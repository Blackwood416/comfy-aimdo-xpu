/* Windows Unified Runtime USM allocation interception for AIMDO.
 *
 * This is the Windows counterpart of src-xpu/ur-usm-hook.cpp. Both arbitrate
 * the same entry points for the same reason: PyTorch keeps its native XPU
 * caching allocator, with all of its block splitting, coalescing and
 * empty_cache() behaviour, and AIMDO intercepts the Unified Runtime call that
 * grows device memory so it can apply the AIMDO budget before the request is
 * placed.
 *
 * Only the injection method differs. Linux exports urUSMDeviceAlloc in the
 * LIBUR_LOADER_0.12 version namespace and is injected with LD_PRELOAD. Windows
 * has neither mechanism, but it does not need them: the SYCL runtime obtains
 * the ur_loader.dll module handle from ur_win_proxy_loader.dll and resolves the
 * flat exports by name with GetProcAddress, so the pointer it calls points
 * directly into ur_loader.dll's code. Patching that function with Detours
 * intercepts every caller no matter when the pointer was captured, which is why
 * this hook can be installed from control.init() instead of before the process
 * starts.
 *
 * Why this supersedes arbitrating at zeMemAllocDevice (see ze-detour.c):
 *
 *   ur_loader.dll!urUSMDeviceAlloc      <- AIMDO decides here
 *     -> ur_adapter_level_zero.dll
 *       -> ze_loader.dll!zeMemAllocDevice   <- the driver allocates here
 *
 * A hook at the driver entry point cannot reclaim before the allocation
 * without re-entering Level Zero's own memory management from inside its
 * allocation call, which was observed to corrupt driver state and surface as
 * UR_RESULT_ERROR_DEVICE_LOST. Arbitrating one layer up enables PyTorch's
 * native-cache refusal/retry protocol, but this function is still inside the
 * allocator/UMF call stack.  On Windows, eviction calls therefore only publish
 * a pressure request; an ensuing VBAR/model-owner boundary performs the
 * actual unmap after checking page-retirement fences. Linux retains immediate
 * allocator-time reclaim.
 */

#include <windows.h>
#include <psapi.h>
#include <detours.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Minimal Unified Runtime declarations. ur_api.h is not required to build
 * AIMDO and is not assumed to be installed; only opaque handles and three
 * result codes are needed, and those are part of the frozen UR ABI. */
typedef int ur_result_t;

#define UR_RESULT_SUCCESS 0
#define UR_RESULT_ERROR_UNINITIALIZED 37
#define UR_RESULT_ERROR_OUT_OF_HOST_MEMORY 38
#define UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY 39
#define UR_RESULT_ERROR_OUT_OF_RESOURCES 40

typedef void *ur_context_handle_t;
typedef void *ur_device_handle_t;
typedef void *ur_usm_pool_handle_t;
typedef void *ur_native_handle_t;
typedef struct ur_usm_desc_t ur_usm_desc_t;

typedef ur_result_t(__cdecl *PFN_urUSMDeviceAlloc)(
    ur_context_handle_t, ur_device_handle_t, const ur_usm_desc_t *,
    ur_usm_pool_handle_t, size_t, void **);
typedef ur_result_t(__cdecl *PFN_urUSMFree)(ur_context_handle_t, void *);
typedef ur_result_t(__cdecl *PFN_urDeviceGetNativeHandle)(ur_device_handle_t,
                                                          ur_native_handle_t *);
typedef ur_result_t(__cdecl *PFN_urPhysicalMemCreate)(ur_context_handle_t,
                                                      ur_device_handle_t, size_t,
                                                      const void *, void **);
typedef ur_result_t(__cdecl *PFN_urGetProcAddrTable)(uint32_t, void *);
typedef void *(__cdecl *PFN_getPreloadedURLib)(void);

/* Minimal DDI table layouts matching unified-runtime/ur_ddi.h. Used when
 * sycl9.dll (oneAPI 2026.2+) statically links the UR loader and dispatches
 * straight into ur_adapter_level_zero[_v2].dll's DDI tables rather than
 * calling ur_loader.dll!urUSMDeviceAlloc. */
struct AimdoUrUsmDdiTable {
    void *pfnHostAlloc;
    PFN_urUSMDeviceAlloc pfnDeviceAlloc;
    void *pfnSharedAlloc;
    PFN_urUSMFree pfnFree;
    void *pfnGetMemAllocInfo;
    void *pfnPoolCreate;
    void *pfnPoolRetain;
    void *pfnPoolRelease;
    void *pfnPoolGetInfo;
};

struct AimdoUrDeviceDdiTable {
    void *pfnGet;
    void *pfnGetInfo;
    void *pfnRetain;
    void *pfnRelease;
    void *pfnPartition;
    void *pfnSelectBinary;
    PFN_urDeviceGetNativeHandle pfnGetNativeHandle;
    void *pfnCreateWithNativeHandle;
    void *pfnGetGlobalTimestamps;
};

struct AimdoUrPhysicalMemDdiTable {
    PFN_urPhysicalMemCreate pfnCreate;
    void *pfnRetain;
    void *pfnRelease;
    void *pfnGetInfo;
};

#define AIMDO_UR_API_VERSION_0_12 0x0000000cu
#define AIMDO_UR_API_VERSION_0_11 0x0000000bu
#define AIMDO_UR_API_VERSION_0_10 0x0000000au
#define AIMDO_UR_API_VERSION_0_9  0x00000009u

#if defined(__cplusplus)
extern "C" {
#endif

enum AimdoUrLogLevel {
    kAimdoUrLogError = 2,
    kAimdoUrLogWarning = 3,
    kAimdoUrLogInfo = 4,
    kAimdoUrLogDebug = 5,
};

extern void aimdo_log(
    int level, const char *file, int line, const char *format, ...);
extern bool aimdo_xpu_allocation_deficit(int device, size_t size,
                                         int64_t *deficit);
extern bool aimdo_xpu_evict_for_allocation(int device, int64_t deficit);
extern bool aimdo_xpu_account_allocation(int device, int64_t delta);
extern int xpu_device_from_native_handle(uintptr_t native_handle);

/* Statistic order must match src-xpu/ur-usm-hook.cpp so control.py can report
 * both platforms with one table. */
enum UrHookStat {
    kAllocCalls,
    kFreeCalls,
    kPassThroughAllocCalls,
    kTrackedAllocCalls,
    kTrackedAllocBytes,
    kTrackedFreeCalls,
    kTrackedFreeBytes,
    kSyntheticOomCalls,
    kRuntimeOomCalls,
    kNativeReclaimFreeCalls,
    kNativeReclaimFreeBytes,
    kRetryEvictionCalls,
    kRetryEvictionBytes,
    kUnknownDeviceCalls,
    kUnknownFreeCalls,
    kDroppedMetadataCalls,
    kDirectPressureCalls,
    kDirectPressureBytes,
    kDuplicatePointerCalls,
    kPhysicalMemCreateCalls,
    kCacheLeverSkippedCalls,
    kHookStatCount
};

enum RetryReason {
    kRetryNone = 0,
    kRetryBudgetDeficit,
    kRetryRuntimeOom
};

#ifdef AIMDO_XPU_TESTING
/* Seams used by tests/ur_usm_detour_unit.c. They mirror the ones
 * src-xpu/ur-usm-hook.cpp exposes for the Linux unit test, so both platforms
 * verify the same state machine without a device or a Unified Runtime loader. */
enum TestRequestKind {
    kTestRequestAutomatic = 0,
    kTestRequestDirect,
    kTestRequestTorchNative
};
static volatile LONG g_test_request_kind;
#endif

static PFN_urUSMDeviceAlloc true_urUSMDeviceAlloc;
static PFN_urUSMFree true_urUSMFree;
static PFN_urPhysicalMemCreate true_urPhysicalMemCreate;
static PFN_urDeviceGetNativeHandle real_urDeviceGetNativeHandle;

/* Adapter-level hooks for oneAPI 2026.2+ (sycl9.dll statically links the UR
 * loader and invokes ur_adapter_level_zero[_v2].dll's DDI table entries
 * directly, bypassing ur_loader.dll). We attach to both the v1 and v2 Level
 * Zero adapters when loaded; a per-thread nesting guard ensures that if an
 * older sycl8.dll calls ur_loader.dll which then forwards into the adapter,
 * arbitration and accounting run exactly once at the outermost entry. */
#define MAX_UR_ADAPTER_SLOTS 2

struct UrAdapterHookSlot {
    bool attached;
    PFN_urUSMDeviceAlloc true_alloc;
    PFN_urUSMFree true_free;
    PFN_urPhysicalMemCreate true_phys_create;
    PFN_urDeviceGetNativeHandle get_native_handle;
};

static struct UrAdapterHookSlot g_adapter_slots[MAX_UR_ADAPTER_SLOTS];
static __declspec(thread) int t_ur_hook_depth;
static __declspec(thread) PFN_urUSMDeviceAlloc t_active_true_alloc;
static __declspec(thread) PFN_urDeviceGetNativeHandle t_active_get_native_handle;

static volatile LONG g_attached;
static volatile LONG g_enabled;
static volatile LONG64 g_generation;
static volatile LONG64 g_stats[kHookStatCount];
static CRITICAL_SECTION g_hook_lock;
static volatile LONG g_lock_ready;

static __declspec(thread) ur_context_handle_t t_retry_context;
static __declspec(thread) ur_device_handle_t t_retry_device;
static __declspec(thread) ur_usm_pool_handle_t t_retry_pool;
static __declspec(thread) size_t t_retry_size;
static __declspec(thread) int t_retry_reason;
static __declspec(thread) LONG64 t_retry_generation;
static __declspec(thread) LONG64 t_retry_returned_bytes;
static __declspec(thread) int t_ur_accounting_depth;

bool aimdo_xpu_ur_accounting_active(void) {
    return t_ur_accounting_depth > 0;
}

/* Bytes PyTorch is holding in its own cache, per device.
 *
 * The synthetic out-of-memory path is the only way AIMDO can reclaim those
 * bytes, and it is expensive: PyTorch answers a refusal by discarding its
 * whole cache, which was measured to double the number of real driver
 * allocations. Firing it when the cache is already empty pays that cost for
 * nothing: 71 of 71 refusals during a Torch ramp returned zero bytes.
 *
 * This value cannot be derived inside the hook. A cached block is one PyTorch
 * freed but did not return to the driver, so it never reaches urUSMFree and
 * the hook is structurally blind to the cache growing. The hook can only learn
 * that the cache is empty, by observing that a refusal returned nothing.
 * Learning only that direction would latch the mechanism off permanently, so
 * comfy_aimdo publishes the live figure from PyTorch's own statistics. */
#define MAX_TRACKED_DEVICES 16
static volatile LONG64 g_torch_cached_bytes[MAX_TRACKED_DEVICES];

static LONG64 torch_cached_bytes(int device) {
    if (device < 0 || device >= MAX_TRACKED_DEVICES) {
        return 0;
    }
    return InterlockedCompareExchange64(&g_torch_cached_bytes[device], 0, 0);
}

static void torch_cached_bytes_consume(int device, LONG64 returned) {
    LONG64 current;

    if (device < 0 || device >= MAX_TRACKED_DEVICES) {
        return;
    }
    if (returned <= 0) {
        /* The refusal produced nothing, so the estimate was stale and the
         * cache is empty. Believe the observation over the estimate. */
        InterlockedExchange64(&g_torch_cached_bytes[device], 0);
        return;
    }
    current = InterlockedCompareExchange64(&g_torch_cached_bytes[device], 0, 0);
    InterlockedExchange64(&g_torch_cached_bytes[device],
                          current > returned ? current - returned : 0);
}

/* Live tracked allocations. Open addressed with tombstones: USM device
 * allocations are highly aligned and collide heavily, and clearing a slot on
 * removal would truncate the probe chain of every pointer behind it. */
#define POINTER_TABLE_SIZE 262144
#define POINTER_TOMBSTONE ((void *)(uintptr_t)1)

struct TrackedAllocation {
    void *pointer;
    size_t size;
    int device;
};

static struct TrackedAllocation g_allocations[POINTER_TABLE_SIZE];
static volatile LONG64 g_allocation_count;

static void stat_add(enum UrHookStat stat, LONG64 delta) {
    InterlockedAdd64(&g_stats[stat], delta);
}

static size_t pointer_slot(void *pointer) {
    uint64_t value = (uint64_t)(uintptr_t)pointer;

    /* Fold the high bits down before mixing: multiplication only propagates
     * bits upwards, so hashing the low bits alone would map every 2 MiB
     * aligned allocation to one slot. */
    value ^= value >> 33;
    value *= 11400714819323198485ull;
    return (size_t)((value >> 44) & (POINTER_TABLE_SIZE - 1));
}

static bool allocation_insert(void *pointer, size_t size, int device) {
    size_t slot = pointer_slot(pointer);
    size_t probe;

    for (probe = 0; probe < POINTER_TABLE_SIZE; ++probe) {
        size_t index = (slot + probe) & (POINTER_TABLE_SIZE - 1);

        if (g_allocations[index].pointer == pointer) {
            return false;
        }
        if (!g_allocations[index].pointer ||
            g_allocations[index].pointer == POINTER_TOMBSTONE) {
            g_allocations[index].pointer = pointer;
            g_allocations[index].size = size;
            g_allocations[index].device = device;
            InterlockedIncrement64(&g_allocation_count);
            return true;
        }
    }
    return false;
}

static bool allocation_remove(void *pointer, size_t *size, int *device) {
    size_t slot = pointer_slot(pointer);
    size_t probe;

    for (probe = 0; probe < POINTER_TABLE_SIZE; ++probe) {
        size_t index = (slot + probe) & (POINTER_TABLE_SIZE - 1);

        if (!g_allocations[index].pointer) {
            return false;
        }
        if (g_allocations[index].pointer == pointer) {
            *size = g_allocations[index].size;
            *device = g_allocations[index].device;
            g_allocations[index].pointer = POINTER_TOMBSTONE;
            g_allocations[index].size = 0;
            InterlockedDecrement64(&g_allocation_count);
            return true;
        }
    }
    return false;
}

static const wchar_t *base_name(const wchar_t *path) {
    const wchar_t *slash = wcsrchr(path, L'\\');

    return slash ? slash + 1 : path;
}

typedef BOOL(WINAPI *PFN_EnumProcessModules)(HANDLE, HMODULE *, DWORD, LPDWORD);
typedef BOOL(WINAPI *PFN_GetModuleInformation)(HANDLE, HMODULE, LPMODULEINFO,
                                               DWORD);

/* Caller classification.
 *
 * ur-usm-hook.cpp walks the stack with backtrace()/dladdr() looking for
 * libc10_xpu.so. The direct Windows transliteration - GetModuleHandleExW plus
 * GetModuleFileNameW on every frame - resolves and formats a module path up to
 * 48 times per allocation, and it runs precisely when the deficit is positive,
 * which is the hot path under pressure. dladdr() is cheap enough for that
 * shape; the Win32 pair is not.
 *
 * c10_xpu.dll occupies one contiguous image range that cannot move while it is
 * loaded, so the range is resolved once and each frame becomes a pair of
 * integer comparisons. See docs/WINDOWS_XPU_UR_ALLOCATOR_HOOK.md for the
 * measured cost of both forms. */
#define CLASSIFY_MAX_FRAMES 32

static uintptr_t g_torch_module_base;
static uintptr_t g_torch_module_end;
static volatile LONG g_torch_module_state; /* 0 unknown, 1 resolved */

static volatile LONG64 g_classify_calls;
static volatile LONG64 g_classify_ticks;
static volatile LONG64 g_classify_torch_hits;
static volatile LONG64 g_hook_calls;
static volatile LONG64 g_hook_ticks;

static HMODULE find_module_by_name(const wchar_t *wanted) {
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    PFN_EnumProcessModules enumerate;
    HMODULE modules[1024];
    DWORD needed = 0;
    DWORD count;
    DWORD index;

    if (!kernel) {
        return NULL;
    }
    enumerate = (PFN_EnumProcessModules)(void *)GetProcAddress(
        kernel, "K32EnumProcessModules");
    if (!enumerate ||
        !enumerate(GetCurrentProcess(), modules, sizeof(modules), &needed)) {
        return NULL;
    }
    count = needed / (DWORD)sizeof(HMODULE);
    if (count > ARRAYSIZE(modules)) {
        count = ARRAYSIZE(modules);
    }
    for (index = 0; index < count; ++index) {
        wchar_t path[MAX_PATH];

        if (!GetModuleFileNameW(modules[index], path, ARRAYSIZE(path))) {
            continue;
        }
        if (_wcsicmp(base_name(path), wanted) == 0) {
            return modules[index];
        }
    }
    return NULL;
}

/* Resolve a loaded module's image range. Returns false while the module is not
 * loaded yet, which is normal before PyTorch initializes XPU, so the result is
 * only cached once it succeeds. */
static bool module_image_range(const wchar_t *name, uintptr_t *base,
                               uintptr_t *end) {
    HMODULE kernel = GetModuleHandleA("kernel32.dll");
    PFN_GetModuleInformation get_information;
    MODULEINFO information;
    HMODULE module = GetModuleHandleW(name);

    if (!module) {
        module = find_module_by_name(name);
    }
    if (!module || !kernel) {
        return false;
    }
    get_information = (PFN_GetModuleInformation)(void *)GetProcAddress(
        kernel, "K32GetModuleInformation");
    if (!get_information ||
        !get_information(GetCurrentProcess(), module, &information,
                         (DWORD)sizeof(information))) {
        return false;
    }
    *base = (uintptr_t)information.lpBaseOfDll;
    *end = *base + information.SizeOfImage;
    return true;
}

static bool ensure_torch_module(void) {
    uintptr_t base;
    uintptr_t end;

    if (InterlockedCompareExchange(&g_torch_module_state, 0, 0)) {
        return true;
    }
    if (!module_image_range(L"c10_xpu.dll", &base, &end)) {
        return false;
    }
    g_torch_module_base = base;
    g_torch_module_end = end;
    InterlockedExchange(&g_torch_module_state, 1);
    return true;
}

static bool is_torch_native_segment_request(void) {
    void *frames[CLASSIFY_MAX_FRAMES];
    LARGE_INTEGER started;
    LARGE_INTEGER finished;
    USHORT count;
    USHORT index;
    bool found = false;

#ifdef AIMDO_XPU_TESTING
    if (g_test_request_kind != kTestRequestAutomatic) {
        return g_test_request_kind == kTestRequestTorchNative;
    }
#endif
    if (!ensure_torch_module()) {
        return false;
    }

    QueryPerformanceCounter(&started);
    count = RtlCaptureStackBackTrace(1, CLASSIFY_MAX_FRAMES, frames, NULL);
    for (index = 0; index < count; ++index) {
        const uintptr_t address = (uintptr_t)frames[index];

        if (address >= g_torch_module_base && address < g_torch_module_end) {
            found = true;
            break;
        }
    }
    QueryPerformanceCounter(&finished);

    InterlockedAdd64(&g_classify_ticks, finished.QuadPart - started.QuadPart);
    InterlockedIncrement64(&g_classify_calls);
    if (found) {
        InterlockedIncrement64(&g_classify_torch_hits);
    }
    return found;
}

static int resolve_device(ur_device_handle_t device) {
    ur_native_handle_t native = NULL;
    PFN_urDeviceGetNativeHandle get_native =
        t_active_get_native_handle ? t_active_get_native_handle
                                   : real_urDeviceGetNativeHandle;

    if (!get_native) {
        return -1;
    }
    if (get_native(device, &native) != UR_RESULT_SUCCESS) {
        return -1;
    }
    return xpu_device_from_native_handle((uintptr_t)native);
}

static void clear_retry(void) {
    t_retry_context = NULL;
    t_retry_device = NULL;
    t_retry_pool = NULL;
    t_retry_size = 0;
    t_retry_reason = kRetryNone;
    t_retry_generation = 0;
    t_retry_returned_bytes = 0;
}

static bool retry_matches(ur_context_handle_t context, ur_device_handle_t device,
                          ur_usm_pool_handle_t pool, size_t size,
                          LONG64 generation) {
    return t_retry_reason != kRetryNone && t_retry_context == context &&
           t_retry_device == device && t_retry_pool == pool &&
           t_retry_size == size && t_retry_generation == generation;
}

static void arm_retry(ur_context_handle_t context, ur_device_handle_t device,
                      ur_usm_pool_handle_t pool, size_t size, int reason,
                      LONG64 generation) {
    t_retry_context = context;
    t_retry_device = device;
    t_retry_pool = pool;
    t_retry_size = size;
    t_retry_reason = reason;
    t_retry_generation = generation;
    t_retry_returned_bytes = 0;
}

static bool is_oom(ur_result_t result) {
    return result == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
           result == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
           result == UR_RESULT_ERROR_OUT_OF_RESOURCES;
}

static ur_result_t allocate_and_account(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer, int aimdo_device) {
    ur_result_t result;
    PFN_urUSMDeviceAlloc real_alloc =
        t_active_true_alloc ? t_active_true_alloc : true_urUSMDeviceAlloc;

    /* Nested Level Zero hooks observe the underlying zeMemAllocDevice on this
     * thread; mark UR ownership so total_vram_usage is credited once here. */
    ++t_ur_accounting_depth;
    result = real_alloc(context, device, description, pool, size, pointer);
    --t_ur_accounting_depth;

    if (is_oom(result)) {
        stat_add(kRuntimeOomCalls, 1);
        if (result == UR_RESULT_ERROR_OUT_OF_RESOURCES) {
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return result;
    }
    if (result != UR_RESULT_SUCCESS) {
        return result;
    }
    if (!pointer || !*pointer) {
        return result;
    }
    if (!allocation_insert(*pointer, size, aimdo_device)) {
        /* A duplicate pointer means the layer below broke its contract. Do not
         * free it: it may still belong to the first tracked allocation. */
        stat_add(kDuplicatePointerCalls, 1);
        stat_add(kDroppedMetadataCalls, 1);
        return result;
    }
    aimdo_xpu_account_allocation(aimdo_device, (int64_t)size);
    stat_add(kTrackedAllocCalls, 1);
    stat_add(kTrackedAllocBytes, (LONG64)size);
    return result;
}

static ur_result_t aimdo_urUSMDeviceAlloc_body(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer) {
    LONG64 generation;
    int aimdo_device;
    int64_t deficit = 0;
    bool caller_classified;
    bool torch_native_request;
    ur_result_t result;
    PFN_urUSMDeviceAlloc real_alloc =
        t_active_true_alloc ? t_active_true_alloc : true_urUSMDeviceAlloc;

    if (!real_alloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        stat_add(kPassThroughAllocCalls, 1);
        return real_alloc(context, device, description, pool, size, pointer);
    }

    EnterCriticalSection(&g_hook_lock);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        clear_retry();
        LeaveCriticalSection(&g_hook_lock);
        stat_add(kPassThroughAllocCalls, 1);
        return real_alloc(context, device, description, pool, size, pointer);
    }

    generation = InterlockedCompareExchange64(&g_generation, 0, 0);
    aimdo_device = resolve_device(device);
    if (aimdo_device < 0) {
        clear_retry();
        LeaveCriticalSection(&g_hook_lock);
        stat_add(kUnknownDeviceCalls, 1);
        return real_alloc(context, device, description, pool, size, pointer);
    }

    if (!aimdo_xpu_allocation_deficit(aimdo_device, size, &deficit)) {
        clear_retry();
        LeaveCriticalSection(&g_hook_lock);
        return true_urUSMDeviceAlloc(context, device, description, pool, size,
                                     pointer);
    }

    caller_classified = deficit > 0 || t_retry_reason != kRetryNone;
    torch_native_request =
        caller_classified && is_torch_native_segment_request();

    if (torch_native_request &&
        retry_matches(context, device, pool, size, generation)) {
        /* PyTorch is retrying the request this hook failed. It has just called
         * release_cached_blocks(), so its cache has already come back through
         * urUSMFree and been credited. Publish what is still missing and let
         * the allocation through; Windows drains it at the next owner-side
         * VBAR boundary, while Linux still evicts immediately. */
        int reason = t_retry_reason;
        int64_t eviction = deficit > 0 ? deficit : 0;
        LONG64 returned = t_retry_returned_bytes;

        clear_retry();
        /* Believe what the refusal actually produced, not what was estimated
         * before it. This is the half of the signal the hook can observe. */
        torch_cached_bytes_consume(aimdo_device, returned);
        if (reason == kRetryRuntimeOom && eviction < (int64_t)size) {
            eviction = (int64_t)size;
        }
        if (eviction > 0) {
            aimdo_xpu_evict_for_allocation(aimdo_device, eviction);
            stat_add(kRetryEvictionCalls, 1);
            stat_add(kRetryEvictionBytes, eviction);
        }
        result = allocate_and_account(context, device, description, pool, size,
                                      pointer, aimdo_device);
        LeaveCriticalSection(&g_hook_lock);
        return result;
    }

    clear_retry();
    if (deficit > 0) {
        if (torch_native_request && torch_cached_bytes(aimdo_device) > 0) {
            /* Fail the request instead of evicting immediately. PyTorch
             * responds by releasing its cached blocks, which no hook below
             * this one can see because they never reach the driver, and then
             * retries the identical request. That converts Torch's own cache
             * into the first source of reclaim and keeps AIMDO's VBAR pages
             * as the second, which is the right order: rebuilding the cache
             * costs a driver allocation, while refaulting a VBAR page costs a
             * host-to-device stream of the weight.
             *
             * Only worth doing when PyTorch actually has a cache to give up.
             * Refusing a request it cannot answer costs a full cache flush and
             * a retry for nothing. */
            arm_retry(context, device, pool, size, kRetryBudgetDeficit,
                      generation);
            if (pointer) {
                *pointer = NULL;
            }
            stat_add(kSyntheticOomCalls, 1);
            LeaveCriticalSection(&g_hook_lock);
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        if (torch_native_request) {
            stat_add(kCacheLeverSkippedCalls, 1);
        }
        /* Either not a PyTorch segment request, or PyTorch has no cache to
         * surrender. Windows records this for its next owner-side VBAR
         * boundary; Linux evicts directly. */
        if (!aimdo_xpu_evict_for_allocation(aimdo_device, deficit)) {
            if (pointer) {
                *pointer = NULL;
            }
            LeaveCriticalSection(&g_hook_lock);
            return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        stat_add(kDirectPressureCalls, 1);
        stat_add(kDirectPressureBytes, deficit);
    }

    result = allocate_and_account(context, device, description, pool, size,
                                  pointer, aimdo_device);
    if (is_oom(result)) {
        if (!caller_classified) {
            torch_native_request = is_torch_native_segment_request();
        }
        if (torch_native_request) {
            arm_retry(context, device, pool, size, kRetryRuntimeOom, generation);
        }
    }
    LeaveCriticalSection(&g_hook_lock);
    return result;
}

/* Wrapper that times the whole hook, including the driver call it wraps. */
static ur_result_t aimdo_urUSMDeviceAlloc_dispatch(
    PFN_urUSMDeviceAlloc real_alloc, PFN_urDeviceGetNativeHandle get_native,
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer) {
    LARGE_INTEGER started;
    LARGE_INTEGER finished;
    ur_result_t result;
    PFN_urUSMDeviceAlloc prev_alloc;
    PFN_urDeviceGetNativeHandle prev_native;

    if (!real_alloc) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (t_ur_hook_depth > 0) {
        return real_alloc(context, device, description, pool, size, pointer);
    }

    prev_alloc = t_active_true_alloc;
    prev_native = t_active_get_native_handle;
    t_active_true_alloc = real_alloc;
    t_active_get_native_handle = get_native;
    ++t_ur_hook_depth;
    QueryPerformanceCounter(&started);
    result = aimdo_urUSMDeviceAlloc_body(context, device, description, pool,
                                         size, pointer);
    QueryPerformanceCounter(&finished);
    --t_ur_hook_depth;
    t_active_true_alloc = prev_alloc;
    t_active_get_native_handle = prev_native;
    stat_add(kAllocCalls, 1);
    InterlockedAdd64(&g_hook_ticks, finished.QuadPart - started.QuadPart);
    InterlockedIncrement64(&g_hook_calls);
    return result;
}

static ur_result_t __cdecl aimdo_urUSMDeviceAlloc(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_usm_desc_t *description, ur_usm_pool_handle_t pool, size_t size,
    void **pointer) {
    return aimdo_urUSMDeviceAlloc_dispatch(
        true_urUSMDeviceAlloc, real_urDeviceGetNativeHandle, context, device,
        description, pool, size, pointer);
}

static ur_result_t aimdo_urUSMFree_dispatch(PFN_urUSMFree real_free,
                                            ur_context_handle_t context,
                                            void *pointer) {
    ur_result_t result;
    size_t size = 0;
    int device = -1;

    if (!real_free) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (t_ur_hook_depth > 0) {
        return real_free(context, pointer);
    }
    stat_add(kFreeCalls, 1);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        return real_free(context, pointer);
    }

    EnterCriticalSection(&g_hook_lock);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        LeaveCriticalSection(&g_hook_lock);
        return real_free(context, pointer);
    }
    ++t_ur_hook_depth;
    result = real_free(context, pointer);
    --t_ur_hook_depth;
    if (result != UR_RESULT_SUCCESS) {
        LeaveCriticalSection(&g_hook_lock);
        return result;
    }
    if (!allocation_remove(pointer, &size, &device)) {
        stat_add(kUnknownFreeCalls, 1);
        LeaveCriticalSection(&g_hook_lock);
        return result;
    }
    aimdo_xpu_account_allocation(device, -(int64_t)size);
    if (t_retry_reason != kRetryNone &&
        t_retry_generation == InterlockedCompareExchange64(&g_generation, 0, 0)) {
        /* Released while a synthetic failure is outstanding, so this is
         * PyTorch returning its cache in response to it. */
        t_retry_returned_bytes += (LONG64)size;
        stat_add(kNativeReclaimFreeCalls, 1);
        stat_add(kNativeReclaimFreeBytes, (LONG64)size);
    }
    stat_add(kTrackedFreeCalls, 1);
    stat_add(kTrackedFreeBytes, (LONG64)size);
    LeaveCriticalSection(&g_hook_lock);
    return result;
}

static ur_result_t __cdecl aimdo_urUSMFree(ur_context_handle_t context,
                                           void *pointer) {
    return aimdo_urUSMFree_dispatch(true_urUSMFree, context, pointer);
}

/* Counted, never arbitrated. PyTorch allocates through this entry point
 * instead of USM when expandable segments are enabled, in which case
 * urUSMDeviceAlloc is never called and AIMDO would otherwise be silently
 * blind. control.py refuses that configuration; this counter is what proves
 * the refusal is warranted. */
static ur_result_t aimdo_urPhysicalMemCreate_dispatch(
    PFN_urPhysicalMemCreate real_create, ur_context_handle_t context,
    ur_device_handle_t device, size_t size, const void *properties,
    void **handle) {
    ur_result_t result;

    if (!real_create) {
        return UR_RESULT_ERROR_UNINITIALIZED;
    }
    if (t_ur_hook_depth > 0) {
        return real_create(context, device, size, properties, handle);
    }
    stat_add(kPhysicalMemCreateCalls, 1);
    ++t_ur_hook_depth;
    result = real_create(context, device, size, properties, handle);
    --t_ur_hook_depth;
    return result;
}

static ur_result_t __cdecl aimdo_urPhysicalMemCreate(
    ur_context_handle_t context, ur_device_handle_t device, size_t size,
    const void *properties, void **handle) {
    return aimdo_urPhysicalMemCreate_dispatch(
        true_urPhysicalMemCreate, context, device, size, properties, handle);
}

#define DEFINE_UR_ADAPTER_THUNKS(SLOT)                                          \
    static ur_result_t __cdecl aimdo_adapter_alloc_##SLOT(                      \
        ur_context_handle_t context, ur_device_handle_t device,                 \
        const ur_usm_desc_t *description, ur_usm_pool_handle_t pool,            \
        size_t size, void **pointer) {                                          \
        return aimdo_urUSMDeviceAlloc_dispatch(                                 \
            g_adapter_slots[SLOT].true_alloc,                                   \
            g_adapter_slots[SLOT].get_native_handle, context, device,           \
            description, pool, size, pointer);                                  \
    }                                                                           \
    static ur_result_t __cdecl aimdo_adapter_free_##SLOT(                       \
        ur_context_handle_t context, void *pointer) {                           \
        return aimdo_urUSMFree_dispatch(g_adapter_slots[SLOT].true_free,        \
                                        context, pointer);                      \
    }                                                                           \
    static ur_result_t __cdecl aimdo_adapter_phys_create_##SLOT(                \
        ur_context_handle_t context, ur_device_handle_t device, size_t size,    \
        const void *properties, void **handle) {                                \
        return aimdo_urPhysicalMemCreate_dispatch(                              \
            g_adapter_slots[SLOT].true_phys_create, context, device, size,      \
            properties, handle);                                                \
    }

DEFINE_UR_ADAPTER_THUNKS(0)
DEFINE_UR_ADAPTER_THUNKS(1)

static const PFN_urUSMDeviceAlloc kAdapterAllocThunks[MAX_UR_ADAPTER_SLOTS] = {
    aimdo_adapter_alloc_0,
    aimdo_adapter_alloc_1,
};
static const PFN_urUSMFree kAdapterFreeThunks[MAX_UR_ADAPTER_SLOTS] = {
    aimdo_adapter_free_0,
    aimdo_adapter_free_1,
};
static const PFN_urPhysicalMemCreate
    kAdapterPhysCreateThunks[MAX_UR_ADAPTER_SLOTS] = {
        aimdo_adapter_phys_create_0,
        aimdo_adapter_phys_create_1,
};

static bool query_adapter_ddi_table(HMODULE adapter, const char *export_name,
                                    void *table_out, size_t table_size) {
    static const uint32_t kVersions[] = {
        AIMDO_UR_API_VERSION_0_12,
        AIMDO_UR_API_VERSION_0_11,
        AIMDO_UR_API_VERSION_0_10,
        AIMDO_UR_API_VERSION_0_9,
    };
    PFN_urGetProcAddrTable getter =
        (PFN_urGetProcAddrTable)(void *)GetProcAddress(adapter, export_name);
    size_t i;

    if (!getter) {
        return false;
    }
    for (i = 0; i < ARRAYSIZE(kVersions); ++i) {
        memset(table_out, 0, table_size);
        if (getter(kVersions[i], table_out) == UR_RESULT_SUCCESS) {
            return true;
        }
    }
    return false;
}

static bool attach_adapter_slot(int slot_index, const wchar_t *dll_name) {
    HMODULE adapter = GetModuleHandleW(dll_name);
    struct AimdoUrUsmDdiTable usm_tbl;
    struct AimdoUrDeviceDdiTable dev_tbl;
    struct AimdoUrPhysicalMemDdiTable phys_tbl;
    struct UrAdapterHookSlot *slot = &g_adapter_slots[slot_index];
    LONG status;

    if (!adapter) {
        adapter = find_module_by_name(dll_name);
    }
    if (!adapter) {
        return false;
    }
    if (!query_adapter_ddi_table(adapter, "urGetUSMProcAddrTable", &usm_tbl,
                                 sizeof(usm_tbl)) ||
        !query_adapter_ddi_table(adapter, "urGetDeviceProcAddrTable", &dev_tbl,
                                 sizeof(dev_tbl)) ||
        !usm_tbl.pfnDeviceAlloc || !usm_tbl.pfnFree ||
        !dev_tbl.pfnGetNativeHandle) {
        return false;
    }

    slot->true_alloc = usm_tbl.pfnDeviceAlloc;
    slot->true_free = usm_tbl.pfnFree;
    slot->get_native_handle = dev_tbl.pfnGetNativeHandle;
    slot->true_phys_create = NULL;
    if (query_adapter_ddi_table(adapter, "urGetPhysicalMemProcAddrTable",
                                &phys_tbl, sizeof(phys_tbl))) {
        slot->true_phys_create = phys_tbl.pfnCreate;
    }

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    status = DetourAttach((void **)&slot->true_alloc,
                          kAdapterAllocThunks[slot_index]);
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&slot->true_free,
                              kAdapterFreeThunks[slot_index]);
    }
    if (status == NO_ERROR && slot->true_phys_create) {
        status = DetourAttach((void **)&slot->true_phys_create,
                              kAdapterPhysCreateThunks[slot_index]);
    }
    if (status != NO_ERROR) {
        DetourTransactionAbort();
        memset(slot, 0, sizeof(*slot));
        aimdo_log(kAimdoUrLogWarning, __FILE__, __LINE__,
                  "%s: DetourAttach failed for %ls: %ld\n", __func__, dll_name,
                  (long)status);
        return false;
    }
    status = DetourTransactionCommit();
    if (status != NO_ERROR) {
        memset(slot, 0, sizeof(*slot));
        aimdo_log(kAimdoUrLogWarning, __FILE__, __LINE__,
                  "%s: DetourTransactionCommit failed for %ls: %ld\n", __func__,
                  dll_name, (long)status);
        return false;
    }
    slot->attached = true;
    return true;
}

static void detach_adapter_slots(void) {
    int i;

    for (i = 0; i < MAX_UR_ADAPTER_SLOTS; ++i) {
        struct UrAdapterHookSlot *slot = &g_adapter_slots[i];
        LONG status;

        if (!slot->attached) {
            continue;
        }
        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourDetach((void **)&slot->true_alloc, kAdapterAllocThunks[i]);
        DetourDetach((void **)&slot->true_free, kAdapterFreeThunks[i]);
        if (slot->true_phys_create) {
            DetourDetach((void **)&slot->true_phys_create,
                         kAdapterPhysCreateThunks[i]);
        }
        status = DetourTransactionCommit();
        if (status == NO_ERROR) {
            memset(slot, 0, sizeof(*slot));
        } else {
            aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                      "%s: DetourDetach failed for adapter slot %d: %ld\n",
                      __func__, i, (long)status);
        }
    }
}



/* Resolve the loader SYCL itself uses.
 *
 * GetModuleHandleA("ur_loader.dll") is correct in every environment measured
 * so far, but a process may hold more than one copy, so the authoritative
 * handle is the one ur_win_proxy_loader.dll hands to the SYCL runtime. When
 * both are available and disagree, the proxy wins and the mismatch is logged,
 * because hooking the wrong copy would silently arbitrate nothing.
 *
 * Note that GetModuleHandleA cannot find the proxy itself in a live PyTorch
 * process even though it is loaded, hence the module walk. */
static bool ur_hook_disabled_by_request(void) {
    char value[8];
    DWORD length =
        GetEnvironmentVariableA("AIMDO_XPU_DISABLE_UR_HOOK", value,
                                (DWORD)sizeof(value));

    return length > 0 && length < sizeof(value) && value[0] == '1';
}

static HMODULE resolve_ur_loader(void) {
    HMODULE proxy = find_module_by_name(L"ur_win_proxy_loader.dll");
    HMODULE by_name = GetModuleHandleA("ur_loader.dll");
    if (!by_name) {
        by_name = GetModuleHandleA("ur_loader_shared.dll");
    }
    HMODULE from_proxy = NULL;

    if (proxy) {
        PFN_getPreloadedURLib get_lib = (PFN_getPreloadedURLib)(void *)
            GetProcAddress(proxy, "?getPreloadedURLib@@YAPEAXXZ");

        if (get_lib) {
            from_proxy = (HMODULE)get_lib();
        }
    }
    if (from_proxy && by_name && from_proxy != by_name) {
        aimdo_log(kAimdoUrLogWarning, __FILE__, __LINE__,
                  "%s: more than one ur_loader.dll is loaded; hooking the copy "
                  "used by the SYCL runtime\n", __func__);
    }
    return from_proxy ? from_proxy : by_name;
}

#ifdef AIMDO_XPU_TESTING
/* Bring the hook up without Detours or a Unified Runtime loader. The unit test
 * installs its own entry points into the true_* pointers first. */
static void test_force_enable(void) {
    if (!InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
        InitializeCriticalSection(&g_hook_lock);
        InterlockedExchange(&g_lock_ready, 1);
    }
    InterlockedExchange(&g_attached, 1);
    InterlockedIncrement64(&g_generation);
    clear_retry();
    InterlockedExchange(&g_enabled, 1);
}

static void test_reset_state(void) {
    size_t index;

    for (index = 0; index < (size_t)kHookStatCount; ++index) {
        InterlockedExchange64(&g_stats[index], 0);
    }
    InterlockedExchange64(&g_allocation_count, 0);
    memset(g_allocations, 0, sizeof(g_allocations));
    clear_retry();
}
#endif

void *aimdo_xpu_ur_loader(void) {
    return (void *)resolve_ur_loader();
}

static int attach_missing_adapter_slots(void) {
    int attached = 0;

    if (!g_adapter_slots[0].attached) {
        attach_adapter_slot(0, L"ur_adapter_level_zero.dll");
    }
    if (!g_adapter_slots[1].attached) {
        attach_adapter_slot(1, L"ur_adapter_level_zero_v2.dll");
    }
    if (g_adapter_slots[0].attached) {
        ++attached;
    }
    if (g_adapter_slots[1].attached) {
        ++attached;
    }
    return attached;
}

bool aimdo_xpu_ur_hook_install(void) {
    HMODULE loader;
    LONG status;
    bool loader_attached = false;
    int adapter_attached_count = 0;

    if (InterlockedCompareExchange(&g_attached, 0, 0)) {
        /* If control.init() ran before torch/sycl9 loaded the Level Zero
         * adapters, pick up any newly loaded adapter DLLs on subsequent
         * calls (such as xpu_ur_hook_enable() inside control.init_devices()). */
        if (InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
            EnterCriticalSection(&g_hook_lock);
            attach_missing_adapter_slots();
            LeaveCriticalSection(&g_hook_lock);
        }
        return true;
    }
    /* Checked here rather than only at the ze-detour call site: control.py
     * also reaches this through xpu_ur_hook_is_interposed(), so testing the
     * flag in one place only left the detours attached and both control loops
     * arbitrating the same pressure. */
    if (ur_hook_disabled_by_request()) {
        return false;
    }
    if (!InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
        InitializeCriticalSection(&g_hook_lock);
        InterlockedExchange(&g_lock_ready, 1);
    }

    loader = resolve_ur_loader();
    if (loader) {
        true_urUSMDeviceAlloc =
            (PFN_urUSMDeviceAlloc)(void *)GetProcAddress(loader, "urUSMDeviceAlloc");
        true_urUSMFree = (PFN_urUSMFree)(void *)GetProcAddress(loader, "urUSMFree");
        true_urPhysicalMemCreate = (PFN_urPhysicalMemCreate)(void *)GetProcAddress(
            loader, "urPhysicalMemCreate");
        real_urDeviceGetNativeHandle =
            (PFN_urDeviceGetNativeHandle)(void *)GetProcAddress(
                loader, "urDeviceGetNativeHandle");
        if (true_urUSMDeviceAlloc && true_urUSMFree &&
            real_urDeviceGetNativeHandle) {
            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            status = DetourAttach((void **)&true_urUSMDeviceAlloc,
                                  aimdo_urUSMDeviceAlloc);
            if (status == NO_ERROR) {
                status = DetourAttach((void **)&true_urUSMFree, aimdo_urUSMFree);
            }
            if (status == NO_ERROR && true_urPhysicalMemCreate) {
                status = DetourAttach((void **)&true_urPhysicalMemCreate,
                                      aimdo_urPhysicalMemCreate);
            }
            if (status != NO_ERROR) {
                DetourTransactionAbort();
                aimdo_log(kAimdoUrLogWarning, __FILE__, __LINE__,
                          "%s: DetourAttach on ur_loader failed: %ld\n",
                          __func__, (long)status);
                true_urUSMDeviceAlloc = NULL;
                true_urUSMFree = NULL;
                true_urPhysicalMemCreate = NULL;
            } else {
                status = DetourTransactionCommit();
                if (status != NO_ERROR) {
                    aimdo_log(kAimdoUrLogWarning, __FILE__, __LINE__,
                              "%s: DetourTransactionCommit on ur_loader failed: %ld\n",
                              __func__, (long)status);
                    true_urUSMDeviceAlloc = NULL;
                    true_urUSMFree = NULL;
                    true_urPhysicalMemCreate = NULL;
                } else {
                    loader_attached = true;
                }
            }
        }
    }

    /* Intel oneAPI 2026.2+ (sycl9.dll) statically links the UR loader and
     * dispatches straight into ur_adapter_level_zero[_v2].dll's DDI tables,
     * bypassing ur_loader.dll entirely. Attach to the loaded Level Zero
     * adapters as well; t_ur_hook_depth ensures older sycl8.dll stacks that
     * enter via ur_loader.dll do not double-arbitrate when forwarding down. */
    adapter_attached_count = attach_missing_adapter_slots();

    if (!loader_attached && adapter_attached_count == 0) {
        aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                  "%s: neither ur_loader.dll nor ur_adapter_level_zero.dll "
                  "could be hooked; import torch before initializing AIMDO\n",
                  __func__);
        return false;
    }

    InterlockedExchange(&g_attached, 1);
    aimdo_log(kAimdoUrLogInfo, __FILE__, __LINE__,
              "%s: arbitrating Unified Runtime USM allocations "
              "(ur_loader=%s, l0_adapters=%d); PyTorch caching allocator retained\n",
              __func__, loader_attached ? "yes" : "no", adapter_attached_count);
    return true;
}

void aimdo_xpu_ur_hook_remove(void) {
    LONG status;

    if (!InterlockedCompareExchange(&g_attached, 0, 0)) {
        return;
    }
    InterlockedExchange(&g_enabled, 0);
    detach_adapter_slots();
    if (true_urUSMDeviceAlloc && true_urUSMFree) {
        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourDetach((void **)&true_urUSMDeviceAlloc, aimdo_urUSMDeviceAlloc);
        DetourDetach((void **)&true_urUSMFree, aimdo_urUSMFree);
        if (true_urPhysicalMemCreate) {
            DetourDetach((void **)&true_urPhysicalMemCreate,
                         aimdo_urPhysicalMemCreate);
        }
        status = DetourTransactionCommit();
        if (status != NO_ERROR) {
            /* The hooks are still live; keeping the trampolines is the only way
             * they can still reach the real entry points. */
            aimdo_log(kAimdoUrLogError, __FILE__, __LINE__,
                      "%s: DetourDetach failed: %ld; hooks remain installed\n",
                      __func__, (long)status);
            return;
        }
        true_urUSMDeviceAlloc = NULL;
        true_urUSMFree = NULL;
        true_urPhysicalMemCreate = NULL;
    }
    InterlockedExchange(&g_attached, 0);
}

/* The four entry points below carry the same names and meanings as the Linux
 * hook in src-xpu/ur-usm-hook.cpp, so comfy_aimdo/control.py drives both
 * platforms through one interface. On Linux "interposed" means the dynamic
 * linker bound the symbol to AIMDO; here it means Detours patched it. */
__declspec(dllexport) bool xpu_ur_hook_is_interposed(void) {
    return aimdo_xpu_ur_hook_install();
}

__declspec(dllexport) bool xpu_ur_hook_enable(void) {
    if (!xpu_ur_hook_is_interposed()) {
        return false;
    }
    EnterCriticalSection(&g_hook_lock);
    if (!InterlockedCompareExchange(&g_enabled, 0, 0)) {
        InterlockedIncrement64(&g_generation);
        clear_retry();
        InterlockedExchange(&g_enabled, 1);
    }
    LeaveCriticalSection(&g_hook_lock);
    return true;
}

/* Stop arbitrating.
 *
 * The Linux hook refuses while tracked segments are live, because there AIMDO
 * *is* Torch's allocator and a live segment means its own accounting would be
 * dropped. Windows is not symmetric: the hook only observes allocations that
 * PyTorch owns, and PyTorch legitimately retains some of them across
 * empty_cache(). Refusing here made control.deinit() raise in any process that
 * had ever allocated, so the table is dropped instead and the count reported.
 * The memory itself is unaffected: PyTorch frees it through the unhooked entry
 * point, and AIMDO's accounting is torn down immediately afterwards. */
__declspec(dllexport) bool xpu_ur_hook_disable(void) {
    LONG64 live;

    if (!InterlockedCompareExchange(&g_lock_ready, 0, 0)) {
        return true;
    }
    EnterCriticalSection(&g_hook_lock);
    InterlockedExchange(&g_enabled, 0);
    InterlockedIncrement64(&g_generation);
    clear_retry();
    live = InterlockedExchange64(&g_allocation_count, 0);
    memset(g_allocations, 0, sizeof(g_allocations));
    LeaveCriticalSection(&g_hook_lock);

    if (live > 0) {
        aimdo_log(kAimdoUrLogInfo, __FILE__, __LINE__,
                  "%s: stopped arbitrating with %lld PyTorch segments still "
                  "live; they remain owned by PyTorch\n", __func__,
                  (long long)live);
    }
    return true;
}

/* Publish PyTorch's freed-but-cached bytes for a device.
 *
 * comfy_aimdo calls this from Python, where torch.xpu.memory_stats() is
 * available. The hook cannot derive it: a cached block never reaches the
 * driver, so the growth of the cache is invisible from here. Without this the
 * hook could only ever learn that the cache is empty, and would latch its one
 * reclaim mechanism off for the rest of the process. */
__declspec(dllexport) void xpu_ur_hook_set_torch_cached_bytes(
    int device, uint64_t bytes) {
    if (device < 0 || device >= MAX_TRACKED_DEVICES) {
        return;
    }
    InterlockedExchange64(&g_torch_cached_bytes[device], (LONG64)bytes);
}

__declspec(dllexport) bool xpu_ur_hook_get_stats(uint64_t *values,
                                                 size_t count) {
    size_t index;

    if (!values || count < (size_t)kHookStatCount) {
        return false;
    }
    for (index = 0; index < (size_t)kHookStatCount; ++index) {
        values[index] =
            (uint64_t)InterlockedCompareExchange64(&g_stats[index], 0, 0);
    }
    return true;
}

/* Caller classification cost, reported separately from the shared statistics
 * table so the Linux and Windows tables stay identical. Nanoseconds are
 * derived here because QueryPerformanceFrequency is fixed for the process. */
__declspec(dllexport) bool xpu_ur_hook_get_classify_timing(
    uint64_t *calls, uint64_t *nanoseconds, uint64_t *torch_hits) {
    LARGE_INTEGER frequency;
    LONG64 ticks;

    if (!calls || !nanoseconds || !torch_hits) {
        return false;
    }
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart == 0) {
        return false;
    }
    ticks = InterlockedCompareExchange64(&g_classify_ticks, 0, 0);
    *calls = (uint64_t)InterlockedCompareExchange64(&g_classify_calls, 0, 0);
    *torch_hits =
        (uint64_t)InterlockedCompareExchange64(&g_classify_torch_hits, 0, 0);
    *nanoseconds =
        (uint64_t)((double)ticks * 1e9 / (double)frequency.QuadPart);
    return true;
}

/* Total wall time spent inside the allocation hook, including the real driver
 * call it wraps. This is what decides whether a workload regression belongs to
 * the hook itself or to the consequences of its decisions, such as PyTorch
 * discarding its entire cache after a synthetic failure. */
__declspec(dllexport) bool xpu_ur_hook_get_hook_timing(uint64_t *calls,
                                                       uint64_t *nanoseconds) {
    LARGE_INTEGER frequency;
    LONG64 ticks;

    if (!calls || !nanoseconds) {
        return false;
    }
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart == 0) {
        return false;
    }
    ticks = InterlockedCompareExchange64(&g_hook_ticks, 0, 0);
    *calls = (uint64_t)InterlockedCompareExchange64(&g_hook_calls, 0, 0);
    *nanoseconds =
        (uint64_t)((double)ticks * 1e9 / (double)frequency.QuadPart);
    return true;
}

#if defined(__cplusplus)
}
#endif
