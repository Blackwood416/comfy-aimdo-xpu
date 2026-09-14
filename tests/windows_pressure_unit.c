/* Execute the production Windows pressure calculation with scripted DXGI and
 * device-memory samples. No GPU, driver hook, or SYCL runtime is required. */
#include "../src-win/compiler.h"
#include "../src/plat.h"
#include "../src/aimdo-time.h"
#include <dxgi1_4.h>

static uint64_t test_tick = 10000;
#undef GET_TICK
#define GET_TICK() test_tick

#ifndef AIMDO_PRESSURE_SOURCE
#define AIMDO_PRESSURE_SOURCE "../src-win/shmem-detect.c"
#endif
#include AIMDO_PRESSURE_SOURCE
#include "../src/xpu-copy-pressure.h"

AimdoCudaDispatch g_cuda;
_Thread_local AimdoContext *g_devctx;
int64_t simple_vram_headroom;
int log_level = DEBUG;
uint64_t log_shot_counter;

static AimdoContext context;
static DXGI_QUERY_VIDEO_MEMORY_INFO local_sample, nonlocal_sample;
static size_t device_free;
static unsigned sample_calls, memory_calls, log_calls, failures;
static HRESULT local_result = S_OK;
static size_t reclaimable_bytes, reclaim_calls, copy_polls;
static ssize_t reclaim_target;
static ssize_t requested_reclaim;

#define MIB(value) ((uint64_t)(value) * 1024 * 1024)
#define CHECK(condition) do {                                                 \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __func__, __LINE__, #condition); \
        ++failures;                                                          \
    }                                                                        \
} while (0)

void aimdo_log(int level, const char *file, int line, const char *format, ...) {
    (void)level; (void)file; (void)line; (void)format;
    ++log_calls;
}

bool aimdo_xpu_copy_residency_poll(bool wait) {
    CHECK(!wait);
    ++copy_polls;
    return true;
}

size_t vbars_free_retired(ssize_t size) {
    size_t pages = ((size_t)size + MIB(32) - 1) / MIB(32);
    size_t freed = MIN(reclaimable_bytes, pages * MIB(32));
    CHECK(copy_polls == reclaim_calls + 1);
    ++reclaim_calls;
    reclaim_target = size;
    total_vram_usage -= freed;
    reclaimable_bytes -= freed;
    return pages - freed / MIB(32);
}

void vbars_request_reclaim(ssize_t size) { requested_reclaim = size; }

static HRESULT STDMETHODCALLTYPE sample_memory(
    IDXGIAdapter3 *adapter, UINT node, DXGI_MEMORY_SEGMENT_GROUP segment,
    DXGI_QUERY_VIDEO_MEMORY_INFO *sample) {
    (void)adapter; (void)node;
    ++sample_calls;
    if (segment == DXGI_MEMORY_SEGMENT_GROUP_LOCAL) {
        *sample = local_sample;
        return local_result;
    }
    *sample = nonlocal_sample;
    return S_OK;
}

static CUresult fake_memory_info(size_t *free_bytes, size_t *total_bytes) {
    ++memory_calls;
    *free_bytes = device_free;
    *total_bytes = (size_t)vram_capacity;
    return CUDA_SUCCESS;
}

static IDXGIAdapter3Vtbl adapter_vtable = {
    .QueryVideoMemoryInfo = sample_memory,
};
static IDXGIAdapter3 adapter = { &adapter_vtable };

static void reset_case(void) {
    memset(&context, 0, sizeof(context));
    memset(&local_sample, 0, sizeof(local_sample));
    memset(&nonlocal_sample, 0, sizeof(nonlocal_sample));
    memset(&g_cuda, 0, sizeof(g_cuda));
    set_devctx(&context);
    g_cuda.p_cuMemGetInfo = fake_memory_info;
    g_wddm_adapter = &adapter;
    vram_capacity = MIB(16384);
    total_vram_usage = MIB(10000);
    local_sample.Budget = MIB(15360);
    local_sample.CurrentUsage = MIB(10000);
    nonlocal_sample.CurrentUsage = MIB(64);
    wddm_nonlocal_usage_baseline = MIB(64);
    simple_vram_headroom = MIB(512);
    device_free = MIB(8192);
    sample_calls = memory_calls = 0;
    local_result = S_OK;
    reclaimable_bytes = reclaim_calls = copy_polls = 0;
    reclaim_target = 0;
    requested_reclaim = 0;
    test_tick = 10000;
}

static void test_copy_uses_already_accounted_destination(void) {
    reset_case();
    /* Sixteen MiB of margin is sufficient for a copy into an existing 64 MiB
     * destination. Counting that payload twice would flush all 12 GiB. */
    total_vram_usage = local_sample.CurrentUsage = MIB(15344);
    reclaimable_bytes = MIB(12288);
    CHECK(budget_deficit(MIB(64)) == (ssize_t)MIB(48));
    AimdoXpuCopyPressure pressure = aimdo_xpu_prepare_h2d();
    CHECK(pressure.fit_deficit == -(ssize_t)MIB(16));
    CHECK(pressure.remaining_pages == 0);
    CHECK(reclaim_calls == 0 && requested_reclaim == 0);
    CHECK(copy_polls == 0);
    CHECK(total_vram_usage == MIB(15344));
}

static void test_copy_still_recovers_from_live_pressure(void) {
    reset_case();
    total_vram_usage = local_sample.CurrentUsage = MIB(15424);
    reclaimable_bytes = MIB(256);
    AimdoXpuCopyPressure pressure = aimdo_xpu_prepare_h2d();
    CHECK(pressure.fit_deficit == (ssize_t)MIB(64));
    CHECK(pressure.remaining_pages == 0);
    CHECK(reclaimable_bytes == MIB(192));
    CHECK(reclaim_target == (ssize_t)MIB(64));
    CHECK(pressure.post_reclaim_deficit == 0);
    CHECK(reclaim_calls == 1 && requested_reclaim == 0);

    reset_case();
    total_vram_usage = local_sample.CurrentUsage = MIB(15424);
    pressure = aimdo_xpu_prepare_h2d();
    CHECK(pressure.remaining_pages == 2);
    CHECK(pressure.post_reclaim_deficit == (ssize_t)MIB(64));
    CHECK(reclaim_calls == 1 && requested_reclaim == (ssize_t)MIB(64));

    reset_case();
    total_vram_usage = local_sample.CurrentUsage = MIB(15367);
    reclaimable_bytes = MIB(12288);
    pressure = aimdo_xpu_prepare_h2d();
    CHECK(pressure.fit_deficit == (ssize_t)MIB(7));
    CHECK(pressure.remaining_pages == 0);
    CHECK(reclaim_target == (ssize_t)MIB(7));
    CHECK(reclaimable_bytes == MIB(12256));
    CHECK(pressure.post_reclaim_deficit == -(ssize_t)MIB(25));
    CHECK(requested_reclaim == 0);
}

static void test_os_reservation_is_counted_once(void) {
    /* Expected margins are measured from physical capacity, while a smaller
     * OS budget remains a hard upper bound. Cases straddle both limits. */
    static const struct {
        unsigned budget_mib, usage_mib, reserve_mib;
        int expected_mib;
    } cases[] = {
        {15360, 15104, 512, -256},
        {15360, 14080, 2048, -256},
        {15360, 14592, 2048, 256},
        {16384, 16128, 512, 256},
        {12288, 12544, 2048, 256},
        {12288, 12032, 2048, -256},
    };
    for (size_t i = 0; i < ARRAY_SIZE(cases); ++i) {
        const char *method = NULL;
        reset_case();
        local_sample.Budget = MIB(cases[i].budget_mib);
        local_sample.CurrentUsage = MIB(cases[i].usage_mib);
        simple_vram_headroom = MIB(cases[i].reserve_mib);
        CHECK(poll_budget_deficit(&method));
        CHECK(deficit_sync == (ssize_t)cases[i].expected_mib * 1024 * 1024);
        CHECK(strcmp(method, "WDDM budget") == 0);
    }
}

static void test_physical_pressure_covers_untracked_allocations(void) {
    const char *method = NULL;
    reset_case();
    simple_vram_headroom = MIB(1024);
    device_free = MIB(256);
    CHECK(poll_budget_deficit(&method));
    CHECK(deficit_sync == (ssize_t)MIB(768));
    CHECK(strcmp(method, "cuMemGetInfo (Windows)") == 0);
}

static void test_fallback_and_nonlocal_pressure(void) {
    const char *method = NULL;
    reset_case();
    local_result = E_FAIL;
    total_vram_usage = MIB(14848);
    simple_vram_headroom = MIB(2048);
    CHECK(poll_budget_deficit(&method));
    CHECK(deficit_sync == (ssize_t)MIB(512));

    reset_case();
    g_wddm_adapter = NULL;
    total_vram_usage = MIB(14848);
    simple_vram_headroom = MIB(2048);
    CHECK(poll_budget_deficit(&method));
    CHECK(deficit_sync == (ssize_t)MIB(512));
    CHECK(strcmp(method, "physical capacity") == 0);

    reset_case();
    nonlocal_sample.CurrentUsage = MIB(192);
    CHECK(poll_budget_deficit(&method));
    CHECK(deficit_sync < 0);
    test_tick += 2000;
    nonlocal_sample.CurrentUsage = MIB(704);
    CHECK(poll_budget_deficit(&method));
    CHECK(deficit_sync == (ssize_t)MIB(640));
    CHECK(strcmp(method, "WDDM non-local usage") == 0);
}

static void test_cached_sample_tracks_allocations_and_frees(void) {
    reset_case();
    simple_vram_headroom = MIB(2048);
    local_sample.CurrentUsage = MIB(14080);
    CHECK(budget_deficit(MIB(512)) == (ssize_t)MIB(256));
    CHECK(sample_calls == 2 && memory_calls == 1);

    total_vram_usage += MIB(128);
    test_tick += 1999;
    CHECK(budget_deficit(MIB(512)) == (ssize_t)MIB(384));
    CHECK(sample_calls == 2 && memory_calls == 1);

    total_vram_usage -= MIB(1024);
    CHECK(budget_deficit(MIB(512)) == -(ssize_t)MIB(640));
    CHECK(sample_calls == 2 && memory_calls == 1);

    /* A model boundary must sample even before the normal poll interval. */
    aimdo_wddm_force_poll();
    local_sample.CurrentUsage = MIB(12032);
    CHECK(budget_deficit(MIB(512)) == -(ssize_t)MIB(1792));
    CHECK(sample_calls == 4 && memory_calls == 2);
    CHECK(total_vram_last_check == total_vram_usage);

    test_tick += 2000;
    extra_vram_headroom = MIB(256);
    CHECK(budget_deficit(MIB(512)) == -(ssize_t)MIB(1536));
    CHECK(sample_calls == 6 && memory_calls == 3);
}

static void emit_normal_log(void) { log(INFO, "normal\n"); }
static void emit_shot_log(void) { log_shot(INFO, "shot\n"); }

static void test_normal_logging_does_not_require_a_shot_reset(void) {
    log_calls = 0;
    log_shot_counter = 0;
    for (unsigned i = 0; i < 3; ++i) emit_normal_log();
    CHECK(log_calls == 3);
    log_shot_counter = 1;
    for (unsigned i = 0; i < 3; ++i) emit_shot_log();
    CHECK(log_calls == 4);
    ++log_shot_counter;
    emit_shot_log();
    CHECK(log_calls == 5);
    log_level = WARNING;
    emit_normal_log();
    CHECK(log_calls == 5);
}

int main(void) {
    test_copy_uses_already_accounted_destination();
    test_copy_still_recovers_from_live_pressure();
    test_os_reservation_is_counted_once();
    test_physical_pressure_covers_untracked_allocations();
    test_fallback_and_nonlocal_pressure();
    test_cached_sample_tracks_allocations_and_frees();
    test_normal_logging_does_not_require_a_shot_reset();
    printf("Windows pressure native tests: %s (%u failures)\n",
           failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
