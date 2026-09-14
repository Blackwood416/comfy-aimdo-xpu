/* A Level Zero copy names every physical mapping in each VMM reservation it
 * touches. A page's last kernel event therefore does not prove that a later
 * copy into a sibling page has stopped referring to its driver allocation.
 * Observe the existing UR copy events and retain reservation ownership until
 * completion. No wait or VMM operation runs on the submission callback stack.
 */
#include <windows.h>
#include <detours.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef int CopyResult;
typedef void *CopyQueue;
typedef void *CopyEvent;
typedef CopyResult (__cdecl *CopyFn)(CopyQueue, bool, void *, const void *,
                                    size_t, uint32_t, const CopyEvent *, CopyEvent *);
typedef CopyResult (__cdecl *Copy2DFn)(CopyQueue, bool, void *, size_t,
                                      const void *, size_t, size_t, size_t,
                                      uint32_t, const CopyEvent *, CopyEvent *);
typedef CopyResult (__cdecl *FillFn)(CopyQueue, void *, size_t, const void *,
                                    size_t, uint32_t, const CopyEvent *, CopyEvent *);
typedef CopyResult (__cdecl *Fill2DFn)(CopyQueue, void *, size_t, size_t,
                                      const void *, size_t, size_t, uint32_t,
                                      const CopyEvent *, CopyEvent *);
typedef CopyResult (__cdecl *EventInfoFn)(CopyEvent, int, size_t, void *, size_t *);
typedef CopyResult (__cdecl *EventWaitFn)(uint32_t, const CopyEvent *);
typedef CopyResult (__cdecl *HandleFn)(void *);

extern void *aimdo_xpu_ur_loader(void);
extern int aimdo_vbar_copy_acquire(const void *, size_t, const void *, size_t,
                                  uint64_t *, size_t);
extern void aimdo_vbar_copy_release(const uint64_t *, size_t);

typedef struct CopyOwner {
    struct CopyOwner *next;
    uint64_t reservations[2];
    size_t count;
    CopyQueue queue;
    CopyEvent event;
    bool complete;
    bool require_queue_wait;
} CopyOwner;

static CopyFn original_copy;
static Copy2DFn original_copy_2d;
static FillFn original_fill;
static Fill2DFn original_fill_2d;
static EventInfoFn event_info;
static EventWaitFn event_wait;
static HandleFn event_retain, event_release, queue_retain, queue_release, queue_finish;
static SRWLOCK copy_lock = SRWLOCK_INIT;
static CopyOwner *pending_copies;
static bool copy_hook_attached;
static volatile LONG64 copy_stats[5]; /* tracked, completed, pending, failed, polls */

static void keep_copy(CopyOwner *owner) {
    AcquireSRWLockExclusive(&copy_lock);
    owner->next = pending_copies;
    pending_copies = owner;
    ReleaseSRWLockExclusive(&copy_lock);
}

static void release_copy(CopyOwner *owner) {
    aimdo_vbar_copy_release(owner->reservations, owner->count);
    if (owner->event) {
        event_release(owner->event);
    }
    if (owner->queue) {
        queue_release(owner->queue);
    }
    InterlockedIncrement64(&copy_stats[1]);
    InterlockedDecrement64(&copy_stats[2]);
    free(owner);
}

/* A negative result refuses submission before any memory can be consumed.
 * The VBAR lookup uses try-lock to avoid inversion with a SYCL queue lock. */
static int prepare_copy(CopyQueue queue, void *dst, size_t dst_size,
                         const void *src, size_t src_size, CopyOwner **out) {
    CopyOwner *owner = calloc(1, sizeof(*owner));
    if (!owner) {
        return -1;
    }
    int count = aimdo_vbar_copy_acquire(
        dst, dst_size, src, src_size, owner->reservations, 2);
    if (count <= 0) {
        free(owner);
        return count;
    }
    owner->count = (size_t)count;
    InterlockedIncrement64(&copy_stats[0]);
    InterlockedIncrement64(&copy_stats[2]);
    if (queue_retain(queue) != 0) {
        /* Release the CPU hold later, outside the caller's SYCL queue lock. */
        owner->complete = true;
        keep_copy(owner);
        return -1;
    }
    owner->queue = queue;
    *out = owner;
    return count;
}

static void finish_copy(CopyOwner *owner, CopyResult result, bool blocking,
                         CopyEvent submitted, CopyEvent *caller_event) {
    if (caller_event) {
        *caller_event = submitted;
    }
    if (submitted && (!caller_event || event_retain(submitted) == 0)) {
        owner->event = submitted;
    }
    owner->complete = result == 0 && blocking;
    owner->require_queue_wait = result != 0;
    if (result != 0) {
        /* Even a failed call may have appended work. An owner-side queue
         * wait is required when no valid completion event was returned. */
        InterlockedIncrement64(&copy_stats[3]);
    }
    keep_copy(owner);
}

static CopyResult __cdecl tracked_copy(
    CopyQueue queue, bool blocking, void *dst, const void *src, size_t size,
    uint32_t wait_count, const CopyEvent *wait_list, CopyEvent *out_event) {
    CopyOwner *owner = NULL;
    int tracked = prepare_copy(queue, dst, size, src, size, &owner);
    if (tracked < 0) {
        return 38; /* UR_RESULT_ERROR_OUT_OF_HOST_MEMORY; nothing submitted */
    }
    if (!tracked) {
        return original_copy(queue, blocking, dst, src, size,
                             wait_count, wait_list, out_event);
    }
    CopyEvent event = NULL;
    CopyResult result = original_copy(queue, blocking, dst, src, size,
                                      wait_count, wait_list, &event);
    finish_copy(owner, result, blocking, event, out_event);
    return result;
}

static CopyResult __cdecl tracked_copy_2d(
    CopyQueue queue, bool blocking, void *dst, size_t dst_pitch,
    const void *src, size_t src_pitch, size_t width, size_t height,
    uint32_t wait_count, const CopyEvent *wait_list, CopyEvent *out_event) {
    if ((height && dst_pitch > SIZE_MAX / height) ||
        (height && src_pitch > SIZE_MAX / height)) {
        return 38;
    }
    CopyOwner *owner = NULL;
    int tracked = prepare_copy(queue, dst, dst_pitch * height,
                                src, src_pitch * height, &owner);
    if (tracked < 0) {
        return 38;
    }
    if (!tracked) {
        return original_copy_2d(queue, blocking, dst, dst_pitch, src, src_pitch,
                                width, height, wait_count, wait_list, out_event);
    }
    CopyEvent event = NULL;
    CopyResult result = original_copy_2d(queue, blocking, dst, dst_pitch,
        src, src_pitch, width, height, wait_count, wait_list, &event);
    finish_copy(owner, result, blocking, event, out_event);
    return result;
}

/* Level Zero fill commands use the same reservation residency expansion. */
static CopyResult __cdecl tracked_fill(
    CopyQueue queue, void *dst, size_t pattern_size, const void *pattern,
    size_t size, uint32_t wait_count, const CopyEvent *wait_list, CopyEvent *out_event) {
    CopyOwner *owner = NULL;
    int tracked = prepare_copy(queue, dst, size, NULL, 0, &owner);
    if (tracked < 0) {
        return 38;
    }
    if (!tracked) {
        return original_fill(queue, dst, pattern_size, pattern, size,
                              wait_count, wait_list, out_event);
    }
    CopyEvent event = NULL;
    CopyResult result = original_fill(queue, dst, pattern_size, pattern, size,
                                      wait_count, wait_list, &event);
    finish_copy(owner, result, false, event, out_event);
    return result;
}

static CopyResult __cdecl tracked_fill_2d(
    CopyQueue queue, void *dst, size_t pitch, size_t pattern_size,
    const void *pattern, size_t width, size_t height, uint32_t wait_count,
    const CopyEvent *wait_list, CopyEvent *out_event) {
    if (height && pitch > SIZE_MAX / height) {
        return 38;
    }
    CopyOwner *owner = NULL;
    int tracked = prepare_copy(queue, dst, pitch * height, NULL, 0, &owner);
    if (tracked < 0) {
        return 38;
    }
    if (!tracked) {
        return original_fill_2d(queue, dst, pitch, pattern_size, pattern,
            width, height, wait_count, wait_list, out_event);
    }
    CopyEvent event = NULL;
    CopyResult result = original_fill_2d(queue, dst, pitch, pattern_size, pattern,
        width, height, wait_count, wait_list, &event);
    finish_copy(owner, result, false, event, out_event);
    return result;
}

/* Model-owner call only, outside VBAR, allocator and submission locks. Normal
 * reclaim polls; existing explicit synchronize/teardown boundaries may wait. */
bool aimdo_xpu_copy_residency_poll(bool wait) {
    CopyOwner *list;
    bool success = true;
    AcquireSRWLockExclusive(&copy_lock);
    list = pending_copies;
    pending_copies = NULL;
    ReleaseSRWLockExclusive(&copy_lock);
    while (list) {
        CopyOwner *owner = list;
        list = owner->next;
        bool complete = owner->complete;
        if (!complete && wait) {
            complete = (owner->event && !owner->require_queue_wait
                ? event_wait(1, &owner->event) : queue_finish(owner->queue)) == 0;
            success = success && complete;
        } else if (!complete && owner->event && !owner->require_queue_wait) {
            int status = 3; /* UR_EXECUTION_INFO_QUEUED */
            InterlockedIncrement64(&copy_stats[4]);
            complete = event_info(owner->event, 3, sizeof(status), &status, NULL) == 0
                       && status == 0; /* UR_EVENT_INFO_COMMAND_EXECUTION_STATUS */
        }
        if (complete) {
            release_copy(owner);
        } else {
            keep_copy(owner);
        }
    }
    return success;
}

__declspec(dllexport) bool xpu_copy_residency_get_stats(uint64_t *out, size_t count) {
    if (!out || count < 5) {
        return false;
    }
    for (size_t i = 0; i < 5; ++i) {
        out[i] = (uint64_t)InterlockedCompareExchange64(&copy_stats[i], 0, 0);
    }
    return true;
}

bool aimdo_xpu_copy_residency_install(void) {
    if (copy_hook_attached) {
        return true;
    }
    HMODULE loader = (HMODULE)aimdo_xpu_ur_loader();
    if (!loader) {
        return false;
    }
    original_copy = (CopyFn)(void *)GetProcAddress(loader, "urEnqueueUSMMemcpy");
    original_copy_2d = (Copy2DFn)(void *)GetProcAddress(loader, "urEnqueueUSMMemcpy2D");
    original_fill = (FillFn)(void *)GetProcAddress(loader, "urEnqueueUSMFill");
    original_fill_2d = (Fill2DFn)(void *)GetProcAddress(loader, "urEnqueueUSMFill2D");
    event_info = (EventInfoFn)(void *)GetProcAddress(loader, "urEventGetInfo");
    event_wait = (EventWaitFn)(void *)GetProcAddress(loader, "urEventWait");
    event_retain = (HandleFn)(void *)GetProcAddress(loader, "urEventRetain");
    event_release = (HandleFn)(void *)GetProcAddress(loader, "urEventRelease");
    queue_retain = (HandleFn)(void *)GetProcAddress(loader, "urQueueRetain");
    queue_release = (HandleFn)(void *)GetProcAddress(loader, "urQueueRelease");
    queue_finish = (HandleFn)(void *)GetProcAddress(loader, "urQueueFinish");
    if (!original_copy || !original_copy_2d || !original_fill || !original_fill_2d || !event_info || !event_wait ||
        !event_retain || !event_release || !queue_retain || !queue_release ||
        !queue_finish) {
        return false;
    }
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    LONG status = DetourAttach((void **)&original_copy, tracked_copy);
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&original_copy_2d, tracked_copy_2d);
    }
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&original_fill, tracked_fill);
    }
    if (status == NO_ERROR) {
        status = DetourAttach((void **)&original_fill_2d, tracked_fill_2d);
    }
    if (status != NO_ERROR) {
        DetourTransactionAbort();
        return false;
    }
    if (DetourTransactionCommit() != NO_ERROR) {
        return false;
    }
    copy_hook_attached = true;
    return true;
}

void aimdo_xpu_copy_residency_remove(void) {
    if (!copy_hook_attached || !aimdo_xpu_copy_residency_poll(true)) {
        return;
    }
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    LONG status = DetourDetach((void **)&original_copy, tracked_copy);
    if (status == NO_ERROR) {
        status = DetourDetach((void **)&original_copy_2d, tracked_copy_2d);
    }
    if (status == NO_ERROR) {
        status = DetourDetach((void **)&original_fill, tracked_fill);
    }
    if (status == NO_ERROR) {
        status = DetourDetach((void **)&original_fill_2d, tracked_fill_2d);
    }
    if (status != NO_ERROR) {
        DetourTransactionAbort();
        return;
    }
    if (DetourTransactionCommit() == NO_ERROR) {
        copy_hook_attached = false;
    }
}
