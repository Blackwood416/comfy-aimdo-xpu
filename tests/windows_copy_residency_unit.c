/* Drive the production copy hook with delayed and failed UR completions. */
#include <stdio.h>
#include "../src-xpu/ur-copy-residency.c"

static int failures, acquire_count = 1, holds, calls, event_refs, queue_refs;
static int event_state = 3, query_result, submit_result, wait_result, retain_result;
static int event_waits, queue_waits;
static bool return_event = true;
static size_t last_dst_size, last_src_size;
static int event_storage, queue_storage;

#define CHECK(x) do { if (!(x)) { printf("FAIL %s:%d %s\n", __func__, __LINE__, #x); failures++; } } while (0)

void *aimdo_xpu_ur_loader(void) { return NULL; }
int aimdo_vbar_copy_acquire(const void *dst, size_t ds, const void *src,
                           size_t ss, uint64_t *ids, size_t capacity) {
    (void)dst; (void)src; (void)capacity;
    last_dst_size = ds;
    last_src_size = ss;
    for (int i = 0; i < acquire_count; ++i) { ids[i] = i + 1; }
    holds += acquire_count > 0 ? acquire_count : 0;
    return acquire_count;
}
void aimdo_vbar_copy_release(const uint64_t *ids, size_t count) {
    CHECK(ids[0] == 1);
    holds -= (int)count;
}
static CopyResult __cdecl fake_copy(CopyQueue q, bool blocking, void *dst,
    const void *src, size_t size, uint32_t n, const CopyEvent *wait, CopyEvent *event) {
    (void)q; (void)blocking; (void)dst; (void)src; (void)size; (void)n; (void)wait;
    calls++;
    if (event && return_event) { *event = &event_storage; event_refs++; }
    return submit_result;
}
static CopyResult __cdecl fake_copy_2d(CopyQueue q, bool blocking, void *dst,
    size_t dp, const void *src, size_t sp, size_t width, size_t height,
    uint32_t n, const CopyEvent *wait, CopyEvent *event) {
    (void)dp; (void)sp;
    return fake_copy(q, blocking, dst, src, width * height, n, wait, event);
}
static CopyResult __cdecl fake_info(CopyEvent e, int property, size_t size, void *out, size_t *ret) {
    (void)e; (void)ret;
    CHECK(property == 3 && size == sizeof(int));
    *(int *)out = event_state;
    return query_result;
}
static CopyResult __cdecl fake_fill(CopyQueue q, void *dst, size_t ps,
    const void *pattern, size_t size, uint32_t n, const CopyEvent *wait, CopyEvent *event) {
    (void)ps;
    return fake_copy(q, false, dst, pattern, size, n, wait, event);
}
static CopyResult __cdecl fake_fill_2d(CopyQueue q, void *dst, size_t pitch,
    size_t ps, const void *pattern, size_t width, size_t height,
    uint32_t n, const CopyEvent *wait, CopyEvent *event) {
    (void)pitch; (void)ps;
    return fake_copy(q, false, dst, pattern, width * height, n, wait, event);
}
static CopyResult __cdecl fake_event_retain(void *e) {
    (void)e;
    if (!retain_result) { event_refs++; }
    return retain_result;
}
static CopyResult __cdecl fake_event_release(void *e) { (void)e; event_refs--; return 0; }
static CopyResult __cdecl fake_queue_retain(void *q) { (void)q; queue_refs++; return 0; }
static CopyResult __cdecl fake_queue_release(void *q) { (void)q; queue_refs--; return 0; }
static CopyResult __cdecl fake_wait(uint32_t n, const CopyEvent *e) {
    (void)e; CHECK(n == 1); event_waits++; return wait_result;
}
static CopyResult __cdecl fake_finish(void *q) { (void)q; queue_waits++; return wait_result; }

int main(void) {
    original_copy = fake_copy;
    original_copy_2d = fake_copy_2d;
    original_fill = fake_fill;
    original_fill_2d = fake_fill_2d;
    event_info = fake_info;
    event_retain = fake_event_retain;
    event_release = fake_event_release;
    event_wait = fake_wait;
    queue_retain = fake_queue_retain;
    queue_release = fake_queue_release;
    queue_finish = fake_finish;
    CopyEvent caller_event = NULL;

    CHECK(tracked_copy(&queue_storage, false, (void *)100, (void *)200, 128, 0, NULL, &caller_event) == 0);
    CHECK(holds == 1 && event_refs == 2 && queue_refs == 1);
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 1 && event_waits == 0 && queue_waits == 0);
    event_state = 0;
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 0 && event_refs == 1 && queue_refs == 0);
    fake_event_release(caller_event);

    acquire_count = 2;
    CHECK(tracked_copy_2d(&queue_storage, false, (void *)100, 64, (void *)200, 32, 16, 3, 0, NULL, NULL) == 0);
    CHECK(last_dst_size == 192 && last_src_size == 96 && holds == 2);
    query_result = 1;
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 2);
    query_result = 0;
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 0 && event_refs == 0);
    acquire_count = 1;

    submit_result = 39;
    return_event = false;
    CHECK(tracked_copy(&queue_storage, false, (void *)100, (void *)200, 128, 0, NULL, NULL) == 39);
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 1 && queue_waits == 0);
    wait_result = 1;
    CHECK(!aimdo_xpu_copy_residency_poll(true));
    CHECK(holds == 1);
    wait_result = 0;
    CHECK(aimdo_xpu_copy_residency_poll(true));
    CHECK(holds == 0 && queue_refs == 0);
    submit_result = 0;
    return_event = true;

    /* A failed submission's event need not cover all partially appended work. */
    submit_result = 39;
    CHECK(tracked_copy(&queue_storage, false, (void *)100, (void *)200, 128, 0, NULL, NULL) == 39);
    event_state = 0;
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 1);
    CHECK(aimdo_xpu_copy_residency_poll(true));
    CHECK(holds == 0 && event_refs == 0);
    submit_result = 0;

    retain_result = 1;
    CHECK(tracked_copy(&queue_storage, false, (void *)100, (void *)200, 128, 0, NULL, &caller_event) == 0);
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 1);
    CHECK(aimdo_xpu_copy_residency_poll(true));
    CHECK(holds == 0 && event_refs == 1);
    fake_event_release(caller_event);
    retain_result = 0;

    event_state = 3;
    CHECK(tracked_copy(&queue_storage, true, (void *)100, (void *)200, 128, 0, NULL, NULL) == 0);
    CHECK(holds == 1); /* Callback itself never takes the VBAR release lock. */
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 0 && event_refs == 0 && queue_refs == 0);

    CHECK(tracked_fill(&queue_storage, (void *)100, 1, (void *)200, 128, 0, NULL, NULL) == 0);
    CHECK(last_dst_size == 128 && last_src_size == 0 && holds == 1);
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 1);
    event_state = 0;
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 0 && event_refs == 0);
    CHECK(tracked_fill_2d(&queue_storage, (void *)100, 64, 1, (void *)200, 16, 3, 0, NULL, NULL) == 0);
    CHECK(last_dst_size == 192 && last_src_size == 0 && holds == 1);
    CHECK(aimdo_xpu_copy_residency_poll(false));
    CHECK(holds == 0 && event_refs == 0);

    acquire_count = -1;
    int before = calls;
    CHECK(tracked_copy(&queue_storage, false, (void *)100, (void *)200, 128, 0, NULL, NULL) == 38);
    CHECK(calls == before && holds == 0);
    acquire_count = 0;
    return_event = false;
    CHECK(tracked_copy(&queue_storage, false, (void *)100, (void *)200, 128, 0, NULL, NULL) == 0);
    CHECK(calls == before + 1 && holds == 0);

    uint64_t stats[5];
    CHECK(xpu_copy_residency_get_stats(stats, 5));
    CHECK(stats[2] == 0 && stats[0] == stats[1] && pending_copies == NULL);
    printf("copy residency %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures != 0;
}
