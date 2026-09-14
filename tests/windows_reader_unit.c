/* Exercise the production ring with delayed copies and injected failures.
 * No GPU or real files: a pending copy detects any premature staging reuse. */
#include "../src-win/compiler.h"
#include "../src/plat.h"
#include "../src/hostbuf-file-reader.c"

AimdoCudaDispatch g_cuda;
_Thread_local AimdoContext *g_devctx;
int64_t simple_vram_headroom;
int log_level = DEBUG;
uint64_t log_shot_counter;
static AimdoContext context;
static int failures, create_fail, record_fail, wait_fail, copy_fail;
static unsigned reads, copies, frees, waits;

struct CUevent_st { void *buffer; };
struct Pending { const uint8_t *source; uint8_t expected; bool complete; };
static struct Pending pending[128];

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++failures; \
} } while (0)

void aimdo_log(int level, const char *file, int line, const char *format, ...) {}
bool set_devctx_for_device(int device) { g_devctx = &context; return device == 0; }
bool poll_budget_deficit(const char **method) { *method = "test"; return true; }
size_t vbars_free_retired(ssize_t size) { return 0; }
bool aimdo_xpu_copy_residency_poll(bool wait) { return true; }
void vbars_request_reclaim(ssize_t size) {}

bool xfer_file_read(uint64_t handle, uint64_t offset, void *destination,
                    size_t size, bool cold) {
    ++reads;
    ((uint8_t *)destination)[0] = (uint8_t)(offset * 37 + 11);
    return true;
}

static CUresult fake_error(CUresult result, const char **message) {
    *message = "injected";
    return CUDA_SUCCESS;
}
static CUresult fake_alloc(void **pointer, size_t size) {
    *pointer = malloc(size);
    return *pointer ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
static bool belongs(const struct Pending *p, const void *buffer) {
    uintptr_t address = (uintptr_t)p->source, base = (uintptr_t)buffer;
    return address >= base && address - base < HOSTBUF_FILE_READER_WINDOW;
}
static CUresult fake_free(void *buffer) {
    for (unsigned i = 0; i < copies; ++i) {
        CHECK(!belongs(&pending[i], buffer) || pending[i].complete);
    }
    ++frees;
    free(buffer);
    return CUDA_SUCCESS;
}
static CUresult fake_copy(CUdeviceptr dest, const void *source, size_t size, CUstream stream) {
    CHECK(copies < 128);
    pending[copies].source = source;
    pending[copies].expected = ((const uint8_t *)source)[0];
    pending[copies++].complete = false;
    return copy_fail ? 999 : CUDA_SUCCESS;
}
static CUresult fake_create(CUevent *event, unsigned flags) {
    if (create_fail) return 999;
    *event = calloc(1, sizeof(**event));
    return CUDA_SUCCESS;
}
CUresult aimdo_xpu_record_reader_event(CUevent event, void *buffer) {
    if (record_fail) return 999;
    event->buffer = buffer;
    return CUDA_SUCCESS;
}
static CUresult fake_wait(CUevent event) {
    ++waits;
    if (wait_fail) return 999;
    for (unsigned i = 0; i < copies; ++i) {
        if (belongs(&pending[i], event->buffer) && !pending[i].complete) {
            CHECK(*pending[i].source == pending[i].expected);
            pending[i].complete = true;
        }
    }
    return CUDA_SUCCESS;
}
static CUresult fake_destroy(CUevent event) { free(event); return CUDA_SUCCESS; }

static void reset(void) {
    memset(&context, 0, sizeof(context));
    memset(pending, 0, sizeof(pending));
    g_devctx = &context;
    context._hostbuf_file_reader_active = -1;
    context._vram_capacity = 16ULL << 30;
    context._deficit_sync = -(1LL << 30);
    create_fail = record_fail = wait_fail = copy_fail = 0;
    reads = copies = frees = waits = 0;
}
static bool read_slice(unsigned id, unsigned stream, size_t size) {
    return hostbuf_file_reader_read(0, 1, id, size,
        (cudaStream_t)(uintptr_t)(stream + 1), 4096, false);
}

int main(void) {
    g_cuda.p_cuGetErrorString = fake_error;
    g_cuda.p_cuMemAllocHost = fake_alloc;
    g_cuda.p_cuMemFreeHost = fake_free;
    g_cuda.p_cuMemcpyHtoDAsync = fake_copy;
    g_cuda.p_cuEventCreate = fake_create;
    g_cuda.p_cuEventSynchronize = fake_wait;
    g_cuda.p_cuEventDestroy = fake_destroy;

    reset();
    for (unsigned i = 0; i < 12; ++i) CHECK(read_slice(i, i % 2, HOSTBUF_FILE_READER_WINDOW));
    CHECK(hostbuf_file_reader_cleanup_checked());
    CHECK(copies == 12 && frees == 3 && waits >= 12);

    reset();
    for (unsigned i = 0; i < 3; ++i) CHECK(read_slice(i, 0, HOSTBUF_FILE_READER_WINDOW));
    wait_fail = 1;
    CHECK(!read_slice(4, 0, HOSTBUF_FILE_READER_WINDOW));
    CHECK(reads == 3 && copies == 3);
    CHECK(!hostbuf_file_reader_cleanup_checked());
    CHECK(frees == 0);
    wait_fail = 0;
    CHECK(hostbuf_file_reader_cleanup_checked());
    CHECK(frees == 3);

    reset();
    CHECK(read_slice(0, 0, 17));
    create_fail = 1;
    CHECK(!read_slice(1, 1, 17));
    CHECK(copies == 1 && reads == 1);
    create_fail = 0;
    CHECK(hostbuf_file_reader_cleanup_checked());
    CHECK(frees == 1);

    reset();
    CHECK(read_slice(0, 0, 17));
    record_fail = 1;
    CHECK(!read_slice(1, 1, 17));
    CHECK(!hostbuf_file_reader_cleanup_checked());
    CHECK(frees == 0);
    record_fail = 0;
    CHECK(hostbuf_file_reader_cleanup_checked());
    CHECK(frees == 1);

    reset();
    copy_fail = 1;
    CHECK(!read_slice(7, 0, 17));
    CHECK(context._hostbuf_file_reader_slots[0].offset == 17);
    copy_fail = 0;
    CHECK(hostbuf_file_reader_cleanup_checked());
    CHECK(waits == 1 && frees == 1);
    printf("reader lifecycle %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
