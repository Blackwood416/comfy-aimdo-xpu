#include "plat.h"
#include "xfer-file.h"

#define HOSTBUF_FILE_READER_WINDOW (64ULL * 1024ULL * 1024ULL)
#define LEAD_IN_THRESHOLD (HOSTBUF_FILE_READER_WINDOW - 16ULL * 1024ULL * 1024ULL)

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
#include <windows.h>
#include "xpu-copy-pressure.h"

/* Diagnostic switch for the streaming-path reclaim, so its effect can be
 * measured on the real workload instead of assumed. */
static bool aimdo_stream_reclaim_disabled(void) {
    static int disabled = -1;
    char value[8];

    if (disabled < 0) {
        DWORD length = GetEnvironmentVariableA(
            "AIMDO_XPU_NO_STREAM_RECLAIM", value, sizeof(value));
        disabled = (length > 0 && length < sizeof(value) && value[0] == '1');
    }
    return disabled != 0;
}

static bool aimdo_stream_reclaim_trace_enabled(void) {
    static int enabled = -1;
    char value[8];

    if (enabled < 0) {
        DWORD length = GetEnvironmentVariableA(
            "AIMDO_XPU_SYNC_TRACE", value, sizeof(value));
        enabled = length > 0 && length < sizeof(value) && value[0] == '1';
    }
    return enabled != 0;
}
#endif

static bool hostbuf_file_reader_retire_active(void) {
    HostbufFileReaderSlot *slot;
    CUevent event = NULL;
    CUresult recorded;

    if (g_devctx->_hostbuf_file_reader_active < 0) {
        return true;
    }

    slot = &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
    if (!slot->offset) {
        return true;
    }
    if (slot->event) {
        // A failed reuse/cleanup leaves this slot retired and retryable.
        return true;
    }
    if (!CHECK_CU(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING))) {
        return false;
    }
#if defined(AIMDO_XPU)
    recorded = aimdo_xpu_record_reader_event(event, slot->buffer);
#else
    recorded = cuEventRecord(event, (CUstream)slot->stream);
#endif
    if (!CHECK_CU(recorded)) {
        CHECK_CU(cuEventDestroy(event));
        return false;
    }
    slot->event = event;
    return true;
}

static HostbufFileReaderSlot *hostbuf_file_reader_next(cudaStream_t stream) {
    HostbufFileReaderSlot *slot;
    int next = (g_devctx->_hostbuf_file_reader_active + 1) % HOSTBUF_FILE_READER_SLOTS;

    slot = &g_devctx->_hostbuf_file_reader_slots[next];

    if (slot->buffer && slot->event) {
        if (!CHECK_CU(cuEventSynchronize(slot->event)) ||
            !CHECK_CU(cuEventDestroy(slot->event))) {
            return NULL;
        }
        slot->event = NULL;
    }

    if (!slot->buffer &&
        !CHECK_CU_OOM_ERROR(cuMemAllocHost((void **)&slot->buffer, HOSTBUF_FILE_READER_WINDOW))) {
        return NULL;
    }

    slot->offset = 0;
    slot->stream = (CUstream)stream;
    g_devctx->_hostbuf_file_reader_active = next;
    return slot;
}

SHARED_EXPORT
bool hostbuf_file_reader_read(int device, uint64_t file_handle, uint64_t file_offset,
                              uint64_t size, cudaStream_t stream,
                              uint64_t device_ptr, bool mark_cold) {
    if (size == 0) {
        return true;
    }
    if (!device_ptr || device < 0 || !set_devctx_for_device(device)) {
        log(AIMDO_LOG_ERROR, "%s: input validation failed device_ptr=%p device=%d\n",
            __func__, (void *)(uintptr_t)device_ptr, device);
        return false;
    }

    while (size) {
        HostbufFileReaderSlot *slot = g_devctx->_hostbuf_file_reader_active < 0 ? NULL :
            &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
        size_t chunk;

        if (!slot || slot->event || slot->stream != (CUstream)stream ||
            (slot->offset + size >= HOSTBUF_FILE_READER_WINDOW &&
             slot->offset >= LEAD_IN_THRESHOLD)) {
            if (!hostbuf_file_reader_retire_active() ||
                !(slot = hostbuf_file_reader_next(stream))) {
                return false;
            }
        }

        chunk = (size_t)MIN(size, HOSTBUF_FILE_READER_WINDOW - slot->offset);
        if (!xfer_file_read(file_handle, file_offset, slot->buffer + slot->offset,
                            chunk, mark_cold)) {
            log(AIMDO_LOG_ERROR, "%s: file read failed handle=0x%llx offset=%llu size=%zu\n",
                __func__, (ull)file_handle, (ull)file_offset, chunk);
            return false;
        }
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        /* Reclaim the live shortage before submitting the copy. The VBAR
         * destination is already accounted and pinned; pending copies also
         * hold its siblings. The scan never waits for unfinished consumers. */
        if (!aimdo_stream_reclaim_disabled()) {
            AimdoXpuCopyPressure pressure = aimdo_xpu_prepare_h2d();
            if (aimdo_stream_reclaim_trace_enabled()) {
                fprintf(stderr,
                        "[AIMDO XPU RECLAIM] op=pre_h2d destination=%p "
                        "size=%zu fit_deficit=%lld remaining_pages=%zu "
                        "post_reclaim_deficit=%lld\n",
                        (void *)(uintptr_t)device_ptr, chunk,
                        (long long)pressure.fit_deficit,
                        pressure.remaining_pages,
                        (long long)pressure.post_reclaim_deficit);
                fflush(stderr);
            }
        }
#endif
        CUresult copy_result = cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,
                                                 slot->buffer + slot->offset,
                                                 chunk, (CUstream)stream);
        // A failed submission can still own its source. Reserve the slice
        // until the slot's completion token has been checked on reuse.
        slot->offset += chunk;
        if (!CHECK_CU(copy_result)) {
            log(AIMDO_LOG_ERROR, "%s: device copy failed result=%d device_ptr=%p device=%d stream=%p size=%zu\n",
                __func__, (int)copy_result, (void *)(uintptr_t)device_ptr, device,
                (void *)stream, chunk);
            return false;
        }

        file_offset += chunk;
        device_ptr += chunk;
        size -= chunk;
    }

    return true;
}

SHARED_EXPORT
bool hostbuf_file_reader_cleanup_checked(void) {
    bool complete = true;
    if (!g_devctx) {
        return true;
    }

    if (!hostbuf_file_reader_retire_active()) {
        return false;
    }
    for (unsigned i = 0; i < HOSTBUF_FILE_READER_SLOTS; i++) {
        HostbufFileReaderSlot *slot = &g_devctx->_hostbuf_file_reader_slots[i];

        if (slot->buffer && slot->event) {
            if (!CHECK_CU(cuEventSynchronize(slot->event))) {
                complete = false;
                continue;
            }
            if (!CHECK_CU(cuEventDestroy(slot->event))) {
                complete = false;
                continue;
            }
            slot->event = NULL;
        }
        if (slot->buffer && !CHECK_CU(cuMemFreeHost(slot->buffer))) {
            complete = false;
            continue;
        }
        memset(slot, 0, sizeof(*slot));
        if ((int)i == g_devctx->_hostbuf_file_reader_active) {
            g_devctx->_hostbuf_file_reader_active = -1;
        }
    }
    return complete;
}

SHARED_EXPORT
void hostbuf_file_reader_cleanup(void) {
    (void)hostbuf_file_reader_cleanup_checked();
}
