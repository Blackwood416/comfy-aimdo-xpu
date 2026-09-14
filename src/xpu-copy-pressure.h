#ifndef AIMDO_XPU_COPY_PRESSURE_H
#define AIMDO_XPU_COPY_PRESSURE_H

/* Called on the direct file reader's model-owner stack, never in an allocator
 * callback. The destination is already allocated (and VBAR destinations are
 * pinned), so the copy payload is not another device-memory allocation. */
typedef struct AimdoXpuCopyPressure {
    ssize_t fit_deficit;
    ssize_t post_reclaim_deficit;
    size_t remaining_pages;
} AimdoXpuCopyPressure;

static inline AimdoXpuCopyPressure aimdo_xpu_prepare_h2d(void) {
    AimdoXpuCopyPressure pressure = {0};

    pressure.fit_deficit = budget_deficit(0);
    pressure.post_reclaim_deficit = pressure.fit_deficit;
    if (pressure.fit_deficit > 0) {
        /* A small shortage must not invalidate the entire model. Completed
         * copy owners can now release sibling-page holds before the existing
         * bounded reclaim selects pages whose consumers have retired. */
        (void)aimdo_xpu_copy_residency_poll(false);
        pressure.remaining_pages = vbars_free_retired(pressure.fit_deficit);
        pressure.post_reclaim_deficit = budget_deficit(0);
        if (pressure.post_reclaim_deficit > 0) {
            vbars_request_reclaim(pressure.post_reclaim_deficit);
        }
    }
    return pressure;
}

#endif
