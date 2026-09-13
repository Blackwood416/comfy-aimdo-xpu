#ifndef AIMDO_XPU_COPY_PRESSURE_H
#define AIMDO_XPU_COPY_PRESSURE_H

/* Called on the direct file reader's model-owner stack, never in an allocator
 * callback. The destination is already allocated (and VBAR destinations are
 * pinned), so the copy payload is not another device-memory allocation. */
typedef struct AimdoXpuCopyPressure {
    ssize_t fit_deficit;
    ssize_t post_reclaim_deficit;
    size_t reclaimed_pages;
} AimdoXpuCopyPressure;

static inline AimdoXpuCopyPressure aimdo_xpu_prepare_h2d(void) {
    AimdoXpuCopyPressure pressure = {0};

    pressure.fit_deficit = budget_deficit(0);
    pressure.post_reclaim_deficit = pressure.fit_deficit;
    if (pressure.fit_deficit > 0) {
        /* Preserve the WDDM progress recovery for a real live shortage. Its
         * non-blocking scan cannot select pinned or unfinished consumers. */
        pressure.reclaimed_pages = vbars_free_all_retired();
        pressure.post_reclaim_deficit = budget_deficit(0);
        if (pressure.post_reclaim_deficit > 0) {
            vbars_request_reclaim(pressure.post_reclaim_deficit);
        }
    }
    return pressure;
}

#endif
