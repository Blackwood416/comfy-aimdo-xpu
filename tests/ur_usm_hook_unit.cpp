#define AIMDO_XPU_TESTING
#include "../src-xpu/ur-usm-hook.cpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>

namespace {

std::atomic<uint64_t> g_real_alloc_calls{0};
std::atomic<uint64_t> g_real_free_calls{0};
std::atomic<int64_t> g_accounted_bytes{0};
std::atomic<int64_t> g_deficit{0};
std::atomic<int64_t> g_evicted_bytes{0};
std::atomic<uintptr_t> g_next_pointer{0x10000000};
std::atomic<int> g_alloc_failures{0};
std::atomic<bool> g_account_fail{false};

ur_result_t fake_device_alloc(
    ur_context_handle_t,
    ur_device_handle_t,
    const ur_usm_desc_t *,
    ur_usm_pool_handle_t,
    size_t,
    void **pointer) {
    g_real_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (g_alloc_failures.load() > 0) {
        g_alloc_failures.fetch_sub(1);
        if (pointer) *pointer = nullptr;
        return UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (pointer) {
        *pointer = reinterpret_cast<void *>(
            g_next_pointer.fetch_add(0x1000, std::memory_order_relaxed));
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t fake_free(ur_context_handle_t, void *) {
    g_real_free_calls.fetch_add(1, std::memory_order_relaxed);
    return UR_RESULT_SUCCESS;
}

ur_result_t fake_get_native_handle(
    ur_device_handle_t device, ur_native_handle_t *native) {
    if (!native) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    *native = reinterpret_cast<ur_native_handle_t>(device);
    return UR_RESULT_SUCCESS;
}

void reset_state() {
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    g_enabled.store(false, std::memory_order_relaxed);
    g_generation.store(0, std::memory_order_relaxed);
    g_allocations.clear();
    g_torch_cached_bytes.clear();
    g_cache_lever_skipped_calls.store(0);
    g_alloc_failures.store(0);
    g_account_fail.store(false);
    clear_retry();
    for (auto &stat : g_stats) {
        stat.store(0, std::memory_order_relaxed);
    }
    g_test_device_alloc = fake_device_alloc;
    g_test_free = fake_free;
    g_test_device_get_native_handle = fake_get_native_handle;
    g_test_request_kind.store(
        TestRequestKind::kAutomatic, std::memory_order_relaxed);
    g_test_after_fast_enabled_check = nullptr;
    g_real_alloc_calls.store(0, std::memory_order_relaxed);
    g_real_free_calls.store(0, std::memory_order_relaxed);
    g_accounted_bytes.store(0, std::memory_order_relaxed);
    g_deficit.store(0, std::memory_order_relaxed);
    g_evicted_bytes.store(0, std::memory_order_relaxed);
    g_next_pointer.store(0x10000000, std::memory_order_relaxed);
}

void enable_for_test(uint64_t generation = 1) {
    g_generation.store(generation, std::memory_order_relaxed);
    g_enabled.store(true, std::memory_order_release);
}

void test_direct_request_does_not_require_retry() {
    reset_state();
    enable_for_test();
    g_test_request_kind.store(
        TestRequestKind::kDirect, std::memory_order_relaxed);
    g_deficit.store(4096, std::memory_order_relaxed);

    void *pointer = nullptr;
    const auto context = reinterpret_cast<ur_context_handle_t>(0x11);
    const auto device = reinterpret_cast<ur_device_handle_t>(0x22);
    assert(urUSMDeviceAlloc(
               context, device, nullptr, nullptr, 8192, &pointer) ==
           UR_RESULT_SUCCESS);
    assert(pointer != nullptr);
    assert(g_real_alloc_calls.load(std::memory_order_relaxed) == 1);
    assert(g_evicted_bytes.load(std::memory_order_relaxed) == 4096);
    assert(g_stats[kSyntheticOomCalls].load(std::memory_order_relaxed) == 0);
    assert(g_stats[kDirectPressureCalls].load(std::memory_order_relaxed) == 1);
    assert(g_stats[kDirectPressureBytes].load(std::memory_order_relaxed) == 4096);
    assert(urUSMFree(context, pointer) == UR_RESULT_SUCCESS);
    assert(g_accounted_bytes.load(std::memory_order_relaxed) == 0);
}

void test_torch_request_preserves_two_stage_retry() {
    reset_state();
    enable_for_test(7);
    xpu_ur_hook_set_torch_cached_bytes(0, 4096);
    g_test_request_kind.store(
        TestRequestKind::kTorchNative, std::memory_order_relaxed);
    g_deficit.store(4096, std::memory_order_relaxed);

    void *pointer = nullptr;
    const auto context = reinterpret_cast<ur_context_handle_t>(0x33);
    const auto device = reinterpret_cast<ur_device_handle_t>(0x44);
    assert(urUSMDeviceAlloc(
               context, device, nullptr, nullptr, 8192, &pointer) ==
           UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(pointer == nullptr);
    assert(g_real_alloc_calls.load(std::memory_order_relaxed) == 0);
    assert(g_stats[kSyntheticOomCalls].load(std::memory_order_relaxed) == 1);

    assert(urUSMDeviceAlloc(
               context, device, nullptr, nullptr, 8192, &pointer) ==
           UR_RESULT_SUCCESS);
    assert(pointer != nullptr);
    assert(g_real_alloc_calls.load(std::memory_order_relaxed) == 1);
    assert(g_evicted_bytes.load(std::memory_order_relaxed) == 4096);
    assert(urUSMFree(context, pointer) == UR_RESULT_SUCCESS);
}

std::atomic<bool> g_after_fast_enabled{false};

void mark_after_fast_enabled() {
    g_after_fast_enabled.store(true, std::memory_order_release);
}

void test_allocation_rechecks_state_after_disable_transition() {
    reset_state();
    enable_for_test(3);
    g_test_request_kind.store(
        TestRequestKind::kDirect, std::memory_order_relaxed);
    g_test_after_fast_enabled_check = mark_after_fast_enabled;
    g_after_fast_enabled.store(false, std::memory_order_relaxed);

    void *pointer = nullptr;
    std::unique_lock<std::mutex> state_guard(g_hook_mutex);
    std::thread worker([&] {
        assert(urUSMDeviceAlloc(
                   reinterpret_cast<ur_context_handle_t>(0x55),
                   reinterpret_cast<ur_device_handle_t>(0x66),
                   nullptr, nullptr, 8192, &pointer) == UR_RESULT_SUCCESS);
    });
    while (!g_after_fast_enabled.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    g_enabled.store(false, std::memory_order_release);
    g_generation.fetch_add(1, std::memory_order_relaxed);
    state_guard.unlock();
    worker.join();
    g_test_after_fast_enabled_check = nullptr;

    assert(pointer != nullptr);
    assert(g_allocations.empty());
    assert(g_accounted_bytes.load(std::memory_order_relaxed) == 0);
    assert(g_stats[kPassThroughAllocCalls].load(std::memory_order_relaxed) == 1);
}

void test_retry_generation_invalidates_worker_state() {
    reset_state();
    enable_for_test(9);
    xpu_ur_hook_set_torch_cached_bytes(0, 4096);
    g_test_request_kind.store(
        TestRequestKind::kTorchNative, std::memory_order_relaxed);
    g_deficit.store(4096, std::memory_order_relaxed);
    std::atomic<bool> armed{false};
    std::atomic<bool> check{false};
    std::atomic<ur_result_t> result{UR_RESULT_SUCCESS};
    std::atomic<uint64_t> retry_generation{0};
    const auto context = reinterpret_cast<ur_context_handle_t>(0x77);
    const auto device = reinterpret_cast<ur_device_handle_t>(0x88);

    std::thread worker([&] {
        arm_retry(
            context, device, nullptr, 8192,
            RetryReason::kBudgetDeficit,
            g_generation.load(std::memory_order_relaxed));
        armed.store(true, std::memory_order_release);
        while (!check.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        void *pointer = nullptr;
        result.store(
            urUSMDeviceAlloc(
                context, device, nullptr, nullptr, 8192, &pointer),
            std::memory_order_release);
        retry_generation.store(g_retry.generation, std::memory_order_release);
        assert(pointer == nullptr);
    });
    while (!armed.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    g_generation.fetch_add(1, std::memory_order_relaxed);
    check.store(true, std::memory_order_release);
    worker.join();
    assert(
        result.load(std::memory_order_acquire) ==
        UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(g_real_alloc_calls.load(std::memory_order_relaxed) == 0);
    assert(g_evicted_bytes.load(std::memory_order_relaxed) == 0);
    assert(g_stats[kSyntheticOomCalls].load(std::memory_order_relaxed) == 1);
    assert(
        retry_generation.load(std::memory_order_acquire) ==
        g_generation.load(std::memory_order_relaxed));
}

void test_duplicate_pointer_is_not_double_accounted_or_freed() {
    reset_state();
    void *pointer = reinterpret_cast<void *>(0x12345000);
    assert(account_success(pointer, 8192, 0) == AccountResult::kSuccess);
    assert(
        account_success(pointer, 16384, 0) ==
        AccountResult::kDuplicatePointer);
    assert(g_stats[kTrackedAllocCalls].load(std::memory_order_relaxed) == 1);
    assert(g_stats[kDuplicatePointerCalls].load(std::memory_order_relaxed) == 1);
    assert(g_real_free_calls.load(std::memory_order_relaxed) == 0);
    assert(g_accounted_bytes.load(std::memory_order_relaxed) == 8192);
}

void test_empty_or_consumed_hint_skips_synthetic_oom() {
    reset_state();
    enable_for_test();
    g_test_request_kind.store(TestRequestKind::kTorchNative);
    g_deficit.store(4096);
    const auto context = reinterpret_cast<ur_context_handle_t>(0x91);
    const auto device = reinterpret_cast<ur_device_handle_t>(0x92);
    void *pointer = nullptr;
    assert(urUSMDeviceAlloc(context, device, nullptr, nullptr, 8192, &pointer)
           == UR_RESULT_SUCCESS);
    assert(g_cache_lever_skipped_calls.load() == 1);
    assert(g_stats[kSyntheticOomCalls].load() == 0);
    assert(urUSMFree(context, pointer) == UR_RESULT_SUCCESS);
    xpu_ur_hook_set_torch_cached_bytes(0, 4096);
    assert(urUSMDeviceAlloc(context, device, nullptr, nullptr, 8192, &pointer)
           == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
    assert(urUSMDeviceAlloc(context, device, nullptr, nullptr, 8192, &pointer)
           == UR_RESULT_SUCCESS);
    assert(urUSMFree(context, pointer) == UR_RESULT_SUCCESS);
    // No new owner publication: a positive estimate cannot request another flush.
    assert(urUSMDeviceAlloc(context, device, nullptr, nullptr, 8192, &pointer)
           == UR_RESULT_SUCCESS);
    assert(g_stats[kSyntheticOomCalls].load() == 1);
    assert(g_cache_lever_skipped_calls.load() == 2);
    assert(urUSMFree(context, pointer) == UR_RESULT_SUCCESS);
    assert(g_accounted_bytes.load() == 0);
}

void test_runtime_oom_reclaims_only_residual_after_same_device_free() {
    for (int variant = 0; variant != 4; ++variant) {
        reset_state();
        enable_for_test();
        g_test_request_kind.store(TestRequestKind::kTorchNative);
        const auto context = reinterpret_cast<ur_context_handle_t>(0xa1);
        const auto device = reinterpret_cast<ur_device_handle_t>(0xa2);
        void *cached = reinterpret_cast<void *>(0xb0);
        assert(account_success(cached, 4096, variant == 2 ? 1 : 0)
               == AccountResult::kSuccess);
        void *pointer = nullptr;
        g_alloc_failures.store(1);
        assert(urUSMDeviceAlloc(context, device, nullptr, nullptr, 8192, &pointer)
               == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        assert(g_retry.reason == RetryReason::kRuntimeOom);
        if (variant == 3) g_generation.fetch_add(1);
        const auto free_context = variant == 1
            ? reinterpret_cast<ur_context_handle_t>(0xc1) : context;
        assert(urUSMFree(free_context, cached) == UR_RESULT_SUCCESS);
        assert(g_retry.returned_bytes == (variant == 0 ? 4096 : 0));
        assert(urUSMDeviceAlloc(context, device, nullptr, nullptr, 8192, &pointer)
               == UR_RESULT_SUCCESS);
        // A new generation discards the obsolete retry entirely.
        assert(g_evicted_bytes.load() == (variant == 0 ? 4096 : variant == 3 ? 0 : 8192));
        assert(urUSMFree(context, pointer) == UR_RESULT_SUCCESS);
        assert(g_accounted_bytes.load() == 0);
    }
}

void test_failed_account_rolls_back_owned_allocation() {
    reset_state();
    enable_for_test();
    g_test_request_kind.store(TestRequestKind::kTorchNative);
    g_account_fail.store(true);
    void *pointer = nullptr;
    assert(urUSMDeviceAlloc(reinterpret_cast<ur_context_handle_t>(0xd1),
        reinterpret_cast<ur_device_handle_t>(0xd2), nullptr, nullptr, 8192, &pointer)
        == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY);
    assert(pointer == nullptr);
    assert(g_allocations.empty());
    assert(g_real_free_calls.load() == 1);
    assert(g_accounted_bytes.load() == 0);
}

}  // namespace

extern "C" bool aimdo_xpu_allocation_deficit(
    int, size_t, int64_t *deficit) {
    if (!deficit) {
        return false;
    }
    *deficit = g_deficit.load(std::memory_order_relaxed);
    return true;
}

extern "C" bool aimdo_xpu_evict_for_allocation(int, int64_t deficit) {
    g_evicted_bytes.fetch_add(deficit, std::memory_order_relaxed);
    return true;
}

extern "C" bool aimdo_xpu_account_allocation(int, int64_t delta) {
    if (g_account_fail.load()) return false;
    g_accounted_bytes.fetch_add(delta, std::memory_order_relaxed);
    return true;
}

extern "C" int xpu_device_from_native_handle(uintptr_t) {
    return 0;
}

int main() {
    test_direct_request_does_not_require_retry();
    test_torch_request_preserves_two_stage_retry();
    test_allocation_rechecks_state_after_disable_transition();
    test_retry_generation_invalidates_worker_state();
    test_duplicate_pointer_is_not_double_accounted_or_freed();
    test_empty_or_consumed_hint_skips_synthetic_oom();
    test_runtime_oom_reclaims_only_residual_after_same_device_free();
    test_failed_account_rolls_back_owned_allocation();
    return 0;
}
