"""Owner completion must precede cache release, including non-current queues."""

import sys
from types import SimpleNamespace

import pytest

from comfy_aimdo import control, model_vbar


@pytest.fixture
def completion(monkeypatch):
    events = []
    state = {"ready": True, "owned": {0, 1}}

    def native_wait(device, queue):
        events.append(("queue_wait", device, queue))
        return state["ready"]

    def current_stream(device):
        return SimpleNamespace(sycl_queue=100 + device)

    def original_sync(device=None):
        events.append(("device_wait", device))

    def original_empty():
        events.append("cache_release")

    xpu = SimpleNamespace(
        synchronize=original_sync, empty_cache=original_empty,
        memory=SimpleNamespace(empty_cache=original_empty),
        current_stream=current_stream, device_count=lambda: 2,
    )
    module = SimpleNamespace(xpu=xpu)
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setattr(control, "lib", SimpleNamespace(
        get_devctx=lambda device: device in state["owned"],
        xpu_synchronize_queues=native_wait,
    ))
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "devctxs", [1, 2])
    monkeypatch.setattr(control, "_xpu_device_index", lambda d: 0 if d is None else d)
    monkeypatch.setattr(control, "_windows_xpu_completion_hooks", [])
    control._install_windows_xpu_completion_hooks(module)
    yield module, events, state, original_sync, original_empty
    control._restore_windows_xpu_completion_hooks()


def test_device_wait_flushes_owned_sycl_queues_before_runtime_wait(completion):
    module, events, _, _, _ = completion
    module.xpu.synchronize(1)
    assert events == [("queue_wait", 1, 101), ("device_wait", 1)]


@pytest.mark.parametrize("namespace", ["xpu", "memory"])
def test_cache_release_waits_all_owned_devices(completion, namespace):
    module, events, _, _, _ = completion
    owner = module.xpu if namespace == "xpu" else module.xpu.memory
    owner.empty_cache()
    assert events == [("queue_wait", 0, 100), ("queue_wait", 1, 101),
                      "cache_release"]


def test_unowned_device_retains_torch_sync(completion):
    module, events, state, _, _ = completion
    state["owned"].remove(1)
    module.xpu.synchronize(1)
    assert events == [("device_wait", 1)]


@pytest.mark.parametrize("operation", ["synchronize", "empty_cache"])
def test_failed_queue_completion_never_releases_storage(completion, operation):
    module, events, state, _, _ = completion
    state["ready"] = False
    with pytest.raises(RuntimeError, match="queue completion failed"):
        getattr(module.xpu, operation)()
    assert events == [("queue_wait", 0, 100)]


def test_install_is_idempotent_and_deinit_restores_originals(completion):
    module, events, _, original_sync, original_empty = completion
    control._install_windows_xpu_completion_hooks(module)
    module.xpu.synchronize()
    assert events == [("queue_wait", 0, 100), ("device_wait", None)]
    control._restore_windows_xpu_completion_hooks()
    assert module.xpu.synchronize is original_sync
    assert module.xpu.empty_cache is original_empty
    assert module.xpu.memory.empty_cache is original_empty


def test_failed_completion_prevents_vbar_refault(completion, monkeypatch):
    _, events, state, _, _ = completion
    state["ready"] = False
    monkeypatch.setattr(model_vbar.sys, "platform", "win32")
    monkeypatch.setattr(control, "_xpu_allocator_mode", "native_hook")
    monkeypatch.setattr(control, "publish_torch_cached_bytes", lambda d: None)
    monkeypatch.setattr(model_vbar, "_release_native_cache", lambda d: False)

    def fault(*args):
        events.append("fault")
        return 1

    monkeypatch.setattr(model_vbar, "lib", SimpleNamespace(vbar_fault=fault))
    vbar = SimpleNamespace(base_addr=4096, device=0, _devctx=1, _ptr=2)
    with pytest.raises(RuntimeError, match="queue completion failed"):
        model_vbar.ModelVBAR.fault(vbar, 4096, 512)
    assert events == ["fault", ("queue_wait", 0, 100)]
