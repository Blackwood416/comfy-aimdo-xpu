"""Windows cache hints follow the model owner, without flushing or waiting."""

from types import SimpleNamespace

import pytest
import torch

from comfy_aimdo import control, model_vbar


@pytest.fixture
def owner(monkeypatch):
    events = []
    state = {"reserved": 96, "allocated": 32}

    def stats(device):
        assert device == 0
        events.append("stats")
        return {"reserved_bytes.all.current": state["reserved"],
                "allocated_bytes.all.current": state["allocated"]}

    def hint(device, value):
        events.append(("hint", device, value))

    def fault(ctx, ptr, offset, size, signature):
        events.append("fault")
        signature[0] = 17
        return 0

    def unexpected(*args, **kwargs):
        pytest.fail("a cache hint must not flush storage or synchronize the GPU")

    native = SimpleNamespace(xpu_ur_hook_set_torch_cached_bytes=hint,
                             vbar_fault=fault, vbar_get_watermark=lambda *a: 0)
    monkeypatch.setattr(model_vbar, "lib", native)
    monkeypatch.setattr(model_vbar.sys, "platform", "win32")
    monkeypatch.setattr(control, "lib", native)
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "_xpu_allocator_mode", "native_hook")
    monkeypatch.setattr(torch.xpu, "memory_stats", stats)
    monkeypatch.setattr(torch.xpu, "empty_cache", unexpected)
    monkeypatch.setattr(torch.xpu, "synchronize", unexpected)
    vbar = SimpleNamespace(base_addr=4096, device=0, _devctx=1, _ptr=2)
    return vbar, state, events, native


def test_resident_fault_refreshes_current_cache_before_native_access(owner):
    vbar, state, events, _ = owner
    assert model_vbar.ModelVBAR.fault(vbar, 4096, 512)[0] == 17
    state["allocated"] = 80
    assert model_vbar.ModelVBAR.fault(vbar, 4096, 512)[0] == 17
    assert events == ["stats", ("hint", 0, 64), "fault",
                      "stats", ("hint", 0, 16), "fault"]


@pytest.mark.parametrize("platform,implementation,mode", [
    ("linux", "xpu", "native_hook"),
    ("win32", "cuda", "native_hook"),
    ("win32", "xpu", "global"),
])
def test_other_allocator_owners_keep_existing_fault_path(
        owner, monkeypatch, platform, implementation, mode):
    vbar, _, events, _ = owner
    monkeypatch.setattr(model_vbar.sys, "platform", platform)
    monkeypatch.setattr(control, "implementation", implementation)
    monkeypatch.setattr(control, "_xpu_allocator_mode", mode)
    assert model_vbar.ModelVBAR.fault(vbar, 4096, 512)[0] == 17
    assert events == ["fault"]


@pytest.mark.parametrize("failure", ["missing_symbol", "stats", "publisher"])
def test_unavailable_hint_does_not_replace_native_fault(owner, monkeypatch, failure):
    vbar, _, events, native = owner

    def unavailable(*args):
        raise RuntimeError("statistics unavailable")

    if failure == "missing_symbol":
        del native.xpu_ur_hook_set_torch_cached_bytes
    elif failure == "stats":
        monkeypatch.setattr(torch.xpu, "memory_stats", unavailable)
    else:
        monkeypatch.setattr(control, "publish_torch_cached_bytes", unavailable)
    assert model_vbar.ModelVBAR.fault(vbar, 4096, 512)[0] == 17
    assert events == ["fault"]


def test_negative_cached_bytes_are_clamped(owner):
    vbar, state, events, _ = owner
    state["reserved"] = 16
    assert model_vbar.ModelVBAR.fault(vbar, 4096, 512)[0] == 17
    assert events == ["stats", ("hint", 0, 0), "fault"]
