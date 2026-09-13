"""Recover native cache before the existing reader's VBAR-pressure fallback."""

from types import SimpleNamespace

import pytest

from comfy_aimdo import control, host_buffer, model_vbar


@pytest.fixture
def copy_owner(monkeypatch):
    events = []
    state = {"deficit": 1}

    def deficit(device, allocation_bytes=0):
        events.append(("budget", device, allocation_bytes))
        return state["deficit"]

    def recover(device):
        events.append(("recover", device))
        return True

    monkeypatch.setattr(host_buffer, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(control, "implementation", "xpu")
    monkeypatch.setattr(control, "get_xpu_memory_deficit", deficit)
    monkeypatch.setattr(model_vbar, "_release_native_cache", recover)
    monkeypatch.setattr(host_buffer, "_device_stream_ptr", lambda s, d: 17)
    monkeypatch.setattr(host_buffer, "_file_handle", lambda f: 23)

    def read(*args):
        events.append("copy")
        return True

    monkeypatch.setattr(host_buffer, "lib", SimpleNamespace(
        hostbuf_file_reader_read=read, hostbuf_read_file_slice=read,
    ))
    return events, state


@pytest.mark.parametrize("reader", ["ring", "hostbuf"])
def test_copy_pressure_recovers_cache_before_native_fallback(copy_owner, reader):
    events, _ = copy_owner
    if reader == "ring":
        host_buffer.read_file_to_device(None, 0, 64 << 20, 17, 4096, 0)
    else:
        obj = SimpleNamespace(_ptr=1, size=0)
        host_buffer.HostBuffer.read_file_slice(obj, None, 0, 64 << 20,
                                               device_ptr=4096, device=0)
    assert events == [("budget", 0, 0), ("recover", 0), "copy"]


@pytest.mark.parametrize("deficit", [None, -1, 0])
def test_no_live_shortage_does_not_flush(copy_owner, deficit):
    events, state = copy_owner
    state["deficit"] = deficit
    host_buffer.read_file_to_device(None, 0, 64 << 20, 17, 4096, 0)
    assert events == [("budget", 0, 0), "copy"]


@pytest.mark.parametrize("device,pointer,size", [(-1, 4096, 1), (0, 0, 1), (0, 4096, 0)])
def test_non_device_read_does_not_query_budget(copy_owner, device, pointer, size):
    events, _ = copy_owner
    host_buffer._prepare_xpu_copy(device, pointer, size)
    assert events == []


def test_completion_failure_prevents_copy(copy_owner, monkeypatch):
    events, _ = copy_owner

    def failed(device):
        raise RuntimeError("queue completion failed")

    monkeypatch.setattr(model_vbar, "_release_native_cache", failed)
    with pytest.raises(RuntimeError, match="queue completion failed"):
        host_buffer.read_file_to_device(None, 0, 1024, 17, 4096, 0)
    assert events == [("budget", 0, 0)]
