from types import SimpleNamespace
from unittest import mock

import pytest

from comfy_aimdo import control, malloc_graph


@pytest.mark.parametrize("record", [control.record, malloc_graph.record])
@pytest.mark.parametrize("initialized", [False, True])
def test_xpu_record_is_rejected_before_native_or_stream_access(record, initialized):
    stream = SimpleNamespace(device=SimpleNamespace(type="xpu", index=0))
    with mock.patch.object(control, "implementation", "xpu" if initialized else None), \
            mock.patch.object(control, "lib", None):
        with pytest.raises(NotImplementedError, match="memory compiler.*not yet supported on XPU"):
            record(stream)


def test_cuda_record_keeps_upstream_native_arguments():
    native = mock.Mock()
    native.malloc_graph_create.return_value = 123
    stream = SimpleNamespace(device=SimpleNamespace(type="cuda", index=2), cuda_stream=456)
    with mock.patch.object(control, "implementation", "cuda"), \
            mock.patch.object(control, "lib", native), \
            mock.patch.object(control, "get_devctx", return_value=789) as get_devctx:
        graph = control.record(stream, assert_graph_breaks=True)
        get_devctx.assert_called_once_with(2)
        args = native.malloc_graph_create.call_args.args
        assert args[0] == 789
        assert args[1].value == 456
        assert args[2] is True
        graph.__del__()
        native.malloc_graph_destroy.assert_called_once_with(123)


@pytest.mark.parametrize("value", [0, 4 * 1024**3, 1 << 60])
def test_simple_headroom_runtime_api(value):
    native = mock.Mock()
    native.get_simple_vram_headroom.return_value = value
    with mock.patch.object(control, "lib", native):
        control.set_simple_vram_headroom(value)
        native.set_simple_vram_headroom.assert_called_once_with(value)
        assert control.get_simple_vram_headroom() == value


@pytest.mark.parametrize("value", [-1, (1 << 60) + 1])
def test_simple_headroom_rejects_invalid_values_before_native_call(value):
    with mock.patch.object(control, "lib", None):
        with pytest.raises(ValueError):
            control.set_simple_vram_headroom(value)
