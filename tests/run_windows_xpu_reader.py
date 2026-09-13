"""Bounded reader-ring reuse, stream dependency and teardown checks on XPU.

No model or pressure allocation. Each round compares every byte with the
independent CPU file contents before dropping the device tensors.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from comfy_aimdo import control


def exercise():
    from comfy_aimdo.host_buffer import HostBuffer, cleanup_file_reader, read_file_to_device
    from comfy_aimdo.model_vbar import ModelVBAR
    from comfy_aimdo.torch import aimdo_to_tensor, hostbuf_to_tensor

    device = torch.device("xpu", 0)
    payload = np.arange(40 << 20, dtype=np.uint8) * np.uint8(17) + np.uint8(3)
    rows = []
    with tempfile.TemporaryFile() as handle:
        handle.write(payload.tobytes())
        handle.flush()
        for cycle in range(3):
            streams = [torch.xpu.Stream(device=device) for _ in range(2)]
            outputs = []
            before = control.get_xpu_vmm_stats()
            started = time.perf_counter()
            for index in range(15):
                size = (16 << 20) + (index * 257) % 4096
                offset = 97 * index + 13 * cycle
                destination = torch.empty(size, device=device, dtype=torch.uint8)
                stream = streams[index % 2]
                with torch.xpu.stream(stream):
                    read_file_to_device(handle, offset, size, int(stream.sycl_queue),
                                        destination.data_ptr(), 0, mark_cold=False)
                # The consumer must see native reader submissions through the
                # existing Torch stream dependency, without a host-side wait.
                torch.xpu.current_stream(device).wait_stream(stream)
                increment = (index + cycle) % 19
                observed = destination + increment
                destination.record_stream(torch.xpu.current_stream(device))
                outputs.append((observed, offset, size, increment))
            del destination, observed, stream, streams
            gc.collect()
            # Includes the partially filled active slot after Python no longer
            # owns its streams. No explicit stream/device wait before cleanup.
            cleanup_file_reader()
            checked = 0
            for observed, offset, size, increment in outputs:
                expected = payload[offset:offset + size] + np.uint8(increment)
                actual = observed.cpu().numpy()
                errors = int(np.count_nonzero(actual != expected))
                assert errors == 0, (cycle, offset, errors)
                checked += size
            del observed, outputs, actual, expected
            after = control.get_xpu_vmm_stats()
            async_calls = after["asynchronous_host_to_device_calls"] - before["asynchronous_host_to_device_calls"]
            if os.environ.get("AIMDO_XPU_ASYNC_FILE_READER") == "1":
                assert async_calls >= 15, after
            assert after["event_sync_calls"] > before["event_sync_calls"]
            assert after["host_buffer_wait_failures"] == 0
            row = {"cycle": cycle, "checked": checked, "errors": 0,
                   "async_calls": async_calls,
                   "wall_seconds": time.perf_counter() - started}
            rows.append(row)
            print(json.dumps(row), flush=True)

        # A regular HostBuffer may be changed by its owner immediately after
        # read_file_slice returns. It must not inherit the ring's async policy.
        size = (2 << 20) + 11
        host = HostBuffer(0, 0, 4 << 20, mark_cold=False)
        host.extend(size, register=False)
        destination = torch.empty(size, device=device, dtype=torch.uint8)
        before = control.get_xpu_vmm_stats()
        host.read_file_slice(handle, 0, size, device=0, device_ptr=destination.data_ptr())
        hostbuf_to_tensor(host).fill_(255)
        assert np.array_equal(destination.cpu().numpy(), payload[:size])
        after = control.get_xpu_vmm_stats()
        assert after["asynchronous_host_to_device_calls"] == before["asynchronous_host_to_device_calls"]
        assert after["synchronous_host_to_device_completions"] > before["synchronous_host_to_device_completions"]
        del destination, host

        # VBAR pins and the normal unpin/free boundary must protect queued
        # reader copies and their consumer as well as ordinary USM outputs.
        vbar = ModelVBAR(32 << 20, 0)
        allocation = vbar.alloc(32 << 20)
        assert vbar.fault(allocation[1], allocation[2]) is not None
        destination = aimdo_to_tensor(allocation, device)
        stream = torch.xpu.Stream(device=device)
        with torch.xpu.stream(stream):
            read_file_to_device(handle, 17, size, int(stream.sycl_queue),
                                destination.data_ptr(), 0, mark_cold=False)
            observed = destination[:size].clone()
            vbar.unpin(allocation[1], allocation[2])
        assert vbar.free_memory(32 << 20) == 32 << 20
        assert np.array_equal(observed.cpu().numpy(), payload[17:17 + size])
        cleanup_file_reader()
        del destination, observed, stream, allocation
        vbar.__del__()
        del vbar
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    watchdog = threading.Timer(90, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    assert os.environ.get("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "0") != "1"
    assert control.init(implementation="xpu")
    assert control.init_devices([0])
    try:
        rows = exercise()
        args.output.write_text(json.dumps({
            "device": torch.xpu.get_device_name(), "torch": torch.__version__,
            "rounds": rows, "vmm_stats": control.get_xpu_vmm_stats(),
        }, indent=2), encoding="utf-8")
    finally:
        gc.collect()
        control.deinit()
        watchdog.cancel()


if __name__ == "__main__":
    main()
