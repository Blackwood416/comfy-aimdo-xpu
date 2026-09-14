"""Bounded real-device check of native batches at existing owner boundaries.

Uses the previously validated H3 SDP shape, with uniform inputs so the
reference is a constant rather than another GPU implementation. It allocates
under 2 GiB, does not create memory pressure, and keeps batched submission.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omni_xpu_kernel import sdp
from comfy_aimdo import control


def exercise():
    device = torch.xpu.current_device()
    current = torch.xpu.current_stream(device)
    other = torch.xpu.Stream(device=device)
    # This validates that a registered queue remains covered after its Python
    # context has ended and another queue becomes current.
    with torch.xpu.stream(other):
        assert control.synchronize_xpu_queues(device)
    shape = (1, 16473, 56, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16, device="xpu")
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    torch.xpu.synchronize()
    rows = []
    with torch.inference_mode():
        for mode in ("synchronize", "empty_cache", "memory.empty_cache"):
            for iteration in range(3):
                value = (iteration + 1) * 0.25
                v.fill_(value)
                torch.xpu.synchronize()
                started = time.perf_counter()
                with torch.xpu.stream(other):
                    output = sdp.sdp(q, k, v)
                submitted = time.perf_counter()
                if mode == "synchronize":
                    torch.xpu.synchronize(device)
                elif mode == "empty_cache":
                    torch.xpu.empty_cache()
                else:
                    torch.xpu.memory.empty_cache()
                completed = time.perf_counter()
                ready = other.query()
                assert ready, f"{mode} returned before registered queue completion"
                # Uniform QK scores make the exact CPU reference constant.
                sample = output[:, ::257].cpu().float()
                errors = int((sample.sub(value).abs() > 0.004).sum())
                assert errors == 0, (mode, iteration, errors)
                row = {"mode": mode, "iteration": iteration,
                       "submit_ms": 1000 * (submitted - started),
                       "complete_ms": 1000 * (completed - started),
                       "queue_complete": ready, "errors": errors,
                       "reference_value": value, "checked": sample.numel()}
                rows.append(row)
                print(json.dumps(row), flush=True)
                del output, sample
    assert current.query()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    watchdog = threading.Timer(90, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    assert os.environ.get("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "0") != "1"
    assert control.init(implementation="xpu")
    assert control.init_devices([torch.xpu.current_device()])
    try:
        rows = exercise()
        args.output.write_text(json.dumps({
            "torch": torch.__version__, "device": torch.xpu.get_device_name(),
            "measurements": rows,
            "vmm_stats": control.get_xpu_vmm_stats(),
        }, indent=2), encoding="utf-8")
    finally:
        gc.collect()
        control.deinit()
        watchdog.cancel()


if __name__ == "__main__":
    main()
