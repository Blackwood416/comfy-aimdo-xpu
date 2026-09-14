"""Bounded real-device regression for Windows VBAR sibling copy residency.

Two pages share one VA reservation; a third page belongs to a separate VBAR.
After page A's own consumer completes, a copy into page B waits behind a warmed
GEMM. Reclaim must retain A, release the unrelated page, then release A after
the copy completes. The budget request allocates no memory; live tensors and
VBAR mappings stay below 1 GiB. Run in a fresh process on one idle XPU.
"""
import argparse
import ctypes
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading

faulthandler.enable()
ROOT = Path(__file__).resolve().parents[1]
if "--installed-provider" in sys.argv:
    provider = importlib.metadata.distribution("comfy-aimdo-xpu-runtime")
    sys.path.insert(0, str(provider.locate_file("comfy_aimdo_xpu_runtime/_vendor")))
else:
    sys.path.insert(0, str(ROOT))

import torch
from comfy_aimdo import control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-provider", action="store_true")
    parser.add_argument("--threaded-copy", action="store_true")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    watchdog = threading.Timer(60, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    assert control.init(implementation="xpu")
    assert control.init_devices([0])
    from comfy_aimdo.model_vbar import ModelVBAR
    from comfy_aimdo.torch import aimdo_to_tensor

    page = 32 << 20
    vbar = ModelVBAR(2 * page, 0)
    a, b = vbar.alloc(page), vbar.alloc(page)
    unrelated = ModelVBAR(page, 0)
    c = unrelated.alloc(page)
    source = torch.full((page,), 7, dtype=torch.uint8, device="xpu")
    matrix = torch.ones((12288, 12288), dtype=torch.bfloat16, device="xpu")
    stream = torch.xpu.current_stream()
    product = torch.mm(matrix, matrix)
    stream.synchronize()
    del product
    rows = []

    def pressure(preserved=None):
        deficit = control.get_xpu_memory_deficit(0)
        request = max(0, -deficit) + 2 * page
        control.lib.vbars_prepare_allocation(vbar._devctx, preserved, request)

    with torch.inference_mode():
        for cycle in range(args.cycles):
            vbar.set_watermark(2 * page)
            unrelated.set_watermark(page)
            for owner, allocation in ((vbar, a), (vbar, b), (unrelated, c)):
                assert owner.fault(allocation[1], allocation[2]) is not None
            av, bv, cv = (aimdo_to_tensor(x, "xpu:0") for x in (a, b, c))
            av.copy_(source)
            cv.copy_(source)
            stream.synchronize()
            vbar.unpin(a[1], a[2])
            unrelated.unpin(c[1], c[2])
            # Complete the two pages' own retirement fences before the next copy.
            # C is kept pinned while this closes the shared queue's token batch.
            assert unrelated.fault(c[1], c[2]) is not None
            pressure(vbar._ptr)
            stream.synchronize()
            unrelated.set_watermark_limit(page)
            unrelated.unpin(c[1], c[2])
            pressure(vbar._ptr)
            stream.synchronize()
            unrelated.set_watermark_limit(0)

            product = torch.mm(matrix, matrix)
            worker_errors = []

            def copy_sibling():
                try:
                    with torch.inference_mode(), torch.xpu.stream(stream):
                        bv.copy_(source, non_blocking=True)
                except BaseException as error:
                    worker_errors.append(error)

            if args.threaded_copy:
                worker = threading.Thread(target=copy_sibling)
                worker.start()
                worker.join(timeout=5)
                assert not worker.is_alive()
            else:
                copy_sibling()
            assert not worker_errors, worker_errors
            pressure()
            during = vbar.get_residency()
            assert during == [1, 3], during
            assert unrelated.loaded_size() == 0
            stream.synchronize()
            actual = bv.cpu()
            errors = int((actual != 7).sum())
            assert errors == 0
            torch.testing.assert_close(product[::256, ::256].cpu(),
                torch.full((48, 48), 12288, dtype=torch.bfloat16))
            pressure()
            assert vbar.get_residency() == [0, 3]
            vbar.unpin(b[1], b[2])
            assert vbar.free_memory(2 * page) == page
            rows.append({"cycle": cycle, "errors": errors, "checked": page,
                         "pending_copy_residency": during})
            print(json.dumps(rows[-1]), flush=True)
            del av, bv, cv, product, actual

    del a, b, c, source, matrix
    vbar.__del__()
    unrelated.__del__()
    values = (ctypes.c_uint64 * 5)()
    control.lib.xpu_copy_residency_get_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    control.lib.xpu_copy_residency_get_stats.restype = ctypes.c_bool
    assert control.lib.xpu_copy_residency_get_stats(values, 5)
    stats = dict(zip(("tracked", "completed", "pending", "failed", "polls"), values))
    assert stats["pending"] == 0 and stats["tracked"] == stats["completed"]
    report = {"complete": True, "rows": rows, "copy_stats": stats,
              "threaded_copy": args.threaded_copy, "torch": torch.__version__,
              "device": torch.xpu.get_device_name(0), "provider": control.__file__,
              "dll_sha256": hashlib.sha256(Path(control.__file__).with_name("aimdo_xpu.dll").read_bytes()).hexdigest()}
    control.deinit()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    watchdog.cancel()


if __name__ == "__main__":
    main()
