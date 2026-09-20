"""Reproduce the VRAMBuffer destroy -> recreate -> H2D copy access violation.

Cycle: create a VRAMBuffer (16 GiB reservation), grow it to N chunks (multiple
zeVirtualMemMap calls), copy a file slice into it on a stream, then drop it
(vrambuf_destroy: unmap, physical release, zeVirtualMemFree). Repeat. A crash on
the copy of a later cycle means the driver kept stale mapping state across the
reservation lifetime. Before the per-chunk unmap in vrambuf_destroy() this
crashed at cycle 1 with --chunks >= 2 on A770 (driver 32.0.101.8860); --chunks 1
and --hold never crashed.

--chunks   how many 16 MiB chunks to grow (1 = single mapping, >1 = multi)
--cycles   number of create/copy/destroy cycles
--hold     keep every buffer alive (control: no destroy)
--checkout import comfy_aimdo from this checkout instead of the installed provider
"""
import argparse
import faulthandler
import importlib.metadata
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

faulthandler.enable()
w = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if "--checkout" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
else:
    provider = importlib.metadata.distribution("comfy-aimdo-xpu-runtime")
    sys.path.insert(0, str(provider.locate_file("comfy_aimdo_xpu_runtime/_vendor")))
import torch
from comfy_aimdo import control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--checkout", action="store_true")
    ap.add_argument("--reserve-gib", type=int, default=16)
    ap.add_argument("--stream-mode", choices=("current", "side"), default="side")
    args = ap.parse_args()
    timer = threading.Timer(240, lambda: os._exit(124)); timer.daemon = True; timer.start()
    assert control.init(implementation="xpu")
    assert control.init_devices([0])
    from comfy_aimdo.vram_buffer import VRAMBuffer
    from comfy_aimdo.torch import aimdo_to_tensor
    from comfy_aimdo.host_buffer import read_file_to_device, cleanup_file_reader

    chunk = 16 << 20
    size = args.chunks * chunk
    held = []
    torch.xpu.init()
    side = torch.xpu.Stream(device=torch.device("xpu", 0)) if args.stream_mode == "side" else None
    with tempfile.TemporaryFile() as handle:
        handle.write(b"\x05" * size); handle.flush()
        for cycle in range(args.cycles):
            t0 = time.perf_counter()
            buf = VRAMBuffer(args.reserve_gib << 30, 0)
            base = buf.base_addr
            # grow chunk by chunk like cast_modules_with_vbar does (offset walks forward)
            views = []
            for i in range(args.chunks):
                views.append(aimdo_to_tensor(buf.get(chunk, i * chunk), "xpu:0"))
            stream = side if side is not None else torch.xpu.current_stream()
            with torch.xpu.stream(stream):
                for i, v in enumerate(views):
                    read_file_to_device(handle, i * chunk, chunk, int(stream.sycl_queue), v.data_ptr(), 0, mark_cold=False)
            stream.synchronize()
            ok = all(bool((v[:4096].cpu() == 5).all()) for v in views)
            vmm = control.get_xpu_vmm_stats()
            print(f"cycle={cycle} base=0x{base:x} allocated={buf.size() >> 20}MiB verify={ok} "
                  f"map={vmm['map_calls']} unmap={vmm['unmap_calls']} phys_rel={vmm['physical_release_calls']} "
                  f"dt={time.perf_counter() - t0:.2f}s", file=w); w.flush()
            del views
            if args.hold:
                held.append(buf)
            else:
                torch.xpu.synchronize()
                del buf
    cleanup_file_reader()
    print("DONE", file=w); w.flush()
    control.deinit()
    timer.cancel()


if __name__ == "__main__":
    main()
