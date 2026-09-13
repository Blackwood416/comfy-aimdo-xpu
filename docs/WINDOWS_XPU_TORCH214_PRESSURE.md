# Windows XPU pressure recovery on the Torch 2.14 stack

The Windows XPU provider must keep large models streaming through Dynamic VRAM
without evicting a resident working set because the OS reserve was counted
twice. The pressure fix is based on PR #22 (`0b80da6`) and is committed as
`80ba77c`.

## Behavior

DXGI `CurrentUsage` remains the complete-process sample, including allocations
that are not tracked by AIMDO. When `Budget` is below physical capacity, its
existing OS reserve counts toward the requested physical-memory margin. Only
the remaining margin is subtracted from the DXGI budget. A smaller OS budget
still limits residency, and Level Zero free-memory pressure and significant
non-local usage can still require reclaim.

The existing two-second sample interval is retained. Model boundaries request
a fresh sample before applying the anticipated allocation size. Recorded
allocation/free deltas continue to adjust the sample between polls.

On a Windows XPU page fault, reclaim first tries other models, then allows the
active model's already-retired pages to satisfy the remaining shortage. Pages
with live pins, incomplete queue retirement, unknown ownership, or consumer
leases remain protected. The speculative model-boundary path still preserves
the active model. Reclaim never adds a device-wide wait to the fault path.

Normal logging no longer depends on a prior shot-counter reset. `log_shot`
retains its once-per-generation behavior.

## Regression coverage

`tests/windows_pressure_unit.c` executes the production `shmem-detect.c` and
`plat.h` code with scripted DXGI and memory samples. It requires no GPU. The
test covers:

- Both sides of the physical reserve and OS-budget boundaries.
- Physical pressure from untracked allocations.
- Failed/missing DXGI samples and non-local memory pressure.
- Cached allocation/free deltas, forced refresh, and additional headroom.
- Normal logging, shot suppression, and log-level filtering.

Run from a Windows checkout with Visual Studio Build Tools installed:

```powershell
python -m pytest -q tests/test_windows_pressure_native.py tests/test_windows_pressure_source.py
```

The native harness passes on the corrected code. As a negative control,
substituting only `shmem-detect.c` from `0b80da6` produces 11 assertion failures.
The broader pressure, VBAR range/budget, provider packaging, and opt-in suite
passes 50 tests. These host tests do not replace real-device retirement and
workflow validation.

The native entry-point ordering also passes all three Linux source-contract
checks. This is source validation on Windows, not a Linux runtime result.

## Reproducible provider build

Build the DLL with `scripts\build-windows-xpu.cmd`. Package it using a Python
build environment that actually contains `setuptools-scm`; `--no-build-isolation`
with that dependency missing can silently produce a `0.0.0` source wheel.

For a provider compatible with canonical AIMDO 0.5.3:

```powershell
$env:SETUPTOOLS_SCM_PRETEND_VERSION='0.5.3'
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist/provider-source
python packaging/xpu_runtime_provider/build_wheel.py `
  --source-wheel dist/provider-source/comfy_aimdo-0.5.3-cp39-abi3-win_amd64.whl `
  --output-dir dist/provider-release `
  --source-revision (git rev-parse HEAD) `
  --torch-version 2.14.0+xpu --xpu-target dg2
```

Install the provider wheel into the target environment after ComfyUI exits.
The source wheel is packaging input; canonical `comfy-aimdo` remains the
official distribution. The provider builder regenerates all manifest and wheel
hashes, including the exact source revision.

## Runtime compilation prerequisites

Installing the current kernel wheel restores the public `torch.compile`
boundaries, but Windows Inductor/Triton also needs a working host compiler
environment. Initialize the Visual Studio x64 and oneAPI environments, expose
Python's development `Include` and `libs` directories, and set
`LEVEL_ZERO_V1_SDK_PATH` to an SDK containing `include/level_zero/ze_api.h` and
`lib/ze_loader.lib`. The embedded Python distribution does not include those
Python development files by default.

Without these prerequisites, compile tests can fail on `cstddef`,
`level_zero/ze_api.h`, or `Python.h` before reaching an OmniXPU operation.
Do not add MSVC's `CL=/utf-8` option to a GNU-style `icpx` invocation.
The validation harness supplies these dependencies locally; it does not
modify ComfyUI, Torch, or Triton source.

## Environment and benchmark identity

The September 13 verification used A770 16 GB, Windows build 26200, driver
`32.0.101.8860`, oneAPI 2026.1, Visual Studio 18.3.3/MSVC 14.50.35717,
Python 3.13.12, Torch `2.14.0+xpu`, and ComfyUI `40c4fcdf` (0.35.0).

Historical handoff values require two corrections:

- The installed `omni_xpu_kernel 0.2.0b1+torch214.dg2` wheel predates the
  repository's compile-boundary merge. It lacks `_compile_ops.py` and
  `_compile_meta.py`, and nine API wrappers differ from the merged source.
  `0.2.0b1+torch214.dg2.1` was rebuilt from kernel commit `7f12671` and verified
  against both the source and actual installation.
- The H3 output associated with the roughly 550-second historical run is
  **864 x 480, 56 frames, 20 steps**, as verified from its MP4 stream and embedded
  API prompt. It is not a 768p/53-frame timing.

The Krea2 control uses 1024 x 1024, 8 Euler/simple steps, CFG 1, and the saved
API prompt from `Krea2_bench_test_00013_.png`. The sampling model connects
directly to the UNet; the disconnected LoRA node is not executed. The first
run is cold. Three following old-wheel runs took 31.84, 31.67, and 31.64 seconds
through the local API, with median step intervals 3.840, 3.771, and 3.764 seconds.
No recurring slowdown occurred in that control. These are pipeline/step wall
times, not isolated kernel device times.

With the rebuilt kernel and provider, the same four seeds produced images
that are pixel-identical to the control. The cold request took 49.40 seconds;
three warm requests took 30.68, 30.80, and 30.77 seconds, with median step
intervals 3.698, 3.712, and 3.700 seconds. Warm pipeline median changed from
31.67 to 30.77 seconds (2.8% lower), with 0.20% coefficient of variation in
the three new warm requests. This sequential comparison verifies output and
short-run stability; it is not an isolated-kernel speedup claim.

Full inputs, logs, wheel verification, and subsequent device/workflow results
are retained in `E:\RiderProjects\results\omnixpu-takeover-20260913` on the
validation host. Keep Dynamic VRAM and `OMNIXPU_PROVIDER_BOOTSTRAP=auto` enabled
when reproducing the workflow tests.

## Reconnecting Windows owner completion

On this Torch 2.14 runtime, `torch.xpu.synchronize()` can return while a native
SYCL kernel remains in an unsubmitted command-list batch. The isolated H3 SDP
probe observed a roughly 3 ms return, a false stream completion query, and
another roughly 261 ms wait at readback. Stream-level waits complete the work.
Immediate command lists also avoid that omission in isolation, but full Krea2
pressure runs later produced device loss; they are not the production fix.

The provider now connects existing Torch `synchronize` and `empty_cache`
boundaries to its owned queue registry. The native entry registers the caller's
current queue, copies owned queue handles under the registry lock, and waits
only after releasing that lock. It reuses the VBAR release implementation.
The original Torch call still runs afterward. Cache release covers every
initialized device, including calls through `torch.xpu.memory.empty_cache`.
Deinitialization restores the Python entry points. These waits never run in an
allocator callback and add no synchronization to successful VBAR faults.

Windows model prioritization and failed-fault recovery propagate completion
errors instead of continuing into storage release or refault. Linux keeps its
existing retry-only-after-actual-cache-release behavior; the Windows retry
after retirement completion is explicitly scoped to Windows XPU.

Validation on the stack above:

- 77 owner-completion, cache, budget, bootstrap, source-contract and provider
  packaging tests passed; native pressure/hook harnesses passed two tests with
  one platform skip.
- `tests/run_windows_xpu_completion.py` uses the already validated H3 SDP shape
  `[1,16473,56,128]` without memory pressure. All three completion entry points
  covered a registered non-current queue, each passed three repetitions, and
  all 465,920 checked values per repetition matched the independent constant
  reference. Warm synchronize completion was approximately 262–264 ms.
- The real-device VBAR mapped-hit, explicit eviction and refault test passed.

These checks validate the missing completion bridge. They do not establish
that it fixes the separately reported ClipProj access violation or meets the
workflow performance thresholds. Full workflow results remain in the external
acceptance directory.
