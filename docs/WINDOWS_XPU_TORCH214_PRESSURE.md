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

## Native cache before file-reader pressure recovery

The native allocation retry cannot return Torch cache when a file-reader copy
is the first operation to observe pressure: the destination was already
allocated, so no native allocation hook runs. The reader could therefore flush
the retired VBAR working set while Torch retained 1–2 GiB of reusable cache.

Both Python file-reader entry points now query the existing AIMDO physical
budget with zero additional allocation bytes. Only a positive live deficit
enters the existing native-cache recovery, retaining its two-second attempt
limit and 32 MiB minimum cache check. The completion bridge above joins queues
before releasing storage. Cache/retry hints are republished after actual
release. The native reader still rechecks pressure and retains its full VBAR
recovery if cache release does not resolve the shortage. CPU reads and other
platforms do not enter this Windows policy.

In the two-run Krea2 diagnostic, the old order produced a warm median of
2.609 seconds with 8.840/8.548-second spikes. The new order invoked recovery
twice, returning 1302 and 930 MiB in 49 and 26 ms; no full-model spike appeared.
Cold late steps were 1.746–1.819 seconds, and warm steps were 2.166–2.249 seconds.
This is an improvement, not the final <=2-second acceptance pass. The isolated
reader/cache/completion regression group passed 60 tests.

## Opt-in reader completion bridge

`AIMDO_XPU_ASYNC_FILE_READER=1` reconnects the common reader's three-slot
pipeline on Windows XPU. It remains opt-in while workflow validation runs.
Only malloc-backed `cuMemAllocHost` slots may submit asynchronously, and only
on in-order queues. Ordinary HostBuffer copies keep their blocking contract.
No pinned allocation, model cache, or attention geometry is introduced.

Each slot owns its latest copy event on every queue that used it, including a
copy of the queue itself. Retirement snapshots these actual owners rather than
dereferencing the saved Python stream pointer. Reuse and cleanup wait for those
events. A failed submission retains a queue-wide completion obligation; failed
event creation/record/wait preserves staging, and checked Python cleanup raises
instead of concealing failure. Host free independently verifies completion.

The production C reader passed delayed-copy/failure-injection checks for ring
reuse, event creation, event recording, wait failure, and partial submission.
The selected cache/completion/pressure/reader regression group passed 74 tests.
`tests/run_windows_xpu_reader.py` passed three real-device rounds, checking
755,055,675 bytes in full with zero errors across alternating streams and
cleanup. It also checked unchanged synchronous HostBuffer behavior and VBAR
copy/consumer completion before explicit eviction. These are bounded lifecycle
checks, not a pressure or workflow acceptance result.

The preceding H3 diagnostic with the copy-cache recovery and two sampling-only
offload streams completed without a crash but had a 53.39-second step median.
Disabling only `AIMDO_XPU_NATIVE_CACHE_TRIM` in the same configuration reduced
the median to 37.71 seconds. Both miss the 25-second requirement. Keep these
negative results when evaluating the reader bridge; do not treat the Krea2
cache-recovery improvement as a universal H3 policy.

The full H3 reader-overlap trial subsequently failed in its first step with
an access violation during LoRA prefetch. The after-exception native snapshot
does not identify the original fault instruction. Keep this route disabled
by default; the bounded reader tests above do not establish pressure safety.

Disabling the UR USM allocator (`UR_L0_DISABLE_USM_ALLOCATOR=1`) also failed
the second Krea2 workflow with an access violation in `ze_intel_gpu64.dll`
1.15.38308.0 at offset `0x308b4e`. The first workflow had a 2.45-second median;
the second slowed to 5.2–5.5 seconds before failing. Keep the default UR pool.
The ten-iteration CPU-reference recovery GEMM passed after this failure.

## VMM sibling-copy residency lifetime

A bounded two-page reproduction now isolates a missing ownership dependency.
Page A's own consumer and retirement fence complete. A device copy into page B
of the same reservation is then queued behind a warmed 12288-square BF16 GEMM.
Reclaiming A before submitting the pending copy produces access violation
`c0000005` at `ze_intel_gpu64.dll` 1.15.38308.0 offset `0x308b4e`, also observed
in the Krea2 pressure failure. The probe uses less than 1 GiB, including only
64 MiB of VBAR mappings; it does not allocate the budget requested to exercise
reclaim. Waiting for B's copy before reclaim passes eight cycles on the same
installation, and the recovery GEMM passed after the original failure.

The Intel compute-runtime source at `9bc5aeb1faf4a13a2f13b1f595c51e47ea4ad289`
explains the extra dependency: `CommandListCoreFamily::addVirtualReservationToResidency`
adds all mapped siblings in a copy argument's virtual reservation to the
command list. `Context::destroyPhysicalMem` destroys the graphics allocation.
A per-page kernel fence alone therefore cannot protect the driver's pending
copy residency list. The source reference is a diagnostic explanation, not a
claim that this source revision built the installed driver.

The Windows provider now observes existing UR 1D/2D copy and fill completions. A copy
holds each touched VBAR reservation until its retained event completes; the
source and destination are both covered. Normal owner reclaim polls without
waiting. Existing synchronization boundaries drain copy owners too. Submission
callbacks only acquire a metadata hold with try-lock and retain handles; they
do not wait, unmap, or release VBAR ownership while inside the SYCL callback.
Failed or partially submitted copies retain their hold until a successful
owner-side queue wait. The allocation arbitration policy remains separate.

The initial fixed two-page probe passed eight cycles, retained A while B's copy
was pending, reclaimed A afterward, and checked every copied byte. Full
The native hook regression, completion, budget, cache, reader and packaging
group passed 54 tests, with one Linux-only skip. The real-device regression
also passed eight normal and eight worker-thread cycles, checking all 512 MiB
copied, preserving the pending copy's siblings, freeing an unrelated VBAR,
and leaving zero pending completion owners. Full
workflow acceptance is still required; these bounded results alone do not
establish the Krea2/H3 performance or ClipProj reload result.

## Rejected Krea2 SwiGLU gate expansion

The `dg2.3` kernel trial with the Krea2 SwiGLU gate expanded to DG2 completed
one cold workflow but failed after the next workflow's first step, at driver
offset `0x308f1c`. The isolated complete FFN measurement also found an
independent memory regression: for M4192/K6144/N16384, the original expression
peaked at 302,596,608 bytes, while fusion peaked at 536,887,808 bytes (or
415,236,096 with gate/up locals deleted). Three interleaved groups had original
medians 21.931/21.770/21.907 ms and fused medians around 21.3–21.5 ms, with zero
reference errors. That small speed difference does not justify the additional
peak for this full-memory model. The plugin's original BMG-only gate is restored.
This does not identify SwiGLU as the sole cause of the workflow access violation.

## Cast-buffer reservation teardown on DG2

ComfyUI's `reset_cast_buffers()` drops the per-stream `VRAMBuffer` at every
node boundary, and `vrambuf_destroy()` then issued one `cuMemUnmap` covering
the whole grown range before releasing the physical handles and freeing the
reservation. Level Zero on Windows/DG2 (`ze_intel_gpu64.dll` 1.15.38308) does
not tear down every mapping when one `zeVirtualMemUnmap` spans several
`zeVirtualMemMap` ranges: the reservation is freed with stale mapped
allocations behind it, and the next command that references a fresh
reservation in that context faults at driver offset `0x308b4e`. In ComfyUI
this is the MiniMax H3 + LoRA second-round access violation inside
`read_file_to_device` during LoRA-patch prefetch; it needs the async prefetch
streams only because they are what makes the cast buffer grow past one chunk.

`tests/run_xpu_vrambuf_recycle.py` reproduces it without ComfyUI: create a
`VRAMBuffer`, grow it by two or more 16 MiB chunks, copy into it, destroy it,
repeat. With the single-range unmap the second cycle crashes on the first
copy; a one-chunk buffer never does, and holding every buffer (no destroy)
never does. `vrambuf_destroy()` now unmaps chunk by chunk on the XPU build, in
the same ranges `vrambuf_grow()` mapped. Eight four-chunk cycles and six
twelve-chunk cycles pass on A770 after the change.
