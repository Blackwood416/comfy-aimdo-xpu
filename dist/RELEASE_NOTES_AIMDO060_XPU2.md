# comfy-aimdo-xpu 0.6.0+xpu.2 (Windows / DG2 / oneAPI 2026.2)

A770 Windows provider maintenance build for **Intel oneAPI 2026.2 Toolkit**, **PyTorch 2.15.0+xpu**, and **ComfyUI 0.44.x** (Python 3.13, Windows 10/11 x64).
This release adapts the Unified Runtime USM memory detour hook to Intel oneAPI 2026.2's renamed loader (`ur_loader_shared.dll`) with full bidirectional compatibility.

## What changed

### 1. Intel oneAPI 2026.2 UR Loader Adaptation
- In Intel oneAPI 2026.2, the Unified Runtime loader dynamic library was officially renamed from `ur_loader.dll` to `ur_loader_shared.dll`.
- Updated `src-xpu/ur-usm-detour.c:resolve_ur_loader()` with fallback resolution to `GetModuleHandleA("ur_loader_shared.dll")`.
- Recompiled `comfy_aimdo/aimdo_xpu.dll` using Intel oneAPI DPC++/C++ 2026.2.0 + MSVC 19.44 to ensure complete symbol alignment.
- Verified seamless hook attachment and DynamicVRAM offloading under clean oneAPI 2026.2 installations.

### 2. Preserved Windows & DG2 Architectural Safeguards
- Retained chunked VRAMBuffer unmapping to prevent Level Zero stale reservation driver faults.
- Retained double-slot advance fix in `src/hostbuf-file-reader.c`.
- Retained safe PyTorch import sequencing in Windows XPU completion hooks.

## Assets

1. **`comfy_aimdo_xpu_runtime-0.6.0+xpu.2.g7e7b686-cp39-abi3-win_amd64.whl`**
   - Co-installable runtime provider wheel.
   - Activates automatically via `comfyui_omnixpu.runtime_providers` entry points alongside official `comfy-aimdo 0.6.0`.
   - **SHA256**: `c49819c05df568c18963c5021357e954eb83afc38b5ef2ca577a08ff35815b2a`

2. **`comfy_aimdo-0.6.0+xpu.2.g7e7b686-cp39-abi3-win_amd64.whl`**
   - Standalone AIMDO package embedding updated `aimdo_xpu.dll`.
   - **SHA256**: `d1a41afee6308443ffd91e716eec07c5c05ddcb8b0e69c551817af9f1bef88a0`

## Install (Provider)

```powershell
pip install --force-reinstall --no-deps comfy_aimdo_xpu_runtime-0.6.0+xpu.2.g7e7b686-cp39-abi3-win_amd64.whl
```
