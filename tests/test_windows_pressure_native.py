"""Run native budget regressions without reserving or using a GPU."""
import os
import platform
import subprocess
from pathlib import Path

import pytest


def test_windows_pressure_and_logging_native():
    if platform.system() != "Windows":
        pytest.skip("Windows DXGI pressure calculation")
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
    vswhere /= "Microsoft Visual Studio/Installer/vswhere.exe"
    if not os.environ.get("VS_PATH") and not vswhere.exists():
        pytest.skip("requires Visual Studio Build Tools")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(root / "scripts" / "test-windows-pressure.cmd")],
        cwd=root, capture_output=True, text=True, errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED (0 failures)" in result.stdout
