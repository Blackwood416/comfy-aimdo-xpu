"""Fault injection against the production reader without a GPU."""
import platform
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows C toolchain")
def test_reader_completion_failure_and_reuse():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(root / "scripts/test-windows-reader.cmd")], cwd=root,
        capture_output=True, text=True, errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED (0 failures)" in result.stdout
