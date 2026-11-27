"""Backend selection: native C++ engine vs pure-Python reference."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import predictive_audit
from predictive_audit import _backend

PROBE = "import predictive_audit as pa; print(pa.backend())"


def _run_with_backend(value: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, _backend.ENV_VAR: value}
    return subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)


def test_python_backend_can_always_be_forced() -> None:
    proc = _run_with_backend("python")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "python"


def test_auto_prefers_native_when_available() -> None:
    proc = _run_with_backend("auto")
    assert proc.returncode == 0, proc.stderr
    expected = "native" if _backend.native_available() else "python"
    assert proc.stdout.strip() == expected


def test_invalid_backend_is_rejected() -> None:
    proc = _run_with_backend("fortran")
    assert proc.returncode != 0
    assert _backend.ENV_VAR in proc.stderr


def test_backend_matches_environment() -> None:
    requested = os.environ.get(_backend.ENV_VAR, "auto").strip().lower() or "auto"
    if requested == "python":
        assert predictive_audit.backend() == "python"
    elif requested == "native":
        assert predictive_audit.backend() == "native"


def test_use_switches_and_restores() -> None:
    before = predictive_audit.backend()
    with _backend.use("python"):
        assert predictive_audit.backend() == "python"
    assert predictive_audit.backend() == before


@pytest.mark.skipif(not _backend.native_available(), reason="native extension not built")
class TestNativeModule:
    def test_native_backend_can_be_forced(self) -> None:
        proc = _run_with_backend("native")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "native"

    def test_version_matches_package(self) -> None:
        from predictive_audit import _native

        assert _native.version() == predictive_audit.__version__

    def test_build_info(self) -> None:
        from predictive_audit import _native

        info = _native.build_info()
        assert info["version"] == predictive_audit.__version__
        assert info["cplusplus"] >= 202002
        assert info["hardware_threads"] >= 0
        assert info["compiler"]
