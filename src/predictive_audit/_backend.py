"""Choose between the native C++ engine and the pure-Python reference implementation.

Set ``PREDICTIVE_AUDIT_BACKEND`` before importing the package:

- ``auto`` (default): use the native engine if the extension is importable, else Python.
- ``native``: require the native engine; importing fails if the extension is missing.
- ``python``: always use the pure-Python implementation.

The Python code paths are kept permanently as the reference oracle: every native kernel is
added behind the same function signature and dispatches on :data:`native` at call time, so
parity tests can run the same inputs through both backends (see :func:`use`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Literal

ENV_VAR = "PREDICTIVE_AUDIT_BACKEND"
BackendName = Literal["native", "python"]
_CHOICES = ("auto", "native", "python")


def _load_native() -> ModuleType | None:
    try:
        from . import _native
    except ImportError:
        return None
    return _native


def _resolve(choice: str) -> ModuleType | None:
    choice = choice.strip().lower() or "auto"
    if choice not in _CHOICES:
        raise ValueError(f"{ENV_VAR} must be one of {_CHOICES}; got {choice!r}")
    if choice == "python":
        return None
    mod = _load_native()
    if mod is None and choice == "native":
        raise ImportError(
            f"{ENV_VAR}=native but the predictive_audit._native extension is not available; "
            "reinstall the package with a C++20 compiler or use a prebuilt wheel"
        )
    return mod


#: The native extension module when active, else ``None``. Kernels check this at call time.
native: ModuleType | None = _resolve(os.environ.get(ENV_VAR, "auto"))


def name() -> BackendName:
    """Name of the active backend."""
    return "native" if native is not None else "python"


def native_available() -> bool:
    """Whether the native extension can be imported, regardless of the active backend."""
    return _load_native() is not None


@contextmanager
def use(backend: BackendName) -> Iterator[None]:
    """Temporarily switch backends (for parity tests and benchmarks). Not thread-safe."""
    global native
    previous = native
    native = _resolve(backend)
    try:
        yield
    finally:
        native = previous
