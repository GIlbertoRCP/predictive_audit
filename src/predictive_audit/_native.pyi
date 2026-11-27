"""Type stubs for the nanobind extension module built from native/python/module.cpp."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]

class BuildInfo(TypedDict):
    version: str
    compiler: str
    build_type: str
    cplusplus: int
    hardware_threads: int

def version() -> str: ...
def build_info() -> BuildInfo: ...
def num_threads() -> int: ...
def set_num_threads(n: int) -> None: ...
def compute(metric: str, y_true: _F64, y_pred: _F64, threshold: float = 0.5) -> float: ...
def has_native_compute(metric: str) -> bool: ...
def block_bootstrap_diff(
    metric: str,
    y_true: _F64,
    model: _F64,
    baseline: _F64,
    block: int,
    n_boot: int,
    seed: int,
    groups: _I64 | None = None,
    times_i64: _I64 | None = None,
    times_f64: _F64 | None = None,
    threshold: float = 0.5,
) -> tuple[float, float]: ...
def estimate_block_length(x: _F64, max_block: int | None = None) -> int: ...
def default_block_length(
    n_total: int, groups: _I64 | None = None, times_i64: _I64 | None = None, times_f64: _F64 | None = None
) -> int: ...
def is_aligned_panel(
    groups: _I64 | None, times_i64: _I64 | None = None, times_f64: _F64 | None = None
) -> tuple[bool, _I64 | None, int]: ...
def group_circular_layout(groups: _I64, pad: int) -> tuple[_I64, _I64]: ...
def diebold_mariano(
    metric: str,
    y_true: _F64,
    model: _F64,
    baseline: _F64,
    horizon: int = 1,
    groups: _I64 | None = None,
    times_i64: _I64 | None = None,
    times_f64: _F64 | None = None,
    threshold: float = 0.5,
) -> tuple[float, float]: ...
def benjamini_hochberg(p: _F64) -> _F64: ...
def holm_bonferroni(p: _F64) -> _F64: ...
def student_t_two_sided_p(t: float, df: float) -> float: ...
def rng_integers(seed: int, high: int, size: int) -> _I64: ...
