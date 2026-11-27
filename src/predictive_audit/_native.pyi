"""Type stubs for the nanobind extension module built from native/python/module.cpp."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_U8 = npt.NDArray[np.uint8]

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
def lag(values: _F64, groups: _I64 | None, k: int) -> _F64: ...
def rolling_mean(values: _F64, groups: _I64 | None, window: int, min_periods: int) -> _F64: ...
def ewm_mean(values: _F64, groups: _I64 | None, span: float) -> _F64: ...
def linear_ar(
    target: _F64,
    groups: _I64 | None,
    horizon: int,
    fit_mask: _U8,
    lags: int,
    alpha: float,
    clip_unit_interval: bool,
) -> _F64: ...
def estimate_season(
    target: _F64, groups: _I64 | None, fit_mask: _U8, max_lag: int = 2000, min_lag: int = 2
) -> int | None: ...
def optimal_threshold(metric: str, y_true: _F64, y_score: _F64) -> float: ...
def rng_integers(seed: int, high: int, size: int) -> _I64: ...
