"""Phase 2 native kernels vs the pandas / scikit-learn reference path.

Bit-identical: group lags, pandas rolling means (Kahan add/remove), ROC-AUC, average precision,
their bootstrap intervals, and the optimal-threshold search. Two documented exceptions:
- EWMA: pandas' compiled kernel fuses `a*b + c` into an FMA on some platforms (its arm64 wheels),
  so its result is platform-dependent. The native kernel never fuses, so it matches pandas
  exactly where pandas does not fuse (x86-64) and to ~1e-13 elsewhere.
- Ridge AR: NumPy solves with LAPACK/BLAS, whose summation order is unspecified.
"""

from __future__ import annotations

import platform
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predictive_audit import _backend, baselines, metrics

pytestmark = [
    pytest.mark.skipif(not _backend.native_available(), reason="native extension not built"),
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning"),
]

PANDAS_FUSES_FMA = platform.machine().lower() in {"arm64", "aarch64"}


def _both(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    with _backend.use("native"):
        native = fn(*args, **kwargs)
    with _backend.use("python"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        python = fn(*args, **kwargs)
    return native, python


def _same(a: Any, b: Any) -> bool:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a_arr.shape == b_arr.shape and bool(np.all((a_arr == b_arr) | (np.isnan(a_arr) & np.isnan(b_arr))))


def _close(a: Any, b: Any, rel: float) -> bool:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a_arr.shape == b_arr.shape and bool(
        np.allclose(a_arr, b_arr, rtol=rel, atol=1e-300, equal_nan=True)
    )


@st.composite
def series_and_groups(draw: st.DrawFn) -> tuple[pd.Series, pd.Series | None]:
    n = draw(st.integers(1, 400))
    seed = draw(st.integers(0, 2**32))
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n).cumsum() * draw(st.sampled_from([1.0, 1e-3, 1e6]))
    if draw(st.booleans()):
        y[rng.random(n) < 0.1] = np.nan
    if draw(st.booleans()):
        y[: n // 3] = 0.1  # constant stretch exercises pandas' same-value guard
    if draw(st.booleans()):
        y = -np.abs(y)  # all-negative windows exercise the sign guard
    if draw(st.booleans()) and n > 2:
        y[int(rng.integers(0, n))] = np.inf
    n_groups = draw(st.integers(0, 5))
    if n_groups == 0:
        return pd.Series(y, name="y"), None
    codes = rng.integers(0, n_groups, n) if draw(st.booleans()) else np.sort(rng.integers(0, n_groups, n))
    return pd.Series(y, name="y"), pd.Series(codes)


@settings(max_examples=80, deadline=None)
@given(data=series_and_groups(), horizon=st.integers(0, 4), window=st.integers(1, 9))
def test_lag_and_moving_average_are_bit_identical(
    data: tuple[pd.Series, pd.Series | None], horizon: int, window: int
) -> None:
    target, groups = data
    assert _same(*_both(baselines.lag_target, target, groups, horizon))
    assert _same(*_both(baselines.moving_average, target, groups, horizon, window))


@settings(max_examples=80, deadline=None)
@given(data=series_and_groups(), horizon=st.integers(0, 4), span=st.integers(1, 9))
def test_ewma_matches_pandas(data: tuple[pd.Series, pd.Series | None], horizon: int, span: int) -> None:
    target, groups = data
    native, python = _both(baselines.ewma, target, groups, horizon, span)
    if PANDAS_FUSES_FMA:
        assert _close(native, python, rel=1e-12)
    else:
        assert _same(native, python)


@settings(max_examples=40, deadline=None)
@given(data=series_and_groups(), lags=st.integers(1, 5), horizon=st.integers(1, 3), clf=st.booleans())
def test_linear_ar_matches(
    data: tuple[pd.Series, pd.Series | None], lags: int, horizon: int, clf: bool
) -> None:
    target, groups = data
    target = target.replace([np.inf, -np.inf], np.nan)
    fit = pd.Series(np.arange(len(target)) % 5 != 0, index=target.index)
    task = "classification" if clf else "regression"
    native, python = _both(baselines.linear_ar, target, groups, horizon, fit, lags=lags, task=task)
    # Two correct solvers agree to about cond(X'X + alpha*I) * eps, so scale the tolerance by the
    # worst-conditioned system the baseline actually solves.
    rtol = max(1e-9, _ridge_condition(target, groups, horizon, fit, lags) * 1e-14)
    scale = float(np.nanmax(np.abs(python.to_numpy()))) if python.notna().any() else 1.0
    np.testing.assert_allclose(native, python, rtol=rtol, atol=rtol * max(scale, 1.0), equal_nan=True)


def _ridge_condition(
    target: pd.Series, groups: pd.Series | None, horizon: int, fit: pd.Series, lags: int, alpha: float = 1.0
) -> float:
    with _backend.use("python"):
        cols = [baselines.lag_target(target, groups, horizon + k).to_numpy() for k in range(lags)]
    x = np.column_stack([*cols, np.ones(len(target))])
    ok = fit.to_numpy() & ~np.isnan(x).any(axis=1) & ~np.isnan(target.to_numpy())
    reg = np.eye(lags + 1) * alpha
    reg[-1, -1] = 0.0
    masks = [ok]
    if groups is not None:
        g = groups.to_numpy()
        masks += [ok & (g == v) for v in np.unique(g) if int((ok & (g == v)).sum()) >= lags + 5]
    conds = [float(np.linalg.cond(x[m].T @ x[m] + reg)) for m in masks if m.any()]
    return max(conds, default=1.0)


@settings(max_examples=40, deadline=None)
@given(data=series_and_groups(), period=st.integers(3, 60), noise=st.floats(0.0, 2.0))
def test_estimate_season_matches(data: tuple[pd.Series, pd.Series | None], period: int, noise: float) -> None:
    target, groups = data
    t = np.arange(len(target))
    jitter = np.nan_to_num(target.to_numpy(), posinf=0.0, neginf=0.0) * 1e-6
    seasonal = pd.Series(np.sin(2 * np.pi * t / period) + noise * jitter)
    fit = pd.Series(t % 7 != 0)
    for values in (seasonal, target):
        assert _same(*_both(baselines.estimate_season, values, groups, fit))


@st.composite
def labels_and_scores(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = draw(st.integers(2, 600))
    rng = np.random.default_rng(draw(st.integers(0, 2**32)))
    y = (rng.random(n) < draw(st.floats(0.0, 1.0))).astype(float)
    decimals = draw(st.integers(0, 4))  # few decimals -> heavy ties
    s1 = np.round(np.clip(y * 0.3 + rng.random(n), 0, 2), decimals)
    s2 = np.round(rng.random(n), decimals)
    return y, s1, s2


@settings(max_examples=80, deadline=None)
@given(data=labels_and_scores())
def test_rank_metrics_are_bit_identical(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y, s1, _ = data
    for metric in ("roc_auc", "pr_auc"):
        assert _same(*_both(metrics.compute, metric, y, s1))


@settings(max_examples=40, deadline=None)
@given(
    data=labels_and_scores(),
    block=st.integers(-2, 40),
    seed=st.integers(0, 2**32),
    mode=st.sampled_from(["plain", "groups", "panel"]),
)
def test_rank_bootstrap_is_bit_identical(
    data: tuple[np.ndarray, np.ndarray, np.ndarray], block: int, seed: int, mode: str
) -> None:
    y, s1, s2 = data
    n = len(y)
    groups = np.sort(np.arange(n) % 4)
    times = np.concatenate([np.arange(int((groups == g).sum())) for g in range(4)])
    kwargs: dict[str, Any] = {}
    if mode != "plain":
        kwargs["groups"] = groups
    if mode == "panel":
        kwargs["times"] = times
    for metric in ("roc_auc", "pr_auc"):
        native, python = _both(metrics.block_bootstrap_diff, metric, y, s1, s2, block, 30, seed, **kwargs)
        assert _same(native, python), (metric, native, python)


@settings(max_examples=60, deadline=None)
@given(data=labels_and_scores(), metric=st.sampled_from(["f1", "mcc", "balanced_accuracy", "accuracy"]))
def test_optimal_threshold_is_bit_identical(
    data: tuple[np.ndarray, np.ndarray, np.ndarray], metric: str
) -> None:
    y, s1, s2 = data
    for scores in (s1, s2, np.where(np.arange(len(s2)) % 9 == 0, np.nan, s2)):
        assert _same(*_both(metrics.optimal_threshold, metric, y, scores))


def test_unsupported_inputs_fall_back_to_python() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=300).cumsum())
    # float32 targets and non-integer group keys take the pandas path, unchanged.
    assert _same(*_both(baselines.moving_average, y.astype(np.float32), None, 1, 3))
    groups = pd.Series(np.where(np.arange(300) < 150, "a", "b"))
    assert _same(*_both(baselines.ewma, y, groups, 1, 3))
    # Groups indexed differently from the target are aligned by pandas, so they fall back.
    shifted = pd.Series(np.arange(300) // 100, index=np.arange(300) + 1)
    assert _same(*_both(baselines.lag_target, y, shifted, 1))
    # Labels other than 0/1, or non-finite scores, keep sklearn's behaviour.
    y3 = np.array([0.0, 2.0, 2.0, 0.0])
    s = np.array([0.1, 0.9, 0.8, 0.2])
    assert _same(*_both(metrics.compute, "roc_auc", y3, s))
    with pytest.raises(ValueError), _backend.use("native"):
        metrics.compute("roc_auc", np.array([0.0, 1.0]), np.array([0.5, np.inf]))


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        # pandas shifts a bool Series to object dtype (True/False/NaN), not float.
        (baselines.lag_target, (pd.Series([True, False, True, True]), None, 1)),
        # shift(0) keeps an integer dtype.
        (baselines.lag_target, (pd.Series([1, 2, 3]), None, 0)),
        # Nullable group keys: pandas drops pd.NA groups (their rows get NaN).
        (baselines.lag_target, (pd.Series([1.0, 2, 3, 4]), pd.Series([0, pd.NA, 0, 1], dtype="Int64"), 1)),
        (
            baselines.moving_average,
            (pd.Series([1.0, 2, 3, 4, 5]), pd.Series([0, 0, pd.NA, 0, 0], dtype="Int64"), 1, 2),
        ),
        (baselines.ewma, (pd.Series([1.0, 2, 3, 4]), pd.Series([0, pd.NA, 0, 0], dtype="Int64"), 1, 2)),
        # Nullable float targets carry pd.NA through pandas' own arithmetic.
        (baselines.lag_target, (pd.Series([1.0, None, 3.0], dtype="Float64"), None, 1)),
        # Integer targets are fine natively: same values and float64 dtype as pandas.
        (baselines.lag_target, (pd.Series([1, 2, 3, 4]), pd.Series([0, 1, 0, 1]), 1)),
        (baselines.moving_average, (pd.Series([True, False, True, True, False]), None, 1, 2)),
    ],
)
def test_dtype_edge_cases_match_pandas_exactly(fn: Callable[..., pd.Series], args: tuple[Any, ...]) -> None:
    native, python = _both(fn, *args)
    pd.testing.assert_series_equal(native, python, check_exact=True)
