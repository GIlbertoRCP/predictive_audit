"""Causal baselines built from the target column alone.

Convention: row t holds ``target[t] = y[t + h]`` for horizon h, rows ordered in time
within each group. The value of y known at time t is therefore ``target[t - h]``,
so every baseline here only looks at targets whose lag is >= h (already observed).
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from . import _backend

# --- native dispatch helpers ----------------------------------------------------


def _native_inputs(
    target: pd.Series, groups: pd.Series | None
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """(values_f64, group_codes_i64) for the native engine, or None to use the pandas path.

    Only plain NumPy dtypes go native: float64 or integer targets, and integer/bool group codes
    aligned with the target (pandas would otherwise align them by index). Bool targets (pandas
    shifts them to object dtype) and extension dtypes such as nullable ``Int64`` / ``Float64``
    (whose ``pd.NA`` pandas treats as a missing group or value) keep the pandas path.
    """
    if _backend.native is None:
        return None
    numeric = target.dtype == np.float64 or target.dtype.kind in "iu"
    if not isinstance(target.dtype, np.dtype) or not numeric:
        return None
    if groups is not None:
        if not isinstance(groups.dtype, np.dtype) or groups.dtype.kind not in "iub":
            return None
        if not groups.index.equals(target.index):
            return None
        if groups.dtype.kind == "u" and len(groups) and int(groups.max()) > np.iinfo(np.int64).max:
            return None
    values = np.ascontiguousarray(target.to_numpy(dtype=np.float64))
    codes = None if groups is None else np.ascontiguousarray(groups.to_numpy(dtype=np.int64))
    return values, codes


def _as_series(values: np.ndarray, like: pd.Series, *, keep_name: bool = True) -> pd.Series:
    name = like.name if keep_name else None
    return cast(pd.Series, pd.Series(values, index=like.index, dtype=float, name=name))

def lag_target(target: pd.Series, groups: pd.Series | None, k: int) -> pd.Series:
    """target[t - k] within each group (NaN where the group has no history yet)."""
    # pandas keeps an integer dtype for shift(0); every other lag of a numeric series is float64.
    use_native = k > 0 or (k == 0 and target.dtype == np.float64)
    native_in = _native_inputs(target, groups) if use_native else None
    if native_in is not None and _backend.native is not None:
        return _as_series(_backend.native.lag(native_in[0], native_in[1], int(k)), target)
    if groups is None:
        return target.shift(k)
    return target.groupby(groups, sort=False).shift(k)


def persistence(target: pd.Series, groups: pd.Series | None, horizon: int) -> pd.Series:
    """y[t + h] := y[t]. What a reactive system implicitly assumes."""
    return lag_target(target, groups, horizon)


def seasonal_naive(
    target: pd.Series, groups: pd.Series | None, horizon: int, season: int
) -> pd.Series:
    """y[t + h] := y[t + h - k*m], the most recent observed value one or more seasons back."""
    k = math.ceil(horizon / season) * season
    return lag_target(target, groups, k)


def moving_average(
    target: pd.Series, groups: pd.Series | None, horizon: int, window: int
) -> pd.Series:
    """Mean of the last ``window`` observed values. For a 0/1 label this is a score in [0, 1]."""
    native_in = _native_inputs(target, groups) if window >= 1 and horizon >= 0 else None
    if native_in is not None and _backend.native is not None:
        values, codes = native_in
        observed_n = _backend.native.lag(values, codes, int(horizon))
        return _as_series(_backend.native.rolling_mean(observed_n, codes, int(window), int(window)), target)
    observed = lag_target(target, groups, horizon)
    if groups is None:
        return observed.rolling(window, min_periods=window).mean()
    res = observed.groupby(groups, sort=False).rolling(window, min_periods=window).mean().droplevel(0)
    return cast(pd.Series, res if res.index.equals(target.index) else res.reindex(target.index))


def ewma(
    target: pd.Series, groups: pd.Series | None, horizon: int, span: int
) -> pd.Series:
    """Exponentially weighted moving average over observed values."""
    native_in = _native_inputs(target, groups) if span >= 1 and horizon >= 0 else None
    if native_in is not None and _backend.native is not None:
        values, codes = native_in
        observed_n = _backend.native.lag(values, codes, int(horizon))
        return _as_series(_backend.native.ewm_mean(observed_n, codes, float(span)), target)
    observed = lag_target(target, groups, horizon)
    if groups is None:
        return observed.ewm(span=span, adjust=False, min_periods=1).mean()
    res = observed.groupby(groups, sort=False).ewm(span=span, adjust=False, min_periods=1).mean().droplevel(0)
    return cast(pd.Series, res if res.index.equals(target.index) else res.reindex(target.index))


def constant(target: pd.Series, fit_mask: pd.Series, task: str) -> pd.Series:
    """Climatology: training-set mean (regression) or majority class (classification)."""
    fit = target[fit_mask].dropna()
    value = float(fit.mean() >= 0.5) if task == "classification" else float(fit.mean())
    return cast(pd.Series, pd.Series(value, index=target.index, dtype=float))


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge regression with an unpenalized intercept in the last column."""
    d = X.shape[1]
    reg = np.eye(d, dtype=np.float64) * alpha
    reg[-1, -1] = 0.0
    xtx = X.T @ X + reg
    xty = X.T @ y
    try:
        return np.asarray(np.linalg.solve(xtx, xty), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=np.float64)


def linear_ar(
    target: pd.Series,
    groups: pd.Series | None,
    horizon: int,
    fit_mask: pd.Series,
    *,
    lags: int = 5,
    alpha: float = 1.0,
    task: str = "regression",
) -> pd.Series:
    """Causal Ridge AR(p) fit strictly on ``fit_mask`` using observed lags [h .. h+p-1]."""
    if lags < 1:
        raise ValueError("ar_lags must be >= 1")
    native_in = _native_inputs(target, groups) if horizon >= 0 and len(fit_mask) == len(target) else None
    if native_in is not None and _backend.native is not None:
        try:
            preds_n = _backend.native.linear_ar(
                native_in[0], native_in[1], int(horizon),
                np.ascontiguousarray(fit_mask.to_numpy(dtype=bool), dtype=np.uint8),
                int(lags), float(alpha), task == "classification",
            )
            return _as_series(preds_n, target, keep_name=False)
        except RuntimeError:
            pass  # singular system: NumPy's least-squares fallback below
    lag_cols = [lag_target(target, groups, horizon + k).to_numpy(dtype=np.float64) for k in range(lags)]
    X = np.column_stack([*lag_cols, np.ones(len(target), dtype=np.float64)])
    y = target.to_numpy(dtype=np.float64)
    fit = fit_mask.to_numpy(dtype=bool)
    valid_x = ~np.isnan(X).any(axis=1)
    valid_fit = fit & valid_x & ~np.isnan(y)

    preds = np.full(len(target), np.nan, dtype=np.float64)
    if not valid_fit.any():
        return cast(pd.Series, pd.Series(preds, index=target.index, dtype=float))

    # Pooled fallback weights for any group with too few training rows
    pooled_w = _fit_ridge(X[valid_fit], y[valid_fit], alpha)

    if groups is None:
        preds[valid_x] = X[valid_x] @ pooled_w
    else:
        g_arr = groups.to_numpy()
        for g_val in np.unique(g_arr):
            g_mask = g_arr == g_val
            g_fit = g_mask & valid_fit
            g_pred = g_mask & valid_x
            if not g_pred.any():
                continue
            w = _fit_ridge(X[g_fit], y[g_fit], alpha) if int(g_fit.sum()) >= lags + 5 else pooled_w
            preds[g_pred] = X[g_pred] @ w

    if task == "classification":
        preds = np.clip(preds, 0.0, 1.0)
    return cast(pd.Series, pd.Series(preds, index=target.index, dtype=float))


def estimate_season(
    target: pd.Series,
    groups: pd.Series | None,
    fit_mask: pd.Series,
    max_lag: int = 2000,
    min_lag: int = 2,
) -> int | None:
    """Dominant period of the target, from the autocorrelation of fit (non-test) rows only.

    Autocorrelation is pooled across groups. A period counts only if the autocorrelation
    first decays below 0.5 and then rises again by at least 0.2 to a peak of 0.3 or more;
    smooth trending series (random walks) have no such rise and return None.
    """
    native_in = _native_inputs(target, groups) if fit_mask.index.equals(target.index) else None
    if native_in is not None and _backend.native is not None and max_lag >= 0 and min_lag >= 0:
        mask = np.ascontiguousarray(fit_mask.to_numpy(dtype=bool), dtype=np.uint8)
        season = _backend.native.estimate_season(native_in[0], native_in[1], mask, int(max_lag), int(min_lag))
        return None if season is None else int(season)
    fit = target.where(fit_mask)
    keys = groups if groups is not None else pd.Series(0, index=target.index)
    series = [s.dropna().to_numpy(float) for _, s in fit.groupby(keys, sort=False)]
    series = [s - s.mean() for s in series if len(s) > 4 * min_lag and s.std() > 0]
    if not series:
        return None
    top = min(max_lag, max(len(s) for s in series) // 2)
    num, den = np.zeros(top + 1), 0.0
    for s in series:
        f = np.fft.rfft(s, 2 * len(s))
        acov = np.fft.irfft(f * np.conj(f))[: min(len(s), top + 1)]
        num[: len(acov)] += acov
        den += float(s @ s)
    ac = num / den
    decayed = np.nonzero(ac[min_lag:] < 0.5)[0]
    if len(decayed) == 0:
        return None
    start = min_lag + int(decayed[0])
    peak = start + int(np.argmax(ac[start:]))
    trough = float(np.min(ac[start : peak + 1]))
    if ac[peak] < 0.3 or ac[peak] - trough < 0.2:
        return None
    return peak


def all_baselines(
    target: pd.Series,
    groups: pd.Series | None,
    horizon: int,
    task: str,
    fit_mask: pd.Series,
    season: int | None = None,
    window: int | None = None,
    *,
    include_ewma: bool = False,
    ar_lags: int | None = None,
) -> dict[str, pd.Series]:
    """Dict of baseline name -> prediction Series aligned to ``target``."""
    window = window or max(3, horizon)
    out: dict[str, pd.Series] = {
        "persistence": persistence(target, groups, horizon),
        f"moving average ({window})": moving_average(target, groups, horizon, window),
        "majority class" if task == "classification" else "training mean": constant(
            target, fit_mask, task
        ),
    }
    if include_ewma:
        out[f"ewma ({window})"] = ewma(target, groups, horizon, window)
    if season:
        out[f"seasonal naive (m={season})"] = seasonal_naive(target, groups, horizon, season)
    if ar_lags:
        out[f"linear AR({ar_lags})"] = linear_ar(
            target, groups, horizon, fit_mask, lags=ar_lags, task=task
        )
    return out
