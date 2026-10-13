"""Trivial forecasters built from the target column alone.

Convention: row t holds ``target[t] = y[t + h]`` for horizon h, rows ordered in time
within each group. The value of y known at time t is therefore ``target[t - h]``,
so every baseline here is a within-group lag of the target column. A baseline is
only allowed to look at targets whose lag is >= h, i.e. values already observed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def lag_target(target: pd.Series, groups: pd.Series | None, k: int) -> pd.Series:
    """target[t - k] within each group (NaN where the group has no history yet)."""
    if groups is None:
        return target.shift(k)
    return target.groupby(groups, sort=False).shift(k)


def persistence(target, groups, horizon):
    """y[t + h] := y[t]. What a reactive system implicitly assumes."""
    return lag_target(target, groups, horizon)


def seasonal_naive(target, groups, horizon, season):
    """y[t + h] := y[t + h - k*m], the most recent observed value one or more seasons back."""
    k = math.ceil(horizon / season) * season
    return lag_target(target, groups, k)


def moving_average(target, groups, horizon, window):
    """Mean of the last ``window`` observed values. For a 0/1 label this is a score in [0, 1]."""
    observed = lag_target(target, groups, horizon)
    if groups is None:
        return observed.rolling(window, min_periods=window).mean()
    return observed.groupby(groups, sort=False).transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )


def constant(target, fit_mask, task):
    """Climatology: training-set mean (regression) or majority class (classification)."""
    fit = target[fit_mask].dropna()
    if task == "classification":
        value = float(fit.mean() >= 0.5)
    else:
        value = float(fit.mean())
    return pd.Series(value, index=target.index)


def estimate_season(target, groups, fit_mask, max_lag=2000, min_lag=2):
    """Dominant period of the target, from the autocorrelation of fit (non-test) rows only.

    Autocorrelation is pooled across groups. A period counts only if the autocorrelation
    first decays below 0.5 and then rises again by at least 0.2 to a peak of 0.3 or more;
    smooth trending series (random walks) have no such rise and return None.
    """
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
        acov = np.fft.irfft(f * np.conj(f))[: min(len(s), top + 1)]  # sum of s[:-k] * s[k:]
        num[: len(acov)] += acov
        den += float(s @ s)
    ac = num / den
    decayed = np.nonzero(ac[min_lag:] < 0.5)[0]
    if len(decayed) == 0:
        return None
    start = min_lag + int(decayed[0])
    peak = start + int(np.argmax(ac[start:]))
    trough = float(ac[start:peak + 1].min())
    if ac[peak] < 0.3 or ac[peak] - trough < 0.2:
        return None
    return peak


def all_baselines(target, groups, horizon, task, fit_mask, season=None, window=None):
    """Dict of baseline name -> prediction Series aligned to ``target``."""
    window = window or max(3, horizon)
    out = {
        "persistence": persistence(target, groups, horizon),
        f"moving average ({window})": moving_average(target, groups, horizon, window),
        "majority class" if task == "classification" else "training mean": constant(target, fit_mask, task),
    }
    if season:
        out[f"seasonal naive (m={season})"] = seasonal_naive(target, groups, horizon, season)
    return out

