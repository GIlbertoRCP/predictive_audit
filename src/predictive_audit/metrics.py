"""Metrics, all oriented so that higher is better (error metrics are negated in ``score``)."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

CLASSIFICATION = ("f1", "pr_auc", "mcc", "balanced_accuracy", "accuracy", "roc_auc")
REGRESSION = ("mae", "rmse", "r2")
LOWER_IS_BETTER = {"mae", "rmse"}


def _labels(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_score >= threshold).astype(int)


def compute(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raw metric value (MAE/RMSE stay as errors here)."""
    if name == "f1":
        return skm.f1_score(y_true, _labels(y_pred), zero_division=0)
    if name == "accuracy":
        return skm.accuracy_score(y_true, _labels(y_pred))
    if name == "balanced_accuracy":
        return skm.balanced_accuracy_score(y_true, _labels(y_pred))
    if name == "mcc":
        return skm.matthews_corrcoef(y_true, _labels(y_pred))
    if name == "pr_auc":
        return _undefined_if_one_class(skm.average_precision_score, y_true, y_pred)
    if name == "roc_auc":
        return _undefined_if_one_class(skm.roc_auc_score, y_true, y_pred)
    if name == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if name == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if name == "r2":
        return skm.r2_score(y_true, y_pred)
    raise ValueError(f"unknown metric {name!r}")


def score(name: str, y_true, y_pred) -> float:
    """Higher-is-better version of ``compute``."""
    value = compute(name, y_true, y_pred)
    return -value if name in LOWER_IS_BETTER else value


def table(task: str, y_true, predictions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = CLASSIFICATION if task == "classification" else REGRESSION
    return {model: {m: compute(m, y_true, p) for m in names} for model, p in predictions.items()}


def _undefined_if_one_class(fn, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_pred))


def block_bootstrap_diff(metric, y_true, model, baseline, block, n_boot, seed):
    """Moving-block bootstrap of score(model) - score(baseline).

    Rows must be in time order (per group, groups concatenated). Blocks keep the
    autocorrelation that an i.i.d. bootstrap would destroy, which matters because
    i.i.d. resampling makes intervals on autocorrelated series far too narrow.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        yt = y_true[idx]
        diffs[i] = score(metric, yt, model[idx]) - score(metric, yt, baseline[idx])
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
