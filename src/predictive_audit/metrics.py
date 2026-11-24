"""Metrics, prefix-sum vectorized block bootstrap, Diebold-Mariano HAC tests, and FDR control.

All metrics are oriented so that higher is better in ``score`` (error metrics are negated).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from sklearn import metrics as skm

CLASSIFICATION: tuple[str, ...] = (
    "f1",
    "pr_auc",
    "mcc",
    "balanced_accuracy",
    "accuracy",
    "roc_auc",
    "brier",
)
REGRESSION: tuple[str, ...] = ("mae", "rmse", "r2", "smape")
LOWER_IS_BETTER: set[str] = {"mae", "rmse", "smape", "brier"}
ADDITIVE_LOSS_METRICS: set[str] = {"mae", "rmse", "smape", "accuracy", "brier"}
THRESHOLDED_METRICS: set[str] = {"f1", "mcc", "balanced_accuracy", "accuracy"}
_PREFIX_METRICS: set[str] = {
    "mae",
    "rmse",
    "r2",
    "smape",
    "accuracy",
    "brier",
    "f1",
    "mcc",
    "balanced_accuracy",
}


def _labels(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_score >= threshold).astype(np.int8)


def _confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> tuple[float, float, float, float]:
    yt = y_true >= 0.5
    yp = y_pred >= threshold
    tp = float(np.count_nonzero(yt & yp))
    fp = float(np.count_nonzero((~yt) & yp))
    fn = float(np.count_nonzero(yt & (~yp)))
    tn = float(np.count_nonzero((~yt) & (~yp)))
    return tp, fp, fn, tn


def compute(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5
) -> float:
    """Raw metric value (MAE/RMSE/sMAPE/Brier stay as errors here)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if name == "f1":
        tp, fp, fn, _ = _confusion_counts(yt, yp, threshold)
        denom = 2.0 * tp + fp + fn
        return (2.0 * tp / denom) if denom > 0.0 else 0.0
    if name == "accuracy":
        return float(np.mean((yt >= 0.5) == (yp >= threshold)))
    if name == "balanced_accuracy":
        tp, fp, fn, tn = _confusion_counts(yt, yp, threshold)
        pos = tp + fn
        neg = tn + fp
        if pos > 0.0 and neg > 0.0:
            return 0.5 * (tp / pos + tn / neg)
        if pos > 0.0:
            return tp / pos
        return (tn / neg) if neg > 0.0 else 0.0
    if name == "mcc":
        tp, fp, fn, tn = _confusion_counts(yt, yp, threshold)
        denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        return ((tp * tn - fp * fn) / denom) if denom > 0.0 else 0.0
    if name == "brier":
        return float(np.mean((yt - yp) ** 2))
    if name == "pr_auc":
        return _undefined_if_one_class(skm.average_precision_score, yt, yp)
    if name == "roc_auc":
        return _undefined_if_one_class(skm.roc_auc_score, yt, yp)
    if name == "mae":
        return float(np.mean(np.abs(yt - yp)))
    if name == "rmse":
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    if name == "r2":
        sse = float(np.sum((yt - yp) ** 2))
        centered = yt - float(np.mean(yt))
        sst = float(np.dot(centered, centered))
        if sst == 0.0:
            return 1.0 if sse == 0.0 else 0.0
        return 1.0 - sse / sst
    if name == "smape":
        smape_denom = np.abs(yt) + np.abs(yp)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(smape_denom == 0.0, 0.0, 2.0 * np.abs(yt - yp) / smape_denom)
        return float(np.mean(terms))
    raise ValueError(f"unknown metric {name!r}")


def score(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5
) -> float:
    """Higher-is-better version of ``compute``."""
    value = compute(name, y_true, y_pred, threshold=threshold)
    return -value if name in LOWER_IS_BETTER else value


def table(
    task: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    names = CLASSIFICATION if task == "classification" else REGRESSION
    return {
        model: {m: compute(m, y_true, p, threshold=threshold) for m in names}
        for model, p in predictions.items()
    }


def optimal_threshold(
    metric: str, y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """Select the classification threshold maximizing ``metric`` on non-test rows."""
    if metric not in THRESHOLDED_METRICS:
        return 0.5
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_score, dtype=np.float64)
    ok = ~(np.isnan(yt) | np.isnan(ys))
    yt, ys = yt[ok], ys[ok]
    if len(yt) < 10 or np.ptp(ys) == 0.0:
        return 0.5
    uniq = np.unique(ys)
    if len(uniq) <= 2:
        return 0.5
    candidates = uniq if len(uniq) <= 100 else np.quantile(ys, np.linspace(0.02, 0.98, 97))
    best_t, best_val = 0.5, score(metric, yt, ys, threshold=0.5)
    for t in candidates:
        val = score(metric, yt, ys, threshold=float(t))
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_t = float(t)
    return best_t


def _undefined_if_one_class(
    fn: Callable[[np.ndarray, np.ndarray], Any], y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_pred))


# --- prefix-sum vectorized & panel-aware circular block bootstrap -------------


def _prefix(arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(arr) + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(arr, dtype=np.float64, out=out[1:])
    return out


def _block_sums(
    cs: np.ndarray, starts: np.ndarray, block: int, rem: int
) -> np.ndarray:
    """Evaluate sum over ``n_blocks`` contiguous blocks in O(1) per block."""
    if starts.shape[1] == 1:
        s0 = starts[:, 0]
        return np.asarray(cs[s0 + rem] - cs[s0], dtype=np.float64)
    full = starts[:, :-1]
    last = starts[:, -1]
    total = np.sum(cs[full + block] - cs[full], axis=1)
    return np.asarray(total + (cs[last + rem] - cs[last]), dtype=np.float64)


def _sufficient_arrays(
    metric: str,
    y_true: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Per-observation additive sufficient statistics for prefix-sum evaluation."""
    if metric == "mae":
        return {"em": np.abs(y_true - model), "eb": np.abs(y_true - baseline)}
    if metric == "rmse":
        return {"em": (y_true - model) ** 2, "eb": (y_true - baseline) ** 2}
    if metric == "r2":
        return {
            "em": (y_true - model) ** 2,
            "eb": (y_true - baseline) ** 2,
            "y": y_true,
            "y2": y_true ** 2,
        }
    if metric == "smape":
        dm = np.abs(y_true) + np.abs(model)
        db = np.abs(y_true) + np.abs(baseline)
        with np.errstate(divide="ignore", invalid="ignore"):
            em = np.where(dm == 0.0, 0.0, 2.0 * np.abs(y_true - model) / dm)
            eb = np.where(db == 0.0, 0.0, 2.0 * np.abs(y_true - baseline) / db)
        return {"em": em, "eb": eb}
    if metric == "brier":
        return {"em": (y_true - model) ** 2, "eb": (y_true - baseline) ** 2}
    if metric == "accuracy":
        yt = y_true >= 0.5
        return {
            "cm": (yt == (model >= threshold)).astype(np.float64),
            "cb": (yt == (baseline >= threshold)).astype(np.float64),
        }
    # Confusion-matrix metrics: f1, mcc, balanced_accuracy
    yt = y_true >= 0.5
    pm = model >= threshold
    pb = baseline >= threshold
    return {
        "tp_m": (yt & pm).astype(np.float64),
        "fp_m": ((~yt) & pm).astype(np.float64),
        "fn_m": (yt & (~pm)).astype(np.float64),
        "tn_m": ((~yt) & (~pm)).astype(np.float64),
        "tp_b": (yt & pb).astype(np.float64),
        "fp_b": ((~yt) & pb).astype(np.float64),
        "fn_b": (yt & (~pb)).astype(np.float64),
        "tn_b": ((~yt) & (~pb)).astype(np.float64),
    }


def _eval_from_sums(metric: str, sums: dict[str, np.ndarray], denom_n: np.ndarray) -> np.ndarray:
    """Compute score(model) - score(baseline) vectorized over a chunk of bootstrap draws."""
    if metric in ("mae", "smape", "brier"):
        return np.asarray((sums["eb"] - sums["em"]) / denom_n, dtype=np.float64)
    if metric == "rmse":
        return np.asarray(np.sqrt(sums["eb"] / denom_n) - np.sqrt(sums["em"] / denom_n), dtype=np.float64)
    if metric == "r2":
        sst = sums["y2"] - (sums["y"] ** 2) / denom_n
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_m = np.where(sst == 0.0, np.where(sums["em"] == 0.0, 1.0, 0.0), 1.0 - sums["em"] / sst)
            r2_b = np.where(sst == 0.0, np.where(sums["eb"] == 0.0, 1.0, 0.0), 1.0 - sums["eb"] / sst)
        return np.asarray(r2_m - r2_b, dtype=np.float64)
    if metric == "accuracy":
        return np.asarray((sums["cm"] - sums["cb"]) / denom_n, dtype=np.float64)
    if metric == "f1":
        dm = 2.0 * sums["tp_m"] + sums["fp_m"] + sums["fn_m"]
        db = 2.0 * sums["tp_b"] + sums["fp_b"] + sums["fn_b"]
        with np.errstate(divide="ignore", invalid="ignore"):
            fm = np.where(dm > 0.0, 2.0 * sums["tp_m"] / dm, 0.0)
            fb = np.where(db > 0.0, 2.0 * sums["tp_b"] / db, 0.0)
        return np.asarray(fm - fb, dtype=np.float64)
    if metric == "balanced_accuracy":
        def _ba(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
            pos = tp + fn
            neg = tn + fp
            with np.errstate(divide="ignore", invalid="ignore"):
                tpr = np.where(pos > 0.0, tp / pos, 0.0)
                tnr = np.where(neg > 0.0, tn / neg, 0.0)
            both = (pos > 0.0) & (neg > 0.0)
            only_pos = (pos > 0.0) & (neg == 0.0)
            res = np.where(both, 0.5 * (tpr + tnr), np.where(only_pos, tpr, tnr))
            return np.asarray(res, dtype=np.float64)

        return np.asarray(
            _ba(sums["tp_m"], sums["fp_m"], sums["fn_m"], sums["tn_m"])
            - _ba(sums["tp_b"], sums["fp_b"], sums["fn_b"], sums["tn_b"]),
            dtype=np.float64,
        )
    if metric == "mcc":
        def _mcc(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
            d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.asarray(np.where(d > 0.0, (tp * tn - fp * fn) / d, 0.0), dtype=np.float64)

        return np.asarray(
            _mcc(sums["tp_m"], sums["fp_m"], sums["fn_m"], sums["tn_m"])
            - _mcc(sums["tp_b"], sums["fp_b"], sums["fn_b"], sums["tn_b"]),
            dtype=np.float64,
        )
    raise ValueError(f"unsupported prefix metric {metric!r}")


def _pad_circular(arr: np.ndarray, pad: int) -> np.ndarray:
    """Wrap-pad a 1D series by ``pad`` elements for bias-free Circular Block Bootstrap."""
    if pad <= 0 or len(arr) == 0:
        return arr
    idx = np.arange(len(arr) + pad) % len(arr)
    return arr[idx]


def _group_circular_layout(groups: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Build flat gather indices and exact-uniform valid starts for within-group circular padding.

    Each contiguous group of length L_g is wrap-padded by ``pad`` entries from itself so:
      1. Blocks never cross group boundaries.
      2. Every original observation has exact uniform sampling probability 1/n (zero edge bias).
    """
    n = len(groups)
    changes = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [n]))
    gather_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    offset = 0
    for s, e in zip(boundaries[:-1], boundaries[1:], strict=True):
        length = int(e - s)
        idx = s + (np.arange(length + pad, dtype=np.int64) % length)
        gather_parts.append(idx)
        start_parts.append(np.arange(offset, offset + length, dtype=np.int64))
        offset += length + pad
    return np.concatenate(gather_parts), np.concatenate(start_parts)


def _is_aligned_panel(
    groups: np.ndarray | None, times: np.ndarray | None
) -> tuple[bool, np.ndarray | None, int]:
    """Detect whether multi-group observations share aligned time indices."""
    if groups is None or times is None or len(groups) < 2:
        return False, None, 0
    if np.all(groups == groups[0]):
        return False, None, 0
    if np.issubdtype(times.dtype, np.integer):
        tmin = int(np.min(times))
        tmax = int(np.max(times))
        span = tmax - tmin + 1
        if 10 <= span <= len(times) and (len(groups) / span) >= 1.5:
            inv = (times - tmin).astype(np.int64, copy=False)
            counts = np.bincount(inv, minlength=span)
            n_present = int(np.count_nonzero(counts))
            if n_present == span:
                return True, inv, span
    _, inv = np.unique(times, return_inverse=True)
    n_times = int(np.max(inv)) + 1 if len(inv) > 0 else 0
    if n_times >= 10 and (len(groups) / n_times) >= 1.5:
        return True, inv, n_times
    return False, None, 0


def estimate_block_length(x: np.ndarray, *, max_block: int | None = None) -> int:
    """Politis-White (2004) / Patton-Politis-White (2009) optimal CBB block length.

    Estimates the correlation scale directly from the sample autocovariances of ``x``
    using an O(T log T) FFT and flat-top trapezoid lag window.
    """
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    cap = max_block if max_block is not None else max(1, n // 2)
    if n < 20:
        return max(1, min(10, cap))
    xc = arr - float(np.mean(arr))
    var = float(np.dot(xc, xc)) / n
    if var <= 0.0:
        return max(1, min(10, cap))

    # O(n log n) sample autocovariance via FFT
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, n_fft)
    acov = np.fft.irfft(f * np.conj(f), n_fft)[:n] / n
    acf = acov / var

    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    thresh = 2.0 * float(np.sqrt(np.log10(n) / n))
    max_m = min(n // 2, max(10, int(np.ceil(np.sqrt(n))) + kn))
    m_hat = max_m
    for lag in range(1, max_m - kn + 1):
        if np.all(np.abs(acf[lag : lag + kn]) < thresh):
            m_hat = max(1, lag)
            break
    bw = min(2 * m_hat, n // 2)
    if bw < 1:
        return 1

    lags = np.arange(1, bw + 1, dtype=np.float64)
    u = lags / float(bw)
    weights = np.where(u <= 0.5, 1.0, 2.0 * (1.0 - u))
    g_hat = 2.0 * float(np.sum(weights * lags * acov[1 : bw + 1]))
    g0_hat = float(acov[0] + 2.0 * np.sum(weights * acov[1 : bw + 1]))
    if g0_hat <= 0.0 or g_hat == 0.0:
        return max(1, min(int(round(np.sqrt(n))), cap))

    d_cbb = (4.0 / 3.0) * (g0_hat ** 2)
    b_opt = ((2.0 * (g_hat ** 2)) / d_cbb) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    return max(1, min(int(np.ceil(b_opt)), cap))


def default_block_length(
    n_total: int,
    *,
    groups: np.ndarray | None = None,
    times: np.ndarray | None = None,
) -> int:
    """Choose a sensible block length based on effective series length (never n_groups * T)."""
    is_panel, _, n_times = _is_aligned_panel(groups, times)
    if is_panel:
        t_eff = n_times
    elif groups is not None and len(groups) > 0:
        changes = np.flatnonzero(groups[1:] != groups[:-1]) + 1
        boundaries = np.concatenate(([0], changes, [len(groups)]))
        lengths = np.diff(boundaries)
        t_eff = int(np.median(lengths))
    else:
        t_eff = n_total
    raw = max(10, int(round(np.sqrt(max(1, t_eff)))))
    return max(1, min(raw, max(1, t_eff // 2))) if t_eff >= 20 else max(1, min(raw, t_eff))


def block_bootstrap_diff(
    metric: str,
    y_true: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    block: int,
    n_boot: int,
    seed: int,
    *,
    groups: np.ndarray | None = None,
    times: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Moving-block bootstrap 95% CI of ``score(model) - score(baseline)``.

    Uses O(1)-per-block prefix-sum vectorization for all additive and confusion-matrix
    metrics. When ``groups`` (and optionally ``times``) are provided:
      - Time-aligned panels resample contiguous blocks of timestamps across all series
        simultaneously, preserving both autocorrelation and cross-sectional dependence.
      - Unaligned multi-series data restricts block starts strictly within each group so
        blocks never cross series boundaries.
    """
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=np.float64)
    pm = np.asarray(model, dtype=np.float64)
    pb = np.asarray(baseline, dtype=np.float64)
    n = len(yt)
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan")

    is_panel, time_inv, n_times = _is_aligned_panel(groups, times)

    if metric in _PREFIX_METRICS:
        raw_stats = _sufficient_arrays(metric, yt, pm, pb, threshold)

        if is_panel and time_inv is not None:
            # Aggregate sufficient statistics by timestamp -> length T circularly padded array
            eff_n = n_times
            b_len = max(1, min(block, eff_n))
            pad = b_len - 1
            agg_stats = {
                k: np.bincount(time_inv, weights=v, minlength=n_times)
                for k, v in raw_stats.items()
            }
            counts = np.bincount(time_inv, minlength=n_times).astype(np.float64)
            prefixes = {k: _prefix(_pad_circular(v, pad)) for k, v in agg_stats.items()}
            cnt_prefix = _prefix(_pad_circular(counts, pad))
            n_blocks = int(np.ceil(eff_n / b_len))
            rem = eff_n - (n_blocks - 1) * b_len

            diffs = np.empty(n_boot, dtype=np.float64)
            chunk = 256
            for i0 in range(0, n_boot, chunk):
                i1 = min(n_boot, i0 + chunk)
                sz = i1 - i0
                starts = rng.integers(0, eff_n, size=(sz, n_blocks))
                denom_n = _block_sums(cnt_prefix, starts, b_len, rem)
                sums = {k: _block_sums(cs, starts, b_len, rem) for k, cs in prefixes.items()}
                diffs[i0:i1] = _eval_from_sums(metric, sums, denom_n)
        else:
            b_len = max(1, min(block, n))
            pad = b_len - 1
            n_blocks = int(np.ceil(n / b_len))
            rem = n - (n_blocks - 1) * b_len
            if groups is not None:
                gather_idx, valid_starts = _group_circular_layout(groups, pad)
                prefixes = {k: _prefix(v[gather_idx]) for k, v in raw_stats.items()}
            else:
                valid_starts = None
                prefixes = {k: _prefix(_pad_circular(v, pad)) for k, v in raw_stats.items()}

            diffs = np.empty(n_boot, dtype=np.float64)
            chunk = 256
            denom_n = np.full(chunk, float(n), dtype=np.float64)
            for i0 in range(0, n_boot, chunk):
                i1 = min(n_boot, i0 + chunk)
                sz = i1 - i0
                if valid_starts is not None:
                    starts = valid_starts[rng.integers(0, len(valid_starts), size=(sz, n_blocks))]
                else:
                    starts = rng.integers(0, n, size=(sz, n_blocks))
                sums = {k: _block_sums(cs, starts, b_len, rem) for k, cs in prefixes.items()}
                diffs[i0:i1] = _eval_from_sums(metric, sums, denom_n[:sz])
    else:
        # Rank-based metrics (roc_auc, pr_auc) require resampled observation vectors
        if is_panel and time_inv is not None:
            order = np.argsort(time_inv, kind="stable")
            sorted_inv = time_inv[order]
            splits = np.flatnonzero(np.diff(sorted_inv)) + 1
            time_buckets = np.split(order, splits)
            b_len = max(1, min(block, n_times))
            n_blocks = int(np.ceil(n_times / b_len))
            offsets = np.arange(b_len)
            diffs = np.empty(n_boot, dtype=np.float64)
            for i in range(n_boot):
                t_starts = rng.integers(0, n_times, size=n_blocks)
                t_idx = ((t_starts[:, None] + offsets) % n_times).ravel()[:n_times]
                idx = np.concatenate([time_buckets[t] for t in t_idx])
                ys = yt[idx]
                diffs[i] = score(metric, ys, pm[idx], threshold=threshold) - score(
                    metric, ys, pb[idx], threshold=threshold
                )
        else:
            b_len = max(1, min(block, n))
            n_blocks = int(np.ceil(n / b_len))
            offsets = np.arange(b_len)
            if groups is not None:
                gather_idx, valid_starts = _group_circular_layout(groups, b_len - 1)
            else:
                gather_idx, valid_starts = None, None
            diffs = np.empty(n_boot, dtype=np.float64)
            for i in range(n_boot):
                if gather_idx is not None and valid_starts is not None:
                    starts = valid_starts[rng.integers(0, len(valid_starts), size=n_blocks)]
                    idx = gather_idx[(starts[:, None] + offsets).ravel()[:n]]
                else:
                    starts = rng.integers(0, n, size=n_blocks)
                    idx = ((starts[:, None] + offsets) % n).ravel()[:n]
                ys = yt[idx]
                diffs[i] = score(metric, ys, pm[idx], threshold=threshold) - score(
                    metric, ys, pb[idx], threshold=threshold
                )

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --- Diebold-Mariano HAC test & multiple-testing correction -------------------


def _step_loss_differential(
    metric: str,
    y_true: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray | None:
    """Return d_t = L(y_t, baseline_t) - L(y_t, model_t) so d_t > 0 favors model."""
    if metric == "mae":
        return np.asarray(np.abs(y_true - baseline) - np.abs(y_true - model), dtype=np.float64)
    if metric in ("rmse", "brier"):
        return np.asarray((y_true - baseline) ** 2 - (y_true - model) ** 2, dtype=np.float64)
    if metric == "smape":
        db = np.abs(y_true) + np.abs(baseline)
        dm = np.abs(y_true) + np.abs(model)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.where(db == 0.0, 0.0, 2.0 * np.abs(y_true - baseline) / db)
            lm = np.where(dm == 0.0, 0.0, 2.0 * np.abs(y_true - model) / dm)
        return np.asarray(lb - lm, dtype=np.float64)
    if metric == "accuracy":
        yt = y_true >= 0.5
        return np.asarray(
            (yt == (model >= threshold)).astype(np.float64)
            - (yt == (baseline >= threshold)).astype(np.float64),
            dtype=np.float64,
        )
    return None


def diebold_mariano(
    metric: str,
    y_true: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    horizon: int = 1,
    groups: np.ndarray | None = None,
    times: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Diebold-Mariano test with Newey-West Bartlett HAC variance and HLN adjustment.

    Positive statistic means ``model`` beats ``baseline``. Returns ``(stat, two_sided_p)``
    or ``(nan, nan)`` for non-additive metrics or degenerate series.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    pm = np.asarray(model, dtype=np.float64)
    pb = np.asarray(baseline, dtype=np.float64)
    d = _step_loss_differential(metric, yt, pm, pb, threshold=threshold)
    if d is None or len(d) < 5:
        return float("nan"), float("nan")

    is_panel, time_inv, n_times = _is_aligned_panel(groups, times)
    if is_panel and time_inv is not None:
        counts = np.bincount(time_inv, minlength=n_times).astype(np.float64)
        sums = np.bincount(time_inv, weights=d, minlength=n_times)
        d = sums / np.maximum(counts, 1.0)

    n = len(d)
    if n < 5:
        return float("nan"), float("nan")

    mean_d = float(np.mean(d))
    dc = d - mean_d
    gamma0 = float(np.dot(dc, dc)) / n
    if gamma0 == 0.0:
        return (0.0, 1.0) if mean_d == 0.0 else (float(np.sign(mean_d) * np.inf), 0.0)

    max_lag = min(n - 1, max(horizon - 1, int(round(n ** (1.0 / 3.0)))))
    lrv = gamma0
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)
        gamma_k = float(np.dot(dc[:-k], dc[k:])) / n
        lrv += 2.0 * weight * gamma_k

    if lrv <= 0.0:
        lrv = gamma0

    dm = mean_d / float(np.sqrt(lrv / n))
    # Harvey-Leybourne-Newbold finite-sample correction
    h_eff = min(max(1, horizon), max(1, n // 2))
    hln_factor = float(np.sqrt(max(1e-12, (n + 1.0 - 2.0 * h_eff + h_eff * (h_eff - 1.0) / n) / n)))
    stat = dm * hln_factor
    p_val = float(2.0 * sp_stats.t.sf(abs(stat), df=n - 1))
    return stat, p_val


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (NaNs preserved)."""
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    valid = p[ok]
    m = len(valid)
    if m == 0:
        return q
    order = np.argsort(valid)
    ranked = valid[order]
    adj = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    q[ok] = out
    return q


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs preserved)."""
    p = np.asarray(p_values, dtype=np.float64)
    adj_all = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    valid = p[ok]
    m = len(valid)
    if m == 0:
        return adj_all
    order = np.argsort(valid)
    ranked = valid[order]
    adj = ranked * np.arange(m, 0, -1, dtype=np.float64)
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    adj_all[ok] = out
    return adj_all
