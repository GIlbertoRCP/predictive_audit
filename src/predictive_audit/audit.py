"""Run every check on one forecasting result and collect them into a report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import baselines, metrics

SEVERITY = {"INFO": 0, "PASS": 1, "WARN": 2, "FAIL": 3}


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | INFO
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    task: str
    target: str
    horizon: int
    metric: str
    model: str | None
    season: int | None
    n_eval: int
    prevalence: float | None
    best_baseline: str
    table: dict[str, dict[str, float]]
    checks: list[Check]

    @property
    def verdict(self) -> str:
        return max((c.status for c in self.checks), key=SEVERITY.__getitem__, default="INFO")

    def to_dict(self) -> dict:
        return {**asdict(self), "verdict": self.verdict}


def audit(
    df: pd.DataFrame,
    target: str,
    *,
    task: str = "classification",
    horizon: int = 1,
    pred: str | None = None,
    model_name: str | None = None,
    groups: str | list[str] | None = None,
    time: str | None = None,
    split: str | None = None,
    features: str | list[str] | None = None,
    season: int | str | None = None,
    window: int | None = None,
    metric: str | None = None,
    reported: float | None = None,
    n_boot: int = 500,
    block: int | None = None,
    seed: int = 0,
) -> AuditReport:
    """Audit a forecast.

    ``target`` is the value being forecast, already aligned so that row t holds
    y[t + horizon]. Rows are ordered by ``time`` (or kept in file order) within each
    group in ``groups``. ``pred`` is the model's prediction column (a score or label for
    classification). ``split`` marks rows ``test`` (scored), ``train`` (checked for temporal
    overlap with test) or anything else (history only, e.g. warm-up rows).
    ``reported`` is a published score on ``metric`` to check against trivial baselines.
    ``season`` adds a seasonal-naive baseline; ``"auto"`` estimates the period from non-test rows.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    groups, features = _as_list(groups), _as_list(features)
    _require_columns(df, [target, pred, time, split, *groups, *features])

    metric = metric or ("f1" if task == "classification" else "mae")
    allowed = metrics.CLASSIFICATION if task == "classification" else metrics.REGRESSION
    if metric not in allowed:
        raise ValueError(f"metric {metric!r} is not valid for {task}; choose from {allowed}")

    sort_cols = groups + ([time] if time else [])
    data = df.sort_values(sort_cols, kind="stable") if sort_cols else df
    gkey = data.groupby(groups, sort=False).ngroup() if groups else None
    y = data[target].astype(float)
    if task == "classification" and not set(y.dropna().unique()) <= {0.0, 1.0}:
        raise ValueError("classification target must be binary 0/1")

    if split:
        labels = data[split].astype(str).str.lower()
        test_mask = labels.eq("test")
        fit_mask = ~test_mask  # any non-test row is history the baselines may fit on
        if not test_mask.any():
            raise ValueError(f"no rows with {split} == 'test'")
    else:
        test_mask = fit_mask = pd.Series(True, index=data.index)

    preds = {}
    if pred:
        model_name = model_name or pred
        preds[model_name] = data[pred].astype(float)
    if season == "auto":
        season = baselines.estimate_season(y, gkey, fit_mask)
    baseline_preds = baselines.all_baselines(y, gkey, horizon, task, fit_mask, season, window)
    preds.update(baseline_preds)

    eval_mask = test_mask & y.notna()
    for p in preds.values():
        eval_mask &= p.notna()
    n_eval = int(eval_mask.sum())
    if n_eval < 10:
        raise ValueError(f"only {n_eval} rows can be scored; need test rows with enough history")

    y_eval = y[eval_mask].to_numpy()
    p_eval = {k: v[eval_mask].to_numpy() for k, v in preds.items()}
    table = metrics.table(task, y_eval, p_eval)
    best = max(baseline_preds, key=lambda b: _nan_low(metrics.score(metric, y_eval, p_eval[b])))

    checks = [_label_memory(task, metric, y_eval, p_eval["persistence"], horizon)]
    if pred:
        checks.append(
            _beats_baselines(metric, y_eval, p_eval[model_name], p_eval[best], best, block, n_boot, seed)
        )
    if reported is not None:
        checks.append(_reported_score(metric, reported, table, baseline_preds))
    if split:
        checks.append(_temporal_split(data, split, gkey))
    if features:
        checks.append(_feature_leakage(data, y, gkey, features, horizon))

    return AuditReport(
        task=task,
        target=target,
        horizon=horizon,
        metric=metric,
        model=model_name,
        season=season,
        n_eval=n_eval,
        prevalence=float(y_eval.mean()) if task == "classification" else None,
        best_baseline=best,
        table=table,
        checks=checks,
    )


# --- checks -------------------------------------------------------------------


def _label_memory(task, metric, y_true, persist, horizon) -> Check:
    """How much of the 'forecast' is the target simply remembering itself?"""
    autocorr = _corr(y_true, persist)
    persist_value = metrics.compute(metric, y_true, persist)
    details = {"lag_autocorrelation": autocorr, f"persistence_{metric}": persist_value}
    if task == "classification":
        details["agreement_with_current_state"] = float(np.mean(y_true == persist))
    bounded = metric not in metrics.LOWER_IS_BETTER
    high = (bounded and persist_value >= 0.9) or (np.isfinite(autocorr) and autocorr >= 0.95)
    if high:
        return Check(
            "label memory",
            "WARN",
            f"The target at t+{horizon} is mostly a copy of the target at t "
            f"(lag-{horizon} autocorrelation {autocorr:.3f}, persistence {metric} {persist_value:.3f}). "
            "Any model will post a high score here; only the margin over persistence means anything.",
            details,
        )
    return Check(
        "label memory",
        "INFO",
        f"Lag-{horizon} autocorrelation of the target is {autocorr:.3f}; "
        f"persistence scores {metric} {persist_value:.3f}.",
        details,
    )


def _beats_baselines(metric, y_true, model, baseline, baseline_name, block, n_boot, seed) -> Check:
    diff = metrics.score(metric, y_true, model) - metrics.score(metric, y_true, baseline)
    block = block or max(10, int(round(np.sqrt(len(y_true)))))
    lo, hi = metrics.block_bootstrap_diff(metric, y_true, model, baseline, block, n_boot, seed)
    details = {"baseline": baseline_name, "difference": diff, "ci95": [lo, hi], "block_length": block}
    what = f"{metric} reduction vs {baseline_name}" if metric in metrics.LOWER_IS_BETTER else f"{metric} gain over {baseline_name}"
    margin = f"{what} = {diff:+.4f} (95% block-bootstrap CI {lo:+.4f} to {hi:+.4f})"
    if not np.isfinite(diff) or diff <= 0:
        return Check("beats trivial baselines", "FAIL", f"Does not beat the best trivial baseline: {margin}.", details)
    if not np.isfinite(lo) or lo <= 0:
        return Check(
            "beats trivial baselines",
            "WARN",
            f"Ahead of the best trivial baseline, but the margin is within noise: {margin}.",
            details,
        )
    return Check("beats trivial baselines", "PASS", f"Beats the best trivial baseline: {margin}.", details)


def _reported_score(metric, reported, table, baseline_names) -> Check:
    lower = metric in metrics.LOWER_IS_BETTER
    name = max(baseline_names, key=lambda b: _nan_low(-table[b][metric] if lower else table[b][metric]))
    value = table[name][metric]
    gap = (value - reported) if lower else (reported - value)  # > 0 means the claim is ahead
    details = {"reported": reported, "baseline": name, "baseline_value": value, "gap": gap}
    text = f"Reported {metric} {reported:.4g}; {name} alone gets {value:.4g} on the same rows"
    tolerance = 0.05 * abs(value) if lower else 0.02
    if not np.isfinite(value):
        return Check("reported score", "INFO", f"{metric} is undefined on these rows.", details)
    if gap <= 0:
        return Check("reported score", "FAIL", f"{text}. The claim is not evidence of forecasting skill.", details)
    if gap < tolerance:
        return Check("reported score", "WARN", f"{text}. The claimed improvement is marginal.", details)
    return Check("reported score", "PASS", f"{text}.", details)


def _temporal_split(data, split, gkey) -> Check:
    labels = data[split].astype(str).str.lower()
    keys = gkey if gkey is not None else pd.Series(0, index=data.index)
    clock = keys.groupby(keys).cumcount()  # data is already in time order within each group
    frame = pd.DataFrame({"g": keys, "t": clock, "test": labels.eq("test"), "train": labels.eq("train")})
    if not frame.train.any():
        return Check("temporal split", "INFO", "No rows are labelled 'train', so there is no overlap to check.")

    last_train = frame[frame.train].groupby("g")["t"].max()
    test = frame[frame.test]
    before = test["t"] < test["g"].map(last_train)  # NaN map (no train rows) compares False
    share_before = float(before.mean())
    runs = frame.groupby("g")["test"].apply(lambda s: int((s & ~s.shift(fill_value=False)).sum()))
    mean_runs = float(runs[runs > 0].mean())
    details = {"test_rows_before_last_train_row": share_before, "mean_test_segments_per_group": mean_runs}

    if share_before == 0:
        return Check("temporal split", "PASS", "Every test row comes after the training data in time.", details)
    if mean_runs <= 3:
        return Check(
            "temporal split",
            "WARN",
            f"{share_before:.0%} of test rows precede training rows, in a few contiguous blocks. "
            "Fine for blocked cross-validation with a gap; otherwise information leaks backward in time.",
            details,
        )
    return Check(
        "temporal split",
        "FAIL",
        f"Test rows are interleaved with training rows ({share_before:.0%} precede the last training row, "
        f"~{mean_runs:.0f} separate test segments per series). This is a shuffled split: with overlapping "
        "windows or autocorrelated rows, test examples are near-copies of training examples.",
        details,
    )


def _feature_leakage(data, y, gkey, features, horizon) -> Check:
    """A feature that tracks y[t+j] for 0 < j <= h is information from the future."""
    leaks, strongest = [], {}
    for f in features:
        x = data[f].astype(float)
        best = 0.0
        for k in range(horizon):  # lag_target(k) for k < h is y[t + h - k], still unobserved at t
            r = abs(_corr(x.to_numpy(), baselines.lag_target(y, gkey, k).to_numpy()))
            if np.isfinite(r) and r > best:
                best = r
            if np.isfinite(r) and r >= 0.999:
                leaks.append(f"{f} ~ y[t+{horizon - k}] (|r|={r:.4f})")
        strongest[f] = best
    details = {"max_abs_corr_with_future_target": strongest}
    if leaks:
        return Check("feature leakage", "FAIL", "Features copy future target values: " + "; ".join(leaks) + ".", details)
    return Check("feature leakage", "PASS", "No feature is a near-copy of a future target value.", details)


# --- helpers ------------------------------------------------------------------


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _nan_low(x: float) -> float:
    return -np.inf if np.isnan(x) else x


def _as_list(x) -> list[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _require_columns(df, cols) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}")
