"""Run every check on one forecasting result (or a batch of series) and collect into a report."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from . import _backend, baselines, metrics
from ._backend import reference_module

SCHEMA_VERSION = "1.0"
SEVERITY: dict[str, int] = {"INFO": 0, "PASS": 1, "WARN": 2, "FAIL": 3}

CheckFn = Callable[[pd.DataFrame, dict[str, np.ndarray], np.ndarray], "Check"]


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | INFO
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


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
    schema_version: str = SCHEMA_VERSION
    threshold: float | None = None
    mase: float | None = None

    @property
    def verdict(self) -> str:
        return max((c.status for c in self.checks), key=SEVERITY.__getitem__, default="INFO")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "verdict": self.verdict}


@dataclass
class BatchAuditReport:
    """Aggregated per-group audit with Benjamini-Hochberg FDR multiple-testing control."""

    group_col: str
    metric: str
    model: str | None
    reports: dict[str, AuditReport]
    rows: list[dict[str, Any]]
    schema_version: str = SCHEMA_VERSION

    @property
    def tally(self) -> dict[str, int]:
        counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
        for r in self.rows:
            status = str(r["status"])
            counts[status] = counts.get(status, 0) + 1
        return counts

    @property
    def n_fdr_significant(self) -> int:
        return sum(
            1
            for r in self.rows
            if r.get("diff", 0.0) > 0.0
            and isinstance(r.get("fdr_q_value"), (int, float))
            and np.isfinite(r["fdr_q_value"])
            and r["fdr_q_value"] < 0.05
        )

    @property
    def verdict(self) -> str:
        active = [str(r["status"]) for r in self.rows if r["status"] != "INFO"]
        if not active:
            return "INFO"
        return max(active, key=SEVERITY.__getitem__)

    def summary_frame(self) -> pd.DataFrame:
        return cast(pd.DataFrame, pd.DataFrame(self.rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "group_col": self.group_col,
            "metric": self.metric,
            "model": self.model,
            "verdict": self.verdict,
            "tally": self.tally,
            "n_groups": len(self.rows),
            "n_fdr_significant": self.n_fdr_significant,
            "rows": self.rows,
        }


def audit(
    df: pd.DataFrame,
    target: str,
    *,
    task: str = "classification",
    horizon: int = 1,
    pred: str | None = None,
    model_name: str | None = None,
    groups: str | Sequence[str] | None = None,
    time: str | None = None,
    split: str | None = None,
    features: str | Sequence[str] | None = None,
    season: int | str | None = None,
    window: int | None = None,
    include_ewma: bool = False,
    ar_lags: int | None = None,
    metric: str | None = None,
    reported: float | None = None,
    threshold: float | Literal["auto"] = 0.5,
    n_boot: int = 500,
    block: int | Literal["auto"] | None = None,
    seed: int = 0,
    extra_baselines: dict[str, pd.Series] | None = None,
    extra_checks: Sequence[CheckFn] | None = None,
) -> AuditReport:
    """Audit a forecast.

    ``target`` is the value being forecast, already aligned so that row t holds
    y[t + horizon]. Rows are ordered by ``time`` (or kept in file order) within each
    group in ``groups``. ``pred`` is the model's prediction column (a score or label for
    classification). ``split`` marks rows ``test`` (scored), ``train`` (checked for temporal
    overlap, embargo gaps, and global scaler leakage) or anything else (history only).
    ``reported`` is a published score on ``metric`` to check against trivial baselines.
    ``season`` adds a seasonal-naive baseline; ``"auto"`` estimates the period from non-test rows.
    ``ar_lags`` adds a causal Ridge AR(p) baseline fit strictly on non-test rows.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    group_cols, feature_cols = _as_list(groups), _as_list(features)
    _require_columns(df, [target, pred, time, split, *group_cols, *feature_cols])

    metric = metric or ("f1" if task == "classification" else "mae")
    allowed = metrics.CLASSIFICATION if task == "classification" else metrics.REGRESSION
    if metric not in allowed:
        raise ValueError(f"metric {metric!r} is not valid for {task}; choose from {allowed}")

    prepared = _native_prepare(
        df, target, task=task, horizon=horizon, pred=pred, model_name=model_name, group_cols=group_cols,
        time=time, split=split, feature_cols=feature_cols, season=season, window=window,
        include_ewma=include_ewma, ar_lags=ar_lags, metric=metric, reported=reported, threshold=threshold,
        n_boot=n_boot, block=block, seed=seed, extra_baselines=extra_baselines,
        want_eval_arrays=bool(extra_checks),
    )
    if prepared is not None and _backend.native is not None:
        return _report_from_native(
            _backend.native.audit(*prepared), df, extra_checks, metric,
            horizon=horizon, model_name=model_name, pred=pred, reported=reported,
        )

    sort_cols = group_cols + ([time] if time else [])
    data = df.sort_values(sort_cols, kind="stable") if sort_cols else df
    gkey = data.groupby(group_cols, sort=False).ngroup() if group_cols else None
    y = data[target].astype(float)
    if task == "classification" and not set(y.dropna().unique()) <= {0.0, 1.0}:
        raise ValueError("classification target must be binary 0/1")

    if split:
        labels = data[split].astype(str).str.lower()
        test_mask = labels.eq("test")
        train_mask = labels.eq("train")
        fit_mask = ~test_mask
        if not test_mask.any():
            raise ValueError(f"no rows with {split} == 'test'")
    else:
        test_mask = fit_mask = pd.Series(True, index=data.index)
        train_mask = pd.Series(False, index=data.index)

    preds: dict[str, pd.Series] = {}
    if pred:
        model_name = model_name or pred
        preds[model_name] = data[pred].astype(float)
    resolved_season: int | None
    if season == "auto":
        resolved_season = baselines.estimate_season(y, gkey, fit_mask)
    elif isinstance(season, int):
        resolved_season = season
    elif season is None:
        resolved_season = None
    else:
        raise ValueError(f"season must be an int, 'auto', or None; got {season!r}")

    baseline_preds = baselines.all_baselines(
        y,
        gkey,
        horizon,
        task,
        fit_mask,
        resolved_season,
        window,
        include_ewma=include_ewma,
        ar_lags=ar_lags,
    )
    if extra_baselines:
        for b_name, b_series in extra_baselines.items():
            baseline_preds[b_name] = b_series.reindex(data.index).astype(float)
    preds.update(baseline_preds)

    # Resolve classification threshold (if "auto", fit strictly on non-test rows)
    resolved_threshold = 0.5
    if task == "classification":
        if threshold == "auto":
            if pred and model_name is not None and fit_mask.any():
                resolved_threshold = metrics.optimal_threshold(
                    metric, y[fit_mask].to_numpy(), preds[model_name][fit_mask].to_numpy()
                )
        else:
            resolved_threshold = float(threshold)

    eval_mask = test_mask & y.notna()
    for p in preds.values():
        eval_mask &= p.notna()
    n_eval = int(eval_mask.sum())
    if n_eval < 10:
        raise ValueError(f"only {n_eval} rows can be scored; need test rows with enough history")

    y_eval = y[eval_mask].to_numpy()
    p_eval = {k: v[eval_mask].to_numpy() for k, v in preds.items()}
    g_eval = gkey[eval_mask].to_numpy() if gkey is not None else None
    t_eval = data.loc[eval_mask, time].to_numpy() if time else None

    table = metrics.table(task, y_eval, p_eval, threshold=resolved_threshold)
    best = max(
        baseline_preds,
        key=lambda b: _nan_low(metrics.score(metric, y_eval, p_eval[b], threshold=resolved_threshold)),
    )

    mase_val: float | None = None
    if task == "regression" and pred and model_name is not None:
        persist_mae = table["persistence"]["mae"]
        model_mae = table[model_name]["mae"]
        mase_val = (model_mae / persist_mae) if persist_mae > 0.0 else float("nan")

    checks: list[Check] = [
        _label_memory(task, metric, y_eval, p_eval["persistence"], horizon, resolved_threshold)
    ]
    if pred and model_name is not None:
        checks.append(
            _beats_baselines(
                metric,
                y_eval,
                p_eval[model_name],
                p_eval[best],
                best,
                block,
                n_boot,
                seed,
                horizon=horizon,
                groups=g_eval,
                times=t_eval,
                threshold=resolved_threshold,
            )
        )
    if reported is not None:
        checks.append(_reported_score(metric, reported, table, list(baseline_preds)))
    if split:
        checks.append(
            _temporal_split(
                data,
                split,
                gkey,
                horizon=horizon,
                time=time,
                train_mask=train_mask,
                test_mask=test_mask,
            )
        )
        scaler_check = _scaler_leakage(
            data, target, split, feature_cols, train_mask=train_mask, test_mask=test_mask
        )
        if scaler_check is not None:
            checks.append(scaler_check)
    if feature_cols:
        checks.append(_feature_leakage(data, y, gkey, feature_cols, horizon))
    if extra_checks:
        for fn in extra_checks:
            checks.append(fn(data.loc[eval_mask], p_eval, y_eval))

    return AuditReport(
        task=task,
        target=target,
        horizon=horizon,
        metric=metric,
        model=model_name,
        season=resolved_season,
        n_eval=n_eval,
        prevalence=float(y_eval.mean()) if task == "classification" else None,
        best_baseline=best,
        table=table,
        checks=checks,
        threshold=resolved_threshold if task == "classification" else None,
        mase=mase_val,
    )


def batch_audit(
    df: pd.DataFrame,
    target: str,
    *,
    group_col: str,
    **audit_kwargs: Any,
) -> BatchAuditReport:
    """Audit each series in ``group_col`` independently with Benjamini-Hochberg FDR control."""
    if group_col not in df.columns:
        raise ValueError(f"columns not found: {[group_col]}")
    kwargs = dict(audit_kwargs)
    kwargs.pop("groups", None)

    reports: dict[str, AuditReport] = {}
    rows: list[dict[str, Any]] = []
    dm_pvals: list[float] = []
    metric_name = ""
    model_name: str | None = None

    for key, rep in _batch_reports(df, target, group_col, kwargs):
        reports[key] = rep
        metric_name = rep.metric
        model_name = rep.model
        beat = next((c for c in rep.checks if c.name == "beats trivial baselines"), None)
        if beat is not None:
            status = beat.status
            diff = float(beat.details.get("difference", float("nan")))
            dm_p = float(beat.details.get("dm_p_value", float("nan")))
        else:
            status = rep.verdict
            diff = float("nan")
            dm_p = float("nan")
        dm_pvals.append(dm_p)
        row: dict[str, Any] = {
            group_col: key,
            "n_eval": rep.n_eval,
            "status": status,
            "best_baseline": rep.best_baseline,
            "baseline_score": rep.table[rep.best_baseline][rep.metric],
            "diff": diff,
            "dm_p_value": dm_p,
            "season": rep.season,
        }
        if rep.model is not None:
            row["model_score"] = rep.table[rep.model][rep.metric]
            if rep.mase is not None:
                row["mase"] = rep.mase
        rows.append(row)

    q_values = metrics.benjamini_hochberg(dm_pvals)
    for row, q in zip(rows, q_values, strict=True):
        row["fdr_q_value"] = float(q) if np.isfinite(q) else float("nan")

    return BatchAuditReport(
        group_col=group_col,
        metric=metric_name,
        model=model_name,
        reports=reports,
        rows=rows,
    )


def _batch_reports(
    df: pd.DataFrame, target: str, group_col: str, kwargs: dict[str, Any]
) -> Iterator[tuple[str, AuditReport]]:
    """(str(group), report) per series, in groupby(sort=False) order: natively in parallel when
    possible, otherwise one Python audit() call per group."""
    native = _backend.native
    hooks = kwargs.get("extra_baselines") or kwargs.get("extra_checks")
    # Categorical group columns stay on groupby, whose handling of unused categories changed
    # between pandas 2 and 3 (``observed=``).
    categorical = isinstance(df[group_col].dtype, pd.CategoricalDtype)
    if native is not None and not hooks and not categorical and set(kwargs) <= _AUDIT_KWARGS:
        codes, uniques = pd.factorize(df[group_col], sort=False)
        if len(uniques) > 0:
            task = kwargs.get("task", "classification")
            horizon = kwargs.get("horizon", 1)
            if task not in ("classification", "regression"):
                raise ValueError("task must be 'classification' or 'regression'")
            if horizon < 1:
                raise ValueError("horizon must be >= 1")
            feature_cols = _as_list(kwargs.get("features"))
            pred, time, split = kwargs.get("pred"), kwargs.get("time"), kwargs.get("split")
            _require_columns(df, [target, pred, time, split, *feature_cols])
            metric = kwargs.get("metric") or ("f1" if task == "classification" else "mae")
            allowed = metrics.CLASSIFICATION if task == "classification" else metrics.REGRESSION
            if metric not in allowed:
                raise ValueError(f"metric {metric!r} is not valid for {task}; choose from {allowed}")
            prepared = _native_prepare(
                df, target, task=task, horizon=horizon, pred=pred, model_name=kwargs.get("model_name"),
                group_cols=[], time=time, split=split, feature_cols=feature_cols,
                season=kwargs.get("season"), window=kwargs.get("window"),
                include_ewma=kwargs.get("include_ewma", False), ar_lags=kwargs.get("ar_lags"), metric=metric,
                reported=kwargs.get("reported"), threshold=kwargs.get("threshold", 0.5),
                n_boot=kwargs.get("n_boot", 500), block=kwargs.get("block"), seed=kwargs.get("seed", 0),
                extra_baselines=None, want_eval_arrays=False,
            )
            if prepared is not None:
                raws = native.audit_each_series(
                    *prepared, np.ascontiguousarray(codes, dtype=np.int64), len(uniques)
                )
                for k, raw in enumerate(raws):
                    yield str(uniques[k]), _report_from_native(
                        raw, df, None, metric, horizon=horizon, model_name=kwargs.get("model_name"),
                        pred=pred, reported=kwargs.get("reported"),
                    )
                return
    for grp, gdf in df.groupby(group_col, sort=False):
        yield str(grp), audit(gdf, target, **kwargs)


# --- native dispatch -------------------------------------------------------------


_AUDIT_KWARGS = {
    "task", "horizon", "pred", "model_name", "groups", "time", "split", "features", "season", "window",
    "include_ewma", "ar_lags", "metric", "reported", "threshold", "n_boot", "block", "seed",
    "extra_baselines", "extra_checks",
}


def _key_column(col: pd.Series) -> np.ndarray | None:
    """Order- and equality-preserving int64/float64 sort key for a column, or None to stay in pandas.

    Integers and floats pass through; datetimes become their int64 ticks; anything else
    (strings, categoricals, objects) becomes ``pd.factorize(sort=True)`` codes. Missing values
    (NaN/NaT/None/pd.NA) keep the pandas path, where they have special sort and grouping rules.
    """
    dtype = col.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "iub":
        if dtype.kind == "u" and len(col) and int(col.max()) > np.iinfo(np.int64).max:
            return None
        return np.ascontiguousarray(col.to_numpy(dtype=np.int64))
    if isinstance(dtype, np.dtype) and dtype.kind == "f":
        values = np.ascontiguousarray(col.to_numpy(dtype=np.float64))
        return None if np.isnan(values).any() else values
    if isinstance(dtype, np.dtype) and dtype.kind in "mM":
        values = col.to_numpy()
        return None if np.isnat(values).any() else np.ascontiguousarray(values.view(np.int64))
    if col.isna().any():
        return None
    try:
        codes, _ = pd.factorize(col, sort=True)
    except TypeError:
        return None
    return np.ascontiguousarray(codes, dtype=np.int64)


def _split_kinds(col: pd.Series) -> np.ndarray:
    """0 = other, 1 = train, 2 = test per row, matching ``col.astype(str).str.lower()``."""
    codes, uniques = pd.factorize(col)
    labels = [str(u).lower() for u in uniques]
    kinds = [2 if lab == "test" else 1 if lab == "train" else 0 for lab in labels]
    kind_of = np.array([*kinds, 0], dtype=np.uint8)
    return np.ascontiguousarray(kind_of[codes])  # code -1 (missing) indexes the trailing 0


def _native_prepare(
    df: pd.DataFrame,
    target: str,
    *,
    task: str,
    horizon: int,
    pred: str | None,
    model_name: str | None,
    group_cols: list[str],
    time: str | None,
    split: str | None,
    feature_cols: list[str],
    season: int | str | None,
    window: int | None,
    include_ewma: bool,
    ar_lags: int | None,
    metric: str,
    reported: float | None,
    threshold: float | str,
    n_boot: int,
    block: int | str | None,
    seed: int,
    extra_baselines: dict[str, pd.Series] | None,
    want_eval_arrays: bool,
) -> tuple[Any, dict[str, Any]] | None:
    """Inputs and config for the native pipeline, or None to run the pandas reference path."""
    native = _backend.native
    if native is None:
        return None
    # Reports carry the caller's own labels (table keys, detail keys, column names). The native
    # engine speaks strings, so non-string labels keep the pandas path.
    labels = [target, *feature_cols, *(extra_baselines or {})]
    if pred:
        labels += [pred, model_name or pred]
    if not all(isinstance(label, str) for label in labels):
        return None
    # Arguments the engine would read differently from pandas/NumPy keep the reference path, which
    # then uses or rejects them as before: bools (``True`` is an int, but pandas rejects
    # ``window=True`` and the reports would show ``m=True``), floats where ints are expected, and
    # non-numeric scores.
    native_seed = metrics._native_seed(seed)
    if native_seed is None or not _is_int(n_boot) or not _is_int(horizon):
        return None
    if window is not None and (not _is_int(window) or window < 0):
        return None
    if ar_lags is not None and not _is_int(ar_lags):
        return None
    if reported is not None and not _is_real(reported):
        return None
    threshold_value = 0.5
    if task == "classification" and threshold != "auto":
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            return None
    season_mode, season_value, season_error = "none", 0, None
    if season == "auto":
        season_mode = "auto"
    elif isinstance(season, bool):
        return None
    elif isinstance(season, int):
        if season < 0:
            return None
        season_mode, season_value = "fixed", int(season)
    elif season is not None:
        season_error = f"season must be an int, 'auto', or None; got {season!r}"
    if block == "auto":
        block_mode, block_value = "auto", 0
    elif block is None:
        block_mode, block_value = "default", 0
    else:
        try:
            block_mode, block_value = "fixed", int(block)
        except (TypeError, ValueError):
            return None

    try:
        keys = [_key_column(df[c]) for c in group_cols]
        time_key = _key_column(df[time]) if time else None
        if any(k is None for k in keys) or (time and time_key is None):
            return None
        inputs = native.AuditInputs()
        inputs.target = np.ascontiguousarray(df[target].astype(float).to_numpy(), dtype=np.float64)
        pred_values = None
        if pred:
            pred_values = np.ascontiguousarray(df[pred].astype(float).to_numpy(), dtype=np.float64)
            inputs.pred = pred_values
        inputs.group_keys = keys
        if time_key is not None:
            if time_key.dtype == np.float64:
                inputs.time_f64 = time_key
            else:
                inputs.time_i64 = time_key
        if split:
            inputs.split = _split_kinds(df[split])
        inputs.feature_names = list(feature_cols)
        inputs.features = [
            np.ascontiguousarray(df[f].astype(float).to_numpy(), dtype=np.float64) for f in feature_cols
        ]
        extras = extra_baselines or {}
        inputs.extra_names = [str(k) for k in extras]
        inputs.extras = [
            np.ascontiguousarray(v.reindex(df.index).astype(float).to_numpy(), dtype=np.float64)
            for v in extras.values()
        ]
    except (TypeError, ValueError, KeyError):
        return None  # let the pandas path raise its own error, in its own order
    if task == "classification":
        # scikit-learn rejects infinite scores with its own message; keep that behaviour.
        scored = [a for a in (pred_values, *inputs.extras) if a is not None]
        if any(np.isinf(a).any() for a in scored):
            return None

    config: dict[str, Any] = {
        "task": task,
        "horizon": int(horizon),
        "target_name": str(target),
        "model_name": (model_name or pred) if pred else None,
        "split_name": str(split) if split else "",
        "metric": metric,
        "season_mode": season_mode,
        "season": season_value,
        "season_error": season_error,
        "window": None if window is None else int(window),
        "include_ewma": bool(include_ewma),
        "ar_lags": None if ar_lags is None else int(ar_lags),
        "reported": None if reported is None else float(reported),
        "threshold_auto": task == "classification" and threshold == "auto",
        "threshold": threshold_value,
        "n_boot": int(n_boot),
        "block_mode": block_mode,
        "block": block_value,
        "seed": native_seed,
        "ewm_com1_update": baselines._PANDAS_EWM_COM1_UPDATE,
        "want_eval_arrays": want_eval_arrays,
    }
    return inputs, config


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    numeric = isinstance(value, (int, float, np.integer, np.floating))
    return numeric and not isinstance(value, (bool, np.bool_))


def _report_from_native(
    raw: dict[str, Any],
    df: pd.DataFrame,
    extra_checks: Sequence[CheckFn] | None,
    metric: str,
    *,
    horizon: int,
    model_name: str | None,
    pred: str | None,
    reported: float | None,
) -> AuditReport:
    checks = [Check(name, status, summary, details) for name, status, summary, details in raw["checks"]]
    for check in checks:
        if check.name == "reported score":
            check.details["reported"] = reported  # the caller's object, e.g. an int stays an int
    if extra_checks:
        rows = raw["eval_rows"]
        p_eval = dict(zip(raw["table"], raw["eval_predictions"], strict=True))
        for fn in extra_checks:
            checks.append(fn(df.iloc[rows], p_eval, raw["eval_target"]))
    return AuditReport(
        task=raw["task"],
        target=raw["target"],
        horizon=horizon,  # the caller's objects, as the Python path stores them
        metric=metric,
        model=(model_name or pred) if pred else model_name,
        season=raw["season"],
        n_eval=raw["n_eval"],
        prevalence=raw["prevalence"],
        best_baseline=raw["best_baseline"],
        table=raw["table"],
        checks=checks,
        threshold=raw["threshold"],
        mase=raw["mase"],
    )


# --- checks -------------------------------------------------------------------


def _label_memory(
    task: str,
    metric: str,
    y_true: np.ndarray,
    persist: np.ndarray,
    horizon: int,
    threshold: float = 0.5,
) -> Check:
    """How much of the 'forecast' is the target simply remembering itself?"""
    autocorr = _corr(y_true, persist)
    persist_value = metrics.compute(metric, y_true, persist, threshold=threshold)
    details: dict[str, Any] = {
        "lag_autocorrelation": autocorr,
        f"persistence_{metric}": persist_value,
    }
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


def _beats_baselines(
    metric: str,
    y_true: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    baseline_name: str,
    block: int | Literal["auto"] | None,
    n_boot: int,
    seed: int,
    *,
    horizon: int = 1,
    groups: np.ndarray | None = None,
    times: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Check:
    if np.ptp(y_true) == 0:
        return Check(
            "beats trivial baselines",
            "INFO",
            f"The target is constant ({y_true[0]:g}) over the scored rows, so there is nothing to forecast.",
            {"baseline": baseline_name},
        )
    model_raw = metrics.compute(metric, y_true, model, threshold=threshold)
    base_raw = metrics.compute(metric, y_true, baseline, threshold=threshold)
    diff = metrics.score(metric, y_true, model, threshold=threshold) - metrics.score(
        metric, y_true, baseline, threshold=threshold
    )
    if block == "auto":
        d_step = metrics._step_loss_differential(metric, y_true, model, baseline, threshold=threshold)
        if d_step is None:
            d_step = y_true - baseline
        is_panel, time_inv, n_times = metrics._is_aligned_panel(groups, times)
        if is_panel and time_inv is not None:
            counts = np.bincount(time_inv, minlength=n_times).astype(np.float64)
            d_series = np.bincount(time_inv, weights=d_step, minlength=n_times) / np.maximum(counts, 1.0)
        else:
            d_series = d_step
        b_len = metrics.estimate_block_length(d_series)
    elif block is not None:
        b_len = int(block)
    else:
        b_len = metrics.default_block_length(len(y_true), groups=groups, times=times)
    lo, hi = metrics.block_bootstrap_diff(
        metric,
        y_true,
        model,
        baseline,
        b_len,
        n_boot,
        seed,
        groups=groups,
        times=times,
        threshold=threshold,
    )
    dm_stat, dm_p = metrics.diebold_mariano(
        metric,
        y_true,
        model,
        baseline,
        horizon=horizon,
        groups=groups,
        times=times,
        threshold=threshold,
    )
    details: dict[str, Any] = {
        "baseline": baseline_name,
        "difference": diff,
        "ci95": [lo, hi],
        "block_length": b_len,
        "dm_stat": dm_stat,
        "dm_p_value": dm_p,
    }
    if metric in metrics.LOWER_IS_BETTER and base_raw > 0.0:
        details["skill_score"] = 1.0 - (model_raw / base_raw)

    what = (
        f"{metric} reduction vs {baseline_name}"
        if metric in metrics.LOWER_IS_BETTER
        else f"{metric} gain over {baseline_name}"
    )
    dm_note = f", DM p={dm_p:.3g}" if np.isfinite(dm_p) else ""
    margin = f"{what} = {diff:+.4f} (95% block-bootstrap CI {lo:+.4f} to {hi:+.4f}{dm_note})"
    if not np.isfinite(diff) or diff <= 0:
        return Check(
            "beats trivial baselines",
            "FAIL",
            f"Does not beat the best trivial baseline: {margin}.",
            details,
        )
    if not np.isfinite(lo) or lo <= 0:
        return Check(
            "beats trivial baselines",
            "WARN",
            f"Ahead of the best trivial baseline, but the margin is within noise: {margin}.",
            details,
        )
    return Check("beats trivial baselines", "PASS", f"Beats the best trivial baseline: {margin}.", details)


def _reported_score(
    metric: str,
    reported: float,
    table: dict[str, dict[str, float]],
    baseline_names: list[str],
) -> Check:
    lower = metric in metrics.LOWER_IS_BETTER
    name = max(baseline_names, key=lambda b: _nan_low(-table[b][metric] if lower else table[b][metric]))
    value = table[name][metric]
    gap = (value - reported) if lower else (reported - value)
    details: dict[str, Any] = {
        "reported": reported,
        "baseline": name,
        "baseline_value": value,
        "gap": gap,
    }
    text = f"Reported {metric} {reported:.4g}; {name} alone gets {value:.4g} on the same rows"
    tolerance = 0.05 * abs(value) if lower else 0.02
    if not np.isfinite(value):
        return Check("reported score", "INFO", f"{metric} is undefined on these rows.", details)
    if gap <= 0:
        return Check(
            "reported score",
            "FAIL",
            f"{text}. The claim is not evidence of forecasting skill.",
            details,
        )
    if gap < tolerance:
        return Check("reported score", "WARN", f"{text}. The claimed improvement is marginal.", details)
    return Check("reported score", "PASS", f"{text}.", details)


def _temporal_split(
    data: pd.DataFrame,
    split: str,
    gkey: pd.Series | None,
    *,
    horizon: int = 1,
    time: str | None = None,
    train_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
) -> Check:
    if train_mask is None or test_mask is None:
        labels = data[split].astype(str).str.lower()
        train_arr = labels.eq("train").to_numpy(dtype=bool)
        test_arr = labels.eq("test").to_numpy(dtype=bool)
    else:
        train_arr = train_mask.to_numpy(dtype=bool)
        test_arr = test_mask.to_numpy(dtype=bool)

    if not train_arr.any():
        return Check(
            "temporal split",
            "INFO",
            "No rows are labelled 'train', so there is no overlap to check.",
        )

    n = len(data)
    if gkey is None:
        g_arr = np.zeros(n, dtype=np.int64)
        n_groups = 1
        clock = np.arange(n, dtype=np.int64)
        grp_start = np.zeros(n, dtype=bool)
        grp_start[0] = True
    else:
        # ngroup() is NaN where a group key is missing. Those rows sort last, so they form one
        # contiguous block: give it its own group instead of casting NaN to int64, which is
        # INT64_MIN on x86-64 (an IndexError below) and 0 on arm64 (merged into group 0).
        g_float = gkey.to_numpy(dtype=np.float64)
        missing = np.isnan(g_float)
        g_arr = np.where(missing, 0.0, g_float).astype(np.int64)
        if missing.any():
            g_arr[missing] = int(np.max(g_arr[~missing])) + 1 if (~missing).any() else 0
        n_groups = int(np.max(g_arr)) + 1
        grp_start = np.empty(n, dtype=bool)
        grp_start[0] = True
        grp_start[1:] = g_arr[1:] != g_arr[:-1]
        # Vectorized within-group cumcount (assumes data is sorted by groups, guaranteed by audit())
        idx = np.arange(n, dtype=np.int64)
        run_starts = np.maximum.accumulate(np.where(grp_start, idx, 0))
        clock = idx - run_starts

    # Per-group max train clock and min test clock in O(n) NumPy
    last_train = np.full(n_groups, -1, dtype=np.int64)
    np.maximum.at(last_train, g_arr[train_arr], clock[train_arr])
    has_train = last_train >= 0

    test_g = g_arr[test_arr]
    test_t = clock[test_arr]
    before = has_train[test_g] & (test_t < last_train[test_g])
    share_before = float(before.mean())

    # Count contiguous test segments per group in O(n) NumPy
    prev_test = np.empty(n, dtype=bool)
    prev_test[0] = False
    prev_test[1:] = test_arr[:-1]
    seg_starts = test_arr & (~prev_test | grp_start)
    runs_per_g = np.bincount(g_arr, weights=seg_starts.astype(np.int64), minlength=n_groups)
    active_runs = runs_per_g[runs_per_g > 0]
    mean_runs = float(active_runs.mean()) if len(active_runs) > 0 else 0.0

    # Check horizon embargo gap between last_train and first_test
    embargo_violated = 0
    if horizon > 1:
        first_test = np.full(n_groups, n + 1, dtype=np.int64)
        np.minimum.at(first_test, test_g, test_t)
        both = has_train & (first_test <= n)
        if both.any():
            embargo_violated = int(np.count_nonzero((first_test[both] - last_train[both]) < horizon))

    # Check duplicate timestamps within each group (data is already sorted by group + time!)
    dup_timestamps = 0
    if time is not None and n > 1:
        ts = data[time].to_numpy()
        same_group = ~grp_start[1:]
        dup_timestamps = int(np.count_nonzero(same_group & (ts[1:] == ts[:-1])))

    details: dict[str, Any] = {
        "test_rows_before_last_train_row": share_before,
        "mean_test_segments_per_group": mean_runs,
        "embargo_violations": embargo_violated,
        "duplicate_timestamps": dup_timestamps,
    }

    if share_before == 0:
        if embargo_violated > 0:
            return Check(
                "temporal split",
                "WARN",
                f"Test rows follow training rows, but {embargo_violated} series have a train/test gap "
                f"smaller than horizon={horizon}, so late training targets overlap the test period.",
                details,
            )
        if dup_timestamps > 0:
            return Check(
                "temporal split",
                "WARN",
                f"Every test row follows training rows, but {dup_timestamps} duplicate timestamps "
                "were found within series (lag alignment may be ambiguous).",
                details,
            )
        return Check(
            "temporal split",
            "PASS",
            "Every test row comes after the training data in time.",
            details,
        )
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


def _feature_leakage(
    data: pd.DataFrame,
    y: pd.Series,
    gkey: pd.Series | None,
    features: list[str],
    horizon: int,
) -> Check:
    """A feature tracking y[t+j] (0 < j <= h) linearly or monotonically is future leakage."""
    leaks: list[str] = []
    strongest: dict[str, float] = {}
    y_obs = baselines.lag_target(y, gkey, horizon).to_numpy()
    for f in features:
        x = data[f].astype(float).to_numpy()
        best = 0.0
        obs_r = abs(_corr(x, y_obs))
        obs_rho = abs(_spearman(x, y_obs))
        n_unique = len(np.unique(x[~np.isnan(x)]))
        for k in range(horizon):  # lag_target(k) for k < h is y[t + h - k], unobserved at t
            future_y = baselines.lag_target(y, gkey, k).to_numpy()
            r = abs(_corr(x, future_y))
            if np.isfinite(r) and r > best:
                best = r
            if np.isfinite(r) and r >= 0.999 and (
                not np.isfinite(obs_r) or r > obs_r + 1e-4 or obs_r < 0.999
            ):
                leaks.append(f"{f} ~ y[t+{horizon - k}] (|r|={r:.4f})")
                continue
            if n_unique >= 10:
                rho = abs(_spearman(x, future_y))
                if np.isfinite(rho) and rho > best:
                    best = rho
                if np.isfinite(rho) and rho >= 0.999 and (not np.isfinite(obs_rho) or rho > obs_rho + 1e-4):
                    leaks.append(f"{f} ~ y[t+{horizon - k}] (monotonic |rho|={rho:.4f})")
        strongest[f] = best
    details: dict[str, Any] = {"max_abs_corr_with_future_target": strongest}
    if leaks:
        return Check(
            "feature leakage",
            "FAIL",
            "Features copy future target values: " + "; ".join(leaks) + ".",
            details,
        )
    return Check("feature leakage", "PASS", "No feature is a near-copy of a future target value.", details)


def _scaler_leakage(
    data: pd.DataFrame,
    target: str,
    split: str,
    features: list[str],
    *,
    train_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
) -> Check | None:
    """Detect [0, 1] min-max scaling fit on train+test combined (test sets the global bound)."""
    if train_mask is None or test_mask is None:
        labels = data[split].astype(str).str.lower()
        tr_arr = labels.eq("train").to_numpy(dtype=bool)
        te_arr = labels.eq("test").to_numpy(dtype=bool)
    else:
        tr_arr = train_mask.to_numpy(dtype=bool)
        te_arr = test_mask.to_numpy(dtype=bool)
    if not tr_arr.any() or not te_arr.any():
        return None

    suspect: list[str] = []
    cols = [target, *features]
    for col in cols:
        vals = data[col].to_numpy(dtype=np.float64, copy=False)
        tr = vals[tr_arr]
        te = vals[te_arr]
        tr = tr[~np.isnan(tr)]
        te = te[~np.isnan(te)]
        if len(tr) < 10 or len(te) < 5:
            continue
        tr_min, tr_max = float(tr.min()), float(tr.max())
        te_min, te_max = float(te.min()), float(te.max())
        gmin, gmax = min(tr_min, te_min), max(tr_max, te_max)
        if not (gmin >= -1e-6 and gmax <= 1.0 + 1e-6 and gmin <= 1e-6 and gmax >= 1.0 - 1e-6):
            continue
        # Skip binary 0/1 columns (fast check without hashing 1M values)
        if not (np.any((tr > 1e-6) & (tr < 1.0 - 1e-6)) or np.any((te > 1e-6) & (te < 1.0 - 1e-6))):
            continue
        test_sets_min = te_min <= 1e-6 and tr_min > 1e-3
        test_sets_max = te_max >= 1.0 - 1e-6 and tr_max < 1.0 - 1e-3
        if test_sets_min or test_sets_max:
            suspect.append(col)

    if suspect:
        return Check(
            "scaler leakage",
            "WARN",
            "Columns appear min-max scaled across train+test combined "
            "(exact 0/1 boundary occurs only in test): "
            + ", ".join(suspect)
            + ".",
            {"suspect_columns": suspect},
        )
    return None


# --- helpers ------------------------------------------------------------------


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return float("nan")
    res = reference_module("scipy.stats").spearmanr(a, b)
    return float(res.statistic)


def _nan_low(x: float) -> float:
    return -np.inf if np.isnan(x) else x


def _as_list(x: str | Sequence[str] | None) -> list[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _require_columns(df: pd.DataFrame, cols: list[str | None]) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}")
