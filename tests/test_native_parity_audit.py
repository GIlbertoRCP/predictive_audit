"""Phase 3: the native audit() / batch_audit() pipeline vs the pandas reference path.

Whole reports are compared: verdicts, check statuses and summary text must match exactly, and
numbers to 1e-9 (absolute 1e-12 near zero). Two documented exceptions come from LAPACK/BLAS,
whose results differ between libraries (Accelerate vs OpenBLAS) even for the reference itself:
- the classification Ridge AR baseline: its fitted weights differ by ~1 ulp, and with a 0/1 target
  many of its scores nearly tie, so its ROC/PR-AUC (and checks that use it as the best baseline)
  can move;
- R^2 near zero, whose denominator NumPy computes with a BLAS dot product.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predictive_audit import Check, _backend, audit, batch_audit, report

pytestmark = [
    pytest.mark.skipif(not _backend.native_available(), reason="native extension not built"),
    pytest.mark.filterwarnings("ignore"),
]


def _run(backend: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    with _backend.use(backend), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - compared across backends
            return ("raised", type(exc).__name__, str(exc))


def _make_frame(seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_series, steps = int(rng.integers(1, 5)), int(rng.integers(30, 250))
    key_kind = ["int", "str", "float"][int(rng.integers(3))]
    time_kind = ["int", "float", "datetime", "str", "sparse"][int(rng.integers(5))]
    parts = []
    for g in range(n_series):
        t = np.arange(steps)
        period = int(rng.integers(4, 30))
        y = np.sin(2 * np.pi * t / period) + rng.normal(scale=0.4, size=steps).cumsum() * 0.1
        d = pd.DataFrame({"t": t, "y": np.roll(y, -1), "pred": y + rng.normal(scale=0.3, size=steps)})
        d["g"] = {"int": g, "str": f"od_{g}", "float": g * 0.5}[key_kind]
        d["leak"] = d["y"] * 2 + 1 if rng.random() < 0.3 else rng.normal(size=steps)
        d["mono"] = np.exp(d["y"]) if rng.random() < 0.3 else rng.normal(size=steps)
        d["scaled"] = (d["y"] - d["y"].min()) / (d["y"].max() - d["y"].min())
        split = np.where(t < int(0.7 * steps), "train", "test")
        if rng.random() < 0.25:
            rng.shuffle(split)
        if rng.random() < 0.2:
            split = np.where(t < 5, "warmup", split)
        d["split"] = pd.Series(split).str.upper() if rng.random() < 0.3 else split
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    if time_kind == "float":
        df["t"] = df["t"] * 0.25
    elif time_kind == "datetime":
        df["t"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df["t"], unit="h")
    elif time_kind == "str":
        df["t"] = df["t"].map(lambda v: f"{v:05d}")
    elif time_kind == "sparse":
        df["t"] = df["t"] * 3
    if rng.random() < 0.5:
        df = df.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    clf = rng.random() < 0.4
    if clf:
        df["y"] = (df["y"] > df["y"].median()).astype(float)
        df["pred"] = 1 / (1 + np.exp(-df["pred"]))
    metrics_for_task = (
        ["f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "accuracy", "brier"]
        if clf
        else ["mae", "rmse", "r2", "smape"]
    )
    kwargs: dict[str, Any] = {
        "task": "classification" if clf else "regression",
        "pred": "pred",
        "groups": "g" if rng.random() < 0.7 else None,
        "time": "t" if rng.random() < 0.9 else None,
        "split": "split" if rng.random() < 0.9 else None,
        "features": ["leak", "mono", "scaled"] if rng.random() < 0.6 else None,
        "season": ["auto", None, 7][int(rng.integers(3))],
        "include_ewma": bool(rng.random() < 0.5),
        "ar_lags": [None, 3][int(rng.integers(2))],
        "horizon": [1, 1, 3][int(rng.integers(3))],
        "n_boot": 60,
        "seed": seed,
        "metric": metrics_for_task[int(rng.integers(len(metrics_for_task)))],
        "threshold": [0.5, "auto", 0.3][int(rng.integers(3))] if clf else 0.5,
        "reported": [None, 0.5, 1.0][int(rng.integers(3))],
        "block": [None, "auto", 10][int(rng.integers(3))],
    }
    return df, kwargs


def _assert_same(native: Any, python: Any, path: str = "", *, lapack_ties: bool = False) -> None:
    if isinstance(python, dict):
        assert list(native) == list(python), path
        for k in python:
            if lapack_ties and "linear AR" in str(k) and path == "/table":
                continue  # rank metrics of a near-tied, BLAS-dependent baseline
            _assert_same(native[k], python[k], f"{path}/{k}", lapack_ties=lapack_ties)
    elif isinstance(python, list):
        assert len(native) == len(python), path
        for i, (a, b) in enumerate(zip(native, python, strict=True)):
            _assert_same(a, b, f"{path}[{i}]", lapack_ties=lapack_ties)
    elif isinstance(python, float):
        assert isinstance(native, float), path
        assert (math.isnan(native) and math.isnan(python)) or native == pytest.approx(
            python, rel=1e-9, abs=1e-12
        ), (path, native, python)
    else:
        assert type(native) is type(python) and native == python, (path, native, python)


def _compare_reports(native: Any, python: Any, kwargs: dict[str, Any]) -> None:
    if isinstance(python, tuple):  # both must raise the same error
        assert native == python
        return
    dn, dp = native.to_dict(), python.to_dict()
    assert dn["verdict"] == dp["verdict"]
    assert [c["status"] for c in dn["checks"]] == [c["status"] for c in dp["checks"]]
    lapack_ties = kwargs.get("task") == "classification" and bool(kwargs.get("ar_lags"))
    if lapack_ties and "linear AR" in dp["best_baseline"]:
        # The checks are computed against a BLAS-dependent best baseline: compare structure only.
        _assert_same(dn["table"], dp["table"], "/table", lapack_ties=True)
        return
    _assert_same(dn, dp, lapack_ties=lapack_ties)
    if not lapack_ties:
        assert report.to_text(native) == report.to_text(python)
        assert report.to_markdown(native) == report.to_markdown(python)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31))
def test_audit_reports_match(seed: int) -> None:
    df, kwargs = _make_frame(seed)
    native, python = _run("native", audit, df, "y", **kwargs), _run("python", audit, df, "y", **kwargs)
    _compare_reports(native, python, kwargs)


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**31))
def test_batch_audit_matches(seed: int) -> None:
    df, kwargs = _make_frame(seed)
    kwargs.pop("groups")
    native = _run("native", batch_audit, df, "y", group_col="g", **kwargs)
    python = _run("python", batch_audit, df, "y", group_col="g", **kwargs)
    if isinstance(python, tuple):
        assert native == python
        return
    assert list(native.reports) == list(python.reports)
    for key in python.reports:
        _compare_reports(native.reports[key], python.reports[key], kwargs)
    assert [r["status"] for r in native.rows] == [r["status"] for r in python.rows]
    assert native.tally == python.tally
    if not (kwargs["task"] == "classification" and kwargs["ar_lags"]):
        _assert_same(native.to_dict(), python.to_dict())


def test_batch_audit_categorical_groups_match() -> None:
    df = _frame().assign(g=lambda d: pd.Categorical(d["g"].map({0: "a", 1: "b"}), categories=["a", "b", "c"]))
    kwargs: dict[str, Any] = dict(task="regression", pred="pred", time="t", split="split", n_boot=40)
    native = _run("native", batch_audit, df, "y", group_col="g", **kwargs)
    python = _run("python", batch_audit, df, "y", group_col="g", **kwargs)
    if isinstance(python, tuple):
        assert native == python
        return
    _assert_same(native.to_dict(), python.to_dict())


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.sin(t / 5) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({
        "t": t,
        "g": t % 2,
        "y": np.roll(y, -1),
        "pred": y + rng.normal(scale=0.1, size=n),
        "split": np.where(t < 140, "train", "test"),
    })


def test_extra_baselines_and_checks_match() -> None:
    df = _frame()
    oracle = df["y"] + 0.01

    def custom(data: pd.DataFrame, preds: dict[str, np.ndarray], y: np.ndarray) -> Check:
        return Check("custom", "INFO", f"{len(data)} rows, first index {data.index[0]}, "
                     f"{sorted(preds)}, mean y {float(np.mean(y)):.6f}")

    kwargs: dict[str, Any] = dict(task="regression", pred="pred", groups="g", time="t", split="split",
                                  extra_baselines={"oracle": oracle}, extra_checks=[custom], n_boot=50)
    native, python = _run("native", audit, df, "y", **kwargs), _run("python", audit, df, "y", **kwargs)
    _compare_reports(native, python, kwargs)


@pytest.mark.parametrize(
    ("mutate", "kwargs"),
    [
        # Data-dependent errors must carry the reference's message, raised in the reference's order.
        (lambda d: d.assign(y=d["y"] * 3), {"task": "classification"}),
        (lambda d: d.assign(split="train"), {}),
        (lambda d: d.iloc[:12], {}),
        (lambda d: d.assign(y=d["y"] * 3), {"task": "classification", "season": "weekly"}),
        (lambda d: d, {"season": "weekly"}),
        (lambda d: d, {"ar_lags": -2}),
        # Inputs the native path hands back to pandas: results (or errors) are unchanged.
        (lambda d: d.assign(g=np.where(d.index % 7 == 0, np.nan, d["g"])), {}),
        (lambda d: d.assign(t=pd.Series(pd.NaT, index=d.index).where(d.index % 9 == 0,
                                                                       pd.Timestamp("2026-01-01")
                                                                       + pd.to_timedelta(d["t"], "h"))), {}),
        (lambda d: d.rename(columns={"pred": 7}), {"pred": 7}),
        (
            lambda d: d.assign(pred=np.where(d.index == 150, np.inf, d["pred"] > 0), y=(d["y"] > 0) * 1.0),
            {"task": "classification"},
        ),
        (lambda d: d.assign(g=d["g"].astype("category")), {}),
        (lambda d: d.assign(split=d["split"].astype("string")), {}),
        # Unusual argument types: the native path must read them exactly as the reference does.
        (lambda d: d, {"threshold": "AUTO"}),  # unused for regression
        (lambda d: d.assign(y=(d["y"] > 0) * 1.0), {"task": "classification", "threshold": "AUTO"}),
        (lambda d: d.assign(y=(d["y"] > 0) * 1.0), {"task": "classification", "threshold": "0.4"}),
        (lambda d: d, {"season": True}),
        (lambda d: d, {"season": np.int64(7)}),
        (lambda d: d, {"window": True}),
        (lambda d: d, {"ar_lags": True}),
        (lambda d: d, {"horizon": True}),
        (lambda d: d, {"horizon": np.int64(2)}),
        (lambda d: d, {"horizon": 2.0}),
        (lambda d: d, {"n_boot": True}),
        (lambda d: d, {"seed": 1.5}),
        (lambda d: d, {"seed": True}),
        (lambda d: d, {"reported": "0.5"}),
        (lambda d: d, {"reported": 1}),
        (lambda d: d, {"reported": True}),
        (lambda d: d, {"pred": None, "model_name": "mine"}),
        (lambda d: d, {"model_name": "mine"}),
    ],
)
def test_errors_and_fallbacks_match(mutate: Any, kwargs: dict[str, Any]) -> None:
    df = mutate(_frame())
    base: dict[str, Any] = dict(
        task="regression", pred="pred", groups="g", time="t", split="split", n_boot=40
    )
    base.update(kwargs)
    _compare_reports(_run("native", audit, df, "y", **base), _run("python", audit, df, "y", **base), base)
