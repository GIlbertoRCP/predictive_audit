"""Tests for upgraded leakage checks, causal baselines, batch_audit, and randomized invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictive_audit import Check, audit, baselines, batch_audit, wide_to_long


def _series_df(y: np.ndarray, horizon: int = 1) -> pd.DataFrame:
    df = pd.DataFrame({"t": np.arange(len(y)), "y": y.astype(float)})
    df["target"] = df["y"].shift(-horizon)
    return df.dropna().reset_index(drop=True)


def test_embargo_violation_flagged_for_multistep_horizon():
    y = np.random.default_rng(1).normal(size=500).cumsum()
    df = _series_df(y, horizon=5)
    # Train immediately followed by test (gap = 1 < horizon = 5)
    df["split_tight"] = np.where(df["t"] < 350, "train", "test")
    # Proper embargo: 4 history rows between train and test (gap = 5 == horizon)
    split_gap = np.where(df["t"] < 350, "train", "test")
    split_gap[350:354] = "history"
    df["split_embargo"] = split_gap

    r_tight = audit(df, "target", task="regression", horizon=5, time="t", split="split_tight")
    r_gap = audit(df, "target", task="regression", horizon=5, time="t", split="split_embargo")
    c_tight = {c.name: c for c in r_tight.checks}["temporal split"]
    c_gap = {c.name: c for c in r_gap.checks}["temporal split"]
    assert c_tight.status == "WARN"
    assert c_tight.details["embargo_violations"] == 1
    assert c_gap.status == "PASS"
    assert c_gap.details["embargo_violations"] == 0


def test_duplicate_timestamps_warned():
    y = np.random.default_rng(2).normal(size=200).cumsum()
    df = _series_df(y)
    df.loc[10, "t"] = df.loc[9, "t"]
    df["split"] = np.where(np.arange(len(df)) < 140, "train", "test")
    r = audit(df, "target", task="regression", time="t", split="split")
    c = {c.name: c for c in r.checks}["temporal split"]
    assert c.status == "WARN"
    assert c.details["duplicate_timestamps"] == 1


def test_nonlinear_monotonic_feature_leakage_detected():
    rng = np.random.default_rng(3)
    y = np.exp(rng.normal(0.0, 0.3, size=600).cumsum() * 0.1) + 1.0
    df = _series_df(y, horizon=2)
    df["legit_log"] = np.log(df["y"])  # observed at t
    df["leaky_exp"] = np.exp(df["y"].shift(-1) * 0.5)  # monotonic transform of y[t+1]
    df = df.dropna().reset_index(drop=True)

    r = audit(df, "target", task="regression", horizon=2, time="t", features=["legit_log", "leaky_exp"])
    c = {c.name: c for c in r.checks}["feature leakage"]
    assert c.status == "FAIL"
    assert "leaky_exp" in c.summary
    assert "legit_log" not in c.summary


def test_global_minmax_scaler_leakage_detected():
    rng = np.random.default_rng(4)
    # Upward trend where test holds the global maximum -> global MinMax puts 1.0 exclusively in test
    raw = np.linspace(0.0, 100.0, 500) + rng.normal(scale=1.0, size=500)
    global_scaled = (raw - raw.min()) / (raw.max() - raw.min())
    train_min, train_max = raw[:400].min(), raw[:400].max()
    train_scaled = (raw - train_min) / (train_max - train_min)

    df_bad = _series_df(global_scaled)
    df_bad["split"] = np.where(df_bad["t"] < 400, "train", "test")
    r_bad = audit(df_bad, "target", task="regression", time="t", split="split")
    assert {c.name: c for c in r_bad.checks}["scaler leakage"].status == "WARN"

    df_ok = _series_df(train_scaled)
    df_ok["split"] = np.where(df_ok["t"] < 400, "train", "test")
    r_ok = audit(df_ok, "target", task="regression", time="t", split="split")
    assert "scaler leakage" not in {c.name for c in r_ok.checks}


def test_causal_ar_and_ewma_baselines():
    rng = np.random.default_rng(5)
    # Stationary AR(2) process: y[t] = 0.6*y[t-1] - 0.5*y[t-2] + eps
    n = 1200
    y = np.zeros(n)
    eps = rng.normal(scale=0.5, size=n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.5 * y[t - 2] + eps[t]
    df = _series_df(y)
    df["split"] = np.where(df["t"] < 800, "train", "test")

    r = audit(df, "target", task="regression", time="t", split="split", include_ewma=True, ar_lags=3)
    assert "ewma (3)" in r.table
    assert "linear AR(3)" in r.table
    assert r.table["linear AR(3)"]["mae"] < r.table["persistence"]["mae"]

    # Causality check: modifying test targets must NOT change fitted AR weights or earlier predictions
    fit_mask = df["split"] != "test"
    p1 = baselines.linear_ar(df["target"], None, 1, fit_mask, lags=3)
    df_mod = df.copy()
    df_mod.loc[df_mod["split"] == "test", "target"] += 1e6
    p2 = baselines.linear_ar(df_mod["target"], None, 1, fit_mask, lags=3)
    # Prediction at the first test row only depends on train targets + train-fitted weights
    first_test = int(df.index[df["split"] == "test"][0])
    assert p1.iloc[first_test] == pytest.approx(p2.iloc[first_test])


def test_extensibility_extra_baselines_and_checks():
    y = np.random.default_rng(6).normal(size=300).cumsum()
    df = _series_df(y)
    df["pred"] = df["y"]
    oracle = pd.Series(df["target"].to_numpy(), index=df.index)

    def custom_check(_eval_df, _preds, _y_eval) -> Check:
        return Check("custom", "PASS", "all good", {"ok": True})

    r = audit(
        df,
        "target",
        task="regression",
        pred="pred",
        time="t",
        extra_baselines={"oracle": oracle},
        extra_checks=[custom_check],
        n_boot=50,
    )
    assert "oracle" in r.table
    assert r.best_baseline == "oracle"
    assert any(c.name == "custom" and c.status == "PASS" for c in r.checks)


def test_batch_audit_with_fdr():
    rng = np.random.default_rng(8)
    frames = []
    for g in range(6):
        n = 400
        y = rng.normal(size=n).cumsum()
        df = _series_df(y)
        df["series"] = f"od_{g}"
        df["split"] = np.where(df["t"] < 280, "train", "test")
        # First 2 groups genuinely beat persistence; last 4 are noisy persistence
        if g < 2:
            df["pred"] = df["target"] + rng.normal(scale=0.1, size=len(df))
        else:
            df["pred"] = df["y"] + rng.normal(scale=1.0, size=len(df))
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)

    br = batch_audit(
        panel,
        "target",
        group_col="series",
        task="regression",
        pred="pred",
        time="t",
        split="split",
        n_boot=100,
    )
    summary = br.summary_frame()
    assert len(summary) == 6
    assert "fdr_q_value" in summary.columns
    assert br.tally["PASS"] == 2
    assert br.tally["FAIL"] == 4
    assert br.n_fdr_significant == 2
    assert br.verdict == "FAIL"
    assert br.to_dict()["n_groups"] == 6


def test_randomized_invariants_causality_and_row_order():
    """Property-style test across 25 random multi-group panels."""
    rng = np.random.default_rng(99)
    for trial in range(25):
        n_groups = int(rng.integers(2, 6))
        horizon = int(rng.integers(1, 5))
        window = int(rng.integers(2, 6))
        parts = []
        for g in range(n_groups):
            length = int(rng.integers(30, 60))
            vals = rng.normal(loc=g * 100.0, scale=1.0, size=length)
            parts.append(pd.DataFrame({"g": g, "t": np.arange(length), "target": vals}))
        df = pd.concat(parts, ignore_index=True)

        # Invariant 1: baselines never leak across groups (first `horizon` rows of each group are NaN)
        p = baselines.persistence(df["target"], df["g"], horizon)
        ma = baselines.moving_average(df["target"], df["g"], horizon, window)
        for _, grp in df.groupby("g", sort=False):
            idx = grp.index
            assert p.loc[idx[:horizon]].isna().all()
            assert ma.loc[idx[: horizon + window - 1]].isna().all()

        # Invariant 2: audit() is invariant to arbitrary input row shuffling when time & groups are set
        shuffled = df.sample(frac=1.0, random_state=trial).reset_index(drop=True)
        r1 = audit(df, "target", task="regression", groups="g", time="t", horizon=horizon, window=window)
        r2 = audit(
            shuffled, "target", task="regression", groups="g", time="t", horizon=horizon, window=window
        )
        assert r1.table == r2.table
        assert r1.n_eval == r2.n_eval


def test_wide_to_long_exact_column_order():
    rng = np.random.default_rng(12)
    wide = pd.DataFrame({
        "a": rng.normal(size=20),
        "b": rng.normal(size=20),
        "c": rng.normal(size=20),
        "split": ["train"] * 15 + ["test"] * 5,
    })
    long = wide_to_long(wide, keep=["split"], series_name="od", value_name="y")
    assert list(long.columns) == ["od", "step", "split", "y"]
    assert len(long) == 60
    np.testing.assert_array_equal(long.loc[:19, "y"].to_numpy(), wide["a"].to_numpy())
    np.testing.assert_array_equal(long.loc[20:39, "y"].to_numpy(), wide["b"].to_numpy())
    np.testing.assert_array_equal(long.loc[40:59, "y"].to_numpy(), wide["c"].to_numpy())
