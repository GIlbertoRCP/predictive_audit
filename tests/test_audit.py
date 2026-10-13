import numpy as np
import pandas as pd
import pytest

from predictive_audit import audit, baselines, wide_to_long
from predictive_audit.cli import main


def sticky_labels(n=4000, p_flip=0.02, seed=0):
    """Binary series that rarely changes state: persistence is nearly perfect."""
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < p_flip
    return (np.cumsum(flips) % 2).astype(int)


def frame_from_series(y, horizon=1):
    df = pd.DataFrame({"t": np.arange(len(y)), "y": y.astype(float)})
    df["target"] = df["y"].shift(-horizon)
    return df.dropna().reset_index(drop=True)


def test_persistence_is_lagged_target():
    df = frame_from_series(np.arange(10.0), horizon=2)
    p = baselines.persistence(df["target"], None, 2)
    # target[t] = y[t+2], persistence must equal y[t]
    np.testing.assert_array_equal(p.iloc[2:].to_numpy(), df["y"].iloc[2:].to_numpy())


def test_baselines_respect_groups():
    df = pd.DataFrame({"g": [0, 0, 0, 1, 1, 1], "target": [1.0, 2, 3, 10, 20, 30]})
    p = baselines.persistence(df["target"], df["g"], 1)
    assert np.isnan(p.iloc[3])  # no history leaks across series
    assert p.iloc[4] == 10


def test_seasonal_naive_never_peeks():
    df = frame_from_series(np.arange(30.0), horizon=5)
    s = baselines.seasonal_naive(df["target"], None, horizon=5, season=3)
    # k = 6 >= horizon: predicts y[t+5] with y[t-1], already observed at t
    assert s.iloc[10] == df["y"].iloc[9]


def test_model_that_copies_persistence_fails():
    df = frame_from_series(sticky_labels())
    df["pred"] = df["y"]  # "model" outputs the current state
    r = audit(df, "target", pred="pred", time="t", n_boot=100)
    check = {c.name: c for c in r.checks}["beats trivial baselines"]
    assert check.status == "FAIL"
    assert {c.name: c for c in r.checks}["label memory"].status == "WARN"
    assert r.verdict == "FAIL"


def test_model_with_real_lead_information_passes():
    rng = np.random.default_rng(1)
    n = 5000
    driver = rng.random(n) < 0.3  # an exogenous event announced one step early
    y = np.roll(driver, 1).astype(int)
    df = pd.DataFrame({"t": np.arange(n), "y": y})
    df["target"] = df["y"].shift(-1)
    df["pred"] = driver.astype(float)  # knows y[t+1] legitimately (e.g. a scheduler)
    df = df.dropna()
    r = audit(df, "target", pred="pred", time="t", n_boot=100)
    assert {c.name: c for c in r.checks}["beats trivial baselines"].status == "PASS"


def test_reported_score_matched_by_persistence():
    df = frame_from_series(sticky_labels())
    r = audit(df, "target", reported=0.95, metric="f1")
    check = {c.name: c for c in r.checks}["reported score"]
    assert check.status == "FAIL"
    assert check.details["baseline"] == "persistence"


def test_accuracy_claim_beaten_by_majority_class():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.03).astype(int)  # rare, unpredictable events
    df = frame_from_series(y)
    r = audit(df, "target", metric="accuracy", reported=0.96)
    check = {c.name: c for c in r.checks}["reported score"]
    assert check.status == "FAIL"
    assert check.details["baseline"] == "majority class"


def test_shuffled_split_detected_and_chronological_passes():
    df = frame_from_series(sticky_labels(2000))
    rng = np.random.default_rng(0)
    df["shuffled"] = np.where(rng.random(len(df)) < 0.2, "test", "train")
    df["chrono"] = np.where(df["t"] >= 1600, "test", "train")
    bad = {c.name: c for c in audit(df, "target", time="t", split="shuffled").checks}
    good = {c.name: c for c in audit(df, "target", time="t", split="chrono").checks}
    assert bad["temporal split"].status == "FAIL"
    assert good["temporal split"].status == "PASS"


def test_datetime_time_and_history_only_rows():
    """Training data lives elsewhere: warm-up rows are history, and timestamps are datetimes."""
    df = frame_from_series(np.random.default_rng(7).normal(size=600).cumsum())
    df["time"] = pd.date_range("2016-03-01", periods=len(df), freq="5min")
    df["split"] = ["history"] * 12 + ["test"] * (len(df) - 12)
    r = audit(df, "target", task="regression", time="time", split="split")
    checks = {c.name: c for c in r.checks}
    assert checks["temporal split"].status == "INFO"
    assert np.isfinite(r.table["training mean"]["mae"])  # fit on the history rows
    assert r.n_eval == len(df) - 12  # baselines may draw on history rows, so every test row scores


def test_feature_leakage_only_flags_future_values():
    y = np.random.default_rng(3).normal(size=1000).cumsum()
    df = frame_from_series(y, horizon=2)
    df["y_now"] = df["y"]  # legitimate: observed at t
    df["y_next"] = df["y"].shift(-1)  # leak: y[t+1] is unobserved at t
    df = df.dropna()
    r = audit(df, "target", task="regression", horizon=2, time="t", features=["y_now", "y_next"])
    check = {c.name: c for c in r.checks}["feature leakage"]
    assert check.status == "FAIL"
    assert "y_next" in check.summary and "y_now" not in check.summary


def test_regression_random_walk_persistence_is_best():
    y = np.random.default_rng(4).normal(size=3000).cumsum()
    df = frame_from_series(y)
    df["pred"] = df["y"] + np.random.default_rng(5).normal(scale=0.5, size=len(df))
    r = audit(df, "target", task="regression", pred="pred", time="t", n_boot=100)
    assert r.best_baseline == "persistence"
    assert {c.name: c for c in r.checks}["beats trivial baselines"].status == "FAIL"


def test_reported_error_metric_compares_against_best_baseline():
    y = np.random.default_rng(6).normal(size=2000).cumsum()
    df = frame_from_series(y)
    r = audit(df, "target", task="regression", metric="mae", reported=10.0)
    check = {c.name: c for c in r.checks}["reported score"]
    assert check.details["baseline"] == "persistence"  # lowest error, not highest
    assert check.status == "FAIL"  # persistence MAE ~0.8 beats a claimed 10.0


def test_estimate_season_finds_replay_loop_and_ignores_random_walk():
    rng = np.random.default_rng(8)
    loop = rng.normal(size=127)  # one "pcap" replayed over and over, plus noise
    periodic = pd.Series(np.tile(loop, 30) + rng.normal(scale=0.3, size=127 * 30))
    walk = pd.Series(rng.normal(size=3000).cumsum())
    everything = pd.Series(True, index=periodic.index)
    assert baselines.estimate_season(periodic, None, everything) == 127
    assert baselines.estimate_season(walk, None, pd.Series(True, index=walk.index)) is None


def test_auto_season_uses_only_non_test_rows():
    rng = np.random.default_rng(9)
    y = np.r_[np.tile(rng.normal(size=50), 40), rng.normal(size=1000)]  # periodic train, noise test
    df = frame_from_series(y)
    df["split"] = np.where(np.arange(len(df)) < 2000, "train", "test")
    r = audit(df, "target", task="regression", split="split", season="auto")
    assert r.season == 50
    assert "seasonal naive (m=50)" in r.table


def test_wide_to_long_audits_each_column_as_its_own_series():
    rng = np.random.default_rng(10)
    wide = pd.DataFrame({"od_a": np.arange(100.0), "od_b": 1000 + rng.normal(size=100)})
    wide["split"] = np.where(np.arange(100) < 80, "train", "test")
    long = wide_to_long(wide, keep=["split"])
    assert set(long["series"]) == {"od_a", "od_b"} and len(long) == 200
    r = audit(long, "value", task="regression", groups="series", time="step", split="split")
    # persistence must not carry od_a's last value into od_b's first row
    assert r.n_eval == 40


def test_input_validation():
    df = frame_from_series(sticky_labels(100))
    with pytest.raises(ValueError, match="columns not found"):
        audit(df, "missing")
    df["target"] = df["target"] * 3
    with pytest.raises(ValueError, match="binary"):
        audit(df, "target")


def test_cli_exit_codes(tmp_path, capsys):
    df = frame_from_series(sticky_labels())
    df["pred"] = df["y"]
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    assert main([str(path), "--target", "target", "--pred", "pred", "--time", "t", "--n-boot", "50"]) == 1
    assert "VERDICT: FAIL" in capsys.readouterr().out
    assert main([str(path), "--target", "target", "--fail-on", "never", "--format", "json"]) == 0
