"""Hypothesis property-based tests and thorough edge-case coverage across all modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predictive_audit import audit, baselines, batch_audit, metrics, report, wide_to_long
from predictive_audit.cli import main

# --- Hypothesis property-based tests ------------------------------------------


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=25, max_value=120),
    horizon=st.integers(min_value=1, max_value=5),
    block=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_prefix_bootstrap_exact_for_mae(n: int, horizon: int, block: int, seed: int):
    """Prefix-sum CBB must match explicit slice-and-compute CBB for any (n, block, seed)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    m = y + rng.normal(scale=0.3, size=n)
    b = y + rng.normal(scale=0.5, size=n)

    fast_lo, fast_hi = metrics.block_bootstrap_diff("mae", y, m, b, block=block, n_boot=40, seed=seed)

    ref_rng = np.random.default_rng(seed)
    b_len = max(1, min(block, n))
    n_blocks = int(np.ceil(n / b_len))
    diffs = np.empty(40)
    for i in range(40):
        starts = ref_rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(b_len)) % n).ravel()[:n]
        diffs[i] = metrics.score("mae", y[idx], m[idx]) - metrics.score("mae", y[idx], b[idx])
    assert fast_lo == pytest.approx(float(np.percentile(diffs, 2.5)), abs=1e-10)
    assert fast_hi == pytest.approx(float(np.percentile(diffs, 97.5)), abs=1e-10)


@settings(max_examples=25, deadline=None)
@given(
    n_groups=st.integers(min_value=2, max_value=5),
    length=st.integers(min_value=20, max_value=45),
    horizon=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=5_000),
)
def test_property_baselines_strictly_causal_under_future_perturbation(
    n_groups: int, length: int, horizon: int, seed: int
):
    """Altering future test targets must never leak backward into earlier baseline predictions."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), length)
    target = pd.Series(rng.normal(size=n_groups * length))
    g_ser = pd.Series(groups)
    fit_mask = pd.Series(np.tile(np.arange(length) < length - 5, n_groups))

    base1 = baselines.all_baselines(
        target, g_ser, horizon, "regression", fit_mask, include_ewma=True, ar_lags=2
    )
    # Perturb the final 5 rows of every group (test period)
    perturbed = target.copy()
    perturbed[~fit_mask] += 1e5
    base2 = baselines.all_baselines(
        perturbed, g_ser, horizon, "regression", fit_mask, include_ewma=True, ar_lags=2
    )
    # Every row in the train period AND the very first test row must be bit-identical
    check_rows = fit_mask.to_numpy().copy()
    for g in range(n_groups):
        check_rows[g * length + (length - 5)] = True
    for name in base1:
        np.testing.assert_allclose(
            base1[name].to_numpy()[check_rows],
            base2[name].to_numpy()[check_rows],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )


@settings(max_examples=25, deadline=None)
@given(
    pvals=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=30),
)
def test_property_fdr_and_holm_bounds(pvals: list[float]):
    q = metrics.benjamini_hochberg(pvals)
    h = metrics.holm_bonferroni(pvals)
    p = np.asarray(pvals)
    assert np.all((q >= p - 1e-12) & (q <= 1.0 + 1e-12))
    assert np.all((h >= q - 1e-12) & (h <= 1.0 + 1e-12))


# --- Politis-White adaptive block length & rank/panel bootstrap coverage ------


def test_politis_white_scales_with_autocorrelation():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=1000)
    ar_high = np.zeros(1000)
    eps = rng.normal(size=1000)
    for t in range(1, 1000):
        ar_high[t] = 0.92 * ar_high[t - 1] + eps[t]

    b_iid = metrics.estimate_block_length(iid)
    b_high = metrics.estimate_block_length(ar_high)
    assert b_high > b_iid
    assert metrics.estimate_block_length(np.zeros(100)) >= 1
    assert metrics.estimate_block_length(np.array([1.0, 2.0, 3.0])) >= 1


def test_rank_metrics_and_panel_bootstrap_paths():
    rng = np.random.default_rng(42)
    n_groups, n_steps = 4, 40
    groups = np.repeat(np.arange(n_groups), n_steps)
    times = np.tile(np.arange(n_steps), n_groups)
    y = (rng.random(n_groups * n_steps) < 0.4).astype(float)
    m = np.clip(y * 0.7 + rng.random(len(y)) * 0.3, 0.0, 1.0)
    b = np.clip(y * 0.3 + rng.random(len(y)) * 0.7, 0.0, 1.0)

    # Aligned panel rank-metric bootstrap (roc_auc)
    lo, hi = metrics.block_bootstrap_diff("roc_auc", y, m, b, 6, 30, 0, groups=groups, times=times)
    assert lo > 0.0 and hi > lo

    # Unaligned groups rank-metric bootstrap (pr_auc)
    lo2, hi2 = metrics.block_bootstrap_diff("pr_auc", y, m, b, 6, 30, 0, groups=groups)
    assert lo2 > 0.0 and hi2 > lo2

    # Single-series rank-metric bootstrap
    lo3, hi3 = metrics.block_bootstrap_diff("roc_auc", y, m, b, 6, 30, 0)
    assert lo3 > 0.0 and hi3 > lo3

    # Empty / degenerate bootstrap
    empty = np.array([])
    assert np.all(np.isnan(metrics.block_bootstrap_diff("mae", empty, empty, empty, 5, 10, 0)))

    # Panel Diebold-Mariano across all additive metrics
    for metric_name in ("mae", "rmse", "brier", "smape", "accuracy"):
        stat, p_val = metrics.diebold_mariano(metric_name, y, m, b, groups=groups, times=times)
        assert np.isfinite(stat) and 0.0 <= p_val <= 1.0

    # Single-class balanced_accuracy & r2 constant target edge cases
    assert metrics.compute("balanced_accuracy", np.ones(10), np.ones(10)) == 1.0
    assert metrics.compute("balanced_accuracy", np.zeros(10), np.zeros(10)) == 1.0
    assert metrics.compute("r2", np.ones(10), np.ones(10)) == 1.0
    assert metrics.compute("r2", np.ones(10), np.zeros(10)) == 0.0
    assert metrics.optimal_threshold("roc_auc", y, m) == 0.5
    assert metrics.optimal_threshold("f1", y[:5], m[:5]) == 0.5
    assert metrics.optimal_threshold("f1", y, np.round(y)) == 0.5
    assert np.all(np.isnan(metrics.benjamini_hochberg([float("nan")])))
    assert np.all(np.isnan(metrics.holm_bonferroni([float("nan")])))


def test_audit_edge_cases_and_cli_options(tmp_path):
    rng = np.random.default_rng(7)
    n = 300
    y = rng.normal(size=n).cumsum()
    df = pd.DataFrame({
        "t": np.arange(n - 1),
        "target": y[1:],
        "pred": y[:-1] + rng.normal(scale=0.2, size=n - 1),
        "split": ["train"] * 100 + ["test"] * 50 + ["train"] * 100 + ["test"] * (n - 251),
    })
    # Blocked CV (2 test segments) -> WARN
    r_cv = audit(
        df, "target", task="regression", pred="pred", time="t", split="split", block="auto", n_boot=40
    )
    assert {c.name: c for c in r_cv.checks}["temporal split"].status == "WARN"
    assert {c.name: c for c in r_cv.checks}["beats trivial baselines"].details["block_length"] >= 1

    # Reported score WARN (marginal) and PASS (clear)
    persist_mae = r_cv.table["persistence"]["mae"]
    r_warn = audit(df, "target", task="regression", split="split", reported=persist_mae * 0.98, metric="mae")
    r_pass = audit(df, "target", task="regression", split="split", reported=persist_mae * 0.50, metric="mae")
    assert {c.name: c for c in r_warn.checks}["reported score"].status == "WARN"
    assert {c.name: c for c in r_pass.checks}["reported score"].status == "PASS"

    # Reported score INFO when metric is undefined (single-class pr_auc)
    df_one = pd.DataFrame({"target": np.zeros(50), "t": np.arange(50)})
    r_info = audit(df_one, "target", task="classification", metric="pr_auc", reported=0.5)
    assert {c.name: c for c in r_info.checks}["reported score"].status == "INFO"

    # Validation errors
    with pytest.raises(ValueError, match="task must be"):
        audit(df, "target", task="ranking")
    with pytest.raises(ValueError, match="horizon must be"):
        audit(df, "target", horizon=0)
    with pytest.raises(ValueError, match="not valid"):
        audit(df, "target", task="regression", metric="f1")
    with pytest.raises(ValueError, match="season must be"):
        audit(df, "target", task="regression", season="weekly")
    with pytest.raises(ValueError, match="no rows with"):
        audit(df.assign(sp="train"), "target", task="regression", split="sp")
    with pytest.raises(ValueError, match="only .* rows"):
        audit(df.iloc[:5], "target", task="regression")
    with pytest.raises(ValueError, match="columns not found"):
        batch_audit(df, "target", group_col="missing")
    with pytest.raises(ValueError, match="no value columns"):
        wide_to_long(pd.DataFrame({"split": ["train", "test"]}), keep=["split"])
    with pytest.raises(ValueError, match="ar_lags"):
        baselines.linear_ar(df["target"], None, 1, pd.Series(True, index=df.index), lags=0)

    # Classification threshold="auto" + EWMA with groups + Ridge AR with small/large groups
    g_df = pd.DataFrame({
        "g": ["big"] * 80 + ["tiny"] * 12,
        "t": list(range(80)) + list(range(12)),
        "target": (rng.random(92) < 0.3).astype(float),
        "split": ["train"] * 50 + ["test"] * 30 + ["train"] * 4 + ["test"] * 8,
    })
    g_df["pred"] = np.where(
        g_df["target"] == 1.0,
        rng.uniform(0.25, 0.45, size=92),
        rng.uniform(0.0, 0.15, size=92),
    )
    r_cls = audit(
        g_df,
        "target",
        task="classification",
        pred="pred",
        groups="g",
        time="t",
        split="split",
        threshold="auto",
        include_ewma=True,
        ar_lags=2,
        n_boot=30,
    )
    assert r_cls.threshold is not None and r_cls.threshold < 0.5
    assert "classification threshold:" in report.to_text(r_cls)
    assert report._sanitize_json({"arr": np.array([1.0, np.nan]), "flag": np.bool_(True)}) == {
        "arr": [1.0, None],
        "flag": True,
    }

    # batch_audit without pred -> INFO verdict
    br_nopred = batch_audit(
        g_df[g_df["g"] == "big"], "target", group_col="g", task="classification", time="t"
    )
    assert br_nopred.verdict in ("INFO", "WARN")

    # CLI flags: --block auto, --threshold auto, --season auto, --fail-on warn, --format markdown
    csv_path = tmp_path / "cls.csv"
    g_df.to_csv(csv_path, index=False)
    code = main([
        str(csv_path),
        "--target", "target",
        "--pred", "pred",
        "--group", "g",
        "--time", "t",
        "--split", "split",
        "--threshold", "auto",
        "--block", "auto",
        "--season", "auto",
        "--format", "markdown",
        "--fail-on", "warn",
        "--n-boot", "30",
    ])
    assert code in (0, 1)

    # CLI bad TOML table
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text('audit = "not_a_table"\n', encoding="utf-8")
    assert main([str(csv_path), "--config", str(bad_toml), "--target", "target"]) == 2
