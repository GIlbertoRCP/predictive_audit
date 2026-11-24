"""Tests for metrics, prefix-sum bootstrap equivalence, DM-HAC, and FDR control."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from predictive_audit import metrics


def _reference_bootstrap_diff(
    metric: str,
    y_true: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    block: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Slow, explicit slice-and-compute reference circular block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    b_len = max(1, min(block, n))
    n_blocks = int(np.ceil(n / b_len))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(b_len)) % n).ravel()[:n]
        yt = y_true[idx]
        diffs[i] = metrics.score(metric, yt, model[idx]) - metrics.score(metric, yt, baseline[idx])
    diffs = diffs[~np.isnan(diffs)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def test_compute_matches_sklearn_exact():
    rng = np.random.default_rng(0)
    y_bin = (rng.random(500) < 0.35).astype(float)
    p_score = np.clip(y_bin * 0.6 + rng.random(500) * 0.4, 0.0, 1.0)
    p_lbl = (p_score >= 0.5).astype(int)

    assert metrics.compute("f1", y_bin, p_score) == pytest.approx(skm.f1_score(y_bin, p_lbl, zero_division=0))
    assert metrics.compute("accuracy", y_bin, p_score) == pytest.approx(skm.accuracy_score(y_bin, p_lbl))
    assert metrics.compute("balanced_accuracy", y_bin, p_score) == pytest.approx(
        skm.balanced_accuracy_score(y_bin, p_lbl)
    )
    assert metrics.compute("mcc", y_bin, p_score) == pytest.approx(skm.matthews_corrcoef(y_bin, p_lbl))
    assert metrics.compute("brier", y_bin, p_score) == pytest.approx(skm.brier_score_loss(y_bin, p_score))
    assert metrics.compute("roc_auc", y_bin, p_score) == pytest.approx(skm.roc_auc_score(y_bin, p_score))
    assert metrics.compute("pr_auc", y_bin, p_score) == pytest.approx(
        skm.average_precision_score(y_bin, p_score)
    )

    y_reg = rng.normal(10.0, 3.0, size=500)
    p_reg = y_reg + rng.normal(0.0, 1.0, size=500)
    assert metrics.compute("r2", y_reg, p_reg) == pytest.approx(skm.r2_score(y_reg, p_reg))
    assert metrics.compute("mae", y_reg, p_reg) == pytest.approx(skm.mean_absolute_error(y_reg, p_reg))
    assert metrics.compute("rmse", y_reg, p_reg) == pytest.approx(skm.root_mean_squared_error(y_reg, p_reg))
    assert metrics.compute("smape", np.zeros(10), np.zeros(10)) == 0.0

    with pytest.raises(ValueError, match="unknown metric"):
        metrics.compute("nonexistent", y_reg, p_reg)


@pytest.mark.parametrize("metric_name", ["mae", "rmse", "r2", "smape"])
def test_prefix_sum_bootstrap_matches_reference_regression(metric_name: str):
    rng = np.random.default_rng(7)
    n = 317  # non-multiple of block length to test partial final block
    y = rng.normal(5.0, 2.0, size=n).cumsum()
    m = y + rng.normal(0.0, 0.4, size=n)
    b = y + rng.normal(0.0, 0.6, size=n)

    fast_lo, fast_hi = metrics.block_bootstrap_diff(metric_name, y, m, b, block=17, n_boot=120, seed=42)
    ref_lo, ref_hi = _reference_bootstrap_diff(metric_name, y, m, b, block=17, n_boot=120, seed=42)
    assert fast_lo == pytest.approx(ref_lo, abs=1e-10)
    assert fast_hi == pytest.approx(ref_hi, abs=1e-10)


@pytest.mark.parametrize("metric_name", ["f1", "accuracy", "balanced_accuracy", "mcc", "brier"])
def test_prefix_sum_bootstrap_matches_reference_classification(metric_name: str):
    rng = np.random.default_rng(11)
    n = 413
    y = (rng.random(n) < 0.3).astype(float)
    m = np.clip(y * 0.7 + rng.random(n) * 0.35, 0.0, 1.0)
    b = np.clip(y * 0.5 + rng.random(n) * 0.5, 0.0, 1.0)

    fast_lo, fast_hi = metrics.block_bootstrap_diff(metric_name, y, m, b, block=19, n_boot=120, seed=9)
    ref_lo, ref_hi = _reference_bootstrap_diff(metric_name, y, m, b, block=19, n_boot=120, seed=9)
    assert fast_lo == pytest.approx(ref_lo, abs=1e-10)
    assert fast_hi == pytest.approx(ref_hi, abs=1e-10)


def test_panel_bootstrap_block_length_uses_series_length_not_pooled_n():
    # 100 series x 50 timestamps = 5000 rows; default block must scale with T=50 (~7-10), not sqrt(5000)=71
    groups = np.repeat(np.arange(100), 50)
    times = np.tile(np.arange(50), 100)
    b_len = metrics.default_block_length(len(groups), groups=groups, times=times)
    assert 1 <= b_len <= 25


def test_unaligned_groups_never_cross_boundaries():
    # 5 short groups of length 15 each: every circular block stays strictly inside its own group
    groups = np.repeat(np.arange(5), 15)
    gather_idx, valid_starts = metrics._group_circular_layout(groups, pad=9)
    assert len(valid_starts) == len(groups)  # exact 1:1 uniform weight per row
    for s in valid_starts:
        block_groups = groups[gather_idx[s : s + 10]]
        assert np.all(block_groups == block_groups[0])


def test_diebold_mariano_hac_detects_skill_and_null():
    rng = np.random.default_rng(19)
    n = 1000
    y = rng.normal(size=n)
    good_model = y + rng.normal(scale=0.3, size=n)
    bad_base = y + rng.normal(scale=0.6, size=n)

    stat, p_val = metrics.diebold_mariano("mae", y, good_model, bad_base, horizon=1)
    assert stat > 5.0 and p_val < 1e-6

    # Reversed order gives negative stat with identical p-value
    stat_rev, p_rev = metrics.diebold_mariano("mae", y, bad_base, good_model, horizon=1)
    assert stat_rev == pytest.approx(-stat)
    assert p_rev == pytest.approx(p_val)

    # Identical forecasts -> stat=0, p=1
    stat0, p0 = metrics.diebold_mariano("mae", y, good_model, good_model)
    assert stat0 == 0.0 and p0 == 1.0

    # Non-additive metric -> (nan, nan)
    stat_nan, p_nan = metrics.diebold_mariano("roc_auc", (y > 0).astype(float), good_model, bad_base)
    assert np.isnan(stat_nan) and np.isnan(p_nan)


def test_multiple_testing_corrections():
    pvals = [0.001, 0.04, 0.03, 0.5, float("nan")]
    q = metrics.benjamini_hochberg(pvals)
    assert np.isnan(q[4])
    assert q[0] == pytest.approx(0.004)
    assert q[2] == pytest.approx(0.0533333333, rel=1e-5)
    assert q[1] == pytest.approx(0.0533333333, rel=1e-5)
    assert q[3] == pytest.approx(0.5)

    h = metrics.holm_bonferroni(pvals)
    assert np.isnan(h[4])
    assert h[0] == pytest.approx(0.004)
    assert h[2] == pytest.approx(0.09)
    assert h[1] == pytest.approx(0.09)
    assert h[3] == pytest.approx(0.5)


def test_optimal_threshold_improves_skewed_f1():
    rng = np.random.default_rng(21)
    y = (rng.random(1000) < 0.08).astype(float)
    # Well-separated probabilities, but scaled below 0.5 so default 0.5 gets F1=0
    score = np.where(y == 1.0, rng.uniform(0.25, 0.45, size=1000), rng.uniform(0.0, 0.15, size=1000))
    assert metrics.compute("f1", y, score, threshold=0.5) == 0.0
    thr = metrics.optimal_threshold("f1", y, score)
    assert 0.14 <= thr <= 0.30
    assert metrics.compute("f1", y, score, threshold=thr) > 0.95
