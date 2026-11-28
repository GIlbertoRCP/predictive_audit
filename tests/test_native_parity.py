"""Native C++ kernels vs the pure-Python reference: same inputs, same answers.

Bootstrap intervals, metrics and block lengths must match bit for bit (the native engine
reproduces NumPy's PCG64 draws and pairwise-summation order). Diebold-Mariano may differ in the
last few ulps because NumPy's dot products go through BLAS.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from predictive_audit import _backend, metrics

pytestmark = pytest.mark.skipif(not _backend.native_available(), reason="native extension not built")

PREFIX_METRICS = sorted(metrics._PREFIX_METRICS)
REGRESSION = {"mae", "rmse", "r2", "smape"}


def _both(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    with _backend.use("native"):
        native = fn(*args, **kwargs)
    with _backend.use("python"):
        python = fn(*args, **kwargs)
    return native, python


def _same(a: Any, b: Any) -> bool:
    """Bitwise float equality that treats NaN == NaN."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a_arr.shape == b_arr.shape and bool(np.all((a_arr == b_arr) | (np.isnan(a_arr) & np.isnan(b_arr))))


def _series(seed: int, n: int, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if metric in REGRESSION:
        y = rng.normal(size=n).cumsum()
        return y, y + rng.normal(scale=0.3, size=n), y + rng.normal(scale=0.5, size=n)
    y = (rng.random(n) < 0.3).astype(float)
    return (
        y,
        np.clip(y * 0.6 + rng.random(n) * 0.5, 0.0, 1.0),
        np.clip(y * 0.4 + rng.random(n) * 0.6, 0.0, 1.0),
    )


def _panel(n: int, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.repeat(np.arange(n_groups), -(-n // n_groups))[:n]
    times = np.concatenate([np.arange(int((groups == g).sum())) for g in range(n_groups)])
    return groups, times


def test_rng_reproduces_numpy_draws() -> None:
    from predictive_audit import _native

    for seed, high, size in [(0, 317, 1000), (42, 2, 257), (2**63 + 5, 10**6, 513), (7, 2**32, 64)]:
        expected = np.random.default_rng(seed).integers(0, high, size=size)
        np.testing.assert_array_equal(_native.rng_integers(seed, high, size), expected)


@settings(max_examples=60, deadline=None)
@given(
    metric=st.sampled_from(PREFIX_METRICS),
    n=st.integers(12, 900),
    seed=st.integers(0, 2**32),
    block=st.integers(-3, 60),
    n_groups=st.integers(1, 5),
    mode=st.sampled_from(["plain", "groups", "panel_int", "panel_float", "panel_datetime"]),
    threshold=st.sampled_from([0.5, 0.3]),
)
def test_bootstrap_is_bit_identical(
    metric: str, n: int, seed: int, block: int, n_groups: int, mode: str, threshold: float
) -> None:
    y, m, b = _series(seed, n, metric)
    groups, times = _panel(n, n_groups)
    kwargs: dict[str, Any] = {"threshold": threshold}
    if mode != "plain":
        kwargs["groups"] = groups
    if mode == "panel_int":
        kwargs["times"] = times
    elif mode == "panel_float":
        kwargs["times"] = times.astype(float) * 0.5
    elif mode == "panel_datetime":
        kwargs["times"] = np.datetime64("2026-01-01") + times.astype("timedelta64[h]")
    native, python = _both(metrics.block_bootstrap_diff, metric, y, m, b, block, 50, seed, **kwargs)
    assert _same(native, python), (native, python)


@settings(max_examples=60, deadline=None)
@given(metric=st.sampled_from(PREFIX_METRICS), n=st.integers(1, 1500), seed=st.integers(0, 2**32))
def test_compute_is_bit_identical(metric: str, n: int, seed: int) -> None:
    y, m, _ = _series(seed, n, metric)
    for threshold in (0.5, 0.25):
        native, python = _both(metrics.compute, metric, y, m, threshold=threshold)
        if metric == "r2" and np.isfinite(python):
            # The reference's total sum of squares is np.dot, i.e. BLAS, whose summation order
            # (and so its last bit) differs between OpenBLAS and Accelerate.
            assert native == pytest.approx(python, rel=1e-13, abs=1e-15)
        else:
            assert _same(native, python)


def test_compute_handles_nan_and_degenerate_inputs() -> None:
    y = np.array([1.0, np.nan, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.2, np.nan, 0.1, 0.0])
    for metric in PREFIX_METRICS:
        assert _same(*_both(metrics.compute, metric, y, p))
        assert _same(*_both(metrics.compute, metric, np.zeros(4), np.zeros(4)))


@settings(max_examples=40, deadline=None)
@given(n=st.integers(0, 3000), seed=st.integers(0, 2**32), phi=st.floats(0.0, 0.99))
def test_block_lengths_are_identical(n: int, seed: int, phi: float) -> None:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    assert _same(*_both(metrics.estimate_block_length, x))
    assert _same(*_both(metrics.estimate_block_length, x, max_block=7))
    if n:
        groups, times = _panel(n, 3)
        for kwargs in ({}, {"groups": groups}, {"groups": groups, "times": times}):
            assert _same(*_both(metrics.default_block_length, n, **kwargs))


@settings(max_examples=40, deadline=None)
@given(n=st.integers(20, 400), seed=st.integers(0, 2**32), k=st.integers(1, 6))
def test_block_length_of_sparse_series_is_identical(n: int, seed: int, k: int) -> None:
    # A loss differential that is mostly 0 (e.g. two classifiers that usually agree): its true
    # autocovariances are exactly 0, and the FFT returns platform-dependent rounding noise.
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[rng.choice(n, size=min(k, n), replace=False)] = rng.choice([-1.0, 1.0], size=min(k, n))
    assert _same(*_both(metrics.estimate_block_length, x))


@settings(max_examples=40, deadline=None)
@given(
    metric=st.sampled_from(PREFIX_METRICS),
    n=st.integers(3, 1200),
    seed=st.integers(0, 2**32),
    horizon=st.integers(1, 12),
    panel=st.booleans(),
)
def test_diebold_mariano_matches(metric: str, n: int, seed: int, horizon: int, panel: bool) -> None:
    y, m, b = _series(seed, n, metric)
    kwargs: dict[str, Any] = {"horizon": horizon}
    if panel:
        kwargs["groups"], kwargs["times"] = _panel(n, 4)
    (stat_n, p_n), (stat_p, p_p) = _both(metrics.diebold_mariano, metric, y, m, b, **kwargs)
    if np.isnan(stat_p):
        assert np.isnan(stat_n) and np.isnan(p_n)
        return
    assert stat_n == pytest.approx(stat_p, rel=1e-9, abs=1e-12)
    assert p_n == pytest.approx(p_p, rel=1e-8, abs=1e-15)


def test_student_t_tail_matches_scipy() -> None:
    from predictive_audit import _native

    for df in [1, 2, 3, 4, 7, 10, 30, 99, 500, 4_999, 99_999, 1_000_000]:
        for t in [0.0, 1e-3, 0.1, 0.5, 1.0, 1.96, 2.5, 4.0, 8.0, 15.0, 40.0, 120.0]:
            ref = 2.0 * stats.t.sf(t, df)
            assert _native.student_t_two_sided_p(t, df) == pytest.approx(ref, rel=1e-8, abs=1e-300)


@settings(max_examples=40, deadline=None)
@given(p=st.lists(st.one_of(st.floats(0.0, 1.0), st.just(float("nan"))), min_size=0, max_size=40))
def test_fdr_corrections_match(p: list[float]) -> None:
    for fn in (metrics.benjamini_hochberg, metrics.holm_bonferroni):
        native, python = _both(fn, p)
        np.testing.assert_allclose(native, python, rtol=1e-15, atol=0.0, equal_nan=True)


def test_panel_detection_and_layout_match_python_helpers() -> None:
    from predictive_audit import _native

    groups, times = _panel(600, 4)
    aligned, inv, n_times = _native.is_aligned_panel(groups, times_i64=times)
    ref_aligned, ref_inv, ref_n = metrics._is_aligned_panel(groups, times)
    assert (aligned, n_times) == (ref_aligned, ref_n)
    np.testing.assert_array_equal(inv, ref_inv)

    gather, starts = _native.group_circular_layout(groups, 9)
    ref_gather, ref_starts = metrics._group_circular_layout(groups, 9)
    np.testing.assert_array_equal(gather, ref_gather)
    np.testing.assert_array_equal(starts, ref_starts)


def test_unsupported_inputs_fall_back_to_python() -> None:
    y, m, b = _series(0, 200, "mae")
    groups = np.array(["a"] * 100 + ["b"] * 100, dtype=object)  # non-integer codes -> Python path
    native, python = _both(metrics.block_bootstrap_diff, "mae", y, m, b, 10, 40, 1, groups=groups)
    assert _same(native, python)
    # Rank metrics stay in Python for now; results are unchanged.
    yb, mb, bb = _series(0, 200, "f1")
    assert _same(*_both(metrics.block_bootstrap_diff, "roc_auc", yb, mb, bb, 10, 20, 1))


def test_results_do_not_depend_on_thread_count() -> None:
    from predictive_audit import _native

    y, m, b = _series(5, 20_000, "f1")
    before = _native.num_threads()
    try:
        _native.set_num_threads(1)
        with _backend.use("native"):
            one = metrics.block_bootstrap_diff("f1", y, m, b, 37, 400, 9)
        _native.set_num_threads(8)
        with _backend.use("native"):
            many = metrics.block_bootstrap_diff("f1", y, m, b, 37, 400, 9)
    finally:
        _native.set_num_threads(0)
    assert one == many
    assert _native.num_threads() == before


def _bootstrap_in_child(conn: Any, expected: tuple[float, float]) -> None:  # pragma: no cover - child process
    y, m, b = _series(5, 20_000, "f1")
    with _backend.use("native"):
        conn.send(metrics.block_bootstrap_diff("f1", y, m, b, 37, 200, 9) == expected)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX-only")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # Python 3.12+ warns about fork() with threads
def test_pool_survives_fork_during_parallel_work() -> None:
    """A child forked while another thread is inside a native bootstrap must not inherit a locked pool."""
    import multiprocessing as mp
    import threading
    import time

    from predictive_audit import _native

    y, m, b = _series(5, 20_000, "f1")
    with _backend.use("native"):
        expected = metrics.block_bootstrap_diff("f1", y, m, b, 37, 200, 9)
    # Small setup, long parallel phase: the busy thread spends nearly all its time inside the pool.
    big_y, big_m, big_b = _series(6, 200_000, "f1")
    _native.set_num_threads(4)
    stop = threading.Event()

    def keep_pool_busy() -> None:
        with _backend.use("native"):
            while not stop.is_set():
                metrics.block_bootstrap_diff("f1", big_y, big_m, big_b, 5, 2_000, 1)

    busy = threading.Thread(target=keep_pool_busy, daemon=True)
    busy.start()
    try:
        ctx = mp.get_context("fork")
        for _ in range(8):  # fork repeatedly while the pool is mid-job in the other thread
            time.sleep(0.15)
            parent_conn, child_conn = ctx.Pipe()
            child = ctx.Process(target=_bootstrap_in_child, args=(child_conn, expected))
            child.start()
            child.join(timeout=30)
            if child.is_alive():
                child.kill()
                pytest.fail("forked child deadlocked in the native thread pool")
            assert child.exitcode == 0
            assert parent_conn.recv() is True
    finally:
        stop.set()
        busy.join()
        _native.set_num_threads(0)
