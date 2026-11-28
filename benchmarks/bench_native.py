"""Python reference vs native C++ kernels: latency, end-to-end audit, and thread scaling.

Every native result is checked against the Python reference before it is timed (bootstrap
intervals must be bit-identical), so the speedups compare equal work.

Usage:
    uv run python benchmarks/bench_native.py [--runs 5] [--out benchmarks/native_kernels.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predictive_audit import _backend, audit, baselines, batch_audit, metrics

if not _backend.native_available():
    raise SystemExit("native extension not built; run `uv sync` first")

from predictive_audit import _native  # noqa: E402


def _timed(fn: Callable[[], Any], runs: int) -> tuple[float, Any]:
    fn()  # warm-up
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), result


def _compare(name: str, fn: Callable[[], Any], runs: int, *, exact: bool = True) -> dict[str, Any]:
    with _backend.use("python"):
        py_ms, py_res = _timed(fn, runs)
    with _backend.use("native"):
        nat_ms, nat_res = _timed(fn, runs)
    if exact:
        assert np.array_equal(np.asarray(py_res, float), np.asarray(nat_res, float), equal_nan=True), name
    row = {
        "workload": name,
        "python_ms": round(py_ms, 3),
        "native_ms": round(nat_ms, 3),
        "speedup": round(py_ms / nat_ms, 1),
    }
    print(f"{name:<52} {py_ms:>10.2f} ms {nat_ms:>10.2f} ms {py_ms / nat_ms:>7.1f}x")
    return row


def _regression(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.cumsum(rng.normal(size=n))
    return y, y + rng.normal(scale=0.1, size=n), np.roll(y, 1)


def _classification(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = (np.cumsum(rng.random(n) < 0.02) % 2).astype(float)
    m = np.clip(y + rng.normal(scale=0.15, size=n), 0.0, 1.0)
    return y, m, np.roll(y, 1)


def bench_kernels(runs: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    rows = []
    for n in (10_000, 100_000, 500_000):
        y, m, b = _regression(n, rng)
        block = metrics.default_block_length(n)
        boot = partial(metrics.block_bootstrap_diff, "mae", y, m, b, block, 500, 0)
        rows.append(_compare(f"bootstrap mae, {n:,} rows, 500 draws", boot, runs))
        yc, mc, bc = _classification(n, rng)
        boot = partial(metrics.block_bootstrap_diff, "f1", yc, mc, bc, block, 500, 0)
        rows.append(_compare(f"bootstrap f1, {n:,} rows, 500 draws", boot, runs))
    # Aligned panel: 500 series x 2,000 timestamps (1M rows)
    g, t = np.repeat(np.arange(500), 2000), np.tile(np.arange(2000), 500)
    y, m, b = _regression(len(g), rng)
    boot = partial(metrics.block_bootstrap_diff, "mae", y, m, b, 45, 500, 0, groups=g, times=t)
    rows.append(_compare("bootstrap mae, 500 x 2,000 aligned panel", boot, runs))
    y, m, b = _regression(100_000, rng)
    rows.append(_compare("Diebold-Mariano HAC, 100,000 rows",
                         lambda: metrics.diebold_mariano("mae", y, m, b, horizon=1), runs, exact=False))
    rows.append(_compare("Politis-White block length, 100,000 rows",
                         lambda: metrics.estimate_block_length(m - y), runs))
    return rows


def bench_phase2(runs: int) -> list[dict[str, Any]]:
    """Baselines, rank metrics and threshold search at Case 03 scale (785 series x 1,000 steps)."""
    rng = np.random.default_rng(5)
    n_series, steps = 785, 1_000
    g = pd.Series(np.repeat(np.arange(n_series), steps))
    t = np.tile(np.arange(steps), n_series)
    y = pd.Series(np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=len(g)).cumsum() * 0.01)
    fit = pd.Series(t < 800)
    slow = max(1, min(runs, 3))
    rows = [
        _compare("moving average (3), 785 series x 1,000",
                 partial(baselines.moving_average, y, g, 1, 3), runs),
        _compare("EWMA (3), 785 series x 1,000",
                 partial(baselines.ewma, y, g, 1, 3), runs, exact=False),
        _compare("Ridge AR(5) per series, 785 series x 1,000",
                 partial(baselines.linear_ar, y, g, 1, fit, lags=5), slow, exact=False),
        _compare("season estimate (FFT), 785 series x 1,000",
                 partial(baselines.estimate_season, y, g, fit), runs),
    ]
    yc, mc, bc = _classification(1_000_000, rng)
    rows.append(_compare("ROC-AUC, 1,000,000 rows", partial(metrics.compute, "roc_auc", yc, mc), runs))
    rows.append(_compare("optimal threshold (F1), 1,000,000 rows",
                         partial(metrics.optimal_threshold, "f1", yc, mc), slow))
    yc, mc, bc = _classification(20_000, rng)
    bc = np.clip(bc + rng.normal(scale=0.3, size=len(bc)), 0, 1)
    rows.append(_compare("bootstrap ROC-AUC, 20,000 rows, 500 draws",
                         partial(metrics.block_bootstrap_diff, "roc_auc", yc, mc, bc, 50, 500, 0), slow))
    rows.append(_compare("bootstrap PR-AUC, 20,000 rows, 500 draws",
                         partial(metrics.block_bootstrap_diff, "pr_auc", yc, mc, bc, 50, 500, 0), slow))
    return rows


def bench_audit(runs: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    rows = []
    for task, metric in (("regression", "mae"), ("classification", "f1")):
        for n in (10_000, 500_000):
            y, m, _ = _regression(n, rng) if task == "regression" else _classification(n, rng)
            df = pd.DataFrame({
                "t": np.arange(n),
                "target": np.roll(y, -1),
                "pred": m,
                "split": np.where(np.arange(n) < int(0.8 * n), "train", "test"),
            }).iloc[:-1]

            def run(df: pd.DataFrame = df, task: str = task, metric: str = metric) -> Any:
                r = audit(df, "target", task=task, pred="pred", time="t", split="split", metric=metric,
                          n_boot=500, seed=0)
                return r.checks[1].details["ci95"]

            rows.append(_compare(f"audit() {metric}, {n:,} rows (end to end)", run, runs))

    # Full-featured panel audit at Case 03 scale: every baseline, automatic season and threshold.
    n_series, steps = 785, 1_000
    t = np.tile(np.arange(steps), n_series)
    y = np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=len(t)).cumsum() * 0.01
    panel = pd.DataFrame({
        "od": np.repeat(np.arange(n_series), steps), "t": t, "y": np.roll(y, -1),
        "pred": y + rng.normal(scale=0.05, size=len(t)), "split": np.where(t < 800, "train", "test"),
    })

    def run_panel() -> Any:
        r = audit(panel, "y", task="regression", pred="pred", groups="od", time="t", split="split",
                  season="auto", include_ewma=True, ar_lags=5, n_boot=500, seed=0)
        return [r.verdict, *(c.status for c in r.checks)]

    with _backend.use("python"):
        py_ms, py_res = _timed(run_panel, max(1, min(runs, 3)))
    with _backend.use("native"):
        nat_ms, nat_res = _timed(run_panel, max(1, min(runs, 3)))
    assert py_res == nat_res, (py_res, nat_res)
    name = "audit() all baselines, 785 x 1,000 panel"
    print(f"{name:<52} {py_ms:>10.2f} ms {nat_ms:>10.2f} ms {py_ms / nat_ms:>7.1f}x")
    rows.append({"workload": name, "python_ms": round(py_ms, 3), "native_ms": round(nat_ms, 3),
                 "speedup": round(py_ms / nat_ms, 1)})

    def statuses(r: Any) -> list[str]:
        return [r.verdict, *(c.status for c in r.checks)]

    # Single series with a feature column (leakage check: Pearson + Spearman over 500k rows).
    n = 500_000
    yc, mc, _ = _classification(n, rng)
    leaky = pd.DataFrame({
        "t": np.arange(n), "target": np.roll(yc, -1), "pred": mc, "f1": rng.normal(size=n),
        "split": np.where(np.arange(n) < int(0.8 * n), "train", "test"),
    }).iloc[:-1]
    extra = [
        ("audit() F1 + feature leakage, 500,000 rows",
         lambda: statuses(audit(leaky, "target", task="classification", pred="pred", time="t", split="split",
                                metric="f1", features="f1", n_boot=500, seed=0))),
        ("batch_audit() 785 series x 1,000, every baseline",
         lambda: [r["status"] for r in batch_audit(panel, "y", group_col="od", task="regression", pred="pred",
                                                    time="t", split="split", season="auto", include_ewma=True,
                                                    ar_lags=5, n_boot=200, seed=0).rows]),
    ]
    for name, fn in extra:
        with _backend.use("python"):
            py_ms, py_res = _timed(fn, max(1, min(runs, 3)))
        with _backend.use("native"):
            nat_ms, nat_res = _timed(fn, max(1, min(runs, 3)))
        assert py_res == nat_res, name
        print(f"{name:<52} {py_ms:>10.2f} ms {nat_ms:>10.2f} ms {py_ms / nat_ms:>7.1f}x")
        rows.append({"workload": name, "python_ms": round(py_ms, 3), "native_ms": round(nat_ms, 3),
                     "speedup": round(py_ms / nat_ms, 1)})
    return rows


def bench_threads(runs: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(3)
    y, m, b = _classification(1_000_000, rng)
    rows = []
    hw = _native.build_info()["hardware_threads"]
    counts = sorted({1, 2, 4, 8, hw} - {0})
    base = None
    try:
        for k in counts:
            _native.set_num_threads(k)
            with _backend.use("native"):
                ms, _ = _timed(lambda: metrics.block_bootstrap_diff("f1", y, m, b, 20, 2000, 0), runs)
            base = base or ms
            rows.append({"threads": k, "native_ms": round(ms, 2), "scaling": round(base / ms, 2)})
            print(f"threads={k:<3} {ms:>9.2f} ms  {base / ms:>5.2f}x")
    finally:
        _native.set_num_threads(0)
    return rows


def _run_ms(cmd: list[str], runs: int, env: dict[str, str] | None = None) -> float:
    """Median wall time of a fresh process (cold start included), in ms."""
    import os
    import subprocess

    full_env = {**os.environ, **(env or {})}
    subprocess.run(cmd, env=full_env, capture_output=True, check=False)  # warm the file cache
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, env=full_env, capture_output=True, check=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def bench_cli(runs: int, binary: Path) -> list[dict[str, Any]]:
    """The standalone executable vs the Python console script, as separate processes."""
    import sys
    import tempfile

    rows: list[dict[str, Any]] = []
    if not binary.exists():
        print(f"(skipping CLI benchmarks: {binary} not built; run `make native`)")
        return rows
    rng = np.random.default_rng(4)
    tmp = Path(tempfile.mkdtemp())
    single = tmp / "single.csv"
    n = 10_000
    t = np.arange(n)
    y = np.sin(t / 30) + rng.normal(scale=0.3, size=n)
    pd.DataFrame({"t": t, "y": np.roll(y, -1), "pred": y + rng.normal(scale=0.2, size=n),
                  "split": np.where(t < 7000, "train", "test")}).to_csv(single, index=False)
    panel = tmp / "panel.csv"
    k, steps = 785, 1000
    tt = np.tile(np.arange(steps), k)
    yy = np.sin(tt / 24) + rng.normal(scale=0.3, size=k * steps)
    pd.DataFrame({
        "od": np.repeat(np.arange(k), steps), "t": tt, "y": yy,
        "pred": yy + rng.normal(scale=0.2, size=k * steps), "split": np.where(tt < 700, "train", "test"),
    }).to_csv(panel, index=False)

    py = [sys.executable, "-m", "predictive_audit.cli"]
    common = ["--task", "regression", "--target", "y", "--pred", "pred", "--time", "t", "--split", "split",
              "--fail-on", "never"]
    workloads = [
        ("startup: --version", ["--version"]),
        ("CLI audit, 10,000 rows", [str(single), *common]),
        ("CLI audit, 785 series x 1,000 steps (785k rows)", [str(panel), *common, "--group", "od"]),
    ]
    head = f"{'python':>13} {'package':>13} {'binary':>13} {'speedup':>8}"
    print(f"\n{'CLI (fresh process each run)':<52} {head}")
    for name, args in workloads:
        ref_ms = _run_ms([*py, *args], runs, {"PREDICTIVE_AUDIT_BACKEND": "python"})
        pkg_ms = _run_ms([*py, *args], runs, {"PREDICTIVE_AUDIT_BACKEND": "native"})
        bin_ms = _run_ms([str(binary), *args], runs)
        rows.append({
            "workload": name, "python_cli_ms": round(ref_ms, 1), "package_native_ms": round(pkg_ms, 1),
            "binary_ms": round(bin_ms, 1), "speedup": round(ref_ms / bin_ms, 1),
        })
        print(f"{name:<52} {ref_ms:>10.1f} ms {pkg_ms:>10.1f} ms {bin_ms:>10.1f} ms {ref_ms / bin_ms:>7.1f}x")

    text = panel.read_text(encoding="utf-8")
    pd_ms, _ = _timed(lambda: pd.read_csv(panel), runs)
    nat_ms, _ = _timed(lambda: _native.read_csv(text, ",", True), runs)
    rows.append({"workload": "read_csv, 785k rows x 5 columns", "python_ms": round(pd_ms, 1),
                 "native_ms": round(nat_ms, 1), "speedup": round(pd_ms / nat_ms, 1)})
    print(f"{'read_csv, 785k rows x 5 (pandas vs native)':<52} {pd_ms:>10.1f} ms {nat_ms:>10.1f} ms "
          f"{pd_ms / nat_ms:>7.1f}x")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/native_kernels.json"))
    parser.add_argument("--cli", type=Path, default=Path("build/native/predictive-audit"),
                        help="standalone executable to time (built by `make native`)")
    args = parser.parse_args()

    print(f"{'workload':<52} {'python':>13} {'native':>13} {'speedup':>8}")
    kernels = bench_kernels(args.runs)
    phase2 = bench_phase2(args.runs)
    end_to_end = bench_audit(args.runs)
    print("\nthread scaling: f1 bootstrap, 1,000,000 rows, block 20, 2,000 draws")
    threads = bench_threads(args.runs)
    cli = bench_cli(args.runs, args.cli)

    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native_build": _native.build_info(),
        "runs": args.runs,
        "kernels": kernels,
        "phase2_kernels": phase2,
        "audit": end_to_end,
        "thread_scaling": threads,
        "cli": cli,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
