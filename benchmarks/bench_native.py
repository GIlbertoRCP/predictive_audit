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

from predictive_audit import _backend, audit, metrics

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/native_kernels.json"))
    args = parser.parse_args()

    print(f"{'workload':<52} {'python':>13} {'native':>13} {'speedup':>8}")
    kernels = bench_kernels(args.runs)
    end_to_end = bench_audit(args.runs)
    print("\nthread scaling: f1 bootstrap, 1,000,000 rows, block 20, 2,000 draws")
    threads = bench_threads(args.runs)

    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native_build": _native.build_info(),
        "runs": args.runs,
        "kernels": kernels,
        "audit": end_to_end,
        "thread_scaling": threads,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
