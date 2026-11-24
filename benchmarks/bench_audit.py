"""End-to-end latency, throughput, and peak memory benchmarks for predictive-audit.

Measures:
  1. Single-series regression (MAE) and classification (F1) audit scaling across 10k, 100k, 500k rows.
  2. Wide-to-long reshaping + multi-series panel audit on a 500 x 2,000 (1M cell) traffic matrix.

Usage:
    uv run python benchmarks/bench_audit.py [--out benchmarks/baseline_audit.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_audit import audit, wide_to_long


def bench_single_series(
    sizes: tuple[int, ...] = (10_000, 100_000, 500_000),
    n_boot: int = 500,
    runs: int = 5,
) -> list[dict]:
    rng = np.random.default_rng(42)
    results = []
    for task, metric in [("regression", "mae"), ("classification", "f1")]:
        for n in sizes:
            if task == "regression":
                y = np.cumsum(rng.normal(size=n))
                pred = y + rng.normal(scale=0.1, size=n)
            else:
                flips = rng.random(n) < 0.02
                y = (np.cumsum(flips) % 2).astype(float)
                pred = np.clip(y + rng.normal(scale=0.15, size=n), 0.0, 1.0)

            df = pd.DataFrame({
                "t": np.arange(n),
                "target": np.roll(y, -1),
                "pred": pred,
                "split": np.where(np.arange(n) < int(0.8 * n), "train", "test"),
            }).iloc[:-1]

            # Warmup run
            audit(
                df, "target", task=task, pred="pred", time="t",
                split="split", metric=metric, n_boot=min(20, n_boot), seed=0,
            )

            times_ms: list[float] = []
            peak_mb = 0.0
            for _ in range(runs):
                tracemalloc.start()
                t0 = time.perf_counter()
                audit(
                    df, "target", task=task, pred="pred", time="t",
                    split="split", metric=metric, n_boot=n_boot, seed=0,
                )
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = max(peak_mb, peak / (1024 ** 2))

            arr = np.asarray(times_ms)
            p50 = float(np.percentile(arr, 50))
            results.append({
                "task": task,
                "metric": metric,
                "n_rows": n,
                "n_boot": n_boot,
                "runs": runs,
                "p50_ms": round(p50, 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
                "throughput_rows_s": int(n / (p50 / 1000.0)),
                "peak_mem_mb": round(peak_mb, 2),
            })
    return results


def bench_wide_panel(n_series: int = 500, n_steps: int = 2000, n_boot: int = 500) -> dict:
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(
        rng.normal(size=(n_steps, n_series)).cumsum(axis=0),
        columns=[f"od_{i}" for i in range(n_series)],
    )
    wide["split"] = np.where(np.arange(n_steps) < int(0.8 * n_steps), "train", "test")

    tracemalloc.start()
    t0 = time.perf_counter()
    long = wide_to_long(wide, keep=["split"])
    t_melt = time.perf_counter() - t0
    _, peak_melt = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    long["pred"] = long["value"].to_numpy() + rng.normal(scale=0.1, size=len(long))

    tracemalloc.start()
    t1 = time.perf_counter()
    audit(
        long, "value", task="regression", pred="pred", groups="series",
        time="step", split="split", metric="mae", n_boot=n_boot, seed=0,
    )
    t_audit = time.perf_counter() - t1
    _, peak_audit = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n_series": n_series,
        "n_steps": n_steps,
        "cells": n_series * n_steps,
        "n_boot": n_boot,
        "wide_to_long_ms": round(t_melt * 1000.0, 2),
        "wide_to_long_peak_mb": round(peak_melt / (1024 ** 2), 2),
        "panel_audit_ms": round(t_audit * 1000.0, 2),
        "panel_audit_peak_mb": round(peak_audit / (1024 ** 2), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="Optional JSON output path")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    payload = {
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "single_series": bench_single_series(runs=args.runs),
        "wide_panel_1m": bench_wide_panel(),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
