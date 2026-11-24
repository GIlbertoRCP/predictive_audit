"""Monte Carlo Type I error calibration benchmark for autocorrelated series.

Under H0, the candidate model has the exact same expected loss as persistence
(difference in step-level loss is a zero-mean AR(1) process with parameter phi).
A well-calibrated 95% confidence interval should reject H0 at ~5% (nominal alpha=0.05).

Usage:
    uv run python benchmarks/bench_calibration.py [--out benchmarks/calibration.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predictive_audit.metrics import block_bootstrap_diff, diebold_mariano, estimate_block_length


def _ar1(phi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(size=n)
    if phi == 0.0:
        return eps
    out = np.empty(n, dtype=float)
    out[0] = eps[0]
    scale = np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + scale * eps[t]
    return out


def simulate_fpr(
    phi: float,
    *,
    n: int = 1000,
    n_trials: int = 400,
    n_boot: int = 400,
) -> dict[str, float]:
    """Empirical false-positive rate when E[|y - model|] == E[|y - baseline|] with AR(1) loss diffs."""
    rng = np.random.default_rng(12345)
    rej_iid = 0
    rej_b32 = 0
    rej_b64 = 0
    rej_pw = 0
    rej_dm = 0
    pw_blocks: list[int] = []
    y_true = np.zeros(n)
    for trial in range(n_trials):
        regime = _ar1(phi, n, rng)
        base_err = np.abs(rng.normal(scale=1.0, size=n))
        delta = 0.25 * np.tanh(regime)
        err_m = np.maximum(0.01, base_err - delta)
        err_b = np.maximum(0.01, base_err + delta)

        lo, hi = block_bootstrap_diff("mae", y_true, err_m, err_b, block=1, n_boot=n_boot, seed=trial)
        if lo > 0.0 or hi < 0.0:
            rej_iid += 1
        lo, hi = block_bootstrap_diff("mae", y_true, err_m, err_b, block=32, n_boot=n_boot, seed=trial)
        if lo > 0.0 or hi < 0.0:
            rej_b32 += 1
        lo, hi = block_bootstrap_diff("mae", y_true, err_m, err_b, block=64, n_boot=n_boot, seed=trial)
        if lo > 0.0 or hi < 0.0:
            rej_b64 += 1
        b_pw = estimate_block_length(err_b - err_m)
        pw_blocks.append(b_pw)
        lo, hi = block_bootstrap_diff("mae", y_true, err_m, err_b, block=b_pw, n_boot=n_boot, seed=trial)
        if lo > 0.0 or hi < 0.0:
            rej_pw += 1
        _, p_dm = diebold_mariano("mae", y_true, err_m, err_b, horizon=1)
        if p_dm < 0.05:
            rej_dm += 1
    return {
        "phi": phi,
        "iid_fpr": round(rej_iid / n_trials, 4),
        "moving_block_32_fpr": round(rej_b32 / n_trials, 4),
        "moving_block_64_fpr": round(rej_b64 / n_trials, 4),
        "politis_white_fpr": round(rej_pw / n_trials, 4),
        "politis_white_median_block": int(np.median(pw_blocks)),
        "dm_hac_fpr": round(rej_dm / n_trials, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = []
    for phi in (0.0, 0.70, 0.90, 0.95, 0.98):
        row = simulate_fpr(phi, n_trials=args.n_trials, n_boot=args.n_boot)
        rows.append(row)
        print(
            f"phi={phi:.2f} | i.i.d.={row['iid_fpr']:5.1%} | "
            f"CBB(32)={row['moving_block_32_fpr']:5.1%} | "
            f"CBB(64)={row['moving_block_64_fpr']:5.1%} | "
            f"PW(med B={row['politis_white_median_block']:2d})={row['politis_white_fpr']:5.1%} | "
            f"DM-HAC={row['dm_hac_fpr']:5.1%}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n_trials": args.n_trials, "n_boot": args.n_boot, "results": rows}
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
