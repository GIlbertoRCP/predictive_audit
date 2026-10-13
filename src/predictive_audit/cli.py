"""Command-line entry point: ``predictive-audit data.csv --target y_next --pred model ...``"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import report
from .audit import SEVERITY, audit


def _csv_list(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predictive-audit",
        description="Check whether a time-series forecasting result beats trivial baselines.",
    )
    p.add_argument("data", help="CSV with one row per (series, time step)")
    p.add_argument("--target", required=True, help="column holding y[t + horizon], the value being forecast")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--horizon", type=int, default=1, help="steps ahead the target is (default 1)")
    p.add_argument("--pred", help="column with the model's predictions/scores")
    p.add_argument("--model-name", help="display name for the model")
    p.add_argument("--group", help="comma-separated columns identifying each series")
    p.add_argument("--time", help="time column (otherwise file order within each series)")
    p.add_argument("--split", help="column with train/test labels; only test rows are scored")
    p.add_argument("--features", help="comma-separated feature columns to scan for future leakage")
    p.add_argument("--season", type=int, help="season length for a seasonal-naive baseline")
    p.add_argument("--window", type=int, help="moving-average window (default max(3, horizon))")
    p.add_argument("--metric", help="headline metric (default f1 / mae)")
    p.add_argument("--reported", type=float, help="a published score on --metric to check")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--block", type=int, help="bootstrap block length (default sqrt(n))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    p.add_argument("--out", help="write the report here instead of stdout")
    p.add_argument(
        "--fail-on",
        choices=["fail", "warn", "never"],
        default="fail",
        help="exit non-zero at this verdict or worse (for CI)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit(
            pd.read_csv(args.data),
            args.target,
            task=args.task,
            horizon=args.horizon,
            pred=args.pred,
            model_name=args.model_name,
            groups=_csv_list(args.group),
            time=args.time,
            split=args.split,
            features=_csv_list(args.features),
            season=args.season,
            window=args.window,
            metric=args.metric,
            reported=args.reported,
            n_boot=args.n_boot,
            block=args.block,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"predictive-audit: error: {e}", file=sys.stderr)
        return 2

    render = {"text": report.to_text, "markdown": report.to_markdown, "json": report.to_json}[args.format]
    text = render(result)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)

    if args.fail_on == "never":
        return 0
    threshold = SEVERITY["FAIL" if args.fail_on == "fail" else "WARN"]
    return 1 if SEVERITY[result.verdict] >= threshold else 0


if __name__ == "__main__":
    sys.exit(main())
