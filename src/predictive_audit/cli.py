"""Command-line entry point: ``predictive-audit data.csv --target y_next --pred model ...``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from . import __version__, _backend, baselines, report
from .audit import SEVERITY, audit
from .reshape import wide_to_long

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# The native CLI reproduces the installed pandas' read_csv, whose float parser changed in 3.1.
_PANDAS_31 = tuple(int(v) for v in pd.__version__.split(".")[:2] if v.isdigit()) >= (3, 1)


def _csv_list(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _season(value: str) -> int | str:
    return value if value == "auto" else int(value)


def _threshold(value: str) -> float | Literal["auto"]:
    return "auto" if value == "auto" else float(value)


def _block(value: str) -> int | Literal["auto"]:
    return "auto" if value == "auto" else int(value)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ValueError("reading Parquet requires 'pyarrow' or 'fastparquet'") from exc
    if suffix == ".feather":
        try:
            return pd.read_feather(path)
        except ImportError as exc:
            raise ValueError("reading Feather requires 'pyarrow'") from exc
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_toml_defaults(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise ValueError("--config requires Python >= 3.11 or 'tomli'")
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    section = raw.get("audit", raw)
    if not isinstance(section, dict):
        raise ValueError("config file must contain a top-level mapping or [audit] table")
    out: dict[str, Any] = {}
    for k, v in section.items():
        key = k.replace("-", "_")
        if isinstance(v, list):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predictive-audit",
        description="Check whether a time-series forecasting result beats trivial baselines.",
    )
    p.add_argument("data", help="CSV, TSV, Parquet, or Feather table")
    p.add_argument("--config", type=Path, help="TOML config file with default audit options")
    p.add_argument("--target", help="column holding y[t + horizon] (auto-set to 'value' with --wide)")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--horizon", type=int, default=1, help="steps ahead the target is (default 1)")
    p.add_argument("--pred", help="column with the model's predictions/scores")
    p.add_argument("--model-name", help="display name for the model")
    p.add_argument("--group", help="comma-separated columns identifying each series")
    p.add_argument("--time", help="time column (otherwise file order within each series)")
    p.add_argument(
        "--split",
        help="column labelling rows test / train / other; only test rows are scored, other rows are history",
    )
    p.add_argument("--features", help="comma-separated feature columns to scan for future leakage")
    p.add_argument(
        "--wide",
        action="store_true",
        help="reshape wide matrix (one column per series) before auditing",
    )
    p.add_argument("--wide-keep", help="comma-separated non-series columns to preserve with --wide")
    p.add_argument("--season", type=_season, help="season length for a seasonal-naive baseline, or 'auto'")
    p.add_argument("--window", type=int, help="moving-average window (default max(3, horizon))")
    p.add_argument("--ewma", action="store_true", help="include an EWMA baseline")
    p.add_argument("--ar-lags", type=int, help="include a causal Ridge AR(p) baseline fit on non-test rows")
    p.add_argument("--metric", help="headline metric (default f1 / mae)")
    p.add_argument("--reported", type=float, help="a published score on --metric to check")
    p.add_argument(
        "--threshold",
        type=_threshold,
        default=0.5,
        help="classification score threshold (float or 'auto' tuned on non-test rows)",
    )
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument(
        "--block",
        type=_block,
        help="bootstrap block length (int, 'auto' Politis-White, or default)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    p.add_argument("--out", help="write the report here instead of stdout")
    p.add_argument(
        "--fail-on",
        choices=["fail", "warn", "never"],
        default="fail",
        help="exit non-zero at this verdict or worse (for CI)",
    )
    p.add_argument(
        "--threads",
        type=int,
        help="engine threads (default: PREDICTIVE_AUDIT_THREADS or all cores)",
    )
    p.add_argument("--version", action="version", version=f"predictive-audit {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    native = _backend.native
    if native is not None:
        # One CLI implementation: the C++ one (the same code as the standalone executable).
        # It hands back inputs only pandas reads (Parquet, Feather, missing key values, ...),
        # and those run through the reference implementation below.
        code, out, err, fallback, _reason = native.cli_main(
            args_list, baselines._PANDAS_EWM_COM1_UPDATE, _PANDAS_31
        )
        if not fallback:
            sys.stdout.write(out)
            sys.stderr.write(err)
            return int(code)
    return _main_python(args_list)


def _main_python(argv: list[str]) -> int:
    parser = build_parser()
    # Two-pass parse so CLI flags always override TOML config defaults
    prelim, _ = parser.parse_known_args(argv)
    if prelim.config is not None:
        try:
            parser.set_defaults(**_load_toml_defaults(prelim.config))
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"predictive-audit: error: {exc}", file=sys.stderr)
            return 2

    args = parser.parse_args(argv)
    if args.threads is not None:
        if args.threads < 1:
            print("predictive-audit: error: --threads must be >= 1", file=sys.stderr)
            return 2
        if _backend.native is not None:
            _backend.native.set_num_threads(args.threads)
    try:
        df = _load_frame(Path(args.data))
        groups = _csv_list(args.group)
        time_col = args.time
        target_col = args.target

        if args.wide:
            keep = _csv_list(args.wide_keep)
            if args.split and args.split not in keep:
                keep.append(args.split)
            df = wide_to_long(df, time=time_col, keep=keep)
            target_col = target_col or "value"
            groups = groups or ["series"]
            time_col = time_col or "step"

        if not target_col:
            raise ValueError("--target is required unless --wide is set")

        result = audit(
            df,
            target_col,
            task=args.task,
            horizon=args.horizon,
            pred=args.pred,
            model_name=args.model_name,
            groups=groups or None,
            time=time_col,
            split=args.split,
            features=_csv_list(args.features) or None,
            season=args.season,
            window=args.window,
            include_ewma=args.ewma,
            ar_lags=args.ar_lags,
            metric=args.metric,
            reported=args.reported,
            threshold=args.threshold,
            n_boot=args.n_boot,
            block=args.block,
            seed=args.seed,
        )
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        print(f"predictive-audit: error: {exc}", file=sys.stderr)
        return 2

    render = {"text": report.to_text, "markdown": report.to_markdown, "json": report.to_json}[args.format]
    text = render(result)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.fail_on == "never":
        return 0
    threshold = SEVERITY["FAIL" if args.fail_on == "fail" else "WARN"]
    return 1 if SEVERITY[result.verdict] >= threshold else 0


if __name__ == "__main__":
    sys.exit(main())
