"""Regenerate the golden CLI fixtures and expected outputs from the pure-Python reference CLI.

    PREDICTIVE_AUDIT_BACKEND=python uv run python tests/golden/regenerate.py

Writes the fixture tables (deterministic synthetic data), then runs every case in cases.json
through predictive_audit.cli with the reference backend and stores stdout, stderr and the exit
code under expected/. tests/golden/check.py compares any CLI (the standalone executable, or the
package's console script) against these files.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _fixtures() -> None:
    rng = np.random.default_rng(20261127)

    # Three regression series with a leaky feature and a min-max scaled target copy.
    parts = []
    for g, period in enumerate((12, 24, 7)):
        t = np.arange(300)
        y = np.sin(2 * np.pi * t / period) + rng.normal(scale=0.3, size=t.size).cumsum() * 0.05
        d = pd.DataFrame({"t": t, "site": f"site-{g}", "y_next": np.roll(y, -1)})
        d["pred"] = y + rng.normal(scale=0.25, size=t.size)
        d["leak"] = d["y_next"] * 2 + rng.normal(scale=0.01, size=t.size)
        d["load"] = rng.normal(size=t.size)
        d["split"] = np.where(t < 210, "train", "test")
        parts.append(d.iloc[:-1])
    pd.concat(parts, ignore_index=True).to_csv(HERE / "regression_panel.csv", index=False)

    # Binary classification with timestamp strings, upper-case split labels and a probability score.
    n = 1500
    t = pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(n) * 5, unit="min")
    state = (np.sin(np.arange(n) / 40) + rng.normal(scale=0.6, size=n) > 0.3).astype(int)
    df = pd.DataFrame({
        "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
        "link": np.where(np.arange(n) % 2 == 0, "core,a", "edge b"),
        "congested_next": np.roll(state, -1),
        "score": np.clip(state * 0.6 + rng.random(n) * 0.5, 0, 1),
        "split": np.where(np.arange(n) < 1000, "TRAIN", "Test"),
    })
    df.iloc[:-1].to_csv(HERE / "classification.csv", index=False)

    # A wide matrix: one column per series, plus a split column, as TSV.
    steps = 160
    wide = pd.DataFrame({f"od_{k}": rng.normal(size=steps).cumsum() for k in range(4)})
    wide["split"] = np.where(np.arange(steps) < 120, "train", "test")
    wide.to_csv(HERE / "wide.tsv", sep="\t", index=False)

    # CSV dialect details: BOM, CRLF, quoted fields, NA spellings, a blank line.
    rows = ['﻿time,"series id",target,pred,split']
    for i in range(120):
        target = "NA" if i == 40 else f"{np.sin(i / 6):.6f}"
        rows.append(f'{i},"s ""{i % 2}""",{target},{np.sin((i - 1) / 6):.6f},{"train" if i < 80 else "test"}')
        if i == 60:
            rows.append("")
    (HERE / "messy.csv").write_bytes(("\r\n".join(rows) + "\r\n").encode("utf-8"))

    (HERE / "audit.toml").write_text(
        '[audit]\ntask = "regression"\nmetric = "rmse"\nar-lags = 3\newma = true\nseason = "auto"\n'
        'features = ["leak", "load"]\nreported = 0.25\n',
        encoding="utf-8",
    )


def run_reference(args: list[str]) -> tuple[int, str, str]:
    from predictive_audit import _backend
    from predictive_audit.cli import main

    out, err = io.StringIO(), io.StringIO()
    with _backend.use("python"), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(args)
        except SystemExit as exc:  # argparse errors
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


def main() -> None:
    _fixtures()
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    expected_dir = HERE / "expected"
    expected_dir.mkdir(exist_ok=True)
    os.chdir(HERE)
    for case in cases:
        code, out, err = run_reference(case["args"])
        name = case["name"]
        (expected_dir / f"{name}.out").write_text(out, encoding="utf-8", newline="")
        (expected_dir / f"{name}.meta.json").write_text(
            json.dumps({"exit_code": code, "stderr": err}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: exit {code}", file=sys.stderr)


if __name__ == "__main__":
    main()
