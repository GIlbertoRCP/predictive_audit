"""Tests for strict RFC 8259 JSON serialization, report renderers, and CLI features/errors."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from predictive_audit import SCHEMA_VERSION, audit, report
from predictive_audit.cli import main


def _strict_json_loads(text: str) -> dict:
    def _forbid_non_standard(token: str) -> None:
        raise ValueError(f"Non-standard JSON token emitted: {token}")

    return json.loads(text, parse_constant=_forbid_non_standard)


def test_to_json_is_strict_rfc8259_even_with_nans():
    # Single-class binary target makes roc_auc and pr_auc NaN
    df = pd.DataFrame({"t": np.arange(100), "target": np.zeros(100), "pred": np.zeros(100)})
    r = audit(df, "target", task="classification", pred="pred", time="t", n_boot=20)
    raw = report.to_json(r)
    parsed = _strict_json_loads(raw)
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["table"]["pred"]["roc_auc"] is None
    assert parsed["table"]["pred"]["pr_auc"] is None


def test_report_text_and_markdown_render_metadata():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300).cumsum()
    df = pd.DataFrame({
        "t": np.arange(len(y) - 1),
        "target": y[1:],
        "pred": y[:-1] + rng.normal(scale=0.1, size=len(y) - 1),
        "split": ["train"] * 200 + ["test"] * (len(y) - 201),
    })
    r = audit(df, "target", task="regression", pred="pred", time="t", split="split", season=24, n_boot=50)
    txt = report.to_text(r)
    md = report.to_markdown(r)
    assert "VERDICT:" in txt
    assert "MASE (vs persistence):" in txt
    assert "seasonal period: 24 steps" in txt
    assert "**Verdict:" in md
    assert "| check | status | finding |" in md


def test_cli_wide_mode_and_config_and_out(tmp_path, capsys):
    rng = np.random.default_rng(3)
    wide = pd.DataFrame({
        "od_1": rng.normal(size=120).cumsum(),
        "od_2": rng.normal(size=120).cumsum(),
        "split": ["train"] * 90 + ["test"] * 30,
    })
    tsv_path = tmp_path / "tm.tsv"
    wide.to_csv(tsv_path, sep="\t", index=False)

    cfg = tmp_path / "audit.toml"
    cfg.write_text(
        '[audit]\ntask = "regression"\nmetric = "mae"\nar_lags = 3\newma = true\n',
        encoding="utf-8",
    )

    out_json = tmp_path / "reports" / "out.json"
    code = main([
        str(tsv_path),
        "--config", str(cfg),
        "--wide",
        "--split", "split",
        "--format", "json",
        "--out", str(out_json),
        "--fail-on", "never",
    ])
    assert code == 0
    parsed = _strict_json_loads(out_json.read_text(encoding="utf-8"))
    assert "linear AR(3)" in parsed["table"]
    assert "ewma (3)" in parsed["table"]
    assert parsed["n_eval"] == 60


def test_cli_error_handling_returns_exit_code_2(tmp_path, capsys):
    # Missing file
    assert main([str(tmp_path / "does_not_exist.csv"), "--target", "y"]) == 2
    assert "not found" in capsys.readouterr().err

    # Empty CSV
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert main([str(empty), "--target", "y"]) == 2
    assert "error:" in capsys.readouterr().err

    # Missing --target without --wide
    valid = tmp_path / "ok.csv"
    pd.DataFrame({"y": [0, 1] * 20}).to_csv(valid, index=False)
    assert main([str(valid)]) == 2
    assert "--target is required" in capsys.readouterr().err

    # Bad config file
    assert main([str(valid), "--config", str(tmp_path / "missing.toml"), "--target", "y"]) == 2
    assert "not found" in capsys.readouterr().err
