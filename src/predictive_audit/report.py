"""Render an AuditReport as plain text, Markdown, or strict RFC 8259 JSON."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .audit import AuditReport


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.4f}"


def _rows(report: AuditReport) -> tuple[list[str], list[list[str]]]:
    names = list(next(iter(report.table.values())).keys())
    rows: list[list[str]] = []
    for model, values in report.table.items():
        label = model
        if model == report.model:
            label += "  <- model"
        elif model == report.best_baseline:
            label += "  <- best baseline"
        rows.append([label] + [_fmt(values[n]) for n in names])
    return names, rows


def _header(report: AuditReport) -> list[str]:
    lines = [
        f"target: {report.target}   task: {report.task}   horizon: {report.horizon}   "
        f"metric: {report.metric}   rows scored: {report.n_eval}"
    ]
    if report.season:
        lines.append(f"seasonal period: {report.season} steps")
    if report.prevalence is not None:
        lines.append(f"positive rate: {report.prevalence:.3f}")
    if report.threshold is not None and report.threshold != 0.5:
        lines.append(f"classification threshold: {report.threshold:.4f}")
    if report.mase is not None and math.isfinite(report.mase):
        lines.append(f"MASE (vs persistence): {report.mase:.4f}")
    return lines


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf and NumPy scalars so JSON strictly conforms to RFC 8259."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    return obj


def to_text(report: AuditReport) -> str:
    names, rows = _rows(report)
    widths = [max(len(r[i]) for r in rows + [["model"] + names]) for i in range(len(names) + 1)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    out = [f"VERDICT: {report.verdict}", *_header(report), "", line(["model"] + names)]
    out += [line(r) for r in rows]
    out.append("")
    for c in report.checks:
        out.append(f"[{c.status}] {c.name}: {c.summary}")
    return "\n".join(out)


def to_markdown(report: AuditReport) -> str:
    names, rows = _rows(report)
    out = [f"**Verdict: {report.verdict}**", "", *(f"- {h}" for h in _header(report)), ""]
    out.append("| model | " + " | ".join(names) + " |")
    out.append("|---" * (len(names) + 1) + "|")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out += ["", "| check | status | finding |", "|---|---|---|"]
    out += [f"| {c.name} | {c.status} | {c.summary} |" for c in report.checks]
    return "\n".join(out)


def to_json(report: AuditReport) -> str:
    """Serialize report to strict RFC 8259 JSON (NaN/Inf encoded as null)."""
    return json.dumps(_sanitize_json(report.to_dict()), indent=2, allow_nan=False)
