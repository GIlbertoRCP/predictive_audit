"""Render an AuditReport as plain text, Markdown, or JSON."""

from __future__ import annotations

import json
import math

from .audit import AuditReport


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _rows(report: AuditReport):
    names = list(next(iter(report.table.values())).keys())
    rows = []
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
    if report.prevalence is not None:
        lines.append(f"positive rate: {report.prevalence:.3f}")
    return lines


def to_text(report: AuditReport) -> str:
    names, rows = _rows(report)
    widths = [max(len(r[i]) for r in rows + [["model"] + names]) for i in range(len(names) + 1)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
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
    return json.dumps(report.to_dict(), indent=2, default=float)
