"""Assemble site/data/summary.json from committed case outputs (no retraining).

    uv run python site/build_data.py

Cases 01, 02 and 04 write their own excerpts with ``run.py --export``; this script adds the
per-OD results of cases 03 and 04 and the reported-score comparison of case 05.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cases"))
from _export import write  # noqa: E402

CASES = ROOT / "cases"


def markdown_tables(text: str) -> dict[str, dict[str, dict[str, float]]]:
    """{section heading: {row label: {column: value}}} for every model table in RESULTS.md text."""
    out, heading, header = {}, None, None
    for line in text.splitlines():
        if line.startswith("#"):
            heading, header = line.lstrip("# ").strip(), None
        elif line.startswith("| model |"):
            header = [c.strip() for c in line.strip("|").split("|")][1:]
        elif header and line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            label = re.sub(r"\s+<-.*$", "", cells[0])
            try:
                out.setdefault(heading, {})[label] = dict(zip(header, map(float, cells[1:])))
            except ValueError:
                header = None  # left the model table
        elif not line.startswith("|"):
            header = None
    return out


def case03():
    per_od = pd.read_csv(CASES / "03_thu_traffic_matrix" / "RESULTS_per_od.csv")
    lstm = per_od[per_od["model"] == "LSTM"]
    # OD pairs that are constant over the whole test period (persistence error exactly 0) have nothing
    # to forecast; the audit now reports these as INFO, so count them apart instead of as failures
    inactive = lstm["persistence_mae"] == 0
    lstm, n_inactive = lstm[~inactive], inactive.groupby(lstm["network"]).sum().to_dict()
    # the pooled-table headings repeat for each network, so parse one network section at a time
    text = (CASES / "03_thu_traffic_matrix" / "RESULTS.md").read_text()
    pooled = {}
    for network in ("abilene", "geant", "cernet"):
        rows = markdown_tables(text.split(f"## {network}:")[1].split("\n## ")[0])
        pooled[network] = {
            "lstm": rows["Pooled over OD pairs: LSTM"]["LSTM"]["mae"],
            "ar": rows["Pooled over OD pairs: Linear AR(10)"]["Linear AR(10)"]["mae"],
            "persistence": rows["Pooled over OD pairs: LSTM"]["persistence"]["mae"],
        }
    return {
        "pooled_mae": pooled,
        "inactive_in_test": n_inactive,
        "tally": {n: g["status"].value_counts().to_dict() for n, g in lstm.groupby("network")},
        "ratio_vs_persistence": {n: (g["mae"] / g["persistence_mae"]).tolist() for n, g in lstm.groupby("network")},
    }


def case04():
    t = pd.read_csv(CASES / "04_sdn_tm_prediction" / "RESULTS_summary.csv")
    b = t[t["model"] == "BiLSTM"]
    return {"per_od": b[["od", "rmse", "best_baseline", "baseline_rmse", "status", "season"]].to_dict("records")}


def case05():
    tables = markdown_tables((CASES / "05_sdn_tm_lstm_geant_zamren" / "RESULTS.md").read_text())
    out = {}
    for heading, rows in tables.items():
        name = heading.split(":")[0]
        reported = float(re.search(r"RMSE ([0-9.]+)", heading).group(1))
        out[name] = {"reported_rmse": reported, **{k: v["rmse"] for k, v in rows.items()}}
    return out


if __name__ == "__main__":
    path = write("summary", {"case03": case03(), "case04": case04(), "case05": case05()})
    print("wrote", path, path.stat().st_size, "bytes")
    print(json.dumps(json.loads(path.read_text())["case05"], indent=1))
