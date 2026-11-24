"""Case 06: BiswajitPadhi99 Azure VM CPU utilization forecasting (LSTM / GRU / IndRNN).

Reported-score audit on the Microsoft Azure 2017 public VM trace (8,640 five-minute steps,
30 days, aggregate min/max/avg CPU). Reproduces the notebooks' exact split and test windowing:
  - 80/20 chronological split: train = first 6,912 rows, test = last 1,728 rows
  - look_back = 5 -> create_training_dataset loops range(len - 5 - 1), scoring 1,722 test rows
    (indices 6,917 .. 8,638) across all 3 CPU series (5,166 scored cells)
  - Global MinMaxScaler(0, 1) fit across all 8,640 rows before splitting

Note two evaluation bugs in the published notebooks:
  1. Test MAE is computed as ``math.sqrt(mean_absolute_error(testY, testPredict))`` (130-137),
     so true test MAE is the square of the printed number (16,941-18,712).
  2. Correlation in the GRU notebook calls ``np.corrcoef(testPredict, testY)[0, 1]`` on a
     (1722, 3) matrix, correlating row 0 with row 1 of testPredict rather than pred vs actual.

    uv run python cases/06_azure_vm_cpu/run.py [--export]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _export import write as export_json  # noqa: E402
from _fetch import fetch  # noqa: E402

from predictive_audit import audit, baselines, report, wide_to_long  # noqa: E402

REPO = "BiswajitPadhi99/predicting-cloud-CPU-utilization-on-Azure-dataset-using-deeplearning"
COMMIT = "6c5d567a077eaf6b3e1050e7ab1c5cfc0b658048"
DATA = (
    "AzureReadings_at_a_timestamp.csv",
    "e65b19b0ec857b146e5a7ce96d4d73be66b9218c2c572f0db27be2f488e9f306",
)
COLS = ["min cpu", "max cpu", "avg cpu"]
LOOK_BACK = 5
SEASON = 288  # 24 hours of 5-minute intervals

# Printed outputs from the three committed notebooks at COMMIT
REPORTED = {
    "GRU (128)": {
        "raw_rmse": 28808.21,
        "printed_mae_sqrt": 133.156378,
        "true_raw_mae": 133.156378 ** 2,
        "mape": 0.011713,
    },
    "LSTM (128)": {
        "raw_rmse": 29059.44,
        "printed_mae_sqrt": 130.157027,
        "true_raw_mae": 130.157027 ** 2,
        "mape": 0.011205,
        "scaled_rmse": math.sqrt(0.00051842911516962),
        "scaled_mae": 0.015751611441373825,
    },
    "IndRNN (128)": {
        "raw_rmse": 29480.69,
        "printed_mae_sqrt": 136.790292,
        "true_raw_mae": 136.790292 ** 2,
        "mape": 0.012492,
        "scaled_rmse": math.sqrt(0.0005520622850488763),
        "scaled_mae": 0.01743340492248535,
    },
}


def prepare(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    n = len(df)
    train_size = int(n * 0.8)  # 6912
    split = np.full(n, "train", dtype=object)
    split[train_size : train_size + LOOK_BACK] = "history"
    split[train_size + LOOK_BACK : n - 1] = "test"  # 1,722 rows matching range(len(test) - 5 - 1)
    split[n - 1] = "dropped"

    raw = df[COLS].astype("float64").copy()
    raw["split"] = split
    long_raw = wide_to_long(raw, keep=["split"], series_name="series", value_name="cpu")

    # Reproduce notebooks' global MinMaxScaler(0, 1) fit before train/test split
    vals32 = df[COLS].astype("float32")
    scaled = ((vals32 - vals32.min()) / (vals32.max() - vals32.min())).astype("float64")
    scaled["split"] = split
    long_scaled = wide_to_long(scaled, keep=["split"], series_name="series", value_name="cpu_scaled")
    return df, long_raw, long_scaled


def main() -> None:
    path = fetch(REPO, COMMIT, DATA[0], DATA[1], HERE / ".cache")
    df, long_raw, long_scaled = prepare(path)

    sections = [
        "# Case 06: Azure VM CPU Utilization (LSTM / GRU / IndRNN)",
        "",
        f"Source: [{REPO}@{COMMIT[:7]}](https://github.com/{REPO}/tree/{COMMIT}). "
        "Reported-score audit on the Microsoft Azure 2017 5-minute VM CPU trace (5,166 scored cells).",
        "",
        "## Summary (raw CPU units, 1,722 test steps × 3 series)",
        "",
    ]

    # Audit each model's published raw RMSE against trivial baselines + EWMA + causal Linear AR(5)
    summary_rows = []
    base_raw_report = None
    for name, rep in REPORTED.items():
        r = audit(
            long_raw,
            "cpu",
            task="regression",
            groups="series",
            time="step",
            split="split",
            metric="rmse",
            reported=rep["raw_rmse"],
            season=SEASON,
            include_ewma=True,
            ar_lags=LOOK_BACK,
        )
        base_raw_report = r
        chk = {c.name: c for c in r.checks}["reported score"]
        summary_rows.append(
            f"| {name} | {rep['raw_rmse']:,.2f} | {rep['true_raw_mae']:,.2f} "
            f"(printed √MAE: {rep['printed_mae_sqrt']:.2f}) | {rep['mape'] * 100:.2f}% | **{chk.status}** |"
        )
        print(f"\n=== Raw RMSE: {name} ===\n{report.to_text(r)}")
        sections += [f"## Raw units: {name}", "", report.to_markdown(r), ""]

    assert base_raw_report is not None
    t = base_raw_report.table
    # Compute exact persistence MAPE on the same test cells for direct comparison
    test_mask = long_raw["split"] == "test"
    y_test = long_raw.loc[test_mask, "cpu"].to_numpy()
    persist_all = baselines.persistence(long_raw["cpu"], long_raw.groupby("series", sort=False).ngroup(), 1)
    p_test = persist_all.loc[test_mask].to_numpy()
    persist_mape = float(np.mean(np.abs(p_test - y_test) / np.abs(y_test)))

    ar_all = baselines.linear_ar(
        long_raw["cpu"],
        long_raw.groupby("series", sort=False).ngroup(),
        1,
        long_raw["split"] != "test",
        lags=LOOK_BACK,
    )
    ar_test = ar_all.loc[test_mask].to_numpy()
    ar_mape = float(np.mean(np.abs(ar_test - y_test) / np.abs(y_test)))

    table_header = [
        "| Forecaster | Test RMSE | Test MAE | Test MAPE | Verdict vs baselines |",
        "|---|---:|---:|---:|---|",
        *summary_rows,
        f"| **Linear AR(5) (causal Ridge)** | **{t['linear AR(5)']['rmse']:,.2f}** | "
        f"**{t['linear AR(5)']['mae']:,.2f}** | **{ar_mape * 100:.2f}%** | baseline |",
        f"| **EWMA (3)** | **{t['ewma (3)']['rmse']:,.2f}** | "
        f"**{t['ewma (3)']['mae']:,.2f}** | — | baseline |",
        f"| **Persistence (lag 1)** | **{t['persistence']['rmse']:,.2f}** | "
        f"**{t['persistence']['mae']:,.2f}** | **{persist_mape * 100:.2f}%** | baseline |",
        "",
    ]
    sections[5:5] = table_header

    # Scaled evaluation check (matching model.evaluate(testX, testY))
    for name in ("LSTM (128)", "IndRNN (128)"):
        rep = REPORTED[name]
        r_s = audit(
            long_scaled,
            "cpu_scaled",
            task="regression",
            groups="series",
            time="step",
            split="split",
            metric="rmse",
            reported=rep["scaled_rmse"],
            season=SEASON,
            include_ewma=True,
            ar_lags=LOOK_BACK,
        )
        print(f"\n=== Scaled RMSE: {name} ===\n{report.to_text(r_s)}")
        sections += [f"## Scaled units (`model.evaluate`): {name}", "", report.to_markdown(r_s), ""]

    (HERE / "RESULTS.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'RESULTS.md'}")

    if "--export" in sys.argv:
        # Export 3 days (864 five-minute steps) of test avg CPU + baselines + RMSE summary for site/
        avg_mask = (long_raw["series"] == "avg cpu") & (long_raw["split"] == "test")
        y_avg = long_raw.loc[avg_mask, "cpu"].to_numpy()[:576]
        p_avg = persist_all.loc[avg_mask].to_numpy()[:576]
        ar_avg = ar_all.loc[avg_mask].to_numpy()[:576]
        out = export_json(
            "case06",
            {
                "minutes_per_step": 5,
                "reported_rmse": {k: v["raw_rmse"] for k, v in REPORTED.items()},
                "baseline_rmse": {
                    "linear AR(5)": t["linear AR(5)"]["rmse"],
                    "ewma (3)": t["ewma (3)"]["rmse"],
                    "persistence": t["persistence"]["rmse"],
                    "moving average (3)": t["moving average (3)"]["rmse"],
                },
                "persistence_r2": t["persistence"]["r2"],
                "persistence_mape": round(persist_mape * 100, 2),
                "y_avg_m": (y_avg / 1e6).tolist(),
                "persist_avg_m": (p_avg / 1e6).tolist(),
                "ar_avg_m": (ar_avg / 1e6).tolist(),
            },
        )
        print("exported", out)


if __name__ == "__main__":
    main()
