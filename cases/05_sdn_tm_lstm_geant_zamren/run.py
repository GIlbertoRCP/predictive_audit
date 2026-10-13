"""Case 05: Annongkhanh/Traffic-matrix-prediction, LSTM traffic-matrix prediction on GÉANT and a Mininet testbed.

Reported-score audit (no retraining). The notebooks print one number each:
  train/Zamren.ipynb  dataset/test.csv   (Mininet + POX + tcpreplay, Zamren topology)  MSE 0.04135
  train/Geant.ipynb   dataset/Geant.csv  (GÉANT 2005, 15-minute TMs)                   MSE 0.02596

Both use the same protocol, reproduced here: MinMaxScaler per OD column fit on the whole table,
first 80% of rows train / last 20% test, 10 previous matrices -> next matrix, and
sklearn mean_squared_error over every (test step, OD) cell. The baselines are scored on exactly those
cells and in the same scaled units, so their RMSE compares directly with sqrt(reported MSE).

    uv run --group cases python cases/05_sdn_tm_lstm_geant_zamren/run.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _fetch import fetch  # noqa: E402

from predictive_audit import audit, report, wide_to_long  # noqa: E402

REPO = "Annongkhanh/Traffic-matrix-prediction"
COMMIT = "0ab331d932f6567464bd93738c2da11c265eb888"
DATASETS = {
    # name: (path, sha256, time column, reported MSE)
    "Zamren testbed": ("dataset/test.csv", "326b51c364ea4ac828c109bf39edf982263f97dd4b9f628c432e5e38ad1a138f", 0, 0.04135),
    "GÉANT": ("dataset/Geant.csv", "a2296c6c2b886195ae0c05dc861eeac7c08984c7ceada299938dbb80bd8e08d2", "Column1", 0.02596),
}
WINDOW = 10


def prepare(path, time_col):
    df = pd.read_csv(path)
    time_name = df.columns[time_col] if isinstance(time_col, int) else time_col
    values = df.drop(columns=[time_name]).astype("float32")
    lo, hi = values.min(), values.max()
    scaled = ((values - lo) / (hi - lo).replace(0, 1)).copy()  # constant columns -> 0, as MinMaxScaler
    n_train = int(len(scaled) * 0.8)
    split = np.full(len(scaled), "train", dtype=object)
    split[n_train:n_train + WINDOW] = "history"  # first test rows are inputs only
    split[n_train + WINDOW:] = "test"
    scaled = scaled.assign(split=split)
    return scaled, values.shape[1], len(df) - n_train - WINDOW


def main() -> None:
    sections = ["# Case 05: Annongkhanh/Traffic-matrix-prediction", ""]
    for name, (path, sha, time_col, reported_mse) in DATASETS.items():
        local = fetch(REPO, COMMIT, path, sha, HERE / ".cache")
        wide, n_series, n_test_steps = prepare(local, time_col)
        long = wide_to_long(wide, keep=["split"], series_name="od", value_name="y")
        result = audit(
            long,
            "y",
            task="regression",
            groups="od",
            time="step",
            split="split",
            metric="rmse",
            reported=math.sqrt(reported_mse),
            season="auto",
        )
        title = f"{name}: reported MSE {reported_mse} (RMSE {math.sqrt(reported_mse):.4f}), {n_series} OD columns x {n_test_steps} test steps"
        print(f"\n=== {title} ===\n{report.to_text(result)}")
        sections += [f"## {title}", "", report.to_markdown(result), ""]
    (HERE / "RESULTS.md").write_text("\n".join(sections))


if __name__ == "__main__":
    main()
