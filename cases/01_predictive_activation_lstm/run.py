"""Case 01: the congestion forecaster from Predictive_Activation v0 (CSCI 4930, CU Denver).

Rebuilds the original task from legacy/train_lstm.py:
  label     = latency_ms > 100 on the *next* step, per (switch, port)
  features  = last 5 steps of 6 telemetry columns
  protocol  = shuffle all windows, then 80/20 split

The original LSTM needs PyTorch and its checkpoint predicts a broken label, so a
gradient-boosted classifier on the same windows stands in for it. The point of
the case is the evaluation protocol, not the architecture.

    uv run python cases/01_predictive_activation_lstm/run.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from predictive_audit import audit, report

HERE = Path(__file__).parent
FEATURES = ["rx_mbps", "tx_mbps", "rx_loss", "tx_loss", "avg_queue_depth", "latency_ms"]
GROUPS = ["switch_id", "port_no"]
WINDOW = 5
SEED = 42


def load() -> pd.DataFrame:
    df = pd.read_csv(HERE / "telemetry.csv").sort_values(GROUPS + ["timestamp"]).reset_index(drop=True)
    df["is_congested"] = (df["latency_ms"] > 100.0).astype(int)
    df["future_congestion"] = df.groupby(GROUPS)["is_congested"].shift(-1)
    # window features: value of each column at t, t-1, ..., t-4 (all observed at t)
    lagged = {
        f"{c}_lag{k}": df.groupby(GROUPS)[c].shift(k) for c in FEATURES for k in range(WINDOW)
    }
    df = pd.concat([df, pd.DataFrame(lagged)], axis=1)
    return df.dropna(subset=["future_congestion", *lagged]).reset_index(drop=True)


def fit_predict(df: pd.DataFrame, split: pd.Series) -> pd.Series:
    cols = [c for c in df.columns if "_lag" in c]
    train = split == "train"
    model = HistGradientBoostingClassifier(random_state=SEED).fit(df.loc[train, cols], df.loc[train, "future_congestion"])
    pred = pd.Series(np.nan, index=df.index)
    pred[~train] = model.predict_proba(df.loc[~train, cols])[:, 1]
    return pred


def main() -> None:
    df = load()
    rng = np.random.default_rng(SEED)

    # A: the original protocol, shuffled windows
    df["split_shuffled"] = np.where(rng.random(len(df)) < 0.8, "train", "test")
    # B: chronological, last 20% of each series is test
    position = df.groupby(GROUPS).cumcount() / df.groupby(GROUPS)["timestamp"].transform("size")
    df["split_chrono"] = np.where(position < 0.8, "train", "test")

    sections = ["# Case 01: Predictive_Activation v0 congestion forecaster", ""]
    for split, title in [
        ("split_shuffled", "A. Original protocol (shuffled 80/20 split)"),
        ("split_chrono", "B. Chronological split (last 20% of each series)"),
    ]:
        df[f"pred_{split}"] = fit_predict(df, df[split])
        result = audit(
            df,
            "future_congestion",
            pred=f"pred_{split}",
            model_name="GBM (LSTM stand-in)",
            groups=GROUPS,
            time="timestamp",
            split=split,
            features=FEATURES,
            metric="f1",
        )
        print(f"\n=== {title} ===\n{report.to_text(result)}")
        sections += [f"## {title}", "", report.to_markdown(result), ""]

    (HERE / "RESULTS.md").write_text("\n".join(sections))
    print(f"\nwrote {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
