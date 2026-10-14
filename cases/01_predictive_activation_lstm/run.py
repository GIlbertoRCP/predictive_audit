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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from predictive_audit import audit, report

sys.path.insert(0, str(Path(__file__).parent.parent))
from _export import write as export_json  # noqa: E402

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

    results = {}
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
        results[split] = {"model_f1": result.table["GBM (LSTM stand-in)"]["f1"],
                          "persistence_f1": result.table["persistence"]["f1"],
                          "best_baseline": result.best_baseline,
                          "best_baseline_f1": result.table[result.best_baseline]["f1"],
                          "verdict": result.verdict}
        sections += [f"## {title}", "", report.to_markdown(result), ""]

    (HERE / "RESULTS.md").write_text("\n".join(sections))
    print(f"\nwrote {HERE / 'RESULTS.md'}")
    if "--export" in sys.argv:
        # the 400-step window (any port) where the label changes state most often, so both states show
        best = None
        for port, g in df.groupby(GROUPS):
            flips = g["is_congested"].diff().abs().fillna(0).rolling(400).sum()
            if flips.notna().any() and (best is None or flips.max() > best[0]):
                end = g.index.get_loc(flips.idxmax())
                best = (flips.max(), port, g.iloc[end - 399:end + 1])
        _, port, one = best
        print("exported", export_json("case01", {
            "port": [int(port[0]), int(port[1])], "results": results,
            "latency_ms": one["latency_ms"].to_numpy(), "congested": one["is_congested"].to_numpy(),
        }))


if __name__ == "__main__":
    main()
