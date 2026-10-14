"""Case 02: xiaochus/TrafficFlowPrediction, PeMS freeway flow with LSTM / GRU / SAEs (681 stars).

Audits the repo's own trained weights (model/*.h5, Keras 2.1.4), run through a numpy
re-implementation of the Keras forward pass. Reproducing the README's metrics from those
weights is the check that the re-implementation is faithful.

    uv run --group cases python cases/02_xiaochus_pems/run.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _export import write as export_json  # noqa: E402
from _fetch import fetch_all  # noqa: E402

from predictive_audit import audit, report  # noqa: E402

REPO = "xiaochus/TrafficFlowPrediction"
COMMIT = "dec8243eb4ebe728d44d1ac73f38c2d3aaa9fd81"
FILES = {
    "data/train.csv": "cf46f219b4595fa214d99d42a85ccd5e7137499efaa54e4705eaf25653b09f14",
    "data/test.csv": "c3abee88094bb733e04c74812ae6b749ac5f21f6ccedf53120b5e3d8df2fc7ba",
    "model/lstm.h5": "06386e62902e460ba05310c67886694ffa2e1ba410f47f65063b5aa8620a233a",
    "model/gru.h5": "cb51c580ecfd7e5a2190af5a0ddd6d69711b98853d382a18e4db4f89b7aab774",
    "model/saes.h5": "3491a54133edd25a3e10ac5c7268eabc51ac686d469ac2b74c25861999f74697",
}
ATTR = "Lane 1 Flow (Veh/5 Minutes)"
LAG = 12
# README table at the pinned commit
REPORTED = {
    "LSTM": {"mae": 7.21, "rmse": 9.90, "r2": 0.9396},
    "GRU": {"mae": 7.20, "rmse": 9.97, "r2": 0.9389},
    "SAEs": {"mae": 7.06, "rmse": 9.60, "r2": 0.9433},
}


# --- Keras 2.1 forward pass ---------------------------------------------------


def hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0.0, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weights(path):
    with h5py.File(path, "r") as h:
        out = {}
        h["model_weights"].visititems(
            lambda name, obj: out.__setitem__(name.split("/")[-1].split(":")[0] + "@" + name.split("/")[0], obj[()])
            if isinstance(obj, h5py.Dataset)
            else None
        )
        config = json.loads(h.attrs["model_config"])
    return out, config


def lstm_layer(x, W, U, b):
    """Keras LSTM, gate order i, f, c, o; hard_sigmoid recurrent activation."""
    n, steps, _ = x.shape
    units = U.shape[0]
    h, c = np.zeros((n, units)), np.zeros((n, units))
    seq = []
    for t in range(steps):
        z = x[:, t] @ W + h @ U + b
        i, f, g, o = (z[:, k * units:(k + 1) * units] for k in range(4))
        c = hard_sigmoid(f) * c + hard_sigmoid(i) * np.tanh(g)
        h = hard_sigmoid(o) * np.tanh(c)
        seq.append(h)
    return np.stack(seq, axis=1)


def gru_layer(x, W, U, b):
    """Keras 2.1 GRU (reset gate applied before the matmul), gate order z, r, h."""
    n, steps, _ = x.shape
    units = U.shape[0]
    h = np.zeros((n, units))
    Wz, Wr, Wh = (W[:, k * units:(k + 1) * units] for k in range(3))
    Uz, Ur, Uh = (U[:, k * units:(k + 1) * units] for k in range(3))
    bz, br, bh = (b[k * units:(k + 1) * units] for k in range(3))
    seq = []
    for t in range(steps):
        xt = x[:, t]
        z = hard_sigmoid(xt @ Wz + h @ Uz + bz)
        r = hard_sigmoid(xt @ Wr + h @ Ur + br)
        hh = np.tanh(xt @ Wh + (r * h) @ Uh + bh)
        h = z * h + (1 - z) * hh
        seq.append(h)
    return np.stack(seq, axis=1)


def predict(path, X):
    w, config = weights(path)
    layers = config["config"] if isinstance(config["config"], list) else config["config"]["layers"]
    out = X[:, :, None]
    for layer in layers:
        kind, cfg = layer["class_name"], layer["config"]
        name = cfg["name"]
        if kind in ("LSTM", "GRU"):
            assert cfg["recurrent_activation"] == "hard_sigmoid" and cfg["activation"] == "tanh"
            fn = lstm_layer if kind == "LSTM" else gru_layer
            out = fn(out, w[f"kernel@{name}"], w[f"recurrent_kernel@{name}"], w[f"bias@{name}"])
            if not cfg["return_sequences"]:
                out = out[:, -1]
        elif kind == "Dense":
            if out.ndim == 3:
                out = out[:, :, 0]
            out = out @ w[f"kernel@{name}"] + w[f"bias@{name}"]
            act = cfg["activation"]
            assert act in ("linear", "sigmoid"), act
            out = sigmoid(out) if act == "sigmoid" else out
        elif kind == "Activation":
            assert cfg["activation"] == "sigmoid"
            out = sigmoid(out)
        elif kind == "Dropout":
            pass  # inactive at inference
        else:
            raise ValueError(f"unhandled layer {kind}")
    return out.reshape(-1)


# --- case ---------------------------------------------------------------------


def main() -> None:
    files = fetch_all(REPO, COMMIT, FILES, HERE / ".cache")
    train = pd.read_csv(files["data/train.csv"], encoding="utf-8-sig").fillna(0)
    test = pd.read_csv(files["data/test.csv"], encoding="utf-8-sig").fillna(0)
    t_train, t_test = (pd.to_datetime(d["5 Minutes"], format="%d/%m/%Y %H:%M") for d in (train, test))
    print(f"train {t_train.min()} .. {t_train.max()}   test {t_test.min()} .. {t_test.max()}")

    # their preprocessing: MinMaxScaler fit on train flow, 12-step windows over the test file
    lo, hi = train[ATTR].min(), train[ATTR].max()
    flow = ((test[ATTR] - lo) / (hi - lo)).to_numpy(float)
    X = np.stack([flow[i - LAG:i] for i in range(LAG, len(flow))])

    # one row per 5-minute step; the first LAG rows are history only, as in the original
    df = pd.DataFrame({"time": t_test, "flow": test[ATTR].astype(float)})
    df["split"] = ["history"] * LAG + ["test"] * (len(df) - LAG)
    reproduced = {}
    for name in REPORTED:
        pred = predict(files[f"model/{name.lower()}.h5"], X) * (hi - lo) + lo
        df[name] = np.r_[np.full(LAG, np.nan), pred]
        y = df["flow"].to_numpy()[LAG:]
        reproduced[name] = {
            "mae": float(np.mean(np.abs(y - pred))),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
        }

    sections = [
        "# Case 02: xiaochus/TrafficFlowPrediction (PeMS freeway flow)",
        "",
        f"Source: [{REPO}@{COMMIT[:7]}](https://github.com/{REPO}/tree/{COMMIT}). "
        "Audited with the repo's own trained weights.",
        "",
        "## Reproduction check (README vs. weights run here)",
        "",
        "| model | MAE README / here | RMSE README / here | R² README / here |",
        "|---|---|---|---|",
    ]
    for name, rep in REPORTED.items():
        got = reproduced[name]
        sections.append(
            f"| {name} | {rep['mae']} / {got['mae']:.2f} | {rep['rmse']} / {got['rmse']:.2f} "
            f"| {rep['r2']} / {got['r2']:.4f} |"
        )
    print("\n".join(sections[-5:]))

    for name in REPORTED:
        for metric in ("mae", "r2"):
            result = audit(
                df,
                "flow",
                task="regression",
                pred=name,
                time="time",
                split="split",
                season=288,
                metric=metric,
                reported=REPORTED[name][metric],
            )
            print(f"\n=== {name} / {metric} ===\n{report.to_text(result)}")
            sections += ["", f"## {name}, headline metric {metric.upper()}", "", report.to_markdown(result)]

    (HERE / "RESULTS.md").write_text("\n".join(sections) + "\n")
    if "--export" in sys.argv:
        scored = df.iloc[LAG:]
        y = scored["flow"].to_numpy()
        persistence = df["flow"].shift(1).iloc[LAG:].to_numpy()
        ma3 = df["flow"].shift(1).rolling(3).mean().iloc[LAG:].to_numpy()
        r2 = lambda p: 1 - np.nansum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)  # noqa: E731
        mae = lambda p: float(np.nanmean(np.abs(y - p)))  # noqa: E731
        print("exported", export_json("case02", {
            "reported": REPORTED["SAEs"],
            "scores": {name: {"mae": round(mae(p), 4), "r2": round(float(r2(p)), 4)} for name, p in
                       [("SAEs", scored["SAEs"].to_numpy()), ("persistence", persistence), ("moving average (3)", ma3)]},
            "minutes_per_step": 5,
            "y": y[:864], "model": scored["SAEs"].to_numpy()[:864], "persistence": persistence[:864],
        }))
    print(f"\nwrote {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
