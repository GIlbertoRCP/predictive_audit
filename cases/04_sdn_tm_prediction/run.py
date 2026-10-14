"""Case 04: duchuyle108/SDN-TMprediction, "An AI-based Traffic Matrix Prediction Solution for SDN" (ICC 2021).

Re-implements training/Model-training.ipynb in PyTorch and audits the result:
  data      testbed_flat_tms.csv (Mininet BSO topology, POX measurement app), read exactly as the
            notebook does (pd.read_csv with the first data row consumed as the header)
  series    one OD column at a time; the notebook uses column 52
  scaling   MinMaxScaler fit on the whole series (as in the notebook)
  split     first 80% train / last 20% test; 10 lags -> next value
  models    LSTM(100, relu), BiLSTM(100, relu), GRU(100, relu) on a (1 step x 10 features) input,
            Dense(1), Adam, MSE, 200 epochs, batch 32, validation_split=0.3 (Keras trains on the first
            70% of the training windows only), final-epoch weights

Also audits a plain linear AR(10) on the same windows as a non-trivial reference point.

    uv run --group cases python cases/04_sdn_tm_prediction/run.py [--ods 52,10,...] [--epochs 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _export import write as export_json  # noqa: E402
from _fetch import fetch  # noqa: E402

from predictive_audit import audit, baselines, report  # noqa: E402

REPO = "duchuyle108/SDN-TMprediction"
COMMIT = "986c8689cc19c2cab5f9dfb3c2ef8b52a54a2fb7"
DATA = ("dataset/testbed_flat_tms.csv", "d40e64d15029cb98c8f9f4b3d376d2d48dc66c96d480e439ae391836adefc957")
LOOK_BACK = 10


# --- Keras-equivalent models --------------------------------------------------


class KerasRNN(nn.Module):
    """Keras LSTM/GRU layer (activation=relu, sigmoid gates) over a (batch, steps, features) input."""

    def __init__(self, kind, n_in, units):
        super().__init__()
        self.kind, self.units = kind, units
        gates = 4 if kind == "lstm" else 3
        self.W = nn.Parameter(torch.empty(n_in, gates * units))
        self.U = nn.Parameter(torch.empty(units, gates * units))
        self.b = nn.Parameter(torch.zeros(gates * units))
        nn.init.xavier_uniform_(self.W)  # glorot_uniform
        nn.init.orthogonal_(self.U)
        if kind == "lstm":
            with torch.no_grad():
                self.b[units:2 * units] = 1.0  # unit_forget_bias
        else:
            self.b_rec = nn.Parameter(torch.zeros(3 * units))  # TF2 GRU: reset_after=True

    def forward(self, x):
        n, u = x.shape[0], self.units
        h = x.new_zeros(n, u)
        c = x.new_zeros(n, u)
        for t in range(x.shape[1]):
            xw = x[:, t] @ self.W + self.b
            if self.kind == "lstm":
                z = xw + h @ self.U
                i, f, g, o = z.split(u, dim=1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.relu(g)
                h = torch.sigmoid(o) * torch.relu(c)
            else:
                hu = h @ self.U + self.b_rec
                xz, xr, xh = xw.split(u, dim=1)
                hz, hr, hh = hu.split(u, dim=1)
                zt, rt = torch.sigmoid(xz + hz), torch.sigmoid(xr + hr)
                cand = torch.relu(xh + rt * hh)
                h = zt * h + (1 - zt) * cand
        return h


class Net(nn.Module):
    def __init__(self, kind, units=100):
        super().__init__()
        self.kind = kind
        if kind == "bilstm":
            self.fwd = KerasRNN("lstm", LOOK_BACK, units)
            self.bwd = KerasRNN("lstm", LOOK_BACK, units)
            width = 2 * units
        else:
            self.rnn = KerasRNN(kind, LOOK_BACK, units)
            width = units
        self.out = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if self.kind == "bilstm":
            h = torch.cat([self.fwd(x), self.bwd(torch.flip(x, dims=[1]))], dim=1)
        else:
            h = self.rnn(x)
        return self.out(h).squeeze(1)


def train(kind, X, y, epochs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_fit = int(len(X) * (1 - 0.3))  # Keras validation_split takes the LAST 30%, unshuffled
    Xf = torch.tensor(X[:n_fit], dtype=torch.float32).reshape(-1, 1, LOOK_BACK)
    yf = torch.tensor(y[:n_fit], dtype=torch.float32)
    model = Net(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)  # Keras Adam defaults
    for _ in range(epochs):
        order = torch.tensor(rng.permutation(n_fit))
        for start in range(0, n_fit, 32):
            idx = order[start:start + 32]
            opt.zero_grad()
            loss = torch.mean((model(Xf[idx]) - yf[idx]) ** 2)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X[n_fit:], dtype=torch.float32).reshape(-1, 1, LOOK_BACK)
        fit_mse = float(torch.mean((model(Xf) - yf) ** 2))
        val_mse = float(torch.mean((model(Xv) - torch.tensor(y[n_fit:], dtype=torch.float32)) ** 2))
    return model, fit_mse, val_mse


def windows(series):
    """The notebook's create_dataset: note it drops the last usable window (range(len - look_back - 1))."""
    X = np.stack([series[i:i + LOOK_BACK] for i in range(len(series) - LOOK_BACK - 1)])
    y = series[LOOK_BACK:len(series) - 1]
    return X, y


# --- case ---------------------------------------------------------------------


def run_od(df, od, epochs, seed):
    raw = df.iloc[:, od].astype("float32").to_numpy()
    lo, hi = raw.min(), raw.max()
    scaled = (raw - lo) / (hi - lo)
    n_train = int(len(scaled) * 0.8)
    Xtr, ytr = windows(scaled[:n_train])
    Xte, yte = windows(scaled[n_train:])

    preds, losses = {}, {}
    for kind, label in [("lstm", "Vanilla LSTM"), ("bilstm", "BiLSTM"), ("gru", "GRU")]:
        model, fit_mse, val_mse = train(kind, Xtr, ytr, epochs, seed)
        losses[label] = (fit_mse, val_mse)
        with torch.no_grad():
            preds[label] = model(torch.tensor(Xte, dtype=torch.float32).reshape(-1, 1, LOOK_BACK)).numpy()
    preds["Linear AR(10)"] = LinearRegression().fit(Xtr, ytr).predict(Xte)

    # long frame over the whole series, in the notebook's scaled units
    frame = pd.DataFrame({"y": scaled.astype(float)})
    frame["split"] = "train"
    test_rows = np.arange(n_train + LOOK_BACK, len(scaled) - 1)  # rows whose value is a test target
    frame.loc[n_train:, "split"] = "history"
    frame.loc[test_rows, "split"] = "test"
    for name, p in preds.items():
        frame[name] = np.nan
        frame.loc[test_rows, name] = p
    rmse = {name: float(np.sqrt(np.mean((yte - p) ** 2))) for name, p in preds.items()}
    return frame, rmse, losses


def export(frame, od, rmse):
    """Test-period series for the explorer: actual, BiLSTM, replay-loop baseline, and the training ACF."""
    fit = frame["split"] != "test"
    season = baselines.estimate_season(frame["y"], None, fit)
    seasonal = baselines.seasonal_naive(frame["y"], None, 1, season)
    test = frame["split"] == "test"
    train = frame.loc[frame["split"] == "train", "y"].to_numpy()
    centered = train - train.mean()
    acf = [float(centered[:-k] @ centered[k:] / (centered @ centered)) for k in range(1, 601)]
    persistence = frame["y"].shift(1)
    y = frame.loc[test, "y"].to_numpy()
    path = export_json("case04", {
        "od": od, "season": season, "minutes_per_step": 1,
        "rmse": {**rmse, "seasonal": float(np.sqrt(np.mean((y - seasonal[test].to_numpy()) ** 2))),
                 "persistence": float(np.sqrt(np.mean((y - persistence[test].to_numpy()) ** 2))),
                 "mean": float(np.sqrt(np.mean((y - train.mean()) ** 2)))},
        "y": y, "bilstm": frame.loc[test, "BiLSTM"].to_numpy(), "seasonal": seasonal[test].to_numpy(),
        "acf": acf,
    })
    print(f"exported {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ods", default="52", help="comma-separated column indices (the notebook uses 52), or 'sample'")
    ap.add_argument("--n-sample", type=int, default=12, help="extra active OD columns drawn with --ods sample")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--export", action="store_true", help="write site/data/case04.json for the first OD")
    args = ap.parse_args()

    path = fetch(REPO, COMMIT, DATA[0], DATA[1], HERE / ".cache")
    df = pd.read_csv(path)  # deliberately as in the notebook: first data row becomes the header

    sections = ["# Case 04: SDN-TMprediction (ICC 2021), testbed dataset", ""]
    summary = []
    if args.ods == "sample":
        active = [i for i in range(1, df.shape[1]) if (df.iloc[:, i] != 0).mean() > 0.9 and i != 52]
        ods = [52] + sorted(np.random.default_rng(args.seed).choice(active, args.n_sample, replace=False).tolist())
    else:
        ods = [int(o) for o in args.ods.split(",")]
    for od in ods:
        frame, rmse, losses = run_od(df, od, args.epochs, args.seed)
        if args.export and od == ods[0]:
            export(frame, od, rmse)
        print(f"OD {od} final train/val MSE: " + ", ".join(f"{k} {a:.4f}/{b:.4f}" for k, (a, b) in losses.items()))
        sections += [f"## OD column {od}", "", "Scaled RMSE (the paper's metric): "
                     + ", ".join(f"{k} {v:.4f}" for k, v in rmse.items()), "",
                     "Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, "
                     "BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): "
                     + ", ".join(f"{k} {a:.4f}/{b:.4f}" for k, (a, b) in losses.items()), ""]
        for name in rmse:
            r = audit(frame, "y", task="regression", pred=name, split="split", metric="rmse", season="auto")
            beat = {c.name: c for c in r.checks}["beats trivial baselines"]
            summary.append({"od": od, "model": name, "rmse": rmse[name], "best_baseline": r.best_baseline,
                            "baseline_rmse": r.table[r.best_baseline]["rmse"], "status": beat.status,
                            "season": r.season,
                            "lag1_autocorr": {c.name: c for c in r.checks}["label memory"].details["lag_autocorrelation"]})
            print(f"\n=== OD {od} / {name} ===\n{report.to_text(r)}")
            sections += [f"### {name}", "", report.to_markdown(r), ""]

    table = pd.DataFrame(summary)
    print("\n", table.to_string(index=False))
    (HERE / args.out).write_text("\n".join(["# Summary", "", table.to_markdown(index=False), "", *sections]) + "\n")
    table.to_csv(HERE / (Path(args.out).stem + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
