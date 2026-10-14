"""Case 03: THU-INSC-NAD/Traffic-Matrix-Prediction, per-OD LSTM traffic-matrix prediction (ISCC 2019 code).

Re-implements <network>/LSTM_predict_OD.py for Abilene, GÉANT and CERNET:
  data      OD_pair/<network>-OD_pair_<date>.csv, the files named in each script's __main__
  series    every OD pair except self-pairs (the scripts skip i-i), one independent model each
  scaling   min-max per OD over the whole series (as in read_data)
  windows   10 lags -> next value; train on the first floor(0.8 * n / 50) * 50 windows, test on the rest
  model     nn.LSTM(1, 30, 1 layer) + Linear(30, 1) on the last step, MSE, Adagrad lr 0.01,
            batch 50, shuffled; 100 epochs (Abilene) / 50 (GÉANT, CERNET); negative test outputs are
            flipped positive, as in the scripts

All OD models of one network are trained at once as a stack of independent models: the loss is
the sum of per-OD mean losses and Adagrad is elementwise, so each model gets exactly the update it
would get alone. (One shuffle order is shared across ODs instead of one per OD.)

    uv run --group cases python cases/03_thu_traffic_matrix/run.py [--networks abilene,geant,cernet]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _fetch import fetch  # noqa: E402

from predictive_audit import audit, report, wide_to_long  # noqa: E402

REPO = "THU-INSC-NAD/Traffic-Matrix-Prediction"
COMMIT = "3f30e5825692d6683ed38b6089809b66e95110b1"
NETWORKS = {
    # name: (file, sha256, epochs, steps per day)
    "abilene": ("OD_pair/Abilene-OD_pair_2004-08-01.csv",
                "6fc05315ebbf6d0ba697bc7d845fa302e7965967ebb43c1e2faf125af506e942", 100, 288),
    "geant": ("OD_pair/GEANT-OD_pair_2005-07-26.csv",
              "bd526578eabeb03401981ca72873c8d1bc0240cc856d6ac6a9d2eec4584cebb0", 50, 96),
    "cernet": ("OD_pair/CERNET-OD_pair_2013-03-01.csv",
               "77be8c91ee77c2b88bce950ecdbdde3d05571de8ec17ba1167db10833cec85eb", 50, 288),
}
K, HIDDEN, BATCH, LR = 10, 30, 50, 0.01


class StackedLSTM(torch.nn.Module):
    """M independent copies of nn.LSTM(1, H) + nn.Linear(H, 1), with PyTorch's default init."""

    def __init__(self, m, hidden=HIDDEN):
        super().__init__()
        b = 1 / math.sqrt(hidden)
        u = lambda *shape: torch.nn.Parameter(torch.empty(*shape).uniform_(-b, b))  # noqa: E731
        self.w_ih, self.w_hh = u(m, 1, 4 * hidden), u(m, hidden, 4 * hidden)
        self.b_ih, self.b_hh = u(m, 1, 4 * hidden), u(m, 1, 4 * hidden)
        self.w_out, self.b_out = u(m, hidden, 1), u(m, 1, 1)  # Linear default bound is also 1/sqrt(fan_in)
        self.hidden = hidden

    def forward(self, x):  # x: (M, B, K)
        m, n, steps = x.shape
        h = x.new_zeros(m, n, self.hidden)
        c = x.new_zeros(m, n, self.hidden)
        for t in range(steps):
            z = x[:, :, t:t + 1] @ self.w_ih + h @ self.w_hh + self.b_ih + self.b_hh
            i, f, g, o = z.chunk(4, dim=2)  # PyTorch gate order
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        return (h @ self.w_out + self.b_out).squeeze(2)  # (M, B)


def load(network):
    path, sha, epochs, per_day = NETWORKS[network]
    df = pd.read_csv(fetch(REPO, COMMIT, path, sha, HERE / ".cache"))
    values = df.drop(columns="time")
    self_pairs = [c for c in values if len(set(c.split("_")[1].split("-"))) == 1]
    values = values.drop(columns=self_pairs)
    constant = [c for c in values if values[c].max() == values[c].min()]
    return df["time"], values.drop(columns=constant).astype("float64"), len(constant), epochs, per_day


def train_and_predict(values, epochs, seed, hidden=HIDDEN, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = values.to_numpy().T  # (M, T)
    lo, hi = data.min(1, keepdims=True), data.max(1, keepdims=True)
    scaled = (data - lo) / (hi - lo)
    n_win = scaled.shape[1] - K
    X = np.stack([scaled[:, i:i + K] for i in range(n_win)], axis=1)  # (M, n_win, K)
    Y = scaled[:, K:]  # (M, n_win)
    train_len = int(int(n_win * 0.8) / 50) * 50

    Xt = torch.tensor(X[:, :train_len], dtype=torch.float32)
    Yt = torch.tensor(Y[:, :train_len], dtype=torch.float32)
    model = StackedLSTM(len(values.columns), hidden)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.tensor(rng.permutation(train_len))
        for s in range(0, train_len, BATCH):
            idx = order[s:s + BATCH]
            opt.zero_grad()
            per_od = ((model(Xt[:, idx]) - Yt[:, idx]) ** 2).mean(dim=1)
            per_od.sum().backward()
            opt.step()

    with torch.no_grad():
        pred = model(torch.tensor(X[:, train_len:], dtype=torch.float32)).abs().numpy()
    lstm = pred * (hi - lo) + lo

    # linear AR(10) on the same scaled windows, one least-squares fit per OD
    ar = np.empty_like(lstm)
    for j in range(len(values.columns)):
        A = np.c_[X[j, :train_len], np.ones(train_len)]
        coef, *_ = np.linalg.lstsq(A, Y[j, :train_len], rcond=None)
        ar[j] = (np.c_[X[j, train_len:], np.ones(n_win - train_len)] @ coef) * (hi[j] - lo[j]) + lo[j]
    return lstm, ar, train_len + K  # first row index that is a test target


def frame_for(values, preds, first_test):
    wide = values.copy()
    wide["split"] = np.where(np.arange(len(values)) < first_test, "train", "test")
    long = wide_to_long(wide, keep=["split"], series_name="od", value_name="y")
    for name, p in preds.items():
        col = np.full((len(values.columns), len(values)), np.nan)
        col[:, first_test:] = p
        long[name] = col.ravel()  # wide_to_long stacks column by column, in the same order
    return long


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", default="abilene,geant,cernet")
    ap.add_argument("--epochs", type=int, help="override (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=HIDDEN, help="the scripts' comment calls 100 'best'; __main__ uses 30")
    ap.add_argument("--lr", type=float, default=LR, help="the scripts' comment calls 0.065 'best'; __main__ uses 0.01")
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()

    rows, sections = [], ["# Case 03: THU-INSC-NAD/Traffic-Matrix-Prediction", ""]
    for network in args.networks.split(","):
        _, values, n_constant, epochs, per_day = load(network)
        epochs = args.epochs or epochs
        t0 = time.time()
        lstm, ar, first_test = train_and_predict(values, epochs, args.seed, args.hidden, args.lr)
        print(f"{network}: {values.shape[1]} ODs ({n_constant} constant ODs dropped), "
              f"{epochs} epochs, hidden {args.hidden}, lr {args.lr}, trained in {time.time() - t0:.0f}s")
        long = frame_for(values, {"LSTM": lstm, "Linear AR(10)": ar}, first_test)

        # network-wide: all ODs pooled, raw traffic units
        sections += [f"## {network}: {values.shape[1]} OD pairs, {n_constant} constant pairs excluded", ""]
        for model in ("LSTM", "Linear AR(10)"):
            r = audit(long, "y", task="regression", pred=model, groups="od", time="step",
                      split="split", metric="mae", season=per_day)
            print(f"\n=== {network} pooled / {model} ===\n{report.to_text(r)}")
            sections += [f"### Pooled over OD pairs: {model}", "", report.to_markdown(r), ""]

        # per OD pair
        for od, g in long.groupby("od", sort=False):
            for model in ("LSTM", "Linear AR(10)"):
                r = audit(g, "y", task="regression", pred=model, time="step", split="split",
                          metric="mae", season="auto", n_boot=200)
                beat = {c.name: c for c in r.checks}["beats trivial baselines"]
                rows.append({"network": network, "od": od, "model": model, "status": beat.status,
                             "mae": r.table[model]["mae"], "best_baseline": r.best_baseline,
                             "baseline_mae": r.table[r.best_baseline]["mae"],
                             "persistence_mae": r.table["persistence"]["mae"], "season": r.season})

    per_od = pd.DataFrame(rows)
    per_od.to_csv(HERE / (Path(args.out).stem + "_per_od.csv"), index=False)
    tally = (per_od.pivot_table(index=["network", "model"], columns="status", values="od",
                                aggfunc="count", fill_value=0).reset_index())
    winners = (per_od.groupby(["network", "model"])["best_baseline"]
               .agg(lambda s: s.str.replace(r" \(.*\)", "", regex=True).value_counts().to_dict()).reset_index())
    print("\n", tally.to_string(index=False), "\n", winners.to_string(index=False))
    (HERE / args.out).write_text("\n".join([
        "# Per-OD verdicts (beats trivial baselines?)", "", tally.to_markdown(index=False), "",
        "Which baseline was best, per OD:", "", winners.to_markdown(index=False), "", *sections]) + "\n")


if __name__ == "__main__":
    main()
