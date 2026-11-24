"""Benchmark Case 03's batched StackedLSTM vs sequential per-OD torch.nn.LSTM training.

Verifies numerical equivalence under elementwise Adagrad and measures exact wall-time speedup.

Usage:
    uv run --group cases python benchmarks/bench_stacked_lstm.py [--out benchmarks/stacked_lstm.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cases" / "03_thu_traffic_matrix"))
from run import StackedLSTM  # noqa: E402


def bench(
    n_ods: int = 132,
    n_win: int = 800,
    k: int = 10,
    hidden: int = 30,
    epochs: int = 10,
    batch: int = 50,
) -> dict:
    rng = np.random.default_rng(0)
    X_np = rng.random((n_ods, n_win, k), dtype=np.float32)
    Y_np = rng.random((n_ods, n_win), dtype=np.float32)
    Xt = torch.tensor(X_np)
    Yt = torch.tensor(Y_np)

    # Pre-generate identical epoch permutations
    orders = [torch.tensor(rng.permutation(n_win)) for _ in range(epochs)]

    # 1. Batched StackedLSTM
    torch.manual_seed(0)
    stacked = StackedLSTM(n_ods, hidden)
    init_state = {k_: v.detach().clone() for k_, v in stacked.named_parameters()}
    opt_s = torch.optim.Adagrad(stacked.parameters(), lr=0.01)

    t0 = time.perf_counter()
    for order in orders:
        for s in range(0, n_win, batch):
            idx = order[s:s + batch]
            opt_s.zero_grad()
            loss = ((stacked(Xt[:, idx]) - Yt[:, idx]) ** 2).mean(dim=1).sum()
            loss.backward()
            opt_s.step()
    t_stacked = time.perf_counter() - t0
    with torch.no_grad():
        pred_stacked = stacked(Xt[:, :64]).numpy()

    # 2. Sequential per-OD nn.LSTM + nn.Linear with identical initial weights
    models, opts = [], []
    for j in range(n_ods):
        lstm = nn.LSTM(1, hidden, batch_first=True)
        head = nn.Linear(hidden, 1)
        with torch.no_grad():
            lstm.weight_ih_l0.copy_(init_state["w_ih"][j].T)
            lstm.weight_hh_l0.copy_(init_state["w_hh"][j].T)
            lstm.bias_ih_l0.copy_(init_state["b_ih"][j].squeeze(0))
            lstm.bias_hh_l0.copy_(init_state["b_hh"][j].squeeze(0))
            head.weight.copy_(init_state["w_out"][j].T)
            head.bias.copy_(init_state["b_out"][j].squeeze(0))
        mod = nn.ModuleList([lstm, head])
        models.append(mod)
        opts.append(torch.optim.Adagrad(mod.parameters(), lr=0.01))

    t1 = time.perf_counter()
    for j in range(n_ods):
        lstm, head = models[j]
        opt = opts[j]
        xj = Xt[j].unsqueeze(-1)
        yj = Yt[j]
        for order in orders:
            for s in range(0, n_win, batch):
                idx = order[s:s + batch]
                opt.zero_grad()
                out, _ = lstm(xj[idx])
                pred = head(out[:, -1]).squeeze(-1)
                loss = ((pred - yj[idx]) ** 2).mean()
                loss.backward()
                opt.step()
    t_seq = time.perf_counter() - t1

    pred_seq = np.empty_like(pred_stacked)
    with torch.no_grad():
        for j in range(n_ods):
            lstm, head = models[j]
            out, _ = lstm(Xt[j, :64].unsqueeze(-1))
            pred_seq[j] = head(out[:, -1]).squeeze(-1).numpy()

    max_abs_diff = float(np.max(np.abs(pred_stacked - pred_seq)))
    return {
        "n_ods": n_ods,
        "n_win": n_win,
        "epochs": epochs,
        "stacked_s": round(t_stacked, 3),
        "sequential_s": round(t_seq, 3),
        "speedup_x": round(t_seq / t_stacked, 2),
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = bench()
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
