# Case 05: Annongkhanh/Traffic-matrix-prediction

**Verdict: FAIL.** On both datasets, the published LSTM's error is *higher* than that of a trivial baseline.

- **Source:** [Annongkhanh/Traffic-matrix-prediction@0ab331d](https://github.com/Annongkhanh/Traffic-matrix-prediction/tree/0ab331d932f6567464bd93738c2da11c265eb888)
- **Task:** predict the next full traffic matrix from the previous 10 (all OD pairs jointly)
- **Model:** LSTM(500) followed by five Dense(500) layers (2.6M–3.6M parameters), Keras, 20 epochs
- **Claims (notebook outputs):** test MSE **0.02596** on GÉANT and **0.04135** on the Mininet "Zamren" testbed, with no baseline shown

## How it was audited

This is a **reported-score audit**, with no retraining. `run.py` rebuilds the notebooks' exact evaluation set from the
repo's data: min-max scaling per OD column fit on the whole table, a chronological 80/20 split, and 10-matrix input
windows. That gives 529 OD columns × 52 test steps for GÉANT and 144 × 167 for Zamren, matching the notebooks' array
shapes. Trivial baselines are scored on the same cells, in the same scaled units, and compared with the square root of
the reported MSE.

## Findings

| Dataset | Reported LSTM | Persistence | 3-step moving average | Training mean |
|---|---|---|---|---|
| GÉANT (RMSE) | 0.161 | **0.101** | 0.102 | 0.186 |
| Zamren testbed (RMSE) | 0.203 | 0.146 | **0.141** | 0.318 |

1. **The LSTM loses to persistence.** On GÉANT, "the next matrix equals the current one" has an MSE of 0.0102,
   about 2.5× lower than the reported 0.0260. On the testbed, a 3-step moving average beats it by a similar margin.
2. **The model sits much closer to the training-mean baseline than to persistence.** Its error is roughly what you'd
   expect from a network that learned the average traffic level and little of its dynamics, which is plausible for a
   3.6M-parameter model trained for 20 epochs on 189 GÉANT windows (237, minus Keras's 20% validation split).
3. **No baseline anywhere in the repo.** The only reference point in the notebooks is a single plotted matrix.

## Caveats

- The notebooks read their CSVs from a personal Google Drive. The repo's `dataset/` files have the same shapes
  (309 × 530 and 885 × 145), but byte-identity with the Drive copies can't be verified.
- The min-max scaler is fit on the whole table, including the test period. That's mild leakage in the model's favour,
  and the baselines here use the same scaling.
- The GÉANT test window is short (52 steps, about 13 hours). The 2.5× gap is far too large to be noise, but the exact ratio is uncertain.

## Reproduce

```bash
uv run --group cases python cases/05_sdn_tm_lstm_geant_zamren/run.py
```
