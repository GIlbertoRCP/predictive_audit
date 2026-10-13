# Case 02: xiaochus/TrafficFlowPrediction

**Verdict: holds up (WARN).** The models really do beat every trivial baseline; the headline R² overstates how much.

- **Source:** [xiaochus/TrafficFlowPrediction@dec8243](https://github.com/xiaochus/TrafficFlowPrediction/tree/dec8243eb4ebe728d44d1ac73f38c2d3aaa9fd81) (681★)
- **Task:** freeway traffic flow (PeMS, one detector, 5-minute counts), one step ahead from the previous 12 steps
- **Models:** LSTM, GRU and stacked autoencoders (Keras 2.1.4)
- **Claim (README):** MAE 7.06–7.21, R² 0.939–0.943

## How it was audited

The repo ships its trained weights, so this audit uses **their exact published models** rather than a retrained copy.
`run.py` re-implements the Keras 2.1 forward pass in numpy (including Keras's `hard_sigmoid` recurrent gate) and
applies the repo's own preprocessing: a min-max scaler fit on the training file, and 12-step windows over the test file.

**Reproduction check:** the weights run here match the README to two decimals for all three models
(e.g. SAEs MAE 7.06 / 7.06, R² 0.9433 / 0.9433). See `RESULTS.md`.

## Findings

| | MAE | R² |
|---|---|---|
| SAEs (best model) | **7.01** | **0.943** |
| LSTM | 7.16 | 0.940 |
| GRU | 7.17 | 0.939 |
| 3-step moving average | 7.77 | 0.930 |
| Persistence | 8.32 | 0.921 |
| Seasonal naive (same time yesterday) | 10.33 | 0.875 |

(Rows scored: 4,032 of 4,308. The seasonal-naive baseline needs one day of history, so the first test day is excluded from every row.)

1. **The models are genuinely better than trivial baselines.** All three cut MAE against a 3-step moving average
   by 0.6–0.8 vehicles per 5 minutes, about 8–10%. The 95% block-bootstrap intervals exclude zero.
2. **R² is the wrong headline.** Persistence alone scores R² 0.92 (lag-1 autocorrelation 0.96). Of the reported
   0.94, only about 0.01–0.02 is skill beyond a 3-step moving average. MAE relative to a naive baseline (MASE-style)
   would describe the result more honestly.
3. **Protocol is clean.** Train is Jan–Feb 2016 and test is March 2016, and the scaler is fit on train only.
   Training windows are shuffled *within* the training set, which is harmless.
4. **Minor:** the test file holds 15 non-consecutive days (mostly weekdays with gaps). The windows treat the gaps as
   continuous, so about 14 day boundaries produce windows that span missing days. That's too small to change the conclusions.

## Caveats

- The "training mean" baseline is fit only on the 12 warm-up rows before the first test window (the original
  training file isn't part of the scored frame), so its very large error is not meaningful. It doesn't affect the verdict.
- One detector and one month of test data: the result says nothing about other detectors or seasons.

## Reproduce

```bash
uv run --group cases python cases/02_xiaochus_pems/run.py
```

Downloads the pinned data and weights (SHA-256 verified) into `.cache/`.
