# Case 04: SDN-TMprediction (ICC 2021)

**Verdict: FAIL on most series.** On replayed testbed traffic, "repeat the value from one replay loop ago" beats the
paper's RNNs on 9 of 13 OD pairs, including the one the notebook showcases, where it's 2.4× more accurate.

- **Source:** [duchuyle108/SDN-TMprediction@986c868](https://github.com/duchuyle108/SDN-TMprediction/tree/986c8689cc19c2cab5f9dfb3c2ef8b52a54a2fb7),
  code for *An AI-based Traffic Matrix Prediction Solution for Software-Defined Network* (IEEE ICC 2021)
- **Data:** `testbed_flat_tms.csv`, 6,257 one-minute traffic matrices from a Mininet BSO topology. Traffic comes
  from **tcpreplay looping pcap files** on each host, measured by the paper's POX application.
- **Models:** LSTM, BiLSTM and GRU (100 units, ReLU), 10 lags → next value, one OD pair at a time
- **Claim:** RNN variants compared with each other by scaled RMSE. No naive baseline appears in the repo or the README.

## How it was audited

This is a **full audit**. `run.py` re-implements `training/Model-training.ipynb` in PyTorch, matching:
- the Keras cell maths (ReLU activations, sigmoid gates, TF2 GRU `reset_after`), initialisers and Adam defaults;
- `validation_split=0.3` (Keras never trains on the last 30% of training windows);
- the notebook's quirks: the CSV's first data row is consumed as a header, the scaler is fit on the whole series,
  and `create_dataset` drops the final window.

**Faithfulness check, OD 52:** final train/validation MSE is LSTM 0.0088/0.0087, BiLSTM 0.0071/0.0071 and
GRU 0.0074/0.0071. The notebook logged 0.0081/0.0076, 0.0064/0.0061 and 0.0068/0.0064: same ranking, within about 10%.

The notebook's OD 52 plus 12 randomly drawn active OD pairs (seed 0) were each audited. The seasonal-naive baseline
uses a period estimated **from training rows only** (`season="auto"`).

## Findings

| OD | BiLSTM RMSE | Best baseline | Baseline RMSE | Verdict |
|---|---|---|---|---|
| **52 (notebook's choice)** | 0.085 | seasonal naive, 339 min | **0.035** | FAIL |
| 4 | 0.087 | seasonal naive, 134 min | **0.017** | FAIL |
| 9 | 0.049 | seasonal naive, 128 min | **0.038** | FAIL |
| 15 | 0.0257 | training mean | 0.0257 | FAIL (pure noise; nothing beats the mean) |
| 34 | 0.058 | seasonal naive, 250 min | **0.053** | FAIL |
| 53 | 0.067 | seasonal naive, 99 min | **0.031** | FAIL |
| 59 | 0.044 | seasonal naive, 210 min | **0.030** | FAIL |
| 96 | **0.040** | seasonal naive, 29 min | 0.054 | PASS |
| 118 | 0.055 | seasonal naive, 133 min | **0.039** | FAIL |
| 127 | 0.042 | seasonal naive, 21 min | 0.042 | WARN (tie) |
| 157 | 0.034 | seasonal naive, 100 min | **0.030** | FAIL |
| 158 | **0.035** | seasonal naive, 17 min | 0.043 | PASS |
| 179 | **0.034** | seasonal naive, 70 min | 0.045 | PASS |

LSTM and GRU follow the same pattern (each passes on the same 3 OD pairs). Across the 13 pairs, the median BiLSTM
error is 1.16× the best baseline's, ranging from 0.74× to 5.3×. Per-model numbers are in `RESULTS.md` and `RESULTS_summary.csv`.

1. **The testbed's traffic is a loop.** Each host replays a fixed pcap, so every OD series repeats with a period of
   roughly 1.5–6 hours (autocorrelation 0.84–0.94 at the replay lag). A 10-minute input window can't see that structure;
   a baseline that remembers one loop back can.
2. **The showcased series is the clearest failure.** OD 52 has lag-1 autocorrelation of just 0.06, so the RNNs
   learn something real compared with persistence (RMSE 0.085 vs. 0.12 for the mean). The replay baseline still gets 0.035.
3. **The RNNs do win somewhere.** On 3 of 13 pairs the estimated period is short (17–70 min) or the replay
   structure is weaker, and the RNNs beat it with CIs clear of zero.
4. **Lesson for testbed papers.** Replayed traffic makes a forecasting benchmark partly a memorisation benchmark.
   Reporting a seasonal-naive baseline at the replay period would have exposed this immediately.

## Caveats

- This is a re-implementation, not the authors' trained weights (the repo ships none). Final losses are close to the
  notebook's but not identical; a single seed was used.
- The paper's own numbers (in the ICC paper) are not reproduced here, only the repo's protocol. GÉANT, the paper's second
  dataset, isn't audited here; case 05 covers GÉANT with a different model.
- Period estimation can lock onto a sub-harmonic. The three PASS rows all have short estimated periods, so a better
  period estimate may turn some of them into FAILs. They are reported as measured.

## Reproduce

```bash
uv run --group cases python cases/04_sdn_tm_prediction/run.py --ods sample   # ~10 min on CPU
uv run --group cases python cases/04_sdn_tm_prediction/run.py               # OD 52 only, ~1 min
```
