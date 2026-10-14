# Case 03: THU-INSC-NAD/Traffic-Matrix-Prediction

**Verdict: FAIL.** On all three backbone networks, the per-OD LSTM is significantly worse than "the next 5/15 minutes
look like the last ones." It beats every trivial baseline on only 14% of the OD pairs that carry traffic in the test period.

- **Source:** [THU-INSC-NAD/Traffic-Matrix-Prediction@3f30e58](https://github.com/THU-INSC-NAD/Traffic-Matrix-Prediction/tree/3f30e5825692d6683ed38b6089809b66e95110b1),
  code for *Traffic Matrix Prediction Based on Deep Learning for Dynamic Traffic Engineering* (ISCC 2019)
- **Data:** the three standard public traffic-matrix datasets, in the files each script names:
  Abilene (23 days, 5 min), GÉANT (36 days, 15 min) and CERNET (21 days, 5 min)
- **Model:** one LSTM per OD pair (`<network>/LSTM_predict_OD.py`), 10 lags → next value
- **Claim:** the paper's tables are not in the repo and the paper isn't openly available, so this audits the
  repo's protocol rather than a printed number (as in case 04)

## How it was audited

This is a **full audit** of a PyTorch re-implementation that follows the scripts line by line: min-max scaling per OD
over the full series, train windows = ⌊0.8·n / 50⌋·50, `nn.LSTM(1, 30)` + `Linear(30, 1)`, Adagrad (lr 0.01), batch 50,
100 epochs (Abilene) or 50 (GÉANT, CERNET), and negative test predictions flipped positive. Self-pairs are skipped,
as in the scripts. Constant all-zero OD pairs (8 in GÉANT, 27 in CERNET) are excluded because there is nothing to forecast.

All 785 remaining OD models are trained, using a stack of independent models per network. A check confirmed that a
model trained in the stack matches the same model trained alone to within 1e-7. Each OD pair is audited on its own
(seasonal period estimated from training rows), and each network is audited pooled across OD pairs in raw traffic units
(seasonal period = 1 day). A linear AR(10) on the same windows is included as a non-trivial reference.

## Findings

**Pooled over each network (MAE, raw units, test period)**

| Network | LSTM | Linear AR(10) | Persistence | LSTM vs persistence (95% CI) |
|---|---|---|---|---|
| Abilene | 2,646 | 2,231 | **2,229** | +19% worse (CI excludes 0) |
| GÉANT | 1,579 | 1,460 | **1,427** | +11% worse (CI excludes 0) |
| CERNET | 7,697 | **5,636** | 6,346 | +21% worse (CI excludes 0) |

**Per OD pair: does the LSTM beat every trivial baseline?**

| Network | PASS | WARN (within noise) | FAIL | Silent in test (excluded) |
|---|---|---|---|---|
| Abilene (132) | 31 | 34 | 67 | 0 |
| GÉANT (498) | 44 | 104 | 323 | 27 |
| CERNET (155) | 29 | 17 | 99 | 10 |
| **Total (748 with test traffic)** | **104 (14%)** | 155 (21%) | **489 (65%)** | 37 |

37 OD pairs carry traffic during training but are constant zero for the whole test period. Persistence is trivially
perfect there, so they say nothing about the model and are excluded. (The first version of this table counted them as
failures; the tool now reports a constant test target as INFO instead.)

1. **Persistence is the bar, and the LSTM doesn't clear it.** The best trivial baseline is persistence for 635 of the 748 active
   OD pairs and a 3-step moving average for most of the rest. Daily seasonality barely matters at 5–15-minute horizons.
2. **A linear model does better than the LSTM everywhere.** AR(10) on the same 10 lags ties persistence on Abilene,
   is slightly behind on GÉANT, and beats it on CERNET (−11% MAE). The LSTM is behind AR(10) on all three.
3. **This isn't a tuning artefact.** The scripts' comments call `hidden_size = 100, LR = 0.065` the "best parameters",
   while `__main__` uses 30 / 0.01. With the commented settings on Abilene, the LSTM gets worse pooled (MAE 2,866) and the
   per-OD tally is essentially unchanged (30 PASS / 35 WARN / 67 FAIL). See `RESULTS_best_params.md`.
4. **Why it matters downstream.** The paper feeds these predictions into traffic engineering (OSPF weights, MCF split
   ratios). A predictor that is worse than "use the last measured matrix" would make the TE pipeline worse than the
   reactive baseline it is meant to improve on. That deserves a direct check, but it is out of scope here.

## Caveats

- This is a re-implementation; the authors' trained models and result files were removed from the repo. One seed was used.
- One shuffle order is shared across OD pairs in each stack instead of one per OD. That doesn't change the expected outcome.
- The scaler is fit on the whole series, including test (as in the scripts). This leaks slightly toward the model, and the baselines are unaffected.
- The paper's own reported numbers couldn't be checked.

## Reproduce

```bash
uv run --group cases python cases/03_thu_traffic_matrix/run.py                     # ~25 min CPU
uv run --group cases python cases/03_thu_traffic_matrix/run.py --networks abilene \
    --hidden 100 --lr 0.065 --out RESULTS_best_params.md                           # sensitivity, ~20 min
```
