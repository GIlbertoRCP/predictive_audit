# Phase 2 candidate shortlist

Goal: 10–15 public network / systems forecasting results, audited the same way as case 01.
Status: **approved 2026-10-13.** Third-party data is never committed; each case fetches it, pinned and checksummed, into a gitignored `.cache/`.

Progress: 02, 04, 05 done · 03, 06–10 to do.

## Audit tiers

- **Tier R: reported-score audit.** Needs only the data, the exact test split, and the published number.
  No retraining. Cheap, so it's the default first pass for every candidate.
- **Tier F: full audit.** Re-run (or faithfully re-implement) the model to get its per-row predictions, then
  run every check, including the bootstrap margin over baselines and split/leakage checks on their pipeline.

## Candidates

| # | Repo | Domain | Data in repo? | Claim to check | Tier | Notes |
|---|---|---|---|---|---|---|
| 02 ✅ | [xiaochus/TrafficFlowPrediction](https://github.com/xiaochus/TrafficFlowPrediction) (681★) | Road traffic (PeMS), LSTM/GRU/SAEs | yes (train/test CSV + trained .h5) | MAE 7.06–7.21, R² 0.94 | R done, F possible | **Spot-checked:** model beats a 3-step moving average (MAE 7.06 vs 7.77); the headline R² is mostly autocorrelation (moving average alone: 0.93). Adjacent domain, but very visible. Train set is shuffled *within* train only (fine). |
| 03 | [THU-INSC-NAD/Traffic-Matrix-Prediction](https://github.com/THU-INSC-NAD/Traffic-Matrix-Prediction) | Traffic matrices: Abilene, GÉANT, CERNET; LSTM/GRU/TCN/DBN | yes (OD-pair CSVs, ~100 MB) | Paper tables (ISSCC 2019) | F | Result files were removed from the repo, so retrain (PyTorch). Min-max scaler fit on the full series before splitting (mild leakage to note). Three networks in one case. |
| 04 ✅ | [duchuyle108/SDN-TMprediction](https://github.com/duchuyle108/SDN-TMprediction) | GÉANT TMs + Mininet testbed TMs; ICC 2021 paper | yes (48 MB + 6 MB) | Paper's prediction error | R then F | Same Mininet/POX setting as your old project, so it makes a strong narrative pair with case 01. |
| 05 ✅ | [Annongkhanh/Traffic-matrix-prediction](https://github.com/Annongkhanh/Traffic-matrix-prediction) | GÉANT + Mininet Zamren; LSTM | yes (small) | Notebook outputs | F | Small and Colab-based; quick to rerun. |
| 06 | [chuanting/STDenseNet](https://github.com/chuanting/STDenseNet) | Milan cellular traffic (Telecom Italia); IEEE Comm. Letters 2018 | yes (h5, 24 MB) | Paper RMSE/MAE | R | Well-cited paper; the Milan grid is the standard cellular benchmark. Has daily/weekly seasonality, so seasonal-naive matters here. |
| 07 | [VarunKurri/Cellular-Traffic-Prediction-using-LSTM](https://github.com/VarunKurri/Cellular-Traffic-Prediction-using-LSTM) | 4G LTE cell traffic, hourly, 1 year | yes (3 cells) | LSTM vs ARIMA | F | Small; hourly data with daily cycles, so seasonal-naive is the key baseline. |
| 08 | [CN-UPB/ml-traffic-prediction](https://github.com/CN-UPB/ml-traffic-prediction) | GÉANT / TOTEM link traffic; master's thesis | results only (data from TOTEM) | LSTM vs ARIMA results tables | R | University lab; a thesis has explicit evaluation tables. |
| 09 | [BiswajitPadhi99/…Azure…](https://github.com/BiswajitPadhi99/predicting-cloud-CPU-utilization-on-Azure-dataset-using-deeplearning) | Azure VM CPU utilization; LSTM/GRU/IndRNN | yes (650 KB) | Notebook metrics | F | Systems rather than networking; cloud workload forecasting has the same failure modes. |
| 10 | [xdbdilab/APM](https://github.com/xdbdilab/APM) | Alibaba 2018 + Google 2019 cluster traces | yes | Paper ("accurate prediction of cloud workloads") | R then F | Published paper with real production traces. |

## Not pursued (and why)

- **abdelhadi-azouni/NeuMat**: a toy example on airline passengers, not network data.
- **peymanbarca/deep-cloud-predictor**: data is "hint" files only (NASA/Calgary HTTP logs must be fetched); reconsider later.
- **RemexHu/BandwidthPrediction-LSTM, n13eho/Schaferct**: no data in the repo (MMSys challenge data is external).
- **Large road-traffic GNN repos (ASTGCN, D2STGNN…)**: high-profile, but the field already has baseline critiques, and GPU training is costly. A possible stretch case.

## Tool work this will need

1. ✅ **Wide → long helper** (`predictive_audit.wide_to_long`). Traffic matrices come as one column per OD pair. Add `--wide` (or a `melt` helper) so each pair becomes a group.
2. ✅ **Seasonal baselines.** `season="auto"` estimates the period from training rows only (added for case 04's replayed pcaps).
3. **Multi-step horizons.** Some papers predict k steps ahead; the target convention already supports it, so this needs tests only.
4. **Faithful reruns.** Pin each repo's framework where it still installs; otherwise re-implement in PyTorch and document every deviation in the case README.

## Suggested order

02 (finish as a "passes" example) → 04 + 05 (Mininet/SDN, closest to case 01) → 03 (three networks) → 06, 07 (cellular, seasonal) →
08–10 (systems/workload). That gives a mix of PASS / WARN / FAIL instead of a takedown list.
