# Case 06: Azure VM CPU Utilization (Cloud Workload Forecasting)

- **Repository:** [BiswajitPadhi99/predicting-cloud-CPU-utilization-on-Azure-dataset-using-deeplearning](https://github.com/BiswajitPadhi99/predicting-cloud-CPU-utilization-on-Azure-dataset-using-deeplearning)
- **Pinned commit:** `6c5d567a077eaf6b3e1050e7ab1c5cfc0b658048`
- **Domain:** Cloud infrastructure (Microsoft Azure 2017 public VM trace — 30 days at 5-minute granularity: aggregate `min cpu`, `max cpu`, `avg cpu`)
- **Tier:** Reported-score audit (exact test cells & scaling reproduced from committed notebook outputs)

## Verdict: **FAIL**

1. **Persistence alone reaches $R^2 = 0.9978$ and 1.11% MAPE** (lag-1 autocorrelation `0.999`). The published LSTM, GRU, and IndRNN notebooks report 1.12%–1.25% MAPE—equal to or *worse* than repeating the previous 5-minute reading.
2. **In the notebooks' own scaled `model.evaluate` metrics**, persistence beats *both* LSTM (scaled MAE `0.0155` vs. `0.0158`, scaled RMSE `0.02276` vs. `0.02277`) and IndRNN (scaled MAE `0.0155` vs. `0.0174`, scaled RMSE `0.02276` vs. `0.02350`).
3. **A causal 1-line EWMA(3) beats the 128-unit IndRNN** on raw RMSE (`29,320` vs. `29,481`), and a closed-form **causal Ridge `Linear AR(5)`** on the exact same 5-step lookback beats **all three deep networks** on both raw RMSE (`27,920` vs. `28,808`–`29,481`) and true MAE (`15,816` vs. `16,941`–`18,712`).
4. **Two metric bugs in the notebooks inflate apparent accuracy:**
   - All three notebooks compute `math.sqrt(mean_absolute_error(testY, testPredict))`, printing `130.16–136.79 MAE` instead of the true `16,941–18,712 MAE` (130× lower than reality).
   - The GRU notebook computes `np.corrcoef(testPredict, testY)[0, 1]` on a `(1722, 3)` array, which correlates step 0 with step 1 of `testPredict` rather than predictions against ground truth, printing a spurious `1.000000 COR`.

## Caveats

- This is a **reported-score audit** comparing the notebooks' printed test metrics against causal baselines on the exact same 1,722 test steps (`5,166` cells across `min`, `max`, `avg` CPU); the Keras models were not retrained.
- The notebooks' LSTM and GRU slightly edge out *unsmoothed* lag-1 persistence on raw RMSE (`29,059` and `28,808` vs. `30,441`, ~5% lower), though both lose to causal `Linear AR(5)` (`27,920`) and `LSTM` loses to persistence in scaled units.
- The notebooks fit `MinMaxScaler` on the full 8,640-row trace before the 80/20 split (mild scaling leakage).

## Run

```bash
uv run python cases/06_azure_vm_cpu/run.py --export
```
