# Case 06: Azure VM CPU Utilization (LSTM / GRU / IndRNN)

Source: [BiswajitPadhi99/predicting-cloud-CPU-utilization-on-Azure-dataset-using-deeplearning@6c5d567](https://github.com/BiswajitPadhi99/predicting-cloud-CPU-utilization-on-Azure-dataset-using-deeplearning/tree/6c5d567a077eaf6b3e1050e7ab1c5cfc0b658048). Reported-score audit on the Microsoft Azure 2017 5-minute VM CPU trace (5,166 scored cells).

## Summary (raw CPU units, 1,722 test steps × 3 series)
| Forecaster | Test RMSE | Test MAE | Test MAPE | Verdict vs baselines |
|---|---:|---:|---:|---|
| GRU (128) | 28,808.21 | 17,730.62 (printed √MAE: 133.16) | 1.17% | **FAIL** |
| LSTM (128) | 29,059.44 | 16,940.85 (printed √MAE: 130.16) | 1.12% | **FAIL** |
| IndRNN (128) | 29,480.69 | 18,711.58 (printed √MAE: 136.79) | 1.25% | **FAIL** |
| **Linear AR(5) (causal Ridge)** | **27,920.34** | **15,816.19** | **1.03%** | baseline |
| **EWMA (3)** | **29,320.05** | **16,980.21** | — | baseline |
| **Persistence (lag 1)** | **30,440.71** | **17,072.21** | **1.11%** | baseline |


## Raw units: GRU (128)

**Verdict: FAIL**

- target: cpu   task: regression   horizon: 1   metric: rmse   rows scored: 5166
- seasonal period: 288 steps

| model | mae | rmse | r2 | smape |
|---|---|---|---|---|
| persistence | 17072.2117 | 30440.7079 | 0.9978 | 0.0111 |
| moving average (3) | 17751.8432 | 30995.0979 | 0.9977 | 0.0117 |
| training mean | 546618.2690 | 646544.9908 | -0.0119 | 0.3901 |
| ewma (3) | 16980.2129 | 29320.0477 | 0.9979 | 0.0112 |
| seasonal naive (m=288) | 52141.2014 | 71580.3012 | 0.9876 | 0.0398 |
| linear AR(5)  <- best baseline | 15816.1916 | 27920.3426 | 0.9981 | 0.0104 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.999, persistence rmse 30440.708). Any model will post a high score here; only the margin over persistence means anything. |
| reported score | FAIL | Reported rmse 2.881e+04; linear AR(5) alone gets 2.792e+04 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |

## Raw units: LSTM (128)

**Verdict: FAIL**

- target: cpu   task: regression   horizon: 1   metric: rmse   rows scored: 5166
- seasonal period: 288 steps

| model | mae | rmse | r2 | smape |
|---|---|---|---|---|
| persistence | 17072.2117 | 30440.7079 | 0.9978 | 0.0111 |
| moving average (3) | 17751.8432 | 30995.0979 | 0.9977 | 0.0117 |
| training mean | 546618.2690 | 646544.9908 | -0.0119 | 0.3901 |
| ewma (3) | 16980.2129 | 29320.0477 | 0.9979 | 0.0112 |
| seasonal naive (m=288) | 52141.2014 | 71580.3012 | 0.9876 | 0.0398 |
| linear AR(5)  <- best baseline | 15816.1916 | 27920.3426 | 0.9981 | 0.0104 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.999, persistence rmse 30440.708). Any model will post a high score here; only the margin over persistence means anything. |
| reported score | FAIL | Reported rmse 2.906e+04; linear AR(5) alone gets 2.792e+04 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |

## Raw units: IndRNN (128)

**Verdict: FAIL**

- target: cpu   task: regression   horizon: 1   metric: rmse   rows scored: 5166
- seasonal period: 288 steps

| model | mae | rmse | r2 | smape |
|---|---|---|---|---|
| persistence | 17072.2117 | 30440.7079 | 0.9978 | 0.0111 |
| moving average (3) | 17751.8432 | 30995.0979 | 0.9977 | 0.0117 |
| training mean | 546618.2690 | 646544.9908 | -0.0119 | 0.3901 |
| ewma (3) | 16980.2129 | 29320.0477 | 0.9979 | 0.0112 |
| seasonal naive (m=288) | 52141.2014 | 71580.3012 | 0.9876 | 0.0398 |
| linear AR(5)  <- best baseline | 15816.1916 | 27920.3426 | 0.9981 | 0.0104 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.999, persistence rmse 30440.708). Any model will post a high score here; only the margin over persistence means anything. |
| reported score | FAIL | Reported rmse 2.948e+04; linear AR(5) alone gets 2.792e+04 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |

## Scaled units (`model.evaluate`): LSTM (128)

**Verdict: FAIL**

- target: cpu_scaled   task: regression   horizon: 1   metric: rmse   rows scored: 5166
- seasonal period: 288 steps

| model | mae | rmse | r2 | smape |
|---|---|---|---|---|
| persistence | 0.0155 | 0.0228 | 0.9622 | 0.0580 |
| moving average (3) | 0.0165 | 0.0238 | 0.9587 | 0.0599 |
| training mean | 0.1080 | 0.1402 | -0.4340 | 0.3753 |
| ewma (3) | 0.0157 | 0.0224 | 0.9633 | 0.0574 |
| seasonal naive (m=288) | 0.0554 | 0.0728 | 0.6133 | 0.2060 |
| linear AR(5)  <- best baseline | 0.0148 | 0.0215 | 0.9663 | 0.0545 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.981, persistence rmse 0.023). Any model will post a high score here; only the margin over persistence means anything. |
| reported score | FAIL | Reported rmse 0.02277; linear AR(5) alone gets 0.02149 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |

## Scaled units (`model.evaluate`): IndRNN (128)

**Verdict: FAIL**

- target: cpu_scaled   task: regression   horizon: 1   metric: rmse   rows scored: 5166
- seasonal period: 288 steps

| model | mae | rmse | r2 | smape |
|---|---|---|---|---|
| persistence | 0.0155 | 0.0228 | 0.9622 | 0.0580 |
| moving average (3) | 0.0165 | 0.0238 | 0.9587 | 0.0599 |
| training mean | 0.1080 | 0.1402 | -0.4340 | 0.3753 |
| ewma (3) | 0.0157 | 0.0224 | 0.9633 | 0.0574 |
| seasonal naive (m=288) | 0.0554 | 0.0728 | 0.6133 | 0.2060 |
| linear AR(5)  <- best baseline | 0.0148 | 0.0215 | 0.9663 | 0.0545 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.981, persistence rmse 0.023). Any model will post a high score here; only the margin over persistence means anything. |
| reported score | FAIL | Reported rmse 0.0235; linear AR(5) alone gets 0.02149 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |

