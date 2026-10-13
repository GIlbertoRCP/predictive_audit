# Case 05: Annongkhanh/Traffic-matrix-prediction

## Zamren testbed: reported MSE 0.04135 (RMSE 0.2033), 144 OD columns x 167 test steps

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 24048
- seasonal period: 10 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| persistence | 0.0755 | 0.1455 | 0.7904 |
| moving average (3)  <- best baseline | 0.0756 | 0.1408 | 0.8038 |
| training mean | 0.2614 | 0.3179 | -0.0010 |
| seasonal naive (m=10) | 0.0710 | 0.1461 | 0.7887 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.895; persistence scores rmse 0.145. |
| reported score | FAIL | Reported rmse 0.2033; moving average (3) alone gets 0.1408 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |

## GÉANT: reported MSE 0.02596 (RMSE 0.1611), 529 OD columns x 52 test steps

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 27508

| model | mae | rmse | r2 |
|---|---|---|---|
| persistence  <- best baseline | 0.0462 | 0.1011 | 0.6963 |
| moving average (3) | 0.0495 | 0.1017 | 0.6927 |
| training mean | 0.1466 | 0.1856 | -0.0233 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.848; persistence scores rmse 0.101. |
| reported score | FAIL | Reported rmse 0.1611; persistence alone gets 0.1011 on the same rows. The claim is not evidence of forecasting skill. |
| temporal split | PASS | Every test row comes after the training data in time. |
