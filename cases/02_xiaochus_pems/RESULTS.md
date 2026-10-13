# Case 02: xiaochus/TrafficFlowPrediction (PeMS freeway flow)

Source: [xiaochus/TrafficFlowPrediction@dec8243](https://github.com/xiaochus/TrafficFlowPrediction/tree/dec8243eb4ebe728d44d1ac73f38c2d3aaa9fd81). Audited with the repo's own trained weights.

## Reproduction check (README vs. weights run here)

| model | MAE README / here | RMSE README / here | R² README / here |
|---|---|---|---|
| LSTM | 7.21 / 7.22 | 9.9 / 9.90 | 0.9396 / 0.9396 |
| GRU | 7.2 / 7.20 | 9.97 / 9.97 | 0.9389 / 0.9389 |
| SAEs | 7.06 / 7.06 | 9.6 / 9.60 | 0.9433 / 0.9433 |

## LSTM, headline metric MAE

**Verdict: WARN**

- target: flow   task: regression   horizon: 1   metric: mae   rows scored: 4032
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| LSTM  <- model | 7.1639 | 9.8684 | 0.9399 |
| persistence | 8.3234 | 11.2949 | 0.9212 |
| moving average (3)  <- best baseline | 7.7722 | 10.6757 | 0.9296 |
| training mean | 60.3435 | 71.9230 | -2.1933 |
| seasonal naive (m=288) | 10.3271 | 14.2139 | 0.8753 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.961, persistence mae 8.323). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: mae reduction vs moving average (3) = +0.6082 (95% block-bootstrap CI +0.3950 to +0.8606). |
| reported score | PASS | Reported mae 7.21; moving average (3) alone gets 7.772 on the same rows. |
| temporal split | INFO | No rows are labelled 'train', so there is no overlap to check. |

## LSTM, headline metric R2

**Verdict: WARN**

- target: flow   task: regression   horizon: 1   metric: r2   rows scored: 4032
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| LSTM  <- model | 7.1639 | 9.8684 | 0.9399 |
| persistence | 8.3234 | 11.2949 | 0.9212 |
| moving average (3)  <- best baseline | 7.7722 | 10.6757 | 0.9296 |
| training mean | 60.3435 | 71.9230 | -2.1933 |
| seasonal naive (m=288) | 10.3271 | 14.2139 | 0.8753 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.961, persistence r2 0.921). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: r2 gain over moving average (3) = +0.0102 (95% block-bootstrap CI +0.0059 to +0.0153). |
| reported score | WARN | Reported r2 0.9396; moving average (3) alone gets 0.9296 on the same rows. The claimed improvement is marginal. |
| temporal split | INFO | No rows are labelled 'train', so there is no overlap to check. |

## GRU, headline metric MAE

**Verdict: WARN**

- target: flow   task: regression   horizon: 1   metric: mae   rows scored: 4032
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 7.1652 | 9.9662 | 0.9387 |
| persistence | 8.3234 | 11.2949 | 0.9212 |
| moving average (3)  <- best baseline | 7.7722 | 10.6757 | 0.9296 |
| training mean | 60.3435 | 71.9230 | -2.1933 |
| seasonal naive (m=288) | 10.3271 | 14.2139 | 0.8753 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.961, persistence mae 8.323). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: mae reduction vs moving average (3) = +0.6070 (95% block-bootstrap CI +0.3713 to +0.8638). |
| reported score | PASS | Reported mae 7.2; moving average (3) alone gets 7.772 on the same rows. |
| temporal split | INFO | No rows are labelled 'train', so there is no overlap to check. |

## GRU, headline metric R2

**Verdict: WARN**

- target: flow   task: regression   horizon: 1   metric: r2   rows scored: 4032
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 7.1652 | 9.9662 | 0.9387 |
| persistence | 8.3234 | 11.2949 | 0.9212 |
| moving average (3)  <- best baseline | 7.7722 | 10.6757 | 0.9296 |
| training mean | 60.3435 | 71.9230 | -2.1933 |
| seasonal naive (m=288) | 10.3271 | 14.2139 | 0.8753 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.961, persistence r2 0.921). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: r2 gain over moving average (3) = +0.0090 (95% block-bootstrap CI +0.0037 to +0.0145). |
| reported score | WARN | Reported r2 0.9389; moving average (3) alone gets 0.9296 on the same rows. The claimed improvement is marginal. |
| temporal split | INFO | No rows are labelled 'train', so there is no overlap to check. |

## SAEs, headline metric MAE

**Verdict: WARN**

- target: flow   task: regression   horizon: 1   metric: mae   rows scored: 4032
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| SAEs  <- model | 7.0116 | 9.5662 | 0.9435 |
| persistence | 8.3234 | 11.2949 | 0.9212 |
| moving average (3)  <- best baseline | 7.7722 | 10.6757 | 0.9296 |
| training mean | 60.3435 | 71.9230 | -2.1933 |
| seasonal naive (m=288) | 10.3271 | 14.2139 | 0.8753 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.961, persistence mae 8.323). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: mae reduction vs moving average (3) = +0.7606 (95% block-bootstrap CI +0.5700 to +0.9957). |
| reported score | PASS | Reported mae 7.06; moving average (3) alone gets 7.772 on the same rows. |
| temporal split | INFO | No rows are labelled 'train', so there is no overlap to check. |

## SAEs, headline metric R2

**Verdict: WARN**

- target: flow   task: regression   horizon: 1   metric: r2   rows scored: 4032
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| SAEs  <- model | 7.0116 | 9.5662 | 0.9435 |
| persistence | 8.3234 | 11.2949 | 0.9212 |
| moving average (3)  <- best baseline | 7.7722 | 10.6757 | 0.9296 |
| training mean | 60.3435 | 71.9230 | -2.1933 |
| seasonal naive (m=288) | 10.3271 | 14.2139 | 0.8753 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.961, persistence r2 0.921). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: r2 gain over moving average (3) = +0.0139 (95% block-bootstrap CI +0.0100 to +0.0192). |
| reported score | WARN | Reported r2 0.9433; moving average (3) alone gets 0.9296 on the same rows. The claimed improvement is marginal. |
| temporal split | INFO | No rows are labelled 'train', so there is no overlap to check. |
