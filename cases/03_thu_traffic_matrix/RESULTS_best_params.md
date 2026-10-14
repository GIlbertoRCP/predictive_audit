# Per-OD verdicts (beats trivial baselines?)

| network   | model         |   FAIL |   PASS |   WARN |
|:----------|:--------------|-------:|-------:|-------:|
| abilene   | LSTM          |     67 |     30 |     35 |
| abilene   | Linear AR(10) |     69 |     37 |     26 |

Which baseline was best, per OD:

| network   | model         | best_baseline                                                                       |
|:----------|:--------------|:------------------------------------------------------------------------------------|
| abilene   | LSTM          | {'persistence': 104, 'moving average': 26, 'seasonal naive': 1, 'training mean': 1} |
| abilene   | Linear AR(10) | {'persistence': 104, 'moving average': 26, 'seasonal naive': 1, 'training mean': 1} |

# Case 03: THU-INSC-NAD/Traffic-Matrix-Prediction

## abilene: 132 OD pairs, 0 constant pairs excluded

### Pooled over OD pairs: LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 168432
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| LSTM  <- model | 2866.3561 | 11262.9342 | 0.8506 |
| persistence  <- best baseline | 2228.5651 | 8437.0282 | 0.9162 |
| moving average (3) | 2387.7006 | 8957.6744 | 0.9055 |
| training mean | 16755.9944 | 29139.2816 | -0.0000 |
| seasonal naive (m=288) | 5441.2594 | 26685.1802 | 0.1613 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.958, persistence mae 2228.565). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: mae reduction vs persistence = -637.7910 (95% block-bootstrap CI -1339.9665 to -184.3784). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Pooled over OD pairs: Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 168432
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 2230.6238 | 8492.2334 | 0.9151 |
| persistence  <- best baseline | 2228.5651 | 8437.0282 | 0.9162 |
| moving average (3) | 2387.7006 | 8957.6744 | 0.9055 |
| training mean | 16755.9944 | 29139.2816 | -0.0000 |
| seasonal naive (m=288) | 5441.2594 | 26685.1802 | 0.1613 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.958, persistence mae 2228.565). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: mae reduction vs persistence = -2.0587 (95% block-bootstrap CI -61.8544 to +39.6360). |
| temporal split | PASS | Every test row comes after the training data in time. |

