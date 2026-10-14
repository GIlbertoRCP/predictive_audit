# Per-OD verdicts (beats trivial baselines?)

| network   | model         |   FAIL |   PASS |   WARN |
|:----------|:--------------|-------:|-------:|-------:|
| abilene   | LSTM          |     67 |     31 |     34 |
| abilene   | Linear AR(10) |     69 |     37 |     26 |
| cernet    | LSTM          |    109 |     29 |     17 |
| cernet    | Linear AR(10) |     28 |    106 |     21 |
| geant     | LSTM          |    350 |     44 |    104 |
| geant     | Linear AR(10) |    343 |     53 |    102 |

Which baseline was best, per OD:

| network   | model         | best_baseline                                                                       |
|:----------|:--------------|:------------------------------------------------------------------------------------|
| abilene   | LSTM          | {'persistence': 104, 'moving average': 26, 'seasonal naive': 1, 'training mean': 1} |
| abilene   | Linear AR(10) | {'persistence': 104, 'moving average': 26, 'seasonal naive': 1, 'training mean': 1} |
| cernet    | LSTM          | {'persistence': 154, 'training mean': 1}                                            |
| cernet    | Linear AR(10) | {'persistence': 154, 'training mean': 1}                                            |
| geant     | LSTM          | {'persistence': 414, 'moving average': 78, 'seasonal naive': 3, 'training mean': 3} |
| geant     | Linear AR(10) | {'persistence': 414, 'moving average': 78, 'seasonal naive': 3, 'training mean': 3} |

# Case 03: THU-INSC-NAD/Traffic-Matrix-Prediction

## abilene: 132 OD pairs, 0 constant pairs excluded

### Pooled over OD pairs: LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 168432
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| LSTM  <- model | 2646.2645 | 9863.5203 | 0.8854 |
| persistence  <- best baseline | 2228.5651 | 8437.0282 | 0.9162 |
| moving average (3) | 2387.7006 | 8957.6744 | 0.9055 |
| training mean | 16755.9944 | 29139.2816 | -0.0000 |
| seasonal naive (m=288) | 5441.2594 | 26685.1802 | 0.1613 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.958, persistence mae 2228.565). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: mae reduction vs persistence = -417.6994 (95% block-bootstrap CI -923.2931 to -92.0023). |
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

## geant: 498 OD pairs, 8 constant pairs excluded

### Pooled over OD pairs: LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 327684
- seasonal period: 96 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| LSTM  <- model | 1578.8599 | 6965.6736 | 0.9842 |
| persistence  <- best baseline | 1427.0792 | 6567.5391 | 0.9860 |
| moving average (3) | 1604.0451 | 7472.7300 | 0.9818 |
| training mean | 19681.2176 | 55424.6488 | -0.0004 |
| seasonal naive (m=96) | 4652.7807 | 25214.3166 | 0.7930 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.993, persistence mae 1427.079). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: mae reduction vs persistence = -151.7807 (95% block-bootstrap CI -207.0110 to -103.9719). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Pooled over OD pairs: Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 327684
- seasonal period: 96 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 1459.6406 | 6435.5122 | 0.9865 |
| persistence  <- best baseline | 1427.0792 | 6567.5391 | 0.9860 |
| moving average (3) | 1604.0451 | 7472.7300 | 0.9818 |
| training mean | 19681.2176 | 55424.6488 | -0.0004 |
| seasonal naive (m=96) | 4652.7807 | 25214.3166 | 0.7930 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.993, persistence mae 1427.079). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: mae reduction vs persistence = -32.5614 (95% block-bootstrap CI -62.8039 to -8.2959). |
| temporal split | PASS | Every test row comes after the training data in time. |

## cernet: 155 OD pairs, 27 constant pairs excluded

### Pooled over OD pairs: LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 191890
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| LSTM  <- model | 7696.6185 | 20158.4657 | 0.9934 |
| persistence  <- best baseline | 6345.5252 | 15824.6505 | 0.9959 |
| moving average (3) | 9310.8686 | 23102.1950 | 0.9913 |
| training mean | 141388.4314 | 248114.5072 | -0.0000 |
| seasonal naive (m=288) | 29599.6850 | 72116.2998 | 0.9155 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.998, persistence mae 6345.525). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: mae reduction vs persistence = -1351.0932 (95% block-bootstrap CI -1677.4219 to -1065.7652). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Pooled over OD pairs: Linear AR(10)

**Verdict: WARN**

- target: y   task: regression   horizon: 1   metric: mae   rows scored: 191890
- seasonal period: 288 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 5635.6012 | 13626.8975 | 0.9970 |
| persistence  <- best baseline | 6345.5252 | 15824.6505 | 0.9959 |
| moving average (3) | 9310.8686 | 23102.1950 | 0.9913 |
| training mean | 141388.4314 | 248114.5072 | -0.0000 |
| seasonal naive (m=288) | 29599.6850 | 72116.2998 | 0.9155 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.998, persistence mae 6345.525). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: mae reduction vs persistence = +709.9240 (95% block-bootstrap CI +607.5737 to +810.2093). |
| temporal split | PASS | Every test row comes after the training data in time. |

