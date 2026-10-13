# Summary

|   od | model         |      rmse | best_baseline          |   baseline_rmse | status   |   season |   lag1_autocorr |
|-----:|:--------------|----------:|:-----------------------|----------------:|:---------|---------:|----------------:|
|   52 | Vanilla LSTM  | 0.0943723 | seasonal naive (m=339) |       0.0350369 | FAIL     |      339 |       0.0641884 |
|   52 | BiLSTM        | 0.0854281 | seasonal naive (m=339) |       0.0350369 | FAIL     |      339 |       0.0641884 |
|   52 | GRU           | 0.0862735 | seasonal naive (m=339) |       0.0350369 | FAIL     |      339 |       0.0641884 |
|   52 | Linear AR(10) | 0.11632   | seasonal naive (m=339) |       0.0350369 | FAIL     |      339 |       0.0641884 |
|    4 | Vanilla LSTM  | 0.0884349 | seasonal naive (m=134) |       0.0164542 | FAIL     |      134 |       0.403502  |
|    4 | BiLSTM        | 0.0866058 | seasonal naive (m=134) |       0.0164542 | FAIL     |      134 |       0.403502  |
|    4 | GRU           | 0.0868317 | seasonal naive (m=134) |       0.0164542 | FAIL     |      134 |       0.403502  |
|    4 | Linear AR(10) | 0.0924341 | seasonal naive (m=134) |       0.0164542 | FAIL     |      134 |       0.403502  |
|    9 | Vanilla LSTM  | 0.0601672 | seasonal naive (m=128) |       0.0382331 | FAIL     |      128 |       0.724043  |
|    9 | BiLSTM        | 0.0490643 | seasonal naive (m=128) |       0.0382331 | FAIL     |      128 |       0.724043  |
|    9 | GRU           | 0.0535555 | seasonal naive (m=128) |       0.0382331 | FAIL     |      128 |       0.724043  |
|    9 | Linear AR(10) | 0.143339  | seasonal naive (m=128) |       0.0382331 | FAIL     |      128 |       0.724043  |
|   15 | Vanilla LSTM  | 0.0258209 | training mean          |       0.0256874 | FAIL     |      nan |      -0.0104028 |
|   15 | BiLSTM        | 0.025694  | training mean          |       0.0256874 | FAIL     |      nan |      -0.0104028 |
|   15 | GRU           | 0.0257858 | training mean          |       0.0256874 | FAIL     |      nan |      -0.0104028 |
|   15 | Linear AR(10) | 0.0256642 | training mean          |       0.0256874 | WARN     |      nan |      -0.0104028 |
|   34 | Vanilla LSTM  | 0.0701662 | seasonal naive (m=250) |       0.0531835 | FAIL     |      250 |       0.394282  |
|   34 | BiLSTM        | 0.0579518 | seasonal naive (m=250) |       0.0531835 | FAIL     |      250 |       0.394282  |
|   34 | GRU           | 0.0602692 | seasonal naive (m=250) |       0.0531835 | FAIL     |      250 |       0.394282  |
|   34 | Linear AR(10) | 0.125758  | seasonal naive (m=250) |       0.0531835 | FAIL     |      250 |       0.394282  |
|   53 | Vanilla LSTM  | 0.0732411 | seasonal naive (m=99)  |       0.0308061 | FAIL     |       99 |       0.306237  |
|   53 | BiLSTM        | 0.0668669 | seasonal naive (m=99)  |       0.0308061 | FAIL     |       99 |       0.306237  |
|   53 | GRU           | 0.0670756 | seasonal naive (m=99)  |       0.0308061 | FAIL     |       99 |       0.306237  |
|   53 | Linear AR(10) | 0.0783369 | seasonal naive (m=99)  |       0.0308061 | FAIL     |       99 |       0.306237  |
|   59 | Vanilla LSTM  | 0.0464327 | seasonal naive (m=210) |       0.0304305 | FAIL     |      210 |       0.534274  |
|   59 | BiLSTM        | 0.0439838 | seasonal naive (m=210) |       0.0304305 | FAIL     |      210 |       0.534274  |
|   59 | GRU           | 0.0436076 | seasonal naive (m=210) |       0.0304305 | FAIL     |      210 |       0.534274  |
|   59 | Linear AR(10) | 0.049093  | seasonal naive (m=210) |       0.0304305 | FAIL     |      210 |       0.534274  |
|   96 | Vanilla LSTM  | 0.0480129 | seasonal naive (m=29)  |       0.0536892 | PASS     |       29 |      -0.468076  |
|   96 | BiLSTM        | 0.0396167 | seasonal naive (m=29)  |       0.0536892 | PASS     |       29 |      -0.468076  |
|   96 | GRU           | 0.0396146 | seasonal naive (m=29)  |       0.0536892 | PASS     |       29 |      -0.468076  |
|   96 | Linear AR(10) | 0.0794211 | seasonal naive (m=29)  |       0.0536892 | FAIL     |       29 |      -0.468076  |
|  118 | Vanilla LSTM  | 0.061013  | seasonal naive (m=133) |       0.0390483 | FAIL     |      133 |       0.690648  |
|  118 | BiLSTM        | 0.0549308 | seasonal naive (m=133) |       0.0390483 | FAIL     |      133 |       0.690648  |
|  118 | GRU           | 0.0580262 | seasonal naive (m=133) |       0.0390483 | FAIL     |      133 |       0.690648  |
|  118 | Linear AR(10) | 0.0920856 | seasonal naive (m=133) |       0.0390483 | FAIL     |      133 |       0.690648  |
|  127 | Vanilla LSTM  | 0.0465412 | seasonal naive (m=21)  |       0.0419801 | FAIL     |       21 |       0.480119  |
|  127 | BiLSTM        | 0.0418309 | seasonal naive (m=21)  |       0.0419801 | WARN     |       21 |       0.480119  |
|  127 | GRU           | 0.0429593 | seasonal naive (m=21)  |       0.0419801 | FAIL     |       21 |       0.480119  |
|  127 | Linear AR(10) | 0.141731  | seasonal naive (m=21)  |       0.0419801 | FAIL     |       21 |       0.480119  |
|  157 | Vanilla LSTM  | 0.0370766 | seasonal naive (m=100) |       0.0295312 | FAIL     |      100 |       0.430927  |
|  157 | BiLSTM        | 0.0341665 | seasonal naive (m=100) |       0.0295312 | FAIL     |      100 |       0.430927  |
|  157 | GRU           | 0.035386  | seasonal naive (m=100) |       0.0295312 | FAIL     |      100 |       0.430927  |
|  157 | Linear AR(10) | 0.0387108 | seasonal naive (m=100) |       0.0295312 | FAIL     |      100 |       0.430927  |
|  158 | Vanilla LSTM  | 0.0373489 | seasonal naive (m=17)  |       0.0433962 | PASS     |       17 |       0.43513   |
|  158 | BiLSTM        | 0.0353839 | seasonal naive (m=17)  |       0.0433962 | PASS     |       17 |       0.43513   |
|  158 | GRU           | 0.0365321 | seasonal naive (m=17)  |       0.0433962 | PASS     |       17 |       0.43513   |
|  158 | Linear AR(10) | 0.113929  | seasonal naive (m=17)  |       0.0433962 | FAIL     |       17 |       0.43513   |
|  179 | Vanilla LSTM  | 0.0378934 | seasonal naive (m=70)  |       0.0450156 | PASS     |       70 |      -0.389871  |
|  179 | BiLSTM        | 0.034369  | seasonal naive (m=70)  |       0.0450156 | PASS     |       70 |      -0.389871  |
|  179 | GRU           | 0.0367706 | seasonal naive (m=70)  |       0.0450156 | PASS     |       70 |      -0.389871  |
|  179 | Linear AR(10) | 0.0799032 | seasonal naive (m=70)  |       0.0450156 | FAIL     |       70 |      -0.389871  |

# Case 04: SDN-TMprediction (ICC 2021), testbed dataset

## OD column 52

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0944, BiLSTM 0.0854, GRU 0.0863, Linear AR(10) 0.1163

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0088/0.0087, BiLSTM 0.0071/0.0071, GRU 0.0074/0.0071

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 339 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0668 | 0.0944 | 0.4082 |
| persistence | 0.0990 | 0.1679 | -0.8723 |
| moving average (3) | 0.0880 | 0.1374 | -0.2544 |
| training mean | 0.0819 | 0.1227 | -0.0001 |
| seasonal naive (m=339)  <- best baseline | 0.0268 | 0.0350 | 0.9184 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.064; persistence scores rmse 0.168. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=339) = -0.0593 (95% block-bootstrap CI -0.0642 to -0.0546). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 339 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0595 | 0.0854 | 0.5150 |
| persistence | 0.0990 | 0.1679 | -0.8723 |
| moving average (3) | 0.0880 | 0.1374 | -0.2544 |
| training mean | 0.0819 | 0.1227 | -0.0001 |
| seasonal naive (m=339)  <- best baseline | 0.0268 | 0.0350 | 0.9184 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.064; persistence scores rmse 0.168. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=339) = -0.0504 (95% block-bootstrap CI -0.0549 to -0.0463). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 339 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0599 | 0.0863 | 0.5054 |
| persistence | 0.0990 | 0.1679 | -0.8723 |
| moving average (3) | 0.0880 | 0.1374 | -0.2544 |
| training mean | 0.0819 | 0.1227 | -0.0001 |
| seasonal naive (m=339)  <- best baseline | 0.0268 | 0.0350 | 0.9184 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.064; persistence scores rmse 0.168. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=339) = -0.0512 (95% block-bootstrap CI -0.0552 to -0.0473). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 339 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0757 | 0.1163 | 0.1009 |
| persistence | 0.0990 | 0.1679 | -0.8723 |
| moving average (3) | 0.0880 | 0.1374 | -0.2544 |
| training mean | 0.0819 | 0.1227 | -0.0001 |
| seasonal naive (m=339)  <- best baseline | 0.0268 | 0.0350 | 0.9184 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.064; persistence scores rmse 0.168. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=339) = -0.0813 (95% block-bootstrap CI -0.0883 to -0.0755). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 4

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0884, BiLSTM 0.0866, GRU 0.0868, Linear AR(10) 0.0924

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0033/0.0071, BiLSTM 0.0026/0.0064, GRU 0.0030/0.0066

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 134 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0614 | 0.0884 | 0.2582 |
| persistence | 0.0647 | 0.1121 | -0.1929 |
| moving average (3) | 0.0656 | 0.1079 | -0.1040 |
| training mean | 0.0731 | 0.1027 | -0.0005 |
| seasonal naive (m=134)  <- best baseline | 0.0106 | 0.0165 | 0.9743 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.404; persistence scores rmse 0.112. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=134) = -0.0720 (95% block-bootstrap CI -0.0767 to -0.0678). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 134 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0573 | 0.0866 | 0.2885 |
| persistence | 0.0647 | 0.1121 | -0.1929 |
| moving average (3) | 0.0656 | 0.1079 | -0.1040 |
| training mean | 0.0731 | 0.1027 | -0.0005 |
| seasonal naive (m=134)  <- best baseline | 0.0106 | 0.0165 | 0.9743 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.404; persistence scores rmse 0.112. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=134) = -0.0702 (95% block-bootstrap CI -0.0757 to -0.0654). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 134 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0578 | 0.0868 | 0.2848 |
| persistence | 0.0647 | 0.1121 | -0.1929 |
| moving average (3) | 0.0656 | 0.1079 | -0.1040 |
| training mean | 0.0731 | 0.1027 | -0.0005 |
| seasonal naive (m=134)  <- best baseline | 0.0106 | 0.0165 | 0.9743 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.404; persistence scores rmse 0.112. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=134) = -0.0704 (95% block-bootstrap CI -0.0755 to -0.0665). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 134 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0603 | 0.0924 | 0.1895 |
| persistence | 0.0647 | 0.1121 | -0.1929 |
| moving average (3) | 0.0656 | 0.1079 | -0.1040 |
| training mean | 0.0731 | 0.1027 | -0.0005 |
| seasonal naive (m=134)  <- best baseline | 0.0106 | 0.0165 | 0.9743 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.404; persistence scores rmse 0.112. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=134) = -0.0760 (95% block-bootstrap CI -0.0830 to -0.0701). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 9

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0602, BiLSTM 0.0491, GRU 0.0536, Linear AR(10) 0.1433

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0032/0.0035, BiLSTM 0.0021/0.0024, GRU 0.0025/0.0028

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 128 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0443 | 0.0602 | 0.9268 |
| persistence | 0.1062 | 0.1652 | 0.4486 |
| moving average (3) | 0.1222 | 0.1764 | 0.3710 |
| training mean | 0.1961 | 0.2225 | -0.0000 |
| seasonal naive (m=128)  <- best baseline | 0.0240 | 0.0382 | 0.9705 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.724; persistence scores rmse 0.165. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=128) = -0.0219 (95% block-bootstrap CI -0.0257 to -0.0176). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 128 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0355 | 0.0491 | 0.9514 |
| persistence | 0.1062 | 0.1652 | 0.4486 |
| moving average (3) | 0.1222 | 0.1764 | 0.3710 |
| training mean | 0.1961 | 0.2225 | -0.0000 |
| seasonal naive (m=128)  <- best baseline | 0.0240 | 0.0382 | 0.9705 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.724; persistence scores rmse 0.165. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=128) = -0.0108 (95% block-bootstrap CI -0.0150 to -0.0065). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 128 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0392 | 0.0536 | 0.9420 |
| persistence | 0.1062 | 0.1652 | 0.4486 |
| moving average (3) | 0.1222 | 0.1764 | 0.3710 |
| training mean | 0.1961 | 0.2225 | -0.0000 |
| seasonal naive (m=128)  <- best baseline | 0.0240 | 0.0382 | 0.9705 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.724; persistence scores rmse 0.165. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=128) = -0.0153 (95% block-bootstrap CI -0.0194 to -0.0109). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 128 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.1034 | 0.1433 | 0.5848 |
| persistence | 0.1062 | 0.1652 | 0.4486 |
| moving average (3) | 0.1222 | 0.1764 | 0.3710 |
| training mean | 0.1961 | 0.2225 | -0.0000 |
| seasonal naive (m=128)  <- best baseline | 0.0240 | 0.0382 | 0.9705 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.724; persistence scores rmse 0.165. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=128) = -0.1051 (95% block-bootstrap CI -0.1107 to -0.0993). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 15

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0258, BiLSTM 0.0257, GRU 0.0258, Linear AR(10) 0.0257

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0006/0.0007, BiLSTM 0.0006/0.0007, GRU 0.0006/0.0007

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0175 | 0.0258 | -0.0153 |
| persistence | 0.0250 | 0.0364 | -1.0206 |
| moving average (3) | 0.0213 | 0.0297 | -0.3420 |
| training mean  <- best baseline | 0.0162 | 0.0257 | -0.0048 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.010; persistence scores rmse 0.036. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs training mean = -0.0001 (95% block-bootstrap CI -0.0003 to +0.0001). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0171 | 0.0257 | -0.0053 |
| persistence | 0.0250 | 0.0364 | -1.0206 |
| moving average (3) | 0.0213 | 0.0297 | -0.3420 |
| training mean  <- best baseline | 0.0162 | 0.0257 | -0.0048 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.010; persistence scores rmse 0.036. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs training mean = -0.0000 (95% block-bootstrap CI -0.0001 to +0.0002). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0174 | 0.0258 | -0.0125 |
| persistence | 0.0250 | 0.0364 | -1.0206 |
| moving average (3) | 0.0213 | 0.0297 | -0.3420 |
| training mean  <- best baseline | 0.0162 | 0.0257 | -0.0048 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.010; persistence scores rmse 0.036. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs training mean = -0.0001 (95% block-bootstrap CI -0.0002 to +0.0001). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: WARN**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0164 | 0.0257 | -0.0030 |
| persistence | 0.0250 | 0.0364 | -1.0206 |
| moving average (3) | 0.0213 | 0.0297 | -0.3420 |
| training mean  <- best baseline | 0.0162 | 0.0257 | -0.0048 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.010; persistence scores rmse 0.036. |
| beats trivial baselines | WARN | Ahead of the best trivial baseline, but the margin is within noise: rmse reduction vs training mean = +0.0000 (95% block-bootstrap CI -0.0000 to +0.0001). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 34

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0702, BiLSTM 0.0580, GRU 0.0603, Linear AR(10) 0.1258

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0050/0.0052, BiLSTM 0.0032/0.0035, GRU 0.0036/0.0037

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 250 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0505 | 0.0702 | 0.7568 |
| persistence | 0.0992 | 0.1566 | -0.2111 |
| moving average (3) | 0.1008 | 0.1520 | -0.1419 |
| training mean | 0.1089 | 0.1423 | -0.0000 |
| seasonal naive (m=250)  <- best baseline | 0.0337 | 0.0532 | 0.8603 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.394; persistence scores rmse 0.157. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=250) = -0.0170 (95% block-bootstrap CI -0.0211 to -0.0118). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 250 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0402 | 0.0580 | 0.8341 |
| persistence | 0.0992 | 0.1566 | -0.2111 |
| moving average (3) | 0.1008 | 0.1520 | -0.1419 |
| training mean | 0.1089 | 0.1423 | -0.0000 |
| seasonal naive (m=250)  <- best baseline | 0.0337 | 0.0532 | 0.8603 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.394; persistence scores rmse 0.157. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=250) = -0.0048 (95% block-bootstrap CI -0.0094 to +0.0007). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 250 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0422 | 0.0603 | 0.8205 |
| persistence | 0.0992 | 0.1566 | -0.2111 |
| moving average (3) | 0.1008 | 0.1520 | -0.1419 |
| training mean | 0.1089 | 0.1423 | -0.0000 |
| seasonal naive (m=250)  <- best baseline | 0.0337 | 0.0532 | 0.8603 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.394; persistence scores rmse 0.157. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=250) = -0.0071 (95% block-bootstrap CI -0.0115 to -0.0019). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 250 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0916 | 0.1258 | 0.2186 |
| persistence | 0.0992 | 0.1566 | -0.2111 |
| moving average (3) | 0.1008 | 0.1520 | -0.1419 |
| training mean | 0.1089 | 0.1423 | -0.0000 |
| seasonal naive (m=250)  <- best baseline | 0.0337 | 0.0532 | 0.8603 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.394; persistence scores rmse 0.157. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=250) = -0.0726 (95% block-bootstrap CI -0.0777 to -0.0660). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 53

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0732, BiLSTM 0.0669, GRU 0.0671, Linear AR(10) 0.0783

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0058/0.0061, BiLSTM 0.0047/0.0052, GRU 0.0049/0.0053

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 99 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0473 | 0.0732 | 0.2423 |
| persistence | 0.0599 | 0.0991 | -0.3866 |
| moving average (3) | 0.0607 | 0.0978 | -0.3516 |
| training mean | 0.0573 | 0.0843 | -0.0035 |
| seasonal naive (m=99)  <- best baseline | 0.0218 | 0.0308 | 0.8659 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.306; persistence scores rmse 0.099. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=99) = -0.0424 (95% block-bootstrap CI -0.0459 to -0.0374). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 99 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0428 | 0.0669 | 0.3684 |
| persistence | 0.0599 | 0.0991 | -0.3866 |
| moving average (3) | 0.0607 | 0.0978 | -0.3516 |
| training mean | 0.0573 | 0.0843 | -0.0035 |
| seasonal naive (m=99)  <- best baseline | 0.0218 | 0.0308 | 0.8659 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.306; persistence scores rmse 0.099. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=99) = -0.0361 (95% block-bootstrap CI -0.0386 to -0.0325). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 99 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0424 | 0.0671 | 0.3645 |
| persistence | 0.0599 | 0.0991 | -0.3866 |
| moving average (3) | 0.0607 | 0.0978 | -0.3516 |
| training mean | 0.0573 | 0.0843 | -0.0035 |
| seasonal naive (m=99)  <- best baseline | 0.0218 | 0.0308 | 0.8659 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.306; persistence scores rmse 0.099. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=99) = -0.0363 (95% block-bootstrap CI -0.0386 to -0.0327). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 99 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0546 | 0.0783 | 0.1331 |
| persistence | 0.0599 | 0.0991 | -0.3866 |
| moving average (3) | 0.0607 | 0.0978 | -0.3516 |
| training mean | 0.0573 | 0.0843 | -0.0035 |
| seasonal naive (m=99)  <- best baseline | 0.0218 | 0.0308 | 0.8659 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.306; persistence scores rmse 0.099. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=99) = -0.0475 (95% block-bootstrap CI -0.0521 to -0.0414). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 59

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0464, BiLSTM 0.0440, GRU 0.0436, Linear AR(10) 0.0491

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0023/0.0022, BiLSTM 0.0021/0.0020, GRU 0.0020/0.0019

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 210 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0329 | 0.0464 | 0.4920 |
| persistence | 0.0447 | 0.0629 | 0.0685 |
| moving average (3) | 0.0491 | 0.0688 | -0.1148 |
| training mean | 0.0498 | 0.0651 | -0.0001 |
| seasonal naive (m=210)  <- best baseline | 0.0214 | 0.0304 | 0.7818 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.534; persistence scores rmse 0.063. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=210) = -0.0160 (95% block-bootstrap CI -0.0180 to -0.0132). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 210 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0314 | 0.0440 | 0.5441 |
| persistence | 0.0447 | 0.0629 | 0.0685 |
| moving average (3) | 0.0491 | 0.0688 | -0.1148 |
| training mean | 0.0498 | 0.0651 | -0.0001 |
| seasonal naive (m=210)  <- best baseline | 0.0214 | 0.0304 | 0.7818 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.534; persistence scores rmse 0.063. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=210) = -0.0136 (95% block-bootstrap CI -0.0156 to -0.0108). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 210 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0313 | 0.0436 | 0.5519 |
| persistence | 0.0447 | 0.0629 | 0.0685 |
| moving average (3) | 0.0491 | 0.0688 | -0.1148 |
| training mean | 0.0498 | 0.0651 | -0.0001 |
| seasonal naive (m=210)  <- best baseline | 0.0214 | 0.0304 | 0.7818 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.534; persistence scores rmse 0.063. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=210) = -0.0132 (95% block-bootstrap CI -0.0154 to -0.0106). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 210 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0349 | 0.0491 | 0.4321 |
| persistence | 0.0447 | 0.0629 | 0.0685 |
| moving average (3) | 0.0491 | 0.0688 | -0.1148 |
| training mean | 0.0498 | 0.0651 | -0.0001 |
| seasonal naive (m=210)  <- best baseline | 0.0214 | 0.0304 | 0.7818 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.534; persistence scores rmse 0.063. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=210) = -0.0187 (95% block-bootstrap CI -0.0207 to -0.0160). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 96

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0480, BiLSTM 0.0396, GRU 0.0396, Linear AR(10) 0.0794

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0011/0.0022, BiLSTM 0.0012/0.0014, GRU 0.0013/0.0015

### Vanilla LSTM

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 29 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0397 | 0.0480 | 0.9588 |
| persistence | 0.3318 | 0.4057 | -1.9411 |
| moving average (3) | 0.1987 | 0.2455 | -0.0772 |
| training mean | 0.1812 | 0.2381 | -0.0132 |
| seasonal naive (m=29)  <- best baseline | 0.0402 | 0.0537 | 0.9485 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.468; persistence scores rmse 0.406. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=29) = +0.0057 (95% block-bootstrap CI +0.0034 to +0.0087). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 29 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0317 | 0.0396 | 0.9720 |
| persistence | 0.3318 | 0.4057 | -1.9411 |
| moving average (3) | 0.1987 | 0.2455 | -0.0772 |
| training mean | 0.1812 | 0.2381 | -0.0132 |
| seasonal naive (m=29)  <- best baseline | 0.0402 | 0.0537 | 0.9485 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.468; persistence scores rmse 0.406. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=29) = +0.0141 (95% block-bootstrap CI +0.0118 to +0.0169). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 29 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0314 | 0.0396 | 0.9720 |
| persistence | 0.3318 | 0.4057 | -1.9411 |
| moving average (3) | 0.1987 | 0.2455 | -0.0772 |
| training mean | 0.1812 | 0.2381 | -0.0132 |
| seasonal naive (m=29)  <- best baseline | 0.0402 | 0.0537 | 0.9485 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.468; persistence scores rmse 0.406. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=29) = +0.0141 (95% block-bootstrap CI +0.0118 to +0.0169). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 29 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0631 | 0.0794 | 0.8873 |
| persistence | 0.3318 | 0.4057 | -1.9411 |
| moving average (3) | 0.1987 | 0.2455 | -0.0772 |
| training mean | 0.1812 | 0.2381 | -0.0132 |
| seasonal naive (m=29)  <- best baseline | 0.0402 | 0.0537 | 0.9485 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.468; persistence scores rmse 0.406. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=29) = -0.0257 (95% block-bootstrap CI -0.0286 to -0.0225). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 118

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0610, BiLSTM 0.0549, GRU 0.0580, Linear AR(10) 0.0921

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0040/0.0041, BiLSTM 0.0032/0.0034, GRU 0.0035/0.0037

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 133 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0446 | 0.0610 | 0.8041 |
| persistence | 0.0739 | 0.1086 | 0.3796 |
| moving average (3) | 0.0737 | 0.1032 | 0.4390 |
| training mean | 0.1219 | 0.1379 | -0.0000 |
| seasonal naive (m=133)  <- best baseline | 0.0288 | 0.0390 | 0.9198 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.691; persistence scores rmse 0.109. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=133) = -0.0220 (95% block-bootstrap CI -0.0259 to -0.0188). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 133 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0395 | 0.0549 | 0.8412 |
| persistence | 0.0739 | 0.1086 | 0.3796 |
| moving average (3) | 0.0737 | 0.1032 | 0.4390 |
| training mean | 0.1219 | 0.1379 | -0.0000 |
| seasonal naive (m=133)  <- best baseline | 0.0288 | 0.0390 | 0.9198 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.691; persistence scores rmse 0.109. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=133) = -0.0159 (95% block-bootstrap CI -0.0196 to -0.0135). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 133 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0412 | 0.0580 | 0.8228 |
| persistence | 0.0739 | 0.1086 | 0.3796 |
| moving average (3) | 0.0737 | 0.1032 | 0.4390 |
| training mean | 0.1219 | 0.1379 | -0.0000 |
| seasonal naive (m=133)  <- best baseline | 0.0288 | 0.0390 | 0.9198 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.691; persistence scores rmse 0.109. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=133) = -0.0190 (95% block-bootstrap CI -0.0227 to -0.0160). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 133 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0702 | 0.0921 | 0.5538 |
| persistence | 0.0739 | 0.1086 | 0.3796 |
| moving average (3) | 0.0737 | 0.1032 | 0.4390 |
| training mean | 0.1219 | 0.1379 | -0.0000 |
| seasonal naive (m=133)  <- best baseline | 0.0288 | 0.0390 | 0.9198 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.691; persistence scores rmse 0.109. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=133) = -0.0530 (95% block-bootstrap CI -0.0573 to -0.0490). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 127

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0465, BiLSTM 0.0418, GRU 0.0430, Linear AR(10) 0.1417

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0021/0.0024, BiLSTM 0.0017/0.0020, GRU 0.0018/0.0021

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 21 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0332 | 0.0465 | 0.9669 |
| persistence | 0.1648 | 0.2609 | -0.0397 |
| moving average (3) | 0.2595 | 0.3057 | -0.4270 |
| training mean | 0.2236 | 0.2559 | -0.0000 |
| seasonal naive (m=21)  <- best baseline | 0.0256 | 0.0420 | 0.9731 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.480; persistence scores rmse 0.261. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=21) = -0.0046 (95% block-bootstrap CI -0.0084 to -0.0013). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: WARN**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 21 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0289 | 0.0418 | 0.9733 |
| persistence | 0.1648 | 0.2609 | -0.0397 |
| moving average (3) | 0.2595 | 0.3057 | -0.4270 |
| training mean | 0.2236 | 0.2559 | -0.0000 |
| seasonal naive (m=21)  <- best baseline | 0.0256 | 0.0420 | 0.9731 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.480; persistence scores rmse 0.261. |
| beats trivial baselines | WARN | Ahead of the best trivial baseline, but the margin is within noise: rmse reduction vs seasonal naive (m=21) = +0.0001 (95% block-bootstrap CI -0.0035 to +0.0032). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 21 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0311 | 0.0430 | 0.9718 |
| persistence | 0.1648 | 0.2609 | -0.0397 |
| moving average (3) | 0.2595 | 0.3057 | -0.4270 |
| training mean | 0.2236 | 0.2559 | -0.0000 |
| seasonal naive (m=21)  <- best baseline | 0.0256 | 0.0420 | 0.9731 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.480; persistence scores rmse 0.261. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=21) = -0.0010 (95% block-bootstrap CI -0.0047 to +0.0021). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 21 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.1052 | 0.1417 | 0.6932 |
| persistence | 0.1648 | 0.2609 | -0.0397 |
| moving average (3) | 0.2595 | 0.3057 | -0.4270 |
| training mean | 0.2236 | 0.2559 | -0.0000 |
| seasonal naive (m=21)  <- best baseline | 0.0256 | 0.0420 | 0.9731 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.480; persistence scores rmse 0.261. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=21) = -0.0998 (95% block-bootstrap CI -0.1053 to -0.0943). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 157

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0371, BiLSTM 0.0342, GRU 0.0354, Linear AR(10) 0.0387

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0014/0.0014, BiLSTM 0.0012/0.0012, GRU 0.0013/0.0013

### Vanilla LSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 100 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0282 | 0.0371 | 0.5724 |
| persistence | 0.0434 | 0.0605 | -0.1384 |
| moving average (3) | 0.0505 | 0.0648 | -0.3079 |
| training mean | 0.0445 | 0.0567 | -0.0000 |
| seasonal naive (m=100)  <- best baseline | 0.0199 | 0.0295 | 0.7287 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.431; persistence scores rmse 0.060. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=100) = -0.0075 (95% block-bootstrap CI -0.0095 to -0.0054). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 100 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0257 | 0.0342 | 0.6369 |
| persistence | 0.0434 | 0.0605 | -0.1384 |
| moving average (3) | 0.0505 | 0.0648 | -0.3079 |
| training mean | 0.0445 | 0.0567 | -0.0000 |
| seasonal naive (m=100)  <- best baseline | 0.0199 | 0.0295 | 0.7287 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.431; persistence scores rmse 0.060. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=100) = -0.0046 (95% block-bootstrap CI -0.0065 to -0.0027). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 100 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0263 | 0.0354 | 0.6105 |
| persistence | 0.0434 | 0.0605 | -0.1384 |
| moving average (3) | 0.0505 | 0.0648 | -0.3079 |
| training mean | 0.0445 | 0.0567 | -0.0000 |
| seasonal naive (m=100)  <- best baseline | 0.0199 | 0.0295 | 0.7287 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.431; persistence scores rmse 0.060. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=100) = -0.0059 (95% block-bootstrap CI -0.0078 to -0.0039). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 100 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0292 | 0.0387 | 0.5338 |
| persistence | 0.0434 | 0.0605 | -0.1384 |
| moving average (3) | 0.0505 | 0.0648 | -0.3079 |
| training mean | 0.0445 | 0.0567 | -0.0000 |
| seasonal naive (m=100)  <- best baseline | 0.0199 | 0.0295 | 0.7287 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.431; persistence scores rmse 0.060. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=100) = -0.0092 (95% block-bootstrap CI -0.0111 to -0.0073). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 158

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0373, BiLSTM 0.0354, GRU 0.0365, Linear AR(10) 0.1139

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0014/0.0012, BiLSTM 0.0012/0.0012, GRU 0.0013/0.0012

### Vanilla LSTM

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 17 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0207 | 0.0373 | 0.9810 |
| persistence | 0.1637 | 0.2881 | -0.1293 |
| moving average (3) | 0.2637 | 0.3418 | -0.5894 |
| training mean | 0.2407 | 0.2711 | -0.0000 |
| seasonal naive (m=17)  <- best baseline | 0.0233 | 0.0434 | 0.9744 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.435; persistence scores rmse 0.288. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=17) = +0.0060 (95% block-bootstrap CI +0.0016 to +0.0111). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 17 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0195 | 0.0354 | 0.9830 |
| persistence | 0.1637 | 0.2881 | -0.1293 |
| moving average (3) | 0.2637 | 0.3418 | -0.5894 |
| training mean | 0.2407 | 0.2711 | -0.0000 |
| seasonal naive (m=17)  <- best baseline | 0.0233 | 0.0434 | 0.9744 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.435; persistence scores rmse 0.288. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=17) = +0.0080 (95% block-bootstrap CI +0.0036 to +0.0132). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 17 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0203 | 0.0365 | 0.9818 |
| persistence | 0.1637 | 0.2881 | -0.1293 |
| moving average (3) | 0.2637 | 0.3418 | -0.5894 |
| training mean | 0.2407 | 0.2711 | -0.0000 |
| seasonal naive (m=17)  <- best baseline | 0.0233 | 0.0434 | 0.9744 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.435; persistence scores rmse 0.288. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=17) = +0.0069 (95% block-bootstrap CI +0.0026 to +0.0119). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 17 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0849 | 0.1139 | 0.8234 |
| persistence | 0.1637 | 0.2881 | -0.1293 |
| moving average (3) | 0.2637 | 0.3418 | -0.5894 |
| training mean | 0.2407 | 0.2711 | -0.0000 |
| seasonal naive (m=17)  <- best baseline | 0.0233 | 0.0434 | 0.9744 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is 0.435; persistence scores rmse 0.288. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=17) = -0.0705 (95% block-bootstrap CI -0.0788 to -0.0643). |
| temporal split | PASS | Every test row comes after the training data in time. |

## OD column 179

Scaled RMSE (the paper's metric): Vanilla LSTM 0.0379, BiLSTM 0.0344, GRU 0.0368, Linear AR(10) 0.0799

Final-epoch train / validation MSE (notebook, OD 52: LSTM 0.0081/0.0076, BiLSTM 0.0064/0.0061, GRU 0.0068/0.0064): Vanilla LSTM 0.0014/0.0014, BiLSTM 0.0012/0.0012, GRU 0.0013/0.0013

### Vanilla LSTM

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 70 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Vanilla LSTM  <- model | 0.0290 | 0.0379 | 0.9407 |
| persistence | 0.2222 | 0.2593 | -1.7797 |
| moving average (3) | 0.1664 | 0.1930 | -0.5396 |
| training mean | 0.1279 | 0.1555 | -0.0000 |
| seasonal naive (m=70)  <- best baseline | 0.0311 | 0.0450 | 0.9162 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.390; persistence scores rmse 0.259. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=70) = +0.0071 (95% block-bootstrap CI +0.0042 to +0.0100). |
| temporal split | PASS | Every test row comes after the training data in time. |

### BiLSTM

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 70 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| BiLSTM  <- model | 0.0255 | 0.0344 | 0.9512 |
| persistence | 0.2222 | 0.2593 | -1.7797 |
| moving average (3) | 0.1664 | 0.1930 | -0.5396 |
| training mean | 0.1279 | 0.1555 | -0.0000 |
| seasonal naive (m=70)  <- best baseline | 0.0311 | 0.0450 | 0.9162 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.390; persistence scores rmse 0.259. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=70) = +0.0106 (95% block-bootstrap CI +0.0077 to +0.0139). |
| temporal split | PASS | Every test row comes after the training data in time. |

### GRU

**Verdict: PASS**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 70 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| GRU  <- model | 0.0275 | 0.0368 | 0.9441 |
| persistence | 0.2222 | 0.2593 | -1.7797 |
| moving average (3) | 0.1664 | 0.1930 | -0.5396 |
| training mean | 0.1279 | 0.1555 | -0.0000 |
| seasonal naive (m=70)  <- best baseline | 0.0311 | 0.0450 | 0.9162 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.390; persistence scores rmse 0.259. |
| beats trivial baselines | PASS | Beats the best trivial baseline: rmse reduction vs seasonal naive (m=70) = +0.0082 (95% block-bootstrap CI +0.0054 to +0.0115). |
| temporal split | PASS | Every test row comes after the training data in time. |

### Linear AR(10)

**Verdict: FAIL**

- target: y   task: regression   horizon: 1   metric: rmse   rows scored: 1241
- seasonal period: 70 steps

| model | mae | rmse | r2 |
|---|---|---|---|
| Linear AR(10)  <- model | 0.0647 | 0.0799 | 0.7361 |
| persistence | 0.2222 | 0.2593 | -1.7797 |
| moving average (3) | 0.1664 | 0.1930 | -0.5396 |
| training mean | 0.1279 | 0.1555 | -0.0000 |
| seasonal naive (m=70)  <- best baseline | 0.0311 | 0.0450 | 0.9162 |

| check | status | finding |
|---|---|---|
| label memory | INFO | Lag-1 autocorrelation of the target is -0.390; persistence scores rmse 0.259. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: rmse reduction vs seasonal naive (m=70) = -0.0349 (95% block-bootstrap CI -0.0389 to -0.0296). |
| temporal split | PASS | Every test row comes after the training data in time. |

