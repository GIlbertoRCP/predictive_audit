# Case 01: Predictive_Activation v0 congestion forecaster

## A. Original protocol (shuffled 80/20 split)

**Verdict: FAIL**

- target: future_congestion   task: classification   horizon: 1   metric: f1   rows scored: 4934
- positive rate: 0.428

| model | f1 | pr_auc | mcc | balanced_accuracy | accuracy | roc_auc |
|---|---|---|---|---|---|---|
| GBM (LSTM stand-in)  <- model | 0.9991 | 1.0000 | 0.9983 | 0.9992 | 0.9992 | 1.0000 |
| persistence | 0.9931 | 0.9895 | 0.9880 | 0.9939 | 0.9941 | 0.9939 |
| moving average (3)  <- best baseline | 0.9945 | 0.9981 | 0.9905 | 0.9949 | 0.9953 | 0.9986 |
| majority class | 0.0000 | 0.4276 | 0.0000 | 0.5000 | 0.5724 | 0.5000 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.988, persistence f1 0.993). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | PASS | Beats the best trivial baseline: f1 gain over moving average (3) = +0.0045 (95% block-bootstrap CI +0.0021 to +0.0083). |
| temporal split | FAIL | Test rows are interleaved with training rows (100% precede the last training row, ~125 separate test segments per series). This is a shuffled split: with overlapping windows or autocorrelated rows, test examples are near-copies of training examples. |
| feature leakage | PASS | No feature is a near-copy of a future target value. |

## B. Chronological split (last 20% of each series)

**Verdict: FAIL**

- target: future_congestion   task: classification   horizon: 1   metric: f1   rows scored: 4960
- positive rate: 0.652

| model | f1 | pr_auc | mcc | balanced_accuracy | accuracy | roc_auc |
|---|---|---|---|---|---|---|
| GBM (LSTM stand-in)  <- model | 0.9835 | 0.9991 | 0.9533 | 0.9792 | 0.9786 | 0.9982 |
| persistence | 0.9806 | 0.9758 | 0.9446 | 0.9732 | 0.9748 | 0.9732 |
| moving average (3)  <- best baseline | 0.9868 | 0.9943 | 0.9625 | 0.9835 | 0.9829 | 0.9935 |
| majority class | 0.0000 | 0.6522 | 0.0000 | 0.5000 | 0.3478 | 0.5000 |

| check | status | finding |
|---|---|---|
| label memory | WARN | The target at t+1 is mostly a copy of the target at t (lag-1 autocorrelation 0.945, persistence f1 0.981). Any model will post a high score here; only the margin over persistence means anything. |
| beats trivial baselines | FAIL | Does not beat the best trivial baseline: f1 gain over moving average (3) = -0.0033 (95% block-bootstrap CI -0.0061 to -0.0011). |
| temporal split | PASS | Every test row comes after the training data in time. |
| feature leakage | PASS | No feature is a near-copy of a future target value. |
