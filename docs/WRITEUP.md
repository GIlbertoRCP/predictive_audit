# Did the model beat yesterday?

*Auditing published network-traffic forecasting results against trivial baselines.*
*Gilberto Romero-Cano · interactive version: [`site/`](../site/index.html)*

---

My team's course project reported a network-congestion forecaster with an F1 score around 0.99. Months later I checked
what the simplest possible forecaster scores on the same data, "the next step is congested if this step is": **F1 0.993**.
The model had learned that the label rarely changes. A shuffled train/test split hid it.

That made me wonder how common this is in published work. I built a small tool, **predictive-audit**, that scores any
forecasting result against the baselines it has to beat before it means anything, and against the evaluation
mistakes that make a result look better than it is. Then I ran it on five public results. Four don't clearly beat
trivial baselines.

## The bar

Network traffic is autocorrelated: the next five minutes look like the last five. Three forecasters need no learning:

- **Persistence:** the next value equals the current one.
- **Moving average:** the mean of the last few values.
- **Seasonal naive:** the value one cycle ago, such as the same minute yesterday or one replay loop ago. The cycle length
  is estimated from training data only.

A model's score only means something next to these. On 5-minute traffic with lag-1 autocorrelation 0.95, persistence
alone gets R² ≈ 0.9. A paper that reports R² = 0.94 and stops there hasn't told you much.

The tool also checks four other things:

- whether the margin over the best baseline survives a **moving-block bootstrap**, since an i.i.d. bootstrap is far too
  confident on autocorrelated data;
- whether the train/test split is **chronological**;
- whether any feature is a **near-copy of a future target**;
- whether a **published number** is already matched by a baseline.

## Results

| | Result | Verdict | Finding |
|---|---|---|---|
| 01 | My course project's congestion LSTM | **FAIL** | With a chronological split it loses to a 3-step moving average |
| 02 | xiaochus/TrafficFlowPrediction (681★, PeMS) | **holds up** | Beats baselines by 8–10% MAE; headline R² mostly autocorrelation |
| 03 | Traffic-Matrix-Prediction (ISCC 2019) | **FAIL** | LSTM 11–21% worse than persistence on Abilene, GÉANT and CERNET |
| 04 | SDN-TMprediction (IEEE ICC 2021) | **FAIL** | "One replay loop ago" wins on 9 of 13 flows; 2.4× better on the showcased one |
| 05 | Annongkhanh/Traffic-matrix-prediction | **FAIL** | Reported error is higher than persistence's on both datasets |

### The traffic was on a loop (case 04)

This IEEE ICC 2021 paper builds an SDN testbed in Mininet, generates traffic by replaying packet captures with
`tcpreplay`, and trains LSTM, BiLSTM and GRU models to predict each origin–destination flow from its last ten minutes. The
models are compared only with each other.

Replayed captures loop. In the training data, the flow the notebook showcases has a lag-1 autocorrelation of 0.07, which is
nearly noise, and an autocorrelation of **0.89 at 339 minutes**, the length of its replay cycle. A ten-minute input window
can't see that. A baseline that predicts "the value one loop ago", with the period estimated from training data, scores RMSE
0.035 against the BiLSTM's 0.085.

I re-implemented the notebook in PyTorch: Keras's ReLU LSTM cell, its initialisers and Adam defaults, and its quirks
(including `validation_split`, which never trains on the last 30% of the training windows). The final training and
validation losses land within about 10% of the losses the notebook logged, with the same ranking across models. Across the
showcased flow plus 12 random ones, the replay baseline beats the RNNs on 9 of 13. The general lesson for testbed papers:
replayed traffic turns a forecasting benchmark partly into a memorisation benchmark, and one seasonal baseline exposes it.

### 785 LSTMs on real backbones (case 03)

The Abilene, GÉANT and CERNET traffic matrices are the standard public benchmarks. This codebase (ISCC 2019) trains one LSTM
per origin–destination pair and feeds the forecasts into traffic engineering. I retrained all 785 non-trivial pairs with
the repository's settings. Since each pair's model is independent and Adagrad updates each weight on its own, all of a
network's models can be trained as one stacked model with identical results; I checked that to within 1e-7.

Pooled over each network, the LSTM's MAE is **19%, 11% and 21% worse than persistence**, and the bootstrap intervals exclude
zero. Per OD pair, it beats every trivial baseline on 104 of the 748 pairs that carry traffic during the test period (14%).
A plain linear model on the same ten inputs beats the LSTM on all three networks. The configuration the code's own comments
call "best" made the LSTM worse.

### A number worse than doing nothing (case 05)

Some audits need no retraining. This project prints one test MSE per dataset. Scoring persistence on the same test cells,
in the same scaled units, gives 0.0102 on GÉANT, against the published LSTM's **0.0260**. The 3.6M-parameter model was trained
on 189 windows.

### One that holds up (case 02)

An audit that fails everything would be worthless. xiaochus/TrafficFlowPrediction ships its trained weights, so I ran the
exact published models through a numpy re-implementation of the Keras 2.1 forward pass. They reproduce the README to two
decimals, and they **genuinely beat every baseline**, by 8–10% MAE with intervals clear of zero. The headline R² of 0.94 still
oversells it: persistence alone scores 0.92.

### My own model (case 01)

I re-ran the original pipeline both ways, with a gradient-boosted model standing in for the LSTM. With the original
shuffled split, the model edges out the baselines. With a chronological split, a three-step moving average of the label beats
it, F1 0.987 to 0.984, and the interval excludes zero. The tool flags the shuffled split directly: every test row sits
between training rows in time.

## What I'd ask of any forecasting result

1. Report **persistence and a seasonal-naive baseline** on the same rows, with the period chosen from training data.
2. Report the **margin**, with an interval that respects autocorrelation.
3. **Split by time.** Never shuffle overlapping windows before splitting.
4. Treat **R² and accuracy** on smooth or sticky series as near-meaningless on their own.
5. Include **one simple learned model** (a linear AR on the same inputs) before reaching for an LSTM.

## How the audits were done, and their limits

- Third-party data and weights are downloaded at a pinned commit and checked against SHA-256. Nothing is redistributed
  beyond the small excerpts plotted on the explorer page.
- Case 02 uses the published weights. Cases 03 and 04 are faithful re-implementations, run with a single seed; the authors'
  weights weren't available. Case 05 compares printed numbers.
- The papers' own result tables for cases 03 and 04 aren't in their repositories and weren't openly available, so those
  audits check the released code's protocol, not the printed tables.
- Each case's README lists its caveats in full, including the ones that cut against the finding.

## Try it

```bash
uv sync
uv run predictive-audit predictions.csv --target y_next --pred model --group series \
    --time t --split split --season auto --metric mae --reported 0.123
```

Every case reruns from `cases/<n>/run.py`.

*Thanks to the authors of every audited repository for publishing code and data. That is what made checking possible.*
