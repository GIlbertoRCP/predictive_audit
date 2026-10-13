# predictive-audit

**Does your forecasting model beat "the next step looks like this one"?**

A small tool that checks a time-series forecasting result against the baselines it has to beat
before it means anything, and against the evaluation mistakes that make it look better than it is:

| Check | What it catches |
|---|---|
| **Beats trivial baselines** | Model vs. persistence, moving average, seasonal-naive and majority class/mean on the same rows, with a moving-block bootstrap CI (an i.i.d. bootstrap is far too optimistic on autocorrelated data). |
| **Reported score** | A published number that a trivial baseline already reaches on the same data. |
| **Label memory** | Targets so autocorrelated that any model posts a high score; only the margin over persistence matters. |
| **Temporal split** | Shuffled train/test splits where test rows sit between training rows, so overlapping windows leak. |
| **Feature leakage** | Feature columns that are near-copies of target values not yet observed at prediction time. |

## Why this exists

My team's course project ([Predictive_Activation](https://github.com/GIlbertoRCP/Predictive_Activation), v0) reported a congestion forecaster with F1 ≈ 0.99.
It turned out that "next step is congested if this step is" scores F1 0.993 on the same data. The model had learned
autocorrelation, and a shuffled split of overlapping windows hid that. This tool automates the checks that would have
caught it, so I can run them on other published results.

## Case studies

| # | Result audited | Claimed | Verdict | Finding |
|---|---|---|---|---|
| 01 | [Predictive_Activation v0 congestion forecaster](cases/01_predictive_activation_lstm/) | F1 ≈ 0.99 | **FAIL** | Shuffled split leaks; with a chronological split the model *loses* to a 3-step moving average (F1 0.984 vs 0.987, CI excludes 0). |

More cases (public network-telemetry and traffic forecasting results) are in progress.

## Usage

```bash
uv sync
uv run predictive-audit data.csv \
    --target future_congestion --pred model_score \
    --group switch_id,port_no --time timestamp --split split \
    --features rx_mbps,latency_ms --metric f1 --reported 0.99
```

Or from Python:

```python
from predictive_audit import audit, report
r = audit(df, "future_congestion", pred="model_score", groups=["switch_id", "port_no"],
          time="timestamp", split="split", metric="f1")
print(report.to_text(r))   # also to_markdown / to_json
```

**Data convention:** one row per (series, time step). The `--target` column holds the value being forecast,
already shifted so that row *t* contains y[t + horizon]. Baselines are rebuilt from that column alone, so
no extra inputs are needed. Only rows with `--split` = `test` are scored. `--fail-on warn|fail|never` sets the exit
code, so the audit can gate CI.

## Development

```bash
uv run pytest
uv run python cases/01_predictive_activation_lstm/run.py
```

## Limitations

- Baselines need ordered history per series. A dataset that only ships pre-built windows can't be audited until it is unrolled.
- The best baseline is picked on the test rows, which slightly favours the baselines. That's conservative toward the model.
- Bootstrap blocks default to √n rows and may cross series boundaries. Set `--block` when you know the correlation length.
- Feature leakage detection catches near-copies (|r| ≥ 0.999) of future targets, not subtler leaks such as scalers fit on the full dataset.
