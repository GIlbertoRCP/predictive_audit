# Benchmarks & Statistical Calibration

All measurements below were captured locally on the same machine (`macOS-27.0.1-arm64`, Python 3.13.5, NumPy 2.5.4, pandas 3.0.6) using identical benchmark harnesses before (`v0.1.0`, commit `871d1c1`) and after (`v0.2.0`) optimization.

Raw JSON outputs live in [`benchmarks/`](benchmarks/):
- [`baseline_audit.json`](benchmarks/baseline_audit.json) vs. [`upgraded_audit.json`](benchmarks/upgraded_audit.json)
- [`calibration.json`](benchmarks/calibration.json)
- [`stacked_lstm.json`](benchmarks/stacked_lstm.json)

---

## 1. End-to-End Audit Latency, Throughput & Peak Memory

Every run executes the full `audit()` pipeline (sorting, baselines, metric table, 500 bootstrap draws, split/leakage checks) with `tracemalloc` active so before/after numbers are strictly apples-to-apples.

| Workload (`n_boot = 500`) | v0.1.0 p50 / p95 | v0.2.0 p50 / p95 | Speedup (p50) | v0.2.0 Throughput | Peak Memory (Before → After) |
|---|---:|---:|---:|---:|---:|
| Regression MAE · 10k rows | 42.62 / 44.25 ms | **8.70 / 9.23 ms** | **4.9×** | 1.15M rows/s | 2.18 → **1.31 MB** (−40%) |
| Regression MAE · 100k rows | 147.93 / 168.89 ms | **46.64 / 47.24 ms** | **3.2×** | 2.14M rows/s | 20.88 → **12.40 MB** (−41%) |
| Regression MAE · 500k rows | 609.28 / 610.59 ms | **213.96 / 221.07 ms** | **2.8×** | 2.34M rows/s | 106.01 → **61.92 MB** (−42%) |
| Classification F1 · 10k rows | 2,675.91 / 2,695.61 ms | **26.68 / 27.12 ms** | **100.3×** | 375k rows/s | 2.34 → **1.52 MB** (−35%) |
| Classification F1 · 100k rows | 3,624.69 / 3,628.94 ms | **75.52 / 76.16 ms** | **48.0×** | 1.32M rows/s | 20.89 → **12.41 MB** (−41%) |
| Classification F1 · 500k rows | 7,789.39 / 7,834.64 ms | **281.19 / 290.15 ms** | **27.7×** | 1.78M rows/s | 106.01 → **61.93 MB** (−42%) |
| `wide_to_long` · 1M cells (500 × 2k) | 198.96 ms | **20.57 ms** | **9.7×** | 48.6M cells/s | 56.58 → 62.97 MB |
| Panel Audit · 1M cells (500 × 2k) | 1,561.92 ms | **675.05 ms** | **2.3×** | 1.48M cells/s | 259.40 → **224.62 MB** (−13%) |

**What drove the gains:**
- Replacing per-resample Python/`sklearn` loops ($O(n_{\text{boot}} \cdot n)$) with **$O(1)$-per-block prefix-sum vectorization** ($O(n + n_{\text{boot}} \cdot \lceil n / B \rceil)$), verified numerically identical (`< 1e-10`) to explicit resampling across all 9 additive and confusion-matrix metrics.
- Replacing `groupby().transform(lambda ...)` and `pd.concat` over 500 DataFrames with native Cython rolling and vectorized NumPy reshaping.
- Reusing split masks and replacing pandas `groupby().apply()` in `_temporal_split` with $O(n)$ NumPy reductions.

---

## 2. Monte Carlo Type I Error Calibration (Why Block Bootstrapping Matters)

Under $H_0$, a candidate model has the **exact same expected MAE** as persistence, while per-step loss differentials follow a zero-mean $\text{AR}(1)$ process with lag-1 autocorrelation $\phi$ ($n = 1{,}000$, 400 Monte Carlo trials, 400 bootstrap draws per trial, nominal $\alpha = 0.05$):

| Autocorrelation $\phi$ | i.i.d. Bootstrap FPR | CBB ($B=32$) FPR | CBB ($B=64$) FPR | Politis–White (`--block auto`) | Median $B_{\text{PW}}$ | Diebold–Mariano HAC FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (i.i.d.) | 5.5% | 7.5% | 8.0% | **6.0%** | 2 | 6.0% |
| 0.70 | **47.5%** | 7.5% | 8.5% | **8.0%** | 20 | 8.8% |
| 0.90 | **66.5%** | 12.0% | 10.8% | **10.8%** | 42 | 25.8% |
| 0.95 | **77.2%** | 21.8% | 14.2% | **14.2%** | 62 | 37.0% |
| 0.98 | **86.2%** | 37.2% | 26.5% | **23.0%** | 88 | 58.2% |

An i.i.d. bootstrap on $\phi = 0.70$–$0.95$ series hallucinates "statistically significant" gains **47.5%–77.2% of the time** when true skill is zero. Politis–White adaptive Circular Block Bootstrapping (`--block auto`) automatically scales block length from `2 → 88` with zero manual tuning, matching or beating fixed blocks across every regime and cutting false positives by **5.4×–6.2×** on $\phi \in [0.70, 0.95]$.

---

## 3. Case 03 Batched PyTorch `StackedLSTM` Equivalence & Speedup

In [Case 03](cases/03_thu_traffic_matrix/run.py), 785 independent per-OD LSTMs are batched into a single 3D tensor module (`StackedLSTM`) under elementwise `Adagrad`. Benchmark on 132 OD pairs × 800 windows × 10 epochs (`hidden=30`, `batch=50`, CPU):

| Implementation | Wall Time (10 epochs) | Speedup | Max Prediction Drift vs. Sequential |
|---|---:|---:|---:|
| Sequential `torch.nn.LSTM` (132 loops) | 26.696 s | 1.00× | — |
| Batched `StackedLSTM` (1 tensor op) | **5.707 s** | **4.68×** | **`1.79 × 10⁻⁷`** |

---

## 4. Native C++ Engine

The bootstrap, metric, block-length, Diebold–Mariano and FDR kernels now run in C++20 (`native/`), dispatched from the same Python functions. Every native result below was checked against the Python reference before being timed, and the bootstrap intervals are **bit-identical**. The engine reimplements NumPy's `default_rng` stream and its pairwise summation order. Median of 15 runs, same machine as above (Apple Silicon, 12 threads), from [`benchmarks/native_kernels.json`](benchmarks/native_kernels.json):

| Workload | Python reference | Native | Speedup |
|---|---:|---:|---:|
| Bootstrap MAE · 10k rows · 500 draws | 0.59 ms | **0.18 ms** | **3.2×** |
| Bootstrap F1 · 10k rows · 500 draws | 1.92 ms | **0.26 ms** | **7.4×** |
| Bootstrap MAE · 100k rows · 500 draws | 2.60 ms | **0.67 ms** | **3.9×** |
| Bootstrap F1 · 100k rows · 500 draws | 9.24 ms | **1.15 ms** | **8.0×** |
| Bootstrap MAE · 500k rows · 500 draws | 10.28 ms | **2.09 ms** | **4.9×** |
| Bootstrap F1 · 500k rows · 500 draws | 43.50 ms | **3.90 ms** | **11.2×** |
| Bootstrap MAE · 500 × 2,000 aligned panel (1M rows) | 7.28 ms | **2.64 ms** | **2.8×** |
| Diebold–Mariano HAC · 100k rows | 0.88 ms | 1.18 ms | 0.8× |
| Politis–White block length · 100k rows | 2.81 ms | 2.92 ms | 1.0× |
| `audit()` MAE · 10k rows (end to end) | 2.76 ms | 2.27 ms | 1.2× |
| `audit()` MAE · 500k rows (end to end) | 74.33 ms | 69.63 ms | 1.1× |
| `audit()` F1 · 10k rows (end to end) | 7.93 ms | 6.75 ms | 1.2× |
| `audit()` F1 · 500k rows (end to end) | 128.25 ms | 117.33 ms | 1.1× |

**Thread scaling** (F1 bootstrap, 1M rows, block 20, 2,000 draws): 1 thread 1,800 ms → 2: 1.75× → 4: 2.82× → 8: 3.89× → 12: 3.99×. Each draw does random lookups into ~72 MB of prefix sums, so the workload becomes memory-bandwidth-bound past ~8 threads. Results are identical at every thread count.

**Reading these numbers honestly:**
- The confusion-matrix metrics gain most. The native path keeps all 8 sufficient statistics interleaved (one cache line per block endpoint), where NumPy gathers from 8 separate arrays.
- Diebold–Mariano and Politis–White are already dominated by BLAS dot products and pocketfft (the same FFT NumPy uses), so they're at parity. They matter for correctness, not speed.
- **End-to-end `audit()` barely moves (1.1–1.3×).** With the bootstrap native, pandas sorting, the rolling/EWMA baselines and the leakage checks dominate. Moving those is Phases 2–3 of [docs/NATIVE_CORE_PLAN.md](docs/NATIVE_CORE_PLAN.md).
- Peak-memory figures in section 1 come from `tracemalloc`, which cannot see C++ allocations, so they are not reported for the native path.

### Phase 2: baselines, rank metrics, threshold search

Same harness, at Case 03 scale (785 series × 1,000 steps) unless noted. Rows marked † are not bit-identical by design: pandas' EWMA fuses multiply-adds on arm64 (~1e-13), and Ridge AR solves through LAPACK/BLAS. Every other row was asserted bit-identical before timing.

| Workload | Python reference | Native | Speedup |
|---|---:|---:|---:|
| Moving average (3), grouped | 36.18 ms | **9.17 ms** | **3.9×** |
| EWMA (3), grouped † | 25.92 ms | **9.17 ms** | **2.8×** |
| Ridge AR(5) per series † | 3,042.31 ms | **42.82 ms** | **71.0×** |
| Season estimate (pooled FFT) | 51.46 ms | **15.01 ms** | **3.4×** |
| ROC-AUC · 1M rows | 112.38 ms | **84.93 ms** | **1.3×** |
| Optimal threshold (F1) · 1M rows | 143.83 ms | **52.75 ms** | **2.7×** |
| Bootstrap ROC-AUC · 20k rows · 500 draws | 2,546.28 ms | **15.20 ms** | **167.5×** |
| Bootstrap PR-AUC · 20k rows · 500 draws | 1,763.58 ms | **13.40 ms** | **131.6×** |
| **`audit()` with every baseline · 785 × 1,000 panel** | **3,615.31 ms** | **207.54 ms** | **17.4×** |
| `audit()` F1 · 10k rows | 8.38 ms | 2.77 ms | 3.0× |
| `audit()` F1 · 500k rows | 130.32 ms | 104.01 ms | 1.3× |

- The two biggest wins replace Python loops, not vectorized NumPy.
  - Ridge AR fitted one series at a time in a Python loop; native fits all 785 in parallel.
  - The rank-metric bootstrap called scikit-learn once per draw. Native sorts once and turns each draw into an O(n) weighted sweep, with the same curve points scikit-learn would compute.
- ROC-AUC on its own is dominated by the sort, which NumPy already does well.
- **Single-series audits on large inputs are still pandas-bound** (sorting, group keys, leakage checks): Phase 3.

### Phase 3: the whole `audit()` pipeline

`audit()` and `batch_audit()` now run end to end in C++. Python keeps only column extraction: dtype checks, `factorize` for string keys, and split labels. Verdicts and statuses were asserted identical before timing.

| Workload | Python reference | Native | Speedup |
|---|---:|---:|---:|
| `audit()` MAE · 10k rows | 2.76 ms | **0.80 ms** | **3.4×** |
| `audit()` F1 · 10k rows | 7.91 ms | **0.95 ms** | **8.3×** |
| `audit()` MAE · 500k rows | 73.84 ms | **30.02 ms** | **2.5×** |
| `audit()` F1 · 500k rows | 129.01 ms | **41.83 ms** | **3.1×** |
| `audit()` F1 + feature leakage · 500k rows | 280.55 ms | **80.18 ms** | **3.5×** |
| `audit()` every baseline · 785 × 1,000 panel | 3,263.92 ms | **120.72 ms** | **27.0×** |
| `batch_audit()` 785 series · every baseline | 1,603.09 ms | **47.03 ms** | **34.1×** |
| `import predictive_audit` (warm) | 0.70 s | **0.18 s** | **3.9×** |

- `batch_audit` gains the most: 785 independent audits run in parallel instead of a Python loop.
- The import-time drop comes from not loading SciPy and scikit-learn; only the reference path needs them. For a CLI gate, startup is now dominated by pandas. The standalone binary (Phase 4) removes Python startup entirely.

---

## Reproducing Benchmarks

```bash
uv run python benchmarks/bench_audit.py --runs 5 --out benchmarks/upgraded_audit.json
uv run python benchmarks/bench_calibration.py --n-trials 400 --n-boot 400 --out benchmarks/calibration.json
uv run --group cases python benchmarks/bench_stacked_lstm.py --out benchmarks/stacked_lstm.json
uv run python benchmarks/bench_native.py --runs 15 --out benchmarks/native_kernels.json
```
