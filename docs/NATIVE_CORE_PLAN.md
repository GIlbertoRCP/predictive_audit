# Native C++ Core — Implementation Plan

**Status:** Phase 0 done · **Target release:** v0.3.0 (kernels) → v0.4.0 (standalone binary)

## Goal

Move the audit engine into a C++20 library (`libpaudit`) that powers three front ends:

1. **`predictive-audit` standalone binary.** One static executable with no Python. It reads CSV/TSV and keeps today's flags, reports and 0/1/2 exit codes.
2. **The Python package.** `from predictive_audit import audit, batch_audit, report` keeps working unchanged. The package becomes a thin nanobind wrapper.
3. **The GitHub Action.** It downloads the prebuilt binary instead of setting up Python and uv.

The pure-Python implementation stays in the repo as the **reference oracle** that every native result is checked against. Each public function keeps its Python body and dispatches to the native kernel when `_backend.native` is set, so both paths live side by side and parity tests can run the same inputs through either.

## Why: where the time goes today

| What | Today (Python 0.2.0) | Why native helps |
|---|---:|---|
| CLI startup (imports only) | **~1.1 s** (scipy.stats alone ~0.6 s) | A native binary starts in a few ms; this is >100× the audit itself in CI. |
| Single audit, 10k rows, MAE | 8.7 ms | Fewer temporaries; modest 2–5× win. |
| Single audit, 500k rows, F1 | 281 ms | Fused sufficient-stat passes, no 8× float64 confusion arrays. |
| `batch_audit`, 785 OD series (Case 03) | sequential Python loop | Embarrassingly parallel across groups → scales with cores. |
| Monte Carlo calibration (400 trials × 400 boots) | minutes | Parallel trials → can run 10k trials and tighten the calibration table. |
| ROC-AUC / PR-AUC bootstrap | sklearn call per draw, O(n log n) each | Sort once, then O(n) weighted sweep per draw (algorithmic win, see Phase 2). |

Record every one of these before and after in `BENCHMARKS.md`. They are the headline numbers for the write-up and the resume.

## Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Language / standard | C++20 | `std::span`, concepts, `std::format`, `<bit>`; supported by GCC 12+, Clang 15+, MSVC 19.3x. |
| Python bindings | **nanobind** | Smaller, faster-compiling and lower-overhead than pybind11; zero-copy `ndarray` → `std::span<const double>`. |
| Build | CMake ≥ 3.24 + **scikit-build-core** | Replaces setuptools in `pyproject.toml`; same CMake builds the CLI, tests and wheel. |
| Wheels | **cibuildwheel** | manylinux x86_64/aarch64, macOS arm64/x86_64, Windows x64; Python 3.10–3.13. |
| Threading | Small in-house fixed thread pool (`std::jthread`) | Avoids OpenMP (no libomp on Apple Clang by default); deterministic work partitioning. |
| RNG | PCG64 with **per-chunk streams** derived from `(seed, chunk_id)` | Results identical regardless of thread count. Not bit-compatible with NumPy (see Risks). |
| FFT | **pocketfft** (header-only C++) | It is NumPy's own FFT backend, so autocovariances match to ~1e-15. |
| Linear algebra | Hand-rolled Cholesky for the (p+1)×(p+1) Ridge system, QR fallback | Systems are ≤ ~20×20; no need for Eigen. Add Eigen later only if needed. |
| Student-t tail (DM p-value) | **Boost.Math** standalone headers (`students_t` cdf) | Replaces `scipy.stats.t.sf`; header-only, Boost licence. |
| CSV/TSV parsing | In-house, `mmap` + chunked multithreaded parse + **fast_float** | Keeps the binary small and static; good engineering showcase. |
| Parquet / Feather | Optional `PAUDIT_WITH_ARROW=ON` build flag | Arrow is heavy; ship it in the wheel path (pyarrow) first, native later. |
| CLI args / config | **CLI11** + **toml++** | Header-only; matches today's argparse + TOML-defaults behaviour. |
| JSON | Hand-written strict RFC 8259 writer | Small surface; must reproduce `json.dumps(indent=2)` layout and NaN→`null`. |
| Tests / benchmarks | **Catch2** + **Google Benchmark** | Native unit tests; Python parity tests stay in pytest + hypothesis. |
| Dependency fetch | CMake `FetchContent` pinned by commit/tag | No system packages required. |

## Repository layout

```
native/
  CMakeLists.txt
  cmake/                      # warnings, sanitizers, FetchContent pins
  include/paudit/
    types.hpp                 # Column, Table, Mask, GroupIndex, AuditConfig, Check, AuditReport
    metrics.hpp               # compute/score/table, optimal_threshold
    bootstrap.hpp             # prefix-sum CBB, panel layout, Politis–White
    inference.hpp             # Diebold–Mariano HAC, BH-FDR, Holm
    baselines.hpp             # persistence, MA, EWMA, seasonal-naive, Ridge AR(p), estimate_season
    checks.hpp                # label memory, beats baselines, reported score, split, leakage, scaler
    audit.hpp                 # audit(), batch_audit()
    report.hpp                # text / markdown / json renderers
    io/csv.hpp  io/toml.hpp
    rng.hpp  thread_pool.hpp
  src/…                       # one .cpp per header
  cli/main.cpp                # standalone `predictive-audit`
  python/module.cpp           # nanobind module → predictive_audit._native
  tests/                      # Catch2 unit tests
  bench/                      # Google Benchmark microbenchmarks
src/predictive_audit/
  _backend.py                 # PREDICTIVE_AUDIT_BACKEND switch + use() context manager for parity tests
  _native.pyi                 # type stubs for the extension
  audit.py, metrics.py, …     # public API; each function dispatches native → Python reference
tests/
  parity/                     # native vs reference (hypothesis + fixed fixtures)
  golden/                     # CLI outputs on case datasets
```

`_backend.py` reads `PREDICTIVE_AUDIT_BACKEND=auto|native|python` (default `auto`: native if importable). The full existing pytest suite runs against **both** backends in CI throughout the migration.

> **Change from the first draft:** instead of moving today's code into a separate `_reference/` package, the Python bodies stay where they are and each function dispatches at call time. That avoids duplicating code and keeps private helpers (`metrics._is_aligned_panel`, …) importable from the same place.

## Phases

Each phase ends with something shippable and a benchmark delta.

### Phase 0: Scaffolding (≈1 week)

- [x] `native/` CMake project, C++20, strict warnings (`PAUDIT_WERROR`), clang-format (pinned 19.1.7), clang-tidy config.
- [x] Switch `pyproject.toml` build backend to scikit-build-core; `_native` module exposing `version()` / `build_info()`; version read from `pyproject.toml`.
- [x] `_backend.py` switch with `use()` context manager; `predictive_audit.backend()`; no behaviour change.
- [x] CI jobs: native build + Catch2 on Linux/macOS/Windows; ASan+UBSan job; clang-format check; pytest on both backends; wheel build on three OSes.
- [x] Release workflow moved to cibuildwheel (pulled forward from Phase 5, since the old single `uv build` would have shipped one Linux wheel only).

**Done when:** `uv sync && uv run pytest` builds the extension and passes with both backends, and CI is green on all three OSes.

### Phase 1: Hot-path kernels (≈2–3 weeks)

Port `metrics.py` behind the same Python signatures:

- [ ] `compute` / `score` / `table` for all additive and confusion metrics (one fused pass that builds every sufficient statistic, not 8 separate arrays).
- [ ] `_is_aligned_panel`, `_group_circular_layout`, prefix sums, `_block_sums`, `_eval_from_sums`.
- [ ] `block_bootstrap_diff` for the 9 prefix metrics, parallel over 256-draw chunks with per-chunk RNG streams.
- [ ] `estimate_block_length` (Politis–White via pocketfft), `default_block_length`.
- [ ] `diebold_mariano` (Newey–West Bartlett + HLN, Boost.Math t-tail), `benjamini_hochberg`, `holm_bonferroni`.
- [ ] `np.percentile`-compatible linear interpolation for CI endpoints.
- [ ] nanobind: accept contiguous float64 / int64 `ndarray`s zero-copy; release the GIL around every kernel.
- [ ] **Test hook:** `block_bootstrap_diff(..., starts=<int64[n_boot, n_blocks]>)` lets parity tests inject the exact block starts NumPy drew, so CIs can be compared exactly, independent of RNG.

**Parity bar:** deterministic functions ≤ 1e-12 relative (hypothesis-generated inputs, including NaN, ties, constant series and single-group panels). Bootstrap with injected starts ≤ 1e-12. Bootstrap with native RNG: verdicts identical on all six cases, and CI endpoints within Monte Carlo error over 20 seeds.

**Done when:** pytest passes on both backends, and `bench_audit.py` shows the native speedup for 10k/100k/500k rows.

### Phase 2: Baselines, rank metrics, reshape (≈2 weeks)

- [ ] Group-aware `lag_target` / `persistence` / `seasonal_naive` on contiguous group runs.
- [ ] `moving_average` matching pandas `rolling(window, min_periods=window).mean()` (its add/remove running sum; tolerance 1e-12).
- [ ] `ewma` matching pandas `ewm(span, adjust=False, min_periods=1)` **including its NaN semantics** (`ignore_na=False`).
- [ ] `constant`, `linear_ar` (per-group Ridge with pooled fallback below `lags + 5` rows; fit groups in parallel).
- [ ] `estimate_season` (pooled FFT autocovariance, same decay/peak rules).
- [ ] `optimal_threshold` (unique values or `np.quantile` candidates).
- [ ] `roc_auc` / `pr_auc` matching sklearn `roc_auc_score` / `average_precision_score` tie handling.
- [ ] **Rank-metric bootstrap rewrite:** sort scores once; each draw becomes a vector of multiplicities, and AUC/AP become an O(n) weighted sweep over the precomputed order. This replaces O(n_boot · n log n) sklearn calls.
- [ ] `wide_to_long`.

**Done when:** these pass reference parity tests, and the rank-metric bootstrap shows its speedup in the benchmarks.

### Phase 3: Checks and the full pipeline in C++ (≈2–3 weeks)

- [ ] Columnar `Table` (double columns, group-key codes, time as int64/double/string, split labels).
- [ ] Sort/group semantics identical to `df.sort_values(group_cols + [time], kind="stable")` + `groupby(...).ngroup()`. That means stable sort, NaN last, and NaN group keys → code −1, which matches pandas' `dropna=True`.
- [ ] All checks: label memory, beats baselines, reported score, temporal split + embargo + duplicate timestamps, feature leakage (Pearson + Spearman with average ranks), scaler leakage. Port the summary strings verbatim.
- [ ] `audit(const Table&, const AuditConfig&) -> AuditReport` and `batch_audit` **parallel across groups**, with BH-FDR applied after all groups finish.
- [ ] Python `audit()` becomes: DataFrame → columns (zero-copy where possible) → native → `AuditReport` dataclass.
  - `extra_baselines` are converted to columns and passed in.
  - `extra_checks` keep running in Python; the native call returns `eval_mask` and the evaluation arrays so these callbacks see exactly what they see today.
- [ ] Drop `scipy` and `scikit-learn` from runtime dependencies (they move to the `dev` group for the reference oracle). Re-measure import time.

**Done when:** all six `cases/*/run.py` produce identical verdicts and tables on the native backend, and Case 03's `batch_audit` shows its thread-scaling numbers.

### Phase 4: Standalone CLI (≈2 weeks)

- [ ] `cli/main.cpp` with the same flags as `cli.py` (CLI11), including the two-pass TOML defaults (CLI flags override `[audit]` table) and `--wide` / `--wide-keep`.
- [ ] CSV/TSV reader matching `pd.read_csv` defaults that matter:
  - numeric vs string column inference;
  - the default NA strings (`""`, `NA`, `N/A`, `NaN`, `nan`, `null`, `NULL`, `#N/A`, …);
  - quoted fields;
  - BOM;
  - CRLF line endings.
- [ ] Time columns: numeric sort if the column parses as numeric, otherwise lexicographic. This matches today's behaviour, because the CLI does not parse dates.
- [ ] Renderers: text / markdown byte-identical to `report.py`. JSON must follow the same key order and `indent=2` layout, use NaN/Inf → `null`, and print floats with shortest round-trip `std::to_chars` + `.0` for integral floats (as Python `repr` does).
- [ ] `--threads N` (default: hardware concurrency).
- [ ] Python's `predictive-audit` console script calls `_native.cli_main(argv)`, so there is one CLI implementation.

**Golden tests:** run the 0.2.0 Python CLI on the case datasets and synthetic fixtures, then store the outputs in `tests/golden/`.
- Deterministic fields must match exactly.
- Bootstrap CIs are compared with a tolerance.
- Text and markdown are snapshot-tested against native output once validated.

**Done when:** `predictive-audit data.csv …` runs from the native binary with no Python on the machine, and its startup is measured in ms.

### Phase 5: Distribution (≈1–2 weeks)

- [ ] `release.yml`: cibuildwheel wheels → PyPI (trusted publishing already configured). Also publish static binaries for:
  - linux-x86_64 / linux-aarch64 (musl, fully static)
  - macos-universal2
  - windows-x64

  Upload them with SHA-256 checksums to the GitHub Release.
- [ ] Homebrew tap (`brew install gilbertorcp/tap/predictive-audit`).
- [ ] `action.yml`: detect runner OS/arch, download the matching binary for the action's tag, verify its checksum, then run it. Keep the uv path as a fallback input (`engine: python`).

**Done when:** the Action runs without `setup-uv`, and its step time is recorded before and after.

### Phase 6: Benchmarks and write-up (≈1 week)

- [ ] Extend `BENCHMARKS.md`:
  - Python vs native for every workload;
  - CLI cold start;
  - thread scaling (1/2/4/8 threads) for `batch_audit` and calibration;
  - peak RSS;
  - binary size.
- [ ] Port `bench_calibration.py`'s Monte Carlo loop to native and rerun it with 10k trials, which gives tighter false-positive-rate estimates.
- [ ] Update `docs/DESIGN.md` (architecture diagram, determinism guarantees, parity strategy) and the README badges.
- [ ] Blog post / HN write-up: the findings, with the native engine as the second act.

### Later (optional)

- Native Parquet/Feather via Arrow behind the build flag.
- Terminal UI viewer (FTXUI) or desktop viewer (Dear ImGui + ImPlot) that reads the report JSON. The `site/` explorer stays as the public write-up page.
- SIMD (`std::experimental::simd` / Highway) for the fused sufficient-stat pass if profiling shows it's memory-bound.

## Determinism contract

- The same input, flags and `--seed` give the same report on every OS and with any `--threads` value.
- Bootstrap draws are split into fixed 256-draw chunks. Chunk *k* gets its own PCG64 stream seeded from `(seed, k)` (via a SeedSequence-style hash), and results are written by chunk index.
- `batch_audit` gives every group the same `seed`, as it does today.
- Floating-point reductions use a fixed order, never order-of-completion across threads.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Bootstrap CIs change slightly vs 0.2.0 for the same seed** (native RNG ≠ NumPy's PCG64 + Lemire bounded-int stream). | Exact parity is proven via injected starts; end-to-end verdict parity is checked on all cases. Call this out in `CHANGELOG.md` as a seed-semantics change in 0.3.0. Bit-compatible NumPy RNG was considered and rejected: its variable rejection count prevents jump-ahead, which would serialize draw generation and cap thread scaling. |
| pandas semantics drift (rolling, ewm NaN handling, `read_csv` inference, sort NaN placement). | Each is an explicit parity test with adversarial fixtures; the reference oracle runs in CI permanently. |
| sklearn AUC/AP tie handling. | Port the exact formulas; hypothesis tests with heavy ties. |
| Float formatting differences in reports. | Golden files; `std::to_chars` for JSON, `std::format` with Python-equivalent specs for text. |
| Two-language build friction for contributors. | `uv sync` still just works (scikit-build-core); the reference backend means pure-Python contributors can still run everything. |
| Scope creep. | Phases 1–3 alone deliver the Python speedup; Phase 4 is the standalone win. Ship each as a release. |

## Rough timeline (part-time)

| Phase | Effort | Release |
|---|---|---|
| 0 Scaffolding | ~1 wk | — |
| 1 Kernels | ~2–3 wks | v0.3.0a1 |
| 2 Baselines & rank metrics | ~2 wks | v0.3.0a2 |
| 3 Full pipeline | ~2–3 wks | **v0.3.0** |
| 4 Standalone CLI | ~2 wks | v0.4.0a1 |
| 5 Distribution | ~1–2 wks | **v0.4.0** |
| 6 Benchmarks & write-up | ~1 wk | — |

About 11–14 weeks part-time. Phase 1 alone is a meaningful, demoable result.
