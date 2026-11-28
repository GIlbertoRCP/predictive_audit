# Native C++ Core — Implementation Plan

**Status:** Phases 0–4 done · **Target release:** v0.3.0 (kernels) → v0.4.0 (standalone binary)

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
| RNG | **Bit-exact NumPy** `SeedSequence` + `PCG64` + `Generator.integers` (Lemire) | Same seed, same draws as `np.random.default_rng`, so bootstrap CIs match the Python reference exactly. Draws are generated sequentially; evaluation runs in parallel (see Determinism contract). |
| FFT | **pocketfft** (header-only C++) | It is NumPy's own FFT backend, so autocovariances match to ~1e-15. |
| Linear algebra | Hand-rolled Cholesky for the (p+1)×(p+1) Ridge system, QR fallback | Systems are ≤ ~20×20; no need for Eigen. Add Eigen later only if needed. |
| Student-t tail (DM p-value) | In-house regularized incomplete beta (Lentz continued fraction, Stirling-differenced log-beta for large df) | Matches `scipy.stats.t.sf` to < 2e-9 relative over df 1–5e7 without a large Boost download. |
| CSV/TSV parsing | In-house tokenizer + per-column parallel conversion; pandas' own float converters (a port of `precise_xstrtod` for pandas ≤ 3.0, **fast_float** for ≥ 3.1) | Results must equal `pd.read_csv` bit for bit, so a correctly rounded parser alone is not enough. |
| Parquet / Feather | Optional `PAUDIT_WITH_ARROW=ON` build flag | Arrow is heavy; ship it in the wheel path (pyarrow) first, native later. |
| CLI args / config | Hand-written argparse-compatible parser + **toml++** | Matches argparse (prefix abbreviations, negative-number values, messages) and the TOML-defaults behaviour exactly. |
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

- [x] `compute` / `score` / `table` for all additive and confusion metrics (one pass per metric; NumPy pairwise-summation order).
- [x] `_is_aligned_panel`, `_group_circular_layout`, prefix sums, `_block_sums`, `_eval_from_sums`. Prefix columns are interleaved (one cache line per block endpoint for all K statistics) and summed with a K-wide pairwise sum that keeps NumPy's order per column.
- [x] `block_bootstrap_diff` for the 9 prefix metrics, draws evaluated in parallel on a persistent thread pool.
- [x] `estimate_block_length` (Politis–White via pocketfft), `default_block_length`.
- [x] `diebold_mariano` (Newey–West Bartlett + HLN, in-house t-tail), `benjamini_hochberg`, `holm_bonferroni`.
- [x] `np.percentile`-compatible linear interpolation (including NumPy's `_lerp`) for CI endpoints.
- [x] nanobind: contiguous float64 / int64 `ndarray`s zero-copy; GIL released around every kernel. Inputs whose dtype semantics differ in C++ (object/string groups, NaT times, seeds outside uint64) fall back to Python.
- [x] ~~Test hook for injected block starts~~ — not needed: the RNG is bit-exact, so parity tests compare whole calls.
- [x] `-ffp-contract=off` on GCC/Clang so no a·b+c is fused into an FMA (FMA would change last bits vs NumPy and between CPUs).

**Parity bar (met):** bootstrap intervals, metrics and block lengths are **bit-identical** to the Python reference on hypothesis-generated inputs (all 9 metrics; plain, per-group, and int/float/datetime-aligned panels; NaN and degenerate inputs). Diebold–Mariano agrees to ~1e-13 relative (NumPy's dot products use BLAS). Native results do not depend on the thread count. Clean under ASan, UBSan and TSan.

**Result:** the bootstrap is 3–11× faster per call (single process, all cores), with up to ~4× from threads on large memory-bound workloads (`benchmarks/native_kernels.json`). End-to-end `audit()` improves only 1.1–1.3×. Once the bootstrap is native, pandas sorting, baselines and checks dominate, which is exactly what Phases 2–3 move.

### Phase 2: Baselines, rank metrics, reshape (≈2 weeks)

- [x] Group-aware `lag_target` / `persistence` / `seasonal_naive`, following pandas `groupby(sort=False)`: groups in order of first appearance, rows need not be contiguous. Groups run in parallel.
- [x] `moving_average`: pandas' `roll_mean` reproduced exactly, including Kahan add/remove compensation, the constant-window guard (GH#42064), the all-positive/all-negative sign guards, and ±inf treated as NaN. **Bit-identical.**
- [x] `ewma`: pandas `ewm(span, adjust=False, min_periods=1)`, including NaN gaps and the `com == 1` special case. Bit-identical wherever pandas doesn't fuse `a*b + c` into an FMA. pandas' own arm64 wheels do fuse, so its result there differs by ~1e-13. The native kernel never fuses, so it gives the same answer on every platform. pandas 3.0 also changed span-3 (`com == 1`) weighting after NaN gaps, so the kernel follows the installed pandas major version (CI's Python 3.10 job runs pandas 2.3).
- [x] `linear_ar`: per-group Ridge with pooled fallback below `lags + 5` rows, groups fit in parallel. LU with partial pivoting, like LAPACK `dgesv`. Agrees with NumPy to about cond(X'X + αI)·ε, since BLAS/LAPACK order is unspecified. A singular system raises and falls back to NumPy's least squares. `constant` stays in Python (a single mean).
- [x] `estimate_season`: pooled FFT autocovariance via pocketfft, same decay/peak rules. Same period on every tested input.
- [x] `optimal_threshold` in O(n log n): sort positive and negative scores once, then count by binary search per candidate (previously O(candidates · n)). **Bit-identical.**
- [x] `roc_auc` / `pr_auc` reproduce scikit-learn's `roc_curve(drop_intermediate=True)` + trapezoid and `precision_recall_curve` + step integral. **Bit-identical.** Labels other than 0/1 or non-finite scores fall back to scikit-learn, which defines those cases.
- [x] **Rank-metric bootstrap rewrite:** scores sorted once; each draw becomes per-row multiplicities, and AUC/AP an O(n) weighted sweep. Curve points depend only on counts per distinct score, so this is **bit-identical** to calling scikit-learn on each resampled array. 130–170× faster.
- [ ] ~~`wide_to_long`~~ moved to Phase 4: from Python it still has to build a DataFrame, so there's nothing to gain until the standalone CLI reads wide files directly.

**Result:** at Case 03 scale (785 series × 1,000 steps, every baseline on, automatic season), `audit()` goes from 3.6 s to 0.21 s (17×) with identical verdicts. Single-series audits on 500k rows are still only 1.1–1.3× faster: pandas sorting and the leakage checks dominate, and those are Phase 3.

### Phase 3: Checks and the full pipeline in C++ (≈2–3 weeks)

- [x] Columnar input (`AuditColumns`) instead of a general `Table` class. Python keeps only cheap column extraction:
  - integer and float sort keys pass through;
  - datetimes pass through as int64 ticks;
  - strings, categoricals and objects become `pd.factorize(sort=True)` codes;
  - split labels become one train/test/other code per distinct label.

  Every audit use of time (sorting, duplicate detection, panel alignment) depends only on order and equality, so codes give identical results; panel alignment was checked case by case. NaN group keys, NaT/NaN times, and non-string column labels keep the pandas path.
- [x] Sort and group semantics identical to `df.sort_values(group_cols + [time], kind="stable")` + `groupby(...).ngroup()`, with an O(n) already-sorted fast path.
- [x] All six checks in C++, summary strings ported verbatim. Python's f-string number formats (`.3f`, `+.4f`, `.3g`, `.4g`, `g`, `.0%`) are reproduced with `snprintf` plus explicit NaN/inf spelling and tested against CPython output. `std::format` was avoided because it needs macOS 13.3+ for floating point under the macOS 11 wheel target.
- [x] `audit(AuditColumns, AuditConfig) -> AuditReport`, and `audit_each_series` for `batch_audit`, running series in parallel (BH-FDR in Python afterwards, as before). Data errors carry the reference's messages, raised in the reference's order.
- [x] Python `audit()` / `batch_audit()` dispatch to the native pipeline. `extra_baselines` are passed as columns; `extra_checks` run in Python on the sorted evaluation rows and arrays the native call returns.
- [x] SciPy and scikit-learn are imported only on first use by the reference path, and moved to an optional `[reference]` extra (plus the dev group and wheel tests). Package import time: **0.70 s → 0.18 s**.
- [x] Feature leakage: average ranks via a plain sort (tie order doesn't matter), and all correlations computed in parallel before the reference-order leak decisions.

**Result:** 480 randomized full audits gave identical verdicts and check statuses, with identical numbers and report text apart from the LAPACK/BLAS-bound cases (classification Ridge AR rank metrics, R² near zero), which the parity tests document. Case 01 output is byte-identical. Single-series `audit()` is 3.4–8.3× faster at 10k rows and 2.5–3.5× at 500k (with feature leakage), a 785-series panel audit is 27× faster, and `batch_audit` over 785 series is 34× faster. Found and fixed along the way: `report.to_text()` crashed on non-string prediction column labels (both backends).

### Phase 4: Standalone CLI (≈2 weeks)

- [x] `paudit::cli::run` (`native/src/cli.cpp`) with the same flags, defaults, messages and 0/1/2 exit codes as `cli.py`, plus `--threads` and `--version`:
  - The argument parser is hand-written instead of CLI11, because the goal is argparse's behaviour: unique-prefix abbreviations (`--targ`), `--flag=value`, negative numbers as values, and argparse's error messages.
  - TOML config uses the same two-pass rule: flags override the `[audit]` table. String values go through each flag's type conversion and other TOML types are used as they are, as `set_defaults` does. TOML is read with toml++.
  - `--wide` / `--wide-keep` reshape natively, which also covers the `wide_to_long` item moved from Phase 2.
- [x] The CSV/TSV reader (`native/src/csv.cpp`) gives the same columns as `pd.read_csv` with default options:
  - **Column types:** int64, float64, bool or string, with pandas' inference order; ints with missing values become float64.
  - **Header and text:** pandas' 19 default NA strings; quoted fields with doubled quotes; BOM; CRLF; blank and whitespace-only lines skipped; `Unnamed: i` and `.1`/`.2` header names; pandas' `ParserError`/`EmptyDataError` messages, including its line numbering.
  - **Floats are bit-identical, which meant porting pandas' converter.** pandas' default float parser is not correctly rounded, and two findings shaped the port:
    - pandas **≤ 3.0** (`precise_xstrtod`) keeps 17 digits in a double. Its **macOS arm64 wheels are compiled with FMA contraction** and its Linux and Windows wheels are not, so the last bit differs between platforms for ~6% of 17-digit inputs. Checked against pandas 2.3.3 and 3.0.6 on macOS arm64, Linux arm64 and x86-64.
    - pandas **3.1** switched to fast_float, plus a rewritten fallback for tokens fast_float declines (a `+` sign with surrounding whitespace, …).

    The reader implements both and follows the installed pandas. The standalone binary follows pandas ≥ 3.1.
  - **Left to pandas (`Unsupported`):** integers beyond int64, a first row longer than the header (pandas turns it into an index), invalid UTF-8, and bare-CR line endings, where pandas' tokenizer has bugs (repeated fields after a CR followed by whitespace, a dropped leading field after a blank line).
- [x] Time columns: numeric if the column parses as numbers, otherwise sorted as text, as today.
- [x] Renderers (`native/src/report.cpp`): text and Markdown byte-identical to `report.py`. JSON follows `json.dumps(indent=2)`: same key order, ASCII escapes with surrogate pairs, NaN/Inf → `null`, and Python `repr` floats. `std::to_chars` needs macOS 13.3+, so floats use shortest round-trip via `snprintf`/`strtod`, with the power-of-two neighbour case handled. That matches `repr` on 2M random doubles and every power of two.
- [x] `--threads N` (default: `PREDICTIVE_AUDIT_THREADS`, else all cores). Reports don't depend on it.
- [x] The `predictive-audit` console script calls `_native.cli_main(argv)`, so there is one CLI implementation. Inputs only pandas reads (Parquet, Feather, compressed files, missing key values, a text column used as numbers, …) come back as `fallback` and run through the reference implementation. The standalone binary reports them as errors.

**Golden tests** (`tests/golden/`): 11 cases, covering all three formats, a TOML config, `--wide` on TSV, a BOM/CRLF/quoted file and four errors. They were recorded from the pure-Python reference CLI and are checked by a stdlib-only script, so CI runs them against the standalone binary on Linux, macOS and Windows. Exit codes, stderr, text and Markdown must match byte for byte. JSON floats are compared to 1e-9 because two reference values come from BLAS dot products.

**Result:**
- **Startup:** `predictive-audit --version` takes **4.7 ms** against 246 ms for the Python CLI.
- **Small audit:** a 10,000-row CLI audit runs in **7.3 ms** against 744 ms (100×).
- **Panel audit:** a 785-series × 1,000-step CSV takes 0.30 s against 1.04 s; parsing the CSV is 1.8× faster than `pd.read_csv`.
- **Parity:** 8,000 generated CSV files read identically to pandas, and 300 random CLI audits give the same output as the Python CLI.
- **Bug found and fixed:** the reference's temporal-split check cast missing group keys to int64. That crashed on x86-64 and silently merged those rows into the first group on arm64.

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
- Bootstrap block starts come from one NumPy-identical PCG64 stream, generated sequentially in bounded chunks (≤ 4M starts at a time). Each draw's statistic is then evaluated in parallel and written to its own slot, so results equal NumPy's and never depend on scheduling.
- `batch_audit` gives every group the same `seed`, as it does today.
- Floating-point reductions use a fixed order, never order-of-completion across threads.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| ~~Bootstrap CIs change vs 0.2.0 for the same seed~~ **Resolved in Phase 1.** | The native engine reimplements NumPy's RNG exactly, so there is no seed-semantics change. Draw generation is sequential, but evaluation dominates (several prefix lookups per drawn start), so threads still scale. |
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
