# Changelog

All notable changes to **predictive-audit** are documented here.

## [Unreleased]

### Added
- **Native C++20 engine scaffold** (`native/`, Phase 0 of [docs/NATIVE_CORE_PLAN.md](docs/NATIVE_CORE_PLAN.md)): `libpaudit` static library, standalone `predictive-audit` executable (`--version` / `--build-info` for now), Catch2 unit tests, strict warnings, opt-in ASan/UBSan.
- **`predictive_audit._native`** nanobind extension and **`PREDICTIVE_AUDIT_BACKEND=auto|native|python`** backend switch; `predictive_audit.backend()` reports the active one. The pure-Python implementation is kept as the reference oracle.
- CI: native build + tests on Linux/macOS/Windows, sanitizer job, clang-format check, and the full pytest suite on both backends.

- **Native Phase 1 kernels** behind the same Python signatures: `compute`/`score`/`table` for all nine additive and confusion-matrix metrics, the prefix-sum circular block bootstrap (plain, per-group, and time-aligned panels), Politis–White and default block lengths, Diebold–Mariano HAC, and Benjamini–Hochberg / Holm–Bonferroni.
- **Bit-identical results:** the engine reimplements NumPy's `SeedSequence` + `PCG64` + `Generator.integers` (Lemire) and NumPy's pairwise summation, `cumsum` and `percentile` order, so bootstrap intervals, metrics and block lengths match the Python reference exactly for the same seed. Diebold–Mariano agrees to ~1e-13 (NumPy's dot products go through BLAS).
- **Parallel bootstrap** on a persistent thread pool (`PREDICTIVE_AUDIT_THREADS`); results do not depend on the thread count. The pool is fork-safe (a `fork()`ed child, e.g. `multiprocessing` on Linux, starts a fresh pool) and can be resized while jobs run. ROC-AUC / PR-AUC stay on the Python path until Phase 2.
- **Native Phase 2 kernels:**
  - **Baselines:** group lags, pandas' Kahan rolling mean (bit-identical), EWMA, per-group Ridge AR(p) fit in parallel, and FFT season estimation.
  - **Rank metrics:** scikit-learn-exact ROC-AUC and average precision (bit-identical), and an O(n log n) optimal-threshold search.
  - **Rank-metric bootstrap:** sorts once, then runs a weighted O(n) sweep per draw, bit-identical to calling scikit-learn on each resample.
  - **Result:** a 785-series audit with every baseline runs 17× faster (3.6 s → 0.21 s); the Ridge AR baseline is 71× faster and the ROC/PR bootstraps 130–170× faster.
- **Native Phase 3 pipeline:**
  - `audit()` and `batch_audit()` run end to end in C++: sort, group, baselines, metric table, and all six checks with the same summary text.
  - Results: single-series audits are 2.5–8.3× faster, a 785-series panel audit 27×, and `batch_audit` over 785 series 34× (series in parallel).
  - Verdicts, statuses, report text and numbers match the reference except the documented LAPACK/BLAS-bound cases.
  - `extra_baselines` and `extra_checks` keep working unchanged.
- **Native Phase 4: standalone CLI.**
  - The `predictive-audit` executable runs a full audit with no Python on the machine. It has the same flags, TOML config handling, `--wide` mode, reports and 0/1/2 exit codes, plus `--threads N` and `--version`.
  - Startup takes 4.7 ms (the Python CLI: 246 ms), and a 10,000-row audit takes 7 ms (744 ms).
  - The package's console script runs the same C++ code. Only inputs that need pandas (Parquet, Feather, compressed files, missing values in key columns, …) go through the Python implementation.
  - Its CSV/TSV reader returns the same columns as `pandas.read_csv`, floats bit for bit. It follows the installed pandas: its float parser differs between pandas 2.x, 3.0 and 3.1, and before 3.1 it also rounds differently on macOS arm64.
  - Golden output tests in `tests/golden/` also run against the standalone binary in CI.
- `tests/test_native_parity.py`, `tests/test_native_parity_baselines.py`, `tests/test_native_parity_audit.py`, `tests/test_native_csv.py` and `tests/test_native_cli.py` (hypothesis-driven native-vs-reference checks), `benchmarks/bench_native.py`, ThreadSanitizer CI job.

### Changed
- **SciPy and scikit-learn are no longer runtime dependencies.** Only the pure-Python reference backend (`PREDICTIVE_AUDIT_BACKEND=python`) uses them; install them with `pip install 'predictive-audit[reference]'`. They are imported on first use, so `import predictive_audit` drops from 0.70 s to 0.18 s.
- Build backend switched from setuptools to **scikit-build-core** + nanobind (C++20 compiler and CMake ≥ 3.24 needed for source builds).
- Release workflow builds platform wheels with **cibuildwheel** (Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64; CPython 3.10–3.13) and tests each wheel.

### Fixed
- `report.to_text()` / `to_markdown()` crashed when the prediction column label was not a string.
- The temporal-split check crashed with an `IndexError` on x86-64 when a group column had missing values. On arm64 it silently merged those rows into the first group. Rows without a group key now form their own group.
- `block="auto"` (Politis–White) no longer depends on floating-point noise when the loss differential is mostly zero, e.g. a classifier that usually agrees with its baseline. Autocovariances that are exactly zero come back from the FFT as ~1e-19 noise that differs by platform (NumPy's complex multiply is fused on arm64). An exact-zero test then picked √n or 1 depending on that noise. The estimate now shrinks smoothly to 1 as serial correlation vanishes, and a long-run variance within 1e-10 of the variance counts as zero.

## [0.2.0] - 2026-11-24

### Added
- **Prefix-sum vectorized Circular Block Bootstrap (CBB)** in `metrics.py` (up to 100× faster on classification F1, 42% lower peak memory) with panel-aligned timestamp resampling and zero-edge-bias within-group circular padding.
- **Diebold–Mariano HAC test** (Newey–West Bartlett kernel + Harvey–Leybourne–Newbold finite-sample adjustment) and **Benjamini–Hochberg / Holm–Bonferroni** multiple-testing corrections.
- **`batch_audit()`** API for multi-series auditing with automatic FDR q-values and aggregated tallies.
- **Expanded baselines:** native Cython rolling `moving_average`, causal `ewma`, and closed-form Ridge `linear_ar` (`--ar-lags`).
- **Expanded metrics & calibration:** `smape`, `brier`, `mase`, `skill_score`, and train-only `--threshold auto` classification threshold tuning.
- **Expanded leakage checks:** horizon embargo gap validation, duplicate timestamp detection, monotonic nonlinear leakage (Spearman), and global `[0, 1]` min-max scaler leakage detection.
- **Extensibility & CLI polish:** `extra_baselines` / `extra_checks` hooks, `--wide` CLI mode, `.tsv` / `.parquet` / `.feather` support, TOML `--config`, strict RFC 8259 JSON (`allow_nan=False`, `schema_version = "1.0"`), and graceful exit code `2` error handling.
- **Production tooling:** GitHub Actions CI (`ci.yml` across Python 3.10–3.13), reusable GitHub Action (`action.yml`), `Makefile`, `LICENSE` (MIT), `py.typed`, `docs/DESIGN.md`, `BENCHMARKS.md`, and 46 automated tests.

## [0.1.0] - 2026-10-13

- Initial release of `predictive-audit` CLI/library, 5 audited network-traffic forecasting case studies, and dependency-free interactive SVG explorer.
