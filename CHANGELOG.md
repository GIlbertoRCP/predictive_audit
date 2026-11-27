# Changelog

All notable changes to **predictive-audit** are documented here.

## [Unreleased]

### Added
- **Native C++20 engine scaffold** (`native/`, Phase 0 of [docs/NATIVE_CORE_PLAN.md](docs/NATIVE_CORE_PLAN.md)): `libpaudit` static library, standalone `predictive-audit` executable (`--version` / `--build-info` for now), Catch2 unit tests, strict warnings, opt-in ASan/UBSan.
- **`predictive_audit._native`** nanobind extension and **`PREDICTIVE_AUDIT_BACKEND=auto|native|python`** backend switch; `predictive_audit.backend()` reports the active one. The pure-Python implementation is kept as the reference oracle.
- CI: native build + tests on Linux/macOS/Windows, sanitizer job, clang-format check, and the full pytest suite on both backends.

### Changed
- Build backend switched from setuptools to **scikit-build-core** + nanobind (C++20 compiler and CMake ≥ 3.24 needed for source builds).
- Release workflow builds platform wheels with **cibuildwheel** (Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64; CPython 3.10–3.13) and tests each wheel.

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
