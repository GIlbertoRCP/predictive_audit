# Contributing to predictive-audit

Contributions are welcome—whether adding new forecasting baselines, leakage checks, benchmarks, or audited case studies.

## Local setup

You need a C++20 compiler (GCC 12+, Clang 15+/Apple Clang 15+, or MSVC 2022) and CMake ≥ 3.24.

```bash
uv sync --group dev   # builds the native extension (predictive_audit._native) via scikit-build-core
make check            # ruff + mypy + pytest coverage gate + native C++ unit tests
```

`uv sync` / `uv run` rebuild the extension automatically when files under `native/` change.

## Native engine and backends

The engine is being moved into C++ (`native/`) in phases; see [docs/NATIVE_CORE_PLAN.md](docs/NATIVE_CORE_PLAN.md).
The pure-Python code paths stay as the reference oracle, and `PREDICTIVE_AUDIT_BACKEND` picks one:

```bash
make test-both            # full suite on the native engine and on the Python reference
make native-test          # standalone CMake build: library, predictive-audit binary, Catch2 tests
make native-asan          # same, with AddressSanitizer + UndefinedBehaviorSanitizer
make native-tsan          # same, with ThreadSanitizer (the bootstrap runs on a thread pool)
make bench-native         # Python reference vs native kernels -> benchmarks/native_kernels.json
make format               # clang-format (pinned 19.1.7) on native/
```

New native kernels keep the existing Python signature and dispatch on `_backend.native` at call
time, so every kernel gets a parity test in `tests/test_native_parity.py` that runs identical
inputs through both backends. Where NumPy's evaluation order is defined (pairwise `np.sum`,
`np.cumsum`, `np.percentile`, `default_rng` draws), native results must be **bit-identical**;
reproduce NumPy's order rather than loosening a tolerance.

Native kernels use all cores by default; set `PREDICTIVE_AUDIT_THREADS=N` (or call
`predictive_audit._native.set_num_threads(N)`) to limit them. Results never depend on the
thread count.

## Adding a new case study

1. Never commit third-party datasets or weights unless explicitly licensed; fetch them at a pinned commit with SHA-256 verification via `cases/_fetch.py`.
2. Document every preprocessing detail, reproduction check, and caveat (including caveats that cut against the finding) in `cases/<nn>_<name>/README.md`.
3. Keep reproduction scripts runnable via `uv run --group cases python cases/<nn>_<name>/run.py`.

## Pull request checklist

- All new code includes PEP 484 type annotations (`make typecheck`).
- Code passes `ruff check` (`make lint`).
- Unit and invariant tests cover new checks, metrics, or baselines (`make test`).
- Native changes pass `make native-test` and `make native-asan`, are clang-formatted (`make format`), and add a native-vs-Python parity test.
