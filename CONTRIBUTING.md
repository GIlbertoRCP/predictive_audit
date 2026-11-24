# Contributing to predictive-audit

Contributions are welcome—whether adding new forecasting baselines, leakage checks, benchmarks, or audited case studies.

## Local setup

```bash
uv sync --group dev
make check      # ruff lint + mypy strict typecheck + pytest coverage gate
```

## Adding a new case study

1. Never commit third-party datasets or weights unless explicitly licensed; fetch them at a pinned commit with SHA-256 verification via `cases/_fetch.py`.
2. Document every preprocessing detail, reproduction check, and caveat (including caveats that cut against the finding) in `cases/<nn>_<name>/README.md`.
3. Keep reproduction scripts runnable via `uv run --group cases python cases/<nn>_<name>/run.py`.

## Pull request checklist

- All new code includes PEP 484 type annotations (`make typecheck`).
- Code passes `ruff check` (`make lint`).
- Unit and invariant tests cover new checks, metrics, or baselines (`make test`).
