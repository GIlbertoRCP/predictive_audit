.PHONY: test cov lint typecheck bench site check

test:
	uv run pytest -v

cov:
	uv run pytest --cov=predictive_audit --cov-report=term-missing --cov-fail-under=90

lint:
	uv run ruff check src tests benchmarks

typecheck:
	uv run mypy src

bench:
	uv run python benchmarks/bench_audit.py --out benchmarks/upgraded_audit.json
	uv run python benchmarks/bench_calibration.py --out benchmarks/calibration.json

site:
	uv run python site/build_data.py

check: lint typecheck cov
