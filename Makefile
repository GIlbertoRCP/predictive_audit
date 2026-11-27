.PHONY: test test-both cov lint typecheck bench bench-native site check native native-test native-asan native-tsan format

test:
	uv run pytest -v

# Run the whole suite on the native engine and on the pure-Python reference.
test-both:
	PREDICTIVE_AUDIT_BACKEND=native uv run pytest -q
	PREDICTIVE_AUDIT_BACKEND=python uv run pytest -q

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

# Standalone C++ build: library, `predictive-audit` binary, Catch2 tests.
native:
	cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DPAUDIT_WERROR=ON
	cmake --build build/native --parallel

native-test: native
	ctest --test-dir build/native --output-on-failure

native-asan:
	cmake -S native -B build/asan -DCMAKE_BUILD_TYPE=Debug -DPAUDIT_WERROR=ON -DPAUDIT_SANITIZE=address,undefined
	cmake --build build/asan --parallel
	ctest --test-dir build/asan --output-on-failure

native-tsan:
	cmake -S native -B build/tsan -DCMAKE_BUILD_TYPE=Debug -DPAUDIT_WERROR=ON -DPAUDIT_SANITIZE=thread
	cmake --build build/tsan --parallel
	ctest --test-dir build/tsan --output-on-failure

bench-native:
	uv run python benchmarks/bench_native.py --runs 15 --out benchmarks/native_kernels.json

format:
	find native \( -name '*.cpp' -o -name '*.hpp' \) | xargs uvx clang-format==19.1.7 -i

check: lint typecheck cov native-test
