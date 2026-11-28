"""Compare a predictive-audit CLI against the golden outputs in expected/ (standard library only).

    python tests/golden/check.py build/native/predictive-audit   # the standalone executable
    python tests/golden/check.py                                 # predictive_audit's console script

Exit codes, stderr, and text/Markdown output must match byte for byte. JSON output must have
the same structure, types and non-float values, with floats equal to 1e-9 relative: NumPy's
reference computes a few dot products (autocorrelation, Diebold-Mariano) through BLAS, whose
last bit is not portable.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
Runner = Callable[[list[str]], tuple[int, str, str]]


def _same_json(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(b, dict):
        if not isinstance(a, dict) or list(a) != list(b):
            return [f"{path}: keys {list(a) if isinstance(a, dict) else a!r} != {list(b)}"]
        return [m for k in b for m in _same_json(a[k], b[k], f"{path}/{k}")]
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            return [f"{path}: {a!r} != {b!r}"]
        pairs = enumerate(zip(a, b, strict=True))
        return [m for i, (x, y) in pairs for m in _same_json(x, y, f"{path}[{i}]")]
    if isinstance(b, float) and isinstance(a, float):
        return [] if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) else [f"{path}: {a!r} != {b!r}"]
    return [] if (type(a) is type(b) and a == b) else [f"{path}: {a!r} != {b!r}"]


def compare(name: str, args: list[str], run: Runner) -> list[str]:
    """Differences between `run(args)` and the golden output of case `name` (empty if none)."""
    code, out, err = run(args)
    expected_out = (HERE / "expected" / f"{name}.out").read_text(encoding="utf-8")
    meta = json.loads((HERE / "expected" / f"{name}.meta.json").read_text(encoding="utf-8"))
    problems = []
    if code != meta["exit_code"]:
        problems.append(f"exit code {code} != {meta['exit_code']}")
    if err != meta["stderr"]:
        problems.append(f"stderr {err!r} != {meta['stderr']!r}")
    if "--format" in args and args[args.index("--format") + 1] == "json" and expected_out:
        problems += _same_json(json.loads(out), json.loads(expected_out))
    elif out != expected_out:
        problems.append("stdout differs:\n" + out + "\n--- expected ---\n" + expected_out)
    return [f"{name}: {p}" for p in problems]


def cases() -> list[dict[str, Any]]:
    return list(json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"])


def binary_runner(binary: str) -> Runner:
    def run(args: list[str]) -> tuple[int, str, str]:
        proc = subprocess.run([binary, *args], cwd=HERE, capture_output=True, text=True, encoding="utf-8")
        return proc.returncode, proc.stdout, proc.stderr

    return run


def main(argv: list[str]) -> int:
    binary = argv[0] if argv else None
    if binary is None:
        run = binary_runner(sys.executable)
        run_args = lambda a: run(["-m", "predictive_audit.cli", *a])  # noqa: E731
    else:
        binary = str(Path(binary).resolve())
        run_args = binary_runner(binary)
    problems = [p for case in cases() for p in compare(case["name"], case["args"], run_args)]
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{len(cases())} golden cases, {len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
