"""Phase 4: the native CLI (predictive_audit._native.cli_main, the code behind the standalone
executable) vs the pure-Python reference CLI.

- Golden files: both CLIs must reproduce tests/golden/expected (see tests/golden/check.py).
- Random audits written to CSV: same exit code, stderr, and text/Markdown output byte for byte;
  JSON equal up to the documented BLAS/LAPACK-bound float digits.
- Inputs only pandas reads hand back to the reference implementation with the same result.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predictive_audit import _backend, baselines
from predictive_audit.cli import _PANDAS_31, main

NATIVE = _backend._load_native()  # the extension itself, whichever backend is active
pytestmark = [
    pytest.mark.skipif(NATIVE is None, reason="native extension not built"),
    pytest.mark.filterwarnings("ignore"),
]

GOLDEN = Path(__file__).parent / "golden"
sys.path.insert(0, str(GOLDEN))
import check  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity_audit import _assert_same, _make_frame, same_text  # noqa: E402

_COMPAT = (baselines._PANDAS_EWM_COM1_UPDATE, _PANDAS_31)  # the installed pandas' behaviour


def _reference(args: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with _backend.use("python"), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(args)
        except SystemExit as exc:  # argparse
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


def _native(args: list[str]) -> tuple[int, str, str]:
    code, out, err, fallback, reason = NATIVE.cli_main(args, *_COMPAT)
    assert not fallback, reason
    return code, out, err


@contextlib.contextmanager
def _cwd(path: Path) -> Any:
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@pytest.mark.parametrize("case", check.cases(), ids=lambda c: c["name"])
@pytest.mark.parametrize("runner", ["native", "reference"])
def test_golden(case: dict[str, Any], runner: str) -> None:
    with _cwd(GOLDEN):
        problems = check.compare(case["name"], case["args"], _native if runner == "native" else _reference)
    assert not problems, "\n".join(problems)


def _cli_args(path: str, kwargs: dict[str, Any], fmt: str, fail_on: str) -> list[str]:
    args = [path, "--target", "y", "--task", kwargs["task"], "--pred", "pred", "--format", fmt,
            "--fail-on", fail_on]
    flags = [("--group", "groups"), ("--time", "time"), ("--split", "split"), ("--season", "season"),
             ("--ar-lags", "ar_lags"), ("--horizon", "horizon"), ("--n-boot", "n_boot"), ("--seed", "seed"),
             ("--metric", "metric"), ("--threshold", "threshold"), ("--reported", "reported"),
             ("--block", "block")]
    for flag, key in flags:
        if kwargs.get(key) is not None:
            args += [flag, str(kwargs[key])]
    if kwargs.get("features"):
        args += ["--features", ",".join(kwargs["features"])]
    if kwargs.get("include_ewma"):
        args.append("--ewma")
    return args


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31), fmt=st.sampled_from(["text", "markdown", "json"]),
       fail_on=st.sampled_from(["fail", "warn", "never"]), tsv=st.booleans())
def test_cli_matches_reference(seed: int, fmt: str, fail_on: str, tsv: bool) -> None:
    df, kwargs = _make_frame(seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / ("data.tsv" if tsv else "data.csv"))
        df.to_csv(path, sep="\t" if tsv else ",", index=False)
        args = _cli_args(path, kwargs, fmt, fail_on)
        native, reference = _native(args), _reference(args)
    assert native[0] == reference[0] and native[2] == reference[2], (args, native, reference)
    lapack = kwargs["task"] == "classification" and bool(kwargs["ar_lags"])
    if reference[0] == 2 or not reference[1]:
        assert native[1] == reference[1]
    elif fmt == "json":
        dn, dp = json.loads(native[1]), json.loads(reference[1])
        assert dn["verdict"] == dp["verdict"]
        if not (lapack and "linear AR" in dp["best_baseline"]):
            _assert_same(dn, dp, lapack_ties=lapack)
    elif not lapack:
        assert same_text(native[1], reference[1]), args


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    t = np.arange(n)
    y = np.sin(t / 5) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({
        "t": t, "g": t % 2, "y": np.roll(y, -1), "pred": y + rng.normal(scale=0.1, size=n),
        "split": np.where(t < 140, "train", "test"),
    })


def _underscored(v: float) -> str:
    return f"{v:.4f}".replace("0.", "0_0.")


BASE = ["--task", "regression", "--target", "y", "--pred", "pred", "--group", "g", "--time", "t",
        "--split", "split"]


@pytest.mark.parametrize(
    ("name", "write", "extra"),
    [
        ("missing group key", lambda d, p: d.assign(g=d["g"].where(d.index % 9 > 0)).to_csv(p, index=False),
         []),
        ("bare CR line endings",
         lambda d, p: Path(p).write_bytes(d.to_csv(index=False).replace("\n", "\r").encode()), []),
        ("parquet", None, []),
        ("negative seed", lambda d, p: d.to_csv(p, index=False), ["--seed", "-3"]),
        # Text pandas keeps as strings but Python's float() accepts ("0_0.5" is 0.5).
        ("text target", lambda d, p: d.assign(y=d["y"].map(_underscored)).to_csv(p, index=False), []),
        ("int64 overflow",
         lambda d, p: d.assign(t=[str(2**63 + int(v)) for v in d["t"]]).to_csv(p, index=False), []),
    ],
)
def test_fallbacks_match_reference(name: str, write: Any, extra: list[str], tmp_path: Path) -> None:
    df = _frame()
    if write is None:
        pytest.importorskip("pyarrow")
        path = tmp_path / "data.parquet"
        df.to_parquet(path)
    else:
        path = tmp_path / "data.csv"
        write(df, str(path))
    args = [str(path), *BASE, *extra]
    *_, fallback, reason = NATIVE.cli_main(args, *_COMPAT)
    assert fallback, name
    assert reason
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)  # the console script: falls back to the reference implementation
    assert (code, out.getvalue(), err.getvalue()) == _reference(args)


@pytest.mark.parametrize(
    "extra",
    [
        ["--horizon", "0"],
        ["--metric", "nope"],
        ["--target", "missing"],
        ["--season", "weekly"],
        ["--horizon", "x"],
        ["--task", "forecast"],
        ["--n-boot"],
        ["--bogus"],
        ["--s", "1"],
        ["--wide=yes"],
        ["--threads", "0"],
    ],
)
def test_errors_match_reference(extra: list[str], tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    _frame().to_csv(path, index=False)
    args = [str(path), *BASE, *extra]
    native, reference = _native(args), _reference(args)
    assert native[0] == reference[0] == 2
    # argparse wraps its usage line to the terminal; compare the error itself. (Its "invalid
    # choice" wording quotes the choices before Python 3.12.8 and not after.)
    last_n, last_p = native[2].splitlines()[-1], reference[2].splitlines()[-1]
    if "invalid choice" in last_p:
        last_n, last_p = last_n.split(" (choose from")[0], last_p.split(" (choose from")[0]
    assert last_n == last_p


def test_threads_do_not_change_the_report(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    _frame(2000).to_csv(path, index=False)
    outs = {_native([str(path), *BASE, "--threads", str(k), "--format", "json"])[1] for k in (1, 2, 7)}
    NATIVE.set_num_threads(0)
    assert len(outs) == 1


def test_help_and_version() -> None:
    code, out, _ = _native(["--help"])
    assert code == 0 and out.startswith("usage: predictive-audit") and "--threads" in out
    code, out, _ = _native(["--version"])
    assert code == 0 and out.startswith("predictive-audit ")


@settings(max_examples=2000, deadline=None)
@given(x=st.one_of(st.floats(allow_nan=False, allow_infinity=False),
                   st.integers(0, 2**64 - 1).map(lambda b: float(np.uint64(b).view(np.float64)))))
def test_json_floats_are_python_repr(x: float) -> None:
    if np.isfinite(x):
        assert NATIVE.py_repr(x) == repr(x)
