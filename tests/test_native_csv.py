"""Phase 4: the native CSV/TSV reader vs pandas.read_csv with default options.

Column names, column types and values must match pandas exactly, floats bit for bit: pandas'
default converter is not correctly rounded (and is compiled with FMA on macOS arm64), so the reader
ports it rather than using strtod. Errors must carry pandas' messages. Inputs the reader leaves
to pandas (NotImplementedError) are skipped.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predictive_audit import _backend
from predictive_audit.cli import _PANDAS_31

NATIVE = _backend._load_native()  # the extension itself, whichever backend is active
pytestmark = pytest.mark.skipif(NATIVE is None, reason="native extension not built")

NA = ["", "NA", "N/A", "n/a", "NaN", "nan", "-NaN", "-nan", "null", "NULL", "None", "<NA>", "#N/A", "#NA",
      "#N/A N/A", "1.#IND", "-1.#IND", "1.#QNAN", "-1.#QNAN"]
NEAR_NA = ["NAN", "na", "Null", " NA", "none", "+nan", "N/a"]
BOOLS = ["True", "False", "TRUE", "false", "tRuE", "FaLsE"]
INFS = ["inf", "-inf", "+inf", "Infinity", "-INFINITY", "INF", "+Infinity", " inf", "infinityy"]


def _digits(draw: Any, lo: int, hi: int) -> str:
    return draw(st.text("0123456789", min_size=lo, max_size=hi))


@st.composite
def numbers(draw: Any) -> str:
    sign = draw(st.sampled_from(["", "", "-", "+"]))
    whole = _digits(draw, 0, 25)
    frac = draw(st.one_of(st.none(), st.just(""), st.text("0123456789", min_size=1, max_size=25)))
    body = whole + ("" if frac is None else "." + frac)
    if not any(c.isdigit() for c in body):
        body += draw(st.sampled_from(["0", "7", "1"]))
    if draw(st.booleans()):
        exp = draw(st.sampled_from([0, 1, 5, 22, 290, 308, 309, 320, 340, 616, 617, 700, 9999]))
        body += draw(st.sampled_from(["e", "E"])) + draw(st.sampled_from(["", "+", "-"])) + str(exp)
    pad_l = draw(st.sampled_from(["", "", "", " ", "\t"]))
    pad_r = draw(st.sampled_from(["", "", "", " "]))
    return pad_l + sign + body + pad_r


@st.composite
def integers(draw: Any) -> str:
    v = draw(st.one_of(st.integers(-1000, 1000), st.integers(-(2**63), 2**63 - 1),
                       st.sampled_from([2**63, -(2**63) - 1, 2**64, 10**25])))
    return draw(st.sampled_from(["", "", "+", " "])) + str(v) if v >= 0 else str(v)


texts = st.text(alphabet=st.sampled_from(list("abxyz ,\"'\n\r\t;é€😀-._0123456789")), min_size=0, max_size=8)


@st.composite
def column(draw: Any, n: int) -> list[str]:
    family = draw(st.sampled_from(["int", "float", "bool", "text", "mixed"]))
    base = {"int": integers(), "float": numbers(), "bool": st.sampled_from(BOOLS), "text": texts}
    cells = []
    for _ in range(n):
        r = draw(st.integers(0, 99))
        if r < 8:
            cells.append(draw(st.sampled_from(NA)))
        elif r < 10:
            cells.append(draw(st.sampled_from(NEAR_NA + INFS)))
        elif family == "mixed":
            cells.append(draw(st.one_of(*base.values())))
        else:
            cells.append(draw(base[family]))
    return cells


def _quote(cell: str, sep: str, force: bool) -> str:
    if force or any(c in cell for c in (sep, '"', "\n", "\r")) or cell.strip(" \t") == "" and cell != "":
        return '"' + cell.replace('"', '""') + '"'
    return cell


@st.composite
def csv_text(draw: Any) -> tuple[str, str]:
    sep = draw(st.sampled_from([",", ",", "\t"]))
    n_cols = draw(st.integers(1, 5))
    n_rows = draw(st.integers(0, 40))
    names = [draw(st.sampled_from(["a", "b", "a.1", "", "x y", "a", "é"])) for _ in range(n_cols)]
    cols = [draw(column(n_rows)) for _ in range(n_cols)]
    eol = draw(st.sampled_from(["\n", "\r\n", "\r"]))
    lines = [sep.join(_quote(c, sep, draw(st.integers(0, 9)) == 0) for c in names)]
    for r in range(n_rows):
        row = [cols[c][r] for c in range(n_cols)]
        if draw(st.integers(0, 19)) == 0:
            row = row[: draw(st.integers(1, n_cols))]  # short row
        if all(c == "" for c in row) and len(row) == 1:
            row = ['""']  # a lone empty field would be a blank line
            lines.append(row[0])
        else:
            lines.append(sep.join(_quote(c, sep, draw(st.integers(0, 9)) == 0) for c in row))
        if draw(st.integers(0, 14)) == 0:
            lines.append(draw(st.sampled_from(["", "  ", "\t"])))  # blank line
    if draw(st.integers(0, 19)) == 0 and n_rows > 1:
        lines.insert(draw(st.integers(2, len(lines))), sep.join(["1"] * (n_cols + 1)))  # ragged row
    text = eol.join(lines) + (eol if draw(st.booleans()) else "")
    if draw(st.integers(0, 9)) == 0:
        text = "﻿" + text
    return text, sep


def _pandas(path: str, sep: str) -> Any:
    try:
        return pd.read_csv(path, sep=sep)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return ("error", str(exc))


def _native(text: str, sep: str) -> Any:
    try:
        return NATIVE.read_csv(text, sep, _PANDAS_31)
    except ValueError as exc:
        return ("error", str(exc))


def _same_floats(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    return bool((nan_a == nan_b).all() and (a[~nan_a].view(np.uint64) == b[~nan_b].view(np.uint64)).all())


def _compare(text: str, sep: str) -> None:
    with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as f:
        f.write(text.encode("utf-8"))
        path = f.name
    try:
        expected = _pandas(path, sep)
        try:
            got = _native(text, sep)
        except NotImplementedError:
            return  # left to pandas on purpose
    finally:
        os.unlink(path)
    if isinstance(expected, tuple) or isinstance(got, tuple):
        assert got == expected, (text, got, expected)
        return
    assert [c[0] for c in got] == list(expected.columns), text
    for name, kind, values in got:
        col = expected[name]
        dtype = str(col.dtype)
        if kind == "int64":
            assert dtype == "int64", (name, dtype, text)
            assert np.array_equal(values, col.to_numpy()), (name, text)
        elif kind == "bool":
            assert dtype == "bool", (name, dtype, text)
            assert list(values) == [int(v) for v in col], (name, text)
        elif kind == "float64":
            if dtype == "object":  # True/False with missing values
                assert all(v is np.nan or isinstance(v, bool) or v != v for v in col), (name, text)
                ref = np.array([float(v) if isinstance(v, bool) else np.nan for v in col])
            else:
                assert dtype == "float64", (name, dtype, text)
                ref = col.to_numpy()
            assert _same_floats(values, ref), (name, list(values), list(ref), text)
        else:
            assert dtype in ("str", "object", "string"), (name, dtype, text)
            ref = [None if (isinstance(v, float) and np.isnan(v)) else v for v in col]
            assert values == ref, (name, values, ref, text)


@settings(max_examples=400, deadline=None)
@given(data=csv_text())
def test_reader_matches_pandas(data: tuple[str, str]) -> None:
    _compare(*data)


@settings(max_examples=300, deadline=None)
@given(cells=st.lists(numbers(), min_size=1, max_size=60))
def test_float_conversion_is_bit_identical(cells: list[str]) -> None:
    _compare("x\n" + "\n".join(cells) + "\n1.5\n", ",")


@pytest.mark.parametrize(
    "text",
    [
        "", "\n\n", "a,b\n", "a,b\n1,2\n3,4,5\n", "\n\na,b\n1,2\n\n3,4,5\n", 'a\n"unterminated\n',
        'a,b\n"x\ny",3\n4,5,6\n', "a,a,a.1\n1,2,3\n", "a,,c\n1,2,3\n", 'a,b\n"ab"cd,2\nx"y,3\n',
        "a,b\r\n1,2\r\n\r\n3,4\r\n", "a,b\r1,2\r3,4", "﻿a,b\n1,2\n", "a\nTrue\nNA\nFalse\n",
        "a\n1e400\n-1e400\n1e-400\n-0e999\n4.9e-324\n2.2250738585072014e-308\n", "a\n1.5E+\n2\n", "a\n1e 5\n",
        "a,b\n,\n1,2\n", "a,b\n1,\n2,\n", 'a\n""\n"NA"\n2\n', "a\n007\n-0\n+5\n", "a;b\n1;2\n",
    ],
)
def test_reader_edge_cases(text: str) -> None:
    _compare(text, ",")
