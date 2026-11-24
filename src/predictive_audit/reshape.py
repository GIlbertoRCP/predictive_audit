"""Turn wide tables (one column per series, e.g. one per OD pair) into the long shape audit() expects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd


def wide_to_long(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    time: str | None = None,
    keep: Sequence[str] | None = None,
    series_name: str = "series",
    value_name: str = "value",
) -> pd.DataFrame:
    """Stack ``columns`` (default: every column except ``time`` and ``keep``) into rows.

    Each output row is one (series, time step). Row order within a series follows the
    input row order, which is preserved in a ``step`` column so ``audit(time="step")``
    works even when the table has no usable timestamp. Columns in ``keep`` (such as a
    split label) are repeated for every series.

    Vectorized in NumPy to avoid O(n_series) intermediate DataFrame copies while
    preserving exact column-by-column stacking order.
    """
    keep_list = list(keep or [])
    fixed = keep_list + ([time] if time else [])
    cols = list(columns) if columns is not None else [c for c in df.columns if c not in fixed]
    n_rows = len(df)
    n_cols = len(cols)
    if n_cols == 0:
        raise ValueError("no value columns found to reshape")

    # Stack column-by-column: all rows of col 0, then all rows of col 1, ...
    values = df[cols].to_numpy(copy=False).T.reshape(-1)
    data: dict[str, object] = {
        series_name: np.repeat(np.asarray(cols, dtype=object), n_rows),
        "step": np.tile(np.arange(n_rows, dtype=np.int64), n_cols),
    }
    for col in fixed:
        data[col] = np.tile(df[col].to_numpy(copy=False), n_cols)
    data[value_name] = values
    return cast(pd.DataFrame, pd.DataFrame(data))
