"""Turn wide tables (one column per series, e.g. one per OD pair) into the long shape audit() expects."""

from __future__ import annotations

import pandas as pd


def wide_to_long(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    time: str | None = None,
    keep: list[str] | None = None,
    series_name: str = "series",
    value_name: str = "value",
) -> pd.DataFrame:
    """Stack ``columns`` (default: every column except ``time`` and ``keep``) into rows.

    Each output row is one (series, time step). Row order within a series follows the
    input row order, which is preserved in a ``step`` column so ``audit(time="step")``
    works even when the table has no usable timestamp. Columns in ``keep`` (such as a
    split label) are repeated for every series.
    """
    keep = list(keep or [])
    fixed = keep + ([time] if time else [])
    columns = list(columns) if columns is not None else [c for c in df.columns if c not in fixed]
    base = df[fixed].copy()
    base["step"] = range(len(df))
    long = (
        pd.concat([base.assign(**{series_name: c, value_name: df[c].to_numpy()}) for c in columns])
        .reset_index(drop=True)
    )
    return long[[series_name, "step", *fixed, value_name]]
