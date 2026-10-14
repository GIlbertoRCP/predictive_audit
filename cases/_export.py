"""Write small JSON excerpts of case results for the explorer page (site/)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SITE_DATA = Path(__file__).resolve().parent.parent / "site" / "data"


def _clean(x, digits):
    if isinstance(x, dict):
        return {k: _clean(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v, digits) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else float(f"{float(x):.{digits}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def write(name: str, payload: dict, digits: int = 5) -> Path:
    """Floats are kept to ``digits`` significant figures."""
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    path = SITE_DATA / f"{name}.json"
    path.write_text(json.dumps(_clean(payload, digits), separators=(",", ":")))
    return path
