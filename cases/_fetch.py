"""Download third-party case-study files pinned to an exact commit, verified by SHA-256.

Audited repos mostly carry no license, so their data is never committed here; each case
fetches it into its own gitignored ``.cache/`` on first run.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


def fetch(repo: str, commit: str, path: str, sha256: str | None, cache: Path) -> Path:
    """Return a local copy of ``repo@commit:path``, downloading it if needed."""
    dest = cache / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha256 is not None and digest != sha256:
        dest.unlink()
        raise RuntimeError(f"{path}: checksum mismatch (got {digest})")
    return dest


def fetch_all(repo: str, commit: str, files: dict[str, str | None], cache: Path) -> dict[str, Path]:
    return {p: fetch(repo, commit, p, h, cache) for p, h in files.items()}
