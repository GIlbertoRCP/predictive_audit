"""Download third-party case-study files pinned to an exact commit, verified by SHA-256.

Audited repos mostly carry no license, so their data is never committed here; each case
fetches it into its own gitignored ``.cache/`` on first run with retries and atomic writes.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    repo: str,
    commit: str,
    path: str,
    sha256: str | None,
    cache: Path,
    *,
    retries: int = 3,
    timeout: int = 120,
) -> Path:
    """Return a local copy of ``repo@commit:path``, downloading atomically if needed."""
    dest = cache / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    tmp.write_bytes(r.read())
                digest = _sha256(tmp)
                if sha256 is not None and digest != sha256:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"{path}: checksum mismatch (got {digest})")
                os.replace(tmp, dest)
                last_err = None
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                tmp.unlink(missing_ok=True)
                last_err = exc
                if attempt + 1 < retries:
                    time.sleep(2.0 ** attempt)
        if last_err is not None:
            raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last_err}") from last_err

    digest = _sha256(dest)
    if sha256 is not None and digest != sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: checksum mismatch (got {digest})")
    return dest


def fetch_all(repo: str, commit: str, files: dict[str, str | None], cache: Path) -> dict[str, Path]:
    return {p: fetch(repo, commit, p, h, cache) for p, h in files.items()}
