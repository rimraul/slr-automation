"""Canonical hashing for run-identity binding -- deliberately self-contained
(does not import FT_SCREENING's own `ft_screening.hashing`, per this
package's own "do not import FT_SCREENING internals" boundary; the two
projects' hashes are not meant to be interchangeable, just similarly
disciplined). NFC-normalized, sorted-key JSON, matching the NFC-normalize-
then-hash convention already established in `slr_pipeline.nodes.fulltext`'s
own `_stable_run_id()`.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """UTF-8, NFC-normalized strings, sorted object keys, compact
    separators -- the same input always produces the same bytes,
    independent of dict insertion order or incidental Unicode
    normalization differences in the source data."""

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, str):
            return unicodedata.normalize("NFC", obj)
        if isinstance(obj, dict):
            return {str(k): _normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_normalize(v) for v in obj]
        return obj

    return json.dumps(
        _normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Any, *, chunk_size: int = 1 << 20) -> str:
    """Streamed (1 MiB chunks) -- never reads a whole PDF into memory just
    to hash it."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
