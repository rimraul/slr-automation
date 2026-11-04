"""Shared helpers for the `continuation` package."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from slr_pipeline.track3_run import Track3ConfigError


def file_sha256(path: Path) -> str:
    """sha256 of a local file's bytes, streamed (same chunking convention as
    `fixture_corpus_digest.py::_sha256_file`) -- used throughout this package
    for copy-integrity checks against a real on-disk PDF, never against a
    bundle's own `payload_manifest.json`-recorded hash (that lookup is
    `Bundle.file_sha256()`/`Bundle.file_sha256_or_none()`, a different thing:
    a manifest lookup, not a from-bytes recomputation)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_and_verify(source_path: Path, dest_path: Path, *, error_context: str) -> None:
    """Idempotent copy-integrity check shared by `preserved_pdfs.py`/
    `retrieved_pdfs.py` (Unit A items 12/15): skip if `dest_path` already
    exists and already verifies against `source_path`; overwrite and
    re-verify if it exists but does not (stale/corrupt, e.g. a partial copy
    left by a killed prior invocation). Raises `Track3ConfigError` naming
    `error_context` on a post-copy hash mismatch -- never a bare
    `assert`."""
    source_sha256 = file_sha256(source_path)
    if dest_path.exists() and file_sha256(dest_path) == source_sha256:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, dest_path)
    if file_sha256(dest_path) != source_sha256:
        raise Track3ConfigError(
            f"{error_context}: copy from {source_path} to {dest_path} failed its own integrity check")
