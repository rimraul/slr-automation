"""Verify one file's sha256 against one named field in one `track2/*.lock`
file.

track2_pipeline_readiness_plan_2026-08-30.md §5.2 item 5 / §5.3 / §5.5: the
one shared, single-purpose re-verification step every asset download (the
FT_SCREENING wheel, the Docling archive, and -- at step 19 -- the
`slr-pipeline` release wheel) runs immediately after fetching the asset,
before installing/extracting it. Every `track2/*.lock` file this repo
writes is the same tiny Python-literal-assignment format `build_slr_
pipeline_wheel.py::_write_lock` established (`name = 'value'` lines,
`exec`'d into a namespace dict) -- this script is generic over that
format and the field name, so `ft-screening-wheel.lock`'s `sha256` and
`DOCLING_ASSETS.lock`'s `archive_sha256` both verify through the same
code path.

Usage:
    python scripts/verify_asset_hash.py <file> <lock_file> --field sha256

Exits non-zero (and prints a clear mismatch) on any hash mismatch, missing
file, or missing lock field -- never silently continues.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(lock_path: Path) -> dict:
    if not lock_path.exists():
        raise FileNotFoundError(f"lock file does not exist: {lock_path}")
    namespace: dict = {}
    exec(lock_path.read_text(encoding="utf-8"), {}, namespace)
    return namespace


def verify_asset_hash(file_path: Path, lock_path: Path, *, field: str) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"asset file does not exist: {file_path}")
    lock_ns = load_lock(lock_path)
    if field not in lock_ns:
        raise KeyError(f"{lock_path} has no field {field!r} (has: {sorted(lock_ns)})")
    expected = lock_ns[field]
    actual = sha256_of(file_path)
    if actual != expected:
        raise ValueError(
            f"sha256 mismatch for {file_path}:\n"
            f"  expected ({field} in {lock_path.name}): {expected}\n"
            f"  actual:                                 {actual}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="the downloaded asset to verify")
    parser.add_argument("lock_file", type=Path, help="the track2/*.lock file recording the expected hash")
    parser.add_argument("--field", default="sha256", help="the lock field holding the expected sha256 (default: sha256)")
    args = parser.parse_args(argv)

    verify_asset_hash(args.file, args.lock_file, field=args.field)
    print(f"OK: {args.file} matches {args.lock_file.name}:{args.field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
