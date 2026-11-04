"""Extract marked code listings from the chapter's solution files.

Marker syntax (flat only, one complete function/class per region):

    # --- listing:<id> start ---
    def foo(...):
        ...
    # --- listing:<id> end ---

Used later by the manuscript-rendering pipeline; the committed
``examples/chapter/listings_manifest.json`` plus
``tests/chapter_examples/test_extract_listings.py`` keep every printed listing
in lock-step with the tested source.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

_START = re.compile(r"^\s*#\s*---\s*listing:([A-Za-z0-9_-]+)\s+start\s*---\s*$")
_END = re.compile(r"^\s*#\s*---\s*listing:([A-Za-z0-9_-]+)\s+end\s*---\s*$")


def _scan(source_path: Path) -> dict[str, tuple[int, int]]:
    """Return {listing_id: (start_line_idx, end_line_idx)} for the body lines
    between (exclusive of) the markers. Raises ValueError on any malformed
    marker structure, naming the file and line."""
    lines = source_path.read_text(encoding="utf-8").splitlines()
    spans: dict[str, tuple[int, int]] = {}
    open_id: str | None = None
    open_line = 0
    for lineno, line in enumerate(lines, start=1):
        start = _START.match(line)
        end = _END.match(line)
        if start:
            if open_id is not None:
                raise ValueError(f"{source_path}:{lineno}: nested listing marker (still inside '{open_id}')")
            open_id = start.group(1)
            open_line = lineno
            if open_id in spans:
                raise ValueError(f"{source_path}:{lineno}: duplicate listing id '{open_id}' in file")
        elif end:
            if open_id is None:
                raise ValueError(f"{source_path}:{lineno}: listing end marker with no open start")
            if end.group(1) != open_id:
                raise ValueError(
                    f"{source_path}:{lineno}: listing end '{end.group(1)}' does not match open '{open_id}'"
                )
            spans[open_id] = (open_line, lineno)  # 1-indexed marker lines
            open_id = None
    if open_id is not None:
        raise ValueError(f"{source_path}:{open_line}: listing '{open_id}' is never closed")
    return spans


def list_listings(source_path: str | Path) -> list[str]:
    return list(_scan(Path(source_path)).keys())


def extract_listing(source_path: str | Path, listing_id: str) -> str:
    """Return the source between the ``listing:<id>`` markers, with common
    leading whitespace removed and exactly one trailing newline. That strip is
    the ONLY transformation applied."""
    path = Path(source_path)
    spans = _scan(path)
    if listing_id not in spans:
        raise KeyError(listing_id)
    start_marker, end_marker = spans[listing_id]
    lines = path.read_text(encoding="utf-8").splitlines()
    body = lines[start_marker:end_marker - 1]  # between the markers, exclusive
    return textwrap.dedent("\n".join(body)).rstrip("\n") + "\n"


def extract_all(manifest_path: str | Path) -> dict[str, str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    repo_root = Path(manifest_path).resolve().parents[2]
    out: dict[str, str] = {}
    for row in manifest["listings"]:
        out[row["id"]] = extract_listing(repo_root / row["source"], row["id"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="examples/chapter/listings_manifest.json")
    parser.add_argument("--id", help="print just this listing id")
    parser.add_argument("--source", help="required with --id")
    args = parser.parse_args()

    if args.id:
        if not args.source:
            parser.error("--source is required with --id")
        print(extract_listing(args.source, args.id), end="")
        return
    for listing_id, text in extract_all(args.manifest).items():
        print(f"# ===== {listing_id} =====")
        print(text)


if __name__ == "__main__":
    main()
