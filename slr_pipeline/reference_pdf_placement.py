"""Shared §6.3 PDF pre-positioning: the hash-verified copy of the 6
fixture-corpus PDFs whose record legitimately resolves to a
`Record.source`/`Record.source_id` pair (see `PDF_PLACEMENT`'s own table)
from `tests/reference_run/fixtures/corpus/pdfs/<record_id>.pdf` into a
review workspace's `pdfs/<source>_<source_id>.pdf` -- the exact filename
`fulltext.py`'s `_find_local_pdf`/`_dest_path` look for
(`f"{source}_{source_id}.pdf"`).

Used by two callers with different purposes: `scripts/author_reference_
cassette.py` (dev-only, step 12(d) -- bootstraps the FIXED cassette-
generation workspace before any cassette exists) and
`slr_pipeline.reference_run_child.py` (shipped, step 13 -- pre-positions
PDFs the same way before replaying the real reference run against that
frozen cassette, so the same `pdf_path` strings -- and therefore the same
`extraction_run_id`/`evidence_id`/request signatures -- reproduce).
Extracted here (not duplicated) because the 6-row table is a single,
hash-critical source of truth -- `author_reference_cassette.py`'s own
`FIXED_WORKSPACE` comment records a prior reproducibility incident (a
46/50-vs-4/50 cassette-entry diff) caused by exactly this kind of value
drifting between call sites.

`source_dir` is an explicit parameter, not a module-level constant, so
this module makes no assumption about a repository checkout being present
-- `reference_run_child.py` may be running from an installed wheel with a
parent-supplied absolute path and no `tests/` tree anywhere near it."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from slr_pipeline.extraction.hashing import file_sha256

# (record_id, fixture source `fx_<db>`, native_key) -- source/source_id become
# `Record.source`/`Record.source_id` (`fixture_search_adapter.py::_record_from_
# native`: `source=f"fx_{db}"`, `source_id=native_key`), which `fulltext.py`'s
# `_find_local_pdf`/`_dest_path` use verbatim as `f"{source}_{source_id}.pdf"`.
# native_key verified directly against the committed `native/fx_<db>.json`
# files (all follow `NK-<record_id>`) -- this is the same table
# `author_reference_cassette.py` used to author the frozen cassette; drifting
# it here would silently break replay signature matching.
PDF_PLACEMENT: Tuple[Tuple[str, str, str], ...] = (
    ("fx-0002", "fx_web_of_science", "NK-fx-0002"),
    ("fx-0005", "fx_scopus", "NK-fx-0005"),
    ("fx-0009", "fx_pubmed", "NK-fx-0009"),
    ("fx-0013", "fx_cinahl", "NK-fx-0013"),
    ("fx-0018", "fx_engineering_village", "NK-fx-0018"),
    ("fx-0024", "fx_acm", "NK-fx-0024"),
)


# Relative (from repo root) path segments of the FIXED workspace whose
# resolved absolute string `author_reference_cassette.py` (step 12(d)) baked
# into the frozen cassette's synthesis_map/synthesis_refine request
# signatures, via extraction_run_id -> evidence_id -> answer_id (see
# `extraction/run_identity.py::compute_run_id()`'s own docstring: it hashes
# `pdf_path` verbatim). `reference_run_child.py` (step 13) MUST resolve its
# own `cfg.output_dir` to this exact same absolute path -- not merely a
# self-consistent path of its own choosing -- or its own evidence_id strings
# will not byte-match what the cassette's synthesis_refine entry already has
# recorded, and replay will latch-fatal on "unrecorded signature". A single
# tuple of parts (not a `Path`) so each caller resolves it against its own
# `REPO_ROOT`, keeping this module itself checkout-agnostic.
REFERENCE_EXTRACTION_WORKSPACE_PARTS: Tuple[str, ...] = (
    "tests", "reference_run", ".cassette_generation_workspace",
)


def place_pdfs(pdf_dir: Path, oracle: Dict[str, Any], source_dir: Path) -> None:
    """Copies each `PDF_PLACEMENT` row's PDF from `source_dir/<record_id>.pdf`
    into `pdf_dir/<source>_<source_id>.pdf`, verifying the source's, then the
    just-written destination's, sha256 against `oracle["pdf_sha256"]
    [<record_id>]` -- both checks happen here, before any caller is allowed
    to proceed to a graph invocation, per §6.3."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for record_id, source, source_id in PDF_PLACEMENT:
        src = source_dir / f"{record_id}.pdf"
        dest = pdf_dir / f"{source}_{source_id}.pdf"
        expected_sha = oracle["pdf_sha256"][record_id]
        actual_src_sha = file_sha256(src)
        if actual_src_sha != expected_sha:
            raise RuntimeError(f"pdf {src} sha256 mismatch: expected {expected_sha}, got {actual_src_sha}")
        dest.write_bytes(src.read_bytes())
        actual_dest_sha = file_sha256(dest)
        if actual_dest_sha != expected_sha:
            raise RuntimeError(
                f"pdf {dest} sha256 mismatch after copy: expected {expected_sha}, got {actual_dest_sha}"
            )
