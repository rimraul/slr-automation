"""Unit A item 15 -- retrieved-PDF physical-acquisition mechanism, mirroring
`preserved_pdfs.py`'s own design. `dest_path` is computed via a FRESH call
to `nodes/fulltext.py`'s own `_dest_path()` (not the parent's already-
recorded basename, since a `retrieved_ids` record has no such prior
`full_text_path` to reuse -- it is newly retrieved this run). This matters
because `_find_local_pdf()` (called later, inside
`acquire_and_confirm_records()`) does NOT call `_dest_path()` -- the two
are independently-duplicated inline naming formulas that happen to produce
identical output for the same record; what guarantees the copy is found is
that both formulas are kept in lockstep, not that one calls the other."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from slr_pipeline.continuation._util import copy_and_verify, file_sha256
from slr_pipeline.continuation.manifest import ContinuationManifest
from slr_pipeline.nodes.fulltext import _dest_path
from slr_pipeline.state import Record
from slr_pipeline.track3_run import Track3ConfigError


def acquire_retrieved_pdfs(
    retrieved_ids: List[str],
    included_records: List[Record],
    continuation_manifest: ContinuationManifest,
    pdfs_dir: Path,
    pdf_dir: Path,
) -> Dict[str, Dict[str, str]]:
    if not retrieved_ids:
        return {}

    docs_by_worklist_id = {d.worklist_id: d for d in continuation_manifest.documents}
    records_by_id = {r.id: r for r in included_records}
    pdfs_dir = Path(pdfs_dir)
    pdf_dir = Path(pdf_dir)

    out: Dict[str, Dict[str, str]] = {}
    for record_id in retrieved_ids:
        doc = docs_by_worklist_id.get(record_id)
        if doc is None:
            raise Track3ConfigError(
                f"retrieved record {record_id!r}: no matching worklist_id entry in continuation_manifest")
        record = records_by_id.get(record_id)
        if record is None:
            raise Track3ConfigError(
                f"retrieved record {record_id!r}: no matching entry in included_records")

        source_path = pdfs_dir / doc.save_as
        if not source_path.is_file():
            raise Track3ConfigError(f"retrieved record {record_id!r}: expected PDF not found at {source_path}")

        # Authenticity: does the file CURRENTLY at source_path still match
        # what the manifest recorded at manifest-creation time? Checked
        # BEFORE the copy-fidelity check below, and before any cost
        # estimate/authorization -- v39/round-36 finding 5.
        if file_sha256(source_path) != doc.file_sha256:
            raise Track3ConfigError(
                f"retrieved record {record_id!r}: {source_path} no longer matches the manifest's recorded "
                f"file_sha256={doc.file_sha256!r} -- the source file may have been replaced since the "
                "manifest was created")

        dest_path = _dest_path(record, pdf_dir)
        copy_and_verify(source_path, dest_path, error_context=f"retrieved record {record_id!r}")

        out[record_id] = {"dest_path": str(dest_path), "file_sha256": file_sha256(dest_path)}
    return out
