"""Unit A item 12 -- preserved-included PDF physical-acquisition mechanism.

`--parent-workspace` is the ONLY available physical source for a
preserved-included record's actual PDF bytes (the signed parent release
bundle never stages them for a `track3_live`-mode payload -- confirmed:
`finalize_bundle()`'s PDF-staging code, `reference_bundle.py:1193`, only
stages PDFs for `mode_profile="track2_offline"`'s fixtures corpus).
`extraction_node()` always re-parses the PDF fresh from an existing local
file, so a physical copy into the continuation's own `pdf_dir` is required
before extraction can run -- this is a file re-acquisition, not a
re-screening or a new eligibility decision; no ledger write happens here."""
from __future__ import annotations

from pathlib import Path
from typing import List

from slr_pipeline.continuation._util import copy_and_verify
from slr_pipeline.nodes.reporting import ReportingSourceSnapshot
from slr_pipeline.state import Record
from slr_pipeline.track3_run import Track3ConfigError


def acquire_preserved_included_pdfs(
    preserved_included_ids: List[str],
    parent_snapshot: ReportingSourceSnapshot,
    parent_workspace: Path,
    pdf_dir: Path,
) -> List[Record]:
    if not preserved_included_ids:
        return []

    by_id = {r.id: r for r in parent_snapshot.full_text_records}
    pdf_dir = Path(pdf_dir)
    out: List[Record] = []
    for record_id in preserved_included_ids:
        record = by_id.get(record_id)
        if record is None:
            raise Track3ConfigError(
                f"preserved-included record {record_id!r}: no matching entry in "
                "parent_snapshot.full_text_records")
        if not record.full_text_path:
            raise Track3ConfigError(
                f"preserved-included record {record_id!r}: full_text_records entry has no full_text_path")

        source_path = Path(parent_workspace) / "pdfs" / Path(record.full_text_path).name
        if not source_path.is_file():
            raise Track3ConfigError(
                f"preserved-included record {record_id!r}: expected PDF not found at {source_path}")

        dest_path = pdf_dir / Path(record.full_text_path).name
        copy_and_verify(source_path, dest_path, error_context=f"preserved-included record {record_id!r}")

        out.append(record.model_copy(update={"full_text_path": str(dest_path)}))
    return out
