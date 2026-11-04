"""
Lean prototype: wire ft_criteria.thabit_criteria_set() into the 20-record
full-text OA pilot manifest and run real FT_SCREENING (docling parser +
lexical_bm25 retriever + gpt-5.6-terra adjudicator via
slr_pipeline.adapters.ft_screening_adapter.chapter_screening_config()).

LIVE PAID CALL -- one OpenAI request per (item, criterion) pair, minus
cache hits. 3 criteria x up to 16 items = up to 48 provider calls.

Only items with a REAL PDF file are screened -- FT_SCREENING's docling
parser needs an actual PDF. 4/20 pilot records had no PDF on disk
(Elsevier Article Retrieval API returns XML text directly, one record was
recovered from a .docx deposit): wos-a799cafa17fa9259a377,
wos-9140e6832d04f557e051, wos-d02908a43cfd326b63ab,
wos-0be9828c8c8318cd9b72 -- see make_pdfs_from_text.py, which renders
their already-extracted .txt full text into plain PDFs so all 20/20 are
screenable. Any record still missing a PDF is skipped and reported.

tier1_manifest.json's own "status"/"pdf_source"/"text_chars" fields are
STALE for Tier 2/3 records (fulltext_pilot_reconcile.py and the manual
tier3 pass wrote text/pdf files directly without updating the manifest) --
this script re-derives real status from what's actually on disk in
fulltext_pilot/text/ and fulltext_pilot/pdfs/, not from those fields.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, r"c:/projects")
from SEARCHING.keys import get_key

from ft_criteria import thabit_criteria_set

from slr_pipeline.adapters.ft_screening_adapter import (
    build_full_text_screening_request,
    chapter_screening_config,
    record_to_screening_item,
)
from slr_pipeline.state import Record

from ft_screening import run_ft_screening
from ft_screening.errors import ComponentResolutionError

HERE = Path(__file__).parent
PILOT_DIR = HERE / "fulltext_pilot"
MANIFEST_PATH = PILOT_DIR / "tier1_manifest.json"
PDF_DIR = PILOT_DIR / "pdfs"
OUTPUT_DIR = PILOT_DIR / "ft_screening_run"
RUN_ID = "thabit2025-ft-pilot-v2"


def safe_id(rec_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", rec_id)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        key = get_key("openai")
        if not key:
            print("No OpenAI key found (SEARCHING.keys / OPENAI_API_KEY env) -- aborting.")
            return
        os.environ["OPENAI_API_KEY"] = key

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    items = []
    skipped_no_pdf = []
    for rec in manifest:
        rid = rec["rec_id"]
        pdf_path = PDF_DIR / f"{safe_id(rid)}.pdf"
        if not pdf_path.exists():
            skipped_no_pdf.append(rid)
            continue
        record = Record(id=rid, source="wos", source_id=rid, doi=rec.get("doi"), title=rec["title"])
        items.append(record_to_screening_item(record, pdf_path=pdf_path))

    print(f"Manifest: {len(manifest)} records -> {len(items)} with a real PDF, "
          f"{len(skipped_no_pdf)} skipped (no PDF, not screenable by docling):")
    for rid in skipped_no_pdf:
        print(f"  skip  {rid}")

    criteria_set = thabit_criteria_set()
    config = chapter_screening_config()

    request = build_full_text_screening_request(
        items=items, criteria_set=criteria_set, config=config,
        output_dir=OUTPUT_DIR, run_id=RUN_ID,
    )

    def on_progress(event) -> None:
        print(f"  [{event.event}] {event.completed_items}/{event.total_items} "
              f"{event.document_id or ''} {event.message or ''}")

    try:
        result = run_ft_screening(request, progress_cb=on_progress)
    except ComponentResolutionError as exc:
        print(f"ComponentResolutionError: {exc}")
        return

    group_by_id = {r["rec_id"]: r.get("group", "?") for r in manifest}
    print(f"\nRun {result.run_id}  status={result.run_status}")
    print(f"  total={result.summary.total} completed={result.summary.completed} "
          f"included={result.summary.included} excluded={result.summary.excluded} "
          f"needs_human_review={result.summary.needs_human_review} "
          f"cache_hits={result.summary.cache_hits}")
    print(f"\n{'rec_id':30s} {'group':16s} {'decision':20s} exclusion_reason_codes")
    for item in result.items:
        if item.eligibility is None:
            print(f"{item.document_id:30s} {group_by_id.get(item.document_id, '?'):16s} "
                  f"{item.execution_status}")
            continue
        elig = item.eligibility
        print(f"{item.document_id:30s} {group_by_id.get(item.document_id, '?'):16s} "
              f"{elig.final_decision:20s} {elig.exclusion_reason_codes}")

    print(f"\nArtifacts:\n  {result.artifacts.run_result_json}\n"
          f"  {result.artifacts.item_results_jsonl}\n  {result.artifacts.ledger_jsonl}")


if __name__ == "__main__":
    main()
