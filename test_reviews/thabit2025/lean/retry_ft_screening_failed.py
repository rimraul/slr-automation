"""
Lean prototype: retry only the 4 items that failed in
thabit2025-ft-pilot-v2 (processing_failed -- fpdf2-generated PDFs lacked an
explicit per-page /MediaBox, now fixed in make_pdfs_from_text.py). Avoids
re-paying for the 16 items that already completed successfully.
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
RUN_ID = "thabit2025-ft-pilot-v2-retry"

FAILED_IDS = {
    "wos-a799cafa17fa9259a377",
    "wos-9140e6832d04f557e051",
    "wos-d02908a43cfd326b63ab",
    "wos-0be9828c8c8318cd9b72",
}


def safe_id(rec_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", rec_id)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        key = get_key("openai")
        if not key:
            print("No OpenAI key found -- aborting.")
            return
        os.environ["OPENAI_API_KEY"] = key

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    items = []
    for rec in manifest:
        rid = rec["rec_id"]
        if rid not in FAILED_IDS:
            continue
        pdf_path = PDF_DIR / f"{safe_id(rid)}.pdf"
        record = Record(id=rid, source="wos", source_id=rid, doi=rec.get("doi"), title=rec["title"])
        items.append(record_to_screening_item(record, pdf_path=pdf_path))

    print(f"Retrying {len(items)} previously-failed items: {[i.document_id for i in items]}")

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
