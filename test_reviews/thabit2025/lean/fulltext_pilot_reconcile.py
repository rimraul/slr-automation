"""
Post-session reconciliation for the Tier 2 Selenium pass -- pattern borrowed
from egpa_test/fulltext_extraction_local_v1.py::parse_selenium_downloads():
don't rely on the live watcher's auto-rename catching every download (it
missed 10/11 here when multiple files landed close together); instead scan
the whole download dir after the fact, parse whatever's there, and match
back to rec_id by title. More robust than depending on real-time rename.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"c:/projects")
from DOWNLOADING.tier1_auto import pdf_to_text, looks_like_stub, safe_id

from screening_test import HERE

PDF_DIR = HERE / "fulltext_pilot" / "pdfs"
TEXT_DIR = HERE / "fulltext_pilot" / "text"

# filename -> rec_id, matched by extracting each PDF's own title text and
# cross-checking against fulltext_pilot/tier2_queue.json (see conversation).
FILE_TO_RECID = {
    "admsci-11-00074.pdf": "wos-93a6e9f90f555e17a764",
    "liddle-2016-aligning-vertical-structures-and-horizontal-relationships-collaborative-leadership-and-accountability.pdf": "wos-b819f01215901882ed2e",
    "New development  Mitigating and negotiating the co-creation of dis value Elinor Ostrom s design principles and co-creating public value.pdf": "wos-7883cf623877d48524cf",
    "Paradoxes of collaborative governance  investigating the real-life dynamics of multi-agency collaborations using a quasi-experimental action-research .pdf": "wos-1e158b40efa0486eb5ef",
    "sustainability-10-04172.pdf": "wos-29070e56a72e19b40428",
    "sustainability-12-06385-v2.pdf": "wos-a159f93cd244636d67d7",
    "sustainability-12-06385-v2 (1).pdf": "wos-a159f93cd244636d67d7",  # duplicate download, same rec_id
    "sustainability-13-04247-v4.pdf": "wos-15de79f98e1e4757cf96",
    "sustainability-13-10292.pdf": "wos-1a29994c4f45a2b78369",
    "sustainability-13-12596-v2.pdf": "wos-2459061c90526eebe243",
    "sustainability-14-11270.pdf": "wos-8d7db893940a795359dd",
}


def main() -> None:
    queue = json.loads((HERE / "fulltext_pilot" / "tier2_queue.json").read_text(encoding="utf-8"))
    queue_by_id = {r["rec_id"]: r for r in queue}

    n_parsed, n_stub, n_skipped_dup = 0, 0, 0
    covered_ids = set()
    for fname, rec_id in FILE_TO_RECID.items():
        pdf_path = PDF_DIR / fname
        if not pdf_path.exists():
            print(f"MISSING FILE: {fname}")
            continue
        text_path = TEXT_DIR / f"{safe_id(rec_id)}.txt"
        if rec_id in covered_ids:
            n_skipped_dup += 1
            continue  # duplicate download of the same paper, first copy wins
        text = pdf_to_text(pdf_path)
        stub_reason = looks_like_stub(text, pdf_path) if text else "empty extraction"
        if stub_reason:
            print(f"STUB/REJECTED: {fname} -- {stub_reason}")
            n_stub += 1
            continue
        text_path.write_text(text, encoding="utf-8")
        renamed_pdf = PDF_DIR / f"{safe_id(rec_id)}.pdf"
        if not renamed_pdf.exists():
            pdf_path.replace(renamed_pdf)
        covered_ids.add(rec_id)
        n_parsed += 1
        title = queue_by_id.get(rec_id, {}).get("title", "?")
        print(f"OK  {rec_id}  ({len(text)} chars)  {title[:60]}")

    print(f"\nReconciled: {n_parsed} parsed, {n_stub} stub/rejected, {n_skipped_dup} duplicate downloads skipped")
    still_missing = [r for r in queue if r["rec_id"] not in covered_ids]
    print(f"Still missing full text: {len(still_missing)}/{len(queue)}")
    for r in still_missing:
        print(f"  {r['rec_id']}  {r['title'][:70]}")


if __name__ == "__main__":
    main()
