"""
OA status check for Appendix B's 59 known-included records via Unpaywall
(DOI lookup, free, no auth) -- reuses DOWNLOADING.tier1_auto.resolve_unpaywall
rather than re-querying Scopus, since the original scopus_pull.py never
extracted the openaccessFlag field (view=COMPLETE was used but parse_entry()
only kept doi/title/abstract/year/journal/cited_by/url -- that OA signal was
never persisted, so it can't be recovered from local cached data).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, r"c:/projects")
from DOWNLOADING.tier1_auto import resolve_unpaywall

from screening_test import build_known_included, HERE


def main() -> None:
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    known_included, _ = build_known_included(dedup)

    rows = []
    for i, r in enumerate(known_included, 1):
        doi = (r.get("doi") or "").strip()
        row = {"title": r.get("title"), "doi": doi, "journal": r.get("journal")}
        if not doi:
            row["oa_status"] = "no_doi"
            row["oa_pdf_url"] = ""
        else:
            url = resolve_unpaywall(doi)
            row["oa_status"] = "oa" if url else "not_oa_or_unresolved"
            row["oa_pdf_url"] = url
            time.sleep(0.3)
        rows.append(row)
        safe_title = (row["title"] or "")[:70].encode("ascii", "replace").decode("ascii")
        print(f"[{i}/{len(known_included)}] {row['oa_status']:20s} {safe_title}")

    counts = {}
    for r in rows:
        counts[r["oa_status"]] = counts.get(r["oa_status"], 0) + 1
    print(f"\nn={len(rows)}  {counts}")

    (HERE / "oa_check_appendix_b.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("-> oa_check_appendix_b.json")


if __name__ == "__main__":
    main()
