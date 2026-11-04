"""
Chapter demonstration pilot: 10 known_included (Appendix B) OA records +
10 known_excluded_rest OA records, randomly sampled, run through
DOWNLOADING's three-tier retrieval (Tier 1 auto now; Tier 2 Selenium queue
+ Tier 3 manual xlsx produced for whatever Tier 1 misses).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, r"c:/projects")
from DOWNLOADING.tier1_auto import run_tier1
from DOWNLOADING.tier3_manual_report import build_manual_report

from screening_test import build_known_included, build_known_excluded, HERE

SEED = 42
N_PER_GROUP = 10

OUT_DIR = HERE / "fulltext_pilot"
PDF_DIR = OUT_DIR / "pdfs"
TEXT_DIR = OUT_DIR / "text"


def main() -> None:
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    journal_yes = json.loads((HERE / "journal_filtered_records.json").read_text(encoding="utf-8"))
    wos_raw = json.loads((HERE / "wos_raw.json").read_text(encoding="utf-8"))
    scopus_raw = json.loads((HERE / "scopus_raw.json").read_text(encoding="utf-8"))
    oa_by_id = {r["id"]: r["is_open_access"] for r in wos_raw + scopus_raw}

    known_included, matched_ids = build_known_included(dedup)
    used59 = build_known_excluded(journal_yes, matched_ids, n=len(known_included))
    used_titles = {r["title"] for r in used59}
    pool = [
        r for r in journal_yes
        if r["id"] not in matched_ids and (r.get("abstract") or "").strip()
    ]
    rest261 = [r for r in pool if r["title"] not in used_titles]

    included_oa = [r for r in known_included if oa_by_id.get(r.get("id")) is True]
    excluded_oa = [r for r in rest261 if oa_by_id.get(r.get("id")) is True]
    print(f"known_included OA pool: n={len(included_oa)}  known_excluded_rest OA pool: n={len(excluded_oa)}")

    rng = random.Random(SEED)
    sample_included = rng.sample(included_oa, N_PER_GROUP)
    rng2 = random.Random(SEED)
    sample_excluded = rng2.sample(excluded_oa, N_PER_GROUP)

    for group_name, sample in (("known_included", sample_included), ("known_excluded", sample_excluded)):
        (HERE / f"fulltext_pilot_sample_{group_name}.json").write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"-> fulltext_pilot_sample_{group_name}.json ({len(sample)} records)")

    all_items = []
    for group_name, sample in (("known_included", sample_included), ("known_excluded", sample_excluded)):
        for r in sample:
            all_items.append({"rec_id": r["id"], "title": r["title"], "doi": r.get("doi") or "",
                               "group": group_name})

    manifest = run_tier1(all_items, pdf_dir=PDF_DIR, text_dir=TEXT_DIR)
    for m, it in zip(manifest, all_items):
        m["group"] = it["group"]

    (OUT_DIR / "tier1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    counts = {}
    for m in manifest:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    print(f"\nTier 1 results (n={len(manifest)}): {counts}")

    leftover = [m for m in manifest if m["status"] not in ("parsed",)]
    if leftover:
        manual_items = [
            {"rec_id": m["rec_id"], "title": m["title"], "doi": m["doi"],
             "priority": m["group"], "reason": m["status"]}
            for m in leftover
        ]
        report_counts = build_manual_report(
            manual_items,
            out_xlsx=OUT_DIR / "tier3_manual_worklist.xlsx",
            save_folder=str(PDF_DIR) + "\\",
        )
        print(f"-> tier3_manual_worklist.xlsx ({len(leftover)} records, by group: {report_counts})")

        selenium_queue = [{"rec_id": m["rec_id"], "title": m["title"], "doi": m["doi"]} for m in leftover]
        (OUT_DIR / "tier2_queue.json").write_text(
            json.dumps(selenium_queue, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"-> tier2_queue.json ({len(selenium_queue)} records, for optional Selenium run)")
    else:
        print("All 20 retrieved by Tier 1 -- no Tier 2/3 needed.")


if __name__ == "__main__":
    main()
