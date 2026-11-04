"""
OA percentages for the 59 known_included (Appendix B) and 261
known_excluded_rest groups, using source-metadata is_open_access from
wos_raw.json (Open Access Designations column) and scopus_raw.json
(openaccess/freetoread API fields) -- joined onto deduplicated_records.json
by id (source-id hash, stable across raw -> dedup).
"""
from __future__ import annotations

import json

from screening_test import build_known_included, build_known_excluded, HERE


def main() -> None:
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    journal_yes = json.loads((HERE / "journal_filtered_records.json").read_text(encoding="utf-8"))
    wos_raw = json.loads((HERE / "wos_raw.json").read_text(encoding="utf-8"))
    scopus_raw = json.loads((HERE / "scopus_raw.json").read_text(encoding="utf-8"))

    oa_by_id = {}
    for r in wos_raw + scopus_raw:
        oa_by_id[r["id"]] = r["is_open_access"]

    known_included, matched_ids = build_known_included(dedup)
    used59 = build_known_excluded(journal_yes, matched_ids, n=len(known_included))
    used_titles = {r["title"] for r in used59}
    pool = [
        r for r in journal_yes
        if r["id"] not in matched_ids and (r.get("abstract") or "").strip()
    ]
    rest261 = [r for r in pool if r["title"] not in used_titles]

    def stats(name: str, records: list) -> None:
        n = len(records)
        oa_true = sum(1 for r in records if oa_by_id.get(r.get("id")) is True)
        oa_false = sum(1 for r in records if oa_by_id.get(r.get("id")) is False)
        oa_unknown = sum(1 for r in records if r.get("id") not in oa_by_id or oa_by_id.get(r.get("id")) is None)
        pct_of_all = 100 * oa_true / n if n else 0
        pct_of_known = 100 * oa_true / (oa_true + oa_false) if (oa_true + oa_false) else 0
        print(f"{name}: n={n}")
        print(f"  OA=yes:     {oa_true:4d}  ({pct_of_all:.1f}% of all, {pct_of_known:.1f}% of known-status)")
        print(f"  OA=no:      {oa_false:4d}")
        print(f"  unknown:    {oa_unknown:4d}")

    print(f"known_included matched via 51 Appendix B fuzzy-match + missing8 (missing8 has no doi/source id -> unknown)")
    stats("known_included (Appendix B, n=59)", known_included)
    print()
    stats("known_excluded_rest (n=261)", rest261)

    out = {
        "known_included": [
            {"id": r.get("id"), "title": r["title"], "is_open_access": oa_by_id.get(r.get("id"))}
            for r in known_included
        ],
        "known_excluded_rest": [
            {"id": r.get("id"), "title": r["title"], "is_open_access": oa_by_id.get(r.get("id"))}
            for r in rest261
        ],
    }
    (HERE / "oa_group_stats.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n-> oa_group_stats.json")


if __name__ == "__main__":
    main()
