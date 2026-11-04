"""
Lean prototype: calibrate a WoS Starter TS=(...) query against Thabit et al.
2025's own reported WoS count (682, Table A1) -- same combinatorial-sweep
approach used for the Scopus calibration this session (SRCTYPE/LANGUAGE/
PUBYEAR dimensions), adapted to WoS Advanced Search field tags.

Count-only: max_results=1 per combination -> one API request each (metadata.total
comes back on page 1 regardless of limit), safe under any Starter API rate tier.

Live calls -- run only with explicit per-invocation authorization.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"c:/projects")

from SEARCHING.adapters.wos_starter import search_wos_starter

TARGET = 682

THEME1 = (
    '"cross-sector* partnership*" OR "cross-sector* project*" OR '
    '"cross-sector* collaboration*" OR "multi-sector* partnership*" OR '
    '"multi-sector* project*" OR "multi-sector* collaboration*" OR '
    '"multisector* partnership*" OR "multisector* project*" OR '
    '"multisector* collaboration*" OR "collaborative* governance*" OR '
    '"multi-agenc*" OR "multi-actor*" OR "multiactor*" OR "co-creation" OR '
    '"cocreation" OR "co-production" OR "coproduction"'
)

THEME2 = (
    '"public value*" OR "public interest*" OR "public benefit*" OR '
    '"public good*" OR "common interest*" OR "common good*" OR '
    '"collective value*" OR "collective interest*" OR "collective good*" OR '
    '"societ* value*" OR "societ* goal*" OR "development goal*" OR '
    '"public* sphere*" OR "democra* value*" OR "democra* interest*" OR '
    '"democratic* accountabilit*" OR "project* outcome*" OR "project* result*"'
)

BASE = f'TS=({THEME1}) AND TS=({THEME2})'


def build_query(*, py_cutoff: str | None, dt_article: bool, la_english: bool) -> str:
    q = BASE
    if dt_article:
        q += ' AND DT=Article'
    if la_english:
        q += ' AND LA=English'
    if py_cutoff:
        q += f' AND PY=({py_cutoff})'
    return q


COMBINATIONS = [
    {"py_cutoff": None,         "dt_article": False, "la_english": False},
    {"py_cutoff": "1900-2022",  "dt_article": False, "la_english": False},
    {"py_cutoff": "1900-2021",  "dt_article": False, "la_english": False},
    {"py_cutoff": None,         "dt_article": True,  "la_english": False},
    {"py_cutoff": "1900-2022",  "dt_article": True,  "la_english": False},
    {"py_cutoff": "1900-2021",  "dt_article": True,  "la_english": False},
    {"py_cutoff": None,         "dt_article": False, "la_english": True},
    {"py_cutoff": "1900-2022",  "dt_article": False, "la_english": True},
    {"py_cutoff": "1900-2021",  "dt_article": False, "la_english": True},
    {"py_cutoff": None,         "dt_article": True,  "la_english": True},
    {"py_cutoff": "1900-2022",  "dt_article": True,  "la_english": True},
    {"py_cutoff": "1900-2021",  "dt_article": True,  "la_english": True},
]


def main() -> None:
    results = []
    for combo in COMBINATIONS:
        query = build_query(**combo)
        _, total = search_wos_starter(query, max_results=1)
        diff = total - TARGET
        results.append({**combo, "total": total, "diff": diff})
        print(f"py={combo['py_cutoff']!s:12} DT=Article={combo['dt_article']!s:5} "
              f"LA=English={combo['la_english']!s:5} -> total={total:6d}  diff={diff:+6d}")

    print(f"\nTarget: {TARGET}")
    results.sort(key=lambda r: abs(r["diff"]))
    print("\nBest candidates (closest to target):")
    for r in results[:5]:
        print(f"  py={r['py_cutoff']!s:12} DT=Article={r['dt_article']!s:5} "
              f"LA=English={r['la_english']!s:5} -> total={r['total']:6d}  diff={r['diff']:+6d}")


if __name__ == "__main__":
    main()
