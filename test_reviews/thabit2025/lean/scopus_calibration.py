"""
Lean prototype: calibrate a Scopus TITLE-ABS-KEY query against Thabit et al.
2025's own reported Scopus count (564, Table A1), using the SAME
theme1/theme2 term lists and the SAME (PY-cutoff x doctype-restriction x
language) dimension grid as wos_calibration.py, so results are directly
comparable pattern-for-pattern against the WoS sweep.

Count-only: count=1 per combination -> minimal cost, one request each.
Live calls -- run only with explicit per-invocation authorization.
"""
from __future__ import annotations

import requests

TARGET = 564
ENDPOINT = "https://api.elsevier.com/content/search/scopus"

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

BASE = f'TITLE-ABS-KEY(({THEME1}) AND ({THEME2}))'

# Mirrors wos_calibration.py's PY=(1900-2022)/(1900-2021) upper-bound cutoffs:
# WoS PY=(1900-2022) == "up to and including 2022" == Scopus PUBYEAR < 2023.
PY_CUTOFF_TO_PUBYEAR_LT = {
    None: None,
    "1900-2022": 2023,
    "1900-2021": 2022,
}


def _headers(api_key: str, inst_token: str | None) -> dict:
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token
    return headers


def build_query(*, py_cutoff: str | None, dt_article: bool, la_english: bool) -> str:
    q = BASE
    if dt_article:
        q += ' AND SRCTYPE(j)'
    if la_english:
        q += ' AND LANGUAGE(english)'
    pubyear_lt = PY_CUTOFF_TO_PUBYEAR_LT[py_cutoff]
    if pubyear_lt:
        q += f' AND PUBYEAR < {pubyear_lt}'
    return q


# Identical grid to wos_calibration.COMBINATIONS -- dt_article here means
# SRCTYPE(j) (closest Scopus analogue to WoS's DT=Article restriction).
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


def load_env(path: str) -> dict:
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> None:
    env = load_env("../../../.env")
    api_key = env["SCOPUS_API_KEY"]
    inst_token = env.get("SCOPUS_INSTITUTIONAL_KEY")

    results = []
    for combo in COMBINATIONS:
        query = build_query(**combo)
        params = {"query": query, "count": 1, "view": "COMPLETE"}
        resp = requests.get(ENDPOINT, params=params, headers=_headers(api_key, inst_token), timeout=30)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} for combo {combo}: {resp.text[:200]}")
            continue
        total = int(resp.json()["search-results"]["opensearch:totalResults"])
        diff = total - TARGET
        results.append({**combo, "total": total, "diff": diff})
        print(f"py={combo['py_cutoff']!s:12} SRCTYPE(j)={combo['dt_article']!s:5} "
              f"LANGUAGE(english)={combo['la_english']!s:5} -> total={total:6d}  diff={diff:+6d}")

    print(f"\nTarget: {TARGET}")
    results.sort(key=lambda r: abs(r["diff"]))
    print("\nBest candidates (closest to target):")
    for r in results[:5]:
        print(f"  py={r['py_cutoff']!s:12} SRCTYPE(j)={r['dt_article']!s:5} "
              f"LANGUAGE(english)={r['la_english']!s:5} -> total={r['total']:6d}  diff={r['diff']:+6d}")

    import json
    with open("scopus_calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
