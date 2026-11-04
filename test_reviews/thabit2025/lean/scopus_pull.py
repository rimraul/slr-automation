"""
Lean prototype: pull the full 674-record Scopus set for the calibrated
best-pair query (PUBYEAR<2023, SRCTYPE(j), LANGUAGE(english)) -- Record-shaped
JSON, ready to append into raw_records.json alongside wos_raw.json before
dedup_node.

Field mapping copied from slr_pipeline/search_backends/scopus.py::_parse_entry
(the reviewed, production mapping) but without that module's checkpoint/
retry/SearchBackendRequest scaffolding -- lean cut: single run, no resume,
no rate-limit backoff beyond the standard 1.1s delay.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from scopus_calibration import BASE, _headers, load_env

ENDPOINT = "https://api.elsevier.com/content/search/scopus"
PER_PAGE = 25
DELAY_S = 1.1

QUERY = BASE + " AND SRCTYPE(j) AND LANGUAGE(english) AND PUBYEAR < 2023"


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def year_from_cover_date(cover_date: Optional[str]) -> Optional[int]:
    if not cover_date:
        return None
    try:
        return int(cover_date[:4])
    except ValueError:
        return None


def parse_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    eid = str(entry.get("eid"))

    authors: List[str] = []
    for au in as_list(entry.get("author")):
        name = au.get("authname") or au.get("ce:indexed-name", "")
        if name:
            authors.append(name)
    if not authors:
        first = entry.get("dc:creator", "")
        if first:
            authors.append(first)

    keywords: List[str] = []
    auth_kw_raw = entry.get("authkeywords") or ""
    if auth_kw_raw:
        keywords = [k.strip() for k in auth_kw_raw.split("|") if k.strip()]

    scopus_url = None
    for link in as_list(entry.get("link")):
        if link.get("@ref") == "scopus":
            scopus_url = link.get("@href")
            break

    import hashlib
    digest = hashlib.sha256(eid.encode("utf-8")).hexdigest()[:20]

    is_oa = entry.get("openaccess") in ("1", 1, True)
    oa_flag = entry.get("freetoread") or entry.get("openaccessFlag")

    return {
        "id": f"scopus-{digest}",
        "source": "scopus",
        "source_id": eid,
        "doi": entry.get("prism:doi"),
        "title": (entry.get("dc:title") or "").strip(),
        "abstract": (entry.get("dc:description") or "").strip() or None,
        "authors": authors,
        "year": year_from_cover_date(entry.get("prism:coverDate")),
        "journal": entry.get("prism:publicationName"),
        "keywords": keywords,
        "cited_by": int(entry["citedby-count"]) if entry.get("citedby-count") not in (None, "") else None,
        "url": scopus_url or entry.get("prism:url"),
        "is_open_access": is_oa,
        "open_access_flag": oa_flag,
    }


def main() -> None:
    env = load_env("../../../.env")
    api_key = env["SCOPUS_API_KEY"]
    inst_token = env.get("SCOPUS_INSTITUTIONAL_KEY")

    records: List[Dict[str, Any]] = []
    cursor = "*"
    page = 0
    total_hits = None

    while True:
        params = {"query": QUERY, "count": PER_PAGE, "cursor": cursor, "view": "COMPLETE"}
        resp = requests.get(ENDPOINT, params=params, headers=_headers(api_key, inst_token), timeout=30)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} at page {page}: {resp.text[:300]}")
            break
        payload = resp.json().get("search-results") or {}
        if total_hits is None:
            total_hits = int(payload.get("opensearch:totalResults", 0))
            print(f"total_hits reported: {total_hits}")

        entries = payload.get("entry") or []
        # Scopus returns a single-element placeholder entry with an
        # "error" key when a cursor page is genuinely empty.
        entries = [e for e in entries if "error" not in e]
        if not entries:
            break

        for e in entries:
            records.append(parse_entry(e))

        next_cursor = (payload.get("cursor") or {}).get("@next")
        page += 1
        print(f"page {page}: +{len(entries)} records (total so far {len(records)})")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(DELAY_S)

    out_path = Path("scopus_raw.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    n_with_abstract = sum(1 for r in records if r["abstract"])
    print(f"\nWrote {len(records)} records ({n_with_abstract} with abstract) -> {out_path}")


if __name__ == "__main__":
    main()
