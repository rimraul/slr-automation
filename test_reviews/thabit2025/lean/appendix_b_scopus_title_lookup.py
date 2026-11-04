"""
Lean prototype: live Scopus exact-title lookup for the 8 Appendix B titles
not found in raw_records.json, mirroring appendix_b_wos_title_lookup's WoS
check. count=1 per title -> minimal cost.

Confirms/refutes whether the same "present in DB, missed by our calibrated
theme query" pattern holds in Scopus too (supporting the expert-addition
theory for these 59-included-study gaps).
"""
from __future__ import annotations

import json

import requests

from scopus_calibration import _headers, load_env

ENDPOINT = "https://api.elsevier.com/content/search/scopus"

NOT_FOUND_TITLES = [
    "A Decision Theory Perspective on Wicked Problems, SDGs and Stakeholders: The Case of Deforestation",
    "Organizational Maturity for Co-creation: Towards a Multi-attribute Decision Support Model for Public Organizations",
    "Achieving Sustainable Development Goals Through Collaborative Innovation: Evidence from Four European Initiatives",
    "Paradox and Collaboration in Network Management",
    "Ambidexterity in Cross-Sector Collaborations Involving Public Organizations",
    "Coping with Value Conflicts in Interorganizational Collaborations",
    "Community Foundations as Network Conveners: Structuring Collective Agency for Child Education and Development System Impact",
    "Co-initiation of Collaborative Innovation in Urban Spaces",
]


def main() -> None:
    env = load_env("../../../.env")
    api_key = env["SCOPUS_API_KEY"]
    inst_token = env.get("SCOPUS_INSTITUTIONAL_KEY")
    headers = _headers(api_key, inst_token)

    rows = []
    for i, title in enumerate(NOT_FOUND_TITLES, 1):
        query = f'TITLE("{title}")'
        params = {"query": query, "count": 1, "view": "COMPLETE"}
        resp = requests.get(ENDPOINT, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"{i}\t{title}\tHTTP {resp.status_code}: {resp.text[:200]}")
            rows.append({"n": i, "title": title, "error": f"HTTP {resp.status_code}"})
            continue
        data = resp.json()["search-results"]
        total = int(data["opensearch:totalResults"])
        found_title = None
        found_year = None
        if total >= 1 and data.get("entry"):
            entry = data["entry"][0]
            found_title = entry.get("dc:title")
            cover = entry.get("prism:coverDate", "")
            found_year = cover[:4] if cover else None
        print(f"{i}\ttotal={total}\t{title[:70]}" + (f"  -> {found_title!r} ({found_year})" if found_title else ""))
        rows.append({"n": i, "title": title, "total": total, "scopus_title": found_title, "year": found_year})

    with open("appendix_b_scopus_title_lookup.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\n-> appendix_b_scopus_title_lookup.json")


if __name__ == "__main__":
    main()
