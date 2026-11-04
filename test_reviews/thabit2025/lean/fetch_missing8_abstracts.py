"""
Fetch abstracts (Scopus dc:description, COMPLETE view) for the 8 Appendix B
titles absent from our own search corpus, so they can be added to the
known_included screening-test pool (51 -> 59). WoS Starter never returns
abstracts (confirmed in SEARCHING/adapters/wos_starter.py) -- Scopus
COMPLETE view is the only abstract source available here.
"""
from __future__ import annotations

import json

import requests

from scopus_calibration import _headers, load_env

ENDPOINT = "https://api.elsevier.com/content/search/scopus"

MISSING_8_TITLES = [
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

    records = []
    for title in MISSING_8_TITLES:
        query = f'TITLE("{title}")'
        params = {"query": query, "count": 1, "view": "COMPLETE"}
        resp = requests.get(ENDPOINT, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()["search-results"]
        total = int(data["opensearch:totalResults"])
        if total < 1 or not data.get("entry"):
            print(f"MISSING abstract: {title[:70]}")
            continue
        entry = data["entry"][0]
        abstract = (entry.get("dc:description") or "").strip()
        journal = entry.get("prism:publicationName")
        cover = entry.get("prism:coverDate", "")
        year = int(cover[:4]) if cover else None
        has_abstract = bool(abstract)
        print(f"{'OK' if has_abstract else 'NO ABSTRACT'}: {title[:70]}  (len={len(abstract)})")
        records.append({
            "title": entry.get("dc:title") or title,
            "abstract": abstract or None,
            "journal": journal,
            "year": year,
            "source": "scopus_manual_fetch",
        })

    (open("missing8_records.json", "w", encoding="utf-8")).write(
        json.dumps(records, indent=2, ensure_ascii=False)
    )
    print(f"\n-> missing8_records.json ({len(records)} records, {sum(1 for r in records if r['abstract'])} with abstract)")


if __name__ == "__main__":
    main()
