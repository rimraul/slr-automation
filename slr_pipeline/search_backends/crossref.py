"""slr_pipeline.search_backends.crossref -- Track 2 plan §2.6 row `crossref`."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from slr_pipeline.search_backends._util import strip_jats
from slr_pipeline.search_backends.base import SearchBackendRequest, finalize_result, record_id
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

ENDPOINT = "https://api.crossref.org/works"
PAGE_CAP = 200


def _date_filter(date_from: Optional[str], date_to: Optional[str]) -> Optional[str]:
    parts = []
    if date_from:
        parts.append(f"from-pub-date:{date_from}-01-01")
    if date_to:
        parts.append(f"until-pub-date:{date_to}-12-31")
    return ",".join(parts) if parts else None


def _parse_item(item: Dict[str, Any]) -> Record:
    doi = item.get("DOI")
    if not doi:
        raise ValueError("crossref item missing DOI")
    titles = item.get("title") or []
    authors = []
    for a in item.get("author") or []:
        name = " ".join(p for p in (a.get("given"), a.get("family")) if p)
        if name:
            authors.append(name)
    year = None
    date_parts = ((item.get("published") or {}).get("date-parts") or [[None]])[0]
    if date_parts and date_parts[0]:
        year = int(date_parts[0])
    journal = (item.get("container-title") or [None])[0]

    return Record(
        id=record_id("crossref", doi.lower()),
        source="crossref",
        source_id=doi.lower(),
        doi=doi.lower(),
        title=titles[0] if titles else "",
        abstract=strip_jats(item.get("abstract")),
        authors=authors,
        year=year,
        journal=journal,
        cited_by=item.get("is-referenced-by-count"),
        url=item.get("URL"),
    )


class CrossrefBackend:
    name = "crossref"

    def search(self, request: SearchBackendRequest):
        client = HttpClient()
        records: List[Record] = []
        skipped = 0
        total_hits: Optional[int] = None
        failure_detail: Optional[str] = None
        cursor = "*"

        try:
            while len(records) < request.max_results:
                params: Dict[str, Any] = {
                    "query.bibliographic": request.query,
                    "cursor": cursor,
                    "rows": min(PAGE_CAP, request.max_results - len(records)),
                }
                date_filter = _date_filter(request.date_from, request.date_to)
                if date_filter:
                    params["filter"] = date_filter
                if request.mailto:
                    params["mailto"] = request.mailto
                resp = client.get(ENDPOINT, params=params)
                message = resp.json().get("message", {})
                if total_hits is None:
                    total_hits = (message.get("total-results"))
                items = message.get("items") or []
                for item in items:
                    try:
                        records.append(_parse_item(item))
                    except Exception:
                        skipped += 1
                next_cursor = message.get("next-cursor")
                if not next_cursor or not items:
                    break
                cursor = next_cursor
        except Exception as exc:
            failure_detail = f"crossref pagination error: {exc}"

        records = records[: request.max_results]
        return finalize_result("crossref", request, records, skipped, total_hits, failure_detail)
