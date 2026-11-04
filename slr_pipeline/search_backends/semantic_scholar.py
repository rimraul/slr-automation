"""slr_pipeline.search_backends.semantic_scholar -- Track 2 plan §2.6 row
`semantic_scholar`."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from slr_pipeline.search_backends.base import SearchBackendRequest, finalize_result, record_id
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
PAGE_CAP = 1000
FIELDS = "externalIds,title,abstract,authors,year,venue,citationCount,url"


def _year_range(date_from: Optional[str], date_to: Optional[str]) -> Optional[str]:
    if date_from and date_to:
        return f"{date_from}-{date_to}"
    if date_from:
        return f"{date_from}-"
    if date_to:
        return f"-{date_to}"
    return None


class SemanticScholarBackend:
    name = "semantic_scholar"

    def search(self, request: SearchBackendRequest):
        client = HttpClient()
        api_key = os.environ.get("S2_API_KEY")
        headers = {"x-api-key": api_key} if api_key else None

        params: Dict[str, Any] = {"query": request.query, "fields": FIELDS}
        year_range = _year_range(request.date_from, request.date_to)
        if year_range:
            params["year"] = year_range

        records: List[Record] = []
        skipped = 0
        total_hits: Optional[int] = None
        failure_detail: Optional[str] = None
        token: Optional[str] = None

        try:
            while len(records) < request.max_results:
                page_params = dict(params)
                if token:
                    page_params["token"] = token
                resp = client.get(ENDPOINT, params=page_params, headers=headers)
                payload = resp.json()
                if total_hits is None and payload.get("total") is not None:
                    total_hits = payload["total"]
                data = payload.get("data") or []
                for paper in data:
                    paper_id = paper.get("paperId")
                    if not paper_id:
                        skipped += 1
                        continue
                    try:
                        external_ids = paper.get("externalIds") or {}
                        records.append(Record(
                            id=record_id("semantic_scholar", paper_id),
                            source="semantic_scholar",
                            source_id=paper_id,
                            doi=external_ids.get("DOI"),
                            title=paper.get("title") or "",
                            abstract=paper.get("abstract"),
                            authors=[a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")],
                            year=paper.get("year"),
                            journal=paper.get("venue") or None,
                            cited_by=paper.get("citationCount"),
                            url=paper.get("url"),
                        ))
                    except Exception:
                        skipped += 1
                    if len(records) >= request.max_results:
                        break
                token = payload.get("token")
                if not token or not data:
                    break
        except Exception as exc:
            failure_detail = f"semantic_scholar pagination error: {exc}"

        records = records[: request.max_results]
        return finalize_result("semantic_scholar", request, records, skipped, total_hits, failure_detail)
