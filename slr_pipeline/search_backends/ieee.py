"""slr_pipeline.search_backends.ieee -- Track 2 plan §2.6 row `ieee`.

`IEEE_API_KEY` is required; its absence is a `.search()`-entry failure
(never a network call), mirroring `springer.py`."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from slr_pipeline.search_backends.base import (
    SearchBackendRequest,
    SearchBackendResult,
    finalize_result,
    record_id,
)
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

ENDPOINT = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
PAGE_CAP = 200


def _parse_article(article: Dict[str, Any]) -> Record:
    article_number = article.get("article_number")
    if not article_number:
        raise ValueError("ieee article missing article_number")
    article_number = str(article_number)
    authors = [
        a.get("full_name", "")
        for a in (article.get("authors") or {}).get("authors") or []
        if a.get("full_name")
    ]

    return Record(
        id=record_id("ieee", article_number),
        source="ieee",
        source_id=article_number,
        doi=article.get("doi"),
        title=article.get("title") or "",
        abstract=article.get("abstract") or None,
        authors=authors,
        year=article.get("publication_year"),
        journal=article.get("publication_title"),
        cited_by=article.get("citing_paper_count"),
        url=article.get("html_url"),
    )


class IeeeBackend:
    name = "ieee"

    def search(self, request: SearchBackendRequest) -> SearchBackendResult:
        api_key = os.environ.get("IEEE_API_KEY")
        if not api_key:
            return finalize_result("ieee", request, [], 0, None, "IEEE_API_KEY not set")

        client = HttpClient()
        records: List[Record] = []
        skipped = 0
        total_hits: Optional[int] = None
        failure_detail: Optional[str] = None
        start_record = 1  # 1-based

        try:
            while len(records) < request.max_results:
                params: Dict[str, Any] = {
                    "apikey": api_key,
                    "querytext": request.query,
                    "start_record": start_record,
                    "max_records": min(PAGE_CAP, request.max_results - len(records)),
                }
                if request.date_from:
                    params["start_year"] = request.date_from
                if request.date_to:
                    params["end_year"] = request.date_to
                resp = client.get(ENDPOINT, params=params)
                payload = resp.json()
                if total_hits is None and payload.get("total_records") is not None:
                    total_hits = payload["total_records"]
                articles = payload.get("articles") or []
                for article in articles:
                    try:
                        records.append(_parse_article(article))
                    except Exception:
                        skipped += 1
                if not articles or len(articles) < params["max_records"]:
                    break
                start_record += len(articles)
        except Exception as exc:
            failure_detail = f"ieee pagination error: {exc}"

        records = records[: request.max_results]
        return finalize_result("ieee", request, records, skipped, total_hits, failure_detail)
