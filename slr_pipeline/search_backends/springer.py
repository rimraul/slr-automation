"""slr_pipeline.search_backends.springer -- Track 2 plan §2.6 row `springer`.

Legacy `http://api.springernature.com` -- HTTPS rewrite. `SPRINGER_API_KEY`
is required; its absence is a `.search()`-entry failure per plan §2.4's
`retrieval_status == "failed"` credential-absent case, never a network
call."""
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

ENDPOINT = "https://api.springernature.com/meta/v2/json"
PAGE_CAP = 100


def _parse_record(rec: Dict[str, Any]) -> Record:
    doi = rec.get("doi")
    if not doi:
        raise ValueError("springer record missing doi")
    authors = [c.get("creator", "") for c in rec.get("creators") or [] if c.get("creator")]
    pub_date = rec.get("publicationDate") or ""
    year = int(pub_date[:4]) if pub_date[:4].isdigit() else None
    url_entries = rec.get("url") or []
    url = url_entries[0].get("value") if url_entries and isinstance(url_entries[0], dict) else None

    return Record(
        id=record_id("springer", doi.lower()),
        source="springer",
        source_id=doi.lower(),
        doi=doi.lower(),
        title=rec.get("title") or "",
        abstract=rec.get("abstract") or None,
        authors=authors,
        year=year,
        journal=rec.get("publicationName"),
        cited_by=None,
        url=url,
    )


class SpringerBackend:
    name = "springer"

    def search(self, request: SearchBackendRequest) -> SearchBackendResult:
        api_key = os.environ.get("SPRINGER_API_KEY")
        if not api_key:
            return finalize_result(
                "springer", request, [], 0, None, "SPRINGER_API_KEY not set"
            )

        client = HttpClient()
        records: List[Record] = []
        skipped = 0
        total_hits: Optional[int] = None
        failure_detail: Optional[str] = None

        q = request.query
        if request.date_from or request.date_to:
            date_from = f"{request.date_from}-01-01" if request.date_from else "*"
            date_to = f"{request.date_to}-12-31" if request.date_to else "*"
            q = f"{q} onlinedatefrom:{date_from} onlinedateto:{date_to}"
        if request.language:
            q = f"{q} language:{request.language}"

        start = 1  # 1-based
        try:
            while len(records) < request.max_results:
                params: Dict[str, Any] = {
                    "api_key": api_key,
                    "q": q,
                    "s": start,
                    "p": min(PAGE_CAP, request.max_results - len(records)),
                }
                resp = client.get(ENDPOINT, params=params)
                payload = resp.json()
                result_envelope = (payload.get("result") or [{}])[0]
                if total_hits is None and result_envelope.get("total") is not None:
                    total_hits = int(result_envelope["total"])
                page_records = payload.get("records") or []
                for rec in page_records:
                    try:
                        records.append(_parse_record(rec))
                    except Exception:
                        skipped += 1
                if not page_records or len(page_records) < params["p"]:
                    break
                start += len(page_records)
        except Exception as exc:
            failure_detail = f"springer pagination error: {exc}"

        records = records[: request.max_results]
        return finalize_result("springer", request, records, skipped, total_hits, failure_detail)
