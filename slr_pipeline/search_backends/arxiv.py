"""slr_pipeline.search_backends.arxiv -- Track 2 plan §2.6 row `arxiv`.

Legacy `http://export.arxiv.org` -- HTTPS rewrite (confirmed live: the
legacy endpoint 301-redirects to `https://`). Neither date nor language
filtering is supported server-side; date is approximated by a client-side
post-filter on the parsed `published` year (`base.DATE_UNSUPPORTED`), both
deviations are recorded by `search_node` via the frozen
`LANGUAGE_UNSUPPORTED`/`DATE_UNSUPPORTED` tables, not by this adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import xmltodict

from slr_pipeline.search_backends._util import as_list, text_of, year_from_text
from slr_pipeline.search_backends.base import SearchBackendRequest, finalize_result, record_id
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

ENDPOINT = "https://export.arxiv.org/api/query"
PAGE_CAP = 100


def _parse_entry(entry: Dict[str, Any]) -> Record:
    entry_id = text_of(entry.get("id"))
    if not entry_id:
        raise ValueError("arXiv entry missing id")

    doi = entry.get("arxiv:doi")
    doi = text_of(doi) if doi else None

    title = text_of(entry.get("title")).strip()
    abstract = text_of(entry.get("summary")).strip() or None

    authors = []
    for a in as_list(entry.get("author")):
        if isinstance(a, dict) and a.get("name"):
            authors.append(text_of(a["name"]))

    year = year_from_text(text_of(entry.get("published")))

    return Record(
        id=record_id("arxiv", entry_id),
        source="arxiv",
        source_id=entry_id,
        doi=doi or None,
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        journal="arXiv",
        cited_by=None,
        url=entry_id,
    )


class ArxivBackend:
    name = "arxiv"

    def search(self, request: SearchBackendRequest):
        client = HttpClient()
        records: List[Record] = []
        skipped = 0
        total_hits: Optional[int] = None
        failure_detail: Optional[str] = None
        start = 0

        try:
            # Over-fetch: the date post-filter below may discard entries,
            # so keep paging until either max_results usable records are
            # collected or the source is exhausted.
            while len(records) < request.max_results:
                params: Dict[str, Any] = {
                    "search_query": request.query,
                    "start": start,
                    "max_results": PAGE_CAP,
                }
                resp = client.get(ENDPOINT, params=params)
                payload = xmltodict.parse(resp.text)
                feed = payload.get("feed", {}) or {}
                if total_hits is None:
                    raw_total = feed.get("opensearch:totalResults")
                    if raw_total is not None:
                        try:
                            total_hits = int(text_of(raw_total))
                        except ValueError:
                            total_hits = None
                entries = as_list(feed.get("entry"))
                if not entries:
                    break
                for entry in entries:
                    try:
                        rec = _parse_entry(entry)
                    except Exception:
                        skipped += 1
                        continue
                    if request.date_from and (rec.year is None or rec.year < int(request.date_from)):
                        continue
                    if request.date_to and (rec.year is None or rec.year > int(request.date_to)):
                        continue
                    records.append(rec)
                    if len(records) >= request.max_results:
                        break
                start += len(entries)
                if len(entries) < PAGE_CAP:
                    break
        except Exception as exc:
            failure_detail = f"arxiv pagination error: {exc}"

        records = records[: request.max_results]
        if request.date_from or request.date_to:
            # The client-side date post-filter (module docstring) drops
            # entries the source already returned -- `total_hits` (the
            # source's own unfiltered opensearch:totalResults) no longer
            # describes a population comparable to `records`, so it must not
            # feed `finalize_result`'s unexplained-retrieval-gap check
            # (review finding P1, 2026-09-24): that check is for a source
            # under-returning its own reported hits, not for records this
            # adapter chose to discard after fetching them.
            total_hits = None
        return finalize_result("arxiv", request, records, skipped, total_hits, failure_detail)
