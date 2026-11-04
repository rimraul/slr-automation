"""slr_pipeline.search_backends.openalex -- Track 2 plan §2.6 row `openalex`."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from slr_pipeline.search_backends._util import int_or_none
from slr_pipeline.search_backends.base import (
    SearchBackendRequest,
    SearchBackendResult,
    SearchCheckpointHooks,
    SearchConfigError,
    SearchPageCheckpoint,
    compute_request_fingerprint,
    finalize_result,
    record_id,
)
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

# 429 excluded from this adapter's own retry set (unlike HttpClient's
# DEFAULT_RETRY_STATUSES) so a rate-limited/budget-exhausted response
# reaches .search() as a genuine single-request requests.exceptions.HTTPError
# with real X-RateLimit-* response headers intact, rather than being
# auto-retried 3x by urllib3 into a header-less requests.exceptions.RetryError
# (live-run finding 2026-09-23: OpenAlex retired the mailto polite-pool
# mechanism in February 2026 -- mailto is now ignored -- and a keyless
# request's daily credit budget, once exhausted, returns 429 until midnight
# UTC; retrying within the same call is futile, only OPENALEX_API_KEY raises
# the budget).
_RETRY_STATUSES = frozenset({500, 502, 503, 504})

ENDPOINT = "https://api.openalex.org/works"
PAGE_CAP = 100

# Bumped from "1" (Checkpoint 3, review finding P1 2026-09-24): the
# retrieval semantics materially changed from full-text-inclusive
# (`default.search`) to title/abstract-only (`title_and_abstract.search`)
# -- old and new searches must be distinguishable in the persisted log.
ADAPTER_CONFIG_VERSION = "2"


def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not inverted_index:
        return None
    positions: Dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _date_filter(date_from: Optional[str], date_to: Optional[str]) -> Optional[str]:
    parts = []
    if date_from:
        parts.append(f"from_publication_date:{date_from}-01-01")
    if date_to:
        parts.append(f"to_publication_date:{date_to}-12-31")
    return ",".join(parts) if parts else None


class OpenAlexBackend:
    name = "openalex"

    def search(
        self, request: SearchBackendRequest,
        *, checkpoint: Optional[SearchCheckpointHooks] = None,
    ) -> SearchBackendResult:
        # Review finding P2 (2026-09-24): `filter` values are joined with
        # `,` (see `_date_filter()` and the `",".join(filters)` below) --
        # OpenAlex itself treats `,` as the AND-separator between filter
        # clauses, so a literal comma inside `request.query` would silently
        # splice in an extra, unintended filter clause instead of being
        # part of the search text (even inside a quoted phrase -- our join
        # is a plain string operation with no quoting awareness). Reject
        # pre-network rather than risk a silently-wrong query.
        if "," in request.query:
            raise SearchConfigError(
                "openalex: query must not contain a literal ',' -- OpenAlex's "
                "filter syntax treats ',' as the AND-separator between filter "
                "clauses, so a comma inside the query text (even in a quoted "
                f"phrase) would be misinterpreted: {request.query!r}"
            )
        effective_query = f"title_and_abstract.search:{request.query}"

        fingerprint = compute_request_fingerprint("openalex", request, ADAPTER_CONFIG_VERSION)
        existing_checkpoint: Optional[SearchPageCheckpoint] = (
            checkpoint.load(fingerprint) if checkpoint is not None else None
        )
        if existing_checkpoint is not None and existing_checkpoint.completed:
            # Restart safety: already finished this exact request in a
            # prior run -- return the cached result with zero network calls
            # (Checkpoint 4, same discipline as pubmed.py).
            restored = [Record.from_dict(d) for d in existing_checkpoint.records_so_far]
            return finalize_result(
                "openalex", request, restored, 0, existing_checkpoint.total_hits, None,
                source_exhausted=existing_checkpoint.source_exhausted,
                termination_reason=existing_checkpoint.termination_reason,
                effective_query=existing_checkpoint.effective_query or effective_query,
                adapter_config_version=ADAPTER_CONFIG_VERSION,
            )
        resumed = existing_checkpoint is not None and not existing_checkpoint.completed
        resume_state: Dict[str, Any] = existing_checkpoint.cursor_state if resumed else {}

        # Never carried on SearchBackendRequest (base.py's no-secrets-in-request
        # contract) -- read directly from os.environ, mirroring scopus.py's
        # SCOPUS_API_KEY/SCOPUS_INSTITUTIONAL_KEY pattern.
        api_key = os.environ.get("OPENALEX_API_KEY")
        client = HttpClient(retry_statuses=_RETRY_STATUSES)
        records = [Record.from_dict(d) for d in existing_checkpoint.records_so_far] if resumed else []
        skipped = 0
        total_hits: Optional[int] = resume_state.get("total_hits") if resumed else None
        cursor = resume_state.get("cursor", "*") if resumed else "*"
        failure_detail: Optional[str] = None
        rate_limit_remaining: Optional[int] = None
        rate_limit_reset_seconds: Optional[int] = None
        exhausted_by_cursor = False
        # Review finding 7 (2026-09-25): page_count is persisted but was
        # never incremented, so every real checkpoint reported zero pages.
        pages_fetched = existing_checkpoint.page_count if resumed else 0

        def _save_page_checkpoint(*, completed: bool, source_exhausted: Optional[bool] = None,
                                   termination_reason_: Optional[str] = None) -> None:
            if checkpoint is None:
                return
            checkpoint.save(SearchPageCheckpoint(
                request_fingerprint=fingerprint,
                adapter="openalex",
                adapter_config_version=ADAPTER_CONFIG_VERSION,
                cursor_state={"cursor": cursor, "total_hits": total_hits},
                records_so_far=tuple(r.to_dict() for r in records),
                total_hits=total_hits,
                page_count=pages_fetched,
                completed=completed,
                source_exhausted=source_exhausted,
                termination_reason=termination_reason_,
                effective_query=effective_query if completed else None,
            ))

        try:
            while len(records) < request.max_results:
                # Checkpoint 3 query harmonization (2026-09-24): the plain
                # `search` param matches OpenAlex's `default.search`, which
                # includes fulltext -- confirmed live-run cause of the
                # 267,400-hit inflation (Revision 5 trigger). `filter=
                # title_and_abstract.search:...` scopes matching to title +
                # abstract only, comparable to PubMed's `[tiab]` and (module
                # docstring's caveat aside) Scopus's `TITLE-ABS-KEY`.
                filters = [effective_query]
                date_filter = _date_filter(request.date_from, request.date_to)
                if date_filter:
                    filters.append(date_filter)
                params: Dict[str, Any] = {
                    "filter": ",".join(filters),
                    "cursor": cursor,
                    "per_page": min(PAGE_CAP, request.max_results - len(records)),
                }
                if request.mailto:
                    params["mailto"] = request.mailto
                request_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
                try:
                    resp = client.get(ENDPOINT, params=params, headers=request_headers)
                except requests.exceptions.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 429:
                        headers = exc.response.headers
                        rate_limit_remaining = int_or_none(headers.get("X-RateLimit-Remaining"))
                        rate_limit_reset_seconds = int_or_none(headers.get("X-RateLimit-Reset"))
                        failure_detail = (
                            f"openalex rate limit or daily budget exceeded at cursor={cursor!r} "
                            f"(rate_limit_remaining={rate_limit_remaining!r}, "
                            f"rate_limit_reset_seconds={rate_limit_reset_seconds!r}); "
                            "not retrying immediately -- see reset value above; "
                            "set OPENALEX_API_KEY for a higher daily budget"
                        )
                    else:
                        # str(exc) on a real requests HTTPError includes the
                        # full request URL (querystring and all) -- never
                        # interpolate it into a persisted failure_detail;
                        # only the exception's own status/class survive.
                        failure_detail = (
                            f"openalex request failed (pagination cursor {cursor!r}): "
                            f"HTTP status {status!r}, {type(exc).__name__}"
                        )
                    break
                payload = resp.json()
                pages_fetched += 1
                if total_hits is None:
                    total_hits = payload.get("meta", {}).get("count")
                for work in payload.get("results", []):
                    raw_id = work.get("id")
                    if not raw_id:
                        skipped += 1
                        continue
                    native_key = raw_id.replace("https://openalex.org/", "")
                    try:
                        doi = work.get("doi")
                        records.append(Record(
                            id=record_id("openalex", native_key),
                            source="openalex",
                            source_id=native_key,
                            doi=doi.replace("https://doi.org/", "") if doi else None,
                            title=work.get("title") or "",
                            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
                            authors=[
                                a.get("author", {}).get("display_name", "")
                                for a in work.get("authorships", [])
                                if a.get("author", {}).get("display_name")
                            ],
                            year=work.get("publication_year"),
                            journal=(work.get("primary_location") or {}).get("source", {}).get("display_name")
                            if (work.get("primary_location") or {}).get("source") else None,
                            cited_by=work.get("cited_by_count"),
                            url=raw_id,
                        ))
                    except Exception:
                        skipped += 1
                next_cursor = payload.get("meta", {}).get("next_cursor")
                if not next_cursor or not payload.get("results"):
                    # OpenAlex's own genuine end-of-results signal (a null
                    # `next_cursor`, or an explicitly empty `results` page)
                    # -- not a short-page inference, positive proof of
                    # exhaustion (Revision 5 point 1(b)).
                    exhausted_by_cursor = True
                    break
                cursor = next_cursor
                _save_page_checkpoint(completed=False)
        except Exception as exc:
            # Same rationale as above -- a requests.exceptions.ConnectionError's
            # str() also embeds the full request URL; class name only.
            failure_detail = f"openalex pagination error (pagination cursor {cursor!r}): {type(exc).__name__}"

        records = records[: request.max_results]
        source_exhausted: Optional[bool] = None
        termination_reason: Optional[str] = None
        if failure_detail is None and skipped == 0:
            if exhausted_by_cursor and (total_hits is None or len(records) == total_hits):
                source_exhausted = True
                termination_reason = "zero_hits" if total_hits == 0 else "cursor_exhausted_empty_next_page"
            elif len(records) >= request.max_results:
                source_exhausted = False
                termination_reason = "max_results_cap_reached"
            else:
                source_exhausted = False
                termination_reason = "pagination_end_unverified"
        if failure_detail is None:
            _save_page_checkpoint(
                completed=True, source_exhausted=source_exhausted,
                termination_reason_=termination_reason,
            )
        return finalize_result(
            "openalex",
            request,
            records,
            skipped,
            total_hits,
            failure_detail,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset_seconds=rate_limit_reset_seconds,
            source_exhausted=source_exhausted,
            termination_reason=termination_reason,
            effective_query=effective_query,
            adapter_config_version=ADAPTER_CONFIG_VERSION,
        )
