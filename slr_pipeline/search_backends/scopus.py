"""slr_pipeline.search_backends.scopus -- Track 3 plan Unit C, ported from
`C:\\projects\\SEARCHING\\adapters\\scopus.py` (structural template:
`ieee.py`'s missing-key -> zero-network `finalize_result`, shared
`HttpClient()`, per-item try/except with a `skipped` counter).

Auth via `SCOPUS_API_KEY` / `SCOPUS_INSTITUTIONAL_KEY` env vars directly --
never `SEARCHING.keys.get_key`, never `my_api_keys.py` (user's explicit
instruction). Diverges from `ieee.py`'s shape exactly where the real
Elsevier API does:
  - Cursor-based pagination (`cursor="*"`, then `search-results.cursor.
    @next`), not `start=` offset -- Elsevier caps offset pagination at
    5,000 results (donor adapter's own comment).
  - `time.sleep(1.1)` before every request (personal-key limit is 1
    req/sec) -- monkeypatched away in tests.
  - 401: if an institutional token was in use, retry once with it dropped
    (forces `view=STANDARD`); a still-401 fails this adapter's whole
    result closed (`finalize_result(..., failure_detail=...)`), never a
    raise -- matches this codebase's non-crashing adapter convention. 401
    is never in `HttpClient`'s own `status_forcelist`, so it always
    reaches this adapter as a genuine `requests.exceptions.HTTPError`
    with `.response.status_code == 401` after a single real request.
  - 429/5xx: `HttpClient`'s own built-in
    `Retry(status_forcelist=[429, 500, 502, 503, 504], raise_on_status=True
    (urllib3 default, not overridden in http.py))` already retries a
    transient one of those up to 3 times internally -- but if every
    attempt still comes back forced-status, `requests.Session.get()`
    itself raises `requests.exceptions.RetryError`, NOT `HTTPError`; no
    raise-able `Response`/`.status_code` is ever available to this
    adapter at that point (confirmed empirically, code-review round 1,
    2026-09-10 -- the plan's original design assumed a bare 429
    `HTTPError` reached this adapter directly, which real `HttpClient`
    never produces). This adapter catches `RetryError` itself, sleeps 60s,
    and retries the SAME cursor -- bounded to
    `_MAX_RETRY_ATTEMPTS_PER_CURSOR` (3) attempts before giving up and
    failing this result closed, so a persistently-unavailable/rate-limited
    Scopus cannot hang this adapter forever. Sleeps are monkeypatched away
    in tests.
  - Pagination termination checks BOTH `not next_cursor` AND
    `next_cursor == cursor` (Scopus's own documented terminal-page signal)
    -- omitting the latter re-fetches and duplicates the last page
    forever whenever total hits is an exact multiple of `_PER_PAGE`
    (round-2 code-review Finding 1, fixed 2026-09-10).

SCOPE DECISION (round-2 code-review Finding 3, revised round-3, 2026-09-10):
the `RetryError`-bounded-retry mechanism above is added to THIS adapter
only. Every other real adapter in this package (`ieee`, `pubmed`,
`openalex`, `crossref`, `springer`, `semantic_scholar`, `arxiv`) wraps its
whole pagination loop in a broad `except Exception`, which already
catches `RetryError` too -- so none of them crash on a sustained 429/5xx
burst; they fail closed immediately on the first one instead of
backing off and retrying like this adapter now does (confirmed via grep,
round-3 code-review -- round 2's framing of this as a "crash" gap was
too strong; it's a lesser resilience/UX gap: fewer transient bursts
survived, not an uncaught exception). Fixing that -- giving every adapter
the same bounded backoff-and-retry this one has -- is a cross-cutting
`search_backends`-wide hardening change, out of this plan's Unit C scope
(Scopus port only) and out of this port's own blast radius (it would
touch six already-shipped, previously-reviewed adapter files unrelated
to Track 3). Flagged forward as a deferred, separately-scoped hardening
item -- not fixed here, same treatment as Unit B round 2's
`dedup_embedding` preflight finding being flagged forward to Unit C
rather than fixed in place.

`Record` field mapping only covers fields with a real home on
`slr_pipeline.state.Record` (`authkeywords` -> `keywords`, etc.) -- ASJC
codes, affiliations, Scopus author IDs, open-access flags and other
Scopus-specific extras are dropped, matching `ieee.py`'s established
"drop fields with no home" precedent. No new `Record` field was added as
part of this port.

`base.py`'s `LANGUAGE_UNSUPPORTED`/`DATE_UNSUPPORTED` membership: this
adapter's `params["date"]` cleanly expresses `request.date_from`/
`date_to` (both already 4-digit years, matching Scopus's own `date=YYYY`
/ `date=YYYY-YYYY` forms directly) -- `scopus` is NOT in
`DATE_UNSUPPORTED`. `request.language` is a BCP-47 2-3 letter primary
subtag (`base.py`'s own `SearchBackendRequest` contract), but Scopus's
Advanced Search `LANGUAGE` field code expects a full language NAME
(e.g. `LANGUAGE(english)`), not an ISO code, and no code-to-name lookup
table exists anywhere in this codebase -- `scopus` IS in
`LANGUAGE_UNSUPPORTED`, this adapter never reads `request.language`.

Round-5 code-review fixes (2026-09-10): the `date_to`-only case
(previously silently dropped the filter -- every sibling real adapter
supports this case, this one didn't) is now covered (semantics corrected
again in round 6, below); each page request's `count` is now capped to
what's actually still needed (`min(_PER_PAGE, request.max_results -
len(records))`, matching `ieee.py`'s own `PAGE_CAP` precedent) instead of
always requesting a full `_PER_PAGE`; `author`/`link` entry fields are
read via the shared `as_list()` helper, not a bare `or []`, since
Scopus's XML-derived JSON can collapse a singleton repeatable field to a
bare dict instead of a one-element list; `year` parsing reuses the shared
`year_from_text()` helper instead of a hand-rolled equivalent.

Round-6 code-review fixes (2026-09-10): round 5's `date_from`-only /
`date_to`-only handling both set `params["date"]` to a bare `YYYY`, but
on Scopus that means an EXACT single year, not an open "from X onward" /
"through Y" range (donor adapter's own documented `date` formats: exact
year, year range, or day-level ISO range -- no open-bound form). That
silently narrowed a "papers since 2020" search down to "papers in 2020
only". Fixed: a single date bound is now expressed via Scopus's
documented `PUBYEAR >` / `PUBYEAR <` query-string operators appended to
`request.query` instead of `params["date"]` (only the closed
`date_from`-and-`date_to` range still uses `params["date"]`, which
correctly expresses that case). Also: `search-results.entry` (the
top-level per-record field) now goes through the same `as_list()`
protection as `author`/`link` -- it is the identical kind of Atom-feed
repeatable field and can equally collapse to a bare dict on a genuine
single-hit query, which previously turned a successful 1-result search
into a reported `retrieval_status="failed"` with 0 records.

Checkpoint 3 query harmonization (2026-09-24): `request.query` reaching
this adapter is expected to already be Scopus's native `TITLE-ABS-KEY(...)`
syntax (produced by `search_strategy_builder.py`'s db-translation step,
enforced pre-network by `nodes/search.py`'s `_STRICT_TRANSLATION_DBS`
membership) -- this adapter itself does not wrap or validate the field
scope. `TITLE-ABS-KEY` is documented (here and in `_DB_DESCRIPTIONS`) as
matching Scopus's author/indexed keywords in addition to title/abstract
text, which is broader coverage than PubMed's `[tiab]` or OpenAlex's
`title_and_abstract.search` filter -- never treated as equivalent scoping
when comparing hit counts across databases.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from slr_pipeline.search_backends.base import (
    SearchBackendRequest,
    SearchBackendResult,
    SearchCheckpointHooks,
    SearchPageCheckpoint,
    compute_request_fingerprint,
    finalize_result,
    record_id,
)
from slr_pipeline.search_backends._util import as_list, int_or_none, sanitize_remote_error_text, year_from_text
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

ENDPOINT = "https://api.elsevier.com/content/search/scopus"
_PER_PAGE = 25                   # max count for the COMPLETE/STANDARD view tiers this key uses
_RATE_LIMIT_DELAY_S = 1.1        # personal-key limit is 1 req/sec
_RATE_LIMIT_RETRY_SLEEP_S = 60
_MAX_RETRY_ATTEMPTS_PER_CURSOR = 3


def _headers(api_key: str, inst_token: Optional[str]) -> Dict[str, str]:
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token
    return headers


def _parse_entry(entry: Dict[str, Any], query: str) -> Record:
    eid = entry.get("eid")
    if not eid:
        raise ValueError("scopus entry missing eid")
    eid = str(eid)

    # `as_list()` handles Scopus's XML-derived JSON quirk of collapsing a
    # singleton repeatable field (single author/link) to a bare dict
    # instead of a one-element list (round-5 code-review finding) -- a
    # raw `entry.get("author") or []` would iterate a dict's string keys
    # instead, raising AttributeError on `.get()` and silently miscounting
    # a valid single-author record as malformed.
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
        # Scopus uses "|" to separate multiple keywords; a single keyword
        # phrase (no "|") is never space-split -- a bare-space fallback
        # would shred a multi-word phrase like "machine learning" into
        # ["machine", "learning"] (round-2 code-review Finding 2).
        # `str.split("|")` already returns `[auth_kw_raw]` unchanged when
        # no "|" is present, so no conditional is needed (round-4 finding).
        keywords = [k.strip() for k in auth_kw_raw.split("|") if k.strip()]

    year = year_from_text(entry.get("prism:coverDate"))

    cited_by = int_or_none(entry.get("citedby-count"))

    scopus_url = None
    for link in as_list(entry.get("link")):
        if link.get("@ref") == "scopus":
            scopus_url = link.get("@href")
            break

    return Record(
        id=record_id("scopus", eid),
        source="scopus",
        source_id=eid,
        doi=entry.get("prism:doi"),
        title=(entry.get("dc:title") or "").strip(),
        abstract=(entry.get("dc:description") or "").strip() or None,
        authors=authors,
        year=year,
        journal=entry.get("prism:publicationName"),
        keywords=keywords,
        cited_by=cited_by,
        url=scopus_url or entry.get("prism:url"),
    )


class ScopusBackend:
    name = "scopus"

    def search(
        self, request: SearchBackendRequest,
        *, checkpoint: Optional[SearchCheckpointHooks] = None,
    ) -> SearchBackendResult:
        api_key = os.environ.get("SCOPUS_API_KEY")
        if not api_key:
            return finalize_result("scopus", request, [], 0, None, "SCOPUS_API_KEY not set")
        inst_token = os.environ.get("SCOPUS_INSTITUTIONAL_KEY") or None

        # Review finding 6 (2026-09-25): bind the REQUESTED access tier into
        # checkpoint identity -- a checkpoint started while an institutional
        # token was configured (tier="COMPLETE") must never silently resume
        # under a personal-key-only run (tier="STANDARD"), or vice versa;
        # each tier's records carry different metadata completeness.
        requested_access_tier = "COMPLETE" if inst_token is not None else "STANDARD"
        fingerprint = compute_request_fingerprint(
            "scopus", request, "1", identity_extra={"requested_access_tier": requested_access_tier},
        )
        existing_checkpoint: Optional[SearchPageCheckpoint] = (
            checkpoint.load(fingerprint) if checkpoint is not None else None
        )
        if existing_checkpoint is not None and existing_checkpoint.completed:
            # Restart safety: already finished this exact request in a
            # prior run -- return the cached result with zero network calls
            # (Checkpoint 4, same discipline as pubmed.py/openalex.py).
            restored = [Record.from_dict(d) for d in existing_checkpoint.records_so_far]
            return finalize_result(
                "scopus", request, restored, 0, existing_checkpoint.total_hits, None,
                existing_checkpoint.cursor_state.get("access_mode"),
                source_exhausted=existing_checkpoint.source_exhausted,
                termination_reason=existing_checkpoint.termination_reason,
            )
        resumed = existing_checkpoint is not None and not existing_checkpoint.completed
        resume_state: Dict[str, Any] = existing_checkpoint.cursor_state if resumed else {}

        client = HttpClient()
        records = [Record.from_dict(d) for d in existing_checkpoint.records_so_far] if resumed else []
        skipped = 0
        total_hits: Optional[int] = resume_state.get("total_hits") if resumed else None
        failure_detail: Optional[str] = None
        cursor = resume_state.get("cursor", "*") if resumed else "*"
        retry_attempts = 0
        # Live-run finding (2026-09-22): the persisted search artifact had no
        # way to prove whether this run got Elsevier's fuller COMPLETE view
        # (needs a live institutional token) or silently fell back to
        # STANDARD (insufficient abstract metadata for screening in some
        # cases) -- tracked here and surfaced via finalize_result's
        # access_mode, not just the view= sent on the last page.
        requested_complete = inst_token is not None
        # Review finding 6 (2026-09-25): a resumed run's `records_so_far`
        # already contains pages fetched under whatever access mode the
        # PRIOR run's checkpoint actually reached, which can be a downgrade
        # even when today's `requested_complete` is True again (same
        # institutional token env var, but a previous 401 forced STANDARD
        # mid-run). Restore that downgrade so the resumed result never
        # reports "COMPLETE" for a corpus that includes STANDARD-fetched
        # earlier records.
        prior_access_mode = resume_state.get("access_mode") if resumed else None
        downgraded_to_standard = prior_access_mode is not None and prior_access_mode != "COMPLETE"
        exhausted_by_cursor = False
        # Review finding 7 (2026-09-25): page_count is persisted but was
        # never incremented, so every real checkpoint reported zero pages.
        pages_fetched = existing_checkpoint.page_count if resumed else 0

        def _current_access_mode() -> str:
            # Review finding 6 (2026-09-25): `downgraded_to_standard` is only
            # ever set True, never reset back to False, so this always
            # reflects the CUMULATIVE WORST access mode seen so far -- once a
            # run downgrades, every subsequent page (and the final result)
            # reports the downgraded tier, never silently reverting to
            # "COMPLETE" just because a later retry happened to succeed
            # without needing the token again.
            if not requested_complete:
                return "STANDARD"
            if downgraded_to_standard:
                return "STANDARD (downgraded from COMPLETE after a 401 on the institutional token)"
            return "COMPLETE"

        def _save_page_checkpoint(*, completed: bool, source_exhausted: Optional[bool] = None,
                                   termination_reason_: Optional[str] = None) -> None:
            if checkpoint is None:
                return
            access_mode_ = _current_access_mode()
            checkpoint.save(SearchPageCheckpoint(
                request_fingerprint=fingerprint,
                adapter="scopus",
                adapter_config_version="1",
                cursor_state={"cursor": cursor, "total_hits": total_hits, "access_mode": access_mode_},
                page_count=pages_fetched,
                records_so_far=tuple(r.to_dict() for r in records),
                total_hits=total_hits,
                completed=completed,
                source_exhausted=source_exhausted,
                termination_reason=termination_reason_,
            ))

        try:
            while len(records) < request.max_results:
                page_count = min(_PER_PAGE, request.max_results - len(records))
                query_str = request.query
                params: Dict[str, Any] = {
                    "count": page_count,
                    "cursor": cursor,
                    "view": "COMPLETE" if inst_token else "STANDARD",
                }
                if request.date_from and request.date_to:
                    # Scopus's `date` param cleanly expresses a closed range.
                    params["date"] = f"{request.date_from}-{request.date_to}"
                elif request.date_from:
                    # `date` alone means an EXACT single year on Scopus (donor
                    # adapter's own documented formats), never an open "from
                    # X onward" range -- unlike every sibling adapter's
                    # single-bound semantics. A bare `params["date"] =
                    # request.date_from` would silently narrow "papers since
                    # 2020" down to "papers in 2020 only" (round-6 code-review
                    # finding). Express the open lower bound via Scopus's
                    # documented `PUBYEAR >` query operator instead.
                    # Parenthesize the base query: `request.query` can carry
                    # a top-level OR (e.g. nodes/search.py's `_sanitize_query`
                    # multi-phrase rebuilds), and AND binds tighter than OR
                    # in this query-language family -- an unparenthesized
                    # append would silently scope the date filter to only
                    # the last OR-clause (round-7 code-review finding).
                    query_str = f"({request.query}) AND PUBYEAR > {int(request.date_from) - 1}"
                elif request.date_to:
                    query_str = f"({request.query}) AND PUBYEAR < {int(request.date_to) + 1}"
                params["query"] = query_str

                time.sleep(_RATE_LIMIT_DELAY_S)
                try:
                    resp = client.get(ENDPOINT, params=params, headers=_headers(api_key, inst_token))
                except requests.exceptions.HTTPError as exc:
                    # 401 is never in HttpClient's own status_forcelist, so it always
                    # reaches here as a genuine single-request HTTPError.
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 401 and inst_token:
                        # institutional token requires institutional IP -- drop it and
                        # retry the SAME cursor once with personal-key/STANDARD view
                        inst_token = None
                        downgraded_to_standard = True
                        continue
                    failure_detail = f"scopus request failed at cursor={cursor!r}: {exc}"
                    break
                except requests.exceptions.RetryError as exc:
                    # HttpClient's own Retry(status_forcelist=[429,500,502,503,504],
                    # raise_on_status=True) already retried 3 times internally and
                    # still saw a forced status on every attempt -- no raise-able
                    # Response/.status_code exists at this point, only this
                    # exception. Back off further and retry the SAME cursor,
                    # bounded, rather than fail closed on a single sustained burst.
                    retry_attempts += 1
                    if retry_attempts > _MAX_RETRY_ATTEMPTS_PER_CURSOR:
                        failure_detail = (
                            f"scopus request repeatedly rate-limited/unavailable at "
                            f"cursor={cursor!r} after {retry_attempts} attempts: {exc}"
                        )
                        break
                    time.sleep(_RATE_LIMIT_RETRY_SLEEP_S)
                    continue

                retry_attempts = 0
                pages_fetched += 1
                payload = resp.json().get("search-results") or {}
                # `search-results.entry` is the same kind of Atom-feed
                # repeatable field as `author`/`link` -- a single-hit query
                # can collapse it to a bare dict instead of a one-element
                # list (round-6 code-review finding, same root cause as the
                # round-5 author/link fix, one level shallower).
                entries = as_list(payload.get("entry"))
                if not entries:
                    # A genuinely empty page -- Scopus's own real end of
                    # results, not a short-page inference.
                    exhausted_by_cursor = True
                    break
                if len(entries) == 1 and isinstance(entries[0], dict) and "error" in entries[0]:
                    # Scopus's OWN documented empty-result-set signal is
                    # exactly this message; anything else in the `error`
                    # key is a genuine error (bad request, malformed
                    # query, quota/auth issue not already caught as a 401,
                    # etc.) that must never be reported as a clean 0-hit
                    # "complete" result -- confirmed root cause of a real
                    # live run silently treating a real Scopus error as a
                    # genuine empty search.
                    error_text = str(entries[0]["error"])
                    if error_text.strip().lower() == "result set was empty":
                        # Scopus's own documented zero-hit signal -- real
                        # proof of exhaustion, not an inferred short page.
                        exhausted_by_cursor = True
                    else:
                        # `base.py`'s SearchBackendResult.failure_detail contract:
                        # "sanitised: never a URL query string, never a token" --
                        # this text comes verbatim from Scopus's own response
                        # body, outside this adapter's control, so it must be
                        # bounded and credential-redacted before being embedded.
                        failure_detail = (
                            f"scopus request failed at cursor={cursor!r}: "
                            f"{sanitize_remote_error_text(error_text)}"
                        )
                    break
                if total_hits is None:
                    total_hits = int_or_none(payload.get("opensearch:totalResults"))

                for entry in entries:
                    try:
                        records.append(_parse_entry(entry, request.query))
                    except Exception:
                        skipped += 1

                next_cursor = (payload.get("cursor") or {}).get("@next")
                # `next_cursor == cursor` is Scopus's own documented terminal-
                # page signal; missing this check re-fetches and duplicates
                # the last page forever whenever total hits is an exact
                # multiple of _PER_PAGE (round-2 code-review Finding 1 -- an
                # uncaught pydantic ValidationError from finalize_result()
                # once `returned` exceeds the first-page-frozen `total_hits`).
                if not next_cursor or next_cursor == cursor:
                    # Scopus's own documented terminal-page signal --
                    # positive proof of exhaustion (Revision 5 point 1(b)).
                    exhausted_by_cursor = True
                    break
                if len(entries) < page_count:
                    # A short page WITHOUT an explicit terminal-cursor
                    # signal is ambiguous, not proof -- do not claim
                    # exhaustion from this alone.
                    break
                cursor = next_cursor
                # Checkpoint 4 fix (2026-09-25): must save AFTER `cursor`
                # advances to the next page's cursor, not before -- saving
                # the just-used cursor here would make a resumed run
                # re-request the SAME page it just finished (Scopus's
                # cursor selects which page to fetch), duplicating records
                # instead of continuing forward. Mirrors openalex.py's
                # save-after-advance placement.
                _save_page_checkpoint(completed=False)
        except Exception as exc:
            failure_detail = f"scopus pagination error: {exc}"

        records = records[: request.max_results]
        # `total_hits` is frozen from the FIRST page's `opensearch:totalResults`;
        # a long-running, many-page fetch can outlast a live increase in
        # Scopus's own index between page fetches, so `returned` can end up
        # > that frozen figure. finalize_result()'s SearchBackendResult
        # pydantic invariant requires total_hits >= returned -- clamp it up
        # rather than let a stale hint raise an uncaught ValidationError
        # (round-3 code-review Finding 1).
        if total_hits is not None and total_hits < len(records):
            total_hits = len(records)
        access_mode = _current_access_mode()
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
            "scopus", request, records, skipped, total_hits, failure_detail, access_mode,
            source_exhausted=source_exhausted, termination_reason=termination_reason,
        )
