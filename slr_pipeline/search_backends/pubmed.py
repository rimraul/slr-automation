"""slr_pipeline.search_backends.pubmed -- Track 2 plan §2.6 row `pubmed`.

PubMed's own self-contained esearch/efetch logic (previously
`nodes/search.py::_search_pubmed`) is refactored into this backend's class
shape; the esearch/efetch mechanics are unchanged."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import xmltodict

from slr_pipeline.search_backends._util import as_list, sanitize_remote_error_text, text_of, year_from_text
from slr_pipeline.search_backends.base import (
    SearchBackendRequest,
    SearchCheckpointHooks,
    SearchPageCheckpoint,
    compute_request_fingerprint,
    finalize_result,
    record_id,
)
from slr_pipeline.search_backends.http import HttpClient
from slr_pipeline.state import Record

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PAGE_CAP = 200

# PubMed's `Language` field tag is `[la]` (NOT `[lang]`, which is not a
# valid E-utilities field tag and silently matches nothing), and its
# accepted values are spelled-out language names, not BCP-47/ISO codes --
# `request.language` is a 2-3 letter subtag (base.py's own
# `SearchBackendRequest` contract), so it must be mapped, never passed
# through raw. Confirmed root cause of a real live run returning 0 PubMed
# hits: `f"{term} AND {request.language}[lang]"` with `language="en"`
# produced an unsupported field tag AND-ed against an unmatchable value.
_ISO_TO_PUBMED_LANGUAGE = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "ja": "japanese",
    "zh": "chinese", "ru": "russian", "pl": "polish", "sv": "swedish",
    "da": "danish", "no": "norwegian", "fi": "finnish", "tr": "turkish",
    "ko": "korean", "ar": "arabic",
}


@dataclass
class PubMedBoundaryDiagnostics:
    """Revision 5 point 2 boundary instrumentation. Populated in place when a
    caller passes an instance to `PubMedBackend.search(request, diagnostics=...)`
    -- NOT part of the persisted `SearchBackendResult`/`DatabaseSearchLog`
    artifact (the plan's own words: "logged/asserted during the
    investigation ... not necessarily all persisted to the final artifact").
    Offline root-cause diagnosis and regression-test tooling only for the
    PubMed 300/303 gap."""
    esearch_reported_count: Optional[int] = None
    used_usehistory: bool = False
    # First-page ESearch idlist size only (review finding P1, 2026-09-24:
    # further pages are drawn from EFetch, not repeated ESearch calls, so
    # there is normally only ever one entry here -- a second only appears
    # on the defensive no-history fallback path).
    ids_per_esearch_page: List[int] = field(default_factory=list)
    # Items (PubmedArticle + PubmedBookArticle) returned per EFetch call,
    # regardless of whether that call used an explicit id= list or
    # WebEnv/query_key history paging.
    items_per_efetch_page: List[int] = field(default_factory=list)
    unique_pmids_collected: int = 0
    duplicate_pmids_seen: List[str] = field(default_factory=list)
    pubmed_article_count: int = 0
    pubmed_book_article_count: int = 0
    parse_failed_pmids: List[str] = field(default_factory=list)
    # Per-batch PMID-set diffing (review finding P1 #2, 2026-09-24) --
    # only populated where an explicit requested-id set exists to diff
    # against (id=-list EFetch calls); the history-paged EFetch path has
    # no independent expected-id list to compare.
    missing_requested_pmids: List[str] = field(default_factory=list)
    unexpected_efetch_pmids: List[str] = field(default_factory=list)
    requested_pmids_sent_to_efetch: int = 0
    final_records_produced: int = 0
    cursor_confirmed_empty_page: bool = False


def _year_from_pub_date(pub_date: Any) -> Optional[int]:
    """A PubDate/ArticleDate element carries either a structured `Year`, or
    -- when the publisher didn't supply a fully structured date -- a
    free-text `MedlineDate` (e.g. "2026 Sep-Oct"); NCBI's E-utilities DTD
    permits both forms interchangeably (live-run finding, PMID 42201346:
    `<PubDate><MedlineDate>2026 Sep-Oct</MedlineDate></PubDate>` with no
    `Year` sibling at all)."""
    if not isinstance(pub_date, dict):
        return None
    year_raw = pub_date.get("Year")
    if year_raw and str(year_raw).isdigit():
        return int(year_raw)
    return year_from_text(pub_date.get("MedlineDate"))


def _article_year(article: Dict[str, Any]) -> Optional[int]:
    """Journal/JournalIssue/PubDate (the print/journal publication date)
    first; only if that is wholly absent, fall back to ArticleDate (the
    electronic publication date -- DTD-fixed `DateType="Electronic"`,
    never Received/Accepted/Completed/Revised, which live only under
    PubmedData/History/PubMedPubDate and are never consulted here)."""
    journal_issue = (article.get("Journal", {}) or {}).get("JournalIssue", {}) or {}
    year = _year_from_pub_date(journal_issue.get("PubDate"))
    if year is not None:
        return year
    for article_date in as_list(article.get("ArticleDate")):
        year = _year_from_pub_date(article_date)
        if year is not None:
            return year
    return None


def _parse_article(art: Dict[str, Any]) -> Record:
    med = art.get("MedlineCitation", {}) or {}
    article = med.get("Article", {}) or {}
    pmid = text_of(med.get("PMID"))
    if not pmid:
        raise ValueError("PubMed article missing PMID")

    doi = None
    # ELocationID may be a single dict or a list of dicts
    for eloc in as_list(article.get("ELocationID")):
        if isinstance(eloc, dict) and eloc.get("@EIdType") == "doi":
            doi = eloc.get("#text")
    if doi is None:
        pubmed_data = art.get("PubmedData", {}) or {}
        article_ids = as_list((pubmed_data.get("ArticleIdList") or {}).get("ArticleId"))
        for aid in article_ids:
            if isinstance(aid, dict) and aid.get("@IdType") == "doi":
                doi = aid.get("#text")

    title = article.get("ArticleTitle", "")
    if isinstance(title, dict):
        title = title.get("#text", "")

    abstract_obj = article.get("Abstract", {}) or {}
    abstract_text = abstract_obj.get("AbstractText", "")
    if isinstance(abstract_text, list):
        abstract_text = " ".join(text_of(t) for t in abstract_text)
    elif isinstance(abstract_text, dict):
        abstract_text = abstract_text.get("#text", "")

    author_list = as_list((article.get("AuthorList") or {}).get("Author"))
    authors = []
    for a in author_list:
        if not isinstance(a, dict):
            continue
        last = a.get("LastName", "")
        fore = a.get("Initials", "") or a.get("ForeName", "")
        if last:
            authors.append(f"{last} {fore}".strip())

    year = _article_year(article)
    journal = (article.get("Journal", {}) or {}).get("Title")

    return Record(
        id=record_id("pubmed", pmid),
        source="pubmed",
        source_id=pmid,
        doi=doi,
        title=title or "",
        abstract=abstract_text or None,
        authors=authors,
        year=year,
        journal=journal,
        cited_by=None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def _pmid_of(art: Dict[str, Any], book: bool = False) -> str:
    """Best-effort PMID lookup for diagnostics only -- used to identify
    WHICH record a parse failure hit (Revision 5 point 2: previously only a
    bare `skipped` counter, PMID identity discarded entirely). Never raises;
    `"<unknown>"` if even this defensive lookup fails."""
    container = (art.get("BookDocument") if book else art.get("MedlineCitation")) or {}
    return text_of(container.get("PMID")) or "<unknown>"


def _parse_book_article(art: Dict[str, Any]) -> Record:
    """`<PubmedBookArticle>` (book chapters/monographs indexed in PubMed)
    uses a `BookDocument` element, structurally distinct from
    `_parse_article`'s `MedlineCitation`/`Article` shape -- previously
    entirely invisible to `as_list((parsed.get("PubmedArticleSet") or {}).
    get("PubmedArticle"))`, which only ever looks at the sibling
    `PubmedArticle` element (Revision 5 point 2, defect (b): a real
    candidate cause of the PubMed 300/303 gap)."""
    doc = art.get("BookDocument", {}) or {}
    pmid = text_of(doc.get("PMID"))
    if not pmid:
        raise ValueError("PubMed book article missing PMID")

    book = doc.get("Book", {}) or {}

    doi = None
    # Real DTD location: ELocationID lives under Book, NOT directly under
    # BookDocument (review finding P2, 2026-09-24; NCBI BookDocument
    # structure, https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/180101/
    # el-BookDocument.html).
    for eloc in as_list(book.get("ELocationID")):
        if isinstance(eloc, dict) and eloc.get("@EIdType") == "doi":
            doi = eloc.get("#text")
    if doi is None:
        # PubmedBookData is a sibling of BookDocument (mirroring PubmedData
        # for a regular PubmedArticle) and carries the definitive
        # ArticleIdList for the book-level doi/pmcid (NCBI PubmedBookArticle
        # structure, https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/190101/
        # el-PubmedBookArticle.html).
        book_data = art.get("PubmedBookData", {}) or {}
        for aid in as_list((book_data.get("ArticleIdList") or {}).get("ArticleId")):
            if isinstance(aid, dict) and aid.get("@IdType") == "doi":
                doi = aid.get("#text")
    if doi is None:
        for aid in as_list((doc.get("ArticleIdList") or {}).get("ArticleId")):
            if isinstance(aid, dict) and aid.get("@IdType") == "doi":
                doi = aid.get("#text")

    title = doc.get("ArticleTitle", "")
    if isinstance(title, dict):
        title = title.get("#text", "")
    if not title:
        # A whole-book citation (no specific chapter) has no ArticleTitle
        # at all -- Book/BookTitle is the DTD-correct fallback (review
        # finding P2, 2026-09-24).
        book_title = book.get("BookTitle", "")
        title = book_title.get("#text", "") if isinstance(book_title, dict) else book_title

    abstract_obj = doc.get("Abstract", {}) or {}
    abstract_text = abstract_obj.get("AbstractText", "")
    if isinstance(abstract_text, list):
        abstract_text = " ".join(text_of(t) for t in abstract_text)
    elif isinstance(abstract_text, dict):
        abstract_text = abstract_text.get("#text", "")

    author_list = as_list((doc.get("AuthorList") or {}).get("Author"))
    authors = []
    for a in author_list:
        if not isinstance(a, dict):
            continue
        last = a.get("LastName", "")
        fore = a.get("Initials", "") or a.get("ForeName", "")
        if last:
            authors.append(f"{last} {fore}".strip())

    year = _year_from_pub_date(book.get("PubDate"))
    journal = book.get("BookTitle")
    if isinstance(journal, dict):
        journal = journal.get("#text")

    return Record(
        id=record_id("pubmed", pmid),
        source="pubmed",
        source_id=pmid,
        doi=doi,
        title=title or "",
        abstract=abstract_text or None,
        authors=authors,
        year=year,
        journal=journal,
        cited_by=None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


class PubMedBackend:
    name = "pubmed"

    def search(
        self, request: SearchBackendRequest,
        *, diagnostics: Optional[PubMedBoundaryDiagnostics] = None,
        checkpoint: Optional[SearchCheckpointHooks] = None,
    ):
        client = HttpClient()
        api_key = os.environ.get("PUBMED_API_KEY")
        email = os.environ.get("PUBMED_EMAIL")

        fingerprint = compute_request_fingerprint("pubmed", request, "1")
        existing_checkpoint: Optional[SearchPageCheckpoint] = (
            checkpoint.load(fingerprint) if checkpoint is not None else None
        )
        if existing_checkpoint is not None and existing_checkpoint.completed:
            # Already finished this exact request in a prior run -- restart
            # safety means skip-if-exists here, same discipline as
            # `UpdateWaveSearchArtifactStore` at the orchestration layer, not
            # re-fetching a single record over the network.
            restored = [Record.from_dict(d) for d in existing_checkpoint.records_so_far]
            return finalize_result(
                "pubmed", request, restored, 0, existing_checkpoint.total_hits, None,
                source_exhausted=existing_checkpoint.source_exhausted,
                termination_reason=existing_checkpoint.termination_reason,
            )
        resumed = existing_checkpoint is not None and not existing_checkpoint.completed
        resume_state: Dict[str, Any] = existing_checkpoint.cursor_state if resumed else {}

        term = request.query
        if request.language:
            pubmed_language = _ISO_TO_PUBMED_LANGUAGE.get(request.language.lower())
            if pubmed_language is None:
                # Fail closed, pre-network: an unmapped code must never be
                # sent raw (it would silently match nothing, same failure
                # mode as the `[lang]`/ISO-code bug this replaces).
                return finalize_result(
                    "pubmed", request, [], 0, None,
                    f"pubmed: no [la] field mapping for language code {request.language!r}",
                )
            term = f"{term} AND {pubmed_language}[la]"

        # `usehistory=y` (Revision 5 point 2's "stable-search-history
        # mechanism"): once NCBI hands back a `webenv`/`querykey` pair, all
        # pages beyond the first are drawn from that fixed, server-side
        # result set via EFetch(WebEnv, query_key, retstart, retmax) -- NOT
        # by feeding query_key back into a second ESearch call, which NCBI
        # documents as performing a Boolean intersection against `term`,
        # not a page of the stored set (review finding P1, 2026-09-24; NCBI
        # E-Utilities "Application 3: Retrieving large datasets",
        # https://www.ncbi.nlm.nih.gov/books/NBK25499/). This also avoids
        # ESearch's own ~10,000-UID retrieval ceiling for later pages.
        base_params: Dict[str, Any] = {"db": "pubmed", "term": term, "retmode": "json", "usehistory": "y"}
        if request.date_from or request.date_to:
            base_params["datetype"] = "pdat"
            # Confirmed root cause of a real live run returning records from
            # 1994 through 2027 despite date_from="2024": NCBI's ESearch
            # silently ignores mindate/maxdate UNLESS BOTH are supplied
            # together -- sending mindate alone (the open-lower-bound case
            # this codebase's other update-wave code paths assume is valid)
            # applies no date filtering at all. An open bound is therefore
            # always paired with a deliberately far sentinel on the other
            # side, never omitted.
            base_params["mindate"] = f"{request.date_from}/01/01" if request.date_from else "1800/01/01"
            base_params["maxdate"] = f"{request.date_to}/12/31" if request.date_to else "3000/12/31"
        if api_key:
            base_params["api_key"] = api_key
        if email:
            base_params["email"] = email

        total_hits: Optional[int] = None
        failure_detail: Optional[str] = None
        # Declared here (not just at the final status-derivation block) so
        # a mid-pagination reconciliation failure (review findings P1,
        # 2026-09-24) can record its own accurate cause instead of falling
        # through to finalize_result's generic "malformed source record(s)"
        # default text.
        termination_reason: Optional[str] = None
        web_env: Optional[str] = None
        query_key: Optional[str] = None
        # Set ONLY on a genuinely empty response -- never inferred from a
        # merely short page (Revision 5 point 2, defect (a): a short page
        # used to be treated as proof of exhaustion, which a mid-run index
        # drift or an off-by-one could also produce; the fix is to keep
        # paging until the NEXT page is itself confirmed empty).
        cursor_confirmed_empty = False
        records: List[Record] = []
        skipped = 0
        seen_source_ids: Set[str] = set()
        resume_cursor: Dict[str, Any] = dict(resume_state)
        if resumed:
            # Restart-safe resume (Checkpoint 4): restore prior progress
            # verbatim instead of re-fetching every already-completed page.
            records = [Record.from_dict(d) for d in existing_checkpoint.records_so_far]
            seen_source_ids = {r.source_id for r in records}
            total_hits = existing_checkpoint.total_hits
            web_env = resume_state.get("web_env")
            query_key = resume_state.get("query_key")
            cursor_confirmed_empty = resume_state.get("cursor_confirmed_empty", False)

        def _save_page_checkpoint(*, completed: bool, source_exhausted: Optional[bool] = None,
                                   termination_reason_: Optional[str] = None) -> None:
            if checkpoint is None:
                return
            checkpoint.save(SearchPageCheckpoint(
                request_fingerprint=fingerprint,
                adapter="pubmed",
                adapter_config_version="1",
                cursor_state={
                    "web_env": web_env,
                    "query_key": query_key,
                    "offset": resume_cursor.get("offset", 0),
                    "have_hits": resume_cursor.get("have_hits", True),
                    "cursor_confirmed_empty": cursor_confirmed_empty,
                },
                records_so_far=tuple(r.to_dict() for r in records),
                total_hits=total_hits,
                page_count=resume_cursor.get("page_count", 0),
                completed=completed,
                source_exhausted=source_exhausted,
                termination_reason=termination_reason_,
            ))

        def _parse_efetch_batch(resp_text: str, requested_ids: Optional[Set[str]]) -> int:
            """Parses one EFetch XML response into `records`. Returns the
            number of items (articles + book articles) it contained.
            `requested_ids`, when given (the explicit id=-list EFetch path),
            enables per-batch missing/unexpected-PMID diagnostics (review
            finding P1 #2, 2026-09-24) -- `None` for the history-paged
            EFetch path, which has no independently-known expected id set
            to diff against."""
            nonlocal skipped, failure_detail, termination_reason
            parsed = xmltodict.parse(resp_text)
            article_set = parsed.get("PubmedArticleSet") or {}
            articles = as_list(article_set.get("PubmedArticle"))
            # `<PubmedBookArticle>` (book chapters/monographs) is a sibling
            # element previously never looked at -- silently invisible, not
            # even counted as skipped (Revision 5 point 2, defect (b)).
            book_articles = as_list(article_set.get("PubmedBookArticle"))
            if diagnostics is not None:
                diagnostics.items_per_efetch_page.append(len(articles) + len(book_articles))
                diagnostics.pubmed_article_count += len(articles)
                diagnostics.pubmed_book_article_count += len(book_articles)
            # Tag every returned item with its PMID BEFORE any record is
            # ingested (review finding P1, 2026-09-24): reconciling
            # identifiers only after ingestion let an unrequested PMID enter
            # `records` unfiltered, and a missing PMID surfaced only as a
            # generic "malformed source record(s)" count -- not what
            # actually happened, since EFetch simply never returned it.
            tagged = [
                (art, is_book, parse_fn, _pmid_of(art, book=is_book))
                for art, is_book, parse_fn in (
                    [(a, False, _parse_article) for a in articles]
                    + [(b, True, _parse_book_article) for b in book_articles]
                )
            ]
            returned_ids = {pmid for *_rest, pmid in tagged}
            unexpected_ids: Set[str] = set()
            if requested_ids is not None:
                missing = requested_ids - returned_ids
                # "<unknown>" (`_pmid_of`'s defensive fallback for an item
                # with no PMID at all) is never treated as a genuine
                # unexpected identity to exclude pre-parse -- it still goes
                # through normal parsing below, so its real parse failure
                # (not a reconciliation mismatch) is what gets recorded.
                unexpected_ids = returned_ids - requested_ids - {"<unknown>"}
                if diagnostics is not None:
                    diagnostics.missing_requested_pmids.extend(sorted(missing))
                    diagnostics.unexpected_efetch_pmids.extend(sorted(unexpected_ids))
                if missing or unexpected_ids:
                    # A requested PMID EFetch silently never returned, or an
                    # unrequested PMID it returned instead, is an explicit
                    # EFetch-omission/unexpected-ID failure cause -- never
                    # just an unexplained final-count mismatch, and never
                    # silently ingested (unexpected records are excluded
                    # below, before parsing).
                    skipped += len(missing) + len(unexpected_ids)
                    parts = []
                    if missing:
                        parts.append(f"{len(missing)} requested PMID(s) never returned: {sorted(missing)}")
                    if unexpected_ids:
                        parts.append(
                            f"{len(unexpected_ids)} unrequested PMID(s) returned and excluded: {sorted(unexpected_ids)}"
                        )
                    failure_detail = failure_detail or (
                        "pubmed efetch id reconciliation failed: " + "; ".join(parts)
                    )
                    if missing and unexpected_ids:
                        termination_reason = termination_reason or "efetch_missing_and_unexpected_pmids"
                    elif missing:
                        termination_reason = termination_reason or "efetch_missing_requested_pmids"
                    else:
                        termination_reason = termination_reason or "efetch_unexpected_pmids"
            for art, is_book, parse_fn, pmid in tagged:
                if pmid in unexpected_ids:
                    continue
                if pmid in seen_source_ids:
                    if diagnostics is not None:
                        diagnostics.duplicate_pmids_seen.append(pmid)
                    continue
                seen_source_ids.add(pmid)
                try:
                    records.append(parse_fn(art))
                except Exception:
                    skipped += 1
                    if diagnostics is not None:
                        diagnostics.parse_failed_pmids.append(pmid)
            return len(articles) + len(book_articles)

        def _fetch_by_id_list(id_list: List[str]) -> None:
            for i in range(0, len(id_list), PAGE_CAP):
                batch_ids = id_list[i : i + PAGE_CAP]
                fetch_params: Dict[str, Any] = {"db": "pubmed", "id": ",".join(batch_ids), "retmode": "xml"}
                if api_key:
                    fetch_params["api_key"] = api_key
                resp = client.get(EFETCH_URL, params=fetch_params)
                if diagnostics is not None:
                    diagnostics.requested_pmids_sent_to_efetch += len(batch_ids)
                _parse_efetch_batch(resp.text, requested_ids=set(batch_ids))

        page1_ids: List[str] = []
        have_hits = resume_cursor.get("have_hits", True) if resumed else True
        if not resumed:
            try:
                page1_params = dict(base_params)
                page1_params["retstart"] = 0
                page1_params["retmax"] = min(PAGE_CAP, request.max_results)
                resp = client.get(ESEARCH_URL, params=page1_params)
                data = resp.json().get("esearchresult", {})
                if data.get("count") is not None:
                    total_hits = int(data["count"])
                    if diagnostics is not None:
                        diagnostics.esearch_reported_count = total_hits
                # A malformed/unsupported term (bad field tag, unrecognised
                # syntax, ...) can come back as HTTP 200 with a genuine `idlist`
                # of [] (or even a partial one) alongside a non-empty
                # `errorlist`/`warninglist` -- previously ignored entirely,
                # silently reported as a clean 0-hit "complete" result rather
                # than the malformed-query failure it is.
                error_list = {k: v for k, v in (data.get("errorlist") or {}).items() if v}
                warning_list = {k: v for k, v in (data.get("warninglist") or {}).items() if v}
                if error_list or warning_list:
                    parts = []
                    if error_list:
                        parts.append(f"errorlist={error_list}")
                    if warning_list:
                        parts.append(f"warninglist={warning_list}")
                    failure_detail = (
                        f"pubmed esearch reported issues: "
                        f"{sanitize_remote_error_text('; '.join(parts))}"
                    )
                else:
                    page1_ids = data.get("idlist", [])
                    have_hits = bool(page1_ids)
                    if diagnostics is not None:
                        diagnostics.ids_per_esearch_page.append(len(page1_ids))
                    if data.get("webenv") and data.get("querykey"):
                        web_env, query_key = data["webenv"], data["querykey"]
                        if diagnostics is not None:
                            diagnostics.used_usehistory = True
            except Exception as exc:
                failure_detail = f"pubmed esearch error: {exc}"

        if failure_detail is None:
            limit = request.max_results if total_hits is None else min(total_hits, request.max_results)
            if not have_hits:
                # Zero hits, or the very first page came back genuinely
                # empty -- confirmed, not inferred (defect (a) applies from
                # the first page onward too).
                cursor_confirmed_empty = True
            else:
                if not resumed:
                    try:
                        _fetch_by_id_list(page1_ids)
                    except Exception as exc:
                        failure_detail = f"pubmed efetch error: {exc}"
                    offset = len(page1_ids)
                    resume_cursor["offset"] = offset
                    resume_cursor["have_hits"] = have_hits
                    # Review finding 7 (2026-09-25): page_count was persisted
                    # but never incremented, so every real checkpoint
                    # reported zero pages.
                    resume_cursor["page_count"] = resume_cursor.get("page_count", 0) + 1
                    _save_page_checkpoint(completed=False)
                else:
                    offset = resume_cursor.get("offset", len(records))

                if failure_detail is None and offset < limit:
                    if web_env and query_key:
                        # PRIMARY continuation path: further pages come
                        # straight from NCBI's history-set via EFetch, never
                        # via a second ESearch call (review finding P1,
                        # 2026-09-24).
                        try:
                            while offset < limit:
                                batch_size = min(PAGE_CAP, limit - offset)
                                fetch_params = {
                                    "db": "pubmed", "WebEnv": web_env, "query_key": query_key,
                                    "retstart": offset, "retmax": batch_size, "retmode": "xml",
                                }
                                if api_key:
                                    fetch_params["api_key"] = api_key
                                resp = client.get(EFETCH_URL, params=fetch_params)
                                page_count = _parse_efetch_batch(resp.text, requested_ids=None)
                                if page_count == 0:
                                    cursor_confirmed_empty = True
                                    break
                                if page_count != batch_size:
                                    # `retstart` addresses positions in
                                    # NCBI's stored history result set, not
                                    # items already parsed -- advancing by
                                    # anything other than the exact
                                    # requested window size risks either
                                    # overlap (re-parsing already-seen
                                    # positions) or a silent gap (skipping
                                    # unretrieved positions). A nonzero
                                    # page_count that doesn't match the
                                    # requested batch_size is therefore an
                                    # explicit partial retrieval -- stop
                                    # rather than keep paging on a
                                    # misaligned cursor (review finding P1,
                                    # 2026-09-24).
                                    failure_detail = failure_detail or (
                                        f"pubmed efetch history page misaligned: requested "
                                        f"retstart={offset} retmax={batch_size}, received "
                                        f"{page_count} item(s) -- stopping to avoid retstart drift"
                                    )
                                    termination_reason = termination_reason or "history_page_size_mismatch"
                                    break
                                offset += batch_size
                                resume_cursor["offset"] = offset
                                resume_cursor["page_count"] = resume_cursor.get("page_count", 0) + 1
                                _save_page_checkpoint(completed=False)
                        except Exception as exc:
                            failure_detail = f"pubmed efetch error: {exc}"
                    else:
                        # Defensive fallback -- NCBI did not grant a history
                        # set (an anomaly; virtually always present for
                        # usehistory=y). Degrades to plain stateless
                        # ESearch-idlist pagination (defect (a) fix
                        # retained: a short page never alone proves
                        # exhaustion) with its own explicit-id EFetch calls,
                        # WITHOUT ever reusing query_key inside a second
                        # ESearch call -- bounded by ESearch's own
                        # ~10,000-UID retrieval ceiling.
                        if resumed:
                            pmids = list(resume_cursor.get("fallback_pmids", []))
                            seen_ids: Set[str] = set(resume_cursor.get("fallback_seen_ids", []))
                            retstart = resume_cursor.get("fallback_retstart", offset)
                            already_fetched = len(pmids)
                        else:
                            pmids = list(page1_ids)
                            seen_ids = set(page1_ids)
                            retstart = offset
                            already_fetched = len(page1_ids)
                        try:
                            while len(pmids) < limit:
                                if total_hits is not None and retstart >= total_hits:
                                    break
                                page_params = dict(base_params)
                                page_params["retstart"] = retstart
                                page_params["retmax"] = min(PAGE_CAP, limit - len(pmids))
                                resp = client.get(ESEARCH_URL, params=page_params)
                                data = resp.json().get("esearchresult", {})
                                error_list = {k: v for k, v in (data.get("errorlist") or {}).items() if v}
                                warning_list = {k: v for k, v in (data.get("warninglist") or {}).items() if v}
                                if error_list or warning_list:
                                    parts = []
                                    if error_list:
                                        parts.append(f"errorlist={error_list}")
                                    if warning_list:
                                        parts.append(f"warninglist={warning_list}")
                                    failure_detail = (
                                        f"pubmed esearch reported issues: "
                                        f"{sanitize_remote_error_text('; '.join(parts))}"
                                    )
                                    break
                                batch = data.get("idlist", [])
                                if diagnostics is not None:
                                    diagnostics.ids_per_esearch_page.append(len(batch))
                                if not batch:
                                    cursor_confirmed_empty = True
                                    break
                                new_ids = [pmid for pmid in batch if pmid not in seen_ids]
                                if diagnostics is not None:
                                    diagnostics.duplicate_pmids_seen.extend(
                                        pmid for pmid in batch if pmid in seen_ids
                                    )
                                seen_ids.update(new_ids)
                                pmids.extend(new_ids)
                                retstart += len(batch)
                                resume_cursor["fallback_pmids"] = pmids
                                resume_cursor["fallback_seen_ids"] = sorted(seen_ids)
                                resume_cursor["fallback_retstart"] = retstart
                                resume_cursor["offset"] = retstart
                                # No `_save_page_checkpoint()` here: this
                                # loop only collects IDs via ESearch, it
                                # does not fetch records yet (the batch
                                # EFetch below is a single combined call,
                                # matching this adapter's established call
                                # sequence -- Checkpoint 2 regression
                                # tests pin ESearch(page2) -> ESearch(page3)
                                # -> EFetch(page2+page3 combined) as one
                                # batch, not per-page EFetch calls).
                                # `records`/`seen_source_ids` therefore stay
                                # unchanged until the batch fetch below
                                # succeeds; a resume before that point
                                # replays this ESearch-only loop from
                                # `resume_cursor`, never re-fetching a page
                                # already-parsed record.
                        except Exception as exc:
                            failure_detail = f"pubmed esearch error: {exc}"

                        new_pmids = pmids[already_fetched:]
                        if new_pmids and failure_detail is None:
                            try:
                                _fetch_by_id_list(new_pmids)
                            except Exception as exc:
                                failure_detail = f"pubmed efetch error: {exc}"
                            else:
                                already_fetched = len(pmids)
                                resume_cursor["fallback_pmids"] = pmids
                                resume_cursor["fallback_seen_ids"] = sorted(seen_ids)
                                resume_cursor["fallback_retstart"] = retstart
                                resume_cursor["offset"] = retstart
                                resume_cursor["page_count"] = resume_cursor.get("page_count", 0) + 1
                                _save_page_checkpoint(completed=False)

        if diagnostics is not None:
            diagnostics.unique_pmids_collected = len(seen_source_ids)
            diagnostics.cursor_confirmed_empty_page = cursor_confirmed_empty
            diagnostics.final_records_produced = len(records)

        retrieved = len(records)
        source_exhausted: Optional[bool] = None
        if failure_detail is None and skipped == 0:
            if total_hits is not None and retrieved == total_hits:
                # Positive proof (Revision 5 point 1(a)): every reported hit
                # was actually retrieved.
                source_exhausted = True
                termination_reason = "zero_hits" if total_hits == 0 else "total_hits_equals_retrieved"
            elif retrieved >= request.max_results:
                source_exhausted = False
                termination_reason = "max_results_cap_reached"
            elif cursor_confirmed_empty and total_hits is None:
                # The pagination cursor itself proved no more results exist
                # (a genuinely empty next page, Revision 5 point 1(b)) --
                # usable as exhaustion proof only when there is no reported
                # total_hits to contradict it.
                source_exhausted = True
                termination_reason = "cursor_exhausted_empty_next_page"
            elif cursor_confirmed_empty:
                # The cursor confirmed empty, but the earlier-reported
                # total_hits still does not match what was retrieved -- a
                # STRICTLY confirmed gap (e.g. duplicate IDs eaten by
                # within-run dedup, or genuine index drift), not merely an
                # unverified pagination end.
                source_exhausted = False
                termination_reason = "unexplained_retrieval_gap"
            else:
                # The esearch loop stopped without the next page being
                # confirmed empty (e.g. the count-based short-circuit fired
                # first) and without total_hits confirming exhaustion.
                source_exhausted = False
                termination_reason = "pagination_end_unverified"
        if failure_detail is None:
            # Only a clean, error-free termination is marked complete
            # (Checkpoint 4 design rule) -- an error path leaves the last
            # per-page checkpoint (`completed=False`) untouched so a
            # restart resumes pagination rather than treating a failed run
            # as finished.
            _save_page_checkpoint(
                completed=True, source_exhausted=source_exhausted,
                termination_reason_=termination_reason,
            )
        return finalize_result(
            "pubmed", request, records, skipped, total_hits, failure_detail,
            source_exhausted=source_exhausted, termination_reason=termination_reason,
        )
