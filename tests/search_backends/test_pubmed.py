from __future__ import annotations

import slr_pipeline.search_backends.pubmed as pubmed_module
from slr_pipeline.search_backends.base import SearchBackendRequest, record_id

_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Test Title</ArticleTitle>
        <Abstract><AbstractText>Some abstract</AbstractText></Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName><Initials>J</Initials></Author>
        </AuthorList>
        <Journal>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
          <Title>J Test</Title>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def _articles_xml(pmids):
    articles = "".join(
        f"""<PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <ArticleTitle>Title {pmid}</ArticleTitle>
        <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue><Title>J Test</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>"""
        for pmid in pmids
    )
    return f'<?xml version="1.0"?>\n<PubmedArticleSet>{articles}</PubmedArticleSet>'


def test_pubmed_short_page_no_longer_stops_pagination_and_recovers_remaining_records(
    fake_response, fake_http_client, monkeypatch,
):
    """Checkpoint 2 regression (Revision 5 point 2, defect (a)): the ORIGINAL
    300/303 bug's leading candidate cause -- a short second ESearch page
    (100 < retmax=200) used to be treated as proof of exhaustion and
    immediately stopped pagination, even though a genuinely-empty-next-page
    check would have found 3 more PMIDs sitting at the next offset. With the
    fix, that short page no longer stops anything; a 3rd ESearch call is
    issued, finds the remaining 3 IDs, and the result becomes fully
    exhaustive (303/303), not silently short by 3."""
    page1_ids = [str(i) for i in range(1, 201)]
    page2_ids = [str(i) for i in range(201, 301)]
    page3_ids = [str(i) for i in range(301, 304)]
    # No webenv/querykey in any payload here -- exercises the no-history
    # fallback path (plain ESearch-idlist pagination), so call order is:
    # ESearch(page1) -> EFetch(page1 ids) -> ESearch(page2) -> ESearch(page3)
    # -> EFetch(page2+page3 ids combined into one <=PAGE_CAP batch).
    client = fake_http_client([
        fake_response(json_data={"esearchresult": {"count": "303", "idlist": page1_ids}}),
        fake_response(text_data=_articles_xml(page1_ids)),
        fake_response(json_data={"esearchresult": {"count": "303", "idlist": page2_ids}}),
        fake_response(json_data={"esearchresult": {"count": "303", "idlist": page3_ids}}),
        fake_response(text_data=_articles_xml(page2_ids + page3_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=1000))

    assert result.total_hits == 303
    assert result.returned == 303
    assert result.retrieval_status == "exhaustive"
    assert result.termination_reason == "total_hits_equals_retrieved"


def test_pubmed_page_count_increments_across_history_pages_and_is_not_stuck_at_zero(
    fake_response, fake_http_client, checkpoint_store, monkeypatch,
):
    """Review finding 7 (2026-09-25): every checkpoint save -- both the
    page-1 intra-loop save and every subsequent history-based EFetch page
    save -- must carry a genuinely incrementing `page_count`, not a value
    stuck at 0 forever."""
    page1_ids = [str(i) for i in range(1, 201)]
    page2_ids = [str(i) for i in range(201, 251)]
    client = fake_http_client([
        fake_response(json_data={
            "esearchresult": {"count": "250", "idlist": page1_ids, "webenv": "WE1", "querykey": "1"},
        }),
        fake_response(text_data=_articles_xml(page1_ids)),
        fake_response(text_data=_articles_xml(page2_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(
        _request(max_results=250), checkpoint=checkpoint_store.hooks_for("pubmed"),
    )

    assert result.returned == 250
    intermediate_saves = [cp for cp in checkpoint_store.saved if not cp.completed]
    assert [cp.page_count for cp in intermediate_saves] == [1, 2]
    final_saves = [cp for cp in checkpoint_store.saved if cp.completed]
    assert final_saves[-1].page_count == 2


def test_pubmed_confirmed_empty_next_page_with_total_hits_mismatch_is_unexplained_gap(
    fake_response, fake_http_client, monkeypatch,
):
    """The other candidate cause: the 3rd (post-short-page) ESearch call
    comes back genuinely empty (`idlist: []`), so the cursor itself confirms
    pagination end -- yet ESearch's own earlier `count=303` still does not
    match the 300 PMIDs actually collected. This is a STRICTLY confirmed
    gap (not merely an unverified short page), and must surface as
    'partial'/'unexplained_retrieval_gap', never 'capped'."""
    page1_ids = [str(i) for i in range(1, 201)]
    page2_ids = [str(i) for i in range(201, 301)]
    # No webenv/querykey -- no-history fallback path. Call order: ESearch
    # (page1) -> EFetch(page1) -> ESearch(page2) -> ESearch(page3, empty)
    # -> EFetch(page2 -- the only newly-discovered ids).
    client = fake_http_client([
        fake_response(json_data={"esearchresult": {"count": "303", "idlist": page1_ids}}),
        fake_response(text_data=_articles_xml(page1_ids)),
        fake_response(json_data={"esearchresult": {"count": "303", "idlist": page2_ids}}),
        fake_response(json_data={"esearchresult": {"count": "303", "idlist": []}}),
        fake_response(text_data=_articles_xml(page2_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=1000))

    assert result.total_hits == 303
    assert result.returned == 300
    assert result.retrieval_status == "partial"
    assert result.termination_reason == "unexplained_retrieval_gap"
    assert "303" in result.failure_detail
    assert "300" in result.failure_detail


def test_pubmed_duplicate_ids_across_pages_are_deduplicated_within_run(
    fake_response, fake_http_client, monkeypatch,
):
    """NCBI's stateless offset paging (no `usehistory`/`WebEnv` provided by
    the fake server here) is not guaranteed stable across requests -- a
    mid-run index shift can hand back an ID already seen on an earlier page.
    Such a duplicate must never be double-counted or double-fetched."""
    page1_ids = ["1", "2", "3"]
    page2_ids = ["3", "4", "5"]  # "3" repeats -- simulated index drift
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    # No webenv/querykey -- no-history fallback path. Call order: ESearch
    # (page1) -> EFetch(page1=[1,2,3]) -> ESearch(page2, dup "3") ->
    # EFetch(only the newly-discovered [4, 5]).
    client = fake_http_client([
        fake_response(json_data={"esearchresult": {"count": "5", "idlist": page1_ids}}),
        fake_response(text_data=_articles_xml(["1", "2", "3"])),
        fake_response(json_data={"esearchresult": {"count": "5", "idlist": page2_ids}}),
        fake_response(text_data=_articles_xml(["4", "5"])),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(
        _request(max_results=5), diagnostics=diagnostics,
    )

    assert diagnostics.duplicate_pmids_seen == ["3"]
    assert diagnostics.unique_pmids_collected == 5
    assert diagnostics.requested_pmids_sent_to_efetch == 5
    assert result.returned == 5


def test_pubmed_efetch_missing_requested_pmid_is_explicit_cause(
    fake_response, fake_http_client, monkeypatch,
):
    """Review finding P1, 2026-09-24: a requested PMID EFetch silently never
    returns must be recorded as an explicit EFetch-omission cause, not left
    to surface only as finalize_result's generic "malformed source
    record(s)" text."""
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    esearch_payload = {"esearchresult": {"count": "3", "idlist": ["1", "2", "3"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=_articles_xml(["1", "2"])),  # "3" never returned
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=3), diagnostics=diagnostics)

    assert result.returned == 2
    assert diagnostics.missing_requested_pmids == ["3"]
    assert diagnostics.unexpected_efetch_pmids == []
    assert result.termination_reason == "efetch_missing_requested_pmids"
    assert "3" in result.failure_detail
    assert result.retrieval_status == "partial"


def test_pubmed_efetch_unexpected_pmid_is_excluded_and_explicit_cause(
    fake_response, fake_http_client, monkeypatch,
):
    """Review finding P1, 2026-09-24: an EFetch response carrying a PMID
    that was never requested must be excluded from ingestion (never enter
    `records`) and recorded as an explicit unexpected-ID cause."""
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    esearch_payload = {"esearchresult": {"count": "2", "idlist": ["1", "2"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=_articles_xml(["1", "2", "99"])),  # "99" never requested
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=2), diagnostics=diagnostics)

    assert result.returned == 2
    assert all(r.source_id != "99" for r in result.records)
    assert diagnostics.unexpected_efetch_pmids == ["99"]
    assert diagnostics.missing_requested_pmids == []
    assert result.termination_reason == "efetch_unexpected_pmids"
    assert "99" in result.failure_detail
    assert result.retrieval_status == "partial"


def test_pubmed_efetch_replaced_pmid_is_both_missing_and_unexpected(
    fake_response, fake_http_client, monkeypatch,
):
    """Review finding P1, 2026-09-24 ("replacement" case): EFetch returning
    an unrequested PMID in place of a requested one keeps the returned
    count unchanged, but must still be caught as both a missing- and an
    unexpected-ID cause, not silently accepted as a like-for-like swap."""
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    esearch_payload = {"esearchresult": {"count": "2", "idlist": ["1", "2"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=_articles_xml(["1", "99"])),  # "2" replaced by "99"
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=2), diagnostics=diagnostics)

    assert result.returned == 1
    assert result.records[0].source_id == "1"
    assert diagnostics.missing_requested_pmids == ["2"]
    assert diagnostics.unexpected_efetch_pmids == ["99"]
    assert result.termination_reason == "efetch_missing_and_unexpected_pmids"
    assert "2" in result.failure_detail
    assert "99" in result.failure_detail
    assert result.retrieval_status == "partial"


def test_pubmed_history_page_size_mismatch_stops_pagination(
    fake_response, fake_http_client, monkeypatch,
):
    """Review finding P1, 2026-09-24: `retstart` addresses positions in
    NCBI's stored history result set, not items actually parsed --
    advancing it by the returned `page_count` instead of the requested
    window risks retstart drift (overlap or a silent gap). A nonzero page
    that returns a different count than the requested window must stop
    pagination as an explicit partial retrieval, not keep paging."""
    page1_ids = [str(i) for i in range(1, 101)]
    short_page_ids = [str(i) for i in range(101, 131)]  # 30 items, not the requested 50
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    client = fake_http_client([
        fake_response(json_data={
            "esearchresult": {"count": "150", "idlist": page1_ids, "webenv": "WE1", "querykey": "1"},
        }),
        fake_response(text_data=_articles_xml(page1_ids)),
        fake_response(text_data=_articles_xml(short_page_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=150), diagnostics=diagnostics)

    assert len(client.calls) == 3
    assert result.returned == 130
    assert result.retrieval_status == "partial"
    assert result.termination_reason == "history_page_size_mismatch"
    assert "retstart=100" in result.failure_detail
    assert "retmax=50" in result.failure_detail


def test_pubmed_usehistory_pages_via_efetch_not_esearch(
    fake_response, fake_http_client, monkeypatch,
):
    """Once NCBI hands back `webenv`/`querykey`, all pages beyond the first
    are drawn from that fixed history set via EFetch(WebEnv, query_key,
    retstart, retmax) -- never by feeding query_key back into a second
    ESearch call, which NCBI documents as a Boolean intersection against
    `term`, not a page of the stored set (review finding P1, 2026-09-24;
    NCBI E-Utilities "Application 3: Retrieving large datasets",
    https://www.ncbi.nlm.nih.gov/books/NBK25499/). Call order: ESearch
    (page1, gets webenv/querykey) -> EFetch(id=page1 ids) -> EFetch(WebEnv/
    query_key/retstart=100/retmax=50) -- no second ESearch call at all."""
    page1_ids = [str(i) for i in range(1, 101)]
    page2_ids = [str(i) for i in range(101, 151)]
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    client = fake_http_client([
        fake_response(json_data={
            "esearchresult": {"count": "150", "idlist": page1_ids, "webenv": "WE123", "querykey": "1"},
        }),
        fake_response(text_data=_articles_xml(page1_ids)),
        fake_response(text_data=_articles_xml(page2_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(
        _request(max_results=1000), diagnostics=diagnostics,
    )

    assert diagnostics.used_usehistory is True
    assert len(client.calls) == 3
    history_call = client.calls[2]
    assert history_call["url"] == pubmed_module.EFETCH_URL
    assert history_call["params"]["WebEnv"] == "WE123"
    assert history_call["params"]["query_key"] == "1"
    assert history_call["params"]["retstart"] == 100
    assert history_call["params"]["retmax"] == 50
    assert "term" not in history_call["params"]
    assert "id" not in history_call["params"]
    assert result.retrieval_status == "exhaustive"
    assert result.returned == 150


_BOOK_ARTICLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>1</PMID>
      <Article>
        <ArticleTitle>Journal Article</ArticleTitle>
        <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue><Title>J Test</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedBookArticle>
    <BookDocument>
      <PMID>2</PMID>
      <ArticleTitle>Book Chapter Title</ArticleTitle>
      <Abstract><AbstractText>Book abstract</AbstractText></Abstract>
      <AuthorList>
        <Author><LastName>Smith</LastName><Initials>A</Initials></Author>
      </AuthorList>
      <Book>
        <BookTitle>Some Book</BookTitle>
        <PubDate><Year>2020</Year></PubDate>
        <ELocationID EIdType="doi">10.1234/book.doi</ELocationID>
      </Book>
    </BookDocument>
    <PubmedBookData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1234/book.doi</ArticleId>
      </ArticleIdList>
    </PubmedBookData>
  </PubmedBookArticle>
</PubmedArticleSet>"""

_WHOLE_BOOK_CITATION_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedBookArticle>
    <BookDocument>
      <PMID>3</PMID>
      <Book>
        <BookTitle>Whole Book Title</BookTitle>
        <PubDate><Year>2019</Year></PubDate>
      </Book>
    </BookDocument>
    <PubmedBookData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.5678/wholebook.doi</ArticleId>
      </ArticleIdList>
    </PubmedBookData>
  </PubmedBookArticle>
</PubmedArticleSet>"""


def test_pubmed_book_article_is_parsed_and_counted_separately(
    fake_response, fake_http_client, monkeypatch,
):
    """Checkpoint 2 regression (Revision 5 point 2, defect (b)):
    `<PubmedBookArticle>` entries were previously entirely invisible --
    neither parsed nor even counted as skipped. They must now be parsed
    into records and counted separately from `<PubmedArticle>`."""
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    esearch_payload = {"esearchresult": {"count": "2", "idlist": ["1", "2"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=_BOOK_ARTICLE_XML),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=2), diagnostics=diagnostics)

    assert result.returned == 2
    assert diagnostics.pubmed_article_count == 1
    assert diagnostics.pubmed_book_article_count == 1
    book_record = next(r for r in result.records if r.source_id == "2")
    assert book_record.title == "Book Chapter Title"
    assert book_record.abstract == "Book abstract"
    assert book_record.authors == ["Smith A"]
    assert book_record.year == 2020
    assert book_record.journal == "Some Book"
    assert book_record.doi == "10.1234/book.doi"


def test_pubmed_whole_book_citation_falls_back_to_book_title(
    fake_response, fake_http_client, monkeypatch,
):
    """Review finding P2, 2026-09-24: a whole-book citation (no specific
    chapter) has no `ArticleTitle` at all per the official DTD example --
    `Book/BookTitle` is the correct fallback for the record title, and
    `PubmedBookData/ArticleIdList` is the correct doi source when `Book`
    carries no `ELocationID`."""
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    esearch_payload = {"esearchresult": {"count": "1", "idlist": ["3"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=_WHOLE_BOOK_CITATION_XML),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=1), diagnostics=diagnostics)

    assert result.returned == 1
    book_record = result.records[0]
    assert book_record.title == "Whole Book Title"
    assert book_record.journal == "Whole Book Title"
    assert book_record.doi == "10.5678/wholebook.doi"
    assert book_record.year == 2019


def test_pubmed_parse_failure_records_the_offending_pmid(
    fake_response, fake_http_client, monkeypatch,
):
    """Checkpoint 2 regression: a parse failure must identify WHICH PMID
    failed, not just bump a bare counter."""
    xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <Article><ArticleTitle>Missing PMID</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
    diagnostics = pubmed_module.PubMedBoundaryDiagnostics()
    esearch_payload = {"esearchresult": {"count": "1", "idlist": ["99"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=xml),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request(max_results=1), diagnostics=diagnostics)

    assert result.returned == 0
    assert diagnostics.parse_failed_pmids == ["<unknown>"]


def test_happy_path_esearch_then_efetch(fake_response, fake_http_client, monkeypatch):
    esearch_payload = {"esearchresult": {"count": "1", "idlist": ["12345"]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=_EFETCH_XML),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.termination_reason == "total_hits_equals_retrieved"
    assert result.total_hits == 1
    assert result.returned == 1
    rec = result.records[0]
    assert rec.id == record_id("pubmed", "12345")
    assert rec.source_id == "12345"
    assert rec.title == "Test Title"
    assert rec.abstract == "Some abstract"
    assert rec.authors == ["Doe J"]
    assert rec.year == 2021
    assert rec.journal == "J Test"
    assert rec.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_no_pmids_found_is_exhaustive_with_zero_records(fake_response, fake_http_client, monkeypatch):
    esearch_payload = {"esearchresult": {"count": "0", "idlist": []}}
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.termination_reason == "zero_hits"
    assert result.returned == 0


def test_language_filter_appended_to_term(fake_response, fake_http_client, monkeypatch):
    """`[lang]` is not a valid PubMed E-utilities field tag (the real tag is
    `[la]`), and PubMed's `Language` field expects a spelled-out name, not
    the ISO code -- confirmed root cause of a real live run returning 0
    PubMed hits (the malformed AND-clause matched nothing)."""
    esearch_payload = {"esearchresult": {"count": "0", "idlist": []}}
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    pubmed_module.PubMedBackend().search(_request(language="en"))

    assert "english[la]" in client.calls[0]["params"]["term"]
    assert "[lang]" not in client.calls[0]["params"]["term"]


def test_tiab_scoped_query_passed_through_verbatim(fake_response, fake_http_client, monkeypatch):
    """Checkpoint 3 query harmonization (2026-09-24): unlike openalex/scopus,
    PubMed needs no adapter-level change -- `[tiab]` field-tag scoping is
    authored directly into the query text upstream (nodes/search.py's
    `_PUBMED_SYNTAX_DBS` membership passes it straight through unmodified).
    This adapter must never strip or alter that scoping."""
    esearch_payload = {"esearchresult": {"count": "0", "idlist": []}}
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    query = "(cats[tiab] OR dogs[tiab]) AND pets[tiab]"
    pubmed_module.PubMedBackend().search(_request(query=query))

    assert client.calls[0]["params"]["term"] == query


def test_esearch_errorlist_is_not_a_clean_zero_hit_result(fake_response, fake_http_client, monkeypatch):
    """A malformed/unsupported term (bad field tag, unrecognised syntax,
    ...) can come back as HTTP 200 with `idlist: []` alongside a non-empty
    `errorlist` -- previously ignored entirely, silently reported as a
    clean 0-hit 'complete' result rather than the malformed-query failure
    it actually is."""
    esearch_payload = {
        "esearchresult": {
            "count": "0",
            "idlist": [],
            "errorlist": {"fieldnotfound": ["la"]},
        }
    }
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert result.returned == 0
    assert "fieldnotfound" in result.failure_detail


def test_esearch_warninglist_is_not_a_clean_zero_hit_result(fake_response, fake_http_client, monkeypatch):
    esearch_payload = {
        "esearchresult": {
            "count": "0",
            "idlist": [],
            "warninglist": {"quotedphrasenotfound": ['"healthcare access"']},
        }
    }
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert result.returned == 0
    assert "quotedphrasenotfound" in result.failure_detail


def test_esearch_errorlist_with_only_empty_values_is_still_clean(fake_response, fake_http_client, monkeypatch):
    """An `errorlist`/`warninglist` key present but with an empty list
    value is not a real problem (PubMed's own shape can include these
    empty) -- must not be misclassified as a failure."""
    esearch_payload = {
        "esearchresult": {
            "count": "1",
            "idlist": ["12345"],
            "errorlist": {"fieldnotfound": []},
            "warninglist": {"quotedphrasenotfound": []},
        }
    }
    client = fake_http_client([fake_response(json_data=esearch_payload), fake_response(text_data=_EFETCH_XML)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    result = pubmed_module.PubMedBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.termination_reason == "total_hits_equals_retrieved"
    assert result.returned == 1


def test_unmapped_language_code_is_pre_network_failure(never_called_http_client, monkeypatch):
    monkeypatch.setattr(pubmed_module, "HttpClient", never_called_http_client)

    result = pubmed_module.PubMedBackend().search(_request(language="xx"))

    assert result.retrieval_status == "failed"
    assert "xx" in result.failure_detail
    assert result.returned == 0


def test_date_from_only_still_sends_a_paired_maxdate(fake_response, fake_http_client, monkeypatch):
    """Live-run finding (2026-09-22): NCBI's ESearch silently ignores
    mindate/maxdate UNLESS BOTH are supplied together -- sending mindate
    alone applies no date filtering at all, confirmed by a real update-wave
    search with date_from="2024" returning PubMed records dated 1994-2027.
    An open lower bound must always be paired with a far-future maxdate
    sentinel, never send mindate on its own."""
    esearch_payload = {"esearchresult": {"count": "0", "idlist": []}}
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    pubmed_module.PubMedBackend().search(_request(date_from="2024"))

    params = client.calls[0]["params"]
    assert params["mindate"] == "2024/01/01"
    assert params["maxdate"] == "3000/12/31"


def test_date_to_only_still_sends_a_paired_mindate(fake_response, fake_http_client, monkeypatch):
    esearch_payload = {"esearchresult": {"count": "0", "idlist": []}}
    client = fake_http_client([fake_response(json_data=esearch_payload)])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)

    pubmed_module.PubMedBackend().search(_request(date_to="2023"))

    params = client.calls[0]["params"]
    assert params["mindate"] == "1800/01/01"
    assert params["maxdate"] == "2023/12/31"


def _efetch_article_xml(pmid: str, journal_pub_date_xml: str, article_date_xml: str = "") -> str:
    return f"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <ArticleTitle>Test Title</ArticleTitle>
        {article_date_xml}
        <Journal>
          <JournalIssue>{journal_pub_date_xml}</JournalIssue>
          <Title>J Test</Title>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _search_one_article(fake_response, fake_http_client, monkeypatch, pmid: str, xml: str):
    esearch_payload = {"esearchresult": {"count": "1", "idlist": [pmid]}}
    client = fake_http_client([
        fake_response(json_data=esearch_payload),
        fake_response(text_data=xml),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)
    return pubmed_module.PubMedBackend().search(_request())


def test_year_parsed_from_structured_pub_date_year(fake_response, fake_http_client, monkeypatch):
    xml = _efetch_article_xml("1", "<PubDate><Year>2021</Year></PubDate>")

    result = _search_one_article(fake_response, fake_http_client, monkeypatch, "1", xml)

    assert result.records[0].year == 2021


def test_year_falls_back_to_medline_date_free_text(fake_response, fake_http_client, monkeypatch):
    """No structured `Year` sibling at all -- only NCBI's free-text
    `MedlineDate` fallback form, e.g. a season/month range."""
    xml = _efetch_article_xml("2", "<PubDate><MedlineDate>2026 Sep-Oct</MedlineDate></PubDate>")

    result = _search_one_article(fake_response, fake_http_client, monkeypatch, "2", xml)

    assert result.records[0].year == 2026


def test_year_falls_back_to_electronic_article_date_when_journal_pub_date_absent(
    fake_response, fake_http_client, monkeypatch,
):
    """Journal/JournalIssue/PubDate wholly absent -- falls back to
    ArticleDate, the electronic publication date (DTD-fixed
    `DateType="Electronic"`)."""
    xml = _efetch_article_xml(
        "3", "<PubDate></PubDate>",
        article_date_xml='<ArticleDate DateType="Electronic"><Year>2026</Year><Month>05</Month><Day>28</Day></ArticleDate>',
    )

    result = _search_one_article(fake_response, fake_http_client, monkeypatch, "3", xml)

    assert result.records[0].year == 2026


def test_year_is_none_when_no_legitimate_publication_date_field_present(
    fake_response, fake_http_client, monkeypatch,
):
    """No `Year`, no `MedlineDate`, no `ArticleDate` -- the record is
    genuinely yearless (fail-closed downstream, never silently discarded
    here as malformed)."""
    xml = _efetch_article_xml("4", "<PubDate></PubDate>")

    result = _search_one_article(fake_response, fake_http_client, monkeypatch, "4", xml)

    assert result.records[0].year is None
    assert result.skipped_malformed == 0


def test_pmid_42201346_shaped_regression_medline_date_and_electronic_article_date(
    fake_response, fake_http_client, monkeypatch,
):
    """Live-run finding (2026-09-23), PMID 42201346: print publication
    "2026 Sep-Oct" (MedlineDate, no structured Year) plus an electronic
    ArticleDate of 2026-05-28 -- the adapter previously read only
    `JournalIssue/PubDate/Year` and returned `year=None`, tripping
    `enforce_record_temporal_window`'s fail-closed guard on a record that
    was never actually yearless."""
    xml = _efetch_article_xml(
        "42201346",
        "<PubDate><MedlineDate>2026 Sep-Oct</MedlineDate></PubDate>",
        article_date_xml='<ArticleDate DateType="Electronic"><Year>2026</Year><Month>05</Month><Day>28</Day></ArticleDate>',
    )

    result = _search_one_article(fake_response, fake_http_client, monkeypatch, "42201346", xml)

    assert result.records[0].year == 2026
