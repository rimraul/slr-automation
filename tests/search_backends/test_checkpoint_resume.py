"""Checkpoint 4: restart-safe pagination end-to-end, adapter + the real
``FileSearchCheckpointStore`` together. Proves (a) a killed/resumed run
picks up from the last saved page instead of re-fetching it, (b) a
completed checkpoint short-circuits with zero network calls, and (c) a
changed request fingerprint never resumes stale progress."""

from __future__ import annotations

import slr_pipeline.search_backends.pubmed as pubmed_module
from slr_pipeline.evaluation.update_driver import FileSearchCheckpointStore
from slr_pipeline.search_backends.base import SearchBackendRequest


def _request(**overrides):
    base = dict(query="cats", max_results=1000, date_from=None, date_to=None, language=None, mailto=None)
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


def test_killed_run_resumes_from_last_checkpoint_without_refetching(
    fake_response, fake_http_client, tmp_path, monkeypatch,
):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("pubmed")

    page1_ids = [str(i) for i in range(1, 201)]
    # First run: ESearch(page1, count=250, webenv/querykey) -> EFetch(page1
    # ids) -> the history-paged EFetch for the remaining 50 IDs is never
    # queued, so it raises inside the adapter's own try/except and is
    # recorded as a failure -- simulating a crash mid-pagination. The
    # per-page checkpoint saved right after EFetch(page1) survives untouched
    # (no completed=True save happens on a failure path).
    crashing_client = fake_http_client([
        fake_response(json_data={
            "esearchresult": {"count": "250", "idlist": page1_ids, "webenv": "WE1", "querykey": "1"},
        }),
        fake_response(text_data=_articles_xml(page1_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", crashing_client)
    first = pubmed_module.PubMedBackend().search(_request(), checkpoint=hooks)
    assert first.failure_detail is not None  # the simulated crash

    # Resume run: only the remaining history-paged EFetch page is queued.
    # If the adapter re-issued ESearch or re-fetched page 1's EFetch, this
    # fake client would raise "no more fake pages queued" or feed the wrong
    # response shape into the wrong parser.
    remaining_ids = [str(i) for i in range(201, 251)]
    resume_client = fake_http_client([
        fake_response(text_data=_articles_xml(remaining_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", resume_client)
    second = pubmed_module.PubMedBackend().search(_request(), checkpoint=hooks)

    assert second.failure_detail is None
    assert second.total_hits == 250
    assert second.returned == 250
    assert second.retrieval_status == "exhaustive"
    assert len(resume_client.calls) == 1  # only the missing page, nothing re-fetched


def test_completed_checkpoint_short_circuits_with_zero_network_calls(
    fake_response, fake_http_client, never_called_http_client, tmp_path, monkeypatch,
):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("pubmed")

    ids = [str(i) for i in range(1, 6)]
    client = fake_http_client([
        fake_response(json_data={"esearchresult": {"count": "5", "idlist": ids}}),
        fake_response(text_data=_articles_xml(ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client)
    first = pubmed_module.PubMedBackend().search(_request(), checkpoint=hooks)
    assert first.retrieval_status == "exhaustive"
    assert first.returned == 5

    monkeypatch.setattr(pubmed_module, "HttpClient", never_called_http_client)
    second = pubmed_module.PubMedBackend().search(_request(), checkpoint=hooks)
    assert second.returned == 5
    assert second.total_hits == 5
    assert second.retrieval_status == "exhaustive"


def test_changed_query_fingerprint_never_resumes_stale_progress(
    fake_response, fake_http_client, tmp_path, monkeypatch,
):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("pubmed")

    cats_ids = [str(i) for i in range(1, 6)]
    client_a = fake_http_client([
        fake_response(json_data={"esearchresult": {"count": "5", "idlist": cats_ids}}),
        fake_response(text_data=_articles_xml(cats_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client_a)
    first = pubmed_module.PubMedBackend().search(_request(query="cats"), checkpoint=hooks)
    assert first.retrieval_status == "exhaustive"

    # Different query -> different request_fingerprint. A full page set must
    # be provided again -- if the adapter incorrectly resumed the "cats"
    # checkpoint, this client's pages would either go unused (AssertionError
    # never raised, but the result would wrongly reflect the cats records)
    # or be consumed out of order.
    dogs_ids = [str(i) for i in range(101, 104)]
    client_b = fake_http_client([
        fake_response(json_data={"esearchresult": {"count": "3", "idlist": dogs_ids}}),
        fake_response(text_data=_articles_xml(dogs_ids)),
    ])
    monkeypatch.setattr(pubmed_module, "HttpClient", client_b)
    second = pubmed_module.PubMedBackend().search(_request(query="dogs"), checkpoint=hooks)

    assert second.retrieval_status == "exhaustive"
    assert second.total_hits == 3
    assert second.returned == 3
    assert len(client_b.calls) == 2  # ESearch + EFetch -- a genuine fresh start
    assert {r.source_id for r in second.records} == set(dogs_ids)
