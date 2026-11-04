"""Checkpoint 4: OpenAlex adapter restart-safe pagination, same three
properties as pubmed's ``test_checkpoint_resume.py``: resume without
re-fetching a completed page, a completed checkpoint short-circuiting with
zero network calls, and a changed request fingerprint never resuming stale
progress."""

from __future__ import annotations

import slr_pipeline.search_backends.openalex as openalex_module
from slr_pipeline.evaluation.update_driver import FileSearchCheckpointStore
from slr_pipeline.search_backends.base import SearchBackendRequest


def _request(**overrides):
    base = dict(query="cats", max_results=1000, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def _work(native_id):
    return {"id": f"https://openalex.org/W{native_id}", "doi": None, "title": f"Title {native_id}",
             "abstract_inverted_index": None, "authorships": [], "publication_year": 2021,
             "primary_location": None, "cited_by_count": 0}


def _page(ids, next_cursor, count=None):
    return {"meta": {"count": count, "next_cursor": next_cursor}, "results": [_work(i) for i in ids]}


def test_killed_run_resumes_from_last_checkpoint_without_refetching(
    fake_response, fake_http_client, tmp_path, monkeypatch,
):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("openalex")

    # First run: page 1 (100 results, cursor advances) is fetched and
    # checkpointed, then the run "crashes" -- no second page queued, so
    # the adapter's own try/except records it as a failure. The per-page
    # checkpoint saved right after `cursor = next_cursor` survives untouched.
    page1_ids = list(range(1, 101))
    crashing_client = fake_http_client([
        fake_response(json_data=_page(page1_ids, next_cursor="CURSOR2", count=150)),
    ])
    monkeypatch.setattr(openalex_module, "HttpClient", crashing_client)
    first = openalex_module.OpenAlexBackend().search(_request(), checkpoint=hooks)
    assert first.failure_detail is not None  # the simulated crash

    # Resume run: only the missing second page is queued. If the adapter
    # incorrectly restarted at cursor="*", this fake client would raise
    # "no more fake pages queued".
    remaining_ids = list(range(101, 151))
    resume_client = fake_http_client([
        fake_response(json_data=_page(remaining_ids, next_cursor=None, count=150)),
    ])
    monkeypatch.setattr(openalex_module, "HttpClient", resume_client)
    second = openalex_module.OpenAlexBackend().search(_request(), checkpoint=hooks)

    assert second.failure_detail is None
    assert second.total_hits == 150
    assert second.returned == 150
    assert second.retrieval_status == "exhaustive"
    assert len(resume_client.calls) == 1  # only the missing page, nothing re-fetched


def test_completed_checkpoint_short_circuits_with_zero_network_calls(
    fake_response, fake_http_client, never_called_http_client, tmp_path, monkeypatch,
):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("openalex")

    ids = list(range(1, 6))
    client = fake_http_client([
        fake_response(json_data=_page(ids, next_cursor=None, count=5)),
    ])
    monkeypatch.setattr(openalex_module, "HttpClient", client)
    first = openalex_module.OpenAlexBackend().search(_request(), checkpoint=hooks)
    assert first.retrieval_status == "exhaustive"
    assert first.returned == 5

    monkeypatch.setattr(openalex_module, "HttpClient", never_called_http_client)
    second = openalex_module.OpenAlexBackend().search(_request(), checkpoint=hooks)
    assert second.returned == 5
    assert second.total_hits == 5
    assert second.retrieval_status == "exhaustive"


def test_changed_query_fingerprint_never_resumes_stale_progress(
    fake_response, fake_http_client, tmp_path, monkeypatch,
):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("openalex")

    cats_ids = list(range(1, 6))
    client_a = fake_http_client([
        fake_response(json_data=_page(cats_ids, next_cursor=None, count=5)),
    ])
    monkeypatch.setattr(openalex_module, "HttpClient", client_a)
    first = openalex_module.OpenAlexBackend().search(_request(query="cats"), checkpoint=hooks)
    assert first.retrieval_status == "exhaustive"

    # Different query -> different request_fingerprint -- a full page set
    # must be provided again from cursor="*".
    dogs_ids = list(range(101, 104))
    client_b = fake_http_client([
        fake_response(json_data=_page(dogs_ids, next_cursor=None, count=3)),
    ])
    monkeypatch.setattr(openalex_module, "HttpClient", client_b)
    second = openalex_module.OpenAlexBackend().search(_request(query="dogs"), checkpoint=hooks)

    assert second.retrieval_status == "exhaustive"
    assert second.total_hits == 3
    assert second.returned == 3
    assert len(client_b.calls) == 1  # a genuine fresh start from cursor="*"
    assert {r.source_id for r in second.records} == {f"W{i}" for i in dogs_ids}
