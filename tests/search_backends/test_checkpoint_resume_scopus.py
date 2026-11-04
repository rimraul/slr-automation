"""Checkpoint 4: Scopus adapter restart-safe pagination, same three
properties as pubmed's/openalex's checkpoint-resume test files: resume
without re-fetching a completed page, a completed checkpoint short-
circuiting with zero network calls, and a changed request fingerprint
never resuming stale progress."""

from __future__ import annotations

import slr_pipeline.search_backends.scopus as scopus_module
from slr_pipeline.evaluation.update_driver import FileSearchCheckpointStore
from slr_pipeline.search_backends.base import SearchBackendRequest


def _request(**overrides):
    base = dict(query="TITLE-ABS-KEY(cats)", max_results=1000, date_from=None, date_to=None,
                language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def _entry(eid):
    return {"eid": eid, "dc:title": f"Title {eid}", "dc:description": "abstract text"}


def _page(eids, next_cursor, total):
    return {"search-results": {
        "opensearch:totalResults": str(total),
        "entry": [_entry(e) for e in eids],
        "cursor": {"@next": next_cursor},
    }}


def _no_sleep(monkeypatch):
    monkeypatch.setattr(scopus_module.time, "sleep", lambda seconds: None)


def test_killed_run_resumes_from_last_checkpoint_without_refetching(
    fake_response, fake_http_client, tmp_path, monkeypatch,
):
    _no_sleep(monkeypatch)
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("scopus")

    page1_eids = [f"e{i}" for i in range(1, 26)]
    crashing_client = fake_http_client([
        fake_response(json_data=_page(page1_eids, "CURSOR2", total=30)),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", crashing_client)
    first = scopus_module.ScopusBackend().search(_request(), checkpoint=hooks)
    assert first.failure_detail is not None  # the simulated crash (no second page queued)

    remaining_eids = [f"e{i}" for i in range(26, 31)]
    resume_client = fake_http_client([
        fake_response(json_data=_page(remaining_eids, "CURSOR2", total=30)),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", resume_client)
    second = scopus_module.ScopusBackend().search(_request(), checkpoint=hooks)

    assert second.failure_detail is None
    assert second.total_hits == 30
    assert second.returned == 30
    assert second.retrieval_status == "exhaustive"
    assert len(resume_client.calls) == 1  # only the missing page, nothing re-fetched


def test_completed_checkpoint_short_circuits_with_zero_network_calls(
    fake_response, fake_http_client, never_called_http_client, tmp_path, monkeypatch,
):
    _no_sleep(monkeypatch)
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("scopus")

    eids = [f"e{i}" for i in range(1, 6)]
    client = fake_http_client([
        fake_response(json_data=_page(eids, next_cursor=None, total=5)),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    first = scopus_module.ScopusBackend().search(_request(), checkpoint=hooks)
    assert first.retrieval_status == "exhaustive"
    assert first.returned == 5

    monkeypatch.setattr(scopus_module, "HttpClient", never_called_http_client)
    second = scopus_module.ScopusBackend().search(_request(), checkpoint=hooks)
    assert second.returned == 5
    assert second.total_hits == 5
    assert second.retrieval_status == "exhaustive"


def test_changed_query_fingerprint_never_resumes_stale_progress(
    fake_response, fake_http_client, tmp_path, monkeypatch,
):
    _no_sleep(monkeypatch)
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("scopus")

    cats_eids = [f"e{i}" for i in range(1, 6)]
    client_a = fake_http_client([
        fake_response(json_data=_page(cats_eids, next_cursor=None, total=5)),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client_a)
    first = scopus_module.ScopusBackend().search(
        _request(query="TITLE-ABS-KEY(cats)"), checkpoint=hooks,
    )
    assert first.retrieval_status == "exhaustive"

    dogs_eids = [f"d{i}" for i in range(1, 4)]
    client_b = fake_http_client([
        fake_response(json_data=_page(dogs_eids, next_cursor=None, total=3)),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client_b)
    second = scopus_module.ScopusBackend().search(
        _request(query="TITLE-ABS-KEY(dogs)"), checkpoint=hooks,
    )

    assert second.retrieval_status == "exhaustive"
    assert second.total_hits == 3
    assert second.returned == 3
    assert len(client_b.calls) == 1  # a genuine fresh start from cursor="*"
    assert {r.source_id for r in second.records} == set(dogs_eids)
