from __future__ import annotations

import slr_pipeline.search_backends.semantic_scholar as s2_module
from slr_pipeline.search_backends.base import SearchBackendRequest, record_id


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def test_happy_path_parses_records_and_skips_malformed(fake_response, fake_http_client, monkeypatch):
    payload = {
        "total": 2,
        "token": None,
        "data": [
            {
                "paperId": "p1",
                "externalIds": {"DOI": "10.1/x"},
                "title": "T",
                "abstract": "A",
                "authors": [{"name": "Jane Doe"}],
                "year": 2020,
                "venue": "V",
                "citationCount": 3,
                "url": "http://x",
            },
            {"paperId": None},  # missing native_key -> skipped
        ],
    }
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(s2_module, "HttpClient", client)

    result = s2_module.SemanticScholarBackend().search(_request())

    assert result.retrieval_status == "partial"
    assert result.skipped_malformed == 1
    assert result.total_hits == 2
    rec = result.records[0]
    assert rec.id == record_id("semantic_scholar", "p1")
    assert rec.doi == "10.1/x"
    assert rec.authors == ["Jane Doe"]
    assert rec.cited_by == 3


def test_api_key_sent_as_header_when_set(fake_response, fake_http_client, monkeypatch):
    payload = {"total": 0, "token": None, "data": []}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(s2_module, "HttpClient", client)
    monkeypatch.setenv("S2_API_KEY", "secret-key")

    s2_module.SemanticScholarBackend().search(_request())

    assert client.calls[0]["headers"] == {"x-api-key": "secret-key"}
