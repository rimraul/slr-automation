from __future__ import annotations

import slr_pipeline.search_backends.ieee as ieee_module
from slr_pipeline.search_backends.base import SearchBackendRequest, record_id


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def test_missing_api_key_fails_without_network_call(never_called_http_client, monkeypatch):
    monkeypatch.delenv("IEEE_API_KEY", raising=False)
    monkeypatch.setattr(ieee_module, "HttpClient", never_called_http_client)

    result = ieee_module.IeeeBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert result.failure_detail == "IEEE_API_KEY not set"
    assert result.returned == 0


def test_happy_path_parses_records_and_skips_malformed(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")
    payload = {
        "total_records": 2,
        "articles": [
            {
                "article_number": "12345",
                "doi": "10.1/x",
                "title": "A Title",
                "abstract": "An abstract",
                "authors": {"authors": [{"full_name": "Jane Doe"}]},
                "publication_year": 2020,
                "publication_title": "J Test",
                "citing_paper_count": 2,
                "html_url": "http://x",
            },
            {"title": "missing article_number"},  # missing native_key -> skipped
        ],
    }
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(ieee_module, "HttpClient", client)

    result = ieee_module.IeeeBackend().search(_request())

    assert result.retrieval_status == "partial"
    assert result.skipped_malformed == 1
    assert result.total_hits == 2
    rec = result.records[0]
    assert rec.id == record_id("ieee", "12345")
    assert rec.authors == ["Jane Doe"]
    assert rec.cited_by == 2
    assert client.calls[0]["params"]["apikey"] == "test-key"


def test_date_range_params_forwarded_only_when_set(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")
    payload = {"total_records": 0, "articles": []}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(ieee_module, "HttpClient", client)

    ieee_module.IeeeBackend().search(_request(date_from="2020", date_to="2021"))

    assert client.calls[0]["params"]["start_year"] == "2020"
    assert client.calls[0]["params"]["end_year"] == "2021"


def test_date_range_params_absent_when_unset(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("IEEE_API_KEY", "test-key")
    payload = {"total_records": 0, "articles": []}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(ieee_module, "HttpClient", client)

    ieee_module.IeeeBackend().search(_request())

    assert "start_year" not in client.calls[0]["params"]
    assert "end_year" not in client.calls[0]["params"]
