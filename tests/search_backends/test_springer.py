from __future__ import annotations

import slr_pipeline.search_backends.springer as springer_module
from slr_pipeline.search_backends.base import SearchBackendRequest, record_id


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def test_missing_api_key_fails_without_network_call(never_called_http_client, monkeypatch):
    monkeypatch.delenv("SPRINGER_API_KEY", raising=False)
    monkeypatch.setattr(springer_module, "HttpClient", never_called_http_client)

    result = springer_module.SpringerBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert result.failure_detail == "SPRINGER_API_KEY not set"
    assert result.returned == 0


def test_happy_path_parses_records(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SPRINGER_API_KEY", "test-key")
    payload = {
        "result": [{"total": "1"}],
        "records": [
            {
                "doi": "10.1000/ABC",
                "title": "A Title",
                "abstract": "An abstract",
                "creators": [{"creator": "Jane Doe"}],
                "publicationDate": "2020-05-01",
                "publicationName": "J Test",
                "url": [{"value": "http://x"}],
            }
        ],
    }
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(springer_module, "HttpClient", client)

    result = springer_module.SpringerBackend().search(_request())

    assert result.retrieval_status == "capped"
    assert result.total_hits == 1
    rec = result.records[0]
    assert rec.id == record_id("springer", "10.1000/abc")
    assert rec.doi == "10.1000/abc"
    assert rec.authors == ["Jane Doe"]
    assert rec.year == 2020
    assert rec.journal == "J Test"
    assert client.calls[0]["params"]["api_key"] == "test-key"


def test_language_and_date_appended_to_q(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SPRINGER_API_KEY", "test-key")
    payload = {"result": [{"total": "0"}], "records": []}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(springer_module, "HttpClient", client)

    springer_module.SpringerBackend().search(_request(language="en", date_from="2020", date_to="2021"))

    q = client.calls[0]["params"]["q"]
    assert "onlinedatefrom:2020-01-01" in q
    assert "onlinedateto:2021-12-31" in q
    assert "language:en" in q
