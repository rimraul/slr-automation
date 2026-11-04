from __future__ import annotations

import slr_pipeline.search_backends.crossref as crossref_module
from slr_pipeline.search_backends.base import SearchBackendRequest, record_id


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def test_happy_path_parses_records_and_skips_malformed(fake_response, fake_http_client, monkeypatch):
    payload = {
        "message": {
            "total-results": 2,
            "next-cursor": None,
            "items": [
                {
                    "DOI": "10.1000/ABC",
                    "title": ["A Title"],
                    "abstract": "<jats:p>Some abstract</jats:p>",
                    "author": [{"given": "Jane", "family": "Doe"}],
                    "published": {"date-parts": [[2020]]},
                    "container-title": ["J Test"],
                    "is-referenced-by-count": 5,
                    "URL": "http://x",
                },
                {"title": ["missing DOI"]},  # missing native_key -> skipped
            ],
        }
    }
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(crossref_module, "HttpClient", client)

    result = crossref_module.CrossrefBackend().search(_request())

    assert result.retrieval_status == "partial"
    assert result.skipped_malformed == 1
    assert result.total_hits == 2
    rec = result.records[0]
    assert rec.id == record_id("crossref", "10.1000/abc")
    assert rec.doi == "10.1000/abc"
    assert rec.abstract == "Some abstract"
    assert rec.authors == ["Jane Doe"]
    assert rec.year == 2020
    assert rec.journal == "J Test"


def test_date_filter_forwarded(fake_response, fake_http_client, monkeypatch):
    payload = {"message": {"total-results": 0, "next-cursor": None, "items": []}}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(crossref_module, "HttpClient", client)

    crossref_module.CrossrefBackend().search(_request(date_from="2020", date_to="2021"))

    assert client.calls[0]["params"]["filter"] == "from-pub-date:2020-01-01,until-pub-date:2021-12-31"
