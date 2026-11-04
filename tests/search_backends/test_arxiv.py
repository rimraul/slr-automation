from __future__ import annotations

import slr_pipeline.search_backends.arxiv as arxiv_module
from slr_pipeline.search_backends.base import SearchBackendRequest, record_id

_FEED_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <title>Test Title</title>
    <summary>Test summary</summary>
    <author><name>Jane Doe</name></author>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
</feed>"""

_FEED_XML_NO_ID = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <title>No id here</title>
  </entry>
</feed>"""


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def test_happy_path_parses_entry(fake_response, fake_http_client, monkeypatch):
    client = fake_http_client([fake_response(text_data=_FEED_XML)])
    monkeypatch.setattr(arxiv_module, "HttpClient", client)

    result = arxiv_module.ArxivBackend().search(_request())

    assert result.retrieval_status == "capped"
    assert result.total_hits == 1
    rec = result.records[0]
    assert rec.id == record_id("arxiv", "http://arxiv.org/abs/2401.01234v1")
    assert rec.title == "Test Title"
    assert rec.abstract == "Test summary"
    assert rec.authors == ["Jane Doe"]
    assert rec.year == 2024
    assert rec.journal == "arXiv"


def test_missing_id_is_skipped(fake_response, fake_http_client, monkeypatch):
    client = fake_http_client([fake_response(text_data=_FEED_XML_NO_ID)])
    monkeypatch.setattr(arxiv_module, "HttpClient", client)

    result = arxiv_module.ArxivBackend().search(_request())

    assert result.skipped_malformed == 1
    assert result.returned == 0
    assert result.retrieval_status == "failed"   # 0 usable + skip -> failed, per finalize_result


def test_client_side_date_post_filter(fake_response, fake_http_client, monkeypatch):
    client = fake_http_client([fake_response(text_data=_FEED_XML)])
    monkeypatch.setattr(arxiv_module, "HttpClient", client)

    result = arxiv_module.ArxivBackend().search(_request(date_from="2025", date_to="2026"))

    # the entry is from 2024 -- outside [2025, 2026], filtered client-side,
    # source exhausted after one page -> zero usable records, no error.
    assert result.returned == 0
    assert result.retrieval_status == "capped"
