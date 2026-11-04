from __future__ import annotations

import requests

import slr_pipeline.search_backends.scopus as scopus_module
from slr_pipeline.search_backends.base import DATE_UNSUPPORTED, LANGUAGE_UNSUPPORTED, SearchBackendRequest, record_id


def _request(**overrides):
    base = dict(query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def _entry(eid="2-s2.0-1", **overrides):
    base = {
        "eid": eid,
        "prism:doi": "10.1/x",
        "dc:title": "A Title",
        "dc:description": "An abstract",
        "author": [{"authname": "Doe J."}],
        "prism:coverDate": "2020-05-01",
        "prism:publicationName": "J Test",
        "authkeywords": "kw1 | kw2",
        "citedby-count": "3",
        "link": [{"@ref": "scopus", "@href": "http://x"}],
    }
    base.update(overrides)
    return base


def _page(entries, *, cursor_next=None, total="2"):
    search_results = {"entry": entries, "opensearch:totalResults": total}
    if cursor_next is not None:
        search_results["cursor"] = {"@next": cursor_next}
    return {"search-results": search_results}


class _FakeResponseWithStatus:
    """Local, scopus-only fake: unlike `conftest.py`'s shared `_FakeResponse`
    (which never simulates a non-2xx status), `_FakeHttpClientWithStatus.
    get()` raises for a non-200 `status_code`, mirroring what the real
    `HttpClient.get()` actually raises for each case: 401 (never in
    `HttpClient`'s own `status_forcelist`) reaches a caller as a genuine
    single-request `requests.exceptions.HTTPError` with `.response` set;
    429/500/502/503/504 (all IN `status_forcelist`, with `raise_on_status`
    at urllib3's default `True`) only ever reach a caller -- after
    `HttpClient`'s own internal `Retry` exhausts 3 attempts -- as
    `requests.exceptions.RetryError`, which carries no `.response`/
    `.status_code` at all (confirmed empirically against the real
    `HttpClient`, code-review round 1, 2026-09-10)."""

    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data


class _FakeHttpClientWithStatus:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def __call__(self):
        return self

    def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self._pages:
            raise AssertionError(f"no more fake pages queued for {url!r}")
        page = self._pages.pop(0)
        if page.status_code == 401:
            raise requests.exceptions.HTTPError(response=page)
        if page.status_code in (429, 500, 502, 503, 504):
            raise requests.exceptions.RetryError(f"too many {page.status_code} error responses")
        return page


def test_missing_api_key_fails_without_network_call(never_called_http_client, monkeypatch):
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.setattr(scopus_module, "HttpClient", never_called_http_client)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert result.failure_detail == "SCOPUS_API_KEY not set"
    assert result.returned == 0


def test_happy_path_single_page_parses_records(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([_entry()], cursor_next=None, total="1")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.total_hits == 1
    rec = result.records[0]
    assert rec.id == record_id("scopus", "2-s2.0-1")
    assert rec.source == "scopus"
    assert rec.source_id == "2-s2.0-1"
    assert rec.doi == "10.1/x"
    assert rec.title == "A Title"
    assert rec.abstract == "An abstract"
    assert rec.authors == ["Doe J."]
    assert rec.year == 2020
    assert rec.journal == "J Test"
    assert rec.keywords == ["kw1", "kw2"]
    assert rec.cited_by == 3
    assert rec.url == "http://x"
    assert client.calls[0]["params"]["query"] == "cats"
    assert client.calls[0]["params"]["cursor"] == "*"
    assert client.calls[0]["params"]["view"] == "STANDARD"
    assert client.calls[0]["headers"]["X-ELS-APIKey"] == "test-key"
    assert "X-ELS-Insttoken" not in client.calls[0]["headers"]
    assert result.access_mode == "STANDARD"


def test_singleton_author_and_link_collapsed_to_bare_dict_still_parses(
    fake_response, fake_http_client, monkeypatch
):
    """Scopus's XML-derived JSON can collapse a singleton repeatable field
    (a single author, a single link) to a bare dict instead of a
    one-element list -- must still parse, not raise/skip (round-5
    code-review finding)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    entry = _entry(author={"authname": "Doe J."}, link={"@ref": "scopus", "@href": "http://x"})
    payload = _page([entry], cursor_next=None, total="1")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.skipped_malformed == 0
    assert result.records[0].authors == ["Doe J."]
    assert result.records[0].url == "http://x"


def test_institutional_token_sets_complete_view_and_inst_header(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "inst-token")
    payload = _page([_entry()], total="1")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert client.calls[0]["params"]["view"] == "COMPLETE"
    assert client.calls[0]["headers"]["X-ELS-Insttoken"] == "inst-token"
    assert result.access_mode == "COMPLETE"


def test_multi_page_cursor_continuation(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    page1_entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    page1 = _page(page1_entries, cursor_next="cursor-2", total="26")
    page2 = _page([_entry(eid="2-s2.0-26")], cursor_next=None, total="26")
    client = fake_http_client([fake_response(json_data=page1), fake_response(json_data=page2)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request(max_results=100))

    assert result.returned == scopus_module._PER_PAGE + 1
    assert client.calls[0]["params"]["cursor"] == "*"
    assert client.calls[1]["params"]["cursor"] == "cursor-2"


def test_401_with_inst_token_retries_once_without_it(fake_response, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "inst-token")
    payload = _page([_entry()], total="1")
    client = _FakeHttpClientWithStatus([
        _FakeResponseWithStatus(status_code=401),
        _FakeResponseWithStatus(json_data=payload, status_code=200),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.returned == 1
    assert len(client.calls) == 2
    assert client.calls[0]["params"]["view"] == "COMPLETE"
    assert "X-ELS-Insttoken" in client.calls[0]["headers"]
    assert client.calls[1]["params"]["view"] == "STANDARD"
    assert "X-ELS-Insttoken" not in client.calls[1]["headers"]
    # retried the SAME cursor, not advanced
    assert client.calls[0]["params"]["cursor"] == client.calls[1]["params"]["cursor"] == "*"
    assert result.access_mode == "STANDARD (downgraded from COMPLETE after a 401 on the institutional token)"


def test_401_hard_fails_without_inst_token(fake_response, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    client = _FakeHttpClientWithStatus([_FakeResponseWithStatus(status_code=401)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert "scopus request failed" in result.failure_detail
    assert len(client.calls) == 1


def test_429_sleeps_and_retries_same_cursor(fake_response, monkeypatch):
    """A 429 that outlasts `HttpClient`'s own internal 3-attempt `Retry`
    surfaces to this adapter as `requests.exceptions.RetryError`, never a
    raise-able 429 `HTTPError` (see `_FakeResponseWithStatus`'s own
    docstring) -- this test exercises that real path, not an imagined
    single-HTTPError-per-429 interface."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([_entry()], total="1")
    client = _FakeHttpClientWithStatus([
        _FakeResponseWithStatus(status_code=429),
        _FakeResponseWithStatus(json_data=payload, status_code=200),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    sleep_calls = []
    monkeypatch.setattr(scopus_module.time, "sleep", lambda s: sleep_calls.append(s))

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.returned == 1
    assert len(client.calls) == 2
    assert client.calls[0]["params"]["cursor"] == client.calls[1]["params"]["cursor"] == "*"
    assert scopus_module._RATE_LIMIT_RETRY_SLEEP_S in sleep_calls


def test_429_gives_up_after_max_retry_attempts(fake_response, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    pages = [_FakeResponseWithStatus(status_code=429) for _ in range(scopus_module._MAX_RETRY_ATTEMPTS_PER_CURSOR + 1)]
    client = _FakeHttpClientWithStatus(pages)
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert "repeatedly rate-limited" in result.failure_detail
    assert len(client.calls) == scopus_module._MAX_RETRY_ATTEMPTS_PER_CURSOR + 1
    assert all(c["params"]["cursor"] == "*" for c in client.calls)


def test_malformed_entry_skip_counting(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([_entry(), {"dc:title": "missing eid"}], total="2")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "partial"
    assert result.skipped_malformed == 1
    assert result.returned == 1


def test_max_results_truncation(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    payload = _page(entries, cursor_next="cursor-2", total="25")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request(max_results=3))

    assert result.returned == 3
    assert len(client.calls) == 1  # loop condition (len(records) < max_results) stopped further pages
    assert client.calls[0]["params"]["count"] == 3  # only requests what's still needed (round-5 finding)


def test_date_range_params_forwarded_only_when_set(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    scopus_module.ScopusBackend().search(_request(date_from="2020", date_to="2021"))

    assert client.calls[0]["params"]["date"] == "2020-2021"


def test_title_abs_key_query_passed_through_verbatim(fake_response, fake_http_client, monkeypatch):
    """Checkpoint 3 query harmonization (2026-09-24): this adapter never wraps
    or validates field scope itself -- `nodes/search.py`'s
    `_STRICT_TRANSLATION_DBS` enforcement guarantees `request.query` already
    carries Scopus's native `TITLE-ABS-KEY(...)` syntax before it reaches
    here. Must pass it through unchanged, not silently unwrap/mangle it."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    query = 'TITLE-ABS-KEY("cats" OR "dogs")'
    scopus_module.ScopusBackend().search(_request(query=query))

    assert client.calls[0]["params"]["query"] == query


def test_date_from_only_uses_open_ended_pubyear_operator(fake_response, fake_http_client, monkeypatch):
    """`date_from` with no `date_to` must mean "from that year onward", not
    an exact single year -- Scopus's `date` param alone can only express an
    EXACT year or a closed range (donor adapter's own documented formats),
    never an open lower bound, so a bare `params["date"] = date_from` would
    silently narrow "papers since 2020" down to "papers in 2020 only"
    (round-6 code-review finding). Must use Scopus's documented `PUBYEAR >`
    query operator instead, and must NOT also set `params["date"]`."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    scopus_module.ScopusBackend().search(_request(date_from="2020"))

    assert "date" not in client.calls[0]["params"]
    assert client.calls[0]["params"]["query"] == "(cats) AND PUBYEAR > 2019"


def test_date_to_only_uses_open_ended_pubyear_operator(fake_response, fake_http_client, monkeypatch):
    """`date_to` with no `date_from` is a legitimate, unvalidated config
    state (`base.py`'s `SearchBackendRequest.__post_init__` never requires
    `date_from` when `date_to` is set) -- must mean "through that year",
    not an exact single year, for the same reason as the `date_from`-only
    case above (round-6 code-review finding)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    scopus_module.ScopusBackend().search(_request(date_to="2021"))

    assert "date" not in client.calls[0]["params"]
    assert client.calls[0]["params"]["query"] == "(cats) AND PUBYEAR < 2022"


def test_date_from_only_parenthesizes_or_containing_query(fake_response, fake_http_client, monkeypatch):
    """Without parenthesizing `request.query`, an unparenthesized `AND
    PUBYEAR > N` appended after a top-level OR would silently scope the
    date filter to only the last OR-clause (AND binds tighter than OR in
    this query-language family) -- round-7 code-review finding."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    scopus_module.ScopusBackend().search(_request(query='"phrase1" OR "phrase2"', date_from="2020"))

    assert client.calls[0]["params"]["query"] == '("phrase1" OR "phrase2") AND PUBYEAR > 2019'


def test_date_to_only_parenthesizes_or_containing_query(fake_response, fake_http_client, monkeypatch):
    """Symmetric to `test_date_from_only_parenthesizes_or_containing_query`
    for the `date_to`-only branch (round-8 code-review finding: only the
    `date_from`-only branch had this coverage)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    scopus_module.ScopusBackend().search(_request(query='"phrase1" OR "phrase2"', date_to="2021"))

    assert client.calls[0]["params"]["query"] == '("phrase1" OR "phrase2") AND PUBYEAR < 2022'


def test_date_param_absent_when_unset(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([], total="0")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    scopus_module.ScopusBackend().search(_request())

    assert "date" not in client.calls[0]["params"]


def test_singleton_top_level_entry_collapsed_to_bare_dict_still_parses(
    fake_response, fake_http_client, monkeypatch
):
    """`search-results.entry` is the same kind of Atom-feed repeatable
    field as `author`/`link` -- a single-hit query can collapse it to a
    bare dict instead of a one-element list. Without `as_list()`, `for
    entry in entries:` would iterate the dict's field-name strings,
    every one of which fails `_parse_entry`, silently turning a genuine
    single-result search into a reported `retrieval_status="failed"`
    with 0 records (round-6 code-review finding)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = {"search-results": {"entry": _entry(), "opensearch:totalResults": "1"}}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.returned == 1
    assert result.records[0].source_id == "2-s2.0-1"


def test_empty_result_stops_pagination(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = {"search-results": {"entry": [{"error": "Result set was empty"}]}}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "exhaustive"
    assert result.returned == 0
    assert result.total_hits is None


def test_non_empty_scopus_error_becomes_failure_not_clean_zero_hit(fake_response, fake_http_client, monkeypatch):
    """Only Scopus's OWN documented empty-result-set message ("Result set
    was empty") may be treated as a clean 0-hit `exhaustive` result. Any
    other content in the `error` key is a genuine error (bad request,
    malformed query, quota/auth issue, etc.) and must surface via
    `failure_detail` with a 'failed' retrieval_status -- confirmed
    root cause of a real live run silently treating a real Scopus error as
    a genuine empty search."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = {"search-results": {"entry": [{"error": "Invalid query syntax"}]}}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert result.returned == 0
    assert "Invalid query syntax" in result.failure_detail


def test_non_empty_scopus_error_is_bounded_and_credential_redacted(fake_response, fake_http_client, monkeypatch):
    """`base.py`'s `SearchBackendResult.failure_detail` contract requires
    the field be "sanitised: never a URL query string, never a token" --
    Scopus's own remote `error` text is outside this adapter's control, so
    any credential-looking key=value pair it contains must be redacted and
    the overall text bounded before being embedded verbatim."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    long_tail = "x" * 500
    error_text = f"Invalid request apikey=SECRETVALUE123 details: {long_tail}"
    payload = {"search-results": {"entry": [{"error": error_text}]}}
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.retrieval_status == "failed"
    assert "SECRETVALUE123" not in result.failure_detail
    assert "apikey=<redacted>" in result.failure_detail
    assert len(result.failure_detail) < len(error_text)


def test_non_empty_scopus_error_on_later_page_still_keeps_earlier_records(
    fake_response, fake_http_client, monkeypatch
):
    """A real error surfacing on page 2 (after page 1 already returned
    usable records) must be `partial`, carrying the earlier records plus
    the error detail -- not silently discarded as a clean stop."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    page1_entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    page1 = _page(page1_entries, cursor_next="cursor-2", total=str(scopus_module._PER_PAGE + 1))
    page2 = {"search-results": {"entry": [{"error": "Server error occurred"}]}}
    client = fake_http_client([fake_response(json_data=page1), fake_response(json_data=page2)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request(max_results=100))

    assert result.retrieval_status == "partial"
    assert result.returned == scopus_module._PER_PAGE
    assert "Server error occurred" in result.failure_detail


def test_scopus_registered_language_unsupported_not_date_unsupported():
    assert "scopus" in LANGUAGE_UNSUPPORTED
    assert "scopus" not in DATE_UNSUPPORTED


def test_terminal_cursor_equal_to_current_stops_pagination_cleanly(fake_response, fake_http_client, monkeypatch):
    """Exact-multiple-of-_PER_PAGE total hits: Scopus signals "no more
    pages" by returning the SAME cursor back as `@next`, not an absent
    `@next`. Without checking `next_cursor == cursor`, the adapter would
    re-fetch this page forever until `returned` exceeds the first-page-
    frozen `total_hits`, crashing `finalize_result()`'s pydantic validator
    (round-2 code-review Finding 1)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    payload = _page(entries, cursor_next="*", total=str(scopus_module._PER_PAGE))
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request(max_results=1000))

    assert result.retrieval_status == "exhaustive"
    assert result.returned == scopus_module._PER_PAGE
    assert len(client.calls) == 1  # did not re-fetch the terminal page


def test_retry_error_on_later_page_resets_counter_after_earlier_success(fake_response, monkeypatch):
    """`retry_attempts` must be scoped per-page, not global across the
    whole search: page 1 itself spends 2 of its 3-attempt budget on
    `RetryError`s before succeeding, then page 2 needs the FULL 3-attempt
    budget again. Without the post-success reset (scopus.py's
    `retry_attempts = 0` after a page fetch succeeds), page 2 would start
    at 2 already-spent attempts, exceed `_MAX_RETRY_ATTEMPTS_PER_CURSOR`
    after only 1 more retry, and fail closed instead of completing --
    this test fails if that reset line is removed (verified)."""
    assert scopus_module._MAX_RETRY_ATTEMPTS_PER_CURSOR == 3
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    page1_entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    page1 = _page(page1_entries, cursor_next="cursor-2", total="26")
    page2 = _page([_entry(eid="2-s2.0-26")], cursor_next=None, total="26")
    client = _FakeHttpClientWithStatus([
        _FakeResponseWithStatus(status_code=429),  # page 1, attempt 1
        _FakeResponseWithStatus(status_code=429),  # page 1, attempt 2
        _FakeResponseWithStatus(json_data=page1, status_code=200),  # page 1, succeeds
        _FakeResponseWithStatus(status_code=429),  # page 2, attempt 1
        _FakeResponseWithStatus(status_code=429),  # page 2, attempt 2
        _FakeResponseWithStatus(status_code=429),  # page 2, attempt 3 -- needs the full budget
        _FakeResponseWithStatus(json_data=page2, status_code=200),  # page 2, succeeds
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request(max_results=100))

    assert result.retrieval_status == "exhaustive"
    assert result.returned == scopus_module._PER_PAGE + 1
    assert len(client.calls) == 7
    assert all(c["params"]["cursor"] == "*" for c in client.calls[:3])
    assert all(c["params"]["cursor"] == "cursor-2" for c in client.calls[3:])


def test_authkeywords_single_multiword_phrase_not_space_split(fake_response, fake_http_client, monkeypatch):
    """No `"|"` present => the whole string is one keyword phrase, never
    shredded on spaces (round-2 code-review Finding 2)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([_entry(authkeywords="machine learning")], total="1")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)

    result = scopus_module.ScopusBackend().search(_request())

    assert result.records[0].keywords == ["machine learning"]


def test_rate_limit_delay_applied_before_every_request(fake_response, fake_http_client, monkeypatch):
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    payload = _page([_entry()], total="1")
    client = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)
    sleep_calls = []
    monkeypatch.setattr(scopus_module.time, "sleep", lambda s: sleep_calls.append(s))

    scopus_module.ScopusBackend().search(_request())

    assert sleep_calls == [scopus_module._RATE_LIMIT_DELAY_S]


def test_requested_access_tier_is_bound_into_checkpoint_identity(
    fake_response, fake_http_client, checkpoint_store, monkeypatch,
):
    """Review finding 6: a checkpoint completed under one requested access
    tier must never be reused for a search run under a DIFFERENT requested
    tier -- the fingerprint must differ, forcing a fresh fetch instead of
    silently returning a cached STANDARD result as if it were COMPLETE (or
    vice versa)."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)
    store = checkpoint_store

    payload = _page([_entry()], cursor_next=None, total="1")
    client1 = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client1)
    result1 = scopus_module.ScopusBackend().search(_request(), checkpoint=store.hooks_for("scopus"))
    assert result1.access_mode == "STANDARD"
    assert len(client1.calls) == 1

    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "inst-token")
    client2 = fake_http_client([fake_response(json_data=payload)])
    monkeypatch.setattr(scopus_module, "HttpClient", client2)
    result2 = scopus_module.ScopusBackend().search(_request(), checkpoint=store.hooks_for("scopus"))

    # A DIFFERENT requested tier must trigger a real network call, not a
    # zero-network cached-completed short-circuit under the wrong tier.
    assert len(client2.calls) == 1
    assert result2.access_mode == "COMPLETE"


def test_intra_loop_checkpoint_reports_cumulative_worst_access_mode(fake_response, checkpoint_store, monkeypatch):
    """Review finding 6: `page_count` is not the only thing an intra-loop
    save must get right -- every INTERMEDIATE (`completed=False`) checkpoint
    save after a downgrade must already report the downgraded tier, not
    `None`/`COMPLETE`, so a crash right after the downgrade can never leave
    a resumable checkpoint that looks like it was still on the institutional
    token."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "inst-token")
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)
    store = checkpoint_store

    # A FULL page (== _PER_PAGE entries) is required here so pagination
    # advances via the terminal-cursor check rather than the ambiguous
    # short-page break, which would exit before the intra-loop checkpoint
    # save this test targets.
    page1_entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    page1 = _page(page1_entries, cursor_next="cursor-2", total=str(scopus_module._PER_PAGE + 1))
    page2 = _page([_entry(eid="2-s2.0-last")], cursor_next=None, total=str(scopus_module._PER_PAGE + 1))
    client = _FakeHttpClientWithStatus([
        _FakeResponseWithStatus(status_code=401),
        _FakeResponseWithStatus(json_data=page1, status_code=200),
        _FakeResponseWithStatus(json_data=page2, status_code=200),
    ])
    monkeypatch.setattr(scopus_module, "HttpClient", client)

    result = scopus_module.ScopusBackend().search(_request(max_results=100), checkpoint=store.hooks_for("scopus"))

    assert result.access_mode == "STANDARD (downgraded from COMPLETE after a 401 on the institutional token)"
    # At least one intermediate (completed=False) checkpoint was saved
    # between page 1 and page 2, after the downgrade -- it must already
    # carry the downgraded access mode, not None.
    intermediate_saves = [cp for cp in store.saved if not cp.completed]
    assert intermediate_saves, "expected at least one intra-loop checkpoint save"
    for cp in intermediate_saves:
        assert cp.cursor_state["access_mode"] == (
            "STANDARD (downgraded from COMPLETE after a 401 on the institutional token)"
        )


def test_page_count_increments_across_pages_and_is_not_stuck_at_zero(
    fake_response, fake_http_client, checkpoint_store, monkeypatch,
):
    """Review finding 7: page_count was persisted but never incremented, so
    every real checkpoint reported zero pages regardless of how many pages
    were actually fetched."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)
    store = checkpoint_store

    page1_entries = [_entry(eid=f"2-s2.0-{i}") for i in range(scopus_module._PER_PAGE)]
    page1 = _page(page1_entries, cursor_next="cursor-2", total=str(scopus_module._PER_PAGE + 1))
    page2 = _page([_entry(eid="2-s2.0-last")], cursor_next=None, total=str(scopus_module._PER_PAGE + 1))
    client = fake_http_client([fake_response(json_data=page1), fake_response(json_data=page2)])
    monkeypatch.setattr(scopus_module, "HttpClient", client)

    result = scopus_module.ScopusBackend().search(_request(max_results=100), checkpoint=store.hooks_for("scopus"))

    assert result.retrieval_status == "exhaustive"
    intermediate_saves = [cp for cp in store.saved if not cp.completed]
    assert [cp.page_count for cp in intermediate_saves] == [1]
    final_saves = [cp for cp in store.saved if cp.completed]
    assert final_saves[-1].page_count == 2


def test_resume_after_downgrade_never_reports_complete_even_with_token_present(
    fake_response, checkpoint_store, monkeypatch,
):
    """Review finding 6: a resumed run must never report `COMPLETE` when
    earlier `records_so_far` were actually fetched under a downgraded
    STANDARD view, even though the institutional token is present again
    this run and no 401 recurs."""
    monkeypatch.setenv("SCOPUS_API_KEY", "test-key")
    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "inst-token")
    monkeypatch.setattr(scopus_module.time, "sleep", lambda *_a, **_kw: None)
    store = checkpoint_store

    request = _request(max_results=100)
    fingerprint = scopus_module.compute_request_fingerprint(
        "scopus", request, "1", identity_extra={"requested_access_tier": "COMPLETE"},
    )
    from slr_pipeline.search_backends.base import SearchPageCheckpoint

    prior_checkpoint = SearchPageCheckpoint(
        request_fingerprint=fingerprint,
        adapter="scopus",
        adapter_config_version="1",
        cursor_state={"cursor": "cursor-2", "total_hits": 2, "access_mode":
                      "STANDARD (downgraded from COMPLETE after a 401 on the institutional token)"},
        records_so_far=({"id": record_id("scopus", "2-s2.0-1"), "source": "scopus", "source_id": "2-s2.0-1",
                          "title": "A Title"},),
        total_hits=2,
        completed=False,
    )
    hooks = store.hooks_for("scopus")
    hooks.save(prior_checkpoint)

    page2 = _page([_entry(eid="2-s2.0-2")], cursor_next=None, total="2")

    class _SinglePageClient:
        def __init__(self, page):
            self._page = page
            self.calls = []

        def __call__(self):
            return self

        def get(self, url, *, params=None, headers=None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            return fake_response(json_data=self._page)

    single_client = _SinglePageClient(page2)
    monkeypatch.setattr(scopus_module, "HttpClient", single_client)

    result = scopus_module.ScopusBackend().search(request, checkpoint=hooks)

    assert result.access_mode == "STANDARD (downgraded from COMPLETE after a 401 on the institutional token)"
    assert len(single_client.calls) == 1
    assert single_client.calls[0]["params"]["cursor"] == "cursor-2"
