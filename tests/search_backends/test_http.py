"""`slr_pipeline.search_backends.http.HttpClient` -- unlike every adapter
test in this directory, this file does NOT replace `HttpClient` wholesale
with `_FakeHttpClient`. That fake stands in for the whole class, so it can
never exercise `HttpClient.get()`'s own body -- which is exactly why
`/code-review` (Track 2 step-20 round 2) found a real connection leak here
that no test had ever caught. These tests mock only the network boundary
(`requests.Session.get`, which `pytest-socket` blocks for real anyway) and
run `HttpClient.get()`'s actual logic on top of it."""
from __future__ import annotations

import pytest
import requests

from slr_pipeline.search_backends.http import HttpClient


def _error_response(status_code: int, url: str = "https://example.test/x") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp.url = url
    resp._content = b""
    return resp


def test_default_retry_statuses_forcelist_unchanged():
    # Every existing adapter calls HttpClient() with zero constructor args --
    # the new retry_statuses parameter must default to the exact same
    # forcelist as before (2026-09-23 OpenAlex live-run fix).
    client = HttpClient()
    adapter = client._session.get_adapter("https://example.test/x")
    assert sorted(adapter.max_retries.status_forcelist) == [429, 500, 502, 503, 504]


def test_retry_statuses_can_exclude_a_status_code():
    client = HttpClient(retry_statuses=frozenset({500, 502, 503, 504}))
    adapter = client._session.get_adapter("https://example.test/x")
    assert sorted(adapter.max_retries.status_forcelist) == [500, 502, 503, 504]


def test_get_closes_response_on_5xx_error_response(monkeypatch):
    # NOTE: this is a unit-level check of HttpClient.get()'s own try/except,
    # not a simulation of real forcelisted-5xx traffic. In production, 500 is
    # in HttpClient's own Retry(status_forcelist=...), so a persistent 500
    # from a real server is retried by the mounted HTTPAdapter and, once
    # exhausted, surfaces as requests.exceptions.RetryError raised INSIDE
    # `self._session.get(...)` itself -- no Response object is ever returned
    # for `raise_for_status()` to see, so there is nothing to leak on that
    # real path either way (verified empirically against installed
    # urllib3/requests). Monkeypatching `_session.get` to directly return a
    # 500 Response bypasses that retry machinery on purpose, to exercise the
    # close-on-error branch itself in isolation for any HTTPError-raising
    # status code, forcelisted or not.
    client = HttpClient()
    resp = _error_response(500)
    closed = {"called": False}
    monkeypatch.setattr(resp, "close", lambda: closed.__setitem__("called", True))
    monkeypatch.setattr(client._session, "get", lambda *a, **kw: resp)

    with pytest.raises(requests.HTTPError):
        client.get("https://example.test/x")

    assert closed["called"] is True


def test_get_closes_response_on_client_http_error(monkeypatch):
    # Same guard, a 4xx status this time -- raise_for_status() raises on
    # both 4xx and 5xx, and the leak-fix must close the connection either way.
    client = HttpClient()
    resp = _error_response(404)
    closed = {"called": False}
    monkeypatch.setattr(resp, "close", lambda: closed.__setitem__("called", True))
    monkeypatch.setattr(client._session, "get", lambda *a, **kw: resp)

    with pytest.raises(requests.HTTPError):
        client.get("https://example.test/x")

    assert closed["called"] is True


def test_get_does_not_close_response_on_success(monkeypatch):
    # The fix must only close on the error path -- a successful response's
    # body is still consumed further down (`iter_content`), so closing it
    # early in the try/except would break that.
    client = HttpClient()
    resp = _error_response(200)
    resp.raw = None
    monkeypatch.setattr(resp, "iter_content", lambda chunk_size: iter([b"ok"]))
    closed = {"called": False}
    monkeypatch.setattr(resp, "close", lambda: closed.__setitem__("called", True))
    monkeypatch.setattr(client._session, "get", lambda *a, **kw: resp)

    out = client.get("https://example.test/x")

    assert out.content == b"ok"
    assert closed["called"] is False


def test_get_closes_response_on_mid_stream_error(monkeypatch):
    # /code-review round 3: the round-2 fix only closed on raise_for_status()
    # raising -- a real connection that returns 200 but then drops or
    # truncates mid-body (ChunkedEncodingError etc. from iter_content) hit
    # the exact same leak, just via a different trigger. Cover it directly.
    client = HttpClient()
    resp = _error_response(200)

    def _broken_iter(chunk_size):
        yield b"partial"
        raise requests.exceptions.ChunkedEncodingError("connection dropped mid-body")

    monkeypatch.setattr(resp, "iter_content", _broken_iter)
    closed = {"called": False}
    monkeypatch.setattr(resp, "close", lambda: closed.__setitem__("called", True))
    monkeypatch.setattr(client._session, "get", lambda *a, **kw: resp)

    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        client.get("https://example.test/x")

    assert closed["called"] is True
