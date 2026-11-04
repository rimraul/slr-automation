"""Shared fakes for slr_pipeline/search_backends/* unit tests -- no real
network call is ever made: every test monkeypatches the adapter module's
own `HttpClient` name with `_FakeHttpClient`, which returns pre-built
`_FakeResponse` pages in order and records every call it received."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from slr_pipeline.search_backends.base import SearchCheckpointHooks, SearchPageCheckpoint


class _FakeResponse:
    def __init__(self, json_data: Any = None, text_data: Optional[str] = None):
        self._json_data = json_data
        self.text = text_data if text_data is not None else ""

    def json(self):
        return self._json_data


class _FakeHttpClient:
    """Drop-in replacement for `search_backends.http.HttpClient` -- same
    `.get(url, *, params=None, headers=None)` shape, no network, no retry."""

    def __init__(self, pages: List[_FakeResponse]):
        self._pages = list(pages)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs):
        # Adapters do `client = HttpClient()` or `HttpClient(retry_statuses=...)`
        # -- this lets the SAME fake instance (with its own call log) be
        # substituted for the class regardless of constructor kwargs.
        return self

    def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if not self._pages:
            raise AssertionError(f"no more fake pages queued for {url!r}")
        return self._pages.pop(0)


class _NeverCalledHttpClient:
    def __call__(self, **kwargs):
        return self

    def get(self, *a, **kw):
        raise AssertionError("HttpClient.get must not be called")


class InMemoryCheckpointStore:
    """Minimal fake of `slr_pipeline.evaluation.update_driver.
    FileSearchCheckpointStore` -- keyed by `request_fingerprint` only,
    exactly matching the contract adapters actually rely on through
    `SearchCheckpointHooks`. Records every saved checkpoint (including
    intermediate, `completed=False` ones) in `.saved`, in order."""

    def __init__(self) -> None:
        self.saved: List[SearchPageCheckpoint] = []
        self._by_fingerprint: Dict[str, SearchPageCheckpoint] = {}

    def hooks_for(self, database: str) -> SearchCheckpointHooks:
        def load(fingerprint: str) -> Optional[SearchPageCheckpoint]:
            return self._by_fingerprint.get(fingerprint)

        def save(cp: SearchPageCheckpoint) -> None:
            self._by_fingerprint[cp.request_fingerprint] = cp
            self.saved.append(cp)

        return SearchCheckpointHooks(load=load, save=save)


@pytest.fixture
def fake_response():
    return _FakeResponse


@pytest.fixture
def fake_http_client():
    return _FakeHttpClient


@pytest.fixture
def never_called_http_client():
    return _NeverCalledHttpClient()


@pytest.fixture
def checkpoint_store():
    return InMemoryCheckpointStore()
