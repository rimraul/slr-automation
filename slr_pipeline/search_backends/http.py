"""
slr_pipeline.search_backends.http — the one shared HTTP client every real
adapter uses (Track 2 plan §2.4, "`HttpClient`" bullet, verbatim contract).

No import-time state, no module-level client -- each adapter constructs its
own `HttpClient()` instance inside `.search()`.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 30
MAX_RESPONSE_BYTES = 25 * 1024 * 1024   # 25 MiB streamed response cap
_CHUNK_SIZE = 65_536
DEFAULT_RETRY_STATUSES: FrozenSet[int] = frozenset({429, 500, 502, 503, 504})


class HttpClient:
    """One shared `requests.Session` per instance; HTTPS-only;
    `timeout=(connect=5, read=30)`; `Retry(total=3, backoff_factor=1.0,
    status_forcelist=<retry_statuses>, allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True)`; a 25 MiB streamed response cap.

    `retry_statuses` defaults to `DEFAULT_RETRY_STATUSES`
    (`{429,500,502,503,504}`) -- every existing caller is unaffected. A
    status code excluded from this set is never auto-retried at the
    transport level, so it reaches the caller as a genuine single-request
    `requests.exceptions.HTTPError` with `.response` (status, headers)
    intact, rather than a header-less `requests.exceptions.RetryError` once
    retries exhaust. OpenAlex's adapter excludes 429 for exactly this reason
    -- it needs the real `X-RateLimit-*` response headers, which only a
    genuine `HTTPError` carries."""

    def __init__(self, *, retry_statuses: FrozenSet[int] = DEFAULT_RETRY_STATUSES) -> None:
        self._session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=list(retry_statuses),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)

    def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        if urlsplit(url).scheme != "https":
            raise ValueError(f"HttpClient is HTTPS-only, got: {url!r}")
        response = self._session.get(
            url,
            params=params,
            headers=headers,
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            stream=True,
        )
        # stream=True leaves the underlying urllib3 connection checked out
        # of the pool until the Response is closed or GC'd. Any failure
        # from here on -- a non-2xx status, the 25 MiB cap tripping, or the
        # connection dropping/truncating mid-download (ChunkedEncodingError
        # etc. from iter_content itself) -- must close it before propagating,
        # or that connection leaks for the lifetime of this HttpClient.
        try:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError(
                        f"response from {url!r} exceeded the {MAX_RESPONSE_BYTES}-byte cap"
                    )
        except Exception:
            response.close()
            raise
        # Streamed manually above (so the cap check runs mid-download);
        # store the accumulated bytes so .content/.text/.json() on the
        # returned Response behave exactly as a non-streamed call would.
        response._content = bytes(body)
        return response

    def close(self) -> None:
        self._session.close()
