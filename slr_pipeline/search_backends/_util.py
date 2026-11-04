"""Small helpers shared by two or more adapters -- not part of the public
`search_backends` API surface (leading underscore), never imported outside
this package."""
from __future__ import annotations

import re
from typing import Any, List, Optional


def year_from_text(text: Optional[str]) -> Optional[int]:
    """First 4-digit year found in a date-ish string, else None."""
    if not text:
        return None
    m = re.search(r"[12][0-9]{3}", text)
    return int(m.group(0)) if m else None


def strip_jats(text: Optional[str]) -> Optional[str]:
    """Crossref abstracts are JATS XML fragments (`<jats:p>...</jats:p>`).
    Strips tags, collapses whitespace. `None` in, `None` out."""
    if not text:
        return None
    stripped = re.sub(r"<[^>]+>", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or None


def text_of(node: Any) -> str:
    """`xmltodict` returns a bare string for a text-only XML element, but a
    `{"#text": ..., "@attr": ...}` dict once the element carries any
    attribute. Handles both uniformly."""
    if isinstance(node, dict):
        return str(node.get("#text", ""))
    return str(node or "")


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def int_or_none(value: Any) -> Optional[int]:
    """`int(value)` if `value` is a non-`None`, digit-only string/int, else
    `None` -- never raises on a missing or non-numeric API field."""
    if value is None:
        return None
    return int(value) if str(value).isdigit() else None


_SECRET_LOOKING_KV_RE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|insttoken|inst[_-]?token|token|authorization|secret)\b\s*[:=]\s*\S+"
)
_MAX_REMOTE_ERROR_TEXT_LEN = 300


def sanitize_remote_error_text(text: Any, max_len: int = _MAX_REMOTE_ERROR_TEXT_LEN) -> str:
    """Bound and redact arbitrary remote-supplied error text before it is
    ever embedded in a `SearchBackendResult.failure_detail` -- `base.py`'s
    own contract requires `failure_detail` be "sanitised: never a URL
    query string, never a token", but an adapter that copies a remote
    error message verbatim (Scopus's `error` field, PubMed's
    `errorlist`/`warninglist` payloads) has no control over what the
    remote service chose to echo back (a key/token accidentally echoed in
    a diagnostic message, or an unbounded-length string). Any
    `key=value`/`key: value` pair whose key looks like a credential is
    redacted; the result is then truncated to `max_len` characters."""
    s = str(text)
    s = _SECRET_LOOKING_KV_RE.sub(lambda m: f"{m.group(1)}=<redacted>", s)
    if len(s) > max_len:
        s = s[:max_len] + "...(truncated)"
    return s
