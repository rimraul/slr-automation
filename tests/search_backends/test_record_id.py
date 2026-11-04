"""`test_real_adapter_record_id_is_safe_token` (plan §2.6): the deterministic
`record_id()` formula always produces a `ft_screening_adapter.py`
`_SAFE_ID_RE = ^[A-Za-z0-9_-]{1,128}$`-safe token, even for native_keys that
themselves contain characters the regex forbids (a DOI with `/`, a colon,
unicode)."""
from __future__ import annotations

import re

import pytest

from slr_pipeline.search_backends.base import record_id

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_NATIVE_KEYS = [
    "W2755950973",
    "10.1000/xyz123",                 # DOI, contains '/'
    "10.1000/xyz:with:colons",
    "2401.01234v1",
    "article number with spaces",
    "unicode-éèê-key",
    "",                                # even an (already-filtered-upstream) empty string
]


@pytest.mark.parametrize("native_key", _NATIVE_KEYS)
@pytest.mark.parametrize("source", ["openalex", "pubmed", "semantic_scholar", "arxiv",
                                     "crossref", "springer", "ieee"])
def test_real_adapter_record_id_is_safe_token(source, native_key):
    rid = record_id(source, native_key)
    assert _SAFE_ID_RE.fullmatch(rid), rid
    assert rid.startswith(f"{source}-")


def test_record_id_is_deterministic():
    assert record_id("openalex", "W1") == record_id("openalex", "W1")


def test_record_id_collision_safe_across_distinct_native_keys():
    ids = {record_id("openalex", k) for k in _NATIVE_KEYS}
    assert len(ids) == len(_NATIVE_KEYS)


def test_record_id_digest_is_20_lowercase_hex_chars():
    rid = record_id("crossref", "10.1000/abc")
    digest = rid.split("-", 1)[1]
    assert len(digest) == 20
    assert re.fullmatch(r"[0-9a-f]{20}", digest)
