"""Stable, versioned content IDs for evidence and candidate values used by
the extraction resolution-suggestion workflow (amber-quilting-osprey.md
sec. 3.2).

Both ID functions hash a structured value through the existing
`sha256_hex()` canonical-JSON utility (never an ad-hoc `\\x1f`-joined
f-string) so `page=None` becomes JSON `null` -- structurally distinct
from the string `"None"` -- and so a future change to what's hashed can
be made explicit via a schema-version bump rather than silently
reinterpreting old IDs as if they were computed the same way.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.results import TypedValue

_EVIDENCE_ID_SCHEMA_VERSION = 1
_CANDIDATE_ID_SCHEMA_VERSION = 1
_SAFE_KEY_SCHEMA_VERSION = 1

SourceCategory = Literal["value", "provisional_value", "invalid_value"]


def safe_key(extraction_run_id: str, record_id: str, field_id: str) -> str:
    """Content-derived, filesystem/locator-safe stand-in for the raw
    `(record_id, field_id)` pair, which is not guaranteed to satisfy
    `locators.py`'s `_LOCATOR_RE` grammar (alphanumeric plus `.`/`_`/`-`
    only, <=128 chars) or to be safe as a filesystem path component
    (`record_id` is an arbitrary source-assigned string). Always exactly
    24 lowercase hex characters. The real `record_id`/`field_id` remain
    fully present, unmangled, inside the suggestion file's own content
    and the generated index -- only the filename/locator subid is
    content-derived."""
    return sha256_hex(
        {
            "id_schema_version": _SAFE_KEY_SCHEMA_VERSION,
            "extraction_run_id": extraction_run_id,
            "record_id": record_id,
            "field_id": field_id,
        }
    )[:24]


def evidence_id(
    document_id: str,
    field_id: str,
    chunk_id: str,
    excerpt: str,
    page: Optional[str],
    section: str,
) -> str:
    return sha256_hex(
        {
            "id_schema_version": _EVIDENCE_ID_SCHEMA_VERSION,
            "document_id": document_id,
            "field_id": field_id,
            "chunk_id": chunk_id,
            "excerpt": excerpt,
            "page": page,
            "section": section,
        }
    )[:24]


def candidate_id(
    value: str,
    typed_value: Optional[TypedValue],
    source_category: SourceCategory,
    evidence_ids: Sequence[str],
) -> str:
    return sha256_hex(
        {
            "id_schema_version": _CANDIDATE_ID_SCHEMA_VERSION,
            "value": value,
            "typed_value": typed_value,
            "source_category": source_category,
            "evidence_ids": sorted(evidence_ids),
        }
    )[:24]
