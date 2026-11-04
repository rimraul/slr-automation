"""Unit tests for `slr_pipeline.replay.cassette` -- the strict
`REPLAY_CASSETTE_VERSION` schema (`track2_pipeline_readiness_plan_2026-08-30.md`
Sec 6.1, step 10)."""
from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from slr_pipeline.replay.cassette import (
    REFERENCE_SURFACE_CALL_COUNTS,
    REPLAY_CASSETTE_VERSION,
    ReplayCassette,
    ReplayCassetteEntry,
)

_HEX = "a" * 64
_HEX2 = "b" * 64


def _entry(entry_id="e1", surface="query_expansion", signature=_HEX, expected_calls=1):
    return {
        "entry_id": entry_id,
        "surface": surface,
        "signature": signature,
        "endpoint": "/v1/chat/completions",
        "request_preview": "hello",
        "response_body": {"ok": True},
        "expected_calls": expected_calls,
    }


def test_cassette_version_pin():
    assert REPLAY_CASSETTE_VERSION == "1"


def test_entry_accepts_all_eight_surfaces():
    for surface in (
        "query_expansion", "db_translation", "title_abstract_screening",
        "qc_second_rater", "ft_adjudication", "extraction_batch",
        "synthesis_map", "synthesis_refine",
    ):
        entry = ReplayCassetteEntry.model_validate(_entry(surface=surface))
        assert entry.surface == surface


def test_entry_rejects_ft_retrieval_surface():
    with pytest.raises(ValidationError):
        ReplayCassetteEntry.model_validate(_entry(surface="ft_retrieval"))


def test_entry_rejects_non_hex64_signature():
    with pytest.raises(ValidationError):
        ReplayCassetteEntry.model_validate(_entry(signature="not-hex"))


def test_entry_rejects_wrong_endpoint():
    bad = _entry()
    bad["endpoint"] = "/v1/other"
    with pytest.raises(ValidationError):
        ReplayCassetteEntry.model_validate(bad)


def test_entry_forbids_extra_fields():
    bad = _entry()
    bad["extra_field"] = "nope"
    with pytest.raises(ValidationError):
        ReplayCassetteEntry.model_validate(bad)


def test_entry_rejects_non_positive_expected_calls():
    with pytest.raises(ValidationError):
        ReplayCassetteEntry.model_validate(_entry(expected_calls=0))


def test_cassette_from_list_roundtrip():
    raw = [_entry(entry_id="e1", signature=_HEX), _entry(entry_id="e2", signature=_HEX2)]
    cassette = ReplayCassette.from_list(raw)
    assert len(cassette.entries) == 2
    assert set(cassette.by_signature()) == {_HEX, _HEX2}


def test_cassette_rejects_duplicate_entry_id():
    raw = [_entry(entry_id="e1", signature=_HEX), _entry(entry_id="e1", signature=_HEX2)]
    with pytest.raises(ValidationError, match="entry_id"):
        ReplayCassette.from_list(raw)


def test_cassette_rejects_duplicate_signature():
    raw = [_entry(entry_id="e1", signature=_HEX), _entry(entry_id="e2", signature=_HEX)]
    with pytest.raises(ValidationError, match="signature"):
        ReplayCassette.from_list(raw)


def test_cassette_forbids_unknown_top_level_field():
    with pytest.raises(ValidationError):
        ReplayCassette(entries=[], extra_field="nope")


def test_cassette_surface_totals():
    """Sec 4.1's frozen per-surface call topology, verbatim: a cassette
    shaped like the real Track-2 reference run's must sum to exactly the
    same per-surface totals as `REFERENCE_SURFACE_CALL_COUNTS`, and to 50
    overall (== `CORPUS_ORACLE.total_replay_calls`, Sec 6.1/11.2)."""
    raw = []
    for surface, count in REFERENCE_SURFACE_CALL_COUNTS.items():
        raw.append(_entry(entry_id=f"{surface}-1", surface=surface,
                           signature=hashlib.sha256(surface.encode()).hexdigest(),
                           expected_calls=count))
    cassette = ReplayCassette.from_list(raw)
    totals = cassette.surface_totals()
    assert totals == REFERENCE_SURFACE_CALL_COUNTS
    assert sum(totals.values()) == 50
    assert REFERENCE_SURFACE_CALL_COUNTS["ft_adjudication"] == 18
    assert REFERENCE_SURFACE_CALL_COUNTS["extraction_batch"] == 5
