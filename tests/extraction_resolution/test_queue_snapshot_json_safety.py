"""JSON-safety and binding tests for QueueSnapshot/QueueSnapshotResolutionEntry
(amber-quilting-osprey.md sec. 3.7, round-6 fix)."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from slr_pipeline.extraction.resolution_schemas import QueueSnapshot, QueueSnapshotEntry, QueueSnapshotResolutionEntry


def _queue_entry(record_id="rec-1", field_id="population") -> dict:
    return {
        "record_id": record_id, "field_id": field_id, "status": "unclear",
        "field_result": {"field_id": field_id, "status": "unclear", "requires_human_review": True},
    }


def _resolution_entry(**overrides) -> dict:
    base = dict(
        record_id="rec-2", field_id="limitations", original_status="unclear",
        original_field_result_hash="a" * 24, outcome="extracted", reviewer="alice",
        note="clear from table 2", resolved_at="2026-09-01T00:00:00+00:00",
        run_id="extrun-1", manifest_hash="b" * 64, extraction_schema_hash="c" * 64,
        value="Small sample", evidence=[{"quotation_or_description": "n=12", "locator": "p.4"}], attested=True,
    )
    base.update(overrides)
    return base


def _snapshot(**overrides) -> QueueSnapshot:
    base = dict(
        run_id="extrun-1", manifest_hash="b" * 64, extraction_schema_hash="c" * 64,
        checkpoint_hashes={"rec-1": "d" * 64, "rec-2": "e" * 64},
        queue=[_queue_entry()], resolution_history_as_of_render=[_resolution_entry()],
    )
    base.update(overrides)
    return QueueSnapshot(**base)


def test_queue_snapshot_round_trips_through_real_json_dumps_loads():
    snapshot = _snapshot()
    dumped = json.dumps(snapshot.model_dump(mode="json"))
    reloaded = QueueSnapshot.model_validate(json.loads(dumped))
    assert reloaded == snapshot


def test_queue_snapshot_queue_is_a_flat_list_never_a_tuple_keyed_mapping():
    snapshot = _snapshot()
    assert isinstance(snapshot.queue, list)
    for entry in snapshot.queue:
        assert isinstance(entry, QueueSnapshotEntry)
        assert entry.record_id and entry.field_id


def test_queue_snapshot_checkpoint_hashes_is_a_string_keyed_dict():
    snapshot = _snapshot()
    assert isinstance(snapshot.checkpoint_hashes, dict)
    for key, value in snapshot.checkpoint_hashes.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_queue_snapshot_resolution_history_defaults_to_empty_list():
    snapshot = QueueSnapshot(
        run_id="extrun-1", manifest_hash="b" * 64, extraction_schema_hash="c" * 64,
        checkpoint_hashes={}, queue=[],
    )
    assert snapshot.resolution_history_as_of_render == []


def test_queue_snapshot_resolution_entry_extracted_outcome_carries_value_evidence_attested():
    entry = QueueSnapshotResolutionEntry(**_resolution_entry())
    assert entry.outcome == "extracted"
    assert entry.value == "Small sample"
    assert entry.evidence == [{"quotation_or_description": "n=12", "locator": "p.4"}]
    assert entry.attested is True


def test_queue_snapshot_resolution_entry_not_reported_outcome_may_omit_value_evidence_attested():
    entry = QueueSnapshotResolutionEntry(
        **_resolution_entry(outcome="not_reported", value=None, evidence=None, attested=None)
    )
    assert entry.value is None
    assert entry.evidence is None
    assert entry.attested is None


def test_queue_snapshot_resolution_entry_rejects_unknown_outcome():
    with pytest.raises(ValidationError):
        QueueSnapshotResolutionEntry(**_resolution_entry(outcome="unclear"))


def test_queue_snapshot_rejects_extra_fields():
    with pytest.raises(ValidationError):
        _snapshot(bogus_field="nope")


def test_queue_snapshot_entry_rejects_extra_fields():
    with pytest.raises(ValidationError):
        QueueSnapshotEntry(**_queue_entry(), bogus_field="nope")


def test_queue_snapshot_rejects_malformed_queue_entry_at_validation_not_later_as_a_keyerror():
    malformed = {"record_id": "rec-1", "field_id": "population"}  # missing status/field_result
    with pytest.raises(ValidationError):
        _snapshot(queue=[malformed])


def test_queue_snapshot_rejects_queue_entry_with_field_result_violating_its_own_invariants():
    invalid_field_result = {
        "record_id": "rec-1", "field_id": "population", "status": "unclear",
        "field_result": {"field_id": "population", "status": "unclear", "requires_human_review": False},
    }
    with pytest.raises(ValidationError):
        _snapshot(queue=[invalid_field_result])
