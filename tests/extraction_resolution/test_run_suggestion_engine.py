"""Offline tests for `run_suggestion_engine()` -- the crash-safe
three-step state machine and startup reconciliation of orphaned
`.reserved.json` markers (amber-quilting-osprey.md sec. 3.3/3.4/3.8).

No network call anywhere in this file: every "LLM response" comes from a
fake, in-memory client object shaped like the real OpenAI SDK's
`client.chat.completions.create(...)` call site, per the standing
offline-tests-only authorization.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from slr_pipeline.ai_traceability import AIEventLockError, AIUseEventStore, attempt_lock_path, compute_event_id, kernel_lock
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.resolution_ids import safe_key
from slr_pipeline.extraction.resolution_schemas import (
    AIUseEventDraft,
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    FailedCallRecord,
    RejectedSuggestion,
    ReservedAttemptMarker,
    ValidatedSuggestion,
    compute_expected_work_item_id,
)
from slr_pipeline.extraction.resolution_suggestions import (
    ReviewPacket,
    ReviewPacketEntry,
    _AlreadyHandled,
    _atomic_write_model,
    _fsync_dir,
    _fsync_dir_windows,
    _win32_flush_directory,
    _candidates_for_field,
    _completed_path,
    _content_path,
    _dispatch_one,
    _evidence_index_for_field,
    _interrupted_path,
    _load_attempt_record,
    _publish_event_for_record,
    _reserved_path,
    build_suggestion_prompt,
    parse_suggestion_response,
    reconcile_suggestion_attempts,
    run_suggestion_engine,
    validate_suggestion,
)
from slr_pipeline.extraction.results import EvidenceSpanModel, FieldResult, FieldStatus
from slr_pipeline.extraction.schema import ExtractionFieldSpec, FieldOrigin, FieldProvenance, ValueKind

EXTRACTION_RUN_ID = "extrun-engine-0001"
REVIEW_RUN_ID = "review-run-engine-0001"
_FIXED_NOW = "2026-09-18T00:00:00Z"


def _provenance() -> FieldProvenance:
    return FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")


def _field_spec(field_id: str) -> ExtractionFieldSpec:
    return ExtractionFieldSpec(
        field_id=field_id, question="What is the study's sample size?",
        description="Total number of participants enrolled.", value_kind=ValueKind.INTEGER,
        cardinality="single", provenance=_provenance(),
    )


def _field_result(field_id: str) -> FieldResult:
    return FieldResult(
        field_id=field_id, status=FieldStatus.UNCLEAR,
        review_evidence=(
            EvidenceSpanModel(chunk_id="c1", excerpt="N=120 participants were enrolled.", page="3", section="Methods"),
        ),
        rationale="Ambiguous phrasing around enrollment vs. completion counts.", requires_human_review=True,
    )


def _entry(record_id: str, field_id: str) -> ReviewPacketEntry:
    field_result = _field_result(field_id)
    evidence_index = _evidence_index_for_field(record_id=record_id, field_id=field_id, field_result=field_result)
    candidates = _candidates_for_field(record_id=record_id, field_id=field_id, field_result=field_result)
    return ReviewPacketEntry(
        record_id=record_id, field_id=field_id, field_spec=_field_spec(field_id),
        field_result=field_result, evidence_index=evidence_index, candidates=candidates,
    )


def _packet(*entries: ReviewPacketEntry) -> ReviewPacket:
    return ReviewPacket(
        run_id=EXTRACTION_RUN_ID, manifest_hash="a" * 64, extraction_schema_hash="b" * 64,
        entries=tuple(entries), resolution_history=(),
    )


def _valid_response_json(evidence_id: str) -> str:
    return json.dumps({
        "outcome": "extracted", "value": "120", "evidence_ids": [evidence_id],
        "candidate_ids_merged": [], "candidate_ids_retained": [], "candidate_ids_rejected": [],
        "rationale": "clear", "confidence": "high", "uncertainty": None, "requires_human_review": False,
    })


def _rejected_response_json() -> str:
    return json.dumps({
        "outcome": "extracted", "value": "120", "evidence_ids": ["nonexistent-evidence-id"],
        "candidate_ids_merged": [], "candidate_ids_retained": [], "candidate_ids_rejected": [],
        "rationale": "clear", "confidence": "high", "uncertainty": None, "requires_human_review": False,
    })


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeClient:
    """Shaped like the real `OpenAI` client at exactly the one call site
    `run_suggestion_engine()` uses (`client.chat.completions.create(...)`)
    -- `actions` is consumed one per call, in order; an entry that is a
    `BaseException` instance is raised instead of returning a response."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, messages):
        action = self._actions[self.calls]
        self.calls += 1
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, str):
            return _FakeResponse(action)
        return action  # already a response-shaped (possibly malformed) object


class _CapExceededError(Exception):
    def __init__(self) -> None:
        super().__init__("cost cap exceeded")
        self.status_code = 402


def _build_validated_record(entry, evidence_id, *, work_item_id, event_id, attempt=0):
    prompt = build_suggestion_prompt(entry)
    draft = AIUseEventDraft(
        review_run_id=REVIEW_RUN_ID, work_item_id=work_item_id, attempt=attempt, model="gpt-test",
        provider="openai", component_versions={"resolution_suggestions": "1"}, access_info=None,
        task_purpose="extraction_resolution_suggestion", model_config_id="test-config-1",
        prompt_template_version="1", prompt_template_text=prompt, prompt_parameters={},
        rendered_prompt_hash=sha256_hex(prompt),
        input_artifact_refs=[f"extraction_review_packet_entry:{entry.record_id}:{entry.field_id}"],
        event_timestamp=_FIXED_NOW,
    )
    parsed = parse_suggestion_response(_valid_response_json(evidence_id))
    return validate_suggestion(
        parsed, field_spec=entry.field_spec, field_result=entry.field_result, evidence_index=entry.evidence_index,
        extraction_run_id=EXTRACTION_RUN_ID, record_id=entry.record_id, field_id=entry.field_id, attempt=attempt,
        ai_use_event_id=event_id, ai_use_event_draft=draft, created_at=_FIXED_NOW,
        prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost_usd=None,
    )


def _make_inputs(tmp_path, entries):
    packet = _packet(*entries)
    suggestions_dir = tmp_path / "suggestions"
    event_store = AIUseEventStore(output_dir=tmp_path, review_run_id=REVIEW_RUN_ID)
    return packet, suggestions_dir, event_store


def _run(packet, suggestions_dir, event_store, client, **kwargs):
    return run_suggestion_engine(
        packet, extraction_run_id=EXTRACTION_RUN_ID, review_run_id=REVIEW_RUN_ID,
        suggestions_dir=suggestions_dir, event_store=event_store, client=client, model="gpt-test",
        provider="openai", component_versions={"resolution_suggestions": "1"}, model_config_id="test-config-1",
        now=lambda: _FIXED_NOW, **kwargs,
    )


def test_successful_dispatch_produces_validated_suggestion(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    [evidence_id] = list(entry.evidence_index.keys())
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([_valid_response_json(evidence_id)])

    result = _run(packet, suggestions_dir, event_store, client)

    assert len(result.items) == 1
    item = result.items[0]
    assert item.status == "dispatched"
    assert item.attempt == 0
    assert isinstance(item.attempt_record, ValidatedSuggestion)

    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    assert _completed_path(suggestions_dir, key, 0).is_file()
    assert not _reserved_path(suggestions_dir, key, 0).exists()
    assert _content_path(suggestions_dir, key, 0).is_file()

    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].call_status == "success"
    assert result.interrupted_total == 0
    assert result.cap_exceeded is False


def test_content_rejection_still_publishes_success_call_status(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([_rejected_response_json()])

    result = _run(packet, suggestions_dir, event_store, client)

    item = result.items[0]
    assert isinstance(item.attempt_record, RejectedSuggestion)
    events = event_store.load_all()
    assert events[0].call_status == "success"


def test_caught_call_exception_produces_failed_call_record(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([RuntimeError("boom: Authorization: Bearer sk-should-be-redacted")])

    result = _run(packet, suggestions_dir, event_store, client)

    item = result.items[0]
    assert isinstance(item.attempt_record, FailedCallRecord)
    assert item.attempt_record.error_type == "RuntimeError"
    assert "REDACTED" in item.attempt_record.error_message
    assert "sk-should-be-redacted" not in item.attempt_record.error_message

    events = event_store.load_all()
    assert events[0].call_status == "failure"


def test_validated_suggestion_never_retried(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    [evidence_id] = list(entry.evidence_index.keys())
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([_valid_response_json(evidence_id)])
    _run(packet, suggestions_dir, event_store, client)
    assert client.calls == 1

    client2 = _FakeClient([])
    result2 = _run(packet, suggestions_dir, event_store, client2)

    assert result2.items[0].status == "skipped_already_validated"
    assert client2.calls == 0


def test_rejected_suggestion_retried_only_when_explicitly_named(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([_rejected_response_json()])
    _run(packet, suggestions_dir, event_store, client)
    assert client.calls == 1

    client2 = _FakeClient([])
    result2 = _run(packet, suggestions_dir, event_store, client2)
    assert result2.items[0].status == "skipped_not_retried"
    assert client2.calls == 0

    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    [evidence_id] = list(entry.evidence_index.keys())
    client3 = _FakeClient([_valid_response_json(evidence_id)])
    result3 = _run(packet, suggestions_dir, event_store, client3, retry_work_item_ids=[work_item_id])

    assert result3.items[0].status == "dispatched"
    assert result3.items[0].attempt == 1
    assert isinstance(result3.items[0].attempt_record, ValidatedSuggestion)


def test_true_interruption_archived_and_retry_gets_fresh_attempt(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
        work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
        reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    _reserved_path(suggestions_dir, key, 0).write_text(marker.model_dump_json(indent=2), encoding="utf-8")

    client = _FakeClient([])  # a bare reconciliation pass must never call the model
    result = _run(packet, suggestions_dir, event_store, client)

    assert result.reconciliation.archived_interrupted == 1
    assert result.interrupted_total == 1
    assert _interrupted_path(suggestions_dir, key, 0).is_file()
    assert not _reserved_path(suggestions_dir, key, 0).exists()
    assert result.items[0].status == "skipped_not_retried"
    assert client.calls == 0

    [evidence_id] = list(entry.evidence_index.keys())
    client2 = _FakeClient([_valid_response_json(evidence_id)])
    result2 = _run(packet, suggestions_dir, event_store, client2, retry_work_item_ids=[work_item_id])

    assert result2.items[0].status == "dispatched"
    assert result2.items[0].attempt == 1
    assert isinstance(result2.items[0].attempt_record, ValidatedSuggestion)


def test_cap_exceeded_stops_further_dispatch_but_records_triggering_attempt(tmp_path) -> None:
    entry_a = _entry("doc-1", "sample_size")
    entry_b = _entry("doc-1", "study_design")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry_a, entry_b])
    client = _FakeClient([_CapExceededError()])

    result = _run(packet, suggestions_dir, event_store, client)

    assert client.calls == 1
    assert result.cap_exceeded is True
    first, second = result.items
    assert first.status == "dispatched"
    assert isinstance(first.attempt_record, FailedCallRecord)
    assert second.status == "skipped_cap_exceeded"

    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].call_status == "failure"


def test_reconciliation_completes_orphaned_content_without_event(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    [evidence_id] = list(entry.evidence_index.keys())
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size", reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), marker)
    record = _build_validated_record(entry, evidence_id, work_item_id=work_item_id, event_id=event_id)
    _atomic_write_model(_content_path(suggestions_dir, key, 0), record)

    assert event_store.load_all() == []
    result = reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW)

    assert result.completed == 1
    assert result.archived_interrupted == 0
    assert not _reserved_path(suggestions_dir, key, 0).exists()
    assert _completed_path(suggestions_dir, key, 0).is_file()
    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].event_id == event_id
    assert events[0].call_status == "success"


def test_reconciliation_content_with_already_published_event_is_idempotent(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    [evidence_id] = list(entry.evidence_index.keys())
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size", reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), marker)
    record = _build_validated_record(entry, evidence_id, work_item_id=work_item_id, event_id=event_id)
    _atomic_write_model(_content_path(suggestions_dir, key, 0), record)
    _publish_event_for_record(event_store, record, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE)
    assert len(event_store.load_all()) == 1

    result = reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW)

    assert result.completed == 1
    assert not _reserved_path(suggestions_dir, key, 0).exists()
    assert _completed_path(suggestions_dir, key, 0).is_file()
    events = event_store.load_all()
    assert len(events) == 1  # republish is a no-op, never a duplicate


def test_reconciliation_skips_reservation_whose_lock_is_still_live(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size", reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), marker)

    lock_path = attempt_lock_path(event_store, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id)
    with kernel_lock(lock_path, error_cls=AIEventLockError, description="simulated-live-call", timeout=1.0):
        result = reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW, lock_timeout=0.2)

    assert result.archived_interrupted == 0
    assert result.completed == 0
    assert _reserved_path(suggestions_dir, key, 0).is_file()
    assert event_store.load_all() == []


def test_reconciliation_archival_is_idempotent_across_crash_before_marker_unlink(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size", reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), marker)

    result = reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW)
    assert result.archived_interrupted == 1
    original = _interrupted_path(suggestions_dir, key, 0).read_text(encoding="utf-8")

    # Simulate a crash that happened after the archive was written but before
    # the `.reserved.json` marker was unlinked -- on restart the marker is
    # still there, so reconciliation runs again over the same attempt.
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), marker)

    result2 = reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: "2026-09-19T00:00:00Z")
    assert result2.archived_interrupted == 0  # must not overwrite the existing archive
    assert not _reserved_path(suggestions_dir, key, 0).exists()
    assert _interrupted_path(suggestions_dir, key, 0).read_text(encoding="utf-8") == original


def test_malformed_response_produces_rejected_suggestion_not_failed_call(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient(["not valid json at all"])

    result = _run(packet, suggestions_dir, event_store, client)

    item = result.items[0]
    assert isinstance(item.attempt_record, RejectedSuggestion)
    assert item.attempt_record.raw_response == "not valid json at all"
    events = event_store.load_all()
    assert events[0].call_status == "success"


def test_load_attempt_record_rejects_content_under_mismatched_filename(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    [evidence_id] = list(entry.evidence_index.keys())
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    suggestions_dir.mkdir(parents=True)
    record = _build_validated_record(entry, evidence_id, work_item_id=work_item_id, event_id=event_id, attempt=0)
    _atomic_write_model(_content_path(suggestions_dir, key, 0), record)

    other_entry = _entry("doc-2", "sample_size")
    other_key = safe_key(EXTRACTION_RUN_ID, "doc-2", "sample_size")
    renamed_path = _content_path(suggestions_dir, other_key, 0)
    _content_path(suggestions_dir, key, 0).rename(renamed_path)

    with pytest.raises(ValueError, match="does not match this file's own filename key"):
        _load_attempt_record(renamed_path, key=other_key, attempt=0)


def test_dispatch_one_raises_already_handled_for_concurrently_validated_work_item(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    [evidence_id] = list(entry.evidence_index.keys())
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    suggestions_dir.mkdir(parents=True)
    record = _build_validated_record(entry, evidence_id, work_item_id=work_item_id, event_id=event_id, attempt=0)
    _atomic_write_model(_content_path(suggestions_dir, key, 0), record)
    _publish_event_for_record(event_store, record, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE)

    client = _FakeClient([])  # must never be called
    with pytest.raises(_AlreadyHandled) as excinfo:
        _dispatch_one(
            entry, extraction_run_id=EXTRACTION_RUN_ID, review_run_id=REVIEW_RUN_ID, suggestions_dir=suggestions_dir,
            event_store=event_store, client=client, model="gpt-test", provider="openai",
            component_versions={"resolution_suggestions": "1"}, model_config_id="test-config-1",
            task_purpose="extraction_resolution_suggestion", prompt_template_version="1", access_info=None,
            retry_work_item_ids=(), now=lambda: _FIXED_NOW, is_cap_exceeded=lambda exc: False,
        )

    assert excinfo.value.status == "skipped_already_validated"
    assert client.calls == 0


def test_reconciliation_rejects_marker_content_mismatched_with_filename(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size", reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    # A marker describing doc-1 written under a filename claiming a different
    # key entirely -- e.g. the result of a rename -- must never be trusted.
    wrong_key = safe_key(EXTRACTION_RUN_ID, "doc-2", "sample_size")
    _atomic_write_model(_reserved_path(suggestions_dir, wrong_key, 0), marker)

    with pytest.raises(ValueError, match="does not match this file's own filename key"):
        reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW)


def test_reconciliation_rejects_existing_archive_mismatched_with_marker(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
    )
    event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=0,
    )
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size", reserved_at=_FIXED_NOW,
    )
    suggestions_dir.mkdir(parents=True)
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), marker)
    result = reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW)
    assert result.archived_interrupted == 1

    # A second, distinct reservation for this same attempt slot must never
    # arise in practice (extra_occupied prevents slot reuse) -- but if a
    # `.reserved.json` with a genuinely different identity is ever found
    # sitting next to an existing archive, that must be treated as
    # corruption, never silently reconciled away.
    other_marker = ReservedAttemptMarker(
        work_item_id=work_item_id, attempt=0, event_id=event_id, review_run_id=REVIEW_RUN_ID,
        extraction_run_id=EXTRACTION_RUN_ID, record_id="doc-1", field_id="sample_size",
        reserved_at="2026-09-19T00:00:00Z",
    )
    _atomic_write_model(_reserved_path(suggestions_dir, key, 0), other_marker)

    with pytest.raises(ValueError, match="does not match this reservation's own"):
        reconcile_suggestion_attempts(suggestions_dir, event_store, now=lambda: _FIXED_NOW)


def test_malformed_response_envelope_produces_rejected_suggestion(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([SimpleNamespace()])  # no .choices at all

    result = _run(packet, suggestions_dir, event_store, client)

    item = result.items[0]
    assert isinstance(item.attempt_record, RejectedSuggestion)
    assert "malformed response envelope" in item.attempt_record.rejection_reasons[0]
    events = event_store.load_all()
    assert events[0].call_status == "success"

    # Never leaves a dangling .reserved.json for reconciliation to later
    # misclassify as an unknown interruption.
    key = safe_key(EXTRACTION_RUN_ID, "doc-1", "sample_size")
    assert not _reserved_path(suggestions_dir, key, 0).exists()
    assert _completed_path(suggestions_dir, key, 0).is_file()


def test_malformed_response_envelope_empty_choices_produces_rejected_suggestion(tmp_path) -> None:
    entry = _entry("doc-1", "sample_size")
    packet, suggestions_dir, event_store = _make_inputs(tmp_path, [entry])
    client = _FakeClient([SimpleNamespace(choices=[])])  # IndexError path

    result = _run(packet, suggestions_dir, event_store, client)

    assert isinstance(result.items[0].attempt_record, RejectedSuggestion)
    assert event_store.load_all()[0].call_status == "success"


def test_fsync_dir_does_not_raise_on_a_real_directory(tmp_path) -> None:
    _fsync_dir(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 FlushFileBuffers API is Windows-only")
def test_win32_flush_directory_returns_true_on_success(tmp_path) -> None:
    # Asserts the actual Win32 API result, not merely the absence of a
    # raised exception -- the GENERIC_READ-instead-of-GENERIC_WRITE bug
    # this guards against made FlushFileBuffers() return False on every
    # real call while raising nothing, so a does-not-raise-only test
    # would never have caught it.
    assert _win32_flush_directory(tmp_path) is True


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 FlushFileBuffers API is Windows-only")
def test_fsync_dir_windows_raises_on_directory_that_does_not_exist(tmp_path) -> None:
    with pytest.raises(OSError):
        _fsync_dir_windows(tmp_path / "does-not-exist")
