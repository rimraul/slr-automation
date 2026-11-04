from __future__ import annotations

import pytest
from pydantic import ValidationError

from slr_pipeline.ai_traceability import compute_event_id
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.resolution_schemas import (
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    AIUseEventDraft,
    FailedCallRecord,
    InterruptedCallRecord,
    RejectedSuggestion,
    ValidatedSuggestion,
    build_failed_call_record,
    build_interrupted_call_record,
    build_rejected_suggestion,
    build_validated_suggestion,
    compute_attempt_record_id,
    compute_expected_work_item_id,
)

_FABRICATED_ATTEMPT_RECORD_ID = "a" * 24


def _recompute_self_hash_for_tampered(model_cls: type, payload: dict) -> str:
    """Mirrors resolution_schemas._compute_self_hash exactly, without
    depending on the private helper, so a test can build a record whose
    self_hash is genuinely, correctly derived from tampered content."""
    payload_no_hash = {k: v for k, v in payload.items() if k != "self_hash"}
    placeholder = model_cls.model_construct(**payload_no_hash, self_hash="")
    dumped = {k: v for k, v in placeholder.model_dump(mode="json").items() if k != "self_hash"}
    return sha256_hex(dumped)


def _fabricate_attempt_record_id(model_cls: type, valid_record) -> dict:
    """Takes a validly-built record, swaps in a fabricated (but still
    24-hex, still self-hash-consistent) attempt_record_id/suggestion_id,
    and returns a dict ready for model_validate() -- the exact attack
    _check_identity_consistency's attempt_record_id check must catch."""
    payload = valid_record.model_dump(mode="json")
    payload["attempt_record_id"] = _FABRICATED_ATTEMPT_RECORD_ID
    if "suggestion_id" in payload:
        payload["suggestion_id"] = _FABRICATED_ATTEMPT_RECORD_ID
    payload["self_hash"] = _recompute_self_hash_for_tampered(model_cls, payload)
    return payload

REVIEW_RUN_ID = "review-1"
EXTRACTION_RUN_ID = "run-1"
RECORD_ID = "record-1"
FIELD_ID = "field-1"


def _work_item_id(*, extraction_run_id=EXTRACTION_RUN_ID, record_id=RECORD_ID, field_id=FIELD_ID) -> str:
    return compute_expected_work_item_id(extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id)


def _event_id(*, review_run_id=REVIEW_RUN_ID, work_item_id=None, attempt=0) -> str:
    return compute_event_id(
        review_run_id=review_run_id, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
        work_item_id=work_item_id or _work_item_id(), attempt=attempt,
    )


def _draft(**overrides) -> AIUseEventDraft:
    fields = dict(
        review_run_id=REVIEW_RUN_ID,
        work_item_id=_work_item_id(),
        attempt=0,
        model="gpt-test",
        provider="openai",
        component_versions={"slr_pipeline": "0.0.0"},
        task_purpose="extraction_resolution_suggestion",
        model_config_id="cfg-1",
        input_artifact_refs=["ref-1"],
        event_timestamp="2026-09-17T00:00:00Z",
    )
    fields.update(overrides)
    return AIUseEventDraft(**fields)


def _identity(**overrides):
    fields = dict(
        extraction_run_id=EXTRACTION_RUN_ID, record_id=RECORD_ID,
        field_id=FIELD_ID, attempt=0, original_field_result_hash="f" * 64,
    )
    fields.update(overrides)
    return fields


def _interrupted_identity(**overrides):
    fields = _identity(**overrides)
    fields.pop("original_field_result_hash")
    return fields


def test_compute_attempt_record_id_deterministic_and_24_hex():
    a = compute_attempt_record_id(extraction_run_id="run-1", record_id="record-1", field_id="field-1", attempt=0)
    b = compute_attempt_record_id(extraction_run_id="run-1", record_id="record-1", field_id="field-1", attempt=0)
    assert a == b
    assert len(a) == 24
    int(a, 16)  # valid hex


def test_build_validated_suggestion_extracted_requires_value_and_evidence():
    vs = build_validated_suggestion(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        outcome="extracted",
        value="42",
        typed_value=42,
        evidence_ids=("ev1",),
        created_at="2026-09-17T00:00:00Z",
    )
    assert vs.suggestion_id == vs.attempt_record_id
    assert isinstance(vs, ValidatedSuggestion)


def test_validated_suggestion_extracted_without_evidence_rejected():
    with pytest.raises(ValidationError):
        build_validated_suggestion(
            **_identity(),
            ai_use_event_id=_event_id(),
            ai_use_event_draft=_draft(),
            outcome="extracted",
            value="42",
            evidence_ids=(),
            created_at="2026-09-17T00:00:00Z",
        )


def test_validated_suggestion_non_extracted_must_not_carry_value():
    with pytest.raises(ValidationError):
        build_validated_suggestion(
            **_identity(),
            ai_use_event_id=_event_id(),
            ai_use_event_draft=_draft(),
            outcome="not_reported",
            value="42",
            created_at="2026-09-17T00:00:00Z",
        )


def test_validated_suggestion_non_extracted_must_not_carry_typed_value():
    with pytest.raises(ValidationError):
        build_validated_suggestion(
            **_identity(),
            ai_use_event_id=_event_id(),
            ai_use_event_draft=_draft(),
            outcome="not_reported",
            typed_value=True,
            created_at="2026-09-17T00:00:00Z",
        )


def test_validated_suggestion_self_hash_tamper_detected():
    vs = build_validated_suggestion(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        outcome="extracted",
        value="42",
        evidence_ids=("ev1",),
        created_at="2026-09-17T00:00:00Z",
    )
    tampered = vs.model_dump(mode="json")
    tampered["value"] = "99"
    with pytest.raises(ValidationError):
        ValidatedSuggestion.model_validate(tampered)


def test_validated_suggestion_rejects_false_ai_use_event_id():
    with pytest.raises(ValidationError):
        build_validated_suggestion(
            **_identity(),
            ai_use_event_id="aiuse-" + "0" * 48,
            ai_use_event_draft=_draft(),
            outcome="extracted",
            value="42",
            evidence_ids=("ev1",),
            created_at="2026-09-17T00:00:00Z",
        )


def test_validated_suggestion_rejects_work_item_id_mismatch():
    with pytest.raises(ValidationError):
        build_validated_suggestion(
            **_identity(),
            ai_use_event_id=_event_id(),
            ai_use_event_draft=_draft(work_item_id="run-1:record-1:field-WRONG"),
            outcome="extracted",
            value="42",
            evidence_ids=("ev1",),
            created_at="2026-09-17T00:00:00Z",
        )


def test_validated_suggestion_rejects_draft_attempt_mismatch():
    with pytest.raises(ValidationError):
        build_validated_suggestion(
            **_identity(attempt=0),
            ai_use_event_id=_event_id(attempt=0),
            ai_use_event_draft=_draft(attempt=1),
            outcome="not_reported",
            created_at="2026-09-17T00:00:00Z",
        )


def test_build_rejected_suggestion_requires_reasons_and_carries_usage_fields():
    rs = build_rejected_suggestion(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        raw_response={"outcome": "bogus"},
        rejection_reasons=("unknown outcome value",),
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.001,
        created_at="2026-09-17T00:00:00Z",
    )
    assert rs.suggestion_id == rs.attempt_record_id
    assert rs.total_tokens == 15
    with pytest.raises(ValidationError):
        build_rejected_suggestion(
            **_identity(),
            ai_use_event_id=_event_id(),
            ai_use_event_draft=_draft(),
            raw_response={},
            rejection_reasons=(),
            created_at="2026-09-17T00:00:00Z",
        )


def test_build_failed_call_record_field_names():
    fc = build_failed_call_record(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        error_type="TimeoutError",
        error_message="call timed out after 30s",
        failed_at="2026-09-17T00:00:00Z",
    )
    assert isinstance(fc, FailedCallRecord)
    assert fc.error_message == "call timed out after 30s"
    assert fc.failed_at == "2026-09-17T00:00:00Z"


def test_build_interrupted_call_record_has_no_draft_field_but_has_identity():
    ic = build_interrupted_call_record(
        **_interrupted_identity(),
        ai_use_event_id=_event_id(),
        review_run_id=REVIEW_RUN_ID,
        work_item_id=_work_item_id(),
        reserved_at="2026-09-17T00:00:00Z",
        archived_at="2026-09-17T00:05:00Z",
    )
    assert isinstance(ic, InterruptedCallRecord)
    assert not hasattr(ic, "ai_use_event_draft")
    assert ic.review_run_id == REVIEW_RUN_ID
    assert ic.work_item_id == _work_item_id()


def test_interrupted_call_record_rejects_false_ai_use_event_id():
    with pytest.raises(ValidationError):
        build_interrupted_call_record(
            **_identity(),
            ai_use_event_id="aiuse-" + "0" * 48,
            review_run_id=REVIEW_RUN_ID,
            work_item_id=_work_item_id(),
            reserved_at="2026-09-17T00:00:00Z",
            archived_at="2026-09-17T00:05:00Z",
        )


def test_different_attempt_numbers_get_different_attempt_record_ids():
    vs0 = build_validated_suggestion(
        **_identity(attempt=0),
        ai_use_event_id=_event_id(attempt=0),
        ai_use_event_draft=_draft(attempt=0),
        outcome="not_reported",
        created_at="2026-09-17T00:00:00Z",
    )
    vs1 = build_validated_suggestion(
        **_identity(attempt=1),
        ai_use_event_id=_event_id(attempt=1),
        ai_use_event_draft=_draft(attempt=1),
        outcome="not_reported",
        created_at="2026-09-17T00:00:00Z",
    )
    assert vs0.attempt_record_id != vs1.attempt_record_id


def test_validated_suggestion_rejects_fabricated_attempt_record_id_despite_consistent_self_hash():
    vs = build_validated_suggestion(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        outcome="not_reported",
        created_at="2026-09-17T00:00:00Z",
    )
    tampered = _fabricate_attempt_record_id(ValidatedSuggestion, vs)
    with pytest.raises(ValidationError):
        ValidatedSuggestion.model_validate(tampered)


def test_rejected_suggestion_rejects_fabricated_attempt_record_id_despite_consistent_self_hash():
    rs = build_rejected_suggestion(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        raw_response={"outcome": "bogus"},
        rejection_reasons=("unknown outcome value",),
        created_at="2026-09-17T00:00:00Z",
    )
    tampered = _fabricate_attempt_record_id(RejectedSuggestion, rs)
    with pytest.raises(ValidationError):
        RejectedSuggestion.model_validate(tampered)


def test_failed_call_record_rejects_fabricated_attempt_record_id_despite_consistent_self_hash():
    fc = build_failed_call_record(
        **_identity(),
        ai_use_event_id=_event_id(),
        ai_use_event_draft=_draft(),
        error_type="TimeoutError",
        error_message="call timed out after 30s",
        failed_at="2026-09-17T00:00:00Z",
    )
    tampered = _fabricate_attempt_record_id(FailedCallRecord, fc)
    with pytest.raises(ValidationError):
        FailedCallRecord.model_validate(tampered)


def test_interrupted_call_record_rejects_fabricated_attempt_record_id_despite_consistent_self_hash():
    ic = build_interrupted_call_record(
        **_interrupted_identity(),
        ai_use_event_id=_event_id(),
        review_run_id=REVIEW_RUN_ID,
        work_item_id=_work_item_id(),
        reserved_at="2026-09-17T00:00:00Z",
        archived_at="2026-09-17T00:05:00Z",
    )
    tampered = _fabricate_attempt_record_id(InterruptedCallRecord, ic)
    with pytest.raises(ValidationError):
        InterruptedCallRecord.model_validate(tampered)
