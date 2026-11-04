"""Offline tests for the LLM suggestion contract boundary --
`build_suggestion_prompt()`, `parse_suggestion_response()`, and
`validate_suggestion()` (amber-quilting-osprey.md sec. 3.3/4.4/4.5).

No network call anywhere in this file: every "LLM response" is a hand-built
JSON string or dict, per the standing offline-tests-only authorization.
"""
from __future__ import annotations

import pytest

from slr_pipeline.ai_traceability import compute_event_id
from slr_pipeline.extraction.resolution_schemas import (
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    AIUseEventDraft,
    RejectedSuggestion,
    ValidatedSuggestion,
    compute_expected_work_item_id,
)
from slr_pipeline.extraction.resolution_suggestions import (
    CandidateItem,
    RawSuggestion,
    ReviewPacketEntry,
    SuggestionParseError,
    _candidates_for_field,
    _evidence_index_for_field,
    build_suggestion_prompt,
    parse_suggestion_response,
    validate_suggestion,
)
from slr_pipeline.extraction.results import (
    EvidenceSpanModel,
    ExtractedValue,
    FieldResult,
    FieldStatus,
)
from slr_pipeline.extraction.schema import (
    ExtractionFieldSpec,
    FieldOrigin,
    FieldProvenance,
    ValueKind,
)

RECORD_ID = "doc-1"
FIELD_ID = "sample_size"
EXTRACTION_RUN_ID = "extrun-test-0001"
REVIEW_RUN_ID = "review-run-test-0001"


def _provenance() -> FieldProvenance:
    return FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")


def _integer_field_spec() -> ExtractionFieldSpec:
    return ExtractionFieldSpec(
        field_id=FIELD_ID,
        question="What is the study's sample size?",
        description="Total number of participants enrolled.",
        value_kind=ValueKind.INTEGER,
        cardinality="single",
        provenance=_provenance(),
    )


def _unclear_field_result_with_review_evidence() -> FieldResult:
    return FieldResult(
        field_id=FIELD_ID,
        status=FieldStatus.UNCLEAR,
        review_evidence=(
            EvidenceSpanModel(chunk_id="c1", excerpt="N=120 participants were enrolled.", page="3", section="Methods"),
        ),
        rationale="Ambiguous phrasing around enrollment vs. completion counts.",
        requires_human_review=True,
    )


def _field_result_with_value_candidate() -> FieldResult:
    return FieldResult(
        field_id=FIELD_ID,
        status=FieldStatus.EXTRACTED,
        values=(
            ExtractedValue(
                value="120",
                evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="N=120 participants were enrolled.", page="3", section="Methods"),),
            ),
        ),
        confidence="high",
        requires_human_review=True,
    )


def _packet_entry(field_spec: ExtractionFieldSpec, field_result: FieldResult) -> ReviewPacketEntry:
    evidence_index = _evidence_index_for_field(record_id=RECORD_ID, field_id=FIELD_ID, field_result=field_result)
    candidates = _candidates_for_field(record_id=RECORD_ID, field_id=FIELD_ID, field_result=field_result)
    return ReviewPacketEntry(
        record_id=RECORD_ID, field_id=FIELD_ID, field_spec=field_spec, field_result=field_result,
        evidence_index=evidence_index, candidates=candidates,
    )


def _identity_kwargs(*, attempt: int = 0) -> dict:
    work_item_id = compute_expected_work_item_id(extraction_run_id=EXTRACTION_RUN_ID, record_id=RECORD_ID, field_id=FIELD_ID)
    ai_use_event_id = compute_event_id(
        review_run_id=REVIEW_RUN_ID, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=attempt,
    )
    draft = AIUseEventDraft(
        review_run_id=REVIEW_RUN_ID, work_item_id=work_item_id, attempt=attempt,
        model="gpt-test", provider="openai", component_versions={"resolution_suggestions": "1"},
        access_info=None, task_purpose="extraction_resolution_suggestion", model_config_id="test-config-1",
        prompt_template_version="1", prompt_template_text="prompt text", prompt_parameters={},
        rendered_prompt_hash=None, input_artifact_refs=[], event_timestamp="2026-09-18T00:00:00+00:00",
    )
    return dict(
        extraction_run_id=EXTRACTION_RUN_ID, record_id=RECORD_ID, field_id=FIELD_ID, attempt=attempt,
        ai_use_event_id=ai_use_event_id, ai_use_event_draft=draft, created_at="2026-09-18T00:00:01+00:00",
    )


class TestBuildSuggestionPrompt:
    def test_includes_field_and_evidence_ids_never_locators(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        [evidence_id] = list(packet_entry.evidence_index.keys())

        prompt = build_suggestion_prompt(packet_entry)

        assert FIELD_ID in prompt
        assert evidence_id in prompt
        assert "N=120 participants were enrolled." in prompt
        assert RECORD_ID in prompt
        assert "never invent a new locator" in prompt

    def test_includes_candidate_ids(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _field_result_with_value_candidate()
        packet_entry = _packet_entry(field_spec, field_result)
        [candidate] = packet_entry.candidates

        prompt = build_suggestion_prompt(packet_entry)

        assert candidate.candidate_id in prompt
        assert "120" in prompt


class TestParseSuggestionResponse:
    def test_parses_valid_json_string(self) -> None:
        raw_text = (
            '{"outcome": "extracted", "value": "120", "evidence_ids": ["e1"], '
            '"candidate_ids_merged": [], "candidate_ids_retained": [], "candidate_ids_rejected": [], '
            '"rationale": "clear", "confidence": "high", "uncertainty": null, "requires_human_review": false}'
        )
        result = parse_suggestion_response(raw_text)
        assert isinstance(result, RawSuggestion)
        assert result.outcome == "extracted"
        assert result.value == "120"
        assert result.evidence_ids == ("e1",)

    def test_parses_dict_directly(self) -> None:
        result = parse_suggestion_response({"outcome": "not_reported"})
        assert result.outcome == "not_reported"
        assert result.value is None

    def test_invalid_json_raises_parse_error(self) -> None:
        with pytest.raises(SuggestionParseError, match="not valid JSON"):
            parse_suggestion_response("{not json")

    def test_unexpected_field_raises_parse_error(self) -> None:
        with pytest.raises(SuggestionParseError, match="expected suggestion schema"):
            parse_suggestion_response({"outcome": "not_reported", "unexpected_field": "x"})

    def test_wrong_type_raises_parse_error(self) -> None:
        with pytest.raises(SuggestionParseError):
            parse_suggestion_response({"outcome": "extracted", "confidence": "extremely_high"})

    def test_non_string_non_dict_raises_parse_error(self) -> None:
        with pytest.raises(SuggestionParseError, match="must be a JSON string or object"):
            parse_suggestion_response(12345)


class TestValidateSuggestion:
    def _validate(self, raw: RawSuggestion, field_spec: ExtractionFieldSpec, field_result: FieldResult, evidence_index):
        return validate_suggestion(
            raw, field_spec=field_spec, field_result=field_result, evidence_index=evidence_index,
            **_identity_kwargs(),
        )

    def test_valid_extracted_suggestion_is_accepted(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        [evidence_id] = list(packet_entry.evidence_index.keys())
        raw = RawSuggestion(outcome="extracted", value="120", evidence_ids=(evidence_id,), confidence="high")

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, ValidatedSuggestion)
        assert result.outcome == "extracted"
        assert result.value == "120"
        assert result.typed_value == 120
        assert result.evidence_ids == (evidence_id,)
        assert result.extraction_run_id == EXTRACTION_RUN_ID
        assert result.record_id == RECORD_ID
        assert result.field_id == FIELD_ID

    def test_not_reported_suggestion_is_accepted(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        raw = RawSuggestion(outcome="not_reported")

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, ValidatedSuggestion)
        assert result.outcome == "not_reported"
        assert result.value is None
        assert result.evidence_ids == ()

    def test_schema_invalid_value_is_rejected(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        [evidence_id] = list(packet_entry.evidence_index.keys())
        raw = RawSuggestion(outcome="extracted", value="not-a-number", evidence_ids=(evidence_id,))

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, RejectedSuggestion)
        assert any("not valid" in reason for reason in result.rejection_reasons)

    def test_unknown_evidence_id_is_rejected(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        raw = RawSuggestion(outcome="extracted", value="120", evidence_ids=("nonexistent-evidence-id",))

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, RejectedSuggestion)
        assert any("evidence index" in reason for reason in result.rejection_reasons)

    def test_unknown_candidate_id_is_rejected(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _field_result_with_value_candidate()
        packet_entry = _packet_entry(field_spec, field_result)
        [evidence_id] = list(packet_entry.evidence_index.keys())
        raw = RawSuggestion(
            outcome="extracted", value="120", evidence_ids=(evidence_id,),
            candidate_ids_merged=("nonexistent-candidate-id",),
        )

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, RejectedSuggestion)
        assert any("candidate index" in reason for reason in result.rejection_reasons)

    def test_extracted_with_no_evidence_is_rejected(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        raw = RawSuggestion(outcome="extracted", value="120", evidence_ids=())

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, RejectedSuggestion)
        assert any("at least one evidence_id" in reason for reason in result.rejection_reasons)

    def test_not_reported_with_value_is_rejected(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        raw = RawSuggestion(outcome="not_reported", value="120")

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, RejectedSuggestion)
        assert any("must not carry a value" in reason for reason in result.rejection_reasons)

    def test_invalid_outcome_string_is_rejected(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        raw = RawSuggestion(outcome="maybe")

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert isinstance(result, RejectedSuggestion)
        assert any("is not one of" in reason for reason in result.rejection_reasons)

    def test_rejected_suggestion_still_carries_identity(self) -> None:
        field_spec = _integer_field_spec()
        field_result = _unclear_field_result_with_review_evidence()
        packet_entry = _packet_entry(field_spec, field_result)
        raw = RawSuggestion(outcome="maybe")

        result = self._validate(raw, field_spec, field_result, packet_entry.evidence_index)

        assert result.extraction_run_id == EXTRACTION_RUN_ID
        assert result.record_id == RECORD_ID
        assert result.field_id == FIELD_ID
        assert result.raw_response == raw.model_dump(mode="json")
