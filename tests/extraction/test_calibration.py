from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.calibration import (
    CalibrationCase, CalibrationCorpus, EvidenceAudit, EvidenceAuditCoverageError, FieldJudgment, QuotationJudgment,
    ReferenceEvidence, build_calibration_worksheet_rows, build_evidence_audit_worksheet_rows,
)
from slr_pipeline.extraction.checkpoint import DocumentCheckpoint
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.parsing import DoclingParserOptions
from slr_pipeline.extraction.results import (
    BatchExecutionRecord, DocumentExtractionResult, EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus,
)
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind


def test_field_judgment_extracted_requires_values():
    judgment = FieldJudgment(field_id="f", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R")
    assert judgment.correct_values == ("RCT",)


def test_field_judgment_extracted_without_values_raises():
    with pytest.raises(Exception, match="requires at least one correct_value"):
        FieldJudgment(field_id="f", correct_status=FieldStatus.EXTRACTED, correct_values=(), reviewer="R")


def test_field_judgment_not_reported_without_values_is_fine():
    judgment = FieldJudgment(field_id="f", correct_status=FieldStatus.NOT_REPORTED, reviewer="R")
    assert judgment.correct_values == ()


def test_field_judgment_not_reported_with_values_raises():
    with pytest.raises(Exception, match="must not carry correct_values"):
        FieldJudgment(field_id="f", correct_status=FieldStatus.NOT_REPORTED, correct_values=("RCT",), reviewer="R")


def test_field_judgment_requires_reviewer():
    with pytest.raises(Exception):
        FieldJudgment(field_id="f", correct_status=FieldStatus.NOT_REPORTED, reviewer="")


def test_field_judgment_reference_evidence_must_support_a_known_correct_value():
    with pytest.raises(Exception, match="supports_value"):
        FieldJudgment(
            field_id="f", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R",
            reference_evidence=(ReferenceEvidence(excerpt="randomized", supports_value="cohort study"),),
        )


def test_field_judgment_reference_evidence_valid_when_supporting_known_value():
    judgment = FieldJudgment(
        field_id="f", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R",
        reference_evidence=(ReferenceEvidence(excerpt="randomized controlled trial", supports_value="RCT", page="3"),),
    )
    assert judgment.reference_evidence[0].page == "3"


def _case(document_id="rec-a", difficulty="routine") -> CalibrationCase:
    return CalibrationCase(
        document_id=document_id, pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category=difficulty,
        field_judgments=(FieldJudgment(field_id="study_design", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
    )


def test_calibration_corpus_difficulty_counts():
    corpus = CalibrationCorpus(
        corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash="a" * 64, expected_field_ids=("study_design",),
        cases=(_case("a", "plumbing"), _case("b", "routine"), _case("c", "routine")),
    )
    assert corpus.difficulty_counts == {"plumbing": 1, "routine": 2}


def test_calibration_corpus_requires_at_least_one_case():
    with pytest.raises(Exception):
        CalibrationCorpus(
            corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
            extraction_schema_hash="a" * 64, expected_field_ids=("study_design",), cases=(),
        )


def test_calibration_corpus_rejects_duplicate_document_ids():
    with pytest.raises(Exception, match="duplicate"):
        CalibrationCorpus(
            corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
            extraction_schema_hash="a" * 64, expected_field_ids=("study_design",), cases=(_case("a"), _case("a")),
        )


def test_calibration_corpus_rejects_case_missing_an_expected_field():
    """Review finding N's central adversarial reproduction: a two-field
    schema whose corpus judges only one field must be refused at
    construction -- never silently accepted as a valid (if partial)
    corpus that could later report coverage_complete=True."""
    with pytest.raises(Exception, match="missing"):
        CalibrationCorpus(
            corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
            extraction_schema_hash="a" * 64, expected_field_ids=("study_design", "sample_size"),
            cases=(_case("a"),),  # only judges "study_design", missing "sample_size"
        )


def test_calibration_corpus_rejects_case_with_unexpected_extra_field():
    judgment_a = FieldJudgment(field_id="study_design", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R")
    judgment_extra = FieldJudgment(field_id="not_in_schema", correct_status=FieldStatus.NOT_REPORTED, reviewer="R")
    case = CalibrationCase(
        document_id="a", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
        field_judgments=(judgment_a, judgment_extra),
    )
    with pytest.raises(Exception, match="extra"):
        CalibrationCorpus(
            corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
            extraction_schema_hash="a" * 64, expected_field_ids=("study_design",), cases=(case,),
        )


def test_calibration_corpus_rejects_duplicate_expected_field_ids():
    with pytest.raises(Exception, match="duplicate"):
        CalibrationCorpus(
            corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
            extraction_schema_hash="a" * 64, expected_field_ids=("study_design", "study_design"), cases=(_case("a"),),
        )


def test_calibration_corpus_content_hash_changes_with_content():
    base_kwargs = dict(
        corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash="a" * 64, expected_field_ids=("study_design",),
    )
    corpus_v1 = CalibrationCorpus(**base_kwargs, cases=(_case("a"),))
    corpus_v2 = CalibrationCorpus(**base_kwargs, cases=(_case("a"), _case("b")))
    assert corpus_v1.content_hash != corpus_v2.content_hash
    corpus_v1_again = CalibrationCorpus(**base_kwargs, cases=(_case("a"),))
    assert corpus_v1.content_hash == corpus_v1_again.content_hash


def _build_schema_single_field(field_id="study_design") -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id=field_id, question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    return ExtractionSchema(schema_id="s", version="1", fields=[field])


def _build_schema_two_fields() -> ExtractionSchema:
    fields = [
        ExtractionFieldSpec(
            field_id=fid, question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
            provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
        )
        for fid in ("study_design", "sample_size")
    ]
    return ExtractionSchema(schema_id="s", version="1", fields=fields)


def test_calibration_corpus_build_derives_identity_from_real_schema():
    """Review finding N-residual's own required correction: "derive the
    corpus schema identity and expected field IDs from an actual
    ExtractionSchema, rather than accepting mutually independent
    copies." `.build()` reads `extraction_schema_id`/
    `extraction_schema_version`/`extraction_schema_hash`/
    `expected_field_ids` ALL from the real schema object -- there is no
    parameter through which a caller could supply a mismatched label."""
    from slr_pipeline.extraction.hashing import sha256_hex

    schema = _build_schema_single_field()
    corpus = CalibrationCorpus.build(corpus_id="c", version="1", schema=schema, cases=(_case("a"),))
    assert corpus.extraction_schema_id == schema.schema_id
    assert corpus.extraction_schema_version == schema.version
    assert corpus.extraction_schema_hash == sha256_hex(schema.model_dump(mode="json"))
    assert corpus.expected_field_ids == ("study_design",)


def test_calibration_corpus_build_with_two_field_schema_derives_both_expected_field_ids():
    schema = _build_schema_two_fields()
    two_field_case = CalibrationCase(
        document_id="a", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
        field_judgments=(
            FieldJudgment(field_id="study_design", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),
            FieldJudgment(field_id="sample_size", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),
        ),
    )
    corpus = CalibrationCorpus.build(corpus_id="c", version="1", schema=schema, cases=(two_field_case,))
    assert set(corpus.expected_field_ids) == {"study_design", "sample_size"}


def test_calibration_corpus_build_rejects_case_not_matching_the_real_schema():
    """A case judging a field the SCHEMA passed to `.build()` doesn't
    actually declare must still be refused -- `.build()` derives
    `expected_field_ids` from the schema and then goes through the SAME
    `_every_case_covers_exactly_the_expected_fields` validator as direct
    construction; there is no bypass."""
    schema = _build_schema_single_field()  # declares only "study_design"
    case_with_wrong_field = CalibrationCase(
        document_id="a", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
        field_judgments=(FieldJudgment(field_id="a_field_not_in_schema", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),),
    )
    with pytest.raises(Exception, match="missing|extra"):
        CalibrationCorpus.build(corpus_id="c", version="1", schema=schema, cases=(case_with_wrong_field,))


def test_calibration_corpus_build_binds_correctly_to_a_checkpoint_from_the_same_schema():
    """The end-to-end regression the review's own required correction is
    FOR: a `CalibrationCorpus` built via `.build()` from a real schema
    must bind cleanly (no `CalibrationBindingError`) against a REAL
    `DocumentCheckpoint` built from that SAME schema object -- both the
    schema hash AND the complete field set agree, because both were
    derived from the identical source rather than independently typed."""
    from slr_pipeline.extraction.reports import build_audited_accuracy_report

    schema = _build_schema_single_field()
    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    result = DocumentExtractionResult(
        document_id="a",
        fields=(FieldResult(
            field_id="study_design", status=FieldStatus.EXTRACTED,
            values=(ExtractedValue(value="RCT", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="randomized", section="Methods"),)),),
            requires_human_review=False,
        ),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
        planned_chunk_ids_by_batch=(("c1",),),
    )
    checkpoint = DocumentCheckpoint.build(
        run_id="extrun-abc", document_id="a", pdf_path="/x.pdf", pdf_sha256="a" * 64,
        schema=schema, parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=config, result=result,
    )
    corpus = CalibrationCorpus.build(corpus_id="c", version="1", schema=schema, cases=(_case("a"),))

    report = build_audited_accuracy_report([checkpoint], corpus)  # must NOT raise CalibrationBindingError
    assert report.correct_count == 1
    assert report.coverage_complete is True


def test_calibration_case_requires_at_least_one_judgment():
    with pytest.raises(Exception):
        CalibrationCase(document_id="a", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine", field_judgments=())


def test_calibration_case_rejects_duplicate_field_judgments():
    judgment = FieldJudgment(field_id="f", correct_status=FieldStatus.NOT_REPORTED, reviewer="R")
    with pytest.raises(Exception, match="duplicate"):
        CalibrationCase(
            document_id="a", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
            field_judgments=(judgment, judgment),
        )


def test_case_map_and_judgment_map():
    corpus = CalibrationCorpus(
        corpus_id="c", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash="a" * 64, expected_field_ids=("study_design",), cases=(_case("a"),),
    )
    assert "a" in corpus.case_map()
    assert "study_design" in corpus.case_map()["a"].judgment_map()


def test_build_calibration_worksheet_rows():
    result = DocumentExtractionResult(
        document_id="rec-a",
        fields=(
            FieldResult(
                field_id="study_design", status=FieldStatus.EXTRACTED,
                values=(ExtractedValue(value="RCT", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="randomized", page="3", section="Methods"),)),),
                confidence="high", requires_human_review=False,
            ),
        ),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
        planned_chunk_ids_by_batch=(("c1",),),
    )
    rows = build_calibration_worksheet_rows(result)
    assert len(rows) == 1
    row = rows[0]
    assert row["document_id"] == "rec-a"
    assert row["field_id"] == "study_design"
    assert row["system_status"] == "extracted"
    assert row["system_values"] == "RCT"
    assert "c1" in row["system_evidence"]
    assert row["human_correct_status"] == ""  # blank -- for a human to fill in


def test_quotation_judgment_construction():
    judgment = QuotationJudgment(
        field_id="f", claimed_value="RCT", chunk_id="c1", excerpt="randomized",
        supports_claimed_value=True, locator_correct=True, reviewer="R",
    )
    assert judgment.supports_claimed_value is True
    assert judgment.claimed_value == "RCT"


def test_quotation_judgment_review_only_span_requires_none_support():
    """A review-only span (`claimed_value=""`) is not evidence for any
    claim -- `supports_claimed_value` must be `None`, structurally
    distinct from a `True`/`False` verdict (review finding F)."""
    judgment = QuotationJudgment(
        field_id="f", claimed_value="", chunk_id="c1", excerpt="ambiguous",
        supports_claimed_value=None, locator_correct=True, reviewer="R",
    )
    assert judgment.supports_claimed_value is None


def test_quotation_judgment_review_only_span_rejects_boolean_support():
    with pytest.raises(Exception, match="review-only span"):
        QuotationJudgment(
            field_id="f", claimed_value="", chunk_id="c1", excerpt="ambiguous",
            supports_claimed_value=False, locator_correct=True, reviewer="R",
        )


def test_quotation_judgment_claim_bearing_span_requires_explicit_support():
    with pytest.raises(Exception, match="requires an explicit supports_claimed_value"):
        QuotationJudgment(
            field_id="f", claimed_value="RCT", chunk_id="c1", excerpt="randomized",
            supports_claimed_value=None, locator_correct=True, reviewer="R",
        )


def test_evidence_audit_rejects_duplicate_judgment_keys():
    judgment = QuotationJudgment(
        field_id="f", claimed_value="RCT", chunk_id="c1", excerpt="randomized",
        supports_claimed_value=True, locator_correct=True, reviewer="R",
    )
    with pytest.raises(Exception, match="duplicate"):
        EvidenceAudit(
            audit_id="ea-1", run_id="extrun-abc", document_id="rec-a", checkpoint_content_hash="a" * 64,
            quotation_judgments=(judgment, judgment),
        )


def test_evidence_audit_allows_empty_quotation_judgments():
    """A result with zero evidence spans (nothing to audit) is a real,
    valid case -- not a construction error (review refinement: "An
    empty/no-quotation system result must remain representable and
    report zero yield rather than making an audit impossible")."""
    audit = EvidenceAudit(audit_id="ea-1", run_id="extrun-abc", document_id="rec-a", checkpoint_content_hash="a" * 64)
    assert audit.quotation_judgments == ()
    assert audit.precision == 0.0
    assert audit.locator_accuracy == 0.0


def test_evidence_audit_precision_excludes_review_only_spans_from_denominator():
    """Review finding F: a review-only span (`supports_claimed_value is
    None`) must be excluded from BOTH the numerator and denominator of
    `precision` -- it is not evidence for any claim, so it can neither
    support nor fail to support one. `locator_accuracy` still counts it
    (every span has SOME expected locator)."""
    audit = EvidenceAudit(
        audit_id="ea-1", run_id="extrun-abc", document_id="rec-a", checkpoint_content_hash="a" * 64,
        quotation_judgments=(
            QuotationJudgment(field_id="f", claimed_value="RCT", chunk_id="c1", excerpt="a", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="f", claimed_value="RCT", chunk_id="c2", excerpt="b", supports_claimed_value=False, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="f", claimed_value="", chunk_id="c3", excerpt="c", supports_claimed_value=None, locator_correct=False, reviewer="R"),
        ),
    )
    assert audit.precision == 0.5  # 1 of 2 CLAIM-BEARING judgments supports its claim -- c3 excluded
    assert audit.locator_accuracy == 2 / 3  # c3 still counts here


def _schema_for_audit() -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    return ExtractionSchema(schema_id="s", version="1", fields=[field])


def _config_for_audit() -> OpenAIExtractionConfig:
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )


def _result_for_audit() -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_id="rec-a",
        fields=(
            FieldResult(
                field_id="study_design", status=FieldStatus.EXTRACTED,
                values=(ExtractedValue(value="RCT", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="randomized", page="3", section="Methods"),)),),
                review_evidence=(EvidenceSpanModel(chunk_id="c2", excerpt="ambiguous mention", page="4", section="Discussion"),),
                requires_human_review=True,
            ),
        ),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1", "c2"), status="completed"),),
        planned_chunk_ids_by_batch=(("c1", "c2"),),
    )


def _checkpoint_for(result: DocumentExtractionResult, *, run_id: str = "extrun-abc") -> DocumentCheckpoint:
    return DocumentCheckpoint.build(
        run_id=run_id, document_id=result.document_id, pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=result,
    )


def test_evidence_audit_build_requires_exact_coverage_rejects_missing_span():
    """Review finding B's adversarial reproduction: a one-row audit
    selecting only one good quotation, while the result actually returned
    two spans, must be REFUSED at construction -- never silently allowed
    to report precision == 1.0 over a cherry-picked subset."""
    checkpoint = _checkpoint_for(_result_for_audit())
    partial_judgments = (
        QuotationJudgment(
            field_id="study_design", claimed_value="RCT", chunk_id="c1", excerpt="randomized",
            supports_claimed_value=True, locator_correct=True, reviewer="R",
        ),
    )
    with pytest.raises(EvidenceAuditCoverageError, match="missing"):
        EvidenceAudit.build(audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=partial_judgments)


def test_evidence_audit_build_rejects_extra_stale_span():
    checkpoint = _checkpoint_for(_result_for_audit())
    full_coverage = (
        QuotationJudgment(field_id="study_design", claimed_value="RCT", chunk_id="c1", excerpt="randomized", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        QuotationJudgment(field_id="study_design", claimed_value="", chunk_id="c2", excerpt="ambiguous mention", supports_claimed_value=None, locator_correct=True, reviewer="R"),
        QuotationJudgment(field_id="study_design", claimed_value="RCT", chunk_id="c9", excerpt="stale span from an earlier run", supports_claimed_value=True, locator_correct=True, reviewer="R"),
    )
    with pytest.raises(EvidenceAuditCoverageError, match="extra"):
        EvidenceAudit.build(audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=full_coverage)


def test_evidence_audit_build_derives_identity_from_checkpoint():
    """Review finding F: `.build()` no longer accepts independent
    run_id/document_id/checkpoint_content_hash -- identity is READ from
    the checkpoint, so there is no parameter through which a caller could
    pass mismatched metadata."""
    checkpoint = _checkpoint_for(_result_for_audit())
    full_coverage = (
        QuotationJudgment(field_id="study_design", claimed_value="RCT", chunk_id="c1", excerpt="randomized", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        QuotationJudgment(field_id="study_design", claimed_value="", chunk_id="c2", excerpt="ambiguous mention", supports_claimed_value=None, locator_correct=True, reviewer="R"),
    )
    audit = EvidenceAudit.build(audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=full_coverage)
    assert audit.run_id == checkpoint.run_id
    assert audit.document_id == checkpoint.document_id
    assert audit.checkpoint_content_hash == checkpoint.content_hash
    assert audit.precision == 1.0  # the one claim-bearing judgment supports its claim


def test_evidence_audit_build_covers_provisional_values():
    """Review finding B: `provisional_values` (an incomplete document's
    candidate findings) must also be part of the required coverage, not
    just final `values` and `review_evidence`."""
    result = DocumentExtractionResult(
        document_id="rec-a",
        fields=(
            FieldResult(
                field_id="study_design", status=FieldStatus.EXTRACTION_FAILED, requires_human_review=True,
                provisional_values=(ExtractedValue(value="RCT", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="randomized", page="3", section="Methods"),)),),
            ),
        ),
        batch_executions=(
            BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),
            BatchExecutionRecord(batch_index=1, chunk_ids=("c2",), status="failed", error="ExtractionCallError"),
        ),
        planned_chunk_ids_by_batch=(("c1",), ("c2",)),
    )
    checkpoint = _checkpoint_for(result)
    with pytest.raises(EvidenceAuditCoverageError, match="missing"):
        EvidenceAudit.build(
            audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=(),  # omits the provisional value's own evidence span
        )
    audit = EvidenceAudit.build(
        audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="study_design", claimed_value="RCT", chunk_id="c1", excerpt="randomized", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    assert audit.precision == 1.0


def test_build_evidence_audit_worksheet_rows_covers_values_and_review_evidence():
    result = _result_for_audit()
    rows = build_evidence_audit_worksheet_rows(document_id="rec-a", checkpoint_content_hash="a" * 64, result=result)
    assert len(rows) == 2
    chunk_ids = {row["chunk_id"] for row in rows}
    assert chunk_ids == {"c1", "c2"}
    for row in rows:
        assert row["checkpoint_content_hash"] == "a" * 64
        assert row["supports_claimed_value"] == ""
