from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.calibration import (
    CalibrationBindingError, CalibrationCase, CalibrationCorpus, EvidenceAudit, EvidenceAuditCoverageError,
    EvidenceAuditIdentityMismatchError, FieldJudgment, QuotationJudgment, ReferenceEvidence,
)
from slr_pipeline.extraction.checkpoint import (
    CheckpointIntegrityError, DocumentCheckpoint, ExtractionCheckpointStore, RunManifest, TelemetrySummary,
    VerifiedRunBundle,
)
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.parsing import DoclingParserOptions
from slr_pipeline.extraction.reports import (
    build_ad_hoc_extraction_report, build_audited_accuracy_report, build_coverage_report,
    build_evidence_quality_report, build_full_report,
)
from slr_pipeline.extraction.results import (
    BatchExecutionRecord, DocumentExtractionResult, EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus,
)
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind


def _schema_for_audit() -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id="a", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    return ExtractionSchema(schema_id="s", version="1", fields=[field])


_REAL_SCHEMA_HASH = sha256_hex(_schema_for_audit().model_dump(mode="json"))


def _config_for_audit() -> OpenAIExtractionConfig:
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )


def _checkpoint_for(result: DocumentExtractionResult, *, run_id: str = "extrun-abc", pdf_sha256: str = "a" * 64) -> DocumentCheckpoint:
    return DocumentCheckpoint.build(
        run_id=run_id, document_id=result.document_id, pdf_path="/a.pdf", pdf_sha256=pdf_sha256,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=result,
    )


def _evidence() -> EvidenceSpanModel:
    return EvidenceSpanModel(chunk_id="c1", excerpt="x", page="1", section="Methods")


def _extracted(field_id="a", value="RCT") -> FieldResult:
    return FieldResult(
        field_id=field_id, status=FieldStatus.EXTRACTED,
        values=(ExtractedValue(value=value, evidence=(_evidence(),)),), requires_human_review=False,
    )


def _not_reported(field_id="a") -> FieldResult:
    return FieldResult(field_id=field_id, status=FieldStatus.NOT_REPORTED, requires_human_review=False)


def _unclear(field_id="a") -> FieldResult:
    return FieldResult(field_id=field_id, status=FieldStatus.UNCLEAR, requires_human_review=True)


def _result(document_id, fields, batches=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),)) -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_id=document_id, fields=tuple(fields), batch_executions=tuple(batches),
        planned_chunk_ids_by_batch=tuple(b.chunk_ids for b in batches),
    )


def test_coverage_report_basic_counts():
    results = [
        _result("d1", [_extracted("a"), _not_reported("b")]),
        _result("d2", [_not_reported("a"), _unclear("b")]),
    ]
    report = build_coverage_report(results)
    assert report.total_documents == 2
    assert report.complete_documents == 2
    assert report.field_occurrence["a"]["extracted"] == 1
    assert report.field_occurrence["a"]["not_reported"] == 1
    assert report.field_occurrence["b"]["unclear"] == 1
    assert report.review_required_count == 1  # only the unclear one


def test_coverage_report_zero_occurrence_fields():
    results = [_result("d1", [_not_reported("a"), _not_reported("b")])]
    report = build_coverage_report(results)
    assert "a" in report.zero_occurrence_field_ids
    assert "b" in report.zero_occurrence_field_ids


def test_coverage_report_field_ever_extracted_not_zero_occurrence():
    results = [_result("d1", [_extracted("a")]), _result("d2", [_not_reported("a")])]
    report = build_coverage_report(results)
    assert "a" not in report.zero_occurrence_field_ids


def test_coverage_report_incomplete_documents_and_failed_batches():
    results = [
        _result(
            "d1", [FieldResult(field_id="a", status=FieldStatus.EXTRACTION_FAILED, requires_human_review=True)],
            batches=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="failed", error="ExtractionCallError"),),
        ),
    ]
    report = build_coverage_report(results)
    assert report.incomplete_documents == 1
    assert report.failed_batches == 1


def test_coverage_report_empty_results():
    report = build_coverage_report([])
    assert report.total_documents == 0
    assert report.review_required_rate == 0.0


def _corpus(**overrides) -> CalibrationCorpus:
    defaults = dict(
        corpus_id="c1", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash=_REAL_SCHEMA_HASH, expected_field_ids=("a",),
        cases=(
            CalibrationCase(
                document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
                field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
            ),
        ),
    )
    defaults.update(overrides)
    return CalibrationCorpus(**defaults)


def test_audited_accuracy_correct():
    checkpoints = [_checkpoint_for(_result("d1", [_extracted("a", "RCT")]))]
    report = build_audited_accuracy_report(checkpoints, _corpus())
    assert report.correct_count == 1
    assert report.accuracy_rate == 1.0
    assert report.value_correctness_rate == 1.0
    assert report.coverage_complete is True


def test_audited_accuracy_value_mismatch():
    checkpoints = [_checkpoint_for(_result("d1", [_extracted("a", "Cohort study")]))]
    report = build_audited_accuracy_report(checkpoints, _corpus())
    assert report.value_mismatch_count == 1
    assert report.value_correctness_rate == 0.0


def test_audited_accuracy_false_not_reported():
    checkpoints = [_checkpoint_for(_result("d1", [_not_reported("a")]))]
    report = build_audited_accuracy_report(checkpoints, _corpus())
    assert report.false_not_reported_count == 1
    assert report.false_not_reported_rate == 1.0


def test_audited_accuracy_over_extraction():
    corpus = _corpus(cases=(
        CalibrationCase(
            document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="genuinely_not_reported",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),),
        ),
    ))
    checkpoints = [_checkpoint_for(_result("d1", [_extracted("a", "RCT")]))]
    report = build_audited_accuracy_report(checkpoints, corpus)
    assert report.over_extraction_count == 1
    assert report.over_extraction_rate == 1.0


def _conflicting(field_id="a") -> FieldResult:
    return FieldResult(
        field_id=field_id, status=FieldStatus.CONFLICTING,
        values=(ExtractedValue(value="RCT", evidence=(_evidence(),)), ExtractedValue(value="Cohort study", evidence=(_evidence(),))),
        requires_human_review=True,
    )


def test_audited_accuracy_conflicting_against_not_reported_is_over_extraction():
    """Review finding O's adversarial reproduction: a system `conflicting`
    field still asserts one or more values, so it is over-extraction
    against a human `not_reported` verdict -- not a generic
    status_mismatch, which previously understated over_extraction_rate."""
    corpus = _corpus(cases=(
        CalibrationCase(
            document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="genuinely_not_reported",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),),
        ),
    ))
    checkpoints = [_checkpoint_for(_result("d1", [_conflicting("a")]))]
    report = build_audited_accuracy_report(checkpoints, corpus)
    assert report.over_extraction_count == 1
    assert report.status_mismatch_count == 0
    assert report.over_extraction_rate == 1.0


def test_audited_accuracy_skips_documents_not_in_run_and_reports_explicitly():
    """A calibration case for a document that wasn't part of THIS run is
    counted as an explicit skip -- not silently absorbed, and
    `coverage_complete` reflects that the audit is incomplete."""
    report = build_audited_accuracy_report([], _corpus())
    assert report.audited_field_count == 0
    assert report.accuracy_rate == 0.0  # not NaN, not a crash
    assert report.skipped_missing_document_field_count == 1
    assert report.coverage_complete is False
    assert report.expected_field_count == 1


def test_audited_accuracy_rejects_corpus_field_not_present_in_checkpoint():
    """A corpus judgment naming a field_id that does not actually exist
    in the audited checkpoint's own result (e.g. a corpus-authoring
    typo) is now caught by the review finding N-residual defense-in-depth
    check (`expected_field_ids` must equal the checkpoint's own complete
    field set) BEFORE the per-judgment tally loop even runs -- this
    supersedes the older "skipped_missing_field_count" trigger for this
    exact scenario (a hard, immediate rejection is a stronger outcome
    than a silently-partial audit that merely looked "explicit" via a
    skip counter; `skipped_missing_field_count` remains in the report
    schema as defense in depth for any future relaxation of the
    stricter corpus/checkpoint binding, but is not reachable via this
    construction now that the binding is enforced this early)."""
    checkpoints = [_checkpoint_for(_result("d1", [_extracted("a", "RCT")]))]
    corpus_with_typo_field = _corpus(
        expected_field_ids=("a", "nonexistent_field"),
        cases=(
            CalibrationCase(
                document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
                field_judgments=(
                    FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),
                    FieldJudgment(field_id="nonexistent_field", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),
                ),
            ),
        ),
    )
    with pytest.raises(CalibrationBindingError, match="expected_field_ids"):
        build_audited_accuracy_report(checkpoints, corpus_with_typo_field)


def test_value_correctness_rate_excludes_non_value_bearing_correct_judgments():
    """Review finding 5's adversarial reproduction: a corpus containing
    only one CORRECT not_reported judgment (no value-bearing judgments at
    all) must NOT report value_correctness_rate=1.0 -- there is nothing
    to measure value correctness against."""
    corpus = CalibrationCorpus(
        corpus_id="c1", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash=_REAL_SCHEMA_HASH, expected_field_ids=("a",),
        cases=(
            CalibrationCase(
                document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="genuinely_not_reported",
                field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),),
            ),
        ),
    )
    checkpoints = [_checkpoint_for(_result("d1", [_not_reported("a")]))]
    report = build_audited_accuracy_report(checkpoints, corpus)
    assert report.correct_count == 1  # the status match is still tallied
    assert report.value_bearing_judgment_count == 0
    assert report.value_correctness_rate == 0.0  # NOT 1.0


def test_audited_accuracy_rejects_duplicate_checkpoint_documents():
    """Review finding L's adversarial reproduction: duplicate
    document_ids across `checkpoints` must be refused explicitly, never
    silently collapsed via a dict overwrite."""
    checkpoint = _checkpoint_for(_result("d1", [_extracted("a", "RCT")]))
    with pytest.raises(ValueError, match="duplicate"):
        build_audited_accuracy_report([checkpoint, checkpoint], _corpus())


def test_audited_accuracy_checks_every_checkpoint_schema_not_just_first():
    """Review finding L's central adversarial reproduction: 'checkpoints
    from different runs and different schemas were accepted together;
    the first happened to match the corpus, so both contributed to a
    perfect accuracy result.' The SECOND checkpoint's own schema hash
    must be checked too, not only the first one encountered."""
    matching = _checkpoint_for(_result("d1", [_extracted("a", "RCT")]))
    other_schema = ExtractionSchema(
        schema_id="different-schema", version="1",
        fields=[ExtractionFieldSpec(
            field_id="a", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
            provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
        )],
    )
    mismatched = DocumentCheckpoint.build(
        run_id="extrun-abc", document_id="d2", pdf_path="/b.pdf", pdf_sha256="b" * 64,
        schema=other_schema, parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=_result("d2", [_extracted("a", "RCT")]),
    )
    corpus = _corpus(cases=(
        CalibrationCase(
            document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
        ),
        CalibrationCase(
            document_id="d2", pdf_path="/b.pdf", pdf_sha256="b" * 64, difficulty_category="routine",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
        ),
    ))
    with pytest.raises(CalibrationBindingError, match="extraction_schema_hash"):
        build_audited_accuracy_report([matching, mismatched], corpus)


def test_audited_accuracy_rejects_mixed_run_id_checkpoints():
    """Review finding L: checkpoints from DIFFERENT runs (even if each
    individually matches the corpus's schema) must never be combined
    into one accuracy report."""
    checkpoint_run_a = _checkpoint_for(_result("d1", [_extracted("a", "RCT")]), run_id="extrun-aaa")
    checkpoint_run_b = _checkpoint_for(_result("d2", [_extracted("a", "RCT")]), run_id="extrun-bbb")
    corpus = _corpus(cases=(
        CalibrationCase(
            document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
        ),
        CalibrationCase(
            document_id="d2", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
        ),
    ))
    with pytest.raises(CalibrationBindingError, match="multiple run_ids"):
        build_audited_accuracy_report([checkpoint_run_a, checkpoint_run_b], corpus)


def test_audited_accuracy_rejects_incompatible_schema_hash():
    """Review finding I's adversarial reproduction: a checkpoint for
    schema `s@1` paired with a corpus declaring a DIFFERENT
    extraction_schema_hash must be refused BEFORE any tally, never
    silently compared."""
    checkpoints = [_checkpoint_for(_result("d1", [_extracted("a", "RCT")]))]
    incompatible_corpus = _corpus(extraction_schema_hash="f" * 64)
    with pytest.raises(CalibrationBindingError, match="extraction_schema_hash"):
        build_audited_accuracy_report(checkpoints, incompatible_corpus)


def test_audited_accuracy_rejects_incompatible_pdf_hash():
    """Review finding I's adversarial reproduction: a calibration case
    declaring a DIFFERENT pdf_sha256 than the audited checkpoint's own
    must be refused -- a substituted PDF must never be silently compared
    against ground truth collected from a different document."""
    checkpoints = [_checkpoint_for(_result("d1", [_extracted("a", "RCT")]), pdf_sha256="a" * 64)]
    incompatible_corpus = _corpus(cases=(
        CalibrationCase(
            document_id="d1", pdf_path="/x.pdf", pdf_sha256="b" * 64, difficulty_category="routine",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
        ),
    ))
    with pytest.raises(CalibrationBindingError, match="pdf_sha256"):
        build_audited_accuracy_report(checkpoints, incompatible_corpus)


def test_ad_hoc_report_audited_accuracy_none_without_corpus():
    results = [_result("d1", [_extracted("a")])]
    report = build_ad_hoc_extraction_report(run_id="extrun-abc", results=results)
    assert report.audited_accuracy is None
    assert report.coverage.total_documents == 1
    assert report.manifest_hash is None


def test_ad_hoc_report_includes_audited_accuracy_with_corpus():
    result = _result("d1", [_extracted("a", "RCT")])
    checkpoints = [_checkpoint_for(result)]
    report = build_ad_hoc_extraction_report(
        run_id="extrun-abc", results=[result], checkpoints=checkpoints, calibration_corpus=_corpus(),
    )
    assert report.audited_accuracy is not None
    assert report.audited_accuracy.correct_count == 1


def test_ad_hoc_report_requires_checkpoints_when_corpus_supplied():
    results = [_result("d1", [_extracted("a", "RCT")])]
    with pytest.raises(ValueError, match="checkpoints"):
        build_ad_hoc_extraction_report(run_id="extrun-abc", results=results, calibration_corpus=_corpus())


def _two_evidence_result(document_id="d1", value="RCT") -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_id=document_id,
        fields=(FieldResult(
            field_id="a", status=FieldStatus.EXTRACTED,
            values=(ExtractedValue(value=value, evidence=(
                EvidenceSpanModel(chunk_id="c1", excerpt="x", section="S"),
                EvidenceSpanModel(chunk_id="c2", excerpt="y", section="S"),
            )),),
            requires_human_review=False,
        ),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1", "c2"), status="completed"),),
        planned_chunk_ids_by_batch=(("c1", "c2"),),
    )


def test_evidence_quality_report_precision_and_locator_accuracy():
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=False, locator_correct=True, reviewer="R"),
        ),
    )
    report = build_evidence_quality_report([(audit, checkpoint)])
    assert report.audited_quotation_count == 2
    assert report.support_eligible_quotation_count == 2
    assert report.supported_quotation_count == 1
    assert report.support_precision == 0.5
    assert report.locator_accuracy == 1.0
    assert report.reference_evidence_recall is None  # no corpus supplied


def test_evidence_quality_report_support_eligible_count_excludes_review_only_spans():
    """Review finding J: `support_eligible_quotation_count` is the TRUE
    denominator of `support_precision`, distinct from
    `audited_quotation_count` (which still includes review-only spans) --
    `support_precision == supported_quotation_count / support_eligible_quotation_count`
    must be reconstructable from the persisted counts alone."""
    result = DocumentExtractionResult(
        document_id="d1",
        fields=(FieldResult(
            field_id="a", status=FieldStatus.EXTRACTED,
            values=(ExtractedValue(value="RCT", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="x", section="S"),)),),
            review_evidence=(EvidenceSpanModel(chunk_id="c2", excerpt="y", section="S"),),
            requires_human_review=True,
        ),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1", "c2"), status="completed"),),
        planned_chunk_ids_by_batch=(("c1", "c2"),),
    )
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="", chunk_id="c2", excerpt="y", supports_claimed_value=None, locator_correct=False, reviewer="R"),
        ),
    )
    report = build_evidence_quality_report([(audit, checkpoint)])
    assert report.audited_quotation_count == 2
    assert report.support_eligible_quotation_count == 1  # the review-only span excluded
    assert report.supported_quotation_count == 1
    assert report.support_precision == 1.0  # == 1 / 1, reconstructable from persisted counts
    assert report.support_precision == report.supported_quotation_count / report.support_eligible_quotation_count
    assert report.locator_accuracy == 0.5  # locator accuracy still counts both spans


def test_evidence_quality_report_empty_audits_no_crash():
    report = build_evidence_quality_report([])
    assert report.audited_quotation_count == 0
    assert report.support_precision == 0.0


def test_evidence_quality_report_rejects_identity_mismatched_pair():
    """Review finding F: `build_evidence_quality_report()` re-verifies
    every (audit, checkpoint) pair -- a directly-deserialized audit
    claiming a DIFFERENT run/document/hash than the checkpoint it is
    paired with must be refused, never silently trusted."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    mismatched_audit = EvidenceAudit(
        audit_id="ea-1", run_id="extrun-WRONG", document_id="d1", checkpoint_content_hash=checkpoint.content_hash,
        quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    with pytest.raises(EvidenceAuditIdentityMismatchError):
        build_evidence_quality_report([(mismatched_audit, checkpoint)])


def test_evidence_quality_report_rejects_partial_coverage_bypassing_direct_construction():
    """Review finding F's central adversarial reproduction: direct
    `EvidenceAudit(...)` construction (bypassing `.build()`) with only
    ONE of the result's two quotations judged must still be refused when
    building the report -- coverage is re-verified here, at the point of
    use, not merely trusted from construction."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    partial_audit = EvidenceAudit(
        audit_id="ea-1", run_id=checkpoint.run_id, document_id=checkpoint.document_id,
        checkpoint_content_hash=checkpoint.content_hash,
        quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    with pytest.raises(EvidenceAuditCoverageError):
        build_evidence_quality_report([(partial_audit, checkpoint)])


def test_evidence_quality_report_rejects_duplicate_audit_documents():
    """Review finding L's adversarial reproduction: duplicate
    document_ids across `audits_with_checkpoints` must be refused
    explicitly, never silently overwritten via a dict."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_evidence_quality_report([(audit, checkpoint), (audit, checkpoint)])


def test_evidence_quality_report_rejects_mixed_run_checkpoints_without_manifest():
    """Review finding L: checkpoints from DIFFERENT runs must never be
    combined into one evidence-quality report when no run_manifest binds
    them to a single published run."""
    result_a = _two_evidence_result(document_id="d1")
    checkpoint_a = _checkpoint_for(result_a, run_id="extrun-aaa")
    audit_a = EvidenceAudit.build(
        audit_id="ea-a", checkpoint=checkpoint_a, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    result_b = _two_evidence_result(document_id="d2")
    checkpoint_b = _checkpoint_for(result_b, run_id="extrun-bbb")
    audit_b = EvidenceAudit.build(
        audit_id="ea-b", checkpoint=checkpoint_b, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    with pytest.raises(EvidenceAuditIdentityMismatchError, match="multiple run_ids"):
        build_evidence_quality_report([(audit_a, checkpoint_a), (audit_b, checkpoint_b)])


def test_evidence_quality_report_rejects_invalid_manifest_hash():
    """Review finding L's adversarial reproduction: a `run_manifest`
    deserialized with a deliberately false self-hash must be refused
    before it can bind anything -- `build_evidence_quality_report()`
    must not merely compare its `run_id`/`checkpoint_hashes` fields
    against an object nobody has verified came from
    `ExtractionCheckpointStore.load_manifest()`."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    fake_manifest = RunManifest(
        run_id=checkpoint.run_id, extraction_schema_id=checkpoint.extraction_schema_id,
        extraction_schema_version=checkpoint.extraction_schema_version,
        extraction_schema_hash=checkpoint.extraction_schema_hash, parser_name=checkpoint.parser_name,
        parser_version=checkpoint.parser_version, parser_options=checkpoint.parser_options,
        max_batch_chars=checkpoint.max_batch_chars, prompt_template_version=checkpoint.prompt_template_version,
        output_schema_version=checkpoint.output_schema_version, model=checkpoint.model,
        model_config_id=checkpoint.model_config_id, reasoning_effort=checkpoint.reasoning_effort,
        max_completion_tokens=checkpoint.max_completion_tokens,
        expected_provider_model=checkpoint.expected_provider_model, document_ids=("d1",),
        checkpointed_document_ids=("d1",), complete_document_ids=("d1",), incomplete_document_ids=(),
        checkpoint_hashes={"d1": checkpoint.content_hash},
        telemetry_summary=TelemetrySummary(
            total_prompt_tokens=0, total_completion_tokens=0, total_tokens=0,
            distinct_provider_models=(), provider_model_consistent=True, provider_mismatch_count=0,
        ),
        created_at="2026-08-21T00:00:00+00:00", manifest_hash="f" * 64,  # deliberately WRONG
    )
    with pytest.raises(Exception, match="manifest_hash mismatch"):
        build_evidence_quality_report([(audit, checkpoint)], run_manifest=fake_manifest)


def test_evidence_quality_report_rejects_checkpoint_with_invalid_content_hash():
    """Review finding H's adversarial reproduction: a checkpoint whose
    own `content_hash` was replaced (e.g. after an ordinary
    deserialization roundtrip that then had the field tampered) must be
    rejected by `checkpoint.verify()` -- matching an audit's identity
    fields against an UNVERIFIED checkpoint object proves nothing about
    whether the checkpoint itself is a genuine artifact."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    tampered_checkpoint = DocumentCheckpoint.model_validate(
        {**checkpoint.model_dump(mode="json"), "content_hash": "f" * 64},
    )
    audit = EvidenceAudit.build(
        audit_id="ea-1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    # Re-point the audit at the tampered checkpoint's own (also tampered)
    # claimed hash, so identity matching alone would not catch it --
    # only checkpoint.verify()'s own content-hash recomputation does.
    audit_for_tampered = EvidenceAudit(
        audit_id=audit.audit_id, run_id=audit.run_id, document_id=audit.document_id,
        checkpoint_content_hash=tampered_checkpoint.content_hash, quotation_judgments=audit.quotation_judgments,
    )
    with pytest.raises(Exception, match="content_hash mismatch"):
        build_evidence_quality_report([(audit_for_tampered, tampered_checkpoint)])


def test_evidence_quality_report_reference_evidence_recall():
    corpus = CalibrationCorpus(
        corpus_id="c1", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash=_REAL_SCHEMA_HASH, expected_field_ids=("a",),
        cases=(
            CalibrationCase(
                document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
                field_judgments=(
                    FieldJudgment(
                        field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R",
                        reference_evidence=(
                            ReferenceEvidence(excerpt="x", supports_value="RCT", page="3"),
                            ReferenceEvidence(excerpt="never found by the system", supports_value="RCT", page="9"),
                        ),
                    ),
                ),
            ),
        ),
    )
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    report = build_evidence_quality_report([(audit, checkpoint)], corpus)
    assert report.reference_evidence_total == 2
    assert report.reference_evidence_recalled == 1  # only the one the audited system actually found+supported
    assert report.reference_evidence_recall == 0.5


def test_evidence_quality_report_recall_excludes_cross_field_and_cross_value_matches():
    """Review finding F's adversarial reproduction: a supported quote
    attached to a DIFFERENT field, or a DIFFERENT claimed value, than the
    reference evidence's own (field_id, supports_value) must NOT count as
    recalling it -- matching is on the full relation, not bare excerpt
    text within a document."""
    corpus = CalibrationCorpus(
        corpus_id="c1", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash=_REAL_SCHEMA_HASH, expected_field_ids=("a",),
        cases=(
            CalibrationCase(
                document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="routine",
                field_judgments=(
                    FieldJudgment(
                        field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R",
                        reference_evidence=(ReferenceEvidence(excerpt="x", supports_value="RCT", page="3"),),
                    ),
                ),
            ),
        ),
    )
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    # The SAME excerpt text "x" is supported, but for a DIFFERENT claimed
    # value ("cohort study", not "RCT") -- must not count as recall.
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=False, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    report = build_evidence_quality_report([(audit, checkpoint)], corpus)
    assert report.reference_evidence_total == 1
    assert report.reference_evidence_recalled == 0  # "x" was judged unsupported for RCT -- no recall
    assert report.reference_evidence_recall == 0.0


def test_evidence_quality_report_recall_none_absent_reference_evidence_is_zero_recall_not_crash():
    corpus = CalibrationCorpus(
        corpus_id="c1", version="1", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash=_REAL_SCHEMA_HASH, expected_field_ids=("a",),
        cases=(
            CalibrationCase(
                document_id="d1", pdf_path="/x.pdf", pdf_sha256="a" * 64, difficulty_category="genuinely_not_reported",
                field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.NOT_REPORTED, reviewer="R"),),
            ),
        ),
    )
    report = build_evidence_quality_report([], corpus)
    assert report.reference_evidence_total == 0
    assert report.reference_evidence_recall == 0.0  # explicit zero, never None when a corpus WAS supplied


def test_evidence_quality_report_rejects_incompatible_schema_hash():
    """Review finding I's adversarial reproduction: a checkpoint for
    schema `s@1` paired with a corpus declaring a DIFFERENT
    extraction_schema_hash must be refused before computing recall,
    which otherwise silently reported reference_evidence_recall == 1.0
    against an incompatible schema."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    incompatible_corpus = _corpus(extraction_schema_hash="f" * 64)
    with pytest.raises(CalibrationBindingError, match="extraction_schema_hash"):
        build_evidence_quality_report([(audit, checkpoint)], incompatible_corpus)


def test_evidence_quality_report_rejects_incompatible_pdf_hash():
    """Review finding I's adversarial reproduction: a calibration case
    declaring a DIFFERENT pdf_sha256 than the audited checkpoint's own
    must be refused."""
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result, pdf_sha256="a" * 64)
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    incompatible_corpus = _corpus(cases=(
        CalibrationCase(
            document_id="d1", pdf_path="/x.pdf", pdf_sha256="b" * 64, difficulty_category="routine",
            field_judgments=(FieldJudgment(field_id="a", correct_status=FieldStatus.EXTRACTED, correct_values=("RCT",), reviewer="R"),),
        ),
    ))
    with pytest.raises(CalibrationBindingError, match="pdf_sha256"):
        build_evidence_quality_report([(audit, checkpoint)], incompatible_corpus)


def test_ad_hoc_report_includes_evidence_quality_when_audits_supplied():
    result = _two_evidence_result()
    checkpoint = _checkpoint_for(result)
    audit = EvidenceAudit.build(
        audit_id="ea-d1", checkpoint=checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    report = build_ad_hoc_extraction_report(
        run_id="extrun-abc", results=[result], evidence_audits_with_checkpoints=[(audit, checkpoint)],
    )
    assert report.evidence_quality is not None
    assert report.evidence_quality.support_precision == 1.0
    assert report.audited_accuracy is None  # no calibration corpus supplied -- these stay independent
    assert report.manifest_hash is None


def test_ad_hoc_report_evidence_quality_none_without_audits():
    results = [_result("d1", [_extracted("a")])]
    report = build_ad_hoc_extraction_report(run_id="extrun-abc", results=results)
    assert report.evidence_quality is None


def _real_run_id_for_report(document_id="d1", pdf_path="/a.pdf", pdf_sha256="a" * 64) -> str:
    from slr_pipeline.extraction.parsing import PARSER_NAME, installed_docling_version
    from slr_pipeline.extraction.run_identity import DocumentIdentity
    from slr_pipeline.extraction.run_identity import compute_run_id as _compute_run_id

    config = _config_for_audit()
    return _compute_run_id(
        documents=[DocumentIdentity(record_id=document_id, pdf_path=pdf_path, pdf_sha256=pdf_sha256)],
        extraction_schema_hash=_REAL_SCHEMA_HASH,
        parser_name=PARSER_NAME, parser_version=installed_docling_version(),
        parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        prompt_template_version=config.prompt_template_version, output_schema_version=config.output_schema_version,
        model=config.model, model_config_id=config.model_config_id, reasoning_effort=config.reasoning_effort,
        max_completion_tokens=config.max_completion_tokens, expected_provider_model=config.expected_provider_model,
    )


def test_build_full_report_uses_verified_bundle_from_store(tmp_path):
    """Review finding L's own required fix, exercised end-to-end: a
    report built via `VerifiedRunBundle.from_store(store)` has its
    run_id/coverage/audited_accuracy all derived from the SAME
    published, re-verified manifest -- there is no parameter through
    which unrelated results/checkpoints/run_id could be combined
    (contrast with `build_ad_hoc_extraction_report`, which accepts them
    independently and performs no such cross-check)."""
    run_id = _real_run_id_for_report()
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    checkpoint = DocumentCheckpoint.build(
        run_id=run_id, document_id="d1", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=_result("d1", [_extracted("a", "RCT")]),
    )
    store.write_checkpoint(checkpoint)
    store.publish_manifest(document_ids=("d1",), created_at="2026-08-21T00:00:00+00:00")

    bundle = VerifiedRunBundle.from_store(store)
    report = build_full_report(bundle=bundle, calibration_corpus=_corpus())
    assert report.run_id == run_id
    assert report.manifest_hash == bundle.manifest.manifest_hash
    assert report.coverage.total_documents == 1
    assert report.audited_accuracy is not None
    assert report.audited_accuracy.correct_count == 1


def test_build_full_report_rejects_evidence_audit_for_document_outside_bundle(tmp_path):
    """An evidence audit for a document_id that isn't part of the
    bundle's own verified manifest must be refused -- proving there is
    no way to smuggle an unrelated document's audit into a formally
    run-bound report (review finding L's central adversarial
    reproduction, closed structurally: `build_full_report()` takes only
    a `bundle` and a bare list of audits, looking up each one's
    checkpoint FROM the bundle itself)."""
    run_id = _real_run_id_for_report()
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    checkpoint = DocumentCheckpoint.build(
        run_id=run_id, document_id="d1", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=_result("d1", [_extracted("a", "RCT")]),
    )
    store.write_checkpoint(checkpoint)
    store.publish_manifest(document_ids=("d1",), created_at="2026-08-21T00:00:00+00:00")
    bundle = VerifiedRunBundle.from_store(store)

    other_checkpoint = _checkpoint_for(_two_evidence_result(document_id="d9"), run_id="extrun-unrelated")
    foreign_audit = EvidenceAudit.build(
        audit_id="ea-foreign", checkpoint=other_checkpoint, quotation_judgments=(
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c1", excerpt="x", supports_claimed_value=True, locator_correct=True, reviewer="R"),
            QuotationJudgment(field_id="a", claimed_value="RCT", chunk_id="c2", excerpt="y", supports_claimed_value=True, locator_correct=True, reviewer="R"),
        ),
    )
    with pytest.raises(ValueError, match="not part of this bundle"):
        build_full_report(bundle=bundle, evidence_audits=[foreign_audit])


def test_verified_run_bundle_checkpoints_by_id_is_immutable(tmp_path):
    """Review finding L-residual: `bundle.checkpoints_by_id` is a
    read-only view -- assignment must raise, never silently substitute a
    checkpoint that was already verified by `from_store()`."""
    run_id = _real_run_id_for_report()
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    checkpoint = DocumentCheckpoint.build(
        run_id=run_id, document_id="d1", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=_result("d1", [_extracted("a", "RCT")]),
    )
    store.write_checkpoint(checkpoint)
    store.publish_manifest(document_ids=("d1",), created_at="2026-08-21T00:00:00+00:00")
    bundle = VerifiedRunBundle.from_store(store)

    with pytest.raises(TypeError):
        bundle.checkpoints_by_id["d1"] = checkpoint


def test_build_full_report_rejects_mutation_of_bundle_dict_passed_to_constructor(tmp_path):
    """Review finding L-residual's central adversarial reproduction:
    even though `checkpoints_by_id` is defensively copied at
    construction (so external mutation of the CALLER's own dict cannot
    reach an already-built bundle), a bundle built directly with a
    SUBSTITUTED checkpoint (bypassing `from_store()` entirely, e.g. by
    swapping in a different, internally self-consistent checkpoint for
    the same run/schema/PDF but a different extracted value) must still
    be rejected -- `bundle.verify()`, called by `build_full_report()`,
    catches this via the checkpoint-hash-vs-manifest-binding check even
    though the substituted checkpoint's OWN self-hash is perfectly
    valid."""
    run_id = _real_run_id_for_report()
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    original_checkpoint = DocumentCheckpoint.build(
        run_id=run_id, document_id="d1", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=_result("d1", [_extracted("a", "RCT")]),
    )
    store.write_checkpoint(original_checkpoint)
    store.publish_manifest(document_ids=("d1",), created_at="2026-08-21T00:00:00+00:00")
    manifest = store.load_manifest()

    # A DIFFERENT, internally self-consistent checkpoint for the SAME
    # run/schema/PDF, but a different extracted value -- never written
    # to the store, so its own content_hash was never bound by the
    # manifest.
    substituted_checkpoint = DocumentCheckpoint.build(
        run_id=run_id, document_id="d1", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema_for_audit(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config_for_audit(), result=_result("d1", [_extracted("a", "Cohort study")]),
    )
    assert substituted_checkpoint.content_hash != original_checkpoint.content_hash

    forged_bundle = VerifiedRunBundle(manifest=manifest, checkpoints_by_id={"d1": substituted_checkpoint})
    with pytest.raises(CheckpointIntegrityError, match="does not match the hash bound by the manifest"):
        build_full_report(bundle=forged_bundle, calibration_corpus=_corpus())
