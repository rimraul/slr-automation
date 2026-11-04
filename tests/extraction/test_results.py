from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.results import (
    BatchExecutionRecord, DocumentExtractionResult, EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus,
)


def _evidence() -> EvidenceSpanModel:
    return EvidenceSpanModel(chunk_id="c1", excerpt="quote", page="5", section="Methods")


def test_field_result_extracted_requires_values():
    with pytest.raises(Exception, match="at least one value"):
        FieldResult(field_id="f", status=FieldStatus.EXTRACTED, values=(), requires_human_review=False)


def test_field_result_not_reported_must_not_carry_values():
    with pytest.raises(Exception, match="must not carry"):
        FieldResult(
            field_id="f", status=FieldStatus.NOT_REPORTED,
            values=(ExtractedValue(value="x", evidence=(_evidence(),)),), requires_human_review=False,
        )


def test_field_result_unclear_must_set_requires_human_review():
    with pytest.raises(Exception, match="requires_human_review"):
        FieldResult(field_id="f", status=FieldStatus.UNCLEAR, requires_human_review=False)


def test_field_result_conflicting_must_set_requires_human_review():
    with pytest.raises(Exception, match="requires_human_review"):
        FieldResult(
            field_id="f", status=FieldStatus.CONFLICTING,
            values=(ExtractedValue(value="a", evidence=(_evidence(),)),), requires_human_review=False,
        )


def test_field_result_extraction_failed_must_set_requires_human_review():
    with pytest.raises(Exception, match="requires_human_review"):
        FieldResult(field_id="f", status=FieldStatus.EXTRACTION_FAILED, requires_human_review=False)


def test_field_result_extracted_can_have_review_false():
    result = FieldResult(
        field_id="f", status=FieldStatus.EXTRACTED,
        values=(ExtractedValue(value="a", evidence=(_evidence(),)),), requires_human_review=False,
    )
    assert result.requires_human_review is False


def test_extracted_value_requires_nonempty_evidence():
    with pytest.raises(Exception):
        ExtractedValue(value="a", evidence=())


def test_extracted_value_carries_typed_value():
    value = ExtractedValue(value="42", typed_value=42, evidence=(_evidence(),))
    assert value.typed_value == 42
    assert value.value == "42"  # raw string preserved regardless of typed form


def test_batch_execution_completed_must_not_carry_error():
    with pytest.raises(Exception, match="must not carry"):
        BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed", error="oops")


def test_batch_execution_failed_requires_error():
    with pytest.raises(Exception, match="requires an error"):
        BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="failed", error=None)


def test_batch_execution_requires_at_least_one_chunk_id():
    with pytest.raises(Exception):
        BatchExecutionRecord(batch_index=0, chunk_ids=(), status="completed")


def test_batch_execution_records_provider_telemetry():
    record = BatchExecutionRecord(
        batch_index=0, chunk_ids=("c1", "c2"), status="completed",
        provider_model="gpt-5.6-terra", prompt_tokens=10, completion_tokens=5, total_tokens=15,
    )
    assert record.provider_model == "gpt-5.6-terra"
    assert record.total_tokens == 15


def test_batch_execution_failed_batch_has_no_telemetry_by_default():
    record = BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="failed", error="ExtractionCallError")
    assert record.provider_model is None
    assert record.prompt_tokens is None


def test_field_result_provisional_values_forbidden_outside_extraction_failed():
    with pytest.raises(Exception, match="provisional_values"):
        FieldResult(
            field_id="f", status=FieldStatus.NOT_REPORTED, requires_human_review=False,
            provisional_values=(ExtractedValue(value="a", evidence=(_evidence(),)),),
        )


def test_field_result_extraction_failed_may_carry_provisional_values():
    result = FieldResult(
        field_id="f", status=FieldStatus.EXTRACTION_FAILED, requires_human_review=True,
        provisional_values=(ExtractedValue(value="a", evidence=(_evidence(),)),),
    )
    assert result.provisional_values[0].value == "a"
    assert result.values == ()  # ordinary `values` still forbidden on extraction_failed


def test_field_result_review_evidence_requires_human_review():
    with pytest.raises(Exception, match="review_evidence"):
        FieldResult(
            field_id="f", status=FieldStatus.EXTRACTED, requires_human_review=False,
            values=(ExtractedValue(value="a", evidence=(_evidence(),)),),
            review_evidence=(_evidence(),),
        )


def test_field_result_extracted_may_carry_review_evidence_when_flagged():
    result = FieldResult(
        field_id="f", status=FieldStatus.EXTRACTED, requires_human_review=True,
        values=(ExtractedValue(value="a", evidence=(_evidence(),)),),
        review_evidence=(_evidence(),),
    )
    assert result.review_evidence[0].chunk_id == "c1"


def _field(status=FieldStatus.NOT_REPORTED, **overrides) -> FieldResult:
    defaults = dict(field_id="f", status=status, requires_human_review=(status != FieldStatus.NOT_REPORTED and status != FieldStatus.EXTRACTED))
    if status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING):
        defaults["values"] = (ExtractedValue(value="a", evidence=(_evidence(),)),)
        defaults["requires_human_review"] = status == FieldStatus.CONFLICTING
    defaults.update(overrides)
    return FieldResult(**defaults)


def test_document_result_rejects_not_reported_when_batches_incomplete():
    with pytest.raises(Exception, match="not_reported"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="failed", error="ExtractionCallError"),),
            planned_chunk_ids_by_batch=(("c1",),),
        )


def test_document_result_allows_not_reported_when_batches_complete():
    result = DocumentExtractionResult(
        document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
        planned_chunk_ids_by_batch=(("c1",),),
    )
    assert result.is_complete is True


def test_document_result_rejects_duplicate_field_ids():
    with pytest.raises(Exception, match="duplicate"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(), _field()),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
            planned_chunk_ids_by_batch=(("c1",),),
        )


def test_document_result_is_complete_false_on_any_failed_batch():
    result = DocumentExtractionResult(
        document_id="d1",
        fields=(_field(FieldStatus.EXTRACTION_FAILED, requires_human_review=True),),
        batch_executions=(
            BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),
            BatchExecutionRecord(batch_index=1, chunk_ids=("c2",), status="failed", error="ExtractionCallError"),
        ),
        planned_chunk_ids_by_batch=(("c1",), ("c2",)),
    )
    assert result.is_complete is False


def test_document_result_rejects_extracted_field_when_document_incomplete():
    """Invariant 3: an incomplete document can never expose an ordinary
    `extracted` value -- defense in depth, even though `aggregation.py`
    is what is actually supposed to prevent this from arising."""
    with pytest.raises(Exception, match="extracted"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.EXTRACTED),),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="failed", error="ExtractionCallError"),),
            planned_chunk_ids_by_batch=(("c1",),),
        )


def test_document_result_rejects_conflicting_field_when_document_incomplete():
    with pytest.raises(Exception, match="conflicting"):
        DocumentExtractionResult(
            document_id="d1",
            fields=(_field(FieldStatus.CONFLICTING, values=(
                ExtractedValue(value="a", evidence=(_evidence(),)),
                ExtractedValue(value="b", evidence=(_evidence(),)),
            )),),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="failed", error="ExtractionCallError"),),
            planned_chunk_ids_by_batch=(("c1",),),
        )


def test_document_result_allows_extraction_failed_with_provisional_values_when_incomplete():
    result = DocumentExtractionResult(
        document_id="d1",
        fields=(
            FieldResult(
                field_id="f", status=FieldStatus.EXTRACTION_FAILED, requires_human_review=True,
                provisional_values=(ExtractedValue(value="a", evidence=(_evidence(),)),),
            ),
        ),
        batch_executions=(
            BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),
            BatchExecutionRecord(batch_index=1, chunk_ids=("c2",), status="failed", error="ExtractionCallError"),
        ),
        planned_chunk_ids_by_batch=(("c1",), ("c2",)),
    )
    assert result.is_complete is False
    assert result.fields[0].provisional_values[0].value == "a"


def test_document_result_rejects_stray_batch_index_not_matching_planned_count():
    """Review finding M's adversarial reproduction: a lone completed
    record at batch_index=99 must not be accepted as "complete" for a
    document whose real plan was a single batch (index 0) -- the planned
    range 0..len(planned_chunk_ids_by_batch)-1 must be covered EXACTLY."""
    with pytest.raises(Exception, match="batch_index"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
            batch_executions=(BatchExecutionRecord(batch_index=99, chunk_ids=("c1",), status="completed"),),
            planned_chunk_ids_by_batch=(("c1",),),
        )


def test_document_result_rejects_duplicate_batch_index():
    """Review finding M's adversarial reproduction: two records both
    claiming batch_index=0 must not be accepted as covering a two-batch
    plan -- duplicates never satisfy the exact contiguous-range
    requirement."""
    with pytest.raises(Exception, match="batch_index"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
            batch_executions=(
                BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),
                BatchExecutionRecord(batch_index=0, chunk_ids=("c2",), status="completed"),
            ),
            planned_chunk_ids_by_batch=(("c1",), ("c2",)),
        )


def test_document_result_rejects_fewer_records_than_planned_batch_count():
    """A single completed record cannot satisfy a two-batch plan -- the
    SECOND planned batch was simply never attempted/recorded, which must
    be visible as an incomplete plan, not silently treated as if only
    one batch had ever been planned."""
    with pytest.raises(Exception, match="batch_index"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
            planned_chunk_ids_by_batch=(("c1",), ("c2",)),
        )


def test_document_result_rejects_chunk_ids_substituted_from_the_real_plan():
    """Review finding M-residual's central adversarial reproduction: a
    result declaring one planned batch and one completed batch_index=0
    record whose `chunk_ids` do NOT match the real planned allocation
    for that batch must be refused -- binding only a COUNT let a caller
    declare a matching batch_index/count while supplying an arbitrary
    (possibly wrong) chunk set and still construct a "complete" result."""
    with pytest.raises(Exception, match="do not match the planned"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("only-first-of-real-plan",), status="completed"),),
            planned_chunk_ids_by_batch=(("only-first-of-real-plan", "second-chunk-of-real-plan"),),
        )


def test_document_result_rejects_chunk_ids_omitted_from_the_real_plan():
    """A batch whose real plan allocated TWO chunks but whose execution
    record only carries ONE of them must be refused -- a subset of the
    real plan is not the real plan."""
    with pytest.raises(Exception, match="do not match the planned"):
        DocumentExtractionResult(
            document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
            batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
            planned_chunk_ids_by_batch=(("c1", "c2"),),
        )


def test_document_result_accepts_matching_multi_chunk_batch_plan():
    result = DocumentExtractionResult(
        document_id="d1", fields=(_field(FieldStatus.NOT_REPORTED),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1", "c2"), status="completed"),),
        planned_chunk_ids_by_batch=(("c1", "c2"),),
    )
    assert result.is_complete is True
