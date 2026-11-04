from __future__ import annotations

import sys
from pathlib import Path

from slr_pipeline.extraction.aggregation import BatchFieldObservation, aggregate_field
from slr_pipeline.extraction.results import EvidenceSpanModel, FieldStatus, InvalidValueObservation
from slr_pipeline.extraction.schema import ExtractionFieldSpec, FieldOrigin, FieldProvenance, ValueKind


def _spec(**overrides) -> ExtractionFieldSpec:
    defaults = dict(
        field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    defaults.update(overrides)
    return ExtractionFieldSpec(**defaults)


def _evidence(chunk_id="c1") -> EvidenceSpanModel:
    return EvidenceSpanModel(chunk_id=chunk_id, excerpt="quote", page="1", section="Methods")


def test_found_incomplete_document_becomes_extraction_failed_with_provisional_value():
    """A value found before a LATER batch failed is real evidence, but
    not a trustworthy final answer -- it must land in
    `provisional_values` on an `extraction_failed` result, never in
    `values` on a clean `extracted` one (results.py's own structural
    validators would refuse the latter)."""
    obs = [BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence(),))]
    result = aggregate_field(_spec(), obs, document_is_complete=False)
    assert result.status == FieldStatus.EXTRACTION_FAILED
    assert result.values == ()
    assert result.provisional_values[0].value == "RCT"
    assert result.requires_human_review is True


def test_no_observations_not_reported_when_complete():
    result = aggregate_field(_spec(), [], document_is_complete=True)
    assert result.status == FieldStatus.NOT_REPORTED
    assert result.requires_human_review is False


def test_no_observations_extraction_failed_when_incomplete():
    result = aggregate_field(_spec(), [], document_is_complete=False)
    assert result.status == FieldStatus.EXTRACTION_FAILED
    assert result.requires_human_review is True


def test_all_not_found_observations_same_as_no_observations():
    obs = [BatchFieldObservation(field_id="study_design", outcome="not_found")]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.NOT_REPORTED


def test_single_found_value_extracted():
    obs = [BatchFieldObservation(field_id="study_design", outcome="found", value="RCT", evidence=(_evidence(),))]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert result.values[0].value == "RCT"
    assert result.requires_human_review is False


def test_identical_values_across_batches_deduplicated():
    obs = [
        BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence("c1"),)),
        BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence("c2"),)),
    ]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert len(result.values) == 1  # deduplicated to one ExtractedValue
    assert len(result.values[0].evidence) == 2  # but BOTH pieces of evidence retained


def test_differing_values_single_valued_field_conflicting():
    obs = [
        BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence("c1"),)),
        BatchFieldObservation(field_id="f", outcome="found", value="Cohort study", evidence=(_evidence("c2"),)),
    ]
    spec = _spec(cardinality="single")
    result = aggregate_field(spec, obs, document_is_complete=True)
    assert result.status == FieldStatus.CONFLICTING
    assert result.requires_human_review is True
    assert {v.value for v in result.values} == {"RCT", "Cohort study"}


def test_differing_values_multi_valued_field_retained_not_conflicting():
    obs = [
        BatchFieldObservation(field_id="f", outcome="found", value="anxiety", evidence=(_evidence("c1"),)),
        BatchFieldObservation(field_id="f", outcome="found", value="depression", evidence=(_evidence("c2"),)),
    ]
    spec = _spec(cardinality="multiple", value_kind=ValueKind.TEXT_LIST)
    result = aggregate_field(spec, obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert result.requires_human_review is False
    assert {v.value for v in result.values} == {"anxiety", "depression"}


def test_unclear_when_no_found_but_unclear_present():
    obs = [BatchFieldObservation(field_id="f", outcome="unclear", rationale="ambiguous mention", evidence=(_evidence(),))]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.UNCLEAR
    assert result.requires_human_review is True
    assert "ambiguous" in result.rationale
    assert result.review_evidence == (_evidence(),)  # ambiguous evidence preserved, not discarded


def test_invalid_value_count_propagates_from_observations_to_field_result():
    invalid_obs = InvalidValueObservation(raw_value="NOT_ALLOWED", value_kind="categorical", reason="not in allowed_values")
    obs = [
        BatchFieldObservation(
            field_id="f", outcome="unclear", value="NOT_ALLOWED", invalid_value_count=1,
            invalid_value_observation=invalid_obs,
        ),
        BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence(),)),
    ]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert result.invalid_value_count == 1
    assert result.invalid_values == (invalid_obs,)
    assert result.requires_human_review is True  # an invalid value elsewhere still forces review


def test_found_with_unclear_elsewhere_stays_extracted_but_forces_review_and_keeps_evidence():
    """A value found in one batch and an ambiguous signal from ANOTHER
    batch must not silently erase the ambiguity -- status stays
    'extracted' (the found value is still the field's answer), but
    review is now required and the unclear evidence survives into
    `review_evidence` rather than being discarded (review finding 4)."""
    ambiguous_evidence = _evidence("c9")
    obs = [
        BatchFieldObservation(
            field_id="f", outcome="unclear", rationale="ambiguous in section A", evidence=(ambiguous_evidence,),
        ),
        BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence(),)),
    ]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert result.values[0].value == "RCT"
    assert result.requires_human_review is True
    assert ambiguous_evidence in result.review_evidence


def test_found_with_no_unclear_stays_clean_no_review_evidence():
    obs = [BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence(),))]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert result.requires_human_review is False
    assert result.review_evidence == ()


def test_rejected_evidence_count_forces_review_even_when_extracted():
    obs = [
        BatchFieldObservation(
            field_id="f", outcome="found", value="RCT", evidence=(_evidence(),), rejected_evidence_count=2,
        ),
    ]
    result = aggregate_field(_spec(), obs, document_is_complete=True)
    assert result.status == FieldStatus.EXTRACTED
    assert result.requires_human_review is True
    assert result.rejected_evidence_count == 2


def test_no_semantic_rewriting_distinct_strings_stay_distinct():
    """"RCT" and "Randomized controlled trial" are the SAME concept but
    DIFFERENT strings -- aggregation must not merge them (requirement 9:
    "no semantic rewriting or construct harmonisation during
    extraction")."""
    obs = [
        BatchFieldObservation(field_id="f", outcome="found", value="RCT", evidence=(_evidence("c1"),)),
        BatchFieldObservation(
            field_id="f", outcome="found", value="Randomized controlled trial", evidence=(_evidence("c2"),),
        ),
    ]
    result = aggregate_field(_spec(cardinality="single"), obs, document_is_complete=True)
    assert result.status == FieldStatus.CONFLICTING
    assert len(result.values) == 2
