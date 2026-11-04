"""Deterministic aggregation of per-batch field observations into one
final `FieldResult` per field (requirement 9): identical supported values
are deduplicated (by exact normalized-string equality -- see
`evidence.normalize_for_matching`, reused here, NEVER semantic
similarity); differing values for a single-valued field become
`conflicting`; multiple-valued fields retain every distinct supported
value; nothing here rewrites, paraphrases, or harmonizes a value.

Two review-driven invariants this module is directly responsible for:
  - An INCOMPLETE document's found value(s) become `provisional_values`
    on an `extraction_failed` result, never ordinary `values` -- a value
    seen before a later batch failed is real evidence, but a still-
    unprocessed batch could still contradict it (results.py's own
    structural validators refuse to let this module get this wrong).
  - An `unclear` observation's evidence is NEVER silently discarded just
    because some OTHER batch found a clean value for the same field --
    it survives into `review_evidence`, and its presence forces
    `requires_human_review=True` even on an otherwise-clean `extracted`
    result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .evidence import normalize_for_matching
from .results import EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus, InvalidValueObservation, TypedValue
from .schema import ExtractionFieldSpec


@dataclass(frozen=True)
class BatchFieldObservation:
    """One field's observation from ONE batch, after evidence
    verification (rejected evidence items are counted, not carried
    forward as evidence) and value-kind validation (an invalid value is
    counted, not carried forward as a found value). `outcome`:
      - "found": the batch located a value that passed BOTH evidence
        verification and `schema.validate_value_for_field`; `value` is
        set and `evidence` is non-empty (an observation claiming "found"
        with zero verified evidence, or an invalid value, is a caller bug
        -- see `engine.py`, which never constructs one for either case).
      - "not_found": this batch's text does not address the field at all.
      - "unclear": this batch's text seems to touch the field but no
        confident, valid value could be determined; `evidence` may still
        be non-empty (pointing at the ambiguous passage, or at the
        rejected/invalid claim's source for audit).
    """

    field_id: str
    outcome: Literal["found", "not_found", "unclear"]
    value: Optional[str] = None
    typed_value: Optional[TypedValue] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    rationale: Optional[str] = None
    evidence: tuple[EvidenceSpanModel, ...] = ()
    rejected_evidence_count: int = 0
    invalid_value_count: int = 0
    invalid_value_observation: Optional[InvalidValueObservation] = None


def _dedupe_found(found: list[BatchFieldObservation]) -> tuple[ExtractedValue, ...]:
    """Exact normalized-string dedup, first-seen order preserved (batch
    order) -- shared by both the `values` path (complete document) and
    the `provisional_values` path (incomplete document); the dedup rule
    itself does not depend on which one a caller is building."""
    by_normalized: dict[str, ExtractedValue] = {}
    for observation in found:
        key = normalize_for_matching(observation.value or "")
        if key in by_normalized:
            existing = by_normalized[key]
            by_normalized[key] = ExtractedValue(
                value=existing.value, typed_value=existing.typed_value,
                evidence=existing.evidence + observation.evidence,
            )
        else:
            by_normalized[key] = ExtractedValue(
                value=observation.value, typed_value=observation.typed_value, evidence=observation.evidence,
            )
    return tuple(by_normalized.values())


def aggregate_field(
    field_spec: ExtractionFieldSpec,
    observations: list[BatchFieldObservation],
    *,
    document_is_complete: bool,
) -> FieldResult:
    """`observations` is every batch's observation of this one field, in
    batch order (missing/empty is fine -- a batch that never mentions a
    field simply contributes no observation for it). `document_is_complete`
    is `DocumentExtractionResult.is_complete` computed by the caller from
    the SAME batch run this field's observations came from -- required to
    enforce requirement 8's not_reported invariant (and this module's own
    provisional-values invariant) here, at the one place that actually
    knows whether every batch succeeded.
    """
    rejected_evidence_count = sum(o.rejected_evidence_count for o in observations)
    invalid_values = tuple(
        o.invalid_value_observation for o in observations if o.invalid_value_observation is not None
    )
    invalid_value_count = len(invalid_values)

    found = [o for o in observations if o.outcome == "found"]
    unclear = [o for o in observations if o.outcome == "unclear"]

    unclear_evidence = tuple(e for o in unclear for e in o.evidence)
    unclear_rationale = "; ".join(o.rationale for o in unclear if o.rationale) or None

    if found:
        deduped_values = _dedupe_found(found)

        if not document_is_complete:
            # A value was found before some LATER batch failed -- real
            # evidence, but not a trustworthy final answer (the failed
            # batch could have contradicted it). Recorded as a candidate
            # only; results.py's own validators refuse to let this
            # become ordinary `values` on anything but `extraction_failed`.
            return FieldResult(
                field_id=field_spec.field_id,
                status=FieldStatus.EXTRACTION_FAILED,
                provisional_values=deduped_values,
                review_evidence=unclear_evidence,
                rationale=(
                    "document incomplete: a value was found before a later batch failed, so it is "
                    "recorded as provisional only -- its uniqueness/correctness cannot be confirmed "
                    "until every batch has been processed"
                ),
                requires_human_review=True,
                rejected_evidence_count=rejected_evidence_count,
                invalid_values=invalid_values,
                invalid_value_count=invalid_value_count,
            )

        if field_spec.cardinality == "single" and len(deduped_values) > 1:
            return FieldResult(
                field_id=field_spec.field_id,
                status=FieldStatus.CONFLICTING,
                values=deduped_values,
                review_evidence=unclear_evidence,
                rationale=unclear_rationale,
                requires_human_review=True,
                rejected_evidence_count=rejected_evidence_count,
                invalid_values=invalid_values,
                invalid_value_count=invalid_value_count,
            )

        # A single (or multi-valued-field's several) clean value(s) --
        # but an ambiguous signal from another batch, rejected evidence,
        # or an invalid value elsewhere still forces review and keeps its
        # own evidence attached, rather than a value found in one batch
        # silently erasing what a DIFFERENT batch flagged as ambiguous
        # (requirement 7: "route unverifiable quotations to review";
        # review finding 4: aggregation must not erase ambiguity just
        # because a value was also found).
        has_ambiguity = bool(unclear)
        requires_review = has_ambiguity or rejected_evidence_count > 0 or invalid_value_count > 0
        rationale_parts = [p for p in (found[0].rationale, unclear_rationale if has_ambiguity else None) if p]
        return FieldResult(
            field_id=field_spec.field_id,
            status=FieldStatus.EXTRACTED,
            values=deduped_values,
            review_evidence=unclear_evidence if requires_review else (),
            confidence=found[0].confidence,
            rationale="; ".join(rationale_parts) or None,
            requires_human_review=requires_review,
            rejected_evidence_count=rejected_evidence_count,
            invalid_values=invalid_values,
            invalid_value_count=invalid_value_count,
        )

    if unclear:
        return FieldResult(
            field_id=field_spec.field_id,
            status=FieldStatus.UNCLEAR,
            confidence=None,
            rationale=unclear_rationale,
            review_evidence=unclear_evidence,
            requires_human_review=True,
            rejected_evidence_count=rejected_evidence_count,
            invalid_values=invalid_values,
            invalid_value_count=invalid_value_count,
        )

    if not document_is_complete:
        return FieldResult(
            field_id=field_spec.field_id,
            status=FieldStatus.EXTRACTION_FAILED,
            rationale="document incomplete: not every batch completed, so absence cannot be confirmed",
            requires_human_review=True,
            rejected_evidence_count=rejected_evidence_count,
            invalid_values=invalid_values,
            invalid_value_count=invalid_value_count,
        )

    return FieldResult(
        field_id=field_spec.field_id,
        status=FieldStatus.NOT_REPORTED,
        requires_human_review=False,
        rejected_evidence_count=rejected_evidence_count,
        invalid_values=invalid_values,
        invalid_value_count=invalid_value_count,
    )
