"""Result models -- what a single extracted field, and a whole document's
extraction, actually contain. Several invariants are enforced STRUCTURALLY
(construction fails, not just "should be checked later"), matching this
whole project's established "structural elimination beats a runtime
check" convention:

  1. `not_reported` is permitted only when every planned batch for that
     document completed (requirement 8) -- enforced by
     `DocumentExtractionResult`'s own validator, not by caller discipline.
  2. A field whose status is unclear/conflicting/extraction_failed MUST
     set `requires_human_review=True` -- there is no way to construct a
     `FieldResult` that silently disagrees with its own status.
  3. An INCOMPLETE document can never carry an `extracted`/`conflicting`
     field -- a value found before a later batch failed is real evidence,
     but it is not a trustworthy FINAL answer (a still-unprocessed batch
     could contradict it); such a value belongs only in
     `provisional_values` on an `extraction_failed` result, never in the
     ordinary `values` a reader would treat as the document's answer.
  4. `review_evidence` (ambiguous/conflicting-with-a-found-value evidence
     that would otherwise be silently discarded once a field has a clean
     `found` value from some other batch) may only be attached to a
     result that already requires human review -- it exists so that
     evidence is never thrown away just because a value was also found,
     never so a "clean" result can quietly carry unexplained baggage.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import VerifiedEvidence

TypedValue = Union[str, int, float, bool]


class FieldStatus(str, Enum):
    EXTRACTED = "extracted"
    NOT_REPORTED = "not_reported"
    UNCLEAR = "unclear"
    CONFLICTING = "conflicting"
    EXTRACTION_FAILED = "extraction_failed"


class EvidenceSpanModel(BaseModel):
    """The JSON-serializable form of `evidence.VerifiedEvidence` -- the
    ONLY evidence shape a result ever stores. `page`/`section` are always
    the trusted chunk's own values (see `evidence.py`), never anything a
    model asserted directly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    excerpt: str
    page: Optional[str] = None
    section: str

    @classmethod
    def from_verified(cls, evidence: VerifiedEvidence) -> "EvidenceSpanModel":
        return cls(chunk_id=evidence.chunk_id, excerpt=evidence.excerpt, page=evidence.page, section=evidence.section)


class InvalidValueObservation(BaseModel):
    """A structured, queryable record of ONE model-returned value that
    FAILED `schema.validate_value_for_field` -- review finding C: "the
    raw value, validation reason, and supporting evidence cannot be
    reliably queried or round-tripped as an audit object" when this is
    only present as formatted prose inside `rationale`. `raw_value` is
    preserved verbatim (never rewritten); it NEVER becomes a typed final
    value regardless of what this record says."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_value: str = Field(min_length=1)
    value_kind: str
    reason: str = Field(min_length=1)
    evidence: tuple[EvidenceSpanModel, ...] = ()


class ExtractedValue(BaseModel):
    """One raw, supported value for a field (requirement 12: "Persist raw
    extracted values", never a semantically rewritten/harmonized form --
    harmonization, if ever added, is a strictly later, separate stage
    reading these raw values, never something this module produces).
    `evidence` is non-empty for every value here -- a value with zero
    verified evidence never survives aggregation into an ExtractedValue at
    all (see `aggregation.py`).

    `value` is always the RAW model-returned string, preserved verbatim,
    regardless of whether it passed `schema.validate_value_for_field()`
    (a value only becomes an `ExtractedValue` at all once it HAS passed
    that check -- see `aggregation.py` -- so `value` here is always a
    validated string, but it is still the original string, never a
    reformatted one). `typed_value` is the canonicalized typed form that
    validation computed alongside it (`int`/`float`/`bool` for
    integer/number/boolean fields, the matched string for categorical,
    `None` for text/text_list, where the raw string already IS the
    value)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1)
    typed_value: Optional[TypedValue] = None
    evidence: tuple[EvidenceSpanModel, ...] = Field(min_length=1)


class FieldResult(BaseModel):
    """One field's final, aggregated result for one document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str
    status: FieldStatus
    values: tuple[ExtractedValue, ...] = ()
    # Candidate values seen before an incomplete document's coverage gap
    # made them untrustworthy as a FINAL answer -- only ever present on
    # an `extraction_failed` result (invariant 3 above). Never treated as
    # this field's answer by anything downstream; present purely so a
    # human reviewer (or a later resumed/re-run extraction) isn't working
    # from nothing.
    provisional_values: tuple[ExtractedValue, ...] = ()
    # Evidence a human should look at that is NOT captured by `values`
    # itself -- an ambiguous passage noticed alongside a clean found
    # value elsewhere, or the ambiguous passages behind an `unclear`
    # verdict. Never silently computed and discarded (invariant 4 above).
    review_evidence: tuple[EvidenceSpanModel, ...] = ()
    confidence: Optional[Literal["high", "medium", "low"]] = None
    rationale: Optional[str] = None
    requires_human_review: bool
    # Instrumentation (lit_review_protocol requirement: "Instrument and
    # count every fallback ... and unverified quotation. They must never
    # resemble genuine not_reported results.") -- visible on the result
    # itself, not just in a log line.
    rejected_evidence_count: int = Field(default=0, ge=0)
    # Structured, queryable records of every model-returned value that
    # FAILED `schema.validate_value_for_field` (e.g. a categorical value
    # outside `allowed_values`, or an unparseable integer/number/boolean)
    # -- routed to review via an `unclear` observation, and NEVER allowed
    # to become a typed final value (review finding 3: "an invalid value
    # must become an instrumented review/failure outcome, never a clean
    # extraction"; review finding C: the raw value/reason/evidence must
    # be structured audit data, not prose parsed out of `rationale`).
    invalid_values: tuple[InvalidValueObservation, ...] = ()
    invalid_value_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _requires_review_matches_status(self) -> "FieldResult":
        needs_review_statuses = {FieldStatus.UNCLEAR, FieldStatus.CONFLICTING, FieldStatus.EXTRACTION_FAILED}
        if self.status in needs_review_statuses and not self.requires_human_review:
            raise ValueError(f"status={self.status.value!r} must set requires_human_review=True")
        return self

    @model_validator(mode="after")
    def _values_match_status(self) -> "FieldResult":
        if self.status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING):
            if not self.values:
                raise ValueError(f"status={self.status.value!r} requires at least one value")
        else:
            if self.values:
                raise ValueError(f"status={self.status.value!r} must not carry any values")
        return self

    @model_validator(mode="after")
    def _provisional_values_only_on_extraction_failed(self) -> "FieldResult":
        if self.provisional_values and self.status != FieldStatus.EXTRACTION_FAILED:
            raise ValueError(
                f"provisional_values may only be set when status='extraction_failed', got {self.status.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _review_evidence_requires_human_review(self) -> "FieldResult":
        if self.review_evidence and not self.requires_human_review:
            raise ValueError("review_evidence may only be set when requires_human_review=True")
        return self

    @model_validator(mode="after")
    def _invalid_values_require_human_review(self) -> "FieldResult":
        if self.invalid_values and not self.requires_human_review:
            raise ValueError("invalid_values may only be set when requires_human_review=True")
        return self

    @model_validator(mode="after")
    def _invalid_value_count_matches_list(self) -> "FieldResult":
        if len(self.invalid_values) != self.invalid_value_count:
            raise ValueError(
                f"invalid_value_count ({self.invalid_value_count}) must equal len(invalid_values) "
                f"({len(self.invalid_values)}) -- the count is derived from the structured list, "
                "never tracked separately"
            )
        return self


class BatchExecutionRecord(BaseModel):
    """What actually happened for ONE deterministic batch (see
    `batching.py`) of one document -- the ledger `requirement 8`'s
    `not_reported` invariant is checked against, and the durable,
    checkpoint-worthy record of exactly what was sent and what came back.

    `chunk_ids` binds exactly which chunks were sent in THIS batch (in
    order) -- part of what a checkpoint's own content hash covers,
    so a checkpoint cannot be silently replayed against a differently-
    batched re-run of the same document without that being detectable.

    `provider_model`/`*_tokens` are populated only when a call actually
    completed and returned telemetry (`None` on a transport/coherence
    failure, where usage is genuinely unavailable -- not a sign of a
    missing/broken record; recorded as `None`, never omitted or faked)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_index: int = Field(ge=0)
    chunk_ids: tuple[str, ...] = Field(min_length=1)
    status: Literal["completed", "failed", "skipped"]
    error: Optional[str] = None  # sanitized (type name only), set iff status != "completed"
    provider_model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    rejected_evidence_count: int = Field(default=0, ge=0)
    invalid_value_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _error_matches_status(self) -> "BatchExecutionRecord":
        if self.status == "completed" and self.error is not None:
            raise ValueError("a completed batch must not carry an error")
        if self.status != "completed" and self.error is None:
            raise ValueError(f"status={self.status!r} requires an error description")
        return self


class DocumentExtractionResult(BaseModel):
    """One document's complete extraction result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    fields: tuple[FieldResult, ...] = Field(min_length=1)
    batch_executions: tuple[BatchExecutionRecord, ...] = Field(min_length=1)
    # The REAL planned batch/chunk allocation, in document-batch order,
    # as produced by `batching.build_batches()` BEFORE any provider call
    # was made -- review finding M-residual: binding only a COUNT let a
    # caller declare `planned_batch_count=1` and supply a single
    # completed record for an ARBITRARY (possibly wrong) set of
    # chunk_ids and still construct a "complete" result, regardless of
    # what the real parsed document's batch plan actually was.
    # `planned_batch_count` is deliberately NOT a separate field anymore
    # (per the review's own "ideally derive planned_batch_count from the
    # persisted plan instead of accepting both independently") --
    # `len(planned_chunk_ids_by_batch)` IS the planned batch count, a
    # single source of truth.
    planned_chunk_ids_by_batch: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @property
    def is_complete(self) -> bool:
        return all(b.status == "completed" for b in self.batch_executions)

    @model_validator(mode="after")
    def _batch_executions_cover_exactly_the_planned_batches(self) -> "DocumentExtractionResult":
        """Review finding M (and its residual): `is_complete` previously
        meant only "every PRESENT record says completed" -- it never
        bound the planned batch count/order/CHUNK ALLOCATION, so a
        result with one completed record at batch_index=99, two
        duplicate batch_index=0 records, or (the residual) one completed
        record whose `chunk_ids` simply didn't match what the real batch
        plan actually put in batch 0, was all accepted as a legitimate
        (and possibly "complete") document. Requiring batch_index values
        to be the exact, unique, contiguous range
        `0..len(planned_chunk_ids_by_batch)-1`, AND requiring each
        record's own `chunk_ids` to exactly equal the planned allocation
        for that SAME index, closes all three: a gap-with-a-stray-index,
        a duplicate, and a substituted/incomplete chunk set."""
        planned_count = len(self.planned_chunk_ids_by_batch)
        indices = sorted(b.batch_index for b in self.batch_executions)
        if indices != list(range(planned_count)):
            raise ValueError(
                f"batch_executions must cover exactly batch_index 0..{planned_count - 1} "
                f"(planned batch count={planned_count}), got indices "
                f"{[b.batch_index for b in self.batch_executions]}"
            )
        for b in self.batch_executions:
            planned_chunk_ids = self.planned_chunk_ids_by_batch[b.batch_index]
            if b.chunk_ids != planned_chunk_ids:
                raise ValueError(
                    f"batch_index={b.batch_index}'s chunk_ids {b.chunk_ids!r} do not match the planned "
                    f"chunk allocation {planned_chunk_ids!r} for that batch"
                )
        return self

    @model_validator(mode="after")
    def _not_reported_requires_complete_document(self) -> "DocumentExtractionResult":
        """Requirement 8, enforced structurally: 'not_reported is
        permitted only when every planned document batch completed
        successfully. A failed or skipped batch makes the document
        incomplete, not "not reported."' A DocumentExtractionResult
        claiming any field is not_reported while its own batch executions
        show an incomplete run cannot be constructed at all."""
        if not self.is_complete:
            offending = [f.field_id for f in self.fields if f.status == FieldStatus.NOT_REPORTED]
            if offending:
                raise ValueError(
                    f"field(s) {offending} are status='not_reported' but this document's batches are "
                    "incomplete -- an incomplete document must use 'extraction_failed' for fields it "
                    "never successfully addressed, never 'not_reported'"
                )
        return self

    @model_validator(mode="after")
    def _incomplete_document_cannot_expose_final_values(self) -> "DocumentExtractionResult":
        """Invariant 3 (module docstring): an incomplete document can
        never claim `extracted`/`conflicting` for any field -- a value
        seen before a later batch failed is a CANDIDATE
        (`provisional_values` on an `extraction_failed` result), never a
        trustworthy final answer. Defense in depth alongside
        `aggregation.py`'s own logic, which is the thing that is actually
        supposed to guarantee this never happens in the first place."""
        if not self.is_complete:
            offending = [
                f.field_id for f in self.fields if f.status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING)
            ]
            if offending:
                raise ValueError(
                    f"field(s) {offending} claim status='extracted'/'conflicting' but this document's "
                    "batches are incomplete -- an incomplete document must use 'extraction_failed' with "
                    "any candidate value(s) in provisional_values instead"
                )
        return self

    @model_validator(mode="after")
    def _unique_field_ids(self) -> "DocumentExtractionResult":
        ids = [f.field_id for f in self.fields]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate field_id(s) in fields")
        return self
