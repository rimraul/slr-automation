"""Per-attempt artifact schemas for the extraction resolution-suggestion
workflow (amber-quilting-osprey.md sec. 3.1/3.4).

Every attempt to resolve one queued `(record_id, field_id)` terminates in
exactly one of four durable, immutable variants -- `AttemptRecord =
Union[ValidatedSuggestion, RejectedSuggestion, FailedCallRecord,
InterruptedCallRecord]`, discriminated by `kind`. All four share a common
identity (`attempt_record_id`/`self_hash`, plus the precomputed
`ai_use_event_id` from the crash-safe reservation step, sec. 3.4) even
though only three of the four (never `InterruptedCallRecord`, which never
gets a published `AIUseEvent`) carry the pre-call `AIUseEventDraft`
needed to actually construct one.

Every variant's `ai_use_event_id` is cross-checked against
`ai_traceability.compute_event_id()`, and every variant's `work_item_id`
(on the draft, or directly for `InterruptedCallRecord`) is cross-checked
against `compute_expected_work_item_id()` -- a record whose identity
fields and declared event id/work-item id do not actually agree is
rejected at construction, never silently trusted.

Builders (`build_validated_suggestion()` etc.) are the only place these
records are constructed -- same self-verifying-hash pattern as
`ai_traceability.build_ai_use_event()`: construct once with a placeholder
hash via `model_construct`, compute the real hash from that, then
`model_validate` the final, self-consistent record.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.ai_traceability import compute_event_id
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.results import FieldResult, TypedValue

_ATTEMPT_RECORD_ID_SCHEMA_VERSION = 1

# Shared with the ai_traceability_report.py STAGE_ORDER wiring (sec. 3.4) --
# defined once here so both sides can never drift apart.
EXTRACTION_RESOLUTION_SUGGESTION_STAGE = "extraction_resolution_suggestion"

Outcome = Literal["extracted", "not_reported", "unclear", "conflicting", "extraction_failed"]

_SELF_HASH_EXCLUDED_FIELDS = {"self_hash"}


def compute_attempt_record_id(*, extraction_run_id: str, record_id: str, field_id: str, attempt: int) -> str:
    return sha256_hex(
        {
            "id_schema_version": _ATTEMPT_RECORD_ID_SCHEMA_VERSION,
            "extraction_run_id": extraction_run_id,
            "record_id": record_id,
            "field_id": field_id,
            "attempt": attempt,
        }
    )[:24]


def compute_expected_work_item_id(*, extraction_run_id: str, record_id: str, field_id: str) -> str:
    return f"{extraction_run_id}:{record_id}:{field_id}"


def _check_identity_consistency(
    *,
    extraction_run_id: str,
    record_id: str,
    field_id: str,
    attempt: int,
    attempt_record_id: str,
    review_run_id: str,
    work_item_id: str,
    draft_attempt: Optional[int],
    ai_use_event_id: str,
) -> None:
    """Shared cross-check run by every variant's own identity validator:
    `attempt_record_id` must be exactly `compute_attempt_record_id()` of
    this record's own identity fields (a fabricated-but-24-hex-char id
    with an otherwise-consistent, correctly-recomputed `self_hash` is
    rejected here, not merely accepted because its own hash is
    internally consistent), the declared `work_item_id` must be exactly
    what this record's own `extraction_run_id`/`record_id`/`field_id`
    implies, a carried draft's own `attempt` must match this record's
    `attempt`, and `ai_use_event_id` must be exactly `compute_event_id()`
    of this record's own `review_run_id`/`work_item_id`/`attempt` -- a
    caller-supplied event id that does not actually derive from this
    record's own identity is rejected here, never merely stored as-is."""
    expected_attempt_record_id = compute_attempt_record_id(
        extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, attempt=attempt,
    )
    if attempt_record_id != expected_attempt_record_id:
        raise ValueError(
            f"attempt_record_id {attempt_record_id!r} does not match compute_attempt_record_id(...) = "
            f"{expected_attempt_record_id!r} for this record's own extraction_run_id/record_id/field_id/attempt"
        )
    expected_work_item_id = compute_expected_work_item_id(
        extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id,
    )
    if work_item_id != expected_work_item_id:
        raise ValueError(
            f"work_item_id {work_item_id!r} does not match expected {expected_work_item_id!r} derived from "
            "this record's own extraction_run_id/record_id/field_id"
        )
    if draft_attempt is not None and draft_attempt != attempt:
        raise ValueError(
            f"ai_use_event_draft.attempt ({draft_attempt}) does not match this record's own attempt ({attempt})"
        )
    expected_event_id = compute_event_id(
        review_run_id=review_run_id, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
        work_item_id=work_item_id, attempt=attempt,
    )
    if ai_use_event_id != expected_event_id:
        raise ValueError(
            f"ai_use_event_id {ai_use_event_id!r} does not match compute_event_id(...) = {expected_event_id!r} "
            "for this record's own review_run_id/work_item_id/attempt"
        )


class AIUseEventDraft(BaseModel):
    """Every immutable input `ai_traceability.build_ai_use_event()` needs.
    Assembled once this attempt's terminal record is written (sec. 3.4
    step 2, after the call has concluded) -- most fields (model, provider,
    prompt template, parameters, ...) are already fixed as of the
    reservation in step 1, but `event_timestamp` necessarily reflects when
    the call actually concluded, not the earlier reservation time, so it
    cannot itself be "fixed at reservation time." `stage`/`tool_kind`/
    `call_status`/`failure_detail`/`output_artifact_refs` are deliberately
    absent -- they are either fixed constants for this workflow
    (`stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE`, `tool_kind="llm"`) or
    only knowable once the call concludes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_run_id: str
    work_item_id: str
    attempt: int = Field(ge=0)
    model: str
    provider: str
    component_versions: Dict[str, str]
    access_info: Optional[str] = None
    task_purpose: str
    model_config_id: str
    prompt_template_version: Optional[str] = None
    prompt_template_text: Optional[str] = None
    prompt_parameters: Dict[str, Any] = Field(default_factory=dict)
    rendered_prompt_hash: Optional[str] = None
    input_artifact_refs: List[str]
    event_timestamp: str


def _compute_self_hash(model_cls: type, **fields: Any) -> str:
    placeholder = model_cls.model_construct(**fields, self_hash="")
    payload = {
        name: value
        for name, value in placeholder.model_dump(mode="json").items()
        if name not in _SELF_HASH_EXCLUDED_FIELDS
    }
    return sha256_hex(payload)


class ValidatedSuggestion(BaseModel):
    """A call that succeeded and passed `validate_suggestion()` (sec.
    3.3) -- the LLM's structured, grounded recommendation. `evidence_ids`/
    `candidate_ids_*` reference only IDs minted by `render_review_packet()`
    (sec. 3.2); the human-facing excerpt/locator is always rehydrated from
    the packet's evidence index by ID, never taken from LLM text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["validated_suggestion"] = "validated_suggestion"
    attempt_record_id: str
    self_hash: str
    suggestion_id: str
    ai_use_event_id: str
    ai_use_event_draft: AIUseEventDraft

    extraction_run_id: str
    record_id: str
    field_id: str
    attempt: int = Field(ge=0)
    original_field_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    outcome: Outcome
    value: Optional[str] = None
    typed_value: Optional[TypedValue] = None
    evidence_ids: tuple[str, ...] = ()
    candidate_ids_merged: tuple[str, ...] = ()
    candidate_ids_retained: tuple[str, ...] = ()
    candidate_ids_rejected: tuple[str, ...] = ()
    rationale: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    uncertainty: Optional[str] = None
    requires_human_review: bool = True

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    created_at: str

    @model_validator(mode="after")
    def _suggestion_id_matches_attempt_record_id(self) -> "ValidatedSuggestion":
        if self.suggestion_id != self.attempt_record_id:
            raise ValueError(
                f"suggestion_id {self.suggestion_id!r} must equal attempt_record_id {self.attempt_record_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _extracted_outcome_requires_value_and_evidence(self) -> "ValidatedSuggestion":
        if self.outcome == "extracted":
            if self.value is None:
                raise ValueError("outcome='extracted' requires a non-None value")
            if not self.evidence_ids:
                raise ValueError("outcome='extracted' requires at least one evidence_id")
        else:
            if self.value is not None:
                raise ValueError(f"outcome={self.outcome!r} must not carry a value")
            if self.typed_value is not None:
                raise ValueError(f"outcome={self.outcome!r} must not carry a typed_value")
            if self.evidence_ids:
                raise ValueError(f"outcome={self.outcome!r} must not carry evidence_ids")
        return self

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> "ValidatedSuggestion":
        _check_identity_consistency(
            extraction_run_id=self.extraction_run_id, record_id=self.record_id, field_id=self.field_id,
            attempt=self.attempt, attempt_record_id=self.attempt_record_id,
            review_run_id=self.ai_use_event_draft.review_run_id,
            work_item_id=self.ai_use_event_draft.work_item_id, draft_attempt=self.ai_use_event_draft.attempt,
            ai_use_event_id=self.ai_use_event_id,
        )
        return self

    @model_validator(mode="after")
    def _self_hash_is_verifiable(self) -> "ValidatedSuggestion":
        expected = _compute_self_hash(
            type(self), **{k: v for k, v in self.__dict__.items() if k != "self_hash"}
        )
        if self.self_hash != expected:
            raise ValueError(f"self_hash {self.self_hash!r} does not match recomputed hash {expected!r}")
        return self


class RejectedSuggestion(BaseModel):
    """An LLM response that failed structural/schema/evidence-existence
    validation -- persisted, never silently dropped, so a retry or a
    human reviewer can see exactly what was rejected and why. The call
    itself succeeded (it consumed real tokens/cost), so it carries the
    same usage fields as `ValidatedSuggestion`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["rejected_suggestion"] = "rejected_suggestion"
    attempt_record_id: str
    self_hash: str
    suggestion_id: str
    ai_use_event_id: str
    ai_use_event_draft: AIUseEventDraft

    extraction_run_id: str
    record_id: str
    field_id: str
    attempt: int = Field(ge=0)
    original_field_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    raw_response: Any
    rejection_reasons: tuple[str, ...] = Field(min_length=1)

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    created_at: str

    @model_validator(mode="after")
    def _suggestion_id_matches_attempt_record_id(self) -> "RejectedSuggestion":
        if self.suggestion_id != self.attempt_record_id:
            raise ValueError(
                f"suggestion_id {self.suggestion_id!r} must equal attempt_record_id {self.attempt_record_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> "RejectedSuggestion":
        _check_identity_consistency(
            extraction_run_id=self.extraction_run_id, record_id=self.record_id, field_id=self.field_id,
            attempt=self.attempt, attempt_record_id=self.attempt_record_id,
            review_run_id=self.ai_use_event_draft.review_run_id,
            work_item_id=self.ai_use_event_draft.work_item_id, draft_attempt=self.ai_use_event_draft.attempt,
            ai_use_event_id=self.ai_use_event_id,
        )
        return self

    @model_validator(mode="after")
    def _self_hash_is_verifiable(self) -> "RejectedSuggestion":
        expected = _compute_self_hash(
            type(self), **{k: v for k, v in self.__dict__.items() if k != "self_hash"}
        )
        if self.self_hash != expected:
            raise ValueError(f"self_hash {self.self_hash!r} does not match recomputed hash {expected!r}")
        return self


class FailedCallRecord(BaseModel):
    """A caught exception during the real provider call -- `error_message`
    has already been through `sanitize_call_error_text()` (sec. 3.4)
    exactly once, before this record was ever constructed; no field on
    this model is ever populated from an un-sanitized error string."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["failed_call"] = "failed_call"
    attempt_record_id: str
    self_hash: str
    ai_use_event_id: str
    ai_use_event_draft: AIUseEventDraft

    extraction_run_id: str
    record_id: str
    field_id: str
    attempt: int = Field(ge=0)
    original_field_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    error_type: str
    error_message: str
    failed_at: str

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> "FailedCallRecord":
        _check_identity_consistency(
            extraction_run_id=self.extraction_run_id, record_id=self.record_id, field_id=self.field_id,
            attempt=self.attempt, attempt_record_id=self.attempt_record_id,
            review_run_id=self.ai_use_event_draft.review_run_id,
            work_item_id=self.ai_use_event_draft.work_item_id, draft_attempt=self.ai_use_event_draft.attempt,
            ai_use_event_id=self.ai_use_event_id,
        )
        return self

    @model_validator(mode="after")
    def _self_hash_is_verifiable(self) -> "FailedCallRecord":
        expected = _compute_self_hash(
            type(self), **{k: v for k, v in self.__dict__.items() if k != "self_hash"}
        )
        if self.self_hash != expected:
            raise ValueError(f"self_hash {self.self_hash!r} does not match recomputed hash {expected!r}")
        return self


class InterruptedCallRecord(BaseModel):
    """A reservation (`.reserved.json`) whose slot had no per-attempt file
    at all on restart -- true process termination between reservation and
    call conclusion. Archived to `.interrupted_unknown.json`, never
    auto-retried, and this attempt slot is never reused (an explicit
    retry gets a fresh, higher attempt number). No `AIUseEventDraft` --
    the call may never have actually reached the provider, so there is no
    real event to ever publish for this attempt. `review_run_id`/
    `work_item_id` are carried directly (rather than via a draft) since
    they were already fixed, and known, at the earlier reservation step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["interrupted_call"] = "interrupted_call"
    attempt_record_id: str
    self_hash: str
    ai_use_event_id: str
    review_run_id: str
    work_item_id: str

    extraction_run_id: str
    record_id: str
    field_id: str
    attempt: int = Field(ge=0)

    reserved_at: str
    archived_at: str

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> "InterruptedCallRecord":
        _check_identity_consistency(
            extraction_run_id=self.extraction_run_id, record_id=self.record_id, field_id=self.field_id,
            attempt=self.attempt, attempt_record_id=self.attempt_record_id,
            review_run_id=self.review_run_id, work_item_id=self.work_item_id,
            draft_attempt=None, ai_use_event_id=self.ai_use_event_id,
        )
        return self

    @model_validator(mode="after")
    def _self_hash_is_verifiable(self) -> "InterruptedCallRecord":
        expected = _compute_self_hash(
            type(self), **{k: v for k, v in self.__dict__.items() if k != "self_hash"}
        )
        if self.self_hash != expected:
            raise ValueError(f"self_hash {self.self_hash!r} does not match recomputed hash {expected!r}")
        return self


AttemptRecord = Union[ValidatedSuggestion, RejectedSuggestion, FailedCallRecord, InterruptedCallRecord]


class ReservedAttemptMarker(BaseModel):
    """Step 1 of the crash-safe three-step state machine (sec. 3.4):
    written+fsynced to `<safe_key>__attempt-<n>.reserved.json` BEFORE the
    real network call, under `reserved_attempt()`'s own per-work-item
    lock -- so a genuine process termination between reservation and call
    conclusion leaves durable evidence an attempt was ever dispatched.
    `event_id` is the precomputed `compute_event_id()` value; no network
    call or `store.publish()` is required to compute it. Renamed to
    `.completed.json` (never rewritten) once step 3 (event publish) has
    completed, or replaced by an archived `InterruptedCallRecord` if
    startup reconciliation finds no per-attempt content file for it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_item_id: str
    attempt: int = Field(ge=0)
    event_id: str
    review_run_id: str
    extraction_run_id: str
    record_id: str
    field_id: str
    reserved_at: str

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> "ReservedAttemptMarker":
        expected_work_item_id = compute_expected_work_item_id(
            extraction_run_id=self.extraction_run_id, record_id=self.record_id, field_id=self.field_id,
        )
        if self.work_item_id != expected_work_item_id:
            raise ValueError(
                f"work_item_id {self.work_item_id!r} does not match expected {expected_work_item_id!r} derived "
                "from this marker's own extraction_run_id/record_id/field_id"
            )
        expected_event_id = compute_event_id(
            review_run_id=self.review_run_id, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
            work_item_id=self.work_item_id, attempt=self.attempt,
        )
        if self.event_id != expected_event_id:
            raise ValueError(
                f"event_id {self.event_id!r} does not match compute_event_id(...) = {expected_event_id!r} "
                "for this marker's own review_run_id/work_item_id/attempt"
            )
        return self


class QueueSnapshotResolutionEntry(BaseModel):
    """One already-resolved `(record_id, field_id)` as it stood at the
    moment a `QueueSnapshot` was captured -- the same fields
    `resolve_extraction_review_queue()`'s own `log_entry` writes into
    `extraction_resolutions.json` (`nodes/extraction.py`), carried here
    verbatim rather than re-derived, since this snapshot must remain
    reproducible even after the live resolution history has grown past
    the point it was captured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    field_id: str
    original_status: str
    original_field_result_hash: str
    outcome: Literal["extracted", "not_reported"]
    reviewer: str
    note: str
    resolved_at: str
    run_id: str
    manifest_hash: str
    extraction_schema_hash: str
    value: Optional[Any] = None
    evidence: Optional[List[Dict[str, str]]] = None
    attested: Optional[bool] = None


class QueueSnapshotEntry(BaseModel):
    """One still-queued `(record_id, field_id)` as it stood at the moment
    a `QueueSnapshot` was captured -- the same `record_id`/`field_id`/
    `status`/`field_result` shape `extraction_review_queue` entries
    already carry (`nodes/extraction.py`). A typed model rather than a
    bare `Dict[str, Any]` (round-review finding): a malformed entry must
    fail HERE, at construction/validation time, not later as an
    unguarded `KeyError` inside `render_review_packet()`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    field_id: str
    status: str
    field_result: FieldResult


class QueueSnapshot(BaseModel):
    """Immutable, JSON-safe baseline captured once at first review-packet
    render (sec. 3.7): the run's own queue (`extraction_review_queue`
    entries as they existed at that moment) plus the resolution history
    as of that moment, bound to the exact manifest/schema/checkpoint
    content it was captured from. `render_review_packet()` takes this,
    never the live, currently-mutable `extraction_review_queue` graph-
    state field, which shrinks as resolutions are made and therefore
    cannot serve as a reproducible input once any resolution has
    occurred. A flat list structure throughout -- never a dict keyed by
    the `(record_id, field_id)` tuple, which cannot round-trip through
    real `json.dumps`/`json.loads` (JSON object keys must be strings)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    manifest_hash: str
    extraction_schema_hash: str
    checkpoint_hashes: Dict[str, str]
    queue: List[QueueSnapshotEntry]
    resolution_history_as_of_render: List[QueueSnapshotResolutionEntry] = Field(default_factory=list)


def _build(model_cls: type, **fields: Any):
    attempt_record_id = compute_attempt_record_id(
        extraction_run_id=fields["extraction_run_id"],
        record_id=fields["record_id"],
        field_id=fields["field_id"],
        attempt=fields["attempt"],
    )
    base = dict(fields, attempt_record_id=attempt_record_id)
    if "suggestion_id" in model_cls.model_fields:
        base["suggestion_id"] = attempt_record_id
    self_hash = _compute_self_hash(model_cls, **base)
    return model_cls.model_validate(dict(base, self_hash=self_hash))


def build_validated_suggestion(**fields: Any) -> ValidatedSuggestion:
    return _build(ValidatedSuggestion, **fields)


def build_rejected_suggestion(**fields: Any) -> RejectedSuggestion:
    return _build(RejectedSuggestion, **fields)


def build_failed_call_record(**fields: Any) -> FailedCallRecord:
    return _build(FailedCallRecord, **fields)


def build_interrupted_call_record(**fields: Any) -> InterruptedCallRecord:
    return _build(InterruptedCallRecord, **fields)
