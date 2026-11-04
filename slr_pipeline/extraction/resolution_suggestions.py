"""Review-packet rendering for the extraction resolution-suggestion
workflow (amber-quilting-osprey.md sec. 3.2/3.3).

`render_review_packet()` is pure and offline: no LLM call, no filesystem
write, no mutation of its inputs. It builds, for every queued
`(record_id, field_id)` in an already-captured `QueueSnapshot`, the
stable evidence and candidate indexes (sec. 3.2) an LLM suggestion prompt
and a human reviewer both read from -- never raw excerpt text an LLM
could echo back unverified, always an ID minted here from a real
checkpoint item.

Every queue entry is cross-checked against the bundle's own,
independently verified `FieldResult` for that key before any of its
content is trusted -- the same tamper-detection discipline
`resolve_extraction_review_queue()` (`nodes/extraction.py`) already
applies to the live `extraction_review_queue`, applied here to the
immutable `QueueSnapshot` instead.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from slr_pipeline.ai_traceability import (
    AIEventLockError,
    AIUseEvent,
    AIUseEventStore,
    attempt_lock_path,
    build_ai_use_event,
    compute_event_id,
    kernel_lock,
    reserved_attempt,
    sanitize_call_error_text,
)
from slr_pipeline.extraction.checkpoint import VerifiedRunBundle
from slr_pipeline.extraction.hashing import canonical_json_bytes, sha256_hex
from slr_pipeline.extraction.resolution_ids import SourceCategory, candidate_id, evidence_id, safe_key
from slr_pipeline.extraction.resolution_schemas import (
    AIUseEventDraft,
    AttemptRecord,
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    FailedCallRecord,
    InterruptedCallRecord,
    Outcome,
    QueueSnapshot,
    QueueSnapshotEntry,
    QueueSnapshotResolutionEntry,
    RejectedSuggestion,
    ReservedAttemptMarker,
    ValidatedSuggestion,
    build_failed_call_record,
    build_interrupted_call_record,
    build_rejected_suggestion,
    build_validated_suggestion,
    compute_expected_work_item_id,
)
from slr_pipeline.extraction.results import EvidenceSpanModel, FieldResult, TypedValue
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, validate_value_for_field
from slr_pipeline.extraction.verified_bundle import bundle_review_fields, verify_existing_resolution

_ALLOWED_OUTCOMES = frozenset(get_args(Outcome))


class EvidenceItem(BaseModel):
    """One evidence span, keyed by its own `evidence_id()` (sec. 3.2) --
    the only shape a suggestion prompt or a human reviewer is ever shown
    excerpt/page/section text through."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    document_id: str
    field_id: str
    chunk_id: str
    excerpt: str
    page: Optional[str] = None
    section: str


class CandidateItem(BaseModel):
    """One candidate value, keyed by its own `candidate_id()` (sec. 3.2).
    `value_kind`/`reason` are populated only for `source_category=
    "invalid_value"` candidates (from `InvalidValueObservation`), where
    they carry real information; `None` otherwise."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    source_category: SourceCategory
    value: str
    typed_value: Optional[TypedValue] = None
    evidence_ids: Tuple[str, ...] = ()
    value_kind: Optional[str] = None
    reason: Optional[str] = None


class ReviewPacketEntry(BaseModel):
    """One queued `(record_id, field_id)`'s complete packet content: the
    field's own schema spec, its real, verified `FieldResult`, and the
    evidence/candidate indexes minted from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    field_id: str
    field_spec: ExtractionFieldSpec
    field_result: FieldResult
    evidence_index: Dict[str, EvidenceItem]
    candidates: Tuple[CandidateItem, ...]


class ReviewPacket(BaseModel):
    """One run's entire current review packet -- bound to the exact
    run/manifest/schema it was rendered from, so a packet can never be
    mistaken for describing a different run's queue. `resolution_history`
    is carried straight through from the `QueueSnapshot` that produced
    this packet (already immutable and already bound to the same
    run/manifest/schema, checked in `render_review_packet()`) -- the
    "existing resolution history, if any" a review packet must show."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    manifest_hash: str
    extraction_schema_hash: str
    entries: Tuple[ReviewPacketEntry, ...]
    resolution_history: Tuple[QueueSnapshotResolutionEntry, ...]


def review_packet_json_bytes(packet: ReviewPacket) -> bytes:
    """Canonical bytes used for both publication and re-derivation."""
    return canonical_json_bytes(packet.model_dump(mode="json"))


def render_review_packet_markdown(packet: ReviewPacket) -> str:
    """Human-readable, self-contained rendering of one immutable packet."""
    lines = [
        "# Extraction resolution review packet",
        "",
        f"- Extraction run: `{packet.run_id}`",
        f"- Manifest hash: `{packet.manifest_hash}`",
        f"- Extraction schema hash: `{packet.extraction_schema_hash}`",
        "",
    ]
    for entry in packet.entries:
        lines.extend([
            f"## `{entry.record_id}` / `{entry.field_id}`",
            "",
            f"**Question:** {entry.field_spec.question}",
            "",
            f"**Review status:** `{entry.field_result.status.value}`",
            "",
            "### Raw candidates",
            "",
        ])
        if not entry.candidates:
            lines.append("_No candidate values were recorded._")
        for candidate in entry.candidates:
            lines.extend([
                f"- `{candidate.candidate_id}` ({candidate.source_category}): {candidate.value}",
                f"  - Evidence IDs: {', '.join(f'`{eid}`' for eid in candidate.evidence_ids) or '_none_'}",
            ])
            if candidate.reason:
                lines.append(f"  - Reason: {candidate.reason}")
        lines.extend(["", "### Evidence", ""])
        if not entry.evidence_index:
            lines.append("_No evidence spans were recorded._")
        for evidence_id in sorted(entry.evidence_index):
            item = entry.evidence_index[evidence_id]
            locator = f"section={item.section}; page={item.page or 'unknown'}; chunk={item.chunk_id}"
            lines.extend([f"- `{item.evidence_id}` ({locator})", f"  - {item.excerpt}"])
        lines.append("")
    lines.extend(["## Resolution history at packet creation", ""])
    if not packet.resolution_history:
        lines.append("_No prior resolutions._")
    for item in packet.resolution_history:
        lines.append(
            f"- `{item.record_id}` / `{item.field_id}`: `{item.outcome}`; "
            f"reviewer={item.reviewer}; resolved_at={item.resolved_at}"
        )
    return "\n".join(lines).rstrip() + "\n"


def review_packet_markdown_bytes(packet: ReviewPacket) -> bytes:
    return render_review_packet_markdown(packet).encode("utf-8")


def _evidence_index_for_field(*, record_id: str, field_id: str, field_result: FieldResult) -> Dict[str, EvidenceItem]:
    index: Dict[str, EvidenceItem] = {}
    sources: Tuple[Tuple[EvidenceSpanModel, ...], ...] = (
        tuple(span for value in field_result.values for span in value.evidence),
        tuple(span for value in field_result.provisional_values for span in value.evidence),
        tuple(span for invalid in field_result.invalid_values for span in invalid.evidence),
        field_result.review_evidence,
    )
    for spans in sources:
        for span in spans:
            eid = evidence_id(record_id, field_id, span.chunk_id, span.excerpt, span.page, span.section)
            if eid not in index:
                index[eid] = EvidenceItem(
                    evidence_id=eid, document_id=record_id, field_id=field_id,
                    chunk_id=span.chunk_id, excerpt=span.excerpt, page=span.page, section=span.section,
                )
    return index


def _evidence_ids_for_spans(*, record_id: str, field_id: str, spans: Tuple[EvidenceSpanModel, ...]) -> Tuple[str, ...]:
    return tuple(sorted(evidence_id(record_id, field_id, s.chunk_id, s.excerpt, s.page, s.section) for s in spans))


def _candidates_for_field(*, record_id: str, field_id: str, field_result: FieldResult) -> Tuple[CandidateItem, ...]:
    candidates: list[CandidateItem] = []
    for value in field_result.values:
        eids = _evidence_ids_for_spans(record_id=record_id, field_id=field_id, spans=value.evidence)
        candidates.append(
            CandidateItem(
                candidate_id=candidate_id(value.value, value.typed_value, "value", eids),
                source_category="value", value=value.value, typed_value=value.typed_value, evidence_ids=eids,
            )
        )
    for value in field_result.provisional_values:
        eids = _evidence_ids_for_spans(record_id=record_id, field_id=field_id, spans=value.evidence)
        candidates.append(
            CandidateItem(
                candidate_id=candidate_id(value.value, value.typed_value, "provisional_value", eids),
                source_category="provisional_value", value=value.value, typed_value=value.typed_value,
                evidence_ids=eids,
            )
        )
    for invalid in field_result.invalid_values:
        eids = _evidence_ids_for_spans(record_id=record_id, field_id=field_id, spans=invalid.evidence)
        candidates.append(
            CandidateItem(
                candidate_id=candidate_id(invalid.raw_value, None, "invalid_value", eids),
                source_category="invalid_value", value=invalid.raw_value, typed_value=None, evidence_ids=eids,
                value_kind=invalid.value_kind, reason=invalid.reason,
            )
        )
    return tuple(candidates)


def render_review_packet(bundle: VerifiedRunBundle, schema: ExtractionSchema, queue_snapshot: QueueSnapshot) -> ReviewPacket:
    """Pure, offline, no LLM call. `queue_snapshot` must be bound to this
    exact `bundle`/`schema` (run_id, manifest_hash, extraction_schema_hash,
    checkpoint_hashes all cross-checked) -- a snapshot captured against a
    different run/manifest/schema/checkpoint set is refused outright,
    never silently rendered against the wrong bundle.

    `bundle` is re-verified here, at this public boundary, rather than
    trusted at face value -- a caller-constructed or previously-mutated
    `VerifiedRunBundle` must not be able to reach the rest of this
    function without its own internal consistency (manifest hash,
    checkpoint hashes, per-checkpoint content) being re-checked first."""
    bundle.verify()
    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    if schema_hash != bundle.manifest.extraction_schema_hash:
        raise ValueError(
            f"schema hash {schema_hash!r} does not match the verified bundle's own manifest "
            f"extraction_schema_hash {bundle.manifest.extraction_schema_hash!r}"
        )
    if queue_snapshot.run_id != bundle.run_id:
        raise ValueError(f"queue_snapshot.run_id {queue_snapshot.run_id!r} does not match bundle.run_id {bundle.run_id!r}")
    if queue_snapshot.manifest_hash != bundle.manifest.manifest_hash:
        raise ValueError(
            f"queue_snapshot.manifest_hash {queue_snapshot.manifest_hash!r} does not match the verified "
            f"bundle's own manifest_hash {bundle.manifest.manifest_hash!r}"
        )
    if queue_snapshot.extraction_schema_hash != schema_hash:
        raise ValueError(
            f"queue_snapshot.extraction_schema_hash {queue_snapshot.extraction_schema_hash!r} does not match "
            f"the schema's own hash {schema_hash!r}"
        )
    if dict(queue_snapshot.checkpoint_hashes) != dict(bundle.manifest.checkpoint_hashes):
        raise ValueError(
            "queue_snapshot.checkpoint_hashes does not match the verified bundle's own manifest checkpoint_hashes"
        )

    field_map = schema.field_map()
    verified_fields = bundle_review_fields(bundle.results)
    verified_keys = set(verified_fields.keys())

    # Every resolution_history entry is cross-checked against the SAME
    # verified, hash-recomputed source of truth `verify_existing_resolution()`
    # (`extraction/verified_bundle.py`, relocated 2026-09-17 from
    # `nodes/extraction.py._verify_existing_resolution`) already checks a
    # fresh resolution against -- not merely bound to the right
    # run/manifest/schema, but proven to correspond to a real
    # review-required field, to describe that field's own original
    # status/hash accurately, AND (the part a hand-rolled partial check here
    # previously missed) to satisfy the outcome-conditional business rules:
    # a "not_reported" entry must carry no value/evidence/attestation; an
    # "extracted" entry must carry a schema-valid value, non-empty
    # structured evidence, and attested=True.
    history_keys: set = set()
    for resolved in queue_snapshot.resolution_history_as_of_render:
        key = (resolved.record_id, resolved.field_id)
        if key in history_keys:
            raise ValueError(f"resolution_history_as_of_render contains duplicate entry for {key!r}")
        history_keys.add(key)

        verify_existing_resolution(
            resolved.model_dump(mode="json"), run_id=bundle.run_id, manifest_hash=bundle.manifest.manifest_hash,
            schema_hash=schema_hash, review_fields=verified_fields, field_map=field_map,
        )

    queue_entries: list[QueueSnapshotEntry] = list(queue_snapshot.queue)
    queue_keys_list = [(e.record_id, e.field_id) for e in queue_entries]
    queue_keys = set(queue_keys_list)
    if len(queue_keys) != len(queue_keys_list):
        raise ValueError("queue_snapshot.queue contains a duplicate (record_id, field_id) entry")

    overlap = queue_keys & history_keys
    if overlap:
        raise ValueError(f"queue_snapshot.queue and resolution_history_as_of_render overlap on {sorted(overlap)!r} -- a field cannot be both still-queued and already resolved")

    covered_keys = queue_keys | history_keys
    if covered_keys != verified_keys:
        missing = verified_keys - covered_keys
        extra = covered_keys - verified_keys
        raise ValueError(
            "queue_snapshot does not exactly cover the verified bundle's own review-required fields -- "
            f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )

    entries: list[ReviewPacketEntry] = []
    for raw_entry in queue_entries:
        record_id, field_id = raw_entry.record_id, raw_entry.field_id
        key = (record_id, field_id)

        verified_field = verified_fields[key]
        if raw_entry.field_result != verified_field or raw_entry.status != verified_field.status.value:
            raise ValueError(f"queue_snapshot entry for {key!r} does not match the verified bundle's own field result/status -- refusing to render a packet from possibly tampered queue-snapshot content")

        field_spec = field_map.get(field_id)
        if field_spec is None:
            raise ValueError(f"field_id {field_id!r} is not present in the run's own extraction schema")

        evidence_index = _evidence_index_for_field(record_id=record_id, field_id=field_id, field_result=verified_field)
        candidates = _candidates_for_field(record_id=record_id, field_id=field_id, field_result=verified_field)
        entries.append(
            ReviewPacketEntry(
                record_id=record_id, field_id=field_id, field_spec=field_spec, field_result=verified_field,
                evidence_index=evidence_index, candidates=candidates,
            )
        )

    return ReviewPacket(
        run_id=bundle.run_id, manifest_hash=bundle.manifest.manifest_hash, extraction_schema_hash=schema_hash,
        entries=tuple(entries), resolution_history=tuple(queue_snapshot.resolution_history_as_of_render),
    )


class RawSuggestion(BaseModel):
    """Structural shape of one LLM suggestion response (amber-quilting-osprey.md
    sec. 4.4) -- `parse_suggestion_response()`'s only output. `extra="forbid"`
    makes schema-shape validation (sec. 4.5 point 1) a property of parsing
    itself: an unexpected field fails here, before any content-level rule in
    `validate_suggestion()` runs. `outcome` is left as a bare `str` here
    (rather than the `Outcome` literal) so an out-of-set value is rejected by
    `validate_suggestion()`'s own explicit check, with a message naming the
    offending value, rather than by an opaque pydantic literal-mismatch error
    at parse time."""

    model_config = ConfigDict(extra="forbid")

    outcome: str
    value: Optional[str] = None
    evidence_ids: Tuple[str, ...] = ()
    candidate_ids_merged: Tuple[str, ...] = ()
    candidate_ids_retained: Tuple[str, ...] = ()
    candidate_ids_rejected: Tuple[str, ...] = ()
    rationale: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    uncertainty: Optional[str] = None
    requires_human_review: bool = True


class SuggestionParseError(ValueError):
    """`raw` could not be parsed into a structurally valid `RawSuggestion` --
    a lower-level failure than `validate_suggestion()`'s deterministic
    domain-rule checks. The provider call itself still succeeded (it
    returned real, if unparseable, content), so the caller (`run_suggestion_engine`)
    persists this as a `RejectedSuggestion` with `call_status="success"`,
    never a `FailedCallRecord` -- `raw_response` carries the best-effort
    decoded content (or the raw text itself, if not even valid JSON) rather
    than a validated `RawSuggestion.model_dump()`, since `raw_response: Any`
    does not require one (amber-quilting-osprey.md sec. 3.4)."""


def build_suggestion_prompt(packet_entry: ReviewPacketEntry) -> str:
    """The LLM contract boundary (sec. 4.4): presents only evidence IDs +
    excerpts and candidate IDs + values already minted by `render_review_packet()`
    (sec. 3.2) -- the model is never asked to invent a locator, an evidence
    ID, or a candidate ID, only to select among the ones listed here."""
    field_spec = packet_entry.field_spec
    field_result = packet_entry.field_result

    field_block = (
        f'field_id: "{field_spec.field_id}"\n'
        f"question: {field_spec.question}\n"
        f"description: {field_spec.description}\n"
        f"value_kind: {field_spec.value_kind.value}\n"
        f"cardinality: {field_spec.cardinality}"
    )
    if field_spec.allowed_values:
        values_text = "\n".join(f'  * "{av.value}": {av.definition}' for av in field_spec.allowed_values)
        field_block += f"\nallowed_values:\n{values_text}"

    trigger_block = (
        f"status: {field_result.status.value}\n"
        f"requires_human_review: {field_result.requires_human_review}"
    )
    if field_result.rationale:
        trigger_block += f"\nmachine_rationale: {field_result.rationale}"

    evidence_blocks = [
        f'[evidence_id: {item.evidence_id} | section: {item.section} | page: {item.page or "unknown"}]\n{item.excerpt}'
        for item in packet_entry.evidence_index.values()
    ]
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "(no evidence spans recorded)"

    candidate_blocks: List[str] = []
    for candidate in packet_entry.candidates:
        line = (
            f'- candidate_id: "{candidate.candidate_id}"\n'
            f"  source_category: {candidate.source_category}\n"
            f"  value: {candidate.value}\n"
            f"  evidence_ids: {list(candidate.evidence_ids)}"
        )
        if candidate.reason:
            line += f"\n  reason: {candidate.reason}"
        candidate_blocks.append(line)
    candidates_text = "\n".join(candidate_blocks) if candidate_blocks else "(no candidate values recorded)"

    return f"""You are adjudicating ONE field of ONE document in a systematic literature review extraction.
Use ONLY the evidence and candidate values listed below -- never outside knowledge, and never a locator,
evidence_id, or candidate_id that is not explicitly listed here.

Document: {packet_entry.record_id}

Field:
{field_block}

Review trigger (why this field needs adjudication):
{trigger_block}

Candidate values already observed by the machine extractor (each with a stable candidate_id):
{candidates_text}

Evidence spans available (each with a stable evidence_id -- cite these IDs only, never invent a new locator):
{evidence_text}

Synthesize overlapping/duplicate formulations of the same fact into one value. If this field's cardinality
is "multiple", preserve genuinely distinct values rather than merging them into one. Never infer a category
or value that is not directly supported by the evidence above.

Respond with exactly one JSON object of this shape:
{{
  "outcome": "extracted" | "not_reported" | "unclear" | "conflicting" | "extraction_failed",
  "value": "<schema-valid value, or null>",
  "evidence_ids": ["<evidence_id from the list above>", ...],
  "candidate_ids_merged": ["<candidate_id>", ...],
  "candidate_ids_retained": ["<candidate_id>", ...],
  "candidate_ids_rejected": ["<candidate_id>", ...],
  "rationale": "<one short sentence>",
  "confidence": "high" | "medium" | "low",
  "uncertainty": "<remaining uncertainty, or null>",
  "requires_human_review": true | false
}}"""


def parse_suggestion_response(raw: Any) -> RawSuggestion:
    """Parses the raw LLM output (a JSON string, bytes, or an already-decoded
    object) into a structurally valid `RawSuggestion` -- sec. 4.5 point 1's
    "validate the response schema" is satisfied by this parse itself
    (`RawSuggestion`'s own `extra="forbid"` and field types). Raises
    `SuggestionParseError` on anything that is not valid JSON or does not
    match the expected shape; `validate_suggestion()` is never reached with
    a malformed response."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SuggestionParseError(f"response is not valid JSON: {exc}") from exc
    elif isinstance(raw, dict):
        parsed = raw
    else:
        raise SuggestionParseError(f"response must be a JSON string or object, got {type(raw).__name__}")
    try:
        return RawSuggestion.model_validate(parsed)
    except ValidationError as exc:
        raise SuggestionParseError(f"response does not match the expected suggestion schema: {exc}") from exc


def validate_suggestion(
    raw: RawSuggestion,
    *,
    field_spec: ExtractionFieldSpec,
    field_result: FieldResult,
    evidence_index: Dict[str, EvidenceItem],
    extraction_run_id: str,
    record_id: str,
    field_id: str,
    attempt: int,
    ai_use_event_id: str,
    ai_use_event_draft: AIUseEventDraft,
    created_at: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    estimated_cost_usd: Optional[float] = None,
) -> "ValidatedSuggestion | RejectedSuggestion":
    """Deterministic post-validation (sec. 4.5 points 2-6). Point 1 (schema
    validation) is already satisfied by `parse_suggestion_response()`; point 7
    (attempt numbering) is the caller's responsibility -- `attempt`,
    `ai_use_event_id`, and `ai_use_event_draft` must already come from sec.
    3.4's crash-safe reservation, never invented here.

    The candidate-ID index (sec. 3.2, covering `values`/`provisional_values`/
    `invalid_values` together) is recomputed here via the same
    `_candidates_for_field()` helper `render_review_packet()` uses, rather
    than taken as a separate parameter -- a pure function of `record_id`/
    `field_id`/`field_result`, so it is always exactly the same index a
    packet built from this same verified `field_result` would carry."""
    candidates = _candidates_for_field(record_id=record_id, field_id=field_id, field_result=field_result)
    candidate_index = {c.candidate_id: c for c in candidates}

    reasons: list[str] = []
    typed_value: Optional[TypedValue] = None
    outcome = raw.outcome

    if outcome not in _ALLOWED_OUTCOMES:
        reasons.append(f"outcome {outcome!r} is not one of {sorted(_ALLOWED_OUTCOMES)!r}")

    bad_evidence_ids = [eid for eid in raw.evidence_ids if eid not in evidence_index]
    if bad_evidence_ids:
        reasons.append(f"evidence_ids not present in this field's evidence index: {bad_evidence_ids!r}")

    for label, ids in (
        ("candidate_ids_merged", raw.candidate_ids_merged),
        ("candidate_ids_retained", raw.candidate_ids_retained),
        ("candidate_ids_rejected", raw.candidate_ids_rejected),
    ):
        bad_ids = [cid for cid in ids if cid not in candidate_index]
        if bad_ids:
            reasons.append(f"{label} references candidate_ids not present in this field's candidate index: {bad_ids!r}")

    if outcome == "extracted":
        if raw.value is None or not raw.value.strip():
            reasons.append("outcome='extracted' requires a non-blank value")
        else:
            is_valid, typed_value = validate_value_for_field(field_spec, raw.value)
            if not is_valid:
                reasons.append(
                    f"value {raw.value!r} is not valid for value_kind={field_spec.value_kind.value!r}"
                )
        if not raw.evidence_ids:
            reasons.append("outcome='extracted' requires at least one evidence_id")
    else:
        if raw.value is not None:
            reasons.append(f"outcome={outcome!r} must not carry a value")
        if raw.evidence_ids:
            reasons.append(f"outcome={outcome!r} must not carry evidence_ids")

    identity_fields = dict(
        extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, attempt=attempt,
        original_field_result_hash=sha256_hex(field_result.model_dump(mode="json")),
        ai_use_event_id=ai_use_event_id, ai_use_event_draft=ai_use_event_draft,
    )
    usage_fields = dict(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=total_tokens, estimated_cost_usd=estimated_cost_usd, created_at=created_at,
    )

    if reasons:
        return build_rejected_suggestion(
            **identity_fields, raw_response=raw.model_dump(mode="json"),
            rejection_reasons=tuple(reasons), **usage_fields,
        )

    return build_validated_suggestion(
        **identity_fields,
        outcome=outcome,
        value=raw.value if outcome == "extracted" else None,
        typed_value=typed_value if outcome == "extracted" else None,
        evidence_ids=raw.evidence_ids if outcome == "extracted" else (),
        candidate_ids_merged=raw.candidate_ids_merged,
        candidate_ids_retained=raw.candidate_ids_retained,
        candidate_ids_rejected=raw.candidate_ids_rejected,
        rationale=raw.rationale,
        confidence=raw.confidence,
        uncertainty=raw.uncertainty,
        requires_human_review=raw.requires_human_review,
        **usage_fields,
    )


# ---------------------------------------------------------------------------
# run_suggestion_engine() (sec. 3.3/3.4/3.8) -- crash-safe three-step state
# machine around the real LLM call, plus startup reconciliation of orphaned
# `.reserved.json` markers into `.interrupted_unknown.json` archives.
#
# On-disk layout under `suggestions_dir`, one file per attempt, named from
# the content-derived `safe_key()` (never the raw record_id/field_id):
#   <safe_key>__attempt-<n>.reserved.json            -- step 1 (pre-call)
#   <safe_key>__attempt-<n>.json                     -- step 2 (immutable content)
#   <safe_key>__attempt-<n>.completed.json           -- step 3 (post-publish rename)
#   <safe_key>__attempt-<n>.interrupted_unknown.json -- reconciliation archive
# ---------------------------------------------------------------------------

_RESERVED_RE = re.compile(r"^([0-9a-f]{24})__attempt-(\d+)\.reserved\.json$")
_COMPLETED_RE = re.compile(r"^([0-9a-f]{24})__attempt-(\d+)\.completed\.json$")
_INTERRUPTED_RE = re.compile(r"^([0-9a-f]{24})__attempt-(\d+)\.interrupted_unknown\.json$")
_CONTENT_RE = re.compile(r"^([0-9a-f]{24})__attempt-(\d+)\.json$")

_ATTEMPT_RECORD_MODEL_BY_KIND = {
    "validated_suggestion": ValidatedSuggestion,
    "rejected_suggestion": RejectedSuggestion,
    "failed_call": FailedCallRecord,
}


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_is_cap_exceeded(exc: BaseException) -> bool:
    """Detects a `CostCapProxy`-refused call (HTTP 402, `cost_cap/proxy.py`)
    surfaced through the OpenAI SDK client -- an injectable predicate
    (never hardcoded to a specific SDK exception class) so offline tests
    can simulate the cap-exceeded path with a plain fake exception that
    merely carries a matching `status_code` attribute."""
    return getattr(exc, "status_code", None) == 402


def _reserved_path(suggestions_dir: Path, key: str, attempt: int) -> Path:
    return suggestions_dir / f"{key}__attempt-{attempt}.reserved.json"


def _content_path(suggestions_dir: Path, key: str, attempt: int) -> Path:
    return suggestions_dir / f"{key}__attempt-{attempt}.json"


def _completed_path(suggestions_dir: Path, key: str, attempt: int) -> Path:
    return suggestions_dir / f"{key}__attempt-{attempt}.completed.json"


def _interrupted_path(suggestions_dir: Path, key: str, attempt: int) -> Path:
    return suggestions_dir / f"{key}__attempt-{attempt}.interrupted_unknown.json"


def _any_marker_exists(suggestions_dir: Path, key: str, attempt: int) -> bool:
    """The `extra_occupied` predicate passed to `reserved_attempt()`
    (sec. 3.4, round-6 fix): an attempt slot already claimed by a local
    marker -- reserved, completed, archived-interrupted, OR a bare content
    file with no marker yet renamed over it (defense in depth: this state
    should never arise without an accompanying `.reserved.json`, but an
    attempt slot with ANY on-disk trace must never be handed out again
    regardless) -- must never be handed out again, even though no
    `AIUseEvent` may exist there."""
    return (
        _reserved_path(suggestions_dir, key, attempt).exists()
        or _content_path(suggestions_dir, key, attempt).exists()
        or _completed_path(suggestions_dir, key, attempt).exists()
        or _interrupted_path(suggestions_dir, key, attempt).exists()
    )


def _win32_flush_directory(path: Path) -> bool:
    """Opens `path` as a directory HANDLE and calls `FlushFileBuffers()`
    on it -- the documented technique for flushing a directory entry
    (not just file content) to stable storage on NTFS (the same
    approach SQLite's own Windows VFS uses) -- returning the raw Win32
    result as a Python `bool`. Kept separate from `_fsync_dir_windows()`
    so tests can assert on the actual API result, not merely on the
    absence of a raised exception: an earlier version of this function
    opened the handle with `GENERIC_READ` alone, which MSDN documents
    (https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers)
    as insufficient for `FlushFileBuffers()` -- it returned `FALSE`
    (`ERROR_ACCESS_DENIED`, error 5) on every real call, silently
    discarded by the caller, so the directory entry was never actually
    flushed despite reporting success. Fixed to `GENERIC_WRITE`, which
    `FlushFileBuffers()` requires.

    Raises `OSError` if the directory handle itself cannot be opened
    (fail closed, same as this function's own `FlushFileBuffers()`
    result is never silently discarded by its caller)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    generic_write = 0x40000000
    share_all = 0x00000001 | 0x00000002 | 0x00000004  # READ | WRITE | DELETE
    open_existing = 3
    flag_backup_semantics = 0x02000000
    invalid_handle_value = wintypes.HANDLE(-1).value

    ctypes.set_last_error(0)
    handle = kernel32.CreateFileW(
        str(path), generic_write, share_all, None, open_existing, flag_backup_semantics, None,
    )
    if handle == invalid_handle_value:
        err = ctypes.get_last_error()
        raise OSError(f"CreateFileW({str(path)!r}) failed opening directory for fsync, Windows error {err}")
    try:
        ctypes.set_last_error(0)
        result = kernel32.FlushFileBuffers(handle)
        return bool(result)
    finally:
        kernel32.CloseHandle(handle)


def _fsync_dir_windows(path: Path) -> None:
    """Fails closed: raises `OSError` if `_win32_flush_directory()`
    reports failure, since silently discarding that result (as a prior
    version of this function did) defeats the entire point of this
    durability guarantee -- an unflushed directory entry that
    `_atomic_write_model()`/reconciliation callers believe is durable
    is exactly the crash-recovery gap sec. 3.4/3.8 exists to close."""
    if not _win32_flush_directory(path):
        raise OSError(f"FlushFileBuffers({str(path)!r}) reported failure during directory fsync")


def _fsync_dir(path: Path) -> None:
    """Directory-entry fsync after an atomic create/rename/unlink in
    `suggestions_dir`, so the directory entry itself (not just file
    content) survives a crash. POSIX uses `os.O_DIRECTORY` +
    `os.fsync()`; Windows (the actual deployment target for this
    pipeline) uses `_fsync_dir_windows()`. Both fail closed -- a failure
    to actually flush the directory entry raises `OSError` rather than
    being silently ignored, on either platform."""
    if os.name == "posix":
        fd = os.open(str(path), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    elif sys.platform == "win32":
        _fsync_dir_windows(path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Same write-tmp/fsync/os.replace convention as
    `nodes/reporting.py::_write_artifact_and_verify()`, plus this module's
    fail-closed directory-entry fsync (`_fsync_dir()`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync_dir(path.parent)


def _atomic_write_model(path: Path, model: BaseModel) -> None:
    """Atomically persist a typed attempt or marker model."""
    atomic_write_bytes(path, model.model_dump_json(indent=2).encode("utf-8"))


def _verify_key_and_attempt(record: AttemptRecord, *, path: Path, key: str, attempt: int) -> None:
    """A record's own self-hash/identity validators (`resolution_schemas.py`)
    already prove it is internally self-consistent, but they know nothing
    about the FILENAME it was actually found at -- a swapped, renamed, or
    otherwise misplaced file would pass every internal check while
    describing a different work item or attempt than its path claims to.
    Cross-checked here, at every load site, rather than trusted implicitly."""
    expected_key = safe_key(record.extraction_run_id, record.record_id, record.field_id)
    if expected_key != key:
        raise ValueError(
            f"{path}: content's own extraction_run_id/record_id/field_id derive key {expected_key!r}, "
            f"which does not match this file's own filename key {key!r}"
        )
    if record.attempt != attempt:
        raise ValueError(
            f"{path}: content's own attempt {record.attempt} does not match this file's own filename "
            f"attempt {attempt}"
        )


def _verify_marker_identity(marker: ReservedAttemptMarker, *, path: Path, key: str, attempt: int) -> None:
    """Mirrors `_verify_key_and_attempt()` for a `.reserved.json` marker:
    the FILENAME a reservation is found at is never trusted as the
    source of truth for which work item it reserves -- a renamed marker
    file would otherwise let reconciliation archive or complete the
    wrong attempt slot under a name that doesn't match its own content."""
    expected_key = safe_key(marker.extraction_run_id, marker.record_id, marker.field_id)
    if expected_key != key:
        raise ValueError(
            f"{path}: marker's own extraction_run_id/record_id/field_id derive key {expected_key!r}, "
            f"which does not match this file's own filename key {key!r}"
        )
    if marker.attempt != attempt:
        raise ValueError(
            f"{path}: marker's own attempt {marker.attempt} does not match this file's own filename "
            f"attempt {attempt}"
        )
    expected_work_item_id = compute_expected_work_item_id(
        extraction_run_id=marker.extraction_run_id, record_id=marker.record_id, field_id=marker.field_id,
    )
    if expected_work_item_id != marker.work_item_id:
        raise ValueError(
            f"{path}: marker's own extraction_run_id/record_id/field_id derive work_item_id "
            f"{expected_work_item_id!r}, which does not match marker.work_item_id {marker.work_item_id!r}"
        )


def _load_attempt_record(path: Path, *, key: str, attempt: int) -> AttemptRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_cls = _ATTEMPT_RECORD_MODEL_BY_KIND.get(data.get("kind"))
    if model_cls is None:
        raise ValueError(f"{path}: unknown or missing attempt record kind {data.get('kind')!r}")
    record = model_cls.model_validate(data)
    _verify_key_and_attempt(record, path=path, key=key, attempt=attempt)
    return record


def _load_interrupted_record(path: Path, *, key: str, attempt: int) -> InterruptedCallRecord:
    record = InterruptedCallRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
    _verify_key_and_attempt(record, path=path, key=key, attempt=attempt)
    return record


def _scan_attempts_for_key(suggestions_dir: Path, key: str) -> Dict[int, AttemptRecord]:
    """Every terminal attempt this work item has -- from real content
    files (`ValidatedSuggestion`/`RejectedSuggestion`/`FailedCallRecord`)
    and archived-interrupted files alike. Bare `.reserved.json`/
    `.completed.json` marker files are bookkeeping only, never a content
    source, and are deliberately not read here."""
    records: Dict[int, AttemptRecord] = {}
    if not suggestions_dir.is_dir():
        return records
    for path in suggestions_dir.iterdir():
        m = _CONTENT_RE.match(path.name)
        if m and m.group(1) == key:
            attempt = int(m.group(2))
            records[attempt] = _load_attempt_record(path, key=key, attempt=attempt)
            continue
        m = _INTERRUPTED_RE.match(path.name)
        if m and m.group(1) == key:
            attempt = int(m.group(2))
            records[attempt] = _load_interrupted_record(path, key=key, attempt=attempt)
    return records


def load_attempt_records(suggestions_dir: Path) -> Tuple[Tuple[Path, AttemptRecord], ...]:
    """Load every immutable terminal attempt file in deterministic order.

    Marker files are deliberately excluded. Filename/content identity and
    each model's self hash are rechecked by the existing loaders.
    """
    loaded: List[Tuple[Path, AttemptRecord]] = []
    identities: set[Tuple[str, int]] = set()
    if not suggestions_dir.is_dir():
        return ()
    for path in sorted(suggestions_dir.iterdir(), key=lambda p: p.name):
        content_match = _CONTENT_RE.match(path.name)
        interrupted_match = _INTERRUPTED_RE.match(path.name)
        if content_match:
            key, attempt = content_match.group(1), int(content_match.group(2))
            record: AttemptRecord = _load_attempt_record(path, key=key, attempt=attempt)
        elif interrupted_match:
            key, attempt = interrupted_match.group(1), int(interrupted_match.group(2))
            record = _load_interrupted_record(path, key=key, attempt=attempt)
        else:
            continue
        identity = (record.attempt_record_id, record.attempt)
        if identity in identities:
            raise ValueError(f"duplicate terminal attempt identity {identity!r} in {suggestions_dir}")
        identities.add(identity)
        loaded.append((path, record))
    return tuple(loaded)


def eligible_suggestion_entries(
    packet: ReviewPacket, *, suggestions_dir: Path,
    retry_work_item_ids: Sequence[str] = (),
) -> Tuple[ReviewPacketEntry, ...]:
    """Return exactly the packet entries a dispatch pass would call now."""
    retry_ids = set(retry_work_item_ids)
    known_ids = {
        compute_expected_work_item_id(
            extraction_run_id=packet.run_id,
            record_id=entry.record_id,
            field_id=entry.field_id,
        )
        for entry in packet.entries
    }
    unknown_retry_ids = retry_ids - known_ids
    if unknown_retry_ids:
        raise ValueError(
            f"retry_work_item_ids are absent from this review packet: {sorted(unknown_retry_ids)!r}"
        )
    eligible: List[ReviewPacketEntry] = []
    for entry in packet.entries:
        work_item_id = compute_expected_work_item_id(
            extraction_run_id=packet.run_id,
            record_id=entry.record_id,
            field_id=entry.field_id,
        )
        records = _scan_attempts_for_key(
            Path(suggestions_dir),
            safe_key(packet.run_id, entry.record_id, entry.field_id),
        )
        if _find_validated(records) is not None:
            continue
        if records and work_item_id not in retry_ids:
            continue
        eligible.append(entry)
    return tuple(eligible)


def suggestions_index_bytes(
    attempts: Sequence[Tuple[Path, AttemptRecord]], *, extraction_run_id: str,
) -> bytes:
    """Rebuild the generated suggestions index from verified attempts."""
    rows: List[Dict[str, Any]] = []
    for path, record in attempts:
        if record.extraction_run_id != extraction_run_id:
            raise ValueError(
                f"{path}: extraction_run_id {record.extraction_run_id!r} does not match "
                f"index run {extraction_run_id!r}"
            )
        row: Dict[str, Any] = {
            "attempt_record_id": record.attempt_record_id,
            "kind": record.kind,
            "record_id": record.record_id,
            "field_id": record.field_id,
            "safe_key": safe_key(record.extraction_run_id, record.record_id, record.field_id),
            "attempt": record.attempt,
            "path": f"extraction_resolution_suggestions/{path.name}",
            "self_hash": record.self_hash,
            "ai_use_event_id": record.ai_use_event_id,
            "outcome": getattr(record, "outcome", None),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["safe_key"], row["attempt"], row["kind"]))
    return canonical_json_bytes({
        "index_schema_version": 1,
        "extraction_run_id": extraction_run_id,
        "attempts": rows,
    })


def count_interrupted_suggestions(suggestions_dir: Path) -> int:
    """The real, on-disk count `resolution_suggestion_interrupted`
    (`RunModeDocument`, sec. 3.4 round-5 addition) is derived from --
    every `.interrupted_unknown.json` archive is loaded and its own
    identity/self-hash re-verified, never merely counted by filename
    match alone."""
    if not suggestions_dir.is_dir():
        return 0
    count = 0
    for path in suggestions_dir.iterdir():
        m = _INTERRUPTED_RE.match(path.name)
        if m:
            _load_interrupted_record(path, key=m.group(1), attempt=int(m.group(2)))
            count += 1
    return count


def _publish_event_for_record(event_store: AIUseEventStore, record: AttemptRecord, *, stage: str) -> AIUseEvent:
    """Step 3: construct+publish the real `AIUseEvent` from a
    `ValidatedSuggestion`/`RejectedSuggestion`/`FailedCallRecord`'s own
    embedded `ai_use_event_draft` -- every immutable constructor input
    was already captured at step 2, before any of it could go stale.
    `AIUseEventStore.publish()` is itself idempotent on a benign retry
    (same `event_id`, same `semantic_hash`), so this is safe to call
    again during reconciliation even if the event was already published."""
    draft = record.ai_use_event_draft
    if record.kind == "failed_call":
        call_status: Literal["success", "failure", "heuristic_fallback"] = "failure"
        failure_detail: Optional[str] = record.error_message
    else:
        call_status = "success"
        failure_detail = None
    key = safe_key(record.extraction_run_id, record.record_id, record.field_id)
    event = build_ai_use_event(
        review_run_id=draft.review_run_id, stage=stage, work_item_id=draft.work_item_id, attempt=draft.attempt,
        tool_kind="llm", model=draft.model, provider=draft.provider, component_versions=draft.component_versions,
        access_info=draft.access_info, task_purpose=draft.task_purpose, model_config_id=draft.model_config_id,
        input_artifact_refs=list(draft.input_artifact_refs),
        # Content-hash ref alone requires already having the file in hand to
        # check it against; the leading locator-style ref lets later bundle
        # cross-validation (sec. 3.7) map this event back to its own exact
        # attempt file on disk deterministically, by name, not just by hash.
        output_artifact_refs=[f"extraction_resolution_attempt:{key}:{record.attempt}", f"sha256:{record.self_hash}"],
        call_status=call_status, failure_detail=failure_detail,
        prompt_template_version=draft.prompt_template_version, prompt_template_text=draft.prompt_template_text,
        prompt_parameters=dict(draft.prompt_parameters), rendered_prompt_hash=draft.rendered_prompt_hash,
        timestamp=draft.event_timestamp,
    )
    if event.event_id != record.ai_use_event_id:
        raise ValueError(
            f"constructed AIUseEvent.event_id {event.event_id!r} does not match this record's own "
            f"ai_use_event_id {record.ai_use_event_id!r} -- refusing to publish a mismatched event"
        )
    return event_store.publish(event)


@dataclass(frozen=True)
class ReconciliationResult:
    archived_interrupted: int = 0
    completed: int = 0


def reconcile_suggestion_attempts(
    suggestions_dir: Path, event_store: AIUseEventStore, *,
    stage: str = EXTRACTION_RESOLUTION_SUGGESTION_STAGE, now: Callable[[], str] = _utc_now_iso,
    lock_timeout: float = 5.0,
) -> ReconciliationResult:
    """Sec. 3.4/3.8's startup reconciliation of every orphaned
    `.reserved.json` marker in `suggestions_dir` into one of the three
    well-defined recovery outcomes, run once before any new dispatch:
    - no per-attempt content file at all -> true process termination;
      archive to `.interrupted_unknown.json` (this attempt slot is never
      reused) and delete the now-superseded `.reserved.json`.
    - a per-attempt content file exists (event published or not) ->
      the call's outcome is already fully, immutably known; complete the
      publish (idempotent if already published) and rename the marker to
      `.completed.json` -- never a new network call, never new spend.

    A reservation is only "orphaned" if the process that made it is truly
    gone -- a `.reserved.json` file on disk alone does not prove that.
    Each candidate is therefore re-examined under `reserved_attempt()`'s
    own per-`(stage, work_item_id)` kernel lock (via `attempt_lock_path()`,
    the identical path that function itself locks on), and its state is
    re-read only after that lock is held: if the owning process is still
    genuinely running, this lock is currently held, `AIEventLockError`
    is raised after `lock_timeout`, and this attempt is left untouched
    for a later pass -- never archived out from under a live call.
    `path.exists()` is re-checked immediately after acquiring the lock too,
    since the owning process (or a concurrent reconciliation run) may have
    already resolved this exact attempt while this pass was waiting."""
    archived = 0
    completed = 0
    if not suggestions_dir.is_dir():
        return ReconciliationResult()
    for path in sorted(suggestions_dir.iterdir()):
        m = _RESERVED_RE.match(path.name)
        if not m:
            continue
        key, attempt = m.group(1), int(m.group(2))
        try:
            marker = ReservedAttemptMarker.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            continue  # resolved by its own owning process between the directory listing and this read
        _verify_marker_identity(marker, path=path, key=key, attempt=attempt)
        if marker.review_run_id != event_store.review_run_id:
            raise ValueError(
                f"{path}: marker.review_run_id {marker.review_run_id!r} does not match this reconciliation "
                f"pass's own event_store.review_run_id {event_store.review_run_id!r}"
            )

        lock_path = attempt_lock_path(event_store, stage=stage, work_item_id=marker.work_item_id)
        try:
            with kernel_lock(
                lock_path, error_cls=AIEventLockError,
                description=f"reconcile:{stage}:{marker.work_item_id}", timeout=lock_timeout,
            ):
                if not path.exists():
                    continue  # the owning process (or another reconciliation pass) already resolved this
                content_path = _content_path(suggestions_dir, key, attempt)
                if not content_path.is_file():
                    interrupted_path = _interrupted_path(suggestions_dir, key, attempt)
                    if interrupted_path.is_file():
                        # A crash between writing the archive and deleting this
                        # reservation must never re-archive with a fresh timestamp
                        # on the next restart -- but the archive's mere presence is
                        # not proof it describes THIS reservation. Load and
                        # cross-check it against the marker before deleting the
                        # marker out from under a mismatched archive.
                        existing = _load_interrupted_record(interrupted_path, key=key, attempt=attempt)
                        if existing.ai_use_event_id != marker.event_id or existing.reserved_at != marker.reserved_at:
                            raise ValueError(
                                f"{interrupted_path}: existing archive (ai_use_event_id={existing.ai_use_event_id!r}, "
                                f"reserved_at={existing.reserved_at!r}) does not match this reservation's own "
                                f"marker (event_id={marker.event_id!r}, reserved_at={marker.reserved_at!r})"
                            )
                    else:
                        interrupted_record = build_interrupted_call_record(
                            extraction_run_id=marker.extraction_run_id, record_id=marker.record_id,
                            field_id=marker.field_id, attempt=marker.attempt, ai_use_event_id=marker.event_id,
                            review_run_id=marker.review_run_id, work_item_id=marker.work_item_id,
                            reserved_at=marker.reserved_at, archived_at=now(),
                        )
                        _atomic_write_model(interrupted_path, interrupted_record)
                        archived += 1
                    path.unlink()
                    _fsync_dir(suggestions_dir)
                    continue
                record = _load_attempt_record(content_path, key=key, attempt=attempt)
                if record.ai_use_event_id != marker.event_id:
                    raise ValueError(
                        f"{content_path}: record.ai_use_event_id {record.ai_use_event_id!r} does not match "
                        f"its own reservation marker's event_id {marker.event_id!r}"
                    )
                _publish_event_for_record(event_store, record, stage=stage)
                os.replace(path, _completed_path(suggestions_dir, key, attempt))
                _fsync_dir(suggestions_dir)
                completed += 1
        except AIEventLockError:
            # Still actively held by the process that made this reservation --
            # a genuinely in-flight call, not a crash artifact. Leave it alone.
            continue
    return ReconciliationResult(archived_interrupted=archived, completed=completed)


class _CapExceededStop(Exception):
    """Internal signal only -- never escapes `run_suggestion_engine()`.
    Raised after the triggering attempt's own `FailedCallRecord` has
    already been durably written and published, so the cap-exceeded stop
    never loses the record of the call that tripped it."""

    def __init__(self, record: FailedCallRecord) -> None:
        super().__init__("cost cap exceeded")
        self.record = record


class _AlreadyHandled(Exception):
    """Internal signal only -- never escapes `run_suggestion_engine()`.
    Raised from INSIDE `_dispatch_one()`'s own `reserved_attempt()` lock,
    after re-scanning this work item's attempts under that lock, when a
    concurrent process (or an earlier pass of this same call) already
    reached a terminal-enough state for this exact work item between
    `run_suggestion_engine()`'s own pre-lock fast-path check and this
    lock's acquisition. That outer check alone is only a fast path -- it
    is NOT the authoritative eligibility decision, precisely so a second,
    concurrent engine run against the same `suggestions_dir` can never
    make a duplicate paid call for a work item another process just
    finished."""

    def __init__(
        self, status: Literal["skipped_already_validated", "skipped_not_retried"], record: AttemptRecord,
    ) -> None:
        super().__init__(status)
        self.status = status
        self.record = record


def _best_effort_raw_response(raw_text: Any) -> Any:
    """`RejectedSuggestion.raw_response: Any` never requires a validated
    `RawSuggestion` -- for a response `parse_suggestion_response()`
    rejected, this decodes as much real structure as actually exists (a
    dict, if the text was at least valid JSON) rather than nothing at all,
    falling back to the literal raw text only when even that fails."""
    if isinstance(raw_text, (bytes, bytearray)):
        raw_text = raw_text.decode("utf-8", errors="replace")
    if isinstance(raw_text, str):
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
    return raw_text


@dataclass(frozen=True)
class SuggestionEngineItemOutcome:
    record_id: str
    field_id: str
    work_item_id: str
    status: Literal[
        "skipped_already_validated", "skipped_not_retried", "skipped_cap_exceeded", "dispatched",
    ]
    attempt: Optional[int] = None
    attempt_record: Optional[AttemptRecord] = None


@dataclass(frozen=True)
class SuggestionEngineResult:
    items: Tuple[SuggestionEngineItemOutcome, ...]
    reconciliation: ReconciliationResult
    interrupted_total: int
    cap_exceeded: bool


def _find_validated(records: Dict[int, AttemptRecord]) -> Optional[ValidatedSuggestion]:
    for record in records.values():
        if isinstance(record, ValidatedSuggestion):
            return record
    return None


def _dispatch_one(
    entry: ReviewPacketEntry, *, extraction_run_id: str, review_run_id: str, suggestions_dir: Path,
    event_store: AIUseEventStore, client: Any, model: str, provider: str, component_versions: Dict[str, str],
    model_config_id: str, task_purpose: str, prompt_template_version: str, access_info: Optional[str],
    retry_work_item_ids: Sequence[str], now: Callable[[], str], is_cap_exceeded: Callable[[BaseException], bool],
    max_completion_tokens: Optional[int] = None,
) -> Tuple[int, AttemptRecord]:
    """Sec. 3.4's crash-safe three-step state machine for exactly one
    real (or injected-fake, for offline tests) LLM call. All three steps
    -- reserve, call, publish+complete -- run under `reserved_attempt()`'s
    single per-work-item lock, matching the `dedup.py` instrumentation
    convention this whole codebase already uses for every other
    instrumented call site.

    Raises `_AlreadyHandled` (never returns a spurious dispatch) if a
    re-scan taken under that same lock finds this work item already
    terminal -- see `_AlreadyHandled`'s own docstring for why this
    re-check is authoritative and `run_suggestion_engine()`'s own
    pre-lock check is not."""
    record_id, field_id = entry.record_id, entry.field_id
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id,
    )
    stage = EXTRACTION_RESOLUTION_SUGGESTION_STAGE
    key = safe_key(extraction_run_id, record_id, field_id)
    prompt = build_suggestion_prompt(entry)
    original_field_result_hash = sha256_hex(entry.field_result.model_dump(mode="json"))
    input_artifact_refs = [
        f"extraction_review_packet_entry:{record_id}:{field_id}",
        f"original_field_result_sha256:{original_field_result_hash}",
    ]

    def _extra_occupied(candidate_attempt: int) -> bool:
        return _any_marker_exists(suggestions_dir, key, candidate_attempt)

    with reserved_attempt(event_store, stage=stage, work_item_id=work_item_id, extra_occupied=_extra_occupied) as attempt:
        records = _scan_attempts_for_key(suggestions_dir, key)
        validated = _find_validated(records)
        if validated is not None:
            raise _AlreadyHandled("skipped_already_validated", validated)
        if records and work_item_id not in retry_work_item_ids:
            raise _AlreadyHandled("skipped_not_retried", records[max(records)])

        event_id = compute_event_id(review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=attempt)
        marker = ReservedAttemptMarker(
            work_item_id=work_item_id, attempt=attempt, event_id=event_id, review_run_id=review_run_id,
            extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, reserved_at=now(),
        )
        _atomic_write_model(_reserved_path(suggestions_dir, key, attempt), marker)

        try:
            call_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if max_completion_tokens is not None:
                call_kwargs["max_completion_tokens"] = max_completion_tokens
            raw_response = client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            draft = AIUseEventDraft(
                review_run_id=review_run_id, work_item_id=work_item_id, attempt=attempt, model=model,
                provider=provider, component_versions=component_versions, access_info=access_info,
                task_purpose=task_purpose, model_config_id=model_config_id,
                prompt_template_version=prompt_template_version, prompt_template_text=prompt,
                prompt_parameters={}, rendered_prompt_hash=sha256_hex(prompt),
                input_artifact_refs=input_artifact_refs, event_timestamp=now(),
            )
            record: AttemptRecord = build_failed_call_record(
                extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, attempt=attempt,
                original_field_result_hash=original_field_result_hash,
                ai_use_event_id=event_id, ai_use_event_draft=draft, error_type=type(exc).__name__,
                error_message=sanitize_call_error_text(str(exc)), failed_at=now(),
            )
            _atomic_write_model(_content_path(suggestions_dir, key, attempt), record)
            _publish_event_for_record(event_store, record, stage=stage)
            os.replace(_reserved_path(suggestions_dir, key, attempt), _completed_path(suggestions_dir, key, attempt))
            _fsync_dir(suggestions_dir)
            if is_cap_exceeded(exc):
                raise _CapExceededStop(record) from exc
            return attempt, record

        try:
            raw_text = raw_response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            envelope_error: Optional[Exception] = exc
            raw_text = None
        else:
            envelope_error = None
        usage = getattr(raw_response, "usage", None)
        draft = AIUseEventDraft(
            review_run_id=review_run_id, work_item_id=work_item_id, attempt=attempt, model=model,
            provider=provider, component_versions=component_versions, access_info=access_info,
            task_purpose=task_purpose, model_config_id=model_config_id,
            prompt_template_version=prompt_template_version, prompt_template_text=prompt,
            prompt_parameters={}, rendered_prompt_hash=sha256_hex(prompt),
            input_artifact_refs=input_artifact_refs, event_timestamp=now(),
        )
        if envelope_error is not None:
            # The provider call itself succeeded -- it returned SOME response
            # object -- but its shape doesn't match the expected
            # choices[0].message.content envelope. Same rule as a parse
            # failure below: a rejected successful call, never a
            # FailedCallRecord, and never an un-terminated .reserved.json
            # left for reconciliation to later misclassify as an
            # interruption of unknown outcome.
            record = build_rejected_suggestion(
                extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, attempt=attempt,
                original_field_result_hash=original_field_result_hash,
                ai_use_event_id=event_id, ai_use_event_draft=draft, raw_response=repr(raw_response),
                rejection_reasons=(sanitize_call_error_text(f"malformed response envelope: {envelope_error}"),),
                prompt_tokens=None, completion_tokens=None, total_tokens=None, estimated_cost_usd=None,
                created_at=now(),
            )
            _atomic_write_model(_content_path(suggestions_dir, key, attempt), record)
            _publish_event_for_record(event_store, record, stage=stage)
            os.replace(_reserved_path(suggestions_dir, key, attempt), _completed_path(suggestions_dir, key, attempt))
            _fsync_dir(suggestions_dir)
            return attempt, record

        try:
            parsed = parse_suggestion_response(raw_text)
        except SuggestionParseError as exc:
            # The provider call itself succeeded -- it returned real, if unparseable,
            # content -- so this is a content-validation rejection (call_status="success"),
            # never a FailedCallRecord (docstring, above). raw_response carries the
            # best-effort decoded content since no validated RawSuggestion exists.
            record = build_rejected_suggestion(
                extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, attempt=attempt,
                original_field_result_hash=original_field_result_hash,
                ai_use_event_id=event_id, ai_use_event_draft=draft,
                raw_response=_best_effort_raw_response(raw_text),
                rejection_reasons=(sanitize_call_error_text(str(exc)),),
                prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None), estimated_cost_usd=None, created_at=now(),
            )
            _atomic_write_model(_content_path(suggestions_dir, key, attempt), record)
            _publish_event_for_record(event_store, record, stage=stage)
            os.replace(_reserved_path(suggestions_dir, key, attempt), _completed_path(suggestions_dir, key, attempt))
            _fsync_dir(suggestions_dir)
            return attempt, record

        record = validate_suggestion(
            parsed, field_spec=entry.field_spec, field_result=entry.field_result, evidence_index=entry.evidence_index,
            extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id, attempt=attempt,
            ai_use_event_id=event_id, ai_use_event_draft=draft, created_at=now(),
            prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None), estimated_cost_usd=None,
        )
        _atomic_write_model(_content_path(suggestions_dir, key, attempt), record)
        _publish_event_for_record(event_store, record, stage=stage)
        os.replace(_reserved_path(suggestions_dir, key, attempt), _completed_path(suggestions_dir, key, attempt))
        _fsync_dir(suggestions_dir)
        return attempt, record


def run_suggestion_engine(
    packet: ReviewPacket, *, extraction_run_id: str, review_run_id: str, suggestions_dir: Path,
    event_store: AIUseEventStore, client: Any, model: str, provider: str, component_versions: Dict[str, str],
    model_config_id: str, task_purpose: str = "extraction_resolution_suggestion",
    prompt_template_version: str = "1", access_info: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    retry_work_item_ids: Sequence[str] = (), now: Callable[[], str] = _utc_now_iso,
    is_cap_exceeded: Callable[[BaseException], bool] = _default_is_cap_exceeded,
) -> SuggestionEngineResult:
    """Orchestrates the whole resolution-suggestion dispatch pass for one
    packet (sec. 3.3/3.4/3.8): startup reconciliation first (never new
    dispatch before an interrupted prior run's markers are resolved),
    then one crash-safe attempt per still-eligible `(record_id,
    field_id)` -- a key with a persisted `ValidatedSuggestion` is
    permanently skipped; a key whose latest attempt is a
    `RejectedSuggestion`/`FailedCallRecord`/archived `InterruptedCallRecord`
    is skipped too, unless its own `work_item_id` is explicitly named in
    `retry_work_item_ids`, in which case it gets a genuinely fresh,
    higher attempt number via `reserved_attempt()`'s own allocator.

    `client`/`model` are injectable (mirroring `nodes/screening.py`'s own
    `client=`/`model=` call-site convention) so this function is fully
    offline-testable via a fake client object -- no network access is
    ever performed by this function itself, per the standing
    offline-tests-only authorization; `client` is the caller's
    responsibility to construct (a real `OpenAI(api_key=...)` routed
    through `CostCapProxy` via `OPENAI_BASE_URL`, or a fake, in-memory
    stub in tests).

    Once `is_cap_exceeded()` reports a cap-exceeded call, the triggering
    attempt is still durably recorded as a `FailedCallRecord` (never
    lost), but no further item in this call is dispatched -- each
    remaining item is reported as `skipped_cap_exceeded` rather than
    silently omitted.

    The eligibility check below (per-item, before acquiring any lock) is
    only a fast path to avoid needlessly locking/scanning an obviously
    already-terminal item -- it is NOT the authoritative decision. The
    authoritative recheck happens inside `_dispatch_one()`, under its own
    per-work-item lock (`_AlreadyHandled`), so a second, concurrent
    `run_suggestion_engine()` call against the same `suggestions_dir` can
    never make a duplicate paid call for a work item another process
    completed in the meantime."""
    if packet.run_id != extraction_run_id:
        raise ValueError(f"packet.run_id {packet.run_id!r} does not match extraction_run_id {extraction_run_id!r}")
    if event_store.review_run_id != review_run_id:
        raise ValueError(
            f"event_store.review_run_id {event_store.review_run_id!r} does not match review_run_id {review_run_id!r}"
        )

    suggestions_dir = Path(suggestions_dir)
    reconciliation = reconcile_suggestion_attempts(suggestions_dir, event_store, now=now)

    items: List[SuggestionEngineItemOutcome] = []
    cap_exceeded = False
    for entry in packet.entries:
        record_id, field_id = entry.record_id, entry.field_id
        work_item_id = compute_expected_work_item_id(
            extraction_run_id=extraction_run_id, record_id=record_id, field_id=field_id,
        )
        key = safe_key(extraction_run_id, record_id, field_id)
        records = _scan_attempts_for_key(suggestions_dir, key)

        validated = _find_validated(records)
        if validated is not None:
            items.append(
                SuggestionEngineItemOutcome(record_id, field_id, work_item_id, "skipped_already_validated", attempt_record=validated)
            )
            continue
        if records and work_item_id not in retry_work_item_ids:
            items.append(
                SuggestionEngineItemOutcome(record_id, field_id, work_item_id, "skipped_not_retried", attempt_record=records[max(records)])
            )
            continue
        if cap_exceeded:
            items.append(SuggestionEngineItemOutcome(record_id, field_id, work_item_id, "skipped_cap_exceeded"))
            continue

        try:
            attempt, record = _dispatch_one(
                entry, extraction_run_id=extraction_run_id, review_run_id=review_run_id,
                suggestions_dir=suggestions_dir, event_store=event_store, client=client, model=model,
                provider=provider, component_versions=component_versions, model_config_id=model_config_id,
                task_purpose=task_purpose, prompt_template_version=prompt_template_version,
                access_info=access_info, retry_work_item_ids=retry_work_item_ids, now=now,
                is_cap_exceeded=is_cap_exceeded,
                max_completion_tokens=max_completion_tokens,
            )
        except _CapExceededStop as stop:
            cap_exceeded = True
            items.append(
                SuggestionEngineItemOutcome(record_id, field_id, work_item_id, "dispatched", attempt_record=stop.record)
            )
            continue
        except _AlreadyHandled as already:
            items.append(
                SuggestionEngineItemOutcome(record_id, field_id, work_item_id, already.status, attempt_record=already.record)
            )
            continue
        items.append(SuggestionEngineItemOutcome(record_id, field_id, work_item_id, "dispatched", attempt, record))

    return SuggestionEngineResult(
        items=tuple(items), reconciliation=reconciliation,
        interrupted_total=count_interrupted_suggestions(suggestions_dir), cap_exceeded=cap_exceeded,
    )
