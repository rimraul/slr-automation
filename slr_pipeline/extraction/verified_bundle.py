"""Shared "verified extraction run" API surface.

Relocated (2026-09-17, amber-quilting-osprey.md sec. 3.0) out of
`slr_pipeline.nodes.extraction`, where `load_and_verify_bundle()` /
`bundle_review_fields()` originated as private module internals. New
extraction-package code (the review-packet renderer, suggestion validator)
needs the exact same "verified bundle" guarantees they provide, and
importing private `nodes` internals from `extraction/` would be the wrong
dependency direction -- the `extraction` package owns bundle verification;
`nodes` imports it, not the reverse. `nodes/extraction.py` imports these
same four names back for its own existing use; `resolve_extraction_review_queue()`
itself did not move and its external behavior is unchanged.

Self-contained by design: this module imports nothing from
`slr_pipeline.nodes`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

from slr_pipeline.config import ReviewConfig
from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore, VerifiedRunBundle
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, validate_value_for_field
from slr_pipeline.state import ReviewState

REVIEW_STATUSES = {"unclear", "conflicting", "extraction_failed"}


def _stripped(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_schema(cfg: ReviewConfig) -> ExtractionSchema:
    if not cfg.extraction_schema:
        raise ValueError(
            "ReviewConfig.extraction_schema is not set, but full_text_records is "
            "non-empty and extraction_node was invoked. A serialized ExtractionSchema "
            "(schema.ExtractionSchema(...).model_dump(mode='json')) must be authored "
            "explicitly before extraction can run -- it is never silently defaulted."
        )
    return ExtractionSchema.model_validate(cfg.extraction_schema)


def bundle_review_fields(results) -> Dict[Tuple[str, str], Any]:
    """The ONE place every caller derives "which fields CURRENTLY require
    review" from -- always from a freshly-computed (`run_extraction()`'s
    own return) or independently verified (`VerifiedRunBundle.results`)
    result set, never from state. `results` is an iterable of real
    `DocumentExtractionResult` objects."""
    fields: Dict[Tuple[str, str], Any] = {}
    for result in results:
        for field in result.fields:
            if field.requires_human_review or field.status.value in REVIEW_STATUSES:
                fields[(result.document_id, field.field_id)] = field
    return fields


def load_and_verify_bundle(
    state: ReviewState, run_id: str,
) -> Tuple[ExtractionCheckpointStore, VerifiedRunBundle, ExtractionSchema, str]:
    """Shared entry point for every caller that must treat a run's
    checkpoints/manifest as authoritative (`resolve_extraction_review_queue`,
    `load_verified_extraction_resolution_artifact`, the review-packet
    renderer): builds the store, loads+verifies a `VerifiedRunBundle` from
    it, and cross-checks the extraction schema in `state["config"]` against
    the bundle's own verified `extraction_schema_hash` -- a mutated/mismatched
    config is rejected before it can be used to validate anything. Returns
    `(store, bundle, schema, schema_hash)`."""
    cfg = ReviewConfig.from_dict(state["config"])
    schema = load_schema(cfg)
    output_dir = cfg.ensure_output_dir()
    extraction_output_dir = output_dir / "extraction_runs"
    store = ExtractionCheckpointStore(output_dir=extraction_output_dir, run_id=run_id)

    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()

    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    if schema_hash != bundle.manifest.extraction_schema_hash:
        raise ValueError(
            f"the extraction schema in state['config'] (hash={schema_hash!r}) does not match this run's "
            f"own verified manifest schema hash ({bundle.manifest.extraction_schema_hash!r}) -- refusing "
            "to validate human-supplied values against a schema that is not structurally the run's own"
        )
    return store, bundle, schema, schema_hash


def validate_human_value(field_spec: ExtractionFieldSpec, value: Any, key: Tuple[str, str]) -> None:
    """Relocated (2026-09-17) from `slr_pipeline.nodes.extraction._validate_human_value`
    so `render_review_packet()`'s resolution-history checks can reuse the same
    validation `resolve_extraction_review_queue()` applies to a fresh resolution,
    rather than duplicating (and risking drifting from) it.

    Codex review finding D / round-2 finding D1: a human-supplied value must
    pass the SAME lexical/type/controlled-value boundary
    (`schema.validate_value_for_field`) the machine's own values pass -- an
    out-of-schema value requires a versioned schema change, never a bypass
    through attestation. Cardinality is enforced STRICTLY, not just accepted
    opportunistically: `cardinality="multiple"` REQUIRES a non-empty list of
    strings (a bare scalar is rejected, not silently wrapped);
    `cardinality="single"` REQUIRES a single scalar string (a list is
    rejected, not silently treated as one candidate). A blank/whitespace-only
    string never passes, regardless of value_kind, and a `multiple` field's
    list may not contain duplicate values (a duplicate has no defined meaning
    in this schema)."""
    if field_spec.cardinality == "multiple":
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"resolution for {key!r}: cardinality='multiple' field requires a non-empty list of "
                f"string values, got {value!r}"
            )
        candidates = value
    else:
        if isinstance(value, list):
            raise ValueError(
                f"resolution for {key!r}: cardinality='single' field requires a single scalar string "
                f"value, not a list ({value!r})"
            )
        if value is None:
            raise ValueError(f"resolution for {key!r}: outcome='extracted' requires a non-empty 'value'")
        candidates = [value]

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"resolution for {key!r}: value(s) must be non-blank strings, got {candidate!r}")
        is_valid, _typed = validate_value_for_field(field_spec, candidate)
        if not is_valid:
            raise ValueError(
                f"resolution for {key!r}: value {candidate!r} is not valid for "
                f"value_kind={field_spec.value_kind.value!r}"
                + (f", allowed_values={[av.value for av in field_spec.allowed_values]}"
                   if field_spec.allowed_values else "")
            )

    if field_spec.cardinality == "multiple" and len(set(candidates)) != len(candidates):
        raise ValueError(f"resolution for {key!r}: duplicate values in a 'multiple' field's value list: {candidates!r}")


def validate_human_evidence(evidence: Any, key: Tuple[str, str]) -> None:
    """Relocated (2026-09-17) from `slr_pipeline.nodes.extraction._validate_human_evidence`.

    Codex review finding D: bare truthiness (`evidence=[""]` passing) let an
    empty string satisfy "non-empty evidence". Every item must be a
    structured record with, at minimum, a non-blank `quotation_or_description`
    AND a non-blank `locator` (page/section/table/or another explicit
    reference a later reader could check) -- never re-verified against a
    chunk_map the way the machine's own evidence is (this human record IS the
    attestation), but never a bare unstructured string either."""
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"resolution for {key!r}: outcome='extracted' requires non-empty structured 'evidence'")
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"resolution for {key!r}: each evidence item must be a dict, got {item!r}")
        if not _stripped(item.get("quotation_or_description")):
            raise ValueError(
                f"resolution for {key!r}: each evidence item requires a non-blank 'quotation_or_description'"
            )
        if not _stripped(item.get("locator")):
            raise ValueError(
                f"resolution for {key!r}: each evidence item requires a non-blank 'locator' "
                "(e.g. a page, section, or table reference)"
            )


def verify_existing_resolution(
    resolution: Dict[str, Any], *, run_id: str, manifest_hash: str, schema_hash: str,
    review_fields: Dict[Tuple[str, str], Any], field_map: Dict[str, ExtractionFieldSpec],
) -> None:
    """Relocated (2026-09-17) from `slr_pipeline.nodes.extraction._verify_existing_resolution`
    so every consumer of a resolution -- the live queue-resolution path in
    `nodes/extraction.py` AND `render_review_packet()`'s
    `resolution_history_as_of_render` checks (`resolution_suggestions.py`) --
    validate against the exact same rules, not a partial duplicate of them.

    Codex review round-3 (hardened round-5): re-validates a resolution --
    whether ALREADY-CONSTRUCTED (loaded from state, from the durable
    artifact, or merged from both) or JUST-CONSTRUCTED immediately before
    publication -- before it is trusted to override a machine value or count
    toward "already resolved." A same-run resolution folded in from state
    was previously accepted after checking only its `run_id` and
    `(record_id, field_id)` key; a freshly-constructed resolution was
    previously published without passing through this at all, letting a
    tampered top-level queue-entry `"status"` flow straight into
    `original_status` unchecked (round-5 finding 1). Every binding and
    value/evidence rule is checked identically regardless of a resolution's
    origin."""
    key = (resolution.get("record_id"), resolution.get("field_id"))
    if resolution.get("run_id") != run_id:
        raise ValueError(f"resolution for {key!r} is not bound to run_id={run_id!r}")
    if resolution.get("manifest_hash") != manifest_hash:
        raise ValueError(f"resolution for {key!r} is not bound to this run's verified manifest_hash")
    if resolution.get("extraction_schema_hash") != schema_hash:
        raise ValueError(f"resolution for {key!r} is not bound to this run's verified extraction_schema_hash")

    field_result = review_fields.get(key)
    if field_result is None:
        raise ValueError(
            f"resolution for {key!r} does not correspond to any field the verified result set currently "
            "flags for review"
        )
    expected_hash = sha256_hex(field_result.model_dump(mode="json"))
    if resolution.get("original_field_result_hash") != expected_hash:
        raise ValueError(
            f"resolution for {key!r}: original_field_result_hash does not match the verified field result"
        )
    # Codex review round-4 finding 3 (closed at construction time too,
    # round-5 finding 1): original_status must match the VERIFIED field
    # result's own status -- never merely a queue entry's own top-level,
    # possibly-tampered "status" string.
    if resolution.get("original_status") != field_result.status.value:
        raise ValueError(
            f"resolution for {key!r}: original_status {resolution.get('original_status')!r} does not "
            f"match the verified field result's own status {field_result.status.value!r}"
        )

    reviewer = _stripped(resolution.get("reviewer"))
    note = _stripped(resolution.get("note"))
    resolved_at = _stripped(resolution.get("resolved_at"))
    if not reviewer or not note or not resolved_at:
        raise ValueError(f"resolution for {key!r} is missing a non-blank reviewer/note/resolved_at")
    try:
        datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"resolution for {key!r}: resolved_at {resolved_at!r} is not a valid ISO-8601 timestamp")

    outcome = resolution.get("outcome")
    if outcome not in ("extracted", "not_reported"):
        raise ValueError(f"resolution for {key!r}: outcome must be 'extracted' or 'not_reported', got {outcome!r}")

    if outcome == "extracted":
        field_spec = field_map.get(key[1])
        if field_spec is None:
            raise ValueError(f"resolution for {key!r}: field_id is not present in the run's own extraction schema")
        validate_human_value(field_spec, resolution.get("value"), key)
        validate_human_evidence(resolution.get("evidence"), key)
        if resolution.get("attested") is not True:
            raise ValueError(f"resolution for {key!r}: outcome='extracted' requires attested=True")
    else:
        # Codex review round-4 finding 3: a not_reported resolution
        # previously had no rule against ALSO carrying a (contradictory,
        # stray) value/evidence/attested -- the freshly-constructed path
        # never sets them for not_reported, so a resolution that DOES
        # have any of them here can only be a corrupted/forged entry.
        if resolution.get("value") is not None or resolution.get("evidence") is not None or resolution.get("attested") is not None:
            raise ValueError(
                f"resolution for {key!r}: outcome='not_reported' must not carry a value/evidence/attested"
            )
