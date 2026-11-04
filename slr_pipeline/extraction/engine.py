"""Per-document and per-run orchestration: parse -> deterministic batches
-> call (injected) -> verify evidence -> validate value -> aggregate ->
checkpoint. The batch-call function is INJECTED (`BatchCallFn`)
specifically so tests never need a real OpenAI call or a real Docling
parse of a real PDF to exercise this logic --
`openai_runner.run_openai_batch_extraction` is just one implementation of
that signature; a later local-model runner would be another, wired in by
the CALLER (never auto-detected here).

Every ATTEMPTED batch -- whether the call transport-failed, the response
was incoherent, or it completed cleanly -- produces exactly one
`BatchExecutionRecord` (review finding: "Every attempted batch gets a
BatchExecutionRecord, including transport, coherence, and value-
validation failures"). Nothing is silently skipped or discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .aggregation import BatchFieldObservation, aggregate_field
from .batch_cap import ExtractionBatchCapExceededError, enforce_extraction_batch_cap
from .batching import DEFAULT_MAX_BATCH_CHARS, build_batches
from .checkpoint import BlockedDocumentStore, DocumentCheckpoint, ExtractionCheckpointStore
from .evidence import EvidenceVerificationError, verify_evidence
from .openai_runner import (
    BatchCallTelemetry, ExtractionCallError, OpenAIExtractionConfig, RawBatchResponse, RawFieldResponse,
)
from .parsing import DoclingParserOptions, DocumentParseError, ParsedChunk, parse_pdf
from .results import BatchExecutionRecord, DocumentExtractionResult, EvidenceSpanModel, FieldResult, InvalidValueObservation
from .run_identity import DocumentIdentity, compute_run_id
from .schema import ExtractionFieldSpec, ExtractionSchema, validate_value_for_field

# (list[ParsedChunk]) -> (RawBatchResponse, BatchCallTelemetry); raises ExtractionCallError.
BatchCallFn = Callable[[list[ParsedChunk]], tuple[RawBatchResponse, BatchCallTelemetry]]


def _check_batch_coherence(schema: ExtractionSchema, raw_response: RawBatchResponse) -> Optional[str]:
    """Batch-level response-coherence check, run BEFORE any per-field
    processing -- a batch either passes this wholly, or the WHOLE batch
    (not just the offending field) is marked failed. Returns an error
    message, or `None` if the response is coherent.

    Checks (review finding 3's "structural response-coherence
    validators" + finding: "Unknown fields or incoherent outcome/findings
    combinations fail that batch explicitly; they must not be silently
    discarded"):
      - every schema field_id is present in the response (missing);
      - the response contains no field_id outside the schema (unknown --
        previously silently ignored, now an explicit failure);
      - no field reports outcome="not_found" while still carrying
        findings (previously silently discarded findings);
      - no field reports outcome="found" with an EMPTY findings list
        (previously silently downgraded to "unclear" per field -- now a
        batch-level coherence failure, since a model claiming "found"
        with nothing to show is a malformed response, not a legitimate
        per-field ambiguity)."""
    expected_ids = set(schema.field_map())
    response_ids = set(raw_response.fields)

    missing = expected_ids - response_ids
    if missing:
        return f"response missing field_id(s): {sorted(missing)}"

    unknown = response_ids - expected_ids
    if unknown:
        return f"response contains unknown field_id(s): {sorted(unknown)}"

    incoherent = [
        field_id
        for field_id, raw_field in raw_response.fields.items()
        if (raw_field.outcome == "not_found" and raw_field.findings)
        or (raw_field.outcome == "found" and not raw_field.findings)
    ]
    if incoherent:
        return f"incoherent outcome/findings combination for field_id(s): {sorted(incoherent)}"

    return None


def _process_field_response(
    field_id: str,
    field_spec: ExtractionFieldSpec,
    raw_field: RawFieldResponse,
    chunk_map: dict[str, ParsedChunk],
    seen_keys: set[tuple[str, str]],
) -> list[BatchFieldObservation]:
    """Turns one batch's raw response for one field into 0+ verified,
    value-validated observations. By the time this runs,
    `_check_batch_coherence` has already guaranteed `raw_field.outcome`
    and `raw_field.findings` are a coherent pair (never "not_found" with
    findings, never "found" with none), so this function does not need to
    re-check that combination.

    A finding surviving with at least one VERIFIED evidence item AND a
    value that passes `schema.validate_value_for_field` becomes a "found"
    observation. Anything else -- unverifiable evidence, or evidence that
    verifies but backs an INVALID value -- becomes "unclear" instead: an
    unverifiable or invalid claim is never silently accepted as a value,
    and never silently dropped either (requirement 7: "route unverifiable
    quotations to review"; review finding 3: an invalid value "must
    become an instrumented review/failure outcome, never a clean
    extraction")."""
    if raw_field.outcome == "not_found":
        return [BatchFieldObservation(field_id=field_id, outcome="not_found")]

    if raw_field.outcome == "unclear" and not raw_field.findings:
        return [BatchFieldObservation(field_id=field_id, outcome="unclear", rationale=raw_field.rationale)]

    observations: list[BatchFieldObservation] = []
    total_rejected = 0

    for finding in raw_field.findings:
        verified: list[EvidenceSpanModel] = []
        for raw_ev in finding.evidence:
            try:
                verified_evidence = verify_evidence(
                    chunk_id=raw_ev.chunk_id, quotation=raw_ev.quotation, chunk_map=chunk_map, seen_keys=seen_keys,
                )
                verified.append(EvidenceSpanModel.from_verified(verified_evidence))
            except EvidenceVerificationError:
                total_rejected += 1

        if not verified:
            observations.append(
                BatchFieldObservation(
                    field_id=field_id, outcome="unclear",
                    rationale=finding.rationale or f"unverifiable claim: {finding.value!r}",
                )
            )
            continue

        if raw_field.outcome != "found":
            # outcome == "unclear" with findings: ambiguous by the
            # model's own report -- even fully-verified evidence stays
            # "unclear", never promoted to "found" just because its
            # quotation happens to check out.
            observations.append(
                BatchFieldObservation(
                    field_id=field_id, outcome="unclear", rationale=finding.rationale, evidence=tuple(verified),
                )
            )
            continue

        is_valid, typed_value = validate_value_for_field(field_spec, finding.value)
        if not is_valid:
            reason = f"value not valid for value_kind={field_spec.value_kind.value!r}: {finding.value!r}"
            observations.append(
                BatchFieldObservation(
                    field_id=field_id, outcome="unclear", value=finding.value,
                    rationale=f"{reason} (raw string retained for audit; never used as a final value)",
                    evidence=tuple(verified), invalid_value_count=1,
                    invalid_value_observation=InvalidValueObservation(
                        raw_value=finding.value, value_kind=field_spec.value_kind.value, reason=reason,
                        evidence=tuple(verified),
                    ),
                )
            )
            continue

        observations.append(
            BatchFieldObservation(
                field_id=field_id, outcome="found", value=finding.value, typed_value=typed_value,
                confidence=finding.confidence, rationale=finding.rationale, evidence=tuple(verified),
            )
        )

    if total_rejected and observations:
        last = observations[-1]
        observations[-1] = replace(last, rejected_evidence_count=last.rejected_evidence_count + total_rejected)

    return observations


@dataclass(frozen=True)
class DocumentExtractionOutcome:
    result: DocumentExtractionResult


def extract_document(
    *,
    document_id: str,
    pdf_path: str | Path,
    schema: ExtractionSchema,
    call_batch_fn: BatchCallFn,
    parser_options: Optional[DoclingParserOptions] = None,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    max_batches: Optional[int] = None,
    expected_artifacts_manifest_digest: Optional[str] = None,
) -> DocumentExtractionOutcome:
    """Extracts every field of `schema` for one document. Raises
    `DocumentParseError` if the PDF cannot be parsed at all (no fallback
    parser exists -- see `parsing.py`); the caller decides what "route to
    review" means for a document that never even got parsed.

    `max_batches` (additive, default `None` -- unchanged behavior for
    every existing caller) enforces a ceiling on the document's own
    deterministic batch count via `batch_cap.enforce_extraction_batch_cap`
    -- the SAME parse+batch computation used for the check is then reused
    for the real batch loop below, never a second independent parse.
    Raises `ExtractionBatchCapExceededError` (never truncates) if
    exceeded; a caller enabling this MUST also pass
    `expected_artifacts_manifest_digest` (required together -- there is
    no safe default for a caller newly opting into an offline-only
    guarantee)."""
    if max_batches is not None:
        if expected_artifacts_manifest_digest is None:
            raise ValueError("expected_artifacts_manifest_digest is required when max_batches is set")
        parsed, batches = enforce_extraction_batch_cap(
            pdf_path, expected_artifacts_manifest_digest=expected_artifacts_manifest_digest,
            max_batches=max_batches, parser_options=parser_options, max_batch_chars=max_batch_chars,
        )
    else:
        parsed = parse_pdf(
            pdf_path, options=parser_options, expected_artifacts_manifest_digest=expected_artifacts_manifest_digest,
        )
        batches = build_batches(parsed.chunks, max_batch_chars=max_batch_chars)
    chunk_map = parsed.chunk_map()

    observations_by_field: dict[str, list[BatchFieldObservation]] = {f.field_id: [] for f in schema.fields}
    batch_executions: list[BatchExecutionRecord] = []

    for batch in batches:
        batch_chunks = [chunk_map[cid] for cid in batch.chunk_ids]
        chunk_ids = tuple(c.chunk_id for c in batch_chunks)

        try:
            raw_response, telemetry = call_batch_fn(batch_chunks)
        except ExtractionCallError as exc:
            # A provider-model mismatch (review finding P) carries the
            # ACTUAL returned model/usage on the exception itself --
            # `getattr(..., None)` duck-types so an ordinary
            # ExtractionCallError (no such attributes) still records
            # `None`s, exactly as before. The response CONTENT is
            # rejected either way; only its identity/usage survives.
            batch_executions.append(
                BatchExecutionRecord(
                    batch_index=batch.batch_index, chunk_ids=chunk_ids, status="failed",
                    error=f"{type(exc).__name__}",
                    provider_model=getattr(exc, "actual_provider_model", None),
                    prompt_tokens=getattr(exc, "prompt_tokens", None),
                    completion_tokens=getattr(exc, "completion_tokens", None),
                    total_tokens=getattr(exc, "total_tokens", None),
                )
            )
            continue

        coherence_error = _check_batch_coherence(schema, raw_response)
        if coherence_error is not None:
            batch_executions.append(
                BatchExecutionRecord(
                    batch_index=batch.batch_index, chunk_ids=chunk_ids, status="failed",
                    error=f"ValueError: {coherence_error}",
                    provider_model=telemetry.provider_model, prompt_tokens=telemetry.prompt_tokens,
                    completion_tokens=telemetry.completion_tokens, total_tokens=telemetry.total_tokens,
                )
            )
            continue

        batch_rejected = 0
        batch_invalid = 0
        for field_id, raw_field in raw_response.fields.items():
            field_spec = schema.field_map()[field_id]
            seen_keys: set[tuple[str, str]] = set()
            field_observations = _process_field_response(field_id, field_spec, raw_field, chunk_map, seen_keys)
            observations_by_field[field_id].extend(field_observations)
            batch_rejected += sum(o.rejected_evidence_count for o in field_observations)
            batch_invalid += sum(o.invalid_value_count for o in field_observations)

        batch_executions.append(
            BatchExecutionRecord(
                batch_index=batch.batch_index, chunk_ids=chunk_ids, status="completed",
                provider_model=telemetry.provider_model, prompt_tokens=telemetry.prompt_tokens,
                completion_tokens=telemetry.completion_tokens, total_tokens=telemetry.total_tokens,
                rejected_evidence_count=batch_rejected, invalid_value_count=batch_invalid,
            )
        )

    document_is_complete = bool(batch_executions) and all(b.status == "completed" for b in batch_executions)

    field_results: list[FieldResult] = [
        aggregate_field(field_spec, observations_by_field[field_spec.field_id], document_is_complete=document_is_complete)
        for field_spec in schema.fields
    ]

    result = DocumentExtractionResult(
        document_id=document_id, fields=tuple(field_results), batch_executions=tuple(batch_executions),
        planned_chunk_ids_by_batch=tuple(batch.chunk_ids for batch in batches),
    )
    return DocumentExtractionOutcome(result=result)


def run_extraction(
    *,
    documents: list[tuple[str, str]],  # [(document_id, pdf_path), ...], in order
    schema: ExtractionSchema,
    call_batch_fn: BatchCallFn,
    output_dir: str | Path,
    model_config: OpenAIExtractionConfig,
    parser_options: Optional[DoclingParserOptions] = None,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    max_batches: Optional[int] = None,
    expected_artifacts_manifest_digest: Optional[str] = None,
) -> tuple[str, list[DocumentExtractionResult]]:
    """Runs (or resumes) extraction over `documents`, checkpointing after
    each document. `model_config` is the SAME `OpenAIExtractionConfig`
    passed to `call_batch_fn`'s own construction (a test may pass a
    fake/local config shape with the same field names instead -- the run
    identity only reads `.model`/`.model_config_id`/`.reasoning_effort`/
    `.max_completion_tokens`/`.expected_provider_model`/
    `.prompt_template_version`/`.output_schema_version`, never constructs
    a client from it).

    `max_batches` (additive, default `None` -- unchanged behavior) is
    passed through to `extract_document()`. A document whose own batch
    plan exceeds it is never billed: it is durably recorded via
    `BlockedDocumentStore` (reserved BEFORE the cap check runs, so a
    restart never re-attempts a document already known to be blocked)
    and excluded from both `results` and the published `RunManifest`'s
    `document_ids` -- it was never attempted, so it must not be
    represented as a (even incomplete) checkpoint. Query
    `BlockedDocumentStore(output_dir=output_dir, run_id=run_id).
    blocked_document_ids()` to see which documents were held back."""
    document_ids = [document_id for document_id, _ in documents]
    duplicates = sorted({d for d in document_ids if document_ids.count(d) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate document_id(s) in documents: {duplicates} -- each document_id must be unique "
            "within a run (a duplicate would silently collide on the same checkpoint)"
        )

    from .hashing import sha256_hex
    from .parsing import PARSER_NAME, installed_docling_version

    parser_opts = parser_options or DoclingParserOptions()
    document_identities = [
        DocumentIdentity(
            record_id=document_id,
            pdf_path=str(Path(pdf_path).resolve()),
            pdf_sha256=_file_sha256(pdf_path),
        )
        for document_id, pdf_path in documents
    ]
    run_id = compute_run_id(
        documents=document_identities,
        extraction_schema_hash=sha256_hex(schema.model_dump(mode="json")),
        parser_name=PARSER_NAME, parser_version=installed_docling_version(), parser_options=parser_opts,
        max_batch_chars=max_batch_chars,
        prompt_template_version=model_config.prompt_template_version,
        output_schema_version=model_config.output_schema_version,
        model=model_config.model, model_config_id=model_config.model_config_id,
        reasoning_effort=model_config.reasoning_effort, max_completion_tokens=model_config.max_completion_tokens,
        expected_provider_model=model_config.expected_provider_model,
        artifacts_manifest_digest=expected_artifacts_manifest_digest,
        max_batches=max_batches,
    )
    store = ExtractionCheckpointStore(output_dir=output_dir, run_id=run_id)
    blocked_store = BlockedDocumentStore(output_dir=output_dir, run_id=run_id)
    identity_by_id = {d.record_id: d for d in document_identities}

    results: list[DocumentExtractionResult] = []
    checkpoints: dict[str, DocumentCheckpoint] = {}
    attempted_document_ids: list[str] = []
    for document_id, pdf_path in documents:
        if store.has_checkpoint(document_id):
            checkpoint = store.load_checkpoint(document_id)
            results.append(checkpoint.result)
            checkpoints[document_id] = checkpoint
            attempted_document_ids.append(document_id)
            continue
        if max_batches is not None and blocked_store.is_blocked(document_id):
            # Already durably blocked by a prior run/attempt -- never
            # auto-retried, and never counted in results, but named in
            # the manifest's own requested_document_ids/blocked_document_ids.
            continue
        identity = identity_by_id[document_id]
        try:
            outcome = extract_document(
                document_id=document_id, pdf_path=pdf_path, schema=schema, call_batch_fn=call_batch_fn,
                parser_options=parser_opts, max_batch_chars=max_batch_chars, max_batches=max_batches,
                expected_artifacts_manifest_digest=expected_artifacts_manifest_digest,
            )
        except ExtractionBatchCapExceededError as exc:
            # Reserve/persist the blocked state BEFORE any batch of this
            # document could have been billed -- extract_document() raised
            # this before calling call_batch_fn even once (batches are
            # built, never executed, on this path).
            blocked_store.record_blocked(
                document_id=document_id, pdf_path=identity.pdf_path, pdf_sha256=identity.pdf_sha256,
                reason=str(exc), batch_count=len(exc.batches),
                max_batches=max_batches, parsed_document_sha256=exc.parsed.document_sha256,
                blocked_at=datetime.now(timezone.utc).isoformat(),
            )
            continue
        checkpoint = DocumentCheckpoint.build(
            run_id=run_id, document_id=document_id, pdf_path=identity.pdf_path, pdf_sha256=identity.pdf_sha256,
            schema=schema, parser_options=parser_opts, max_batch_chars=max_batch_chars, model_config=model_config,
            result=outcome.result, artifacts_manifest_digest=expected_artifacts_manifest_digest,
            max_batches=max_batches,
        )
        store.write_checkpoint(checkpoint)
        results.append(outcome.result)
        checkpoints[document_id] = checkpoint
        attempted_document_ids.append(document_id)

    # publish_manifest() derives EVERY manifest field itself, directly
    # from the verified on-disk checkpoints/blocked-records -- it is
    # never handed a caller-built RunManifest to trust (review finding
    # E). Blocked documents are excluded from `document_ids` (never
    # attempted, so they cannot be represented by the manifest's
    # checkpoint-coverage invariant) but ARE named in
    # `requested_document_ids`/`blocked_document_ids` (review finding:
    # a mixed blocked/attempted publish previously recomputed run_id
    # from only the attempted subset, which could never match this
    # store's own run_id -- always raising).
    if attempted_document_ids:
        store.publish_manifest(
            document_ids=tuple(attempted_document_ids), requested_document_ids=tuple(document_ids),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    return run_id, results


def _file_sha256(pdf_path: str | Path) -> str:
    from .hashing import file_sha256

    return file_sha256(pdf_path)
