"""Automatic reports, deliberately split into sections that must never be
blended together:

  - CoverageReport: computed from execution telemetry ALONE (no ground
    truth needed) -- how much ran, how much succeeded, what got flagged
    for review, which fields were never extracted anywhere in the
    corpus. This is yield, not correctness.
  - AuditedAccuracyReport: computed ONLY against a real
    `calibration.CalibrationCorpus` with human field-level judgments --
    value correctness, false-not_reported rate, over-extraction rate.
    Absent (never fabricated) when no calibration corpus is supplied.
  - EvidenceQualityReport: computed from `calibration.EvidenceAudit`
    objects (evidence-support PRECISION, locator accuracy -- did each
    system-returned quotation actually support its claim and carry a
    correct page/section) and, when a `CalibrationCorpus` is also
    supplied, reference-evidence RECALL (did the system find evidence a
    human reviewer already knows exists) -- review finding B: "reports.py
    exposes only field/status/value accuracy; it has no evidence-support,
    reference-evidence recall/completeness, or locator-accuracy report."

lit_review_protocol's explicit warning this split exists to satisfy:
"Instrument and count every fallback ... They must never resemble
genuine not_reported results" and "Evaluate both yield and quality" as
two DISTINCT things, never conflated into one "success rate" number.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .calibration import (
    CalibrationBindingError, CalibrationCorpus, EvidenceAudit, EvidenceAuditCoverageError,
    EvidenceAuditIdentityMismatchError, evidence_projection,
)
from .checkpoint import DocumentCheckpoint, RunManifest, VerifiedRunBundle
from .evidence import normalize_for_matching
from .results import DocumentExtractionResult, FieldResult, FieldStatus


class CoverageReport(BaseModel):
    """Yield/coverage -- what happened, not whether it was right."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_documents: int = Field(ge=0)
    complete_documents: int = Field(ge=0)
    incomplete_documents: int = Field(ge=0)
    total_batches: int = Field(ge=0)
    completed_batches: int = Field(ge=0)
    failed_batches: int = Field(ge=0)
    skipped_batches: int = Field(ge=0)
    total_rejected_evidence: int = Field(ge=0)
    field_occurrence: dict[str, dict[str, int]]  # field_id -> {status.value: count}
    zero_occurrence_field_ids: tuple[str, ...]  # fields NEVER "extracted" anywhere in this corpus
    review_required_count: int = Field(ge=0)
    review_required_rate: float = Field(ge=0.0, le=1.0)


def build_coverage_report(results: list[DocumentExtractionResult]) -> CoverageReport:
    total_documents = len(results)
    complete_documents = sum(1 for r in results if r.is_complete)
    total_batches = sum(len(r.batch_executions) for r in results)
    completed_batches = sum(1 for r in results for b in r.batch_executions if b.status == "completed")
    failed_batches = sum(1 for r in results for b in r.batch_executions if b.status == "failed")
    skipped_batches = sum(1 for r in results for b in r.batch_executions if b.status == "skipped")
    total_rejected_evidence = sum(f.rejected_evidence_count for r in results for f in r.fields)
    review_required_count = sum(1 for r in results for f in r.fields if f.requires_human_review)
    total_field_observations = sum(len(r.fields) for r in results)

    field_occurrence: dict[str, dict[str, int]] = {}
    for r in results:
        for f in r.fields:
            counts = field_occurrence.setdefault(f.field_id, {s.value: 0 for s in FieldStatus})
            counts[f.status.value] += 1

    zero_occurrence = tuple(
        field_id for field_id, counts in field_occurrence.items() if counts[FieldStatus.EXTRACTED.value] == 0
    )

    return CoverageReport(
        total_documents=total_documents,
        complete_documents=complete_documents,
        incomplete_documents=total_documents - complete_documents,
        total_batches=total_batches,
        completed_batches=completed_batches,
        failed_batches=failed_batches,
        skipped_batches=skipped_batches,
        total_rejected_evidence=total_rejected_evidence,
        field_occurrence=field_occurrence,
        zero_occurrence_field_ids=zero_occurrence,
        review_required_count=review_required_count,
        review_required_rate=(review_required_count / total_field_observations) if total_field_observations else 0.0,
    )


_ACCURACY_OUTCOMES = ("correct", "value_mismatch", "false_not_reported", "over_extraction", "status_mismatch")


class AuditedAccuracyReport(BaseModel):
    """Quality, against real human ground truth -- ONLY exists when a
    calibration corpus was supplied. Never present alongside a bare
    CoverageReport by accident; see `build_full_report()`.

    Coverage is reported EXPLICITLY, never silently absorbed (review
    finding 5: "reports silently skip missing documents and fields,
    permitting incomplete audits to look clean") -- a corpus case whose
    document never ran, or whose field is missing from that document's
    result, is counted in `skipped_missing_document_field_count` and
    `coverage_complete` becomes `False`, rather than the audit simply
    looking smaller with no signal that anything was skipped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_corpus_id: str
    calibration_corpus_version: str
    calibration_corpus_content_hash: str = Field(min_length=64, max_length=64)
    expected_field_count: int = Field(ge=0)
    audited_document_count: int = Field(ge=0)
    audited_field_count: int = Field(ge=0)
    skipped_missing_document_field_count: int = Field(ge=0)
    skipped_missing_field_count: int = Field(ge=0)
    coverage_complete: bool
    correct_count: int = Field(ge=0)
    value_mismatch_count: int = Field(ge=0)
    false_not_reported_count: int = Field(ge=0)
    over_extraction_count: int = Field(ge=0)
    status_mismatch_count: int = Field(ge=0)
    value_bearing_judgment_count: int = Field(ge=0)
    accuracy_rate: float = Field(ge=0.0, le=1.0)
    value_correctness_rate: float = Field(ge=0.0, le=1.0)
    false_not_reported_rate: float = Field(ge=0.0, le=1.0)
    over_extraction_rate: float = Field(ge=0.0, le=1.0)
    outcomes_by_field: dict[str, dict[str, int]]


def _classify_accuracy_outcome(system_field: FieldResult, correct_status: FieldStatus, correct_values: set[str]) -> str:
    system_status = system_field.status
    system_values = {v.value for v in system_field.values}

    if system_status == FieldStatus.NOT_REPORTED and correct_status != FieldStatus.NOT_REPORTED:
        return "false_not_reported"
    # Both value-bearing final statuses count as over-extraction against
    # genuine absence -- review finding O: a `conflicting` field still
    # asserts one or more values, so it is over-extraction against a
    # human `not_reported`, not a generic status_mismatch (which
    # understated the published over_extraction_rate).
    if system_status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING) and correct_status == FieldStatus.NOT_REPORTED:
        return "over_extraction"
    if system_status == correct_status:
        if system_status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING):
            return "correct" if system_values == correct_values else "value_mismatch"
        return "correct"
    return "status_mismatch"


def build_audited_accuracy_report(
    checkpoints: list[DocumentCheckpoint], corpus: CalibrationCorpus,
) -> AuditedAccuracyReport:
    """Takes VERIFIED `DocumentCheckpoint` objects, never bare
    `DocumentExtractionResult` objects -- review finding I: only a
    checkpoint carries the `extraction_schema_hash`/`pdf_sha256` a
    `CalibrationCorpus`'s own bindings must be checked against BEFORE any
    accuracy metric is computed ("storing these bindings but never
    enforcing them turns the fields described as scientific identity
    bindings into passive metadata"). Each checkpoint is verified
    (`checkpoint.verify()`, review finding H) before use, and the
    corpus's `extraction_schema_hash` must match every audited
    checkpoint's own (checked once, since `publish_manifest()` already
    guarantees all checkpoints in one run share identical configuration);
    each corpus case's `pdf_sha256` must match its own document's
    checkpoint. Either mismatch raises `CalibrationBindingError` before
    any tally happens -- an incompatible corpus/run pairing fails
    explicitly, never silently compared.

    Audits (document_id, field_id) pairs present in BOTH `checkpoints`
    and `corpus`. A corpus case for a document that isn't part of this
    run, or a judgment for a field missing from that document's result,
    is counted explicitly (`skipped_missing_document_field_count` /
    `skipped_missing_field_count`) rather than silently absorbed --
    `coverage_complete=False` signals an incomplete audit rather than
    letting it look clean by omission (review finding 5).

    Review finding L: EVERY checkpoint's own `extraction_schema_hash` is
    checked against the corpus (not merely the first one -- "checkpoints
    from different runs and different schemas were accepted together;
    the first happened to match the corpus, so both contributed to a
    perfect accuracy result"), a duplicate `document_id` across
    `checkpoints` is refused explicitly (not silently overwritten in a
    dict), and every checkpoint must share the SAME `run_id` -- checkpoints
    from different runs are never combined into one accuracy report.

    Review finding N-residual: matching schema HASHES is not enough on
    its own -- `corpus.expected_field_ids` must also equal EVERY audited
    checkpoint's own complete result field set (defense in depth,
    independent of whether the corpus was authored via
    `CalibrationCorpus.build()`, which derives `expected_field_ids` from
    the real schema, or hand-constructed with a stale/cherry-picked
    label). "Build a valid checkpoint for a two-field schema, give the
    corpus that exact schema hash but set expected_field_ids=('a',) and
    judge only 'a' -- coverage_complete=True, perfect accuracy" is
    refused here, before any tally."""
    document_ids_seen = [cp.document_id for cp in checkpoints]
    if len(set(document_ids_seen)) != len(document_ids_seen):
        dupes = sorted({d for d in document_ids_seen if document_ids_seen.count(d) > 1})
        raise ValueError(f"duplicate document_id(s) in checkpoints: {dupes}")

    checkpoint_by_id: dict[str, DocumentCheckpoint] = {}
    run_ids_seen: set[str] = set()
    for checkpoint in checkpoints:
        checkpoint.verify()
        run_ids_seen.add(checkpoint.run_id)
        if checkpoint.extraction_schema_hash != corpus.extraction_schema_hash:
            raise CalibrationBindingError(
                f"calibration corpus extraction_schema_hash={corpus.extraction_schema_hash!r} "
                f"(schema {corpus.extraction_schema_id!r}@{corpus.extraction_schema_version!r}) does not "
                f"match checkpoint document_id={checkpoint.document_id!r}'s own extraction_schema_hash="
                f"{checkpoint.extraction_schema_hash!r} (schema "
                f"{checkpoint.extraction_schema_id!r}@{checkpoint.extraction_schema_version!r}) "
                "-- refusing to compute accuracy against an incompatible schema"
            )
        checkpoint_field_ids = {f.field_id for f in checkpoint.result.fields}
        if set(corpus.expected_field_ids) != checkpoint_field_ids:
            raise CalibrationBindingError(
                f"calibration corpus expected_field_ids={sorted(corpus.expected_field_ids)!r} do not "
                f"match checkpoint document_id={checkpoint.document_id!r}'s own complete result "
                f"field set {sorted(checkpoint_field_ids)!r} -- refusing to compute accuracy against "
                "a partial/mismatched field declaration"
            )
        checkpoint_by_id[checkpoint.document_id] = checkpoint

    if len(run_ids_seen) > 1:
        raise CalibrationBindingError(
            f"checkpoints span multiple run_ids ({sorted(run_ids_seen)!r}) -- refusing to compute "
            "accuracy across checkpoints from different runs"
        )

    outcomes_by_field: dict[str, dict[str, int]] = {}
    tally = {outcome: 0 for outcome in _ACCURACY_OUTCOMES}
    value_bearing_correct = 0
    value_bearing_total = 0
    skipped_missing_document_field_count = 0
    skipped_missing_field_count = 0

    expected_field_count = sum(len(case.field_judgments) for case in corpus.cases)

    for case in corpus.cases:
        checkpoint = checkpoint_by_id.get(case.document_id)
        if checkpoint is None:
            skipped_missing_document_field_count += len(case.field_judgments)
            continue
        if checkpoint.pdf_sha256 != case.pdf_sha256:
            raise CalibrationBindingError(
                f"calibration case for document_id={case.document_id!r} declares "
                f"pdf_sha256={case.pdf_sha256!r} but the audited checkpoint's own "
                f"pdf_sha256={checkpoint.pdf_sha256!r} differs -- refusing to compute accuracy "
                "against a substituted PDF"
            )
        field_result_map = {f.field_id: f for f in checkpoint.result.fields}
        for judgment in case.field_judgments:
            system_field = field_result_map.get(judgment.field_id)
            if system_field is None:
                skipped_missing_field_count += 1
                continue
            outcome = _classify_accuracy_outcome(system_field, judgment.correct_status, set(judgment.correct_values))
            tally[outcome] += 1
            field_counts = outcomes_by_field.setdefault(judgment.field_id, {})
            field_counts[outcome] = field_counts.get(outcome, 0) + 1
            if judgment.correct_status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING):
                value_bearing_total += 1
                if outcome == "correct":
                    value_bearing_correct += 1

    audited_field_count = sum(tally.values())
    denom = audited_field_count or 1

    return AuditedAccuracyReport(
        calibration_corpus_id=corpus.corpus_id,
        calibration_corpus_version=corpus.version,
        calibration_corpus_content_hash=corpus.content_hash,
        expected_field_count=expected_field_count,
        audited_document_count=len({c.document_id for c in corpus.cases if c.document_id in checkpoint_by_id}),
        audited_field_count=audited_field_count,
        skipped_missing_document_field_count=skipped_missing_document_field_count,
        skipped_missing_field_count=skipped_missing_field_count,
        coverage_complete=(skipped_missing_document_field_count == 0 and skipped_missing_field_count == 0),
        correct_count=tally["correct"],
        value_mismatch_count=tally["value_mismatch"],
        false_not_reported_count=tally["false_not_reported"],
        over_extraction_count=tally["over_extraction"],
        status_mismatch_count=tally["status_mismatch"],
        value_bearing_judgment_count=value_bearing_total,
        accuracy_rate=tally["correct"] / denom,
        # Only computed over VALUE-BEARING judgments (correct_status in
        # EXTRACTED/CONFLICTING) -- a correct non-value status (e.g.
        # not_reported) must never inflate this rate (review finding 5's
        # adversarial reproduction: "one correctly absent field and zero
        # extracted values ... reports value_correctness_rate=1.0").
        value_correctness_rate=(value_bearing_correct / value_bearing_total) if value_bearing_total else 0.0,
        false_not_reported_rate=tally["false_not_reported"] / denom,
        over_extraction_rate=tally["over_extraction"] / denom,
        outcomes_by_field=outcomes_by_field,
    )


class EvidenceQualityReport(BaseModel):
    """Evidence-level quality -- SEPARATE from `AuditedAccuracyReport`
    (which measures whether a VALUE was correct, not whether its
    evidence substantively supports it or points to the right place).

      - `support_precision`: of every system-returned quotation a human
        actually judged, what fraction substantively support their
        claimed value (review finding B/5: exact-substring verification
        only proves the text EXISTS, never that it supports the claim).
      - `locator_accuracy`: of the same judged quotations, what fraction
        carry a correct page/section attribution.
      - `reference_evidence_recall`: of every piece of REUSABLE reference
        evidence in the (optional) `CalibrationCorpus` -- evidence a
        human reviewer already identified in the publication, independent
        of any system run -- what fraction was actually found (as a
        SUPPORTED quotation) by the audited system output. `None` (never
        `0.0`) when no corpus was supplied, so "no reference evidence to
        check" is never confused with "the system found none of it."

    Every rate carries its own explicit numerator/denominator alongside
    it (review requirement: "explicit numerators, denominators, and
    missing coverage") -- never a bare float a reader has to trust."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audited_document_count: int = Field(ge=0)
    audited_quotation_count: int = Field(ge=0)
    # The precision DENOMINATOR -- review finding J: after correctly
    # excluding review-only spans (`supports_claimed_value is None`)
    # from precision, `audited_quotation_count` (which still includes
    # them) was no longer support_precision's true denominator, so the
    # rate could not be reconstructed from the persisted counts alone.
    # This is that denominator, persisted explicitly:
    # support_precision == supported_quotation_count / support_eligible_quotation_count.
    support_eligible_quotation_count: int = Field(ge=0)
    supported_quotation_count: int = Field(ge=0)
    locator_correct_count: int = Field(ge=0)
    support_precision: float = Field(ge=0.0, le=1.0)
    locator_accuracy: float = Field(ge=0.0, le=1.0)
    reference_evidence_total: Optional[int] = Field(default=None, ge=0)
    reference_evidence_recalled: Optional[int] = Field(default=None, ge=0)
    reference_evidence_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibration_corpus_content_hash: Optional[str] = Field(default=None, min_length=64, max_length=64)


def build_evidence_quality_report(
    audits_with_checkpoints: list[tuple[EvidenceAudit, DocumentCheckpoint]],
    calibration_corpus: Optional[CalibrationCorpus] = None,
    *, run_manifest: Optional[RunManifest] = None,
) -> EvidenceQualityReport:
    """Takes `(audit, checkpoint)` PAIRS, never a bare list of audits --
    review finding F: "direct `EvidenceAudit(...)` construction remains
    public and `build_evidence_quality_report()` explicitly trusts such
    objects without re-verification." Trustworthiness is enforced HERE,
    at the point of use, not merely at an optional `.build()` factory a
    caller could bypass: every pair is re-verified before its judgments
    are allowed to contribute to the report --
      - `checkpoint.verify()` -- review finding H: matching an audit's
        claimed identity against a checkpoint object proves nothing if
        the checkpoint ITSELF is never confirmed to be a genuine,
        untampered artifact (a deserialized checkpoint with a replaced
        `content_hash` still constructs successfully as a plain Pydantic
        object -- this call is what actually catches that);
      - `audit.run_id == checkpoint.run_id`,
        `audit.document_id == checkpoint.document_id`, and
        `audit.checkpoint_content_hash == checkpoint.content_hash`
        (raises `EvidenceAuditIdentityMismatchError` otherwise);
      - when `run_manifest` is supplied (a formally run-bound report):
        `checkpoint.run_id == run_manifest.run_id` and
        `checkpoint.content_hash == run_manifest.checkpoint_hashes[checkpoint.document_id]`
        -- refusing an arbitrary self-consistent checkpoint detached
        from the actually-published run (review finding H's second
        requirement). Pass a `RunManifest` from
        `ExtractionCheckpointStore.load_manifest()` (itself already
        fully re-verified) to get this guarantee; omit it only for an ad
        hoc report built directly from verified checkpoints with no
        published run to bind against;
      - `audit.quotation_judgments` exactly covers
        `checkpoint.result`'s own evidence projection (raises
        `EvidenceAuditCoverageError` otherwise, the SAME check
        `EvidenceAudit.build()` performs at construction time -- applied
        again here because a directly-deserialized audit may never have
        gone through `.build()` at all).
    A non-conforming pair is never silently skipped or partially
    counted -- the whole call fails, the same way a single invalid input
    fails validation elsewhere in this package.

    When `calibration_corpus` is supplied, its `extraction_schema_hash`
    must match every audited checkpoint's own, and each corpus case's
    `pdf_sha256` must match its own document's checkpoint (review finding
    I) -- raises `CalibrationBindingError` before computing recall
    against an incompatible schema or substituted PDF.

    `reference_evidence_recall` matches on the FULL `(document_id,
    field_id, value, normalized excerpt)` relation -- never bare
    excerpt-within-document (review finding F). Note this is MECHANICAL
    exact-excerpt-text recall (after the same whitespace/Unicode
    normalization `evidence.py` uses for quotation matching, never
    fuzzy/semantic matching), not semantic evidence recall.

    Review finding L: `run_manifest`'s OWN self-hash is re-verified here
    (never merely a `run_id`/`checkpoint_hashes` field comparison against
    an unverified object -- a manifest deserialized with a deliberately
    false self-hash is refused before it can bind anything). A duplicate
    `document_id` across `audits_with_checkpoints` is refused explicitly.
    Every audited checkpoint's own schema hash is checked against
    `calibration_corpus` (not merely one arbitrary checkpoint), and
    (when no `run_manifest` was supplied) all checkpoints must share the
    SAME `run_id` -- mixed-run checkpoints are never combined into one
    evidence-quality report."""
    if run_manifest is not None:
        run_manifest.verify_manifest_hash()

    audit_document_ids = [audit.document_id for audit, _ in audits_with_checkpoints]
    if len(set(audit_document_ids)) != len(audit_document_ids):
        dupes = sorted({d for d in audit_document_ids if audit_document_ids.count(d) > 1})
        raise ValueError(f"duplicate document_id(s) across audits_with_checkpoints: {dupes}")

    audits: list[EvidenceAudit] = []
    checkpoint_by_document_id: dict[str, DocumentCheckpoint] = {}
    run_ids_seen: set[str] = set()
    for audit, checkpoint in audits_with_checkpoints:
        checkpoint.verify()
        run_ids_seen.add(checkpoint.run_id)
        if (audit.run_id, audit.document_id, audit.checkpoint_content_hash) != (
            checkpoint.run_id, checkpoint.document_id, checkpoint.content_hash,
        ):
            raise EvidenceAuditIdentityMismatchError(
                f"audit_id={audit.audit_id!r} claims (run_id={audit.run_id!r}, "
                f"document_id={audit.document_id!r}, checkpoint_content_hash={audit.checkpoint_content_hash!r}) "
                f"but the supplied checkpoint is (run_id={checkpoint.run_id!r}, "
                f"document_id={checkpoint.document_id!r}, content_hash={checkpoint.content_hash!r})"
            )
        if run_manifest is not None:
            if checkpoint.run_id != run_manifest.run_id:
                raise EvidenceAuditIdentityMismatchError(
                    f"checkpoint for document_id={checkpoint.document_id!r} has run_id="
                    f"{checkpoint.run_id!r}, which does not match the supplied run_manifest's own "
                    f"run_id={run_manifest.run_id!r}"
                )
            bound_hash = run_manifest.checkpoint_hashes.get(checkpoint.document_id)
            if bound_hash != checkpoint.content_hash:
                raise EvidenceAuditIdentityMismatchError(
                    f"checkpoint for document_id={checkpoint.document_id!r} (content_hash="
                    f"{checkpoint.content_hash!r}) is not the checkpoint bound by the verified run "
                    f"manifest (bound hash={bound_hash!r}) -- refusing to report against a checkpoint "
                    "detached from the published run"
                )
        expected = {
            (field_id, claimed_value, evidence.chunk_id, evidence.excerpt)
            for field_id, claimed_value, evidence in evidence_projection(checkpoint.result)
        }
        provided = {(j.field_id, j.claimed_value, j.chunk_id, j.excerpt) for j in audit.quotation_judgments}
        missing, extra = expected - provided, provided - expected
        if missing or extra:
            raise EvidenceAuditCoverageError(
                f"audit_id={audit.audit_id!r} does not exactly cover its checkpoint's evidence spans -- "
                f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
            )
        if calibration_corpus is not None:
            if checkpoint.extraction_schema_hash != calibration_corpus.extraction_schema_hash:
                raise CalibrationBindingError(
                    f"calibration corpus extraction_schema_hash={calibration_corpus.extraction_schema_hash!r} "
                    f"does not match checkpoint document_id={checkpoint.document_id!r}'s own "
                    f"extraction_schema_hash={checkpoint.extraction_schema_hash!r} -- refusing to "
                    "compute evidence quality against an incompatible schema"
                )
            # Review finding N-residual: matching schema HASHES alone is
            # not enough -- expected_field_ids must also equal this
            # checkpoint's own complete result field set.
            checkpoint_field_ids = {f.field_id for f in checkpoint.result.fields}
            if set(calibration_corpus.expected_field_ids) != checkpoint_field_ids:
                raise CalibrationBindingError(
                    f"calibration corpus expected_field_ids={sorted(calibration_corpus.expected_field_ids)!r} "
                    f"do not match checkpoint document_id={checkpoint.document_id!r}'s own complete "
                    f"result field set {sorted(checkpoint_field_ids)!r} -- refusing to compute "
                    "evidence quality against a partial/mismatched field declaration"
                )
        audits.append(audit)
        checkpoint_by_document_id[audit.document_id] = checkpoint

    if run_manifest is None and len(run_ids_seen) > 1:
        raise EvidenceAuditIdentityMismatchError(
            f"checkpoints span multiple run_ids ({sorted(run_ids_seen)!r}) -- refusing to compute "
            "evidence quality across checkpoints from different runs"
        )

    if calibration_corpus is not None:
        for case in calibration_corpus.cases:
            checkpoint = checkpoint_by_document_id.get(case.document_id)
            if checkpoint is not None and checkpoint.pdf_sha256 != case.pdf_sha256:
                raise CalibrationBindingError(
                    f"calibration case for document_id={case.document_id!r} declares "
                    f"pdf_sha256={case.pdf_sha256!r} but the audited checkpoint's own "
                    f"pdf_sha256={checkpoint.pdf_sha256!r} differs -- refusing to compute evidence "
                    "quality against a substituted PDF"
                )

    claim_bearing_judgments = [
        j for a in audits for j in a.quotation_judgments if j.supports_claimed_value is not None
    ]
    all_judgments = [j for a in audits for j in a.quotation_judgments]
    audited_quotation_count = len(all_judgments)
    support_eligible_count = len(claim_bearing_judgments)
    supported_count = sum(1 for j in claim_bearing_judgments if j.supports_claimed_value)
    locator_correct_count = sum(1 for j in all_judgments if j.locator_correct)
    precision_denom = support_eligible_count or 1
    locator_denom = audited_quotation_count or 1

    reference_total: Optional[int] = None
    reference_recalled: Optional[int] = None
    reference_recall: Optional[float] = None

    if calibration_corpus is not None:
        # (document_id, field_id, claimed_value, normalized_excerpt) keys
        # actually SUPPORTED by a judged quotation -- the full relation,
        # never bare excerpt text alone.
        supported_keys: set[tuple[str, str, str, str]] = {
            (audit.document_id, j.field_id, j.claimed_value, normalize_for_matching(j.excerpt))
            for audit in audits for j in audit.quotation_judgments
            if j.supports_claimed_value is True and j.claimed_value
        }

        reference_total = 0
        reference_recalled = 0
        for case in calibration_corpus.cases:
            for judgment in case.field_judgments:
                for reference in judgment.reference_evidence:
                    reference_total += 1
                    key = (
                        case.document_id, judgment.field_id, reference.supports_value,
                        normalize_for_matching(reference.excerpt),
                    )
                    if key in supported_keys:
                        reference_recalled += 1
        reference_recall = (reference_recalled / reference_total) if reference_total else 0.0

    return EvidenceQualityReport(
        audited_document_count=len({a.document_id for a in audits}),
        audited_quotation_count=audited_quotation_count,
        support_eligible_quotation_count=support_eligible_count,
        supported_quotation_count=supported_count,
        locator_correct_count=locator_correct_count,
        support_precision=supported_count / precision_denom,
        locator_accuracy=locator_correct_count / locator_denom,
        reference_evidence_total=reference_total,
        reference_evidence_recalled=reference_recalled,
        reference_evidence_recall=reference_recall,
        calibration_corpus_content_hash=calibration_corpus.content_hash if calibration_corpus is not None else None,
    )


class FullExtractionReport(BaseModel):
    """The top-level report artifact. `audited_accuracy` /
    `evidence_quality` are `None` when no calibration corpus / no
    evidence audits were supplied -- callers must never read `coverage`
    alone and present it as if it said anything about correctness.

    `manifest_hash` is set ONLY by `build_full_report()` (the
    formally run-bound path, built from a `checkpoint.VerifiedRunBundle`)
    -- its presence is itself the signal that `run_id`/`coverage`/
    `audited_accuracy`/`evidence_quality` are all projections of ONE
    verified manifest, never independently-supplied objects that merely
    happen to agree (review finding L). It is always `None` on a report
    from `build_ad_hoc_extraction_report()`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    coverage: CoverageReport
    audited_accuracy: Optional[AuditedAccuracyReport] = None
    evidence_quality: Optional[EvidenceQualityReport] = None
    manifest_hash: Optional[str] = Field(default=None, min_length=64, max_length=64)


def build_full_report(
    *, bundle: VerifiedRunBundle, calibration_corpus: Optional[CalibrationCorpus] = None,
    evidence_audits: Optional[list[EvidenceAudit]] = None,
) -> FullExtractionReport:
    """The ONLY supported entry point for a FORMALLY RUN-BOUND report
    (review finding L). `bundle` (a `checkpoint.VerifiedRunBundle`, built
    via `VerifiedRunBundle.from_store(store)`) is the SINGLE source of
    `run_id`, `results`, and `checkpoints` -- all three are projections
    of the ONE manifest `bundle` was derived from, so there is no
    parameter through which a caller could cross-join an unrelated
    `run_id`, results from one document set, and checkpoints from
    another (review finding L's central adversarial reproduction: "A
    report with run_id='WRONG-RUN', coverage from document A, and
    accuracy from document B constructed successfully").

    `evidence_audits` is a bare list of `EvidenceAudit` (never
    independent `(audit, checkpoint)` pairs) -- each audit's own
    `document_id` is used to look up ITS checkpoint from `bundle` itself,
    so a caller cannot pair an audit with an unrelated checkpoint either.
    Duplicate `document_id`s among `evidence_audits`, or an audit for a
    `document_id` not present in `bundle`, are refused explicitly.

    Review finding L-residual: `bundle.verify()` is called FIRST, before
    anything else -- a bundle that was directly (forged) constructed
    from an inconsistent manifest/checkpoint pairing, or one whose
    `checkpoints_by_id` was mutated after `VerifiedRunBundle.from_store()`
    originally verified it, is rejected here at the point of use, not
    merely trusted because `from_store()` exists as an option."""
    bundle.verify()

    if evidence_audits is not None:
        audit_document_ids = [a.document_id for a in evidence_audits]
        if len(set(audit_document_ids)) != len(audit_document_ids):
            dupes = sorted({d for d in audit_document_ids if audit_document_ids.count(d) > 1})
            raise ValueError(f"duplicate document_id(s) in evidence_audits: {dupes}")
        for document_id in audit_document_ids:
            if document_id not in bundle.checkpoints_by_id:
                raise ValueError(
                    f"evidence audit for document_id={document_id!r} is not part of this bundle's "
                    f"verified manifest (run_id={bundle.run_id!r})"
                )

    coverage = build_coverage_report(bundle.results)
    audited = (
        build_audited_accuracy_report(bundle.checkpoints, calibration_corpus)
        if calibration_corpus is not None else None
    )
    evidence_quality = None
    if evidence_audits is not None:
        audits_with_checkpoints = [(audit, bundle.checkpoints_by_id[audit.document_id]) for audit in evidence_audits]
        evidence_quality = build_evidence_quality_report(
            audits_with_checkpoints, calibration_corpus, run_manifest=bundle.manifest,
        )
    return FullExtractionReport(
        run_id=bundle.run_id, coverage=coverage, audited_accuracy=audited, evidence_quality=evidence_quality,
        manifest_hash=bundle.manifest.manifest_hash,
    )


def build_ad_hoc_extraction_report(
    *, run_id: str, results: list[DocumentExtractionResult],
    checkpoints: Optional[list[DocumentCheckpoint]] = None,
    calibration_corpus: Optional[CalibrationCorpus] = None,
    evidence_audits_with_checkpoints: Optional[list[tuple[EvidenceAudit, DocumentCheckpoint]]] = None,
    run_manifest: Optional[RunManifest] = None,
) -> FullExtractionReport:
    """An AD HOC report -- deliberately named so it cannot be mistaken
    for a formally run-bound one (review finding L: "Keep a clearly
    named ad-hoc report path only if it cannot be mistaken for a
    formally run-bound report"). `run_id`, `results`, `checkpoints`, and
    `evidence_audits_with_checkpoints` are INDEPENDENT parameters this
    function does NOT cross-verify against each other -- each individual
    section builder (`build_audited_accuracy_report`,
    `build_evidence_quality_report`) still enforces its OWN internal
    checks (schema/PDF binding, checkpoint self-hash, single-run
    coherence within that section), but nothing here proves `results`
    actually came from `run_id`, or that `checkpoints` and
    `evidence_audits_with_checkpoints` describe the same run as each
    other. `manifest_hash` is always `None` on the result. Use this ONLY
    for genuinely ad hoc/exploratory comparisons with no published
    manifest to bind against; use `build_full_report(bundle=...)`
    whenever a real published run exists.

    `checkpoints` is required whenever `calibration_corpus` is given
    (review finding I: `build_audited_accuracy_report()` needs verified
    checkpoints, not bare results, to enforce the corpus's schema/PDF
    bindings). `run_manifest`, when supplied, is forwarded to
    `build_evidence_quality_report()` for its run-bound checkpoint-hash
    check (review finding H)."""
    coverage = build_coverage_report(results)
    if calibration_corpus is not None:
        if checkpoints is None:
            raise ValueError(
                "calibration_corpus requires checkpoints -- build_audited_accuracy_report() needs "
                "verified DocumentCheckpoint objects to enforce the corpus's schema/PDF bindings"
            )
        audited = build_audited_accuracy_report(checkpoints, calibration_corpus)
    else:
        audited = None
    evidence_quality = (
        build_evidence_quality_report(evidence_audits_with_checkpoints, calibration_corpus, run_manifest=run_manifest)
        if evidence_audits_with_checkpoints is not None else None
    )
    return FullExtractionReport(run_id=run_id, coverage=coverage, audited_accuracy=audited, evidence_quality=evidence_quality)
