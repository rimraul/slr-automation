"""Workstream C Checkpoint 3 -- offline enforcement contracts for the
Berg post-2023 evidence-update wave's incremental-update driver. No
network access, no live search, no paid LLM call: every function here
either runs a purely local computation (PDF parsing, batch counting) or
inspects/constructs a client object without issuing a request.

Checkpoint-2 review named two forward-looking Checkpoint-3 contracts,
both implemented here:

- "enforce the eight-batch limit before any paid extraction call, after
  local PDF parsing; fail and preserve the checkpoint when exceeded.
  Never truncate a document to satisfy the bound" ->
  ``enforce_extraction_batch_cap()`` (this module's thin, estimator-bound
  wrapper) delegates to ``extraction.batch_cap`` -- the SAME module
  ``extraction/engine.py``'s ``extract_document()``/``run_extraction()``
  use internally (via their additive ``max_batches``/
  ``expected_artifacts_manifest_digest`` parameters) for their own real,
  billed batch loop. There is exactly ONE parse+batch computation for a
  given document, never a second independent one that could diverge from
  what gets billed (Checkpoint-3 review finding: "check-versus-use gap").
  ``run_extraction()`` reserves/persists a durable ``BlockedDocumentStore``
  record BEFORE any batch of an over-cap document could be billed, and
  excludes that document from its published ``RunManifest`` entirely
  (Checkpoint-3 review finding: "does not preserve a checkpoint").
  ``expected_artifacts_manifest_digest`` is mandatory throughout this
  path -- Docling's mutable default model cache is never silently used
  (Checkpoint-3 review finding: "may ... potentially fetch artifacts").
- "bind the driver's resolution model, output ceiling, and retry policy
  to the estimator constants, or refuse execution when they differ" ->
  ``verify_resolution_bounds()``.

Checkpoint-2 approval (Round 6) added a further contract: "Pin or
validate screening/QC retry behavior." ``nodes/screening.py``/
``nodes/screening_qc.py`` gained an additive, backward-compatible
``client_factory`` keyword-only parameter (defaulting to their existing
unpatched behavior, so Checkpoint 2's approved
``SCREENING_MAX_ATTEMPTS``/``SCREENING_QC_MAX_ATTEMPTS`` figures remain
valid for every other caller). This module's
``pinned_and_validated_openai_client_factory()`` builds the factory the
Checkpoint-3 driver passes instead: it PINS ``max_retries`` explicitly
(never relying on the installed SDK's own default persisting) and
VALIDATES the constructed client's actual ``max_retries`` against the
estimator constant it is priced against, refusing execution on any
mismatch -- both halves of "pin or validate", not just one.
``run_update_wave_screening()`` (Checkpoint-3 review finding: "retry-bound
factories exist but are not wired into any graph") is the actual, tested
production caller that passes ``screening_client_factory``/
``screening_qc_client_factory`` into ``screening_node``/
``screening_qc_node`` -- this driver never uses LangGraph's
``builder.add_node("screening", screening_node)`` bare default wiring.

"Refuse execution when any bound differs" is the general contract every
function below follows: a mismatch always raises (never logs-and-
continues, never silently substitutes a corrected value).
"""

from __future__ import annotations

import json
import contextlib
import inspect
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, model_validator

from slr_pipeline import credentials
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key
from slr_pipeline.doi import normalise_doi, normalise_title
from slr_pipeline.evaluation.bundle_artifacts import VerifiedReproductionArtifactSet
from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.evaluation.update_cost_estimate import (
    ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE,
    RESOLUTION_SUGGESTION_MAX_ATTEMPTS,
    RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION,
    RESOLUTION_SUGGESTION_MODEL_ASSUMPTION,
    SCREENING_MAX_ATTEMPTS,
    SCREENING_QC_MAX_ATTEMPTS,
    CENSUS_VS_SAMPLING_ESTIMATOR_VERSION,
    CensusVsSamplingDecision,
    SampleBudgetExceededError as SampleBudgetExceededError,
    decide_census_vs_sampling,
    draw_sample_records,
)
from slr_pipeline.evaluation.update_lineage import UpdateLineage
from slr_pipeline.extraction.resolution_suggestions import _fsync_dir
from slr_pipeline.extraction.batch_cap import ExtractionBatchCapExceededError as ExtractionBatchCapExceededError
from slr_pipeline.extraction.batch_cap import enforce_extraction_batch_cap as _enforce_extraction_batch_cap
from slr_pipeline.extraction.batching import DEFAULT_MAX_BATCH_CHARS, ExtractionBatch
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedDocument
from slr_pipeline.ai_traceability import AIUseEventStore, kernel_lock
from slr_pipeline.nodes.screening import screening_node
from slr_pipeline.nodes.screening_qc import SecondRaterResultStore, screening_qc_node
from slr_pipeline.nodes.search import search_node
from slr_pipeline.search_backends.base import SearchCheckpointHooks, SearchPageCheckpoint
from slr_pipeline.state import DatabaseSearchLog, Record, SearchLog, initial_state

_SAFE_DATABASE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def enforce_extraction_batch_cap(
    pdf_path: str | Path,
    *,
    expected_artifacts_manifest_digest: str,
    parser_options: Optional[DoclingParserOptions] = None,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
    max_batches: int = ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE,
) -> tuple[ParsedDocument, tuple[ExtractionBatch, ...]]:
    """Estimator-bound wrapper over ``extraction.batch_cap.
    enforce_extraction_batch_cap`` -- defaults ``max_batches`` to the
    estimator's own
    ``ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE``,
    but is otherwise the exact same single-parse-single-batch primitive
    ``extraction/engine.py`` uses for its real, billed batch loop -- never
    a second, independent parse that could diverge from what gets billed."""
    return _enforce_extraction_batch_cap(
        pdf_path, expected_artifacts_manifest_digest=expected_artifacts_manifest_digest,
        max_batches=max_batches, parser_options=parser_options, max_batch_chars=max_batch_chars,
    )


class RetryPolicyMismatchError(Exception):
    """Raised when a constructed client's actual retry policy does not
    match the ``max_attempts`` the cost estimator prices for that stage."""


class ResolutionBoundMismatchError(Exception):
    """Raised when the resolution-suggestion pass's model, output-token
    ceiling, or retry policy differs from
    ``RESOLUTION_SUGGESTION_MODEL_ASSUMPTION``/
    ``RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION``/
    ``RESOLUTION_SUGGESTION_MAX_ATTEMPTS``."""


def verify_client_retry_policy(client: Any, *, expected_max_attempts: int, stage: str) -> None:
    """Raises ``RetryPolicyMismatchError`` unless ``client.max_retries``
    (the ``openai`` SDK client's own real, introspectable attribute --
    confirmed to reflect the value passed at construction) implies exactly
    ``expected_max_attempts`` total attempts (``1 + max_retries``)."""
    actual_max_attempts = 1 + client.max_retries
    if actual_max_attempts != expected_max_attempts:
        raise RetryPolicyMismatchError(
            f"{stage}: client configured for {actual_max_attempts} max attempt(s) "
            f"(max_retries={client.max_retries}), but the cost estimator prices "
            f"{expected_max_attempts} for this stage. Refusing execution."
        )


def pinned_and_validated_openai_client_factory(
    *, max_retries: int, expected_max_attempts: int, stage: str,
) -> Callable[[], Any]:
    """Returns a zero-argument factory that PINS ``max_retries`` explicitly
    (never relying on the installed SDK's own default persisting) and
    VALIDATES the resulting client against ``expected_max_attempts``
    before returning it, raising ``RetryPolicyMismatchError`` on
    mismatch -- satisfies "pin or validate" as both, not either."""

    def factory() -> Any:
        from openai import OpenAI

        client = OpenAI(api_key=resolve_openai_api_key(), max_retries=max_retries)
        verify_client_retry_policy(client, expected_max_attempts=expected_max_attempts, stage=stage)
        return client

    return factory


# Bound 1:1 to the exact constants Checkpoint 2's estimator prices for
# these two stages -- ``max_retries=expected_max_attempts - 1`` by
# construction, so pinning and pricing can never silently drift apart.
screening_client_factory: Callable[[], Any] = pinned_and_validated_openai_client_factory(
    max_retries=SCREENING_MAX_ATTEMPTS - 1, expected_max_attempts=SCREENING_MAX_ATTEMPTS, stage="screening",
)
screening_qc_client_factory: Callable[[], Any] = pinned_and_validated_openai_client_factory(
    max_retries=SCREENING_QC_MAX_ATTEMPTS - 1, expected_max_attempts=SCREENING_QC_MAX_ATTEMPTS, stage="screening_qc",
)


def verify_resolution_bounds(
    *,
    model: str,
    max_completion_tokens: int,
    client_factory: Callable[[], Any],
) -> None:
    """Refuses execution (raises ``ResolutionBoundMismatchError``) unless
    ``model``/``max_completion_tokens`` -- the exact arguments the driver
    is about to pass to ``resolution_runtime.run_paid_suggestion_pass()``
    -- match ``RESOLUTION_SUGGESTION_MODEL_ASSUMPTION``/
    ``RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION``, and unless
    ``client_factory()``'s constructed client's retry policy matches
    ``RESOLUTION_SUGGESTION_MAX_ATTEMPTS``."""
    if model != RESOLUTION_SUGGESTION_MODEL_ASSUMPTION:
        raise ResolutionBoundMismatchError(
            f"resolution_suggestions: model {model!r} does not match the cost estimator's "
            f"RESOLUTION_SUGGESTION_MODEL_ASSUMPTION {RESOLUTION_SUGGESTION_MODEL_ASSUMPTION!r}. "
            "Refusing execution."
        )
    if max_completion_tokens != RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION:
        raise ResolutionBoundMismatchError(
            f"resolution_suggestions: max_completion_tokens={max_completion_tokens} does not match "
            "the cost estimator's RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION="
            f"{RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION}. Refusing execution."
        )
    client = client_factory()
    try:
        verify_client_retry_policy(
            client, expected_max_attempts=RESOLUTION_SUGGESTION_MAX_ATTEMPTS, stage="resolution_suggestions",
        )
    except RetryPolicyMismatchError as exc:
        raise ResolutionBoundMismatchError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Cross-parent deduplication (plan 0.3/0.8): a candidate "new" record is
# genuinely new only if it matches NEITHER another new-side candidate NOR
# any record already in the parent's own raw pool. Reuses the exact
# identity-matching primitives ``nodes/dedup.py``'s own
# ``_dedup_doi_only()`` fallback uses (``normalise_doi``/
# ``normalise_title``, same exact-equality-only semantics, same >10-char
# floor on title matching) rather than reimplementing duplicate-detection
# logic -- but does NOT reuse ``_dedup_doi_only()`` itself, since that
# function mutates its input records' ``is_duplicate`` in place, which
# here would incorrectly mark the PARENT's own frozen, already-verified
# raw_records. This function never mutates either input list.
# ---------------------------------------------------------------------------


def classify_new_records_against_parent(
    *, parent_raw_records: Sequence[Record], new_raw_records: Sequence[Record],
) -> tuple[Record, ...]:
    """Returns the subset of ``new_raw_records`` that are genuinely new:
    not an exact-DOI or exact-title match of any ``parent_raw_records``
    entry, and not a duplicate of an earlier (order-preserved) entry
    within ``new_raw_records`` itself. Non-mutating; a dropped
    ``new_raw_records`` entry is simply omitted from the result, never
    flagged via a shared mutable field."""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    for r in parent_raw_records:
        if r.doi:
            doi_norm = normalise_doi(r.doi)
            if doi_norm:
                seen_dois.add(doi_norm)
        title_norm = normalise_title(r.title)
        if title_norm and len(title_norm) > 10:
            seen_titles.add(title_norm)

    survivors: list[Record] = []
    for r in new_raw_records:
        doi_norm = normalise_doi(r.doi) if r.doi else None
        if doi_norm and doi_norm in seen_dois:
            continue
        title_norm = normalise_title(r.title)
        title_is_matchable = bool(title_norm) and len(title_norm) > 10
        if title_is_matchable and title_norm in seen_titles:
            continue
        if doi_norm:
            seen_dois.add(doi_norm)
        if title_is_matchable:
            seen_titles.add(title_norm)
        survivors.append(r)
    return tuple(survivors)


def compute_cross_source_membership(
    *, parent_raw_records: Sequence[Record], new_raw_records: Sequence[Record],
) -> Dict[str, Tuple[Dict[str, str], ...]]:
    """Maps a record id (parent-origin or new-wave-survivor) to the extra
    ``{"source", "source_id"}`` memberships discovered for it during the
    same exact-DOI/exact-title matching ``classify_new_records_against_parent``
    performs -- e.g. a parent record also retrieved by the new wave's
    Scopus search, or a new-wave survivor also retrieved by a second
    database within the same wave. Mirrors that function's matching
    order (DOI match wins first, else title match) so the two functions
    always agree on which records are duplicates of which.

    Returned as a standalone side-channel report, deliberately never
    embedded into a ``Record`` field or the ``StageCorpus``/
    ``VerifiedReproductionArtifactSet`` returned by
    ``assemble_update_corpus``: every ``Record`` field feeds
    ``StageCorpus``'s hash-pinned ``raw_records_hash``/
    ``deduplicated_records_hash`` (``stage_corpus.py``'s ``_records_hash``
    hashes the FULL ``model_dump``), and the real, already-persisted
    frozen Berg parent artifact's hashes were computed under today's
    ``Record`` schema -- adding a field to ``Record`` (even one that
    defaults to empty) changes every record's serialised content and
    breaks that parent's hash verification on load. (Revision 5 point 5:
    dedup must preserve every database membership for a record found in
    more than one source, not collapse to a single winning source --
    satisfied here via this separate report rather than a schema
    change.)"""
    seen_dois: Dict[str, str] = {}
    seen_titles: Dict[str, str] = {}
    membership: Dict[str, List[Dict[str, str]]] = {}

    def _entry(r: Record) -> Dict[str, str]:
        return {"source": r.source, "source_id": r.source_id}

    def _record_membership(owner_id: str, r: Record) -> None:
        bucket = membership.setdefault(owner_id, [])
        entry = _entry(r)
        if entry not in bucket:
            bucket.append(entry)

    for r in parent_raw_records:
        if r.doi:
            doi_norm = normalise_doi(r.doi)
            if doi_norm:
                seen_dois.setdefault(doi_norm, r.id)
        title_norm = normalise_title(r.title)
        if title_norm and len(title_norm) > 10:
            seen_titles.setdefault(title_norm, r.id)

    for r in new_raw_records:
        doi_norm = normalise_doi(r.doi) if r.doi else None
        title_norm = normalise_title(r.title)
        title_is_matchable = bool(title_norm) and len(title_norm) > 10

        owner_id: Optional[str] = None
        if doi_norm and doi_norm in seen_dois:
            owner_id = seen_dois[doi_norm]
        elif title_is_matchable and title_norm in seen_titles:
            owner_id = seen_titles[title_norm]

        if owner_id is not None:
            _record_membership(owner_id, r)
            continue

        if doi_norm:
            seen_dois[doi_norm] = r.id
        if title_is_matchable:
            seen_titles[title_norm] = r.id

    return {rid: tuple(entries) for rid, entries in membership.items()}


# ---------------------------------------------------------------------------
# Update-wave corpus assembly: parent's 483 deduplicated_records (origin=
# parent, all re-screened under the broader protocol) + genuinely-new
# records (origin=new, first_screen), bound to an UpdateLineage ledger.
# ---------------------------------------------------------------------------


class UpdateWaveQCConfigError(Exception):
    """Update-wave screening QC was enabled with a null
    ``screening_qc.calibration_seed`` -- review finding 4: an unseeded
    sample could silently differ across a restart, and this driver's
    restart-safety guarantee (crash-safe reservation, resumable per-record
    caching) depends on a pinned, reproducible sample. Never fall back to
    an unseeded draw; refuse execution instead."""


class UpdateCorpusAssemblyError(Exception):
    """Raised when the assembled corpus would violate the "each of the
    parent's deduplicated_records is processed exactly once" contract --
    e.g. a duplicate id within the parent set itself, or a new-origin
    record whose id collides with a parent-origin one."""


def assemble_update_corpus(
    *,
    parent: VerifiedReproductionArtifactSet,
    update_spec: ReproductionSpec,
    review_config: ReviewConfig,
    new_raw_records: Sequence[Record],
    review_id: str,
) -> tuple[tuple[Record, ...], UpdateLineage, Dict[str, Tuple[Dict[str, str], ...]]]:
    """Cross-parent-deduplicates ``new_raw_records`` against
    ``parent.stage_corpus.raw_records`` (0.3: "deduplicate new hits
    against the parent's raw_records/deduplicated_records" -- the raw
    pool is the superset, so this also covers everything already
    reachable via ``deduplicated_records``), then builds the combined
    screening corpus (every one of ``parent.stage_corpus.
    deduplicated_records`` once, plus every surviving new record once),
    its binding ``UpdateLineage`` ledger, and (review finding 2) the
    cross-source dedup-membership map from ``compute_cross_source_
    membership()`` -- every canonical record id's extra database
    memberships, so production dedup no longer discards this alongside
    the winning record it already keeps.

    Review finding 6: this function previously accepted caller-supplied
    records AND caller-supplied hashes for those records, calling
    ``UpdateLineage.build()`` directly -- nothing verified that the
    supplied hashes actually described the supplied records, or that the
    "parent" was ever a real, self-consistent, verified reproduction
    artifact set at all. Requiring a ``VerifiedReproductionArtifactSet``
    (only producible via ``bundle_artifacts.verify_reproduction_artifact_set()``),
    a ``ReproductionSpec``, and a ``ReviewConfig``, and deriving every
    record/hash internally via ``UpdateLineage.from_verified_parent()``,
    closes that gap: a lineage can only ever be built from an artifact
    set that has already proven its own internal consistency.

    Raises ``UpdateCorpusAssemblyError`` if the parent's own
    ``deduplicated_records`` contain a duplicate id, if a genuinely-new
    record's id collides with a parent-origin id, or if ``update_spec``/
    ``review_config`` do not bind to this ``parent``/``review_id`` --
    the "exactly once" contract must hold before any screening call is
    ever made."""
    parent_deduplicated_records = parent.stage_corpus.deduplicated_records
    parent_raw_records = parent.stage_corpus.raw_records

    new_survivors = classify_new_records_against_parent(
        parent_raw_records=parent_raw_records, new_raw_records=new_raw_records,
    )
    new_ids = tuple(r.id for r in new_survivors)
    overlap = {r.id for r in parent_deduplicated_records} & set(new_ids)
    if overlap:
        raise UpdateCorpusAssemblyError(
            f"new-origin record id(s) collide with parent-origin id(s): {sorted(overlap)}"
        )

    try:
        lineage = UpdateLineage.from_verified_parent(
            review_id=review_id, parent=parent, update_spec=update_spec, review_config=review_config,
            new_record_ids=new_ids,
        )
    except ValueError as exc:
        raise UpdateCorpusAssemblyError(str(exc)) from exc

    dedup_membership = compute_cross_source_membership(
        parent_raw_records=parent_raw_records, new_raw_records=new_raw_records,
    )
    combined_records = tuple(parent_deduplicated_records) + tuple(new_survivors)
    return combined_records, lineage, dedup_membership


def true_deduplicated_volume(
    *, parent_deduplicated_records: Sequence[Record], new_raw_records: Sequence[Record], parent_raw_records: Sequence[Record],
) -> int:
    """The real post-dedup record count for the update wave: every one of
    the parent's already-deduplicated records, plus every genuinely-new
    record surviving both cross-parent and within-wave deduplication.
    This is the volume the census-vs-sampling decision (Revision 5, plan
    section 0.4) must be made against -- never the raw per-database hit
    counts or a capped/ranked subset."""
    new_survivors = classify_new_records_against_parent(
        parent_raw_records=parent_raw_records, new_raw_records=new_raw_records,
    )
    return len(parent_deduplicated_records) + len(new_survivors)


# ---------------------------------------------------------------------------
# Screening wiring (Checkpoint-3 review finding: "retry-bound factories
# exist but are not wired into any graph"). This driver calls
# ``screening_node``/``screening_qc_node`` DIRECTLY with the pinned/
# validated factories -- it does not register them into a LangGraph
# ``StateGraph`` at all, so there is no bare-default add_node() call site
# that could silently ignore them.
# ---------------------------------------------------------------------------


def run_update_wave_screening(
    *, cfg: ReviewConfig, combined_records: Sequence[Record], review_run_id: str,
) -> Dict[str, Any]:
    """Screens ``combined_records`` (parent-origin + new-origin, already
    assembled by ``assemble_update_corpus``) using the REAL
    ``screening_node``/``screening_qc_node``, with the pinned/validated
    ``screening_client_factory``/``screening_qc_client_factory``.

    Restart-durability comes from ``screening_node``'s own existing
    ``ScreeningBatchResultStore`` get-or-compute checkpoint (keyed by a
    content fingerprint of each batch's record ids + criteria + model +
    policy): calling this again with the SAME ``combined_records``/
    ``cfg``/``review_run_id`` after an interruption reuses every
    already-computed batch's persisted result rather than re-screening or
    re-billing it -- no new checkpoint mechanism is needed here.

    Review finding 4: when screening QC is enabled, ``cfg.screening_qc.
    calibration_seed`` must be non-null (``UpdateWaveQCConfigError``
    otherwise), and QC's IRR sample/second-rater calls are routed through
    a ``SecondRaterResultStore`` scoped to this update wave's own
    ``output_dir``/``review_run_id`` -- giving the same crash-safe
    reservation -> paid-call -> durable-publication restart-safety that
    ``screening_node`` already has, without touching any other
    ``screening_qc_node`` caller's default (unseeded, uncached) behavior."""
    if cfg.screening_qc.enabled and cfg.screening_qc.calibration_seed is None:
        raise UpdateWaveQCConfigError(
            "update-wave screening QC is enabled but cfg.screening_qc.calibration_seed is None -- "
            "a non-null seed is required so a restart reserves the SAME IRR sample rather than "
            "silently drawing a different one"
        )
    state: Dict[str, Any] = dict(initial_state(cfg.to_dict()))
    state["deduplicated_records"] = [r.to_dict() for r in combined_records]
    state["review_run_id"] = review_run_id
    screening_result = screening_node(state, client_factory=screening_client_factory)
    state.update(screening_result)
    if not cfg.screening_qc.enabled:
        return state
    # Review finding 5: `screening_node()` already set `screened_records_full`
    # to the COMPLETE screened population (include + exclude + borderline) --
    # overwriting it with `included_records` here fed screening_qc_node()
    # only the included subset, silently narrowing its sampling population.
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    result_store = SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id, event_store=event_store,
    )
    qc_result = screening_qc_node(state, client_factory=screening_qc_client_factory, result_store=result_store)
    state.update(qc_result)
    return state


# ---------------------------------------------------------------------------
# Architectural gap (Checkpoint 3 review): nothing durably binds
# ``assemble_update_corpus()``'s assembled record ids/lineage hash to the
# corpus ``run_update_wave_screening()`` actually screens -- a caller could
# reassemble a DIFFERENT corpus (parent moved on, spec/config edited,
# new_raw_records changed) under the SAME review_run_id and silently screen
# it as if it were a resume of the original assembly. UpdateWaveAssemblyManifest
# closes this: assembled ids + their content hash + the lineage's own
# records_hash are reserved atomically BEFORE screening ever starts, and a
# resumed run must recompute an IDENTICAL manifest or fail loudly.
# ---------------------------------------------------------------------------

class UpdateWaveAssemblyIntegrityError(RuntimeError):
    """A persisted ``UpdateWaveAssemblyManifest`` failed self-verification,
    or a resumed assembly no longer matches the one originally reserved
    for this ``review_run_id`` -- never silently re-assemble or screen a
    corpus that drifted from what was durably recorded."""


_SAFE_UPDATE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_UPDATE_WAVE_ASSEMBLY_MANIFEST_SCHEMA_VERSION = "update_wave_assembly_manifest_v1"


class UpdateWaveAssemblyManifest(BaseModel):
    """The durable binding between one ``review_run_id`` and the exact
    assembled corpus (ordered record ids + a content hash over them) and
    the COMPLETE ``UpdateLineage`` (``lineage_hash`` -- a whole-object
    hash, not just ``records_hash``, so it also transitively covers
    ``parent_reproduction_spec_hash``/``parent_gold_standard_hash``/
    ``update_reproduction_spec_hash``/``review_config_hash``, none of
    which a records-only hash would catch drifting) that
    ``assemble_update_corpus()`` produced for it -- reserved once,
    verified (never silently recomputed) on every resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["update_wave_assembly_manifest_v1"] = _UPDATE_WAVE_ASSEMBLY_MANIFEST_SCHEMA_VERSION
    review_run_id: str
    review_id: str
    lineage_hash: str
    combined_record_ids: Tuple[str, ...]
    combined_records_hash: str
    # Review finding 2: compute_cross_source_membership() was computed only
    # by tests -- production deduplication discarded its output. Persisting
    # it here (rather than as a side file) means it is automatically covered
    # by manifest_hash below (self.model_dump() includes every field), so a
    # tampered or dropped membership entry is detected exactly like a
    # tampered combined_record_ids/combined_records_hash would be.
    dedup_membership: Dict[str, Tuple[Dict[str, str], ...]]
    manifest_hash: str

    def compute_manifest_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "manifest_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_manifest_hash()
        if self.manifest_hash != expected:
            raise UpdateWaveAssemblyIntegrityError(
                f"UpdateWaveAssemblyManifest for review_run_id={self.review_run_id!r} failed manifest_hash "
                f"self-verification: stored {self.manifest_hash!r} != recomputed {expected!r} -- file "
                "corrupted or tampered"
            )

    def fingerprint(self) -> Tuple[Any, ...]:
        return (
            self.review_id, self.lineage_hash, self.combined_record_ids, self.combined_records_hash,
            self.dedup_membership,
        )


def compute_lineage_hash(lineage: UpdateLineage) -> str:
    """Whole-object hash over the COMPLETE ``UpdateLineage``, not just its
    ``records_hash`` -- covers every field the model itself already binds
    and self-verifies (parent/spec/config hashes included)."""
    return sha256_hex(lineage.model_dump(mode="json"))


def _build_update_wave_assembly_manifest(
    *,
    review_run_id: str,
    review_id: str,
    combined_records: Sequence[Record],
    lineage: UpdateLineage,
    dedup_membership: Dict[str, Tuple[Dict[str, str], ...]],
) -> UpdateWaveAssemblyManifest:
    combined_record_ids = tuple(r.id for r in combined_records)
    combined_records_hash = sha256_hex([r.to_dict() for r in combined_records])
    base = dict(
        schema_version=_UPDATE_WAVE_ASSEMBLY_MANIFEST_SCHEMA_VERSION, review_run_id=review_run_id,
        review_id=review_id, lineage_hash=compute_lineage_hash(lineage),
        combined_record_ids=combined_record_ids, combined_records_hash=combined_records_hash,
        dedup_membership=dedup_membership,
    )
    manifest_hash = UpdateWaveAssemblyManifest.model_construct(**base, manifest_hash="").compute_manifest_hash()
    return UpdateWaveAssemblyManifest.model_validate(dict(base, manifest_hash=manifest_hash))


class UpdateWaveAssemblyManifestStore:
    """Restart-safe home for one update wave's assembly manifest -- same
    safe-token-plus-resolved-containment and atomic-write-then-verify
    discipline as every other Stage 3 store in this codebase."""

    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        if not _SAFE_UPDATE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        manifests_root = Path(output_dir) / "update_wave_assembly_manifests"
        run_dir = manifests_root
        if not run_dir.resolve().is_relative_to(manifests_root.resolve()):
            raise ValueError(
                f"review_run_id {review_run_id!r} resolves outside the update_wave_assembly_manifests "
                "root -- refusing to construct store"
            )
        self.review_run_id = review_run_id
        self.manifests_root = manifests_root

    def _path(self) -> Path:
        return self.manifests_root / f"{self.review_run_id}.json"

    def _lineage_path(self) -> Path:
        return self.manifests_root / f"{self.review_run_id}.lineage.json"

    def _verify_or_persist_lineage(self, *, manifest: UpdateWaveAssemblyManifest, lineage: UpdateLineage) -> None:
        """Review finding 4: durably persists the COMPLETE ``UpdateLineage``
        artifact plan 0.2 requires, bound to ``manifest.lineage_hash`` --
        called before ``run_update_wave()`` ever screens, since this
        method itself is always called before that. Rejects a stale or
        mismatched persisted lineage rather than silently trusting it."""
        path = self._lineage_path()
        if path.is_file():
            existing_lineage = UpdateLineage.load(path)
            existing_hash = compute_lineage_hash(existing_lineage)
            if existing_hash != manifest.lineage_hash:
                raise UpdateWaveAssemblyIntegrityError(
                    f"lineage artifact at {path} hashes to {existing_hash!r}, which does not match this "
                    f"run's assembly manifest lineage_hash={manifest.lineage_hash!r} -- refusing to trust "
                    "a stale or mismatched lineage artifact"
                )
            return

        self.manifests_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifests_root / f".tmp.{uuid.uuid4().hex}.lineage.json"
        payload = lineage.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            existing_lineage = UpdateLineage.load(path)
            existing_hash = compute_lineage_hash(existing_lineage)
            if existing_hash != manifest.lineage_hash:
                raise UpdateWaveAssemblyIntegrityError(
                    f"lineage artifact for review_run_id={self.review_run_id!r} was concurrently "
                    "persisted with a different lineage_hash -- refusing to trust either."
                )
        else:
            _fsync_dir(path.parent)
        finally:
            tmp_path.unlink(missing_ok=True)

    def reserve_or_verify(
        self,
        *,
        review_id: str,
        combined_records: Sequence[Record],
        lineage: UpdateLineage,
        dedup_membership: Dict[str, Tuple[Dict[str, str], ...]],
    ) -> UpdateWaveAssemblyManifest:
        """Computes this invocation's own manifest from the CURRENT
        assembly output, then either publishes it (first call for this
        run) or verifies a persisted manifest's fingerprint still matches
        it (resume) -- raising ``UpdateWaveAssemblyIntegrityError`` rather
        than silently screening a corpus that drifted from the one
        originally reserved for this ``review_run_id``. Also durably
        persists (or verifies) the complete ``UpdateLineage`` artifact
        bound to the manifest's ``lineage_hash``, before returning.

        ``dedup_membership`` (review finding 2) is the hash-bound
        dedup-provenance artifact: every canonical record id mapped to the
        extra database memberships discovered for it, from
        ``compute_cross_source_membership()`` -- persisted as part of this
        same manifest so it is covered by ``manifest_hash`` and by
        ``fingerprint()``'s drift check, rather than being computed and
        silently discarded."""
        fresh = _build_update_wave_assembly_manifest(
            review_run_id=self.review_run_id, review_id=review_id, combined_records=combined_records,
            lineage=lineage, dedup_membership=dedup_membership,
        )
        path = self._path()
        if path.is_file():
            existing = UpdateWaveAssemblyManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_hashes()
            if existing.review_run_id != self.review_run_id:
                raise UpdateWaveAssemblyIntegrityError(
                    f"assembly manifest at {path} has review_run_id={existing.review_run_id!r}, expected "
                    f"this store's own {self.review_run_id!r} -- refusing to trust a foreign manifest"
                )
            if existing.fingerprint() != fresh.fingerprint():
                raise UpdateWaveAssemblyIntegrityError(
                    f"assembly manifest at {path} no longer matches the corpus this invocation just "
                    "re-assembled -- the parent, update spec, review config, or new_raw_records changed "
                    f"since review_run_id={self.review_run_id!r}'s corpus was first reserved. Refusing to "
                    "silently screen a different corpus under the same run id."
                )
            self._verify_or_persist_lineage(manifest=existing, lineage=lineage)
            return existing

        self.manifests_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifests_root / f".tmp.{uuid.uuid4().hex}.json"
        payload = fresh.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            existing = UpdateWaveAssemblyManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_hashes()
            if existing.fingerprint() != fresh.fingerprint():
                raise UpdateWaveAssemblyIntegrityError(
                    f"assembly manifest for review_run_id={self.review_run_id!r} was concurrently "
                    "reserved with a different fingerprint -- refusing to trust either."
                )
            result = existing
        else:
            _fsync_dir(path.parent)
            result = fresh
        finally:
            tmp_path.unlink(missing_ok=True)
        self._verify_or_persist_lineage(manifest=result, lineage=lineage)
        return result


# ---------------------------------------------------------------------------
# Review finding 3 (2026-09-25): decide_census_vs_sampling() was computed
# only by tests -- run_update_wave() could screen the full assembled corpus
# without ever producing or verifying a durable decision bound to the
# deduplicated corpus, configuration, cap, and sample seed. This artifact +
# store close that gap: reserve_or_verify() is called BEFORE any screening,
# mirroring UpdateWaveAssemblyManifestStore's recompute-and-compare
# discipline (never silently re-decide under drifted inputs).
# ---------------------------------------------------------------------------


class UpdateWaveCensusDecisionIntegrityError(RuntimeError):
    """A persisted ``UpdateWaveCensusDecisionRecord`` failed self-
    verification, or a resumed decision no longer matches the corpus/
    configuration/cap/sample parameters originally reserved for this
    ``review_run_id`` -- never silently re-decide or screen under drifted
    inputs."""


_UPDATE_WAVE_CENSUS_DECISION_SCHEMA_VERSION = "update_wave_census_decision_v3"


class UpdateWaveCensusDecisionRecord(BaseModel):
    """The durable binding between one ``review_run_id`` and the exact
    census-vs-sampling decision made for it: which deduplicated corpus
    (``combined_records_hash``), which configuration
    (``review_config_hash``), which authorized cap, which sample
    size/seed/estimator version/pricing scenario produced
    ``recommendation``, and (review finding 5, 2026-09-25 second round)
    which exact sample records that recommendation was computed against
    (``sample_record_ids_hash``) and the sample's own worst-case cost
    (``sample_scenario_max_output_usd``) -- reserved once, verified (never
    silently recomputed) on every resume, so ``run_update_wave()`` can
    never screen a corpus under a decision that does not actually describe
    it, and a stale decision can never pass verification merely because
    its final ``recommendation`` string happens to still match while the
    estimator, pricing scenario, or underlying sample selection changed.

    Review finding 1 (2026-09-25, third round): ``decide_census_vs_sampling()``
    gates its ``recommendation`` on the COMBINED title-screening + screening-QC
    worst-case cost (``census_total_scenario_max_output_usd``/
    ``sample_total_scenario_max_output_usd``), but this record used to persist
    only the title-screening cost lines -- ``render()`` therefore understated
    the true priced cost, and ``fingerprint()`` could not detect a QC-only
    cost drift (e.g. a pricing-table or QC-assumption change that leaves
    ``review_config_hash`` unchanged but moves the QC cost lines). Schema
    bumped to v3 and widened with the four QC cost fields below -- both
    title-only and QC-only figures are persisted so no cost figure is ever
    silently dropped, and ``fingerprint()`` binds all four worst-case cost
    figures (title x2, QC x2), not just the sample title figure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["update_wave_census_decision_v3"] = _UPDATE_WAVE_CENSUS_DECISION_SCHEMA_VERSION
    review_run_id: str
    combined_records_hash: str
    review_config_hash: str
    cap_usd: float
    sample_size: int
    sample_seed: int
    cost_scenario: str
    estimator_version: str
    recommendation: Literal["census", "sample"]
    census_expected_usd: float
    census_scenario_max_output_usd: float
    census_qc_expected_usd: float
    census_qc_scenario_max_output_usd: float
    sample_expected_usd: float
    sample_scenario_max_output_usd: float
    sample_qc_expected_usd: float
    sample_qc_scenario_max_output_usd: float
    sample_record_ids_hash: str
    decision_hash: str

    @property
    def census_total_expected_usd(self) -> float:
        return self.census_expected_usd + self.census_qc_expected_usd

    @property
    def census_total_scenario_max_output_usd(self) -> float:
        return self.census_scenario_max_output_usd + self.census_qc_scenario_max_output_usd

    @property
    def sample_total_expected_usd(self) -> float:
        return self.sample_expected_usd + self.sample_qc_expected_usd

    @property
    def sample_total_scenario_max_output_usd(self) -> float:
        return self.sample_scenario_max_output_usd + self.sample_qc_scenario_max_output_usd

    def compute_decision_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "decision_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_decision_hash()
        if self.decision_hash != expected:
            raise UpdateWaveCensusDecisionIntegrityError(
                f"UpdateWaveCensusDecisionRecord for review_run_id={self.review_run_id!r} failed "
                f"decision_hash self-verification: stored {self.decision_hash!r} != recomputed "
                f"{expected!r} -- file corrupted or tampered"
            )

    def fingerprint(self) -> Tuple[Any, ...]:
        return (
            self.combined_records_hash, self.review_config_hash, self.cap_usd, self.sample_size,
            self.sample_seed, self.cost_scenario, self.estimator_version, self.recommendation,
            self.sample_scenario_max_output_usd, self.sample_record_ids_hash,
            self.census_scenario_max_output_usd, self.census_qc_scenario_max_output_usd,
            self.sample_qc_scenario_max_output_usd,
        )

    def render(self) -> str:
        """Review finding 2 (2026-09-25, second round): human-readable
        rendering for the ``plan-screen`` CLI subcommand -- the persisted
        decision a reviewer must read before authorizing a separate
        ``screen`` invocation to consume it. Review finding 1 (2026-09-25,
        third round): reports title-screening, QC, and combined (the figure
        the recommendation is actually gated on) cost lines separately, not
        just the title-screening figure alone."""
        lines = [
            f"Planned census-vs-sampling decision for review_run_id={self.review_run_id!r}:",
            f"  cap_usd={self.cap_usd:.4f}, cost_scenario={self.cost_scenario!r}, "
            f"estimator_version={self.estimator_version!r}",
            f"  census title-screen: expected_usd={self.census_expected_usd:.4f}, "
            f"scenario_max_output_usd={self.census_scenario_max_output_usd:.4f}",
            f"  census QC: expected_usd={self.census_qc_expected_usd:.4f}, "
            f"scenario_max_output_usd={self.census_qc_scenario_max_output_usd:.4f}",
            f"  census combined: expected_usd={self.census_total_expected_usd:.4f}, "
            f"scenario_max_output_usd={self.census_total_scenario_max_output_usd:.4f}",
            f"  sample (size={self.sample_size}, seed={self.sample_seed}) title-screen: "
            f"expected_usd={self.sample_expected_usd:.4f}, "
            f"scenario_max_output_usd={self.sample_scenario_max_output_usd:.4f}",
            f"  sample QC: expected_usd={self.sample_qc_expected_usd:.4f}, "
            f"scenario_max_output_usd={self.sample_qc_scenario_max_output_usd:.4f}",
            f"  sample combined: expected_usd={self.sample_total_expected_usd:.4f}, "
            f"scenario_max_output_usd={self.sample_total_scenario_max_output_usd:.4f}, "
            f"record_ids_hash={self.sample_record_ids_hash!r}",
            f"  recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)


def _build_update_wave_census_decision_record(
    *,
    review_run_id: str,
    combined_records: Sequence[Record],
    cfg: ReviewConfig,
    cap_usd: float,
    sample_size: int,
    sample_seed: int,
    scenario: "Scenario" = "base",
) -> UpdateWaveCensusDecisionRecord:
    decision = decide_census_vs_sampling(
        deduplicated_records=list(combined_records), cfg=cfg, cap_usd=cap_usd,
        sample_size=sample_size, sample_seed=sample_seed, scenario=scenario,
    )
    if sample_size == 0:
        sample_record_ids: Sequence[str] = ()
    else:
        sample_record_ids = tuple(
            r.id for r in draw_sample_records(combined_records, sample_size=sample_size, sample_seed=sample_seed)
        )
    base = dict(
        schema_version=_UPDATE_WAVE_CENSUS_DECISION_SCHEMA_VERSION,
        review_run_id=review_run_id,
        combined_records_hash=sha256_hex([r.to_dict() for r in combined_records]),
        review_config_hash=sha256_hex(cfg.to_dict()),
        cap_usd=cap_usd,
        sample_size=sample_size,
        sample_seed=sample_seed,
        cost_scenario=str(scenario),
        estimator_version=CENSUS_VS_SAMPLING_ESTIMATOR_VERSION,
        recommendation=decision.recommendation,
        census_expected_usd=decision.census.expected_usd,
        census_scenario_max_output_usd=decision.census.scenario_max_output_usd,
        census_qc_expected_usd=decision.census_qc.expected_usd,
        census_qc_scenario_max_output_usd=decision.census_qc.scenario_max_output_usd,
        sample_expected_usd=decision.sample.expected_usd,
        sample_scenario_max_output_usd=decision.sample.scenario_max_output_usd,
        sample_qc_expected_usd=decision.sample_qc.expected_usd,
        sample_qc_scenario_max_output_usd=decision.sample_qc.scenario_max_output_usd,
        sample_record_ids_hash=sha256_hex(list(sample_record_ids)),
    )
    decision_hash = UpdateWaveCensusDecisionRecord.model_construct(**base, decision_hash="").compute_decision_hash()
    return UpdateWaveCensusDecisionRecord.model_validate(dict(base, decision_hash=decision_hash))


class UpdateWaveCensusDecisionStore:
    """Restart-safe home for one update wave's census-vs-sampling decision
    -- same safe-token-plus-resolved-containment, atomic-write-then-verify,
    and recompute-and-compare-on-resume discipline as
    ``UpdateWaveAssemblyManifestStore``."""

    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        if not _SAFE_UPDATE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        decisions_root = Path(output_dir) / "update_wave_census_decisions"
        run_dir = decisions_root
        if not run_dir.resolve().is_relative_to(decisions_root.resolve()):
            raise ValueError(
                f"review_run_id {review_run_id!r} resolves outside the update_wave_census_decisions "
                "root -- refusing to construct store"
            )
        self.review_run_id = review_run_id
        self.decisions_root = decisions_root

    def _path(self) -> Path:
        return self.decisions_root / f"{self.review_run_id}.json"

    def reserve_or_verify(
        self,
        *,
        combined_records: Sequence[Record],
        cfg: ReviewConfig,
        cap_usd: float,
        sample_size: int,
        sample_seed: int,
        scenario: "Scenario" = "base",
    ) -> UpdateWaveCensusDecisionRecord:
        """Computes this invocation's own decision from the CURRENT corpus/
        configuration/cap/sample parameters, then either publishes it
        (first call for this run) or verifies a persisted decision's
        fingerprint still matches it (resume) -- raising
        ``UpdateWaveCensusDecisionIntegrityError`` rather than silently
        screening under a decision that no longer describes this corpus,
        configuration, cap, or sample parameters."""
        fresh = _build_update_wave_census_decision_record(
            review_run_id=self.review_run_id, combined_records=combined_records, cfg=cfg, cap_usd=cap_usd,
            sample_size=sample_size, sample_seed=sample_seed, scenario=scenario,
        )
        path = self._path()
        if path.is_file():
            existing = UpdateWaveCensusDecisionRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_hashes()
            if existing.review_run_id != self.review_run_id:
                raise UpdateWaveCensusDecisionIntegrityError(
                    f"census decision at {path} has review_run_id={existing.review_run_id!r}, expected "
                    f"this store's own {self.review_run_id!r} -- refusing to trust a foreign decision"
                )
            if existing.fingerprint() != fresh.fingerprint():
                raise UpdateWaveCensusDecisionIntegrityError(
                    f"census decision at {path} no longer matches the corpus/configuration/cap/sample "
                    "parameters this invocation just recomputed -- the deduplicated corpus, review "
                    f"config, cap_usd, sample_size, or sample_seed changed since review_run_id="
                    f"{self.review_run_id!r}'s decision was first reserved. Refusing to silently screen "
                    "under a stale decision."
                )
            return existing

        self.decisions_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.decisions_root / f".tmp.{uuid.uuid4().hex}.json"
        payload = fresh.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            existing = UpdateWaveCensusDecisionRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_hashes()
            if existing.fingerprint() != fresh.fingerprint():
                raise UpdateWaveCensusDecisionIntegrityError(
                    f"census decision for review_run_id={self.review_run_id!r} was concurrently reserved "
                    "with a different fingerprint -- refusing to trust either."
                )
            return existing
        else:
            _fsync_dir(path.parent)
            return fresh
        finally:
            tmp_path.unlink(missing_ok=True)

    def require_existing_and_verify(
        self,
        *,
        combined_records: Sequence[Record],
        cfg: ReviewConfig,
        cap_usd: float,
        sample_size: int,
        sample_seed: int,
        scenario: "Scenario" = "base",
    ) -> UpdateWaveCensusDecisionRecord:
        """Review finding 2 (2026-09-25, second round): the screening-time
        half of the plan/screen split -- loads an EXISTING persisted
        decision and verifies it against the CURRENT corpus/configuration/
        cap/sample/scenario parameters, exactly like ``reserve_or_verify()``
        on resume. Unlike ``reserve_or_verify()``, this NEVER creates a
        decision when none exists: it raises
        ``UpdateWaveCensusDecisionIntegrityError`` instead, forcing a prior,
        separate ``plan_update_wave()`` call (offline, no screening) so a
        human reviews the decision before any paid screening call can be
        made under it."""
        path = self._path()
        if not path.is_file():
            raise UpdateWaveCensusDecisionIntegrityError(
                f"no census-vs-sampling decision has been planned yet for review_run_id="
                f"{self.review_run_id!r} (expected at {path}) -- call plan_update_wave() (or the "
                "'plan-screen' CLI subcommand), review its decision, then re-invoke screening. "
                "run_update_wave() never creates a decision itself."
            )
        fresh = _build_update_wave_census_decision_record(
            review_run_id=self.review_run_id, combined_records=combined_records, cfg=cfg, cap_usd=cap_usd,
            sample_size=sample_size, sample_seed=sample_seed, scenario=scenario,
        )
        existing = UpdateWaveCensusDecisionRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        existing.verify_hashes()
        if existing.review_run_id != self.review_run_id:
            raise UpdateWaveCensusDecisionIntegrityError(
                f"census decision at {path} has review_run_id={existing.review_run_id!r}, expected "
                f"this store's own {self.review_run_id!r} -- refusing to trust a foreign decision"
            )
        if existing.fingerprint() != fresh.fingerprint():
            raise UpdateWaveCensusDecisionIntegrityError(
                f"census decision at {path} no longer matches the corpus/configuration/cap/sample/"
                "scenario parameters this invocation just recomputed -- the deduplicated corpus, "
                f"review config, cap_usd, sample_size, sample_seed, or cost_scenario changed since "
                f"review_run_id={self.review_run_id!r}'s decision was planned. Refusing to silently "
                "screen under a stale decision -- re-run plan_update_wave() and review the new "
                "decision before screening."
            )
        return existing


# ---------------------------------------------------------------------------
# User-named findings (2026-09-22): `run_update_wave()` accepted a caller-
# supplied `new_raw_records` with no real search step and no runtime volume/
# provenance check, so an arbitrarily large or unattributed record list could
# reach paid screening. `enforce_search_volume_and_provenance()` closes the
# second gap; `run_update_wave_search()`/`UpdateWaveSearchArtifactStore`
# close the first by wrapping the real, production `search_node()` (never a
# reimplementation of adapter dispatch) in a restart-safe, hashed artifact
# that a resumed run reuses verbatim rather than re-issuing a live search.
# ---------------------------------------------------------------------------

REQUIRED_UPDATE_SEARCH_DATABASES = frozenset({"pubmed", "openalex", "scopus"})
# Review finding 1 (2026-09-25): the original 1000/3000 values were a
# pilot-run assumption (the capped berg2023-update-20260923 volume), not a
# safety ceiling -- any database with a genuine comprehensive-frame hit
# count above 1000 became architecturally impossible to retrieve exhaustively,
# directly contradicting Revision 5's comprehensive-retrieval goal. These
# constants are now a runaway-query safety rail only, anchored to
# SearchBackendRequest.__post_init__'s own existing max_results ceiling
# (100_000) -- not an expected-volume assumption -- for the 3 required
# databases (100_000 * 3 = 300_000).
MAX_RECORDS_PER_ADAPTER = 100_000
MAX_RECORDS_TOTAL = 300_000


class SearchVolumeError(RuntimeError):
    """Raised when real search output exceeds the per-adapter/total record
    caps, or contains a record not attributed to one of the authorized
    backends -- refuses to forward any of it to a paid LLM call."""


def enforce_search_volume_and_provenance(
    *,
    records: Sequence[Record],
    allowed_databases: Sequence[str],
    max_per_adapter: int = MAX_RECORDS_PER_ADAPTER,
    max_total: int = MAX_RECORDS_TOTAL,
) -> Dict[str, int]:
    """Groups ``records`` by their own ``source`` field and enforces, before
    any of them can reach a paid LLM call: every record is attributed to one
    of ``allowed_databases`` (never blank, never an unrecognized backend);
    no single backend contributes more than ``max_per_adapter`` records; the
    combined total does not exceed ``max_total``. Returns the per-database
    counts on success. Raises ``SearchVolumeError`` on any violation."""
    allowed = set(allowed_databases)
    counts: Dict[str, int] = {db: 0 for db in allowed}
    for record in records:
        if record.source not in allowed:
            raise SearchVolumeError(
                f"record id={record.id!r} source={record.source!r} is not attributed to one of the "
                f"authorized backends {sorted(allowed)} -- refusing to accept unattributed search output"
            )
        counts[record.source] += 1
    for db, count in counts.items():
        if count > max_per_adapter:
            raise SearchVolumeError(
                f"backend {db!r} returned {count} records, exceeding the per-adapter cap of {max_per_adapter}"
            )
    total = sum(counts.values())
    if total > max_total:
        raise SearchVolumeError(
            f"total record count {total} exceeds the combined cap of {max_total} across {sorted(allowed)}"
        )
    return counts


class SearchTemporalWindowError(RuntimeError):
    """Raised when a raw search record's year falls outside the review
    config's ``[date_from, date_to]`` window, or has no year at all. A
    live-run finding (2026-09-22) proved a real database can silently
    ignore its own date-filter parameters (NCBI's ESearch drops
    ``mindate``/``maxdate`` unless both are supplied together) -- this is a
    second, independent, post-retrieval check that never trusts an
    adapter's own date filtering to have actually worked. Refuses to
    persist a search result containing an out-of-window or undated record
    rather than silently letting it through as a genuine 'new' study."""


def enforce_record_temporal_window(
    *, records: Sequence[Record], date_from: str, date_to: Optional[str],
) -> None:
    """Checked immediately after ``enforce_search_volume_and_provenance()``,
    before any raw record is persisted. ``date_from`` is always required by
    ``run_update_wave_search()`` before this point; ``date_to`` may be
    ``None``, in which case only the lower bound and the no-year case are
    enforced."""
    min_year = int(date_from)
    max_year = int(date_to) if date_to else None
    violations: List[str] = []
    for record in records:
        if record.year is None:
            violations.append(f"{record.source}:{record.source_id} has no year")
        elif record.year < min_year:
            violations.append(f"{record.source}:{record.source_id} year={record.year} < {min_year}")
        elif max_year is not None and record.year > max_year:
            violations.append(f"{record.source}:{record.source_id} year={record.year} > {max_year}")
    if violations:
        window = f"[{min_year}, {max_year}]" if max_year is not None else f"[{min_year}, unbounded)"
        raise SearchTemporalWindowError(
            f"{len(violations)} record(s) fall outside the configured search window {window} or have "
            f"no year at all -- refusing to persist a search result a database may have filtered "
            f"incorrectly; first violations: {violations[:5]}"
        )


class UpdateWaveSearchArtifactIntegrityError(RuntimeError):
    """A persisted ``UpdateWaveSearchArtifact`` failed self-verification, or
    belongs to a different ``review_run_id`` -- never silently trust a
    corrupted, tampered, or foreign search artifact."""


_UPDATE_WAVE_SEARCH_ARTIFACT_SCHEMA_VERSION = "update_wave_search_artifact_v1"


class UpdateWaveSearchArtifact(BaseModel):
    """The durable, hashed record of one real search-node invocation for a
    given ``review_run_id`` -- ``UpdateWaveSearchArtifactStore`` persists
    exactly one of these per run and, on any subsequent call, returns it
    verbatim instead of re-invoking ``search_node()`` (unlike the assembly
    manifest above, search is never cheap/pure to re-run, so restart safety
    here means skip-if-exists, not recompute-and-verify)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["update_wave_search_artifact_v1"] = _UPDATE_WAVE_SEARCH_ARTIFACT_SCHEMA_VERSION
    review_run_id: str
    databases: Tuple[str, ...]
    per_database_counts: Dict[str, int]
    total_records: int
    raw_records: Tuple[Dict[str, Any], ...]
    raw_records_hash: str
    search_log: Dict[str, Any]
    review_config_hash: str
    db_queries_hash: str
    execution_dates: Dict[str, str]
    artifact_hash: str

    def compute_artifact_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "artifact_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected_records_hash = sha256_hex(list(self.raw_records))
        if self.raw_records_hash != expected_records_hash:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"UpdateWaveSearchArtifact for review_run_id={self.review_run_id!r} failed raw_records_hash "
                f"self-verification: stored {self.raw_records_hash!r} != recomputed {expected_records_hash!r} "
                "-- file corrupted or tampered"
            )
        expected = self.compute_artifact_hash()
        if self.artifact_hash != expected:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"UpdateWaveSearchArtifact for review_run_id={self.review_run_id!r} failed artifact_hash "
                f"self-verification: stored {self.artifact_hash!r} != recomputed {expected!r} -- file "
                "corrupted or tampered"
            )

    def verify_bindings(self, *, review_config_hash: str, db_queries_hash: str) -> None:
        """Called on every resume-search load (user-named finding: a
        resumed search used to reuse a stale artifact under a silently
        changed review config or translated queries). Raises
        ``UpdateWaveSearchArtifactIntegrityError`` if either input that
        could invalidate the persisted search result has changed since it
        was produced."""
        if self.review_config_hash != review_config_hash:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"UpdateWaveSearchArtifact for review_run_id={self.review_run_id!r} was produced under a "
                f"different review config (stored hash {self.review_config_hash!r} != current "
                f"{review_config_hash!r}) -- refusing to resume under a changed configuration"
            )
        if self.db_queries_hash != db_queries_hash:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"UpdateWaveSearchArtifact for review_run_id={self.review_run_id!r} was produced under "
                f"different db_queries (stored hash {self.db_queries_hash!r} != current {db_queries_hash!r}) "
                "-- refusing to resume under changed queries"
            )

    def records(self) -> Tuple[Record, ...]:
        return tuple(Record.model_validate(d) for d in self.raw_records)

    def parsed_search_log(self) -> SearchLog:
        """Parses the raw ``search_log`` dict through the real ``SearchLog``/
        ``DatabaseSearchLog`` models (review finding P1/#3, 2026-09-24) --
        ``search_log`` stays an untyped dict on this frozen model so
        ``artifact_hash`` keeps hashing exactly the bytes originally
        persisted (never re-serialising a migrated shape), but every
        decision must be made off THIS parsed view, not the raw dict: a
        legacy v2 artifact's old ``"complete"`` labels only get downgraded
        to the honest ``"capped"`` here, in memory, never on disk. Callers
        must call ``verify_hashes()`` first (``load_if_exists()`` already
        does) -- this method never re-verifies integrity itself."""
        return SearchLog.model_validate(self.search_log)


def _build_update_wave_search_artifact(
    *,
    review_run_id: str,
    databases: Sequence[str],
    per_database_counts: Dict[str, int],
    raw_records: Sequence[Dict[str, Any]],
    search_log: Dict[str, Any],
    review_config_hash: str = "",
    db_queries_hash: str = "",
    execution_dates: Optional[Dict[str, str]] = None,
) -> UpdateWaveSearchArtifact:
    raw_records_tuple = tuple(raw_records)
    base = dict(
        schema_version=_UPDATE_WAVE_SEARCH_ARTIFACT_SCHEMA_VERSION,
        review_run_id=review_run_id,
        databases=tuple(databases),
        per_database_counts=dict(per_database_counts),
        total_records=sum(per_database_counts.values()),
        raw_records=raw_records_tuple,
        raw_records_hash=sha256_hex(list(raw_records_tuple)),
        search_log=search_log,
        review_config_hash=review_config_hash,
        db_queries_hash=db_queries_hash,
        execution_dates=dict(execution_dates or {}),
    )
    artifact_hash = UpdateWaveSearchArtifact.model_construct(**base, artifact_hash="").compute_artifact_hash()
    return UpdateWaveSearchArtifact.model_validate(dict(base, artifact_hash=artifact_hash))


class UpdateWaveSearchArtifactStore:
    """Restart-safe home for one update wave's real search output -- same
    safe-token-plus-resolved-containment and atomic-write discipline as
    ``UpdateWaveAssemblyManifestStore``, but ``load_if_exists()`` is the
    entry point callers must check FIRST: if an artifact already exists for
    this ``review_run_id``, the real search must never be re-issued."""

    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        if not _SAFE_UPDATE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        artifacts_root = Path(output_dir) / "update_wave_search_artifacts"
        if not artifacts_root.resolve().is_relative_to(artifacts_root.resolve()):
            raise ValueError(
                f"review_run_id {review_run_id!r} resolves outside the update_wave_search_artifacts "
                "root -- refusing to construct store"
            )
        self.review_run_id = review_run_id
        self.artifacts_root = artifacts_root

    def _path(self) -> Path:
        return self.artifacts_root / f"{self.review_run_id}.json"

    def load_if_exists(self) -> Optional[UpdateWaveSearchArtifact]:
        """Returns the persisted artifact for this ``review_run_id``, or
        ``None`` if none exists yet. Callers MUST check this before issuing
        a real search -- a non-``None`` result means the search already
        happened and must not be repeated."""
        path = self._path()
        if not path.is_file():
            return None
        artifact = UpdateWaveSearchArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
        artifact.verify_hashes()
        if artifact.review_run_id != self.review_run_id:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"search artifact at {path} has review_run_id={artifact.review_run_id!r}, expected this "
                f"store's own {self.review_run_id!r} -- refusing to trust a foreign artifact"
            )
        return artifact

    def persist(self, artifact: UpdateWaveSearchArtifact) -> UpdateWaveSearchArtifact:
        """Atomically, durably publishes ``artifact`` -- first-writer-wins
        on a race, exactly like ``UpdateWaveAssemblyManifestStore``'s own
        writes. Callers reach this only after ``load_if_exists()`` returned
        ``None`` and a fresh real search has just completed."""
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.artifacts_root / f".tmp.{uuid.uuid4().hex}.json"
        payload = artifact.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        path = self._path()
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            existing = UpdateWaveSearchArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_hashes()
            result = existing
        else:
            _fsync_dir(path.parent)
            result = artifact
        finally:
            tmp_path.unlink(missing_ok=True)
        return result


class FileSearchCheckpointStore:
    """Restart-safe, atomic-overwrite-per-page home for one update wave's
    in-progress pagination state, keyed by ``(review_run_id, database)``
    (Checkpoint 4, bullets 1-2). Unlike ``UpdateWaveSearchArtifactStore``'s
    write-once hardlink discipline (a finished search result is immutable),
    a page checkpoint is expected to be overwritten many times over one
    adapter's pagination -- ``save()`` uses a temp-file-plus-``os.replace()``
    atomic overwrite instead, so a crash mid-write can never leave a
    corrupted or half-written checkpoint behind for the next ``load()``.
    Integrity is verified via ``envelope_hash`` (review finding 5), which
    covers the complete checkpoint envelope, not just ``records_so_far``."""

    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        if not _SAFE_UPDATE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        self.review_run_id = review_run_id
        self.checkpoints_root = Path(output_dir) / "update_wave_search_checkpoints" / review_run_id

    def _path(self, database: str) -> Path:
        if not _SAFE_DATABASE_NAME_RE.fullmatch(database):
            raise ValueError(
                f"database {database!r} is not a safe token -- must match [a-z][a-z0-9_]{{0,63}}"
            )
        return self.checkpoints_root / f"{database}.json"

    def load(self, database: str, fingerprint: str) -> Optional[SearchPageCheckpoint]:
        """Returns the persisted checkpoint for ``database``, or ``None`` if
        none exists yet, OR if one exists but was produced under a
        DIFFERENT ``request_fingerprint`` -- a changed query/config/adapter
        version must never silently resume stale progress (Checkpoint 4
        design rule); the caller starts that database over from page 1
        instead, exactly as if no checkpoint existed."""
        path = self._path(database)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_fingerprint") != fingerprint:
            return None
        expected_envelope_hash = payload.pop("envelope_hash", None)
        checkpoint = SearchPageCheckpoint(
            request_fingerprint=payload["request_fingerprint"],
            adapter=payload["adapter"],
            adapter_config_version=payload["adapter_config_version"],
            cursor_state=dict(payload.get("cursor_state") or {}),
            records_so_far=tuple(payload.get("records_so_far") or ()),
            total_hits=payload.get("total_hits"),
            page_count=payload.get("page_count", 0),
            completed=payload.get("completed", False),
            source_exhausted=payload.get("source_exhausted"),
            termination_reason=payload.get("termination_reason"),
            effective_query=payload.get("effective_query"),
        )
        # Review finding 5 (2026-09-25, first round): verify the FULL envelope
        # hash, not just records_hash -- cursor_state/total_hits/page_count/
        # completed/source_exhausted/termination_reason/effective_query/
        # adapter/adapter_config_version must all be tamper-evident, not just
        # the record payloads.
        # Review finding 6 (2026-09-25, second round): a MISSING envelope_hash
        # key must fail closed, not silently skip verification -- no legacy
        # checkpoint file ever existed without one, so there is no migration
        # case to preserve.
        if expected_envelope_hash is None:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"search checkpoint at {path} has no envelope_hash -- refusing to trust an "
                "unverifiable checkpoint. No legacy checkpoint format predates envelope_hash."
            )
        if checkpoint.envelope_hash != expected_envelope_hash:
            raise UpdateWaveSearchArtifactIntegrityError(
                f"search checkpoint at {path} failed envelope_hash self-verification: stored "
                f"{expected_envelope_hash!r} != recomputed {checkpoint.envelope_hash!r} -- file "
                "corrupted or tampered"
            )
        return checkpoint

    def save(self, database: str, checkpoint: SearchPageCheckpoint) -> None:
        path = self._path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "request_fingerprint": checkpoint.request_fingerprint,
            "adapter": checkpoint.adapter,
            "adapter_config_version": checkpoint.adapter_config_version,
            "cursor_state": checkpoint.cursor_state,
            "records_so_far": list(checkpoint.records_so_far),
            "envelope_hash": checkpoint.envelope_hash,
            "total_hits": checkpoint.total_hits,
            "page_count": checkpoint.page_count,
            "completed": checkpoint.completed,
            "source_exhausted": checkpoint.source_exhausted,
            "termination_reason": checkpoint.termination_reason,
            "effective_query": checkpoint.effective_query,
        }
        tmp_path = path.parent / f".tmp.{uuid.uuid4().hex}.json"
        with open(tmp_path, "wb") as fh:
            fh.write(json.dumps(payload, indent=2).encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)

    def hooks_for(self, database: str) -> SearchCheckpointHooks:
        """Binds this store to one ``database``, returning the
        ``SearchCheckpointHooks`` an adapter's ``.search()`` call expects --
        ``load``/``save`` here take no ``database`` argument, unlike this
        store's own methods, since one adapter call only ever concerns
        itself with its own database."""
        return SearchCheckpointHooks(
            load=lambda fingerprint: self.load(database, fingerprint),
            save=lambda checkpoint: self.save(database, checkpoint),
        )


class SearchRetrievalError(RuntimeError):
    """Raised when any database's real search call did not complete
    successfully (``retrieval_status not in ("exhaustive", "capped")``) --
    e.g. a Scopus call that failed for want of credentials and came back as
    a technically-valid, zero-record response. Refuses to persist a failed
    retrieval as if it were a genuine, error-free zero-hit search."""


class SearchAccessModeError(RuntimeError):
    """Raised when a search-log entry reports a non-``None`` ``access_mode``
    other than ``"COMPLETE"`` (e.g. Scopus's ``STANDARD`` view, or a mid-run
    401 downgrade from COMPLETE). ``access_mode`` is ``None`` for every
    adapter with no COMPLETE/STANDARD-style distinction -- nothing to
    enforce there. STANDARD abstracts carry materially less metadata than
    COMPLETE, so silently accepting them could screen an update wave off
    worse data than the run believed it had; refuses to persist or screen
    under a weaker-than-required database access tier."""


def enforce_required_access_mode(search_log: SearchLog) -> None:
    """Called both before persisting a fresh search
    (``run_update_wave_search``) and again before screening an already
    -persisted one (``verify_search_artifact_and_execution_spec_before_screening``)
    -- a fresh search catches this at the earliest point, but a persisted
    artifact from before this check existed (or one hand-edited) must be
    caught at screening time too. Takes the parsed ``SearchLog`` (review
    finding P1/#3, 2026-09-24), never a raw dict, so a legacy-migrated entry
    is checked on its real, migrated field values. Raises
    ``SearchAccessModeError`` on any entry whose reported access mode is
    neither ``None`` nor ``"COMPLETE"``."""
    for entry in search_log.entries:
        access_mode = entry.access_mode
        if access_mode is not None and access_mode != "COMPLETE":
            raise SearchAccessModeError(
                f"database {entry.database!r} reported access_mode={access_mode!r}, not "
                "'COMPLETE' -- refusing to persist/screen an update wave built on a weaker-than-"
                "required database access tier"
            )


class SearchNotExhaustiveError(RuntimeError):
    """Raised when any database's search-log entry is not
    ``retrieval_status == "exhaustive"`` (review finding P1, 2026-09-24) --
    a ``"capped"`` (top-N/ranked) result must never silently reach paid
    screening as if it were a comprehensive evidence frame. No hash-bound
    sampling-plan bypass exists yet (Revision 5 point 5's still-pending
    census-vs-sample decision); until one is designed and authorized, every
    database in an update-wave search must be exhaustive."""


def enforce_exhaustive_retrieval(search_log: SearchLog) -> None:
    """Called both before persisting a fresh search
    (``run_update_wave_search``) and again before screening an already
    -persisted one (``verify_search_artifact_and_execution_spec_before_screening``)
    -- mirrors ``enforce_required_access_mode``'s dual-call-site pattern.
    Takes the parsed ``SearchLog`` so a legacy v2 artifact's ``"complete"``
    label (migrated in memory to ``"capped"``, never ``"exhaustive"``) is
    caught here too, not just a freshly-produced one. Raises
    ``SearchNotExhaustiveError`` on any entry whose ``retrieval_status`` is
    not ``"exhaustive"``."""
    non_exhaustive = [entry for entry in search_log.entries if entry.retrieval_status != "exhaustive"]
    if non_exhaustive:
        details = "; ".join(
            f"{entry.database}: {entry.retrieval_status} ({entry.termination_reason})"
            for entry in non_exhaustive
        )
        raise SearchNotExhaustiveError(
            f"{len(non_exhaustive)} database(s) are not exhaustive -- refusing to persist/screen a "
            f"capped/ranked or incomplete retrieval as the update wave's evidence frame: {details}"
        )


class SearchAlreadyInProgressError(RuntimeError):
    """Raised when a second ``run_update_wave_search()`` call for the same
    ``review_run_id`` is attempted while one is already in flight --
    refuses to let two concurrent invocations both pass the existence
    check and both issue a real, duplicate search."""


@contextlib.contextmanager
def _update_wave_search_lock(*, artifacts_root: Path, review_run_id: str):
    """Held across the whole load -> search -> persist sequence, so two
    concurrent calls can never both observe "no artifact yet" and both
    issue a live search. User-named finding: an earlier version used a
    plain ``O_CREAT|O_EXCL`` lock FILE, which survives process
    termination (crash, kill -9, power loss) -- nothing but a clean
    ``finally`` ever removed it, so a killed run permanently wedged every
    future restart for that ``review_run_id``. Reuses
    ``ai_traceability.kernel_lock()``'s OS-level ``flock``/``msvcrt`` lock
    instead: the kernel releases it automatically when the holding
    process's file descriptor closes, including on a crash, so a dead
    holder can never block a future run. ``timeout=0.0`` preserves the
    original fail-fast contract -- a genuinely live concurrent holder is
    detected and rejected immediately, never waited out."""
    artifacts_root.mkdir(parents=True, exist_ok=True)
    lock_path = artifacts_root / f".{review_run_id}.search.lock"
    with kernel_lock(
        lock_path,
        error_cls=SearchAlreadyInProgressError,
        description=f"update-wave search review_run_id={review_run_id!r}",
        timeout=0.0,
    ):
        yield


def run_update_wave_search(
    *,
    cfg: ReviewConfig,
    review_run_id: str,
    db_queries: Optional[Dict[str, str]] = None,
) -> UpdateWaveSearchArtifact:
    """The restart-safe, search-only stage the review named as missing: runs
    the real, production ``search_node()`` (never a reimplementation of
    adapter dispatch) exactly once per ``review_run_id``, enforces
    ``enforce_search_volume_and_provenance()`` on its output BEFORE
    persisting anything, and returns a durable, hashed
    ``UpdateWaveSearchArtifact``. A subsequent call for the same
    ``review_run_id`` returns the persisted artifact verbatim and makes zero
    calls into ``search_node()`` -- but only if this call's own
    ``cfg``/``db_queries`` still hash-match what the persisted artifact was
    produced under; otherwise ``UpdateWaveSearchArtifactIntegrityError`` is
    raised rather than silently reusing a stale result under a changed
    configuration or changed queries.

    ``cfg.search.databases`` must equal exactly
    ``REQUIRED_UPDATE_SEARCH_DATABASES``, ``cfg.search.max_results_per_database``
    must not exceed ``MAX_RECORDS_PER_ADAPTER``, and ``cfg.search.date_from``
    must be set -- all checked before touching the store or issuing any
    search, so a misconfigured review config fails closed. The ``date_from``
    check exists because an update wave is defined as searching only the
    period after the original review's cutoff; an unbound search could
    re-pull records the parent corpus's own cap already excluded, which
    would then be misclassified as newly found. ``cfg.search.date_to`` is
    not required -- the execution spec records the actual search date.

    ``db_queries``, if given, is loaded into
    ``state["search_strategy"]["db_queries"]`` before dispatch -- required
    for any database in ``search.py``'s ``_STRICT_TRANSLATION_DBS`` (e.g.
    scopus), which refuses to substitute a lossy rebuilt query.

    Scopus's real adapter reads ``SCOPUS_API_KEY``/``SCOPUS_INSTITUTIONAL_KEY``,
    and OpenAlex's real adapter reads ``OPENALEX_API_KEY``, from
    ``os.environ`` at call time -- ``credentials.loaded_credentials()``
    populates them for just the duration of the real ``search_node()`` call.
    After the call, every entry in the returned search log is checked: any
    ``retrieval_status not in ("exhaustive", "capped")`` (e.g. Scopus
    failing closed with 0 records for want of credentials) raises
    ``SearchRetrievalError`` rather than persisting a failed retrieval as a
    genuine empty result. ``enforce_exhaustive_retrieval()`` is then applied
    on top, requiring every entry to be ``"exhaustive"`` -- a ``"capped"``
    (ranked top-N) or ``"partial"`` retrieval raises ``SearchNotExhaustiveError``
    rather than being persisted as the update wave's evidence frame (review
    finding P1, 2026-09-24). The whole
    load -> search -> persist sequence runs under a per-``review_run_id``
    lock (``SearchAlreadyInProgressError`` on a concurrent attempt)."""
    databases = tuple(cfg.search.databases)
    if set(databases) != REQUIRED_UPDATE_SEARCH_DATABASES:
        raise ValueError(
            f"cfg.search.databases={sorted(databases)} does not equal the required update-wave set "
            f"{sorted(REQUIRED_UPDATE_SEARCH_DATABASES)} -- refusing to search"
        )
    if cfg.search.max_results_per_database > MAX_RECORDS_PER_ADAPTER:
        raise ValueError(
            f"cfg.search.max_results_per_database={cfg.search.max_results_per_database} exceeds the "
            f"per-adapter cap of {MAX_RECORDS_PER_ADAPTER} -- refusing to search"
        )
    if not cfg.search.date_from:
        raise ValueError(
            "cfg.search.date_from is unset -- an update-wave search must search only the period "
            "after the original review's cutoff. An unbound search could pull in records the "
            "parent corpus's own cap already excludes, which would then be misclassified as "
            "'new' rather than recognised as out-of-scope. Refusing to search; set a lower date "
            "bound (e.g. \"2024\") on cfg.search.date_from. cfg.search.date_to may remain unset -- "
            "the execution spec records the actual search date."
        )

    review_config_hash = sha256_hex(cfg.to_dict())
    db_queries_hash = sha256_hex(dict(db_queries) if db_queries else {})
    store = UpdateWaveSearchArtifactStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)

    with _update_wave_search_lock(artifacts_root=store.artifacts_root, review_run_id=review_run_id):
        existing = store.load_if_exists()
        if existing is not None:
            existing.verify_bindings(review_config_hash=review_config_hash, db_queries_hash=db_queries_hash)
            return existing

        state: Dict[str, Any] = dict(initial_state(cfg.to_dict()))
        if db_queries:
            state["search_strategy"] = {"db_queries": dict(db_queries)}
        checkpoint_store = FileSearchCheckpointStore(
            output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id,
        )
        with credentials.loaded_credentials(
            ("SCOPUS_API_KEY", "SCOPUS_INSTITUTIONAL_KEY", "OPENALEX_API_KEY")
        ):
            if not os.environ.get("OPENALEX_API_KEY"):
                # A prior live run already exhausted OpenAlex's anonymous/keyless
                # daily credit budget (429 until midnight UTC, live-run finding
                # 2026-09-23) -- an update-wave search must never proceed
                # anonymously and risk repeating that exact failure. Scoped to
                # THIS driver only; general OpenAlex callers may still use
                # anonymous access.
                raise ValueError(
                    "OPENALEX_API_KEY is not set (env, OS keyring, or credential file) -- refusing "
                    "to run an update-wave search on OpenAlex's anonymous/keyless daily budget. Set "
                    "it first: python -m slr_pipeline.credentials set OPENALEX_API_KEY"
                )
            # Additive, backward-compatible: only passed when the callable
            # bound to `search_node` actually declares this parameter --
            # keeps every existing `fake_search_node(state)` test double
            # (which stands in for the real function via monkeypatch)
            # working unchanged, same discipline as `_dispatch_one()`'s own
            # adapter-level `inspect.signature` check (nodes/search.py).
            search_node_kwargs: Dict[str, Any] = {}
            if "checkpoint_hooks" in inspect.signature(search_node).parameters:
                search_node_kwargs["checkpoint_hooks"] = checkpoint_store.hooks_for
            result = search_node(state, **search_node_kwargs)
        parsed_log = SearchLog.model_validate(result["search_log"])
        failed_entries = [
            entry for entry in parsed_log.entries
            if entry.retrieval_status not in ("exhaustive", "capped")
        ]
        if failed_entries:
            details = "; ".join(
                f"{entry.database}: {entry.retrieval_status} ({entry.failure_detail})"
                for entry in failed_entries
            )
            raise SearchRetrievalError(
                f"review_run_id={review_run_id!r}: {len(failed_entries)} database(s) did not complete "
                f"successfully -- refusing to persist a failed retrieval as a genuine result: {details}"
            )
        enforce_exhaustive_retrieval(parsed_log)
        enforce_required_access_mode(parsed_log)
        raw_records = result["raw_records"]
        records = tuple(Record.model_validate(d) for d in raw_records)
        per_database_counts = enforce_search_volume_and_provenance(records=records, allowed_databases=databases)
        enforce_record_temporal_window(records=records, date_from=cfg.search.date_from, date_to=cfg.search.date_to)
        execution_dates = {entry.database: entry.date for entry in parsed_log.entries}
        artifact = _build_update_wave_search_artifact(
            review_run_id=review_run_id, databases=databases, per_database_counts=per_database_counts,
            raw_records=raw_records, search_log=result["search_log"],
            review_config_hash=review_config_hash, db_queries_hash=db_queries_hash,
            execution_dates=execution_dates,
        )
        return store.persist(artifact)


class UpdateWaveExecutionSpecIntegrityError(RuntimeError):
    """A persisted update-wave execution spec failed self-verification,
    belongs to a foreign ``review_run_id``, or a caller attempted to
    persist a spec that differs from one already on disk for this
    ``review_run_id`` -- once written, an execution spec is never
    mutated."""


def build_update_execution_spec(
    *, draft_spec: ReproductionSpec, search_artifact: UpdateWaveSearchArtifact,
) -> ReproductionSpec:
    """Mints the immutable execution spec the review named as missing:
    screening a draft spec whose ``search_dates`` are all ``"unknown"`` is
    a protocol violation once a real search has actually run. Every field
    is copied from ``draft_spec`` except ``search_dates``/
    ``queries_executed``, which come from ``search_artifact``'s own real
    per-database execution dates and each backend's actual executed query
    -- never the draft's placeholder or a re-derived query."""
    queries_executed = {
        entry["database"]: entry["query"] for entry in search_artifact.search_log.get("entries", [])
    }
    payload = dict(draft_spec.model_dump(mode="json"))
    payload["search_dates"] = dict(search_artifact.execution_dates)
    payload["queries_executed"] = queries_executed
    return ReproductionSpec.model_validate(payload)


class UpdateWaveExecutionSpecStore:
    """Write-once home for one update wave's immutable execution spec --
    same safe-token-plus-resolved-containment and atomic-write discipline
    as ``UpdateWaveSearchArtifactStore``, but ``persist()`` refuses to
    overwrite an existing spec with a DIFFERENT one (``spec_hash()``
    mismatch) rather than first-writer-wins -- an execution spec, once
    written, must never be silently replaced."""

    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        if not _SAFE_UPDATE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        self.review_run_id = review_run_id
        self.specs_root = Path(output_dir) / "update_wave_execution_specs"

    def _path(self) -> Path:
        return self.specs_root / f"{self.review_run_id}.json"

    def _load_envelope(self, path: Path) -> ReproductionSpec:
        """``ReproductionSpec`` itself carries no ``review_run_id`` field, so
        the envelope this store writes wraps it with one -- otherwise a
        spec misfiled under the wrong ``review_run_id`` would load
        successfully with nothing to catch it, unlike
        ``UpdateWaveSearchArtifactStore``'s own foreign-artifact check.

        The envelope also carries the ``spec_hash`` it was minted with
        (user-named finding: without a stored expected hash, a manually
        corrupted ``spec`` payload would simply recompute a different,
        self-consistent ``spec_hash()`` and load as though it were another
        genuinely valid spec, with nothing to catch the tamper)."""
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("review_run_id") != self.review_run_id:
            raise UpdateWaveExecutionSpecIntegrityError(
                f"execution spec at {path} has review_run_id={envelope.get('review_run_id')!r}, expected "
                f"this store's own {self.review_run_id!r} -- refusing to trust a foreign execution spec"
            )
        spec = ReproductionSpec.model_validate(envelope["spec"])
        expected_hash = envelope.get("spec_hash")
        if spec.spec_hash() != expected_hash:
            raise UpdateWaveExecutionSpecIntegrityError(
                f"execution spec at {path} failed spec_hash self-verification: envelope declares "
                f"{expected_hash!r}, recomputed {spec.spec_hash()!r} -- file corrupted or tampered"
            )
        return spec

    def load_if_exists(self) -> Optional[ReproductionSpec]:
        path = self._path()
        if not path.is_file():
            return None
        return self._load_envelope(path)

    def persist(self, spec: ReproductionSpec) -> ReproductionSpec:
        self.specs_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.specs_root / f".tmp.{uuid.uuid4().hex}.json"
        envelope = {
            "review_run_id": self.review_run_id,
            "spec_hash": spec.spec_hash(),
            "spec": spec.model_dump(mode="json"),
        }
        payload = json.dumps(envelope, indent=2, ensure_ascii=False).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        path = self._path()
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            existing = self._load_envelope(path)
            if existing.spec_hash() != spec.spec_hash():
                raise UpdateWaveExecutionSpecIntegrityError(
                    f"an execution spec already exists for review_run_id={self.review_run_id!r} "
                    f"(spec_hash={existing.spec_hash()!r}) that differs from the one just built "
                    f"(spec_hash={spec.spec_hash()!r}) -- an execution spec is never mutated once written"
                )
            result = existing
        else:
            _fsync_dir(path.parent)
            result = spec
        finally:
            tmp_path.unlink(missing_ok=True)
        return result


def verify_search_artifact_and_execution_spec_before_screening(
    *,
    cfg: ReviewConfig,
    search_artifact: UpdateWaveSearchArtifact,
    update_spec: ReproductionSpec,
) -> Dict[str, int]:
    """Called by ``screen`` immediately before any proxy/LLM client is
    constructed (user-named findings, round 3): loading a persisted search
    artifact and a persisted execution spec and trusting both outright,
    with no check that either still matches what ``screen`` was actually
    invoked with, let two independently-valid-looking but mismatched files
    reach a paid call together -- e.g. an artifact produced under one
    review config screened under a since-edited one, or an execution spec
    whose ``search_dates``/``queries_executed`` do not actually describe
    the artifact it claims to have been minted from (hand-edited, or
    copied from a different run's spec file). Also re-runs
    ``enforce_search_volume_and_provenance()`` over the artifact's own
    ``records()`` and compares the result against its persisted
    ``per_database_counts`` -- that field is unverified data on load
    otherwise, so a corrupted or hand-edited artifact JSON with an
    inflated ``raw_records`` payload but an untouched ``per_database_counts``
    would otherwise sail through to a paid screening call unnoticed. Also
    re-applies ``enforce_exhaustive_retrieval()`` on the artifact's own
    (parsed, migration-aware) search log -- an old artifact predating this
    gate, or one hand-edited to look ``"capped"``/``"complete"``, is caught
    here too, not only at pre-persist time (review finding P1, 2026-09-24).
    Returns the recomputed counts on success; raises
    ``UpdateWaveSearchArtifactIntegrityError``/
    ``UpdateWaveExecutionSpecIntegrityError``/``SearchNotExhaustiveError`` on
    any mismatch."""
    review_config_hash = sha256_hex(cfg.to_dict())
    if search_artifact.review_config_hash != review_config_hash:
        raise UpdateWaveSearchArtifactIntegrityError(
            f"search artifact for review_run_id={search_artifact.review_run_id!r} was produced under a "
            f"different review config (stored hash {search_artifact.review_config_hash!r} != current "
            f"{review_config_hash!r}) -- refusing to screen under a changed configuration"
        )
    parsed_log = search_artifact.parsed_search_log()
    enforce_exhaustive_retrieval(parsed_log)
    enforce_required_access_mode(parsed_log)

    expected_dates = dict(search_artifact.execution_dates)
    if dict(update_spec.search_dates) != expected_dates:
        raise UpdateWaveExecutionSpecIntegrityError(
            f"execution spec search_dates {dict(update_spec.search_dates)!r} do not match search artifact "
            f"execution_dates {expected_dates!r} for review_run_id={search_artifact.review_run_id!r} -- "
            "refusing to screen under a spec that does not describe this artifact's real search"
        )
    expected_queries = {entry.database: entry.query for entry in parsed_log.entries}
    if dict(update_spec.queries_executed) != expected_queries:
        raise UpdateWaveExecutionSpecIntegrityError(
            f"execution spec queries_executed {dict(update_spec.queries_executed)!r} do not match search "
            f"artifact search_log queries {expected_queries!r} for review_run_id="
            f"{search_artifact.review_run_id!r} -- refusing to screen under a mismatched spec"
        )

    recomputed_counts = enforce_search_volume_and_provenance(
        records=search_artifact.records(), allowed_databases=search_artifact.databases
    )
    enforce_record_temporal_window(
        records=search_artifact.records(), date_from=cfg.search.date_from, date_to=cfg.search.date_to
    )
    if recomputed_counts != dict(search_artifact.per_database_counts):
        raise UpdateWaveSearchArtifactIntegrityError(
            f"recomputed per-database counts {recomputed_counts!r} do not match persisted artifact "
            f"per_database_counts {dict(search_artifact.per_database_counts)!r} for review_run_id="
            f"{search_artifact.review_run_id!r} -- refusing to screen a corrupted or hand-edited artifact"
        )
    return recomputed_counts


def run_update_wave(
    *,
    cfg: ReviewConfig,
    parent: VerifiedReproductionArtifactSet,
    update_spec: ReproductionSpec,
    new_raw_records: Sequence[Record],
    review_id: str,
    review_run_id: str,
    cap_usd: float,
    sample_size: int,
    sample_seed: int,
    cost_scenario: "Scenario" = "base",
) -> Dict[str, Any]:
    """The single restart-safe orchestration entry point Checkpoint 3's
    review named as missing: binds ``assemble_update_corpus()``'s
    assembled ids/lineage hash to a durable ``UpdateWaveAssemblyManifest``
    BEFORE any screening call, then screens the assembled corpus via
    ``run_update_wave_screening()``.

    On a resumed call (same ``review_run_id``), ``assemble_update_corpus()``
    is run again (it is pure/offline -- no cost to re-run), and its output
    is verified against the manifest already reserved for this
    ``review_run_id`` -- if the parent, update spec, review config, or
    ``new_raw_records`` changed since the original reservation, this
    raises ``UpdateWaveAssemblyIntegrityError`` rather than silently
    screening a different corpus under the same run id.

    Review finding 3 (first round): a durable ``UpdateWaveCensusDecisionRecord``
    is bound to the assembled corpus, ``cfg``, ``cap_usd``, ``sample_size``,
    and ``sample_seed`` -- this is REQUIRED before screening, not merely
    available. When the decision recommends ``"sample"``, only the
    deterministic seeded sample (``draw_sample_records()``, the exact same
    draw ``decide_census_vs_sampling()`` priced) is screened, never the
    full assembled corpus.

    Review finding 2 (2026-09-25, second round): this function no longer
    CREATES that decision -- it requires one already planned by a prior,
    separate ``plan_update_wave()`` call (offline, no screening), reviewed
    by a human, and raises ``UpdateWaveCensusDecisionIntegrityError`` if
    none exists yet. This is the human decision checkpoint between
    deciding census-vs-sample and spending money screening under that
    decision.

    Returns ``{"lineage": UpdateLineage, "dedup_membership": ...,
    "census_decision": UpdateWaveCensusDecisionRecord,
    **run_update_wave_screening()'s state}``."""
    combined_records, lineage, dedup_membership = assemble_update_corpus(
        parent=parent, update_spec=update_spec, review_config=cfg, new_raw_records=new_raw_records,
        review_id=review_id,
    )
    manifest_store = UpdateWaveAssemblyManifestStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    manifest_store.reserve_or_verify(
        review_id=review_id, combined_records=combined_records, lineage=lineage, dedup_membership=dedup_membership,
    )

    decision_store = UpdateWaveCensusDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    census_decision = decision_store.require_existing_and_verify(
        combined_records=combined_records, cfg=cfg, cap_usd=cap_usd, sample_size=sample_size,
        sample_seed=sample_seed, scenario=cost_scenario,
    )
    if census_decision.recommendation == "census":
        records_to_screen: Sequence[Record] = combined_records
    else:
        records_to_screen = draw_sample_records(combined_records, sample_size=sample_size, sample_seed=sample_seed)

    state = run_update_wave_screening(cfg=cfg, combined_records=records_to_screen, review_run_id=review_run_id)
    return {"lineage": lineage, "dedup_membership": dedup_membership, "census_decision": census_decision, **state}


def plan_update_wave(
    *,
    cfg: ReviewConfig,
    parent: VerifiedReproductionArtifactSet,
    update_spec: ReproductionSpec,
    new_raw_records: Sequence[Record],
    review_id: str,
    review_run_id: str,
    cap_usd: float,
    sample_size: int,
    sample_seed: int,
    cost_scenario: "Scenario" = "base",
) -> UpdateWaveCensusDecisionRecord:
    """Review finding 2 (2026-09-25, second round): the offline "plan-screen"
    half of what ``run_update_wave()`` used to do in one uninterrupted
    step -- assembles the corpus, reserves/verifies its assembly manifest,
    and reserves/verifies the census-vs-sampling decision, but performs NO
    screening and makes NO LLM call. A human reads this decision's
    ``render()`` output; only a SEPARATE, later ``run_update_wave()``
    invocation may consume it, via ``UpdateWaveCensusDecisionStore.
    require_existing_and_verify()``, which refuses to proceed if no
    decision has been planned yet via this function."""
    combined_records, lineage, dedup_membership = assemble_update_corpus(
        parent=parent, update_spec=update_spec, review_config=cfg, new_raw_records=new_raw_records,
        review_id=review_id,
    )
    manifest_store = UpdateWaveAssemblyManifestStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    manifest_store.reserve_or_verify(
        review_id=review_id, combined_records=combined_records, lineage=lineage, dedup_membership=dedup_membership,
    )
    decision_store = UpdateWaveCensusDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    return decision_store.reserve_or_verify(
        combined_records=combined_records, cfg=cfg, cap_usd=cap_usd, sample_size=sample_size,
        sample_seed=sample_seed, scenario=cost_scenario,
    )


def estimate_update_wave(
    *,
    cfg: ReviewConfig,
    parent: VerifiedReproductionArtifactSet,
    update_spec: ReproductionSpec,
    new_raw_records: Sequence[Record],
    review_id: str,
    cap_usd: float,
    sample_size: int,
    sample_seed: int,
    cost_scenario: "Scenario" = "base",
) -> CensusVsSamplingDecision:
    """Review finding 3 (2026-09-25, third round): ``plan_update_wave()``
    immediately persists an IMMUTABLE decision, but a human cannot know the
    true deduplicated volume/cost for THIS review_run_id until the real
    search artifact and parent are assembled -- and once persisted, trying a
    different ``--cap-usd``/``--sample-size``/``--sample-seed`` conflicts
    with the existing record (``UpdateWaveCensusDecisionIntegrityError``).
    This function is the non-persisting preview: it assembles the real
    corpus (offline, pure -- no cost, no side effect) and prices both
    options via ``decide_census_vs_sampling()``, but writes NOTHING to disk
    (no assembly manifest, no census decision file) and makes no LLM call.
    It may be called any number of times with any candidate cap/sample
    parameters with zero persisted state; once a human is satisfied with a
    specific choice, ``plan_update_wave()`` persists it for real."""
    combined_records, _lineage, _dedup_membership = assemble_update_corpus(
        parent=parent, update_spec=update_spec, review_config=cfg, new_raw_records=new_raw_records,
        review_id=review_id,
    )
    return decide_census_vs_sampling(
        deduplicated_records=list(combined_records), cfg=cfg, cap_usd=cap_usd,
        sample_size=sample_size, sample_seed=sample_seed, scenario=cost_scenario,
    )


__all__ = [
    "ExtractionBatchCapExceededError",
    "RetryPolicyMismatchError",
    "ResolutionBoundMismatchError",
    "UpdateCorpusAssemblyError",
    "UpdateWaveQCConfigError",
    "UpdateWaveAssemblyIntegrityError",
    "UpdateWaveAssemblyManifest",
    "UpdateWaveAssemblyManifestStore",
    "SearchVolumeError",
    "REQUIRED_UPDATE_SEARCH_DATABASES",
    "MAX_RECORDS_PER_ADAPTER",
    "MAX_RECORDS_TOTAL",
    "enforce_search_volume_and_provenance",
    "SearchTemporalWindowError",
    "enforce_record_temporal_window",
    "UpdateWaveSearchArtifactIntegrityError",
    "UpdateWaveSearchArtifact",
    "UpdateWaveSearchArtifactStore",
    "SearchRetrievalError",
    "SearchAccessModeError",
    "enforce_required_access_mode",
    "SearchNotExhaustiveError",
    "enforce_exhaustive_retrieval",
    "SearchAlreadyInProgressError",
    "run_update_wave_search",
    "UpdateWaveExecutionSpecIntegrityError",
    "build_update_execution_spec",
    "UpdateWaveExecutionSpecStore",
    "verify_search_artifact_and_execution_spec_before_screening",
    "run_update_wave",
    "estimate_update_wave",
    "enforce_extraction_batch_cap",
    "verify_client_retry_policy",
    "pinned_and_validated_openai_client_factory",
    "screening_client_factory",
    "screening_qc_client_factory",
    "verify_resolution_bounds",
    "classify_new_records_against_parent",
    "assemble_update_corpus",
    "run_update_wave_screening",
    "compute_cross_source_membership",
    "true_deduplicated_volume",
    "FileSearchCheckpointStore",
    "UpdateWaveCensusDecisionIntegrityError",
    "UpdateWaveCensusDecisionRecord",
    "UpdateWaveCensusDecisionStore",
    "plan_update_wave",
]
