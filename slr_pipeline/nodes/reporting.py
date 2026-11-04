"""
reporting_node -- Final step of the SLR pipeline
(`aggregation_export_reporting_plan_2026-08-25.md` v17).

Generates the complete reporting bundle under
`output_dir/reporting/<review_run_id>/`:
  - `prisma_flow.png`             -- PRISMA diagram, AI-vs-human-vs-rule-
                                     based-decision overlay
  - `methods.md`                  -- auto-drafted methods section
  - `ai_traceability.json`/`.md`  -- aggregated AIUseEvent inventory,
                                     PRISMA-trAIce crosswalk (see
                                     slr_pipeline.ai_traceability_report)
  - `quality_appraisal.csv`/`.json` -- flat/full-fidelity QA export
  - `evidence_map.csv`            -- PRISMA-ScR characteristics table
  - `synthesis_claims.json`/`synthesis.json`/`synthesis.md`
  - `export_manifest.json`        -- the sole commit marker (INV-R15):
                                     unlinked as the very first step,
                                     written last, self-verified on write

Node input keys:  config, review_run_id, selection_decision_ledger_dir,
                   deduplicated_records, full_text_records, prisma_counts,
                   search_log, screening_review_resolutions,
                   full_text_review_queue, screening_qc, synthesis,
                   oversight_mode, automation_origin, extraction_run_id,
                   quality_appraisal_run_id, ft_screening_run_id,
                   ft_screening_execution_fingerprint,
                   ft_screening_artifact_paths
Node output keys: prisma_diagram_path, methods_text, ai_traceability_json_path,
                   ai_traceability_md_path, quality_appraisal_csv_path,
                   quality_appraisal_json_path, evidence_map_csv_path,
                   synthesis_claims_json_path, synthesis_json_path,
                   synthesis_md_path, claim_audit_json_path, claim_audit_md_path,
                   export_manifest_path,
                   prisma_scr_report_md, prisma_scr_conformance,
                   prisma_scr_report_sha256, prisma_scr_conformance_sha256
                   (step 14, [C13] -- both content keys are None/empty when
                   `reference_context` is not set, matching every other
                   conditionally-applicable renderer's own state contract)
"""

from __future__ import annotations

import copy
import csv as csv_module
import dataclasses
import hashlib
import io
import json
import logging
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, TypeAdapter, field_validator, model_validator

from slr_pipeline.state import PRISMACounts, Record, ReviewState, SearchLog, SynthesisOutput
from slr_pipeline.config import ReviewConfig
from slr_pipeline.reference_spec import ReferenceReportingContext
from slr_pipeline.run_identity import resolve_review_run_id
from slr_pipeline.human_resolution import compute_resolution_id

from ft_screening import ScreeningItemResult, ScreeningRunResult

from slr_pipeline.ai_traceability import AIEventIntegrityError, AIUseEvent, AIUseEventStore, classify_automation_origin_optional
from slr_pipeline.extraction.hashing import canonical_json_bytes, file_sha256, sha256_hex
from slr_pipeline.extraction.resolution_journal import resolution_matches_intent, verify_decision_journal_bytes
from slr_pipeline.extraction.resolution_schemas import (
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    FailedCallRecord,
    InterruptedCallRecord,
    QueueSnapshot,
    RejectedSuggestion,
    ValidatedSuggestion,
)
from slr_pipeline.extraction.resolution_suggestions import (
    ReviewPacket,
    load_attempt_records,
    render_review_packet,
    review_packet_json_bytes,
    review_packet_markdown_bytes,
    suggestions_index_bytes,
)
from slr_pipeline.export_manifest import M1ProtocolInfo, ReportingStageEntry, ReportingStageManifest
from slr_pipeline.locators import LocatorGlob, LogicalLocator
from slr_pipeline.prisma_scr import render_prisma_scr_conformance, render_prisma_scr_report_md
from slr_pipeline.selection_ledger import (
    SelectionDecisionStore, _key_digest, _tails_by_key, derive_source_event_id, is_dedup_duplicate_decision,
    is_fulltext_exclusion_decision, is_screening_exclusion_decision, ledger_automation_origin_and_oversight,
)
from slr_pipeline.ai_traceability_report import _render_ai_traceability_dict, _render_ai_traceability_md
from slr_pipeline.continuation.claim_audit import ClaimAuditSources, build_claim_evidence_audit
from slr_pipeline.extraction.results import FieldResult
from slr_pipeline.extraction import verified_bundle
from slr_pipeline.nodes.extraction import load_verified_extraction_resolution_artifact
from slr_pipeline.nodes.quality_appraisal import (
    DocumentAppraisal, QACheckpointStore, VerifiedQARunBundle, load_verified_quality_appraisal_resolution_artifact,
)
from slr_pipeline.nodes.screening import ScreeningBatchResultStore, compute_current_screening_batches, compute_screening_batch_identity
from slr_pipeline.nodes.screening_qc import compute_qc_second_rater_call_identity, compute_qc_eligible_records
from slr_pipeline.nodes.search_strategy_builder import (
    canonical_query_expansion_output, compute_db_translation_call_identity, compute_query_expansion_call_identity,
)
from slr_pipeline.nodes.synthesis import (
    _automation_origin_for_stage, _build_evidence_map, build_synthesis_verification_context,
    canonicalize_corpus_order, verify_synthesis_output_sources,
)

logger = logging.getLogger(__name__)


class ReportingConsistencyError(RuntimeError):
    """A reporting-phase invariant was violated -- durable/graph-state
    source data disagrees with what a re-derivation or reconstruction
    independently produces, a required artifact is missing, or a path
    escapes its own containment root. Propagates out of `reporting_node`
    uncaught (INV-R11), except the one named renderer permitted to fail-
    without-aborting (the PRISMA diagram, §2.11)."""


# ---------------------------------------------------------------------------
# §2.1 -- ReportingSourceSnapshot and its component dataclasses
# ---------------------------------------------------------------------------

def _canonicalize_for_hash(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _canonicalize_for_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(v) for v in value]
    return value


def _hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_canonicalize_for_hash(value))).hexdigest()


@dataclass(frozen=True)
class VerifiedFile:
    """A genuine on-disk file this phase re-loaded and re-verified against
    its own producing store's contract (INV-R2 bucket 1, `durable_reverified`)."""
    model: Any
    path: Path
    file_sha256: str
    internal_hash: str


@dataclass(frozen=True)
class ValidatedStateValue:
    """A deep-copied graph-state value with NO backing file of its own
    (INV-R2 bucket 2, `graph_state`)."""
    model: Any
    content_hash: str


def _state_value(raw: Any) -> ValidatedStateValue:
    """Deep-copies `raw` before hashing/storing -- INV-R22: a live,
    in-place mutation of the ORIGINAL state object after capture must
    never be able to desync the stored hash from the stored model."""
    independent = copy.deepcopy(raw)
    return ValidatedStateValue(model=independent, content_hash=_hash_value(independent))


def _validate_list_of_dicts(value: Any, *, name: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ReportingConsistencyError(f"{name} must be a list of dicts, got {type(value)!r}")
    return value


@dataclass(frozen=True)
class VerifiedComponent:
    """One physical file belonging to a `SourceBundle` -- `internal_hash`
    is the producer's OWN recorded hash for this exact file (a manifest's
    `manifest_hash`/`checkpoint_hashes[...]` entry, or an envelope's own
    `content_hash`), never independently invented here."""
    path: Path
    file_sha256: str
    internal_hash: Optional[str]
    document_id: Optional[str] = None  # set only for a checkpoint component -- step 9: checkpoint files are
    # hash-named on disk (`ExtractionCheckpointStore`/`QACheckpointStore`'s own `_checkpoint_filename`), so
    # `document_id` cannot be recovered from `path` alone by a downstream `slr://…/checkpoint/<run_id>/<document_id>`
    # locator builder -- it must be carried on the component itself.


@dataclass(frozen=True)
class ExtractionReviewArtifacts:
    """Verified files that make extraction adjudication auditable."""

    review_packet_json: Optional[VerifiedComponent] = None
    review_packet_md: Optional[VerifiedComponent] = None
    queue_snapshot: Optional[VerifiedComponent] = None
    suggestions: Tuple[VerifiedComponent, ...] = ()
    suggestions_index: Optional[VerifiedComponent] = None
    decision_journal: Optional[VerifiedComponent] = None


@dataclass(frozen=True)
class ReproductionEvaluationArtifacts:
    """Verified, mutually bound files for one reproduction attempt."""

    reproduction_spec: VerifiedComponent
    gold_standard: VerifiedComponent
    stage_corpus: VerifiedComponent
    pipeline_evaluation: VerifiedComponent
    gate_report: VerifiedComponent
    adjudication_journal: VerifiedComponent


@dataclass(frozen=True)
class SourceBundle:
    """A complete extraction- or quality-appraisal-run bundle: the raw,
    already-verified artifact dict (`raw`) plus every physical component
    file that backs it, individually re-verified (INV-R23/INV-R24)."""
    raw: Dict[str, Any]
    manifest: VerifiedComponent
    checkpoints: List[VerifiedComponent]
    resolutions: Optional[VerifiedComponent]
    confirmations: Optional[VerifiedComponent]
    computed_content_hash: str
    run_id: str  # step 9: needed to build `slr://extraction/…/<run_id>` / `slr://quality_appraisal/…/<run_id>` locators
    extraction_review: Optional[ExtractionReviewArtifacts] = None
    reproduction_evaluation: Optional[ReproductionEvaluationArtifacts] = None


@dataclass(frozen=True)
class VerifiedProjection:
    """`SynthesisOutput`, re-verified via `verify_synthesis_output_sources()`
    (INV-R2 bucket 1) -- has no backing file of its own (it lives inside
    `ReviewState`), so it is content-hashed rather than file-hashed."""
    model: "SynthesisOutput"
    content_hash: str


# ---------------------------------------------------------------------------
# Evidence map -- validated shape (X3), full reconstruction-and-compare
# (W1/X1/X2/Y2)
# ---------------------------------------------------------------------------

_NonNegStrictInt = Annotated[StrictInt, Field(ge=0)]
# X3: extraction's own charting projection ALWAYS stores ExtractedValue.value
# -- the raw string -- never a typed numeric/boolean value, regardless of the
# field's declared schema ValueKind (confirmed directly, extraction.py:
# `raw[field_id] = values[0] if len(values) == 1 else values`, where each
# `value` is itself a plain string).
EvidenceCellValue = Union[StrictStr, List[StrictStr]]


class EvidenceMapRow(BaseModel):
    """The real, exact shape of one `_build_evidence_map()` (`nodes/synthesis.py`)
    `characteristics_table` row -- `extra="forbid"` so a fabricated/renamed
    field is rejected. `year` is `StrictInt`-only (matches `Record.year:
    Optional[int]` exactly -- this comes from `Record`, not the charting
    projection, so it is genuinely numeric). `country`/`study_design` are
    `StrictStr`-only -- the charting projection never places a typed value
    there. `population`/`concept`/`context` additionally accept
    `List[StrictStr]` for a genuinely multi-valued field -- never a bare
    number/boolean/dict, none of which the projection ever actually produces."""
    model_config = ConfigDict(extra="forbid")
    citation: str
    year: Optional[StrictInt] = None
    country: Optional[StrictStr] = None
    study_design: Optional[StrictStr] = None
    population: Optional[EvidenceCellValue] = None
    concept: Optional[EvidenceCellValue] = None
    context: Optional[EvidenceCellValue] = None


class EvidenceMapModel(BaseModel):
    """Real, validated shape for `_build_evidence_map()`'s actual return.
    Kept deliberately CHEAP here (n_studies >= 1, EXACTLY
    len(characteristics_table), every count StrictInt/non-negative) since
    the authoritative content proof is the capture-time full
    reconstruction-and-compare, not a second, error-prone hand-rolled
    recomputation inside this model."""
    model_config = ConfigDict(extra="forbid")
    n_studies: StrictInt = Field(ge=1)
    by_year: Dict[str, _NonNegStrictInt]
    by_study_design: Dict[str, _NonNegStrictInt]
    by_country: Dict[str, _NonNegStrictInt]
    characteristics_table: List[EvidenceMapRow]

    @model_validator(mode="after")
    def _row_count_conserves(self) -> "EvidenceMapModel":
        if self.n_studies != len(self.characteristics_table):
            raise ValueError(f"n_studies ({self.n_studies}) != len(characteristics_table) ({len(self.characteristics_table)})")
        return self


def _reconstruct_expected_evidence_map(
    *, full_text_records: List["Record"], extraction_raw: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Reorders `full_text_records` into `charted_data`'s own canonical
    order via `canonicalize_corpus_order()` -- the SAME function
    `synthesis_node` itself uses -- THEN calls the SAME, UNMODIFIED
    `_build_evidence_map()`. Returns `None` if there is no extraction data
    to build from."""
    if not extraction_raw:
        return None
    charted = list(extraction_raw.get("resolved_projection", []))
    ordered_records = canonicalize_corpus_order(
        full_text_records=full_text_records, charted=charted,
        error_prefix="reporting evidence-map reconstruction",
    )
    charted_by_id: Dict[str, Dict[str, Any]] = {c["record_id"]: c.get("raw_extraction", {}) for c in charted}
    return _build_evidence_map(ordered_records, charted_by_id)


_ANY_TYPE_ADAPTER = TypeAdapter(Any)


def _pydantic_json_mode(value: Any) -> Any:
    """Converts `value` into pydantic's OWN JSON-mode representation, using
    the actual mechanism (`TypeAdapter(Any).dump_python(value, mode="json")`)
    `SynthesisOutput.model_dump(mode="json")` itself already relies on for
    nested `Any`-typed content -- not an approximation of it. `canonical_
    json_bytes()`/`_canonicalize_for_hash()` stringify a dict key via
    Python's own `str()`, a THIRD, different convention from both stdlib
    `json.dumps` and pydantic's own JSON-mode key coercion -- using the
    real mechanism directly means this comparison stays correct even for a
    key type neither side has been explicitly tested against."""
    return _ANY_TYPE_ADAPTER.dump_python(value, mode="json")


# ---------------------------------------------------------------------------
# §2.13 -- report-input validation for screening_qc's own graph state
# ---------------------------------------------------------------------------

class ReportingIRRState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_attempted: StrictInt = Field(ge=0)
    n_valid_pairs: StrictInt = Field(ge=0)
    n_failed: StrictInt = Field(ge=0)
    pct_include_avg: Optional[float] = None
    pct_agreement: Optional[float] = None
    cohens_kappa: Optional[float] = None
    pabak: Optional[float] = None
    gwet_ac1: Optional[float] = None
    level: Optional[str] = None
    note: Optional[str] = None
    sampled_record_ids: List[str] = Field(default_factory=list)
    # Codex round-5 review, finding 2: the exact AIUseEvent.event_id THIS
    # invocation published per sampled record (screening_qc.py's own
    # `_compute_irr()`) -- lets reporting's own cross-check verify against
    # this specific invocation's events, not every historical
    # screening_qc_second_rater event ever published under this
    # review_run_id.
    producing_event_ids: List[str] = Field(default_factory=list)

    @field_validator("sampled_record_ids")
    @classmethod
    def _unique(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("sampled_record_ids must not contain duplicates")
        return v

    @field_validator("producing_event_ids")
    @classmethod
    def _unique_events(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("producing_event_ids must not contain duplicates")
        return v

    @model_validator(mode="after")
    def _counts_conserve(self) -> "ReportingIRRState":
        if self.n_valid_pairs + self.n_failed != self.n_attempted:
            raise ValueError(f"n_valid_pairs ({self.n_valid_pairs}) + n_failed ({self.n_failed}) != n_attempted ({self.n_attempted})")
        if self.n_attempted > 0 and len(self.sampled_record_ids) != self.n_attempted:
            raise ValueError(f"sampled_record_ids has {len(self.sampled_record_ids)} entries, expected n_attempted={self.n_attempted}")
        if self.n_attempted > 0 and len(self.producing_event_ids) != self.n_attempted:
            raise ValueError(f"producing_event_ids has {len(self.producing_event_ids)} entries, expected n_attempted={self.n_attempted}")
        if self.n_attempted == 0 and self.producing_event_ids:
            raise ValueError("producing_event_ids must be empty when n_attempted == 0")
        return self


class ReportingScreeningQCState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    irr: ReportingIRRState
    saturation: Optional[Dict[str, Any]] = None


def _validate_screening_qc_state(raw: Optional[Dict[str, Any]]) -> Optional[ReportingScreeningQCState]:
    if raw is None:
        return None
    try:
        return ReportingScreeningQCState.model_validate(raw)
    except Exception as exc:
        raise ReportingConsistencyError(f"state['screening_qc'] failed validation: {exc}") from exc


# ---------------------------------------------------------------------------
# §2.6 -- FT_SCREENING native provenance validation (INV-R17)
# ---------------------------------------------------------------------------

_ALLOWED_FT_KEYS = {"run_result_json", "item_results_jsonl", "ledger_jsonl"}


def capture_ft_native_provenance(state: ReviewState, cfg: ReviewConfig) -> Optional[Dict[str, Any]]:
    run_id = state.get("ft_screening_run_id")
    fp = state.get("ft_screening_execution_fingerprint")
    paths = state.get("ft_screening_artifact_paths")
    present = [x is not None for x in (run_id, fp, paths)]
    if not any(present):
        return None
    if not all(present):
        raise ReportingConsistencyError(
            "ft_screening_run_id/execution_fingerprint/artifact_paths must be all-set or all-None")
    if set(paths.keys()) != _ALLOWED_FT_KEYS:
        raise ReportingConsistencyError(f"ft_screening_artifact_paths keys {set(paths.keys())} != {_ALLOWED_FT_KEYS}")
    resolved = {k: Path(v).resolve() for k, v in paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ReportingConsistencyError("ft_screening_artifact_paths contains duplicate/aliased paths")
    output_root = cfg.ensure_output_dir().resolve()
    for key, p in resolved.items():
        if not p.is_file():
            raise ReportingConsistencyError(f"ft_screening_artifact_paths[{key!r}]={p} is not an existing regular file")
        if not p.is_relative_to(output_root):
            raise ReportingConsistencyError(f"ft_screening_artifact_paths[{key!r}]={p} is outside {output_root}")
    # Finding 7 (Codex round-1 review): store a DETACHED copy -- resolved,
    # absolute path strings computed from `resolved` (already validated
    # above) -- never the original `paths` dict/values pulled straight out
    # of live `state`. Without this, a later in-place mutation of
    # `state["ft_screening_artifact_paths"]` could silently desync this
    # already-captured snapshot value from the file hashes already
    # computed against it.
    detached_paths = {k: str(p) for k, p in resolved.items()}
    # Codex round-4 review, finding 4, P1: the full_text_eligibility
    # ledger-completeness check (compute_selection_splits) needs an
    # AUTHORITATIVE, genuinely-independent expected record-ID set --
    # neither the ledger itself (circular) nor `full_text_records` (only
    # "include" outcomes) nor `included_records` (pre-acquisition, doesn't
    # know which records got a confirmed PDF) can serve. FT_SCREENING's own
    # `item_results.jsonl` -- a "materialized-current-view" (one line per
    # item ever submitted to the one real FT_SCREENING request this run,
    # ft_screening/store.py's own `write_item_result()`) -- is written by a
    # DIFFERENT subsystem, already byte-hash-verified above. This is
    # precisely the set of records `fulltext.py`'s own acquisition loop
    # found a confirmed PDF for (`full_text_status == "parsed"`) and
    # therefore submitted for eligibility assessment -- i.e. exactly
    # `prisma.assessed_full_text`'s own record-level membership, not
    # merely its count.
    #
    # Codex round-5 review, finding 1, P1: an earlier version of this
    # function pulled `document_id` off an untyped `json.loads()` dict per
    # line -- accepting arbitrary bytes/shapes as "the item result" despite
    # this artifact's own `classification` (below) explicitly disclaiming
    # semantic re-verification. Fixed to reuse FT_SCREENING's OWN real,
    # fully-validating pydantic models (`ScreeningRunResult`/
    # `ScreeningItemResult`, the exact types `fulltext.py`'s own
    # `_run_full_text_eligibility()` receives from `run_ft_screening()`/
    # `store.py`'s own `_read_item_results()`) -- never a second, weaker,
    # hand-rolled parser -- and to cross-verify `run_result.json`'s own
    # `run_id`/`execution_fingerprint` against `state["ft_screening_run_id"]`/
    # `state["ft_screening_execution_fingerprint"]`, AND `item_results.
    # jsonl`'s own (document_id, report_id) item set against
    # `run_result.json`'s own `items` list -- two independently-written
    # artifacts of the SAME run must agree on exactly which items exist
    # before either is trusted as the authoritative confirmed-record set.
    # `record_to_screening_item()` (adapters/ft_screening_adapter.py) sets
    # `document_id == report_id == record.id`; `.report_id` is used here
    # (not `.document_id`) because it is FT_SCREENING's own documented join
    # key back to the originating `Record` (see
    # `ChapterFullTextOutcome.record_id`'s own docstring, which
    # `fulltext.py` itself relies on for the identical join).
    run_result_path = resolved["run_result_json"]
    try:
        run_result = ScreeningRunResult.model_validate_json(run_result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportingConsistencyError(
            f"ft_screening_artifact_paths['run_result_json'] does not contain a valid "
            f"ScreeningRunResult: {exc}") from exc
    if run_result.run_id != run_id:
        raise ReportingConsistencyError(
            f"run_result.json's own run_id {run_result.run_id!r} does not match "
            f"state['ft_screening_run_id'] {run_id!r}")
    if run_result.execution_fingerprint != fp:
        raise ReportingConsistencyError(
            f"run_result.json's own execution_fingerprint {run_result.execution_fingerprint!r} does not "
            f"match state['ft_screening_execution_fingerprint'] {fp!r}")

    item_results_path = resolved["item_results_jsonl"]
    item_results: List[ScreeningItemResult] = []
    for line in item_results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item_results.append(ScreeningItemResult.model_validate_json(line))
        except Exception as exc:
            raise ReportingConsistencyError(
                f"ft_screening_artifact_paths['item_results_jsonl'] contains a line that is not a "
                f"valid ScreeningItemResult: {exc}") from exc

    report_ids = [item.report_id for item in item_results]
    if len(set(report_ids)) != len(report_ids):
        raise ReportingConsistencyError(
            "ft_screening_artifact_paths['item_results_jsonl'] contains duplicate report_id(s) -- "
            "cannot derive an authoritative confirmed-record set from it")

    item_pairs = {(item.document_id, item.report_id) for item in item_results}
    run_result_pairs = {(item.document_id, item.report_id) for item in run_result.items}
    if item_pairs != run_result_pairs:
        raise ReportingConsistencyError(
            "item_results.jsonl's own (document_id, report_id) item set does not match run_result."
            f"json's own items -- missing_from_item_results={sorted(run_result_pairs - item_pairs)!r}, "
            f"unexpected_in_item_results={sorted(item_pairs - run_result_pairs)!r}")

    confirmed_record_ids = sorted(report_ids)
    return {
        "ft_screening_run_id": run_id, "ft_screening_execution_fingerprint": fp,
        "ft_screening_artifact_paths": detached_paths,
        "artifact_file_hashes": {k: file_sha256(p) for k, p in resolved.items()},
        "confirmed_record_ids": sorted(confirmed_record_ids),
        "classification": "state_referenced_byte_hashed_not_semantically_reverified",
    }


# ---------------------------------------------------------------------------
# §2.9 -- ReportingConfigProjection (the ONLY config-shaped value any
# renderer -- or the completed snapshot -- ever sees; no ReviewConfig,
# no openai_api_key, anywhere downstream of this)
# ---------------------------------------------------------------------------

class ReportingConfigProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_type: str
    research_questions: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    screening_llm_model: str
    generate_evidence_map: bool
    run_bertopic: bool
    n_topics: int
    quality_appraisal_checklist_id: Optional[str]
    search_query: str
    population: Optional[str]
    concept: Optional[str]
    context: Optional[str]
    intervention: Optional[str]
    comparator: Optional[str]
    outcome: Optional[str]
    protocol_uri: Optional[str]
    protocol_hash: Optional[str]
    protocol_registration_id: Optional[str]
    ai_use_prespecified: Optional[bool]
    protocol_deviations: List[str]
    data_governance_declaration: Optional[str]
    # Step 14 ([C13], PRISMA-ScR renderers) additions -- a minimal extension,
    # only the fields the 22-item matrix actually needs and no others (items
    # 7, 9, 13, 22 respectively):
    search_databases: List[str]
    search_as_of_date: Optional[str]
    screening_uncertain_threshold: float
    generate_claim_backed_results: bool
    refine_batch_size: int
    funding_statement: Optional[str]

    @classmethod
    def from_config(cls, cfg: ReviewConfig) -> "ReportingConfigProjection":
        return cls(
            review_type=cfg.review_type, research_questions=cfg.research_questions,
            inclusion_criteria=cfg.inclusion_criteria, exclusion_criteria=cfg.exclusion_criteria,
            screening_llm_model=cfg.screening.llm_model,
            generate_evidence_map=cfg.synthesis.generate_evidence_map, run_bertopic=cfg.synthesis.run_bertopic,
            n_topics=cfg.synthesis.n_topics,
            quality_appraisal_checklist_id=(cfg.quality_appraisal.checklist_id if cfg.review_type != "scoping" else None),
            search_query=cfg.search.query,
            population=cfg.population, concept=cfg.concept, context=cfg.context,
            intervention=cfg.intervention, comparator=cfg.comparator, outcome=cfg.outcome,
            protocol_uri=cfg.protocol_uri, protocol_hash=cfg.protocol_hash,
            protocol_registration_id=cfg.protocol_registration_id, ai_use_prespecified=cfg.ai_use_prespecified,
            protocol_deviations=cfg.protocol_deviations, data_governance_declaration=cfg.data_governance_declaration,
            search_databases=list(cfg.search.databases), search_as_of_date=cfg.search.as_of_date,
            screening_uncertain_threshold=cfg.screening.uncertain_threshold,
            generate_claim_backed_results=cfg.synthesis.generate_claim_backed_results,
            refine_batch_size=cfg.synthesis.refine_batch_size,
            funding_statement=cfg.funding_statement,
        )


# ---------------------------------------------------------------------------
# §2.1 -- ReportingSourceSnapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportingSourceSnapshot:
    review_run_id: str
    config_projection: ReportingConfigProjection
    config_projection_hash: str
    raw_records: ValidatedStateValue   # {r["id"] for r in .model} is the authoritative expected
                                          # record-ID set for the "dedup" ledger stage (Codex round-4
                                          # review, finding 4) -- dedup_node emits exactly one ledger
                                          # decision per raw record, unconditionally.
    screening_records: List["Record"]
    full_text_records: List["Record"]
    screened_records_full: List["Record"]   # Codex round-10 review, finding 3: captured ONCE here (was
                                               # previously read live/ad-hoc from state at cross-check time)
                                               # -- same INV-R1/INV-R19 "captured once, immutable" reasoning
                                               # as every other graph-state value on this snapshot.
    prisma: ValidatedStateValue
    search_log: ValidatedStateValue
    screening_review_resolutions: ValidatedStateValue
    full_text_review_resolutions: ValidatedStateValue
    full_text_review_queue: ValidatedStateValue
    screening_qc_state: Optional[ReportingScreeningQCState]
    oversight_mode_state: Dict[str, str]
    automation_origin_state: Dict[str, str]
    events: List[VerifiedFile]
    ledger_records: List[VerifiedFile]
    current_screening_batch_results: List[VerifiedFile]
    extraction: Optional[SourceBundle]
    quality_appraisal: Optional[SourceBundle]
    synthesis_output: VerifiedProjection
    evidence_map: Optional[EvidenceMapModel]
    ft_native_provenance: Optional[Dict[str, Any]]
    # Track 2 plan §9/§10.10 field #24: the reference run's own reporting
    # context (`cfg.reference_reporting_context`, validated here). None in
    # production -- only the Track 2 reference run sets `cfg.reference_
    # reporting_context`. `capture_reporting_snapshot` cannot build this
    # itself (it only ever sees `state`/`cfg`/`run_id`), so it is carried
    # through unchanged from config, the same way every other config-
    # sourced field on this snapshot is.
    reference_context: Optional[ReferenceReportingContext] = None
    reproduction_evaluation: Optional[ReproductionEvaluationArtifacts] = None


# ---------------------------------------------------------------------------
# One ordered, single-read (within this function) component-loading
# primitive (INV-R25)
# ---------------------------------------------------------------------------

def _load_component(
    path: Path, *, run_root: Path, seen: set, label: str, required: bool,
) -> Tuple[Optional[VerifiedComponent], Optional[bytes]]:
    """Containment -> existence -> is_file -> uniqueness -> read bytes ONCE,
    all within THIS function's own execution -- says nothing about, and
    does not eliminate, reads a pre-existing verified loader already
    performed earlier, independently, to prove the artifact correct in
    the first place (INV-R19)."""
    resolved = path.resolve()
    if not resolved.is_relative_to(run_root):
        raise ReportingConsistencyError(f"{label} path {resolved} escapes its own run_dir {run_root}")
    if not resolved.exists():
        if required:
            raise ReportingConsistencyError(f"{label} expected at {path} but not found")
        return None, None
    if not resolved.is_file():
        raise ReportingConsistencyError(f"{label} path {resolved} exists but is not a regular file")
    if resolved in seen:
        raise ReportingConsistencyError(f"{label} path {resolved} duplicates/aliases another component path")
    seen.add(resolved)
    raw_bytes = resolved.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    return VerifiedComponent(path=path, file_sha256=file_hash, internal_hash=None), raw_bytes


def _load_extraction_review_artifacts(
    *, run_root: Path, seen: set, bundle: Any, schema: Any,
    events: List[AIUseEvent], run_id: str, resolutions_raw: Optional[Dict[str, Any]] = None,
) -> Optional[ExtractionReviewArtifacts]:
    """Load and cross-verify the Section 3.7 extraction-review layer."""
    packet_json_path = run_root / "extraction_review_packet.json"
    packet_md_path = run_root / "extraction_review_packet.md"
    snapshot_path = run_root / "review_queue_snapshot.json"
    suggestions_dir = run_root / "extraction_resolution_suggestions"
    index_path = run_root / "extraction_resolution_suggestions.json"
    journal_path = run_root / "extraction_resolution_decisions.jsonl"
    if not any(p.exists() for p in (
        packet_json_path, packet_md_path, snapshot_path, suggestions_dir, index_path, journal_path,
    )):
        return None

    packet_component, packet_bytes = _load_component(
        packet_json_path, run_root=run_root, seen=seen,
        label="extraction review packet JSON", required=True,
    )
    markdown_component, markdown_bytes = _load_component(
        packet_md_path, run_root=run_root, seen=seen,
        label="extraction review packet Markdown", required=True,
    )
    snapshot_component, snapshot_bytes = _load_component(
        snapshot_path, run_root=run_root, seen=seen,
        label="extraction review queue snapshot", required=True,
    )
    assert packet_component and packet_bytes is not None
    assert markdown_component and markdown_bytes is not None
    assert snapshot_component and snapshot_bytes is not None
    try:
        packet = ReviewPacket.model_validate_json(packet_bytes)
        queue_snapshot = QueueSnapshot.model_validate_json(snapshot_bytes)
        rerendered = render_review_packet(bundle, schema, queue_snapshot)
    except Exception as exc:
        raise ReportingConsistencyError(f"invalid extraction review packet/snapshot: {exc}") from exc
    if review_packet_json_bytes(packet) != packet_bytes:
        raise ReportingConsistencyError("extraction_review_packet.json is not canonical JSON")
    if review_packet_json_bytes(rerendered) != packet_bytes:
        raise ReportingConsistencyError(
            "extraction_review_packet.json does not re-derive from its verified checkpoints and queue snapshot"
        )
    if review_packet_markdown_bytes(packet) != markdown_bytes:
        raise ReportingConsistencyError(
            "extraction_review_packet.md does not re-render from extraction_review_packet.json"
        )
    if canonical_json_bytes(queue_snapshot.model_dump(mode="json")) != snapshot_bytes:
        raise ReportingConsistencyError("review_queue_snapshot.json is not canonical JSON")

    attempts = load_attempt_records(suggestions_dir)
    packet_entries = {(entry.record_id, entry.field_id): entry for entry in packet.entries}
    events_by_id = {event.event_id: event for event in events}
    suggestion_components: List[VerifiedComponent] = []
    records_by_id: Dict[str, Tuple[Any, Path]] = {}
    for path, record in attempts:
        if record.extraction_run_id != run_id:
            raise ReportingConsistencyError(
                f"{path}: attempt extraction_run_id {record.extraction_run_id!r} != {run_id!r}"
            )
        entry = packet_entries.get((record.record_id, record.field_id))
        if entry is None:
            raise ReportingConsistencyError(f"{path}: attempt has no matching review-packet entry")
        if not isinstance(record, InterruptedCallRecord):
            actual_field_hash = sha256_hex(entry.field_result.model_dump(mode="json"))
            if record.original_field_result_hash != actual_field_hash:
                raise ReportingConsistencyError(
                    f"{path}: original_field_result_hash does not match bundled review packet"
                )
        if isinstance(record, ValidatedSuggestion):
            unknown_evidence = set(record.evidence_ids) - set(entry.evidence_index)
            unknown_candidates = (
                set(record.candidate_ids_merged)
                | set(record.candidate_ids_retained)
                | set(record.candidate_ids_rejected)
            ) - {candidate.candidate_id for candidate in entry.candidates}
            if unknown_evidence or unknown_candidates:
                raise ReportingConsistencyError(
                    f"{path}: suggestion references unknown evidence/candidate IDs"
                )
        elif isinstance(record, RejectedSuggestion) and isinstance(record.raw_response, dict):
            referenced_evidence = set(record.raw_response.get("evidence_ids") or [])
            referenced_candidates: set[str] = set()
            for name in ("candidate_ids_merged", "candidate_ids_retained", "candidate_ids_rejected"):
                referenced_candidates.update(record.raw_response.get(name) or [])
            if (referenced_evidence - set(entry.evidence_index) or
                    referenced_candidates - {candidate.candidate_id for candidate in entry.candidates}):
                raise ReportingConsistencyError(
                    f"{path}: rejected suggestion references unknown evidence/candidate IDs"
                )
        event = events_by_id.get(record.ai_use_event_id)
        if isinstance(record, InterruptedCallRecord):
            if event is not None:
                raise ReportingConsistencyError(
                    f"{path}: interrupted attempt unexpectedly has a published AIUseEvent"
                )
        else:
            if event is None:
                raise ReportingConsistencyError(f"{path}: attempt AIUseEvent is absent")
            expected_status = "failure" if isinstance(record, FailedCallRecord) else "success"
            draft = record.ai_use_event_draft
            if any((
                event.stage != EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
                event.work_item_id != draft.work_item_id,
                event.attempt != record.attempt,
                event.model != draft.model,
                event.call_status != expected_status,
            )):
                raise ReportingConsistencyError(f"{path}: attempt does not match its published AIUseEvent")
        component, _ = _load_component(
            path, run_root=run_root, seen=seen, label=f"extraction suggestion {path.name}", required=True,
        )
        assert component is not None
        locator_subid = path.name.split(".json", 1)[0].replace(".interrupted_unknown", "")
        suggestion_components.append(dataclasses.replace(
            component, internal_hash=record.self_hash, document_id=locator_subid,
        ))
        records_by_id[record.attempt_record_id] = (record, path)

    index_component, index_bytes = _load_component(
        index_path, run_root=run_root, seen=seen,
        label="extraction suggestions index", required=bool(attempts) or index_path.exists(),
    )
    rebuilt_index = suggestions_index_bytes(attempts, extraction_run_id=run_id)
    if index_bytes is not None and index_bytes != rebuilt_index:
        raise ReportingConsistencyError(
            "extraction_resolution_suggestions.json does not byte-match its verified attempt files"
        )
    if attempts and index_bytes is None:
        raise ReportingConsistencyError("suggestion attempt files exist without their generated index")

    journal_component, journal_bytes = _load_component(
        journal_path, run_root=run_root, seen=seen,
        label="extraction resolution decision journal", required=False,
    )
    if journal_bytes is not None:
        try:
            decisions = verify_decision_journal_bytes(journal_bytes)
        except Exception as exc:
            raise ReportingConsistencyError(f"invalid extraction resolution decision journal: {exc}") from exc
        for decision in decisions:
            if decision.run_id != run_id:
                raise ReportingConsistencyError(
                    f"decision {decision.journal_id!r} belongs to run {decision.run_id!r}, not {run_id!r}"
                )
            if decision.suggestion_id is not None:
                bound = records_by_id.get(decision.suggestion_id)
                if bound is None:
                    raise ReportingConsistencyError(
                        f"decision {decision.journal_id!r} does not bind a bundled suggestion attempt"
                    )
                suggestion, suggestion_path = bound
                expected_attempt_path = suggestion_path.relative_to(run_root).as_posix()
                if (
                    not isinstance(suggestion, (ValidatedSuggestion, RejectedSuggestion))
                    or suggestion.suggestion_id != decision.suggestion_id
                    or suggestion.self_hash != decision.self_hash
                    or suggestion.record_id != decision.record_id
                    or suggestion.field_id != decision.field_id
                    or expected_attempt_path != decision.attempt_file_path
                    or suggestion.ai_use_event_draft.model != decision.model
                    or suggestion.ai_use_event_draft.prompt_template_version
                    != decision.prompt_template_version
                    or suggestion.ai_use_event_id != decision.call_event_id
                ):
                    raise ReportingConsistencyError(
                        f"decision {decision.journal_id!r} does not bind a bundled suggestion attempt"
                    )
            if decision.status == "applied":
                candidates = [row for row in (resolutions_raw or {}).get("resolutions", []) if
                              row.get("record_id") == decision.record_id and
                              row.get("field_id") == decision.field_id]
                if len(candidates) != 1 or not resolution_matches_intent(candidates[0], decision, packet):
                    raise ReportingConsistencyError(
                        f"applied decision {decision.journal_id!r} does not match extraction_resolutions.json"
                    )

    return ExtractionReviewArtifacts(
        review_packet_json=dataclasses.replace(packet_component, internal_hash=packet_component.file_sha256),
        review_packet_md=dataclasses.replace(markdown_component, internal_hash=markdown_component.file_sha256),
        queue_snapshot=dataclasses.replace(snapshot_component, internal_hash=snapshot_component.file_sha256),
        suggestions=tuple(suggestion_components),
        suggestions_index=(dataclasses.replace(index_component, internal_hash=index_component.file_sha256)
                           if index_component else None),
        decision_journal=(dataclasses.replace(journal_component, internal_hash=journal_component.file_sha256)
                          if journal_component else None),
    )


def _load_extraction_bundle(
    state: ReviewState, cfg: ReviewConfig, *, events: Optional[List[AIUseEvent]] = None,
) -> Optional[SourceBundle]:
    run_id = state.get("extraction_run_id")
    if not run_id:
        return None
    raw = load_verified_extraction_resolution_artifact(state)
    if raw is None:
        return None
    store, bundle, schema, _schema_hash = verified_bundle.load_and_verify_bundle(state, run_id)
    manifest = bundle.manifest
    run_root = store.run_dir.resolve()
    seen: set = set()

    manifest_component, _ = _load_component(store.manifest_path(), run_root=run_root, seen=seen, label="extraction manifest", required=True)
    manifest_component = dataclasses.replace(manifest_component, internal_hash=manifest.manifest_hash)

    checkpoints = []
    for document_id in sorted(manifest.document_ids):
        component, _ = _load_component(store.checkpoint_path(document_id), run_root=run_root, seen=seen,
                                        label=f"extraction checkpoint[{document_id}]", required=True)
        checkpoints.append(dataclasses.replace(component, internal_hash=manifest.checkpoint_hashes[document_id],
                                                document_id=document_id))

    resolutions_component, _ = _load_component(store.run_dir / "extraction_resolutions.json", run_root=run_root,
                                                seen=seen, label="extraction resolutions", required=True)
    resolutions_component = dataclasses.replace(resolutions_component, internal_hash=raw["content_hash"])

    review_events = events
    if review_events is None:
        review_events = [vf.model for vf in _load_verified_events(AIUseEventStore(
            output_dir=cfg.ensure_output_dir(), review_run_id=state["review_run_id"],
        ))]
    review_artifacts = _load_extraction_review_artifacts(
        run_root=run_root, seen=seen, bundle=bundle, schema=schema,
        events=review_events, run_id=run_id,
        resolutions_raw=raw,
    )
    return SourceBundle(raw=raw, manifest=manifest_component, checkpoints=checkpoints,
                        resolutions=resolutions_component, confirmations=None,
                        computed_content_hash=_hash_value(raw), run_id=run_id,
                        extraction_review=review_artifacts)


def _load_qa_bundle(state: ReviewState, cfg: ReviewConfig) -> Optional[SourceBundle]:
    qa_run_id = state.get("quality_appraisal_run_id")
    if not qa_run_id:
        return None
    raw = load_verified_quality_appraisal_resolution_artifact(state)
    if raw is None:
        return None
    store = QACheckpointStore(output_dir=cfg.ensure_output_dir(), qa_run_id=qa_run_id)
    bundle = VerifiedQARunBundle.from_store(store)
    bundle.verify()
    manifest = bundle.manifest
    if raw["manifest_hash"] != manifest.manifest_hash:
        raise ReportingConsistencyError(
            f"QA resolution artifact's manifest_hash {raw['manifest_hash']!r} does not match the "
            f"freshly-reloaded manifest's own {manifest.manifest_hash!r}")
    run_root = store.run_dir.resolve()
    seen: set = set()

    manifest_component, _ = _load_component(store.manifest_path(), run_root=run_root, seen=seen, label="QA manifest", required=True)
    manifest_component = dataclasses.replace(manifest_component, internal_hash=manifest.manifest_hash)

    checkpoints = []
    for record_id in sorted(manifest.document_ids):
        component, _ = _load_component(store.checkpoint_path(record_id), run_root=run_root, seen=seen,
                                        label=f"QA checkpoint[{record_id}]", required=True)
        checkpoints.append(dataclasses.replace(component, internal_hash=manifest.checkpoint_hashes[record_id],
                                                document_id=record_id))

    resolutions_component, res_bytes = _load_component(store.resolutions_path(), run_root=run_root, seen=seen, label="QA resolutions", required=False)
    if resolutions_component is not None:
        resolutions_component = dataclasses.replace(resolutions_component, internal_hash=json.loads(res_bytes)["content_hash"])
    confirmations_component, conf_bytes = _load_component(store.confirmations_path(), run_root=run_root, seen=seen, label="QA confirmations", required=False)
    if confirmations_component is not None:
        confirmations_component = dataclasses.replace(confirmations_component, internal_hash=json.loads(conf_bytes)["content_hash"])

    return SourceBundle(raw=raw, manifest=manifest_component, checkpoints=checkpoints,
                         resolutions=resolutions_component, confirmations=confirmations_component,
                         computed_content_hash=_hash_value(raw), run_id=qa_run_id)


def _load_verified_events(event_store: AIUseEventStore) -> List[VerifiedFile]:
    events = event_store.load_all()  # already self-verified (AIUseEventStore's own contract)
    out = []
    for event in events:
        p = event_store.event_path(event.event_id)
        out.append(VerifiedFile(model=event, path=p, file_sha256=file_sha256(p), internal_hash=event.content_hash))
    return out


def _load_verified_ledger_records(ledger_store: SelectionDecisionStore) -> List[VerifiedFile]:
    records = ledger_store.load_validated()  # already self-verified + chain-structure-validated
    out = []
    for record in records:
        p = ledger_store.record_path(record_id=record.record_id, stage=record.stage, source_event_id=record.source_event_id)
        out.append(VerifiedFile(model=record, path=p, file_sha256=file_sha256(p), internal_hash=record.content_hash))
    return out


def _load_current_screening_batches(
    *, records: List["Record"], cfg: ReviewConfig, review_run_id: str, event_store: AIUseEventStore,
) -> List[VerifiedFile]:
    batch_store = ScreeningBatchResultStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id, event_store=event_store)
    out = []
    for batch in compute_current_screening_batches(records=records, cfg=cfg):
        artifact_id, fingerprint, record_ids = compute_screening_batch_identity(batch=batch, cfg=cfg, review_run_id=review_run_id)
        result = batch_store.load(artifact_id, expected_batch_fingerprint=fingerprint, expected_record_ids=record_ids)
        if result is None:
            raise ReportingConsistencyError(f"expected ScreeningBatchResult {artifact_id} not found")
        p = batch_store.artifact_path(artifact_id)
        out.append(VerifiedFile(model=result, path=p, file_sha256=file_sha256(p), internal_hash=result.compute_content_hash()))
    return out


# ---------------------------------------------------------------------------
# §2.6/T2 -- BERTopic consistency guard + rendered note (5 mutually-
# exclusive branches, INV-R28)
# ---------------------------------------------------------------------------

def _validate_bertopic_consistency(
    *, run_bertopic: bool, n_records: int, excluded: int, topics: List[Dict[str, Any]],
) -> None:
    """Matches `synthesis.py`'s ACTUAL semantics precisely -- `excluded` is
    only ever incremented inside `if cfg.synthesis.run_bertopic and
    records:`, so a disabled or empty-corpus run's counter is EXACTLY 0 by
    construction, never merely "small". `excluded`'s own type/range is
    primarily enforced by `SynthesisOutput.bertopic_excluded_empty_evidence_count`'s
    own `StrictInt`/`ge=0` field -- by the time this function runs, a
    bool/numeric-string has ALREADY been rejected at model-construction
    time; this function's own defensive re-check expresses the
    exact-value/exclusivity relationships StrictInt alone cannot."""
    if not run_bertopic or n_records == 0:
        if excluded != 0:
            raise ReportingConsistencyError(
                f"bertopic_excluded_empty_evidence_count must be exactly 0 when run_bertopic=False "
                f"or there are no full-text records, got {excluded}")
        if topics:
            raise ReportingConsistencyError(
                "BERTopic topics must be empty when run_bertopic=False or there are no full-text records")
        return
    if not (0 <= excluded <= n_records):
        raise ReportingConsistencyError(f"bertopic_excluded_empty_evidence_count {excluded} is not in [0, {n_records}]")
    if excluded == n_records and topics:
        raise ReportingConsistencyError(
            "BERTopic topics must be empty when every full-text record was excluded for empty evidence")


def render_bertopic_note(snapshot: ReportingSourceSnapshot) -> str:
    """By the time this runs, `_validate_bertopic_consistency()` has
    ALREADY confirmed (at capture time, INV-R28) that these five
    conditions are mutually exclusive and jointly exhaustive -- this
    function does not need, and does not perform, any consistency
    checking of its own."""
    n_records = len(snapshot.full_text_records)
    excluded = snapshot.synthesis_output.model.bertopic_excluded_empty_evidence_count
    n_eligible = n_records - excluded
    topics = snapshot.synthesis_output.model.topics
    if not snapshot.config_projection.run_bertopic:
        return "BERTopic thematic clustering was disabled for this run (synthesis.run_bertopic=False)."
    if n_records == 0:
        return "BERTopic was enabled, but no full-text records were available to cluster."
    if n_eligible == 0:
        return ("BERTopic was enabled and records were available, but every one was excluded for "
                "having no usable evidence text to cluster.")
    if topics:
        return (f"BERTopic produced {len(topics)} thematic topics from the included corpus this run. "
                "Execution-level traceability (versions, success/failure) is not yet instrumented for this stage.")
    return ("BERTopic was enabled with eligible records available, but produced zero topics this run. "
            "Execution-level traceability is not yet instrumented for this stage, so this specific "
            "result cannot be distinguished from a silent execution failure.")


# ---------------------------------------------------------------------------
# INV-R7 -- oversight_mode/automation_origin cross-check. Re-derives each
# stage's expected value using ITS OWN actual algorithm and requires
# agreement with what state actually carries -- except §0.2's one named
# limitation (title_abstract_screening's oversight_mode).
# ---------------------------------------------------------------------------

def _check_call_identity_fields(*, event: "AIUseEvent", expected: Dict[str, Any], stage_label: str, event_label: str) -> None:
    """Codex round-9 review, finding 3, P2: widens the round-8 completeness
    binding (work_item_id/rendered_prompt_hash/prompt_parameters/
    input_artifact_refs) to cover EVERY other deterministic field a call
    site's shared identity-builder (`compute_query_expansion_call_identity()`/
    `compute_db_translation_call_identity()`/`compute_qc_second_rater_call_
    identity()`) publishes -- model/provider/task_purpose/prompt_template_
    version/prompt_template_text. `prompt_template_hash` is deliberately NOT
    separately checked here: `AIUseEvent`'s own self-consistency validator
    already ties it to `prompt_template_text` for any VALID event, so
    checking `prompt_template_text` alone already implies `prompt_template_
    hash` agreement. `component_versions`/`access_info`/`attempt`/
    `timestamp`/`tool_kind`/`model_config_id` are deliberately excluded from
    THIS helper -- the first four are runtime/environment-dependent (not
    part of a call's own deterministic identity), and `tool_kind`/
    `model_config_id` are already checked by each call site's own dedicated,
    more specific error message above this helper's use."""
    for field in ("work_item_id", "rendered_prompt_hash", "prompt_parameters", "input_artifact_refs",
                  "model", "provider", "task_purpose", "prompt_template_version", "prompt_template_text"):
        actual = getattr(event, field)
        if actual != expected[field]:
            raise ReportingConsistencyError(
                f"{stage_label}: the {event_label} event's {field} {actual!r} does not match the CURRENT "
                f"invocation's own recomputed {expected[field]!r} -- an unrelated event (a different "
                "prompt/config) may have been substituted (INV-R7)")


def _extract_record_id_from_ref(input_artifact_refs: List[str]) -> Optional[str]:
    """Extracts the record id from a `record:<id>:sha256:<hash>` entry
    (the exact format `_record_content_ref()`/`_criteria_ref()`'s sibling
    helpers in dedup.py/screening.py/screening_qc.py all produce) --
    Codex round-3 review, finding 1(b): reconciling a stage's own events
    against a durable record-identity set (not just a category/count)
    requires recovering WHICH record each event was actually called for.
    `rpartition(':sha256:')` splits on the rightmost occurrence of that
    literal delimiter, so this is correct even if a record id itself
    happened to contain a colon (real ids are UUIDs and never do, but the
    parse does not rely on that). Returns `None` if no ref matches."""
    for ref in input_artifact_refs:
        if ref.startswith("record:") and ":sha256:" in ref:
            prefix, _, _ = ref.rpartition(":sha256:")
            return prefix[len("record:"):]
    return None


def _load_verified_event_by_id(event_store: AIUseEventStore, event_id: str, *, context: str) -> "AIUseEvent":
    """Loads and fully verifies one event by id (format, path containment,
    self-hash, and THIS store's own `review_run_id` -- via
    `AIUseEventStore.load_event()`) for use as one entry in a durable
    per-invocation `producing_event_ids` list (Codex round-5 review,
    findings 2/3). `context` names the caller for a clear error message;
    never silently swallows a missing/corrupt/foreign-run event id."""
    if not isinstance(event_id, str):
        raise ReportingConsistencyError(f"{context}: producing_event_ids contains a non-string entry: {event_id!r}")
    try:
        return event_store.load_event(event_id)
    except AIEventIntegrityError as exc:
        raise ReportingConsistencyError(
            f"{context}: producing_event_ids contains {event_id!r}, which could not be loaded/verified: {exc}"
        ) from exc


def _cross_check_oversight_and_automation_origin(
    *, cfg: ReviewConfig, oversight_mode_state: Dict[str, str], automation_origin_state: Dict[str, str],
    ledger_store: SelectionDecisionStore, event_store: AIUseEventStore,
    current_screening_batches: List[VerifiedFile], screening_review_resolutions: List[Dict[str, Any]],
    synthesis_requested_targets: List[str], imrad_narrative_enabled: bool,
    screening_qc_state: Optional["ReportingScreeningQCState"], query_expansion_enabled: bool,
    search_strategy_producing_event_ids: List[str], search_strategy_db_queries: Dict[str, str],
    search_strategy_expanded_query: str, search_strategy_concepts: List[Dict[str, Any]],
    search_strategy_notes: List[str], search_strategy_seed_query: str,
    screened_records_full: List["Record"], screening_records: List["Record"],
    skip_search_strategy_check: bool = False,
) -> None:
    all_events = event_store.load_all()   # loaded once -- every branch below that needs "every event this
                                             # run" (as opposed to a specific invocation's own recorded ids)
                                             # reuses this single, already-verified list.

    def _check(stage: str, expected_origin: Optional[str], expected_oversight: Optional[str]) -> None:
        actual_origin = automation_origin_state.get(stage)
        actual_oversight = oversight_mode_state.get(stage)
        if actual_origin != expected_origin:
            raise ReportingConsistencyError(
                f"automation_origin[{stage!r}] = {actual_origin!r} in state, but re-derivation "
                f"produces {expected_origin!r} (INV-R7)")
        if actual_oversight != expected_oversight:
            raise ReportingConsistencyError(
                f"oversight_mode[{stage!r}] = {actual_oversight!r} in state, but re-derivation "
                f"produces {expected_oversight!r} (INV-R7)")

    # full_text_eligibility: ledger-tail-based, unconditional -- zero
    # ledger entries and an absent state key both re-derive/read as None,
    # so no gating guard is needed here.
    origin, oversight = ledger_automation_origin_and_oversight(ledger_store, "full_text_eligibility")
    _check("full_text_eligibility", origin, oversight)

    # dedup (Codex round-1 review, finding 1, P1): EVENT-PRESENCE-based
    # (plan §2.8), NOT ledger-based. dedup_node always publishes exactly
    # one dedup_embedding OR dedup_exact AIUseEvent per run, depending on
    # DedupConfig.mode (Track 2 plan §3.1/§3.2) -- including for a
    # zero-record corpus, where the attempt still genuinely happens -- but
    # emits ZERO ledger decisions when there are zero records to decide
    # on. A ledger-based re-derivation therefore wrongly re-derives
    # (None, None) for a valid empty-corpus run even though dedup_node
    # itself always sets both state keys unconditionally. dedup_node's own
    # decision_maker is ALWAYS "rule_based_automation" (the unique/
    # duplicate decision is deterministic given the dedup step's output,
    # never an AI judgment) regardless of that event's own call_status
    # (success or heuristic_fallback).
    dedup_events_exist = any(e.stage in ("dedup_embedding", "dedup_exact") for e in all_events)
    if dedup_events_exist:
        _check("dedup", "rule_based_automation", "automation_unreviewed")
    else:
        _check("dedup", None, None)

    # title_abstract_screening (§0.2 accepted limitation): automation_origin
    # from the CURRENT screening batches' own call_status; oversight_mode
    # reproduces screening.py's own rule (screening_review_resolutions
    # non-empty), NOT a ledger-tail-based rule.
    if current_screening_batches:
        makers = {"ai" if vf.model.call_status == "success" else "rule_based_automation"
                  for vf in current_screening_batches}
        if makers == {"ai"}:
            screening_origin = "ai"
        elif makers == {"rule_based_automation"}:
            screening_origin = "rule_based_automation"
        else:
            screening_origin = "mixed"
        screening_oversight = "human_resolved" if screening_review_resolutions else "automation_unreviewed"
    else:
        screening_origin, screening_oversight = None, None
    _check("title_abstract_screening", screening_origin, screening_oversight)

    if not skip_search_strategy_check:
        # search_strategy: gated on ITS OWN durable events, not on
        # oversight_mode_state (Codex round-2 review, finding 1, P1:
        # `stage not in oversight_mode_state` let deleting BOTH state keys
        # silently bypass this check). search_strategy_builder_node is the
        # very first node of every review graph (START -> search_strategy_
        # builder -> search, in scoping.py/rapid.py/slr.py alike), and its own
        # docstring: "Always runs (even when disabled) to produce per-database
        # query variants via LLM translation" -- Step 2 (db-translation) is
        # UNCONDITIONAL; Step 1 (query expansion) runs ONLY when
        # `cfg.search.strategy_builder.enabled` is True.
        #
        # Codex round-3 review, finding 1(a): the OLD `if
        # search_strategy_events:` gate made the db-translation guarantee
        # OPTIONAL from reporting's own point of view. Codex round-4 review,
        # finding 2: requiring only "at least one event in EITHER stage" was
        # still too permissive.
        #
        # Codex round-5 review, finding 3, P1: round-4's own fix scanned EVERY
        # historical event ever published under this review_run_id for these
        # two stage names -- append-only across every invocation, not scoped
        # to the CURRENT one. That is wrong in both directions: (a) an earlier
        # invocation's own query_expansion event, published back when
        # `strategy_cfg.enabled` was True, would wrongly satisfy the presence
        # check for a LATER invocation whose flag is now False; (b) deleting
        # the CURRENT invocation's own db_translation event could be masked by
        # an unrelated, older db_translation event surviving from a prior
        # invocation. Fixed by anchoring on `state["search_strategy"][
        # "producing_event_ids"]` (`SearchStrategy.producing_event_ids`,
        # search_strategy_builder.py) -- the exact event_id(s) THIS invocation
        # itself published -- loading and verifying each one individually
        # (format, containment, self-hash, this run's own review_run_id) via
        # `_load_verified_event_by_id()`, rather than re-scanning the whole
        # stage-name-keyed event pool.
        # Codex round-6 review, finding 1, P1: round-5's own fix wrongly reused
        # `producing_event_ids` (THIS invocation's own events) for BOTH the
        # completeness/inventory checks below AND automation_origin
        # re-derivation. That is right for completeness (proving THIS
        # invocation's own claimed evidence is real and internally
        # consistent), but WRONG for origin: `search_strategy_builder_node`
        # itself classifies origin over EVERY historical event for these two
        # stage names ever published under this `review_run_id`
        # (`event_store.load_all()` filtered by stage, search_strategy_
        # builder.py's own code) -- an append-only population that can span
        # multiple invocations. Re-deriving origin from only the CURRENT
        # invocation's ids would wrongly reject a legitimate rerun whose own
        # call outcome differs from an earlier invocation's (e.g. an earlier
        # db_translation succeeded, this one fell back -- the producer's own
        # state says "mixed", but a current-only re-derivation would say
        # "rule_based_automation"). Origin re-derivation below therefore uses
        # `all_events` filtered by stage (the SAME population the producer
        # itself classifies over), never `search_strategy_producing_event_ids`.
        #
        # Codex round-10 review, finding 1, P1: the failure-branch's own
        # `expanded_query == seed_query` check (below) only proves INTERNAL
        # consistency within the (possibly entirely substituted) SearchStrategy
        # state blob -- it never anchors `seed_query` itself to the one durably-
        # trusted source. `search_strategy_builder.py`'s own real code sets
        # `seed_query = cfg.search.query` unconditionally, success or failure --
        # checked here, UNCONDITIONALLY (not just inside the failure branch),
        # before any of the per-event checks below.
        if search_strategy_seed_query != cfg.search.query:
            raise ReportingConsistencyError(
                f"search_strategy: SearchStrategy.seed_query ({search_strategy_seed_query!r}) != "
                f"cfg.search.query ({cfg.search.query!r}) -- search_strategy_builder_node always sets "
                "seed_query verbatim from the active config, unconditionally (INV-R7)")
        search_strategy_events = [
            _load_verified_event_by_id(event_store, eid, context="search_strategy")
            for eid in search_strategy_producing_event_ids
        ]
        for e in search_strategy_events:
            if e.stage not in ("search_strategy_query_expansion", "search_strategy_db_translation"):
                raise ReportingConsistencyError(
                    f"search_strategy: producing_event_ids contains event {e.event_id!r}, whose own stage "
                    f"{e.stage!r} is neither search_strategy_query_expansion nor search_strategy_db_translation")
        db_translation_events = [e for e in search_strategy_events if e.stage == "search_strategy_db_translation"]
        # Codex round-6 review, finding 4, P2: "at least one" let duplicate ids
        # or a smuggled-in extra historical translation event pass; require
        # EXACTLY one. Combined with the exact query-expansion count below and
        # the stage-membership check above (every id is either a translation or
        # an expansion event), this also proves NO other event is present --
        # `state.py`'s own new uniqueness validator on `producing_event_ids`
        # (Codex round-6 review, finding 4) rules out the same real event id
        # being counted twice.
        if len(db_translation_events) != 1:
            if not db_translation_events:
                detail = "no search_strategy_db_translation event exists"
            else:
                detail = f"{len(db_translation_events)} search_strategy_db_translation events exist (expected exactly one)"
            raise ReportingConsistencyError(
                f"search_strategy: {detail} among state['search_strategy']['producing_event_ids'] -- "
                "search_strategy_builder_node always attempts DB query translation exactly once per "
                "invocation, so this means the CURRENT invocation's own event was deleted, never recorded, "
                "an unrelated (e.g. historical) event was smuggled in alongside it, or state was tampered "
                "with (INV-R7)")
        query_expansion_events = [e for e in search_strategy_events if e.stage == "search_strategy_query_expansion"]
        if query_expansion_enabled and len(query_expansion_events) != 1:
            if not query_expansion_events:
                detail = "no search_strategy_query_expansion event exists"
            else:
                detail = f"{len(query_expansion_events)} search_strategy_query_expansion events exist (expected exactly one)"
            raise ReportingConsistencyError(
                f"search_strategy: {detail} among state['search_strategy']['producing_event_ids'], but "
                "cfg.search.strategy_builder.enabled=True -- search_strategy_builder_node's own Step 1 "
                "always attempts query expansion exactly once per invocation when this flag is set (INV-R7)")
        if not query_expansion_enabled and query_expansion_events:
            raise ReportingConsistencyError(
                "search_strategy: a search_strategy_query_expansion event exists among state['search_strategy']"
                "['producing_event_ids'], but cfg.search.strategy_builder.enabled=False -- "
                "search_strategy_builder_node's own Step 1 never attempts query expansion when this flag is "
                "unset (INV-R7)")

        # Codex round-7 review, finding 2, P1: presence + cardinality alone let
        # an impossible event SHAPE (e.g. a query_expansion event with
        # call_status="heuristic_fallback", which this call site can never
        # produce) or a valid-but-UNRELATED historical event of the right stage
        # (deleting the current event and substituting an old one with stale
        # output) pass. Validate the COMPLETE per-call-site contract for both
        # events -- call_status vocabulary, tool_kind/model_config_id identity,
        # and EXACT output-reference-list equality (not membership) bound to
        # the CURRENT SearchStrategy state -- confirmed directly against
        # search_strategy_builder.py's own two `build_ai_use_event(...)` call
        # sites (never assumed to mirror each other).
        if query_expansion_events:
            expansion_event = query_expansion_events[0]
            if expansion_event.call_status not in ("success", "failure"):
                raise ReportingConsistencyError(
                    f"search_strategy: the query_expansion event has call_status={expansion_event.call_status!r}, "
                    "but _run_query_expansion() has no computed fallback -- this call site can only ever "
                    "produce 'success' or 'failure' (INV-R7)")
            if expansion_event.tool_kind != "llm":
                raise ReportingConsistencyError(
                    f"search_strategy: the query_expansion event has tool_kind={expansion_event.tool_kind!r}, "
                    "expected 'llm' (INV-R7)")
            if expansion_event.model_config_id != "search_strategy_query_expansion_v1":
                raise ReportingConsistencyError(
                    f"search_strategy: the query_expansion event has model_config_id="
                    f"{expansion_event.model_config_id!r}, expected 'search_strategy_query_expansion_v1' (INV-R7)")
            if expansion_event.call_status == "failure":
                if expansion_event.output_artifact_refs != []:
                    raise ReportingConsistencyError(
                        f"search_strategy: the query_expansion event has call_status='failure' but "
                        f"output_artifact_refs={expansion_event.output_artifact_refs!r} -- a failure never "
                        "has an output (INV-06)")
                # Codex round-10 review, finding 2, P1: `AIUseEvent` has no
                # cross-field validator tying call_status to failure_detail
                # (ai_traceability.py) -- a failure_detail=None event self-
                # verifies and loads fine, so this must be enforced here.
                # _run_query_expansion()'s own real code (search_strategy_
                # builder.py) always sets failure_detail=str(exc) on this path.
                if expansion_event.failure_detail is None:
                    raise ReportingConsistencyError(
                        "search_strategy: the query_expansion event has call_status='failure' but "
                        "failure_detail is None -- a genuine failure always carries a real error "
                        "description (INV-R7)")
                # Codex round-9 review, finding 1, P1: output-emptiness alone
                # cannot distinguish THIS invocation's own failure event from a
                # DIFFERENT, genuinely-existing failure event of an unrelated
                # prompt/config -- every failure event shares the identical
                # empty output by construction (INV-06). `_run_query_expansion()`'s
                # own failure path (search_strategy_builder.py) never reassigns
                # `expanded_query`/`concepts` away from their initial seed-query/
                # empty-list values, and always leaves EXACTLY two entries in
                # `notes` in a fixed, reconstructible order: the "Query expanded
                # with N concept groups." banner (N==0 on failure, since
                # `concepts` was never populated) followed by
                # f"Expansion failed: {failure_exc}" -- the SAME string the event's
                # own `failure_detail` field stores. Reconciling the CURRENT
                # state's own notes against the CURRENT event's own failure_detail
                # (rather than merely checking output emptiness) closes the
                # substitution gap: an older, same-shaped failure event from a
                # different prompt/config would have a DIFFERENT failure_detail,
                # which would no longer match state's own note text.
                if search_strategy_expanded_query != search_strategy_seed_query:
                    raise ReportingConsistencyError(
                        f"search_strategy: the query_expansion event has call_status='failure', but "
                        f"SearchStrategy.expanded_query ({search_strategy_expanded_query!r}) != seed_query "
                        f"({search_strategy_seed_query!r}) -- _run_query_expansion()'s own failure path never "
                        "reassigns expanded_query away from the seed query (INV-R7)")
                if search_strategy_concepts != []:
                    raise ReportingConsistencyError(
                        "search_strategy: the query_expansion event has call_status='failure', but "
                        f"SearchStrategy.concepts is non-empty ({search_strategy_concepts!r}) -- "
                        "_run_query_expansion()'s own failure path never populates concepts (INV-R7)")
                expected_failure_notes = [
                    f"Query expanded with {len(search_strategy_concepts)} concept groups.",
                    f"Expansion failed: {expansion_event.failure_detail}",
                ]
                if search_strategy_notes != expected_failure_notes:
                    raise ReportingConsistencyError(
                        f"search_strategy: the query_expansion event's own failure_detail "
                        f"{expansion_event.failure_detail!r} does not reconcile with the CURRENT "
                        f"SearchStrategy.notes {search_strategy_notes!r} -- expected exactly "
                        f"{expected_failure_notes!r} (_run_query_expansion()'s own real note-construction on "
                        "a failure path) -- an unrelated failure event (a different prompt/config, or a "
                        "hand-edited failure_detail) may have been substituted (INV-R7)")
            else:  # call_status == "success"
                # Codex round-10 review, finding 2, P1: symmetric to the failure
                # branch's own new check above -- a genuine success never
                # carries a failure description.
                if expansion_event.failure_detail is not None:
                    raise ReportingConsistencyError(
                        "search_strategy: the query_expansion event has call_status='success' but a "
                        f"non-None failure_detail ({expansion_event.failure_detail!r}) -- a genuine success "
                        "never carries a failure description (INV-R7)")
                expected_expansion_output_ref = f"sha256:{sha256_hex(canonical_query_expansion_output(expanded_query=search_strategy_expanded_query, concepts=search_strategy_concepts, notes=search_strategy_notes))}"
                if expansion_event.output_artifact_refs != [expected_expansion_output_ref]:
                    raise ReportingConsistencyError(
                        f"search_strategy: the query_expansion event's output_artifact_refs "
                        f"{expansion_event.output_artifact_refs!r} does not exactly equal "
                        f"[{expected_expansion_output_ref!r}], the hash of the CURRENT SearchStrategy's own "
                        "expanded_query/concepts/notes -- state may have been edited independently of its "
                        "claimed producing event, or an unrelated event was substituted (INV-R7)")
            # Codex round-8 review, finding 2, P1: call_status/tool_kind/
            # model_config_id/output alone cannot distinguish THIS invocation's
            # own event from a DIFFERENT, genuinely-existing event whose
            # call_status and output happen to coincide -- every failure event
            # for this stage has the identical empty output ([]) by
            # construction, and a stale success event could in principle share
            # an output hash with the current one. Reconstruct the COMPLETE
            # immutable call identity (work_item_id, rendered_prompt_hash,
            # prompt_parameters, input_artifact_refs) from LIVE config via the
            # SAME shared builder `_run_query_expansion()` itself uses to
            # publish the event, and require exact agreement on every field.
            expected_expansion_identity = compute_query_expansion_call_identity(cfg)
            _check_call_identity_fields(
                event=expansion_event, expected=expected_expansion_identity,
                stage_label="search_strategy", event_label="query_expansion")

        translation_event = db_translation_events[0]
        if translation_event.call_status not in ("success", "heuristic_fallback"):
            raise ReportingConsistencyError(
                f"search_strategy: the db_translation event has call_status={translation_event.call_status!r}, "
                "but _generate_db_queries() always has a real computed fallback (same query for every DB) -- "
                "this call site can never produce 'failure' (INV-R7)")
        if translation_event.tool_kind != "llm":
            raise ReportingConsistencyError(
                f"search_strategy: the db_translation event has tool_kind={translation_event.tool_kind!r}, "
                "expected 'llm' (INV-R7)")
        if translation_event.model_config_id != "search_strategy_db_translation_v2":
            raise ReportingConsistencyError(
                f"search_strategy: the db_translation event has model_config_id="
                f"{translation_event.model_config_id!r}, expected 'search_strategy_db_translation_v2' (INV-R7)")
        # Codex round-10 review, finding 2, P1: symmetric failure_detail
        # invariant -- `_generate_db_queries()`'s own real code sets
        # failure_detail=str(exc) exactly when call_status="heuristic_fallback"
        # and failure_detail=None exactly when call_status="success" (one-to-one,
        # search_strategy_builder.py). `AIUseEvent` has no cross-field validator
        # enforcing this itself.
        if translation_event.call_status == "success" and translation_event.failure_detail is not None:
            raise ReportingConsistencyError(
                "search_strategy: the db_translation event has call_status='success' but a non-None "
                f"failure_detail ({translation_event.failure_detail!r}) -- a genuine success never carries "
                "a failure description (INV-R7)")
        if translation_event.call_status == "heuristic_fallback" and translation_event.failure_detail is None:
            raise ReportingConsistencyError(
                "search_strategy: the db_translation event has call_status='heuristic_fallback' but "
                "failure_detail is None -- a genuine fallback is always triggered by a real error (INV-R7)")
        # Codex round-7 review, finding 2, P1: `not in` (membership) let an
        # extra, unrelated ref alongside the correct one pass -- the producer's
        # own shape is always EXACTLY one element.
        expected_translation_output_ref = f"sha256:{sha256_hex(search_strategy_db_queries)}"
        if translation_event.output_artifact_refs != [expected_translation_output_ref]:
            raise ReportingConsistencyError(
                f"search_strategy: the db_translation event's output_artifact_refs "
                f"{translation_event.output_artifact_refs!r} does not exactly equal "
                f"[{expected_translation_output_ref!r}], the hash of the CURRENT SearchStrategy.db_queries -- "
                "state may have been edited independently of its claimed producing event, or an unrelated "
                "event was substituted (INV-R7)")
        # Codex round-12 review, finding 2, P2: the output-hash check above only
        # proves `db_queries` matches WHATEVER the state currently claims -- it
        # never proves that claim is itself a shape `_generate_db_queries()`
        # could actually produce. A self-consistent (hash-matching) event/state
        # pair that's missing a configured database, has a blank query for one,
        # or (in the fallback case) varies per-DB would previously pass.
        # `_generate_db_queries()`'s own real code (search_strategy_builder.py):
        # `dbs_needed = sorted(set(databases) | {"pubmed"})`; success requires
        # EVERY dbs_needed key present with a non-blank string value (`missing`/
        # `empty` checks, raising to the fallback branch otherwise); the
        # fallback is EXACTLY `{db: query for db in dbs_needed}` -- the
        # SAME value (the current invocation's own `expanded_query`) for every
        # database, never per-DB variation.
        expected_dbs_needed = sorted(set(cfg.search.databases) | {"pubmed"})
        if set(search_strategy_db_queries.keys()) != set(expected_dbs_needed):
            raise ReportingConsistencyError(
                f"search_strategy: db_queries key set {sorted(search_strategy_db_queries.keys())} does not "
                f"exactly equal the expected database set {expected_dbs_needed} "
                "(sorted(set(cfg.search.databases) | {'pubmed'})) -- _generate_db_queries() can never produce "
                "a db_queries dict missing or adding a database (INV-R7)")
        non_string = [db for db, q in search_strategy_db_queries.items() if not isinstance(q, str)]
        if non_string:
            raise ReportingConsistencyError(
                f"search_strategy: db_queries has a non-string query for database(s) {sorted(non_string)} -- "
                "_generate_db_queries() only ever accepts or fabricates string queries (INV-R7)")
        # The "non-blank" requirement is only a property of the SUCCESS path's
        # own validation (`empty = [db for db, q in candidate.items() if not
        # q.strip()]`, raising to the fallback branch otherwise) -- the
        # FALLBACK path (`db_queries = {db: query for db in dbs_needed}`)
        # blindly reuses whatever `expanded_query` it was given, which is
        # itself only ever guaranteed non-`None`, never guaranteed non-blank
        # (e.g. a genuinely blank `cfg.search.query` with a failed expansion
        # step legitimately leaves `expanded_query == ""`) -- a blank fallback
        # value is real, producer-reachable output, not a shape violation.
        if translation_event.call_status == "success":
            blank = [db for db, q in search_strategy_db_queries.items() if not q.strip()]
            if blank:
                raise ReportingConsistencyError(
                    f"search_strategy: db_translation call_status='success', but db_queries has a blank query "
                    f"for database(s) {sorted(blank)} -- a genuine success always requires every database to "
                    "have a real, non-empty query (INV-R7)")
        if translation_event.call_status == "heuristic_fallback":
            wrong_value = {db: q for db, q in search_strategy_db_queries.items() if q != search_strategy_expanded_query}
            if wrong_value:
                raise ReportingConsistencyError(
                    f"search_strategy: db_translation call_status='heuristic_fallback', but db_queries has "
                    f"value(s) differing from the current expanded_query for database(s) "
                    f"{sorted(wrong_value.keys())} -- the real fallback is ALWAYS the identical query for "
                    "every database, never per-DB variation (INV-R7)")

        # Codex round-8 review, finding 2, P1: same complete-call-identity binding
        # as query expansion above, via the SAME shared builder
        # `_generate_db_queries()` itself uses. `query` is
        # `search_strategy_expanded_query` -- THIS invocation's own actual input
        # to db translation (whatever Step 1 actually produced, success,
        # fallback, or disabled) -- never independently re-derived.
        expected_translation_identity = compute_db_translation_call_identity(
            query=search_strategy_expanded_query, databases=cfg.search.databases,
            model=cfg.search.strategy_builder.llm_model,
        )
        _check_call_identity_fields(
            event=translation_event, expected=expected_translation_identity,
            stage_label="search_strategy", event_label="db_translation")

        origin_events = [e for e in all_events if e.stage in ("search_strategy_query_expansion", "search_strategy_db_translation")]
        origin = classify_automation_origin_optional(origin_events)
        _check("search_strategy", origin, "automation_unreviewed")

    # screening_qc_second_rater: gated on `screening_qc_state is not None`
    # (Codex round-2 review, finding 1, P1), NOT on event presence and NOT
    # on oversight_mode_state. screening_qc_node only ever early-returns
    # {"screening_qc": None} when qc_cfg.enabled is False -- an ENABLED run
    # with zero eligible records for IRR sampling still executes fully and
    # still unconditionally sets oversight_mode["screening_qc_second_rater"]
    # = "automation_unreviewed", even though it may publish ZERO events for
    # this stage (automation_origin's own key is then correctly left unset,
    # matching classify_automation_origin_optional([]) -> None). Gating on
    # event presence would wrongly skip verification for that legitimate
    # zero-event case; gating on oversight_mode_state's own key would let
    # deleting it bypass the check even though screening_qc_state (durable,
    # captured straight from state, validated by _validate_screening_qc_state)
    # already proves the stage ran.
    if screening_qc_state is not None:
        irr = screening_qc_state.irr

        # Codex round-5 review, finding 2, P1: scanning EVERY historical
        # screening_qc_second_rater event under this review_run_id (as
        # round-3/round-4 both did) conflates every invocation's own
        # attempts -- `screening_qc_node` draws a FRESH `random.sample()`
        # on every call, so a legitimate rerun with a different sample
        # leaves both the old and new invocations' events on disk, and
        # taking their union against ONLY the latest `irr.sampled_record_
        # ids` can never match (nor can a raw count comparison, since
        # `reserved_attempt()`'s own `attempt` field lets one record
        # legitimately accumulate multiple historical events across
        # reruns -- ai_traceability.py's own AIUseEvent docstring).
        # Anchored instead on `irr.producing_event_ids` (`_compute_irr()`,
        # screening_qc.py) -- the exact event_id(s) THIS invocation
        # published, one per sampled record, success or failure alike --
        # loaded and verified individually via `_load_verified_event_by_id()`.
        screening_qc_events = [
            _load_verified_event_by_id(event_store, eid, context="screening_qc_second_rater")
            for eid in irr.producing_event_ids
        ]
        for e in screening_qc_events:
            if e.stage != "screening_qc_second_rater":
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater: irr.producing_event_ids contains event {e.event_id!r}, "
                    f"whose own stage {e.stage!r} is not screening_qc_second_rater")

        # Codex round-6 review, finding 2, P1: `n_failed`/`n_valid_pairs`
        # were only checked for INTERNAL arithmetic consistency
        # (`_counts_conserve`) -- never reconciled against the actual
        # `call_status` of the durable events THIS invocation itself
        # published. A state claiming `n_valid_pairs=1, n_failed=0` while
        # its sole bound event's own `call_status="failure"` passed
        # unnoticed. This call site never produces a `heuristic_fallback`
        # substitute (screening_qc.py's own docstring/field-contract), so
        # every event here must be exactly "success" or "failure".
        for e in screening_qc_events:
            if e.call_status not in ("success", "failure"):
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater event {e.event_id!r} has call_status={e.call_status!r}, "
                    "but this call site never produces a heuristic_fallback substitute -- every event "
                    "must be 'success' or 'failure' (INV-R7)")
        failure_event_count = sum(1 for e in screening_qc_events if e.call_status == "failure")
        success_event_count = sum(1 for e in screening_qc_events if e.call_status == "success")
        if failure_event_count != irr.n_failed:
            raise ReportingConsistencyError(
                f"screening_qc_second_rater: {failure_event_count} of THIS invocation's own "
                f"producing_event_ids have call_status='failure', but irr.n_failed={irr.n_failed} "
                "(INV-R7)")
        if success_event_count != irr.n_valid_pairs:
            raise ReportingConsistencyError(
                f"screening_qc_second_rater: {success_event_count} of THIS invocation's own "
                f"producing_event_ids have call_status='success', but irr.n_valid_pairs="
                f"{irr.n_valid_pairs} (INV-R7)")

        # Codex round-6 review, finding 1, P1: origin re-derivation must
        # use the SAME event population `screening_qc_node` itself
        # classifies over -- every historical `screening_qc_second_rater`
        # event ever published under this `review_run_id`
        # (`event_store.load_all()` filtered by stage, screening_qc.py's
        # own code), NOT `irr.producing_event_ids` (THIS invocation's own
        # ids only). A legitimate rerun whose own call outcome differs
        # from an earlier invocation's (e.g. an earlier call succeeded,
        # this one failed) would otherwise be wrongly rejected: the
        # producer's own state says "ai", a current-only re-derivation
        # would say "no_automated_output". `irr.producing_event_ids`
        # remains the right anchor for the completeness/count/record-set
        # checks above and below, which ask a different question ("did
        # THIS invocation's own claimed evidence really happen"), not
        # "what does the stage's overall origin classification say".
        origin_events = [e for e in all_events if e.stage == "screening_qc_second_rater"]
        origin = classify_automation_origin_optional(origin_events)
        _check("screening_qc_second_rater", origin, "automation_unreviewed")

        # The DISTINCT record-ID SET these specific events were actually
        # called for (a record contributes at most one id, however many
        # attempts it has across ITS OWN history) must equal
        # `irr.sampled_record_ids` exactly -- since `producing_event_ids`
        # is now scoped to THIS invocation alone (one entry per sampled
        # record, by construction of `_compute_irr()`), this also proves
        # `len(producing_event_ids) == n_attempted` was genuinely honored,
        # not merely declared.
        event_record_ids: set = set()
        for e in screening_qc_events:
            record_id = _extract_record_id_from_ref(e.input_artifact_refs)
            if record_id is None:
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater event {e.event_id!r} has no 'record:<id>:sha256:...' "
                    "entry in its own input_artifact_refs -- cannot verify which record it was called for")
            event_record_ids.add(record_id)
        if event_record_ids != set(irr.sampled_record_ids):
            raise ReportingConsistencyError(
                f"screening_qc_second_rater: the record_id set actually called by THIS invocation's own "
                f"producing_event_ids ({sorted(event_record_ids)}) does not match irr.sampled_record_ids "
                f"({sorted(irr.sampled_record_ids)}) (INV-R7)")

        # Codex round-9 review, finding 2, P1: stage/status-count/record-ID-
        # set checks alone cannot distinguish THIS invocation's own event
        # (for a given record) from a DIFFERENT, genuinely-existing event
        # for the SAME record whose criteria/model/model_config_id/
        # work_item_id/decision-output are unrelated. Reconstruct the
        # COMPLETE immutable call identity from the CURRENT config and the
        # CURRENT record content via the SAME shared builder
        # `_second_rater_screen()` itself uses, and require exact agreement
        # on every field -- plus, for a success event specifically, that its
        # own output corresponds to the SAME decision persisted as this
        # record's CURRENT `second_rater_decision`.
        qc_model = cfg.screening_qc.second_rater_model or cfg.screening.llm_model
        # Codex round-10 review, finding 3, P1: a plain dict-comprehension
        # silently keeps only the LAST entry for a repeated record_id --
        # never checked before. Reject duplicates outright rather than
        # silently picking one.
        screened_full_ids = [r.id for r in screened_records_full]
        if len(set(screened_full_ids)) != len(screened_full_ids):
            duplicates = sorted({rid for rid in screened_full_ids if screened_full_ids.count(rid) > 1})
            raise ReportingConsistencyError(
                f"state['screened_records_full'] contains duplicate record_id(s): {duplicates} -- cannot "
                "unambiguously bind a screening_qc_second_rater event to a single record (INV-R7)")
        # Codex round-11 review, finding 1, P1: round-10's duplicate check
        # only proved `screened_records_full` has no INTERNAL duplicates --
        # it never proved this population is the SAME SIZE/SET as the
        # durably-captured `screening_records` corpus. Shrinking (or
        # padding) `screened_records_full` while consistently shrinking
        # `n_attempted`/`n_valid_pairs`/`sampled_record_ids`/
        # `producing_event_ids` together previously passed every check
        # above, since none of them ever looked at the population `_compute_
        # irr()` actually drew its sample FROM.
        screening_ids_set = {r.id for r in screening_records}
        screened_full_ids_set = set(screened_full_ids)
        if screened_full_ids_set != screening_ids_set:
            raise ReportingConsistencyError(
                f"state['screened_records_full'] record_id set ({sorted(screened_full_ids_set)}) does not "
                f"exactly match the durably-captured screening corpus's own record_id set "
                f"({sorted(screening_ids_set)}) -- missing="
                f"{sorted(screening_ids_set - screened_full_ids_set)!r}, "
                f"foreign={sorted(screened_full_ids_set - screening_ids_set)!r} (INV-R7)")

        # Codex round-12 review, finding 1, P1: round-11's eligibility
        # recomputation still read `screening_decision` directly off
        # `screened_records_full`'s own (mutable, self-declared) entries --
        # round-11's checks only proved the record_id SET matches
        # `screening_records`, never that each record's own
        # `screening_decision` field is genuine. An attacker could shrink
        # the eligible pool by setting an UNSAMPLED-but-eligible record's
        # `screening_decision` to None on `screened_records_full` while
        # consistently shrinking `n_attempted`/`n_valid_pairs`/
        # `sampled_record_ids`/`producing_event_ids` together -- every check
        # above still passes, since none of them touch this field for a
        # record outside the sampled subset.
        #
        # Fix: derive each record's screening decision from the FROZEN,
        # already-verified `current_screening_batches` (each a
        # `ScreeningBatchResult` with `record_ids`/`responses` in 1:1 order,
        # itself cross-verified against its own producing `AIUseEvent` --
        # see `ScreeningBatchResultStore._verify_result()`) instead of
        # trusting `screened_records_full.screening_decision` directly.
        #
        # `compute_current_screening_batches(records=screening_records, cfg=cfg)`
        # partitions the durably-captured `screening_records` corpus into
        # disjoint contiguous batches, and `_load_current_screening_batches()`
        # requires a verified `ScreeningBatchResult` matching
        # `expected_record_ids=<that batch's own record ids>` to exist for
        # EVERY batch (raising otherwise) -- so the combined record_ids
        # across all of `current_screening_batches` is STRUCTURALLY
        # guaranteed to equal `{r.id for r in screening_records}` already;
        # no separate key-set check is needed here (verified by reading
        # `_load_current_screening_batches`/`compute_current_screening_batches`
        # directly, not assumed). The cheap assertion below is defense in
        # depth against that guarantee ever regressing, not a new gap-close.
        batch_derived_decision_by_id: Dict[str, str] = {}
        for vf in current_screening_batches:
            for rid, resp in zip(vf.model.record_ids, vf.model.responses):
                batch_derived_decision_by_id[rid] = resp.decision
        if set(batch_derived_decision_by_id) != screening_ids_set:
            raise ReportingConsistencyError(
                "current_screening_batches' own combined record_id set does not exactly match "
                "screening_records -- this should be structurally impossible per "
                "_load_current_screening_batches()'s own per-batch verification (INV-R7)")
        # `Record.screening_decision` is a required, non-Optional-in-practice
        # field screening_node ALWAYS sets for every record before any
        # human-review branching (screening.py: `record.screening_decision =
        # resp.decision` inside `for i, record in enumerate(records):`,
        # unconditional -- confirmed directly, not assumed) -- and
        # `ScreeningResponse.decision` (what the batch-result projection
        # carries) is itself a required, non-`None` `Literal[...]` field, so
        # every record in `screening_records` is, by construction, ALWAYS
        # "eligible" once anchored to this verified projection. Reusing
        # `compute_qc_eligible_records()` (rather than reimplementing its
        # filter) over records annotated with the VERIFIED decision --
        # instead of `screened_records_full`'s own mutable field -- keeps
        # both sides using the identical eligibility definition while making
        # the input to that definition tamper-proof.
        verified_screened_records = [
            r.model_copy(update={"screening_decision": batch_derived_decision_by_id[r.id]})
            for r in screening_records
        ]
        expected_eligible = compute_qc_eligible_records(verified_screened_records)
        expected_n_attempted = min(cfg.screening_qc.calibration_sample_size, len(expected_eligible))
        if irr.n_attempted != expected_n_attempted:
            raise ReportingConsistencyError(
                f"screening_qc_second_rater: irr.n_attempted={irr.n_attempted}, but the deterministic "
                f"sample size min(calibration_sample_size={cfg.screening_qc.calibration_sample_size}, "
                f"len(eligible_records)={len(expected_eligible)}) over the REAL, reconciled screening "
                f"population is {expected_n_attempted} (INV-R7)")

        screened_records_by_id = {r.id: r for r in screened_records_full}
        # Codex round-10 review, finding 3, P1: `screened_records_full` is a
        # SEPARATE graph-state projection from the durably-captured
        # `snapshot.screening_records` (state["deduplicated_records"]) --
        # both share the SAME title/abstract/doi per record by construction
        # (screening_node only ever ANNOTATES a record with a screening
        # decision, never rewrites its bibliographic content), but nothing
        # previously reconciled the two. A stale/tampered
        # `screened_records_full` row could otherwise agree with a
        # stale/tampered QC event while disagreeing with the actual,
        # durably-captured screening corpus. `screening_records` is the
        # anchor of truth for content; only `second_rater_decision` (which
        # exists ONLY on `screened_records_full`) is trusted from there,
        # and only after this content-match passes.
        screening_records_by_id = {r.id: r for r in screening_records}
        for e in screening_qc_events:
            record_id = _extract_record_id_from_ref(e.input_artifact_refs)
            matched_record = screened_records_by_id.get(record_id)
            if matched_record is None:
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater event {e.event_id!r} was called for record_id "
                    f"{record_id!r}, which is not present in state['screened_records_full'] -- cannot "
                    "verify this event's own call identity against the CURRENT record content (INV-R7)")
            anchor_record = screening_records_by_id.get(record_id)
            if anchor_record is None:
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater event {e.event_id!r} was called for record_id "
                    f"{record_id!r}, which is not present in the durably-captured screening corpus "
                    "(state['deduplicated_records']) -- cannot reconcile screened_records_full against "
                    "the actual current screening inputs (INV-R7)")
            if (matched_record.title, matched_record.abstract, matched_record.doi) != (
                    anchor_record.title, anchor_record.abstract, anchor_record.doi):
                raise ReportingConsistencyError(
                    f"screened_records_full's own entry for record_id {record_id!r} "
                    f"(title={matched_record.title!r}, abstract={matched_record.abstract!r}, "
                    f"doi={matched_record.doi!r}) disagrees with the SAME record_id's entry in the "
                    f"durably-captured screening corpus (title={anchor_record.title!r}, "
                    f"abstract={anchor_record.abstract!r}, doi={anchor_record.doi!r}) -- "
                    "screened_records_full may have been tampered with independently of the actual "
                    "current screening checkpoint (INV-R7)")
            expected_identity = compute_qc_second_rater_call_identity(cfg=cfg, record=matched_record, model=qc_model)
            # Unlike the search-strategy call sites, QC has no separate,
            # earlier tool_kind/model_config_id check of its own --
            # `_check_call_identity_fields()` deliberately excludes both
            # (they're checked by each call site's OWN dedicated check),
            # so they must be checked here explicitly, or a substituted
            # event with the correct work_item_id/output but a wrong
            # model_config_id/tool_kind would pass uncaught (Codex round-9
            # review, finding 2's own confirmed "wrong model-config ID"
            # bypass).
            if e.tool_kind != expected_identity["tool_kind"]:
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                    f"tool_kind={e.tool_kind!r}, expected {expected_identity['tool_kind']!r} (INV-R7)")
            if e.model_config_id != expected_identity["model_config_id"]:
                raise ReportingConsistencyError(
                    f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                    f"model_config_id={e.model_config_id!r}, expected "
                    f"{expected_identity['model_config_id']!r} (INV-R7)")
            _check_call_identity_fields(
                event=e, expected=expected_identity, stage_label="screening_qc_second_rater",
                event_label=f"record={record_id!r}")
            # Codex round-10 review, finding 2, P1: only the success branch's
            # OWN output was ever checked -- a failed event could retain a
            # non-empty output_artifact_refs, its matched record could retain
            # a non-None second_rater_decision (an impossible state per
            # _second_rater_screen()'s own real code: `decision` is `None`
            # on ANY failure, never a fabricated value), and failure_detail
            # could be `None` even on a "failure" call_status -- none of
            # which the pre-existing success-only check could ever catch.
            # `AIUseEvent` itself has no cross-field validator tying
            # call_status to failure_detail (ai_traceability.py) -- a
            # `failure_detail=None, call_status="failure"` event loads and
            # self-verifies just fine, so this must be enforced here.
            if e.call_status == "success":
                if matched_record.second_rater_decision is None:
                    raise ReportingConsistencyError(
                        f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                        "call_status='success', but the matched record's own second_rater_decision is "
                        "None -- _second_rater_screen() never leaves a successful call's decision unset "
                        "(INV-R7)")
                if e.failure_detail is not None:
                    raise ReportingConsistencyError(
                        f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                        f"call_status='success' but a non-None failure_detail ({e.failure_detail!r}) -- a "
                        "genuine success never carries a failure description (INV-R7)")
                expected_output_ref = f"sha256:{sha256_hex({'decision': matched_record.second_rater_decision})}"
                if e.output_artifact_refs != [expected_output_ref]:
                    raise ReportingConsistencyError(
                        f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                        f"output_artifact_refs={e.output_artifact_refs!r}, which does not match "
                        f"[{expected_output_ref!r}] -- the hash of the CURRENT "
                        f"Record.second_rater_decision ({matched_record.second_rater_decision!r}) for "
                        "this record -- state may have been edited independently of its claimed "
                        "producing event, or an unrelated event was substituted (INV-R7)")
            else:  # call_status == "failure"
                if matched_record.second_rater_decision is not None:
                    raise ReportingConsistencyError(
                        f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                        "call_status='failure', but the matched record's own second_rater_decision is "
                        f"{matched_record.second_rater_decision!r} (expected None) -- "
                        "_second_rater_screen() never leaves a failed call's decision set (INV-R7)")
                if e.output_artifact_refs != []:
                    raise ReportingConsistencyError(
                        f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                        f"call_status='failure' but output_artifact_refs={e.output_artifact_refs!r} -- a "
                        "failure never has an output (INV-06)")
                if e.failure_detail is None:
                    raise ReportingConsistencyError(
                        f"screening_qc_second_rater event {e.event_id!r} (record_id={record_id!r}) has "
                        "call_status='failure' but failure_detail is None -- a genuine failure always "
                        "carries a real error description (INV-R7)")
    # Codex round-6 review, finding 1, P1 (second half): round-5's own
    # "screening_qc_state is None -> zero screening_qc_second_rater events
    # may exist anywhere" check is deleted -- it is WRONG given the event
    # store's own intentionally append-only design. `screening_qc_node`
    # can legitimately run (with real, durable events) in an EARLIER
    # invocation of this same `review_run_id`, then be disabled in a
    # LATER invocation (`screening_qc_state is None` in the current
    # state) -- the earlier invocation's own events correctly remain on
    # disk and must not be treated as "stray"/tampering evidence. No
    # positive-side check replaces this: `screening_qc_state is None` is
    # not itself a fact this cross-check can verify further, since there
    # is no "current invocation's own producing_event_ids" to anchor a
    # completeness check on when QC did not run this invocation.

    # synthesis_claim_map / _claim_refine / _imrad_narrative (Codex round-1
    # review, finding 2, P1): gated on the SAME configuration/output facts
    # synthesis_node itself gates on (`if requested_targets:` for
    # claim_map/claim_refine, `if cfg.synthesis.generate_legacy_imrad_
    # narrative:` for imrad_narrative) -- deliberately NOT on
    # `stage in oversight_mode_state`. Gating on the state dict's own key
    # let deleting BOTH oversight_mode/automation_origin entries for a
    # stage silently bypass this cross-check even when the durable,
    # already-verified synthesis output (`requested_targets`) or the
    # active configuration prove the stage genuinely executed.
    # `synthesis_output.model.requested_targets` is durable/re-verified
    # (VerifiedProjection), not itself part of what a caller could erase
    # by tampering with the mutable oversight_mode/automation_origin dicts.
    if synthesis_requested_targets:
        for stage in ("synthesis_claim_map", "synthesis_claim_refine"):
            events = [e for e in all_events if e.stage == stage]
            origin = _automation_origin_for_stage(events)
            _check(stage, origin, "automation_unreviewed")
    if imrad_narrative_enabled:
        events = [e for e in all_events if e.stage == "synthesis_imrad_narrative"]
        origin = _automation_origin_for_stage(events)
        _check("synthesis_imrad_narrative", origin, "automation_unreviewed")


# ---------------------------------------------------------------------------
# §2.7 -- synthesis verification (reporting-side entry point)
# ---------------------------------------------------------------------------

def load_and_verify_synthesis_output(
    *, cfg: ReviewConfig, records: List["Record"], extraction_raw: Dict[str, Any],
    qa_raw: Optional[Dict[str, Any]], review_run_id: str, event_store: AIUseEventStore,
    raw_synthesis_state: Optional[Dict[str, Any]],
) -> VerifiedProjection:
    if raw_synthesis_state is None:
        raise ReportingConsistencyError("state['synthesis'] is missing -- every review graph routes through synthesis_node")
    output = SynthesisOutput(**raw_synthesis_state)
    ctx = build_synthesis_verification_context(
        cfg=cfg, records=records, extraction_raw=extraction_raw, qa_raw=qa_raw,
        review_run_id=review_run_id, event_store=event_store,
    )
    verify_synthesis_output_sources(output, **ctx)
    return VerifiedProjection(model=output, content_hash=_hash_value(output))


# ---------------------------------------------------------------------------
# §2.1 -- capture_reporting_snapshot()
# ---------------------------------------------------------------------------

def capture_reporting_snapshot(
    state: ReviewState, cfg: ReviewConfig, review_run_id: str,
    skip_search_strategy_cross_check: bool = False,
) -> ReportingSourceSnapshot:
    screening_records = [Record.from_dict(r) for r in state["deduplicated_records"]]
    full_text_records = [Record.from_dict(r) for r in state["full_text_records"]]
    # Codex round-10 review, finding 3, P1: captured ONCE here, deep-copied
    # via Record.from_dict, instead of being re-read live/ad-hoc from state
    # at cross-check time -- same "captured once, immutable" discipline as
    # every other graph-state value on this snapshot (INV-R1/INV-R19).
    screened_records_full = [Record.from_dict(r) for r in state.get("screened_records_full") or []]

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)

    events = _load_verified_events(event_store)
    ledger_records = _load_verified_ledger_records(ledger_store)
    current_batches = _load_current_screening_batches(
        records=screening_records, cfg=cfg, review_run_id=review_run_id, event_store=event_store)
    extraction_bundle = _load_extraction_bundle(state, cfg, events=[vf.model for vf in events])

    # Finding 3 (Codex round-1 review, P1): canonicalize `full_text_records`
    # into `charted_data`'s own canonical order BEFORE synthesis
    # verification, not only inside the evidence-map reconstruction.
    # `build_synthesis_verification_context()`'s own docstring requires
    # `records` to ALREADY be in this canonical order ("this function does
    # not reorder") -- passing the live, possibly-reordered
    # `state["full_text_records"]` list straight through violated that
    # documented precondition, so a harmless reorder of otherwise-valid
    # full-text state (record identity/content unchanged) could make
    # reporting abort. Canonicalizing once here, up front, means every
    # downstream consumer (synthesis verification, evidence-map
    # reconstruction, BERTopic validation, the persisted snapshot field
    # itself) sees the SAME canonical order `synthesis_node` itself used.
    if extraction_bundle is not None:
        full_text_records = canonicalize_corpus_order(
            full_text_records=full_text_records,
            charted=list(extraction_bundle.raw.get("resolved_projection", [])),
            error_prefix="reporting canonical corpus order",
        )

    qa_bundle = _load_qa_bundle(state, cfg)
    screening_qc_state = _validate_screening_qc_state(state.get("screening_qc"))
    ft_native = capture_ft_native_provenance(state, cfg)

    screening_review_resolutions_raw = _validate_list_of_dicts(
        state.get("screening_review_resolutions") or [], name="screening_review_resolutions")
    # Finding 5 (Codex round-1 review, P1/M8): full_text_review_resolutions
    # was not captured into the snapshot at all -- M8's own crosswalk claim
    # ("every review-queue's resolution/confirmation audit trail") was
    # false without it. Captured the same way screening_review_resolutions
    # already is: a validated, deep-copied graph_state value.
    full_text_review_resolutions_raw = _validate_list_of_dicts(
        state.get("full_text_review_resolutions") or [], name="full_text_review_resolutions")
    oversight_mode_state = copy.deepcopy(dict(state.get("oversight_mode") or {}))
    automation_origin_state = copy.deepcopy(dict(state.get("automation_origin") or {}))

    synthesis_output = load_and_verify_synthesis_output(
        cfg=cfg, records=full_text_records,
        extraction_raw=extraction_bundle.raw if extraction_bundle is not None else {},
        qa_raw=qa_bundle.raw if qa_bundle is not None else None,
        review_run_id=review_run_id, event_store=event_store,
        raw_synthesis_state=state.get("synthesis"))

    # Codex round-5 review, finding 3: the search_strategy cross-check
    # verifies against THIS invocation's own recorded event_id(s)
    # (`SearchStrategy.producing_event_ids`), not the whole stage-name-
    # keyed event pool -- see `_cross_check_oversight_and_automation_
    # origin()`'s own docstring note at that check.
    search_strategy_producing_event_ids = list((state.get("search_strategy") or {}).get("producing_event_ids") or [])
    # Codex round-6/7 review: the CURRENT db_queries/expanded_query/
    # concepts/notes content, used to bind both events' own claimed output
    # to this invocation's actual SearchStrategy state.
    _search_strategy_state = state.get("search_strategy") or {}
    search_strategy_db_queries = dict(_search_strategy_state.get("db_queries") or {})
    search_strategy_expanded_query = _search_strategy_state.get("expanded_query") or ""
    search_strategy_concepts = list(_search_strategy_state.get("concepts") or [])
    search_strategy_notes = list(_search_strategy_state.get("notes") or [])
    # Codex round-9 review, finding 1: needed to prove a failed expansion
    # left expanded_query genuinely unchanged from the seed query.
    search_strategy_seed_query = _search_strategy_state.get("seed_query") or ""

    # Finding 2 (Codex round-1 review, P1): the cross-check runs AFTER
    # synthesis_output is available, so the synthesis-stage gate can use
    # its own durable, re-verified `requested_targets` fact rather than
    # the mutable oversight_mode/automation_origin dicts it is itself
    # meant to be checking.
    _cross_check_oversight_and_automation_origin(
        cfg=cfg, oversight_mode_state=oversight_mode_state, automation_origin_state=automation_origin_state,
        ledger_store=ledger_store, event_store=event_store, current_screening_batches=current_batches,
        screening_review_resolutions=screening_review_resolutions_raw,
        synthesis_requested_targets=synthesis_output.model.requested_targets,
        imrad_narrative_enabled=cfg.synthesis.generate_legacy_imrad_narrative,
        screening_qc_state=screening_qc_state,
        query_expansion_enabled=cfg.search.strategy_builder.enabled,
        search_strategy_producing_event_ids=search_strategy_producing_event_ids,
        search_strategy_db_queries=search_strategy_db_queries,
        search_strategy_expanded_query=search_strategy_expanded_query,
        search_strategy_concepts=search_strategy_concepts,
        search_strategy_notes=search_strategy_notes,
        search_strategy_seed_query=search_strategy_seed_query,
        screened_records_full=screened_records_full, screening_records=screening_records,
        skip_search_strategy_check=skip_search_strategy_cross_check,
    )

    config_projection = ReportingConfigProjection.from_config(cfg)

    # T1/INV-R27: validate, don't just is-not-None-check. A raw {} or
    # malformed dict raises HERE.
    raw_evidence_map = synthesis_output.model.evidence_map
    validated_evidence_map = EvidenceMapModel.model_validate(raw_evidence_map) if raw_evidence_map is not None else None
    expected_evidence_map_present = bool(config_projection.generate_evidence_map and full_text_records)
    if expected_evidence_map_present != (validated_evidence_map is not None):
        raise ReportingConsistencyError(
            f"evidence_map presence ({validated_evidence_map is not None}) does not match "
            f"generate_evidence_map({config_projection.generate_evidence_map}) and "
            f"full_text_records({len(full_text_records)}) -- expected {expected_evidence_map_present}")
    # V2/INV-R4: tie the validated map back to the ACTUAL corpus size.
    if validated_evidence_map is not None and validated_evidence_map.n_studies != len(full_text_records):
        raise ReportingConsistencyError(
            f"evidence_map.n_studies ({validated_evidence_map.n_studies}) != len(full_text_records) "
            f"({len(full_text_records)}) -- the evidence map does not account for the full corpus")
    # W1/X1/X2/Y2: full reconstruction-and-compare -- the authoritative
    # content proof.
    if validated_evidence_map is not None:
        expected_raw_evidence_map = _reconstruct_expected_evidence_map(
            full_text_records=full_text_records,
            extraction_raw=extraction_bundle.raw if extraction_bundle is not None else {})
        if (expected_raw_evidence_map is None
                or _pydantic_json_mode(expected_raw_evidence_map) != raw_evidence_map):
            raise ReportingConsistencyError(
                "evidence_map does not match the map reconstructed from full_text_records and the "
                "verified extraction projection -- rows, categories, or values disagree with the actual corpus")

    # T2/INV-R28: BERTopic consistency, checked once, before any render branch is chosen.
    _validate_bertopic_consistency(
        run_bertopic=config_projection.run_bertopic, n_records=len(full_text_records),
        excluded=synthesis_output.model.bertopic_excluded_empty_evidence_count,
        topics=synthesis_output.model.topics)

    return ReportingSourceSnapshot(
        review_run_id=review_run_id,
        config_projection=config_projection, config_projection_hash=_hash_value(config_projection),
        raw_records=_state_value(_validate_list_of_dicts(state["raw_records"], name="raw_records")),
        screening_records=screening_records, full_text_records=full_text_records,
        screened_records_full=screened_records_full,
        prisma=_state_value(PRISMACounts(**state["prisma_counts"])),
        search_log=_state_value(SearchLog(**state["search_log"])),
        screening_review_resolutions=_state_value(screening_review_resolutions_raw),
        full_text_review_resolutions=_state_value(full_text_review_resolutions_raw),
        full_text_review_queue=_state_value(_validate_list_of_dicts(state.get("full_text_review_queue") or [], name="full_text_review_queue")),
        screening_qc_state=screening_qc_state,
        oversight_mode_state=oversight_mode_state,
        automation_origin_state=automation_origin_state,
        events=events, ledger_records=ledger_records, current_screening_batch_results=current_batches,
        extraction=extraction_bundle, quality_appraisal=qa_bundle,
        synthesis_output=synthesis_output, evidence_map=validated_evidence_map, ft_native_provenance=ft_native,
        reference_context=(
            ReferenceReportingContext.model_validate(cfg.reference_reporting_context)
            if cfg.reference_reporting_context is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# §2.3 -- PRISMA diagram overlay -- unconditionally fatal, all three boxes
# ---------------------------------------------------------------------------

# Codex round-7 review, finding 1, P1: the EXACT, complete key set
# `fulltext.py`'s own `resolve_full_text_review_queue()` writes for an
# `"acquisition"`-stage `full_text_review_resolutions` entry (its
# `log_entry` dict literal in the `stage == "acquisition"` branch) --
# nothing else, ever. Used to reject a genuine ft_screening resolution
# relabeled to "acquisition" while its FT-only fields (resolution_id,
# reviewed_baseline_decision_id, reason, ...) were left attached.
_ACQUISITION_RESOLUTION_KEYS = frozenset({"record_id", "stage", "action", "reviewer", "note", "resolved_at"})


def compute_selection_splits(*, snapshot: ReportingSourceSnapshot, prisma: PRISMACounts) -> Dict[str, Dict[str, int]]:
    ledger_records = [vf.model for vf in snapshot.ledger_records]

    # Codex round-3 review, finding 2, P1: the exclusion-sum check below
    # (unchanged) is necessary but not sufficient -- it says nothing about
    # whether the ledger has a current tail for EVERY record that should
    # have one. Codex round-4 review, finding 4, P1: round-3's own fix
    # compared only total COUNTS per stage -- an attacker could swap one
    # record's current-tail entry for an unrelated/stale record's entry,
    # preserving the total count (and the exclusion-predicate sum) while
    # corrupting WHICH records are actually covered. Fixed to exact
    # record-ID SET equality (which subsumes a pure count check) against
    # an authoritative expected-ID set per stage, confirmed directly
    # against each producing node's own real behavior:
    #   - dedup: dedup_node (nodes/dedup.py) emits exactly one ledger
    #     decision per RAW record, unconditionally, regardless of
    #     unique/duplicate outcome -> {r["id"] for r in snapshot.raw_records.model}.
    #   - title_abstract_screening: screening_node (nodes/screening.py)
    #     emits exactly one ledger decision per DEDUPLICATED record,
    #     unconditionally, in both automated and human-review modes ->
    #     {r.id for r in snapshot.screening_records}.
    #   - full_text_eligibility: fulltext_node (nodes/fulltext.py) emits
    #     an `ai` entry only for a confirmed PDF whose FT_SCREENING
    #     execution_status=="completed"; every OTHER confirmed PDF
    #     (needs_human_review, or a not-"completed" outcome) is queued
    #     instead and later gets its own ledger entry (an `ai`-then-
    #     `human` supersession, or a fresh `human` root) once resolved --
    #     never both an `ai` entry AND a fresh `human` root for the same
    #     record. By the time reporting_node runs, `full_text_review_
    #     queue` must already be empty (the graph's own routing gate
    #     never reaches reporting_node otherwise), so EVERY confirmed
    #     record has exactly one current tail. The authoritative
    #     "confirmed" set itself is FT_SCREENING's own `item_results.jsonl`
    #     (a genuinely independent, separately-hash-verified artifact --
    #     see `capture_ft_native_provenance()`), captured as
    #     `snapshot.ft_native_provenance["confirmed_record_ids"]`.
    # This closes: "deleting every unique/include decision still passes
    # when exclusion counts match", "an empty ledger passes a one-record,
    # zero-exclusion fixture", AND "replacing a missing expected record
    # with an unknown/stale record preserves the count and exclusion
    # totals" -- all invisible to a count-only or exclusion-only check.
    expected_id_sets = {
        "dedup": {r["id"] for r in snapshot.raw_records.model},
        "title_abstract_screening": {r.id for r in snapshot.screening_records},
        "full_text_eligibility": (
            set(snapshot.ft_native_provenance["confirmed_record_ids"])
            if snapshot.ft_native_provenance is not None else set()
        ),
    }
    for stage, expected_ids in expected_id_sets.items():
        stage_tails = _tails_by_key([r for r in ledger_records if r.stage == stage])
        actual_ids = {record_id for record_id, _stage in stage_tails}
        if actual_ids != expected_ids:
            raise ReportingConsistencyError(
                f"{stage}: ledger current-tail record-ID set does not match the expected corpus -- "
                f"missing={sorted(expected_ids - actual_ids)!r}, "
                f"unexpected={sorted(actual_ids - expected_ids)!r} (exactly one current tail must "
                "exist for every, and only every, record this stage decides on)")

    # (b) Resolution-to-ledger linkage: a record named in the durable
    # human-resolution audit trail MUST have a matching ledger entry
    # SOMEWHERE in its own history (record_id, stage, producer_artifact_id
    # == this resolution's own resolution_id, decision_maker=="human") --
    # NOT necessarily as the current tail. Codex round-4 review, finding
    # 1, P1: round-3's own fix wrongly required the CURRENT tail to be
    # "human", which rejects the plan's own explicitly APPROVED
    # `A0 -> H -> A1` regression (aggregation_export_reporting_plan_2026-
    # 08-25.md's screening-regression table: "A0 -> H -> A1 -> origin
    # reflects A1, oversight stays 'human_resolved'") -- a LATER
    # legitimate AI/rule_based rerun may validly supersede a human
    # decision; oversight_mode staying "human_resolved" means a human
    # resolution occurred SOMEWHERE in this record's history, not that it
    # remains current. Anchoring the search on this exact resolution's own
    # `resolution_id` (rather than merely "some human entry exists for
    # this record/stage") still catches genuine tampering: deleting the
    # human decision's own ledger FILE removes it from `ledger_records`
    # entirely, so no matching entry is found at all, regardless of
    # whether it would have been the tail.
    # Codex round-5 review, finding 4, P2: the round-4 linkage check only
    # confirmed SOME ledger entry existed with a matching `record_id`/
    # `stage`/`producer_artifact_id`/`decision_maker` -- it never
    # recomputed `resolution_id` from the resolution row's own fields
    # (so a hand-tampered `resolution_id` unrelated to the row's real
    # content would still "link"), never bound the match via the FULL
    # expected `source_event_id` (only the looser `producer_artifact_id`),
    # and never cross-checked that the linked ledger entry's own
    # `decision` actually equals what the resolution row claims it
    # decided. Fixed to: (1) recompute `resolution_id` from the row's own
    # fields via the SAME `compute_resolution_id()` the producing node
    # itself used and require an exact match; (2) recompute the FULL
    # expected `source_event_id` (`derive_source_event_id()`) and require
    # the linked ledger entry's OWN `source_event_id` field equals it
    # exactly -- `SelectionDecisionRecord._check_producer_artifact_id_
    # binding`'s own validator already ties `source_event_id` to
    # `producer_artifact_id` structurally, so this subsumes the older,
    # looser `producer_artifact_id`-only match; (3) require the linked
    # entry's own `decision` field equals the resolution row's own claimed
    # decision/action -- a resolution claiming "exclude" must never link
    # to a ledger entry recording "include".
    for resolution in snapshot.screening_review_resolutions.model:
        record_id = resolution["record_id"]
        resolution_id = resolution["resolution_id"]
        expected_resolution_id = compute_resolution_id(
            stage="title_abstract_screening", record_id=record_id, action=resolution["decision"],
            reviewer=resolution["reviewer"], note=resolution["note"], resolved_at=resolution["resolved_at"],
            reviewed_baseline_decision_id=resolution["reviewed_baseline_decision_id"],
        )
        if resolution_id != expected_resolution_id:
            raise ReportingConsistencyError(
                f"title_abstract_screening: screening_review_resolutions entry for record {record_id!r} "
                f"has resolution_id={resolution_id!r}, but recomputing compute_resolution_id() from this "
                f"same entry's own fields yields {expected_resolution_id!r} -- the audit-log row is "
                "malformed or has been tampered with")
        expected_source_event_id = derive_source_event_id(
            producer_artifact_id=resolution_id, record_id=record_id, stage="title_abstract_screening")
        match = next(
            (r for r in ledger_records
             if r.stage == "title_abstract_screening" and r.record_id == record_id
             and r.source_event_id == expected_source_event_id and r.decision_maker == "human"),
            None)
        if match is None:
            raise ReportingConsistencyError(
                f"title_abstract_screening: record {record_id!r} was human-resolved "
                f"(resolution_id={resolution_id!r}) per screening_review_resolutions, but no ledger "
                f"entry with source_event_id={expected_source_event_id!r} and decision_maker='human' "
                "exists anywhere in this record's history -- the human decision's own ledger file may "
                "have been deleted")
        if match.decision != resolution["decision"]:
            raise ReportingConsistencyError(
                f"title_abstract_screening: record {record_id!r}'s linked ledger entry records "
                f"decision={match.decision!r}, but screening_review_resolutions claims "
                f"decision={resolution['decision']!r} for this same resolution -- the audit-log row "
                "does not match what was actually decided")
        # Codex round-6 review, finding 3(a), P1: matching by
        # `source_event_id`/`decision_maker`/`decision` alone does not
        # prove WHICH baseline was actually superseded -- a self-
        # consistent resolution_id/source_event_id could still link to a
        # ledger entry that supersedes a DIFFERENT decision than the one
        # this resolution claims it reviewed. Require the linked entry's
        # own `supersedes_decision_id` to equal this resolution's own
        # claimed `reviewed_baseline_decision_id` exactly.
        if match.supersedes_decision_id != resolution["reviewed_baseline_decision_id"]:
            raise ReportingConsistencyError(
                f"title_abstract_screening: record {record_id!r}'s linked ledger entry supersedes "
                f"{match.supersedes_decision_id!r}, but screening_review_resolutions claims this "
                f"resolution reviewed baseline {resolution['reviewed_baseline_decision_id']!r} -- the "
                "resolution's own claimed baseline does not match what its ledger entry actually "
                "superseded")

    for resolution in snapshot.full_text_review_resolutions.model:
        # Codex round-6 review, finding 3(b), P1: a `full_text_review_
        # resolutions` entry whose `stage` was anything other than exactly
        # `"ft_screening"` was previously silently SKIPPED entirely --
        # changing a genuine ft_screening resolution's own `stage` field
        # to `"acquisition"` (or any unknown/tampered value) bypassed
        # every linkage check below with no error at all. `stage` must be
        # a member of the closed set fulltext.py's own
        # `resolve_full_text_review_queue()` actually produces.
        stage = resolution.get("stage")
        if stage not in ("acquisition", "ft_screening"):
            raise ReportingConsistencyError(
                f"full_text_review_resolutions: entry for record {resolution.get('record_id')!r} has "
                f"stage={stage!r}, which is neither 'acquisition' nor 'ft_screening' -- an unknown or "
                "tampered stage value must never silently bypass linkage validation (INV-R7)")
        if stage == "acquisition":
            # No confirmed report ever reached FT_SCREENING for this
            # record -- there is genuinely no full_text_eligibility ledger
            # entry to link against (fulltext.py's own resolve_full_text_
            # review_queue(), stage=="acquisition" branch: the ONLY legal
            # action is "not_retrieved", and the row carries no
            # resolution_id/reviewed_baseline_decision_id/source_event_id/
            # reason at all -- validate the row's own real shape instead
            # of attempting a ledger linkage that cannot exist).
            #
            # Codex round-7 review, finding 1, P1: the round-6 fix above
            # validated the required fields' PRESENCE but never rejected
            # EXTRA keys, and never cross-checked the record against any
            # independent evidence of whether it genuinely reached
            # FT_SCREENING. A genuine ft_screening resolution (which DOES
            # carry resolution_id/reviewed_baseline_decision_id/reason) can
            # be relabeled stage="acquisition"/action="not_retrieved" while
            # simply KEEPING those extra fields -- the old check ignored
            # them and still skipped ledger linkage entirely, letting a
            # report simultaneously claim "not retrieved" in M8 while the
            # very same record has a real full-text human decision in the
            # ledger. Fixed with three independent checks: (1) the row's
            # OWN key set must be EXACTLY the real acquisition shape
            # `fulltext.py`'s `resolve_full_text_review_queue()` itself
            # produces (confirmed directly against that function's own
            # `log_entry` dict literal for the acquisition branch: {
            # "record_id", "stage", "action", "reviewer", "note",
            # "resolved_at"} -- nothing else, ever) -- any extra key
            # raises; (2) the record must be ABSENT from FT_SCREENING's own
            # independently-verified `confirmed_record_ids` (established in
            # an earlier round from `item_results.jsonl`'s own real typed
            # models) -- presence there proves the record demonstrably
            # reached FT_SCREENING; (3) NO `full_text_eligibility`-stage
            # ledger record (historical OR current, not just "not the
            # current tail") may exist anywhere for this record_id --
            # existence there proves a real eligibility decision was made.
            actual_keys = frozenset(resolution.keys())
            if actual_keys != _ACQUISITION_RESOLUTION_KEYS:
                raise ReportingConsistencyError(
                    f"full_text_review_resolutions: acquisition-stage entry for record "
                    f"{resolution.get('record_id')!r} has key set {sorted(actual_keys)!r}, expected "
                    f"exactly {sorted(_ACQUISITION_RESOLUTION_KEYS)!r} -- fulltext.py's own "
                    "resolve_full_text_review_queue() never attaches resolution_id/"
                    "reviewed_baseline_decision_id/reason/source_event_id to an acquisition-stage entry "
                    "(no assessment ever happened); an extra key means a genuine ft_screening resolution "
                    "may have been relabeled to hide it from ledger linkage")
            record_id = resolution.get("record_id")
            if not record_id or not isinstance(record_id, str):
                raise ReportingConsistencyError(
                    "full_text_review_resolutions: an acquisition-stage entry is missing a well-formed "
                    "record_id")
            if resolution.get("action") != "not_retrieved":
                raise ReportingConsistencyError(
                    f"full_text_review_resolutions: acquisition-stage entry for record {record_id!r} has "
                    f"action={resolution.get('action')!r}, but an acquisition-stage entry (no confirmed "
                    "report ever reached FT_SCREENING) can only ever be resolved 'not_retrieved'")
            for field in ("reviewer", "note", "resolved_at"):
                if not resolution.get(field):
                    raise ReportingConsistencyError(
                        f"full_text_review_resolutions: acquisition-stage entry for record {record_id!r} "
                        f"is missing a non-blank {field!r}")
            confirmed_ids = (
                set(snapshot.ft_native_provenance["confirmed_record_ids"])
                if snapshot.ft_native_provenance is not None else set()
            )
            if record_id in confirmed_ids:
                raise ReportingConsistencyError(
                    f"full_text_review_resolutions: record {record_id!r} is labeled stage='acquisition' "
                    "(claiming no confirmed report ever reached FT_SCREENING), but it IS present in "
                    "FT_SCREENING's own independently-verified confirmed_record_ids -- this record "
                    "demonstrably reached FT_SCREENING, contradicting the acquisition label")
            if any(r.record_id == record_id and r.stage == "full_text_eligibility" for r in ledger_records):
                raise ReportingConsistencyError(
                    f"full_text_review_resolutions: record {record_id!r} is labeled stage='acquisition', "
                    "but a full_text_eligibility ledger record exists for it (historical or current) -- "
                    "this record demonstrably received a real full-text eligibility decision, "
                    "contradicting the acquisition label")
            continue
        # stage == "ft_screening"
        record_id = resolution["record_id"]
        resolution_id = resolution["resolution_id"]
        expected_resolution_id = compute_resolution_id(
            stage="full_text_eligibility", record_id=record_id, action=resolution["action"],
            reviewer=resolution["reviewer"], note=resolution["note"], resolved_at=resolution["resolved_at"],
            reviewed_baseline_decision_id=resolution["reviewed_baseline_decision_id"],
            exclusion_reason=resolution.get("reason"),
        )
        if resolution_id != expected_resolution_id:
            raise ReportingConsistencyError(
                f"full_text_eligibility: full_text_review_resolutions entry for record {record_id!r} "
                f"has resolution_id={resolution_id!r}, but recomputing compute_resolution_id() from this "
                f"same entry's own fields yields {expected_resolution_id!r} -- the audit-log row is "
                "malformed or has been tampered with")
        expected_source_event_id = derive_source_event_id(
            producer_artifact_id=resolution_id, record_id=record_id, stage="full_text_eligibility")
        match = next(
            (r for r in ledger_records
             if r.stage == "full_text_eligibility" and r.record_id == record_id
             and r.source_event_id == expected_source_event_id and r.decision_maker == "human"),
            None)
        if match is None:
            raise ReportingConsistencyError(
                f"full_text_eligibility: record {record_id!r} was human-resolved "
                f"(resolution_id={resolution_id!r}) per full_text_review_resolutions, but no ledger "
                f"entry with source_event_id={expected_source_event_id!r} and decision_maker='human' "
                "exists anywhere in this record's history -- the human decision's own ledger file may "
                "have been deleted")
        if match.decision != resolution["action"]:
            raise ReportingConsistencyError(
                f"full_text_eligibility: record {record_id!r}'s linked ledger entry records "
                f"decision={match.decision!r}, but full_text_review_resolutions claims "
                f"action={resolution['action']!r} for this same resolution -- the audit-log row does not "
                "match what was actually decided")
        # Codex round-6 review, finding 3(a), P1 (see the equivalent
        # screening-stage check above for the full rationale).
        if match.supersedes_decision_id != resolution["reviewed_baseline_decision_id"]:
            raise ReportingConsistencyError(
                f"full_text_eligibility: record {record_id!r}'s linked ledger entry supersedes "
                f"{match.supersedes_decision_id!r}, but full_text_review_resolutions claims this "
                f"resolution reviewed baseline {resolution['reviewed_baseline_decision_id']!r} -- the "
                "resolution's own claimed baseline does not match what its ledger entry actually "
                "superseded")

    splits = {}
    for box, stage, predicate, expected_total in [
        ("dedup_removed", "dedup", is_dedup_duplicate_decision, prisma.total_identified - prisma.after_dedup),
        ("title_abstract_excluded", "title_abstract_screening", is_screening_exclusion_decision, prisma.excluded_title_abstract),
        ("full_text_excluded", "full_text_eligibility", is_fulltext_exclusion_decision, prisma.excluded_full_text),
    ]:
        tails = _tails_by_key([r for r in ledger_records if r.stage == stage])
        by_maker = Counter(rec.decision_maker for rec in tails.values() if predicate(rec.decision))
        if sum(by_maker.values()) != expected_total:
            raise ReportingConsistencyError(f"{box}: ledger split totals {sum(by_maker.values())} != PRISMACounts {expected_total}")
        splits[box] = dict(by_maker)
    return splits


# ---------------------------------------------------------------------------
# §2.9 -- built_from dependency closure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 9 -- `built_from` as `slr://` locator references (plan Sec 10.1/10.3)
# ---------------------------------------------------------------------------
# Supersedes the pre-step-9 `build_source_hashes()`/`BUILT_FROM_TABLE` pair
# (an internal-hash-keyed dict, e.g. {"quality_appraisal_resolution_content_
# sha256": "<hex>"}) -- `ReportingStageEntry.built_from` is now a validated
# `List[str]` of `slr://` locators/globs (export_manifest.py), never a raw
# hash dict: a hash proves two renders were byte-identical but cannot be
# resolved back to WHAT was depended on, whereas a locator names the exact
# other manifest entry (or entry family, via a `LocatorGlob`) that fed this
# renderer -- and the byte-level determinism guarantee `build_source_hashes`
# existed for is unaffected: each entry's own `file_sha256`/`internal_hash`
# still changes whenever its source changes, and `manifest_hash` still
# covers every entry, so any upstream change is still visible end-to-end.

def _renderer_locator_pool(snapshot: ReportingSourceSnapshot) -> Dict[str, List[str]]:
    """The pool of `slr://` locators/globs each renderer's own `built_from`
    is assembled from -- one list per named building block, empty when the
    underlying source has no data (a glob over zero matches is still a
    well-formed locator family, but listing it would claim a dependency
    that materially did not exist this run)."""
    ft_run_id = snapshot.ft_native_provenance["ft_screening_run_id"] if snapshot.ft_native_provenance else None
    extraction_review = snapshot.extraction.extraction_review if snapshot.extraction else None
    extraction_review_locators: List[str] = []
    if extraction_review is not None and snapshot.extraction is not None:
        xid = snapshot.extraction.run_id
        for subkind, subid, component in (
            ("review_packet", "json", extraction_review.review_packet_json),
            ("review_packet", "md", extraction_review.review_packet_md),
            ("queue_snapshot", None, extraction_review.queue_snapshot),
            ("suggestions_index", None, extraction_review.suggestions_index),
            ("decision_journal", None, extraction_review.decision_journal),
        ):
            if component is not None:
                extraction_review_locators.append(str(LogicalLocator("extraction", subkind, xid, subid)))
        extraction_review_locators.extend(
            str(LogicalLocator("extraction", "suggestion", xid, component.document_id))
            for component in extraction_review.suggestions
        )
    return {
        "events": [str(LocatorGlob("ai_use_events", "event"))] if snapshot.events else [],
        "ledger": [str(LocatorGlob("selection_ledger", "decision"))] if snapshot.ledger_records else [],
        "batches": [str(LocatorGlob("screening_batch", "result"))] if snapshot.current_screening_batch_results else [],
        "extraction": [str(LogicalLocator("extraction", "manifest", snapshot.extraction.run_id))] if snapshot.extraction else [],
        "extraction_review": extraction_review_locators,
        "quality_appraisal": [str(LogicalLocator("quality_appraisal", "manifest", snapshot.quality_appraisal.run_id))] if snapshot.quality_appraisal else [],
        "ft_screening": (
            [str(LogicalLocator("ft_screening", component, ft_run_id)) for component in ("run_result", "item_results", "ledger")]
            if ft_run_id else []
        ),
        # Step 14 ([C13], PRISMA-ScR renderers) additions. The plan's own §9/
        # §10.10 tables literally list bare tokens ("config_projection",
        # "reference_context") in these two renderers' frozen `built_from`
        # lists -- but `ReportingStageEntry.built_from`'s own validator
        # (export_manifest.py) requires every entry to be a real `slr://`
        # locator or glob, and the plan's own consumer of that exact table
        # (`open_bundle()`/`from_bundle()`, §10.10) is not implemented
        # anywhere in the codebase yet. `methods_md` above already diverges
        # from its own identically-shaped plan row (also literally "
        # config_projection"/"reference_context") the same way: real `slr://`
        # locators only, via this pool/`_BUILT_FROM_BLOCKS` mechanism.
        #
        # Deliberately NOT using `LogicalLocator("state", "snapshot", ...)`
        # here: `locators.py`'s own module docstring reserves the `state`/
        # `run_evidence`/`fixture`/`bundle` kinds for `PayloadManifestV1`
        # alone ("no code in this step ever constructs one") -- that
        # manifest is a step-15 concern, not yet implemented. Every bare
        # graph-state value this renderer reads (`prisma`, `search_log`,
        # `config_projection`, `reference_context`, `synthesis_output`) has
        # no `slr://` locator of its own for the same reason
        # `config_projection`/`reference_context` were dropped from
        # `methods_md`'s own built_from -- only the file-backed sources
        # below (already-published artifacts, extraction's own manifest)
        # get one.
        "ai_traceability_artifact": [str(LogicalLocator("reporting", "artifact", "ai_traceability_json"))],
        "prisma_scr_report_artifact": [str(LogicalLocator("reporting", "artifact", "prisma_scr_report_md"))],
    }


_BUILT_FROM_BLOCKS: Dict[str, List[str]] = {
    "prisma_diagram": ["ledger"],
    "methods_md": ["extraction", "quality_appraisal"],
    "ai_traceability_json": ["events", "ledger", "batches", "extraction", "quality_appraisal", "ft_screening"],
    "ai_traceability_md": ["events", "ledger", "batches", "extraction", "quality_appraisal", "ft_screening"],
    "quality_appraisal_csv": ["quality_appraisal"],
    "quality_appraisal_json": ["quality_appraisal"],
    "evidence_map_csv": ["extraction"],
    "synthesis_claims_json": ["events", "extraction", "quality_appraisal"],
    "synthesis_json": ["events", "extraction", "quality_appraisal"],
    "synthesis_md": ["events", "extraction", "quality_appraisal"],
    "claim_audit_json": ["events", "extraction", "quality_appraisal"],
    "claim_audit_md": ["events", "extraction", "quality_appraisal"],
    "extraction_resolution_review_json": ["extraction", "extraction_review"],
    "extraction_resolution_review_md": ["extraction", "extraction_review"],
    "prisma_scr_report_md": ["extraction", "ai_traceability_artifact"],
    "prisma_scr_conformance_json": ["ai_traceability_artifact", "prisma_scr_report_artifact"],
}


def _renderer_built_from(name: str, pool: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for block in _BUILT_FROM_BLOCKS[name]:
        out.extend(pool[block])
    return sorted(out)


# ---------------------------------------------------------------------------
# §2.10 -- ledger path validation
# ---------------------------------------------------------------------------

def _validate_ledger_dir(state: ReviewState, cfg: ReviewConfig, review_run_id: str) -> None:
    ledger_dir_claimed = state.get("selection_decision_ledger_dir")
    if not ledger_dir_claimed:
        raise ReportingConsistencyError("no selection_decision_ledger_dir in state")
    ledger_store = SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    if Path(ledger_dir_claimed).resolve() != ledger_store.decisions_dir.resolve():
        raise ReportingConsistencyError("selection_decision_ledger_dir does not match this run's own constructed directory")
    if not ledger_store.decisions_dir.resolve().is_relative_to(cfg.ensure_output_dir().resolve()):
        raise ReportingConsistencyError("selection_decision_ledger_dir resolves outside the configured output root")


# ---------------------------------------------------------------------------
# §2.14 -- atomic write + read-back self-verify
# ---------------------------------------------------------------------------

def _write_artifact_and_verify(path: Path, content_bytes: bytes) -> str:
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")
    try:
        with open(tmp, "wb") as f:
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    on_disk = path.read_bytes()
    if on_disk != content_bytes:
        raise ReportingConsistencyError(f"{path}: on-disk bytes do not match what was just written")
    return hashlib.sha256(on_disk).hexdigest()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_methods_md(snapshot: ReportingSourceSnapshot) -> str:
    """Adapted from the pre-v17 `_draft_methods_text()` -- SAME content
    sections, now reading from the snapshot instead of loose arguments,
    and with a deterministic search-date fallback (§2.15) instead of
    `date.today()`."""
    cfg = snapshot.config_projection
    prisma = snapshot.prisma.model
    search_log = snapshot.search_log.model
    review_queue = snapshot.full_text_review_queue.model
    extraction_artifact: Dict[str, Any] = snapshot.extraction.raw if snapshot.extraction else {}

    # Finding 6 (Codex round-1 review, P2): `prisma.identified` is a dict
    # whose insertion order is NOT part of `prisma_sha256` (a canonical-
    # JSON hash, key-order-independent) -- rendering it in raw dict
    # iteration order let two semantically-identical PRISMACounts values
    # (same content, different insertion order) produce the same hash but
    # different methods.md bytes. Sorted by key for determinism.
    db_list = ", ".join(sorted(prisma.identified.keys()))
    search_dates = sorted({e.date for e in search_log.entries})
    date_str = ", ".join(search_dates) if search_dates else "search date not recorded"

    rq_text = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(cfg.research_questions))
    inc_text = "\n".join(f"  - {c}" for c in cfg.inclusion_criteria)
    exc_text = "\n".join(f"  - {c}" for c in cfg.exclusion_criteria)

    review_label = {
        "scoping": "scoping review",
        "rapid": "rapid review",
        "slr": "systematic literature review",
    }.get(cfg.review_type, "review")

    framework = "Population-Concept-Context (PCC)" if cfg.review_type == "scoping" else "PICO"
    if cfg.review_type == "scoping":
        framework_detail = (
            f"  - Population: {cfg.population or 'Not specified'}\n"
            f"  - Concept: {cfg.concept or 'Not specified'}\n"
            f"  - Context: {cfg.context or 'Not specified'}"
        )
    else:
        framework_detail = (
            f"  - Population: {cfg.population or 'Not specified'}\n"
            f"  - Intervention: {cfg.intervention or 'Not specified'}\n"
            f"  - Comparator: {cfg.comparator or 'Not specified'}\n"
            f"  - Outcome: {cfg.outcome or 'Not specified'}"
        )

    # Finding 2 (Codex round-2 review, P1): three genuinely distinct cases,
    # not a scoping/non-scoping binary -- "not conducted, as is standard
    # practice" is a REVIEW-TYPE fact (never applicable to scoping,
    # regardless of what happened this run); a rapid/SLR review with
    # `snapshot.quality_appraisal is None` is a run-specific fact (QA WAS
    # configured to run, but there were no eligible full-text records to
    # appraise -- a valid empty-corpus outcome, not evidence QA "was
    # conducted"); only a rapid/SLR review with an actual QA bundle present
    # may claim QA was genuinely performed.
    if cfg.review_type == "scoping":
        qc_note = "Quality appraisal was not conducted, as is standard practice for scoping reviews."
    elif snapshot.quality_appraisal is not None:
        qc_note = f"Quality appraisal was conducted using the {cfg.quality_appraisal_checklist_id} checklist."
    else:
        qc_note = (
            "Quality appraisal was not performed this run: no eligible full-text records were "
            "available to appraise (see the PRISMA full-text assessment count above)."
        )

    retrieval = prisma.fulltext_retrieval
    not_retrieved_n = sum(n for status, n in retrieval.items() if status != "parsed")
    retrieval_rows = "\n".join(
        f"| {status.replace('_', ' ').capitalize()} | {n} |" for status, n in sorted(retrieval.items())
    ) or "| (no full-text retrieval attempted) | 0 |"

    queue = review_queue or []
    queue_note = (
        "No records remained in the full-text eligibility review queue."
        if not queue
        else (
            f"**{len(queue)} record(s) remain UNRESOLVED in the full-text eligibility review "
            "queue** (needs human review, a processing/provider failure, or full text that was "
            "never retrieved) -- this methods section and the PRISMA counts below do NOT yet "
            "reflect a final decision for them."
        )
    )

    # Finding 2 (Codex round-2 review, P1): "Data were extracted..." must
    # only be claimed when `snapshot.extraction is not None` -- a valid
    # empty-corpus run (zero full_text_records) never invokes
    # extraction_node at all, so extraction_artifact is `{}` and every
    # field below would be a fabricated N/A-filled claim rather than an
    # honest "did not run" statement.
    if snapshot.extraction is not None:
        artifact = extraction_artifact
        remaining_keys = artifact.get("remaining_queue_keys", [])
        resolutions = artifact.get("resolutions", [])
        total_fields = artifact.get("total_fields", 0)
        run_id = artifact.get("run_id")
        # Step 9 (plan Sec 10.5): the reporting boundary never renders an
        # absolute path -- `slr://extraction/manifest/<run_id>` names the
        # manifest logically; `extraction_run_id` (an id, not a path, fixed
        # within a single bundle) stays in the prose.
        manifest_locator = str(LogicalLocator("extraction", "manifest", run_id)) if run_id else None
        schema_hash = artifact.get("extraction_schema_hash")

        extraction_queue_note = (
            "No fields remained in the structured-extraction review queue."
            if not remaining_keys
            else (
                f"**{len(remaining_keys)} field(s) across the included full-text corpus remain "
                "UNRESOLVED in the structured-extraction review queue** (unclear, conflicting, or a "
                "failed extraction) -- charted_data does not yet reflect a final, human-reviewed "
                "value for them, and synthesis/reporting have not yet run over this corpus."
            )
        )

        resolved_extracted = sum(1 for r in resolutions if r.get("outcome") == "extracted")
        resolved_not_reported = sum(1 for r in resolutions if r.get("outcome") == "not_reported")
        total_resolved = len(resolutions)
        machine_final = max(total_fields - total_resolved - len(remaining_keys), 0)
        resolution_locator = str(LogicalLocator("extraction", "resolution", run_id)) if run_id else "N/A"
        review_summary = (
            "No field required human review; every field-level result below is the direct, "
            "unmodified machine extraction output."
            if total_resolved == 0
            else (
                f"{total_resolved} of {total_fields} field-level result(s) required human review and "
                f"resolution ({resolved_extracted} resolved as a human-attested extracted value, "
                f"{resolved_not_reported} resolved as not_reported); the remaining {machine_final} "
                "field(s) are the direct, unmodified machine extraction output. See the verified, "
                f"content-hashed extraction_resolutions.json artifact for this run ({resolution_locator}) "
                "for the full, audited reviewer/note/timestamp trail per "
                "field -- every count above is drawn from it, deduplicated by (record_id, field_id) "
                "and checked against the exact machine review set at publish time, never re-derived "
                "here or guessed."
            )
        )

        data_charting_intro = (
            "Data were extracted using the evidence-verified structured extraction stage described "
            "below, not the legacy descriptive charting methods (examples/legacy/charting.py)."
        )
        extraction_section = f"""
### Structured data extraction
Data were extracted using the evidence-verified structured-extraction engine
described in slr_pipeline/extraction/PROTOCOL.md (schema-driven, checkpointed,
provider-explicit). Run identity: extraction_run_id={run_id or 'N/A'},
extraction_schema_hash={schema_hash or 'N/A'} (see the complete frozen run
manifest: {manifest_locator or 'N/A'}).

{review_summary}

{extraction_queue_note}

NOTE: `charted_data` (used by the synthesis stage below) is a deterministic,
LOSSY compatibility projection of the complete extraction result -- one raw
value per field (the human-resolved value where applicable), with
evidence/confidence/rationale discarded -- and is never the scientific
artifact itself. Synthesis below consumes this resolved projection by
reloading and re-verifying the SAME durable extraction_resolutions.json
artifact this section's own counts came from, not raw graph state.
"""
    else:
        data_charting_intro = (
            "No structured data extraction was performed this run: no full-text records were "
            "available to extract from (extraction_node did not run)."
        )
        extraction_section = ""

    # Finding 4 (Codex round-1 review, P1): every claim below is rendered
    # from the actual snapshot, never asserted unconditionally regardless
    # of what genuinely happened this run.
    screening_review_resolutions = snapshot.screening_review_resolutions.model
    human_validation_note = (
        "Borderline cases received human validation via the screening review queue."
        if screening_review_resolutions
        else "No borderline cases required human validation this run (automated-mode routing, "
             "or none were queued for review)."
    )
    # Finding 2 (Codex round-2 review, P1): `n_valid_pairs > 0` alone is
    # NOT proof kappa was assessed -- the producer can legitimately return
    # `cohens_kappa=None` for a non-estimable sample (e.g. one rater's
    # ratings have zero variance). Three genuinely distinct outcomes:
    # QC didn't run/had no valid pairs at all; QC ran and had valid pairs
    # but kappa itself was not estimable; QC ran and kappa is a real value.
    qc_state = snapshot.screening_qc_state
    if qc_state is None or qc_state.irr.n_valid_pairs == 0:
        kappa_note = (
            "Inter-rater reliability (Cohen's kappa) was NOT assessed this run (screening QC did "
            "not run, or produced no valid second-rater pairs)."
        )
    elif qc_state.irr.cohens_kappa is None:
        kappa_note = (
            f"Inter-rater reliability was attempted with {qc_state.irr.n_valid_pairs} valid "
            "second-rater pair(s), but Cohen's kappa was not estimable for this sample (see "
            "ai_traceability.json's M9/R2 block for the full statistics)."
        )
    else:
        kappa_note = (
            "Inter-rater reliability was assessed using Cohen's kappa (see ai_traceability.json's "
            "M9/R2 block for the full statistics)."
        )
    evidence_map_note = "An evidence map was generated." if snapshot.evidence_map is not None else ""
    data_governance_text = cfg.data_governance_declaration or "Not provided."

    return f"""## Methods

### Study design
This study is a {review_label} conducted according to the {framework} framework.

### Research questions
{rq_text}

### {framework} framework
{framework_detail}

### Search strategy
A systematic search was conducted on {date_str} across the following databases:
{db_list}.

The search string was: `{cfg.search_query}`

### Inclusion and exclusion criteria

**Inclusion criteria:**
{inc_text}

**Exclusion criteria:**
{exc_text}

### Screening
Records were screened at title and abstract stage using an LLM-assisted screening
pipeline ({cfg.screening_llm_model}). {human_validation_note}
{kappa_note}

### Data charting
{data_charting_intro}
{extraction_section}
### {qc_note}

### Synthesis
{render_bertopic_note(snapshot)}
{evidence_map_note}

### Data governance and ethics
{data_governance_text}

### Full-text retrieval
| Outcome | n |
|---------|---|
{retrieval_rows}

Reports not retrieved (no confirmed full text -- excluded from full-text eligibility
assessment, never from acquisition attempts): {not_retrieved_n}.

### Full-text eligibility review queue
{queue_note}

### PRISMA flow
| Stage | n |
|-------|---|
| Identified | {prisma.total_identified} |
| After deduplication | {prisma.after_dedup} |
| Screened | {prisma.screened_title_abstract} |
| Excluded (title/abstract) | {prisma.excluded_title_abstract} |
| Full-text assessed (from a confirmed, retrieved PDF) | {prisma.assessed_full_text} |
| Full-text reports not retrieved | {not_retrieved_n} |
| Excluded (full-text) | {prisma.excluded_full_text} |
| Included | {prisma.included} |
"""


def _qa_export_rows(snapshot: ReportingSourceSnapshot) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if snapshot.quality_appraisal is None:
        return rows
    for proj in snapshot.quality_appraisal.raw["resolved_quality_projection"]:
        da = DocumentAppraisal.model_validate(proj)
        document_is_provisional = da.is_provisional
        overall_risk = da.rob2.overall_risk if da.rob2 is not None else None
        for field_id, leaf in da.leaves().items():
            if da.checklist_id == "rob2_inspired_study_level_teaching_v1":
                domain_id = field_id.split(".", 1)[0]
                domain = next(d for d in da.rob2.domains if d.domain_id == domain_id)
                domain_risk, domain_risk_rationale, basis = domain.domain_risk, domain.domain_risk_rationale, domain.basis
                status_or_answer = leaf.answer
            else:
                domain_id = domain_risk = domain_risk_rationale = basis = None
                status_or_answer = leaf.status
            rows.append({
                "record_id": da.record_id, "checklist_id": da.checklist_id, "field_id": field_id,
                "domain_id": domain_id, "status_or_answer": status_or_answer,
                "decision_origin": leaf.decision_origin, "leaf_is_provisional": leaf.provisional_pending_human_confirmation,
                "human_confirmation": leaf.human_confirmation.model_dump(mode="json") if leaf.human_confirmation is not None else None,
                "evidence": [e.model_dump(mode="json") for e in leaf.evidence],
                "evidence_status": leaf.evidence_status, "rationale": leaf.rationale,
                "coverage_complete": leaf.coverage_complete, "queue_reason": leaf.queue_reason,
                "conflicting_evidence": leaf.conflicting_evidence,
                "domain_risk": domain_risk, "domain_risk_rationale": domain_risk_rationale, "basis": basis,
                "document_is_provisional": document_is_provisional, "overall_risk": overall_risk,
            })
    return rows


_QA_CSV_FIELDNAMES = [
    "record_id", "checklist_id", "field_id", "domain_id", "status_or_answer",
    "decision_origin", "leaf_is_provisional", "human_confirmed", "evidence_count",
    "evidence_status", "rationale", "coverage_complete", "queue_reason",
    "conflicting_evidence", "domain_risk", "domain_risk_rationale", "basis",
    "document_is_provisional", "overall_risk",
]


def _render_qa_csv(snapshot: ReportingSourceSnapshot) -> bytes:
    rows = _qa_export_rows(snapshot)
    buf = io.StringIO()
    writer = csv_module.DictWriter(buf, fieldnames=_QA_CSV_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        csv_row = {k: v for k, v in row.items() if k not in ("evidence", "human_confirmation")}
        csv_row["human_confirmed"] = row["human_confirmation"] is not None
        csv_row["evidence_count"] = len(row["evidence"])
        csv_row["conflicting_evidence"] = (
            canonical_json_bytes(row["conflicting_evidence"]).decode("utf-8")
            if row["conflicting_evidence"] is not None else ""
        )
        writer.writerow(csv_row)
    return buf.getvalue().encode("utf-8")


def _render_qa_json_dict(snapshot: ReportingSourceSnapshot) -> Dict[str, Any]:
    return {"rows": _qa_export_rows(snapshot)}


def _render_evidence_map_csv(snapshot: ReportingSourceSnapshot) -> bytes:
    em = snapshot.evidence_map
    buf = io.StringIO()
    writer = csv_module.writer(buf)
    writer.writerow(["citation", "year", "country", "study_design", "population", "concept", "context"])
    for row in em.characteristics_table:
        cells = []
        for value in (row.citation, row.year, row.country, row.study_design, row.population, row.concept, row.context):
            if value is None:
                cells.append("")
            elif isinstance(value, list):
                cells.append(canonical_json_bytes(value).decode("utf-8"))
            else:
                cells.append(str(value))
        writer.writerow(cells)
    return buf.getvalue().encode("utf-8")


def _render_synthesis_claims_dict(snapshot: ReportingSourceSnapshot) -> Dict[str, Any]:
    output = snapshot.synthesis_output.model
    return {
        "claims": [c.model_dump(mode="json") for c in output.claims],
        "rejected_targets": [r.model_dump(mode="json") for r in output.rejected_targets],
        "claim_construction_coverage": [c.model_dump(mode="json") for c in output.claim_construction_coverage],
        "target_coverage": output.target_coverage,
    }


def _render_synthesis_md(snapshot: ReportingSourceSnapshot) -> str:
    output = snapshot.synthesis_output.model
    extraction_index = snapshot.extraction.raw.get("evidence_index", {}) if snapshot.extraction else {}
    qa_index = snapshot.quality_appraisal.raw.get("evidence_index", {}) if snapshot.quality_appraisal else {}

    def _resolve_quotations(refs) -> List[str]:
        quotations = []
        for ref in refs:
            index = qa_index if ref.artifact_kind == "quality_appraisal" else extraction_index
            for entry in index.get(ref.record_id, {}).get(ref.field_id, []):
                if entry.evidence_id == ref.evidence_id:
                    quotations.append(entry.quotation)
        return quotations

    lines = ["# Synthesis", ""]
    targets = sorted({c.rq_or_section for c in output.claims} | {r.rq_or_section for r in output.rejected_targets})
    rejected_by_target = {r.rq_or_section: r for r in output.rejected_targets}
    for target in targets:
        lines.append(f"## {target}")
        target_claims = [c for c in output.claims if c.rq_or_section == target]
        for c in target_claims:
            lines.append(f"- {c.claim_text}")
            for q in _resolve_quotations(c.references):
                lines.append(f"  > {q}")
        rejected = rejected_by_target.get(target)
        if rejected is not None:
            lines.append(f"_Rejected: {rejected.rejection_reason}_")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_claim_audit_sources(snapshot: ReportingSourceSnapshot) -> ClaimAuditSources:
    extraction_raw = snapshot.extraction.raw if snapshot.extraction else {}
    extraction_field_results: Dict[str, Dict[str, FieldResult]] = {
        doc["document_id"]: {f["field_id"]: FieldResult.model_validate(f) for f in doc["fields"]}
        for doc in extraction_raw.get("extraction_results", [])
    }
    extraction_index = extraction_raw.get("evidence_index", {})
    qa_index = snapshot.quality_appraisal.raw.get("evidence_index", {}) if snapshot.quality_appraisal else {}
    return ClaimAuditSources(
        synthesis_output=snapshot.synthesis_output.model, extraction_field_results=extraction_field_results,
        extraction_evidence_index=extraction_index, qa_evidence_index=qa_index,
    )


def _render_claim_audit_dict(snapshot: ReportingSourceSnapshot) -> Dict[str, Any]:
    return build_claim_evidence_audit(_build_claim_audit_sources(snapshot)).model_dump(mode="json")


def _render_claim_audit_md(snapshot: ReportingSourceSnapshot) -> str:
    audit = build_claim_evidence_audit(_build_claim_audit_sources(snapshot))
    lines = ["# Claim-Evidence Audit", ""]
    lines.append("## Unsupported targets (no claims constructed)")
    if audit.unsupported_targets:
        for target in audit.unsupported_targets:
            lines.append(f"- {target}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Claims")
    if not audit.claims:
        lines.append("(no claims)")
    for c in audit.claims:
        flags = []
        if c.single_study_supported:
            flags.append("single-study-supported")
        if c.conflicting_evidence:
            flags.append(f"conflicting evidence: {', '.join(c.conflicting_field_ids)}")
        flag_text = f" [{'; '.join(flags)}]" if flags else ""
        lines.append(f"- {c.claim_id} ({c.rq_or_section}){flag_text} -- records: {', '.join(c.record_ids)}")
    return "\n".join(lines) + "\n"


def _render_extraction_resolution_review_dict(snapshot: ReportingSourceSnapshot) -> Dict[str, Any]:
    extraction = snapshot.extraction
    artifacts = extraction.extraction_review if extraction else None
    if artifacts is None or artifacts.review_packet_json is None:
        raise ReportingConsistencyError("extraction review artifacts are unavailable")
    def read_verified(component: VerifiedComponent) -> bytes:
        payload = component.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != component.file_sha256:
            raise ReportingConsistencyError(f"extraction review component changed after snapshot: {component.path}")
        return payload

    packet = ReviewPacket.model_validate_json(read_verified(artifacts.review_packet_json))
    attempts: List[Any] = []
    for component in artifacts.suggestions:
        data = json.loads(read_verified(component))
        attempts.append(data)
    decisions = []
    if artifacts.decision_journal is not None:
        decisions = [entry.model_dump(mode="json") for entry in verify_decision_journal_bytes(
            read_verified(artifacts.decision_journal)
        )]
    fields: List[Dict[str, Any]] = []
    for entry in packet.entries:
        matching_attempts = [row for row in attempts if
                             row.get("record_id") == entry.record_id and row.get("field_id") == entry.field_id
                             and row.get("kind") in {"validated_suggestion", "rejected_suggestion"}]
        matching_decisions = [row for row in decisions if
                              row["record_id"] == entry.record_id and row["field_id"] == entry.field_id
                              and row["status"] == "applied"]
        fields.append({
            "record_id": entry.record_id,
            "field_id": entry.field_id,
            "raw_candidates": [candidate.model_dump(mode="json") for candidate in entry.candidates],
            "machine_recommendations": matching_attempts,
            "final_applied_decision": matching_decisions[-1] if matching_decisions else None,
        })
    return {"extraction_run_id": packet.run_id, "fields": fields}


def _render_extraction_resolution_review_md(snapshot: ReportingSourceSnapshot) -> str:
    report = _render_extraction_resolution_review_dict(snapshot)
    lines = ["# Extraction Resolution Review", ""]
    for field in report["fields"]:
        lines.extend([
            f"## `{field['record_id']}` / `{field['field_id']}`", "",
            "### Raw candidates", "",
        ])
        if field["raw_candidates"]:
            for candidate in field["raw_candidates"]:
                lines.append(f"- `{candidate['candidate_id']}`: {candidate['value']}")
        else:
            lines.append("_None._")
        lines.extend(["", "### Machine recommendation", ""])
        if field["machine_recommendations"]:
            for recommendation in field["machine_recommendations"]:
                lines.append(
                    f"- attempt {recommendation['attempt']} ({recommendation['kind']}): "
                    f"{recommendation.get('value') or recommendation.get('outcome') or 'no suggestion'}"
                )
        else:
            lines.append("_None._")
        lines.extend(["", "### Final applied decision", ""])
        decision = field["final_applied_decision"]
        lines.append(json.dumps(decision, ensure_ascii=False, sort_keys=True) if decision else "_None._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _draw_prisma_diagram(prisma: PRISMACounts, splits: Dict[str, Dict[str, int]]):
    """Returns a matplotlib `Figure` (never writes to a path itself) --
    §2.11's renderer contract requires every renderer to return `bytes`
    from a caller that owns the Figure's own lifecycle (`finally: plt.close(fig)`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def box(x, y, w, h, text, color="#dce8f0"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#2c5f8a", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#2c5f8a", lw=1.5))

    def split_text(split: Dict[str, int]) -> str:
        if not split:
            return ""
        return "\n" + "; ".join(f"{maker}: {n}" for maker, n in sorted(split.items()))

    total_id = prisma.total_identified
    # Finding 6 (Codex round-1 review, P2): sorted for the same reason as
    # `_render_methods_md()`'s own `db_list` -- determinism relative to
    # `prisma_sha256`, which is insertion-order-independent.
    sources = ", ".join(f"{db}: {n}" for db, n in sorted(prisma.identified.items()))
    box(1, 12, 8, 1.5, f"Records identified\n(n = {total_id})\n{sources}", color="#e8f4fb")
    arrow(5, 12, 5, 11.2)

    dupes = total_id - prisma.after_dedup
    box(1, 10, 8, 1.0, f"Records after deduplication\n(n = {prisma.after_dedup}; removed {dupes}"
                       f"{split_text(splits.get('dedup_removed', {}))})", color="#dce8f0")
    arrow(5, 10, 5, 9.2)

    box(1, 8, 8, 1.0, f"Records screened (title/abstract)\n(n = {prisma.screened_title_abstract})", color="#dce8f0")
    box(7.5, 7, 2, 0.8, f"Excluded\n(n = {prisma.excluded_title_abstract}"
                        f"{split_text(splits.get('title_abstract_excluded', {}))})", color="#fde8e8")
    ax.annotate("", xy=(7.5, 7.4), xytext=(7, 8.5), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    arrow(5, 8, 5, 7.2)

    box(1, 6, 8, 1.0, f"Full-text assessed\n(n = {prisma.assessed_full_text})", color="#dce8f0")
    ft_excluded = prisma.excluded_full_text
    if ft_excluded:
        box(7.5, 5, 2, 0.8, f"Excluded\n(n = {ft_excluded}"
                            f"{split_text(splits.get('full_text_excluded', {}))})", color="#fde8e8")
        ax.annotate("", xy=(7.5, 5.4), xytext=(7, 6.5), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    arrow(5, 6, 5, 5.2)

    y_included = 4
    if prisma.quality_assessed is not None:
        box(1, 4, 8, 1.0, f"Quality appraised\n(n = {prisma.quality_assessed}; "
                          f"excluded {prisma.excluded_quality})", color="#dce8f0")
        arrow(5, 4, 5, 3.2)
        y_included = 2

    box(1, y_included, 8, 1.0, f"Included in review\n(n = {prisma.included})", color="#d5e8d4")

    ax.set_title("PRISMA 2020 Flow Diagram (AI / rule-based / human)", fontsize=13, fontweight="bold", pad=10)
    return fig


def _render_prisma_diagram_png(snapshot: ReportingSourceSnapshot, splits: Dict[str, Dict[str, int]]) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = _draw_prisma_diagram(snapshot.prisma.model, splits)
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# §2.11 -- Publication
# ---------------------------------------------------------------------------

ARTIFACT_RENDERERS = [
    ("prisma_diagram", "prisma_flow.png", lambda s: True, _render_prisma_diagram_png, True),
    ("methods_md", "methods.md", lambda s: True, lambda s, sp: _render_methods_md(s).encode("utf-8"), False),
    ("ai_traceability_json", "ai_traceability.json", lambda s: True, lambda s, sp: canonical_json_bytes(_render_ai_traceability_dict(s)), False),
    ("ai_traceability_md", "ai_traceability.md", lambda s: True, lambda s, sp: _render_ai_traceability_md(s).encode("utf-8"), False),
    ("quality_appraisal_csv", "quality_appraisal.csv", lambda s: s.quality_appraisal is not None, lambda s, sp: _render_qa_csv(s), False),
    ("quality_appraisal_json", "quality_appraisal.json", lambda s: s.quality_appraisal is not None, lambda s, sp: canonical_json_bytes(_render_qa_json_dict(s)), False),
    ("evidence_map_csv", "evidence_map.csv", lambda s: s.evidence_map is not None, lambda s, sp: _render_evidence_map_csv(s), False),
    ("synthesis_claims_json", "synthesis_claims.json", lambda s: True, lambda s, sp: canonical_json_bytes(_render_synthesis_claims_dict(s)), False),
    ("synthesis_json", "synthesis.json", lambda s: True, lambda s, sp: canonical_json_bytes(s.synthesis_output.model.model_dump(mode="json")), False),
    ("synthesis_md", "synthesis.md", lambda s: True, lambda s, sp: _render_synthesis_md(s).encode("utf-8"), False),
    ("claim_audit_json", "claim_evidence_audit.json", lambda s: True, lambda s, sp: canonical_json_bytes(_render_claim_audit_dict(s)), False),
    ("claim_audit_md", "claim_evidence_audit.md", lambda s: True, lambda s, sp: _render_claim_audit_md(s).encode("utf-8"), False),
    ("extraction_resolution_review_json", "extraction_resolution_review.json",
     lambda s: s.extraction is not None and s.extraction.extraction_review is not None,
     lambda s, sp: canonical_json_bytes(_render_extraction_resolution_review_dict(s)), False),
    ("extraction_resolution_review_md", "extraction_resolution_review.md",
     lambda s: s.extraction is not None and s.extraction.extraction_review is not None,
     lambda s, sp: _render_extraction_resolution_review_md(s).encode("utf-8"), False),
    # Step 14 ([C13]) -- both gated on `reference_context is not None`: the
    # Track 2 reference run is the only run that ever sets
    # `cfg.reference_reporting_context` (every field the PRISMA-ScR
    # renderers read from `reference_context` is otherwise unavailable), so
    # a production run never produces these two artifacts.
    ("prisma_scr_report_md", "prisma_scr_report.md", lambda s: s.reference_context is not None,
     lambda s, sp: render_prisma_scr_report_md(s).encode("utf-8"), False),
    ("prisma_scr_conformance_json", "prisma_scr_conformance.json", lambda s: s.reference_context is not None,
     lambda s, sp: canonical_json_bytes(render_prisma_scr_conformance(s)), False),
]
FILENAMES = {name: filename for name, filename, *_ in ARTIFACT_RENDERERS}


def _qa_not_applicable_reason(snapshot: ReportingSourceSnapshot) -> str:
    """Finding 2 (Codex round-2 review, P1): a single static reason string
    conflated two genuinely different causes -- for `review_type ==
    "scoping"` QA is never applicable BY REVIEW TYPE (a configuration
    fact, true regardless of what happened this run); for rapid/SLR with
    `snapshot.quality_appraisal is None`, QA WAS configured to run but had
    no eligible full-text records to appraise (a run-specific fact) --
    reusing the scoping wording there would directly contradict that
    review's own configuration."""
    if snapshot.config_projection.review_type == "scoping":
        return "quality appraisal not performed for this review_type"
    return "quality appraisal was not performed: no eligible full-text records were available"


NOT_APPLICABLE_REASONS: Dict[str, Union[str, Callable[["ReportingSourceSnapshot"], str]]] = {
    "quality_appraisal_csv": _qa_not_applicable_reason,
    "quality_appraisal_json": _qa_not_applicable_reason,
    "evidence_map_csv": "generate_evidence_map is off, or full_text_records is empty",
    "extraction_resolution_review_json": "no extraction review packet was produced",
    "extraction_resolution_review_md": "no extraction review packet was produced",
    "prisma_scr_report_md": "reference_context is not set -- PRISMA-ScR reporting is a Track 2 reference-run-only artifact",
    "prisma_scr_conformance_json": "reference_context is not set -- PRISMA-ScR reporting is a Track 2 reference-run-only artifact",
}


def _component_reporting_entry(
    *, entry_kind: str, locator_kind: str, run_id: str, component: str, vc: VerifiedComponent,
) -> ReportingStageEntry:
    """One `extraction_component`/`qa_component` entry (plan Sec 10.3) for a
    single physical component file (`vc`) of an already-verified
    `SourceBundle` -- `document_id` is carried only for `component ==
    "checkpoint"` (`VerifiedComponent.document_id`, step 9)."""
    document_id = vc.document_id if component == "checkpoint" else None
    locator = LogicalLocator(locator_kind, component, run_id, document_id)
    return ReportingStageEntry(
        entry_kind=entry_kind, logical_path=str(locator), status="present",
        file_sha256=vc.file_sha256, internal_hash=vc.internal_hash,
        run_id=run_id, component=component, document_id=document_id,
    )


def _bundle_component_entries(*, entry_kind: str, locator_kind: str, bundle: SourceBundle) -> List[ReportingStageEntry]:
    entries = [_component_reporting_entry(entry_kind=entry_kind, locator_kind=locator_kind, run_id=bundle.run_id,
                                           component="manifest", vc=bundle.manifest)]
    entries.extend(_component_reporting_entry(entry_kind=entry_kind, locator_kind=locator_kind, run_id=bundle.run_id,
                                               component="checkpoint", vc=c) for c in bundle.checkpoints)
    if bundle.resolutions is not None:
        entries.append(_component_reporting_entry(entry_kind=entry_kind, locator_kind=locator_kind, run_id=bundle.run_id,
                                                    component="resolution", vc=bundle.resolutions))
    if bundle.confirmations is not None:  # extraction bundles never have one (SourceBundle.confirmations stays None)
        entries.append(_component_reporting_entry(entry_kind=entry_kind, locator_kind=locator_kind, run_id=bundle.run_id,
                                                    component="confirmation", vc=bundle.confirmations))
    if locator_kind == "extraction" and bundle.extraction_review is not None:
        review = bundle.extraction_review
        for component, subid, vc in (
            ("review_packet", "json", review.review_packet_json),
            ("review_packet", "md", review.review_packet_md),
            ("queue_snapshot", None, review.queue_snapshot),
            ("suggestions_index", None, review.suggestions_index),
            ("decision_journal", None, review.decision_journal),
        ):
            if vc is not None:
                entries.append(ReportingStageEntry(
                    entry_kind=entry_kind,
                    logical_path=str(LogicalLocator("extraction", component, bundle.run_id, subid)),
                    status="present", file_sha256=vc.file_sha256, internal_hash=vc.internal_hash,
                    run_id=bundle.run_id, component=component,
                ))
        for vc in review.suggestions:
            entries.append(ReportingStageEntry(
                entry_kind=entry_kind,
                logical_path=str(LogicalLocator("extraction", "suggestion", bundle.run_id, vc.document_id)),
                status="present", file_sha256=vc.file_sha256, internal_hash=vc.internal_hash,
                run_id=bundle.run_id, component="suggestion",
            ))
    return entries


def publish_report_bundle(*, report_dir: Path, snapshot: ReportingSourceSnapshot, splits: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    manifest_path = report_dir / "export_manifest.json"
    cp = snapshot.config_projection
    pool = _renderer_locator_pool(snapshot)

    entries: List[ReportingStageEntry] = []
    for name, filename, applicable_fn, render_fn, allow_failure in ARTIFACT_RENDERERS:
        target_path = report_dir / filename
        locator = str(LogicalLocator("reporting", "artifact", name))
        if not applicable_fn(snapshot):
            if target_path.exists():
                target_path.unlink()
            reason_entry = NOT_APPLICABLE_REASONS[name]
            reason = reason_entry(snapshot) if callable(reason_entry) else reason_entry
            entries.append(ReportingStageEntry(entry_kind="renderer_artifact", logical_path=locator,
                                                status="not_applicable", reason=reason, name=name))
            continue
        try:
            content_bytes = render_fn(snapshot, splits)
        except Exception as e:
            if not allow_failure:
                raise
            if target_path.exists():
                target_path.unlink()
            entries.append(ReportingStageEntry(entry_kind="renderer_artifact", logical_path=locator,
                                                status="failed", reason=str(e), name=name))
            continue
        file_sha256_val = _write_artifact_and_verify(target_path, content_bytes)
        entries.append(ReportingStageEntry(entry_kind="renderer_artifact", logical_path=locator, status="present",
                                            file_sha256=file_sha256_val, built_from=_renderer_built_from(name, pool),
                                            name=name, live_relpath=filename))

    for vf in snapshot.ledger_records:
        record = vf.model
        key_digest = _key_digest(record.record_id, record.stage)
        source_event_digest = sha256_hex(record.source_event_id)
        locator = str(LogicalLocator("selection_ledger", "decision", key_digest, source_event_digest))
        entries.append(ReportingStageEntry(entry_kind="selection_decision", logical_path=locator, status="present",
                                            file_sha256=vf.file_sha256, internal_hash=vf.internal_hash,
                                            record_id=record.record_id, stage=record.stage, decision_id=record.decision_id))

    for vf in snapshot.current_screening_batch_results:
        result = vf.model
        locator = str(LogicalLocator("screening_batch", "result", result.artifact_id))
        entries.append(ReportingStageEntry(entry_kind="screening_batch", logical_path=locator, status="present",
                                            file_sha256=vf.file_sha256, internal_hash=vf.internal_hash,
                                            artifact_id=result.artifact_id, batch_fingerprint=result.batch_fingerprint))

    if snapshot.extraction is not None:
        entries.extend(_bundle_component_entries(entry_kind="extraction_component", locator_kind="extraction", bundle=snapshot.extraction))
    if snapshot.quality_appraisal is not None:
        entries.extend(_bundle_component_entries(entry_kind="qa_component", locator_kind="quality_appraisal", bundle=snapshot.quality_appraisal))

    for vf in snapshot.events:
        event = vf.model
        locator = str(LogicalLocator("ai_use_events", "event", event.event_id))
        entries.append(ReportingStageEntry(entry_kind="ai_use_event", logical_path=locator, status="present",
                                            file_sha256=vf.file_sha256, internal_hash=vf.internal_hash,
                                            event_id=event.event_id, work_item_id=event.work_item_id, stage=event.stage))

    if snapshot.ft_native_provenance is not None:
        ftp = snapshot.ft_native_provenance
        ft_run_id = ftp["ft_screening_run_id"]
        key_to_component = {"run_result_json": "run_result", "item_results_jsonl": "item_results", "ledger_jsonl": "ledger"}
        for key, component in sorted(key_to_component.items()):
            locator = str(LogicalLocator("ft_screening", component, ft_run_id))
            entries.append(ReportingStageEntry(entry_kind="ft_run_artifact", logical_path=locator, status="present",
                                                file_sha256=ftp["artifact_file_hashes"][key],
                                                ft_screening_run_id=ft_run_id, component=component))

    entries.sort(key=lambda e: e.logical_path)

    manifest_obj = ReportingStageManifest(
        review_run_id=snapshot.review_run_id, review_type=cp.review_type,
        m1_protocol=M1ProtocolInfo(protocol_uri=cp.protocol_uri, protocol_hash=cp.protocol_hash,
                                    protocol_registration_id=cp.protocol_registration_id,
                                    ai_use_prespecified=cp.ai_use_prespecified, protocol_deviations=cp.protocol_deviations),
        m10_data_governance=cp.data_governance_declaration,
        entries=entries, manifest_hash="0" * 64,  # placeholder -- replaced below, before anything is written
    )
    manifest_dict = manifest_obj.model_dump(mode="json")
    manifest_dict["manifest_hash"] = _hash_value({k: v for k, v in manifest_dict.items() if k != "manifest_hash"})
    # Re-validate with the real hash so a manifest that would fail its own
    # schema (or self-hash convention) never reaches disk.
    manifest_dict = ReportingStageManifest.model_validate(manifest_dict).model_dump(mode="json")

    _write_artifact_and_verify(manifest_path, canonical_json_bytes(manifest_dict))

    reloaded = json.loads(manifest_path.read_bytes())
    recomputed = _hash_value({k: v for k, v in reloaded.items() if k != "manifest_hash"})
    if recomputed != reloaded["manifest_hash"]:
        raise ReportingConsistencyError("export_manifest.json failed its own self-hash check immediately after being written")
    return reloaded


def _renderer_entry(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Looks up the one `renderer_artifact` entry named `name` in an
    already-published `ReportingStageManifest` dict -- replaces the old
    `manifest["artifacts"][name]` dict-keyed lookup, now that `entries` is a
    flat, `logical_path`-sorted list (plan Sec 10.1(b))."""
    for entry in manifest["entries"]:
        if entry["entry_kind"] == "renderer_artifact" and entry["name"] == name:
            return entry
    raise ReportingConsistencyError(f"export_manifest.json has no renderer_artifact entry named {name!r}")


# ---------------------------------------------------------------------------
# §3 -- Workflow, manifest invalidated first
# ---------------------------------------------------------------------------

def reporting_node(
    state: ReviewState, cumulative_snapshot: Optional[ReportingSourceSnapshot] = None,
) -> Dict[str, Any]:
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = resolve_review_run_id(state, allow_config=True)
    reporting_root = (cfg.ensure_output_dir() / "reporting").resolve()
    report_dir = (reporting_root / review_run_id).resolve()
    if not report_dir.is_relative_to(reporting_root):
        raise ReportingConsistencyError(f"resolved report_dir {report_dir} escapes the reporting root")
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "export_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    _validate_ledger_dir(state, cfg, review_run_id)
    snapshot = cumulative_snapshot if cumulative_snapshot is not None else capture_reporting_snapshot(state, cfg, review_run_id)
    splits = compute_selection_splits(snapshot=snapshot, prisma=snapshot.prisma.model)
    manifest = publish_report_bundle(report_dir=report_dir, snapshot=snapshot, splits=splits)

    def _renderer_path(name: str) -> Optional[str]:
        entry = _renderer_entry(manifest, name)
        if entry["status"] != "present":
            return None
        path = report_dir / FILENAMES[name]
        if not path.exists():
            raise ReportingConsistencyError(f"manifest claims {name!r} is present but {path} does not exist on disk")
        return str(path)

    prisma_diagram_path = _renderer_path("prisma_diagram")
    methods_md_path = _renderer_path("methods_md")
    prisma_scr_report_md_path = _renderer_path("prisma_scr_report_md")
    prisma_scr_conformance_path = _renderer_path("prisma_scr_conformance_json")

    return {
        "prisma_diagram_path": prisma_diagram_path,
        "methods_text": Path(methods_md_path).read_text(encoding="utf-8"),
        "ai_traceability_json_path": _renderer_path("ai_traceability_json"),
        "ai_traceability_md_path": _renderer_path("ai_traceability_md"),
        "quality_appraisal_csv_path": _renderer_path("quality_appraisal_csv"),
        "quality_appraisal_json_path": _renderer_path("quality_appraisal_json"),
        "evidence_map_csv_path": _renderer_path("evidence_map_csv"),
        "synthesis_claims_json_path": _renderer_path("synthesis_claims_json"),
        "synthesis_json_path": _renderer_path("synthesis_json"),
        "synthesis_md_path": _renderer_path("synthesis_md"),
        "claim_audit_json_path": _renderer_path("claim_audit_json"),
        "claim_audit_md_path": _renderer_path("claim_audit_md"),
        "export_manifest_path": str(manifest_path),
        "prisma_scr_report_md": (
            Path(prisma_scr_report_md_path).read_text(encoding="utf-8") if prisma_scr_report_md_path else None
        ),
        "prisma_scr_conformance": (
            json.loads(Path(prisma_scr_conformance_path).read_bytes()) if prisma_scr_conformance_path else None
        ),
        "prisma_scr_report_sha256": (
            _renderer_entry(manifest, "prisma_scr_report_md")["file_sha256"] if prisma_scr_report_md_path else None
        ),
        "prisma_scr_conformance_sha256": (
            _renderer_entry(manifest, "prisma_scr_conformance_json")["file_sha256"] if prisma_scr_conformance_path else None
        ),
    }
