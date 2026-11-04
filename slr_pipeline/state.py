"""
ReviewState and core data models for the SLR automation pipeline.

Every LangGraph node receives the full ReviewState and returns a dict
with only the keys it updates. All data models use Pydantic for
validation and JSON serialisation (required by LangGraph checkpointers).
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Any, Dict, FrozenSet, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator
import uuid

from slr_pipeline.config import ReviewConfig


# ---------------------------------------------------------------------------
# Core bibliographic record
# ---------------------------------------------------------------------------

class Record(BaseModel):
    """Standardised bibliographic record, source-agnostic."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str                        # 'openalex', 'pubmed', 'arxiv', ...
    source_id: str                     # database-native identifier
    doi: Optional[str] = None
    title: str = ""
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    cited_by: Optional[int] = None
    url: Optional[str] = None

    # Set by dedup_node
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None   # id of canonical record

    # Set by screening_node
    screening_decision: Optional[Literal["include", "borderline", "exclude"]] = None
    screening_reason: Optional[str] = None
    screening_confidence: Optional[float] = None
    screening_relevant_topics: List[str] = Field(default_factory=list)
    # Set by screening_qc_node, on the sampled subset only
    second_rater_decision: Optional[Literal["include", "borderline", "exclude"]] = None

    # Set by fulltext_node
    full_text_path: Optional[str] = None
    full_text_status: Optional[
        Literal["parsed", "rejected_stub", "no_oa_fulltext_found", "no_doi"]
    ] = None
    full_text_source: Optional[str] = None   # which resolver found it: 'unpaywall', 'semantic_scholar', ...

    # quality_appraisal_node NEVER mutates Record (see
    # nodes.quality_appraisal's own module docstring, Section 1.2 of
    # quality_appraisal_synthesis_plan_2026-08-22.md) -- there is no
    # per-record quality_score/quality_details field any more; quality
    # appraisal results live in ReviewState's own
    # quality_appraisal_results/-review_queue/-resolutions, keyed by
    # record_id, never embedded here.

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Record":
        return cls(**d)


# ---------------------------------------------------------------------------
# Charted / extracted data
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """
    A single piece of extracted evidence tied to a research question.
    Carries page location and confidence — adapted from
    identity_leadership_screening_20260121.ipynb (create_evidence_based_prompt).

    NOTE: page_range is only populated when Docling is used for parsing;
    pymupdf4llm fallback does not currently preserve page numbers.
    """
    question_key: str                      # e.g. "rq_0", "rq_1"
    answer: Optional[str] = None           # direct answer to the RQ
    supporting_excerpts: List[str] = Field(default_factory=list)  # verbatim quotes
    confidence: Literal["high", "medium", "low", "not_found"] = "not_found"
    page_range: Optional[str] = None       # "12-14" or "7" — from Docling metadata
    section: Optional[str] = None          # section heading this evidence came from


class SectionClassification(BaseModel):
    """
    Per-section classification against research questions.
    Adapted from classify_sections_with_llm in NB1.

    NOTE: The classification prompt is our own design; NB1 only showed the
    calling code, not the prompt itself. May need tuning per review topic.
    """
    section_name: str
    category: Literal["background", "methods", "results", "discussion", "other"]
    relevant_rq_indices: List[int] = Field(default_factory=list)  # 0-based RQ indices
    confidence: Literal["high", "medium", "low"] = "low"
    reasoning: Optional[str] = None


class ChartedRecord(BaseModel):
    """
    `charted_data`'s per-record shape. `charting_method="pydantic"`/
    `"chunking"` are the legacy charting_node outputs (nodes/charting.py,
    moved to examples/legacy/charting.py -- no longer used by any
    production graph); `charting_method="extraction_projection"` is the
    current one, produced for ALL THREE review types (scoping included,
    since Stage 2 of quality_appraisal_synthesis_plan_2026-08-22.md) by
    nodes.extraction._project_charted_data() from the frozen Phase A
    extraction engine's own results.

    charting_method="pydantic"  → raw_extraction only (fast, no page tracking)
    charting_method="chunking"  → raw_extraction + evidence_items with page refs
    """
    record_id: str
    raw_extraction: Dict[str, Any] = Field(default_factory=dict)
    # Page-aware evidence extraction (chunking method only)
    evidence_items: List[Dict[str, Any]] = Field(default_factory=list)
    # Section-to-RQ classifications (chunking method only)
    section_classifications: List[Dict[str, Any]] = Field(default_factory=list)
    validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    # "extraction_projection" -- built by nodes.extraction._project_charted_data()
    # from the Phase A structured-extraction engine's own DocumentExtractionResult
    # (see nodes/extraction.py's module docstring). This value is a DETERMINISTIC,
    # LOSSY downstream compatibility view (one raw value per field; evidence,
    # confidence, rationale, and every non-primary value discarded) -- never the
    # scientific artifact itself, which remains state["extraction_results"] plus
    # the frozen Phase A checkpoint/manifest files on disk.
    charting_method: Literal["pydantic", "chunking", "extraction_projection"] = "pydantic"


# ---------------------------------------------------------------------------
# PRISMA flow counts
# ---------------------------------------------------------------------------

class PRISMACounts(BaseModel):
    """Tracks record counts at every PRISMA stage."""

    identified: Dict[str, int] = Field(default_factory=dict)  # {source: count}
    after_dedup: int = 0
    screened_title_abstract: int = 0
    excluded_title_abstract: int = 0
    assessed_full_text: int = 0          # documents fulltext_node actually ran FT_SCREENING against
                                          # from a confirmed PDF -- NOT every title/abstract inclusion
    excluded_full_text: int = 0
    excluded_full_text_reasons: Dict[str, int] = Field(default_factory=dict)
    # {ft_screening.CriterionSpec.exclusion_reason_code: count} -- a structured,
    # stable code (e.g. "wrong_population"), never a free-text LLM sentence.
    fulltext_retrieval: Dict[str, int] = Field(default_factory=dict)  # {'parsed': n, 'rejected_stub': n, ...}
    quality_assessed: Optional[int] = None   # None means step was skipped
    excluded_quality: Optional[int] = None
    included: int = 0

    @property
    def total_identified(self) -> int:
        return sum(self.identified.values())


# ---------------------------------------------------------------------------
# Search log
# ---------------------------------------------------------------------------

_V1_REQUIRED_FIELDS = {"database", "query", "date", "retrieved"}
_V1_ALLOWED_FIELDS = _V1_REQUIRED_FIELDS | {"hits", "schema_version"}


class TerminationReason(str, Enum):
    """Closed set of `termination_reason` values (review finding P2,
    2026-09-24): free text let contradictory pairs like
    `retrieval_status="exhaustive"` + `termination_reason="max_results_cap_reached"`
    validate. Every member is compatible with exactly the
    `retrieval_status` value(s) listed in `TERMINATION_REASONS_BY_STATUS`
    below -- adding a new reason means adding it there too."""
    ZERO_HITS = "zero_hits"
    TOTAL_HITS_EQUALS_RETRIEVED = "total_hits_equals_retrieved"
    CURSOR_EXHAUSTED_EMPTY_NEXT_PAGE = "cursor_exhausted_empty_next_page"
    MAX_RESULTS_CAP_REACHED = "max_results_cap_reached"
    PAGINATION_END_UNVERIFIED = "pagination_end_unverified"
    NOT_YET_INSTRUMENTED = "not_yet_instrumented"
    MIGRATED_FROM_V2_UNSPECIFIED = "migrated_from_v2_unspecified"
    UNEXPLAINED_RETRIEVAL_GAP = "unexplained_retrieval_gap"
    RETRIEVAL_ERROR = "retrieval_error"
    PARTIAL_RETRIEVAL_WITH_SKIPS_OR_ERRORS = "partial_retrieval_with_skips_or_errors"
    # PubMed EFetch id-reconciliation/history-pagination causes (review
    # findings P1, 2026-09-24): distinct, explicit causes so a silently
    # dropped/substituted PMID or a misaligned history cursor is never
    # reported under the generic "malformed source record(s)" text.
    EFETCH_MISSING_REQUESTED_PMIDS = "efetch_missing_requested_pmids"
    EFETCH_UNEXPECTED_PMIDS = "efetch_unexpected_pmids"
    EFETCH_MISSING_AND_UNEXPECTED_PMIDS = "efetch_missing_and_unexpected_pmids"
    HISTORY_PAGE_SIZE_MISMATCH = "history_page_size_mismatch"


TERMINATION_REASONS_BY_STATUS: Dict[str, FrozenSet[TerminationReason]] = {
    "exhaustive": frozenset({
        TerminationReason.ZERO_HITS,
        TerminationReason.TOTAL_HITS_EQUALS_RETRIEVED,
        TerminationReason.CURSOR_EXHAUSTED_EMPTY_NEXT_PAGE,
    }),
    "capped": frozenset({
        TerminationReason.MAX_RESULTS_CAP_REACHED,
        TerminationReason.PAGINATION_END_UNVERIFIED,
        TerminationReason.NOT_YET_INSTRUMENTED,
        TerminationReason.MIGRATED_FROM_V2_UNSPECIFIED,
    }),
    "partial": frozenset({
        TerminationReason.UNEXPLAINED_RETRIEVAL_GAP,
        TerminationReason.PAGINATION_END_UNVERIFIED,
        TerminationReason.PARTIAL_RETRIEVAL_WITH_SKIPS_OR_ERRORS,
        TerminationReason.MIGRATED_FROM_V2_UNSPECIFIED,
        TerminationReason.EFETCH_MISSING_REQUESTED_PMIDS,
        TerminationReason.EFETCH_UNEXPECTED_PMIDS,
        TerminationReason.EFETCH_MISSING_AND_UNEXPECTED_PMIDS,
        TerminationReason.HISTORY_PAGE_SIZE_MISMATCH,
    }),
    "failed": frozenset({
        TerminationReason.UNEXPLAINED_RETRIEVAL_GAP,
        TerminationReason.RETRIEVAL_ERROR,
        TerminationReason.MIGRATED_FROM_V2_UNSPECIFIED,
        TerminationReason.EFETCH_MISSING_REQUESTED_PMIDS,
        TerminationReason.EFETCH_UNEXPECTED_PMIDS,
        TerminationReason.EFETCH_MISSING_AND_UNEXPECTED_PMIDS,
        TerminationReason.HISTORY_PAGE_SIZE_MISMATCH,
    }),
}


class DatabaseSearchLog(BaseModel):
    """v3 (Track 2 plan §2.8, Revision 5 point 1) -- the model never holds a
    hybrid v1/v2/v3 shape: `_migrate_legacy` fully upgrades a legacy
    `{database, query, date, hits, retrieved}` v1 doc, or an unversioned/
    v2 doc predating the exhaustive/capped/partial/failed contract, into a
    complete v3 object before any field validation runs. A migrated v2
    `"complete"` status is downgraded to `"capped"`, never upgraded to
    `"exhaustive"` -- v2 never recorded a real exhaustion proof, so
    claiming exhaustion for it would be dishonest. The already-persisted
    `berg2023-update-20260923` artifact is never rewritten on disk; this
    migration only applies in memory to any code that loads it."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["database_search_log_v3"] = "database_search_log_v3"
    database: str                         # CANONICAL name
    execution_adapter: str
    adapter_config_version: str
    query: str
    date: str                             # "YYYY-MM-DD" (== spec.as_of_date)
    retrieval_status: Literal["exhaustive", "capped", "partial", "failed"]
    termination_reason: TerminationReason
    failure_detail: Optional[str] = None
    unsupported_filter_deviations: List[str] = Field(default_factory=list)
    skipped_malformed: int = Field(default=0, ge=0)
    requested: int = Field(ge=0)
    hits: int = Field(ge=0)               # source total, OR == retrieved when the adapter's total_hits is None
    retrieved: int = Field(ge=0)
    # Mirrors SearchBackendResult.access_mode (search_backends/base.py) --
    # None for every adapter with no COMPLETE/STANDARD-style access-tier
    # distinction; Scopus is the only current source of a non-None value.
    access_mode: Optional[str] = None
    # Mirrors SearchBackendResult.rate_limit_remaining/rate_limit_reset_seconds
    # (search_backends/base.py) -- None for every adapter/outcome except an
    # OpenAlex 429 failure, which records the daily-budget metadata from its
    # response headers.
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset_seconds: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data):
        # A legacy v1 doc: no schema_version (or "v1") and only the old key set.
        if isinstance(data, dict) and data.get("schema_version") in (None, "v1") \
           and set(data) <= _V1_ALLOWED_FIELDS:
            # `set(data) <= _V1_ALLOWED_FIELDS` only guards against extra/unknown
            # keys -- it says nothing about whether the required v1 fields are
            # actually PRESENT (a subset check passes trivially for `{}` too).
            # A hand-edited or truncated legacy checkpoint missing e.g.
            # "retrieved" must fail with a clean ValidationError here, not an
            # unhandled KeyError three lines down.
            missing = _V1_REQUIRED_FIELDS - set(data)
            if missing:
                raise ValueError(
                    f"legacy v1 search_log entry missing required field(s): {sorted(missing)}"
                )
            r = int(data["retrieved"])
            data = {"schema_version": "database_search_log_v2", "database": data["database"],
                    "execution_adapter": data["database"], "adapter_config_version": "unknown",
                    "query": data["query"], "date": data["date"], "retrieval_status": "complete",
                    "failure_detail": None, "unsupported_filter_deviations": [], "skipped_malformed": 0,
                    "requested": r, "hits": int(data.get("hits", r)), "retrieved": r}
        if isinstance(data, dict) and data.get("schema_version") == "database_search_log_v2":
            # v2 never recorded a real exhaustion proof -- its "complete" is
            # downgraded to "capped", never upgraded to "exhaustive".
            data = dict(data)
            data["schema_version"] = "database_search_log_v3"
            if data.get("retrieval_status") == "complete":
                data["retrieval_status"] = "capped"
            data.setdefault("termination_reason", TerminationReason.MIGRATED_FROM_V2_UNSPECIFIED.value)
            # Same unexplained-gap invariant `finalize_result()` applies at
            # creation time (review finding P1, 2026-09-24): a migrated
            # "capped" entry whose own persisted counts show hits > retrieved
            # and retrieved < requested (e.g. the preserved PubMed 300/303
            # entry) is not a clean capped result -- v2 had no way to record
            # that gap, and blindly downgrading "complete" to "capped" would
            # hide it, letting reference_acceptance.py describe it as
            # error-free. Reclassify as "partial" instead.
            requested, hits, retrieved = data.get("requested"), data.get("hits"), data.get("retrieved")
            if (
                data["retrieval_status"] == "capped"
                and isinstance(requested, (int, float))
                and isinstance(hits, (int, float))
                and isinstance(retrieved, (int, float))
                and hits > retrieved and retrieved < requested
            ):
                data["retrieval_status"] = "partial"
                data["termination_reason"] = TerminationReason.UNEXPLAINED_RETRIEVAL_GAP.value
                data["failure_detail"] = (
                    f"unexplained retrieval gap discovered during v2->v3 migration: "
                    f"total_hits={hits} exceeds retrieved={retrieved}, and retrieved did not reach "
                    f"the recorded requested cap ({requested})"
                )
        return data

    @model_validator(mode="after")
    def _conditional(self):
        if self.skipped_malformed > 0 and self.retrieval_status in ("exhaustive", "capped"):
            raise ValueError("skipped_malformed > 0 forces retrieval_status in {partial, failed}")
        if self.skipped_malformed > 0 and self.retrieved == 0 and self.retrieval_status != "failed":
            raise ValueError("skipped >= 1 and 0 usable records => 'failed'")
        if self.retrieval_status == "failed" and self.retrieved != 0:
            raise ValueError("a failed database search retrieved 0")
        if self.retrieval_status == "partial" and self.retrieved < 1:
            raise ValueError("partial requires >= 1 retrieved record (else 'failed')")
        if self.retrieval_status in ("exhaustive", "capped") and self.failure_detail is not None:
            raise ValueError("exhaustive/capped search has no failure_detail")
        if self.retrieval_status in ("partial", "failed") and not self.failure_detail:
            raise ValueError("partial/failed search MUST carry a non-empty failure_detail")
        if self.retrieved > self.requested:
            raise ValueError("retrieved cannot exceed requested")
        if self.hits < self.retrieved:
            raise ValueError("hits cannot be < retrieved")
        if self.retrieval_status == "capped" and self.hits > self.retrieved and self.retrieved < self.requested:
            raise ValueError(
                "'capped' with hits > retrieved and retrieved < requested is an unexplained "
                "retrieval gap, not a clean cap -- use retrieval_status='partial' with "
                "termination_reason='unexplained_retrieval_gap' instead"
            )
        allowed_reasons = TERMINATION_REASONS_BY_STATUS[self.retrieval_status]
        if self.termination_reason not in allowed_reasons:
            raise ValueError(
                f"termination_reason {self.termination_reason.value!r} is not compatible with "
                f"retrieval_status {self.retrieval_status!r} -- allowed: "
                f"{sorted(r.value for r in allowed_reasons)}"
            )
        if self.retrieval_status == "exhaustive" and self.hits != self.retrieved:
            raise ValueError("'exhaustive' requires hits == retrieved")
        return self


class SearchLog(BaseModel):
    entries: List[DatabaseSearchLog] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol (PCC for scoping, PICO for SLR)
# ---------------------------------------------------------------------------

class Protocol(BaseModel):
    title: str
    review_type: Literal["scoping", "rapid", "slr"]
    research_questions: List[str]
    # PCC (scoping/rapid)
    population: Optional[str] = None
    concept: Optional[str] = None
    context: Optional[str] = None
    # PICO (SLR)
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    outcome: Optional[str] = None
    # Criteria
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    date_range: Optional[str] = None
    language_filter: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Search strategy
# ---------------------------------------------------------------------------

class SearchStrategy(BaseModel):
    """Output of search_strategy_builder_node."""
    seed_query: str = ""
    expanded_query: str = ""          # generic / PubMed query used by default
    db_queries: Dict[str, str] = Field(default_factory=dict)  # db-specific variants
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    backend: Optional[str] = None
    llm_model: Optional[str] = None
    # Codex round-5 review, finding 3 (aggregation/export/reporting plan
    # v17): the exact AIUseEvent.event_id(s) THIS invocation published
    # (query expansion, when attempted, and db translation, always) --
    # lets reporting's own cross-check verify against the CURRENT
    # invocation's own events specifically, rather than every historical
    # event ever published for these stage names under this review_run_id
    # (an append-only log an earlier, differently-configured invocation
    # may also have written to).
    producing_event_ids: List[str] = Field(default_factory=list)

    @field_validator("producing_event_ids")
    @classmethod
    def _unique_producing_event_ids(cls, v: List[str]) -> List[str]:
        # Codex round-6 review, finding 4, P2: no uniqueness/cardinality
        # validation let [same_translation_id, same_translation_id] pass
        # as if it were two distinct events. Reporting's own exact-count
        # checks (aggregation/export/reporting plan v17,
        # _cross_check_oversight_and_automation_origin) rely on this list
        # containing no duplicate event ids.
        if len(set(v)) != len(v):
            raise ValueError("producing_event_ids must not contain duplicates")
        return v


# ---------------------------------------------------------------------------
# Synthesis -- structured, evidence-verified claims and traceability, the
# Stage 3 claim-construction slice of quality_appraisal_synthesis_plan_2026-
# 08-22.md's own §3.2 ("Structural traceability, not claim validation").
# Self-contained: zero internal imports beyond stdlib+pydantic.
# ---------------------------------------------------------------------------

_EVIDENCE_ID_RE = re.compile(r"^evidence-[0-9a-f]{64}$")
_CLAIM_ID_RE = re.compile(r"^claim-[0-9a-f]{64}$")
_ANSWER_ID_RE = re.compile(r"^answer-[0-9a-f]{64}$")
_AIUSE_EVENT_ID_RE = re.compile(r"^aiuse-[0-9a-f]{48}$")


class SupportingFieldSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_kind: Literal["extraction", "quality_appraisal"]
    field_id: str = Field(min_length=1)

    @field_validator("field_id")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field_id must not be blank/whitespace-only")
        return v


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_kind: Literal["extraction", "quality_appraisal"]
    artifact_run_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    field_id: str = Field(min_length=1)
    evidence_id: str

    @field_validator("artifact_run_id", "record_id", "field_id")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("evidence_id")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if not _EVIDENCE_ID_RE.fullmatch(v):
            raise ValueError(f"evidence_id {v!r} is not well-formed")
        return v


def _reference_sort_key(r: EvidenceReference):
    return (r.artifact_kind, r.artifact_run_id, r.record_id, r.field_id, r.evidence_id)


def compute_answer_id(*, rq_or_section: str, record_id: str, answer_text: str,
                       selectors: List[SupportingFieldSelector],
                       evidence_references: List[EvidenceReference]) -> str:
    """answer_id is a pure function of the answer's own CONTENT (target, record, text, cited
    selectors, cited evidence) -- reproducible independent of which physical map call produced it.
    producing_event_id is carried on MapAnswer as a required, separately format-checked, separately
    verified PROVENANCE field -- it is never a hash input."""
    payload = {"rq_or_section": rq_or_section, "record_id": record_id, "answer_text": answer_text,
               "selectors": sorted([(s.artifact_kind, s.field_id) for s in selectors]),
               "evidence_references": sorted([(r.artifact_kind, r.artifact_run_id, r.record_id, r.field_id, r.evidence_id)
                                               for r in evidence_references])}
    return "answer-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class MapAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    answer_id: str
    rq_or_section: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    selectors: List[SupportingFieldSelector] = Field(min_length=1)
    evidence_references: List[EvidenceReference] = Field(min_length=1)
    producing_event_id: str = Field(min_length=1)   # provenance ONLY -- the physical
                                                        # synthesis_claim_map event that produced
                                                        # this exact content; verified separately by
                                                        # verify_synthesis_output_sources(),
                                                        # deliberately NOT an answer_id hash input

    @field_validator("rq_or_section", "record_id", "answer_text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("selectors")
    @classmethod
    def _unique_sorted_selectors(cls, v: List[SupportingFieldSelector]) -> List[SupportingFieldSelector]:
        keys = [(s.artifact_kind, s.field_id) for s in v]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate selector in selectors")
        return sorted(v, key=lambda s: (s.artifact_kind, s.field_id))

    @field_validator("evidence_references")
    @classmethod
    def _canonicalize_references(cls, v: List[EvidenceReference]) -> List[EvidenceReference]:
        deduped = list({_reference_sort_key(r): r for r in v}.values())
        return sorted(deduped, key=_reference_sort_key)

    @field_validator("answer_id")
    @classmethod
    def _check_answer_id_format(cls, v: str) -> str:
        if not _ANSWER_ID_RE.fullmatch(v):
            raise ValueError(f"answer_id {v!r} is not well-formed")
        return v

    @field_validator("producing_event_id")
    @classmethod
    def _check_event_id_format(cls, v: str) -> str:
        if not _AIUSE_EVENT_ID_RE.fullmatch(v):
            raise ValueError(f"producing_event_id {v!r} is not a well-formed AIUseEvent id")
        return v

    @model_validator(mode="after")
    def _verify(self) -> "MapAnswer":
        foreign = [r for r in self.evidence_references if r.record_id != self.record_id]
        if foreign:
            raise ValueError("evidence_references contains reference(s) for a different record_id")
        selector_keys = {(s.artifact_kind, s.field_id) for s in self.selectors}
        reference_keys = {(r.artifact_kind, r.field_id) for r in self.evidence_references}
        if selector_keys != reference_keys:
            raise ValueError(f"selectors {selector_keys} do not exactly match the reference field-set {reference_keys}")
        expected = compute_answer_id(rq_or_section=self.rq_or_section, record_id=self.record_id,
                                      answer_text=self.answer_text, selectors=self.selectors,
                                      evidence_references=self.evidence_references)
        if self.answer_id != expected:
            raise ValueError(f"answer_id {self.answer_id!r} does not match recomputed {expected!r}")
        return self


def build_map_answer(*, rq_or_section: str, record_id: str, answer_text: str,
                      selectors: List[SupportingFieldSelector], evidence_references: List[EvidenceReference],
                      producing_event_id: str) -> MapAnswer:
    canonical_selectors = sorted(selectors, key=lambda s: (s.artifact_kind, s.field_id))
    canonical_refs = sorted({_reference_sort_key(r): r for r in evidence_references}.values(), key=_reference_sort_key)
    answer_id = compute_answer_id(rq_or_section=rq_or_section, record_id=record_id, answer_text=answer_text,
                                   selectors=canonical_selectors, evidence_references=canonical_refs)
    return MapAnswer(answer_id=answer_id, rq_or_section=rq_or_section, record_id=record_id,
                      answer_text=answer_text, selectors=selectors, evidence_references=evidence_references,
                      producing_event_id=producing_event_id)


def _compute_claim_id(claim_text: str, references: List[EvidenceReference], rq_or_section: str) -> str:
    """EXACTLY the parent design doc's own literal three-field hash
    (quality_appraisal_synthesis_plan_2026-08-22.md:380) -- no additions. supporting_answer_ids and
    producing_event_id are both real fields on SynthesisClaim but neither is a hash input."""
    payload = {"claim_text": claim_text,
               "references": [{"artifact_kind": r.artifact_kind, "artifact_run_id": r.artifact_run_id,
                                "record_id": r.record_id, "field_id": r.field_id, "evidence_id": r.evidence_id}
                              for r in sorted(references, key=_reference_sort_key)],
               "rq_or_section": rq_or_section}
    return "claim-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SynthesisClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str
    claim_text: str = Field(min_length=1)
    references: List[EvidenceReference] = Field(min_length=1)
    supporting_answer_ids: List[str] = Field(min_length=1)
    rq_or_section: str = Field(min_length=1)
    producing_event_id: str = Field(min_length=1)   # provenance ONLY -- the LAST synthesis_claim_refine
                                                        # event whose own response set this claim's CURRENT
                                                        # (claim_text, supporting_answer_ids); verified
                                                        # separately by verify_synthesis_output_sources(),
                                                        # deliberately NOT a claim_id hash input

    @field_validator("claim_text", "rq_or_section")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("references")
    @classmethod
    def _canonicalize_references(cls, v: List[EvidenceReference]) -> List[EvidenceReference]:
        deduped = list({_reference_sort_key(r): r for r in v}.values())
        return sorted(deduped, key=_reference_sort_key)

    @field_validator("supporting_answer_ids")
    @classmethod
    def _canonicalize_answer_ids(cls, v: List[str]) -> List[str]:
        for a in v:
            if not _ANSWER_ID_RE.fullmatch(a):
                raise ValueError(f"malformed answer_id {a!r} in supporting_answer_ids")
        return sorted(set(v))

    @field_validator("claim_id")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if not _CLAIM_ID_RE.fullmatch(v):
            raise ValueError(f"claim_id {v!r} is not well-formed")
        return v

    @field_validator("producing_event_id")
    @classmethod
    def _check_event_id_format(cls, v: str) -> str:
        if not _AIUSE_EVENT_ID_RE.fullmatch(v):
            raise ValueError(f"producing_event_id {v!r} is not a well-formed AIUseEvent id")
        return v

    @model_validator(mode="after")
    def _verify_claim_id(self) -> "SynthesisClaim":
        expected = _compute_claim_id(self.claim_text, self.references, self.rq_or_section)
        if self.claim_id != expected:
            raise ValueError(f"claim_id {self.claim_id!r} does not match recomputed {expected!r}")
        return self


def build_synthesis_claim(claim_text: str, contributing_answers: List[MapAnswer], rq_or_section: str,
                           producing_event_id: str) -> SynthesisClaim:
    if not contributing_answers:
        raise ValueError("a SynthesisClaim requires at least one contributing MapAnswer")
    seen: Dict[tuple, EvidenceReference] = {}
    for a in contributing_answers:
        for r in a.evidence_references:
            seen.setdefault(_reference_sort_key(r), r)
    references = sorted(seen.values(), key=_reference_sort_key)
    supporting_answer_ids = sorted({a.answer_id for a in contributing_answers})
    claim_id = _compute_claim_id(claim_text, references, rq_or_section)
    return SynthesisClaim(claim_id=claim_id, claim_text=claim_text, references=references,
                           supporting_answer_ids=supporting_answer_ids, rq_or_section=rq_or_section,
                           producing_event_id=producing_event_id)


class MapOutcomeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rq_or_section: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    outcome: Literal["relevant", "not_relevant", "call_failed", "invalid_response", "no_citeable_evidence"]
    hallucinated_selector: StrictBool = False
    producing_event_id: Optional[str] = None

    @field_validator("rq_or_section", "record_id")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("producing_event_id")
    @classmethod
    def _check_event_id_format_if_present(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _AIUSE_EVENT_ID_RE.fullmatch(v):
            raise ValueError(f"producing_event_id {v!r} is not a well-formed AIUseEvent id")
        return v

    @model_validator(mode="after")
    def _verify(self) -> "MapOutcomeRecord":
        if self.outcome == "no_citeable_evidence":
            if self.producing_event_id is not None:
                raise ValueError("no_citeable_evidence means no physical call was made -- "
                                  "producing_event_id must be None")
        elif self.producing_event_id is None:
            raise ValueError(f"outcome={self.outcome!r} means a physical call WAS made -- "
                              "producing_event_id is required")
        if self.hallucinated_selector and self.outcome != "invalid_response":
            raise ValueError("hallucinated_selector may only be True when outcome == 'invalid_response'")
        return self


class RefineClaimSnapshot(BaseModel):
    """One accumulator's own (claim_text, supporting_answer_ids, producing_event_id) state at a
    single point in a target's own refine batch sequence -- POSITIONAL: this entry's own index
    within the containing list is the SAME claim_index a later refine batch's own applied_updates
    addresses by position, and the same index the live orchestration's own in-memory `accumulators`
    list uses. Always stored with supporting_answer_ids SORTED, matching the live
    `_ClaimAccumulator`'s own invariant (never re-sorted on read). producing_event_id (codex
    code-review round 3, 2026-08-26 finding 2) is the SAME field `_ClaimAccumulator.producing_event_id`
    tracks at runtime -- the LAST refine event that set THIS accumulator position's current content
    -- carried through replay so a final claim can be bound to the event that ACTUALLY produced its
    own current position, not merely to any applied event for the target."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_text: str = Field(min_length=1)
    supporting_answer_ids: List[str] = Field(min_length=1)
    producing_event_id: str = Field(min_length=1)

    @field_validator("claim_text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim_text must not be blank/whitespace-only")
        return v

    @field_validator("supporting_answer_ids")
    @classmethod
    def _well_formed_sorted_unique(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate answer_id in supporting_answer_ids")
        for a in v:
            if not _ANSWER_ID_RE.fullmatch(a):
                raise ValueError(f"malformed answer_id {a!r} in supporting_answer_ids")
        if v != sorted(v):
            raise ValueError("supporting_answer_ids must be stored sorted, matching the live "
                              "_ClaimAccumulator's own invariant")
        return v

    @field_validator("producing_event_id")
    @classmethod
    def _check_event_id_format(cls, v: str) -> str:
        if not _AIUSE_EVENT_ID_RE.fullmatch(v):
            raise ValueError(f"producing_event_id {v!r} is not a well-formed AIUseEvent id")
        return v


class RefineAppliedUpdate(BaseModel):
    """One entry of the model's own real RefineResponse.claim_updates, in ORIGINAL response order,
    persisted ONLY for outcome=='applied' batches -- lets a verifier recompute the EXACT expected
    output_artifact_refs (response:/update: hashes, via the SAME _canonical_refine_response_payload/
    _canonical_refine_update_payload the live orchestration itself hashes) and the EXACT resulting
    post-batch accumulator state, rather than trusting either the event's own claimed hashes or the
    next batch's own pre_batch_claim_states as self-asserted."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_index: Optional[StrictInt] = None
    claim_text: str = Field(min_length=1)
    final_support_answer_ids: List[str] = Field(min_length=1)

    @field_validator("claim_text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim_text must not be blank/whitespace-only")
        return v

    @field_validator("final_support_answer_ids")
    @classmethod
    def _well_formed_unique(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate answer_id in final_support_answer_ids")
        for a in v:
            if not _ANSWER_ID_RE.fullmatch(a):
                raise ValueError(f"malformed answer_id {a!r} in final_support_answer_ids")
        return v


class RefineBatchOutcomeRecord(BaseModel):
    """One immutable row per refine batch ATTEMPT -- published regardless of outcome, since every
    attempt (applied, call_failed, invalid_response) always produces a real, durably-persisted
    AIUseEvent. This is the ONE addressable, source-verifiable record of a batch's own existence
    once its accumulator update may have been overwritten by a later batch and its own event
    becomes otherwise unreachable from any surviving SynthesisClaim. A row with outcome in
    {"call_failed", "invalid_response"} whose own event carries an "internal_error: " failure_detail
    can NEVER legitimately appear here -- the live orchestration always raises internal_exc
    immediately after publishing such an event and BEFORE ever appending a record for it; the
    verifier treats such a record as proof of a hand-constructed or tampered output.

    pre_batch_claim_states/applied_updates (codex code-review round 2, 2026-08-26 finding 1): the
    live refine call renders its own prompt (and hashes it into work_item_id/rendered_prompt_hash)
    from the FULL prior accumulator state, which was previously not persisted anywhere a verifier
    could reconstruct -- a genuinely valid same-run refine event, produced against a DIFFERENT prior
    accumulator state, could be substituted for this batch's own event as long as target/batch
    answers/output-hash SHAPE happened to match. Persisting the exact pre-batch accumulator state
    plus the model's own real per-update response lets SynthesisOutput's own topology validator
    (below) walk every target's own batch sequence from empty state and prove EACH batch's own
    pre_batch_claim_states is exactly the state produced by every batch before it, closing the
    substitution gap structurally rather than merely checking each batch in isolation."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    rq_or_section: str = Field(min_length=1)
    batch_index: StrictInt = Field(ge=0)
    batch_answer_ids: List[str] = Field(min_length=1)   # ORDER is semantically significant -- this
                                                            # is the exact, ordered slice _chunk()
                                                            # produced, never re-sorted (below)
    outcome: Literal["applied", "call_failed", "invalid_response"]
    is_no_op: StrictBool = False
    producing_event_id: str = Field(min_length=1)
    pre_batch_claim_states: List[RefineClaimSnapshot] = Field(default_factory=list)
    applied_updates: List[RefineAppliedUpdate] = Field(default_factory=list)

    @field_validator("rq_or_section")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("batch_answer_ids")
    @classmethod
    def _well_formed_unique_ordered(cls, v: List[str]) -> List[str]:
        """Does not sort -- ORDER is load-bearing (it must reproduce _chunk()'s own real,
        deterministic slice exactly, SynthesisOutput's own topology check), so this validator only
        checks well-formedness/uniqueness and returns v UNCHANGED."""
        if len(v) != len(set(v)):
            raise ValueError("duplicate answer_id in batch_answer_ids")
        for a in v:
            if not _ANSWER_ID_RE.fullmatch(a):
                raise ValueError(f"malformed answer_id {a!r} in batch_answer_ids")
        return v

    @field_validator("producing_event_id")
    @classmethod
    def _check_event_id_format(cls, v: str) -> str:
        if not _AIUSE_EVENT_ID_RE.fullmatch(v):
            raise ValueError(f"producing_event_id {v!r} is not a well-formed AIUseEvent id")
        return v

    @model_validator(mode="after")
    def _applied_updates_consistency(self) -> "RefineBatchOutcomeRecord":
        if self.outcome != "applied" and self.applied_updates:
            raise ValueError("applied_updates must be empty unless outcome == 'applied'")
        if self.outcome == "applied" and self.is_no_op and self.applied_updates:
            raise ValueError("is_no_op == True means the model's own response had zero claim_updates -- "
                              "applied_updates must be empty")
        if self.outcome == "applied" and not self.is_no_op and not self.applied_updates:
            raise ValueError("a non-no-op applied batch must have at least one applied_updates entry")
        # Codex code-review round 3 (2026-08-26) finding 1: RefineResponse (nodes/synthesis.py)
        # forbids two updates targeting the same existing claim_index -- the live parser can never
        # produce this shape. applied_updates had no equivalent constraint, so a fabricated event
        # could represent a response production would have rejected as invalid_response.
        non_none_indices = [u.claim_index for u in self.applied_updates if u.claim_index is not None]
        if len(non_none_indices) != len(set(non_none_indices)):
            raise ValueError("two applied_updates entries target the same existing claim_index -- this "
                              "response shape is impossible from the live parser (RefineResponse forbids "
                              "two updates targeting the same existing claim_index)")
        return self

    @model_validator(mode="after")
    def _verify(self) -> "RefineBatchOutcomeRecord":
        if self.is_no_op and self.outcome != "applied":
            raise ValueError("is_no_op may only be True when outcome == 'applied'")
        return self


class ClaimConstructionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rq_or_section: str = Field(min_length=1)
    # StrictInt on all five counters -- a plain `int` field silently coerces bool (True -> 1) and
    # numeric strings ("1" -> 1) under pydantic's own lax mode, letting malformed serialized data
    # pass the derived-count equality checks even though it was never a genuine int to begin with.
    # StrictInt also rejects integral floats (5.0), which plain `int` would likewise have coerced.
    refine_batches_total: StrictInt = Field(0, ge=0)
    refine_batches_applied: StrictInt = Field(0, ge=0)
    refine_batches_applied_no_op: StrictInt = Field(0, ge=0)
    refine_batches_call_failed: StrictInt = Field(0, ge=0)
    refine_batches_invalid_response: StrictInt = Field(0, ge=0)
    unincorporated_answer_ids: List[str] = Field(default_factory=list)

    @field_validator("rq_or_section")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("unincorporated_answer_ids")
    @classmethod
    def _well_formed_unique(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate answer_id in unincorporated_answer_ids")
        for a in v:
            if not _ANSWER_ID_RE.fullmatch(a):
                raise ValueError(f"malformed answer_id {a!r} in unincorporated_answer_ids")
        return v

    @model_validator(mode="after")
    def _conservation(self) -> "ClaimConstructionCoverage":
        if self.refine_batches_applied + self.refine_batches_call_failed + self.refine_batches_invalid_response != self.refine_batches_total:
            raise ValueError("refine batch outcome counts do not sum to refine_batches_total")
        if self.refine_batches_applied_no_op > self.refine_batches_applied:
            raise ValueError("refine_batches_applied_no_op cannot exceed refine_batches_applied")
        return self


class RejectedSynthesisTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rq_or_section: str = Field(min_length=1)
    rejection_reason: Literal["no_claim_candidate_produced"] = "no_claim_candidate_produced"

    @field_validator("rq_or_section")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v


class ImradSectionOutcomeRecord(BaseModel):
    """One immutable row per legacy IMRAD section attempt (introduction/methods/discussion) -- the
    SAME "one addressable, source-verifiable outcome per attempt" pattern established for
    MapOutcomeRecord/RefineBatchOutcomeRecord. outcome="no_call" is the ONE case with no physical
    call and therefore no event -- the corpus/evidence-triviality short-circuit never invokes the
    real IMRAD call for any section, matching the SAME "no call means no event" contract
    MapOutcomeRecord's own outcome="no_citeable_evidence" already establishes. outcome="call_failed"/
    "invalid_response" mirror MapOutcomeRecord's own literal outcome vocabulary; "internal_error: "
    can NEVER legitimately appear here, for the SAME reason established for RefineBatchOutcomeRecord
    -- the real orchestration always raises internal_exc immediately after publishing such an event
    and BEFORE ever appending a record for it."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    section: Literal["introduction", "methods", "discussion"]
    outcome: Literal["success", "call_failed", "invalid_response", "no_call"]
    producing_event_id: Optional[str] = None
    failure_detail: Optional[str] = None

    @field_validator("producing_event_id")
    @classmethod
    def _check_event_id_format_if_present(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _AIUSE_EVENT_ID_RE.fullmatch(v):
            raise ValueError(f"producing_event_id {v!r} is not a well-formed AIUseEvent id")
        return v

    @model_validator(mode="after")
    def _verify(self) -> "ImradSectionOutcomeRecord":
        if self.outcome == "no_call":
            if self.producing_event_id is not None:
                raise ValueError("outcome='no_call' means no physical call was made -- "
                                  "producing_event_id must be None")
        elif self.producing_event_id is None:
            raise ValueError(f"outcome={self.outcome!r} means a physical call WAS made -- "
                              "producing_event_id is required")
        if self.outcome == "success":
            if self.failure_detail is not None:
                raise ValueError("outcome='success' must have failure_detail=None")
        elif not self.failure_detail or not self.failure_detail.strip():
            raise ValueError(f"outcome={self.outcome!r} requires a non-blank failure_detail")
        return self


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topics: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_map: Optional[Dict[str, Any]] = None
    # StrictInt/ge=0 (aggregation/export/reporting plan v17, U2): a plain
    # `int` field would silently coerce a bool (True -> 1) or a numeric
    # string ("2" -> 2) under pydantic's own lax mode, BEFORE any
    # reporting-side consistency check (_validate_bertopic_consistency())
    # ever sees the value -- closing the type-level hole at its source
    # rather than relying solely on a downstream check to catch it.
    bertopic_excluded_empty_evidence_count: StrictInt = Field(default=0, ge=0)

    extraction_run_id: Optional[str] = None
    quality_appraisal_run_id: Optional[str] = None
    requested_targets: List[str] = Field(default_factory=list)
    synthesis_record_ids: List[str] = Field(default_factory=list)

    map_outcome_records: List[MapOutcomeRecord] = Field(default_factory=list)
    map_answers: List[MapAnswer] = Field(default_factory=list)
    claims: List[SynthesisClaim] = Field(default_factory=list)
    rejected_targets: List[RejectedSynthesisTarget] = Field(default_factory=list)
    refine_batch_outcomes: List[RefineBatchOutcomeRecord] = Field(default_factory=list)
    refine_batch_size: StrictInt = Field(ge=1)   # the EFFECTIVE cfg.synthesis.refine_batch_size used
                                                    # for this run -- persisted so the topology validator
                                                    # can reconstruct the EXACT expected _chunk() output
                                                    # per target. NO default -- REQUIRED at construction,
                                                    # even for a zero-record/zero-target run (synthesis_node()
                                                    # always passes cfg.synthesis.refine_batch_size explicitly).
    claim_construction_coverage: List[ClaimConstructionCoverage] = Field(default_factory=list)
    target_coverage: Optional[float] = None

    rq_synthesis: Dict[str, str] = Field(default_factory=dict)
    imrad_claim_sections: Dict[str, str] = Field(default_factory=dict)
    imrad_narrative_sections: Dict[str, str] = Field(default_factory=dict)
    imrad_narrative_failures: Dict[str, str] = Field(default_factory=dict)
    imrad_section_outcomes: List[ImradSectionOutcomeRecord] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict)

    @field_validator("extraction_run_id", "quality_appraisal_run_id")
    @classmethod
    def _non_blank_run_id_if_present(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("if provided, must not be blank/whitespace-only")
        return v

    @field_validator("requested_targets")
    @classmethod
    def _requested_targets_canonical(cls, v: List[str]) -> List[str]:
        if any(t != t.strip() for t in v):
            raise ValueError("requested_targets entries must already be in canonical (stripped) form")
        if any(not t for t in v):
            raise ValueError("entries must not be blank")
        if len(v) != len(set(v)):
            raise ValueError("duplicate entries")
        return v

    @field_validator("synthesis_record_ids")
    @classmethod
    def _non_blank_entries_no_dupes(cls, v: List[str]) -> List[str]:
        if any(not s.strip() for s in v):
            raise ValueError("entries must not be blank/whitespace-only")
        if len(v) != len(set(v)):
            raise ValueError("duplicate entries")
        return v

    @model_validator(mode="after")
    def _verify_topology_and_derived_views(self) -> "SynthesisOutput":
        if bool(self.synthesis_record_ids) == (self.extraction_run_id is None):
            raise ValueError("synthesis_record_ids being non-empty must correspond exactly to "
                              "extraction_run_id being set, and vice versa")
        if not self.synthesis_record_ids and self.quality_appraisal_run_id is not None:
            raise ValueError("quality_appraisal_run_id must be None for an empty corpus")

        coverage_targets = [c.rq_or_section for c in self.claim_construction_coverage]
        if len(coverage_targets) != len(set(coverage_targets)):
            raise ValueError("duplicate rq_or_section in claim_construction_coverage")
        coverage_target_set = set(coverage_targets)

        requested_target_set = set(self.requested_targets)
        if coverage_target_set != requested_target_set:
            raise ValueError(f"claim_construction_coverage targets {coverage_target_set} != "
                              f"requested_targets {requested_target_set}")

        bad_imrad_targets = {t for t in coverage_target_set if t.startswith("imrad:")} - {"imrad:results"}
        if bad_imrad_targets:
            raise ValueError(f"only 'imrad:results' may be an imrad:-prefixed target, found {bad_imrad_targets}")

        claim_ids = [c.claim_id for c in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("duplicate claim_id in claims")

        accepted_targets = {c.rq_or_section for c in self.claims}
        if not accepted_targets <= coverage_target_set:
            raise ValueError(f"claims reference untracked target(s): {accepted_targets - coverage_target_set}")

        rejected_names = [r.rq_or_section for r in self.rejected_targets]
        if len(rejected_names) != len(set(rejected_names)):
            raise ValueError("duplicate rq_or_section in rejected_targets")
        rejected_target_set = set(rejected_names)
        if not rejected_target_set <= coverage_target_set:
            raise ValueError(f"rejected_targets reference untracked target(s): {rejected_target_set - coverage_target_set}")

        if accepted_targets & rejected_target_set:
            raise ValueError(f"target(s) both accepted and rejected: {accepted_targets & rejected_target_set}")
        if (accepted_targets | rejected_target_set) != coverage_target_set:
            raise ValueError(f"target(s) neither accepted nor rejected: "
                              f"{coverage_target_set - (accepted_targets | rejected_target_set)}")

        record_id_set = set(self.synthesis_record_ids)

        outcome_pairs = [(r.rq_or_section, r.record_id) for r in self.map_outcome_records]
        if len(outcome_pairs) != len(set(outcome_pairs)):
            raise ValueError("duplicate (rq_or_section, record_id) in map_outcome_records")
        expected_pairs = {(t, rid) for t in coverage_target_set for rid in record_id_set}
        if set(outcome_pairs) != expected_pairs:
            raise ValueError(f"map_outcome_records does not exactly cover every (target, record) pair -- "
                              f"missing={expected_pairs - set(outcome_pairs)}, "
                              f"foreign={set(outcome_pairs) - expected_pairs}")

        event_ids = [r.producing_event_id for r in self.map_outcome_records if r.producing_event_id is not None]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate producing_event_id across map_outcome_records")

        outcome_by_pair = {(r.rq_or_section, r.record_id): r for r in self.map_outcome_records}
        # A dict comprehension silently keeps only the LAST entry for a repeated key -- checked
        # explicitly BEFORE building/using answer_by_pair, so two distinct MapAnswers for the same
        # (target, record) pair (each with its own, different, legitimately-unique answer_id, so
        # the later global answer_id-uniqueness check alone would never catch this) can no longer
        # be silently collapsed to one.
        answer_pair_keys = [(a.rq_or_section, a.record_id) for a in self.map_answers]
        if len(answer_pair_keys) != len(set(answer_pair_keys)):
            raise ValueError("duplicate (rq_or_section, record_id) pair across map_answers -- at "
                              "most one MapAnswer may exist per (target, record)")
        answer_by_pair = {(a.rq_or_section, a.record_id): a for a in self.map_answers}
        relevant_pairs = {(r.rq_or_section, r.record_id) for r in self.map_outcome_records if r.outcome == "relevant"}
        if relevant_pairs != set(answer_by_pair):
            raise ValueError("map_outcome_records' 'relevant' pairs do not exactly match map_answers' own "
                              "(target, record) pairs")

        # map_answers' own LIST ORDER itself must equal the ONE true canonical (target, record)
        # iteration order -- derived INDEPENDENTLY of map_answers itself, from requested_targets x
        # synthesis_record_ids (both already-validated ordered lists, matching synthesis_node()'s
        # own real `for target in requested_targets: for record in records:` iteration) filtered to
        # map_outcome_records' own "relevant" entries. Without this, map_answers could be
        # arbitrarily reordered/permuted while still passing every set-based check above, and the
        # per-target exact _chunk() reconstruction (below) would then treat that reordering as if
        # it were genuinely canonical, since it would otherwise have no independent source of truth
        # for "canonical order" at all.
        expected_answer_pair_sequence = [(t, rid) for t in self.requested_targets for rid in self.synthesis_record_ids
                                          if outcome_by_pair.get((t, rid)) is not None
                                          and outcome_by_pair[(t, rid)].outcome == "relevant"]
        actual_answer_pair_sequence = [(a.rq_or_section, a.record_id) for a in self.map_answers]
        if actual_answer_pair_sequence != expected_answer_pair_sequence:
            raise ValueError("map_answers' own list order does not match the ONE canonical "
                              "(target, record) order (requested_targets x synthesis_record_ids, "
                              "filtered to 'relevant' map_outcome_records) -- map_answers may have "
                              "been reordered or permuted")

        for pair, a in answer_by_pair.items():
            if outcome_by_pair[pair].producing_event_id != a.producing_event_id:
                raise ValueError(f"MapAnswer {a.answer_id}: producing_event_id disagrees with its own "
                                  "MapOutcomeRecord's producing_event_id")

        answer_ids_list = [a.answer_id for a in self.map_answers]
        if len(answer_ids_list) != len(set(answer_ids_list)):
            raise ValueError("duplicate answer_id in map_answers")
        foreign_records = {a.record_id for a in self.map_answers} - record_id_set
        if foreign_records:
            raise ValueError(f"map_answers reference record_id(s) outside synthesis_record_ids: {foreign_records}")
        for a in self.map_answers:
            for r in a.evidence_references:
                expected_run_id = self.quality_appraisal_run_id if r.artifact_kind == "quality_appraisal" else self.extraction_run_id
                if r.artifact_run_id != expected_run_id:
                    raise ValueError(f"MapAnswer {a.answer_id}: reference artifact_run_id {r.artifact_run_id!r} "
                                      f"does not match the active {r.artifact_kind} run {expected_run_id!r}")

        known_answers = {a.answer_id: a for a in self.map_answers}
        for c in self.claims:
            missing = set(c.supporting_answer_ids) - set(known_answers)
            if missing:
                raise ValueError(f"claim {c.claim_id} references unknown answer_id(s): {missing}")
            expected_refs: Dict[tuple, EvidenceReference] = {}
            for aid in c.supporting_answer_ids:
                for r in known_answers[aid].evidence_references:
                    expected_refs.setdefault(_reference_sort_key(r), r)
            expected_sorted = sorted(expected_refs.values(), key=_reference_sort_key)
            if c.references != expected_sorted:
                raise ValueError(f"claim {c.claim_id}: references do not equal the canonical union of its "
                                  "supporting answers' own evidence_references")

        coverage_by_target = {c.rq_or_section: c for c in self.claim_construction_coverage}

        # refine_batch_outcomes is the ONE addressable, source-verifiable record of every refine
        # batch ATTEMPT -- ClaimConstructionCoverage's own counts are no longer trusted as
        # freestanding integers, they are DERIVED from and checked exactly against this raw data,
        # the same "derived view checked against raw data" pattern used for rq_synthesis/
        # imrad_claim_sections above.
        rbo_keys = [(rbo.rq_or_section, rbo.batch_index) for rbo in self.refine_batch_outcomes]
        if len(rbo_keys) != len(set(rbo_keys)):
            raise ValueError("duplicate (rq_or_section, batch_index) pair in refine_batch_outcomes")
        rbo_event_ids = [rbo.producing_event_id for rbo in self.refine_batch_outcomes]
        if len(rbo_event_ids) != len(set(rbo_event_ids)):
            raise ValueError("duplicate producing_event_id across refine_batch_outcomes")
        foreign_rbo_targets = {rbo.rq_or_section for rbo in self.refine_batch_outcomes} - coverage_target_set
        if foreign_rbo_targets:
            raise ValueError(f"refine_batch_outcomes reference untracked target(s): {foreign_rbo_targets}")

        rbo_by_target: Dict[str, List[RefineBatchOutcomeRecord]] = {}
        for rbo in self.refine_batch_outcomes:
            rbo_by_target.setdefault(rbo.rq_or_section, []).append(rbo)

        for target in coverage_target_set:
            relevant_answer_ids = {a.answer_id for a in self.map_answers if a.rq_or_section == target}
            cov = coverage_by_target[target]
            supported_ids = {aid for c in self.claims if c.rq_or_section == target for aid in c.supporting_answer_ids}
            unincorporated_ids = set(cov.unincorporated_answer_ids)
            if supported_ids & unincorporated_ids:
                raise ValueError(f"target {target!r}: answer id(s) both supported and unincorporated")
            if (supported_ids | unincorporated_ids) != relevant_answer_ids:
                raise ValueError(f"target {target!r}: supported ∪ unincorporated does not equal this "
                                  "target's own relevant MapAnswers")

            target_rbos = rbo_by_target.get(target, [])
            target_batch_indices = sorted(rbo.batch_index for rbo in target_rbos)
            if target_batch_indices != list(range(len(target_rbos))):
                raise ValueError(f"target {target!r}: refine_batch_outcomes batch_index values are not "
                                  f"contiguous from 0 (found {target_batch_indices!r})")

            # A SET-based partition check (pairwise disjoint + exhaustive union) is not enough --
            # for answers [A,B,C] and refine_batch_size=2, RBO rows claiming batches [A,C] and [B]
            # are ALSO disjoint and exhaustive, yet do not represent the real, deterministic
            # _chunk() execution ([A,B],[C]) at all; batch ORDER was never enforced either.
            # Reconstruct the EXACT expected chunking from this target's own canonical answer order
            # -- self.map_answers preserves construction order (never reordered by any
            # field_validator, unlike selectors/evidence_references), which IS the same order
            # _finalize_target()'s own `target_answers = [a for a in map_answers if a.rq_or_section
            # == target]` sliced via _chunk() in the first place -- and require EXACT
            # list-of-lists equality against this target's own refine_batch_outcomes, sorted by
            # batch_index, in order.
            target_ordered_answer_ids = [a.answer_id for a in self.map_answers if a.rq_or_section == target]
            expected_chunks = [target_ordered_answer_ids[i:i + self.refine_batch_size]
                                for i in range(0, len(target_ordered_answer_ids), self.refine_batch_size)]
            target_rbos_by_index = sorted(target_rbos, key=lambda rbo: rbo.batch_index)
            actual_chunks = [rbo.batch_answer_ids for rbo in target_rbos_by_index]
            if actual_chunks != expected_chunks:
                raise ValueError(f"target {target!r}: refine_batch_outcomes does not reproduce the EXACT "
                                  "deterministic _chunk() execution over this target's own canonical "
                                  f"answer order with refine_batch_size={self.refine_batch_size} -- "
                                  f"expected {expected_chunks!r}, got {actual_chunks!r}")

            expected_total = len(target_rbos)
            expected_applied = sum(1 for rbo in target_rbos if rbo.outcome == "applied")
            expected_applied_no_op = sum(1 for rbo in target_rbos if rbo.outcome == "applied" and rbo.is_no_op)
            expected_call_failed = sum(1 for rbo in target_rbos if rbo.outcome == "call_failed")
            expected_invalid_response = sum(1 for rbo in target_rbos if rbo.outcome == "invalid_response")
            if (cov.refine_batches_total, cov.refine_batches_applied, cov.refine_batches_applied_no_op,
                cov.refine_batches_call_failed, cov.refine_batches_invalid_response) != (
                    expected_total, expected_applied, expected_applied_no_op,
                    expected_call_failed, expected_invalid_response):
                raise ValueError(f"target {target!r}: ClaimConstructionCoverage counts do not match what "
                                  "refine_batch_outcomes actually records -- "
                                  f"expected total={expected_total} applied={expected_applied} "
                                  f"applied_no_op={expected_applied_no_op} call_failed={expected_call_failed} "
                                  f"invalid_response={expected_invalid_response}, got total={cov.refine_batches_total} "
                                  f"applied={cov.refine_batches_applied} applied_no_op={cov.refine_batches_applied_no_op} "
                                  f"call_failed={cov.refine_batches_call_failed} "
                                  f"invalid_response={cov.refine_batches_invalid_response}")

            # Codex code-review round 2 (2026-08-26) finding 1: walk this target's own batch
            # sequence from EMPTY accumulator state, replaying each "applied" batch's own
            # applied_updates through the identical allowed-set/simulate logic
            # _validate_and_simulate() applies at runtime (duplicated here in pure form since
            # state.py sits below nodes/synthesis.py and cannot import from it) -- proving each
            # batch's own pre_batch_claim_states is EXACTLY the state produced by every real batch
            # before it, never a substituted/stale accumulator snapshot, and that the target's own
            # FINAL simulated state exactly equals its own persisted `claims`.
            #
            # Codex code-review round 3 (2026-08-26) finding 2: each accumulator entry ALSO carries
            # its own producing_event_id, set for a TOUCHED index to this batch's own
            # rbo.producing_event_id and left UNCHANGED for an untouched index -- EXACTLY
            # _commit_simulated_state()'s own real runtime behavior -- so a final claim can be bound
            # to the event that ACTUALLY produced its own current accumulator position, not merely
            # to ANY applied event for the target (which a claim with content identical to some
            # OTHER position/batch could previously satisfy).
            accumulator_state: List[Tuple[str, Tuple[str, ...], str]] = []
            for rbo in target_rbos_by_index:
                actual_pre = [(s.claim_text, tuple(s.supporting_answer_ids), s.producing_event_id)
                              for s in rbo.pre_batch_claim_states]
                if actual_pre != accumulator_state:
                    raise ValueError(f"target {target!r} batch {rbo.batch_index}: pre_batch_claim_states does "
                                      "not match the accumulator state produced by this target's own prior "
                                      "batches -- a valid event from a DIFFERENT accumulator context may have "
                                      "been substituted here")
                if rbo.outcome != "applied":
                    continue
                batch_answer_ids = set(rbo.batch_answer_ids)
                simulated = list(accumulator_state)
                touched: set = set()
                for u in rbo.applied_updates:
                    if u.claim_index is None:
                        allowed = batch_answer_ids
                    else:
                        if not (0 <= u.claim_index < len(accumulator_state)):
                            raise ValueError(f"target {target!r} batch {rbo.batch_index}: applied_updates "
                                              f"claim_index {u.claim_index} is out of range 0.."
                                              f"{len(accumulator_state) - 1} for the accumulator state at "
                                              "this point in the sequence")
                        allowed = batch_answer_ids | set(accumulator_state[u.claim_index][1])
                    bad = set(u.final_support_answer_ids) - allowed
                    if bad:
                        raise ValueError(f"target {target!r} batch {rbo.batch_index}: applied_updates "
                                          f"references answer id(s) {sorted(bad)} outside the allowed set")
                    entry = (u.claim_text, tuple(sorted(set(u.final_support_answer_ids))), rbo.producing_event_id)
                    if u.claim_index is None:
                        simulated.append(entry)
                        touched.add(len(simulated) - 1)
                    else:
                        simulated[u.claim_index] = entry
                        touched.add(u.claim_index)
                # Content-only dedup (ignores producing_event_id), matching _validate_and_simulate()'s
                # own real runtime check exactly -- two accumulator entries with the SAME text/answer-
                # ids but DIFFERENT producing_event_id are still "identical claims" content-wise.
                content_only = [(text, ids) for text, ids, _ in simulated]
                if len(set(content_only)) != len(content_only):
                    raise ValueError(f"target {target!r} batch {rbo.batch_index}: applied_updates would "
                                      "produce two claims with identical claim_text and "
                                      "final_support_answer_ids")
                accumulator_state = simulated

            expected_final_claims = [(c.claim_text, tuple(c.supporting_answer_ids), c.producing_event_id)
                                       for c in self.claims if c.rq_or_section == target]
            if accumulator_state != expected_final_claims:
                raise ValueError(f"target {target!r}: the final accumulator state reconstructed from "
                                  "refine_batch_outcomes' own pre_batch_claim_states/applied_updates chain "
                                  "does not match this target's own persisted `claims` (text, answer ids, "
                                  "or producing_event_id disagree)")

        # Every claim's own producing_event_id must correspond to a REAL, listed "applied" batch
        # outcome for that SAME target -- ties claim provenance back to the raw batch record, not
        # just to an isolated event_id string.
        applied_event_ids_by_target: Dict[str, set] = {}
        for rbo in self.refine_batch_outcomes:
            if rbo.outcome == "applied":
                applied_event_ids_by_target.setdefault(rbo.rq_or_section, set()).add(rbo.producing_event_id)
        for c in self.claims:
            if c.producing_event_id not in applied_event_ids_by_target.get(c.rq_or_section, set()):
                raise ValueError(f"claim {c.claim_id}: producing_event_id does not correspond to any "
                                  "'applied' entry in refine_batch_outcomes for this claim's own target")

        grouped: Dict[str, List[str]] = {}
        for c in self.claims:
            grouped.setdefault(c.rq_or_section, []).append(c.claim_text)
        joined = {k: "\n\n".join(v) for k, v in grouped.items()}
        expected_rq = {k: v for k, v in joined.items() if not k.startswith("imrad:")}
        expected_imrad_claim = {k[len("imrad:"):]: v for k, v in joined.items() if k.startswith("imrad:")}
        if self.rq_synthesis != expected_rq:
            raise ValueError("rq_synthesis is not the deterministic ordered join of `claims`")
        if self.imrad_claim_sections != expected_imrad_claim:
            raise ValueError("imrad_claim_sections is not the deterministic ordered join of `claims`")
        if self.sections != self.imrad_claim_sections:
            raise ValueError("sections (deprecated view) must equal imrad_claim_sections exactly")

        allowed_narrative_keys = {"introduction", "methods", "discussion"}
        if set(self.imrad_narrative_sections) - allowed_narrative_keys:
            raise ValueError("imrad_narrative_sections may only contain introduction/methods/discussion")
        if set(self.imrad_narrative_failures) - allowed_narrative_keys:
            raise ValueError("imrad_narrative_failures may only contain introduction/methods/discussion")
        for reason in self.imrad_narrative_failures.values():
            if not reason.strip():
                raise ValueError("imrad_narrative_failures value must not be blank")
        if set(self.imrad_claim_sections) & set(self.imrad_narrative_sections):
            raise ValueError("imrad_claim_sections and imrad_narrative_sections must be disjoint")
        if set(self.imrad_narrative_sections) & set(self.imrad_narrative_failures):
            raise ValueError("a section cannot be both a successful narrative output and a recorded failure")

        # imrad_section_outcomes is the ONE addressable, source-verifiable record of every legacy
        # IMRAD section ATTEMPT -- imrad_narrative_sections/imrad_narrative_failures are no longer
        # trusted as freestanding, unverifiable strings, they are checked for exact correspondence
        # against this raw data, the same "derived view checked against raw data" pattern already
        # used for refine_batch_outcomes.
        imrad_outcome_sections = [r.section for r in self.imrad_section_outcomes]
        if len(imrad_outcome_sections) != len(set(imrad_outcome_sections)):
            raise ValueError("duplicate section in imrad_section_outcomes")
        imrad_outcome_section_set = set(imrad_outcome_sections)
        expected_imrad_outcome_section_set = set(self.imrad_narrative_sections) | set(self.imrad_narrative_failures)
        if imrad_outcome_section_set != expected_imrad_outcome_section_set:
            raise ValueError(f"imrad_section_outcomes covers {imrad_outcome_section_set!r} but "
                              "imrad_narrative_sections/imrad_narrative_failures together name "
                              f"{expected_imrad_outcome_section_set!r} -- must cover exactly the same sections")
        for r in self.imrad_section_outcomes:
            if r.outcome == "success":
                if r.section not in self.imrad_narrative_sections:
                    raise ValueError(f"imrad_section_outcomes: section {r.section!r} outcome='success' "
                                      "but not present in imrad_narrative_sections")
            else:
                if r.section not in self.imrad_narrative_failures:
                    raise ValueError(f"imrad_section_outcomes: section {r.section!r} outcome={r.outcome!r} "
                                      "but not present in imrad_narrative_failures")
                if r.failure_detail != self.imrad_narrative_failures[r.section]:
                    raise ValueError(f"imrad_section_outcomes: section {r.section!r} failure_detail "
                                      "disagrees with imrad_narrative_failures' own recorded reason")

        total_targets = len(self.claim_construction_coverage)
        expected_target_coverage = None if total_targets == 0 else len(accepted_targets) / total_targets
        if self.target_coverage != expected_target_coverage:
            raise ValueError(f"target_coverage={self.target_coverage!r} != recomputed {expected_target_coverage!r}")
        if self.target_coverage is not None and not (0.0 <= self.target_coverage <= 1.0):
            raise ValueError(f"target_coverage={self.target_coverage!r} out of [0,1]")
        return self


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------

class ReviewState(TypedDict):
    """
    Shared state that flows through every node in the pipeline.

    LangGraph passes the full state to each node; nodes return a dict
    containing only the keys they modify. Keys not in the returned dict
    are left unchanged.
    """

    # --- Review setup ---
    config: Dict[str, Any]          # ReviewConfig serialised as dict
    review_type: str                 # "scoping" | "rapid" | "slr"
    protocol: Optional[Dict[str, Any]]
    search_strategy: Optional[Dict[str, Any]]   # after search_strategy_builder_node

    # --- Records at each pipeline stage (all stored as dicts for JSON compat) ---
    raw_records: List[Dict[str, Any]]           # after search_node
    deduplicated_records: List[Dict[str, Any]]  # after dedup_node
    included_records: List[Dict[str, Any]]      # after screening_node (title/abstract)
    screened_records_full: List[Dict[str, Any]] # ALL records screening_node saw, incl. excluded
    full_text_records: List[Dict[str, Any]]     # after fulltext_node -- ONLY records FT_SCREENING
                                                 # decided "include"; never populated by an abstract/
                                                 # title fallback or by include-on-error (see fulltext.py)
    charted_data: List[Dict[str, Any]]          # after extraction_node (all three review types)

    # --- Full-text eligibility review queue (set by fulltext_node) ---
    # Records that could not be given an "include"/"exclude" full-text
    # decision -- EITHER because no confirmed PDF ever reached FT_SCREENING
    # (stage="acquisition": no_doi/no_oa_fulltext_found/rejected_stub) OR
    # because FT_SCREENING itself returned needs_human_review/processing_
    # failed/input_unavailable/duplicate_input (stage="ft_screening", the
    # full ChapterFullTextOutcome dump -- item_result/run_artifacts included,
    # so a reviewer has the complete criterion-level evidence/rationale or
    # error detail without leaving state). These records are NEVER included
    # in full_text_records -- conservative-by-construction, not a policy
    # applied after the fact.
    full_text_review_queue: List[Dict[str, Any]]

    # --- Full-text review resolution audit trail (set by
    # nodes.fulltext.resolve_full_text_review_queue(), never by
    # fulltext_node itself) -- one entry per resolved queue record: who
    # decided (reviewer), why (note, and reason when excluding), and when
    # (resolved_at). Never embedded into Record itself, which has no
    # fields for this and is used far beyond the full-text stage.
    full_text_review_resolutions: List[Dict[str, Any]]

    # --- FT_SCREENING run provenance (set by fulltext_node; None if no
    # confirmed PDF ever required a run) -- the frozen run files themselves
    # stay on disk, referenced by path here, never copied/simplified into
    # state beyond what full_text_review_queue's item_result entries already
    # carry. ---
    ft_screening_run_id: Optional[str]
    ft_screening_execution_fingerprint: Optional[str]
    ft_screening_artifact_paths: Optional[Dict[str, str]]  # run_result_json/item_results_jsonl/ledger_jsonl

    # --- Structured data extraction, via nodes.extraction and the frozen
    # Phase A engine in slr_pipeline/extraction/* -- ALL THREE review
    # types since Stage 2 of quality_appraisal_synthesis_plan_2026-08-22.md
    # (scoping's own charting is re-platformed onto this same engine via
    # slr_pipeline.adapters.scoping_charting_adapter). ---

    # Complete, evidence-bearing per-document results -- one
    # DocumentExtractionResult.model_dump(mode="json") per record in
    # full_text_records, in the SAME order. This, not charted_data, is the
    # scientific artifact (see ChartedRecord.charting_method's own note).
    extraction_results: List[Dict[str, Any]]

    # One entry per field with requires_human_review=True or status
    # unclear/conflicting/extraction_failed (set by nodes.extraction.
    # extraction_node) -- {"record_id", "field_id", "status",
    # "field_result": <complete FieldResult dump>}. Never mutated once
    # written; nodes.extraction.resolve_extraction_review_queue() only
    # ever removes entries here, appending to
    # extraction_review_resolutions instead.
    extraction_review_queue: List[Dict[str, Any]]

    # Human resolution audit trail (set by nodes.extraction.
    # resolve_extraction_review_queue(), never by extraction_node itself)
    # -- one entry per resolved (record_id, field_id): who decided
    # (reviewer), why (note), when (resolved_at), and the resolved
    # outcome ("extracted" with value/evidence/attested, or
    # "not_reported"). The ORIGINAL machine FieldResult is never
    # overwritten -- it stays exactly as extraction_node produced it in
    # extraction_results; this is a SEPARATE, additive record.
    extraction_review_resolutions: List[Dict[str, Any]]

    # --- Extraction run provenance (set by extraction_node; None if
    # extraction never ran, e.g. full_text_records was empty) -- the
    # frozen Phase A checkpoint/manifest files stay on disk, referenced
    # by path/hash here, never copied or simplified into state. ---
    extraction_run_id: Optional[str]
    extraction_manifest_path: Optional[str]
    extraction_manifest_hash: Optional[str]
    extraction_artifact_paths: Optional[Dict[str, str]]  # run_dir/checkpoints_dir/manifest_path
    extraction_schema_hash: Optional[str]

    # --- Quality appraisal (rapid / SLR only; empty if skipped -- see
    # nodes.quality_appraisal's own module docstring, Stage 1 of
    # quality_appraisal_synthesis_plan_2026-08-22.md). Renamed from the
    # prior `quality_scores` -- there is no aggregate score any more (no
    # `gate_on_threshold`, no numeric quality_threshold); this stage NEVER
    # mutates full_text_records, under any circumstance. ---

    # Raw, process-local per-document DocumentAppraisal.model_dump()
    # results, one per record in full_text_records, in the SAME order --
    # analogous to extraction_results. NOT the authoritative artifact for
    # downstream consumers: synthesis_node/reporting_node reload and
    # re-verify the durable resolution artifact via
    # nodes.quality_appraisal.load_verified_quality_appraisal_resolution_artifact()
    # instead of trusting this list at face value (same reasoning as
    # extraction_results vs. the extraction resolution artifact).
    quality_appraisal_results: List[Dict[str, Any]]

    # One entry per leaf (item, or "domain_id.question_id" signalling
    # question) that currently needs a human action -- EITHER a
    # resolution (unclear / evidence-verification-failed / conflicting-
    # cross-batch-evidence / batch-call-failed; cleared via
    # resolve_quality_appraisal_review_queue()) OR a confirmation (a
    # machine-verified negative/high-risk leaf under
    # negative_high_risk_policy="require_human_confirmation"; cleared via
    # confirm_quality_appraisal_provisional_item(), which preserves the
    # machine's own evidence rather than overriding it). Each entry's own
    # "queue_kind" field ("resolution" | "confirmation") says which.
    quality_appraisal_review_queue: List[Dict[str, Any]]

    # Human resolution audit trail (set by
    # nodes.quality_appraisal.resolve_quality_appraisal_review_queue(),
    # never by quality_appraisal_node itself).
    quality_appraisal_resolutions: List[Dict[str, Any]]

    # Human confirmation audit trail (set by
    # nodes.quality_appraisal.confirm_quality_appraisal_provisional_item())
    # -- a confirmation is NOT a resolution: it never touches evidence/
    # decision_origin/evidence_status/answer/status, only records that a
    # human looked at an already-verified negative/high-risk machine
    # judgment and agreed.
    quality_appraisal_confirmations: List[Dict[str, Any]]

    # Non-blocking record of provisional negative/high-risk leaves under
    # negative_high_risk_policy="allow_provisional_ai_only" -- these never
    # enter quality_appraisal_review_queue and never block the graph gate
    # (make_post_quality_appraisal_router only checks the queue above).
    quality_appraisal_provisional_ledger: List[Dict[str, Any]]

    # --- Quality appraisal run provenance (set by quality_appraisal_node;
    # None if quality appraisal never ran, e.g. full_text_records was
    # empty) -- the frozen manifest/checkpoints files stay on disk,
    # referenced by path/hash here, never copied or simplified into
    # state. Independent of extraction_run_id/extraction_manifest_hash --
    # see plan Section 4's "Worries" note. ---
    quality_appraisal_run_id: Optional[str]
    quality_appraisal_manifest_path: Optional[str]
    quality_appraisal_manifest_hash: Optional[str]
    quality_appraisal_artifact_paths: Optional[Dict[str, str]]

    # --- Screening quality control ---
    screening_qc: Optional[Dict[str, Any]]

    # --- Synthesis ---
    synthesis: Optional[Dict[str, Any]]

    # --- Tracking ---
    prisma_counts: Dict[str, Any]
    search_log: Dict[str, Any]
    pipeline_errors: List[str]

    # --- Human-in-the-loop ---
    requires_human_review: bool
    human_feedback: Optional[str]

    # --- Final outputs ---
    prisma_diagram_path: Optional[str]
    methods_text: Optional[str]
    manual_worklist_path: Optional[str]   # set by fulltext_node (Tier 3 CSV)

    # --- Stage 3 traceability/selection-ledger identity (set by whichever
    # instrumented node runs first this invocation -- dedup_node,
    # screening_node, screening_qc_node, search_strategy_builder_node, or
    # fulltext_node; see slr_pipeline.run_identity.resolve_review_run_id().
    # Constant for the whole graph invocation once set -- every
    # instrumented node re-derives/validates it against the active
    # LangGraph thread_id and returns it unchanged. ---
    review_run_id: Optional[str]
    # Run-isolated ledger directory, mirroring how other durable Stage 3
    # artifacts are referenced by path from state rather than copied in --
    # output_dir/selection_decisions/<review_run_id>/ exactly (never its
    # shared parent). Set alongside review_run_id by any node that touches
    # the selection ledger (dedup_node/screening_node/fulltext_node).
    selection_decision_ledger_dir: Optional[str]

    # --- Stage 3 oversight/automation-origin bookkeeping, stage-keyed.
    # oversight_mode (human-involvement axis): "automation_unreviewed" |
    # "automation_human_verified" | "human_resolved" | "human_generated".
    # automation_origin (mechanism axis, independent of oversight_mode):
    # "ai" | "rule_based_automation" | "mixed" | "no_automated_output".
    # Every instrumented node reads-copies-updates ONLY its own stage's
    # key -- never wholesale-overwrites another stage's entry (this dict
    # field has no LangGraph reducer, so a plain node return would
    # otherwise silently drop earlier stages' keys). Absent key means
    # that stage did not execute this run, never any of the 4/4 values. ---
    oversight_mode: Dict[str, str]
    automation_origin: Dict[str, str]

    # --- Screening human-resolution audit trail (set by screening_node's
    # own human-apply step, the interrupt()/resume analogue of
    # full_text_review_resolutions below) -- one entry per resolved
    # borderline/queued record: record_id, decision, reviewer, note,
    # resolved_at, resolution_id, reviewed_baseline_decision_id. ---
    screening_review_resolutions: List[Dict[str, Any]]

    # --- Aggregation/export/reporting phase (aggregation_export_reporting_
    # plan_2026-08-25.md v17) -- set by reporting_node, the terminal node.
    # Every path here has already passed its own manifest self-hash check
    # (export_manifest_path's own contents) before reporting_node returns;
    # a failed run leaves NONE of these set (manifest-as-commit-marker). ---
    ai_traceability_json_path: Optional[str]
    ai_traceability_md_path: Optional[str]
    quality_appraisal_csv_path: Optional[str]
    quality_appraisal_json_path: Optional[str]
    evidence_map_csv_path: Optional[str]
    synthesis_claims_json_path: Optional[str]
    synthesis_json_path: Optional[str]
    synthesis_md_path: Optional[str]
    claim_audit_json_path: Optional[str]
    claim_audit_md_path: Optional[str]
    export_manifest_path: Optional[str]

    # --- Step 14 ([C13]) PRISMA-ScR checklist report -- content, not a
    # path (mirrors methods_text above): both keys are None whenever
    # `reference_context` is unset (every production run), since neither
    # renderer is applicable then. `prisma_scr_conformance` is the parsed
    # `prisma_scr_conformance.json` dict, not its raw bytes. ---
    prisma_scr_report_md: Optional[str]
    prisma_scr_conformance: Optional[Dict[str, Any]]
    prisma_scr_report_sha256: Optional[str]
    prisma_scr_conformance_sha256: Optional[str]


def initial_state(config: Dict[str, Any]) -> ReviewState:
    """Return a blank ReviewState for starting a new review.

    Codex review (Phase A follow-up, finding 1 -- P0): `config` is
    sanitized through `ReviewConfig` before being stored, rather than
    stored verbatim. The public `run_scoping_review()`/`run_rapid_review()`/
    `run_slr()` functions all accept a raw `dict`, not necessarily an
    already-validated `ReviewConfig` object -- a hand-built or historical
    dict carrying a real `openai_api_key` would otherwise bypass
    `ReviewConfig`'s own credential-exclusion entirely and land verbatim in
    `ReviewState` (and therefore in any persisted/checkpointed state),
    reproduced directly by Codex: a sentinel value survived in
    `state["config"]` and in serialized state before this fix.
    Round-tripping through `ReviewConfig.from_dict(...).to_dict()` here
    means the credential is stripped at this ONE boundary, regardless of
    whether the caller passed an already-sanitized `cfg.to_dict()` or an
    arbitrary raw dict. `ReviewConfig` has no `extra="forbid"` of its own,
    so unrecognized keys are silently dropped by the round-trip, not
    raised -- an intentional, harmless side effect of enforcing this
    invariant uniformly."""
    config = ReviewConfig.from_dict(config).to_dict()
    return ReviewState(
        config=config,
        review_type=config.get("review_type", "scoping"),
        protocol=None,
        search_strategy=None,
        raw_records=[],
        deduplicated_records=[],
        included_records=[],
        screened_records_full=[],
        full_text_records=[],
        charted_data=[],
        full_text_review_queue=[],
        full_text_review_resolutions=[],
        ft_screening_run_id=None,
        ft_screening_execution_fingerprint=None,
        ft_screening_artifact_paths=None,
        extraction_results=[],
        extraction_review_queue=[],
        extraction_review_resolutions=[],
        extraction_run_id=None,
        extraction_manifest_path=None,
        extraction_manifest_hash=None,
        extraction_artifact_paths=None,
        extraction_schema_hash=None,
        quality_appraisal_results=[],
        quality_appraisal_review_queue=[],
        quality_appraisal_resolutions=[],
        quality_appraisal_confirmations=[],
        quality_appraisal_provisional_ledger=[],
        quality_appraisal_run_id=None,
        quality_appraisal_manifest_path=None,
        quality_appraisal_manifest_hash=None,
        quality_appraisal_artifact_paths=None,
        screening_qc=None,
        synthesis=None,
        prisma_counts=PRISMACounts().model_dump(),
        search_log=SearchLog().model_dump(),
        pipeline_errors=[],
        requires_human_review=False,
        human_feedback=None,
        prisma_diagram_path=None,
        methods_text=None,
        manual_worklist_path=None,
        review_run_id=None,
        selection_decision_ledger_dir=None,
        oversight_mode={},
        automation_origin={},
        screening_review_resolutions=[],
        ai_traceability_json_path=None,
        ai_traceability_md_path=None,
        quality_appraisal_csv_path=None,
        quality_appraisal_json_path=None,
        evidence_map_csv_path=None,
        synthesis_claims_json_path=None,
        synthesis_json_path=None,
        synthesis_md_path=None,
        claim_audit_json_path=None,
        claim_audit_md_path=None,
        export_manifest_path=None,
        prisma_scr_report_md=None,
        prisma_scr_conformance=None,
        prisma_scr_report_sha256=None,
        prisma_scr_conformance_sha256=None,
    )
