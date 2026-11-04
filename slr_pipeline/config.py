"""
ReviewConfig — all settings for a pipeline run.

Keep config as a plain Pydantic model so it serialises cleanly into
ReviewState (which must be JSON-serialisable for LangGraph checkpointing).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class SearchStrategyConfig(BaseModel):
    """Settings for the LLM-assisted query expansion node."""
    enabled: bool = False              # opt-in; False = pass seed query through
    llm_model: str = "gpt-5.4-mini"
    max_synonyms_per_concept: int = 5
    include_framework_terms: bool = True
    prefer_title_abstract_fields: bool = True


class SearchConfig(BaseModel):
    databases: List[str] = Field(
        default=["openalex", "pubmed", "semantic_scholar", "arxiv"],
        description="Databases to query. Options: openalex, pubmed, crossref, "
                    "semantic_scholar, arxiv, springer, ieee",
    )
    max_results_per_database: int = 1000
    query: str = ""                    # primary Boolean seed query
    date_from: Optional[str] = None    # "YYYY"
    date_to: Optional[str] = None
    language: Optional[str] = "en"
    strategy_builder: SearchStrategyConfig = Field(default_factory=SearchStrategyConfig)
    # Track 2 plan §2.5/§4.4: when set, search_node dispatches each configured
    # database through the mapped adapter registry key instead of the database
    # name itself (Track 2's own rehearsal maps every canonical database to a
    # "fx_"-prefixed fixture adapter). None (production default) -> unmapped
    # dispatch, `key = db`. When set, its keys MUST equal `set(databases)`
    # exactly and every value MUST be a real `_ADAPTER_REGISTRY` key, checked
    # BEFORE any network call (search.py::_resolve_adapter_keys).
    execution_adapter_map: Optional[Dict[str, str]] = None

    # Track 2 plan §8: None (default) = production unchanged -- search_node
    # stamps every DatabaseSearchLog entry's `date` with `date.today()`. Set
    # explicitly to make a run's logged search date reproducible independent
    # of the wall-clock day it's replayed on.
    as_of_date: Optional[str] = None    # "YYYY-MM-DD"

    # Deprecated (Track 2 plan §2.7 -- same treatment as
    # ReviewConfig.openai_api_key below): these three fields used to be
    # exposed with no serialisation exclusion, so a real value set here
    # landed verbatim in `state["config"]` and every LangGraph checkpoint.
    # The real search backends (slr_pipeline/search_backends/*) read
    # PUBMED_API_KEY/PUBMED_EMAIL/SPRINGER_API_KEY/IEEE_API_KEY from
    # `os.environ` exclusively -- never from this config. `exclude=True`/
    # `repr=False` mean these are never re-serialised or printed by
    # `to_dict()`/`model_dump()`/`repr()`/`str()`; the validator discards
    # the value outright so it doesn't even persist in memory on the
    # instance.
    pubmed_email: Optional[str] = Field(default=None, exclude=True, repr=False)
    springer_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    ieee_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)

    @field_validator("pubmed_email", "springer_api_key", "ieee_api_key")
    @classmethod
    def _deprecated_credential_field(cls, v: Optional[str], info) -> Optional[str]:
        if v:
            warnings.warn(
                f"SearchConfig.{info.field_name} is deprecated and never used -- "
                "set the documented environment variable instead. The value is discarded.",
                DeprecationWarning,
                stacklevel=2,
            )
        return None


class DedupConfig(BaseModel):
    """Track 2 plan §3.1. `"auto"` (default) preserves production behavior
    unchanged: `dedup_node` always attempts the embedding pipeline first,
    falling back to the deterministic DOI/title-exact path on
    `ImportError`/any exception (a `heuristic_fallback` `AIUseEvent`,
    stage `dedup_embedding`). `"exact"` skips the embedding pipeline
    entirely and always runs the deterministic DOI/title-exact path
    directly, publishing a `dedup_exact` `AIUseEvent`
    (`tool_kind="non_llm_tool"`) instead. `"semantic"` is embedding-only
    with no fallback -- it raises if `unified_screening_pipeline_v1` is
    unavailable rather than silently degrading to the exact path (a
    fail-closed preflight)."""
    mode: Literal["exact", "semantic", "auto"] = "auto"


class ScreeningConfig(BaseModel):
    # LLM model for abstract screening
    llm_model: str = "gpt-5.4-mini"
    # Minimum confidence to auto-include without human review
    auto_include_threshold: float = 0.80
    # Records below this go to human review queue
    uncertain_threshold: float = 0.50
    # Sentence-transformer model for semantic similarity
    embedding_model: str = "all-MiniLM-L6-v2"
    # Track 2 plan §4.4: None (default) = production unchanged --
    # screening.py's own SCREENING_BATCH_SIZE / SCREENING_WORKERS env
    # constants govern batch partitioning and thread-pool concurrency. Set
    # explicitly to make a run's batching reproducible independent of the
    # environment it's replayed in -- see effective_batch_size()/
    # effective_workers() in screening.py, the sole readers of these fields.
    batch_size: Optional[int] = None
    workers: Optional[int] = None


class ScreeningQCConfig(BaseModel):
    enabled: bool = True
    # Number of records re-screened by an independent second rater for IRR
    # (kappa/PABAK/Gwet's AC1). Was previously ScreeningConfig.calibration_
    # sample_size, a field declared but never read anywhere in the pipeline.
    calibration_sample_size: int = 50
    # Second rater's LLM model. None -> reuse the primary screening model
    # (still an independent call/prompt, not a no-op).
    second_rater_model: Optional[str] = None
    n_bootstrap: int = 200
    saturation_levels: List[float] = Field(default_factory=lambda: [0.80, 0.90, 0.95])
    # Track 2 plan §4.4: None (default) = production unchanged --
    # screening_qc.py's IRR sample is drawn with the unseeded module-level
    # `random.sample(...)`. When set, `random.Random(calibration_seed).sample(...)`
    # is used instead, so a rerun over the same eligible pool draws the same
    # sample.
    calibration_seed: Optional[int] = None


class ChartingConfig(BaseModel):
    method: Literal["pydantic", "chunking"] = "pydantic"
    llm_model: str = "gpt-4o"
    # Path to Pydantic schema definition (for method="pydantic")
    schema_path: Optional[str] = None
    # Chunking parameters (for method="chunking")
    chunk_max_tokens: int = 1024
    chunk_buffer_size: int = 1
    chunk_breakpoint_percentile: int = 95


class QualityAppraisalConfig(BaseModel):
    """Stage 1 of quality_appraisal_synthesis_plan_2026-08-22.md. Only two
    checklist identities exist -- both honestly named ("teaching_appraisal_v1",
    never claiming JBI conformance; "rob2_inspired_study_level_teaching_v1",
    never claiming real per-outcome RoB2 fidelity). There is deliberately no
    `quality_threshold`/`gate_on_threshold` field: quality_appraisal_node
    NEVER mutates full_text_records (Section 1.2) -- appraisal and
    eligibility are never conflated."""

    checklist_id: Literal["teaching_appraisal_v1", "rob2_inspired_study_level_teaching_v1"] = "teaching_appraisal_v1"
    # Negative/high-risk machine judgments ("no" / "N" / "PN") are always
    # provisional until a human acts (Section 1.6). This policy controls
    # whether that provisional state BLOCKS the graph gate
    # (require_human_confirmation, the teaching default) or is recorded
    # non-blockingly for a demonstrable AI-only run (allow_provisional_ai_only).
    negative_high_risk_policy: Literal["require_human_confirmation", "allow_provisional_ai_only"] = "require_human_confirmation"


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_bertopic: bool = True
    n_topics: int = 20
    llm_model: str = "gpt-4o"
    generate_evidence_map: bool = False
    # generate_imrad removed, replaced by two independent flags (structured synthesis claims and
    # evidence integration -- the Stage 3 claim-construction slice of
    # quality_appraisal_synthesis_plan_2026-08-22.md's own §3.2). generate_claim_backed_results --
    # structured, evidence-verified claims for the Results section (map/refine over each study's
    # own charted/appraised data, nodes/synthesis.py). generate_legacy_imrad_narrative -- the
    # free-text, non-claim-backed Introduction/Methods/Discussion narrative (off by default; opt
    # in explicitly).
    generate_claim_backed_results: bool = False
    generate_legacy_imrad_narrative: bool = False
    # Number of per-document RQ answers folded into the running synthesis per refine iteration.
    # Keeps each refine call's prompt bounded regardless of corpus size instead of growing linearly
    # with n_records. StrictInt rejects bool AND numeric strings outright (no coercion at all --
    # an exact, unambiguous policy), Field(ge=1) rejects non-positive values; no field_validator
    # needed (a mode="after" field_validator would run AFTER pydantic's own lax coercion had
    # already turned True into 1, letting a bool value through).
    refine_batch_size: StrictInt = Field(5, ge=1)


class ReviewConfig(BaseModel):
    """Master configuration for a single review run."""

    review_type: Literal["scoping", "rapid", "slr"] = "scoping"
    title: str = "Untitled Review"
    output_dir: str = "outputs"

    # Research question(s) and framework
    research_questions: List[str] = Field(default_factory=list)
    # PCC (scoping / rapid)
    population: Optional[str] = None
    concept: Optional[str] = None
    context: Optional[str] = None
    # PICO (slr)
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    outcome: Optional[str] = None

    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)

    # Full-text eligibility screening (fulltext_node, via FT_SCREENING).
    # `inclusion_criteria`/`exclusion_criteria` above are free-text,
    # title/abstract-screening-era strings -- NEVER auto-converted into
    # this. A real ft_screening.CriteriaSet (serialised via
    # `.model_dump(mode="json")`) must be authored explicitly and set
    # here before fulltext_node can screen any confirmed PDF; if a
    # confirmed PDF exists and this is still None, fulltext_node raises
    # rather than silently skipping full-text eligibility or falling
    # back to the abstract. See slr_pipeline.adapters.ft_screening_adapter's
    # module docstring for why this is never derived automatically.
    full_text_criteria_set: Optional[Dict[str, Any]] = None
    # Optional override of the LLM/component configuration fulltext_node
    # uses (a serialised ft_screening.ScreeningConfig). None -> the
    # book's own calibrated pilot preset,
    # slr_pipeline.adapters.ft_screening_adapter.chapter_screening_config().
    full_text_screening_config: Optional[Dict[str, Any]] = None
    # What fulltext_node does with a record whose full text could not be
    # acquired at all (no DOI, no OA copy found, or a downloaded PDF
    # rejected as a preview/stub -- i.e. full_text_status != "parsed", so
    # it never reaches FT_SCREENING). "review" (default) preserves the
    # original behavior: queue it into full_text_review_queue for a human
    # to close out (see fulltext_node's own module docstring). "exclude"
    # instead drops it immediately as a terminal "report not retrieved"
    # outcome and never queues it -- required for a fully autonomous run,
    # where a non-empty full_text_review_queue at terminal state is a hard
    # error (see slr_pipeline/track3_run_child.py). This does NOT increment
    # PRISMACounts.excluded_full_text: that box is reserved for records that
    # reached FT_SCREENING and were evaluated-but-ineligible, each backed by
    # a SelectionDecisionRecord in the eligibility ledger, and
    # reporting_node's ledger-vs-PRISMA reconciliation requires the two to
    # match exactly. An acquisition failure never reaches FT_SCREENING and
    # is a deterministic, non-AI decision (no evidence was ever available to
    # assess), so it never creates a SelectionDecisionRecord -- it is
    # represented solely via PRISMACounts.fulltext_retrieval's non-"parsed"
    # counts (see reporting.py's _render_methods_md, which sums those as
    # "not retrieved").
    full_text_acquisition_failure_policy: Literal["review", "exclude"] = "review"

    # Structured data extraction (extraction_node, via the frozen Phase A
    # engine in slr_pipeline/extraction/*; ALL THREE review types --
    # scoping's own charting is re-platformed onto this same engine too,
    # via slr_pipeline.adapters.scoping_charting_adapter, per Stage 2 of
    # quality_appraisal_synthesis_plan_2026-08-22.md. `charting`/
    # `ChartingConfig` below are kept only for the legacy, non-production
    # example moved to examples/legacy/charting.py.). Every field here is
    # None by default and extraction_node RAISES if it is invoked with a
    # non-empty full_text_records while schema/model_config are still
    # None -- there is deliberately no fallback default model/reasoning_
    # effort/token budget the way `charting.llm_model`/
    # `quality_appraisal.llm_model` above have one (see
    # extraction/openai_runner.py's own OpenAIExtractionConfig docstring
    # for why this package never silently chooses those).

    # A serialised `slr_pipeline.extraction.schema.ExtractionSchema`
    # (`.model_dump(mode="json")`) -- loaded with
    # `ExtractionSchema.model_validate(...)`, never Python code (see
    # schema.py's own module docstring on why `charting.schema_path`'s
    # arbitrary-`.py`-import pattern is not repeated here).
    extraction_schema: Optional[Dict[str, Any]] = None
    # A serialised `slr_pipeline.extraction.openai_runner.
    # OpenAIExtractionConfig` (`dataclasses.asdict(...)`) -- explicit
    # model/model_config_id/reasoning_effort/max_completion_tokens/
    # expected_provider_model, authored by whoever configures the
    # review. Carries no credential field to serialise (credentials are
    # read lazily from OPENAI_API_KEY by openai_runner.py itself).
    extraction_model_config: Optional[Dict[str, Any]] = None
    # A serialised `slr_pipeline.extraction.parsing.DoclingParserOptions`
    # (`dataclasses.asdict(...)`). None -> extraction_node passes no
    # override, and `run_extraction()`/`extract_document()` fall back to
    # DoclingParserOptions()'s own defaults (the ENGINE's default, not
    # something this config silently chooses on its behalf).
    extraction_parser_options: Optional[Dict[str, Any]] = None
    # Mirrors `slr_pipeline.extraction.batching.DEFAULT_MAX_BATCH_CHARS`
    # (not imported here -- config.py is imported bare, without the repo
    # root on sys.path, by modules that never otherwise need it; see
    # nodes/fulltext.py's own docstring on that sys.path split). Passed
    # through to run_extraction() unchanged.
    extraction_max_batch_chars: int = 12_000

    # Quality appraisal (quality_appraisal_node, Stage 1 of
    # quality_appraisal_synthesis_plan_2026-08-22.md; rapid/SLR only --
    # scoping never runs quality appraisal). Mirrors extraction's own
    # "explicit model config, no silent default" discipline immediately
    # above: quality_appraisal_node RAISES if invoked with a non-empty
    # full_text_records while quality_appraisal_model_config is still None.

    # A serialised `slr_pipeline.extraction.openai_runner.
    # OpenAIExtractionConfig` (`dataclasses.asdict(...)`) -- the SAME
    # dataclass extraction uses (explicit model/model_config_id/
    # reasoning_effort/max_completion_tokens/expected_provider_model);
    # reused rather than re-declared, since it is exactly the same
    # "one explicit LLM call configuration" contract, not an
    # extraction-specific concept.
    quality_appraisal_model_config: Optional[Dict[str, Any]] = None
    # A serialised `slr_pipeline.extraction.parsing.DoclingParserOptions`.
    # None -> quality_appraisal_node passes no override, falling back to
    # DoclingParserOptions()'s own defaults.
    quality_appraisal_parser_options: Optional[Dict[str, Any]] = None
    quality_appraisal_max_batch_chars: int = 12_000

    # Protocol/registration provenance (PRISMA-trAIce M1, Section 3.3 --
    # recorded here, on ReviewConfig, since quality_appraisal_node's own
    # qa_manifest.json is the natural place an implementer looks for it
    # first; a human fills these in when configuring a review. All optional
    # -- explicitly null/empty when not supplied, never silently omitted.
    protocol_uri: Optional[str] = None
    protocol_hash: Optional[str] = None
    protocol_registration_id: Optional[str] = None
    ai_use_prespecified: Optional[bool] = None
    protocol_deviations: List[str] = Field(default_factory=list)
    # PRISMA-trAIce M10 -- a free-text declaration of the review's own data-
    # governance posture (e.g. what personal/sensitive data, if any, the
    # corpus contains and how it is handled). Optional, explicitly null
    # when not supplied -- never silently omitted from the reporting phase's
    # aggregated ai_traceability.json/export_manifest.json.
    data_governance_declaration: Optional[str] = None
    # Track 2 plan §4.4/§9: a free-text funding declaration (PRISMA-ScR item
    # 22), reported verbatim by the reporting stage. Optional, explicitly
    # null when not supplied -- same "never silently omitted" discipline as
    # `data_governance_declaration` above.
    funding_statement: Optional[str] = None
    # Track 2 plan §4.4/§9: a serialised
    # `slr_pipeline.reference_spec.ReferenceReportingContext`
    # (`.model_dump(mode="json")`) -- the reference run's own reporting
    # context, which `capture_reporting_snapshot` cannot build itself since
    # it only ever sees `state`/`cfg`/`run_id`. None (production default) --
    # only the Track 2 reference run sets this.
    reference_reporting_context: Optional[Dict[str, Any]] = None

    # Sub-configs
    search: SearchConfig = Field(default_factory=SearchConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    screening_qc: ScreeningQCConfig = Field(default_factory=ScreeningQCConfig)
    charting: ChartingConfig = Field(default_factory=ChartingConfig)
    quality_appraisal: QualityAppraisalConfig = Field(
        default_factory=QualityAppraisalConfig
    )
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)

    # LLM backend selection
    llm_backend: Literal["openai", "ollama", "hf_server"] = "openai"
    ollama_base_url: str = "http://localhost:11434/v1"
    hf_server_url: str = "http://localhost:8080/v1"
    # DEPRECATED, never used for authentication (Track 1 Phase A credential-
    # safety fix). Retained ONLY so a historical config file that still has
    # a real string here can be loaded (`from_dict`/`model_validate`) without
    # raising -- it is never read by any client-construction code path and
    # `exclude=True`/`repr=False` mean it is never re-serialised or printed
    # by `to_dict()`/`model_dump()`/`repr()`/`str()` either, by construction,
    # not by relying on every call site to redact it individually. Real
    # credential resolution is `resolve_openai_api_key()` below, which reads
    # `OPENAI_API_KEY` from the environment exclusively -- no fallback to
    # this field. A caller that genuinely needs to inject a credential
    # programmatically (e.g. a test) passes it as an ephemeral argument
    # directly to the OpenAI client at the call site, never through this
    # field or through ReviewState.
    openai_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)

    @field_validator("openai_api_key")
    @classmethod
    def _warn_legacy_credential_field(cls, v: Optional[str]) -> Optional[str]:
        if v:
            warnings.warn(
                "ReviewConfig.openai_api_key is deprecated and is never used for "
                "authentication -- set the OPENAI_API_KEY environment variable "
                "instead. This field is accepted only for backward-compatible "
                "parsing of historical config files; its value is discarded and "
                "never serialised or passed to any client.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Codex review finding 2: this previously returned `v` unchanged --
        # the docstring/warning claimed the value was "discarded" but it
        # factually remained on the instance (`cfg.openai_api_key` still held
        # the real secret in memory after validation). Returning None here
        # means the field is actually gone after construction, not merely
        # hidden from serialization.
        return None

    def to_dict(self) -> Dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict) -> "ReviewConfig":
        return cls(**d)

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def resolve_openai_api_key() -> Optional[str]:
    """The ONE shared OpenAI-credential resolver -- every OpenAI-client
    construction site in this pipeline calls this instead of inlining its
    own `cfg.openai_api_key or ...` pattern. Reads `OPENAI_API_KEY` from the
    environment EXCLUSIVELY. Codex review (Phase A follow-up, finding 3):
    previously took an unused `cfg: ReviewConfig` parameter "for a stable
    call signature" -- made zero-argument instead, since `cfg.openai_api_key`
    was never consulted and a real `ReviewConfig` instance is not always in
    scope at every call site (e.g. slr_pipeline/extraction/openai_runner.py's
    low-level functions only ever see an `OpenAIExtractionConfig`, never a
    `ReviewConfig`) -- removing the unused parameter lets those sites call
    this resolver too, rather than being permitted exceptions to the "every
    call site uses the resolver" contract. A historical config file carrying
    a real (possibly committed) key in its own `openai_api_key` field can
    never be silently reactivated just because no environment variable
    happens to be set -- that field is never read here. Returns `None` when
    the environment variable is absent -- passing that `None` straight into
    `OpenAI(api_key=None)` is the correct, safe behavior: the SDK itself
    raises a clear `OpenAIError` rather than authenticating with an
    empty/stale credential."""
    return os.environ.get("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Convenience constructors for each review family
# ---------------------------------------------------------------------------

def scoping_review_config(**kwargs) -> ReviewConfig:
    """
    Default config for a scoping review.
    - No quality appraisal (scoping never runs quality_appraisal_node)
    - Charting via the extraction engine (Stage 2) -- extraction_schema/
      extraction_model_config are deliberately NOT set here; a caller sets
      them explicitly via slr_pipeline.adapters.scoping_charting_adapter
    - BERTopic synthesis + evidence map
    """
    return ReviewConfig(
        review_type="scoping",
        synthesis=SynthesisConfig(generate_evidence_map=True),
        **kwargs,
    )


def rapid_review_config(**kwargs) -> ReviewConfig:
    """
    Default config for a rapid review.
    - Reduced database set
    - Simplified quality appraisal (teaching_appraisal_v1) -- caller sets
      quality_appraisal_model_config explicitly; there is no default model
    - Claim-backed Results section (traceable) by default; legacy free-text
      Introduction/Methods/Discussion narrative is off by default -- opt in
      via synthesis=SynthesisConfig(generate_claim_backed_results=True,
      generate_legacy_imrad_narrative=True) (both flags must be set explicitly
      when overriding `synthesis=` directly, since doing so replaces this
      function's own default SynthesisConfig entirely)
    """
    quality_appraisal = kwargs.pop("quality_appraisal", QualityAppraisalConfig(checklist_id="teaching_appraisal_v1"))
    synthesis = kwargs.pop("synthesis", SynthesisConfig(generate_claim_backed_results=True))
    return ReviewConfig(
        review_type="rapid",
        search=SearchConfig(databases=["pubmed", "openalex"], max_results_per_database=500),
        quality_appraisal=quality_appraisal, synthesis=synthesis, **kwargs,
    )


def slr_config(**kwargs) -> ReviewConfig:
    """
    Default config for a full systematic review.
    - All databases
    - RoB2-inspired quality appraisal -- caller sets
      quality_appraisal_model_config explicitly; there is no default model
    - Claim-backed Results section (traceable) by default; legacy free-text
      Introduction/Methods/Discussion narrative is off by default -- opt in
      via synthesis=SynthesisConfig(generate_claim_backed_results=True,
      generate_legacy_imrad_narrative=True) (both flags must be set explicitly
      when overriding `synthesis=` directly)
    """
    quality_appraisal = kwargs.pop("quality_appraisal",
                                    QualityAppraisalConfig(checklist_id="rob2_inspired_study_level_teaching_v1"))
    synthesis = kwargs.pop("synthesis", SynthesisConfig(generate_claim_backed_results=True))
    return ReviewConfig(
        review_type="slr",
        search=SearchConfig(databases=["openalex", "pubmed", "semantic_scholar", "arxiv",
                                        "crossref", "springer", "ieee"], max_results_per_database=2000),
        quality_appraisal=quality_appraisal, synthesis=synthesis, **kwargs,
    )
