"""`ReferenceRunSpec` -- the strict, fully-inline configuration contract for
Track 2's pipeline-readiness rehearsal (`track2_pipeline_readiness_plan_
2026-08-30.md` §4.2, step 11).

This module also owns `ReferenceReportingContext` (§9 -- the "complete,
pre-`graph.invoke()` context" `capture_reporting_snapshot` cannot itself
build, since it only ever sees `state`/`cfg`/`run_id`) and
`reference_run_config()`, which builds a `ReviewConfig` **purely** from a
`ReferenceRunSpec` instance.

**Scope deliberately NOT covered here** (documented, not silently
dropped -- each lands in the step that actually produces the artifact it
needs, per §0.4/§13):
  - `reference_spec.json` itself is **not written** by this step.
    `dump_reference_spec()` is defined and ready, but it depends on the
    per-surface request snapshots `author_reference_cassette.py` writes
    at step 12(d), plus `fixture_corpus_digest`/`prisma_scr_matrix_sha256`
    (step 12(f)) -- calling it before those exist raises, by construction
    (every one of `ReferenceRunSpec`'s hash fields is a required,
    pattern-constrained string with no default).
  - The four **hash-target-owning** dumps this step DOES write --
    `extraction_schema.json`/`extraction_model_config.json`
    (`reference_charting_adapter.py`), `ft_screening_config.json`
    (`chapter_screening_config()`, already built), and
    `reference_reporting_context.json` (this module) -- are the four
    build outputs §0.4 commits at step 11. Their hash targets are
    registered by `scripts/register_reference_hash_targets.py`, not next
    to this pure content itself (Track-2 packaging-boundary fix, CI run
    34155102694 -- this module is wheel-packaged and must not assume a
    checkout-relative `tests/` sibling).
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, conlist, constr, model_validator

from slr_pipeline.adapters.ft_screening_adapter import chapter_screening_config
from slr_pipeline.config import (
    DedupConfig,
    ReviewConfig,
    ScreeningConfig,
    ScreeningQCConfig,
    SearchConfig,
    SearchStrategyConfig,
    SynthesisConfig,
)
from slr_pipeline.reference_charting_adapter import (
    REFERENCE_EXTRACTION_SCHEMA_ID,
    REFERENCE_EXTRACTION_SCHEMA_VERSION,
    reference_extraction_model_config,
    reference_extraction_schema,
)

REFERENCE_SPEC_VERSION = "1"

# The 7 canonical Berg2023 databases (plan §2.6) this run maps 1:1 to the
# `fx_<db>` fixture adapters (plan §2.5/§6).
_CANONICAL_DATABASES: tuple = (
    "pubmed", "web_of_science", "cinahl", "scopus", "acm",
    "engineering_village", "business_source_complete",
)

# The 8 call-making surfaces (plan §4.1/§4.3) -- no `ft_retrieval` (the FT
# retriever is the non-model `lexical_bm25`, §4.7).
REFERENCE_REQUEST_SNAPSHOT_SURFACES: tuple = (
    "query_expansion", "db_translation", "title_abstract_screening",
    "qc_second_rater", "ft_adjudication", "extraction_batch",
    "synthesis_map", "synthesis_refine",
)

# --- shared frozen literal text (single source, quoted once, reused by
#     both ReferenceRunSpec and ReferenceReportingContext below) ---
_REVIEW_TITLE = ("Pipeline-readiness rehearsal: MBSE and computer-simulation "
                 "methods for rural healthcare disaster preparedness (scoping review)")
_REVIEW_RATIONALE = (
    "This run rehearses the chapter's end-to-end scoping-review automation "
    "pipeline on a synthetic corpus with replayed model calls, to demonstrate "
    "that every methodological artifact the chapter cites is produced and "
    "internally consistent before the audited Track-3 run on real data."
)
_FUNDING_STATEMENT = "This pipeline-readiness rehearsal received no external funding."
_DATA_GOVERNANCE_DECLARATION = (
    "Track 2 pipeline-readiness rehearsal: synthetic corpus, replayed model "
    "calls, no human subjects, no external data-governance obligations."
)

# Verbatim from test_reviews/berg2023/gold_standard.json (verified 2026-08-31).
_RESEARCH_QUESTIONS = [
    "What is the current body of evidence on the application of computer simulation based on "
    "model-based systems engineering approaches to disaster preparedness and response for rural "
    "healthcare systems?",
    "What MBSE methods, healthcare settings, disaster types, and contexts have been studied in "
    "rural healthcare disaster planning?",
]
_PCC_POPULATION = "Rural healthcare systems and rural communities involved in disaster preparedness and response"
_PCC_CONCEPT = ("Application of model-based systems engineering (MBSE), including computer "
                "simulation and systems thinking, to rural healthcare disaster planning")
_PCC_CONTEXT = "Disaster preparedness and response planning in rural healthcare settings"
_INCLUSION_CRITERIA = [
    "Studies addressing computer simulation or MBSE approaches applied to disaster preparedness "
    "or response",
    "Studies focused on rural healthcare systems or rural healthcare settings",
    "Studies relevant to disaster planning, preparedness, or response",
]
_EXCLUSION_CRITERIA = [
    "Studies not involving rural healthcare systems",
    "Studies not using MBSE or computer simulation approaches",
    "Studies not related to disaster preparedness or response",
]
_SEED_QUERY = ('((rural OR non-urban OR frontier) AND (healthcare OR "health care") AND '
               '(disaster OR pandemic OR epidemic) AND (simulation OR "agent based model" '
               'OR "system dynamics" OR "machine learning" OR "artificial intelligence"))')

# Plan §4.5, frozen JSON, verbatim.
FULL_TEXT_CRITERIA_SET: Dict[str, Any] = {
    "criteria_set_id": "track2_readiness_v1",
    "version": "1",
    "criteria": [
        {"criterion_id": "mbse_or_simulation", "version": "1",
         "eligibility_requirement": "The study applies computer simulation or model-based systems engineering (MBSE) methods (discrete-event, agent-based, or system-dynamics simulation, or systems modelling) to a disaster-planning problem.",
         "missing_evidence_policy": "review", "exclusion_reason_code": "no_mbse_simulation",
         "exclusion_reason_label": "No MBSE or computer-simulation method", "priority": 1, "allowed_sources": None},
        {"criterion_id": "rural_healthcare", "version": "1",
         "eligibility_requirement": "The study concerns rural healthcare systems, rural healthcare facilities, or rural communities' healthcare, in a disaster context.",
         "missing_evidence_policy": "review", "exclusion_reason_code": "not_rural_healthcare",
         "exclusion_reason_label": "Not a rural healthcare setting", "priority": 2, "allowed_sources": None},
        {"criterion_id": "disaster_context", "version": "1",
         "eligibility_requirement": "The study addresses disaster preparedness, planning, or response (natural disaster, pandemic, epidemic, or mass-casualty event).",
         "missing_evidence_policy": "review", "exclusion_reason_code": "not_disaster_context",
         "exclusion_reason_label": "Not a disaster-preparedness or -response context", "priority": 3, "allowed_sources": None},
    ],
}

# Plan §4.2's `extraction_parser_options` Literal -- a `DoclingParserOptions`
# with both toggles off, chunk cap unchanged.
EXTRACTION_PARSER_OPTIONS: Dict[str, Any] = {
    "do_ocr": False, "do_table_structure": False, "max_chunk_chars": 3000,
}

_HEX64 = r"^[0-9a-f]{64}$"


class ReferenceRunSpec(BaseModel):
    """Plan §4.2, verbatim. Every field either a frozen `Literal` (exact
    equality, round-11 finding 1) or a build-output hash/mapping this
    module's own callers (the step-12/13 authoring scripts, the driver)
    supply once the value they hash actually exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_spec_version: Literal["1"] = "1"

    # --- review identity (PRISMA-ScR items 1, 3, 4) ---
    review_title: Literal[_REVIEW_TITLE]
    review_rationale: Literal[_REVIEW_RATIONALE]
    report_language: Literal["en"] = "en"
    research_questions: Literal[_RESEARCH_QUESTIONS]
    pcc_population: Literal[_PCC_POPULATION]
    pcc_concept: Literal[_PCC_CONCEPT]
    pcc_context: Literal[_PCC_CONTEXT]
    inclusion_criteria: Literal[_INCLUSION_CRITERIA]
    exclusion_criteria: Literal[_EXCLUSION_CRITERIA]
    seed_query: Literal[_SEED_QUERY]

    # --- protocol ---
    protocol_uri: Literal["track2/readiness_protocol.md"] = "track2/readiness_protocol.md"
    protocol_sha256: constr(pattern=_HEX64)
    protocol_registration_id: Literal["TRACK2-READINESS-REHEARSAL-NONE"] = "TRACK2-READINESS-REHEARSAL-NONE"
    ai_use_prespecified: Literal[True] = True
    protocol_deviations: Literal[[]] = []
    date_from: None = None
    date_to: None = None
    language: None = None

    # --- databases ---
    databases: Literal[list(_CANONICAL_DATABASES)]
    execution_adapter_map: Dict[str, str]

    # --- search / dedup / screening ---
    as_of_date: Literal["2026-01-01"] = "2026-01-01"
    max_results_per_database: Literal[50] = 50
    strategy_builder_enabled: Literal[True] = True
    strategy_builder_llm_model: Literal["gpt-5.4-mini"] = "gpt-5.4-mini"
    strategy_builder_max_synonyms_per_concept: Literal[5] = 5
    strategy_builder_prefer_title_abstract_fields: Literal[True] = True
    dedup_mode: Literal["exact"] = "exact"
    screening_llm_model: Literal["gpt-5.4-mini"] = "gpt-5.4-mini"
    screening_batch_size: Literal[5] = 5
    screening_workers: Literal[1] = 1
    screening_auto_include_threshold: Literal[0.80] = 0.80
    screening_uncertain_threshold: Literal[0.0] = 0.0
    qc_enabled: Literal[True] = True
    qc_second_rater_model: None = None
    qc_calibration_sample_size: Literal[6] = 6
    qc_calibration_seed: Literal[20260101] = 20260101
    qc_n_bootstrap: Literal[200] = 200
    qc_saturation_levels: Literal[[0.80, 0.90, 0.95]]

    # --- synthesis ---
    run_bertopic: Literal[False] = False
    synthesis_llm_model: Literal["gpt-4o"] = "gpt-4o"
    generate_evidence_map: Literal[True] = True
    # False, not True: `synthesis.py`'s docstring/code add a third,
    # optional "imrad:results" map/refine target only when this flag is
    # True (`synthesis.py:1188,1742`) -- a third target would make the
    # map loop issue 3 x 5 = 15 calls (and refine correspondingly more),
    # contradicting the plan's own frozen call-topology table (row 7/8:
    # "for target in requested_targets (2)" -> synthesis_map=10,
    # synthesis_refine=4) and `CORPUS_ORACLE.json`'s already-committed
    # `synthesis_map_calls: 10`/`synthesis_refine_calls: 4`/
    # `total_replay_calls: 50`. `SynthesisConfig`'s own default is also
    # `False` (`config.py`); only `build_rapid_review_config`/
    # `build_slr_review_config` opt into `True`, for review types this
    # scoping-review reference run does not use.
    generate_claim_backed_results: Literal[False] = False
    generate_legacy_imrad_narrative: Literal[False] = False
    refine_batch_size: Literal[3] = 3

    # --- extraction (charting) ---
    extraction_schema_sha256: constr(pattern=_HEX64)
    extraction_model_config_sha256: constr(pattern=_HEX64)
    extraction_max_batch_chars: Literal[12000] = 12000
    extraction_parser_options: Literal[EXTRACTION_PARSER_OPTIONS]

    # --- full text ---
    full_text_criteria_set: Dict[str, Any]
    ft_screening_config_sha256: constr(pattern=_HEX64)

    # --- per-surface request snapshots (§4.3) ---
    request_snapshot_sha256: Dict[str, str]

    # --- reporting / governance ---
    funding_statement: Literal[_FUNDING_STATEMENT]
    data_governance_declaration: Literal[_DATA_GOVERNANCE_DECLARATION]
    llm_transport: Literal["replay"] = "replay"

    # --- embedded build-output hashes ---
    reference_reporting_context_sha256: constr(pattern=_HEX64)
    fixture_corpus_digest: constr(pattern=_HEX64)
    prisma_scr_matrix_sha256: constr(pattern=_HEX64)

    @model_validator(mode="after")
    def _check_execution_adapter_map(self) -> "ReferenceRunSpec":
        expected = {db: f"fx_{db}" for db in self.databases}
        if self.execution_adapter_map != expected:
            raise ValueError(
                f"execution_adapter_map must be exactly {expected!r}, "
                f"got {self.execution_adapter_map!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_request_snapshot_keys(self) -> "ReferenceRunSpec":
        expected_keys = set(REFERENCE_REQUEST_SNAPSHOT_SURFACES)
        actual_keys = set(self.request_snapshot_sha256)
        if actual_keys != expected_keys:
            raise ValueError(
                f"request_snapshot_sha256 keys must be exactly {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )
        for surface, digest in self.request_snapshot_sha256.items():
            if not re.match(_HEX64, digest):
                raise ValueError(f"request_snapshot_sha256[{surface!r}] is not a sha256 hex digest: {digest!r}")
        return self


def reference_run_config(spec: ReferenceRunSpec) -> ReviewConfig:
    """Builds a `ReviewConfig` **purely** from `spec` -- no other input,
    no silently-chosen default beyond what `ReviewConfig`'s own
    sub-configs already default (plan §4.2). Track 2 leaves
    `full_text_screening_config` unset (§4.7 -- `_run_full_text_
    eligibility` falls back to `chapter_screening_config()`)."""
    return ReviewConfig(
        review_type="scoping",
        title=spec.review_title,
        research_questions=list(spec.research_questions),
        population=spec.pcc_population,
        concept=spec.pcc_concept,
        context=spec.pcc_context,
        inclusion_criteria=list(spec.inclusion_criteria),
        exclusion_criteria=list(spec.exclusion_criteria),
        full_text_criteria_set=spec.full_text_criteria_set,
        extraction_schema=reference_extraction_schema().model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(reference_extraction_model_config()),
        extraction_parser_options=dict(spec.extraction_parser_options),
        extraction_max_batch_chars=spec.extraction_max_batch_chars,
        protocol_uri=spec.protocol_uri,
        protocol_hash=spec.protocol_sha256,
        protocol_registration_id=spec.protocol_registration_id,
        ai_use_prespecified=spec.ai_use_prespecified,
        protocol_deviations=list(spec.protocol_deviations),
        data_governance_declaration=spec.data_governance_declaration,
        funding_statement=spec.funding_statement,
        reference_reporting_context=reference_reporting_context().model_dump(mode="json"),
        search=SearchConfig(
            databases=list(spec.databases),
            max_results_per_database=spec.max_results_per_database,
            query=spec.seed_query,
            date_from=spec.date_from,
            date_to=spec.date_to,
            language=spec.language,
            strategy_builder=SearchStrategyConfig(
                enabled=spec.strategy_builder_enabled,
                llm_model=spec.strategy_builder_llm_model,
                max_synonyms_per_concept=spec.strategy_builder_max_synonyms_per_concept,
                prefer_title_abstract_fields=spec.strategy_builder_prefer_title_abstract_fields,
            ),
            execution_adapter_map=dict(spec.execution_adapter_map),
            as_of_date=spec.as_of_date,
        ),
        dedup=DedupConfig(mode=spec.dedup_mode),
        screening=ScreeningConfig(
            llm_model=spec.screening_llm_model,
            auto_include_threshold=spec.screening_auto_include_threshold,
            uncertain_threshold=spec.screening_uncertain_threshold,
            batch_size=spec.screening_batch_size,
            workers=spec.screening_workers,
        ),
        screening_qc=ScreeningQCConfig(
            enabled=spec.qc_enabled,
            calibration_sample_size=spec.qc_calibration_sample_size,
            second_rater_model=spec.qc_second_rater_model,
            n_bootstrap=spec.qc_n_bootstrap,
            saturation_levels=list(spec.qc_saturation_levels),
            calibration_seed=spec.qc_calibration_seed,
        ),
        synthesis=SynthesisConfig(
            run_bertopic=spec.run_bertopic,
            llm_model=spec.synthesis_llm_model,
            generate_evidence_map=spec.generate_evidence_map,
            generate_claim_backed_results=spec.generate_claim_backed_results,
            generate_legacy_imrad_narrative=spec.generate_legacy_imrad_narrative,
            refine_batch_size=spec.refine_batch_size,
        ),
    )


def dump_reference_spec(spec: ReferenceRunSpec, path: Path) -> str:
    """Writes `spec` as canonical JSON (`indent=2, sort_keys=True`,
    trailing newline -- the same convention `scientific_projection.json`
    established at step 8) and returns the written text. **Not invoked to
    produce `reference_spec.json` at step 11** -- see this module's own
    docstring for why (step 12 dependency)."""
    text = json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# ReferenceReportingContext -- plan §9
# ---------------------------------------------------------------------------

class ExtractionFieldItem(BaseModel):
    """One of the 9 charting fields (§4.6)."""
    model_config = ConfigDict(extra="forbid")
    field_id: str
    question: str


class ReferenceReportingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_reporting_context_version: Literal["1"] = "1"
    review_title: str
    review_rationale: str
    report_language: Literal["en"]
    date_from_declaration: Literal["No publication-date limit was applied."]
    date_to_declaration: Literal["No publication-date limit was applied."]
    language_declaration: Literal["No language restriction was applied."]
    review_funding_statement: str
    protocol_registration_id: Literal["TRACK2-READINESS-REHEARSAL-NONE"]
    ai_use_prespecified: Literal[True]
    extraction_schema_id: Literal["reference-charting-v1"]
    extraction_schema_version: Literal["1"]
    extraction_model: Literal["gpt-5.6-terra"]
    extraction_reasoning_effort: Literal["medium"]
    extraction_field_items: conlist(ExtractionFieldItem, min_length=9, max_length=9)
    charting_verification_policy: Literal[
        "Every extracted value is evidence-verified against a "
        "quoted, page-located span; unclear/conflicting fields go "
        "to a human review queue (empty this run)."
    ]
    llm_transport: Literal["replay"]
    search_source: Literal["fixture"]
    bertopic: Literal["not_run"]
    dedup_mode: Literal["exact"]
    non_instrumented_stages: Literal[["extraction", "quality_appraisal", "synthesis_bertopic"]]
    full_text_eligibility_provenance_note: Literal[
        "full_text_eligibility is instrumentation_status='instrumented' with "
        "event_count=0 (ai_traceability_report.py's _STAGE_EVENT_STAGES maps it to "
        "an empty AIUseEvent set) -- it is not one of the non_instrumented_stages "
        "above. Its evidence trail is FT_SCREENING's own byte-hashed "
        "ft_native_provenance block, not an AIUseEvent."
    ]
    synthetic_corpus_caveat: Literal[
        "This run used a synthetic corpus and replayed model calls; "
        "it demonstrates pipeline readiness, not a scientific finding. "
        "The audited run on real data is Track 3."
    ]


def reference_reporting_context() -> ReferenceReportingContext:
    """Builds the ONE reference-run reporting context (plan §9). Its
    `extraction_field_items` are read from `reference_extraction_schema()`
    itself (field_id + question only) rather than re-typed a third time,
    so a schema edit here can never silently drift from what
    `prisma_scr_report.md`'s item 11 renders."""
    field_items = [
        ExtractionFieldItem(field_id=f.field_id, question=f.question)
        for f in reference_extraction_schema().fields
    ]
    return ReferenceReportingContext(
        review_title=_REVIEW_TITLE,
        review_rationale=_REVIEW_RATIONALE,
        report_language="en",
        date_from_declaration="No publication-date limit was applied.",
        date_to_declaration="No publication-date limit was applied.",
        language_declaration="No language restriction was applied.",
        review_funding_statement=_FUNDING_STATEMENT,
        protocol_registration_id="TRACK2-READINESS-REHEARSAL-NONE",
        ai_use_prespecified=True,
        extraction_schema_id=REFERENCE_EXTRACTION_SCHEMA_ID,
        extraction_schema_version=REFERENCE_EXTRACTION_SCHEMA_VERSION,
        extraction_model="gpt-5.6-terra",
        extraction_reasoning_effort="medium",
        extraction_field_items=field_items,
        charting_verification_policy=(
            "Every extracted value is evidence-verified against a "
            "quoted, page-located span; unclear/conflicting fields go "
            "to a human review queue (empty this run)."
        ),
        llm_transport="replay",
        search_source="fixture",
        bertopic="not_run",
        dedup_mode="exact",
        non_instrumented_stages=["extraction", "quality_appraisal", "synthesis_bertopic"],
        full_text_eligibility_provenance_note=(
            "full_text_eligibility is instrumentation_status='instrumented' with "
            "event_count=0 (ai_traceability_report.py's _STAGE_EVENT_STAGES maps it to "
            "an empty AIUseEvent set) -- it is not one of the non_instrumented_stages "
            "above. Its evidence trail is FT_SCREENING's own byte-hashed "
            "ft_native_provenance block, not an AIUseEvent."
        ),
        synthetic_corpus_caveat=(
            "This run used a synthetic corpus and replayed model calls; "
            "it demonstrates pipeline readiness, not a scientific finding. "
            "The audited run on real data is Track 3."
        ),
    )


# ---------------------------------------------------------------------------
# `FT_SCREENING_CONFIG_FIXTURE`/`REFERENCE_REPORTING_CONTEXT_FIXTURE`/
# `READINESS_PROTOCOL_PATH` and their hash-target registrations (step 11/
# 12(b), plan §0.4) moved to `scripts/register_reference_hash_targets.py`:
# this module is `slr_pipeline`-wheel-packaged and must not assume it sits
# inside a git checkout with `tests/`/`track2/` as siblings (Track-2
# packaging-boundary fix, CI run 34155102694). `dump_ft_screening_config_
# fixture()`/`dump_reference_reporting_context_fixture()` below still live
# here (the pure serialization convention belongs next to `chapter_
# screening_config()`/`reference_reporting_context()`), but now take the
# destination `path` explicitly -- callers (e.g. `scripts/register_
# reference_hash_targets.py`, which owns the real fixture path) supply it,
# rather than this module assuming a wheel-sibling `tests/` directory.
# ---------------------------------------------------------------------------


def dump_ft_screening_config_fixture(path: Path) -> str:
    """Writes `chapter_screening_config().model_dump()` to `path` (plan
    §4.7 -- "the entire ... `model_dump()` is captured"). `reference_
    preflight` asserts the live value still equals this file byte-for-byte
    before a reference run."""
    text = json.dumps(chapter_screening_config().model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def dump_reference_reporting_context_fixture(path: Path) -> str:
    text = json.dumps(reference_reporting_context().model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
