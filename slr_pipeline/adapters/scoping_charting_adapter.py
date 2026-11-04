"""slr_pipeline <-> structured-extraction adapter for scoping-review charting.

Stage 2 of quality_appraisal_synthesis_plan_2026-08-22.md ("Charting via
the Extraction Engine"): scoping's charting is re-platformed onto the
already-frozen, already-nine-times-reviewed `extraction_node`/Phase A
engine (`slr_pipeline/extraction/*`) rather than reinventing a third,
independently-durable artifact system on top of `nodes/charting.py`'s own
abstract-fallback-prone pydantic/chunking methods (moved to
`examples/legacy/charting.py`; see graphs/scoping.py).

This module maps `nodes/charting.py`'s old nine-field `ScopingChartFields`
onto SEVEN methodologically distinct extraction fields (plan Section 2.2):
  - `author_year` is NOT an extraction field at all -- derived
    deterministically from already-structured bibliographic metadata
    (`derive_author_year`), never sent through the LLM/evidence-
    verification path.
  - `country`, `study_design`, `population`, `concept`, `context`,
    `limitations` are six real, evidence-verified TEXT/single fields.
  - `key_findings` is a TEXT_LIST/multiple field: several atomic,
    independently evidence-backed findings (one quotation+locator per
    finding), not one unverifiable free-text paragraph.
  - `relevance_notes` (the old schema's ninth field) is REMOVED from the
    extraction schema entirely -- it is reviewer interpretation produced
    AFTER extraction (why a study matters to THIS review's own research
    questions), not a fact the article itself states, and is not built by
    this module at all.

Adapter location follows `ft_screening_adapter.py`'s own precedent:
`config.py` is deliberately kept free of `extraction`-package imports (it
is sometimes imported as a bare, standalone module), so this glue code
lives here instead. `scoping_review_config()` in config.py sets NO
`extraction_schema`/`extraction_model_config` default of its own -- the
same "no silently-mirrored default" discipline `rapid_review_config()`/
`slr_config()` already follow for `extraction_model_config`. A caller
invokes `scoping_chart_schema()`/`scoping_extraction_model_config()`
explicitly, exactly like `ft_screening_adapter.py`'s own
`chapter_screening_config()`.
"""

from __future__ import annotations

from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.schema import (
    ExtractionFieldSpec,
    ExtractionSchema,
    FieldOrigin,
    FieldProvenance,
    ValueKind,
)
from slr_pipeline.state import Record

__all__ = [
    "SCOPING_CHART_SCHEMA_ID",
    "SCOPING_MODEL_CONFIG_ID",
    "derive_author_year",
    "scoping_chart_schema",
    "scoping_extraction_model_config",
]

# Versioned so a future revision to the field set/wording is itself a new,
# distinguishable schema_id -- never a silent redefinition of what "v1"
# meant (same discipline ExtractionSchema.schema_id/version already
# requires of every caller).
SCOPING_CHART_SCHEMA_ID = "scoping-charting-v1"
SCOPING_CHART_SCHEMA_VERSION = "1"

# Mirrors CHAPTER_MODEL_CONFIG_ID's own naming convention in
# ft_screening_adapter.py.
SCOPING_MODEL_CONFIG_ID = "slr-chapter-scoping-charting-preset-v1"

_PROVENANCE = FieldProvenance(
    origin=FieldOrigin.THEORY_DERIVED, version="1",
    rationale=(
        "Field set adapted from PRISMA-ScR / Arksey and O'Malley's PCC "
        "(Population, Concept, Context) scoping-review charting "
        "convention -- see quality_appraisal_synthesis_plan_2026-08-22.md "
        "Section 2.2."
    ),
)


def scoping_chart_schema() -> ExtractionSchema:
    """The chapter's one scoping-review charting schema: six evidence-
    verified TEXT/single fields plus `key_findings` as a TEXT_LIST/
    multiple field. `author_year`/`relevance_notes` are deliberately NOT
    here -- see this module's own docstring."""
    fields = (
        ExtractionFieldSpec(
            field_id="country",
            question="In which country or countries was the study conducted?",
            description=(
                "The country or countries where the study's data collection took "
                "place (the study SITE, not the authors' institutional affiliation "
                "when the two differ). Report exactly what the paper states; do not "
                "infer a country from an author's affiliation or language alone."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="study_design",
            question="What is the study's design or type?",
            description=(
                "The methodological design or type of study reported (e.g. "
                "randomised controlled trial, qualitative case study, cross-"
                "sectional survey, systematic review, conceptual/discussion "
                "paper). Use the paper's own stated design/terminology where "
                "given; do not infer a design category the paper never names."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="population",
            question="What population or sample does the study describe?",
            description=(
                "The Population element of the review's PCC framework: who or "
                "what the study is about (e.g. participant characteristics, "
                "sample size, inclusion criteria for participants) as stated in "
                "the paper."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="concept",
            question="What is the key concept the study addresses?",
            description=(
                "The Concept element of the review's PCC framework: the core "
                "idea, intervention, phenomenon, or practice the study "
                "investigates, as the paper itself frames it."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="context",
            question="What setting or context does the study describe?",
            description=(
                "The Context element of the review's PCC framework: the "
                "setting in which the study/intervention/phenomenon took place "
                "(e.g. clinical, educational, organisational, cultural, "
                "geographic setting), as stated in the paper."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="limitations",
            question="What limitations does the study's own text state?",
            description=(
                "Limitations the study's AUTHORS themselves state (typically in "
                "a Limitations/Discussion section) -- never a reviewer's own "
                "critique of the study; only limitations explicitly acknowledged "
                "in the paper's own text."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="key_findings",
            question="What are the study's key findings?",
            description=(
                "The study's own key findings/results, reported as separate, "
                "atomic statements -- one finding per list entry, each "
                "independently traceable to its own quoted, page-located "
                "evidence, rather than one unverifiable free-text paragraph. "
                "Report only findings the paper's Results/Findings/Discussion "
                "text actually states, not a reviewer's interpretation of their "
                "relevance."
            ),
            value_kind=ValueKind.TEXT_LIST,
            cardinality="multiple",
            provenance=_PROVENANCE,
        ),
    )
    return ExtractionSchema(
        schema_id=SCOPING_CHART_SCHEMA_ID, version=SCOPING_CHART_SCHEMA_VERSION, fields=fields,
    )


def scoping_extraction_model_config() -> OpenAIExtractionConfig:
    """The chapter's ONE calibrated teaching-preset model configuration for
    scoping-review charting -- takes no parameters on purpose, the same
    "every value is fixed and named" discipline
    `ft_screening_adapter.chapter_screening_config()` established: a
    reader following the book's examples reproduces this configuration
    exactly, and a different configuration is a different, explicitly
    named preset, never an optional argument to this one. Uses the same
    `gpt-5.6-terra` / `reasoning_effort="medium"` pairing as the chapter's
    full-text-screening preset, for the same reason -- one calibrated
    model identity the book's worked examples are validated under, not an
    arbitrary example value."""
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra",
        model_config_id=SCOPING_MODEL_CONFIG_ID,
        reasoning_effort="medium",
        max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )


def derive_author_year(record: Record) -> str:
    """Deterministic, non-LLM, non-evidence-verified derivation of the old
    `ScopingChartFields.author_year` field from already-structured
    bibliographic metadata the pipeline has from search/dedup, long
    before any PDF is parsed -- never an extraction field (plan Section
    2.2). Mirrors `nodes/synthesis.py`'s own `_citation_for()` fallback
    convention exactly, so a citation style already established elsewhere
    in this pipeline is not reinvented here."""
    if record.authors:
        return f"{record.authors[0]} et al., {record.year or 'n.d.'}"
    if record.title:
        return f"{record.title[:40]}, {record.year or 'n.d.'}"
    return f"Unknown author, {record.year or 'n.d.'}"
