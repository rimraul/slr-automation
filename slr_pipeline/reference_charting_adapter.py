"""The Track 2 reference run's own extraction (charting) schema + model
configuration -- `track2_pipeline_readiness_plan_2026-08-30.md` §4.6,
step 11.

This is a **frozen, reproduced-verbatim** copy of
`slr_pipeline.adapters.scoping_charting_adapter.scoping_chart_schema()`'s
seven fields, plus two PRISMA-ScR item-22 funding fields
(`funding_sources`, `funder_role`) -- deliberately **not imported** from
that module. A later edit to the real scoping-charting schema (field
wording, provenance, cardinality) must never silently change what the
frozen reference run demonstrates; if the two ever need to diverge on
purpose, `REFERENCE_EXTRACTION_SCHEMA_VERSION` bumps here independently
of `SCOPING_CHART_SCHEMA_VERSION` over there.

`reference_extraction_schema()`'s dump feeds `extraction_schema_sha256`
(plan §0.4); `reference_extraction_model_config()`'s dump feeds
`extraction_model_config_sha256`. Both hash targets are registered by
`scripts/register_reference_hash_targets.py`, not here: this module is
`slr_pipeline`-wheel-packaged and must not assume it sits inside a git
checkout with `tests/` as a sibling (Track-2 packaging-boundary fix, CI run
34155102694) -- only the pure content lives here."""
from __future__ import annotations

from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.schema import (
    ExtractionFieldSpec,
    ExtractionSchema,
    FieldOrigin,
    FieldProvenance,
    ValueKind,
)

REFERENCE_EXTRACTION_SCHEMA_ID = "reference-charting-v1"
REFERENCE_EXTRACTION_SCHEMA_VERSION = "1"

# Mirrors CHAPTER_MODEL_CONFIG_ID / SCOPING_MODEL_CONFIG_ID's own naming
# convention (ft_screening_adapter.py / scoping_charting_adapter.py).
REFERENCE_MODEL_CONFIG_ID = "reference-charting-preset-v1"

_PROVENANCE = FieldProvenance(
    origin=FieldOrigin.THEORY_DERIVED, version="1",
    rationale=(
        "PRISMA-ScR / PCC charting convention plus PRISMA-ScR item 22 "
        "funding items; see the Track-2 pipeline-readiness plan."
    ),
)


def reference_extraction_schema() -> ExtractionSchema:
    """The reference run's nine-field charting schema: the six evidence-
    verified TEXT/single PCC fields, `key_findings` (TEXT_LIST/multiple),
    and two PRISMA-ScR item-22 funding fields. `author_year` is not a
    field here either (same `derive_author_year` treatment as the real
    scoping schema); `relevance_notes` is not reintroduced."""
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
        ExtractionFieldSpec(
            field_id="funding_sources",
            question="What are the study's sources of funding?",
            description=(
                "The named funders of the study, exactly as the paper states "
                "them (typically in a Funding, Acknowledgements, or "
                "Declarations section). If the paper explicitly states there "
                "was no external funding, return the single item \"No "
                "external funding\". Report only what the paper's own text "
                "states."
            ),
            value_kind=ValueKind.TEXT_LIST,
            cardinality="multiple",
            provenance=_PROVENANCE,
        ),
        ExtractionFieldSpec(
            field_id="funder_role",
            question="What role did the funder(s) play in the study?",
            description=(
                "The funder's stated role in the study design, conduct, "
                "analysis, or reporting. If the paper states the funder had "
                "no such role, return \"The funder had no role in the "
                "study.\" Report only what the paper states."
            ),
            value_kind=ValueKind.TEXT,
            cardinality="single",
            provenance=_PROVENANCE,
        ),
    )
    return ExtractionSchema(
        schema_id=REFERENCE_EXTRACTION_SCHEMA_ID,
        version=REFERENCE_EXTRACTION_SCHEMA_VERSION,
        fields=fields,
    )


def reference_extraction_model_config() -> OpenAIExtractionConfig:
    """The reference run's ONE calibrated extraction configuration --
    `gpt-5.6-terra` / `reasoning_effort="medium"` / `max_completion_
    tokens=4096`, the same model/effort pairing the chapter's other
    calibrated presets use (`chapter_screening_config()`,
    `scoping_extraction_model_config()`). `prompt_template_version` /
    `output_schema_version` take `OpenAIExtractionConfig`'s own defaults
    (`PROMPT_TEMPLATE_VERSION` / `OUTPUT_SCHEMA_VERSION`,
    `openai_runner.py:22-23`) -- not restated here."""
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra",
        model_config_id=REFERENCE_MODEL_CONFIG_ID,
        reasoning_effort="medium",
        max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )


