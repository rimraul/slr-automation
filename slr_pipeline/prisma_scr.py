"""PRISMA-ScR (2018) checklist -- the frozen 22-row matrix (Track 2 plan §9)
plus the two renderers step 14 ([C13]) adds to `slr_pipeline.nodes.reporting`:
`render_prisma_scr_report_md()` (the human-readable `prisma_scr_report.md`)
and `render_prisma_scr_conformance()` (the machine-readable
`prisma_scr_conformance.json`).

`ITEMS`/`PRISMA_SCR_VERSION`/`PRISMA_SCR_MATRIX_VERSION` are the SINGLE
canonical source for the matrix -- `scripts/author_prisma_scr_matrix.py`
imports them from here rather than keeping its own copy (the same
anti-duplication move step 13 made for `reference_pdf_placement.py`, after a
literal drifting between two call sites there caused a real incident).
`tests/reference_run/fixtures/prisma_scr_matrix.json` is the frozen,
byte-checked dump of `matrix_payload()`.

Both renderers are gated by their CALLER (`reporting.py`'s
`ARTIFACT_RENDERERS`, `lambda s: s.reference_context is not None`) -- this
module never checks that itself, matching every other renderer in
`reporting.py` (the applicability check lives in the dispatch table, not in
the render function). `reference_context` is `None` in every production run
(only the Track 2 reference run sets `cfg.reference_reporting_context`), so
these renderers are exercised only there.

No import of `slr_pipeline.nodes.reporting` at module load time -- that
module imports THIS one (for `ITEMS`/the two renderers), so the reverse
import would be circular. `ReportingSourceSnapshot` is used only as a type
hint, under `TYPE_CHECKING`.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from slr_pipeline.nodes.reporting import ReportingSourceSnapshot

PRISMA_SCR_VERSION = "2018"
PRISMA_SCR_MATRIX_VERSION = "1"

# Plan §9's 22-row table, transcribed verbatim (column order: #, Section,
# Short title, Tier, anchor, run-state inputs, status rule). `tier` uses the
# plan's own two literal tokens, "ess"/"opt", rather than expanded words, so
# the byte-equality check against `prisma_scr_matrix.json` has one
# unambiguous vocabulary to match, not two.
ITEMS: List[Dict[str, Any]] = [
    {
        "item_id": 1, "section": "TITLE", "short_title": "Identify as a scoping review",
        "tier": "ess", "anchor": "title",
        "run_state_inputs": "spec.review_title",
        "status_rule": 'non-empty, contains "scoping review"',
    },
    {
        "item_id": 2, "section": "ABSTRACT", "short_title": "Structured summary",
        "tier": "ess", "anchor": "abstract",
        "run_state_inputs": "prisma_counts; synthesis.rq_synthesis; RQs",
        "status_rule": "all six sub-parts render",
    },
    {
        "item_id": 3, "section": "INTRO", "short_title": "Rationale",
        "tier": "ess", "anchor": "rationale",
        "run_state_inputs": "spec.review_rationale",
        "status_rule": "non-empty",
    },
    {
        "item_id": 4, "section": "INTRO", "short_title": "Objectives",
        "tier": "ess", "anchor": "objectives",
        "run_state_inputs": "RQs; PCC",
        "status_rule": "RQs + PCC render",
    },
    {
        "item_id": 5, "section": "METHODS", "short_title": "Protocol and registration",
        "tier": "ess", "anchor": "methods.protocol",
        "run_state_inputs": (
            "protocol_uri/protocol_sha256; ai_use_prespecified; "
            "protocol_registration_id; protocol_deviations"
        ),
        "status_rule": 'ref + prespec + registration id + "no deviations" render',
    },
    {
        "item_id": 6, "section": "METHODS", "short_title": "Eligibility criteria",
        "tier": "ess", "anchor": "methods.eligibility",
        "run_state_inputs": (
            "inclusion/exclusion; date_from/date_to None -> \"no date limit\"; "
            "language None -> \"no language restriction\""
        ),
        "status_rule": 'all six criteria + both "no limit" statements render',
    },
    {
        "item_id": 7, "section": "METHODS", "short_title": "Information sources",
        "tier": "ess", "anchor": "methods.sources",
        "run_state_inputs": "the seven canonical databases; as_of_date",
        "status_rule": "the seven names + the date render",
    },
    {
        "item_id": 8, "section": "METHODS", "short_title": "Search",
        "tier": "ess", "anchor": "methods.search",
        "run_state_inputs": "seed_query; search_strategy.db_queries",
        "status_rule": "seed query + >=1 full per-DB translation render",
    },
    {
        "item_id": 9, "section": "METHODS", "short_title": "Selection of sources of evidence",
        "tier": "ess", "anchor": "methods.selection",
        "run_state_inputs": (
            "screening + full-text decisions; uncertain_threshold=0.0; "
            '"single automated reviewer, replayed model calls, no second human"'
        ),
        "status_rule": "the two-stage process + the automation statement render",
    },
    {
        "item_id": 10, "section": "METHODS", "short_title": "Data charting process",
        "tier": "ess", "anchor": "methods.charting",
        "run_state_inputs": (
            "reference-charting-v1 schema id/version; gpt-5.6-terra / "
            "reasoning_effort=medium; verification + review-queue policy"
        ),
        "status_rule": "schema ref + model + policy render",
    },
    {
        "item_id": 11, "section": "METHODS", "short_title": "Data items",
        "tier": "ess", "anchor": "methods.data_items",
        "run_state_inputs": "the nine reference_extraction_schema() field ids + questions",
        "status_rule": "every field id + question renders",
    },
    {
        "item_id": 12, "section": "METHODS",
        "short_title": "Critical appraisal of individual sources",
        "tier": "opt", "anchor": "methods.appraisal",
        "run_state_inputs": "—",
        "status_rule": (
            "optional_not_conducted; rationale: \"Critical appraisal of individual "
            "sources is optional for scoping reviews (PRISMA-ScR items 12 and 16) "
            'and was not performed in this pipeline-readiness run."'
        ),
    },
    {
        "item_id": 13, "section": "METHODS", "short_title": "Synthesis of results",
        "tier": "ess", "anchor": "methods.synthesis",
        "run_state_inputs": (
            "generate_evidence_map=True; generate_claim_backed_results=False "
            "(map/refine over the 2 RQs is unconditional; this flag would add a "
            "third, optional \"imrad:results\" target -- off here, matching the "
            "frozen 10/4/50 call topology); refine_batch_size=3; run_bertopic=False"
        ),
        "status_rule": (
            'evidence-map + claim-backed map/refine method + "no BERTopic '
            'topic modelling" render'
        ),
    },
    {
        "item_id": 14, "section": "RESULTS", "short_title": "Selection of sources of evidence",
        "tier": "ess", "anchor": "results.selection",
        "run_state_inputs": "prisma_counts (full S12 flow); ref to prisma_flow.png",
        "status_rule": (
            "every S12 flow count renders and prisma_flow raises no "
            "PrismaInconsistencyError and the counts are consistent with "
            "full_text_review_queue == []"
        ),
    },
    {
        "item_id": 15, "section": "RESULTS", "short_title": "Characteristics of sources",
        "tier": "ess", "anchor": "results.characteristics",
        "run_state_inputs": "synthesis.evidence_map; charted_data",
        "status_rule": "one characteristics row per eligible source (5)",
    },
    {
        "item_id": 16, "section": "RESULTS", "short_title": "Critical appraisal within sources",
        "tier": "opt", "anchor": "results.appraisal",
        "run_state_inputs": "—",
        "status_rule": "optional_not_conducted; same rationale as 12",
    },
    {
        "item_id": 17, "section": "RESULTS", "short_title": "Results of individual sources",
        "tier": "ess", "anchor": "results.individual",
        "run_state_inputs": "per-study charted fields; per-study map_answers",
        "status_rule": "one per-study block per eligible source (5)",
    },
    {
        "item_id": 18, "section": "RESULTS", "short_title": "Synthesis of results",
        "tier": "ess", "anchor": "results.synthesis",
        "run_state_inputs": "synthesis.rq_synthesis; cited study ids per RQ",
        "status_rule": "a narrative per RQ with its per-study citations",
    },
    {
        "item_id": 19, "section": "DISCUSSION", "short_title": "Summary of evidence",
        "tier": "ess", "anchor": "discussion.summary",
        "run_state_inputs": "synthesis; evidence_map; RQs",
        "status_rule": "a summary paragraph per RQ",
    },
    {
        "item_id": 20, "section": "DISCUSSION", "short_title": "Limitations",
        "tier": "ess", "anchor": "discussion.limitations",
        "run_state_inputs": (
            'non-instrumented stages (extraction, quality_appraisal, synthesis_bertopic '
            '-- from ai_traceability.json, instrumentation_status="not_instrumented"); '
            "the full_text_eligibility provenance note (instrumentation_status="
            '"instrumented", event_count=0, evidenced by its own ft_native_provenance '
            "block, not an AIUseEvent); \"BERTopic not run\"; the synthetic-corpus "
            'caveat; dedup_mode="exact"; "replayed model calls, not a live run"'
        ),
        "status_rule": "all named limitations render",
    },
    {
        "item_id": 21, "section": "DISCUSSION", "short_title": "Conclusions",
        "tier": "ess", "anchor": "discussion.conclusions",
        "run_state_inputs": (
            'synthesis; objectives; "this run demonstrates pipeline readiness; '
            'the audited scientific run is Track 3"'
        ),
        "status_rule": "a conclusions paragraph renders",
    },
    {
        "item_id": 22, "section": "FUNDING", "short_title": "Funding",
        "tier": "ess", "anchor": "funding",
        "run_state_inputs": (
            "ReviewConfig.funding_statement; charted funding_sources + "
            "funder_role per eligible source (5)"
        ),
        "status_rule": (
            "the review-level statement renders and a funding + funder-role "
            "line per eligible source"
        ),
    },
]

assert len(ITEMS) == 22, f"expected 22 PRISMA-ScR items, got {len(ITEMS)}"
assert [row["item_id"] for row in ITEMS] == list(range(1, 23))
ESSENTIAL_ITEM_IDS = frozenset(range(1, 23)) - {12, 16}
assert {row["item_id"] for row in ITEMS if row["tier"] == "ess"} == ESSENTIAL_ITEM_IDS, (
    "essential/optional split must be exactly 'every item except 12 and 16' (plan §9)"
)
OPTIONAL_ITEM_IDS = frozenset({12, 16})

_OPTIONAL_APPRAISAL_RATIONALE = (
    "Critical appraisal of individual sources is optional for scoping reviews "
    "(PRISMA-ScR items 12 and 16) and was not performed in this pipeline-readiness run."
)


def matrix_payload() -> Dict[str, Any]:
    return {
        "prisma_scr_matrix_version": PRISMA_SCR_MATRIX_VERSION,
        "prisma_scr_version": PRISMA_SCR_VERSION,
        "items": ITEMS,
    }


def matrix_text() -> str:
    return json.dumps(matrix_payload(), indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _anchor_heading(item: Dict[str, Any]) -> str:
    return f'## {item["item_id"]}. {item["short_title"]}\n<a id="{item["anchor"]}"></a>\n'


def _item_1_title(snapshot: "ReportingSourceSnapshot") -> str:
    return f"**Title:** {snapshot.reference_context.review_title}"


def _item_2_abstract(snapshot: "ReportingSourceSnapshot") -> str:
    ctx = snapshot.reference_context
    cfg = snapshot.config_projection
    prisma = snapshot.prisma.model
    output = snapshot.synthesis_output.model
    rq_lines = "\n".join(f"- {rq}: {output.rq_synthesis.get(rq, '(no synthesis narrative)')}"
                          for rq in cfg.research_questions)
    return (
        f"**Objectives:** {ctx.review_rationale}\n\n"
        f"**Eligibility criteria:** {len(cfg.inclusion_criteria)} inclusion, "
        f"{len(cfg.exclusion_criteria)} exclusion criterion/criteria (see item 6).\n\n"
        f"**Sources of evidence:** {len(cfg.search_databases)} database(s) searched "
        f"(item 7); {prisma.total_identified} record(s) identified.\n\n"
        f"**Charting methods:** structured, evidence-verified extraction (item 10).\n\n"
        f"**Results:** {prisma.included} source(s) included; findings per research "
        f"question:\n{rq_lines}\n\n"
        f"**Conclusions:** see item 21."
    )


def _item_3_rationale(snapshot: "ReportingSourceSnapshot") -> str:
    return snapshot.reference_context.review_rationale


def _item_4_objectives(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    rq_lines = "\n".join(f"- {rq}" for rq in cfg.research_questions)
    pcc = (
        f"- Population: {cfg.population or 'Not specified'}\n"
        f"- Concept: {cfg.concept or 'Not specified'}\n"
        f"- Context: {cfg.context or 'Not specified'}"
    )
    return f"**Research questions:**\n{rq_lines}\n\n**Population-Concept-Context:**\n{pcc}"


def _item_5_protocol(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    return (
        f"**Protocol:** {cfg.protocol_uri or 'Not registered'} "
        f"(hash: {cfg.protocol_hash or 'N/A'})\n\n"
        f"**Registration id:** {cfg.protocol_registration_id or 'N/A'}\n\n"
        f"**AI use pre-specified:** {cfg.ai_use_prespecified}\n\n"
        f"**Protocol deviations:** "
        f"{'; '.join(cfg.protocol_deviations) if cfg.protocol_deviations else 'None.'}"
    )


def _item_6_eligibility(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    ctx = snapshot.reference_context
    inc = "\n".join(f"- {c}" for c in cfg.inclusion_criteria)
    exc = "\n".join(f"- {c}" for c in cfg.exclusion_criteria)
    return (
        f"**Inclusion criteria:**\n{inc}\n\n"
        f"**Exclusion criteria:**\n{exc}\n\n"
        f"**Publication date limit:** {ctx.date_from_declaration}\n\n"
        f"**Publication date limit (to):** {ctx.date_to_declaration}\n\n"
        f"**Language restriction:** {ctx.language_declaration}"
    )


def _item_7_sources(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    db_lines = "\n".join(f"- {db}" for db in cfg.search_databases)
    return f"**Databases searched:**\n{db_lines}\n\n**As-of date:** {cfg.search_as_of_date}"


def _item_8_search(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    entries = snapshot.search_log.model.entries
    translations = "\n".join(f"- **{e.database}** ({e.date}): `{e.query}`" for e in entries)
    return f"**Seed query:** `{cfg.search_query}`\n\n**Per-database translations:**\n{translations}"


def _item_9_selection(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    prisma = snapshot.prisma.model
    ctx = snapshot.reference_context
    return (
        f"**Title/abstract screening:** {prisma.screened_title_abstract} screened, "
        f"{prisma.excluded_title_abstract} excluded.\n\n"
        f"**Full-text assessment:** {prisma.assessed_full_text} assessed, "
        f"{prisma.excluded_full_text} excluded.\n\n"
        f"**Uncertain threshold:** {cfg.screening_uncertain_threshold}\n\n"
        f"**Reviewer model:** single automated reviewer ({ctx.llm_transport} model calls), "
        "no second human reviewer this run."
    )


def _item_10_charting(snapshot: "ReportingSourceSnapshot") -> str:
    ctx = snapshot.reference_context
    return (
        f"**Extraction schema:** {ctx.extraction_schema_id} v{ctx.extraction_schema_version}\n\n"
        f"**Model:** {ctx.extraction_model} (reasoning effort: {ctx.extraction_reasoning_effort})\n\n"
        f"**Verification policy:** {ctx.charting_verification_policy}"
    )


def _item_11_data_items(snapshot: "ReportingSourceSnapshot") -> str:
    ctx = snapshot.reference_context
    lines = "\n".join(f"- **{fi.field_id}**: {fi.question}" for fi in ctx.extraction_field_items)
    return f"**Charted fields:**\n{lines}"


def _item_12_appraisal_not_conducted(snapshot: "ReportingSourceSnapshot") -> str:
    return f"_Not conducted._ {_OPTIONAL_APPRAISAL_RATIONALE}"


def _item_13_synthesis_methods(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    bertopic_note = (
        "BERTopic topic modelling was run this run."
        if cfg.run_bertopic
        else "BERTopic topic modelling was NOT run this run."
    )
    return (
        f"**Evidence map generated:** {cfg.generate_evidence_map}\n\n"
        f"**Claim-backed IMRAD results:** {cfg.generate_claim_backed_results}\n\n"
        f"**Refine batch size:** {cfg.refine_batch_size}\n\n"
        f"{bertopic_note}"
    )


def _item_14_selection_results(snapshot: "ReportingSourceSnapshot") -> str:
    p = snapshot.prisma.model
    queue = snapshot.full_text_review_queue.model or []
    return (
        f"| Stage | n |\n|---|---|\n"
        f"| Identified | {p.total_identified} |\n"
        f"| After deduplication | {p.after_dedup} |\n"
        f"| Screened | {p.screened_title_abstract} |\n"
        f"| Excluded (title/abstract) | {p.excluded_title_abstract} |\n"
        f"| Full-text assessed | {p.assessed_full_text} |\n"
        f"| Excluded (full-text) | {p.excluded_full_text} |\n"
        f"| Included | {p.included} |\n\n"
        f"Full-text eligibility review queue: {len(queue)} unresolved record(s). "
        f"See `prisma_flow.png` for the complete PRISMA 2020 flow diagram."
    )


def _item_15_characteristics(snapshot: "ReportingSourceSnapshot") -> str:
    em = snapshot.evidence_map
    if em is None:
        return "No eligible sources this run -- no characteristics table to render."
    rows = "\n".join(
        f"| {r.citation} | {r.year if r.year is not None else ''} | {r.country or ''} | "
        f"{r.study_design or ''} |"
        for r in em.characteristics_table
    )
    return f"| Citation | Year | Country | Study design |\n|---|---|---|---|\n{rows}"


def _item_16_appraisal_within_not_conducted(snapshot: "ReportingSourceSnapshot") -> str:
    return f"_Not conducted._ {_OPTIONAL_APPRAISAL_RATIONALE}"


_CHARTED_FIELD_ORDER = (
    "study_design", "population", "concept", "context", "limitations", "key_findings",
)


def _charted_by_id(snapshot: "ReportingSourceSnapshot") -> Dict[str, Dict[str, Any]]:
    """Single shared reconstruction of `{record_id: raw_extraction}` from
    `extraction.raw["resolved_projection"]` -- the SAME pattern already used
    at `reporting.py`'s own `_reconstruct_expected_evidence_map()` -- shared
    between items 17 and 22 rather than duplicated per-renderer."""
    charted_by_id: Dict[str, Dict[str, Any]] = {}
    if snapshot.extraction is not None:
        for row in snapshot.extraction.raw.get("resolved_projection", []):
            charted_by_id[row["record_id"]] = row.get("raw_extraction", {})
    return charted_by_id


def _format_field_value(value: Any) -> str:
    """`reference_extraction_schema()` fields are a mix of `ValueKind.TEXT`
    (a plain string) and `ValueKind.TEXT_LIST` (`key_findings`/
    `funding_sources`, a `List[str]`), and any field may be `None` when
    charting found nothing -- interpolating the raw Python value directly
    would render a list repr or the literal string 'None' into the report."""
    if value is None:
        return "Not reported"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) if value else "Not reported"
    return str(value)


def _item_17_individual_results(snapshot: "ReportingSourceSnapshot") -> str:
    output = snapshot.synthesis_output.model
    charted_by_id = _charted_by_id(snapshot)
    answers_by_record: Dict[str, List] = {}
    for a in output.map_answers:
        answers_by_record.setdefault(a.record_id, []).append(a)

    blocks = []
    for record_id in output.synthesis_record_ids:
        fields = charted_by_id.get(record_id, {})
        field_lines = "\n".join(
            f"  - {f}: {_format_field_value(fields[f])}" for f in _CHARTED_FIELD_ORDER if f in fields
        )
        answers = answers_by_record.get(record_id, [])
        answer_lines = "\n".join(f"  - [{a.rq_or_section}] {a.answer_text}" for a in answers)
        blocks.append(
            f"### Source: {record_id}\n{field_lines}\n\nMap answers:\n{answer_lines or '  (none)'}"
        )
    return "\n\n".join(blocks) if blocks else "No eligible sources this run."


def _item_18_synthesis_results(snapshot: "ReportingSourceSnapshot") -> str:
    output = snapshot.synthesis_output.model
    by_target: Dict[str, List[str]] = {}
    for a in output.map_answers:
        by_target.setdefault(a.rq_or_section, []).append(a.record_id)
    blocks = []
    for target in sorted(output.rq_synthesis):
        cited = ", ".join(sorted(by_target.get(target, [])))
        blocks.append(f"### {target}\n{output.rq_synthesis[target]}\n\nCited sources: {cited or '(none)'}")
    return "\n\n".join(blocks) if blocks else "No synthesis narrative was produced this run."


def _item_19_summary(snapshot: "ReportingSourceSnapshot") -> str:
    output = snapshot.synthesis_output.model
    n_sources = len(output.synthesis_record_ids)
    blocks = []
    for target in sorted(output.rq_synthesis):
        blocks.append(
            f"- **{target}:** {output.rq_synthesis[target]} "
            f"(drawn from {n_sources} eligible source(s))"
        )
    return "\n".join(blocks) if blocks else "No sources were eligible for synthesis this run."


def _item_20_limitations(snapshot: "ReportingSourceSnapshot") -> str:
    ctx = snapshot.reference_context
    cfg = snapshot.config_projection
    non_instrumented = ", ".join(ctx.non_instrumented_stages)
    bertopic_note = "BERTopic topic modelling was not run this run." if not cfg.run_bertopic else ""
    return (
        f"**Non-instrumented stages:** {non_instrumented} (see ai_traceability.json, "
        "instrumentation_status=\"not_instrumented\").\n\n"
        f"**Full-text eligibility provenance:** {ctx.full_text_eligibility_provenance_note}\n\n"
        f"{bertopic_note}\n\n"
        f"{ctx.synthetic_corpus_caveat}\n\n"
        f"**Deduplication mode:** {ctx.dedup_mode}\n\n"
        f"**Model calls:** {ctx.llm_transport} -- replayed, not a live run."
    )


def _item_21_conclusions(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    ctx = snapshot.reference_context
    n_rq = len(cfg.research_questions)
    return (
        f"This scoping review addressed {n_rq} research question(s) across "
        f"{len(snapshot.synthesis_output.model.synthesis_record_ids)} eligible source(s). "
        f"{ctx.synthetic_corpus_caveat}"
    )


def _item_22_funding(snapshot: "ReportingSourceSnapshot") -> str:
    cfg = snapshot.config_projection
    output = snapshot.synthesis_output.model
    charted_by_id = _charted_by_id(snapshot)
    lines = []
    for record_id in output.synthesis_record_ids:
        fields = charted_by_id.get(record_id, {})
        funding = _format_field_value(fields.get("funding_sources"))
        role = _format_field_value(fields.get("funder_role"))
        lines.append(f"- **{record_id}**: funding = {funding}; funder role = {role}")
    funding_lines = "\n".join(lines) if lines else "(no eligible sources this run)"
    return (
        f"**Review-level funding statement:** {cfg.funding_statement or 'Not provided.'}\n\n"
        f"**Per-source funding:**\n{funding_lines}"
    )


_ITEM_RENDERERS = {
    1: _item_1_title, 2: _item_2_abstract, 3: _item_3_rationale, 4: _item_4_objectives,
    5: _item_5_protocol, 6: _item_6_eligibility, 7: _item_7_sources, 8: _item_8_search,
    9: _item_9_selection, 10: _item_10_charting, 11: _item_11_data_items,
    12: _item_12_appraisal_not_conducted, 13: _item_13_synthesis_methods,
    14: _item_14_selection_results, 15: _item_15_characteristics,
    16: _item_16_appraisal_within_not_conducted, 17: _item_17_individual_results,
    18: _item_18_synthesis_results, 19: _item_19_summary, 20: _item_20_limitations,
    21: _item_21_conclusions, 22: _item_22_funding,
}


def render_prisma_scr_report_md(snapshot: "ReportingSourceSnapshot") -> str:
    """The full 22-item, deterministic, state-templated PRISMA-ScR report.
    Caller (`reporting.py`'s `ARTIFACT_RENDERERS`) is responsible for the
    `snapshot.reference_context is not None` applicability gate -- every
    item body below reads `snapshot.reference_context` unconditionally."""
    sections = [f"# PRISMA-ScR Checklist Report (PRISMA-ScR {PRISMA_SCR_VERSION})\n"]
    for item in ITEMS:
        body = _ITEM_RENDERERS[item["item_id"]](snapshot)
        sections.append(_anchor_heading(item) + "\n" + body + "\n")
    return "\n".join(sections)


def render_prisma_scr_conformance(snapshot: "ReportingSourceSnapshot") -> Dict[str, Any]:
    """The machine-readable per-item conformance record:
    `{item_id, status, locator?, rationale?}`. `locator` is set only for the
    handful of items that name one concrete, already-published `slr://`
    artifact as their primary evidence trail (item 10's charting run and
    item 20's non-instrumented-stage evidence both live in
    `ai_traceability.json`); every other item's evidence is the report
    prose itself, so no locator is claimed for it."""
    from slr_pipeline.locators import LogicalLocator  # local import: avoids a module-load-time
                                                          # dependency on locators.py for the common
                                                          # case (matrix_payload()/ITEMS) that never
                                                          # needs it.

    ai_traceability_locator = str(LogicalLocator("reporting", "artifact", "ai_traceability_json"))
    items_out: List[Dict[str, Any]] = []
    for item in ITEMS:
        item_id = item["item_id"]
        entry: Dict[str, Any] = {"item_id": item_id}
        if item_id in OPTIONAL_ITEM_IDS:
            entry["status"] = "optional_not_conducted"
            entry["rationale"] = _OPTIONAL_APPRAISAL_RATIONALE
        else:
            entry["status"] = "reported"
            if item_id in (10, 20):
                entry["locator"] = ai_traceability_locator
        items_out.append(entry)
    return {
        "prisma_scr_conformance_version": "1",
        "prisma_scr_version": PRISMA_SCR_VERSION,
        "items": items_out,
    }
