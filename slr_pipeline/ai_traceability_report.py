"""
AI-traceability aggregation -- the `ai_traceability.json`/`ai_traceability.md`
half of the aggregation/export/reporting phase
(`aggregation_export_reporting_plan_2026-08-25.md` v17, §2.2/§2.4/§2.8/§2.12).

Deliberately separate from `nodes/reporting.py`: this module owns the
per-stage AIUseEvent aggregation, the M9/R2 (screening QC) block, and the
static PRISMA-trAIce crosswalk table, while `nodes/reporting.py` owns
snapshot capture, PRISMA/evidence-map/QA/synthesis exports, and
publication. `reporting.py`'s own `ARTIFACT_RENDERERS` calls into this
module's `_render_ai_traceability_dict`/`_render_ai_traceability_md`.

No runtime dependency on `nodes.reporting` -- `ReportingSourceSnapshot` is
imported only under `TYPE_CHECKING` (deferred string annotations via
`from __future__ import annotations`), so `nodes/reporting.py` can safely
import THIS module at its own top level without a circular import.
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.resolution_schemas import EXTRACTION_RESOLUTION_SUGGESTION_STAGE

if TYPE_CHECKING:  # pragma: no cover - type-only, no runtime import
    from slr_pipeline.nodes.reporting import ReportingSourceSnapshot

# ---------------------------------------------------------------------------
# Stage inventory (INV-R14) -- 12 stages, in the plan's own declared order.
# "extraction_resolution_suggestion" (amber-quilting-osprey.md sec. 3.4) is a
# genuinely new, real LLM call site -- distinct from "extraction" (the
# deterministic extraction node, still correctly listed as not-instrumented
# below): the crosswalk table must never conflate "the deterministic
# extraction node has no AI events" with "the new suggestion engine's calls
# are unaccounted for."
# ---------------------------------------------------------------------------

STAGE_ORDER: List[str] = [
    "dedup", "search_strategy", "title_abstract_screening", "screening_qc_second_rater",
    "full_text_eligibility", "synthesis_claim_map", "synthesis_claim_refine",
    "synthesis_imrad_narrative", "synthesis_bertopic", "extraction", "quality_appraisal",
    "extraction_resolution_suggestion",
]

# Maps a stage-inventory key (== the SAME key oversight_mode/automation_origin
# dicts use) to the set of real AIUseEvent.stage values that feed it -- these
# two granularities differ for "dedup" (its own event stages are
# "dedup_embedding"/"dedup_exact" depending on DedupConfig.mode, Track 2 plan
# §3.1/§3.2, nodes/dedup.py) and "search_strategy" (its own two event
# stages are "search_strategy_query_expansion"/"search_strategy_db_translation",
# nodes/search_strategy_builder.py). "full_text_eligibility" has NO AIUseEvent
# stage of its own at all -- FT_SCREENING's own internal provenance is a
# separate, byte-hashed-not-semantically-reverified block
# (`ft_native_provenance`), never aggregated into this event-count table.
_STAGE_EVENT_STAGES: Dict[str, frozenset] = {
    "dedup": frozenset({"dedup_embedding", "dedup_exact"}),
    "search_strategy": frozenset({"search_strategy_query_expansion", "search_strategy_db_translation"}),
    "title_abstract_screening": frozenset({"title_abstract_screening"}),
    "screening_qc_second_rater": frozenset({"screening_qc_second_rater"}),
    "full_text_eligibility": frozenset(),
    "synthesis_claim_map": frozenset({"synthesis_claim_map"}),
    "synthesis_claim_refine": frozenset({"synthesis_claim_refine"}),
    "synthesis_imrad_narrative": frozenset({"synthesis_imrad_narrative"}),
    "extraction_resolution_suggestion": frozenset({EXTRACTION_RESOLUTION_SUGGESTION_STAGE}),
}

# INV-R14 -- named explicitly, not merely absent from the instrumented set.
_NOT_INSTRUMENTED_REASONS: Dict[str, str] = {
    "synthesis_bertopic": (
        "BERTopic clustering has no AIUseEvent instrumentation yet -- deferred to a follow-up "
        "phase (plan §2.6, full instrumentation design preserved but unimplemented)."
    ),
    "extraction": (
        "extraction_node emits zero AIUseEvents as of this phase's base commit -- deferred to a "
        "follow-up phase (plan §0.1)."
    ),
    "quality_appraisal": (
        "quality_appraisal_node emits zero AIUseEvents as of this phase's base commit -- deferred "
        "to a follow-up phase (plan §0.1)."
    ),
}

# The M9/R2 note is attached to graph_state, non-durably-reverified data
# (INV-R2 bucket 2) -- screening_qc's own IRR result is read from
# ReviewState.screening_qc as last written by screening_qc_node, not from a
# durably re-verified artifact the way every other stage's own event
# aggregation is.
LIMITATION_TEXT = (
    "screening_qc's IRR statistics are graph_state (ReviewState.screening_qc, deep-copied at "
    "capture time), not a durably re-verified artifact like an AIUseEvent or a checkpoint store -- "
    "and screening_qc's own second-rater agreement is an second_ai_rater comparison, never a human "
    "reference standard (PRISMA-trAIce M9/R2, see the crosswalk below)."
)


def _md_not_provided(value: Any) -> Any:
    """INV-R9: Markdown renders an absent value via the fixed 'Not
    provided' convention -- JSON keeps the typed `None`/`null`. Applied to
    every M1 field individually (Codex round-1 review, finding 8) rather
    than only to `protocol_deviations`/M10, which already used it."""
    return value if value is not None else "Not provided"


def _render_resolution_list_md(resolutions: Optional[List[Dict[str, Any]]]) -> str:
    """Renders a review-queue resolution audit trail's FULL contents
    (every key/value the producing node persisted -- reviewer, note,
    resolved_at, decision/action, resolution_id, ...) generically, since
    screening_review_resolutions and full_text_review_resolutions do not
    share an identical key set (nodes/screening.py vs. nodes/fulltext.py).
    Codex round-1 review, finding 5 (M8): the JSON side
    (`m8_audit_trails`) carries the complete, unmodified dicts; this is
    the human-readable rendering of the SAME data, not a separate source.
    Codex round-11 review, finding 2: `None` (the underlying bundle itself
    doesn't exist -- empty corpus / scoping review) is distinguished from
    `[]` (the bundle exists but genuinely has zero resolutions/
    confirmations this run), per INV-R9's "not applicable" vs. "applicable
    but empty" convention."""
    if resolutions is None:
        return "  - Not applicable (no extraction/quality-appraisal run this review).\n"
    if not resolutions:
        return "  - None this run.\n"
    lines = []
    for r in resolutions:
        parts = ", ".join(f"{k}={v!r}" for k, v in sorted(r.items()))
        lines.append(f"  - {parts}")
    return "\n".join(lines) + "\n"


def _interrupted_suggestion_count(snapshot: "ReportingSourceSnapshot") -> Optional[int]:
    """Real count of archived `*.interrupted_unknown.json` recovery markers
    for this run's extraction_resolution_suggestion stage
    (`count_interrupted_suggestions()`, resolution_suggestions.py) --
    these attempts never produced a published AIUseEvent
    (nodes/reporting.py's `_load_extraction_review_artifacts` explicitly
    forbids one), so the stage's own `event_count` above silently excludes
    them. `None` (not 0), INV-R9's "not applicable" convention, when no
    extraction-review bundle was captured this run at all -- distinct from
    an extraction-review bundle that genuinely recorded zero interrupted
    attempts."""
    if snapshot.extraction is None or snapshot.extraction.extraction_review is None:
        return None
    return sum(
        1 for component in snapshot.extraction.extraction_review.suggestions
        if component.path.name.endswith(".interrupted_unknown.json")
    )


def _stage_entry(snapshot: "ReportingSourceSnapshot", stage: str) -> Dict[str, Any]:
    interrupted_count = (
        _interrupted_suggestion_count(snapshot) if stage == "extraction_resolution_suggestion" else None
    )
    if stage in _NOT_INSTRUMENTED_REASONS:
        return {
            "instrumentation_status": "not_instrumented",
            "not_instrumented_reason": _NOT_INSTRUMENTED_REASONS[stage],
            "event_count": 0, "by_call_status": {}, "by_tool_kind": {}, "models": [],
            "component_versions_union": {},
            "oversight_mode": snapshot.oversight_mode_state.get(stage),
            "automation_origin": snapshot.automation_origin_state.get(stage),
            "notable_events": [],
            "interrupted_count": interrupted_count,
        }

    event_stages = _STAGE_EVENT_STAGES[stage]
    events = [vf.model for vf in snapshot.events if vf.model.stage in event_stages]
    component_versions_union: Dict[str, str] = {}
    for e in events:
        component_versions_union.update(e.component_versions)
    return {
        "instrumentation_status": "instrumented",
        "not_instrumented_reason": None,
        "event_count": len(events),
        "by_call_status": dict(Counter(e.call_status for e in events)),
        "by_tool_kind": dict(Counter(e.tool_kind for e in events)),
        "models": sorted({e.model for e in events}),
        "component_versions_union": component_versions_union,
        "oversight_mode": snapshot.oversight_mode_state.get(stage),
        "automation_origin": snapshot.automation_origin_state.get(stage),
        # Every non-"success" event this stage produced this run -- a
        # heuristic_fallback/failure is exactly the case a human reviewer
        # of this file most needs surfaced without having to scan every
        # individual event record.
        "notable_events": sorted(e.event_id for e in events if e.call_status != "success"),
        "interrupted_count": interrupted_count,
    }


def _m9_r2_dict(snapshot: "ReportingSourceSnapshot") -> Dict[str, Any]:
    qc_state = snapshot.screening_qc_state
    if qc_state is None:
        return {"evaluation_status": "not_run", "reference_standard": None,
                "n_attempted": None, "n_valid_pairs": None, "n_failed": None, "irr_stats": None,
                "source_kind": "graph_state", "note": LIMITATION_TEXT}
    irr = qc_state.irr
    base = {"n_attempted": irr.n_attempted, "n_valid_pairs": irr.n_valid_pairs, "n_failed": irr.n_failed,
            "source_kind": "graph_state", "note": LIMITATION_TEXT}
    if irr.n_attempted == 0:
        return {**base, "evaluation_status": "not_run_no_eligible_records", "reference_standard": None, "irr_stats": None}
    if irr.n_valid_pairs == 0:
        return {**base, "evaluation_status": "attempted_no_valid_pairs", "reference_standard": "second_ai_rater", "irr_stats": None}
    return {**base, "evaluation_status": "completed", "reference_standard": "second_ai_rater", "irr_stats": irr.model_dump()}


def _render_ai_traceability_dict(snapshot: "ReportingSourceSnapshot") -> Dict[str, Any]:
    cp = snapshot.config_projection
    events_sorted = sorted(
        (vf.model.model_dump(mode="json") for vf in snapshot.events),
        key=lambda e: (e["stage"], e["work_item_id"], e["attempt"]),
    )
    body: Dict[str, Any] = {
        "review_run_id": snapshot.review_run_id,
        "stages": {stage: _stage_entry(snapshot, stage) for stage in STAGE_ORDER},
        "full_text_eligibility_native_provenance": snapshot.ft_native_provenance,
        "events": events_sorted,
        "m9_r2": _m9_r2_dict(snapshot),
        "m1_protocol": {
            "protocol_uri": cp.protocol_uri, "protocol_hash": cp.protocol_hash,
            "protocol_registration_id": cp.protocol_registration_id,
            "ai_use_prespecified": cp.ai_use_prespecified, "protocol_deviations": cp.protocol_deviations,
            "source_kind": "config",
        },
        "m10_data_governance": {"value": cp.data_governance_declaration, "source_kind": "config"},
        # Codex round-1 review, finding 5 (M8): the FULL contents of every
        # review-queue resolution audit trail this phase captures --
        # ledger records alone cannot substitute (they omit reviewer/
        # resolved_at/note), so M8's own crosswalk claim ("every
        # review-queue's resolution/confirmation audit trail") requires
        # this to actually be exported, not merely referenced by count.
        # Codex round-11 review, finding 2 (M8): the base plan's own
        # crosswalk (quality_appraisal_synthesis_plan_2026-08-22.md:663)
        # requires M8 to cover FOUR review systems' resolution/confirmation
        # audit trails -- screening and full-text (already exported since
        # round 1) PLUS extraction's own `resolutions` and QA's own
        # `resolutions`/`confirmations`. All four are already present in
        # the verified `snapshot.extraction.raw`/`snapshot.quality_
        # appraisal.raw` bundles (loaded/verified once at snapshot-capture
        # time, INV-R19) -- this only surfaces what was already in hand.
        # `None` (not `[]`) when the underlying bundle itself doesn't exist
        # (empty corpus / scoping review), matching this codebase's own
        # "not applicable" vs. "applicable but empty" convention (a bundle
        # that DOES exist but genuinely has zero resolutions/confirmations
        # renders as `[]`, per extraction.py/quality_appraisal.py's own
        # `raw["resolutions"]`/`raw["confirmations"]` list-typed contract).
        # INV-R2: this block genuinely mixes two source kinds -- screening/
        # full-text resolutions are graph_state (deep-copied ReviewState
        # lists), while extraction/QA resolutions/confirmations are
        # durable_reverified (sourced from the already-verified extraction/
        # QA bundles) -- so each entry carries its OWN source_kind rather
        # than one flat, now-inaccurate block-level label.
        "m8_audit_trails": {
            "screening_review_resolutions": list(snapshot.screening_review_resolutions.model),
            "full_text_review_resolutions": list(snapshot.full_text_review_resolutions.model),
            "extraction_review_resolutions": (
                list(snapshot.extraction.raw["resolutions"]) if snapshot.extraction is not None else None
            ),
            "quality_appraisal_resolutions": (
                list(snapshot.quality_appraisal.raw["resolutions"]) if snapshot.quality_appraisal is not None else None
            ),
            "quality_appraisal_confirmations": (
                list(snapshot.quality_appraisal.raw["confirmations"]) if snapshot.quality_appraisal is not None else None
            ),
            "source_kind": {
                "screening_review_resolutions": "graph_state",
                "full_text_review_resolutions": "graph_state",
                "extraction_review_resolutions": "durable_reverified",
                "quality_appraisal_resolutions": "durable_reverified",
                "quality_appraisal_confirmations": "durable_reverified",
            },
        },
    }
    body["content_hash"] = sha256_hex(body)
    return body


# ---------------------------------------------------------------------------
# PRISMA-trAIce crosswalk -- static, corrected per the plan's own §2.12
# erratum (M2-M7 downgraded from "covered"; M1/M8/M10 stay "covered"; M9/R2
# keep their own pre-existing, explicitly-labeled limitation).
# ---------------------------------------------------------------------------

_CROSSWALK_ROWS = [
    ("M1", "Protocol and registration", "Mandatory",
     "ReviewConfig.protocol_uri/.protocol_hash/.protocol_registration_id/.ai_use_prespecified/"
     ".protocol_deviations, recorded in export_manifest.json's m1_protocol block", "covered"),
    ("M2", "AI tool identification and access", "Mandatory",
     "AIUseEvent.model/.provider/.component_versions/.access_info, aggregated per stage in "
     "ai_traceability.json", "partially_covered -- see 2026-08-26 erratum (extraction/QA/BERTopic not instrumented)"),
    ("M3", "Purpose and stage of application", "Mandatory",
     "AIUseEvent.stage/.task_purpose + input_artifact_refs/output_artifact_refs",
     "partially_covered -- see 2026-08-26 erratum (extraction/QA/BERTopic not instrumented)"),
    ("M4", "Input data", "Mandatory", "AIUseEvent.input_artifact_refs",
     "partially_covered -- see 2026-08-26 erratum (extraction/QA/BERTopic not instrumented)"),
    ("M5", "Output data", "Mandatory", "AIUseEvent.output_artifact_refs + the referenced artifact's own schema",
     "partially_covered -- see 2026-08-26 erratum (extraction/QA/BERTopic not instrumented)"),
    ("M6", "Prompt engineering", "Mandatory",
     "AIUseEvent.prompt_template_text/.prompt_template_hash/.prompt_parameters/.rendered_prompt_hash",
     "partially_covered -- see 2026-08-26 erratum (extraction/QA/BERTopic not instrumented)"),
    ("M7", "Operational details/settings (non-LLM tools)", "Highly Recommended",
     "AIUseEvent generalized to non-LLM tool use (dedup's embedding step); BERTopic itself not yet instrumented",
     "partially_covered -- see 2026-08-26 erratum (extraction/QA/BERTopic not instrumented)"),
    ("M8", "Human-AI interaction and oversight", "Mandatory",
     "oversight_mode + all four review systems' resolution/confirmation audit trails "
     "(screening_review_resolutions, full_text_review_resolutions, extraction_review_resolutions, "
     "quality_appraisal_resolutions/confirmations), exported in full in m8_audit_trails", "covered"),
    ("M9", "AI performance evaluation (methods)", "Mandatory",
     "screening_qc_node's kappa/PABAK/Gwet's AC1 -- labeled explicitly as second_ai_rater agreement, "
     "not a human reference standard", "partially_covered -- explicitly labeled limitation"),
    ("M10", "Data governance and ethics", "Recommended",
     "ReviewConfig.data_governance_declaration, rendered into methods.md/ai_traceability.md", "covered"),
    ("R1", "Study selection (AI-assisted) -- PRISMA flow distinguishing AI vs. human decisions", "Mandatory",
     "selection_decision_ledger (SelectionDecisionRecord) + reporting.py's extended PRISMA diagram", "covered"),
    ("R2", "AI performance metrics (results)", "Mandatory",
     "Same screening_qc output as M9, same explicit labeling", "partially_covered -- explicitly labeled limitation"),
]


def _render_crosswalk_md() -> str:
    header = "| Item | Description | Priority | SLR_chapter artifact/field | Status |\n"
    header += "|---|---|---|---|---|\n"
    rows = "\n".join(
        f"| {item} | {desc} | {priority} | {artifact} | {status} |"
        for item, desc, priority, artifact, status in _CROSSWALK_ROWS
    )
    return header + rows + "\n"


def _render_ai_traceability_md(snapshot: "ReportingSourceSnapshot") -> str:
    d = _render_ai_traceability_dict(snapshot)
    lines = [f"# AI traceability -- review_run_id={d['review_run_id']}", ""]

    lines.append("## Stage inventory")
    lines.append("| Stage | Instrumentation | Events | Oversight mode | Automation origin |")
    lines.append("|---|---|---|---|---|")
    for stage in STAGE_ORDER:
        entry = d["stages"][stage]
        # Finding 3 (Codex round-2 review, P2): oversight_mode/
        # automation_origin are genuinely None for a stage that did not
        # execute this run -- INV-R9's "Not provided" convention applies
        # here exactly as it does to M1, not only to M1.
        lines.append(
            f"| {stage} | {entry['instrumentation_status']} | {entry['event_count']} | "
            f"{_md_not_provided(entry['oversight_mode'])} | {_md_not_provided(entry['automation_origin'])} |"
        )
    lines.append("")

    for stage in STAGE_ORDER:
        entry = d["stages"][stage]
        if entry["instrumentation_status"] == "not_instrumented":
            lines.append(f"**{stage}**: not instrumented -- {entry['not_instrumented_reason']}")
            lines.append("")

    lines.append("## Extraction resolution suggestion -- interrupted / unknown-outcome attempts")
    interrupted_count = d["stages"]["extraction_resolution_suggestion"]["interrupted_count"]
    if interrupted_count is None:
        lines.append("Not applicable (no extraction-review bundle captured this review).")
    else:
        lines.append(
            f"{interrupted_count} interrupted/unknown-outcome attempt(s) -- archived "
            "`*.interrupted_unknown.json` recovery markers whose true LLM call outcome is genuinely "
            "unknown (the process ended before a definitive success/failure could be recorded). "
            "Excluded from extraction_resolution_suggestion's event_count above, since these attempts "
            "never produced a published AIUseEvent, and never auto-retried."
        )
    lines.append("")

    ft = d["full_text_eligibility_native_provenance"]
    lines.append("## Full-text eligibility native provenance")
    if ft is None:
        lines.append("No FT_SCREENING run occurred this review (no confirmed PDF ever reached full-text eligibility).")
    else:
        lines.append(f"- ft_screening_run_id: `{ft['ft_screening_run_id']}`")
        lines.append(f"- ft_screening_execution_fingerprint: `{ft['ft_screening_execution_fingerprint']}`")
        lines.append(f"- classification: {ft['classification']}")
    lines.append("")

    m9 = d["m9_r2"]
    lines.append("## M9/R2 -- AI performance evaluation (screening QC)")
    lines.append(f"- evaluation_status: {m9['evaluation_status']}")
    # Finding 3 (Codex round-2 review, P2): reference_standard/n_attempted/
    # n_valid_pairs/n_failed are all genuinely None when screening QC never
    # ran (evaluation_status="not_run") -- same INV-R9 convention as M1.
    lines.append(f"- reference_standard: {_md_not_provided(m9['reference_standard'])}")
    lines.append(
        f"- n_attempted / n_valid_pairs / n_failed: {_md_not_provided(m9['n_attempted'])} / "
        f"{_md_not_provided(m9['n_valid_pairs'])} / {_md_not_provided(m9['n_failed'])}"
    )
    lines.append(f"- note: {m9['note']}")
    lines.append("")

    m1 = d["m1_protocol"]
    lines.append("## M1 -- Protocol and registration")
    lines.append(f"- protocol_uri: {_md_not_provided(m1['protocol_uri'])}")
    lines.append(f"- protocol_hash: {_md_not_provided(m1['protocol_hash'])}")
    lines.append(f"- protocol_registration_id: {_md_not_provided(m1['protocol_registration_id'])}")
    lines.append(f"- ai_use_prespecified: {_md_not_provided(m1['ai_use_prespecified'])}")
    lines.append(f"- protocol_deviations: {m1['protocol_deviations'] or 'Not provided'}")
    lines.append("")

    m8 = d["m8_audit_trails"]
    lines.append("## M8 -- Human-AI interaction and oversight")
    for key in ("screening_review_resolutions", "full_text_review_resolutions",
                "extraction_review_resolutions", "quality_appraisal_resolutions",
                "quality_appraisal_confirmations"):
        entries = m8[key]
        count_label = "not applicable" if entries is None else f"{len(entries)} entries"
        lines.append(f"- {key} ({count_label}):")
        lines.append(_render_resolution_list_md(entries))
    lines.append("")

    lines.append("## M10 -- Data governance and ethics")
    lines.append(d["m10_data_governance"]["value"] or "Not provided")
    lines.append("")

    lines.append("## PRISMA-trAIce crosswalk")
    lines.append(_render_crosswalk_md())

    lines.append(f"content_hash: `{d['content_hash']}`")
    return "\n".join(lines) + "\n"
