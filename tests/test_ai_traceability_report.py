"""Tests for `slr_pipeline/ai_traceability_report.py` -- the
`ai_traceability.json`/`.md` half of the aggregation/export/reporting
phase (`aggregation_export_reporting_plan_2026-08-25.md` v17, §2.2/§2.4/
§2.8/§2.12). Complements `tests/nodes/test_reporting_node.py`'s own
full-pipeline smoke test, which already exercises these functions
end-to-end over real captured snapshots; the tests here isolate this
module's own stage-inventory/M9-R2/crosswalk logic against hand-built
snapshots.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore, build_ai_use_event
from slr_pipeline.ai_traceability_report import (
    STAGE_ORDER, _interrupted_suggestion_count, _m9_r2_dict, _render_ai_traceability_dict,
    _render_ai_traceability_md,
)
from slr_pipeline.config import ReviewConfig
from slr_pipeline.state import SynthesisOutput

from slr_pipeline.nodes.reporting import (
    ExtractionReviewArtifacts, ReportingConfigProjection, ReportingIRRState, ReportingScreeningQCState,
    ReportingSourceSnapshot, SourceBundle, VerifiedComponent, VerifiedFile, VerifiedProjection,
    _hash_value, _state_value,
)


def _empty_snapshot(cfg: ReviewConfig, *, oversight_mode=None, automation_origin=None,
                     events=None, screening_qc_state=None, ft_native_provenance=None,
                     screening_review_resolutions=None, full_text_review_resolutions=None,
                     extraction=None, quality_appraisal=None) -> ReportingSourceSnapshot:
    config_projection = ReportingConfigProjection.from_config(cfg)
    return ReportingSourceSnapshot(
        review_run_id="run-1", config_projection=config_projection, config_projection_hash="a" * 64,
        raw_records=_state_value([]),
        screening_records=[], full_text_records=[], screened_records_full=[],
        prisma=_state_value({}), search_log=_state_value({}),
        screening_review_resolutions=_state_value(screening_review_resolutions or []),
        full_text_review_resolutions=_state_value(full_text_review_resolutions or []),
        full_text_review_queue=_state_value([]),
        screening_qc_state=screening_qc_state, oversight_mode_state=oversight_mode or {},
        automation_origin_state=automation_origin or {},
        events=events or [], ledger_records=[], current_screening_batch_results=[],
        extraction=extraction, quality_appraisal=quality_appraisal,
        synthesis_output=VerifiedProjection(model=SynthesisOutput(refine_batch_size=1), content_hash="b" * 64),
        evidence_map=None, ft_native_provenance=ft_native_provenance,
    )


def _fake_bundle(*, resolutions=None, confirmations=None, extraction_review=None) -> SourceBundle:
    """A minimal SourceBundle stand-in for ai_traceability_report tests --
    only `.raw` (and, for the interrupted-suggestion tests, `.extraction_review`)
    is read by this module, so the other fields are placeholders."""
    raw: Dict[str, Any] = {"resolutions": resolutions if resolutions is not None else []}
    if confirmations is not None:
        raw["confirmations"] = confirmations
    component = VerifiedComponent(path=Path("x"), file_sha256="d" * 64, internal_hash="e" * 64)
    return SourceBundle(raw=raw, manifest=component, checkpoints=[], resolutions=None,
                         confirmations=None, computed_content_hash="f" * 64, run_id="fake-run",
                         extraction_review=extraction_review)


def _fake_suggestion_component(name: str) -> VerifiedComponent:
    return VerifiedComponent(path=Path(name), file_sha256="d" * 64, internal_hash="e" * 64)


def _event_file(*, stage: str, work_item_id: str, call_status: str = "success") -> VerifiedFile:
    event = build_ai_use_event(
        review_run_id="run-1", stage=stage, work_item_id=work_item_id, tool_kind="non_llm_tool",
        model="gpt-5.6-terra", provider="openai", component_versions={"gpt-5.6-terra": "2026-07-30"},
        task_purpose="test", model_config_id="cfg-1",
        input_artifact_refs=["in-1"], output_artifact_refs=(["out-1"] if call_status != "failure" else []),
        call_status=call_status,
    )
    return VerifiedFile(model=event, path=Path("x"), file_sha256="c" * 64, internal_hash=event.content_hash)


# ---------------------------------------------------------------------------
# Stage inventory (INV-R14)
# ---------------------------------------------------------------------------

def test_stage_order_has_twelve_stages():
    assert len(STAGE_ORDER) == 12


def test_extraction_resolution_suggestion_is_instrumented_separately_from_extraction():
    """amber-quilting-osprey.md sec. 3.4: a new, real LLM call site, never
    conflated with the still-correctly-uninstrumented deterministic
    "extraction" stage entry."""
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    entry = d["stages"]["extraction_resolution_suggestion"]
    assert entry["instrumentation_status"] == "instrumented"
    assert entry["event_count"] == 0
    assert d["stages"]["extraction"]["instrumentation_status"] == "not_instrumented"


def test_not_instrumented_stages_named_explicitly():
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    for stage in ("extraction", "quality_appraisal", "synthesis_bertopic"):
        entry = d["stages"][stage]
        assert entry["instrumentation_status"] == "not_instrumented"
        assert entry["not_instrumented_reason"]
        assert entry["event_count"] == 0


def test_instrumented_stage_with_zero_events_reports_zero():
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    entry = d["stages"]["dedup"]
    assert entry["instrumentation_status"] == "instrumented"
    assert entry["event_count"] == 0
    assert entry["by_call_status"] == {}


def test_dedup_stage_aggregates_dedup_embedding_events():
    """'dedup' the stage-inventory key groups the REAL underlying
    AIUseEvent.stage="dedup_embedding" events, not a literal "dedup" stage
    string (nodes/dedup.py)."""
    cfg = ReviewConfig(title="t")
    events = [_event_file(stage="dedup_embedding", work_item_id="w1", call_status="success")]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    assert d["stages"]["dedup"]["event_count"] == 1
    assert d["stages"]["dedup"]["by_call_status"] == {"success": 1}


def test_dedup_stage_aggregates_dedup_exact_events():
    """Step-4 follow-up: DedupConfig.mode="exact" publishes AIUseEvent.stage=
    "dedup_exact" (nodes/dedup.py), not "dedup_embedding" -- the 'dedup'
    stage-inventory key's frozenset({"dedup_embedding", "dedup_exact"})
    (ai_traceability_report.py:49) must aggregate this underlying stage too,
    not just the embedding one already covered above."""
    cfg = ReviewConfig(title="t")
    events = [_event_file(stage="dedup_exact", work_item_id="w1", call_status="success")]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    assert d["stages"]["dedup"]["event_count"] == 1
    assert d["stages"]["dedup"]["by_call_status"] == {"success": 1}


def test_dedup_stage_never_mixes_exact_and_embedding_events_in_one_run():
    """A single run is always mode="exact" XOR "semantic"/"auto" -- but the
    'dedup' stage-inventory key groups by underlying-stage *set*, so if both
    happened to be present it must still sum correctly (guards against a
    future regression narrowing the aggregation back to one literal stage
    string instead of the frozenset)."""
    cfg = ReviewConfig(title="t")
    events = [
        _event_file(stage="dedup_embedding", work_item_id="w1", call_status="success"),
        _event_file(stage="dedup_exact", work_item_id="w2", call_status="success"),
    ]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    assert d["stages"]["dedup"]["event_count"] == 2


def test_search_strategy_stage_aggregates_both_underlying_event_stages():
    cfg = ReviewConfig(title="t")
    events = [
        _event_file(stage="search_strategy_query_expansion", work_item_id="w1", call_status="success"),
        _event_file(stage="search_strategy_db_translation", work_item_id="w2", call_status="heuristic_fallback"),
    ]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    assert d["stages"]["search_strategy"]["event_count"] == 2
    assert d["stages"]["search_strategy"]["by_call_status"] == {"success": 1, "heuristic_fallback": 1}


def test_full_text_eligibility_stage_never_aggregates_events():
    """full_text_eligibility has NO AIUseEvent stage of its own -- its
    events count stays 0 regardless of what other stages' events exist."""
    cfg = ReviewConfig(title="t")
    events = [_event_file(stage="title_abstract_screening", work_item_id="w1")]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    assert d["stages"]["full_text_eligibility"]["event_count"] == 0


def test_notable_events_excludes_success_only():
    cfg = ReviewConfig(title="t")
    events = [
        _event_file(stage="dedup_embedding", work_item_id="w1", call_status="success"),
        _event_file(stage="dedup_embedding", work_item_id="w2", call_status="heuristic_fallback"),
    ]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    notable = d["stages"]["dedup"]["notable_events"]
    assert len(notable) == 1
    assert notable[0] == events[1].model.event_id


# ---------------------------------------------------------------------------
# Interrupted / unknown-outcome extraction-resolution-suggestion attempts
# ---------------------------------------------------------------------------

def test_interrupted_suggestion_count_none_when_no_extraction_bundle():
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    assert d["stages"]["extraction_resolution_suggestion"]["interrupted_count"] is None
    assert _interrupted_suggestion_count(_empty_snapshot(cfg)) is None


def test_interrupted_suggestion_count_none_when_bundle_has_no_review():
    cfg = ReviewConfig(title="t")
    bundle = _fake_bundle()
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, extraction=bundle))
    assert d["stages"]["extraction_resolution_suggestion"]["interrupted_count"] is None


def test_interrupted_suggestion_count_zero_when_review_has_no_interrupted_attempts():
    cfg = ReviewConfig(title="t")
    review = ExtractionReviewArtifacts(suggestions=(_fake_suggestion_component("a__attempt-1.json"),))
    bundle = _fake_bundle(extraction_review=review)
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, extraction=bundle))
    assert d["stages"]["extraction_resolution_suggestion"]["interrupted_count"] == 0


def test_interrupted_suggestion_count_counts_real_interrupted_markers():
    cfg = ReviewConfig(title="t")
    review = ExtractionReviewArtifacts(suggestions=(
        _fake_suggestion_component("a__attempt-1.json"),
        _fake_suggestion_component("b__attempt-1.interrupted_unknown.json"),
        _fake_suggestion_component("c__attempt-2.interrupted_unknown.json"),
    ))
    bundle = _fake_bundle(extraction_review=review)
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, extraction=bundle))
    assert d["stages"]["extraction_resolution_suggestion"]["interrupted_count"] == 2


def test_interrupted_suggestion_markdown_not_applicable_when_no_bundle():
    cfg = ReviewConfig(title="t")
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    assert "## Extraction resolution suggestion -- interrupted / unknown-outcome attempts" in md
    assert "Not applicable (no extraction-review bundle captured this review)." in md


def test_interrupted_suggestion_markdown_reports_real_count():
    cfg = ReviewConfig(title="t")
    review = ExtractionReviewArtifacts(suggestions=(
        _fake_suggestion_component("b__attempt-1.interrupted_unknown.json"),
    ))
    bundle = _fake_bundle(extraction_review=review)
    md = _render_ai_traceability_md(_empty_snapshot(cfg, extraction=bundle))
    assert "1 interrupted/unknown-outcome attempt(s)" in md
    assert "never auto-retried" in md


def test_stage_reads_oversight_and_automation_origin_from_snapshot():
    cfg = ReviewConfig(title="t")
    snapshot = _empty_snapshot(cfg, oversight_mode={"dedup": "automation_unreviewed"},
                                automation_origin={"dedup": "rule_based_automation"})
    d = _render_ai_traceability_dict(snapshot)
    assert d["stages"]["dedup"]["oversight_mode"] == "automation_unreviewed"
    assert d["stages"]["dedup"]["automation_origin"] == "rule_based_automation"


# ---------------------------------------------------------------------------
# M9/R2 (§2.4)
# ---------------------------------------------------------------------------

def test_m9_r2_not_run_when_no_qc_state():
    cfg = ReviewConfig(title="t")
    d = _m9_r2_dict(_empty_snapshot(cfg))
    assert d["evaluation_status"] == "not_run"
    assert d["source_kind"] == "graph_state"
    assert d["note"]


def test_m9_r2_not_run_no_eligible_records():
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(n_attempted=0, n_valid_pairs=0, n_failed=0))
    d = _m9_r2_dict(_empty_snapshot(cfg, screening_qc_state=qc))
    assert d["evaluation_status"] == "not_run_no_eligible_records"
    assert d["reference_standard"] is None


def test_m9_r2_attempted_no_valid_pairs():
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=3, n_valid_pairs=0, n_failed=3, sampled_record_ids=["a", "b", "c"],
        producing_event_ids=["e1", "e2", "e3"]))
    d = _m9_r2_dict(_empty_snapshot(cfg, screening_qc_state=qc))
    assert d["evaluation_status"] == "attempted_no_valid_pairs"
    assert d["reference_standard"] == "second_ai_rater"
    assert d["irr_stats"] is None


def test_m9_r2_completed_never_a_human_reference_standard():
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=2, n_valid_pairs=2, n_failed=0, sampled_record_ids=["a", "b"],
        pct_agreement=100.0, cohens_kappa=1.0, producing_event_ids=["e1", "e2"]))
    d = _m9_r2_dict(_empty_snapshot(cfg, screening_qc_state=qc))
    assert d["evaluation_status"] == "completed"
    assert d["reference_standard"] == "second_ai_rater"
    assert "human" not in d["reference_standard"]
    assert d["irr_stats"]["n_attempted"] == 2


# ---------------------------------------------------------------------------
# content_hash / M1 / M10 / full_text_eligibility_native_provenance
# ---------------------------------------------------------------------------

def test_content_hash_is_self_consistent():
    cfg = ReviewConfig(title="t", data_governance_declaration="No PII collected.")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    recomputed = _hash_value({k: v for k, v in d.items() if k != "content_hash"})
    from slr_pipeline.extraction.hashing import sha256_hex
    assert d["content_hash"] == sha256_hex({k: v for k, v in d.items() if k != "content_hash"})


def test_m10_data_governance_passed_through_verbatim():
    cfg = ReviewConfig(title="t", data_governance_declaration="No PII collected.")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    assert d["m10_data_governance"] == {"value": "No PII collected.", "source_kind": "config"}


def test_m1_protocol_fields_null_when_not_supplied():
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    assert d["m1_protocol"]["protocol_uri"] is None
    assert d["m1_protocol"]["source_kind"] == "config"


def test_ft_native_provenance_none_when_no_ft_screening_run():
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    assert d["full_text_eligibility_native_provenance"] is None


def test_ft_native_provenance_passed_through_when_present():
    cfg = ReviewConfig(title="t")
    ft = {"ft_screening_run_id": "ftrun-1", "ft_screening_execution_fingerprint": "fp",
          "ft_screening_artifact_paths": {}, "artifact_file_hashes": {},
          "classification": "state_referenced_byte_hashed_not_semantically_reverified"}
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, ft_native_provenance=ft))
    assert d["full_text_eligibility_native_provenance"]["ft_screening_run_id"] == "ftrun-1"


# ---------------------------------------------------------------------------
# events list (§2.2) -- ordering
# ---------------------------------------------------------------------------

def test_events_list_sorted_by_stage_work_item_attempt():
    cfg = ReviewConfig(title="t")
    events = [
        _event_file(stage="title_abstract_screening", work_item_id="w2"),
        _event_file(stage="dedup_embedding", work_item_id="w1"),
    ]
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, events=events))
    stages_in_order = [e["stage"] for e in d["events"]]
    assert stages_in_order == sorted(stages_in_order)


# ---------------------------------------------------------------------------
# Markdown rendering -- crosswalk erratum + basic structure
# ---------------------------------------------------------------------------

def test_markdown_contains_stage_inventory_and_crosswalk():
    cfg = ReviewConfig(title="t")
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    assert "Stage inventory" in md
    assert "PRISMA-trAIce crosswalk" in md
    assert "M9" in md and "M10" in md


def test_markdown_crosswalk_reflects_erratum_for_m2_through_m7():
    cfg = ReviewConfig(title="t")
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    assert "2026-08-26 erratum" in md
    assert "| M1 |" in md and "covered |" in md


def test_markdown_m9_reference_standard_never_claims_human():
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=2, n_valid_pairs=2, n_failed=0, sampled_record_ids=["a", "b"],
        producing_event_ids=["e1", "e2"]))
    md = _render_ai_traceability_md(_empty_snapshot(cfg, screening_qc_state=qc))
    assert "second_ai_rater" in md
    assert "human reference standard" not in md.lower() or "not a human reference standard" in md.lower()


# ---------------------------------------------------------------------------
# Codex round-1 review, finding 8: Markdown "Not provided" convention (INV-R9)
# ---------------------------------------------------------------------------

def test_markdown_m1_fields_render_not_provided_when_absent():
    """M1 fields must use the plan's own 'Not provided' Markdown convention
    (INV-R9), never Python's raw `None`/`str(None)`."""
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    assert d["m1_protocol"]["protocol_uri"] is None  # JSON keeps typed None
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    assert "protocol_uri: Not provided" in md
    assert "protocol_hash: Not provided" in md
    assert "protocol_registration_id: Not provided" in md
    assert "ai_use_prespecified: Not provided" in md
    assert "protocol_uri: None" not in md
    assert "None" not in md.split("## M1")[1].split("## M8")[0]


def test_markdown_m1_fields_render_actual_value_when_present():
    cfg = ReviewConfig(title="t", protocol_uri="https://osf.io/x", ai_use_prespecified=False)
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    assert "protocol_uri: https://osf.io/x" in md
    assert "ai_use_prespecified: False" in md


# ---------------------------------------------------------------------------
# Codex round-2 review, finding 3 (P2): `_md_not_provided()` was applied
# only to M1 -- the stage table and M9/R2 block still rendered raw `None`.
# ---------------------------------------------------------------------------

def test_markdown_stage_table_uses_not_provided_convention_not_bare_none():
    """A minimal report (no events, no oversight/automation_origin state)
    produced eleven stage rows containing raw `None` -- Codex's own
    reproduction. Every stage this run never touched must render 'Not
    provided' for oversight_mode/automation_origin instead."""
    cfg = ReviewConfig(title="t")
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    stage_table = md.split("## Stage inventory")[1].split("##")[0]
    assert "None" not in stage_table
    assert "Not provided" in stage_table


def test_markdown_m9_r2_uses_not_provided_convention_not_bare_none():
    """Codex's own reproduction: `reference_standard: None` (and the
    n_attempted/n_valid_pairs/n_failed triple) must use the 'Not provided'
    convention when screening QC never ran -- JSON keeps typed `None`."""
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    assert d["m9_r2"]["reference_standard"] is None  # JSON keeps typed None
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    m9_block = md.split("## M9/R2")[1].split("##")[0]
    assert "reference_standard: None" not in m9_block
    assert "reference_standard: Not provided" in m9_block
    assert "n_attempted / n_valid_pairs / n_failed: Not provided / Not provided / Not provided" in m9_block


# ---------------------------------------------------------------------------
# Codex round-1 review, finding 5 (M8): full resolution audit-trail export
# ---------------------------------------------------------------------------

def test_m8_audit_trails_exports_full_screening_resolution_contents():
    cfg = ReviewConfig(title="t")
    resolution = {"record_id": "r1", "decision": "include", "reviewer": "alice",
                  "note": "clearly relevant", "resolved_at": "2026-08-01T00:00:00Z",
                  "resolution_id": "res-1", "reviewed_baseline_decision_id": "dec-0"}
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, screening_review_resolutions=[resolution]))
    assert d["m8_audit_trails"]["source_kind"]["screening_review_resolutions"] == "graph_state"
    assert d["m8_audit_trails"]["source_kind"]["extraction_review_resolutions"] == "durable_reverified"
    assert d["m8_audit_trails"]["source_kind"]["quality_appraisal_resolutions"] == "durable_reverified"
    assert d["m8_audit_trails"]["source_kind"]["quality_appraisal_confirmations"] == "durable_reverified"
    assert d["m8_audit_trails"]["screening_review_resolutions"] == [resolution]
    assert d["m8_audit_trails"]["full_text_review_resolutions"] == []
    # Codex round-11 review, finding 2: extraction/QA bundles don't exist in
    # this empty-snapshot fixture -- their audit trails must be None (not
    # []), the established "not applicable" convention.
    assert d["m8_audit_trails"]["extraction_review_resolutions"] is None
    assert d["m8_audit_trails"]["quality_appraisal_resolutions"] is None
    assert d["m8_audit_trails"]["quality_appraisal_confirmations"] is None


def test_m8_audit_trails_exports_full_text_resolution_contents():
    cfg = ReviewConfig(title="t")
    resolution = {"record_id": "r2", "stage": "ft_screening", "action": "exclude", "reviewer": "bob",
                  "note": "wrong population", "resolved_at": "2026-08-02T00:00:00Z", "reason": "wrong_population"}
    d = _render_ai_traceability_dict(_empty_snapshot(cfg, full_text_review_resolutions=[resolution]))
    assert d["m8_audit_trails"]["full_text_review_resolutions"] == [resolution]


def test_m8_audit_trails_exports_extraction_and_qa_resolutions_when_bundles_present():
    """Codex round-11 review, finding 2: M8 previously exported only 2 of
    the 4 review systems' resolution/confirmation audit trails the base
    plan's crosswalk requires (quality_appraisal_synthesis_plan_2026-08-22.md:663)
    despite being labeled 'covered'. Extraction/QA data is already present
    in the verified bundles -- this proves it's now actually surfaced."""
    cfg = ReviewConfig(title="t")
    extraction_resolution = {"record_id": "r1", "field_id": "sample_size", "outcome": "extracted",
                              "reviewer": "carol", "note": "confirmed from table 2"}
    qa_resolution = {"record_id": "r1", "field_id": "randomization", "status_or_answer": "yes",
                      "reviewer": "dave", "note": "explicit in methods"}
    qa_confirmation = {"record_id": "r1", "field_id": "blinding", "confirmed_by": "erin"}
    extraction_bundle = _fake_bundle(resolutions=[extraction_resolution])
    qa_bundle = _fake_bundle(resolutions=[qa_resolution], confirmations=[qa_confirmation])
    d = _render_ai_traceability_dict(
        _empty_snapshot(cfg, extraction=extraction_bundle, quality_appraisal=qa_bundle))
    m8 = d["m8_audit_trails"]
    assert m8["extraction_review_resolutions"] == [extraction_resolution]
    assert m8["quality_appraisal_resolutions"] == [qa_resolution]
    assert m8["quality_appraisal_confirmations"] == [qa_confirmation]


def test_m8_audit_trails_extraction_and_qa_not_applicable_when_bundles_absent():
    cfg = ReviewConfig(title="t")
    d = _render_ai_traceability_dict(_empty_snapshot(cfg))
    m8 = d["m8_audit_trails"]
    assert m8["extraction_review_resolutions"] is None
    assert m8["quality_appraisal_resolutions"] is None
    assert m8["quality_appraisal_confirmations"] is None


def test_m8_markdown_renders_all_four_systems_and_not_applicable_convention():
    cfg = ReviewConfig(title="t")
    extraction_bundle = _fake_bundle(resolutions=[{"record_id": "r1", "field_id": "f1", "outcome": "extracted"}])
    md = _render_ai_traceability_md(_empty_snapshot(cfg, extraction=extraction_bundle))
    assert "extraction_review_resolutions (1 entries):" in md
    assert "quality_appraisal_resolutions (not applicable):" in md
    assert "quality_appraisal_confirmations (not applicable):" in md
    assert "Not applicable (no extraction/quality-appraisal run this review)." in md


def test_m8_audit_trails_rendered_in_markdown():
    cfg = ReviewConfig(title="t")
    resolution = {"record_id": "r1", "decision": "include", "reviewer": "alice",
                  "note": "clearly relevant", "resolved_at": "2026-08-01T00:00:00Z",
                  "resolution_id": "res-1", "reviewed_baseline_decision_id": None}
    md = _render_ai_traceability_md(_empty_snapshot(cfg, screening_review_resolutions=[resolution]))
    assert "M8 -- Human-AI interaction" in md
    assert "reviewer='alice'" in md
    assert "resolved_at='2026-08-01T00:00:00Z'" in md


def test_m8_audit_trails_empty_shows_none_this_run():
    cfg = ReviewConfig(title="t")
    md = _render_ai_traceability_md(_empty_snapshot(cfg))
    assert "None this run." in md
