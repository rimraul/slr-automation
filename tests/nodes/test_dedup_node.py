"""Offline tests for slr_pipeline.nodes.dedup.dedup_node -- Stage 3 node
instrumentation (the Stage 3 node-instrumentation design record): one
`non_llm_tool` AIUseEvent per run, one SelectionDecisionRecord per record
(decision_maker="rule_based_automation" always), producer_artifact_id ==
that event's own event_id (no interrupt loop, no A0->H hazard for this
stage)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore
from slr_pipeline.config import ReviewConfig
from slr_pipeline.selection_ledger import SelectionDecisionStore
from slr_pipeline.state import Record, initial_state

import slr_pipeline.nodes.dedup as dedup_module
from slr_pipeline.nodes.dedup import dedup_node


def _config(tmp_path) -> ReviewConfig:
    return ReviewConfig(title="Test review", output_dir=str(tmp_path / "outputs"), research_questions=["q"])


def _record(**overrides) -> Record:
    defaults = dict(id=f"rec-{overrides.get('source_id', '0')}", source="test", title="A study", abstract="abc")
    defaults.update(overrides)
    return Record(**defaults)


def _state(cfg: ReviewConfig, records: list[Record]) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["raw_records"] = [r.to_dict() for r in records]
    st["review_run_id"] = "dedup-test-run"
    return st


def test_dedup_node_success_path_emits_one_event_and_per_record_ledger_entries(tmp_path, monkeypatch):
    unique = [_record(source_id="1"), _record(source_id="2", title="Different paper")]
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda records, cfg: unique)

    cfg = _config(tmp_path)
    records = [_record(source_id="1"), _record(source_id="2", title="Different paper"), _record(source_id="3", title="A study")]
    state = _state(cfg, records)

    result = dedup_node(state)

    assert {r["id"] for r in result["deduplicated_records"]} == {r.id for r in unique}
    assert result["oversight_mode"]["dedup"] == "automation_unreviewed"
    assert result["automation_origin"]["dedup"] == "rule_based_automation"

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id="dedup-test-run")
    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].call_status == "success"
    assert events[0].tool_kind == "non_llm_tool"
    assert "sentence-transformers" in events[0].component_versions

    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id="dedup-test-run")
    tails = ledger_store.project_current_tails()
    unique_ids = {r.id for r in unique}
    for record in records:
        tail = tails[(record.id, "dedup")]
        assert tail.decision_maker == "rule_based_automation"
        assert tail.decision == ("unique" if record.id in unique_ids else "duplicate")
        assert tail.producer_artifact_id == events[0].event_id


def test_dedup_node_fallback_path_emits_heuristic_fallback_event_with_failure_detail(tmp_path, monkeypatch):
    def _raise(records, cfg):
        raise RuntimeError("sentence-transformers unavailable")
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", _raise)

    cfg = _config(tmp_path)
    records = [_record(source_id="1")]
    state = _state(cfg, records)

    result = dedup_node(state)
    assert len(result["deduplicated_records"]) == 1

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id="dedup-test-run")
    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].call_status == "heuristic_fallback"
    assert "sentence-transformers unavailable" in events[0].failure_detail
    # F21: the failure branch never attempts the sentence_transformers
    # import at all -- component_versions is unconditionally {} on
    # fallback, tolerating an unimportable package without a second crash.
    assert events[0].component_versions == {}


def test_dedup_node_membership_governs_decision_not_is_duplicate_field(tmp_path, monkeypatch):
    """_dedup_with_pipeline()'s own return value is the sole source of
    truth for duplicate/unique -- even if a returned "unique" record still
    has is_duplicate=True set on it (a pre-existing, separately-tracked
    gap in the underlying pipeline), the ledger decision must follow
    membership in the returned set, not that stale field."""
    kept = _record(source_id="1")
    kept.is_duplicate = True   # deliberately inconsistent with membership
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda records, cfg: [kept])

    cfg = _config(tmp_path)
    records = [_record(source_id="1"), _record(source_id="2")]
    state = _state(cfg, records)
    dedup_node(state)

    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id="dedup-test-run")
    tails = ledger_store.project_current_tails()
    assert tails[("rec-1", "dedup")].decision == "unique"
    assert tails[("rec-2", "dedup")].decision == "duplicate"


def test_dedup_node_idempotent_rerun_with_identical_input(tmp_path, monkeypatch):
    unique = [_record(source_id="1")]
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda records, cfg: unique)

    cfg = _config(tmp_path)
    records = [_record(source_id="1")]
    state = _state(cfg, records)

    dedup_node(state)
    dedup_node(state)   # a genuine second run with identical input

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id="dedup-test-run")
    # Two DIFFERENT events are expected here (dedup has no checkpoint/cache
    # -- each run makes a real call and attempt increments), but the
    # LEDGER stays a clean two-entry chain (harmless re-decision), never a
    # fork or duplicate-decision error, since there's no human entry here
    # to ever be clobbered by a redundant re-decision (unlike screening).
    events = event_store.load_all()
    assert len(events) == 2

    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id="dedup-test-run")
    chain = ledger_store.load_key(record_id="rec-1", stage="dedup")
    assert len(chain) == 2
    tails = ledger_store.project_current_tails()
    assert tails[("rec-1", "dedup")].decision == "unique"
