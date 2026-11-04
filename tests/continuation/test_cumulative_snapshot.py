"""Unit E -- `slr_pipeline.continuation.cumulative_snapshot`."""
from __future__ import annotations

import pytest

import slr_pipeline.reference_bundle as rb
from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.cumulative_snapshot import (
    _PARENT_ONLY_EVENT_STAGES,
    build_cumulative_reporting_snapshot,
)
from slr_pipeline.continuation.partition import ContinuationPartition
from slr_pipeline.graphs.scoping_continuation import build_scoping_continuation_graph
from slr_pipeline.selection_ledger import _tails_by_key
from slr_pipeline.state import initial_state
from slr_pipeline.track3_run import Track3ConfigError
from tests.continuation.conftest import build_parent_bundle


def _ft_citations(parent_snapshot):
    tails = _tails_by_key(
        [vf.model for vf in parent_snapshot.ledger_records if vf.model.stage == "full_text_eligibility"])
    return [
        {"record_id": record_id, "decision_id": rec.decision_id, "content_hash": rec.content_hash}
        for (record_id, _stage), rec in tails.items()
    ]


def _continuation_cfg(tmp_path) -> ReviewConfig:
    return ReviewConfig(title="Unit E continuation", output_dir=str(tmp_path / "continuation_outputs"))


def _continuation_state(cfg: ReviewConfig, thread_id: str, monkeypatch) -> dict:
    """Runs the real continuation graph up through `synthesis` (with
    "reporting" swapped for a passthrough) so the returned state carries
    every field `capture_reporting_snapshot()` requires (`synthesis`,
    `extraction`, `quality_appraisal`, ...) exactly as a genuine
    continuation run would produce them -- a hand-built `initial_state()`
    dict alone is missing all of these (no node ever populated them). The
    continuation's own `review_run_id` (resolved from `thread_id`, per
    `resolve_review_run_id()`) is intentionally independent of the parent
    bundle's own `review_run_id` -- a continuation is a genuinely new run."""

    def _passthrough(state):
        return {}

    monkeypatch.setattr(
        "slr_pipeline.graphs.scoping_continuation.continuation_reporting_node", _passthrough)
    graph = build_scoping_continuation_graph()

    partition = ContinuationPartition(
        preserved_included_ids=[], preserved_excluded_ids=[], retrieved_ids=[], not_retrieved_ids=[])
    seeded = dict(initial_state(cfg.to_dict()))
    seeded["included_records"] = []
    seeded["continuation_partition"] = partition.model_dump()
    seeded["preserved_included_records"] = []
    seeded["retrieved_pdf_manifest"] = {}
    seeded["parent_bundle_payload_root"] = ""
    seeded["preserved_decision_citations"] = []
    run_config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(seeded, run_config)


def _build_snapshot(tmp_path, monkeypatch, *, citations_fn=lambda citations: citations):
    bundle, parent_state = build_parent_bundle(tmp_path, monkeypatch)
    parent_snapshot = rb.from_bundle(bundle)
    citations = citations_fn(_ft_citations(parent_snapshot))

    continuation_cfg = _continuation_cfg(tmp_path)
    thread_id = "unit-e-continuation-test-run"
    continuation_state = _continuation_state(continuation_cfg, thread_id, monkeypatch)

    snapshot = build_cumulative_reporting_snapshot(
        parent_bundle=bundle, continuation_state=continuation_state, cfg=continuation_cfg,
        review_run_id=thread_id, preserved_decision_citations=citations,
    )
    return snapshot, parent_snapshot, citations


def test_build_cumulative_reporting_snapshot_unions_parent_corpus(tmp_path, monkeypatch):
    snapshot, parent_snapshot, citations = _build_snapshot(tmp_path, monkeypatch)

    assert [r["id"] for r in snapshot.raw_records.model] == [r["id"] for r in parent_snapshot.raw_records.model]
    assert [r.id for r in snapshot.screening_records] == [r.id for r in parent_snapshot.screening_records]
    dedup_tails = _tails_by_key([vf.model for vf in snapshot.ledger_records if vf.model.stage == "dedup"])
    parent_dedup_tails = _tails_by_key(
        [vf.model for vf in parent_snapshot.ledger_records if vf.model.stage == "dedup"])
    assert set(dedup_tails) == set(parent_dedup_tails)
    ft_tails = _tails_by_key(
        [vf.model for vf in snapshot.ledger_records if vf.model.stage == "full_text_eligibility"])
    assert {record_id for record_id, _stage in ft_tails} == {c["record_id"] for c in citations}


def test_build_cumulative_reporting_snapshot_events_union_excludes_synthesis(tmp_path, monkeypatch):
    snapshot, parent_snapshot, _citations = _build_snapshot(tmp_path, monkeypatch)

    parent_synthesis_stage_present = any(vf.model.stage.startswith("synthesis_") for vf in parent_snapshot.events)
    assert parent_synthesis_stage_present, "fixture should have at least one parent synthesis-stage event"
    assert not any(vf.model.stage.startswith("synthesis_") for vf in snapshot.events)

    parent_only_events_present = [vf for vf in parent_snapshot.events if vf.model.stage in _PARENT_ONLY_EVENT_STAGES]
    if parent_only_events_present:
        result_event_ids = {vf.model.event_id for vf in snapshot.events}
        assert any(vf.model.event_id in result_event_ids for vf in parent_only_events_present)


def test_build_cumulative_reporting_snapshot_tampered_citation_raises(tmp_path, monkeypatch):
    def _tamper(citations):
        assert citations, "fixture must have at least one full_text_eligibility decision to tamper"
        tampered = [dict(citations[0])]
        tampered[0]["content_hash"] = "0" * 64
        return tampered

    with pytest.raises(Track3ConfigError, match="tampered citation"):
        _build_snapshot(tmp_path, monkeypatch, citations_fn=_tamper)


def test_build_cumulative_reporting_snapshot_unknown_decision_id_raises(tmp_path, monkeypatch):
    def _bogus(citations):
        assert citations
        bogus = [dict(citations[0])]
        bogus[0]["decision_id"] = "does-not-exist"
        return bogus

    with pytest.raises(Track3ConfigError, match="no full_text_eligibility decision"):
        _build_snapshot(tmp_path, monkeypatch, citations_fn=_bogus)


def test_build_cumulative_reporting_snapshot_duplicate_citation_raises(tmp_path, monkeypatch):
    def _duplicate(citations):
        assert citations
        return citations + [dict(citations[0])]

    with pytest.raises(Track3ConfigError, match="cited more than once"):
        _build_snapshot(tmp_path, monkeypatch, citations_fn=_duplicate)
