"""Unit D -- `slr_pipeline.graphs.scoping_continuation.build_scoping_continuation_graph`.

Two dedicated properties, both mandated by the plan's own Unit D Test
paragraph:
  - restart-safety: invoking the compiled graph twice against the SAME
    fresh thread_id resumes rather than re-running `continuation_fulltext_
    node` a second time (`track3_run_child.py::_invoke_graph`'s own
    `graph.get_state(run_config)` / `graph.invoke(None, run_config)`
    pattern, reused here rather than reinvented).
  - zero-search-backend-dispatch: `nodes.search.search_node`,
    `nodes.dedup.dedup_node`, `nodes.screening.screening_node` are
    monkeypatched to raise if invoked; a real end-to-end continuation
    graph run never triggers any of them (call-count/mock-based, not
    import-absence -- `nodes/__init__.py` unconditionally imports all
    three at package-init time, making import-absence untestable).

Every test seeds `included_records` directly and an already-empty
`continuation_partition` (all `not_retrieved_ids`) so `continuation_
fulltext_node` runs its cheapest real path (`confirmed == []`, no
FT_SCREENING call) and `extraction_node`/`synthesis_node` both take their
own documented empty-input fast paths -- no PDF, no OpenAI-shaped mock,
no network of any kind anywhere in this file.
"""
from __future__ import annotations

import inspect
import uuid

import pytest

import slr_pipeline.reference_bundle as rb
from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.partition import ContinuationPartition
from slr_pipeline.graphs.scoping_continuation import (
    ContinuationReviewState,
    build_scoping_continuation_graph,
)
from slr_pipeline.selection_ledger import _tails_by_key
from slr_pipeline.state import Record, initial_state
from tests.continuation.conftest import build_parent_bundle


def _config(tmp_path) -> ReviewConfig:
    return ReviewConfig(title="Test continuation graph", output_dir=str(tmp_path / "outputs"))


def _ft_citations(parent_snapshot):
    tails = _tails_by_key(
        [vf.model for vf in parent_snapshot.ledger_records if vf.model.stage == "full_text_eligibility"])
    return [
        {"record_id": record_id, "decision_id": rec.decision_id, "content_hash": rec.content_hash}
        for (record_id, _stage), rec in tails.items()
    ]


def _seed_parent_bundle(tmp_path, monkeypatch):
    """Builds the genuine parent bundle -- must run BEFORE any test
    monkeypatches `search_node`/`dedup_node`/`screening_node` to raise,
    since this itself dispatches the real (offline, fixture-backed)
    versions of those very nodes to produce the parent's own corpus."""
    bundle, _parent_state = build_parent_bundle(tmp_path, monkeypatch)
    parent_snapshot = rb.from_bundle(bundle)
    return str(bundle.payload_root), _ft_citations(parent_snapshot)


def _seeded_state(cfg: ReviewConfig, tmp_path, monkeypatch) -> dict:
    """Unit E requires a genuine, `open_bundle()`-loadable parent bundle
    (`parent_bundle_payload_root`) and its own full_text_eligibility
    ledger decisions cited (`preserved_decision_citations`) -- `r-missing`
    itself is a synthetic id absent from the parent's own corpus (Unit D's
    own point: it takes `continuation_fulltext_node`'s cheapest real,
    zero-FT_SCREENING-call path), so citing ALL of the parent's own
    decisions here (its own genuinely preserved corpus) keeps `reporting`'s
    `compute_selection_splits()` self-consistent."""
    payload_root, citations = _seed_parent_bundle(tmp_path, monkeypatch)
    return _seed_state_dict(cfg, payload_root, citations)


def _seed_state_dict(cfg: ReviewConfig, payload_root: str, citations: list) -> dict:
    r_missing = Record(id="r-missing", source="test", source_id="missing", doi=None)
    partition = ContinuationPartition(
        preserved_included_ids=[], preserved_excluded_ids=[], retrieved_ids=[],
        not_retrieved_ids=["r-missing"],
    )
    st = dict(initial_state(cfg.to_dict()))
    st["included_records"] = [r_missing.to_dict()]
    st["continuation_partition"] = partition.model_dump()
    st["preserved_included_records"] = []
    st["retrieved_pdf_manifest"] = {}
    st["parent_bundle_payload_root"] = payload_root
    st["preserved_decision_citations"] = citations
    return st


# ---------------------------------------------------------------------------
# Restart-safety.
# ---------------------------------------------------------------------------


def test_restart_safety_second_invoke_does_not_rerun_fulltext_node(tmp_path, monkeypatch):
    calls = []
    import slr_pipeline.nodes.fulltext_continuation as fulltext_continuation_module

    real_node = fulltext_continuation_module.continuation_fulltext_node

    def _counting_node(state):
        calls.append(1)
        return real_node(state)

    monkeypatch.setattr(
        "slr_pipeline.graphs.scoping_continuation.continuation_fulltext_node", _counting_node,
    )

    cfg = _config(tmp_path)
    state = _seeded_state(cfg, tmp_path, monkeypatch)
    thread_id = f"track3-continuation-test-{uuid.uuid4()}"
    run_config = {"configurable": {"thread_id": thread_id}}

    graph = build_scoping_continuation_graph()
    first = graph.invoke(state, run_config)
    assert len(calls) == 1
    assert first["full_text_records"] == []

    existing = graph.get_state(run_config)
    assert existing.values

    # Second invoke against the SAME thread_id, with input=None -- resumes
    # from the already-terminal checkpoint rather than restarting the graph
    # from its entry node.
    second = graph.invoke(None, run_config)
    assert len(calls) == 1
    assert second["full_text_records"] == []


def test_restart_safety_fresh_thread_id_runs_independently(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    graph = build_scoping_continuation_graph()

    # Separate parent-bundle roots (`build_parent_bundle()` writes a fixed
    # "driver_workspace" dir directly under its own `tmp_path` argument) --
    # each of the two independent continuation runs below needs its own.
    parent_root_a = tmp_path / "parent_a"
    parent_root_a.mkdir()
    parent_root_b = tmp_path / "parent_b"
    parent_root_b.mkdir()

    state_a = _seeded_state(cfg, parent_root_a, monkeypatch)
    config_a = {"configurable": {"thread_id": f"track3-continuation-test-{uuid.uuid4()}"}}
    result_a = graph.invoke(state_a, config_a)
    assert result_a["review_run_id"] == config_a["configurable"]["thread_id"]

    state_b = _seeded_state(cfg, parent_root_b, monkeypatch)
    config_b = {"configurable": {"thread_id": f"track3-continuation-test-{uuid.uuid4()}"}}
    result_b = graph.invoke(state_b, config_b)
    assert result_b["review_run_id"] == config_b["configurable"]["thread_id"]

    assert result_a["review_run_id"] != result_b["review_run_id"]


# ---------------------------------------------------------------------------
# Zero-search-backend-dispatch.
# ---------------------------------------------------------------------------


def test_zero_search_backend_dispatch(tmp_path, monkeypatch):
    # Parent bundle built FIRST -- it genuinely dispatches the real
    # dedup_node/screening_node (offline, fixture-backed) to produce the
    # parent's own corpus, before those are monkeypatched to raise below.
    cfg = _config(tmp_path / "continuation")
    payload_root, citations = _seed_parent_bundle(tmp_path / "parent", monkeypatch)
    state = _seed_state_dict(cfg, payload_root, citations)

    def _boom(*args, **kwargs):
        raise AssertionError("search/dedup/screening must never be dispatched in a continuation graph")

    import slr_pipeline.nodes.search as search_module
    import slr_pipeline.nodes.dedup as dedup_module
    import slr_pipeline.nodes.screening as screening_module

    monkeypatch.setattr(search_module, "search_node", _boom)
    monkeypatch.setattr(dedup_module, "dedup_node", _boom)
    monkeypatch.setattr(screening_module, "screening_node", _boom)

    run_config = {"configurable": {"thread_id": f"track3-continuation-test-{uuid.uuid4()}"}}

    graph = build_scoping_continuation_graph()
    result = graph.invoke(state, run_config)

    assert result["full_text_records"] == []


def test_module_source_never_references_search_dedup_screening_node_symbols():
    import slr_pipeline.graphs.scoping_continuation as module

    source = inspect.getsource(module)
    for forbidden in ("search_node", "dedup_node", "screening_node", "screening_qc_node"):
        assert forbidden not in source


def test_module_source_never_references_fulltext_node_symbol():
    """Distinct from the search/dedup/screening check above -- proves this
    graph's entry point is exclusively Unit C's `continuation_fulltext_
    node`, never the real `fulltext_node` (which would re-run the full
    acquisition/eligibility loop against a different, non-continuation
    contract)."""
    import slr_pipeline.graphs.scoping_continuation as module

    source = inspect.getsource(module)
    assert "fulltext_node" not in source.replace("continuation_fulltext_node", "")


# ---------------------------------------------------------------------------
# ContinuationReviewState schema -- proves the seeded keys actually reach
# the entry node (round-27/round-29 fix: schema, not just seeding, is
# what determines which keys survive `graph.invoke()`'s input mapping).
# ---------------------------------------------------------------------------


def test_extraction_resolution_update_resumes_after_extraction_without_rerun(monkeypatch):
    """Section 3.9: update-as-extraction re-evaluates only the outgoing route."""
    import slr_pipeline.graphs.scoping_continuation as graph_module
    from langgraph.checkpoint.memory import MemorySaver

    calls = {"fulltext": 0, "extraction": 0, "synthesis": 0, "reporting": 0}

    def fulltext(state):
        calls["fulltext"] += 1
        return {"full_text_review_queue": []}

    def extraction(state):
        calls["extraction"] += 1
        return {"extraction_review_queue": [{"record_id": "r", "field_id": "f"}]}

    def synthesis(state):
        calls["synthesis"] += 1
        return {}

    def reporting(state):
        calls["reporting"] += 1
        return {}

    monkeypatch.setattr(graph_module, "continuation_fulltext_node", fulltext)
    monkeypatch.setattr(graph_module, "extraction_node", extraction)
    monkeypatch.setattr(graph_module, "synthesis_node", synthesis)
    monkeypatch.setattr(graph_module, "continuation_reporting_node", reporting)
    graph = graph_module.build_scoping_continuation_graph(checkpointer=MemorySaver())
    run_config = {"configurable": {"thread_id": "resolution-resume-test"}}
    seed = dict(initial_state(ReviewConfig(title="resume").to_dict()))
    seed.update({
        "continuation_partition": {},
        "preserved_included_records": [],
        "retrieved_pdf_manifest": [],
        "parent_bundle_payload_root": "unused",
        "preserved_decision_citations": [],
    })
    first = graph.invoke(seed, run_config)
    assert first["extraction_review_queue"]
    assert calls == {"fulltext": 1, "extraction": 1, "synthesis": 0, "reporting": 0}

    graph.update_state(run_config, {"extraction_review_queue": []}, as_node="extraction")
    resumed = graph.invoke(None, run_config)
    assert resumed["extraction_review_queue"] == []
    assert calls == {"fulltext": 1, "extraction": 1, "synthesis": 1, "reporting": 1}


def test_continuation_review_state_declares_seeded_fields():
    hints = ContinuationReviewState.__annotations__
    for field in ("continuation_partition", "preserved_included_records", "retrieved_pdf_manifest"):
        assert field in hints
