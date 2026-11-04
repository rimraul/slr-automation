"""Proves graphs/routing.py's conditional gate stops each REAL compiled
graph (scoping/rapid/slr) before downstream analysis while
full_text_review_queue is non-empty, and lets it proceed once empty
(codex review, Phase B finding 1 -- "unresolved studies disappear from
the analyzed corpus while reporting still produces apparently final
outputs").

Every node in the graph EXCEPT the routing edge itself is replaced with a
recording stub -- this test is about the WIRING (does the compiled graph
object actually stop/continue where make_post_fulltext_router() says it
should), not about re-testing search/dedup/screening/fulltext's own
logic, which are covered elsewhere. Stubs are installed on each graph
MODULE's own namespace (slr_pipeline.graphs.scoping/slr_pipeline.graphs.rapid/slr_pipeline.graphs.slr), not
nodes.fulltext etc., because build_*_graph() resolves those names from
its own module globals at call time.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from langgraph.config import get_config

from slr_pipeline.config import ReviewConfig
from slr_pipeline.state import Record, initial_state

from slr_pipeline.nodes.fulltext import resolve_full_text_review_queue

GRAPH_CASES = [
    ("slr_pipeline.graphs.scoping", "build_scoping_graph", "extraction_node"),
    ("slr_pipeline.graphs.rapid", "build_rapid_graph", "extraction_node"),
    ("slr_pipeline.graphs.slr", "build_slr_graph", "extraction_node"),
]


def _install_stub_nodes(monkeypatch, module, calls, *, queue_after_fulltext):
    """Replace every `*_node` name in `module`'s own namespace with a
    stub recording its own name in `calls`. `fulltext_node`'s stub is
    the only one that returns anything meaningful: the review-queue
    state this test wants to drive the routing decision with."""

    def _make_stub(name):
        def _stub(state):
            calls.append(name)
            return {}
        return _stub

    def _make_fulltext_stub():
        def _stub(state):
            calls.append("fulltext_node")
            # Stage 3: resolve_full_text_review_queue() now needs
            # review_run_id in state (mirrors the real fulltext_node's own
            # resolve_review_run_id() call) -- this stub is invoked inside
            # a real graph.invoke(), so an active LangGraph config exists.
            return {
                "full_text_review_queue": list(queue_after_fulltext),
                "requires_human_review": bool(queue_after_fulltext),
                "full_text_records": [],
                "review_run_id": get_config()["configurable"]["thread_id"],
            }
        return _stub

    for name in list(vars(module)):
        if not name.endswith("_node"):
            continue
        monkeypatch.setattr(module, name, _make_fulltext_stub() if name == "fulltext_node" else _make_stub(name))


@pytest.mark.parametrize("graph_module_name,builder_name,first_downstream", GRAPH_CASES)
def test_graph_stops_before_downstream_while_queue_nonempty(monkeypatch, graph_module_name, builder_name, first_downstream):
    mod = importlib.import_module(graph_module_name)
    calls: list[str] = []
    _install_stub_nodes(
        monkeypatch, mod, calls,
        queue_after_fulltext=[{"stage": "ft_screening", "record_id": "rec-pending"}],
    )
    graph = getattr(mod, builder_name)()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    result = graph.invoke(state, {"configurable": {"thread_id": f"test-{graph_module_name}"}})

    assert "fulltext_node" in calls
    assert first_downstream not in calls
    assert "synthesis_node" not in calls
    assert "reporting_node" not in calls
    assert result["full_text_review_queue"]
    assert result["requires_human_review"] is True


@pytest.mark.parametrize("graph_module_name,builder_name,first_downstream", GRAPH_CASES)
def test_graph_proceeds_to_downstream_when_queue_empty(monkeypatch, graph_module_name, builder_name, first_downstream):
    mod = importlib.import_module(graph_module_name)
    calls: list[str] = []
    _install_stub_nodes(monkeypatch, mod, calls, queue_after_fulltext=[])
    graph = getattr(mod, builder_name)()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    result = graph.invoke(state, {"configurable": {"thread_id": f"test-{graph_module_name}-proceed"}})

    assert "fulltext_node" in calls
    assert first_downstream in calls
    assert "synthesis_node" in calls
    assert "reporting_node" in calls
    assert result["full_text_review_queue"] == []
    assert result["requires_human_review"] is False


# ---------------------------------------------------------------------------
# Resolution/resume path: resolve the queue, update the SAME checkpointed
# thread, resume -- fulltext_node itself is never re-run, only the routing
# decision changes.
# ---------------------------------------------------------------------------


def test_graph_resumes_past_the_gate_after_resolving_the_queue(monkeypatch):
    """Genuine resolver -> update_state() -> resume integration test
    (codex review, Phase B round-2: the previous version manually cleared
    full_text_review_queue instead of calling
    resolve_full_text_review_queue(), which would have hidden a real bug
    in the resolver itself). Uses a real stage="ft_screening" queue entry
    -- complete with the preserved mutated `record` (full_text_path set)
    -- and the real resolver to produce the state update."""
    mod = importlib.import_module("slr_pipeline.graphs.scoping")
    calls: list[str] = []

    pending_record = Record(
        id="rec-pending", source="test", source_id="pending", title="Pending study",
        full_text_path="/pdfs/rec-pending.pdf", full_text_status="parsed",
    )
    queue_entry = {
        "stage": "ft_screening", "record_id": "rec-pending", "record": pending_record.to_dict(),
        "reviewed_baseline_decision_id": None,
    }
    _install_stub_nodes(monkeypatch, mod, calls, queue_after_fulltext=[queue_entry])
    graph = mod.build_scoping_graph()
    config = {"configurable": {"thread_id": "resume-test"}}

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    first = graph.invoke(state, config)
    assert "extraction_node" not in calls
    assert first["full_text_review_queue"]

    # Human resolves the queue via the REAL resolver -- update the SAME
    # thread's checkpoint with its output, never re-invoke from scratch
    # with a fresh state.
    resolution_update = resolve_full_text_review_queue(
        first,
        {"rec-pending": {
            "action": "include", "reviewer": "R. Rauleckas",
            "note": "manual full-text read confirms eligibility", "resolved_at": "2026-08-21T12:00:00Z",
        }},
    )
    assert resolution_update["full_text_review_queue"] == []
    assert resolution_update["full_text_records"][0]["full_text_path"] == "/pdfs/rec-pending.pdf"

    graph.update_state(config, resolution_update)
    second = graph.invoke(None, config)

    assert calls.count("fulltext_node") == 1  # fulltext itself was NOT re-run
    assert "extraction_node" in calls
    assert "reporting_node" in calls
    assert second["full_text_review_queue"] == []
    assert second["full_text_records"][0]["id"] == "rec-pending"
    assert second["full_text_review_resolutions"][0]["action"] == "include"
