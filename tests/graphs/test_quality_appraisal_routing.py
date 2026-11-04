"""Proves graphs/routing.py's make_post_quality_appraisal_router() gate
stops each REAL compiled rapid/SLR graph before synthesis/reporting while
state["quality_appraisal_review_queue"] is non-empty, and lets it proceed
(and resume) once empty -- Section 1.10/1.11 of
quality_appraisal_synthesis_plan_2026-08-22.md, following
tests/graphs/test_extraction_routing.py's exact structure and
stub-installation convention.

Also proves the new pipeline order (Section 1.9): fulltext -> extraction
-> quality_appraisal -> synthesis, not the prior quality_appraisal ->
extraction drift.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from slr_pipeline.config import ReviewConfig
from slr_pipeline.state import initial_state

from slr_pipeline.nodes.quality_appraisal import confirm_quality_appraisal_provisional_item

GRAPH_CASES = [
    ("slr_pipeline.graphs.rapid", "build_rapid_graph"),
    ("slr_pipeline.graphs.slr", "build_slr_graph"),
]


def _install_stub_nodes(monkeypatch, module, calls, *, qa_queue_after):
    def _make_stub(name):
        def _stub(state):
            calls.append(name)
            return {}
        return _stub

    def _make_fulltext_stub():
        def _stub(state):
            calls.append("fulltext_node")
            return {"full_text_review_queue": [], "requires_human_review": False, "full_text_records": []}
        return _stub

    def _make_extraction_stub():
        def _stub(state):
            calls.append("extraction_node")
            return {"extraction_review_queue": [], "requires_human_review": False, "extraction_results": []}
        return _stub

    def _make_quality_appraisal_stub():
        def _stub(state):
            calls.append("quality_appraisal_node")
            return {
                "quality_appraisal_review_queue": list(qa_queue_after),
                "requires_human_review": bool(qa_queue_after),
                "quality_appraisal_results": [],
                "quality_appraisal_run_id": "qarun-test" if qa_queue_after else None,
            }
        return _stub

    for name in list(vars(module)):
        if not name.endswith("_node"):
            continue
        if name == "fulltext_node":
            monkeypatch.setattr(module, name, _make_fulltext_stub())
        elif name == "extraction_node":
            monkeypatch.setattr(module, name, _make_extraction_stub())
        elif name == "quality_appraisal_node":
            monkeypatch.setattr(module, name, _make_quality_appraisal_stub())
        else:
            monkeypatch.setattr(module, name, _make_stub(name))


@pytest.mark.parametrize("graph_module_name,builder_name", GRAPH_CASES)
def test_graph_stops_before_synthesis_while_qa_queue_nonempty(monkeypatch, graph_module_name, builder_name):
    mod = importlib.import_module(graph_module_name)
    calls: list[str] = []
    _install_stub_nodes(monkeypatch, mod, calls, qa_queue_after=[{"record_id": "rec-a", "field_id": "clear_research_question", "queue_kind": "resolution", "reason": "no_evidence_found"}])
    graph = getattr(mod, builder_name)()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    result = graph.invoke(state, {"configurable": {"thread_id": f"qa-test-{graph_module_name}"}})

    assert "extraction_node" in calls
    assert "quality_appraisal_node" in calls
    assert "synthesis_node" not in calls
    assert "reporting_node" not in calls
    assert result["quality_appraisal_review_queue"]
    assert result["requires_human_review"] is True


@pytest.mark.parametrize("graph_module_name,builder_name", GRAPH_CASES)
def test_graph_proceeds_to_synthesis_when_qa_queue_empty(monkeypatch, graph_module_name, builder_name):
    mod = importlib.import_module(graph_module_name)
    calls: list[str] = []
    _install_stub_nodes(monkeypatch, mod, calls, qa_queue_after=[])
    graph = getattr(mod, builder_name)()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    result = graph.invoke(state, {"configurable": {"thread_id": f"qa-test-{graph_module_name}-proceed"}})

    assert "quality_appraisal_node" in calls
    assert "synthesis_node" in calls
    assert "reporting_node" in calls
    assert result["quality_appraisal_review_queue"] == []
    assert result["requires_human_review"] is False


def test_extraction_runs_before_quality_appraisal(monkeypatch):
    mod = importlib.import_module("slr_pipeline.graphs.rapid")
    calls: list[str] = []
    _install_stub_nodes(monkeypatch, mod, calls, qa_queue_after=[])
    graph = mod.build_rapid_graph()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    graph.invoke(state, {"configurable": {"thread_id": "qa-order-test"}})

    assert calls.index("extraction_node") < calls.index("quality_appraisal_node")
    assert calls.index("quality_appraisal_node") < calls.index("synthesis_node")


# ---------------------------------------------------------------------------
# Resolution/resume path: a graph stopped at END with only confirmation
# entries queued genuinely un-blocks once every one of them is confirmed
# via confirm_quality_appraisal_provisional_item() (Section 1.11's own
# direct regression for Codex point 1 -- the router genuinely un-blocks,
# not just clears a flag with the queue left stale).
# ---------------------------------------------------------------------------

def _real_provisional_qa_run(monkeypatch, tmp_path):
    import dataclasses

    import slr_pipeline.nodes.quality_appraisal as real_qa_mod
    from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
    from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
    from slr_pipeline.state import Record

    chunk = ParsedChunk(chunk_id="rec-pending-chunk-0000", order_index=0, section="Methods", page_min=1, page_max=1, text="The study explicitly states no research question was formulated.")
    monkeypatch.setattr(real_qa_mod, "parse_pdf", lambda *a, **k: ParsedDocument(document_sha256="a" * 64, parser_name="docling", parser_version="2.61.2", parser_options=DoclingParserOptions(), chunks=(chunk,)))

    def _fake(**kw):
        return real_qa_mod.RawTeachingBatchResponse(items={
            i["item_id"]: real_qa_mod.RawTeachingItemBatchResponse(outcome="no", rationale="explicit", evidence=[real_qa_mod.RawAppraisalEvidenceItem(chunk_id=chunk.chunk_id, quotation=chunk.text)])
            for i in real_qa_mod.TEACHING_APPRAISAL_ITEMS
        })
    monkeypatch.setattr(real_qa_mod, "run_openai_teaching_batch", _fake)

    model_config = OpenAIExtractionConfig(model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra")
    pdf_path = tmp_path / "rec-pending.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    from slr_pipeline.config import QualityAppraisalConfig
    cfg = ReviewConfig(
        title="t", review_type="rapid", output_dir=str(tmp_path / "out"),
        quality_appraisal=QualityAppraisalConfig(checklist_id="teaching_appraisal_v1", negative_high_risk_policy="require_human_confirmation"),
        quality_appraisal_model_config=dataclasses.asdict(model_config),
    )
    pending_record = Record(id="rec-pending", source="test", source_id="pending", title="Pending study", full_text_path=str(pdf_path), full_text_status="parsed")
    qa_state = initial_state(cfg.to_dict())
    qa_state["full_text_records"] = [pending_record.to_dict()]
    qa_output = real_qa_mod.quality_appraisal_node(qa_state)
    assert qa_output["quality_appraisal_review_queue"], "setup failed: expected a genuinely queued confirmation entry"
    return cfg, pending_record, qa_output


def test_graph_resumes_past_qa_gate_after_confirming_queue(monkeypatch, tmp_path):
    mod = importlib.import_module("slr_pipeline.graphs.rapid")
    cfg, pending_record, qa_output = _real_provisional_qa_run(monkeypatch, tmp_path)

    calls_local: list[str] = []

    def _make_stub(name):
        def _stub(state):
            calls_local.append(name)
            return {}
        return _stub

    def _make_fulltext_stub():
        def _stub(state):
            calls_local.append("fulltext_node")
            return {"full_text_review_queue": [], "requires_human_review": False, "full_text_records": [pending_record.to_dict()]}
        return _stub

    def _make_extraction_stub():
        def _stub(state):
            calls_local.append("extraction_node")
            return {"extraction_review_queue": [], "requires_human_review": False, "extraction_results": []}
        return _stub

    def _make_qa_stub():
        def _stub(state):
            calls_local.append("quality_appraisal_node")
            return dict(qa_output)
        return _stub

    for name in list(vars(mod)):
        if not name.endswith("_node"):
            continue
        if name == "fulltext_node":
            monkeypatch.setattr(mod, name, _make_fulltext_stub())
        elif name == "extraction_node":
            monkeypatch.setattr(mod, name, _make_extraction_stub())
        elif name == "quality_appraisal_node":
            monkeypatch.setattr(mod, name, _make_qa_stub())
        else:
            monkeypatch.setattr(mod, name, _make_stub(name))

    graph = mod.build_rapid_graph()
    config = {"configurable": {"thread_id": "resume-qa-test"}}

    state = initial_state(cfg.to_dict())
    first = graph.invoke(state, config)
    assert "synthesis_node" not in calls_local
    assert first["quality_appraisal_review_queue"]

    confirmations = {
        (e["record_id"], e["field_id"]): {"reviewer": "R. Rauleckas", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}
        for e in first["quality_appraisal_review_queue"]
    }
    confirm_update = confirm_quality_appraisal_provisional_item(first, confirmations)
    assert confirm_update["quality_appraisal_review_queue"] == []

    graph.update_state(config, confirm_update)
    second = graph.invoke(None, config)

    assert calls_local.count("quality_appraisal_node") == 1  # quality_appraisal itself was NOT re-run
    assert "synthesis_node" in calls_local
    assert "reporting_node" in calls_local
    assert second["quality_appraisal_review_queue"] == []
