"""Proves graphs/routing.py's make_post_extraction_router() gate stops each
REAL compiled rapid/SLR graph before synthesis/reporting while
state["extraction_review_queue"] is non-empty, and lets it proceed (and
resume) once empty -- the extraction-stage analogue of
tests/graphs/test_fulltext_routing.py's own fulltext gate proof, following
that file's exact structure and stub-installation convention.

Also proves that all three graphs (scoping, rapid, SLR) are wired to
extraction_node, never the legacy charting_node (moved to
examples/legacy/charting.py by Stage 2 of
quality_appraisal_synthesis_plan_2026-08-22.md, "Charting via the
Extraction Engine") -- see tests/graphs/test_scoping_routing.py for
scoping's own dedicated extraction-gate coverage (fulltext -> extraction
-> synthesis, same shape as rapid/SLR).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from slr_pipeline.config import ReviewConfig
from slr_pipeline.state import Record, initial_state

from slr_pipeline.nodes.extraction import resolve_extraction_review_queue

GRAPH_CASES = [
    ("slr_pipeline.graphs.scoping", "build_scoping_graph"),
    ("slr_pipeline.graphs.rapid", "build_rapid_graph"),
    ("slr_pipeline.graphs.slr", "build_slr_graph"),
]


def _install_stub_nodes(monkeypatch, module, calls, *, extraction_queue_after, full_text_records=None):
    """Replace every `*_node` name in `module`'s own namespace with a stub
    recording its own name in `calls` -- same pattern as
    tests/graphs/test_fulltext_routing.py's own `_install_stub_nodes`.
    `fulltext_node`'s stub always reports an EMPTY full_text_review_queue
    (that gate is proven separately) so every case here reaches
    `extraction_node`; `extraction_node`'s stub is the only other one that
    returns anything meaningful."""

    def _make_stub(name):
        def _stub(state):
            calls.append(name)
            return {}
        return _stub

    def _make_fulltext_stub():
        def _stub(state):
            calls.append("fulltext_node")
            return {
                "full_text_review_queue": [], "requires_human_review": False,
                "full_text_records": list(full_text_records or []),
            }
        return _stub

    def _make_extraction_stub():
        def _stub(state):
            calls.append("extraction_node")
            return {
                "extraction_review_queue": list(extraction_queue_after),
                "requires_human_review": bool(extraction_queue_after),
                "extraction_results": [],
            }
        return _stub

    for name in list(vars(module)):
        if not name.endswith("_node"):
            continue
        if name == "fulltext_node":
            monkeypatch.setattr(module, name, _make_fulltext_stub())
        elif name == "extraction_node":
            monkeypatch.setattr(module, name, _make_extraction_stub())
        else:
            monkeypatch.setattr(module, name, _make_stub(name))


@pytest.mark.parametrize("graph_module_name,builder_name", GRAPH_CASES)
def test_graph_stops_before_synthesis_while_extraction_queue_nonempty(monkeypatch, graph_module_name, builder_name):
    mod = importlib.import_module(graph_module_name)
    calls: list[str] = []
    _install_stub_nodes(
        monkeypatch, mod, calls,
        extraction_queue_after=[{"record_id": "rec-pending", "field_id": "study_design", "status": "unclear"}],
    )
    graph = getattr(mod, builder_name)()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    result = graph.invoke(state, {"configurable": {"thread_id": f"extraction-test-{graph_module_name}"}})

    assert "extraction_node" in calls
    assert "synthesis_node" not in calls
    assert "reporting_node" not in calls
    assert result["extraction_review_queue"]
    assert result["requires_human_review"] is True


@pytest.mark.parametrize("graph_module_name,builder_name", GRAPH_CASES)
def test_graph_proceeds_to_synthesis_when_extraction_queue_empty(monkeypatch, graph_module_name, builder_name):
    mod = importlib.import_module(graph_module_name)
    calls: list[str] = []
    _install_stub_nodes(monkeypatch, mod, calls, extraction_queue_after=[])
    graph = getattr(mod, builder_name)()

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    result = graph.invoke(state, {"configurable": {"thread_id": f"extraction-test-{graph_module_name}-proceed"}})

    assert "extraction_node" in calls
    assert "synthesis_node" in calls
    assert "reporting_node" in calls
    assert result["extraction_review_queue"] == []
    assert result["requires_human_review"] is False


def test_slr_graph_with_meta_analysis_also_gated_by_extraction_queue(monkeypatch):
    mod = importlib.import_module("slr_pipeline.graphs.slr")
    calls: list[str] = []
    _install_stub_nodes(
        monkeypatch, mod, calls,
        extraction_queue_after=[{"record_id": "rec-pending", "field_id": "study_design", "status": "unclear"}],
    )
    graph = mod.build_slr_graph(include_meta_analysis=True)

    cfg = ReviewConfig(title="t")
    state = initial_state(cfg.to_dict())
    graph.invoke(state, {"configurable": {"thread_id": "extraction-meta-test"}})

    assert "extraction_node" in calls
    assert "meta_analysis" not in calls
    assert "synthesis_node" not in calls


# ---------------------------------------------------------------------------
# Resolution/resume path: resolve the queue with the REAL resolver, apply
# via update_state() on the same checkpointed thread, resume -- extraction
# itself is never re-run, only the routing decision changes.
# ---------------------------------------------------------------------------

def _real_unclear_extraction_run(monkeypatch, tmp_path):
    """Produces a REAL extraction_node() run (parse_pdf + provider both
    faked, exactly like tests/nodes/test_extraction_node.py's own
    convention) with a genuine on-disk checkpoint/manifest -- needed
    because resolve_extraction_review_queue() now derives/verifies a
    VerifiedRunBundle against ACTUAL Phase A artifacts on disk (codex
    review round-2 finding C1); a purely stub-driven graph run has no
    backing store for that. Returns (cfg, pending_record, ext_output) --
    `cfg` (with extraction_schema/extraction_model_config already set)
    must be reused for the graph's own initial_state so
    state["config"]["extraction_schema"] matches the run that actually
    produced the artifacts on disk."""
    import dataclasses

    import slr_pipeline.extraction.engine as engine_module
    import slr_pipeline.nodes.extraction as real_extraction_mod
    from slr_pipeline.extraction.openai_runner import (
        BatchCallTelemetry, OpenAIExtractionConfig, RawBatchResponse, RawEvidenceItem, RawFieldResponse, RawFinding,
    )
    from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
    from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind

    chunk = ParsedChunk(
        chunk_id="rec-pending-chunk-0000", order_index=0, section="Methods",
        page_min=1, page_max=1, text="Some ambiguous passage about the study design.",
    )
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: ParsedDocument(
        document_sha256="a" * 64, parser_name="docling", parser_version="2.61.2",
        parser_options=DoclingParserOptions(), chunks=(chunk,),
    ))

    def fake_call(*, schema, batch_chunks, config):
        c = batch_chunks[0]
        # A finding whose evidence quotation does NOT verbatim-match the
        # chunk text -> evidence verification fails -> status='unclear'.
        return RawBatchResponse(fields={"study_design": RawFieldResponse(outcome="found", findings=[
            RawFinding(value="RCT", confidence="high", rationale="x",
                       evidence=[RawEvidenceItem(chunk_id=c.chunk_id, quotation="text that does not match verbatim")]),
        ])}), BatchCallTelemetry(provider_model="gpt-5.6-terra", prompt_tokens=1, completion_tokens=1, total_tokens=2)

    monkeypatch.setattr(real_extraction_mod, "run_openai_batch_extraction", fake_call)

    schema = ExtractionSchema(schema_id="s", version="1", fields=[ExtractionFieldSpec(
        field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
        cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )])
    model_config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )

    pdf_path = tmp_path / "rec-pending.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = ReviewConfig(
        title="t", review_type="rapid", output_dir=str(tmp_path / "out"),
        extraction_schema=schema.model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(model_config),
    )
    pending_record = Record(
        id="rec-pending", source="test", source_id="pending", title="Pending study",
        full_text_path=str(pdf_path), full_text_status="parsed",
    )
    ext_state = initial_state(cfg.to_dict())
    ext_state["full_text_records"] = [pending_record.to_dict()]
    ext_output = real_extraction_mod.extraction_node(ext_state)
    assert ext_output["extraction_review_queue"], "setup failed: expected a genuinely queued field"
    return cfg, pending_record, ext_output


def test_graph_resumes_past_extraction_gate_after_resolving_queue(monkeypatch, tmp_path):
    mod = importlib.import_module("slr_pipeline.graphs.rapid")

    cfg, pending_record, ext_output = _real_unclear_extraction_run(monkeypatch, tmp_path)

    calls_local: list[str] = []

    def _make_stub(name):
        def _stub(state):
            calls_local.append(name)
            return {}
        return _stub

    def _make_fulltext_stub():
        def _stub(state):
            calls_local.append("fulltext_node")
            return {
                "full_text_review_queue": [], "requires_human_review": False,
                "full_text_records": [pending_record.to_dict()],
            }
        return _stub

    def _make_extraction_stub():
        def _stub(state):
            calls_local.append("extraction_node")
            # Returns the REAL output from the genuine extraction_node()
            # run above -- the graph-level stub controls call COUNTING
            # (proving extraction is never re-run) without re-exercising
            # extraction_node's own internals, already covered in
            # tests/nodes/test_extraction_node.py.
            return dict(ext_output)
        return _stub

    for name in list(vars(mod)):
        if not name.endswith("_node"):
            continue
        if name == "fulltext_node":
            monkeypatch.setattr(mod, name, _make_fulltext_stub())
        elif name == "extraction_node":
            monkeypatch.setattr(mod, name, _make_extraction_stub())
        else:
            monkeypatch.setattr(mod, name, _make_stub(name))

    graph = mod.build_rapid_graph()
    config = {"configurable": {"thread_id": "resume-extraction-test"}}

    state = initial_state(cfg.to_dict())
    first = graph.invoke(state, config)
    assert "synthesis_node" not in calls_local
    assert first["extraction_review_queue"]

    resolution_update = resolve_extraction_review_queue(
        first,
        {("rec-pending", "study_design"): {
            "outcome": "extracted", "value": "RCT",
            "evidence": [{"quotation_or_description": "human-confirmed via full text", "locator": "Table 2"}],
            "attested": True,
            "reviewer": "R. Rauleckas", "note": "manual read confirms RCT design",
            "resolved_at": "2026-08-21T12:00:00Z",
        }},
    )
    assert resolution_update["extraction_review_queue"] == []
    assert resolution_update["charted_data"][0]["raw_extraction"]["study_design"] == "RCT"

    graph.update_state(config, resolution_update)
    second = graph.invoke(None, config)

    assert calls_local.count("extraction_node") == 1  # extraction itself was NOT re-run
    assert "synthesis_node" in calls_local
    assert "reporting_node" in calls_local
    assert second["extraction_review_queue"] == []


# ---------------------------------------------------------------------------
# Wiring: scoping, rapid, and SLR all use extraction_node (never the legacy
# charting_node).
# ---------------------------------------------------------------------------

def test_all_three_graphs_are_wired_to_extraction_not_charting():
    import slr_pipeline.graphs.rapid as rapid_mod
    import slr_pipeline.graphs.scoping as scoping_mod
    import slr_pipeline.graphs.slr as slr_mod

    for mod, builder in (
        (scoping_mod, "build_scoping_graph"),
        (rapid_mod, "build_rapid_graph"),
        (slr_mod, "build_slr_graph"),
    ):
        assert hasattr(mod, "extraction_node")
        assert not hasattr(mod, "charting_node")
        graph = getattr(mod, builder)()
        node_names = set(graph.get_graph().nodes)
        assert "extraction" in node_names
        assert "charting" not in node_names
