"""Proves Stage 2 of quality_appraisal_synthesis_plan_2026-08-22.md
("Charting via the Extraction Engine") for the scoping graph: the same
fulltext -> extraction -> synthesis conditional-edge gate rapid/SLR already
have (proven generically in tests/graphs/test_extraction_routing.py's own
GRAPH_CASES, which now includes scoping), PLUS scoping-specific coverage
this file owns:
  - the resume/resolve sequence through a genuinely queued extraction field
    (mirrors tests/graphs/test_extraction_routing.py's own
    test_graph_resumes_past_extraction_gate_after_resolving_queue, using
    scoping_chart_schema()/scoping_extraction_model_config() instead of an
    ad hoc single-field schema);
  - Section 2.5's explicit "real-data check against Berg 2023 reports
    key_findings' specific unclear/not_reported rate, alongside the other
    six fields' rates" -- run against the REAL Berg 2023 PDF
    (test_reviews/berg2023/review.pdf) through the REAL Docling parser (no
    network/money cost), with only the OpenAI provider call faked (a
    deterministic keyword-heuristic stand-in, NOT a real LLM judgement --
    this is a structural/plumbing check that the real document's real
    chunked content reaches every field's prompt correctly, not a claim
    about real-world extraction quality; the genuine LLM-based empirical
    check is a manual follow-up the user runs with OPENAI_API_KEY set).
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import slr_pipeline.extraction.engine as engine_module
from slr_pipeline.adapters.scoping_charting_adapter import (
    scoping_chart_schema, scoping_extraction_model_config,
)
from slr_pipeline.config import ReviewConfig, scoping_review_config
from slr_pipeline.extraction.openai_runner import (
    BatchCallTelemetry, RawBatchResponse, RawEvidenceItem, RawFieldResponse, RawFinding,
)
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.state import Record, initial_state

import slr_pipeline.nodes.extraction as real_extraction_mod
from slr_pipeline.nodes.extraction import resolve_extraction_review_queue


# ---------------------------------------------------------------------------
# Resume/resolve sequence through the scoping-specific schema.
# ---------------------------------------------------------------------------

def _real_unclear_scoping_extraction_run(monkeypatch, tmp_path):
    """Mirrors tests/graphs/test_extraction_routing.py's own
    `_real_unclear_extraction_run`, but using the real
    scoping_chart_schema()/scoping_extraction_model_config() adapter
    functions instead of an ad hoc single-field schema -- a genuine
    on-disk checkpoint/manifest for the scoping schema, needed because
    resolve_extraction_review_queue() derives/verifies a VerifiedRunBundle
    against actual Phase A artifacts on disk."""
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
        # study_design's evidence quotation does NOT verbatim-match the
        # chunk text -> evidence verification fails -> status='unclear'.
        # Every other field is a clean not_found.
        fields = {"study_design": RawFieldResponse(outcome="found", findings=[
            RawFinding(value="RCT", confidence="high", rationale="x",
                       evidence=[RawEvidenceItem(chunk_id=c.chunk_id, quotation="text that does not match verbatim")]),
        ])}
        for field_id in schema.field_map():
            if field_id != "study_design":
                fields[field_id] = RawFieldResponse(outcome="not_found")
        return RawBatchResponse(fields=fields), BatchCallTelemetry(
            provider_model="gpt-5.6-terra", prompt_tokens=1, completion_tokens=1, total_tokens=2,
        )

    monkeypatch.setattr(real_extraction_mod, "run_openai_batch_extraction", fake_call)

    schema = scoping_chart_schema()
    model_config = scoping_extraction_model_config()

    pdf_path = tmp_path / "rec-pending.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = scoping_review_config(
        title="t", output_dir=str(tmp_path / "out"),
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


def test_scoping_graph_resumes_past_extraction_gate_after_resolving_queue(monkeypatch, tmp_path):
    mod = importlib.import_module("slr_pipeline.graphs.scoping")

    cfg, pending_record, ext_output = _real_unclear_scoping_extraction_run(monkeypatch, tmp_path)

    calls_local: List[str] = []

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

    graph = mod.build_scoping_graph()
    config = {"configurable": {"thread_id": "resume-scoping-extraction-test"}}

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
# Section 2.5's real-data check against Berg 2023 -- real PDF, real Docling
# parse, faked (keyword-heuristic) provider call. See this file's own
# module docstring for exactly what this does and does not prove.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
_BERG_2023_PDF = REPO_ROOT / "test_reviews" / "berg2023" / "review.pdf"


def _heuristic_field_response(field_id: str, chunk: ParsedChunk) -> RawFieldResponse:
    """A deterministic, keyword-driven stand-in for a real LLM judgement
    -- NOT a claim of extraction quality, only a structural exercise of
    "does real, Docling-parsed Berg 2023 text reach each field's prompt
    and produce a plausible found/not_found/unclear split." A real
    empirical check requires a real model call (see module docstring)."""
    text = chunk.text
    keyword_map = {
        "country": [r"\b[A-Z][a-z]+ (?:Islands?|Republic)\b", r"\bUnited States\b", r"\bU\.S\.A?\.?\b"],
        "study_design": [r"\bsimulation\b", r"\bmodel[- ]based\b", r"\bcase study\b", r"\bmethodology\b"],
        "population": [r"\bpopulation\b", r"\bresidents?\b", r"\bpatients?\b"],
        "concept": [r"\bsystems engineering\b", r"\bMBSE\b", r"\bdisaster\b"],
        "context": [r"\bhealthcare\b", r"\brural\b", r"\bsetting\b"],
        "limitations": [r"\blimitations?\b", r"\bfuture work\b"],
        "key_findings": [r"\bresults?\b", r"\bfound that\b", r"\bfindings?\b", r"\bconclu"],
    }
    for pattern in keyword_map.get(field_id, []):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            quotation = text[match.start():match.start() + min(60, len(text) - match.start())]
            if field_id == "key_findings":
                return RawFieldResponse(outcome="found", findings=[
                    RawFinding(value=quotation.strip(), confidence="medium", rationale="keyword heuristic match",
                               evidence=[RawEvidenceItem(chunk_id=chunk.chunk_id, quotation=quotation)]),
                ])
            return RawFieldResponse(outcome="found", findings=[
                RawFinding(value=quotation.strip(), confidence="medium", rationale="keyword heuristic match",
                           evidence=[RawEvidenceItem(chunk_id=chunk.chunk_id, quotation=quotation)]),
            ])
    return RawFieldResponse(outcome="not_found")


@pytest.mark.skipif(not _BERG_2023_PDF.exists(), reason="Berg 2023 fixture PDF not present in this checkout")
def test_scoping_extraction_over_real_berg2023_pdf_reports_per_field_rates(monkeypatch, tmp_path, capsys):
    """Real Docling parse of the real Berg 2023 PDF (no network/money
    cost) through extraction_node with scoping_chart_schema(), a faked
    (keyword-heuristic, non-LLM) provider call. Reports the per-field
    extracted/not_reported/unclear rate, with key_findings' own rate
    called out explicitly -- Section 2.5's own requirement -- as a
    structural precondition check, not a substitute for a real,
    credentialed LLM run."""

    def fake_call(*, schema, batch_chunks, config):
        fields = {}
        for field_id in schema.field_map():
            # First chunk with a keyword match wins; batches are processed
            # one at a time by run_extraction's own aggregation, so this
            # per-batch response only needs to answer for THIS batch.
            best = RawFieldResponse(outcome="not_found")
            for chunk in batch_chunks:
                candidate = _heuristic_field_response(field_id, chunk)
                if candidate.outcome == "found":
                    best = candidate
                    break
            fields[field_id] = best
        return RawBatchResponse(fields=fields), BatchCallTelemetry(
            provider_model="gpt-5.6-terra", prompt_tokens=1, completion_tokens=1, total_tokens=2,
        )

    monkeypatch.setattr(real_extraction_mod, "run_openai_batch_extraction", fake_call)

    schema = scoping_chart_schema()
    model_config = scoping_extraction_model_config()
    cfg = scoping_review_config(
        title="Berg 2023 real-data check", output_dir=str(tmp_path / "out"),
        extraction_schema=schema.model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(model_config),
    )
    record = Record(
        id="berg2023", source="test", source_id="berg2023", title="Berg 2023",
        full_text_path=str(_BERG_2023_PDF), full_text_status="parsed",
    )
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [record.to_dict()]

    output = real_extraction_mod.extraction_node(state)

    assert output["extraction_results"], "extraction produced no results at all"
    result = output["extraction_results"][0]
    status_by_field: Dict[str, str] = {f["field_id"]: f["status"] for f in result["fields"]}
    assert set(status_by_field) == set(schema.field_map())

    print("\nBerg 2023 real-data per-field status (Docling-parsed, heuristic-faked provider call):")
    for field_id, status in sorted(status_by_field.items()):
        print(f"  {field_id:15s} -> {status}")
    print(f"  key_findings status: {status_by_field['key_findings']!r} "
          f"(unclear/not_reported here would indicate the atomic-evidence structure "
          "needs prompt tuning against a real LLM before Stage 2 is fully validated)")

    # Structural assertion: at least SOME field found real evidence in the
    # real document (proves chunks/schema/verification genuinely connect
    # end to end over Berg 2023's real content) -- not an assertion about
    # which specific fields, since that depends on the heuristic's own
    # keyword luck, not a real model's judgement.
    assert any(status == "extracted" for status in status_by_field.values()), (
        "no field found ANY evidence in the real, Docling-parsed Berg 2023 text -- "
        "either the heuristic's keywords are miscalibrated or the schema/chunking "
        "connection is broken; a real LLM run would need investigation either way"
    )
