"""Offline tests for Extraction Phase B's shared node (nodes/extraction.py):
extraction_node, resolve_extraction_review_queue, and the internal
_project_charted_data compatibility projection.

Two boundaries are monkeypatched, matching tests/extraction/test_engine.py's
own convention exactly:
  - `slr_pipeline.extraction.engine.parse_pdf` -- replaced with a fixed,
    hand-built ParsedDocument (no real Docling parse of a real PDF).
  - `nodes.extraction.run_openai_batch_extraction` -- the ONE provider call
    this node wires up (requirement 4); replaced with a scripted fake so no
    real OpenAI credential or network call is ever made.
Everything else (run_extraction's own checkpointing/manifest/run-identity,
evidence verification, aggregation, batching) runs for real, through the
frozen Phase A engine, unmodified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest
import slr_pipeline
import slr_pipeline.extraction.engine as engine_module
from slr_pipeline.config import ReviewConfig
from slr_pipeline.extraction.checkpoint import CheckpointIntegrityError, ExtractionCheckpointStore, VerifiedRunBundle
from slr_pipeline.extraction.openai_runner import (
    BatchCallTelemetry, ExtractionCallError, OpenAIExtractionConfig, RawBatchResponse,
    RawEvidenceItem, RawFieldResponse, RawFinding,
)
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.extraction.schema import AllowedValue, ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind
from slr_pipeline.state import Record, initial_state

import slr_pipeline.nodes.extraction as extraction_module
from slr_pipeline.nodes.extraction import (
    _project_charted_data, _resolution_artifact_path, extraction_node,
    load_verified_extraction_resolution_artifact, resolve_extraction_review_queue,
)

SLR_PIPELINE_DIR = Path(slr_pipeline.__file__).resolve().parent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _chunk(index: int, text: str, section: str = "Methods") -> ParsedChunk:
    return ParsedChunk(
        chunk_id=f"doc-chunk-{index:04d}", order_index=index, section=section,
        page_min=index + 1, page_max=index + 1, text=text,
    )


def _parsed_document(chunks: tuple) -> ParsedDocument:
    return ParsedDocument(
        document_sha256="a" * 64, parser_name="docling", parser_version="2.61.2",
        parser_options=DoclingParserOptions(), chunks=chunks,
    )


def _schema(**overrides) -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    defaults = dict(schema_id="s", version="1", fields=[field])
    defaults.update(overrides)
    return ExtractionSchema(**defaults)


def _model_config(**overrides) -> OpenAIExtractionConfig:
    defaults = dict(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )
    defaults.update(overrides)
    return OpenAIExtractionConfig(**defaults)


def _telemetry() -> BatchCallTelemetry:
    return BatchCallTelemetry(provider_model="gpt-5.6-terra", prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _found_response(field_id: str, value: str, chunk_id: str, quotation: str) -> RawBatchResponse:
    return RawBatchResponse(fields={
        field_id: RawFieldResponse(outcome="found", findings=[
            RawFinding(value=value, confidence="high", rationale="explicit statement",
                       evidence=[RawEvidenceItem(chunk_id=chunk_id, quotation=quotation)]),
        ]),
    })


def _not_found_response(field_id: str) -> RawBatchResponse:
    return RawBatchResponse(fields={field_id: RawFieldResponse(outcome="not_found")})


def _record(id: str, pdf_path) -> Record:
    return Record(
        id=id, source="test", source_id=id, title=f"Study {id}",
        full_text_path=str(pdf_path), full_text_status="parsed",
    )


def _cfg(tmp_path, *, schema=None, model_config=None) -> ReviewConfig:
    kwargs: Dict[str, Any] = dict(title="t", review_type="rapid", output_dir=str(tmp_path / "out"))
    if schema is not None:
        kwargs["extraction_schema"] = schema.model_dump(mode="json")
    if model_config is not None:
        import dataclasses
        kwargs["extraction_model_config"] = dataclasses.asdict(model_config)
    return ReviewConfig(**kwargs)


def _state(cfg: ReviewConfig, records) -> Dict[str, Any]:
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [r.to_dict() for r in records]
    return state


# ---------------------------------------------------------------------------
# Requirement 2: confirmed-PDF-only boundary -- no title/abstract fallback,
# no include-on-error, no re-screening.
# ---------------------------------------------------------------------------

def test_extraction_node_no_records_is_a_clean_noop(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [])
    output = extraction_node(state)
    assert output["extraction_results"] == []
    assert output["extraction_review_queue"] == []
    assert output["requires_human_review"] is False
    assert output["charted_data"] == []
    assert output["extraction_run_id"] is None


def test_extraction_node_rejects_record_with_no_confirmed_pdf_path(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    bad_record = Record(id="rec-a", source="test", source_id="rec-a", title="No PDF")
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        extraction_node(state)


def test_extraction_node_rejects_non_parsed_status_even_with_a_path(tmp_path):
    """Codex review finding E: path truthiness alone previously let a
    record with full_text_status=None (or any non-'parsed' status) and
    ANY path string through this boundary."""
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    bad_record = Record(
        id="rec-a", source="test", source_id="rec-a", title="t",
        full_text_path=str(pdf_path), full_text_status=None,
    )
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        extraction_node(state)


def test_extraction_node_rejects_rejected_stub_status(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    bad_record = Record(
        id="rec-a", source="test", source_id="rec-a", title="t",
        full_text_path=str(pdf_path), full_text_status="rejected_stub",
    )
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        extraction_node(state)


def test_extraction_node_rejects_path_that_is_a_directory(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    directory_path = tmp_path / "not_a_file"
    directory_path.mkdir()
    bad_record = Record(
        id="rec-a", source="test", source_id="rec-a", title="t",
        full_text_path=str(directory_path), full_text_status="parsed",
    )
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        extraction_node(state)


def test_extraction_node_rejects_nonexistent_path(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    bad_record = Record(
        id="rec-a", source="test", source_id="rec-a", title="t",
        full_text_path=str(tmp_path / "does_not_exist.pdf"), full_text_status="parsed",
    )
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        extraction_node(state)


def test_extraction_node_pdf_boundary_rejects_before_any_schema_or_provider_work(monkeypatch, tmp_path):
    """Zero parser/provider calls when the PDF boundary itself fails --
    the check happens before schema/model-config loading or run_extraction."""
    def fail_if_called(*a, **k):
        raise AssertionError("must never be called -- PDF boundary check happens first")

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fail_if_called)
    monkeypatch.setattr(engine_module, "parse_pdf", fail_if_called)

    # No extraction_schema/model_config configured at all -- if the PDF
    # boundary check didn't happen first, this would fail with a DIFFERENT
    # (schema/model-config) error instead.
    cfg = _cfg(tmp_path, schema=None, model_config=None)
    bad_record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        extraction_node(state)


def test_extraction_node_never_references_abstract_field(monkeypatch):
    """Structural check: this module must never read Record.abstract as a
    fallback source of text -- extraction is confirmed-PDF-only by
    construction (requirement 2), not merely by convention."""
    source = Path(SLR_PIPELINE_DIR / "nodes" / "extraction.py").read_text(encoding="utf-8")
    assert ".abstract" not in source


# ---------------------------------------------------------------------------
# Requirement 3: schema and model configuration required when extraction
# runs -- never silently defaulted.
# ---------------------------------------------------------------------------

def test_extraction_node_requires_extraction_schema(tmp_path):
    cfg = _cfg(tmp_path, schema=None, model_config=_model_config())
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    with pytest.raises(ValueError, match="extraction_schema is not set"):
        extraction_node(state)


def test_extraction_node_requires_extraction_model_config(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=None)
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    with pytest.raises(ValueError, match="extraction_model_config is not set"):
        extraction_node(state)


def test_extraction_node_invalid_model_config_dict_raises(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    cfg.extraction_model_config = {"model": "gpt-5.6-terra"}  # missing required fields
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    with pytest.raises(TypeError):
        extraction_node(state)


def test_extraction_node_invalid_schema_dict_raises(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    cfg.extraction_schema = {"schema_id": "s"}  # missing required fields (version, fields)
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    with pytest.raises(Exception):  # pydantic.ValidationError
        extraction_node(state)


# ---------------------------------------------------------------------------
# Requirement 4: no backend substitution -- structural checks.
# ---------------------------------------------------------------------------

def test_extraction_node_module_never_references_another_backend():
    source = Path(SLR_PIPELINE_DIR / "nodes" / "extraction.py").read_text(encoding="utf-8")
    for forbidden in ("ollama", "local_runner", "hf_server", "llm_backends"):
        assert forbidden not in source


def test_extraction_node_module_import_performs_no_network_or_credential_activity(monkeypatch):
    """An ISOLATED load of nodes/extraction.py alone (never the `nodes`
    package __init__, which eagerly `from openai import OpenAI`s via
    charting.py -- see nodes/fulltext.py's own docstring on that split)
    must not import `openai` as a top-level side effect: this module's
    own provider call (openai_runner.run_openai_batch_extraction) is lazy,
    reading OPENAI_API_KEY only inside a function body at call time."""
    import importlib.util

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    extraction_path = SLR_PIPELINE_DIR / "nodes" / "extraction.py"
    spec = importlib.util.spec_from_file_location("_test_isolated_copy_of_extraction_node", extraction_path)
    isolated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated_module)

    assert "openai" not in sys.modules


# ---------------------------------------------------------------------------
# Requirement 5/6: run_extraction() used directly; complete results,
# review queue, run provenance preserved in state.
# ---------------------------------------------------------------------------

def test_extraction_node_found_value_no_review_queue_and_charted_data_projected(monkeypatch, tmp_path):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = extraction_node(state)

    assert output["extraction_review_queue"] == []
    assert output["requires_human_review"] is False
    assert output["extraction_run_id"] is not None
    assert output["extraction_manifest_path"] is not None
    assert output["extraction_manifest_hash"] is not None
    assert output["extraction_schema_hash"] is not None
    assert output["extraction_artifact_paths"]["checkpoints_dir"]

    assert len(output["extraction_results"]) == 1
    assert output["extraction_results"][0]["document_id"] == "rec-a"
    assert output["extraction_results"][0]["fields"][0]["status"] == "extracted"

    assert output["charted_data"] == [{
        "record_id": "rec-a", "raw_extraction": {"study_design": "RCT"},
        "evidence_items": [], "section_classifications": [],
        "validated": True, "validation_errors": [], "charting_method": "extraction_projection",
    }]


def test_extraction_node_unclear_value_populates_review_queue_no_charted_data(monkeypatch, tmp_path):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        # Paraphrase, not verbatim -- evidence fails verification -> unclear.
        return _found_response(
            "study_design", "RCT", chunk.chunk_id, "the study used a randomized design approach",
        ), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = extraction_node(state)

    assert output["requires_human_review"] is True
    assert len(output["extraction_review_queue"]) == 1
    entry = output["extraction_review_queue"][0]
    assert entry["record_id"] == "rec-a"
    assert entry["field_id"] == "study_design"
    assert entry["status"] == "unclear"
    assert entry["field_result"]["requires_human_review"] is True
    # requirement 9: not projected while queue is non-empty -- explicitly
    # cleared to [] (codex review finding C's "never leave charted_data
    # stale under partial-update semantics"), not merely omitted.
    assert output["charted_data"] == []


def test_extraction_node_multiple_fields_extraction_failed_and_not_found_both_queue_or_pass_correctly(monkeypatch, tmp_path):
    """extraction_failed fields queue; not_reported fields don't."""
    chunk = _chunk(0, "Some text about funding sources only.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        raise ExtractionCallError("simulated API failure")

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = extraction_node(state)
    assert output["requires_human_review"] is True
    assert len(output["extraction_review_queue"]) == 1
    assert output["extraction_review_queue"][0]["status"] == "extraction_failed"


# ---------------------------------------------------------------------------
# Checkpoint resume after a simulated interruption -- no repeated provider
# calls for a document that already has a checkpoint on disk.
# ---------------------------------------------------------------------------

def test_extraction_node_resumes_after_simulated_interruption_without_recalling(monkeypatch, tmp_path):
    chunk_a = _chunk(0, "We used a randomized controlled trial design.")
    chunk_b = _chunk(1, "This cohort study followed participants over time.")

    def fake_parse(pdf_path, *a, **k):
        if "rec-a" in str(pdf_path):
            return _parsed_document((chunk_a,))
        return _parsed_document((chunk_b,))

    monkeypatch.setattr(engine_module, "parse_pdf", fake_parse)

    pdf_a = tmp_path / "rec-a.pdf"
    pdf_a.write_bytes(b"fake pdf bytes a")
    pdf_b = tmp_path / "rec-b.pdf"
    pdf_b.write_bytes(b"fake pdf bytes b")

    call_count = {"rec-a": 0, "rec-b": 0}

    def crashing_call(*, schema, batch_chunks, config):
        chunk = batch_chunks[0]
        if chunk.chunk_id == chunk_a.chunk_id:
            call_count["rec-a"] += 1
            return _found_response("study_design", "RCT", chunk_a.chunk_id, "randomized controlled trial"), _telemetry()
        call_count["rec-b"] += 1
        # Simulates a hard crash mid-run (NOT ExtractionCallError, so it is
        # never caught inside extract_document -- it propagates all the way
        # out of run_extraction, exactly like an interrupted process: rec-a
        # is already checkpointed by the time this raises, rec-b never gets
        # its own checkpoint written at all.
        raise RuntimeError("simulated process crash")

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", crashing_call)

    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_a), _record("rec-b", pdf_b)])

    with pytest.raises(RuntimeError, match="simulated process crash"):
        extraction_node(state)
    assert call_count == {"rec-a": 1, "rec-b": 1}

    # Second attempt: rec-b's call now succeeds; rec-a must NOT be recalled.
    def recovering_call(*, schema, batch_chunks, config):
        chunk = batch_chunks[0]
        if chunk.chunk_id == chunk_a.chunk_id:
            call_count["rec-a"] += 1
            return _found_response("study_design", "RCT", chunk_a.chunk_id, "randomized controlled trial"), _telemetry()
        call_count["rec-b"] += 1
        return _found_response("study_design", "cohort study", chunk_b.chunk_id, "cohort study followed participants"), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", recovering_call)

    output = extraction_node(state)
    assert call_count["rec-a"] == 1  # NOT called again -- resumed from checkpoint
    assert call_count["rec-b"] == 2  # first (crashed) + second (succeeded) attempt
    assert {r["document_id"] for r in output["extraction_results"]} == {"rec-a", "rec-b"}
    assert output["extraction_review_queue"] == []
    assert "charted_data" in output


# ---------------------------------------------------------------------------
# resolve_extraction_review_queue() -- requirement 8 + codex review round-2
# finding C1: every call now re-derives and cross-checks a VerifiedRunBundle
# from REAL on-disk checkpoint/manifest files -- so every test below first
# runs a real (parse/provider-faked) extraction_node() call to produce a
# genuine unclear queue entry with real artifacts on disk, rather than
# hand-crafting queue/results dicts with no backing store.
# ---------------------------------------------------------------------------

_VALID_TS = "2026-08-21T12:00:00Z"


def _evidence(description="human-confirmed via full text", locator="Table 2"):
    return [{"quotation_or_description": description, "locator": locator}]


def _bad_evidence_response(field_ids, chunk_id) -> RawBatchResponse:
    """A 'found' response whose evidence quotation does NOT verbatim-match
    the chunk text -- fails evidence verification -> status='unclear',
    for every field_id given, regardless of value_kind."""
    return RawBatchResponse(fields={
        fid: RawFieldResponse(outcome="found", findings=[
            RawFinding(value="SOME_VALUE", confidence="high", rationale="x",
                       evidence=[RawEvidenceItem(chunk_id=chunk_id, quotation="text that does not match verbatim")]),
        ]) for fid in field_ids
    })


def _run_to_unclear(monkeypatch, tmp_path, schema, field_ids, record_id: str = "rec-a") -> Dict[str, Any]:
    """Runs extraction_node() for REAL (parse_pdf + provider both faked)
    so that every field in `field_ids` ends up status='unclear' with a
    genuine checkpoint/manifest on disk -- the on-disk artifacts
    resolve_extraction_review_queue's own VerifiedRunBundle-based
    verification now requires (codex review round-2 finding C1). Returns
    the state dict, merged with extraction_node's own output, ready to
    pass into resolve_extraction_review_queue()."""
    chunk = _chunk(0, "Some ambiguous passage mentioning the field.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        return _bad_evidence_response(field_ids, chunk.chunk_id), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / f"{record_id}.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=schema, model_config=_model_config())
    state = _state(cfg, [_record(record_id, pdf_path)])
    output = extraction_node(state)
    state.update(output)
    assert state["extraction_review_queue"], "setup failed: expected queued field(s)"
    return state


def test_resolve_extraction_review_queue_unknown_key_raises_keyerror(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(KeyError):
        resolve_extraction_review_queue(state, {("rec-a", "wrong_field"): {"outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS}})


def test_resolve_extraction_review_queue_requires_reviewer_note_timestamp(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="non-blank reviewer, note, and resolved_at"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {"outcome": "not_reported", "note": "n", "resolved_at": _VALID_TS}})


def test_resolve_extraction_review_queue_rejects_whitespace_only_reviewer(monkeypatch, tmp_path):
    """Codex review finding D: whitespace-only metadata previously passed
    the bare truthiness check."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="non-blank reviewer, note, and resolved_at"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "   ", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_rejects_invalid_timestamp(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="ISO-8601"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": "not-a-timestamp",
        }})


def test_resolve_extraction_review_queue_rejects_invalid_outcome(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="outcome must be"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "still_unclear", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_extracted_requires_value(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="non-empty 'value'"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "evidence": _evidence(), "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_rejects_whitespace_only_value(monkeypatch, tmp_path):
    """Codex review finding D: a whitespace-only value previously cleared
    the queue and projected into charted_data unmodified."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="non-blank strings"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "   ", "evidence": _evidence(), "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_rejects_list_value_for_single_cardinality_field(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])  # cardinality="single"
    with pytest.raises(ValueError, match="requires a single scalar string value, not a list"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": ["RCT", "cohort"], "evidence": _evidence(), "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_rejects_invalid_categorical_value(monkeypatch, tmp_path):
    """Codex review finding D: a categorical value outside allowed_values
    previously bypassed schema validation entirely."""
    field = ExtractionFieldSpec(
        field_id="has_control_group", question="q", description="d", value_kind=ValueKind.CATEGORICAL,
        cardinality="single", allowed_values=(AllowedValue(value="yes", definition="Has a control group"),),
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(fields=[field]), ["has_control_group"])
    with pytest.raises(ValueError, match="is not valid for value_kind"):
        resolve_extraction_review_queue(state, {("rec-a", "has_control_group"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "NOT_ALLOWED", "evidence": _evidence(), "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_rejects_invalid_boolean_value(monkeypatch, tmp_path):
    field = ExtractionFieldSpec(
        field_id="randomized", question="q", description="d", value_kind=ValueKind.BOOLEAN,
        cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(fields=[field]), ["randomized"])
    with pytest.raises(ValueError, match="is not valid for value_kind"):
        resolve_extraction_review_queue(state, {("rec-a", "randomized"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "yes", "evidence": _evidence(), "attested": True,  # "yes" is not accepted -- only true/false
        }})


def test_resolve_extraction_review_queue_extracted_rejects_scalar_value_for_multiple_cardinality_field(monkeypatch, tmp_path):
    """Codex review round-2 finding D1's exact reproduction: a scalar
    string previously slipped through for a cardinality='multiple' field
    -- silently wrapped into a singleton list instead of being rejected."""
    field = ExtractionFieldSpec(
        field_id="designs", question="q", description="d", value_kind=ValueKind.TEXT,
        cardinality="multiple", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(fields=[field]), ["designs"])
    with pytest.raises(ValueError, match="requires a non-empty list of"):
        resolve_extraction_review_queue(state, {("rec-a", "designs"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "SINGLE_SCALAR", "evidence": _evidence(), "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_accepts_list_value_for_multiple_cardinality_field(monkeypatch, tmp_path):
    field = ExtractionFieldSpec(
        field_id="designs", question="q", description="d", value_kind=ValueKind.TEXT,
        cardinality="multiple", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(fields=[field]), ["designs"])
    output = resolve_extraction_review_queue(state, {("rec-a", "designs"): {
        "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        "value": ["RCT", "cohort"], "evidence": _evidence(), "attested": True,
    }})
    assert output["charted_data"][0]["raw_extraction"]["designs"] == ["RCT", "cohort"]


def test_resolve_extraction_review_queue_extracted_rejects_duplicate_values_for_multiple_field(monkeypatch, tmp_path):
    field = ExtractionFieldSpec(
        field_id="designs", question="q", description="d", value_kind=ValueKind.TEXT,
        cardinality="multiple", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(fields=[field]), ["designs"])
    with pytest.raises(ValueError, match="duplicate values"):
        resolve_extraction_review_queue(state, {("rec-a", "designs"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": ["RCT", "RCT"], "evidence": _evidence(), "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_requires_evidence(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="non-empty structured 'evidence'"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "RCT", "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_rejects_blank_string_evidence(monkeypatch, tmp_path):
    """Codex review finding D's exact reproduction: evidence=[""]
    previously passed as "non-empty"."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="each evidence item must be a dict"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "RCT", "evidence": [""], "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_rejects_evidence_missing_locator(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="non-blank 'locator'"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "RCT", "evidence": [{"quotation_or_description": "confirmed via full text"}],
            "attested": True,
        }})


def test_resolve_extraction_review_queue_extracted_requires_explicit_attestation(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    with pytest.raises(ValueError, match="human attestation"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
            "value": "RCT", "evidence": _evidence(), "attested": False,
        }})


def test_resolve_extraction_review_queue_extracted_preserves_original_result_and_builds_audit_trail(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    original_field_result = state["extraction_review_queue"][0]["field_result"]
    real_run_id = state["extraction_run_id"]

    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "extracted", "reviewer": "R. Rauleckas", "note": "manual read confirms RCT",
        "resolved_at": "2026-08-21T12:00:00Z", "value": "RCT",
        "evidence": _evidence(), "attested": True,
    }})

    assert output["extraction_review_queue"] == []
    assert output["requires_human_review"] is False
    assert "prisma_counts" not in output  # requirement 8: never touched

    # Original machine result untouched in the SOURCE state.
    assert state["extraction_review_queue"][0]["field_result"] == original_field_result

    assert len(output["extraction_review_resolutions"]) == 1
    audit = output["extraction_review_resolutions"][0]
    assert audit["record_id"] == "rec-a"
    assert audit["field_id"] == "study_design"
    assert audit["original_status"] == "unclear"
    assert audit["outcome"] == "extracted"
    assert audit["reviewer"] == "R. Rauleckas"
    assert audit["value"] == "RCT"
    assert audit["attested"] is True
    # codex review round-2 finding C1: bound to the run's OWN verified
    # manifest/schema hashes and the original field result's own hash --
    # not merely a nullable run_id label.
    assert audit["run_id"] == real_run_id
    assert audit["manifest_hash"] == state["extraction_manifest_hash"]
    assert audit["extraction_schema_hash"] == state["extraction_schema_hash"]
    assert audit["original_field_result_hash"]

    assert output["charted_data"][0]["raw_extraction"]["study_design"] == "RCT"


def test_resolve_extraction_review_queue_not_reported_clears_without_value(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "confirmed absent on manual read", "resolved_at": _VALID_TS,
    }})
    assert output["extraction_review_queue"] == []
    assert output["charted_data"][0]["raw_extraction"]["study_design"] is None


def test_resolve_extraction_review_queue_partial_resolution_leaves_others_queued_and_no_projection(monkeypatch, tmp_path):
    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])
    assert len(state["extraction_review_queue"]) == 2

    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    assert len(output["extraction_review_queue"]) == 1
    assert output["extraction_review_queue"][0]["field_id"] == "country"
    assert output["requires_human_review"] is True
    assert output["charted_data"] == []  # still gated -- one field remains queued


# ---------------------------------------------------------------------------
# Codex review round-2 finding C1: resolutions are re-derived/verified
# against a fresh VerifiedRunBundle every call, never trusted from
# possibly-stale/forged state.
# ---------------------------------------------------------------------------

def test_resolve_extraction_review_queue_rejects_null_run_id(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    state["extraction_run_id"] = None
    with pytest.raises(ValueError, match="extraction_run_id"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_rejects_run_id_with_no_published_manifest(tmp_path):
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", tmp_path / "a.pdf")])
    state["extraction_run_id"] = "extrun-" + "0" * 58
    state["extraction_review_queue"] = []
    with pytest.raises(ValueError, match="no published manifest"):
        resolve_extraction_review_queue(state, {})


def test_resolve_extraction_review_queue_rejects_mutated_schema_in_config(monkeypatch, tmp_path):
    """The schema in state['config'] must match the run's own VERIFIED
    manifest schema hash -- a config mutated after the run (e.g. someone
    edited a field's definition) is rejected, not silently trusted as
    'the run's own schema'."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    mutated_field = ExtractionFieldSpec(
        field_id="study_design", question="a DIFFERENT question", description="d", value_kind=ValueKind.TEXT,
        cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    state["config"]["extraction_schema"] = _schema(fields=[mutated_field]).model_dump(mode="json")
    with pytest.raises(ValueError, match="does not match this run's own verified manifest schema hash"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_rejects_altered_queue_entry(monkeypatch, tmp_path):
    """A copied/altered queue entry using a GENUINE run_id string must
    still be rejected -- its field_result must match the verified
    bundle's own field result exactly."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    state["extraction_review_queue"][0]["field_result"] = dict(
        state["extraction_review_queue"][0]["field_result"], rationale="TAMPERED",
    )
    with pytest.raises(ValueError, match="does not match the verified run bundle's own field result"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_rejects_queue_missing_a_verified_unresolved_field(monkeypatch, tmp_path):
    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])
    state["extraction_review_queue"] = state["extraction_review_queue"][:1]  # drop "country"
    with pytest.raises(ValueError, match="does not exactly match the verified run bundle's own"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_rejects_queue_with_a_foreign_extra_entry(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    forged_entry = dict(state["extraction_review_queue"][0], record_id="rec-forged")
    state["extraction_review_queue"] = state["extraction_review_queue"] + [forged_entry]
    with pytest.raises(ValueError, match="does not exactly match the verified run bundle's own"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_reconstructs_from_minimal_reloaded_state(monkeypatch, tmp_path):
    """Process-reload proxy: a minimal state dict carrying only what a
    real checkpointer/reload would restore (config, run_id, full_text_records,
    the queue, and extraction_results) resolves successfully purely because
    resolve_extraction_review_queue re-derives and verifies everything
    against the ON-DISK Phase A artifacts, never assuming a richer
    in-memory state survived."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    reloaded_state = {
        "config": state["config"],
        "extraction_run_id": state["extraction_run_id"],
        "full_text_records": state["full_text_records"],
        "extraction_review_queue": state["extraction_review_queue"],
        "extraction_results": state["extraction_results"],
        "extraction_review_resolutions": [],
    }
    output = resolve_extraction_review_queue(reloaded_state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    assert output["extraction_review_queue"] == []
    assert output["charted_data"][0]["raw_extraction"]["study_design"] is None


# ---------------------------------------------------------------------------
# Durable resolution artifact (codex review round-2 finding C1):
# extraction_resolutions.json -- self-hashed, append-only, reload/verify.
# ---------------------------------------------------------------------------

def test_extraction_node_publishes_resolution_artifact_even_with_no_review_needed(monkeypatch, tmp_path):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = extraction_node(state)
    state.update(output)

    artifact = load_verified_extraction_resolution_artifact(state)
    assert artifact is not None
    assert artifact["resolutions"] == []
    assert artifact["remaining_queue_keys"] == []
    assert artifact["resolved_projection"] == output["charted_data"]
    assert artifact["total_fields"] == 1
    assert artifact["run_id"] == state["extraction_run_id"]


def test_load_verified_extraction_resolution_artifact_none_without_a_run():
    state = {"config": ReviewConfig(title="t").to_dict(), "extraction_run_id": None}
    assert load_verified_extraction_resolution_artifact(state) is None


def test_resolution_artifact_accumulates_across_multiple_resolve_calls(monkeypatch, tmp_path):
    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])

    first = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(first)
    artifact_1 = load_verified_extraction_resolution_artifact(state)
    assert len(artifact_1["resolutions"]) == 1
    assert artifact_1["remaining_queue_keys"] == [["rec-a", "country"]]
    assert artifact_1["resolved_projection"] == []

    second = resolve_extraction_review_queue(state, {("rec-a", "country"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(second)
    artifact_2 = load_verified_extraction_resolution_artifact(state)
    assert len(artifact_2["resolutions"]) == 2  # append-only: first resolution never dropped
    assert artifact_2["remaining_queue_keys"] == []
    assert artifact_2["resolved_projection"]


def test_load_verified_extraction_resolution_artifact_detects_tampering(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)
    assert load_verified_extraction_resolution_artifact(state) is not None

    cfg = ReviewConfig.from_dict(state["config"])
    store = ExtractionCheckpointStore(
        output_dir=Path(cfg.output_dir) / "extraction_runs", run_id=state["extraction_run_id"],
    )
    artifact_path = _resolution_artifact_path(store)
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["remaining_queue_keys"] = [["rec-a", "TAMPERED_FIELD"]]
    artifact_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError):
        load_verified_extraction_resolution_artifact(state)


# ---------------------------------------------------------------------------
# Codex review round-3 (P1): the loader must RE-DERIVE artifact semantics
# from a verified bundle, not just trust a self-consistent content_hash --
# a file edited AND re-hashed to match its own new (wrong) content must
# still be rejected.
# ---------------------------------------------------------------------------

def _rehash_and_write(artifact_path: Path, data: Dict[str, Any]) -> None:
    from slr_pipeline.extraction.hashing import sha256_hex
    payload = {k: v for k, v in data.items() if k != "content_hash"}
    data["content_hash"] = sha256_hex(payload)
    artifact_path.write_text(json.dumps(data), encoding="utf-8")


def _store_for(state) -> ExtractionCheckpointStore:
    cfg = ReviewConfig.from_dict(state["config"])
    return ExtractionCheckpointStore(
        output_dir=Path(cfg.output_dir) / "extraction_runs", run_id=state["extraction_run_id"],
    )


def test_load_verified_extraction_resolution_artifact_rejects_self_consistent_wrong_total_fields(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["total_fields"] = 999
    _rehash_and_write(artifact_path, data)  # self-hash is now internally VALID for the tampered content

    with pytest.raises(ValueError, match="total_fields does not match"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_self_consistent_wrong_remaining_queue_keys(monkeypatch, tmp_path):
    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["remaining_queue_keys"] = []  # falsely claims fully resolved -- "country" is still unresolved
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="remaining_queue_keys does not match"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_forged_resolved_projection(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        "value": "RCT", "evidence": _evidence(), "attested": True,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["resolved_projection"][0]["raw_extraction"]["study_design"] = "FORGED_PROJECTION_VALUE"
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="resolved_projection is not derivable"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_duplicate_resolution_keys(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["resolutions"] = data["resolutions"] * 2  # duplicate the single resolution
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="duplicate resolution"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_resolution_with_wrong_original_hash(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        "value": "RCT", "evidence": _evidence(), "attested": True,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["resolutions"][0]["original_field_result_hash"] = "0" * 64
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="original_field_result_hash does not match"):
        load_verified_extraction_resolution_artifact(state)


# ---------------------------------------------------------------------------
# Codex review round-3 (P1c): resolved_projection/total_fields must be
# built from the VERIFIED bundle, never mutable state directly.
# ---------------------------------------------------------------------------

def test_resolve_extraction_review_queue_projection_ignores_tampered_clean_field_in_state(monkeypatch, tmp_path):
    """A CLEAN (never-queued) field's value in state["extraction_results"]
    must never influence resolved_projection -- only the VERIFIED bundle's
    own value does."""
    chunk = _chunk(0, "We used a randomized controlled trial design in the USA.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])

    def fake_call(*, schema, batch_chunks, config):
        c = batch_chunks[0]
        return RawBatchResponse(fields={
            "study_design": RawFieldResponse(outcome="found", findings=[
                RawFinding(value="RCT", confidence="high", rationale="x",
                           evidence=[RawEvidenceItem(chunk_id=c.chunk_id, quotation="text that does not match verbatim")]),
            ]),
            "country": RawFieldResponse(outcome="found", findings=[
                RawFinding(value="USA", confidence="high", rationale="x",
                           evidence=[RawEvidenceItem(chunk_id=c.chunk_id, quotation="randomized controlled trial design in the USA")]),
            ]),
        }), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=schema, model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = extraction_node(state)
    state.update(output)
    assert len(state["extraction_review_queue"]) == 1
    assert state["extraction_review_queue"][0]["field_id"] == "study_design"

    tampered = json.loads(json.dumps(state["extraction_results"]))
    for field in tampered[0]["fields"]:
        if field["field_id"] == "country":
            field["values"][0]["value"] = "TAMPERED_COUNTRY"
    state["extraction_results"] = tampered

    resolution_update = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    assert resolution_update["charted_data"][0]["raw_extraction"]["country"] == "USA"  # NOT "TAMPERED_COUNTRY"


def test_resolve_extraction_review_queue_rejects_document_coverage_mismatch(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    state["full_text_records"] = []  # document dropped from state entirely
    with pytest.raises(ValueError, match="does not match the verified run manifest's own document_ids"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


# ---------------------------------------------------------------------------
# Codex review round-3 (P2a): artifact-ahead-of-state recovery.
# ---------------------------------------------------------------------------

def test_merge_resolution_log_recovers_entries_state_does_not_know_about(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    # `state` itself is NEVER updated with the resolve() output -- simulates
    # a crash between the (already successful) artifact publish and the
    # graph checkpoint persisting the returned state update.
    from slr_pipeline.nodes.extraction import _merge_resolution_log

    verified_artifact = load_verified_extraction_resolution_artifact(state)
    merged = _merge_resolution_log(state.get("extraction_review_resolutions"), verified_artifact)
    assert len(merged) == 1
    assert merged[0]["field_id"] == "study_design"


def test_extraction_node_same_run_reentry_recovers_when_state_is_behind_artifact(monkeypatch, tmp_path):
    """extraction_node()'s own same-run branch also merges with the
    durable artifact -- even if state's own extraction_review_resolutions
    never got the already-published resolution (state crashed before
    persisting it), a same-run resume still excludes the resolved field
    from the rebuilt queue."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    # `state["extraction_review_resolutions"]` stays [] -- never updated.
    second_output = extraction_node(state)
    assert second_output["extraction_review_queue"] == []
    assert second_output["charted_data"][0]["raw_extraction"]["study_design"] is None


# ---------------------------------------------------------------------------
# Codex review round-4 finding 1: a deterministic run_id must be
# recoverable from FRESH state, not just a state that carried
# extraction_run_id forward.
# ---------------------------------------------------------------------------

def test_extraction_node_recovers_resolutions_from_completely_fresh_state(monkeypatch, tmp_path):
    """Exact round-4 reproduction: run_id is a deterministic hash of
    documents+schema+config -- a completely FRESH state (no
    extraction_run_id, no extraction_review_resolutions carried over at
    all) that recomputes the SAME run_id as a prior, already
    human-resolved run must recover that run's durable resolutions, not
    treat it as new and crash trying to publish an empty resolutions list
    over the existing non-empty artifact."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    resolve_update = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(resolve_update)
    real_run_id = state["extraction_run_id"]

    cfg = ReviewConfig.from_dict(state["config"])
    records = [Record.from_dict(r) for r in state["full_text_records"]]
    fresh_state = _state(cfg, records)  # a BRAND NEW state dict from scratch
    assert fresh_state.get("extraction_run_id") is None
    assert fresh_state.get("extraction_review_resolutions") == []

    output = extraction_node(fresh_state)  # must NOT raise
    assert output["extraction_run_id"] == real_run_id
    assert output["extraction_review_queue"] == []
    assert len(output["extraction_review_resolutions"]) == 1
    assert output["charted_data"][0]["raw_extraction"]["study_design"] is None


# ---------------------------------------------------------------------------
# Codex review round-4 finding 2: set-based checks must not permit
# duplicates that poison the append-only artifact.
# ---------------------------------------------------------------------------

def test_resolve_extraction_review_queue_rejects_duplicate_queue_entries(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    state["extraction_review_queue"] = state["extraction_review_queue"] * 2
    with pytest.raises(ValueError, match=r"duplicate \(record_id, field_id\) entries"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_resolve_extraction_review_queue_rejects_duplicate_full_text_records(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    state["full_text_records"] = state["full_text_records"] * 2
    with pytest.raises(ValueError, match="duplicate record_id"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})


def test_publish_resolution_artifact_rejects_duplicate_resolution_keys(monkeypatch, tmp_path):
    from slr_pipeline.nodes.extraction import _publish_resolution_artifact

    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    store = _store_for(state)
    dup_resolution = {
        "record_id": "rec-a", "field_id": "study_design", "outcome": "not_reported",
        "original_status": "unclear", "original_field_result_hash": "x" * 64, "reviewer": "R", "note": "n",
        "resolved_at": _VALID_TS, "run_id": state["extraction_run_id"], "manifest_hash": "m", "extraction_schema_hash": "s",
    }
    with pytest.raises(ValueError, match="duplicate resolution key"):
        _publish_resolution_artifact(
            store, run_id=state["extraction_run_id"], manifest_hash="m", extraction_schema_hash="s",
            total_fields=1, resolutions=[dup_resolution, dict(dup_resolution)],
            remaining_queue_keys=[], resolved_projection=[], existing_verified_artifact=None,
        )


def test_publish_resolution_artifact_rejects_duplicate_remaining_queue_keys(monkeypatch, tmp_path):
    from slr_pipeline.nodes.extraction import _publish_resolution_artifact

    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    store = _store_for(state)
    with pytest.raises(ValueError, match="duplicate key"):
        _publish_resolution_artifact(
            store, run_id=state["extraction_run_id"], manifest_hash="m", extraction_schema_hash="s",
            total_fields=1, resolutions=[],
            remaining_queue_keys=[("rec-a", "study_design"), ("rec-a", "study_design")],
            resolved_projection=[], existing_verified_artifact=None,
        )


def test_publish_resolution_artifact_duplicate_rejection_leaves_existing_artifact_untouched(monkeypatch, tmp_path):
    """Codex review round-4: rejection must happen BEFORE replacing the
    existing artifact -- the on-disk file must be byte-identical after a
    rejected (duplicate-containing) publish attempt."""
    from slr_pipeline.nodes.extraction import _publish_resolution_artifact

    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)
    store = _store_for(state)
    artifact_path = _resolution_artifact_path(store)
    original_bytes = artifact_path.read_bytes()
    existing_verified_artifact = load_verified_extraction_resolution_artifact(state)

    valid_resolution = state["extraction_review_resolutions"][0]
    with pytest.raises(ValueError, match="duplicate resolution key"):
        _publish_resolution_artifact(
            store, run_id=state["extraction_run_id"], manifest_hash=state["extraction_manifest_hash"],
            extraction_schema_hash=state["extraction_schema_hash"], total_fields=1,
            resolutions=[valid_resolution, dict(valid_resolution)],
            remaining_queue_keys=[], resolved_projection=state["charted_data"],
            existing_verified_artifact=existing_verified_artifact,
        )
    assert artifact_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Codex review round-4 finding 3: the loader must re-derive EVERY
# persisted semantic field, including remaining_queue_keys canonical
# shape, original_status, manifest_path, and not_reported shape.
# ---------------------------------------------------------------------------

def test_load_verified_extraction_resolution_artifact_rejects_duplicate_remaining_queue_keys(monkeypatch, tmp_path):
    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["remaining_queue_keys"] = data["remaining_queue_keys"] * 2
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="duplicate entries"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_noncanonical_remaining_queue_order(monkeypatch, tmp_path):
    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert len(data["remaining_queue_keys"]) == 2
    data["remaining_queue_keys"] = list(reversed(data["remaining_queue_keys"]))
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="canonical sorted order"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_rehashed_wrong_original_status(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["resolutions"][0]["original_status"] = "extraction_failed"  # wrong -- was "unclear"
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="original_status .* does not match"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_rehashed_wrong_manifest_path(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["manifest_path"] = str(Path(data["manifest_path"]).with_name("some_other_manifest.json"))
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="manifest_path does not match"):
        load_verified_extraction_resolution_artifact(state)


def test_load_verified_extraction_resolution_artifact_rejects_not_reported_with_stray_value(monkeypatch, tmp_path):
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["resolutions"][0]["value"] = "STRAY_VALUE"
    _rehash_and_write(artifact_path, data)

    with pytest.raises(ValueError, match="must not carry a value/evidence/attested"):
        load_verified_extraction_resolution_artifact(state)


# ---------------------------------------------------------------------------
# Codex review round-5 closing regressions.
# ---------------------------------------------------------------------------

def test_semantic_tampering_causes_error_and_leaves_artifact_bytes_unchanged(monkeypatch, tmp_path):
    """Round-5 finding 2's closing regression: a re-hashed, semantically
    wrong artifact must cause EVERY recovery path to raise, and the
    on-disk bytes must remain EXACTLY the tampered version afterward --
    never silently "healed" by a subsequent republish with corrected
    values (the centralized verifier + the no-independent-re-read
    publisher together make this structurally impossible)."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }})
    state.update(output)

    artifact_path = _resolution_artifact_path(_store_for(state))
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data["total_fields"] = 999
    _rehash_and_write(artifact_path, data)
    tampered_bytes = artifact_path.read_bytes()

    # 1. The public loader rejects it -- bytes untouched.
    with pytest.raises(ValueError, match="total_fields does not match"):
        load_verified_extraction_resolution_artifact(state)
    assert artifact_path.read_bytes() == tampered_bytes

    # 2. A fresh extraction_node() recovery attempt (same deterministic
    # run_id, completely fresh state) ALSO rejects it -- never silently
    # republishes corrected values over the tampered file.
    fresh_state = _state(
        ReviewConfig.from_dict(state["config"]), [Record.from_dict(r) for r in state["full_text_records"]],
    )
    with pytest.raises(ValueError, match="total_fields does not match"):
        extraction_node(fresh_state)
    assert artifact_path.read_bytes() == tampered_bytes

    # 3. resolve_extraction_review_queue() also rejects it -- bytes
    # untouched even though its own queue is already fully resolved.
    with pytest.raises(ValueError, match="total_fields does not match"):
        resolve_extraction_review_queue(state, {})
    assert artifact_path.read_bytes() == tampered_bytes


def test_fresh_state_recovery_succeeds_for_untouched_artifact(monkeypatch, tmp_path):
    """Pairs with the tampering regression above: an UNTOUCHED artifact
    must still be recoverable from completely fresh state (no
    extraction_run_id/extraction_review_resolutions carried over) --
    proving the verifier correctly accepts the genuinely valid case, not
    just always rejecting."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    output = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        "value": "RCT", "evidence": _evidence(), "attested": True,
    }})
    state.update(output)

    fresh_state = _state(
        ReviewConfig.from_dict(state["config"]), [Record.from_dict(r) for r in state["full_text_records"]],
    )
    assert fresh_state.get("extraction_run_id") is None
    assert load_verified_extraction_resolution_artifact(fresh_state) is None  # no run_id yet -- nothing to load

    recovered = extraction_node(fresh_state)  # must NOT raise
    assert recovered["extraction_review_queue"] == []
    assert recovered["charted_data"][0]["raw_extraction"]["study_design"] == "RCT"

    fresh_state.update(recovered)
    artifact_after = load_verified_extraction_resolution_artifact(fresh_state)
    assert artifact_after is not None
    assert len(artifact_after["resolutions"]) == 1


def test_new_resolution_status_tampering_is_rejected_before_any_write(monkeypatch, tmp_path):
    """Round-5 finding 1's exact reproduction: change ONLY a queue
    entry's top-level 'status' (leave its verified field_result
    untouched), then attempt to resolve it -- must be rejected BEFORE any
    artifact is (re)published, never silently accepted with a false
    original_status a later loader would then reject."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    artifact_path = _resolution_artifact_path(_store_for(state))
    original_bytes = artifact_path.read_bytes()

    state["extraction_review_queue"][0]["status"] = "extraction_failed"  # was "unclear" -- field_result untouched

    with pytest.raises(ValueError, match="does not match the verified run bundle's own field result/status"):
        resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        }})
    assert artifact_path.read_bytes() == original_bytes  # nothing was (re)published


def test_reversed_full_text_records_yields_consistent_manifest_order_projection(monkeypatch, tmp_path):
    """Round-5 finding 3's closing regression: reversing (or otherwise
    reordering) state["full_text_records"] must never change
    resolved_projection's own order -- a resolve() call and the public
    loader, both given a REVERSED state, must agree on the SAME canonical
    (manifest) document order the original publish used."""
    chunk_a = _chunk(0, "First study text about design A.")
    chunk_b = _chunk(1, "Second study text about design B.")

    def fake_parse(pdf_path, *a, **k):
        if "rec-a" in str(pdf_path):
            return _parsed_document((chunk_a,))
        return _parsed_document((chunk_b,))

    monkeypatch.setattr(engine_module, "parse_pdf", fake_parse)

    def fake_call(*, schema, batch_chunks, config):
        chunk = batch_chunks[0]
        if chunk.chunk_id == chunk_a.chunk_id:
            return _found_response("study_design", "Design A", chunk_a.chunk_id, chunk_a.text), _telemetry()
        return _found_response("study_design", "Design B", chunk_b.chunk_id, chunk_b.text), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_a = tmp_path / "rec-a.pdf"
    pdf_a.write_bytes(b"fake pdf bytes a")
    pdf_b = tmp_path / "rec-b.pdf"
    pdf_b.write_bytes(b"fake pdf bytes b")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_a), _record("rec-b", pdf_b)])
    output = extraction_node(state)
    state.update(output)
    assert state["extraction_review_queue"] == []
    assert [c["record_id"] for c in state["charted_data"]] == ["rec-a", "rec-b"]

    # Reverse full_text_records (SAME extraction_run_id, SAME document
    # set, just reordered) -- resolve() must still project in the SAME
    # canonical (manifest) order, never the reversed one.
    reversed_state = dict(state)
    reversed_state["full_text_records"] = list(reversed(state["full_text_records"]))
    resolve_output = resolve_extraction_review_queue(reversed_state, {})
    assert [c["record_id"] for c in resolve_output["charted_data"]] == ["rec-a", "rec-b"]

    # The public loader, given the SAME reversed state, agrees.
    reloaded_artifact = load_verified_extraction_resolution_artifact(reversed_state)
    assert [c["record_id"] for c in reloaded_artifact["resolved_projection"]] == ["rec-a", "rec-b"]


# ---------------------------------------------------------------------------
# Codex review round-6 finding 1: the read-verify-merge-write transaction
# must be serialized -- a lock-free version would let two concurrent
# resolvers, both reading the SAME artifact snapshot before either writes,
# silently lose one of their two resolutions on the later os.replace(...).
# ---------------------------------------------------------------------------

def test_resolution_artifact_lock_serializes_concurrent_acquirers(monkeypatch, tmp_path):
    """Direct proof that _resolution_artifact_lock() provides real mutual
    exclusion: a second acquirer blocks until the first releases, rather
    than proceeding concurrently."""
    import threading

    from slr_pipeline.nodes.extraction import _resolution_artifact_lock

    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    store = _store_for(state)

    order: list[str] = []
    first_acquired = threading.Event()
    first_may_release = threading.Event()

    def first():
        with _resolution_artifact_lock(store):
            order.append("first-acquired")
            first_acquired.set()
            first_may_release.wait(timeout=5)
            order.append("first-released")

    def second():
        first_acquired.wait(timeout=5)
        with _resolution_artifact_lock(store):
            order.append("second-acquired")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    first_acquired.wait(timeout=5)
    t2.start()
    time.sleep(0.2)  # second must still be blocked on the lock
    assert order == ["first-acquired"]

    first_may_release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert order == ["first-acquired", "first-released", "second-acquired"]


def test_concurrent_resolves_do_not_silently_lose_updates(monkeypatch, tmp_path):
    """Codex review round-6 finding 1's exact reproduction: two concurrent
    resolvers reading the SAME (momentarily accurate) state/artifact
    snapshot, each resolving a DIFFERENT field, must never have one
    silently overwrite the other's already-durable write. With the
    per-run lock serializing the whole read-verify-merge-write
    transaction, exactly ONE of the two succeeds; the other is rejected
    LOUDLY (its own queue snapshot is now stale relative to the winner's
    committed resolution) rather than silently discarding the winner's
    resolution."""
    import threading

    schema = _schema(fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
        ExtractionFieldSpec(field_id="country", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])
    state = _run_to_unclear(monkeypatch, tmp_path, schema, ["study_design", "country"])

    results: Dict[str, Any] = {}
    errors: Dict[str, Exception] = {}
    start_barrier = threading.Barrier(2)

    def resolve_field(name, field_id):
        try:
            start_barrier.wait(timeout=5)
            out = resolve_extraction_review_queue(state, {("rec-a", field_id): {
                "outcome": "not_reported", "reviewer": name, "note": "n", "resolved_at": _VALID_TS,
            }})
            results[name] = out
        except Exception as e:  # noqa: BLE001 -- captured for assertion, not swallowed
            errors[name] = e

    t1 = threading.Thread(target=resolve_field, args=("R1", "study_design"))
    t2 = threading.Thread(target=resolve_field, args=("R2", "country"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert len(results) + len(errors) == 2
    assert len(results) == 1, f"expected exactly one success, got results={list(results)}, errors={errors}"
    assert len(errors) == 1

    winner_output = next(iter(results.values()))
    assert len(winner_output["extraction_review_resolutions"]) == 1

    state.update(winner_output)
    artifact = load_verified_extraction_resolution_artifact(state)
    assert len(artifact["resolutions"]) == 1  # exactly the winner's -- never lost, never duplicated/corrupted


def test_resolution_artifact_lock_recovers_after_holder_is_hard_killed(tmp_path):
    """Codex review round-7 finding 1: an exclusively-CREATED lock file
    (`open(path, "x")`) is never cleaned up by a hard-killed holder --
    SIGKILL/TerminateProcess bypasses `finally`, so the lock file is left
    behind FOREVER and every future resumer waits out the full timeout
    and fails, undermining checkpoint recovery. The kernel-backed lock
    (`fcntl.flock` / `msvcrt.locking`) must instead be released by the OS
    itself the instant the killed process's file descriptors are torn
    down, so a fresh acquirer succeeds near-immediately, not after the
    timeout."""
    import subprocess

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    run_id = "run-hardkill"

    script = f"""
from pathlib import Path
from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
from slr_pipeline.nodes.extraction import _resolution_artifact_lock

store = ExtractionCheckpointStore(output_dir=Path({str(output_dir)!r}), run_id={run_id!r})
with _resolution_artifact_lock(store):
    print("locked", flush=True)
    import time
    time.sleep(60)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # The interpreter's own site/env startup may print unrelated noise
        # (e.g. this machine's conda env prints an API-keys banner) before
        # our script's output -- scan for "locked" rather than assuming
        # it is the very first line.
        deadline = time.monotonic() + 15.0
        saw_locked = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if line.strip() == "locked":
                saw_locked = True
                break
        assert saw_locked, f"child failed to acquire lock: stderr={proc.stderr.read()}"

        proc.kill()  # hard kill: TerminateProcess (Windows) / SIGKILL (POSIX) -- bypasses `finally`
        proc.wait(timeout=10)

        from slr_pipeline.nodes.extraction import _resolution_artifact_lock

        store = ExtractionCheckpointStore(output_dir=output_dir, run_id=run_id)
        start = time.monotonic()
        with _resolution_artifact_lock(store, timeout=5.0):
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, (
            f"lock acquisition took {elapsed:.2f}s after the holder was hard-killed -- expected "
            "near-immediate reclaim via the OS-released kernel lock, not a timeout wait"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_resolution_artifact_lock_propagates_non_contention_error_posix_adapter(monkeypatch, tmp_path):
    """Codex review round-8 finding (P2): a non-contention OSError from
    `fcntl.flock` (e.g. ENOLCK -- lock table exhausted) must propagate
    immediately with its own errno/cause, NOT be swallowed into
    `_LockBusyError` and retried for the full timeout, which would then
    surface as a misleading "another live process is holding this lock"
    report. Monkeypatches the module's `fcntl` attribute directly so this
    runs regardless of the host OS's real platform."""
    import errno as errno_module

    import slr_pipeline.nodes.extraction as extraction_module

    class _FakeFcntl:
        LOCK_EX = 0
        LOCK_NB = 0
        LOCK_UN = 0

        @staticmethod
        def flock(fd, flags):
            raise OSError(errno_module.ENOLCK, "simulated: no locks available")

    monkeypatch.setattr(extraction_module, "fcntl", _FakeFcntl)
    monkeypatch.setattr(extraction_module, "msvcrt", None)

    class _FakeStore:
        run_dir = tmp_path

    start = time.monotonic()
    with pytest.raises(OSError) as excinfo:
        with extraction_module._resolution_artifact_lock(_FakeStore(), timeout=30.0):
            pass
    elapsed = time.monotonic() - start

    assert excinfo.value.errno == errno_module.ENOLCK
    assert not isinstance(excinfo.value, extraction_module.ResolutionArtifactLockError)
    assert elapsed < 2.0, f"non-contention error took {elapsed:.2f}s to surface -- it must not be retried"


def test_resolution_artifact_lock_propagates_non_contention_error_windows_adapter(monkeypatch, tmp_path):
    """Same as the POSIX-adapter test above, for `msvcrt.locking`: a
    non-contention OSError (e.g. EBADF -- invalid file descriptor) must
    propagate immediately, never be mistaken for lock contention."""
    import errno as errno_module

    import slr_pipeline.nodes.extraction as extraction_module

    class _FakeMsvcrt:
        LK_NBLCK = 0
        LK_UNLCK = 0

        @staticmethod
        def locking(fd, mode, nbytes):
            raise OSError(errno_module.EBADF, "simulated: bad file descriptor")

    monkeypatch.setattr(extraction_module, "fcntl", None)
    monkeypatch.setattr(extraction_module, "msvcrt", _FakeMsvcrt)

    class _FakeStore:
        run_dir = tmp_path

    start = time.monotonic()
    with pytest.raises(OSError) as excinfo:
        with extraction_module._resolution_artifact_lock(_FakeStore(), timeout=30.0):
            pass
    elapsed = time.monotonic() - start

    assert excinfo.value.errno == errno_module.EBADF
    assert not isinstance(excinfo.value, extraction_module.ResolutionArtifactLockError)
    assert elapsed < 2.0, f"non-contention error took {elapsed:.2f}s to surface -- it must not be retried"


def test_resolution_artifact_lock_retries_posix_eacces_as_contention(monkeypatch, tmp_path):
    """Codex review round-9 finding (P2): the Python 3.12 fcntl docs
    document that `flock(..., LOCK_NB)` contention may be reported as
    EACCES rather than EAGAIN/EWOULDBLOCK, depending on the OS. EACCES
    must therefore be RETRIED (treated as contention), not propagated
    immediately as a PermissionError -- simulates a holder that reports
    EACCES for its first two attempts, then releases (raises no more),
    and asserts the lock is still acquired successfully within the
    timeout, proving EACCES is now in the POSIX contention set rather
    than the non-contention path."""
    import errno as errno_module

    import slr_pipeline.nodes.extraction as extraction_module

    acquire_attempts = {"count": 0}

    class _FakeFcntl:
        # Distinct nonzero flag values so the fake can tell an acquire
        # attempt (LOCK_EX | LOCK_NB) apart from the release call
        # (LOCK_UN) made by `_resolution_artifact_lock`'s own `finally`.
        LOCK_EX = 0b01
        LOCK_NB = 0b10
        LOCK_UN = 0b100

        @staticmethod
        def flock(fd, flags):
            if flags == _FakeFcntl.LOCK_UN:
                return  # release: always succeeds, not part of the acquire-retry count
            acquire_attempts["count"] += 1
            if acquire_attempts["count"] <= 2:
                raise OSError(errno_module.EACCES, "simulated: lock held by another process (EACCES)")
            # third acquire attempt: "holder" has released -- lock succeeds

    monkeypatch.setattr(extraction_module, "fcntl", _FakeFcntl)
    monkeypatch.setattr(extraction_module, "msvcrt", None)

    class _FakeStore:
        run_dir = tmp_path

    with extraction_module._resolution_artifact_lock(_FakeStore(), timeout=5.0, poll_interval=0.01):
        pass

    assert acquire_attempts["count"] == 3, "expected EACCES to be retried (as contention), not raised immediately"


# ---------------------------------------------------------------------------
# Same-run re-entry / stale-run reset (codex review findings C and C1).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Codex review finding C: resolutions must be bound to their own run and
# never silently apply to a different run's machine result.
# ---------------------------------------------------------------------------

def test_extraction_node_stale_resolution_from_prior_run_never_overrides_new_clean_machine_result(monkeypatch, tmp_path):
    """Direct reproduction + fix verification of the review's own
    finding: a stale resolution `design=STALE_OLD_VALUE` left over from a
    DIFFERENT prior run must never override a later, clean machine
    `design=NEW_VALUE` result produced by a genuinely new run."""
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])

    # A resolution left over from a completely different, older run,
    # keyed on the SAME (record_id, field_id) this new run will also
    # produce a result for.
    state["extraction_run_id"] = "stale-old-run-id"
    state["extraction_review_resolutions"] = [{
        "record_id": "rec-a", "field_id": "study_design", "outcome": "extracted",
        "value": "STALE_OLD_VALUE", "run_id": "stale-old-run-id",
        "reviewer": "R", "note": "n", "resolved_at": "2020-01-01T00:00:00Z",
    }]

    output = extraction_node(state)

    assert output["extraction_run_id"] != "stale-old-run-id"
    assert output["extraction_review_resolutions"] == []  # explicitly reset for the new run
    assert output["extraction_review_queue"] == []
    assert output["charted_data"][0]["raw_extraction"]["study_design"] == "RCT"  # NOT "STALE_OLD_VALUE"


def test_extraction_node_preserves_resolutions_when_resuming_the_same_run(monkeypatch, tmp_path):
    """Codex review round-2 finding C1's 'same-run re-entry' complaint:
    re-invoking extraction_node() for the SAME run (documents/schema/
    config unchanged) must fold in resolutions already recorded for that
    run -- excluding the already-resolved field from the rebuilt queue and
    reconstructing charted_data WITH the human override applied, never
    reopening it or discarding the resolved projection. Uses the REAL
    resolve_extraction_review_queue() to produce a genuinely, fully bound
    resolution (round-3: a same-run resolution is now re-verified in
    full, not just checked for a matching run_id/field key)."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    real_run_id = state["extraction_run_id"]
    assert state["extraction_review_queue"]  # sanity: genuinely queued on the first call

    resolution_update = resolve_extraction_review_queue(state, {("rec-a", "study_design"): {
        "outcome": "extracted", "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
        "value": "HUMAN_OVERRIDE", "evidence": _evidence(), "attested": True,
    }})
    state.update(resolution_update)
    assert state["extraction_review_queue"] == []

    second_output = extraction_node(state)
    assert second_output["extraction_run_id"] == real_run_id
    # codex review round-4: always explicitly set (never conditionally
    # omitted) -- here it must equal the SAME resolved set, recovered
    # correctly rather than reset to [].
    assert len(second_output["extraction_review_resolutions"]) == 1
    assert second_output["extraction_review_queue"] == []  # the only queued field was already resolved
    assert second_output["charted_data"][0]["raw_extraction"]["study_design"] == "HUMAN_OVERRIDE"


def test_extraction_node_rejects_incompletely_bound_same_run_resolution(monkeypatch, tmp_path):
    """Codex review round-3: a same-run resolution folded in from state
    was previously accepted after checking only its run_id and field
    key -- a fabricated entry omitting manifest/schema/original-field-
    result-hash binding must now be rejected outright."""
    state = _run_to_unclear(monkeypatch, tmp_path, _schema(), ["study_design"])
    real_run_id = state["extraction_run_id"]

    # An UNDER-SPECIFIED resolution -- missing manifest_hash,
    # extraction_schema_hash, original_field_result_hash, evidence, and
    # attested -- the exact shape the pre-round-3 same-run branch trusted.
    state["extraction_review_resolutions"] = [{
        "record_id": "rec-a", "field_id": "study_design", "outcome": "extracted",
        "value": "HUMAN_OVERRIDE", "run_id": real_run_id,
        "reviewer": "R", "note": "n", "resolved_at": "2026-01-01T00:00:00Z",
    }]

    with pytest.raises(ValueError, match="not bound to this run's verified manifest_hash"):
        extraction_node(state)


def test_extraction_node_rejects_foreign_resolution_on_same_run_reentry(monkeypatch, tmp_path):
    """A same-run resolution that does not correspond to any field the
    FRESH machine result actually flags for review is corrupted/foreign
    state -- rejected outright, not silently ignored, EVEN when it is
    otherwise correctly bound to this run's real manifest/schema hash."""
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(*, schema, batch_chunks, config):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    state = _state(cfg, [_record("rec-a", pdf_path)])
    first_output = extraction_node(state)
    state.update(first_output)
    real_run_id = state["extraction_run_id"]

    # study_design extracted CLEANLY (not queued) -- a resolution claiming
    # to resolve it for this same run, EVEN correctly bound to the real
    # manifest/schema hash, is foreign/forged state (no verified field
    # result behind it).
    state["extraction_review_resolutions"] = [{
        "record_id": "rec-a", "field_id": "study_design", "outcome": "extracted",
        "value": "FORGED", "evidence": _evidence(), "attested": True,
        "run_id": real_run_id, "manifest_hash": state["extraction_manifest_hash"],
        "extraction_schema_hash": state["extraction_schema_hash"],
        "original_field_result_hash": "0" * 64,
        "reviewer": "R", "note": "n", "resolved_at": _VALID_TS,
    }]

    with pytest.raises(ValueError, match="does not correspond to any field the verified result set"):
        extraction_node(state)


def test_project_charted_data_ignores_resolution_from_a_different_run():
    record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    result = {"document_id": "rec-a", "fields": [{"field_id": "study_design", "status": "extracted", "values": [{"value": "NEW"}]}]}
    stale_resolution = {"record_id": "rec-a", "field_id": "study_design", "outcome": "extracted", "value": "OLD", "run_id": "old-run"}
    out = _project_charted_data([record], [result], [stale_resolution], current_run_id="new-run")
    assert out[0]["raw_extraction"]["study_design"] == "NEW"


def test_project_charted_data_honors_resolution_from_the_current_run():
    record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    result = {"document_id": "rec-a", "fields": [{"field_id": "study_design", "status": "unclear", "values": []}]}
    resolution = {"record_id": "rec-a", "field_id": "study_design", "outcome": "extracted", "value": "HUMAN", "run_id": "run-1"}
    out = _project_charted_data([record], [result], [resolution], current_run_id="run-1")
    assert out[0]["raw_extraction"]["study_design"] == "HUMAN"


# ---------------------------------------------------------------------------
# _project_charted_data -- deterministic compatibility projection.
# ---------------------------------------------------------------------------

def test_project_charted_data_is_deterministic():
    record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    result = {
        "document_id": "rec-a",
        "fields": [{"field_id": "study_design", "status": "extracted", "values": [{"value": "RCT"}]}],
    }
    out_1 = _project_charted_data([record], [result], [], current_run_id=None)
    out_2 = _project_charted_data([record], [result], [], current_run_id=None)
    assert out_1 == out_2
    assert out_1[0]["raw_extraction"] == {"study_design": "RCT"}


def test_project_charted_data_multi_value_field_becomes_a_list():
    record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    result = {
        "document_id": "rec-a",
        "fields": [{"field_id": "designs", "status": "extracted", "values": [{"value": "RCT"}, {"value": "cohort"}]}],
    }
    out = _project_charted_data([record], [result], [], current_run_id=None)
    assert out[0]["raw_extraction"]["designs"] == ["RCT", "cohort"]


def test_project_charted_data_not_reported_becomes_none():
    record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    result = {"document_id": "rec-a", "fields": [{"field_id": "country", "status": "not_reported", "values": []}]}
    out = _project_charted_data([record], [result], [], current_run_id=None)
    assert out[0]["raw_extraction"]["country"] is None


def test_project_charted_data_missing_result_is_unvalidated_empty_record():
    record = Record(id="rec-a", source="test", source_id="rec-a", title="t")
    out = _project_charted_data([record], [], [], current_run_id=None)
    assert out[0]["raw_extraction"] == {}
    assert out[0]["validated"] is False
