"""Offline tests for engine.py's orchestration. Two boundaries are
monkeypatched, and ONLY these two:
  - `engine.parse_pdf` (needs a real PDF + real Docling otherwise) --
    replaced with a fixed, hand-built ParsedDocument.
  - `call_batch_fn` -- injected per the engine's own public contract (this
    is not a monkeypatch at all; injection is the designed seam), a
    scripted fake returning controlled RawBatchResponse objects.
Evidence verification, aggregation, batching, checkpointing all run for
real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import openai

import slr_pipeline.extraction.batch_cap as batch_cap_module
import slr_pipeline.extraction.engine as engine_module
from slr_pipeline.extraction.checkpoint import BlockedDocumentStore, ExtractionCheckpointStore
from slr_pipeline.extraction.engine import extract_document, run_extraction
from slr_pipeline.extraction.openai_runner import (
    BatchCallTelemetry, ExtractionCallError, ProviderModelMismatchError, RawBatchResponse, RawEvidenceItem,
    RawFieldResponse, RawFinding, OpenAIExtractionConfig, run_openai_batch_extraction,
)
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.extraction.results import FieldStatus
from slr_pipeline.extraction.schema import AllowedValue, ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind


def _chunk(index: int, text: str, section: str = "Methods") -> ParsedChunk:
    return ParsedChunk(chunk_id=f"doc-chunk-{index:04d}", order_index=index, section=section, page_min=index + 1, page_max=index + 1, text=text)


def _parsed_document(chunks: tuple[ParsedChunk, ...]) -> ParsedDocument:
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


def _categorical_schema() -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id="has_control_group", question="q", description="d", value_kind=ValueKind.CATEGORICAL,
        cardinality="single", allowed_values=(AllowedValue(value="yes", definition="Has a control group"),),
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    return ExtractionSchema(schema_id="s", version="1", fields=[field])


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


def test_extract_document_found_value(monkeypatch):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    field = outcome.result.fields[0]
    assert field.status == FieldStatus.EXTRACTED
    assert field.values[0].value == "RCT"
    assert field.values[0].evidence[0].page == "1"
    assert field.values[0].evidence[0].section == "Methods"
    assert outcome.result.is_complete is True
    assert len(outcome.result.batch_executions) == 1
    assert outcome.result.batch_executions[0].provider_model == "gpt-5.6-terra"
    assert outcome.result.batch_executions[0].total_tokens == 15
    assert outcome.result.batch_executions[0].chunk_ids == (chunk.chunk_id,)


def test_extract_document_not_found_becomes_not_reported(monkeypatch):
    chunk = _chunk(0, "Some unrelated text about funding sources.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return _not_found_response("study_design"), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    assert outcome.result.fields[0].status == FieldStatus.NOT_REPORTED


def test_extract_document_batch_failure_becomes_extraction_failed_not_not_reported(monkeypatch):
    chunk = _chunk(0, "Some text.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        raise ExtractionCallError("simulated API failure")

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    field = outcome.result.fields[0]
    assert field.status == FieldStatus.EXTRACTION_FAILED  # NOT not_reported -- requirement 8
    assert field.requires_human_review is True
    assert outcome.result.is_complete is False
    assert outcome.result.batch_executions[0].status == "failed"


def test_extract_document_unknown_chunk_id_evidence_becomes_unclear_not_silently_accepted(monkeypatch):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return _found_response("study_design", "RCT", "chunk-does-not-exist", "randomized controlled trial"), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    field = outcome.result.fields[0]
    assert field.status == FieldStatus.UNCLEAR
    assert field.requires_human_review is True


def test_extract_document_paraphrased_quotation_rejected(monkeypatch):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        # A paraphrase, not a verbatim quote -- must not verify.
        return _found_response("study_design", "RCT", chunk.chunk_id, "the study used a randomized design approach"), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    assert outcome.result.fields[0].status == FieldStatus.UNCLEAR


def test_extract_document_response_missing_field_id_marks_batch_failed(monkeypatch):
    chunk = _chunk(0, "Some text.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return RawBatchResponse(fields={}), _telemetry()  # missing "study_design" entirely

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    assert outcome.result.batch_executions[0].status == "failed"
    assert outcome.result.fields[0].status == FieldStatus.EXTRACTION_FAILED


def test_extract_document_unknown_field_id_in_response_marks_batch_failed(monkeypatch):
    chunk = _chunk(0, "Some text.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return RawBatchResponse(fields={
            "study_design": RawFieldResponse(outcome="not_found"),
            "unexpected_field": RawFieldResponse(outcome="not_found"),
        }), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    assert outcome.result.batch_executions[0].status == "failed"
    assert "unknown field_id" in outcome.result.batch_executions[0].error
    assert outcome.result.fields[0].status == FieldStatus.EXTRACTION_FAILED


def test_extract_document_not_found_with_findings_is_incoherent_batch_failure(monkeypatch):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return RawBatchResponse(fields={
            "study_design": RawFieldResponse(outcome="not_found", findings=[
                RawFinding(value="RCT", evidence=[RawEvidenceItem(chunk_id=chunk.chunk_id, quotation="randomized controlled trial")]),
            ]),
        }), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    assert outcome.result.batch_executions[0].status == "failed"
    assert "incoherent" in outcome.result.batch_executions[0].error
    assert outcome.result.fields[0].status == FieldStatus.EXTRACTION_FAILED


def test_extract_document_found_with_no_findings_is_incoherent_batch_failure(monkeypatch):
    chunk = _chunk(0, "Some text.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return RawBatchResponse(fields={"study_design": RawFieldResponse(outcome="found", findings=[])}), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call)
    assert outcome.result.batch_executions[0].status == "failed"
    assert "incoherent" in outcome.result.batch_executions[0].error


def test_extract_document_invalid_categorical_value_routed_to_review_not_extracted(monkeypatch):
    """Review finding 3's adversarial reproduction: a categorical field
    whose only allowed value is "yes" must never accept "NOT_ALLOWED" as
    a clean extraction."""
    chunk = _chunk(0, "No control group was used in this study.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return RawBatchResponse(fields={
            "has_control_group": RawFieldResponse(outcome="found", findings=[
                RawFinding(value="NOT_ALLOWED", evidence=[RawEvidenceItem(chunk_id=chunk.chunk_id, quotation="No control group was used")]),
            ]),
        }), _telemetry()

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_categorical_schema(), call_batch_fn=fake_call)
    field = outcome.result.fields[0]
    assert field.status != FieldStatus.EXTRACTED
    assert field.status == FieldStatus.UNCLEAR
    assert field.requires_human_review is True
    assert field.invalid_value_count == 1
    assert outcome.result.batch_executions[0].invalid_value_count == 1
    # Review finding C: the raw value must be queryable as STRUCTURED
    # audit data, not only present inside a formatted rationale string.
    assert len(field.invalid_values) == 1
    assert field.invalid_values[0].raw_value == "NOT_ALLOWED"
    assert field.invalid_values[0].value_kind == "categorical"
    assert field.invalid_values[0].evidence[0].chunk_id == chunk.chunk_id


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50, total_tokens=150) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeMismatchedResponse:
    """Simulates the API answering with a DIFFERENT model than requested
    -- real usage IS present on this response, exactly as a genuine
    substitution would look."""

    def __init__(self, content: str, model: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.model = model
        self.usage = _FakeUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)


class _FakeMismatchedOpenAIClient:
    def __init__(self, content: str, model: str) -> None:
        self._content = content
        self._model = model

    def __call__(self, *a, **kw):
        return self

    @property
    def chat(self):
        outer = self

        class _Completions:
            def create(self, **kwargs):
                return _FakeMismatchedResponse(outer._content, outer._model)

        class _Chat:
            completions = _Completions()

        return _Chat()


def test_extract_document_provider_mismatch_preserves_telemetry_through_real_runner(monkeypatch):
    """Review finding P's explicitly-required regression: through the
    REAL runner-to-engine boundary (not a fake telemetry object bypassing
    the real early raise) -- `call_batch_fn` here IS
    `run_openai_batch_extraction` itself, wired to a faked OpenAI client
    that answers with a substituted model. The resulting
    BatchExecutionRecord must still carry the ACTUAL returned model and
    usage, even though the batch is marked "failed" and the response
    content is rejected."""
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    valid_json = (
        '{"fields": {"study_design": {"outcome": "found", "findings": '
        '[{"value": "RCT", "confidence": "high", "rationale": "explicit", '
        f'"evidence": [{{"chunk_id": "{chunk.chunk_id}", "quotation": "randomized controlled trial"}}]}}]}}}}'
    )
    fake_client = _FakeMismatchedOpenAIClient(valid_json, model="gpt-5.6-terra-SUBSTITUTED")
    monkeypatch.setattr(openai, "OpenAI", fake_client)

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )

    def call_batch_fn(batch_chunks):
        return run_openai_batch_extraction(schema=_schema(), batch_chunks=batch_chunks, config=config)

    outcome = extract_document(document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=call_batch_fn)
    record = outcome.result.batch_executions[0]
    assert record.status == "failed"
    assert record.error == "ProviderModelMismatchError"
    assert record.provider_model == "gpt-5.6-terra-SUBSTITUTED"
    assert record.prompt_tokens == 11
    assert record.completion_tokens == 7
    assert record.total_tokens == 18


def test_extract_document_multi_batch_conflicting_values(monkeypatch):
    chunk_a = _chunk(0, "We used a randomized controlled trial design.", section="Methods")
    chunk_b = _chunk(1, "This cohort study followed participants over time.", section="Methods")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))

    call_log = []

    def fake_call(batch_chunks):
        call_log.append([c.chunk_id for c in batch_chunks])
        chunk = batch_chunks[0]
        if chunk.chunk_id == chunk_a.chunk_id:
            return _found_response("study_design", "RCT", chunk_a.chunk_id, "randomized controlled trial"), _telemetry()
        return _found_response("study_design", "cohort study", chunk_b.chunk_id, "cohort study followed participants"), _telemetry()

    # Force 2 separate batches (one chunk each) via a tiny max_batch_chars.
    outcome = extract_document(
        document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call, max_batch_chars=10,
    )
    assert len(call_log) == 2  # both batches actually called -- whole document covered, no truncation
    assert outcome.result.fields[0].status == FieldStatus.CONFLICTING


def test_run_extraction_checkpoints_and_resumes_without_recalling(monkeypatch, tmp_path):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    call_count = {"n": 0}

    def fake_call(batch_chunks):
        call_count["n"] += 1
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )

    run_id_1, results_1 = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
    )
    assert call_count["n"] == 1
    assert results_1[0].fields[0].status == FieldStatus.EXTRACTED

    run_id_2, results_2 = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
    )
    assert run_id_2 == run_id_1
    assert call_count["n"] == 1  # NOT called again -- resumed from checkpoint
    assert results_2[0].document_id == "rec-a"


def test_run_extraction_writes_manifest(monkeypatch, tmp_path):
    chunk = _chunk(0, "text")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_call(batch_chunks):
        return _not_found_response("study_design"), _telemetry()

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
    )

    from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=run_id)
    manifest = store.load_manifest()
    assert manifest is not None
    assert manifest.document_ids == ("rec-a",)
    assert manifest.checkpointed_document_ids == ("rec-a",)
    assert manifest.complete_document_ids == ("rec-a",)
    assert manifest.incomplete_document_ids == ()
    assert manifest.model_config_id == "cfg-1"
    assert manifest.model == "gpt-5.6-terra"
    assert manifest.reasoning_effort == "medium"
    assert manifest.max_completion_tokens == 4096
    assert manifest.expected_provider_model == "gpt-5.6-terra"
    assert manifest.max_batch_chars == 12_000
    assert set(manifest.checkpoint_hashes) == {"rec-a"}
    assert manifest.telemetry_summary.total_tokens == 15
    assert manifest.telemetry_summary.distinct_provider_models == ("gpt-5.6-terra",)
    assert manifest.telemetry_summary.provider_model_consistent is True

    # The checkpoint envelope itself also binds the complete model
    # configuration (review finding A) -- not just the manifest.
    checkpoint = store.load_checkpoint("rec-a")
    assert checkpoint.model == "gpt-5.6-terra"
    assert checkpoint.reasoning_effort == "medium"
    assert checkpoint.max_completion_tokens == 4096
    assert checkpoint.run_id == run_id


def test_run_extraction_provider_model_inconsistency_surfaces_in_telemetry_summary(monkeypatch, tmp_path):
    """If a batch's own telemetry ever carries a provider_model that
    does not match the run's expected_provider_model (e.g. captured
    before openai_runner.py's own per-call check existed, or replayed
    from an inconsistent source), the manifest's telemetry_summary must
    surface that rather than silently reporting consistency."""
    chunk = _chunk(0, "text")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_call(batch_chunks):
        return _not_found_response("study_design"), BatchCallTelemetry(
            provider_model="gpt-5.6-terra-UNEXPECTED", prompt_tokens=1, completion_tokens=1, total_tokens=2,
        )

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
    )

    from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=run_id)
    manifest = store.load_manifest()
    assert manifest.telemetry_summary.provider_model_consistent is False
    assert "gpt-5.6-terra-UNEXPECTED" in manifest.telemetry_summary.distinct_provider_models


def test_run_extraction_provider_mismatch_error_surfaces_in_manifest_telemetry(monkeypatch, tmp_path):
    """Review finding P: a REAL `ProviderModelMismatchError` (the kind
    `run_openai_batch_extraction` actually raises) must still leave a
    telemetry trace on the published manifest -- `provider_mismatch_count`
    counts it explicitly, and `provider_model_consistent` becomes False
    because the substituted model is now part of `distinct_provider_models`
    (review finding P's core requirement: a run must never be labeled
    provider-consistent when a recorded provider response violated the
    expected-model policy)."""
    chunk = _chunk(0, "text")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_call(batch_chunks):
        raise ProviderModelMismatchError(
            expected_provider_model="gpt-5.6-terra", actual_provider_model="gpt-5.6-terra-SUBSTITUTED",
            prompt_tokens=11, completion_tokens=7, total_tokens=18,
        )

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
    )

    from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=run_id)
    manifest = store.load_manifest()
    assert manifest.telemetry_summary.provider_mismatch_count == 1
    assert manifest.telemetry_summary.provider_model_consistent is False
    assert "gpt-5.6-terra-SUBSTITUTED" in manifest.telemetry_summary.distinct_provider_models
    assert manifest.telemetry_summary.total_tokens == 18


def test_run_extraction_propagates_parse_error_no_silent_fallback(monkeypatch, tmp_path):
    from slr_pipeline.extraction.parsing import DocumentParseError

    def failing_parse(*a, **k):
        raise DocumentParseError("simulated docling failure")

    monkeypatch.setattr(engine_module, "parse_pdf", failing_parse)
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_call(batch_chunks):
        raise AssertionError("must never be called -- parse failed before any batch call")

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    with pytest.raises(DocumentParseError):
        run_extraction(
            documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
            output_dir=tmp_path / "out", model_config=config,
        )


def test_run_extraction_rejects_duplicate_document_ids(tmp_path):
    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )

    def fake_call(batch_chunks):
        raise AssertionError("must never be called -- duplicate-id check happens before any work")

    with pytest.raises(ValueError, match="duplicate document_id"):
        run_extraction(
            documents=[("rec-a", "a.pdf"), ("rec-a", "b.pdf")], schema=_schema(), call_batch_fn=fake_call,
            output_dir=tmp_path / "out", model_config=config,
        )


def test_run_extraction_different_max_batch_chars_do_not_share_run_id_or_checkpoints(monkeypatch, tmp_path):
    """Review finding 1's adversarial reproduction: changing
    `max_batch_chars` between two otherwise-identical `run_extraction()`
    calls must produce a different run_id, so the second run cannot
    silently reuse the first run's checkpoints (which were produced under
    a different, model-visible batching plan)."""
    chunk_a = _chunk(0, "We used a randomized controlled trial design.")
    chunk_b = _chunk(1, "This cohort study followed participants over time.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    call_count = {"n": 0}

    def fake_call(batch_chunks):
        call_count["n"] += 1
        chunk = batch_chunks[0]
        cid = chunk_a.chunk_id if chunk.chunk_id == chunk_a.chunk_id else chunk_b.chunk_id
        return _found_response("study_design", "RCT", cid, chunk.text[:20]), _telemetry()

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )

    run_id_1, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config, max_batch_chars=10,
    )
    calls_after_run_1 = call_count["n"]

    run_id_2, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config, max_batch_chars=10_000,
    )
    assert run_id_2 != run_id_1
    # A genuinely different (non-shared) checkpoint namespace: the second
    # run had to actually call the provider again, not resume.
    assert call_count["n"] > calls_after_run_1


_FAKE_DIGEST = "0" * 64


def test_extract_document_max_batches_requires_expected_artifacts_manifest_digest(monkeypatch):
    chunk = _chunk(0, "text")
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        raise AssertionError("must never be called -- the missing-digest check happens first")

    with pytest.raises(ValueError, match="expected_artifacts_manifest_digest is required"):
        extract_document(
            document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call, max_batches=1,
        )


def test_extract_document_max_batches_within_bound_extracts_normally(monkeypatch):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def fake_call(batch_chunks):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    outcome = extract_document(
        document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call,
        max_batches=8, expected_artifacts_manifest_digest=_FAKE_DIGEST,
    )
    assert outcome.result.fields[0].status == FieldStatus.EXTRACTED


def test_extract_document_max_batches_exceeded_raises_before_any_billed_call(monkeypatch):
    # Two chunks, tiny max_batch_chars -- forces 2 batches, exceeding a cap of 1.
    chunk_a = _chunk(0, "a" * 20)
    chunk_b = _chunk(1, "b" * 20)
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))

    def fake_call(batch_chunks):
        raise AssertionError("must never be called -- the cap is exceeded before any batch is billed")

    from slr_pipeline.extraction.batch_cap import ExtractionBatchCapExceededError

    with pytest.raises(ExtractionBatchCapExceededError) as excinfo:
        extract_document(
            document_id="rec-a", pdf_path="unused.pdf", schema=_schema(), call_batch_fn=fake_call,
            max_batches=1, expected_artifacts_manifest_digest=_FAKE_DIGEST, max_batch_chars=20,
        )
    assert len(excinfo.value.batches) == 2


def test_run_extraction_blocks_document_exceeding_cap_without_billing_or_checkpointing(monkeypatch, tmp_path):
    chunk_a = _chunk(0, "a" * 20)
    chunk_b = _chunk(1, "b" * 20)
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    call_count = {"n": 0}

    def fake_call(batch_chunks):
        call_count["n"] += 1
        return _not_found_response("study_design"), _telemetry()

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id, results = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
        max_batches=1, expected_artifacts_manifest_digest=_FAKE_DIGEST, max_batch_chars=20,
    )
    assert call_count["n"] == 0  # never billed
    assert results == []  # blocked document excluded from results

    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=run_id)
    assert not store.has_checkpoint("rec-a")
    assert store.load_manifest() is None  # nothing attempted -- no manifest published

    blocked_store = BlockedDocumentStore(output_dir=tmp_path / "out", run_id=run_id)
    assert blocked_store.is_blocked("rec-a")


def test_run_extraction_restart_skips_previously_blocked_document_without_reattempting(monkeypatch, tmp_path):
    chunk_a = _chunk(0, "a" * 20)
    chunk_b = _chunk(1, "b" * 20)
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    parse_call_count = {"n": 0}

    def counting_parse_pdf(*a, **k):
        parse_call_count["n"] += 1
        return _parsed_document((chunk_a, chunk_b))

    monkeypatch.setattr(batch_cap_module, "parse_pdf", counting_parse_pdf)

    def fake_call(batch_chunks):
        raise AssertionError("must never be called -- capped document")

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id_1, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
        max_batches=1, expected_artifacts_manifest_digest=_FAKE_DIGEST, max_batch_chars=20,
    )
    assert parse_call_count["n"] == 1

    run_id_2, results_2 = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
        max_batches=1, expected_artifacts_manifest_digest=_FAKE_DIGEST, max_batch_chars=20,
    )
    assert run_id_2 == run_id_1
    assert parse_call_count["n"] == 1  # not re-parsed/re-checked -- skipped via BlockedDocumentStore
    assert results_2 == []


def test_run_extraction_records_artifacts_manifest_digest_on_checkpoint_and_manifest(monkeypatch, tmp_path):
    chunk = _chunk(0, "We used a randomized controlled trial design.")
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_call(batch_chunks):
        return _found_response("study_design", "RCT", chunk.chunk_id, "randomized controlled trial"), _telemetry()

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id, results = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
        max_batches=8, expected_artifacts_manifest_digest=_FAKE_DIGEST,
    )
    assert results[0].fields[0].status == FieldStatus.EXTRACTED

    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=run_id)
    checkpoint = store.load_checkpoint("rec-a")
    assert checkpoint.artifacts_manifest_digest == _FAKE_DIGEST
    manifest = store.load_manifest()
    assert manifest.artifacts_manifest_digest == _FAKE_DIGEST


def test_run_extraction_different_artifacts_manifest_digest_do_not_share_run_id(monkeypatch, tmp_path):
    """A differently-pinned Docling artifacts digest must not silently
    reuse another digest's checkpoint namespace -- the run identity binds
    it (mirrors the existing max_batch_chars run-identity regression)."""
    chunk = _chunk(0, "text")
    monkeypatch.setattr(batch_cap_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    def fake_call(batch_chunks):
        return _not_found_response("study_design"), _telemetry()

    config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    run_id_1, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
        max_batches=8, expected_artifacts_manifest_digest="a" * 64,
    )
    run_id_2, _ = run_extraction(
        documents=[("rec-a", str(pdf_path))], schema=_schema(), call_batch_fn=fake_call,
        output_dir=tmp_path / "out", model_config=config,
        max_batches=8, expected_artifacts_manifest_digest="b" * 64,
    )
    assert run_id_2 != run_id_1
