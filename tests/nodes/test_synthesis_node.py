"""Tests for the structured, evidence-verified synthesis_node -- the Stage 3
claim-construction slice of quality_appraisal_synthesis_plan_2026-08-22.md's
own §3.2 ("Structural traceability, not claim validation").

Every OpenAI call synthesis_node itself makes is replaced with a fake
client dispatching on the rendered system prompt (map / refine / IMRAD
narrative) -- no real network/credential activity. `extraction_node` is
exercised for REAL (with only `parse_pdf` and the OpenAI provider call
faked, exactly like tests/nodes/test_extraction_node.py) so a genuine,
verified resolution artifact -- and a real, non-empty evidence_index --
exists on disk for synthesis_node's own map step to cite.

This is a representative test suite, not an exhaustive transcription of
every acceptance-criterion fixture named across the plan's 23 review
rounds (~50+ named regressions) -- it covers: corpus/coverage selection
(unchanged from the free-text implementation, including the no-fallback-
on-missing/all-not_reported-extraction and canonical-record-order variants),
arbitrary (non-hardcoded) extraction-schema fields reaching both the claim-
backed map/refine pipeline and the legacy IMRAD narrative, the verified-
artifact boundary against a directly-polluted raw state["charted_data"],
the golden-path map/refine claim-construction pipeline with real evidence
citations (including a real rapid-review run where quality-appraisal
evidence genuinely reaches a claim's own references), refine batching
across multiple batches, legacy IMRAD narrative (success/no-call/disabled),
the four IMRAD/claim-backed-results flag combinations, and a representative
sample of verify_synthesis_output_sources()'s own tamper-detection checks
(exercised directly against hand-tampered SynthesisOutput instances, the
same technique the plan's own acceptance criteria use) -- including an
adversarial map event and an adversarial refine event each produced against
a genuinely DIFFERENT call context (model/prompt config; prior accumulator
state) from what they claim, proving the verifier reconstructs the complete
call identity rather than trusting input/output content hashes alone.
Codex code-review round 2 (2026-08-26) named three specific still-missing
regressions from the old free-text suite (arbitrary-schema-fields for
map/refine, arbitrary-schema-fields for legacy IMRAD, raw-state-pollution)
and one specific mechanism gap (refine provenance) -- all four are covered
above; high-risk lifecycle/tamper cases beyond these remain a representative
sample, not an exhaustive one.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import slr_pipeline.extraction.engine as engine_module
from slr_pipeline.ai_traceability import AIUseEventStore, build_ai_use_event
from slr_pipeline.config import ReviewConfig, SynthesisConfig
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.openai_runner import (
    BatchCallTelemetry, OpenAIExtractionConfig, RawBatchResponse, RawEvidenceItem, RawFieldResponse, RawFinding,
)
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind
from slr_pipeline.state import (
    EvidenceReference, MapOutcomeRecord, Record, SupportingFieldSelector, SynthesisOutput, build_map_answer,
    initial_state,
)

import slr_pipeline.nodes.extraction as extraction_module
from slr_pipeline.nodes.extraction import extraction_node

import slr_pipeline.nodes.synthesis as synthesis_module
from slr_pipeline.nodes.synthesis import synthesis_node, verify_synthesis_output_sources


# ---------------------------------------------------------------------------
# Fake OpenAI client -- dispatches on the rendered system prompt so one
# client can serve map/refine/IMRAD calls within a single test.
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.model = "fake-model"


def _relevant_map_response(answer_text: str, selectors: List[Dict[str, str]]) -> str:
    return json.dumps({"relevant": True, "answer": answer_text, "supporting_selectors": selectors})


def _not_relevant_map_response() -> str:
    return json.dumps({"relevant": False})


def _refine_response(claim_updates: List[Dict[str, Any]]) -> str:
    return json.dumps({"claim_updates": claim_updates})


def _make_dispatching_openai_class(responder: Callable[[Dict[str, Any]], str], captured: List[Dict[str, Any]]):
    """`responder(kwargs) -> content_str` inspects kwargs['messages'][0]['content'] (the system
    prompt) to decide which stage this call belongs to and returns the right canned JSON/text."""
    class _FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return _FakeResponse(responder(kwargs))

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()
            self.base_url = "https://fake.example/v1"

    return _FakeClient


def _all_prompt_text(captured: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(m.get("content", "")) for call in captured for m in call.get("messages", [])
    )


def _system_prompt(kwargs: Dict[str, Any]) -> str:
    return kwargs["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Real (parse/provider-faked) extraction setup -- produces a genuine,
# verified resolution artifact (with a real, non-empty evidence_index) on
# disk, matching tests/nodes/test_extraction_node.py's own convention.
# ---------------------------------------------------------------------------

def _extraction_schema(field_ids: List[str]) -> ExtractionSchema:
    fields = [
        ExtractionFieldSpec(
            field_id=fid, question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
            provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
        )
        for fid in field_ids
    ]
    return ExtractionSchema(schema_id="s", version="1", fields=fields)


def _model_config() -> OpenAIExtractionConfig:
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )


def _telemetry() -> BatchCallTelemetry:
    return BatchCallTelemetry(provider_model="gpt-5.6-terra", prompt_tokens=1, completion_tokens=1, total_tokens=2)


def _run_extraction_for_synthesis(
    monkeypatch, tmp_path, *,
    review_type: str,
    schema_field_ids: List[str],
    field_values: Dict[str, Dict[str, str]],
    research_questions: List[str] = None,
    synthesis_config: SynthesisConfig = None,
    with_qa_model_config: bool = False,
) -> Dict[str, Any]:
    """Runs a REAL extraction_node() over one record per key of `field_values`, each with a canned
    CLEAN (verbatim-quoted) result for every field named in its own sub-dict (any schema field not
    named there comes back `not_found` -> `not_reported` for that record) -- producing a genuine,
    verified, published resolution artifact on disk. Returns the merged state, ready to pass into
    synthesis_node."""
    schema = _extraction_schema(schema_field_ids)

    def _chunk_text(fields: Dict[str, str]) -> str:
        return " ".join(f"[{fid}] {value}" for fid, value in fields.items()) or "placeholder text, no fields."

    chunks_by_record = {
        record_id: ParsedChunk(
            chunk_id=f"{record_id}-chunk-0000", order_index=0, section="Methods",
            page_min=1, page_max=1, text=_chunk_text(fields),
        )
        for record_id, fields in field_values.items()
    }

    def fake_parse(pdf_path, *a, **k):
        record_id = Path(pdf_path).stem
        chunk = chunks_by_record[record_id]
        return ParsedDocument(
            document_sha256="a" * 64, parser_name="docling", parser_version="2.61.2",
            parser_options=DoclingParserOptions(), chunks=(chunk,),
        )

    monkeypatch.setattr(engine_module, "parse_pdf", fake_parse)

    def fake_call(*, schema, batch_chunks, config):
        chunk = batch_chunks[0]
        record_id = chunk.chunk_id.rsplit("-chunk-", 1)[0]
        fields = field_values.get(record_id, {})
        response_fields = {}
        for fid in schema_field_ids:
            value = fields.get(fid)
            if value:
                quotation = f"[{fid}] {value}"
                response_fields[fid] = RawFieldResponse(outcome="found", findings=[
                    RawFinding(value=value, confidence="high", rationale="x",
                               evidence=[RawEvidenceItem(chunk_id=chunk.chunk_id, quotation=quotation)]),
                ])
            else:
                response_fields[fid] = RawFieldResponse(outcome="not_found")
        return RawBatchResponse(fields=response_fields), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)

    records = []
    for record_id in field_values:
        pdf_path = tmp_path / f"{record_id}.pdf"
        pdf_path.write_bytes(b"fake pdf bytes")
        records.append(Record(
            id=record_id, source="s", source_id=record_id, title=f"Study {record_id}",
            full_text_path=str(pdf_path), full_text_status="parsed",
        ))

    cfg_kwargs: Dict[str, Any] = dict(
        title="t", review_type=review_type, output_dir=str(tmp_path / "out"),
        extraction_schema=schema.model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(_model_config()),
        synthesis=synthesis_config or SynthesisConfig(run_bertopic=False, generate_evidence_map=False),
    )
    if research_questions:
        cfg_kwargs["research_questions"] = research_questions
    if with_qa_model_config:
        cfg_kwargs["quality_appraisal_model_config"] = dataclasses.asdict(_model_config())
    cfg = ReviewConfig(**cfg_kwargs)

    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [r.to_dict() for r in records]
    output = extraction_node(state)
    state.update(output)
    assert state["extraction_review_queue"] == [], "setup failed: expected a clean, unqueued extraction"
    # synthesis_node() (unlike extraction_node) resolves/requires review_run_id -- outside a real
    # LangGraph invocation (no active thread_id), state must already carry one explicitly.
    state["review_run_id"] = "test-review-run"
    return state


def _run_extraction_and_qa_for_synthesis(
    monkeypatch, tmp_path, *,
    review_type: str,
    schema_field_ids: List[str],
    field_values: Dict[str, Dict[str, str]],
    research_questions: List[str] = None,
    synthesis_config: SynthesisConfig = None,
) -> Dict[str, Any]:
    """Like `_run_extraction_for_synthesis()`, but ALSO runs a REAL `quality_appraisal_node()`
    (teaching checklist, every one of `TEACHING_APPRAISAL_ITEMS` genuinely `"yes"` with real,
    verifier-checked evidence) -- producing a genuine, verified QA resolution artifact, with its own
    real, non-empty `evidence_index`, on disk. Required for `review_type in ('rapid', 'slr')`, which
    `synthesis_node()` now REQUIRES a verified QA artifact for (§6 QA binding) -- exercises the path
    the reviewer's own finding named as missing: a real rapid/SLR run where QA evidence genuinely
    reaches a claim's own `references`."""
    import slr_pipeline.nodes.quality_appraisal as qa_module
    from slr_pipeline.nodes.quality_appraisal import (
        RawAppraisalEvidenceItem, RawTeachingBatchResponse, RawTeachingItemBatchResponse,
        TEACHING_APPRAISAL_ITEMS, quality_appraisal_node,
    )

    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type=review_type, schema_field_ids=schema_field_ids,
        field_values=field_values, research_questions=research_questions,
        synthesis_config=synthesis_config, with_qa_model_config=True)

    qa_chunks_by_record = {
        record_id: ParsedChunk(chunk_id=f"{record_id}-qa-chunk-0000", order_index=0, section="Methods",
                                page_min=1, page_max=1,
                                text="The research question and methods are clearly reported in this study.")
        for record_id in field_values
    }

    def fake_qa_parse(pdf_path, *a, **k):
        record_id = Path(pdf_path).stem
        chunk = qa_chunks_by_record[record_id]
        return ParsedDocument(document_sha256="b" * 64, parser_name="docling", parser_version="2.61.2",
                               parser_options=DoclingParserOptions(), chunks=(chunk,))

    monkeypatch.setattr(qa_module, "parse_pdf", fake_qa_parse)

    def fake_teaching_batch(*, record_title, batch_chunks, config):
        chunk = batch_chunks[0]
        return RawTeachingBatchResponse(items={
            item["item_id"]: RawTeachingItemBatchResponse(
                outcome="yes", rationale="clearly and explicitly reported",
                evidence=[RawAppraisalEvidenceItem(chunk_id=chunk.chunk_id, quotation=chunk.text)])
            for item in TEACHING_APPRAISAL_ITEMS
        })

    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", fake_teaching_batch)

    qa_output = quality_appraisal_node(state)
    state.update(qa_output)
    assert state["quality_appraisal_review_queue"] == [], "QA setup failed: expected a clean, unqueued appraisal"
    return state


# ---------------------------------------------------------------------------
# Corpus/coverage selection -- unchanged from the free-text implementation.
# ---------------------------------------------------------------------------

def test_synthesis_node_scoping_uses_full_text_records_not_included_records(monkeypatch, tmp_path):
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-ft": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=True),
    )
    excluded_record = Record(
        id="rec-excluded", source="s", source_id="rec-excluded", title="Excluded study",
        abstract="SECRET_ABSTRACT_TEXT_NEVER_SEEN_BY_SYNTHESIS",
    )
    state["included_records"] = state["full_text_records"] + [excluded_record.to_dict()]

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)

    assert output["synthesis"]["evidence_map"]["n_studies"] == 1  # NOT 2
    assert "SECRET_ABSTRACT_TEXT_NEVER_SEEN_BY_SYNTHESIS" not in _all_prompt_text(captured)


def test_synthesis_node_rapid_uses_full_text_records_not_included_records(monkeypatch, tmp_path):
    """Codex code-review finding 2 (2026-08-26): the previous version of this test actually
    constructed a SCOPING review despite its own name -- genuinely exercises review_type='rapid'
    here, which requires a real, verified QA artifact (§6 QA binding)."""
    state = _run_extraction_and_qa_for_synthesis(
        monkeypatch, tmp_path, review_type="rapid", schema_field_ids=["study_design"],
        field_values={"rec-ft": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=True),
    )
    excluded_record = Record(
        id="rec-excluded", source="s", source_id="rec-excluded", title="Excluded study",
        abstract="SECRET_ABSTRACT_TEXT_NEVER_SEEN_BY_SYNTHESIS",
    )
    state["included_records"] = state["full_text_records"] + [excluded_record.to_dict()]

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)

    assert output["synthesis"]["evidence_map"]["n_studies"] == 1  # NOT 2
    assert "SECRET_ABSTRACT_TEXT_NEVER_SEEN_BY_SYNTHESIS" not in _all_prompt_text(captured)


def test_synthesis_node_slr_also_uses_full_text_records(monkeypatch, tmp_path):
    state = _run_extraction_and_qa_for_synthesis(
        monkeypatch, tmp_path, review_type="slr", schema_field_ids=["study_design"],
        field_values={"rec-ft": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=True),
    )
    other_record = Record(id="rec-other", source="s", source_id="rec-other", title="t2")
    state["included_records"] = state["full_text_records"] + [other_record.to_dict()]

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)
    assert output["synthesis"]["evidence_map"]["n_studies"] == 1


def test_synthesis_node_rapid_qa_evidence_reaches_a_claim(monkeypatch, tmp_path):
    """Codex code-review finding 2 (2026-08-26): there was no successful rapid/SLR synthesis test
    proving QA evidence reaches a claim. A real, verified `quality_appraisal_node()` run makes every
    teaching-checklist item genuinely 'yes' with real evidence; the map call cites BOTH an extraction
    field and a quality_appraisal field, and the resulting claim's own `references` must include a
    real `artifact_kind='quality_appraisal'` entry, structurally traced back to the real QA run."""
    state = _run_extraction_and_qa_for_synthesis(
        monkeypatch, tmp_path, review_type="rapid", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        research_questions=["What designs were used?"],
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False, refine_batch_size=5),
    )
    import slr_pipeline.nodes.quality_appraisal as _qa_module
    qa_field_id = _qa_module.TEACHING_APPRAISAL_ITEMS[0]["item_id"]

    import re as _re

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if "assisting a systematic literature review" in sysprompt:
            return _relevant_map_response(
                "This study reports a design and was appraised as methodologically sound.",
                [{"artifact_kind": "extraction", "field_id": "study_design"},
                 {"artifact_kind": "quality_appraisal", "field_id": qa_field_id}])
        if "drafting the synthesis section" in sysprompt:
            user_content = kw["messages"][1]["content"]
            new_ids = _re.findall(r"(answer-[0-9a-f]{64})",
                                    user_content.split("New batch of per-study answers:", 1)[1])
            return _refine_response([{"claim_index": None, "claim_text": "The included study was RCT-designed and methodologically sound.",
                                       "final_support_answer_ids": new_ids}])
        raise AssertionError(f"unexpected call: {sysprompt[:80]}")

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(responder, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    assert len(syn["claims"]) == 1
    reference_kinds = {ref["artifact_kind"] for ref in syn["claims"][0]["references"]}
    assert reference_kinds == {"extraction", "quality_appraisal"}
    qa_run_id = syn["quality_appraisal_run_id"]
    assert qa_run_id is not None
    qa_refs = [ref for ref in syn["claims"][0]["references"] if ref["artifact_kind"] == "quality_appraisal"]
    assert qa_refs[0]["artifact_run_id"] == qa_run_id
    assert qa_refs[0]["field_id"] == qa_field_id


def test_synthesis_node_rapid_raises_when_full_text_record_missing_charted_data(monkeypatch):
    cfg = ReviewConfig(title="t", review_type="rapid")
    record = Record(id="rec-a", source="s", source_id="rec-a", title="t")
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [record.to_dict()]
    state["review_run_id"] = "test-review-run"

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    with pytest.raises(ValueError, match="missing="):
        synthesis_node(state)
    assert captured == []  # no LLM call was ever made


def test_synthesis_node_rapid_rejects_duplicate_full_text_record_ids(monkeypatch):
    record = Record(id="rec-a", source="s", source_id="rec-a", title="t")
    cfg = ReviewConfig(title="t", review_type="rapid")
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [record.to_dict(), record.to_dict()]
    state["extraction_run_id"] = "extrun-fake"
    state["review_run_id"] = "test-review-run"
    monkeypatch.setattr(
        synthesis_module, "load_verified_extraction_resolution_artifact",
        lambda s: {"resolved_projection": [{"record_id": "rec-a", "raw_extraction": {}}],
                   "run_id": "extrun-fake", "evidence_index": {}},
    )
    with pytest.raises(ValueError, match="duplicate record_id"):
        synthesis_node(state)


def test_synthesis_node_rapid_rejects_duplicate_charted_data_ids(monkeypatch):
    record = Record(id="rec-a", source="s", source_id="rec-a", title="t")
    cfg = ReviewConfig(title="t", review_type="rapid")
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [record.to_dict()]
    state["extraction_run_id"] = "extrun-fake"
    state["review_run_id"] = "test-review-run"
    monkeypatch.setattr(
        synthesis_module, "load_verified_extraction_resolution_artifact",
        lambda s: {"resolved_projection": [
            {"record_id": "rec-a", "raw_extraction": {}},
            {"record_id": "rec-a", "raw_extraction": {}},
        ], "run_id": "extrun-fake", "evidence_index": {}},
    )
    with pytest.raises(ValueError, match="charted_data contains duplicate"):
        synthesis_node(state)


def test_synthesis_node_default_bertopic_never_reads_abstract_for_rapid(monkeypatch, tmp_path):
    """Codex code-review finding 2 (2026-08-26): this test also actually constructed a SCOPING
    review despite its own name -- genuinely exercises review_type='rapid' here (real QA artifact
    required, §6). SynthesisConfig.run_bertopic defaults to True -- the DEFAULT rapid/SLR synthesis
    path must never feed BERTopic record.abstract/.title; only verified extracted content."""
    state = _run_extraction_and_qa_for_synthesis(
        monkeypatch, tmp_path, review_type="rapid", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(),  # DEFAULT config -- run_bertopic=True
    )
    records = [Record.from_dict(r) for r in state["full_text_records"]]
    records[0].abstract = "SECRET_ABSTRACT_MARKER_NEVER_FED_TO_BERTOPIC"
    state["full_text_records"] = [r.to_dict() for r in records]

    captured_bertopic_texts: List[str] = []

    def fake_run_bertopic(records, n_topics, texts):
        captured_bertopic_texts.extend(texts)
        return []

    monkeypatch.setattr(synthesis_module, "_run_bertopic", fake_run_bertopic)
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    synthesis_node(state)

    assert captured_bertopic_texts  # BERTopic really was invoked (default run_bertopic=True)
    assert "SECRET_ABSTRACT_MARKER_NEVER_FED_TO_BERTOPIC" not in " ".join(captured_bertopic_texts)
    assert "study_design: RCT" in captured_bertopic_texts[0]


def test_synthesis_node_default_bertopic_never_reads_abstract_for_scoping(monkeypatch, tmp_path):
    """Mirrors test_synthesis_node_default_bertopic_never_reads_abstract_for_rapid exactly, for
    scoping (no QA artifact involved/required)."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(),  # DEFAULT config -- run_bertopic=True
    )
    records = [Record.from_dict(r) for r in state["full_text_records"]]
    records[0].abstract = "SECRET_ABSTRACT_MARKER_NEVER_FED_TO_BERTOPIC"
    state["full_text_records"] = [r.to_dict() for r in records]

    captured_bertopic_texts: List[str] = []

    def fake_run_bertopic(records, n_topics, texts):
        captured_bertopic_texts.extend(texts)
        return []

    monkeypatch.setattr(synthesis_module, "_run_bertopic", fake_run_bertopic)
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    synthesis_node(state)

    assert captured_bertopic_texts  # BERTopic really was invoked (default run_bertopic=True)
    assert "SECRET_ABSTRACT_MARKER_NEVER_FED_TO_BERTOPIC" not in " ".join(captured_bertopic_texts)
    assert "study_design: RCT" in captured_bertopic_texts[0]


def test_synthesis_node_bertopic_excludes_evidence_empty_records(monkeypatch, tmp_path):
    """A record with no verified extracted content must be EXCLUDED from BERTopic entirely, not
    padded in with a placeholder."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}, "rec-b": {}},
        synthesis_config=SynthesisConfig(run_bertopic=True, generate_evidence_map=False),
    )
    captured_bertopic_record_ids: List[str] = []

    def fake_run_bertopic(records, n_topics, texts):
        captured_bertopic_record_ids.extend(r.id for r in records)
        return []

    monkeypatch.setattr(synthesis_module, "_run_bertopic", fake_run_bertopic)
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)

    assert captured_bertopic_record_ids == ["rec-a"]
    assert output["synthesis"]["bertopic_excluded_empty_evidence_count"] == 1


def test_synthesis_node_bertopic_excluded_count_zero_when_all_records_have_evidence(monkeypatch, tmp_path):
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=True, generate_evidence_map=False),
    )
    monkeypatch.setattr(synthesis_module, "_run_bertopic", lambda records, n_topics, texts: [])
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)
    assert output["synthesis"]["bertopic_excluded_empty_evidence_count"] == 0


def test_synthesis_node_scoping_raises_when_full_text_record_missing_charted_data(monkeypatch):
    """Scoping shares rapid/SLR's no-fallback-when-extraction-absent contract -- mirrors
    test_synthesis_node_rapid_raises_when_full_text_record_missing_charted_data."""
    cfg = ReviewConfig(title="t", review_type="scoping")
    record = Record(id="rec-a", source="s", source_id="rec-a", title="t")
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [record.to_dict()]
    state["review_run_id"] = "test-review-run"

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    with pytest.raises(ValueError, match="missing="):
        synthesis_node(state)
    assert captured == []  # no LLM call was ever made


def test_synthesis_node_rapid_never_falls_back_to_abstract_when_all_fields_not_reported(monkeypatch, tmp_path):
    """A normal all-not_reported EXTRACTION outcome (the machine looked and found nothing -- not a
    malformed/missing projection) must never fall back to the record's abstract/title, and the
    not-reported field itself must never appear in the map context (real QA evidence still exists for
    this record, via the required QA artifact, so a map call for the QA-backed selector is expected)."""
    state = _run_extraction_and_qa_for_synthesis(
        monkeypatch, tmp_path, review_type="rapid", schema_field_ids=["study_design"],
        field_values={"rec-a": {}},  # nothing found -> study_design == not_reported
        research_questions=["What design was used?"],
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False),
    )
    assert state["charted_data"][0]["raw_extraction"] == {"study_design": None}

    records = [Record.from_dict(r) for r in state["full_text_records"]]
    records[0].abstract = "SECRET_ABSTRACT_MARKER_NEVER_SEEN_BY_SYNTHESIS"
    state["full_text_records"] = [r.to_dict() for r in records]

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    synthesis_node(state)

    prompt_text = _all_prompt_text(captured)
    assert "SECRET_ABSTRACT_MARKER_NEVER_SEEN_BY_SYNTHESIS" not in prompt_text
    assert "study_design:" not in prompt_text  # not_reported extraction field never reaches the prompt


def test_synthesis_node_record_order_is_canonical_not_state_order(monkeypatch, tmp_path):
    """Reversing state["full_text_records"] must not change synthesis_record_ids, BERTopic's own
    per-record text order, or evidence_map's own characteristics_table order -- all of them must
    follow the verified extraction artifact's own canonical (manifest) order, never mutable state's."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}, "rec-b": {"study_design": "Cohort"}},
        research_questions=["What is the study design?"],
        synthesis_config=SynthesisConfig(run_bertopic=True, generate_evidence_map=True),
    )
    canonical_order = [c["record_id"] for c in state["charted_data"]]
    assert set(canonical_order) == {"rec-a", "rec-b"}

    def _first_list_marker(texts: List[str]) -> str:
        first = texts[0]
        if "RCT" in first:
            return "rec-a"
        assert "Cohort" in first, f"unexpected BERTopic text: {first!r}"
        return "rec-b"

    bertopic_texts_forward: List[str] = []
    monkeypatch.setattr(
        synthesis_module, "_run_bertopic",
        lambda records, n_topics, texts: bertopic_texts_forward.extend(texts) or [],
    )
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))
    forward_output = synthesis_node(state)
    forward_evidence_order = [row["citation"] for row in forward_output["synthesis"]["evidence_map"]["characteristics_table"]]
    forward_record_ids_order = forward_output["synthesis"]["synthesis_record_ids"]
    forward_bertopic_first = _first_list_marker(bertopic_texts_forward)

    reversed_state = dict(state)
    reversed_state["full_text_records"] = list(reversed(state["full_text_records"]))
    bertopic_texts_reversed: List[str] = []
    monkeypatch.setattr(
        synthesis_module, "_run_bertopic",
        lambda records, n_topics, texts: bertopic_texts_reversed.extend(texts) or [],
    )
    captured2: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured2))
    reversed_output = synthesis_node(reversed_state)
    reversed_evidence_order = [row["citation"] for row in reversed_output["synthesis"]["evidence_map"]["characteristics_table"]]
    reversed_record_ids_order = reversed_output["synthesis"]["synthesis_record_ids"]
    reversed_bertopic_first = _first_list_marker(bertopic_texts_reversed)

    assert forward_evidence_order == reversed_evidence_order
    assert forward_record_ids_order == reversed_record_ids_order
    assert forward_bertopic_first == reversed_bertopic_first


# ---------------------------------------------------------------------------
# 2026-08-22 correction-pass finding: _citation_for()'s own fallback used to
# reimplement scoping_charting_adapter.derive_author_year()'s exact logic
# inline instead of calling it -- leaving the adapter function unused in
# production. Now delegates directly.
# ---------------------------------------------------------------------------

def test_citation_for_delegates_to_derive_author_year_when_not_charted():
    from slr_pipeline.adapters.scoping_charting_adapter import derive_author_year
    from slr_pipeline.nodes.synthesis import _citation_for

    record = Record(id="r1", source="test", source_id="r1", title="A Study of Things", authors=["Smith, J."], year=2021)
    assert _citation_for(record, {}) == derive_author_year(record)

    no_author_record = Record(id="r2", source="test", source_id="r2", title="Another Study", year=2019)
    assert _citation_for(no_author_record, {}) == derive_author_year(no_author_record)


def test_citation_for_prefers_charted_author_year_when_present():
    from slr_pipeline.nodes.synthesis import _citation_for

    record = Record(id="r1", source="test", source_id="r1", title="A Study of Things", authors=["Smith, J."], year=2021)
    assert _citation_for(record, {"author_year": "Jones et al., 2020"}) == "Jones et al., 2020"


# ---------------------------------------------------------------------------
# Golden-path map/refine claim construction: real evidence citations, real
# AIUseEvents, real verify_synthesis_output_sources() pass.
# ---------------------------------------------------------------------------

def test_synthesis_node_produces_verified_claim_from_real_evidence(monkeypatch, tmp_path):
    """One record, one relevant map answer, one refine batch producing one new claim -- the
    resulting SynthesisOutput must construct successfully (all self-verifying models pass) AND
    verify_synthesis_output_sources() (called internally at the end of synthesis_node) must pass
    without raising, proving the claim's own evidence_references/producing_event_id genuinely
    trace back to real, durable AIUseEvents."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        research_questions=["What designs were used?"],
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False,
                                          generate_claim_backed_results=False, refine_batch_size=5),
    )

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if "assisting a systematic literature review" in sysprompt:
            return _relevant_map_response(
                "This study used a randomized controlled trial design.",
                [{"artifact_kind": "extraction", "field_id": "study_design"}])
        if "drafting the synthesis section" in sysprompt:
            return _refine_response([{"claim_index": None, "claim_text": "Included studies used RCT designs.",
                                       "final_support_answer_ids": ["__ANY__"]}])
        raise AssertionError(f"unexpected call: {sysprompt[:80]}")

    captured: List[Dict[str, Any]] = []

    def _dispatching_with_answer_id_fill(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if "drafting the synthesis section" in sysprompt:
            # Pull the real answer_id(s) out of the rendered user prompt (new_answers_block lists
            # "<answer_id> (...)") so the refine response cites a real, allowed id.
            user_content = kw["messages"][1]["content"]
            import re as _re
            answer_ids = _re.findall(r"(answer-[0-9a-f]{64})", user_content)
            return _refine_response([{"claim_index": None, "claim_text": "Included studies used RCT designs.",
                                       "final_support_answer_ids": answer_ids}])
        return responder(kw)

    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        _dispatching_with_answer_id_fill, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    assert len(syn["map_answers"]) == 1
    assert len(syn["claims"]) == 1
    assert syn["claims"][0]["claim_text"] == "Included studies used RCT designs."
    assert syn["claims"][0]["references"], "claim must carry at least one evidence reference"
    assert syn["target_coverage"] == 1.0

    # verify_synthesis_output_sources() already ran inside synthesis_node() without raising
    # (otherwise synthesis_node itself would have raised) -- re-run it explicitly here too, against
    # a freshly reconstructed SynthesisOutput, as an extra positive-path confirmation.
    review_run_id = output["review_run_id"]
    cfg = ReviewConfig.from_dict(state["config"])
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    reconstructed = SynthesisOutput.model_validate(syn)
    assert reconstructed.claims  # round-trips through model_validate cleanly


def test_synthesis_node_map_and_refine_reach_arbitrary_extraction_schema_fields(monkeypatch, tmp_path):
    """Codex code-review round 2 (2026-08-26) finding 2: every fixture in this suite used only
    'study_design' -- restores the old free-text implementation's own regression proving the map/
    refine pipeline is schema-generic, not hardcoded to a fixed set of legacy field names. Two
    ARBITRARY extraction-schema fields (never referenced anywhere else in this test module) must
    both reach the map call's own rendered context, and the resulting claim must carry a real
    supporting selector for each."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping",
        schema_field_ids=["intervention_effect", "sample_size"],
        field_values={"rec-a": {"intervention_effect": "Reduced risk by 40 percent", "sample_size": "123"}},
        research_questions=["What was the intervention effect?"],
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False, refine_batch_size=5),
    )

    import re as _re

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if "assisting a systematic literature review" in sysprompt:
            return _relevant_map_response(
                "The intervention reduced risk by 40 percent across a sample of 123 participants.",
                [{"artifact_kind": "extraction", "field_id": "intervention_effect"},
                 {"artifact_kind": "extraction", "field_id": "sample_size"}])
        if "drafting the synthesis section" in sysprompt:
            user_content = kw["messages"][1]["content"]
            new_ids = _re.findall(r"(answer-[0-9a-f]{64})",
                                    user_content.split("New batch of per-study answers:", 1)[1])
            return _refine_response([{"claim_index": None,
                                       "claim_text": "The intervention reduced risk by 40 percent.",
                                       "final_support_answer_ids": new_ids}])
        raise AssertionError(f"unexpected call: {sysprompt[:80]}")

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(responder, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    prompt_text = _all_prompt_text(captured)
    assert "intervention_effect: Reduced risk by 40 percent" in prompt_text
    assert "sample_size: 123" in prompt_text

    assert len(syn["claims"]) == 1
    field_ids = {ref["field_id"] for ref in syn["claims"][0]["references"]}
    assert field_ids == {"intervention_effect", "sample_size"}


def test_synthesis_node_legacy_imrad_reaches_arbitrary_extraction_schema_fields(monkeypatch, tmp_path):
    """Mirrors test_synthesis_node_map_and_refine_reach_arbitrary_extraction_schema_fields for the
    legacy free-text IMRAD narrative path -- its own evidence_context must also be schema-generic,
    not limited to a fixed set of legacy field names, and must never render the old hardcoded-fields
    fallback's own garbled '- Unknown:' placeholder line."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping",
        schema_field_ids=["intervention_effect", "sample_size"],
        field_values={"rec-a": {"intervention_effect": "Reduced risk by 40 percent", "sample_size": "123"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False,
                                          generate_claim_backed_results=False,
                                          generate_legacy_imrad_narrative=True),
    )

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if sysprompt.startswith("You are writing the"):
            return "A full narrative section with enough words to be a real section."
        return _not_relevant_map_response()

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(responder, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    prompt_text = _all_prompt_text(captured)
    assert "intervention_effect: Reduced risk by 40 percent" in prompt_text
    assert "sample_size: 123" in prompt_text
    assert "- Unknown:" not in prompt_text
    assert set(syn["imrad_narrative_sections"]) == {"introduction", "methods", "discussion"}


def test_synthesis_node_ignores_foreign_charted_data_left_in_raw_state(monkeypatch, tmp_path):
    """The verified-artifact boundary must never regress: synthesis_node() reads the extraction
    corpus via the durable, run-bound resolution artifact (load_verified_extraction_resolution_
    artifact()), never raw state["charted_data"] directly -- so even a directly-polluted
    state["charted_data"] entry (a stale leftover from a bug elsewhere, or a copy-paste error) must
    never reach a prompt. Kept as a regression guard even though the current implementation
    structurally cannot read state["charted_data"] at all -- this test is what would catch a FUTURE
    change that reintroduces such a read."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False),
    )
    state["charted_data"] = state["charted_data"] + [{
        "record_id": "rec-foreign", "raw_extraction": {"study_design": "FOREIGN_MARKER_VALUE"},
    }]

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    synthesis_node(state)
    assert "FOREIGN_MARKER_VALUE" not in _all_prompt_text(captured)


def test_synthesis_node_refine_batches_across_multiple_batches(monkeypatch, tmp_path):
    """refine_batch_size=1 with two relevant records forces two separate refine batches for one
    target -- ClaimConstructionCoverage/refine_batch_outcomes must reflect exactly two attempts,
    both independently source-verified."""
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}, "rec-b": {"study_design": "Cohort"}},
        research_questions=["What designs were used?"],
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False, refine_batch_size=1),
    )

    import re as _re

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if "assisting a systematic literature review" in sysprompt:
            return _relevant_map_response(
                "This study reports a design.", [{"artifact_kind": "extraction", "field_id": "study_design"}])
        if "drafting the synthesis section" in sysprompt:
            user_content = kw["messages"][1]["content"]
            # Only the NEW batch's own answer ids are allowed for a claim_index=None (new claim)
            # update -- an id shown only in "Current claim candidates" (an EARLIER batch's own
            # already-committed claim) is not part of THIS batch's own allowed set.
            new_batch_text = user_content.split("New batch of per-study answers:", 1)[1]
            new_ids = _re.findall(r"(answer-[0-9a-f]{64})", new_batch_text)
            return _refine_response([{"claim_index": None, "claim_text": f"Claim for {new_ids[0]}",
                                       "final_support_answer_ids": new_ids}])
        raise AssertionError("unexpected call")

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(responder, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    assert len(syn["map_answers"]) == 2
    assert len(syn["claims"]) == 2
    coverage = syn["claim_construction_coverage"][0]
    assert coverage["refine_batches_total"] == 2
    assert coverage["refine_batches_applied"] == 2
    assert len(syn["refine_batch_outcomes"]) == 2
    assert {rbo["batch_index"] for rbo in syn["refine_batch_outcomes"]} == {0, 1}


def test_synthesis_node_zero_record_construction(monkeypatch):
    """A zero-record run must construct successfully and never call OpenAI at all."""
    cfg = ReviewConfig(title="t", review_type="scoping", research_questions=["RQ1"],
                        synthesis=SynthesisConfig(run_bertopic=False))
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = []
    state["review_run_id"] = "test-review-run"

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)
    assert captured == []
    assert output["synthesis"]["synthesis_record_ids"] == []
    assert output["synthesis"]["refine_batch_size"] == cfg.synthesis.refine_batch_size


# ---------------------------------------------------------------------------
# Legacy IMRAD narrative (off by default) -- success / no-call / disabled.
# ---------------------------------------------------------------------------

def test_synthesis_node_imrad_narrative_success_records_section_outcomes(monkeypatch, tmp_path):
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False,
                                          generate_claim_backed_results=False,
                                          generate_legacy_imrad_narrative=True),
    )

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if sysprompt.startswith("You are writing the"):
            return "A full narrative section with enough words to be a real section."
        return _not_relevant_map_response()

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(responder, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    assert set(syn["imrad_narrative_sections"]) == {"introduction", "methods", "discussion"}
    assert len(syn["imrad_section_outcomes"]) == 3
    assert all(r["outcome"] == "success" for r in syn["imrad_section_outcomes"])
    assert all(r["producing_event_id"] is not None for r in syn["imrad_section_outcomes"])


def test_synthesis_node_imrad_narrative_no_call_for_empty_corpus(monkeypatch):
    cfg = ReviewConfig(title="t", review_type="scoping",
                        synthesis=SynthesisConfig(run_bertopic=False,
                                                   generate_legacy_imrad_narrative=True))
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = []
    state["review_run_id"] = "test-review-run"

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: (_ for _ in ()).throw(AssertionError("no call should occur")), captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    assert len(syn["imrad_section_outcomes"]) == 3
    assert all(r["outcome"] == "no_call" for r in syn["imrad_section_outcomes"])
    assert all(r["producing_event_id"] is None for r in syn["imrad_section_outcomes"])
    assert all(r["failure_detail"] == "no full_text_records available to synthesize"
               for r in syn["imrad_section_outcomes"])
    assert captured == []


def test_synthesis_node_imrad_disabled_produces_no_section_outcomes(monkeypatch, tmp_path):
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False,
                                          generate_legacy_imrad_narrative=False),
    )
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(
        lambda kw: _not_relevant_map_response(), captured))

    output = synthesis_node(state)
    syn = output["synthesis"]
    assert syn["imrad_section_outcomes"] == []
    assert syn["imrad_narrative_sections"] == {}
    assert syn["imrad_narrative_failures"] == {}


# ---------------------------------------------------------------------------
# All four generate_claim_backed_results / generate_legacy_imrad_narrative
# flag combinations.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim_backed,legacy_narrative", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_synthesis_node_imrad_flag_combinations(monkeypatch, tmp_path, claim_backed, legacy_narrative):
    state = _run_extraction_for_synthesis(
        monkeypatch, tmp_path, review_type="scoping", schema_field_ids=["study_design"],
        field_values={"rec-a": {"study_design": "RCT"}},
        research_questions=["What designs were used?"],
        synthesis_config=SynthesisConfig(run_bertopic=False, generate_evidence_map=False,
                                          generate_claim_backed_results=claim_backed,
                                          generate_legacy_imrad_narrative=legacy_narrative),
    )

    import re as _re

    def responder(kw: Dict[str, Any]) -> str:
        sysprompt = _system_prompt(kw)
        if sysprompt.startswith("You are writing the"):
            return "A full legacy narrative section with enough words."
        if "assisting a systematic literature review" in sysprompt:
            return _not_relevant_map_response()  # keep the claim-backed path simple: no claims
        raise AssertionError(f"unexpected call: {sysprompt[:80]}")

    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(responder, captured))

    output = synthesis_node(state)
    syn = output["synthesis"]

    if legacy_narrative:
        assert len(syn["imrad_section_outcomes"]) == 3
    else:
        assert syn["imrad_section_outcomes"] == []

    expected_targets = ["What designs were used?"] + (["imrad:results"] if claim_backed else [])
    assert syn["requested_targets"] == expected_targets


# ---------------------------------------------------------------------------
# verify_synthesis_output_sources() -- representative tamper-detection
# checks, exercised directly against hand-tampered SynthesisOutput.
# ---------------------------------------------------------------------------

def _minimal_verify_kwargs(**overrides) -> Dict[str, Any]:
    base = dict(
        review_run_id="run-1",
        build_map_context_fn=lambda record_id: ([], ""),
        extraction_index={}, qa_index={},
        expected_refine_batch_size=5,
        expected_requested_targets=[],
        expected_synthesis_record_ids=[],
        expected_generate_legacy_imrad_narrative=False,
        expected_imrad_records_exist=False,
        expected_imrad_any_real_evidence=False,
        expected_evidence_context="",
        expected_canonical_research_questions=[],
        expected_llm_model="gpt-4o",
    )
    base.update(overrides)
    return base


def test_verify_rejects_tampered_requested_targets(tmp_path):
    """A deleted target: SynthesisOutput construction succeeds (internally coherent, since nothing
    else references the deleted target), but verify_synthesis_output_sources() must reject the
    mismatch against the real, live expected_requested_targets."""
    output = SynthesisOutput(refine_batch_size=5, requested_targets=["RQ1"],
                              claim_construction_coverage=[{"rq_or_section": "RQ1"}],
                              rejected_targets=[{"rq_or_section": "RQ1"}], target_coverage=0.0)
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")
    with pytest.raises(RuntimeError, match="does not match"):
        verify_synthesis_output_sources(output, event_store=event_store, **_minimal_verify_kwargs(
            expected_requested_targets=["RQ1", "RQ2"]))


def test_verify_rejects_wrong_refine_batch_size(tmp_path):
    output = SynthesisOutput(refine_batch_size=10)
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")
    with pytest.raises(RuntimeError, match="refine_batch_size"):
        verify_synthesis_output_sources(output, event_store=event_store, **_minimal_verify_kwargs(
            expected_refine_batch_size=5))


def test_verify_rejects_no_call_when_corpus_has_usable_evidence(tmp_path):
    """Three internally coherent no_call rows must be REJECTED when the live corpus genuinely has
    usable evidence -- closing the v22-finding gap fixed in v23."""
    output = SynthesisOutput(
        refine_batch_size=5,
        imrad_narrative_failures={"introduction": "no full_text_records available to synthesize",
                                   "methods": "no full_text_records available to synthesize",
                                   "discussion": "no full_text_records available to synthesize"},
        imrad_section_outcomes=[
            {"section": s, "outcome": "no_call", "producing_event_id": None,
             "failure_detail": "no full_text_records available to synthesize"}
            for s in ("introduction", "methods", "discussion")
        ],
    )
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")
    with pytest.raises(RuntimeError, match="usable evidence"):
        verify_synthesis_output_sources(output, event_store=event_store, **_minimal_verify_kwargs(
            expected_generate_legacy_imrad_narrative=True,
            expected_imrad_records_exist=True, expected_imrad_any_real_evidence=True))


def test_verify_rejects_disabled_generation_with_nonempty_output(tmp_path):
    output = SynthesisOutput(
        refine_batch_size=5,
        imrad_narrative_failures={"introduction": "no full_text_records available to synthesize"},
        imrad_section_outcomes=[{"section": "introduction", "outcome": "no_call",
                                  "producing_event_id": None,
                                  "failure_detail": "no full_text_records available to synthesize"}],
    )
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")
    with pytest.raises(RuntimeError, match="disabled"):
        verify_synthesis_output_sources(output, event_store=event_store, **_minimal_verify_kwargs(
            expected_generate_legacy_imrad_narrative=False))


def test_verify_rejects_map_event_with_unrelated_call_identity(tmp_path):
    """Codex code-review finding 1 (2026-08-26): an adversarial event can carry the CORRECT
    target/context/output hash (so input_artifact_refs/output_artifact_refs alone pass) while its
    own model/prompt_parameters/prompt_template_version/model_config_id/rendered_prompt_hash are
    completely UNRELATED to what the real call site would have used. Map's own prompt is fully
    reconstructable, so verify_synthesis_output_sources() must recompute the COMPLETE call identity
    (work_item_id, in particular, which is itself a hash over all of these fields) and reject a
    mismatch -- input/output refs alone are not enough to prove WHICH model/configuration actually
    produced them."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")

    record_id, target = "rec-a", "RQ1"
    reconstructed_context = "study_design: RCT"

    tampered_event = build_ai_use_event(
        review_run_id="run-1", stage="synthesis_claim_map", work_item_id="synthesis_claim_map:unrelated",
        attempt=0, tool_kind="llm", model="unrelated-model", provider="openai", component_versions={},
        task_purpose="x", model_config_id="unrelated-config-id", prompt_template_version="v999",
        prompt_template_text="unrelated template text",
        prompt_parameters={"temperature": 0.99}, rendered_prompt_hash="0" * 64,
        # These two ARE the correct, genuinely-reconstructable refs for (target, record_id,
        # reconstructed_context) -- the adversarial probe -- everything else is unrelated.
        input_artifact_refs=[f"target:{target}", f"record:{record_id}:sha256:{sha256_hex(reconstructed_context)}"],
        output_artifact_refs=[f"sha256:{sha256_hex({'relevant': False})}"],
        call_status="success",
    )
    published = event_store.publish(tampered_event)

    outcome_record = MapOutcomeRecord(rq_or_section=target, record_id=record_id, outcome="not_relevant",
                                       producing_event_id=published.event_id)
    output = SynthesisOutput(
        refine_batch_size=5, extraction_run_id="extrun-1", synthesis_record_ids=[record_id],
        requested_targets=[target], map_outcome_records=[outcome_record],
        claim_construction_coverage=[{"rq_or_section": target}],
        rejected_targets=[{"rq_or_section": target}], target_coverage=0.0,
    )

    with pytest.raises(RuntimeError, match="work_item_id"):
        verify_synthesis_output_sources(
            output, event_store=event_store,
            **_minimal_verify_kwargs(
                build_map_context_fn=lambda rid: (
                    [SupportingFieldSelector(artifact_kind="extraction", field_id="study_design")],
                    reconstructed_context),
                expected_requested_targets=[target], expected_synthesis_record_ids=[record_id],
            ))


def test_verify_rejects_refine_event_from_stale_accumulator_state(tmp_path):
    """Codex code-review round 2 (2026-08-26) finding 1: refine's own live call hashes its rendered
    prompt -- which shows the FULL prior accumulator state -- into work_item_id/rendered_prompt_hash.
    A genuinely valid, correctly-shaped refine event, produced against a DIFFERENT prior accumulator
    state, could previously be substituted for a batch as long as target/batch answers/output hash
    all happened to match, since the verifier declined to reconstruct refine's own rendered prompt at
    all. Here batch 1's own RefineBatchOutcomeRecord truthfully claims the pre_batch_claim_states
    that batch 0's own real output actually produced (Claim A) -- so SynthesisOutput's own internal
    chain-consistency check (state.py) is satisfied -- but its own producing_event_id points at a
    REAL, validly-published event that was actually rendered assuming an EMPTY prior state, as if it
    were itself the very first batch. Old input-shape/output-hash checks alone pass this (same
    target, same batch answer, same output content); only the freshly-reconstructed work_item_id/
    rendered_prompt_hash catches it."""
    import types as _types

    # NOTE: every state.py model instance constructed below uses synthesis_module's OWN bound
    # classes (synthesis_module.EvidenceReference etc., from its own `from state import ...`) rather
    # than this test file's top-level `from slr_pipeline.state import ...` names -- the two are
    # DIFFERENT classes at runtime (this repo's dual sys.path setup imports `state` under two
    # different module identities), and pydantic model equality requires matching classes, which
    # verify_synthesis_output_sources()'s own internal reconstruction (e.g. _references_for_selectors())
    # always produces via the `state`-module identity, never `slr_pipeline.state`.
    RefineAppliedUpdate = synthesis_module.RefineAppliedUpdate
    RefineClaimSnapshot = synthesis_module.RefineClaimSnapshot
    RefineBatchOutcomeRecord = synthesis_module.RefineBatchOutcomeRecord
    build_synthesis_claim = synthesis_module.build_synthesis_claim
    EvidenceReference = synthesis_module.EvidenceReference
    MapOutcomeRecord = synthesis_module.MapOutcomeRecord
    SynthesisOutput = synthesis_module.SynthesisOutput
    build_map_answer = synthesis_module.build_map_answer
    SupportingFieldSelector = synthesis_module.SupportingFieldSelector

    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")
    target = "RQ1"
    selectors = [SupportingFieldSelector(artifact_kind="extraction", field_id="study_design")]

    extraction_index = {
        "rec-a": {"study_design": [_types.SimpleNamespace(evidence_id="evidence-" + "1" * 64)]},
        "rec-b": {"study_design": [_types.SimpleNamespace(evidence_id="evidence-" + "2" * 64)]},
    }
    ref1 = EvidenceReference(artifact_kind="extraction", artifact_run_id="extrun-1", record_id="rec-a",
                              field_id="study_design", evidence_id="evidence-" + "1" * 64)
    ref2 = EvidenceReference(artifact_kind="extraction", artifact_run_id="extrun-1", record_id="rec-b",
                              field_id="study_design", evidence_id="evidence-" + "2" * 64)

    def _publish_valid_map_event(record_id, rendered_context, answer_text):
        work_item_id, _messages, rendered_prompt_hash, prompt_parameters = synthesis_module._build_map_work_item(
            target=target, record_id=record_id, rendered_context=rendered_context, model="gpt-4o")
        event = build_ai_use_event(
            review_run_id="run-1", stage="synthesis_claim_map", work_item_id=work_item_id, attempt=0,
            tool_kind="llm", model="gpt-4o", provider="openai", component_versions={"gpt-4o": "x"},
            access_info="https://api.openai.com/v1", task_purpose="x",
            model_config_id=synthesis_module._SYNTHESIS_MAP_MODEL_CONFIG_ID,
            prompt_template_version=synthesis_module._SYNTHESIS_MAP_PROMPT_VERSION,
            prompt_template_text="template", prompt_parameters=prompt_parameters,
            rendered_prompt_hash=rendered_prompt_hash,
            input_artifact_refs=[f"target:{target}", f"record:{record_id}:sha256:{sha256_hex(rendered_context)}"],
            output_artifact_refs=[f"sha256:{sha256_hex(synthesis_module._canonical_map_response_payload(answer_text=answer_text, selectors=selectors))}"],
            call_status="success",
        )
        return event_store.publish(event)

    map_event_a = _publish_valid_map_event("rec-a", "study_design: RCT", "Study A used an RCT design.")
    map_event_b = _publish_valid_map_event("rec-b", "study_design: Cohort", "Study B used a cohort design.")

    answer1 = build_map_answer(rq_or_section=target, record_id="rec-a", answer_text="Study A used an RCT design.",
                                selectors=selectors, evidence_references=[ref1], producing_event_id=map_event_a.event_id)
    answer2 = build_map_answer(rq_or_section=target, record_id="rec-b", answer_text="Study B used a cohort design.",
                                selectors=selectors, evidence_references=[ref2], producing_event_id=map_event_b.event_id)
    known_answers_by_id = {answer1.answer_id: answer1, answer2.answer_id: answer2}

    def _publish_refine_batch(*, batch_index, batch, pre_batch_claim_states, applied_updates):
        work_item_id, _messages, rendered_prompt_hash, prompt_parameters = synthesis_module._build_refine_work_item(
            target=target, batch_index=batch_index, batch=batch,
            pre_batch_claim_states=[(s.claim_text, s.supporting_answer_ids) for s in pre_batch_claim_states],
            answer_lookup=known_answers_by_id, model="gpt-4o")
        output_refs = (
            [f"response:sha256:{sha256_hex(synthesis_module._canonical_refine_response_payload(applied_updates))}"]
            + [f"update:sha256:{sha256_hex(synthesis_module._canonical_refine_update_payload(u.claim_text, u.final_support_answer_ids))}"
               for u in applied_updates])
        event = build_ai_use_event(
            review_run_id="run-1", stage="synthesis_claim_refine", work_item_id=work_item_id, attempt=0,
            tool_kind="llm", model="gpt-4o", provider="openai", component_versions={"gpt-4o": "x"},
            access_info="https://api.openai.com/v1", task_purpose="x",
            model_config_id=synthesis_module._SYNTHESIS_REFINE_MODEL_CONFIG_ID,
            prompt_template_version=synthesis_module._SYNTHESIS_REFINE_PROMPT_VERSION,
            prompt_template_text="template", prompt_parameters=prompt_parameters,
            rendered_prompt_hash=rendered_prompt_hash,
            input_artifact_refs=(
                [f"target:{target}", f"batch:{batch_index}"] + [f"answer:{a.answer_id}" for a in batch]
                + [f"claim:{i}:sha256:{sha256_hex({'claim_text': s.claim_text, 'supporting_answer_ids': s.supporting_answer_ids})}"
                   for i, s in enumerate(pre_batch_claim_states)]),
            output_artifact_refs=output_refs, call_status="success", failure_detail=None,
        )
        return event_store.publish(event)

    # Batch 0: the target's REAL first batch -- genuinely empty prior state, applies one new claim.
    batch0_update = RefineAppliedUpdate(claim_index=None, claim_text="Claim A",
                                         final_support_answer_ids=[answer1.answer_id])
    event0 = _publish_refine_batch(batch_index=0, batch=[answer1], pre_batch_claim_states=[],
                                    applied_updates=[batch0_update])
    rbo0 = RefineBatchOutcomeRecord(rq_or_section=target, batch_index=0, batch_answer_ids=[answer1.answer_id],
                                     outcome="applied", is_no_op=False, producing_event_id=event0.event_id,
                                     pre_batch_claim_states=[], applied_updates=[batch0_update])

    # A REAL, validly-published event for batch 1's own (target, batch_answer_ids, output content) --
    # but rendered assuming an EMPTY prior accumulator state, as if it were itself batch 0. Valid on
    # its own; wrong for batch 1's own REAL prior state (which genuinely already has Claim A in it).
    batch1_update = RefineAppliedUpdate(claim_index=None, claim_text="Claim B",
                                         final_support_answer_ids=[answer2.answer_id])
    stale_event = _publish_refine_batch(batch_index=1, batch=[answer2], pre_batch_claim_states=[],
                                         applied_updates=[batch1_update])

    # The tampered RefineBatchOutcomeRecord: CLAIMS the real, chain-consistent pre_batch_claim_states
    # (satisfying SynthesisOutput's own internal chain check), but its own producing_event_id is the
    # STALE event above, produced against a DIFFERENT (empty) prior state.
    tampered_pre_state = [RefineClaimSnapshot(claim_text="Claim A", supporting_answer_ids=[answer1.answer_id],
                                               producing_event_id=event0.event_id)]
    rbo1 = RefineBatchOutcomeRecord(rq_or_section=target, batch_index=1, batch_answer_ids=[answer2.answer_id],
                                     outcome="applied", is_no_op=False, producing_event_id=stale_event.event_id,
                                     pre_batch_claim_states=tampered_pre_state, applied_updates=[batch1_update])

    claim_a = build_synthesis_claim(claim_text="Claim A", contributing_answers=[answer1], rq_or_section=target,
                                     producing_event_id=event0.event_id)
    claim_b = build_synthesis_claim(claim_text="Claim B", contributing_answers=[answer2], rq_or_section=target,
                                     producing_event_id=stale_event.event_id)

    output = SynthesisOutput(
        refine_batch_size=1, extraction_run_id="extrun-1", synthesis_record_ids=["rec-a", "rec-b"],
        requested_targets=[target],
        map_outcome_records=[
            MapOutcomeRecord(rq_or_section=target, record_id="rec-a", outcome="relevant",
                              producing_event_id=map_event_a.event_id),
            MapOutcomeRecord(rq_or_section=target, record_id="rec-b", outcome="relevant",
                              producing_event_id=map_event_b.event_id),
        ],
        map_answers=[answer1, answer2], claims=[claim_a, claim_b],
        rq_synthesis={target: "Claim A\n\nClaim B"},
        refine_batch_outcomes=[rbo0, rbo1],
        claim_construction_coverage=[{
            "rq_or_section": target, "refine_batches_total": 2, "refine_batches_applied": 2,
            "refine_batches_applied_no_op": 0, "refine_batches_call_failed": 0,
            "refine_batches_invalid_response": 0, "unincorporated_answer_ids": [],
        }],
        target_coverage=1.0,
    )

    def _build_map_context_fn(rid: str):
        return (selectors, "study_design: RCT" if rid == "rec-a" else "study_design: Cohort")

    with pytest.raises(RuntimeError, match="work_item_id"):
        verify_synthesis_output_sources(
            output, event_store=event_store,
            **_minimal_verify_kwargs(
                build_map_context_fn=_build_map_context_fn, extraction_index=extraction_index,
                expected_requested_targets=[target], expected_synthesis_record_ids=["rec-a", "rec-b"],
                expected_refine_batch_size=1,
            ))


def test_refine_batch_outcome_rejects_duplicate_claim_index_in_applied_updates():
    """Codex code-review round 3 (2026-08-26) finding 1: RefineResponse -- the live parser --
    forbids two updates in one response targeting the SAME existing claim_index
    (nodes/synthesis.py's RefineResponse._check_internal_consistency). RefineBatchOutcomeRecord.
    applied_updates had no equivalent constraint, so an outcome record containing two claim_index=0
    updates constructed successfully -- a fabricated, self-consistent event could represent a
    response production would have classified as invalid_response. Must now be rejected at
    construction, before any event/replay machinery is even involved."""
    from pydantic import ValidationError
    RefineAppliedUpdate = synthesis_module.RefineAppliedUpdate
    RefineClaimSnapshot = synthesis_module.RefineClaimSnapshot
    RefineBatchOutcomeRecord = synthesis_module.RefineBatchOutcomeRecord

    with pytest.raises(ValidationError, match="same existing claim_index"):
        RefineBatchOutcomeRecord(
            rq_or_section="RQ1", batch_index=1, batch_answer_ids=["answer-" + "a" * 64],
            outcome="applied", is_no_op=False, producing_event_id="aiuse-" + "a" * 48,
            pre_batch_claim_states=[RefineClaimSnapshot(
                claim_text="Claim A", supporting_answer_ids=["answer-" + "b" * 64],
                producing_event_id="aiuse-" + "b" * 48)],
            applied_updates=[
                RefineAppliedUpdate(claim_index=0, claim_text="Claim A v2",
                                     final_support_answer_ids=["answer-" + "a" * 64]),
                RefineAppliedUpdate(claim_index=0, claim_text="Claim A v3",
                                     final_support_answer_ids=["answer-" + "a" * 64]),
            ],
        )


def test_verify_rejects_claim_reassigned_to_stale_earlier_event(tmp_path):
    """Codex code-review round 3 (2026-08-26) finding 2: a final claim was bound to its producing
    event by CONTENT membership and "any applied event for this target" -- never by WHICH batch
    actually last touched that accumulator position. Batch 1 here genuinely re-touches position 0
    (claim_index=0), restating the EXACT SAME text/answer ids batch 0 originally set -- a
    legitimate, content-unchanged refine call (the model simply didn't add new evidence this round).
    The real, honest producing_event_id for the final claim is batch 1's own event (the live
    _commit_simulated_state() always overwrites producing_event_id on any touch, even a
    content-repeating one). The tampered output instead claims the STALE, earlier batch 0 event --
    content-identical, so undetectable by any content-only check; only comparing producing_event_id
    against the replayed accumulator chain (SynthesisOutput's own topology validator) catches it."""
    from pydantic import ValidationError
    RefineAppliedUpdate = synthesis_module.RefineAppliedUpdate
    RefineClaimSnapshot = synthesis_module.RefineClaimSnapshot
    RefineBatchOutcomeRecord = synthesis_module.RefineBatchOutcomeRecord
    build_synthesis_claim = synthesis_module.build_synthesis_claim
    EvidenceReference = synthesis_module.EvidenceReference
    MapOutcomeRecord = synthesis_module.MapOutcomeRecord
    SynthesisOutput = synthesis_module.SynthesisOutput
    build_map_answer = synthesis_module.build_map_answer
    SupportingFieldSelector = synthesis_module.SupportingFieldSelector

    import types as _types

    cfg = ReviewConfig(title="t", output_dir=str(tmp_path))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1")
    target = "RQ1"
    selectors = [SupportingFieldSelector(artifact_kind="extraction", field_id="study_design")]

    extraction_index = {
        "rec-a": {"study_design": [_types.SimpleNamespace(evidence_id="evidence-" + "1" * 64)]},
        "rec-b": {"study_design": [_types.SimpleNamespace(evidence_id="evidence-" + "2" * 64)]},
    }
    ref1 = EvidenceReference(artifact_kind="extraction", artifact_run_id="extrun-1", record_id="rec-a",
                              field_id="study_design", evidence_id="evidence-" + "1" * 64)
    ref2 = EvidenceReference(artifact_kind="extraction", artifact_run_id="extrun-1", record_id="rec-b",
                              field_id="study_design", evidence_id="evidence-" + "2" * 64)

    def _publish_valid_map_event(record_id, rendered_context, answer_text):
        work_item_id, _messages, rendered_prompt_hash, prompt_parameters = synthesis_module._build_map_work_item(
            target=target, record_id=record_id, rendered_context=rendered_context, model="gpt-4o")
        event = build_ai_use_event(
            review_run_id="run-1", stage="synthesis_claim_map", work_item_id=work_item_id, attempt=0,
            tool_kind="llm", model="gpt-4o", provider="openai", component_versions={"gpt-4o": "x"},
            access_info="https://api.openai.com/v1", task_purpose="x",
            model_config_id=synthesis_module._SYNTHESIS_MAP_MODEL_CONFIG_ID,
            prompt_template_version=synthesis_module._SYNTHESIS_MAP_PROMPT_VERSION,
            prompt_template_text="template", prompt_parameters=prompt_parameters,
            rendered_prompt_hash=rendered_prompt_hash,
            input_artifact_refs=[f"target:{target}", f"record:{record_id}:sha256:{sha256_hex(rendered_context)}"],
            output_artifact_refs=[f"sha256:{sha256_hex(synthesis_module._canonical_map_response_payload(answer_text=answer_text, selectors=selectors))}"],
            call_status="success",
        )
        return event_store.publish(event)

    map_event_a = _publish_valid_map_event("rec-a", "study_design: RCT", "Study A used an RCT design.")
    map_event_b = _publish_valid_map_event("rec-b", "study_design: Cohort", "Study B used a cohort design.")

    answer1 = build_map_answer(rq_or_section=target, record_id="rec-a", answer_text="Study A used an RCT design.",
                                selectors=selectors, evidence_references=[ref1], producing_event_id=map_event_a.event_id)
    answer2 = build_map_answer(rq_or_section=target, record_id="rec-b", answer_text="Study B used a cohort design.",
                                selectors=selectors, evidence_references=[ref2], producing_event_id=map_event_b.event_id)
    known_answers_by_id = {answer1.answer_id: answer1, answer2.answer_id: answer2}

    def _publish_refine_batch(*, batch_index, batch, pre_batch_claim_states, applied_updates):
        work_item_id, _messages, rendered_prompt_hash, prompt_parameters = synthesis_module._build_refine_work_item(
            target=target, batch_index=batch_index, batch=batch,
            pre_batch_claim_states=[(s.claim_text, s.supporting_answer_ids) for s in pre_batch_claim_states],
            answer_lookup=known_answers_by_id, model="gpt-4o")
        output_refs = (
            [f"response:sha256:{sha256_hex(synthesis_module._canonical_refine_response_payload(applied_updates))}"]
            + [f"update:sha256:{sha256_hex(synthesis_module._canonical_refine_update_payload(u.claim_text, u.final_support_answer_ids))}"
               for u in applied_updates])
        event = build_ai_use_event(
            review_run_id="run-1", stage="synthesis_claim_refine", work_item_id=work_item_id, attempt=0,
            tool_kind="llm", model="gpt-4o", provider="openai", component_versions={"gpt-4o": "x"},
            access_info="https://api.openai.com/v1", task_purpose="x",
            model_config_id=synthesis_module._SYNTHESIS_REFINE_MODEL_CONFIG_ID,
            prompt_template_version=synthesis_module._SYNTHESIS_REFINE_PROMPT_VERSION,
            prompt_template_text="template", prompt_parameters=prompt_parameters,
            rendered_prompt_hash=rendered_prompt_hash,
            input_artifact_refs=(
                [f"target:{target}", f"batch:{batch_index}"] + [f"answer:{a.answer_id}" for a in batch]
                + [f"claim:{i}:sha256:{sha256_hex({'claim_text': s.claim_text, 'supporting_answer_ids': s.supporting_answer_ids})}"
                   for i, s in enumerate(pre_batch_claim_states)]),
            output_artifact_refs=output_refs, call_status="success", failure_detail=None,
        )
        return event_store.publish(event)

    # Batch 0: creates position 0 -- "Claim X" supported by answer1 only.
    batch0_update = RefineAppliedUpdate(claim_index=None, claim_text="Claim X",
                                         final_support_answer_ids=[answer1.answer_id])
    event0 = _publish_refine_batch(batch_index=0, batch=[answer1], pre_batch_claim_states=[],
                                    applied_updates=[batch0_update])
    rbo0 = RefineBatchOutcomeRecord(rq_or_section=target, batch_index=0, batch_answer_ids=[answer1.answer_id],
                                     outcome="applied", is_no_op=False, producing_event_id=event0.event_id,
                                     pre_batch_claim_states=[], applied_updates=[batch0_update])

    # Batch 1: genuinely RE-TOUCHES position 0 (claim_index=0), restating the EXACT SAME text/ids
    # batch 0 already set -- a legitimate, content-unchanged refine call that does not even cite
    # its own new answer2. The REAL post-batch producing_event_id for position 0 is event1's own id.
    pre1 = [RefineClaimSnapshot(claim_text="Claim X", supporting_answer_ids=[answer1.answer_id],
                                 producing_event_id=event0.event_id)]
    batch1_update = RefineAppliedUpdate(claim_index=0, claim_text="Claim X",
                                         final_support_answer_ids=[answer1.answer_id])
    event1 = _publish_refine_batch(batch_index=1, batch=[answer2], pre_batch_claim_states=pre1,
                                    applied_updates=[batch1_update])
    rbo1 = RefineBatchOutcomeRecord(rq_or_section=target, batch_index=1, batch_answer_ids=[answer2.answer_id],
                                     outcome="applied", is_no_op=False, producing_event_id=event1.event_id,
                                     pre_batch_claim_states=pre1, applied_updates=[batch1_update])

    # The tampered claim: correct content ("Claim X", [answer1] -- unchanged from batch 0's own
    # original content, since batch 1's update restated it verbatim), but producing_event_id
    # reassigned to the STALE, earlier batch 0 event instead of the REAL, actually-last-touching
    # batch 1 event.
    claim_x = build_synthesis_claim(claim_text="Claim X", contributing_answers=[answer1], rq_or_section=target,
                                     producing_event_id=event0.event_id)

    with pytest.raises(ValidationError, match="does not match this target's own persisted"):
        SynthesisOutput(
            refine_batch_size=1, extraction_run_id="extrun-1", synthesis_record_ids=["rec-a", "rec-b"],
            requested_targets=[target],
            map_outcome_records=[
                MapOutcomeRecord(rq_or_section=target, record_id="rec-a", outcome="relevant",
                                  producing_event_id=map_event_a.event_id),
                MapOutcomeRecord(rq_or_section=target, record_id="rec-b", outcome="relevant",
                                  producing_event_id=map_event_b.event_id),
            ],
            map_answers=[answer1, answer2], claims=[claim_x],
            rq_synthesis={target: "Claim X"},
            refine_batch_outcomes=[rbo0, rbo1],
            claim_construction_coverage=[{
                "rq_or_section": target, "refine_batches_total": 2, "refine_batches_applied": 2,
                "refine_batches_applied_no_op": 0, "refine_batches_call_failed": 0,
                "refine_batches_invalid_response": 0, "unincorporated_answer_ids": [answer2.answer_id],
            }],
            target_coverage=1.0,
        )
