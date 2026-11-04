"""Offline tests for Stage 1's rebuilt quality-appraisal node
(nodes/quality_appraisal.py): the roll-up algorithms, the model provenance
contracts, quality_appraisal_node itself, and the resolve/confirm resume
paths -- per Section 1.11 of quality_appraisal_synthesis_plan_2026-08-22.md.

Two boundaries are monkeypatched, matching tests/nodes/test_extraction_node.py's
own convention exactly:
  - `nodes.quality_appraisal.parse_pdf` -- replaced with a fixed, hand-built
    ParsedDocument (no real Docling parse of a real PDF).
  - `nodes.quality_appraisal.run_openai_teaching_batch` /
    `run_openai_rob2_domain_batch` -- the two provider call shapes this
    node wires up; replaced with scripted fakes so no real OpenAI
    credential or network call is ever made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from slr_pipeline.config import ReviewConfig, QualityAppraisalConfig
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.state import Record, initial_state

import slr_pipeline.nodes.quality_appraisal as qa_module
from slr_pipeline.nodes.quality_appraisal import (
    AppraisalEvidence, AppraisalItemResponse, DomainJudgment, HumanConfirmation,
    QAAppraisalCheckpoint, QACheckpointStore, RawAppraisalEvidenceItem,
    RawDomainBatchResponse, RawSignallingQuestionBatchResponse, RawTeachingBatchResponse,
    RawTeachingItemBatchResponse, RoB2InspiredAppraisal, SignallingQuestionResponse,
    TEACHING_APPRAISAL_ITEMS, TeachingAppraisal, compute_qa_run_id,
    confirm_quality_appraisal_provisional_item, load_verified_quality_appraisal_resolution_artifact,
    quality_appraisal_node, resolve_quality_appraisal_review_queue, roll_up_domain_risk, roll_up_overall_risk,
)

import dataclasses


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _chunk(index: int, text: str, section: str = "Methods") -> ParsedChunk:
    return ParsedChunk(chunk_id=f"doc-chunk-{index:04d}", order_index=index, section=section, page_min=index + 1, page_max=index + 1, text=text)


def _parsed_document(chunks: tuple) -> ParsedDocument:
    return ParsedDocument(document_sha256="a" * 64, parser_name="docling", parser_version="2.61.2", parser_options=DoclingParserOptions(), chunks=chunks)


def _model_config(**overrides) -> OpenAIExtractionConfig:
    defaults = dict(model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra")
    defaults.update(overrides)
    return OpenAIExtractionConfig(**defaults)


def _record(id: str, pdf_path) -> Record:
    return Record(id=id, source="test", source_id=id, title=f"Study {id}", full_text_path=str(pdf_path), full_text_status="parsed")


def _cfg(tmp_path, *, checklist_id="teaching_appraisal_v1", policy="require_human_confirmation", model_config=None) -> ReviewConfig:
    kwargs: Dict[str, Any] = dict(
        title="t", review_type="rapid", output_dir=str(tmp_path / "out"),
        quality_appraisal=QualityAppraisalConfig(checklist_id=checklist_id, negative_high_risk_policy=policy),
    )
    kwargs["quality_appraisal_model_config"] = dataclasses.asdict(model_config or _model_config())
    return ReviewConfig(**kwargs)


def _state(cfg: ReviewConfig, records) -> Dict[str, Any]:
    state = initial_state(cfg.to_dict())
    state["full_text_records"] = [r.to_dict() for r in records]
    return state


def _all_items_response(outcome: str, chunk_id: str, quotation: str) -> RawTeachingBatchResponse:
    return RawTeachingBatchResponse(items={
        i["item_id"]: RawTeachingItemBatchResponse(
            outcome=outcome, rationale="stated explicitly",
            evidence=[RawAppraisalEvidenceItem(chunk_id=chunk_id, quotation=quotation)] if outcome in ("yes", "no") else [],
        )
        for i in TEACHING_APPRAISAL_ITEMS
    })


def _all_not_found_response() -> RawTeachingBatchResponse:
    return RawTeachingBatchResponse(items={i["item_id"]: RawTeachingItemBatchResponse(outcome="not_found") for i in TEACHING_APPRAISAL_ITEMS})


# ---------------------------------------------------------------------------
# Requirement: no title/abstract fallback, no include-on-error.
# ---------------------------------------------------------------------------

def test_no_records_is_a_clean_noop(tmp_path):
    cfg = _cfg(tmp_path)
    state = _state(cfg, [])
    output = quality_appraisal_node(state)
    assert output["quality_appraisal_results"] == []
    assert output["quality_appraisal_review_queue"] == []
    assert output["requires_human_review"] is False
    assert output["quality_appraisal_run_id"] is None
    assert "full_text_records" not in output


def test_rejects_record_with_no_confirmed_pdf_path(tmp_path):
    cfg = _cfg(tmp_path)
    bad_record = Record(id="rec-a", source="test", source_id="rec-a", title="No PDF")
    state = _state(cfg, [bad_record])
    with pytest.raises(ValueError, match="no confirmed, existing full-text PDF"):
        quality_appraisal_node(state)


def test_no_config_raises(tmp_path):
    cfg = ReviewConfig(title="t", review_type="rapid", output_dir=str(tmp_path / "out"))
    pdf_path = tmp_path / "rec.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    with pytest.raises(ValueError, match="quality_appraisal_model_config"):
        quality_appraisal_node(state)


# ---------------------------------------------------------------------------
# Teaching rubric: verified yes/no is finalized, never queued.
# ---------------------------------------------------------------------------

def test_verified_yes_finalizes_with_verifier_derived_locator(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("yes", chunk.chunk_id, "The research question is clearly stated"))

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    record = _record("rec-a", pdf_path)
    state = _state(cfg, [record])

    output = quality_appraisal_node(state)
    assert output["quality_appraisal_review_queue"] == []
    assert output["requires_human_review"] is False
    doc = output["quality_appraisal_results"][0]
    item = doc["teaching"]["items"][0]
    assert item["status"] == "yes"
    assert item["decision_origin"] == "ai"
    assert item["evidence_status"] == "verified_from_chunk"
    assert item["evidence"][0]["locator"] == chunk.page_range
    assert item["evidence"][0]["chunk_id"] == chunk.chunk_id
    assert item["provisional_pending_human_confirmation"] is False


def test_genuine_absence_of_evidence_is_queued_as_unclear(tmp_path, monkeypatch):
    chunk = _chunk(0, "Some unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    assert output["requires_human_review"] is True
    queue = output["quality_appraisal_review_queue"]
    assert len(queue) == len(TEACHING_APPRAISAL_ITEMS)
    for entry in queue:
        assert entry["queue_kind"] == "resolution"
        assert entry["reason"] == "no_evidence_found"
    doc = output["quality_appraisal_results"][0]
    for item in doc["teaching"]["items"]:
        assert item["status"] == "unclear"
        assert item["evidence"] == []
        assert item["decision_origin"] == "ai"
        assert item["evidence_status"] == "not_provided"


def test_evidence_verification_failure_is_queued_with_distinct_reason(tmp_path, monkeypatch):
    chunk = _chunk(0, "The actual chunk text about methodology.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("yes", chunk.chunk_id, "text that does not appear verbatim anywhere"))

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    assert output["requires_human_review"] is True
    for entry in output["quality_appraisal_review_queue"]:
        assert entry["reason"] == "evidence_verification_failed"


def test_batch_call_failure_is_queued_with_distinct_reason(tmp_path, monkeypatch):
    chunk = _chunk(0, "Some text.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def _raise(**kw):
        raise qa_module.QACallError("boom")
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _raise)

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    assert output["requires_human_review"] is True
    for entry in output["quality_appraisal_review_queue"]:
        assert entry["reason"] == "batch_call_failed"


def test_conflicting_evidence_across_batches_is_never_auto_resolved(tmp_path, monkeypatch):
    chunk_a = _chunk(0, "A" * 5000)  # forces two separate batches
    chunk_b = _chunk(1, "B" * 5000)
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))
    monkeypatch.setattr(qa_module, "extraction_max_batch_chars_override", None, raising=False)

    def _fake(*, record_title, batch_chunks, config):
        c = batch_chunks[0]
        if c.chunk_id == chunk_a.chunk_id:
            return _all_items_response("yes", chunk_a.chunk_id, chunk_a.text)
        return _all_items_response("no", chunk_b.chunk_id, chunk_b.text)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    cfg.quality_appraisal_max_batch_chars = 100  # forces two batches given the two 5000-char chunks
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    reasons = {e["reason"] for e in output["quality_appraisal_review_queue"]}
    assert "conflicting_evidence_across_batches" in reasons
    conflicting_entry = next(e for e in output["quality_appraisal_review_queue"] if e["reason"] == "conflicting_evidence_across_batches")
    assert set(conflicting_entry["conflicting_evidence"].keys()) == {"yes", "no"}


# ---------------------------------------------------------------------------
# Negative/high-risk provisional policy (Section 1.6, 1.10).
# ---------------------------------------------------------------------------

def test_negative_finding_is_provisional_and_blocks_under_require_confirmation(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    assert output["requires_human_review"] is True
    assert output["quality_appraisal_provisional_ledger"] == []
    confirmation_entries = [e for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "confirmation"]
    assert len(confirmation_entries) == len(TEACHING_APPRAISAL_ITEMS)
    for doc in output["quality_appraisal_results"]:
        for item in doc["teaching"]["items"]:
            assert item["provisional_pending_human_confirmation"] is True
            assert item["status"] == "no"
            assert item["decision_origin"] == "ai"
            assert item["evidence_status"] == "verified_from_chunk"


def test_negative_finding_is_non_blocking_under_allow_provisional_ai_only(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))

    cfg = _cfg(tmp_path, policy="allow_provisional_ai_only")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    assert output["requires_human_review"] is False
    assert output["quality_appraisal_review_queue"] == []
    assert len(output["quality_appraisal_provisional_ledger"]) == len(TEACHING_APPRAISAL_ITEMS)


# ---------------------------------------------------------------------------
# Roll-up algorithms -- pure, deterministic, no LLM/mocking needed.
# ---------------------------------------------------------------------------

def _sq(question_id, answer, evidence=None, decision_origin="ai"):
    return SignallingQuestionResponse(
        question_id=question_id, answer=answer, evidence=evidence or [], rationale="r",
        decision_origin=decision_origin,
        evidence_status=("human_attested" if decision_origin == "human" else ("verified_from_chunk" if answer != "NI" else "not_provided")),
        # A machine N/PN must be provisional (Section 1.6's own machine-
        # negative-implies-provisional invariant, enforced structurally
        # since the round-4 correction pass).
        provisional_pending_human_confirmation=(decision_origin == "ai" and answer in ("N", "PN")),
    )


def _evidence():
    return [AppraisalEvidence(chunk_id="c1", quotation="q", locator="p1")]


@pytest.mark.parametrize("answers,expected_risk,expected_basis", [
    (["Y", "Y"], "low", "evidence_based"),
    (["Y", "PY"], "low", "evidence_based"),
    (["NI", "NI"], "some_concerns", "no_information_default"),
    (["Y", "PN"], "some_concerns", "evidence_based"),
    (["Y", "N"], "high", "evidence_based"),
    (["N", "N"], "high", "evidence_based"),
])
def test_roll_up_domain_risk_exhaustive(answers, expected_risk, expected_basis):
    questions = [_sq(f"q{i}", a, _evidence() if a != "NI" else []) for i, a in enumerate(answers)]
    risk, basis = roll_up_domain_risk(questions)
    assert risk == expected_risk
    assert basis == expected_basis


def test_roll_up_overall_risk_takes_highest_risk_domain():
    # "selection_of_the_reported_result" has exactly ONE canonical
    # question ("5.1"), the simplest domain to construct a full,
    # validator-passing DomainJudgment for.
    low = DomainJudgment(domain_id="selection_of_the_reported_result", signalling_questions=[_sq("5.1", "Y", _evidence())], domain_risk="low", domain_risk_rationale="r", basis="evidence_based")
    high = DomainJudgment(
        domain_id="missing_outcome_data",
        signalling_questions=[_sq("3.1", "N", _evidence()), _sq("3.2", "NI")],
        domain_risk="high", domain_risk_rationale="r", basis="evidence_based",
    )
    assert roll_up_overall_risk([low, high]) == "high"


# ---------------------------------------------------------------------------
# Provenance validators.
# ---------------------------------------------------------------------------

def test_item_response_rejects_disallowed_provenance_combination():
    with pytest.raises(Exception):
        AppraisalItemResponse(item_id="x", status="yes", evidence=_evidence(), rationale="r", decision_origin="human", evidence_status="verified_from_chunk")


def test_item_response_requires_evidence_for_yes_no():
    with pytest.raises(Exception):
        AppraisalItemResponse(item_id="x", status="yes", evidence=[], rationale="r", decision_origin="ai", evidence_status="verified_from_chunk")


def test_signalling_question_ni_must_not_carry_evidence():
    with pytest.raises(Exception):
        SignallingQuestionResponse(question_id="1.1", answer="NI", evidence=_evidence(), rationale="r", decision_origin="ai", evidence_status="not_provided")


def test_domain_judgment_is_provisional_derived_not_settable():
    q_confirmed = _sq("1.1", "N", _evidence())
    q_confirmed = q_confirmed.model_copy(update={"provisional_pending_human_confirmation": False, "human_confirmation": HumanConfirmation(reviewer="r", note="n", resolved_at="2026-08-22T00:00:00Z")})
    q_still_provisional = _sq("1.2", "N", _evidence())
    domain = DomainJudgment(domain_id="randomisation_process", signalling_questions=[q_confirmed, q_still_provisional], domain_risk="high", domain_risk_rationale="r", basis="evidence_based")
    assert domain.is_provisional is True
    assert not hasattr(domain, "provisional_pending_human_confirmation")


# ---------------------------------------------------------------------------
# qa_run_id: content-addressed, path-excluded, policy-included.
# ---------------------------------------------------------------------------

def _run_id_kwargs(**overrides):
    defaults = dict(
        documents=[("rec-a", "a" * 64)], checklist_id="teaching_appraisal_v1", checklist_version="1",
        checklist_content_hash="h", roll_up_method_ids=[], prompt_template_version="v1",
        model="m", model_config_id="c1", reasoning_effort="medium", max_completion_tokens=100,
        expected_provider_model="m", parser_name="docling", parser_version="1.0",
        parser_options=DoclingParserOptions(), max_batch_chars=100, evidence_verifier_version="v1",
        negative_high_risk_policy="require_human_confirmation", citation_metadata_hash="t",
    )
    defaults.update(overrides)
    return defaults


def test_qa_run_id_same_pdf_content_different_path_is_identical():
    id1 = compute_qa_run_id(**_run_id_kwargs(documents=[("rec-a", "a" * 64)]))
    id2 = compute_qa_run_id(**_run_id_kwargs(documents=[("rec-a", "a" * 64)]))
    assert id1 == id2


def test_qa_run_id_differs_by_negative_high_risk_policy():
    id1 = compute_qa_run_id(**_run_id_kwargs(negative_high_risk_policy="require_human_confirmation"))
    id2 = compute_qa_run_id(**_run_id_kwargs(negative_high_risk_policy="allow_provisional_ai_only"))
    assert id1 != id2


def test_qa_run_id_differs_by_citation_metadata():
    id1 = compute_qa_run_id(**_run_id_kwargs(citation_metadata_hash="a"))
    id2 = compute_qa_run_id(**_run_id_kwargs(citation_metadata_hash="b"))
    assert id1 != id2


def test_qa_run_id_artifacts_manifest_digest_defaults_to_none_and_matches_omitting_it():
    """§5.4: omitting `artifacts_manifest_digest` and passing it as
    explicit `None` must be indistinguishable."""
    id1 = compute_qa_run_id(**_run_id_kwargs())
    id2 = compute_qa_run_id(**_run_id_kwargs(artifacts_manifest_digest=None))
    assert id1 == id2


def test_qa_run_id_differs_by_artifacts_manifest_digest():
    id1 = compute_qa_run_id(**_run_id_kwargs(artifacts_manifest_digest="a" * 64))
    id2 = compute_qa_run_id(**_run_id_kwargs(artifacts_manifest_digest="b" * 64))
    assert id1 != id2


def test_qa_run_id_none_vs_set_artifacts_manifest_digest_differs():
    id1 = compute_qa_run_id(**_run_id_kwargs(artifacts_manifest_digest=None))
    id2 = compute_qa_run_id(**_run_id_kwargs(artifacts_manifest_digest="a" * 64))
    assert id1 != id2


def test_qa_manifest_artifacts_manifest_digest_defaults_to_none():
    """§5.4: production-unchanged default on the QAManifest envelope
    itself, mirroring RunManifest."""
    base_kwargs = dict(
        qa_run_id="qarun-test", checklist_id="teaching_appraisal_v1", checklist_version="1",
        checklist_content_hash="h", roll_up_method_ids=[], prompt_template_version="v",
        model="m", model_config_id="c", reasoning_effort=None, max_completion_tokens=10,
        expected_provider_model="m", parser_name="docling", parser_version="1",
        parser_options={}, max_batch_chars=10, evidence_verifier_version="v",
        negative_high_risk_policy="require_human_confirmation", citation_metadata_hash="h",
        document_ids=["a"], checkpoint_hashes={"a": "h" * 16}, created_at="2026-01-01T00:00:00Z",
    )
    m = qa_module.QAManifest(**base_kwargs, manifest_hash="0" * 64)
    assert m.artifacts_manifest_digest is None

    m2 = qa_module.QAManifest(**base_kwargs, artifacts_manifest_digest="a" * 64, manifest_hash="0" * 64)
    assert m2.artifacts_manifest_digest == "a" * 64


# ---------------------------------------------------------------------------
# Confirm vs resolve -- distinct actions (Section 1.6).
# ---------------------------------------------------------------------------

def test_confirm_preserves_machine_evidence_and_dequeues(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    confirmations = {
        (e["record_id"], e["field_id"]): {"reviewer": "R. Rauleckas", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}
        for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "confirmation"
    }
    result = confirm_quality_appraisal_provisional_item(state, confirmations)
    assert result["requires_human_review"] is False
    assert result["quality_appraisal_review_queue"] == []
    assert len(result["quality_appraisal_confirmations"]) == len(TEACHING_APPRAISAL_ITEMS)

    projection = load_verified_quality_appraisal_resolution_artifact({**state, **result})
    for doc in projection["resolved_quality_projection"]:
        for item in doc["teaching"]["items"]:
            assert item["status"] == "no"
            assert item["decision_origin"] == "ai"
            assert item["evidence_status"] == "verified_from_chunk"
            assert item["evidence"]
            assert item["provisional_pending_human_confirmation"] is False
            assert item["human_confirmation"] is not None


def test_resolve_overrides_unclear_with_human_attested_value(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    resolutions = {
        (e["record_id"], e["field_id"]): {
            "value": "yes", "evidence": [{"quotation": "human-read confirmation", "locator": "p3"}],
            "reviewer": "R. Rauleckas", "note": "manual read", "resolved_at": "2026-08-22T00:00:00Z",
        }
        for e in output["quality_appraisal_review_queue"]
    }
    result = resolve_quality_appraisal_review_queue(state, resolutions)
    assert result["requires_human_review"] is False
    assert result["quality_appraisal_review_queue"] == []

    projection = load_verified_quality_appraisal_resolution_artifact({**state, **result})
    for doc in projection["resolved_quality_projection"]:
        for item in doc["teaching"]["items"]:
            assert item["status"] == "yes"
            assert item["decision_origin"] == "human"
            assert item["evidence_status"] == "human_attested"


# ---------------------------------------------------------------------------
# RoB2 recomputation: category-unchanged rationale staleness fix (Section 1.7).
# ---------------------------------------------------------------------------

def _all_ni_rob2_response(domain_spec):
    return RawDomainBatchResponse(questions={q["question_id"]: RawSignallingQuestionBatchResponse(answer="NI") for q in domain_spec["questions"]})


# Plan Section 1.7's own two required direct regressions: (1) a domain's
# ONLY "N" signalling question resolved to "Y" recomputes domain_risk/
# overall_risk and replaces the rationale with the templated recomputation
# sentence; (2) a domain with TWO independently-"N" leaves, only ONE
# resolved, keeps domain_risk unchanged (the OTHER leaf still earns "high")
# while the rationale is STILL replaced (v3.2.3's own "stale rationale can
# survive an unchanged risk category" fix) -- the original test here
# performed no resolution at all (round-3 review finding), fixed below by
# using require_human_confirmation and resolving via
# resolve_quality_appraisal_review_queue() (Finding 1's own fix: a
# provisional "N" finding is resolvable, not just confirmable).

def test_rob2_single_N_resolved_to_Y_recomputes_domain_risk_and_rationale(tmp_path, monkeypatch):
    chunk = _chunk(0, "The randomization was performed with a coin flip, explicitly not concealed.")

    def _fake_rob2(*, record_title, domain_spec, batch_chunks, config):
        c = batch_chunks[0]
        if domain_spec["domain_id"] == "randomisation_process":
            return RawDomainBatchResponse(questions={
                "1.1": RawSignallingQuestionBatchResponse(answer="N", rationale="explicit", evidence=[RawAppraisalEvidenceItem(chunk_id=c.chunk_id, quotation=c.text)]),
                "1.2": RawSignallingQuestionBatchResponse(answer="NI"),
            })
        return _all_ni_rob2_response(domain_spec)

    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_rob2_domain_batch", _fake_rob2)

    cfg = _cfg(tmp_path, checklist_id="rob2_inspired_study_level_teaching_v1", policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    doc = output["quality_appraisal_results"][0]
    domain = next(d for d in doc["rob2"]["domains"] if d["domain_id"] == "randomisation_process")
    assert domain["domain_risk"] == "high"
    original_rationale = domain["domain_risk_rationale"]

    confirmation_entries = [e for e in output["quality_appraisal_review_queue"] if e["record_id"] == "rec-a" and e["field_id"] == "randomisation_process.1.1"]
    assert len(confirmation_entries) == 1, "setup failed: expected 1.1's negative finding to be a queued confirmation entry"

    key = ("rec-a", "randomisation_process.1.1")
    resolutions = {key: {
        "value": "Y", "evidence": [{"quotation": "on closer reading, allocation WAS random", "locator": "p2"}],
        "reviewer": "R. Rauleckas", "note": "overriding after manual review", "resolved_at": "2026-08-22T00:00:00Z",
    }}
    result = resolve_quality_appraisal_review_queue(state, resolutions)

    projection = load_verified_quality_appraisal_resolution_artifact({**state, **result})
    proj_domain = next(d for d in projection["resolved_quality_projection"][0]["rob2"]["domains"] if d["domain_id"] == "randomisation_process")
    # 1.1 is now "Y" (severity 0), 1.2 is still "NI" (severity 2) -> max
    # severity 2 -> "some_concerns", a genuine change away from "high".
    assert proj_domain["domain_risk"] == "some_concerns"
    assert proj_domain["domain_risk_rationale"] != original_rationale
    assert "Recomputed after" in proj_domain["domain_risk_rationale"]


def test_rob2_one_of_two_negative_leaves_resolved_keeps_risk_but_replaces_rationale(tmp_path, monkeypatch):
    chunk = _chunk(0, "The randomization was performed with a coin flip, explicitly not concealed, and outcome data were missing for most participants.")

    def _fake_rob2(*, record_title, domain_spec, batch_chunks, config):
        c = batch_chunks[0]
        if domain_spec["domain_id"] == "randomisation_process":
            return RawDomainBatchResponse(questions={
                "1.1": RawSignallingQuestionBatchResponse(answer="N", rationale="explicit re 1.1", evidence=[RawAppraisalEvidenceItem(chunk_id=c.chunk_id, quotation=c.text)]),
                "1.2": RawSignallingQuestionBatchResponse(answer="N", rationale="explicit re 1.2", evidence=[RawAppraisalEvidenceItem(chunk_id=c.chunk_id, quotation=c.text)]),
            })
        return _all_ni_rob2_response(domain_spec)

    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_rob2_domain_batch", _fake_rob2)

    cfg = _cfg(tmp_path, checklist_id="rob2_inspired_study_level_teaching_v1", policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    doc = output["quality_appraisal_results"][0]
    domain = next(d for d in doc["rob2"]["domains"] if d["domain_id"] == "randomisation_process")
    assert domain["domain_risk"] == "high"  # both 1.1 and 1.2 independently earn "high"
    original_rationale = domain["domain_risk_rationale"]

    # Resolve ONLY 1.1 to "Y" -- 1.2 stays "N" untouched.
    key = ("rec-a", "randomisation_process.1.1")
    resolutions = {key: {
        "value": "Y", "evidence": [{"quotation": "on closer reading, allocation WAS random", "locator": "p2"}],
        "reviewer": "R. Rauleckas", "note": "overriding after manual review", "resolved_at": "2026-08-22T00:00:00Z",
    }}
    result = resolve_quality_appraisal_review_queue(state, resolutions)

    projection = load_verified_quality_appraisal_resolution_artifact({**state, **result})
    proj_domain = next(d for d in projection["resolved_quality_projection"][0]["rob2"]["domains"] if d["domain_id"] == "randomisation_process")
    # domain_risk arithmetic is correct: STAYS "high", since 1.2 alone
    # still earns it.
    assert proj_domain["domain_risk"] == "high"
    # But the rationale must NOT be presented as still-current -- it
    # described BOTH leaves as negative, which is no longer true of 1.1.
    assert proj_domain["domain_risk_rationale"] != original_rationale
    assert "Recomputed after" in proj_domain["domain_risk_rationale"]

    proj_1_1 = next(q for q in proj_domain["signalling_questions"] if q["question_id"] == "1.1")
    proj_1_2 = next(q for q in proj_domain["signalling_questions"] if q["question_id"] == "1.2")
    assert proj_1_1["answer"] == "Y"
    assert proj_1_1["decision_origin"] == "human"
    assert proj_1_2["answer"] == "N"
    assert proj_1_2["decision_origin"] == "ai"  # untouched


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-22 correction-pass review findings
# (relayed verbatim by the user, independently verified against the code
# before fixing -- see the conversation's own finding-by-finding analysis).
# ---------------------------------------------------------------------------

# Finding 1: a provisional negative judgment could only be CONFIRMED, never
# OVERRIDDEN -- resolve_quality_appraisal_review_queue() only accepted
# queue_kind="resolution" entries, so a reviewer who disagreed with a
# machine "no" had no path to change it to "yes".
def test_finding1_provisional_negative_finding_can_be_overridden_via_resolve(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    confirmation_entries = [e for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "confirmation"]
    assert confirmation_entries, "setup failed: expected provisional confirmation-kind entries"

    resolutions = {
        (e["record_id"], e["field_id"]): {
            "value": "yes", "evidence": [{"quotation": "on closer human reading, the question IS stated", "locator": "p2"}],
            "reviewer": "R. Rauleckas", "note": "overriding the machine's negative finding", "resolved_at": "2026-08-22T00:00:00Z",
        }
        for e in confirmation_entries
    }
    result = resolve_quality_appraisal_review_queue(state, resolutions)
    assert result["quality_appraisal_review_queue"] == []
    assert result["requires_human_review"] is False

    projection = load_verified_quality_appraisal_resolution_artifact({**state, **result})
    for doc in projection["resolved_quality_projection"]:
        for item in doc["teaching"]["items"]:
            assert item["status"] == "yes"
            assert item["decision_origin"] == "human"
            assert item["evidence_status"] == "human_attested"
            assert item["provisional_pending_human_confirmation"] is False


# Finding 2: the checkpoint layer was not actually resumable -- every
# invocation re-appraised every document (re-calling the LLM) before
# qa_run_id was even computed, and checkpoints were blindly overwritten.
def test_finding2_identical_retry_never_recalls_the_llm_and_reuses_checkpoints(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    call_count = {"n": 0}

    def _fake(**kw):
        call_count["n"] += 1
        return _all_items_response("yes", chunk.chunk_id, "The research question is clearly stated")
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output1 = quality_appraisal_node(state)
    assert call_count["n"] == 1

    output2 = quality_appraisal_node(state)  # identical retry, same state/config
    assert call_count["n"] == 1, "the LLM must not be called again for an already-checkpointed document"
    assert output1["quality_appraisal_run_id"] == output2["quality_appraisal_run_id"]
    assert output1["quality_appraisal_manifest_hash"] == output2["quality_appraisal_manifest_hash"]
    assert output1["quality_appraisal_results"] == output2["quality_appraisal_results"]


# Finding 3: a verified, unanimous finding was finalized silently even
# when OTHER batches for the SAME leaf failed the call -- reported as
# complete despite incomplete coverage.
def test_finding3_partial_batch_failure_preserves_finding_but_flags_incomplete_coverage(tmp_path, monkeypatch):
    chunk_a = _chunk(0, "A" * 5000)
    chunk_b = _chunk(1, "B" * 5000)
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))

    def _fake(*, record_title, batch_chunks, config):
        c = batch_chunks[0]
        if c.chunk_id == chunk_a.chunk_id:
            return _all_items_response("yes", chunk_a.chunk_id, chunk_a.text)
        raise qa_module.QACallError("simulated transient failure")
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    cfg.quality_appraisal_max_batch_chars = 100  # forces two separate batches
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    assert output["requires_human_review"] is True
    incomplete_entries = [e for e in output["quality_appraisal_review_queue"] if e["reason"] == "incomplete_coverage"]
    assert len(incomplete_entries) == len(TEACHING_APPRAISAL_ITEMS)

    doc = output["quality_appraisal_results"][0]
    for item in doc["teaching"]["items"]:
        assert item["status"] == "yes"  # the real, verified finding is preserved, never discarded
        assert item["evidence"]
        assert item["coverage_complete"] is False
        assert item["queue_reason"] == "incomplete_coverage"


# Finding 4a: a parse failure assigned pdf_sha256="" unconditionally, so
# two genuinely different (byte-different) PDFs that both fail to parse
# collided on the same run identity.
def test_finding4a_parse_failure_still_gets_a_real_distinct_pdf_sha256(tmp_path, monkeypatch):
    def _raise_parse_error(*a, **k):
        raise qa_module.DocumentParseError("simulated parse failure")
    monkeypatch.setattr(qa_module, "parse_pdf", _raise_parse_error)

    cfg = _cfg(tmp_path)
    pdf_path_a = tmp_path / "rec-a.pdf"
    pdf_path_a.write_bytes(b"content A, deliberately broken")
    pdf_path_b = tmp_path / "rec-b.pdf"
    pdf_path_b.write_bytes(b"content B, a totally different, also-broken file")

    state_a = _state(cfg, [_record("rec-a", pdf_path_a)])
    output_a = quality_appraisal_node(state_a)
    state_b = _state(cfg, [_record("rec-a", pdf_path_b)])  # SAME record_id, DIFFERENT (also-broken) PDF bytes
    output_b = quality_appraisal_node(state_b)

    assert output_a["quality_appraisal_run_id"] != output_b["quality_appraisal_run_id"]
    # And every leaf is still routed to review, exactly like any other
    # unresolved leaf -- never silently treated as passing.
    assert output_a["requires_human_review"] is True
    assert all(e["reason"] == "batch_call_failed" for e in output_a["quality_appraisal_review_queue"])


# Finding 4b: duplicate record_ids in full_text_records silently collapsed
# onto one checkpoint.
def test_finding4b_duplicate_record_id_is_rejected(tmp_path, monkeypatch):
    chunk = _chunk(0, "text")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    record = _record("rec-a", pdf_path)
    state = _state(cfg, [record, record])  # same record_id twice
    with pytest.raises(ValueError, match="duplicate record_id"):
        quality_appraisal_node(state)


# Finding 4c: QACheckpointStore accepted an arbitrary qa_run_id as a raw
# path component, with no safe-token validation (unlike ExtractionCheckpointStore).
def test_finding4c_checkpoint_store_rejects_unsafe_run_id(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        QACheckpointStore(output_dir=tmp_path, qa_run_id="../../../etc")


# Finding 4d: loading a manifest never recomputed qa_run_id from the
# verified checkpoints' own content, nor rejected duplicate document_ids.
def test_finding4d_manifest_rejects_duplicate_document_ids():
    with pytest.raises(Exception):
        qa_module.QAManifest(
            qa_run_id="qarun-x", checklist_id="teaching_appraisal_v1", checklist_version="1",
            checklist_content_hash="h", roll_up_method_ids=[], prompt_template_version="v",
            model="m", model_config_id="c", reasoning_effort=None, max_completion_tokens=10,
            expected_provider_model="m", parser_name="docling", parser_version="1",
            parser_options={}, max_batch_chars=10, evidence_verifier_version="v",
            negative_high_risk_policy="require_human_confirmation", citation_metadata_hash="h",
            document_ids=["a", "a"], checkpoint_hashes={"a": "h" * 16},
            created_at="2026-01-01T00:00:00Z", manifest_hash="0" * 64,
        )


def test_finding4d_tampered_manifest_that_recomputes_a_different_run_id_is_rejected(tmp_path, monkeypatch):
    chunk = _chunk(0, "text")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    manifest_path = Path(output["quality_appraisal_manifest_path"])
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["model"] = "some-other-model-entirely"  # tamper a config field
    # Re-derive the file's OWN self-hash so the bare content_hash/manifest_hash
    # self-check alone would NOT catch this -- only the recompute-from-
    # checkpoints cross-check (Finding 4d fix) should.
    recomputed_self_hash = qa_module.sha256_hex({k: v for k, v in data.items() if k != "manifest_hash"})
    data["manifest_hash"] = recomputed_self_hash
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(qa_module.CheckpointIntegrityError, match="does not describe the request"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


# Finding 5a: the provenance validator accepted an AI "yes"/"Y" response
# carrying evidence_status="not_provided" (only checked tuple membership,
# never cross-checked against the specific status/answer).
def test_finding5a_ai_yes_with_not_provided_evidence_status_is_rejected():
    with pytest.raises(Exception):
        AppraisalItemResponse(
            item_id="x", status="yes", evidence=_evidence(), rationale="r",
            decision_origin="ai", evidence_status="not_provided",
        )


def test_finding5a_ai_signalling_question_Y_with_not_provided_evidence_status_is_rejected():
    with pytest.raises(Exception):
        SignallingQuestionResponse(
            question_id="1.1", answer="Y", evidence=_evidence(), rationale="r",
            decision_origin="ai", evidence_status="not_provided",
        )


# Finding 5b: a human resolution to "not_applicable" crashed at
# projection time -- the projection always set evidence_status=
# "human_attested", but the (then-unconditional) model validator required
# evidence_status="not_applicable" whenever status=="not_applicable".
def test_finding5b_human_resolution_to_not_applicable_does_not_crash(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    resolutions = {
        (e["record_id"], e["field_id"]): {
            "value": "not_applicable", "evidence": [],
            "reviewer": "R. Rauleckas", "note": "genuinely not applicable to this study design", "resolved_at": "2026-08-22T00:00:00Z",
        }
        for e in output["quality_appraisal_review_queue"]
    }
    result = resolve_quality_appraisal_review_queue(state, resolutions)
    # Must not raise -- this is the direct regression for the crash.
    projection = load_verified_quality_appraisal_resolution_artifact({**state, **result})
    for doc in projection["resolved_quality_projection"]:
        for item in doc["teaching"]["items"]:
            assert item["status"] == "not_applicable"
            assert item["decision_origin"] == "human"
            assert item["evidence_status"] == "human_attested"


# Finding 6a: expected_provider_model is bound into qa_run_id's own hash,
# but neither OpenAI call function ever checked the provider's returned
# model against it.
def test_finding6a_teaching_batch_call_rejects_provider_model_mismatch(monkeypatch):
    chunk = _chunk(0, "text")

    class _FakeMessage:
        content = '{"items": {}}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        model = "some-unexpected-model"
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = _FakeChat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    config = _model_config(expected_provider_model="gpt-5.6-terra")
    with pytest.raises(qa_module.QACallError, match="provider model mismatch"):
        qa_module.run_openai_teaching_batch(record_title="t", batch_chunks=[chunk], config=config)


def test_finding6a_rob2_domain_batch_call_rejects_provider_model_mismatch(monkeypatch):
    chunk = _chunk(0, "text")

    class _FakeMessage:
        content = '{"questions": {}}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        model = "some-unexpected-model"
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = _FakeChat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    config = _model_config(expected_provider_model="gpt-5.6-terra")
    domain_spec = qa_module.ROB2_DOMAINS[0]
    with pytest.raises(qa_module.QACallError, match="provider model mismatch"):
        qa_module.run_openai_rob2_domain_batch(record_title="t", domain_spec=domain_spec, batch_chunks=[chunk], config=config)


# Calibration-cap review regression: the installed openai SDK (2.7.2)
# defaults OpenAI()'s own max_retries to 2 -- a bare OpenAI() would
# transparently retry transient failures at the HTTP layer, making both
# QA call functions' own "No retry, no fallback" docstring claims false.
def test_run_openai_teaching_batch_disables_sdk_retries(monkeypatch):
    chunk = _chunk(0, "text")

    class _FakeMessage:
        content = '{"items": {}}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        model = "gpt-5.6-terra"
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    constructor_calls = []

    class _FakeClient:
        def __init__(self, *a, **k):
            constructor_calls.append(k)
            self.chat = _FakeChat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    config = _model_config(expected_provider_model="gpt-5.6-terra")
    qa_module.run_openai_teaching_batch(record_title="t", batch_chunks=[chunk], config=config)
    assert len(constructor_calls) == 1
    assert constructor_calls[0].get("max_retries") == 0


def test_run_openai_rob2_domain_batch_disables_sdk_retries(monkeypatch):
    chunk = _chunk(0, "text")

    class _FakeMessage:
        content = '{"questions": {}}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        model = "gpt-5.6-terra"
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    constructor_calls = []

    class _FakeClient:
        def __init__(self, *a, **k):
            constructor_calls.append(k)
            self.chat = _FakeChat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    config = _model_config(expected_provider_model="gpt-5.6-terra")
    domain_spec = qa_module.ROB2_DOMAINS[0]
    qa_module.run_openai_rob2_domain_batch(record_title="t", domain_spec=domain_spec, batch_chunks=[chunk], config=config)
    assert len(constructor_calls) == 1
    assert constructor_calls[0].get("max_retries") == 0


# Track 1 Phase A follow-up (Codex review, finding 3): both QA call sites
# must be wired through the shared `resolve_openai_api_key()` resolver, not
# left to the SDK's own implicit environment lookup -- an explicit,
# auditable, testable contract, mirroring extraction/openai_runner.py's own
# equivalent regression.
def test_run_openai_teaching_batch_uses_resolved_api_key(monkeypatch):
    chunk = _chunk(0, "text")

    class _FakeMessage:
        content = '{"items": {}}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        model = "gpt-5.6-terra"
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    constructor_calls = []

    class _FakeClient:
        def __init__(self, *a, **k):
            constructor_calls.append(k)
            self.chat = _FakeChat()

    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-value")
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    config = _model_config(expected_provider_model="gpt-5.6-terra")
    qa_module.run_openai_teaching_batch(record_title="t", batch_chunks=[chunk], config=config)
    assert len(constructor_calls) == 1
    assert constructor_calls[0].get("api_key") == "sk-real-env-value"


def test_run_openai_rob2_domain_batch_uses_resolved_api_key(monkeypatch):
    chunk = _chunk(0, "text")

    class _FakeMessage:
        content = '{"questions": {}}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        model = "gpt-5.6-terra"
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    constructor_calls = []

    class _FakeClient:
        def __init__(self, *a, **k):
            constructor_calls.append(k)
            self.chat = _FakeChat()

    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-value")
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    config = _model_config(expected_provider_model="gpt-5.6-terra")
    domain_spec = qa_module.ROB2_DOMAINS[0]
    qa_module.run_openai_rob2_domain_batch(record_title="t", domain_spec=domain_spec, batch_chunks=[chunk], config=config)
    assert len(constructor_calls) == 1
    assert constructor_calls[0].get("api_key") == "sk-real-env-value"


# Finding 7: resolve_quality_appraisal_review_queue() only verified a queue
# entry's (record_id, field_id) KEY against the verified bundle, never its
# own leaf_response CONTENT -- a tampered/stale leaf_response would flow
# straight into the permanent qa_resolutions.json audit trail.
def test_finding7_resolve_rejects_queue_entry_with_tampered_leaf_response(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    tampered_queue = [dict(e) for e in output["quality_appraisal_review_queue"]]
    tampered_queue[0] = dict(tampered_queue[0])
    tampered_queue[0]["leaf_response"] = dict(tampered_queue[0]["leaf_response"])
    tampered_queue[0]["leaf_response"]["rationale"] = "TAMPERED RATIONALE"
    state["quality_appraisal_review_queue"] = tampered_queue

    key = (tampered_queue[0]["record_id"], tampered_queue[0]["field_id"])
    resolutions = {key: {
        "value": "yes", "evidence": [{"quotation": "x", "locator": "p1"}],
        "reviewer": "R. Rauleckas", "note": "n", "resolved_at": "2026-08-22T00:00:00Z",
    }}
    with pytest.raises(ValueError, match="does not match the verified run"):
        resolve_quality_appraisal_review_queue(state, resolutions)


def test_finding7_confirm_rejects_queue_entry_with_tampered_leaf_response(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    tampered_queue = [dict(e) for e in output["quality_appraisal_review_queue"]]
    tampered_queue[0] = dict(tampered_queue[0])
    tampered_queue[0]["leaf_response"] = dict(tampered_queue[0]["leaf_response"])
    tampered_queue[0]["leaf_response"]["rationale"] = "TAMPERED RATIONALE"
    state["quality_appraisal_review_queue"] = tampered_queue

    key = (tampered_queue[0]["record_id"], tampered_queue[0]["field_id"])
    confirmations = {key: {"reviewer": "R. Rauleckas", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}}
    with pytest.raises(ValueError, match="does not match the verified run"):
        confirm_quality_appraisal_provisional_item(state, confirmations)


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-22 ROUND-3 correction-pass review
# findings (relayed verbatim by the user, independently verified against
# the code before fixing).
# ---------------------------------------------------------------------------

# Round-3 finding 1: confirm_quality_appraisal_provisional_item() only
# validated its own confirmation-kind subset of the caller's queue, then
# reconstructed the remaining queue's resolution-kind entries FROM THAT
# SAME (possibly truncated) caller-supplied queue instead of the verified
# bundle -- a caller who omitted genuine resolution entries from their own
# state before confirming would see them silently vanish.
def test_round3_finding1_confirm_rejects_a_queue_missing_genuine_resolution_entries(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def _mixed(**kw):
        # item 0 -> genuinely negative (provisional/confirmation-kind);
        # every other item -> not_found (queued as resolution-kind).
        items = {}
        for i, spec in enumerate(TEACHING_APPRAISAL_ITEMS):
            if i == 0:
                items[spec["item_id"]] = qa_module.RawTeachingItemBatchResponse(
                    outcome="no", rationale="explicit", evidence=[qa_module.RawAppraisalEvidenceItem(chunk_id=chunk.chunk_id, quotation=chunk.text)],
                )
            else:
                items[spec["item_id"]] = qa_module.RawTeachingItemBatchResponse(outcome="not_found")
        return RawTeachingBatchResponse(items=items)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _mixed)

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    resolution_entries = [e for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "resolution"]
    confirmation_entries = [e for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "confirmation"]
    assert resolution_entries and confirmation_entries, "setup failed: expected both a resolution-kind and a confirmation-kind entry"

    # Caller drops the genuine resolution-kind entries from their own copy
    # of the queue before confirming the one negative finding.
    truncated_queue = list(confirmation_entries)
    state["quality_appraisal_review_queue"] = truncated_queue

    key = (confirmation_entries[0]["record_id"], confirmation_entries[0]["field_id"])
    confirmations = {key: {"reviewer": "R. Rauleckas", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}}
    with pytest.raises(ValueError, match="does not exactly match"):
        confirm_quality_appraisal_provisional_item(state, confirmations)


def test_round3_finding1_confirm_restores_dropped_resolution_entries_when_queue_is_otherwise_valid(tmp_path, monkeypatch):
    """The FULL, non-truncated queue confirms correctly and the returned
    queue still contains every genuine resolution-kind entry, proving the
    fix reconstructs from the verified bundle rather than merely copying
    whatever subset the caller happened to pass in."""
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def _mixed(**kw):
        items = {}
        for i, spec in enumerate(TEACHING_APPRAISAL_ITEMS):
            if i == 0:
                items[spec["item_id"]] = qa_module.RawTeachingItemBatchResponse(
                    outcome="no", rationale="explicit", evidence=[qa_module.RawAppraisalEvidenceItem(chunk_id=chunk.chunk_id, quotation=chunk.text)],
                )
            else:
                items[spec["item_id"]] = qa_module.RawTeachingItemBatchResponse(outcome="not_found")
        return RawTeachingBatchResponse(items=items)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _mixed)

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    state.update(output)

    resolution_entries = [e for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "resolution"]
    confirmation_entries = [e for e in output["quality_appraisal_review_queue"] if e["queue_kind"] == "confirmation"]

    key = (confirmation_entries[0]["record_id"], confirmation_entries[0]["field_id"])
    confirmations = {key: {"reviewer": "R. Rauleckas", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}}
    result = confirm_quality_appraisal_provisional_item(state, confirmations)

    remaining_resolution_keys = {(e["record_id"], e["field_id"]) for e in result["quality_appraisal_review_queue"] if e["queue_kind"] == "resolution"}
    expected_resolution_keys = {(e["record_id"], e["field_id"]) for e in resolution_entries}
    assert remaining_resolution_keys == expected_resolution_keys
    assert result["requires_human_review"] is True  # the resolution-kind entries are still genuinely outstanding


# Round-3 finding 2: a MISSING item_id/question_id key in the model's raw
# response was treated identically to an explicit "not_found"/"NI" --
# silently reported as complete coverage instead of an incomplete/failed
# batch response.
def test_round3_finding2_missing_response_key_is_treated_as_incomplete_not_complete(tmp_path, monkeypatch):
    chunk_a = _chunk(0, "A" * 5000)
    chunk_b = _chunk(1, "B" * 5000)
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))

    first_item_id = TEACHING_APPRAISAL_ITEMS[0]["item_id"]

    def _fake(*, record_title, batch_chunks, config):
        c = batch_chunks[0]
        if c.chunk_id == chunk_a.chunk_id:
            return _all_not_found_response()
        # Second batch: the response is well-formed JSON but OMITS the
        # first item_id's key entirely (a real, reproducible schema
        # violation, not a legitimate "not_found").
        items = {i["item_id"]: qa_module.RawTeachingItemBatchResponse(outcome="not_found") for i in TEACHING_APPRAISAL_ITEMS if i["item_id"] != first_item_id}
        return RawTeachingBatchResponse(items=items)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    cfg.quality_appraisal_max_batch_chars = 100  # forces two batches
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    doc = output["quality_appraisal_results"][0]
    first_item = next(i for i in doc["teaching"]["items"] if i["item_id"] == first_item_id)
    assert first_item["coverage_complete"] is False
    assert first_item["queue_reason"] == "batch_call_failed"
    matching_entries = [e for e in output["quality_appraisal_review_queue"] if e["field_id"] == first_item_id]
    assert matching_entries and matching_entries[0]["reason"] == "batch_call_failed"


def test_round3_finding2_missing_response_key_in_rob2_domain_batch(tmp_path, monkeypatch):
    chunk = _chunk(0, "text")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))

    def _fake(*, record_title, domain_spec, batch_chunks, config):
        if domain_spec["domain_id"] != "randomisation_process":
            return _all_ni_rob2_response(domain_spec)
        # Omit "1.1" entirely from the response.
        return RawDomainBatchResponse(questions={"1.2": RawSignallingQuestionBatchResponse(answer="NI")})
    monkeypatch.setattr(qa_module, "run_openai_rob2_domain_batch", _fake)

    cfg = _cfg(tmp_path, checklist_id="rob2_inspired_study_level_teaching_v1")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])

    output = quality_appraisal_node(state)
    doc = output["quality_appraisal_results"][0]
    domain = next(d for d in doc["rob2"]["domains"] if d["domain_id"] == "randomisation_process")
    q1_1 = next(q for q in domain["signalling_questions"] if q["question_id"] == "1.1")
    assert q1_1["coverage_complete"] is False
    assert q1_1["queue_reason"] == "batch_call_failed"


# Round-3 finding 3: qa_resolutions.json's own semantic content (record_id/
# field_id eligibility, original_leaf_response fidelity) was never
# cross-checked against the verified bundle -- only the ENVELOPE's own
# self-hash/run/manifest binding was. A directly-forged, self-hashed file
# was accepted and projected as if genuine.
def test_round3_finding3_forged_self_hashed_resolution_entry_is_rejected(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)

    qa_run_id = output["quality_appraisal_run_id"]
    manifest_hash = output["quality_appraisal_manifest_hash"]
    resolutions_path = Path(output["quality_appraisal_artifact_paths"]["resolutions_path"])

    forged_entry = {
        "record_id": "rec-a", "field_id": TEACHING_APPRAISAL_ITEMS[0]["item_id"], "value": "yes",
        "evidence": [{"chunk_id": "human-attested", "quotation": "fabricated", "locator": "p1"}],
        "human_rationale": "forged", "reviewer": "Attacker", "note": "forged entry", "resolved_at": "2026-08-22T00:00:00Z",
        "qa_run_id": qa_run_id, "manifest_hash": manifest_hash,
        # A FABRICATED original_leaf_response that does not match the real
        # checkpoint's own leaf content for this item.
        "original_leaf_response": {"item_id": TEACHING_APPRAISAL_ITEMS[0]["item_id"], "status": "unclear", "evidence": [], "rationale": "fabricated original", "decision_origin": "ai", "evidence_status": "not_provided", "chunks_examined": [], "batches_examined": 1, "provisional_pending_human_confirmation": False, "human_confirmation": None, "coverage_complete": True, "queue_reason": "no_evidence_found", "conflicting_evidence": None},
    }
    payload = {"qa_run_id": qa_run_id, "manifest_hash": manifest_hash, "entries": [forged_entry], "created_at": "2026-08-22T00:00:00Z"}
    content_hash = qa_module.sha256_hex(payload)
    envelope = dict(payload, content_hash=content_hash)
    resolutions_path.parent.mkdir(parents=True, exist_ok=True)
    resolutions_path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(qa_module.CheckpointIntegrityError, match="original_leaf_response does not match"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


def test_round3_finding3_checkpoint_inner_record_id_must_match_envelope():
    da_foreign = qa_module.DocumentAppraisal(
        record_id="rec-FOREIGN", checklist_id="teaching_appraisal_v1",
        teaching=qa_module.TeachingAppraisal(items=[
            AppraisalItemResponse(item_id=i["item_id"], status="unclear", rationale="r", decision_origin="ai", evidence_status="not_provided")
            for i in TEACHING_APPRAISAL_ITEMS
        ]),
    )
    with pytest.raises(Exception, match="does not match"):
        qa_module.QAAppraisalCheckpoint.build(
            qa_run_id="qarun-x", record_id="rec-a", pdf_sha256="a" * 64,
            checklist_id="teaching_appraisal_v1", checklist_version="1", document_appraisal=da_foreign,
        )


# Round-3 finding 4a: write_manifest() blindly overwrote an existing,
# different published manifest -- the safety lived only in the CALLER's
# own pre-check, not the method itself.
def test_round3_finding4a_write_manifest_rejects_overwriting_different_content(tmp_path):
    store = QACheckpointStore(output_dir=tmp_path, qa_run_id="qarun-test")
    base_kwargs = dict(
        qa_run_id="qarun-test", checklist_id="teaching_appraisal_v1", checklist_version="1",
        checklist_content_hash="h", roll_up_method_ids=[], prompt_template_version="v",
        model="m", model_config_id="c", reasoning_effort=None, max_completion_tokens=10,
        expected_provider_model="m", parser_name="docling", parser_version="1",
        parser_options={}, max_batch_chars=10, evidence_verifier_version="v",
        negative_high_risk_policy="require_human_confirmation", citation_metadata_hash="h",
        document_ids=["a"], checkpoint_hashes={"a": "h" * 16}, created_at="2026-01-01T00:00:00Z",
    )
    def _manifest(**overrides):
        kwargs = dict(base_kwargs)
        kwargs.update(overrides)
        m = qa_module.QAManifest(**kwargs, manifest_hash="0" * 64)
        return m.model_copy(update={"manifest_hash": qa_module.sha256_hex(m.model_dump(mode="json", exclude={"manifest_hash"}))})

    store.write_manifest(_manifest())
    with pytest.raises(qa_module.CheckpointIntegrityError, match="immutable"):
        store.write_manifest(_manifest(model="some-other-model"))


# Round-3 finding 4b: a corrupted existing checkpoint triggered a fresh
# (potentially PAID) LLM call before the inevitable integrity failure was
# ever discovered -- violating this project's own paid-run discipline.
def test_round3_finding4b_corrupted_checkpoint_fails_before_any_llm_call(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("yes", chunk.chunk_id, "The research question is clearly stated"))

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)

    # Corrupt the on-disk checkpoint (tamper its content after the fact).
    checkpoints_dir = Path(output["quality_appraisal_artifact_paths"]["checkpoints_dir"])
    checkpoint_files = list(checkpoints_dir.glob("*.json"))
    assert checkpoint_files
    data = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
    data["pdf_sha256"] = "0" * 64  # tamper a field without fixing content_hash
    checkpoint_files[0].write_text(json.dumps(data), encoding="utf-8")

    def _must_not_be_called(**kw):
        raise AssertionError("run_openai_teaching_batch must NOT be called for a corrupted-checkpoint retry")
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _must_not_be_called)

    with pytest.raises(qa_module.CheckpointIntegrityError):
        quality_appraisal_node(state)


# Round-3 finding 4c: after a resolution is durably published but the
# caller's own state update never lands (simulating a crash before
# graph.update_state()), retrying the IDENTICAL resolve call against the
# stale state must recover idempotently, not raise.
def test_round3_finding4c_identical_resolve_retry_after_simulated_crash_recovers_idempotently(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    stale_state = {**state, **output}  # the state the crashed caller is stuck with

    resolutions = {
        (e["record_id"], e["field_id"]): {
            "value": "yes", "evidence": [{"quotation": "human-read confirmation", "locator": "p3"}],
            "reviewer": "R. Rauleckas", "note": "manual read", "resolved_at": "2026-08-22T00:00:00Z",
        }
        for e in output["quality_appraisal_review_queue"]
    }

    first_result = resolve_quality_appraisal_review_queue(stale_state, resolutions)
    assert first_result["quality_appraisal_review_queue"] == []
    # Simulate the crash: graph.update_state() never happened, so a retry
    # uses the SAME stale_state (still showing the pre-resolution queue).
    second_result = resolve_quality_appraisal_review_queue(stale_state, resolutions)
    assert second_result["quality_appraisal_review_queue"] == []
    assert second_result["quality_appraisal_resolutions"] == first_result["quality_appraisal_resolutions"]


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-22 ROUND-4 correction-pass review
# findings (relayed verbatim by the user, independently verified against
# the code before fixing).
# ---------------------------------------------------------------------------

def _run_clean_appraisal(tmp_path, monkeypatch):
    """Every item verified 'yes' -- no queue entries at all, a genuinely
    'clean' run to forge attacks against."""
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("yes", chunk.chunk_id, chunk.text))
    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    assert output["quality_appraisal_review_queue"] == []
    return state, output


def _write_forged_envelope(path, *, qa_run_id, manifest_hash, entries):
    payload = {"qa_run_id": qa_run_id, "manifest_hash": manifest_hash, "entries": entries, "created_at": "2026-08-22T00:00:00Z"}
    content_hash = qa_module.sha256_hex(payload)
    envelope = dict(payload, content_hash=content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")


# Round-4 finding 1a: a self-consistent resolution overriding a clean,
# never-queued "yes" leaf was accepted and projected as if genuine.
def test_round4_finding1a_resolution_on_a_never_queued_leaf_is_rejected(tmp_path, monkeypatch):
    state, output = _run_clean_appraisal(tmp_path, monkeypatch)
    qa_run_id = output["quality_appraisal_run_id"]
    manifest_hash = output["quality_appraisal_manifest_hash"]
    doc = output["quality_appraisal_results"][0]
    clean_item_id = TEACHING_APPRAISAL_ITEMS[0]["item_id"]
    clean_leaf = next(i for i in doc["teaching"]["items"] if i["item_id"] == clean_item_id)
    assert clean_leaf["queue_reason"] is None and not clean_leaf["provisional_pending_human_confirmation"]

    forged_entry = {
        "record_id": "rec-a", "field_id": clean_item_id, "value": "no", "evidence": [{"chunk_id": "human-attested", "quotation": "forged", "locator": "p1"}],
        "human_rationale": "forged override", "reviewer": "Attacker", "note": "forged", "resolved_at": "2026-08-22T00:00:00Z",
        "qa_run_id": qa_run_id, "manifest_hash": manifest_hash,
        "original_leaf_response": clean_leaf,  # a genuine, byte-accurate copy of the REAL (never-queued) leaf
    }
    resolutions_path = Path(output["quality_appraisal_artifact_paths"]["resolutions_path"])
    _write_forged_envelope(resolutions_path, qa_run_id=qa_run_id, manifest_hash=manifest_hash, entries=[forged_entry])

    with pytest.raises(qa_module.CheckpointIntegrityError, match="never a canonical queue entry"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


# Round-4 finding 1b: a confirmation attached to an unclear, non-provisional
# leaf was accepted.
def test_round4_finding1b_confirmation_on_a_non_provisional_leaf_is_rejected(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())
    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)

    qa_run_id = output["quality_appraisal_run_id"]
    manifest_hash = output["quality_appraisal_manifest_hash"]
    doc = output["quality_appraisal_results"][0]
    unclear_item_id = TEACHING_APPRAISAL_ITEMS[0]["item_id"]
    unclear_leaf = next(i for i in doc["teaching"]["items"] if i["item_id"] == unclear_item_id)
    assert unclear_leaf["queue_reason"] is not None and not unclear_leaf["provisional_pending_human_confirmation"]

    forged_entry = {
        "record_id": "rec-a", "field_id": unclear_item_id, "reviewer": "Attacker", "note": "forged agreement",
        "resolved_at": "2026-08-22T00:00:00Z", "qa_run_id": qa_run_id, "manifest_hash": manifest_hash,
        "original_leaf_response": unclear_leaf,
    }
    confirmations_path = Path(output["quality_appraisal_artifact_paths"]["confirmations_path"])
    _write_forged_envelope(confirmations_path, qa_run_id=qa_run_id, manifest_hash=manifest_hash, entries=[forged_entry])

    with pytest.raises(qa_module.CheckpointIntegrityError, match="never a canonical queue_kind='confirmation' entry"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


# Round-4 finding 1c: the same (record_id, field_id) key may appear in
# BOTH qa_resolutions.json and qa_confirmations.json -- projection then
# applied both.
def test_round4_finding1c_key_in_both_resolutions_and_confirmations_is_rejected(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))
    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)

    qa_run_id = output["quality_appraisal_run_id"]
    manifest_hash = output["quality_appraisal_manifest_hash"]
    doc = output["quality_appraisal_results"][0]
    negative_item_id = TEACHING_APPRAISAL_ITEMS[0]["item_id"]
    negative_leaf = next(i for i in doc["teaching"]["items"] if i["item_id"] == negative_item_id)
    assert negative_leaf["provisional_pending_human_confirmation"] is True

    resolution_entry = {
        "record_id": "rec-a", "field_id": negative_item_id, "value": "yes",
        "evidence": [{"chunk_id": "human-attested", "quotation": "override", "locator": "p1"}],
        "human_rationale": "override", "reviewer": "R1", "note": "override", "resolved_at": "2026-08-22T00:00:00Z",
        "qa_run_id": qa_run_id, "manifest_hash": manifest_hash, "original_leaf_response": negative_leaf,
    }
    confirmation_entry = {
        "record_id": "rec-a", "field_id": negative_item_id, "reviewer": "R2", "note": "agreed",
        "resolved_at": "2026-08-22T00:00:00Z", "qa_run_id": qa_run_id, "manifest_hash": manifest_hash,
        "original_leaf_response": negative_leaf,
    }
    resolutions_path = Path(output["quality_appraisal_artifact_paths"]["resolutions_path"])
    confirmations_path = Path(output["quality_appraisal_artifact_paths"]["confirmations_path"])
    _write_forged_envelope(resolutions_path, qa_run_id=qa_run_id, manifest_hash=manifest_hash, entries=[resolution_entry])
    _write_forged_envelope(confirmations_path, qa_run_id=qa_run_id, manifest_hash=manifest_hash, entries=[confirmation_entry])

    with pytest.raises(qa_module.CheckpointIntegrityError, match="mutually exclusive"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


# Round-4 finding 2: a resolve retry from stale state with DIFFERENT
# content than what was already published was silently discarded as if
# it were a benign idempotent replay.
def test_round4_finding2_resolve_retry_with_different_value_raises(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())
    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    stale_state = {**state, **output}

    key = (output["quality_appraisal_review_queue"][0]["record_id"], output["quality_appraisal_review_queue"][0]["field_id"])
    first_resolutions = {key: {
        "value": "yes", "evidence": [{"quotation": "human-read confirmation", "locator": "p3"}],
        "reviewer": "R. Rauleckas", "note": "manual read", "resolved_at": "2026-08-22T00:00:00Z",
    }}
    resolve_quality_appraisal_review_queue(stale_state, first_resolutions)

    # Retry from the SAME stale state, but with a genuinely DIFFERENT value.
    conflicting_resolutions = {key: {
        "value": "no", "evidence": [{"quotation": "a different human decided differently", "locator": "p4"}],
        "reviewer": "R2", "note": "conflicting read", "resolved_at": "2026-08-22T01:00:00Z",
    }}
    with pytest.raises(qa_module.CheckpointIntegrityError, match="DIFFERENT"):
        resolve_quality_appraisal_review_queue(stale_state, conflicting_resolutions)


# Round-4 finding 2 continued: a resolve-versus-confirm conflict (the SAME
# key closed via one action, retried via the OTHER) must raise.
def test_round4_finding2_confirm_after_resolve_conflict_raises(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))
    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    stale_state = {**state, **output}

    confirmation_entry = output["quality_appraisal_review_queue"][0]
    key = (confirmation_entry["record_id"], confirmation_entry["field_id"])

    # Resolve (override) it first -- this closes it via RESOLUTION.
    resolve_quality_appraisal_review_queue(stale_state, {key: {
        "value": "yes", "evidence": [{"quotation": "override", "locator": "p1"}],
        "reviewer": "R1", "note": "override", "resolved_at": "2026-08-22T00:00:00Z",
    }})

    # Now retry a CONFIRM against the same key from the SAME stale state
    # (which still shows it as an outstanding confirmation entry).
    with pytest.raises(qa_module.CheckpointIntegrityError, match="already closed via RESOLUTION"):
        confirm_quality_appraisal_provisional_item(stale_state, {key: {"reviewer": "R2", "note": "agreed", "resolved_at": "2026-08-22T01:00:00Z"}})


def test_round4_finding2_resolve_after_confirm_conflict_raises(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))
    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    stale_state = {**state, **output}

    confirmation_entry = output["quality_appraisal_review_queue"][0]
    key = (confirmation_entry["record_id"], confirmation_entry["field_id"])

    confirm_quality_appraisal_provisional_item(stale_state, {key: {"reviewer": "R1", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}})

    with pytest.raises(qa_module.CheckpointIntegrityError, match="already closed via CONFIRMATION"):
        resolve_quality_appraisal_review_queue(stale_state, {key: {
            "value": "yes", "evidence": [{"quotation": "override anyway", "locator": "p1"}],
            "reviewer": "R2", "note": "override", "resolved_at": "2026-08-22T01:00:00Z",
        }})


# Round-4 finding 3: checkpoint semantic structure under-validated.
def test_round4_finding3_checkpoint_checklist_version_mismatch_rejected(tmp_path, monkeypatch):
    """Tampers the MANIFEST's own checklist_version (leaving the real
    checkpoint untouched, so its content_hash still matches the manifest's
    own checkpoint_hashes binding -- the pre-existing hash-binding check
    would otherwise catch a tampered CHECKPOINT before this NEW
    checkpoint-vs-manifest checklist_id/version cross-check ever runs)."""
    state, output = _run_clean_appraisal(tmp_path, monkeypatch)
    manifest_path = Path(output["quality_appraisal_manifest_path"])
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["checklist_version"] = "999"
    recomputed_hash = qa_module.sha256_hex({k: v for k, v in data.items() if k != "manifest_hash"})
    data["manifest_hash"] = recomputed_hash
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(qa_module.CheckpointIntegrityError, match="checklist_version"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


def test_round4_finding3_teaching_appraisal_with_missing_item_is_rejected():
    with pytest.raises(Exception, match="does not exactly match"):
        qa_module.TeachingAppraisal(items=[
            AppraisalItemResponse(item_id=TEACHING_APPRAISAL_ITEMS[0]["item_id"], status="unclear", rationale="r", decision_origin="ai", evidence_status="not_provided"),
        ])


def test_round4_finding3_domain_judgment_with_contradictory_risk_is_rejected():
    # Both signalling questions answer "Y" (severity 0 -> "low"), but the
    # stored label claims "high" -- contradicts roll_up_domain_risk()'s
    # own deterministic result for these SAME questions.
    with pytest.raises(Exception, match="does not match"):
        DomainJudgment(
            domain_id="randomisation_process",
            signalling_questions=[_sq("1.1", "Y", _evidence()), _sq("1.2", "Y", _evidence())],
            domain_risk="high", domain_risk_rationale="r", basis="evidence_based",
        )


def test_round4_finding3_rob2_appraisal_with_only_one_domain_is_rejected():
    domain = DomainJudgment(
        domain_id="randomisation_process",
        signalling_questions=[_sq("1.1", "Y", _evidence()), _sq("1.2", "Y", _evidence())],
        domain_risk="low", domain_risk_rationale="r", basis="evidence_based",
    )
    with pytest.raises(Exception, match="does not exactly match the 5 canonical"):
        qa_module.RoB2InspiredAppraisal(domains=[domain], overall_risk="low", overall_risk_rationale="r")


# Round-4 finding 4: preflight discovers a LATER record's corrupted
# checkpoint before any LLM call is made for ANY record, including
# EARLIER ones that have no checkpoint at all yet.
def test_round4_finding4_preflight_discovers_corruption_before_any_llm_call(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    call_count = {"n": 0}

    def _fake(**kw):
        call_count["n"] += 1
        return _all_items_response("yes", chunk.chunk_id, chunk.text)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    pdf_path_a = tmp_path / "rec-a.pdf"
    pdf_path_a.write_bytes(b"x")
    pdf_path_b = tmp_path / "rec-b.pdf"
    pdf_path_b.write_bytes(b"y")

    # Appraise BOTH records together first (real checkpoints for both,
    # same qa_run_id/directory), then simulate "rec-a's checkpoint was
    # never actually written yet" by deleting it, and corrupt rec-b's --
    # both under the SAME qa_run_id a re-run over [rec-a, rec-b] (rec-a
    # iterated FIRST) would actually use.
    state_both = _state(cfg, [_record("rec-a", pdf_path_a), _record("rec-b", pdf_path_b)])
    output_first = quality_appraisal_node(state_both)
    assert call_count["n"] == 2

    checkpoints_dir = Path(output_first["quality_appraisal_artifact_paths"]["checkpoints_dir"])
    rec_a_ckpt_path = qa_module.QACheckpointStore(
        output_dir=Path(cfg.output_dir), qa_run_id=output_first["quality_appraisal_run_id"],
    ).checkpoint_path("rec-a")
    rec_b_ckpt_path = qa_module.QACheckpointStore(
        output_dir=Path(cfg.output_dir), qa_run_id=output_first["quality_appraisal_run_id"],
    ).checkpoint_path("rec-b")
    rec_a_ckpt_path.unlink()  # rec-a "not yet appraised"
    data = json.loads(rec_b_ckpt_path.read_text(encoding="utf-8"))
    data["pdf_sha256"] = "0" * 64  # tamper without fixing content_hash
    rec_b_ckpt_path.write_text(json.dumps(data), encoding="utf-8")

    call_count["n"] = 0
    state_both_retry = _state(cfg, [_record("rec-a", pdf_path_a), _record("rec-b", pdf_path_b)])
    with pytest.raises(qa_module.CheckpointIntegrityError):
        quality_appraisal_node(state_both_retry)
    assert call_count["n"] == 0, "no LLM call should have been made for rec-a before rec-b's corruption was discovered"


# Round-4 finding 4: _atomic_write_json's temp filename is now unique per
# call -- two concurrent writers targeting the same final path never
# collide on the SAME temp file (the previous fixed "<name>.tmp" let one
# process's partial write be clobbered mid-write by another before either
# reached its own atomic replace). Windows' own `MoveFileEx` can still
# transiently refuse a concurrent replace of the SAME final path with
# PermissionError when uncoordinated by an external lock (real callers
# always hold `_qa_resolutions_lock` around this -- this test exercises
# `_atomic_write_json` directly, without that lock, specifically to prove
# the TEMP-FILE-corruption class of race is gone; a transient replace
# contention is retried here, matching how a real caller would retry
# after the lock briefly serializes it).
def test_round4_finding4_atomic_write_json_concurrent_writes_do_not_corrupt(tmp_path):
    import threading
    target = tmp_path / "shared.json"
    errors = []

    def _writer(n):
        for _ in range(10):
            for attempt in range(20):
                try:
                    qa_module._atomic_write_json(target, {"n": n})
                    break
                except PermissionError:
                    time_module.sleep(0.001)
            else:
                errors.append(RuntimeError(f"writer {n} could not publish after retries"))

    import time as time_module
    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # The final file must be valid, parseable JSON -- never a partially
    # overwritten mix of two writers' content (proves no TEMP-FILE
    # collision corrupted anything, regardless of which writer's replace
    # happened to land last).
    final = json.loads(target.read_text(encoding="utf-8"))
    assert "n" in final
    # No leftover temp files.
    assert list(tmp_path.glob("shared.json.*.tmp")) == []


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-22 ROUND-5 correction-pass review
# findings (relayed verbatim by the user, independently verified against
# the code before fixing).
# ---------------------------------------------------------------------------

# Round-5 finding 1a: eligibility ignored `negative_high_risk_policy` -- a
# provisional negative under allow_provisional_ai_only is DELIBERATELY
# absent from the blocking queue (routed to the non-blocking ledger
# instead), yet a forged resolution targeting it was accepted because the
# old check only looked at the leaf's own provisional flag.
def test_round5_finding1a_resolution_on_ai_only_provisional_ledger_leaf_is_rejected(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))

    cfg = _cfg(tmp_path, policy="allow_provisional_ai_only")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    assert output["quality_appraisal_review_queue"] == []  # never queued under this policy
    assert output["quality_appraisal_provisional_ledger"]  # but IS on the non-blocking ledger

    qa_run_id = output["quality_appraisal_run_id"]
    manifest_hash = output["quality_appraisal_manifest_hash"]
    doc = output["quality_appraisal_results"][0]
    negative_item_id = TEACHING_APPRAISAL_ITEMS[0]["item_id"]
    negative_leaf = next(i for i in doc["teaching"]["items"] if i["item_id"] == negative_item_id)
    assert negative_leaf["provisional_pending_human_confirmation"] is True
    assert negative_leaf["queue_reason"] is None

    forged_entry = {
        "record_id": "rec-a", "field_id": negative_item_id, "value": "yes",
        "evidence": [{"chunk_id": "human-attested", "quotation": "forged", "locator": "p1"}],
        "human_rationale": "forged override", "reviewer": "Attacker", "note": "forged", "resolved_at": "2026-08-22T00:00:00Z",
        "qa_run_id": qa_run_id, "manifest_hash": manifest_hash, "original_leaf_response": negative_leaf,
    }
    resolutions_path = Path(output["quality_appraisal_artifact_paths"]["resolutions_path"])
    _write_forged_envelope(resolutions_path, qa_run_id=qa_run_id, manifest_hash=manifest_hash, entries=[forged_entry])

    with pytest.raises(qa_module.CheckpointIntegrityError, match="never a canonical queue entry"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


# Round-5 finding 1b: a confirmation must additionally require the leaf's
# own CANONICAL queue_kind to be specifically "confirmation" -- a leaf
# that is BOTH queued for resolution (queue_reason set, e.g.
# "incomplete_coverage") AND provisional is a "resolution"-kind entry by
# precedence (per _build_review_queue_and_ledger's own rule), never
# "confirmation"-kind, even though its own provisional flag is True.
def test_round5_finding1b_confirmation_on_a_resolution_kind_leaf_is_rejected(tmp_path, monkeypatch):
    chunk_a = _chunk(0, "A" * 5000)
    chunk_b = _chunk(1, "B" * 5000)
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk_a, chunk_b)))

    def _fake(*, record_title, batch_chunks, config):
        c = batch_chunks[0]
        if c.chunk_id == chunk_a.chunk_id:
            return _all_items_response("no", chunk_a.chunk_id, chunk_a.text)
        raise qa_module.QACallError("simulated transient failure")
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    cfg.quality_appraisal_max_batch_chars = 100  # forces two separate batches
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)

    doc = output["quality_appraisal_results"][0]
    mixed_item_id = TEACHING_APPRAISAL_ITEMS[0]["item_id"]
    mixed_leaf = next(i for i in doc["teaching"]["items"] if i["item_id"] == mixed_item_id)
    # Both provisional (negative finding) AND queued as a RESOLUTION
    # (incomplete coverage) -- this is the exact overlapping case.
    assert mixed_leaf["provisional_pending_human_confirmation"] is True
    assert mixed_leaf["queue_reason"] == "incomplete_coverage"
    queue_entry = next(e for e in output["quality_appraisal_review_queue"] if e["field_id"] == mixed_item_id)
    assert queue_entry["queue_kind"] == "resolution"  # precedence: never "confirmation"

    qa_run_id = output["quality_appraisal_run_id"]
    manifest_hash = output["quality_appraisal_manifest_hash"]
    forged_entry = {
        "record_id": "rec-a", "field_id": mixed_item_id, "reviewer": "Attacker", "note": "forged agreement",
        "resolved_at": "2026-08-22T00:00:00Z", "qa_run_id": qa_run_id, "manifest_hash": manifest_hash,
        "original_leaf_response": mixed_leaf,
    }
    confirmations_path = Path(output["quality_appraisal_artifact_paths"]["confirmations_path"])
    _write_forged_envelope(confirmations_path, qa_run_id=qa_run_id, manifest_hash=manifest_hash, entries=[forged_entry])

    with pytest.raises(qa_module.CheckpointIntegrityError, match="never a canonical queue_kind='confirmation' entry"):
        load_verified_quality_appraisal_resolution_artifact({**state, **output})


# Round-5 finding 2a: a resolve retry from stale state with the SAME
# value/evidence but DIFFERENT reviewer/note/resolved_at metadata was
# silently discarded as a benign no-op.
def test_round5_finding2a_resolve_retry_with_different_reviewer_raises(tmp_path, monkeypatch):
    chunk = _chunk(0, "Unrelated text about funding sources.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_not_found_response())
    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    stale_state = {**state, **output}

    key = (output["quality_appraisal_review_queue"][0]["record_id"], output["quality_appraisal_review_queue"][0]["field_id"])
    first_resolutions = {key: {
        "value": "yes", "evidence": [{"quotation": "human-read confirmation", "locator": "p3"}],
        "reviewer": "R1", "note": "manual read", "resolved_at": "2026-08-22T00:00:00Z",
    }}
    resolve_quality_appraisal_review_queue(stale_state, first_resolutions)

    # SAME value/evidence, but a genuinely DIFFERENT reviewer.
    conflicting_reviewer_resolutions = {key: {
        "value": "yes", "evidence": [{"quotation": "human-read confirmation", "locator": "p3"}],
        "reviewer": "R2", "note": "manual read", "resolved_at": "2026-08-22T00:00:00Z",
    }}
    with pytest.raises(qa_module.CheckpointIntegrityError, match="DIFFERENT content"):
        resolve_quality_appraisal_review_queue(stale_state, conflicting_reviewer_resolutions)


# Round-5 finding 2b: a confirm retry from stale state with a DIFFERENT
# reviewer/note/timestamp was treated as automatically identical.
def test_round5_finding2b_confirm_retry_with_different_reviewer_raises(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))
    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    stale_state = {**state, **output}

    confirmation_entry = output["quality_appraisal_review_queue"][0]
    key = (confirmation_entry["record_id"], confirmation_entry["field_id"])

    confirm_quality_appraisal_provisional_item(stale_state, {key: {"reviewer": "R1", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}})

    with pytest.raises(qa_module.CheckpointIntegrityError, match="DIFFERENT content"):
        confirm_quality_appraisal_provisional_item(stale_state, {key: {"reviewer": "R2", "note": "agreed", "resolved_at": "2026-08-22T00:00:00Z"}})


# Round-5 finding 3: a validly-hashed seven-item machine checkpoint where
# every answer is unresolved ("unclear") but NO item has a queue_reason
# loaded successfully -- the checkpoint envelope itself must enforce
# machine-layer invariants, since AppraisalItemResponse alone legitimately
# allows this shape for the PROJECTED (human-resolved-to-"unclear") layer.
def test_round5_finding3_unresolved_machine_checkpoint_without_queue_reason_is_rejected():
    items = [
        AppraisalItemResponse(
            item_id=i["item_id"], status="unclear", rationale="r",
            decision_origin="ai", evidence_status="not_provided",
            queue_reason=None,  # the illegitimate shape: unresolved but no reason
        )
        for i in TEACHING_APPRAISAL_ITEMS
    ]
    da = qa_module.DocumentAppraisal(record_id="rec-a", checklist_id="teaching_appraisal_v1", teaching=qa_module.TeachingAppraisal(items=items))
    with pytest.raises(Exception, match="requires the leaf's own status/answer to be definitive"):
        qa_module.QAAppraisalCheckpoint.build(
            qa_run_id="qarun-x", record_id="rec-a", pdf_sha256="a" * 64,
            checklist_id="teaching_appraisal_v1", checklist_version="1", document_appraisal=da,
        )


def test_round5_finding3_checkpoint_with_human_decision_origin_is_rejected():
    items = [
        AppraisalItemResponse(item_id=i["item_id"], status="not_applicable", rationale="r", decision_origin="ai", evidence_status="not_applicable")
        for i in TEACHING_APPRAISAL_ITEMS[1:]
    ]
    # One leaf carries a HUMAN decision_origin -- illegal for a frozen
    # machine checkpoint (human overrides live in qa_resolutions.json,
    # applied only at projection time).
    items.insert(0, AppraisalItemResponse(
        item_id=TEACHING_APPRAISAL_ITEMS[0]["item_id"], status="yes",
        evidence=[AppraisalEvidence(chunk_id="human-attested", quotation="q", locator="p1")],
        rationale="r", decision_origin="human", evidence_status="human_attested",
    ))
    da = qa_module.DocumentAppraisal(record_id="rec-a", checklist_id="teaching_appraisal_v1", teaching=qa_module.TeachingAppraisal(items=items))
    with pytest.raises(Exception, match="decision_origin='ai'"):
        qa_module.QAAppraisalCheckpoint.build(
            qa_run_id="qarun-x", record_id="rec-a", pdf_sha256="a" * 64,
            checklist_id="teaching_appraisal_v1", checklist_version="1", document_appraisal=da,
        )


# Round-5 finding 4a: manifest preflight discovers a deleted manifest-bound
# checkpoint before any (paid) LLM call is made.
def test_round5_finding4a_deleted_manifest_bound_checkpoint_fails_before_any_llm_call(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    call_count = {"n": 0}

    def _fake(**kw):
        call_count["n"] += 1
        return _all_items_response("yes", chunk.chunk_id, chunk.text)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    assert call_count["n"] == 1
    assert Path(output["quality_appraisal_manifest_path"]).exists()

    # Delete the manifest-BOUND checkpoint (manifest itself remains).
    checkpoints_dir = Path(output["quality_appraisal_artifact_paths"]["checkpoints_dir"])
    checkpoint_files = list(checkpoints_dir.glob("*.json"))
    assert checkpoint_files
    checkpoint_files[0].unlink()

    call_count["n"] = 0
    with pytest.raises(qa_module.CheckpointIntegrityError):
        quality_appraisal_node(state)
    assert call_count["n"] == 0, "no LLM call should have been made once a manifest already exists but a bound checkpoint is missing"


# Round-5 finding 4b: relocating identical PDF bytes to a new path
# preserves qa_run_id (path is not part of the hash), and resuming from
# the new path must NOT attempt to rebuild/republish the manifest with
# the new path -- it must reuse the ORIGINAL published manifest verbatim.
def test_round5_finding4b_relocated_pdf_resume_reuses_original_manifest(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    call_count = {"n": 0}

    def _fake(**kw):
        call_count["n"] += 1
        return _all_items_response("yes", chunk.chunk_id, chunk.text)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    original_path = tmp_path / "original_location.pdf"
    original_path.write_bytes(b"identical pdf content")
    state_original = _state(cfg, [_record("rec-a", original_path)])
    output_original = quality_appraisal_node(state_original)
    assert call_count["n"] == 1

    # Relocate: identical bytes, new path.
    relocated_path = tmp_path / "subdir" / "relocated.pdf"
    relocated_path.parent.mkdir(parents=True, exist_ok=True)
    relocated_path.write_bytes(b"identical pdf content")

    call_count["n"] = 0
    state_relocated = _state(cfg, [_record("rec-a", relocated_path)])
    output_relocated = quality_appraisal_node(state_relocated)

    assert call_count["n"] == 0, "byte-identical relocated PDF must not trigger a fresh LLM call"
    assert output_relocated["quality_appraisal_run_id"] == output_original["quality_appraisal_run_id"]
    assert output_relocated["quality_appraisal_manifest_hash"] == output_original["quality_appraisal_manifest_hash"]

    # The manifest itself must retain the ORIGINAL path as its own
    # provenance -- never silently rewritten to the relocated one.
    manifest_data = json.loads(Path(output_relocated["quality_appraisal_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest_data["pdf_paths"]["rec-a"] == str(original_path)


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-22 ROUND-6 correction-pass review
# findings (relayed verbatim by the user, independently verified against
# the code before fixing).
# ---------------------------------------------------------------------------

def _clean_teaching_items(*, exclude_first: bool = False) -> list:
    specs = TEACHING_APPRAISAL_ITEMS[1:] if exclude_first else TEACHING_APPRAISAL_ITEMS
    return [
        AppraisalItemResponse(
            item_id=i["item_id"], status="yes",
            evidence=[AppraisalEvidence(chunk_id="c1", quotation="q", locator="p1")],
            rationale="r", decision_origin="ai", evidence_status="verified_from_chunk",
        )
        for i in specs
    ]


# Round-6 finding 1a: a machine "yes" leaf with
# provisional_pending_human_confirmation=True would incorrectly enter the
# confirmation queue -- only a DEFINITIVE NEGATIVE ("no"/"N"/"PN") machine
# finding may ever be provisional.
def test_round6_finding1a_machine_yes_with_provisional_true_is_rejected():
    items = _clean_teaching_items(exclude_first=True)
    items.insert(0, AppraisalItemResponse(
        item_id=TEACHING_APPRAISAL_ITEMS[0]["item_id"], status="yes",
        evidence=[AppraisalEvidence(chunk_id="c1", quotation="q", locator="p1")],
        rationale="r", decision_origin="ai", evidence_status="verified_from_chunk",
        provisional_pending_human_confirmation=True,  # illegitimate for a "yes"
    ))
    da = qa_module.DocumentAppraisal(record_id="rec-a", checklist_id="teaching_appraisal_v1", teaching=qa_module.TeachingAppraisal(items=items))
    with pytest.raises(Exception, match="does not match the expected"):
        qa_module.QAAppraisalCheckpoint.build(
            qa_run_id="qarun-x", record_id="rec-a", pdf_sha256="a" * 64,
            checklist_id="teaching_appraisal_v1", checklist_version="1", document_appraisal=da,
        )


# Round-6 finding 1b: a machine "yes" leaf with queue_reason="made_up_reason"
# would incorrectly enter the resolution queue -- queue_reason must be one
# of _aggregate_leaf()'s own real outcomes.
def test_round6_finding1b_machine_yes_with_fabricated_queue_reason_is_rejected():
    items = _clean_teaching_items(exclude_first=True)
    items.insert(0, AppraisalItemResponse(
        item_id=TEACHING_APPRAISAL_ITEMS[0]["item_id"], status="yes",
        evidence=[AppraisalEvidence(chunk_id="c1", quotation="q", locator="p1")],
        rationale="r", decision_origin="ai", evidence_status="verified_from_chunk",
        queue_reason="made_up_reason",  # not one of the real aggregation outcomes
    ))
    da = qa_module.DocumentAppraisal(record_id="rec-a", checklist_id="teaching_appraisal_v1", teaching=qa_module.TeachingAppraisal(items=items))
    with pytest.raises(Exception, match="is not one of the aggregation's own real outcomes"):
        qa_module.QAAppraisalCheckpoint.build(
            qa_run_id="qarun-x", record_id="rec-a", pdf_sha256="a" * 64,
            checklist_id="teaching_appraisal_v1", checklist_version="1", document_appraisal=da,
        )


# Round-6 finding 1c: a machine "unclear" leaf claiming
# queue_reason="incomplete_coverage" (which requires a RESOLVED leaf) while
# ALSO claiming coverage_complete=True (which that reason always pairs
# with False) -- an internally contradictory producer-impossible shape.
def test_round6_finding1c_unclear_with_incomplete_coverage_reason_but_coverage_complete_true_is_rejected():
    items = _clean_teaching_items(exclude_first=True)
    items.insert(0, AppraisalItemResponse(
        item_id=TEACHING_APPRAISAL_ITEMS[0]["item_id"], status="unclear", rationale="r",
        decision_origin="ai", evidence_status="not_provided",
        queue_reason="incomplete_coverage", coverage_complete=True,  # contradictory
    ))
    da = qa_module.DocumentAppraisal(record_id="rec-a", checklist_id="teaching_appraisal_v1", teaching=qa_module.TeachingAppraisal(items=items))
    with pytest.raises(Exception, match="requires the leaf's own status/answer to be definitive"):
        qa_module.QAAppraisalCheckpoint.build(
            qa_run_id="qarun-x", record_id="rec-a", pdf_sha256="a" * 64,
            checklist_id="teaching_appraisal_v1", checklist_version="1", document_appraisal=da,
        )


# Round-6 finding 1, continuity check: _build_review_queue_and_ledger()
# now delegates to the SAME _canonical_queue_kind_for_leaf() helper the
# verifiers use -- a real run's queue construction and the verifiers'
# eligibility checks can no longer independently drift apart.
def test_round6_finding1_build_review_queue_uses_canonical_helper(tmp_path, monkeypatch):
    chunk = _chunk(0, "The study explicitly states no research question was formulated.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", lambda **kw: _all_items_response("no", chunk.chunk_id, chunk.text))
    cfg = _cfg(tmp_path, policy="require_human_confirmation")
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"x")
    state = _state(cfg, [_record("rec-a", pdf_path)])
    output = quality_appraisal_node(state)
    for entry in output["quality_appraisal_review_queue"]:
        da = qa_module.DocumentAppraisal.model_validate(output["quality_appraisal_results"][0])
        leaf = da.leaves()[entry["field_id"]]
        assert qa_module._canonical_queue_kind_for_leaf(leaf, negative_high_risk_policy="require_human_confirmation") == entry["queue_kind"]


# Round-6 finding 2: a self-hashed but forged checkpoint (WRONG pdf_sha256)
# placed into a partial run with NO manifest yet is reused verbatim (zero
# LLM calls) and a manifest is published that cannot actually reproduce
# its own qa_run_id when checked against its own bound checkpoints.
# Two-document regression: one existing checkpoint is INVALID (wrong
# pdf_sha256), the other is MISSING entirely -- the invalid checkpoint
# must be detected before any paid call for the missing document.
def test_round6_finding2_forged_pdf_sha256_checkpoint_detected_before_missing_documents_llm_call(tmp_path, monkeypatch):
    chunk = _chunk(0, "The research question is clearly stated in the introduction.")
    monkeypatch.setattr(qa_module, "parse_pdf", lambda *a, **k: _parsed_document((chunk,)))
    call_count = {"n": 0}

    def _fake(**kw):
        call_count["n"] += 1
        return _all_items_response("yes", chunk.chunk_id, chunk.text)
    monkeypatch.setattr(qa_module, "run_openai_teaching_batch", _fake)

    cfg = _cfg(tmp_path)
    pdf_path_a = tmp_path / "rec-a.pdf"
    pdf_path_a.write_bytes(b"content A")
    pdf_path_b = tmp_path / "rec-b.pdf"
    pdf_path_b.write_bytes(b"content B")

    # Compute the qa_run_id this two-document run would actually use, and
    # construct its store directly (no manifest published yet).
    from slr_pipeline.extraction.hashing import file_sha256
    from slr_pipeline.extraction.parsing import DoclingParserOptions, installed_docling_version
    model_config = _model_config()
    real_pdf_sha256_a = file_sha256(pdf_path_a)
    real_pdf_sha256_b = file_sha256(pdf_path_b)
    qa_run_id = qa_module.compute_qa_run_id(
        documents=[("rec-a", real_pdf_sha256_a), ("rec-b", real_pdf_sha256_b)],
        checklist_id="teaching_appraisal_v1", checklist_version=qa_module.CHECKLIST_VERSION,
        checklist_content_hash=qa_module._checklist_content_hash("teaching_appraisal_v1"),
        roll_up_method_ids=qa_module._roll_up_method_ids_for("teaching_appraisal_v1"),
        prompt_template_version=qa_module.PROMPT_TEMPLATE_VERSION_TEACHING, model=model_config.model,
        model_config_id=model_config.model_config_id, reasoning_effort=model_config.reasoning_effort,
        max_completion_tokens=model_config.max_completion_tokens, expected_provider_model=model_config.expected_provider_model,
        parser_name="docling", parser_version=installed_docling_version(), parser_options=DoclingParserOptions(),
        max_batch_chars=cfg.quality_appraisal_max_batch_chars, evidence_verifier_version=qa_module.EVIDENCE_VERIFIER_VERSION,
        negative_high_risk_policy="require_human_confirmation", citation_metadata_hash=qa_module.sha256_hex({"rec-a": "Study rec-a", "rec-b": "Study rec-b"}),
    )
    store = qa_module.QACheckpointStore(output_dir=Path(cfg.output_dir), qa_run_id=qa_run_id)

    # Plant a FORGED but self-hashed checkpoint for rec-a: internally
    # consistent (its own content_hash matches its own fields), correctly
    # tagged qa_run_id/record_id, but claiming a WRONG pdf_sha256.
    forged_items = [
        AppraisalItemResponse(item_id=i["item_id"], status="yes", evidence=[AppraisalEvidence(chunk_id="c1", quotation="q", locator="p1")], rationale="r", decision_origin="ai", evidence_status="verified_from_chunk")
        for i in TEACHING_APPRAISAL_ITEMS
    ]
    forged_da = qa_module.DocumentAppraisal(record_id="rec-a", checklist_id="teaching_appraisal_v1", teaching=qa_module.TeachingAppraisal(items=forged_items))
    forged_checkpoint = qa_module.QAAppraisalCheckpoint.build(
        qa_run_id=qa_run_id, record_id="rec-a", pdf_sha256="0" * 64,  # WRONG -- does not match real_pdf_sha256_a
        checklist_id="teaching_appraisal_v1", checklist_version=qa_module.CHECKLIST_VERSION, document_appraisal=forged_da,
    )
    store.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    store.checkpoint_path("rec-a").write_text(json.dumps(forged_checkpoint.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    # rec-b has NO checkpoint at all -- would require a real (paid) LLM call.

    state = _state(cfg, [_record("rec-a", pdf_path_a), _record("rec-b", pdf_path_b)])
    call_count["n"] = 0
    with pytest.raises(qa_module.CheckpointIntegrityError, match="does not match this run's own current PDF content"):
        quality_appraisal_node(state)
    assert call_count["n"] == 0, "the forged rec-a checkpoint must be detected before any paid call for the missing rec-b document"
