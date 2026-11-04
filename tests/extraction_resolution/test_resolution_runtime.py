from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import slr_pipeline.extraction.resolution_runtime as runtime_module
from slr_pipeline.config import ReviewConfig
from slr_pipeline.cost_cap.ledger import CostLedger
from slr_pipeline.extraction.resolution_runtime import (
    ResolutionDecisionBatch,
    ResolutionDecisionRequest,
    apply_resolution_decision_batch,
    estimate_suggestion_cost,
    prepare_review_packet,
    run_paid_suggestion_pass,
)
from slr_pipeline.extraction.resolution_suggestions import load_attempt_records
from slr_pipeline.extraction.verified_bundle import bundle_review_fields
from slr_pipeline.nodes.extraction import resolve_extraction_review_queue
from slr_pipeline.state import Record
from tests.extraction_resolution.test_review_packet import _build_bundle, _model_config


def _state_and_config(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    output_dir = tmp_path / "review_output"
    output_dir.mkdir()
    shutil.move(str(tmp_path / "out"), str(output_dir / "extraction_runs"))
    cfg = ReviewConfig(
        title="Resolution runtime test",
        output_dir=str(output_dir),
        extraction_schema=schema.model_dump(mode="json"),
        extraction_model_config=asdict(_model_config()),
    )
    queue = [
        {
            "record_id": record_id,
            "field_id": field_id,
            "status": field.status.value,
            "field_result": field.model_dump(mode="json"),
        }
        for (record_id, field_id), field in bundle_review_fields(bundle.results).items()
    ]
    state = {
        "config": cfg.to_dict(),
        "review_run_id": "review-runtime-1",
        "extraction_run_id": bundle.run_id,
        "extraction_review_queue": queue,
        "full_text_records": [
            Record(id="rec-1", source="test", source_id="rec-1", title="Test record").to_dict()
        ],
    }
    state.update(resolve_extraction_review_queue(state, {}))
    return state, cfg


class _FakeClient:
    def __init__(self, captured):
        self._captured = captured
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self._captured.append((os.environ.get("OPENAI_BASE_URL"), kwargs))
        payload = {
            "outcome": "not_reported",
            "value": None,
            "evidence_ids": [],
            "candidate_ids_merged": [],
            "candidate_ids_retained": [],
            "candidate_ids_rejected": [],
            "rationale": "No single value is sufficiently supported.",
            "confidence": "low",
            "uncertainty": "Human adjudication remains required.",
            "requires_human_review": True,
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _seed_ledger(path, *, cap=7.5, committed=0.3216):
    ledger = CostLedger(path, cap_usd=cap)
    ledger.reserve("prior-call", committed + 0.1)
    ledger.commit(
        "prior-call", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
        cached_tokens=0, actual_usd=committed,
    )


def test_prepare_packet_is_immutable_and_byte_reproducible(tmp_path):
    state, _cfg = _state_and_config(tmp_path)
    first = prepare_review_packet(state)
    before = {
        path.name: path.read_bytes()
        for path in (
            first.run_dir / "review_queue_snapshot.json",
            first.run_dir / "extraction_review_packet.json",
            first.run_dir / "extraction_review_packet.md",
        )
    }
    second = prepare_review_packet(state)
    assert second.packet == first.packet
    assert {path.name: path.read_bytes() for path in (
        second.run_dir / "review_queue_snapshot.json",
        second.run_dir / "extraction_review_packet.json",
        second.run_dir / "extraction_review_packet.md",
    )} == before


def test_suggestion_pass_reuses_nonzero_ledger_and_proxy(tmp_path, monkeypatch):
    state, cfg = _state_and_config(tmp_path)
    prepared = prepare_review_packet(state)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ledger_path = workspace / "cost_cap_ledger.json"
    _seed_ledger(ledger_path)

    estimate = estimate_suggestion_cost(
        prepared.packet,
        suggestions_dir=prepared.run_dir / "extraction_resolution_suggestions",
        ledger_path=ledger_path,
        model="gpt-5.6-terra",
        max_completion_tokens=123,
    )
    assert estimate.already_committed_usd == pytest.approx(0.3216)
    assert estimate.remaining_before_calls_usd == pytest.approx(7.1784)
    assert estimate.eligible_calls == len(prepared.packet.entries)

    captured = []
    old_base_url = os.environ.get("OPENAI_BASE_URL")
    reconciled = []
    result = run_paid_suggestion_pass(
        state,
        prepared.packet,
        cfg=cfg,
        workspace=workspace,
        cap_usd=7.5,
        model="gpt-5.6-terra",
        model_config_id="cfg-1",
        max_completion_tokens=123,
        client_factory=lambda: _FakeClient(captured),
        reconcile_journal_fn=lambda actual_state, packet: reconciled.append((actual_state, packet)),
    )
    assert len(reconciled) == 1
    assert len(captured) == len(prepared.packet.entries)
    assert all(url and url.startswith("http://127.0.0.1:") for url, _ in captured)
    assert all(kwargs["max_completion_tokens"] == 123 for _, kwargs in captured)
    assert os.environ.get("OPENAI_BASE_URL") == old_base_url
    assert result.already_committed_usd == pytest.approx(0.3216)
    assert result.remaining_usd == pytest.approx(7.1784)
    assert (prepared.run_dir / "extraction_resolution_suggestions.json").is_file()
    assert (workspace / "cost_cap_call_log.jsonl").is_file()


def test_suggestion_pass_restores_proxy_environment_if_index_rebuild_fails(tmp_path, monkeypatch):
    state, cfg = _state_and_config(tmp_path)
    prepared = prepare_review_packet(state)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _seed_ledger(workspace / "cost_cap_ledger.json")
    prior = "http://prior.example/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", prior)

    def fail_index(_suggestions_dir):
        raise RuntimeError("index rebuild failed")

    monkeypatch.setattr(runtime_module, "load_attempt_records", fail_index)
    with pytest.raises(RuntimeError, match="index rebuild failed"):
        run_paid_suggestion_pass(
            state,
            prepared.packet,
            cfg=cfg,
            workspace=workspace,
            cap_usd=7.5,
            model="gpt-5.6-terra",
            model_config_id="cfg-1",
            max_completion_tokens=123,
            client_factory=lambda: _FakeClient([]),
            reconcile_journal_fn=lambda _state, _packet: None,
        )
    assert os.environ["OPENAI_BASE_URL"] == prior


def test_human_acceptance_is_audited_checkpointable_and_idempotent(tmp_path):
    state, cfg = _state_and_config(tmp_path)
    prepared = prepare_review_packet(state)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _seed_ledger(workspace / "cost_cap_ledger.json")
    run_paid_suggestion_pass(
        state,
        prepared.packet,
        cfg=cfg,
        workspace=workspace,
        cap_usd=7.5,
        model="gpt-5.6-terra",
        model_config_id="cfg-1",
        max_completion_tokens=123,
        client_factory=lambda: _FakeClient([]),
        reconcile_journal_fn=lambda _state, _packet: None,
    )
    _path, suggestion = load_attempt_records(
        prepared.run_dir / "extraction_resolution_suggestions"
    )[0]
    batch = ResolutionDecisionBatch(
        extraction_run_id=prepared.packet.run_id,
        decisions=(ResolutionDecisionRequest(
            record_id=suggestion.record_id,
            field_id=suggestion.field_id,
            decision_mode="accept_suggestion",
            suggestion_id=suggestion.suggestion_id,
            reviewer="human-reviewer",
            note="Accepted after checking the packet evidence.",
            resolved_at="2026-09-20T08:00:00Z",
        ),),
    )
    updates = []
    first = apply_resolution_decision_batch(
        state, prepared.packet, batch, on_state_update=updates.append,
    )
    assert first.applied == 1
    assert first.already_applied == 0
    assert first.remaining_queue == len(prepared.packet.entries) - 1
    assert len(updates) == 2  # durable-artifact projection, then this decision

    second = apply_resolution_decision_batch(first.state, prepared.packet, batch)
    assert second.applied == 0
    assert second.already_applied == 1
    assert second.remaining_queue == first.remaining_queue
