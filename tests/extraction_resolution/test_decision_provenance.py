"""Offline tests for section 3.6's resolution-decision transaction."""

from __future__ import annotations

import json
import contextlib
from pathlib import Path

import pytest

import slr_pipeline.nodes.extraction as extraction_module
from slr_pipeline.ai_traceability import compute_event_id
from slr_pipeline.extraction import verified_bundle
from slr_pipeline.extraction.resolution_ids import safe_key
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.resolution_journal import (
    ResolutionDecisionIntent,
    append_decision_entry,
    decision_journal_path,
    read_decision_journal,
)
from slr_pipeline.extraction.resolution_schemas import (
    AIUseEventDraft,
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    QueueSnapshot,
    QueueSnapshotEntry,
    build_validated_suggestion,
    compute_expected_work_item_id,
)
from slr_pipeline.extraction.resolution_suggestions import render_review_packet
from slr_pipeline.nodes.extraction import (
    _resolution_artifact_path,
    _resolution_artifact_lock,
    apply_extraction_resolution_decision,
    reconcile_resolution_journal,
    resolve_extraction_review_queue,
    verify_resolution_decisions,
)
from tests.nodes.test_extraction_node import (
    _VALID_TS,
    _cfg,
    _chunk,
    _found_response,
    _model_config,
    _parsed_document,
    _record,
    _schema,
    _state,
    _telemetry,
    engine_module,
)


def _state_and_packet(monkeypatch, tmp_path):
    chunk_a = _chunk(0, "We used a randomized controlled trial design.")
    chunk_b = _chunk(1, "This cohort study followed participants over time.")
    monkeypatch.setattr(engine_module, "parse_pdf", lambda *args, **kwargs: _parsed_document((chunk_a, chunk_b)))

    def fake_call(*, schema, batch_chunks, config):
        chunk = batch_chunks[0]
        if chunk.chunk_id == chunk_a.chunk_id:
            return _found_response(
                "study_design", "RCT", chunk_a.chunk_id, "randomized controlled trial",
            ), _telemetry()
        return _found_response(
            "study_design", "cohort study", chunk_b.chunk_id, "cohort study followed participants",
        ), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_call)
    pdf_path = tmp_path / "rec-a.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")
    cfg = _cfg(tmp_path, schema=_schema(), model_config=_model_config())
    cfg.extraction_max_batch_chars = 10
    state = _state(cfg, [_record("rec-a", pdf_path)])
    state.update(extraction_module.extraction_node(state))
    assert state["extraction_review_queue"][0]["status"] == "conflicting"
    store, bundle, schema, schema_hash = verified_bundle.load_and_verify_bundle(
        state, state["extraction_run_id"],
    )
    snapshot = QueueSnapshot(
        run_id=store.run_id,
        manifest_hash=bundle.manifest.manifest_hash,
        extraction_schema_hash=schema_hash,
        checkpoint_hashes=dict(bundle.manifest.checkpoint_hashes),
        queue=[QueueSnapshotEntry.model_validate(entry) for entry in state["extraction_review_queue"]],
    )
    return state, store, render_review_packet(bundle, schema, snapshot)


def _evidence_id(packet) -> str:
    assert packet.entries[0].evidence_index
    return next(iter(packet.entries[0].evidence_index))


def _human_intent(state, *, outcome="not_reported", packet=None) -> ResolutionDecisionIntent:
    evidence_ids = (_evidence_id(packet),) if outcome == "extracted" else ()
    return ResolutionDecisionIntent(
        run_id=state["extraction_run_id"], record_id="rec-a", field_id="study_design",
        outcome=outcome, final_value="RCT" if outcome == "extracted" else None,
        selected_evidence_ids=evidence_ids, reviewer="human-reviewer", note="checked full text",
        resolved_at=_VALID_TS, attested=True if outcome == "extracted" else None,
        decision_origin="human",
    )


def test_human_decision_writes_prepared_and_applied_and_matches_authority(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    update = apply_extraction_resolution_decision(
        state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
        reviewer="human-reviewer", note="checked full text", resolved_at=_VALID_TS,
        decision_origin="human",
    )
    assert update["extraction_review_queue"] == []
    entries = read_decision_journal(store)
    assert [entry.status for entry in entries] == ["prepared", "applied"]
    assert entries[0].decision_origin == "human"
    assert entries[0].suggestion_id is None
    verify_resolution_decisions(state, packet)


def test_extracted_decision_rehydrates_evidence_and_sets_human_attestation(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    evidence_id = _evidence_id(packet)
    update = apply_extraction_resolution_decision(
        state, packet, record_id="rec-a", field_id="study_design", outcome="extracted",
        final_value="RCT", selected_evidence_ids=(evidence_id,), reviewer="R", note="human checked",
        resolved_at=_VALID_TS, decision_origin="human",
    )
    resolved = update["extraction_review_resolutions"][0]
    source = packet.entries[0].evidence_index[evidence_id]
    assert resolved["value"] == "RCT"
    assert resolved["attested"] is True
    assert resolved["evidence"][0]["quotation_or_description"] == source.excerpt
    assert source.chunk_id in resolved["evidence"][0]["locator"]
    verify_resolution_decisions(state, packet)


def test_machine_only_is_rejected_before_any_journal_write(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="machine_only"):
        apply_extraction_resolution_decision(
            state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
            reviewer="R", note="n", resolved_at=_VALID_TS, decision_origin="machine_only",
        )
    assert not decision_journal_path(store).exists()


def test_not_reported_rejects_stray_value_or_evidence_before_write(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="cannot carry"):
        apply_extraction_resolution_decision(
            state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
            final_value="stray", selected_evidence_ids=(_evidence_id(packet),),
            reviewer="R", note="n", resolved_at=_VALID_TS, decision_origin="human",
        )
    assert not decision_journal_path(store).exists()


def test_forged_packet_evidence_is_rederived_and_rejected(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    evidence_id = _evidence_id(packet)
    original_entry = packet.entries[0]
    forged_index = dict(original_entry.evidence_index)
    forged_index[evidence_id] = forged_index[evidence_id].model_copy(update={"excerpt": "invented evidence"})
    forged_entry = original_entry.model_copy(update={"evidence_index": forged_index})
    forged_packet = packet.model_copy(update={"entries": (forged_entry,)})
    with pytest.raises(ValueError, match="indexes.*do not match"):
        apply_extraction_resolution_decision(
            state, forged_packet, record_id="rec-a", field_id="study_design", outcome="extracted",
            final_value="RCT", selected_evidence_ids=(evidence_id,), reviewer="R", note="n",
            resolved_at=_VALID_TS, decision_origin="human",
        )
    assert not decision_journal_path(store).exists()


def test_machine_recommended_decision_binds_immutable_attempt(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    evidence_id = _evidence_id(packet)
    work_item_id = compute_expected_work_item_id(
        extraction_run_id=store.run_id, record_id="rec-a", field_id="study_design",
    )
    draft = AIUseEventDraft(
        review_run_id="review-run-1", work_item_id=work_item_id, attempt=0,
        model="gpt-test", provider="openai", component_versions={"resolution_suggestions": "1"},
        task_purpose="resolve extraction ambiguity", model_config_id="cfg-test",
        prompt_template_version="resolution-v1", prompt_template_text="prompt",
        rendered_prompt_hash="a" * 64, input_artifact_refs=[],
        event_timestamp="2026-09-18T12:00:00Z",
    )
    event_id = compute_event_id(
        review_run_id=draft.review_run_id, stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
        work_item_id=work_item_id, attempt=0,
    )
    suggestion = build_validated_suggestion(
        ai_use_event_id=event_id, ai_use_event_draft=draft, extraction_run_id=store.run_id,
        record_id="rec-a", field_id="study_design", attempt=0, outcome="extracted",
        original_field_result_hash=sha256_hex(packet.entries[0].field_result.model_dump(mode="json")),
        value="RCT", typed_value="RCT", evidence_ids=(evidence_id,), rationale="best synthesis",
        confidence="medium", requires_human_review=True, created_at="2026-09-18T12:00:00Z",
    )
    suggestions_dir = store.run_dir / "extraction_resolution_suggestions"
    suggestions_dir.mkdir()
    path = suggestions_dir / f"{safe_key(store.run_id, 'rec-a', 'study_design')}__attempt-0.json"
    path.write_text(suggestion.model_dump_json(indent=2), encoding="utf-8")

    apply_extraction_resolution_decision(
        state, packet, record_id="rec-a", field_id="study_design", outcome="extracted",
        final_value="RCT", selected_evidence_ids=(evidence_id,), reviewer="R", note="approved",
        resolved_at=_VALID_TS, decision_origin="machine_recommended_human_approved",
        suggestion_file=path,
    )
    prepared = read_decision_journal(store)[0]
    assert prepared.suggestion_id == suggestion.suggestion_id
    assert prepared.self_hash == suggestion.self_hash
    assert prepared.call_event_id == suggestion.ai_use_event_id
    assert prepared.model == "gpt-test"
    assert prepared.attempt_file_path == f"extraction_resolution_suggestions/{path.name}"


def test_caught_resolver_failure_gets_failed_terminal_inside_transaction(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    monkeypatch.setattr(
        extraction_module, "_resolve_extraction_review_queue_locked",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad decision")),
    )
    with pytest.raises(ValueError, match="bad decision"):
        apply_extraction_resolution_decision(
            state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
            reviewer="R", note="n", resolved_at=_VALID_TS, decision_origin="human",
        )
    entries = read_decision_journal(store)
    assert [entry.status for entry in entries] == ["prepared", "failed"]
    assert "bad decision" in entries[-1].reason


def test_restart_reconciles_orphan_without_resolution_to_failed(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    original = extraction_module._resolve_extraction_review_queue_locked

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("hard stop after prepared append")

    monkeypatch.setattr(extraction_module, "_resolve_extraction_review_queue_locked", interrupted)
    with pytest.raises(KeyboardInterrupt):
        apply_extraction_resolution_decision(
            state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
            reviewer="human-reviewer", note="checked full text", resolved_at=_VALID_TS,
            decision_origin="human",
        )
    monkeypatch.setattr(extraction_module, "_resolve_extraction_review_queue_locked", original)
    assert [entry.status for entry in read_decision_journal(store)] == ["prepared"]
    reconcile_resolution_journal(state, packet)
    entries = read_decision_journal(store)
    assert [entry.status for entry in entries] == ["prepared", "failed"]
    assert entries[-1].reason == "interrupted, requires manual re-decision"


def test_restart_reconciles_orphan_with_exact_resolution_to_applied(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    intent = _human_intent(state)
    with _resolution_artifact_lock(store):
        append_decision_entry(store, intent, status="prepared")
    resolve_extraction_review_queue(state, {
        ("rec-a", "study_design"): {
            "outcome": "not_reported", "reviewer": "human-reviewer",
            "note": "checked full text", "resolved_at": _VALID_TS,
        }
    })
    reconcile_resolution_journal(state, packet)
    assert [entry.status for entry in read_decision_journal(store)] == ["prepared", "applied"]
    verify_resolution_decisions(state, packet)


def test_reconciliation_scan_and_append_hold_resolution_lock(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    with _resolution_artifact_lock(store):
        append_decision_entry(store, _human_intent(state), status="prepared")
    original_lock = extraction_module._resolution_artifact_lock
    observed = []

    @contextlib.contextmanager
    def observed_lock(target_store, *args, **kwargs):
        with original_lock(target_store, *args, **kwargs):
            observed.append(getattr(extraction_module._RESOLUTION_LOCK_HELD, "value", False))
            yield

    monkeypatch.setattr(extraction_module, "_resolution_artifact_lock", observed_lock)
    reconcile_resolution_journal(state, packet)
    assert observed == [True]
    assert [entry.status for entry in read_decision_journal(store)] == ["prepared", "failed"]


def test_prepared_append_and_resolver_core_share_the_same_lock(monkeypatch, tmp_path):
    state, _store, packet = _state_and_packet(monkeypatch, tmp_path)
    original_append = extraction_module.append_decision_entry
    original_core = extraction_module._resolve_extraction_review_queue_locked
    observed = []

    def checked_append(*args, **kwargs):
        observed.append(("append", getattr(extraction_module._RESOLUTION_LOCK_HELD, "value", False)))
        return original_append(*args, **kwargs)

    def checked_core(*args, **kwargs):
        observed.append(("core", getattr(extraction_module._RESOLUTION_LOCK_HELD, "value", False)))
        return original_core(*args, **kwargs)

    monkeypatch.setattr(extraction_module, "append_decision_entry", checked_append)
    monkeypatch.setattr(extraction_module, "_resolve_extraction_review_queue_locked", checked_core)
    apply_extraction_resolution_decision(
        state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
        reviewer="R", note="n", resolved_at=_VALID_TS, decision_origin="human",
    )
    assert observed == [("append", True), ("core", True), ("append", True)]


def test_verify_rejects_full_payload_mismatch_even_when_authority_rehashed(monkeypatch, tmp_path):
    state, store, packet = _state_and_packet(monkeypatch, tmp_path)
    apply_extraction_resolution_decision(
        state, packet, record_id="rec-a", field_id="study_design", outcome="not_reported",
        reviewer="R", note="original note", resolved_at=_VALID_TS, decision_origin="human",
    )
    path = _resolution_artifact_path(store)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["resolutions"][0]["note"] = "different but still schema-valid note"
    artifact["content_hash"] = sha256_hex({key: value for key, value in artifact.items() if key != "content_hash"})
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        verify_resolution_decisions(state, packet)
