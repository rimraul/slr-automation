from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slr_pipeline.ai_traceability import AIUseEventStore, compute_event_id
from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
from slr_pipeline.extraction.hashing import canonical_json_bytes
from slr_pipeline.extraction.resolution_suggestions import (
    load_attempt_records,
    render_review_packet,
    review_packet_json_bytes,
    review_packet_markdown_bytes,
    run_suggestion_engine,
    suggestions_index_bytes,
)
from slr_pipeline.extraction.resolution_journal import (
    ResolutionDecisionIntent,
    append_decision_entry,
)
from slr_pipeline.extraction.resolution_ids import safe_key
from slr_pipeline.extraction.resolution_schemas import (
    EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
    build_failed_call_record,
    build_interrupted_call_record,
    build_rejected_suggestion,
)
from slr_pipeline.nodes.reporting import (
    ReportingConsistencyError,
    SourceBundle,
    VerifiedComponent,
    _load_extraction_review_artifacts,
    _bundle_component_entries,
    _render_extraction_resolution_review_dict,
    _render_extraction_resolution_review_md,
)
from tests.extraction_resolution.test_review_packet import _build_bundle, _snapshot_from_bundle


class _Response:
    def __init__(self) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=json.dumps({
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
        })))]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class _Client:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: _Response()))


def _prepared_review(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=bundle.run_id)
    run_root = store.run_dir
    (run_root / "extraction_review_packet.json").write_bytes(review_packet_json_bytes(packet))
    (run_root / "extraction_review_packet.md").write_bytes(review_packet_markdown_bytes(packet))
    (run_root / "review_queue_snapshot.json").write_bytes(
        canonical_json_bytes(snapshot.model_dump(mode="json"))
    )
    suggestions_dir = run_root / "extraction_resolution_suggestions"
    event_store = AIUseEventStore(output_dir=tmp_path / "out", review_run_id="review-1")
    result = run_suggestion_engine(
        packet, extraction_run_id=bundle.run_id, review_run_id="review-1",
        suggestions_dir=suggestions_dir, event_store=event_store, client=_Client(),
        model="gpt-test", provider="openai", component_versions={"resolver": "1"},
        model_config_id="cfg-test", now=lambda: "2026-09-19T00:00:00Z",
    )
    assert all(item.status == "dispatched" for item in result.items)
    attempts = load_attempt_records(suggestions_dir)
    (run_root / "extraction_resolution_suggestions.json").write_bytes(
        suggestions_index_bytes(attempts, extraction_run_id=bundle.run_id)
    )
    events = event_store.load_all()
    artifacts = _load_extraction_review_artifacts(
        run_root=run_root.resolve(), seen=set(), bundle=bundle, schema=schema,
        events=events, run_id=bundle.run_id,
    )
    assert artifacts is not None
    source = SourceBundle(
        raw={},
        manifest=VerifiedComponent(store.manifest_path(), "a" * 64, bundle.manifest.manifest_hash),
        checkpoints=[], resolutions=None, confirmations=None,
        computed_content_hash="b" * 64, run_id=bundle.run_id, extraction_review=artifacts,
    )
    return run_root, bundle, schema, events, artifacts, source


def test_review_artifacts_verify_index_and_render_three_separate_groups(tmp_path):
    _run_root, _bundle, _schema, _events, artifacts, source = _prepared_review(tmp_path)
    assert len(artifacts.suggestions) == 2
    report = _render_extraction_resolution_review_dict(SimpleNamespace(extraction=source))
    assert report["fields"]
    assert set(report["fields"][0]) >= {
        "raw_candidates", "machine_recommendations", "final_applied_decision",
    }
    markdown = _render_extraction_resolution_review_md(SimpleNamespace(extraction=source))
    assert "### Raw candidates" in markdown
    assert "### Machine recommendation" in markdown
    assert "### Final applied decision" in markdown
    entries = _bundle_component_entries(
        entry_kind="extraction_component", locator_kind="extraction", bundle=source,
    )
    suggestion_entries = [entry for entry in entries if entry.component == "suggestion"]
    assert len(suggestion_entries) == 2
    assert all("/suggestion/" in entry.logical_path for entry in suggestion_entries)


def test_generated_index_mismatch_is_hard_failure(tmp_path):
    run_root, bundle, schema, events, _artifacts, _source = _prepared_review(tmp_path)
    index_path = run_root / "extraction_resolution_suggestions.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["attempts"] = []
    index_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ReportingConsistencyError, match="does not byte-match"):
        _load_extraction_review_artifacts(
            run_root=run_root.resolve(), seen=set(), bundle=bundle, schema=schema,
            events=events, run_id=bundle.run_id,
        )


def test_packet_markdown_must_rerender_from_packet_json(tmp_path):
    run_root, bundle, schema, events, _artifacts, _source = _prepared_review(tmp_path)
    (run_root / "extraction_review_packet.md").write_text("stale\n", encoding="utf-8")
    with pytest.raises(ReportingConsistencyError, match="does not re-render"):
        _load_extraction_review_artifacts(
            run_root=run_root.resolve(), seen=set(), bundle=bundle, schema=schema,
            events=events, run_id=bundle.run_id,
        )


def test_generated_index_represents_all_four_attempt_record_variants(tmp_path):
    run_root, bundle, _schema, _events, _artifacts, _source = _prepared_review(tmp_path)
    path0, validated = load_attempt_records(run_root / "extraction_resolution_suggestions")[0]
    key = safe_key(bundle.run_id, validated.record_id, validated.field_id)

    def event_id(attempt):
        return compute_event_id(
            review_run_id=validated.ai_use_event_draft.review_run_id,
            stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
            work_item_id=validated.ai_use_event_draft.work_item_id,
            attempt=attempt,
        )

    draft1 = validated.ai_use_event_draft.model_copy(update={"attempt": 1})
    rejected = build_rejected_suggestion(
        extraction_run_id=bundle.run_id, record_id=validated.record_id, field_id=validated.field_id,
        attempt=1, original_field_result_hash=validated.original_field_result_hash,
        ai_use_event_id=event_id(1), ai_use_event_draft=draft1,
        raw_response={"bad": True}, rejection_reasons=("invalid response",),
        created_at="2026-09-19T00:00:01Z",
    )
    draft2 = validated.ai_use_event_draft.model_copy(update={"attempt": 2})
    failed = build_failed_call_record(
        extraction_run_id=bundle.run_id, record_id=validated.record_id, field_id=validated.field_id,
        attempt=2, original_field_result_hash=validated.original_field_result_hash,
        ai_use_event_id=event_id(2), ai_use_event_draft=draft2,
        error_type="TimeoutError", error_message="timeout", failed_at="2026-09-19T00:00:02Z",
    )
    interrupted = build_interrupted_call_record(
        extraction_run_id=bundle.run_id, record_id=validated.record_id, field_id=validated.field_id,
        attempt=3, ai_use_event_id=event_id(3), review_run_id=draft1.review_run_id,
        work_item_id=draft1.work_item_id, reserved_at="2026-09-19T00:00:03Z",
        archived_at="2026-09-19T00:00:04Z",
    )
    attempts = (
        (path0, validated),
        (run_root / f"{key}__attempt-1.json", rejected),
        (run_root / f"{key}__attempt-2.json", failed),
        (run_root / f"{key}__attempt-3.interrupted_unknown.json", interrupted),
    )
    payload = json.loads(suggestions_index_bytes(attempts, extraction_run_id=bundle.run_id))
    assert {row["kind"] for row in payload["attempts"]} == {
        "validated_suggestion", "rejected_suggestion", "failed_call", "interrupted_call",
    }


def test_interrupted_attempt_rejects_an_event_that_was_actually_published(tmp_path):
    run_root, bundle, schema, events, _artifacts, _source = _prepared_review(tmp_path)
    suggestions_dir = run_root / "extraction_resolution_suggestions"
    path, validated = load_attempt_records(suggestions_dir)[0]
    interrupted = build_interrupted_call_record(
        extraction_run_id=bundle.run_id, record_id=validated.record_id, field_id=validated.field_id,
        attempt=validated.attempt, ai_use_event_id=validated.ai_use_event_id,
        review_run_id=validated.ai_use_event_draft.review_run_id,
        work_item_id=validated.ai_use_event_draft.work_item_id,
        reserved_at="2026-09-19T00:00:00Z", archived_at="2026-09-19T00:01:00Z",
    )
    path.unlink()
    interrupted_path = suggestions_dir / path.name.replace(".json", ".interrupted_unknown.json")
    interrupted_path.write_text(interrupted.model_dump_json(indent=2), encoding="utf-8")
    attempts = load_attempt_records(suggestions_dir)
    (run_root / "extraction_resolution_suggestions.json").write_bytes(
        suggestions_index_bytes(attempts, extraction_run_id=bundle.run_id)
    )
    with pytest.raises(ReportingConsistencyError, match="unexpectedly has a published AIUseEvent"):
        _load_extraction_review_artifacts(
            run_root=run_root.resolve(), seen=set(), bundle=bundle, schema=schema,
            events=events, run_id=bundle.run_id,
        )


def test_decision_journal_must_bind_the_exact_bundled_attempt_file(tmp_path):
    run_root, bundle, schema, events, _artifacts, _source = _prepared_review(tmp_path)
    _path, suggestion = load_attempt_records(
        run_root / "extraction_resolution_suggestions"
    )[0]
    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=bundle.run_id)
    append_decision_entry(
        store,
        ResolutionDecisionIntent(
            run_id=bundle.run_id,
            record_id=suggestion.record_id,
            field_id=suggestion.field_id,
            outcome="not_reported",
            reviewer="reviewer-1",
            note="Reviewed the machine recommendation.",
            resolved_at="2026-09-19T01:00:00Z",
            decision_origin="machine_recommended_human_approved",
            suggestion_id=suggestion.suggestion_id,
            self_hash=suggestion.self_hash,
            attempt_file_path="extraction_resolution_suggestions/wrong-attempt.json",
            model=suggestion.ai_use_event_draft.model,
            prompt_template_version=suggestion.ai_use_event_draft.prompt_template_version,
            call_event_id=suggestion.ai_use_event_id,
        ),
        status="prepared",
    )
    with pytest.raises(ReportingConsistencyError, match="does not bind a bundled suggestion attempt"):
        _load_extraction_review_artifacts(
            run_root=run_root.resolve(), seen=set(), bundle=bundle, schema=schema,
            events=events, run_id=bundle.run_id,
        )
