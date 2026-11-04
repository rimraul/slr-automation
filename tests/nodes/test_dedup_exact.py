"""Offline tests for dedup_node's `DedupConfig.mode` dispatch (Track 2 plan
§3.1/§3.2) -- in-memory synthetic `Record` objects only, never the
reference-run corpus fixtures (round-12 finding 3: the Track-2 reference
run's own dedup coverage is the normalised-title exact-match path only,
since every fixture record has doi=None; the DOI-exact path is exercised
here instead)."""

from __future__ import annotations

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore
from slr_pipeline.config import DedupConfig, ReviewConfig
from slr_pipeline.selection_ledger import SelectionDecisionStore
from slr_pipeline.state import Record, initial_state

import slr_pipeline.nodes.dedup as dedup_module
from slr_pipeline.nodes.dedup import dedup_node


def _config(tmp_path, mode: str = "exact") -> ReviewConfig:
    return ReviewConfig(
        title="Test review", output_dir=str(tmp_path / "outputs"), research_questions=["q"],
        dedup=DedupConfig(mode=mode),
    )


def _record(**overrides) -> Record:
    defaults = dict(id=f"rec-{overrides.get('source_id', '0')}", source="test", title="A study", abstract="abc")
    defaults.update(overrides)
    return Record(**defaults)


def _state(cfg: ReviewConfig, records: list[Record]) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["raw_records"] = [r.to_dict() for r in records]
    st["review_run_id"] = "dedup-exact-test-run"
    return st


def test_exact_mode_never_calls_embedding_pipeline(tmp_path, monkeypatch):
    def _fail(records, cfg):
        raise AssertionError("mode='exact' must never attempt the embedding pipeline")
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", _fail)

    cfg = _config(tmp_path, mode="exact")
    records = [_record(source_id="1")]
    dedup_node(_state(cfg, records))   # would raise via the monkeypatched stub if called


def test_exact_mode_merges_records_sharing_normalised_doi(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = _config(tmp_path, mode="exact")
    records = [
        _record(source_id="1", doi="10.1000/x", title="Study A"),
        _record(source_id="2", doi="10.1000/X", title="Study A duplicate title text differs"),
    ]
    state = _state(cfg, records)
    result = dedup_node(state)
    ids = {r["id"] for r in result["deduplicated_records"]}
    assert ids == {"rec-1"}


def test_exact_mode_merges_dx_doi_org_prefix_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = _config(tmp_path, mode="exact")
    records = [
        _record(source_id="1", doi="10.1000/x", title="Study A"),
        _record(source_id="2", doi="http://dx.doi.org/10.1000/x", title="A different title entirely"),
    ]
    state = _state(cfg, records)
    result = dedup_node(state)
    ids = {r["id"] for r in result["deduplicated_records"]}
    assert ids == {"rec-1"}


def test_exact_mode_keeps_non_matching_doi(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = _config(tmp_path, mode="exact")
    records = [
        _record(source_id="1", doi="10.1000/x", title="Study A"),
        _record(source_id="2", doi="10.1000/y", title="Study B, unrelated"),
    ]
    state = _state(cfg, records)
    result = dedup_node(state)
    ids = {r["id"] for r in result["deduplicated_records"]}
    assert ids == {"rec-1", "rec-2"}


def test_exact_mode_merges_doi_less_duplicate_by_normalised_title(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = _config(tmp_path, mode="exact")
    records = [
        _record(source_id="1", title="A Study of Cats"),
        _record(source_id="2", title="  a study   OF cats  "),
    ]
    state = _state(cfg, records)
    result = dedup_node(state)
    ids = {r["id"] for r in result["deduplicated_records"]}
    assert ids == {"rec-1"}


def test_exact_mode_publishes_one_dedup_exact_event(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = _config(tmp_path, mode="exact")
    records = [_record(source_id="1"), _record(source_id="2", title="Different paper")]
    state = _state(cfg, records)
    dedup_node(state)

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id="dedup-exact-test-run")
    events = event_store.load_all()
    assert len(events) == 1
    event = events[0]
    assert event.stage == "dedup_exact"
    assert event.tool_kind == "non_llm_tool"
    assert event.model == "doi_title_exact_match"
    assert event.provider == "slr_pipeline"
    assert event.model_config_id == "dedup_exact_v1"
    assert event.call_status == "success"
    assert event.failure_detail is None
    assert event.prompt_template_version is None
    assert event.prompt_template_text is None
    assert "python" in event.component_versions


def test_exact_mode_ledger_reason_and_decision_maker(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    cfg = _config(tmp_path, mode="exact")
    records = [_record(source_id="1"), _record(source_id="2", doi="10.1000/x", title="Study A")]
    state = _state(cfg, records)
    result = dedup_node(state)
    assert result["automation_origin"]["dedup"] == "rule_based_automation"

    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id="dedup-exact-test-run")
    tails = ledger_store.project_current_tails()
    for record in records:
        assert tails[(record.id, "dedup")].reason == "exact DOI/normalised-title deduplication"
        assert tails[(record.id, "dedup")].decision_maker == "rule_based_automation"


def test_semantic_mode_raises_when_pipeline_unavailable(tmp_path, monkeypatch):
    def _raise(records, cfg):
        raise ImportError("sentence-transformers not installed")
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", _raise)

    cfg = _config(tmp_path, mode="semantic")
    records = [_record(source_id="1")]
    state = _state(cfg, records)
    with pytest.raises(ImportError):
        dedup_node(state)


def test_semantic_mode_success_publishes_dedup_embedding_event(tmp_path, monkeypatch):
    unique = [_record(source_id="1")]
    monkeypatch.setattr(dedup_module, "_dedup_with_pipeline", lambda records, cfg: unique)

    cfg = _config(tmp_path, mode="semantic")
    records = [_record(source_id="1")]
    state = _state(cfg, records)
    dedup_node(state)

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id="dedup-exact-test-run")
    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].stage == "dedup_embedding"
    assert events[0].call_status == "success"


def test_auto_mode_default_is_unchanged_production_behavior(tmp_path):
    cfg = ReviewConfig(title="Test review", output_dir=str(tmp_path / "outputs"), research_questions=["q"])
    assert cfg.dedup.mode == "auto"
