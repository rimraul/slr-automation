"""Offline tests for slr_pipeline.nodes.screening.screening_node -- Stage 3
node instrumentation (the Stage 3 node-instrumentation design record): checkpoint
reuse across interrupt()/resume (INV-02), double-checked-locking (INV-10),
record-identity/policy binding in work_item_id (INV-12), the atomic human-
resolution precondition (INV-03), and automated-mode disposition
normalization (INV-09)."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore
from slr_pipeline.config import ReviewConfig
from slr_pipeline.selection_ledger import SelectionDecisionStore, SelectionLedgerPreconditionError, emit_selection_decision
from slr_pipeline.state import Record, initial_state
from slr_pipeline.human_resolution import StaleReviewError

import slr_pipeline.nodes.screening as screening_module
from slr_pipeline.nodes.screening import ScreeningBatchResultStore, _get_or_compute_batch_result, screening_node


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content, model="gpt-5.4-mini-2026-01-01"):
        self.choices = [_FakeChoice(content)]
        self.model = model


def _config(tmp_path) -> ReviewConfig:
    return ReviewConfig(
        title="Test review", output_dir=str(tmp_path / "outputs"), research_questions=["q"],
        inclusion_criteria=["c1"], exclusion_criteria=["c2"],
    )


def _record(**overrides) -> Record:
    defaults = dict(id=f"rec-{overrides.get('source_id', '0')}", source="test", title="A study", abstract="abc")
    defaults.update(overrides)
    return Record(**defaults)


def _state(cfg: ReviewConfig, records: list[Record]) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["deduplicated_records"] = [r.to_dict() for r in records]
    st["review_run_id"] = "screen-test-run"
    return st


def _install_fake_client(create_fn):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = create_fn
    fake_client.base_url = "https://api.openai.com/v1/"
    screening_module.OpenAI = lambda **kw: fake_client
    return fake_client


def _all_decision_response(decision: str, n: int, confidence: float = 0.9):
    def _create(*, model, response_format, messages, temperature):
        items = [{"idx": i + 1, "decision": decision, "reason": "r", "relevant_topics": [], "confidence": confidence} for i in range(n)]
        return _FakeResp(json.dumps({"items": items}))
    return _create


def _stores(cfg: ReviewConfig, review_run_id: str):
    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    batch_store = ScreeningBatchResultStore(output_dir=output_dir, review_run_id=review_run_id, event_store=event_store)
    return event_store, ledger_store, batch_store


# ---------------------------------------------------------------------------
# INV-09: automated-mode disposition normalization
# ---------------------------------------------------------------------------

def test_automated_mode_borderline_collapses_to_include_in_ledger_not_raw_label(tmp_path):
    fake_client = _install_fake_client(_all_decision_response("borderline", 1, confidence=0.3))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0   # automated mode
    records = [_record(source_id="1")]
    result = screening_node(_state(cfg, records))

    assert len(result["included_records"]) == 1
    updated = Record.from_dict(result["included_records"][0])
    assert updated.screening_decision == "borderline"   # raw label preserved on the record itself

    _, ledger_store, _ = _stores(cfg, "screen-test-run")
    tail = ledger_store.project_current_tails()[("rec-1", "title_abstract_screening")]
    assert tail.decision == "include"   # ledger records the OPERATIONAL final value
    assert "auto-included" in tail.reason


def test_non_automated_mode_borderline_keeps_raw_label_pending_human(tmp_path):
    _install_fake_client(_all_decision_response("borderline", 1, confidence=0.3))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.5
    cfg.screening.auto_include_threshold = 0.8
    records = [_record(source_id="1")]

    def _fake_interrupt(payload):
        return [{"record_id": "rec-1", "decision": "include", "reviewer": "alice", "note": "ok", "resolved_at": "2026-08-24T00:00:00Z"}]
    screening_module.interrupt = _fake_interrupt

    screening_node(_state(cfg, records))
    _, ledger_store, _ = _stores(cfg, "screen-test-run")
    chain = ledger_store.load_key(record_id="rec-1", stage="title_abstract_screening")
    ai_entry = [r for r in chain if r.decision_maker == "ai"][0]
    assert ai_entry.decision == "borderline"   # no final disposition existed yet without the human step


# ---------------------------------------------------------------------------
# INV-02: checkpoint reuse across a simulated interrupt()/resume
# ---------------------------------------------------------------------------

def test_identical_resume_reuses_checkpoint_no_new_provider_call_or_event(tmp_path):
    fake_client = _install_fake_client(_all_decision_response("include", 2))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    records = [_record(source_id="1"), _record(source_id="2")]
    state = _state(cfg, records)

    screening_node(state)
    calls_after_first = fake_client.chat.completions.create.call_count
    screening_node(state)   # simulated resume: identical content
    calls_after_second = fake_client.chat.completions.create.call_count

    assert calls_after_first == 1
    assert calls_after_second == 1   # no new physical call

    event_store, _, batch_store = _stores(cfg, "screen-test-run")
    assert len(event_store.load_all()) == 1


def test_changed_batch_content_creates_a_new_checkpoint_and_fresh_review(tmp_path):
    call_log = []

    def _create(*, model, response_format, messages, temperature):
        call_log.append(messages)
        items = [{"idx": 1, "decision": "include", "reason": "r", "relevant_topics": [], "confidence": 0.9}]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    state1 = _state(cfg, [_record(source_id="1", title="First title")])
    screening_node(state1)
    state2 = _state(cfg, [_record(source_id="1", title="A genuinely different title")])
    screening_node(state2)

    assert len(call_log) == 2   # two genuinely different batch fingerprints -> two real calls
    event_store, _, _ = _stores(cfg, "screen-test-run")
    assert len(event_store.load_all()) == 2


# ---------------------------------------------------------------------------
# INV-10/F15: double-checked locking around the checkpoint cache
# ---------------------------------------------------------------------------

def test_concurrent_identical_batch_one_provider_call_one_event_one_checkpoint(tmp_path):
    call_count = {"n": 0}
    call_lock = threading.Lock()

    def _create(*, model, response_format, messages, temperature):
        with call_lock:
            call_count["n"] += 1
        time.sleep(0.1)   # widen the race window
        items = [{"idx": 1, "decision": "include", "reason": "r", "relevant_topics": [], "confidence": 0.9}]
        return _FakeResp(json.dumps({"items": items}))

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create
    fake_client.base_url = "https://api.openai.com/v1/"

    cfg = _config(tmp_path)
    event_store, _, batch_store = _stores(cfg, "screen-test-run")
    batch = [_record(source_id="1")]

    results = []
    errors = []

    def _worker():
        try:
            r = _get_or_compute_batch_result(
                batch=batch, client=fake_client, model="gpt-5.4-mini",
                inclusion_criteria=cfg.inclusion_criteria, exclusion_criteria=cfg.exclusion_criteria,
                research_questions=cfg.research_questions, automated=True, auto_include_threshold=0.8,
                uncertain_threshold=0.0, event_store=event_store, batch_store=batch_store,
                review_run_id="screen-test-run",
            )
            results.append(r)
        except Exception as e:   # pragma: no cover - surfaced via errors list
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    assert call_count["n"] == 1
    assert len(event_store.load_all()) == 1
    assert len({r.artifact_id for r in results}) == 1


# ---------------------------------------------------------------------------
# INV-12: record identity and policy binding in work_item_id
# ---------------------------------------------------------------------------

def test_two_records_with_byte_identical_prompt_text_get_distinct_work_items(tmp_path):
    """Two different records that happen to render an identical prompt
    (same title/abstract) must not collide -- record identity is bound
    into the fingerprint explicitly, not left implicit in the rendered
    text."""
    _install_fake_client(_all_decision_response("include", 1))
    cfg = _config(tmp_path)
    event_store, _, batch_store = _stores(cfg, "screen-test-run")
    client = screening_module.OpenAI()

    same_title_a = _record(id="rec-a", source_id="a", title="Identical title", abstract="Identical abstract")
    same_title_b = _record(id="rec-b", source_id="b", title="Identical title", abstract="Identical abstract")

    r1 = _get_or_compute_batch_result(
        batch=[same_title_a], client=client, model="gpt-5.4-mini",
        inclusion_criteria=cfg.inclusion_criteria, exclusion_criteria=cfg.exclusion_criteria,
        research_questions=cfg.research_questions, automated=True, auto_include_threshold=0.8,
        uncertain_threshold=0.0, event_store=event_store, batch_store=batch_store,
        review_run_id="screen-test-run",
    )
    r2 = _get_or_compute_batch_result(
        batch=[same_title_b], client=client, model="gpt-5.4-mini",
        inclusion_criteria=cfg.inclusion_criteria, exclusion_criteria=cfg.exclusion_criteria,
        research_questions=cfg.research_questions, automated=True, auto_include_threshold=0.8,
        uncertain_threshold=0.0, event_store=event_store, batch_store=batch_store,
        review_run_id="screen-test-run",
    )
    assert r1.artifact_id != r2.artifact_id
    assert r1.record_ids == ["rec-a"]
    assert r2.record_ids == ["rec-b"]
    assert len(event_store.load_all()) == 2


def test_policy_change_creates_new_checkpoint_not_stale_reuse(tmp_path):
    """Same batch content, but automated/threshold policy differs -- must
    not reuse a cached checkpoint whose own already-derived disposition
    would now be wrong under the new policy."""
    _install_fake_client(_all_decision_response("borderline", 1, confidence=0.3))
    cfg = _config(tmp_path)
    event_store, _, batch_store = _stores(cfg, "screen-test-run")
    client = screening_module.OpenAI()
    batch = [_record(source_id="1")]

    kwargs = dict(
        batch=batch, client=client, model="gpt-5.4-mini",
        inclusion_criteria=cfg.inclusion_criteria, exclusion_criteria=cfg.exclusion_criteria,
        research_questions=cfg.research_questions, auto_include_threshold=0.8,
        event_store=event_store, batch_store=batch_store, review_run_id="screen-test-run",
    )
    r_automated = _get_or_compute_batch_result(automated=True, uncertain_threshold=0.0, **kwargs)
    r_manual = _get_or_compute_batch_result(automated=False, uncertain_threshold=0.5, **kwargs)

    assert r_automated.artifact_id != r_manual.artifact_id
    assert len(event_store.load_all()) == 2


# ---------------------------------------------------------------------------
# F23: malformed screening batch response with a real fallback still
# retains resp.model.
# ---------------------------------------------------------------------------

def test_malformed_batch_response_heuristic_fallback_retains_response_model(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        return _FakeResp("not valid json{{{", model="gpt-5.4-mini-parsefail")
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    records = [_record(source_id="1")]
    screening_node(_state(cfg, records))

    event_store, _, _ = _stores(cfg, "screen-test-run")
    events = event_store.load_all()
    assert len(events) == 1
    assert events[0].call_status == "heuristic_fallback"
    assert events[0].component_versions == {"gpt-5.4-mini": "gpt-5.4-mini-parsefail"}
    assert events[0].output_artifact_refs != []


# ---------------------------------------------------------------------------
# INV-03: human resolution -- baseline capture, precondition, staleness
# ---------------------------------------------------------------------------

def test_human_resolution_supersedes_ai_baseline_two_ledger_entries(tmp_path):
    _install_fake_client(_all_decision_response("borderline", 1, confidence=0.3))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.5
    records = [_record(source_id="1")]

    def _fake_interrupt(payload):
        rec_id = payload["records_for_review"][0]["record"]["id"]
        return [{"record_id": rec_id, "decision": "exclude", "reviewer": "bob", "note": "not relevant", "resolved_at": "2026-08-24T00:00:00Z"}]
    screening_module.interrupt = _fake_interrupt

    result = screening_node(_state(cfg, records))
    assert result["screening_review_resolutions"][0]["decision"] == "exclude"
    assert result["oversight_mode"]["title_abstract_screening"] == "human_resolved"

    _, ledger_store, _ = _stores(cfg, "screen-test-run")
    chain = ledger_store.load_key(record_id="rec-1", stage="title_abstract_screening")
    assert len(chain) == 2
    tail = ledger_store.project_current_tails()[("rec-1", "title_abstract_screening")]
    assert tail.decision_maker == "human"
    assert tail.decision == "exclude"


def test_malformed_human_submission_missing_record_raises_before_any_side_effect(tmp_path):
    _install_fake_client(_all_decision_response("borderline", 2, confidence=0.3))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.5
    records = [_record(source_id="1"), _record(source_id="2")]

    def _fake_interrupt(payload):
        # only resolves ONE of the two queued records -- incomplete submission
        rec_id = payload["records_for_review"][0]["record"]["id"]
        return [{"record_id": rec_id, "decision": "include", "reviewer": "alice", "note": "ok", "resolved_at": "2026-08-24T00:00:00Z"}]
    screening_module.interrupt = _fake_interrupt

    with pytest.raises(ValueError):
        screening_node(_state(cfg, records))

    # zero human-resolution writes -- neither record has a "human"
    # decision_maker entry (the ai baselines from the batch computation
    # that ran earlier in this same execution ARE expected to exist).
    _, ledger_store, _ = _stores(cfg, "screen-test-run")
    for rec_id in ("rec-1", "rec-2"):
        chain = ledger_store.load_key(record_id=rec_id, stage="title_abstract_screening")
        assert all(r.decision_maker != "human" for r in chain)


def test_stale_human_resolution_rejected_when_tail_moved_before_apply(tmp_path):
    """A competing ai decision supersedes the record's tail AFTER the
    human queue entry captured its own reviewed_baseline_decision_id, but
    BEFORE the human resolution is actually applied -- staleness must be
    caught and propagate, never be silently downgraded. This injects the
    race before screening_node's own preflight check runs, so the fast
    preflight catches it here (StaleReviewError); emit_selection_decision()'s
    own atomic, in-lock precondition (SelectionLedgerPreconditionError) is
    the authoritative guarantee for a race the preflight itself cannot see
    -- covered directly, with real concurrent threads, in
    test_selection_ledger.py::test_concurrent_tail_change_between_preflight_and_emission_raises."""
    _install_fake_client(_all_decision_response("borderline", 1, confidence=0.3))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.5
    records = [_record(source_id="1")]
    _, ledger_store, _ = _stores(cfg, "screen-test-run")

    def _fake_interrupt(payload):
        rec_id = payload["records_for_review"][0]["record"]["id"]
        baseline = payload["records_for_review"][0]["reviewed_baseline_decision_id"]
        # Simulate a race: something else supersedes the baseline right
        # here, between the queue being built and the human resolution
        # being applied.
        emit_selection_decision(
            store=ledger_store, record_id=rec_id, stage="title_abstract_screening",
            decision="exclude", decision_maker="ai", source_event_id="race-evt-1",
        )
        return [{"record_id": rec_id, "decision": "include", "reviewer": "alice", "note": "ok", "resolved_at": "2026-08-24T00:00:00Z"}]
    screening_module.interrupt = _fake_interrupt

    with pytest.raises((StaleReviewError, SelectionLedgerPreconditionError)):
        screening_node(_state(cfg, records))


# ---------------------------------------------------------------------------
# Real StateGraph interrupt()/resume -- not just two direct screening_node()
# calls (codex review: a real LangGraph pause/resume cycle is what INV-02's
# checkpoint-reuse guarantee is actually meant to survive).
# ---------------------------------------------------------------------------

def test_real_langgraph_interrupt_resume_reuses_checkpoint_and_chains_correctly(tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt as real_interrupt

    from slr_pipeline.state import ReviewState

    screening_module.interrupt = real_interrupt   # undo any prior test's monkeypatch

    call_count = {"n": 0}

    def _create(*, model, response_format, messages, temperature):
        call_count["n"] += 1
        items = [{"idx": 1, "decision": "borderline", "reason": "r", "relevant_topics": [], "confidence": 0.3}]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)

    builder = StateGraph(ReviewState)
    builder.add_node("screening", screening_node)
    builder.add_edge(START, "screening")
    builder.add_edge("screening", END)
    graph = builder.compile(checkpointer=MemorySaver())

    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.5
    cfg.screening.auto_include_threshold = 0.8
    records = [_record(source_id="1")]
    state = _state(cfg, records)
    config = {"configurable": {"thread_id": "screen-test-run"}}

    first = graph.invoke(state, config)
    assert "__interrupt__" in first
    assert call_count["n"] == 1

    resume_payload = [{
        "record_id": "rec-1", "decision": "include", "reviewer": "alice",
        "note": "manual check", "resolved_at": "2026-08-24T00:00:00Z",
    }]
    second = graph.invoke(Command(resume=resume_payload), config)
    assert "__interrupt__" not in second
    assert [r["id"] for r in second["included_records"]] == ["rec-1"]

    # The whole node body re-executes on resume (LangGraph's own
    # semantics), but the checkpoint means the batch call itself is NOT
    # repeated -- exactly one provider call total across both invocations.
    assert call_count["n"] == 1

    event_store, ledger_store, batch_store = _stores(cfg, "screen-test-run")
    assert len(event_store.load_all()) == 1
    tail = ledger_store.project_current_tails()[("rec-1", "title_abstract_screening")]
    assert tail.decision_maker == "human"
    assert tail.decision == "include"
    chain = ledger_store.load_key(record_id="rec-1", stage="title_abstract_screening")
    assert len(chain) == 2   # ai baseline -> human, connected, not forked


# ---------------------------------------------------------------------------
# ScreeningBatchResultStore.publish() cross-verifies a FRESH write, not
# only a collision (codex review, finding 2).
# ---------------------------------------------------------------------------

def test_fresh_checkpoint_publish_rejects_wrong_producing_event(tmp_path):
    from slr_pipeline.ai_traceability import build_ai_use_event
    from slr_pipeline.nodes.screening import ScreeningBatchResultIntegrityError, ScreeningResponse, build_screening_batch_result

    cfg = _config(tmp_path)
    event_store, _, batch_store = _stores(cfg, "screen-test-run")

    # A REAL, validly-published event -- but for an UNRELATED work item.
    unrelated_event = build_ai_use_event(
        review_run_id="screen-test-run", stage="title_abstract_screening", work_item_id="unrelated:xyz",
        attempt=0, tool_kind="llm", model="gpt-5.4-mini", provider="openai",
        component_versions={"gpt-5.4-mini": "x"}, task_purpose="unrelated",
        model_config_id="title_abstract_screening_batch_v1",
        prompt_template_version="v1", prompt_template_text="t", prompt_parameters={},
        rendered_prompt_hash="h", input_artifact_refs=[], output_artifact_refs=["sha256:unrelated"],
        call_status="success",
    )
    event_store.publish(unrelated_event)

    responses = [ScreeningResponse(decision="include", reason="r", relevant_topics=[], confidence=0.9)]
    bad_result = build_screening_batch_result(
        review_run_id="screen-test-run", stage="title_abstract_screening",
        batch_fingerprint="deadbeef" * 8, record_ids=["rec-1"],
        producing_event_id=unrelated_event.event_id,   # references a REAL but UNRELATED event
        call_status="success", responses=responses,
    )
    with pytest.raises(ScreeningBatchResultIntegrityError):
        batch_store.publish(bad_result)
    # And it must never have been persisted.
    assert batch_store.artifact_path(bad_result.artifact_id).exists() is False


# ---------------------------------------------------------------------------
# Semantic response validation (codex review, finding 1): a schema-shaped
# but semantically-invalid response must fall through to the heuristic
# fallback, never be accepted as "success".
# ---------------------------------------------------------------------------

def test_missing_decision_field_falls_back_to_heuristic(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        items = [{"idx": 1, "reason": "r", "relevant_topics": [], "confidence": 0.9}]   # no "decision"
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    screening_node(_state(cfg, [_record(source_id="1")]))
    event_store, _, _ = _stores(cfg, "screen-test-run")
    assert event_store.load_all()[0].call_status == "heuristic_fallback"


def test_out_of_range_confidence_falls_back_to_heuristic(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        items = [{"idx": 1, "decision": "include", "reason": "r", "relevant_topics": [], "confidence": 1.5}]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    screening_node(_state(cfg, [_record(source_id="1")]))
    event_store, _, _ = _stores(cfg, "screen-test-run")
    assert event_store.load_all()[0].call_status == "heuristic_fallback"


def test_reordered_response_indices_assigned_to_correct_records(tmp_path):
    """The response array's own position must never be trusted for
    record assignment -- only "idx"."""
    def _create(*, model, response_format, messages, temperature):
        # Deliberately reversed order relative to idx.
        items = [
            {"idx": 2, "decision": "exclude", "reason": "r2", "relevant_topics": [], "confidence": 0.9},
            {"idx": 1, "decision": "include", "reason": "r1", "relevant_topics": [], "confidence": 0.9},
        ]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    records = [_record(source_id="1"), _record(source_id="2")]
    result = screening_node(_state(cfg, records))
    included_ids = {r["id"] for r in result["included_records"]}
    assert included_ids == {"rec-1"}   # idx=1 (include) -> rec-1, idx=2 (exclude) -> rec-2, regardless of array order


def test_duplicate_response_indices_falls_back_to_heuristic(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        items = [
            {"idx": 1, "decision": "include", "reason": "r", "relevant_topics": [], "confidence": 0.9},
            {"idx": 1, "decision": "exclude", "reason": "r", "relevant_topics": [], "confidence": 0.9},
        ]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    screening_node(_state(cfg, [_record(source_id="1"), _record(source_id="2")]))
    event_store, _, _ = _stores(cfg, "screen-test-run")
    assert event_store.load_all()[0].call_status == "heuristic_fallback"


# ---------------------------------------------------------------------------
# Stage-dictionary preservation (codex review, finding 8): dedup then
# screening in sequence must not clobber each other's oversight_mode/
# automation_origin keys.
# ---------------------------------------------------------------------------

def test_dedup_then_screening_preserve_both_stage_keys(tmp_path):
    import slr_pipeline.nodes.dedup as dedup_module

    monkeypatch_pipeline = lambda records, cfg: records   # no-op "unique everything"
    dedup_module._dedup_with_pipeline = monkeypatch_pipeline

    _install_fake_client(_all_decision_response("include", 1))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    records = [_record(source_id="1")]
    state = _state(cfg, records)
    state["raw_records"] = [r.to_dict() for r in records]

    dedup_result = dedup_module.dedup_node(state)
    state.update(dedup_result)
    state["deduplicated_records"] = dedup_result["deduplicated_records"]

    screening_result = screening_node(state)

    assert "dedup" in screening_result["oversight_mode"]
    assert "dedup" in screening_result["automation_origin"]
    assert "title_abstract_screening" in screening_result["oversight_mode"]
    assert "title_abstract_screening" in screening_result["automation_origin"]


# ---------------------------------------------------------------------------
# INV-05: a store-integrity error inside a node call site propagates,
# never caught by the surrounding LLM-call except and downgraded.
# ---------------------------------------------------------------------------

def test_forced_event_store_error_propagates_out_of_screening_node(tmp_path):
    from slr_pipeline.ai_traceability import AIEventIntegrityError

    _install_fake_client(_all_decision_response("include", 1))
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0

    def _boom(self, event):
        raise AIEventIntegrityError("simulated store corruption")
    from slr_pipeline.ai_traceability import AIUseEventStore
    original_publish = AIUseEventStore.publish
    AIUseEventStore.publish = _boom
    try:
        with pytest.raises(AIEventIntegrityError):
            screening_node(_state(cfg, [_record(source_id="1")]))
    finally:
        AIUseEventStore.publish = original_publish


# ---------------------------------------------------------------------------
# codex review (second round): whitespace-only reason and boolean idx.
# ---------------------------------------------------------------------------

def test_whitespace_only_reason_falls_back_to_heuristic(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        items = [{"idx": 1, "decision": "include", "reason": "   ", "relevant_topics": [], "confidence": 0.9}]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    screening_node(_state(cfg, [_record(source_id="1")]))
    event_store, _, _ = _stores(cfg, "screen-test-run")
    assert event_store.load_all()[0].call_status == "heuristic_fallback"


def test_boolean_idx_falls_back_to_heuristic(tmp_path):
    """Python's bool is an int subclass -- idx=true must never be accepted
    as if it were the integer 1."""
    def _create(*, model, response_format, messages, temperature):
        items = [{"idx": True, "decision": "include", "reason": "r", "relevant_topics": [], "confidence": 0.9}]
        return _FakeResp(json.dumps({"items": items}))
    _install_fake_client(_create)
    cfg = _config(tmp_path)
    cfg.screening.uncertain_threshold = 0.0
    screening_node(_state(cfg, [_record(source_id="1")]))
    event_store, _, _ = _stores(cfg, "screen-test-run")
    assert event_store.load_all()[0].call_status == "heuristic_fallback"


# ---------------------------------------------------------------------------
# Track 2 plan §4.4: effective_batch_size()/effective_workers().
# ---------------------------------------------------------------------------

def test_effective_batch_size_falls_back_to_env_constant_when_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(screening_module, "SCREENING_BATCH_SIZE", 7)
    cfg = _config(tmp_path)
    assert cfg.screening.batch_size is None
    assert screening_module.effective_batch_size(cfg) == 7


def test_effective_batch_size_uses_explicit_override_when_set(tmp_path, monkeypatch):
    monkeypatch.setattr(screening_module, "SCREENING_BATCH_SIZE", 7)
    cfg = _config(tmp_path)
    cfg.screening.batch_size = 3
    assert screening_module.effective_batch_size(cfg) == 3


def test_effective_workers_falls_back_to_env_constant_when_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(screening_module, "SCREENING_WORKERS", 4)
    cfg = _config(tmp_path)
    assert cfg.screening.workers is None
    assert screening_module.effective_workers(cfg) == 4


def test_effective_workers_uses_explicit_override_when_set(tmp_path, monkeypatch):
    monkeypatch.setattr(screening_module, "SCREENING_WORKERS", 4)
    cfg = _config(tmp_path)
    cfg.screening.workers = 2
    assert screening_module.effective_workers(cfg) == 2


# ---------------------------------------------------------------------------
# client_factory (Workstream C Checkpoint 3: additive, backward-compatible
# retry-policy injection seam)
# ---------------------------------------------------------------------------

def test_screening_node_default_client_factory_behavior_is_unchanged(tmp_path):
    # No client_factory passed -- LangGraph's own calling convention
    # (screening_node(state)) must still work exactly as before.
    fake_client = _install_fake_client(_all_decision_response("include", 1))
    cfg = _config(tmp_path)
    records = [_record(source_id="1")]
    result = screening_node(_state(cfg, records))
    assert len(result["included_records"]) == 1
    fake_client.chat.completions.create.assert_called()


def test_screening_node_uses_the_injected_client_factory_instead_of_the_default(tmp_path):
    injected_client = MagicMock()
    injected_client.chat.completions.create.side_effect = _all_decision_response("include", 1)
    injected_client.base_url = "https://api.openai.com/v1/"
    calls = {"factory": 0}

    def factory():
        calls["factory"] += 1
        return injected_client

    default_client = _install_fake_client(_all_decision_response("exclude", 1))
    cfg = _config(tmp_path)
    records = [_record(source_id="1")]
    result = screening_node(_state(cfg, records), client_factory=factory)

    assert calls["factory"] == 1
    injected_client.chat.completions.create.assert_called()
    default_client.chat.completions.create.assert_not_called()
    updated = Record.from_dict(result["included_records"][0])
    assert updated.screening_decision == "include"   # came from the injected client, not the default
