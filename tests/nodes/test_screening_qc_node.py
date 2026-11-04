"""Offline tests for slr_pipeline.nodes.screening_qc.screening_qc_node --
Stage 3 node instrumentation: one AIUseEvent per second-rater call, a
failed call returns None (never a fabricated "exclude") and is excluded
from the IRR pair count, zero-valid-pairs renders every stat None without
a divide-by-zero, and automation_origin correctly excludes failures from
its classification (INV-06)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore
from slr_pipeline.config import ReviewConfig
from slr_pipeline.state import Record, initial_state

import slr_pipeline.nodes.screening_qc as qc_module
from slr_pipeline.nodes.screening_qc import screening_qc_node


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content, model="gpt-4o-2026-01-01"):
        self.choices = [_FakeChoice(content)]
        self.model = model


def _config(tmp_path) -> ReviewConfig:
    cfg = ReviewConfig(
        title="Test review", output_dir=str(tmp_path / "outputs"), research_questions=["q"],
        inclusion_criteria=["c1"], exclusion_criteria=["c2"],
    )
    cfg.screening_qc.enabled = True
    return cfg


def _record(**overrides) -> Record:
    defaults = dict(
        id=f"rec-{overrides.get('source_id', '0')}", source="test", title="A study", abstract="abc",
        screening_decision="include", screening_relevant_topics=["x"],
    )
    defaults.update(overrides)
    return Record(**defaults)


def _state(cfg: ReviewConfig, records: list[Record]) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["screened_records_full"] = [r.to_dict() for r in records]
    st["review_run_id"] = "qc-test-run"
    return st


def _install_fake_client(create_fn):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = create_fn
    fake_client.base_url = "https://api.openai.com/v1/"
    qc_module.OpenAI = lambda **kw: fake_client
    return fake_client


def test_all_second_raters_succeed(tmp_path):
    _install_fake_client(lambda **kw: _FakeResp(json.dumps({"decision": "include"})))
    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 2
    records = [_record(source_id="1"), _record(source_id="2")]
    result = screening_qc_node(_state(cfg, records))

    irr = result["screening_qc"]["irr"]
    assert irr["n_attempted"] == 2
    assert irr["n_valid_pairs"] == 2
    assert irr["n_failed"] == 0
    assert irr["pct_agreement"] == 100.0
    assert result["oversight_mode"]["screening_qc_second_rater"] == "automation_unreviewed"
    assert result["automation_origin"]["screening_qc_second_rater"] == "ai"


def test_transport_failure_excluded_from_pairs_never_scored_as_exclude(tmp_path):
    def _raise(**kw):
        raise RuntimeError("simulated transport failure")
    _install_fake_client(_raise)

    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 1
    records = [_record(source_id="1")]
    result = screening_qc_node(_state(cfg, records))

    irr = result["screening_qc"]["irr"]
    assert irr["n_attempted"] == 1
    assert irr["n_valid_pairs"] == 0
    assert irr["n_failed"] == 1
    # every agreement statistic is None, never computed from a fabricated
    # "exclude" -- and no ZeroDivisionError was raised getting here.
    for key in ("pct_include_avg", "pct_agreement", "cohens_kappa", "pabak", "gwet_ac1", "level"):
        assert irr[key] is None

    # second_rater_decision on the record itself is None, never "exclude"
    updated = Record.from_dict(result["screened_records_full"][0])
    assert updated.second_rater_decision is None

    assert result["automation_origin"]["screening_qc_second_rater"] == "no_automated_output"


def test_all_second_raters_fail_no_exception_and_stats_none(tmp_path):
    def _raise(**kw):
        raise RuntimeError("boom")
    _install_fake_client(_raise)

    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 3
    records = [_record(source_id=str(i)) for i in range(3)]
    result = screening_qc_node(_state(cfg, records))   # must not raise

    irr = result["screening_qc"]["irr"]
    assert irr["n_attempted"] == 3
    assert irr["n_valid_pairs"] == 0
    assert irr["n_failed"] == 3


def test_some_success_some_failure_automation_origin_stays_ai_not_mixed(tmp_path):
    """QC has no fallback at this call site -- a failure produced no
    automated output at all, so it must be excluded from the
    classification entirely rather than counted as a third category that
    would make the result 'mixed' (INV-06)."""
    call_count = {"n": 0}

    def _create(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure")
        return _FakeResp(json.dumps({"decision": "include"}))
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 2
    records = [_record(source_id="1"), _record(source_id="2")]
    result = screening_qc_node(_state(cfg, records))

    assert result["automation_origin"]["screening_qc_second_rater"] == "ai"


def test_parse_failure_retains_response_model_transport_failure_does_not(tmp_path):
    """F23: call_status/output-existence and response-version availability
    are independent axes. QC has no fallback, so both outcomes are
    call_status='failure' -- but component_versions differs: a parse
    failure still had a real response (resp.model known), a transport
    failure never got one."""
    call_count = {"n": 0}

    def _create(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transport failure")
        return _FakeResp("not valid json{{{", model="gpt-4o-parsefail")
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 2
    records = [_record(source_id="1"), _record(source_id="2")]
    screening_qc_node(_state(cfg, records))

    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id="qc-test-run")
    events = event_store.load_all()
    assert len(events) == 2
    assert all(e.call_status == "failure" for e in events)
    # one event has an empty component_versions (transport failure), the
    # other carries the model's response snapshot (parse failure) --
    # independent axes, per F23.
    empties = [e for e in events if e.component_versions == {}]
    non_empties = [e for e in events if e.component_versions != {}]
    assert len(empties) == 1
    assert len(non_empties) == 1
    model_used = cfg.screening_qc.second_rater_model or cfg.screening.llm_model
    assert non_empties[0].component_versions == {model_used: "gpt-4o-parsefail"}


# ---------------------------------------------------------------------------
# Semantic validation (codex review, finding 1): an unrecognized decision
# value must never be treated as a successful, usable second-rater
# judgment -- it would otherwise silently fall through _compute_irr()'s own
# binary collapse as an unintended "exclude".
# ---------------------------------------------------------------------------

def test_invalid_decision_value_treated_as_failure_not_success(tmp_path):
    _install_fake_client(lambda **kw: _FakeResp(json.dumps({"decision": "nonsense"})))
    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 1
    records = [_record(source_id="1")]
    result = screening_qc_node(_state(cfg, records))

    irr = result["screening_qc"]["irr"]
    assert irr["n_valid_pairs"] == 0
    assert irr["n_failed"] == 1
    updated = Record.from_dict(result["screened_records_full"][0])
    assert updated.second_rater_decision is None

    output_dir = cfg.ensure_output_dir()
    events = AIUseEventStore(output_dir=output_dir, review_run_id="qc-test-run").load_all()
    assert events[0].call_status == "failure"


# ---------------------------------------------------------------------------
# Track 2 plan §4.4: ScreeningQCConfig.calibration_seed.
# ---------------------------------------------------------------------------

def test_calibration_seed_reproducible_sample_selection(tmp_path, monkeypatch):
    """`None` (default) preserves production's unseeded `random.sample`;
    a set `calibration_seed` must make the sampled subset (and its order
    of second-rater calls) reproducible across separate invocations, and
    genuinely different for a different seed."""
    cfg = _config(tmp_path)
    records = [_record(source_id=str(i)) for i in range(20)]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-seed-test")

    calls: list[str] = []

    def _fake_second_rater(record, client, model, cfg_, event_store_, review_run_id_):
        calls.append(record.id)
        return None, "evt-x"

    monkeypatch.setattr(qc_module, "_second_rater_screen", _fake_second_rater)

    qc_module._compute_irr(records, None, "m", cfg, 5, event_store, "qc-seed-test", calibration_seed=42)
    first = list(calls)
    calls.clear()
    qc_module._compute_irr(records, None, "m", cfg, 5, event_store, "qc-seed-test", calibration_seed=42)
    second = list(calls)
    assert first == second

    calls.clear()
    qc_module._compute_irr(records, None, "m", cfg, 5, event_store, "qc-seed-test", calibration_seed=99)
    third = list(calls)
    assert third != first  # different seed -- different 5-of-20 sample


def test_calibration_seed_none_still_samples_correct_count(tmp_path):
    """Regression guard: the `calibration_seed=None` default path must
    still produce a usable sample of the requested size -- not silently
    break the unseeded fallback while wiring the seeded branch in."""
    _install_fake_client(lambda **kw: _FakeResp(json.dumps({"decision": "include"})))
    cfg = _config(tmp_path)
    assert cfg.screening_qc.calibration_seed is None
    cfg.screening_qc.calibration_sample_size = 2
    records = [_record(source_id="1"), _record(source_id="2")]
    result = screening_qc_node(_state(cfg, records))
    assert result["screening_qc"]["irr"]["n_attempted"] == 2


# ---------------------------------------------------------------------------
# client_factory (Workstream C Checkpoint 3: additive, backward-compatible
# retry-policy injection seam)
# ---------------------------------------------------------------------------

def test_screening_qc_node_default_client_factory_behavior_is_unchanged(tmp_path):
    fake_client = _install_fake_client(lambda **kw: _FakeResp(json.dumps({"decision": "include"})))
    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 2
    records = [_record(source_id="1"), _record(source_id="2")]
    result = screening_qc_node(_state(cfg, records))
    assert result["screening_qc"]["irr"]["n_attempted"] == 2
    fake_client.chat.completions.create.assert_called()


def test_screening_qc_node_uses_the_injected_client_factory_instead_of_the_default(tmp_path):
    injected_client = MagicMock()
    injected_client.chat.completions.create.side_effect = lambda **kw: _FakeResp(json.dumps({"decision": "include"}))
    calls = {"factory": 0}

    def factory():
        calls["factory"] += 1
        return injected_client

    default_client = _install_fake_client(lambda **kw: _FakeResp(json.dumps({"decision": "exclude"})))
    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_sample_size = 2
    records = [_record(source_id="1"), _record(source_id="2")]
    result = screening_qc_node(_state(cfg, records), client_factory=factory)

    assert calls["factory"] == 1
    injected_client.chat.completions.create.assert_called()
    default_client.chat.completions.create.assert_not_called()
    assert result["screening_qc"]["irr"]["pct_agreement"] == 100.0   # agreed via the injected "include"


# ---------------------------------------------------------------------------
# Review finding 4: restart-safe update-wave QC (SecondRaterResultStore).
# ---------------------------------------------------------------------------

def _counting_client(decision="include"):
    calls: list = []

    def create(**kw):
        calls.append(kw)
        return _FakeResp(json.dumps({"decision": decision}))

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    client.base_url = "https://api.openai.com/v1/"
    return client, calls


def _seeded_config(tmp_path, *, seed=7, sample_size=3) -> ReviewConfig:
    cfg = _config(tmp_path)
    cfg.screening_qc.calibration_seed = seed
    cfg.screening_qc.calibration_sample_size = sample_size
    return cfg


def test_reserve_sample_requires_non_null_calibration_seed(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.screening_qc.calibration_seed is None
    records = [_record(source_id=str(i)) for i in range(5)]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test")
    store = qc_module.SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test", event_store=event_store,
    )
    with pytest.raises(ValueError, match="calibration_seed"):
        store.reserve_sample(eligible_records=records, cfg=cfg, model="m")


def test_completed_qc_pass_makes_zero_duplicate_calls_on_restart(tmp_path):
    """A fully-completed update-wave QC pass, rerun with the SAME pool/
    config, must make ZERO additional provider calls -- every sampled
    record's result is already durably published."""
    cfg = _seeded_config(tmp_path, seed=7, sample_size=3)
    records = [_record(source_id=str(i)) for i in range(5)]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test")
    store = qc_module.SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test", event_store=event_store,
    )
    client, calls = _counting_client()

    first = qc_module._compute_irr(
        records, client, "m", cfg, cfg.screening_qc.calibration_sample_size, event_store, "qc-restart-test",
        calibration_seed=cfg.screening_qc.calibration_seed, result_store=store,
    )
    assert first["n_attempted"] == 3
    assert len(calls) == 3

    second = qc_module._compute_irr(
        records, client, "m", cfg, cfg.screening_qc.calibration_sample_size, event_store, "qc-restart-test",
        calibration_seed=cfg.screening_qc.calibration_seed, result_store=store,
    )
    assert second["n_attempted"] == 3
    assert len(calls) == 3  # no new calls -- every sampled record was cached
    assert first["sampled_record_ids"] == second["sampled_record_ids"]


def test_partially_interrupted_qc_resumes_without_repeating_completed_calls(tmp_path):
    """Simulates a crash after 2 of a 3-record sample were durably
    resolved: resuming must call the provider only for the ONE record
    still missing a result, never re-calling for the 2 already done."""
    cfg = _seeded_config(tmp_path, seed=7, sample_size=3)
    records = [_record(source_id=str(i)) for i in range(5)]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test")
    store = qc_module.SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test", event_store=event_store,
    )
    pool = qc_module.compute_qc_eligible_records(records)
    sample = store.reserve_sample(eligible_records=pool, cfg=cfg, model="m")
    assert len(sample) == 3

    client, calls = _counting_client()
    # Pre-resolve all but the last sampled record, exactly as a completed
    # get_or_compute call would have durably published it before a crash.
    for record in sample[:-1]:
        qc_module._get_or_compute_second_rater_result(
            record, client, "m", cfg, event_store, store, "qc-restart-test",
        )
    assert len(calls) == 2

    result = qc_module._compute_irr(
        records, client, "m", cfg, cfg.screening_qc.calibration_sample_size, event_store, "qc-restart-test",
        calibration_seed=cfg.screening_qc.calibration_seed, result_store=store,
    )
    assert result["n_attempted"] == 3
    assert len(calls) == 3  # exactly one new call -- the previously-unresolved record


def test_crash_between_paid_call_and_publication_requires_adjudication_never_repeats_call(tmp_path):
    """Review finding 1: a process death after the real provider call
    succeeds but before the event/result are durably published must NOT
    cause a duplicate paid call on resume. Simulates the crash by
    reserving attempt 0 directly (mirroring what `_get_or_compute_second_
    rater_result` does BEFORE dispatching its call) and never publishing
    any terminal event/result for it -- then reconciling, exactly as a
    resumed pass does before any new dispatch."""
    cfg = _seeded_config(tmp_path, seed=7, sample_size=1)
    records = [_record(source_id="0")]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-crash-test")
    store = qc_module.SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id="qc-crash-test", event_store=event_store,
    )
    identity = qc_module.compute_qc_second_rater_call_identity(cfg=cfg, record=records[0], model="m")
    work_item_id = identity["work_item_id"]
    artifact_id = "secondrater-" + qc_module.sha256_hex(
        {"review_run_id": "qc-crash-test", "work_item_id": work_item_id}
    )
    # Simulate the crash: reservation marker written, but the provider
    # call's outcome (success or failure) was never durably recorded.
    store.reserve_attempt(work_item_id=work_item_id, artifact_id=artifact_id, record_id=records[0].id, attempt=0)

    archived = store.reconcile()
    assert len(archived) == 1
    assert archived[0].artifact_id == artifact_id

    client, calls = _counting_client()
    with pytest.raises(qc_module.SecondRaterAdjudicationRequiredError):
        qc_module._get_or_compute_second_rater_result(
            records[0], client, "m", cfg, event_store, store, "qc-crash-test",
        )
    assert len(calls) == 0  # zero duplicate provider calls -- never auto-retried


def test_second_rater_result_store_rejects_tampered_stored_result(tmp_path):
    cfg = _seeded_config(tmp_path, seed=7, sample_size=1)
    records = [_record(source_id="0")]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test")
    store = qc_module.SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test", event_store=event_store,
    )
    client, _calls = _counting_client()
    result = qc_module._get_or_compute_second_rater_result(
        records[0], client, "m", cfg, event_store, store, "qc-restart-test",
    )
    path = store._artifact_path(result.artifact_id)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["decision"] = "exclude"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(qc_module.SecondRaterResultIntegrityError):
        store.load(result.artifact_id)


def test_qc_sample_manifest_rejects_changed_eligible_pool_on_resume(tmp_path):
    cfg = _seeded_config(tmp_path, seed=7, sample_size=2)
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test")
    store = qc_module.SecondRaterResultStore(
        output_dir=cfg.ensure_output_dir(), review_run_id="qc-restart-test", event_store=event_store,
    )
    records = [_record(source_id=str(i)) for i in range(5)]
    store.reserve_sample(eligible_records=records, cfg=cfg, model="m")

    changed_records = [_record(source_id=str(i)) for i in range(6)]  # pool grew -- config changed since reservation
    with pytest.raises(qc_module.QCSampleManifestMismatchError):
        store.reserve_sample(eligible_records=changed_records, cfg=cfg, model="m")


def test_result_store_none_default_behavior_is_byte_for_byte_unchanged(tmp_path):
    """Regression proof for 'keep existing non-update screening behavior
    unchanged': the default (`result_store=None`) path draws its sample
    via the existing unseeded/seeded random.sample logic, never touching
    SecondRaterResultStore at all."""
    cfg = _config(tmp_path)
    records = [_record(source_id=str(i)) for i in range(5)]
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="qc-default-test")
    client, calls = _counting_client()

    result = qc_module._compute_irr(
        records, client, "m", cfg, 2, event_store, "qc-default-test", calibration_seed=3,
    )
    assert result["n_attempted"] == 2
    assert len(calls) == 2
