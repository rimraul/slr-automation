"""Unit tests for `slr_pipeline.cost_cap.ledger.CostLedger` -- the
running-total half of the Track 3 live-run spend cap
(`track3_live_run_plan_2026-09-06.md` Sec 5, user authorization
2026-09-08). No network, no OpenAI client -- pure ledger arithmetic and
persistence."""
from __future__ import annotations

import json
import threading

import pytest

from slr_pipeline.cost_cap.ledger import BudgetExceededError, CostLedger


def test_reserve_within_budget_succeeds_and_reduces_remaining(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 3.0)
    assert ledger.remaining_usd() == pytest.approx(7.0)


def test_reserve_exceeding_remaining_raises_and_reserves_nothing(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 9.0)
    with pytest.raises(BudgetExceededError):
        ledger.reserve("call-2", 2.0)
    # the failed reservation must not have been partially applied
    assert ledger.remaining_usd() == pytest.approx(1.0)


def test_commit_replaces_reservation_with_actual_cost(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 5.0)  # worst-case estimate
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                   cached_tokens=0, actual_usd=1.5)  # real cost turned out lower
    # remaining should reflect the ACTUAL cost, not the worst-case reservation
    assert ledger.remaining_usd() == pytest.approx(8.5)


def test_release_drops_reservation_without_charging(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 5.0)
    ledger.release("call-1")
    assert ledger.remaining_usd() == pytest.approx(10.0)


def test_release_of_unknown_call_id_is_a_safe_no_op(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.release("never-reserved")  # must not raise
    assert ledger.remaining_usd() == pytest.approx(10.0)


def test_commit_persists_to_disk_and_survives_reload(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 5.0)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                   cached_tokens=0, actual_usd=2.0)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["total_committed_usd"] == pytest.approx(2.0)
    assert on_disk["call_count"] == 1
    assert on_disk["calls"][0]["call_id"] == "call-1"

    reloaded = CostLedger(path, cap_usd=10.0)
    assert reloaded.remaining_usd() == pytest.approx(8.0)


def test_reload_with_different_cap_refuses(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=10, completion_tokens=10,
                   cached_tokens=0, actual_usd=0.1)
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=30.0)


def test_double_reserve_same_call_id_raises(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    with pytest.raises(ValueError):
        ledger.reserve("call-1", 1.0)


def test_commit_without_reservation_raises(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    with pytest.raises(ValueError):
        ledger.commit("never-reserved", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                       cached_tokens=0, actual_usd=0.01)


def test_reserve_non_finite_estimated_usd_raises_and_reserves_nothing(tmp_path):
    """Round-3 `/code-review` finding, most severe of that round: NaN/inf
    comparisons are always False, so `estimated_usd > remaining` would
    silently let a non-finite reservation through the one gate that exists
    to enforce the cap. Round-4 `/code-review` finding: this guard -- the
    module's last line of defense -- had no direct unit test of its own."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    with pytest.raises(ValueError):
        ledger.reserve("call-1", float("nan"))
    with pytest.raises(ValueError):
        ledger.reserve("call-2", float("inf"))
    assert ledger.remaining_usd() == pytest.approx(10.0)


def test_reserve_non_numeric_estimated_usd_raises_valueerror_not_typeerror(tmp_path):
    """Round-5 `/code-review` finding: before the switch to the shared
    `pricing.is_finite_number()` predicate, this guard was a bare
    `math.isfinite(estimated_usd)` call, which itself raises an
    undocumented `TypeError` for a non-numeric input (e.g. a string)
    instead of this method's own documented `ValueError`."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    with pytest.raises(ValueError):
        ledger.reserve("call-1", "not-a-number")
    assert ledger.remaining_usd() == pytest.approx(10.0)


def test_commit_non_finite_actual_usd_raises(tmp_path):
    """Companion to the above for `commit()`'s own guard -- a NaN/inf
    `actual_usd` must never enter `_committed_usd`, where it would poison
    every later `remaining_usd()` call for the rest of the run."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    with pytest.raises(ValueError):
        ledger.commit("call-1", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                       cached_tokens=0, actual_usd=float("nan"))


def test_construct_with_nan_cap_usd_raises_valueerror(tmp_path):
    """Round-10 `/code-review` finding: `cap_usd` itself never crossed the
    `is_finite_number()` gate rounds 3-9 hardened everywhere else. A NaN
    `cap_usd` makes `estimated_usd > remaining` always False in `reserve()`
    -- silently disabling the entire spend cap with zero refusals ever."""
    with pytest.raises(ValueError):
        CostLedger(tmp_path / "ledger.json", cap_usd=float("nan"))


def test_construct_with_huge_int_cap_usd_raises_valueerror(tmp_path):
    """Companion to the above: a huge-int `cap_usd` (e.g. `10**400`) used
    to raise an uncaught `OverflowError` from plain float arithmetic in
    `_remaining_locked()`, permanently breaking every request the proxy
    ever handles instead of failing fast, once, at construction time."""
    with pytest.raises(ValueError):
        CostLedger(tmp_path / "ledger.json", cap_usd=10 ** 400)


def test_reload_with_corrupted_huge_int_persisted_cap_raises_friendly_valueerror(tmp_path):
    """Round-10 `/code-review` finding, reproduced live: a hand-corrupted
    ledger.json with a huge-int `cap_usd` used to raise an uncaught
    `OverflowError: int too large to convert to float` from
    `_load_if_exists()`'s cap-mismatch check, instead of the deliberate,
    friendly `ValueError` that check exists to provide for exactly this
    kind of bad-ledger-file scenario."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10 ** 400,
        "total_committed_usd": 0.0,
        "call_count": 0,
        "calls": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_reload_with_corrupted_huge_int_committed_usd_raises_friendly_valueerror(tmp_path):
    """Symmetric case for `total_committed_usd`: a corrupted running total
    must not crash every future run with an uncaught `OverflowError`
    either."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10.0,
        "total_committed_usd": 10 ** 400,
        "call_count": 0,
        "calls": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_reserve_negative_estimated_usd_raises_and_reserves_nothing(tmp_path):
    """Round-11 `/code-review` finding: `is_finite_number()` checks type
    and finiteness, not sign -- a negative estimate would pass both that
    check and the `estimated_usd > remaining` gate trivially, then inflate
    `_remaining_locked()`'s reported remaining budget for every other call
    for as long as the reservation stayed open."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    with pytest.raises(ValueError):
        ledger.reserve("call-1", -5.0)
    assert ledger.remaining_usd() == pytest.approx(10.0)


def test_commit_negative_actual_usd_raises(tmp_path):
    """Companion to the above for `commit()` -- a negative committed
    amount would act as a "credit," increasing `remaining_usd()` beyond
    the true cap for every later call."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    with pytest.raises(ValueError):
        ledger.commit("call-1", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                       cached_tokens=0, actual_usd=-1.0)


def test_commit_actual_usd_exceeding_reservation_still_commits_but_warns(tmp_path, caplog):
    """Round-11 `/code-review` finding: the real cost of a completed call
    can exceed its own worst-case reservation if the pre-flight token
    estimate undercounted. The call already happened and already cost
    real money -- `commit()` must record the true cost, not refuse or
    clamp it -- but the overshoot must be surfaced loudly, since
    cumulative spend can now be closer to (or over) the cap than the
    `reserve()` gates alone would suggest."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    with caplog.at_level("WARNING"):
        ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                       cached_tokens=0, actual_usd=1.5)
    assert ledger.remaining_usd() == pytest.approx(8.5)
    assert any("exceeded its own reservation" in rec.message for rec in caplog.records)


def test_reload_with_syntactically_invalid_json_raises_friendly_valueerror(tmp_path):
    """Round-11 `/code-review` finding: a truncated/corrupted ledger.json
    (outside the atomic tmp-file-then-os.replace() write path) used to
    raise a bare `json.JSONDecodeError` instead of this module's own
    documented, friendly `ValueError`."""
    path = tmp_path / "ledger.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_reload_with_non_object_top_level_json_raises_friendly_valueerror(tmp_path):
    """Companion: a ledger.json that parses but isn't a JSON object (e.g.
    a bare list) must not crash the `.get()` calls with an uncaught
    `AttributeError`."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_reload_with_non_list_calls_field_raises_friendly_valueerror(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10.0, "total_committed_usd": 0.0, "call_count": 0, "calls": "not-a-list",
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_reload_with_call_record_missing_a_key_raises_friendly_valueerror(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10.0, "total_committed_usd": 0.0, "call_count": 1,
        "calls": [{"call_id": "x"}],  # missing model/prompt_tokens/.../ts
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_reload_with_negative_committed_usd_raises_friendly_valueerror(tmp_path):
    """Round-12 `/code-review` finding: `is_finite_number()` accepts any
    finite value regardless of sign -- a hand-corrupted `total_committed_usd`
    of e.g. -20.0 used to pass that check, then `_remaining_locked()`
    computed `cap_usd - (-20.0)`, inflating the resumed run's remaining
    budget above the authorized cap and silently defeating the whole
    mechanism for the rest of the run."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10.0, "total_committed_usd": -20.0, "call_count": 0, "calls": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        CostLedger(path, cap_usd=10.0)


def test_construct_with_negative_cap_usd_raises_valueerror(tmp_path):
    """Round-13 `/code-review` finding: `is_finite_number()` accepts any
    finite value regardless of sign -- a negative `cap_usd` happened to
    "fail safe" (every `reserve()` is immediately refused since
    `remaining_usd()` starts negative), but silently violates this class's
    own documented invariant that `cap_usd` is a non-negative authorized
    spend limit."""
    with pytest.raises(ValueError):
        CostLedger(tmp_path / "ledger.json", cap_usd=-5.0)


def test_commit_persist_typeerror_does_not_lose_in_memory_committed_spend(tmp_path, monkeypatch):
    """Round-13 `/code-review` finding: `commit()`'s persist-failure guard
    only caught `OSError`, unlike the structurally identical
    `json.dumps()` risk in this module's sibling `_CallLogWriter.append()`
    (round-5 fix widened that one to `OSError, TypeError, ValueError`).
    A `TypeError` from `json.dumps()` (e.g. a future caller passing a
    non-JSON-serializable value) must not escape `commit()` after the
    in-memory running total has already been updated -- the caller would
    otherwise fall through to its own exception handler and RELEASE a
    reservation that was already, correctly, committed in memory."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 1.0)

    def _boom(*args, **kwargs):
        raise TypeError("not JSON serializable")

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _boom)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                   cached_tokens=0, actual_usd=1.5)  # must not raise
    assert ledger.remaining_usd() == pytest.approx(8.5)


def test_commit_persist_exotic_exception_does_not_escape_or_lose_committed_spend(tmp_path, monkeypatch):
    """Round-43 `/code-review` finding: `_persist_with_retry()`/
    `_persist_once_locked()` only catch `(OSError, TypeError, ValueError)`
    -- an exotic exception outside that set (e.g. `RecursionError`/
    `MemoryError`) used to escape `commit()` uncaught, AFTER the in-memory
    running total had already been updated, violating this method's own
    documented invariant that a persist failure must never make `commit()`
    raise. A caller with no surrounding try/except of its own (every
    pre-200 fail-safe path in `proxy.py`) would see `commit()` raise
    despite the ledger's accounting already being correct."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    ledger.reserve("call-1", 1.0)

    def _boom(*args, **kwargs):
        raise RecursionError("simulated exotic persist failure")

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _boom)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                   cached_tokens=0, actual_usd=1.5)  # must not raise
    assert ledger.remaining_usd() == pytest.approx(8.5)
    assert ledger._persist_dirty is True


def test_commit_usage_verified_defaults_true_and_persists(tmp_path):
    """Round-14 `/code-review` finding: without a flag distinguishing
    genuinely `usage`-verified records from worst-case-charged ones, the
    ledger's own audit trail couldn't tell real telemetry from a guess
    without cross-referencing the separate call-log file."""
    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                   cached_tokens=0, actual_usd=1.0)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["calls"][0]["usage_verified"] is True


def test_commit_usage_verified_false_persists_and_survives_reload(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 1.0)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=0,
                   cached_tokens=0, actual_usd=1.0, usage_verified=False)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["calls"][0]["usage_verified"] is False
    reloaded = CostLedger(path, cap_usd=10.0)
    assert reloaded.remaining_usd() == pytest.approx(9.0)


def test_reload_old_ledger_json_without_usage_verified_field_defaults_true(tmp_path):
    """A `ledger.json` written before this field existed must still
    reload cleanly, defaulting every pre-existing record to the common
    "verified" case rather than refusing to resume an otherwise-valid
    in-progress run."""
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10.0, "total_committed_usd": 1.0, "call_count": 1,
        "calls": [{
            "call_id": "call-1", "model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50,
            "cached_tokens": 0, "actual_usd": 1.0, "ts": 0.0,
            # no "usage_verified" key -- pre-round-14 shape
        }],
    }), encoding="utf-8")
    reloaded = CostLedger(path, cap_usd=10.0)  # must not raise
    assert reloaded.remaining_usd() == pytest.approx(9.0)


def test_concurrent_reservations_never_overshoot_cap(tmp_path):
    """The scenario `reserve()`'s docstring exists to prevent: N threads
    (mirroring `screening.py`'s `ThreadPoolExecutor`) racing to reserve
    against a tight budget. Exactly enough for 10 calls at $1 each must
    succeed; any extra must be refused -- never let more than 10 through."""
    ledger = CostLedger(tmp_path / "ledger.json", cap_usd=10.0)
    successes = []
    lock = threading.Lock()

    def worker(i):
        try:
            ledger.reserve(f"call-{i}", 1.0)
            with lock:
                successes.append(i)
        except BudgetExceededError:
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 10
    assert ledger.remaining_usd() == pytest.approx(0.0)


def test_load_if_exists_os_error_raises_friendly_valueerror_not_uncaught_exception(tmp_path, monkeypatch):
    """Round-36 `/code-review` finding, reproduced live: `_load_if_exists()`'s
    raw `self._path.read_text(...)` had no guard against `OSError` (e.g. a
    Windows `PermissionError` from a transient antivirus scan or another
    process briefly holding an exclusive handle on this exact file at the
    moment a resumed run's `CostLedger(...)` is constructed) -- unlike
    every other corrupted-ledger scenario a few lines below in this same
    method, which is deliberately converted into this module's own
    established friendly `ValueError`, not left to crash the caller with
    an uncaught internal exception."""
    import pathlib

    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "cap_usd": 10.0,
        "total_committed_usd": 0.0,
        "call_count": 0,
        "calls": [],
    }), encoding="utf-8")

    real_read_text = pathlib.Path.read_text

    def _flaky_read_text(self, *args, **kwargs):
        if self == path:
            raise PermissionError("simulated transient exclusive lock")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(ledger_module.Path, "read_text", _flaky_read_text)
    with pytest.raises(ValueError, match="could not be read"):
        CostLedger(path, cap_usd=10.0)


def test_load_if_exists_invalid_utf8_raises_friendly_valueerror_not_unicodedecodeerror(tmp_path):
    """Round-37 `/code-review` finding, reproduced live: the round-36
    OSError guard on `self._path.read_text(...)` in `_load_if_exists()`
    did not also catch `UnicodeDecodeError` -- a `ValueError` subclass,
    not an `OSError` subclass -- so a ledger.json with invalid UTF-8
    bytes (a hand-edit mistake, or disk-level bit corruption, both
    threat vectors this method's own docstring/comments already name)
    slipped past that guard and crashed the constructor with a raw
    `UnicodeDecodeError` instead of this method's own established
    friendly `ValueError`, unlike every other corrupted-ledger scenario
    in this same method."""
    path = tmp_path / "ledger.json"
    path.write_bytes(b'{"cap_usd": 10.0, "total_committed_usd": 0.0, '
                      b'"call_count": 0, "calls": [\xff\xfe]}')
    with pytest.raises(ValueError, match="could not be read"):
        CostLedger(path, cap_usd=10.0)


def test_commit_persist_transient_oserror_is_retried_and_recovers(tmp_path, monkeypatch):
    """Round-38 `/code-review` finding, reproduced live: `commit()`'s
    persist call used to give up after a single failed attempt -- if that
    was the LAST commit() before the process exited, real, already-billed
    spend never reached disk, and a later resumed run reloaded only the
    stale, lower total. Most such failures are transient (e.g. the
    Windows antivirus/exclusive-handle lock this file's own comments
    already name), so `_persist_with_retry_locked()` retries a bounded
    number of times on `OSError` before giving up. Simulate a persist
    that fails twice then succeeds and confirm the on-disk file ends up
    correct, not stale, with no need for any later call in this process."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 5.0)

    real_persist_locked = ledger_module.CostLedger._persist_locked
    calls = {"n": 0}

    def _flaky_persist_locked(self):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("simulated transient exclusive lock")
        return real_persist_locked(self)

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _flaky_persist_locked)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                   cached_tokens=0, actual_usd=5.0)

    assert calls["n"] == 3  # failed twice, succeeded on the third attempt
    assert ledger._persist_dirty is False

    # A brand-new process reloading this same path must see the real
    # committed total, not a stale one from a lost persist.
    reloaded = CostLedger(path, cap_usd=10.0)
    assert reloaded.remaining_usd() == pytest.approx(5.0)


def test_commit_persist_exhausted_retries_self_heals_on_next_reserve(tmp_path, monkeypatch):
    """Round-38 `/code-review` finding, reproduced live: when a persist
    failure outlasts `commit()`'s own retry budget, the ledger is left
    dirty on disk -- but the process doesn't necessarily exit immediately.
    `reserve()`/`release()` opportunistically retry a dirty persist on
    their next lock acquisition, so a LATER call in the same process (not
    just the one that originally failed) gets a free chance to flush the
    lost commit to disk before the process actually exits. Simulate a
    persist that fails for the whole duration of commit()'s retries, then
    starts succeeding again by the time reserve() is called."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 5.0)

    real_persist_locked = ledger_module.CostLedger._persist_locked
    failing = {"active": True}

    def _sometimes_failing_persist_locked(self):
        if failing["active"]:
            raise OSError("simulated exclusive lock, still held")
        return real_persist_locked(self)

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _sometimes_failing_persist_locked)
    ledger.commit("call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                   cached_tokens=0, actual_usd=5.0)
    assert ledger._persist_dirty is True  # every retry attempt failed

    # The lock clears before the next call -- reserve() gets a free
    # opportunistic chance to flush the earlier commit's lost write.
    failing["active"] = False
    ledger.reserve("call-2", 1.0)
    assert ledger._persist_dirty is False

    reloaded = CostLedger(path, cap_usd=10.0)
    assert reloaded.remaining_usd() == pytest.approx(5.0)  # the earlier commit's $5 was not lost


def test_dirty_flush_does_not_serialize_concurrent_calls_during_sustained_failure(tmp_path, monkeypatch):
    """Round-39 `/code-review` finding, reproduced live: an earlier
    version's opportunistic dirty-flush in `reserve()`/`release()` ran
    the SAME multi-attempt retry-with-backoff loop `commit()` uses INSIDE
    the caller's own critical section -- against a SUSTAINED (non-
    transient) disk problem, every `reserve()` call paid the full
    3-attempt/0.05s-backoff budget while holding `self._lock` (measured
    live: 5 threads x 10 calls each took ~5s wall-clock, fully
    serialized). `_flush_dirty_persist_if_any()` now takes a single quick
    attempt (no retry/backoff) via its own brief lock acquisition, so a
    sustained outage should no longer collapse concurrency this way.
    Assert 5 threads x 10 reserve() calls each complete well under what
    the old serialized behavior would have taken."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=1000.0)
    ledger.reserve("seed", 1.0)

    def _always_failing_persist_locked(self):
        raise OSError("simulated sustained disk problem")

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _always_failing_persist_locked)
    ledger.commit("seed", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                   cached_tokens=0, actual_usd=1.0)
    assert ledger._persist_dirty is True

    errors = []

    def worker(thread_index):
        try:
            for i in range(10):
                ledger.reserve(f"call-{thread_index}-{i}", 1.0)
        except Exception as exc:  # pragma: no cover -- surfaced via `errors` below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    import time as time_module

    start = time_module.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    elapsed = time_module.monotonic() - start

    assert not errors
    assert not any(t.is_alive() for t in threads)
    # The old serialized behavior measured ~5s for this exact workload;
    # a single quick attempt per call should finish in a small fraction
    # of that, even with scheduling/test-runner overhead.
    assert elapsed < 2.0, f"5 threads x 10 reserve() calls took {elapsed:.3f}s -- looks serialized again"


def test_dirty_flush_failure_warning_is_throttled_not_spammed(tmp_path, monkeypatch, caplog):
    """Round-39 `/code-review` finding, reproduced live: a repeat
    opportunistic dirty-flush failure in `reserve()`/`release()` used to
    be completely silent -- only the ONE original `commit()` failure was
    ever logged, so an operator running a long unattended live run had no
    way to notice the on-disk ledger was still stale minutes or hours
    later. `_flush_dirty_persist_if_any()` now logs a throttled
    `logger.warning()` on each repeat failure. Verify: (a) many rapid
    calls within the default throttle window produce at most one such
    warning, not one per call; (b) once the throttle window has
    conceptually elapsed, the next failure logs again."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=1000.0)
    ledger.reserve("seed", 1.0)

    def _always_failing_persist_locked(self):
        raise OSError("simulated sustained disk problem")

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _always_failing_persist_locked)
    with caplog.at_level("WARNING"):
        ledger.commit("seed", model="gpt-4o", prompt_tokens=1, completion_tokens=1,
                       cached_tokens=0, actual_usd=1.0)
        caplog.clear()
        for i in range(20):
            ledger.reserve(f"call-{i}", 1.0)
        dirty_warnings = [r for r in caplog.records if "still dirty" in r.message]
        assert len(dirty_warnings) <= 1, (
            f"expected the throttle to allow at most one warning across 20 rapid calls, got "
            f"{len(dirty_warnings)}"
        )

        caplog.clear()
        # Simulate the throttle window having elapsed: with the throttle
        # constant at 0, every subsequent failed flush attempt must log.
        monkeypatch.setattr(ledger_module, "_DIRTY_LOG_THROTTLE_S", 0.0)
        ledger.reserve("call-after-throttle-1", 1.0)
        ledger.reserve("call-after-throttle-2", 1.0)
        dirty_warnings_after = [r for r in caplog.records if "still dirty" in r.message]
        assert len(dirty_warnings_after) == 2


def test_dirty_flag_set_before_the_first_persist_attempt_not_after(tmp_path, monkeypatch):
    """Round-40 `/code-review` finding, reproduced live: round 39 moved
    the persist-retry loop OUTSIDE commit()'s own state-mutation lock (so
    a sustained outage doesn't hold self._lock across backoff sleeps) --
    which opened a window between mutating self._committed_usd/self.
    _calls and commit()'s own first persist attempt where `self._persist_
    dirty` still read `False` even though the on-disk file did not yet
    reflect the commit. An operator polling `snapshot()['persist_dirty']`
    for staleness during that window had a blind spot for "committed but
    not yet even attempted to persist," as opposed to "persist attempted
    and failed." Fixed by marking dirty pre-emptively in the SAME lock
    acquisition as the mutation, before it is ever released.

    `_persist_once_locked()` (and the slow persist call below) always run
    while `self._lock` is held, so `snapshot()` -- which needs the same
    lock -- cannot observe an in-flight persist attempt from another
    thread without simply blocking until it finishes; reading the raw
    `_persist_dirty` attribute directly (a plain, GIL-atomic bool
    read/write, not protected by -- or needing -- the lock itself) is
    what lets this test observe the state genuinely mid-flight, between
    the mutation and the first persist attempt actually reaching disk."""
    import threading as threading_module

    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    ledger.reserve("call-1", 5.0)

    persist_started = threading_module.Event()
    release_persist = threading_module.Event()
    real_persist_locked = ledger_module.CostLedger._persist_locked

    def _slow_persist_locked(self):
        persist_started.set()
        release_persist.wait(timeout=10)
        return real_persist_locked(self)

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _slow_persist_locked)

    commit_thread = threading_module.Thread(
        target=ledger.commit,
        kwargs=dict(call_id="call-1", model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                    cached_tokens=0, actual_usd=5.0),
    )
    commit_thread.start()
    try:
        assert persist_started.wait(timeout=5), "commit() never reached its first persist attempt"
        # The mutation has already happened (committed_usd reflects it)
        # and the slow persist call is genuinely still blocked mid-flight
        # -- dirty must already read True here, not False, and the file
        # must not exist on disk yet.
        assert ledger._committed_usd == pytest.approx(5.0)
        assert ledger._persist_dirty is True
        assert not path.exists()
    finally:
        release_persist.set()
        commit_thread.join(timeout=10)

    assert not commit_thread.is_alive()
    final = ledger.snapshot()
    assert final["persist_dirty"] is False
    assert path.exists()


# ---------------------------------------------------------------------------
# Code-review corrective pass, round 3 (P2): `ensure_persisted()` must route
# through the same `_persist_with_retry()` machinery `commit()` uses, not
# call `_persist_locked()` directly -- so a transient failure here retries,
# marks `_persist_dirty`, and (unlike `commit()`) is actually raised on
# exhaustion, since this method's whole purpose is guaranteeing a file
# lands on disk.
# ---------------------------------------------------------------------------


def test_ensure_persisted_writes_file_with_zero_commits(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)
    assert not path.exists()

    ledger.ensure_persisted()

    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["total_committed_usd"] == 0.0
    assert ledger._persist_dirty is False


def test_ensure_persisted_retries_transient_oserror_and_recovers(tmp_path, monkeypatch):
    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)

    real_persist_locked = ledger_module.CostLedger._persist_locked
    calls = {"n": 0}

    def _flaky_persist_locked(self):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("simulated transient exclusive lock")
        return real_persist_locked(self)

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _flaky_persist_locked)
    ledger.ensure_persisted()

    assert calls["n"] == 3  # failed twice, succeeded on the third attempt
    assert ledger._persist_dirty is False
    assert path.exists()


def test_ensure_persisted_raises_after_exhausting_retries_and_stays_dirty(tmp_path, monkeypatch):
    """Unlike `commit()` (which must never raise on a persist failure --
    the in-memory cap enforcement is already correct and the call already
    happened), `ensure_persisted()` has no other purpose than guaranteeing
    a file lands on disk, so a failure that outlasts every retry must be
    raised, not silently swallowed."""
    import slr_pipeline.cost_cap.ledger as ledger_module

    path = tmp_path / "ledger.json"
    ledger = CostLedger(path, cap_usd=10.0)

    def _always_failing_persist_locked(self):
        raise OSError("simulated sustained disk problem")

    monkeypatch.setattr(ledger_module.CostLedger, "_persist_locked", _always_failing_persist_locked)

    with pytest.raises(RuntimeError, match="failed to persist ledger"):
        ledger.ensure_persisted()

    assert ledger._persist_dirty is True
    assert not path.exists()
