"""Regression tests for slr_pipeline.ai_traceability's hardened AIUseEvent
store -- Stage 3, Section 3.3/3.5 acceptance criteria."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pydantic
import pytest

from slr_pipeline.ai_traceability import (
    AIEventIntegrityError,
    AIEventLockError,
    AIUseEvent,
    AIUseEventStore,
    build_ai_use_event,
    compute_event_id,
    kernel_lock,
    reserved_attempt,
    sanitize_call_error_text,
)


def _llm_event(**overrides):
    kwargs = dict(
        review_run_id="run-1", stage="screening", work_item_id="rec-1:batch-0",
        tool_kind="llm", model="gpt-5.6-terra", provider="openai",
        component_versions={"gpt-5.6-terra": "2026-07-30"},
        task_purpose="title/abstract screening decision",
        model_config_id="slr-chapter-teaching-preset-v1",
        input_artifact_refs=["rec-1"], output_artifact_refs=["decision-rec-1"],
        call_status="success",
        prompt_template_version="screen-v1", prompt_template_text="Decide include/exclude...",
        rendered_prompt_hash="cafef00d",
        prompt_parameters={"temperature": 0.0},
    )
    kwargs.update(overrides)
    return build_ai_use_event(**kwargs)


def test_llm_event_requires_prompt_fields():
    with pytest.raises(ValueError):
        build_ai_use_event(
            review_run_id="run-1", stage="screening", work_item_id="rec-1:batch-0",
            tool_kind="llm", model="gpt-5.6-terra", provider="openai",
            component_versions={"gpt-5.6-terra": "2026-07-30"},
            task_purpose="x", model_config_id="preset-1",
            input_artifact_refs=[], output_artifact_refs=[], call_status="success",
        )


def test_non_llm_tool_event_allows_none_prompt_fields():
    event = build_ai_use_event(
        review_run_id="run-1", stage="synthesis", work_item_id="bertopic-run-1",
        tool_kind="non_llm_tool", model="bertopic", provider="maartengr/bertopic",
        component_versions={
            "bertopic": "0.16.x", "sentence-transformers": "3.x",
            "umap-learn": "0.5.x", "hdbscan": "0.8.x",
        },
        task_purpose="exploratory topic modelling over charted key_findings",
        model_config_id="bertopic-default-v1",
        input_artifact_refs=["extrun-1"], output_artifact_refs=["topics.json"],
        call_status="success", prompt_parameters={"min_cluster_size": 5},
    )
    assert event.prompt_template_text is None
    assert event.prompt_template_hash is None
    assert event.rendered_prompt_hash is None
    assert event.prompt_template_version is None
    assert len(event.component_versions) == 4


def test_publish_and_load_roundtrip(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event()
    published = store.publish(event)
    assert published.event_id == event.event_id
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].event_id == event.event_id
    assert loaded[0].content_hash == event.content_hash


def test_content_hash_matches_persisted_file(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event()
    store.publish(event)
    on_disk = json.loads(store.event_path(event.event_id).read_text(encoding="utf-8"))
    persisted = AIUseEvent.model_validate(on_disk)
    assert persisted.content_hash == persisted.compute_content_hash()


def test_idempotent_retry_same_work_item_attempt_is_a_noop(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    first = _llm_event()
    store.publish(first)
    retry = _llm_event()  # identical inputs -> identical event_id/semantic_hash, different timestamp allowed by construction (same timestamp here since build is deterministic-ish)
    result = store.publish(retry)
    assert result.event_id == first.event_id
    files = list(store.events_dir.glob("*.json"))
    assert len(files) == 1


def test_different_work_item_forced_identical_event_id_raises(tmp_path, monkeypatch):
    """Direct regression for the reviewer's own reproduction (finding 1):
    two events with genuinely DIFFERENT work_item_id, otherwise IDENTICAL
    metadata -- task_purpose deliberately left unchanged, unlike the
    prior version of this test -- that collide on event_id (forced via a
    monkeypatched compute_event_id, since a real sha256 collision is
    infeasible to construct honestly) must never be accepted as a benign
    retry: work_item_id is now hashed into semantic_hash, so the two
    events' semantic_hash genuinely differs despite the forced event_id
    collision, and publish() must raise rather than silently treat the
    second as a duplicate of the first."""
    monkeypatch.setattr("slr_pipeline.ai_traceability.compute_event_id", lambda **kwargs: "aiuse-forced-collision")
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    first = _llm_event(work_item_id="rec-1:batch-0")
    second = _llm_event(work_item_id="rec-2:batch-0")  # task_purpose UNCHANGED -- only work_item_id differs
    assert first.event_id == second.event_id == "aiuse-forced-collision"
    assert first.semantic_hash != second.semantic_hash

    store.publish(first)
    with pytest.raises(AIEventIntegrityError):
        store.publish(second)
    leftover_temps = [p for p in store.events_dir.glob("*.tmp.*")]
    assert len(leftover_temps) == 1


def test_work_item_id_and_attempt_change_semantic_hash():
    a = _llm_event(work_item_id="rec-1:batch-0", attempt=0)
    b = _llm_event(work_item_id="rec-2:batch-0", attempt=0)
    c = _llm_event(work_item_id="rec-1:batch-0", attempt=1)
    assert a.semantic_hash != b.semantic_hash
    assert a.semantic_hash != c.semantic_hash
    assert a.event_id != b.event_id
    assert a.event_id != c.event_id


def test_negative_attempt_rejected():
    with pytest.raises(pydantic.ValidationError):
        _llm_event(attempt=-1)


def test_event_id_not_matching_identity_fields_rejected_at_construction():
    """Direct regression for finding 3's 'a self-consistent forged
    event_id loaded successfully': hand-crafting a JSON payload whose
    event_id does not match compute_event_id() for its own
    review_run_id/stage/work_item_id/attempt is rejected immediately --
    both for a fresh construction and for model_validate() on load,
    since it is the same validator either way."""
    event = _llm_event(work_item_id="rec-1:batch-0")
    payload = event.model_dump(mode="json")
    payload["event_id"] = "aiuse-" + "0" * 48  # does not match compute_event_id() for these fields
    with pytest.raises(pydantic.ValidationError):
        AIUseEvent.model_validate(payload)


def test_review_run_id_path_traversal_rejected(tmp_path):
    """Direct regression for finding 2: review_run_id='../escaped' must
    never be able to write outside ai_traceability/."""
    with pytest.raises(ValueError):
        AIUseEventStore(output_dir=tmp_path, review_run_id="../escaped")
    # ai_traceability/../escaped resolves to tmp_path/escaped (one level up
    # from ai_traceability/, back into tmp_path) -- confirm nothing was
    # written there. Construction raises before any mkdir either way, so
    # this is a belt-and-suspenders check on the actual would-be escape
    # target, not tmp_path.parent (a directory this store could never
    # reach even without the safe-token/containment checks).
    assert not (tmp_path / "escaped").exists()


def test_publish_rejects_foreign_run_event(tmp_path):
    """Direct regression for finding 3's 'wrong_run_accepted=run-B'."""
    store_a = AIUseEventStore(output_dir=tmp_path, review_run_id="run-A")
    event_for_run_b = _llm_event(review_run_id="run-B", work_item_id="rec-1:batch-0")
    with pytest.raises(AIEventIntegrityError):
        store_a.publish(event_for_run_b)


def test_load_all_rejects_foreign_run_event_found_in_directory(tmp_path):
    """A foreign-run event's file placed directly in run-A's events
    directory (bypassing publish()'s own check entirely) must still be
    caught at load time, not just at publish time."""
    store_a = AIUseEventStore(output_dir=tmp_path, review_run_id="run-A")
    store_a.events_dir.mkdir(parents=True, exist_ok=True)
    foreign_event = _llm_event(review_run_id="run-B", work_item_id="rec-1:batch-0")
    (store_a.events_dir / f"{foreign_event.event_id}.json").write_text(
        foreign_event.model_dump_json(indent=2), encoding="utf-8"
    )
    with pytest.raises(AIEventIntegrityError):
        store_a.load_all()


def test_load_all_rejects_renamed_copy(tmp_path):
    """Direct regression for finding 3's 'A renamed copy produced
    renamed_duplicate_count=2': a valid event file copied under a
    different filename must be rejected, not silently loaded as a second
    valid event."""
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event()
    store.publish(event)
    original_path = store.event_path(event.event_id)
    renamed_copy = store.events_dir / "renamed-copy.json"
    renamed_copy.write_text(original_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(AIEventIntegrityError):
        store.load_all()


def test_extra_field_injection_rejected(tmp_path):
    """Direct regression for finding 4: an unknown injected JSON field
    (which neither hash would ever notice, since hashes are computed
    only from known model fields) must be rejected outright by
    extra='forbid', never silently dropped/ignored."""
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event()
    store.publish(event)
    path = store.event_path(event.event_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["injected_extra_field"] = "unexpected"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        store.load_all()


def test_event_is_frozen():
    event = _llm_event()
    with pytest.raises(pydantic.ValidationError):
        event.task_purpose = "mutated after construction"


def test_non_llm_tool_event_with_populated_prompt_fields_rejected():
    """Direct regression for finding 5: the LLM/non-LLM distinction must
    be enforced bidirectionally -- a non_llm_tool event with all four
    prompt fields populated must be rejected, not silently accepted."""
    with pytest.raises(pydantic.ValidationError):
        AIUseEvent.model_validate(
            {
                "review_run_id": "run-1", "event_id": "placeholder", "stage": "synthesis",
                "work_item_id": "bertopic-run-1", "attempt": 0, "tool_kind": "non_llm_tool",
                "model": "bertopic", "provider": "maartengr/bertopic",
                "component_versions": {"bertopic": "0.16.x"}, "access_info": None,
                "task_purpose": "topic modelling", "model_config_id": "bertopic-default-v1",
                "prompt_template_version": "v1", "prompt_template_text": "some template",
                "prompt_template_hash": "deadbeef", "prompt_parameters": {},
                "rendered_prompt_hash": "cafef00d", "input_artifact_refs": [], "output_artifact_refs": [],
                "call_status": "success", "failure_detail": None, "timestamp": "2026-08-22T00:00:00Z",
                "semantic_hash": "placeholder", "content_hash": "placeholder",
            }
        )


def test_prompt_template_hash_must_match_stored_template():
    """Direct regression for finding 5's second half: prompt_template_hash
    is derived from prompt_template_text, never an arbitrary
    caller-supplied string -- constructing an event where the two
    disagree is rejected."""
    event = _llm_event()
    payload = event.model_dump(mode="json")
    payload["prompt_template_hash"] = "deadbeef"  # does not match sha256 of prompt_template_text
    with pytest.raises(pydantic.ValidationError):
        AIUseEvent.model_validate(payload)


def test_build_ai_use_event_derives_prompt_template_hash():
    event = _llm_event(prompt_template_text="Decide include/exclude...")
    from slr_pipeline.ai_traceability import compute_prompt_template_hash

    assert event.prompt_template_hash == compute_prompt_template_hash("Decide include/exclude...")


def test_crash_between_fsync_and_link_leaves_no_truncated_final_file(tmp_path, monkeypatch):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event()

    real_link = __import__("os").link

    def _boom(*args, **kwargs):
        raise OSError("simulated crash between fsync and hard-link publish")

    monkeypatch.setattr("slr_pipeline.ai_traceability.os.link", _boom)
    with pytest.raises(OSError):
        store.publish(event)

    final_path = store.event_path(event.event_id)
    assert not final_path.exists()
    tmp_files = list(store.events_dir.glob("*.tmp.*"))
    assert len(tmp_files) == 1
    # the temp file itself is fully written (fsync happened before the
    # simulated crash) -- never a truncated/partial write
    persisted = AIUseEvent.model_validate(json.loads(tmp_files[0].read_text(encoding="utf-8")))
    persisted.verify_hashes()


def test_timestamp_tampering_detected_by_content_hash_on_load(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event()
    store.publish(event)

    path = store.event_path(event.event_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["timestamp"] = "2099-01-01T00:00:00Z"  # tamper timestamp only, hashes left untouched
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(AIEventIntegrityError):
        store.load_all()


def test_concurrent_publish_same_work_item_attempt_never_corrupts(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    errors = []

    def _worker():
        try:
            store.publish(_llm_event())
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    files = [p for p in store.events_dir.glob("*.json")]
    assert len(files) == 1
    loaded = store.load_all()
    assert len(loaded) == 1
    loaded[0].verify_hashes()


def test_kernel_lock_times_out_when_already_held(tmp_path):
    lock_path = tmp_path / "some.lock"
    holder_ready = threading.Event()
    release_holder = threading.Event()

    def _hold():
        with kernel_lock(lock_path, timeout=5.0):
            holder_ready.set()
            release_holder.wait(timeout=5.0)

    holder_thread = threading.Thread(target=_hold)
    holder_thread.start()
    assert holder_ready.wait(timeout=5.0)
    try:
        start = time.monotonic()
        with pytest.raises(AIEventLockError):
            with kernel_lock(lock_path, timeout=0.3, poll_interval=0.02):
                pass  # pragma: no cover - must never be reached while held
        elapsed = time.monotonic() - start
        assert elapsed >= 0.3
    finally:
        release_holder.set()
        holder_thread.join(timeout=5.0)


def test_kernel_lock_releases_after_exception(tmp_path):
    lock_path = tmp_path / "some.lock"

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with kernel_lock(lock_path, timeout=1.0):
            raise _Boom("failure inside the locked section")

    # the lock must have been released despite the exception -- a fresh
    # acquire immediately afterward must succeed without waiting out a
    # timeout
    acquired = []
    with kernel_lock(lock_path, timeout=1.0):
        acquired.append(True)
    assert acquired == [True]


# ---------------------------------------------------------------------------
# load_event() -- INV-11: a persisted, caller-supplied event_id string is
# never interpolated into a filesystem path without a format check plus
# resolved-path containment first.
# ---------------------------------------------------------------------------

def test_load_event_roundtrip(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    published = store.publish(_llm_event())
    loaded = store.load_event(published.event_id)
    assert loaded.event_id == published.event_id
    assert loaded.content_hash == published.content_hash


@pytest.mark.parametrize("bad_id", ["not-an-event-id", "aiuse-tooshort", "../../etc/passwd", "aiuse-" + "g" * 48])
def test_load_event_rejects_malformed_or_path_traversal_ids(tmp_path, bad_id):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    with pytest.raises(AIEventIntegrityError):
        store.load_event(bad_id)


def test_load_event_rejects_nonexistent_wellformed_id(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    fake_id = compute_event_id(review_run_id="run-1", stage="x", work_item_id="y", attempt=0)
    with pytest.raises(AIEventIntegrityError):
        store.load_event(fake_id)


def test_load_event_rejects_foreign_run_event(tmp_path):
    """A well-formed, self-consistent event genuinely published under a
    DIFFERENT review_run_id physically lives under that run's own
    directory tree (review_run_id determines the store's own root), so
    reaching this specific check requires a file placed directly rather
    than a legitimate publish -- covers the same review_run_id-binding
    discipline load_all() already enforces, now shared via
    _load_and_verify_event()."""
    store_a = AIUseEventStore(output_dir=tmp_path, review_run_id="run-a")
    published = store_a.publish(_llm_event(review_run_id="run-a"))

    store_b = AIUseEventStore(output_dir=tmp_path, review_run_id="run-b")
    import shutil
    dest = store_b.event_path(published.event_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(store_a.event_path(published.event_id), dest)
    with pytest.raises(AIEventIntegrityError):
        store_b.load_event(published.event_id)


# ---------------------------------------------------------------------------
# publish()'s collision path now uses the SAME shared _load_and_verify_event
# as load_all()/load_event() -- closing a real gap in the already-accepted
# store (the collision path previously skipped the review_run_id/filename
# checks load_all() already had). event_id's own structural binding to
# review_run_id (_check_event_id_matches_identity) makes a genuine foreign-
# run collision at the exact same path unreachable through legitimate
# construction; the filename-binding check is exercised directly instead.
# ---------------------------------------------------------------------------

def test_publish_collision_path_rejects_tampered_existing_file(tmp_path):
    """A file exists at the exact path publish() would collide on, but
    its own persisted content has been tampered with (breaking self-
    verification) -- publish()'s collision path must catch this via the
    shared verified loader (`_load_and_verify_event`) before ever
    comparing semantic_hash, exactly like every other store's own
    collision path already does."""
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event = _llm_event(work_item_id="wi-1", attempt=0)
    store.publish(event)

    path = store.event_path(event.event_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["task_purpose"] = "tampered after publish"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AIEventIntegrityError):
        store.publish(_llm_event(work_item_id="wi-1", attempt=0))


# ---------------------------------------------------------------------------
# reserved_attempt() -- concurrency-safe attempt allocation (F1/F15).
# ---------------------------------------------------------------------------

def test_reserved_attempt_starts_at_zero_and_increments(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    with reserved_attempt(store, stage="screening", work_item_id="wi-1") as attempt:
        assert attempt == 0
        store.publish(_llm_event(work_item_id="wi-1", attempt=attempt))

    with reserved_attempt(store, stage="screening", work_item_id="wi-1") as attempt:
        assert attempt == 1
        store.publish(_llm_event(work_item_id="wi-1", attempt=attempt))

    with reserved_attempt(store, stage="screening", work_item_id="wi-1") as attempt:
        assert attempt == 2


def test_reserved_attempt_different_work_items_start_at_zero_independently(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    with reserved_attempt(store, stage="screening", work_item_id="wi-a") as attempt:
        store.publish(_llm_event(work_item_id="wi-a", attempt=attempt))
    with reserved_attempt(store, stage="screening", work_item_id="wi-b") as attempt:
        assert attempt == 0


def test_reserved_attempt_holds_lock_across_the_whole_block(tmp_path):
    """Two real threads reserving the SAME work item concurrently must be
    serialized -- both a lock-contention artifact (only one can be inside
    the block at a time) and a correctness one (they must get DIFFERENT
    attempt numbers, never both 0)."""
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    seen_attempts: list = []
    lock_obj = threading.Lock()
    concurrent_entries = []

    def _worker():
        with reserved_attempt(store, stage="screening", work_item_id="wi-shared", timeout=5.0) as attempt:
            with lock_obj:
                concurrent_entries.append(1)
            time.sleep(0.05)   # widen the window a genuine race would need
            with lock_obj:
                concurrent_entries.append(-1)
                assert sum(concurrent_entries) <= 1, "two threads were inside the locked block simultaneously"
            store.publish(_llm_event(work_item_id="wi-shared", attempt=attempt))
            seen_attempts.append(attempt)

    threads = [threading.Thread(target=_worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert sorted(seen_attempts) == [0, 1, 2]


def test_reserved_attempt_raises_on_corrupted_occupied_attempt(tmp_path):
    """A file exists at attempt=0's own event_id path but fails self-
    verification (tampered content) -- reserved_attempt() must raise
    immediately, never silently allocate past a corrupted event and
    treat attempt=1 as the next free slot."""
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    event0 = _llm_event(work_item_id="wi-1", attempt=0)
    store.publish(event0)
    path = store.event_path(event0.event_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["task_purpose"] = "tampered"   # stored semantic_hash/content_hash now stale
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AIEventIntegrityError):
        with reserved_attempt(store, stage="screening", work_item_id="wi-1"):
            pass  # pragma: no cover - must raise before yielding


def test_reserved_attempt_extra_occupied_defaults_to_none_and_changes_nothing(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    with reserved_attempt(store, stage="screening", work_item_id="wi-1") as attempt:
        assert attempt == 0


def test_reserved_attempt_extra_occupied_skips_a_slot_with_no_published_event(tmp_path):
    """amber-quilting-osprey.md sec. 3.4 round-6 fix: an attempt with a
    durable local marker but no published AIUseEvent (e.g. an archived
    interrupted-call marker) must never be handed out again."""
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    with reserved_attempt(store, stage="screening", work_item_id="wi-1", extra_occupied=lambda a: a == 0) as attempt:
        assert attempt == 1


def test_reserved_attempt_extra_occupied_checked_alongside_published_events(tmp_path):
    store = AIUseEventStore(output_dir=tmp_path, review_run_id="run-1")
    with reserved_attempt(store, stage="screening", work_item_id="wi-1") as attempt:
        store.publish(_llm_event(work_item_id="wi-1", attempt=attempt))
    # attempt 0 is published; extra_occupied additionally marks attempt 1 occupied.
    with reserved_attempt(store, stage="screening", work_item_id="wi-1", extra_occupied=lambda a: a == 1) as attempt:
        assert attempt == 2


def test_sanitize_call_error_text_redacts_bearer_and_authorization_and_sk_patterns():
    text = "call failed: Authorization: Bearer sk-abc123XYZ-secret, header Authorization: sk-other-token"
    sanitized = sanitize_call_error_text(text)
    assert "sk-abc123XYZ-secret" not in sanitized
    assert "sk-other-token" not in sanitized
    assert "Bearer sk-abc123XYZ-secret" not in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_call_error_text_redacts_exact_currently_loaded_credential(monkeypatch):
    monkeypatch.setattr(
        "slr_pipeline.config.resolve_openai_api_key", lambda: "not-shaped-like-a-normal-key-at-all",
    )
    text = "connection reset while using key not-shaped-like-a-normal-key-at-all for auth"
    sanitized = sanitize_call_error_text(text)
    assert "not-shaped-like-a-normal-key-at-all" not in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_call_error_text_tolerates_credential_resolution_failure(monkeypatch):
    def _raise():
        raise RuntimeError("no credential configured")

    monkeypatch.setattr("slr_pipeline.config.resolve_openai_api_key", _raise)
    sanitized = sanitize_call_error_text("plain error text with no secrets")
    assert sanitized == "plain error text with no secrets"


def test_sanitize_call_error_text_leaves_ordinary_text_unchanged():
    assert sanitize_call_error_text("timeout after 30s") == "timeout after 30s"
