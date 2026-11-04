"""Regression tests for slr_pipeline.selection_ledger -- Stage 3, Section
3.3/3.5 acceptance criteria for the SelectionDecisionRecord ledger:
chain validation, replay behavior, and concurrent auto-supersession."""
from __future__ import annotations

import json
import threading
import time

import pydantic
import pytest

from slr_pipeline.ai_traceability import kernel_lock
from slr_pipeline.selection_ledger import (
    SelectionDecisionRecord,
    SelectionDecisionStore,
    SelectionLedgerIntegrityError,
    SelectionLedgerLockError,
    SelectionLedgerPreconditionError,
    build_selection_decision_record,
    derive_source_event_id,
    emit_selection_decision,
    source_event_filename,
)


def _record(**overrides) -> SelectionDecisionRecord:
    kwargs = dict(
        review_run_id="run-1", decision_id="dec-1", source_event_id="evt-1",
        record_id="rec-1", stage="title_abstract_screening", decision="exclude",
        decision_maker="ai", reason="off-topic",
    )
    kwargs.update(overrides)
    if kwargs["decision_id"] == "dec-1":
        kwargs["decision_id"] = "11111111-1111-4111-8111-111111111111"
    return build_selection_decision_record(**kwargs)


def _uuid4(tag: str) -> str:
    """Deterministic, valid UUID4-shaped strings for test fixtures that
    need distinct decision_ids without depending on uuid.uuid4()'s
    randomness."""
    import hashlib

    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:12]
    return f"00000000-0000-4000-8000-{digest}"


# ---------------------------------------------------------------------------
# Basic chain: ai decision -> human supersession, tail projection
# ---------------------------------------------------------------------------

def test_ai_then_human_chain_end_to_end(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    ai_record = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )
    human_record = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
    )
    assert human_record.supersedes_decision_id == ai_record.decision_id

    all_records = store.load_all()
    assert len(all_records) == 2

    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision_id == human_record.decision_id


def test_retry_idempotent_same_source_event_id_is_a_noop(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    first = emit_selection_decision(
        store=store, record_id="rec-1", stage="dedup", decision="duplicate",
        decision_maker="rule_based_automation", source_event_id="dedup-evt-1",
    )
    second = emit_selection_decision(
        store=store, record_id="rec-1", stage="dedup", decision="duplicate",
        decision_maker="rule_based_automation", source_event_id="dedup-evt-1",
    )
    assert second.decision_id == first.decision_id
    assert len(store.load_all()) == 1


def test_batch_growth_new_source_event_id_supersedes_prior(tmp_path):
    """Analog of the batch-growth scenario: a genuinely different
    source_event_id (e.g. a new ft_screening_run_id after the confirmed
    batch grows) for the SAME (record_id, stage) produces a SECOND
    record, auto-superseding the first -- one connected two-decision
    chain, not two disconnected roots."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    first = emit_selection_decision(
        store=store, record_id="rec-1", stage="full_text_eligibility", decision="include",
        decision_maker="ai", source_event_id="ftrun-A",
    )
    second = emit_selection_decision(
        store=store, record_id="rec-1", stage="full_text_eligibility", decision="include",
        decision_maker="ai", source_event_id="ftrun-B",
    )
    assert second.supersedes_decision_id == first.decision_id
    tails = store.project_current_tails()
    assert tails[("rec-1", "full_text_eligibility")].decision_id == second.decision_id
    assert len(store.load_all()) == 2


def test_replay_of_superseded_source_event_id_returns_original_unchanged(tmp_path):
    """Direct regression for the A -> H -> replay-A sequence: an ai
    decision A published, a human decision H supersedes it, then
    emit_selection_decision() called a third time with A's own
    source_event_id, proposing the SAME content A originally had --
    confirms no third file is created, the call returns A's own
    original record unchanged, and the tail remains H."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-A",
    )
    h = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-H",
    )
    replay = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-A",
    )
    assert replay.decision_id == a.decision_id
    assert replay.content_hash == a.content_hash
    assert len(store.load_all()) == 2
    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision_id == h.decision_id


# ---------------------------------------------------------------------------
# Replay semantic-mismatch rejection (finding 1)
# ---------------------------------------------------------------------------

def test_replay_with_different_content_on_current_tail_raises(tmp_path):
    """A replay whose source_event_id matches the CURRENT TAIL but whose
    proposed decision/decision_maker/reason differs from what was
    persisted must raise, never silently return the stale original."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1", reason="off-topic",
    )
    with pytest.raises(SelectionLedgerIntegrityError, match="replay"):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="title_abstract_screening",
            decision="include", decision_maker="human", source_event_id="ai-evt-1", reason="override",
        )
    # no new record was created by the failed replay
    assert len(store.load_all()) == 1


def test_replay_with_different_content_on_historical_entry_raises(tmp_path):
    """Same mismatch check, but for a source_event_id that belongs to a
    HISTORICAL (already-superseded) entry, not the current tail -- the
    candidate must be built against that entry's own original
    predecessor, not the current tail, and still raise on mismatch."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-A", reason="off-topic",
    )
    emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-H",
    )
    with pytest.raises(SelectionLedgerIntegrityError, match="replay"):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="title_abstract_screening",
            decision="include", decision_maker="ai", source_event_id="ai-evt-A",  # A's id, different content
        )
    assert len(store.load_all()) == 2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_emit_same_key_different_source_events_serializes_into_one_chain(tmp_path):
    """Direct regression for the fork-via-race finding: two threads call
    emit_selection_decision() concurrently for the SAME
    (review_run_id, record_id, stage) key with two genuinely different
    source_event_ids and no prior tail -- the per-key lock must
    serialize them into ONE linear two-decision chain, never two
    disconnected roots or a fork."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    results = {}
    errors = []

    def _emit(tag, source_event_id):
        try:
            results[tag] = emit_selection_decision(
                store=store, record_id="rec-1", stage="title_abstract_screening",
                decision="exclude", decision_maker="ai", source_event_id=source_event_id,
            )
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    t1 = threading.Thread(target=_emit, args=("a", "evt-A"))
    t2 = threading.Thread(target=_emit, args=("b", "evt-B"))
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not errors
    validated = store.load_validated()  # raises if a fork/disconnected-root snuck through
    assert len(validated) == 2
    tails = store.project_current_tails()
    assert ("rec-1", "title_abstract_screening") in tails
    # exactly one record has supersedes_decision_id=None, the other supersedes it
    roots = [r for r in validated if r.supersedes_decision_id is None]
    assert len(roots) == 1
    tail_ids = {r.decision_id for r in validated} - {r.supersedes_decision_id for r in validated if r.supersedes_decision_id}
    assert len(tail_ids) == 1


def test_concurrent_emit_different_keys_never_block_each_other(tmp_path):
    """Proves per-KEY locking, not just "fast enough to pass under a
    global lock": explicitly holds key A's own lock for longer than the
    assertion window while key B's emit runs concurrently -- B must
    complete well before A's lock is released, which a global lock could
    never satisfy."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    lock_a = store.lock_path_for(record_id="rec-A", stage="title_abstract_screening")
    hold_seconds = 2.0
    release_event = threading.Event()
    holder_acquired = threading.Event()

    def _hold_lock_a():
        with kernel_lock(
            lock_a, error_cls=SelectionLedgerLockError, description="test-hold-key-a", timeout=10.0,
        ):
            holder_acquired.set()
            release_event.wait(hold_seconds)

    holder = threading.Thread(target=_hold_lock_a)
    holder.start()
    assert holder_acquired.wait(timeout=5.0)

    start = time.monotonic()
    emit_selection_decision(
        store=store, record_id="rec-B", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="evt-B", timeout=5.0,
    )
    elapsed = time.monotonic() - start

    release_event.set()
    holder.join(timeout=10.0)

    assert elapsed < hold_seconds  # B must not have waited on A's still-held lock
    assert len(store.load_all()) == 1


# ---------------------------------------------------------------------------
# Batch-level producer disambiguation (finding 4)
# ---------------------------------------------------------------------------

def test_derive_source_event_id_disambiguates_records_sharing_one_batch_producer(tmp_path):
    """screening_node makes one model call per multi-record batch, so one
    AIUseEvent.event_id (the producer_artifact_id) legitimately backs
    several records' decisions. Using that id directly as
    source_event_id would let only the first record's decision persist
    -- derive_source_event_id() must fold in record_id/stage so every
    record in the batch gets its own distinct, stable source_event_id."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    batch_event_id = "aiuse-batch-123"

    published = []
    for record_id in ("rec-1", "rec-2", "rec-3"):
        source_event_id = derive_source_event_id(
            producer_artifact_id=batch_event_id, record_id=record_id, stage="title_abstract_screening",
        )
        published.append(
            emit_selection_decision(
                store=store, record_id=record_id, stage="title_abstract_screening",
                decision="exclude", decision_maker="ai", source_event_id=source_event_id,
                producer_artifact_id=batch_event_id,
            )
        )

    assert len({r.source_event_id for r in published}) == 3
    assert len(store.load_all()) == 3
    for record_id, rec in zip(("rec-1", "rec-2", "rec-3"), published):
        assert store.project_current_tails()[(record_id, "title_abstract_screening")].decision_id == rec.decision_id
    # the raw upstream producer id must stay traceable, not lost inside
    # each record's own (disambiguated, opaque) source_event_id hash
    assert all(rec.producer_artifact_id == batch_event_id for rec in published)

    # deterministic: re-deriving for the same (producer, record, stage) reproduces the identical id
    again = derive_source_event_id(
        producer_artifact_id=batch_event_id, record_id="rec-1", stage="title_abstract_screening",
    )
    assert again == published[0].source_event_id


# ---------------------------------------------------------------------------
# source_event_id -> filename hashing, empty/hostile values (finding 3)
# ---------------------------------------------------------------------------

def test_source_event_filename_rejects_empty():
    with pytest.raises(ValueError):
        source_event_filename("")


def test_dot_prefixed_source_event_id_still_loadable(tmp_path):
    """A source_event_id that LOOKS like a hidden-file name (leading dot)
    must not become permanently invisible to the loader -- the on-disk
    filename is always a hash of source_event_id, never the raw value,
    so this case cannot arise via the normal publish path at all."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    record = emit_selection_decision(
        store=store, record_id="rec-1", stage="dedup", decision="duplicate",
        decision_maker="rule_based_automation", source_event_id=".hidden",
    )
    path = store.record_path(record_id="rec-1", stage="dedup", source_event_id=".hidden")
    assert path.exists()
    assert not path.name.startswith(".")
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].source_event_id == ".hidden"


def test_empty_source_event_id_rejected_at_construction():
    with pytest.raises(pydantic.ValidationError):
        build_selection_decision_record(
            review_run_id="run-1", decision_id=_uuid4("a"), source_event_id="",
            record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
        )


# ---------------------------------------------------------------------------
# decision_id UUID4 contract
# ---------------------------------------------------------------------------

def test_non_uuid4_decision_id_rejected():
    with pytest.raises(pydantic.ValidationError):
        _record(decision_id="not-a-uuid4")


def test_uuid1_shaped_decision_id_rejected():
    # a syntactically valid UUID, but not version 4
    with pytest.raises(pydantic.ValidationError):
        _record(decision_id="11111111-1111-1111-8111-111111111111")


def test_uppercase_uuid4_decision_id_rejected():
    """Regression for the lowercasing bug: the validator's own error
    message claims canonical lowercase is required, but previously
    compared against `self.decision_id.lower()`, which made an
    uppercase-but-otherwise-valid UUID4 string pass silently. Uses a
    UUID containing actual hex letters (a/b) -- an all-digit UUID has no
    case to differ, so it would not exercise this bug at all."""
    with pytest.raises(pydantic.ValidationError):
        _record(decision_id="aabbccdd-aabb-4abc-8abc-aabbccddeeff".upper())


def test_non_uuid4_supersedes_decision_id_rejected():
    """The shared UUID4 validator must cover supersedes_decision_id too,
    not just decision_id -- previously a malformed predecessor id like
    "not-a-uuid4" was accepted into a well-formed record."""
    with pytest.raises(pydantic.ValidationError):
        _record(decision_id=_uuid4("x"), supersedes_decision_id="not-a-uuid4")


def test_uuid1_shaped_supersedes_decision_id_rejected():
    with pytest.raises(pydantic.ValidationError):
        _record(decision_id=_uuid4("x"), supersedes_decision_id="11111111-1111-1111-8111-111111111111")


# ---------------------------------------------------------------------------
# producer_artifact_id <-> source_event_id structural binding (finding 1)
# ---------------------------------------------------------------------------

def test_producer_artifact_id_with_non_derived_source_event_id_rejected_at_construction():
    """A producer_artifact_id paired with an arbitrary, non-derived
    source_event_id must be rejected structurally, on every construction
    path -- not just when going through emit_selection_decision()'s own
    calling convention -- since accepting it would silently conceal
    incorrect audit linkage."""
    with pytest.raises(pydantic.ValidationError):
        build_selection_decision_record(
            review_run_id="run-1", decision_id=_uuid4("a"), source_event_id="arbitrary-not-derived",
            producer_artifact_id="producer-A", record_id="rec-1", stage="dedup",
            decision="duplicate", decision_maker="ai",
        )


def test_producer_artifact_id_with_correctly_derived_source_event_id_accepted():
    source_event_id = derive_source_event_id(
        producer_artifact_id="producer-A", record_id="rec-1", stage="dedup",
    )
    record = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("a"), source_event_id=source_event_id,
        producer_artifact_id="producer-A", record_id="rec-1", stage="dedup",
        decision="duplicate", decision_maker="ai",
    )
    assert record.producer_artifact_id == "producer-A"


def test_replay_claiming_different_producer_artifact_id_rejected(tmp_path):
    """Direct regression for "replaying producer A's event while
    claiming producer_artifact_id='producer-B'": emit_selection_decision()
    must reject a replay whose source_event_id matches an already-
    persisted record but whose producer_artifact_id claim differs, even
    when decision/decision_maker/reason are otherwise identical."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    source_event_id = derive_source_event_id(
        producer_artifact_id="producer-A", record_id="rec-1", stage="dedup",
    )
    emit_selection_decision(
        store=store, record_id="rec-1", stage="dedup", decision="duplicate",
        decision_maker="ai", source_event_id=source_event_id, producer_artifact_id="producer-A",
    )
    with pytest.raises(SelectionLedgerIntegrityError, match="producer"):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="dedup", decision="duplicate",
            decision_maker="ai", source_event_id=source_event_id, producer_artifact_id="producer-B",
        )
    assert len(store.load_all()) == 1


def test_publish_collision_with_producer_artifact_id_mismatch_rejected(tmp_path):
    """Defense-in-depth mirror of the replay check, at the publish()
    collision path directly. producer_artifact_id is deliberately NOT a
    semantic_hash input (the decision's own substance is unchanged by
    which upstream call produced it), so without this check a persisted
    record with producer_artifact_id=None and one claiming
    producer_artifact_id="producer-A" -- otherwise semantically identical
    -- would have the SAME semantic_hash and be silently accepted as a
    benign retry, discarding the producer claim entirely."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    # an arbitrary source_event_id that happens to equal what
    # derive_source_event_id() would produce for producer-A -- legal
    # here since producer_artifact_id=None imposes no derivation
    # constraint on this record.
    shared_source_event_id = derive_source_event_id(
        producer_artifact_id="producer-A", record_id="rec-1", stage="dedup",
    )
    existing = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("a"), source_event_id=shared_source_event_id,
        producer_artifact_id=None, record_id="rec-1", stage="dedup",
        decision="duplicate", decision_maker="ai",
    )
    store.publish(existing)

    colliding = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("b"), source_event_id=shared_source_event_id,
        producer_artifact_id="producer-A", record_id="rec-1", stage="dedup",
        decision="duplicate", decision_maker="ai",
    )
    assert colliding.semantic_hash == existing.semantic_hash  # confirms the gap this test closes
    with pytest.raises(SelectionLedgerIntegrityError, match="producer_artifact_id"):
        store.publish(colliding)


# ---------------------------------------------------------------------------
# Publish-time collision binding (finding 2)
# ---------------------------------------------------------------------------

def _write_record_raw(store: SelectionDecisionStore, record: SelectionDecisionRecord, *, filename: str | None = None) -> None:
    key_dir = store.key_dir(record_id=record.record_id, stage=record.stage)
    key_dir.mkdir(parents=True, exist_ok=True)
    name = filename if filename is not None else source_event_filename(record.source_event_id)
    (key_dir / name).write_text(record.model_dump_json(indent=2), encoding="utf-8")


def test_publish_collision_with_foreign_run_record_rejected(tmp_path):
    """A file sitting at the exact path a new record would publish to,
    whose OWN review_run_id differs from this store's run (but whose
    semantic_hash happens to match, since review_run_id is not itself a
    semantic_hash input) must not be silently accepted as "the same
    record, therefore a benign retry" -- publish() must reject it."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    foreign = build_selection_decision_record(
        review_run_id="run-FOREIGN", decision_id=_uuid4("f"), source_event_id="evt-1",
        record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
    )
    _write_record_raw(store, foreign)

    new_record = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("n"), source_event_id="evt-1",
        record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
    )
    with pytest.raises(SelectionLedgerIntegrityError, match="review_run_id"):
        store.publish(new_record)


def test_publish_collision_with_filename_source_mismatch_rejected(tmp_path):
    """A file at the target path whose OWN source_event_id does not
    match the filename it's stored under (i.e. someone placed record B's
    content at record A's expected path) must be rejected, not silently
    treated as a valid existing record to compare hashes against."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    mismatched = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("m"), source_event_id="evt-OTHER",
        record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
    )
    # place it at the filename for "evt-1", not its own "evt-OTHER"
    _write_record_raw(store, mismatched, filename=source_event_filename("evt-1"))

    new_record = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("n"), source_event_id="evt-1",
        record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
    )
    with pytest.raises(SelectionLedgerIntegrityError, match="filename"):
        store.publish(new_record)


# ---------------------------------------------------------------------------
# Crash safety + emission scaling (finding 5, plus required test hardening)
# ---------------------------------------------------------------------------

def test_crash_between_fsync_and_link_leaves_no_truncated_final_file(tmp_path, monkeypatch):
    """selection_ledger.py duplicates AIUseEventStore's write+fsync-temp
    / atomic-hard-link publisher rather than sharing it -- so it needs
    its own regression for the same crash window, not just a reference
    to the AI-event store's existing test."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    record = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("a"), source_event_id="evt-1",
        record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
    )

    def _boom(*args, **kwargs):
        raise OSError("simulated crash between fsync and hard-link publish")

    monkeypatch.setattr("slr_pipeline.selection_ledger.os.link", _boom)
    with pytest.raises(OSError):
        store.publish(record)

    final_path = store.record_path(record_id="rec-1", stage="dedup", source_event_id="evt-1")
    assert not final_path.exists()
    key_dir = store.key_dir(record_id="rec-1", stage="dedup")
    tmp_files = list(key_dir.glob(".tmp.*"))
    assert len(tmp_files) == 1
    # the temp file itself is fully written (fsync happened before the
    # simulated crash) -- never a truncated/partial write
    persisted = SelectionDecisionRecord.model_validate(json.loads(tmp_files[0].read_text(encoding="utf-8")))
    persisted.verify_hashes()


def test_emission_reads_bounded_files_independent_of_ledger_size(tmp_path, monkeypatch):
    """Non-fragile regression for the quadratic-emission finding --
    counts actual file reads instead of depending on a wall-clock
    threshold. Populates the ledger with many UNRELATED (record_id,
    stage) keys, then emits a decision for a BRAND NEW key and asserts
    the number of record files read stays small and bounded, never
    scaling with the number of unrelated keys already in the ledger
    (which the old load_validated()-per-emission design would have
    read every single time)."""
    import slr_pipeline.selection_ledger as ledger_mod

    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    unrelated_key_count = 150
    for i in range(unrelated_key_count):
        emit_selection_decision(
            store=store, record_id=f"rec-{i}", stage="dedup", decision="duplicate",
            decision_maker="rule_based_automation", source_event_id=f"evt-{i}",
        )
    assert len(store.load_all()) == unrelated_key_count

    read_paths: list = []
    original_load = ledger_mod._load_record_file

    def _counting_load(path):
        read_paths.append(path)
        return original_load(path)

    monkeypatch.setattr(ledger_mod, "_load_record_file", _counting_load)

    emit_selection_decision(
        store=store, record_id="rec-NEW", stage="dedup", decision="duplicate",
        decision_maker="rule_based_automation", source_event_id="evt-NEW",
    )

    # a brand-new key's own partition starts empty, so this should read
    # zero pre-existing files -- well under the 150 unrelated records,
    # which is what actually distinguishes O(chain length) from O(N).
    assert len(read_paths) < 5


# ---------------------------------------------------------------------------
# Loader integrity checks -- malformed fixtures
# ---------------------------------------------------------------------------

def _write_record(store: SelectionDecisionStore, record: SelectionDecisionRecord) -> None:
    _write_record_raw(store, record)


def test_missing_predecessor_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    orphan = _record(
        decision_id=_uuid4("2"), source_event_id="evt-2", supersedes_decision_id=_uuid4("does-not-exist"),
    )
    _write_record(store, orphan)
    with pytest.raises(SelectionLedgerIntegrityError, match="missing predecessor"):
        store.load_validated()


def test_cross_record_supersession_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a = _record(decision_id=_uuid4("a"), source_event_id="evt-A", record_id="rec-1")
    b = _record(
        decision_id=_uuid4("b"), source_event_id="evt-B", record_id="rec-2",  # different record_id
        supersedes_decision_id=_uuid4("a"),
    )
    _write_record(store, a)
    _write_record(store, b)
    with pytest.raises(SelectionLedgerIntegrityError, match="different record_id/stage"):
        store.load_validated()


def test_cross_stage_supersession_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a = _record(decision_id=_uuid4("a"), source_event_id="evt-A", stage="title_abstract_screening")
    b = _record(
        decision_id=_uuid4("b"), source_event_id="evt-B", stage="full_text_eligibility",  # different stage
        supersedes_decision_id=_uuid4("a"),
    )
    _write_record(store, a)
    _write_record(store, b)
    with pytest.raises(SelectionLedgerIntegrityError, match="different record_id/stage"):
        store.load_validated()


def test_cycle_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a = _record(decision_id=_uuid4("a"), source_event_id="evt-A", supersedes_decision_id=_uuid4("b"))
    b = _record(decision_id=_uuid4("b"), source_event_id="evt-B", supersedes_decision_id=_uuid4("a"))
    _write_record(store, a)
    _write_record(store, b)
    with pytest.raises(SelectionLedgerIntegrityError, match="cycle"):
        store.load_validated()


def test_fork_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    root = _record(decision_id=_uuid4("root"), source_event_id="evt-root")
    a = _record(decision_id=_uuid4("a"), source_event_id="evt-A", supersedes_decision_id=_uuid4("root"))
    b = _record(decision_id=_uuid4("b"), source_event_id="evt-B", supersedes_decision_id=_uuid4("root"))
    _write_record(store, root)
    _write_record(store, a)
    _write_record(store, b)
    with pytest.raises(SelectionLedgerIntegrityError, match="fork"):
        store.load_validated()


def test_disconnected_multiple_roots_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a = _record(decision_id=_uuid4("a"), source_event_id="evt-A", supersedes_decision_id=None)
    b = _record(decision_id=_uuid4("b"), source_event_id="evt-B", supersedes_decision_id=None)
    _write_record(store, a)
    _write_record(store, b)
    with pytest.raises(SelectionLedgerIntegrityError, match="multiple roots"):
        store.load_validated()


def test_duplicate_decision_id_mismatched_content_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a = _record(decision_id=_uuid4("a"), source_event_id="evt-A", decision="exclude")
    b = _record(decision_id=_uuid4("a"), source_event_id="evt-B", decision="include")  # same decision_id, different content
    _write_record(store, a)
    _write_record(store, b)
    with pytest.raises(SelectionLedgerIntegrityError, match="duplicate decision_id"):
        store.load_validated()


def test_renamed_copy_rejected(tmp_path):
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    record = emit_selection_decision(
        store=store, record_id="rec-1", stage="dedup", decision="duplicate",
        decision_maker="rule_based_automation", source_event_id="evt-1",
    )
    original_path = store.record_path(record_id=record.record_id, stage=record.stage, source_event_id=record.source_event_id)
    renamed = original_path.parent / "renamed-copy.json"
    renamed.write_text(original_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SelectionLedgerIntegrityError, match="filename"):
        store.load_all()


def test_foreign_run_record_rejected_on_load(tmp_path):
    store_a = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-A")
    foreign = _record(review_run_id="run-B", decision_id=_uuid4("b"), source_event_id="evt-B")
    _write_record_raw(store_a, foreign)
    with pytest.raises(SelectionLedgerIntegrityError, match="review_run_id"):
        store_a.load_all()


def test_timestamp_tampering_then_genuine_collision_self_verifies_first(tmp_path):
    """A persisted record is directly edited to change only its
    timestamp (stored semantic_hash/content_hash left untouched), then a
    genuine collision on its source_event_id is triggered -- the
    collision path in publish() must recompute+verify both hashes
    against the tampered file's CURRENT field values first and raise
    immediately, before ever reaching the semantic_hash retry-comparison."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    original = emit_selection_decision(
        store=store, record_id="rec-1", stage="dedup", decision="duplicate",
        decision_maker="rule_based_automation", source_event_id="evt-1",
    )
    path = store.record_path(record_id=original.record_id, stage=original.stage, source_event_id=original.source_event_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["timestamp"] = "2099-01-01T00:00:00Z"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    colliding = build_selection_decision_record(
        review_run_id="run-1", decision_id=_uuid4("new"), source_event_id="evt-1",
        record_id="rec-1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation",
    )
    with pytest.raises(SelectionLedgerIntegrityError):
        store.publish(colliding)


def test_extra_field_and_frozen(tmp_path):
    record = _record()
    with pytest.raises(pydantic.ValidationError):
        record.decision = "mutated"
    payload = record.model_dump(mode="json")
    payload["sneaky"] = "extra"
    with pytest.raises(pydantic.ValidationError):
        SelectionDecisionRecord.model_validate(payload)


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        SelectionDecisionStore(output_dir=tmp_path, review_run_id="../escaped")
    assert not (tmp_path / "escaped").exists()


# ---------------------------------------------------------------------------
# expected_current_tail_decision_id -- Stage 3 node-instrumentation plan v9,
# F13/F19: a human resolution's precondition, checked atomically inside
# emit_selection_decision()'s own per-key lock. Three exact scenarios from
# the plan's own §4.2, plus the concurrent-tail-change race.
# ---------------------------------------------------------------------------

def test_precondition_matching_current_tail_proceeds(tmp_path):
    """The ordinary case: a human resolution's own expected baseline
    equals the actual current tail -- proceeds and supersedes it."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    ai_record = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )
    human_record = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=ai_record.decision_id,
    )
    assert human_record.supersedes_decision_id == ai_record.decision_id
    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision_id == human_record.decision_id


def test_precondition_none_baseline_requires_empty_chain(tmp_path):
    """A human resolution whose reviewed baseline was `None` (no prior ai/
    rule_based decision ever existed) may only become the chain's own
    root -- if a tail already exists, the precondition (None) does not
    match it and this raises."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    root = emit_selection_decision(
        store=store, record_id="rec-1", stage="full_text_eligibility",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=None,
    )
    assert root.supersedes_decision_id is None

    with pytest.raises(SelectionLedgerPreconditionError):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="full_text_eligibility",
            decision="exclude", decision_maker="human", source_event_id="human-evt-2",
            expected_current_tail_decision_id=None,
        )


def test_stale_state_replay_of_human_decision_is_idempotent_when_still_current(tmp_path):
    """A0 -> H, then a stale-state replay of H's OWN source_event_id
    (H's own resolution resubmitted, e.g. a retried caller) -- H is still
    the tail, so this succeeds idempotently and returns H unchanged."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a0 = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )
    h = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=a0.decision_id,
    )
    replay = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=a0.decision_id,   # same stale-but-still-valid baseline
    )
    assert replay.decision_id == h.decision_id
    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision_id == h.decision_id
    assert len(store.load_all()) == 2   # no third file was created


def test_replay_of_human_decision_after_further_supersession_raises(tmp_path):
    """A0 -> H -> A1 (A1: a genuinely new ai decision, e.g. a changed
    batch fingerprint, legitimately supersedes H). A stale replay of H's
    own source_event_id must NOT be silently reported as current -- A1 is
    the real tail now, and this specific bug (returning H unconditionally)
    is exactly what this plan round's own F19 fix closes."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a0 = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )
    h = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=a0.decision_id,
    )
    a1 = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-2",   # genuinely different content
    )
    assert a1.supersedes_decision_id == h.decision_id

    with pytest.raises(SelectionLedgerPreconditionError):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="title_abstract_screening",
            decision="include", decision_maker="human", source_event_id="human-evt-1",
            expected_current_tail_decision_id=a0.decision_id,
        )

    # The failed replay attempt must leave the ledger exactly as it was --
    # A1 stays current, no new file, no state mutation.
    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision_id == a1.decision_id
    assert len(store.load_all()) == 3


def test_replay_with_no_precondition_stays_unconditional_legacy_behavior(tmp_path):
    """A0 -> H, replay of A0's OWN source_event_id with NO
    expected_current_tail_decision_id (the ordinary, non-human-resolution
    caller -- e.g. a genuine ai/rule_based retry) -- unconditional,
    harmless historical replay; H remains current throughout. This is the
    pre-existing accepted behavior (see
    test_replay_of_superseded_source_event_id_returns_original_unchanged
    above), re-asserted here explicitly alongside the two new precondition
    scenarios so all three of §4.2's exact cases live together."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a0 = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )
    h = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=a0.decision_id,
    )
    replay = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
        # no expected_current_tail_decision_id -- legacy/_UNSET path
    )
    assert replay.decision_id == a0.decision_id
    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision_id == h.decision_id
    assert len(store.load_all()) == 2


def test_precondition_mismatch_on_genuinely_new_content_raises(tmp_path):
    """A precondition mismatch for genuinely NEW content (not a replay of
    anything already in the chain) -- e.g. two concurrent writers both
    trying to supersede the same stale-by-now baseline -- raises before
    any construction/publish."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a0 = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )
    emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="include", decision_maker="human", source_event_id="human-evt-1",
        expected_current_tail_decision_id=a0.decision_id,
    )
    with pytest.raises(SelectionLedgerPreconditionError):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="title_abstract_screening",
            decision="exclude", decision_maker="human", source_event_id="human-evt-2",   # different content
            expected_current_tail_decision_id=a0.decision_id,   # stale -- H already superseded this
        )
    assert len(store.load_all()) == 2   # the failed attempt wrote nothing


def test_concurrent_tail_change_between_preflight_and_emission_raises(tmp_path):
    """A second writer moves the tail (publishes a new ai decision) WHILE
    a human-resolution's own emit_selection_decision() call is waiting to
    acquire the per-key lock -- the atomic, in-lock precondition check
    must catch this race that an earlier, unlocked-only preflight read
    could not (the TOCTOU gap F13 identifies)."""
    store = SelectionDecisionStore(output_dir=tmp_path, review_run_id="run-1")
    a0 = emit_selection_decision(
        store=store, record_id="rec-1", stage="title_abstract_screening",
        decision="exclude", decision_maker="ai", source_event_id="ai-evt-1",
    )

    lock_path = store.lock_path_for(record_id="rec-1", stage="title_abstract_screening")
    release_event = threading.Event()
    started_event = threading.Event()

    def _hold_lock_then_publish_a1():
        with kernel_lock(lock_path, description="test-hold"):
            started_event.set()
            release_event.wait(timeout=5.0)
        emit_selection_decision(
            store=store, record_id="rec-1", stage="title_abstract_screening",
            decision="exclude", decision_maker="ai", source_event_id="ai-evt-2",
        )

    holder = threading.Thread(target=_hold_lock_then_publish_a1)
    holder.start()
    started_event.wait(timeout=5.0)
    # The human resolution's own precondition was read (conceptually) as
    # a0.decision_id BEFORE this race -- release the holder so it
    # publishes a1 first, then let the human resolution's own call
    # proceed against the now-stale a0 baseline.
    release_event.set()
    time.sleep(0.1)   # give the holder a moment to actually publish a1

    with pytest.raises(SelectionLedgerPreconditionError):
        emit_selection_decision(
            store=store, record_id="rec-1", stage="title_abstract_screening",
            decision="include", decision_maker="human", source_event_id="human-evt-1",
            expected_current_tail_decision_id=a0.decision_id,   # now stale -- a1 is the real tail
        )
    holder.join(timeout=5.0)

    tails = store.project_current_tails()
    assert tails[("rec-1", "title_abstract_screening")].decision == "exclude"
    assert tails[("rec-1", "title_abstract_screening")].decision_maker == "ai"
