"""Shared selection-decision ledger -- Stage 3 (`quality_appraisal_synthesis_plan_2026-08-22.md`
Section 3.3, "Selection-decision ledger -- closes R1"). Records every
dedup/title-abstract-screening/full-text-eligibility decision as an
append-only, content-addressed, chain-linked history so a record's
CURRENT decision at a stage is always the well-defined tail of a single
connected chain, never ambiguous between a superseded AI decision and a
later human override.

Reuses `slr_pipeline.ai_traceability`'s hashing conventions and
`kernel_lock()` rather than reinventing either -- this module owns only
the ledger's own record shape, store, chain-validation/projection, and
the single `emit_selection_decision()` entry point every node calls.

On-disk layout is partitioned by `(record_id, stage)`:
`selection_decisions/<review_run_id>/<key_digest>/<source_event_hash>.json`.
`emit_selection_decision()` therefore only ever loads and validates the
one partition its own key touches (`SelectionDecisionStore.load_key()`),
never the whole run -- keeping emission cost independent of total ledger
size. Whole-run validation (`load_validated()`/`load_all()`, used for
reporting/export and for the duplicate-`decision_id` cross-key check)
still walks every partition; it is deliberately not on emission's own
hot path.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator

from slr_pipeline.ai_traceability import AIEventLockError, kernel_lock
from slr_pipeline.extraction.hashing import sha256_hex

import time


class SelectionLedgerIntegrityError(RuntimeError):
    """A persisted `SelectionDecisionRecord` file (or the chain it is
    part of) failed self-verification or structural validation -- never
    silently trusted, worked around, or dropped."""


class SelectionLedgerLockError(AIEventLockError):
    """Could not acquire the per-`(review_run_id, record_id, stage)`
    publish lock within the timeout. Subclasses `AIEventLockError` so
    callers that already catch the AI-event lock error also catch this
    one, while still being independently distinguishable."""


class SelectionLedgerPreconditionError(RuntimeError):
    """A caller-supplied `expected_current_tail_decision_id` precondition
    (see `emit_selection_decision()`) did not hold: either the tail moved
    before this call's own per-key lock was acquired (a genuine race
    against another writer), or an exact `source_event_id` match's own
    historical record is no longer the current tail (it was superseded
    since the caller -- typically a human reviewing a queue entry -- last
    saw it). An ordinary, EXPECTED operational condition -- the caller
    should re-read current state and retry/re-review -- never evidence of
    corruption or tampering. Deliberately NOT a `SelectionLedgerIntegrityError`
    subclass, even though both are raised from the same module: that class
    means "the persisted data itself is untrustworthy"; this one means
    "the data is fine, but the world moved since you last looked at it.\""""


_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_UNSET = object()

_SEMANTIC_HASH_FIELDS = ("record_id", "stage", "decision", "decision_maker", "reason", "supersedes_decision_id")
_CONTENT_HASH_EXCLUDED_FIELDS = {"semantic_hash", "content_hash"}


def source_event_filename(source_event_id: str) -> str:
    """The persisted filename for a record is always the sha256 hex of
    its own `source_event_id`, never the raw string. `source_event_id`
    is caller-controlled (a batch `AIUseEvent.event_id`, an
    `ft_screening_run_id`, ...); using it verbatim as a filename would
    let a pathological value (empty string, a dot-prefixed name the
    loader's own temp-file skip rule would then hide forever, path
    separators, ...) corrupt or permanently hide a published record.
    Raises on an empty `source_event_id` rather than silently hashing
    `""` into a valid-looking filename."""
    if not source_event_id:
        raise ValueError("source_event_id must be a non-empty string")
    return sha256_hex(source_event_id) + ".json"


def _key_digest(record_id: str, stage: str) -> str:
    return sha256_hex({"record_id": record_id, "stage": stage})[:32]


def _validate_canonical_uuid4(value: str, *, field_name: str) -> None:
    """Shared UUID4-shape validator -- covers both `decision_id`
    (required) and `supersedes_decision_id` (optional, checked by the
    caller only when not `None`), so a malformed identifier can never
    slip in through one field just because only the other was checked.
    Compares against `str(parsed)` directly, NOT `value.lower()` --
    lowercasing the caller's own value before comparing would make an
    uppercase-but-otherwise-valid UUID4 string pass despite the error
    message's own claim that canonical lowercase is required."""
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} {value!r} is not a valid UUID string") from exc
    if parsed.version != 4:
        raise ValueError(f"{field_name} {value!r} is not a UUID4 (parsed version={parsed.version})")
    if str(parsed) != value:
        raise ValueError(f"{field_name} {value!r} is not a canonical lowercase UUID4 string")


def derive_source_event_id(*, producer_artifact_id: str, record_id: str, stage: str) -> str:
    """The one shared derivation for turning a (possibly batch-level)
    producer-artifact id into a per-`(record_id, stage)`
    `source_event_id`. `screening_node`'s own model calls are one call
    per multi-record batch, while `AIUseEvent` is correctly one event
    per actual call -- so a single `AIUseEvent.event_id` can legitimately
    be the `producer_artifact_id` behind SEVERAL records' decisions.
    Reusing that event id directly AS `source_event_id` would let only
    the first record's decision persist (every other record in the same
    batch would collide on the identical filename). Every node MUST
    derive `source_event_id` through this function rather than inventing
    its own scheme."""
    return "src-" + sha256_hex(
        {"producer_artifact_id": producer_artifact_id, "record_id": record_id, "stage": stage}
    )


class SelectionDecisionRecord(BaseModel):
    """`decision_id` is a fresh UUID4 minted by the node making the
    decision at the moment it's made -- a human-loggable unique tag, but
    NOT what retry-idempotency keys on (`source_event_id` is). See
    Section 3.3 for the full field-by-field rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_run_id: str
    decision_id: str
    source_event_id: str
    producer_artifact_id: Optional[str] = None
    record_id: str
    stage: Literal["dedup", "title_abstract_screening", "full_text_eligibility"]
    decision: str
    decision_maker: Literal["ai", "human", "rule_based_automation"]
    reason: Optional[str] = None
    supersedes_decision_id: Optional[str] = None
    timestamp: str
    semantic_hash: str
    content_hash: str

    @model_validator(mode="after")
    def _check_uuid4_fields(self) -> "SelectionDecisionRecord":
        """The plan's own UUID4 contract, enforced on EVERY construction
        path including `model_validate()` on load, for BOTH `decision_id`
        (required) and `supersedes_decision_id` (optional -- checked only
        when not `None`) via the one shared `_validate_canonical_uuid4()`
        helper. A malformed or non-UUID4 identifier in either field (e.g.
        `"not-a-uuid4"`) must never be silently accepted into the
        ledger."""
        _validate_canonical_uuid4(self.decision_id, field_name="decision_id")
        if self.supersedes_decision_id is not None:
            _validate_canonical_uuid4(self.supersedes_decision_id, field_name="supersedes_decision_id")
        return self

    @model_validator(mode="after")
    def _check_source_event_id_non_empty(self) -> "SelectionDecisionRecord":
        if not self.source_event_id:
            raise ValueError("source_event_id must be a non-empty string")
        return self

    @model_validator(mode="after")
    def _check_producer_artifact_id_binding(self) -> "SelectionDecisionRecord":
        """When `producer_artifact_id` is set, `source_event_id` MUST be
        exactly `derive_source_event_id(producer_artifact_id, record_id,
        stage)` -- structurally, on every construction path, not just
        the `emit_selection_decision()` calling convention. Without this,
        an arbitrary non-derived `source_event_id` could be paired with a
        `producer_artifact_id` claim that has no real relationship to it,
        silently concealing incorrect audit linkage. `producer_artifact_id
        =None` is unconstrained (a single-record producer with no batch
        disambiguation need not use `derive_source_event_id()` at all)."""
        if self.producer_artifact_id is not None:
            expected = derive_source_event_id(
                producer_artifact_id=self.producer_artifact_id, record_id=self.record_id, stage=self.stage,
            )
            if self.source_event_id != expected:
                raise ValueError(
                    f"producer_artifact_id={self.producer_artifact_id!r} is set, so source_event_id must "
                    f"equal derive_source_event_id(producer_artifact_id={self.producer_artifact_id!r}, "
                    f"record_id={self.record_id!r}, stage={self.stage!r}) = {expected!r}, but got "
                    f"{self.source_event_id!r} instead"
                )
        return self

    def compute_semantic_hash(self) -> str:
        payload = {name: getattr(self, name) for name in _SEMANTIC_HASH_FIELDS}
        return sha256_hex(payload)

    def compute_content_hash(self) -> str:
        payload = {
            name: value
            for name, value in self.model_dump(mode="json").items()
            if name not in _CONTENT_HASH_EXCLUDED_FIELDS
        }
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected_semantic = self.compute_semantic_hash()
        if self.semantic_hash != expected_semantic:
            raise SelectionLedgerIntegrityError(
                f"SelectionDecisionRecord decision_id={self.decision_id!r} failed semantic_hash "
                f"self-verification: stored {self.semantic_hash!r} != recomputed {expected_semantic!r} "
                "-- file corrupted or tampered"
            )
        expected_content = self.compute_content_hash()
        if self.content_hash != expected_content:
            raise SelectionLedgerIntegrityError(
                f"SelectionDecisionRecord decision_id={self.decision_id!r} failed content_hash "
                f"self-verification: stored {self.content_hash!r} != recomputed {expected_content!r} -- "
                "file corrupted or tampered (a timestamp-only edit would be caught here even though "
                "semantic_hash deliberately excludes timestamp)"
            )


def build_selection_decision_record(
    *, review_run_id: str, decision_id: str, source_event_id: str, record_id: str, stage: str,
    decision: str, decision_maker: str, reason: Optional[str] = None,
    supersedes_decision_id: Optional[str] = None, timestamp: Optional[str] = None,
    producer_artifact_id: Optional[str] = None,
) -> SelectionDecisionRecord:
    """`producer_artifact_id`, when given, is stored verbatim (never
    hashed) as the audit-trail link back to whatever upstream artifact
    produced this decision (an `AIUseEvent.event_id`, an
    `ft_screening_run_id`, ...) -- distinct from `source_event_id`, which
    is the (possibly `derive_source_event_id()`-disambiguated) retry-
    idempotency key. It deliberately does NOT participate in
    `semantic_hash` (the decision's own substance -- what/who/why -- is
    unchanged by which upstream call happened to produce it)."""
    ts = timestamp if timestamp is not None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base = dict(
        review_run_id=review_run_id, decision_id=decision_id, source_event_id=source_event_id,
        producer_artifact_id=producer_artifact_id,
        record_id=record_id, stage=stage, decision=decision, decision_maker=decision_maker,
        reason=reason, supersedes_decision_id=supersedes_decision_id, timestamp=ts,
    )
    semantic_hash = SelectionDecisionRecord.model_construct(**base, semantic_hash="", content_hash="").compute_semantic_hash()
    with_semantic = dict(base, semantic_hash=semantic_hash)
    content_hash = SelectionDecisionRecord.model_construct(**with_semantic, content_hash="").compute_content_hash()
    return SelectionDecisionRecord.model_validate(dict(with_semantic, content_hash=content_hash))


def _load_record_file(path: Path) -> SelectionDecisionRecord:
    import json

    return SelectionDecisionRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))


class SelectionDecisionStore:
    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        """Same safe-token-plus-resolved-containment discipline as
        `AIUseEventStore`/`QACheckpointStore` -- `review_run_id` must
        never be trusted as an already-safe path component."""
        decisions_root = Path(output_dir) / "selection_decisions"
        if not _SAFE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        run_dir = decisions_root / review_run_id
        if not run_dir.resolve().is_relative_to(decisions_root.resolve()):
            raise ValueError(f"review_run_id {review_run_id!r} resolves outside the selection_decisions root -- refusing to construct store")
        self.review_run_id = review_run_id
        self.decisions_dir = run_dir

    def key_dir(self, *, record_id: str, stage: str) -> Path:
        return self.decisions_dir / _key_digest(record_id, stage)

    def lock_path_for(self, *, record_id: str, stage: str) -> Path:
        return self.decisions_dir / "locks" / f"{_key_digest(record_id, stage)}.lock"

    def record_path(self, *, record_id: str, stage: str, source_event_id: str) -> Path:
        return self.key_dir(record_id=record_id, stage=stage) / source_event_filename(source_event_id)

    def _load_and_verify(
        self, path: Path, *, expected_record_id: Optional[str] = None,
        expected_stage: Optional[str] = None, expected_source_event_id: Optional[str] = None,
        expected_producer_artifact_id: object = _UNSET,
    ) -> SelectionDecisionRecord:
        """The ONE binding verifier for a persisted record, reused
        identically by `load_all()`/`load_key()` (aggregation) and
        `publish()`'s `FileExistsError` collision path -- both must apply
        the same discipline (hash self-verification, this store's own
        `review_run_id`, filename-vs-`source_event_id` identity, and
        optionally the caller's own expected identity) before a
        persisted file's content is trusted for ANY further comparison,
        including the semantic-hash equality check that decides whether
        a collision is a benign retry or a real conflict. `expected_
        producer_artifact_id` defaults to the `_UNSET` sentinel (not
        checked) rather than `None`, since `None` is itself a valid,
        meaningful value to assert (a caller collision-checking a record
        with no producer)."""
        record = _load_record_file(path)
        record.verify_hashes()
        if record.review_run_id != self.review_run_id:
            raise SelectionLedgerIntegrityError(
                f"record at {path} has review_run_id={record.review_run_id!r}, expected this store's own "
                f"{self.review_run_id!r} -- refusing to trust a foreign-run record found in this run's "
                "directory tree"
            )
        expected_filename = source_event_filename(record.source_event_id)
        if path.name != expected_filename:
            raise SelectionLedgerIntegrityError(
                f"record at {path} has source_event_id={record.source_event_id!r}, whose deterministic "
                f"filename is {expected_filename!r}, which does not match this file's own name -- refusing "
                "to trust a renamed, mismatched, or corrupted file"
            )
        if expected_source_event_id is not None and record.source_event_id != expected_source_event_id:
            raise SelectionLedgerIntegrityError(
                f"record at {path} has source_event_id={record.source_event_id!r}, expected "
                f"{expected_source_event_id!r} -- refusing to trust a record found at the wrong path"
            )
        if expected_producer_artifact_id is not _UNSET and record.producer_artifact_id != expected_producer_artifact_id:
            raise SelectionLedgerIntegrityError(
                f"record at {path} has producer_artifact_id={record.producer_artifact_id!r}, expected "
                f"{expected_producer_artifact_id!r} -- refusing to trust a record whose audit-linked "
                "producer does not match the one being collision-checked against"
            )
        if expected_record_id is not None and record.record_id != expected_record_id:
            raise SelectionLedgerIntegrityError(
                f"record at {path} has record_id={record.record_id!r}, expected {expected_record_id!r} -- "
                "refusing to trust a record found under the wrong partition"
            )
        if expected_stage is not None and record.stage != expected_stage:
            raise SelectionLedgerIntegrityError(
                f"record at {path} has stage={record.stage!r}, expected {expected_stage!r} -- refusing to "
                "trust a record found under the wrong partition"
            )
        return record

    def publish(self, record: SelectionDecisionRecord) -> SelectionDecisionRecord:
        """Low-level, write+fsync-temp / atomic-hard-link publish -- same
        mechanism as `AIUseEventStore.publish()`. This should only ever
        be called from within `emit_selection_decision()`'s own per-key
        lock; a genuine `FileExistsError` here should be structurally
        near-impossible (the lock plus the full-chain search already
        confirmed no record exists for this `source_event_id`), but is
        still handled with the same self-verify-then-bind-then-compare
        discipline as defense in depth, never a bare field-value
        comparison against an unverified file."""
        record.verify_hashes()
        if record.review_run_id != self.review_run_id:
            raise SelectionLedgerIntegrityError(
                f"refusing to publish record with review_run_id={record.review_run_id!r} into the store "
                f"for review_run_id={self.review_run_id!r} -- a record may only be published into its own "
                "run's store"
            )
        target_dir = self.key_dir(record_id=record.record_id, stage=record.stage)
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / source_event_filename(record.source_event_id)
        tmp_path = target_dir / f".tmp.{uuid.uuid4().hex}.json"
        payload = record.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, final_path)
        except FileExistsError:
            existing = self._load_and_verify(
                final_path, expected_record_id=record.record_id, expected_stage=record.stage,
                expected_source_event_id=record.source_event_id,
                expected_producer_artifact_id=record.producer_artifact_id,
            )
            if existing.semantic_hash == record.semantic_hash:
                tmp_path.unlink(missing_ok=True)
                return existing
            raise SelectionLedgerIntegrityError(
                f"SelectionDecisionRecord collision for source_event_id={record.source_event_id!r}: "
                f"persisted semantic_hash {existing.semantic_hash!r} != new record's {record.semantic_hash!r}. "
                "This should be structurally unreachable under the per-key lock plus full-chain search -- "
                f"leaving the temp file at {tmp_path} for forensic inspection rather than deleting evidence."
            )
        else:
            tmp_path.unlink(missing_ok=True)
            return record

    def load_all(self) -> List[SelectionDecisionRecord]:
        """Walks every `(record_id, stage)` partition under this run's
        directory. Self-verifies every record (hashes, run binding,
        filename-vs-`source_event_id` identity, and that the record
        actually lives under the partition its own `record_id`/`stage`
        hash to) and rejects a duplicate `source_event_id` within a
        partition -- but does NOT run the cross-partition chain-structure
        integrity checks (missing predecessor, cycles, forks, duplicate
        `decision_id`, ...). Use `load_validated()` for that. Kept
        separate so a caller that only wants "every record that exists,
        individually trustworthy" doesn't pay for (or get blocked by) a
        full-ledger structural validation it doesn't need."""
        if not self.decisions_dir.is_dir():
            return []
        records: List[SelectionDecisionRecord] = []
        seen_by_key: Dict[Tuple[str, str], set] = {}
        for key_subdir in sorted(p for p in self.decisions_dir.iterdir() if p.is_dir() and p.name != "locks"):
            for path in sorted(key_subdir.glob("*.json")):
                if path.name.startswith("."):
                    continue
                record = self._load_and_verify(path)
                expected_dir = self.key_dir(record_id=record.record_id, stage=record.stage)
                if key_subdir.resolve() != expected_dir.resolve():
                    raise SelectionLedgerIntegrityError(
                        f"record at {path} (record_id={record.record_id!r}, stage={record.stage!r}) is "
                        f"stored under partition directory {key_subdir}, but its own record_id/stage hash "
                        f"to {expected_dir} -- refusing to trust a record found in the wrong partition"
                    )
                key = (record.record_id, record.stage)
                seen = seen_by_key.setdefault(key, set())
                if record.source_event_id in seen:
                    raise SelectionLedgerIntegrityError(
                        f"duplicate source_event_id {record.source_event_id!r} found across multiple files "
                        f"in {key_subdir}"
                    )
                seen.add(record.source_event_id)
                records.append(record)
        return records

    def load_key(self, *, record_id: str, stage: str) -> List[SelectionDecisionRecord]:
        """Loads and chain-validates ONLY the one `(record_id, stage)`
        partition -- the operation `emit_selection_decision()` actually
        needs on every call. Cost is bounded by that key's own chain
        length, never by the whole ledger's size (the fix for the
        emission-quadratic-in-ledger-size finding: previously every
        `emit_selection_decision()` call ran a full `load_validated()`
        over the entire run)."""
        key_subdir = self.key_dir(record_id=record_id, stage=stage)
        if not key_subdir.is_dir():
            return []
        records: List[SelectionDecisionRecord] = []
        seen: set = set()
        for path in sorted(key_subdir.glob("*.json")):
            if path.name.startswith("."):
                continue
            record = self._load_and_verify(path, expected_record_id=record_id, expected_stage=stage)
            if record.source_event_id in seen:
                raise SelectionLedgerIntegrityError(
                    f"duplicate source_event_id {record.source_event_id!r} found across multiple files in "
                    f"{key_subdir}"
                )
            seen.add(record.source_event_id)
            records.append(record)
        _validate_chain_structure(records)
        return records

    def load_validated(self) -> List[SelectionDecisionRecord]:
        """`load_all()` plus the full chain-structure integrity checks
        (Section 3.3's "Loader integrity checks, made explicit"):
        duplicate `decision_id` (checked globally, across partitions --
        the one integrity property `load_key()`'s partition-local
        validation cannot see), missing predecessor, cross-record/
        stage/run supersession, cycles, forks, and disconnected multiple
        roots -- each raises `SelectionLedgerIntegrityError` rather than
        silently working around or dropping data. Only after all of
        these pass is the "chain's own tail is current" projection rule
        (`project_current_tails`) meaningful to run. This is the
        whole-run validation path (reporting/export), deliberately not
        on `emit_selection_decision()`'s own hot path."""
        records = self.load_all()
        _validate_chain_structure(records)
        return records

    def project_current_tails(self) -> Dict[Tuple[str, str], SelectionDecisionRecord]:
        """Validates the whole ledger, then returns, for every
        `(record_id, stage)` key that has at least one decision, the
        chain's own tail (the entry nothing supersedes) -- the single
        CURRENT decision for that key, per the event-history semantics
        in Section 3.3 (a record's decision for a stage can genuinely
        change over the run's life; only the tail counts toward any
        current-state aggregation)."""
        records = self.load_validated()
        return _tails_by_key(records)


def _validate_chain_structure(records: List[SelectionDecisionRecord]) -> None:
    # Duplicate decision_id check doubles as building the global lookup
    # map every later check needs -- a duplicate here would make any
    # predecessor lookup ambiguous, so this must happen first.
    by_decision_id: Dict[str, SelectionDecisionRecord] = {}
    for record in records:
        if record.decision_id in by_decision_id:
            other = by_decision_id[record.decision_id]
            raise SelectionLedgerIntegrityError(
                f"duplicate decision_id {record.decision_id!r} found in two different records "
                f"(source_event_id={other.source_event_id!r} and source_event_id={record.source_event_id!r}) "
                "with different content -- a loader-time integrity failure, never silently merged or "
                "arbitrarily picked between"
            )
        by_decision_id[record.decision_id] = record

    # Missing predecessor + cross-record/stage/run supersession: every
    # supersedes_decision_id must resolve to a predecessor that (a)
    # exists among the loaded records and (b) shares this record's own
    # record_id/stage (review_run_id is already uniform -- load_all()
    # already rejected any foreign-run record).
    for record in records:
        if record.supersedes_decision_id is None:
            continue
        predecessor = by_decision_id.get(record.supersedes_decision_id)
        if predecessor is None:
            raise SelectionLedgerIntegrityError(
                f"record decision_id={record.decision_id!r} (record_id={record.record_id!r}, "
                f"stage={record.stage!r}) claims supersedes_decision_id={record.supersedes_decision_id!r}, "
                "but no loaded record has that decision_id -- missing predecessor"
            )
        if predecessor.record_id != record.record_id or predecessor.stage != record.stage:
            raise SelectionLedgerIntegrityError(
                f"record decision_id={record.decision_id!r} (record_id={record.record_id!r}, "
                f"stage={record.stage!r}) claims to supersede decision_id={record.supersedes_decision_id!r}, "
                f"which belongs to a different record_id/stage ({predecessor.record_id!r}/{predecessor.stage!r}) "
                "-- a decision may only supersede another decision for the identical (record_id, stage)"
            )

    # From here on every supersedes_decision_id edge is known to stay
    # within its own (record_id, stage) group, so cycle/fork/root checks
    # can work purely group-by-group.
    groups: Dict[Tuple[str, str], List[SelectionDecisionRecord]] = {}
    for record in records:
        groups.setdefault((record.record_id, record.stage), []).append(record)

    for key, group in groups.items():
        # Cycles: from every node, walk supersedes_decision_id links; the
        # walk must reach None within at most len(group) steps without
        # revisiting a node -- otherwise it loops.
        for start in group:
            visited: set[str] = set()
            current: Optional[SelectionDecisionRecord] = start
            steps = 0
            while current is not None:
                if current.decision_id in visited:
                    raise SelectionLedgerIntegrityError(
                        f"cycle detected in the supersession chain for (record_id={key[0]!r}, stage={key[1]!r}) "
                        f"starting from decision_id={start.decision_id!r} -- following supersedes_decision_id "
                        "links never reaches a root"
                    )
                visited.add(current.decision_id)
                steps += 1
                if steps > len(group):
                    raise SelectionLedgerIntegrityError(
                        f"supersession chain for (record_id={key[0]!r}, stage={key[1]!r}) starting from "
                        f"decision_id={start.decision_id!r} did not terminate within {len(group)} steps -- "
                        "cycle detected"
                    )
                current = by_decision_id.get(current.supersedes_decision_id) if current.supersedes_decision_id else None

        # Forks: at most one record may claim to supersede any given
        # decision_id.
        superseded_by: Dict[str, SelectionDecisionRecord] = {}
        for record in group:
            if record.supersedes_decision_id is None:
                continue
            if record.supersedes_decision_id in superseded_by:
                other = superseded_by[record.supersedes_decision_id]
                raise SelectionLedgerIntegrityError(
                    f"fork detected for (record_id={key[0]!r}, stage={key[1]!r}): both "
                    f"decision_id={other.decision_id!r} and decision_id={record.decision_id!r} claim to "
                    f"supersede the same decision_id={record.supersedes_decision_id!r}"
                )
            superseded_by[record.supersedes_decision_id] = record

        # Disconnected multiple roots: more than one entry with
        # supersedes_decision_id=None in the same group.
        roots = [r for r in group if r.supersedes_decision_id is None]
        if len(roots) > 1:
            raise SelectionLedgerIntegrityError(
                f"disconnected multiple roots detected for (record_id={key[0]!r}, stage={key[1]!r}): "
                f"{len(roots)} records with supersedes_decision_id=None "
                f"({[r.decision_id for r in roots]}) -- exactly one root is required per key"
            )
        if len(roots) == 0:
            raise SelectionLedgerIntegrityError(
                f"no root found for (record_id={key[0]!r}, stage={key[1]!r}) among {len(group)} record(s) "
                "-- every non-empty chain must have exactly one entry with supersedes_decision_id=None"
            )


def _tails_by_key(records: List[SelectionDecisionRecord]) -> Dict[Tuple[str, str], SelectionDecisionRecord]:
    """Assumes `records` already passed `_validate_chain_structure` --
    each (record_id, stage) group is a single connected chain with
    exactly one root, so the tail is simply the one entry nothing
    supersedes."""
    superseded_ids = {r.supersedes_decision_id for r in records if r.supersedes_decision_id is not None}
    groups: Dict[Tuple[str, str], List[SelectionDecisionRecord]] = {}
    for record in records:
        groups.setdefault((record.record_id, record.stage), []).append(record)
    tails: Dict[Tuple[str, str], SelectionDecisionRecord] = {}
    for key, group in groups.items():
        candidates = [r for r in group if r.decision_id not in superseded_ids]
        # _validate_chain_structure's cycle/fork/root checks already
        # guarantee this is exactly one record.
        tails[key] = candidates[0]
    return tails


def is_dedup_duplicate_decision(decision: str) -> bool:
    """`dedup_node`'s own real decision vocabulary is exactly `"unique"` /
    `"duplicate"` (see `nodes/dedup.py`'s own ledger emission) -- this
    predicate is the ONE place that vocabulary's "excluded from the
    corpus" meaning is named, so the PRISMA split computation (reporting
    phase) and any other caller never re-derive it independently."""
    return decision == "duplicate"


def is_screening_exclusion_decision(decision: str) -> bool:
    """`title_abstract_screening`'s ledger `decision` field is the
    OPERATIONALLY FINAL value (INV-09, `nodes/screening.py`) -- raw
    `"borderline"` already collapses to `"include"` in automated mode
    before it ever reaches the ledger, so `"exclude"` is the one value
    meaning genuinely excluded here."""
    return decision == "exclude"


def is_fulltext_exclusion_decision(decision: str) -> bool:
    """`full_text_eligibility`'s ledger `decision` field is FT_SCREENING's
    own `outcome.decision` (ai) or a human resolution's `action`
    (human) -- both vocabularies agree that `"exclude"` is the one value
    meaning genuinely excluded (`nodes/fulltext.py`)."""
    return decision == "exclude"


def ledger_automation_origin_and_oversight(
    ledger_store: "SelectionDecisionStore", stage: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Shared `automation_origin`/`oversight_mode` re-derivation for any
    ledger-populating stage (`dedup`, `title_abstract_screening`,
    `full_text_eligibility`) -- extracted verbatim from
    `nodes/fulltext.py`'s own `_full_text_origin_and_oversight()`
    (aggregation/export/reporting plan v17, §2.8), generalized over
    `stage` instead of hardcoding `full_text_eligibility`. Returns
    `(None, None)` when this stage has no ledger entries at all this run
    -- absent, never defaulted.

    `automation_origin` and `oversight_mode` deliberately read from TWO
    DIFFERENT scopes: `automation_origin` is a PROVENANCE question --
    which mechanism(s) EVER produced an automated decision for this stage
    this run -- so it is derived from ALL of the stage's ledger entries,
    not just the current tail (a tails-only derivation would report
    "no_automated_output" after a genuine `ai -> human` chain, even
    though an `ai` decision genuinely was produced and later superseded).
    `oversight_mode` IS a current-state question ("is a human decision
    backing this record right now") and correctly stays tail-based."""
    all_records = [r for r in ledger_store.load_validated() if r.stage == stage]
    if not all_records:
        return None, None

    non_human_makers = {r.decision_maker for r in all_records if r.decision_maker != "human"}
    if not non_human_makers:
        origin = "no_automated_output"
    elif non_human_makers == {"ai"}:
        origin = "ai"
    elif non_human_makers == {"rule_based_automation"}:
        origin = "rule_based_automation"
    else:
        origin = "mixed"

    tails = _tails_by_key(all_records)
    oversight = "human_resolved" if any(r.decision_maker == "human" for r in tails.values()) else "automation_unreviewed"
    return origin, oversight


def emit_selection_decision(
    *, store: SelectionDecisionStore, record_id: str, stage: str, decision: str,
    decision_maker: str, source_event_id: str, reason: Optional[str] = None,
    producer_artifact_id: Optional[str] = None, timeout: float = 30.0,
    expected_current_tail_decision_id: object = _UNSET,
) -> SelectionDecisionRecord:
    """The ONE place any node constructs and publishes a
    `SelectionDecisionRecord` -- every stage calls this rather than
    building/publishing records independently, so retry-discard and
    auto-supersession are structural guarantees everywhere this ledger
    is populated. The entire lookup-construct-publish sequence runs
    under one per-`(review_run_id, record_id, stage)` lock, acquired
    before the lookup and held through publish's own success/no-op/raise
    outcome -- see Section 3.3 for why the lookup and publish must be
    one atomic transaction (a genuine race between two concurrent,
    different source events for the same key could otherwise both read
    the same tail and both publish, forking the chain).

    The lookup itself only ever loads THIS key's own partition
    (`store.load_key()`), not the whole ledger -- emission cost is
    bounded by this key's own chain length, independent of total ledger
    size.

    `producer_artifact_id`, when given, is recorded verbatim as the
    audit-trail link back to whatever upstream artifact produced this
    decision (an `AIUseEvent.event_id`, an `ft_screening_run_id`, ...).
    For a batch-level producer backing SEVERAL records' decisions (e.g.
    one `screening_node` LLM call deciding many records at once), pass
    `source_event_id=derive_source_event_id(producer_artifact_id=...,
    record_id=record_id, stage=stage)` so each record still gets its own
    distinct, stable, non-colliding retry key, while `producer_artifact_id`
    itself keeps the raw upstream id traceable rather than lost inside a
    hash.

    `expected_current_tail_decision_id` (Stage 3 node-instrumentation,
    default `_UNSET` meaning "no precondition, legacy behavior"): a human
    resolution passes the `decision_id` of the ledger entry it actually
    reviewed (or `None` for a fresh root) here. Checked ATOMICALLY, inside
    this same per-key lock, both against a fresh non-replay tail read AND
    against an exact `source_event_id` replay's own historical match --
    an unlocked, checked-then-raced precondition read (e.g. a preflight
    convenience check before this call) cannot close this gap by itself,
    since another writer can move the tail in between. See the module-
    level design note in `human_resolution.py` for why existence of the
    baseline in the chain is not sufficient -- it must still equal the
    CURRENT tail, or the call must be an exact, already-applied replay of
    that same resolution."""
    lock_path = store.lock_path_for(record_id=record_id, stage=stage)
    with kernel_lock(
        lock_path, error_cls=SelectionLedgerLockError,
        description=f"selection_decision:{record_id}:{stage}", timeout=timeout,
    ):
        group = store.load_key(record_id=record_id, stage=stage)
        tail = _tails_by_key(group).get((record_id, stage)) if group else None

        # Step 1: search the ENTIRE chain for this key for this exact
        # source_event_id, not just the current tail -- an old event can
        # be replayed after a newer one (human or machine) has already
        # superseded it, and that replay must return the ORIGINAL record
        # unchanged, never mint a new one superseding the current tail.
        # A replay is only a benign no-op if the PROPOSED decision is
        # semantically identical to what was persisted originally -- a
        # genuine replay of the same event can never legitimately
        # propose different content, so the candidate is constructed
        # against the historical record's OWN original predecessor
        # (never the current tail, which may have moved on) and compared
        # by semantic_hash before trusting the early return.
        for existing in group:
            if existing.source_event_id == source_event_id:
                if existing.producer_artifact_id != producer_artifact_id:
                    raise SelectionLedgerIntegrityError(
                        f"replay of source_event_id={source_event_id!r} for (record_id={record_id!r}, "
                        f"stage={stage!r}) claims producer_artifact_id={producer_artifact_id!r}, but the "
                        f"already-persisted decision_id={existing.decision_id!r} was recorded with "
                        f"producer_artifact_id={existing.producer_artifact_id!r} -- a genuine replay of "
                        "the same source event must always claim the identical producer, never a "
                        "different one; refusing to silently conceal the mismatch"
                    )
                candidate = build_selection_decision_record(
                    review_run_id=store.review_run_id, decision_id=existing.decision_id,
                    source_event_id=source_event_id, producer_artifact_id=producer_artifact_id,
                    record_id=record_id, stage=stage,
                    decision=decision, decision_maker=decision_maker, reason=reason,
                    supersedes_decision_id=existing.supersedes_decision_id, timestamp=existing.timestamp,
                )
                if candidate.semantic_hash != existing.semantic_hash:
                    raise SelectionLedgerIntegrityError(
                        f"replay of source_event_id={source_event_id!r} for (record_id={record_id!r}, "
                        f"stage={stage!r}) proposes decision={decision!r}/decision_maker={decision_maker!r}/"
                        f"reason={reason!r}, which does not match the already-persisted "
                        f"decision_id={existing.decision_id!r} (semantic_hash {existing.semantic_hash!r} != "
                        f"recomputed {candidate.semantic_hash!r}) -- a genuine replay of the same source "
                        "event must always propose identical content; refusing to silently keep either "
                        "version"
                    )

                # Stage 3: an exact source_event_id match is only a safe,
                # meaningful "yes, this exact event already happened"
                # answer for a precondition-checked (human-resolution)
                # caller when it's ALSO still the current tail -- i.e. an
                # "artifact-ahead-of-state" replay of the caller's own
                # already-applied write. If the match is a HISTORICAL,
                # already-superseded entry (A0 -> H -> A1, then a stale
                # replay of H's own resolution_id), returning it
                # unconditionally would let a caller believe a superseded
                # decision is current -- state and ledger would disagree.
                # A caller with NO precondition (expected is _UNSET, the
                # legacy/non-human-resolution path) keeps the original,
                # unconditional replay behavior exactly as before.
                if expected_current_tail_decision_id is _UNSET:
                    return existing
                if tail is not None and tail.decision_id == existing.decision_id:
                    return existing
                raise SelectionLedgerPreconditionError(
                    f"replay of source_event_id={source_event_id!r} for (record_id={record_id!r}, "
                    f"stage={stage!r}) matches historical decision_id={existing.decision_id!r}, but the "
                    f"current tail is {(tail.decision_id if tail is not None else None)!r} -- this "
                    "resolution exists historically but has since been superseded; re-review against the "
                    "current state rather than trusting a stale replay"
                )

        # Step 2: no matching source_event_id anywhere in the chain --
        # for a precondition-checked caller, the current tail must equal
        # exactly what it expected to be superseding (including `None`
        # for a fresh root) BEFORE constructing anything -- otherwise
        # this call would silently supersede a decision the caller never
        # actually reviewed (e.g. a newer AI/rule-based decision that
        # arrived after a human queue entry was created but before this
        # resolution was applied).
        if expected_current_tail_decision_id is not _UNSET:
            actual_tail_decision_id = tail.decision_id if tail is not None else None
            if actual_tail_decision_id != expected_current_tail_decision_id:
                raise SelectionLedgerPreconditionError(
                    f"emit_selection_decision for (record_id={record_id!r}, stage={stage!r}) expected "
                    f"current tail decision_id={expected_current_tail_decision_id!r}, but the actual "
                    f"current tail is {actual_tail_decision_id!r} -- refusing to supersede a decision "
                    "that does not match the precondition the caller was reviewing against"
                )

        # Step 3/4: construct a genuinely new record, auto-superseding
        # the current tail (if any) or starting a fresh root.
        record = build_selection_decision_record(
            review_run_id=store.review_run_id, decision_id=str(uuid.uuid4()), source_event_id=source_event_id,
            producer_artifact_id=producer_artifact_id,
            record_id=record_id, stage=stage, decision=decision, decision_maker=decision_maker,
            reason=reason, supersedes_decision_id=(tail.decision_id if tail is not None else None),
        )
        return store.publish(record)
