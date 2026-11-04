"""Shared AI-use traceability event store -- Stage 3 (`quality_appraisal_synthesis_plan_2026-08-22.md`
Section 3.3). One `AIUseEvent` per actual model/tool call (never per
node-run), durably persisted with crash-safe, collision-detecting
publication and re-verify-on-load integrity, independent of any single
node so `extraction`, `quality_appraisal`, `screening`, `screening_qc`,
`search_strategy_builder`, and `synthesis`'s BERTopic call can all reuse
the identical mechanism rather than each inventing their own.

Not shared with `slr_pipeline.extraction`/`slr_pipeline.nodes.quality_appraisal`'s
own checkpoint stores -- this is a parallel, independent artifact concept
(a call-level provenance log, not a resumable-computation checkpoint).
"""
from __future__ import annotations

import contextlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.extraction.hashing import sha256_hex

try:
    import fcntl  # POSIX: kernel-backed advisory lock, released on process exit
except ImportError:  # pragma: no cover - exercised only on POSIX
    fcntl = None

try:
    import msvcrt  # Windows: kernel-backed advisory lock, released on process exit
except ImportError:  # pragma: no cover - exercised only on Windows
    msvcrt = None


class AIEventIntegrityError(RuntimeError):
    """A persisted `AIUseEvent` file failed self-verification (its stored
    `semantic_hash`/`content_hash` does not match a fresh recomputation
    from its own field values) -- the file is corrupted or tampered,
    never silently trusted."""


class AIEventLockError(RuntimeError):
    """Could not acquire the per-`(review_run_id, work_item_id, attempt)`
    publish lock within the timeout."""


class _LockBusyError(Exception):
    pass


_POSIX_LOCK_CONTENTION_ERRNOS = {11, 35}  # EAGAIN, EWOULDBLOCK (platform-dependent value)
_WINDOWS_LOCK_CONTENTION_ERRNOS = {13, 36}  # errno.EACCES / errno.EDEADLOCK-ish values msvcrt.locking raises


# ---------------------------------------------------------------------------
# AIUseEvent
# ---------------------------------------------------------------------------

# Fields hashed into semantic_hash: every field EXCEPT event_id, timestamp,
# semantic_hash, content_hash themselves. event_id is excluded from the
# HASH INPUT because it is itself a pure function of (review_run_id, stage,
# work_item_id, attempt) -- already covered by the other fields, so hashing
# it too would be redundant -- but that derivation is independently
# enforced below (`_check_event_id_matches_identity`), so event_id can
# never silently drift from the identity fields that determine it.
_SEMANTIC_HASH_FIELDS = (
    "review_run_id", "stage", "work_item_id", "attempt", "tool_kind", "model", "provider",
    "component_versions", "access_info", "task_purpose", "model_config_id",
    "prompt_template_version", "prompt_template_text", "prompt_template_hash", "prompt_parameters",
    "rendered_prompt_hash", "input_artifact_refs", "output_artifact_refs",
    "call_status", "failure_detail",
)
_CONTENT_HASH_EXCLUDED_FIELDS = {"semantic_hash", "content_hash"}

_LLM_ONLY_PROMPT_FIELDS = (
    "prompt_template_version", "prompt_template_text", "prompt_template_hash", "rendered_prompt_hash",
)


def compute_prompt_template_hash(prompt_template_text: str) -> str:
    """`prompt_template_hash` must be a verifiable function of the stored
    `prompt_template_text` (the AUTHORED template), never an arbitrary
    caller-supplied string -- otherwise the hash carries no real
    integrity guarantee at all."""
    return sha256_hex(prompt_template_text)


class AIUseEvent(BaseModel):
    """See Section 3.3 of the plan for the full design rationale (LLM vs.
    non_llm_tool as a first-class, not holes-punched-in-LLM-shape, case;
    component_versions as structured multi-package data; the
    semantic_hash/content_hash split). `extra="forbid"` so an unknown
    injected field can never silently ride along uncovered by either
    hash; `frozen=True` matches the intended immutable-record contract --
    an `AIUseEvent`, once constructed, is never mutated in place."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_run_id: str
    event_id: str
    stage: str
    work_item_id: str
    attempt: int = Field(ge=0)
    tool_kind: Literal["llm", "non_llm_tool"]
    model: str
    provider: str
    component_versions: Dict[str, str]
    access_info: Optional[str] = None
    task_purpose: str
    model_config_id: str
    prompt_template_version: Optional[str] = None
    prompt_template_text: Optional[str] = None
    prompt_template_hash: Optional[str] = None
    prompt_parameters: Dict[str, Any] = Field(default_factory=dict)
    rendered_prompt_hash: Optional[str] = None
    input_artifact_refs: List[str]
    output_artifact_refs: List[str]
    call_status: Literal["success", "failure", "heuristic_fallback"]
    failure_detail: Optional[str] = None
    timestamp: str
    semantic_hash: str
    content_hash: str

    @model_validator(mode="after")
    def _check_event_id_matches_identity(self) -> "AIUseEvent":
        """Runs on every construction path, including `model_validate()`
        on load -- an `event_id` that does not match what
        `compute_event_id()` derives from this same event's own
        `review_run_id`/`stage`/`work_item_id`/`attempt` is rejected
        immediately, never silently accepted as if it were bound to
        those fields."""
        expected_event_id = compute_event_id(
            review_run_id=self.review_run_id, stage=self.stage,
            work_item_id=self.work_item_id, attempt=self.attempt,
        )
        if self.event_id != expected_event_id:
            raise ValueError(
                f"event_id {self.event_id!r} does not match compute_event_id() = {expected_event_id!r} "
                "for this event's own review_run_id/stage/work_item_id/attempt -- event_id must be a pure "
                "function of those four fields, never independently set"
            )
        return self

    @model_validator(mode="after")
    def _check_prompt_fields_match_tool_kind(self) -> "AIUseEvent":
        """Bidirectional: `tool_kind='llm'` requires every LLM-only prompt
        field to be set (and `prompt_template_hash` to actually be the
        hash of `prompt_template_text`); `tool_kind='non_llm_tool'`
        requires all of them to be `None` -- a well-formed first-class
        case, not an LLM-shaped event with holes left unfilled."""
        if self.tool_kind == "llm":
            missing = [name for name in _LLM_ONLY_PROMPT_FIELDS if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"tool_kind='llm' requires {missing} to be non-None -- an LLM event must carry its "
                    "own prompt-template provenance"
                )
            expected_hash = compute_prompt_template_hash(self.prompt_template_text)
            if self.prompt_template_hash != expected_hash:
                raise ValueError(
                    f"prompt_template_hash {self.prompt_template_hash!r} does not match sha256 of the "
                    f"stored prompt_template_text (expected {expected_hash!r}) -- the hash must verifiably "
                    "correspond to the template it claims to represent"
                )
        else:
            present = [name for name in _LLM_ONLY_PROMPT_FIELDS if getattr(self, name) is not None]
            if present:
                raise ValueError(
                    f"tool_kind='non_llm_tool' requires {list(_LLM_ONLY_PROMPT_FIELDS)} to all be None, "
                    f"but {present} were set -- prompt-template provenance is an LLM-only concept"
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
            raise AIEventIntegrityError(
                f"AIUseEvent {self.event_id!r} failed semantic_hash self-verification: stored "
                f"{self.semantic_hash!r} != recomputed {expected_semantic!r} -- file corrupted or tampered"
            )
        expected_content = self.compute_content_hash()
        if self.content_hash != expected_content:
            raise AIEventIntegrityError(
                f"AIUseEvent {self.event_id!r} failed content_hash self-verification: stored "
                f"{self.content_hash!r} != recomputed {expected_content!r} -- file corrupted or tampered "
                "(a timestamp-only edit would be caught here even though semantic_hash deliberately "
                "excludes timestamp)"
            )


def compute_event_id(*, review_run_id: str, stage: str, work_item_id: str, attempt: int) -> str:
    return "aiuse-" + sha256_hex(
        {"review_run_id": review_run_id, "stage": stage, "work_item_id": work_item_id, "attempt": attempt}
    )[:48]


def build_ai_use_event(
    *, review_run_id: str, stage: str, work_item_id: str, attempt: int = 0,
    tool_kind: Literal["llm", "non_llm_tool"], model: str, provider: str,
    component_versions: Dict[str, str], task_purpose: str, model_config_id: str,
    input_artifact_refs: List[str], output_artifact_refs: List[str],
    call_status: Literal["success", "failure", "heuristic_fallback"],
    access_info: Optional[str] = None, failure_detail: Optional[str] = None,
    prompt_template_version: Optional[str] = None, prompt_template_text: Optional[str] = None,
    prompt_parameters: Optional[Dict[str, Any]] = None,
    rendered_prompt_hash: Optional[str] = None, timestamp: Optional[str] = None,
) -> AIUseEvent:
    """The one place an `AIUseEvent` is constructed -- computes `event_id`
    and both hashes itself, so no call site can construct a
    self-inconsistent event. `prompt_template_hash` is deliberately NOT a
    parameter here -- it is always derived from `prompt_template_text`
    (`compute_prompt_template_hash`), never accepted as an independent,
    unverifiable caller-supplied string."""
    event_id = compute_event_id(review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=attempt)
    ts = timestamp if timestamp is not None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prompt_template_hash = (
        compute_prompt_template_hash(prompt_template_text) if prompt_template_text is not None else None
    )
    base = dict(
        review_run_id=review_run_id, event_id=event_id, stage=stage, work_item_id=work_item_id, attempt=attempt,
        tool_kind=tool_kind, model=model,
        provider=provider, component_versions=component_versions, access_info=access_info,
        task_purpose=task_purpose, model_config_id=model_config_id,
        prompt_template_version=prompt_template_version, prompt_template_text=prompt_template_text,
        prompt_template_hash=prompt_template_hash, prompt_parameters=prompt_parameters or {},
        rendered_prompt_hash=rendered_prompt_hash, input_artifact_refs=input_artifact_refs,
        output_artifact_refs=output_artifact_refs, call_status=call_status, failure_detail=failure_detail,
        timestamp=ts,
    )
    semantic_hash = AIUseEvent.model_construct(**base, semantic_hash="", content_hash="").compute_semantic_hash()
    with_semantic = dict(base, semantic_hash=semantic_hash)
    content_hash = AIUseEvent.model_construct(**with_semantic, content_hash="").compute_content_hash()
    return AIUseEvent.model_validate(dict(with_semantic, content_hash=content_hash))


# ---------------------------------------------------------------------------
# Hardened event store: write+fsync temp, publish via atomic hard-link,
# self-verify-before-compare on collision, self-verify-on-load for
# aggregation. See Section 3.3 for the full rationale.
# ---------------------------------------------------------------------------

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_EVENT_ID_RE = re.compile(r"^aiuse-[0-9a-f]{48}$")   # matches compute_event_id()'s own output shape exactly


class AIUseEventStore:
    def __init__(self, *, output_dir: Path, review_run_id: str) -> None:
        """Same safe-token-plus-resolved-containment discipline as
        `ExtractionCheckpointStore`/`QACheckpointStore` -- `review_run_id`
        is often derived from mutable, LangGraph-checkpointed state and
        must never be trusted as an already-safe path component (a value
        like `"../escaped"` must not be able to write outside
        `ai_traceability/`)."""
        traceability_root = Path(output_dir) / "ai_traceability"
        if not _SAFE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        run_dir = traceability_root / review_run_id
        if not run_dir.resolve().is_relative_to(traceability_root.resolve()):
            raise ValueError(f"review_run_id {review_run_id!r} resolves outside the ai_traceability root -- refusing to construct store")
        self.review_run_id = review_run_id
        self.events_dir = run_dir / "events"

    def event_path(self, event_id: str) -> Path:
        return self.events_dir / f"{event_id}.json"

    def _load_and_verify_event(self, path: Path, *, expected_event_id: Optional[str] = None) -> AIUseEvent:
        """The ONE shared verified-single-event loader -- used by
        `publish()`'s collision path, `load_all()`, `load_event()`, and
        `reserved_attempt()`'s attempt-scan alike, so all four apply the
        identical discipline (self-hash verification, this store's own
        `review_run_id`, filename-vs-`event_id` identity, and optionally
        the caller's own expected `event_id`) before a persisted file's
        content is trusted for anything. Stage 3 node-instrumentation
        review round: `publish()`'s collision path previously only called
        `_load_event_file(path); existing.verify_hashes()` -- it never
        checked `review_run_id`/filename binding the way `load_all()`
        already did, a real gap in the already-accepted store, closed
        here rather than left as new-code-only discipline."""
        event = _load_event_file(path)
        event.verify_hashes()
        if event.review_run_id != self.review_run_id:
            raise AIEventIntegrityError(
                f"event at {path} has review_run_id={event.review_run_id!r}, expected this store's own "
                f"{self.review_run_id!r} -- refusing to trust a foreign-run event found in this run's directory"
            )
        if path.stem != event.event_id:
            raise AIEventIntegrityError(
                f"event at {path} has event_id={event.event_id!r}, which does not match its own filename "
                f"stem {path.stem!r} -- refusing to trust a renamed or mismatched file"
            )
        if expected_event_id is not None and event.event_id != expected_event_id:
            raise AIEventIntegrityError(
                f"event at {path} has event_id={event.event_id!r}, expected {expected_event_id!r} -- "
                "refusing to trust an event found at the wrong path"
            )
        return event

    def load_event(self, event_id: str) -> AIUseEvent:
        """Safe accessor for a persisted event by id -- validates the
        format of `event_id` and enforces resolved-path containment
        BEFORE ever building a path from it, so a persisted-but-
        unvalidated string (e.g. `ScreeningBatchResult.producing_event_id`,
        loaded from another store's own file) is never interpolated into
        a filesystem path unguarded. Prefer this over
        `event_path()`/`_load_and_verify_event()` directly whenever
        `event_id` comes from data this store did not itself just
        compute."""
        if not _EVENT_ID_RE.fullmatch(event_id):
            raise AIEventIntegrityError(f"event_id {event_id!r} is not a well-formed event id")
        path = self.event_path(event_id)
        if not path.resolve().is_relative_to(self.events_dir.resolve()):
            raise AIEventIntegrityError(f"event_id {event_id!r} resolves outside this store's events directory")
        if not path.is_file():
            raise AIEventIntegrityError(f"no event found for event_id {event_id!r} in this store")
        return self._load_and_verify_event(path, expected_event_id=event_id)

    def publish(self, event: AIUseEvent) -> AIUseEvent:
        """Returns the canonical persisted event: the one just written on a
        genuine first publish, or the original persisted record on a
        benign idempotent-retry republish. Raises on a genuine collision
        (same event_id, different semantic_hash), a foreign-run event, or
        a corrupted existing file."""
        event.verify_hashes()
        if event.review_run_id != self.review_run_id:
            raise AIEventIntegrityError(
                f"refusing to publish event with review_run_id={event.review_run_id!r} into the store for "
                f"review_run_id={self.review_run_id!r} -- an event may only be published into its own run's store"
            )
        self.events_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.event_path(event.event_id)
        tmp_path = self.events_dir / f".{event.event_id}.json.tmp.{uuid.uuid4().hex}"
        payload = event.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, final_path)
        except FileExistsError:
            existing = self._load_and_verify_event(final_path, expected_event_id=event.event_id)
            if existing.semantic_hash == event.semantic_hash:
                tmp_path.unlink(missing_ok=True)
                return existing
            raise AIEventIntegrityError(
                f"AIUseEvent collision for event_id={event.event_id!r}: persisted semantic_hash "
                f"{existing.semantic_hash!r} != new event's {event.semantic_hash!r}. This should be "
                "structurally unreachable under a well-formed work_item_id/attempt scheme -- leaving the "
                f"temp file at {tmp_path} for forensic inspection rather than deleting evidence."
            )
        else:
            tmp_path.unlink(missing_ok=True)
            return event

    def load_all(self) -> List[AIUseEvent]:
        """Self-verifies every event before returning it (via the shared
        `_load_and_verify_event()`) -- a corrupted, foreign-run, renamed,
        or tampered file raises rather than being silently trusted or
        dropped. Also rejects a duplicate `event_id` appearing across
        multiple files (a loader-time integrity failure, not silently
        deduplicated)."""
        if not self.events_dir.is_dir():
            return []
        events: List[AIUseEvent] = []
        seen_event_ids: set[str] = set()
        for path in sorted(self.events_dir.glob("*.json")):
            if path.name.startswith("."):
                continue
            event = self._load_and_verify_event(path)
            if event.event_id in seen_event_ids:
                raise AIEventIntegrityError(
                    f"duplicate event_id {event.event_id!r} found across multiple files in {self.events_dir}"
                )
            seen_event_ids.add(event.event_id)
            events.append(event)
        return events


def classify_automation_origin_optional(events: List[AIUseEvent]) -> Optional[str]:
    """Shared `automation_origin` classification for a stage with NO
    selection-ledger records (aggregation/export/reporting plan v17,
    §2.8) -- `search_strategy` (query expansion + db translation combined)
    and `screening_qc_second_rater`, extracted verbatim from what each of
    those nodes computed inline before this refactor (no behavior change:
    `screening_qc`'s own call site never produces a `heuristic_fallback`
    event, so `statuses == {"heuristic_fallback"}`/`"mixed"` are simply
    never reached there -- this function's four-way branch reduces to the
    exact same two branches that site already had).

    `call_status="failure"` events are excluded entirely (INV-06) -- a
    failure produced no automated output at all, so it must never be
    counted toward `"rule_based_automation"`. Returns `None` (never
    defaulted) when zero events exist for this stage this run -- absence
    of any call is not itself a "no_automated_output" mechanism claim."""
    if not events:
        return None
    output_producing = [e for e in events if e.call_status in ("success", "heuristic_fallback")]
    if not output_producing:
        return "no_automated_output"
    statuses = {e.call_status for e in output_producing}
    if statuses == {"success"}:
        return "ai"
    if statuses == {"heuristic_fallback"}:
        return "rule_based_automation"
    return "mixed"


def _load_event_file(path: Path) -> AIUseEvent:
    import json

    return AIUseEvent.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Reusable kernel-backed per-key lock -- same pattern as
# slr_pipeline.nodes.quality_appraisal's own _qa_resolutions_lock, factored
# out here so Stage 3's SelectionDecisionRecord ledger (which needs a
# per-(review_run_id, record_id, stage) lock, not a per-run one) can reuse
# it rather than reimplementing flock/msvcrt handling a third time.
# ---------------------------------------------------------------------------

def _try_acquire_os_lock(fh) -> None:
    fh.seek(0)
    if fcntl is not None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _POSIX_LOCK_CONTENTION_ERRNOS:
                raise _LockBusyError() from exc
            raise
    elif msvcrt is not None:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in _WINDOWS_LOCK_CONTENTION_ERRNOS:
                raise _LockBusyError() from exc
            raise
    else:  # pragma: no cover - no supported platform lock primitive
        raise RuntimeError("no kernel-backed file-locking primitive available on this platform")


def _release_os_lock(fh) -> None:
    fh.seek(0)
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


@contextlib.contextmanager
def kernel_lock(lock_path: Path, *, error_cls: type = AIEventLockError, description: str = "", timeout: float = 30.0, poll_interval: float = 0.02):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        fh.seek(0, 2)
        if fh.tell() == 0:
            fh.write(b"\0")
            fh.flush()
    except Exception:
        fh.close()
        raise
    deadline = time.monotonic() + timeout
    locked = False
    try:
        while not locked:
            try:
                _try_acquire_os_lock(fh)
                locked = True
            except _LockBusyError:
                if time.monotonic() >= deadline:
                    raise error_cls(f"could not acquire lock {lock_path} ({description}) within {timeout}s")
                time.sleep(poll_interval)
        yield
    finally:
        if locked:
            _release_os_lock(fh)
        fh.close()


# ---------------------------------------------------------------------------
# Concurrency-safe attempt allocation for a logical work item -- Stage 3
# node-instrumentation. See ai_traceability.py's own AIUseEvent docstring:
# `event_id = sha256(review_run_id, stage, work_item_id, attempt)`, where
# `attempt` increments only for a genuine retry of the SAME work item.
# `attempt=0` cannot be hardcoded at every call site: LangGraph re-executes
# a node's full body on `interrupt()`/resume, so a resumed call can
# genuinely re-dispatch the same logical work item's provider call, which
# must get a fresh attempt, never silently collide with (or blindly reuse)
# attempt 0.
# ---------------------------------------------------------------------------

_BEARER_TOKEN_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(r"Authorization\s*:\s*\S+", re.IGNORECASE)
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9-]+")


def sanitize_call_error_text(text: str) -> str:
    """Redacts credential-shaped substrings from a caught provider-call
    exception's text (amber-quilting-osprey.md sec. 3.4, P1 finding 1
    sanitization half). Must run exactly once, at the point the exception
    is caught, before the resulting string is used for ANY purpose --
    logged, placed into `AIUseEvent.failure_detail`, or persisted into a
    `FailedCallRecord.error_message`. No unsanitized copy of the original
    exception text is ever constructed after that point; every consumer
    reads this same already-sanitized value.

    Pattern-based redaction (`Bearer <token>`/`Authorization: ...`/
    `sk-...`) catches the common shapes; exact-value redaction of the
    real, currently-resolved OpenAI credential additionally catches a
    genuine leak that does not happen to match any of those patterns."""
    sanitized = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text)
    sanitized = _AUTH_HEADER_RE.sub("Authorization: [REDACTED]", sanitized)
    sanitized = _OPENAI_KEY_RE.sub("[REDACTED]", sanitized)

    try:
        from slr_pipeline.config import resolve_openai_api_key
        actual_key = resolve_openai_api_key()
    except Exception:
        actual_key = None
    if actual_key:
        sanitized = sanitized.replace(actual_key, "[REDACTED]")

    return sanitized


def attempt_lock_path(store: "AIUseEventStore", *, stage: str, work_item_id: str) -> Path:
    """The per-`(stage, work_item_id)` kernel-lock path `reserved_attempt()`
    itself locks on -- exposed so a caller needing to coordinate with that
    SAME lock from outside `reserved_attempt()` (e.g. startup reconciliation
    re-checking whether a reservation is still genuinely live before
    archiving it) computes the identical path, never a parallel,
    independently-derived one that would fail to actually exclude the real
    lock holder."""
    return store.events_dir.parent / "attempt_locks" / (
        sha256_hex({"stage": stage, "work_item_id": work_item_id})[:32] + ".lock"
    )


@contextlib.contextmanager
def reserved_attempt(
    store: AIUseEventStore, *, stage: str, work_item_id: str, timeout: float = 30.0,
    extra_occupied: Optional[Callable[[int], bool]] = None,
):
    """Reserves the next free `attempt` for one `(stage, work_item_id)`
    logical work item and holds a per-work-item lock across the caller's
    entire reserve -> real call -> publish sequence -- work items are
    already disjoint per concurrent batch/record dispatch, so this never
    serializes unrelated work, but it does prevent two genuinely
    concurrent physical calls for the IDENTICAL work item from racing
    onto the same attempt number.

    Scans attempt=0,1,2,... using the store's own shared verified loader,
    bound to the exact `event_id` each attempt slot must have -- a
    self-hashed-but-foreign-run or renamed/mismatched file at that path
    raises immediately (via `_load_and_verify_event`) rather than being
    treated as merely "this attempt is occupied."

    `extra_occupied` (amber-quilting-osprey.md sec. 3.4, round-6 fix):
    optional, defaults to `None` -- preserves this function's exact
    existing behavior (occupancy determined solely by a published event
    file) for every call site that does not pass it. When given, an
    attempt is ALSO treated as occupied when `extra_occupied(attempt)`
    returns `True`, even though no published `AIUseEvent` exists there yet
    -- lets a caller with its own durable pre-publish markers (e.g. a
    `.reserved.json`/`.completed.json`/`.interrupted_unknown.json` marker
    file) prevent an attempt slot from being handed out twice, without
    this allocator needing to know anything about that caller's own
    on-disk marker format."""
    lock_path = attempt_lock_path(store, stage=stage, work_item_id=work_item_id)
    with kernel_lock(
        lock_path, error_cls=AIEventLockError,
        description=f"ai_use_attempt:{stage}:{work_item_id}", timeout=timeout,
    ):
        attempt = 0
        while True:
            event_id = compute_event_id(
                review_run_id=store.review_run_id, stage=stage, work_item_id=work_item_id, attempt=attempt,
            )
            path = store.event_path(event_id)
            occupied = path.exists()
            if occupied:
                store._load_and_verify_event(path, expected_event_id=event_id)
            elif extra_occupied is not None and extra_occupied(attempt):
                occupied = True
            if not occupied:
                break
            attempt += 1
        yield attempt
