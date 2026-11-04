"""
extraction_node -- shared structured-extraction step for the rapid, SLR,
AND scoping graphs, built directly on the frozen, adversarially-reviewed
Phase A engine (`slr_pipeline/extraction/*`, commit 1fb164b). Originally
scoping was UNCHANGED by this module and kept using the legacy
`charting_node` (nodes/charting.py); Stage 2 of
quality_appraisal_synthesis_plan_2026-08-22.md re-platformed scoping's own
charting onto this SAME engine (via
`slr_pipeline.adapters.scoping_charting_adapter`), so `graphs/scoping.py`
now imports and uses `extraction_node` too -- `nodes/charting.py` was
moved to `examples/legacy/charting.py`, preserved as a teaching example,
no longer imported by any production graph.

Two clearly separated layers, mirroring nodes/fulltext.py's own
acquisition/eligibility split:

  1. THIS MODULE (thin orchestration only): turns `state["full_text_records"]`
     (confirmed-PDF-only, by construction -- see fulltext.py's own
     docstring) into the `(document_id, pdf_path)` pairs `run_extraction()`
     expects, builds the ONE explicit OpenAI provider call via
     `openai_runner.run_openai_batch_extraction` (requirement 4: no
     unified backend abstraction, no local-model fallback -- a future
     local runner is a SEPARATE node, never auto-detected here), and maps
     the engine's own field-level results into a review queue.
  2. THE FROZEN PHASE A ENGINE (`extraction/engine.py`'s `run_extraction()`,
     used directly and unmodified -- requirement 5): parsing, batching,
     evidence verification, value validation, aggregation, checkpointing,
     manifest publication. This module reimplements NONE of that.

No title/abstract fallback, no include-on-error, no re-screening
(requirement 2): every record in `full_text_records` MUST already carry a
confirmed `full_text_path` (fulltext_node's own guarantee); this node
raises rather than silently falling back to an abstract for any record
that somehow doesn't. There is no LLM eligibility call here at all --
eligibility was already decided upstream by FT_SCREENING (nodes/fulltext.py).

DURABLE RESOLUTION ARTIFACT (codex review, Phase B round-2 finding C1;
hardened rounds 3-5): `ReviewState`/`MemorySaver` alone cannot reproduce
which human-resolved values a synthesis run actually consumed -- state is
process-local and mutable. `resolve_extraction_review_queue()` (and
`extraction_node()`, on a resume -- including from FRESH state that merely
recomputes the same deterministic run_id, round-4 finding 1) therefore
publish a self-hashed, append-only-immutable JSON envelope per run
(`extraction_resolutions.json`, next to Phase A's own `manifest.json`,
under the SAME `ExtractionCheckpointStore` run directory) binding every
human resolution to the run's OWN verified manifest hash and extraction
schema hash.

ONE centralized semantic verifier (`_verify_resolution_artifact_semantics`,
reached only via `_read_and_verify_resolution_artifact`) is the SOLE path
every recovery consumer uses -- the public loader
(`load_verified_extraction_resolution_artifact`), `extraction_node`'s own
discovery, and `resolve_extraction_review_queue`'s merge all go through it
(round-5 finding 2: recovery previously read the artifact via its bare
self-hash/run-id check alone, so a re-hashed-but-semantically-wrong
artifact was not just accepted by a fresh recovery -- it was silently
"healed" by republishing corrected fields, permanently erasing the
tampered file's bytes/timestamp instead of raising). It re-derives EVERY
persisted field (resolution bindings/values/original_status, canonical
remaining-queue coverage, manifest_path, total_fields, resolved_projection
in canonical MANIFEST document order -- round-5 finding 3: order is bound
into the run identity, so it is derived from the verified manifest, never
from mutable state's own possibly-reordered `full_text_records`) against
a fresh `results`/`manifest` pair (a genuine `VerifiedRunBundle`, or
`extraction_node`'s own just-computed, equally authoritative
`results`/`manifest`). `_publish_resolution_artifact()` itself never reads
the raw file independently either (round-5 follow-up): every caller passes
in the SAME already-verified artifact (or `None`) it obtained from
`_read_and_verify_resolution_artifact`, so the append-only/no-op
comparison a publish performs is always against a semantically-verified
baseline, never a fresh, unverified re-read that could reintroduce the
same bypass.

Every resolution -- whether freshly constructed by
`resolve_extraction_review_queue()` (verified again, immediately, before
projection/publication -- round-5 finding 1: a freshly-built resolution
was previously published WITHOUT ever passing through the verifier, and
its `original_status` was copied from the queue entry's own top-level
`"status"` field, which the tamper-check never actually compared), folded
back in on a resume, or reloaded from the durable artifact -- is
re-verified via the SAME `_verify_existing_resolution()` against a fresh
bundle/result set.

Node input keys:  full_text_records, config, prisma_counts
Node output keys: extraction_results, extraction_review_queue,
                   requires_human_review, extraction_run_id,
                   extraction_manifest_path, extraction_manifest_hash,
                   extraction_artifact_paths, extraction_schema_hash,
                   prisma_counts, charted_data (ONLY once
                   extraction_review_queue is empty -- see
                   `_project_charted_data`'s own docstring)
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    import fcntl  # POSIX: kernel-backed advisory lock, released on process exit
except ImportError:  # pragma: no cover - exercised only on POSIX
    fcntl = None

try:
    import msvcrt  # Windows: kernel-backed advisory lock, released on process exit
except ImportError:  # pragma: no cover - exercised only on Windows
    msvcrt = None

from slr_pipeline.state import ChartedRecord, PRISMACounts, Record, ReviewState
from slr_pipeline.config import ReviewConfig
from slr_pipeline.extraction.checkpoint import CheckpointIntegrityError, ExtractionCheckpointStore, VerifiedRunBundle
from slr_pipeline.extraction.engine import run_extraction
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig, run_openai_batch_extraction
from slr_pipeline.extraction.parsing import DoclingParserOptions
from slr_pipeline.extraction.resolution_ids import safe_key
from slr_pipeline.extraction.resolution_journal import (
    DecisionOrigin,
    ResolutionDecisionIntent,
    append_decision_entry,
    packet_entry,
    read_decision_journal,
    repair_torn_journal_tail,
    rehydrate_evidence,
    resolution_matches_intent,
    validate_journal_transitions,
)
from slr_pipeline.extraction.resolution_schemas import ValidatedSuggestion
from slr_pipeline.extraction.resolution_suggestions import (
    ReviewPacket,
    _candidates_for_field,
    _evidence_index_for_field,
)
from slr_pipeline.extraction.schema import ExtractionSchema
from slr_pipeline.extraction import verified_bundle
from slr_pipeline.synthesis_evidence import _human_attested_chunk_id, build_evidence_index_for_field

logger = logging.getLogger(__name__)

# The resolver lock is deliberately non-reentrant.  This thread-local flag
# lets the extracted core fail closed if a future caller accidentally invokes
# it without the one outer transaction lock required by section 3.6.
_RESOLUTION_LOCK_HELD = threading.local()


def _stripped(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# Explicit config loading -- requirement 3: schema and model configuration
# are REQUIRED when extraction runs; nothing here silently chooses a
# model, reasoning effort, or token budget on the caller's behalf.
# ---------------------------------------------------------------------------

def _load_model_config(cfg: ReviewConfig) -> OpenAIExtractionConfig:
    if not cfg.extraction_model_config:
        raise ValueError(
            "ReviewConfig.extraction_model_config is not set, but full_text_records is "
            "non-empty and extraction_node was invoked. An explicit OpenAIExtractionConfig "
            "(model/model_config_id/reasoning_effort/max_completion_tokens/"
            "expected_provider_model, via dataclasses.asdict(...)) must be authored "
            "explicitly before extraction can run -- there is no default model, reasoning "
            "effort, or token budget."
        )
    return OpenAIExtractionConfig(**cfg.extraction_model_config)


def _load_parser_options(cfg: ReviewConfig) -> Optional[DoclingParserOptions]:
    if cfg.extraction_parser_options is None:
        return None
    return DoclingParserOptions(**cfg.extraction_parser_options)


def _verify_and_canonicalize_document_order(records: List[Record], manifest) -> List[Record]:
    """Codex review round-3/round-4: `resolved_projection`/`total_fields`
    must be built from the VERIFIED run, never mutable state -- but
    `records` (needed for title/authors/etc., which the manifest does not
    carry) still comes from `state["full_text_records"]`. Duplicates are
    rejected outright; the remaining IDs must be exactly the manifest's
    own `document_ids` SET (an omitted or added document is rejected).

    Codex review round-5 finding 3: a `sorted(...)` multiset comparison
    cannot detect a REORDERING -- since the manifest/run identity bind
    document order (`compute_run_id()`'s own ordered binding) but
    `_project_charted_data`'s own output order simply follows whatever
    order `records` is given in, a merely-reordered (same-multiset)
    `full_text_records` previously produced a differently-ordered (though
    content-identical) projection for the SAME run -- and the loader
    accepted it right back, since it re-derives from whatever order it is
    given too. Durable ordering must never be derived from mutable state:
    this function returns `records` REORDERED into the manifest's own
    canonical `document_ids` order, so every caller's projection is
    deterministic regardless of `state["full_text_records"]`'s own
    (irrelevant) order."""
    record_ids = [r.id for r in records]
    duplicate_ids = sorted({rid for rid in record_ids if record_ids.count(rid) > 1})
    if duplicate_ids:
        raise ValueError(f"full_text_records contains duplicate record_id(s): {duplicate_ids}")
    if set(record_ids) != set(manifest.document_ids):
        raise ValueError(
            f"full_text_records document set ({sorted(record_ids)!r}) does not match the verified run "
            f"manifest's own document_ids ({sorted(manifest.document_ids)!r}) -- refusing to project "
            "against a document set that differs from what this run actually processed"
        )
    records_by_id = {r.id: r for r in records}
    return [records_by_id[doc_id] for doc_id in manifest.document_ids]


# ---------------------------------------------------------------------------
# charted_data compatibility projection (requirement 9)
# ---------------------------------------------------------------------------

def _project_charted_data(
    records: List[Record],
    extraction_results: List[Dict[str, Any]],
    resolutions: List[Dict[str, Any]],
    *,
    current_run_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Deterministic, LOSSY compatibility projection of the complete,
    evidence-bearing extraction result into charted_data's existing
    `{record_id, raw_extraction, ...}` shape, so existing synthesis.py/
    quality_appraisal.py code (which only ever reads `charted_data`) keeps
    working unmodified. `raw_extraction` maps field_id -> a single raw
    value (or a list, for a field with more than one supported value) --
    every evidence span, confidence rating, rationale, and non-primary
    value from the real result is discarded here.

    `records` is expected to ALREADY be in canonical manifest document
    order (see `_verify_and_canonicalize_document_order`) -- this
    function itself does not reorder; every call site is responsible for
    passing a canonicalized list so the OUTPUT order is deterministic
    regardless of what order mutable state happened to carry.

    `current_run_id` is the ONE authoritative filter against stale-run
    replay (codex review, Phase B finding C): every resolution in
    `resolutions` whose own `"run_id"` does not equal `current_run_id` is
    IGNORED here. Every call site now only ever passes resolutions/
    `extraction_results` it has ALREADY verified against a fresh
    `VerifiedRunBundle` (or, for `extraction_node`, its own just-computed
    `results`) -- this filter is defense in depth, not the only guard.

    THIS IS NEVER THE SCIENTIFIC ARTIFACT. The complete, evidence-bearing
    result remains `state["extraction_results"]` (the full, frozen
    per-document FieldResult set) and the underlying Phase A checkpoint/
    manifest files on disk (referenced by `extraction_manifest_path`/
    `extraction_artifact_paths`) -- both untouched and unsimplified by
    this function. `charted_data` exists ONLY so downstream
    synthesis/quality-appraisal code (out of scope for this phase to
    redesign) can keep running temporarily against a familiar shape.

    Called ONLY once the review queue is empty. `resolutions` OVERRIDES
    the machine value for any (record_id, field_id) it covers -- a
    human-resolved 'extracted' outcome projects the human-supplied value;
    a human-resolved 'not_reported' outcome projects `None`. Every other
    field projects its OWN machine value directly: `extracted` -> its
    first (or, for a multi-valued field, every) raw value; `not_reported`
    -> `None`. A field that is still `unclear`/`conflicting`/
    `extraction_failed` with NO matching resolution is left out of
    `raw_extraction` entirely rather than guessed at -- this should never
    actually happen, since both call sites only ever invoke this function
    once their own review queue is empty."""
    resolved_by_key = {
        (r["record_id"], r["field_id"]): r for r in resolutions if r.get("run_id") == current_run_id
    }
    results_by_id = {r["document_id"]: r for r in extraction_results}

    charted: List[ChartedRecord] = []
    for record in records:
        result = results_by_id.get(record.id)
        raw: Dict[str, Any] = {}
        if result is not None:
            for field in result["fields"]:
                field_id = field["field_id"]
                key = (record.id, field_id)
                resolution = resolved_by_key.get(key)
                if resolution is not None:
                    raw[field_id] = resolution["value"] if resolution["outcome"] == "extracted" else None
                    continue

                status = field["status"]
                if status == "extracted":
                    values = [v["value"] for v in field["values"]]
                    raw[field_id] = values[0] if len(values) == 1 else values
                elif status == "not_reported":
                    raw[field_id] = None
                # unclear/conflicting/extraction_failed with no resolution:
                # left out of raw_extraction -- see docstring above.

        charted.append(
            ChartedRecord(
                record_id=record.id,
                raw_extraction=raw,
                validated=result is not None,
                charting_method="extraction_projection",
            )
        )
    return [c.model_dump() for c in charted]


# ---------------------------------------------------------------------------
# Durable resolution artifact (codex review, Phase B round-2 finding C1) --
# a self-hashed, append-only-immutable envelope binding every human
# resolution for a run to that run's own verified manifest/schema
# identity. Lives next to Phase A's own manifest.json under the SAME
# ExtractionCheckpointStore run directory, but is defined HERE (not in
# slr_pipeline/extraction/checkpoint.py) since Phase A itself is frozen
# and has no field for a Phase B, human-review concept.
# ---------------------------------------------------------------------------

def _resolution_artifact_path(store: ExtractionCheckpointStore) -> Path:
    return store.run_dir / "extraction_resolutions.json"


def _resolution_artifact_lock_path(store: ExtractionCheckpointStore) -> Path:
    return store.run_dir / "extraction_resolutions.lock"


class ResolutionArtifactLockError(Exception):
    """Raised when the per-run resolution-artifact lock cannot be
    acquired within the timeout -- another process/thread appears to be
    reading/verifying/merging/writing the SAME run's resolution artifact
    concurrently."""


class _LockBusyError(Exception):
    """Internal signal: a non-blocking OS lock attempt found the lock
    already held by someone else. Never escapes `_resolution_artifact_lock`."""


# Codex review round-8 finding (P2): only THESE errno values mean "someone
# else currently holds the lock" for each platform's non-blocking primitive.
# `fcntl.flock(..., LOCK_NB)` signals contention with EWOULDBLOCK (aliased
# to EAGAIN on Linux, distinct on some BSDs -- both are listed defensively).
# Codex review round-9 finding (P2): the Python 3.12 fcntl docs
# (https://docs.python.org/3.12/library/fcntl.html) additionally document
# that contention on `flock(..., LOCK_NB)` may be reported as EACCES
# instead of EAGAIN/EWOULDBLOCK, depending on the OS -- omitting it meant
# ordinary contention on those systems propagated immediately as a
# PermissionError instead of being retried until the holder released.
# `msvcrt.locking(..., LK_NBLCK, ...)` signals contention with EACCES (the
# CRT's `_locking` reports "Permission denied" for an already-locked
# region). Any OTHER errno -- EIO, EBADF, ENOLCK (lock table exhausted),
# EINVAL (locking unsupported on this filesystem), etc. -- is a real
# failure, not contention, and must propagate immediately with its
# original cause rather than being retried for the full 30s timeout and
# then reported as a misleading "another live process" error.
_POSIX_LOCK_CONTENTION_ERRNOS = {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}
_WINDOWS_LOCK_CONTENTION_ERRNOS = {errno.EACCES}


def _try_acquire_os_lock(fh) -> None:
    """Attempt a non-blocking, kernel-backed advisory lock on `fh`. Raises
    `_LockBusyError` if another live process/handle already holds it;
    re-raises any other `OSError` (unsupported locking, exhausted lock
    resources, I/O failure, invalid descriptor, ...) unchanged, with its
    original cause intact, since retrying those for the full timeout would
    only produce a misleading "another live process" report.

    Deliberately NOT based on file *existence* (an exclusively-created
    lock file, `open(path, "x")`): that approach leaves a permanent
    orphaned lock file behind if the holder is hard-killed (SIGKILL /
    TerminateProcess bypasses `finally`), so every future resumer waits
    out the full timeout and fails forever -- undermining checkpoint
    recovery (Codex review round-7 finding 1). A kernel-backed lock
    (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows) is instead tied
    to the OS's own record of which open file descriptions hold it: the
    OS releases it automatically when the holding process dies, for any
    reason, without any cooperating code needing to run."""
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
def _resolution_artifact_lock(store: ExtractionCheckpointStore, *, timeout: float = 30.0, poll_interval: float = 0.02):
    """Codex review round-6 finding 1: the read-verify-merge-write
    sequence for a run's resolution artifact must be a single atomic
    transaction. Without this, two concurrent resolvers can both read the
    SAME existing artifact snapshot E0 (a genuine race -- both reads
    happen before either write), each independently add a DIFFERENT new
    resolution, and each then publish comparing against that SAME stale
    E0: the append-only check on either side passes (E0 has nothing to
    protect yet from that writer's own point of view), so whichever
    writer's `os.replace(...)` runs SECOND silently overwrites the FIRST
    writer's already-durable resolution -- a lost update the two writers'
    own "queue must exactly match" checks do NOT catch, since both read
    an identical, momentarily-accurate state/artifact pair before either
    committed.

    Wrapping the entire read+verify+merge+process+publish sequence in
    this lock serializes the two transactions: whichever thread acquires
    it first completes its FULL transaction (durably) before the second
    can even begin its OWN read, so the second thread's read now reflects
    the first's committed resolution -- its own "queue must exactly
    match" check then correctly (and loudly) rejects a stale queue,
    rather than silently losing data.

    Codex review round-7 finding 1: uses a kernel-backed advisory lock
    (`fcntl.flock` / `msvcrt.locking`) held on a persistent lock-file
    handle, NOT an exclusively-created lock file. The lock file itself is
    created once and then left on disk permanently -- it is never used as
    a mutex by its own existence, only as an anchor for the OS-held
    byte-range lock, so there is nothing to reclaim or go stale if a
    holder is hard-killed: the OS drops the lock the instant the killed
    process's file descriptors are torn down, and the very next
    `_try_acquire_os_lock` call (from this process or any other) succeeds
    immediately, no different from a clean release."""
    if getattr(_RESOLUTION_LOCK_HELD, "value", False):
        raise RuntimeError("_resolution_artifact_lock is non-reentrant")
    lock_path = _resolution_artifact_lock_path(store)
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
                    raise ResolutionArtifactLockError(
                        f"could not acquire the resolution-artifact lock for run_id={store.run_id!r} within "
                        f"{timeout}s -- another live process/thread appears to be resolving this run "
                        "concurrently (if a prior holder was killed, the OS releases its lock "
                        "automatically, so a timeout here means a genuinely live holder, not a stale one)"
                    )
                time.sleep(poll_interval)
        _RESOLUTION_LOCK_HELD.value = True
        yield
    finally:
        if locked:
            _RESOLUTION_LOCK_HELD.value = False
        if locked:
            _release_os_lock(fh)
        fh.close()


def _artifact_content_hash(payload: Dict[str, Any]) -> str:
    return sha256_hex(payload)


def _read_resolution_artifact_file(store: ExtractionCheckpointStore) -> Optional[Dict[str, Any]]:
    """The bare, self-hash-only reader. Codex review round-5 finding 2:
    this is NEVER an acceptable "recovery" path on its own -- every caller
    that wants to TRUST an artifact's content goes through
    `_read_and_verify_resolution_artifact()` instead, which additionally
    runs the FULL semantic verifier. This function is called from exactly
    two places: internally by `_read_and_verify_resolution_artifact()`
    (which then verifies what it reads), and nowhere else -- deliberately
    not even by `_publish_resolution_artifact()`, which now takes an
    already-verified artifact from its caller rather than re-reading."""
    path = _resolution_artifact_path(store)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    check_hash = _artifact_content_hash({k: v for k, v in data.items() if k != "content_hash"})
    if check_hash != data.get("content_hash"):
        raise CheckpointIntegrityError(
            f"extraction resolution artifact for run_id={store.run_id!r} is tampered/corrupted -- its "
            "content_hash does not match its own contents"
        )
    if data.get("run_id") != store.run_id:
        raise CheckpointIntegrityError(
            f"extraction resolution artifact claims run_id={data.get('run_id')!r} but was loaded from "
            f"the store for run_id={store.run_id!r}"
        )
    return data


def _verify_resolution_artifact_semantics(
    artifact: Dict[str, Any], *, store: ExtractionCheckpointStore, results, manifest, schema: ExtractionSchema,
    schema_hash: str, canonical_records: List[Record],
) -> None:
    """THE ONE semantic re-derivation/verification path for a resolution
    artifact's COMPLETE persisted content (codex review round-5 finding
    2). Used exclusively via `_read_and_verify_resolution_artifact()` --
    every recovery consumer (public loader, `extraction_node` discovery,
    `resolve_extraction_review_queue` merging) goes through that, never
    trusting `_read_resolution_artifact_file()`'s bare self-hash check as
    sufficient on its own. Raises on ANY mismatch -- a tampered-but-
    re-hashed artifact must cause the WHOLE calling operation to fail
    loudly, never be silently "healed" by a subsequent republish with
    corrected values (round-5's core complaint).

    `results`/`manifest` may come from a genuine `VerifiedRunBundle`
    (loader, resolver) or from `extraction_node`'s own just-computed,
    equally authoritative `results`/`manifest` -- both are treated
    identically here. `canonical_records` MUST already be in the
    manifest's own document order (see
    `_verify_and_canonicalize_document_order`)."""
    if artifact["manifest_hash"] != manifest.manifest_hash or artifact["extraction_schema_hash"] != schema_hash:
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r} does not match the CURRENT "
            "run's manifest_hash/extraction_schema_hash -- refusing to trust a stale or foreign artifact"
        )
    if artifact.get("manifest_path") != str(store.manifest_path()):
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r}: manifest_path does not match "
            "this store's own manifest_path -- refusing to trust a relocated or foreign artifact"
        )

    field_map = schema.field_map()
    review_fields = verified_bundle.bundle_review_fields(results)

    resolutions = artifact.get("resolutions", [])
    seen_keys: set = set()
    for r in resolutions:
        key = (r.get("record_id"), r.get("field_id"))
        if key in seen_keys:
            raise ValueError(
                f"extraction resolution artifact for run_id={store.run_id!r} contains a duplicate "
                f"resolution for {key!r}"
            )
        seen_keys.add(key)
        verified_bundle.verify_existing_resolution(
            r, run_id=store.run_id, manifest_hash=manifest.manifest_hash, schema_hash=schema_hash,
            review_fields=review_fields, field_map=field_map,
        )

    # Codex review round-4 finding 3: `remaining_queue_keys` must be
    # duplicate-free AND in the SAME canonical sorted order
    # `_publish_resolution_artifact` always writes it in -- a set-based
    # comparison alone would miss both.
    remaining_list = [tuple(k) for k in artifact.get("remaining_queue_keys", [])]
    if len(set(remaining_list)) != len(remaining_list):
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r}: remaining_queue_keys contains "
            "duplicate entries"
        )
    if remaining_list != sorted(remaining_list):
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r}: remaining_queue_keys is not in "
            "canonical sorted order"
        )
    expected_remaining = set(review_fields) - seen_keys
    if set(remaining_list) != expected_remaining:
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r}: remaining_queue_keys does not "
            "match the verified result set's own unresolved-field coverage"
        )

    expected_total_fields = sum(len(result.fields) for result in results)
    if artifact.get("total_fields") != expected_total_fields:
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r}: total_fields does not match "
            "the verified result set's own field count"
        )

    extraction_results_dicts = [result.model_dump(mode="json") for result in results]
    expected_projection = (
        _project_charted_data(canonical_records, extraction_results_dicts, resolutions, current_run_id=store.run_id)
        if not expected_remaining
        else []
    )
    if artifact.get("resolved_projection") != expected_projection:
        raise ValueError(
            f"extraction resolution artifact for run_id={store.run_id!r}: resolved_projection is not "
            "derivable from the verified result set (in canonical manifest document order) and its own "
            "(now-verified) resolutions -- refusing to trust it"
        )


def _read_and_verify_resolution_artifact(
    store: ExtractionCheckpointStore, *, results, manifest, schema: ExtractionSchema, schema_hash: str,
    records: List[Record],
) -> Tuple[Optional[Dict[str, Any]], List[Record]]:
    """THE ONE recovery entry point for reading a resolution artifact off
    disk (codex review round-5 finding 2). ALWAYS canonicalizes `records`
    into the manifest's own document order first (round-5 finding 3,
    `_verify_and_canonicalize_document_order`) -- regardless of whether an
    artifact exists -- then, if one does, runs it through the full
    semantic verifier (`_verify_resolution_artifact_semantics`) before
    returning it. Returns `(artifact_or_None, canonical_records)`.

    Every caller that wants to RECOVER/TRUST a resolution artifact's
    content -- the public loader, `extraction_node`'s own discovery, and
    `resolve_extraction_review_queue`'s merge -- goes through this, never
    `_read_resolution_artifact_file()` directly. `_publish_resolution_
    artifact()` does not read the file at all any more; every caller
    passes it the exact artifact object obtained here (or `None`)."""
    canonical_records = _verify_and_canonicalize_document_order(records, manifest)
    artifact = _read_resolution_artifact_file(store)
    if artifact is None:
        return None, canonical_records
    _verify_resolution_artifact_semantics(
        artifact, store=store, results=results, manifest=manifest, schema=schema, schema_hash=schema_hash,
        canonical_records=canonical_records,
    )
    return artifact, canonical_records


def _merge_resolution_log(
    state_resolutions: Optional[List[Dict[str, Any]]], verified_artifact: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Codex review round-3 ("artifact-ahead-of-state recovery"): the
    durable artifact is published BEFORE a resolve call returns its state
    update, so a process crash between a successful publish and the graph
    checkpoint actually persisting that update leaves the ON-DISK artifact
    AHEAD of `state["extraction_review_resolutions"]`. Without this merge,
    a later resolve call -- built only from the stale, behind-the-artifact
    state list -- would omit an already-published resolution from the
    NEXT publish call's `resolutions`, which `_publish_resolution_artifact`'s
    own append-only check correctly (but confusingly) rejects as "removing"
    it.

    Codex review round-5 finding 2: `verified_artifact` MUST already have
    passed `_verify_resolution_artifact_semantics` (via
    `_read_and_verify_resolution_artifact`) -- this function performs NO
    verification of its own and never reads anything from disk itself.

    Merges state's own log with the verified artifact's own resolutions,
    keyed by (record_id, field_id) -- ADDING entries state doesn't know
    about yet, never removing or overriding what state already has (state
    can only ever be BEHIND the artifact under normal operation)."""
    merged = list(state_resolutions or [])
    seen = {(r["record_id"], r["field_id"]) for r in merged}
    if verified_artifact is not None:
        for r in verified_artifact.get("resolutions", []):
            key = (r["record_id"], r["field_id"])
            if key not in seen:
                merged.append(r)
                seen.add(key)
    return merged


def _publish_resolution_artifact(
    store: ExtractionCheckpointStore, *, run_id: str, manifest_hash: str, extraction_schema_hash: str,
    total_fields: int, resolutions: List[Dict[str, Any]], remaining_queue_keys: List[Tuple[str, str]],
    resolved_projection: List[Dict[str, Any]], existing_verified_artifact: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """The ONE supported way to publish the extraction resolution
    artifact. Mirrors `ExtractionCheckpointStore.publish_manifest()`'s own
    re-verify-existing / no-op-if-identical / refuse-if-different
    discipline (reimplemented here, not imported, since Phase A's own
    checkpoint.py is frozen): an already-published resolution for a given
    (record_id, field_id) can never be silently removed or changed by a
    later publish -- only genuinely NEW resolutions may be added.

    Codex review round-5 follow-up: this function does NOT read the
    artifact file itself -- `existing_verified_artifact` (or `None`, if
    none exists yet) MUST be supplied by the caller, already obtained from
    `_read_and_verify_resolution_artifact()`. An independent raw re-read
    here would (a) reintroduce round-5 finding 2's semantic-verification
    bypass (comparing against a merely self-hash-checked, not semantically
    verified, file) and (b) open a check/use race between whatever the
    caller already verified and what this function would separately read.
    There is exactly ONE on-disk read of this artifact per operation, and
    it is always the verified one.

    Codex review round-4 finding 2: independently rejects duplicate
    `resolutions`/`remaining_queue_keys` entries itself -- never assumes a
    caller already deduplicated -- and does so BEFORE comparing against
    (let alone replacing) any existing artifact, so a duplicate can never
    reach disk even transiently."""
    resolution_keys = [(r["record_id"], r["field_id"]) for r in resolutions]
    duplicate_resolution_keys = sorted({k for k in resolution_keys if resolution_keys.count(k) > 1})
    if duplicate_resolution_keys:
        raise ValueError(
            f"refusing to publish the extraction resolution artifact for run_id={run_id!r}: duplicate "
            f"resolution key(s) {duplicate_resolution_keys!r} in `resolutions`"
        )
    remaining_keys_list = [tuple(k) for k in remaining_queue_keys]
    duplicate_remaining_keys = sorted({k for k in remaining_keys_list if remaining_keys_list.count(k) > 1})
    if duplicate_remaining_keys:
        raise ValueError(
            f"refusing to publish the extraction resolution artifact for run_id={run_id!r}: duplicate "
            f"key(s) {duplicate_remaining_keys!r} in `remaining_queue_keys`"
        )

    payload = {
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "manifest_path": str(store.manifest_path()),
        "extraction_schema_hash": extraction_schema_hash,
        "total_fields": total_fields,
        "resolutions": resolutions,
        "remaining_queue_keys": [list(k) for k in remaining_queue_keys],
        "resolved_projection": resolved_projection,
    }
    existing = existing_verified_artifact

    if existing is not None:
        existing_core = {k: v for k, v in existing.items() if k not in ("content_hash", "created_at")}
        if existing_core == payload:
            return existing  # true no-op -- identical content, keep original created_at/content_hash

        existing_by_key = {(r["record_id"], r["field_id"]): r for r in existing["resolutions"]}
        new_by_key = {(r["record_id"], r["field_id"]): r for r in resolutions}
        for key, old in existing_by_key.items():
            new = new_by_key.get(key)
            if new is None or new != old:
                raise CheckpointIntegrityError(
                    f"refusing to publish the extraction resolution artifact for run_id={run_id!r}: an "
                    f"already-published resolution for {key!r} would be removed or changed -- published "
                    "resolutions are append-only"
                )

    envelope = dict(payload, created_at=datetime.now(timezone.utc).isoformat())
    envelope["content_hash"] = _artifact_content_hash(envelope)

    path = _resolution_artifact_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    return envelope


def _build_extraction_evidence_index(
    extraction_results_dicts: List[Dict[str, Any]], resolutions: List[Dict[str, Any]], current_run_id: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Builds the per-record, per-field evidence index synthesis_node's own map step needs to cite
    exact evidence spans -- additive to, never mutating, the on-disk resolution artifact. Human-resolved:
    one entry per resolution['evidence'] item, chunk_id derived via _human_attested_chunk_id() (the
    RAW quotation_or_description/locator preserved verbatim, page/section None -- a human-resolved
    field has no machine-derived page/section location). Machine status=='extracted', no resolution:
    one entry per (value, evidence-span) pair, page/section preserved, locator None. Everything else
    (not_reported, unclear/conflicting/extraction_failed with no resolution, or a resolution with
    outcome=='not_reported'): no entries -- there is no evidence to cite for a field with no
    reported value."""
    resolved_by_key = {
        (r["record_id"], r["field_id"]): r for r in resolutions if r.get("run_id") == current_run_id
    }
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for result in extraction_results_dicts:
        record_id = result["document_id"]
        record_index: Dict[str, List[Dict[str, Any]]] = {}
        for field in result["fields"]:
            field_id = field["field_id"]
            resolution = resolved_by_key.get((record_id, field_id))
            entries_raw: List[Dict[str, Optional[str]]] = []
            if resolution is not None:
                if resolution["outcome"] == "extracted":
                    for item in resolution["evidence"]:
                        entries_raw.append({
                            "chunk_id": _human_attested_chunk_id(item["locator"]),
                            "quotation": item["quotation_or_description"],
                            "page": None, "section": None, "locator": item["locator"],
                        })
            elif field["status"] == "extracted":
                for value in field["values"]:
                    for ev in value["evidence"]:
                        entries_raw.append({
                            "chunk_id": ev["chunk_id"], "quotation": ev["excerpt"],
                            "page": ev["page"], "section": ev["section"], "locator": None,
                        })
            if entries_raw:
                record_index[field_id] = build_evidence_index_for_field(
                    entries_raw, artifact_kind="extraction", artifact_run_id=current_run_id,
                    record_id=record_id, field_id=field_id,
                )
        if record_index:
            index[record_id] = record_index
    return index


def load_verified_extraction_resolution_artifact(state: ReviewState) -> Optional[Dict[str, Any]]:
    """The ONE supported way for a caller (synthesis_node, reporting_node,
    or a human-facing tool) to read back the durable resolution artifact
    for `state["extraction_run_id"]` -- loads+verifies a `VerifiedRunBundle`
    (`verified_bundle.load_and_verify_bundle`) and delegates entirely to
    `_read_and_verify_resolution_artifact()` (see its own and
    `_verify_resolution_artifact_semantics()`'s docstrings for the complete
    re-derivation this performs). Returns `None` if extraction never ran
    for this state (no run_id) or no artifact has been published yet for
    it. Raises `CheckpointIntegrityError`/`ValueError` on any tampering,
    staleness, or semantic inconsistency -- a malformed artifact is never
    trusted just because its own content_hash happens to be internally
    consistent."""
    run_id = state.get("extraction_run_id")
    if not run_id:
        return None

    store, bundle, schema, schema_hash = verified_bundle.load_and_verify_bundle(state, run_id)
    records = [Record.from_dict(r) for r in state["full_text_records"]]
    artifact, _canonical_records = _read_and_verify_resolution_artifact(
        store, results=bundle.results, manifest=bundle.manifest, schema=schema, schema_hash=schema_hash,
        records=records,
    )
    if artifact is None:
        return None
    # Additive, in-memory-only, never written back to disk (pure read-only projection over the
    # already-verified `artifact`/`bundle.results`) -- see _build_extraction_evidence_index().
    # `extraction_results` (Unit I, squishy-petting-starlight.md) surfaces this same already-computed
    # list -- the complete, typed-value-bearing `DocumentExtractionResult` dicts `bundle.results` itself
    # provides -- so a caller (the claim-evidence audit) can resolve a `FieldResult`'s real typed_value
    # without re-deriving it from the lossy `resolved_projection`/charted_data string-only view.
    extraction_results_dicts = [result.model_dump(mode="json") for result in bundle.results]
    evidence_index = _build_extraction_evidence_index(extraction_results_dicts, artifact["resolutions"], run_id)
    return dict(artifact, evidence_index=evidence_index, extraction_results=extraction_results_dicts)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def extraction_node(state: ReviewState) -> Dict[str, Any]:
    """Runs (or resumes) structured extraction over every record in
    `state["full_text_records"]` via the frozen Phase A `run_extraction()`,
    then maps its field-level results into `extraction_review_queue`.
    Never called for the scoping graph (see this module's own docstring).

    Deterministic-run-id recovery (codex review, Phase B round-2 finding
    C1; hardened round-4 finding 1): `run_id` is a deterministic hash of
    documents+schema+config, so THIS SAME run_id can be recomputed from
    fresh or independently reconstructed state that never carried
    `state["extraction_run_id"]` forward at all -- this node therefore
    ALWAYS discovers and verifies that run_id's own durable resolution
    artifact (via `_read_and_verify_resolution_artifact` +
    `_merge_resolution_log`), regardless of what `state["extraction_run_id"]`
    said beforehand. Prior state's own `extraction_review_resolutions` list
    only contributes when it agrees (`previous_run_id == run_id`) that this
    is the same logical run it was already tracking; the durable artifact
    contributes unconditionally. Already-resolved fields are excluded from
    the rebuilt review queue, and `charted_data` is reconstructed WITH
    those resolutions applied, rather than reopening resolved fields or
    discarding the resolved projection just because the node ran again --
    whether that resume happened via surviving state or via artifact-only
    recovery."""
    cfg = ReviewConfig.from_dict(state["config"])
    records = [Record.from_dict(r) for r in state["full_text_records"]]

    prisma = PRISMACounts(**state["prisma_counts"])

    if not records:
        prisma.included = 0
        # No extraction to do -- explicitly reset every run-scoped field
        # (including extraction_review_resolutions, codex review finding C)
        # rather than leaving a PRIOR run's resolutions/charted_data
        # dangling under LangGraph's partial-update semantics.
        return {
            "extraction_results": [],
            "extraction_review_queue": [],
            "extraction_review_resolutions": [],
            "requires_human_review": False,
            "extraction_run_id": None,
            "extraction_manifest_path": None,
            "extraction_manifest_hash": None,
            "extraction_artifact_paths": None,
            "extraction_schema_hash": None,
            "prisma_counts": prisma.model_dump(),
            "charted_data": [],
        }

    # Requirement 2: confirmed-PDF-only, no title/abstract fallback, no
    # include-on-error -- structural check, not caller discipline. A
    # record must be status="parsed" AND carry a non-blank path AND that
    # path must resolve to an existing regular file (codex review finding
    # E: path truthiness alone let a rejected_stub/None-status record, or
    # a path string pointing nowhere, reach schema/provider work before
    # failing only incidentally, deep inside Phase A's own file hashing).
    invalid_pdf: List[str] = []
    for r in records:
        path = r.full_text_path
        if r.full_text_status != "parsed" or not path or not path.strip() or not Path(path).is_file():
            invalid_pdf.append(r.id)
    if invalid_pdf:
        raise ValueError(
            f"full_text_records contains record(s) with no confirmed, existing full-text PDF "
            f"(full_text_status != 'parsed', a blank path, or a path that is not an existing file): "
            f"{invalid_pdf} -- extraction_node only extracts from a confirmed PDF path; there is no "
            "title/abstract fallback and no include-on-error"
        )

    schema = verified_bundle.load_schema(cfg)
    model_config = _load_model_config(cfg)
    parser_options = _load_parser_options(cfg)
    max_batch_chars = cfg.extraction_max_batch_chars

    def call_batch_fn(batch_chunks):
        # Requirement 4: the ONE explicit provider wire-up, through
        # openai_runner.py directly -- no unified backend abstraction, no
        # local-model fallback.
        return run_openai_batch_extraction(schema=schema, batch_chunks=batch_chunks, config=model_config)

    output_dir = cfg.ensure_output_dir()
    extraction_output_dir = output_dir / "extraction_runs"

    previous_run_id = state.get("extraction_run_id")
    documents: List[Tuple[str, str]] = [(record.id, record.full_text_path) for record in records]

    # Requirement 5: run_extraction() used directly, with its own
    # checkpoint/resume behaviour -- run identity, batching, checkpointing,
    # manifests, and telemetry are NOT reimplemented here.
    run_id, results = run_extraction(
        documents=documents,
        schema=schema,
        call_batch_fn=call_batch_fn,
        output_dir=extraction_output_dir,
        model_config=model_config,
        parser_options=parser_options,
        max_batch_chars=max_batch_chars,
    )

    store = ExtractionCheckpointStore(output_dir=extraction_output_dir, run_id=run_id)
    manifest = store.load_manifest()
    if manifest is None:
        raise RuntimeError(
            f"run_extraction() completed for run_id={run_id!r} but publish_manifest() left no manifest "
            "on disk -- this should never happen (run_extraction() always publishes one before returning)"
        )

    # Machine-derived set of fields that CURRENTLY require review, from
    # the just-computed, authoritative `results` -- never from state.
    machine_review_fields = verified_bundle.bundle_review_fields(results)

    # Codex review round-6 finding 1: the ENTIRE read-verify-merge-write
    # transaction below is held under a single per-run lock -- two
    # concurrent resolvers (or resolver+extraction_node resume) reading
    # the SAME existing artifact snapshot and each publishing based on it
    # would otherwise let the later writer's os.replace(...) silently
    # discard the earlier writer's already-durable resolution (see
    # `_resolution_artifact_lock`'s own docstring).
    with _resolution_artifact_lock(store):
        # Codex review round-4 finding 1 / round-5 finding 2: ALWAYS
        # discover and fully verify this run's own durable artifact
        # (regardless of `previous_run_id`), via the SAME centralized
        # verifier every other recovery path uses -- never a raw,
        # unverified read. This also canonicalizes `records` into the
        # manifest's own document order (round-5 finding 3).
        existing_artifact, records = _read_and_verify_resolution_artifact(
            store, results=results, manifest=manifest, schema=schema,
            schema_hash=manifest.extraction_schema_hash, records=records,
        )

        # `previous_run_id` is used ONLY to decide whether state's own
        # `extraction_review_resolutions` list is even eligible to
        # contribute (state's list describes a DIFFERENT logical run's
        # history from state's own point of view when it disagrees, even
        # if a stray entry in it happens to carry a matching run_id tag)
        # -- it never gates discovery of the artifact itself, which is
        # what previously made an identical run unrecoverable from
        # config+documents+checkpoints+artifact alone without mutable
        # state also surviving.
        state_resolutions_for_this_run = (
            [r for r in (state.get("extraction_review_resolutions") or []) if r.get("run_id") == run_id]
            if previous_run_id == run_id
            else []
        )
        current_run_resolutions = _merge_resolution_log(state_resolutions_for_this_run, existing_artifact)
        for r in current_run_resolutions:
            verified_bundle.verify_existing_resolution(
                r, run_id=run_id, manifest_hash=manifest.manifest_hash,
                schema_hash=manifest.extraction_schema_hash,
                review_fields=machine_review_fields, field_map=schema.field_map(),
            )

        resolved_keys = {(r["record_id"], r["field_id"]) for r in current_run_resolutions}
        review_queue: List[Dict[str, Any]] = [
            {
                "record_id": record_id, "field_id": field_id, "status": field.status.value,
                "field_result": field.model_dump(mode="json"),
            }
            for (record_id, field_id), field in machine_review_fields.items()
            if (record_id, field_id) not in resolved_keys
        ]

        logger.info(
            f"Extraction run {run_id}: {len(results)} document(s), "
            f"{len(review_queue)} field(s) in review queue"
        )

        prisma.included = len(records)
        extraction_results_dicts = [r.model_dump(mode="json") for r in results]
        total_fields = sum(len(r.fields) for r in results)

        resolved_projection = (
            _project_charted_data(records, extraction_results_dicts, current_run_resolutions, current_run_id=run_id)
            if not review_queue
            else []
        )

        _publish_resolution_artifact(
            store, run_id=run_id, manifest_hash=manifest.manifest_hash,
            extraction_schema_hash=manifest.extraction_schema_hash, total_fields=total_fields,
            resolutions=current_run_resolutions,
            remaining_queue_keys=sorted({(e["record_id"], e["field_id"]) for e in review_queue}),
            resolved_projection=resolved_projection,
            existing_verified_artifact=existing_artifact,
        )

    return {
        "extraction_results": extraction_results_dicts,
        "extraction_review_queue": review_queue,
        "requires_human_review": bool(review_queue),
        "extraction_run_id": run_id,
        "extraction_manifest_path": str(store.manifest_path()),
        "extraction_manifest_hash": manifest.manifest_hash,
        "extraction_artifact_paths": {
            "run_dir": str(store.run_dir),
            "checkpoints_dir": str(store.checkpoints_dir),
            "manifest_path": str(store.manifest_path()),
            "resolution_artifact_path": str(_resolution_artifact_path(store)),
        },
        "extraction_schema_hash": manifest.extraction_schema_hash,
        "prisma_counts": prisma.model_dump(),
        # requirement 9: charted_data is ALWAYS set here, one way or the
        # other -- never left to a PRIOR run's stale value under
        # LangGraph's partial-update semantics (codex review finding C).
        "charted_data": resolved_projection,
        # Codex review round-4 finding 1: ALWAYS set explicitly to the
        # freshly discovered-and-verified `current_run_resolutions` --
        # never conditionally reset/omitted.
        "extraction_review_resolutions": current_run_resolutions,
    }


# ---------------------------------------------------------------------------
# Human-review resolution -- the "resume path" for a non-empty
# extraction_review_queue. Not a LangGraph node: called between graph
# invocations once a human has looked at the queued (record_id, field_id)
# entries, producing a state-update dict in the same "only the changed
# keys" shape every node returns -- apply it via
# `graph.update_state(config, resolve_extraction_review_queue(state, {...}))`
# then resume with `graph.invoke(None, config)`.
# ---------------------------------------------------------------------------

def _resolve_extraction_review_queue_locked(
    state: ReviewState,
    resolutions: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    verified_context: Optional[Tuple[ExtractionCheckpointStore, VerifiedRunBundle, ExtractionSchema, str]] = None,
) -> Dict[str, Any]:
    """Applies human decisions to queued (record_id, field_id) entries.

    Codex review, Phase B round-2 finding C1 (hardened rounds 3-5): `state`
    is NEVER trusted at face value here. Before anything else, this
    function:
      1. Requires `state["extraction_run_id"]` to be set (a `None`/empty
         run_id is refused outright).
      2. Loads a `VerifiedRunBundle` and cross-checks the schema in
         `state["config"]` against its own verified schema hash
         (`verified_bundle.load_and_verify_bundle`).
      3. Discovers and FULLY verifies this run's own durable artifact
         (`_read_and_verify_resolution_artifact` -- the SAME centralized
         verifier `extraction_node`/the public loader use, round-5 finding
         2), which also canonicalizes `records` into the manifest's own
         document order (round-5 finding 3).
      4. Merges state's own resolution log with that verified artifact
         (`_merge_resolution_log` -- round-3's "artifact-ahead-of-state
         recovery"), then re-verifies EVERY already-resolved-for-this-run
         entry via `_verify_existing_resolution`.
      5. Re-derives the CURRENTLY review-required (record_id, field_id)
         set directly from the verified bundle's own `results` (never
         from state), subtracts the now-verified already-resolved keys,
         and requires this to EXACTLY equal `state["extraction_review_queue"]`'s
         key set (itself first checked for duplicates) -- a stale,
         cross-run, tampered, or duplicated queue is rejected.
      6. Requires every remaining queue entry's own `field_result` AND
         top-level `status` to be identical to the verified bundle's own
         field result (round-5 finding 1: previously only `field_result`
         was compared, so a tampered top-level `status` -- which flows
         into a newly-published resolution's `original_status` -- went
         undetected).

    `resolutions` maps `(record_id, field_id)` -> a resolution dict:
      - "reviewer", "note", "resolved_at": REQUIRED for every resolution
        -- who decided, why, and when. Each must be a non-BLANK string
        after stripping; `resolved_at` must additionally parse as an
        ISO-8601 timestamp. Appended to the new
        `extraction_review_resolutions` audit-trail state field (never
        used to mutate the original machine `FieldResult`, which is
        preserved unchanged in `extraction_results`/the queue entry's own
        `field_result`).
      - "outcome": REQUIRED, and must be exactly `"extracted"` or
        `"not_reported"` -- these are the ONLY two outcomes that may
        clear a queue entry (requirement 8). To leave an entry queued
        (still undecided, needs more evidence, etc.), simply omit its
        `(record_id, field_id)` key from `resolutions` entirely --
        resolution can be partial across multiple calls, exactly like
        `nodes.fulltext.resolve_full_text_review_queue()`.
      - "value", "evidence", "attested": REQUIRED when
        outcome == "extracted":
          * "value" is validated against the run's OWN, HASH-VERIFIED
            extraction schema (`_validate_human_value` -- the same
            `validate_value_for_field` boundary the machine's own values
            pass, with STRICT cardinality: a list for `multiple`, a
            scalar for `single`, never the other).
          * "evidence" is a non-empty list of STRUCTURED items, each with
            a non-blank `quotation_or_description` and a non-blank
            `locator` (`_validate_human_evidence`).
          * "attested" must be exactly `True`.
        Not used/required when outcome == "not_reported".

    Every resolution is bound to the run's `run_id`, verified
    `manifest_hash`, verified `extraction_schema_hash`, an
    `original_status` DERIVED DIRECTLY from the verified field result's
    own status (round-5 finding 1 -- never from the queue entry's own
    top-level `status`), and a hash of the ORIGINAL (verified) field
    result it resolves (`original_field_result_hash`).

    Codex review round-5 finding 1: EVERY final current-run resolution
    (both pre-existing/merged AND freshly constructed in this call) is
    run through `_verify_existing_resolution()` again, immediately before
    the projection is computed and the artifact is published -- not just
    the pre-existing ones checked earlier.

    PRISMA counts are NEVER touched here (requirement 8) -- this
    function's return dict has no `prisma_counts` key at all.

    `total_fields` and the extraction results fed into the projection are
    built from `bundle.results` (verified), NEVER from mutable
    `state["extraction_results"]`. `records` is the CANONICAL, manifest-
    ordered list `_read_and_verify_resolution_artifact` returns (round-5
    finding 3) -- never state's own (irrelevant) order.

    Once every current-run resolution + the remaining queue is computed,
    this publishes the durable, self-hashed, append-only
    `extraction_resolutions.json` artifact for this run (see
    `_publish_resolution_artifact`'s own docstring -- passed the SAME
    already-verified existing artifact obtained in step 3, never a fresh
    re-read) BEFORE returning -- `charted_data` in the returned state
    update is exactly that artifact's own `resolved_projection` (empty
    until the queue is fully resolved).
    """
    run_id = state.get("extraction_run_id")
    if not run_id:
        raise ValueError(
            "state['extraction_run_id'] is not set -- there is no verifiable extraction run to resolve "
            "against; resolve_extraction_review_queue() refuses to operate on state with no run identity"
        )

    store, bundle, schema, schema_hash = (
        verified_context
        if verified_context is not None
        else verified_bundle.load_and_verify_bundle(state, run_id)
    )
    if store.run_id != run_id or bundle.run_id != run_id:
        raise ValueError("verified resolution context does not belong to state['extraction_run_id']")
    field_map = schema.field_map()

    # The verified, machine-derived set of fields that ACTUALLY require
    # review for THIS run -- never trusted from state, always recomputed
    # from the verified bundle's own results.
    bundle_review_fields = verified_bundle.bundle_review_fields(bundle.results)

    records = [Record.from_dict(r) for r in state["full_text_records"]]

    # The caller MUST hold `_resolution_artifact_lock(store)` for this
    # complete read-verify-merge-write transaction.  Keeping the core
    # lock-free is intentional: the provenance-journal transaction needs
    # one non-reentrant acquisition spanning its prepared append, this
    # resolver core, and its terminal append.
    if not getattr(_RESOLUTION_LOCK_HELD, "value", False):
        raise RuntimeError("_resolve_extraction_review_queue_locked() requires _resolution_artifact_lock")
    existing_artifact, records = _read_and_verify_resolution_artifact(
        store, results=bundle.results, manifest=bundle.manifest, schema=schema, schema_hash=schema_hash,
        records=records,
    )

    # Codex review round-3: merge state's own resolution log with the
    # FULLY VERIFIED artifact (artifact-ahead-of-state recovery), then
    # re-verify EVERY already-resolved-for-this-run entry -- never trust
    # one merely because its run_id/field key happen to match.
    resolution_log = _merge_resolution_log(state.get("extraction_review_resolutions"), existing_artifact)
    for r in resolution_log:
        if r.get("run_id") != run_id:
            continue
        verified_bundle.verify_existing_resolution(
            r, run_id=run_id, manifest_hash=bundle.manifest.manifest_hash, schema_hash=schema_hash,
            review_fields=bundle_review_fields, field_map=field_map,
        )
    already_resolved_keys = {
        (r["record_id"], r["field_id"]) for r in resolution_log if r.get("run_id") == run_id
    }

    # Codex review round-4 finding 2: a duplicated (but otherwise valid)
    # queue entry collapses invisibly under a set-based coverage check --
    # the processing loop below would then append TWO resolutions for the
    # same key, which the publisher's own dict-keyed append-only check
    # silently collapses while still writing the raw duplicate list to
    # disk, producing an artifact the verifier then permanently rejects
    # with no way to republish under the append-only contract. Rejected
    # here instead, before any of that can happen.
    queue_key_list = [(e["record_id"], e["field_id"]) for e in state["extraction_review_queue"]]
    duplicate_queue_keys = sorted({k for k in queue_key_list if queue_key_list.count(k) > 1})
    if duplicate_queue_keys:
        raise ValueError(
            f"state['extraction_review_queue'] contains duplicate (record_id, field_id) entries: "
            f"{duplicate_queue_keys!r} -- refusing to resolve against a queue with ambiguous membership"
        )

    expected_remaining_keys = set(bundle_review_fields) - already_resolved_keys
    actual_queue_keys = set(queue_key_list)
    if actual_queue_keys != expected_remaining_keys:
        raise ValueError(
            "state['extraction_review_queue'] does not exactly match the verified run bundle's own "
            f"unresolved fields (expected {sorted(expected_remaining_keys)!r}, got "
            f"{sorted(actual_queue_keys)!r}) -- refusing to resolve against possibly stale, tampered, "
            "or cross-run queue state"
        )

    queue = list(state["extraction_review_queue"])
    for entry in queue:
        key = (entry["record_id"], entry["field_id"])
        verified_field = bundle_review_fields[key]
        # Codex review round-5 finding 1: exact-check the COMPLETE queue
        # entry -- not just the embedded field_result, but the SEPARATE
        # top-level "status" too (previously untouched by tampering
        # detection even though a resolver used to copy it, unverified,
        # into a newly-published resolution's own original_status).
        if entry.get("field_result") != verified_field.model_dump(mode="json") or entry.get("status") != verified_field.status.value:
            raise ValueError(
                f"extraction_review_queue entry for {key!r} does not match the verified run bundle's "
                "own field result/status -- refusing to resolve against possibly tampered queue state"
            )

    queue_by_key = {(e["record_id"], e["field_id"]): e for e in queue}
    unknown = set(resolutions) - set(queue_by_key)
    if unknown:
        raise KeyError(f"(record_id, field_id) pair(s) not in extraction_review_queue: {sorted(unknown)}")

    remaining_queue: List[Dict[str, Any]] = []

    for entry in queue:
        key = (entry["record_id"], entry["field_id"])
        if key not in resolutions:
            remaining_queue.append(entry)
            continue

        resolution = resolutions[key]
        reviewer = _stripped(resolution.get("reviewer"))
        note = _stripped(resolution.get("note"))
        resolved_at = _stripped(resolution.get("resolved_at"))
        outcome = resolution.get("outcome")
        if not reviewer or not note or not resolved_at:
            raise ValueError(f"resolution for {key!r} must set non-blank reviewer, note, and resolved_at")
        try:
            datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"resolution for {key!r}: resolved_at {resolved_at!r} is not a valid ISO-8601 timestamp")
        if outcome not in ("extracted", "not_reported"):
            raise ValueError(
                f"resolution for {key!r}: outcome must be 'extracted' or 'not_reported' to clear the "
                f"queue, got {outcome!r} -- omit this (record_id, field_id) from `resolutions` to leave "
                "it queued instead"
            )

        verified_field = bundle_review_fields[key]
        original_field_result_hash = sha256_hex(verified_field.model_dump(mode="json"))

        log_entry: Dict[str, Any] = {
            "record_id": entry["record_id"],
            "field_id": entry["field_id"],
            # Codex review round-5 finding 1: derived DIRECTLY from the
            # VERIFIED field result's own status -- never from the queue
            # entry's own top-level "status" (already exact-checked above,
            # but never relied upon alone for this).
            "original_status": verified_field.status.value,
            "original_field_result_hash": original_field_result_hash,
            "outcome": outcome,
            "reviewer": reviewer,
            "note": note,
            "resolved_at": resolved_at,
            "run_id": run_id,
            "manifest_hash": bundle.manifest.manifest_hash,
            "extraction_schema_hash": schema_hash,
        }

        if outcome == "extracted":
            field_spec = field_map.get(entry["field_id"])
            if field_spec is None:
                raise ValueError(
                    f"resolution for {key!r}: field_id {entry['field_id']!r} is not present in the "
                    "run's own extraction schema -- cannot validate a human-supplied value against it"
                )
            value = resolution.get("value")
            evidence = resolution.get("evidence")
            attested = resolution.get("attested")
            verified_bundle.validate_human_value(field_spec, value, key)
            verified_bundle.validate_human_evidence(evidence, key)
            if attested is not True:
                raise ValueError(
                    f"resolution for {key!r}: outcome='extracted' requires explicit human attestation "
                    "('attested': True) -- a human-supplied value is never accepted without it"
                )
            log_entry["value"] = value
            log_entry["evidence"] = evidence
            log_entry["attested"] = True

        resolution_log.append(log_entry)

    current_run_resolutions = [r for r in resolution_log if r.get("run_id") == run_id]

    # Codex review round-5 finding 1: run EVERY final current-run
    # resolution (both merged/pre-existing AND freshly constructed above)
    # through the full verifier immediately before projection/publication
    # -- a freshly-constructed resolution was previously published
    # without ever passing through this.
    for r in current_run_resolutions:
        verified_bundle.verify_existing_resolution(
            r, run_id=run_id, manifest_hash=bundle.manifest.manifest_hash, schema_hash=schema_hash,
            review_fields=bundle_review_fields, field_map=field_map,
        )

    extraction_results_dicts = [result.model_dump(mode="json") for result in bundle.results]
    total_fields = sum(len(result.fields) for result in bundle.results)
    remaining_keys = sorted({(e["record_id"], e["field_id"]) for e in remaining_queue})

    resolved_projection = (
        _project_charted_data(records, extraction_results_dicts, current_run_resolutions, current_run_id=run_id)
        if not remaining_queue
        else []
    )

    _publish_resolution_artifact(
        store, run_id=run_id, manifest_hash=bundle.manifest.manifest_hash,
        extraction_schema_hash=schema_hash, total_fields=total_fields,
        resolutions=current_run_resolutions, remaining_queue_keys=remaining_keys,
        resolved_projection=resolved_projection, existing_verified_artifact=existing_artifact,
    )

    return {
        "extraction_review_queue": remaining_queue,
        "extraction_review_resolutions": resolution_log,
        "requires_human_review": bool(remaining_queue),
        "charted_data": resolved_projection,
    }


def resolve_extraction_review_queue(
    state: ReviewState,
    resolutions: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """Public, lock-owning wrapper for the existing human resolver.

    The implementation lives in `_resolve_extraction_review_queue_locked()`
    so the provenance-journal transaction can call it while holding the same
    non-reentrant lock as its `prepared` and terminal journal writes.
    """
    run_id = state.get("extraction_run_id")
    if not run_id:
        raise ValueError(
            "state['extraction_run_id'] is not set -- there is no verifiable extraction run to resolve "
            "against; resolve_extraction_review_queue() refuses to operate on state with no run identity"
        )
    context = verified_bundle.load_and_verify_bundle(state, run_id)
    with _resolution_artifact_lock(context[0]):
        return _resolve_extraction_review_queue_locked(state, resolutions, verified_context=context)


def _load_validated_suggestion_file(
    path: Path, *, store: ExtractionCheckpointStore, extraction_run_id: str,
    record_id: str, field_id: str,
) -> ValidatedSuggestion:
    """Load one immutable suggestion and bind its content to its path."""
    expected_dir = (store.run_dir / "extraction_resolution_suggestions").resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != expected_dir:
        raise ValueError("suggestion file must live in this extraction run's suggestions directory")
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    suggestion = ValidatedSuggestion.model_validate(data)
    if (
        suggestion.extraction_run_id != extraction_run_id
        or suggestion.record_id != record_id
        or suggestion.field_id != field_id
    ):
        raise ValueError("suggestion file identity does not match the decision work item")
    expected_name = f"{safe_key(extraction_run_id, record_id, field_id)}__attempt-{suggestion.attempt}.json"
    if path.name != expected_name:
        raise ValueError(f"suggestion filename {path.name!r} does not match expected {expected_name!r}")
    return suggestion


def _verify_decision_packet_entry(
    packet: ReviewPacket, *, record_id: str, field_id: str,
    context: Tuple[ExtractionCheckpointStore, VerifiedRunBundle, ExtractionSchema, str],
) -> None:
    """Re-derive one packet entry from the verified checkpoint bundle."""
    _store, bundle, schema, _schema_hash = context
    entry = packet_entry(packet, record_id, field_id)
    verified_fields = verified_bundle.bundle_review_fields(bundle.results)
    key = (record_id, field_id)
    field_result = verified_fields.get(key)
    field_spec = schema.field_map().get(field_id)
    if field_result is None or field_spec is None:
        raise ValueError(f"review packet work item {key!r} is absent from the verified run")
    if entry.field_result != field_result or entry.field_spec != field_spec:
        raise ValueError(f"review packet entry for {key!r} does not match the verified run")
    expected_evidence = _evidence_index_for_field(
        record_id=record_id, field_id=field_id, field_result=field_result,
    )
    expected_candidates = _candidates_for_field(
        record_id=record_id, field_id=field_id, field_result=field_result,
    )
    if entry.evidence_index != expected_evidence or entry.candidates != expected_candidates:
        raise ValueError(f"review packet indexes for {key!r} do not match the verified run")


def _resolution_artifact_entries_locked(
    state: ReviewState,
    context: Tuple[ExtractionCheckpointStore, VerifiedRunBundle, ExtractionSchema, str],
) -> List[Dict[str, Any]]:
    store, bundle, schema, schema_hash = context
    records = [Record.from_dict(record) for record in state["full_text_records"]]
    artifact, _records = _read_and_verify_resolution_artifact(
        store, results=bundle.results, manifest=bundle.manifest, schema=schema,
        schema_hash=schema_hash, records=records,
    )
    return list((artifact or {}).get("resolutions", []))


def _verify_resolution_decisions_locked(
    state: ReviewState,
    packet: ReviewPacket,
    context: Tuple[ExtractionCheckpointStore, VerifiedRunBundle, ExtractionSchema, str],
) -> None:
    store, bundle, _schema, schema_hash = context
    if (
        packet.run_id != store.run_id
        or packet.manifest_hash != bundle.manifest.manifest_hash
        or packet.extraction_schema_hash != schema_hash
    ):
        raise ValueError("review packet is not bound to the verified extraction run being reconciled")
    entries = read_decision_journal(store)
    validate_journal_transitions(entries)
    resolutions = _resolution_artifact_entries_locked(state, context)
    for entry in entries:
        if entry.status != "applied":
            continue
        _verify_decision_packet_entry(
            packet, record_id=entry.record_id, field_id=entry.field_id, context=context,
        )
        matches = [
            resolution for resolution in resolutions
            if resolution.get("run_id") == entry.run_id
            and resolution.get("record_id") == entry.record_id
            and resolution.get("field_id") == entry.field_id
            and resolution.get("resolved_at") == entry.resolved_at
        ]
        if len(matches) != 1 or not resolution_matches_intent(matches[0], entry, packet):
            raise ValueError(
                f"applied decision journal entry {entry.journal_id!r} does not exactly match "
                "extraction_resolutions.json"
            )


def apply_extraction_resolution_decision(
    state: ReviewState,
    packet: ReviewPacket,
    *,
    record_id: str,
    field_id: str,
    outcome: Literal["extracted", "not_reported"],
    final_value: Any = None,
    selected_evidence_ids: Tuple[str, ...] = (),
    reviewer: str,
    note: str,
    resolved_at: str,
    decision_origin: DecisionOrigin,
    suggestion_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Apply one human decision as a prepared -> applied/failed transaction.

    ``attested=True`` is minted here from the human action.  It is never
    copied from a model confidence or suggestion field.
    """
    run_id = state.get("extraction_run_id")
    if not run_id:
        raise ValueError("state['extraction_run_id'] is required")
    context = verified_bundle.load_and_verify_bundle(state, run_id)
    store, bundle, _schema, schema_hash = context
    if (
        packet.run_id != run_id
        or packet.manifest_hash != bundle.manifest.manifest_hash
        or packet.extraction_schema_hash != schema_hash
    ):
        raise ValueError("review packet is not bound to this verified extraction run")
    _verify_decision_packet_entry(
        packet, record_id=record_id, field_id=field_id, context=context,
    )
    if outcome == "not_reported" and (final_value is not None or selected_evidence_ids):
        raise ValueError("outcome='not_reported' cannot carry final_value or selected_evidence_ids")

    # Proves that this work item and every selected ID occur exactly once in
    # the packet before any journal bytes are written.
    evidence = rehydrate_evidence(
        packet, record_id=record_id, field_id=field_id, evidence_ids=selected_evidence_ids,
    ) if outcome == "extracted" else []

    suggestion: Optional[ValidatedSuggestion] = None
    attempt_path: Optional[Path] = None
    if suggestion_file is not None:
        attempt_path = Path(suggestion_file)
        suggestion = _load_validated_suggestion_file(
            attempt_path, store=store, extraction_run_id=run_id,
            record_id=record_id, field_id=field_id,
        )

    if decision_origin == "human":
        if suggestion is not None:
            raise ValueError("a fully human decision cannot reference a suggestion file")
        provenance: Dict[str, Any] = {}
        proposed_value = None
    elif decision_origin == "machine_only":
        raise ValueError("decision_origin='machine_only' is reserved and disabled in human-gated mode")
    else:
        if suggestion is None or attempt_path is None:
            raise ValueError("machine_recommended_human_approved requires a validated suggestion file")
        packet_item = packet_entry(packet, record_id, field_id)
        known_evidence_ids = set(packet_item.evidence_index)
        known_candidate_ids = {candidate.candidate_id for candidate in packet_item.candidates}
        if not set(suggestion.evidence_ids).issubset(known_evidence_ids):
            raise ValueError("suggestion references evidence IDs absent from the verified review packet")
        suggestion_candidate_ids = (
            set(suggestion.candidate_ids_merged)
            | set(suggestion.candidate_ids_retained)
            | set(suggestion.candidate_ids_rejected)
        )
        if not suggestion_candidate_ids.issubset(known_candidate_ids):
            raise ValueError("suggestion references candidate IDs absent from the verified review packet")
        relative_attempt_path = attempt_path.resolve().relative_to(store.run_dir.resolve()).as_posix()
        provenance = {
            "suggestion_id": suggestion.suggestion_id,
            "self_hash": suggestion.self_hash,
            "attempt_file_path": relative_attempt_path,
            "model": suggestion.ai_use_event_draft.model,
            "prompt_template_version": suggestion.ai_use_event_draft.prompt_template_version,
            "call_event_id": suggestion.ai_use_event_id,
        }
        proposed_value = suggestion.typed_value if suggestion.typed_value is not None else suggestion.value

    intent = ResolutionDecisionIntent.model_validate({
        "run_id": run_id,
        "record_id": record_id,
        "field_id": field_id,
        "outcome": outcome,
        "proposed_value": proposed_value,
        "final_value": final_value if outcome == "extracted" else None,
        "selected_evidence_ids": selected_evidence_ids if outcome == "extracted" else (),
        "reviewer": reviewer.strip() if isinstance(reviewer, str) else reviewer,
        "note": note.strip() if isinstance(note, str) else note,
        "resolved_at": resolved_at.strip() if isinstance(resolved_at, str) else resolved_at,
        "attested": True if outcome == "extracted" else None,
        "decision_origin": decision_origin,
        **provenance,
    })
    resolution: Dict[str, Any] = {
        "outcome": intent.outcome,
        "reviewer": intent.reviewer,
        "note": intent.note,
        "resolved_at": intent.resolved_at,
    }
    if intent.outcome == "extracted":
        resolution.update(value=intent.final_value, evidence=evidence, attested=True)

    with _resolution_artifact_lock(store):
        append_decision_entry(store, intent, status="prepared")
        try:
            update = _resolve_extraction_review_queue_locked(
                state, {(record_id, field_id): resolution}, verified_context=context,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:2000]
            append_decision_entry(store, intent, status="failed", reason=reason)
            raise
        append_decision_entry(store, intent, status="applied")
        _verify_resolution_decisions_locked(state, packet, context)
    return update


def reconcile_resolution_journal(state: ReviewState, packet: ReviewPacket) -> None:
    """Repair a torn tail and terminalize any orphaned prepared entries."""
    run_id = state.get("extraction_run_id")
    if not run_id:
        raise ValueError("state['extraction_run_id'] is required")
    context = verified_bundle.load_and_verify_bundle(state, run_id)
    store = context[0]
    with _resolution_artifact_lock(store):
        entries = repair_torn_journal_tail(store)
        validate_journal_transitions(entries)
        resolutions = _resolution_artifact_entries_locked(state, context)
        grouped: Dict[str, List[Any]] = {}
        for entry in entries:
            grouped.setdefault(entry.journal_id, []).append(entry)
        for group in grouped.values():
            if len(group) != 1 or group[0].status != "prepared":
                continue
            prepared = group[0]
            _verify_decision_packet_entry(
                packet, record_id=prepared.record_id, field_id=prepared.field_id, context=context,
            )
            matches = [
                resolution for resolution in resolutions
                if resolution.get("run_id") == prepared.run_id
                and resolution.get("record_id") == prepared.record_id
                and resolution.get("field_id") == prepared.field_id
                and resolution.get("resolved_at") == prepared.resolved_at
            ]
            if len(matches) == 1 and resolution_matches_intent(matches[0], prepared, packet):
                append_decision_entry(store, prepared, status="applied")
            else:
                append_decision_entry(
                    store, prepared, status="failed",
                    reason="interrupted, requires manual re-decision",
                )
        _verify_resolution_decisions_locked(state, packet, context)


def verify_resolution_decisions(state: ReviewState, packet: ReviewPacket) -> None:
    """Fail closed unless every applied decision matches the authority."""
    run_id = state.get("extraction_run_id")
    if not run_id:
        raise ValueError("state['extraction_run_id'] is required")
    context = verified_bundle.load_and_verify_bundle(state, run_id)
    with _resolution_artifact_lock(context[0]):
        _verify_resolution_decisions_locked(state, packet, context)
