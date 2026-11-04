"""Shared human-resolution helpers -- Stage 3 node instrumentation
(the Stage 3 node-instrumentation design record, section 2.5). Used by
`screening_node`'s own human-apply step and
`slr_pipeline.nodes.fulltext.resolve_full_text_review_queue()`:

  - `compute_resolution_id()`: a durable, content-addressed identity for
    one human resolution, bound to the exact ledger baseline the human
    was reviewing.
  - `preflight_check_baseline_still_current()`: a fast, UNLOCKED check
    mirroring `emit_selection_decision()`'s own authoritative in-lock
    `expected_current_tail_decision_id` precondition -- fails an
    obviously-stale submission early, before other per-record work.
    Never the source of truth by itself; see its own docstring.
  - `validate_submission_coverage()`: the shared duplicate/unknown/
    (optionally) missing record-id check every human-resolution submission
    validates BEFORE any side effect is applied.
  - `StaleReviewError`: raised by the preflight check above.

Existence of a reviewed baseline in the ledger chain is NOT sufficient by
itself -- the baseline must still equal the CURRENT tail (or the
resolution must be an exact, already-applied replay of that same
baseline), or `emit_selection_decision()` would silently supersede a
decision the human never actually reviewed. This module's preflight check
and `emit_selection_decision()`'s own atomic precondition apply the
IDENTICAL three-way logic; only the atomic, in-lock version is
authoritative (a second writer can move the tail between this module's
own unlocked read and the caller's later `emit_selection_decision()` call
-- a genuine race the preflight check alone cannot close).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.selection_ledger import SelectionDecisionStore, _tails_by_key

RESOLUTION_SCHEMA_VERSION = "resolution_v1"


class StaleReviewError(RuntimeError):
    """A human resolution's own persisted `reviewed_baseline_decision_id`
    is no longer the current ledger tail for its `(record_id, stage)`, and
    this call is not an exact replay of an already-applied resolution for
    that same baseline -- the review is stale. Raised only by
    `preflight_check_baseline_still_current()` (the fast, unlocked check);
    `emit_selection_decision()`'s own in-lock `expected_current_tail_decision_id`
    precondition (`SelectionLedgerPreconditionError`) is the authoritative
    guarantee -- this preflight exists purely to fail obviously-stale
    submissions early, before any other per-record work, never to replace
    that atomic check."""


def compute_resolution_id(
    *, stage: str, record_id: str, action: str, reviewer: str, note: str, resolved_at: str,
    reviewed_baseline_decision_id: Optional[str], exclusion_reason: Optional[str] = None,
) -> str:
    """Content-addressed, baseline-bound identity for one human
    resolution -- hashes every decision-relevant field, INCLUDING the
    baseline the human was reviewing at resolution time (not just
    record_id/stage/action/reviewer/note/resolved_at). Binding the
    baseline in means a resubmission against a genuinely DIFFERENT
    current baseline (the reviewer's own earlier decision was itself
    superseded since they last looked) is structurally a different
    `resolution_id`, never confused with a stale replay of the old one.
    `reviewed_baseline_decision_id=None` is itself a meaningful, distinct
    value (a fresh root -- no prior ai/rule_based baseline ever existed
    for this record), never conflated with "not yet computed."""
    return "humanres-" + sha256_hex({
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "stage": stage,
        "record_id": record_id,
        "action": action,
        "exclusion_reason": exclusion_reason,
        "reviewer": reviewer,
        "note": note,
        "resolved_at": resolved_at,
        "reviewed_baseline_decision_id": reviewed_baseline_decision_id,
    })


def preflight_check_baseline_still_current(
    *, store: SelectionDecisionStore, record_id: str, stage: str,
    reviewed_baseline_decision_id: Optional[str], resolution_source_event_id: str,
) -> None:
    """Fast, UNLOCKED preflight mirroring `emit_selection_decision()`'s
    own authoritative in-lock logic exactly: the baseline still equals
    the current tail; OR this exact resolution was already applied and
    IS (or matches) the current tail (an idempotent artifact-ahead-of-
    state replay); otherwise the review is stale and this raises. A
    `None` baseline (no prior ai/rule_based decision ever existed for
    this record) requires either an empty chain (this resolution would
    become the chain's own root) or an exact replay of the resolution
    that already created that root.

    This is a convenience fast-fail only -- see the module docstring for
    why it cannot, by itself, close the TOCTOU race that
    `emit_selection_decision()`'s own atomic precondition exists to
    close.

    Codex review round: a prior version treated "resolution_source_event_id
    exists ANYWHERE in the chain" as sufficient for a valid replay -- for
    `A0 -> H -> A1`, a stale replay of `H`'s own source_event_id then
    incorrectly passed this preflight (since `H` is genuinely IN the
    chain, just no longer the tail), even though `A1` is current. An
    existing match only counts as a valid replay when the CURRENT TAIL
    ITSELF is that match -- exactly mirroring `emit_selection_decision()`'s
    own authoritative logic (`tail.decision_id == existing.decision_id`
    after finding the source_event_id match), never "found somewhere."""
    chain = store.load_key(record_id=record_id, stage=stage)
    tail = _tails_by_key(chain).get((record_id, stage)) if chain else None
    tail_is_the_replayed_entry = tail is not None and tail.source_event_id == resolution_source_event_id

    if reviewed_baseline_decision_id is None:
        if tail is None or tail_is_the_replayed_entry:
            return
        raise StaleReviewError(
            f"record {record_id!r}/{stage!r}: this resolution was reviewed against an empty chain "
            f"(no prior ai/rule_based baseline), but the current tail is {tail.decision_id!r} -- "
            "something else has since decided this record; re-review against the current state"
        )

    if tail is not None and tail.decision_id == reviewed_baseline_decision_id:
        return
    if tail_is_the_replayed_entry:
        return
    raise StaleReviewError(
        f"record {record_id!r}/{stage!r}: human reviewed baseline {reviewed_baseline_decision_id!r}, "
        f"but the current tail is {(tail.decision_id if tail is not None else None)!r} -- the review "
        "is stale, something else changed the decision since this human looked at it; re-review "
        "against the current state rather than silently superseding a decision this resolution "
        "never saw"
    )


def append_resolution_log_entry_deduplicated(
    resolution_log: List[Dict[str, Any]], entry: Dict[str, Any], *, resolution_id: str,
) -> List[Dict[str, Any]]:
    """Appends `entry` (which must itself carry `"resolution_id": resolution_id`)
    to `resolution_log`, UNLESS an entry with the identical `resolution_id`
    is already present -- an identical human resubmission is idempotent at
    the LEDGER level (`emit_selection_decision()`'s own replay match), and
    the audit-trail log must be equally idempotent, never gain a duplicate
    entry for a resubmission that changed nothing (codex review: a probe
    replaying an identical submission grew the audit log from 1 to 2
    entries with no new ledger decision or provider call). Raises if an
    existing entry with the same `resolution_id` has DIFFERENT content --
    `resolution_id` is content-addressed over exactly these fields, so a
    mismatch here means something is structurally wrong, never silently
    kept as two versions."""
    for existing in resolution_log:
        if existing.get("resolution_id") == resolution_id:
            if existing != entry:
                raise ValueError(
                    f"resolution_id {resolution_id!r} already exists in the resolution log with "
                    f"different content ({existing!r} != {entry!r}) -- resolution_id is content-"
                    "addressed, so this should be structurally unreachable; refusing to silently "
                    "keep either version"
                )
            return resolution_log
    return resolution_log + [entry]


def validate_submission_coverage(
    *, submitted_record_ids: Iterable[str], queued_record_ids: Iterable[str], require_full_coverage: bool,
) -> None:
    """Shared coverage check applied BEFORE inspecting any individual
    submitted entry's own fields, so a badly-shaped submission never gets
    partially processed: no duplicate submitted record IDs; no unknown
    record IDs (submitted but not actually queued); and, only when
    `require_full_coverage=True` (screening's own contract -- exact
    response-set equality, never a default-to-exclude for a missing
    entry), no missing record IDs either (queued but not submitted).
    `require_full_coverage=False` preserves `resolve_full_text_review_queue()`'s
    own pre-existing, tested, and intentional support for resolving a
    queue partially across multiple separate calls."""
    submitted_list = list(submitted_record_ids)
    queued_set = set(queued_record_ids)

    seen: set = set()
    duplicates = sorted({rid for rid in submitted_list if rid in seen or seen.add(rid)})
    if duplicates:
        raise ValueError(f"duplicate record_id(s) in submission: {duplicates}")

    submitted_set = set(submitted_list)
    unknown = sorted(submitted_set - queued_set)
    if unknown:
        raise ValueError(f"record_id(s) not in queue: {unknown}")

    if require_full_coverage:
        missing = sorted(queued_set - submitted_set)
        if missing:
            raise ValueError(f"record_id(s) in queue but missing from submission: {missing}")
