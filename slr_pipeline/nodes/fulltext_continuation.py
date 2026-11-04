"""
continuation_fulltext_node -- Track 3 continuation plan Unit C.

Thin orchestration only: reuses `nodes/fulltext.py`'s own
`acquire_and_confirm_records()` (Unit C's verbatim-moved acquisition/
eligibility-gate loop) and `_run_full_text_eligibility()` unmodified,
applied to a smaller input set -- no acquisition/eligibility logic is
duplicated here.

A bare `fn(state)` graph entry point (`graphs/scoping_continuation.py`,
Unit D), like every other node this continuation's graph reuses -- no
closure/factory binding exists anywhere in this plan. It therefore never
receives `parent_snapshot`/`continuation_manifest`/`parent_bundle_path`
(launcher-scope-only objects with no mechanism to reach a bare `fn(state)`
node); it reads the launcher's already-computed values straight off
seeded state instead:

  - `state["continuation_partition"]` (Unit A item 13): the launcher's own
    `derive_continuation_partition()` result, `model_dump()`d --
    `preserved_included_ids`/`preserved_excluded_ids`/`retrieved_ids`/
    `not_retrieved_ids`, never recomputed here.
  - `state["preserved_included_records"]` (Unit A item 14): the launcher's
    own `acquire_preserved_included_pdfs()` return value, `model_dump()`d
    -- each record already pointed at its local, freshly-copied
    `full_text_path`. No copy, no hash check, no `Track3ConfigError`
    handling happens here; all of that already happened at launch.

For `retrieved_ids`: this node performs NO copy/hash logic of its own.
The launcher's `acquire_retrieved_pdfs()` (Unit A item 15) has already
copied and verified each one's PDF into this continuation's own `pdf_dir`
under the SAME naming convention `_dest_path()`/`_find_local_pdf()` both
use -- so `acquire_and_confirm_records()`'s own local-PDF short-circuit
finds them transparently, and the OA-download/network branch is never
taken for this bucket. `_run_full_text_eligibility()` -- the only place a
real FT_SCREENING/OpenAI call happens -- runs `if confirmed:` only,
mirroring `fulltext_node()`'s own guard, since its documented precondition
("never called with an empty confirmed list") would otherwise be
violated whenever `retrieved_ids` is empty (e.g. a pure
report-regeneration-pass continuation).

For `not_retrieved_ids`: never passed into `acquire_and_confirm_records()`
at all -- no PDF, no queue entry, no FT_SCREENING call, no exclusion. This
is what makes "not retrieved, never an eligibility exclusion" true by
construction, generically, for whichever ids end up in this bucket.

For `preserved_ids` (both include and exclude): never passed into
`acquire_and_confirm_records()`/`_run_full_text_eligibility()` -- FT_SCREENING
never re-runs for either. `preserved_included_ids`' own records are merged
read-only into this node's `full_text_records` output.
`preserved_excluded_ids` are dropped from `full_text_records`, same as a
normal `fulltext_node()` run drops any FT_SCREENING exclusion.

`prisma_counts` is built ADDITIVELY against the item 9-seeded parent
baseline already present in `state["prisma_counts"]` on entry (unlike
`fulltext_node()`'s own plain-overwrite tail, correct only for a run
starting from zero): `assessed_full_text`/`excluded_full_text` are summed;
`excluded_full_text_reasons`/`fulltext_retrieval` are instead RECOMPUTED as
a fresh histogram over the CURRENT status of every `included_records` id,
derived structurally from the partition (never a second independent
oracle) -- key-wise-summing `fulltext_retrieval` against the parent's own
dict double-counts every record whose status transitions this run (a
record the parent recorded as `"not_retrieved"` that is now `"parsed"`
this session), since the parent's own `"not_retrieved"` count already
included it. The resulting histogram totals `len(included_records)` by
construction -- asserted below, not merely assumed. (`excluded_full_text_
reasons` has no such transition risk: the parent's own seeded dict only
ever reflects `preserved_excluded_ids`' reasons -- untouched this run --
and this node's own newly-computed reasons dict only ever concerns a
disjoint subset of `retrieved_ids`, so key-wise summing the two is safe.)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from slr_pipeline.config import ReviewConfig
from slr_pipeline.nodes.fulltext import (
    FULL_TEXT_ELIGIBILITY_STAGE,
    _full_text_origin_and_oversight,
    _run_full_text_eligibility,
    acquire_and_confirm_records,
)
from slr_pipeline.run_identity import resolve_review_run_id
from slr_pipeline.selection_ledger import SelectionDecisionStore
from slr_pipeline.state import PRISMACounts, Record, ReviewState
from slr_pipeline.track3_run import Track3ConfigError


def continuation_fulltext_node(state: ReviewState) -> Dict[str, Any]:
    cfg = ReviewConfig.from_dict(state["config"])
    included_records = [Record.from_dict(r) for r in state["included_records"]]

    partition = state["continuation_partition"]
    preserved_included_ids = set(partition["preserved_included_ids"])
    preserved_excluded_ids = set(partition["preserved_excluded_ids"])
    retrieved_ids = set(partition["retrieved_ids"])
    not_retrieved_ids = set(partition["not_retrieved_ids"])

    output_dir = cfg.ensure_output_dir()
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    review_run_id = resolve_review_run_id(state, allow_config=True)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)

    contact_email = os.environ.get("PUBMED_EMAIL")
    if not contact_email or contact_email.endswith("@example.com"):
        contact_email = None

    retrieved_records = [r for r in included_records if r.id in retrieved_ids]
    confirmed, _manual_worklist, retrieval_counts, review_queue = acquire_and_confirm_records(
        records=retrieved_records, cfg=cfg, pdf_dir=pdf_dir, contact_email=contact_email,
    )

    passing: List[Record] = []
    excluded_count = 0
    excluded_reasons: Dict[str, int] = {}
    ft_run_id: Optional[str] = None
    ft_execution_fingerprint: Optional[str] = None
    ft_artifact_paths: Optional[Dict[str, str]] = None
    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})

    if confirmed:
        (
            passing, ft_queue_entries, excluded_count, excluded_reasons,
            ft_run_id, ft_execution_fingerprint, ft_artifact_paths,
        ) = _run_full_text_eligibility(confirmed, cfg, output_dir, ledger_store)
        review_queue.extend(ft_queue_entries)

        origin, oversight = _full_text_origin_and_oversight(ledger_store)
        if oversight is not None:
            oversight_mode[FULL_TEXT_ELIGIBILITY_STAGE] = oversight
        if origin is not None:
            automation_origin[FULL_TEXT_ELIGIBILITY_STAGE] = origin

    preserved_included_records = [
        Record.from_dict(r) for r in (state.get("preserved_included_records") or [])
    ]
    full_text_records = preserved_included_records + passing

    fulltext_retrieval: Dict[str, int] = {}
    if preserved_included_ids or preserved_excluded_ids:
        fulltext_retrieval["parsed"] = len(preserved_included_ids) + len(preserved_excluded_ids)
    for status, count in retrieval_counts.items():
        fulltext_retrieval[status] = fulltext_retrieval.get(status, 0) + count
    if not_retrieved_ids:
        fulltext_retrieval["not_retrieved"] = fulltext_retrieval.get("not_retrieved", 0) + len(not_retrieved_ids)

    total_represented = sum(fulltext_retrieval.values())
    if total_represented != len(included_records):
        raise Track3ConfigError(
            f"continuation_fulltext_node: recomputed fulltext_retrieval histogram "
            f"represents {total_represented} record(s), expected "
            f"{len(included_records)} (len(included_records)) -- partition "
            f"buckets do not exhaustively/uniquely cover included_records."
        )

    excluded_full_text_reasons: Dict[str, int] = dict(
        state["prisma_counts"].get("excluded_full_text_reasons") or {}
    )
    for reason, count in excluded_reasons.items():
        excluded_full_text_reasons[reason] = excluded_full_text_reasons.get(reason, 0) + count

    prisma = PRISMACounts(**state["prisma_counts"])
    prisma.assessed_full_text = state["prisma_counts"]["assessed_full_text"] + len(confirmed)
    prisma.excluded_full_text = state["prisma_counts"]["excluded_full_text"] + excluded_count
    prisma.excluded_full_text_reasons = excluded_full_text_reasons
    prisma.fulltext_retrieval = fulltext_retrieval

    return {
        "full_text_records": [r.to_dict() for r in full_text_records],
        "full_text_review_queue": review_queue,
        "requires_human_review": bool(review_queue),
        "prisma_counts": prisma.model_dump(),
        "ft_screening_run_id": ft_run_id,
        "ft_screening_execution_fingerprint": ft_execution_fingerprint,
        "ft_screening_artifact_paths": ft_artifact_paths,
        "review_run_id": review_run_id,
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
        "selection_decision_ledger_dir": str(ledger_store.decisions_dir),
    }
