"""Track 3 continuation plan (squishy-petting-starlight.md) Unit E --
cumulative `ReportingSourceSnapshot` for a continuation run.

`compute_selection_splits()` (`nodes/reporting.py`) asserts EXACT
record-ID-set equality per stage (`dedup`, `title_abstract_screening`,
`full_text_eligibility`) between its own expected sets and the snapshot's
ledger tails. A continuation never re-runs `dedup`/`title_abstract_
screening`, so a snapshot built purely from the continuation's own live
state (`capture_reporting_snapshot()`) can never satisfy those checks --
the whole parent corpus is missing. `build_cumulative_reporting_snapshot()`
closes that gap by building the snapshot as a verified UNION of the
parent bundle's own published snapshot (`from_bundle(parent_bundle)`) and
the continuation's own live snapshot, never by re-deriving/re-computing
anything the parent already decided.

`preserved_decision_citations` (a new `ContinuationReviewState` field) is
never trusted on its own: every citation triple is independently
re-verified against `parent_snapshot.ledger_records`'s own matching
record before being unioned in -- a tampered citation (wrong
`content_hash`) or one naming a decision that doesn't actually exist in
the parent bundle both raise `Track3ConfigError`, never silently pass
through.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from slr_pipeline.ai_traceability_report import _STAGE_EVENT_STAGES
from slr_pipeline.config import ReviewConfig
from slr_pipeline.nodes.reporting import ReportingSourceSnapshot, VerifiedFile, capture_reporting_snapshot
from slr_pipeline.reference_bundle import Bundle, apply_parent_oversight_union, from_bundle
from slr_pipeline.state import ReviewState
from slr_pipeline.track3_run import Track3ConfigError

# The scoped union of parent-only-stage AIUseEvent stages: `full_text_
# eligibility`/`extraction`/`quality_appraisal`/`synthesis_*` are always
# sourced from the continuation's OWN base snapshot alone (it genuinely
# runs, or re-runs, those stages itself) -- only `dedup`/`search_strategy`/
# `title_abstract_screening`/`screening_qc_second_rater` events, which a
# continuation never re-produces, need to be pulled in from the parent.
_PARENT_ONLY_EVENT_STAGES = frozenset().union(
    _STAGE_EVENT_STAGES["dedup"],
    _STAGE_EVENT_STAGES["search_strategy"],
    _STAGE_EVENT_STAGES["title_abstract_screening"],
    _STAGE_EVENT_STAGES["screening_qc_second_rater"],
)

_PARENT_LEDGER_STAGES = ("dedup", "title_abstract_screening")


def _union_ledger_records(
    *, parent_snapshot: ReportingSourceSnapshot, base_snapshot: ReportingSourceSnapshot,
    preserved_decision_citations: List[Dict[str, Any]],
) -> List[VerifiedFile]:
    # dedup/title_abstract_screening: 100% the parent's own full history
    # (not just current tails -- `compute_selection_splits()`'s own
    # `_tails_by_key()` call reduces that itself) -- a continuation never
    # re-runs either stage.
    result: List[VerifiedFile] = [
        vf for vf in parent_snapshot.ledger_records if vf.model.stage in _PARENT_LEDGER_STAGES
    ]

    # full_text_eligibility, preserved half: every citation is
    # independently re-verified against the parent's own ledger before
    # being trusted -- never taken from the citation file alone.
    cited_record_ids: List[str] = []
    for citation in preserved_decision_citations:
        record_id = citation["record_id"]
        decision_id = citation["decision_id"]
        content_hash = citation["content_hash"]
        matching = [
            vf for vf in parent_snapshot.ledger_records
            if vf.model.stage == "full_text_eligibility"
            and vf.model.record_id == record_id and vf.model.decision_id == decision_id
        ]
        if not matching:
            raise Track3ConfigError(
                f"preserved_decision_citations: no full_text_eligibility decision {decision_id!r} for "
                f"record {record_id!r} exists in the parent bundle's own ledger")
        vf = matching[0]
        if vf.model.content_hash != content_hash:
            raise Track3ConfigError(
                f"preserved_decision_citations: citation for record {record_id!r} (decision "
                f"{decision_id!r}) claims content_hash {content_hash!r}, but the parent bundle's own "
                f"matching ledger decision has content_hash {vf.model.content_hash!r} -- tampered citation")
        if record_id in cited_record_ids:
            raise Track3ConfigError(
                f"preserved_decision_citations: record {record_id!r} is cited more than once")
        cited_record_ids.append(record_id)
        result.append(vf)

    # full_text_eligibility, retrieved half: the continuation's own live
    # ledger tails. Overlap with the preserved half would mean the same
    # record is claimed as both already-decided and newly-decided this
    # session -- a partition invariant `derive_continuation_partition()`
    # already enforces upstream, re-checked here defensively.
    continuation_ft_record_ids = {
        vf.model.record_id for vf in base_snapshot.ledger_records if vf.model.stage == "full_text_eligibility"
    }
    overlap = set(cited_record_ids) & continuation_ft_record_ids
    if overlap:
        raise Track3ConfigError(
            f"ledger union: record(s) {sorted(overlap)!r} appear in both the preserved-decision "
            "citations and the continuation's own new full_text_eligibility decisions")

    result.extend(base_snapshot.ledger_records)
    return result


def _union_ft_native_provenance(
    *, parent_snapshot: ReportingSourceSnapshot, base_snapshot: ReportingSourceSnapshot,
):
    if base_snapshot.ft_native_provenance is None:
        # Degenerate case: the continuation ran no new FT_SCREENING call
        # this session (e.g. a pure report-regeneration pass over an
        # empty `retrieved_ids`) -- nothing of its own to override onto,
        # so the cumulative value is entirely the parent's own.
        return parent_snapshot.ft_native_provenance
    ft_native_provenance = dict(base_snapshot.ft_native_provenance)
    parent_ft_native = parent_snapshot.ft_native_provenance
    parent_confirmed = set(parent_ft_native["confirmed_record_ids"]) if parent_ft_native is not None else set()
    ft_native_provenance["confirmed_record_ids"] = sorted(
        set(ft_native_provenance["confirmed_record_ids"]) | parent_confirmed)
    return ft_native_provenance


def build_cumulative_reporting_snapshot(
    *, parent_bundle: Bundle, continuation_state: ReviewState, cfg: ReviewConfig, review_run_id: str,
    preserved_decision_citations: List[Dict[str, Any]],
) -> ReportingSourceSnapshot:
    parent_snapshot = from_bundle(parent_bundle)
    # `skip_search_strategy_cross_check=True`: a continuation never runs
    # `search_strategy_builder_node`, so the unconditional `seed_query ==
    # cfg.search.query` / producing-event-ids checks that block would
    # otherwise hard-fail on every continuation.
    base_snapshot = capture_reporting_snapshot(
        continuation_state, cfg, review_run_id, skip_search_strategy_cross_check=True)

    ledger_records = _union_ledger_records(
        parent_snapshot=parent_snapshot, base_snapshot=base_snapshot,
        preserved_decision_citations=preserved_decision_citations)

    events = list(base_snapshot.events) + [
        vf for vf in parent_snapshot.events if vf.model.stage in _PARENT_ONLY_EVENT_STAGES
    ]

    ft_native_provenance = _union_ft_native_provenance(
        parent_snapshot=parent_snapshot, base_snapshot=base_snapshot)

    unioned = dataclasses.replace(
        base_snapshot,
        # dedup/title_abstract_screening never re-run -- these 4 fields
        # are the parent's own, verbatim, zero continuation contribution.
        raw_records=parent_snapshot.raw_records,
        screening_records=parent_snapshot.screening_records,
        current_screening_batch_results=parent_snapshot.current_screening_batch_results,
        search_log=parent_snapshot.search_log,
        ledger_records=ledger_records,
        ft_native_provenance=ft_native_provenance,
        events=events,
    )
    return apply_parent_oversight_union(unioned, parent_snapshot)
