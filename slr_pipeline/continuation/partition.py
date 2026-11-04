"""Unit B -- data-derived partition (codex point 1: no hardcoded IDs/counts).

`derive_continuation_partition()` takes NO knowledge of which run it is
being called for -- the real parent run's 14/3/10/1 shape and the NNA
record's specific ID never appear in this function or its production
callers. It reads purely from `included_records` (this continuation's own
included-record list), `parent_snapshot` (the parent run's
`ReportingSourceSnapshot`, Unit A item 1's `resolve_parent_bundle()` +
`from_bundle(open_bundle(...))`), and `continuation_manifest` (Unit A item
15's typed manifest, `manifest.py::ContinuationManifest`)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict

from slr_pipeline.continuation.manifest import ContinuationManifest
from slr_pipeline.nodes.reporting import ReportingSourceSnapshot
from slr_pipeline.selection_ledger import _tails_by_key, is_fulltext_exclusion_decision
from slr_pipeline.state import Record
from slr_pipeline.track3_run import Track3ConfigError

_STAGE = "full_text_eligibility"


class ContinuationPartition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preserved_included_ids: List[str]
    preserved_excluded_ids: List[str]
    retrieved_ids: List[str]
    not_retrieved_ids: List[str]

    @property
    def preserved_ids(self) -> List[str]:
        """Union of the include/exclude split, for callers that don't need
        it (Unit C's PDF-acquisition dispatch needs the split; other
        callers, e.g. exhaustiveness checks, only need the union)."""
        return self.preserved_included_ids + self.preserved_excluded_ids


def derive_continuation_partition(
    included_records: List[Record],
    parent_snapshot: ReportingSourceSnapshot,
    continuation_manifest: ContinuationManifest,
) -> ContinuationPartition:
    ledger_records = [vf.model for vf in parent_snapshot.ledger_records]
    ft_tails = _tails_by_key([r for r in ledger_records if r.stage == _STAGE])

    preserved_included_ids: List[str] = []
    preserved_excluded_ids: List[str] = []
    for record in included_records:
        tail = ft_tails.get((record.id, _STAGE))
        if tail is None:
            continue
        if is_fulltext_exclusion_decision(tail.decision):
            preserved_excluded_ids.append(record.id)
        else:
            preserved_included_ids.append(record.id)
    preserved_ids = set(preserved_included_ids) | set(preserved_excluded_ids)

    retrieved_ids: List[str] = []
    for doc in continuation_manifest.documents:
        if doc.status != "parsed" or doc.identity_flag != "OK":
            continue
        if doc.worklist_id in preserved_ids:
            raise Track3ConfigError(
                f"continuation manifest worklist_id {doc.worklist_id!r} already has a preserved "
                "full_text_eligibility decision in parent_snapshot -- a record cannot be both "
                "already-decided and newly-supplied")
        retrieved_ids.append(doc.worklist_id)

    included_ids = {r.id for r in included_records}
    not_retrieved_ids = sorted(included_ids - preserved_ids - set(retrieved_ids))

    all_buckets = preserved_included_ids + preserved_excluded_ids + retrieved_ids + not_retrieved_ids
    if len(all_buckets) != len(set(all_buckets)) or set(all_buckets) != included_ids:
        raise Track3ConfigError(
            "derive_continuation_partition: exhaustive partition check failed -- "
            f"included_records ids={sorted(included_ids)!r} do not exactly match the union of "
            f"preserved_included_ids={sorted(preserved_included_ids)!r}, "
            f"preserved_excluded_ids={sorted(preserved_excluded_ids)!r}, "
            f"retrieved_ids={sorted(retrieved_ids)!r}, not_retrieved_ids={sorted(not_retrieved_ids)!r} "
            "(missing id, duplicate id across buckets, or a manifest worklist_id not in included_records)")

    return ContinuationPartition(
        preserved_included_ids=preserved_included_ids,
        preserved_excluded_ids=preserved_excluded_ids,
        retrieved_ids=retrieved_ids,
        not_retrieved_ids=not_retrieved_ids,
    )
