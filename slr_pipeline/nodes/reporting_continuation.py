"""Track 3 continuation plan (squishy-petting-starlight.md) Unit E --
`continuation_reporting_node`, the graph-facing thin wrapper.

Opens the parent bundle, builds the cumulative `ReportingSourceSnapshot`
(`continuation/cumulative_snapshot.py`), and delegates the actual report-
writing/bundle-staging work to the ordinary `reporting_node` unchanged --
this node's own job is entirely "assemble the snapshot", never
"duplicate reporting_node's own orchestration".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.cumulative_snapshot import build_cumulative_reporting_snapshot
from slr_pipeline.nodes.reporting import reporting_node
from slr_pipeline.reference_bundle import open_bundle
from slr_pipeline.run_identity import resolve_review_run_id
from slr_pipeline.state import ReviewState


def continuation_reporting_node(state: ReviewState) -> Dict[str, Any]:
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = resolve_review_run_id(state, allow_config=True)

    parent_bundle = open_bundle(Path(state["parent_bundle_payload_root"]))
    snapshot = build_cumulative_reporting_snapshot(
        parent_bundle=parent_bundle,
        continuation_state=state,
        cfg=cfg,
        review_run_id=review_run_id,
        preserved_decision_citations=state["preserved_decision_citations"],
    )
    return reporting_node(state, cumulative_snapshot=snapshot)
