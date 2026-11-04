"""Track 3 continuation plan (squishy-petting-starlight.md) Unit J --
`publish_continuation_bundle()`, the separate, later, out-of-process
bundle-publish step for a continuation run. `track3_continuation_run.py`
itself stops after `graph.invoke()` and never calls `finalize_bundle()`
(plan Unit A/G design) -- this module is the one place that does, for a
continuation.

Writes the two publish-time-only workspace files `finalize_bundle()`
expects to already exist whenever `parent_bundle_id` is given
(`continuation_lineage.json`, `preserved_decisions.json`), builds the
verified cumulative `ReportingSourceSnapshot` (Unit E), then delegates to
`finalize_bundle(cumulative_snapshot=..., parent_bundle_id=...)`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.cumulative_snapshot import build_cumulative_reporting_snapshot
from slr_pipeline.continuation.lineage import write_continuation_lineage
from slr_pipeline.extraction.hashing import file_sha256
from slr_pipeline.reference_bundle import Bundle, PayloadSourceInfo, _now_iso, finalize_bundle, from_bundle


def publish_continuation_bundle(
    *,
    workspace: Path,
    cfg: ReviewConfig,
    review_run_id: str,
    reference_spec_sha256: str,
    source: PayloadSourceInfo,
    bundle_parent: Path,
    parent_bundle: Bundle,
    continuation_manifest_path: Path,
    parent_workspace: Optional[Path] = None,
) -> Path:
    """`workspace` must already hold every §12 file a continuation's real
    `graph.invoke()` writes (`terminal_state.json`, ..., `run_mode.json`/
    `run_mode_statement.md`) -- this function only additionally writes
    `continuation_lineage.json`/`preserved_decisions.json` into it before
    handing off to `finalize_bundle()`. Always publishes `mode_profile=
    "track3_live"` (a continuation is never a `track2_offline` fixture
    run) -- `fixtures` is therefore never passed."""
    workspace = Path(workspace)
    terminal_state = json.loads((workspace / "terminal_state.json").read_bytes())
    preserved_decision_citations = terminal_state.get("preserved_decision_citations", [])

    cumulative_snapshot = build_cumulative_reporting_snapshot(
        parent_bundle=parent_bundle, continuation_state=terminal_state, cfg=cfg,
        review_run_id=review_run_id, preserved_decision_citations=preserved_decision_citations,
    )

    (workspace / "preserved_decisions.json").write_text(
        json.dumps(preserved_decision_citations, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    parent_snapshot = from_bundle(parent_bundle)
    write_continuation_lineage(
        path=workspace / "continuation_lineage.json",
        parent_bundle_id=parent_bundle.bundle_id,
        parent_bundle=parent_bundle,
        parent_snapshot=parent_snapshot,
        parent_workspace=parent_workspace,
        continuation_manifest_path=continuation_manifest_path,
        continuation_manifest_digest=file_sha256(continuation_manifest_path),
        created_utc=_now_iso(),
    )

    return finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=reference_spec_sha256, source=source, bundle_parent=bundle_parent,
        mode_profile="track3_live", cumulative_snapshot=cumulative_snapshot,
        parent_bundle_id=parent_bundle.bundle_id,
    )
