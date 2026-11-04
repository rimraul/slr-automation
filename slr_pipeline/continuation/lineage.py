"""Unit A item 5 -- `continuation_lineage.json`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from slr_pipeline.nodes.reporting import ReportingSourceSnapshot
from slr_pipeline.reference_bundle import Bundle


def write_continuation_lineage(
    *,
    path: Path,
    parent_bundle_id: str,
    parent_bundle: Bundle,
    parent_snapshot: ReportingSourceSnapshot,
    parent_workspace: Optional[Path],
    continuation_manifest_path: Path,
    continuation_manifest_digest: str,
    created_utc: str,
) -> Dict[str, Any]:
    """Parent `payload_hash`/`release_hash` are copied VERBATIM from the
    already-verified registry entry, not recomputed -- recomputation is
    `open_bundle()`/`verify_release_structure()`'s own job, invoked
    separately (`identity.py::resolve_parent_bundle()`) and must
    independently succeed before this is ever called."""
    entry = parent_bundle.run_registry_entry
    if entry is None:
        raise ValueError(
            "parent_bundle.run_registry_entry is required -- pass the Bundle returned by "
            "identity.resolve_parent_bundle(), never a bare open_bundle() result")

    ft_screening_run_id = None
    if parent_snapshot.ft_native_provenance is not None:
        ft_screening_run_id = parent_snapshot.ft_native_provenance.get("ft_screening_run_id")

    lineage = {
        "parent_bundle_id": parent_bundle_id,
        "parent_payload_hash": entry.payload_hash,
        "parent_release_hash": entry.release_hash,
        "parent_review_run_id": parent_snapshot.review_run_id,
        "parent_ft_screening_run_id": ft_screening_run_id,
        "parent_workspace": str(parent_workspace) if parent_workspace is not None else None,
        "continuation_manifest_path": str(continuation_manifest_path),
        "continuation_manifest_digest": continuation_manifest_digest,
        "created_utc": created_utc,
    }
    Path(path).write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lineage
