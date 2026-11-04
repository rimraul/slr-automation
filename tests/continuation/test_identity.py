"""Unit A items 1-2 -- `slr_pipeline.continuation.identity`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from slr_pipeline.continuation.identity import (
    check_parent_workspace_consistency,
    resolve_parent_bundle,
)
from slr_pipeline.reference_bundle import ReleaseNotApproved, ReleaseStructureInvalid
from slr_pipeline.track3_run import Track3ConfigError

from .conftest import write_disk_bundle


# ===========================================================================
# resolve_parent_bundle()
# ===========================================================================

def test_resolve_parent_bundle_success(tmp_path: Path):
    bundles_root = tmp_path / "bundles"
    bundle_id = write_disk_bundle(bundles_root)

    bundle = resolve_parent_bundle(bundle_id, bundles_root)

    assert bundle.bundle_id == bundle_id
    assert bundle.run_registry_key == bundle_id
    assert bundle.run_registry_entry is not None
    assert bundle.run_registry_entry.decision == "APPROVED"


def test_resolve_parent_bundle_missing_registry_file(tmp_path: Path):
    bundles_root = tmp_path / "bundles"
    bundles_root.mkdir()

    with pytest.raises(Track3ConfigError, match="no run_registry.json"):
        resolve_parent_bundle("t2-doesnotexist00000000", bundles_root)


def test_resolve_parent_bundle_missing_registry_entry(tmp_path: Path):
    bundles_root = tmp_path / "bundles"
    write_disk_bundle(bundles_root)

    with pytest.raises(Track3ConfigError, match="no such entry"):
        resolve_parent_bundle("t2-000000000000000000000000", bundles_root)


def test_resolve_parent_bundle_rejected_decision_raises_release_not_approved(tmp_path: Path):
    bundles_root = tmp_path / "bundles"
    bundle_id = write_disk_bundle(bundles_root, decision="REJECTED")

    with pytest.raises(ReleaseNotApproved):
        resolve_parent_bundle(bundle_id, bundles_root)


def test_resolve_parent_bundle_tampered_rejected_bundle_raises_structure_invalid_not_not_approved(tmp_path: Path):
    """A REJECTED bundle whose sidecars are ALSO internally inconsistent
    must fail on structure (`ReleaseStructureInvalid`), not be reported as
    merely not-approved -- `verify_release_structure()`'s own documented
    guarantee (`reference_bundle.py:600-603`)."""
    bundles_root = tmp_path / "bundles"
    bundle_id = write_disk_bundle(bundles_root, decision="REJECTED")

    acceptance_report_path = bundles_root / bundle_id / "acceptance_report.json"
    data = json.loads(acceptance_report_path.read_text(encoding="utf-8"))
    data["checks"][0]["status"] = "fail"
    acceptance_report_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReleaseStructureInvalid):
        resolve_parent_bundle(bundle_id, bundles_root)


def test_resolve_parent_bundle_registry_run_registry_key_mismatch_raises_structure_invalid(tmp_path: Path):
    bundles_root = tmp_path / "bundles"
    bundle_id = write_disk_bundle(bundles_root)

    registry_path = bundles_root / "run_registry.json"
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = data["runs"].pop(bundle_id)
    other_key = "t2-" + ("0" * 24)
    data["runs"][other_key] = entry
    registry_path.write_text(json.dumps(data), encoding="utf-8")
    shutil.copytree(bundles_root / bundle_id, bundles_root / other_key)

    with pytest.raises(Track3ConfigError, match="no such entry"):
        resolve_parent_bundle(bundle_id, bundles_root)

    with pytest.raises(ReleaseStructureInvalid):
        resolve_parent_bundle(other_key, bundles_root)


# ===========================================================================
# check_parent_workspace_consistency()
# ===========================================================================

def test_check_parent_workspace_consistency_none_is_noop():
    check_parent_workspace_consistency(None)


def test_check_parent_workspace_consistency_missing_directory(tmp_path: Path):
    with pytest.raises(Track3ConfigError, match="not a directory"):
        check_parent_workspace_consistency(tmp_path / "does-not-exist")


@pytest.mark.parametrize("missing", ["terminal_state.json", "thread_id.txt", "graph_checkpoints.sqlite"])
def test_check_parent_workspace_consistency_missing_file(tmp_path: Path, missing: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("terminal_state.json", "thread_id.txt", "graph_checkpoints.sqlite"):
        if name != missing:
            (workspace / name).write_text("x", encoding="utf-8")

    with pytest.raises(Track3ConfigError, match=f"missing expected file '{missing}'"):
        check_parent_workspace_consistency(workspace)


def test_check_parent_workspace_consistency_all_present(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("terminal_state.json", "thread_id.txt", "graph_checkpoints.sqlite"):
        (workspace / name).write_text("x", encoding="utf-8")

    check_parent_workspace_consistency(workspace)
