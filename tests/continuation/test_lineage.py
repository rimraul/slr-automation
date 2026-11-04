"""Unit A item 5 -- `slr_pipeline.continuation.lineage`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slr_pipeline.continuation.identity import resolve_parent_bundle
from slr_pipeline.continuation.lineage import write_continuation_lineage

from .conftest import minimal_reporting_source_snapshot, write_disk_bundle


def _resolved(tmp_path: Path, **kw):
    bundles_root = tmp_path / "bundles"
    bundle_id = write_disk_bundle(bundles_root, **kw)
    return bundle_id, resolve_parent_bundle(bundle_id, bundles_root)


def test_write_continuation_lineage_copies_registry_hashes_verbatim(tmp_path: Path):
    bundle_id, bundle = _resolved(tmp_path)
    snapshot = minimal_reporting_source_snapshot(
        review_run_id="run-1", ft_native_provenance={"ft_screening_run_id": "ftrun-42"})

    out_path = tmp_path / "continuation_lineage.json"
    result = write_continuation_lineage(
        path=out_path, parent_bundle_id=bundle_id, parent_bundle=bundle, parent_snapshot=snapshot,
        parent_workspace=tmp_path / "workspace", continuation_manifest_path=tmp_path / "manifest.json",
        continuation_manifest_digest="d" * 64, created_utc="2026-09-15T00:00:00Z")

    assert result["parent_bundle_id"] == bundle_id
    assert result["parent_payload_hash"] == bundle.run_registry_entry.payload_hash
    assert result["parent_release_hash"] == bundle.run_registry_entry.release_hash
    assert result["parent_review_run_id"] == "run-1"
    assert result["parent_ft_screening_run_id"] == "ftrun-42"
    assert result["parent_workspace"] == str(tmp_path / "workspace")
    assert result["continuation_manifest_path"] == str(tmp_path / "manifest.json")
    assert result["continuation_manifest_digest"] == "d" * 64
    assert result["created_utc"] == "2026-09-15T00:00:00Z"

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result


def test_write_continuation_lineage_handles_missing_ft_native_provenance(tmp_path: Path):
    bundle_id, bundle = _resolved(tmp_path)
    snapshot = minimal_reporting_source_snapshot(review_run_id="run-1", ft_native_provenance=None)

    result = write_continuation_lineage(
        path=tmp_path / "lineage.json", parent_bundle_id=bundle_id, parent_bundle=bundle,
        parent_snapshot=snapshot, parent_workspace=None, continuation_manifest_path=tmp_path / "manifest.json",
        continuation_manifest_digest="e" * 64, created_utc="2026-09-15T00:00:00Z")

    assert result["parent_ft_screening_run_id"] is None
    assert result["parent_workspace"] is None


def test_write_continuation_lineage_handles_ft_native_provenance_without_run_id_key(tmp_path: Path):
    bundle_id, bundle = _resolved(tmp_path)
    snapshot = minimal_reporting_source_snapshot(review_run_id="run-1", ft_native_provenance={"other": "value"})

    result = write_continuation_lineage(
        path=tmp_path / "lineage.json", parent_bundle_id=bundle_id, parent_bundle=bundle,
        parent_snapshot=snapshot, parent_workspace=None, continuation_manifest_path=tmp_path / "manifest.json",
        continuation_manifest_digest="f" * 64, created_utc="2026-09-15T00:00:00Z")

    assert result["parent_ft_screening_run_id"] is None


def test_write_continuation_lineage_requires_run_registry_entry(tmp_path: Path):
    import dataclasses

    bundle_id, bundle = _resolved(tmp_path)
    bundle_without_entry = dataclasses.replace(bundle, run_registry_entry=None)
    snapshot = minimal_reporting_source_snapshot(review_run_id="run-1")

    with pytest.raises(ValueError, match="run_registry_entry is required"):
        write_continuation_lineage(
            path=tmp_path / "lineage.json", parent_bundle_id=bundle_id, parent_bundle=bundle_without_entry,
            parent_snapshot=snapshot, parent_workspace=None, continuation_manifest_path=tmp_path / "manifest.json",
            continuation_manifest_digest="0" * 64, created_utc="2026-09-15T00:00:00Z")
