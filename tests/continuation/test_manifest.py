import json

import pytest

from slr_pipeline.continuation.manifest import load_continuation_manifest
from slr_pipeline.track3_run import Track3ConfigError

_VALID = {
    "continuation_workspace": "/tmp/cont",
    "documents": [
        {
            "worklist_id": "pubmed-abc123",
            "save_as": "10_1000_abc.pdf",
            "file_sha256": "a" * 64,
            "status": "parsed",
            "identity_flag": "OK",
            "expected_doi": "10.1000/abc",  # informational extra field
        }
    ],
}


def test_loads_valid_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_VALID), encoding="utf-8")
    manifest = load_continuation_manifest(path)
    assert manifest.continuation_workspace == "/tmp/cont"
    assert len(manifest.documents) == 1
    doc = manifest.documents[0]
    assert doc.worklist_id == "pubmed-abc123"
    assert doc.save_as == "10_1000_abc.pdf"
    assert doc.file_sha256 == "a" * 64
    assert doc.status == "parsed"
    assert doc.identity_flag == "OK"


def test_missing_file_raises_track3_config_error(tmp_path):
    with pytest.raises(Track3ConfigError, match="file not found"):
        load_continuation_manifest(tmp_path / "does_not_exist.json")


def test_malformed_json_raises_track3_config_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(Track3ConfigError, match="malformed JSON"):
        load_continuation_manifest(path)


def test_schema_mismatch_raises_track3_config_error(tmp_path):
    path = tmp_path / "manifest.json"
    bad = dict(_VALID)
    del bad["continuation_workspace"]
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(Track3ConfigError, match="does not match the manifest schema"):
        load_continuation_manifest(path)


def test_document_missing_required_field_raises_track3_config_error(tmp_path):
    path = tmp_path / "manifest.json"
    bad = json.loads(json.dumps(_VALID))
    del bad["documents"][0]["worklist_id"]
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(Track3ConfigError, match="does not match the manifest schema"):
        load_continuation_manifest(path)


def test_real_fixture_manifest_loads():
    """Round-trips the real, already-completed continuation fixture
    manifest -- confirms the model is not merely self-consistent with its
    own synthetic test fixture above, but genuinely compatible with the
    real on-disk artifact this plan cites throughout Unit A."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    real_manifest = repo_root / "track3_runs" / "track3_continuation_20260914T144702Z" / "manifest.json"
    if not real_manifest.is_file():
        pytest.skip("real continuation fixture manifest not present in this checkout")
    manifest = load_continuation_manifest(real_manifest)
    assert manifest.continuation_workspace.endswith("track3_continuation_20260914T144702Z")
    assert len(manifest.documents) == 10
    assert all(doc.status == "parsed" for doc in manifest.documents)
