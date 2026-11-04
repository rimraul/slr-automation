"""Unit A item 15 -- `slr_pipeline.continuation.retrieved_pdfs`."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from slr_pipeline.continuation.manifest import ContinuationManifest, ContinuationManifestDocument
from slr_pipeline.continuation.retrieved_pdfs import acquire_retrieved_pdfs
from slr_pipeline.state import Record
from slr_pipeline.track3_run import Track3ConfigError


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(*docs: ContinuationManifestDocument) -> ContinuationManifest:
    return ContinuationManifest(continuation_workspace="/some/workspace", documents=list(docs))


def _doc(worklist_id: str, save_as: str, data: bytes) -> ContinuationManifestDocument:
    return ContinuationManifestDocument(
        worklist_id=worklist_id, save_as=save_as, file_sha256=_sha256(data), status="parsed", identity_flag="OK")


def test_empty_input_is_a_noop(tmp_path: Path):
    out = acquire_retrieved_pdfs([], [], _manifest(), tmp_path / "pdfs", tmp_path / "pdf_dir")
    assert out == {}
    assert not (tmp_path / "pdf_dir").exists()


def test_copies_verifies_authenticity_and_returns_dest_info(tmp_path: Path):
    data = b"%PDF-1.4 retrieved pdf bytes\n"
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    (pdfs_dir / "rec1.pdf").write_bytes(data)
    doc = _doc("rec1", "rec1.pdf", data)
    record = Record(id="rec1", source="openalex", source_id="W1", doi="10.1/abc")
    pdf_dir = tmp_path / "pdf_dir"

    out = acquire_retrieved_pdfs(["rec1"], [record], _manifest(doc), pdfs_dir, pdf_dir)

    assert set(out.keys()) == {"rec1"}
    dest_path = Path(out["rec1"]["dest_path"])
    assert dest_path.is_file()
    assert dest_path.read_bytes() == data
    assert out["rec1"]["file_sha256"] == _sha256(data)


def test_idempotent_skip_when_dest_already_matches(tmp_path: Path):
    data = b"same bytes\n"
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    (pdfs_dir / "rec1.pdf").write_bytes(data)
    doc = _doc("rec1", "rec1.pdf", data)
    record = Record(id="rec1", source="openalex", source_id="W1", doi="10.1/abc")
    pdf_dir = tmp_path / "pdf_dir"
    pdf_dir.mkdir()
    from slr_pipeline.nodes.fulltext import _dest_path
    dest = _dest_path(record, pdf_dir)
    dest.write_bytes(data)
    original_mtime = dest.stat().st_mtime_ns

    acquire_retrieved_pdfs(["rec1"], [record], _manifest(doc), pdfs_dir, pdf_dir)

    assert dest.stat().st_mtime_ns == original_mtime


def test_overwrites_when_dest_is_stale(tmp_path: Path):
    data = b"correct bytes\n"
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    (pdfs_dir / "rec1.pdf").write_bytes(data)
    doc = _doc("rec1", "rec1.pdf", data)
    record = Record(id="rec1", source="openalex", source_id="W1", doi="10.1/abc")
    pdf_dir = tmp_path / "pdf_dir"
    pdf_dir.mkdir()
    from slr_pipeline.nodes.fulltext import _dest_path
    dest = _dest_path(record, pdf_dir)
    dest.write_bytes(b"stale wrong bytes\n")

    acquire_retrieved_pdfs(["rec1"], [record], _manifest(doc), pdfs_dir, pdf_dir)

    assert dest.read_bytes() == data


def test_missing_manifest_document_raises_track3_config_error(tmp_path: Path):
    record = Record(id="rec1", source="openalex", source_id="W1")
    with pytest.raises(Track3ConfigError, match="no matching worklist_id"):
        acquire_retrieved_pdfs(["rec1"], [record], _manifest(), tmp_path / "pdfs", tmp_path / "pdf_dir")


def test_missing_included_records_entry_raises_track3_config_error(tmp_path: Path):
    doc = _doc("rec1", "rec1.pdf", b"x")
    with pytest.raises(Track3ConfigError, match="no matching entry in included_records"):
        acquire_retrieved_pdfs(["rec1"], [], _manifest(doc), tmp_path / "pdfs", tmp_path / "pdf_dir")


def test_missing_source_pdf_raises_track3_config_error(tmp_path: Path):
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    doc = _doc("rec1", "rec1.pdf", b"x")
    record = Record(id="rec1", source="openalex", source_id="W1")
    with pytest.raises(Track3ConfigError, match="expected PDF not found"):
        acquire_retrieved_pdfs(["rec1"], [record], _manifest(doc), pdfs_dir, tmp_path / "pdf_dir")


def test_authenticity_mismatch_raises_track3_config_error_before_copy(tmp_path: Path):
    original_data = b"original bytes at manifest time\n"
    tampered_data = b"tampered bytes since then\n"
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    (pdfs_dir / "rec1.pdf").write_bytes(tampered_data)
    doc = _doc("rec1", "rec1.pdf", original_data)
    record = Record(id="rec1", source="openalex", source_id="W1")
    pdf_dir = tmp_path / "pdf_dir"

    with pytest.raises(Track3ConfigError, match="no longer matches the manifest's recorded"):
        acquire_retrieved_pdfs(["rec1"], [record], _manifest(doc), pdfs_dir, pdf_dir)

    assert not pdf_dir.exists()
