"""Unit A item 12 -- `slr_pipeline.continuation.preserved_pdfs`."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from slr_pipeline.continuation.preserved_pdfs import acquire_preserved_included_pdfs
from slr_pipeline.state import Record
from slr_pipeline.track3_run import Track3ConfigError

from .conftest import minimal_reporting_source_snapshot


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_workspace_pdf(workspace: Path, filename: str, data: bytes) -> Path:
    path = workspace / "pdfs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_empty_input_is_a_noop(tmp_path: Path):
    snapshot = minimal_reporting_source_snapshot(full_text_records=[])
    out = acquire_preserved_included_pdfs([], snapshot, tmp_path / "workspace", tmp_path / "pdf_dir")
    assert out == []
    assert not (tmp_path / "pdf_dir").exists()


def test_copies_and_verifies_and_updates_full_text_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    data = b"%PDF-1.4 fake pdf bytes\n"
    _make_workspace_pdf(workspace, "rec1.pdf", data)
    record = Record(id="rec1", source="openalex", source_id="W1", full_text_path="/anywhere/rec1.pdf")
    snapshot = minimal_reporting_source_snapshot(full_text_records=[record])
    pdf_dir = tmp_path / "pdf_dir"

    out = acquire_preserved_included_pdfs(["rec1"], snapshot, workspace, pdf_dir)

    assert len(out) == 1
    updated = out[0]
    assert updated.id == "rec1"
    dest = pdf_dir / "rec1.pdf"
    assert dest.is_file()
    assert dest.read_bytes() == data
    assert updated.full_text_path == str(dest)


def test_idempotent_skip_when_dest_already_matches(tmp_path: Path):
    workspace = tmp_path / "workspace"
    data = b"same bytes\n"
    _make_workspace_pdf(workspace, "rec1.pdf", data)
    record = Record(id="rec1", source="openalex", source_id="W1", full_text_path="/anywhere/rec1.pdf")
    snapshot = minimal_reporting_source_snapshot(full_text_records=[record])
    pdf_dir = tmp_path / "pdf_dir"
    pdf_dir.mkdir()
    dest = pdf_dir / "rec1.pdf"
    dest.write_bytes(data)
    original_mtime = dest.stat().st_mtime_ns

    acquire_preserved_included_pdfs(["rec1"], snapshot, workspace, pdf_dir)

    assert dest.stat().st_mtime_ns == original_mtime


def test_overwrites_when_dest_is_stale(tmp_path: Path):
    workspace = tmp_path / "workspace"
    data = b"correct bytes\n"
    _make_workspace_pdf(workspace, "rec1.pdf", data)
    record = Record(id="rec1", source="openalex", source_id="W1", full_text_path="/anywhere/rec1.pdf")
    snapshot = minimal_reporting_source_snapshot(full_text_records=[record])
    pdf_dir = tmp_path / "pdf_dir"
    pdf_dir.mkdir()
    (pdf_dir / "rec1.pdf").write_bytes(b"stale, wrong bytes\n")

    acquire_preserved_included_pdfs(["rec1"], snapshot, workspace, pdf_dir)

    assert (pdf_dir / "rec1.pdf").read_bytes() == data


def test_missing_source_pdf_raises_track3_config_error(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    record = Record(id="rec1", source="openalex", source_id="W1", full_text_path="/anywhere/rec1.pdf")
    snapshot = minimal_reporting_source_snapshot(full_text_records=[record])

    with pytest.raises(Track3ConfigError, match="rec1"):
        acquire_preserved_included_pdfs(["rec1"], snapshot, workspace, tmp_path / "pdf_dir")


def test_missing_full_text_records_entry_raises_track3_config_error(tmp_path: Path):
    snapshot = minimal_reporting_source_snapshot(full_text_records=[])

    with pytest.raises(Track3ConfigError, match="rec-missing"):
        acquire_preserved_included_pdfs(["rec-missing"], snapshot, tmp_path / "workspace", tmp_path / "pdf_dir")


def test_record_without_full_text_path_raises_track3_config_error(tmp_path: Path):
    record = Record(id="rec1", source="openalex", source_id="W1", full_text_path=None)
    snapshot = minimal_reporting_source_snapshot(full_text_records=[record])

    with pytest.raises(Track3ConfigError, match="no full_text_path"):
        acquire_preserved_included_pdfs(["rec1"], snapshot, tmp_path / "workspace", tmp_path / "pdf_dir")
