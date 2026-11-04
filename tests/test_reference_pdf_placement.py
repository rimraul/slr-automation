"""Unit tests for `slr_pipeline.reference_pdf_placement` -- the shared §6.3
hash-verified PDF pre-positioning used by both `scripts/author_reference_
cassette.py` and `slr_pipeline.reference_run_child` (`track2_pipeline_
readiness_plan_2026-08-30.md` §6.3, step 13)."""
from __future__ import annotations

import hashlib

import pytest

from slr_pipeline.reference_pdf_placement import (
    PDF_PLACEMENT,
    REFERENCE_EXTRACTION_WORKSPACE_PARTS,
    place_pdfs,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_source_pdfs(source_dir, contents: dict) -> dict:
    """`contents` maps record_id -> bytes. Returns the oracle dict `place_pdfs` expects."""
    source_dir.mkdir(parents=True, exist_ok=True)
    oracle = {"pdf_sha256": {}}
    for record_id, data in contents.items():
        (source_dir / f"{record_id}.pdf").write_bytes(data)
        oracle["pdf_sha256"][record_id] = _sha256(data)
    return oracle


def test_placement_table_has_six_rows_all_nk_pattern():
    assert len(PDF_PLACEMENT) == 6
    for record_id, source, source_id in PDF_PLACEMENT:
        assert source_id == f"NK-{record_id}"
        assert source.startswith("fx_")


def test_reference_extraction_workspace_parts():
    assert REFERENCE_EXTRACTION_WORKSPACE_PARTS == (
        "tests", "reference_run", ".cassette_generation_workspace",
    )


def test_place_pdfs_copies_all_rows_with_correct_dest_names(tmp_path):
    source_dir = tmp_path / "source"
    contents = {record_id: f"pdf-bytes-{record_id}".encode() for record_id, _, _ in PDF_PLACEMENT}
    oracle = _write_source_pdfs(source_dir, contents)

    pdf_dir = tmp_path / "dest" / "pdfs"
    place_pdfs(pdf_dir, oracle, source_dir)

    for record_id, source, source_id in PDF_PLACEMENT:
        dest = pdf_dir / f"{source}_{source_id}.pdf"
        assert dest.exists()
        assert dest.read_bytes() == contents[record_id]


def test_place_pdfs_creates_dest_dir(tmp_path):
    source_dir = tmp_path / "source"
    contents = {record_id: f"pdf-bytes-{record_id}".encode() for record_id, _, _ in PDF_PLACEMENT}
    oracle = _write_source_pdfs(source_dir, contents)

    pdf_dir = tmp_path / "brand-new" / "nested" / "pdfs"
    assert not pdf_dir.exists()
    place_pdfs(pdf_dir, oracle, source_dir)
    assert pdf_dir.is_dir()


def test_place_pdfs_raises_on_source_hash_mismatch(tmp_path):
    source_dir = tmp_path / "source"
    contents = {record_id: f"pdf-bytes-{record_id}".encode() for record_id, _, _ in PDF_PLACEMENT}
    oracle = _write_source_pdfs(source_dir, contents)

    # Corrupt the first row's source file after the oracle was computed.
    first_record_id = PDF_PLACEMENT[0][0]
    (source_dir / f"{first_record_id}.pdf").write_bytes(b"corrupted")

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        place_pdfs(tmp_path / "dest" / "pdfs", oracle, source_dir)


def test_place_pdfs_raises_on_wrong_oracle_value(tmp_path):
    source_dir = tmp_path / "source"
    contents = {record_id: f"pdf-bytes-{record_id}".encode() for record_id, _, _ in PDF_PLACEMENT}
    oracle = _write_source_pdfs(source_dir, contents)

    first_record_id = PDF_PLACEMENT[0][0]
    oracle["pdf_sha256"][first_record_id] = "0" * 64  # wrong digest, unrelated to the real bytes

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        place_pdfs(tmp_path / "dest" / "pdfs", oracle, source_dir)


def test_place_pdfs_against_real_fixture_corpus():
    """Same source directory and oracle the real driver/child use -- proves
    the table matches what is actually committed under
    `tests/reference_run/fixtures/corpus/`, not just synthetic bytes."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    corpus_dir = repo_root / "tests" / "reference_run" / "fixtures" / "corpus"
    oracle = json.loads((corpus_dir / "CORPUS_ORACLE.json").read_text(encoding="utf-8"))

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pdf_dir = Path(tmp) / "pdfs"
        place_pdfs(pdf_dir, oracle, corpus_dir / "pdfs")
        for record_id, source, source_id in PDF_PLACEMENT:
            assert (pdf_dir / f"{source}_{source_id}.pdf").exists()
