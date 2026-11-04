"""Offline tests for parsing.py. The real Docling conversion call itself
(`converter.convert(...)`) is faked via a minimal fake `docling`/
`docling_core` module tree installed into `sys.modules` -- this exercises
parse_pdf()'s OWN post-processing logic (section-heading citability, empty-
result handling, chunk grouping) without needing a real PDF or a slow real
Docling parse. See test_parsing_fixture_pdf.py for the real-Docling,
real-PDF contract test.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from slr_pipeline.extraction.parsing import (
    DoclingParserOptions, DocumentParseError, _compute_tree_digest, _group_items_into_chunks, parse_pdf,
)


def test_docling_parser_options_rejects_zero_max_chunk_chars():
    with pytest.raises(ValueError, match="max_chunk_chars"):
        DoclingParserOptions(max_chunk_chars=0)


def test_docling_parser_options_rejects_negative_max_chunk_chars():
    with pytest.raises(ValueError, match="max_chunk_chars"):
        DoclingParserOptions(max_chunk_chars=-100)


def test_group_items_section_heading_becomes_its_own_dedicated_chunk():
    """`parse_pdf()` appends a heading as `(section, section, page_no,
    is_heading=True)` -- this proves `_group_items_into_chunks` gives it
    its own dedicated chunk, with its OWN page, never fused with the
    body text that follows on a later page (review finding D)."""
    items = [
        ("Methods", "Methods", 3, True),  # the heading item itself
        ("Methods", "We used a randomized design.", 3, False),
        ("Methods", "Participants were recruited from three sites.", 4, False),
    ]
    grouped = _group_items_into_chunks(items, max_chunk_chars=10_000)
    assert len(grouped) == 2
    heading_section, heading_page_min, heading_page_max, heading_text = grouped[0]
    assert heading_text == "Methods"
    assert heading_page_min == 3 and heading_page_max == 3

    body_section, body_page_min, body_page_max, body_text = grouped[1]
    assert body_text.count("Methods") == 0  # heading text not repeated inside the body chunk
    assert body_page_min == 3 and body_page_max == 4  # body's OWN page range, independent of the heading's


def test_group_items_heading_never_repeated_across_multiple_body_chunks():
    """If the section's own body text overflows the budget and splits
    into a second chunk, the heading text (which only ever appeared once
    in the item stream, as its own dedicated chunk) must not reappear in
    either body chunk."""
    items = [
        ("Methods", "Methods", 1, True),
        ("Methods", "x" * 60, 1, False),
        ("Methods", "y" * 60, 2, False),
    ]
    grouped = _group_items_into_chunks(items, max_chunk_chars=70)
    assert len(grouped) == 3  # heading chunk + 2 split body chunks
    body_text = " ".join(text for *_rest, text in grouped[1:])
    assert body_text.count("Methods") == 0


def _install_fake_docling(monkeypatch, items, *, captured_pipeline_options: dict | None = None):
    """Installs a minimal fake `docling`/`docling_core` package tree so
    `parse_pdf`'s own lazy `from docling.document_converter import ...`
    resolves to a controlled fake. `items` is a list of
    (text, label, page_no) tuples, one per already-"iterated" document
    item; `label` is `"section_header"` for a heading, `None` otherwise.
    `captured_pipeline_options`, when given, is populated in place with
    every kwarg `parse_pdf` passed to `PdfPipelineOptions(...)` -- lets a
    test assert what `artifacts_path` was actually threaded through
    without needing a real Docling installed.
    """

    class _FakeItem:
        def __init__(self, text, label, page_no):
            self.text = text
            self.label = label
            self.prov = [SimpleNamespace(page_no=page_no)] if page_no is not None else []

    class _FakeDocument:
        def iterate_items(self):
            for text, label, page_no in items:
                yield _FakeItem(text, label, page_no), 0

    class _FakeResult:
        document = _FakeDocument()

    class _FakeConverter:
        def __init__(self, *a, **k):
            pass

        def convert(self, *a, **k):
            return _FakeResult()

    document_converter_mod = ModuleType("docling.document_converter")
    document_converter_mod.DocumentConverter = _FakeConverter
    document_converter_mod.PdfFormatOption = lambda **k: None

    def _fake_pdf_pipeline_options(**k):
        if captured_pipeline_options is not None:
            captured_pipeline_options.update(k)
        return None

    pipeline_options_mod = ModuleType("docling.datamodel.pipeline_options")
    pipeline_options_mod.PdfPipelineOptions = _fake_pdf_pipeline_options

    base_models_mod = ModuleType("docling.datamodel.base_models")
    base_models_mod.InputFormat = SimpleNamespace(PDF="pdf")

    docling_core_types_doc_mod = ModuleType("docling_core.types.doc")
    docling_core_types_doc_mod.DocItemLabel = SimpleNamespace(SECTION_HEADER="section_header")

    modules = {
        "docling": ModuleType("docling"),
        "docling.document_converter": document_converter_mod,
        "docling.datamodel": ModuleType("docling.datamodel"),
        "docling.datamodel.pipeline_options": pipeline_options_mod,
        "docling.datamodel.base_models": base_models_mod,
        "docling_core": ModuleType("docling_core"),
        "docling_core.types": ModuleType("docling_core.types"),
        "docling_core.types.doc": docling_core_types_doc_mod,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0-fake")


def test_parse_pdf_empty_document_raises_document_parse_error(monkeypatch, tmp_path):
    _install_fake_docling(monkeypatch, items=[])
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"stand-in bytes -- only hashed, never actually parsed as a real PDF here")
    with pytest.raises(DocumentParseError, match="zero citable chunks"):
        parse_pdf(pdf_path)


def test_parse_pdf_blank_only_items_also_raise_document_parse_error(monkeypatch, tmp_path):
    _install_fake_docling(monkeypatch, items=[("   ", None, 1), ("", None, 2)])
    pdf_path = tmp_path / "blank.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    with pytest.raises(DocumentParseError, match="zero citable chunks"):
        parse_pdf(pdf_path)


def test_parse_pdf_section_heading_is_citable_with_page_provenance(monkeypatch, tmp_path):
    items = [
        ("Methods", "section_header", 3),
        ("We used a randomized controlled trial design.", None, 3),
    ]
    _install_fake_docling(monkeypatch, items=items)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    parsed = parse_pdf(pdf_path)
    all_text = " ".join(c.text for c in parsed.chunks)
    assert "Methods" in all_text
    assert any(c.page_min == 3 for c in parsed.chunks)


# ---------------------------------------------------------------------------
# §5.4 -- Docling identity binding (DOCLING_CACHE_DIR + artifacts_path +
# artifacts_manifest_digest assertion). The real Docling call is faked, as
# above; these tests exercise the digest-binding logic that runs BEFORE
# Docling is even imported (or, on the success path, that it is threaded
# through to the faked `PdfPipelineOptions(...)` call).
# ---------------------------------------------------------------------------

def _basic_items():
    return [("We used a randomized controlled trial design.", None, 1)]


def _write_fake_model_tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "model.bin").write_bytes(b"weights-a")
    (root / "config.json").write_text("{}", encoding="utf-8")
    # bookkeeping that must be excluded from the digest, same as
    # FT_SCREENING's own compute_tree_digest.
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "lockfile").write_bytes(b"transient")


def test_parse_pdf_no_expected_digest_never_touches_docling_cache_dir_env(monkeypatch, tmp_path):
    """§5.4: `None` (the default) is production-unchanged -- no
    `DOCLING_CACHE_DIR` read, no digest assertion, `artifacts_path=None`
    passed straight through to Docling, and the returned document's own
    digest field stays `None`."""
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    captured: dict = {}
    _install_fake_docling(monkeypatch, items=_basic_items(), captured_pipeline_options=captured)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    parsed = parse_pdf(pdf_path)
    assert parsed.artifacts_manifest_digest is None
    assert captured["artifacts_path"] is None


def test_parse_pdf_expected_digest_requires_docling_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    _install_fake_docling(monkeypatch, items=_basic_items())
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    with pytest.raises(DocumentParseError, match="DOCLING_CACHE_DIR is not set"):
        parse_pdf(pdf_path, expected_artifacts_manifest_digest="a" * 64)


def test_parse_pdf_expected_digest_requires_models_dir_to_exist(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()  # deliberately no "models" subdirectory
    monkeypatch.setenv("DOCLING_CACHE_DIR", str(cache_dir))
    _install_fake_docling(monkeypatch, items=_basic_items())
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    with pytest.raises(DocumentParseError, match="not found"):
        parse_pdf(pdf_path, expected_artifacts_manifest_digest="a" * 64)


def test_parse_pdf_expected_digest_mismatch_raises(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    models_dir = cache_dir / "models"
    _write_fake_model_tree(models_dir)
    monkeypatch.setenv("DOCLING_CACHE_DIR", str(cache_dir))
    _install_fake_docling(monkeypatch, items=_basic_items())
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    with pytest.raises(DocumentParseError, match="do not match the expected tree digest"):
        parse_pdf(pdf_path, expected_artifacts_manifest_digest="0" * 64)


def test_parse_pdf_expected_digest_match_binds_artifacts_path_and_records_digest(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    models_dir = cache_dir / "models"
    _write_fake_model_tree(models_dir)
    expected_digest = _compute_tree_digest(models_dir)
    monkeypatch.setenv("DOCLING_CACHE_DIR", str(cache_dir))
    captured: dict = {}
    _install_fake_docling(monkeypatch, items=_basic_items(), captured_pipeline_options=captured)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"stand-in bytes")
    parsed = parse_pdf(pdf_path, expected_artifacts_manifest_digest=expected_digest)
    assert parsed.artifacts_manifest_digest == expected_digest
    assert captured["artifacts_path"] == models_dir


def test_compute_tree_digest_excludes_cache_and_locks_bookkeeping(tmp_path):
    """Same exclusion rule as FT_SCREENING's own `compute_tree_digest` --
    a change under `.cache`/`.locks` must not perturb the digest."""
    models_dir = tmp_path / "models"
    _write_fake_model_tree(models_dir)
    before = _compute_tree_digest(models_dir)
    (models_dir / ".cache" / "huggingface" / "lockfile").write_bytes(b"different transient bytes")
    after = _compute_tree_digest(models_dir)
    assert before == after


def test_compute_tree_digest_changes_when_a_real_model_file_changes(tmp_path):
    models_dir = tmp_path / "models"
    _write_fake_model_tree(models_dir)
    before = _compute_tree_digest(models_dir)
    (models_dir / "sub" / "model.bin").write_bytes(b"different weights")
    after = _compute_tree_digest(models_dir)
    assert before != after


def test_compute_tree_digest_matches_ft_screening_real_implementation(tmp_path):
    """The reimplementation (mandatory -- this module's own
    "Do not import FT_SCREENING internals" boundary, see module
    docstring) must be algorithmically IDENTICAL to
    `ft_screening.docling_artifacts.compute_tree_digest`: the digest
    asserted here is compared against `assets/models/DOCLING_ASSETS.
    lock`'s own `tree_digest`, itself produced by the real function --
    a divergent reimplementation would make every comparison spuriously
    fail (or, worse, spuriously pass)."""
    ft_screening_docling_artifacts = pytest.importorskip("ft_screening.docling_artifacts")
    models_dir = tmp_path / "models"
    _write_fake_model_tree(models_dir)
    assert _compute_tree_digest(models_dir) == ft_screening_docling_artifacts.compute_tree_digest(models_dir)
