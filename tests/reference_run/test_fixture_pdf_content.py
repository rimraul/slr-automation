"""Track 2 Step 12A -- regression tests for `scripts/author_fixture_corpus.py`'s
PDF generation. Codex review (2026-09-03) findings 1 and 2: the previous
`_build_pdf` placed every line at a fixed (x, y) coordinate with no wrapping
and used PyMuPDF's base14 "helv" font, which together (a) let long
evidence-field lines run off the right edge of the page and (b) silently
mangled the mandated banner's em dash (U+2014) into a replacement-character
glyph on extraction. Both defects were confirmed present in all 6 generated
PDFs by direct visual inspection.

These tests build PDFs the same way `scripts/author_fixture_corpus.py`
does (via its own `_build_pdf`/`_pdf_lines_for`) and would FAIL against the
pre-fix code: the em-dash test fails outright (extracts as "�" instead
of "—"), and the clipping test fails because the old fixed-position
`insert_text` calls place glyphs past the page's right margin for every one
of the 6 records (measured overflow 0.5-5.2pt past a 595pt-wide page,
2026-09-03).
"""
from __future__ import annotations

import re

import fitz
import pytest

import scripts.author_fixture_corpus as afc
from slr_pipeline import reference_fixture_answer_key as ak

# The right edge every generated text span must stay within (matches
# `_build_pdf`'s own heading/body/banner rects, all of which end at x=540
# on a page whose default width is 595pt -- a 55pt right margin).
_MAX_SPAN_X1 = 540.0


def _extracted_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _max_span_x1(pdf_bytes: bytes) -> float:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        maxx1 = 0.0
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        maxx1 = max(maxx1, span["bbox"][2])
        return maxx1
    finally:
        doc.close()


@pytest.mark.parametrize("record_id", ak.FULL_TEXT_SOUGHT)
def test_banner_em_dash_round_trips_exactly(record_id: str):
    """The exact banner string, INCLUDING its em dash, must be extractable
    verbatim -- not a substring that happens to exclude the em dash, and
    not a normalised/ASCII-folded comparison that would hide corruption."""
    lines = afc._pdf_lines_for(record_id)
    data = afc._build_pdf(record_id, lines)
    text = _extracted_text(data)
    assert afc.SYNTHETIC_BANNER in text
    assert "—" in text  # the literal em dash, not "�" or "-"


@pytest.mark.parametrize("record_id", ak.FULL_TEXT_SOUGHT)
def test_no_text_clipped_off_the_page(record_id: str):
    """No rendered glyph may extend past the page's right margin. The old
    fixed-position `insert_text` loop had no wrapping and no width check --
    every one of the 6 records overflowed the page by 0.5-5.2pt (measured
    2026-09-03 against the pre-fix code)."""
    lines = afc._pdf_lines_for(record_id)
    data = afc._build_pdf(record_id, lines)
    assert _max_span_x1(data) <= _MAX_SPAN_X1


@pytest.mark.parametrize("record_id", ak.FULL_TEXT_SOUGHT)
def test_full_expected_content_is_present_unclipped(record_id: str):
    """Every non-blank line the answer key contributes must be present in
    the extracted text (whitespace-normalised -- `insert_textbox` reflows
    original line breaks, and PyMuPDF's embedded-font path substitutes
    NBSP for the ASCII space; both are cosmetic, not content loss)."""
    lines = afc._pdf_lines_for(record_id)
    data = afc._build_pdf(record_id, lines)
    text = _extracted_text(data)
    normalized = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    for line in lines:
        if not line.strip():
            continue
        expected = re.sub(r"\s+", " ", line).strip()
        assert expected in normalized, f"{record_id}: clipped or missing: {expected!r}"


def test_build_pdf_raises_on_genuine_overflow():
    """The overflow guard itself must fire: a body far too long for its box
    must raise, not silently clip. Proves `_build_pdf`'s `insert_textbox`
    return-value check is live, not dead code."""
    huge_lines = ["This line is repeated far past what the body text box can hold. " * 40] * 40
    with pytest.raises(ValueError, match="overflowed its text box"):
        afc._build_pdf(ak.FULL_TEXT_SOUGHT[0], huge_lines)


def test_write_pdfs_is_byte_deterministic():
    """Two consecutive `_build_pdf` calls over identical input must be
    byte-identical -- otherwise `--check`'s byte-comparison contract and
    CORPUS_ORACLE.json's pdf_sha256 pin cannot hold across a re-`--write`.
    MuPDF's `subset_fonts()` assigns a random subset-name tag and stamps a
    fresh random /ID trailer pair on every save unless both are
    normalised (2026-09-03 finding, fixed via `no_new_id=1` plus a
    same-length tag substitution)."""
    record_id = ak.FULL_TEXT_SOUGHT[0]
    lines = afc._pdf_lines_for(record_id)
    first = afc._build_pdf(record_id, lines)
    second = afc._build_pdf(record_id, lines)
    assert first == second


def test_on_disk_pdfs_match_check_all():
    """The corpus actually committed under fixtures/corpus/ must currently
    pass the script's own --check (regenerating this test's fixtures is
    covered by scripts/author_fixture_corpus.py --write, exercised
    separately; this only asserts the checked-in state is self-consistent
    right now)."""
    afc.check_all()  # raises on any mismatch; no assertion needed
