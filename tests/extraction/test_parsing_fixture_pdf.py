"""Real-Docling, real-PDF contract tests for parsing.py (review finding 7:
"the real Docling parser boundary is untested"). Two tiers:

  - `test_parse_pdf_fixture_...`: a SMALL, deterministically-generated PDF
    (via PyMuPDF/`fitz`, test-only -- never a production extraction
    dependency, see `parsing.py`'s own module docstring) with a
    distinctively-laid-out heading, a late-page decisive statement, and a
    simple table -- fast enough to run in the default suite.
  - `test_parse_pdf_real_publication_slow_integration` (SKIPPED by
    default): the actual `2025.STMetodologinepanorama.pdf` used
    throughout this chapter, gated behind an explicit opt-in env var
    since a real ~224-page Docling parse takes on the order of a minute.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is test-only, used here purely to generate a fixture PDF")

from slr_pipeline.extraction.parsing import parse_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_fixture_pdf(path: Path) -> None:
    """A small, deterministic, multi-page PDF built directly with
    PyMuPDF -- large bold text on its own line for the heading (Docling's
    heading detector is primarily font-size/style driven, so this layout
    is deliberately distinctive rather than same-size body text), a
    decisive statement on a LATE page (page 4), and a simple table-like
    grid on page 3."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 100), "Methods", fontsize=20, fontname="helv", render_mode=0)
    page1.insert_text((72, 140), "This study used a randomized controlled trial design.", fontsize=11)
    page1.insert_text((72, 160), "Participants were recruited from three clinical sites.", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 100), "Results", fontsize=20, fontname="helv", render_mode=0)
    page2.insert_text((72, 140), "Preliminary findings are summarized in the table below.", fontsize=11)

    page3 = doc.new_page()
    page3.insert_text((72, 100), "Table 1: Participant characteristics", fontsize=12)
    rows = [
        ("Group", "N", "Mean age"),
        ("Treatment", "42", "51.3"),
        ("Control", "40", "50.7"),
    ]
    y = 130
    for row in rows:
        page3.insert_text((72, y), f"{row[0]:<15}{row[1]:<10}{row[2]:<10}", fontsize=11)
        y += 20

    page4 = doc.new_page()
    page4.insert_text((72, 100), "Discussion", fontsize=20, fontname="helv", render_mode=0)
    page4.insert_text((72, 140), "The final sample size was 82 participants after exclusions.", fontsize=11)

    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="module")
def fixture_pdf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("extraction_fixture") / "fixture.pdf"
    _build_fixture_pdf(path)
    return path


@pytest.fixture(scope="module")
def parsed_fixture(fixture_pdf_path: Path):
    return parse_pdf(fixture_pdf_path)


def test_parse_pdf_fixture_produces_chunks(parsed_fixture):
    assert len(parsed_fixture.chunks) > 0


def test_parse_pdf_fixture_heading_is_citable(parsed_fixture):
    all_text = " ".join(c.text for c in parsed_fixture.chunks)
    assert "Methods" in all_text


def test_parse_pdf_fixture_heading_is_its_own_chunk_not_fused_with_body(parsed_fixture):
    """Review finding D's direct regression against real Docling output:
    the "Methods" heading (page 1) must be its own dedicated chunk,
    distinct from the body chunk containing the RCT-design sentence that
    follows it -- not one fused chunk repeating/absorbing the heading."""
    heading_chunks = [c for c in parsed_fixture.chunks if c.text.strip() == "Methods"]
    assert len(heading_chunks) == 1
    assert heading_chunks[0].page_min == 1
    body_chunks = [c for c in parsed_fixture.chunks if "randomized controlled trial" in c.text]
    assert body_chunks
    assert "Methods" not in body_chunks[0].text


def test_parse_pdf_fixture_late_page_decisive_statement_has_correct_page(parsed_fixture):
    matches = [c for c in parsed_fixture.chunks if "82 participants" in c.text]
    assert matches, "decisive late-page statement not found in any parsed chunk"
    assert matches[0].page_min == 4


def test_parse_pdf_fixture_table_text_is_present(parsed_fixture):
    all_text = " ".join(c.text for c in parsed_fixture.chunks)
    assert "Treatment" in all_text
    assert "51.3" in all_text


def test_parse_pdf_fixture_page_provenance_spans_text_pages(parsed_fixture):
    """Pages 1, 2, and 4 carry plain text items and get correct page
    provenance through `parsing.py`'s own `item.prov[0].page_no` path.
    Page 3's TABLE content is a Docling `TableItem`, not a plain text
    item -- `parsing.py`'s current `iterate_items()` loop only reads
    `.text`/`.prov` off items generically, and a `TableItem` does not
    expose page provenance the same way a text item does, so its content
    is still captured (see `test_parse_pdf_fixture_table_text_is_present`)
    but is not currently guaranteed to carry page 3 specifically. This is
    a real, narrower gap (table-specific page provenance) beyond this
    package's v1 scope, not something this test papers over -- it is
    checked here only for the pages this parser DOES handle."""
    pages_seen = {c.page_min for c in parsed_fixture.chunks if c.page_min is not None}
    assert {1, 2, 4}.issubset(pages_seen)


_REAL_PDF_PATH = REPO_ROOT / "2025.STMetodologinepanorama.pdf"


@pytest.mark.skipif(
    not _REAL_PDF_PATH.exists() or os.environ.get("SLR_EXTRACTION_RUN_SLOW_PDF_TEST") != "1",
    reason="opt-in slow integration test: set SLR_EXTRACTION_RUN_SLOW_PDF_TEST=1 to run "
    "(a real ~224-page Docling parse takes roughly a minute)",
)
def test_parse_pdf_real_publication_slow_integration():
    parsed = parse_pdf(_REAL_PDF_PATH)
    assert len(parsed.chunks) > 0
    assert len({c.chunk_id for c in parsed.chunks}) == len(parsed.chunks)  # all unique
    assert [c.order_index for c in parsed.chunks] == list(range(len(parsed.chunks)))  # sequential
