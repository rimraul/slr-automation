"""Track 2 Step 12A, sub-step (a) -- authors the self-contained,
independently-testable portion of the synthetic fixture corpus under
`tests/reference_run/fixtures/corpus/`: the 7 native `fx_<db>.json`
files, the 6 synthetic PDFs (PyMuPDF, each carrying the mandated visible
"synthetic test data" banner), `LICENSE.md`, `PROVENANCE.md`,
`SCENARIO_TABLE.md`, `known_real_doi_denylist.txt`, and `CORPUS_ORACLE.json`
(schema version "2", `oracle_kind="software_conformance"`).

Everything this script writes is generated from, or validated against,
`slr_pipeline.reference_fixture_answer_key` -- the ONE structured answer
key (`track2_step12_revision_proposal_2026-09-02.md`, §2.1). Nothing here
is a second, independently-authored copy of the same facts.

**Deliberately NOT written by this script** (sub-steps (b)-(h), a later
phase -- see `track2_step12_revision_proposal_2026-09-02.md` §2.1's own
(d) note): `track2/readiness_protocol.md`, `fixtures/prisma_scr_matrix.json`,
`fixtures/corpus/replay_cassette.json`, `fixtures/spec/*.request.json`,
`reference_spec.json`, and the `hash_reference_inputs.py` targets that
depend on any of those (`fixture_corpus_digest`, `prisma_scr_matrix_sha256`,
`reference_spec_sha256`). This script's own `CORPUS_ORACLE.json` output
therefore has real, computed `pdf_sha256` values but does not yet carry
per-surface `expected_calls` (those are a step-12(d) cassette-authoring
output).

Usage:
    python scripts/author_fixture_corpus.py --write
    python scripts/author_fixture_corpus.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF -- already a test-only dependency (test_parsing_fixture_pdf.py)
import matplotlib  # locked transitive dependency of pymupdf4llm/pymupdf-layout;
# used here ONLY for its bundled, cross-platform DejaVu Sans font file (see
# _UNICODE_FONT_PATH below) -- never for plotting.

from slr_pipeline import reference_fixture_answer_key as ak
from slr_pipeline.corpus_oracle import CorpusOracle
from slr_pipeline.fixture_corpus_digest import compute_corpus_digest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Direct invocation (`python scripts/author_fixture_corpus.py`) sets
# `sys.path[0]` to this file's own directory (`scripts/`), not the repo
# root, so `scripts` (this file's own parent package, needed below for
# `from scripts.hash_reference_inputs import register_target`) would not
# otherwise be importable -- same fix as author_prisma_scr_matrix.py's own
# top-of-file guard.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CORPUS_ROOT = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "corpus"
PDFS_DIR = CORPUS_ROOT / "pdfs"

# Codex review (2026-09-03) finding 1: PyMuPDF's base14 "helv" font cannot
# encode the banner's em dash (U+2014) -- `page.insert_text(..., fontname=
# "helv")` silently substitutes it with a replacement-character glyph on
# extraction, and single-line, fixed-position `insert_text` calls clip any
# evidence-field line wider than the page's usable width. DejaVu Sans (a
# real TrueType font, bundled with matplotlib -- already a locked,
# cross-platform dependency via pymupdf4llm/pymupdf-layout, never a
# Windows-only system font path) round-trips the em dash exactly and is
# embedded once per generated PDF via `fontfile=` on the first
# `insert_textbox` call. Verified directly: base14 "helv" turns the banner's
# em dash into "�" on extraction; DejaVu Sans does not.
_UNICODE_FONT_PATH = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
_UNICODE_FONT_NAME = "fxunicode"

SYNTHETIC_BANNER = "Synthetic test data — pipeline conformance fixture, not scientific evidence."

PROVENANCE_BANNER = (
    "Track 2 pipeline conformance scenario — synthetic test data, not scientific "
    "evidence. Every finding, funding statement, and country named below is "
    "fabricated to exercise a specific pipeline branch (see SCENARIO_TABLE.md) "
    "and must never be cited as a real study."
)

# Berg2023's four real included-study DOIs (the other two Berg2023 included
# studies have doi: null) -- `chapter_claim_sources.json`'s own
# `relationships` entries key on `source_id` strings like `"lit:prisma2020"`,
# never a DOI, so it contributes zero additional entries here (checked
# 2026-09-03, no `10\.\d{4,9}/...`-shaped substring found in that file).
_DENYLIST_DOIS = (
    "10.1371/journal.pone.0237628",
    "10.2196/31337",
    "10.1055/s-0040-1714090",
    "10.1097/dmp.0000000000000192",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_pdf(record_id: str, lines: list[str]) -> bytes:
    """One synthetic PDF: a `Methods`-style heading, the given content
    lines wrapped into a text box (never a fixed single-line placement --
    a long evidence-field line must never silently clip off the page
    edge), and the mandated visible warning banner (not just page 1 -- a
    reader opening any page sees it). All three text boxes use the
    embedded Unicode font (see `_UNICODE_FONT_PATH` above) and each
    `insert_textbox` call's return value is checked: PyMuPDF returns a
    negative number of points of UNFIT text on overflow, so a negative
    return is a hard authoring error, not a warning."""
    doc = fitz.open()
    page = doc.new_page()

    heading_rect = fitz.Rect(72, 72, 540, 100)
    heading_overflow = page.insert_textbox(
        heading_rect, f"Fixture record {record_id}",
        fontsize=18, fontname=_UNICODE_FONT_NAME, fontfile=_UNICODE_FONT_PATH,
    )
    if heading_overflow < 0:
        raise ValueError(f"{record_id}: heading overflowed its text box by {-heading_overflow:.1f}pt")

    body_rect = fitz.Rect(72, 110, 540, 760)
    body_overflow = page.insert_textbox(
        body_rect, "\n".join(lines), fontsize=10.5, fontname=_UNICODE_FONT_NAME,
    )
    if body_overflow < 0:
        raise ValueError(
            f"{record_id}: body content overflowed its text box by {-body_overflow:.1f}pt -- "
            "enlarge the box or paginate across multiple pages"
        )

    banner_rect = fitz.Rect(72, 770, 540, 792)
    for pg in doc:
        banner_overflow = pg.insert_textbox(
            banner_rect, SYNTHETIC_BANNER, fontsize=8, fontname=_UNICODE_FONT_NAME,
        )
        if banner_overflow < 0:
            raise ValueError(f"{record_id}: banner overflowed its text box by {-banner_overflow:.1f}pt")

    # Subset the embedded font down to only the glyphs actually used, then
    # compress -- without this, every one of the 6 PDFs embeds the ENTIRE
    # DejaVu Sans font (~750 KB) uncompressed; subsetting + deflate brings
    # each fixture PDF back down to a repo-appropriate size (checked
    # 2026-09-03: ~750 KB -> ~40 KB per file) with no change to extracted
    # text content.
    doc.subset_fonts()
    # `no_new_id=1`: MuPDF otherwise stamps a fresh random /ID pair into the
    # trailer on every save, and `subset_fonts()` also assigns a random
    # 6-letter subset tag (e.g. "PHSAUG+DejaVu") to the embedded font's
    # PostScript name -- both are cosmetic (never affect rendering or
    # extracted text) but make two `--write` runs over IDENTICAL input
    # produce different bytes, defeating this script's own `--check`
    # byte-comparison contract and CORPUS_ORACLE.json's pdf_sha256 pin.
    # Verified 2026-09-03: with both of these normalised, two consecutive
    # builds of the same record are byte-identical.
    data = doc.tobytes(garbage=4, deflate=True, no_new_id=1)
    doc.close()
    data = re.sub(rb"[A-Z]{6}\+DejaVu", b"AAAAAA+DejaVu", data)
    return data


def _pdf_lines_for(record_id: str) -> list[str]:
    """Renders the answer key's extraction values (for the 5 eligible
    records) or the FT-eligibility-relevant content (for fx-0018, the one
    full-text exclusion) into plain text lines. This is prose ABOUT the
    answer key's own data, not a second independently-invented text."""
    rec = ak.record(record_id)
    lines = [f"Title: {rec['title']}", "", "Abstract:", rec["abstract"], ""]
    if record_id in ak.EXTRACTION_ANSWERS:
        e = ak.EXTRACTION_ANSWERS[record_id]
        lines += [
            "Methods", f"Study design: {e['study_design']}", f"Country: {e['country']}",
            f"Population: {e['population']}", f"Concept: {e['concept']}", f"Context: {e['context']}", "",
            "Results",
        ]
        lines += [f"- {finding}" for finding in e["key_findings"]]
        lines += ["", "Limitations", e["limitations"], "", "Funding"]
        lines.append(
            "Funding sources: " + (", ".join(e["funding_sources"]) if e["funding_sources"] else "(not stated)")
        )
        lines.append("Funder role: " + (e["funder_role"] if e["funder_role"] else "(not stated)"))
    else:
        # fx-0018 -- the one full-text exclusion: the full text names the
        # setting explicitly as peri-urban/metropolitan-adjacent, the
        # detail the FULL-TEXT read (never the title/abstract) reveals.
        ft = ak.FULL_TEXT_ELIGIBILITY[record_id]
        lines += [
            "Methods",
            ft["criteria"]["mbse_or_simulation"]["evidence"],
            "",
            "Setting",
            ft["criteria"]["rural_healthcare"]["evidence"],
            "",
            "Results",
            ft["criteria"]["disaster_context"]["evidence"],
        ]
    return lines


def write_pdfs() -> dict[str, str]:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_sha256: dict[str, str] = {}
    for record_id in ak.FULL_TEXT_SOUGHT:
        data = _build_pdf(record_id, _pdf_lines_for(record_id))
        path = PDFS_DIR / f"{record_id}.pdf"
        path.write_bytes(data)
        pdf_sha256[record_id] = _sha256_bytes(data)
    return pdf_sha256


def _license_text() -> str:
    return """\
# License

Every file under `tests/reference_run/fixtures/corpus/` is licensed
**CC BY 4.0** (Creative Commons Attribution 4.0 International):
<https://creativecommons.org/licenses/by/4.0/>.

All content is author-created-synthetic (invented for this chapter's Track
2 pipeline-conformance rehearsal; see `PROVENANCE.md`). Attribution: the
SLR_chapter authors.
"""


def write_license() -> None:
    (CORPUS_ROOT / "LICENSE.md").write_text(_license_text(), encoding="utf-8")


def _provenance_text() -> str:
    lines = [
        PROVENANCE_BANNER,
        "",
        "## What this corpus is",
        "",
        "This directory is an entirely synthetic bibliographic corpus (30 records "
        "across 7 fixture databases), authored solely to exercise specific branches "
        "of the chapter's scoping-review automation pipeline: exact-title "
        "deduplication, title/abstract screening, full-text eligibility "
        "adjudication (including its one full-text-only exclusion), 9-field "
        "extraction (including 5 distinct funding-statement presentations), and "
        "map/refine synthesis. No record describes a real study, no country, "
        "funder, or finding named anywhere in this corpus is real, and no PDF in "
        "`pdfs/` should ever be read as reporting an actual result.",
        "",
        "The single structured answer key this corpus is generated from is "
        "`slr_pipeline/reference_fixture_answer_key.py`; `SCENARIO_TABLE.md` "
        "renders it into one row per record, naming the pipeline branch each "
        "record exists to exercise.",
        "",
        "## Licence",
        "",
        "CC BY 4.0 -- see `LICENSE.md`.",
        "",
        "## Real-DOI exclusion",
        "",
        "`known_real_doi_denylist.txt` lists every real DOI this corpus must "
        "never contain (Berg2023's real included-study DOIs, plus every DOI in "
        "`chapter_claim_sources.json`); every fixture record has `doi: null`.",
        "",
    ]
    return "\n".join(lines)


def write_provenance() -> None:
    (CORPUS_ROOT / "PROVENANCE.md").write_text(_provenance_text(), encoding="utf-8")


def _scenario_table_text() -> str:
    return "# Scenario table\n\n" + ak.scenario_table_markdown()


def write_scenario_table() -> None:
    (CORPUS_ROOT / "SCENARIO_TABLE.md").write_text(_scenario_table_text(), encoding="utf-8")


def _denylist_text() -> str:
    lines = [
        "# known_real_doi_denylist.txt",
        "#",
        "# Real DOIs that must NEVER appear anywhere in this synthetic fixture",
        "# corpus. Berg2023's four real included-study DOIs (the other two",
        "# Berg2023 included studies have doi: null) plus every DOI referenced in",
        "# chapter_claim_sources.json (none found, checked 2026-09-03).",
        *_DENYLIST_DOIS,
        "",
    ]
    return "\n".join(lines)


def write_denylist() -> None:
    (CORPUS_ROOT / "known_real_doi_denylist.txt").write_text(_denylist_text(), encoding="utf-8")


def _corpus_oracle_text(pdf_sha256: dict[str, str]) -> str:
    payload = ak.corpus_oracle_payload(pdf_sha256)
    CorpusOracle.model_validate(payload)  # fail closed before writing anything invalid
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_corpus_oracle(pdf_sha256: dict[str, str]) -> None:
    (CORPUS_ROOT / "CORPUS_ORACLE.json").write_text(_corpus_oracle_text(pdf_sha256), encoding="utf-8")


def write_all() -> None:
    ak.dump_native_fixture_files(root=CORPUS_ROOT)
    pdf_sha256 = write_pdfs()
    write_license()
    write_provenance()
    write_scenario_table()
    write_denylist()
    write_corpus_oracle(pdf_sha256)


def _assert_file_matches(path: Path, expected: str) -> None:
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(f"{path} does not match its freshly-regenerated expected content")


def _assert_pdf_content(record_id: str, pdf_bytes: bytes) -> None:
    """Codex review (2026-09-03) finding 2: sha256-hash-comparing PDF bytes
    against CORPUS_ORACLE.json only proves the file hasn't changed since it
    was last written -- it does NOT prove the file's extracted text is
    correct/un-clipped (a corrupted-but-hash-matching file would pass
    forever). Independently re-extracts each PDF's text and asserts every
    expected content line (normalised for whitespace -- PyMuPDF's
    `insert_textbox` reflows original line breaks, and its embedded-font
    path substitutes NBSP for the ASCII space, both cosmetic) is present in
    full, plus the exact banner string (including its em dash) verbatim."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        full_text = "".join(page.get_text() for page in doc)
    finally:
        doc.close()
    if SYNTHETIC_BANNER not in full_text:
        raise ValueError(f"{record_id}.pdf: exact banner text not found verbatim in extracted text")
    normalized = re.sub(r"\s+", " ", full_text.replace("\xa0", " ")).strip()
    for line in _pdf_lines_for(record_id):
        if not line.strip():
            continue
        expected_line = re.sub(r"\s+", " ", line).strip()
        if expected_line not in normalized:
            raise ValueError(f"{record_id}.pdf: expected content clipped or missing: {expected_line!r}")


def check_all() -> None:
    """Read-only: re-derives every deterministic artifact this script
    produces from the answer key, and byte-compares (or, for PDFs,
    semantically validates) each one against what is on disk -- never
    writes. Codex review (2026-09-03) finding 2: previously covered only
    native/*.json and PDF sha256-vs-oracle -- LICENSE.md, PROVENANCE.md,
    SCENARIO_TABLE.md, known_real_doi_denylist.txt, and CORPUS_ORACLE.json's
    other fields could drift or be hand-edited and `--check` would still
    report OK. Every generated file is now covered."""
    for db in ak.CANONICAL_DATABASES:
        path = CORPUS_ROOT / "native" / f"fx_{db}.json"
        expected = json.dumps(ak.native_payload(db), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        _assert_file_matches(path, expected)

    pdf_sha256 = {}
    for record_id in ak.FULL_TEXT_SOUGHT:
        path = PDFS_DIR / f"{record_id}.pdf"
        pdf_bytes = path.read_bytes()
        pdf_sha256[record_id] = _sha256_bytes(pdf_bytes)
        _assert_pdf_content(record_id, pdf_bytes)

    _assert_file_matches(CORPUS_ROOT / "LICENSE.md", _license_text())
    _assert_file_matches(CORPUS_ROOT / "PROVENANCE.md", _provenance_text())
    _assert_file_matches(CORPUS_ROOT / "SCENARIO_TABLE.md", _scenario_table_text())
    _assert_file_matches(CORPUS_ROOT / "known_real_doi_denylist.txt", _denylist_text())
    _assert_file_matches(CORPUS_ROOT / "CORPUS_ORACLE.json", _corpus_oracle_text(pdf_sha256))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.write:
        write_all()
        print(f"wrote fixture corpus sub-step (a) content under {CORPUS_ROOT.relative_to(REPO_ROOT)}")
    else:
        check_all()
        print("OK: fixture corpus sub-step (a) content matches the answer key")
    return 0


def _register_targets() -> None:
    from scripts.hash_reference_inputs import register_target

    # Step 12(e): whole-file sha256 of CORPUS_ORACLE.json, computed once
    # both pdf_sha256[*] (this script, sub-step (a)) and total_replay_calls
    # (also sub-step (a) -- deterministic from the frozen answer key, needs
    # no live cassette run) are already in place, making the file
    # content-final. See §0.4 / §13 step 12(e).
    @register_target("corpus_oracle_pdf_sha256")
    def _corpus_oracle_pdf_sha256_target() -> Path:
        return CORPUS_ROOT / "CORPUS_ORACLE.json"

    # Step 12(f): §7.1's multi-file digest over every regular file under
    # corpus/ -- not a single file's bytes, so this generator returns an
    # already-computed hex string instead of a Path (see
    # hash_reference_inputs.py's Path|str generator contract). Safe only
    # after every corpus-file-touching sub-step ((a) this script, (d)
    # cassette authoring, (e) the oracle's pdf_sha256[*]/total_replay_calls)
    # has already written its final bytes under corpus/ -- called fresh on
    # every --write/--check, so it always reflects the corpus tree's
    # current, real on-disk content, never a stale cached value.
    @register_target("fixture_corpus_digest")
    def _fixture_corpus_digest_target() -> str:
        return compute_corpus_digest(CORPUS_ROOT)


_register_targets()


if __name__ == "__main__":
    raise SystemExit(main())
