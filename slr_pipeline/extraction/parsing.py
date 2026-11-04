"""A small, Docling-only, page-aware parser -- deliberately independent of
FT_SCREENING's own `ft_screening.document.DoclingDocumentParser` (this
package's requirement: "Do not import FT_SCREENING internals"). Simpler
than that class on purpose: no artifact-manifest content validation, no
per-run converter caching/locking -- this is a v1 extraction-specific
parser, not a shared production component.

No silent fallback: if Docling is unavailable or conversion fails, this
raises `DocumentParseError` -- there is no second parser this module
quietly tries instead. `charting.py`'s `_get_section_chunks()` (Docling,
falling back to pymupdf4llm on ANY failure, including "not installed") is
exactly the pattern this module refuses to repeat: a silent fallback
means two different documents in the same run could have been parsed by
two different parsers with different provenance, without that ever being
visible in the result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .hashing import file_sha256

PARSER_NAME = "docling"

# Same segments FT_SCREENING's own `ft_screening.docling_artifacts.
# compute_tree_digest` excludes (huggingface_hub's own bookkeeping --
# `.cache/huggingface/...`, `.locks/...` -- legitimately varies across
# otherwise-identical downloads and would make the digest spuriously
# unstable). Reimplemented here, not imported (this module's
# FT_SCREENING-internals boundary, see module docstring); must stay
# algorithmically identical to that function, since the digest asserted
# here is compared against `assets/models/DOCLING_ASSETS.lock`'s own
# `tree_digest`, itself produced by the real `compute_tree_digest`.
_EXCLUDED_ARTIFACT_PATH_SEGMENTS = (".cache", ".locks")


class DocumentParseError(Exception):
    """Docling is not installed, or conversion of this specific PDF
    failed. There is no fallback parser -- the caller must route this
    document to review, never silently substitute a weaker parse."""


@dataclass(frozen=True)
class DoclingParserOptions:
    """Recorded verbatim into run identity (requirement 11: "parser/
    chunking identity") -- these are the ONLY two toggles this module
    exposes, matching Docling's own most consequential options, plus the
    deterministic chunk-size cap."""

    do_ocr: bool = True
    do_table_structure: bool = True
    max_chunk_chars: int = 3000  # deterministic chunk-size cap; see _group_items_into_chunks

    def __post_init__(self) -> None:
        if self.max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive, got {self.max_chunk_chars!r}")


@dataclass(frozen=True)
class ParsedChunk:
    """One trusted, content-addressed unit of parsed document text.
    `chunk_id` is deterministic (derived from document content + position,
    never random) so the SAME PDF parsed twice with the SAME options
    produces IDENTICAL chunk_ids -- required for evidence verification
    (requirement 7: "verify that the chunk exists") and for run-identity
    reproducibility."""

    chunk_id: str
    order_index: int  # 0-based, strict document order -- the basis for deterministic batching
    section: str
    page_min: Optional[int]
    page_max: Optional[int]
    text: str

    @property
    def page_range(self) -> Optional[str]:
        if self.page_min is None:
            return None
        if self.page_max is None or self.page_max == self.page_min:
            return str(self.page_min)
        return f"{self.page_min}-{self.page_max}"


@dataclass(frozen=True)
class ParsedDocument:
    document_sha256: str
    parser_name: str
    parser_version: str
    parser_options: DoclingParserOptions
    chunks: tuple[ParsedChunk, ...]
    # None in ordinary/production use (no artifacts-directory pinning --
    # Docling resolves its own default cache, exactly as before this
    # field existed). Set to the caller's verified
    # `expected_artifacts_manifest_digest` (see `parse_pdf`) only when a
    # pinned, hash-checked model-artifacts directory was actually bound
    # for this parse -- the reference-run mechanism's own use case.
    artifacts_manifest_digest: Optional[str] = None

    def chunk_map(self) -> dict[str, ParsedChunk]:
        return {c.chunk_id: c for c in self.chunks}


def installed_docling_version() -> str:
    """The ACTUAL installed docling package version -- read via package
    metadata, never a hardcoded literal (same discipline as FT_SCREENING's
    own `DoclingDocumentParser.installed_parser_version()`, reimplemented
    fresh here rather than imported, per this package's FT_SCREENING-
    internals boundary)."""
    return importlib.metadata.version("docling")


def _compute_tree_digest(artifacts_path: Path) -> str:
    """Canonical content digest of every actual model file under
    `artifacts_path`, excluding `_EXCLUDED_ARTIFACT_PATH_SEGMENTS`. See
    that constant's own comment for why this is a local reimplementation,
    not an import."""
    entries: list[str] = []
    for path in sorted(artifacts_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifacts_path)
        if any(part in _EXCLUDED_ARTIFACT_PATH_SEGMENTS for part in relative.parts):
            continue
        file_hash = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                file_hash.update(block)
        entries.append(f"{relative.as_posix()}:{file_hash.hexdigest()}")
    return hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()


def _group_items_into_chunks(
    items: list[tuple[str, str, Optional[int], bool]],  # (section, text, page_no, is_heading) in document order
    max_chunk_chars: int,
) -> list[tuple[str, Optional[int], Optional[int], str]]:  # (section, page_min, page_max, text)
    """Deterministic, fixed-size-with-natural-boundaries grouping -- NEVER
    a semantic splitter (requirement 6: "no ... semantic splitter ...").
    Flushes the accumulated chunk whenever the section changes OR the next
    item would push it over `max_chunk_chars`; never splits an item's own
    text mid-way, so a chunk boundary always falls between two Docling
    items, not inside one.

    A heading item (`is_heading=True`) is ALWAYS emitted as its OWN
    dedicated chunk, with its own page provenance and document-order
    index -- never fused with the body text that follows it (review
    finding D: a heading on page 3 followed by body text on page 4 must
    not collapse into one "3-4" chunk that loses the heading's own page).
    Body items still carry the heading's text as their own `section`
    metadata (set by the caller before appending them), so the heading's
    identity is not lost from body chunks either -- it just isn't
    RE-QUOTED inside them."""
    grouped: list[tuple[str, Optional[int], Optional[int], str]] = []
    current_section: Optional[str] = None
    current_texts: list[str] = []
    current_pages: list[int] = []

    def flush() -> None:
        if not current_texts:
            return
        text = "\n\n".join(current_texts)
        page_min = min(current_pages) if current_pages else None
        page_max = max(current_pages) if current_pages else None
        grouped.append((current_section or "Preamble", page_min, page_max, text))

    for section, text, page_no, is_heading in items:
        if not text.strip():
            continue

        if is_heading:
            flush()
            current_texts, current_pages = [], []
            current_section = section
            current_texts.append(text)
            if page_no is not None:
                current_pages.append(page_no)
            flush()
            current_texts, current_pages = [], []
            continue

        starting_new_section = section != current_section and current_texts
        would_overflow = current_texts and (
            sum(len(t) for t in current_texts) + len(text) > max_chunk_chars
        )
        if starting_new_section or would_overflow:
            flush()
            current_texts = []
            current_pages = []
        current_section = section
        current_texts.append(text)
        if page_no is not None:
            current_pages.append(page_no)

    flush()
    return grouped


def parse_pdf(
    pdf_path: str | Path, *, options: DoclingParserOptions | None = None,
    expected_artifacts_manifest_digest: Optional[str] = None,
) -> ParsedDocument:
    """The ONLY parse entry point this package has. Raises
    `DocumentParseError` on any failure -- callers decide what "route to
    review" means for a document that can't be parsed; this function never
    decides that for them by trying something else.

    `expected_artifacts_manifest_digest` is `None` in ordinary/production
    use -- Docling resolves its own default model cache, exactly as
    before this parameter existed (no behaviour change). When given (the
    reference-run mechanism's own use case), `DOCLING_CACHE_DIR` MUST be
    set: `Path(os.environ["DOCLING_CACHE_DIR"]) / "models"` is resolved,
    hashed via `_compute_tree_digest`, and required to equal this value
    BEFORE Docling is even imported -- a mismatch (wrong, missing, or
    partial model directory) is a controlled `DocumentParseError`, never
    a silent fall-through to whatever Docling would otherwise resolve
    from a mutable, possibly network-fetched, default location. The
    verified directory is then passed to Docling as an explicit
    `artifacts_path`, and the same verified digest is recorded on the
    returned `ParsedDocument`."""
    options = options or DoclingParserOptions()
    pdf_path = Path(pdf_path)
    document_sha256 = file_sha256(pdf_path)

    artifacts_path: Optional[Path] = None
    if expected_artifacts_manifest_digest is not None:
        cache_dir = os.environ.get("DOCLING_CACHE_DIR")
        if not cache_dir:
            raise DocumentParseError(
                "expected_artifacts_manifest_digest was given but DOCLING_CACHE_DIR is not set -- "
                "cannot resolve which model artifacts directory to bind identity to"
            )
        artifacts_path = Path(cache_dir) / "models"
        if not artifacts_path.is_dir():
            raise DocumentParseError(
                f"Docling model artifacts directory not found at {artifacts_path!s} "
                f"(from DOCLING_CACHE_DIR={cache_dir!r})"
            )
        actual_digest = _compute_tree_digest(artifacts_path)
        if actual_digest != expected_artifacts_manifest_digest:
            raise DocumentParseError(
                f"Docling model artifacts at {artifacts_path!s} do not match the expected tree digest "
                f"{expected_artifacts_manifest_digest!r} (actual: {actual_digest!r})"
            )

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling_core.types.doc import DocItemLabel
    except ImportError as exc:
        raise DocumentParseError(f"docling is not installed or unavailable: {type(exc).__name__}") from exc

    try:
        pipeline_options = PdfPipelineOptions(
            do_ocr=options.do_ocr, do_table_structure=options.do_table_structure, artifacts_path=artifacts_path,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(str(pdf_path))
    except Exception as exc:
        raise DocumentParseError(f"Docling conversion failed for {pdf_path.name}: {type(exc).__name__}") from exc

    items: list[tuple[str, str, Optional[int], bool]] = []
    current_section = "Preamble"
    for item, _level in result.document.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        page_no = None
        prov = getattr(item, "prov", None)
        if prov:
            page_no = prov[0].page_no
        if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
            # The heading becomes a citable item IN ITS OWN RIGHT, with
            # its own page provenance -- never silently dropped as pure
            # metadata (the "structural-heading evidence gap" this
            # package's own review flagged: a quotation drawn from a
            # heading must be verifiable like any other text). It is
            # appended exactly ONCE here, at the point Docling reports
            # it, and `_group_items_into_chunks` below ALWAYS gives it
            # its own dedicated chunk (review finding D) -- never fused
            # with the body text that follows.
            current_section = text.strip()
            items.append((current_section, current_section, page_no, True))
            continue
        items.append((current_section, text.strip(), page_no, False))

    grouped = _group_items_into_chunks(items, options.max_chunk_chars)

    chunks = tuple(
        ParsedChunk(
            chunk_id=f"{document_sha256[:16]}-chunk-{index:04d}",
            order_index=index,
            section=section,
            page_min=page_min,
            page_max=page_max,
            text=text,
        )
        for index, (section, page_min, page_max, text) in enumerate(grouped)
    )

    if not chunks:
        # A controlled failure, not a silently-empty ParsedDocument that
        # would later produce zero batches and an opaque
        # DocumentExtractionResult validation error deep inside
        # `results.py` (batch_outcomes requires min_length=1). An empty,
        # image-only/unOCR'd, or otherwise unsupported document must be
        # routed to review explicitly, the same as any other parse
        # failure.
        raise DocumentParseError(
            f"Docling produced zero citable chunks for {pdf_path.name} -- likely an empty, "
            "image-only, or otherwise unsupported document"
        )

    return ParsedDocument(
        document_sha256=document_sha256,
        parser_name=PARSER_NAME,
        parser_version=installed_docling_version(),
        parser_options=options,
        chunks=chunks,
        artifacts_manifest_digest=expected_artifacts_manifest_digest,
    )
