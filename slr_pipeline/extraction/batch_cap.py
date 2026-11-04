"""Single source of truth for a document's parsed content and batch plan,
shared by the pre-billing offline cap check AND `extract_document()`'s own
real, billed batch loop -- so there is exactly ONE parse+batch computation
per document, never two independent ones that could silently diverge (a
different file on disk, different parser artifacts/options, a different
batch config) between the check and what actually gets billed.

`expected_artifacts_manifest_digest` is mandatory here (unlike
`parse_pdf()`'s own optional default): this module exists specifically for
callers that must not silently fall through to Docling's mutable default
model cache (which can resolve to a different, possibly network-fetched,
location) -- see `parsing.py::parse_pdf` for the verification this performs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .batching import DEFAULT_MAX_BATCH_CHARS, ExtractionBatch, build_batches
from .parsing import DoclingParserOptions, ParsedDocument, parse_pdf


class ExtractionBatchCapExceededError(Exception):
    """The document parsed to more extraction batches than the enforced
    ceiling allows. `parsed`/`batches` carry the COMPLETE, untruncated
    parse and batch plan -- the document is never cut down to fit the
    bound; a caller that wants to preserve this for review has the full
    evidence available on the exception itself."""

    def __init__(self, message: str, *, parsed: ParsedDocument, batches: tuple[ExtractionBatch, ...]) -> None:
        super().__init__(message)
        self.parsed = parsed
        self.batches = batches


def enforce_extraction_batch_cap(
    pdf_path: str | Path,
    *,
    expected_artifacts_manifest_digest: str,
    max_batches: int,
    parser_options: Optional[DoclingParserOptions] = None,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> tuple[ParsedDocument, tuple[ExtractionBatch, ...]]:
    """Parses `pdf_path` in full (no network call: Docling itself performs
    no network access, and `expected_artifacts_manifest_digest` forces the
    model-artifacts directory to be pinned/hash-verified rather than
    resolved from a mutable default -- see `parse_pdf`), builds its
    deterministic batch plan, and raises `ExtractionBatchCapExceededError`
    if that plan exceeds `max_batches`. Returns `(parsed, batches)`
    otherwise -- the caller (e.g. `extract_document`) must use THESE exact
    values for the real extraction loop, never re-parse/re-batch."""
    if max_batches < 1:
        raise ValueError(f"max_batches must be >= 1, got {max_batches!r}")
    parsed = parse_pdf(
        pdf_path, options=parser_options, expected_artifacts_manifest_digest=expected_artifacts_manifest_digest,
    )
    batches = build_batches(parsed.chunks, max_batch_chars=max_batch_chars)
    if len(batches) > max_batches:
        raise ExtractionBatchCapExceededError(
            f"{pdf_path}: {len(batches)} extraction batches exceeds the enforced cap of {max_batches}. "
            "The document was parsed in full and is NOT truncated -- route it to review instead of "
            "retrying with a smaller input.",
            parsed=parsed, batches=batches,
        )
    return parsed, batches


__all__ = ["ExtractionBatchCapExceededError", "enforce_extraction_batch_cap"]
