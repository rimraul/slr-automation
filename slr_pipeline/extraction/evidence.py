"""Evidence verification -- requirement 7's exact rules: verify the chunk
exists, verify the quotation against its ACTUAL text (not the model's
claim), derive page/section from the trusted chunk (never a model-
asserted page number), reject duplicate/malformed evidence. This is the
one piece of the whole extraction engine where a model's OWN assertion is
never trusted at face value -- everything here is checked against the
`ParsedChunk` data `parsing.py` produced, which is itself derived
directly from Docling's output, not from anything the LLM said.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .parsing import ParsedChunk


class EvidenceVerificationError(ValueError):
    """A model-returned evidence item cannot be trusted: unknown chunk_id,
    a quotation not actually present in that chunk's text (even after the
    one documented normalization below), or malformed/duplicate input.
    Callers must route the owning field to human review when this is
    raised -- never silently drop the evidence or accept it anyway."""


@dataclass(frozen=True)
class VerifiedEvidence:
    """The ONLY evidence shape that ever reaches a stored result.
    `page`/`section` come from the trusted `ParsedChunk`, never from
    anything the model said (requirement 7: "derive page and section from
    the trusted chunk, never from model-asserted page numbers")."""

    chunk_id: str
    excerpt: str  # the caller-supplied quotation, as given (post-strip), NOT the normalized matching form
    page: Optional[str]
    section: str


def normalize_for_matching(text: str) -> str:
    """The ONLY normalization evidence verification performs, and it is
    exactly this and nothing more (requirement 7: "allowing only
    documented whitespace/Unicode normalisation"):
      1. Unicode NFC normalization (composed form) -- so a quote typed
         with a decomposed accent still matches source text using the
         composed form, or vice versa.
      2. Collapse every run of whitespace (spaces, tabs, newlines --
         including ones introduced by Docling's own paragraph/line
         breaks) to a single space, then strip leading/trailing
         whitespace.
    Deliberately NOT done: case-folding (a "verbatim quote" should match
    case), punctuation stripping, hyphenation/ligature repair, or fuzzy/
    word-overlap matching of the kind `charting.py`'s
    `_find_page_of_excerpt()` uses -- a quotation either matches exactly
    (mod whitespace/Unicode form) or it is rejected, not scored.
    """
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def verify_evidence(
    *,
    chunk_id: str,
    quotation: str,
    chunk_map: dict[str, ParsedChunk],
    seen_keys: set[tuple[str, str]],
) -> VerifiedEvidence:
    """Verifies ONE evidence item. `seen_keys` is mutated in place (add
    the new (chunk_id, normalized_quotation) key on success) -- callers
    verifying a whole list of evidence items for one field pass the SAME
    set across every call in that list, so a second, identical
    (chunk_id, quotation) pair is rejected as a duplicate regardless of
    which earlier call in the list first recorded it."""
    if not chunk_id or not isinstance(chunk_id, str):
        raise EvidenceVerificationError(f"malformed evidence: chunk_id must be a non-empty string, got {chunk_id!r}")
    if not quotation or not isinstance(quotation, str) or not quotation.strip():
        raise EvidenceVerificationError("malformed evidence: quotation must be a non-empty string")

    chunk = chunk_map.get(chunk_id)
    if chunk is None:
        raise EvidenceVerificationError(f"unknown chunk_id: {chunk_id!r} does not exist in this document's parse")

    normalized_quote = normalize_for_matching(quotation)
    normalized_chunk_text = normalize_for_matching(chunk.text)
    if normalized_quote not in normalized_chunk_text:
        raise EvidenceVerificationError(
            f"quotation not found in chunk {chunk_id!r}'s text, even after whitespace/Unicode normalization"
        )

    dedup_key = (chunk_id, normalized_quote)
    if dedup_key in seen_keys:
        raise EvidenceVerificationError(
            f"duplicate evidence: identical (chunk_id, quotation) already recorded for this field"
        )
    seen_keys.add(dedup_key)

    return VerifiedEvidence(chunk_id=chunk_id, excerpt=quotation.strip(), page=chunk.page_range, section=chunk.section)
