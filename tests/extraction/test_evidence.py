from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.evidence import EvidenceVerificationError, normalize_for_matching, verify_evidence
from slr_pipeline.extraction.parsing import ParsedChunk


def _chunk(**overrides) -> ParsedChunk:
    defaults = dict(chunk_id="doc-chunk-0001", order_index=0, section="Methods", page_min=5, page_max=5,
                     text="Participants were randomly assigned to two groups.")
    defaults.update(overrides)
    return ParsedChunk(**defaults)


def test_verify_evidence_success():
    chunk = _chunk()
    result = verify_evidence(
        chunk_id="doc-chunk-0001", quotation="randomly assigned to two groups",
        chunk_map={chunk.chunk_id: chunk}, seen_keys=set(),
    )
    assert result.chunk_id == "doc-chunk-0001"
    assert result.page == "5"
    assert result.section == "Methods"


def test_verify_evidence_unknown_chunk_id():
    chunk = _chunk()
    with pytest.raises(EvidenceVerificationError, match="unknown chunk_id"):
        verify_evidence(
            chunk_id="does-not-exist", quotation="randomly assigned",
            chunk_map={chunk.chunk_id: chunk}, seen_keys=set(),
        )


def test_verify_evidence_quotation_not_in_chunk():
    chunk = _chunk()
    with pytest.raises(EvidenceVerificationError, match="not found"):
        verify_evidence(
            chunk_id=chunk.chunk_id, quotation="this text is not in the chunk at all",
            chunk_map={chunk.chunk_id: chunk}, seen_keys=set(),
        )


def test_verify_evidence_allows_whitespace_normalization():
    chunk = _chunk(text="Participants were   randomly\nassigned to two groups.")
    result = verify_evidence(
        chunk_id=chunk.chunk_id, quotation="randomly assigned to two groups",
        chunk_map={chunk.chunk_id: chunk}, seen_keys=set(),
    )
    assert result.excerpt == "randomly assigned to two groups"


def test_verify_evidence_allows_nfc_unicode_normalization():
    import unicodedata

    composed = unicodedata.normalize("NFC", "café results")
    decomposed = unicodedata.normalize("NFD", "café results")
    chunk = _chunk(text=f"The {composed} were significant.")
    result = verify_evidence(
        chunk_id=chunk.chunk_id, quotation=decomposed, chunk_map={chunk.chunk_id: chunk}, seen_keys=set(),
    )
    assert result is not None


def test_verify_evidence_rejects_paraphrase_not_case_folded_or_fuzzy():
    """Case must still match exactly (mod whitespace/Unicode) -- this is
    NOT fuzzy/word-overlap matching."""
    chunk = _chunk(text="Participants were randomly assigned to two groups.")
    with pytest.raises(EvidenceVerificationError):
        verify_evidence(
            chunk_id=chunk.chunk_id, quotation="RANDOMLY ASSIGNED",
            chunk_map={chunk.chunk_id: chunk}, seen_keys=set(),
        )


def test_verify_evidence_rejects_duplicate():
    chunk = _chunk()
    seen: set = set()
    verify_evidence(chunk_id=chunk.chunk_id, quotation="randomly assigned", chunk_map={chunk.chunk_id: chunk}, seen_keys=seen)
    with pytest.raises(EvidenceVerificationError, match="duplicate"):
        verify_evidence(chunk_id=chunk.chunk_id, quotation="randomly assigned", chunk_map={chunk.chunk_id: chunk}, seen_keys=seen)


def test_verify_evidence_rejects_empty_quotation():
    chunk = _chunk()
    with pytest.raises(EvidenceVerificationError, match="malformed"):
        verify_evidence(chunk_id=chunk.chunk_id, quotation="   ", chunk_map={chunk.chunk_id: chunk}, seen_keys=set())


def test_verify_evidence_rejects_empty_chunk_id():
    chunk = _chunk()
    with pytest.raises(EvidenceVerificationError, match="malformed"):
        verify_evidence(chunk_id="", quotation="randomly assigned", chunk_map={chunk.chunk_id: chunk}, seen_keys=set())


def test_normalize_for_matching_does_not_casefold():
    assert normalize_for_matching("Hello World") != normalize_for_matching("hello world")


def test_normalize_for_matching_collapses_whitespace():
    assert normalize_for_matching("a  \n\t b") == normalize_for_matching("a b")
