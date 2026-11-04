from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.batching import build_batches
from slr_pipeline.extraction.parsing import ParsedChunk


def _chunk(index: int, text: str) -> ParsedChunk:
    return ParsedChunk(chunk_id=f"doc-chunk-{index:04d}", order_index=index, section="S", page_min=1, page_max=1, text=text)


def test_build_batches_empty():
    assert build_batches(()) == ()


def test_build_batches_all_chunks_covered_no_truncation():
    chunks = tuple(_chunk(i, "x" * 100) for i in range(10))
    batches = build_batches(chunks, max_batch_chars=250)
    all_ids_in_batches = [cid for b in batches for cid in b.chunk_ids]
    assert all_ids_in_batches == [c.chunk_id for c in chunks]  # every chunk present, in order, none dropped


def test_build_batches_respects_char_budget():
    chunks = tuple(_chunk(i, "x" * 100) for i in range(10))
    batches = build_batches(chunks, max_batch_chars=250)
    for batch in batches:
        assert batch.total_chars <= 250 or len(batch.chunk_ids) == 1  # oversized single chunk is the only exception


def test_build_batches_oversized_single_chunk_gets_its_own_batch_not_truncated():
    huge_text = "y" * 5000
    chunks = (_chunk(0, huge_text),)
    batches = build_batches(chunks, max_batch_chars=1000)
    assert len(batches) == 1
    assert batches[0].chunk_ids == ("doc-chunk-0000",)
    assert batches[0].total_chars == 5000  # not truncated


def test_build_batches_deterministic():
    chunks = tuple(_chunk(i, "x" * 50) for i in range(7))
    batches1 = build_batches(chunks, max_batch_chars=120)
    batches2 = build_batches(chunks, max_batch_chars=120)
    assert batches1 == batches2


def test_build_batches_indices_are_sequential():
    chunks = tuple(_chunk(i, "x" * 100) for i in range(5))
    batches = build_batches(chunks, max_batch_chars=150)
    assert [b.batch_index for b in batches] == list(range(len(batches)))


def test_build_batches_rejects_zero_max_batch_chars():
    chunks = (_chunk(0, "x"),)
    with pytest.raises(ValueError, match="max_batch_chars"):
        build_batches(chunks, max_batch_chars=0)


def test_build_batches_rejects_negative_max_batch_chars():
    chunks = (_chunk(0, "x"),)
    with pytest.raises(ValueError, match="max_batch_chars"):
        build_batches(chunks, max_batch_chars=-50)
