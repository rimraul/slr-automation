"""Deterministic page/section batching (requirement 6): the complete
parsed document, split into fixed-order, fixed-budget batches for LLM
calls. No whole-document character truncation (every chunk belongs to
exactly one batch; an oversized single chunk still gets its own batch
rather than being cut), no semantic splitter, no vector index, no top-k
retrieval -- batch membership depends only on document order and a fixed
character budget, nothing content- or query-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parsing import ParsedChunk

DEFAULT_MAX_BATCH_CHARS = 12_000


@dataclass(frozen=True)
class ExtractionBatch:
    batch_index: int  # 0-based, strict document order
    chunk_ids: tuple[str, ...]
    total_chars: int


def build_batches(
    chunks: tuple[ParsedChunk, ...], *, max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> tuple[ExtractionBatch, ...]:
    """Every chunk in `chunks` (already in document order, per
    `parsing.py`'s `order_index`) is assigned to exactly one batch.
    Batches are filled greedily in order: a chunk joins the current batch
    unless doing so would exceed `max_batch_chars`, in which case the
    current batch is closed and a new one started. A single chunk longer
    than `max_batch_chars` still becomes its own one-chunk batch -- it is
    never split (chunk boundaries are `parsing.py`'s to decide, not this
    module's) and never dropped/truncated."""
    if max_batch_chars <= 0:
        raise ValueError(f"max_batch_chars must be positive, got {max_batch_chars!r}")
    if not chunks:
        return ()

    batches: list[ExtractionBatch] = []
    current_ids: list[str] = []
    current_chars = 0

    def flush() -> None:
        if current_ids:
            batches.append(
                ExtractionBatch(batch_index=len(batches), chunk_ids=tuple(current_ids), total_chars=current_chars)
            )

    for chunk in chunks:
        chunk_len = len(chunk.text)
        if current_ids and current_chars + chunk_len > max_batch_chars:
            flush()
            current_ids = []
            current_chars = 0
        current_ids.append(chunk.chunk_id)
        current_chars += chunk_len

    flush()
    return tuple(batches)
