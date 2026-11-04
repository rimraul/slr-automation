"""Deterministic run identity (requirement 11). Same discipline as
`slr_pipeline.nodes.fulltext._stable_run_id()` (three review rounds
converged that design on "hash the actual bound values, not a hand-picked
subset") -- reimplemented fresh here, not imported, since this package
does not depend on `slr_pipeline.nodes` (extraction is meant to be usable
standalone, before any graph wiring exists at all -- see this package's
`__init__.py`).

`compute_run_id()` takes PRIMITIVE values (strings, an options dataclass,
a documents list) rather than an `ExtractionSchema`/`OpenAIExtractionConfig`
object -- deliberately, so that a `run_identity.compute_run_id()` call can
be reconstructed purely from a `checkpoint.DocumentCheckpoint`'s own
already-persisted fields, with no access to the original schema/config
objects required (review finding G: `ExtractionCheckpointStore` needs to
PROVE that its checkpoints' own document identities and configuration
reproduce its own `run_id`, using nothing but what a checkpoint already
stores)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .hashing import file_sha256, sha256_hex
from .parsing import DoclingParserOptions


@dataclass(frozen=True)
class DocumentIdentity:
    record_id: str
    pdf_path: str  # normalized (str(Path(...).resolve()))
    pdf_sha256: str


def compute_run_id(
    *,
    documents: list[DocumentIdentity],
    extraction_schema_hash: str,
    parser_name: str,
    parser_version: str,
    parser_options: DoclingParserOptions,
    max_batch_chars: int,
    prompt_template_version: str,
    output_schema_version: str,
    model: str,
    model_config_id: str,
    reasoning_effort: Optional[str],
    max_completion_tokens: int,
    expected_provider_model: str,
    artifacts_manifest_digest: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> str:
    """Binds, per requirement 11: ordered record identities; PDF hashes
    and paths; the extraction-schema hash; parser/chunking identity
    (name, installed version, options -- including `max_chunk_chars`,
    the deterministic-CHUNKING knob); the deterministic-BATCHING knob
    (`max_batch_chars` -- a SEPARATE boundary from `max_chunk_chars`:
    changing which chunks are COMBINED into one model call changes what
    text reaches each call, even though the underlying chunks themselves
    are unchanged; review finding 1: omitting this let two runs with
    different batching plans silently share a run_id and reuse each
    other's checkpoints); prompt/output-schema versions; and the complete
    model configuration (including `expected_provider_model` -- the
    run's own provider-substitution policy, review-required to be part
    of "the complete model configuration" this binds); and
    `artifacts_manifest_digest` (§5.4 -- the pinned Docling model-artifacts
    tree digest; `None` in ordinary/production use, since no artifacts
    directory is pinned there -- two runs differing ONLY in which pinned
    model artifacts they bind must still get different run_ids); and
    `max_batches` (the per-document batch-count cap that decides whether a
    document is attempted or durably blocked -- `None` when the cap is
    disabled; review finding: this cap previously affected which documents
    got blocked WITHOUT affecting `run_id`, so raising or lowering the cap
    reused the SAME run's namespace and silently left previously-blocked
    documents blocked even though the new cap might now admit them; binding
    it here forces a changed cap onto a fresh run_id/checkpoint namespace).
    Order matters
    (list, not sorted) -- the SAME documents processed in a different
    order are a different request, matching FT_SCREENING's own
    request_fingerprint philosophy this project has already converged
    on.

    Every value here is a caller-supplied PRIMITIVE, never re-derived
    internally (e.g. `parser_version` is NOT computed via
    `installed_docling_version()` inside this function) -- the caller is
    responsible for supplying the real, current values. This is what
    makes the function usable BOTH at the moment a run is first started
    (the caller reads the real installed version, real schema hash, etc.)
    AND later, when `checkpoint.ExtractionCheckpointStore` needs to
    recompute this SAME digest purely from a `DocumentCheckpoint`'s own
    already-persisted fields, with no live schema/config objects at
    hand."""
    payload = {
        "documents": [
            {"record_id": d.record_id, "pdf_path": d.pdf_path, "pdf_sha256": d.pdf_sha256} for d in documents
        ],
        "extraction_schema_hash": extraction_schema_hash,
        "parser_name": parser_name,
        "parser_version": parser_version,
        "parser_options": {
            "do_ocr": parser_options.do_ocr,
            "do_table_structure": parser_options.do_table_structure,
            "max_chunk_chars": parser_options.max_chunk_chars,
        },
        "max_batch_chars": max_batch_chars,
        "prompt_template_version": prompt_template_version,
        "output_schema_version": output_schema_version,
        "model_config": {
            "model": model,
            "model_config_id": model_config_id,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
            "expected_provider_model": expected_provider_model,
        },
        "artifacts_manifest_digest": artifacts_manifest_digest,
    }
    if max_batches is not None:
        # Omitted entirely (not even as a `None`-valued key) when unset,
        # so that every run_id computed before this field existed --
        # which always implicitly meant "no cap" -- reproduces BYTE
        # IDENTICALLY here. Unconditionally adding a `"max_batches":
        # None` key would change the hashed payload's shape for every
        # such historical run regardless of the actual cap value,
        # breaking run_id reproducibility for runs that never used this
        # feature at all -- not just capped ones.
        payload["max_batches"] = max_batches
    digest = sha256_hex(payload)
    return f"extrun-{digest[:32]}"


def document_identity_for(record_id: str, pdf_path: str) -> DocumentIdentity:
    from pathlib import Path

    resolved = str(Path(pdf_path).resolve())
    return DocumentIdentity(record_id=record_id, pdf_path=resolved, pdf_sha256=file_sha256(pdf_path))
