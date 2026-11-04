from __future__ import annotations

import sys
from pathlib import Path

from slr_pipeline.extraction.parsing import DoclingParserOptions
from slr_pipeline.extraction.run_identity import DocumentIdentity, compute_run_id


def _identity(record_id="rec-a", pdf_path="/a.pdf", pdf_sha256="a" * 64) -> DocumentIdentity:
    return DocumentIdentity(record_id=record_id, pdf_path=pdf_path, pdf_sha256=pdf_sha256)


def _kwargs(**overrides) -> dict:
    """`compute_run_id()` takes PRIMITIVE values only (review finding G:
    this is what lets `ExtractionCheckpointStore` recompute the SAME
    digest purely from a `DocumentCheckpoint`'s own persisted fields,
    with no schema/config objects at hand) -- these tests exercise it
    directly with fixed primitive values rather than real
    ExtractionSchema/OpenAIExtractionConfig objects."""
    defaults = dict(
        documents=[_identity()],
        extraction_schema_hash="a" * 64,
        parser_name="docling", parser_version="2.61.2", parser_options=DoclingParserOptions(),
        max_batch_chars=12_000,
        prompt_template_version="extraction-openai-prompt-1", output_schema_version="extraction-openai-schema-1",
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    defaults.update(overrides)
    return defaults


def test_identical_inputs_same_run_id():
    id1 = compute_run_id(**_kwargs())
    id2 = compute_run_id(**_kwargs())
    assert id1 == id2


def test_different_schema_hash_different_run_id():
    id1 = compute_run_id(**_kwargs(extraction_schema_hash="a" * 64))
    id2 = compute_run_id(**_kwargs(extraction_schema_hash="b" * 64))
    assert id1 != id2


def test_different_parser_name_different_run_id():
    id1 = compute_run_id(**_kwargs(parser_name="docling"))
    id2 = compute_run_id(**_kwargs(parser_name="other-parser"))
    assert id1 != id2


def test_different_parser_version_different_run_id():
    id1 = compute_run_id(**_kwargs(parser_version="2.61.2"))
    id2 = compute_run_id(**_kwargs(parser_version="2.62.0"))
    assert id1 != id2


def test_different_reasoning_effort_different_run_id():
    id1 = compute_run_id(**_kwargs(reasoning_effort="medium"))
    id2 = compute_run_id(**_kwargs(reasoning_effort="high"))
    assert id1 != id2


def test_different_max_completion_tokens_different_run_id():
    id1 = compute_run_id(**_kwargs(max_completion_tokens=4096))
    id2 = compute_run_id(**_kwargs(max_completion_tokens=8000))
    assert id1 != id2


def test_different_expected_provider_model_different_run_id():
    id1 = compute_run_id(**_kwargs(expected_provider_model="gpt-5.6-terra"))
    id2 = compute_run_id(**_kwargs(expected_provider_model="gpt-5.6-terra-2026-08-01"))
    assert id1 != id2


def test_different_prompt_template_version_different_run_id():
    id1 = compute_run_id(**_kwargs(prompt_template_version="prompt-1"))
    id2 = compute_run_id(**_kwargs(prompt_template_version="prompt-2"))
    assert id1 != id2


def test_different_output_schema_version_different_run_id():
    id1 = compute_run_id(**_kwargs(output_schema_version="schema-1"))
    id2 = compute_run_id(**_kwargs(output_schema_version="schema-2"))
    assert id1 != id2


def test_different_model_different_run_id():
    id1 = compute_run_id(**_kwargs(model="gpt-5.6-terra"))
    id2 = compute_run_id(**_kwargs(model="gpt-5.7-terra"))
    assert id1 != id2


def test_different_model_config_id_different_run_id():
    id1 = compute_run_id(**_kwargs(model_config_id="cfg-1"))
    id2 = compute_run_id(**_kwargs(model_config_id="cfg-2"))
    assert id1 != id2


def test_different_parser_options_different_run_id():
    id1 = compute_run_id(**_kwargs(parser_options=DoclingParserOptions(do_ocr=True)))
    id2 = compute_run_id(**_kwargs(parser_options=DoclingParserOptions(do_ocr=False)))
    assert id1 != id2


def test_different_max_chunk_chars_different_run_id():
    id1 = compute_run_id(**_kwargs(parser_options=DoclingParserOptions(max_chunk_chars=3000)))
    id2 = compute_run_id(**_kwargs(parser_options=DoclingParserOptions(max_chunk_chars=6000)))
    assert id1 != id2


def test_different_max_batch_chars_different_run_id():
    """Review finding 1: `max_batch_chars` determines which chunks are
    COMBINED into one model call -- a separate boundary from
    `max_chunk_chars` (which only affects how the document is initially
    split into chunks). Two runs differing only in `max_batch_chars` must
    get different run_ids so one can never resume from the other's
    (batching-incompatible) checkpoints."""
    id1 = compute_run_id(**_kwargs(max_batch_chars=6_000))
    id2 = compute_run_id(**_kwargs(max_batch_chars=12_000))
    assert id1 != id2


def test_different_pdf_hash_different_run_id():
    id1 = compute_run_id(**_kwargs(documents=[_identity(pdf_sha256="a" * 64)]))
    id2 = compute_run_id(**_kwargs(documents=[_identity(pdf_sha256="b" * 64)]))
    assert id1 != id2


def test_different_pdf_path_same_bytes_different_run_id():
    """Provisional-manifest binding: same content hash, different path,
    must still be a different run identity."""
    id1 = compute_run_id(**_kwargs(documents=[_identity(pdf_path="/location_a/a.pdf")]))
    id2 = compute_run_id(**_kwargs(documents=[_identity(pdf_path="/location_b/a.pdf")]))
    assert id1 != id2


def test_different_document_order_different_run_id():
    a, b = _identity(record_id="a"), _identity(record_id="b", pdf_path="/b.pdf", pdf_sha256="b" * 64)
    id1 = compute_run_id(**_kwargs(documents=[a, b]))
    id2 = compute_run_id(**_kwargs(documents=[b, a]))
    assert id1 != id2


def test_growing_batch_different_run_id():
    a = _identity(record_id="a")
    b = _identity(record_id="b", pdf_path="/b.pdf", pdf_sha256="b" * 64)
    id_a_only = compute_run_id(**_kwargs(documents=[a]))
    id_a_and_b = compute_run_id(**_kwargs(documents=[a, b]))
    assert id_a_only != id_a_and_b


def test_artifacts_manifest_digest_defaults_to_none_and_matches_omitting_it():
    """§5.4: omitting `artifacts_manifest_digest` and passing it as
    explicit `None` must be indistinguishable."""
    id1 = compute_run_id(**_kwargs())
    id2 = compute_run_id(**_kwargs(artifacts_manifest_digest=None))
    assert id1 == id2


def test_different_artifacts_manifest_digest_different_run_id():
    id1 = compute_run_id(**_kwargs(artifacts_manifest_digest="a" * 64))
    id2 = compute_run_id(**_kwargs(artifacts_manifest_digest="b" * 64))
    assert id1 != id2


def test_none_vs_set_artifacts_manifest_digest_different_run_id():
    id1 = compute_run_id(**_kwargs(artifacts_manifest_digest=None))
    id2 = compute_run_id(**_kwargs(artifacts_manifest_digest="a" * 64))
    assert id1 != id2
