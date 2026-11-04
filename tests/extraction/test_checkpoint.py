from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.checkpoint import (
    BlockedDocumentStore, CheckpointIntegrityError, DocumentCheckpoint, ExtractionCheckpointStore, RunManifest,
    TelemetrySummary,
)
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.parsing import DoclingParserOptions
from slr_pipeline.extraction.results import (
    BatchExecutionRecord, DocumentExtractionResult, EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus,
)
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind


def _result(document_id="rec-a") -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_id=document_id,
        fields=(FieldResult(field_id="f", status=FieldStatus.NOT_REPORTED, requires_human_review=False),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
        planned_chunk_ids_by_batch=(("c1",),),
    )


def _schema() -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id="f", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    return ExtractionSchema(schema_id="s", version="1", fields=[field])


def _config() -> OpenAIExtractionConfig:
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )


_PDF_MAP = {"rec-a": ("/a.pdf", "a" * 64), "rec-b": ("/b.pdf", "b" * 64)}


def _real_run_id_for(
    document_ids: tuple[str, ...] = ("rec-a",), *, artifacts_manifest_digest=None, max_batches=None,
) -> str:
    """Review finding G: `ExtractionCheckpointStore._derive_manifest()`
    now proves that a manifest's checkpoints (their own ordered document
    identities and configuration) reproduce the STORE's own `run_id` via
    `compute_run_id()`. Tests that call `publish_manifest()`/
    `load_manifest()` therefore need a store constructed with the REAL
    digest for the documents/config they actually use -- a fixed
    placeholder string like `"extrun-abc"` no longer round-trips."""
    from slr_pipeline.extraction.hashing import sha256_hex
    from slr_pipeline.extraction.parsing import PARSER_NAME, installed_docling_version
    from slr_pipeline.extraction.run_identity import DocumentIdentity
    from slr_pipeline.extraction.run_identity import compute_run_id as _compute_run_id

    identities = [
        DocumentIdentity(record_id=d, pdf_path=_PDF_MAP[d][0], pdf_sha256=_PDF_MAP[d][1]) for d in document_ids
    ]
    config = _config()
    return _compute_run_id(
        documents=identities, extraction_schema_hash=sha256_hex(_schema().model_dump(mode="json")),
        parser_name=PARSER_NAME, parser_version=installed_docling_version(),
        parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        prompt_template_version=config.prompt_template_version, output_schema_version=config.output_schema_version,
        model=config.model, model_config_id=config.model_config_id, reasoning_effort=config.reasoning_effort,
        max_completion_tokens=config.max_completion_tokens, expected_provider_model=config.expected_provider_model,
        artifacts_manifest_digest=artifacts_manifest_digest, max_batches=max_batches,
    )


def _checkpoint(
    document_id="rec-a", run_id="extrun-abc", artifacts_manifest_digest=None, max_batches=None,
) -> DocumentCheckpoint:
    pdf_path, pdf_sha256 = _PDF_MAP.get(document_id, ("/a.pdf", "a" * 64))
    return DocumentCheckpoint.build(
        run_id=run_id, document_id=document_id, pdf_path=pdf_path, pdf_sha256=pdf_sha256,
        schema=_schema(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config(), result=_result(document_id),
        artifacts_manifest_digest=artifacts_manifest_digest, max_batches=max_batches,
    )


def test_store_rejects_path_traversal_run_id(tmp_path):
    """Review finding K's central adversarial reproduction: a run_id
    crafted to escape output_dir via '..' must be rejected at
    construction, before any path is ever built from it."""
    with pytest.raises(ValueError, match="not a safe token"):
        ExtractionCheckpointStore(output_dir=tmp_path, run_id="../../../escaped-run")


def test_store_rejects_run_id_with_backslash_traversal(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        ExtractionCheckpointStore(output_dir=tmp_path, run_id="..\\..\\escaped-run")


def test_store_rejects_absolute_path_run_id(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        ExtractionCheckpointStore(output_dir=tmp_path, run_id="/etc/passwd")


def test_store_rejects_drive_qualified_run_id(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        ExtractionCheckpointStore(output_dir=tmp_path, run_id="C:\\Windows\\System32")


def test_store_rejects_dot_run_id(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        ExtractionCheckpointStore(output_dir=tmp_path, run_id="..")


def test_store_accepts_ordinary_valid_run_id(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-" + ("a" * 32))
    assert store.run_dir.resolve().is_relative_to((tmp_path / "runs").resolve())


def test_checkpoint_round_trip(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    assert not store.has_checkpoint("rec-a")
    store.write_checkpoint(_checkpoint("rec-a"))
    assert store.has_checkpoint("rec-a")
    loaded = store.load_checkpoint("rec-a")
    assert loaded.document_id == "rec-a"
    assert loaded.result.document_id == "rec-a"


def test_checkpoint_namespaced_by_run_id(tmp_path):
    store1 = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-aaa")
    store2 = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-bbb")
    store1.write_checkpoint(_checkpoint("rec-a", run_id="extrun-aaa"))
    assert store1.has_checkpoint("rec-a")
    assert not store2.has_checkpoint("rec-a")  # different run_id -- different checkpoint directory


def test_checkpoint_write_is_atomic_no_leftover_tmp_file(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    store.write_checkpoint(_checkpoint("rec-a"))
    tmp_files = list(store.checkpoints_dir.glob("*.tmp"))
    assert tmp_files == []


def test_checkpoint_filename_is_not_the_raw_document_id():
    """Hash-derived filenames -- the raw document_id never appears in the
    path (review finding 6's own design requirement, checked directly)."""
    store = ExtractionCheckpointStore(output_dir="unused", run_id="extrun-abc")
    path = store.checkpoint_path("rec-a")
    assert "rec-a" not in path.name


def test_checkpoint_path_traversal_document_id_stays_inside_checkpoints_dir(tmp_path):
    """Review finding 6's adversarial reproduction: a document_id crafted
    to look like a path-traversal payload must resolve to an ordinary
    file INSIDE checkpoints_dir, never escape it."""
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    malicious_id = "../../../escape"
    path = store.checkpoint_path(malicious_id)
    assert path.resolve().parent == store.checkpoints_dir.resolve()
    store.write_checkpoint(_checkpoint(malicious_id))
    assert store.load_checkpoint(malicious_id).document_id == malicious_id
    # nothing was written outside checkpoints_dir
    assert not (tmp_path / "escape").exists()


def test_checkpoint_duplicate_document_ids_do_not_collide_via_filename():
    """Two DIFFERENT document_ids must never hash-collide onto the same
    filename in ordinary use -- spot-checked directly since the store has
    no other structural guarantee against a raw hash collision (astronomically
    unlikely, not specifically defended against, but distinct ids here must
    still produce distinct filenames)."""
    store = ExtractionCheckpointStore(output_dir="unused", run_id="extrun-abc")
    assert store.checkpoint_path("rec-a") != store.checkpoint_path("rec-b")


def test_checkpoint_tampered_content_raises_integrity_error(tmp_path):
    """Review finding 2's adversarial reproduction: a checkpoint file
    whose bytes were modified after writing (tampered, corrupted, or
    copied from a different document) must be detected on load, never
    silently trusted just because it parses as valid JSON/Pydantic."""
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    store.write_checkpoint(_checkpoint("rec-a"))
    path = store.checkpoint_path("rec-a")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pdf_sha256"] = "b" * 64  # tamper with a field that is itself part of what content_hash covers
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="content_hash mismatch"):
        store.load_checkpoint("rec-a")


def test_checkpoint_copied_from_a_different_run_id_raises_integrity_error(tmp_path):
    """Review finding A's adversarial reproduction: a self-consistent,
    correctly-hashed checkpoint from `old-run`, copied verbatim into a
    store for `new-run` under the same document_id, must be rejected --
    content-hash verification alone does not catch this (the file IS
    internally consistent); only an explicit run_id cross-check does."""
    old_store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-old")
    old_checkpoint = _checkpoint("rec-a", run_id="extrun-old")
    old_store.write_checkpoint(old_checkpoint)

    new_store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-new")
    new_store.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    # Copy old-run's own (internally self-consistent) checkpoint bytes
    # directly into new-run's checkpoint directory, at the filename
    # new-run would use for the SAME document_id.
    new_store.checkpoint_path("rec-a").write_text(
        json.dumps(old_checkpoint.model_dump(mode="json")), encoding="utf-8",
    )
    with pytest.raises(CheckpointIntegrityError, match="run_id mismatch"):
        new_store.load_checkpoint("rec-a")


def test_checkpoint_copied_from_a_different_document_raises_integrity_error(tmp_path):
    """A checkpoint file for `rec-b`, copied verbatim to the filename
    location for `rec-a`, must be rejected -- its own envelope still
    claims document_id='rec-b', which does not match what was requested."""
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    checkpoint_b = _checkpoint("rec-b")
    # Simulate a copy: write rec-b's own (internally-consistent, correctly
    # hashed) envelope bytes directly at rec-a's filename location.
    store.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    store.checkpoint_path("rec-a").write_text(
        json.dumps(checkpoint_b.model_dump(mode="json")), encoding="utf-8",
    )
    with pytest.raises(CheckpointIntegrityError, match="document_id mismatch"):
        store.load_checkpoint("rec-a")


def _telemetry_summary(**overrides) -> TelemetrySummary:
    defaults = dict(
        total_prompt_tokens=0, total_completion_tokens=0, total_tokens=0,
        distinct_provider_models=(), provider_model_consistent=True, provider_mismatch_count=0,
    )
    defaults.update(overrides)
    return TelemetrySummary(**defaults)


def _manifest(**overrides) -> RunManifest:
    """Direct `RunManifest` construction -- used ONLY to test the model's
    OWN internal-coherence validators (partition/coverage). `manifest_hash`
    is a placeholder (not recomputed/verified at construction time --
    that only happens via `RunManifest.verify_manifest_hash()`, called by
    `ExtractionCheckpointStore.load_manifest()`); these tests never go
    through the store, so the placeholder is irrelevant to what they
    check. `publish_manifest()`-based tests below are what exercise the
    store's own derive-and-verify path against real checkpoints."""
    defaults = dict(
        run_id="extrun-abc", extraction_schema_id="s", extraction_schema_version="1",
        extraction_schema_hash="a" * 64, parser_name="docling", parser_version="2.61.2",
        parser_options={"do_ocr": True, "do_table_structure": True, "max_chunk_chars": 3000},
        max_batch_chars=12_000,
        prompt_template_version="extraction-openai-prompt-1", output_schema_version="extraction-openai-schema-1",
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
        document_ids=("rec-a",), checkpointed_document_ids=("rec-a",),
        complete_document_ids=("rec-a",), incomplete_document_ids=(),
        checkpoint_hashes={"rec-a": "c" * 64},
        telemetry_summary=_telemetry_summary(),
        created_at="2026-08-21T00:00:00+00:00",
        manifest_hash="0" * 64,
    )
    defaults.update(overrides)
    return RunManifest(**defaults)


def test_manifest_rejects_incoherent_coverage_partition():
    with pytest.raises(Exception, match="partition"):
        _manifest(checkpointed_document_ids=("rec-a",), complete_document_ids=(), incomplete_document_ids=())


def test_manifest_rejects_document_in_both_complete_and_incomplete():
    with pytest.raises(Exception, match="both"):
        _manifest(
            checkpointed_document_ids=("rec-a",), complete_document_ids=("rec-a",),
            incomplete_document_ids=("rec-a",), checkpoint_hashes={"rec-a": "c" * 64},
        )


def test_manifest_rejects_hash_coverage_mismatch():
    with pytest.raises(Exception, match="checkpoint_hashes"):
        _manifest(checkpoint_hashes={})


def test_manifest_rejects_checkpointed_ids_not_matching_document_ids():
    """Review finding A: `run_extraction()` always checkpoints every
    requested document before publishing one manifest -- a manifest
    describing fewer (or different) checkpointed documents than were
    actually requested is never a legitimate partial publish, so this
    must be refused structurally, not just by convention."""
    with pytest.raises(Exception, match="checkpointed_document_ids"):
        _manifest(
            document_ids=("rec-a", "rec-b"), checkpointed_document_ids=("rec-a",),
            complete_document_ids=("rec-a",), incomplete_document_ids=(),
            checkpoint_hashes={"rec-a": "c" * 64},
        )


def test_manifest_verify_manifest_hash_detects_tampering():
    manifest = _manifest(manifest_hash="1" * 64)  # deliberately wrong for these actual field values
    with pytest.raises(CheckpointIntegrityError, match="manifest_hash mismatch"):
        manifest.verify_manifest_hash()


def _write_real_checkpoint(store: ExtractionCheckpointStore, document_id: str, run_id: str) -> DocumentCheckpoint:
    checkpoint = _checkpoint(document_id, run_id=run_id)
    store.write_checkpoint(checkpoint)
    return checkpoint


def test_publish_manifest_round_trip(tmp_path):
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    _write_real_checkpoint(store, "rec-a", run_id)
    assert store.load_manifest() is None
    published = store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")
    assert published.run_id == run_id
    assert published.document_ids == ("rec-a",)
    assert published.complete_document_ids == ("rec-a",)

    loaded = store.load_manifest()
    assert loaded is not None
    assert loaded.model_dump(mode="json") == published.model_dump(mode="json")


def test_publish_manifest_derives_configuration_from_checkpoints_never_from_caller(tmp_path):
    """Review finding E's central adversarial reproduction: there is no
    longer any parameter through which a caller could supply a false
    `model`/`model_config_id`/`reasoning_effort`/token budget/expected
    provider model/telemetry -- `publish_manifest()` takes only
    `document_ids`; every other field is READ from the verified
    checkpoints themselves."""
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    checkpoint = _write_real_checkpoint(store, "rec-a", run_id)
    published = store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")
    assert published.model == checkpoint.model
    assert published.model_config_id == checkpoint.model_config_id
    assert published.reasoning_effort == checkpoint.reasoning_effort
    assert published.max_completion_tokens == checkpoint.max_completion_tokens
    assert published.expected_provider_model == checkpoint.expected_provider_model
    assert published.checkpoint_hashes == {"rec-a": checkpoint.content_hash}


def test_publish_manifest_rejects_missing_checkpoint_file(tmp_path):
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    with pytest.raises(FileNotFoundError):
        store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")


def test_publish_manifest_rejects_diverging_configuration_across_checkpoints(tmp_path):
    """A checkpoint for `rec-b` claiming a DIFFERENT model than `rec-a`'s
    checkpoint within the SAME run_id is incoherent -- both were only
    ever supposed to come from the same `compute_run_id()` inputs, so any
    divergence indicates tampering, not a legitimate configuration."""
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    _write_real_checkpoint(store, "rec-a", "extrun-abc")

    diverging_config = OpenAIExtractionConfig(
        model="gpt-5.6-terra-DIFFERENT", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )
    diverging_checkpoint = DocumentCheckpoint.build(
        run_id="extrun-abc", document_id="rec-b", pdf_path="/b.pdf", pdf_sha256="b" * 64,
        schema=_schema(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=diverging_config, result=_result("rec-b"),
    )
    store.write_checkpoint(diverging_checkpoint)

    with pytest.raises(CheckpointIntegrityError, match="different"):
        store.publish_manifest(document_ids=("rec-a", "rec-b"), created_at="2026-08-21T00:00:00+00:00")


def test_checkpoint_build_defaults_artifacts_manifest_digest_to_none():
    """§5.4: production-unchanged default."""
    checkpoint = _checkpoint("rec-a")
    assert checkpoint.artifacts_manifest_digest is None


def test_checkpoint_build_records_artifacts_manifest_digest():
    checkpoint = _checkpoint("rec-a", artifacts_manifest_digest="a" * 64)
    assert checkpoint.artifacts_manifest_digest == "a" * 64


def test_publish_manifest_derives_artifacts_manifest_digest_from_checkpoint(tmp_path):
    digest = "a" * 64
    run_id = _real_run_id_for(("rec-a",), artifacts_manifest_digest=digest)
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    store.write_checkpoint(_checkpoint("rec-a", run_id=run_id, artifacts_manifest_digest=digest))
    published = store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")
    assert published.artifacts_manifest_digest == digest


def test_publish_manifest_rejects_diverging_artifacts_manifest_digest_across_checkpoints(tmp_path):
    """Mirrors `test_publish_manifest_rejects_diverging_configuration_
    across_checkpoints`: two checkpoints within the SAME run_id claiming
    DIFFERENT `artifacts_manifest_digest` values is incoherent -- both
    were only ever supposed to come from the same `compute_run_id()`
    inputs."""
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id="extrun-abc")
    store.write_checkpoint(_checkpoint("rec-a", run_id="extrun-abc", artifacts_manifest_digest="a" * 64))
    store.write_checkpoint(_checkpoint("rec-b", run_id="extrun-abc", artifacts_manifest_digest="b" * 64))

    with pytest.raises(CheckpointIntegrityError, match="artifacts_manifest_digest"):
        store.publish_manifest(document_ids=("rec-a", "rec-b"), created_at="2026-08-21T00:00:00+00:00")


def test_publish_manifest_identical_republish_is_a_noop_preserving_created_at(tmp_path):
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    _write_real_checkpoint(store, "rec-a", run_id)
    first = store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")
    second = store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T09:30:00+00:00")
    assert second.created_at == first.created_at  # NOT the new created_at passed in -- a true no-op
    assert second.manifest_hash == first.manifest_hash
    assert second.model_dump(mode="json") == first.model_dump(mode="json")


def test_publish_manifest_refuses_republish_after_checkpoint_replaced(tmp_path):
    """Review refinement: a published manifest is immutable. Even a
    REAL, self-consistent, on-disk-matching checkpoint replacement (here
    simulating the document being re-extracted with a different result
    after the manifest was already published) must not be allowed to
    silently update the already-published manifest."""
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    _write_real_checkpoint(store, "rec-a", run_id)
    store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")

    different_result = DocumentExtractionResult(
        document_id="rec-a",
        fields=(FieldResult(
            field_id="f", status=FieldStatus.EXTRACTED,
            values=(ExtractedValue(
                value="x", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="x", section="S"),),
            ),),
            requires_human_review=False,
        ),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
        planned_chunk_ids_by_batch=(("c1",),),
    )
    checkpoint_v2 = DocumentCheckpoint.build(
        run_id=run_id, document_id="rec-a", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config(), result=different_result,
    )
    store.write_checkpoint(checkpoint_v2)

    # The already-published manifest is now stale relative to the
    # REPLACED checkpoint -- publish_manifest() re-verifies the existing
    # manifest against current checkpoints first, and refuses rather
    # than silently basing a "no-op vs. overwrite" decision on data that
    # no longer matches reality.
    with pytest.raises(CheckpointIntegrityError, match="no longer matches its own checkpoints"):
        store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T01:00:00+00:00")


def test_load_manifest_detects_post_publication_tampering(tmp_path):
    """Review finding E's adversarial reproduction: a published manifest
    file edited on disk (e.g. its `model` field changed post-publication)
    must be rejected on reload, never silently accepted just because it
    still parses as valid JSON/Pydantic."""
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    _write_real_checkpoint(store, "rec-a", run_id)
    store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")

    path = store.manifest_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"] = "gpt-5.6-terra-TAMPERED"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="manifest_hash mismatch"):
        store.load_manifest()


def test_load_manifest_detects_checkpoint_replaced_after_publication(tmp_path):
    """Review finding G's explicitly-required regression: after a
    manifest is published, replacing its checkpoint with a DIFFERENT
    valid, self-hashed checkpoint for the same run/document must make
    `load_manifest()` (not just `publish_manifest()`) refuse to return
    the now-stale manifest -- its own self-hash is still internally
    valid, so only re-derivation against the CURRENT checkpoint catches
    this."""
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    _write_real_checkpoint(store, "rec-a", run_id)
    store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")

    different_result = DocumentExtractionResult(
        document_id="rec-a",
        fields=(FieldResult(
            field_id="f", status=FieldStatus.EXTRACTED,
            values=(ExtractedValue(
                value="x", evidence=(EvidenceSpanModel(chunk_id="c1", excerpt="x", section="S"),),
            ),),
            requires_human_review=False,
        ),),
        batch_executions=(BatchExecutionRecord(batch_index=0, chunk_ids=("c1",), status="completed"),),
        planned_chunk_ids_by_batch=(("c1",),),
    )
    checkpoint_v2 = DocumentCheckpoint.build(
        run_id=run_id, document_id="rec-a", pdf_path="/a.pdf", pdf_sha256="a" * 64,
        schema=_schema(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config(), result=different_result,
    )
    store.write_checkpoint(checkpoint_v2)

    with pytest.raises(CheckpointIntegrityError, match="no longer matches"):
        store.load_manifest()


def test_load_manifest_rejects_manifest_and_checkpoint_copied_from_a_different_run(tmp_path):
    """Review finding G's explicitly-required regression: a fully
    self-consistent manifest AND its checkpoint, both copied verbatim
    from `old-run`'s directory into a store for `new-run`, must not load
    successfully -- the copied checkpoint's own claimed `run_id` still
    says `old-run`, which `load_checkpoint()` (called from within
    `_derive_manifest()`) catches even though every individual file is,
    in isolation, internally self-consistent."""
    old_run_id = _real_run_id_for(("rec-a",))
    old_store = ExtractionCheckpointStore(output_dir=tmp_path / "old", run_id=old_run_id)
    _write_real_checkpoint(old_store, "rec-a", old_run_id)
    old_store.publish_manifest(document_ids=("rec-a",), created_at="2026-08-21T00:00:00+00:00")

    new_run_id = "extrun-" + ("f" * 32)  # deliberately a DIFFERENT (fake) run_id
    new_store = ExtractionCheckpointStore(output_dir=tmp_path / "new", run_id=new_run_id)
    new_store.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    old_checkpoint = old_store.load_checkpoint("rec-a")
    new_store.checkpoint_path("rec-a").write_text(
        json.dumps(old_checkpoint.model_dump(mode="json")), encoding="utf-8",
    )
    old_manifest_bytes = old_store.manifest_path().read_text(encoding="utf-8")
    new_store.manifest_path().parent.mkdir(parents=True, exist_ok=True)
    new_store.manifest_path().write_text(old_manifest_bytes, encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="run_id mismatch"):
        new_store.load_manifest()


def test_publish_manifest_rejects_duplicate_document_ids(tmp_path):
    """Review finding G's explicitly-required regression: a set-based
    coverage check alone lets a duplicate document_id collapse invisibly
    (two identical entries have the same set membership as one) --
    `_derive_manifest()` checks length, not just set equality, so this
    must be refused explicitly rather than silently accepted."""
    run_id = _real_run_id_for(("rec-a",))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    _write_real_checkpoint(store, "rec-a", run_id)
    with pytest.raises(CheckpointIntegrityError, match="duplicate"):
        store.publish_manifest(document_ids=("rec-a", "rec-a"), created_at="2026-08-21T00:00:00+00:00")


def test_publish_manifest_rejects_reversed_document_order(tmp_path):
    """Review finding G's explicitly-required regression:
    `compute_run_id()` explicitly binds document ORDER -- publishing a
    manifest with the SAME two documents in a DIFFERENT order than the
    store's own `run_id` was computed for must be refused, since the
    reversed order reproduces a DIFFERENT digest than `self.run_id`."""
    forward_run_id = _real_run_id_for(("rec-a", "rec-b"))
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=forward_run_id)
    _write_real_checkpoint(store, "rec-a", forward_run_id)
    checkpoint_b = DocumentCheckpoint.build(
        run_id=forward_run_id, document_id="rec-b", pdf_path="/b.pdf", pdf_sha256="b" * 64,
        schema=_schema(), parser_options=DoclingParserOptions(), max_batch_chars=12_000,
        model_config=_config(), result=_result("rec-b"),
    )
    store.write_checkpoint(checkpoint_b)

    # The forward order is the one this store's run_id actually matches.
    store.publish_manifest(document_ids=("rec-a", "rec-b"), created_at="2026-08-21T00:00:00+00:00")

    # The SAME two checkpoints, requested in REVERSED order, must be
    # refused: compute_run_id(["rec-b", "rec-a"]) != forward_run_id,
    # since document order is explicitly part of the digest.
    with pytest.raises(CheckpointIntegrityError, match="does not match this store"):
        store._derive_manifest(document_ids=("rec-b", "rec-a"), created_at="2026-08-21T00:00:00+00:00")


def test_publish_manifest_mixed_attempted_and_blocked_documents(tmp_path):
    """Review finding 1's explicitly-required regression: a run where one
    requested document is checkpointed (attempted) and another is durably
    blocked (batch-cap exceeded) must still publish a valid manifest --
    `run_id` is bound to BOTH documents via `max_batches`, but only the
    attempted one is checkpointed."""
    max_batches = 8
    run_id = _real_run_id_for(("rec-a", "rec-b"), max_batches=max_batches)
    store = ExtractionCheckpointStore(output_dir=tmp_path, run_id=run_id)
    store.write_checkpoint(_checkpoint("rec-a", run_id=run_id, max_batches=max_batches))
    store._blocked_store().record_blocked(
        document_id="rec-b", pdf_path=_PDF_MAP["rec-b"][0], pdf_sha256=_PDF_MAP["rec-b"][1],
        reason="9 batches exceeds cap of 8", batch_count=9, max_batches=max_batches,
        parsed_document_sha256="c" * 64, blocked_at="2026-09-22T00:00:00+00:00",
    )

    manifest = store.publish_manifest(
        document_ids=("rec-a",), requested_document_ids=("rec-a", "rec-b"),
        created_at="2026-09-22T00:00:00+00:00",
    )

    assert manifest.document_ids == ("rec-a",)
    assert manifest.checkpointed_document_ids == ("rec-a",)
    assert manifest.blocked_document_ids == ("rec-b",)
    assert manifest.requested_document_ids == ("rec-a", "rec-b")

    reloaded = store.load_manifest()
    assert reloaded.blocked_document_ids == ("rec-b",)


# ---------------------------------------------------------------------------
# BlockedDocumentStore
# ---------------------------------------------------------------------------

def test_blocked_store_rejects_path_traversal_run_id(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        BlockedDocumentStore(output_dir=tmp_path, run_id="../../../escaped-run")


def test_blocked_store_accepts_ordinary_run_id_and_starts_empty(tmp_path):
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    assert not store.is_blocked("rec-a")
    assert store.blocked_document_ids() == ()


def test_blocked_store_record_and_query_round_trip(tmp_path):
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    store.record_blocked(
        document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
        reason="9 batches exceeds cap of 8", batch_count=9, max_batches=8,
        parsed_document_sha256="a" * 64, blocked_at="2026-09-21T00:00:00+00:00",
    )
    assert store.is_blocked("rec-a")
    assert not store.is_blocked("rec-b")
    assert store.blocked_document_ids() == ("rec-a",)


def test_blocked_store_filename_is_not_the_raw_document_id(tmp_path):
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    path = store.blocked_path("rec-a")
    assert "rec-a" not in path.name


def test_blocked_store_re_recording_identical_content_is_a_noop(tmp_path):
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    kwargs = dict(
        document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
        reason="9 batches exceeds cap of 8", batch_count=9, max_batches=8,
        parsed_document_sha256="a" * 64,
    )
    store.record_blocked(blocked_at="2026-09-21T00:00:00+00:00", **kwargs)
    # Re-recording with different blocked_at (only) is still a no-op --
    # blocked_at is excluded from the equality check by design.
    store.record_blocked(blocked_at="2026-09-21T01:00:00+00:00", **kwargs)
    assert store.blocked_document_ids() == ("rec-a",)


def test_blocked_store_re_recording_different_content_raises(tmp_path):
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    store.record_blocked(
        document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
        reason="9 batches exceeds cap of 8", batch_count=9, max_batches=8,
        parsed_document_sha256="a" * 64, blocked_at="2026-09-21T00:00:00+00:00",
    )
    with pytest.raises(CheckpointIntegrityError, match="different content"):
        store.record_blocked(
            document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
            reason="10 batches exceeds cap of 8", batch_count=10, max_batches=8,
            parsed_document_sha256="a" * 64, blocked_at="2026-09-21T02:00:00+00:00",
        )


def test_blocked_store_namespaced_by_run_id(tmp_path):
    store1 = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-aaa")
    store2 = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-bbb")
    store1.record_blocked(
        document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
        reason="over cap", batch_count=9, max_batches=8,
        parsed_document_sha256="a" * 64, blocked_at="2026-09-21T00:00:00+00:00",
    )
    assert store1.is_blocked("rec-a")
    assert not store2.is_blocked("rec-a")


def test_blocked_store_is_blocked_rejects_tampered_content(tmp_path):
    """Review finding 3: `is_blocked()` must not trust file existence
    alone -- a blocked-record file whose content_hash no longer matches
    its own body (tampered or corrupted on disk) must raise, not be
    silently treated as a valid block."""
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    store.record_blocked(
        document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
        reason="over cap", batch_count=9, max_batches=8,
        parsed_document_sha256="a" * 64, blocked_at="2026-09-21T00:00:00+00:00",
    )
    path = store.blocked_path("rec-a")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["reason"] = "tampered reason, content_hash now stale"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError):
        store.is_blocked("rec-a")


def test_blocked_store_is_blocked_rejects_wrong_claimed_document_id(tmp_path):
    """Review finding 3: a blocked-record file placed under a different
    document_id's filename than the one it claims internally must be
    refused, not silently accepted as that document's block."""
    store = BlockedDocumentStore(output_dir=tmp_path, run_id="extrun-abc")
    store.record_blocked(
        document_id="rec-a", pdf_path="/docs/rec-a.pdf", pdf_sha256="a" * 64,
        reason="over cap", batch_count=9, max_batches=8,
        parsed_document_sha256="a" * 64, blocked_at="2026-09-21T00:00:00+00:00",
    )
    content = store.blocked_path("rec-a").read_text(encoding="utf-8")
    store.blocked_path("rec-b").write_text(content, encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError):
        store.is_blocked("rec-b")
