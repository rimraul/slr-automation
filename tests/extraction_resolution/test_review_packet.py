"""Offline tests for render_review_packet() (amber-quilting-osprey.md sec.
3.2/3.3). Two fixtures, per the plan's own phase-1 test contract:

  - a synthetic, hand-built two-field bundle covering all four real
    evidence sources (values, provisional_values, invalid_values,
    review_evidence) and all three real candidate sources (values,
    provisional_values, invalid_values), including the confirmed
    real-world case (sec. 3.2) of two evidence items sharing one
    chunk_id with two different excerpts;
  - the real Berg2023 charting-calibration fixture at
    test_reviews/berg2023/charting_calibration/runs/extrun-069b11531b6b9c7cf2b4b7547b555b11/
    (migrated 2026-09-17 from extrun-014f1111817811f736e0775da8719363 to the
    current run-identity algorithm; see calibration_report.json's
    migration_provenance).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import slr_pipeline
from slr_pipeline.adapters.scoping_charting_adapter import scoping_chart_schema
from slr_pipeline.extraction.checkpoint import (
    PARSER_NAME, DocumentCheckpoint, ExtractionCheckpointStore, VerifiedRunBundle, installed_docling_version,
)
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.parsing import DoclingParserOptions
from slr_pipeline.extraction.resolution_schemas import QueueSnapshot
from slr_pipeline.extraction.run_identity import DocumentIdentity, compute_run_id
from slr_pipeline.extraction.resolution_suggestions import render_review_packet
from slr_pipeline.extraction.results import (
    BatchExecutionRecord, DocumentExtractionResult, EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus,
    InvalidValueObservation,
)
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind
from slr_pipeline.extraction.verified_bundle import bundle_review_fields

REPO_ROOT = Path(slr_pipeline.__file__).resolve().parents[1]
BERG_OUTPUT_DIR = REPO_ROOT / "test_reviews" / "berg2023" / "charting_calibration"
BERG_RUN_ID = "extrun-069b11531b6b9c7cf2b4b7547b555b11"

TRACK3_CONTINUATION_OUTPUT_DIR = REPO_ROOT / "test_reviews" / "track3_continuation_six_field_fixture"
TRACK3_CONTINUATION_RUN_ID = "extrun-dbaf4a05e6f8ce0965d24a234f593a09"


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def _field_spec(field_id: str) -> ExtractionFieldSpec:
    return ExtractionFieldSpec(
        field_id=field_id, question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )


def _schema() -> ExtractionSchema:
    return ExtractionSchema(schema_id="s", version="1", fields=[_field_spec("population"), _field_spec("limitations")])


def _model_config() -> OpenAIExtractionConfig:
    return OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )


def _span(chunk_id: str, excerpt: str, page: str = "4", section: str = "Results") -> EvidenceSpanModel:
    return EvidenceSpanModel(chunk_id=chunk_id, excerpt=excerpt, page=page, section=section)


def _population_field() -> FieldResult:
    # Two `values` candidates, one of them citing two evidence items that
    # SHARE a chunk_id with two different excerpts -- the confirmed
    # real-world case sec. 3.2 requires a chunk_id-alone key not to collapse.
    span_shared_1 = _span("chunk-0019", "adults aged 18-65")
    span_shared_2 = _span("chunk-0019", "some children were also enrolled")
    span_review = _span("chunk-0020", "unclear whether adolescents counted separately")
    span_invalid = _span("chunk-0021", "N/A")
    return FieldResult(
        field_id="population", status=FieldStatus.CONFLICTING, requires_human_review=True,
        values=(
            ExtractedValue(value="Adults", evidence=(span_shared_1,)),
            ExtractedValue(value="Children", evidence=(span_shared_2,)),
        ),
        review_evidence=(span_review,),
        invalid_values=(
            InvalidValueObservation(raw_value="N/A", value_kind="text", reason="ambiguous", evidence=(span_invalid,)),
        ),
        invalid_value_count=1,
    )


def _limitations_field() -> FieldResult:
    span_provisional = _span("chunk-0030", "small sample size noted", section="Discussion")
    return FieldResult(
        field_id="limitations", status=FieldStatus.EXTRACTION_FAILED, requires_human_review=True,
        provisional_values=(ExtractedValue(value="Small sample", evidence=(span_provisional,)),),
    )


def _build_bundle(tmp_path) -> tuple:
    schema = _schema()
    model_config = _model_config()
    parser_options = DoclingParserOptions()
    result = DocumentExtractionResult(
        document_id="rec-1", fields=(_population_field(), _limitations_field()),
        batch_executions=(
            BatchExecutionRecord(
                batch_index=0, chunk_ids=("chunk-0019", "chunk-0020", "chunk-0021", "chunk-0030"),
                status="completed", provider_model="gpt-5.6-terra",
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            ),
        ),
        planned_chunk_ids_by_batch=(("chunk-0019", "chunk-0020", "chunk-0021", "chunk-0030"),),
    )
    run_id = compute_run_id(
        documents=[DocumentIdentity(record_id="rec-1", pdf_path="rec-1.pdf", pdf_sha256="a" * 64)],
        extraction_schema_hash=sha256_hex(schema.model_dump(mode="json")),
        parser_name=PARSER_NAME, parser_version=installed_docling_version(), parser_options=parser_options,
        max_batch_chars=12000, prompt_template_version=model_config.prompt_template_version,
        output_schema_version=model_config.output_schema_version, model=model_config.model,
        model_config_id=model_config.model_config_id, reasoning_effort=model_config.reasoning_effort,
        max_completion_tokens=model_config.max_completion_tokens,
        expected_provider_model=model_config.expected_provider_model,
    )
    checkpoint = DocumentCheckpoint.build(
        run_id=run_id, document_id="rec-1", pdf_path="rec-1.pdf", pdf_sha256="a" * 64, schema=schema,
        parser_options=parser_options, max_batch_chars=12000, model_config=model_config, result=result,
    )
    store = ExtractionCheckpointStore(output_dir=tmp_path / "out", run_id=run_id)
    store.write_checkpoint(checkpoint)
    store.publish_manifest(document_ids=("rec-1",), created_at="2026-09-17T00:00:00+00:00")
    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()
    return bundle, schema


def _snapshot_from_bundle(bundle, schema, *, resolved_keys=(), resolution_history_as_of_render=None) -> QueueSnapshot:
    """Builds a snapshot that exactly covers `bundle_review_fields()`'s own
    keys: every key in `resolved_keys` is moved out of the queue and into
    a synthesized resolution-history entry (using the real, recomputed
    original_status/original_field_result_hash), every other review-
    required key stays queued -- satisfying render_review_packet()'s
    queue-union-history-equals-verified-keys / queue-history-disjoint
    completeness contract by construction. `resolution_history_as_of_render`,
    if given explicitly, overrides this derivation entirely (used by tests
    that need a malformed/stray history entry instead)."""
    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    verified_fields = bundle_review_fields(bundle.results)
    resolved_keys_set = set(resolved_keys)
    queue = [
        {"record_id": rid, "field_id": fid, "status": field_result.status.value, "field_result": field_result.model_dump(mode="json")}
        for (rid, fid), field_result in verified_fields.items()
        if (rid, fid) not in resolved_keys_set
    ]
    if resolution_history_as_of_render is None:
        resolution_history_as_of_render = [
            {
                "record_id": rid, "field_id": fid,
                "original_status": verified_fields[(rid, fid)].status.value,
                "original_field_result_hash": sha256_hex(verified_fields[(rid, fid)].model_dump(mode="json")),
                "outcome": "not_reported", "reviewer": "bob", "note": "n/a",
                "resolved_at": "2026-09-01T00:00:00+00:00", "run_id": bundle.run_id,
                "manifest_hash": bundle.manifest.manifest_hash, "extraction_schema_hash": schema_hash,
            }
            for (rid, fid) in resolved_keys
        ]
    return QueueSnapshot(
        run_id=bundle.run_id, manifest_hash=bundle.manifest.manifest_hash, extraction_schema_hash=schema_hash,
        checkpoint_hashes=dict(bundle.manifest.checkpoint_hashes), queue=queue,
        resolution_history_as_of_render=list(resolution_history_as_of_render),
    )


def test_render_review_packet_covers_both_queued_fields(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    assert {e.field_id for e in packet.entries} == {"population", "limitations"}
    assert packet.run_id == bundle.run_id
    assert packet.manifest_hash == bundle.manifest.manifest_hash


def test_render_review_packet_evidence_index_does_not_collapse_shared_chunk_id(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    population_entry = next(e for e in packet.entries if e.field_id == "population")
    excerpts = {item.excerpt for item in population_entry.evidence_index.values()}
    assert "adults aged 18-65" in excerpts
    assert "some children were also enrolled" in excerpts
    chunk_0019_items = [item for item in population_entry.evidence_index.values() if item.chunk_id == "chunk-0019"]
    assert len(chunk_0019_items) == 2
    assert len({item.evidence_id for item in chunk_0019_items}) == 2


def test_render_review_packet_evidence_index_covers_all_four_sources(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    population_entry = next(e for e in packet.entries if e.field_id == "population")
    excerpts = {item.excerpt for item in population_entry.evidence_index.values()}
    assert "unclear whether adolescents counted separately" in excerpts  # review_evidence
    assert "N/A" in excerpts  # invalid_values
    limitations_entry = next(e for e in packet.entries if e.field_id == "limitations")
    limitations_excerpts = {item.excerpt for item in limitations_entry.evidence_index.values()}
    assert "small sample size noted" in limitations_excerpts  # provisional_values


def test_render_review_packet_candidates_cover_all_three_sources(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    population_entry = next(e for e in packet.entries if e.field_id == "population")
    by_category = {c.source_category: c for c in population_entry.candidates}
    assert by_category["value"].value in {"Adults", "Children"}
    assert by_category["invalid_value"].value == "N/A"
    assert by_category["invalid_value"].reason == "ambiguous"
    limitations_entry = next(e for e in packet.entries if e.field_id == "limitations")
    assert limitations_entry.candidates[0].source_category == "provisional_value"
    assert limitations_entry.candidates[0].value == "Small sample"


def test_render_review_packet_candidate_ids_reference_real_evidence_ids(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    population_entry = next(e for e in packet.entries if e.field_id == "population")
    for candidate in population_entry.candidates:
        for eid in candidate.evidence_ids:
            assert eid in population_entry.evidence_index


def test_render_review_packet_is_pure_and_deterministic(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet_a = render_review_packet(bundle, schema, snapshot)
    packet_b = render_review_packet(bundle, schema, snapshot)
    assert packet_a == packet_b


def test_render_review_packet_rejects_snapshot_bound_to_different_run(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    tampered = snapshot.model_copy(update={"run_id": "extrun-other"})
    with pytest.raises(ValueError, match="run_id"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_rejects_snapshot_bound_to_different_manifest_hash(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    tampered = snapshot.model_copy(update={"manifest_hash": "0" * 64})
    with pytest.raises(ValueError, match="manifest_hash"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_rejects_snapshot_bound_to_different_schema_hash(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    tampered = snapshot.model_copy(update={"extraction_schema_hash": "0" * 64})
    with pytest.raises(ValueError, match="extraction_schema_hash"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_rejects_snapshot_bound_to_different_checkpoint_hashes(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    tampered = snapshot.model_copy(update={"checkpoint_hashes": {"rec-1": "0" * 64}})
    with pytest.raises(ValueError, match="checkpoint_hashes"):
        render_review_packet(bundle, schema, tampered)


def _replace_queue(snapshot: QueueSnapshot, queue: list) -> QueueSnapshot:
    """Rebuilds a `QueueSnapshot` through real validation (mirroring how a
    snapshot loaded from disk is always constructed) instead of
    `model_copy(update=...)`, which performs no re-validation and would
    let a raw dict silently masquerade as a `QueueSnapshotEntry`."""
    data = snapshot.model_dump(mode="json")
    data["queue"] = queue
    return QueueSnapshot.model_validate(data)


def test_render_review_packet_rejects_tampered_queue_field_result(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    tampered_queue = [e.model_dump(mode="json") for e in snapshot.queue]
    tampered_queue[0] = dict(tampered_queue[0], status="extracted")
    tampered = _replace_queue(snapshot, tampered_queue)
    with pytest.raises(ValueError, match="tampered"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_rejects_duplicate_queue_entries(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    queue = [e.model_dump(mode="json") for e in snapshot.queue]
    tampered = _replace_queue(snapshot, queue + [queue[0]])
    with pytest.raises(ValueError, match="duplicate"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_rejects_queue_key_not_requiring_review(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    bogus_entry = {
        "record_id": "rec-1", "field_id": "nonexistent_field", "status": "unclear",
        "field_result": {"field_id": "nonexistent_field", "status": "unclear", "requires_human_review": True},
    }
    queue = [e.model_dump(mode="json") for e in snapshot.queue]
    tampered = _replace_queue(snapshot, queue + [bogus_entry])
    with pytest.raises(ValueError, match="nonexistent_field"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_rejects_resolution_history_entry_from_a_different_run(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    stray = {
        "record_id": "rec-1", "field_id": "population", "original_status": "conflicting",
        "original_field_result_hash": "a" * 24, "outcome": "not_reported", "reviewer": "bob",
        "note": "n/a", "resolved_at": "2026-09-01T00:00:00+00:00", "run_id": "extrun-other",
        "manifest_hash": bundle.manifest.manifest_hash, "extraction_schema_hash": sha256_hex(schema.model_dump(mode="json")),
    }
    snapshot = _snapshot_from_bundle(bundle, schema, resolution_history_as_of_render=[stray])
    with pytest.raises(ValueError, match="run_id"):
        render_review_packet(bundle, schema, snapshot)


def test_render_review_packet_rejects_resolution_history_entry_for_a_field_not_requiring_review(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    stray = {
        "record_id": "rec-1", "field_id": "concept", "original_status": "unclear",
        "original_field_result_hash": "a" * 24, "outcome": "not_reported", "reviewer": "bob",
        "note": "n/a", "resolved_at": "2026-09-01T00:00:00+00:00", "run_id": bundle.run_id,
        "manifest_hash": bundle.manifest.manifest_hash, "extraction_schema_hash": schema_hash,
    }
    snapshot = _snapshot_from_bundle(bundle, schema, resolution_history_as_of_render=[stray])
    with pytest.raises(ValueError, match="concept"):
        render_review_packet(bundle, schema, snapshot)


def test_render_review_packet_rejects_resolution_history_entry_with_wrong_original_field_result_hash(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    stray = {
        "record_id": "rec-1", "field_id": "population", "original_status": "conflicting",
        "original_field_result_hash": "0" * 24, "outcome": "not_reported", "reviewer": "bob",
        "note": "n/a", "resolved_at": "2026-09-01T00:00:00+00:00", "run_id": bundle.run_id,
        "manifest_hash": bundle.manifest.manifest_hash, "extraction_schema_hash": schema_hash,
    }
    snapshot = _snapshot_from_bundle(bundle, schema, resolved_keys=(("rec-1", "population"),), resolution_history_as_of_render=[stray])
    with pytest.raises(ValueError, match="original_field_result_hash"):
        render_review_packet(bundle, schema, snapshot)


def test_render_review_packet_rejects_queue_and_history_overlap(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    verified_fields = bundle_review_fields(bundle.results)
    population = verified_fields[("rec-1", "population")]
    duplicate_history_entry = {
        "record_id": "rec-1", "field_id": "population", "original_status": population.status.value,
        "original_field_result_hash": sha256_hex(population.model_dump(mode="json")),
        "outcome": "not_reported", "reviewer": "bob", "note": "n/a",
        "resolved_at": "2026-09-01T00:00:00+00:00", "run_id": bundle.run_id,
        "manifest_hash": bundle.manifest.manifest_hash, "extraction_schema_hash": schema_hash,
    }
    snapshot = _snapshot_from_bundle(bundle, schema, resolution_history_as_of_render=[duplicate_history_entry])
    with pytest.raises(ValueError, match="overlap"):
        render_review_packet(bundle, schema, snapshot)


def test_render_review_packet_rejects_incomplete_queue_missing_a_review_required_field(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema)
    queue = [e.model_dump(mode="json") for e in snapshot.queue if e.field_id != "limitations"]
    tampered = _replace_queue(snapshot, queue)
    with pytest.raises(ValueError, match="missing"):
        render_review_packet(bundle, schema, tampered)


def test_render_review_packet_carries_resolution_history_through(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    snapshot = _snapshot_from_bundle(bundle, schema, resolved_keys=(("rec-1", "population"),))
    packet = render_review_packet(bundle, schema, snapshot)
    assert len(packet.resolution_history) == 1
    assert packet.resolution_history[0].field_id == "population"
    assert {e.field_id for e in packet.entries} == {"limitations"}


def _correctly_bound_history_stray(bundle, schema, *, record_id: str, field_id: str, **overrides) -> dict:
    """A resolution_history_as_of_render entry with a genuinely correct
    run_id/manifest_hash/extraction_schema_hash/original_status/
    original_field_result_hash binding -- so a test built on top of it
    isolates the outcome-conditional business-rule checks
    (`verify_existing_resolution()`) from the identity/hash-binding checks
    already covered above."""
    schema_hash = sha256_hex(schema.model_dump(mode="json"))
    verified_fields = bundle_review_fields(bundle.results)
    field_result = verified_fields[(record_id, field_id)]
    stray = {
        "record_id": record_id, "field_id": field_id, "original_status": field_result.status.value,
        "original_field_result_hash": sha256_hex(field_result.model_dump(mode="json")),
        "outcome": "not_reported", "reviewer": "bob", "note": "n/a",
        "resolved_at": "2026-09-01T00:00:00+00:00", "run_id": bundle.run_id,
        "manifest_hash": bundle.manifest.manifest_hash, "extraction_schema_hash": schema_hash,
    }
    stray.update(overrides)
    return stray


def test_render_review_packet_rejects_resolution_history_not_reported_entry_with_a_value(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    stray = _correctly_bound_history_stray(
        bundle, schema, record_id="rec-1", field_id="population", outcome="not_reported", value="Adults",
    )
    snapshot = _snapshot_from_bundle(
        bundle, schema, resolved_keys=(("rec-1", "population"),), resolution_history_as_of_render=[stray],
    )
    with pytest.raises(ValueError, match="not_reported.*must not carry a value"):
        render_review_packet(bundle, schema, snapshot)


def test_render_review_packet_rejects_resolution_history_extracted_entry_with_empty_evidence(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    stray = _correctly_bound_history_stray(
        bundle, schema, record_id="rec-1", field_id="population", outcome="extracted",
        value="Adults", evidence=[], attested=True,
    )
    snapshot = _snapshot_from_bundle(
        bundle, schema, resolved_keys=(("rec-1", "population"),), resolution_history_as_of_render=[stray],
    )
    with pytest.raises(ValueError, match="non-empty structured 'evidence'"):
        render_review_packet(bundle, schema, snapshot)


def test_render_review_packet_rejects_resolution_history_extracted_entry_with_false_attestation(tmp_path):
    bundle, schema = _build_bundle(tmp_path)
    stray = _correctly_bound_history_stray(
        bundle, schema, record_id="rec-1", field_id="population", outcome="extracted", value="Adults",
        evidence=[{"quotation_or_description": "adults aged 18-65", "locator": "p.4"}], attested=False,
    )
    snapshot = _snapshot_from_bundle(
        bundle, schema, resolved_keys=(("rec-1", "population"),), resolution_history_as_of_render=[stray],
    )
    with pytest.raises(ValueError, match="attested=True"):
        render_review_packet(bundle, schema, snapshot)


# ---------------------------------------------------------------------------
# Real Berg2023 charting-calibration fixture
# ---------------------------------------------------------------------------

def test_render_review_packet_against_real_berg2023_fixture():
    schema = scoping_chart_schema()
    store = ExtractionCheckpointStore(output_dir=BERG_OUTPUT_DIR, run_id=BERG_RUN_ID)
    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    assert {e.field_id for e in packet.entries} == {"population", "concept", "context", "limitations"}
    for entry in packet.entries:
        assert entry.field_result.status.value == "conflicting"
        assert len(entry.candidates) == len(entry.field_result.values)
        assert entry.evidence_index  # every conflicting field has at least one evidence span


def test_render_review_packet_against_real_berg2023_fixture_is_json_round_trippable():
    schema = scoping_chart_schema()
    store = ExtractionCheckpointStore(output_dir=BERG_OUTPUT_DIR, run_id=BERG_RUN_ID)
    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    from slr_pipeline.extraction.resolution_suggestions import ReviewPacket

    reloaded = ReviewPacket.model_validate(packet.model_dump(mode="json"))
    assert reloaded == packet


# ---------------------------------------------------------------------------
# Real Track 3 continuation six-field fixture
# (test_reviews/track3_continuation_six_field_fixture/PROVENANCE.md)
# ---------------------------------------------------------------------------

def test_render_review_packet_against_real_track3_continuation_six_field_fixture():
    schema = scoping_chart_schema()
    store = ExtractionCheckpointStore(output_dir=TRACK3_CONTINUATION_OUTPUT_DIR, run_id=TRACK3_CONTINUATION_RUN_ID)
    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    assert {e.field_id for e in packet.entries} == {
        "study_design", "population", "concept", "context", "limitations", "key_findings",
    }
    for entry in packet.entries:
        assert entry.field_result.requires_human_review is True
        assert len(entry.candidates) == len(entry.field_result.values)


def test_render_review_packet_against_real_track3_continuation_six_field_fixture_is_json_round_trippable():
    schema = scoping_chart_schema()
    store = ExtractionCheckpointStore(output_dir=TRACK3_CONTINUATION_OUTPUT_DIR, run_id=TRACK3_CONTINUATION_RUN_ID)
    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()
    snapshot = _snapshot_from_bundle(bundle, schema)
    packet = render_review_packet(bundle, schema, snapshot)
    from slr_pipeline.extraction.resolution_suggestions import ReviewPacket

    reloaded = ReviewPacket.model_validate(packet.model_dump(mode="json"))
    assert reloaded == packet
