"""Checkpoint 4: restart-safe pagination checkpoint storage
(``FileSearchCheckpointStore``). Covers atomic overwrite-per-page
persistence, fingerprint-mismatch never resuming stale progress, and
``envelope_hash`` integrity verification on load (review finding 5: the
envelope hash covers the complete checkpoint, not just records_so_far)."""

from __future__ import annotations

import json

import pytest

from slr_pipeline.evaluation.update_driver import (
    FileSearchCheckpointStore,
    UpdateWaveSearchArtifactIntegrityError,
)
from slr_pipeline.search_backends.base import SearchPageCheckpoint


def _checkpoint(**overrides) -> SearchPageCheckpoint:
    defaults = dict(
        request_fingerprint="fp-1",
        adapter="pubmed",
        adapter_config_version="1",
        cursor_state={"offset": 20},
        records_so_far=({"id": "r1", "title": "A"},),
        total_hits=100,
        page_count=1,
        completed=False,
    )
    defaults.update(overrides)
    return SearchPageCheckpoint(**defaults)


def test_load_returns_none_when_no_checkpoint_exists(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    assert store.load("pubmed", "fp-1") is None


def test_save_then_load_round_trips(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    cp = _checkpoint()
    store.save("pubmed", cp)
    loaded = store.load("pubmed", "fp-1")
    assert loaded is not None
    assert loaded.request_fingerprint == "fp-1"
    assert loaded.cursor_state == {"offset": 20}
    assert loaded.records_so_far == ({"id": "r1", "title": "A"},)
    assert loaded.completed is False


def test_load_returns_none_on_fingerprint_mismatch(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint(request_fingerprint="fp-old"))
    # A changed query/config/adapter-version must never silently resume
    # stale progress: caller starts page 1 fresh instead.
    assert store.load("pubmed", "fp-new") is None


def test_save_overwrites_previous_page_checkpoint(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint(page_count=1, records_so_far=({"id": "r1"},)))
    store.save("pubmed", _checkpoint(page_count=2, records_so_far=({"id": "r1"}, {"id": "r2"})))
    loaded = store.load("pubmed", "fp-1")
    assert loaded.page_count == 2
    assert loaded.records_so_far == ({"id": "r1"}, {"id": "r2"})


def test_save_leaves_no_temp_files_behind(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint())
    remaining = list(store.checkpoints_root.glob("*"))
    assert remaining == [store.checkpoints_root / "pubmed.json"]


def test_load_raises_on_tampered_records_so_far(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint())
    path = store.checkpoints_root / "pubmed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records_so_far"] = [{"id": "tampered"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UpdateWaveSearchArtifactIntegrityError, match="envelope_hash"):
        store.load("pubmed", "fp-1")


@pytest.mark.parametrize(
    "field, tampered_value",
    [
        ("cursor_state", {"offset": 999}),
        ("total_hits", 999999),
        ("page_count", 999),
        ("completed", True),
        ("source_exhausted", True),
        ("termination_reason", "tampered_reason"),
        ("effective_query", "tampered query"),
        ("adapter_config_version", "tampered-version"),
    ],
)
def test_load_raises_on_tampered_non_records_envelope_field(tmp_path, field, tampered_value):
    """Review finding 5: envelope_hash must cover cursor_state, total_hits,
    page_count, completed, source_exhausted, termination_reason,
    effective_query, and adapter_config_version -- not just records_so_far.
    A checkpoint integrity check covering only records_hash would miss all
    of these tampering cases."""
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint())
    path = store.checkpoints_root / "pubmed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = tampered_value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UpdateWaveSearchArtifactIntegrityError, match="envelope_hash"):
        store.load("pubmed", "fp-1")


def test_load_raises_on_missing_envelope_hash(tmp_path):
    """Review finding 6 (second round): a persisted checkpoint with no
    envelope_hash key at all must fail closed, not be silently trusted.
    No legacy checkpoint format predates envelope_hash."""
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint())
    path = store.checkpoints_root / "pubmed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["envelope_hash"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UpdateWaveSearchArtifactIntegrityError, match="envelope_hash"):
        store.load("pubmed", "fp-1")


def test_different_databases_are_independent(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    store.save("pubmed", _checkpoint(adapter="pubmed"))
    assert store.load("scopus", "fp-1") is None
    assert store.load("pubmed", "fp-1") is not None


def test_different_review_run_ids_are_independent(tmp_path):
    store_a = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-a")
    store_b = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-b")
    store_a.save("pubmed", _checkpoint())
    assert store_b.load("pubmed", "fp-1") is None


def test_rejects_unsafe_review_run_id(tmp_path):
    with pytest.raises(ValueError, match="not a safe token"):
        FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="../escape")


def test_rejects_unsafe_database_name(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    with pytest.raises(ValueError, match="not a safe token"):
        store.save("../escape", _checkpoint())


def test_hooks_for_binds_database(tmp_path):
    store = FileSearchCheckpointStore(output_dir=tmp_path, review_run_id="run-1")
    hooks = store.hooks_for("openalex")
    assert hooks.load("fp-1") is None
    hooks.save(_checkpoint(adapter="openalex"))
    loaded = hooks.load("fp-1")
    assert loaded is not None
    assert loaded.adapter == "openalex"
    assert store.load("openalex", "fp-1") is not None
