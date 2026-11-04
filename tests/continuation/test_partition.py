"""Unit B -- `slr_pipeline.continuation.partition`.

`derive_continuation_partition()` must not be shape-specific to the real
14/3/10/1 parent run (codex point 1) -- proven here with BOTH the real-run
fixture (`conftest.load_real_fixture()`, all 3 preserved decisions are
exclusions) AND an independently-constructed synthetic fixture with
different counts, including a preserved *inclusion* (round-2 code-review
finding 3 -- the real run alone can't prove the include/exclude split
works, since it happens to have zero preserved inclusions)."""
from __future__ import annotations

import pytest

from slr_pipeline.continuation.manifest import ContinuationManifest, ContinuationManifestDocument
from slr_pipeline.continuation.partition import derive_continuation_partition
from slr_pipeline.selection_ledger import SelectionDecisionRecord, build_selection_decision_record
from slr_pipeline.state import Record
from slr_pipeline.track3_run import Track3ConfigError

from .conftest import load_real_fixture, minimal_reporting_source_snapshot

_NNA_ID = "scopus-fb08897e9d4f238ce11c"


def test_real_fixture_reproduces_14_3_10_1_shape():
    included_records, parent_snapshot, continuation_manifest = load_real_fixture()
    assert len(included_records) == 14

    partition = derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)

    assert partition.preserved_included_ids == []
    assert sorted(partition.preserved_excluded_ids) == sorted(
        ["pubmed-246e529a0b53acb7d012", "pubmed-1f9c75fc79c4e6ad2f85", "pubmed-4c6c9f513cf62a9a5fb7"])
    assert len(partition.retrieved_ids) == 10
    assert partition.not_retrieved_ids == [_NNA_ID]
    assert sorted(partition.preserved_ids) == sorted(partition.preserved_excluded_ids)


def _record(record_id: str) -> Record:
    return Record(id=record_id, source="test", source_id=record_id)


def _ft_decision(record_id: str, decision: str, decision_id: str) -> SelectionDecisionRecord:
    return build_selection_decision_record(
        review_run_id="run-synth", decision_id=decision_id, source_event_id=f"src-{record_id}",
        record_id=record_id, stage="full_text_eligibility", decision=decision, decision_maker="ai",
    )


def _manifest(worklist_ids, status="parsed", identity_flag="OK") -> ContinuationManifest:
    return ContinuationManifest(
        continuation_workspace="/tmp/ws",
        documents=[
            ContinuationManifestDocument(
                worklist_id=w, save_as=f"{w}.pdf", file_sha256="a" * 64, status=status, identity_flag=identity_flag)
            for w in worklist_ids
        ],
    )


def test_synthetic_fixture_different_shape_with_preserved_inclusion():
    included_records = [_record("r-inc"), _record("r-exc"), _record("r-new"), _record("r-missing")]
    ledger_records = [
        _ft_decision("r-inc", "include", "11111111-1111-4111-8111-111111111111"),
        _ft_decision("r-exc", "exclude", "22222222-2222-4222-8222-222222222222"),
    ]
    parent_snapshot = minimal_reporting_source_snapshot(review_run_id="run-synth", ledger_records=ledger_records)
    continuation_manifest = _manifest(["r-new"])

    partition = derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)

    assert partition.preserved_included_ids == ["r-inc"]
    assert partition.preserved_excluded_ids == ["r-exc"]
    assert partition.retrieved_ids == ["r-new"]
    assert partition.not_retrieved_ids == ["r-missing"]
    assert sorted(partition.preserved_ids) == ["r-exc", "r-inc"]


def test_manifest_status_or_identity_flag_filters_out_non_parsed_documents():
    included_records = [_record("r-a"), _record("r-b")]
    parent_snapshot = minimal_reporting_source_snapshot(review_run_id="run-synth")
    continuation_manifest = _manifest(["r-a"], status="failed")
    continuation_manifest.documents.append(
        ContinuationManifestDocument(
            worklist_id="r-b", save_as="r-b.pdf", file_sha256="a" * 64, status="parsed", identity_flag="MISMATCH"))

    partition = derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)

    assert partition.retrieved_ids == []
    assert sorted(partition.not_retrieved_ids) == ["r-a", "r-b"]


def test_overlap_between_preserved_and_retrieved_raises():
    included_records = [_record("r-a")]
    ledger_records = [_ft_decision("r-a", "exclude", "33333333-3333-4333-8333-333333333333")]
    parent_snapshot = minimal_reporting_source_snapshot(review_run_id="run-synth", ledger_records=ledger_records)
    continuation_manifest = _manifest(["r-a"])

    with pytest.raises(Track3ConfigError, match="already has a preserved full_text_eligibility decision"):
        derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)


def test_manifest_worklist_id_not_in_included_records_raises_exhaustiveness_error():
    included_records = [_record("r-a")]
    parent_snapshot = minimal_reporting_source_snapshot(review_run_id="run-synth")
    continuation_manifest = _manifest(["r-a", "r-unexpected"])

    with pytest.raises(Track3ConfigError, match="exhaustive partition check failed"):
        derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)


def test_duplicate_worklist_id_across_manifest_documents_raises_exhaustiveness_error():
    included_records = [_record("r-a")]
    parent_snapshot = minimal_reporting_source_snapshot(review_run_id="run-synth")
    continuation_manifest = _manifest(["r-a", "r-a"])

    with pytest.raises(Track3ConfigError, match="exhaustive partition check failed"):
        derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)


def test_no_ledger_and_no_manifest_matches_puts_everything_in_not_retrieved():
    included_records = [_record("r-a"), _record("r-b")]
    parent_snapshot = minimal_reporting_source_snapshot(review_run_id="run-synth")
    continuation_manifest = _manifest([])

    partition = derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)

    assert partition.preserved_included_ids == []
    assert partition.preserved_excluded_ids == []
    assert partition.retrieved_ids == []
    assert sorted(partition.not_retrieved_ids) == ["r-a", "r-b"]
