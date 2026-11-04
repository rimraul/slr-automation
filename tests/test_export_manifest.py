"""Unit tests for `slr_pipeline.export_manifest` -- `ReportingStageEntry`/
`ReportingStageManifest` (`track2_pipeline_readiness_plan_2026-08-30.md`
Sec 10.1(b)/10.3). End-to-end coverage over a real pipeline run lives in
`tests/nodes/test_reporting_node.py::test_export_manifest_schema_coverage`;
these are pure-model field-contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slr_pipeline.export_manifest import M1ProtocolInfo, ReportingStageEntry, ReportingStageManifest

_HEX = "a" * 64
_HEX2 = "b" * 64


def _protocol() -> M1ProtocolInfo:
    return M1ProtocolInfo(protocol_uri=None, protocol_hash=None, protocol_registration_id=None,
                           ai_use_prespecified=None, protocol_deviations=[])


def test_m1_protocol_info_allows_all_none():
    p = _protocol()
    assert p.protocol_uri is None and p.protocol_hash is None


def test_renderer_artifact_present_requires_file_sha256_and_built_from():
    entry = ReportingStageEntry(
        entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/methods_md",
        status="present", file_sha256=_HEX, built_from=["slr://extraction/manifest/extrun-abc"], name="methods_md",
    )
    assert entry.status == "present"

    with pytest.raises(ValidationError, match="file_sha256 is required"):
        ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/methods_md",
                             status="present", built_from=["slr://extraction/manifest/extrun-abc"], name="methods_md")

    with pytest.raises(ValidationError, match="requires built_from"):
        ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/methods_md",
                             status="present", file_sha256=_HEX, name="methods_md")


def test_renderer_artifact_not_applicable_requires_reason_forbids_file_sha256():
    entry = ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/evidence_map_csv",
                                 status="not_applicable", reason="not scoping", name="evidence_map_csv")
    assert entry.status == "not_applicable"

    with pytest.raises(ValidationError, match="requires a reason"):
        ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/evidence_map_csv",
                             status="not_applicable", name="evidence_map_csv")

    with pytest.raises(ValidationError, match="file_sha256 must be None"):
        ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/evidence_map_csv",
                             status="not_applicable", reason="x", file_sha256=_HEX, name="evidence_map_csv")


def test_entry_kind_forbids_fields_of_another_kind():
    with pytest.raises(ValidationError, match="not applicable to it"):
        ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/methods_md",
                             status="present", file_sha256=_HEX, built_from=[], name="methods_md",
                             record_id="rec-a")  # selection_decision-only field


def test_entry_kind_missing_required_field_rejected():
    with pytest.raises(ValidationError, match="missing required field"):
        ReportingStageEntry(entry_kind="selection_decision",
                             logical_path="slr://selection_ledger/decision/" + "a" * 32 + "/" + "b" * 64,
                             status="present", file_sha256=_HEX, internal_hash=_HEX2,
                             record_id="rec-a", stage="dedup")  # missing decision_id


def test_logical_path_kind_must_match_entry_kind():
    with pytest.raises(ValidationError, match="expected kind"):
        ReportingStageEntry(entry_kind="selection_decision", logical_path="slr://reporting/artifact/methods_md",
                             status="present", file_sha256=_HEX, internal_hash=_HEX2,
                             record_id="rec-a", stage="dedup", decision_id="dec-1")


def test_extraction_component_document_id_bound_to_checkpoint():
    with pytest.raises(ValidationError, match="only set when component"):
        ReportingStageEntry(entry_kind="extraction_component", logical_path="slr://extraction/manifest/extrun-abc",
                             status="present", file_sha256=_HEX, internal_hash=_HEX2,
                             run_id="extrun-abc", component="manifest", document_id="rec-a")

    with pytest.raises(ValidationError, match="document_id is required"):
        ReportingStageEntry(entry_kind="extraction_component", logical_path="slr://extraction/checkpoint/extrun-abc/rec-a",
                             status="present", file_sha256=_HEX, internal_hash=_HEX2,
                             run_id="extrun-abc", component="checkpoint")


def test_component_vocab_rejects_unknown_component():
    with pytest.raises(ValidationError, match="must be one of"):
        ReportingStageEntry(entry_kind="extraction_component", logical_path="slr://extraction/manifest/extrun-abc",
                             status="present", file_sha256=_HEX, internal_hash=_HEX2,
                             run_id="extrun-abc", component="confirmation")  # not in extraction's vocab


def test_ft_run_artifact_requires_no_internal_hash():
    entry = ReportingStageEntry(entry_kind="ft_run_artifact", logical_path="slr://ft_screening/run_result/run-1",
                                 status="present", file_sha256=_HEX,
                                 ft_screening_run_id="run-1", component="run_result")
    assert entry.internal_hash is None


def test_built_from_accepts_locator_and_glob_rejects_malformed():
    entry = ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/synthesis_json",
                                 status="present", file_sha256=_HEX,
                                 built_from=["slr://ai_use_events/event/*", "slr://extraction/manifest/extrun-abc"],
                                 name="synthesis_json")
    assert entry.built_from is not None

    with pytest.raises(ValidationError):
        ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/synthesis_json",
                             status="present", file_sha256=_HEX, built_from=["not-a-locator"], name="synthesis_json")


def _minimal_manifest(entries) -> ReportingStageManifest:
    return ReportingStageManifest(
        review_run_id="run-1", review_type="rapid", m1_protocol=_protocol(),
        m10_data_governance=None, entries=entries, manifest_hash=_HEX,
    )


def test_manifest_requires_sorted_unique_entries():
    e1 = ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/a_name",
                              status="present", file_sha256=_HEX, built_from=[], name="a_name")
    e2 = ReportingStageEntry(entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/b_name",
                              status="present", file_sha256=_HEX2, built_from=[], name="b_name")

    manifest = _minimal_manifest([e1, e2])
    assert [e.logical_path for e in manifest.entries] == [e1.logical_path, e2.logical_path]

    with pytest.raises(ValidationError, match="sorted"):
        _minimal_manifest([e2, e1])

    with pytest.raises(ValidationError, match="duplicate"):
        _minimal_manifest([e1, e1])


def test_manifest_extra_field_forbidden():
    with pytest.raises(ValidationError):
        ReportingStageManifest(review_run_id="run-1", review_type="rapid", m1_protocol=_protocol(),
                                m10_data_governance=None, entries=[], manifest_hash=_HEX, extra_field="nope")
