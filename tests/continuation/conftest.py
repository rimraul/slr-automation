"""On-disk bundle fixture builder for `tests/continuation/`.

Also home to `load_real_fixture()`, the "real parent run" fixture Unit B's
own plan section calls for (a trimmed, already-collected, read-only
snapshot of `track3_run_20260913T135526Z_6e3e4fd3762941b383ec86d009463034`'s
real `terminal_state.json['included_records']` (14), the real 3 ledger
`full_text_eligibility` decisions (all exclusions -- this run has no
preserved-inclusion case, hence Unit B's own tests add a synthetic one), and
the real `track3_continuation_20260914T144702Z/manifest.json` (10 parsed/OK
documents) -- files under `tests/fixtures/continuation/`, never imported by
production code.

Unlike `tests/test_reference_bundle.py::_MinimalBundle` (which constructs a
`Bundle` dataclass directly in-memory, sidecars passed as constructor
arguments, never written to disk as JSON), `identity.py::resolve_parent_bundle()`
exercises `open_bundle()`'s own disk-reading path -- it needs a real
`payload_manifest.json` under `payload/`, real sidecar JSON files one level
up (`_maybe_load_sidecar()`), and a real `bundles_root/run_registry.json`.
This module mirrors `_MinimalBundle`'s hash-chain-correct construction
algorithm but adds the one artifact `open_bundle()` unconditionally requires
(`reporting/export_manifest.json`, resolved via the fixed logical path
`slr://reporting/artifact/export_manifest`) and writes every model to disk
instead of holding it only in memory."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

import slr_pipeline.reference_bundle as rb
from slr_pipeline.continuation.manifest import ContinuationManifest
from slr_pipeline.export_manifest import M1ProtocolInfo
from slr_pipeline.locators import LogicalLocator
from slr_pipeline.nodes.reporting import (
    ReportingConfigProjection,
    ReportingSourceSnapshot,
    VerifiedFile,
    VerifiedProjection,
    _hash_value,
    _state_value,
)
from slr_pipeline.selection_ledger import SelectionDecisionRecord
from slr_pipeline.state import Record, SynthesisOutput

_HEX = "a" * 64
_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "continuation"


@pytest.fixture(autouse=True)
def _fake_docling_validate(monkeypatch):
    """Real-data preflight corrective pass: `track3_continuation_run.py`'s
    `_docling_offline_env()` now calls `assert_docling_artifacts_valid()`
    before every `main()`/`run_continuation()` call, which hashes the real,
    ~1.2GB, gitignored `assets/models/track2-docling-cache` bundle --
    unavailable on a fresh checkout/CI runner and needlessly slow even
    where present. Autouse (mirrors `tests/test_track3_launcher.py`'s own
    identical fixture) so every test under `tests/continuation/` gets a
    fast, offline fake by default; tests that specifically exercise
    `_docling_offline_env()`'s own validate-first behavior override this via
    their own `monkeypatch.setattr(docling_artifacts, "validate", ...)`
    call."""
    from ft_screening import docling_artifacts

    monkeypatch.setattr(docling_artifacts, "validate", lambda path, expected_tree_digest=None: "fake" * 16)


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(seed: str, n: int = 64) -> str:
    return _raw_sha256(seed.encode("utf-8"))[:n]


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _artifact(*, logical: LogicalLocator, path: str, data: bytes,
              kind: str = "bundle_generated", media_type: str = "text/markdown",
              producer_stage: str = "reporting") -> rb.PayloadArtifact:
    sha = _raw_sha256(data)
    return rb.PayloadArtifact(
        artifact_id=f"{logical.kind}-{logical.subkind}-{sha[:16]}", logical_path=str(logical),
        path=path, sha256=sha, media_type=media_type, producer_stage=producer_stage, kind=kind,
    )


def _source_info(**overrides: Any) -> rb.PayloadSourceInfo:
    base = dict(
        git_commit=_hex("git", 40), execution_subtree_digest=_hex("subtree"),
        execution_subtree_provenance="runtime_verified",
        execution_subtree_clean=True, python_version="3.12.12",
        runner_image_os="ubuntu-22.04", runner_image_version="20260101.1",
        base_requirements_lock_sha256=_hex("base_req"), build_requirements_lock_sha256=_hex("build_req"),
        ft_screening_wheel_lock_sha256=_hex("ft_wheel_lock"), slr_pipeline_wheel_sha256=_hex("slr_wheel"),
        workflow_sha256=_hex("workflow"), provision_script_sha256=_hex("provision"),
        build_ft_wheel_script_sha256=_hex("build_ft_wheel"), author_cassette_script_sha256=_hex("author_cassette"),
        ft_screening=rb.FTScreeningSourceInfo(
            filename="ft_screening-1.0.0-py3-none-any.whl", sha256=_hex("ft_wheel"),
            licence_commit=_hex("licence", 40), code_parent="9604dc7", release_tag="track2-assets-v1"),
        docling_assets_manifest_digest=_hex("docling_assets"), docling_archive_sha256=_hex("docling_archive"),
        docling_model_manifest_sha256=_hex("docling_model"), fixture_corpus_digest=_hex("fixture_corpus"),
        cassette_sha256=_hex("cassette"), reference_spec_sha256=_hex("reference_spec"),
        rfc8785_version="0.1.4", mode_profile="track2_offline",
    )
    base.update(overrides)
    return rb.PayloadSourceInfo(**base)


def _reporting_stage_manifest_bytes(review_run_id: str) -> bytes:
    """A self-consistent `export_manifest.json` -- `manifest_hash` computed
    the same way `open_bundle()` recomputes it (`_hash_value` over the
    reloaded JSON dict minus `manifest_hash`), not merely a fixed dummy hex
    string (`_MinimalBundle`'s own `_minimal_reporting_manifest()` is NOT
    self-consistent -- it is only ever handed to `Bundle(...)` directly,
    never round-tripped through `open_bundle()`'s own disk-reading path)."""
    protocol = M1ProtocolInfo(protocol_uri=None, protocol_hash=None, protocol_registration_id=None,
                               ai_use_prespecified=None, protocol_deviations=[])
    unsigned = rb.ReportingStageManifest(review_run_id=review_run_id, review_type="rapid", m1_protocol=protocol,
                                         m10_data_governance=None, entries=[], manifest_hash="0" * 64)
    unsigned_bytes = unsigned.model_dump_json().encode("utf-8")
    reloaded = json.loads(unsigned_bytes)
    manifest_hash = _hash_value({k: v for k, v in reloaded.items() if k != "manifest_hash"})
    signed = unsigned.model_copy(update={"manifest_hash": manifest_hash})
    return signed.model_dump_json().encode("utf-8")


def write_disk_bundle(
    bundles_root: Path, *, decision: str = "APPROVED", pm_mode_profile: str = "track3_live",
    review_run_id: str = "run-1", registry_entry_override: Optional[Dict[str, Any]] = None,
    included_records: Optional[List[Record]] = None,
) -> str:
    """Writes a complete, hash-chain-correct, on-disk published bundle at
    `bundles_root/<bundle_id>/payload/` plus sidecars at
    `bundles_root/<bundle_id>/` and a matching `bundles_root/run_registry.json`
    -- the exact layout `identity.py::resolve_parent_bundle()` expects.
    Returns the bundle_id. Merges into any existing `run_registry.json`
    already at `bundles_root` (so a test may call this more than once for
    multiple bundles under the same root).

    `included_records`, when given, additionally stages a minimal
    `raw/state/terminal_state.json` artifact (logical path
    `slr://state/snapshot/terminal_state`, mirroring
    `reference_bundle.py::_collect_state_and_run_evidence()`'s own real
    staging convention) holding only `{"included_records": [...]}` -- enough
    for `Bundle.resolve("slr://state/snapshot/terminal_state")` to work
    against this fixture bundle, without running a real pipeline just to
    produce a genuine `terminal_state.json`."""
    bundles_root = Path(bundles_root)
    is_live = pm_mode_profile == "track3_live"

    run_mode_bytes = json.dumps({"llm_transport": "replay", "search_source": "fixture"}).encode("utf-8")
    export_manifest_bytes = _reporting_stage_manifest_bytes(review_run_id)

    artifacts = [
        _artifact(logical=LogicalLocator("bundle", "run_mode", "main"),
                  path="bundle/run_mode.json", data=run_mode_bytes, media_type="application/json"),
        _artifact(logical=LogicalLocator("reporting", "artifact", "export_manifest"),
                  path="reporting/export_manifest.json", data=export_manifest_bytes,
                  media_type="application/json", producer_stage="reporting"),
    ]
    report_bytes = b"# PRISMA-ScR report\n"
    conformance_bytes = json.dumps({"conformance": "ok"}).encode("utf-8")
    if not is_live:
        artifacts.extend([
            _artifact(logical=LogicalLocator("reporting", "artifact", "prisma_scr_report"),
                      path="reporting/prisma_scr_report.md", data=report_bytes, media_type="text/markdown"),
            _artifact(logical=LogicalLocator("reporting", "artifact", "prisma_scr_conformance"),
                      path="reporting/prisma_scr_conformance.json", data=conformance_bytes,
                      media_type="application/json"),
        ])
    terminal_state_bytes = None
    if included_records is not None:
        terminal_state_bytes = json.dumps(
            {"included_records": [r.to_dict() for r in included_records]}).encode("utf-8")
        artifacts.append(_artifact(
            logical=LogicalLocator("state", "snapshot", "terminal_state"),
            path="raw/state/terminal_state.json", data=terminal_state_bytes, media_type="application/json",
            kind="graph_state_snapshot", producer_stage="driver"))
    artifacts = sorted(artifacts, key=lambda a: a.logical_path)

    pm_source = (_source_info(mode_profile=pm_mode_profile) if pm_mode_profile == "track2_offline"
                 else _source_info(mode_profile=pm_mode_profile,
                                    execution_subtree_provenance="retrospective_reconstruction",
                                    execution_subtree_clean=None,
                                    fixture_corpus_digest=None, cassette_sha256=None))
    bundle_id = rb.compute_bundle_id(reference_spec_sha256=pm_source.reference_spec_sha256,
                                      review_run_id=review_run_id, created_utc=rb._now_iso())
    bundle_dir = bundles_root / bundle_id
    payload_root = bundle_dir / "payload"

    _write(payload_root / "bundle" / "run_mode.json", run_mode_bytes)
    _write(payload_root / "reporting" / "export_manifest.json", export_manifest_bytes)
    if not is_live:
        _write(payload_root / "reporting" / "prisma_scr_report.md", report_bytes)
        _write(payload_root / "reporting" / "prisma_scr_conformance.json", conformance_bytes)
    if terminal_state_bytes is not None:
        _write(payload_root / "raw" / "state" / "terminal_state.json", terminal_state_bytes)

    pm_dict = dict(payload_manifest_version="1", bundle_id=bundle_id, mode_profile=pm_mode_profile,
                   created_utc=rb._now_iso(), source=pm_source,
                   reference_spec_sha256=pm_source.reference_spec_sha256, artifacts=artifacts,
                   payload_hash="0" * 64)
    manifest_dump = rb.PayloadManifestV1(**pm_dict).model_dump(mode="json")
    payload_hash = rb._sha256_rfc8785_jcs({
        k: v for k, v in manifest_dump.items()
        if k != "payload_hash" and not (k == "parent_bundle_id" and v is None)
    })
    pm = rb.PayloadManifestV1(**{**pm_dict, "payload_hash": payload_hash})
    (payload_root / "payload_manifest.json").write_text(pm.model_dump_json(), encoding="utf-8")

    checks = [rb.AcceptanceCheck(id=i, status="pass", detail="ok") for i in range(1, 9)]
    ar_unsigned = rb.AcceptanceReport(
        payload_hash=pm.payload_hash, mode_profile=pm_mode_profile,
        source=pm_source, checks=checks, acceptance_report_hash="0" * 64)
    ar = ar_unsigned.model_copy(update={"acceptance_report_hash": ar_unsigned.self_hash()})

    att_unsigned = rb.ReviewerAttestation(
        payload_hash=pm.payload_hash, acceptance_report_hash=ar.acceptance_report_hash,
        prisma_scr_report_sha256=(rb.PRISMA_SCR_NOT_APPLICABLE_SHA256 if is_live else _raw_sha256(report_bytes)),
        prisma_scr_conformance_sha256=(rb.PRISMA_SCR_NOT_APPLICABLE_SHA256 if is_live
                                        else _raw_sha256(conformance_bytes)),
        reviewer="tester", decision=decision, notes="ok", attested_utc=rb._now_iso(),
        attestation_hash="0" * 64)
    att = att_unsigned.model_copy(update={"attestation_hash": att_unsigned.self_hash()})

    rel_unsigned = rb.ReleaseManifest(
        bundle_id=pm.bundle_id, mode_profile=pm_mode_profile,
        payload_hash=pm.payload_hash, acceptance_report_hash=ar.acceptance_report_hash,
        attestation_hash=att.attestation_hash, decision=decision, release_hash="0" * 64)
    rel = rel_unsigned.model_copy(update={"release_hash": rel_unsigned.self_hash()})

    (bundle_dir / "acceptance_report.json").write_text(ar.model_dump_json(), encoding="utf-8")
    (bundle_dir / "reviewer_attestation.json").write_text(att.model_dump_json(), encoding="utf-8")
    (bundle_dir / "release_manifest.json").write_text(rel.model_dump_json(), encoding="utf-8")

    reg_kwargs = dict(payload_hash=pm.payload_hash, release_hash=rel.release_hash,
                       acceptance_report_hash=ar.acceptance_report_hash, attestation_hash=att.attestation_hash,
                       mode_profile=rel.mode_profile, decision=decision)
    if registry_entry_override:
        reg_kwargs.update(registry_entry_override)
    reg = rb.RunRegistryEntry(**reg_kwargs)

    registry_path = bundles_root / "run_registry.json"
    if registry_path.is_file():
        registry = rb.RunRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
    else:
        registry = rb.RunRegistry(run_registry_version="1", runs={})
    registry.runs[bundle_id] = reg
    registry_path.write_text(registry.model_dump_json(), encoding="utf-8")

    return bundle_id


def _minimal_config_projection() -> ReportingConfigProjection:
    return ReportingConfigProjection(
        review_type="rapid", research_questions=[], inclusion_criteria=[], exclusion_criteria=[],
        screening_llm_model="gpt-4o-mini", generate_evidence_map=False, run_bertopic=False, n_topics=0,
        quality_appraisal_checklist_id=None, search_query="test query",
        population=None, concept=None, context=None, intervention=None, comparator=None, outcome=None,
        protocol_uri=None, protocol_hash=None, protocol_registration_id=None, ai_use_prespecified=None,
        protocol_deviations=[], data_governance_declaration=None, search_databases=[], search_as_of_date=None,
        screening_uncertain_threshold=0.5, generate_claim_backed_results=False, refine_batch_size=10,
        funding_statement=None,
    )


def minimal_reporting_source_snapshot(
    *, full_text_records: Optional[list] = None, ft_native_provenance: Optional[Dict[str, Any]] = None,
    review_run_id: str = "run-1", ledger_records: Optional[List[SelectionDecisionRecord]] = None,
) -> ReportingSourceSnapshot:
    """A `ReportingSourceSnapshot` with every field the `continuation`
    package's own code never reads set to its cheapest valid placeholder --
    only `full_text_records`/`ft_native_provenance`/`review_run_id`/
    `ledger_records` are meaningful to `lineage.py`/`preserved_pdfs.py`/
    `partition.py`. `ledger_records` is given as plain
    `SelectionDecisionRecord`s and wrapped into `VerifiedFile` here (the
    wrapper's `path`/`file_sha256`/`internal_hash` are never read by any
    `continuation` package code -- only `.model` is)."""
    config_projection = _minimal_config_projection()
    synthesis_output = SynthesisOutput(refine_batch_size=1)
    wrapped_ledger_records = [
        VerifiedFile(model=r, path=Path("x"), file_sha256="a" * 64, internal_hash=r.content_hash)
        for r in (ledger_records or [])
    ]
    return ReportingSourceSnapshot(
        review_run_id=review_run_id, config_projection=config_projection,
        config_projection_hash=_hash_value(config_projection.model_dump(mode="json")),
        raw_records=_state_value([]), screening_records=[], full_text_records=(full_text_records or []),
        screened_records_full=[], prisma=_state_value({}), search_log=_state_value([]),
        screening_review_resolutions=_state_value({}), full_text_review_resolutions=_state_value({}),
        full_text_review_queue=_state_value([]), screening_qc_state=None,
        oversight_mode_state={}, automation_origin_state={}, events=[], ledger_records=wrapped_ledger_records,
        current_screening_batch_results=[], extraction=None, quality_appraisal=None,
        synthesis_output=VerifiedProjection(model=synthesis_output, content_hash=_hash_value(synthesis_output)),
        evidence_map=None, ft_native_provenance=ft_native_provenance,
    )


def load_real_fixture() -> Tuple[List[Record], ReportingSourceSnapshot, ContinuationManifest]:
    """Loads the real-run fixture described in this module's own docstring.
    Returns `(included_records, parent_snapshot, continuation_manifest)`.
    `included_records` has 14 entries; `parent_snapshot.ledger_records`
    carries the real 3 `full_text_eligibility` exclusion decisions;
    `continuation_manifest` has the real 10 parsed/OK documents. The
    remaining record (`scopus-fb08897e9d4f238ce11c`, the real NNA paywalled
    record) is neither preserved nor retrieved by design -- it is the real
    `not_retrieved_ids` case."""
    raw_records = json.loads((_FIXTURES_DIR / "included_records.json").read_text(encoding="utf-8"))
    included_records = [Record(**r) for r in raw_records]

    raw_ledger = json.loads((_FIXTURES_DIR / "ledger_full_text_eligibility.json").read_text(encoding="utf-8"))
    ledger_records = [SelectionDecisionRecord(**r) for r in raw_ledger]
    parent_snapshot = minimal_reporting_source_snapshot(
        review_run_id=ledger_records[0].review_run_id if ledger_records else "run-1",
        ledger_records=ledger_records,
    )

    continuation_manifest = ContinuationManifest.model_validate_json(
        (_FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))

    return included_records, parent_snapshot, continuation_manifest


# ---------------------------------------------------------------------------
# Unit E -- a genuine, fully self-consistent, on-disk PUBLISHED parent
# bundle for `build_cumulative_reporting_snapshot()`'s own tests. Unlike
# `write_disk_bundle()` above (Unit A's own fixture -- stages only the
# `bundle/run_mode.json` + `reporting/export_manifest.json` artifacts
# `identity.py::resolve_parent_bundle()` itself reads), `from_bundle()`
# requires a real `raw/state/terminal_state.json` + full raw layer, which
# only a genuine `finalize_bundle()` call over a real (offline,
# hand-orchestrated) pipeline run produces. Reused, not reimplemented,
# from `tests/test_reference_bundle.py`'s own harness -- the same fixture
# shape `test_from_bundle_reconstructs_matching_review_run_id_and_records`/
# `test_refinalize_reporting_from_bundle_reproduces_reporting_tree` already
# prove is self-consistent end-to-end (including `compute_selection_
# splits()`, exercised transitively via `refinalize_reporting_from_bundle()`).
# ---------------------------------------------------------------------------


def _load_reference_bundle_test_module():
    path = Path(__file__).resolve().parent.parent / "test_reference_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "_reference_bundle_test_module_for_continuation_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_verify_bundle(bundle: rb.Bundle, *, decision: str = "APPROVED") -> rb.Bundle:
    """Attaches a self-consistent acceptance_report/reviewer_attestation/
    release_manifest/run_registry_entry (+ matching run_registry_key) to an
    already-`open_bundle()`d bundle, so that `verify_release_structure()`'s
    full 6-check sweep genuinely passes -- not merely a fabricated
    `run_registry_entry` attached in isolation (code-review corrective
    pass, round 3: `from_bundle()` now calls `verify_release_structure()`
    itself, since a bare `run_registry_entry` proves nothing about whether
    the parent was ever actually release-verified). Mirrors `write_disk_
    bundle()`'s own sidecar-construction algorithm above, applied here to a
    real `finalize_bundle()` output (with real on-disk PRISMA-ScR report/
    conformance bytes to hash, for a `track2_offline` bundle) rather than a
    from-scratch minimal payload."""
    import dataclasses

    pm = bundle.payload_manifest
    is_live = pm.mode_profile == "track3_live"
    checks = [rb.AcceptanceCheck(id=i, status="pass", detail="ok") for i in range(1, 9)]
    ar_unsigned = rb.AcceptanceReport(
        payload_hash=pm.payload_hash, mode_profile=pm.mode_profile,
        source=pm.source, checks=checks, acceptance_report_hash="0" * 64)
    ar = ar_unsigned.model_copy(update={"acceptance_report_hash": ar_unsigned.self_hash()})

    if is_live:
        report_sha = rb.PRISMA_SCR_NOT_APPLICABLE_SHA256
        conformance_sha = rb.PRISMA_SCR_NOT_APPLICABLE_SHA256
    else:
        report_sha = bundle.file_sha256_or_none("reporting/prisma_scr_report.md")
        conformance_sha = bundle.file_sha256_or_none("reporting/prisma_scr_conformance.json")
    att_unsigned = rb.ReviewerAttestation(
        payload_hash=pm.payload_hash, acceptance_report_hash=ar.acceptance_report_hash,
        prisma_scr_report_sha256=report_sha, prisma_scr_conformance_sha256=conformance_sha,
        reviewer="tester", decision=decision, notes="ok", attested_utc=rb._now_iso(),
        attestation_hash="0" * 64)
    att = att_unsigned.model_copy(update={"attestation_hash": att_unsigned.self_hash()})

    rel_unsigned = rb.ReleaseManifest(
        bundle_id=pm.bundle_id, mode_profile=pm.mode_profile,
        payload_hash=pm.payload_hash, acceptance_report_hash=ar.acceptance_report_hash,
        attestation_hash=att.attestation_hash, decision=decision, release_hash="0" * 64)
    rel = rel_unsigned.model_copy(update={"release_hash": rel_unsigned.self_hash()})

    reg = rb.RunRegistryEntry(
        payload_hash=pm.payload_hash, release_hash=rel.release_hash,
        acceptance_report_hash=ar.acceptance_report_hash, attestation_hash=att.attestation_hash,
        mode_profile=rel.mode_profile, decision=decision)

    return dataclasses.replace(
        bundle, acceptance_report=ar, reviewer_attestation=att, release_manifest=rel,
        run_registry_entry=reg, run_registry_key=pm.bundle_id)


def build_parent_bundle(tmp_path, monkeypatch) -> Tuple[rb.Bundle, Dict[str, Any]]:
    """Runs the real small offline pipeline (dedup/screening/full-text-
    eligibility/extraction/synthesis/reporting, no network anywhere) and
    publishes it as a `track2_offline` bundle. Returns `(bundle, state)`
    where `state` is the pipeline's own terminal state dict."""
    rb_test_module = _load_reference_bundle_test_module()
    harness = rb_test_module._load_reporting_node_harness()
    bundle_dir, state, _cfg = rb_test_module._finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    return bundle, state
