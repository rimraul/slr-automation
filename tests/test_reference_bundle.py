"""Unit tests for `slr_pipeline.reference_bundle` -- Track 2 pipeline
readiness plan (`track2_pipeline_readiness_plan_2026-08-30.md`) step 15
[C14]: `PayloadManifestV1`/sidecar models (Sec 10.1(a)/10.7), the release
guard `verify_release_structure()` (Sec 10.7's exact 6-check algorithm),
the atomic-publish + per-bundle lock lifecycle (Sec 10.8), locator/`run:`
pointer resolution (Sec 10.6), and `finalize_bundle()`/`open_bundle()`/
`from_bundle()`/`refinalize_reporting_from_bundle()` (Sec 10.2/10.10),
exercised end-to-end over a real (hand-orchestrated, in-process) small
pipeline run -- `tests/nodes/test_reporting_node.py::
_run_full_pipeline_to_reporting`'s own reusable fixture, since Track 2's
own reference-run child process is a real, slow, out-of-process cassette
replay this module's tests do not need."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

import slr_pipeline.reference_bundle as rb
from slr_pipeline.config import DedupConfig, ReviewConfig
from slr_pipeline.export_manifest import M1ProtocolInfo, ReportingStageManifest
from slr_pipeline.extraction.hashing import file_sha256
from slr_pipeline.locators import LogicalLocator

_HEX = "a" * 64
_HEX2 = "b" * 64
REPO_ROOT = Path(__file__).resolve().parent.parent


# `FixtureInputPaths` used to have a `.default()` fallback computed from
# `slr_pipeline.reference_run`'s own `Path(__file__)`-derived constants --
# removed (Track-2 packaging-boundary fix, CI run 34155102694) since that
# assumes `slr_pipeline` sits inside a git checkout. This test file's own
# `REPO_ROOT` (a genuine, always-valid checkout-relative constant, since
# pytest always reads `tests/` from the real checkout) supplies the same
# paths explicitly instead.
_FIXTURES = rb.FixtureInputPaths(
    spec_dir=REPO_ROOT / "tests" / "reference_run" / "fixtures" / "spec",
    corpus_dir=REPO_ROOT / "tests" / "reference_run" / "fixtures" / "corpus",
    protocol_path=REPO_ROOT / "track2" / "readiness_protocol.md",
    prisma_scr_matrix_path=REPO_ROOT / "tests" / "reference_run" / "fixtures" / "prisma_scr_matrix.json",
)


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(seed: str, n: int = 64) -> str:
    return _raw_sha256(seed.encode("utf-8"))[:n]


def _protocol() -> M1ProtocolInfo:
    return M1ProtocolInfo(protocol_uri=None, protocol_hash=None, protocol_registration_id=None,
                           ai_use_prespecified=None, protocol_deviations=[])


def _minimal_reporting_manifest() -> ReportingStageManifest:
    return ReportingStageManifest(review_run_id="run-1", review_type="rapid", m1_protocol=_protocol(),
                                   m10_data_governance=None, entries=[], manifest_hash=_HEX)


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


class _MinimalBundle:
    """Builds a self-consistent (hash-chain-correct, on-disk) minimal
    published bundle directly from the Sec 10.1(a)/10.7 models -- no real
    pipeline run required. Each keyword override lets a test construct a
    bundle that is internally hash-consistent everywhere EXCEPT the one
    cross-field binding under test, matching how `verify_release_structure()`'s
    check 6 is actually reachable (a forged-but-self-signed sidecar, not a
    naive post-hoc field edit that check 3a's self-hash would already
    catch first)."""

    def __init__(self, tmp_path: Path, *, decision: str = "APPROVED",
                 pm_mode_profile: str = "track2_offline", ar_mode_profile: Optional[str] = None,
                 rel_mode_profile: Optional[str] = None, ar_source: Optional[rb.PayloadSourceInfo] = None,
                 ar_payload_hash: Optional[str] = None,
                 run_registry_key: Optional[str] = None, run_mode_document: Optional[Dict[str, Any]] = None,
                 registry_entry_override: Optional[Dict[str, Any]] = None) -> None:
        bundle_dir = tmp_path / "t2-bundle"
        payload_root = bundle_dir / "payload"
        payload_root.mkdir(parents=True)

        is_live = pm_mode_profile == "track3_live"
        report_bytes = b"# PRISMA-ScR report\n"
        conformance_bytes = json.dumps({"conformance": "ok"}).encode("utf-8")
        run_mode_bytes = json.dumps(run_mode_document or {"llm_transport": "replay", "search_source": "fixture"}
                                     ).encode("utf-8")
        _write(payload_root / "bundle" / "run_mode.json", run_mode_bytes)

        artifacts = [
            _artifact(logical=LogicalLocator("bundle", "run_mode", "main"),
                      path="bundle/run_mode.json", data=run_mode_bytes, media_type="application/json"),
        ]
        if not is_live:
            # `track3_live` never stages either PRISMA-ScR artifact (Sec
            # 2.6a) -- a genuinely self-consistent `track3_live` bundle
            # must omit them, not merely tolerate a mismatched hash.
            _write(payload_root / "reporting" / "prisma_scr_report.md", report_bytes)
            _write(payload_root / "reporting" / "prisma_scr_conformance.json", conformance_bytes)
            artifacts.extend([
                _artifact(logical=LogicalLocator("reporting", "artifact", "prisma_scr_report"),
                          path="reporting/prisma_scr_report.md", data=report_bytes, media_type="text/markdown"),
                _artifact(logical=LogicalLocator("reporting", "artifact", "prisma_scr_conformance"),
                          path="reporting/prisma_scr_conformance.json", data=conformance_bytes,
                          media_type="application/json"),
            ])
        artifacts = sorted(artifacts, key=lambda a: a.logical_path)

        pm_source = (_source_info(mode_profile=pm_mode_profile) if pm_mode_profile == "track2_offline"
                     else _source_info(mode_profile=pm_mode_profile,
                                        execution_subtree_provenance="retrospective_reconstruction",
                                        execution_subtree_clean=None,
                                        fixture_corpus_digest=None, cassette_sha256=None))
        bundle_id = rb.compute_bundle_id(reference_spec_sha256=pm_source.reference_spec_sha256,
                                          review_run_id="run-1", created_utc=rb._now_iso())
        pm_dict = dict(payload_manifest_version="1", bundle_id=bundle_id, mode_profile=pm_mode_profile,
                       created_utc=rb._now_iso(), source=pm_source,
                       reference_spec_sha256=pm_source.reference_spec_sha256, artifacts=artifacts,
                       payload_hash="0" * 64)
        payload_hash = rb._sha256_rfc8785_jcs(
            rb.PayloadManifestV1(**pm_dict).model_dump(mode="json", exclude={"payload_hash"}))
        pm = rb.PayloadManifestV1(**{**pm_dict, "payload_hash": payload_hash})

        checks = [rb.AcceptanceCheck(id=i, status="pass", detail="ok") for i in range(1, 9)]
        ar_source = ar_source if ar_source is not None else pm_source
        ar_unsigned = rb.AcceptanceReport(
            payload_hash=(ar_payload_hash if ar_payload_hash is not None else pm.payload_hash),
            mode_profile=(ar_mode_profile or pm_mode_profile),
            source=ar_source, checks=checks, acceptance_report_hash="0" * 64)
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
            bundle_id=pm.bundle_id, mode_profile=(rel_mode_profile or pm_mode_profile),
            payload_hash=pm.payload_hash, acceptance_report_hash=ar.acceptance_report_hash,
            attestation_hash=att.attestation_hash, decision=decision, release_hash="0" * 64)
        rel = rel_unsigned.model_copy(update={"release_hash": rel_unsigned.self_hash()})

        reg_kwargs = dict(payload_hash=pm.payload_hash, release_hash=rel.release_hash,
                           acceptance_report_hash=ar.acceptance_report_hash, attestation_hash=att.attestation_hash,
                           mode_profile=rel.mode_profile, decision=decision)
        if registry_entry_override:
            reg_kwargs.update(registry_entry_override)
        reg = rb.RunRegistryEntry(**reg_kwargs)

        self.bundle = rb.Bundle(
            bundle_id=pm.bundle_id, payload_root=payload_root, bundle_dir=bundle_dir,
            payload_manifest=pm, reporting_stage_manifest=_minimal_reporting_manifest(),
            acceptance_report=ar, reviewer_attestation=att, release_manifest=rel,
            run_registry_entry=reg, run_registry_key=(run_registry_key if run_registry_key is not None
                                                        else pm.bundle_id),
        )


def _build(tmp_path: Path, **kw: Any) -> rb.Bundle:
    return _MinimalBundle(tmp_path, **kw).bundle


# ===========================================================================
# Sec 10.7 -- sidecar schema contract tests
# ===========================================================================

def test_acceptance_check_forbids_extra_field_and_bad_status():
    rb.AcceptanceCheck(id=1, status="pass", detail="ok")
    with pytest.raises(ValidationError):
        rb.AcceptanceCheck(id=1, status="pass", detail="ok", extra="nope")
    with pytest.raises(ValidationError):
        rb.AcceptanceCheck(id=1, status="unknown", detail="ok")


def test_acceptance_report_requires_exactly_eight_checks(tmp_path):
    source = _source_info()
    seven = [rb.AcceptanceCheck(id=i, status="pass", detail="ok") for i in range(1, 8)]
    with pytest.raises(ValidationError):
        rb.AcceptanceReport(payload_hash=_HEX, mode_profile="track2_offline", source=source,
                             checks=seven, acceptance_report_hash=_HEX2)


def test_acceptance_report_has_no_bundle_id_or_semantic_review_field():
    assert "bundle_id" not in rb.AcceptanceReport.model_fields
    assert "created_utc" not in rb.AcceptanceReport.model_fields
    assert "prisma_scr_semantic_review" not in rb.AcceptanceReport.model_fields


def test_reviewer_attestation_has_no_bundle_id_field_and_requires_notes():
    assert "bundle_id" not in rb.ReviewerAttestation.model_fields
    assert "reviewed_utc" not in rb.ReviewerAttestation.model_fields
    with pytest.raises(ValidationError):
        rb.ReviewerAttestation(payload_hash=_HEX, acceptance_report_hash=_HEX, prisma_scr_report_sha256=_HEX,
                                prisma_scr_conformance_sha256=_HEX, reviewer="x", decision="APPROVED",
                                attested_utc=rb._now_iso(), attestation_hash=_HEX)  # missing notes


def test_release_manifest_forbids_finalized_utc():
    assert "finalized_utc" not in rb.ReleaseManifest.model_fields


def test_run_registry_default_is_empty_and_versioned():
    reg = rb.RunRegistry()
    assert reg.run_registry_version == "1"
    assert reg.runs == {}


def test_committed_run_registry_json_parses_empty():
    """`track2/run_registry.json` -- the durable registry Track 2 commits
    as the empty template (Sec 10.7); Track 2 CI never writes to it, only
    Track 3's `finalize_release` does (plan Sec 13 step 20's "durable real
    registry" note). Parses as a schema-valid `RunRegistry` with no runs."""
    path = REPO_ROOT / "track2" / "run_registry.json"
    reg = rb.RunRegistry.model_validate_json(path.read_text(encoding="utf-8"))
    assert reg.run_registry_version == "1"
    assert reg.runs == {}


# ===========================================================================
# Sec 10.7 -- verify_release_structure()'s 6-check release guard
# ===========================================================================

def test_verify_release_structure_accepts_valid_approved_bundle(tmp_path):
    bundle = _build(tmp_path, run_mode_document={"llm_transport": "live", "search_source": "live"})
    rb.verify_release_structure(bundle)  # no raise


def test_verify_release_structure_accepts_valid_track3_live_bundle(tmp_path):
    """A genuinely self-consistent `track3_live` bundle -- neither PRISMA-ScR
    file staged, both attestation hashes the not-applicable sentinel --
    passes the release guard exactly like a `track2_offline` bundle does."""
    bundle = _build(tmp_path, pm_mode_profile="track3_live", ar_mode_profile="track3_live",
                     rel_mode_profile="track3_live",
                     run_mode_document={"mode_profile": "track3_live", "llm_transport": "live",
                                         "search_source": "live"})
    rb.verify_release_structure(bundle)  # no raise


def test_verify_release_structure_rejects_track3_live_bundle_with_prisma_file_staged(tmp_path):
    """Defense in depth: even if a `track3_live` payload_manifest somehow
    lists a staged PRISMA-ScR artifact, the release guard must reject it
    rather than silently accept a Track-2-only file on a live bundle."""
    bundle = _build(tmp_path, pm_mode_profile="track3_live", ar_mode_profile="track3_live",
                     rel_mode_profile="track3_live",
                     run_mode_document={"mode_profile": "track3_live", "llm_transport": "live",
                                         "search_source": "live"})
    report_bytes = b"# PRISMA-ScR report\n"
    _write(bundle.payload_root / "reporting" / "prisma_scr_report.md", report_bytes)
    tampered_artifacts = list(bundle.payload_manifest.artifacts) + [
        _artifact(logical=LogicalLocator("reporting", "artifact", "prisma_scr_report"),
                  path="reporting/prisma_scr_report.md", data=report_bytes, media_type="text/markdown"),
    ]
    tampered_pm = bundle.payload_manifest.model_copy(update={"artifacts": sorted(
        tampered_artifacts, key=lambda a: a.logical_path)})
    tampered = dataclasses.replace(bundle, payload_manifest=tampered_pm)
    with pytest.raises(rb.ReleaseStructureInvalid, match="must not contain"):
        rb.verify_release_structure(tampered)


def test_verify_release_structure_rejects_unknown_mode_profile(tmp_path):
    """A third/bogus `mode_profile` value must fail closed, never silently
    fall into either the track2_offline or track3_live PRISMA-ScR branch."""
    bundle = _build(tmp_path)
    tampered_pm = bundle.payload_manifest.model_copy(update={"mode_profile": "bogus_mode"})
    tampered = dataclasses.replace(bundle, payload_manifest=tampered_pm)
    with pytest.raises(rb.ReleaseStructureInvalid, match="unknown mode_profile"):
        rb.verify_release_structure(tampered)


def test_verify_release_structure_rejects_track3_live_bundle_with_non_sentinel_prisma_hash(tmp_path):
    bundle = _build(tmp_path, pm_mode_profile="track3_live", ar_mode_profile="track3_live",
                     rel_mode_profile="track3_live",
                     run_mode_document={"mode_profile": "track3_live", "llm_transport": "live",
                                         "search_source": "live"})
    # The hash-chain cross-sidecar check (att.acceptance_report_hash/
    # att.payload_hash) and this PRISMA sentinel check both run BEFORE
    # check 3a's self-hash recompute, so this field can be tampered
    # without resigning `attestation_hash` and still reach this check.
    tampered_att = bundle.reviewer_attestation.model_copy(update={"prisma_scr_report_sha256": "f" * 64})
    tampered = dataclasses.replace(bundle, reviewer_attestation=tampered_att)
    with pytest.raises(rb.ReleaseStructureInvalid, match="not-applicable sentinel"):
        rb.verify_release_structure(tampered)


def test_verify_release_structure_accepts_structurally_valid_rejected_bundle(tmp_path):
    """A structurally-valid REJECTED release passes checks 1-6 and is
    rejected ONLY by the final decision check -- never conflated with a
    malformed bundle."""
    bundle = _build(tmp_path, decision="REJECTED")
    with pytest.raises(rb.ReleaseNotApproved):
        rb.verify_release_structure(bundle)


def test_verify_release_structure_requires_all_four_sidecars(tmp_path):
    bundle = _build(tmp_path)
    incomplete = dataclasses.replace(bundle, acceptance_report=None)
    with pytest.raises(rb.ReleaseStructureInvalid, match="missing one of"):
        rb.verify_release_structure(incomplete)


def test_release_guard_rejects_bundle_id_tamper(tmp_path):
    """`bundle.run_registry_key` is caller/directory-supplied, independent
    of any sidecar's own signed content -- an attacker relabelling the
    bundle directory to a different identity than the sidecars themselves
    declare must be rejected by check 6, even though every sidecar's own
    hash chain is perfectly self-consistent."""
    bundle = _build(tmp_path, run_registry_key="t2-" + "f" * 24)
    with pytest.raises(rb.ReleaseStructureInvalid, match="registry key"):
        rb.verify_release_structure(bundle)


def test_release_guard_rejects_mode_tamper(tmp_path):
    bundle = _build(tmp_path, pm_mode_profile="track2_offline", ar_mode_profile="track3_live",
                     rel_mode_profile="track2_offline")
    with pytest.raises(rb.ReleaseStructureInvalid, match="mode_profile"):
        rb.verify_release_structure(bundle)


def test_release_guard_rejects_source_tamper(tmp_path):
    """The acceptance run must attest to the SAME source the payload
    claims -- an acceptance_report signed against a different `source`
    than the payload_manifest's own is rejected even with an otherwise
    perfectly hash-consistent bundle."""
    other_source = _source_info(git_commit=_hex("a-different-git-commit", 40))
    bundle = _build(tmp_path, ar_source=other_source)
    with pytest.raises(rb.ReleaseStructureInvalid, match="SAME source"):
        rb.verify_release_structure(bundle)


def test_release_guard_rejects_acceptance_payload_hash_tamper(tmp_path):
    """The 8 acceptance checks must have run against THIS payload's bytes --
    an acceptance_report that is internally self-consistent (self_hash still
    matches) but whose own `payload_hash` field points at a DIFFERENT
    payload than payload_manifest.payload_hash must be rejected, even though
    the attestation/release/registry all correctly agree with the real
    payload_hash and every other hash-chain link is intact."""
    bundle = _build(tmp_path, ar_payload_hash="f" * 64)
    with pytest.raises(rb.ReleaseStructureInvalid, match="acceptance_report.payload_hash"):
        rb.verify_release_structure(bundle)


def test_release_guard_rejects_registry_mismatch(tmp_path):
    bundle = _build(tmp_path, registry_entry_override={"decision": "REJECTED"})
    with pytest.raises(rb.ReleaseStructureInvalid, match="run_registry"):
        rb.verify_release_structure(bundle)


def test_release_guard_rejects_hash_chain_tamper(tmp_path):
    bundle = _build(tmp_path)
    tampered_att = bundle.reviewer_attestation.model_copy(update={"payload_hash": "0" * 64})
    tampered = dataclasses.replace(bundle, reviewer_attestation=tampered_att)
    with pytest.raises(rb.ReleaseStructureInvalid):
        rb.verify_release_structure(tampered)


def test_release_guard_rejects_self_hash_tamper(tmp_path):
    """Editing a sidecar's own content post-hoc without recomputing its
    own `*_hash` field (e.g. a reviewer's `notes`) must fail check 3a even
    though it never touches any cross-sidecar hash-chain field."""
    bundle = _build(tmp_path)
    tampered_att = bundle.reviewer_attestation.model_copy(update={"notes": "quietly edited after signing"})
    tampered = dataclasses.replace(bundle, reviewer_attestation=tampered_att)
    with pytest.raises(rb.ReleaseStructureInvalid, match="recomputed content hash"):
        rb.verify_release_structure(tampered)


def test_release_guard_rejects_synthetic_provenance_claim(tmp_path):
    """`assert_not_synthetic_evidence()` (composed into
    `assert_evidentiary_release()`) refuses a bundle whose own
    `run_mode.json` discloses a replay/fixture transport, regardless of
    what its sidecars otherwise claim."""
    bundle = _build(tmp_path, run_mode_document={"llm_transport": "replay", "search_source": "fixture"})
    with pytest.raises(rb.SyntheticEvidenceClaim):
        rb.assert_not_synthetic_evidence(bundle)


def test_assert_evidentiary_release_requires_track3_live(tmp_path):
    bundle = _build(tmp_path, run_mode_document={"llm_transport": "live", "search_source": "live"})
    with pytest.raises(rb.NotEvidentiary):
        rb.assert_evidentiary_release(bundle)  # mode_profile is track2_offline here


# ===========================================================================
# Sec 10.6 -- locator/pointer resolution
# ===========================================================================

def test_resolve_json_pointer_whole_document_and_traversal():
    doc = {"a": [{"b": "hit"}]}
    assert rb.resolve_json_pointer(doc, "") == doc
    assert rb.resolve_json_pointer(doc, "/a/0/b") == "hit"
    with pytest.raises(rb.RunLocatorError):
        rb.resolve_json_pointer(doc, "/a/9")
    with pytest.raises(rb.RunLocatorError):
        rb.resolve_json_pointer(doc, "no-leading-slash")


def test_resolve_pointer_json_and_jsonl_and_line_ranges():
    json_bytes = json.dumps({"x": 1}).encode("utf-8")
    assert rb.resolve_pointer(media_type="application/json", data=json_bytes, pointer=None) == json_bytes
    assert json.loads(rb.resolve_pointer(media_type="application/json", data=json_bytes, pointer="/x")) == 1

    jsonl_bytes = b'{"a": 1}\n{"a": 2}\n'
    line2 = rb.resolve_pointer(media_type="application/jsonl", data=jsonl_bytes, pointer="/2")
    assert json.loads(line2) == {"a": 2}
    line2_field = rb.resolve_pointer(media_type="application/jsonl", data=jsonl_bytes, pointer="/2/a")
    assert json.loads(line2_field) == 2

    md_bytes = b"line one\nline two\nline three\n"
    assert rb.resolve_pointer(media_type="text/markdown", data=md_bytes, pointer="L2") == b"line two\n"
    assert rb.resolve_pointer(media_type="text/markdown", data=md_bytes, pointer="L1-2") == b"line one\nline two\n"


def test_parse_run_locator_round_trip_and_rejects_malformed():
    value = f"run:t2-{'a' * 24}@{'b' * 64}/reporting-artifact-{'c' * 16}#/x"
    parsed = rb.parse_run_locator(value)
    assert str(parsed) == value
    with pytest.raises(rb.RunLocatorError):
        rb.parse_run_locator("not-a-run-locator")


def test_compute_bundle_id_deterministic_and_prefixed():
    a = rb.compute_bundle_id(reference_spec_sha256=_HEX, review_run_id="run-1", created_utc="2026-01-01T00:00:00Z")
    b = rb.compute_bundle_id(reference_spec_sha256=_HEX, review_run_id="run-1", created_utc="2026-01-01T00:00:00Z")
    c = rb.compute_bundle_id(reference_spec_sha256=_HEX, review_run_id="run-2", created_utc="2026-01-01T00:00:00Z")
    assert a == b
    assert a != c
    assert rb.BUNDLE_ID_RE.match(a)


# ===========================================================================
# _boot_id() / _pid_alive() -- real (unmocked) ctypes behavior on Windows
# ===========================================================================
# `/code-review` (Track 2 step-20 holistic pass) found GetTickCount64/
# OpenProcess/CloseHandle were called via ctypes.windll without ever setting
# `.restype`, so ctypes defaulted each to a 32-bit signed `c_int` -- silently
# truncating/sign-flipping GetTickCount64's true 64-bit tick count past
# ~24.85 days of uptime, and risking the same for a HANDLE with its high bit
# set. Both existing tests above monkeypatch these functions away entirely,
# so neither would have caught this. These call the real functions.

@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only")
def test_boot_id_sets_gettickcount64_restype_to_uint64():
    boot_id = rb._boot_id()
    assert boot_id.startswith("win-") or boot_id.startswith("unknown-")
    # A c_int-truncated GetTickCount64 wraps negative once uptime exceeds
    # 2**31 ms; restype=c_uint64 must prevent that regardless of this
    # machine's actual current uptime.
    import ctypes
    assert ctypes.windll.kernel32.GetTickCount64.restype == ctypes.c_uint64


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only")
def test_pid_alive_true_for_current_process():
    assert rb._pid_alive(os.getpid()) is True


@pytest.mark.skipif(sys.platform != "win32", reason="ctypes.windll is Windows-only")
def test_pid_alive_false_for_implausible_pid():
    # PID 0 is the Windows System Idle Process (not openable); a very large
    # implausible PID is reliably not a running process either.
    assert rb._pid_alive(2**31 - 1) is False


# ===========================================================================
# Sec 10.8 -- atomic publish primitive + per-bundle lock lifecycle
# ===========================================================================

def test_atomic_new_file_writes_and_refuses_to_replace(tmp_path):
    target = tmp_path / "sub" / "file.json"
    rb.atomic_new_file(target, b'{"a": 1}')
    assert target.read_bytes() == b'{"a": 1}'
    with pytest.raises(rb.PublicationConflict):
        rb.atomic_new_file(target, b'{"a": 2}')
    assert target.read_bytes() == b'{"a": 1}'  # untouched by the failed second write
    assert list(tmp_path.rglob(".*.tmp.*")) == []  # tmp sibling always cleaned up


def test_acquire_lock_blocks_same_host_alive_pid(tmp_path, monkeypatch):
    handle = rb._acquire_lock(tmp_path, "t2-lockme", "test")
    monkeypatch.setattr(rb, "_pid_alive", lambda pid: True)
    with pytest.raises(rb.PublicationLocked):
        rb._acquire_lock(tmp_path, "t2-lockme", "test")
    rb._release_lock(handle)


def test_acquire_lock_recovers_local_stale_lock(tmp_path, monkeypatch):
    rb._acquire_lock(tmp_path, "t2-lockme", "first")
    monkeypatch.setattr(rb, "_pid_alive", lambda pid: False)  # simulate the owning pid having died
    handle2 = rb._acquire_lock(tmp_path, "t2-lockme", "second")
    assert handle2.payload.transition == "second"
    stale_copies = list(tmp_path.glob("t2-lockme.lock.stale-*"))
    assert len(stale_copies) == 1
    rb._release_lock(handle2)


def test_acquire_lock_never_auto_breaks_remote_lock(tmp_path, monkeypatch):
    rb._acquire_lock(tmp_path, "t2-lockme", "first")
    monkeypatch.setattr(rb, "_boot_id", lambda: "a-different-boot")
    with pytest.raises(rb.PublicationLockedRemote):
        rb._acquire_lock(tmp_path, "t2-lockme", "second")


def test_break_remote_lock_clears_and_copies_aside(tmp_path):
    rb._acquire_lock(tmp_path, "t2-lockme", "first")
    rb.break_remote_lock(tmp_path, "t2-lockme")
    assert not rb._lock_path(tmp_path, "t2-lockme").exists()
    assert list(tmp_path.glob("t2-lockme.lock.broken-*"))
    with pytest.raises(FileNotFoundError):
        rb.break_remote_lock(tmp_path, "t2-lockme")


def test_held_lock_releases_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with rb._held_lock(tmp_path, "t2-lockme", "test"):
            raise RuntimeError("boom")
    assert not rb._lock_path(tmp_path, "t2-lockme").exists()


def test_create_payload_tombstone_recovery_after_crash(tmp_path):
    def crashing_build(staging_dir: Path) -> None:
        raise RuntimeError("simulated crash mid-build")

    with pytest.raises(RuntimeError):
        rb.create_payload(bundle_parent=tmp_path, bundle_id="t2-crashy", build_payload=crashing_build)
    # no half-written bundle directory left behind, staging cleaned up
    assert not (tmp_path / "t2-crashy").exists()
    assert list(tmp_path.glob("t2-crashy.staging-*")) == []

    def real_build(staging_dir: Path) -> None:
        rb.atomic_new_file(staging_dir / "payload_manifest.json", b'{"ok": true}')

    bundle_dir = rb.create_payload(bundle_parent=tmp_path, bundle_id="t2-crashy", build_payload=real_build)
    assert (bundle_dir / "payload" / "payload_manifest.json").read_bytes() == b'{"ok": true}'


# ===========================================================================
# Sec 10.2/10.10 -- finalize_bundle() / open_bundle() / from_bundle() /
# refinalize_reporting_from_bundle() end-to-end, over a REAL (hand-
# orchestrated, in-process) small pipeline run. This reuses
# `tests/nodes/test_reporting_node.py::_run_full_pipeline_to_reporting`
# rather than re-implementing it -- that helper already produces every
# durable, on-disk artifact `finalize_bundle()`'s raw-layer collectors
# read (AIUseEvents, selection-ledger records, screening-batch results,
# an extraction bundle, FT_SCREENING native provenance, and a rendered
# `reporting/export_manifest.json`), loaded here via `importlib` (not a
# package import -- neither `tests/` nor `tests/nodes/` is a package).
# ===========================================================================

import importlib.util


def _load_reporting_node_harness():
    path = Path(__file__).parent / "nodes" / "test_reporting_node.py"
    spec = importlib.util.spec_from_file_location("_reporting_node_harness_for_bundle_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_reporting_node_harness()


def _finalize_from_harness(
    harness_mod, monkeypatch, tmp_path, *, review_type: str = "scoping",
    reproduction_artifacts=None,
):
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type=review_type)
    # Lossless round-trip (the same one `initial_state()` itself performs
    # when first storing `cfg.to_dict()`) -- `output_dir` and every other
    # field `finalize_bundle()` reads off `cfg` survive it unchanged, so
    # this is a faithful stand-in for "the SAME ReviewConfig the driver
    # ran with" without needing `_run_full_pipeline_to_reporting()` to
    # also return its own local `cfg` (it doesn't).
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "violations.jsonl").write_text("", encoding="utf-8")
    (workspace / "run_mode.json").write_text(
        json.dumps({"llm_transport": "replay", "search_source": "fixture"}), encoding="utf-8")
    (workspace / "run_mode_statement.md").write_text("# Run mode\n\nReplay/fixture mode (test harness).\n",
                                                       encoding="utf-8")

    source = _source_info(mode_profile="track2_offline")
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track2_offline",
        fixtures=_FIXTURES,
        reproduction_artifacts=reproduction_artifacts,
    )
    return bundle_dir, state, cfg


def _finalize_track3_live_from_harness(harness_mod, monkeypatch, tmp_path, *, review_type: str = "scoping"):
    """`track3_live` sibling of `_finalize_from_harness()`: same real
    pipeline run, but shaped like an actual live workspace -- no
    `reference_reporting_context` (a live run never sets it), no
    `call_log.jsonl`/`violations.jsonl` (netguard/cassette-specific, never
    written by a live run per plan Unit C3), `cost_cap_call_log.jsonl`/
    `cost_cap_ledger.json` instead, and `fixtures=None` (a live bundle has
    no synthetic fixture corpus to stage)."""
    def _track3_live_review_config(*args, **kwargs):
        kwargs.setdefault("dedup", DedupConfig(mode="exact"))
        return ReviewConfig(*args, **kwargs)

    monkeypatch.setattr(harness_mod, "ReviewConfig", _track3_live_review_config)
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type=review_type)
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "cost_cap_call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "cost_cap_ledger.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode.json").write_text(
        json.dumps({"mode_profile": "track3_live", "llm_transport": "live", "search_source": "live"}),
        encoding="utf-8")
    (workspace / "run_mode_statement.md").write_text(
        "# Run mode\n\nLive mode (test harness stub).\n", encoding="utf-8")

    source = _source_info(mode_profile="track3_live", execution_subtree_provenance="retrospective_reconstruction",
                           execution_subtree_clean=None, fixture_corpus_digest=None, cassette_sha256=None)
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track3_live",
    )
    return bundle_dir, state, cfg


def test_finalize_bundle_track2_offline_requires_fixtures(monkeypatch, tmp_path, harness):
    """Track-2 non-regression: `finalize_bundle()` must still reject a
    `track2_offline` call with no fixtures, exactly as it always required
    fixtures to be supplied at all before `fixtures` became `Optional`."""
    state = harness._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="scoping")
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]
    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "terminal_graph_snapshot.json").write_text(json.dumps({
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }), encoding="utf-8")
    (workspace / "call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "violations.jsonl").write_text("", encoding="utf-8")
    (workspace / "run_mode.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode_statement.md").write_text("# Run mode\n", encoding="utf-8")
    source = _source_info(mode_profile="track2_offline")
    with pytest.raises(rb.ReferenceBundleError, match="requires fixtures"):
        rb.finalize_bundle(
            workspace=workspace, cfg=cfg, review_run_id=review_run_id,
            reference_spec_sha256=_hex("test-reference-spec"), source=source,
            bundle_parent=tmp_path / "bundles", mode_profile="track2_offline", fixtures=None,
        )


def test_finalize_bundle_track2_offline_unchanged(monkeypatch, tmp_path, harness):
    """Track-2 non-regression: the exact existing `_finalize_from_harness()`
    path (fixtures supplied, mode_profile='track2_offline') still builds a
    bundle whose `fixture_inputs/` tree is staged, unchanged from before
    `finalize_bundle()` became mode-aware."""
    bundle_dir, state, cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    assert bundle.payload_manifest.mode_profile == "track2_offline"
    fixture_logical_paths = [a.logical_path for a in bundle.payload_manifest.artifacts if a.kind == "fixture_input"]
    assert fixture_logical_paths, "track2_offline must still stage fixture_inputs/"
    assert all(a.path.startswith("fixture_inputs/") for a in bundle.payload_manifest.artifacts
               if a.kind == "fixture_input")


def test_finalize_bundle_track3_live_rejects_fixtures(monkeypatch, tmp_path, harness):
    state = harness._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="scoping")
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]
    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "terminal_graph_snapshot.json").write_text(json.dumps({
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }), encoding="utf-8")
    (workspace / "cost_cap_call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "cost_cap_ledger.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode_statement.md").write_text("# Run mode\n", encoding="utf-8")
    source = _source_info(mode_profile="track3_live", execution_subtree_provenance="retrospective_reconstruction",
                           execution_subtree_clean=None, fixture_corpus_digest=None, cassette_sha256=None)
    with pytest.raises(rb.ReferenceBundleError, match="must not receive fixtures"):
        rb.finalize_bundle(
            workspace=workspace, cfg=cfg, review_run_id=review_run_id,
            reference_spec_sha256=_hex("test-reference-spec"), source=source,
            bundle_parent=tmp_path / "bundles", mode_profile="track3_live", fixtures=_FIXTURES,
        )


def test_finalize_bundle_track3_live_never_stages_fixture_inputs(monkeypatch, tmp_path, harness):
    """Track-3 fixture-absence: a real `track3_live` `finalize_bundle()`
    call (no `fixtures` argument) never stages a `fixture_inputs/` tree at
    all, and reads live-run-shaped evidence (`cost_cap_call_log.jsonl`,
    `cost_cap_ledger.json`) rather than the Track-2 replay filenames."""
    bundle_dir, state, cfg = _finalize_track3_live_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    assert bundle.payload_manifest.mode_profile == "track3_live"
    assert not any(a.kind == "fixture_input" for a in bundle.payload_manifest.artifacts)
    assert not any(a.path.startswith("fixture_inputs/") for a in bundle.payload_manifest.artifacts)
    run_evidence_logical_paths = {
        a.logical_path for a in bundle.payload_manifest.artifacts if a.logical_path.startswith("slr://run_evidence/")
    }
    assert str(LogicalLocator("run_evidence", "cost_cap", "call_log")) in run_evidence_logical_paths
    assert str(LogicalLocator("run_evidence", "cost_cap", "ledger")) in run_evidence_logical_paths
    assert str(LogicalLocator("run_evidence", "replay", "call_log")) not in run_evidence_logical_paths
    assert str(LogicalLocator("run_evidence", "netguard", "violations")) not in run_evidence_logical_paths


def test_finalize_bundle_open_bundle_round_trip(monkeypatch, tmp_path, harness):
    bundle_dir, state, cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    # open_bundle() itself enforces the raw-layer-inventory-exact property
    # (every physical `payload/` file appears in `artifacts` exactly once,
    # and vice versa) -- a mismatch there raises `ReferenceBundleError`
    # before this line ever returns.
    bundle = rb.open_bundle(bundle_dir / "payload")
    assert bundle.bundle_id.startswith("t2-")
    assert bundle.payload_manifest.mode_profile == "track2_offline"
    assert len(bundle.payload_manifest.artifacts) > 0


def test_finalize_bundle_rejects_second_publish_of_same_bundle_id(monkeypatch, tmp_path, harness):
    bundle_dir, state, cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    with pytest.raises(rb.PublicationConflict):
        rb.create_payload(bundle_parent=bundle_dir.parent, bundle_id=bundle.bundle_id,
                           build_payload=lambda staging_dir: None)


def test_from_bundle_reconstructs_matching_review_run_id_and_records(monkeypatch, tmp_path, harness):
    bundle_dir, state, cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    snapshot = rb.from_bundle(bundle)

    assert snapshot.review_run_id == state["review_run_id"]
    assert [r.id for r in snapshot.screening_records] == [r["id"] for r in state["deduplicated_records"]]
    assert [r.id for r in snapshot.full_text_records] == [r["id"] for r in state["full_text_records"]]
    assert snapshot.prisma.model.included == state["prisma_counts"]["included"]
    assert snapshot.synthesis_output.model.claims, "expected at least one reconstructed synthesis claim"
    assert snapshot.extraction is not None
    assert snapshot.reference_context is None  # this harness never sets reference_reporting_context


def test_built_from_present_on_every_present_renderer_artifact(monkeypatch, tmp_path, harness):
    """Every `renderer_artifact` entry the reporting stage actually
    rendered (`status == "present"`) must carry a non-empty `built_from`
    provenance list -- the frozen per-renderer contract table's own
    schema-level guarantee (Sec 10.3/10.10), re-checked here against a
    REAL rendered manifest rather than a hand-built one."""
    bundle_dir, state, cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    present_renderer_entries = [
        e for e in bundle.reporting_stage_manifest.entries
        if e.entry_kind == "renderer_artifact" and e.status == "present"
    ]
    assert present_renderer_entries, "expected at least one rendered artifact"
    for entry in present_renderer_entries:
        assert entry.built_from, f"{entry.logical_path} is present but built_from is empty/None"


def test_refinalize_reporting_from_bundle_reproduces_reporting_tree(monkeypatch, tmp_path, harness):
    """Relocates the bundle's `payload/` tree to a fresh directory (a
    stand-in for moving it to a different machine/mount), refinalizes the
    reporting stage from scratch against ONLY that relocated raw layer,
    and confirms the freshly re-rendered artifacts are byte-for-byte
    identical to what the bundle already carries under `reporting/` --
    proof the raw layer is a complete, sufficient source for the whole
    reporting phase. `PayloadManifestV1` itself must be untouched by the
    refinalize call."""
    import shutil

    bundle_dir, state, cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    original_bundle = rb.open_bundle(bundle_dir / "payload")
    original_manifest_bytes = (bundle_dir / "payload" / "payload_manifest.json").read_bytes()

    relocated_root = tmp_path / "relocated_payload"
    shutil.copytree(bundle_dir / "payload", relocated_root)

    out_dir = tmp_path / "refinalized_reporting"
    rb.refinalize_reporting_from_bundle(relocated_root, out_dir=out_dir)

    reporting_artifacts = [a for a in original_bundle.payload_manifest.artifacts if a.path.startswith("reporting/")]
    assert reporting_artifacts, "expected at least one reporting/ artifact in the original bundle"
    for artifact in reporting_artifacts:
        filename = Path(artifact.path).name
        original_bytes = (bundle_dir / "payload" / artifact.path).read_bytes()
        refinalized_bytes = (out_dir / filename).read_bytes()
        if filename == "ai_traceability.json":
            # `full_text_eligibility_native_provenance.ft_screening_artifact_paths`
            # is, BY DESIGN, not byte-stable across a relocate+refinalize:
            # `_from_bundle_ft_native_provenance()` resolves those paths via
            # `bundle.resolve(...)`, i.e. the ft_screening raw files' CURRENT
            # location under the (relocated) bundle's own `raw/ft_screening/`
            # tree -- not the transient host path the original live pipeline
            # run happened to write to. Re-anchoring to the bundle's own
            # current location is the correct, portable behavior; requiring
            # literal path identity would instead demand the refinalized
            # provenance point at a location that may no longer exist. Assert
            # the two renders diverge ONLY in that path re-anchoring (and its
            # derived `content_hash` self-hash), and that the refinalized
            # paths do point inside the relocated bundle's own raw layer.
            original_json = json.loads(original_bytes)
            refinalized_json = json.loads(refinalized_bytes)
            original_paths = original_json["full_text_eligibility_native_provenance"]["ft_screening_artifact_paths"]
            refinalized_paths = refinalized_json["full_text_eligibility_native_provenance"]["ft_screening_artifact_paths"]
            for key, refinalized_path in refinalized_paths.items():
                assert Path(refinalized_path).is_relative_to(relocated_root), (
                    f"refinalized ft_screening_artifact_paths.{key} must resolve under the relocated bundle's "
                    f"own raw layer, got {refinalized_path!r}")
            normalized_original = dict(original_json,
                full_text_eligibility_native_provenance=dict(
                    original_json["full_text_eligibility_native_provenance"],
                    ft_screening_artifact_paths={k: None for k in original_paths}),
                content_hash=None)
            normalized_refinalized = dict(refinalized_json,
                full_text_eligibility_native_provenance=dict(
                    refinalized_json["full_text_eligibility_native_provenance"],
                    ft_screening_artifact_paths={k: None for k in refinalized_paths}),
                content_hash=None)
            assert normalized_refinalized == normalized_original, (
                "ai_traceability.json differs after relocate+refinalize beyond the expected "
                "ft_screening_artifact_paths re-anchoring")
            continue
        if filename == "ai_traceability.md":
            # Same expected divergence as ai_traceability.json, carried into
            # the .md rendering: it embeds only the trailing `content_hash:`
            # line (see `_render_ai_traceability_md()`), never the raw
            # ft_screening_artifact_paths themselves, so strip that one line
            # from both sides before comparing.
            original_lines = original_bytes.decode("utf-8").splitlines()
            refinalized_lines = refinalized_bytes.decode("utf-8").splitlines()
            assert original_lines[-1].startswith("content_hash: `")
            assert refinalized_lines[-1].startswith("content_hash: `")
            assert refinalized_lines[:-1] == original_lines[:-1], (
                "ai_traceability.md differs after relocate+refinalize beyond the expected content_hash line")
            continue
        if filename == "export_manifest.json":
            # Cascades from the same ai_traceability re-anchoring: export_manifest.json
            # records each renderer artifact's own `file_sha256`, and
            # ai_traceability.json/.md's bytes legitimately changed above, so
            # their two entries' `file_sha256` legitimately change too.
            # Normalize only those two entries' `file_sha256` before comparing.
            original_json = json.loads(original_bytes)
            refinalized_json = json.loads(refinalized_bytes)
            reanchored_logical_paths = {
                "slr://reporting/artifact/ai_traceability_json",
                "slr://reporting/artifact/ai_traceability_md",
            }
            for doc in (original_json, refinalized_json):
                for entry in doc["entries"]:
                    if entry.get("logical_path") in reanchored_logical_paths:
                        entry["file_sha256"] = None
                doc["manifest_hash"] = None  # self-hash over `entries`, cascades from the file_sha256 edits above
            assert refinalized_json == original_json, (
                "export_manifest.json differs after relocate+refinalize beyond the expected "
                "ai_traceability entries' file_sha256")
            continue
        assert refinalized_bytes == original_bytes, f"{filename} differs after relocate+refinalize"

    # payload_manifest.json itself is never touched by a refinalize call.
    assert (relocated_root / "payload_manifest.json").read_bytes() == original_manifest_bytes


def test_extraction_review_artifacts_survive_bundle_round_trip(monkeypatch, tmp_path, harness):
    """Section 3.7: packet/snapshot/index remain verified bundle components."""
    from slr_pipeline.extraction.resolution_suggestions import (
        render_review_packet, review_packet_json_bytes, review_packet_markdown_bytes,
        suggestions_index_bytes,
    )
    from slr_pipeline.extraction.hashing import canonical_json_bytes
    from slr_pipeline.extraction.verified_bundle import load_and_verify_bundle
    from tests.extraction_resolution.test_review_packet import _snapshot_from_bundle

    original_loader = rb._load_extraction_bundle

    def loader_with_review_artifacts(state, cfg):
        run_id = state.get("extraction_run_id")
        if run_id:
            store, verified, schema, _schema_hash = load_and_verify_bundle(state, run_id)
            snapshot = _snapshot_from_bundle(verified, schema)
            packet = render_review_packet(verified, schema, snapshot)
            (store.run_dir / "extraction_review_packet.json").write_bytes(review_packet_json_bytes(packet))
            (store.run_dir / "extraction_review_packet.md").write_bytes(review_packet_markdown_bytes(packet))
            (store.run_dir / "review_queue_snapshot.json").write_bytes(
                canonical_json_bytes(snapshot.model_dump(mode="json"))
            )
            (store.run_dir / "extraction_resolution_suggestions.json").write_bytes(
                suggestions_index_bytes((), extraction_run_id=run_id)
            )
        return original_loader(state, cfg)

    monkeypatch.setattr(rb, "_load_extraction_bundle", loader_with_review_artifacts)
    bundle_dir, _state, _cfg = _finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    snapshot = rb.from_bundle(bundle)
    assert snapshot.extraction is not None
    review = snapshot.extraction.extraction_review
    assert review is not None
    assert review.review_packet_json is not None
    assert review.review_packet_md is not None
    assert review.queue_snapshot is not None
    assert review.suggestions_index is not None
    logical_paths = {artifact.logical_path for artifact in bundle.payload_manifest.artifacts}
    xid = snapshot.extraction.run_id
    assert str(LogicalLocator("extraction", "review_packet", xid, "json")) in logical_paths
    assert str(LogicalLocator("extraction", "review_packet", xid, "md")) in logical_paths
    assert str(LogicalLocator("extraction", "queue_snapshot", xid)) in logical_paths
    assert str(LogicalLocator("extraction", "suggestions_index", xid)) in logical_paths


def _build_reproduction_artifacts(root: Path):
    from slr_pipeline.evaluation.berg2023_gates import evaluate_berg2023_gates
    from slr_pipeline.evaluation.bundle_artifacts import ReproductionArtifactPaths
    from slr_pipeline.evaluation.evaluator import evaluate_pipeline
    from slr_pipeline.evaluation.gold_standard import GoldStandard
    from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
    from slr_pipeline.evaluation.stage_corpus import StageCorpus
    from slr_pipeline.state import Record

    record = Record(
        id="repro-record-1",
        source="pubmed",
        source_id="1",
        doi="10.1000/reproduction",
        title="A reproducible study",
        year=2020,
    )
    gold = GoldStandard.model_validate({
        "review_id": "bundle-reproduction",
        "review_citation": "Example review",
        "included_papers": [{
            "paper_id": "",
            "doi": record.doi,
            "title": record.title,
            "year": record.year,
        }],
    })
    spec = ReproductionSpec.model_validate({
        "reproduction_mode": "search_reproduction",
        "source_mode": "archived_exports",
        "citation": "Example review",
        "original_databases": ["PubMed"],
        "reproduced_databases": ["PubMed"],
        "exact_queries": {},
        "search_dates": {"PubMed": "2023-01-01"},
        "filters": [],
        "manual_additions": [],
        "reported_prisma_counts": {},
        "unknowns": [],
        "review_id": gold.review_id,
        "review_type": "scoping review",
        "title_abstract_eligibility_criteria": "criterion",
        "full_text_eligibility_criteria": "criterion",
        "gold_standard_ref": "gold_standard.json",
        "expected_gold_standard_hash": gold.content_hash(),
    })
    records = (record,)
    corpus = StageCorpus.build(
        raw_records=records,
        deduplicated_records=records,
        screened_records=records,
        full_text_records=records,
        included_records=records,
    )
    evaluation = evaluate_pipeline(spec, gold, corpus)
    gate_report = evaluate_berg2023_gates(evaluation, spec, gold, corpus)

    artifact_dir = root / "reproduction_artifacts"
    artifact_dir.mkdir()
    paths = ReproductionArtifactPaths(
        reproduction_spec=artifact_dir / "reproduction_spec.json",
        gold_standard=artifact_dir / "gold_standard.json",
        stage_corpus=artifact_dir / "stage_corpus.json",
        pipeline_evaluation=artifact_dir / "pipeline_evaluation.json",
        gate_report=artifact_dir / "gate_report.json",
        adjudication_journal=artifact_dir / "evaluation_adjudications.jsonl",
    )
    models = {
        "reproduction_spec": spec,
        "gold_standard": gold,
        "stage_corpus": corpus,
        "pipeline_evaluation": evaluation,
        "gate_report": gate_report,
    }
    for name, model in models.items():
        getattr(paths, name).write_text(
            model.model_dump_json(indent=2), encoding="utf-8"
        )
    paths.adjudication_journal.write_bytes(b"")
    return paths


def test_reproduction_artifacts_survive_bundle_round_trip_and_torn_journal_fails_closed(
    monkeypatch, tmp_path, harness
):
    artifacts = _build_reproduction_artifacts(tmp_path)
    bundle_dir, _state, _cfg = _finalize_from_harness(
        harness,
        monkeypatch,
        tmp_path,
        reproduction_artifacts=artifacts,
    )
    bundle = rb.open_bundle(bundle_dir / "payload")
    snapshot = rb.from_bundle(bundle)

    assert snapshot.reproduction_evaluation is not None
    assert snapshot.extraction is not None
    assert (
        snapshot.extraction.reproduction_evaluation
        == snapshot.reproduction_evaluation
    )
    for subkind in (
        "spec",
        "gold_standard",
        "stage_corpus",
        "pipeline_evaluation",
        "gate_report",
        "adjudication_journal",
    ):
        bundle.resolve(
            str(LogicalLocator("reproduction", subkind, snapshot.review_run_id))
        )

    journal = snapshot.reproduction_evaluation.adjudication_journal.path
    original = journal.read_bytes()
    torn_bytes = original + b'{"torn":'
    journal.write_bytes(torn_bytes)
    journal_locator = str(LogicalLocator(
        "reproduction", "adjudication_journal", snapshot.review_run_id
    ))
    updated_artifacts = [
        artifact.model_copy(update={"sha256": hashlib.sha256(torn_bytes).hexdigest()})
        if artifact.logical_path == journal_locator
        else artifact
        for artifact in bundle.payload_manifest.artifacts
    ]
    parser_test_bundle = dataclasses.replace(
        bundle,
        payload_manifest=bundle.payload_manifest.model_copy(
            update={"artifacts": updated_artifacts}
        ),
    )
    with pytest.raises(rb.ReferenceBundleError, match="torn final line"):
        rb.from_bundle(parser_test_bundle)
    assert journal.read_bytes() == torn_bytes