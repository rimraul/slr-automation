"""
Track 2 plan (`track2_pipeline_readiness_plan_2026-08-30.md`) step 15
([C14]): the bundle writer + publication/lock lifecycle + `open_bundle`/
`from_bundle`/`refinalize_reporting_from_bundle`.

This module is the SOLE owner of `PayloadManifestV1` (Sec 10.1) -- no other
module ever constructs one. Its job is split into four layers, in the order
they are used by a real release lifecycle:

  1. Sec 10.1(a) -- the shared strict models (`PayloadArtifact`,
     `PayloadSourceInfo`, `PayloadManifestV1`) plus the Sec 10.6
     `bundle_id`/`run:` locator computation and resolution.
  2. Sec 10.2 -- `finalize_bundle()`, run by the driver AFTER
     `graph.invoke()` + `graph.get_state()` + replay teardown: walks the
     completed reference-run workspace + the live durable stores + the
     fixed fixture-input locations + `reporting_node`'s own output, and
     stages the complete `payload/` tree.
  3. Sec 10.7/10.8 -- the release-guard sidecar schemas
     (`AcceptanceReport`/`ReviewerAttestation`/`ReleaseManifest`/
     `RunRegistry`), `verify_release_structure()`/`assert_evidentiary_
     release()`, and the atomic-publish + per-bundle lock lifecycle
     (`atomic_new_file`, `create_payload`/`attach_acceptance`/
     `attach_attestation`/`finalize_release`).
  4. Sec 10.10 -- `open_bundle()`, `from_bundle()`, and
     `refinalize_reporting_from_bundle()`.

Scoping note (carried from the step-15 planning discussion, not restated in
the plan verbatim but load-bearing for every function in layer 4): `from_
bundle()` is deliberately VERIFICATION-ONLY. `capture_reporting_snapshot()`
(`slr_pipeline.nodes.reporting`) performs deep, LIVE cross-validation --
re-deriving oversight_mode/automation_origin from scratch, reconstructing
and comparing the evidence map, checking BERTopic consistency, and running
`load_and_verify_synthesis_output()`'s own external cross-check -- all of
which needs live `Store` objects pointed at `cfg.ensure_output_dir()`'s own
directory-naming convention, a convention the bundle's flattened `raw/`
tree does not (and, by design, need not) share. That content was already
fully, semantically validated ONCE, live, by `capture_reporting_snapshot()`
itself, before the very `reporting/*` renderer outputs bundled here were
ever produced from it. `from_bundle()` therefore re-establishes TAMPER-
EVIDENCE over already-validated content (hash/schema/self-consistency
checks, mirroring each field's own existing verification primitive) rather
than re-deriving semantic correctness from first principles.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace as _dataclasses_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, conlist, constr, field_validator, model_validator

from ft_screening import ScreeningItemResult, ScreeningRunResult

from slr_pipeline.ai_traceability import AIUseEvent, AIUseEventStore
from slr_pipeline.config import ReviewConfig
from slr_pipeline.evaluation.bundle_artifacts import (
    ReproductionArtifactPaths, verify_reproduction_artifact_set,
)
from slr_pipeline.export_manifest import ReportingStageManifest
from slr_pipeline.extraction.checkpoint import DocumentCheckpoint, ExtractionCheckpointStore, RunManifest, VerifiedRunBundle
from slr_pipeline.extraction.hashing import canonical_json_bytes, file_sha256, sha256_hex
from slr_pipeline.extraction.verified_bundle import load_schema
from slr_pipeline.locators import (
    LOCATOR_GRAMMAR_VERSION,
    POSIX_RELPATH,
    RUN_LOCATOR_VERSION,
    LocatorGlob,
    LogicalLocator,
    _parse_and_check_locator,
)
from slr_pipeline.nodes.extraction import _build_extraction_evidence_index
from slr_pipeline.nodes.reporting import (
    FILENAMES,
    EvidenceMapModel,
    ReportingConfigProjection,
    ReportingSourceSnapshot,
    ReproductionEvaluationArtifacts,
    SourceBundle,
    VerifiedComponent,
    VerifiedFile,
    VerifiedProjection,
    ValidatedStateValue,
    _hash_value,
    _load_current_screening_batches,
    _load_extraction_bundle,
    _load_verified_events,
    _load_verified_ledger_records,
    _load_extraction_review_artifacts,
    _state_value,
    _validate_list_of_dicts,
    _validate_screening_qc_state,
    capture_ft_native_provenance,
)
from slr_pipeline.nodes.screening import ScreeningBatchResultStore, compute_current_screening_batches
from slr_pipeline.nodes.synthesis import canonicalize_corpus_order
from slr_pipeline.reference_spec import ReferenceReportingContext
from slr_pipeline.replay.signature import rfc8785_jcs
from slr_pipeline.selection_ledger import SelectionDecisionRecord, SelectionDecisionStore, _key_digest
from slr_pipeline.state import PRISMACounts, Record, SearchLog, SynthesisOutput

logger = logging.getLogger(__name__)


class ReferenceBundleError(RuntimeError):
    """A raw-layer artifact is missing, malformed, or fails its own self-
    verification during `finalize_bundle()`'s collection pass or
    `open_bundle()`/`from_bundle()`'s reload."""


# ===========================================================================
# Sec 10.1(a) -- shared strict primitives + payload manifest models
# ===========================================================================

def _check_iso8601_z(v: str) -> str:
    """The regex on `ISO8601_Z` already enforces the lexical shape --
    this AfterValidator additionally rejects a lexically-valid but
    calendar-invalid value (e.g. `2025-02-30T00:00:00Z`)."""
    try:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{v!r} is not a calendar-valid ISO8601 UTC timestamp") from exc
    return v


ISO8601_Z = Annotated[
    constr(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$"),
    AfterValidator(_check_iso8601_z),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PayloadArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: constr(pattern=r"^[a-z_]+-[a-z_]+-[0-9a-f]{16}$")
    logical_path: str
    path: POSIX_RELPATH
    sha256: constr(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal[
        "application/json", "application/jsonl", "text/markdown", "text/csv",
        "text/plain", "image/png", "application/pdf", "application/octet-stream",
    ]
    producer_stage: str
    kind: Literal["durable_reverified", "graph_state_snapshot", "fixture_input", "bundle_generated"]

    @field_validator("logical_path")
    @classmethod
    def _logical_path_well_formed(cls, v: str) -> str:
        return _parse_and_check_locator(v)


class FTScreeningSourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    filename: constr(pattern=r"^ft_screening-[\w.]+-py3-none-any\.whl$")
    sha256: constr(pattern=r"^[0-9a-f]{64}$")
    licence_commit: constr(pattern=r"^[0-9a-f]{40}$")
    code_parent: Literal["9604dc7"]
    release_tag: Literal["track2-assets-v1"]


class PayloadSourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    git_commit: constr(pattern=r"^[0-9a-f]{40}$")
    execution_subtree_digest: constr(pattern=r"^[0-9a-f]{64}$")
    # `execution_subtree_provenance`/`execution_subtree_clean` distinguish a
    # working-tree-clean check performed LIVE, at run time (`"runtime_
    # verified"` -- the only value this codebase's own build guard,
    # `scripts/build_slr_pipeline_wheel.py::_assert_canonical_working_tree()`,
    # can ever produce, since it raises rather than let a dirty build
    # proceed, so `execution_subtree_clean` is always `True` in that branch,
    # never `False`) from a digest recomputed AFTER THE FACT from `git`'s own
    # object store at a recorded commit (`"retrospective_reconstruction"` --
    # this can prove what that commit's subtree contains, but NOT whether
    # the working tree was clean while some earlier, unrecorded run actually
    # executed). `execution_subtree_clean` is `Optional` specifically so a
    # retrospective bundle records "unknown", never a fabricated `True`.
    execution_subtree_provenance: Literal["runtime_verified", "retrospective_reconstruction"]
    execution_subtree_clean: Optional[Literal[True]] = None
    python_version: Literal["3.12.12"]
    runner_image_os: str
    runner_image_version: str
    base_requirements_lock_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    build_requirements_lock_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    ft_screening_wheel_lock_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    slr_pipeline_wheel_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    workflow_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    provision_script_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    build_ft_wheel_script_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    author_cassette_script_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    ft_screening: FTScreeningSourceInfo
    docling_assets_manifest_digest: constr(pattern=r"^[0-9a-f]{64}$")
    docling_archive_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    docling_model_manifest_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    # `fixture_corpus_digest`/`cassette_sha256` are mode-aware for the same
    # reason `reference_run_mode.RunModeDocument` already made its own
    # same-named fields `Optional` (reused precedent, not a new
    # convention): a `track3_live` run has no synthetic fixture corpus and
    # no replay cassette to hash in the first place.
    fixture_corpus_digest: Optional[constr(pattern=r"^[0-9a-f]{64}$")] = None
    cassette_sha256: Optional[constr(pattern=r"^[0-9a-f]{64}$")] = None
    reference_spec_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    rfc8785_version: Literal["0.1.4"]
    mode_profile: Literal["track2_offline", "track3_live"]

    @model_validator(mode="after")
    def _mode_aware_fields_consistent(self) -> "PayloadSourceInfo":
        if self.mode_profile == "track2_offline":
            if self.execution_subtree_provenance != "runtime_verified":
                raise ValueError(
                    "track2_offline requires execution_subtree_provenance == 'runtime_verified' -- "
                    "the CI build guard always verifies working-tree cleanliness live before building")
            if self.execution_subtree_clean is not True:
                raise ValueError("track2_offline requires execution_subtree_clean is True")
            if self.fixture_corpus_digest is None or self.cassette_sha256 is None:
                raise ValueError(
                    "track2_offline requires both fixture_corpus_digest and cassette_sha256 to be present")
        elif self.mode_profile == "track3_live":
            if self.execution_subtree_clean is not None:
                raise ValueError(
                    "track3_live must leave execution_subtree_clean as None -- a live run cannot "
                    "cryptographically prove the working tree was clean while it executed, only that a "
                    "named commit's subtree contains certain bytes; asserting True here would be dishonest")
            if self.fixture_corpus_digest is not None or self.cassette_sha256 is not None:
                raise ValueError(
                    "track3_live must leave both fixture_corpus_digest and cassette_sha256 as None -- a "
                    "live run has no synthetic fixture corpus and no replay cassette to hash")
        return self


class PayloadManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    payload_manifest_version: Literal["1"] = "1"
    bundle_id: constr(pattern=r"^t2-[0-9a-f]{24}$")
    mode_profile: Literal["track2_offline", "track3_live"]
    created_utc: ISO8601_Z
    source: PayloadSourceInfo
    reference_spec_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    artifacts: conlist(PayloadArtifact, min_length=1)
    payload_hash: constr(pattern=r"^[0-9a-f]{64}$")
    parent_bundle_id: Optional[constr(pattern=r"^t2-[0-9a-f]{24}$")] = None

    @model_validator(mode="after")
    def _artifacts_sorted_and_unique(self) -> "PayloadManifestV1":
        logical_paths = [a.logical_path for a in self.artifacts]
        if logical_paths != sorted(logical_paths):
            raise ValueError("artifacts must be sorted by logical_path")
        if len(set(logical_paths)) != len(logical_paths):
            raise ValueError("artifacts contains duplicate logical_path values")
        if len({a.artifact_id for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("artifacts contains duplicate artifact_id values")
        if len({a.path for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("artifacts contains duplicate path values")
        if self.source.mode_profile != self.mode_profile:
            raise ValueError(
                f"source.mode_profile ({self.source.mode_profile!r}) != mode_profile ({self.mode_profile!r})")
        return self

    def resolve(self, logical_path_or_artifact_id: str, payload_root: Path) -> Path:
        """Looks up one artifact by `logical_path` or `artifact_id`,
        containment-checks its `path` against `payload_root`, verifies its
        recorded `sha256` against the live file bytes, and returns the
        resolved `Path`. Raises on any mismatch -- never returns an
        unverified path."""
        for artifact in self.artifacts:
            if logical_path_or_artifact_id in (artifact.logical_path, artifact.artifact_id):
                root = payload_root.resolve()
                resolved = (root / artifact.path).resolve()
                if not resolved.is_relative_to(root):
                    raise ReferenceBundleError(f"artifact path {artifact.path!r} escapes payload_root {root}")
                if not resolved.is_file():
                    raise ReferenceBundleError(f"artifact {artifact.logical_path!r} not found at {resolved}")
                actual = file_sha256(resolved)
                if actual != artifact.sha256:
                    raise ReferenceBundleError(
                        f"artifact {artifact.logical_path!r} sha256 mismatch: manifest says "
                        f"{artifact.sha256!r}, on-disk file is {actual!r}")
                return resolved
        raise KeyError(f"{logical_path_or_artifact_id!r} not found in payload_manifest.artifacts")


def _sha256_rfc8785_jcs(obj: Any) -> str:
    return hashlib.sha256(rfc8785_jcs(obj)).hexdigest()


# ===========================================================================
# Sec 10.6 -- bundle_id + `run:` locator grammar/resolution
# ===========================================================================

BUNDLE_ID_RE = re.compile(r"^t2-[0-9a-f]{24}$")


def compute_bundle_id(*, reference_spec_sha256: str, review_run_id: str, created_utc: str) -> str:
    digest = _sha256_rfc8785_jcs({
        "reference_spec_sha256": reference_spec_sha256,
        "review_run_id": review_run_id,
        "created_utc": created_utc,
    })
    return f"t2-{digest[:24]}"


_RUN_LOCATOR_RE = re.compile(
    r"^run:(?P<bundle_id>t2-[0-9a-f]{24})@(?P<payload_hash>[0-9a-f]{64})/"
    r"(?P<artifact_id>[a-z_]+-[a-z_]+-[0-9a-f]{16})(?:#(?P<pointer>.+))?$"
)


class RunLocatorError(ValueError):
    """Malformed `run:` locator, or a well-formed one whose pointer does
    not resolve against the artifact it addresses."""


@dataclass(frozen=True)
class RunLocator:
    bundle_id: str
    payload_hash: str
    artifact_id: str
    pointer: Optional[str] = None

    def __str__(self) -> str:
        base = f"run:{self.bundle_id}@{self.payload_hash}/{self.artifact_id}"
        return f"{base}#{self.pointer}" if self.pointer else base


def parse_run_locator(value: str) -> RunLocator:
    m = _RUN_LOCATOR_RE.match(value)
    if m is None:
        raise RunLocatorError(f"{value!r} is not a well-formed run: locator")
    return RunLocator(bundle_id=m.group("bundle_id"), payload_hash=m.group("payload_hash"),
                       artifact_id=m.group("artifact_id"), pointer=m.group("pointer"))


def _unescape_json_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def resolve_json_pointer(obj: Any, pointer: str) -> Any:
    """RFC 6901 JSON Pointer resolution. `pointer == ""` returns `obj`
    itself (the whole-document pointer)."""
    if pointer == "":
        return obj
    if not pointer.startswith("/"):
        raise RunLocatorError(f"malformed JSON pointer {pointer!r}: must start with '/'")
    current = obj
    for raw_token in pointer[1:].split("/"):
        token = _unescape_json_pointer_token(raw_token)
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise RunLocatorError(f"malformed JSON pointer array index {token!r}")
            idx = int(token)
            if idx >= len(current):
                raise RunLocatorError(f"JSON pointer index {idx} out of range (len={len(current)})")
            current = current[idx]
        elif isinstance(current, dict):
            if token not in current:
                raise RunLocatorError(f"JSON pointer key {token!r} not found")
            current = current[token]
        else:
            raise RunLocatorError(f"cannot descend into pointer segment {token!r} of a {type(current).__name__}")
    return current


_LINE_POINTER_RE = re.compile(r"^L(?P<a>[1-9][0-9]*)(-(?P<b>[1-9][0-9]*))?$")
_JSONL_POINTER_RE = re.compile(r"^/(?P<line>[1-9][0-9]*)(?P<rest>/.*)?$")


def resolve_pointer(*, media_type: str, data: bytes, pointer: Optional[str]) -> bytes:
    """Sec 10.6's media-specific pointer rules, applied to the already-
    resolved raw bytes of one artifact. `pointer is None` returns `data`
    unchanged (the whole-file address)."""
    if pointer is None:
        return data
    if media_type == "application/json":
        obj = json.loads(data)
        return canonical_json_bytes(resolve_json_pointer(obj, pointer))
    if media_type == "application/jsonl":
        m = _JSONL_POINTER_RE.match(pointer)
        if m is None:
            raise RunLocatorError(f"malformed JSONL pointer {pointer!r} -- expected /<1-based-line>[<rfc6901>]")
        line_no = int(m.group("line"))
        lines = data.splitlines()
        if line_no > len(lines):
            raise RunLocatorError(f"JSONL pointer line {line_no} out of range (1..{len(lines)})")
        line_obj = json.loads(lines[line_no - 1])
        return canonical_json_bytes(resolve_json_pointer(line_obj, m.group("rest") or ""))
    if media_type in ("text/markdown", "text/csv"):
        m = _LINE_POINTER_RE.match(pointer)
        if m is None:
            raise RunLocatorError(f"malformed line-range pointer {pointer!r} -- expected L<n> or L<a>-<b>")
        a, b = int(m.group("a")), int(m.group("b")) if m.group("b") else int(m.group("a"))
        if b < a:
            raise RunLocatorError(f"line range {pointer!r}: end before start")
        lines = data.decode("utf-8").splitlines(keepends=True)
        if a < 1 or b > len(lines):
            raise RunLocatorError(f"line range {pointer!r} out of range (1..{len(lines)})")
        return "".join(lines[a - 1:b]).encode("utf-8")
    raise RunLocatorError(f"media_type {media_type!r} does not accept a pointer")


def resolve_run_locator(value: str, *, payload_manifest: PayloadManifestV1, payload_root: Path) -> bytes:
    loc = parse_run_locator(value)
    if loc.bundle_id != payload_manifest.bundle_id:
        raise RunLocatorError(f"locator bundle_id {loc.bundle_id!r} != payload_manifest.bundle_id "
                               f"{payload_manifest.bundle_id!r}")
    if loc.payload_hash != payload_manifest.payload_hash:
        raise RunLocatorError("locator payload_hash does not match payload_manifest.payload_hash")
    artifact = next((a for a in payload_manifest.artifacts if a.artifact_id == loc.artifact_id), None)
    if artifact is None:
        raise RunLocatorError(f"artifact_id {loc.artifact_id!r} not found in payload_manifest")
    path = payload_manifest.resolve(loc.artifact_id, payload_root)
    return resolve_pointer(media_type=artifact.media_type, data=path.read_bytes(), pointer=loc.pointer)


# ===========================================================================
# Sec 10.7 -- release-guard sidecar/registry schemas + verification
# ===========================================================================

class AcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: int
    status: Literal["pass", "fail"]
    detail: str


class AcceptanceReport(BaseModel):
    """`acceptance_report.json`. No `bundle_id`/`created_utc` field -- the
    acceptance run only attests to a `payload_hash` + `source`; binding
    that attestation to a specific `bundle_id` is `ReleaseManifest`'s job.
    No `prisma_scr_semantic_review` field either -- semantic PRISMA-ScR
    conformance is a human judgment call, recorded exclusively on
    `ReviewerAttestation.decision` (Sec 10.7)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    acceptance_report_version: Literal["1"] = "1"
    payload_hash: constr(pattern=r"^[0-9a-f]{64}$")
    mode_profile: Literal["track2_offline", "track3_live"]
    source: PayloadSourceInfo
    checks: conlist(AcceptanceCheck, min_length=8, max_length=8)
    acceptance_report_hash: constr(pattern=r"^[0-9a-f]{64}$")

    def self_hash(self) -> str:
        return _sha256_rfc8785_jcs(self.model_dump(exclude={"acceptance_report_hash"}))


class ReviewerAttestation(BaseModel):
    """`reviewer_attestation.json`. `decision` **is** the PRISMA-ScR
    semantic-review verdict -- there is no separate field for it anywhere
    on `AcceptanceReport`."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    reviewer_attestation_version: Literal["1"] = "1"
    payload_hash: constr(pattern=r"^[0-9a-f]{64}$")
    acceptance_report_hash: constr(pattern=r"^[0-9a-f]{64}$")
    prisma_scr_report_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    prisma_scr_conformance_sha256: constr(pattern=r"^[0-9a-f]{64}$")
    reviewer: str
    decision: Literal["APPROVED", "REJECTED"]
    notes: str
    attested_utc: ISO8601_Z
    attestation_hash: constr(pattern=r"^[0-9a-f]{64}$")

    def self_hash(self) -> str:
        return _sha256_rfc8785_jcs(self.model_dump(exclude={"attestation_hash"}))


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_manifest_version: Literal["1"] = "1"
    bundle_id: constr(pattern=r"^t2-[0-9a-f]{24}$")
    mode_profile: Literal["track2_offline", "track3_live"]
    payload_hash: constr(pattern=r"^[0-9a-f]{64}$")
    acceptance_report_hash: constr(pattern=r"^[0-9a-f]{64}$")
    attestation_hash: constr(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["APPROVED", "REJECTED"]
    release_hash: constr(pattern=r"^[0-9a-f]{64}$")

    def self_hash(self) -> str:
        return _sha256_rfc8785_jcs(self.model_dump(exclude={"release_hash"}))


class RunRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    payload_hash: constr(pattern=r"^[0-9a-f]{64}$")
    release_hash: constr(pattern=r"^[0-9a-f]{64}$")
    acceptance_report_hash: constr(pattern=r"^[0-9a-f]{64}$")
    attestation_hash: constr(pattern=r"^[0-9a-f]{64}$")
    mode_profile: Literal["track2_offline", "track3_live"]
    decision: Literal["APPROVED", "REJECTED"]


class RunRegistry(BaseModel):
    """`track2/run_registry.json`. Committed as the EMPTY template
    (`run_registry_version="1", runs={}`) -- Track 2 CI never writes to
    the real, durable registry; every test targets a temporary external
    registry path instead (Sec 10.7)."""
    model_config = ConfigDict(extra="forbid")
    run_registry_version: Literal["1"] = "1"
    runs: Dict[constr(pattern=r"^t2-[0-9a-f]{24}$"), RunRegistryEntry] = Field(default_factory=dict)


class ReleaseStructureInvalid(Exception):
    """Raised by `verify_release_structure()` -- any of the 6 named
    checks failed (id-set completeness, the semantic-review-field
    exclusion, hash-chain equality across sidecars, self-hash integrity,
    release_manifest/registry cross-agreement, or the 4-way identity
    binding)."""


class ReleaseNotApproved(Exception):
    def __init__(self, bundle_id: str):
        super().__init__(f"release {bundle_id} is structurally valid but its decision is not APPROVED")
        self.bundle_id = bundle_id


class NotEvidentiary(Exception):
    def __init__(self, bundle_id: str, mode_profile: str):
        super().__init__(f"bundle {bundle_id} has mode_profile={mode_profile!r}, not 'track3_live' -- "
                          "an offline/replay-mode bundle can never be asserted evidentiary")
        self.bundle_id = bundle_id
        self.mode_profile = mode_profile


class SyntheticEvidenceClaim(Exception):
    """Raised by `assert_not_synthetic_evidence()` -- a `mode_profile ==
    'track3_live'` bundle's own `run_mode.json` still shows a replay/
    fixture transport, i.e. the release's evidentiary claim is false on
    its face."""
    def __init__(self, bundle_id: str, llm_transport: Any, search_source: Any):
        super().__init__(f"bundle {bundle_id} claims track3_live but run_mode.json shows "
                          f"llm_transport={llm_transport!r}, search_source={search_source!r}")
        self.bundle_id = bundle_id
        self.llm_transport = llm_transport
        self.search_source = search_source


@dataclass(frozen=True)
class Bundle:
    """A published bundle directory (`<bundle_id>/`), as seen by
    `open_bundle()`. The four top-level sidecars are `None` until their
    own publication transition has run -- `verify_release_structure()`
    requires all four to be present, `open_bundle()` does not."""
    bundle_id: str
    payload_root: Path
    bundle_dir: Path
    payload_manifest: PayloadManifestV1
    reporting_stage_manifest: ReportingStageManifest
    acceptance_report: Optional[AcceptanceReport] = None
    reviewer_attestation: Optional[ReviewerAttestation] = None
    release_manifest: Optional[ReleaseManifest] = None
    run_registry_entry: Optional[RunRegistryEntry] = None
    run_registry_key: Optional[str] = None

    def file_sha256(self, logical_path_or_relpath_or_artifact_id: str) -> str:
        for artifact in self.payload_manifest.artifacts:
            if logical_path_or_relpath_or_artifact_id in (artifact.path, artifact.logical_path, artifact.artifact_id):
                return artifact.sha256
        raise KeyError(f"{logical_path_or_relpath_or_artifact_id!r} not found in payload_manifest.artifacts")

    def file_sha256_or_none(self, logical_path_or_relpath_or_artifact_id: str) -> Optional[str]:
        """Non-raising sibling of `file_sha256()` -- used where a missing
        artifact is a legitimate outcome (a `track3_live` bundle never
        stages the Track-2-only PRISMA-ScR files) rather than an error.
        `file_sha256()`'s own raise-on-miss contract is left untouched for
        its other callers."""
        try:
            return self.file_sha256(logical_path_or_relpath_or_artifact_id)
        except KeyError:
            return None

    def resolve(self, logical_path_or_artifact_id: str) -> Path:
        return self.payload_manifest.resolve(logical_path_or_artifact_id, self.payload_root)


# A permanent "this artifact does not apply to this run's mode_profile"
# sentinel for the two PRISMA-ScR attestation hash fields on a
# `track3_live` bundle, reusing this codebase's existing all-zero-hex64
# convention (`reference_acceptance.py`'s `acceptance_report_hash="0"*64`,
# this module's own `payload_hash="0"*64` placeholder below). Those two
# existing uses are *transient* placeholders overwritten once the real
# hash is computed, not a *permanent* not-applicable marker -- reusing the
# same literal for a semantically different, permanent meaning is a real
# naming/collision risk, flagged here deliberately for reviewer scrutiny
# rather than presented as a clean precedent match.
PRISMA_SCR_NOT_APPLICABLE_SHA256 = "0" * 64


def verify_release_structure(bundle: Bundle) -> None:
    """Sec 10.7's 6-check release guard, mode-profile-AWARE: true for ANY
    correctly-formed release, `track2_offline` or `track3_live`, but check
    3's PRISMA-ScR sub-checks take a different shape per mode (a
    `track3_live` bundle never stages those Track-2-only files -- see
    `PRISMA_SCR_NOT_APPLICABLE_SHA256` above). Every check is an explicit
    `raise`, never a bare `assert` (Python silently strips those under
    `-O`/`PYTHONOPTIMIZE`, which would make a structural-integrity guard
    non-fail-closed in that mode). A structurally-valid REJECTED release
    passes every check below and is rejected ONLY by check 6's own
    `ReleaseNotApproved` -- there is no earlier `decision != "APPROVED"`
    shortcut, so a genuinely audit-closed REJECTED bundle stays
    distinguishable from a malformed one."""
    pm, ar = bundle.payload_manifest, bundle.acceptance_report
    att, rel, reg = bundle.reviewer_attestation, bundle.release_manifest, bundle.run_registry_entry
    if ar is None or att is None or rel is None or reg is None:
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id} is missing one of acceptance_report/reviewer_attestation/"
            "release_manifest/run_registry_entry -- a release requires all four sidecars")

    # 1. exactly ids 1-8, each once, all pass.
    ids = sorted(c.id for c in ar.checks)
    if ids != list(range(1, 9)):
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: acceptance_report.checks must be exactly ids 1..8")
    if not all(c.status == "pass" for c in ar.checks):
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: every acceptance_report check must be status=='pass'")

    # 2. the semantic-review verdict lives only on the attestation, never on
    # the acceptance report -- a SCHEMA/location check (already structurally
    # impossible under AcceptanceReport's own extra="forbid", re-asserted
    # here defensively), never a verdict-value check.
    if hasattr(ar, "prisma_scr_semantic_review"):
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: acceptance_report must not carry a "
                                       "semantic-review field")

    # 3. hash-chain equality (cross-sidecar: two sidecars' STORED hash fields agree).
    if ar.payload_hash != pm.payload_hash:
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: acceptance_report.payload_hash != payload_manifest.payload_hash -- "
            "the 8 acceptance checks must have run against THIS payload's bytes, not some other payload's")
    if att.acceptance_report_hash != ar.acceptance_report_hash:
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: reviewer_attestation.acceptance_report_hash != "
            "acceptance_report.acceptance_report_hash")
    if att.payload_hash != pm.payload_hash:
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: reviewer_attestation.payload_hash != payload_manifest.payload_hash")
    if pm.mode_profile == "track2_offline":
        report_sha = bundle.file_sha256_or_none("reporting/prisma_scr_report.md")
        if report_sha is None:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: reporting/prisma_scr_report.md missing from "
                "payload_manifest.artifacts on a track2_offline bundle")
        if att.prisma_scr_report_sha256 != report_sha:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: reviewer_attestation.prisma_scr_report_sha256 does not match "
                "the bundled reporting/prisma_scr_report.md")
        conformance_sha = bundle.file_sha256_or_none("reporting/prisma_scr_conformance.json")
        if conformance_sha is None:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: reporting/prisma_scr_conformance.json missing from "
                "payload_manifest.artifacts on a track2_offline bundle")
        if att.prisma_scr_conformance_sha256 != conformance_sha:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: reviewer_attestation.prisma_scr_conformance_sha256 does not "
                "match the bundled reporting/prisma_scr_conformance.json")
    elif pm.mode_profile == "track3_live":
        if bundle.file_sha256_or_none("reporting/prisma_scr_report.md") is not None:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: track3_live bundle must not contain "
                "reporting/prisma_scr_report.md")
        if att.prisma_scr_report_sha256 != PRISMA_SCR_NOT_APPLICABLE_SHA256:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: reviewer_attestation.prisma_scr_report_sha256 must be the "
                "not-applicable sentinel on a track3_live bundle")
        if bundle.file_sha256_or_none("reporting/prisma_scr_conformance.json") is not None:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: track3_live bundle must not contain "
                "reporting/prisma_scr_conformance.json")
        if att.prisma_scr_conformance_sha256 != PRISMA_SCR_NOT_APPLICABLE_SHA256:
            raise ReleaseStructureInvalid(
                f"bundle {bundle.bundle_id}: reviewer_attestation.prisma_scr_conformance_sha256 must be "
                "the not-applicable sentinel on a track3_live bundle")
    else:
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: unknown mode_profile {pm.mode_profile!r} -- no release-guard "
            "rule defined for it")

    # 3a. self-hash integrity -- step 3 alone only ever compares one
    # sidecar's STORED hash field against another's STORED hash field;
    # nothing there recomputes a hash FROM a sidecar's own content, so
    # editing e.g. att.reviewer/notes/attested_utc without touching
    # att.attestation_hash would pass every check above. Recompute and
    # compare each sidecar's own `*_hash` field against a fresh hash of
    # its own remaining content.
    if ar.acceptance_report_hash != ar.self_hash():
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: acceptance_report.acceptance_report_hash does not match its own "
            "recomputed content hash")
    if att.attestation_hash != att.self_hash():
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: reviewer_attestation.attestation_hash does not match its own "
            "recomputed content hash")
    if rel.release_hash != rel.self_hash():
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: release_manifest.release_hash does not match its own recomputed "
            "content hash")

    # 4. release_manifest equals upstream, decision equality.
    if rel.payload_hash != pm.payload_hash:
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: release_manifest.payload_hash != "
                                       "payload_manifest.payload_hash")
    if rel.acceptance_report_hash != ar.acceptance_report_hash:
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: release_manifest.acceptance_report_hash != "
                                       "acceptance_report.acceptance_report_hash")
    if rel.attestation_hash != att.attestation_hash:
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: release_manifest.attestation_hash != "
                                       "reviewer_attestation.attestation_hash")
    if rel.decision != att.decision:
        raise ReleaseStructureInvalid(f"bundle {bundle.bundle_id}: release_manifest.decision != "
                                       "reviewer_attestation.decision")

    # 5. registry entry equals the sidecars.
    expected_reg = RunRegistryEntry(
        payload_hash=pm.payload_hash, release_hash=rel.release_hash,
        acceptance_report_hash=ar.acceptance_report_hash, attestation_hash=att.attestation_hash,
        mode_profile=rel.mode_profile, decision=rel.decision,
    )
    if reg.model_dump() != expected_reg.model_dump():
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: run_registry entry {reg.model_dump()!r} != the sidecars "
            f"{expected_reg.model_dump()!r}")

    # 6. bind identity fields to each other, not just to a hash -- catches a
    # bundle that is internally hash-consistent but describes a mismatched
    # mode/source/registry-key combination.
    if not (ar.mode_profile == pm.mode_profile == rel.mode_profile):
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: acceptance/payload/release mode_profile must be the single same value")
    if ar.source != pm.source:
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: the acceptance run must attest to the SAME source as the payload claims")
    if not (bundle.run_registry_key == rel.bundle_id == pm.bundle_id):
        raise ReleaseStructureInvalid(
            f"bundle {bundle.bundle_id}: the registry key an attacker could rename must equal the bundle's "
            "own declared identity")

    if rel.decision != "APPROVED":
        raise ReleaseNotApproved(bundle.bundle_id)   # REJECTED -> audit-closed, permanent, never evidentiary


def _load_run_mode_document(bundle: Bundle) -> Dict[str, Any]:
    path = bundle.resolve("slr://bundle/run_mode/main")
    return json.loads(path.read_bytes())


def assert_not_synthetic_evidence(bundle: Bundle) -> None:
    run_mode = _load_run_mode_document(bundle)
    llm_transport = run_mode.get("llm_transport")
    search_source = run_mode.get("search_source")
    if llm_transport == "replay" or search_source == "fixture":
        raise SyntheticEvidenceClaim(bundle.bundle_id, llm_transport, search_source)


def assert_evidentiary_release(bundle: Bundle) -> None:
    """Composes `verify_release_structure()` with the two additional
    Track-3-only guarantees: `mode_profile == "track3_live"`, and the
    bundle's own `run_mode.json` does not itself disclose replay/fixture
    transports (a bundle cannot claim to be evidentiary while its own
    run-mode statement says otherwise)."""
    verify_release_structure(bundle)
    if bundle.release_manifest.mode_profile != "track3_live":
        raise NotEvidentiary(bundle.bundle_id, bundle.release_manifest.mode_profile)
    assert_not_synthetic_evidence(bundle)


# ===========================================================================
# Sec 10.8 -- atomic publish primitive + per-bundle lock lifecycle
# ===========================================================================

class PublicationConflict(Exception):
    """A `create_payload`/`attach_*`/`finalize_release` transition tried
    to publish a file that already exists (no-replace violated)."""


class PublicationLocked(Exception):
    """A live, same-host, same-boot, alive-pid lock is already held for
    this bundle_id."""


class PublicationLockedRemote(Exception):
    """A lock is held by a different host, or the same host after a
    reboot -- never auto-broken; requires the operator-invoked
    `break-remote-lock` subcommand."""


def atomic_new_file(path: Path, data: bytes) -> None:
    """Sec 10.8's 4-step algorithm: write to a uuid4-suffixed sibling tmp
    file (create-exclusive, flush, fsync, close) fully BEFORE attempting
    the atomic create-or-fail publish (`os.link`); on a `FileExistsError`
    (something else already holds `path`), delete the tmp file and raise
    `PublicationConflict`; on success, best-effort-remove the tmp file
    (its content now also lives at `path` via the hardlink) and read back
    + verify `path`'s own bytes before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        os.link(str(tmp), str(path))
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        raise PublicationConflict(f"{path} already exists -- no-replace violated")
    finally:
        tmp.unlink(missing_ok=True)
    read_back = path.read_bytes()
    if hashlib.sha256(read_back).digest() != hashlib.sha256(data).digest():
        raise RuntimeError(f"atomic_new_file: {path} readback does not match the bytes just written")


_TMP_RE = re.compile(r"^\..+\.tmp\.[0-9a-f]{32}$")


def _recovery_sweep(root: Path, *, not_after_utc: str) -> None:
    """Deletes stale `.<name>.tmp.<uuid>` files under `root`, but ONLY
    ones older than `not_after_utc` -- called while the caller holds the
    sole live lock for this bundle_id, so any qualifying tmp file found
    here belongs to a crashed PRIOR attempt by the same lock holder, never
    a foreign in-flight write."""
    if not root.exists():
        return
    cutoff = datetime.fromisoformat(not_after_utc.replace("Z", "+00:00")).timestamp()
    for p in root.rglob("*"):
        if p.is_file() and _TMP_RE.match(p.name) and p.stat().st_mtime < cutoff:
            logger.warning("recovery sweep: removing stale tmp file %s", p)
            p.unlink(missing_ok=True)


def _boot_id() -> str:
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if boot_id_path.exists():
        try:
            return boot_id_path.read_text().strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            # ctypes defaults an unconfigured function's return type to
            # c_int (32-bit signed) -- GetTickCount64 is a real 64-bit
            # ULONGLONG, and system uptime does exceed 2**31 ms (~24.85
            # days) on any long-running host. Without this, the value
            # truncates/sign-flips past that point.
            ctypes.windll.kernel32.GetTickCount64.restype = ctypes.c_uint64
            uptime_ms = ctypes.windll.kernel32.GetTickCount64()
            # Rounded (not floored) to the nearest 60s bucket -- ordinary
            # per-call jitter between GetTickCount64() and now() must not
            # perturb the boot identity two calls within the SAME boot
            # session return. A hard bucket boundary (whether floor or
            # round) can in principle still be straddled by two calls whose
            # raw boot_epoch values happen to land on opposite sides of it,
            # but round() to a 60s (not 10s) bucket needs the jitter to be
            # within seconds of a multiple of 60 to flip, several orders of
            # magnitude less likely than landing near a multiple of 10.
            boot_epoch = datetime.now(timezone.utc).timestamp() - uptime_ms / 1000.0
            return f"win-{round(boot_epoch / 60)}"
        except (OSError, AttributeError):
            pass
    # Last-resort fallback: stable for the lifetime of this process (not
    # genuinely per-boot), better than raising.
    return f"unknown-{platform.node()}"


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        # Same c_int-default truncation risk as _boot_id()'s GetTickCount64:
        # HANDLE is pointer-sized. Set restype/argtypes explicitly rather
        # than let ctypes guess a 32-bit int return.
        OpenProcess = ctypes.windll.kernel32.OpenProcess
        OpenProcess.restype = ctypes.c_void_p
        OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        CloseHandle = ctypes.windll.kernel32.CloseHandle
        CloseHandle.restype = ctypes.c_int
        CloseHandle.argtypes = [ctypes.c_void_p]

        handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _LockPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_pid: int
    owner_host: str
    owner_boot_id: str
    acquired_utc: ISO8601_Z
    transition: str


def _lock_path(bundle_parent: Path, bundle_id: str) -> Path:
    return bundle_parent / f"{bundle_id}.lock"


@dataclass(frozen=True)
class _LockHandle:
    bundle_parent: Path
    bundle_id: str
    payload: _LockPayload


def _acquire_lock(bundle_parent: Path, bundle_id: str, transition: str, *, _max_attempts: int = 8) -> _LockHandle:
    """Bounded retry loop, not a single try-once-then-decide sequence --
    every branch below that reads `lock_path`'s current state can be
    invalidated by another process between that read and this process's own
    next write, so every such race is handled by re-looping against fresh
    on-disk state rather than proceeding on a stale snapshot:
      - the conflicting lock owner may exit and unlink its lock between our
        failed `atomic_new_file()` and our `read_bytes()` of it (the file is
        simply gone by the time we look -- loop back to a plain acquire
        attempt);
      - during local stale-lock recovery, a second process may win the same
        race and re-acquire the lock between our dead-pid determination and
        our `unlink()` -- re-read the lock file's bytes immediately before
        unlinking and abort the recovery (re-looping) if they no longer
        match what we staleness-checked, rather than blindly deleting
        whatever is there now."""
    lock_path = _lock_path(bundle_parent, bundle_id)
    this_host, this_boot, this_pid = socket.gethostname(), _boot_id(), os.getpid()
    payload = _LockPayload(owner_pid=this_pid, owner_host=this_host, owner_boot_id=this_boot,
                            acquired_utc=_now_iso(), transition=transition)
    data = canonical_json_bytes(payload.model_dump(mode="json"))

    for _attempt in range(_max_attempts):
        try:
            atomic_new_file(lock_path, data)
            return _LockHandle(bundle_parent, bundle_id, payload)
        except PublicationConflict:
            pass
        try:
            existing_bytes = lock_path.read_bytes()
        except FileNotFoundError:
            # The conflicting lock's owner released it between our failed
            # atomic_new_file() above and this read -- it may now be free.
            continue
        existing = _LockPayload.model_validate(json.loads(existing_bytes))
        if existing.owner_host == this_host and existing.owner_boot_id == this_boot:
            if _pid_alive(existing.owner_pid):
                raise PublicationLocked(
                    f"bundle {bundle_id} is locked by this host's own live pid {existing.owner_pid} "
                    f"(transition={existing.transition!r})")
            # Local stale recovery -- same host, same boot, but the owning
            # pid is dead: safe to reclaim, PROVIDED the lock file is still
            # the exact one we just staleness-checked at the moment we
            # unlink it.
            stale_aside = bundle_parent / f"{bundle_id}.lock.stale-{uuid.uuid4().hex}"
            shutil.copy2(lock_path, stale_aside)
            try:
                if lock_path.read_bytes() != existing_bytes:
                    # Someone else already recovered/replaced this lock
                    # since we staleness-checked it -- don't touch it;
                    # re-loop and reassess whatever is there now.
                    stale_aside.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                stale_aside.unlink(missing_ok=True)
                continue
            lock_path.unlink()
            logger.warning("recovered stale local lock for %s (dead pid %s)", bundle_id, existing.owner_pid)
            try:
                atomic_new_file(lock_path, data)
                return _LockHandle(bundle_parent, bundle_id, payload)
            except PublicationConflict:
                # Another process won the re-acquire race right after our
                # unlink -- re-loop and reassess.
                continue
        # Different host, OR same host after a reboot (owner_boot_id
        # differs) -- treated identically as remote: never auto-broken.
        raise PublicationLockedRemote(
            f"bundle {bundle_id} is locked by {existing.owner_host}/{existing.owner_boot_id} (pid "
            f"{existing.owner_pid}, transition={existing.transition!r}); once confirmed dead, run "
            f"`python -m slr_pipeline.reference_bundle break-remote-lock {bundle_id} "
            "--i-confirm-the-other-host-is-dead`"
        )
    raise PublicationLocked(
        f"bundle {bundle_id}: could not acquire the lock after {_max_attempts} attempts -- "
        "contended by other same-host processes racing the same recovery/acquire sequence")


def _release_lock(handle: _LockHandle) -> None:
    _lock_path(handle.bundle_parent, handle.bundle_id).unlink(missing_ok=True)


def break_remote_lock(bundle_parent: Path, bundle_id: str) -> None:
    lock_path = _lock_path(bundle_parent, bundle_id)
    if not lock_path.exists():
        raise FileNotFoundError(f"no lock file for {bundle_id} at {lock_path}")
    aside = bundle_parent / f"{bundle_id}.lock.broken-{uuid.uuid4().hex}"
    shutil.copy2(lock_path, aside)
    lock_path.unlink()
    logger.warning("operator broke remote lock for %s (prior lock copied aside to %s)", bundle_id, aside)


def _staging_cleanup(bundle_parent: Path, bundle_id: str, *, this_host: str, this_boot: str) -> None:
    for marker in bundle_parent.glob(f"{bundle_id}.staging-*.owner.json"):
        staging_dir = bundle_parent / marker.name[: -len(".owner.json")]
        try:
            owner = _LockPayload.model_validate(json.loads(marker.read_bytes()))
        except (OSError, ValueError):
            continue
        if owner.owner_host == this_host and owner.owner_boot_id == this_boot and not _pid_alive(owner.owner_pid):
            logger.warning("removing stale staging directory %s", staging_dir)
            shutil.rmtree(staging_dir, ignore_errors=True)
            marker.unlink(missing_ok=True)


@contextmanager
def _held_lock(bundle_parent: Path, bundle_id: str, transition: str):
    bundle_parent.mkdir(parents=True, exist_ok=True)
    handle = _acquire_lock(bundle_parent, bundle_id, transition)
    try:
        _recovery_sweep(bundle_parent / bundle_id, not_after_utc=handle.payload.acquired_utc)
        _staging_cleanup(bundle_parent, bundle_id, this_host=handle.payload.owner_host,
                          this_boot=handle.payload.owner_boot_id)
        yield handle
    finally:
        _release_lock(handle)


def _hardlink_capable(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".hardlink_probe.{uuid.uuid4().hex}"
    link = directory / f".hardlink_probe.{uuid.uuid4().hex}.link"
    try:
        probe.write_bytes(b"x")
        os.link(str(probe), str(link))
        return True
    except OSError:
        return False
    finally:
        probe.unlink(missing_ok=True)
        link.unlink(missing_ok=True)


def create_payload(*, bundle_parent: Path, bundle_id: str, build_payload: Callable[[Path], None]) -> Path:
    """Sec 10.8's `create_payload` transition. `build_payload(staging_dir)`
    is called with an empty directory to fill with the complete `payload/`
    tree (every file written via `atomic_new_file`, ending with
    `payload_manifest.json` itself); this function then performs the
    tombstone-safe `mkdir` + `rename` publish (a crash between the two
    leaves a recoverable, empty `<bundle_id>/` tombstone directory, never
    a half-written `payload/`)."""
    bundle_parent.mkdir(parents=True, exist_ok=True)
    if not _hardlink_capable(bundle_parent):
        raise RuntimeError(f"{bundle_parent} is not on a hardlink-capable filesystem")
    with _held_lock(bundle_parent, bundle_id, "create_payload") as handle:
        bundle_dir = bundle_parent / bundle_id
        if bundle_dir.exists():
            if (bundle_dir / "payload" / "payload_manifest.json").exists():
                raise PublicationConflict(f"bundle {bundle_id} is already published at {bundle_dir}")
            tombstone = bundle_parent / f"{bundle_id}.tombstone-{uuid.uuid4().hex}"
            logger.warning("recovering tombstone %s -> %s", bundle_dir, tombstone)
            bundle_dir.rename(tombstone)
        staging_dir = bundle_parent / f"{bundle_id}.staging-{uuid.uuid4().hex}"
        staging_dir.mkdir()
        owner_marker = staging_dir.parent / f"{staging_dir.name}.owner.json"
        atomic_new_file(owner_marker, canonical_json_bytes(handle.payload.model_dump(mode="json")))
        try:
            build_payload(staging_dir)
            bundle_dir.mkdir()  # the exclusive tombstone-claim point
            os.rename(str(staging_dir), str(bundle_dir / "payload"))
            owner_marker.unlink(missing_ok=True)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            owner_marker.unlink(missing_ok=True)
            raise
        return bundle_dir


def attach_acceptance(*, bundle_parent: Path, bundle_id: str, acceptance_report: AcceptanceReport) -> Path:
    with _held_lock(bundle_parent, bundle_id, "attach_acceptance"):
        bundle_dir = bundle_parent / bundle_id
        if not (bundle_dir / "payload" / "payload_manifest.json").exists():
            raise PublicationConflict(f"bundle {bundle_id} has no published payload yet")
        target = bundle_dir / "acceptance_report.json"
        if target.exists():
            raise PublicationConflict(f"{target} already published")
        atomic_new_file(target, canonical_json_bytes(acceptance_report.model_dump(mode="json")))
        return target


def attach_attestation(*, bundle_parent: Path, bundle_id: str, reviewer_attestation: ReviewerAttestation) -> Path:
    with _held_lock(bundle_parent, bundle_id, "attach_attestation"):
        bundle_dir = bundle_parent / bundle_id
        if not (bundle_dir / "acceptance_report.json").exists():
            raise PublicationConflict(f"bundle {bundle_id} has no acceptance_report.json yet")
        target = bundle_dir / "reviewer_attestation.json"
        if target.exists():
            raise PublicationConflict(f"{target} already published")
        atomic_new_file(target, canonical_json_bytes(reviewer_attestation.model_dump(mode="json")))
        return target


def _append_registry_entry(registry_path: Path, release_manifest: ReleaseManifest) -> None:
    """Test-only / Track-3-only: appends one entry to an EXTERNAL registry
    file -- never the committed `track2/run_registry.json` template
    itself. A temporary/external registry is single-writer in every real
    use here, so this relies on the caller's own bundle lock rather than
    a second, independent locking mechanism."""
    if registry_path.exists():
        registry = RunRegistry.model_validate(json.loads(registry_path.read_text(encoding="utf-8")))
    else:
        registry = RunRegistry()
    registry.runs[release_manifest.bundle_id] = RunRegistryEntry(
        payload_hash=release_manifest.payload_hash, release_hash=release_manifest.release_hash,
        acceptance_report_hash=release_manifest.acceptance_report_hash,
        attestation_hash=release_manifest.attestation_hash, mode_profile=release_manifest.mode_profile,
        decision=release_manifest.decision,
    )
    registry_path.write_text(
        json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finalize_release(*, bundle_parent: Path, bundle_id: str, release_manifest: ReleaseManifest,
                      registry_path: Optional[Path] = None) -> Path:
    with _held_lock(bundle_parent, bundle_id, "finalize_release"):
        bundle_dir = bundle_parent / bundle_id
        if not (bundle_dir / "reviewer_attestation.json").exists():
            raise PublicationConflict(f"bundle {bundle_id} has no reviewer_attestation.json yet")
        target = bundle_dir / "release_manifest.json"
        if target.exists():
            raise PublicationConflict(f"{target} already published")
        atomic_new_file(target, canonical_json_bytes(release_manifest.model_dump(mode="json")))
        if registry_path is not None:
            _append_registry_entry(registry_path, release_manifest)
        return target


# ===========================================================================
# Sec 10.2 -- fixture-input locations + raw-layer collection + finalize_bundle
# ===========================================================================

@dataclass(frozen=True)
class FixtureInputPaths:
    """No `.default()` classmethod (Track-2 packaging-boundary fix, CI run
    34155102694): a previous version fell back to `slr_pipeline.reference_
    run`'s own `Path(__file__)`-derived `REPO_ROOT`/`FIXTURES_SPEC_DIR`/
    `FIXTURES_CORPUS_DIR` constants, which assume `slr_pipeline` sits inside
    a git checkout with `tests/` as a sibling -- false once installed as a
    real, non-editable wheel. `finalize_bundle()`'s caller must always
    supply these paths explicitly, from wherever it legitimately knows the
    checkout layout (a repo-only script, or a test's own `Path(__file__)`-
    relative fixture location)."""
    spec_dir: Path
    corpus_dir: Path
    protocol_path: Path
    prisma_scr_matrix_path: Path


_MEDIA_TYPE_BY_SUFFIX = {
    ".json": "application/json", ".jsonl": "application/jsonl", ".md": "text/markdown",
    ".csv": "text/csv", ".txt": "text/plain", ".png": "image/png", ".pdf": "application/pdf",
    ".lock": "application/json",
}

_RENDERER_MEDIA_TYPES = {filename: _MEDIA_TYPE_BY_SUFFIX[Path(filename).suffix] for filename in FILENAMES.values()}


def _artifact_id(logical_path: str) -> str:
    parsed = LogicalLocator.parse(logical_path)
    digest = hashlib.sha1(logical_path.encode("utf-8")).hexdigest()[:16]
    return f"{parsed.kind}-{parsed.subkind}-{digest}"


def _stage_file(staging_dir: Path, *, source: Path, payload_relpath: str, logical_path: str,
                 kind: str, media_type: str, producer_stage: str) -> PayloadArtifact:
    data = source.read_bytes()
    atomic_new_file(staging_dir / payload_relpath, data)
    return PayloadArtifact(
        artifact_id=_artifact_id(logical_path), logical_path=logical_path, path=payload_relpath,
        sha256=hashlib.sha256(data).hexdigest(), media_type=media_type, producer_stage=producer_stage, kind=kind,
    )


def _collect_fixture_inputs(staging_dir: Path, *, fixtures: FixtureInputPaths) -> List[PayloadArtifact]:
    out: List[PayloadArtifact] = []
    for pdf_path in sorted((fixtures.corpus_dir / "pdfs").glob("*.pdf")):
        out.append(_stage_file(
            staging_dir, source=pdf_path, payload_relpath=f"fixture_inputs/corpus/pdfs/{pdf_path.name}",
            logical_path=str(LogicalLocator("fixture", "pdf", pdf_path.stem)),
            kind="fixture_input", media_type="application/pdf", producer_stage="fixture"))
    for native_path in sorted((fixtures.corpus_dir / "native").glob("*.json")):
        out.append(_stage_file(
            staging_dir, source=native_path, payload_relpath=f"fixture_inputs/corpus/native/{native_path.name}",
            logical_path=str(LogicalLocator("fixture", "native", native_path.stem)),
            kind="fixture_input", media_type="application/json", producer_stage="fixture"))
    # Sec 10.2's raw-layer table is the closed, authoritative row set --
    # `SCENARIO_TABLE.md` (present in the real fixture directory, a
    # human-readable index only, never hash-anchored or consumed by any
    # verification code) is deliberately NOT one of its rows and must
    # never be staged, or `test_raw_layer_inventory_exact` fails.
    _corpus_singletons = [
        ("CORPUS_ORACLE.json", "native", "corpus_oracle", "application/json"),
        ("LICENSE.md", "native", "corpus_license", "text/markdown"),
        ("PROVENANCE.md", "native", "corpus_provenance", "text/markdown"),
        ("known_real_doi_denylist.txt", "native", "corpus_doi_denylist", "text/plain"),
        ("replay_cassette.json", "cassette", "replay", "application/json"),
    ]
    for filename, subkind, locator_id, media_type in _corpus_singletons:
        path = fixtures.corpus_dir / filename
        out.append(_stage_file(
            staging_dir, source=path, payload_relpath=f"fixture_inputs/corpus/{filename}",
            logical_path=str(LogicalLocator("fixture", subkind, locator_id)),
            kind="fixture_input", media_type=media_type, producer_stage="fixture"))

    # Sec 10.2's exact `slr://fixture/spec/<id>` ids -- NOT a mechanical
    # stem-of-filename derivation (`reference_spec.json` -> id `reference`,
    # not `reference_spec`; `request_snapshots.lock` -> id
    # `request_snapshots_lock`, not `request_snapshots` -- `Path.stem`
    # only strips the LAST suffix, so it would silently drop `_lock`).
    _spec_singletons = [
        ("reference_spec.json", "reference"),
        ("reference_reporting_context.json", "reference_reporting_context"),
        ("request_snapshots.lock", "request_snapshots_lock"),
        ("extraction_schema.json", "extraction_schema"),
        ("extraction_model_config.json", "extraction_model_config"),
        ("ft_screening_config.json", "ft_screening_config"),
    ]
    for filename, locator_id in _spec_singletons:
        path = fixtures.spec_dir / filename
        out.append(_stage_file(
            staging_dir, source=path, payload_relpath=f"fixture_inputs/spec/{filename}",
            logical_path=str(LogicalLocator("fixture", "spec", locator_id)),
            kind="fixture_input", media_type="application/json", producer_stage="fixture"))
    for req_path in sorted(fixtures.spec_dir.glob("*.request.json")):
        surface = req_path.name[: -len(".request.json")]
        out.append(_stage_file(
            staging_dir, source=req_path, payload_relpath=f"fixture_inputs/spec/{req_path.name}",
            logical_path=str(LogicalLocator("fixture", "spec", f"{surface}_request")),
            kind="fixture_input", media_type="application/json", producer_stage="fixture"))
    out.append(_stage_file(
        staging_dir, source=fixtures.prisma_scr_matrix_path, payload_relpath="fixture_inputs/spec/prisma_scr_matrix.json",
        logical_path=str(LogicalLocator("fixture", "spec", "prisma_scr_matrix")),
        kind="fixture_input", media_type="application/json", producer_stage="fixture"))
    out.append(_stage_file(
        staging_dir, source=fixtures.protocol_path, payload_relpath="fixture_inputs/protocol/readiness_protocol.md",
        logical_path=str(LogicalLocator("fixture", "protocol", "readiness")),
        kind="fixture_input", media_type="text/markdown", producer_stage="fixture"))
    return out


def _collect_reporting(staging_dir: Path, *, output_dir: Path, review_run_id: str) -> List[PayloadArtifact]:
    report_dir = output_dir / "reporting" / review_run_id
    manifest_path = report_dir / "export_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = ReportingStageManifest.model_validate_json(manifest_bytes)
    reloaded = json.loads(manifest_bytes)
    recomputed = _hash_value({k: v for k, v in reloaded.items() if k != "manifest_hash"})
    if recomputed != manifest.manifest_hash:
        raise ReferenceBundleError(f"{manifest_path} failed its own manifest_hash self-check")

    out: List[PayloadArtifact] = []
    for entry in manifest.entries:
        if entry.entry_kind != "renderer_artifact" or entry.status != "present":
            continue
        filename = FILENAMES[entry.name]
        source_path = report_dir / filename
        actual = file_sha256(source_path)
        if actual != entry.file_sha256:
            raise ReferenceBundleError(
                f"{source_path} sha256 does not match export_manifest.json's own entry for {entry.name!r}")
        out.append(_stage_file(
            staging_dir, source=source_path, payload_relpath=f"reporting/{filename}",
            logical_path=str(LogicalLocator("reporting", "artifact", entry.name)),
            kind="bundle_generated", media_type=_RENDERER_MEDIA_TYPES[filename], producer_stage="reporting"))
    out.append(_stage_file(
        staging_dir, source=manifest_path, payload_relpath="reporting/export_manifest.json",
        logical_path=str(LogicalLocator("reporting", "artifact", "export_manifest")),
        kind="bundle_generated", media_type="application/json", producer_stage="reporting"))
    return out


def _collect_state_and_run_evidence(
    staging_dir: Path, *, workspace: Path, mode_profile: str = "track2_offline",
) -> List[PayloadArtifact]:
    out = [
        _stage_file(staging_dir, source=workspace / "terminal_state.json", payload_relpath="raw/state/terminal_state.json",
                    logical_path=str(LogicalLocator("state", "snapshot", "terminal_state")),
                    kind="graph_state_snapshot", media_type="application/json", producer_stage="driver"),
        _stage_file(staging_dir, source=workspace / "terminal_graph_snapshot.json",
                    payload_relpath="raw/state/terminal_graph_snapshot.json",
                    logical_path=str(LogicalLocator("state", "snapshot", "terminal_graph_snapshot")),
                    kind="graph_state_snapshot", media_type="application/json", producer_stage="driver"),
    ]
    if mode_profile == "track3_live":
        # A live run never arms `ReferenceNetGuard` (plan Unit C3 scope
        # decision) and has no replay cassette to log calls against --
        # the cost-cap ledger's own call log/ledger snapshot are the real
        # run evidence instead, and no `violations.jsonl` ever exists.
        out.append(_stage_file(
            staging_dir, source=workspace / "cost_cap_call_log.jsonl",
            payload_relpath="raw/run_evidence/cost_cap_call_log.jsonl",
            logical_path=str(LogicalLocator("run_evidence", "cost_cap", "call_log")),
            kind="bundle_generated", media_type="application/jsonl", producer_stage="driver"))
        out.append(_stage_file(
            staging_dir, source=workspace / "cost_cap_ledger.json",
            payload_relpath="raw/run_evidence/cost_cap_ledger.json",
            logical_path=str(LogicalLocator("run_evidence", "cost_cap", "ledger")),
            kind="bundle_generated", media_type="application/json", producer_stage="driver"))
        return out

    out.append(_stage_file(
        staging_dir, source=workspace / "call_log.jsonl", payload_relpath="raw/run_evidence/call_log.jsonl",
        logical_path=str(LogicalLocator("run_evidence", "replay", "call_log")),
        kind="bundle_generated", media_type="application/jsonl", producer_stage="driver"))
    violations_path = workspace / "violations.jsonl"
    if violations_path.read_text(encoding="utf-8").strip():
        raise ReferenceBundleError(f"{violations_path} is not empty -- netguard recorded a violation this run")
    out.append(_stage_file(
        staging_dir, source=violations_path, payload_relpath="raw/run_evidence/violations.jsonl",
        logical_path=str(LogicalLocator("run_evidence", "netguard", "violations")),
        kind="bundle_generated", media_type="application/jsonl", producer_stage="driver"))
    return out


def _collect_run_mode(staging_dir: Path, *, workspace: Path) -> List[PayloadArtifact]:
    return [
        _stage_file(staging_dir, source=workspace / "run_mode.json", payload_relpath="run_mode.json",
                    logical_path=str(LogicalLocator("bundle", "run_mode", "main")),
                    kind="bundle_generated", media_type="application/json", producer_stage="driver"),
        _stage_file(staging_dir, source=workspace / "run_mode_statement.md", payload_relpath="run_mode_statement.md",
                    logical_path=str(LogicalLocator("bundle", "run_mode", "statement")),
                    kind="bundle_generated", media_type="text/markdown", producer_stage="driver"),
    ]


def apply_parent_oversight_union(
    base_snapshot: ReportingSourceSnapshot, parent_snapshot: ReportingSourceSnapshot,
) -> ReportingSourceSnapshot:
    """Track 3 continuation plan (squishy-petting-starlight.md) Unit E --
    unions a continuation's own `oversight_mode_state`/`automation_origin_
    state`/`screening_qc_state` with its parent's. These 3 fields are
    embedded directly in `terminal_state.json` (never staged as separate,
    independently-re-verifiable bundle payload artifacts the way
    `ledger_records`/`events`/`current_screening_batch_results`/
    `ft_native_provenance` are), so they can only ever be made complete for
    a continuation via this snapshot-level union -- never via anything
    persisted in `state`/`terminal_state.json` itself. Used both at
    publish time (`build_cumulative_reporting_snapshot()`) and at
    acceptance-review time (`build_acceptance_report(parent_bundle=...)`),
    so the two never drift via duplicated merge logic."""
    oversight_mode_state = {**parent_snapshot.oversight_mode_state, **base_snapshot.oversight_mode_state}
    automation_origin_state = {**parent_snapshot.automation_origin_state, **base_snapshot.automation_origin_state}
    screening_qc_state = (
        base_snapshot.screening_qc_state if base_snapshot.screening_qc_state is not None
        else parent_snapshot.screening_qc_state
    )
    return _dataclasses_replace(
        base_snapshot,
        oversight_mode_state=oversight_mode_state,
        automation_origin_state=automation_origin_state,
        screening_qc_state=screening_qc_state,
    )


def finalize_bundle(
    *, workspace: Path, cfg: ReviewConfig, review_run_id: str, reference_spec_sha256: str,
    source: PayloadSourceInfo, bundle_parent: Path, fixtures: Optional[FixtureInputPaths] = None,
    mode_profile: str = "track2_offline",
    cumulative_snapshot: Optional[ReportingSourceSnapshot] = None,
    parent_bundle_id: Optional[str] = None,
    reproduction_artifacts: Optional[ReproductionArtifactPaths] = None,
) -> Path:
    """The sole `PayloadManifestV1` owner. Run by the driver AFTER
    `graph.invoke()` + `graph.get_state()` + replay teardown have already
    completed and written `workspace`'s §12 files (`terminal_state.json`,
    `terminal_graph_snapshot.json`, plus mode-specific run evidence --
    `call_log.jsonl`/`violations.jsonl` for `track2_offline`,
    `cost_cap_call_log.jsonl`/`cost_cap_ledger.json` for `track3_live`) --
    `reporting_node` has already run as part of the graph itself, so
    `cfg.ensure_output_dir()/"reporting"/review_run_id/` already holds the
    rendered artifacts this function copies (never re-renders).

    `cfg` MUST be the SAME `ReviewConfig` the reference-run child process
    ran with (its `output_dir` is the live, still-on-disk workspace this
    function reads raw durable artifacts from) -- unlike `from_bundle()`,
    this is never reconstructed from `terminal_state["config"]`, EXCEPT
    that a `track3_live` bundle has no separate fixture-derived config to
    diverge from `terminal_state.json`'s own `config` field in the first
    place -- the caller is expected to have built `cfg` from that same
    preserved `terminal_state.json`, not a fresh fixture spec.

    `fixtures` is mode-aware: required and non-`None` for `track2_offline`
    (Track-2 packaging-boundary fix, CI run 34155102694: this module is
    `slr_pipeline`-wheel-packaged and must not assume a checkout-relative
    fallback -- the caller supplies it explicitly), and required to be
    `None` for `track3_live` -- a live run has no synthetic fixture corpus
    to stage, and staging one would falsely claim the release depends on
    fixture inputs it never actually used.

    `parent_bundle_id` (Unit J): non-`None` only for a Track 3 continuation
    bundle. When given, `workspace / "continuation_lineage.json"` and
    `workspace / "preserved_decisions.json"` must already exist (the
    publish-time caller's own responsibility, mirroring every other §12
    workspace file this function stages but never generates) and are staged
    as `slr://continuation/lineage/main` / `slr://continuation/
    preserved_decisions/main`."""
    if mode_profile not in ("track2_offline", "track3_live"):
        raise ReferenceBundleError(f"finalize_bundle(): unrecognized mode_profile {mode_profile!r}")
    if mode_profile == "track2_offline" and fixtures is None:
        raise ReferenceBundleError("track2_offline finalize_bundle() requires fixtures")
    if mode_profile == "track3_live" and fixtures is not None:
        raise ReferenceBundleError("track3_live finalize_bundle() must not receive fixtures")
    output_dir = cfg.ensure_output_dir()
    terminal_state = json.loads((workspace / "terminal_state.json").read_bytes())
    created_utc = _now_iso()
    bundle_id = compute_bundle_id(
        reference_spec_sha256=reference_spec_sha256, review_run_id=review_run_id, created_utc=created_utc)

    def build_payload(staging_dir: Path) -> None:
        artifacts: List[PayloadArtifact] = []

        event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
        ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)

        # Unit E (squishy-petting-starlight.md): when `cumulative_snapshot`
        # is given (continuation mode only), these 4 fields are sourced
        # from it instead of a fresh, continuation-local-only
        # reconstruction -- `cumulative_snapshot` already carries the
        # verified parent+continuation union `continuation_reporting_node`'s
        # own rendered `export_manifest.json` was built from, so staging
        # anything else here would leave `open_bundle()`'s cross-check
        # unable to find a staged artifact for every manifest entry.
        # `current_screening_batch_results`, NOT a fresh local
        # `screening_records`-based reconstruction, is what's swapped in
        # here: `_load_current_screening_batches()` recomputes EXPECTED
        # batches from whatever `screening_records` it's given, and a
        # continuation's own `ScreeningBatchResultStore` never holds the
        # parent's batch files, so passing the cumulative (parent-inclusive)
        # `screening_records` into it here would guarantee a
        # `ReportingConsistencyError` -- `cumulative_snapshot.
        # current_screening_batch_results` is already fully computed and is
        # staged directly instead.
        if cumulative_snapshot is not None:
            event_vfs = cumulative_snapshot.events
            ledger_vfs = cumulative_snapshot.ledger_records
            current_batches = cumulative_snapshot.current_screening_batch_results
        else:
            screening_records = [Record.from_dict(r) for r in terminal_state["deduplicated_records"]]
            event_vfs = _load_verified_events(event_store)
            ledger_vfs = _load_verified_ledger_records(ledger_store)
            current_batches = _load_current_screening_batches(
                records=screening_records, cfg=cfg, review_run_id=review_run_id, event_store=event_store)

        for vf in event_vfs:
            artifacts.append(_stage_file(
                staging_dir, source=vf.path, payload_relpath=f"raw/ai_use_events/{vf.model.event_id}.json",
                logical_path=str(LogicalLocator("ai_use_events", "event", vf.model.event_id)),
                kind="durable_reverified", media_type="application/json", producer_stage=vf.model.stage))

        for vf in ledger_vfs:
            key_digest = _key_digest(vf.model.record_id, vf.model.stage)
            artifacts.append(_stage_file(
                staging_dir, source=vf.path, payload_relpath=f"raw/selection_ledger/{key_digest}/{vf.path.name}",
                logical_path=str(LogicalLocator("selection_ledger", "decision", key_digest, vf.path.stem)),
                kind="durable_reverified", media_type="application/json", producer_stage=vf.model.stage))

        for vf in current_batches:
            artifacts.append(_stage_file(
                staging_dir, source=vf.path, payload_relpath=f"raw/screening_batches/{vf.model.artifact_id}.json",
                logical_path=str(LogicalLocator("screening_batch", "result", vf.model.artifact_id)),
                kind="durable_reverified", media_type="application/json", producer_stage="screening"))

        extraction_bundle = _load_extraction_bundle(terminal_state, cfg)
        if extraction_bundle is not None:
            xid = extraction_bundle.run_id
            artifacts.append(_stage_file(
                staging_dir, source=extraction_bundle.manifest.path, payload_relpath=f"raw/extraction/{xid}/manifest.json",
                logical_path=str(LogicalLocator("extraction", "manifest", xid)),
                kind="durable_reverified", media_type="application/json", producer_stage="extraction"))
            for component in extraction_bundle.checkpoints:
                artifacts.append(_stage_file(
                    staging_dir, source=component.path,
                    payload_relpath=f"raw/extraction/{xid}/checkpoints/{component.document_id}.json",
                    logical_path=str(LogicalLocator("extraction", "checkpoint", xid, component.document_id)),
                    kind="durable_reverified", media_type="application/json", producer_stage="extraction"))
            if extraction_bundle.resolutions is not None:
                artifacts.append(_stage_file(
                    staging_dir, source=extraction_bundle.resolutions.path,
                    payload_relpath=f"raw/extraction/{xid}/extraction_resolutions.json",
                    logical_path=str(LogicalLocator("extraction", "resolution", xid)),
                    kind="durable_reverified", media_type="application/json", producer_stage="extraction"))
            if extraction_bundle.extraction_review is not None:
                review = extraction_bundle.extraction_review
                for component, subid, vc, filename, media_type in (
                    ("review_packet", "json", review.review_packet_json, "extraction_review_packet.json", "application/json"),
                    ("review_packet", "md", review.review_packet_md, "extraction_review_packet.md", "text/markdown"),
                    ("queue_snapshot", None, review.queue_snapshot, "review_queue_snapshot.json", "application/json"),
                    ("suggestions_index", None, review.suggestions_index, "extraction_resolution_suggestions.json", "application/json"),
                    ("decision_journal", None, review.decision_journal, "extraction_resolution_decisions.jsonl", "application/jsonl"),
                ):
                    if vc is None:
                        continue
                    artifacts.append(_stage_file(
                        staging_dir, source=vc.path,
                        payload_relpath=f"raw/extraction/{xid}/{filename}",
                        logical_path=str(LogicalLocator("extraction", component, xid, subid)),
                        kind="durable_reverified", media_type=media_type, producer_stage="extraction"))
                for vc in review.suggestions:
                    artifacts.append(_stage_file(
                        staging_dir, source=vc.path,
                        payload_relpath=f"raw/extraction/{xid}/extraction_resolution_suggestions/{vc.path.name}",
                        logical_path=str(LogicalLocator("extraction", "suggestion", xid, vc.document_id)),
                        kind="durable_reverified", media_type="application/json", producer_stage="extraction"))

        ft_native = (
            cumulative_snapshot.ft_native_provenance if cumulative_snapshot is not None
            else capture_ft_native_provenance(terminal_state, cfg)
        )
        if ft_native is not None:
            ft_run_id = ft_native["ft_screening_run_id"]
            ft_paths = {k: Path(v) for k, v in ft_native["ft_screening_artifact_paths"].items()}
            artifacts.append(_stage_file(
                staging_dir, source=ft_paths["run_result_json"],
                payload_relpath=f"raw/ft_screening/{ft_run_id}/run_result.json",
                logical_path=str(LogicalLocator("ft_screening", "run_result", ft_run_id)),
                kind="durable_reverified", media_type="application/json", producer_stage="ft_screening"))
            artifacts.append(_stage_file(
                staging_dir, source=ft_paths["item_results_jsonl"],
                payload_relpath=f"raw/ft_screening/{ft_run_id}/item_results.jsonl",
                logical_path=str(LogicalLocator("ft_screening", "item_results", ft_run_id)),
                kind="durable_reverified", media_type="application/jsonl", producer_stage="ft_screening"))
            artifacts.append(_stage_file(
                staging_dir, source=ft_paths["ledger_jsonl"],
                payload_relpath=f"raw/ft_screening/{ft_run_id}/ledger.jsonl",
                logical_path=str(LogicalLocator("ft_screening", "ledger", ft_run_id)),
                kind="durable_reverified", media_type="application/jsonl", producer_stage="ft_screening"))

        artifacts.extend(_collect_state_and_run_evidence(staging_dir, workspace=workspace, mode_profile=mode_profile))
        artifacts.extend(_collect_run_mode(staging_dir, workspace=workspace))
        if mode_profile != "track3_live":
            artifacts.extend(_collect_fixture_inputs(staging_dir, fixtures=fixtures))
        artifacts.extend(_collect_reporting(staging_dir, output_dir=output_dir, review_run_id=review_run_id))

        if reproduction_artifacts is not None:
            staged_paths: Dict[str, Path] = {}
            filenames = {
                "reproduction_spec": ("spec", "reproduction_spec.json", "application/json"),
                "gold_standard": ("gold_standard", "gold_standard.json", "application/json"),
                "stage_corpus": ("stage_corpus", "stage_corpus.json", "application/json"),
                "pipeline_evaluation": ("pipeline_evaluation", "pipeline_evaluation.json", "application/json"),
                "gate_report": ("gate_report", "gate_report.json", "application/json"),
                "adjudication_journal": (
                    "adjudication_journal", "evaluation_adjudications.jsonl", "application/jsonl"
                ),
            }
            for name, source_path in reproduction_artifacts.items():
                subkind, filename, media_type = filenames[name]
                payload_relpath = f"raw/reproduction/{review_run_id}/{filename}"
                artifacts.append(_stage_file(
                    staging_dir, source=source_path, payload_relpath=payload_relpath,
                    logical_path=str(LogicalLocator("reproduction", subkind, review_run_id)),
                    kind="durable_reverified", media_type=media_type,
                    producer_stage="evaluation",
                ))
                staged_paths[name] = staging_dir / payload_relpath
            try:
                verify_reproduction_artifact_set(ReproductionArtifactPaths(**staged_paths))
            except Exception as exc:
                raise ReferenceBundleError(
                    f"staged reproduction artifacts failed verification: {exc}"
                ) from exc

        if parent_bundle_id is not None:
            artifacts.append(_stage_file(
                staging_dir, source=workspace / "continuation_lineage.json",
                payload_relpath="raw/continuation/continuation_lineage.json",
                logical_path=str(LogicalLocator("continuation", "lineage", "main")),
                kind="bundle_generated", media_type="application/json", producer_stage="driver"))
            artifacts.append(_stage_file(
                staging_dir, source=workspace / "preserved_decisions.json",
                payload_relpath="raw/continuation/preserved_decisions.json",
                logical_path=str(LogicalLocator("continuation", "preserved_decisions", "main")),
                kind="bundle_generated", media_type="application/json", producer_stage="driver"))

        artifacts.sort(key=lambda a: a.logical_path)
        manifest = PayloadManifestV1(
            bundle_id=bundle_id, mode_profile=mode_profile, created_utc=created_utc, source=source,
            reference_spec_sha256=reference_spec_sha256, artifacts=artifacts, payload_hash="0" * 64,
            parent_bundle_id=parent_bundle_id,
        )
        manifest_dict = manifest.model_dump(mode="json")
        manifest_dict["payload_hash"] = _sha256_rfc8785_jcs({
            k: v for k, v in manifest_dict.items()
            if k != "payload_hash" and not (k == "parent_bundle_id" and v is None)
        })
        manifest = PayloadManifestV1.model_validate(manifest_dict)
        atomic_new_file(staging_dir / "payload_manifest.json", canonical_json_bytes(manifest.model_dump(mode="json")))

    return create_payload(bundle_parent=bundle_parent, bundle_id=bundle_id, build_payload=build_payload)


# ===========================================================================
# Sec 10.10 -- open_bundle / from_bundle / refinalize_reporting_from_bundle
# ===========================================================================

def _maybe_load_sidecar(path: Path, model):
    if not path.exists():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def open_bundle(payload_root: Path, *, run_registry_entry: Optional[RunRegistryEntry] = None,
                 run_registry_key: Optional[str] = None) -> Bundle:
    """Loads and self-verifies a published bundle from its `payload/`
    directory (per Sec 10.10's literal signature -- sidecars live one
    level up, at `payload_root.parent`): `payload_manifest.json`'s own
    `payload_hash` self-check, a COMPLETE file-inventory match (every
    regular file physically under `payload/` appears in `artifacts`
    exactly once, except `payload_manifest.json` itself), each artifact's
    own recorded sha256, and `reporting/export_manifest.json`'s own
    manifest_hash self-check plus its per-entry cross-check against
    `payload_manifest.artifacts`."""
    payload_root = Path(payload_root).resolve()
    bundle_dir = payload_root.parent
    manifest_path = payload_root / "payload_manifest.json"
    manifest = PayloadManifestV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest_dict = manifest.model_dump(mode="json")
    recomputed = _sha256_rfc8785_jcs({
        k: v for k, v in manifest_dict.items()
        if k != "payload_hash" and not (k == "parent_bundle_id" and v is None)
    })
    if recomputed != manifest.payload_hash:
        raise ReferenceBundleError(f"{manifest_path} failed its own payload_hash self-check")

    on_disk = {p.resolve() for p in payload_root.rglob("*") if p.is_file()}
    on_disk.discard(manifest_path)
    declared = {(payload_root / a.path).resolve() for a in manifest.artifacts}
    if on_disk != declared:
        raise ReferenceBundleError(
            f"payload/ inventory mismatch -- missing={sorted(str(p) for p in declared - on_disk)!r}, "
            f"extra={sorted(str(p) for p in on_disk - declared)!r}")
    for artifact in manifest.artifacts:
        manifest.resolve(artifact.logical_path, payload_root)  # containment + sha256 check; raises on mismatch

    export_manifest_path = manifest.resolve("slr://reporting/artifact/export_manifest", payload_root)
    export_manifest_bytes = export_manifest_path.read_bytes()
    reporting_manifest = ReportingStageManifest.model_validate_json(export_manifest_bytes)
    reloaded = json.loads(export_manifest_bytes)
    recomputed_rm = _hash_value({k: v for k, v in reloaded.items() if k != "manifest_hash"})
    if recomputed_rm != reporting_manifest.manifest_hash:
        raise ReferenceBundleError(f"{export_manifest_path} failed its own manifest_hash self-check")
    artifacts_by_logical_path = {a.logical_path: a for a in manifest.artifacts}
    for entry in reporting_manifest.entries:
        if entry.status != "present" or entry.file_sha256 is None:
            continue
        matching = artifacts_by_logical_path.get(entry.logical_path)
        if matching is None:
            raise ReferenceBundleError(
                f"export_manifest.json entry {entry.logical_path!r} has no corresponding payload_manifest artifact")
        if matching.sha256 != entry.file_sha256:
            raise ReferenceBundleError(
                f"export_manifest.json entry {entry.logical_path!r} file_sha256 disagrees with payload_manifest")

    return Bundle(
        bundle_id=manifest.bundle_id, payload_root=payload_root, bundle_dir=bundle_dir,
        payload_manifest=manifest, reporting_stage_manifest=reporting_manifest,
        acceptance_report=_maybe_load_sidecar(bundle_dir / "acceptance_report.json", AcceptanceReport),
        reviewer_attestation=_maybe_load_sidecar(bundle_dir / "reviewer_attestation.json", ReviewerAttestation),
        release_manifest=_maybe_load_sidecar(bundle_dir / "release_manifest.json", ReleaseManifest),
        run_registry_entry=run_registry_entry, run_registry_key=run_registry_key,
    )


def _from_bundle_events(bundle: Bundle) -> List[VerifiedFile]:
    out = []
    for artifact in bundle.payload_manifest.artifacts:
        if not artifact.logical_path.startswith("slr://ai_use_events/event/"):
            continue
        path = bundle.payload_root / artifact.path
        event = AIUseEvent.model_validate_json(path.read_text(encoding="utf-8"))
        out.append(VerifiedFile(model=event, path=path, file_sha256=artifact.sha256, internal_hash=event.content_hash))
    out.sort(key=lambda vf: vf.model.event_id)
    return out


def _from_bundle_ledger(bundle: Bundle) -> List[VerifiedFile]:
    out = []
    for artifact in bundle.payload_manifest.artifacts:
        if not artifact.logical_path.startswith("slr://selection_ledger/decision/"):
            continue
        path = bundle.payload_root / artifact.path
        record = SelectionDecisionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        out.append(VerifiedFile(model=record, path=path, file_sha256=artifact.sha256, internal_hash=record.content_hash))
    out.sort(key=lambda vf: (vf.model.record_id, vf.model.stage, vf.model.decision_id))
    return out


def _from_bundle_screening_batches(
    bundle: Bundle, *, screening_records: List[Record], cfg: ReviewConfig,
    parent_bundle: Optional["Bundle"] = None,
) -> List[VerifiedFile]:
    """Unit E addition: for a Track 3 continuation bundle,
    `screening_records` (from THIS bundle's own `terminal_state[
    "deduplicated_records"]`, which stays empty -- a continuation never
    re-runs dedup) can never equal the bundled `screening_batch` artifacts'
    combined `record_ids`, which are always the PARENT's cumulative
    corpus (see `build_cumulative_reporting_snapshot()`). When
    `parent_bundle` is given, the expected-id-set derivation uses the
    parent bundle's own `screening_records` instead -- default `None`
    preserves today's non-continuation behavior unchanged."""
    out = []
    for artifact in bundle.payload_manifest.artifacts:
        if not artifact.logical_path.startswith("slr://screening_batch/result/"):
            continue
        path = bundle.payload_root / artifact.path
        result = json.loads(path.read_text(encoding="utf-8"))
        from slr_pipeline.nodes.screening import ScreeningBatchResult
        model = ScreeningBatchResult.model_validate(result)
        out.append(VerifiedFile(model=model, path=path, file_sha256=artifact.sha256, internal_hash=model.compute_content_hash()))
    out.sort(key=lambda vf: vf.model.artifact_id)
    # A `set` union alone would silently collapse a record_id that appears in
    # TWO different bundled batch artifacts down to one entry, letting an
    # overlapping/duplicated batch pass the equality check below even though
    # the raw layer no longer represents a valid disjoint partition of
    # records into batches. Check for duplicates across batches explicitly,
    # before ever comparing against the expected set.
    actual_record_id_list = [rid for vf in out for rid in vf.model.record_ids]
    actual_record_ids = set(actual_record_id_list)
    if len(actual_record_id_list) != len(actual_record_ids):
        duplicates = sorted({rid for rid in actual_record_id_list if actual_record_id_list.count(rid) > 1})
        raise ReferenceBundleError(
            f"from_bundle(): bundled screening_batch results overlap -- record_id(s) {duplicates!r} appear in "
            "more than one batch artifact")
    expected_source_records = (
        from_bundle(parent_bundle).screening_records if parent_bundle is not None else screening_records
    )
    expected_batches = compute_current_screening_batches(records=expected_source_records, cfg=cfg)
    expected_record_ids = {r.id for batch in expected_batches for r in batch}
    if expected_record_ids != actual_record_ids:
        raise ReferenceBundleError(
            "from_bundle(): bundled screening_batch results' combined record_ids do not equal "
            "compute_current_screening_batches()'s own derivation over the bundled deduplicated_records/config")
    return out


def _from_bundle_extraction(
    bundle: Bundle, *, extraction_run_id: Optional[str], cfg: ReviewConfig,
    events: List[VerifiedFile],
) -> Optional[SourceBundle]:
    if not extraction_run_id:
        return None
    xid = extraction_run_id
    manifest_locator = str(LogicalLocator("extraction", "manifest", xid))
    manifest_path = bundle.resolve(manifest_locator)
    run_manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    run_manifest.verify_manifest_hash()
    manifest_component = VerifiedComponent(path=manifest_path, file_sha256=bundle.file_sha256(manifest_locator),
                                            internal_hash=run_manifest.manifest_hash)

    checkpoints: List[VerifiedComponent] = []
    checkpoint_results = []
    checkpoint_models: Dict[str, DocumentCheckpoint] = {}
    for document_id in sorted(run_manifest.document_ids):
        locator = str(LogicalLocator("extraction", "checkpoint", xid, document_id))
        ck_path = bundle.resolve(locator)
        checkpoint = DocumentCheckpoint.model_validate_json(ck_path.read_text(encoding="utf-8"))
        checkpoint.verify(expected_document_id=document_id, expected_run_id=xid)
        checkpoint_models[document_id] = checkpoint
        checkpoints.append(VerifiedComponent(path=ck_path, file_sha256=bundle.file_sha256(locator),
                                              internal_hash=run_manifest.checkpoint_hashes[document_id],
                                              document_id=document_id))
        checkpoint_results.append(checkpoint.result.model_dump(mode="json"))

    resolutions_locator = str(LogicalLocator("extraction", "resolution", xid))
    resolutions_component = None
    raw: Dict[str, Any] = {}
    try:
        resolutions_path = bundle.resolve(resolutions_locator)
    except KeyError:
        resolutions_path = None
    if resolutions_path is not None:
        raw = json.loads(resolutions_path.read_text(encoding="utf-8"))
        expected_hash = sha256_hex({k: v for k, v in raw.items() if k != "content_hash"})
        if expected_hash != raw.get("content_hash"):
            raise ReferenceBundleError(f"{resolutions_path} content_hash does not match its own contents")
        resolutions_component = VerifiedComponent(path=resolutions_path, file_sha256=bundle.file_sha256(resolutions_locator),
                                                    internal_hash=raw["content_hash"])
        # Rebuild the additive, in-memory-only evidence_index projection
        # (never itself written to disk -- see load_verified_extraction_
        # resolution_artifact()'s own docstring) from the already-verified
        # checkpoints + resolutions file content, the same pure
        # transformation the live path applies.
        evidence_index = _build_extraction_evidence_index(checkpoint_results, raw.get("resolutions", []), xid)
        # `extraction_results` (Unit I, squishy-petting-starlight.md): same additive, in-memory-only
        # surfacing as the live path (nodes/extraction.py's load_verified_extraction_resolution_
        # artifact()) -- `checkpoint_results` here is the same already-verified per-document
        # DocumentExtractionResult dict shape, just sourced from a rebuilt checkpoint instead of a
        # freshly-opened VerifiedRunBundle.
        raw = dict(raw, evidence_index=evidence_index, extraction_results=checkpoint_results)

    verified_run = VerifiedRunBundle(manifest=run_manifest, checkpoints_by_id=checkpoint_models)
    verified_run.verify()
    seen = {manifest_path.resolve(), *(component.path.resolve() for component in checkpoints)}
    if resolutions_component is not None:
        seen.add(resolutions_component.path.resolve())
    try:
        review_artifacts = _load_extraction_review_artifacts(
            run_root=manifest_path.parent.resolve(), seen=seen, bundle=verified_run,
            schema=load_schema(cfg), events=[vf.model for vf in events], run_id=xid,
            resolutions_raw=raw,
        )
    except Exception as exc:
        raise ReferenceBundleError(f"bundled extraction review artifacts failed verification: {exc}") from exc
    if review_artifacts is not None:
        for component, subid, verified_component in (
            ("review_packet", "json", review_artifacts.review_packet_json),
            ("review_packet", "md", review_artifacts.review_packet_md),
            ("queue_snapshot", None, review_artifacts.queue_snapshot),
            ("suggestions_index", None, review_artifacts.suggestions_index),
            ("decision_journal", None, review_artifacts.decision_journal),
        ):
            if verified_component is not None:
                locator = str(LogicalLocator("extraction", component, xid, subid))
                if bundle.resolve(locator).resolve() != verified_component.path.resolve():
                    raise ReferenceBundleError(
                        f"bundled extraction review component {component!r} is not registered at {locator!r}"
                    )
        for verified_component in review_artifacts.suggestions:
            locator = str(LogicalLocator("extraction", "suggestion", xid, verified_component.document_id))
            if bundle.resolve(locator).resolve() != verified_component.path.resolve():
                raise ReferenceBundleError(
                    f"bundled extraction suggestion is not registered at {locator!r}"
                )
    return SourceBundle(raw=raw, manifest=manifest_component, checkpoints=checkpoints,
                        resolutions=resolutions_component, confirmations=None,
                        computed_content_hash=_hash_value(raw), run_id=xid,
                        extraction_review=review_artifacts)



def _from_bundle_reproduction(
    bundle: Bundle, *, run_id: str
) -> Optional[ReproductionEvaluationArtifacts]:
    """Reconstruct and strictly re-evaluate an optional reproduction set."""

    definitions = {
        "reproduction_spec": ("spec", "reproduction_spec.json"),
        "gold_standard": ("gold_standard", "gold_standard.json"),
        "stage_corpus": ("stage_corpus", "stage_corpus.json"),
        "pipeline_evaluation": ("pipeline_evaluation", "pipeline_evaluation.json"),
        "gate_report": ("gate_report", "gate_report.json"),
        "adjudication_journal": (
            "adjudication_journal", "evaluation_adjudications.jsonl"
        ),
    }
    paths: Dict[str, Path] = {}
    missing: List[str] = []
    for name, (subkind, _filename) in definitions.items():
        locator = str(LogicalLocator("reproduction", subkind, run_id))
        try:
            paths[name] = bundle.resolve(locator)
        except KeyError:
            missing.append(name)
    if len(missing) == len(definitions):
        return None
    if missing:
        raise ReferenceBundleError(
            "bundled reproduction artifact set is incomplete; missing "
            + ", ".join(sorted(missing))
        )

    artifact_paths = ReproductionArtifactPaths(**paths)
    try:
        verified = verify_reproduction_artifact_set(artifact_paths)
    except Exception as exc:
        raise ReferenceBundleError(
            f"bundled reproduction artifacts failed verification: {exc}"
        ) from exc

    internal_hashes = {
        "reproduction_spec": verified.spec.spec_hash(),
        "gold_standard": verified.gold_standard.content_hash(),
        "stage_corpus": sha256_hex(verified.stage_corpus.model_dump(mode="json")),
        "pipeline_evaluation": sha256_hex(
            verified.pipeline_evaluation.model_dump(mode="json")
        ),
        "gate_report": sha256_hex(verified.gate_report.model_dump(mode="json")),
        "adjudication_journal": bundle.file_sha256(
            str(LogicalLocator("reproduction", "adjudication_journal", run_id))
        ),
    }
    components: Dict[str, VerifiedComponent] = {}
    for name, (subkind, _filename) in definitions.items():
        locator = str(LogicalLocator("reproduction", subkind, run_id))
        components[name] = VerifiedComponent(
            path=paths[name],
            file_sha256=bundle.file_sha256(locator),
            internal_hash=internal_hashes[name],
        )
    return ReproductionEvaluationArtifacts(**components)


def _from_bundle_ft_native_provenance(
    bundle: Bundle, *, terminal_state: Dict[str, Any], parent_bundle: Optional["Bundle"] = None,
) -> Optional[Dict[str, Any]]:
    """Unit E addition: `parent_bundle`, when given, makes this
    continuation-aware. If this bundle ran no FT_SCREENING of its own this
    session (`run_id is None` -- e.g. a pure report-regeneration pass),
    the parent's own reconstruction is returned unchanged (informational
    only, since no new FT_SCREENING artifacts exist to manifest this run).
    If this bundle DID run its own fresh FT_SCREENING, `confirmed_record_
    ids` is unioned with the parent's own -- everything else stays this
    bundle's own single-run reconstruction, unchanged."""
    run_id = terminal_state.get("ft_screening_run_id")
    fp = terminal_state.get("ft_screening_execution_fingerprint")
    if run_id is None:
        if parent_bundle is not None:
            return from_bundle(parent_bundle).ft_native_provenance
        return None
    run_result_locator = str(LogicalLocator("ft_screening", "run_result", run_id))
    item_results_locator = str(LogicalLocator("ft_screening", "item_results", run_id))
    ledger_locator = str(LogicalLocator("ft_screening", "ledger", run_id))
    run_result_path = bundle.resolve(run_result_locator)
    item_results_path = bundle.resolve(item_results_locator)
    ledger_path = bundle.resolve(ledger_locator)

    run_result = ScreeningRunResult.model_validate_json(run_result_path.read_text(encoding="utf-8"))
    if run_result.run_id != run_id or run_result.execution_fingerprint != fp:
        raise ReferenceBundleError(
            "bundled ft_screening run_result.json does not match terminal_state's own "
            "ft_screening_run_id/ft_screening_execution_fingerprint")
    item_results = [
        ScreeningItemResult.model_validate_json(line)
        for line in item_results_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    report_ids = [item.report_id for item in item_results]
    if len(set(report_ids)) != len(report_ids):
        raise ReferenceBundleError("bundled ft_screening item_results.jsonl contains duplicate report_id(s)")
    item_pairs = {(item.document_id, item.report_id) for item in item_results}
    run_result_pairs = {(item.document_id, item.report_id) for item in run_result.items}
    if item_pairs != run_result_pairs:
        raise ReferenceBundleError(
            "bundled ft_screening item_results.jsonl's own item set does not match run_result.json's own items")

    confirmed_record_ids = set(report_ids)
    if parent_bundle is not None:
        parent_ft_native = from_bundle(parent_bundle).ft_native_provenance
        if parent_ft_native is not None:
            confirmed_record_ids |= set(parent_ft_native["confirmed_record_ids"])

    return {
        "ft_screening_run_id": run_id, "ft_screening_execution_fingerprint": fp,
        "ft_screening_artifact_paths": {
            "run_result_json": str(run_result_path), "item_results_jsonl": str(item_results_path),
            "ledger_jsonl": str(ledger_path),
        },
        "artifact_file_hashes": {
            "run_result_json": bundle.file_sha256(run_result_locator),
            "item_results_jsonl": bundle.file_sha256(item_results_locator),
            "ledger_jsonl": bundle.file_sha256(ledger_locator),
        },
        "confirmed_record_ids": sorted(confirmed_record_ids),
        "classification": "state_referenced_byte_hashed_not_semantically_reverified",
    }


def from_bundle(bundle: Bundle, *, parent_bundle: Optional["Bundle"] = None) -> ReportingSourceSnapshot:
    """Rebuilds a `ReportingSourceSnapshot` entirely from a bundle's own
    `raw/state/terminal_state.json` + flattened `raw/` tree -- see this
    module's own top-of-file scoping note for exactly what is and is not
    re-verified here relative to the live `capture_reporting_snapshot()`.

    `parent_bundle` (Unit E addition): required for a Track 3 continuation
    bundle -- a continuation's own `terminal_state["deduplicated_records"]`
    stays empty (dedup never re-runs), so `_from_bundle_screening_batches()`
    can only reconcile against the PARENT's own cumulative
    `screening_records`, and `_from_bundle_ft_native_provenance()` can only
    see the parent's own confirmed full-text records too. Default `None`
    preserves today's non-continuation behavior unchanged. This does NOT
    reconstruct `oversight_mode_state`/`automation_origin_state`/
    `screening_qc_state`'s own parent-inclusive union -- callers that need
    that call `apply_parent_oversight_union()` separately, since those 3
    fields are never staged as bundle payload artifacts and so cannot be
    reconstructed from this bundle's payload alone, unlike the two fields
    this parameter fixes.

    Unit J hardening: `bundle.payload_manifest.parent_bundle_id` (when set)
    is enforced, not merely documented as required -- raises
    `ReferenceBundleError` if `parent_bundle` is omitted, if
    `parent_bundle.bundle_id` does not exactly match the declared id, or if
    the bundle's own `continuation_lineage.json` disagrees with the actual
    parent bundle's registry hashes. A caller can no longer silently
    reconstruct a continuation's snapshot against the wrong (or no)
    parent.

    Code-review corrective pass (post-95c9667, round 2): `parent_bundle.
    run_registry_entry` is now REQUIRED, not merely consulted when present
    -- `open_bundle()`'s own `run_registry_entry` parameter defaults to
    `None`, so a `parent_bundle` opened without wiring it through (any
    caller other than `continuation.identity.resolve_parent_bundle()`,
    which always supplies it) previously made the two lineage-hash checks
    below a silent no-op, leaving only the bare `bundle_id` string
    compared. A continuation's parent must always be a release-verified
    bundle with a real registry entry; anything else raises here instead
    of quietly skipping the hash cross-check.

    Code-review corrective pass, round 3: a `run_registry_entry` being
    merely *present* on `parent_bundle` proves nothing by itself -- any
    caller can attach a fabricated one (e.g. via `dataclasses.replace()`)
    without the parent ever having genuinely passed acceptance/release.
    `verify_release_structure(parent_bundle)` is now called directly
    (raising `ReleaseStructureInvalid`/`ReleaseNotApproved`, the same as
    `continuation.identity.resolve_parent_bundle()`'s own convention --
    never wrapped into `ReferenceBundleError`) before the lineage hashes
    below are ever trusted, so a parent must actually pass every one of the
    6 structural checks (including registry-entry-equals-sidecars and
    decision == APPROVED) before its `run_registry_entry` hashes are used
    for anything."""
    declared_parent_id = bundle.payload_manifest.parent_bundle_id
    if declared_parent_id is not None or parent_bundle is not None:
        if declared_parent_id is None:
            raise ReferenceBundleError(
                f"from_bundle(): parent_bundle was passed but bundle {bundle.bundle_id!r} "
                "does not declare a parent_bundle_id -- it is not a continuation bundle")
        if parent_bundle is None:
            raise ReferenceBundleError(
                f"from_bundle(): bundle {bundle.bundle_id!r} declares parent_bundle_id "
                f"{declared_parent_id!r} -- parent_bundle is required")
        if parent_bundle.bundle_id != declared_parent_id:
            raise ReferenceBundleError(
                f"from_bundle(): bundle {bundle.bundle_id!r} declares parent_bundle_id "
                f"{declared_parent_id!r}, but the parent_bundle passed is "
                f"{parent_bundle.bundle_id!r}")
        lineage = json.loads(bundle.resolve("slr://continuation/lineage/main").read_bytes())
        if lineage.get("parent_bundle_id") != parent_bundle.bundle_id:
            raise ReferenceBundleError(
                f"from_bundle(): continuation_lineage.json's own parent_bundle_id "
                f"{lineage.get('parent_bundle_id')!r} does not match parent_bundle "
                f"{parent_bundle.bundle_id!r}")
        verify_release_structure(parent_bundle)
        if lineage.get("parent_payload_hash") != parent_bundle.run_registry_entry.payload_hash:
            raise ReferenceBundleError(
                "from_bundle(): continuation_lineage.json's own parent_payload_hash does not "
                "match parent_bundle.run_registry_entry.payload_hash")
        if lineage.get("parent_release_hash") != parent_bundle.run_registry_entry.release_hash:
            raise ReferenceBundleError(
                "from_bundle(): continuation_lineage.json's own parent_release_hash does not "
                "match parent_bundle.run_registry_entry.release_hash")

    payload_root = bundle.payload_root
    terminal_state_path = bundle.resolve("slr://state/snapshot/terminal_state")
    terminal_state = json.loads(terminal_state_path.read_bytes())

    graph_snapshot_path = bundle.resolve("slr://state/snapshot/terminal_graph_snapshot")
    graph_snapshot = json.loads(graph_snapshot_path.read_bytes())
    if graph_snapshot.get("terminal_state_sha256") != file_sha256(terminal_state_path):
        raise ReferenceBundleError(
            "terminal_graph_snapshot.terminal_state_sha256 does not match the bundled terminal_state.json")
    if graph_snapshot.get("review_run_id") != terminal_state.get("review_run_id"):
        raise ReferenceBundleError("terminal_graph_snapshot.review_run_id != terminal_state.review_run_id")

    review_run_id = terminal_state["review_run_id"]
    cfg = ReviewConfig.from_dict(terminal_state["config"])
    config_projection = ReportingConfigProjection.from_config(cfg)

    screening_records = [Record.from_dict(r) for r in terminal_state["deduplicated_records"]]
    full_text_records = [Record.from_dict(r) for r in terminal_state["full_text_records"]]
    screened_records_full = [Record.from_dict(r) for r in terminal_state.get("screened_records_full") or []]

    events = _from_bundle_events(bundle)
    extraction = _from_bundle_extraction(
        bundle, extraction_run_id=terminal_state.get("extraction_run_id"), cfg=cfg, events=events,
    )
    reproduction_evaluation = _from_bundle_reproduction(
        bundle, run_id=review_run_id
    )
    if extraction is not None and reproduction_evaluation is not None:
        extraction = _dataclasses_replace(
            extraction, reproduction_evaluation=reproduction_evaluation
        )

    if extraction is not None:
        full_text_records = canonicalize_corpus_order(
            full_text_records=full_text_records, charted=list(extraction.raw.get("resolved_projection", [])),
            error_prefix="from_bundle canonical corpus order",
        )

    if terminal_state.get("quality_appraisal_run_id"):
        raise ReferenceBundleError(
            "from_bundle(): terminal_state carries a quality_appraisal_run_id, but Track 2's reference run "
            "is scoping-only and quality_appraisal is always expected to be absent")

    # Deliberately NOT `load_and_verify_synthesis_output()` -- that call
    # re-derives semantic correctness (cross-checks against extraction/QA,
    # re-runs the evidence-map reconstruct-and-compare, re-validates
    # BERTopic consistency, gates on the live event store) against a
    # RunConfig-shaped `output_dir` layout the bundle's flattened `raw/`
    # tree does not preserve. Per this module's own top-of-file scoping
    # note, that full semantic pass already ran ONCE, live, when
    # `capture_reporting_snapshot()` first produced the very
    # `terminal_state["synthesis"]` bundled here -- `from_bundle()` only
    # re-establishes tamper-evidence (schema validation + hashing) over
    # that already-validated content, same as `extraction`'s reconstruction
    # above (`_build_extraction_evidence_index`, not `_read_and_verify_
    # resolution_artifact`).
    synthesis_model = SynthesisOutput.model_validate(terminal_state.get("synthesis") or {})
    synthesis_output = VerifiedProjection(model=synthesis_model, content_hash=_hash_value(synthesis_model))
    evidence_map = (EvidenceMapModel.model_validate(synthesis_model.evidence_map)
                    if synthesis_model.evidence_map is not None else None)

    # Mirrors capture_reporting_snapshot()'s own final `reference_context=`
    # construction exactly: read off the live `cfg` attribute, never the
    # raw `terminal_state["config"]` dict directly (ReviewConfig.from_dict
    # is the one binding transformation both paths must agree on).
    reference_context = (
        ReferenceReportingContext.model_validate(cfg.reference_reporting_context)
        if cfg.reference_reporting_context is not None else None
    )
    if reference_context is not None:
        # Sec 10.2/10.10 field 24's cross-check: `reference_spec.json`'s
        # `reference_reporting_context_sha256` is a FILE-hash target (see
        # reference_spec.py's own `@register_target`), pointing at the raw
        # bytes of `reference_reporting_context.json` -- not a hash of
        # `context.model_dump()`. Confirm the bundled spec singleton still
        # agrees with the bundled context singleton's own raw file hash.
        reference_spec_dict = json.loads(bundle.resolve("slr://fixture/spec/reference").read_bytes())
        expected_context_sha256 = reference_spec_dict["reference_reporting_context_sha256"]
        actual_context_sha256 = bundle.file_sha256("slr://fixture/spec/reference_reporting_context")
        if actual_context_sha256 != expected_context_sha256:
            raise ReferenceBundleError(
                "bundled fixture_inputs/spec/reference_reporting_context.json does not match "
                "reference_spec.json's own reference_reporting_context_sha256")

    return ReportingSourceSnapshot(
        review_run_id=review_run_id, config_projection=config_projection,
        config_projection_hash=_hash_value(config_projection),
        raw_records=_state_value(_validate_list_of_dicts(terminal_state["raw_records"], name="raw_records")),
        screening_records=screening_records, full_text_records=full_text_records,
        screened_records_full=screened_records_full,
        prisma=_state_value(PRISMACounts(**terminal_state["prisma_counts"])),
        search_log=_state_value(SearchLog(**terminal_state["search_log"])),
        screening_review_resolutions=_state_value(_validate_list_of_dicts(
            terminal_state.get("screening_review_resolutions") or [], name="screening_review_resolutions")),
        full_text_review_resolutions=_state_value(_validate_list_of_dicts(
            terminal_state.get("full_text_review_resolutions") or [], name="full_text_review_resolutions")),
        full_text_review_queue=_state_value(_validate_list_of_dicts(
            terminal_state.get("full_text_review_queue") or [], name="full_text_review_queue")),
        screening_qc_state=_validate_screening_qc_state(terminal_state.get("screening_qc")),
        oversight_mode_state=dict(terminal_state.get("oversight_mode") or {}),
        automation_origin_state=dict(terminal_state.get("automation_origin") or {}),
        events=events, ledger_records=_from_bundle_ledger(bundle),
        current_screening_batch_results=_from_bundle_screening_batches(
            bundle, screening_records=screening_records, cfg=cfg, parent_bundle=parent_bundle),
        extraction=extraction, quality_appraisal=None, synthesis_output=synthesis_output, evidence_map=evidence_map,
        ft_native_provenance=_from_bundle_ft_native_provenance(
            bundle, terminal_state=terminal_state, parent_bundle=parent_bundle),
        reference_context=reference_context,
        reproduction_evaluation=reproduction_evaluation,
    )


def refinalize_reporting_from_bundle(payload_root: Path, *, out_dir: Path) -> Dict[str, Any]:
    """Re-opens the bundle from `payload_root` (per Sec 10.10's literal
    signature), re-derives the PRISMA selection splits from `from_bundle()`'s
    own snapshot, and re-runs `publish_report_bundle()` against a fresh
    `out_dir` -- a relocation/refinalize smoke test: the freshly re-
    rendered `export_manifest.json` (and every renderer artifact it
    references) reproduces byte-for-byte what the bundle already carries
    under `reporting/`, proving the bundle's own raw layer is a complete,
    sufficient source for the reporting phase -- WITH ONE DELIBERATE
    EXCEPTION: `ai_traceability.json`/`.md` embed
    `full_text_eligibility_native_provenance.ft_screening_artifact_paths`,
    which `_from_bundle_ft_native_provenance()` resolves via
    `bundle.resolve(...)`, i.e. the ft_screening raw files' CURRENT location
    under `payload_root`'s own `raw/ft_screening/` tree -- not whatever
    transient host path the original live pipeline run happened to write
    to. That re-anchoring is correct/portable behavior (the whole point of
    calling this after a relocation), not a bug, but it means those two
    artifacts' bytes -- and `export_manifest.json`'s recorded `file_sha256`
    for them -- legitimately change on any refinalize that follows a real
    relocation; a caller diffing this call's output against the bundle's
    existing `reporting/` tree must account for that one field before
    treating any other divergence as tamper-evidence. Only the
    `ReportingStageManifest` and its renderer artifacts are regenerated --
    `PayloadManifestV1` is not touched by this call."""
    from slr_pipeline.nodes.reporting import compute_selection_splits, publish_report_bundle

    bundle = open_bundle(payload_root)
    snapshot = from_bundle(bundle)
    splits = compute_selection_splits(snapshot=snapshot, prisma=snapshot.prisma.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    return publish_report_bundle(report_dir=out_dir, snapshot=snapshot, splits=splits)


# ===========================================================================
# CLI -- `python -m slr_pipeline.reference_bundle break-remote-lock <id> ...`
# ===========================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m slr_pipeline.reference_bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    brk = sub.add_parser("break-remote-lock", help="Force-clear a remote/rebooted-host bundle lock.")
    brk.add_argument("bundle_id")
    brk.add_argument("--bundle-parent", type=Path, required=True)
    brk.add_argument("--i-confirm-the-other-host-is-dead", action="store_true", dest="confirmed")
    args = parser.parse_args(argv)

    if args.command == "break-remote-lock":
        if not args.confirmed:
            parser.error("refusing to break a remote lock without --i-confirm-the-other-host-is-dead")
        break_remote_lock(args.bundle_parent, args.bundle_id)
        print(f"broke remote lock for {args.bundle_id}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(_main())
