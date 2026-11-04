"""Resumable per-document checkpoints + a final run manifest (requirements
11-12). Checkpoints live under `output_dir/runs/<run_id>/checkpoints/`,
namespaced by `run_id` -- a configuration change produces a different
`run_id` (see `run_identity.py`) and therefore a different checkpoint
directory automatically.

Two review-driven guarantees this module is directly responsible for:
  - Every checkpoint is a self-describing, content-hashed ENVELOPE
    (`DocumentCheckpoint`) that binds the run/document/schema/config
    identity alongside the result -- not a bare `DocumentExtractionResult`
    trusted purely because a file with a plausible name exists at a
    plausible path. Loading a checkpoint re-verifies its content hash and
    its own claimed `document_id`; a tampered, copied-from-elsewhere, or
    otherwise-inconsistent file raises `CheckpointIntegrityError` rather
    than being silently accepted.
  - The checkpoint FILENAME is derived from a hash of `document_id`, not
    the raw id interpolated into a path -- a `document_id` such as
    `"../../../escape"` cannot resolve outside `checkpoints_dir`, and two
    different runs' documents never collide on a shared filename purely
    because of overlapping directory naming. The verified envelope still
    carries the ORIGINAL `document_id` inside it (checked on load), so
    nothing about the real identity is lost.

Writes are exclusive-temp-file-then-atomic-replace (`os.replace`) so a
process killed mid-write never leaves a half-written checkpoint that a
later resume could misread as complete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import sha256_hex
from .parsing import PARSER_NAME, DoclingParserOptions, installed_docling_version
from .results import DocumentExtractionResult
from .run_identity import DocumentIdentity, compute_run_id
from .schema import ExtractionSchema


class CheckpointIntegrityError(Exception):
    """A checkpoint or manifest file exists but cannot be trusted: its
    content hash does not match its own claimed contents (tampered,
    corrupted, or copied from a different document/run), its claimed
    `document_id` does not match the filename it was loaded for, or a new
    manifest would silently overwrite an existing one with a
    contradictory checkpoint hash for the same document within the same
    run_id. Callers must treat this the same as any other unrecoverable
    parse/call failure -- never silently proceed with unverified data."""


class DocumentCheckpoint(BaseModel):
    """The complete, self-describing, content-hashed envelope for one
    document's checkpointed result (review finding 2: "use a checkpoint
    envelope binding run ID, document/PDF identity, schema/config
    identity, result, telemetry and its own content hash" -- per-batch
    telemetry itself already lives on `result.batch_executions`, so it is
    not duplicated here separately)."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    run_id: str
    document_id: str
    pdf_path: str
    pdf_sha256: str
    extraction_schema_id: str
    extraction_schema_version: str
    extraction_schema_hash: str
    parser_name: str
    parser_version: str
    parser_options: dict
    # §5.4: `None` in ordinary/production use (no pinned model-artifacts
    # directory). Set only when `parse_pdf()` was actually given an
    # `expected_artifacts_manifest_digest` and verified it -- see
    # `parsing.py`.
    artifacts_manifest_digest: Optional[str] = None
    max_batch_chars: int
    prompt_template_version: str
    output_schema_version: str
    model: str
    model_config_id: str
    reasoning_effort: Optional[str]
    max_completion_tokens: int
    expected_provider_model: str
    max_batches: Optional[int] = None
    result: DocumentExtractionResult
    content_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _document_id_matches_result(self) -> "DocumentCheckpoint":
        """Structural, not just checked in `verify()` -- review finding
        E's adversarial reproduction: "a content-consistent envelope for
        `doc` carrying a result for `OTHER` passed verification." An
        envelope whose own `document_id` disagrees with its own
        `result.document_id` is internally incoherent regardless of any
        external run/document context, so this is unconstructable, the
        same way `DocumentExtractionResult`'s own structural invariants
        are (see `results.py`)."""
        if self.document_id != self.result.document_id:
            raise ValueError(
                f"checkpoint document_id={self.document_id!r} does not match "
                f"result.document_id={self.result.document_id!r}"
            )
        return self

    @classmethod
    def build(
        cls, *, run_id: str, document_id: str, pdf_path: str, pdf_sha256: str, schema: ExtractionSchema,
        parser_options: DoclingParserOptions, max_batch_chars: int, model_config, result: DocumentExtractionResult,
        artifacts_manifest_digest: Optional[str] = None, max_batches: Optional[int] = None,
    ) -> "DocumentCheckpoint":
        # Review finding M: the schema hash alone does not prove the
        # PERSISTED result actually corresponds to that schema -- "it
        # accepted a schema containing only field `a` while persisting a
        # result containing only field `b`." A checkpoint's own
        # self-consistency must include this: exact field_id coverage,
        # not merely a matching hash label.
        schema_field_ids = {f.field_id for f in schema.fields}
        result_field_ids = {f.field_id for f in result.fields}
        if schema_field_ids != result_field_ids:
            missing, extra = schema_field_ids - result_field_ids, result_field_ids - schema_field_ids
            raise ValueError(
                f"result field_ids do not exactly match schema field_ids -- "
                f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
            )

        envelope = dict(
            run_id=run_id, document_id=document_id, pdf_path=pdf_path, pdf_sha256=pdf_sha256,
            extraction_schema_id=schema.schema_id, extraction_schema_version=schema.version,
            extraction_schema_hash=sha256_hex(schema.model_dump(mode="json")),
            parser_name=PARSER_NAME, parser_version=installed_docling_version(),
            parser_options={
                "do_ocr": parser_options.do_ocr, "do_table_structure": parser_options.do_table_structure,
                "max_chunk_chars": parser_options.max_chunk_chars,
            },
            artifacts_manifest_digest=artifacts_manifest_digest,
            max_batch_chars=max_batch_chars,
            prompt_template_version=model_config.prompt_template_version,
            output_schema_version=model_config.output_schema_version,
            model=model_config.model,
            model_config_id=model_config.model_config_id,
            reasoning_effort=model_config.reasoning_effort,
            max_completion_tokens=model_config.max_completion_tokens,
            expected_provider_model=model_config.expected_provider_model,
            max_batches=max_batches,
            result=result.model_dump(mode="json"),
        )
        content_hash = sha256_hex(envelope)
        return cls(**envelope, content_hash=content_hash)

    def _recompute_content_hash(self) -> str:
        envelope = self.model_dump(mode="json", exclude={"content_hash"})
        return sha256_hex(envelope)

    def verify(self, *, expected_document_id: Optional[str] = None, expected_run_id: Optional[str] = None) -> None:
        """Raises `CheckpointIntegrityError` if this checkpoint's content
        hash does not match its own contents, if (when `expected_document_id`
        is given -- the caller's own filename-derived expectation) the
        envelope's claimed `document_id` does not match what was actually
        being loaded, or if (when `expected_run_id` is given) the
        envelope's claimed `run_id` does not match the store it was
        loaded from -- review finding A's adversarial reproduction: "a
        valid checkpoint from `old-run`, copied into `new-run` under the
        same document ID, loads successfully." Content-hash verification
        alone does not catch this, since the copied file IS internally
        self-consistent; only an explicit run_id cross-check does."""
        if expected_document_id is not None and self.document_id != expected_document_id:
            raise CheckpointIntegrityError(
                f"checkpoint document_id mismatch: loaded for {expected_document_id!r} but the "
                f"envelope itself claims document_id={self.document_id!r}"
            )
        if expected_run_id is not None and self.run_id != expected_run_id:
            raise CheckpointIntegrityError(
                f"checkpoint run_id mismatch: loaded from a store for run_id={expected_run_id!r} but "
                f"the envelope itself claims run_id={self.run_id!r} -- this checkpoint belongs to a "
                "DIFFERENT run and must not be reused here, even though its own content_hash is valid"
            )
        if self._recompute_content_hash() != self.content_hash:
            raise CheckpointIntegrityError(
                f"checkpoint content_hash mismatch for document_id={self.document_id!r} -- the file "
                "may be tampered, corrupted, or copied from a different run/document"
            )


class TelemetrySummary(BaseModel):
    """Provider-usage/identity summary aggregated across every batch of
    every checkpointed document in the run (review finding A: "the
    manifest provides no telemetry summary/provider-consistency result
    even though those values exist in its checkpointed batch records").
    `distinct_provider_models` collects every non-null
    `BatchExecutionRecord.provider_model` actually seen -- INCLUDING on
    failed batches, since `openai_runner.ProviderModelMismatchError`
    (review finding P) now carries the actual returned model even for a
    rejected call, so a real substitution is never invisible here just
    because its content was rejected. `provider_model_consistent` is
    `True` only if every one of them equals the run's own
    `expected_provider_model` (vacuously `True` if none were seen at
    all, e.g. a run with only transport-failed batches) -- a `False`
    here now means a REAL provider-substitution response was recorded,
    not merely a hypothetical. `provider_mismatch_count` counts batches
    that failed specifically due to `ProviderModelMismatchError`
    (matched by its sanitized type-name string on
    `BatchExecutionRecord.error`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    distinct_provider_models: tuple[str, ...]
    provider_model_consistent: bool
    provider_mismatch_count: int = Field(ge=0)


class RunManifest(BaseModel):
    """Binds the checkpoints and configuration for one completed (or
    partially completed) run (requirement 12). Coverage is reported as
    THREE disjoint sets, never one blended "completed" count (review
    refinement: "Exact manifest coverage should separately report
    checkpointed, complete, and incomplete documents"):
      - `checkpointed_document_ids`: every document with a checkpoint on
        disk as of this manifest (freshly run OR resumed) -- MUST equal
        `document_ids` exactly (review finding A: `run_extraction()`
        always processes every requested document before writing one
        manifest, so a manifest describing anything less than full
        requested coverage is invalid, never a legitimate partial
        publish).
      - `complete_document_ids` / `incomplete_document_ids`: that same
        set, partitioned by each checkpoint's own
        `result.is_complete` -- a checkpointed-but-incomplete document
        (some batch failed) is a REAL, distinct outcome, not silently
        folded into "completed" just because a checkpoint file exists
        for it.
    `checkpoint_hashes` binds the manifest to the EXACT checkpoint bytes
    it describes. Note that `RunManifest`'s OWN validators only check
    INTERNAL coherence (the fields agree with each other); cross-checking
    `checkpoint_hashes` against the ACTUAL on-disk checkpoint files, and
    refusing to silently overwrite an already-published manifest with
    different content, is `ExtractionCheckpointStore.write_manifest()`'s
    job (a `RunManifest` instance alone, detached from a store, cannot
    verify that -- review finding A: "a fabricated checkpoint hash was
    accepted" required exactly this store-level cross-check, since a
    self-consistent-looking manifest object can still lie about what is
    actually on disk)."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    run_id: str
    extraction_schema_id: str
    extraction_schema_version: str
    extraction_schema_hash: str
    parser_name: str
    parser_version: str
    parser_options: dict
    artifacts_manifest_digest: Optional[str] = None
    max_batch_chars: int
    prompt_template_version: str
    output_schema_version: str
    model: str
    model_config_id: str
    reasoning_effort: Optional[str]
    max_completion_tokens: int
    expected_provider_model: str
    max_batches: Optional[int] = None
    document_ids: tuple[str, ...]  # ordered, as run -- attempted/checkpointed only
    checkpointed_document_ids: tuple[str, ...]
    # Review finding (mixed blocked/attempted documents): a document
    # blocked by `max_batches` before any billed call is never
    # checkpointed, so it cannot be counted in `document_ids`, but it IS
    # part of the run's own request -- `requested_document_ids` names the
    # full originally-requested set (order-preserving), and
    # `blocked_document_ids` names the subset of it that was durably
    # blocked. `document_ids` | `blocked_document_ids` ==
    # `requested_document_ids` exactly (validated below); when a manifest
    # predates this field (or a caller never supplies it), it defaults to
    # `document_ids` itself, matching the prior, blocking-unaware
    # behavior exactly.
    blocked_document_ids: tuple[str, ...] = ()
    requested_document_ids: tuple[str, ...] = ()
    complete_document_ids: tuple[str, ...]
    incomplete_document_ids: tuple[str, ...]
    checkpoint_hashes: dict[str, str]  # document_id -> DocumentCheckpoint.content_hash
    telemetry_summary: TelemetrySummary
    created_at: str  # ISO-8601 UTC
    manifest_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _default_requested_document_ids(cls, data):
        if isinstance(data, dict) and not data.get("requested_document_ids"):
            data = dict(data)
            data["requested_document_ids"] = data.get("document_ids", ())
        return data

    def _recompute_manifest_hash(self) -> str:
        return sha256_hex(self.model_dump(mode="json", exclude={"manifest_hash"}))

    def verify_manifest_hash(self) -> None:
        """Review finding E: `load_manifest()` previously only did
        Pydantic parsing -- a published manifest file edited on disk
        (e.g. its `model` field changed post-publication) would be
        accepted on reload without complaint. `manifest_hash` is this
        object's own content hash (the SAME self-describing-envelope
        pattern `DocumentCheckpoint.content_hash`/`verify()` already
        uses), recomputed and checked on every load."""
        if self._recompute_manifest_hash() != self.manifest_hash:
            raise CheckpointIntegrityError(
                f"manifest_hash mismatch for run_id={self.run_id!r} -- the file may be tampered or corrupted"
            )

    @model_validator(mode="after")
    def _document_ids_are_unique(self) -> "RunManifest":
        """Review finding G: `set()`-based coverage checks alone let a
        duplicate document_id slip through undetected (two identical
        entries collapse to the same set membership as one) -- this
        checks LENGTH, not just set equality, so a duplicate is
        impossible to construct regardless of how many other checks
        happen to also pass."""
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document_ids must not contain duplicates")
        return self

    @model_validator(mode="after")
    def _checkpointed_matches_requested_documents(self) -> "RunManifest":
        if self.checkpointed_document_ids != self.document_ids:
            raise ValueError(
                "checkpointed_document_ids must exactly equal document_ids, in order -- "
                "run_extraction() always checkpoints every requested document before "
                "publishing a manifest, so partial coverage indicates an invalid/tampered manifest"
            )
        return self

    @model_validator(mode="after")
    def _requested_partitions_into_checkpointed_and_blocked(self) -> "RunManifest":
        """Review finding: `document_ids` (attempted) and
        `blocked_document_ids` must be a DISJOINT, EXACT partition of
        `requested_document_ids` -- a document cannot be both attempted
        and blocked, and no requested document may be silently unaccounted
        for by either set."""
        if len(set(self.requested_document_ids)) != len(self.requested_document_ids):
            raise ValueError("requested_document_ids must not contain duplicates")
        if len(set(self.blocked_document_ids)) != len(self.blocked_document_ids):
            raise ValueError("blocked_document_ids must not contain duplicates")
        checkpointed = set(self.document_ids)
        blocked = set(self.blocked_document_ids)
        requested = set(self.requested_document_ids)
        if checkpointed & blocked:
            raise ValueError("a document cannot appear in both document_ids and blocked_document_ids")
        if checkpointed | blocked != requested:
            raise ValueError(
                "document_ids + blocked_document_ids must exactly partition requested_document_ids"
            )
        return self

    @model_validator(mode="after")
    def _coverage_partition_is_exact(self) -> "RunManifest":
        checkpointed = set(self.checkpointed_document_ids)
        complete = set(self.complete_document_ids)
        incomplete = set(self.incomplete_document_ids)
        if complete & incomplete:
            raise ValueError("a document cannot appear in both complete_document_ids and incomplete_document_ids")
        if complete | incomplete != checkpointed:
            raise ValueError(
                "complete_document_ids + incomplete_document_ids must exactly partition "
                "checkpointed_document_ids"
            )
        if set(self.checkpoint_hashes) != checkpointed:
            raise ValueError("checkpoint_hashes must have exactly one entry per checkpointed_document_id")
        return self


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _checkpoint_filename(document_id: str) -> str:
    """Hash-derived, never the raw `document_id` interpolated into a
    path -- an ID like `"../../../escape"` or one containing slashes
    cannot resolve outside `checkpoints_dir` or create nested
    subdirectories (review finding 6). The verified envelope still
    carries the original `document_id`; see `DocumentCheckpoint.verify`."""
    return f"{sha256_hex(document_id)[:32]}.json"


_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ExtractionCheckpointStore:
    def __init__(self, *, output_dir: str | Path, run_id: str) -> None:
        """Review finding K: `run_id` was interpolated directly into a
        path with no validation -- unlike checkpoint FILENAMES (already
        hash-derived), the run DIRECTORY itself had the same class of
        path-traversal exposure the filename fix was meant to close.
        `run_id` must be a safe token (no path separators, no `.`/`..`);
        the resolved run directory is additionally required to remain
        beneath the resolved `output_dir/runs` root as defense in depth
        (provably always true once the regex passes, but checked
        explicitly rather than assumed)."""
        if not _SAFE_RUN_ID_RE.match(run_id):
            raise ValueError(
                f"run_id {run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        runs_root = self.output_dir / "runs"
        self.run_dir = runs_root / run_id
        if not self.run_dir.resolve().is_relative_to(runs_root.resolve()):
            raise ValueError(f"run_id {run_id!r} resolves outside the runs root -- refusing to construct store")
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def checkpoint_path(self, document_id: str) -> Path:
        return self.checkpoints_dir / _checkpoint_filename(document_id)

    def _blocked_store(self) -> "BlockedDocumentStore":
        """The SAME `output_dir`/`run_id` namespace this checkpoint store
        uses, so `_derive_manifest()` can recover a blocked document's
        identity (pdf_path/pdf_sha256) when a manifest describes a mix of
        attempted and blocked documents (review finding: a mixed
        blocked/attempted manifest publish previously recomputed
        `compute_run_id()` from ONLY the attempted subset, which can never
        equal a run_id computed over the full originally-requested set)."""
        return BlockedDocumentStore(output_dir=self.output_dir, run_id=self.run_id)

    def has_checkpoint(self, document_id: str) -> bool:
        return self.checkpoint_path(document_id).exists()

    def load_checkpoint(self, document_id: str) -> DocumentCheckpoint:
        data = json.loads(self.checkpoint_path(document_id).read_text(encoding="utf-8"))
        checkpoint = DocumentCheckpoint.model_validate(data)
        checkpoint.verify(expected_document_id=document_id, expected_run_id=self.run_id)
        return checkpoint

    def write_checkpoint(self, checkpoint: DocumentCheckpoint) -> None:
        checkpoint.verify(expected_document_id=checkpoint.document_id, expected_run_id=self.run_id)
        _atomic_write_json(self.checkpoint_path(checkpoint.document_id), checkpoint.model_dump(mode="json"))

    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    _MANIFEST_CONFIG_FIELDS = (
        "extraction_schema_id", "extraction_schema_version", "extraction_schema_hash",
        "parser_name", "parser_version", "parser_options", "artifacts_manifest_digest", "max_batch_chars",
        "prompt_template_version", "output_schema_version",
        "model", "model_config_id", "reasoning_effort", "max_completion_tokens", "expected_provider_model",
        "max_batches",
    )

    def _read_manifest_file(self) -> Optional[RunManifest]:
        """Parses the manifest file and verifies its OWN self-hash only
        -- no run/checkpoint re-verification. Private: `load_manifest()`
        (the public entry point) always follows this with the full
        `_derive_manifest()` cross-check; this split exists only so
        `publish_manifest()` can read the current file without paying for
        that cross-check twice in the same call."""
        path = self.manifest_path()
        if not path.exists():
            return None
        manifest = RunManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        manifest.verify_manifest_hash()
        return manifest

    def _derive_manifest(
        self, *, document_ids: tuple[str, ...], created_at: str,
        requested_document_ids: Optional[tuple[str, ...]] = None,
    ) -> RunManifest:
        """THE single manifest-verification/derivation code path -- used
        by `publish_manifest()` (to build a fresh manifest, and to
        re-verify an existing one before deciding whether a republish is
        a true no-op) AND by `load_manifest()` (to re-verify a manifest
        read from disk against the CURRENT checkpoints and this store's
        own run_id), so there is exactly one place that computes or
        checks manifest content, never two that could drift apart
        (review finding G's own required correction: "centralize one
        store-bound manifest verifier and call it from both publication
        and load").

        `document_ids` are the ATTEMPTED (checkpointed) documents, order
        matters. `requested_document_ids`, if given, is the FULL
        originally-requested set (also order-preserving); any id in it
        that is not in `document_ids` must be durably blocked (via this
        store's own `BlockedDocumentStore`), never simply missing --
        `None` (the default) means "no blocking in play", i.e.
        `requested_document_ids` is treated as exactly `document_ids`
        (review finding: mixed blocked/attempted documents previously
        recomputed `compute_run_id()` from ONLY the attempted subset,
        which can never equal a run_id computed over the full
        originally-requested set -- this always raised
        `CheckpointIntegrityError` whenever at least one document was
        blocked).

        Every field is derived purely from the verified on-disk
        checkpoints for `document_ids`, PLUS the verified blocked records
        for the requested-but-not-attempted subset:
          - each checkpoint is loaded via `load_checkpoint()` (itself
            re-verifying content_hash/document_id/run_id against THIS
            store); each blocked record is loaded via
            `BlockedDocumentStore.load_blocked()` (itself re-verifying its
            own content_hash/document_id/run_id);
          - all checkpoints must share IDENTICAL configuration (any
            divergence indicates tampering, since checkpoints from the
            same `run_id` are only ever legitimately produced from the
            same `compute_run_id()` inputs);
          - the FULL requested set's ordered document identities
            (record_id/pdf_path/pdf_sha256, drawn from a checkpoint when
            attempted or a blocked record when blocked) and configuration
            (including `max_batches`) are fed back through
            `compute_run_id()` and the result is required to equal THIS
            store's own `run_id` -- proving that the exact document
            ORDER and SET this manifest claims to describe (which
            `compute_run_id()` binds explicitly) is actually the request
            whose digest is this store's `run_id`, not merely a
            document set that happens to have valid-looking checkpoints
            sitting in this directory (review finding G: "duplicate/
            reordered document IDs are accepted" -- reversing or
            duplicating documents changes what `compute_run_id()` would
            have produced, so it now fails this check instead of
            silently matching via set-based coverage logic alone)."""
        if not document_ids:
            raise ValueError("cannot derive a manifest with zero document_ids")
        if len(set(document_ids)) != len(document_ids):
            raise CheckpointIntegrityError(
                f"duplicate document_id(s) in document_ids for run_id={self.run_id!r}: cannot derive "
                "a manifest describing the same document more than once"
            )

        requested = tuple(requested_document_ids) if requested_document_ids is not None else tuple(document_ids)
        if len(set(requested)) != len(requested):
            raise CheckpointIntegrityError(
                f"duplicate document_id(s) in requested_document_ids for run_id={self.run_id!r}"
            )
        attempted_set = set(document_ids)
        if not attempted_set.issubset(set(requested)):
            raise CheckpointIntegrityError(
                f"document_ids {sorted(attempted_set - set(requested))!r} are not present in "
                f"requested_document_ids for run_id={self.run_id!r}"
            )
        blocked_ids = tuple(d for d in requested if d not in attempted_set)

        checkpoints = {document_id: self.load_checkpoint(document_id) for document_id in document_ids}

        first = checkpoints[document_ids[0]]
        for document_id, checkpoint in checkpoints.items():
            for field_name in self._MANIFEST_CONFIG_FIELDS:
                if getattr(checkpoint, field_name) != getattr(first, field_name):
                    raise CheckpointIntegrityError(
                        f"checkpoint for document_id={document_id!r} has a different {field_name!r} "
                        f"than other checkpoints in run_id={self.run_id!r} -- checkpoints within the "
                        "same run must share identical configuration"
                    )

        blocked_store = self._blocked_store()
        identities = []
        for document_id in requested:
            if document_id in checkpoints:
                cp = checkpoints[document_id]
                identities.append(DocumentIdentity(record_id=document_id, pdf_path=cp.pdf_path, pdf_sha256=cp.pdf_sha256))
                continue
            blocked_record = blocked_store.load_blocked(document_id)
            if blocked_record is None:
                raise CheckpointIntegrityError(
                    f"document_id={document_id!r} is neither checkpointed nor recorded as blocked for "
                    f"run_id={self.run_id!r} -- cannot derive a manifest for an unresolved document"
                )
            identities.append(
                DocumentIdentity(
                    record_id=document_id, pdf_path=blocked_record.pdf_path, pdf_sha256=blocked_record.pdf_sha256,
                )
            )

        recomputed_run_id = compute_run_id(
            documents=identities,
            extraction_schema_hash=first.extraction_schema_hash,
            parser_name=first.parser_name, parser_version=first.parser_version,
            parser_options=DoclingParserOptions(**first.parser_options), max_batch_chars=first.max_batch_chars,
            prompt_template_version=first.prompt_template_version, output_schema_version=first.output_schema_version,
            model=first.model, model_config_id=first.model_config_id, reasoning_effort=first.reasoning_effort,
            max_completion_tokens=first.max_completion_tokens, expected_provider_model=first.expected_provider_model,
            artifacts_manifest_digest=first.artifacts_manifest_digest, max_batches=first.max_batches,
        )
        if recomputed_run_id != self.run_id:
            raise CheckpointIntegrityError(
                f"the requested documents' own order/identities and configuration reproduce "
                f"run_id={recomputed_run_id!r}, which does not match this store's own "
                f"run_id={self.run_id!r} -- this document set/order/configuration is not the request "
                "this run_id was computed from"
            )

        checkpointed_ids = tuple(document_ids)
        complete_ids = tuple(d for d in checkpointed_ids if checkpoints[d].result.is_complete)
        incomplete_ids = tuple(d for d in checkpointed_ids if not checkpoints[d].result.is_complete)
        checkpoint_hashes = {d: checkpoints[d].content_hash for d in checkpointed_ids}

        all_batches = [b for cp in checkpoints.values() for b in cp.result.batch_executions]
        provider_models_seen = tuple(sorted({b.provider_model for b in all_batches if b.provider_model is not None}))
        telemetry_summary = TelemetrySummary(
            total_prompt_tokens=sum(b.prompt_tokens or 0 for b in all_batches),
            total_completion_tokens=sum(b.completion_tokens or 0 for b in all_batches),
            total_tokens=sum(b.total_tokens or 0 for b in all_batches),
            distinct_provider_models=provider_models_seen,
            provider_model_consistent=all(pm == first.expected_provider_model for pm in provider_models_seen),
            provider_mismatch_count=sum(1 for b in all_batches if b.error == "ProviderModelMismatchError"),
        )

        base = dict(
            run_id=self.run_id,
            **{field_name: getattr(first, field_name) for field_name in self._MANIFEST_CONFIG_FIELDS},
            document_ids=checkpointed_ids,
            checkpointed_document_ids=checkpointed_ids,
            blocked_document_ids=blocked_ids,
            requested_document_ids=requested,
            complete_document_ids=complete_ids,
            incomplete_document_ids=incomplete_ids,
            checkpoint_hashes=checkpoint_hashes,
            telemetry_summary=telemetry_summary.model_dump(mode="json"),
            created_at=created_at,
        )
        manifest_hash = sha256_hex(base)
        return RunManifest(**base, manifest_hash=manifest_hash)

    def publish_manifest(
        self, *, document_ids: tuple[str, ...], created_at: str,
        requested_document_ids: Optional[tuple[str, ...]] = None,
    ) -> RunManifest:
        """The ONLY supported way to publish a run manifest -- derives
        EVERY field via `_derive_manifest()`, never from a caller-supplied
        `RunManifest` object (review finding E).

        `document_ids` are the ATTEMPTED (checkpointed) documents.
        `requested_document_ids`, additive and optional (default `None` --
        unchanged behavior for every caller that never blocks a document),
        names the full originally-requested set when it differs from
        `document_ids` -- see `_derive_manifest()`.

        A published manifest is IMMUTABLE: the existing manifest is first
        re-verified against the CURRENT checkpoints (via `_derive_manifest`
        using its own recorded `document_ids`/`requested_document_ids`/
        `created_at`) -- if it is already stale or inconsistent,
        publication is refused outright rather than silently basing a
        "no-op" decision on data that no longer matches reality (review
        finding G). If it is still consistent and the newly-requested
        `document_ids`/`requested_document_ids` derive to IDENTICAL
        content, the EXISTING manifest is returned unchanged (a true
        no-op -- no file write, no new `created_at`/`manifest_hash`). Any
        real difference is refused."""
        existing = self._read_manifest_file()
        if existing is not None:
            existing_rederived = self._derive_manifest(
                document_ids=existing.document_ids, created_at=existing.created_at,
                requested_document_ids=existing.requested_document_ids,
            )
            if existing_rederived.model_dump(mode="json") != existing.model_dump(mode="json"):
                raise CheckpointIntegrityError(
                    f"the already-published manifest for run_id={self.run_id!r} no longer matches its "
                    "own checkpoints -- refusing to publish until this inconsistency is resolved"
                )
            candidate = self._derive_manifest(
                document_ids=document_ids, created_at=existing.created_at,
                requested_document_ids=requested_document_ids,
            )
            if candidate.model_dump(mode="json") == existing.model_dump(mode="json"):
                return existing  # true no-op: no write, original created_at/manifest_hash preserved
            raise CheckpointIntegrityError(
                f"refusing to overwrite the already-published manifest for run_id={self.run_id!r} with "
                "different content -- a published manifest is immutable; only a byte-identical "
                "(save for created_at) republish is a no-op, any real difference is refused"
            )

        manifest = self._derive_manifest(
            document_ids=document_ids, created_at=created_at, requested_document_ids=requested_document_ids,
        )
        _atomic_write_json(self.manifest_path(), manifest.model_dump(mode="json"))
        return manifest

    def load_manifest(self) -> Optional[RunManifest]:
        """Re-verifies the loaded manifest against THIS store's run_id
        and the CURRENT on-disk checkpoints/blocked-records it claims to
        bind -- a self-hash alone only proves the file wasn't edited; it
        proves nothing about whether it still matches this store or its
        checkpoints (review finding G). Re-derives a fresh manifest from
        `loaded.document_ids`/`loaded.requested_document_ids` (via
        `_derive_manifest`, the SAME path `publish_manifest()` uses) and
        requires it to match the loaded manifest EXACTLY -- any divergence
        (wrong run, a copied-from-elsewhere manifest, a replaced/stale
        checkpoint or blocked record, tampered coverage/config/telemetry)
        raises `CheckpointIntegrityError` rather than silently returning
        outdated or foreign data."""
        loaded = self._read_manifest_file()
        if loaded is None:
            return None
        rederived = self._derive_manifest(
            document_ids=loaded.document_ids, created_at=loaded.created_at,
            requested_document_ids=loaded.requested_document_ids,
        )
        if rederived.model_dump(mode="json") != loaded.model_dump(mode="json"):
            raise CheckpointIntegrityError(
                f"manifest for run_id={loaded.run_id!r} no longer matches this store's own run_id or "
                "its current checkpoints -- it may belong to a different run, or a checkpoint it "
                "references has changed since publication"
            )
        return loaded


def _blocked_filename(document_id: str) -> str:
    """Same hash-derived-filename discipline as `_checkpoint_filename` --
    a `document_id` cannot escape `blocked_dir` or collide across runs."""
    return f"{sha256_hex(document_id)[:32]}.json"


class BlockedDocumentRecord(BaseModel):
    """The complete, self-describing, content-hashed envelope for one
    document durably blocked before any billed extraction call. Same
    self-hashing discipline as `DocumentCheckpoint` (review finding:
    `BlockedDocumentStore.is_blocked()` previously trusted bare file
    existence with no parsing or verification of the file's own claimed
    run/document/content -- a corrupt or fabricated file at the
    hash-derived path would silently suppress extraction with no error;
    this envelope, checked on every load, closes that)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    document_id: str
    pdf_path: str
    pdf_sha256: str
    reason: str
    batch_count: int
    max_batches: int
    parsed_document_sha256: str
    blocked_at: str
    content_hash: str = Field(min_length=64, max_length=64)

    @classmethod
    def build(
        cls, *, run_id: str, document_id: str, pdf_path: str, pdf_sha256: str, reason: str, batch_count: int,
        max_batches: int, parsed_document_sha256: str, blocked_at: str,
    ) -> "BlockedDocumentRecord":
        envelope = dict(
            run_id=run_id, document_id=document_id, pdf_path=pdf_path, pdf_sha256=pdf_sha256, reason=reason,
            batch_count=batch_count, max_batches=max_batches, parsed_document_sha256=parsed_document_sha256,
            blocked_at=blocked_at,
        )
        content_hash = sha256_hex(envelope)
        return cls(**envelope, content_hash=content_hash)

    def _recompute_content_hash(self) -> str:
        return sha256_hex(self.model_dump(mode="json", exclude={"content_hash"}))

    def verify(self, *, expected_document_id: Optional[str] = None, expected_run_id: Optional[str] = None) -> None:
        if expected_document_id is not None and self.document_id != expected_document_id:
            raise CheckpointIntegrityError(
                f"blocked-document record document_id mismatch: loaded for {expected_document_id!r} but "
                f"the envelope itself claims document_id={self.document_id!r}"
            )
        if expected_run_id is not None and self.run_id != expected_run_id:
            raise CheckpointIntegrityError(
                f"blocked-document record run_id mismatch: loaded from a store for "
                f"run_id={expected_run_id!r} but the envelope itself claims run_id={self.run_id!r} -- "
                "this record belongs to a DIFFERENT run and must not be trusted here"
            )
        if self._recompute_content_hash() != self.content_hash:
            raise CheckpointIntegrityError(
                f"blocked-document record content_hash mismatch for document_id={self.document_id!r} -- "
                "the file may be tampered, corrupted, or copied from a different run/document"
            )


class BlockedDocumentStore:
    """Durable record of a document that was reserved/checked but refused
    BEFORE any billed extraction call -- e.g. `run_extraction()`'s
    `max_batches` cap (`extraction/batch_cap.py`). Distinct from
    `ExtractionCheckpointStore`: a blocked document was never attempted,
    so it must never be represented as a `DocumentCheckpoint` (which
    always carries a real, even if incomplete, `DocumentExtractionResult`)
    and is never counted in a published `RunManifest`'s `document_ids`
    (attempted-only) -- it is instead named in the manifest's own
    `blocked_document_ids`/`requested_document_ids` fields.

    Lives under the same `output_dir/runs/<run_id>/` root as
    `ExtractionCheckpointStore`, in a sibling `blocked/` directory, so a
    restart naturally finds both an existing run's checkpoints and its
    blocked documents together."""

    def __init__(self, *, output_dir: str | Path, run_id: str) -> None:
        if not _SAFE_RUN_ID_RE.match(run_id):
            raise ValueError(
                f"run_id {run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        self.run_id = run_id
        runs_root = Path(output_dir) / "runs"
        self.run_dir = runs_root / run_id
        if not self.run_dir.resolve().is_relative_to(runs_root.resolve()):
            raise ValueError(f"run_id {run_id!r} resolves outside the runs root -- refusing to construct store")
        self.blocked_dir = self.run_dir / "blocked"

    def blocked_path(self, document_id: str) -> Path:
        return self.blocked_dir / _blocked_filename(document_id)

    def _load_and_verify(self, document_id: str) -> BlockedDocumentRecord:
        data = json.loads(self.blocked_path(document_id).read_text(encoding="utf-8"))
        record = BlockedDocumentRecord.model_validate(data)
        record.verify(expected_document_id=document_id, expected_run_id=self.run_id)
        return record

    def is_blocked(self, document_id: str) -> bool:
        """Review finding: bare file existence previously stood in for
        "blocked" with no parsing or verification -- a corrupt or
        fabricated file would silently suppress extraction. Absence still
        means "not blocked"; any file that DOES exist is now parsed and
        verified (self-hash, claimed run_id/document_id) before being
        trusted, raising `CheckpointIntegrityError` on any inconsistency
        rather than silently accepting or silently ignoring it."""
        if not self.blocked_path(document_id).exists():
            return False
        self._load_and_verify(document_id)
        return True

    def load_blocked(self, document_id: str) -> Optional[BlockedDocumentRecord]:
        """The verified record for `document_id`, or `None` if it was
        never blocked -- used by `ExtractionCheckpointStore._derive_manifest()`
        to recover a blocked document's identity (pdf_path/pdf_sha256)
        when recomputing `compute_run_id()` over the full requested set."""
        if not self.blocked_path(document_id).exists():
            return None
        return self._load_and_verify(document_id)

    def record_blocked(
        self, *, document_id: str, pdf_path: str, pdf_sha256: str, reason: str, batch_count: int,
        max_batches: int, parsed_document_sha256: str, blocked_at: str,
    ) -> None:
        """Idempotent: re-recording the SAME document_id with identical
        content (aside from `blocked_at`) is a no-op; recording it with
        DIFFERENT content raises, since a blocked document's reason
        should never silently change underneath a restart."""
        candidate = BlockedDocumentRecord.build(
            run_id=self.run_id, document_id=document_id, pdf_path=pdf_path, pdf_sha256=pdf_sha256, reason=reason,
            batch_count=batch_count, max_batches=max_batches, parsed_document_sha256=parsed_document_sha256,
            blocked_at=blocked_at,
        )
        path = self.blocked_path(document_id)
        if path.exists():
            existing = self._load_and_verify(document_id)
            if existing.model_dump(mode="json", exclude={"blocked_at", "content_hash"}) != candidate.model_dump(
                mode="json", exclude={"blocked_at", "content_hash"}
            ):
                raise CheckpointIntegrityError(
                    f"blocked-document record for document_id={document_id!r} already exists with "
                    f"different content: {existing.model_dump(mode='json')!r} != "
                    f"{candidate.model_dump(mode='json')!r}"
                )
            return
        _atomic_write_json(path, candidate.model_dump(mode="json"))

    def blocked_document_ids(self) -> tuple[str, ...]:
        if not self.blocked_dir.is_dir():
            return ()
        ids = []
        for path in sorted(self.blocked_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            record = BlockedDocumentRecord.model_validate(data)
            record.verify(expected_run_id=self.run_id)
            if path.name != _blocked_filename(record.document_id):
                raise CheckpointIntegrityError(
                    f"blocked-document record at {path} does not live at the filename its own claimed "
                    f"document_id={record.document_id!r} derives -- possible filename/content mismatch"
                )
            ids.append(record.document_id)
        return tuple(ids)


class VerifiedRunBundle:
    """The ONE verified, store-derived source of truth for a formally
    run-bound report (review finding L: "report builders do not
    establish one verified run/artifact graph... each sub-object may be
    locally valid while the top-level report describes no real run that
    ever existed"). `manifest` and every checkpoint in `checkpoints_by_id`
    are ALL derived from a SINGLE `store.load_manifest()` call (itself
    already fully re-verified against the store's own run_id and current
    checkpoints -- see `ExtractionCheckpointStore.load_manifest()`), so
    there is no parameter through which a caller building a report from
    this bundle could cross-join results/checkpoints/run_id from
    different runs -- the bundle's own `run_id`, `results`, and
    `checkpoints` are always projections of that ONE manifest.

    This is the RECOMMENDED way to gather report inputs. A caller with no
    published run to bind against (a genuinely ad hoc, exploratory
    comparison) uses `reports.build_ad_hoc_extraction_report()` instead
    -- deliberately named so it cannot be mistaken for a formally
    run-bound report; that path accepts independent objects and performs
    no cross-run/cross-schema verification of its own beyond what each
    individual report builder already checks.

    Review finding L-residual: the guarantee above held only at the
    MOMENT `from_store()` ran -- `__init__` was public (a caller could
    hand-construct a bundle from an inconsistent manifest/checkpoint
    pairing) and `checkpoints_by_id` was an ordinary mutable dict (a
    caller could replace a bound checkpoint with a different, internally
    self-consistent one AFTER `from_store()` verified it, and
    `build_full_report()` would score the substitution while still
    emitting the original manifest hash). Both are closed here:
    `checkpoints_by_id` is a read-only `MappingProxyType` view over a
    dict `__init__` defensively COPIES (the caller's own dict, and any
    later mutation of it, can never reach the bundle); and `verify()` --
    called by `build_full_report()` at the start of every call, not only
    inside the optional `from_store()` factory -- re-checks the manifest
    self-hash, exact ordered document-key coverage, every checkpoint's
    own content_hash/document_id/run_id, every checkpoint's hash against
    the manifest's own binding, every checkpoint's COMPLETE configuration
    against the manifest's own, and each checkpoint's completeness
    against the manifest's own complete/incomplete classification --
    directly (or forgeably) constructed bundles are caught the same way
    mutated ones are, since both are just `verify()` failing."""

    def __init__(self, *, manifest: RunManifest, checkpoints_by_id: dict[str, DocumentCheckpoint]) -> None:
        self.manifest = manifest
        self._checkpoints_by_id = dict(checkpoints_by_id)

    @classmethod
    def from_store(cls, store: ExtractionCheckpointStore) -> "VerifiedRunBundle":
        manifest = store.load_manifest()
        if manifest is None:
            raise ValueError(f"no published manifest found for run_id={store.run_id!r}")
        checkpoints_by_id: dict[str, DocumentCheckpoint] = {}
        for document_id in manifest.document_ids:
            checkpoint = store.load_checkpoint(document_id)
            claimed_hash = manifest.checkpoint_hashes[document_id]
            if checkpoint.content_hash != claimed_hash:
                raise CheckpointIntegrityError(
                    f"checkpoint for document_id={document_id!r} (content_hash={checkpoint.content_hash!r}) "
                    f"does not match the hash bound by the verified manifest ({claimed_hash!r})"
                )
            checkpoints_by_id[document_id] = checkpoint
        return cls(manifest=manifest, checkpoints_by_id=checkpoints_by_id)

    @property
    def checkpoints_by_id(self) -> MappingProxyType:
        """Read-only view -- `bundle.checkpoints_by_id["d1"] = other`
        raises `TypeError`, never silently substitutes a bound
        checkpoint after verification."""
        return MappingProxyType(self._checkpoints_by_id)

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def results(self) -> list[DocumentExtractionResult]:
        return [self._checkpoints_by_id[d].result for d in self.manifest.document_ids]

    @property
    def checkpoints(self) -> list[DocumentCheckpoint]:
        return [self._checkpoints_by_id[d] for d in self.manifest.document_ids]

    def verify(self) -> None:
        """Re-verifies this bundle's OWN internal consistency -- called
        by `reports.build_full_report()` at the start of every call, so
        neither a directly (forged) constructed bundle nor one mutated
        after `from_store()` can reach the point of computing a metric.
        Raises `CheckpointIntegrityError` on any inconsistency."""
        self.manifest.verify_manifest_hash()

        checkpoint_ids = set(self._checkpoints_by_id)
        manifest_ids = set(self.manifest.document_ids)
        if checkpoint_ids != manifest_ids:
            raise CheckpointIntegrityError(
                f"bundle checkpoints_by_id keys {sorted(checkpoint_ids)!r} do not exactly match "
                f"manifest.document_ids {sorted(manifest_ids)!r}"
            )

        for document_id in self.manifest.document_ids:
            checkpoint = self._checkpoints_by_id[document_id]
            checkpoint.verify(expected_document_id=document_id, expected_run_id=self.manifest.run_id)

            claimed_hash = self.manifest.checkpoint_hashes[document_id]
            if checkpoint.content_hash != claimed_hash:
                raise CheckpointIntegrityError(
                    f"checkpoint for document_id={document_id!r} (content_hash={checkpoint.content_hash!r}) "
                    f"does not match the hash bound by the manifest ({claimed_hash!r})"
                )

            for field_name in ExtractionCheckpointStore._MANIFEST_CONFIG_FIELDS:
                checkpoint_value, manifest_value = getattr(checkpoint, field_name), getattr(self.manifest, field_name)
                if checkpoint_value != manifest_value:
                    raise CheckpointIntegrityError(
                        f"checkpoint for document_id={document_id!r} has {field_name}={checkpoint_value!r}, "
                        f"which does not match the manifest's own {manifest_value!r}"
                    )

            is_complete = checkpoint.result.is_complete
            if is_complete and document_id not in self.manifest.complete_document_ids:
                raise CheckpointIntegrityError(
                    f"document_id={document_id!r}'s checkpoint is complete but the manifest does not "
                    "list it in complete_document_ids"
                )
            if not is_complete and document_id not in self.manifest.incomplete_document_ids:
                raise CheckpointIntegrityError(
                    f"document_id={document_id!r}'s checkpoint is incomplete but the manifest does not "
                    "list it in incomplete_document_ids"
                )
