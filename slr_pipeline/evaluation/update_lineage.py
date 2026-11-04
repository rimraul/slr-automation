"""Origin/processing record ledger for an incremental post-cutoff review update.

An update wave's corpus mixes records from two independent dimensions:

- ``origin``: whether a record is one of the parent reproduction's
  ``deduplicated_records`` (``"parent"``) or was found only by the update wave's
  own new search (``"new"``).
- ``processing``: why a record is screened under the update wave's protocol --
  ``"reused"`` (a parent-origin record whose prior screening decision still
  applies unchanged), ``"protocol_migration_rescreen"`` (a parent-origin record
  re-screened because the eligibility protocol changed), or ``"first_screen"``
  (a new-origin record screened for the first time).

These are independent, not one mutually-exclusive three-way split: a stage's
PRISMA count is reported per (origin, processing) pair. Every record must have
exactly one of the two valid (origin, processing) pairings -- there is no
"unscreened" state in a closed ledger.

Per-stage parent hashes are copied VERBATIM from the parent reproduction's own
verified artifact set (never recomputed here -- the same discipline as
``slr_pipeline.continuation.lineage.write_continuation_lineage``), which is why
``from_verified_parent()`` requires a
``slr_pipeline.evaluation.bundle_artifacts.VerifiedReproductionArtifactSet``,
not a bare directory: an unverified directory (tampered file, stale
cross-reference, mismatched adjudication journal) must never be treated as a
closed parent.

There is currently no durable artifact anywhere binding a screening decision to
the eligibility/criteria hash that produced it, so this module never uses
``"reused"`` for title/abstract screening: every parent-origin record in
``deduplicated_records`` is re-screened as ``"protocol_migration_rescreen"``.
``screened_records_parent_hash`` identifies only the survivors of the parent's
narrower-criteria screen, not the full decision population -- it is recorded
for provenance only; ``screened_records_provenance_note()`` renders this from
the validated ``deduplicated_parent_record_count`` /
``screened_records_survivor_count`` fields rather than a hardcoded string, so
the model stays generic across reviews.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from slr_pipeline.config import ReviewConfig
from slr_pipeline.evaluation.bundle_artifacts import VerifiedReproductionArtifactSet
from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
from slr_pipeline.extraction.hashing import sha256_hex

Origin = Literal["parent", "new"]
Processing = Literal["reused", "protocol_migration_rescreen", "first_screen"]
ParentStatus = Literal["no_verified_parent", "verified"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_VALID_PAIRINGS: frozenset[tuple[Origin, Processing]] = frozenset(
    {
        ("parent", "reused"),
        ("parent", "protocol_migration_rescreen"),
        ("new", "first_screen"),
    }
)


def _is_sha256_hex(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _records_ledger_hash(records: tuple["RecordLineageEntry", ...]) -> str:
    """Content hash over the ledger itself, not just its length.

    A count-only check (``parent_entry_count == deduplicated_parent_record_count``)
    passes unchanged if one record_id is swapped for another with the same
    origin/processing. This hash binds the actual id/origin/processing content
    so any such substitution is detected.
    """

    return sha256_hex(
        [
            {"record_id": e.record_id, "origin": e.origin, "processing": e.processing}
            for e in records
        ]
    )


class RecordLineageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    origin: Origin
    processing: Processing

    @model_validator(mode="after")
    def _check_origin_processing_pairing(self) -> "RecordLineageEntry":
        if (self.origin, self.processing) not in _VALID_PAIRINGS:
            raise ValueError(
                f"record {self.record_id!r}: invalid (origin={self.origin!r}, "
                f"processing={self.processing!r}) pairing; valid pairings are "
                f"{sorted(_VALID_PAIRINGS)}"
            )
        return self


class UpdateLineage(BaseModel):
    """Per-stage provenance, binding hashes, and a per-record origin/processing ledger.

    ``deduplicated_records`` ledger entries must exactly cover the parent's real
    deduplicated-record id set (``build()``/``from_verified_parent()`` enforce
    this): every one of those records is processed exactly once, never
    omitted, never double-counted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    review_id: str

    # Binding: which exact parent/update artifacts this lineage was computed
    # against. All four are required so a lineage file can never be silently
    # replayed against a different spec/config/gold-standard than the one it
    # was built from.
    parent_reproduction_spec_hash: str
    parent_gold_standard_hash: str
    update_reproduction_spec_hash: str
    review_config_hash: str

    raw_records_parent_hash: str
    deduplicated_records_parent_hash: str

    screened_records_parent_hash: str
    deduplicated_parent_record_count: int
    screened_records_survivor_count: int

    full_text_records_parent_hash: Optional[str] = None
    full_text_records_parent_status: ParentStatus = "no_verified_parent"

    included_records_parent_hash: Optional[str] = None
    included_records_parent_status: ParentStatus = "no_verified_parent"

    records: tuple[RecordLineageEntry, ...]
    records_hash: str

    @model_validator(mode="after")
    def _check_ledger_integrity(self) -> "UpdateLineage":
        expected_records_hash = _records_ledger_hash(self.records)
        if self.records_hash != expected_records_hash:
            raise ValueError(
                "records_hash does not match the content of the records ledger "
                "(record_id/origin/processing) -- a record may have been "
                "substituted, reordered, or edited without updating records_hash"
            )
        for field_name in (
            "parent_reproduction_spec_hash",
            "parent_gold_standard_hash",
            "update_reproduction_spec_hash",
            "review_config_hash",
            "raw_records_parent_hash",
            "deduplicated_records_parent_hash",
            "screened_records_parent_hash",
        ):
            value = getattr(self, field_name)
            if not _is_sha256_hex(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")

        for field_name, status_name in (
            ("full_text_records_parent_hash", "full_text_records_parent_status"),
            ("included_records_parent_hash", "included_records_parent_status"),
        ):
            value = getattr(self, field_name)
            status = getattr(self, status_name)
            if status == "no_verified_parent":
                if value is not None:
                    raise ValueError(f"{field_name} must be None when {status_name} is no_verified_parent")
            else:
                if value is None:
                    raise ValueError(f"{field_name} is required when {status_name} is verified")
                if not _is_sha256_hex(value):
                    raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")

        if self.screened_records_survivor_count > self.deduplicated_parent_record_count:
            raise ValueError(
                "screened_records_survivor_count cannot exceed "
                "deduplicated_parent_record_count"
            )
        if self.deduplicated_parent_record_count < 0 or self.screened_records_survivor_count < 0:
            raise ValueError("record counts must be non-negative")

        ids = [entry.record_id for entry in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("records ledger contains duplicate record_id entries")

        parent_entry_count = sum(1 for entry in self.records if entry.origin == "parent")
        if parent_entry_count != self.deduplicated_parent_record_count:
            raise ValueError(
                f"records ledger has {parent_entry_count} origin=parent entries, "
                f"which does not match deduplicated_parent_record_count="
                f"{self.deduplicated_parent_record_count}"
            )
        return self

    def screened_records_provenance_note(self) -> str:
        return (
            f"Parent screened_records_hash identifies only "
            f"{self.screened_records_survivor_count} of "
            f"{self.deduplicated_parent_record_count} deduplicated_records "
            "survivors under the parent's narrower criteria -- not the full "
            "decision population. All "
            f"{self.deduplicated_parent_record_count} are re-screened under "
            "the update wave's protocol; see records below."
        )

    def prisma_counts(self) -> dict[str, int]:
        """Counts keyed as ``"{origin}/{processing}"``, e.g. ``"parent/protocol_migration_rescreen"``."""

        counts: dict[str, int] = {}
        for entry in self.records:
            key = f"{entry.origin}/{entry.processing}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @classmethod
    def build(
        cls,
        *,
        review_id: str,
        parent_reproduction_spec_hash: str,
        parent_gold_standard_hash: str,
        update_reproduction_spec_hash: str,
        review_config_hash: str,
        raw_records_parent_hash: str,
        deduplicated_records_parent_hash: str,
        screened_records_parent_hash: str,
        screened_records_survivor_count: int,
        parent_deduplicated_record_ids: tuple[str, ...],
        new_record_ids: tuple[str, ...] = (),
    ) -> "UpdateLineage":
        """All parent-origin records are protocol_migration_rescreen; all new-origin are first_screen."""

        if len(set(parent_deduplicated_record_ids)) != len(parent_deduplicated_record_ids):
            raise ValueError("parent_deduplicated_record_ids contains duplicates")
        if len(set(new_record_ids)) != len(new_record_ids):
            raise ValueError("new_record_ids contains duplicates")
        overlap = set(parent_deduplicated_record_ids) & set(new_record_ids)
        if overlap:
            raise ValueError(
                f"record ids appear in both parent and new sets: {sorted(overlap)}"
            )

        records = tuple(
            RecordLineageEntry(
                record_id=record_id,
                origin="parent",
                processing="protocol_migration_rescreen",
            )
            for record_id in parent_deduplicated_record_ids
        ) + tuple(
            RecordLineageEntry(
                record_id=record_id,
                origin="new",
                processing="first_screen",
            )
            for record_id in new_record_ids
        )

        return cls.model_validate(
            {
                "review_id": review_id,
                "parent_reproduction_spec_hash": parent_reproduction_spec_hash,
                "parent_gold_standard_hash": parent_gold_standard_hash,
                "update_reproduction_spec_hash": update_reproduction_spec_hash,
                "review_config_hash": review_config_hash,
                "raw_records_parent_hash": raw_records_parent_hash,
                "deduplicated_records_parent_hash": deduplicated_records_parent_hash,
                "screened_records_parent_hash": screened_records_parent_hash,
                "deduplicated_parent_record_count": len(parent_deduplicated_record_ids),
                "screened_records_survivor_count": screened_records_survivor_count,
                "records": records,
                "records_hash": _records_ledger_hash(records),
            }
        )

    @classmethod
    def from_verified_parent(
        cls,
        *,
        review_id: str,
        parent: VerifiedReproductionArtifactSet,
        update_spec: ReproductionSpec,
        review_config: ReviewConfig,
        new_record_ids: tuple[str, ...] = (),
    ) -> "UpdateLineage":
        """Bind a new update-wave ledger to an already-verified parent reproduction.

        ``parent`` must come from
        ``slr_pipeline.evaluation.bundle_artifacts.verify_reproduction_artifact_set()``
        -- this function never re-verifies the parent itself, so an unverified
        directory can never become a "closed parent" here.

        Raises ``ValueError`` if ``review_id`` does not match both the parent
        spec's and the update spec's own ``review_id``, if ``update_spec`` is
        not a ``"review_update"``, or if ``update_spec``'s expected gold
        standard hash does not match the parent's actual gold standard --
        each of these binds the lineage to the one parent/update pair it was
        actually computed from, not merely one with matching stage counts.
        """

        if review_id != parent.spec.review_id:
            raise ValueError(
                f"review_id={review_id!r} does not match parent.spec.review_id="
                f"{parent.spec.review_id!r}"
            )
        if review_id != update_spec.review_id:
            raise ValueError(
                f"review_id={review_id!r} does not match update_spec.review_id="
                f"{update_spec.review_id!r}"
            )
        if update_spec.reproduction_mode != "review_update":
            raise ValueError(
                f"update_spec.reproduction_mode={update_spec.reproduction_mode!r} "
                "must be 'review_update'"
            )
        actual_gold_hash = parent.gold_standard.content_hash()
        if update_spec.expected_gold_standard_hash != actual_gold_hash:
            raise ValueError(
                f"update_spec.expected_gold_standard_hash="
                f"{update_spec.expected_gold_standard_hash!r} does not match "
                f"parent.gold_standard.content_hash()={actual_gold_hash!r}"
            )

        parent_ids = tuple(record.id for record in parent.stage_corpus.deduplicated_records)
        return cls.build(
            review_id=review_id,
            parent_reproduction_spec_hash=parent.spec.spec_hash(),
            parent_gold_standard_hash=parent.gold_standard.content_hash(),
            update_reproduction_spec_hash=update_spec.spec_hash(),
            review_config_hash=sha256_hex(review_config.model_dump(mode="json")),
            raw_records_parent_hash=parent.stage_corpus.raw_records_hash,
            deduplicated_records_parent_hash=parent.stage_corpus.deduplicated_records_hash,
            screened_records_parent_hash=parent.stage_corpus.screened_records_hash,
            screened_records_survivor_count=len(parent.stage_corpus.screened_records),
            parent_deduplicated_record_ids=parent_ids,
            new_record_ids=new_record_ids,
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "UpdateLineage":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "Origin",
    "Processing",
    "ParentStatus",
    "RecordLineageEntry",
    "UpdateLineage",
]
