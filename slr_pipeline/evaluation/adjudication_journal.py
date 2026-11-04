"""Append-only, hash-chained journal for evaluation adjudications."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.ai_traceability import kernel_lock
from slr_pipeline.extraction.hashing import canonical_json_bytes, sha256_hex
from slr_pipeline.extraction.resolution_suggestions import atomic_write_bytes, _fsync_dir

from .adjudication import AdjudicationRecord


MATCHER_VERSION = "gold-matcher-v1"


class EvaluationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_hash: str
    gold_standard_hash: str
    raw_records_hash: str
    deduplicated_records_hash: str
    screened_records_hash: str
    full_text_records_hash: Optional[str] = None
    included_records_hash: Optional[str] = None
    matcher_version: str = MATCHER_VERSION

    @model_validator(mode="after")
    def _validate_hashes(self) -> "EvaluationBinding":
        for name in (
            "spec_hash",
            "gold_standard_hash",
            "raw_records_hash",
            "deduplicated_records_hash",
            "screened_records_hash",
            "full_text_records_hash",
            "included_records_hash",
        ):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if not self.matcher_version.strip():
            raise ValueError("matcher_version must not be blank")
        return self


class AdjudicationJournalEntry(AdjudicationRecord):
    sequence: int = Field(ge=1)
    binding: EvaluationBinding
    previous_entry_hash: Optional[str] = None
    entry_hash: str

    @model_validator(mode="after")
    def _verify_chain_identity(self) -> "AdjudicationJournalEntry":
        if self.previous_entry_hash is not None and (
            len(self.previous_entry_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in self.previous_entry_hash)
        ):
            raise ValueError("previous_entry_hash must be a lowercase SHA-256 digest")
        expected = sha256_hex(self.model_dump(mode="json", exclude={"entry_hash"}))
        if self.entry_hash != expected:
            raise ValueError("entry_hash does not match journal entry content")
        return self


def _build_entry(
    record: AdjudicationRecord,
    *,
    binding: EvaluationBinding,
    sequence: int,
    previous_entry_hash: str | None,
) -> AdjudicationJournalEntry:
    payload = {
        **record.model_dump(mode="json"),
        "sequence": sequence,
        "binding": binding.model_dump(mode="json"),
        "previous_entry_hash": previous_entry_hash,
    }
    return AdjudicationJournalEntry.model_validate(
        {**payload, "entry_hash": sha256_hex(payload)}
    )


def validate_journal(entries: Iterable[AdjudicationJournalEntry]) -> None:
    ordered = list(entries)
    previous_hash: str | None = None
    binding: EvaluationBinding | None = None
    by_id: dict[str, list[AdjudicationJournalEntry]] = {}
    for expected_sequence, entry in enumerate(ordered, start=1):
        if entry.sequence != expected_sequence:
            raise ValueError("adjudication journal sequence is not contiguous")
        if entry.previous_entry_hash != previous_hash:
            raise ValueError("adjudication journal hash chain is broken")
        if binding is None:
            binding = entry.binding
        elif entry.binding != binding:
            raise ValueError("adjudication journal binding changed within one run")
        previous_hash = entry.entry_hash
        by_id.setdefault(entry.adjudication_id, []).append(entry)

    for adjudication_id, history in by_id.items():
        if history[0].resolution != "pending":
            raise ValueError(
                f"adjudication {adjudication_id} must begin with a pending snapshot"
            )
        if len(history) > 2:
            raise ValueError(f"adjudication {adjudication_id} has multiple resolutions")
        if len(history) == 2:
            if history[1].resolution == "pending":
                raise ValueError(
                    f"adjudication {adjudication_id} repeats a pending snapshot"
                )
            first = history[0]
            second = history[1]
            immutable_fields = (
                "run_id",
                "tier",
                "candidate_gold_paper_ids",
                "candidate_pipeline_record_ids",
                "scores",
            )
            if any(getattr(first, name) != getattr(second, name) for name in immutable_fields):
                raise ValueError(
                    f"adjudication {adjudication_id} changed immutable candidates"
                )


def _parse_journal_bytes(
    data: bytes, *, allow_torn_tail: bool
) -> tuple[list[AdjudicationJournalEntry], bytes, bool]:
    if not data:
        return [], b"", False
    lines = data.splitlines(keepends=True)
    entries: list[AdjudicationJournalEntry] = []
    valid_lines: list[bytes] = []
    torn = False
    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        if not line.endswith(b"\n"):
            if allow_torn_tail and is_last:
                torn = True
                break
            raise ValueError("adjudication journal has a torn final line")
        try:
            payload = json.loads(line.decode("utf-8"))
            entry = AdjudicationJournalEntry.model_validate(payload)
        except Exception as exc:
            if allow_torn_tail and is_last:
                torn = True
                break
            raise ValueError(
                f"invalid adjudication journal line {index + 1}"
            ) from exc
        entries.append(entry)
        valid_lines.append(line)
    validate_journal(entries)
    return entries, b"".join(valid_lines), torn


def verify_adjudication_journal_bytes(data: bytes) -> list[AdjudicationJournalEntry]:
    entries, _prefix, torn = _parse_journal_bytes(data, allow_torn_tail=False)
    if torn:  # pragma: no cover - strict parser never returns torn
        raise ValueError("published adjudication journal is torn")
    return entries


class EvaluationAdjudicationStore:
    """Concurrency-safe live-workspace journal; published copies are read-only."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def read(self) -> list[AdjudicationJournalEntry]:
        if not self.path.exists():
            return []
        with kernel_lock(
            self.lock_path,
            description=f"evaluation adjudication journal {self.path}",
        ):
            if not self.path.exists():
                return []
            return verify_adjudication_journal_bytes(self.path.read_bytes())

    def repair_torn_tail(self) -> list[AdjudicationJournalEntry]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with kernel_lock(
            self.lock_path,
            description=f"evaluation adjudication journal {self.path}",
        ):
            return self._repair_torn_tail_locked()

    def _repair_torn_tail_locked(self) -> list[AdjudicationJournalEntry]:
        if not self.path.exists():
            return []
        entries, prefix, torn = _parse_journal_bytes(
            self.path.read_bytes(), allow_torn_tail=True
        )
        if torn:
            atomic_write_bytes(self.path, prefix)
        return entries

    def _append_locked(
        self,
        entries: list[AdjudicationJournalEntry],
        record: AdjudicationRecord,
        binding: EvaluationBinding,
    ) -> AdjudicationJournalEntry:
        entry = _build_entry(
            record,
            binding=binding,
            sequence=len(entries) + 1,
            previous_entry_hash=entries[-1].entry_hash if entries else None,
        )
        validate_journal([*entries, entry])
        line = canonical_json_bytes(entry.model_dump(mode="json")) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.path.parent)
        return entry

    def append(
        self, record: AdjudicationRecord, *, binding: EvaluationBinding
    ) -> AdjudicationJournalEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with kernel_lock(
            self.lock_path,
            description=f"evaluation adjudication journal {self.path}",
        ):
            entries = self._repair_torn_tail_locked()
            return self._append_locked(entries, record, binding)

    def append_pending_if_absent(
        self, record: AdjudicationRecord, *, binding: EvaluationBinding
    ) -> AdjudicationJournalEntry:
        if record.resolution != "pending":
            raise ValueError("append_pending_if_absent requires a pending record")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with kernel_lock(
            self.lock_path,
            description=f"evaluation adjudication journal {self.path}",
        ):
            entries = self._repair_torn_tail_locked()
            if entries and entries[0].binding != binding:
                raise ValueError(
                    "evaluation adjudication journal is bound to different inputs"
                )
            existing = [
                entry
                for entry in entries
                if entry.adjudication_id == record.adjudication_id
            ]
            if existing:
                return existing[-1]
            return self._append_locked(entries, record, binding)

    def latest_state_projection(self) -> dict[str, AdjudicationRecord]:
        return latest_state_projection(self.read())


def latest_state_projection(
    entries: Iterable[AdjudicationJournalEntry],
) -> dict[str, AdjudicationRecord]:
    ordered = list(entries)
    validate_journal(ordered)
    projection: dict[str, AdjudicationRecord] = {}
    record_fields = set(AdjudicationRecord.model_fields)
    for entry in ordered:
        payload = entry.model_dump(mode="json", include=record_fields)
        projection[entry.adjudication_id] = AdjudicationRecord.model_validate(payload)
    return projection


__all__ = [
    "AdjudicationJournalEntry",
    "EvaluationAdjudicationStore",
    "EvaluationBinding",
    "MATCHER_VERSION",
    "latest_state_projection",
    "validate_journal",
    "verify_adjudication_journal_bytes",
]
