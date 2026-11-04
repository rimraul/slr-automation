"""Hash-bound record corpora used for stage-by-stage reproduction evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.state import Record


def _records_hash(records: tuple[Record, ...]) -> str:
    return sha256_hex([record.model_dump(mode="json") for record in records])


class StageCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_records: tuple[Record, ...]
    deduplicated_records: tuple[Record, ...]
    screened_records: tuple[Record, ...]
    full_text_records: Optional[tuple[Record, ...]] = None
    included_records: Optional[tuple[Record, ...]] = None

    raw_records_hash: str
    deduplicated_records_hash: str
    screened_records_hash: str
    full_text_records_hash: Optional[str] = None
    included_records_hash: Optional[str] = None

    @model_validator(mode="after")
    def _verify_hashes_and_stage_subsets(self) -> "StageCorpus":
        stages = (
            ("raw_records", self.raw_records, self.raw_records_hash),
            (
                "deduplicated_records",
                self.deduplicated_records,
                self.deduplicated_records_hash,
            ),
            ("screened_records", self.screened_records, self.screened_records_hash),
            ("full_text_records", self.full_text_records, self.full_text_records_hash),
            ("included_records", self.included_records, self.included_records_hash),
        )
        previous_name: str | None = None
        previous_ids: set[str] | None = None
        for name, records, digest in stages:
            if records is None:
                if digest is not None:
                    raise ValueError(f"{name}_hash must be None when {name} is absent")
                continue
            if digest is None:
                raise ValueError(f"{name}_hash is required when {name} is present")
            record_ids = [record.id for record in records]
            if any(not record_id.strip() for record_id in record_ids):
                raise ValueError(f"{name} contains a blank record id")
            if len(record_ids) != len(set(record_ids)):
                raise ValueError(f"{name} contains duplicate record ids")
            expected_hash = _records_hash(records)
            if digest != expected_hash:
                raise ValueError(f"{name}_hash does not match normalized record content")
            id_set = set(record_ids)
            if previous_ids is not None and not id_set.issubset(previous_ids):
                introduced = sorted(id_set - previous_ids)
                raise ValueError(
                    f"{name} introduces record ids absent from {previous_name}: {introduced}"
                )
            previous_name = name
            previous_ids = id_set
        return self

    @classmethod
    def build(
        cls,
        *,
        raw_records: tuple[Record, ...],
        deduplicated_records: tuple[Record, ...],
        screened_records: tuple[Record, ...],
        full_text_records: Optional[tuple[Record, ...]] = None,
        included_records: Optional[tuple[Record, ...]] = None,
    ) -> "StageCorpus":
        values = {
            "raw_records": raw_records,
            "deduplicated_records": deduplicated_records,
            "screened_records": screened_records,
            "full_text_records": full_text_records,
            "included_records": included_records,
            "raw_records_hash": _records_hash(raw_records),
            "deduplicated_records_hash": _records_hash(deduplicated_records),
            "screened_records_hash": _records_hash(screened_records),
            "full_text_records_hash": (
                _records_hash(full_text_records)
                if full_text_records is not None
                else None
            ),
            "included_records_hash": (
                _records_hash(included_records)
                if included_records is not None
                else None
            ),
        }
        return cls.model_validate(values)

    @classmethod
    def from_files(
        cls,
        *,
        raw_records: Path,
        deduplicated_records: Path,
        screened_records: Path,
        full_text_records: Optional[Path] = None,
        included_records: Optional[Path] = None,
    ) -> "StageCorpus":
        def load(path: Path) -> tuple[Record, ...]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"stage record file must contain a list: {path}")
            return tuple(Record.model_validate(item) for item in payload)

        return cls.build(
            raw_records=load(raw_records),
            deduplicated_records=load(deduplicated_records),
            screened_records=load(screened_records),
            full_text_records=load(full_text_records) if full_text_records else None,
            included_records=load(included_records) if included_records else None,
        )


__all__ = ["StageCorpus"]
