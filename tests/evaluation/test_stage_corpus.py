from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slr_pipeline.evaluation.stage_corpus import StageCorpus
from slr_pipeline.state import Record


def _record(record_id: str) -> Record:
    return Record(id=record_id, source="test", source_id=record_id, title=record_id)


def test_build_hashes_every_stage_and_accepts_monotone_subsets() -> None:
    raw = (_record("a"), _record("b"), _record("c"))
    corpus = StageCorpus.build(
        raw_records=raw,
        deduplicated_records=raw[:2],
        screened_records=raw[:1],
        full_text_records=raw[:1],
        included_records=raw[:1],
    )

    assert len(corpus.raw_records_hash) == 64
    assert len(corpus.full_text_records_hash or "") == 64
    assert len(corpus.included_records_hash or "") == 64


def test_stage_cannot_introduce_a_new_record_id() -> None:
    raw = (_record("a"),)
    valid = StageCorpus.build(
        raw_records=raw,
        deduplicated_records=raw,
        screened_records=raw,
    )
    payload = valid.model_dump(mode="json")
    payload["screened_records"] = [_record("new").model_dump(mode="json")]
    from slr_pipeline.extraction.hashing import sha256_hex

    payload["screened_records_hash"] = sha256_hex(payload["screened_records"])

    with pytest.raises(ValidationError, match="introduces record ids"):
        StageCorpus.model_validate(payload)


def test_included_stage_uses_previous_present_stage_when_full_text_absent() -> None:
    raw = (_record("a"), _record("b"))
    corpus = StageCorpus.build(
        raw_records=raw,
        deduplicated_records=raw,
        screened_records=raw[:1],
        included_records=raw[:1],
    )
    assert corpus.full_text_records is None

    with pytest.raises(ValidationError, match="introduces record ids"):
        StageCorpus.build(
            raw_records=raw,
            deduplicated_records=raw,
            screened_records=raw[:1],
            included_records=(raw[1],),
        )


def test_hash_tampering_and_optional_hash_mismatch_fail_closed() -> None:
    record = (_record("a"),)
    corpus = StageCorpus.build(
        raw_records=record,
        deduplicated_records=record,
        screened_records=record,
    )
    with pytest.raises(ValidationError, match="does not match"):
        StageCorpus.model_validate(
            {**corpus.model_dump(mode="json"), "raw_records_hash": "0" * 64}
        )
    with pytest.raises(ValidationError, match="absent"):
        StageCorpus.model_validate(
            {**corpus.model_dump(mode="json"), "full_text_records_hash": "0" * 64}
        )


def test_duplicate_record_ids_fail_closed() -> None:
    duplicate = (_record("a"), _record("a"))
    with pytest.raises(ValidationError, match="duplicate record ids"):
        StageCorpus.build(
            raw_records=duplicate,
            deduplicated_records=duplicate,
            screened_records=duplicate,
        )


def test_from_files_normalizes_records_and_round_trips(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    dedup_path = tmp_path / "dedup.json"
    screened_path = tmp_path / "screened.json"
    raw = [_record("a").model_dump(mode="json"), _record("b").model_dump(mode="json")]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    dedup_path.write_text(json.dumps(raw[:1]), encoding="utf-8")
    screened_path.write_text(json.dumps(raw[:1]), encoding="utf-8")

    corpus = StageCorpus.from_files(
        raw_records=raw_path,
        deduplicated_records=dedup_path,
        screened_records=screened_path,
    )
    reloaded = StageCorpus.model_validate(corpus.model_dump(mode="json"))

    assert reloaded == corpus
