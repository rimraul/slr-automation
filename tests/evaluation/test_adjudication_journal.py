from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from slr_pipeline.evaluation.adjudication import AdjudicationRecord
from slr_pipeline.evaluation.adjudication_journal import (
    EvaluationAdjudicationStore,
    EvaluationBinding,
    verify_adjudication_journal_bytes,
)
from slr_pipeline.evaluation.gold_standard import GoldStandard
from slr_pipeline.evaluation.matching import match_records_to_gold


def _pending(run_id: str = "run-1", suffix: str = "") -> AdjudicationRecord:
    gold = GoldStandard(
        review_id="gold",
        included_papers=[{"title": f"Systematic review rural health{suffix}"}],
    )
    result = match_records_to_gold(
        [
            {
                "id": f"record{suffix}",
                "source": "pubmed",
                "source_id": f"source{suffix}",
                "title": f"Systematic review of rural health{suffix}",
            }
        ],
        gold,
        run_id=run_id,
    )
    return result.adjudications[0]


def _binding(**changes: str) -> EvaluationBinding:
    values = {
        "spec_hash": "1" * 64,
        "gold_standard_hash": "2" * 64,
        "raw_records_hash": "3" * 64,
        "deduplicated_records_hash": "4" * 64,
        "screened_records_hash": "5" * 64,
    }
    values.update(changes)
    return EvaluationBinding(**values)


def _resolve(pending: AdjudicationRecord) -> AdjudicationRecord:
    return AdjudicationRecord.model_validate(
        {
            **pending.model_dump(mode="json"),
            "resolution": "matched",
            "resolved_gold_paper_id": pending.candidate_gold_paper_ids[0],
            "resolved_pipeline_record_id": pending.candidate_pipeline_record_ids[0],
            "reviewer": "reviewer",
            "note": "verified manually",
            "resolved_at": "2026-09-20T12:00:00Z",
        }
    )


def test_pending_then_resolved_round_trip_and_projection(tmp_path: Path) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    pending = _pending()
    first = store.append(pending, binding=_binding())
    resolved = _resolve(pending)
    second = store.append(resolved, binding=_binding())

    assert first.sequence == 1
    assert first.previous_entry_hash is None
    assert second.sequence == 2
    assert second.previous_entry_hash == first.entry_hash
    assert store.read() == [first, second]
    assert store.latest_state_projection()[pending.adjudication_id] == resolved


def test_entry_tampering_breaks_self_hash(tmp_path: Path) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    store.append(_pending(), binding=_binding())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["scores"][0]["score"] = 0.999
    tampered = json.dumps(payload, separators=(",", ":")).encode() + b"\n"

    with pytest.raises(ValueError, match="invalid adjudication journal line"):
        verify_adjudication_journal_bytes(tampered)


def test_published_verification_rejects_torn_tail_without_mutating_it(
    tmp_path: Path,
) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    store.append(_pending(), binding=_binding())
    with store.path.open("ab") as handle:
        handle.write(b'{"torn":')
    before = store.path.read_bytes()

    with pytest.raises(ValueError, match="torn final line"):
        verify_adjudication_journal_bytes(before)

    assert store.path.read_bytes() == before


def test_live_repair_truncates_only_torn_tail_then_allows_append(
    tmp_path: Path,
) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    pending = _pending()
    first = store.append(pending, binding=_binding())
    with store.path.open("ab") as handle:
        handle.write(b'{"torn":')

    repaired = store.repair_torn_tail()
    second = store.append(_resolve(pending), binding=_binding())

    assert repaired == [first]
    assert second.previous_entry_hash == first.entry_hash
    assert len(store.read()) == 2


def test_binding_cannot_change_within_one_journal(tmp_path: Path) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    pending = _pending()
    store.append(pending, binding=_binding())
    before = store.path.read_bytes()

    with pytest.raises(ValueError, match="binding changed"):
        store.append(
            _resolve(pending),
            binding=_binding(spec_hash="9" * 64),
        )

    assert store.path.read_bytes() == before


def test_resolution_cannot_be_first_or_repeat(tmp_path: Path) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    pending = _pending()
    resolved = _resolve(pending)

    with pytest.raises(ValueError, match="must begin with a pending"):
        store.append(resolved, binding=_binding())

    store.append(pending, binding=_binding())
    store.append(resolved, binding=_binding())
    with pytest.raises(ValueError, match="multiple resolutions"):
        store.append(resolved, binding=_binding())


def test_concurrent_appends_are_serialized_without_lost_updates(tmp_path: Path) -> None:
    store = EvaluationAdjudicationStore(tmp_path / "evaluation_adjudications.jsonl")
    pending_a = _pending(suffix=" A")
    pending_b = _pending(suffix=" B")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.append, pending, binding=_binding())
            for pending in (pending_a, pending_b)
        ]
        for future in futures:
            future.result()

    entries = store.read()
    assert len(entries) == 2
    assert {entry.adjudication_id for entry in entries} == {
        pending_a.adjudication_id,
        pending_b.adjudication_id,
    }
    assert entries[1].previous_entry_hash == entries[0].entry_hash
