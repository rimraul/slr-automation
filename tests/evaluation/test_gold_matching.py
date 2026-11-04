from __future__ import annotations

import pytest
from pydantic import ValidationError

from slr_pipeline.evaluation.adjudication import AdjudicationRecord
from slr_pipeline.evaluation.gold_standard import GoldStandard
from slr_pipeline.evaluation.matching import match_records_to_gold


def _record(
    record_id: str,
    *,
    title: str,
    doi: str | None = None,
    source: str = "pubmed",
    source_id: str | None = None,
    year: int | None = None,
    authors: list[str] | None = None,
) -> dict:
    return {
        "id": record_id,
        "source": source,
        "source_id": source_id or record_id,
        "doi": doi,
        "title": title,
        "year": year,
        "authors": authors or [],
    }


def _gold(*papers: dict) -> GoldStandard:
    return GoldStandard(review_id="gold", included_papers=list(papers))


def _resolve(
    pending: AdjudicationRecord,
    *,
    resolution: str,
    gold_id: str | None = None,
    record_id: str | None = None,
) -> AdjudicationRecord:
    return AdjudicationRecord.model_validate(
        {
            **pending.model_dump(mode="json"),
            "resolution": resolution,
            "resolved_gold_paper_id": gold_id,
            "resolved_pipeline_record_id": record_id,
            "reviewer": "reviewer",
            "note": "checked source records",
            "resolved_at": "2026-09-20T12:00:00Z",
        }
    )


def test_doi_and_title_match_consumes_each_side_once() -> None:
    gold = _gold({"doi": "10.1000/one", "title": "Same paper"})
    result = match_records_to_gold(
        [_record("r1", doi="https://doi.org/10.1000/ONE", title="Same paper")],
        gold,
        run_id="run-1",
    )

    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0].match_method == "doi_exact"
    assert result.unmatched_gold_paper_ids == ()
    assert result.adjudications == ()


def test_other_persistent_id_matches_source_and_id_exactly() -> None:
    gold = _gold(
        {
            "title": "Persistent identity",
            "other_persistent_ids": {"Pub Med": "34567890"},
        }
    )
    result = match_records_to_gold(
        [
            _record(
                "r1",
                title="Different title",
                source="pubmed",
                source_id="34567890",
            )
        ],
        gold,
        run_id="run-1",
    )

    assert result.matched_pairs[0].match_method == "other_persistent_id_exact"


def test_unique_maximum_exact_assignment_is_not_greedy() -> None:
    gold = _gold(
        {
            "title": "Gold one",
            "other_persistent_ids": {"pubmed": "x", "scopus": "y"},
        },
        {
            "title": "Gold two",
            "other_persistent_ids": {"scopus": "y"},
        },
    )
    records = [
        _record("r2", title="r2", source="scopus", source_id="y"),
        _record("r1", title="r1", source="pubmed", source_id="x"),
    ]

    result = match_records_to_gold(records, gold, run_id="run-1")
    pairs = {(pair.gold_paper_id, pair.pipeline_record_id) for pair in result.matched_pairs}

    assert pairs == {
        (gold.included_papers[0].paper_id, "r1"),
        (gold.included_papers[1].paper_id, "r2"),
    }
    assert result.adjudications == ()


def test_equally_maximal_exact_collision_requires_adjudication() -> None:
    gold = _gold(
        {"title": "Identical", "year": 2020},
        {"title": "Identical", "year": 2021},
    )
    records = [
        _record("r2", title="Identical"),
        _record("r1", title="Identical"),
    ]

    forward = match_records_to_gold(records, gold, run_id="run-1")
    reverse = match_records_to_gold(list(reversed(records)), gold, run_id="run-1")

    assert forward.matched_pairs == ()
    assert len(forward.adjudications) == 1
    assert forward.adjudications[0].tier == "exact_title_collision"
    assert forward.adjudications[0].adjudication_id == reverse.adjudications[0].adjudication_id
    assert len(forward.adjudications[0].scores) == 4


def test_every_fuzzy_pair_requires_human_adjudication_even_when_corroborated() -> None:
    gold = _gold(
        {
            "title": "A simulation modelling toolkit for dialysis services",
            "authors": "Allen M et al.",
            "year": 2020,
        }
    )
    record = _record(
        "r1",
        title="A simulation modeling toolkit for dialysis services",
        authors=["Allen M"],
        year=2020,
    )

    result = match_records_to_gold([record], gold, run_id="run-1")

    assert result.matched_pairs == ()
    assert len(result.adjudications) == 1
    adjudication = result.adjudications[0]
    assert adjudication.tier == "fuzzy_title"
    assert adjudication.scores[0].score >= 0.95
    assert adjudication.scores[0].corroborating_fields == (
        "year",
        "first_author_surname",
    )


def test_below_fuzzy_floor_is_not_a_candidate() -> None:
    gold = _gold({"title": "Completely unrelated title"})
    result = match_records_to_gold(
        [_record("r1", title="Another subject entirely")],
        gold,
        run_id="run-1",
    )

    assert result.matched_pairs == ()
    assert result.adjudications == ()


def test_resolved_fuzzy_match_is_forced_on_rerun() -> None:
    gold = _gold({"title": "Systematic review rural health", "year": 2020})
    record = _record("r1", title="Systematic review of rural health", year=2020)
    initial = match_records_to_gold([record], gold, run_id="run-1")
    pending = initial.adjudications[0]
    resolved = _resolve(
        pending,
        resolution="matched",
        gold_id=pending.candidate_gold_paper_ids[0],
        record_id="r1",
    )

    rerun = match_records_to_gold(
        [record],
        gold,
        run_id="run-1",
        adjudication_projection={resolved.adjudication_id: resolved},
    )

    assert rerun.adjudications == ()
    assert rerun.matched_pairs[0].match_method == "adjudicated"


def test_resolved_no_match_suppresses_same_fuzzy_ambiguity() -> None:
    gold = _gold({"title": "Systematic review rural health", "year": 2020})
    record = _record("r1", title="Systematic review of rural health", year=2020)
    pending = match_records_to_gold([record], gold, run_id="run-1").adjudications[0]
    resolved = _resolve(pending, resolution="no_match")

    rerun = match_records_to_gold(
        [record],
        gold,
        run_id="run-1",
        adjudication_projection={resolved.adjudication_id: resolved},
    )

    assert rerun.adjudications == ()
    assert rerun.matched_pairs == ()


def test_duplicate_manifestation_is_separate_from_gold_recall_match() -> None:
    gold = _gold({"doi": "10.1000/one", "title": "Same paper", "year": 2020})
    result = match_records_to_gold(
        [
            _record("doi-record", doi="10.1000/one", title="Different title"),
            _record("title-record", title="Same paper", year=2020),
        ],
        gold,
        run_id="run-1",
    )

    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0].pipeline_record_id == "doi-record"
    assert result.duplicate_manifestations[0].pipeline_record_id == "title-record"
    assert (
        result.duplicate_manifestations[0].duplicate_manifestation_of
        == gold.included_papers[0].paper_id
    )


def test_record_ids_are_required_and_unique() -> None:
    gold = _gold({"title": "Paper"})
    with pytest.raises(ValueError, match="stable ids"):
        match_records_to_gold(
            [{"source": "x", "source_id": "1", "title": "Paper"}],
            gold,
            run_id="run-1",
        )
    with pytest.raises(ValueError, match="duplicate pipeline record id"):
        match_records_to_gold(
            [_record("same", title="a"), _record("same", title="b")],
            gold,
            run_id="run-1",
        )


def test_structurally_ambiguous_legacy_gold_fails_closed() -> None:
    gold = GoldStandard(review_id="legacy", included_titles=["Paper"])
    with pytest.raises(ValueError, match="structurally ambiguous"):
        match_records_to_gold([_record("r1", title="Paper")], gold, run_id="run-1")


def test_adjudication_schema_rejects_fabricated_identity_and_invalid_resolution() -> None:
    gold = _gold({"title": "Systematic review rural health"})
    pending = match_records_to_gold(
        [_record("r1", title="Systematic review of rural health")],
        gold,
        run_id="run-1",
    ).adjudications[0]
    payload = pending.model_dump(mode="json")

    with pytest.raises(ValidationError, match="adjudication_id"):
        AdjudicationRecord.model_validate({**payload, "adjudication_id": "fabricated"})
    with pytest.raises(ValidationError, match="requires reviewer"):
        AdjudicationRecord.model_validate({**payload, "resolution": "no_match"})
