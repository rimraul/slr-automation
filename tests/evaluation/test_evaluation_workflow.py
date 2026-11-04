from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from slr_pipeline.evaluation.adjudication_journal import EvaluationAdjudicationStore
from slr_pipeline.evaluation.evaluator import NotEvaluable
from slr_pipeline.evaluation.gold_standard import GoldStandard
from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
from slr_pipeline.evaluation.stage_corpus import StageCorpus
from slr_pipeline.evaluation.workflow import (
    evaluate_with_adjudication_journal,
    evaluation_binding,
    resolve_adjudication,
)
from slr_pipeline.state import Record


def _inputs():
    gold = GoldStandard(
        review_id="review",
        included_papers=[{"title": "Systematic review rural health", "year": 2020}],
    )
    spec = ReproductionSpec(
        reproduction_mode="search_reproduction",
        source_mode="archived_exports",
        citation="Review",
        original_databases=["PubMed"],
        reproduced_databases=["PubMed"],
        exact_queries={},
        search_dates={"PubMed": "2023-06-01"},
        filters=[],
        manual_additions=[],
        reported_prisma_counts={},
        unknowns=[],
        review_id="review",
        review_type="scoping review",
        title_abstract_eligibility_criteria="criterion",
        full_text_eligibility_criteria="criterion",
        gold_standard_ref="gold.json",
        expected_gold_standard_hash=gold.content_hash(),
    )
    record = Record(
        id="r1",
        source="pubmed",
        source_id="r1",
        title="Systematic review of rural health",
        year=2020,
    )
    corpus = StageCorpus.build(
        raw_records=(record,),
        deduplicated_records=(record,),
        screened_records=(record,),
        included_records=(record,),
    )
    return spec, gold, corpus


def test_workflow_persists_pending_once_then_honors_resolution(tmp_path: Path) -> None:
    spec, gold, corpus = _inputs()
    journal = tmp_path / "evaluation_adjudications.jsonl"

    first = evaluate_with_adjudication_journal(
        spec, gold, corpus, journal_path=journal
    )
    second = evaluate_with_adjudication_journal(
        spec, gold, corpus, journal_path=journal
    )
    store = EvaluationAdjudicationStore(journal)

    assert isinstance(first.final_recall, NotEvaluable)
    assert first == second
    assert len(store.read()) == 1

    pending = first.pending_adjudications[0]
    resolved = resolve_adjudication(
        journal_path=journal,
        binding=evaluation_binding(spec, gold, corpus),
        adjudication_id=pending.adjudication_id,
        resolution="matched",
        resolved_gold_paper_id=pending.candidate_gold_paper_ids[0],
        resolved_pipeline_record_id=pending.candidate_pipeline_record_ids[0],
        reviewer="reviewer",
        note="verified",
        resolved_at="2026-09-20T12:00:00Z",
    )
    rerun = evaluate_with_adjudication_journal(
        spec, gold, corpus, journal_path=journal
    )

    assert resolved.resolution == "matched"
    assert rerun.pending_adjudications == ()
    assert rerun.final_recall == 1.0
    assert len(store.read()) == 2


def test_identical_resolution_is_idempotent_but_conflict_fails(tmp_path: Path) -> None:
    spec, gold, corpus = _inputs()
    journal = tmp_path / "evaluation_adjudications.jsonl"
    evaluation = evaluate_with_adjudication_journal(
        spec, gold, corpus, journal_path=journal
    )
    pending = evaluation.pending_adjudications[0]
    kwargs = dict(
        journal_path=journal,
        binding=evaluation_binding(spec, gold, corpus),
        adjudication_id=pending.adjudication_id,
        resolution="no_match",
        reviewer="reviewer",
        note="not the same paper",
        resolved_at="2026-09-20T12:00:00Z",
    )

    first = resolve_adjudication(**kwargs)
    second = resolve_adjudication(**kwargs)
    assert second == first

    with pytest.raises(ValueError, match="already been resolved differently"):
        resolve_adjudication(**{**kwargs, "note": "changed"})


def test_binding_drift_fails_before_reusing_journal(tmp_path: Path) -> None:
    spec, gold, corpus = _inputs()
    journal = tmp_path / "evaluation_adjudications.jsonl"
    evaluate_with_adjudication_journal(spec, gold, corpus, journal_path=journal)
    changed = spec.model_copy(update={"filters": ["English"]})

    with pytest.raises(ValueError, match="different evaluation inputs"):
        evaluate_with_adjudication_journal(
            changed, gold, corpus, journal_path=journal
        )


def test_concurrent_evaluations_append_same_pending_only_once(tmp_path: Path) -> None:
    spec, gold, corpus = _inputs()
    journal = tmp_path / "evaluation_adjudications.jsonl"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                evaluate_with_adjudication_journal,
                spec,
                gold,
                corpus,
                journal_path=journal,
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert len(EvaluationAdjudicationStore(journal).read()) == 1
