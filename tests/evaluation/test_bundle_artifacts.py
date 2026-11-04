from __future__ import annotations

from pathlib import Path

import pytest

from slr_pipeline.evaluation.berg2023_gates import evaluate_berg2023_gates
from slr_pipeline.evaluation.bundle_artifacts import (
    ReproductionArtifactPaths,
    verify_reproduction_artifact_set,
)
from slr_pipeline.evaluation.evaluator import evaluate_pipeline
from slr_pipeline.evaluation.gold_standard import GoldStandard
from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
from slr_pipeline.evaluation.stage_corpus import StageCorpus
from slr_pipeline.evaluation.workflow import (
    evaluate_with_adjudication_journal,
    evaluation_binding,
)
from slr_pipeline.state import Record


def _inputs_with_pending_adjudication():
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


def _write_artifacts(root: Path, spec, gold, corpus, evaluation, gate_report) -> ReproductionArtifactPaths:
    root.mkdir(parents=True, exist_ok=True)
    paths = ReproductionArtifactPaths(
        reproduction_spec=root / "reproduction_spec.json",
        gold_standard=root / "gold_standard.json",
        stage_corpus=root / "stage_corpus.json",
        pipeline_evaluation=root / "pipeline_evaluation.json",
        gate_report=root / "gate_report.json",
        adjudication_journal=root / "evaluation_adjudications.jsonl",
    )
    paths.reproduction_spec.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    paths.gold_standard.write_text(gold.model_dump_json(indent=2), encoding="utf-8")
    paths.stage_corpus.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")
    paths.pipeline_evaluation.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    paths.gate_report.write_text(gate_report.model_dump_json(indent=2), encoding="utf-8")
    return paths


def test_verify_rejects_pending_adjudications_with_an_empty_journal(tmp_path: Path) -> None:
    spec, gold, corpus = _inputs_with_pending_adjudication()
    evaluation = evaluate_pipeline(spec, gold, corpus)
    gate_report = evaluate_berg2023_gates(evaluation, spec, gold, corpus)
    assert evaluation.pending_adjudications

    paths = _write_artifacts(tmp_path / "artifacts", spec, gold, corpus, evaluation, gate_report)
    paths.adjudication_journal.write_bytes(b"")

    with pytest.raises(ValueError, match="adjudication journal latest projection does not match"):
        verify_reproduction_artifact_set(paths)


def test_verify_accepts_pending_adjudications_journaled_via_the_real_workflow(
    tmp_path: Path,
) -> None:
    spec, gold, corpus = _inputs_with_pending_adjudication()
    journal_path = tmp_path / "evaluation_adjudications.jsonl"
    evaluation = evaluate_with_adjudication_journal(
        spec, gold, corpus, journal_path=journal_path
    )
    gate_report = evaluate_berg2023_gates(evaluation, spec, gold, corpus)
    assert evaluation.pending_adjudications

    paths = _write_artifacts(tmp_path / "artifacts", spec, gold, corpus, evaluation, gate_report)
    paths.adjudication_journal.write_bytes(journal_path.read_bytes())

    verified = verify_reproduction_artifact_set(paths)
    assert verified.pipeline_evaluation == evaluation
    assert verified.gate_report == gate_report
    assert verified.journal_binding == evaluation_binding(spec, gold, corpus)
