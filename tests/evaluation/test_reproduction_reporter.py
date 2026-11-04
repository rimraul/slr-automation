from __future__ import annotations

from pathlib import Path

import pytest

from slr_pipeline.evaluation.berg2023_gates import evaluate_berg2023_gates
from slr_pipeline.evaluation.evaluator import evaluate_pipeline
from slr_pipeline.evaluation.gold_standard import GoldStandard
from slr_pipeline.evaluation.reporter import generate_html_report
from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
from slr_pipeline.evaluation.stage_corpus import StageCorpus
from slr_pipeline.state import Record


def test_report_renders_not_evaluable_and_post_cutoff_separately(
    tmp_path: Path,
) -> None:
    gold = GoldStandard(
        review_id="review",
        review_citation="Source review",
        included_papers=[{"title": "Systematic review rural health", "year": 2020}],
    )
    spec = ReproductionSpec(
        reproduction_mode="search_reproduction",
        source_mode="live_search",
        citation="Source review",
        original_databases=["PubMed", "Scopus"],
        reproduced_databases=["PubMed"],
        exact_queries={},
        search_dates={"PubMed": "2023-06-15"},
        filters=[],
        manual_additions=[],
        reported_prisma_counts={},
        unknowns=["Original PRISMA counts unavailable."],
        review_id="review",
        review_type="scoping review",
        title_abstract_eligibility_criteria="criterion",
        full_text_eligibility_criteria="criterion",
        gold_standard_ref="gold.json",
        expected_gold_standard_hash=gold.content_hash(),
    )
    fuzzy = Record(
        id="fuzzy",
        source="pubmed",
        source_id="fuzzy",
        title="Systematic review of rural health",
        year=2020,
    )
    recent = Record(
        id="recent",
        source="pubmed",
        source_id="recent",
        title="New post-cutoff study",
        year=2024,
    )
    corpus = StageCorpus.build(
        raw_records=(fuzzy, recent),
        deduplicated_records=(fuzzy, recent),
        screened_records=(fuzzy, recent),
    )
    evaluation = evaluate_pipeline(spec, gold, corpus)
    output = tmp_path / "report.html"

    gate_report = evaluate_berg2023_gates(evaluation, spec, gold, corpus)
    generate_html_report(evaluation, gold, output, gate_report=gate_report)
    rendered = output.read_text(encoding="utf-8")

    assert "Not evaluable:" in rendered
    assert "Pending human adjudications" in rendered
    assert "Executable acceptance gates" in rendered
    assert "First-loss distribution" in rendered
    assert evaluation.pending_adjudications[0].adjudication_id in rendered
    assert "New post-cutoff study" in rendered
    assert "not counted as reproduced inclusions" in rendered
    assert evaluation.spec_hash in rendered
    assert evaluation.gold_standard_hash in rendered


def test_report_rejects_different_gold_content(tmp_path: Path) -> None:
    gold = GoldStandard(review_id="review", included_papers=[{"title": "Gold"}])
    spec = ReproductionSpec(
        reproduction_mode="search_reproduction",
        source_mode="archived_exports",
        citation="Review",
        original_databases=[],
        reproduced_databases=[],
        exact_queries={},
        search_dates={},
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
    record = Record(id="r", source="x", source_id="r", title="Gold")
    corpus = StageCorpus.build(
        raw_records=(record,),
        deduplicated_records=(record,),
        screened_records=(record,),
    )
    evaluation = evaluate_pipeline(spec, gold, corpus)
    changed = GoldStandard(
        review_id="review",
        included_papers=[{"title": "Different"}],
    )

    with pytest.raises(ValueError, match="gold_standard_hash"):
        generate_html_report(evaluation, changed, tmp_path / "report.html")


def _gold_spec_evaluation_gate() -> tuple:
    gold = GoldStandard(review_id="review", included_papers=[{"title": "Gold"}])
    spec = ReproductionSpec(
        reproduction_mode="search_reproduction",
        source_mode="archived_exports",
        citation="Review",
        original_databases=[],
        reproduced_databases=[],
        exact_queries={},
        search_dates={},
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
    record = Record(id="r", source="x", source_id="r", title="Gold")
    corpus = StageCorpus.build(
        raw_records=(record,),
        deduplicated_records=(record,),
        screened_records=(record,),
    )
    evaluation = evaluate_pipeline(spec, gold, corpus)
    gate_report = evaluate_berg2023_gates(evaluation, spec, gold, corpus)
    return gold, evaluation, gate_report


def test_report_rejects_gate_report_from_a_different_review_id(tmp_path: Path) -> None:
    gold, evaluation, gate_report = _gold_spec_evaluation_gate()
    mismatched = gate_report.model_copy(update={"review_id": "other-review"})

    with pytest.raises(ValueError, match="review_id"):
        generate_html_report(evaluation, gold, tmp_path / "report.html", gate_report=mismatched)


def test_report_rejects_gate_report_from_a_different_stage_corpus(tmp_path: Path) -> None:
    gold, evaluation, gate_report = _gold_spec_evaluation_gate()
    mismatched = gate_report.model_copy(
        update={"stage_corpus_hashes": {**gate_report.stage_corpus_hashes, "raw_records": "0" * 64}}
    )

    with pytest.raises(ValueError, match="stage-corpus hashes"):
        generate_html_report(evaluation, gold, tmp_path / "report.html", gate_report=mismatched)


def test_report_rejects_gate_report_from_a_different_matcher_version(tmp_path: Path) -> None:
    gold, evaluation, gate_report = _gold_spec_evaluation_gate()
    mismatched = gate_report.model_copy(update={"matcher_version": "stale-version"})

    with pytest.raises(ValueError, match="matcher_version"):
        generate_html_report(evaluation, gold, tmp_path / "report.html", gate_report=mismatched)
