"""Application boundary for evaluation, adjudication persistence, and reruns."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from .adjudication import AdjudicationRecord
from .adjudication_journal import (
    EvaluationAdjudicationStore,
    EvaluationBinding,
    latest_state_projection,
)
from .evaluator import PipelineEvaluation, evaluate_pipeline
from .gold_standard import GoldStandard
from .reproduction_spec import ReproductionSpec
from .stage_corpus import StageCorpus


def evaluation_binding(
    spec: ReproductionSpec,
    gold_standard: GoldStandard,
    stage_corpus: StageCorpus,
) -> EvaluationBinding:
    return EvaluationBinding(
        spec_hash=spec.spec_hash(),
        gold_standard_hash=gold_standard.content_hash(),
        raw_records_hash=stage_corpus.raw_records_hash,
        deduplicated_records_hash=stage_corpus.deduplicated_records_hash,
        screened_records_hash=stage_corpus.screened_records_hash,
        full_text_records_hash=stage_corpus.full_text_records_hash,
        included_records_hash=stage_corpus.included_records_hash,
    )


def _read_bound_entries(
    store: EvaluationAdjudicationStore, binding: EvaluationBinding
):
    store.repair_torn_tail()
    entries = store.read()
    if entries and entries[0].binding != binding:
        raise ValueError(
            "evaluation adjudication journal is bound to different evaluation inputs"
        )
    return entries


def evaluate_with_adjudication_journal(
    spec: ReproductionSpec,
    gold_standard: GoldStandard,
    stage_corpus: StageCorpus,
    *,
    journal_path: Path,
) -> PipelineEvaluation:
    """Evaluate, honoring resolved decisions and durably recording new pending ones."""

    binding = evaluation_binding(spec, gold_standard, stage_corpus)
    store = EvaluationAdjudicationStore(journal_path)
    entries = _read_bound_entries(store, binding)
    projection = latest_state_projection(entries)
    evaluation = evaluate_pipeline(
        spec,
        gold_standard,
        stage_corpus,
        adjudication_projection=projection,
    )
    for pending in evaluation.pending_adjudications:
        if pending.adjudication_id not in projection:
            store.append_pending_if_absent(pending, binding=binding)
            projection[pending.adjudication_id] = pending
    return evaluation


def resolve_adjudication(
    *,
    journal_path: Path,
    binding: EvaluationBinding,
    adjudication_id: str,
    resolution: Literal["matched", "no_match"],
    reviewer: str,
    note: str,
    resolved_at: str,
    resolved_gold_paper_id: Optional[str] = None,
    resolved_pipeline_record_id: Optional[str] = None,
) -> AdjudicationRecord:
    """Append one human resolution without changing the pending snapshot."""

    store = EvaluationAdjudicationStore(journal_path)
    entries = _read_bound_entries(store, binding)
    projection = latest_state_projection(entries)
    current = projection.get(adjudication_id)
    if current is None:
        raise ValueError(f"unknown adjudication_id {adjudication_id!r}")
    if current.resolution != "pending":
        desired = (
            current.resolution == resolution
            and current.resolved_gold_paper_id == resolved_gold_paper_id
            and current.resolved_pipeline_record_id == resolved_pipeline_record_id
            and current.reviewer == reviewer
            and current.note == note
            and current.resolved_at == resolved_at
        )
        if desired:
            return current
        raise ValueError("adjudication has already been resolved differently")

    resolved = AdjudicationRecord.model_validate(
        {
            **current.model_dump(mode="json"),
            "resolution": resolution,
            "resolved_gold_paper_id": resolved_gold_paper_id,
            "resolved_pipeline_record_id": resolved_pipeline_record_id,
            "reviewer": reviewer,
            "note": note,
            "resolved_at": resolved_at,
        }
    )
    store.append(resolved, binding=binding)
    return resolved


__all__ = [
    "evaluate_with_adjudication_journal",
    "evaluation_binding",
    "resolve_adjudication",
]
