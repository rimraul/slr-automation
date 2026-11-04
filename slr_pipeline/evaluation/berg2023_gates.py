"""Executable reproducibility acceptance gates for the Berg2023 case study."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, model_validator

from .adjudication import AdjudicationRecord
from .adjudication_journal import MATCHER_VERSION
from .evaluator import NotEvaluable, PipelineEvaluation
from .gold_standard import GoldStandard
from .matching import GoldMatchResult, match_records_to_gold
from .reproduction_spec import ReproductionSpec
from .stage_corpus import StageCorpus


GateStatus = Literal["pass", "fail", "pending", "not_evaluable"]


class GoldPaperOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    found_at_stages: dict[str, bool]
    first_loss_stage: Union[str, NotEvaluable]


class GateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    final_recall: Union[float, NotEvaluable]
    final_recall_basis: Literal[
        "included_records", "monotonicity_inferred_zero"
    ] | None
    final_recall_lower_bound: Union[float, NotEvaluable]
    final_recall_upper_bound: Union[float, NotEvaluable]
    gold_papers_with_unknown_final_status: Union[int, NotEvaluable]
    per_paper_outcomes: Union[tuple[GoldPaperOutcome, ...], NotEvaluable]
    gold_papers_with_unknown_loss_stage: Union[int, NotEvaluable]
    loss_stage_distribution: Union[dict[str, int], NotEvaluable]
    pending_gold_paper_count: int
    gate_status: dict[str, GateStatus]
    gate_reasons: dict[str, str]
    outcomes_source: Literal["recomputed_from_stage_records"] = (
        "recomputed_from_stage_records"
    )
    spec_hash: str
    gold_standard_hash: str
    stage_corpus_hashes: dict[str, str]
    matcher_version: str
    missing_databases: tuple[str, ...]
    unsupported_databases: tuple[str, ...]
    partial_reproduction: bool

    @model_validator(mode="after")
    def _validate_internal_accounting(self) -> "GateReport":
        required_gates = {
            "gate_1_search",
            "gate_2_deduplication",
            "gate_3_title_abstract",
            "gate_4_full_text",
            "gate_5_post_cutoff",
        }
        if set(self.gate_status) != required_gates:
            raise ValueError("gate_status must contain all five Berg2023 gates")
        if set(self.gate_reasons) != required_gates:
            raise ValueError("gate_reasons must contain all five Berg2023 gates")

        if isinstance(self.per_paper_outcomes, NotEvaluable):
            reason = self.per_paper_outcomes.reason
            linked = (
                self.final_recall_lower_bound,
                self.final_recall_upper_bound,
                self.gold_papers_with_unknown_final_status,
                self.gold_papers_with_unknown_loss_stage,
                self.loss_stage_distribution,
            )
            if any(
                not isinstance(value, NotEvaluable) or value.reason != reason
                for value in linked
            ):
                raise ValueError(
                    "whole-set per-paper identity gaps must propagate one identical "
                    "NotEvaluable value"
                )
            return self

        if not isinstance(self.loss_stage_distribution, dict):
            raise ValueError("loss_stage_distribution must be a dict for per-paper outcomes")
        if sum(self.loss_stage_distribution.values()) != len(self.per_paper_outcomes):
            raise ValueError("loss-stage distribution must account for every gold paper")
        if not isinstance(self.gold_papers_with_unknown_loss_stage, int):
            raise ValueError("unknown loss-stage count must be concrete here")
        if (
            self.loss_stage_distribution.get("not_evaluable", 0)
            != self.gold_papers_with_unknown_loss_stage
        ):
            raise ValueError("not_evaluable loss bucket does not match unknown count")
        if not isinstance(self.gold_papers_with_unknown_final_status, int):
            raise ValueError("unknown final-status count must be concrete here")
        if not isinstance(self.final_recall_lower_bound, float) or not isinstance(
            self.final_recall_upper_bound, float
        ):
            raise ValueError("recall bounds must be concrete for structured gold papers")
        if not 0.0 <= self.final_recall_lower_bound <= self.final_recall_upper_bound <= 1.0:
            raise ValueError("recall bounds are invalid")
        return self


def _stage_inputs(stage_corpus: StageCorpus):
    stages = [
        ("search", stage_corpus.raw_records),
        ("dedup", stage_corpus.deduplicated_records),
        ("screening", stage_corpus.screened_records),
    ]
    if stage_corpus.full_text_records is not None:
        stages.append(("full_text", stage_corpus.full_text_records))
    if stage_corpus.included_records is not None:
        stages.append(("included", stage_corpus.included_records))
    return stages


def _hashes(stage_corpus: StageCorpus) -> dict[str, str]:
    values = {
        "raw_records": stage_corpus.raw_records_hash,
        "deduplicated_records": stage_corpus.deduplicated_records_hash,
        "screened_records": stage_corpus.screened_records_hash,
    }
    if stage_corpus.full_text_records_hash is not None:
        values["full_text_records"] = stage_corpus.full_text_records_hash
    if stage_corpus.included_records_hash is not None:
        values["included_records"] = stage_corpus.included_records_hash
    return values


def _verify_bindings(
    evaluation: PipelineEvaluation,
    spec: ReproductionSpec,
    gold_standard: GoldStandard,
    stage_corpus: StageCorpus,
) -> None:
    gold_hash = gold_standard.content_hash()
    if spec.review_id != gold_standard.review_id or evaluation.review_id != spec.review_id:
        raise ValueError("review_id mismatch among evaluation inputs")
    if spec.expected_gold_standard_hash != gold_hash:
        raise ValueError("spec expected_gold_standard_hash does not match gold standard")
    if evaluation.spec_hash != spec.spec_hash():
        raise ValueError("evaluation spec_hash does not match reproduction spec")
    if evaluation.gold_standard_hash != gold_hash:
        raise ValueError("evaluation gold_standard_hash does not match gold standard")
    evaluation_hashes = {
        "raw_records": evaluation.raw_records_hash,
        "deduplicated_records": evaluation.deduplicated_records_hash,
        "screened_records": evaluation.screened_records_hash,
    }
    if evaluation.full_text_records_hash is not None:
        evaluation_hashes["full_text_records"] = evaluation.full_text_records_hash
    if evaluation.included_records_hash is not None:
        evaluation_hashes["included_records"] = evaluation.included_records_hash
    if evaluation_hashes != _hashes(stage_corpus):
        raise ValueError("evaluation stage-corpus hashes do not match supplied corpus")


def _stage_matches(
    evaluation: PipelineEvaluation,
    spec: ReproductionSpec,
    gold_standard: GoldStandard,
    stage_corpus: StageCorpus,
) -> dict[str, GoldMatchResult]:
    projection = {
        record.adjudication_id: record for record in evaluation.adjudications
    }
    results = {
        name: match_records_to_gold(
            records,
            gold_standard,
            run_id=spec.spec_hash(),
            adjudication_projection=projection,
        )
        for name, records in _stage_inputs(stage_corpus)
    }
    return results


def _status(
    *,
    pending: set[str],
    found: set[str] | None,
    all_gold: set[str],
    unavailable_reason: str | None = None,
) -> tuple[GateStatus, str]:
    if unavailable_reason is not None:
        return "not_evaluable", unavailable_reason
    if pending:
        return "pending", f"{len(pending)} gold paper(s) await human adjudication"
    missing = all_gold - (found or set())
    if missing:
        return "fail", f"{len(missing)} of {len(all_gold)} gold papers are absent"
    return "pass", f"all {len(all_gold)} gold papers are present"


def evaluate_berg2023_gates(
    evaluation: PipelineEvaluation,
    spec: ReproductionSpec,
    gold_standard: GoldStandard,
    stage_corpus: StageCorpus,
) -> GateReport:
    """Recompute and evaluate the five Berg2023 reproducibility gates."""

    _verify_bindings(evaluation, spec, gold_standard, stage_corpus)
    coverage = spec.coverage()
    stage_hashes = _hashes(stage_corpus)

    if gold_standard.structurally_ambiguous:
        identity_gap = NotEvaluable(
            reason=(
                "gold standard loaded from legacy doi/title-only format; pairing "
                "between DOI and title entries is unknown, so per-paper identity "
                "cannot be established"
            )
        )
        statuses = {
            key: "not_evaluable"
            for key in (
                "gate_1_search",
                "gate_2_deduplication",
                "gate_3_title_abstract",
                "gate_4_full_text",
                "gate_5_post_cutoff",
            )
        }
        reasons = {key: identity_gap.reason for key in statuses}
        return GateReport(
            review_id=spec.review_id,
            final_recall=identity_gap,
            final_recall_basis=None,
            final_recall_lower_bound=identity_gap,
            final_recall_upper_bound=identity_gap,
            gold_papers_with_unknown_final_status=identity_gap,
            per_paper_outcomes=identity_gap,
            gold_papers_with_unknown_loss_stage=identity_gap,
            loss_stage_distribution=identity_gap,
            pending_gold_paper_count=0,
            gate_status=statuses,
            gate_reasons=reasons,
            spec_hash=spec.spec_hash(),
            gold_standard_hash=gold_standard.content_hash(),
            stage_corpus_hashes=stage_hashes,
            matcher_version=MATCHER_VERSION,
            missing_databases=coverage.missing_databases,
            unsupported_databases=coverage.unsupported_databases,
            partial_reproduction=coverage.partial_reproduction,
        )

    results = _stage_matches(evaluation, spec, gold_standard, stage_corpus)
    stage_records_by_name = dict(_stage_inputs(stage_corpus))
    stage_order = list(stage_records_by_name)
    found_by_stage = {
        name: {pair.gold_paper_id for pair in result.matched_pairs}
        for name, result in results.items()
    }
    pending_by_stage = {
        name: {
            paper_id
            for item in result.adjudications
            for paper_id in item.candidate_gold_paper_ids
        }
        for name, result in results.items()
    }
    pending_gold = set().union(*pending_by_stage.values()) if pending_by_stage else set()
    all_gold = {paper.paper_id for paper in gold_standard.included_papers}
    latest_stage = stage_order[-1]
    latest_found = found_by_stage[latest_stage]

    included_present = stage_corpus.included_records is not None
    monotonic_zero = not included_present and not latest_found and not pending_gold
    unknown_final = (
        set()
        if included_present or monotonic_zero
        else set(latest_found)
    )
    unknown_loss = pending_gold | unknown_final

    outcomes: list[GoldPaperOutcome] = []
    for paper in sorted(gold_standard.included_papers, key=lambda item: item.paper_id):
        found_at = {name: paper.paper_id in found_by_stage[name] for name in stage_order}
        if paper.paper_id in unknown_loss:
            first_loss: Union[str, NotEvaluable] = NotEvaluable(
                reason=(
                    "paper's match or final inclusion status is undetermined by "
                    "the available stage evidence"
                )
            )
        else:
            first_absent = next(
                (name for name in stage_order if not found_at[name]),
                None,
            )
            first_loss = first_absent or "retained"
        outcomes.append(
            GoldPaperOutcome(
                paper_id=paper.paper_id,
                found_at_stages=found_at,
                first_loss_stage=first_loss,
            )
        )

    distribution = Counter(
        "not_evaluable"
        if isinstance(outcome.first_loss_stage, NotEvaluable)
        else outcome.first_loss_stage
        for outcome in outcomes
    )
    if distribution.get("not_evaluable") == 0:
        distribution.pop("not_evaluable", None)

    denominator = len(all_gold)
    if included_present:
        definite_final_found = set(found_by_stage["included"])
    else:
        definite_final_found = set()
    uncertainty_union = unknown_loss
    lower = (
        len(definite_final_found - uncertainty_union) / denominator
        if denominator
        else 0.0
    )
    upper = (
        len(definite_final_found | uncertainty_union) / denominator
        if denominator
        else 0.0
    )

    if pending_gold:
        final_recall: Union[float, NotEvaluable] = NotEvaluable(
            reason=f"{len(pending_gold)} gold paper(s) pending adjudication"
        )
        final_basis = None
    elif included_present:
        final_recall = len(definite_final_found) / denominator if denominator else 0.0
        final_basis = "included_records"
    elif monotonic_zero:
        final_recall = 0.0
        final_basis = "monotonicity_inferred_zero"
    else:
        final_recall = NotEvaluable(
            reason="included-record stage is absent and at least one gold paper survives"
        )
        final_basis = None

    gate_status: dict[str, GateStatus] = {}
    gate_reasons: dict[str, str] = {}
    status, reason = _status(
        pending=pending_by_stage["search"],
        found=found_by_stage["search"],
        all_gold=all_gold,
    )
    if status in {"pass", "fail"} and (
        spec.source_mode != "archived_exports" or coverage.partial_reproduction
    ):
        status = "fail"
        reason = (
            "gate requires a complete archived-export replay; this attempt is "
            f"source_mode={spec.source_mode}, partial={coverage.partial_reproduction}"
        )
    gate_status["gate_1_search"] = status
    gate_reasons["gate_1_search"] = reason

    for key, stage_name in (
        ("gate_2_deduplication", "dedup"),
        ("gate_3_title_abstract", "screening"),
    ):
        status, reason = _status(
            pending=pending_by_stage[stage_name],
            found=found_by_stage[stage_name],
            all_gold=all_gold,
        )
        gate_status[key] = status
        gate_reasons[key] = reason

    status, reason = _status(
        pending=pending_by_stage.get("full_text", set()),
        found=found_by_stage.get("full_text"),
        all_gold=all_gold,
        unavailable_reason=(
            "full-text stage records were not captured"
            if stage_corpus.full_text_records is None
            else None
        ),
    )
    gate_status["gate_4_full_text"] = status
    gate_reasons["gate_4_full_text"] = reason

    if not included_present:
        gate_status["gate_5_post_cutoff"] = "not_evaluable"
        gate_reasons["gate_5_post_cutoff"] = (
            "included-record stage was not captured; a reproduced inclusion "
            "cannot be determined from an earlier, non-final stage"
        )
    else:
        final_stage_result = results["included"]
        caind_ids = {
            record.id
            for record in stage_records_by_name["included"]
            if record.year is not None
            and record.year >= 2025
            and "caind" in record.title.casefold()
        }
        caind_confirmed = any(
            pair.pipeline_record_id in caind_ids for pair in final_stage_result.matched_pairs
        )
        caind_pending = any(
            record_id in caind_ids
            for adjudication in final_stage_result.adjudications
            for record_id in adjudication.candidate_pipeline_record_ids
        )
        if caind_confirmed:
            gate_status["gate_5_post_cutoff"] = "fail"
            gate_reasons["gate_5_post_cutoff"] = (
                "a 2025-or-later CAIND record was presented as a reproduced "
                "inclusion at the included stage"
            )
        elif caind_pending:
            gate_status["gate_5_post_cutoff"] = "pending"
            gate_reasons["gate_5_post_cutoff"] = (
                "a 2025-or-later CAIND record has a pending adjudication that "
                "could present it as a reproduced inclusion at the included stage"
            )
        else:
            gate_status["gate_5_post_cutoff"] = "pass"
            gate_reasons["gate_5_post_cutoff"] = (
                "no 2025-or-later CAIND record is presented as a reproduced "
                "inclusion at the included stage"
            )

    return GateReport(
        review_id=spec.review_id,
        final_recall=final_recall,
        final_recall_basis=final_basis,
        final_recall_lower_bound=lower,
        final_recall_upper_bound=upper,
        gold_papers_with_unknown_final_status=len(unknown_final),
        per_paper_outcomes=tuple(outcomes),
        gold_papers_with_unknown_loss_stage=len(unknown_loss),
        loss_stage_distribution=dict(distribution),
        pending_gold_paper_count=len(pending_gold),
        gate_status=gate_status,
        gate_reasons=gate_reasons,
        spec_hash=spec.spec_hash(),
        gold_standard_hash=gold_standard.content_hash(),
        stage_corpus_hashes=stage_hashes,
        matcher_version=MATCHER_VERSION,
        missing_databases=coverage.missing_databases,
        unsupported_databases=coverage.unsupported_databases,
        partial_reproduction=coverage.partial_reproduction,
    )


__all__ = ["GateReport", "GoldPaperOutcome", "evaluate_berg2023_gates"]
