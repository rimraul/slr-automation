"""Stage-by-stage reproduction evaluation against a structured gold standard."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from slr_pipeline.doi import normalise_doi
from slr_pipeline.state import Record

from .adjudication import AdjudicationRecord
from .gold_standard import GoldStandard
from .matching import GoldMatchResult, match_records_to_gold
from .reproduction_spec import DatabaseCoverage, ReproductionSpec
from .stage_corpus import StageCorpus


class NotEvaluable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["not_evaluable"] = "not_evaluable"
    reason: str


MetricValue = Union[float, NotEvaluable]
CountValue = Union[int, NotEvaluable]


class PrismaCountPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original: Optional[int]
    reproduced: int
    difference: Optional[int]


class NumericValueComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str
    original_value: float
    reproduced_value: float
    absolute_difference: float


class ThematicConcordance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["theme_overlap_set", "human_rated_concordance"]
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reviewer: Optional[str] = None
    notes: str


class PaperMatch(BaseModel):
    """A pipeline record's evaluated relationship to the gold standard."""

    pipeline_id: str
    doi: Optional[str]
    title: str
    year: Optional[int]
    in_gold_standard: bool
    gold_paper_id: Optional[str] = None
    match_method: Optional[str] = None
    similarity_ratio: Optional[float] = None
    temporal_status: Literal[
        "in_scope",
        "post_cutoff",
        "date_unknown",
        "cutoff_unknown",
        "date_precision_insufficient",
    ]
    duplicate_manifestation_of: Optional[str] = None


class StageEvaluation(BaseModel):
    stage: str
    stage_key: str
    pipeline_count: int
    gold_count: int
    found_in_gold: CountValue
    recall: MetricValue
    precision: MetricValue
    sensitivity: MetricValue
    specificity: MetricValue
    missed_dois: list[str] = Field(default_factory=list)
    missed_titles: list[str] = Field(default_factory=list)
    extra_dois: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    paper_matches: list[PaperMatch] = Field(default_factory=list)
    pending_adjudication_ids: tuple[str, ...] = ()
    exclusion_reason_agreement: MetricValue
    field_level_charting_agreement: MetricValue
    evidence_locator_agreement: MetricValue
    prisma_count_comparison: dict[str, Union[PrismaCountPair, NotEvaluable]] = Field(
        default_factory=dict
    )
    numeric_value_agreement: list[Union[NumericValueComparison, NotEvaluable]] = Field(
        default_factory=list
    )
    thematic_synthesis_concordance: Optional[ThematicConcordance] = None


class PipelineEvaluation(BaseModel):
    review_id: str
    spec_hash: str
    gold_standard_hash: str
    source_mode: Literal["archived_exports", "live_search"]
    reproduction_mode: Literal[
        "search_reproduction", "screening_replay", "review_update"
    ]
    missing_databases: tuple[str, ...]
    unsupported_databases: tuple[str, ...]
    partial_reproduction: bool
    gold_n_included: int
    raw_records_hash: str
    deduplicated_records_hash: str
    screened_records_hash: str
    full_text_records_hash: Optional[str] = None
    included_records_hash: Optional[str] = None
    stages: list[StageEvaluation] = Field(default_factory=list)
    search_recall: Optional[MetricValue] = None
    final_recall: Optional[MetricValue] = None
    final_precision: Optional[MetricValue] = None
    adjudications: tuple[AdjudicationRecord, ...] = ()
    pending_adjudications: tuple[AdjudicationRecord, ...] = ()
    newly_discovered_post_cutoff: tuple[PaperMatch, ...] = ()
    summary_notes: list[str] = Field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"Reproduction evaluation: {self.review_id}")
        print(f"Gold standard: {self.gold_n_included} included papers")
        print("=" * 60)
        for stage in self.stages:
            recall = (
                f"{stage.recall:.1%}"
                if isinstance(stage.recall, float)
                else "not evaluable"
            )
            print(f"  {stage.stage:<22} n={stage.pipeline_count:>5} recall={recall}")
        for note in self.summary_notes:
            print(f"  ! {note}")


_LEGACY_IDENTITY_REASON = (
    "gold standard loaded from legacy doi/title-only format; pairing between "
    "DOI and title entries is unknown, so per-paper identity cannot be established"
)
_INCLUDED_ONLY_REASON = (
    "gold standard documents included papers only; exclusion set is unknown"
)
_STAGE_LABELS = {
    "raw_records": "1. Search",
    "deduplicated_records": "2. After dedup",
    "screened_records": "3. After screening",
    "full_text_records": "4. Full-text screening",
    "included_records": "5. Final included set",
}
_PRISMA_KEYS = {
    "raw_records": "identified",
    "deduplicated_records": "after_dedup",
    "screened_records": "screened",
    "full_text_records": "assessed_fulltext",
    "included_records": "included",
}


def classify_temporal_status(
    record: Record, spec: ReproductionSpec
) -> Literal[
    "in_scope",
    "post_cutoff",
    "date_unknown",
    "cutoff_unknown",
    "date_precision_insufficient",
]:
    if record.year is None:
        return "date_unknown"
    cutoff = spec.cutoff_for_database(record.source or None)
    if cutoff is None:
        return "cutoff_unknown"
    if record.year < cutoff.year:
        return "in_scope"
    if record.year > cutoff.year:
        return "post_cutoff"
    return "date_precision_insufficient"


def _not_evaluable(reason: str) -> NotEvaluable:
    return NotEvaluable(reason=reason)


def _prisma_comparison(
    *, stage_key: str, pipeline_count: int, spec: ReproductionSpec
) -> dict[str, Union[PrismaCountPair, NotEvaluable]]:
    prisma_key = _PRISMA_KEYS[stage_key]
    original = spec.reported_prisma_counts.get(prisma_key)
    if original is None:
        return {
            prisma_key: _not_evaluable(
                f"source review did not report the {prisma_key} PRISMA count"
            )
        }
    return {
        prisma_key: PrismaCountPair(
            original=original,
            reproduced=pipeline_count,
            difference=pipeline_count - original,
        )
    }


def _paper_matches(
    records: tuple[Record, ...], result: GoldMatchResult, spec: ReproductionSpec
) -> list[PaperMatch]:
    matched = {pair.pipeline_record_id: pair for pair in result.matched_pairs}
    duplicates = {
        item.pipeline_record_id: item for item in result.duplicate_manifestations
    }
    return [
        PaperMatch(
            pipeline_id=record.id,
            doi=record.doi,
            title=record.title,
            year=record.year,
            in_gold_standard=record.id in matched,
            gold_paper_id=(
                matched[record.id].gold_paper_id if record.id in matched else None
            ),
            match_method=(
                matched[record.id].match_method if record.id in matched else None
            ),
            similarity_ratio=(
                matched[record.id].score if record.id in matched else None
            ),
            temporal_status=classify_temporal_status(record, spec),
            duplicate_manifestation_of=(
                duplicates[record.id].duplicate_manifestation_of
                if record.id in duplicates
                else None
            ),
        )
        for record in records
    ]


def _evaluate_stage(
    *,
    stage_key: str,
    records: tuple[Record, ...],
    spec: ReproductionSpec,
    gold: GoldStandard,
    projection: dict[str, AdjudicationRecord],
) -> tuple[StageEvaluation, tuple[AdjudicationRecord, ...]]:
    pipeline_count = len(records)
    common_not_evaluable = _not_evaluable(_INCLUDED_ONLY_REASON)
    if gold.structurally_ambiguous:
        identity_gap = _not_evaluable(_LEGACY_IDENTITY_REASON)
        stage = StageEvaluation(
            stage=_STAGE_LABELS[stage_key],
            stage_key=stage_key,
            pipeline_count=pipeline_count,
            gold_count=gold.n_included,
            found_in_gold=identity_gap,
            recall=identity_gap,
            precision=identity_gap,
            sensitivity=identity_gap,
            specificity=common_not_evaluable,
            notes=[_LEGACY_IDENTITY_REASON],
            exclusion_reason_agreement=common_not_evaluable,
            field_level_charting_agreement=_not_evaluable(
                "no bound original charting results were supplied"
            ),
            evidence_locator_agreement=_not_evaluable(
                "no bound original evidence locators were supplied"
            ),
            prisma_count_comparison=_prisma_comparison(
                stage_key=stage_key, pipeline_count=pipeline_count, spec=spec
            ),
        )
        return stage, ()

    result = match_records_to_gold(
        records,
        gold,
        run_id=spec.spec_hash(),
        adjudication_projection=projection,
    )
    pending_gold = {
        paper_id
        for adjudication in result.adjudications
        for paper_id in adjudication.candidate_gold_paper_ids
    }
    found_count = len(result.matched_pairs)
    found: CountValue = found_count
    if pending_gold:
        metric: MetricValue = _not_evaluable(
            f"{len(pending_gold)} gold paper(s) have pending match adjudications"
        )
    else:
        metric = found_count / gold.n_included if gold.n_included else 0.0

    paper_matches = _paper_matches(records, result, spec)
    definite_missed = [
        paper
        for paper in gold.included_papers
        if paper.paper_id in result.unmatched_gold_paper_ids
        and paper.paper_id not in pending_gold
    ]
    assigned_record_ids = {pair.pipeline_record_id for pair in result.matched_pairs}
    duplicate_record_ids = {
        item.pipeline_record_id for item in result.duplicate_manifestations
    }
    extra_dois = [
        normalized
        for record in records
        if record.id not in assigned_record_ids | duplicate_record_ids
        and (normalized := normalise_doi(record.doi))
    ][:20]
    precision: MetricValue
    if pending_gold:
        precision = metric
    else:
        precision = len(result.matched_pairs) / pipeline_count if pipeline_count else 0.0

    stage = StageEvaluation(
        stage=_STAGE_LABELS[stage_key],
        stage_key=stage_key,
        pipeline_count=pipeline_count,
        gold_count=gold.n_included,
        found_in_gold=found,
        recall=metric,
        precision=precision,
        sensitivity=metric,
        specificity=common_not_evaluable,
        missed_dois=[paper.doi for paper in definite_missed if paper.doi],
        missed_titles=[paper.title for paper in definite_missed if paper.title],
        extra_dois=extra_dois,
        notes=(
            [f"{len(pending_gold)} gold paper(s) await adjudication"]
            if pending_gold
            else []
        ),
        paper_matches=paper_matches,
        pending_adjudication_ids=tuple(
            adjudication.adjudication_id for adjudication in result.adjudications
        ),
        exclusion_reason_agreement=common_not_evaluable,
        field_level_charting_agreement=_not_evaluable(
            "no bound original charting results were supplied"
        ),
        evidence_locator_agreement=_not_evaluable(
            "no bound original evidence locators were supplied"
        ),
        prisma_count_comparison=_prisma_comparison(
            stage_key=stage_key, pipeline_count=pipeline_count, spec=spec
        ),
    )
    return stage, result.adjudications


def evaluate_pipeline(
    spec: ReproductionSpec,
    gold_standard: GoldStandard,
    stage_corpus: StageCorpus,
    *,
    adjudication_projection: Optional[dict[str, AdjudicationRecord]] = None,
) -> PipelineEvaluation:
    """Evaluate every available stage against exactly bound external inputs."""

    if spec.review_id != gold_standard.review_id:
        raise ValueError("reproduction spec review_id does not match gold standard")
    gold_hash = gold_standard.content_hash()
    if spec.expected_gold_standard_hash != gold_hash:
        raise ValueError(
            "reproduction spec expected_gold_standard_hash does not match the "
            "supplied gold standard"
        )

    stages_input: list[tuple[str, tuple[Record, ...]]] = [
        ("raw_records", stage_corpus.raw_records),
        ("deduplicated_records", stage_corpus.deduplicated_records),
        ("screened_records", stage_corpus.screened_records),
    ]
    if stage_corpus.full_text_records is not None:
        stages_input.append(("full_text_records", stage_corpus.full_text_records))
    if stage_corpus.included_records is not None:
        stages_input.append(("included_records", stage_corpus.included_records))

    stages: list[StageEvaluation] = []
    pending: dict[str, AdjudicationRecord] = {}
    projection = dict(adjudication_projection or {})
    for stage_key, records in stages_input:
        stage, stage_pending = _evaluate_stage(
            stage_key=stage_key,
            records=records,
            spec=spec,
            gold=gold_standard,
            projection=projection,
        )
        stages.append(stage)
        for adjudication in stage_pending:
            pending[adjudication.adjudication_id] = adjudication

    by_key = {stage.stage_key: stage for stage in stages}
    coverage: DatabaseCoverage = spec.coverage()
    post_cutoff = tuple(
        match
        for stage in stages
        for match in stage.paper_matches
        if not match.in_gold_standard and match.temporal_status == "post_cutoff"
    )
    # The same record can appear at several stages; report it once.
    post_cutoff = tuple({match.pipeline_id: match for match in post_cutoff}.values())
    summary: list[str] = []
    if coverage.partial_reproduction:
        summary.append(
            "Partial reproduction: missing original databases "
            + ", ".join(coverage.missing_databases)
        )
    if pending:
        summary.append(f"{len(pending)} match adjudication(s) remain pending")
    search_recall = by_key["raw_records"].recall
    final_stage = by_key.get("included_records")

    return PipelineEvaluation(
        review_id=spec.review_id,
        spec_hash=spec.spec_hash(),
        gold_standard_hash=gold_hash,
        source_mode=spec.source_mode,
        reproduction_mode=spec.reproduction_mode,
        missing_databases=coverage.missing_databases,
        unsupported_databases=coverage.unsupported_databases,
        partial_reproduction=coverage.partial_reproduction,
        gold_n_included=gold_standard.n_included,
        raw_records_hash=stage_corpus.raw_records_hash,
        deduplicated_records_hash=stage_corpus.deduplicated_records_hash,
        screened_records_hash=stage_corpus.screened_records_hash,
        full_text_records_hash=stage_corpus.full_text_records_hash,
        included_records_hash=stage_corpus.included_records_hash,
        stages=stages,
        search_recall=search_recall,
        final_recall=final_stage.recall if final_stage else None,
        final_precision=final_stage.precision if final_stage else None,
        adjudications=tuple(
            {
                **projection,
                **pending,
            }[key]
            for key in sorted({**projection, **pending})
        ),
        pending_adjudications=tuple(pending[key] for key in sorted(pending)),
        newly_discovered_post_cutoff=post_cutoff,
        summary_notes=summary,
    )


# Kept as a private compatibility name for callers/tests that previously
# reached into evaluator.py. Its semantics are now the structured matcher.
_match_records_to_gold = match_records_to_gold


__all__ = [
    "NotEvaluable",
    "NumericValueComparison",
    "PaperMatch",
    "PipelineEvaluation",
    "PrismaCountPair",
    "StageEvaluation",
    "ThematicConcordance",
    "classify_temporal_status",
    "evaluate_pipeline",
]
