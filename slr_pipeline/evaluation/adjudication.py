"""Human-adjudication records produced by gold-standard matching."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.extraction.hashing import sha256_hex


AdjudicationTier = Literal[
    "doi_collision",
    "other_persistent_id_collision",
    "exact_title_collision",
    "fuzzy_title",
]
AdjudicationResolution = Literal["pending", "matched", "no_match"]


class CandidatePairScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_paper_id: str
    pipeline_record_id: str
    score: float = Field(ge=0.0, le=1.0)
    corroborating_fields: tuple[Literal["year", "first_author_surname"], ...] = ()


class AdjudicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adjudication_id: str
    run_id: str
    tier: AdjudicationTier
    candidate_gold_paper_ids: tuple[str, ...]
    candidate_pipeline_record_ids: tuple[str, ...]
    scores: tuple[CandidatePairScore, ...]
    resolution: AdjudicationResolution = "pending"
    resolved_gold_paper_id: Optional[str] = None
    resolved_pipeline_record_id: Optional[str] = None
    reviewer: Optional[str] = None
    note: Optional[str] = None
    resolved_at: Optional[str] = None

    @model_validator(mode="after")
    def _validate_identity_and_resolution(self) -> "AdjudicationRecord":
        gold_ids = tuple(sorted(set(self.candidate_gold_paper_ids)))
        pipeline_ids = tuple(sorted(set(self.candidate_pipeline_record_ids)))
        if not gold_ids or not pipeline_ids:
            raise ValueError("adjudication candidates must be non-empty")
        if gold_ids != self.candidate_gold_paper_ids:
            raise ValueError("candidate_gold_paper_ids must be unique and sorted")
        if pipeline_ids != self.candidate_pipeline_record_ids:
            raise ValueError("candidate_pipeline_record_ids must be unique and sorted")

        score_pairs = [
            (score.gold_paper_id, score.pipeline_record_id) for score in self.scores
        ]
        if len(score_pairs) != len(set(score_pairs)):
            raise ValueError("candidate pair scores must be unique")
        if any(
            gold_id not in gold_ids or record_id not in pipeline_ids
            for gold_id, record_id in score_pairs
        ):
            raise ValueError("candidate pair score refers outside the candidate sets")
        if not self.scores:
            raise ValueError("adjudication must include at least one candidate score")

        expected_id = compute_adjudication_id(
            run_id=self.run_id,
            tier=self.tier,
            gold_paper_ids=gold_ids,
            pipeline_record_ids=pipeline_ids,
        )
        if self.adjudication_id != expected_id:
            raise ValueError("adjudication_id does not match its candidates")

        resolution_fields = (self.reviewer, self.note, self.resolved_at)
        if self.resolution == "pending":
            if any(value is not None for value in resolution_fields) or any(
                value is not None
                for value in (
                    self.resolved_gold_paper_id,
                    self.resolved_pipeline_record_id,
                )
            ):
                raise ValueError("pending adjudication cannot carry resolution fields")
        else:
            if any(not value or not value.strip() for value in resolution_fields):
                raise ValueError("resolved adjudication requires reviewer, note, and resolved_at")
            if self.resolution == "matched":
                if self.resolved_gold_paper_id not in gold_ids:
                    raise ValueError("resolved gold paper is not a candidate")
                if self.resolved_pipeline_record_id not in pipeline_ids:
                    raise ValueError("resolved pipeline record is not a candidate")
                if (
                    self.resolved_gold_paper_id,
                    self.resolved_pipeline_record_id,
                ) not in set(score_pairs):
                    raise ValueError("resolved pair was not a candidate pair")
            elif any(
                value is not None
                for value in (
                    self.resolved_gold_paper_id,
                    self.resolved_pipeline_record_id,
                )
            ):
                raise ValueError("no_match adjudication cannot name a resolved pair")
        return self


def compute_adjudication_id(
    *,
    run_id: str,
    tier: AdjudicationTier,
    gold_paper_ids: tuple[str, ...],
    pipeline_record_ids: tuple[str, ...],
) -> str:
    return sha256_hex(
        {
            "id_schema_version": 1,
            "run_id": run_id,
            "tier": tier,
            "candidate_gold_paper_ids": sorted(set(gold_paper_ids)),
            "candidate_pipeline_record_ids": sorted(set(pipeline_record_ids)),
        }
    )


def build_pending_adjudication(
    *,
    run_id: str,
    tier: AdjudicationTier,
    scores: tuple[CandidatePairScore, ...],
) -> AdjudicationRecord:
    ordered_scores = tuple(
        sorted(scores, key=lambda item: (item.gold_paper_id, item.pipeline_record_id))
    )
    gold_ids = tuple(sorted({score.gold_paper_id for score in ordered_scores}))
    pipeline_ids = tuple(
        sorted({score.pipeline_record_id for score in ordered_scores})
    )
    return AdjudicationRecord(
        adjudication_id=compute_adjudication_id(
            run_id=run_id,
            tier=tier,
            gold_paper_ids=gold_ids,
            pipeline_record_ids=pipeline_ids,
        ),
        run_id=run_id,
        tier=tier,
        candidate_gold_paper_ids=gold_ids,
        candidate_pipeline_record_ids=pipeline_ids,
        scores=ordered_scores,
    )


__all__ = [
    "AdjudicationRecord",
    "AdjudicationResolution",
    "AdjudicationTier",
    "CandidatePairScore",
    "build_pending_adjudication",
    "compute_adjudication_id",
]
