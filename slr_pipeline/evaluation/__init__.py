"""
Evaluation module for pipeline reproduction studies.

Compares pipeline outputs against a published review's gold standard to measure
stage-specific fidelity and identify reproducibility gaps.
"""

from .adjudication import AdjudicationRecord, CandidatePairScore
from .adjudication_journal import (
    AdjudicationJournalEntry,
    EvaluationAdjudicationStore,
    EvaluationBinding,
)
from .berg2023_gates import GateReport, GoldPaperOutcome, evaluate_berg2023_gates
from .bundle_artifacts import (
    ReproductionArtifactPaths,
    VerifiedReproductionArtifactSet,
    verify_reproduction_artifact_set,
)
from .evaluator import (
    NotEvaluable,
    PaperMatch,
    PipelineEvaluation,
    StageEvaluation,
    evaluate_pipeline,
)
from .gold_standard import GoldPaper, GoldStandard, load_gold_standard
from .matching import GoldMatchResult, MatchedPair, match_records_to_gold
from .reporter import generate_html_report
from .reproduction_spec import DatabaseCoverage, ReproductionSpec
from .stage_corpus import StageCorpus
from .workflow import (
    evaluate_with_adjudication_journal,
    evaluation_binding,
    resolve_adjudication,
)

__all__ = [
    "AdjudicationJournalEntry",
    "AdjudicationRecord",
    "CandidatePairScore",
    "DatabaseCoverage",
    "EvaluationAdjudicationStore",
    "EvaluationBinding",
    "GateReport",
    "GoldMatchResult",
    "GoldPaper",
    "GoldPaperOutcome",
    "GoldStandard",
    "MatchedPair",
    "NotEvaluable",
    "PaperMatch",
    "PipelineEvaluation",
    "ReproductionArtifactPaths",
    "ReproductionSpec",
    "StageCorpus",
    "StageEvaluation",
    "VerifiedReproductionArtifactSet",
    "evaluate_berg2023_gates",
    "evaluate_pipeline",
    "evaluate_with_adjudication_journal",
    "evaluation_binding",
    "generate_html_report",
    "load_gold_standard",
    "match_records_to_gold",
    "resolve_adjudication",
    "verify_reproduction_artifact_set",
]
