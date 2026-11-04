"""Strict loading and cross-verification of reproduction bundle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adjudication_journal import (
    EvaluationBinding,
    latest_state_projection,
    verify_adjudication_journal_bytes,
)
from .berg2023_gates import GateReport, evaluate_berg2023_gates
from .evaluator import PipelineEvaluation, evaluate_pipeline
from .gold_standard import GoldStandard
from .reproduction_spec import ReproductionSpec
from .stage_corpus import StageCorpus
from .workflow import evaluation_binding


@dataclass(frozen=True)
class ReproductionArtifactPaths:
    """The complete, indivisible artifact set for one reproduction attempt."""

    reproduction_spec: Path
    gold_standard: Path
    stage_corpus: Path
    pipeline_evaluation: Path
    gate_report: Path
    adjudication_journal: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return tuple((name, Path(getattr(self, name))) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class VerifiedReproductionArtifactSet:
    paths: ReproductionArtifactPaths
    spec: ReproductionSpec
    gold_standard: GoldStandard
    stage_corpus: StageCorpus
    pipeline_evaluation: PipelineEvaluation
    gate_report: GateReport
    journal_binding: EvaluationBinding


def verify_reproduction_artifact_set(
    paths: ReproductionArtifactPaths,
) -> VerifiedReproductionArtifactSet:
    """Load all files and reproduce both outputs without mutating inputs."""

    resolved: set[Path] = set()
    for name, path in paths.items():
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"reproduction artifact {name!r} is not a file: {path}")
        if path in resolved:
            raise ValueError(f"reproduction artifact {name!r} aliases another artifact")
        resolved.add(path)

    spec = ReproductionSpec.model_validate_json(paths.reproduction_spec.read_bytes())
    gold = GoldStandard.model_validate_json(paths.gold_standard.read_bytes())
    corpus = StageCorpus.model_validate_json(paths.stage_corpus.read_bytes())
    evaluation = PipelineEvaluation.model_validate_json(paths.pipeline_evaluation.read_bytes())
    gate_report = GateReport.model_validate_json(paths.gate_report.read_bytes())
    entries = verify_adjudication_journal_bytes(paths.adjudication_journal.read_bytes())

    binding = evaluation_binding(spec, gold, corpus)
    if any(entry.binding != binding for entry in entries):
        raise ValueError("adjudication journal binding does not match bundled inputs")
    projection = latest_state_projection(entries)
    evaluation_adjudications = {
        record.adjudication_id: record for record in evaluation.adjudications
    }
    if projection != evaluation_adjudications:
        raise ValueError(
            "adjudication journal latest projection does not match the pipeline "
            "evaluation's adjudications"
        )
    rebuilt_evaluation = evaluate_pipeline(spec, gold, corpus, adjudication_projection=projection)
    if rebuilt_evaluation != evaluation:
        raise ValueError("pipeline evaluation does not reproduce from bundled inputs")
    rebuilt_gate_report = evaluate_berg2023_gates(rebuilt_evaluation, spec, gold, corpus)
    if rebuilt_gate_report != gate_report:
        raise ValueError("gate report does not reproduce from bundled inputs")

    return VerifiedReproductionArtifactSet(
        paths=paths,
        spec=spec,
        gold_standard=gold,
        stage_corpus=corpus,
        pipeline_evaluation=evaluation,
        gate_report=gate_report,
        journal_binding=binding,
    )


__all__ = [
    "ReproductionArtifactPaths",
    "VerifiedReproductionArtifactSet",
    "verify_reproduction_artifact_set",
]