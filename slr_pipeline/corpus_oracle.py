"""`CORPUS_ORACLE.json`'s schema + loader -- Track 2 Step 12A
(`track2_step12_revision_proposal_2026-09-02.md`): the frozen, versioned
statement of what a pipeline run over the synthetic fixture corpus MUST
produce (record counts, dedup mappings, screening/full-text/extraction
outcomes, replay-call topology). Authored now (Step 12A, sub-step (a));
the full pipeline-driving acceptance harness that runs the reference run
end-to-end and diffs its actual output against this oracle is a later
step (13) and is not built here.

`corpus_oracle_version` bumps from "1" (the original pipeline-readiness
plan's schema, §7.2) to "2" with this module: the revision proposal's
mandated `oracle_kind` field is new in this version. `oracle_kind` names
WHAT this oracle validates -- pipeline branch-coverage / software
conformance against a synthetic corpus, never a scientific finding -- and
`_check_oracle_kind` below is the acceptance-harness assertion the
proposal requires: a missing or different `oracle_kind` value RAISES; it
is not merely documented in a comment or silently accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

CORPUS_ORACLE_VERSION = "2"
ORACLE_KIND = "software_conformance"


class DuplicateMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dup_id: str
    canonical_id: str
    basis: str


class CorpusOracleError(ValueError):
    """`CORPUS_ORACLE.json` failed schema validation or a cross-field
    consistency check -- an authoring bug, never silently tolerated or
    worked around by a caller."""


class CorpusOracle(BaseModel):
    """`CORPUS_ORACLE.json`'s full schema, version `"2"`."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    corpus_oracle_version: str
    oracle_kind: str
    raw_records: int
    records_per_fixture_db: Dict[str, int]
    duplicates_removed: int
    # Plan §7.2's own explanatory field, kept verbatim (`_dedup_note` on
    # disk) -- Pydantic field names cannot start with an underscore, so
    # the Python attribute is `dedup_note` with an explicit JSON alias.
    dedup_note: str = Field(alias="_dedup_note")
    duplicate_mappings: Tuple[DuplicateMapping, ...]
    unique_after_dedup: int
    title_abstract_inclusions: Tuple[str, ...]
    all_fixture_records_doi_less: bool
    local_pdfs_present: int
    full_text_sought: int
    full_text_review_queue_entries: int
    full_text_exclusions: Tuple[str, ...]
    full_text_eligible: Tuple[str, ...]
    research_questions: int
    map_relevant_by_rq: Dict[str, Tuple[str, ...]]
    screening_batches: int
    qc_sample_size: int
    ft_adjudication_calls: int
    extraction_batch_calls: int
    synthesis_map_calls: int
    synthesis_refine_calls: int
    total_replay_calls: int
    pdf_sha256: Dict[str, str]

    @model_validator(mode="after")
    def _check_oracle_kind(self) -> "CorpusOracle":
        # The acceptance-harness assertion the revision proposal mandates:
        # a missing or wrong oracle_kind must fail loudly here, not be
        # silently accepted, and not merely be noted in a comment.
        if self.oracle_kind != ORACLE_KIND:
            raise CorpusOracleError(
                f"CORPUS_ORACLE.json oracle_kind must be {ORACLE_KIND!r}, got {self.oracle_kind!r} -- "
                "this oracle validates SOFTWARE pipeline conformance against a synthetic corpus, "
                "never a scientific finding; a caller must never mistake this fixture for evidence."
            )
        return self

    @model_validator(mode="after")
    def _check_version(self) -> "CorpusOracle":
        if self.corpus_oracle_version != CORPUS_ORACLE_VERSION:
            raise CorpusOracleError(
                f"CORPUS_ORACLE.json corpus_oracle_version must be {CORPUS_ORACLE_VERSION!r}, "
                f"got {self.corpus_oracle_version!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> "CorpusOracle":
        if self.unique_after_dedup != self.raw_records - self.duplicates_removed:
            raise CorpusOracleError("unique_after_dedup must equal raw_records - duplicates_removed")
        if len(self.duplicate_mappings) != self.duplicates_removed:
            raise CorpusOracleError("duplicate_mappings length must equal duplicates_removed")
        if self.full_text_sought != len(self.title_abstract_inclusions):
            raise CorpusOracleError("full_text_sought must equal len(title_abstract_inclusions)")
        exclusions, eligible, inclusions = (
            set(self.full_text_exclusions), set(self.full_text_eligible), set(self.title_abstract_inclusions),
        )
        if exclusions | eligible != inclusions:
            raise CorpusOracleError("full_text_exclusions + full_text_eligible must partition title_abstract_inclusions")
        if exclusions & eligible:
            raise CorpusOracleError("full_text_exclusions and full_text_eligible must be disjoint")
        if sum(self.records_per_fixture_db.values()) != self.raw_records:
            raise CorpusOracleError("records_per_fixture_db must sum to raw_records")
        if set(self.pdf_sha256) != inclusions:
            raise CorpusOracleError("pdf_sha256 keys must be exactly title_abstract_inclusions")
        for rq, ids in self.map_relevant_by_rq.items():
            if set(ids) - eligible:
                raise CorpusOracleError(f"map_relevant_by_rq[{rq!r}] contains an id outside full_text_eligible")
        expected_total = (
            1 + 1 + self.screening_batches + self.qc_sample_size + self.ft_adjudication_calls
            + self.extraction_batch_calls + self.synthesis_map_calls + self.synthesis_refine_calls
        )
        if self.total_replay_calls != expected_total:
            raise CorpusOracleError(
                f"total_replay_calls ({self.total_replay_calls}) must equal "
                f"query_expansion(1) + db_translation(1) + screening_batches"
                f"({self.screening_batches}) + qc_sample_size({self.qc_sample_size}) + "
                f"ft_adjudication_calls({self.ft_adjudication_calls}) + "
                f"extraction_batch_calls({self.extraction_batch_calls}) + "
                f"synthesis_map_calls({self.synthesis_map_calls}) + "
                f"synthesis_refine_calls({self.synthesis_refine_calls}) = {expected_total}"
            )
        return self


def load_corpus_oracle(path: Path) -> CorpusOracle:
    """Read-only load + full-schema validation of `CORPUS_ORACLE.json`.
    Every check above runs unconditionally on load -- there is no
    "skip validation" path. `path` is required (Track-2 packaging-boundary
    fix, CI run 34155102694): this module is `slr_pipeline`-wheel-packaged
    and must not assume a checkout-relative `tests/` fixture default --
    zero current callers, so this is a signature-only fix for consistency,
    not a behavior change."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CorpusOracle.model_validate(payload)
