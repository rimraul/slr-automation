"""
quality_appraisal_node -- Stage 1 (rapid / SLR reviews only) of
quality_appraisal_synthesis_plan_2026-08-22.md. A structural,
evidence-verified rebuild of the prior LLM quality-appraisal step, closing
six confirmed bugs in the old implementation (Section 1.1): a processing
failure silently counted as passing; a title/abstract fallback survived
alongside a full-text path; an unsupported checklist silently substituted
JBI; appraisal exclusion was hard, synchronous, and unreviewable; no item
carried quoted/located evidence; and the graph gave quality appraisal no
per-review-type conditional-edge treatment at all.

CONFLATION FIX (Section 1.2): eligibility is decided exactly once, by
full-text screening (fulltext_node, via FT_SCREENING). This node -- and
every function in this module -- NEVER mutates `full_text_records`, under
any circumstance (not a processing failure, not an unresolved item, not a
threshold; there is no `gate_on_threshold`/`quality_threshold` config
field at all). A record with an unresolved leaf sits in
`quality_appraisal_review_queue`; the graph gates *synthesis* on that
queue being empty (Section 1.10), never the record's own eligibility.

Two structurally distinct rubrics (Section 1.3): `teaching_appraisal_v1`
(generic, JBI-structure-inspired, never claiming JBI conformance) and
`rob2_inspired_study_level_teaching_v1` (RoB2's real 5-domain/signalling-
question shape, simplified to study level -- `outcome_scope=
"study_level_simplified"`). Full-document, deterministic evidence
retrieval (Section 1.4): chunking/batching are REUSED directly from the
frozen Phase A extraction engine (`extraction/parsing.py`,
`extraction/batching.py`) -- never `nodes/extraction.py` itself (Section
1.8's import boundary). Cross-batch aggregation is a separate, non-LLM,
deterministic step: single-batch or unanimous multi-batch evidence
finalizes an answer; contradictory evidence across batches is NEVER
auto-resolved (no majority vote, no last-batch-wins) -- it is queued with
`reason="conflicting_evidence_across_batches"`, both sides' evidence
preserved on the queue entry for the human reviewer.

Provenance is split, not binary (Section 1.6): `decision_origin` (ai |
human) x `evidence_status` (verified_from_chunk | human_attested |
not_provided | not_applicable). A negative/high-risk machine leaf ("no",
or RoB2's "N"/"PN") is always provisional until a human acts; whether that
provisional state BLOCKS the graph gate or is recorded non-blockingly is
`QualityAppraisalConfig.negative_high_risk_policy`. Human CONFIRMATION
(`confirm_quality_appraisal_provisional_item`) is a distinct action from
human RESOLUTION (`resolve_quality_appraisal_review_queue`): confirming
preserves the machine's own verified evidence byte-for-byte and only adds
an audit-trail `HumanConfirmation`; resolving genuinely overwrites a leaf
with a human-attested value.

Own independent, three-layer durable artifact (Section 1.6, Section 4's
"Worries" note): `qa_manifest.json` (frozen) + `checkpoints/` (frozen,
content-hashed, one per document) + `qa_resolutions.json`/
`qa_confirmations.json` (mutable-by-humans, append-only, self-hashed,
sharing one kernel-backed `qa_resolutions.lock`). Content-addressed
`qa_run_id`, PDF path excluded, `negative_high_risk_policy` included (two
runs differing only in that policy must never collide on one run_id).
Entirely independent of `ExtractionCheckpointStore`/
`extraction_resolutions.json` -- extraction's run identity is recorded in
`qa_manifest.json` as provenance/dependency metadata only, never folded
into `qa_run_id`'s own hash.

No aggregate score is ever computed or treated as authoritative anywhere
in this module.

Node input keys:  full_text_records, config, prisma_counts
Node output keys: quality_appraisal_results, quality_appraisal_review_queue,
                   quality_appraisal_provisional_ledger,
                   requires_human_review, quality_appraisal_run_id,
                   quality_appraisal_manifest_path,
                   quality_appraisal_manifest_hash,
                   quality_appraisal_artifact_paths, prisma_counts
                   (full_text_records is NEVER included in this node's
                   return dict -- it is never touched.)
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import fcntl  # POSIX: kernel-backed advisory lock, released on process exit
except ImportError:  # pragma: no cover - exercised only on POSIX
    fcntl = None

try:
    import msvcrt  # Windows: kernel-backed advisory lock, released on process exit
except ImportError:  # pragma: no cover - exercised only on Windows
    msvcrt = None

from slr_pipeline.state import PRISMACounts, Record, ReviewState
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key
from slr_pipeline.extraction.batching import ExtractionBatch, build_batches
from slr_pipeline.extraction.evidence import EvidenceVerificationError, VerifiedEvidence, normalize_for_matching, verify_evidence
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.parsing import DoclingParserOptions, DocumentParseError, ParsedChunk, ParsedDocument, parse_pdf
from slr_pipeline.synthesis_evidence import _human_attested_chunk_id, build_evidence_index_for_field

logger = logging.getLogger(__name__)

EVIDENCE_VERIFIER_VERSION = "qa-evidence-verifier-1"  # tracks extraction.evidence's verify_evidence() contract
PROMPT_TEMPLATE_VERSION_TEACHING = "qa-teaching-openai-prompt-1"
PROMPT_TEMPLATE_VERSION_ROB2 = "qa-rob2-openai-prompt-1"
CHECKLIST_VERSION = "1"


# ---------------------------------------------------------------------------
# Checklist identity (Section 1.3) -- static, versioned specifications, not
# arbitrary imported Python (same "data, not code" discipline
# extraction/schema.py's own module docstring documents).
# ---------------------------------------------------------------------------

TEACHING_APPRAISAL_ITEMS: Tuple[Dict[str, str], ...] = (
    {
        "item_id": "clear_research_question",
        "question": "Is the research question or objective clearly stated?",
        "description": "Whether the paper states an explicit research question, aim, or objective.",
    },
    {
        "item_id": "appropriate_study_design",
        "question": "Is the study design appropriate for the stated research question?",
        "description": "Whether the chosen design (e.g. RCT, cohort, case study, survey) plausibly fits the question asked.",
    },
    {
        "item_id": "recruitment_or_sampling_adequacy",
        "question": "Is the recruitment or sampling strategy adequately described and appropriate?",
        "description": "Whether participants/units/documents were selected via a described, defensible procedure.",
    },
    {
        "item_id": "data_collection_adequacy",
        "question": "Is the data collection method adequately described and appropriate?",
        "description": "Whether instruments/procedures used to collect data are described in enough detail to assess adequacy.",
    },
    {
        "item_id": "analysis_rigor",
        "question": "Is the data analysis method rigorous and appropriate?",
        "description": "Whether the analytic approach fits the data and research question and is described with enough detail to assess rigor.",
    },
    {
        "item_id": "findings_clearly_stated",
        "question": "Are the findings clearly stated and supported by the reported data?",
        "description": "Whether results are stated explicitly and traceable to the data presented, not merely asserted.",
    },
    {
        "item_id": "limitations_acknowledged",
        "question": "Are the study's limitations acknowledged?",
        "description": "Whether the paper discusses its own limitations, threats to validity, or scope constraints.",
    },
)

# MVP subset of signalling questions per RoB2 domain (Section 1.3: "MVP
# subset of signalling questions per domain"). `selection_of_the_reported_result`
# carries only one question and an explicit caveat (Section 1.3/1.4): a
# real RoB2 selective-reporting judgment requires a registered protocol
# this MVP does not collect.
ROB2_DOMAINS: Tuple[Dict[str, Any], ...] = (
    {
        "domain_id": "randomisation_process",
        "questions": (
            {"question_id": "1.1", "question": "Was the allocation sequence random?"},
            {"question_id": "1.2", "question": "Was the allocation sequence concealed until participants were enrolled and assigned to interventions?"},
        ),
    },
    {
        "domain_id": "deviations_from_intended_interventions",
        "questions": (
            {"question_id": "2.1", "question": "Were participants aware of their assigned intervention during the study?"},
            {"question_id": "2.2", "question": "Were there deviations from the intended intervention that arose because of the trial context?"},
        ),
    },
    {
        "domain_id": "missing_outcome_data",
        "questions": (
            {"question_id": "3.1", "question": "Were outcome data available for all, or nearly all, participants/units randomized or enrolled?"},
            {"question_id": "3.2", "question": "Is there evidence that the result was not biased by missing outcome data?"},
        ),
    },
    {
        "domain_id": "measurement_of_the_outcome",
        "questions": (
            {"question_id": "4.1", "question": "Was the method of measuring the outcome appropriate?"},
            {"question_id": "4.2", "question": "Could measurement or ascertainment of the outcome have differed between comparison groups?"},
        ),
    },
    {
        "domain_id": "selection_of_the_reported_result",
        "questions": (
            {
                "question_id": "5.1",
                "question": (
                    "Was the reported result selected from among multiple outcome measurements or analyses, "
                    "without evidence the paper followed a pre-specified plan? (No protocol/registration was "
                    "supplied to this MVP appraisal -- this question can only be assessed against the paper's "
                    "OWN internal statements, never against an external registered protocol.)"
                ),
            },
        ),
    },
)


def _checklist_content_hash(checklist_id: str) -> str:
    if checklist_id == "teaching_appraisal_v1":
        return sha256_hex({"checklist_id": checklist_id, "items": list(TEACHING_APPRAISAL_ITEMS)})
    return sha256_hex({"checklist_id": checklist_id, "domains": list(ROB2_DOMAINS)})


# ---------------------------------------------------------------------------
# Pydantic result models (Section 1.3, 1.5, 1.6)
# ---------------------------------------------------------------------------

def _expected_evidence_status(*, decision_origin: str, is_definitive: bool, is_not_applicable: bool) -> str:
    """The ONE precise (status/answer, decision_origin) -> required
    evidence_status mapping (Section 1.6's allowed table, closed):
    `decision_origin="human"` ALWAYS implies `evidence_status=
    "human_attested"`, regardless of the resolved value (a human
    resolving to "not_applicable"/"unclear"/"NI" is still an attestation,
    never re-labeled `not_applicable`/`not_provided` -- those two values
    are reserved for the MACHINE's own no-evidence states). For
    `decision_origin="ai"`: `not_applicable` -> `not_applicable`,
    a definitive answer (yes/no, Y/PY/PN/N) -> `verified_from_chunk`,
    anything else (unclear/NI) -> `not_provided`. Replaces an earlier,
    looser "is this tuple in an allowed set" check that let e.g.
    `(status="yes", decision_origin="ai", evidence_status="not_provided")`
    pass validation just because `("ai","not_provided")` is a legitimate
    tuple for a DIFFERENT status."""
    if decision_origin == "human":
        return "human_attested"
    if is_not_applicable:
        return "not_applicable"
    return "verified_from_chunk" if is_definitive else "not_provided"


class AppraisalEvidence(BaseModel):
    """One verified evidence span. A LIST, never a single field, because
    multiple batches can independently corroborate the same item (Section
    1.3)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    quotation: str
    locator: str  # verifier-derived (page/section from the matched chunk), never model-claimed


class HumanConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str
    note: str
    resolved_at: str  # ISO-8601


class AppraisalItemResponse(BaseModel):
    """`teaching_appraisal_v1`'s leaf. There is no aggregation level above
    this for the teaching rubric (Section 1.5) -- this item IS the leaf."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    status: Literal["yes", "no", "unclear", "not_applicable"]
    evidence: List[AppraisalEvidence] = Field(default_factory=list)
    rationale: str
    decision_origin: Literal["ai", "human"] = "ai"
    evidence_status: Literal["verified_from_chunk", "human_attested", "not_provided", "not_applicable"] = "not_provided"
    chunks_examined: List[str] = Field(default_factory=list)
    batches_examined: int = 0
    provisional_pending_human_confirmation: bool = False
    human_confirmation: Optional[HumanConfirmation] = None
    # Persisted so a reloaded checkpoint is self-contained (Section 1.11's
    # "resolved_quality_projection reconstructable from disk alone"):
    # `coverage_complete=False` means at least one batch that COULD have
    # spoken to this leaf failed the call or evidence verification -- even
    # when `status` itself carries a real, verified value (a leaf is never
    # silently finalized as complete just because SOME batch agreed;
    # Section 1.4's own coverage-transparency intent). `queue_reason` is
    # the single source of truth for "does this leaf currently need a
    # RESOLUTION" (None -> no); `conflicting_evidence` is populated only
    # when queue_reason=="conflicting_evidence_across_batches".
    coverage_complete: bool = True
    queue_reason: Optional[str] = None
    conflicting_evidence: Optional[Dict[str, List[Dict[str, str]]]] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "AppraisalItemResponse":
        is_definitive = self.status in ("yes", "no")
        is_not_applicable = self.status == "not_applicable"
        expected = _expected_evidence_status(decision_origin=self.decision_origin, is_definitive=is_definitive, is_not_applicable=is_not_applicable)
        if self.evidence_status != expected:
            raise ValueError(
                f"item {self.item_id!r}: status={self.status!r}, decision_origin={self.decision_origin!r} "
                f"requires evidence_status={expected!r}, got {self.evidence_status!r}"
            )
        if is_definitive:
            if not self.evidence:
                raise ValueError(f"item {self.item_id!r}: status={self.status!r} requires non-empty evidence")
        elif self.evidence:
            raise ValueError(f"item {self.item_id!r}: status={self.status!r} must not carry evidence")
        # Round-4 finding 3 fix: a MACHINE leaf with a negative status must
        # ALWAYS be either still-pending confirmation OR already
        # explicitly confirmed -- structurally enforced, not merely a
        # convention the aggregation code happens to follow. A human
        # override to "no" is exempt (decision_origin=="human" already
        # means a human exercised explicit oversight)."""
        if self.decision_origin == "ai" and self.status == "no" and not self.provisional_pending_human_confirmation and self.human_confirmation is None:
            raise ValueError(f"item {self.item_id!r}: a machine status='no' must be provisional_pending_human_confirmation=True or already carry a human_confirmation")
        return self


_TEACHING_ITEM_IDS = frozenset(i["item_id"] for i in TEACHING_APPRAISAL_ITEMS)


class TeachingAppraisal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[AppraisalItemResponse]

    @model_validator(mode="after")
    def _exact_item_membership(self) -> "TeachingAppraisal":
        """Round-4 finding 3 fix: a teaching appraisal must carry EXACTLY
        the checklist's own 7 canonical item_ids, no more, no fewer, no
        duplicates -- previously unconstrained, so a fragment (e.g. one of
        seven required items) was a valid, constructable `TeachingAppraisal`."""
        ids = [i.item_id for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"TeachingAppraisal.items contains duplicate item_id(s): {sorted({x for x in ids if ids.count(x) > 1})}")
        if set(ids) != _TEACHING_ITEM_IDS:
            missing, extra = _TEACHING_ITEM_IDS - set(ids), set(ids) - _TEACHING_ITEM_IDS
            raise ValueError(f"TeachingAppraisal.items does not exactly match the checklist's own item_ids -- missing={sorted(missing)!r}, extra={sorted(extra)!r}")
        return self

    @property
    def is_provisional(self) -> bool:
        """DERIVED, never an independently-settable field (Section 1.6)."""
        return any(i.provisional_pending_human_confirmation for i in self.items)


class SignallingQuestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str  # stable id within the domain, e.g. "1.1", "1.2"
    answer: Literal["Y", "PY", "PN", "N", "NI"]  # RoB2's real 5-state scale
    evidence: List[AppraisalEvidence] = Field(default_factory=list)  # empty iff answer == "NI"
    rationale: str
    decision_origin: Literal["ai", "human"] = "ai"
    evidence_status: Literal["verified_from_chunk", "human_attested", "not_provided", "not_applicable"] = "not_provided"
    chunks_examined: List[str] = Field(default_factory=list)
    batches_examined: int = 0
    provisional_pending_human_confirmation: bool = False
    human_confirmation: Optional[HumanConfirmation] = None
    coverage_complete: bool = True
    queue_reason: Optional[str] = None
    conflicting_evidence: Optional[Dict[str, List[Dict[str, str]]]] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "SignallingQuestionResponse":
        is_definitive = self.answer != "NI"
        expected = _expected_evidence_status(decision_origin=self.decision_origin, is_definitive=is_definitive, is_not_applicable=False)
        if self.evidence_status != expected:
            raise ValueError(
                f"signalling question {self.question_id!r}: answer={self.answer!r}, decision_origin={self.decision_origin!r} "
                f"requires evidence_status={expected!r}, got {self.evidence_status!r}"
            )
        if is_definitive and not self.evidence:
            raise ValueError(f"signalling question {self.question_id!r}: answer={self.answer!r} requires non-empty evidence")
        if not is_definitive and self.evidence:
            raise ValueError(f"signalling question {self.question_id!r}: answer='NI' must not carry evidence")
        # Round-4 finding 3 fix: same machine-negative-implies-provisional
        # invariant as AppraisalItemResponse's own (above).
        if self.decision_origin == "ai" and self.answer in ("N", "PN") and not self.provisional_pending_human_confirmation and self.human_confirmation is None:
            raise ValueError(f"signalling question {self.question_id!r}: a machine answer={self.answer!r} must be provisional_pending_human_confirmation=True or already carry a human_confirmation")
        return self


_SIGNALLING_SEVERITY = {"Y": 0, "PY": 1, "NI": 2, "PN": 3, "N": 4}


def roll_up_domain_risk(
    signalling_questions: List[SignallingQuestionResponse],
) -> Tuple[Literal["low", "some_concerns", "high"], Literal["evidence_based", "no_information_default"]]:
    """`max_severity_v1` -- the ONE algorithm this MVP uses (Section 1.3),
    fully specified for every possible input, not an example among
    others."""
    severities = [_SIGNALLING_SEVERITY[q.answer] for q in signalling_questions]
    max_severity = max(severities)
    all_ni = all(q.answer == "NI" for q in signalling_questions)
    risk = "low" if max_severity <= 1 else "high" if max_severity == 4 else "some_concerns"
    basis = "no_information_default" if all_ni else "evidence_based"
    return risk, basis


def roll_up_overall_risk(domains: List["DomainJudgment"]) -> Literal["low", "some_concerns", "high"]:
    """`max_domain_severity_v1` (Section 1.3): overall risk is the single
    highest-risk domain."""
    order = {"low": 0, "some_concerns": 1, "high": 2}
    return max(domains, key=lambda d: order[d.domain_risk]).domain_risk


_ROB2_QUESTION_IDS_BY_DOMAIN = {d["domain_id"]: frozenset(q["question_id"] for q in d["questions"]) for d in ROB2_DOMAINS}


class DomainJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_id: Literal[
        "randomisation_process", "deviations_from_intended_interventions",
        "missing_outcome_data", "measurement_of_the_outcome",
        "selection_of_the_reported_result",
    ]
    signalling_questions: List[SignallingQuestionResponse]
    domain_risk: Literal["low", "some_concerns", "high"]
    domain_risk_rationale: str
    basis: Literal["evidence_based", "no_information_default"]
    roll_up_method_id: Literal["max_severity_v1"] = "max_severity_v1"

    @model_validator(mode="after")
    def _exact_question_membership_and_correct_rollup(self) -> "DomainJudgment":
        """Round-4 finding 3 fix: (a) a domain must carry EXACTLY its own
        canonical signalling-question set (Section 1.3's own MVP subset per
        domain), no more/fewer/duplicate; (b) `domain_risk`/`basis` must
        equal what `roll_up_domain_risk()` -- the ONE algorithm this MVP
        uses -- actually produces from these SAME signalling_questions.
        Previously unconstrained: a domain with stored risk labels
        contradicting the deterministic roll-up was a valid,
        constructable `DomainJudgment`."""
        ids = [q.question_id for q in self.signalling_questions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"DomainJudgment({self.domain_id!r}).signalling_questions contains duplicate question_id(s): {sorted({x for x in ids if ids.count(x) > 1})}")
        expected_ids = _ROB2_QUESTION_IDS_BY_DOMAIN[self.domain_id]
        if set(ids) != expected_ids:
            missing, extra = expected_ids - set(ids), set(ids) - expected_ids
            raise ValueError(f"DomainJudgment({self.domain_id!r}).signalling_questions does not exactly match its own canonical question_ids -- missing={sorted(missing)!r}, extra={sorted(extra)!r}")
        expected_risk, expected_basis = roll_up_domain_risk(list(self.signalling_questions))
        if self.domain_risk != expected_risk or self.basis != expected_basis:
            raise ValueError(
                f"DomainJudgment({self.domain_id!r}): domain_risk={self.domain_risk!r}/basis={self.basis!r} does not match "
                f"roll_up_domain_risk()'s own deterministic result ({expected_risk!r}, {expected_basis!r}) for these signalling_questions"
            )
        return self

    @property
    def is_provisional(self) -> bool:
        """DERIVED, never an independently-settable field (Section 1.6)."""
        return any(q.provisional_pending_human_confirmation for q in self.signalling_questions)


_ROB2_DOMAIN_IDS = frozenset(d["domain_id"] for d in ROB2_DOMAINS)


class RoB2InspiredAppraisal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domains: List[DomainJudgment]
    overall_risk: Literal["low", "some_concerns", "high"]
    overall_risk_rationale: str
    overall_roll_up_method_id: Literal["max_domain_severity_v1"] = "max_domain_severity_v1"
    outcome_scope: Literal["study_level_simplified"] = "study_level_simplified"

    @model_validator(mode="after")
    def _exact_domain_membership_and_correct_rollup(self) -> "RoB2InspiredAppraisal":
        """Round-4 finding 3 fix -- same reasoning as `DomainJudgment`'s
        own validator, one level up: exactly the 5 canonical RoB2 domains,
        no duplicates, and `overall_risk` must equal what
        `roll_up_overall_risk()` actually produces from these domains."""
        ids = [d.domain_id for d in self.domains]
        if len(set(ids)) != len(ids):
            raise ValueError(f"RoB2InspiredAppraisal.domains contains duplicate domain_id(s): {sorted({x for x in ids if ids.count(x) > 1})}")
        if set(ids) != _ROB2_DOMAIN_IDS:
            missing, extra = _ROB2_DOMAIN_IDS - set(ids), set(ids) - _ROB2_DOMAIN_IDS
            raise ValueError(f"RoB2InspiredAppraisal.domains does not exactly match the 5 canonical domain_ids -- missing={sorted(missing)!r}, extra={sorted(extra)!r}")
        expected_overall = roll_up_overall_risk(list(self.domains))
        if self.overall_risk != expected_overall:
            raise ValueError(
                f"RoB2InspiredAppraisal: overall_risk={self.overall_risk!r} does not match roll_up_overall_risk()'s "
                f"own deterministic result ({expected_overall!r}) for these domains"
            )
        return self

    @property
    def is_provisional(self) -> bool:
        """DERIVED (Section 1.6): transitively true iff any leaf anywhere
        in this appraisal still is."""
        return any(d.is_provisional for d in self.domains)


class DocumentAppraisal(BaseModel):
    """One document's complete appraisal result -- the content stored
    inside each per-document checkpoint envelope. Exactly one of
    `teaching`/`rob2` is set, matching `checklist_id`."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    checklist_id: Literal["teaching_appraisal_v1", "rob2_inspired_study_level_teaching_v1"]
    teaching: Optional[TeachingAppraisal] = None
    rob2: Optional[RoB2InspiredAppraisal] = None

    @model_validator(mode="after")
    def _validate_exactly_one(self) -> "DocumentAppraisal":
        if self.checklist_id == "teaching_appraisal_v1":
            if self.teaching is None or self.rob2 is not None:
                raise ValueError("checklist_id='teaching_appraisal_v1' requires `teaching` set and `rob2` unset")
        else:
            if self.rob2 is None or self.teaching is not None:
                raise ValueError("checklist_id='rob2_inspired_study_level_teaching_v1' requires `rob2` set and `teaching` unset")
        return self

    @property
    def is_provisional(self) -> bool:
        return self.rob2.is_provisional if self.rob2 is not None else self.teaching.is_provisional

    def leaves(self) -> Dict[str, Any]:
        """field_id -> leaf response object. `field_id` is `item_id` for
        teaching_appraisal_v1, or f"{domain_id}.{question_id}" for
        rob2_inspired_study_level_teaching_v1 (matches Section 3.2's own
        `evidence_id` derivation note)."""
        if self.teaching is not None:
            return {i.item_id: i for i in self.teaching.items}
        out: Dict[str, Any] = {}
        for d in self.rob2.domains:
            for q in d.signalling_questions:
                out[f"{d.domain_id}.{q.question_id}"] = q
        return out


def _build_review_queue_and_ledger(
    document_appraisals: List[DocumentAppraisal], *, resolved_keys: set, confirmed_keys: set,
    negative_high_risk_policy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """THE ONE place every recovery path (a fresh `quality_appraisal_node`
    run, `resolve_quality_appraisal_review_queue`,
    `confirm_quality_appraisal_provisional_item`) derives the current
    `quality_appraisal_review_queue`/`quality_appraisal_provisional_ledger`
    from -- always from `document_appraisals`' own persisted leaf fields
    (`queue_reason`, `provisional_pending_human_confirmation`), never from
    a separately-threaded, possibly-divergent bookkeeping structure
    (Finding 1/3 fix: this is what makes a leaf's need for resolution
    reloadable from a checkpoint alone, and lets EITHER a `resolution`- or
    `confirmation`-flagged leaf be closed via resolution, not just its own
    narrower action).

    A leaf whose key is already in `resolved_keys` never re-enters the
    queue in ANY form, regardless of its own still-machine-recorded
    `queue_reason`/`provisional_pending_human_confirmation` (both of
    which describe the FROZEN checkpoint, not the current, resolution-
    applied state) -- a resolution always takes precedence, since
    resolving is this module's own "override" action, genuinely capable
    of closing a `queue_reason` entry, a `confirmation` entry, or both at
    once (Finding 1: a provisional negative finding is no longer a
    permanent dead end -- a human who disagrees with it calls
    `resolve_quality_appraisal_review_queue`, exactly like any other
    queued leaf).

    A leaf still needing RESOLUTION (`queue_reason is not None`) always
    takes queue-entry precedence over a leaf ALSO being provisional (a
    negative/high-risk finding with e.g. `queue_reason="incomplete_coverage"`
    is queued once, as `queue_kind="resolution"` -- confirming an
    incompletely-covered finding without addressing the coverage gap
    would be premature).

    Round-6 fix: delegates the precedence decision to
    `_canonical_queue_kind_for_leaf()` -- the SAME function
    `_verify_resolution_entry`/`_verify_confirmation_entry` use to
    validate a persisted human action's own eligibility -- rather than
    re-implementing the identical `queue_reason`/`provisional`/policy
    precedence logic a second time inline. Two independent
    implementations of the same rule are exactly how the two could
    silently drift apart; there is now only one."""
    review_queue: List[Dict[str, Any]] = []
    provisional_ledger: List[Dict[str, Any]] = []
    for da in document_appraisals:
        for field_id, leaf in da.leaves().items():
            key = (da.record_id, field_id)
            if key in resolved_keys:
                continue
            kind = _canonical_queue_kind_for_leaf(leaf, negative_high_risk_policy=negative_high_risk_policy)
            if kind == "resolution":
                review_queue.append({
                    "record_id": da.record_id, "field_id": field_id, "queue_kind": "resolution",
                    "reason": leaf.queue_reason, "conflicting_evidence": leaf.conflicting_evidence,
                    "leaf_response": leaf.model_dump(mode="json"),
                })
            elif kind == "confirmation":
                if key in confirmed_keys:
                    continue
                review_queue.append({
                    "record_id": da.record_id, "field_id": field_id, "queue_kind": "confirmation",
                    "leaf_response": leaf.model_dump(mode="json"),
                })
            elif leaf.provisional_pending_human_confirmation:
                # kind is None (not a queue entry at all) but the leaf is
                # still provisional -- this is the allow_provisional_ai_only
                # non-blocking-ledger case, `_canonical_queue_kind_for_leaf`'s
                # own documented "never a queue entry under this policy" path.
                if key in confirmed_keys:
                    continue
                provisional_ledger.append({"record_id": da.record_id, "field_id": field_id, "leaf_response": leaf.model_dump(mode="json")})
    return review_queue, provisional_ledger


# ---------------------------------------------------------------------------
# OpenAI call layer -- mirrors extraction/openai_runner.py's own shape
# (raw pre-verification response, one explicit provider path, credentials
# read lazily by the SDK), but is NOT imported from there: teaching_appraisal_v1's
# "all items in one call" and rob2's "one domain per call" batching shapes
# are QA-specific, never a field-by-field extraction contract.
# ---------------------------------------------------------------------------

class QACallError(Exception):
    """A single batch's LLM call failed -- network/API error, malformed
    JSON, schema-validation failure, or a provider-model mismatch (the
    response's own `.model` did not equal `config.expected_provider_model`
    -- Finding 6a fix: `expected_provider_model` is bound into `qa_run_id`'s
    own hash, but until this fix, neither call function ever actually
    checked the provider's returned model against it, so the commitment
    the run identity makes was unenforced). No retry, no fallback: the
    caller records this batch's contribution as `call_failed` for every
    leaf it covers."""


class RawAppraisalEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    quotation: str


class RawTeachingItemBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: Literal["yes", "no", "not_applicable", "not_found"]
    rationale: Optional[str] = None
    evidence: List[RawAppraisalEvidenceItem] = Field(default_factory=list)


class RawTeachingBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: Dict[str, RawTeachingItemBatchResponse]


class RawSignallingQuestionBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Literal["Y", "PY", "PN", "N", "NI"]
    rationale: Optional[str] = None
    evidence: List[RawAppraisalEvidenceItem] = Field(default_factory=list)


class RawDomainBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: Dict[str, RawSignallingQuestionBatchResponse]


def _chunk_block(chunk: ParsedChunk) -> str:
    page_label = chunk.page_range or "unknown"
    return f"[chunk_id: {chunk.chunk_id} | section: {chunk.section} | page: {page_label}]\n{chunk.text}"


_NO_SEMANTICS_INSTRUCTION = (
    'A "no"/"N"/"PN" answer is only correct when this batch\'s text contains an EXPLICIT statement of the '
    'negative condition. The mere ABSENCE of any mention of the topic is never grounds for "no"/"N"/"PN" -- '
    'report "not_found"/"NI" instead in that case, even if you suspect the true answer is "no". Never guess.'
)


def build_teaching_batch_prompt(*, record_title: str, batch_chunks: List[ParsedChunk]) -> str:
    items_text = "\n".join(f'- item_id: "{i["item_id"]}"\n  question: {i["question"]}\n  guidance: {i["description"]}' for i in TEACHING_APPRAISAL_ITEMS)
    chunks_text = "\n\n".join(_chunk_block(c) for c in batch_chunks)
    return f"""You are appraising the methodological quality of ONE batch of a scientific paper's text ("{record_title}") for a systematic review, using a generic teaching-oriented checklist. Answer ONLY using the text in this batch below -- never use outside knowledge.

For EACH item listed, report exactly one outcome:
- "yes": this batch's text directly supports answering the item's question affirmatively. Give "evidence": a list of {{"chunk_id": ..., "quotation": ...}} pairs, "quotation" copied VERBATIM from the exact chunk_id shown below.
- "no": this batch's text directly supports answering the item's question negatively. {_NO_SEMANTICS_INSTRUCTION} Give non-empty "evidence" as above.
- "not_applicable": this batch's text makes clear the item does not apply to this study at all (e.g. an item about ethical approval for a study design that has no human subjects). No evidence needed.
- "not_found": this batch's text does not address the item. No evidence needed.

Items:
{items_text}

Text batch (each block is one trusted, citable chunk):
{chunks_text}

Respond with exactly one JSON object of this shape, covering EVERY item_id listed above:
{{"items": {{"<item_id>": {{"outcome": "yes"|"no"|"not_applicable"|"not_found", "rationale": "...", "evidence": [{{"chunk_id": "...", "quotation": "..."}}]}}, ...}}}}"""


def build_rob2_domain_batch_prompt(*, record_title: str, domain_spec: Dict[str, Any], batch_chunks: List[ParsedChunk]) -> str:
    questions_text = "\n".join(f'- question_id: "{q["question_id"]}"\n  question: {q["question"]}' for q in domain_spec["questions"])
    chunks_text = "\n\n".join(_chunk_block(c) for c in batch_chunks)
    return f"""You are appraising risk of bias for ONE domain ("{domain_spec['domain_id']}", RoB2-inspired) of ONE batch of a scientific paper's text ("{record_title}") for a systematic review. Answer ONLY using the text in this batch below -- never use outside knowledge.

For EACH signalling question listed, report exactly one answer using RoB2's own 5-point scale:
- "Y" (yes) / "PY" (probably yes) / "PN" (probably no) / "N" (no): this batch's text supports this answer. {_NO_SEMANTICS_INSTRUCTION.replace('"no"/"N"/"PN"', '"N"/"PN"')} Give non-empty "evidence": a list of {{"chunk_id": ..., "quotation": ...}} pairs, "quotation" copied VERBATIM from the exact chunk_id shown below.
- "NI" (no information): this batch's text does not address this question. No evidence needed.

Signalling questions:
{questions_text}

Text batch (each block is one trusted, citable chunk):
{chunks_text}

Respond with exactly one JSON object of this shape, covering EVERY question_id listed above:
{{"questions": {{"<question_id>": {{"answer": "Y"|"PY"|"PN"|"N"|"NI", "rationale": "...", "evidence": [{{"chunk_id": "...", "quotation": "..."}}]}}, ...}}}}"""


def run_openai_teaching_batch(
    *, record_title: str, batch_chunks: List[ParsedChunk], config: OpenAIExtractionConfig,
) -> RawTeachingBatchResponse:
    """The ONE call this function makes. Credentials are read lazily by
    the `openai` SDK from `OPENAI_API_KEY`. No retry, no fallback --
    `max_retries=0` is passed explicitly since the installed SDK (2.7.2)
    otherwise defaults to 2 transparent HTTP-layer retries per call,
    which would make "one attempt" false (see
    extraction/openai_runner.py's own identical fix and docstring note --
    this mirrors it exactly)."""
    from openai import OpenAI  # lazy import -- never at module import time

    prompt = build_teaching_batch_prompt(record_title=record_title, batch_chunks=batch_chunks)
    call_kwargs: Dict[str, Any] = dict(
        model=config.model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a meticulous systematic-review quality appraiser. Return only valid JSON -- no markdown, no explanation outside the JSON object."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
    )
    if config.reasoning_effort is not None:
        call_kwargs["reasoning_effort"] = config.reasoning_effort
    try:
        # Codex review (Phase A follow-up, finding 3): explicitly wired
        # through the shared resolver rather than relying on the SDK's own
        # implicit environment lookup when `api_key` is omitted -- same
        # runtime behavior, now an explicit, auditable, testable contract.
        client = OpenAI(api_key=resolve_openai_api_key(), max_retries=0)
        response = client.chat.completions.create(**call_kwargs)
        if response.model != config.expected_provider_model:
            raise QACallError(
                f"provider model mismatch: expected {config.expected_provider_model!r}, got {response.model!r}"
            )
        raw_json = json.loads(response.choices[0].message.content or "")
        return RawTeachingBatchResponse.model_validate(raw_json)
    except QACallError:
        raise
    except Exception as exc:
        raise QACallError(f"OpenAI teaching-appraisal call failed: {type(exc).__name__}") from exc


def run_openai_rob2_domain_batch(
    *, record_title: str, domain_spec: Dict[str, Any], batch_chunks: List[ParsedChunk], config: OpenAIExtractionConfig,
) -> RawDomainBatchResponse:
    from openai import OpenAI  # lazy import -- never at module import time

    prompt = build_rob2_domain_batch_prompt(record_title=record_title, domain_spec=domain_spec, batch_chunks=batch_chunks)
    call_kwargs: Dict[str, Any] = dict(
        model=config.model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a meticulous systematic-review risk-of-bias appraiser. Return only valid JSON -- no markdown, no explanation outside the JSON object."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
    )
    if config.reasoning_effort is not None:
        call_kwargs["reasoning_effort"] = config.reasoning_effort
    try:
        # Codex review (Phase A follow-up, finding 3): explicitly wired
        # through the shared resolver rather than relying on the SDK's own
        # implicit environment lookup when `api_key` is omitted -- same
        # runtime behavior, now an explicit, auditable, testable contract.
        client = OpenAI(api_key=resolve_openai_api_key(), max_retries=0)
        response = client.chat.completions.create(**call_kwargs)
        if response.model != config.expected_provider_model:
            raise QACallError(
                f"provider model mismatch: expected {config.expected_provider_model!r}, got {response.model!r}"
            )
        raw_json = json.loads(response.choices[0].message.content or "")
        return RawDomainBatchResponse.model_validate(raw_json)
    except QACallError:
        raise
    except Exception as exc:
        raise QACallError(f"OpenAI RoB2-domain-appraisal call failed: {type(exc).__name__}") from exc


# ---------------------------------------------------------------------------
# Deterministic cross-batch aggregation (Section 1.4) -- a separate,
# non-LLM step. No majority vote, no last-batch-wins.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Contribution:
    outcome: Literal["verified_finding", "verification_failed", "call_failed", "no_finding"]
    status: Optional[str] = None  # set only for verified_finding
    evidence: Tuple[VerifiedEvidence, ...] = ()
    rationale: str = ""
    chunk_ids: Tuple[str, ...] = ()  # every chunk_id sent to the model in this batch


def _verify_claimed_evidence(
    *, claimed: List[RawAppraisalEvidenceItem], chunk_map: Dict[str, ParsedChunk],
) -> List[VerifiedEvidence]:
    verified: List[VerifiedEvidence] = []
    seen_keys: set = set()
    for item in claimed:
        try:
            verified.append(verify_evidence(chunk_id=item.chunk_id, quotation=item.quotation, chunk_map=chunk_map, seen_keys=seen_keys))
        except EvidenceVerificationError:
            continue
    return verified


def _aggregate_leaf(contributions: List[_Contribution], *, total_batches: int) -> Dict[str, Any]:
    """Returns a dict of the fields common to `AppraisalItemResponse`/
    `SignallingQuestionResponse`: `status_or_answer` (None iff no
    definitive value could be reached at all), `evidence`, `rationale`,
    `decision_origin`, `evidence_status`, `chunks_examined`,
    `batches_examined`, `coverage_complete`, `queue_reason` (None iff this
    leaf needs no human action at all), and `conflicting_evidence`.

    Finding 3 fix: a verified, unanimous finding is no longer finalized
    silently when OTHER batches for the SAME leaf failed the call or
    evidence verification -- `coverage_complete=False` in that case, and
    `queue_reason="incomplete_coverage"` is set ALONGSIDE the real,
    verified `status_or_answer` (never discarding a genuine finding just
    because coverage was partial elsewhere; a human resolving this entry
    sees the actual machine answer, not a fabricated "unclear")."""
    # Round-4 "smaller corrections" fix: document order (batches, and each
    # batch's own chunk_ids, are already in document order -- Section 1.4's
    # own "every chunk_id actually sent to the model for this item, in
    # order" requirement), not alphabetically sorted -- `dict.fromkeys`
    # dedups while preserving first-seen order.
    chunks_examined = list(dict.fromkeys(cid for c in contributions for cid in c.chunk_ids))
    verified_findings = [c for c in contributions if c.outcome == "verified_finding"]
    verification_failures = [c for c in contributions if c.outcome == "verification_failed"]
    call_failures = [c for c in contributions if c.outcome == "call_failed"]
    coverage_complete = not verification_failures and not call_failures

    base = {"chunks_examined": chunks_examined, "batches_examined": total_batches, "conflicting_evidence": None}

    if verified_findings:
        distinct_statuses = sorted({c.status for c in verified_findings})
        if len(distinct_statuses) == 1:
            status = distinct_statuses[0]
            merged_evidence = [
                AppraisalEvidence(chunk_id=e.chunk_id, quotation=e.excerpt, locator=e.page or e.section)
                for c in verified_findings for e in c.evidence
            ]
            rationales = [c.rationale for c in verified_findings if c.rationale]
            rationale = " | ".join(dict.fromkeys(rationales)) if rationales else f"Consistent across {len(verified_findings)} batch(es)."
            queue_reason = None if coverage_complete else "incomplete_coverage"
            if not coverage_complete:
                rationale += " [Some batches for this leaf failed the call or evidence verification -- coverage is incomplete; see queue_reason.]"
            if status == "not_applicable":
                return {**base, "status_or_answer": status, "evidence": [], "rationale": rationale, "decision_origin": "ai", "evidence_status": "not_applicable", "coverage_complete": coverage_complete, "queue_reason": queue_reason}
            return {**base, "status_or_answer": status, "evidence": merged_evidence, "rationale": rationale, "decision_origin": "ai", "evidence_status": "verified_from_chunk", "coverage_complete": coverage_complete, "queue_reason": queue_reason}
        # Genuine conflict across batches -- never auto-resolved.
        conflicting_evidence = {
            status: [
                {"chunk_id": e.chunk_id, "quotation": e.excerpt, "locator": e.page or e.section}
                for c in verified_findings if c.status == status for e in c.evidence
            ]
            for status in distinct_statuses
        }
        return {
            **base, "status_or_answer": None, "evidence": [],
            "rationale": f"Conflicting evidence across batches: {distinct_statuses}.",
            "decision_origin": "ai", "evidence_status": "not_provided", "coverage_complete": False,
            "queue_reason": "conflicting_evidence_across_batches", "conflicting_evidence": conflicting_evidence,
        }

    if verification_failures:
        return {
            **base, "status_or_answer": None, "evidence": [],
            "rationale": "The model claimed evidence for this leaf, but it could not be verified against the parsed document text.",
            "decision_origin": "ai", "evidence_status": "not_provided", "coverage_complete": False, "queue_reason": "evidence_verification_failed",
        }

    if call_failures:
        return {
            **base, "status_or_answer": None, "evidence": [],
            "rationale": "One or more batch calls failed -- this leaf was not fully evaluated.",
            "decision_origin": "ai", "evidence_status": "not_provided", "coverage_complete": False, "queue_reason": "batch_call_failed",
        }

    return {
        **base, "status_or_answer": None, "evidence": [],
        "rationale": "No batch found any evidence addressing this leaf.",
        "decision_origin": "ai", "evidence_status": "not_provided", "coverage_complete": True, "queue_reason": "no_evidence_found",
    }


_NEGATIVE_TEACHING_STATUSES = {"no"}
_NEGATIVE_ROB2_ANSWERS = {"N", "PN"}


def _finalize_teaching_item(item_spec: Dict[str, str], contributions: List[_Contribution], *, total_batches: int) -> AppraisalItemResponse:
    agg = _aggregate_leaf(contributions, total_batches=total_batches)
    item_id = item_spec["item_id"]
    status = agg["status_or_answer"] if agg["status_or_answer"] is not None else "unclear"
    provisional = status in _NEGATIVE_TEACHING_STATUSES if agg["status_or_answer"] is not None else False
    return AppraisalItemResponse(
        item_id=item_id, status=status, evidence=agg["evidence"], rationale=agg["rationale"],
        decision_origin=agg["decision_origin"], evidence_status=agg["evidence_status"],
        chunks_examined=agg["chunks_examined"], batches_examined=agg["batches_examined"],
        provisional_pending_human_confirmation=provisional,
        coverage_complete=agg["coverage_complete"], queue_reason=agg["queue_reason"],
        conflicting_evidence=agg["conflicting_evidence"],
    )


def _finalize_signalling_question(q_spec: Dict[str, str], contributions: List[_Contribution], *, total_batches: int) -> SignallingQuestionResponse:
    agg = _aggregate_leaf(contributions, total_batches=total_batches)
    question_id = q_spec["question_id"]
    answer = agg["status_or_answer"] if agg["status_or_answer"] is not None else "NI"
    provisional = answer in _NEGATIVE_ROB2_ANSWERS if agg["status_or_answer"] is not None else False
    return SignallingQuestionResponse(
        question_id=question_id, answer=answer, evidence=agg["evidence"], rationale=agg["rationale"],
        decision_origin=agg["decision_origin"], evidence_status=agg["evidence_status"],
        chunks_examined=agg["chunks_examined"], batches_examined=agg["batches_examined"],
        provisional_pending_human_confirmation=provisional,
        coverage_complete=agg["coverage_complete"], queue_reason=agg["queue_reason"],
        conflicting_evidence=agg["conflicting_evidence"],
    )


def _domain_rationale(questions: List[SignallingQuestionResponse], domain_risk: str) -> str:
    parts = [f"{q.question_id}={q.answer}" for q in questions]
    return f"Rolled up (max_severity_v1) from signalling questions [{', '.join(parts)}] to domain_risk={domain_risk!r}."


def _overall_rationale(domains: List[DomainJudgment], overall_risk: str) -> str:
    parts = [f"{d.domain_id}={d.domain_risk}" for d in domains]
    return f"Rolled up (max_domain_severity_v1) from domains [{', '.join(parts)}] to overall_risk={overall_risk!r}."


def _appraise_document(
    *, record: Record, model_config: OpenAIExtractionConfig, checklist_id: str,
    parser_options: DoclingParserOptions, max_batch_chars: int,
) -> DocumentAppraisal:
    """Parses, batches, appraises, and aggregates ONE document. `pdf_sha256`
    is NOT computed or returned here -- the caller (`quality_appraisal_node`)
    computes it via a cheap, direct file hash BEFORE calling this function
    at all, so `qa_run_id` (and therefore whether a checkpoint already
    exists to resume from) never depends on running this expensive,
    LLM-calling function first (Finding 2/4a fix). A parse failure
    (Section 1.1 bug 1: never silently treated as passing) still produces
    a fully-populated DocumentAppraisal where EVERY leaf is unclear/NI
    with `queue_reason="batch_call_failed"` -- routed to review
    identically to any other unresolved leaf, rather than a special-cased
    document-level sentinel. Review-queue entries are NOT built here any
    more -- they are derived uniformly from the returned DocumentAppraisal's
    own persisted `queue_reason`/`provisional_pending_human_confirmation`
    leaf fields by `_build_review_queue_and_ledger()`, the SAME function
    every recovery path (fresh appraisal, resolve, confirm) now uses."""
    try:
        parsed = parse_pdf(record.full_text_path, options=parser_options)
        batches = build_batches(parsed.chunks, max_batch_chars=max_batch_chars)
        chunk_map = parsed.chunk_map()
        parse_failed = False
    except DocumentParseError as exc:
        logger.warning(f"quality_appraisal: parse failed for record {record.id}: {exc}")
        batches = ()
        chunk_map = {}
        parse_failed = True

    if checklist_id == "teaching_appraisal_v1":
        per_item_contributions: Dict[str, List[_Contribution]] = {i["item_id"]: [] for i in TEACHING_APPRAISAL_ITEMS}
        for batch in ((None,) if parse_failed else batches):
            if parse_failed:
                for item_id in per_item_contributions:
                    per_item_contributions[item_id].append(_Contribution(outcome="call_failed", chunk_ids=()))
                continue
            batch_chunks = [chunk_map[cid] for cid in batch.chunk_ids]
            try:
                raw = run_openai_teaching_batch(record_title=record.title, batch_chunks=batch_chunks, config=model_config)
                call_failed = False
            except QACallError:
                raw = None
                call_failed = True
            for item_id in per_item_contributions:
                if call_failed:
                    per_item_contributions[item_id].append(_Contribution(outcome="call_failed", chunk_ids=batch.chunk_ids))
                    continue
                raw_item = raw.items.get(item_id)
                if raw_item is None:
                    # Round-3 finding 2 fix: a MISSING item_id key is a
                    # malformed/incomplete response for this batch (the
                    # model was asked to cover every item_id and did not),
                    # never treated the same as an explicit "not_found" --
                    # conflating the two would let a schema-violating batch
                    # silently masquerade as "genuinely checked, nothing
                    # there" instead of counting toward incomplete coverage.
                    per_item_contributions[item_id].append(_Contribution(outcome="call_failed", chunk_ids=batch.chunk_ids))
                elif raw_item.outcome == "not_found":
                    per_item_contributions[item_id].append(_Contribution(outcome="no_finding", chunk_ids=batch.chunk_ids))
                elif raw_item.outcome == "not_applicable":
                    per_item_contributions[item_id].append(_Contribution(outcome="verified_finding", status="not_applicable", evidence=(), rationale=raw_item.rationale or "", chunk_ids=batch.chunk_ids))
                else:
                    verified = _verify_claimed_evidence(claimed=raw_item.evidence, chunk_map=chunk_map)
                    if not raw_item.evidence or not verified:
                        per_item_contributions[item_id].append(_Contribution(outcome="verification_failed", chunk_ids=batch.chunk_ids))
                    else:
                        per_item_contributions[item_id].append(_Contribution(outcome="verified_finding", status=raw_item.outcome, evidence=tuple(verified), rationale=raw_item.rationale or "", chunk_ids=batch.chunk_ids))

        items: List[AppraisalItemResponse] = [
            _finalize_teaching_item(item_spec, per_item_contributions[item_spec["item_id"]], total_batches=(1 if parse_failed else len(batches)))
            for item_spec in TEACHING_APPRAISAL_ITEMS
        ]
        return DocumentAppraisal(record_id=record.id, checklist_id=checklist_id, teaching=TeachingAppraisal(items=items))

    # rob2_inspired_study_level_teaching_v1
    domains: List[DomainJudgment] = []
    total_batches = 1 if parse_failed else len(batches)
    for domain_spec in ROB2_DOMAINS:
        domain_id = domain_spec["domain_id"]
        per_q_contributions: Dict[str, List[_Contribution]] = {q["question_id"]: [] for q in domain_spec["questions"]}
        for batch in ((None,) if parse_failed else batches):
            if parse_failed:
                for qid in per_q_contributions:
                    per_q_contributions[qid].append(_Contribution(outcome="call_failed", chunk_ids=()))
                continue
            batch_chunks = [chunk_map[cid] for cid in batch.chunk_ids]
            try:
                raw = run_openai_rob2_domain_batch(record_title=record.title, domain_spec=domain_spec, batch_chunks=batch_chunks, config=model_config)
                call_failed = False
            except QACallError:
                raw = None
                call_failed = True
            for qid in per_q_contributions:
                if call_failed:
                    per_q_contributions[qid].append(_Contribution(outcome="call_failed", chunk_ids=batch.chunk_ids))
                    continue
                raw_q = raw.questions.get(qid)
                if raw_q is None:
                    # Round-3 finding 2 fix -- same reasoning as the
                    # teaching-rubric branch above: a missing question_id
                    # key is a malformed response, not a legitimate "NI".
                    per_q_contributions[qid].append(_Contribution(outcome="call_failed", chunk_ids=batch.chunk_ids))
                elif raw_q.answer == "NI":
                    per_q_contributions[qid].append(_Contribution(outcome="no_finding", chunk_ids=batch.chunk_ids))
                else:
                    verified = _verify_claimed_evidence(claimed=raw_q.evidence, chunk_map=chunk_map)
                    if not raw_q.evidence or not verified:
                        per_q_contributions[qid].append(_Contribution(outcome="verification_failed", chunk_ids=batch.chunk_ids))
                    else:
                        per_q_contributions[qid].append(_Contribution(outcome="verified_finding", status=raw_q.answer, evidence=tuple(verified), rationale=raw_q.rationale or "", chunk_ids=batch.chunk_ids))

        questions: List[SignallingQuestionResponse] = [
            _finalize_signalling_question(q_spec, per_q_contributions[q_spec["question_id"]], total_batches=total_batches)
            for q_spec in domain_spec["questions"]
        ]

        domain_risk, basis = roll_up_domain_risk(questions)
        domains.append(DomainJudgment(
            domain_id=domain_id, signalling_questions=questions, domain_risk=domain_risk,
            domain_risk_rationale=_domain_rationale(questions, domain_risk), basis=basis,
        ))

    overall_risk = roll_up_overall_risk(domains)
    rob2 = RoB2InspiredAppraisal(domains=domains, overall_risk=overall_risk, overall_risk_rationale=_overall_rationale(domains, overall_risk))
    return DocumentAppraisal(record_id=record.id, checklist_id=checklist_id, rob2=rob2)


# ---------------------------------------------------------------------------
# Explicit config loading -- no silently-chosen model (mirrors
# nodes/extraction.py's own discipline).
# ---------------------------------------------------------------------------

def _load_qa_model_config(cfg: ReviewConfig) -> OpenAIExtractionConfig:
    if not cfg.quality_appraisal_model_config:
        raise ValueError(
            "ReviewConfig.quality_appraisal_model_config is not set, but full_text_records is non-empty "
            "and quality_appraisal_node was invoked. An explicit OpenAIExtractionConfig "
            "(dataclasses.asdict(...)) must be authored explicitly before quality appraisal can run."
        )
    return OpenAIExtractionConfig(**cfg.quality_appraisal_model_config)


def _load_qa_parser_options(cfg: ReviewConfig) -> DoclingParserOptions:
    if cfg.quality_appraisal_parser_options is None:
        return DoclingParserOptions()
    return DoclingParserOptions(**cfg.quality_appraisal_parser_options)


def compute_qa_run_id(
    *, documents: List[Tuple[str, str]], checklist_id: str, checklist_version: str,
    checklist_content_hash: str, roll_up_method_ids: List[str], prompt_template_version: str,
    model: str, model_config_id: str, reasoning_effort: Optional[str], max_completion_tokens: int,
    expected_provider_model: str, parser_name: str, parser_version: str, parser_options: DoclingParserOptions,
    max_batch_chars: int, evidence_verifier_version: str, negative_high_risk_policy: str,
    citation_metadata_hash: Optional[str], artifacts_manifest_digest: Optional[str] = None,
) -> str:
    """Content-addressed run identity (Section 1.6). PDF PATH is
    deliberately excluded -- only `pdf_sha256` binds document identity.
    `negative_high_risk_policy` is included (two runs differing only in
    that policy must never collide on one qa_run_id -- exactly the
    Extraction Phase B lesson this plan cites). `artifacts_manifest_digest`
    (§5.4) is `None` in ordinary/production use (no pinned Docling
    model-artifacts directory) -- mirrors
    `extraction.run_identity.compute_run_id`'s own field of the same name."""
    payload = {
        "documents": [{"record_id": r, "pdf_sha256": h} for r, h in documents],
        "checklist_id": checklist_id, "checklist_version": checklist_version,
        "checklist_content_hash": checklist_content_hash, "roll_up_method_ids": list(roll_up_method_ids),
        "prompt_template_version": prompt_template_version,
        "model_config": {
            "model": model, "model_config_id": model_config_id, "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens, "expected_provider_model": expected_provider_model,
        },
        "parser": {
            "name": parser_name, "version": parser_version,
            "options": {"do_ocr": parser_options.do_ocr, "do_table_structure": parser_options.do_table_structure, "max_chunk_chars": parser_options.max_chunk_chars},
        },
        "artifacts_manifest_digest": artifacts_manifest_digest,
        "max_batch_chars": max_batch_chars,
        "evidence_verifier_version": evidence_verifier_version,
        "negative_high_risk_policy": negative_high_risk_policy,
        "citation_metadata_hash": citation_metadata_hash,
    }
    digest = sha256_hex(payload)
    return f"qarun-{digest[:32]}"


def _roll_up_method_ids_for(checklist_id: str) -> List[str]:
    return [] if checklist_id == "teaching_appraisal_v1" else ["max_severity_v1", "max_domain_severity_v1"]


# ---------------------------------------------------------------------------
# Durable artifact set (Section 1.6) -- own independent manifest +
# checkpoints + resolutions/confirmations + lock. Never
# ExtractionCheckpointStore/extraction_resolutions.json (Section 4).
# ---------------------------------------------------------------------------

class CheckpointIntegrityError(Exception):
    """A manifest/checkpoint/resolution/confirmation artifact exists but
    cannot be trusted: its content hash does not match its own contents,
    or it claims an identity that does not match the store loading it."""


def _leaf_value(leaf: Any) -> str:
    return leaf.status if hasattr(leaf, "status") else leaf.answer


def _leaf_is_negative(leaf: Any) -> bool:
    value = _leaf_value(leaf)
    return value == "no" or value in ("N", "PN")


# Round-6 finding 1 fix: the EXACT producer matrix `_aggregate_leaf()`
# (below) actually generates -- one entry per real `queue_reason` outcome
# (plus `None`, the "no queue entry" case), each specifying whether the
# leaf's own status/answer must be RESOLVED (definitive or not_applicable
# -- a real, verified finding exists) or UNRESOLVED (unclear/NI -- no
# trustworthy finding), the exact `coverage_complete` value that reason is
# always paired with, and whether `conflicting_evidence` must be set.
# `QAAppraisalCheckpoint._machine_layer_invariants()` is the ONE place
# this is checked; nowhere else in this module hand-rolls the same rules
# a second time.
_QUEUE_REASON_MATRIX: Dict[Optional[str], Dict[str, bool]] = {
    None: {"resolved": True, "coverage_complete": True, "conflicting_evidence": False},
    "incomplete_coverage": {"resolved": True, "coverage_complete": False, "conflicting_evidence": False},
    "conflicting_evidence_across_batches": {"resolved": False, "coverage_complete": False, "conflicting_evidence": True},
    "evidence_verification_failed": {"resolved": False, "coverage_complete": False, "conflicting_evidence": False},
    "batch_call_failed": {"resolved": False, "coverage_complete": False, "conflicting_evidence": False},
    "no_evidence_found": {"resolved": False, "coverage_complete": True, "conflicting_evidence": False},
}


class QAAppraisalCheckpoint(BaseModel):
    """The complete, self-describing, content-hashed envelope for one
    document's checkpointed appraisal result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qa_run_id: str
    record_id: str
    pdf_sha256: str
    checklist_id: str
    checklist_version: str
    document_appraisal: DocumentAppraisal
    content_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _inner_model_matches_envelope(self) -> "QAAppraisalCheckpoint":
        """Round-3 finding 3 fix: mirrors extraction/checkpoint.py's own
        `DocumentCheckpoint._document_id_matches_result` -- an envelope
        whose own `record_id`/`checklist_id` disagree with its own
        `document_appraisal.record_id`/`.checklist_id` is internally
        incoherent, regardless of external run/document context, and must
        be unconstructable rather than merely caught later by a separate,
        skippable check. Without this, a self-hashed checkpoint bound
        (outer fields) to record_id="rec-a" could carry an inner
        DocumentAppraisal for an entirely different record and pass every
        other verification this class performs."""
        if self.document_appraisal.record_id != self.record_id:
            raise ValueError(
                f"checkpoint record_id={self.record_id!r} does not match "
                f"document_appraisal.record_id={self.document_appraisal.record_id!r}"
            )
        if self.document_appraisal.checklist_id != self.checklist_id:
            raise ValueError(
                f"checkpoint checklist_id={self.checklist_id!r} does not match "
                f"document_appraisal.checklist_id={self.document_appraisal.checklist_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _machine_layer_invariants(self) -> "QAAppraisalCheckpoint":
        """Round-5 finding 3 fix: THIS class holds the FROZEN, MACHINE-ONLY
        layer -- `decision_origin="human"`/`human_confirmation` are
        properties of the RESOLUTION/CONFIRMATION layer
        (qa_resolutions.json/qa_confirmations.json), applied only at
        projection time (`_project_document_appraisal`), never baked into
        a checkpoint at construction time. `AppraisalItemResponse`/
        `SignallingQuestionResponse` themselves cannot enforce this,
        because those SAME models legitimately represent BOTH a frozen
        machine leaf and a resolved/confirmed PROJECTED one -- so these
        invariants live here, on the checkpoint envelope specifically,
        never on the shared leaf models.

        Enforces, for EVERY leaf in `document_appraisal`:
          1. `decision_origin == "ai"` -- a checkpoint never carries a
             human-authored leaf.
          2. `human_confirmation is None` -- a checkpoint never carries a
             confirmation record.
          3. `queue_reason` is EXACTLY one of `_aggregate_leaf()`'s own six
             real outcomes (`None` plus the five string reasons it
             actually produces) -- round-6 fix: previously any string
             (e.g. a fabricated `"made_up_reason"`) was accepted as long
             as SOME `queue_reason` was present, which only caught a
             MISSING reason, never an invalid one.
          4. For each of those six outcomes, the EXACT producer matrix
             `_QUEUE_REASON_MATRIX` (round-6 fix, replacing three
             previously-independent, incomplete checks) cross-validates:
             whether the leaf's own status/answer must be resolved
             (definitive or not_applicable -- a real verified finding
             exists) or unresolved (unclear/NI -- no finding, or one that
             could not be trusted), the exact `coverage_complete` value
             `_aggregate_leaf()` always pairs with that reason, and
             whether `conflicting_evidence` must be set. Closes the
             round-5 review's own producer-impossible counterexamples: a
             definitive "yes"/"no" with an invalid queue_reason like
             `"made_up_reason"`; an "unclear" leaf claiming
             `queue_reason="incomplete_coverage"` (which requires a
             RESOLVED leaf) while ALSO claiming `coverage_complete=True`
             (which that reason always pairs with `False`).
          5. `provisional_pending_human_confirmation` is EXACTLY `True`
             iff the leaf's own status/answer is a DEFINITIVE negative
             machine finding (`"no"`, `"N"`, or `"PN"`) -- round-6 fix:
             the existing leaf-level validators (`AppraisalItemResponse`/
             `SignallingQuestionResponse`) only ever checked the
             "negative implies provisional" direction; nothing prevented
             a definitive "yes"/"Y"/"PY" (or `not_applicable`, or an
             unresolved unclear/NI) leaf from ALSO claiming
             `provisional_pending_human_confirmation=True`, which would
             incorrectly enter the confirmation queue for a leaf that was
             never actually negative."""
        for field_id, leaf in self.document_appraisal.leaves().items():
            if leaf.decision_origin != "ai":
                raise ValueError(f"checkpoint leaf {field_id!r}: a frozen machine checkpoint must have decision_origin='ai' for every leaf, got {leaf.decision_origin!r}")
            if leaf.human_confirmation is not None:
                raise ValueError(f"checkpoint leaf {field_id!r}: a frozen machine checkpoint must never carry a human_confirmation")

            if leaf.queue_reason not in _QUEUE_REASON_MATRIX:
                raise ValueError(
                    f"checkpoint leaf {field_id!r}: queue_reason={leaf.queue_reason!r} is not one of the "
                    f"aggregation's own real outcomes {sorted(r for r in _QUEUE_REASON_MATRIX if r is not None)!r}"
                )
            expected = _QUEUE_REASON_MATRIX[leaf.queue_reason]

            is_definitive = _leaf_is_definitive(leaf)
            is_not_applicable = _leaf_is_not_applicable(leaf)
            is_resolved = is_definitive or is_not_applicable
            if is_resolved != expected["resolved"]:
                raise ValueError(
                    f"checkpoint leaf {field_id!r}: queue_reason={leaf.queue_reason!r} requires the leaf's own "
                    f"status/answer to be {'definitive or not_applicable' if expected['resolved'] else 'unclear/NI'}, "
                    f"but it is {'definitive/not_applicable' if is_resolved else 'unclear/NI'}"
                )
            if leaf.coverage_complete != expected["coverage_complete"]:
                raise ValueError(
                    f"checkpoint leaf {field_id!r}: queue_reason={leaf.queue_reason!r} requires "
                    f"coverage_complete={expected['coverage_complete']!r}, got {leaf.coverage_complete!r}"
                )
            has_conflicting_evidence = leaf.conflicting_evidence is not None
            if has_conflicting_evidence != expected["conflicting_evidence"]:
                raise ValueError(
                    f"checkpoint leaf {field_id!r}: queue_reason={leaf.queue_reason!r} requires conflicting_evidence "
                    f"{'to be set' if expected['conflicting_evidence'] else 'to be None'}"
                )

            expected_provisional = is_definitive and _leaf_is_negative(leaf)
            if leaf.provisional_pending_human_confirmation != expected_provisional:
                raise ValueError(
                    f"checkpoint leaf {field_id!r}: provisional_pending_human_confirmation="
                    f"{leaf.provisional_pending_human_confirmation!r} does not match the expected "
                    f"{expected_provisional!r} for this leaf's own status/answer (provisional is True if and "
                    "only if a DEFINITIVE negative machine finding)"
                )
        return self

    @classmethod
    def build(cls, *, qa_run_id: str, record_id: str, pdf_sha256: str, checklist_id: str, checklist_version: str, document_appraisal: DocumentAppraisal) -> "QAAppraisalCheckpoint":
        envelope = dict(
            qa_run_id=qa_run_id, record_id=record_id, pdf_sha256=pdf_sha256, checklist_id=checklist_id,
            checklist_version=checklist_version, document_appraisal=document_appraisal.model_dump(mode="json"),
        )
        content_hash = sha256_hex(envelope)
        return cls(**envelope, content_hash=content_hash)

    def _recompute_content_hash(self) -> str:
        return sha256_hex(self.model_dump(mode="json", exclude={"content_hash"}))

    def verify(self, *, expected_qa_run_id: Optional[str] = None, expected_record_id: Optional[str] = None) -> None:
        if expected_qa_run_id is not None and self.qa_run_id != expected_qa_run_id:
            raise CheckpointIntegrityError(f"checkpoint qa_run_id mismatch: loaded from a store for {expected_qa_run_id!r} but claims {self.qa_run_id!r}")
        if expected_record_id is not None and self.record_id != expected_record_id:
            raise CheckpointIntegrityError(f"checkpoint record_id mismatch: loaded for {expected_record_id!r} but claims {self.record_id!r}")
        if self._recompute_content_hash() != self.content_hash:
            raise CheckpointIntegrityError(f"checkpoint content_hash mismatch for record_id={self.record_id!r} -- the file may be tampered or corrupted")


class QAManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    qa_run_id: str
    checklist_id: str
    checklist_version: str
    checklist_content_hash: str
    roll_up_method_ids: List[str]
    prompt_template_version: str
    model: str
    model_config_id: str
    reasoning_effort: Optional[str]
    max_completion_tokens: int
    expected_provider_model: str
    parser_name: str
    parser_version: str
    parser_options: Dict[str, Any]
    # §5.4: `None` in ordinary/production use (no pinned model-artifacts
    # directory) -- mirrors `extraction.checkpoint.RunManifest`'s own
    # field of the same name.
    artifacts_manifest_digest: Optional[str] = None
    max_batch_chars: int
    evidence_verifier_version: str
    negative_high_risk_policy: str
    # Finding 4d fix: persisted (and part of the manifest's own self-hash,
    # so tamper-evident) precisely so `qa_run_id` can be recomputed on
    # load from this manifest's own OTHER fields + each verified
    # checkpoint's own (record_id, pdf_sha256) -- citation metadata
    # (record titles) is run-level, not persisted on any checkpoint, so
    # it cannot be re-derived from checkpoints alone the way the
    # documents list can.
    citation_metadata_hash: Optional[str] = None
    document_ids: List[str]
    checkpoint_hashes: Dict[str, str]
    # Round-4 "smaller corrections" fix: Section 1.6 states "never the PDF
    # path, which is stored in qa_manifest.json as provenance only" -- this
    # field was named in the plan but never actually added. Provenance
    # only, like the other fields in this block: NEVER part of qa_run_id's
    # own hash (only `pdf_sha256`, bound per-document into `qa_run_id`
    # itself via `compute_qa_run_id()`'s own `documents` list, does that).
    pdf_paths: Dict[str, str] = Field(default_factory=dict)
    # Provenance-only fields -- NEVER part of qa_run_id's own hash.
    extraction_run_id: Optional[str] = None
    extraction_manifest_hash: Optional[str] = None
    protocol_uri: Optional[str] = None
    protocol_hash: Optional[str] = None
    protocol_registration_id: Optional[str] = None
    ai_use_prespecified: Optional[bool] = None
    protocol_deviations: List[str] = Field(default_factory=list)
    created_at: str
    manifest_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _document_ids_are_unique(self) -> "QAManifest":
        """Finding 4b/4d fix: mirrors `RunManifest._document_ids_are_unique`
        (extraction/checkpoint.py) exactly -- a set-based coverage check
        alone lets a duplicate record_id collapse invisibly (two identical
        entries collapse to the same set membership as one)."""
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError(f"document_ids must not contain duplicates, got {self.document_ids!r}")
        return self

    @model_validator(mode="after")
    def _checkpoint_hashes_match_document_ids(self) -> "QAManifest":
        if set(self.checkpoint_hashes) != set(self.document_ids):
            raise ValueError("checkpoint_hashes must have exactly one entry per document_id")
        return self

    def _recompute_manifest_hash(self) -> str:
        return sha256_hex(self.model_dump(mode="json", exclude={"manifest_hash"}))

    def verify_manifest_hash(self) -> None:
        if self._recompute_manifest_hash() != self.manifest_hash:
            raise CheckpointIntegrityError(f"manifest_hash mismatch for qa_run_id={self.qa_run_id!r} -- the file may be tampered or corrupted")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Round-4 finding 4 fix: the temp filename is now unique per call
    (PID + a fresh UUID4), not a fixed `<name>.tmp` -- two concurrent
    writers targeting the SAME final `path` previously raced on that
    SAME temp path too, risking one process's partial write being
    clobbered mid-write by the other before either reached its own
    `.replace()`. `Path.replace()` itself is still the atomic publish
    step (POSIX `rename`/Windows `MoveFileEx` with `MOVEFILE_REPLACE_EXISTING`),
    but callers that need "first writer wins, second detects and
    reconciles" (rather than "last writer silently wins") must still
    hold `_qa_resolutions_lock` around the whole read-check-write
    sequence -- this function alone only prevents temp-file corruption,
    it does not by itself prevent two racing publishers from each
    observing `path` as absent and both proceeding to replace it."""
    import os
    import uuid
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


_SAFE_QA_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class QACheckpointStore:
    """Own independent artifact layout, `output_dir/quality_appraisal_runs/
    <qa_run_id>/` -- structurally parallel to `ExtractionCheckpointStore`
    but never sharing its class/lock/files (Section 4)."""

    def __init__(self, *, output_dir: Path, qa_run_id: str) -> None:
        """Finding 4c fix: `qa_run_id` is always internally computed
        (`compute_qa_run_id()` -> `f"qarun-{digest[:32]}"`) when this store
        is constructed from a fresh `quality_appraisal_node` run, but every
        OTHER caller (`load_verified_quality_appraisal_resolution_artifact`,
        `resolve_quality_appraisal_review_queue`,
        `confirm_quality_appraisal_provisional_item`) constructs it from
        `state["quality_appraisal_run_id"]` -- mutable, LangGraph-checkpointed
        state that must never be trusted as an already-safe path component
        (same reasoning as `ExtractionCheckpointStore`'s own `run_id`
        validation)."""
        runs_root = Path(output_dir) / "quality_appraisal_runs"
        if not _SAFE_QA_RUN_ID_RE.match(qa_run_id):
            raise ValueError(
                f"qa_run_id {qa_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        self.qa_run_id = qa_run_id
        self.run_dir = runs_root / qa_run_id
        if not self.run_dir.resolve().is_relative_to(runs_root.resolve()):
            raise ValueError(f"qa_run_id {qa_run_id!r} resolves outside the quality_appraisal_runs root -- refusing to construct store")
        self.checkpoints_dir = self.run_dir / "checkpoints"

    def manifest_path(self) -> Path:
        return self.run_dir / "qa_manifest.json"

    def resolutions_path(self) -> Path:
        return self.run_dir / "qa_resolutions.json"

    def confirmations_path(self) -> Path:
        return self.run_dir / "qa_confirmations.json"

    def lock_path(self) -> Path:
        return self.run_dir / "qa_resolutions.lock"

    def checkpoint_path(self, record_id: str) -> Path:
        return self.checkpoints_dir / f"{sha256_hex(record_id)[:32]}.json"

    def write_checkpoint(self, checkpoint: QAAppraisalCheckpoint) -> QAAppraisalCheckpoint:
        """No-op-if-identical, raise-if-different (Finding 2 fix) -- a
        published checkpoint is frozen/immutable, mirroring
        `ExtractionCheckpointStore.publish_manifest()`'s own discipline.
        In the normal path, `quality_appraisal_node` never calls this for
        a record whose checkpoint already exists (it reuses the existing
        one instead -- see that function's own resumability logic), so
        this is defense in depth: it should only ever observe a genuine
        first write or a true no-op republish, never a real difference."""
        checkpoint.verify(expected_qa_run_id=self.qa_run_id, expected_record_id=checkpoint.record_id)
        path = self.checkpoint_path(checkpoint.record_id)
        if path.exists():
            existing = QAAppraisalCheckpoint.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify(expected_qa_run_id=self.qa_run_id, expected_record_id=checkpoint.record_id)
            if existing.content_hash == checkpoint.content_hash:
                return existing
            raise CheckpointIntegrityError(
                f"refusing to overwrite the already-published checkpoint for record_id={checkpoint.record_id!r} "
                f"under qa_run_id={self.qa_run_id!r} with different content -- a published checkpoint is immutable"
            )
        _atomic_write_json(path, checkpoint.model_dump(mode="json"))
        return checkpoint

    def load_checkpoint(self, record_id: str) -> QAAppraisalCheckpoint:
        """Round-5 finding 4 fix: a MISSING checkpoint file (e.g. a manifest-
        bound checkpoint deleted after publish) now raises the same
        `CheckpointIntegrityError` every other integrity failure in this
        module raises, rather than a bare `FileNotFoundError` -- both fail
        immediately either way, but a consistent exception type keeps every
        caller's error handling uniform."""
        path = self.checkpoint_path(record_id)
        if not path.exists():
            raise CheckpointIntegrityError(f"no checkpoint file exists for record_id={record_id!r} under qa_run_id={self.qa_run_id!r}")
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = QAAppraisalCheckpoint.model_validate(data)
        checkpoint.verify(expected_qa_run_id=self.qa_run_id, expected_record_id=record_id)
        return checkpoint

    def write_manifest(self, manifest: QAManifest) -> QAManifest:
        """No-op-if-identical, raise-if-different (round-3 finding 4a fix):
        previously this method blindly wrote whatever it was given,
        relying ENTIRELY on caller discipline (`quality_appraisal_node`'s
        own pre-check) to never call it with different content than an
        already-published manifest -- any other/future caller, or a bug
        in that pre-check, could silently overwrite a different valid
        published manifest. The safety now lives in the method itself,
        mirroring `write_checkpoint`'s own discipline exactly."""
        manifest.verify_manifest_hash()
        path = self.manifest_path()
        if path.exists():
            existing = QAManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_manifest_hash()
            if existing.model_dump(mode="json") == manifest.model_dump(mode="json"):
                return existing
            raise CheckpointIntegrityError(
                f"refusing to overwrite the already-published qa_manifest.json for qa_run_id={self.qa_run_id!r} "
                "with different content -- a published QA manifest is immutable"
            )
        _atomic_write_json(path, manifest.model_dump(mode="json"))
        return manifest

    def load_manifest(self) -> Optional[QAManifest]:
        path = self.manifest_path()
        if not path.exists():
            return None
        manifest = QAManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        manifest.verify_manifest_hash()
        if manifest.qa_run_id != self.qa_run_id:
            raise CheckpointIntegrityError(f"manifest qa_run_id mismatch: loaded from a store for {self.qa_run_id!r} but claims {manifest.qa_run_id!r}")
        return manifest


class VerifiedQARunBundle:
    """The ONE verified, store-derived source of truth for a run's
    manifest + every checkpoint it binds."""

    def __init__(self, *, manifest: QAManifest, checkpoints_by_id: Dict[str, QAAppraisalCheckpoint]) -> None:
        self.manifest = manifest
        self._checkpoints_by_id = dict(checkpoints_by_id)

    @classmethod
    def from_store(cls, store: QACheckpointStore) -> "VerifiedQARunBundle":
        manifest = store.load_manifest()
        if manifest is None:
            raise ValueError(f"no published manifest found for qa_run_id={store.qa_run_id!r}")
        checkpoints_by_id: Dict[str, QAAppraisalCheckpoint] = {}
        for record_id in manifest.document_ids:
            checkpoint = store.load_checkpoint(record_id)
            claimed_hash = manifest.checkpoint_hashes[record_id]
            if checkpoint.content_hash != claimed_hash:
                raise CheckpointIntegrityError(f"checkpoint for record_id={record_id!r} does not match the hash bound by the verified manifest")
            checkpoints_by_id[record_id] = checkpoint
        return cls(manifest=manifest, checkpoints_by_id=checkpoints_by_id)

    @property
    def document_appraisals(self) -> List[DocumentAppraisal]:
        return [self._checkpoints_by_id[d].document_appraisal for d in self.manifest.document_ids]

    def verify(self) -> None:
        self.manifest.verify_manifest_hash()
        checkpoint_ids = set(self._checkpoints_by_id)
        manifest_ids = set(self.manifest.document_ids)
        if checkpoint_ids != manifest_ids:
            raise CheckpointIntegrityError("bundle checkpoints do not exactly match manifest.document_ids")
        for record_id in self.manifest.document_ids:
            checkpoint = self._checkpoints_by_id[record_id]
            checkpoint.verify(expected_qa_run_id=self.manifest.qa_run_id, expected_record_id=record_id)
            if checkpoint.content_hash != self.manifest.checkpoint_hashes[record_id]:
                raise CheckpointIntegrityError(f"checkpoint for record_id={record_id!r} does not match the manifest's own binding")
            # Round-4 finding 3 fix: the checkpoint's OWN checklist_id/
            # checklist_version must match the manifest's -- previously
            # unchecked, so e.g. checklist_version="999" under a manifest
            # claiming "1" passed every other check here.
            if checkpoint.checklist_id != self.manifest.checklist_id or checkpoint.checklist_version != self.manifest.checklist_version:
                raise CheckpointIntegrityError(
                    f"checkpoint for record_id={record_id!r} has checklist_id={checkpoint.checklist_id!r}/"
                    f"checklist_version={checkpoint.checklist_version!r}, which does not match the manifest's own "
                    f"checklist_id={self.manifest.checklist_id!r}/checklist_version={self.manifest.checklist_version!r}"
                )

        # Finding 4d fix: `ExtractionCheckpointStore._derive_manifest()`'s
        # own strongest guarantee -- re-derive qa_run_id from the VERIFIED
        # checkpoints' own (record_id, pdf_sha256) pairs (in the manifest's
        # own document_ids order) plus the manifest's own other config
        # fields, and require it reproduces THIS store's own qa_run_id.
        # Without this, a manifest that is internally self-hash-consistent
        # but was relocated into (or copied under) the wrong run_id
        # directory -- or whose document_ids/config were tampered with in
        # a way that still hashes consistently -- would pass every OTHER
        # check above.
        documents = [(record_id, self._checkpoints_by_id[record_id].pdf_sha256) for record_id in self.manifest.document_ids]
        recomputed_qa_run_id = compute_qa_run_id(
            documents=documents, checklist_id=self.manifest.checklist_id, checklist_version=self.manifest.checklist_version,
            checklist_content_hash=self.manifest.checklist_content_hash, roll_up_method_ids=self.manifest.roll_up_method_ids,
            prompt_template_version=self.manifest.prompt_template_version, model=self.manifest.model,
            model_config_id=self.manifest.model_config_id, reasoning_effort=self.manifest.reasoning_effort,
            max_completion_tokens=self.manifest.max_completion_tokens, expected_provider_model=self.manifest.expected_provider_model,
            parser_name=self.manifest.parser_name, parser_version=self.manifest.parser_version,
            parser_options=DoclingParserOptions(**self.manifest.parser_options), max_batch_chars=self.manifest.max_batch_chars,
            evidence_verifier_version=self.manifest.evidence_verifier_version, negative_high_risk_policy=self.manifest.negative_high_risk_policy,
            citation_metadata_hash=self.manifest.citation_metadata_hash,
        )
        if recomputed_qa_run_id != self.manifest.qa_run_id:
            raise CheckpointIntegrityError(
                f"the manifest's own document_ids/config, run back through compute_qa_run_id(), reproduce "
                f"{recomputed_qa_run_id!r}, which does not match this store's own qa_run_id "
                f"{self.manifest.qa_run_id!r} -- this manifest does not describe the request its own "
                "qa_run_id claims to represent"
            )


# --- Kernel-backed lock (same discipline as nodes/extraction.py's
# `_resolution_artifact_lock` -- tied to the OS's own open-file-description
# lifetime, not file existence, so a hard-killed holder never orphans it).

class QAArtifactLockError(Exception):
    """Raised when the per-run qa_resolutions.lock cannot be acquired
    within the timeout -- another live process/thread appears to be
    resolving/confirming this run concurrently."""


class _LockBusyError(Exception):
    pass


_POSIX_LOCK_CONTENTION_ERRNOS = {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}
_WINDOWS_LOCK_CONTENTION_ERRNOS = {errno.EACCES}


def _try_acquire_os_lock(fh) -> None:
    fh.seek(0)
    if fcntl is not None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _POSIX_LOCK_CONTENTION_ERRNOS:
                raise _LockBusyError() from exc
            raise
    elif msvcrt is not None:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in _WINDOWS_LOCK_CONTENTION_ERRNOS:
                raise _LockBusyError() from exc
            raise
    else:  # pragma: no cover - no supported platform lock primitive
        raise RuntimeError("no kernel-backed file-locking primitive available on this platform")


def _release_os_lock(fh) -> None:
    fh.seek(0)
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


@contextlib.contextmanager
def _qa_resolutions_lock(store: QACheckpointStore, *, timeout: float = 30.0, poll_interval: float = 0.02):
    """Covers BOTH qa_resolutions.json and qa_confirmations.json -- one
    lock, two files, since both are part of the same logical human-review
    transaction for a run (Section 1.6)."""
    lock_path = store.lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        fh.seek(0, 2)
        if fh.tell() == 0:
            fh.write(b"\0")
            fh.flush()
    except Exception:
        fh.close()
        raise
    deadline = time.monotonic() + timeout
    locked = False
    try:
        while not locked:
            try:
                _try_acquire_os_lock(fh)
                locked = True
            except _LockBusyError:
                if time.monotonic() >= deadline:
                    raise QAArtifactLockError(
                        f"could not acquire qa_resolutions.lock for qa_run_id={store.qa_run_id!r} within {timeout}s"
                    )
                time.sleep(poll_interval)
        yield
    finally:
        if locked:
            _release_os_lock(fh)
        fh.close()


def _read_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope_content_hash(payload: Dict[str, Any]) -> str:
    return sha256_hex(payload)


def _publish_append_only_envelope(
    path: Path, *, qa_run_id: str, manifest_hash: str, entries: List[Dict[str, Any]], entry_key_fn,
    existing_verified: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Shared publish mechanism for BOTH qa_resolutions.json and
    qa_confirmations.json -- mirrors `nodes.extraction._publish_resolution_
    artifact`'s own re-verify-existing/no-op-if-identical/refuse-if-
    different discipline (reimplemented here, not imported, since QA's
    artifact is independent -- Section 4)."""
    keys = [entry_key_fn(e) for e in entries]
    duplicate_keys = sorted({k for k in keys if keys.count(k) > 1})
    if duplicate_keys:
        raise ValueError(f"refusing to publish {path.name}: duplicate key(s) {duplicate_keys!r}")

    payload = {"qa_run_id": qa_run_id, "manifest_hash": manifest_hash, "entries": entries}
    existing = existing_verified

    if existing is not None:
        existing_core = {k: v for k, v in existing.items() if k not in ("content_hash", "created_at")}
        if existing_core == payload:
            return existing
        existing_by_key = {entry_key_fn(e): e for e in existing["entries"]}
        new_by_key = {entry_key_fn(e): e for e in entries}
        for key, old in existing_by_key.items():
            new = new_by_key.get(key)
            if new is None or new != old:
                raise CheckpointIntegrityError(f"refusing to publish {path.name}: an already-published entry for {key!r} would be removed or changed -- entries are append-only")

    envelope = dict(payload, created_at=datetime.now(timezone.utc).isoformat())
    envelope["content_hash"] = _envelope_content_hash(envelope)
    _atomic_write_json(path, envelope)
    return envelope


def _read_and_verify_envelope(path: Path, *, qa_run_id: str, manifest_hash: str) -> Optional[Dict[str, Any]]:
    data = _read_json_or_none(path)
    if data is None:
        return None
    check_hash = _envelope_content_hash({k: v for k, v in data.items() if k != "content_hash"})
    if check_hash != data.get("content_hash"):
        raise CheckpointIntegrityError(f"{path.name} for qa_run_id={qa_run_id!r} is tampered/corrupted -- content_hash does not match its own contents")
    if data.get("qa_run_id") != qa_run_id:
        raise CheckpointIntegrityError(f"{path.name} claims qa_run_id={data.get('qa_run_id')!r} but was loaded from the store for {qa_run_id!r}")
    if data.get("manifest_hash") != manifest_hash:
        raise CheckpointIntegrityError(f"{path.name} for qa_run_id={qa_run_id!r} does not match the current run's verified manifest_hash -- refusing to trust a stale or foreign artifact")
    return data


def _leaf_is_definitive(leaf: Any) -> bool:
    return leaf.status in ("yes", "no") if hasattr(leaf, "status") else leaf.answer != "NI"


def _leaf_is_not_applicable(leaf: Any) -> bool:
    return getattr(leaf, "status", None) == "not_applicable"


def _canonical_queue_kind_for_leaf(leaf: Any, *, negative_high_risk_policy: str) -> Optional[str]:
    """Round-5 finding 1 fix: THE ONE place that decides what queue_kind
    (if any) a leaf would ACTUALLY have, mirroring
    `_build_review_queue_and_ledger()`'s own precedence rule exactly
    (queue_reason takes precedence over provisional; a provisional leaf is
    a `confirmation`-kind queue entry ONLY under `require_human_
    confirmation` -- under `allow_provisional_ai_only` it is NEVER a queue
    entry at all, routed to the non-blocking provisional ledger instead).
    Returns `"resolution"`, `"confirmation"`, or `None` (not eligible for
    ANY human action via the queue). Used by both `_verify_resolution_
    entry`/`_verify_confirmation_entry` so eligibility is reconstructed
    from the same policy-aware logic the queue itself uses, never a
    leaf-flag-only shortcut that ignores `negative_high_risk_policy`."""
    if leaf.queue_reason is not None:
        return "resolution"
    if leaf.provisional_pending_human_confirmation and negative_high_risk_policy == "require_human_confirmation":
        return "confirmation"
    return None


def _verify_resolution_entry(
    entry: Dict[str, Any], *, qa_run_id: str, manifest_hash: str, checklist_id: str,
    document_appraisals_by_id: Dict[str, Any], negative_high_risk_policy: str,
) -> None:
    """THE semantic re-derivation/verification step for ONE persisted
    resolution entry (round-3 finding 3 fix) -- re-checks it against the
    VERIFIED bundle's own current leaf content, never trusting it merely
    because the enclosing qa_resolutions.json envelope's own self-hash
    happens to check out. Without this, a directly-placed (forged) file
    whose OWN self-hash is internally consistent -- but whose entries
    claim a foreign run/manifest, an unresolvable (record_id, field_id),
    or an `original_leaf_response` that does not match what the frozen
    checkpoint actually contains -- would be accepted by
    `_read_and_verify_envelope`'s bare self-hash/binding check alone."""
    record_id, field_id = entry.get("record_id"), entry.get("field_id")
    key_desc = f"({record_id!r}, {field_id!r})"
    if entry.get("qa_run_id") != qa_run_id:
        raise CheckpointIntegrityError(f"resolution for {key_desc} is not bound to qa_run_id={qa_run_id!r}")
    if entry.get("manifest_hash") != manifest_hash:
        raise CheckpointIntegrityError(f"resolution for {key_desc} is not bound to this run's verified manifest_hash")
    da = document_appraisals_by_id.get(record_id)
    if da is None:
        raise CheckpointIntegrityError(f"resolution references record_id={record_id!r}, which is not part of this verified run")
    original_leaf = da.leaves().get(field_id)
    if original_leaf is None:
        raise CheckpointIntegrityError(f"resolution references field_id={field_id!r}, which is not a real leaf of record_id={record_id!r}'s verified appraisal")
    if entry.get("original_leaf_response") != original_leaf.model_dump(mode="json"):
        raise CheckpointIntegrityError(f"resolution for {key_desc}: original_leaf_response does not match the verified checkpoint's own leaf content")
    # Round-5 finding 1 fix (round-4's own version of this check only
    # looked at the leaf's own flags, ignoring `negative_high_risk_policy`
    # -- so a provisional leaf that is DELIBERATELY absent from the
    # blocking queue under `allow_provisional_ai_only` still passed this
    # check, since it only asked "is this leaf provisional at all," not
    # "would this leaf actually be a queue entry under THIS run's own
    # policy"). Eligibility is now reconstructed via the SAME canonical
    # queue-membership logic `_build_review_queue_and_ledger()` itself
    # uses -- a resolution is legitimate against ANY canonical queue
    # entry, whichever queue_kind it would be.
    if _canonical_queue_kind_for_leaf(original_leaf, negative_high_risk_policy=negative_high_risk_policy) is None:
        raise CheckpointIntegrityError(
            f"resolution for {key_desc}: the referenced leaf was never a canonical queue entry under this run's "
            f"own negative_high_risk_policy={negative_high_risk_policy!r} -- a resolution may only target a leaf "
            "that was actually eligible for one"
        )
    reviewer, note, resolved_at = _stripped(entry.get("reviewer")), _stripped(entry.get("note")), _stripped(entry.get("resolved_at"))
    if not reviewer or not note or not resolved_at:
        raise CheckpointIntegrityError(f"resolution for {key_desc} is missing a non-blank reviewer/note/resolved_at")
    try:
        datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
    except ValueError:
        raise CheckpointIntegrityError(f"resolution for {key_desc}: resolved_at {resolved_at!r} is not a valid ISO-8601 timestamp")
    try:
        _validate_human_resolution_value(field_id, checklist_id, entry.get("value"), list(entry.get("evidence") or []))
    except ValueError as exc:
        raise CheckpointIntegrityError(f"resolution for {key_desc}: {exc}") from exc


def _verify_confirmation_entry(
    entry: Dict[str, Any], *, qa_run_id: str, manifest_hash: str,
    document_appraisals_by_id: Dict[str, Any], negative_high_risk_policy: str,
) -> None:
    """Semantic verification for ONE persisted confirmation entry --
    same reasoning as `_verify_resolution_entry` (round-3 finding 3)."""
    record_id, field_id = entry.get("record_id"), entry.get("field_id")
    key_desc = f"({record_id!r}, {field_id!r})"
    if entry.get("qa_run_id") != qa_run_id:
        raise CheckpointIntegrityError(f"confirmation for {key_desc} is not bound to qa_run_id={qa_run_id!r}")
    if entry.get("manifest_hash") != manifest_hash:
        raise CheckpointIntegrityError(f"confirmation for {key_desc} is not bound to this run's verified manifest_hash")
    da = document_appraisals_by_id.get(record_id)
    if da is None:
        raise CheckpointIntegrityError(f"confirmation references record_id={record_id!r}, which is not part of this verified run")
    original_leaf = da.leaves().get(field_id)
    if original_leaf is None:
        raise CheckpointIntegrityError(f"confirmation references field_id={field_id!r}, which is not a real leaf of record_id={record_id!r}'s verified appraisal")
    if entry.get("original_leaf_response") != original_leaf.model_dump(mode="json"):
        raise CheckpointIntegrityError(f"confirmation for {key_desc}: original_leaf_response does not match the verified checkpoint's own leaf content")
    # Round-5 finding 1 fix: a confirmation may ONLY target a leaf that
    # would ACTUALLY be a `queue_kind="confirmation"` canonical queue
    # entry -- not merely "provisional" (round-4's own check), since
    # `_canonical_queue_kind_for_leaf()` also accounts for
    # `negative_high_risk_policy` (a provisional leaf under
    # `allow_provisional_ai_only` is NEVER a queue entry at all -- it goes
    # to the non-blocking ledger, so confirming it is illegitimate) AND
    # queue_reason precedence (a leaf with BOTH a queue_reason and
    # provisional_pending_human_confirmation is a `resolution`-kind entry,
    # never `confirmation`-kind, per `_build_review_queue_and_ledger()`'s
    # own precedence rule).
    if _canonical_queue_kind_for_leaf(original_leaf, negative_high_risk_policy=negative_high_risk_policy) != "confirmation":
        raise CheckpointIntegrityError(
            f"confirmation for {key_desc}: the referenced leaf was never a canonical queue_kind='confirmation' "
            f"entry under this run's own negative_high_risk_policy={negative_high_risk_policy!r} -- a confirmation "
            "may only target a leaf that was actually pending human confirmation as a real queue entry"
        )
    reviewer, note, resolved_at = _stripped(entry.get("reviewer")), _stripped(entry.get("note")), _stripped(entry.get("resolved_at"))
    if not reviewer or not note or not resolved_at:
        raise CheckpointIntegrityError(f"confirmation for {key_desc} is missing a non-blank reviewer/note/resolved_at")
    try:
        datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
    except ValueError:
        raise CheckpointIntegrityError(f"confirmation for {key_desc}: resolved_at {resolved_at!r} is not a valid ISO-8601 timestamp")


def _load_and_verify_resolutions(
    path: Path, *, qa_run_id: str, manifest_hash: str, checklist_id: str,
    document_appraisals_by_id: Dict[str, Any], negative_high_risk_policy: str,
) -> List[Dict[str, Any]]:
    """THE ONE recovery entry point for a run's qa_resolutions.json --
    every caller (fresh appraisal, resolve, confirm, the public loader)
    goes through this, never `_read_and_verify_envelope` directly (round-3
    finding 3 fix)."""
    env = _read_and_verify_envelope(path, qa_run_id=qa_run_id, manifest_hash=manifest_hash)
    entries = list((env or {}).get("entries", []))
    seen: set = set()
    for e in entries:
        key = (e.get("record_id"), e.get("field_id"))
        if key in seen:
            raise CheckpointIntegrityError(f"qa_resolutions.json contains a duplicate resolution for {key!r}")
        seen.add(key)
        _verify_resolution_entry(
            e, qa_run_id=qa_run_id, manifest_hash=manifest_hash, checklist_id=checklist_id,
            document_appraisals_by_id=document_appraisals_by_id, negative_high_risk_policy=negative_high_risk_policy,
        )
    return entries


def _assert_resolutions_and_confirmations_disjoint(resolutions: List[Dict[str, Any]], confirmations: List[Dict[str, Any]]) -> None:
    """Round-4 finding 1 fix: `qa_resolutions.json` and `qa_confirmations.json`
    must never both claim the same (record_id, field_id) -- resolve
    (override) and confirm (preserve) are mutually exclusive actions per
    leaf. Enforced defensively on every load, not merely at the point
    each file is individually written (which only ever checks its OWN
    key set), so a key present in BOTH files -- however it got there --
    is caught before its content can be silently double-applied in
    projection (`_project_document_appraisal` previously applied a
    resolution AND a confirmation for the same key with no error)."""
    resolved_keys = {(e["record_id"], e["field_id"]) for e in resolutions}
    confirmed_keys = {(e["record_id"], e["field_id"]) for e in confirmations}
    overlap = resolved_keys & confirmed_keys
    if overlap:
        raise CheckpointIntegrityError(
            f"qa_resolutions.json and qa_confirmations.json both claim key(s) {sorted(overlap)!r} -- "
            "resolve and confirm are mutually exclusive actions per (record_id, field_id)"
        )


def _load_and_verify_confirmations(
    path: Path, *, qa_run_id: str, manifest_hash: str,
    document_appraisals_by_id: Dict[str, Any], negative_high_risk_policy: str,
) -> List[Dict[str, Any]]:
    env = _read_and_verify_envelope(path, qa_run_id=qa_run_id, manifest_hash=manifest_hash)
    entries = list((env or {}).get("entries", []))
    seen: set = set()
    for e in entries:
        key = (e.get("record_id"), e.get("field_id"))
        if key in seen:
            raise CheckpointIntegrityError(f"qa_confirmations.json contains a duplicate confirmation for {key!r}")
        seen.add(key)
        _verify_confirmation_entry(
            e, qa_run_id=qa_run_id, manifest_hash=manifest_hash,
            document_appraisals_by_id=document_appraisals_by_id, negative_high_risk_policy=negative_high_risk_policy,
        )
    return entries


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def quality_appraisal_node(state: ReviewState) -> Dict[str, Any]:
    """Runs (or RESUMES -- Finding 2 fix, see below) structural quality
    appraisal over every record in `state["full_text_records"]`. NEVER
    returns a `full_text_records` key -- this stage never mutates
    eligibility (Section 1.2).

    Resumability (Finding 2 fix): `qa_run_id` is now computed from a
    cheap, direct `pdf_sha256` file hash for every record -- BEFORE any
    LLM call is made -- so an identical retry (same documents/config)
    always recomputes the SAME `qa_run_id` and finds its checkpoints
    already on disk. A record whose checkpoint already exists under this
    exact `qa_run_id` is loaded and reused verbatim, never re-appraised;
    only records with no existing checkpoint (a genuinely new run, or one
    that never finished) trigger an LLM call. This closes both halves of
    the original gap: a nondeterministic nth call to the same batch can
    no longer silently replace an already-frozen checkpoint (there IS no
    nth call for an already-checkpointed record), and the manifest is
    published once and never rewritten with different content afterward
    (`store.write_manifest`'s own immutable-once-published discipline,
    below) -- so existing human resolutions/confirmations, bound to that
    one manifest_hash, are never invalidated by a routine resume."""
    cfg = ReviewConfig.from_dict(state["config"])
    records = [Record.from_dict(r) for r in state["full_text_records"]]
    prisma = PRISMACounts(**state["prisma_counts"])

    if not records:
        prisma.quality_assessed = 0
        prisma.excluded_quality = 0
        return {
            "quality_appraisal_results": [], "quality_appraisal_review_queue": [],
            "quality_appraisal_resolutions": [], "quality_appraisal_confirmations": [],
            "quality_appraisal_provisional_ledger": [], "requires_human_review": False,
            "quality_appraisal_run_id": None, "quality_appraisal_manifest_path": None,
            "quality_appraisal_manifest_hash": None, "quality_appraisal_artifact_paths": None,
            "prisma_counts": prisma.model_dump(),
        }

    for r in records:
        if r.full_text_status != "parsed" or not r.full_text_path or not r.full_text_path.strip() or not Path(r.full_text_path).is_file():
            raise ValueError(
                f"full_text_records contains record(s) with no confirmed, existing full-text PDF: {r.id!r} -- "
                "quality_appraisal_node only appraises a confirmed PDF; there is no title/abstract fallback"
            )

    # Finding 4b fix: reject a duplicate record_id outright, before it can
    # collapse two documents onto one checkpoint path (mirrors
    # nodes.extraction._verify_and_canonicalize_document_order's own
    # duplicate-id check).
    record_id_list = [r.id for r in records]
    duplicate_record_ids = sorted({rid for rid in record_id_list if record_id_list.count(rid) > 1})
    if duplicate_record_ids:
        raise ValueError(f"full_text_records contains duplicate record_id(s): {duplicate_record_ids}")

    checklist_id = cfg.quality_appraisal.checklist_id
    negative_high_risk_policy = cfg.quality_appraisal.negative_high_risk_policy
    model_config = _load_qa_model_config(cfg)
    parser_options = _load_qa_parser_options(cfg)
    max_batch_chars = cfg.quality_appraisal_max_batch_chars
    checklist_content_hash = _checklist_content_hash(checklist_id)
    roll_up_method_ids = _roll_up_method_ids_for(checklist_id)
    prompt_template_version = PROMPT_TEMPLATE_VERSION_TEACHING if checklist_id == "teaching_appraisal_v1" else PROMPT_TEMPLATE_VERSION_ROB2

    import slr_pipeline.extraction.parsing as parsing_module
    from slr_pipeline.extraction.hashing import file_sha256
    parser_version = parsing_module.installed_docling_version()

    # Finding 2/4a fix: a REAL content hash of each PDF's actual bytes,
    # computed directly (never derived from a successful Docling parse) --
    # BEFORE any appraisal is attempted. This means (a) qa_run_id can be
    # computed, and existing checkpoints discovered, WITHOUT running a
    # single LLM call first, and (b) two genuinely different (byte-
    # different) PDFs for the same record_id -- including two different
    # PDFs that BOTH happen to fail Docling parsing -- always get
    # DIFFERENT pdf_sha256 values (never the empty-string collision the
    # old parse-failure path produced).
    pdf_sha256_by_record = {r.id: file_sha256(r.full_text_path) for r in records}

    documents = [(r.id, pdf_sha256_by_record[r.id]) for r in records]
    citation_metadata_hash = sha256_hex({r.id: r.title for r in records})
    qa_run_id = compute_qa_run_id(
        documents=documents, checklist_id=checklist_id, checklist_version=CHECKLIST_VERSION,
        checklist_content_hash=checklist_content_hash, roll_up_method_ids=roll_up_method_ids,
        prompt_template_version=prompt_template_version, model=model_config.model,
        model_config_id=model_config.model_config_id, reasoning_effort=model_config.reasoning_effort,
        max_completion_tokens=model_config.max_completion_tokens, expected_provider_model=model_config.expected_provider_model,
        parser_name="docling", parser_version=parser_version, parser_options=parser_options,
        max_batch_chars=max_batch_chars, evidence_verifier_version=EVIDENCE_VERIFIER_VERSION,
        negative_high_risk_policy=negative_high_risk_policy, citation_metadata_hash=citation_metadata_hash,
    )

    output_dir = cfg.ensure_output_dir()
    store = QACheckpointStore(output_dir=output_dir, qa_run_id=qa_run_id)

    # Round-4 finding 4 fix: the ENTIRE publication sequence for this run
    # -- checkpoint preflight, appraisal, checkpoint/manifest publication,
    # AND the resolutions/confirmations read below -- is now one locked
    # transaction, closing the concurrency race the previous version had:
    # `write_checkpoint()`/`write_manifest()`'s own check-then-write
    # (`path.exists()` then `_atomic_write_json`) was NOT protected by any
    # lock, so two concurrent `quality_appraisal_node()` calls for the
    # SAME qa_run_id could both observe "absent" and both proceed,
    # allowing either a last-writer-wins overwrite or (before the
    # `_atomic_write_json` fix above) temporary-file collision. Holding
    # this same per-run lock for the whole function body is safe: two
    # calls for the SAME qa_run_id describe the IDENTICAL request by
    # construction (qa_run_id is content-addressed), so full serialization
    # is the correct behavior, not a throughput concession -- the second
    # caller simply finds everything the first already published.
    with _qa_resolutions_lock(store):
        # Round-5 finding 4 fix: if a manifest for THIS EXACT qa_run_id
        # already exists, load+verify it -- and every checkpoint it binds
        # -- BEFORE any appraisal call for ANY record, and NEVER rebuild or
        # republish it. By construction (qa_run_id is a content hash of
        # the ordered documents+config), a manifest found under THIS
        # store's own qa_run_id necessarily already describes exactly this
        # request. `VerifiedQARunBundle.from_store()` + `.verify()` load
        # and cross-check EVERY bound checkpoint's own content_hash/
        # checklist_id/checklist_version against the manifest in one pass
        # -- a missing (now-deleted) or corrupted bound checkpoint raises
        # immediately here, never silently falls through to a fresh (paid)
        # `_appraise_document()` call for that record. Using the loaded
        # manifest VERBATIM (never reconstructing a fresh `QAManifest(...)`
        # from the CURRENT call's own `records`) is what makes a
        # byte-identical-but-relocated PDF path a non-issue: `pdf_paths`
        # is provenance-only (never part of qa_run_id's own hash), so a
        # freshly rebuilt manifest carrying the NEW path would differ from
        # the published one and spuriously trip write_manifest()'s own
        # immutability check -- reusing the existing manifest object
        # sidesteps that entirely, retaining the ORIGINAL run's own
        # provenance as authoritative.
        if store.manifest_path().exists():
            bundle = VerifiedQARunBundle.from_store(store)
            bundle.verify()
            document_appraisals = list(bundle.document_appraisals)
            manifest = bundle.manifest
        else:
            # Genuinely new (or partial-crash-before-manifest-publish)
            # run: preflight every EXISTING checkpoint for every record
            # FIRST -- raising immediately on ANY corruption found, for
            # ANY record, before a single `_appraise_document()` (paid)
            # call is made for ANY record. Without this, records iterated
            # before a later, corrupted one could still trigger real paid
            # calls before the loop ever reached the corruption.
            existing_checkpoints: Dict[str, Optional[QAAppraisalCheckpoint]] = {}
            for record in records:
                if store.checkpoint_path(record.id).exists():
                    # A checkpoint file that EXISTS but fails to load/verify
                    # (tampered, corrupted, truncated by a prior crash
                    # mid-write) is a HARD failure, raised immediately --
                    # NEVER silently caught and treated as "no checkpoint,
                    # appraise fresh." A corruption needs a human to see it
                    # and decide -- it must never be "fixed" by silently
                    # re-spending money over it.
                    checkpoint = store.load_checkpoint(record.id)
                    # Round-6 finding 2 fix: `load_checkpoint()`'s own
                    # `.verify()` only checks the checkpoint's OWN internal
                    # self-consistency (content_hash, qa_run_id, record_id)
                    # -- it does NOT cross-check the checkpoint's own
                    # claimed `pdf_sha256`/`checklist_id`/`checklist_version`
                    # against what THIS call's own current inputs actually
                    # are. A forged-but-self-consistent checkpoint (correct
                    # qa_run_id/record_id, but a WRONG pdf_sha256) previously
                    # passed here undetected, was reused verbatim (zero LLM
                    # calls), and the resulting manifest -- itself only
                    # self-hash-consistent, never cross-checked against its
                    # own bound checkpoints at THIS point -- was published
                    # and returned as a success. The break only surfaced
                    # later, the first time anyone called
                    # `VerifiedQARunBundle.verify()` (which recomputes
                    # qa_run_id from the checkpoints' own pdf_sha256 values)
                    # against it. Checked here, BEFORE any appraisal call
                    # for ANY record, so a later (genuinely missing)
                    # record's real LLM call never happens before an
                    # earlier record's corruption is caught.
                    if checkpoint.pdf_sha256 != pdf_sha256_by_record[record.id]:
                        raise CheckpointIntegrityError(
                            f"existing checkpoint for record_id={record.id!r} claims pdf_sha256="
                            f"{checkpoint.pdf_sha256!r}, which does not match this run's own current PDF content "
                            f"hash {pdf_sha256_by_record[record.id]!r} -- refusing to reuse it"
                        )
                    if checkpoint.checklist_id != checklist_id or checkpoint.checklist_version != CHECKLIST_VERSION:
                        raise CheckpointIntegrityError(
                            f"existing checkpoint for record_id={record.id!r} claims checklist_id="
                            f"{checkpoint.checklist_id!r}/checklist_version={checkpoint.checklist_version!r}, which "
                            f"does not match this run's own checklist_id={checklist_id!r}/checklist_version="
                            f"{CHECKLIST_VERSION!r} -- refusing to reuse it"
                        )
                    existing_checkpoints[record.id] = checkpoint
                else:
                    existing_checkpoints[record.id] = None

            document_appraisals = []
            checkpoint_hashes: Dict[str, str] = {}
            for record in records:
                existing_checkpoint = existing_checkpoints[record.id]
                if existing_checkpoint is not None:
                    checkpoint = existing_checkpoint
                    doc_appraisal = checkpoint.document_appraisal
                else:
                    doc_appraisal = _appraise_document(
                        record=record, model_config=model_config, checklist_id=checklist_id,
                        parser_options=parser_options, max_batch_chars=max_batch_chars,
                    )
                    checkpoint = QAAppraisalCheckpoint.build(
                        qa_run_id=qa_run_id, record_id=record.id, pdf_sha256=pdf_sha256_by_record[record.id],
                        checklist_id=checklist_id, checklist_version=CHECKLIST_VERSION, document_appraisal=doc_appraisal,
                    )
                    checkpoint = store.write_checkpoint(checkpoint)

                document_appraisals.append(doc_appraisal)
                checkpoint_hashes[record.id] = checkpoint.content_hash

            manifest = QAManifest(
                qa_run_id=qa_run_id, checklist_id=checklist_id, checklist_version=CHECKLIST_VERSION,
                checklist_content_hash=checklist_content_hash, roll_up_method_ids=roll_up_method_ids,
                prompt_template_version=prompt_template_version, model=model_config.model,
                model_config_id=model_config.model_config_id, reasoning_effort=model_config.reasoning_effort,
                max_completion_tokens=model_config.max_completion_tokens, expected_provider_model=model_config.expected_provider_model,
                parser_name="docling", parser_version=parser_version,
                parser_options={"do_ocr": parser_options.do_ocr, "do_table_structure": parser_options.do_table_structure, "max_chunk_chars": parser_options.max_chunk_chars},
                max_batch_chars=max_batch_chars, evidence_verifier_version=EVIDENCE_VERIFIER_VERSION,
                negative_high_risk_policy=negative_high_risk_policy, citation_metadata_hash=citation_metadata_hash,
                document_ids=[r.id for r in records], checkpoint_hashes=checkpoint_hashes,
                pdf_paths={r.id: r.full_text_path for r in records},
                extraction_run_id=state.get("extraction_run_id"), extraction_manifest_hash=state.get("extraction_manifest_hash"),
                protocol_uri=cfg.protocol_uri, protocol_hash=cfg.protocol_hash, protocol_registration_id=cfg.protocol_registration_id,
                ai_use_prespecified=cfg.ai_use_prespecified, protocol_deviations=list(cfg.protocol_deviations),
                created_at=datetime.now(timezone.utc).isoformat(),
                manifest_hash="0" * 64,
            )
            manifest_dict = manifest.model_dump(mode="json", exclude={"manifest_hash"})
            manifest = manifest.model_copy(update={"manifest_hash": sha256_hex(manifest_dict)})
            # A genuine first publish -- store.manifest_path() did not
            # exist at the top of this branch, so this can only be a true
            # first write (never a republish, since we hold the per-run
            # lock the whole time); write_manifest()'s own no-op-if-
            # identical/raise-if-different discipline is defense in depth.
            manifest = store.write_manifest(manifest)

            # Round-6 finding 2 fix, second half: construct and verify the
            # COMPLETE bundle immediately after publishing -- defense in
            # depth beyond the per-checkpoint pdf_sha256/checklist checks
            # above (which check each checkpoint against THIS call's own
            # inputs individually). `VerifiedQARunBundle.verify()`
            # additionally re-derives `qa_run_id` from the ordered
            # documents+config the CHECKPOINTS themselves (not `records`)
            # claim, and requires it reproduce this store's own qa_run_id
            # -- catching any cross-checkpoint inconsistency the per-record
            # checks individually could not (e.g. two checkpoints that are
            # each individually self-consistent and pdf_sha256-correct but
            # disagree on shared config fields). Never returns success
            # without this passing.
            bundle = VerifiedQARunBundle.from_store(store)
            bundle.verify()

        document_appraisals_by_id = {da.record_id: da for da in document_appraisals}
        existing_resolutions = _load_and_verify_resolutions(
            store.resolutions_path(), qa_run_id=qa_run_id, manifest_hash=manifest.manifest_hash,
            checklist_id=checklist_id, document_appraisals_by_id=document_appraisals_by_id,
            negative_high_risk_policy=manifest.negative_high_risk_policy,
        )
        existing_confirmations = _load_and_verify_confirmations(
            store.confirmations_path(), qa_run_id=qa_run_id, manifest_hash=manifest.manifest_hash,
            document_appraisals_by_id=document_appraisals_by_id,
            negative_high_risk_policy=manifest.negative_high_risk_policy,
        )
        _assert_resolutions_and_confirmations_disjoint(existing_resolutions, existing_confirmations)

    resolved_keys = {(e["record_id"], e["field_id"]) for e in existing_resolutions}
    confirmed_keys = {(e["record_id"], e["field_id"]) for e in existing_confirmations}

    review_queue, provisional_ledger = _build_review_queue_and_ledger(
        document_appraisals, resolved_keys=resolved_keys, confirmed_keys=confirmed_keys,
        negative_high_risk_policy=negative_high_risk_policy,
    )

    prisma.quality_assessed = len(records)
    prisma.excluded_quality = 0  # quality appraisal never excludes (Section 1.2)

    return {
        "quality_appraisal_results": [d.model_dump(mode="json") for d in document_appraisals],
        "quality_appraisal_review_queue": review_queue,
        "quality_appraisal_resolutions": existing_resolutions,
        "quality_appraisal_confirmations": existing_confirmations,
        "quality_appraisal_provisional_ledger": provisional_ledger,
        "requires_human_review": bool(review_queue),
        "quality_appraisal_run_id": qa_run_id,
        "quality_appraisal_manifest_path": str(store.manifest_path()),
        "quality_appraisal_manifest_hash": manifest.manifest_hash,
        "quality_appraisal_artifact_paths": {
            "run_dir": str(store.run_dir), "checkpoints_dir": str(store.checkpoints_dir),
            "manifest_path": str(store.manifest_path()), "resolutions_path": str(store.resolutions_path()),
            "confirmations_path": str(store.confirmations_path()),
        },
        "prisma_counts": prisma.model_dump(),
    }


# ---------------------------------------------------------------------------
# Resolution / recomputation helpers (Section 1.7) -- re-run the roll-up
# functions fresh over CURRENT leaves rather than trusting a frozen
# checkpoint's now-possibly-stale domain_risk/overall_risk/rationale.
# ---------------------------------------------------------------------------

_STALE_RATIONALE_TEMPLATE = "Recomputed after {n} human resolution(s) to signalling question(s) {ids}; see qa_resolutions.json for the full audit trail."


def _apply_resolutions_to_leaves(leaves: Dict[str, Any], resolutions: List[Dict[str, Any]], record_id: str) -> Dict[str, Any]:
    """Returns a NEW leaves dict with every resolution for `record_id`
    applied on top of the frozen machine leaves -- never mutates the
    frozen checkpoint objects themselves."""
    result = dict(leaves)
    for res in resolutions:
        if res["record_id"] != record_id:
            continue
        field_id = res["field_id"]
        original = result.get(field_id)
        if original is None:
            continue
        common = dict(
            evidence=[AppraisalEvidence(**e) for e in res["evidence"]],
            rationale=res.get("human_rationale") or "Human-attested resolution.",
            decision_origin="human", evidence_status="human_attested",
            chunks_examined=list(getattr(original, "chunks_examined", [])),
            batches_examined=getattr(original, "batches_examined", 0),
            provisional_pending_human_confirmation=False, human_confirmation=None,
            # A resolution overrides/closes this leaf outright -- it must
            # never re-enter the review queue afterward regardless of what
            # the FROZEN machine leaf's own queue_reason/coverage_complete
            # said (Finding 1/3 fix continuity: a resolution is a genuine
            # override, not merely an annotation on top of the old state).
            coverage_complete=True, queue_reason=None, conflicting_evidence=None,
        )
        if isinstance(original, AppraisalItemResponse):
            result[field_id] = AppraisalItemResponse(item_id=original.item_id, status=res["value"], **common)
        else:
            result[field_id] = SignallingQuestionResponse(question_id=original.question_id, answer=res["value"], **common)
    return result


def _project_document_appraisal(document_appraisal: DocumentAppraisal, resolutions: List[Dict[str, Any]], confirmations: List[Dict[str, Any]]) -> DocumentAppraisal:
    """Applies resolutions AND confirmations on top of a frozen checkpoint's
    DocumentAppraisal, then re-runs the roll-up functions fresh over the
    CURRENT leaves (Section 1.7) -- never trusts the checkpoint's own
    stored domain_risk/overall_risk/rationale once any leaf has changed."""
    record_id = document_appraisal.record_id
    leaves = document_appraisal.leaves()
    leaves = _apply_resolutions_to_leaves(leaves, resolutions, record_id)

    for conf in confirmations:
        if conf["record_id"] != record_id:
            continue
        field_id = conf["field_id"]
        original = leaves.get(field_id)
        if original is None:
            continue
        hc = HumanConfirmation(reviewer=conf["reviewer"], note=conf["note"], resolved_at=conf["resolved_at"])
        leaves[field_id] = original.model_copy(update={"provisional_pending_human_confirmation": False, "human_confirmation": hc})

    if document_appraisal.teaching is not None:
        items = [leaves[i.item_id] for i in document_appraisal.teaching.items]
        return document_appraisal.model_copy(update={"teaching": TeachingAppraisal(items=items)})

    resolved_field_ids_by_domain: Dict[str, List[str]] = {}
    for res in resolutions:
        if res["record_id"] != record_id:
            continue
        domain_id, _, qid = res["field_id"].partition(".")
        resolved_field_ids_by_domain.setdefault(domain_id, []).append(res["field_id"])

    new_domains = []
    any_resolved_overall = False
    for d in document_appraisal.rob2.domains:
        questions = [leaves[f"{d.domain_id}.{q.question_id}"] for q in d.signalling_questions]
        resolved_ids = resolved_field_ids_by_domain.get(d.domain_id, [])
        if resolved_ids:
            any_resolved_overall = True
            domain_risk, basis = roll_up_domain_risk(questions)
            rationale = _STALE_RATIONALE_TEMPLATE.format(n=len(resolved_ids), ids=sorted(resolved_ids))
            new_domains.append(DomainJudgment(domain_id=d.domain_id, signalling_questions=questions, domain_risk=domain_risk, domain_risk_rationale=rationale, basis=basis))
        else:
            new_domains.append(d.model_copy(update={"signalling_questions": questions}))

    if any_resolved_overall:
        overall_risk = roll_up_overall_risk(new_domains)
        n_total = sum(len(v) for v in resolved_field_ids_by_domain.values())
        all_ids = sorted(fid for ids in resolved_field_ids_by_domain.values() for fid in ids)
        overall_rationale = _STALE_RATIONALE_TEMPLATE.format(n=n_total, ids=all_ids)
    else:
        overall_risk = document_appraisal.rob2.overall_risk
        overall_rationale = document_appraisal.rob2.overall_risk_rationale

    new_rob2 = document_appraisal.rob2.model_copy(update={"domains": new_domains, "overall_risk": overall_risk, "overall_risk_rationale": overall_rationale})
    return document_appraisal.model_copy(update={"rob2": new_rob2})


def _build_qa_evidence_index(
    projections: List["DocumentAppraisal"], qa_run_id: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Builds the per-record, per-field evidence index synthesis_node's own map step needs to cite
    exact QA evidence spans -- additive, in-memory only, never written back to disk. `AppraisalEvidence`
    (this module) stores only chunk_id/quotation/locator -- never page/section -- so both real
    branches below pass page/section explicitly as None. evidence_status=='verified_from_chunk':
    one entry per leaf.evidence item, RAW chunk_id/quotation/locator. evidence_status=='human_attested':
    one entry per leaf.evidence item, chunk_id re-derived via _human_attested_chunk_id() (matching the
    extraction-side convention). Everything else ('not_provided'/'not_applicable'): no entries."""
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for da in projections:
        record_index: Dict[str, List[Dict[str, Any]]] = {}
        for field_id, leaf in da.leaves().items():
            entries_raw: List[Dict[str, Optional[str]]] = []
            if leaf.evidence_status == "verified_from_chunk":
                for ev in leaf.evidence:
                    entries_raw.append({"chunk_id": ev.chunk_id, "quotation": ev.quotation,
                                         "page": None, "section": None, "locator": ev.locator})
            elif leaf.evidence_status == "human_attested":
                for ev in leaf.evidence:
                    entries_raw.append({"chunk_id": _human_attested_chunk_id(ev.locator), "quotation": ev.quotation,
                                         "page": None, "section": None, "locator": ev.locator})
            if entries_raw:
                record_index[field_id] = build_evidence_index_for_field(
                    entries_raw, artifact_kind="quality_appraisal", artifact_run_id=qa_run_id,
                    record_id=da.record_id, field_id=field_id,
                )
        if record_index:
            index[da.record_id] = record_index
    return index


def load_verified_quality_appraisal_resolution_artifact(state: ReviewState) -> Optional[Dict[str, Any]]:
    """The ONE supported way for a caller (synthesis_node, reporting_node)
    to read back the CURRENT, resolution/confirmation-applied projection
    for `state["quality_appraisal_run_id"]`. Returns `None` if quality
    appraisal never ran. Raises on tampering/staleness."""
    qa_run_id = state.get("quality_appraisal_run_id")
    if not qa_run_id:
        return None
    cfg = ReviewConfig.from_dict(state["config"])
    output_dir = cfg.ensure_output_dir()
    store = QACheckpointStore(output_dir=output_dir, qa_run_id=qa_run_id)
    bundle = VerifiedQARunBundle.from_store(store)
    bundle.verify()
    document_appraisals_by_id = {da.record_id: da for da in bundle.document_appraisals}

    with _qa_resolutions_lock(store):
        resolutions = _load_and_verify_resolutions(
            store.resolutions_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            checklist_id=bundle.manifest.checklist_id, document_appraisals_by_id=document_appraisals_by_id,
            negative_high_risk_policy=bundle.manifest.negative_high_risk_policy,
        )
        confirmations = _load_and_verify_confirmations(
            store.confirmations_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            document_appraisals_by_id=document_appraisals_by_id,
            negative_high_risk_policy=bundle.manifest.negative_high_risk_policy,
        )
        _assert_resolutions_and_confirmations_disjoint(resolutions, confirmations)

    projections = [_project_document_appraisal(da, resolutions, confirmations) for da in bundle.document_appraisals]

    return {
        "qa_run_id": qa_run_id, "manifest_hash": bundle.manifest.manifest_hash,
        "manifest_path": str(store.manifest_path()),
        "resolutions": resolutions, "confirmations": confirmations,
        "resolved_quality_projection": [p.model_dump(mode="json") for p in projections],
        # Additive, in-memory-only, never written back to disk -- see _build_qa_evidence_index().
        "evidence_index": _build_qa_evidence_index(projections, qa_run_id),
    }


def _stripped(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _validate_human_resolution_value(field_id: str, checklist_id: str, value: Any, evidence: Any) -> None:
    if checklist_id == "teaching_appraisal_v1":
        allowed = {"yes", "no", "unclear", "not_applicable"}
    else:
        allowed = {"Y", "PY", "PN", "N", "NI"}
    if value not in allowed:
        raise ValueError(f"resolution for {field_id!r}: value must be one of {sorted(allowed)}, got {value!r}")
    requires_evidence = value not in ("unclear", "not_applicable", "NI")
    if requires_evidence:
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"resolution for {field_id!r}: value={value!r} requires non-empty structured 'evidence'")
        for item in evidence:
            if not isinstance(item, dict) or not _stripped(item.get("quotation")) or not _stripped(item.get("locator")):
                raise ValueError(f"resolution for {field_id!r}: each evidence item requires non-blank 'quotation' and 'locator'")
            item.setdefault("chunk_id", "human-attested")
    elif evidence:
        raise ValueError(f"resolution for {field_id!r}: value={value!r} must not carry evidence")


def _validate_queue_against_canonical(
    queue: List[Dict[str, Any]], expected_by_key: Dict[Tuple[str, str], Dict[str, Any]], *, already_closed_keys: set,
) -> set:
    """Validates the caller-supplied `state["quality_appraisal_review_queue"]`
    against the canonical `expected_by_key` (recomputed from the verified
    bundle), tolerating "artifact-ahead-of-state" staleness (round-3
    finding 4c): a key ALREADY resolved or confirmed on disk since the
    caller's state snapshot was taken (`already_closed_keys` --
    `resolved_keys_existing | confirmed_keys_existing`) is allowed to
    still be present in `queue` -- the exact staleness window a crash
    between a durable publish and `graph.update_state()` produces. Every
    OTHER key must match the canonical set exactly (never silently
    dropped or added), and every non-stale entry's own `leaf_response`/
    `queue_kind` must match the canonical entry's (Finding 7 continuity --
    a stale entry's own leaf_response describes the PRE-closure state, so
    it is not checked). Returns `stale_keys` (the subset of `queue`'s own
    keys found to be already-closed) for the caller's own idempotent-
    replay handling."""
    keys = [(e["record_id"], e["field_id"]) for e in queue]
    duplicate_keys = sorted({k for k in keys if keys.count(k) > 1})
    if duplicate_keys:
        raise ValueError(f"state['quality_appraisal_review_queue'] contains duplicate entries: {duplicate_keys}")

    actual_keys = set(keys)
    stale_keys = actual_keys & already_closed_keys
    live_keys = actual_keys - stale_keys
    if live_keys != set(expected_by_key):
        raise ValueError(
            "state['quality_appraisal_review_queue'] does not exactly match the verified run's own current "
            f"queue (expected {sorted(expected_by_key)!r}, got {sorted(live_keys)!r}"
            + (f", after discounting {sorted(stale_keys)!r} already resolved/confirmed on disk" if stale_keys else "")
            + ")"
        )

    for entry in queue:
        key = (entry["record_id"], entry["field_id"])
        if key in stale_keys:
            continue
        verified_entry = expected_by_key[key]
        if entry.get("leaf_response") != verified_entry["leaf_response"] or entry.get("queue_kind") != verified_entry["queue_kind"]:
            raise ValueError(
                f"quality_appraisal_review_queue entry for {key!r} does not match the verified run's own "
                "current leaf content -- refusing to act against possibly tampered or stale queue state"
            )
    return stale_keys


def resolve_quality_appraisal_review_queue(state: ReviewState, resolutions: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Applies human decisions to ANY currently-queued (record_id,
    field_id) entry -- both `queue_kind="resolution"` (unclear/evidence-
    verification-failed/conflicting/incomplete-coverage) AND
    `queue_kind="confirmation"` (a provisional negative/high-risk finding
    a human wants to OVERRIDE rather than merely confirm -- Finding 1
    fix: a machine "no"/"N" was previously a permanent dead end, since
    only `queue_kind="resolution"` entries were ever eligible here; a
    provisional finding is now resolvable exactly like any other queued
    leaf, while `confirm_quality_appraisal_provisional_item` remains the
    separate, narrower, evidence-PRESERVING action for a human who simply
    agrees with it). Same contract shape as
    `nodes.extraction.resolve_extraction_review_queue` (Section 1.7):
    requires reviewer/note/resolved_at, requires the queue to exactly
    match the verified run's own current queue (computed via the SAME
    `_build_review_queue_and_ledger()` every other entry point uses),
    structural (human-attested, non-byte-verified) evidence, producing
    decision_origin="human"/evidence_status="human_attested", and NEVER
    touching full_text_records.

    Finding 7 fix: every targeted queue entry's own `leaf_response` is
    cross-checked against the VERIFIED bundle's own leaf content before
    being trusted -- `state["quality_appraisal_review_queue"]` is mutable,
    LangGraph-checkpointed state, and only its (record_id, field_id) KEY
    set was previously verified against the bundle; a tampered/stale
    `leaf_response` on an otherwise-correctly-keyed entry would previously
    flow straight into the permanent `qa_resolutions.json` audit trail
    unchecked (mirrors `nodes.extraction.resolve_extraction_review_queue`'s
    own `entry.get("field_result") != verified_field...` check)."""
    qa_run_id = state.get("quality_appraisal_run_id")
    if not qa_run_id:
        raise ValueError("state['quality_appraisal_run_id'] is not set -- there is no verifiable quality-appraisal run to resolve against")

    cfg = ReviewConfig.from_dict(state["config"])
    output_dir = cfg.ensure_output_dir()
    store = QACheckpointStore(output_dir=output_dir, qa_run_id=qa_run_id)
    bundle = VerifiedQARunBundle.from_store(store)
    bundle.verify()
    checklist_id = bundle.manifest.checklist_id
    negative_high_risk_policy = bundle.manifest.negative_high_risk_policy

    document_appraisals_by_id = {da.record_id: da for da in bundle.document_appraisals}

    with _qa_resolutions_lock(store):
        existing_resolutions = _load_and_verify_resolutions(
            store.resolutions_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            checklist_id=checklist_id, document_appraisals_by_id=document_appraisals_by_id,
            negative_high_risk_policy=negative_high_risk_policy,
        )
        existing_confirmations = _load_and_verify_confirmations(
            store.confirmations_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            document_appraisals_by_id=document_appraisals_by_id, negative_high_risk_policy=negative_high_risk_policy,
        )
        _assert_resolutions_and_confirmations_disjoint(existing_resolutions, existing_confirmations)
        existing_resolutions_env = {"qa_run_id": qa_run_id, "manifest_hash": bundle.manifest.manifest_hash, "entries": existing_resolutions}
        resolved_keys_existing = {(e["record_id"], e["field_id"]) for e in existing_resolutions}
        confirmed_keys_existing = {(e["record_id"], e["field_id"]) for e in existing_confirmations}

        # THE canonical current queue, recomputed directly from the
        # verified bundle -- never trusted from state alone. Every entry
        # in it (both queue_kind values) is eligible for resolution.
        expected_queue, _expected_ledger = _build_review_queue_and_ledger(
            bundle.document_appraisals, resolved_keys=resolved_keys_existing, confirmed_keys=confirmed_keys_existing,
            negative_high_risk_policy=negative_high_risk_policy,
        )
        expected_by_key = {(e["record_id"], e["field_id"]): e for e in expected_queue}

        queue = list(state["quality_appraisal_review_queue"])
        # Round-3 finding 4c fix: tolerates a queue entry whose key is
        # ALREADY resolved/confirmed on disk (the "artifact-ahead-of-
        # state" window a crash between a durable publish and
        # graph.update_state() produces) -- an identical retry recovers
        # idempotently instead of raising.
        already_closed_keys = resolved_keys_existing | confirmed_keys_existing
        stale_keys = _validate_queue_against_canonical(queue, expected_by_key, already_closed_keys=already_closed_keys)

        queue_by_key = {(e["record_id"], e["field_id"]): e for e in queue}
        unknown = set(resolutions) - set(queue_by_key)
        if unknown:
            raise KeyError(f"(record_id, field_id) pair(s) not in quality_appraisal_review_queue: {sorted(unknown)}")

        new_resolutions: List[Dict[str, Any]] = []
        existing_resolutions_by_key = {(e["record_id"], e["field_id"]): e for e in existing_resolutions}
        for entry in queue:
            key = (entry["record_id"], entry["field_id"])
            if key not in resolutions:
                continue

            res = resolutions[key]
            reviewer, note, resolved_at = _stripped(res.get("reviewer")), _stripped(res.get("note")), _stripped(res.get("resolved_at"))
            if not reviewer or not note or not resolved_at:
                raise ValueError(f"resolution for {key!r} must set non-blank reviewer, note, and resolved_at")
            try:
                datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"resolution for {key!r}: resolved_at {resolved_at!r} is not a valid ISO-8601 timestamp")
            value = res.get("value")
            evidence = res.get("evidence") or []
            _validate_human_resolution_value(entry["field_id"], checklist_id, value, evidence)

            if key in stale_keys:
                # Round-5 finding 2 fix: round-4's own version of this
                # check compared only `value`/`evidence` -- a retry
                # carrying a genuinely DIFFERENT reviewer, note,
                # resolved_at, or human_rationale than what was already
                # published still silently passed as a "benign" no-op,
                # discarding that conflicting metadata. The COMPLETE
                # normalized candidate (every field this action actually
                # persists, besides the run-scoped qa_run_id/manifest_hash/
                # original_leaf_response, which are structurally identical
                # for the same key within the same run) is now compared;
                # ANY difference raises, not just a value/evidence one.
                if key not in resolved_keys_existing:
                    raise CheckpointIntegrityError(
                        f"resolution for {key!r}: this leaf was already closed via CONFIRMATION (not resolution) "
                        "on disk -- a resolve attempt against it is a genuine conflict, not a benign replay"
                    )
                existing_entry = existing_resolutions_by_key[key]
                candidate = {
                    "value": value, "evidence": evidence, "human_rationale": res.get("human_rationale") or note,
                    "reviewer": reviewer, "note": note, "resolved_at": resolved_at,
                }
                existing_normalized = {k: existing_entry.get(k) for k in candidate}
                if existing_normalized != candidate:
                    raise CheckpointIntegrityError(
                        f"resolution for {key!r}: this leaf was already resolved on disk with DIFFERENT content "
                        f"(persisted={existing_normalized!r}, candidate={candidate!r}) -- refusing to silently "
                        "discard a genuinely conflicting resolve attempt"
                    )
                continue  # true idempotent replay -- nothing new to write

            # Finding 7 fix: the entry's OWN leaf_response/queue_kind
            # already verified against the canonical bundle by
            # `_validate_queue_against_canonical` above.
            verified_entry = expected_by_key[key]
            new_resolutions.append({
                "record_id": entry["record_id"], "field_id": entry["field_id"], "value": value, "evidence": evidence,
                "human_rationale": res.get("human_rationale") or note, "reviewer": reviewer, "note": note,
                "resolved_at": resolved_at, "qa_run_id": qa_run_id, "manifest_hash": bundle.manifest.manifest_hash,
                "original_leaf_response": verified_entry["leaf_response"],
            })

        all_resolutions = existing_resolutions + new_resolutions
        _publish_append_only_envelope(
            store.resolutions_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            entries=all_resolutions, entry_key_fn=lambda e: (e["record_id"], e["field_id"]),
            existing_verified=existing_resolutions_env,
        )

        # Recompute the remaining queue directly from the bundle again,
        # now with the just-published resolutions applied -- a resolved
        # confirmation-kind entry must not linger as if still awaiting
        # confirmation (Finding 1 continuity).
        new_resolved_keys = resolved_keys_existing | {(r["record_id"], r["field_id"]) for r in new_resolutions}
        remaining_queue, _ledger = _build_review_queue_and_ledger(
            bundle.document_appraisals, resolved_keys=new_resolved_keys, confirmed_keys=confirmed_keys_existing,
            negative_high_risk_policy=negative_high_risk_policy,
        )

    return {
        "quality_appraisal_review_queue": remaining_queue,
        "quality_appraisal_resolutions": all_resolutions,
        "requires_human_review": bool(remaining_queue),
    }


def confirm_quality_appraisal_provisional_item(state: ReviewState, confirmations: Dict[Tuple[str, str], Dict[str, str]]) -> Dict[str, Any]:
    """Confirms already-verified negative/high-risk machine leaves --
    genuinely distinct from `resolve_quality_appraisal_review_queue`
    (Section 1.6): sets `human_confirmation` and clears
    `provisional_pending_human_confirmation`, WITHOUT touching
    `decision_origin`/`evidence_status`/`evidence`/`answer`/`status` at
    all. One locked transaction: verify -> write -> dequeue -> recompute
    `requires_human_review` (Section 1.6/1.7's "same contract shape"
    fix)."""
    qa_run_id = state.get("quality_appraisal_run_id")
    if not qa_run_id:
        raise ValueError("state['quality_appraisal_run_id'] is not set -- there is no verifiable quality-appraisal run to confirm against")

    cfg = ReviewConfig.from_dict(state["config"])
    output_dir = cfg.ensure_output_dir()
    store = QACheckpointStore(output_dir=output_dir, qa_run_id=qa_run_id)
    bundle = VerifiedQARunBundle.from_store(store)
    bundle.verify()
    negative_high_risk_policy = bundle.manifest.negative_high_risk_policy

    document_appraisals_by_id = {da.record_id: da for da in bundle.document_appraisals}

    with _qa_resolutions_lock(store):
        existing_resolutions = _load_and_verify_resolutions(
            store.resolutions_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            checklist_id=bundle.manifest.checklist_id, document_appraisals_by_id=document_appraisals_by_id,
            negative_high_risk_policy=negative_high_risk_policy,
        )
        existing_confirmations = _load_and_verify_confirmations(
            store.confirmations_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            document_appraisals_by_id=document_appraisals_by_id, negative_high_risk_policy=negative_high_risk_policy,
        )
        _assert_resolutions_and_confirmations_disjoint(existing_resolutions, existing_confirmations)
        existing_confirmations_env = {"qa_run_id": qa_run_id, "manifest_hash": bundle.manifest.manifest_hash, "entries": existing_confirmations}
        confirmed_keys_existing = {(e["record_id"], e["field_id"]) for e in existing_confirmations}
        resolved_keys_existing = {(e["record_id"], e["field_id"]) for e in existing_resolutions}

        # THE canonical current queue (Finding 1 continuity: same helper
        # every entry point uses). Round-3 finding 1 fix: the ENTIRE
        # queue (both queue_kind values) is now validated against it --
        # not just the confirmation-kind subset -- so a caller cannot
        # silently drop genuine resolution-kind entries from their own
        # `state["quality_appraisal_review_queue"]` undetected. Only
        # `confirmation`-kind entries are ever ACTED on below --
        # `resolve_quality_appraisal_review_queue` remains the function
        # for overriding a value; this one only ever preserves it.
        expected_queue, _expected_ledger = _build_review_queue_and_ledger(
            bundle.document_appraisals, resolved_keys=resolved_keys_existing, confirmed_keys=confirmed_keys_existing,
            negative_high_risk_policy=negative_high_risk_policy,
        )
        expected_by_key = {(e["record_id"], e["field_id"]): e for e in expected_queue}

        queue = list(state["quality_appraisal_review_queue"])
        already_closed_keys = resolved_keys_existing | confirmed_keys_existing
        stale_keys = _validate_queue_against_canonical(queue, expected_by_key, already_closed_keys=already_closed_keys)

        confirmation_queue_entries = [e for e in queue if e["queue_kind"] == "confirmation"]
        queue_by_key = {(e["record_id"], e["field_id"]): e for e in confirmation_queue_entries}
        unknown = set(confirmations) - set(queue_by_key)
        if unknown:
            raise KeyError(f"(record_id, field_id) pair(s) not in quality_appraisal_review_queue's confirmation entries: {sorted(unknown)}")

        new_confirmations: List[Dict[str, Any]] = []
        existing_confirmations_by_key = {(e["record_id"], e["field_id"]): e for e in existing_confirmations}
        for entry in confirmation_queue_entries:
            key = (entry["record_id"], entry["field_id"])
            if key not in confirmations:
                continue

            # Finding 7 fix: the entry's own leaf_response already
            # verified against the canonical bundle by
            # `_validate_queue_against_canonical` above.
            conf = confirmations[key]
            reviewer, note, resolved_at = _stripped(conf.get("reviewer")), _stripped(conf.get("note")), _stripped(conf.get("resolved_at"))
            if not reviewer or not note or not resolved_at:
                raise ValueError(f"confirmation for {key!r} must set non-blank reviewer, note, and resolved_at")
            try:
                datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"confirmation for {key!r}: resolved_at {resolved_at!r} is not a valid ISO-8601 timestamp")

            if key in stale_keys:
                # Round-5 finding 2 fix: round-4's own version of this
                # check treated EVERY same-kind stale retry as
                # automatically identical ("confirmations carry no value
                # to differ over") -- false: reviewer/note/resolved_at ARE
                # real, meaningfully different metadata a retry with a
                # genuinely different reviewer/note/timestamp silently
                # discarded. The COMPLETE normalized candidate is now
                # compared; any difference raises.
                if key not in confirmed_keys_existing:
                    raise CheckpointIntegrityError(
                        f"confirmation for {key!r}: this leaf was already closed via RESOLUTION (not confirmation) "
                        "on disk -- a confirm attempt against it is a genuine conflict, not a benign replay"
                    )
                existing_entry = existing_confirmations_by_key[key]
                candidate = {"reviewer": reviewer, "note": note, "resolved_at": resolved_at}
                existing_normalized = {k: existing_entry.get(k) for k in candidate}
                if existing_normalized != candidate:
                    raise CheckpointIntegrityError(
                        f"confirmation for {key!r}: this leaf was already confirmed on disk with DIFFERENT content "
                        f"(persisted={existing_normalized!r}, candidate={candidate!r}) -- refusing to silently "
                        "discard a genuinely conflicting confirm attempt"
                    )
                continue  # true idempotent replay -- nothing new to write

            new_confirmations.append({
                "record_id": entry["record_id"], "field_id": entry["field_id"], "reviewer": reviewer, "note": note,
                "resolved_at": resolved_at, "qa_run_id": qa_run_id, "manifest_hash": bundle.manifest.manifest_hash,
                "original_leaf_response": entry["leaf_response"],
            })

        all_confirmations = existing_confirmations + new_confirmations
        _publish_append_only_envelope(
            store.confirmations_path(), qa_run_id=qa_run_id, manifest_hash=bundle.manifest.manifest_hash,
            entries=all_confirmations, entry_key_fn=lambda e: (e["record_id"], e["field_id"]),
            existing_verified=existing_confirmations_env,
        )

        # Round-3 finding 1 fix: reconstructed directly from the verified
        # bundle + the just-published confirmations, exactly mirroring
        # `resolve_quality_appraisal_review_queue`'s own already-correct
        # pattern -- NEVER from `queue`'s own resolution-kind subset
        # (which was the actual bug: a caller who omitted genuine
        # resolution entries from their own mutable-state queue before
        # calling this function previously saw them vanish from the
        # returned queue instead of being restored from disk).
        new_confirmed_keys = confirmed_keys_existing | {(c["record_id"], c["field_id"]) for c in new_confirmations}
        remaining_queue, _ledger = _build_review_queue_and_ledger(
            bundle.document_appraisals, resolved_keys=resolved_keys_existing, confirmed_keys=new_confirmed_keys,
            negative_high_risk_policy=negative_high_risk_policy,
        )

    return {
        "quality_appraisal_review_queue": remaining_queue,
        "quality_appraisal_confirmations": all_confirmations,
        "requires_human_review": bool(remaining_queue),
    }
