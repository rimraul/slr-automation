"""
screening_qc_node -- quality control for the screening stage.

Two independent checks on the title/abstract screening decisions:

  1. Inter-rater reliability (IRR). A random sample of already-screened
     records is re-screened by an independent second-rater LLM call, and
     agreement with the primary screening_node decision is reported via
     Cohen's kappa, PABAK, and Gwet's AC1 (preferred over kappa when
     include/exclude prevalence is imbalanced -- usually true in a
     screening pass, where most records get excluded). Reimplements the
     metrics/thresholds from c:/projects/selection_strategies/
     intercoder_agreement.py.

  2. Topic saturation. Are we still discovering new themes, or would
     screening more records be diminishing returns? Tracks how many
     distinct screening_relevant_topics accumulate as records are
     processed in random order (bootstrapped over many permutations),
     fits a Michaelis-Menten curve, and separately computes an empirical
     (curve-fit-free) saturation point plus Jaccard convergence between
     independent samples. Generalized, domain-agnostic version of
     c:/projects/selection_strategies/v23_saturation_moe_tworater.py's
     concept-pattern rarefaction analysis -- that script tracks fixed
     identity-leadership indicator combinations (leader/trauma/colact/
     socid); this tracks whatever topics the screening LLM itself
     surfaced for THIS review, so it works for any topic.

Node input keys:  screened_records_full, config
Node output keys: screening_qc, screened_records_full (annotated with
                   second_rater_decision on the sampled subset),
                   review_run_id, oversight_mode, automation_origin

Stage 3 instrumentation (the Stage 3 node-instrumentation design record): every
second-rater call is a durable `AIUseEvent` (`screening_qc` has no
selection-ledger records at all -- it is not a selection stage). A failed
second-rater call returns `None`, never `"exclude"` -- a fabricated
decision would silently corrupt the IRR agreement computation (INV-06);
`_compute_irr` drops that sample from both rater lists and reports
`n_failed` separately, rendering every agreement statistic `None` (never
a divide-by-zero) when zero valid pairs remain.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import cohen_kappa_score

from slr_pipeline.state import Record, ReviewState
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key
from slr_pipeline.run_identity import resolve_review_run_id

from slr_pipeline.ai_traceability import (
    AIEventLockError, AIUseEventStore, attempt_lock_path, build_ai_use_event,
    classify_automation_origin_optional, compute_event_id, compute_prompt_template_hash,
    kernel_lock, reserved_attempt,
)
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.resolution_suggestions import _fsync_dir

logger = logging.getLogger(__name__)

STAGE = "screening_qc_second_rater"


def _record_content_ref(r: Record) -> str:
    return f"record:{r.id}:sha256:{sha256_hex({'title': r.title, 'abstract': r.abstract, 'doi': r.doi})}"


def _criteria_ref(inclusion_criteria: List[str], exclusion_criteria: List[str], research_questions: List[str]) -> str:
    return f"criteria:sha256:{sha256_hex({'inclusion_criteria': inclusion_criteria, 'exclusion_criteria': exclusion_criteria, 'research_questions': research_questions})}"


# ---------------------------------------------------------------------------
# Inter-rater reliability
# ---------------------------------------------------------------------------

def _gwet_ac1(a: List[str], b: List[str]) -> Tuple[float, float, float]:
    """Gwet's AC1 for two binary raters -- robust to prevalence imbalance,
    unlike Cohen's kappa, which can look artificially low (even negative)
    when raters agree on almost everything but one category dominates."""
    a_arr, b_arr = np.array(a), np.array(b)
    po = float(np.mean(a_arr == b_arr))
    p1 = float((np.mean(a_arr == "include") + np.mean(b_arr == "include")) / 2)
    pe = 2 * p1 * (1 - p1)
    ac1 = (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")
    return po, pe, ac1


def _pabak(po: float) -> float:
    """Prevalence-Adjusted Bias-Adjusted Kappa; assumes a 50/50 base rate."""
    return 2 * po - 1


def _kappa_label(k: float) -> str:
    if k != k:  # NaN
        return "n/a"
    if k < 0:
        return "Poor"
    if k <= 0.20:
        return "Slight"
    if k <= 0.40:
        return "Fair"
    if k <= 0.60:
        return "Moderate"
    if k <= 0.80:
        return "Substantial"
    return "Almost perfect"


QC_SECOND_RATER_SYSTEM = "You are a second, independent screening reviewer. Return only valid JSON."

QC_SECOND_RATER_PROMPT_TEMPLATE = """\
Screen this paper for inclusion in a systematic review.

Research questions:
{rqs}

Inclusion criteria:
{inc}

Exclusion criteria:
{exc}

Title: {title}
Abstract: {abstract}

Respond with JSON: {{"decision": "include"|"borderline"|"exclude"}}"""


def compute_qc_second_rater_call_identity(*, cfg: ReviewConfig, record: Record, model: str) -> Dict[str, Any]:
    """The complete immutable call-identity fields `_second_rater_screen()`
    itself publishes onto its `AIUseEvent` -- extracted as a shared builder
    (Codex round-9 review, finding 2) so `reporting.py`'s verification can
    reconstruct the SAME values from the CURRENT configuration and the
    CURRENT record content and compare them field-by-field against a
    loaded event, rather than checking only stage/call_status-counts/
    record-id membership -- which cannot distinguish a substituted event
    (a stale criteria hash, a wrong model, a wrong model_config_id, an
    arbitrary work_item_id, or an unrelated decision output) from the
    genuine one for that exact record. `component_versions`/`access_info`/
    `attempt`/`timestamp` are deliberately excluded, same as the
    search-strategy identity builders' own rationale."""
    inclusion_criteria, exclusion_criteria, research_questions = (
        cfg.inclusion_criteria, cfg.exclusion_criteria, cfg.research_questions)
    rqs = "\n".join(f"- {q}" for q in research_questions) or "(not specified)"
    inc = "\n".join(f"- {c}" for c in inclusion_criteria) or "(not specified)"
    exc = "\n".join(f"- {c}" for c in exclusion_criteria) or "(not specified)"
    user_content = QC_SECOND_RATER_PROMPT_TEMPLATE.format(
        rqs=rqs, inc=inc, exc=exc, title=record.title, abstract=(record.abstract or "")[:1000],
    )
    messages = [
        {"role": "system", "content": QC_SECOND_RATER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    canonical_payload = {
        "messages": messages, "model": model, "temperature": 0.0,
        "response_format": {"type": "json_object"}, "record_id": record.id,
    }
    work_item_id = f"{STAGE}:{sha256_hex(canonical_payload)}"
    prompt_template_text = json.dumps(
        [{"role": "system", "content": QC_SECOND_RATER_SYSTEM}, {"role": "user", "content": QC_SECOND_RATER_PROMPT_TEMPLATE}]
    )
    return {
        "messages": messages,
        "work_item_id": work_item_id,
        "rendered_prompt_hash": sha256_hex(messages),
        "prompt_parameters": {"temperature": 0.0, "response_format": {"type": "json_object"}},
        "input_artifact_refs": [
            _criteria_ref(inclusion_criteria, exclusion_criteria, research_questions),
            _record_content_ref(record),
        ],
        "model": model,
        "provider": "openai",
        "tool_kind": "llm",
        "task_purpose": "Independent second-rater title/abstract screening for inter-rater reliability",
        "model_config_id": "screening_qc_second_rater_v1",
        "prompt_template_version": "v1",
        "prompt_template_text": prompt_template_text,
        "prompt_template_hash": compute_prompt_template_hash(prompt_template_text),
    }


def _second_rater_screen(
    record: Record,
    client: OpenAI,
    model: str,
    cfg: ReviewConfig,
    event_store: AIUseEventStore,
    review_run_id: str,
) -> Tuple[Optional[str], str]:
    """Independent single-record screening call -- deliberately not reusing
    the batched primary-screening code path, so this is a genuinely separate
    rater rather than the same call replayed. Returns
    (decision, this call's own published event_id) -- `decision` is `None`
    on ANY failure (transport or parse) -- never `"exclude"`, which would
    silently fabricate a decision and corrupt the IRR agreement computation
    (INV-06). No fallback exists for this call site. The event_id is
    ALWAYS returned, success or failure -- a genuine attempt was always
    made and published, and reporting's own cross-check (Codex round-5
    review, finding 2) needs it to verify the exact set of records this
    invocation's QC actually called, distinct from every historical
    attempt for the same records under repeated reruns."""
    identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model=model)
    messages = identity["messages"]
    work_item_id = identity["work_item_id"]

    with reserved_attempt(event_store, stage=STAGE, work_item_id=work_item_id) as attempt:
        response_received = False
        resp_model: Optional[str] = None
        decision: Optional[str] = None
        failure_exc: Optional[Exception] = None
        try:
            response = client.chat.completions.create(
                model=model, response_format={"type": "json_object"}, messages=messages, temperature=0.0,
            )
            response_received = True
            resp_model = response.model
            raw = json.loads(response.choices[0].message.content)
            decision = raw.get("decision") or None
            if decision not in ("include", "borderline", "exclude"):
                # Semantic validation, not just presence (codex review): an
                # unrecognized value must never silently reach the IRR
                # computation, where "include"/"borderline"/"exclude" are
                # the only strings _compute_irr()'s binary collapse knows
                # how to interpret -- anything else would otherwise fall
                # through as an unintended "exclude".
                raise ValueError(f"second-rater response has invalid decision {decision!r}")
        except Exception as e:
            logger.warning(f"Second-rater screening failed for {record.id}: {e}")
            failure_exc = e
            decision = None

        call_status = "success" if decision is not None else "failure"
        output_artifact_refs = (
            [f"sha256:{sha256_hex({'decision': decision})}"] if decision is not None else []
        )
        component_versions = {model: resp_model} if response_received else {}

        event = build_ai_use_event(
            review_run_id=review_run_id, stage=STAGE, work_item_id=work_item_id, attempt=attempt,
            tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
            component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose=identity["task_purpose"],
            model_config_id=identity["model_config_id"],
            prompt_template_version=identity["prompt_template_version"],
            prompt_template_text=identity["prompt_template_text"],
            prompt_parameters=identity["prompt_parameters"],
            rendered_prompt_hash=identity["rendered_prompt_hash"],
            input_artifact_refs=identity["input_artifact_refs"],
            output_artifact_refs=output_artifact_refs,
            call_status=call_status, failure_detail=(str(failure_exc) if failure_exc is not None else None),
        )
        event = event_store.publish(event)

    return decision, event.event_id


def compute_qc_eligible_records(records: List[Record]) -> List[Record]:
    """The pure eligibility filter `_compute_irr()` draws its random sample
    from -- only records `screening_node` actually rendered a decision for.
    Shared with `reporting.py`'s own deterministic sample-size reconciliation
    (Codex round-11 review, finding 1) so both sides use the IDENTICAL
    definition of "eligible", rather than reporting guessing at it
    independently."""
    return [r for r in records if r.screening_decision is not None]


# ---------------------------------------------------------------------------
# Restart-safe update-wave QC (review finding 4): screening_qc_node's own
# IRR sampling makes a fresh, unseeded random draw and a fresh, uncached
# provider call on every invocation -- resuming an update-wave QC pass after
# a crash would draw a DIFFERENT sample (calibration_seed defaults to None)
# and re-bill every already-completed second-rater call (reserved_attempt()
# always hands out a NEW attempt slot, by design, for traceability -- it is
# not a result cache). SecondRaterResultStore below gives update-wave
# callers (only) a crash-safe reservation -> paid-call -> durable-
# publication path, mirroring nodes/screening.py's ScreeningBatchResultStore
# discipline exactly. It is never constructed or consulted by
# screening_qc_node's own default (`result_store=None`) code path, so every
# existing, non-update caller is byte-for-byte unaffected.
# ---------------------------------------------------------------------------

class SecondRaterResultIntegrityError(RuntimeError):
    """A persisted `SecondRaterResult` or `QCSampleManifest` (or its
    cross-verification against its own `producing_event_id`) failed
    self-verification or structural validation -- never silently trusted,
    worked around, or dropped."""


_SAFE_QC_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SECOND_RATER_RESULT_SCHEMA_VERSION = "second_rater_result_v1"
_QC_SAMPLE_MANIFEST_SCHEMA_VERSION = "qc_sample_manifest_v1"


class SecondRaterResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["second_rater_result_v1"] = _SECOND_RATER_RESULT_SCHEMA_VERSION
    review_run_id: str
    stage: str
    work_item_id: str
    artifact_id: str
    record_id: str
    producing_event_id: str
    call_status: Literal["success", "failure"]
    decision: Optional[Literal["include", "borderline", "exclude"]]
    content_hash: str

    @model_validator(mode="after")
    def _check_artifact_id_binding(self) -> "SecondRaterResult":
        expected = "secondrater-" + sha256_hex({"review_run_id": self.review_run_id, "work_item_id": self.work_item_id})
        if self.artifact_id != expected:
            raise ValueError(
                f"artifact_id {self.artifact_id!r} does not match the expected "
                f"'secondrater-' + sha256({{review_run_id, work_item_id}}) = {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_decision_call_status_consistency(self) -> "SecondRaterResult":
        if self.call_status == "failure" and self.decision is not None:
            raise ValueError("call_status='failure' must never carry a non-null decision (INV-06)")
        if self.call_status == "success" and self.decision is None:
            raise ValueError("call_status='success' must carry a non-null decision")
        return self

    def compute_content_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "content_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_content_hash()
        if self.content_hash != expected:
            raise SecondRaterResultIntegrityError(
                f"SecondRaterResult artifact_id={self.artifact_id!r} failed content_hash self-verification: "
                f"stored {self.content_hash!r} != recomputed {expected!r} -- file corrupted or tampered"
            )


def build_second_rater_result(
    *, review_run_id: str, stage: str, work_item_id: str, record_id: str,
    producing_event_id: str, call_status: str, decision: Optional[str],
) -> SecondRaterResult:
    artifact_id = "secondrater-" + sha256_hex({"review_run_id": review_run_id, "work_item_id": work_item_id})
    base = dict(
        schema_version=_SECOND_RATER_RESULT_SCHEMA_VERSION, review_run_id=review_run_id, stage=stage,
        work_item_id=work_item_id, artifact_id=artifact_id, record_id=record_id,
        producing_event_id=producing_event_id, call_status=call_status, decision=decision,
    )
    content_hash = SecondRaterResult.model_construct(**base, content_hash="").compute_content_hash()
    return SecondRaterResult.model_validate(dict(base, content_hash=content_hash))


def _load_second_rater_result_file(path: Path) -> SecondRaterResult:
    return SecondRaterResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


class QCSampleManifest(BaseModel):
    """The crash-safe RESERVATION for one QC pass's IRR sample -- written
    atomically before any paid second-rater call is made, and verified
    (never silently recomputed) on every resume. Binds the ordered
    eligible-record pool (ids + a hash of their actual content), the QC
    configuration that determines what gets sampled/called (model, sample
    size, seed, prompt template version, criteria), and the deterministic
    `sampled_record_ids` that reservation pins for the rest of this QC
    pass's lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["qc_sample_manifest_v1"] = _QC_SAMPLE_MANIFEST_SCHEMA_VERSION
    review_run_id: str
    stage: str
    eligible_record_ids: Tuple[str, ...]
    eligible_pool_hash: str
    second_rater_model: str
    calibration_sample_size: int
    calibration_seed: int
    prompt_template_version: str
    criteria_hash: str
    sampled_record_ids: Tuple[str, ...]
    manifest_hash: str

    def compute_manifest_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "manifest_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_manifest_hash()
        if self.manifest_hash != expected:
            raise SecondRaterResultIntegrityError(
                f"QCSampleManifest for review_run_id={self.review_run_id!r} failed manifest_hash "
                f"self-verification: stored {self.manifest_hash!r} != recomputed {expected!r} -- file "
                "corrupted or tampered"
            )

    def fingerprint(self) -> Tuple[Any, ...]:
        """The subset of fields that must be IDENTICAL between a resumed
        invocation's freshly recomputed manifest and the one already on
        disk for that resume to be safe -- deliberately excludes
        `sampled_record_ids`/`manifest_hash` (the very values reservation
        is pinning), so a resume compares only the INPUTS that determine
        the sample, not the sample's own already-decided output."""
        return (
            self.eligible_record_ids, self.eligible_pool_hash, self.second_rater_model,
            self.calibration_sample_size, self.calibration_seed, self.prompt_template_version,
            self.criteria_hash,
        )


class QCSampleManifestMismatchError(SecondRaterResultIntegrityError):
    """A resumed QC pass recomputed a manifest whose fingerprint does not
    match the one already reserved on disk -- the eligible pool or QC
    configuration changed since the original reservation. Never silently
    draw a different sample or reuse a stale one."""


def _build_qc_sample_manifest(
    *, review_run_id: str, stage: str, eligible_records: List[Record], cfg: ReviewConfig, model: str,
) -> QCSampleManifest:
    qc_cfg = cfg.screening_qc
    if qc_cfg.calibration_seed is None:
        raise ValueError(
            "update-wave QC requires a non-null screening_qc.calibration_seed -- an unseeded sample "
            "could silently differ across a restart, defeating this store's restart-safety guarantee"
        )
    eligible_record_ids = tuple(r.id for r in eligible_records)
    eligible_pool_hash = sha256_hex([_record_content_ref(r) for r in eligible_records])
    criteria_hash = _criteria_ref(cfg.inclusion_criteria, cfg.exclusion_criteria, cfg.research_questions)
    rng = random.Random(qc_cfg.calibration_seed)
    sample = rng.sample(eligible_records, min(qc_cfg.calibration_sample_size, len(eligible_records)))
    sampled_record_ids = tuple(r.id for r in sample)
    base = dict(
        schema_version=_QC_SAMPLE_MANIFEST_SCHEMA_VERSION, review_run_id=review_run_id, stage=stage,
        eligible_record_ids=eligible_record_ids, eligible_pool_hash=eligible_pool_hash,
        second_rater_model=model, calibration_sample_size=qc_cfg.calibration_sample_size,
        calibration_seed=qc_cfg.calibration_seed, prompt_template_version="v1", criteria_hash=criteria_hash,
        sampled_record_ids=sampled_record_ids,
    )
    manifest_hash = QCSampleManifest.model_construct(**base, manifest_hash="").compute_manifest_hash()
    return QCSampleManifest.model_validate(dict(base, manifest_hash=manifest_hash))


class SecondRaterAdjudicationRequiredError(SecondRaterResultIntegrityError):
    """A prior second-rater call attempt for this work item was reserved
    but its outcome (billed? succeeded? failed?) was never durably
    recorded -- a process death between the real provider call and its
    event/result publication. Whether that call actually happened and
    whether it was billed is now genuinely unknown, so this work item is
    never auto-retried. A human must inspect the archived
    `.interrupted_unknown.json` record and explicitly authorize a retry
    before any further attempt is made (mirrors
    `extraction/resolution_suggestions.py`'s `retry_work_item_ids` gate --
    no such authorization parameter exists yet anywhere in the QC/
    update-wave call chain, so today the only correct behavior is this
    unconditional raise)."""


_SECOND_RATER_RESERVED_MARKER_SCHEMA_VERSION = "second_rater_reserved_marker_v1"
_SECOND_RATER_INTERRUPTED_RECORD_SCHEMA_VERSION = "second_rater_interrupted_record_v1"
_RESERVED_MARKER_RE = re.compile(r"^(.+)__attempt-(\d+)\.reserved\.json$")


class SecondRaterReservedMarker(BaseModel):
    """Crash-safe pre-call reservation for one second-rater attempt --
    written+fsynced BEFORE the real provider call, under
    `reserved_attempt()`'s own per-work-item lock. Mirrors
    `extraction/resolution_schemas.py::ReservedAttemptMarker`'s structure
    and self-validation discipline, scoped to this store's own identity
    fields (review_run_id/work_item_id/artifact_id/record_id) rather than
    extraction-resolution's."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["second_rater_reserved_marker_v1"] = _SECOND_RATER_RESERVED_MARKER_SCHEMA_VERSION
    review_run_id: str
    stage: str
    work_item_id: str
    artifact_id: str
    record_id: str
    attempt: int = Field(ge=0)
    event_id: str
    reserved_at: str
    marker_hash: str

    @model_validator(mode="after")
    def _check_bindings(self) -> "SecondRaterReservedMarker":
        expected_artifact_id = "secondrater-" + sha256_hex(
            {"review_run_id": self.review_run_id, "work_item_id": self.work_item_id}
        )
        if self.artifact_id != expected_artifact_id:
            raise ValueError(
                f"artifact_id {self.artifact_id!r} does not match the expected "
                f"'secondrater-' + sha256({{review_run_id, work_item_id}}) = {expected_artifact_id!r}"
            )
        expected_event_id = compute_event_id(
            review_run_id=self.review_run_id, stage=self.stage, work_item_id=self.work_item_id, attempt=self.attempt,
        )
        if self.event_id != expected_event_id:
            raise ValueError(
                f"event_id {self.event_id!r} does not match the expected compute_event_id(...) = "
                f"{expected_event_id!r} for this marker's own review_run_id/stage/work_item_id/attempt"
            )
        return self

    def compute_marker_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "marker_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_marker_hash()
        if self.marker_hash != expected:
            raise SecondRaterResultIntegrityError(
                f"SecondRaterReservedMarker artifact_id={self.artifact_id!r} attempt={self.attempt} failed "
                f"marker_hash self-verification: stored {self.marker_hash!r} != recomputed {expected!r} -- "
                "file corrupted or tampered"
            )


def build_second_rater_reserved_marker(
    *, review_run_id: str, stage: str, work_item_id: str, record_id: str, attempt: int,
) -> SecondRaterReservedMarker:
    artifact_id = "secondrater-" + sha256_hex({"review_run_id": review_run_id, "work_item_id": work_item_id})
    event_id = compute_event_id(review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=attempt)
    base = dict(
        schema_version=_SECOND_RATER_RESERVED_MARKER_SCHEMA_VERSION, review_run_id=review_run_id, stage=stage,
        work_item_id=work_item_id, artifact_id=artifact_id, record_id=record_id, attempt=attempt, event_id=event_id,
        reserved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    marker_hash = SecondRaterReservedMarker.model_construct(**base, marker_hash="").compute_marker_hash()
    return SecondRaterReservedMarker.model_validate(dict(base, marker_hash=marker_hash))


class SecondRaterInterruptedRecord(BaseModel):
    """Archive of a reservation whose outcome could not be reconciled to
    any durable, verified terminal result -- the adjudication-required
    slot. Mirrors `extraction/resolution_schemas.py::InterruptedCallRecord`.
    Never auto-retried; never reused for a new attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["second_rater_interrupted_record_v1"] = _SECOND_RATER_INTERRUPTED_RECORD_SCHEMA_VERSION
    review_run_id: str
    stage: str
    work_item_id: str
    artifact_id: str
    record_id: str
    attempt: int = Field(ge=0)
    event_id: str
    reserved_at: str
    archived_at: str
    reason: str
    content_hash: str

    def compute_content_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "content_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_content_hash()
        if self.content_hash != expected:
            raise SecondRaterResultIntegrityError(
                f"SecondRaterInterruptedRecord artifact_id={self.artifact_id!r} attempt={self.attempt} failed "
                f"content_hash self-verification: stored {self.content_hash!r} != recomputed {expected!r} -- "
                "file corrupted or tampered"
            )


def build_second_rater_interrupted_record(*, marker: SecondRaterReservedMarker, reason: str) -> SecondRaterInterruptedRecord:
    base = dict(
        schema_version=_SECOND_RATER_INTERRUPTED_RECORD_SCHEMA_VERSION, review_run_id=marker.review_run_id,
        stage=marker.stage, work_item_id=marker.work_item_id, artifact_id=marker.artifact_id,
        record_id=marker.record_id, attempt=marker.attempt, event_id=marker.event_id,
        reserved_at=marker.reserved_at, archived_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), reason=reason,
    )
    content_hash = SecondRaterInterruptedRecord.model_construct(**base, content_hash="").compute_content_hash()
    return SecondRaterInterruptedRecord.model_validate(dict(base, content_hash=content_hash))


class SecondRaterResultStore:
    """Restart-safe home for one update-wave QC pass's sample reservation
    (`QCSampleManifest`) and per-record second-rater results
    (`SecondRaterResult`). Only ever constructed/consulted by update-wave
    callers that explicitly pass it into `_compute_irr()` --
    `screening_qc_node`'s own default path never touches this class."""

    def __init__(self, *, output_dir: Path, review_run_id: str, event_store: AIUseEventStore) -> None:
        if not _SAFE_QC_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        results_root = Path(output_dir) / "screening_qc_second_rater_results"
        run_dir = results_root / review_run_id
        if not run_dir.resolve().is_relative_to(results_root.resolve()):
            raise ValueError(
                f"review_run_id {review_run_id!r} resolves outside the screening_qc_second_rater_results "
                "root -- refusing to construct store"
            )
        self.review_run_id = review_run_id
        self.run_dir = run_dir
        self.event_store = event_store

    def _manifest_path(self) -> Path:
        return self.run_dir / "sample_manifest.json"

    def _atomic_write(self, path: Path, payload: bytes) -> bool:
        """True iff THIS call's bytes are now durably on disk (a genuine
        first write); False on a benign idempotent-retry collision where
        an existing file already exists (the caller compares decoded
        content, not raw bytes, to decide whether that collision is
        benign or a genuine mismatch)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".tmp.{uuid.uuid4().hex}.json"
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            return False
        else:
            _fsync_dir(path.parent)
            return True
        finally:
            tmp_path.unlink(missing_ok=True)

    def _reserved_marker_path(self, artifact_id: str, attempt: int) -> Path:
        return self.run_dir / f"{artifact_id}__attempt-{attempt}.reserved.json"

    def _completed_marker_path(self, artifact_id: str, attempt: int) -> Path:
        return self.run_dir / f"{artifact_id}__attempt-{attempt}.completed.json"

    def _interrupted_marker_path(self, artifact_id: str, attempt: int) -> Path:
        return self.run_dir / f"{artifact_id}__attempt-{attempt}.interrupted_unknown.json"

    def _marker_exists(self, artifact_id: str, attempt: int) -> bool:
        """The `extra_occupied` predicate passed to `reserved_attempt()` --
        an attempt slot with ANY on-disk marker trace (reserved, completed,
        or archived-interrupted) must never be handed out again, even
        though no `AIUseEvent` may exist there."""
        return (
            self._reserved_marker_path(artifact_id, attempt).exists()
            or self._completed_marker_path(artifact_id, attempt).exists()
            or self._interrupted_marker_path(artifact_id, attempt).exists()
        )

    def reject_if_interrupted(self, artifact_id: str) -> None:
        """The adjudication gate (review finding 1): raises unconditionally
        if ANY attempt for this artifact_id was archived as
        interrupted-unknown -- never auto-retried."""
        matches = sorted(self.run_dir.glob(f"{artifact_id}__attempt-*.interrupted_unknown.json"))
        if matches:
            raise SecondRaterAdjudicationRequiredError(
                f"work item for artifact_id={artifact_id!r} has {len(matches)} interrupted-unknown "
                f"second-rater attempt(s) on disk ({[p.name for p in matches]!r}) -- a prior call's "
                "outcome was never durably recorded and must be adjudicated by a human before any "
                "further attempt is made"
            )

    def reserve_attempt(self, *, work_item_id: str, artifact_id: str, record_id: str, attempt: int) -> SecondRaterReservedMarker:
        """Crash-safe step 1: durably records that THIS attempt is about to
        make a real provider call, before that call is dispatched."""
        marker = build_second_rater_reserved_marker(
            review_run_id=self.review_run_id, stage=STAGE, work_item_id=work_item_id, record_id=record_id,
            attempt=attempt,
        )
        path = self._reserved_marker_path(artifact_id, attempt)
        payload = marker.model_dump_json(indent=2).encode("utf-8")
        if self._atomic_write(path, payload):
            return marker
        existing = SecondRaterReservedMarker.model_validate(json.loads(path.read_text(encoding="utf-8")))
        existing.verify_hashes()
        if existing.event_id != marker.event_id:
            raise SecondRaterResultIntegrityError(
                f"SecondRaterReservedMarker collision at {path}: persisted event_id {existing.event_id!r} "
                f"!= new marker's {marker.event_id!r}. This should be structurally unreachable under "
                "reserved_attempt()'s own per-work-item lock."
            )
        return existing

    def mark_attempt_completed(self, *, artifact_id: str, attempt: int) -> None:
        """Crash-safe step 3: renames the reservation marker to a completed
        marker AFTER the terminal result/event have both been durably
        published -- called only once both `event_store.publish()` and
        `self.publish()` have returned."""
        reserved_path = self._reserved_marker_path(artifact_id, attempt)
        completed_path = self._completed_marker_path(artifact_id, attempt)
        if completed_path.exists():
            return
        os.replace(reserved_path, completed_path)
        _fsync_dir(completed_path.parent)

    def reconcile(self) -> List[SecondRaterInterruptedRecord]:
        """Startup reconciliation (review finding 1): scans orphaned
        `.reserved.json` markers with no matching completed marker, and
        archives each to `.interrupted_unknown.json` -- a slot that is
        never reused/auto-retried. Skips (does not archive) any marker
        whose per-work-item lock is currently held by a live process,
        mirroring `extraction/resolution_suggestions.py::
        reconcile_suggestion_attempts()` exactly."""
        archived: List[SecondRaterInterruptedRecord] = []
        if not self.run_dir.is_dir():
            return archived
        for path in sorted(self.run_dir.glob("*__attempt-*.reserved.json")):
            m = _RESERVED_MARKER_RE.match(path.name)
            if not m:
                continue
            artifact_id, attempt_str = m.group(1), int(m.group(2))
            marker = SecondRaterReservedMarker.model_validate(json.loads(path.read_text(encoding="utf-8")))
            marker.verify_hashes()
            lock_path = attempt_lock_path(self.event_store, stage=STAGE, work_item_id=marker.work_item_id)
            try:
                with kernel_lock(
                    lock_path, error_cls=AIEventLockError,
                    description=f"reconcile:{STAGE}:{marker.work_item_id}", timeout=0.5,
                ):
                    if not path.is_file():
                        continue  # completed/archived concurrently
                    existing_result = self.load(artifact_id)
                    if existing_result is not None and existing_result.producing_event_id == marker.event_id:
                        self.mark_attempt_completed(artifact_id=artifact_id, attempt=attempt_str)
                        continue
                    reason = (
                        "no durably-published result matches this reservation's event_id -- the real "
                        "provider call's outcome (billed? succeeded? failed?) was never recorded"
                        if existing_result is None
                        else "a different attempt's result was published for this work item; this "
                        "reservation's own outcome was never recorded"
                    )
                    record = build_second_rater_interrupted_record(marker=marker, reason=reason)
                    interrupted_path = self._interrupted_marker_path(artifact_id, attempt_str)
                    payload = record.model_dump_json(indent=2).encode("utf-8")
                    if not self._atomic_write(interrupted_path, payload):
                        existing_archive = SecondRaterInterruptedRecord.model_validate(
                            json.loads(interrupted_path.read_text(encoding="utf-8"))
                        )
                        existing_archive.verify_hashes()
                        record = existing_archive
                    path.unlink(missing_ok=True)
                    _fsync_dir(path.parent)
                    archived.append(record)
            except AIEventLockError:
                continue  # a live process still owns this reservation -- never archive out from under it
        return archived

    def reserve_sample(
        self, *, eligible_records: List[Record], cfg: ReviewConfig, model: str, stage: str = STAGE,
    ) -> Tuple[Record, ...]:
        """Crash-safe reservation step: computes this invocation's own
        manifest from the CURRENT eligible pool/config, then either
        publishes it (first call for this run) or verifies a persisted
        manifest's fingerprint still matches it (resume) -- raising
        `QCSampleManifestMismatchError` rather than silently drawing a NEW
        sample when the eligible pool or QC configuration changed since
        the original reservation. Returns the pinned sample, in manifest
        order, as real `Record` objects."""
        fresh = _build_qc_sample_manifest(
            review_run_id=self.review_run_id, stage=stage, eligible_records=eligible_records, cfg=cfg, model=model,
        )
        path = self._manifest_path()
        if path.is_file():
            existing = QCSampleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
            existing.verify_hashes()
            if existing.review_run_id != self.review_run_id or existing.stage != stage:
                raise SecondRaterResultIntegrityError(
                    f"QC sample manifest at {path} has review_run_id/stage="
                    f"{existing.review_run_id!r}/{existing.stage!r}, expected this store's own "
                    f"{self.review_run_id!r}/{stage!r} -- refusing to trust a foreign manifest"
                )
            if existing.fingerprint() != fresh.fingerprint():
                raise QCSampleManifestMismatchError(
                    f"QC sample manifest at {path} no longer matches the eligible pool/QC configuration "
                    "this invocation just recomputed -- the eligible pool or QC configuration changed "
                    "since this run's sample was first reserved. Refusing to silently draw a different "
                    "sample or reuse a stale one."
                )
            manifest = existing
        else:
            payload = fresh.model_dump_json(indent=2).encode("utf-8")
            if not self._atomic_write(path, payload):
                existing = QCSampleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
                existing.verify_hashes()
                if existing.fingerprint() != fresh.fingerprint():
                    raise QCSampleManifestMismatchError(
                        f"QC sample manifest for review_run_id={self.review_run_id!r} was concurrently "
                        "reserved with a different fingerprint -- refusing to trust either."
                    )
                manifest = existing
            else:
                manifest = fresh

        by_id = {r.id: r for r in eligible_records}
        return tuple(by_id[rid] for rid in manifest.sampled_record_ids)

    def _artifact_path(self, artifact_id: str) -> Path:
        return self.run_dir / f"{artifact_id}.json"

    def _verify_result(self, result: SecondRaterResult, *, path: Optional[Path] = None) -> None:
        result.verify_hashes()
        location = f"checkpoint at {path}" if path is not None else f"checkpoint artifact_id={result.artifact_id!r}"
        if result.review_run_id != self.review_run_id:
            raise SecondRaterResultIntegrityError(
                f"{location} has review_run_id={result.review_run_id!r}, expected this store's own "
                f"{self.review_run_id!r} -- refusing to trust a foreign-run checkpoint"
            )
        if path is not None and path.stem != result.artifact_id:
            raise SecondRaterResultIntegrityError(
                f"{location} has artifact_id={result.artifact_id!r}, which does not match its own "
                f"filename stem {path.stem!r} -- refusing to trust a renamed or mismatched file"
            )
        event = self.event_store.load_event(result.producing_event_id)
        if event.stage != result.stage or event.work_item_id != result.work_item_id or event.call_status != result.call_status:
            raise SecondRaterResultIntegrityError(
                f"{location}'s producing_event_id={result.producing_event_id!r} refers to an event with "
                f"stage/work_item_id/call_status ({event.stage!r}/{event.work_item_id!r}/{event.call_status!r}) "
                f"that does not match this checkpoint's own ({result.stage!r}/{result.work_item_id!r}/"
                f"{result.call_status!r})"
            )
        expected_output_hash = (
            f"sha256:{sha256_hex({'decision': result.decision})}" if result.decision is not None else None
        )
        if expected_output_hash is not None and expected_output_hash not in event.output_artifact_refs:
            raise SecondRaterResultIntegrityError(
                f"{location}'s own decision hashes to {expected_output_hash!r}, which is not present in "
                f"its producing event's output_artifact_refs {event.output_artifact_refs!r}"
            )

    def load(self, artifact_id: str) -> Optional[SecondRaterResult]:
        path = self._artifact_path(artifact_id)
        if not path.is_file():
            return None
        result = _load_second_rater_result_file(path)
        self._verify_result(result, path=path)
        return result

    def publish(self, result: SecondRaterResult) -> SecondRaterResult:
        self._verify_result(result)
        path = self._artifact_path(result.artifact_id)
        payload = result.model_dump_json(indent=2).encode("utf-8")
        if self._atomic_write(path, payload):
            return result
        existing = _load_second_rater_result_file(path)
        self._verify_result(existing, path=path)
        if existing.content_hash == result.content_hash:
            return existing
        raise SecondRaterResultIntegrityError(
            f"SecondRaterResult collision for artifact_id={result.artifact_id!r}: persisted content_hash "
            f"{existing.content_hash!r} != new result's {result.content_hash!r}. This should be "
            "structurally unreachable under reserved_attempt()'s own lock plus the mandatory "
            "double-checked cache lookup."
        )


def _get_or_compute_second_rater_result(
    record: Record, client: OpenAI, model: str, cfg: ReviewConfig,
    event_store: AIUseEventStore, result_store: SecondRaterResultStore, review_run_id: str,
) -> SecondRaterResult:
    """Restart-safe, get-or-compute counterpart to `_second_rater_screen` --
    mirrors `nodes/screening.py`'s `_get_or_compute_batch_result`'s
    check-before-lock / `reserved_attempt` / re-check-under-lock / call /
    publish discipline exactly, so a resumed QC pass skips every record
    this store already has a durable result for and makes a real provider
    call only for the records still missing one."""
    identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model=model)
    messages = identity["messages"]
    work_item_id = identity["work_item_id"]
    artifact_id = "secondrater-" + sha256_hex({"review_run_id": review_run_id, "work_item_id": work_item_id})

    cached = result_store.load(artifact_id)
    if cached is not None:
        return cached

    # Adjudication gate (review finding 1): a prior attempt for this work
    # item was reserved but its outcome was never durably recorded (a
    # process death after the paid call but before event/result
    # publication). Never auto-retried.
    result_store.reject_if_interrupted(artifact_id)

    with reserved_attempt(
        event_store, stage=STAGE, work_item_id=work_item_id,
        extra_occupied=lambda a: result_store._marker_exists(artifact_id, a),
    ) as attempt:
        # Mandatory re-check, now holding the lock (F15) -- checking only
        # before the lock lets two concurrent callers both miss and both
        # call the provider.
        cached = result_store.load(artifact_id)
        if cached is not None:
            return cached
        result_store.reject_if_interrupted(artifact_id)

        # Crash-safe step 1 (review finding 1): durably record that THIS
        # attempt is about to make a real provider call, BEFORE dispatching
        # it -- so a crash between a successful/failed response and its
        # event/result publication leaves durable evidence on disk instead
        # of silently vanishing.
        result_store.reserve_attempt(
            work_item_id=work_item_id, artifact_id=artifact_id, record_id=record.id, attempt=attempt,
        )

        response_received = False
        resp_model: Optional[str] = None
        decision: Optional[str] = None
        failure_exc: Optional[Exception] = None
        try:
            response = client.chat.completions.create(
                model=model, response_format={"type": "json_object"}, messages=messages, temperature=0.0,
            )
            response_received = True
            resp_model = response.model
            raw = json.loads(response.choices[0].message.content)
            decision = raw.get("decision") or None
            if decision not in ("include", "borderline", "exclude"):
                raise ValueError(f"second-rater response has invalid decision {decision!r}")
        except Exception as e:
            logger.warning(f"Second-rater screening failed for {record.id}: {e}")
            failure_exc = e
            decision = None

        call_status = "success" if decision is not None else "failure"
        output_artifact_refs = (
            [f"sha256:{sha256_hex({'decision': decision})}"] if decision is not None else []
        )
        component_versions = {model: resp_model} if response_received else {}

        event = build_ai_use_event(
            review_run_id=review_run_id, stage=STAGE, work_item_id=work_item_id, attempt=attempt,
            tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
            component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose=identity["task_purpose"],
            model_config_id=identity["model_config_id"],
            prompt_template_version=identity["prompt_template_version"],
            prompt_template_text=identity["prompt_template_text"],
            prompt_parameters=identity["prompt_parameters"],
            rendered_prompt_hash=identity["rendered_prompt_hash"],
            input_artifact_refs=identity["input_artifact_refs"],
            output_artifact_refs=output_artifact_refs,
            call_status=call_status, failure_detail=(str(failure_exc) if failure_exc is not None else None),
        )
        event = event_store.publish(event)

        result = build_second_rater_result(
            review_run_id=review_run_id, stage=STAGE, work_item_id=work_item_id, record_id=record.id,
            producing_event_id=event.event_id, call_status=call_status, decision=decision,
        )
        published = result_store.publish(result)
        # Crash-safe step 3: only rename reserved -> completed AFTER both
        # the event and the result are durably published.
        result_store.mark_attempt_completed(artifact_id=artifact_id, attempt=attempt)
        return published


def _compute_irr(
    records: List[Record],
    client: OpenAI,
    model: str,
    cfg: ReviewConfig,
    sample_size: int,
    event_store: AIUseEventStore,
    review_run_id: str,
    calibration_seed: Optional[int] = None,
    result_store: Optional[SecondRaterResultStore] = None,
) -> Dict[str, Any]:
    """Draws a random sample of already-screened records, re-screens them
    with an independent second rater, and reports agreement. Binary collapse
    (include/borderline -> "include", exclude -> "exclude") matches how
    screening_node itself treats decisions in automated mode -- kappa/AC1 on
    the raw 3-way decision would conflate "clearly excluded" with "borderline
    but nudged one way," which isn't the distinction IRR is meant to check.

    A failed second-rater call (returns None) is excluded from BOTH rater
    lists, not scored as a fabricated "exclude" -- `n_attempted`/
    `n_valid_pairs`/`n_failed` are reported separately so a caller can see
    the difference between "we sampled N records" and "N of those produced
    a usable second-rater judgment." Every agreement statistic is `None`
    (never computed, never a divide-by-zero) when zero valid pairs remain.

    `producing_event_ids` (Codex round-5 review, finding 2): the exact
    `AIUseEvent.event_id` THIS invocation published for every sampled
    record, success or failure alike -- a genuine call is always attempted
    and published per sampled record, so this list always has exactly
    `n_attempted` entries. Lets reporting's own cross-check verify against
    the CURRENT invocation's own events specifically, rather than the
    union of every historical `screening_qc_second_rater` event ever
    published under this review_run_id (which would wrongly include/
    exclude events from an earlier rerun with a DIFFERENT random sample).

    `calibration_seed` (Track 2 plan §4.4): `None` (default) preserves
    production's unseeded `random.sample(...)`; when set, the module-level
    `random` instance is never touched -- a dedicated `random.Random(seed)`
    draws the sample instead, so a rerun over the same eligible pool is
    reproducible.

    `result_store` (review finding 4): `None` (default) preserves every
    existing caller's exact unseeded/uncached behavior -- every sample draw
    and provider call happens exactly as before. When set (update-wave
    callers only), the sample is instead reserved crash-safely via
    `result_store.reserve_sample()` (which requires a non-null
    `calibration_seed`, raising rather than silently drawing an unseeded
    sample), and each record's second-rater call is routed through
    `result_store`'s get-or-compute cache, so a resumed pass skips every
    record already durably resolved and calls the provider only for the
    records still missing a result."""
    pool = compute_qc_eligible_records(records)
    if result_store is not None:
        # Startup reconciliation (review finding 1), run once before any
        # dispatch -- archives orphaned reservations into
        # `.interrupted_unknown.json` so `reject_if_interrupted()` can see
        # them before this pass makes a single new provider call.
        result_store.reconcile()
        sample = list(result_store.reserve_sample(eligible_records=pool, cfg=cfg, model=model))
    else:
        rng = random.Random(calibration_seed) if calibration_seed is not None else random
        sample = rng.sample(pool, min(sample_size, len(pool)))
    n_attempted = len(sample)

    if n_attempted == 0:
        return {
            "n_attempted": 0, "n_valid_pairs": 0, "n_failed": 0,
            "note": "No screened records available to sample.",
            "producing_event_ids": [],
        }

    first_rater: List[str] = []
    second_rater: List[str] = []
    n_failed = 0
    producing_event_ids: List[str] = []
    for record in sample:
        if result_store is not None:
            result = _get_or_compute_second_rater_result(
                record, client, model, cfg, event_store, result_store, review_run_id,
            )
            second, event_id = result.decision, result.producing_event_id
        else:
            second, event_id = _second_rater_screen(
                record, client, model, cfg, event_store, review_run_id,
            )
        producing_event_ids.append(event_id)
        record.second_rater_decision = second
        if second is None:
            n_failed += 1
            continue
        first_rater.append("include" if record.screening_decision in ("include", "borderline") else "exclude")
        second_rater.append("include" if second in ("include", "borderline") else "exclude")

    n_valid_pairs = len(first_rater)

    if n_valid_pairs == 0:
        return {
            "n_attempted": n_attempted, "n_valid_pairs": 0, "n_failed": n_failed,
            "pct_include_avg": None, "pct_agreement": None, "cohens_kappa": None,
            "pabak": None, "gwet_ac1": None, "level": None,
            "note": "Every second-rater call failed; no valid pair to compute agreement over.",
            "sampled_record_ids": [r.id for r in sample],
            "producing_event_ids": producing_event_ids,
        }

    po, _pe, ac1 = _gwet_ac1(first_rater, second_rater)
    pb = _pabak(po)
    try:
        kappa = float(cohen_kappa_score(first_rater, second_rater))
    except Exception:
        kappa = float("nan")

    pct_include = (first_rater.count("include") + second_rater.count("include")) / (2 * n_valid_pairs) * 100

    return {
        "n_attempted": n_attempted,
        "n_valid_pairs": n_valid_pairs,
        "n_failed": n_failed,
        "pct_include_avg": round(pct_include, 1),
        "pct_agreement": round(po * 100, 1),
        "cohens_kappa": round(kappa, 3) if kappa == kappa else None,
        "pabak": round(pb, 3),
        "gwet_ac1": round(ac1, 3) if ac1 == ac1 else None,
        "level": _kappa_label(ac1),
        "sampled_record_ids": [r.id for r in sample],
        "producing_event_ids": producing_event_ids,
    }


# ---------------------------------------------------------------------------
# Topic saturation
# ---------------------------------------------------------------------------

def _michaelis_menten(n, c_max, k):
    return c_max * n / (k + n)


def _rarefaction_curve(
    topic_sets: List[Set[str]], n_bootstrap: int, sat_levels: List[float]
) -> Optional[Dict[str, Any]]:
    """Bootstrapped topic-accumulation curve: shuffle record order many
    times, track cumulative distinct topics discovered, fit a Michaelis-
    Menten saturation curve to estimate the total topic pool size and how
    many records it takes to see 80/90/95% of it."""
    n = len(topic_sets)
    if n < 10:
        return None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        curve_fit = None

    rng = np.random.default_rng(42)
    curves = np.zeros((n_bootstrap, n))
    indices = list(range(n))
    for b in range(n_bootstrap):
        perm = rng.permutation(indices)
        seen: Set[str] = set()
        for i, idx in enumerate(perm):
            seen |= topic_sets[idx]
            curves[b, i] = len(seen)

    mean_curve = curves.mean(axis=0)
    ns = np.arange(1, n + 1)

    sat_n: Dict[float, int] = {}
    c_max_est, k_est = float(mean_curve[-1]), float("nan")
    if curve_fit is not None:
        try:
            popt, _ = curve_fit(
                _michaelis_menten, ns, mean_curve,
                p0=[mean_curve[-1] * 1.2, n / 2],
                bounds=([0, 0], [np.inf, np.inf]), maxfev=5000,
            )
            c_max_est, k_est = float(popt[0]), float(popt[1])
            for lvl in sat_levels:
                target = lvl * c_max_est
                sat_n[lvl] = int(np.ceil(k_est * target / (c_max_est - target))) if c_max_est > target else n
        except Exception as e:
            logger.debug(f"Michaelis-Menten fit failed: {e}")
    if not sat_n:
        sat_n = {lvl: n for lvl in sat_levels}

    return {
        "n_records": n,
        "total_unique_topics": int(mean_curve[-1]),
        "c_max_est": round(c_max_est, 1),
        "k_est": round(k_est, 1) if k_est == k_est else None,
        "sat_n_curve_fit": sat_n,
    }


def _bootstrapped_saturation_point(
    topic_sets: List[Set[str]], n_bootstrap: int, sat_levels: List[float]
) -> Optional[Dict[str, Any]]:
    """Empirical, curve-fit-free complement to the MM estimate above: for
    each bootstrap permutation, find the first n at which cumulative unique
    topics reaches each fraction of the FINAL observed total, then report
    the median and 5-95% range across permutations. More robust than the MM
    extrapolation when the topic space hasn't clearly leveled off yet."""
    n = len(topic_sets)
    if n < 10:
        return None
    final_total = len(set().union(*topic_sets))
    if final_total == 0:
        return None

    rng = np.random.default_rng(42)
    indices = list(range(n))
    hits: Dict[float, List[int]] = {lvl: [] for lvl in sat_levels}

    for _ in range(n_bootstrap):
        perm = rng.permutation(indices)
        seen: Set[str] = set()
        reached: Dict[float, Optional[int]] = {lvl: None for lvl in sat_levels}
        for i, idx in enumerate(perm):
            seen |= topic_sets[idx]
            coverage = len(seen) / final_total
            for lvl in sat_levels:
                if reached[lvl] is None and coverage >= lvl:
                    reached[lvl] = i + 1
        for lvl in sat_levels:
            hits[lvl].append(reached[lvl] or n)

    summary = {}
    for lvl in sat_levels:
        arr = np.array(hits[lvl])
        summary[lvl] = {
            "median_n": int(np.median(arr)),
            "ci_lo": int(np.percentile(arr, 5)),
            "ci_hi": int(np.percentile(arr, 95)),
        }
    return summary


def _jaccard_convergence(
    topic_sets: List[Set[str]], sample_sizes: List[int], n_reps: int = 20
) -> Dict[str, Any]:
    """For each sample size n, draw two independent random samples of n
    records (without replacement) and measure the Jaccard overlap of their
    discovered topic sets -- answers "if two reviewers each screened a
    different n papers, how much topic overlap would they see?" A common
    rule of thumb is the n at which mean Jaccard first reaches 0.80."""
    n = len(topic_sets)
    rng = np.random.default_rng(42)
    results: Dict[int, float] = {}
    for size in sample_sizes:
        if size >= n or size <= 0:
            continue
        jacs = []
        for _ in range(n_reps):
            idx1 = rng.choice(n, size=size, replace=False)
            idx2 = rng.choice(n, size=size, replace=False)
            s1: Set[str] = set().union(*(topic_sets[i] for i in idx1))
            s2: Set[str] = set().union(*(topic_sets[i] for i in idx2))
            union = len(s1 | s2)
            jacs.append(len(s1 & s2) / union if union else 0.0)
        results[size] = round(float(np.mean(jacs)), 3)
    n_80 = next((s for s in sorted(results) if results[s] >= 0.80), None)
    return {"jaccard_by_n": results, "n_jaccard_80": n_80}


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def _default_screening_qc_client_factory() -> Any:
    """Today's exact unpatched behavior: a bare client with no
    ``max_retries`` override -- see ``_default_screening_client_factory``
    in ``nodes/screening.py`` (same rationale) and
    ``evaluation/update_cost_estimate.py``'s ``SCREENING_QC_MAX_ATTEMPTS``,
    which prices this default explicitly."""
    return OpenAI(api_key=resolve_openai_api_key())


def screening_qc_node(
    state: ReviewState, *, client_factory: Callable[[], Any] = _default_screening_qc_client_factory,
    result_store: Optional[SecondRaterResultStore] = None,
) -> Dict[str, Any]:
    """
    Quality control for screening_node's decisions:
      1. IRR on a random sample vs. an independent second rater
      2. Topic saturation across all screened records

    ``client_factory`` (keyword-only, additive) defaults to
    ``_default_screening_qc_client_factory`` -- unchanged behavior for
    every existing caller, including LangGraph itself, which invokes this
    node as ``screening_qc_node(state)``. A caller that must pin or
    validate the client's retry policy supplies its own factory.

    ``result_store`` (keyword-only, additive, review finding 4) defaults
    to ``None`` -- unchanged behavior for every existing caller. Update-
    wave callers pass a ``SecondRaterResultStore`` to get crash-safe
    sample reservation and per-record result caching; see ``_compute_irr``.
    """
    cfg = ReviewConfig.from_dict(state["config"])
    qc_cfg = cfg.screening_qc

    if not qc_cfg.enabled:
        return {"screening_qc": None}

    records = [Record.from_dict(r) for r in state["screened_records_full"]]
    client = client_factory()
    model = qc_cfg.second_rater_model or cfg.screening.llm_model

    output_dir = cfg.ensure_output_dir()
    review_run_id = resolve_review_run_id(state, allow_config=True)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)

    logger.info(f"Screening QC: IRR on a sample of up to {qc_cfg.calibration_sample_size} records")
    irr = _compute_irr(
        records, client, model, cfg, qc_cfg.calibration_sample_size, event_store, review_run_id,
        calibration_seed=qc_cfg.calibration_seed, result_store=result_store,
    )

    topic_sets = [set(r.screening_relevant_topics) for r in records if r.screening_relevant_topics]
    logger.info(f"Screening QC: {len(topic_sets)}/{len(records)} records have topics for saturation analysis")

    saturation = None
    if topic_sets:
        saturation = {
            "n_records_with_topics": len(topic_sets),
            "rarefaction": _rarefaction_curve(topic_sets, qc_cfg.n_bootstrap, qc_cfg.saturation_levels),
            "bootstrapped_saturation_point": _bootstrapped_saturation_point(
                topic_sets, qc_cfg.n_bootstrap, qc_cfg.saturation_levels
            ),
            "jaccard_convergence": (
                _jaccard_convergence(topic_sets, sample_sizes=[10, 20, 30, 50, 80, 100])
                if len(topic_sets) >= 20 else None
            ),
        }

    # automation_origin derived directly from this stage's own AIUseEvents
    # this run -- screening_qc has no selection-ledger records at all (it
    # is not a selection stage). classify_automation_origin_optional()
    # (shared, aggregation/export/reporting plan v17 §2.8) excludes
    # call_status="failure" events entirely (INV-06): a failure produced
    # no automated output, so it must never be counted toward
    # "rule_based_automation" -- there is no fallback at this call site
    # (every non-success outcome IS a failure), so this stage's own events
    # can only ever be all-"success" ("ai") or all-"failure"
    # ("no_automated_output"); the shared function's "heuristic_fallback"/
    # "mixed" branches are simply never reached here, matching this site's
    # prior behavior exactly.
    events_this_stage = [e for e in event_store.load_all() if e.stage == STAGE]
    stage_origin = classify_automation_origin_optional(events_this_stage)

    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    oversight_mode[STAGE] = "automation_unreviewed"   # no human-review step exists for this stage
    if stage_origin is not None:
        automation_origin[STAGE] = stage_origin

    return {
        "screening_qc": {"irr": irr, "saturation": saturation},
        "screened_records_full": [r.to_dict() for r in records],  # carries second_rater_decision
        "review_run_id": review_run_id,
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
    }
