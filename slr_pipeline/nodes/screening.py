"""
screening_node — Step 3 of the SLR pipeline.

Title and abstract screening using:
  - Batch LLM classification (N records per call → include / borderline / exclude)
  - Parallel batch execution via ThreadPoolExecutor
  - Heuristic keyword fallback for any failed batches
  - Optional human-in-the-loop via LangGraph interrupt() for borderline records

Batching reduces 900 serial API calls to ~10 parallel round-trips.
Batch size and parallelism are tunable via SCREENING_BATCH_SIZE / SCREENING_WORKERS.

Node input keys:  deduplicated_records, config, prisma_counts
Node output keys: included_records, prisma_counts, review_run_id,
                   selection_decision_ledger_dir, oversight_mode,
                   automation_origin, screening_review_resolutions

Stage 3 instrumentation (the Stage 3 node-instrumentation design record):
every batch call is a durable `AIUseEvent`, reused across `interrupt()`/
resume via a frozen `ScreeningBatchResult` checkpoint (never recomputed --
INV-02, F11: the same prompt is not guaranteed to reproduce identical
content on a genuine retry, so the checkpoint, not the input identity
alone, is what the ledger's producer_artifact_id points at). Every record
gets a `SelectionDecisionRecord` (`stage="title_abstract_screening"`) for
its ai/rule_based batch outcome, and a second, superseding entry once a
human resolves a queued borderline (F2) -- via the shared
`expected_current_tail_decision_id` precondition (INV-03), never a bare
`emit_selection_decision()` call, so a stale human resolution can never
silently supersede a decision it never reviewed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from langgraph.types import interrupt
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from slr_pipeline.state import Record, PRISMACounts, ReviewState
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key
from slr_pipeline.run_identity import resolve_review_run_id
from slr_pipeline.human_resolution import (
    append_resolution_log_entry_deduplicated,
    compute_resolution_id,
    preflight_check_baseline_still_current,
    validate_submission_coverage,
)

from slr_pipeline.ai_traceability import AIUseEventStore, build_ai_use_event, reserved_attempt
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.selection_ledger import SelectionDecisionStore, derive_source_event_id, emit_selection_decision

logger = logging.getLogger(__name__)

# Tuning knobs — override via env vars if needed
SCREENING_BATCH_SIZE = int(os.environ.get("SCREENING_BATCH_SIZE", "20"))
SCREENING_WORKERS    = int(os.environ.get("SCREENING_WORKERS", "8"))


def _default_screening_client_factory() -> Any:
    """Today's exact unpatched behavior: a bare client with no
    ``max_retries`` override, so the installed ``openai`` SDK's own
    default (currently 2) applies -- see
    ``evaluation/update_cost_estimate.py``'s ``SCREENING_MAX_ATTEMPTS``,
    which prices this default explicitly. Kept as the DEFAULT (rather
    than pinning ``max_retries`` here) so existing callers/tests are
    unaffected; a caller that needs a pinned or validated retry policy
    (e.g. the Workstream C Checkpoint-3 driver) passes its own
    ``client_factory``."""
    return OpenAI(api_key=resolve_openai_api_key())


def effective_batch_size(cfg: ReviewConfig) -> int:
    """Track 2 plan §4.4: `cfg.screening.batch_size` overrides the
    `SCREENING_BATCH_SIZE` env constant when set; `None` (default)
    preserves production's env-driven behavior. The ONE place batch size
    is resolved -- every consumer (batch partitioning, thread-pool sizing,
    progress logging, reporting's own inventory) calls this rather than
    reading either source directly."""
    return cfg.screening.batch_size if cfg.screening.batch_size is not None else SCREENING_BATCH_SIZE


def effective_workers(cfg: ReviewConfig) -> int:
    """Track 2 plan §4.4: analogous to `effective_batch_size()`, for
    `cfg.screening.workers` / `SCREENING_WORKERS`."""
    return cfg.screening.workers if cfg.screening.workers is not None else SCREENING_WORKERS


STAGE = "title_abstract_screening"

# Bumped whenever the automated/threshold-routing CODE (not config values,
# already bound in separately) changes -- part of ScreeningBatchResult's
# own work_item_id fingerprint (F20/INV-12) so a policy-derivation change
# can never silently reuse a cached checkpoint whose already-derived
# ledger disposition would now be wrong.
SCREENING_POLICY_VERSION = "v1"


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class ScreeningResponse(BaseModel):
    decision: Literal["include", "borderline", "exclude"]
    reason: str = Field(min_length=1)
    relevant_topics: List[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, v: str) -> str:
        # codex review: Field(min_length=1) only checks raw character
        # count -- a single space satisfies it. A whitespace-only reason
        # is exactly as ungrounded as an empty one.
        if not v.strip():
            raise ValueError("reason must not be blank or whitespace-only")
        return v


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------

BATCH_SYSTEM = (
    "You are a systematic review screening assistant. "
    "Return only valid JSON — an array, one object per paper, in the same order as the input."
)

BATCH_PROMPT_TEMPLATE = """\
Screen the following {n} papers for inclusion in a systematic review.

Research questions:
{rqs}

Inclusion criteria:
{inc}

Exclusion criteria:
{exc}

Papers:
{papers}

Return a JSON object with key "items" containing exactly {n} objects, one per paper, in input order:
{{
  "items": [
    {{
      "idx": <1-based integer matching the paper number>,
      "decision": "include" | "borderline" | "exclude",
      "reason": "<one sentence>",
      "relevant_topics": ["<topic>"],
      "confidence": <0.0-1.0>
    }},
    ...
  ]
}}
"""


def _format_batch(
    records: List[Record],
    inclusion_criteria: List[str],
    exclusion_criteria: List[str],
    research_questions: List[str],
) -> str:
    rqs  = "\n".join(f"- {q}" for q in research_questions)  or "(not specified)"
    inc  = "\n".join(f"- {c}" for c in inclusion_criteria)  or "(not specified)"
    exc  = "\n".join(f"- {c}" for c in exclusion_criteria)  or "(not specified)"
    papers = "\n\n".join(
        f"[{i+1}] Title: {r.title}\n    Abstract: {(r.abstract or '')[:600]}"
        for i, r in enumerate(records)
    )
    return BATCH_PROMPT_TEMPLATE.format(
        n=len(records), rqs=rqs, inc=inc, exc=exc, papers=papers
    )


# ---------------------------------------------------------------------------
# Heuristic fallback (per-record)
# ---------------------------------------------------------------------------

def _heuristic_screen_record(
    record: Record,
    inclusion_criteria: List[str],
    exclusion_criteria: List[str],
    research_questions: List[str],
) -> ScreeningResponse:
    """Keyword-overlap fallback — no LLM required."""
    text = f"{record.title}\n{record.abstract or ''}".lower()
    terms: set = set()
    for collection in (research_questions, inclusion_criteria):
        for item in collection:
            if not item:
                continue
            for token in item.lower().replace("/", " ").replace("-", " ").split():
                token = token.strip(" ,.;:()[]{}\"'")
                if len(token) >= 4:
                    terms.add(token)
    matched = sorted(t for t in terms if t in text)
    decision: Literal["include", "borderline", "exclude"] = (
        "include" if matched else "exclude"
    )
    return ScreeningResponse(
        decision=decision,
        reason=f"Heuristic. Matched keywords: {matched[:5]}",
        relevant_topics=matched[:8],
        confidence=0.55 if matched else 0.75,
    )


# ---------------------------------------------------------------------------
# Content-addressed reference helpers (Stage 3 field contract)
# ---------------------------------------------------------------------------

def _record_content_ref(r: Record) -> str:
    return f"record:{r.id}:sha256:{sha256_hex({'title': r.title, 'abstract': r.abstract, 'doi': r.doi})}"


def _criteria_ref(inclusion_criteria: List[str], exclusion_criteria: List[str], research_questions: List[str]) -> str:
    return f"criteria:sha256:{sha256_hex({'inclusion_criteria': inclusion_criteria, 'exclusion_criteria': exclusion_criteria, 'research_questions': research_questions})}"


# ---------------------------------------------------------------------------
# ScreeningBatchResult — frozen, content-addressed, self-verifying
# checkpoint of one batch call's raw responses (Stage 3, F11/F14).
# Reused across interrupt()/resume instead of recomputed, closing the
# hazard a bare content-addressed producer id (round-3 draft) could not:
# the same rendered prompt is not guaranteed to reproduce identical output
# on a genuine second physical call, so the LEDGER's producer must point
# at a durable, reused OUTPUT artifact, never an input-identity stand-in.
# ---------------------------------------------------------------------------

class ScreeningBatchResultIntegrityError(RuntimeError):
    """A persisted `ScreeningBatchResult` (or its cross-verification
    against its own `producing_event_id`) failed self-verification or
    structural validation -- never silently trusted, worked around, or
    dropped."""


_SCREENING_BATCH_RESULT_SCHEMA_VERSION = "screening_batch_result_v1"


def compute_current_screening_batches(*, records: List[Record], cfg: ReviewConfig) -> List[List[Record]]:
    """The ONE authoritative batch-partitioning function, extracted
    verbatim from `screening_node`'s own `batches = [...]` construction
    (aggregation/export/reporting plan v17, §2.8) -- reused by
    `screening_node` itself and by reporting's own current-screening-
    batch-results inventory, so the two can never silently partition
    records differently. Track 2 plan §4.4: partitions by
    `effective_batch_size(cfg)`, not the raw env constant, so a pinned
    `cfg.screening.batch_size` reproduces the same batches regardless of
    the environment the run replays in."""
    size = effective_batch_size(cfg)
    return [records[i:i + size] for i in range(0, len(records), size)]


def compute_screening_batch_identity(
    *, batch: List[Record], cfg: ReviewConfig, review_run_id: str,
) -> Tuple[str, str, List[str]]:
    """Extracted verbatim from `_get_or_compute_batch_result`'s own
    identity computation (aggregation/export/reporting plan v17, §2.8) --
    returns `(artifact_id, batch_fingerprint, record_ids)`. Reused by
    reporting's own current-screening-batch-results inventory so a
    checkpoint's identity is never independently re-derived there."""
    model = cfg.screening.llm_model
    automated = cfg.screening.uncertain_threshold == 0.0
    prompt = _format_batch(batch, cfg.inclusion_criteria, cfg.exclusion_criteria, cfg.research_questions)
    messages = [
        {"role": "system", "content": BATCH_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    record_ids = [r.id for r in batch]
    canonical_payload = {
        "messages": messages, "model": model, "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "record_ids": record_ids,
        "automated": automated, "auto_include_threshold": cfg.screening.auto_include_threshold,
        "uncertain_threshold": cfg.screening.uncertain_threshold, "policy_version": SCREENING_POLICY_VERSION,
    }
    batch_fingerprint = sha256_hex(canonical_payload)
    work_item_id = compute_screening_work_item_id(stage=STAGE, batch_fingerprint=batch_fingerprint)
    artifact_id = "screenbatch-" + sha256_hex({"review_run_id": review_run_id, "work_item_id": work_item_id})
    return artifact_id, batch_fingerprint, record_ids


def compute_screening_work_item_id(*, stage: str, batch_fingerprint: str) -> str:
    """Pure function; `batch_index` is deliberately excluded from identity
    -- `batch_fingerprint` already hashes the actual assigned records, and
    a single node run partitions records into disjoint contiguous slices,
    so two different-index batches can never share a fingerprint within
    one run."""
    return f"{stage}:{batch_fingerprint}"


class ScreeningBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["screening_batch_result_v1"] = _SCREENING_BATCH_RESULT_SCHEMA_VERSION
    review_run_id: str
    stage: str
    work_item_id: str
    batch_fingerprint: str
    artifact_id: str
    record_ids: List[str]                 # ordered, 1:1 with responses, no duplicates
    producing_event_id: str
    call_status: Literal["success", "heuristic_fallback"]
    responses: List[ScreeningResponse]
    content_hash: str

    @model_validator(mode="after")
    def _check_counts(self) -> "ScreeningBatchResult":
        if len(self.record_ids) != len(self.responses):
            raise ValueError(
                f"record_ids has {len(self.record_ids)} entries but responses has {len(self.responses)} "
                "-- must be 1:1, in the same order"
            )
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError(f"record_ids contains duplicates: {self.record_ids}")
        return self

    @model_validator(mode="after")
    def _check_work_item_id_binding(self) -> "ScreeningBatchResult":
        expected = compute_screening_work_item_id(stage=self.stage, batch_fingerprint=self.batch_fingerprint)
        if self.work_item_id != expected:
            raise ValueError(
                f"work_item_id {self.work_item_id!r} does not match "
                f"compute_screening_work_item_id(stage={self.stage!r}, batch_fingerprint={self.batch_fingerprint!r}) "
                f"= {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_artifact_id_binding(self) -> "ScreeningBatchResult":
        # artifact_id is derived FROM work_item_id (not independently from
        # batch_fingerprint), so this checkpoint and its producing
        # AIUseEvent -- whose own event_id is ALSO a function of this same
        # work_item_id -- can never both look self-consistent while
        # actually referring to different, unrelated fingerprints (INV-04).
        expected = "screenbatch-" + sha256_hex({"review_run_id": self.review_run_id, "work_item_id": self.work_item_id})
        if self.artifact_id != expected:
            raise ValueError(
                f"artifact_id {self.artifact_id!r} does not match the expected "
                f"'screenbatch-' + sha256({{review_run_id, work_item_id}}) = {expected!r}"
            )
        return self

    def compute_content_hash(self) -> str:
        payload = {name: value for name, value in self.model_dump(mode="json").items() if name != "content_hash"}
        return sha256_hex(payload)

    def verify_hashes(self) -> None:
        expected = self.compute_content_hash()
        if self.content_hash != expected:
            raise ScreeningBatchResultIntegrityError(
                f"ScreeningBatchResult artifact_id={self.artifact_id!r} failed content_hash "
                f"self-verification: stored {self.content_hash!r} != recomputed {expected!r} -- file "
                "corrupted or tampered"
            )


def build_screening_batch_result(
    *, review_run_id: str, stage: str, batch_fingerprint: str, record_ids: List[str],
    producing_event_id: str, call_status: str, responses: List[ScreeningResponse],
) -> ScreeningBatchResult:
    work_item_id = compute_screening_work_item_id(stage=stage, batch_fingerprint=batch_fingerprint)
    artifact_id = "screenbatch-" + sha256_hex({"review_run_id": review_run_id, "work_item_id": work_item_id})
    base = dict(
        schema_version=_SCREENING_BATCH_RESULT_SCHEMA_VERSION, review_run_id=review_run_id, stage=stage,
        work_item_id=work_item_id, batch_fingerprint=batch_fingerprint, artifact_id=artifact_id,
        record_ids=record_ids, producing_event_id=producing_event_id, call_status=call_status,
        responses=responses,
    )
    content_hash = ScreeningBatchResult.model_construct(**base, content_hash="").compute_content_hash()
    return ScreeningBatchResult.model_validate(dict(base, content_hash=content_hash))


def _load_screening_batch_result_file(path: Path) -> ScreeningBatchResult:
    return ScreeningBatchResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ScreeningBatchResultStore:
    def __init__(self, *, output_dir: Path, review_run_id: str, event_store: AIUseEventStore) -> None:
        """Same safe-token-plus-resolved-containment discipline as every
        other Stage 3 store. `event_store` is used for cross-store
        verification against a checkpoint's own `producing_event_id`
        (never a manually-built path from persisted data -- INV-11, uses
        `event_store.load_event()`)."""
        batches_root = Path(output_dir) / "screening_batch_results"
        if not _SAFE_RUN_ID_RE.fullmatch(review_run_id):
            raise ValueError(
                f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
                "(no path separators, no '.', no '..')"
            )
        run_dir = batches_root / review_run_id
        if not run_dir.resolve().is_relative_to(batches_root.resolve()):
            raise ValueError(f"review_run_id {review_run_id!r} resolves outside the screening_batch_results root -- refusing to construct store")
        self.review_run_id = review_run_id
        self.results_dir = run_dir
        self.event_store = event_store

    def artifact_path(self, artifact_id: str) -> Path:
        return self.results_dir / f"{artifact_id}.json"

    def _verify_result(
        self, result: ScreeningBatchResult, *, path: Optional[Path] = None,
        expected_artifact_id: Optional[str] = None, expected_batch_fingerprint: Optional[str] = None,
        expected_record_ids: Optional[List[str]] = None,
    ) -> None:
        """The ONE place a `ScreeningBatchResult` is fully verified --
        self-hash, run binding, filename-vs-`artifact_id` identity (when
        loaded from a path), caller expectations, and cross-store
        verification against its own `producing_event_id`. Called from
        BOTH the loader (`_load_and_verify`) AND `publish()`'s OWN fresh-
        write path, not only its `FileExistsError` collision branch --
        codex review round: a freshly-published checkpoint that skipped
        this check could persist a `producing_event_id` referencing a
        REAL but UNRELATED event, undetected until a later `load()`."""
        result.verify_hashes()
        location = f"checkpoint at {path}" if path is not None else f"checkpoint artifact_id={result.artifact_id!r}"
        if result.review_run_id != self.review_run_id:
            raise ScreeningBatchResultIntegrityError(
                f"{location} has review_run_id={result.review_run_id!r}, expected this store's own "
                f"{self.review_run_id!r} -- refusing to trust a foreign-run checkpoint"
            )
        if path is not None and path.stem != result.artifact_id:
            raise ScreeningBatchResultIntegrityError(
                f"{location} has artifact_id={result.artifact_id!r}, which does not match its own "
                f"filename stem {path.stem!r} -- refusing to trust a renamed or mismatched file"
            )
        if expected_artifact_id is not None and result.artifact_id != expected_artifact_id:
            raise ScreeningBatchResultIntegrityError(
                f"{location} has artifact_id={result.artifact_id!r}, expected {expected_artifact_id!r}"
            )
        if expected_batch_fingerprint is not None and result.batch_fingerprint != expected_batch_fingerprint:
            raise ScreeningBatchResultIntegrityError(
                f"{location} has batch_fingerprint={result.batch_fingerprint!r}, expected "
                f"{expected_batch_fingerprint!r}"
            )
        if expected_record_ids is not None and result.record_ids != expected_record_ids:
            raise ScreeningBatchResultIntegrityError(
                f"{location} has record_ids={result.record_ids!r}, expected {expected_record_ids!r}"
            )
        # Cross-store verification: the checkpoint's claimed producing
        # event must genuinely correspond to this exact content, not just
        # reference a valid-looking event_id.
        event = self.event_store.load_event(result.producing_event_id)
        if event.stage != result.stage or event.work_item_id != result.work_item_id or event.call_status != result.call_status:
            raise ScreeningBatchResultIntegrityError(
                f"{location}'s producing_event_id={result.producing_event_id!r} refers to an event with "
                f"stage/work_item_id/call_status ({event.stage!r}/{event.work_item_id!r}/{event.call_status!r}) "
                f"that does not match this checkpoint's own ({result.stage!r}/{result.work_item_id!r}/"
                f"{result.call_status!r})"
            )
        expected_output_hash = f"sha256:{sha256_hex([r.model_dump(mode='json') for r in result.responses])}"
        if expected_output_hash not in event.output_artifact_refs:
            raise ScreeningBatchResultIntegrityError(
                f"{location}'s own responses hash to {expected_output_hash!r}, which is not present in "
                f"its producing event's output_artifact_refs {event.output_artifact_refs!r}"
            )

    def _load_and_verify(
        self, path: Path, *, expected_artifact_id: Optional[str] = None,
        expected_batch_fingerprint: Optional[str] = None, expected_record_ids: Optional[List[str]] = None,
    ) -> ScreeningBatchResult:
        result = _load_screening_batch_result_file(path)
        self._verify_result(
            result, path=path, expected_artifact_id=expected_artifact_id,
            expected_batch_fingerprint=expected_batch_fingerprint, expected_record_ids=expected_record_ids,
        )
        return result

    def load(
        self, artifact_id: str, *, expected_batch_fingerprint: Optional[str] = None,
        expected_record_ids: Optional[List[str]] = None,
    ) -> Optional[ScreeningBatchResult]:
        path = self.artifact_path(artifact_id)
        if not path.exists():
            return None
        return self._load_and_verify(
            path, expected_artifact_id=artifact_id, expected_batch_fingerprint=expected_batch_fingerprint,
            expected_record_ids=expected_record_ids,
        )

    def publish(self, result: ScreeningBatchResult) -> ScreeningBatchResult:
        # Full verification, including the cross-store check against
        # producing_event_id, BEFORE any write -- not only on the
        # collision path below.
        self._verify_result(result)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.artifact_path(result.artifact_id)
        tmp_path = self.results_dir / f".tmp.{uuid.uuid4().hex}.json"
        payload = result.model_dump_json(indent=2).encode("utf-8")
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp_path, final_path)
        except FileExistsError:
            existing = self._load_and_verify(final_path, expected_artifact_id=result.artifact_id)
            if existing.content_hash == result.content_hash:
                tmp_path.unlink(missing_ok=True)
                return existing
            raise ScreeningBatchResultIntegrityError(
                f"ScreeningBatchResult collision for artifact_id={result.artifact_id!r}: persisted "
                f"content_hash {existing.content_hash!r} != new result's {result.content_hash!r}. This "
                "should be structurally unreachable under reserved_attempt()'s own lock plus the "
                f"mandatory double-checked cache lookup -- leaving the temp file at {tmp_path} for "
                "forensic inspection rather than deleting evidence."
            )
        else:
            tmp_path.unlink(missing_ok=True)
            return result


# ---------------------------------------------------------------------------
# Get-or-compute one batch's ScreeningBatchResult, with double-checked
# locking around the cache and a concurrency-safe attempt allocation for
# the one real call this may need to make (INV-02, INV-10, F15).
# ---------------------------------------------------------------------------

def _get_or_compute_batch_result(
    *, batch: List[Record], client: OpenAI, model: str,
    inclusion_criteria: List[str], exclusion_criteria: List[str], research_questions: List[str],
    automated: bool, auto_include_threshold: float, uncertain_threshold: float,
    event_store: AIUseEventStore, batch_store: ScreeningBatchResultStore, review_run_id: str,
) -> ScreeningBatchResult:
    prompt = _format_batch(batch, inclusion_criteria, exclusion_criteria, research_questions)
    messages = [
        {"role": "system", "content": BATCH_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    record_ids = [r.id for r in batch]
    # This fingerprint computation is mirrored exactly (same fields, same
    # order) by the module-level compute_screening_batch_identity(), which
    # reporting's own current-screening-batch-results inventory uses to
    # reconstruct a checkpoint's identity from cfg alone, without needing
    # this function's own individual (model, inclusion_criteria, ...)
    # parameter breakout. F20/INV-12: record identity AND the screening
    # disposition policy are
    # both decision-relevant and must be bound into the fingerprint -- not
    # left implicit in the rendered text -- so neither a coincidental
    # prompt-text collision across different records/batches, nor a
    # policy/threshold change silently reusing a cached checkpoint whose
    # DERIVED disposition would now differ, can happen.
    canonical_payload = {
        "messages": messages, "model": model, "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "record_ids": record_ids,
        "automated": automated, "auto_include_threshold": auto_include_threshold,
        "uncertain_threshold": uncertain_threshold, "policy_version": SCREENING_POLICY_VERSION,
    }
    batch_fingerprint = sha256_hex(canonical_payload)
    work_item_id = compute_screening_work_item_id(stage=STAGE, batch_fingerprint=batch_fingerprint)
    artifact_id = "screenbatch-" + sha256_hex({"review_run_id": review_run_id, "work_item_id": work_item_id})

    cached = batch_store.load(artifact_id, expected_batch_fingerprint=batch_fingerprint, expected_record_ids=record_ids)
    if cached is not None:
        return cached

    with reserved_attempt(event_store, stage=STAGE, work_item_id=work_item_id) as attempt:
        # Mandatory re-check, now holding the lock (F15) -- checking only
        # before the lock lets two concurrent callers both miss and both
        # call the provider.
        cached = batch_store.load(artifact_id, expected_batch_fingerprint=batch_fingerprint, expected_record_ids=record_ids)
        if cached is not None:
            return cached

        response_received = False
        resp_model: Optional[str] = None
        parsed_items: Optional[List[ScreeningResponse]] = None
        failure_exc: Optional[Exception] = None
        try:
            resp = client.chat.completions.create(
                model=model, response_format={"type": "json_object"}, messages=messages, temperature=0.0,
            )
            response_received = True
            resp_model = resp.model
            raw = json.loads(resp.choices[0].message.content)
            if isinstance(raw, list):
                items = raw
            else:
                items = (
                    raw.get("items")
                    or raw.get("results")
                    or raw.get("papers")
                    or raw.get("decisions")
                    or next((v for v in raw.values() if isinstance(v, list)), [])
                )
            if len(items) != len(batch):
                raise ValueError(f"Got {len(items)} results for {len(batch)} records")
            # Semantic validation, not just shape (codex review): responses
            # are matched to records by their own declared "idx", never by
            # raw array position -- a reordered or duplicated response set
            # must never be silently assigned to the wrong record. No field
            # defaults here either -- a missing/invalid decision, empty
            # reason, or out-of-range confidence must raise (via
            # ScreeningResponse's own Field constraints) and fall through
            # to the heuristic fallback below, never fabricate a plausible-
            # looking but ungrounded response.
            indices = [item.get("idx") for item in items]
            # codex review: isinstance(i, int) accepts True/False too --
            # Python's bool is an int subclass -- so an idx of `true`
            # could pass as 1. type(i) is int rejects bool explicitly
            # (type(True) is bool, not int) while still accepting real ints.
            if any(type(i) is not int for i in indices):
                raise ValueError(f"response indices {indices!r} must all be plain integers (booleans are not accepted)")
            if sorted(indices) != list(range(1, len(batch) + 1)):
                raise ValueError(f"response indices {indices!r} are not an exact permutation of 1..{len(batch)}")
            items_by_idx = {item["idx"]: item for item in items}
            parsed_items = [
                ScreeningResponse(
                    decision=items_by_idx[i].get("decision"),
                    reason=items_by_idx[i].get("reason"),
                    relevant_topics=items_by_idx[i].get("relevant_topics", []),
                    confidence=items_by_idx[i].get("confidence"),
                )
                for i in range(1, len(batch) + 1)
            ]
        except Exception as e:
            logger.warning(f"Batch screening failed ({e}); falling back to heuristic for {len(batch)} records")
            failure_exc = e

        # F23: call_status/output existence and response-version
        # availability are independent axes. Screening always has a real
        # fallback (the keyword heuristic), so this site is always either
        # "success" or "heuristic_fallback", never bare "failure".
        if parsed_items is not None:
            call_status = "success"
            responses = parsed_items
        else:
            call_status = "heuristic_fallback"
            responses = [
                _heuristic_screen_record(r, inclusion_criteria, exclusion_criteria, research_questions)
                for r in batch
            ]
        output_payload = [r.model_dump(mode="json") for r in responses]
        component_versions = {model: resp_model} if response_received else {}

        event = build_ai_use_event(
            review_run_id=review_run_id, stage=STAGE, work_item_id=work_item_id, attempt=attempt,
            tool_kind="llm", model=model, provider="openai", component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose="Classify a batch of records for title/abstract inclusion eligibility",
            model_config_id="title_abstract_screening_batch_v1",
            prompt_template_version="v1",
            prompt_template_text=json.dumps(
                [{"role": "system", "content": BATCH_SYSTEM}, {"role": "user", "content": BATCH_PROMPT_TEMPLATE}]
            ),
            prompt_parameters={"temperature": 0.0, "response_format": {"type": "json_object"}},
            rendered_prompt_hash=sha256_hex(messages),
            input_artifact_refs=(
                [_criteria_ref(inclusion_criteria, exclusion_criteria, research_questions)]
                + [_record_content_ref(r) for r in batch]
            ),
            output_artifact_refs=[f"sha256:{sha256_hex(output_payload)}"],
            call_status=call_status,
            failure_detail=(str(failure_exc) if failure_exc is not None else None),
        )
        event = event_store.publish(event)

        result = build_screening_batch_result(
            review_run_id=review_run_id, stage=STAGE, batch_fingerprint=batch_fingerprint,
            record_ids=record_ids, producing_event_id=event.event_id, call_status=call_status,
            responses=responses,
        )
        return batch_store.publish(result)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def screening_node(
    state: ReviewState, *, client_factory: Callable[[], Any] = _default_screening_client_factory,
) -> Dict[str, Any]:
    """
    Screens title+abstract of deduplicated records using batched parallel LLM calls.

    Workflow:
      1. Split records into batches of SCREENING_BATCH_SIZE
      2. Get-or-compute each batch's ScreeningBatchResult (real call, or a
         frozen checkpoint reused across interrupt()/resume — INV-02)
      3. Emit one ai/rule_based SelectionDecisionRecord per record from
         each batch's own outcome
      4. If uncertain_threshold > 0: queue borderlines for human review
         (interrupt()), capturing each queued record's reviewed ledger
         baseline; apply the eventual human resolutions as a second,
         superseding ledger entry each (F2/F3, INV-03)
         If uncertain_threshold == 0: auto-decide all records (for
         reproduction runs)
      5. Update PRISMA counts, oversight_mode, automation_origin

    ``client_factory`` (keyword-only, additive) defaults to
    ``_default_screening_client_factory`` -- unchanged behavior for every
    existing caller, including LangGraph itself, which invokes this node
    as ``screening_node(state)``. A caller that must pin or validate the
    client's retry policy (see ``evaluation/update_cost_estimate.py``'s
    ``SCREENING_MAX_ATTEMPTS``) supplies its own factory.
    """
    cfg = ReviewConfig.from_dict(state["config"])
    records = [Record.from_dict(r) for r in state["deduplicated_records"]]

    client = client_factory()
    model = cfg.screening.llm_model
    auto_include_threshold = cfg.screening.auto_include_threshold
    uncertain_threshold = cfg.screening.uncertain_threshold
    automated = uncertain_threshold == 0.0   # True in reproduction/smoke-test mode

    inc_crit = cfg.inclusion_criteria
    exc_crit = cfg.exclusion_criteria
    rqs      = cfg.research_questions

    output_dir = cfg.ensure_output_dir()
    review_run_id = resolve_review_run_id(state, allow_config=True)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    batch_store = ScreeningBatchResultStore(output_dir=output_dir, review_run_id=review_run_id, event_store=event_store)

    batch_size = effective_batch_size(cfg)
    workers = effective_workers(cfg)
    logger.info(
        f"Screening {len(records)} records in batches of {batch_size} "
        f"({workers} workers) using {model}"
    )
    print(f"  Screening {len(records)} records "
          f"(batch={batch_size}, workers={workers}, model={model})...")

    # --- Build batches ---
    batches = compute_current_screening_batches(records=records, cfg=cfg)

    # --- Get-or-compute batches in parallel ---
    batch_results: Dict[int, ScreeningBatchResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_batch_idx = {
            pool.submit(
                _get_or_compute_batch_result, batch=batch, client=client, model=model,
                inclusion_criteria=inc_crit, exclusion_criteria=exc_crit, research_questions=rqs,
                automated=automated, auto_include_threshold=auto_include_threshold,
                uncertain_threshold=uncertain_threshold, event_store=event_store, batch_store=batch_store,
                review_run_id=review_run_id,
            ): idx
            for idx, batch in enumerate(batches)
        }
        completed = 0
        for future in as_completed(future_to_batch_idx):
            batch_idx = future_to_batch_idx[future]
            # Deliberately NOT wrapped in a broad try/except: a genuine
            # store-integrity/lock error here must propagate and abort
            # the run (INV-05), never be silently downgraded. Ordinary
            # provider/parse failures never reach this point as an
            # exception at all -- _get_or_compute_batch_result's own
            # narrow try/except already turns those into a
            # heuristic_fallback checkpoint.
            batch_results[batch_idx] = future.result()
            completed += 1
            print(f"  Screened batch {completed}/{len(batches)} "
                  f"({min((batch_idx + 1) * batch_size, len(records))}/{len(records)} records)")

    # --- Per-record AI/rule_based ledger entries + response lookup ---
    responses: Dict[int, ScreeningResponse] = {}
    baseline_decision_id_by_record: Dict[str, Optional[str]] = {}
    global_idx = 0
    for batch_idx, batch in enumerate(batches):
        result = batch_results[batch_idx]
        decision_maker = "ai" if result.call_status == "success" else "rule_based_automation"
        for j, record in enumerate(batch):
            resp = result.responses[j]
            responses[global_idx] = resp

            # INV-09: the ledger's decision is the OPERATIONALLY FINAL
            # value where the pipeline actually treats one as final.
            # Automated mode collapses a raw "borderline" to "include"
            # (record.screening_decision itself stays the raw label,
            # unchanged) -- a fixed, pre-specified policy, not a new
            # judgment, so decision_maker is unaffected.
            if automated and resp.decision == "borderline":
                ledger_decision = "include"
                ledger_reason = f"borderline (auto-included under automated-mode policy); {resp.reason}"
            else:
                ledger_decision = resp.decision
                ledger_reason = resp.reason

            source_event_id = derive_source_event_id(
                producer_artifact_id=result.artifact_id, record_id=record.id, stage=STAGE,
            )
            emitted = emit_selection_decision(
                store=ledger_store, record_id=record.id, stage=STAGE, decision=ledger_decision,
                decision_maker=decision_maker, source_event_id=source_event_id, reason=ledger_reason,
                producer_artifact_id=result.artifact_id,
            )
            baseline_decision_id_by_record[record.id] = emitted.decision_id
            global_idx += 1

    # --- Apply decisions ---
    included: List[Record] = []
    for_human_review: List[Dict[str, Any]] = []
    excluded_count = 0

    for i, record in enumerate(records):
        resp = responses[i]

        record.screening_decision = resp.decision
        record.screening_reason = resp.reason
        record.screening_confidence = resp.confidence
        record.screening_relevant_topics = resp.relevant_topics

        if automated:
            # Automated mode: treat all non-exclude as include, no human queue
            if resp.decision in ("include", "borderline"):
                included.append(record)
            else:
                excluded_count += 1
        else:
            if resp.decision == "include" and resp.confidence >= auto_include_threshold:
                included.append(record)
            elif resp.decision == "exclude" and resp.confidence >= auto_include_threshold:
                excluded_count += 1
            else:
                for_human_review.append({
                    "record": record.to_dict(),
                    "llm_decision": resp.decision,
                    "llm_reason": resp.reason,
                    "confidence": resp.confidence,
                    "reviewed_baseline_decision_id": baseline_decision_id_by_record[record.id],
                })

    screening_review_resolutions = list(state.get("screening_review_resolutions") or [])

    # --- Human-in-the-loop (only when not in automated mode) ---
    if for_human_review and not automated:
        logger.info(f"{len(for_human_review)} records need human review; interrupting...")
        human_decisions: List[Dict[str, Any]] = interrupt({
            "message": (
                f"{len(for_human_review)} records require your screening decision. "
                "For each, respond with 'include' or 'exclude', plus reviewer/note/resolved_at."
            ),
            "records_for_review": for_human_review,
        })

        # --- Two-pass: validate the WHOLE submission, zero side effects
        # on failure, THEN apply (F2/F3, INV-03) ---
        queue_by_id = {item["record"]["id"]: item for item in for_human_review}
        submitted_ids = [d.get("record_id") for d in human_decisions]
        validate_submission_coverage(
            submitted_record_ids=submitted_ids, queued_record_ids=set(queue_by_id),
            require_full_coverage=True,   # exact response-set equality -- never default to "exclude"
        )

        candidates: List[Dict[str, Any]] = []
        for d in human_decisions:
            record_id = d["record_id"]
            decision = d.get("decision")
            reviewer = d.get("reviewer")
            note = d.get("note")
            resolved_at = d.get("resolved_at")
            if decision not in ("include", "exclude"):
                raise ValueError(f"human resolution for {record_id!r} must be 'include' or 'exclude', got {decision!r}")
            if not reviewer or not note or not resolved_at:
                raise ValueError(f"human resolution for {record_id!r} must set reviewer, note, and resolved_at")

            reviewed_baseline_decision_id = queue_by_id[record_id]["reviewed_baseline_decision_id"]
            resolution_id = compute_resolution_id(
                stage=STAGE, record_id=record_id, action=decision, reviewer=reviewer, note=note,
                resolved_at=resolved_at, reviewed_baseline_decision_id=reviewed_baseline_decision_id,
            )
            source_event_id = derive_source_event_id(
                producer_artifact_id=resolution_id, record_id=record_id, stage=STAGE,
            )
            # Fast, unlocked preflight -- fails an obviously-stale
            # submission early. NOT the source of truth: emit_selection_
            # decision()'s own expected_current_tail_decision_id (below)
            # is the atomic, authoritative guarantee (INV-03).
            preflight_check_baseline_still_current(
                store=ledger_store, record_id=record_id, stage=STAGE,
                reviewed_baseline_decision_id=reviewed_baseline_decision_id,
                resolution_source_event_id=source_event_id,
            )
            candidates.append({
                "record_id": record_id, "decision": decision, "reviewer": reviewer, "note": note,
                "resolved_at": resolved_at, "reviewed_baseline_decision_id": reviewed_baseline_decision_id,
                "resolution_id": resolution_id, "source_event_id": source_event_id,
            })

        # Every candidate passed validation + preflight -- apply.
        for c in candidates:
            emit_selection_decision(
                store=ledger_store, record_id=c["record_id"], stage=STAGE, decision=c["decision"],
                decision_maker="human", source_event_id=c["source_event_id"], reason=c["note"],
                producer_artifact_id=c["resolution_id"],
                expected_current_tail_decision_id=c["reviewed_baseline_decision_id"],
            )
            item = queue_by_id[c["record_id"]]
            rec = Record.from_dict(item["record"])
            rec.screening_decision = c["decision"]
            if c["decision"] == "include":
                included.append(rec)
            else:
                excluded_count += 1
            screening_review_resolutions = append_resolution_log_entry_deduplicated(
                screening_review_resolutions,
                {
                    "record_id": c["record_id"], "decision": c["decision"], "reviewer": c["reviewer"],
                    "note": c["note"], "resolved_at": c["resolved_at"], "resolution_id": c["resolution_id"],
                    "reviewed_baseline_decision_id": c["reviewed_baseline_decision_id"],
                },
                resolution_id=c["resolution_id"],
            )

    print(f"  Screening done: {len(included)} included, {excluded_count} excluded"
          + (f", {len(for_human_review)} queued for review" if for_human_review else ""))
    logger.info(f"Screening complete: {len(included)} included, {excluded_count} excluded")

    prisma = PRISMACounts(**state["prisma_counts"])
    prisma.screened_title_abstract = len(records)
    prisma.excluded_title_abstract = excluded_count

    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    if batch_results:
        # Zero batches (zero records screened this run) must never fall
        # through to "mixed" -- an empty decision_makers_used set is not a
        # genuine mix of two mechanisms, it's the absence of any call at
        # all (codex review). Every non-empty batch_results always has at
        # least one of {"ai", "rule_based_automation"}, so "mixed" here is
        # only ever reached with two GENUINE, distinct mechanisms present.
        decision_makers_used = {
            "ai" if br.call_status == "success" else "rule_based_automation"
            for br in batch_results.values()
        }
        if decision_makers_used == {"ai"}:
            stage_origin = "ai"
        elif decision_makers_used == {"rule_based_automation"}:
            stage_origin = "rule_based_automation"
        else:
            stage_origin = "mixed"
        automation_origin[STAGE] = stage_origin
        oversight_mode[STAGE] = "human_resolved" if screening_review_resolutions else "automation_unreviewed"

    return {
        "included_records": [r.to_dict() for r in included],
        "screened_records_full": [r.to_dict() for r in records],
        "prisma_counts": prisma.model_dump(),
        "requires_human_review": False,
        "review_run_id": review_run_id,
        "selection_decision_ledger_dir": str(ledger_store.decisions_dir),
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
        "screening_review_resolutions": screening_review_resolutions,
    }
