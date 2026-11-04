"""Runtime boundary for extraction-resolution suggestion passes.

This module joins the already-audited pieces without weakening their
separate contracts: immutable review packets, crash-safe attempt files,
AI-use events, and the continuation workspace's existing cost ledger and
proxy.  It performs no network operation during packet preparation or cost
estimation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.ai_traceability import AIUseEventStore
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key
from slr_pipeline.cost_cap.ledger import CostLedger
from slr_pipeline.cost_cap.pricing import estimate_worst_case_cost_usd, is_finite_number, is_priced
from slr_pipeline.cost_cap.proxy import CostCapProxy, count_prompt_tokens
from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
from slr_pipeline.extraction.hashing import canonical_json_bytes, sha256_hex
from slr_pipeline.extraction.resolution_schemas import (
    QueueSnapshot,
    QueueSnapshotEntry,
    QueueSnapshotResolutionEntry,
    ValidatedSuggestion,
)
from slr_pipeline.extraction.resolution_suggestions import (
    ReviewPacket,
    SuggestionEngineResult,
    atomic_write_bytes,
    build_suggestion_prompt,
    eligible_suggestion_entries,
    load_attempt_records,
    render_review_packet,
    review_packet_json_bytes,
    review_packet_markdown_bytes,
    run_suggestion_engine,
    suggestions_index_bytes,
)
from slr_pipeline.extraction.verified_bundle import (
    bundle_review_fields,
    load_and_verify_bundle,
)


@dataclass(frozen=True)
class PreparedReviewPacket:
    packet: ReviewPacket
    snapshot: QueueSnapshot
    run_dir: Path


@dataclass(frozen=True)
class SuggestionCostEstimate:
    model: str
    eligible_calls: int
    prompt_tokens: int
    max_completion_tokens_per_call: int
    worst_case_new_spend_usd: float
    ledger_cap_usd: float
    already_committed_usd: float
    remaining_before_calls_usd: float

    @property
    def fits_remaining_cap(self) -> bool:
        return self.worst_case_new_spend_usd <= self.remaining_before_calls_usd + 1e-12

    def render_breakdown(self) -> str:
        return "\n".join((
            "Extraction resolution suggestion estimate",
            f"  model: {self.model}",
            f"  remaining eligible calls: {self.eligible_calls}",
            f"  estimated prompt tokens: {self.prompt_tokens}",
            f"  max completion tokens per call: {self.max_completion_tokens_per_call}",
            f"  worst-case new spend: ${self.worst_case_new_spend_usd:.6f}",
            f"  existing ledger committed: ${self.already_committed_usd:.6f}",
            f"  existing ledger remaining: ${self.remaining_before_calls_usd:.6f}",
            f"  fits remaining cap: {'yes' if self.fits_remaining_cap else 'no'}",
        ))


@dataclass(frozen=True)
class SuggestionPassResult:
    engine: SuggestionEngineResult
    remaining_usd: float
    already_committed_usd: float


class ResolutionDecisionRequest(BaseModel):
    """One explicit human action over a packet field."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    field_id: str
    decision_mode: Literal["accept_suggestion", "edit_suggestion", "human"]
    reviewer: str
    note: str
    resolved_at: str
    suggestion_id: Optional[str] = None
    outcome: Optional[Literal["extracted", "not_reported"]] = None
    final_value: Optional[Any] = None
    selected_evidence_ids: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_mode(self) -> "ResolutionDecisionRequest":
        for name in ("record_id", "field_id", "reviewer", "note", "resolved_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        if self.decision_mode == "accept_suggestion":
            if not self.suggestion_id:
                raise ValueError("accept_suggestion requires suggestion_id")
            if self.outcome is not None or self.final_value is not None or self.selected_evidence_ids:
                raise ValueError("accept_suggestion derives outcome/value/evidence from the immutable suggestion")
        elif self.decision_mode == "edit_suggestion":
            if not self.suggestion_id or self.outcome is None:
                raise ValueError("edit_suggestion requires suggestion_id and outcome")
        else:
            if self.suggestion_id is not None or self.outcome is None:
                raise ValueError("human decisions prohibit suggestion_id and require outcome")
        if self.outcome == "not_reported" and (self.final_value is not None or self.selected_evidence_ids):
            raise ValueError("not_reported decisions cannot carry value/evidence")
        return self


class ResolutionDecisionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    extraction_run_id: str
    decisions: Tuple[ResolutionDecisionRequest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> "ResolutionDecisionBatch":
        keys = [(item.record_id, item.field_id) for item in self.decisions]
        if len(keys) != len(set(keys)):
            raise ValueError("decision batch contains duplicate (record_id, field_id) keys")
        return self


@dataclass(frozen=True)
class AppliedDecisionBatchResult:
    state: Dict[str, Any]
    applied: int
    already_applied: int
    remaining_queue: int


def _snapshot_from_verified_state(
    state: Dict[str, Any], *, bundle: Any, schema: Any,
    resolutions_artifact: Dict[str, Any],
) -> QueueSnapshot:
    verified_fields = bundle_review_fields(bundle.results)
    history = []
    history_fields = tuple(QueueSnapshotResolutionEntry.model_fields)
    for raw in resolutions_artifact.get("resolutions") or []:
        if raw.get("run_id") != bundle.run_id:
            continue
        history.append(QueueSnapshotResolutionEntry.model_validate({
            name: raw.get(name) for name in history_fields
        }))
    resolved_keys = {(entry.record_id, entry.field_id) for entry in history}
    queue = []
    for raw in state.get("extraction_review_queue") or []:
        if (raw.get("record_id"), raw.get("field_id")) in resolved_keys:
            continue
        queue.append(QueueSnapshotEntry.model_validate({
            "record_id": raw.get("record_id"),
            "field_id": raw.get("field_id"),
            "status": raw.get("status"),
            "field_result": raw.get("field_result"),
        }))
    return QueueSnapshot(
        run_id=bundle.run_id,
        manifest_hash=bundle.manifest.manifest_hash,
        extraction_schema_hash=sha256_hex(schema.model_dump(mode="json")),
        checkpoint_hashes=dict(bundle.manifest.checkpoint_hashes),
        queue=queue,
        resolution_history_as_of_render=history,
    )


def prepare_review_packet(state: Dict[str, Any]) -> PreparedReviewPacket:
    """Create or deterministically reopen the immutable packet trio."""
    run_id = state.get("extraction_run_id")
    if not run_id:
        raise ValueError("checkpoint state has no extraction_run_id")
    store, bundle, schema, _schema_hash = load_and_verify_bundle(state, run_id)
    from slr_pipeline.nodes.extraction import load_verified_extraction_resolution_artifact

    resolutions = load_verified_extraction_resolution_artifact(state)
    if resolutions is None:
        raise ValueError("verified extraction_resolutions.json is required before rendering a review packet")

    snapshot_path = store.run_dir / "review_queue_snapshot.json"
    packet_json_path = store.run_dir / "extraction_review_packet.json"
    packet_md_path = store.run_dir / "extraction_review_packet.md"
    if snapshot_path.exists():
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot = QueueSnapshot.model_validate_json(snapshot_bytes)
        if canonical_json_bytes(snapshot.model_dump(mode="json")) != snapshot_bytes:
            raise ValueError("existing review_queue_snapshot.json is not canonical JSON")
    else:
        snapshot = _snapshot_from_verified_state(
            state, bundle=bundle, schema=schema, resolutions_artifact=resolutions,
        )
        snapshot_bytes = canonical_json_bytes(snapshot.model_dump(mode="json"))

    packet = render_review_packet(bundle, schema, snapshot)
    expected = (
        (snapshot_path, snapshot_bytes),
        (packet_json_path, review_packet_json_bytes(packet)),
        (packet_md_path, review_packet_markdown_bytes(packet)),
    )
    for path, payload in expected:
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"existing immutable review artifact does not re-derive byte-identically: {path}")
        else:
            atomic_write_bytes(path, payload)
    return PreparedReviewPacket(packet=packet, snapshot=snapshot, run_dir=store.run_dir)


def _read_existing_ledger(path: Path) -> Tuple[float, float]:
    if not path.is_file():
        raise ValueError(f"existing continuation cost ledger is required: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cap = payload["cap_usd"]
        raw_committed = payload["total_committed_usd"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"existing continuation cost ledger is invalid: {path}: {exc}") from exc
    if not is_finite_number(raw_cap) or not is_finite_number(raw_committed):
        raise ValueError(f"existing continuation cost ledger has non-finite cap/committed totals: {path}")
    cap = float(raw_cap)
    committed = float(raw_committed)
    if cap < 0 or committed < 0 or committed > cap + 1e-9:
        raise ValueError(f"existing continuation cost ledger has invalid cap/committed totals: {path}")
    return cap, committed


def estimate_suggestion_cost(
    packet: ReviewPacket, *, suggestions_dir: Path, ledger_path: Path,
    model: str, max_completion_tokens: int,
    retry_work_item_ids: Sequence[str] = (),
) -> SuggestionCostEstimate:
    """Estimate only the work items a restart would actually dispatch."""
    if not is_priced(model):
        raise ValueError(f"suggestion model {model!r} is absent from the cost-cap pricing table")
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    cap, committed = _read_existing_ledger(Path(ledger_path))
    entries = eligible_suggestion_entries(
        packet, suggestions_dir=Path(suggestions_dir),
        retry_work_item_ids=retry_work_item_ids,
    )
    prompt_tokens = 0
    worst_case = 0.0
    for entry in entries:
        messages = [{"role": "user", "content": build_suggestion_prompt(entry)}]
        tokens = count_prompt_tokens(model, messages)
        prompt_tokens += tokens
        worst_case += estimate_worst_case_cost_usd(
            model=model, prompt_tokens=tokens,
            max_output_tokens=max_completion_tokens,
        )
    return SuggestionCostEstimate(
        model=model,
        eligible_calls=len(entries),
        prompt_tokens=prompt_tokens,
        max_completion_tokens_per_call=max_completion_tokens,
        worst_case_new_spend_usd=worst_case,
        ledger_cap_usd=cap,
        already_committed_usd=committed,
        remaining_before_calls_usd=cap - committed,
    )


def _default_client_factory() -> Any:
    from openai import OpenAI
    return OpenAI(api_key=resolve_openai_api_key(), max_retries=0)


def run_paid_suggestion_pass(
    state: Dict[str, Any], packet: ReviewPacket, *, cfg: ReviewConfig,
    workspace: Path, cap_usd: float, model: str, model_config_id: str,
    max_completion_tokens: int, retry_work_item_ids: Sequence[str] = (),
    client_factory: Callable[[], Any] = _default_client_factory,
    reconcile_journal_fn: Optional[Callable[[Dict[str, Any], ReviewPacket], None]] = None,
) -> SuggestionPassResult:
    """Run one authorized suggestion pass through the existing run cap."""
    review_run_id = state.get("review_run_id")
    if not review_run_id:
        raise ValueError("checkpoint state has no review_run_id")
    if packet.run_id != state.get("extraction_run_id"):
        raise ValueError("review packet does not belong to checkpoint extraction_run_id")
    store = ExtractionCheckpointStore(
        output_dir=cfg.ensure_output_dir() / "extraction_runs",
        run_id=packet.run_id,
    )
    suggestions_dir = store.run_dir / "extraction_resolution_suggestions"
    ledger_path = Path(workspace) / "cost_cap_ledger.json"
    if not ledger_path.is_file():
        raise ValueError(f"refusing to create a fresh suggestion ledger; existing ledger required: {ledger_path}")

    if reconcile_journal_fn is None:
        from slr_pipeline.nodes.extraction import reconcile_resolution_journal
        reconcile_journal_fn = reconcile_resolution_journal
    reconcile_journal_fn(state, packet)

    ledger = CostLedger(ledger_path, cap_usd=cap_usd)
    committed_before = ledger.cap_usd - ledger.remaining_usd()
    call_log_path = Path(workspace) / "cost_cap_call_log.jsonl"
    proxy = CostCapProxy(ledger, call_log_path, port=0)
    proxy.start()
    previous_base_url = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = proxy.base_url
    try:
        client = client_factory()
        result = run_suggestion_engine(
            packet,
            extraction_run_id=packet.run_id,
            review_run_id=review_run_id,
            suggestions_dir=suggestions_dir,
            event_store=AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id),
            client=client,
            model=model,
            provider="openai",
            component_versions={"resolution_suggestions": "1"},
            model_config_id=model_config_id,
            max_completion_tokens=max_completion_tokens,
            retry_work_item_ids=retry_work_item_ids,
        )
    finally:
        try:
            attempts = load_attempt_records(suggestions_dir)
            atomic_write_bytes(
                store.run_dir / "extraction_resolution_suggestions.json",
                suggestions_index_bytes(attempts, extraction_run_id=packet.run_id),
            )
        finally:
            try:
                proxy.stop()
            finally:
                if previous_base_url is None:
                    os.environ.pop("OPENAI_BASE_URL", None)
                else:
                    os.environ["OPENAI_BASE_URL"] = previous_base_url
                ledger.ensure_persisted()
                call_log_path.touch(exist_ok=True)
    return SuggestionPassResult(
        engine=result,
        remaining_usd=ledger.remaining_usd(),
        already_committed_usd=committed_before,
    )


def load_resolution_decision_batch(path: Path) -> ResolutionDecisionBatch:
    """Load a human-authored decision batch without rewriting it."""
    try:
        return ResolutionDecisionBatch.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid extraction resolution decision batch {path}: {exc}") from exc


def _validated_suggestion_by_id(
    suggestions_dir: Path, suggestion_id: str,
) -> Tuple[Path, ValidatedSuggestion]:
    matches = [
        (path, record)
        for path, record in load_attempt_records(suggestions_dir)
        if isinstance(record, ValidatedSuggestion) and record.suggestion_id == suggestion_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"suggestion_id {suggestion_id!r} must identify exactly one validated suggestion; "
            f"found {len(matches)}"
        )
    return matches[0]


def _decision_parameters(
    request: ResolutionDecisionRequest, *, suggestions_dir: Path,
) -> Tuple[Dict[str, Any], Optional[Path], Optional[ValidatedSuggestion]]:
    suggestion_path: Optional[Path] = None
    suggestion: Optional[ValidatedSuggestion] = None
    if request.suggestion_id is not None:
        suggestion_path, suggestion = _validated_suggestion_by_id(
            suggestions_dir, request.suggestion_id,
        )
        if (suggestion.record_id, suggestion.field_id) != (request.record_id, request.field_id):
            raise ValueError("decision's suggestion_id belongs to a different record/field")
    if request.decision_mode == "accept_suggestion":
        assert suggestion is not None
        outcome = suggestion.outcome
        final_value = (
            suggestion.typed_value if suggestion.typed_value is not None else suggestion.value
        ) if outcome == "extracted" else None
        selected_evidence_ids = suggestion.evidence_ids if outcome == "extracted" else ()
    else:
        assert request.outcome is not None
        outcome = request.outcome
        final_value = request.final_value if outcome == "extracted" else None
        selected_evidence_ids = request.selected_evidence_ids if outcome == "extracted" else ()
    return ({
        "record_id": request.record_id,
        "field_id": request.field_id,
        "outcome": outcome,
        "final_value": final_value,
        "selected_evidence_ids": tuple(selected_evidence_ids),
        "reviewer": request.reviewer,
        "note": request.note,
        "resolved_at": request.resolved_at,
        "decision_origin": (
            "human" if request.decision_mode == "human"
            else "machine_recommended_human_approved"
        ),
    }, suggestion_path, suggestion)


def _matches_applied_entry(entry: Any, *, params: Dict[str, Any], suggestion: Optional[ValidatedSuggestion]) -> bool:
    return all((
        entry.status == "applied",
        entry.record_id == params["record_id"],
        entry.field_id == params["field_id"],
        entry.outcome == params["outcome"],
        entry.final_value == params["final_value"],
        tuple(entry.selected_evidence_ids) == tuple(params["selected_evidence_ids"]),
        entry.reviewer == params["reviewer"].strip(),
        entry.note == params["note"].strip(),
        entry.resolved_at == params["resolved_at"].strip(),
        entry.decision_origin == params["decision_origin"],
        entry.suggestion_id == (suggestion.suggestion_id if suggestion else None),
        entry.self_hash == (suggestion.self_hash if suggestion else None),
    ))


def refresh_resolution_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Project any durable resolutions missing from the graph checkpoint."""
    from slr_pipeline.nodes.extraction import (
        load_verified_extraction_resolution_artifact,
        resolve_extraction_review_queue,
    )

    artifact = load_verified_extraction_resolution_artifact(state)
    if artifact is None:
        raise ValueError("verified extraction resolution artifact is absent")
    run_id = state.get("extraction_run_id")
    resolved_keys = {
        (row["record_id"], row["field_id"])
        for row in artifact.get("resolutions") or []
        if row.get("run_id") == run_id
    }
    projected = dict(state)
    projected["extraction_review_queue"] = [
        row for row in state.get("extraction_review_queue") or []
        if (row.get("record_id"), row.get("field_id")) not in resolved_keys
    ]
    return resolve_extraction_review_queue(projected, {})


def apply_resolution_decision_batch(
    state: Dict[str, Any], packet: ReviewPacket, batch: ResolutionDecisionBatch, *,
    on_state_update: Callable[[Dict[str, Any]], None] = lambda _update: None,
) -> AppliedDecisionBatchResult:
    """Apply human decisions one at a time and checkpoint each projection."""
    if batch.extraction_run_id != packet.run_id or packet.run_id != state.get("extraction_run_id"):
        raise ValueError("decision batch, review packet, and checkpoint extraction_run_id differ")
    from slr_pipeline.extraction.resolution_journal import read_decision_journal
    from slr_pipeline.nodes.extraction import (
        apply_extraction_resolution_decision,
        reconcile_resolution_journal,
    )

    store = ExtractionCheckpointStore(
        output_dir=ReviewConfig.from_dict(state["config"]).ensure_output_dir() / "extraction_runs",
        run_id=packet.run_id,
    )
    suggestions_dir = store.run_dir / "extraction_resolution_suggestions"
    reconcile_resolution_journal(state, packet)
    current = dict(state)
    refresh_update = refresh_resolution_state(current)
    current.update(refresh_update)
    on_state_update(refresh_update)
    journal = read_decision_journal(store)
    applied = 0
    already_applied = 0

    for request in batch.decisions:
        params, suggestion_path, suggestion = _decision_parameters(
            request, suggestions_dir=suggestions_dir,
        )
        key = (request.record_id, request.field_id)
        queue_keys = {
            (row.get("record_id"), row.get("field_id"))
            for row in current.get("extraction_review_queue") or []
        }
        if key not in queue_keys:
            if any(_matches_applied_entry(entry, params=params, suggestion=suggestion) for entry in journal):
                already_applied += 1
                continue
            raise ValueError(
                f"decision key {key!r} is not currently queued and has no identical applied journal entry"
            )
        update = apply_extraction_resolution_decision(
            current,
            packet,
            **params,
            suggestion_file=suggestion_path,
        )
        current.update(update)
        on_state_update(update)
        applied += 1

    return AppliedDecisionBatchResult(
        state=current,
        applied=applied,
        already_applied=already_applied,
        remaining_queue=len(current.get("extraction_review_queue") or []),
    )
