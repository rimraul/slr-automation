"""Hash-chained provenance journal for human extraction resolutions.

The authoritative resolved values remain in ``extraction_resolutions.json``.
This companion JSONL records why a human made each decision and, when an LLM
suggestion was involved, binds that decision to one immutable attempt file.
All mutating helpers here assume the caller already holds the extraction
resolution artifact's OS lock; ``nodes.extraction`` owns that transaction.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, model_validator

from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.resolution_suggestions import EvidenceItem, ReviewPacket


DecisionOrigin = Literal["human", "machine_recommended_human_approved", "machine_only"]
DecisionStatus = Literal["prepared", "applied", "failed"]


def decision_journal_path(store: ExtractionCheckpointStore) -> Path:
    return store.run_dir / "extraction_resolution_decisions.jsonl"


def compute_journal_id(
    *, run_id: str, record_id: str, field_id: str, suggestion_id: Optional[str],
    self_hash: Optional[str], resolved_at: str,
) -> str:
    return sha256_hex({
        "id_schema_version": 1,
        "run_id": run_id,
        "record_id": record_id,
        "field_id": field_id,
        "suggestion_id": suggestion_id,
        "self_hash": self_hash,
        "resolved_at": resolved_at,
    })


class ResolutionDecisionIntent(BaseModel):
    """Human-approved decision before journal chain fields are assigned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    record_id: str
    field_id: str
    outcome: Literal["extracted", "not_reported"]
    proposed_value: Optional[Any] = None
    final_value: Optional[Any] = None
    selected_evidence_ids: Tuple[str, ...] = ()
    reviewer: str
    note: str
    resolved_at: str
    attested: Optional[bool] = None
    decision_origin: DecisionOrigin
    suggestion_id: Optional[str] = None
    self_hash: Optional[str] = None
    attempt_file_path: Optional[str] = None
    model: Optional[str] = None
    prompt_template_version: Optional[str] = None
    call_event_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_semantics(self) -> "ResolutionDecisionIntent":
        for name in ("run_id", "record_id", "field_id", "reviewer", "note", "resolved_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        try:
            datetime.fromisoformat(self.resolved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("resolved_at must be an ISO-8601 timestamp") from exc
        if len(set(self.selected_evidence_ids)) != len(self.selected_evidence_ids):
            raise ValueError("selected_evidence_ids must not contain duplicates")
        if self.outcome == "extracted":
            if self.final_value is None or not self.selected_evidence_ids or self.attested is not True:
                raise ValueError("outcome='extracted' requires final_value, evidence IDs, and attested=True")
        elif self.final_value is not None or self.selected_evidence_ids or self.attested is not None:
            raise ValueError("outcome='not_reported' cannot carry final_value, evidence IDs, or attestation")

        provenance = (
            self.suggestion_id, self.self_hash, self.attempt_file_path, self.model,
            self.prompt_template_version, self.call_event_id,
        )
        if self.decision_origin == "machine_only":
            raise ValueError("decision_origin='machine_only' is reserved and disabled in human-gated mode")
        if self.decision_origin == "human":
            if any(value is not None for value in provenance) or self.proposed_value is not None:
                raise ValueError("decision_origin='human' cannot carry suggestion provenance")
        elif any(not isinstance(value, str) or not value.strip() for value in provenance):
            raise ValueError(
                "machine_recommended_human_approved requires suggestion_id, self_hash, attempt_file_path, "
                "model, prompt_template_version, and call_event_id"
            )
        return self


class ResolutionDecisionEntry(ResolutionDecisionIntent):
    """One immutable JSONL line in the decision journal."""

    status: DecisionStatus
    journal_id: str
    reason: Optional[str] = None
    previous_entry_hash: Optional[str] = None
    entry_hash: str

    @model_validator(mode="after")
    def _validate_identity_and_hash(self) -> "ResolutionDecisionEntry":
        expected_id = compute_journal_id(
            run_id=self.run_id, record_id=self.record_id, field_id=self.field_id,
            suggestion_id=self.suggestion_id, self_hash=self.self_hash, resolved_at=self.resolved_at,
        )
        if self.journal_id != expected_id:
            raise ValueError("journal_id does not match this entry's decision identity")
        if self.status == "failed":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("status='failed' requires a non-blank reason")
        elif self.reason is not None:
            raise ValueError("only status='failed' may carry a reason")
        expected_hash = sha256_hex({
            k: v for k, v in self.model_dump(mode="json").items() if k != "entry_hash"
        })
        if self.entry_hash != expected_hash:
            raise ValueError("entry_hash does not match this journal entry's content")
        return self


def build_decision_entry(
    intent: ResolutionDecisionIntent, *, status: DecisionStatus,
    previous_entry_hash: Optional[str], reason: Optional[str] = None,
) -> ResolutionDecisionEntry:
    # ``intent`` may be a previously-read ResolutionDecisionEntry when a
    # terminal line is appended during reconciliation.  Copy only the base
    # intent fields so the prior line's chain/status fields cannot leak into
    # the new entry or its hash input.
    payload = {
        name: getattr(intent, name)
        for name in ResolutionDecisionIntent.model_fields
    }
    payload.update(
        status=status,
        journal_id=compute_journal_id(
            run_id=intent.run_id, record_id=intent.record_id, field_id=intent.field_id,
            suggestion_id=intent.suggestion_id, self_hash=intent.self_hash, resolved_at=intent.resolved_at,
        ),
        reason=reason,
        previous_entry_hash=previous_entry_hash,
    )
    payload["entry_hash"] = sha256_hex(payload)
    return ResolutionDecisionEntry.model_validate(payload)


def _parse_lines(data: bytes, *, allow_torn_tail: bool) -> Tuple[list[ResolutionDecisionEntry], bytes, bool]:
    entries: list[ResolutionDecisionEntry] = []
    valid_lines: list[bytes] = []
    lines = data.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            raise ValueError(f"decision journal contains a blank line at line {index + 1}")
        try:
            decoded = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if allow_torn_tail and index == len(lines) - 1:
                return entries, b"".join(valid_lines), True
            raise ValueError(f"invalid decision journal line {index + 1}: {exc}") from exc
        try:
            entry = ResolutionDecisionEntry.model_validate(decoded)
        except Exception as exc:
            # A syntactically-complete but invalid/tampered final line is not
            # a torn write and must never be truncated away as "repair".
            raise ValueError(f"invalid decision journal line {index + 1}: {exc}") from exc
        expected_previous = entries[-1].entry_hash if entries else None
        if entry.previous_entry_hash != expected_previous:
            raise ValueError(
                f"decision journal hash chain is broken at line {index + 1}: expected "
                f"previous_entry_hash={expected_previous!r}, got {entry.previous_entry_hash!r}"
            )
        entries.append(entry)
        valid_lines.append(json.dumps(entry.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8") + b"\n")
    canonical_prefix = b"".join(valid_lines)
    # A complete JSON object without its terminating newline is a possible
    # partial final append too.  Keep the valid entry, but force the repair
    # path to rewrite the canonical newline before any future append.
    unterminated = bool(data and not data.endswith(b"\n"))
    if unterminated and not allow_torn_tail:
        raise ValueError("decision journal has an unterminated final line; reconcile it before appending")
    return entries, canonical_prefix, unterminated


def read_decision_journal(store: ExtractionCheckpointStore) -> list[ResolutionDecisionEntry]:
    path = decision_journal_path(store)
    if not path.exists():
        return []
    entries, _prefix, _torn = _parse_lines(path.read_bytes(), allow_torn_tail=False)
    return entries


def verify_decision_journal_bytes(data: bytes) -> list[ResolutionDecisionEntry]:
    """Read-only verification for a published journal.

    Released bundles are immutable, so a torn tail is an integrity error;
    the repair path remains restricted to a live workspace.
    """
    entries, _canonical_prefix, _torn = _parse_lines(data, allow_torn_tail=False)
    validate_journal_transitions(entries)
    return entries


def repair_torn_journal_tail(store: ExtractionCheckpointStore) -> list[ResolutionDecisionEntry]:
    """Repair only an unparseable final line; broken valid chain links fail."""
    path = decision_journal_path(store)
    if not path.exists():
        return []
    entries, prefix, torn = _parse_lines(path.read_bytes(), allow_torn_tail=True)
    if not torn:
        return entries
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(prefix)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    from slr_pipeline.extraction.resolution_suggestions import _fsync_dir
    _fsync_dir(path.parent)
    return entries


def append_decision_entry(
    store: ExtractionCheckpointStore, intent: ResolutionDecisionIntent, *, status: DecisionStatus,
    reason: Optional[str] = None,
) -> ResolutionDecisionEntry:
    """Append+fsync one line. Caller must hold the resolution artifact lock."""
    entries = read_decision_journal(store)
    validate_journal_transitions(entries)
    journal_id = compute_journal_id(
        run_id=intent.run_id, record_id=intent.record_id, field_id=intent.field_id,
        suggestion_id=intent.suggestion_id, self_hash=intent.self_hash, resolved_at=intent.resolved_at,
    )
    if status == "prepared":
        if any(entry.journal_id == journal_id for entry in entries):
            raise ValueError(f"decision journal already contains journal_id={journal_id!r}")
    else:
        matching = [entry for entry in entries if entry.journal_id == journal_id]
        if len(matching) != 1 or matching[0].status != "prepared":
            raise ValueError(
                f"cannot append status={status!r} for journal_id={journal_id!r}: "
                "exactly one unterminated prepared entry is required"
            )
        prepared_payload = {
            name: getattr(matching[0], name)
            for name in ResolutionDecisionIntent.model_fields
        }
        terminal_payload = {
            name: getattr(intent, name)
            for name in ResolutionDecisionIntent.model_fields
        }
        if terminal_payload != prepared_payload:
            raise ValueError(
                f"cannot append status={status!r} for journal_id={journal_id!r}: "
                "terminal payload differs from its prepared entry"
            )
    entry = build_decision_entry(
        intent, status=status,
        previous_entry_hash=entries[-1].entry_hash if entries else None,
        reason=reason,
    )
    path = decision_journal_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    from slr_pipeline.extraction.resolution_suggestions import _fsync_dir
    _fsync_dir(path.parent)
    return entry


def packet_entry(packet: ReviewPacket, record_id: str, field_id: str):
    matches = [entry for entry in packet.entries if entry.record_id == record_id and entry.field_id == field_id]
    if len(matches) != 1:
        raise ValueError(
            f"review packet must contain exactly one entry for {(record_id, field_id)!r}, got {len(matches)}"
        )
    return matches[0]


def evidence_locator(item: EvidenceItem) -> str:
    parts = [f"chunk={item.chunk_id}"]
    if item.page is not None:
        parts.append(f"page={item.page}")
    if item.section:
        parts.append(f"section={item.section}")
    return "; ".join(parts)


def rehydrate_evidence(
    packet: ReviewPacket, *, record_id: str, field_id: str, evidence_ids: Sequence[str],
) -> list[Dict[str, str]]:
    entry = packet_entry(packet, record_id, field_id)
    rehydrated: list[Dict[str, str]] = []
    for evidence_id in evidence_ids:
        item = entry.evidence_index.get(evidence_id)
        if item is None:
            raise ValueError(
                f"selected evidence ID {evidence_id!r} is absent from the verified review-packet entry "
                f"for {(record_id, field_id)!r}"
            )
        rehydrated.append({
            "quotation_or_description": item.excerpt,
            "locator": evidence_locator(item),
        })
    return rehydrated


def resolution_matches_intent(
    resolution: Dict[str, Any], intent: ResolutionDecisionIntent, packet: ReviewPacket,
) -> bool:
    expected: Dict[str, Any] = {
        "outcome": intent.outcome,
        "reviewer": intent.reviewer.strip(),
        "note": intent.note.strip(),
        "resolved_at": intent.resolved_at.strip(),
    }
    if intent.outcome == "extracted":
        expected.update(
            value=intent.final_value,
            evidence=rehydrate_evidence(
                packet, record_id=intent.record_id, field_id=intent.field_id,
                evidence_ids=intent.selected_evidence_ids,
            ),
            attested=True,
        )
    else:
        for forbidden in ("value", "evidence", "attested"):
            if forbidden in resolution:
                return False
    return all(resolution.get(name) == value for name, value in expected.items())


def validate_journal_transitions(entries: Iterable[ResolutionDecisionEntry]) -> None:
    by_id: Dict[str, list[ResolutionDecisionEntry]] = {}
    for entry in entries:
        by_id.setdefault(entry.journal_id, []).append(entry)
    for journal_id, grouped in by_id.items():
        if grouped[0].status != "prepared" or len(grouped) > 2:
            raise ValueError(f"journal_id={journal_id!r} has an invalid prepared/terminal transition sequence")
        if len(grouped) == 2 and grouped[1].status not in ("applied", "failed"):
            raise ValueError(f"journal_id={journal_id!r} has no valid terminal transition")
        baseline = grouped[0].model_dump(mode="json", exclude={"status", "reason", "previous_entry_hash", "entry_hash"})
        for terminal in grouped[1:]:
            candidate = terminal.model_dump(mode="json", exclude={"status", "reason", "previous_entry_hash", "entry_hash"})
            if candidate != baseline:
                raise ValueError(f"journal_id={journal_id!r} terminal entry changes the prepared decision payload")
