"""Track 3 continuation plan (squishy-petting-starlight.md) Unit I --
claim-evidence audit (codex point 9): per-claim evidence resolution,
single-study-support flagging, cross-study categorical/boolean conflict
detection, and RQ/target coverage gaps.

Pure aggregation over already-existing `SynthesisClaim`/`EvidenceReference`/
`FieldResult`/`EvidenceIndexEntry` schemas -- no new evidence model, no
disk access of its own. `ClaimAuditSources` is the ONE input shape this
module needs; the caller (`nodes/reporting.py`) is responsible for
assembling it from an already-verified `ReportingSourceSnapshot`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from slr_pipeline.extraction.results import FieldResult
from slr_pipeline.state import EvidenceReference, SynthesisClaim, SynthesisOutput
from slr_pipeline.synthesis_evidence import EvidenceIndexEntry


@dataclass(frozen=True)
class ClaimAuditSources:
    """Everything `build_claim_evidence_audit()` needs. `extraction_field_
    results`/the two evidence indexes are keyed exactly like `evidence_
    index` itself (`record_id -> field_id -> ...`) -- the same shape
    `nodes/synthesis.py`/`nodes/reporting.py` already use."""

    synthesis_output: SynthesisOutput
    extraction_field_results: Dict[str, Dict[str, FieldResult]]
    extraction_evidence_index: Dict[str, Dict[str, List[EvidenceIndexEntry]]]
    qa_evidence_index: Dict[str, Dict[str, List[EvidenceIndexEntry]]]


class ResolvedEvidenceReference(BaseModel):
    """One `EvidenceReference`, resolved back to its evidence-index entry's
    quotation/page/chunk data where a match exists. `chunk_id`/`quotation`/
    `page`/`section` stay `None` if the index has no matching evidence_id
    for this (record_id, field_id) -- surfaced, never hidden, since that
    itself is diagnostic (a claim citing evidence the index cannot resolve)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_kind: str
    record_id: str
    field_id: str
    evidence_id: str
    chunk_id: Optional[str] = None
    quotation: Optional[str] = None
    page: Optional[str] = None
    section: Optional[str] = None


class ClaimAuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str
    rq_or_section: str
    record_ids: List[str]
    resolved_references: List[ResolvedEvidenceReference]
    single_study_supported: bool
    conflicting_evidence: bool
    conflicting_field_ids: List[str] = Field(default_factory=list)


class ClaimEvidenceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claims: List[ClaimAuditEntry]
    unsupported_targets: List[str]


def _resolve_reference(ref: EvidenceReference, sources: ClaimAuditSources) -> ResolvedEvidenceReference:
    index = sources.qa_evidence_index if ref.artifact_kind == "quality_appraisal" else sources.extraction_evidence_index
    entries = index.get(ref.record_id, {}).get(ref.field_id, [])
    match = next((e for e in entries if e.evidence_id == ref.evidence_id), None)
    return ResolvedEvidenceReference(
        artifact_kind=ref.artifact_kind, record_id=ref.record_id, field_id=ref.field_id,
        evidence_id=ref.evidence_id,
        chunk_id=match.chunk_id if match else None, quotation=match.quotation if match else None,
        page=match.page if match else None, section=match.section if match else None,
    )


def _categorical_or_boolean_values(field_result: FieldResult) -> set:
    """`typed_value` is `None` for text/text_list (no typed form), an
    `int`/`float` for integer/number (normal cross-study variation, never
    flagged), and a `bool`/`str` for boolean/categorical respectively (the
    ONLY kinds this audit's conflict rule applies to) -- see `schema.
    validate_value_for_field()`'s own compatibility matrix, which is what
    guarantees `typed_value`'s Python type already discriminates these
    kinds without needing the `ExtractionSchema` itself here."""
    return {v.typed_value for v in field_result.values
            if isinstance(v.typed_value, (bool, str))}


def _conflicting_field_ids(claim: SynthesisClaim, sources: ClaimAuditSources) -> List[str]:
    """A field_id is flagged when two or more DISTINCT records this claim
    cites report categorical/boolean values for that field_id whose value
    sets are mutually exclusive (share nothing) -- per this unit's own
    spec, never triggered by numeric/free-text values (both excluded by
    `_categorical_or_boolean_values()` above). Only `artifact_kind ==
    "extraction"` references are considered: quality-appraisal results
    carry no `ExtractedValue.typed_value`/`ValueKind` concept to apply
    this same rule to."""
    values_by_field_and_record: Dict[str, Dict[str, set]] = {}
    for ref in claim.references:
        if ref.artifact_kind != "extraction":
            continue
        field_result = sources.extraction_field_results.get(ref.record_id, {}).get(ref.field_id)
        if field_result is None:
            continue
        values = _categorical_or_boolean_values(field_result)
        if not values:
            continue
        values_by_field_and_record.setdefault(ref.field_id, {})[ref.record_id] = values

    conflicting: List[str] = []
    for field_id, by_record in values_by_field_and_record.items():
        record_ids = list(by_record)
        if len(record_ids) < 2:
            continue
        if any(by_record[record_ids[i]].isdisjoint(by_record[record_ids[j]])
               for i in range(len(record_ids)) for j in range(i + 1, len(record_ids))):
            conflicting.append(field_id)
    return sorted(conflicting)


def _audit_claim(claim: SynthesisClaim, sources: ClaimAuditSources) -> ClaimAuditEntry:
    record_ids = sorted({r.record_id for r in claim.references})
    conflicting_field_ids = _conflicting_field_ids(claim, sources)
    return ClaimAuditEntry(
        claim_id=claim.claim_id, rq_or_section=claim.rq_or_section, record_ids=record_ids,
        resolved_references=[_resolve_reference(r, sources) for r in claim.references],
        single_study_supported=len(record_ids) == 1,
        conflicting_evidence=bool(conflicting_field_ids),
        conflicting_field_ids=conflicting_field_ids,
    )


def build_claim_evidence_audit(synthesis_state: ClaimAuditSources) -> ClaimEvidenceAudit:
    output = synthesis_state.synthesis_output
    claims = [_audit_claim(c, synthesis_state) for c in output.claims]
    covered_targets = {c.rq_or_section for c in output.claims}
    unsupported_targets = [t for t in output.requested_targets if t not in covered_targets]
    return ClaimEvidenceAudit(claims=claims, unsupported_targets=unsupported_targets)
