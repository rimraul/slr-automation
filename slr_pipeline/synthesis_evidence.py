"""Shared evidence-index construction for structured synthesis claims."""
import hashlib
import json
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slr_pipeline.extraction.evidence import normalize_for_matching

_EVIDENCE_ID_RE = re.compile(r"^evidence-[0-9a-f]{64}$")
_ARTIFACT_KINDS = ("extraction", "quality_appraisal")


def make_evidence_id(*, artifact_kind: str, artifact_run_id: str, record_id: str, field_id: str,
                      chunk_id: str, quotation: str, page: Optional[str], section: Optional[str],
                      locator: Optional[str]) -> str:
    """Diverges from the parent design doc's own narrower signature/hash (this plan's own third
    named divergence -- quality_appraisal_synthesis_plan_2026-08-22.md's own make_evidence_id()
    hashes only (chunk_id, normalize_for_matching(quotation))) -- needed because this module's own
    EvidenceIndexEntry also preserves page/section/locator, and two genuinely different attestations
    of the same quoted text at different locators must not collide. page/section/locator are
    REQUIRED keyword arguments -- no default -- so a caller can never silently omit an
    identity-relevant field. Every text field is hashed RAW, exactly as stored on
    EvidenceIndexEntry -- normalize_for_matching() is deliberately NOT applied here: normalizing
    the hash input while storing the raw value on the entry would break "same evidence_id implies
    same stored content" (e.g. "A  B" and "A B" normalize identically but are different raw
    strings) -- hashing raw values makes evidence_id a true, unambiguous content-address over
    exactly what is stored."""
    if artifact_kind not in _ARTIFACT_KINDS:
        raise ValueError(f"artifact_kind must be one of {_ARTIFACT_KINDS}, got {artifact_kind!r}")
    payload = {"artifact_kind": artifact_kind, "artifact_run_id": artifact_run_id,
               "record_id": record_id, "field_id": field_id, "chunk_id": chunk_id,
               "quotation": quotation, "page": page, "section": section, "locator": locator}
    return "evidence-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _human_attested_chunk_id(locator: str) -> str:
    """A pure function of locator whose OWN output (the derived chunk_id string) is what actually
    gets stored/hashed everywhere downstream; normalizing INSIDE this function carries none of
    make_evidence_id()'s own raw-vs-normalized mismatch, since there is no separate raw copy of the
    derived chunk_id stored anywhere to compare against."""
    return "human-attested:" + hashlib.sha256(normalize_for_matching(locator).encode()).hexdigest()[:16]


class EvidenceIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    chunk_id: str = Field(min_length=1)
    quotation: str = Field(min_length=1)
    page: Optional[str] = None
    section: Optional[str] = None
    locator: Optional[str] = None

    @field_validator("chunk_id", "quotation")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank/whitespace-only")
        return v

    @field_validator("page", "section", "locator")
    @classmethod
    def _non_blank_if_present(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("if provided, must not be blank/whitespace-only")
        return v

    @field_validator("evidence_id")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if not _EVIDENCE_ID_RE.fullmatch(v):
            raise ValueError(f"evidence_id {v!r} is not well-formed")
        return v


def _build_evidence_index_entry(*, artifact_kind: str, artifact_run_id: str, record_id: str, field_id: str,
                                 chunk_id: str, quotation: str, page: Optional[str], section: Optional[str],
                                 locator: Optional[str]) -> EvidenceIndexEntry:
    """page/section/locator have NO default -- every caller (extraction.py/quality_appraisal.py's
    own loader integration) must pass all three explicitly."""
    return EvidenceIndexEntry(
        evidence_id=make_evidence_id(artifact_kind=artifact_kind, artifact_run_id=artifact_run_id,
                                      record_id=record_id, field_id=field_id, chunk_id=chunk_id,
                                      quotation=quotation, page=page, section=section, locator=locator),
        chunk_id=chunk_id, quotation=quotation, page=page, section=section, locator=locator,
    )


def build_evidence_index_for_field(entries_raw: List[Dict[str, Optional[str]]], *, artifact_kind: str,
                                    artifact_run_id: str, record_id: str, field_id: str) -> List[EvidenceIndexEntry]:
    """`entries_raw`: plain dicts, unpacked via **raw -- each dict MUST contain page/section/locator
    keys explicitly (None where a field does not apply), or construction raises TypeError, per
    _build_evidence_index_entry()'s own now-required keyword arguments. Dedups on the COMPLETE
    entry (pydantic's own default __eq__ compares every field). With evidence_id hashing every
    preserved field RAW, hash-equality is logically equivalent to complete-stored-field-equality,
    so a same-evidence_id/different-content pair is not a legitimate, reachable outcome -- the
    RuntimeError below is defense-in-depth against a genuine SHA-256 hash collision only."""
    built = [_build_evidence_index_entry(artifact_kind=artifact_kind, artifact_run_id=artifact_run_id,
                                          record_id=record_id, field_id=field_id, **raw) for raw in entries_raw]
    deduped: Dict[str, EvidenceIndexEntry] = {}
    for entry in built:
        existing = deduped.get(entry.evidence_id)
        if existing is None:
            deduped[entry.evidence_id] = entry
        elif existing == entry:
            continue   # a genuine, byte-identical duplicate -- safe to drop
        else:
            raise RuntimeError(
                f"evidence_id collision within {artifact_kind}/{record_id}/{field_id}: two entries "
                f"share evidence_id {entry.evidence_id!r} but differ in content -- defense-in-depth "
                f"only, since evidence_id now hashes every preserved field RAW (no normalization) "
                f"and should make this structurally unreachable for legitimate data -- "
                f"{existing!r} vs {entry!r}"
            )
    return list(deduped.values())
