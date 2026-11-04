"""Track 3 continuation plan (squishy-petting-starlight.md) Unit I --
offline tests for `build_claim_evidence_audit()`'s classification rules:
single-study-support (N=1), cross-study categorical/boolean conflict
detection (never triggered by numeric/free-text variation), RQ/target
coverage gaps, and evidence-index page/chunk resolution. No network, no
OpenAI client, no ledger/proxy -- pure pydantic-model fixtures."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Dict, List

from slr_pipeline.continuation.claim_audit import ClaimAuditSources, build_claim_evidence_audit
from slr_pipeline.extraction.results import EvidenceSpanModel, ExtractedValue, FieldResult, FieldStatus
from slr_pipeline.state import (
    EvidenceReference, SupportingFieldSelector, SynthesisClaim, build_map_answer, build_synthesis_claim,
)
from slr_pipeline.synthesis_evidence import EvidenceIndexEntry, make_evidence_id

# `build_claim_evidence_audit()` only ever reads `.claims`/`.requested_targets` off
# `synthesis_state.synthesis_output` (see claim_audit.py) -- a `SimpleNamespace` stand-in avoids
# needing to satisfy `SynthesisOutput`'s own much larger topology/refine-batch-replay validator
# (map_answers/map_outcome_records/refine_batch_outcomes chain consistency), which is orthogonal
# to what this module tests. `ClaimAuditSources` is a plain dataclass (not pydantic), so it does
# not itself enforce the `SynthesisOutput` type hint at runtime.

_EVENT_ID = "aiuse-1b8a17e43ecec6f819c59e7b9c0df9b58b30987516759744"


def _event_id(seed: str) -> str:
    return "aiuse-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:48]


def _field_result(field_id: str, typed_values, *, chunk_id: str = "c1", quotation: str = "quote") -> FieldResult:
    ev = EvidenceSpanModel(chunk_id=chunk_id, excerpt=quotation, page="1", section="Results")
    values = tuple(ExtractedValue(value=str(tv), typed_value=tv, evidence=(ev,)) for tv in typed_values)
    return FieldResult(field_id=field_id, status=FieldStatus.EXTRACTED, values=values, requires_human_review=False)


def _reference(*, record_id: str, field_id: str, artifact_kind: str = "extraction", run_id: str = "ext-run-1",
               chunk_id: str = "c1", quotation: str = "quote", page: str = "1", section: str = "Results") -> EvidenceReference:
    evidence_id = make_evidence_id(
        artifact_kind=artifact_kind, artifact_run_id=run_id, record_id=record_id, field_id=field_id,
        chunk_id=chunk_id, quotation=quotation, page=page, section=section, locator=None,
    )
    return EvidenceReference(artifact_kind=artifact_kind, artifact_run_id=run_id, record_id=record_id,
                              field_id=field_id, evidence_id=evidence_id)


def _index_entry(ref: EvidenceReference, *, chunk_id: str = "c1", quotation: str = "quote",
                  page: str = "1", section: str = "Results") -> EvidenceIndexEntry:
    return EvidenceIndexEntry(evidence_id=ref.evidence_id, chunk_id=chunk_id, quotation=quotation,
                               page=page, section=section, locator=None)


def _map_answer_for_record(rq_or_section: str, record_id: str, refs: List[EvidenceReference]):
    selectors = sorted({(r.artifact_kind, r.field_id) for r in refs})
    return build_map_answer(
        rq_or_section=rq_or_section, record_id=record_id, answer_text=f"Answer for {record_id}.",
        selectors=[SupportingFieldSelector(artifact_kind=k, field_id=f) for k, f in selectors],
        evidence_references=refs, producing_event_id=_event_id(f"answer|{rq_or_section}|{record_id}"),
    )


def _claim(rq_or_section: str, refs: List[EvidenceReference]):
    """Builds one covered target's `SynthesisClaim` plus the `MapAnswer`s it was built from
    (one `MapAnswer` per distinct record_id, since a `MapAnswer`'s own evidence_references must
    all share its record_id)."""
    by_record: Dict[str, List[EvidenceReference]] = {}
    for r in refs:
        by_record.setdefault(r.record_id, []).append(r)
    answers = [_map_answer_for_record(rq_or_section, rid, rs) for rid, rs in by_record.items()]
    claim = build_synthesis_claim(f"Claim text for {rq_or_section}.", answers, rq_or_section, _EVENT_ID)
    return claim, answers


def _synthesis_output(targets_with_refs: Dict[str, List[EvidenceReference]], requested_targets: List[str]) -> SimpleNamespace:
    """`targets_with_refs` maps each COVERED target (one that gets a claim) to the evidence
    references its claim cites; any `requested_targets` entry absent from `targets_with_refs`
    is left uncovered (no claim) -- exactly the shape the audit's own unsupported_targets rule
    inspects. Real `SynthesisClaim` instances (built via `build_synthesis_claim()`/`build_map_answer()`,
    both self-contained pydantic validators with no event-store dependency), wrapped in a
    `SimpleNamespace` standing in for the much larger `SynthesisOutput` model -- see note above."""
    claims: List[SynthesisClaim] = [_claim(rq, refs)[0] for rq, refs in targets_with_refs.items()]
    return SimpleNamespace(claims=claims, requested_targets=requested_targets)


def _sources(targets_with_refs: Dict[str, List[EvidenceReference]], requested_targets: List[str],
             field_results: Dict[str, Dict[str, FieldResult]], index_entries: List[EvidenceIndexEntry]) -> ClaimAuditSources:
    extraction_index: Dict[str, Dict[str, List[EvidenceIndexEntry]]] = {}
    for entry, ref in index_entries:
        extraction_index.setdefault(ref.record_id, {}).setdefault(ref.field_id, []).append(entry)
    return ClaimAuditSources(
        synthesis_output=_synthesis_output(targets_with_refs, requested_targets),
        extraction_field_results=field_results,
        extraction_evidence_index=extraction_index,
        qa_evidence_index={},
    )


def test_single_study_supported_claim_flagged_n1():
    ref = _reference(record_id="rec-1", field_id="country")
    target = "What countries were studied?"
    field_results = {"rec-1": {"country": _field_result("country", ["Lithuania"])}}
    audit = build_claim_evidence_audit(_sources({target: [ref]}, [target], field_results, [(_index_entry(ref), ref)]))

    entry = audit.claims[0]
    assert entry.single_study_supported is True
    assert entry.record_ids == ["rec-1"]
    assert entry.conflicting_evidence is False


def test_multi_study_claim_not_single_study_supported():
    ref1 = _reference(record_id="rec-1", field_id="sample_size", quotation="n=100")
    ref2 = _reference(record_id="rec-2", field_id="sample_size", quotation="n=250")
    target = "What sample sizes were reported?"
    field_results = {
        "rec-1": {"sample_size": _field_result("sample_size", [100])},
        "rec-2": {"sample_size": _field_result("sample_size", [250])},
    }
    audit = build_claim_evidence_audit(
        _sources({target: [ref1, ref2]}, [target], field_results,
                 [(_index_entry(ref1), ref1), (_index_entry(ref2), ref2)])
    )

    entry = audit.claims[0]
    assert entry.single_study_supported is False
    assert entry.record_ids == ["rec-1", "rec-2"]
    # numeric variation across studies is normal, never flagged as conflicting evidence
    assert entry.conflicting_evidence is False
    assert entry.conflicting_field_ids == []


def test_categorical_disjoint_values_flagged_conflicting():
    ref1 = _reference(record_id="rec-1", field_id="study_design", quotation="an RCT")
    ref2 = _reference(record_id="rec-2", field_id="study_design", quotation="an observational study")
    target = "What designs were used?"
    field_results = {
        "rec-1": {"study_design": _field_result("study_design", ["RCT"])},
        "rec-2": {"study_design": _field_result("study_design", ["observational"])},
    }
    audit = build_claim_evidence_audit(
        _sources({target: [ref1, ref2]}, [target], field_results,
                 [(_index_entry(ref1), ref1), (_index_entry(ref2), ref2)])
    )

    entry = audit.claims[0]
    assert entry.conflicting_evidence is True
    assert entry.conflicting_field_ids == ["study_design"]


def test_categorical_overlapping_values_not_flagged_conflicting():
    """Two records each report a set of categorical values -- they share
    at least one value, so the sets are NOT mutually exclusive."""
    ref1 = _reference(record_id="rec-1", field_id="outcome_type", quotation="improvement and remission")
    ref2 = _reference(record_id="rec-2", field_id="outcome_type", quotation="improvement only")
    target = "What outcomes were reported?"
    field_results = {
        "rec-1": {"outcome_type": _field_result("outcome_type", ["improvement", "remission"])},
        "rec-2": {"outcome_type": _field_result("outcome_type", ["improvement"])},
    }
    audit = build_claim_evidence_audit(
        _sources({target: [ref1, ref2]}, [target], field_results,
                 [(_index_entry(ref1), ref1), (_index_entry(ref2), ref2)])
    )

    entry = audit.claims[0]
    assert entry.conflicting_evidence is False
    assert entry.conflicting_field_ids == []


def test_boolean_disjoint_values_flagged_conflicting():
    ref1 = _reference(record_id="rec-1", field_id="blinded", quotation="double-blind")
    ref2 = _reference(record_id="rec-2", field_id="blinded", quotation="open-label")
    target = "Was blinding used?"
    field_results = {
        "rec-1": {"blinded": _field_result("blinded", [True])},
        "rec-2": {"blinded": _field_result("blinded", [False])},
    }
    audit = build_claim_evidence_audit(
        _sources({target: [ref1, ref2]}, [target], field_results,
                 [(_index_entry(ref1), ref1), (_index_entry(ref2), ref2)])
    )

    entry = audit.claims[0]
    assert entry.conflicting_evidence is True
    assert entry.conflicting_field_ids == ["blinded"]


def test_unsupported_target_flagged_when_no_claims_reference_it():
    ref = _reference(record_id="rec-1", field_id="country")
    covered_target = "What countries were studied?"
    uncovered_target = "What was the funding source?"
    field_results = {"rec-1": {"country": _field_result("country", ["Lithuania"])}}
    targets = [covered_target, uncovered_target]
    audit = build_claim_evidence_audit(
        _sources({covered_target: [ref]}, targets, field_results, [(_index_entry(ref), ref)])
    )

    assert audit.unsupported_targets == [uncovered_target]
    assert len(audit.claims) == 1


def test_resolved_reference_surfaces_page_and_chunk():
    ref = _reference(record_id="rec-1", field_id="country", chunk_id="chunk-7", quotation="Lithuania", page="12")
    target = "What countries were studied?"
    field_results = {"rec-1": {"country": _field_result("country", ["Lithuania"])}}
    entry = _index_entry(ref, chunk_id="chunk-7", quotation="Lithuania", page="12")
    audit = build_claim_evidence_audit(_sources({target: [ref]}, [target], field_results, [(entry, ref)]))

    resolved = audit.claims[0].resolved_references[0]
    assert resolved.chunk_id == "chunk-7"
    assert resolved.quotation == "Lithuania"
    assert resolved.page == "12"


def test_resolved_reference_none_fields_when_evidence_id_not_in_index():
    ref = _reference(record_id="rec-1", field_id="country")
    target = "What countries were studied?"
    field_results = {"rec-1": {"country": _field_result("country", ["Lithuania"])}}
    # no matching index entry supplied at all
    audit = build_claim_evidence_audit(_sources({target: [ref]}, [target], field_results, []))

    resolved = audit.claims[0].resolved_references[0]
    assert resolved.chunk_id is None
    assert resolved.quotation is None
    assert resolved.page is None


def test_quality_appraisal_references_excluded_from_conflict_check():
    """QA `FieldResult`s carry no `ValueKind`/`typed_value` concept -- a
    claim citing conflicting-looking QA evidence on the same field_id is
    never flagged by this rule, only extraction evidence is."""
    ref1 = _reference(record_id="rec-1", field_id="risk_domain", artifact_kind="quality_appraisal", run_id="qa-run-1")
    ref2 = _reference(record_id="rec-2", field_id="risk_domain", artifact_kind="quality_appraisal", run_id="qa-run-1")
    target = "What was the risk of bias?"
    audit = build_claim_evidence_audit(
        ClaimAuditSources(
            synthesis_output=_synthesis_output({target: [ref1, ref2]}, [target]),
            extraction_field_results={}, extraction_evidence_index={},
            qa_evidence_index={"rec-1": {"risk_domain": [_index_entry(ref1)]},
                                "rec-2": {"risk_domain": [_index_entry(ref2)]}},
        )
    )

    entry = audit.claims[0]
    assert entry.conflicting_evidence is False
    assert entry.resolved_references[0].artifact_kind == "quality_appraisal"
    assert entry.resolved_references[0].chunk_id == "c1"
