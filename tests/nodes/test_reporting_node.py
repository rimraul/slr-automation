"""Tests for `nodes/reporting.py` -- the aggregation/export/reporting phase
(`aggregation_export_reporting_plan_2026-08-25.md` v17).

Structure:
  1. `_render_methods_md()` content tests -- adapted from the pre-v17
     `_draft_methods_text()` suite (codex review Phase B finding 1 /
     Extraction Phase B), now driven through a hand-built
     `ReportingSourceSnapshot` instead of loose positional arguments.
  2. Pure-function/model unit tests for the plan's own acceptance matrix
     (§5): EvidenceMapModel validation (T1/U1/X3), the evidence-map
     reconstruction-and-compare (W1/X1/X2/Y1/Y2), BERTopic consistency
     (T2/U2/U3) and its five rendered-note branches (S5), `_load_component`
     containment/single-read (T3), FT native-provenance path safety,
     report-input validation (ReportingIRRState/ReportingScreeningQCState),
     PRISMA render safety, `_write_artifact_and_verify`/manifest self-
     verification, `compute_selection_splits` (INV-R6), and INV-R16.
  3. A full-pipeline smoke test per review type (scoping/rapid/slr),
     exercising `capture_reporting_snapshot()`/`reporting_node()` end to
     end over a small, hand-orchestrated corpus.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore, build_ai_use_event
from slr_pipeline.config import QualityAppraisalConfig, ReviewConfig, ScreeningConfig, SynthesisConfig
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.extraction.openai_runner import (
    BatchCallTelemetry, OpenAIExtractionConfig, RawBatchResponse, RawEvidenceItem, RawFieldResponse, RawFinding,
)
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind
from slr_pipeline.selection_ledger import SelectionDecisionStore, derive_source_event_id, emit_selection_decision
from slr_pipeline.human_resolution import compute_resolution_id
from slr_pipeline.state import PRISMACounts, Record, SearchLog, SearchStrategy, SynthesisOutput, initial_state

import slr_pipeline.nodes.dedup as dedup_module
from slr_pipeline.nodes.dedup import dedup_node
import slr_pipeline.nodes.screening as screening_module
from slr_pipeline.nodes.screening import (
    ScreeningBatchResultStore, ScreeningResponse, build_screening_batch_result, compute_current_screening_batches,
    compute_screening_batch_identity, screening_node,
)
import slr_pipeline.extraction.engine as engine_module
import slr_pipeline.nodes.extraction as extraction_module
from slr_pipeline.nodes.extraction import extraction_node
import slr_pipeline.nodes.quality_appraisal as qa_module
from slr_pipeline.nodes.quality_appraisal import (
    RawAppraisalEvidenceItem, RawTeachingBatchResponse, RawTeachingItemBatchResponse, TEACHING_APPRAISAL_ITEMS,
    quality_appraisal_node,
)
import slr_pipeline.nodes.synthesis as synthesis_module
from slr_pipeline.nodes.synthesis import synthesis_node
import slr_pipeline.nodes.search_strategy_builder as search_strategy_builder_module
from slr_pipeline.nodes.search_strategy_builder import (
    canonical_query_expansion_output, compute_db_translation_call_identity, compute_query_expansion_call_identity,
    search_strategy_builder_node,
)
from slr_pipeline.nodes.screening_qc import compute_qc_second_rater_call_identity

import slr_pipeline.nodes.reporting as reporting_module
from slr_pipeline.nodes.reporting import (
    ReportingConsistencyError, ReportingConfigProjection, ReportingIRRState, ReportingScreeningQCState,
    ReportingSourceSnapshot, SourceBundle, ValidatedStateValue, VerifiedComponent, VerifiedFile, VerifiedProjection,
    EvidenceMapModel, _canonicalize_for_hash, _hash_value, _load_component, _pydantic_json_mode,
    _reconstruct_expected_evidence_map, _render_methods_md, _state_value, _validate_bertopic_consistency,
    _write_artifact_and_verify, capture_ft_native_provenance, capture_reporting_snapshot, compute_selection_splits,
    render_bertopic_note, reporting_node,
)
from slr_pipeline.nodes.synthesis import canonicalize_corpus_order, _build_evidence_map
from slr_pipeline.export_manifest import ReportingStageManifest

from datetime import datetime, timezone
from ft_screening import ScreeningItemResult, ScreeningRunResult
from ft_screening.schema import RunArtifacts, RunSummary


def _valid_sha256(seed: str) -> str:
    return sha256_hex(seed)


def _build_valid_ft_screening_run_files(
    out_dir: Path, *, run_id: str, execution_fingerprint: str, record_ids: List[str],
) -> Dict[str, str]:
    """Codex round-5 review, finding 1: a genuinely valid, schema-complete
    `ScreeningRunResult`/`ScreeningItemResult` pair -- `capture_ft_native_
    provenance()` now fully parses/cross-verifies both via FT_SCREENING's
    own real pydantic models, so a test fixture can no longer get away
    with arbitrary bytes or an untyped `{"document_id": ...}` dict. Every
    item uses `execution_status="not_processed"` -- the only status that
    needs no `eligibility`/`provenance`/`coverage` payload -- since these
    tests only care about `document_id`/`report_id` identity, not
    eligibility outcomes."""
    items = [
        ScreeningItemResult(
            input_index=i, document_id=rid, report_id=rid, citation={"record_id": rid},
            execution_status="not_processed",
        )
        for i, rid in enumerate(record_ids)
    ]
    run_result_path = out_dir / "run_result.json"
    item_results_path = out_dir / "item_results.jsonl"
    ledger_path = out_dir / "ledger.jsonl"
    run_result = ScreeningRunResult(
        # run_status="failed" (not "completed"): every item here is
        # execution_status="not_processed" (completed=0 in the summary),
        # and ScreeningRunResult._run_status_consistency() requires
        # completed == total for run_status="completed" -- "failed" is the
        # one status consistent with completed == 0 regardless of total.
        # Semantically odd for a fixture that isn't testing failure
        # handling, but these tests only care about document_id/report_id
        # identity, never run_status.
        api_version="1.0", schema_version="1.0", run_id=run_id, run_status="failed",
        criteria_set_id="chapter-criteria", criteria_set_version="v1",
        criteria_set_hash=_valid_sha256("criteria"), execution_fingerprint=execution_fingerprint,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc), completed_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        items=items,
        summary=RunSummary(total=len(items), completed=0, included=0, excluded=0, needs_human_review=0,
                            input_unavailable=0, duplicate_input=0, processing_failed=0,
                            not_processed=len(items), cache_hits=0),
        artifacts=RunArtifacts(run_result_json=run_result_path, item_results_jsonl=item_results_path,
                                ledger_jsonl=ledger_path),
    )
    run_result_path.write_text(run_result.model_dump_json(), encoding="utf-8")
    item_results_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")
    ledger_path.write_text("", encoding="utf-8")
    return {"run_result_json": str(run_result_path), "item_results_jsonl": str(item_results_path),
            "ledger_jsonl": str(ledger_path)}


# ---------------------------------------------------------------------------
# 1. `_render_methods_md()` content tests
# ---------------------------------------------------------------------------

def _prisma(**overrides) -> PRISMACounts:
    defaults = dict(
        assessed_full_text=3, excluded_full_text=1,
        excluded_full_text_reasons={"wrong_population": 1},
        fulltext_retrieval={"parsed": 3, "no_oa_fulltext_found": 2, "rejected_stub": 1},
        included=2,
    )
    defaults.update(overrides)
    return PRISMACounts(**defaults)


def _make_snapshot(
    cfg: ReviewConfig, prisma: PRISMACounts, search_log: SearchLog, *,
    review_queue: Optional[list] = None, extraction_artifact: Optional[Dict[str, Any]] = None,
    screening_review_resolutions: Optional[list] = None,
    screening_qc_state: Optional[ReportingScreeningQCState] = None,
    evidence_map: Optional[EvidenceMapModel] = None,
    quality_appraisal_present: bool = False,
) -> ReportingSourceSnapshot:
    """A minimal snapshot with only the fields `_render_methods_md()` (and
    a handful of other pure renderers) actually reads -- every other
    field is a harmless placeholder. Never constructed via
    `capture_reporting_snapshot()` here; these are unit tests of the
    RENDER function alone."""
    extraction = None
    if extraction_artifact is not None:
        extraction = SourceBundle(
            raw=extraction_artifact,
            manifest=VerifiedComponent(path=Path("manifest.json"), file_sha256="a" * 64, internal_hash="a" * 64),
            checkpoints=[], resolutions=None, confirmations=None, computed_content_hash="b" * 64,
            run_id=extraction_artifact["run_id"],
        )
    quality_appraisal = None
    if quality_appraisal_present:
        quality_appraisal = SourceBundle(
            raw={"resolved_quality_projection": []},
            manifest=VerifiedComponent(path=Path("qa_manifest.json"), file_sha256="e" * 64, internal_hash="e" * 64),
            checkpoints=[], resolutions=None, confirmations=None, computed_content_hash="f" * 64,
            run_id="qarun-abc",
        )
    config_projection = ReportingConfigProjection.from_config(cfg)
    synthesis_output = VerifiedProjection(
        model=SynthesisOutput(refine_batch_size=cfg.synthesis.refine_batch_size), content_hash="c" * 64)
    return ReportingSourceSnapshot(
        review_run_id="run-1", config_projection=config_projection, config_projection_hash="d" * 64,
        raw_records=_state_value([]),
        screening_records=[], full_text_records=[], screened_records_full=[],
        prisma=_state_value(prisma), search_log=_state_value(search_log),
        screening_review_resolutions=_state_value(screening_review_resolutions or []),
        full_text_review_resolutions=_state_value([]),
        full_text_review_queue=_state_value(review_queue if review_queue is not None else []),
        screening_qc_state=screening_qc_state, oversight_mode_state={}, automation_origin_state={},
        events=[], ledger_records=[], current_screening_batch_results=[],
        extraction=extraction, quality_appraisal=quality_appraisal,
        synthesis_output=synthesis_output, evidence_map=evidence_map, ft_native_provenance=None,
    )


def test_methods_text_shows_reports_not_retrieved():
    cfg = ReviewConfig(title="t")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog()))
    assert "Reports not retrieved" in text
    assert "3" in text  # 2 (no_oa_fulltext_found) + 1 (rejected_stub)
    assert "No oa fulltext found" in text or "no_oa_fulltext_found" in text.lower()


def test_methods_text_no_queue_note_when_resolved():
    cfg = ReviewConfig(title="t")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), review_queue=[]))
    assert "No records remained in the full-text eligibility review queue." in text
    assert "UNRESOLVED" not in text


def test_methods_text_flags_unresolved_queue():
    cfg = ReviewConfig(title="t")
    queue = [{"stage": "ft_screening", "record_id": "rec-pending"}]
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), review_queue=queue))
    assert "1 record(s) remain UNRESOLVED" in text


def test_methods_text_deterministic_search_date_fallback():
    """§2.15: `date.today()` is never used -- an empty search log renders
    a fixed, deterministic sentence, and re-rendering twice is
    byte-identical (R4)."""
    cfg = ReviewConfig(title="t")
    snapshot = _make_snapshot(cfg, _prisma(), SearchLog())
    text1 = _render_methods_md(snapshot)
    text2 = _render_methods_md(snapshot)
    assert text1 == text2
    assert "search date not recorded" in text1


def _artifact(*, total_fields: int = 3, resolutions: list = None, remaining_queue_keys: list = None,
              run_id: str = "extrun-abc", manifest_path: str = "/out/manifest.json",
              extraction_schema_hash: str = "deadbeef") -> dict:
    return {
        "run_id": run_id, "manifest_path": manifest_path, "extraction_schema_hash": extraction_schema_hash,
        "total_fields": total_fields, "resolutions": resolutions or [], "remaining_queue_keys": remaining_queue_keys or [],
        "resolved_projection": [],
    }


def test_methods_text_includes_extraction_section_for_scoping():
    """Scoping also gets a genuine extraction section, but only when
    `snapshot.extraction` is actually present (Codex round-2 review,
    finding 2: an absent extraction bundle must not render this section
    with fabricated N/A placeholders -- see
    test_methods_text_no_extraction_bundle_shows_honest_not_performed_note)."""
    cfg = ReviewConfig(title="t", review_type="scoping")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=_artifact()))
    assert "Structured data extraction" in text


def test_methods_text_reports_zero_human_review_when_nothing_resolved():
    cfg = ReviewConfig(title="t", review_type="rapid")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=_artifact()))
    assert "No field required human review" in text
    assert "extrun-abc" in text
    assert "deadbeef" in text


def test_methods_text_no_extraction_bundle_shows_honest_not_performed_note():
    """Codex round-2 review, finding 2: `snapshot.extraction is None` (a
    valid empty-corpus run -- extraction_node never ran) must render an
    honest "did not run" statement, never the old N/A-filled "Data were
    extracted..." boilerplate, which falsely implied extraction occurred."""
    cfg = ReviewConfig(title="t", review_type="rapid")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=None))
    assert "extraction_run_id=N/A" not in text
    assert "Data were extracted using the evidence-verified structured-extraction engine" not in text
    assert "no full-text records were available to extract from" in text


def test_methods_text_reports_human_review_counts_by_outcome():
    cfg = ReviewConfig(title="t", review_type="slr")
    resolutions = [
        {"record_id": "rec-a", "field_id": "f0", "outcome": "extracted"},
        {"record_id": "rec-a", "field_id": "f1", "outcome": "not_reported"},
    ]
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=_artifact(resolutions=resolutions)))
    assert "2 of 3 field-level result(s) required human review" in text
    assert "1 resolved as a human-attested extracted value" in text
    assert "1 resolved as not_reported" in text
    assert "the remaining 1 field(s) are the direct, unmodified machine extraction output" in text


def test_methods_text_flags_unresolved_extraction_queue():
    cfg = ReviewConfig(title="t", review_type="rapid")
    text = _render_methods_md(_make_snapshot(
        cfg, _prisma(), SearchLog(), extraction_artifact=_artifact(remaining_queue_keys=[["rec-a", "f0"]])))
    assert "1 field(s) across the included full-text corpus remain UNRESOLVED" in text


def test_methods_text_excludes_still_unresolved_fields_from_machine_final_count():
    cfg = ReviewConfig(title="t", review_type="rapid")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=_artifact(
        total_fields=3,
        resolutions=[{"record_id": "rec-a", "field_id": "f0", "outcome": "extracted"}],
        remaining_queue_keys=[["rec-a", "f1"]],
    )))
    assert "the remaining 1 field(s) are the direct, unmodified machine extraction output" in text


def test_methods_text_distinguishes_clean_run_from_heavily_reviewed_run():
    cfg = ReviewConfig(title="t", review_type="rapid")
    clean_text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=_artifact()))
    reviewed_text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), extraction_artifact=_artifact(
        resolutions=[{"record_id": "rec-a", "field_id": f"f{i}", "outcome": "extracted"} for i in range(3)],
    )))
    assert clean_text != reviewed_text
    assert "No field required human review" in clean_text
    assert "No field required human review" not in reviewed_text


# ---------------------------------------------------------------------------
# 1a. Codex round-1 review, finding 4: methods.md renders actual snapshot
# facts, not unconditional boilerplate claims.
# ---------------------------------------------------------------------------

def test_methods_text_no_human_validation_claim_when_no_resolutions():
    cfg = ReviewConfig(title="t")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), screening_review_resolutions=[]))
    assert "Borderline cases received human validation" not in text
    assert "No borderline cases required human validation" in text


def test_methods_text_claims_human_validation_when_resolutions_exist():
    cfg = ReviewConfig(title="t")
    resolutions = [{"record_id": "r1", "decision": "include", "reviewer": "alice"}]
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), screening_review_resolutions=resolutions))
    assert "Borderline cases received human validation" in text


def test_methods_text_no_kappa_claim_when_qc_did_not_run():
    cfg = ReviewConfig(title="t")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), screening_qc_state=None))
    assert "Inter-rater reliability was assessed using Cohen's kappa" not in text
    assert "was NOT assessed this run" in text


def test_methods_text_no_kappa_claim_when_qc_ran_with_no_valid_pairs():
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(n_attempted=3, n_valid_pairs=0, n_failed=3,
                                                          sampled_record_ids=["a", "b", "c"],
                                                          producing_event_ids=["e1", "e2", "e3"]))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), screening_qc_state=qc))
    assert "was NOT assessed this run" in text


def test_methods_text_claims_kappa_when_qc_completed():
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(n_attempted=2, n_valid_pairs=2, n_failed=0,
                                                          sampled_record_ids=["a", "b"], cohens_kappa=1.0,
                                                          producing_event_ids=["e1", "e2"]))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), screening_qc_state=qc))
    assert "Inter-rater reliability was assessed using Cohen's kappa" in text


def test_methods_text_bertopic_disabled_note_never_contradicted_by_unconditional_claim():
    """The old, unconditional 'Thematic synthesis was conducted using
    BERTopic topic modelling.' sentence contradicted a disabled/no-op
    render_bertopic_note() -- must no longer appear at all."""
    cfg = ReviewConfig(title="t", synthesis=SynthesisConfig(run_bertopic=False))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog()))
    assert "BERTopic thematic clustering was disabled" in text
    assert "Thematic synthesis was conducted using BERTopic topic modelling." not in text


def test_methods_text_no_evidence_map_claim_when_absent_despite_config_on():
    """Only `snapshot.evidence_map is not None` -- not merely
    `generate_evidence_map` being on in config -- may justify the claim."""
    cfg = ReviewConfig(title="t", synthesis=SynthesisConfig(generate_evidence_map=True))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), evidence_map=None))
    assert "An evidence map was generated." not in text


# ---------------------------------------------------------------------------
# 1c. Codex round-2 review, finding 2 (P1): methods.md's QA/kappa claims
# were proxy-based (review_type, n_valid_pairs > 0) rather than
# artifact-based (snapshot.quality_appraisal presence, actual kappa value).
# ---------------------------------------------------------------------------

def test_methods_text_qa_not_conducted_for_scoping_regardless_of_bundle():
    cfg = ReviewConfig(title="t", review_type="scoping")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), quality_appraisal_present=False))
    assert "Quality appraisal was not conducted, as is standard practice for scoping reviews." in text


def test_methods_text_qa_not_performed_for_rapid_empty_corpus():
    """The exact bug Codex reproduced: a valid empty-corpus rapid/SLR run
    (`snapshot.quality_appraisal is None`) must NOT claim QA "was
    conducted using the X checklist" -- that claim is false when no QA
    bundle exists, regardless of what the review is configured to do."""
    cfg = ReviewConfig(title="t", review_type="rapid",
                        quality_appraisal=QualityAppraisalConfig(checklist_id="rob2_inspired_study_level_teaching_v1"))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), quality_appraisal_present=False))
    assert "Quality appraisal was conducted using" not in text
    assert "Quality appraisal was not performed this run: no eligible full-text records" in text


def test_methods_text_qa_conducted_for_rapid_with_bundle():
    cfg = ReviewConfig(title="t", review_type="rapid",
                        quality_appraisal=QualityAppraisalConfig(checklist_id="rob2_inspired_study_level_teaching_v1"))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), quality_appraisal_present=True))
    assert "Quality appraisal was conducted using the rob2_inspired_study_level_teaching_v1 checklist." in text


def test_methods_text_kappa_not_estimable_distinct_from_assessed_or_not_run():
    """`n_valid_pairs > 0` alone is not proof kappa was assessed -- the
    producer can legitimately return `cohens_kappa=None` for a
    non-estimable sample (Codex's own reproduction: n_valid_pairs=1,
    cohens_kappa=None)."""
    cfg = ReviewConfig(title="t")
    qc = ReportingScreeningQCState(irr=ReportingIRRState(n_attempted=1, n_valid_pairs=1, n_failed=0,
                                                          sampled_record_ids=["a"], cohens_kappa=None,
                                                          producing_event_ids=["e1"]))
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), screening_qc_state=qc))
    assert "Inter-rater reliability was assessed using Cohen's kappa" not in text
    assert "was NOT assessed this run" not in text
    assert "kappa was not estimable for this sample" in text
    assert "1 valid" in text


def test_qa_export_not_applicable_reason_differs_scoping_vs_rapid_empty_corpus():
    """Codex round-2 review, finding 2 (P1): a single static
    NOT_APPLICABLE_REASONS string conflated "never applicable by
    review_type" (scoping) with "configured to run but no eligible
    records" (rapid/SLR empty-corpus) -- reusing the scoping wording for
    the latter would contradict that review's own configuration."""
    scoping_cfg = ReviewConfig(title="t", review_type="scoping")
    scoping_reason = reporting_module._qa_not_applicable_reason(
        _make_snapshot(scoping_cfg, _prisma(), SearchLog()))
    assert scoping_reason == "quality appraisal not performed for this review_type"

    rapid_cfg = ReviewConfig(title="t", review_type="rapid")
    rapid_reason = reporting_module._qa_not_applicable_reason(
        _make_snapshot(rapid_cfg, _prisma(), SearchLog(), quality_appraisal_present=False))
    assert rapid_reason != scoping_reason
    assert "no eligible full-text records" in rapid_reason


def test_methods_md_built_from_includes_quality_appraisal_hashes():
    """methods.md's own `built_from` (Step 9: a list of `slr://` locators,
    not a hash dict -- see `_renderer_locator_pool`/`_renderer_built_from`)
    still depends on `snapshot.quality_appraisal`'s presence/absence
    (Codex round-2 review, finding 2)."""
    cfg = ReviewConfig(title="t", review_type="rapid")
    snapshot = _make_snapshot(
        cfg, _prisma(), SearchLog(), extraction_artifact=_artifact(), quality_appraisal_present=True)
    pool = reporting_module._renderer_locator_pool(snapshot)
    built_from = reporting_module._renderer_built_from("methods_md", pool)
    assert f"slr://extraction/manifest/{snapshot.extraction.run_id}" in built_from
    assert f"slr://quality_appraisal/manifest/{snapshot.quality_appraisal.run_id}" in built_from


def test_methods_text_evidence_map_claim_when_present():
    cfg = ReviewConfig(title="t", synthesis=SynthesisConfig(generate_evidence_map=True))
    em = EvidenceMapModel(n_studies=1, by_year={}, by_study_design={}, by_country={},
                           characteristics_table=[{"citation": "c1"}])
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog(), evidence_map=em))
    assert "An evidence map was generated." in text


def test_methods_text_includes_data_governance_section():
    cfg = ReviewConfig(title="t", data_governance_declaration="No personal data collected.")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog()))
    assert "Data governance and ethics" in text
    assert "No personal data collected." in text


def test_methods_text_data_governance_not_provided_when_absent():
    cfg = ReviewConfig(title="t")
    text = _render_methods_md(_make_snapshot(cfg, _prisma(), SearchLog()))
    assert "Data governance and ethics" in text
    assert "Not provided." in text


# ---------------------------------------------------------------------------
# 1b. Codex round-1 review, finding 6: PRISMA rendering determinism
# relative to prisma_sha256 (insertion-order-independent).
# ---------------------------------------------------------------------------

def test_methods_text_database_list_sorted_regardless_of_insertion_order():
    cfg = ReviewConfig(title="t")
    prisma_a = _prisma(identified={"pubmed": 3, "arxiv": 2})
    prisma_b = _prisma(identified={"arxiv": 2, "pubmed": 3})
    text_a = _render_methods_md(_make_snapshot(cfg, prisma_a, SearchLog()))
    text_b = _render_methods_md(_make_snapshot(cfg, prisma_b, SearchLog()))
    assert text_a == text_b
    assert "arxiv, pubmed" in text_a


def test_prisma_diagram_source_list_sorted_regardless_of_insertion_order():
    prisma_a = _prisma(identified={"pubmed": 3, "arxiv": 2})
    prisma_b = _prisma(identified={"arxiv": 2, "pubmed": 3})
    fig_a = reporting_module._draw_prisma_diagram(prisma_a, {})
    fig_b = reporting_module._draw_prisma_diagram(prisma_b, {})
    texts_a = sorted(t.get_text() for t in fig_a.axes[0].texts)
    texts_b = sorted(t.get_text() for t in fig_b.axes[0].texts)
    assert texts_a == texts_b
    import matplotlib.pyplot as plt
    plt.close(fig_a)
    plt.close(fig_b)


# ---------------------------------------------------------------------------
# 2a. EvidenceMapModel / reconstruction (T1, U1, V2, W1, X1-X3, Y1-Y2)
# ---------------------------------------------------------------------------

def _two_record_corpus():
    records = [
        Record(id="r1", source="s", source_id="r1", title="Study One", year=2020),
        Record(id="r2", source="s", source_id="r2", title="Study Two", year=2021),
    ]
    charted = [
        {"record_id": "r1", "raw_extraction": {"country": "US", "study_design": "RCT", "population": "adults"}},
        {"record_id": "r2", "raw_extraction": {"country": "UK", "study_design": "cohort", "population": ["adults", "elderly"]}},
    ]
    extraction_raw = {"resolved_projection": charted}
    return records, charted, extraction_raw


def test_evidence_map_reconstruction_matches_real_build_X1_X3():
    """X1: a genuine integer-valued `by_year` key round-trips correctly
    via `_pydantic_json_mode()` -- exactly the conversion `SynthesisOutput.
    model_dump(mode="json")` already applies before `evidence_map` ever
    reaches `state["synthesis"]`, so `raw_evidence_map` in the real
    pipeline always has STRING `by_year` keys already; `EvidenceMapModel`
    is therefore validated against the JSON-mode form here too, not
    `_build_evidence_map()`'s own raw (int-keyed) Python return value.
    X3(c): a real multi-valued `population` list is accepted and
    reproduced exactly."""
    records, charted, extraction_raw = _two_record_corpus()
    real_map = _build_evidence_map(records, {c["record_id"]: c["raw_extraction"] for c in charted})
    real_map_json_mode = _pydantic_json_mode(real_map)
    assert set(real_map_json_mode["by_year"].keys()) == {"2020", "2021"}  # int keys -> str keys
    validated = EvidenceMapModel.model_validate(real_map_json_mode)
    assert validated.n_studies == 2
    assert validated.characteristics_table[1].population == ["adults", "elderly"]

    reconstructed = _reconstruct_expected_evidence_map(full_text_records=records, extraction_raw=extraction_raw)
    assert _pydantic_json_mode(reconstructed) == real_map_json_mode


def test_evidence_map_reconstruction_rejects_fabricated_map_W1():
    records, charted, extraction_raw = _two_record_corpus()
    real_map = _build_evidence_map(records, {c["record_id"]: c["raw_extraction"] for c in charted})
    fabricated = copy.deepcopy(_pydantic_json_mode(real_map))
    fabricated["characteristics_table"][0]["citation"] = "Someone Else 1999"
    reconstructed = _reconstruct_expected_evidence_map(full_text_records=records, extraction_raw=extraction_raw)
    assert _pydantic_json_mode(reconstructed) != fabricated


def test_evidence_map_reconstruction_tolerates_reversed_record_order_X2():
    records, charted, extraction_raw = _two_record_corpus()
    reversed_records = list(reversed(records))
    reconstructed = canonicalize_corpus_order(full_text_records=reversed_records, charted=charted, error_prefix="test")
    assert [r.id for r in reconstructed] == [c["record_id"] for c in charted]


def test_canonicalize_corpus_order_reporting_prefix_X2c():
    records, charted, _ = _two_record_corpus()
    bad_charted = charted + [{"record_id": "foreign", "raw_extraction": {}}]
    with pytest.raises(ValueError, match="reporting evidence-map reconstruction"):
        canonicalize_corpus_order(full_text_records=records, charted=bad_charted, error_prefix="reporting evidence-map reconstruction")


def test_evidence_map_row_rejects_typed_country_value_X3b():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({
            "n_studies": 1, "by_year": {}, "by_study_design": {}, "by_country": {},
            "characteristics_table": [{"citation": "c", "country": 2020}],
        })


def test_evidence_map_row_rejects_string_year_X3a():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({
            "n_studies": 1, "by_year": {}, "by_study_design": {}, "by_country": {},
            "characteristics_table": [{"citation": "c", "year": "2020"}],
        })


def test_evidence_map_model_rejects_bool_count_U1a():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({
            "n_studies": True, "by_year": {}, "by_study_design": {}, "by_country": {}, "characteristics_table": [],
        })


def test_evidence_map_model_rejects_numeric_string_count_U1a():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({
            "n_studies": 2, "by_year": {"2020": "2"}, "by_study_design": {}, "by_country": {},
            "characteristics_table": [{"citation": "a"}, {"citation": "b"}],
        })


def test_evidence_map_row_rejects_extra_field_U1c():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({
            "n_studies": 1, "by_year": {}, "by_study_design": {}, "by_country": {},
            "characteristics_table": [{"citation": "c", "unexpected_field": 1}],
        })


def test_evidence_map_model_row_count_mismatch_raises_T1b():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({
            "n_studies": 3, "by_year": {}, "by_study_design": {}, "by_country": {},
            "characteristics_table": [{"citation": "a"}, {"citation": "b"}],
        })


def test_evidence_map_empty_dict_raises_T1a():
    with pytest.raises(Exception):
        EvidenceMapModel.model_validate({})


# ---------------------------------------------------------------------------
# 2b. BERTopic consistency (T2, U3) and rendered note branches (S5)
# ---------------------------------------------------------------------------

def test_bertopic_consistency_disabled_with_topics_raises_T2a():
    with pytest.raises(ReportingConsistencyError):
        _validate_bertopic_consistency(run_bertopic=False, n_records=2, excluded=0, topics=[{"topic_id": 0}])


def test_bertopic_consistency_zero_records_with_topics_raises_T2b():
    with pytest.raises(ReportingConsistencyError):
        _validate_bertopic_consistency(run_bertopic=True, n_records=0, excluded=0, topics=[{"topic_id": 0}])


def test_bertopic_consistency_all_excluded_with_topics_raises_T2c():
    with pytest.raises(ReportingConsistencyError):
        _validate_bertopic_consistency(run_bertopic=True, n_records=2, excluded=2, topics=[{"topic_id": 0}])


def test_bertopic_consistency_disabled_nonzero_excluded_raises_U3():
    with pytest.raises(ReportingConsistencyError):
        _validate_bertopic_consistency(run_bertopic=False, n_records=2, excluded=2, topics=[])


def test_bertopic_consistency_valid_partial_exclusion_passes():
    _validate_bertopic_consistency(run_bertopic=True, n_records=5, excluded=2, topics=[{"topic_id": 0}])


def _bertopic_snapshot(*, run_bertopic: bool, n_records: int, excluded: int, topics: list) -> ReportingSourceSnapshot:
    cfg = ReviewConfig(title="t", synthesis=SynthesisConfig(run_bertopic=run_bertopic))
    records = [Record(id=f"r{i}", source="s", source_id=f"r{i}") for i in range(n_records)]
    snapshot = _make_snapshot(cfg, _prisma(), SearchLog())
    return dataclasses.replace(
        snapshot, full_text_records=records,
        synthesis_output=VerifiedProjection(
            model=SynthesisOutput(refine_batch_size=1, bertopic_excluded_empty_evidence_count=excluded, topics=topics),
            content_hash="x" * 64),
    )


def test_render_bertopic_note_disabled_S5a():
    snapshot = _bertopic_snapshot(run_bertopic=False, n_records=2, excluded=0, topics=[])
    assert "disabled" in render_bertopic_note(snapshot)


def test_render_bertopic_note_zero_records_S5b():
    snapshot = _bertopic_snapshot(run_bertopic=True, n_records=0, excluded=0, topics=[])
    assert "no full-text records were available" in render_bertopic_note(snapshot)


def test_render_bertopic_note_all_excluded_S5c():
    snapshot = _bertopic_snapshot(run_bertopic=True, n_records=2, excluded=2, topics=[])
    assert "every one was excluded" in render_bertopic_note(snapshot)


def test_render_bertopic_note_zero_topics_S5d():
    snapshot = _bertopic_snapshot(run_bertopic=True, n_records=2, excluded=0, topics=[])
    assert "cannot be distinguished from a silent execution failure" in render_bertopic_note(snapshot)


def test_render_bertopic_note_success_S5e():
    snapshot = _bertopic_snapshot(run_bertopic=True, n_records=2, excluded=0, topics=[{"topic_id": 0}, {"topic_id": 1}])
    note = render_bertopic_note(snapshot)
    assert "produced 2 thematic topics" in note


# ---------------------------------------------------------------------------
# 2c. `_load_component()` -- containment/single-read (T3, rescoped)
# ---------------------------------------------------------------------------

def test_load_component_rejects_path_outside_run_root_T3b(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ReportingConsistencyError, match="escapes its own run_dir"):
        _load_component(outside, run_root=run_root, seen=set(), label="test component", required=True)


def test_load_component_reads_bytes_exactly_once_T3a(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    target = run_root / "file.json"
    target.write_text('{"a": 1}')

    real_read_bytes = Path.read_bytes
    call_count = {"n": 0}

    def counting_read_bytes(self):
        if self == target:
            call_count["n"] += 1
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    component, raw_bytes = _load_component(target, run_root=run_root, seen=set(), label="test component", required=True)
    assert call_count["n"] == 1
    assert raw_bytes == b'{"a": 1}'
    assert component.file_sha256 == sha256_hex_of(raw_bytes)


def sha256_hex_of(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_load_component_missing_optional_returns_none(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    missing = run_root / "missing.json"
    component, raw_bytes = _load_component(missing, run_root=run_root, seen=set(), label="optional", required=False)
    assert component is None and raw_bytes is None


def test_load_component_missing_required_raises(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    missing = run_root / "missing.json"
    with pytest.raises(ReportingConsistencyError, match="expected at"):
        _load_component(missing, run_root=run_root, seen=set(), label="required", required=True)


def test_load_component_rejects_duplicate_alias(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    target = run_root / "file.json"
    target.write_text("{}")
    seen: set = set()
    _load_component(target, run_root=run_root, seen=seen, label="first", required=True)
    with pytest.raises(ReportingConsistencyError, match="duplicates/aliases"):
        _load_component(target, run_root=run_root, seen=seen, label="second", required=True)


# ---------------------------------------------------------------------------
# 2d. FT native provenance path safety
# ---------------------------------------------------------------------------

def _ft_cfg(tmp_path) -> ReviewConfig:
    return ReviewConfig(title="t", output_dir=str(tmp_path / "out"))


def test_ft_native_provenance_all_none_returns_none(tmp_path):
    cfg = _ft_cfg(tmp_path)
    state = {"ft_screening_run_id": None, "ft_screening_execution_fingerprint": None, "ft_screening_artifact_paths": None}
    assert capture_ft_native_provenance(state, cfg) is None


def test_ft_native_provenance_partial_set_raises(tmp_path):
    cfg = _ft_cfg(tmp_path)
    state = {"ft_screening_run_id": "run-1", "ft_screening_execution_fingerprint": None, "ft_screening_artifact_paths": None}
    with pytest.raises(ReportingConsistencyError, match="all-set or all-None"):
        capture_ft_native_provenance(state, cfg)


FT_RUN_ID = "run-1"
FT_EXECUTION_FINGERPRINT = _valid_sha256("ft-fp-1")


def _ft_files(cfg: ReviewConfig) -> Dict[str, str]:
    """Codex round-5 review, finding 1: `capture_ft_native_provenance()`
    now fully parses/cross-verifies `run_result.json`/`item_results.jsonl`
    via FT_SCREENING's own real pydantic models (`ScreeningRunResult`/
    `ScreeningItemResult`), so this fixture must build a genuinely valid,
    schema-complete pair via `_build_valid_ft_screening_run_files()`
    rather than arbitrary bytes/an untyped dict. Callers whose own check
    raises BEFORE the parsing step (wrong-key-set/duplicate-path/missing-
    file/outside-output-dir) don't need their state dict's own
    `ft_screening_run_id`/`_execution_fingerprint` to match what's baked
    into these files; the two tests that DO reach the full parse+cross-
    verify step (`..._valid_returns_classification`/
    `..._detached_from_live_state_dict`) use `FT_RUN_ID`/
    `FT_EXECUTION_FINGERPRINT` below, matching exactly."""
    out = cfg.ensure_output_dir()
    return _build_valid_ft_screening_run_files(
        out, run_id=FT_RUN_ID, execution_fingerprint=FT_EXECUTION_FINGERPRINT, record_ids=["ft-item-1"])


def test_ft_native_provenance_wrong_key_set_raises(tmp_path):
    cfg = _ft_cfg(tmp_path)
    paths = _ft_files(cfg)
    del paths["ledger_jsonl"]
    paths["extra_key"] = str(cfg.ensure_output_dir() / "run_result_json.txt")
    state = {"ft_screening_run_id": "run-1", "ft_screening_execution_fingerprint": "fp", "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="artifact_paths keys"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_duplicate_paths_raise(tmp_path):
    cfg = _ft_cfg(tmp_path)
    out = cfg.ensure_output_dir()
    shared = out / "shared.txt"
    shared.write_text("x")
    paths = {"run_result_json": str(shared), "item_results_jsonl": str(shared), "ledger_jsonl": str(shared)}
    state = {"ft_screening_run_id": "run-1", "ft_screening_execution_fingerprint": "fp", "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="duplicate/aliased"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_missing_file_raises(tmp_path):
    cfg = _ft_cfg(tmp_path)
    out = cfg.ensure_output_dir()
    paths = {"run_result_json": str(out / "missing1.txt"), "item_results_jsonl": str(out / "missing2.txt"),
              "ledger_jsonl": str(out / "missing3.txt")}
    state = {"ft_screening_run_id": "run-1", "ft_screening_execution_fingerprint": "fp", "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="not an existing regular file"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_path_outside_output_dir_raises(tmp_path):
    cfg = _ft_cfg(tmp_path)
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    outside_file = outside_dir / "x.txt"
    outside_file.write_text("x")
    paths = _ft_files(cfg)
    paths["ledger_jsonl"] = str(outside_file)
    state = {"ft_screening_run_id": "run-1", "ft_screening_execution_fingerprint": "fp", "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="is outside"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_valid_returns_classification(tmp_path):
    cfg = _ft_cfg(tmp_path)
    paths = _ft_files(cfg)
    state = {"ft_screening_run_id": FT_RUN_ID, "ft_screening_execution_fingerprint": FT_EXECUTION_FINGERPRINT,
              "ft_screening_artifact_paths": paths}
    result = capture_ft_native_provenance(state, cfg)
    assert result["classification"] == "state_referenced_byte_hashed_not_semantically_reverified"
    assert set(result["artifact_file_hashes"]) == {"run_result_json", "item_results_jsonl", "ledger_jsonl"}
    assert result["confirmed_record_ids"] == ["ft-item-1"]


def test_ft_native_provenance_detached_from_live_state_dict(tmp_path):
    """Codex round-1 review, finding 7: the captured `ft_screening_
    artifact_paths` must be a detached copy -- mutating the ORIGINAL
    `paths` dict/state after capture must never change the already-
    captured result (whose file hashes were computed against it)."""
    cfg = _ft_cfg(tmp_path)
    paths = _ft_files(cfg)
    original_run_result_path = paths["run_result_json"]
    state = {"ft_screening_run_id": FT_RUN_ID, "ft_screening_execution_fingerprint": FT_EXECUTION_FINGERPRINT,
              "ft_screening_artifact_paths": paths}
    result = capture_ft_native_provenance(state, cfg)
    assert result["ft_screening_artifact_paths"] is not paths
    paths["run_result_json"] = "/tampered/path.txt"
    assert result["ft_screening_artifact_paths"]["run_result_json"] != "/tampered/path.txt"
    assert result["ft_screening_artifact_paths"]["run_result_json"] == str(Path(original_run_result_path).resolve())


def test_ft_native_provenance_run_result_run_id_mismatch_raises(tmp_path):
    """Codex round-5 review, finding 1: run_result.json's OWN run_id must
    match state['ft_screening_run_id'] -- an independently-written
    artifact claiming a DIFFERENT run must never be trusted."""
    cfg = _ft_cfg(tmp_path)
    paths = _ft_files(cfg)
    state = {"ft_screening_run_id": "a-different-run-id", "ft_screening_execution_fingerprint": FT_EXECUTION_FINGERPRINT,
              "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="run_id"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_execution_fingerprint_mismatch_raises(tmp_path):
    cfg = _ft_cfg(tmp_path)
    paths = _ft_files(cfg)
    state = {"ft_screening_run_id": FT_RUN_ID, "ft_screening_execution_fingerprint": _valid_sha256("a-different-fingerprint"),
              "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="execution_fingerprint"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_malformed_item_results_line_raises(tmp_path):
    """Codex round-5 review, finding 1: a line that is not a valid
    ScreeningItemResult (e.g. an untyped `{"document_id": ...}` dict, the
    exact shape the OLD, weaker fixture used) must be REJECTED at real-
    model-parsing time, never silently accepted as authoritative."""
    cfg = _ft_cfg(tmp_path)
    out = cfg.ensure_output_dir()
    paths = _ft_files(cfg)
    Path(paths["item_results_jsonl"]).write_text(json.dumps({"document_id": "ft-item-1"}) + "\n", encoding="utf-8")
    state = {"ft_screening_run_id": FT_RUN_ID, "ft_screening_execution_fingerprint": FT_EXECUTION_FINGERPRINT,
              "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="not a valid ScreeningItemResult"):
        capture_ft_native_provenance(state, cfg)


def test_ft_native_provenance_item_set_mismatch_between_files_raises(tmp_path):
    """Codex round-5 review, finding 1: item_results.jsonl's own item set
    must match run_result.json's own `items` -- two independently-written
    artifacts of the same run disagreeing on which items exist must never
    both be trusted."""
    cfg = _ft_cfg(tmp_path)
    out = cfg.ensure_output_dir()
    paths = _build_valid_ft_screening_run_files(
        out, run_id=FT_RUN_ID, execution_fingerprint=FT_EXECUTION_FINGERPRINT, record_ids=["ft-item-1", "ft-item-2"])
    # Rewrite item_results.jsonl to drop ft-item-2 -- now disagrees with
    # run_result.json's own two-item `items` list.
    only_item = ScreeningItemResult(
        input_index=0, document_id="ft-item-1", report_id="ft-item-1", citation={"record_id": "ft-item-1"},
        execution_status="not_processed",
    )
    Path(paths["item_results_jsonl"]).write_text(only_item.model_dump_json() + "\n", encoding="utf-8")
    state = {"ft_screening_run_id": FT_RUN_ID, "ft_screening_execution_fingerprint": FT_EXECUTION_FINGERPRINT,
              "ft_screening_artifact_paths": paths}
    with pytest.raises(ReportingConsistencyError, match="does not match run_result"):
        capture_ft_native_provenance(state, cfg)


# ---------------------------------------------------------------------------
# 2e. Report-input validation (ReportingIRRState / ReportingScreeningQCState)
# ---------------------------------------------------------------------------

def test_reporting_irr_state_zero_attempt_constructs():
    state = ReportingIRRState(n_attempted=0, n_valid_pairs=0, n_failed=0)
    assert state.sampled_record_ids == []


def test_reporting_irr_state_rejects_bool_count():
    with pytest.raises(Exception):
        ReportingIRRState(n_attempted=True, n_valid_pairs=0, n_failed=0)


def test_reporting_irr_state_rejects_count_mismatch():
    with pytest.raises(Exception):
        ReportingIRRState(n_attempted=5, n_valid_pairs=2, n_failed=2, sampled_record_ids=["a", "b", "c", "d", "e"])


def test_reporting_irr_state_rejects_duplicate_sampled_ids():
    with pytest.raises(Exception):
        ReportingIRRState(n_attempted=2, n_valid_pairs=2, n_failed=0, sampled_record_ids=["a", "a"])


def test_reporting_irr_state_rejects_sample_count_mismatch():
    with pytest.raises(Exception):
        ReportingIRRState(n_attempted=2, n_valid_pairs=2, n_failed=0, sampled_record_ids=["a"])


def test_reporting_screening_qc_state_rejects_missing_irr():
    with pytest.raises(Exception):
        ReportingScreeningQCState.model_validate({"saturation": None})


# ---------------------------------------------------------------------------
# 2f. PRISMA render safety -- fig closed even when savefig raises
# ---------------------------------------------------------------------------

def test_prisma_render_closes_figure_on_savefig_failure(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    closed = []
    real_close = plt.close

    def spy_close(fig=None):
        closed.append(fig)
        return real_close(fig)

    monkeypatch.setattr(plt, "close", spy_close)

    def raising_savefig(self, *a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", raising_savefig)

    prisma = _prisma()
    with pytest.raises(RuntimeError, match="boom"):
        reporting_module._render_prisma_diagram_png(_make_snapshot(ReviewConfig(title="t"), prisma, SearchLog()), {})
    assert len(closed) == 1


# ---------------------------------------------------------------------------
# 2g. `_write_artifact_and_verify` / manifest self-verification / commit-
# marker policy
# ---------------------------------------------------------------------------

def test_write_artifact_and_verify_roundtrip(tmp_path):
    path = tmp_path / "out.json"
    file_hash = _write_artifact_and_verify(path, b'{"a": 1}')
    assert path.read_bytes() == b'{"a": 1}'
    assert file_hash == sha256_hex_of(b'{"a": 1}')


def _minimal_snapshot_for_publish(cfg: ReviewConfig) -> ReportingSourceSnapshot:
    return _make_snapshot(cfg, _prisma(assessed_full_text=0, excluded_full_text=0, excluded_full_text_reasons={},
                                        fulltext_retrieval={}, included=0), SearchLog())


def test_publish_report_bundle_manifest_self_verifies(tmp_path):
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    manifest = reporting_module.publish_report_bundle(report_dir=tmp_path, snapshot=snapshot, splits={})
    assert (tmp_path / "export_manifest.json").exists()
    assert reporting_module._renderer_entry(manifest, "quality_appraisal_csv")["status"] == "not_applicable"
    assert reporting_module._renderer_entry(manifest, "evidence_map_csv")["status"] == "not_applicable"
    assert reporting_module._renderer_entry(manifest, "synthesis_json")["status"] == "present"


def test_publish_report_bundle_prisma_failure_is_commit_marker_safe(tmp_path, monkeypatch):
    """PRISMA diagram is the ONE renderer with allow_failure=True -- a
    failure there still produces a full manifest, with a 'failed' entry
    for that one artifact."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)

    def raising_draw(*a, **kw):
        raise RuntimeError("prisma boom")

    monkeypatch.setattr(reporting_module, "_draw_prisma_diagram", raising_draw)
    manifest = reporting_module.publish_report_bundle(report_dir=tmp_path, snapshot=snapshot, splits={})
    assert reporting_module._renderer_entry(manifest, "prisma_diagram")["status"] == "failed"
    assert reporting_module._renderer_entry(manifest, "synthesis_json")["status"] == "present"
    assert (tmp_path / "export_manifest.json").exists()


def test_publish_report_bundle_other_failure_leaves_no_manifest(tmp_path, monkeypatch):
    """Any OTHER renderer failing must propagate and leave NO manifest --
    the manifest is written only after every non-allow_failure renderer
    has succeeded."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)

    def raising_render(snapshot):
        raise RuntimeError("synthesis claims boom")

    monkeypatch.setattr(reporting_module, "_render_synthesis_claims_dict", raising_render)
    with pytest.raises(RuntimeError, match="synthesis claims boom"):
        reporting_module.publish_report_bundle(report_dir=tmp_path, snapshot=snapshot, splits={})
    assert not (tmp_path / "export_manifest.json").exists()


def test_manifest_self_hash_check_catches_disk_tampering(tmp_path, monkeypatch):
    """Simulates a corrupted round-trip (the on-disk bytes, once re-parsed,
    no longer hash to their own claimed manifest_hash) -- the read-back
    self-check must catch it rather than trust the write."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)

    real_loads = json.loads

    def tampering_loads(data, *a, **kw):
        result = real_loads(data, *a, **kw)
        if isinstance(result, dict) and "manifest_hash" in result:
            result = {**result, "review_type": "TAMPERED"}
        return result

    monkeypatch.setattr(reporting_module.json, "loads", tampering_loads)
    with pytest.raises(ReportingConsistencyError, match="failed its own self-hash check"):
        reporting_module.publish_report_bundle(report_dir=tmp_path, snapshot=snapshot, splits={})


# ---------------------------------------------------------------------------
# 2h. `compute_selection_splits` (INV-R6)
# ---------------------------------------------------------------------------

def _ledger_verified_file(*, record_id: str, stage: str, decision: str, decision_maker: str,
                           source_event_id: str, supersedes_decision_id=None,
                           producer_artifact_id=None) -> VerifiedFile:
    from slr_pipeline.selection_ledger import build_selection_decision_record
    import uuid as uuid_module
    record = build_selection_decision_record(
        review_run_id="run-1", decision_id=str(uuid_module.uuid4()), source_event_id=source_event_id,
        record_id=record_id, stage=stage, decision=decision, decision_maker=decision_maker,
        supersedes_decision_id=supersedes_decision_id, producer_artifact_id=producer_artifact_id,
    )
    return VerifiedFile(model=record, path=Path("x"), file_sha256="a" * 64, internal_hash=record.content_hash)


def test_compute_selection_splits_matches_prisma_counts():
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    ledger = [
        _ledger_verified_file(record_id="r1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation", source_event_id="e1"),
        _ledger_verified_file(record_id="r2", stage="dedup", decision="unique", decision_maker="rule_based_automation", source_event_id="e2"),
    ]
    snapshot = dataclasses.replace(
        snapshot, ledger_records=ledger, raw_records=_state_value([{"id": "r1"}, {"id": "r2"}]))
    prisma = PRISMACounts(identified={"db": 2}, after_dedup=1)
    splits = compute_selection_splits(snapshot=snapshot, prisma=prisma)
    assert splits["dedup_removed"] == {"rule_based_automation": 1}


def test_compute_selection_splits_mismatch_raises_INV_R6():
    """Isolates the EXCLUSION-SUM check specifically: the record-ID-set
    reconciliation (Codex round-3 review, finding 2(a), strengthened by
    round-4 review, finding 4) passes -- only the exclusion-predicate sum
    disagrees with PRISMACounts.after_dedup."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    ledger = [
        _ledger_verified_file(record_id="r1", stage="dedup", decision="unique", decision_maker="rule_based_automation", source_event_id="e1"),
        _ledger_verified_file(record_id="r2", stage="dedup", decision="unique", decision_maker="rule_based_automation", source_event_id="e2"),
    ]
    snapshot = dataclasses.replace(
        snapshot, ledger_records=ledger, raw_records=_state_value([{"id": "r1"}, {"id": "r2"}]))
    prisma = PRISMACounts(identified={"db": 2}, after_dedup=1)  # claims 1 removed, ledger shows 0 duplicates
    with pytest.raises(ReportingConsistencyError, match="ledger split totals"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_record_id_set_mismatch_raises_dedup():
    """Codex round-3 review, finding 2(a) (P1): the exclusion-sum check
    alone says nothing about whether the ledger has a current tail for
    EVERY record it should -- an incomplete ledger (fewer tails than the
    real raw corpus) must raise even when nothing later would trip the
    exclusion-sum check first. Codex round-4 review, finding 4 (P1):
    strengthened from a raw COUNT comparison to exact record-ID SET
    equality against `snapshot.raw_records` (the authoritative source) --
    5 raw records exist, but only 1 has a ledger tail."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    ledger = [
        _ledger_verified_file(record_id="r1", stage="dedup", decision="duplicate", decision_maker="rule_based_automation", source_event_id="e1"),
    ]
    snapshot = dataclasses.replace(
        snapshot, ledger_records=ledger,
        raw_records=_state_value([{"id": f"r{i}"} for i in range(1, 6)]))  # r1..r5, only r1 has a ledger tail
    prisma = PRISMACounts(identified={"db": 5}, after_dedup=4)
    with pytest.raises(ReportingConsistencyError,
                        match=r"dedup: ledger current-tail record-ID set does not match.*missing="):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_empty_ledger_one_record_zero_exclusions_raises():
    """Codex's own exact reproduction: "an empty ledger passes a
    one-record, zero-exclusion fixture" -- the old exclusion-sum-only
    check saw 0 == 0 and passed; the record-ID-set reconciliation
    correctly rejects an entirely empty ledger against a nonzero expected
    corpus."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[], raw_records=_state_value([{"id": "r1"}]))
    prisma = PRISMACounts(identified={"db": 1}, after_dedup=1)  # 1 record, 0 expected exclusions, EMPTY ledger
    with pytest.raises(ReportingConsistencyError, match=r"dedup: ledger current-tail record-ID set does not match"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_resolution_linkage_allows_later_ai_supersession():
    """Codex round-4 review, finding 1 (P1) -- exact reproduction of the
    plan's own APPROVED `A0 -> H -> A1` regression
    (aggregation_export_reporting_plan_2026-08-25.md's screening-regression
    table: "origin reflects A1, oversight stays 'human_resolved'"): a
    LATER legitimate AI/rule_based rerun may validly supersede a human
    decision. Round-3's own fix wrongly required the CURRENT tail to be
    'human', which would reject exactly this scenario -- must NOT raise;
    the resolution's own ledger entry (H) still exists, correctly linked
    via producer_artifact_id, in r1's full history, even though it is no
    longer the tail (A1 is)."""
    cfg = ReviewConfig(title="t")
    prisma = PRISMACounts(identified={}, after_dedup=0, screened_title_abstract=1, excluded_title_abstract=0,
                           assessed_full_text=0, excluded_full_text=0)
    snapshot = _make_snapshot(cfg, prisma, SearchLog())
    a0 = _ledger_verified_file(record_id="r1", stage="title_abstract_screening", decision="borderline",
                                decision_maker="ai", source_event_id="e-a0")
    # Codex round-5 review, finding 4: resolution_id/reviewed_baseline_
    # decision_id must be real, recomputable values -- reporting.py now
    # recomputes compute_resolution_id() from the resolution row's own
    # fields and requires an exact match.
    resolution_id = compute_resolution_id(
        stage="title_abstract_screening", record_id="r1", action="exclude", reviewer="x", note="n",
        resolved_at="t", reviewed_baseline_decision_id=a0.model.decision_id,
    )
    h = _ledger_verified_file(
        record_id="r1", stage="title_abstract_screening", decision="exclude", decision_maker="human",
        source_event_id=derive_source_event_id(
            producer_artifact_id=resolution_id, record_id="r1", stage="title_abstract_screening"),
        supersedes_decision_id=a0.model.decision_id, producer_artifact_id=resolution_id)
    a1 = _ledger_verified_file(record_id="r1", stage="title_abstract_screening", decision="include",
                                decision_maker="ai", source_event_id="e-a1",
                                supersedes_decision_id=h.model.decision_id)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[a0, h, a1], screening_records=[Record(id="r1", source="s", source_id="r1")],
        screening_review_resolutions=_state_value([{"record_id": "r1", "decision": "exclude", "reviewer": "x",
                                                      "note": "n", "resolved_at": "t",
                                                      "resolution_id": resolution_id,
                                                      "reviewed_baseline_decision_id": a0.model.decision_id}]),
    )
    compute_selection_splits(snapshot=snapshot, prisma=prisma)  # must NOT raise -- A1 is current, H is historical


def test_compute_selection_splits_screening_resolution_requires_human_tail():
    """Codex round-3 review, finding 2(b) (P1), rescoped by round-4 review,
    finding 1: a record named in `screening_review_resolutions` (durable
    proof a human resolved it) must have a MATCHING ledger entry
    (record_id, stage, producer_artifact_id==this resolution's own
    resolution_id, decision_maker=='human') SOMEWHERE in its own history --
    not necessarily as the current tail (see the positive companion test
    above). Here, the human decision's own ledger entry was never
    published at all (only the AI predecessor survives on disk) -- no
    matching entry exists anywhere, so this must still raise."""
    cfg = ReviewConfig(title="t")
    prisma = PRISMACounts(identified={}, after_dedup=0, screened_title_abstract=1, excluded_title_abstract=1,
                           assessed_full_text=0, excluded_full_text=0)
    snapshot = _make_snapshot(cfg, prisma, SearchLog())
    # Only the AI predecessor survives on disk -- the human supersession
    # that screening_review_resolutions claims happened is missing.
    ai_entry = _ledger_verified_file(record_id="r1", stage="title_abstract_screening", decision="exclude",
                                      decision_maker="ai", source_event_id="e1")
    ledger = [ai_entry]
    resolution_id = compute_resolution_id(
        stage="title_abstract_screening", record_id="r1", action="exclude", reviewer="x", note="n",
        resolved_at="t", reviewed_baseline_decision_id=ai_entry.model.decision_id,
    )
    snapshot = dataclasses.replace(
        snapshot, ledger_records=ledger, screening_records=[Record(id="r1", source="s", source_id="r1")],
        screening_review_resolutions=_state_value([{"record_id": "r1", "decision": "exclude", "reviewer": "x",
                                                      "note": "n", "resolved_at": "t", "resolution_id": resolution_id,
                                                      "reviewed_baseline_decision_id": ai_entry.model.decision_id}]),
    )
    with pytest.raises(ReportingConsistencyError, match=r"title_abstract_screening: record 'r1' was human-resolved"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_fulltext_resolution_requires_human_tail():
    """Same as above, for full_text_eligibility -- an ft_screening-stage
    resolution names a record whose current ledger tail is still the AI
    predecessor (needs_human_review), not the human supersession the
    resolution log claims happened."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    ai_entry = _ledger_verified_file(record_id="r1", stage="full_text_eligibility", decision="needs_human_review",
                                      decision_maker="ai", source_event_id="e1")
    ledger = [ai_entry]
    resolution_id = compute_resolution_id(
        stage="full_text_eligibility", record_id="r1", action="exclude", reviewer="x", note="n",
        resolved_at="t", reviewed_baseline_decision_id=ai_entry.model.decision_id, exclusion_reason=None,
    )
    snapshot = dataclasses.replace(
        snapshot, ledger_records=ledger,
        full_text_review_resolutions=_state_value([{"record_id": "r1", "stage": "ft_screening", "action": "exclude",
                                                      "reviewer": "x", "note": "n", "resolved_at": "t",
                                                      "resolution_id": resolution_id,
                                                      "reviewed_baseline_decision_id": ai_entry.model.decision_id}]),
        # Codex round-4 review, finding 4: the completeness SET check now
        # needs an authoritative confirmed-record-ID source too -- set it
        # to match this fixture's own single record so ONLY the
        # resolution-linkage check (this test's actual target) fires.
        ft_native_provenance={"confirmed_record_ids": ["r1"]},
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=1, excluded_full_text=1)
    with pytest.raises(ReportingConsistencyError, match=r"full_text_eligibility: record 'r1' was human-resolved"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_acquisition_stage_resolution_never_checked():
    """An acquisition-stage full-text resolution ('not_retrieved') never
    touches the ledger at all -- the linkage check must skip it, not
    demand a human ledger tail that legitimately never exists."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[],
        full_text_review_resolutions=_state_value([{"record_id": "r1", "stage": "acquisition",
                                                      "action": "not_retrieved", "reviewer": "x", "note": "n",
                                                      "resolved_at": "t"}]),
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    compute_selection_splits(snapshot=snapshot, prisma=prisma)  # must not raise


def test_compute_selection_splits_acquisition_stage_resolution_wrong_action_raises():
    """Codex round-6 review, finding 3(b) (P1): an acquisition-stage
    resolution's OWN shape must now be validated, not merely skipped
    silently -- 'include'/'exclude' are illegal for a record whose report
    was never confirmed (fulltext.py's own resolve_full_text_review_queue()
    only ever accepts 'not_retrieved' for this stage)."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[],
        full_text_review_resolutions=_state_value([{"record_id": "r1", "stage": "acquisition",
                                                      "action": "include", "reviewer": "x", "note": "n",
                                                      "resolved_at": "t"}]),
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    with pytest.raises(ReportingConsistencyError, match=r"acquisition-stage entry for record 'r1' has action='include'"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_acquisition_stage_resolution_extra_key_raises():
    """Codex round-7 review, finding 1 (P1) -- exact reproduction: a
    genuine ft_screening resolution (which DOES carry resolution_id/
    reviewed_baseline_decision_id/reason) relabeled stage='acquisition',
    action='not_retrieved' while KEEPING those FT-only fields must now
    raise -- the round-6 fix validated required-field PRESENCE but never
    rejected EXTRA keys, silently skipping ledger linkage entirely."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[],
        full_text_review_resolutions=_state_value([{
            "record_id": "r1", "stage": "acquisition", "action": "not_retrieved", "reviewer": "x", "note": "n",
            "resolved_at": "t",
            # FT-only fields kept from the original genuine ft_screening
            # resolution this row was relabeled from -- must be rejected.
            "resolution_id": "some-resolution-id", "reviewed_baseline_decision_id": "some-decision-id",
            "reason": "some reason",
        }]),
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    with pytest.raises(ReportingConsistencyError, match=r"acquisition-stage entry for record 'r1' has key set"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_acquisition_stage_resolution_confirmed_record_raises():
    """Codex round-7 review, finding 1 (P1): a correctly-shaped acquisition
    row (exact key set, valid action/fields) for a record that IS present
    in FT_SCREENING's own independently-verified confirmed_record_ids must
    still raise -- the record demonstrably reached FT_SCREENING, so it
    cannot legitimately be labeled 'never retrieved'. A matching current
    full_text_eligibility ledger tail for r1 is included so the round-4
    review's own corpus-completeness check (which runs BEFORE this
    resolution-linkage loop and independently requires exact record-ID-set
    equality between confirmed_record_ids and the ledger's own current
    tails) passes first, isolating THIS finding's own confirmed_record_ids
    cross-check as what actually raises."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    ft_entry = _ledger_verified_file(record_id="r1", stage="full_text_eligibility", decision="include",
                                      decision_maker="ai", source_event_id="e1")
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[ft_entry],
        full_text_review_resolutions=_state_value([{
            "record_id": "r1", "stage": "acquisition", "action": "not_retrieved", "reviewer": "x", "note": "n",
            "resolved_at": "t",
        }]),
        ft_native_provenance={"confirmed_record_ids": ["r1"]},
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    with pytest.raises(ReportingConsistencyError, match=r"record 'r1' is labeled stage='acquisition'.*confirmed_record_ids"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_acquisition_stage_resolution_with_ledger_record_raises():
    """Codex round-7 review, finding 1 (P1): a correctly-shaped acquisition
    row for a record with a real full_text_eligibility ledger entry, but
    NOT present in confirmed_record_ids, must still raise. NOTE: because
    the round-4 review's own corpus-completeness check (earlier in this
    same function) already requires EXACT record-ID-set equality between
    confirmed_record_ids and the ledger's own current full_text_eligibility
    tails, any ledger presence for r1 without a matching confirmed_
    record_ids entry is caught by THAT check first (with its own "does not
    match the expected corpus" message) -- this finding's own explicit
    ledger-existence check is a genuine, deliberate defense-in-depth
    safeguard for the SAME underlying invariant, not independently
    observable via this entry point given the current call order. This
    test asserts the overall (correct) rejection without pinning which of
    the two complementary checks fires."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    stale_entry = _ledger_verified_file(record_id="r1", stage="full_text_eligibility", decision="exclude",
                                         decision_maker="ai", source_event_id="e1")
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[stale_entry],
        full_text_review_resolutions=_state_value([{
            "record_id": "r1", "stage": "acquisition", "action": "not_retrieved", "reviewer": "x", "note": "n",
            "resolved_at": "t",
        }]),
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    with pytest.raises(ReportingConsistencyError,
                        match=r"(does not match the expected corpus|full_text_eligibility ledger record exists)"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_acquisition_stage_resolution_genuinely_absent_passes():
    """Positive companion: a genuine acquisition-stage resolution for a
    record truly absent from BOTH confirmed_record_ids and the ledger must
    still pass -- this is the real, legitimate case. This is exactly
    `test_compute_selection_splits_acquisition_stage_resolution_never_
    checked`'s own scenario, restated here as this finding's own explicit
    positive companion."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[],
        full_text_review_resolutions=_state_value([{
            "record_id": "r1", "stage": "acquisition", "action": "not_retrieved", "reviewer": "x", "note": "n",
            "resolved_at": "t",
        }]),
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    compute_selection_splits(snapshot=snapshot, prisma=prisma)  # must not raise


def test_compute_selection_splits_unknown_full_text_resolution_stage_raises():
    """Codex round-6 review, finding 3(b) (P1) -- exact reproduction:
    changing a genuine full-text resolution's own 'stage' field to
    anything other than 'acquisition'/'ft_screening' previously bypassed
    ALL linkage validation silently (`if resolution.get("stage") !=
    "ft_screening"): continue`); now it must raise immediately."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[],
        full_text_review_resolutions=_state_value([{"record_id": "r1", "stage": "bogus",
                                                      "action": "not_retrieved", "reviewer": "x", "note": "n",
                                                      "resolved_at": "t"}]),
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=0, excluded_full_text=0)
    with pytest.raises(ReportingConsistencyError, match=r"stage='bogus', which is neither 'acquisition' nor 'ft_screening'"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_screening_resolution_baseline_mismatch_raises():
    """Codex round-6 review, finding 3(a) (P1): a resolution whose
    resolution_id/source_event_id/decision all check out, but whose linked
    ledger entry's own supersedes_decision_id does NOT equal this
    resolution's own claimed reviewed_baseline_decision_id, must raise --
    otherwise a self-consistent resolution_id could claim baseline H0 while
    its ledger entry actually supersedes an unrelated decision A0."""
    cfg = ReviewConfig(title="t")
    prisma = PRISMACounts(identified={}, after_dedup=0, screened_title_abstract=1, excluded_title_abstract=1,
                           assessed_full_text=0, excluded_full_text=0)
    snapshot = _make_snapshot(cfg, prisma, SearchLog())
    real_baseline = _ledger_verified_file(record_id="r1", stage="title_abstract_screening", decision="borderline",
                                           decision_maker="ai", source_event_id="e-real-baseline")
    unrelated_baseline = _ledger_verified_file(record_id="r1", stage="title_abstract_screening", decision="include",
                                                decision_maker="ai", source_event_id="e-unrelated-baseline")
    # The resolution's OWN claimed baseline is `real_baseline`, so
    # resolution_id/source_event_id are both correctly derived from it --
    # but the human entry itself is published superseding `unrelated_
    # baseline` instead, which no honest resolution flow could ever
    # produce (a genuine human resolution always supersedes the baseline
    # it actually reviewed).
    resolution_id = compute_resolution_id(
        stage="title_abstract_screening", record_id="r1", action="exclude", reviewer="x", note="n",
        resolved_at="t", reviewed_baseline_decision_id=real_baseline.model.decision_id,
    )
    human_entry = _ledger_verified_file(
        record_id="r1", stage="title_abstract_screening", decision="exclude", decision_maker="human",
        source_event_id=derive_source_event_id(
            producer_artifact_id=resolution_id, record_id="r1", stage="title_abstract_screening"),
        supersedes_decision_id=unrelated_baseline.model.decision_id, producer_artifact_id=resolution_id)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[real_baseline, unrelated_baseline, human_entry],
        screening_records=[Record(id="r1", source="s", source_id="r1")],
        screening_review_resolutions=_state_value([{"record_id": "r1", "decision": "exclude", "reviewer": "x",
                                                      "note": "n", "resolved_at": "t",
                                                      "resolution_id": resolution_id,
                                                      "reviewed_baseline_decision_id": real_baseline.model.decision_id}]),
    )
    with pytest.raises(ReportingConsistencyError, match=r"linked ledger entry supersedes"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


def test_compute_selection_splits_fulltext_resolution_baseline_mismatch_raises():
    """Same as the screening-stage test above, for full_text_eligibility."""
    cfg = ReviewConfig(title="t")
    snapshot = _minimal_snapshot_for_publish(cfg)
    real_baseline = _ledger_verified_file(record_id="r1", stage="full_text_eligibility",
                                           decision="needs_human_review", decision_maker="ai", source_event_id="e-rb")
    unrelated_baseline = _ledger_verified_file(record_id="r1", stage="full_text_eligibility", decision="include",
                                                decision_maker="ai", source_event_id="e-ub")
    resolution_id = compute_resolution_id(
        stage="full_text_eligibility", record_id="r1", action="exclude", reviewer="x", note="n",
        resolved_at="t", reviewed_baseline_decision_id=real_baseline.model.decision_id, exclusion_reason=None,
    )
    human_entry = _ledger_verified_file(
        record_id="r1", stage="full_text_eligibility", decision="exclude", decision_maker="human",
        source_event_id=derive_source_event_id(
            producer_artifact_id=resolution_id, record_id="r1", stage="full_text_eligibility"),
        supersedes_decision_id=unrelated_baseline.model.decision_id, producer_artifact_id=resolution_id)
    snapshot = dataclasses.replace(
        snapshot, ledger_records=[real_baseline, unrelated_baseline, human_entry],
        full_text_review_resolutions=_state_value([{"record_id": "r1", "stage": "ft_screening", "action": "exclude",
                                                      "reviewer": "x", "note": "n", "resolved_at": "t",
                                                      "resolution_id": resolution_id,
                                                      "reviewed_baseline_decision_id": real_baseline.model.decision_id}]),
        ft_native_provenance={"confirmed_record_ids": ["r1"]},
    )
    prisma = PRISMACounts(identified={"db": 0}, after_dedup=0, assessed_full_text=1, excluded_full_text=1)
    with pytest.raises(ReportingConsistencyError, match=r"linked ledger entry supersedes"):
        compute_selection_splits(snapshot=snapshot, prisma=prisma)


# ---------------------------------------------------------------------------
# 2i. Deep-copy independence (INV-R22)
# ---------------------------------------------------------------------------

def test_state_value_deep_copy_independence():
    original = {"a": [1, 2, 3]}
    validated = _state_value(original)
    original["a"].append(4)
    assert validated.model == {"a": [1, 2, 3]}


# ---------------------------------------------------------------------------
# 2j. INV-R16 -- state['synthesis'] required
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2k. Manifest-first invalidation (INV-R15)
# ---------------------------------------------------------------------------

def test_manifest_first_invalidation(tmp_path):
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-manifest-test"
    report_dir = cfg.ensure_output_dir() / "reporting" / review_run_id
    report_dir.mkdir(parents=True)
    old_manifest = report_dir / "export_manifest.json"
    old_manifest.write_text('{"stale": true}')

    state = initial_state(cfg.to_dict())
    state["review_run_id"] = review_run_id
    # No selection_decision_ledger_dir in state -> _validate_ledger_dir raises,
    # AFTER the manifest has already been unlinked as the very first step.
    with pytest.raises(ReportingConsistencyError, match="no selection_decision_ledger_dir"):
        reporting_node(state)
    assert not old_manifest.exists()


# ---------------------------------------------------------------------------
# 2l. INV-R7 cross-check (oversight_mode/automation_origin re-derivation)
# ---------------------------------------------------------------------------

def _cross_check(*, cfg, oversight_mode_state, automation_origin_state, ledger_store, event_store,
                  current_screening_batches=(), screening_review_resolutions=(),
                  synthesis_requested_targets=(), imrad_narrative_enabled=False,
                  screening_qc_state=None, query_expansion_enabled=False,
                  search_strategy_producing_event_ids=(), search_strategy_db_queries=None,
                  search_strategy_expanded_query="", search_strategy_concepts=(), search_strategy_notes=(),
                  search_strategy_seed_query="", screened_records_full=(), screening_records=()):
    return reporting_module._cross_check_oversight_and_automation_origin(
        cfg=cfg, oversight_mode_state=oversight_mode_state, automation_origin_state=automation_origin_state,
        ledger_store=ledger_store, event_store=event_store,
        current_screening_batches=list(current_screening_batches),
        screening_review_resolutions=list(screening_review_resolutions),
        synthesis_requested_targets=list(synthesis_requested_targets),
        imrad_narrative_enabled=imrad_narrative_enabled,
        screening_qc_state=screening_qc_state, query_expansion_enabled=query_expansion_enabled,
        search_strategy_producing_event_ids=list(search_strategy_producing_event_ids),
        # Codex round-6 review, finding 4 stretch goal / round-12 review,
        # finding 2: defaults to a REAL, shape-valid db_queries (built from
        # this SAME `cfg`, matching `_baseline_search_strategy()`'s own
        # event), unless a test explicitly cares about binding/shape.
        search_strategy_db_queries=(search_strategy_db_queries if search_strategy_db_queries is not None else _default_db_queries(cfg)),
        # Codex round-7 review, finding 2: defaults to "" / [] / [] --
        # matching `canonical_query_expansion_output(expanded_query="",
        # concepts=[], notes=[])`'s own hash -- so a test that doesn't
        # enable query expansion never needs to care about this binding.
        search_strategy_expanded_query=search_strategy_expanded_query,
        search_strategy_concepts=list(search_strategy_concepts),
        search_strategy_notes=list(search_strategy_notes),
        # Codex round-9 review, finding 1: defaults to "" -- matching the
        # default search_strategy_expanded_query="" above -- so a test that
        # doesn't exercise the query-expansion failure branch never needs
        # to care about this binding.
        search_strategy_seed_query=search_strategy_seed_query,
        # Codex round-9 review, finding 2: defaults to () -- a test that
        # doesn't exercise the QC success-output-binding check never needs
        # to supply real Record objects.
        screened_records_full=list(screened_records_full),
        # Codex round-10 review, finding 3: defaults to () -- a test that
        # doesn't exercise the QC content-reconciliation check never needs
        # to supply real Record objects here; when a test DOES supply
        # screened_records_full with a matching record_id, it must supply
        # the SAME record's content here too (or the new reconciliation
        # check raises "not present in the durably-captured screening
        # corpus").
        screening_records=list(screening_records),
    )


def _dedup_event(event_store, *, review_run_id, work_item_id, call_status="success", stage="dedup_embedding"):
    event = build_ai_use_event(
        review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=0,
        tool_kind="non_llm_tool", model="m", provider="p", component_versions={}, access_info=None,
        task_purpose="t", model_config_id="c",
        input_artifact_refs=[], output_artifact_refs=[f"sha256:{sha256_hex([])}"], call_status=call_status,
    )
    return event_store.publish(event)


def _default_db_queries(cfg, query: str = "some real query") -> Dict[str, str]:
    """Codex round-12 review, finding 2 (P2): reporting.py now requires
    db_queries' own KEY SET to exactly equal `sorted(set(cfg.search.
    databases) | {'pubmed'})` (the real `dbs_needed` formula
    `_generate_db_queries()` itself uses), plus non-blank string values --
    a fixture's own `db_queries` must therefore be built from the SAME
    `cfg` a test passes to `_cross_check(cfg=...)`, never a fixed/partial
    dict like `{}`/`{"pubmed": "..."}`."""
    return {db: query for db in sorted(set(cfg.search.databases) | {"pubmed"})}


def _baseline_search_strategy(event_store, *, review_run_id, cfg) -> str:
    """Codex round-3 review, finding 1(a) (P1): search_strategy is now
    UNCONDITIONALLY required -- every `_cross_check(...)` call in this
    file that exercises a stage other than search_strategy itself must
    still satisfy that requirement, exactly like a real pipeline run
    always would (search_strategy_builder_node is the first node of every
    graph). Publishes one baseline event and returns its own `event_id` --
    callers pass this into `_cross_check(..., search_strategy_producing_
    event_ids=[...])` (Codex round-5 review, finding 3: the check is now
    anchored on this specific, per-invocation id list, not a re-scan of
    every event under this stage name) alongside the matching
    `oversight_mode_state["search_strategy"]`/`automation_origin_state[
    "search_strategy"]` entries via `_BASELINE_SEARCH_STRATEGY_STATE`
    below.

    Codex round-8 review, finding 2 (P1): reporting.py now binds the
    COMPLETE immutable call identity (work_item_id/rendered_prompt_hash/
    prompt_parameters/input_artifact_refs), reconstructed via the SAME
    shared `compute_db_translation_call_identity()` the real producer uses
    -- a hand-picked `work_item_id="w-baseline-ss"` can no longer pass.
    Built from the given `cfg` (`cfg.search.query`/`cfg.search.databases`/
    `cfg.search.strategy_builder.llm_model`) -- callers must pass the SAME
    `cfg` they pass to `_cross_check(cfg=..., search_strategy_expanded_
    query=...)` so both sides recompute identically (defaults align: a bare
    `ReviewConfig(title="t", output_dir=...)` has `query=""`, matching
    `_cross_check`'s own default `search_strategy_expanded_query=""`)."""
    identity = compute_db_translation_call_identity(
        query=cfg.search.query, databases=cfg.search.databases, model=cfg.search.strategy_builder.llm_model)
    event = build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_db_translation", work_item_id=identity["work_item_id"],
        # Codex round-9 review, finding 3 (P2): reporting.py now validates
        # EVERY deterministic call-site field (model/provider/task_purpose/
        # prompt_template_version/prompt_template_text), not just tool_kind/
        # model_config_id -- built entirely from `identity` (the SAME shared
        # builder the real producer uses) so this fixture can never drift
        # from what reporting.py actually recomputes.
        attempt=0, tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
        component_versions={}, access_info="http://x", task_purpose=identity["task_purpose"],
        model_config_id=identity["model_config_id"],
        prompt_template_version=identity["prompt_template_version"], prompt_template_text=identity["prompt_template_text"],
        prompt_parameters=identity["prompt_parameters"], rendered_prompt_hash=identity["rendered_prompt_hash"],
        input_artifact_refs=identity["input_artifact_refs"],
        # Codex round-6 review, finding 4 stretch goal / round-12 review,
        # finding 2: hashes a REAL, shape-valid db_queries (matching
        # `_cross_check`'s own default `search_strategy_db_queries`, both
        # built from this SAME `cfg` via `_default_db_queries()`), so both
        # the output-binding check AND the newer key-set/blank-value shape
        # checks pass for every test in this file that doesn't specifically
        # exercise db_queries itself.
        output_artifact_refs=[f"sha256:{sha256_hex(_default_db_queries(cfg))}"], call_status="success",
    )
    return event_store.publish(event).event_id


_BASELINE_SEARCH_STRATEGY_STATE: Dict[str, str] = {"search_strategy": "ai"}
_BASELINE_SEARCH_STRATEGY_OVERSIGHT: Dict[str, str] = {"search_strategy": "automation_unreviewed"}


def test_cross_check_dedup_mismatch_raises_when_event_exists(tmp_path):
    """Codex round-1 review, finding 1 (P1): dedup's re-derivation is
    EVENT-PRESENCE-based (stage='dedup_embedding'), not ledger-based --
    dedup_node always publishes exactly one dedup_embedding AIUseEvent per
    run, even for a zero-record corpus, while emitting ZERO ledger
    decisions in that case."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-x"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    _dedup_event(event_store, review_run_id=review_run_id, work_item_id="w1")
    with pytest.raises(ReportingConsistencyError, match=r"automation_origin\['dedup'\]"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"dedup": "automation_unreviewed"},
            automation_origin_state={"dedup": "ai"},  # WRONG -- dedup is always rule_based_automation
            ledger_store=ledger_store, event_store=event_store,
        )


def test_cross_check_dedup_agreement_passes_for_empty_corpus_zero_ledger_entries(tmp_path):
    """The exact bug scenario Codex reproduced: a zero-record corpus --
    dedup_node still publishes its one dedup_embedding event, but emits
    ZERO ledger decisions (there is nothing to decide on). A ledger-based
    re-derivation would incorrectly read (None, None) here and raise on a
    genuinely valid run; the event-presence-based re-derivation must pass."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-y"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    _dedup_event(event_store, review_run_id=review_run_id, work_item_id="w1", call_status="heuristic_fallback")
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    # NOTE: zero ledger entries published at all -- exactly the empty-corpus case.
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"dedup": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state={"dedup": "rule_based_automation", **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise


def test_cross_check_dedup_no_events_expects_absent_state(tmp_path):
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-nodedup"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state=dict(_BASELINE_SEARCH_STRATEGY_OVERSIGHT),
        automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),
        ledger_store=ledger_store, event_store=event_store,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise -- no dedup_embedding event exists at all


def test_cross_check_dedup_exact_agreement_passes_for_empty_corpus_zero_ledger_entries(tmp_path):
    """Step-4 follow-up: DedupConfig.mode="exact" publishes AIUseEvent.stage=
    "dedup_exact", not "dedup_embedding" (nodes/dedup.py) -- the event-
    presence check (reporting.py:851, `e.stage in ("dedup_embedding",
    "dedup_exact")`) was only ever exercised via the "dedup_embedding"
    literal above; this mirrors that same empty-corpus scenario via the
    "dedup_exact" literal, so a future narrowing of that tuple back to one
    stage would be caught here."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-exact"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    _dedup_event(event_store, review_run_id=review_run_id, work_item_id="w1", stage="dedup_exact")
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    # NOTE: zero ledger entries published at all -- exactly the empty-corpus case.
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"dedup": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state={"dedup": "rule_based_automation", **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise


def test_cross_check_synthesis_stage_skipped_when_no_requested_targets_and_imrad_disabled(tmp_path):
    """synthesis_claim_map/_refine are gated on a non-empty
    `synthesis_requested_targets` (the durable, re-verified output fact
    synthesis_node itself gates on); synthesis_imrad_narrative is gated on
    `imrad_narrative_enabled` (the real config fact). Neither state key nor
    any event needs to exist for this to correctly skip."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-z"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state=dict(_BASELINE_SEARCH_STRATEGY_OVERSIGHT),
        automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),
        ledger_store=ledger_store, event_store=event_store,
        synthesis_requested_targets=[], imrad_narrative_enabled=False,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise -- both facts say these stages did not run


def test_cross_check_synthesis_stage_deleting_state_keys_no_longer_bypasses_check(tmp_path):
    """Codex round-1 review, finding 2 (P1) -- the actual bug: deleting
    BOTH oversight_mode/automation_origin entries for synthesis_claim_map
    must NOT bypass this cross-check when `synthesis_requested_targets` (a
    durable, re-verified SynthesisOutput fact) and a real 'ai' event both
    prove the stage genuinely executed this run."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-synth-tamper"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    event = build_ai_use_event(
        review_run_id=review_run_id, stage="synthesis_claim_map", work_item_id="w1", attempt=0,
        tool_kind="non_llm_tool", model="m", provider="openai", component_versions={}, access_info="http://x",
        task_purpose="t", model_config_id="c",
        input_artifact_refs=[], output_artifact_refs=["out-1"], call_status="success",
    )
    event_store.publish(event)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    # BOTH state keys deleted/absent for synthesis_claim_map specifically --
    # the exact tamper scenario -- but search_strategy's OWN state must
    # still agree, so this test isolates the synthesis-stage bug alone.
    with pytest.raises(ReportingConsistencyError, match=r"automation_origin\['synthesis_claim_map'\]"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state=dict(_BASELINE_SEARCH_STRATEGY_OVERSIGHT),
            automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),
            ledger_store=ledger_store, event_store=event_store,
            synthesis_requested_targets=["rq_0"], imrad_narrative_enabled=False,
            search_strategy_producing_event_ids=[ss_event_id],
        )


def test_cross_check_synthesis_stage_agreement_passes(tmp_path):
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-synth-ok"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    for stage in ("synthesis_claim_map", "synthesis_claim_refine"):
        event = build_ai_use_event(
            review_run_id=review_run_id, stage=stage, work_item_id=f"w-{stage}", attempt=0,
            tool_kind="non_llm_tool", model="m", provider="openai", component_versions={}, access_info="http://x",
            task_purpose="t", model_config_id="c",
            input_artifact_refs=[], output_artifact_refs=["out-1"], call_status="success",
        )
        event_store.publish(event)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"synthesis_claim_map": "automation_unreviewed", "synthesis_claim_refine": "automation_unreviewed",
                               **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state={"synthesis_claim_map": "ai", "synthesis_claim_refine": "ai",
                                  **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store,
        synthesis_requested_targets=["rq_0"], imrad_narrative_enabled=False,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise


# ---------------------------------------------------------------------------
# 1b. Codex round-2 review, finding 1 (P1): search_strategy/screening_qc
# provenance checks were bypassable via `stage not in oversight_mode_state`.
# ---------------------------------------------------------------------------

_SEARCH_STRATEGY_MODEL_CONFIG_ID = {
    "search_strategy_db_translation": "search_strategy_db_translation_v2",
    "search_strategy_query_expansion": "search_strategy_query_expansion_v1",
}


def _search_strategy_event(event_store, *, review_run_id, stage, work_item_id, call_status="success",
                            output_artifact_refs=None, cfg=None, failure_detail=None):
    # Codex round-6 review, finding 4 stretch goal: `output_artifact_refs`
    # hashes {} (an empty dict) by default so a `search_strategy_db_
    # translation` event built via this helper matches `_cross_check`'s own
    # default `search_strategy_db_queries={}` output-binding check.
    # Codex round-7 review, finding 2 (P1): reporting.py now validates the
    # COMPLETE per-call-site contract -- `tool_kind`/`model_config_id` must
    # match search_strategy_builder.py's own real literals per stage, and
    # `output_artifact_refs` must exactly equal (not merely contain) the
    # expected single-element list; a "failure" query_expansion event must
    # pass `output_artifact_refs=[]` explicitly (no output on failure).
    # Codex round-10 review, finding 2 (P1): reporting.py now requires a
    # non-None failure_detail on any "failure"/"heuristic_fallback" event
    # (a real call site always sets one, str(exc)) and requires None on any
    # "success" event -- auto-default here so every pre-existing test that
    # only cares about a LATER check still reaches it, unless it explicitly
    # passes its own failure_detail (e.g. to test note-reconciliation).
    if failure_detail is None and call_status in ("failure", "heuristic_fallback"):
        failure_detail = "simulated failure"
    if output_artifact_refs is None:
        if call_status == "failure":
            output_artifact_refs = []
        elif stage == "search_strategy_query_expansion":
            # Must match `_cross_check`'s own default
            # search_strategy_expanded_query=""/concepts=[]/notes=[] via the
            # SAME canonical_query_expansion_output() the producer/reporting
            # both use -- NOT a bare sha256_hex({}) (that shape is only
            # correct for db_translation's own db_queries hash).
            output_artifact_refs = [f"sha256:{sha256_hex(canonical_query_expansion_output(expanded_query='', concepts=[], notes=[]))}"]
        else:
            # Codex round-12 review, finding 2 (P2): must hash a REAL,
            # shape-valid db_queries -- `_default_db_queries(cfg)` when a
            # `cfg` was given (matching `_cross_check`'s own default), else
            # `{}` (a test with no `cfg` never reaches the db_queries shape
            # checks anyway -- it fails an earlier check first).
            output_artifact_refs = [f"sha256:{sha256_hex(_default_db_queries(cfg) if cfg is not None else {})}"]
    # Codex round-8 review, finding 2 (P1): reporting.py now binds the
    # COMPLETE immutable call identity (work_item_id/rendered_prompt_hash/
    # prompt_parameters/input_artifact_refs), reconstructed via the SAME
    # shared identity builders the real producer uses. When a test cares
    # about that binding succeeding (not merely about an earlier check that
    # fires first), pass `cfg=cfg` -- the SAME `cfg` given to `_cross_check`
    # -- so this helper recomputes the REAL identity instead of using the
    # caller's own placeholder `work_item_id`/a fixed `rendered_prompt_hash`.
    if cfg is not None:
        if stage == "search_strategy_query_expansion":
            identity = compute_query_expansion_call_identity(cfg)
        else:
            identity = compute_db_translation_call_identity(
                query=cfg.search.query, databases=cfg.search.databases, model=cfg.search.strategy_builder.llm_model)
        work_item_id = identity["work_item_id"]
        rendered_prompt_hash = identity["rendered_prompt_hash"]
        prompt_parameters = identity["prompt_parameters"]
        input_artifact_refs = identity["input_artifact_refs"]
        # Codex round-9 review, finding 3 (P2): reporting.py now validates
        # EVERY deterministic call-site field, not just work_item_id/
        # rendered_prompt_hash/prompt_parameters/input_artifact_refs --
        # when a test cares about the full-identity binding succeeding
        # (cfg given), these must ALSO come from the real identity dict.
        model = identity["model"]
        task_purpose = identity["task_purpose"]
        prompt_template_text = identity["prompt_template_text"]
        prompt_template_version = identity["prompt_template_version"]
    else:
        rendered_prompt_hash = "h"
        prompt_parameters = {}
        input_artifact_refs = []
        model = "m"
        task_purpose = "t"
        prompt_template_text = "t"
        prompt_template_version = "v1"
    event = build_ai_use_event(
        review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=0,
        tool_kind="llm", model=model, provider="openai", component_versions={}, access_info="http://x",
        task_purpose=task_purpose, model_config_id=_SEARCH_STRATEGY_MODEL_CONFIG_ID[stage],
        # AIUseEvent's own model validator requires prompt-template
        # provenance whenever tool_kind="llm".
        prompt_template_version=prompt_template_version, prompt_template_text=prompt_template_text, rendered_prompt_hash=rendered_prompt_hash,
        prompt_parameters=prompt_parameters,
        input_artifact_refs=input_artifact_refs, output_artifact_refs=output_artifact_refs, call_status=call_status,
        failure_detail=failure_detail,
    )
    return event_store.publish(event)


def test_cross_check_search_strategy_deleting_state_keys_no_longer_bypasses_check(tmp_path):
    """Codex round-2 review, finding 1 (P1) -- exact reproduction: publish
    a real search-strategy event, delete BOTH oversight_mode/
    automation_origin state keys, confirm the cross-check still catches a
    genuine mismatch instead of silently skipping (the old
    `stage not in oversight_mode_state` gate let this through)."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w1", call_status="success", cfg=cfg)
    with pytest.raises(ReportingConsistencyError, match=r"automation_origin\['search_strategy'\]"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={}, automation_origin_state={},  # BOTH deleted
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
        )


def test_cross_check_search_strategy_agreement_passes(tmp_path):
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-ok"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w1", call_status="success", cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"search_strategy": "automation_unreviewed"},
        automation_origin_state={"search_strategy": "ai"},
        ledger_store=ledger_store, event_store=event_store,
        search_strategy_producing_event_ids=[dt_event.event_id],
    )  # must not raise


def test_cross_check_search_strategy_no_events_raises(tmp_path):
    """Codex round-3 review, finding 1(a) (P1): search_strategy_builder_node
    is the FIRST node of every review graph and always publishes at least
    one search_strategy_db_translation event -- a real invocation of
    reporting_node structurally guarantees this. The stage's own event(s)
    being entirely absent must now raise, not silently skip the check (the
    old `if search_strategy_events:` gate let deleting every
    search-strategy event pass with intact state). Codex round-4 review,
    finding 2 (P1): this must specifically require `db_translation` (the
    message now names it explicitly), not merely "either stage" -- see
    `test_cross_check_search_strategy_query_expansion_deleted_raises`
    below for the sibling scenario this round's fix closes."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-none"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    with pytest.raises(ReportingConsistencyError, match="no search_strategy_db_translation"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={}, automation_origin_state={},
            ledger_store=ledger_store, event_store=event_store,
        )


def test_cross_check_search_strategy_translation_deleted_query_expansion_survives_raises(tmp_path):
    """Codex round-4 review, finding 2 (P1) -- exact reproduction: if
    query expansion and translation both ran, deleting the (always-run)
    db_translation event while a query_expansion event survives must still
    raise -- round-3's "any event in either stage" check wrongly let this
    pass (origin stayed "ai" unchanged since classify_automation_origin_
    optional() only looked at the CATEGORY of whichever events remained)."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-translation-deleted"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="success")
    with pytest.raises(ReportingConsistencyError, match="no search_strategy_db_translation"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[qe_event.event_id],
        )


def test_cross_check_search_strategy_query_expansion_required_when_enabled(tmp_path):
    """Codex round-4 review, finding 2 (P1): when
    `cfg.search.strategy_builder.enabled=True`, query expansion always
    runs too -- a surviving db_translation event alone must not be
    sufficient; the query_expansion event's own absence must raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-missing"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    with pytest.raises(ReportingConsistencyError, match="no search_strategy_query_expansion"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id],
        )


def test_cross_check_search_strategy_query_expansion_forbidden_when_disabled(tmp_path):
    """Symmetric case: query expansion is CONFIGURED DISABLED, but a
    query_expansion event exists anyway (contrived/stale) -- must raise,
    since search_strategy_builder_node's own Step 1 never runs when this
    flag is unset."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-unexpected"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="success")
    with pytest.raises(ReportingConsistencyError, match="cfg.search.strategy_builder.enabled=False"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=False,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
        )


def test_search_strategy_producing_event_ids_rejects_duplicates():
    """Codex round-6 review, finding 4 (P2) -- exact reproduction:
    [same_translation_id, same_translation_id] must be rejected at
    construction time, not silently accepted as if it named two distinct
    events."""
    with pytest.raises(Exception, match="producing_event_ids must not contain duplicates"):
        SearchStrategy(producing_event_ids=["aiuse-" + "a" * 48, "aiuse-" + "a" * 48])


def test_cross_check_search_strategy_duplicate_translation_events_raises(tmp_path):
    """Codex round-6 review, finding 4 (P2): TWO distinct, genuinely real
    db_translation events (e.g. a stale historical one smuggled in
    alongside the current one) must raise -- 'at least one' is not the
    same requirement as 'exactly one'."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-two-translations"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt1 = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                  work_item_id="w-dt-1", call_status="success")
    dt2 = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                  work_item_id="w-dt-2", call_status="success")
    with pytest.raises(ReportingConsistencyError, match=r"2 search_strategy_db_translation events exist \(expected exactly one\)"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt1.event_id, dt2.event_id],
        )


def test_cross_check_search_strategy_output_binding_mismatch_raises(tmp_path):
    """Codex round-6 review, finding 4 stretch goal: the db_translation
    event's own output_artifact_refs must contain the hash of the CURRENT
    SearchStrategy.db_queries -- a state value edited independently of its
    claimed producing event (or a genuinely different event smuggled in)
    must raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-output-binding-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    # _search_strategy_event's own output_artifact_refs hashes {} -- but
    # here the CURRENT db_queries claims real content, which the event
    # never actually produced.
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    with pytest.raises(ReportingConsistencyError, match=r"does not exactly equal .* the hash of the CURRENT SearchStrategy\.db_queries"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
            search_strategy_db_queries={"pubmed": "some real query the event never actually produced"},
        )


def test_cross_check_search_strategy_db_queries_missing_database_raises(tmp_path):
    """Codex round-12 review, finding 2 (P2) -- exact reproduction: a
    self-consistent (hash-matching) db_queries missing one of the
    configured databases must raise -- the output-hash check alone proves
    consistency with WHATEVER state claims, never that the claim is itself
    a shape `_generate_db_queries()` could actually produce."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-db-queries-missing-db"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    incomplete = _default_db_queries(cfg)
    del incomplete[cfg.search.databases[0]]   # drop one configured database
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success",
                                       output_artifact_refs=[f"sha256:{sha256_hex(incomplete)}"])
    with pytest.raises(ReportingConsistencyError, match=r"db_queries key set .* does not exactly equal the expected database set"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
            search_strategy_db_queries=incomplete,
        )


def test_cross_check_search_strategy_db_queries_blank_value_on_success_raises(tmp_path):
    """Codex round-12 review, finding 2 (P2): a `call_status='success'`
    db_translation event's own db_queries can never legitimately have a
    blank query for any database -- `_generate_db_queries()`'s own real
    validation (the `empty` check) rejects that response and falls to the
    fallback branch instead."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-db-queries-blank-success"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    blank = _default_db_queries(cfg)
    blank[cfg.search.databases[0]] = "   "   # whitespace-only -- blank after strip()
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success",
                                       output_artifact_refs=[f"sha256:{sha256_hex(blank)}"])
    with pytest.raises(ReportingConsistencyError, match=r"call_status='success', but db_queries has a blank query"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
            search_strategy_db_queries=blank,
        )


def test_cross_check_search_strategy_db_queries_fallback_per_db_variation_raises(tmp_path):
    """Codex round-12 review, finding 2 (P2): a genuine `heuristic_fallback`
    db_queries is ALWAYS the identical query for every database
    (`{db: query for db in dbs_needed}`) -- per-database variation in the
    fallback case is an impossible producer shape."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-db-queries-fallback-variation"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    varying = _default_db_queries(cfg, query="")   # matches search_strategy_expanded_query="" for every db...
    varying[cfg.search.databases[0]] = "a different query for just this one db"   # ...except this one
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="heuristic_fallback",
                                       output_artifact_refs=[f"sha256:{sha256_hex(varying)}"])
    with pytest.raises(ReportingConsistencyError, match=r"call_status='heuristic_fallback', but db_queries has value\(s\) differing from the current expanded_query"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "rule_based_automation"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
            search_strategy_db_queries=varying,
        )


def test_cross_check_search_strategy_db_queries_fallback_blank_value_passes(tmp_path):
    """Codex round-12 review, finding 2 (P2) -- the real edge case that
    motivated NOT extending the blank-value check to the fallback branch:
    a genuinely blank `cfg.search.query` (never configured) combined with
    a failed translation call legitimately produces an all-blank fallback
    db_queries (`_generate_db_queries()`'s fallback blindly reuses whatever
    `expanded_query` it was given, with no non-blank requirement of its
    own) -- this is REAL, producer-reachable output, not a shape violation,
    and must NOT raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))   # cfg.search.query defaults to ""
    review_run_id = "run-ss-db-queries-fallback-blank-ok"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    all_blank = _default_db_queries(cfg, query="")
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="heuristic_fallback", cfg=cfg,
                                       output_artifact_refs=[f"sha256:{sha256_hex(all_blank)}"])
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"search_strategy": "automation_unreviewed"},
        automation_origin_state={"search_strategy": "rule_based_automation"},
        ledger_store=ledger_store, event_store=event_store,
        search_strategy_producing_event_ids=[dt_event.event_id],
        search_strategy_db_queries=all_blank,
    )  # must NOT raise -- expanded_query="" (the default) makes this the real, correct fallback shape


def test_cross_check_search_strategy_query_expansion_enabled_agreement_passes(tmp_path):
    """Both stages present, both configured/expected -- must not raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-ok"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    # Codex round-8 review, finding 2 (P1): both events now need the COMPLETE
    # real call identity (work_item_id/rendered_prompt_hash/prompt_parameters/
    # input_artifact_refs), recomputed from this SAME `cfg` -- `cfg=cfg`
    # routes `_search_strategy_event()` through the real shared identity
    # builders instead of a hand-picked `work_item_id`.
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success", cfg=cfg)
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="success", cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"search_strategy": "automation_unreviewed"},
        automation_origin_state={"search_strategy": "ai"},
        ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
        search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
    )  # must not raise


def test_cross_check_search_strategy_query_expansion_impossible_fallback_status_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1) -- exact reproduction: a
    query_expansion event can only ever be 'success' or 'failure'
    (_run_query_expansion() has no computed fallback); a
    call_status='heuristic_fallback' event is an impossible shape and must
    raise even though it satisfies presence+cardinality."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-impossible-status"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="heuristic_fallback",
                                       output_artifact_refs=["sha256:old-unrelated-output"])
    with pytest.raises(ReportingConsistencyError, match=r"query_expansion event has call_status='heuristic_fallback'"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "mixed"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
        )


def test_cross_check_search_strategy_query_expansion_output_binding_mismatch_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1): a successful query_expansion
    event whose output_artifact_refs does not match a freshly-recomputed
    hash of the CURRENT expanded_query/concepts/notes (stale/unrelated
    content, or an unrelated historical event substituted in place of the
    current one) must raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-output-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    # This helper's default success output hashes canonical_query_expansion_
    # output(expanded_query="", concepts=[], notes=[]) -- but the CURRENT
    # state below claims a genuinely different expanded_query.
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="success")
    with pytest.raises(ReportingConsistencyError, match=r"query_expansion event's output_artifact_refs .* does not exactly equal"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
            search_strategy_expanded_query="a genuinely different expanded query",
        )


def test_cross_check_search_strategy_query_expansion_failure_with_nonempty_output_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1): a call_status='failure'
    query_expansion event must have output_artifact_refs=[] exactly
    (INV-06 -- a failure never has an output)."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-failure-nonempty-output"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="failure",
                                       output_artifact_refs=["sha256:should-not-exist"])
    with pytest.raises(ReportingConsistencyError, match=r"call_status='failure' but output_artifact_refs"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
        )


def test_cross_check_search_strategy_db_translation_impossible_failure_status_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1): _generate_db_queries() always
    has a real computed fallback (same query for every DB) -- a
    call_status='failure' db_translation event is an impossible shape."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-dt-impossible-failure"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="failure", output_artifact_refs=[])
    with pytest.raises(ReportingConsistencyError, match=r"db_translation event has call_status='failure'"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
        )


def test_cross_check_search_strategy_db_translation_extra_output_ref_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1): the old membership check (`not
    in`) let an extra, unrelated ref alongside the correct one pass -- the
    producer's own shape is always EXACTLY one element; exact list
    equality must catch an extra entry."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-dt-extra-ref"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    correct_ref = f"sha256:{sha256_hex({})}"
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success",
                                       output_artifact_refs=[correct_ref, "sha256:extra-unrelated-ref"])
    with pytest.raises(ReportingConsistencyError, match=r"db_translation event's output_artifact_refs .* does not exactly equal"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
        )


def test_cross_check_search_strategy_tool_kind_mismatch_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1): a db_translation event whose
    tool_kind doesn't match the producer's own real literal must raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-dt-tool-kind-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_db_translation", work_item_id="w-dt", attempt=0,
        tool_kind="non_llm_tool", model="m", provider="openai", component_versions={}, access_info="http://x",
        task_purpose="t", model_config_id="search_strategy_db_translation_v2",
        input_artifact_refs=[], output_artifact_refs=[f"sha256:{sha256_hex({})}"], call_status="success",
    ))
    with pytest.raises(ReportingConsistencyError, match=r"db_translation event has tool_kind='non_llm_tool', expected 'llm'"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
        )


def test_cross_check_search_strategy_model_config_id_mismatch_raises(tmp_path):
    """Codex round-7 review, finding 2 (P1): a db_translation event whose
    model_config_id doesn't match the producer's own real literal must
    raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-dt-model-config-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_db_translation", work_item_id="w-dt", attempt=0,
        tool_kind="llm", model="m", provider="openai", component_versions={}, access_info="http://x",
        task_purpose="t", model_config_id="wrong_model_config_id",
        prompt_template_version="v1", prompt_template_text="t", rendered_prompt_hash="h",
        input_artifact_refs=[], output_artifact_refs=[f"sha256:{sha256_hex({})}"], call_status="success",
    ))
    with pytest.raises(ReportingConsistencyError, match=r"db_translation event has model_config_id='wrong_model_config_id'"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[dt_event.event_id],
        )


def test_cross_check_search_strategy_full_contract_happy_path_passes(tmp_path):
    """Positive companion: the full legitimate happy path (real
    query-expansion + db-translation success events with genuinely
    matching content) must still pass cleanly.

    Codex round-8 review, finding 2 (P1): reporting.py now binds the
    COMPLETE immutable call identity (work_item_id/rendered_prompt_hash/
    prompt_parameters/input_artifact_refs). The db_translation event's
    identity must be built from THIS invocation's own actual
    `expanded_query` ("real expanded query" -- what Step 1 of this
    invocation actually produced), NOT `cfg.search.query`'s default ""
    -- so it cannot use `_search_strategy_event`'s own `cfg=` shortcut
    (which always derives db_translation identity from `cfg.search.query`
    directly, matching the common case where no query-expansion override
    is in play). Built directly via the same shared
    `compute_db_translation_call_identity()`/`compute_query_expansion_
    call_identity()` the real producer uses, exactly like
    `_baseline_search_strategy()` does."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-full-contract-happy"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    # Codex round-12 review, finding 2 (P2): db_queries' own key set must
    # exactly equal sorted(set(cfg.search.databases) | {"pubmed"}).
    db_queries = _default_db_queries(cfg, query="real query")
    dt_identity = compute_db_translation_call_identity(
        query="real expanded query", databases=cfg.search.databases, model=cfg.search.strategy_builder.llm_model)
    dt_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_db_translation", work_item_id=dt_identity["work_item_id"],
        attempt=0, tool_kind=dt_identity["tool_kind"], model=dt_identity["model"], provider=dt_identity["provider"],
        component_versions={}, access_info="http://x", task_purpose=dt_identity["task_purpose"],
        model_config_id=dt_identity["model_config_id"],
        prompt_template_version=dt_identity["prompt_template_version"], prompt_template_text=dt_identity["prompt_template_text"],
        prompt_parameters=dt_identity["prompt_parameters"], rendered_prompt_hash=dt_identity["rendered_prompt_hash"],
        input_artifact_refs=dt_identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex(db_queries)}"], call_status="success",
    ))
    concepts = [{"label": "Population", "seed_terms": ["x"], "synonyms": [], "controlled_terms": []}]
    notes = ["a note"]
    expected_qe_output = canonical_query_expansion_output(expanded_query="real expanded query", concepts=concepts, notes=notes)
    qe_identity = compute_query_expansion_call_identity(cfg)
    qe_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_query_expansion", work_item_id=qe_identity["work_item_id"],
        attempt=0, tool_kind=qe_identity["tool_kind"], model=qe_identity["model"], provider=qe_identity["provider"],
        component_versions={}, access_info="http://x", task_purpose=qe_identity["task_purpose"],
        model_config_id="search_strategy_query_expansion_v1",
        prompt_template_version=qe_identity["prompt_template_version"], prompt_template_text=qe_identity["prompt_template_text"],
        prompt_parameters=qe_identity["prompt_parameters"], rendered_prompt_hash=qe_identity["rendered_prompt_hash"],
        input_artifact_refs=qe_identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex(expected_qe_output)}"], call_status="success",
    ))
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"search_strategy": "automation_unreviewed"},
        automation_origin_state={"search_strategy": "ai"},
        ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
        search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
        search_strategy_db_queries=db_queries,
        search_strategy_expanded_query="real expanded query", search_strategy_concepts=concepts,
        search_strategy_notes=notes,
    )  # must not raise


def test_cross_check_search_strategy_query_expansion_wrong_identity_failure_substitution_raises(tmp_path):
    """Codex round-8 review, finding 2 (P1): a call_status='failure' event
    ALWAYS has output_artifact_refs=[] by construction, so every failure
    event for this stage is indistinguishable by output alone -- only the
    complete immutable call identity (work_item_id/rendered_prompt_hash/
    prompt_parameters/input_artifact_refs) can catch a substituted event.
    Publishes a genuine failure event for a DIFFERENT config (different
    research_questions -> different rendered prompt/work_item_id) and
    references it as if it were the current invocation's own event; the
    substitute's output_artifact_refs=[] trivially matches what a REAL
    current failure would also have, so only the identity fields can
    catch this."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["q1"])
    other_cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["an unrelated question"])
    review_run_id = "run-ss-qe-wrong-identity-failure"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    wrong_identity = compute_query_expansion_call_identity(other_cfg)
    # Codex round-9 review, finding 1 (P1): a failure event's own
    # failure_detail must reconcile against the CURRENT state's own notes --
    # given a real, non-blank failure_detail here (and matching notes below)
    # so THAT check passes cleanly, isolating this test to what it actually
    # means to exercise (the work_item_id substitution), same as every other
    # field this event deliberately leaves correct.
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_query_expansion",
        work_item_id=wrong_identity["work_item_id"], attempt=0, tool_kind="llm",
        model=cfg.search.strategy_builder.llm_model, provider="openai", component_versions={},
        access_info="http://x", task_purpose="t", model_config_id="search_strategy_query_expansion_v1",
        prompt_template_version="v1", prompt_template_text="t",
        prompt_parameters=wrong_identity["prompt_parameters"], rendered_prompt_hash=wrong_identity["rendered_prompt_hash"],
        input_artifact_refs=wrong_identity["input_artifact_refs"],
        output_artifact_refs=[], call_status="failure", failure_detail="an unrelated transport failure",
    ))
    dt_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    with pytest.raises(ReportingConsistencyError, match=r"query_expansion event's work_item_id .* does not match the CURRENT invocation's own recomputed"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "no_automated_output"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event_id, wrong_event.event_id],
            search_strategy_notes=["Query expanded with 0 concept groups.", "Expansion failed: an unrelated transport failure"],
        )


def test_cross_check_search_strategy_query_expansion_wrong_identity_success_coincidental_output_raises(tmp_path):
    """Codex round-8 review, finding 2 (P1): a SUCCESS event from a
    DIFFERENT config whose own expanded_query/concepts/notes happen to
    coincide with the current invocation's own (so output_artifact_refs
    matches too) must still be rejected via the identity fields -- output
    equality alone is not proof of the same underlying call."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["q1"])
    other_cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["a different question"])
    review_run_id = "run-ss-qe-wrong-identity-success"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    # Both cfgs produce the SAME expanded_query/concepts/notes here (the
    # test controls what `_cross_check` claims as "current" output), so the
    # output hash coincides by construction -- only the identity fields can
    # distinguish `other_cfg`'s real event from `cfg`'s own expected one.
    coincidental_output = canonical_query_expansion_output(expanded_query="(a OR b)", concepts=[], notes=["ok"])
    wrong_identity = compute_query_expansion_call_identity(other_cfg)
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_query_expansion",
        work_item_id=wrong_identity["work_item_id"], attempt=0, tool_kind="llm",
        model=cfg.search.strategy_builder.llm_model, provider="openai", component_versions={},
        access_info="http://x", task_purpose="t", model_config_id="search_strategy_query_expansion_v1",
        prompt_template_version="v1", prompt_template_text="t",
        prompt_parameters=wrong_identity["prompt_parameters"], rendered_prompt_hash=wrong_identity["rendered_prompt_hash"],
        input_artifact_refs=wrong_identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex(coincidental_output)}"], call_status="success",
    ))
    dt_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    with pytest.raises(ReportingConsistencyError, match=r"query_expansion event's work_item_id .* does not match the CURRENT invocation's own recomputed"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event_id, wrong_event.event_id],
            search_strategy_expanded_query="(a OR b)", search_strategy_concepts=[], search_strategy_notes=["ok"],
        )


def test_cross_check_search_strategy_db_translation_wrong_identity_coincidental_output_raises(tmp_path):
    """Codex round-8 review, finding 2 (P1), db_translation side: a wrong-
    config event whose own db_queries output happens to coincide with the
    current invocation's claimed `search_strategy_db_queries` (the test
    controls this directly, since it's passed into `_cross_check`
    independent of cfg) must still be rejected via the identity fields."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["q1"])
    other_cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["q1"])
    other_cfg.search.query = "a completely different seed query"
    review_run_id = "run-ss-dt-wrong-identity"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    # Codex round-12 review, finding 2 (P2): db_queries' own key set must
    # exactly equal sorted(set(cfg.search.databases) | {"pubmed"}).
    db_queries = _default_db_queries(cfg, query="real query")
    wrong_identity = compute_db_translation_call_identity(
        query=other_cfg.search.query, databases=other_cfg.search.databases, model=other_cfg.search.strategy_builder.llm_model)
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_db_translation",
        work_item_id=wrong_identity["work_item_id"], attempt=0, tool_kind="llm",
        model=cfg.search.strategy_builder.llm_model, provider="openai", component_versions={},
        access_info="http://x", task_purpose="t", model_config_id="search_strategy_db_translation_v2",
        prompt_template_version="v1", prompt_template_text="t",
        prompt_parameters=wrong_identity["prompt_parameters"], rendered_prompt_hash=wrong_identity["rendered_prompt_hash"],
        input_artifact_refs=wrong_identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex(db_queries)}"], call_status="success",
    ))
    with pytest.raises(ReportingConsistencyError, match=r"db_translation event's work_item_id .* does not match the CURRENT invocation's own recomputed"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store,
            search_strategy_producing_event_ids=[wrong_event.event_id],
            search_strategy_db_queries=db_queries,
        )


def test_cross_check_search_strategy_real_node_lifecycle_query_expansion_hash_matches(tmp_path, monkeypatch):
    """Codex round-8 review, finding 1 (P1) -- MANDATORY regression per
    Codex's own explicit instruction ("Add a regression using the actual
    builder output"). Every other test in this module hand-constructs
    `SearchStrategy.notes`/events directly, which is exactly what let the
    real bug slip through: `_run_query_expansion()` used to compute/publish
    its event's output hash from the RAW notes BEFORE the caller
    (`search_strategy_builder_node`) prepended "Query expanded with N
    concept groups." to them, while reporting's `canonical_query_expansion_
    output()` reconstruction hashes the FINAL (already-prepended)
    `state["search_strategy"]["notes"]` -- the two could never match for a
    real, unmodified successful run. This test runs the REAL
    `search_strategy_builder_node()`/`_run_query_expansion()` code path
    (mocked LLM, real internal logic) and feeds its OWN real output
    straight into reporting's cross-check, proving producer and verifier
    now agree end-to-end -- not just that two independently-computed test
    fixtures happen to match."""
    def _responder(kwargs: Dict[str, Any]) -> str:
        user = kwargs["messages"][1]["content"]
        if "adapting a PubMed" in user:
            return json.dumps({"openalex": "translated query", "pubmed": "orig"})
        return json.dumps({
            "concepts": [{"label": "Population", "seed_terms": ["x"], "synonyms": [], "controlled_terms": []}],
            "candidate_query": "(a OR b)", "notes": ["a genuine note from the provider"],
        })

    monkeypatch.setattr(search_strategy_builder_module, "OpenAI", _make_dispatching_openai_class(_responder))

    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["q"])
    cfg.search.databases = ["openalex"]
    cfg.search.strategy_builder.enabled = True
    review_run_id = "run-ss-real-lifecycle"
    state = dict(initial_state(cfg.to_dict()))
    state["review_run_id"] = review_run_id

    result = search_strategy_builder_node(state)
    search_strategy = result["search_strategy"]
    # Confirm the real lifecycle actually exercised the prepend this test
    # is about -- if this assertion ever stops holding, the regression
    # below would silently stop testing what it claims to.
    assert search_strategy["notes"][0].startswith("Query expanded with")

    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)

    _cross_check(
        cfg=cfg,
        oversight_mode_state=result["oversight_mode"], automation_origin_state=result["automation_origin"],
        ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
        search_strategy_producing_event_ids=search_strategy["producing_event_ids"],
        search_strategy_db_queries=search_strategy["db_queries"],
        search_strategy_expanded_query=search_strategy["expanded_query"],
        search_strategy_concepts=search_strategy["concepts"], search_strategy_notes=search_strategy["notes"],
    )  # must NOT raise -- this is the real producer/verifier hash boundary Codex found broken


def test_cross_check_search_strategy_query_expansion_failure_wrong_expanded_query_raises(tmp_path):
    """Codex round-9 review, finding 1 (P1): a call_status='failure'
    query_expansion event means `_run_query_expansion()`'s own failure path
    never reassigns `expanded_query` away from the seed query -- a
    contrived state where they diverge must raise, even though
    output_artifact_refs is correctly empty (output-emptiness alone cannot
    distinguish THIS invocation's own failure from a substituted one)."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-failure-wrong-expanded-query"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="failure", output_artifact_refs=[])
    with pytest.raises(ReportingConsistencyError, match=r"expanded_query \(.*\) != seed_query"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
            search_strategy_expanded_query="a diverged query",   # != seed_query ("" default)
        )


def test_cross_check_search_strategy_query_expansion_failure_nonempty_concepts_raises(tmp_path):
    """Codex round-9 review, finding 1 (P1): a failure path never populates
    `concepts` -- a contrived non-empty value must raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-failure-nonempty-concepts"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="failure", output_artifact_refs=[])
    with pytest.raises(ReportingConsistencyError, match=r"SearchStrategy\.concepts is non-empty"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
            search_strategy_concepts=[{"label": "x", "seed_terms": [], "synonyms": [], "controlled_terms": []}],
        )


def test_cross_check_search_strategy_query_expansion_failure_note_mismatch_raises(tmp_path):
    """Codex round-9 review, finding 1 (P1) -- exact reproduction: an OLDER,
    same-configuration failure event whose own `failure_detail` does NOT
    match the CURRENT state's own note text must raise -- output-emptiness
    alone is identical for every failure event of this stage, so only this
    reconciliation can catch a substituted failure event."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-qe-failure-note-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    dt_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                                       work_item_id="w-dt", call_status="success")
    qe_event = _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_query_expansion",
                                       work_item_id="w-qe", call_status="failure", output_artifact_refs=[])
    with pytest.raises(ReportingConsistencyError, match=r"does not reconcile with the CURRENT"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"search_strategy": "automation_unreviewed"},
            automation_origin_state={"search_strategy": "ai"},
            ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
            search_strategy_producing_event_ids=[dt_event.event_id, qe_event.event_id],
            search_strategy_notes=["Query expanded with 0 concept groups.", "Expansion failed: an unrelated older failure"],
        )


def test_cross_check_search_strategy_real_node_lifecycle_query_expansion_failure_reconciles(tmp_path, monkeypatch):
    """Codex round-9 review, finding 1 (P1) -- the failure-path sibling of
    round-8's own mandatory success-path regression: runs the REAL
    `search_strategy_builder_node()`/`_run_query_expansion()` code path
    with a genuinely failing provider call (mocked transport failure, real
    internal logic) and feeds its own real output straight into reporting's
    cross-check, proving the failure-state reconciliation this round adds
    (expanded_query==seed_query, concepts==[], notes reconciled against
    failure_detail) agrees with what a real failure run actually produces
    -- not just a hand-constructed approximation of it."""
    monkeypatch.setattr(search_strategy_builder_module, "OpenAI", _FailingOpenAI)

    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["q"])
    cfg.search.databases = ["openalex"]
    cfg.search.strategy_builder.enabled = True
    review_run_id = "run-ss-real-lifecycle-failure"
    state = dict(initial_state(cfg.to_dict()))
    state["review_run_id"] = review_run_id

    result = search_strategy_builder_node(state)
    search_strategy = result["search_strategy"]
    # Confirm the real lifecycle actually exercised the failure path this
    # test is about.
    assert search_strategy["expanded_query"] == cfg.search.query
    assert search_strategy["concepts"] == []
    assert search_strategy["notes"][0].startswith("Query expanded with 0 concept groups")
    assert search_strategy["notes"][1].startswith("Expansion failed:")

    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)

    _cross_check(
        cfg=cfg,
        oversight_mode_state=result["oversight_mode"], automation_origin_state=result["automation_origin"],
        ledger_store=ledger_store, event_store=event_store, query_expansion_enabled=True,
        search_strategy_producing_event_ids=search_strategy["producing_event_ids"],
        search_strategy_db_queries=search_strategy["db_queries"],
        search_strategy_expanded_query=search_strategy["expanded_query"],
        search_strategy_concepts=search_strategy["concepts"], search_strategy_notes=search_strategy["notes"],
        search_strategy_seed_query=search_strategy["seed_query"],
    )  # must NOT raise -- the real producer/verifier failure-state boundary this round adds


def test_cross_check_screening_qc_deleting_state_keys_no_longer_bypasses_check(tmp_path):
    """Codex round-2 review, finding 1 (P1): screening_qc's own gate is
    `screening_qc_state is not None`, NOT event presence -- a legitimately
    enabled QC run can have zero eligible-record events yet still ran and
    still set oversight_mode. Deleting both state keys must not bypass the
    check when screening_qc_state itself durably proves the stage ran."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(n_attempted=0, n_valid_pairs=0, n_failed=0))
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    with pytest.raises(ReportingConsistencyError, match=r"oversight_mode\['screening_qc_second_rater'\]"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state=dict(_BASELINE_SEARCH_STRATEGY_OVERSIGHT),
            automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),  # QC's own keys BOTH deleted
            ledger_store=ledger_store, event_store=event_store,
            screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
        )


def test_cross_check_screening_qc_zero_records_zero_events_agreement_passes(tmp_path):
    """The exact bug scenario Codex named: an ENABLED, zero-eligible-record
    QC run has no events at all, yet legitimately ran and set oversight to
    'automation_unreviewed' with automation_origin left unset -- must pass,
    not be mistaken for "did not run"."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-empty"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(n_attempted=0, n_valid_pairs=0, n_failed=0))
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),  # QC's own origin correctly absent -- zero events
        ledger_store=ledger_store, event_store=event_store,
        screening_qc_state=qc_state,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise


def test_cross_check_screening_qc_none_state_skips_check(tmp_path):
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-disabled"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    _cross_check(
        cfg=cfg,
        oversight_mode_state=dict(_BASELINE_SEARCH_STRATEGY_OVERSIGHT),
        automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),
        ledger_store=ledger_store, event_store=event_store,
        screening_qc_state=None,
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise -- QC disabled this run, gate is state-presence not event-presence


def _qc_record(record_id: str) -> "Record":
    """A stable, reproducible Record for QC test fixtures -- title/
    abstract/doi fixed per record_id so `_record_content_ref()`'s own hash
    (part of the call identity) is identical whether computed by `_qc_event()`
    building the event or by a test passing this SAME record into
    `screened_records_full` for `_cross_check`.

    `screening_decision="include"` (Codex round-11 review, finding 1):
    `compute_qc_eligible_records()` -- both the real node's and reporting's
    own deterministic sample-size check -- filters on
    `screening_decision is not None`; a record genuinely eligible for QC
    sampling always has one (screening_node always sets it before
    screening_qc_node ever runs)."""
    return Record(id=record_id, source="s", source_id=record_id, title=f"title-{record_id}", abstract="abstract",
                  doi=None, screening_decision="include")


def _qc_current_screening_batch(records: List["Record"], *, review_run_id: str) -> VerifiedFile:
    """Codex round-12 review, finding 1 (P1): QC eligibility/sample-size is
    now anchored to the FROZEN `current_screening_batches` projection
    (each record's decision zipped from a real `ScreeningBatchResult`),
    never `screened_records_full.screening_decision` directly. Every QC
    test fixture that supplies `screening_records=[...]` must therefore
    also supply a matching `current_screening_batches=[_qc_current_
    screening_batch(records, ...)]` whose own `.responses[i].decision`
    equals `records[i].screening_decision` exactly (`_qc_record()`'s own
    `"include"` by default) -- reporting.py only ever reads `.record_ids`/
    `.responses` off this object; it does not itself re-verify the
    checkpoint against disk/its own producing event in a test context
    (that verification is `_load_current_screening_batches()`'s own job,
    exercised separately by the real-lifecycle/full-pipeline tests)."""
    responses = [
        ScreeningResponse(decision=r.screening_decision, reason="r", relevant_topics=[], confidence=0.9)
        for r in records
    ]
    result = build_screening_batch_result(
        review_run_id=review_run_id, stage="title_abstract_screening", batch_fingerprint="fp-test-fixture",
        record_ids=[r.id for r in records], producing_event_id="test-producing-event",
        call_status="success", responses=responses,
    )
    return VerifiedFile(model=result, path=Path("test-batch-fixture"), file_sha256="a" * 64,
                         internal_hash=result.compute_content_hash())


def _qc_event(event_store, *, review_run_id, record_id, work_item_id=None, call_status, cfg=None, decision="include", attempt=0):
    """Mirrors screening_qc.py's own `_second_rater_screen()` exactly:
    `input_artifact_refs=[criteria_ref, record_ref]`, where `record_ref`
    is `f"record:{record.id}:sha256:{...}"`.

    Codex round-9 review, finding 2 (P1): reporting.py now binds the
    COMPLETE immutable call identity (work_item_id/rendered_prompt_hash/
    prompt_parameters/input_artifact_refs/model/provider/tool_kind/
    task_purpose/prompt_template_version/prompt_template_text), plus, for
    a success event, the output's own correspondence to the CURRENT
    Record.second_rater_decision. When `cfg` is given, this helper builds
    a genuinely correct event via the SAME shared
    `compute_qc_second_rater_call_identity()` the real producer uses,
    for `_qc_record(record_id)` -- a caller that needs the full-identity
    check to PASS must also pass a matching `screened_records_full=
    [_qc_record(record_id), ...]` (with `.second_rater_decision` set to
    `decision` for a success event) into `_cross_check`. When `cfg` is
    `None` (the pre-round-9 default, `work_item_id` then required), this
    stays a bare placeholder event -- only usable by a test where some
    EARLIER check in the cross-check's own execution order (record-id-set
    mismatch, call_status/count mismatch, or a missing/deleted event)
    raises before the full-identity check would ever be reached."""
    # Codex round-10 review, finding 2 (P1): reporting.py now requires a
    # non-None failure_detail on any "failure" event (a real call always
    # sets str(exc)) and None on a "success" event -- auto-default here so
    # every pre-existing test that only cares about a LATER check still
    # reaches it.
    failure_detail = "simulated failure" if call_status == "failure" else None
    if cfg is not None:
        record = _qc_record(record_id)
        model = cfg.screening_qc.second_rater_model or cfg.screening.llm_model
        identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model=model)
        output_artifact_refs = (
            [f"sha256:{sha256_hex({'decision': decision})}"] if call_status == "success" else []
        )
        event = build_ai_use_event(
            review_run_id=review_run_id, stage="screening_qc_second_rater", work_item_id=identity["work_item_id"],
            attempt=attempt, tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
            component_versions={}, access_info="http://x", task_purpose=identity["task_purpose"],
            model_config_id=identity["model_config_id"],
            prompt_template_version=identity["prompt_template_version"], prompt_template_text=identity["prompt_template_text"],
            prompt_parameters=identity["prompt_parameters"], rendered_prompt_hash=identity["rendered_prompt_hash"],
            input_artifact_refs=identity["input_artifact_refs"],
            output_artifact_refs=output_artifact_refs, call_status=call_status, failure_detail=failure_detail,
        )
        return event_store.publish(event)
    assert work_item_id is not None, "work_item_id is required when cfg is not given"
    event = build_ai_use_event(
        review_run_id=review_run_id, stage="screening_qc_second_rater", work_item_id=work_item_id, attempt=0,
        tool_kind="non_llm_tool", model="m", provider="openai", component_versions={}, access_info="http://x",
        task_purpose="t", model_config_id="c",
        input_artifact_refs=[f"criteria:sha256:{sha256_hex({})}", f"record:{record_id}:sha256:{sha256_hex({})}"],
        output_artifact_refs=([f"sha256:{sha256_hex({'decision': 'include'})}"] if call_status == "success" else []),
        call_status=call_status, failure_detail=failure_detail,
    )
    return event_store.publish(event)


def test_cross_check_screening_qc_missing_record_raises_when_event_deleted(tmp_path):
    """Codex round-3 review, finding 1(b) (P1) -- exact reproduction:
    `classify_automation_origin_optional()` is a CATEGORY classification
    (success/heuristic_fallback/failure sets); deleting one of several
    all-failure events re-derives the SAME 'no_automated_output' category
    both before and after, so the origin-only `_check()` above sees no
    disagreement. Two sampled records both fail -> irr.n_attempted=2, but
    only ONE of the two events survives on disk (the other was deleted
    from disk AFTER publication, while `irr.producing_event_ids` --
    persisted at the moment of publication, Codex round-5 review, finding
    2 -- still names it, exactly modeling a post-hoc tamper of the event
    store rather than a node that never called it at all).
    `_load_verified_event_by_id()` now raises immediately when a
    `producing_event_ids` entry cannot be loaded, via a DIFFERENT (more
    specific) error message than the record-ID-set reconciliation this
    round-3 finding originally named -- both are the SAME underlying
    protection (a caller-recorded id must correspond to a REAL, still-
    persisted event), just surfaced at the earliest point that can name
    the problem precisely."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-missing-record"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    event1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", work_item_id="w1", call_status="failure")
    event2 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-2", work_item_id="w2", call_status="failure")
    event_store.event_path(event2.event_id).unlink()   # deleted AFTER publication -- irr still names it
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=2, n_valid_pairs=0, n_failed=2, sampled_record_ids=["rec-1", "rec-2"],
        producing_event_ids=[event1.event_id, event2.event_id],
        note="Every second-rater call failed; no valid pair to compute agreement over."))
    with pytest.raises(ReportingConsistencyError, match=r"screening_qc_second_rater.*could not be loaded/verified"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "no_automated_output", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
        )


def test_cross_check_screening_qc_multi_attempt_retry_does_not_false_reject(tmp_path):
    """Codex round-4 review, finding 3 (P1), reframed per round-5's own
    per-invocation design (finding 2): a rerun of `screening_qc_node`
    (e.g. an earlier, now-stale invocation whose own attempt=0 call for
    rec-1 failed) can leave OLD `screening_qc_second_rater` events on disk
    that are NOT part of THIS invocation's own `irr.producing_event_ids`.
    A raw stage-name-wide event scan would wrongly see THREE events for
    only two sampled records; anchoring on `producing_event_ids` (exactly
    2 entries, one per THIS invocation's own sampled record) ignores the
    stale attempt=0 survivor entirely -- must NOT raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-multi-attempt"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    # Codex round-9 review, finding 2 (P1): the full-identity check requires
    # a genuinely correct work_item_id per record_id -- `attempt` (not a
    # hand-picked placeholder work_item_id) is what real retries under the
    # SAME logical work item vary, matching reserved_attempt()'s own
    # real semantics.
    _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="failure", cfg=cfg, attempt=0)
    event_rec1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="success", cfg=cfg, attempt=1)
    event_rec2 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-2", call_status="success", cfg=cfg)
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=2, n_valid_pairs=2, n_failed=0, sampled_record_ids=["rec-1", "rec-2"],
        producing_event_ids=[event_rec1.event_id, event_rec2.event_id]))
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    rec1.second_rater_decision = "include"
    rec2.second_rater_decision = "include"
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
        search_strategy_producing_event_ids=[ss_event_id],
        screened_records_full=[rec1, rec2], screening_records=[rec1, rec2],
        current_screening_batches=[_qc_current_screening_batch([rec1, rec2], review_run_id=review_run_id)],
    )  # must not raise -- THREE events on disk, but producing_event_ids names only the two THIS invocation made


def test_cross_check_screening_qc_record_id_set_mismatch_raises(tmp_path):
    """Codex round-3 review, finding 1(b) (P1): "QC calls bound to sampled
    records" -- an event count that happens to match n_attempted, but for
    the WRONG record_ids relative to irr.sampled_record_ids, must still
    raise. Publishes an event for 'rec-wrong' instead of the one genuinely
    sampled record 'rec-1'."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-recordid-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    wrong_event = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-wrong", work_item_id="w1", call_status="success")
    # n_valid_pairs=1/n_failed=0 matches wrong_event's own call_status="success"
    # exactly (Codex round-6 review, finding 2's count-reconciliation check
    # must NOT fire here) -- isolates this fixture to the record-ID-set
    # mismatch alone (wrong_event was called for "rec-wrong", not the
    # genuinely sampled "rec-1").
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[wrong_event.event_id]))
    with pytest.raises(ReportingConsistencyError, match=r"screening_qc_second_rater: the record_id set actually called"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
        )


def test_cross_check_screening_qc_count_and_record_id_reconciliation_passes(tmp_path):
    """A genuine, untampered QC run: two sampled records, each with
    exactly one real event, record_id sets and counts all agree."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-reconciled-ok"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    event1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="success", cfg=cfg)
    event2 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-2", call_status="failure", cfg=cfg)
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=2, n_valid_pairs=1, n_failed=1, sampled_record_ids=["rec-1", "rec-2"],
        producing_event_ids=[event1.event_id, event2.event_id]))
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    rec1.second_rater_decision = "include"
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
        search_strategy_producing_event_ids=[ss_event_id],
        screened_records_full=[rec1, rec2], screening_records=[rec1, rec2],
        current_screening_batches=[_qc_current_screening_batch([rec1, rec2], review_run_id=review_run_id)],
    )  # must not raise -- classify_automation_origin_optional excludes the failure, leaving {"success"} -> "ai"


def test_cross_check_screening_qc_sample_size_shrunk_consistently_raises(tmp_path):
    """Codex round-11 review, finding 1 (P1) -- exact reproduction: an
    all-success TWO-record eligible pool, but one event deleted and
    n_attempted/n_valid_pairs/sampled_record_ids/producing_event_ids all
    consistently shrunk from 2 to 1 together. Every pre-round-11 check
    (event existence, call_status/count reconciliation, origin, record-ID-
    set membership, per-event identity/output binding) passes, since all
    four fields agree with each other and with the ONE surviving event --
    none of them ever compared against the deterministic sample size
    min(calibration_sample_size, len(eligible_records)) the real producer
    actually used. Must now raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-shrunk-sample"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    # The REAL eligible pool this run actually had: two screened records.
    event1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="success", cfg=cfg)
    _qc_event(event_store, review_run_id=review_run_id, record_id="rec-2", call_status="success", cfg=cfg)  # deleted below
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    rec1.second_rater_decision = "include"
    rec2.second_rater_decision = "include"
    # Consistently shrink to 1: state, producing_event_ids, and the event
    # itself all agree with each other -- only the REAL eligible population
    # (2 records) reveals the tamper.
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[event1.event_id]))
    with pytest.raises(ReportingConsistencyError, match=r"deterministic sample size min\(calibration_sample_size=50, len\(eligible_records\)=2\) .* is 2"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
            # screened_records_full/screening_records still show BOTH real
            # records -- only irr's own claimed sample was shrunk.
            screened_records_full=[rec1, rec2], screening_records=[rec1, rec2],
            current_screening_batches=[_qc_current_screening_batch([rec1, rec2], review_run_id=review_run_id)],
        )


def test_cross_check_screening_qc_tampered_screening_decision_on_unsampled_record_no_longer_matters(tmp_path):
    """Codex round-12 review, finding 1 (P1) -- exact reproduction of the
    scenario the review named: TWO eligible records, but only 'rec-1' is
    sampled by QC. An attacker sets the UNSAMPLED 'rec-2's own
    `screening_decision` to `None` on `screened_records_full` -- before
    round-12's fix, `compute_qc_eligible_records()` read THIS field
    directly, so the eligible pool would shrink to 1, matching a
    correspondingly-shrunk `n_attempted=1` and making the tamper invisible.
    After the fix, eligibility is derived from the FROZEN, verified
    `current_screening_batches` projection instead -- tampering
    `screened_records_full.screening_decision` for ANY record (sampled or
    not) no longer has ANY effect on the eligible-pool computation, so the
    genuinely correct `n_attempted=1` (rec-1 alone was sampled from a REAL
    2-record eligible pool) still needs `calibration_sample_size` >= 2 to
    legitimately sample only 1 -- use `calibration_sample_size=1` so this
    fixture isolates the "tamper has no effect" claim, not accidentally
    re-triggering the deterministic-sample-size check for an unrelated
    reason."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    cfg.screening_qc.calibration_sample_size = 1
    review_run_id = "run-qc-tamper-unsampled-decision"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    event1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="success", cfg=cfg)
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    rec1.second_rater_decision = "include"
    # The tamper Codex reproduced: rec-2 (unsampled, but genuinely eligible
    # -- current_screening_batches below still shows it with a real
    # screening_decision) has its OWN `screened_records_full` entry's
    # `screening_decision` blanked out.
    rec2.screening_decision = None
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[event1.event_id]))
    # current_screening_batches is built from FRESH _qc_record() calls (both
    # genuinely "include"), independent of the tampered rec2 above -- this
    # is the FROZEN projection that now anchors eligibility.
    batches = [_qc_current_screening_batch([_qc_record("rec-1"), _qc_record("rec-2")], review_run_id=review_run_id)]
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
        search_strategy_producing_event_ids=[ss_event_id],
        screened_records_full=[rec1, rec2], screening_records=[_qc_record("rec-1"), _qc_record("rec-2")],
        current_screening_batches=batches,
    )  # must NOT raise -- the tamper on screened_records_full.screening_decision is inert


def test_cross_check_screening_qc_current_screening_batches_shrunk_relative_to_screening_records_raises(tmp_path):
    """Codex round-12 review, finding 1 (P1): defense-in-depth for the
    (structurally-should-be-impossible-in-production) case where
    `current_screening_batches`' own combined record_id set disagrees with
    `screening_records` -- `_load_current_screening_batches()` itself
    guarantees this can't happen for a real run, but this test constructs
    it directly to confirm the guard actually fires rather than silently
    trusting an incomplete batch-derived decision mapping."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-batches-shrunk"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=0, n_valid_pairs=0, n_failed=0, sampled_record_ids=[], producing_event_ids=[]))
    with pytest.raises(ReportingConsistencyError, match="current_screening_batches' own combined record_id set"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": None, "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
            screened_records_full=[rec1, rec2], screening_records=[rec1, rec2],
            # Only rec-1's batch result exists -- missing rec-2 entirely.
            current_screening_batches=[_qc_current_screening_batch([rec1], review_run_id=review_run_id)],
        )


def test_cross_check_screening_qc_screened_records_full_missing_record_raises(tmp_path):
    """Codex round-11 review, finding 1 (P1): `screened_records_full`
    missing a record present in the durably-captured `screening_records`
    (population shrunk relative to the real corpus) must raise, even
    before the deterministic sample-size formula is even evaluated."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-population-missing"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    event1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="success", cfg=cfg)
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    rec1.second_rater_decision = "include"
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[event1.event_id]))
    with pytest.raises(ReportingConsistencyError, match=r"does not exactly match the durably-captured screening corpus"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
            screened_records_full=[rec1],  # missing rec2
            screening_records=[rec1, rec2],  # the REAL corpus has both
        )


def test_cross_check_screening_qc_screened_records_full_extra_record_raises(tmp_path):
    """Codex round-11 review, finding 1 (P1): `screened_records_full`
    containing an EXTRA record not present in `screening_records` (padded
    relative to the real corpus) must also raise."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-population-extra"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    event1 = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="success", cfg=cfg)
    rec1, rec2 = _qc_record("rec-1"), _qc_record("rec-2")
    rec1.second_rater_decision = "include"
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[event1.event_id]))
    with pytest.raises(ReportingConsistencyError, match=r"does not exactly match the durably-captured screening corpus"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
            screened_records_full=[rec1, rec2],  # rec2 is EXTRA, not in the real corpus
            screening_records=[rec1],  # the REAL corpus has only rec1
        )


def test_cross_check_screening_qc_call_status_mismatch_with_n_failed_raises(tmp_path):
    """Codex round-6 review, finding 2 (P1) -- exact reproduction: a state
    claiming n_attempted=1, n_valid_pairs=1, n_failed=0 must raise when its
    sole bound event's own call_status is actually 'failure'. Previously
    only internal arithmetic (_counts_conserve) was checked; the durable
    event's own call_status was never reconciled against these counts at
    all."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-callstatus-mismatch"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    failed_event = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", work_item_id="w1", call_status="failure")
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[failed_event.event_id]))
    with pytest.raises(ReportingConsistencyError, match=r"call_status='failure', but irr\.n_failed=0"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "no_automated_output", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id],
        )


def test_cross_check_screening_qc_origin_uses_full_population_not_just_current_invocation(tmp_path):
    """Codex round-6 review, finding 1 (P1) -- exact reproduction: an
    EARLIER invocation's own screening_qc_second_rater call succeeded, the
    CURRENT invocation's own (single, distinct) sampled-record call failed.
    screening_qc_node itself classifies origin over EVERY historical event
    for this stage (`event_store.load_all()` filtered by stage), so the
    producer's own state correctly says 'ai' (the earlier success still
    counts) -- a current-invocation-only re-derivation would wrongly say
    'no_automated_output' instead, rejecting a genuinely valid rerun."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-rerun-origin"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    # Earlier invocation: succeeded for a DIFFERENT sampled record -- not
    # named in THIS invocation's own producing_event_ids, so exempt from
    # the full-identity check (only the origin-population scan sees it);
    # a placeholder identity is fine here.
    _qc_event(event_store, review_run_id=review_run_id, record_id="rec-earlier", work_item_id="w-earlier", call_status="success")
    # Current invocation: its own single sampled record's call failed --
    # named in producing_event_ids, so it DOES need a genuine identity
    # (Codex round-9 review, finding 2).
    current_event = _qc_event(event_store, review_run_id=review_run_id, record_id="rec-1", call_status="failure", cfg=cfg)
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=0, n_failed=1, sampled_record_ids=["rec-1"],
        producing_event_ids=[current_event.event_id]))
    _cross_check(
        cfg=cfg,
        # "ai", not "no_automated_output" -- matches classify_automation_origin_optional()
        # over the FULL two-event population (one success, one failure -> output_producing
        # is non-empty -> statuses=={"success"} -> "ai"), exactly what screening_qc_node
        # itself would have derived and stored.
        oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
        screened_records_full=[_qc_record("rec-1")], screening_records=[_qc_record("rec-1")],
        automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
        ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
        search_strategy_producing_event_ids=[ss_event_id],
        current_screening_batches=[_qc_current_screening_batch([_qc_record("rec-1")], review_run_id=review_run_id)],
    )  # must NOT raise


def test_cross_check_screening_qc_disabled_with_surviving_historical_events_does_not_raise(tmp_path):
    """Codex round-6 review, finding 1 (P1), second half: round-5's own
    'screening_qc_state is None -> zero events may exist anywhere' check is
    WRONG given the event store's intentionally append-only design -- QC
    can legitimately have run in an EARLIER invocation of this same
    review_run_id, then be disabled in a LATER one. Real historical events
    surviving on disk must not be treated as tampering evidence."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-qc-disabled-after-earlier-run"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    _qc_event(event_store, review_run_id=review_run_id, record_id="rec-earlier", work_item_id="w-earlier", call_status="success")
    _cross_check(
        cfg=cfg,
        oversight_mode_state=dict(_BASELINE_SEARCH_STRATEGY_OVERSIGHT),
        automation_origin_state=dict(_BASELINE_SEARCH_STRATEGY_STATE),
        ledger_store=ledger_store, event_store=event_store,
        screening_qc_state=None,   # QC disabled THIS invocation
        search_strategy_producing_event_ids=[ss_event_id],
    )  # must not raise, even though a real historical QC event survives on disk


def _qc_setup(tmp_path, review_run_id: str):
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ss_event_id = _baseline_search_strategy(event_store, review_run_id=review_run_id, cfg=cfg)
    return cfg, ledger_store, event_store, ss_event_id


def test_cross_check_screening_qc_wrong_criteria_raises(tmp_path):
    """Codex round-9 review, finding 2 (P1) -- exact reproduction: an event
    for the CORRECT record_id, but produced against a DIFFERENT cfg's
    criteria (research_questions), must raise -- record-id-set membership
    alone cannot catch a stale/foreign criteria hash."""
    cfg, ledger_store, event_store, ss_event_id = _qc_setup(tmp_path, "run-qc-wrong-criteria")
    other_cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), research_questions=["an unrelated question"])
    record = _qc_record("rec-1")
    wrong_identity = compute_qc_second_rater_call_identity(cfg=other_cfg, record=record, model="gpt-5.4-mini")
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id="run-qc-wrong-criteria", stage="screening_qc_second_rater",
        work_item_id=wrong_identity["work_item_id"], attempt=0, tool_kind=wrong_identity["tool_kind"],
        model=wrong_identity["model"], provider=wrong_identity["provider"], component_versions={},
        access_info="http://x", task_purpose=wrong_identity["task_purpose"],
        model_config_id=wrong_identity["model_config_id"],
        prompt_template_version=wrong_identity["prompt_template_version"],
        prompt_template_text=wrong_identity["prompt_template_text"],
        prompt_parameters=wrong_identity["prompt_parameters"], rendered_prompt_hash=wrong_identity["rendered_prompt_hash"],
        input_artifact_refs=wrong_identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex({'decision': 'include'})}"], call_status="success",
    ))
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[wrong_event.event_id]))
    record.second_rater_decision = "include"
    with pytest.raises(ReportingConsistencyError, match=r"screening_qc_second_rater.*does not match the CURRENT"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id], screened_records_full=[record],
            # Codex round-10 review, finding 3 (P1): reporting.py now
            # reconciles screened_records_full's own content against the
            # durably-captured screening corpus (snapshot.screening_records)
            # BEFORE trusting it for call-identity reconstruction -- the SAME
            # record here satisfies that check trivially, isolating this
            # fixture to the substitution/tamper it's actually testing.
            screening_records=[record],
            current_screening_batches=[_qc_current_screening_batch([record], review_run_id="run-qc-wrong-criteria")],
        )


def test_cross_check_screening_qc_wrong_model_raises(tmp_path):
    """Codex round-9 review, finding 2 (P1) -- exact reproduction: an event
    genuinely produced for a DIFFERENT model than the current config's own
    `screening_qc.second_rater_model`/`screening.llm_model` must raise."""
    cfg, ledger_store, event_store, ss_event_id = _qc_setup(tmp_path, "run-qc-wrong-model")
    record = _qc_record("rec-1")
    wrong_identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model="an-unrelated-model")
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id="run-qc-wrong-model", stage="screening_qc_second_rater",
        work_item_id=wrong_identity["work_item_id"], attempt=0, tool_kind=wrong_identity["tool_kind"],
        model=wrong_identity["model"], provider=wrong_identity["provider"], component_versions={},
        access_info="http://x", task_purpose=wrong_identity["task_purpose"],
        model_config_id=wrong_identity["model_config_id"],
        prompt_template_version=wrong_identity["prompt_template_version"],
        prompt_template_text=wrong_identity["prompt_template_text"],
        prompt_parameters=wrong_identity["prompt_parameters"], rendered_prompt_hash=wrong_identity["rendered_prompt_hash"],
        input_artifact_refs=wrong_identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex({'decision': 'include'})}"], call_status="success",
    ))
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[wrong_event.event_id]))
    record.second_rater_decision = "include"
    with pytest.raises(ReportingConsistencyError, match=r"screening_qc_second_rater.*does not match the CURRENT"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id], screened_records_full=[record],
            # Codex round-10 review, finding 3 (P1): reporting.py now
            # reconciles screened_records_full's own content against the
            # durably-captured screening corpus (snapshot.screening_records)
            # BEFORE trusting it for call-identity reconstruction -- the SAME
            # record here satisfies that check trivially, isolating this
            # fixture to the substitution/tamper it's actually testing.
            screening_records=[record],
            current_screening_batches=[_qc_current_screening_batch([record], review_run_id="run-qc-wrong-model")],
        )


def test_cross_check_screening_qc_wrong_model_config_id_raises(tmp_path):
    """Codex round-9 review, finding 2 (P1) -- exact reproduction: every
    OTHER field is genuinely correct (real work_item_id/rendered_prompt_
    hash/prompt_parameters/input_artifact_refs/model/provider/task_purpose/
    prompt_template_*), but `model_config_id` alone is wrong -- must still
    raise. `_check_call_identity_fields()` deliberately excludes tool_kind/
    model_config_id (each call site's own dedicated check catches those),
    so this isolates that QC's OWN dedicated model_config_id check exists
    and fires."""
    cfg, ledger_store, event_store, ss_event_id = _qc_setup(tmp_path, "run-qc-wrong-model-config-id")
    record = _qc_record("rec-1")
    identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model="gpt-5.4-mini")
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id="run-qc-wrong-model-config-id", stage="screening_qc_second_rater",
        work_item_id=identity["work_item_id"], attempt=0, tool_kind=identity["tool_kind"],
        model=identity["model"], provider=identity["provider"], component_versions={},
        access_info="http://x", task_purpose=identity["task_purpose"],
        model_config_id="wrong_model_config_id",   # the ONE deliberately wrong field
        prompt_template_version=identity["prompt_template_version"],
        prompt_template_text=identity["prompt_template_text"],
        prompt_parameters=identity["prompt_parameters"], rendered_prompt_hash=identity["rendered_prompt_hash"],
        input_artifact_refs=identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex({'decision': 'include'})}"], call_status="success",
    ))
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[wrong_event.event_id]))
    record.second_rater_decision = "include"
    with pytest.raises(ReportingConsistencyError, match=r"model_config_id='wrong_model_config_id'"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id], screened_records_full=[record],
            # Codex round-10 review, finding 3 (P1): reporting.py now
            # reconciles screened_records_full's own content against the
            # durably-captured screening corpus (snapshot.screening_records)
            # BEFORE trusting it for call-identity reconstruction -- the SAME
            # record here satisfies that check trivially, isolating this
            # fixture to the substitution/tamper it's actually testing.
            screening_records=[record],
            current_screening_batches=[_qc_current_screening_batch([record], review_run_id="run-qc-wrong-model-config-id")],
        )


def test_cross_check_screening_qc_arbitrary_work_item_id_raises(tmp_path):
    """Codex round-9 review, finding 2 (P1) -- exact reproduction: every
    OTHER field genuinely correct, but `work_item_id` alone is an arbitrary
    string unrelated to the real canonical payload -- must raise (an
    internally self-consistent `AIUseEvent` -- `event_id` correctly
    derived from ITS OWN `work_item_id` -- is not the same as a GENUINE
    call identity)."""
    cfg, ledger_store, event_store, ss_event_id = _qc_setup(tmp_path, "run-qc-arbitrary-work-item-id")
    record = _qc_record("rec-1")
    identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model="gpt-5.4-mini")
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id="run-qc-arbitrary-work-item-id", stage="screening_qc_second_rater",
        work_item_id="screening_qc_second_rater:arbitrary-unrelated-hash",   # the ONE deliberately wrong field
        attempt=0, tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
        component_versions={}, access_info="http://x", task_purpose=identity["task_purpose"],
        model_config_id=identity["model_config_id"],
        prompt_template_version=identity["prompt_template_version"],
        prompt_template_text=identity["prompt_template_text"],
        prompt_parameters=identity["prompt_parameters"], rendered_prompt_hash=identity["rendered_prompt_hash"],
        input_artifact_refs=identity["input_artifact_refs"],
        output_artifact_refs=[f"sha256:{sha256_hex({'decision': 'include'})}"], call_status="success",
    ))
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[wrong_event.event_id]))
    record.second_rater_decision = "include"
    with pytest.raises(ReportingConsistencyError, match=r"work_item_id 'screening_qc_second_rater:arbitrary-unrelated-hash' does not match"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id], screened_records_full=[record],
            # Codex round-10 review, finding 3 (P1): reporting.py now
            # reconciles screened_records_full's own content against the
            # durably-captured screening corpus (snapshot.screening_records)
            # BEFORE trusting it for call-identity reconstruction -- the SAME
            # record here satisfies that check trivially, isolating this
            # fixture to the substitution/tamper it's actually testing.
            screening_records=[record],
            current_screening_batches=[_qc_current_screening_batch([record], review_run_id="run-qc-arbitrary-work-item-id")],
        )


def test_cross_check_screening_qc_unrelated_decision_output_raises(tmp_path):
    """Codex round-9 review, finding 2 (P1) -- exact reproduction: every
    identity field genuinely correct, but the success event's own output
    hash corresponds to a DIFFERENT decision ('exclude') than the CURRENT
    Record.second_rater_decision ('include') -- must raise."""
    cfg, ledger_store, event_store, ss_event_id = _qc_setup(tmp_path, "run-qc-unrelated-decision")
    record = _qc_record("rec-1")
    identity = compute_qc_second_rater_call_identity(cfg=cfg, record=record, model="gpt-5.4-mini")
    wrong_event = event_store.publish(build_ai_use_event(
        review_run_id="run-qc-unrelated-decision", stage="screening_qc_second_rater",
        work_item_id=identity["work_item_id"], attempt=0, tool_kind=identity["tool_kind"],
        model=identity["model"], provider=identity["provider"], component_versions={},
        access_info="http://x", task_purpose=identity["task_purpose"], model_config_id=identity["model_config_id"],
        prompt_template_version=identity["prompt_template_version"],
        prompt_template_text=identity["prompt_template_text"],
        prompt_parameters=identity["prompt_parameters"], rendered_prompt_hash=identity["rendered_prompt_hash"],
        input_artifact_refs=identity["input_artifact_refs"],
        # Deliberately unrelated to the record's own second_rater_decision below.
        output_artifact_refs=[f"sha256:{sha256_hex({'decision': 'exclude'})}"], call_status="success",
    ))
    qc_state = ReportingScreeningQCState(irr=ReportingIRRState(
        n_attempted=1, n_valid_pairs=1, n_failed=0, sampled_record_ids=["rec-1"],
        producing_event_ids=[wrong_event.event_id]))
    record.second_rater_decision = "include"
    with pytest.raises(ReportingConsistencyError, match=r"does not match .* hash of the CURRENT Record\.second_rater_decision"):
        _cross_check(
            cfg=cfg,
            oversight_mode_state={"screening_qc_second_rater": "automation_unreviewed", "title_abstract_screening": "automation_unreviewed", **_BASELINE_SEARCH_STRATEGY_OVERSIGHT},
            automation_origin_state={"screening_qc_second_rater": "ai", "title_abstract_screening": "ai", **_BASELINE_SEARCH_STRATEGY_STATE},
            ledger_store=ledger_store, event_store=event_store, screening_qc_state=qc_state,
            search_strategy_producing_event_ids=[ss_event_id], screened_records_full=[record],
            # Codex round-10 review, finding 3 (P1): reporting.py now
            # reconciles screened_records_full's own content against the
            # durably-captured screening corpus (snapshot.screening_records)
            # BEFORE trusting it for call-identity reconstruction -- the SAME
            # record here satisfies that check trivially, isolating this
            # fixture to the substitution/tamper it's actually testing.
            screening_records=[record],
            current_screening_batches=[_qc_current_screening_batch([record], review_run_id="run-qc-unrelated-decision")],
        )


def test_cross_check_search_strategy_origin_uses_full_population_not_just_current_invocation(tmp_path):
    """Codex round-6 review, finding 1 (P1) -- the search_strategy sibling
    of the QC scenario above: an earlier invocation's own db_translation
    call succeeded, the CURRENT invocation's own db_translation call fell
    back. search_strategy_builder_node itself classifies origin over EVERY
    historical event for these two stage names -- the producer's own state
    correctly says 'mixed' (one success, one heuristic_fallback); a
    current-invocation-only re-derivation would wrongly say
    'rule_based_automation' instead."""
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    review_run_id = "run-ss-rerun-origin"
    output_dir = cfg.ensure_output_dir()
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    _search_strategy_event(event_store, review_run_id=review_run_id, stage="search_strategy_db_translation",
                            work_item_id="w-earlier", call_status="success")
    # Codex round-8 review, finding 2 (P1): the CURRENT invocation's own
    # event -- the one actually named in `producing_event_ids` -- must
    # carry the REAL immutable call identity reporting.py now recomputes
    # from live config (`_cross_check`'s own default
    # `search_strategy_expanded_query=""` matches this bare `cfg.search.
    # query`'s own default ""), even though its `call_status` is the
    # fallback branch. The EARLIER event above is not referenced in
    # `producing_event_ids`, so it's exempt from this binding -- it only
    # needs to be a real, well-formed event of the right stage for the
    # origin-population scan to see.
    current_dt_identity = compute_db_translation_call_identity(
        query=cfg.search.query, databases=cfg.search.databases, model=cfg.search.strategy_builder.llm_model)
    current_dt = build_ai_use_event(
        review_run_id=review_run_id, stage="search_strategy_db_translation",
        work_item_id=current_dt_identity["work_item_id"], attempt=0,
        tool_kind=current_dt_identity["tool_kind"], model=current_dt_identity["model"],
        provider=current_dt_identity["provider"],
        component_versions={}, access_info="http://x",
        task_purpose=current_dt_identity["task_purpose"], model_config_id=current_dt_identity["model_config_id"],
        prompt_template_version=current_dt_identity["prompt_template_version"],
        prompt_template_text=current_dt_identity["prompt_template_text"],
        rendered_prompt_hash=current_dt_identity["rendered_prompt_hash"],
        prompt_parameters=current_dt_identity["prompt_parameters"],
        input_artifact_refs=current_dt_identity["input_artifact_refs"],
        # Codex round-12 review, finding 2 (P2): a genuine "heuristic_fallback"
        # db_queries has the SAME value (the current invocation's own
        # expanded_query, "" here -- `_cross_check`'s own default) for
        # EVERY database in the expected key set -- not an empty dict.
        output_artifact_refs=[f"sha256:{sha256_hex(_default_db_queries(cfg, query=''))}"], call_status="heuristic_fallback",
        # Codex round-10 review, finding 2 (P1): a genuine "heuristic_fallback"
        # event always carries a non-None failure_detail (the real error that
        # triggered the fallback) -- this test builds the event directly
        # (not via `_search_strategy_event()`'s own auto-default), so it must
        # be supplied explicitly here.
        failure_detail="simulated failure",
    )
    current_dt = event_store.publish(current_dt)
    _cross_check(
        cfg=cfg,
        oversight_mode_state={"search_strategy": "automation_unreviewed"},
        automation_origin_state={"search_strategy": "mixed"},
        ledger_store=ledger_store, event_store=event_store,
        search_strategy_producing_event_ids=[current_dt.event_id],
        # Codex round-12 review, finding 2 (P2): must match the CURRENT
        # event's own genuine fallback shape (every db_queries value equal
        # to search_strategy_expanded_query="", not `_cross_check`'s own
        # unrelated default).
        search_strategy_db_queries=_default_db_queries(cfg, query=""),
    )  # must NOT raise -- "mixed" reflects the FULL two-event population, not just the current one


# ---------------------------------------------------------------------------
# 3. Full-pipeline smoke test per review type
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.model = "fake-model"


class _FailingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("no network in tests -- forces the heuristic/rule-based fallback path")


class _FailingChat:
    completions = _FailingCompletions()


class _FailingOpenAI:
    def __init__(self, *a, **kw) -> None:
        self.chat = _FailingChat()
        self.base_url = "https://fake.example/v1"


def _make_dispatching_openai_class(responder: Callable[[Dict[str, Any]], str]):
    class _FakeCompletions:
        def create(self, **kwargs):
            return _FakeResponse(responder(kwargs))

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = _FakeChat()
            self.base_url = "https://fake.example/v1"

    return _FakeClient


def _extraction_schema() -> ExtractionSchema:
    return ExtractionSchema(schema_id="s", version="1", fields=[
        ExtractionFieldSpec(field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT,
                             cardinality="single", provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1")),
    ])


def _model_config() -> OpenAIExtractionConfig:
    return OpenAIExtractionConfig(model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
                                   max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra")


def _telemetry() -> BatchCallTelemetry:
    return BatchCallTelemetry(provider_model="gpt-5.6-terra", prompt_tokens=1, completion_tokens=1, total_tokens=2)


_CORPUS = [
    {"id": "rec-a", "source": "s", "source_id": "rec-a", "title": "Exercise improves outcome in adults",
     "abstract": "A study on exercise improving outcome.", "doi": "10.1/a", "year": 2020},
    {"id": "rec-b", "source": "s", "source_id": "rec-b", "title": "Exercise improves outcome in children",
     "abstract": "Another study on exercise improving outcome.", "doi": "10.1/b", "year": 2021},
]


def _run_full_pipeline_to_reporting(
    monkeypatch, tmp_path, *, review_type: str, pre_reporting_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Hand-orchestrates a small, 2-record run through dedup_node,
    screening_node (heuristic fallback -- OpenAI monkeypatched to fail),
    a HAND-CONSTRUCTED full-text-eligibility stage (fulltext_node itself
    has its own dedicated test suite; this smoke test's own concern is
    reporting.py's consumption contract, not FT_SCREENING's internals),
    a REAL extraction_node (parse/provider faked, matching tests/nodes/
    test_extraction_node.py's own convention), quality_appraisal_node for
    rapid/slr, a REAL synthesis_node (map/refine dispatching fake OpenAI
    client), and finally reporting_node."""
    qa_needed = review_type in ("rapid", "slr")
    cfg = ReviewConfig(
        title="Reporting smoke test", review_type=review_type, output_dir=str(tmp_path / "out"),
        research_questions=["Does exercise improve outcome"],
        inclusion_criteria=["improve"], exclusion_criteria=[],
        screening=ScreeningConfig(uncertain_threshold=0.0),
        extraction_schema=_extraction_schema().model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(_model_config()),
        quality_appraisal_model_config=(dataclasses.asdict(_model_config()) if qa_needed else None),
        quality_appraisal=QualityAppraisalConfig(checklist_id="teaching_appraisal_v1"),
        synthesis=SynthesisConfig(run_bertopic=False, generate_evidence_map=(review_type == "scoping"),
                                   generate_claim_backed_results=True, generate_legacy_imrad_narrative=False,
                                   refine_batch_size=5),
        data_governance_declaration="No personal data; public bibliographic metadata only.",
    )
    state = initial_state(cfg.to_dict())
    state["raw_records"] = [dict(r) for r in _CORPUS]
    # search_node never runs in this hand-orchestrated pipeline -- set
    # PRISMACounts.identified directly, matching what it would have set,
    # so compute_selection_splits()'s own INV-R6 total (total_identified -
    # after_dedup) is a real, non-negative expected dedup-removed count.
    state["prisma_counts"] = PRISMACounts(identified={"test": len(_CORPUS)}).model_dump()
    # Outside a real LangGraph invocation (no active thread_id), state must
    # already carry review_run_id explicitly -- matches tests/nodes/
    # test_synthesis_node.py's own convention.
    state["review_run_id"] = "reporting-smoke-test-run"

    # --- hand-constructed search_strategy stage (Codex round-3 review,
    # finding 1(a): search_strategy_builder_node -- the always-runs-first
    # node of every real graph -- is deliberately never invoked by this
    # hand-orchestrated pipeline, same reasoning as the hand-constructed
    # full-text-eligibility stage below: this smoke test's own concern is
    # reporting.py's consumption contract, not search_strategy_builder's
    # internals. reporting_node now unconditionally requires at least one
    # durable search_strategy_query_expansion/_db_translation event to
    # exist, so one is published here, matching what a real invocation
    # would always have produced). ---
    review_run_id_for_ss = state["review_run_id"]
    ss_event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id_for_ss)
    # Codex round-8 review, finding 2 (P1): reporting.py now binds the
    # COMPLETE immutable call identity (work_item_id/rendered_prompt_hash/
    # prompt_parameters/input_artifact_refs), reconstructed via the SAME
    # shared `compute_db_translation_call_identity()` the real producer
    # uses -- a hand-picked work_item_id/rendered_prompt_hash can no longer
    # pass. `query=cfg.search.query` ("" here, unset above) matches
    # `state["search_strategy"]["expanded_query"]` being absent/"" below.
    ss_identity = compute_db_translation_call_identity(
        query=cfg.search.query, databases=cfg.search.databases, model=cfg.search.strategy_builder.llm_model)
    ss_event = ss_event_store.publish(build_ai_use_event(
        review_run_id=review_run_id_for_ss, stage="search_strategy_db_translation",
        # Codex round-7 review, finding 2 (P1): reporting.py now validates
        # the COMPLETE per-call-site contract -- must match search_strategy_
        # builder.py's own real `_generate_db_queries()` literals exactly.
        work_item_id=ss_identity["work_item_id"], attempt=0, tool_kind=ss_identity["tool_kind"],
        model=ss_identity["model"],
        provider=ss_identity["provider"], component_versions={}, access_info="http://x",
        task_purpose=ss_identity["task_purpose"],
        model_config_id=ss_identity["model_config_id"],
        prompt_template_version=ss_identity["prompt_template_version"], prompt_template_text=ss_identity["prompt_template_text"],
        prompt_parameters=ss_identity["prompt_parameters"], rendered_prompt_hash=ss_identity["rendered_prompt_hash"],
        input_artifact_refs=ss_identity["input_artifact_refs"],
        # Codex round-6 review, finding 4 stretch goal / round-12 review,
        # finding 2 (P2): hashes a REAL, shape-valid db_queries (matching
        # `state["search_strategy"]["db_queries"]" below, both built from
        # THIS `cfg` via `_default_db_queries()`) -- reporting's own
        # output-binding check requires this event's output_artifact_refs
        # to contain the hash of the CURRENT SearchStrategy.db_queries, and
        # its newer key-set/blank-value shape checks require db_queries
        # itself to be a real, `_generate_db_queries()`-producible shape.
        output_artifact_refs=[f"sha256:{sha256_hex(_default_db_queries(cfg))}"], call_status="success",
    ))
    state["oversight_mode"] = {**state["oversight_mode"], "search_strategy": "automation_unreviewed"}
    state["automation_origin"] = {**state["automation_origin"], "search_strategy": "ai"}
    # Codex round-5 review, finding 3: reporting's own cross-check now
    # verifies against THIS invocation's own `state["search_strategy"]
    # ["producing_event_ids"]` -- `cfg.search.strategy_builder.enabled`
    # defaults to False (not overridden above), so only the db_translation
    # event above is required/expected.
    state["search_strategy"] = {"producing_event_ids": [ss_event.event_id], "db_queries": _default_db_queries(cfg)}

    # --- dedup ---
    state.update(dedup_node(state))
    assert len(state["deduplicated_records"]) == 2

    # --- screening (heuristic fallback, automated mode) ---
    monkeypatch.setattr(screening_module, "OpenAI", _FailingOpenAI)
    state.update(screening_node(state))
    assert len(state["included_records"]) == 2

    # --- hand-constructed full-text-eligibility stage ---
    review_run_id = state["review_run_id"]
    ledger_store = SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    full_text_records = []
    for record in state["included_records"]:
        pdf_path = tmp_path / f"{record['id']}.pdf"
        pdf_path.write_bytes(b"fake pdf bytes")
        record = dict(record, full_text_path=str(pdf_path), full_text_status="parsed", full_text_source="test")
        full_text_records.append(record)
        source_event_id = derive_source_event_id(
            producer_artifact_id="manual-ft-screening-smoke-test", record_id=record["id"], stage="full_text_eligibility")
        emit_selection_decision(
            store=ledger_store, record_id=record["id"], stage="full_text_eligibility", decision="include",
            decision_maker="ai", source_event_id=source_event_id,
            reason="hand-constructed for reporting smoke test", producer_artifact_id="manual-ft-screening-smoke-test")
    oversight_mode = dict(state["oversight_mode"])
    automation_origin = dict(state["automation_origin"])
    oversight_mode["full_text_eligibility"] = "automation_unreviewed"
    automation_origin["full_text_eligibility"] = "ai"
    prisma = PRISMACounts(**state["prisma_counts"])
    prisma.assessed_full_text = len(full_text_records)
    prisma.excluded_full_text = 0
    prisma.fulltext_retrieval = {"parsed": len(full_text_records)}
    prisma.included = len(full_text_records)

    # Codex round-4 review, finding 4 (P1): the full_text_eligibility
    # ledger-completeness check's authoritative expected record-ID set now
    # comes from FT_SCREENING's own `item_results.jsonl` (via
    # capture_ft_native_provenance()'s "confirmed_record_ids"), not merely
    # `prisma.assessed_full_text`'s own count. This smoke test hand-
    # constructs the full-text-eligibility ledger stage (see this
    # function's own docstring) without ever running the real
    # fulltext_node/FT_SCREENING adapter, so it must also hand-construct a
    # matching, genuinely on-disk `item_results.jsonl` (plus its two
    # sibling artifact files). Codex round-5 review, finding 1: these must
    # now be genuinely valid, schema-complete FT_SCREENING artifacts
    # (`ScreeningRunResult`/`ScreeningItemResult`), since
    # `capture_ft_native_provenance()` fully parses and cross-verifies
    # both -- built via the same `_build_valid_ft_screening_run_files()`
    # helper the FT-provenance unit tests above use.
    ft_run_dir = cfg.ensure_output_dir() / "ft_screening_runs" / "smoke-run"
    ft_run_dir.mkdir(parents=True, exist_ok=True)
    smoke_ft_run_id = "smoke-run"
    smoke_ft_fingerprint = _valid_sha256("fp-smoke")
    ft_artifact_paths = _build_valid_ft_screening_run_files(
        ft_run_dir, run_id=smoke_ft_run_id, execution_fingerprint=smoke_ft_fingerprint,
        record_ids=[r["id"] for r in full_text_records])

    state.update({
        "full_text_records": full_text_records, "full_text_review_queue": [],
        "oversight_mode": oversight_mode, "automation_origin": automation_origin,
        "prisma_counts": prisma.model_dump(),
        "ft_screening_run_id": smoke_ft_run_id, "ft_screening_execution_fingerprint": smoke_ft_fingerprint,
        "ft_screening_artifact_paths": ft_artifact_paths,
    })

    # --- extraction (real engine, parse/provider faked) ---
    def fake_parse(pdf_path, *a, **k):
        record_id = Path(pdf_path).stem
        chunk = ParsedChunk(chunk_id=f"{record_id}-chunk-0000", order_index=0, section="Methods",
                             page_min=1, page_max=1, text=f"[study_design] RCT for {record_id}.")
        return ParsedDocument(document_sha256="a" * 64, parser_name="docling", parser_version="2.61.2",
                               parser_options=DoclingParserOptions(), chunks=(chunk,))

    monkeypatch.setattr(engine_module, "parse_pdf", fake_parse)

    def fake_extraction_call(*, schema, batch_chunks, config):
        chunk = batch_chunks[0]
        return RawBatchResponse(fields={
            "study_design": RawFieldResponse(outcome="found", findings=[
                RawFinding(value="RCT", confidence="high", rationale="x",
                           evidence=[RawEvidenceItem(chunk_id=chunk.chunk_id, quotation=chunk.text)]),
            ]),
        }), _telemetry()

    monkeypatch.setattr(extraction_module, "run_openai_batch_extraction", fake_extraction_call)
    state.update(extraction_node(state))
    assert state["extraction_review_queue"] == []

    # --- quality appraisal (rapid/slr only) ---
    if qa_needed:
        def fake_qa_parse(pdf_path, *a, **k):
            record_id = Path(pdf_path).stem
            chunk = ParsedChunk(chunk_id=f"{record_id}-qa-chunk-0000", order_index=0, section="Methods",
                                 page_min=1, page_max=1, text="The research question and methods are clearly reported.")
            return ParsedDocument(document_sha256="b" * 64, parser_name="docling", parser_version="2.61.2",
                                   parser_options=DoclingParserOptions(), chunks=(chunk,))

        monkeypatch.setattr(qa_module, "parse_pdf", fake_qa_parse)

        def fake_teaching_batch(*, record_title, batch_chunks, config):
            chunk = batch_chunks[0]
            return RawTeachingBatchResponse(items={
                item["item_id"]: RawTeachingItemBatchResponse(
                    outcome="yes", rationale="clearly and explicitly reported",
                    evidence=[RawAppraisalEvidenceItem(chunk_id=chunk.chunk_id, quotation=chunk.text)])
                for item in TEACHING_APPRAISAL_ITEMS
            })

        monkeypatch.setattr(qa_module, "run_openai_teaching_batch", fake_teaching_batch)
        state.update(quality_appraisal_node(state))
        assert state["quality_appraisal_review_queue"] == []

    # --- synthesis (map/refine dispatching fake client) ---
    import re as re_module

    def synthesis_responder(kw: Dict[str, Any]) -> str:
        system = kw["messages"][0]["content"]
        if "assisting a systematic literature review" in system:
            return json.dumps({"relevant": True, "answer": "Improves outcome via RCT evidence.",
                                "supporting_selectors": [{"artifact_kind": "extraction", "field_id": "study_design"}]})
        if "drafting the synthesis section" in system:
            user_content = kw["messages"][1]["content"]
            answer_ids = re_module.findall(r"(answer-[0-9a-f]{64})", user_content)
            return json.dumps({"claim_updates": [{"claim_index": None, "claim_text": "Exercise improves outcome across included studies.",
                                                    "final_support_answer_ids": answer_ids}]})
        raise AssertionError(f"unexpected synthesis call: {system[:80]}")

    monkeypatch.setattr(synthesis_module, "OpenAI", _make_dispatching_openai_class(synthesis_responder))
    state.update(synthesis_node(state))
    assert state["synthesis"]["claims"], "setup failed: expected at least one synthesized claim"

    # --- reporting ---
    if pre_reporting_hook is not None:
        pre_reporting_hook(state)
    output = reporting_node(state)
    state.update(output)
    return state


@pytest.mark.parametrize("review_type", ["scoping", "rapid", "slr"])
def test_full_pipeline_reporting_smoke(monkeypatch, tmp_path, review_type):
    state = _run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type=review_type)

    manifest_path = Path(state["export_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recomputed = _hash_value({k: v for k, v in manifest.items() if k != "manifest_hash"})
    assert recomputed == manifest["manifest_hash"]

    for key in ("prisma_diagram_path", "ai_traceability_json_path", "ai_traceability_md_path",
                "synthesis_claims_json_path", "synthesis_json_path", "synthesis_md_path",
                "claim_audit_json_path", "claim_audit_md_path"):
        assert Path(state[key]).exists(), f"{key} missing on disk"

    if review_type == "scoping":
        assert state["evidence_map_csv_path"] is not None and Path(state["evidence_map_csv_path"]).exists()
        assert reporting_module._renderer_entry(manifest, "quality_appraisal_csv")["status"] == "not_applicable"
        assert reporting_module._renderer_entry(manifest, "quality_appraisal_json")["status"] == "not_applicable"
    else:
        assert state["quality_appraisal_csv_path"] is not None and Path(state["quality_appraisal_csv_path"]).exists()
        assert state["quality_appraisal_json_path"] is not None and Path(state["quality_appraisal_json_path"]).exists()
        assert reporting_module._renderer_entry(manifest, "evidence_map_csv")["status"] == "not_applicable"

    ai_traceability = json.loads(Path(state["ai_traceability_json_path"]).read_text(encoding="utf-8"))
    assert ai_traceability["stages"]["extraction"]["instrumentation_status"] == "not_instrumented"
    assert ai_traceability["stages"]["quality_appraisal"]["instrumentation_status"] == "not_instrumented"
    assert ai_traceability["stages"]["synthesis_bertopic"]["instrumentation_status"] == "not_instrumented"
    assert ai_traceability["stages"]["dedup"]["instrumentation_status"] == "instrumented"
    assert ai_traceability["m10_data_governance"]["value"] == "No personal data; public bibliographic metadata only."

    synthesis_claims = json.loads(Path(state["synthesis_claims_json_path"]).read_text(encoding="utf-8"))
    assert synthesis_claims["claims"]

    claim_audit = json.loads(Path(state["claim_audit_json_path"]).read_text(encoding="utf-8"))
    assert "claims" in claim_audit and "unsupported_targets" in claim_audit


def test_export_manifest_schema_coverage(monkeypatch, tmp_path):
    """Track 2 readiness plan Step 9: a real `slr`-type run exercises every
    entry_kind in `ReportingStageEntry`'s `Literal` (7 total), and the
    written `export_manifest.json` round-trips through
    `ReportingStageManifest.model_validate()` unchanged -- i.e. what
    `publish_report_bundle` writes to disk is exactly what the strict
    schema accepts, with no entry_kind left untested."""
    state = _run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="slr")
    manifest_path = Path(state["export_manifest_path"])
    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))

    revalidated = ReportingStageManifest.model_validate(manifest_dict)
    assert revalidated.model_dump(mode="json") == manifest_dict

    seen_kinds = {entry["entry_kind"] for entry in manifest_dict["entries"]}
    assert seen_kinds == {
        "renderer_artifact", "selection_decision", "screening_batch",
        "extraction_component", "qa_component", "ai_use_event", "ft_run_artifact",
    }


def test_reporting_renders_no_absolute_paths(monkeypatch, tmp_path):
    """Plan Sec 10.5: the reporting boundary never leaks an absolute
    filesystem path (or any non-`slr://` locator scheme) into either the
    manifest or the renderer output text it writes -- only `http(s)://`
    (a real external citation URL) is exempt; `slr://` locators are the
    only internal reference scheme allowed through."""
    leak_re = re.compile(
        r"file://|ftp://|proj:|(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\[^\s\"']+"
    )

    def _assert_no_leaks(text: str, *, source: str) -> None:
        for line in text.splitlines():
            if "http://" in line or "https://" in line:
                continue  # the one exempt scheme (plan Sec 10.5)
            m = leak_re.search(line)
            assert m is None, f"{source}: absolute-path/foreign-scheme leak {m.group(0)!r} in line: {line!r}"

    state = _run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="slr")
    manifest_path = Path(state["export_manifest_path"])
    _assert_no_leaks(manifest_path.read_text(encoding="utf-8"), source="export_manifest.json")

    for key in ("methods_text",):
        _assert_no_leaks(state[key], source=key)
    for key in ("ai_traceability_md_path", "synthesis_md_path", "claim_audit_md_path"):
        _assert_no_leaks(Path(state[key]).read_text(encoding="utf-8"), source=key)


def test_full_pipeline_reporting_survives_harmless_full_text_records_reorder(monkeypatch, tmp_path):
    """Codex round-1 review, finding 3 (P1): reversing `state["full_text_
    records"]`'s own list order after synthesis (record identity/content
    unchanged -- a harmless live-state reorder) must NOT make reporting_node
    raise. Reproduces the exact scenario Codex's review confirmed against
    genuine extraction and synthesis artifacts."""
    def _reverse_full_text_records(state: Dict[str, Any]) -> None:
        state["full_text_records"] = list(reversed(state["full_text_records"]))

    state = _run_full_pipeline_to_reporting(
        monkeypatch, tmp_path, review_type="rapid", pre_reporting_hook=_reverse_full_text_records,
    )
    assert Path(state["export_manifest_path"]).exists()


def test_full_pipeline_reporting_treats_acquisition_failure_as_not_retrieved_not_excluded(monkeypatch, tmp_path):
    """Regression for the fulltext.py <-> reporting.py integration defect:
    an acquisition-stage full-text failure (full_text_acquisition_failure_
    policy="exclude") must be represented ONLY via PRISMACounts.
    fulltext_retrieval's non-"parsed" counts ("reports not retrieved"),
    never via excluded_full_text/excluded_full_text_reasons -- an
    acquisition failure never reaches FT_SCREENING, so it has no
    eligibility-ledger decision behind it, and reporting_node's ledger-vs-
    PRISMA reconciliation (compute_selection_splits) requires every
    excluded_full_text count to have one, so a wrongly-bumped
    excluded_full_text would make this raise ReportingConsistencyError.
    Simulates exactly the shape fulltext_node now produces for an
    acquisition failure under policy="exclude": no full_text_records
    entry, no ledger entry, only a bumped fulltext_retrieval count."""
    def _add_acquisition_not_retrieved_record(state: Dict[str, Any]) -> None:
        prisma = PRISMACounts(**state["prisma_counts"])
        retrieval = dict(prisma.fulltext_retrieval)
        retrieval["no_oa_fulltext_found"] = retrieval.get("no_oa_fulltext_found", 0) + 1
        prisma.fulltext_retrieval = retrieval
        state["prisma_counts"] = prisma.model_dump()

    state = _run_full_pipeline_to_reporting(
        monkeypatch, tmp_path, review_type="scoping",
        pre_reporting_hook=_add_acquisition_not_retrieved_record,
    )
    assert Path(state["export_manifest_path"]).exists()

    prisma = PRISMACounts(**state["prisma_counts"])
    assert prisma.fulltext_retrieval["no_oa_fulltext_found"] == 1
    # The defining assertion: the acquisition failure did NOT inflate
    # excluded_full_text -- reporting_node's own ledger reconciliation
    # would have raised ReportingConsistencyError above if it had.
    assert prisma.excluded_full_text == 0
    assert "Full-text reports not retrieved | 1" in state["methods_text"]


# ---------------------------------------------------------------------------
# 3b. Codex round-3 review -- on-disk tampering against a REAL full-pipeline
# run (not a synthetic, hand-built snapshot), reproducing the exact class
# of attack the review's findings 1(a) and 2(a) describe.
# ---------------------------------------------------------------------------

def test_full_pipeline_tamper_deleting_all_search_strategy_events_raises(monkeypatch, tmp_path):
    """Real on-disk tamper: run the full pipeline to a genuine, hand-
    published search_strategy_db_translation event, then delete every
    search_strategy_* event FILE from disk before reporting_node runs.
    Codex round-3 review, finding 1(a) -- must now raise, not silently
    pass with intact (but now-orphaned) oversight_mode/automation_origin
    state. Codex round-5 review, finding 3: the check is now anchored on
    `state["search_strategy"]["producing_event_ids"]` rather than a
    re-scan of every event under this stage name -- deleting the event
    file this invocation's own id points at now raises via
    `_load_verified_event_by_id()`'s own "could not be loaded/verified"
    message, not the older "no ... event exists" message."""
    def _delete_search_strategy_events(state: Dict[str, Any]) -> None:
        cfg = ReviewConfig.from_dict(state["config"])
        event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=state["review_run_id"])
        for event in event_store.load_all():
            if event.stage in ("search_strategy_query_expansion", "search_strategy_db_translation"):
                event_store.event_path(event.event_id).unlink()

    with pytest.raises(ReportingConsistencyError, match="search_strategy.*could not be loaded/verified"):
        _run_full_pipeline_to_reporting(
            monkeypatch, tmp_path, review_type="scoping", pre_reporting_hook=_delete_search_strategy_events,
        )


def test_full_pipeline_tamper_deleting_dedup_unique_decisions_raises(monkeypatch, tmp_path):
    """Real on-disk tamper: run the full pipeline to genuine dedup ledger
    decisions (both records this smoke corpus's own dedup step produces
    are 'unique'), then delete every 'unique'-decision ledger FILE from
    disk before reporting_node runs -- the exclusion-sum check alone
    would see 0 duplicates expected/0 duplicates found and pass; the
    record-ID-set reconciliation (Codex round-3 review, finding 2(a),
    strengthened to exact SET equality by round-4 review, finding 4) must
    catch the now-incomplete ledger."""
    def _delete_dedup_unique_decisions(state: Dict[str, Any]) -> None:
        cfg = ReviewConfig.from_dict(state["config"])
        review_run_id = state["review_run_id"]
        ledger_store = SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
        for record in ledger_store.load_all():
            if record.stage == "dedup" and record.decision == "unique":
                ledger_store.record_path(
                    record_id=record.record_id, stage=record.stage, source_event_id=record.source_event_id,
                ).unlink()

    with pytest.raises(ReportingConsistencyError, match=r"dedup: ledger current-tail record-ID set does not match"):
        _run_full_pipeline_to_reporting(
            monkeypatch, tmp_path, review_type="scoping", pre_reporting_hook=_delete_dedup_unique_decisions,
        )


def test_full_pipeline_tamper_swapping_dedup_record_id_raises(monkeypatch, tmp_path):
    """Codex round-4 review, finding 4 (P1) -- exact reproduction:
    replacing a missing expected record's ledger entry with an unknown/
    stale record_id preserves the total COUNT (and the exclusion sum,
    since the swapped-in entry keeps the same 'unique' decision) but must
    still raise, since the record-ID SET no longer matches the real raw
    corpus."""
    def _swap_one_dedup_record_id(state: Dict[str, Any]) -> None:
        import uuid as uuid_module
        from slr_pipeline.selection_ledger import build_selection_decision_record
        cfg = ReviewConfig.from_dict(state["config"])
        review_run_id = state["review_run_id"]
        ledger_store = SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
        records = ledger_store.load_all()
        victim = next(r for r in records if r.stage == "dedup" and r.decision == "unique")
        old_path = ledger_store.record_path(
            record_id=victim.record_id, stage=victim.stage, source_event_id=victim.source_event_id)
        old_path.unlink()
        forged = build_selection_decision_record(
            review_run_id=review_run_id, decision_id=str(uuid_module.uuid4()),
            source_event_id=derive_source_event_id(
                producer_artifact_id="forged-artifact", record_id="unknown-stale-record-id", stage="dedup"),
            record_id="unknown-stale-record-id", stage="dedup", decision="unique",
            decision_maker="rule_based_automation",
        )
        ledger_store.publish(forged)

    with pytest.raises(ReportingConsistencyError, match=r"dedup: ledger current-tail record-ID set does not match"):
        _run_full_pipeline_to_reporting(
            monkeypatch, tmp_path, review_type="scoping", pre_reporting_hook=_swap_one_dedup_record_id,
        )


def test_full_pipeline_tamper_swapping_fulltext_record_id_raises(monkeypatch, tmp_path):
    """Codex round-4 review, finding 4 (P1) -- the full_text_eligibility
    half of the same reproduction: swap one record's genuine ledger
    current-tail entry for an unknown/stale record_id, preserving the
    total count. The AUTHORITATIVE expected set here is FT_SCREENING's own
    `item_results.jsonl` (`ft_native_provenance['confirmed_record_ids']`),
    a genuinely independent, separately-hash-verified artifact -- left
    untouched by this tamper, so it still names the real, original record,
    catching the swap even though the ledger itself now looks internally
    consistent (one tail per record, correct decision)."""
    def _swap_one_fulltext_record_id(state: Dict[str, Any]) -> None:
        import uuid as uuid_module
        from slr_pipeline.selection_ledger import build_selection_decision_record
        cfg = ReviewConfig.from_dict(state["config"])
        review_run_id = state["review_run_id"]
        ledger_store = SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
        records = ledger_store.load_all()
        victim = next(r for r in records if r.stage == "full_text_eligibility")
        old_path = ledger_store.record_path(
            record_id=victim.record_id, stage=victim.stage, source_event_id=victim.source_event_id)
        old_path.unlink()
        forged = build_selection_decision_record(
            review_run_id=review_run_id, decision_id=str(uuid_module.uuid4()),
            source_event_id=derive_source_event_id(
                producer_artifact_id="forged-ft-artifact", record_id="unknown-stale-ft-record-id",
                stage="full_text_eligibility"),
            record_id="unknown-stale-ft-record-id", stage="full_text_eligibility", decision="include",
            decision_maker="ai",
        )
        ledger_store.publish(forged)

    with pytest.raises(ReportingConsistencyError,
                        match=r"full_text_eligibility: ledger current-tail record-ID set does not match"):
        _run_full_pipeline_to_reporting(
            monkeypatch, tmp_path, review_type="scoping", pre_reporting_hook=_swap_one_fulltext_record_id,
        )


def test_full_pipeline_tamper_editing_db_translation_to_impossible_failure_status_raises(monkeypatch, tmp_path):
    """Codex round-7 review, finding 2 (P1) -- real on-disk tamper: replace
    the genuine, already-published search_strategy_db_translation event
    with a legitimately self-hashed one (same event_id -- so `state["
    search_strategy"]["producing_event_ids"]` still points at it) whose
    call_status is 'failure' -- an impossible shape for this call site
    (_generate_db_queries() always has a real same-query-per-DB computed
    fallback) -- and confirm reporting_node now raises rather than
    accepting it."""
    def _corrupt_db_translation_to_failure(state: Dict[str, Any]) -> None:
        cfg = ReviewConfig.from_dict(state["config"])
        event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=state["review_run_id"])
        event_id = state["search_strategy"]["producing_event_ids"][0]
        original = event_store.load_event(event_id)
        assert original.stage == "search_strategy_db_translation"
        event_store.event_path(event_id).unlink()
        forged = build_ai_use_event(
            review_run_id=original.review_run_id, stage=original.stage, work_item_id=original.work_item_id,
            attempt=original.attempt, tool_kind=original.tool_kind, model=original.model, provider=original.provider,
            component_versions=original.component_versions, access_info=original.access_info,
            task_purpose=original.task_purpose, model_config_id=original.model_config_id,
            prompt_template_version=original.prompt_template_version, prompt_template_text=original.prompt_template_text,
            rendered_prompt_hash=original.rendered_prompt_hash,
            input_artifact_refs=original.input_artifact_refs, output_artifact_refs=[],
            call_status="failure",
        )
        assert forged.event_id == event_id
        event_store.publish(forged)

    with pytest.raises(ReportingConsistencyError, match=r"db_translation event has call_status='failure'"):
        _run_full_pipeline_to_reporting(
            monkeypatch, tmp_path, review_type="scoping", pre_reporting_hook=_corrupt_db_translation_to_failure,
        )


def test_full_pipeline_tamper_relabeling_ft_screening_resolution_as_acquisition_raises(monkeypatch, tmp_path):
    """Codex round-7 review, finding 1 (P1) -- reproduction against a
    genuine full pipeline run: inject a real ft_screening-shaped
    full_text_review_resolutions entry (as fulltext.py's own
    resolve_full_text_review_queue() would have produced it, complete with
    resolution_id/reviewed_baseline_decision_id/reason), then relabel its
    stage to 'acquisition'/action to 'not_retrieved' while KEEPING those
    FT-only fields -- reporting_node must now raise rather than silently
    skipping ledger linkage for it."""
    def _relabel_ft_resolution_as_acquisition(state: Dict[str, Any]) -> None:
        cfg = ReviewConfig.from_dict(state["config"])
        review_run_id = state["review_run_id"]
        ledger_store = SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
        records = ledger_store.load_all()
        victim = next(r for r in records if r.stage == "full_text_eligibility")
        resolution_id = compute_resolution_id(
            stage="full_text_eligibility", record_id=victim.record_id, action="exclude", reviewer="x", note="n",
            resolved_at="t", reviewed_baseline_decision_id=victim.decision_id, exclusion_reason="a real reason",
        )
        state["full_text_review_resolutions"] = list(state.get("full_text_review_resolutions") or []) + [{
            "record_id": victim.record_id,
            # Relabeled -- but resolution_id/reviewed_baseline_decision_id/
            # reason (FT-only fields) were left attached.
            "stage": "acquisition", "action": "not_retrieved",
            "reviewer": "x", "note": "n", "resolved_at": "t",
            "resolution_id": resolution_id, "reviewed_baseline_decision_id": victim.decision_id,
            "reason": "a real reason",
        }]

    with pytest.raises(ReportingConsistencyError, match=r"acquisition-stage entry for record .* has key set"):
        _run_full_pipeline_to_reporting(
            monkeypatch, tmp_path, review_type="scoping", pre_reporting_hook=_relabel_ft_resolution_as_acquisition,
        )


# ---------------------------------------------------------------------------
# 3a. §2.8 screening current-batch regressions (1, 3, 4, 5)
# ---------------------------------------------------------------------------

def _publish_screening_checkpoint(*, cfg, review_run_id, event_store, batch_store, batch, call_status: str):
    artifact_id, fingerprint, record_ids = compute_screening_batch_identity(batch=batch, cfg=cfg, review_run_id=review_run_id)
    responses = [ScreeningResponse(decision="include", reason="matched keywords", relevant_topics=[], confidence=0.6)
                 for _ in batch]
    output_payload = [r.model_dump(mode="json") for r in responses]
    work_item_id = f"title_abstract_screening:{fingerprint}"
    event = build_ai_use_event(
        review_run_id=review_run_id, stage="title_abstract_screening", work_item_id=work_item_id, attempt=0,
        tool_kind="llm", model=cfg.screening.llm_model, provider="openai", component_versions={},
        task_purpose="test", model_config_id="cfg-1", prompt_template_version="v1", prompt_template_text="t",
        rendered_prompt_hash="h",
        input_artifact_refs=["in-1"], output_artifact_refs=[f"sha256:{sha256_hex(output_payload)}"],
        call_status=call_status,
    )
    event = event_store.publish(event)
    result = build_screening_batch_result(
        review_run_id=review_run_id, stage="title_abstract_screening", batch_fingerprint=fingerprint,
        record_ids=record_ids, producing_event_id=event.event_id, call_status=call_status, responses=responses,
    )
    return batch_store.publish(result)


def test_screening_regression_1_stale_fingerprint_never_leaks_into_mixed(tmp_path):
    """A policy/threshold change gives the SAME batch a NEW fingerprint --
    an old, now-irrelevant heuristic_fallback checkpoint under the OLD
    fingerprint must never be found or counted; only the CURRENT
    checkpoint's own call_status ("success") determines automation_origin."""
    review_run_id = "run-regr1"
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), screening=ScreeningConfig(auto_include_threshold=0.80))
    old_cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"), screening=ScreeningConfig(auto_include_threshold=0.50))
    records = [Record(id="r1", source="s", source_id="r1", title="Exercise improves outcome")]

    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    batch_store = ScreeningBatchResultStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id, event_store=event_store)

    _publish_screening_checkpoint(cfg=old_cfg, review_run_id=review_run_id, event_store=event_store,
                                   batch_store=batch_store, batch=records, call_status="heuristic_fallback")
    _publish_screening_checkpoint(cfg=cfg, review_run_id=review_run_id, event_store=event_store,
                                   batch_store=batch_store, batch=records, call_status="success")

    current = reporting_module._load_current_screening_batches(
        records=records, cfg=cfg, review_run_id=review_run_id, event_store=event_store)
    assert len(current) == 1
    assert current[0].model.call_status == "success"
    makers = {"ai" if vf.model.call_status == "success" else "rule_based_automation" for vf in current}
    assert makers == {"ai"}


def test_screening_regression_2_origin_reflects_current_oversight_from_resolutions():
    """automation_origin comes from the CURRENT batch checkpoint alone
    (never the ledger's own A0->H->A1 history); oversight_mode is keyed
    purely off screening_review_resolutions non-emptiness (§0.2), fully
    decoupled from automation_origin."""
    origin_from_current_batch = "ai"  # as if re-derived from an "A1" success checkpoint
    screening_review_resolutions = [{"record_id": "r1", "decision": "include"}]  # "H" happened earlier
    oversight = "human_resolved" if screening_review_resolutions else "automation_unreviewed"
    assert origin_from_current_batch == "ai"
    assert oversight == "human_resolved"


def test_screening_regression_3_growing_corpus_counts_both_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(screening_module, "SCREENING_BATCH_SIZE", 1)  # force 2 SEPARATE batches, not 1 combined
    review_run_id = "run-regr3"
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    batch1 = [Record(id="r1", source="s", source_id="r1", title="a")]
    batch2 = [Record(id="r2", source="s", source_id="r2", title="b")]

    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    batch_store = ScreeningBatchResultStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id, event_store=event_store)
    _publish_screening_checkpoint(cfg=cfg, review_run_id=review_run_id, event_store=event_store,
                                   batch_store=batch_store, batch=batch1, call_status="success")
    _publish_screening_checkpoint(cfg=cfg, review_run_id=review_run_id, event_store=event_store,
                                   batch_store=batch_store, batch=batch2, call_status="heuristic_fallback")

    current = reporting_module._load_current_screening_batches(
        records=batch1 + batch2, cfg=cfg, review_run_id=review_run_id, event_store=event_store)
    assert len(current) == 2
    makers = {"ai" if vf.model.call_status == "success" else "rule_based_automation" for vf in current}
    assert makers == {"ai", "rule_based_automation"}  # "mixed", genuinely two distinct mechanisms


def test_screening_regression_4_empty_corpus_returns_no_batches(tmp_path):
    review_run_id = "run-regr4"
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    current = reporting_module._load_current_screening_batches(
        records=[], cfg=cfg, review_run_id=review_run_id, event_store=event_store)
    assert current == []


def test_screening_regression_5_full_text_subset_does_not_affect_batch_lookup(tmp_path):
    """`_load_current_screening_batches()` takes `screening_records` (==
    state["deduplicated_records"]) -- never `full_text_records` -- so a
    full_text_records that is a STRICT SUBSET of the screened corpus (some
    records excluded later at full-text stage) has no bearing on whether
    every screening batch is found; the lookup is keyed purely on the
    screened corpus itself."""
    review_run_id = "run-regr5"
    cfg = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    screening_records = [Record(id="r1", source="s", source_id="r1", title="a"),
                          Record(id="r2", source="s", source_id="r2", title="b")]
    # full_text_records (not passed to _load_current_screening_batches at all) would be
    # a strict subset, e.g. only r1 -- irrelevant to this function's own contract.
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    batch_store = ScreeningBatchResultStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id, event_store=event_store)
    _publish_screening_checkpoint(cfg=cfg, review_run_id=review_run_id, event_store=event_store,
                                   batch_store=batch_store, batch=screening_records, call_status="success")
    current = reporting_module._load_current_screening_batches(
        records=screening_records, cfg=cfg, review_run_id=review_run_id, event_store=event_store)
    assert len(current) == 1
    assert current[0].model.record_ids == ["r1", "r2"]


def test_screening_effective_controls(tmp_path, monkeypatch):
    """Track 2 plan §4.4: `effective_batch_size(cfg)` resolves
    `cfg.screening.batch_size` when set, else `SCREENING_BATCH_SIZE` --
    an explicit override matching the env constant's value must produce
    IDENTICAL batch partitioning, batch identities (artifact_id,
    batch_fingerprint), and reporting's own current-screening-batch
    inventory as the env-fallback path, never a merely-equivalent-looking
    but differently-derived result."""
    monkeypatch.setattr(screening_module, "SCREENING_BATCH_SIZE", 1)  # env fallback value
    review_run_id = "run-effective-controls"
    cfg_env = ReviewConfig(title="t", output_dir=str(tmp_path / "out"))
    cfg_explicit = ReviewConfig(title="t", output_dir=str(tmp_path / "out"),
                                 screening=ScreeningConfig(batch_size=1))  # same value, explicit
    assert cfg_env.screening.batch_size is None  # genuinely exercising the fallback branch

    records = [Record(id="r1", source="s", source_id="r1", title="a"),
               Record(id="r2", source="s", source_id="r2", title="b")]

    batches_env = screening_module.compute_current_screening_batches(records=records, cfg=cfg_env)
    batches_explicit = screening_module.compute_current_screening_batches(records=records, cfg=cfg_explicit)
    assert [[r.id for r in b] for b in batches_env] == [[r.id for r in b] for b in batches_explicit] == [["r1"], ["r2"]]

    for batch_env, batch_explicit in zip(batches_env, batches_explicit):
        identity_env = compute_screening_batch_identity(batch=batch_env, cfg=cfg_env, review_run_id=review_run_id)
        identity_explicit = compute_screening_batch_identity(batch=batch_explicit, cfg=cfg_explicit, review_run_id=review_run_id)
        assert identity_env == identity_explicit

    event_store = AIUseEventStore(output_dir=cfg_env.ensure_output_dir(), review_run_id=review_run_id)
    batch_store = ScreeningBatchResultStore(output_dir=cfg_env.ensure_output_dir(), review_run_id=review_run_id, event_store=event_store)
    for batch in batches_env:
        _publish_screening_checkpoint(cfg=cfg_env, review_run_id=review_run_id, event_store=event_store,
                                       batch_store=batch_store, batch=batch, call_status="success")

    inventory_env = reporting_module._load_current_screening_batches(
        records=records, cfg=cfg_env, review_run_id=review_run_id, event_store=event_store)
    inventory_explicit = reporting_module._load_current_screening_batches(
        records=records, cfg=cfg_explicit, review_run_id=review_run_id, event_store=event_store)
    assert [vf.model.artifact_id for vf in inventory_env] == [vf.model.artifact_id for vf in inventory_explicit]
    assert [vf.model.record_ids for vf in inventory_env] == [["r1"], ["r2"]]


def test_load_and_verify_synthesis_output_requires_synthesis_state():
    cfg = ReviewConfig(title="t")
    with pytest.raises(ReportingConsistencyError, match="state\\['synthesis'\\] is missing"):
        reporting_module.load_and_verify_synthesis_output(
            cfg=cfg, records=[], extraction_raw={}, qa_raw=None, review_run_id="run-1",
            event_store=AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id="run-1"),
            raw_synthesis_state=None,
        )
