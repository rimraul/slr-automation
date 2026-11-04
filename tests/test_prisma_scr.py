"""Tests for `slr_pipeline.prisma_scr` -- the frozen 22-row PRISMA-ScR
matrix (Track 2 plan §9) plus step 14's ([C13]) two renderers,
`render_prisma_scr_report_md()`/`render_prisma_scr_conformance()`."""
from __future__ import annotations

from pathlib import Path

import pytest

from slr_pipeline.ai_traceability import compute_event_id
from slr_pipeline.config import ReviewConfig
from slr_pipeline.nodes.reporting import (
    ARTIFACT_RENDERERS, EvidenceMapModel, EvidenceMapRow, ReportingConfigProjection,
    ReportingSourceSnapshot, SourceBundle, VerifiedComponent, VerifiedProjection, _state_value,
)
from slr_pipeline.prisma_scr import (
    ESSENTIAL_ITEM_IDS, ITEMS, OPTIONAL_ITEM_IDS, PRISMA_SCR_MATRIX_VERSION, PRISMA_SCR_VERSION,
    matrix_payload, matrix_text, render_prisma_scr_conformance, render_prisma_scr_report_md,
)
from slr_pipeline.reference_spec import reference_reporting_context
from slr_pipeline.state import (
    ClaimConstructionCoverage, EvidenceReference, MapOutcomeRecord, PRISMACounts, RefineAppliedUpdate,
    RefineBatchOutcomeRecord, SearchLog, DatabaseSearchLog, SupportingFieldSelector, SynthesisOutput,
    build_map_answer, build_synthesis_claim,
)
from slr_pipeline.synthesis_evidence import make_evidence_id

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_FIXTURE = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "prisma_scr_matrix.json"


# ---------------------------------------------------------------------------
# The matrix itself
# ---------------------------------------------------------------------------

def test_matrix_has_22_items_1_to_22():
    assert len(ITEMS) == 22
    assert [row["item_id"] for row in ITEMS] == list(range(1, 23))


def test_essential_optional_split_is_all_except_12_and_16():
    assert OPTIONAL_ITEM_IDS == frozenset({12, 16})
    assert ESSENTIAL_ITEM_IDS == frozenset(range(1, 23)) - {12, 16}


def test_matrix_byte_equality_against_fixture():
    """Cross-checks `matrix_text()`'s own generation against the frozen
    fixture `scripts/author_prisma_scr_matrix.py --write` authored
    (Track 2 step 12A(c)) -- this module is now that script's canonical
    source (see its own docstring), so the two must always agree."""
    on_disk = MATRIX_FIXTURE.read_text(encoding="utf-8")
    assert on_disk == matrix_text()


def test_matrix_payload_shape():
    payload = matrix_payload()
    assert payload["prisma_scr_matrix_version"] == PRISMA_SCR_MATRIX_VERSION == "1"
    assert payload["prisma_scr_version"] == PRISMA_SCR_VERSION == "2018"
    assert payload["items"] is ITEMS


# ---------------------------------------------------------------------------
# Renderer fixtures
# ---------------------------------------------------------------------------

def _snapshot(*, reference_context, with_data: bool = True) -> ReportingSourceSnapshot:
    cfg = ReviewConfig(
        title="Exemplar Scoping Review", review_type="scoping",
        research_questions=["RQ1: What interventions are reported?"],
        inclusion_criteria=["Peer-reviewed", "English language"],
        exclusion_criteria=["Conference abstracts"],
        population="adults", concept="self-management", context="primary care",
        funding_statement="No dedicated funding was received for this review.",
    )
    config_projection = ReportingConfigProjection.from_config(cfg)
    prisma = PRISMACounts(
        identified={"pubmed": 5}, after_dedup=5, screened_title_abstract=5,
        excluded_title_abstract=3, assessed_full_text=2, excluded_full_text=1, included=1,
    )
    search_log = SearchLog(entries=[
        DatabaseSearchLog(database="pubmed", execution_adapter="fx_pubmed", adapter_config_version="v1",
                           query="diabetes AND education", date="2026-01-01",
                           retrieval_status="exhaustive", termination_reason="total_hits_equals_retrieved",
                           requested=5, hits=5, retrieved=5),
    ])
    extraction = None
    evidence_map = None
    synthesis_output = VerifiedProjection(model=SynthesisOutput(refine_batch_size=cfg.synthesis.refine_batch_size),
                                           content_hash="c" * 64)
    if with_data:
        extraction = SourceBundle(
            raw={"resolved_projection": [{
                "record_id": "rec-0001",
                # key_findings/funding_sources are TEXT_LIST (List[str]) per the real
                # reference_extraction_schema(); funder_role is TEXT but may be None --
                # both shapes exercised here on purpose, see test_report_md_formats_
                # list_and_none_field_values_readably below.
                "raw_extraction": {
                    "country": "USA", "study_design": "RCT", "population": "adults",
                    "concept": "self-management", "context": "primary care",
                    "limitations": "small sample", "key_findings": ["improved outcomes"],
                    "funding_sources": ["NIH"], "funder_role": None,
                },
            }]},
            manifest=VerifiedComponent(path=Path("manifest.json"), file_sha256="a" * 64, internal_hash="a" * 64),
            checkpoints=[], resolutions=None, confirmations=None, computed_content_hash="b" * 64,
            run_id="extrun-1",
        )
        evidence_map = EvidenceMapModel(
            n_studies=1, by_year={"2020": 1}, by_study_design={"RCT": 1}, by_country={"USA": 1},
            characteristics_table=[EvidenceMapRow(citation="Doe 2020", year=2020, country="USA", study_design="RCT")],
        )
        rq = "RQ1: What interventions are reported?"
        event_id = compute_event_id(review_run_id="run-1", stage="synthesis", work_item_id="rec-0001", attempt=1)
        map_answer = build_map_answer(
            rq_or_section=rq, record_id="rec-0001",
            answer_text="Self-management education improved outcomes.",
            selectors=[SupportingFieldSelector(artifact_kind="extraction", field_id="key_findings")],
            evidence_references=[EvidenceReference(
                artifact_kind="extraction", artifact_run_id="extrun-1",
                record_id="rec-0001", field_id="key_findings",
                evidence_id=make_evidence_id(
                    artifact_kind="extraction", artifact_run_id="extrun-1", record_id="rec-0001",
                    field_id="key_findings", chunk_id="chunk-1", quotation="improved outcomes",
                    page="3", section=None, locator=None,
                ),
            )],
            producing_event_id=event_id,
        )
        refine_event_id = compute_event_id(
            review_run_id="run-1", stage="synthesis_refine", work_item_id="rec-0001", attempt=1,
        )
        claim_text = "One study reported a self-management intervention."
        claim = build_synthesis_claim(claim_text, [map_answer], rq, refine_event_id)
        synthesis_output = VerifiedProjection(
            model=SynthesisOutput(
                refine_batch_size=cfg.synthesis.refine_batch_size,
                extraction_run_id="extrun-1",
                synthesis_record_ids=["rec-0001"],
                requested_targets=[rq],
                rq_synthesis={rq: claim_text},
                claims=[claim],
                target_coverage=1.0,
                claim_construction_coverage=[ClaimConstructionCoverage(
                    rq_or_section=rq, unincorporated_answer_ids=[],
                    refine_batches_total=1, refine_batches_applied=1,
                )],
                refine_batch_outcomes=[RefineBatchOutcomeRecord(
                    rq_or_section=rq, batch_index=0, batch_answer_ids=[map_answer.answer_id],
                    outcome="applied", producing_event_id=refine_event_id,
                    applied_updates=[RefineAppliedUpdate(
                        claim_index=None, claim_text=claim_text,
                        final_support_answer_ids=[map_answer.answer_id],
                    )],
                )],
                map_outcome_records=[MapOutcomeRecord(
                    rq_or_section=rq, record_id="rec-0001", outcome="relevant", producing_event_id=event_id,
                )],
                map_answers=[map_answer],
            ),
            content_hash="c" * 64,
        )
    return ReportingSourceSnapshot(
        review_run_id="run-1", config_projection=config_projection, config_projection_hash="d" * 64,
        raw_records=_state_value([]), screening_records=[], full_text_records=[], screened_records_full=[],
        prisma=_state_value(prisma), search_log=_state_value(search_log),
        screening_review_resolutions=_state_value([]), full_text_review_resolutions=_state_value([]),
        full_text_review_queue=_state_value([]),
        screening_qc_state=None, oversight_mode_state={}, automation_origin_state={},
        events=[], ledger_records=[], current_screening_batch_results=[],
        extraction=extraction, quality_appraisal=None,
        synthesis_output=synthesis_output, evidence_map=evidence_map, ft_native_provenance=None,
        reference_context=reference_context,
    )


# ---------------------------------------------------------------------------
# render_prisma_scr_report_md
# ---------------------------------------------------------------------------

def test_report_md_has_every_item_anchor():
    text = render_prisma_scr_report_md(_snapshot(reference_context=reference_reporting_context()))
    for item in ITEMS:
        assert f'<a id="{item["anchor"]}"></a>' in text, f"missing anchor for item {item['item_id']}"


def test_report_md_deterministic():
    snapshot = _snapshot(reference_context=reference_reporting_context())
    assert render_prisma_scr_report_md(snapshot) == render_prisma_scr_report_md(snapshot)


def test_report_md_optional_items_state_not_conducted():
    text = render_prisma_scr_report_md(_snapshot(reference_context=reference_reporting_context()))
    assert text.count("Not conducted") == 2  # items 12 and 16


def test_report_md_renders_charted_and_synthesis_content():
    text = render_prisma_scr_report_md(_snapshot(reference_context=reference_reporting_context()))
    assert "rec-0001" in text
    assert "Self-management education improved outcomes." in text
    assert "One study reported a self-management intervention." in text


def test_report_md_abstract_counts_databases_not_research_questions():
    """Regression: item 2's abstract once reported `len(cfg.research_questions)`
    database(s) searched instead of `len(cfg.search_databases)` -- the fixture's
    1 research question vs. `SearchConfig`'s 4 default databases makes the two
    counts distinguishable, so this would fail again if the bug returned."""
    text = render_prisma_scr_report_md(_snapshot(reference_context=reference_reporting_context()))
    assert "4 database(s) searched" in text
    assert "1 database(s) searched" not in text


def test_report_md_formats_list_and_none_field_values_readably():
    """Regression: `key_findings`/`funding_sources` are `List[str]`
    (TEXT_LIST) in the real extraction schema and `funder_role` may be
    `None` -- interpolating the raw Python value once rendered a list repr
    (`['improved outcomes']`) and the literal string 'None' into the report."""
    text = render_prisma_scr_report_md(_snapshot(reference_context=reference_reporting_context()))
    assert "['improved outcomes']" not in text
    assert "key_findings: improved outcomes" in text
    assert "funder role = None" not in text
    assert "funder role = Not reported" in text


def test_report_md_empty_corpus_does_not_crash():
    text = render_prisma_scr_report_md(_snapshot(reference_context=reference_reporting_context(), with_data=False))
    assert "No eligible sources this run." in text or "No sources were eligible" in text


# ---------------------------------------------------------------------------
# render_prisma_scr_conformance
# ---------------------------------------------------------------------------

def test_conformance_has_22_items_with_correct_statuses():
    conformance = render_prisma_scr_conformance(_snapshot(reference_context=reference_reporting_context()))
    items = conformance["items"]
    assert len(items) == 22
    by_id = {row["item_id"]: row for row in items}
    for item_id in OPTIONAL_ITEM_IDS:
        assert by_id[item_id]["status"] == "optional_not_conducted"
        assert "rationale" in by_id[item_id]
    for item_id in ESSENTIAL_ITEM_IDS:
        assert by_id[item_id]["status"] == "reported"


def test_conformance_locators_are_well_formed_slr_uris():
    from slr_pipeline.locators import LogicalLocator

    conformance = render_prisma_scr_conformance(_snapshot(reference_context=reference_reporting_context()))
    locator_items = [row for row in conformance["items"] if "locator" in row]
    assert locator_items, "expected at least one item to carry a locator"
    for row in locator_items:
        LogicalLocator.parse(row["locator"])  # raises on malformed grammar/unregistered kind


def test_conformance_deterministic():
    snapshot = _snapshot(reference_context=reference_reporting_context())
    assert render_prisma_scr_conformance(snapshot) == render_prisma_scr_conformance(snapshot)


# ---------------------------------------------------------------------------
# Applicability gating -- `reference_context is None` (every production run)
# ---------------------------------------------------------------------------

def _renderer_entry(name):
    for entry in ARTIFACT_RENDERERS:
        if entry[0] == name:
            return entry
    raise AssertionError(f"no ARTIFACT_RENDERERS entry named {name!r}")


@pytest.mark.parametrize("name", ["prisma_scr_report_md", "prisma_scr_conformance_json"])
def test_renderer_not_applicable_when_reference_context_none(name):
    _, _, applicable_fn, _, _ = _renderer_entry(name)
    assert applicable_fn(_snapshot(reference_context=None)) is False
    assert applicable_fn(_snapshot(reference_context=reference_reporting_context())) is True
