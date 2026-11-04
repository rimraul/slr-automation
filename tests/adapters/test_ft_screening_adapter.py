"""Offline tests for slr_pipeline.adapters.ft_screening_adapter (Phase A of
the FT_SCREENING integration). No OpenAI client, credentials, network, PDF
downloads, Docling model loading, or filesystem writes beyond pytest's own
tmp_path -- every FT_SCREENING call here goes through fakes
(ft_screening.fakes + the private _run_ft_screening), never
resolve_components()/run_ft_screening().
"""

from __future__ import annotations

import ast
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from slr_pipeline.adapters.ft_screening_adapter import (
    CHAPTER_MODEL_CONFIG_ID,
    ChapterAdapterError,
    ChapterFullTextOutcome,
    build_full_text_screening_request,
    chapter_screening_config,
    map_run_result_to_chapter_outcomes,
    record_to_screening_item,
)
from slr_pipeline.state import Record

from ft_screening import CriteriaSet, CriterionSpec, ScreeningConfig, ScreeningItem
from ft_screening.errors import ComponentResolutionError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _record(**overrides) -> Record:
    defaults = dict(
        id="rec-0001", source="openalex", source_id="W123", doi="10.1000/xyz",
        title="A study of things", year=2024,
    )
    defaults.update(overrides)
    return Record(**defaults)


def _criteria_set(**overrides) -> CriteriaSet:
    defaults = dict(
        criteria_set_id="chapter-ft-pilot", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1",
                eligibility_requirement="Adult human participants",
                missing_evidence_policy="review",
                exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population",
                priority=1,
            ),
        ],
    )
    defaults.update(overrides)
    return CriteriaSet(**defaults)


# ---------------------------------------------------------------------------
# chapter_screening_config() -- the calibrated pilot preset.
# ---------------------------------------------------------------------------


def test_chapter_screening_config_locked_values():
    config = chapter_screening_config()
    assert config.adjudicator_model == "gpt-5.6-terra"
    assert config.model_config_id == CHAPTER_MODEL_CONFIG_ID
    assert config.model_options["max_completion_tokens"] == 4096
    assert config.model_options["reasoning_effort"] == "medium"
    assert config.parser_name == "docling"
    assert config.retriever_name == "lexical_bm25"
    assert config.adjudicator_provider == "openai"


def test_chapter_screening_config_is_deterministic():
    a = chapter_screening_config()
    b = chapter_screening_config()
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def test_chapter_screening_config_returns_ordinary_screening_config():
    config = chapter_screening_config()
    assert isinstance(config, ScreeningConfig)


def test_chapter_screening_config_requires_no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OPENAI_API_KEY" not in os.environ
    config = chapter_screening_config()  # must not raise
    assert config is not None


def test_chapter_screening_config_creates_no_files_or_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chapter_screening_config()
    assert list(tmp_path.iterdir()) == []


def test_chapter_screening_config_never_imports_openai(monkeypatch):
    # monkeypatch.delitem (not a raw sys.modules.pop()) so this test's
    # removal of the CANONICAL "openai" module entry is automatically
    # restored afterward -- a raw pop() here previously left "openai"
    # permanently absent from sys.modules for the rest of the pytest
    # session, so any OTHER test file's later `from openai import ...`
    # (executed after this test ran) silently re-imported openai as a
    # BRAND NEW, separate module object -- invisible to (and unaffected
    # by) that other test's own `monkeypatch.setattr(openai, ...)`,
    # since it had patched the OLD module object. Same class of bug as
    # this file's own earlier importlib.reload() fix, via a different
    # mechanism (sys.modules removal instead of module reload).
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    chapter_screening_config()
    assert "openai" not in sys.modules


# ---------------------------------------------------------------------------
# record_to_screening_item() -- deterministic Record + confirmed pdf_path
# -> ScreeningItem.
# ---------------------------------------------------------------------------


def test_record_to_screening_item_maps_document_id_and_report_id_to_record_id(tmp_path):
    record = _record(id="rec-0001")
    pdf_path = tmp_path / "rec-0001.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content for a test fixture only")

    item = record_to_screening_item(record, pdf_path=pdf_path)
    assert isinstance(item, ScreeningItem)
    assert item.document_id == "rec-0001"
    assert item.report_id == "rec-0001"
    assert item.pdf_path == pdf_path


def test_record_to_screening_item_is_deterministic(tmp_path):
    record = _record(id="rec-0001")
    pdf_path = tmp_path / "rec-0001.pdf"
    a = record_to_screening_item(record, pdf_path=pdf_path)
    b = record_to_screening_item(record, pdf_path=pdf_path)
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def test_record_to_screening_item_citation_omits_none_fields(tmp_path):
    record = _record(id="rec-0001", doi=None, year=None)
    pdf_path = tmp_path / "rec-0001.pdf"
    item = record_to_screening_item(record, pdf_path=pdf_path)
    assert "doi" not in item.citation
    assert "year" not in item.citation
    assert item.citation["record_id"] == "rec-0001"
    assert item.citation["source"] == "openalex"


def test_record_to_screening_item_citation_never_carries_screening_state(tmp_path):
    """citation is display/audit metadata only -- never any field that
    would leak a PRIOR title/abstract screening decision into the
    full-text adjudication prompt's neighborhood."""
    record = _record(id="rec-0001", screening_decision="include", screening_reason="looked relevant")
    pdf_path = tmp_path / "rec-0001.pdf"
    item = record_to_screening_item(record, pdf_path=pdf_path)
    assert "screening_decision" not in item.citation
    assert "screening_reason" not in item.citation


@pytest.mark.parametrize(
    "bad_id",
    ["", "has spaces", "has/slash", "has.dot", "has:colon", "..", "a" * 129, "héllo"],
)
def test_record_to_screening_item_rejects_unsafe_record_id(tmp_path, bad_id):
    """Record.id itself has no min_length/format constraint (any str is a
    valid Record), so this adapter must be the one to reject an empty or
    unsafe id before it ever reaches FT_SCREENING."""
    record = _record(id=bad_id)
    pdf_path = tmp_path / "x.pdf"
    with pytest.raises(ChapterAdapterError):
        record_to_screening_item(record, pdf_path=pdf_path)


def test_record_to_screening_item_accepts_uuid_style_id(tmp_path):
    """The default Record.id (uuid4 hex+hyphens) must pass -- it's the
    common case, not just hand-picked safe ids."""
    import uuid

    record = _record(id=str(uuid.uuid4()))
    pdf_path = tmp_path / "x.pdf"
    item = record_to_screening_item(record, pdf_path=pdf_path)
    assert item.document_id == record.id


# ---------------------------------------------------------------------------
# build_full_text_screening_request() -- explicit, pure assembly; no
# hidden defaulting; FT_SCREENING's own validation still applies.
# ---------------------------------------------------------------------------


def test_build_full_text_screening_request_assembles_fields(tmp_path):
    record = _record(id="rec-0001")
    pdf_path = tmp_path / "rec-0001.pdf"
    item = record_to_screening_item(record, pdf_path=pdf_path)
    criteria_set = _criteria_set()
    config = chapter_screening_config()

    request = build_full_text_screening_request(
        items=[item], criteria_set=criteria_set, config=config, output_dir=tmp_path / "out",
    )
    assert request.items == [item]
    assert request.criteria_set == criteria_set
    assert request.config == config
    assert request.output_dir == tmp_path / "out"


def test_build_full_text_screening_request_never_derives_criteria_from_protocol_lists(tmp_path):
    """Structural proof, not a behavioral one: the function has no
    parameter that could accept a Protocol/free-text criteria list at
    all -- passing one raises TypeError, the same structural-rejection
    pattern FT_SCREENING's own build_supported_screening_config() uses
    for locked identities."""
    with pytest.raises(TypeError):
        build_full_text_screening_request(
            items=[], inclusion_criteria=["Empirical studies"], output_dir=tmp_path,
        )


def test_build_full_text_screening_request_propagates_ft_screening_validation_for_duplicate_ids(tmp_path):
    """Duplicate record ids reaching this stage is a slr_pipeline-side
    bug -- FT_SCREENING's own ScreeningRequest validator catches it;
    this function does not pre-check or silently disambiguate."""
    record = _record(id="rec-dup")
    pdf_path = tmp_path / "rec-dup.pdf"
    item1 = record_to_screening_item(record, pdf_path=pdf_path)
    item2 = record_to_screening_item(record, pdf_path=pdf_path)  # same id, deterministic

    with pytest.raises(Exception):  # pydantic.ValidationError from ScreeningRequest itself
        build_full_text_screening_request(
            items=[item1, item2], criteria_set=_criteria_set(), config=chapter_screening_config(),
            output_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# ChapterFullTextOutcome / map_run_result_to_chapter_outcomes() -- exercised
# end-to-end through a REAL FT_SCREENING run, but entirely against fakes
# (ft_screening.fakes + the private _run_ft_screening) -- proves the
# mapping against FT_SCREENING's actual result shape, not a hand-rolled
# stand-in, while staying fully offline.
# ---------------------------------------------------------------------------


def _run_fake_screening(
    tmp_path, *, criteria_set, items, status_by_key=None, requires_review_by_key=None, fail_document_ids=None,
):
    from ft_screening.components import Components
    from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
    from ft_screening.orchestrator import _run_ft_screening
    from ft_screening.schema import ScreeningConfig as _SC
    from ft_screening.schema import ScreeningRequest
    from ft_screening.store import FileScreeningStore

    config = _SC(
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="fake", retriever_config_version="1",
        adjudicator_provider="fake", adjudicator_model="fake-1", model_config_id="cfg-1",
        prompt_template_version="1", output_schema_version="1", aggregation_policy_version="1",
    )
    output_dir = tmp_path / "run_output"
    request = ScreeningRequest(
        run_id="chapter-adapter-test-run", items=items, criteria_set=criteria_set, config=config,
        output_dir=output_dir,
    )
    components = Components(
        parser=FakeDocumentParser(fail_document_ids=fail_document_ids or set()),
        retriever=FakeEvidenceRetriever(),
        adjudicator=FakeCriterionAdjudicator(
            status_by_key=status_by_key or {}, requires_review_by_key=requires_review_by_key or {},
        ),
        store=FileScreeningStore(output_dir=output_dir),
    )
    return _run_ft_screening(request, components)


def test_map_run_result_include_exclude_and_missing_pdf(tmp_path):
    criteria_set = _criteria_set()

    include_record = _record(id="rec-include")
    exclude_record = _record(id="rec-exclude")
    missing_record = _record(id="rec-missing")

    include_pdf = tmp_path / "include.pdf"
    include_pdf.write_bytes(b"content for the included record")
    exclude_pdf = tmp_path / "exclude.pdf"
    exclude_pdf.write_bytes(b"content for the excluded record")  # distinct bytes -- avoids duplicate_content_policy
    missing_pdf = tmp_path / "does_not_exist.pdf"  # never written -- input_unavailable

    items = [
        record_to_screening_item(include_record, pdf_path=include_pdf),
        record_to_screening_item(exclude_record, pdf_path=exclude_pdf),
        record_to_screening_item(missing_record, pdf_path=missing_pdf),
    ]
    status_by_key = {
        ("rec-include", "population"): "satisfied",
        ("rec-exclude", "population"): "violated",
    }
    run_result = _run_fake_screening(tmp_path, criteria_set=criteria_set, items=items, status_by_key=status_by_key)

    outcomes = map_run_result_to_chapter_outcomes(run_result)
    assert len(outcomes) == 3
    by_id = {o.record_id: o for o in outcomes}

    assert by_id["rec-include"].decision == "include"
    assert by_id["rec-include"].execution_status == "completed"
    assert by_id["rec-include"].primary_exclusion_reason_code is None

    assert by_id["rec-exclude"].decision == "exclude"
    assert by_id["rec-exclude"].primary_exclusion_reason_code == "wrong_population"
    assert by_id["rec-exclude"].exclusion_reason_codes == ["wrong_population"]

    assert by_id["rec-missing"].decision == "not_assessed"
    assert by_id["rec-missing"].execution_status == "input_unavailable"
    assert by_id["rec-missing"].ft_screening_result_id is None


def test_map_run_result_needs_human_review(tmp_path):
    criteria_set = _criteria_set()
    record = _record(id="rec-review")
    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]

    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items,
        status_by_key={("rec-review", "population"): "unclear"},
        requires_review_by_key={("rec-review", "population"): True},
    )
    outcomes = map_run_result_to_chapter_outcomes(run_result)
    assert outcomes[0].decision == "needs_human_review"
    assert outcomes[0].execution_status == "completed"


def test_needs_human_review_outcome_preserves_criterion_evidence_and_rationale(tmp_path):
    """codex review finding 1's exact repro: a human-review outcome must
    let a reviewer see WHY, not just THAT -- criterion-level status,
    rationale, requires_human_review, and the actual retrieved evidence
    excerpt must all survive the mapping, not just the top-level
    decision."""
    criteria_set = _criteria_set()
    record = _record(id="rec-review")
    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]

    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items,
        status_by_key={("rec-review", "population"): "unclear"},
        requires_review_by_key={("rec-review", "population"): True},
    )
    outcome = map_run_result_to_chapter_outcomes(run_result)[0]

    assessments = outcome.item_result.eligibility.assessments
    assert len(assessments) == 1
    assessment = assessments[0]
    assert assessment.criterion_id == "population"
    assert assessment.status == "unclear"
    assert assessment.requires_human_review is True
    assert "unclear" in assessment.rationale
    assert len(assessment.evidence) == 1
    assert "Fake parsed content for document rec-review" in assessment.evidence[0].excerpt

    # And it's reachable through the model_dump() Phase B will persist into
    # ReviewState, not just on the live Python object.
    dumped = outcome.model_dump(mode="json")
    dumped_assessment = dumped["item_result"]["eligibility"]["assessments"][0]
    assert dumped_assessment["requires_human_review"] is True
    assert dumped_assessment["rationale"] == assessment.rationale


def test_map_run_result_processing_failure_never_becomes_include(tmp_path):
    """Zero include-on-error: a parser failure must map to something
    OTHER than "include", the exact defect this integration must not
    inherit from fulltext.py's _fulltext_screen()."""
    criteria_set = _criteria_set()
    record = _record(id="rec-fails")
    pdf_path = tmp_path / "fails.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]

    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items, fail_document_ids={"rec-fails"},
    )
    outcomes = map_run_result_to_chapter_outcomes(run_result)
    assert outcomes[0].decision == "not_assessed"
    assert outcomes[0].decision != "include"
    assert outcomes[0].execution_status == "processing_failed"


def test_processing_failure_outcome_preserves_error_details(tmp_path):
    """codex review finding 1's second repro: a processing-failure
    outcome must preserve WHY it failed (code/stage/message/exception
    type), not just the fact that it failed."""
    criteria_set = _criteria_set()
    record = _record(id="rec-fails")
    pdf_path = tmp_path / "fails.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]

    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items, fail_document_ids={"rec-fails"},
    )
    outcome = map_run_result_to_chapter_outcomes(run_result)[0]

    errors = outcome.item_result.errors
    assert len(errors) == 1
    error = errors[0]
    assert error.code == "parse_failed"
    assert error.stage == "parse"
    assert error.severity == "fatal"
    assert error.exception_type == "RuntimeError"
    assert "rec-fails" in error.message

    dumped = outcome.model_dump(mode="json")
    dumped_error = dumped["item_result"]["errors"][0]
    assert dumped_error["code"] == "parse_failed"
    assert dumped_error["exception_type"] == "RuntimeError"


def test_outcome_carries_run_artifact_reference(tmp_path):
    """The complete, immutable, authoritative FT_SCREENING run files stay
    referenced by path -- never copied/simplified into this model (Phase
    B's own artifact-immutability requirement)."""
    criteria_set = _criteria_set()
    record = _record(id="rec-include")
    pdf_path = tmp_path / "include.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]

    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items,
        status_by_key={("rec-include", "population"): "satisfied"},
    )
    outcome = map_run_result_to_chapter_outcomes(run_result)[0]
    assert outcome.run_artifacts == run_result.artifacts
    assert outcome.run_artifacts.run_result_json.exists()
    assert outcome.run_artifacts.item_results_jsonl.exists()
    assert outcome.run_artifacts.ledger_jsonl.exists()


def test_chapter_full_text_outcome_derived_properties_are_frozen_and_live(tmp_path):
    criteria_set = _criteria_set()
    record = _record(id="rec-include")
    pdf_path = tmp_path / "include.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]
    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items,
        status_by_key={("rec-include", "population"): "satisfied"},
    )
    outcome = map_run_result_to_chapter_outcomes(run_result)[0]

    assert outcome.record_id == "rec-include"
    assert outcome.decision == "include"
    with pytest.raises(Exception):
        outcome.item_result = outcome.item_result  # frozen -- assignment must fail


def test_chapter_full_text_outcome_model_dump_carries_only_authoritative_fields(tmp_path):
    """codex review, round 3: the derived record_id/decision/execution_status/
    primary_exclusion_reason_code/exclusion_reason_codes/ft_screening_result_id
    must NOT appear in model_dump() output -- only the three fields that
    actually determine them (ft_screening_run_id/run_artifacts/item_result),
    so there is nothing duplicated/stale to persist into ReviewState."""
    criteria_set = _criteria_set()
    record = _record(id="rec-include")
    pdf_path = tmp_path / "include.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]
    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items,
        status_by_key={("rec-include", "population"): "satisfied"},
    )
    outcome = map_run_result_to_chapter_outcomes(run_result)[0]

    dumped = outcome.model_dump(mode="json")
    assert set(dumped.keys()) == {"ft_screening_run_id", "run_artifacts", "item_result"}


def test_chapter_full_text_outcome_serialization_round_trip(tmp_path):
    """codex review, round 3, exact regression: a persisted outcome must
    be reconstructable via model_validate(), not just dumpable -- this is
    the resume/artifact-loading path Phase B needs."""
    criteria_set = _criteria_set()
    record = _record(id="rec-include")
    pdf_path = tmp_path / "include.pdf"
    pdf_path.write_bytes(b"content")
    items = [record_to_screening_item(record, pdf_path=pdf_path)]
    run_result = _run_fake_screening(
        tmp_path, criteria_set=criteria_set, items=items,
        status_by_key={("rec-include", "population"): "satisfied"},
    )
    outcome = map_run_result_to_chapter_outcomes(run_result)[0]

    dumped = outcome.model_dump(mode="json")
    restored = ChapterFullTextOutcome.model_validate(dumped)

    assert restored.item_result == outcome.item_result
    assert restored.decision == outcome.decision
    assert restored.record_id == outcome.record_id


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        dict(record_id="r1", decision="include", execution_status="completed", ft_screening_run_id="run-1"),
        dict(decision="exclude", ft_screening_run_id="run-1"),
        dict(execution_status="processing_failed", ft_screening_run_id="run-1"),
    ],
)
def test_chapter_full_text_outcome_rejects_the_old_flat_field_shape(bad_kwargs):
    """codex review finding 2, verified structurally rather than via a
    validator: record_id/decision/execution_status/exclusion codes are no
    longer independently settable fields at all, so the exact
    "decision=include with execution_status=processing_failed" impossible
    combination the review flagged can't even be constructed -- it fails
    at the schema level (missing run_artifacts/item_result, and
    record_id/decision/execution_status rejected as unknown fields under
    extra="forbid"), not via a runtime check."""
    with pytest.raises(Exception):
        ChapterFullTextOutcome(**bad_kwargs)


# ---------------------------------------------------------------------------
# Static proof: the adapter MODULE's own source imports only the public
# FT_SCREENING surface (never ft_screening.orchestrator/.document/.fakes/
# etc.) -- a property of the production module, not of this test file.
# ---------------------------------------------------------------------------


def test_adapter_module_imports_only_public_ft_screening_surface():
    adapter_path = REPO_ROOT / "slr_pipeline" / "adapters" / "ft_screening_adapter.py"
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ft_screening"):
            assert node.module == "ft_screening", (
                f"adapter module imports from {node.module!r} -- only bare 'ft_screening' "
                "(the public top-level package) is allowed"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ft_screening.") and alias.name != "ft_screening":
                    pytest.fail(f"adapter module does `import {alias.name}` -- not the public surface")


def test_adapter_module_creates_no_files_at_import_time(tmp_path, monkeypatch):
    """Loads a SEPARATE, isolated copy of the module (never
    importlib.reload() on the canonical, already-imported
    `slr_pipeline.adapters.ft_screening_adapter` sys.modules entry) --
    reload() re-executes the module's top-level `class` statements,
    minting a NEW `ChapterAdapterError` (and every other class defined
    here) that is a distinct object from the one every other test file in
    this session already imported and bound at collection time. That
    broke `pytest.raises(ChapterAdapterError)` in
    tests/nodes/test_fulltext_node.py the first time this test ran before
    it in the same session -- isinstance/except stopped matching because
    the two "ChapterAdapterError" names no longer referred to the same
    class. An isolated load under a throwaway sys.modules key proves the
    same "import performs no I/O" property without that cross-test-file
    side effect."""
    import importlib.util

    monkeypatch.chdir(tmp_path)
    adapter_path = REPO_ROOT / "slr_pipeline" / "adapters" / "ft_screening_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "_test_isolated_copy_of_ft_screening_adapter", adapter_path,
    )
    isolated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated_module)
    assert list(tmp_path.iterdir()) == []
