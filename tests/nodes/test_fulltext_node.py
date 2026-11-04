"""Offline tests for slr_pipeline.nodes.fulltext.fulltext_node (Phase B).

Two boundaries are monkeypatched, and ONLY these two -- everything else
(acquisition routing, stub-detection string logic, the FT_SCREENING
request/outcome mapping) runs for real:
  - `_extract_text` (shells out to pymupdf4llm against a real file) --
    replaced with a controllable in-memory string so tests don't need a
    real, parseable PDF fixture.
  - `run_ft_screening` (the real FT_SCREENING public entry point --
    Docling/OpenAI/network/credentials) -- replaced with a factory that
    forwards the REAL request into FT_SCREENING's own private
    `_run_ft_screening` using fakes (`ft_screening.fakes`), the exact
    same technique used in tests/adapters/test_ft_screening_adapter.py.

No network calls happen anywhere in this file: every test record has
`doi=None` and no arxiv source, so `_resolve_oa_url` returns ("", "")
immediately without ever calling urllib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from slr_pipeline.adapters.ft_screening_adapter import ChapterAdapterError, record_to_screening_item
from slr_pipeline.config import ReviewConfig
from slr_pipeline.selection_ledger import SelectionDecisionStore, SelectionLedgerPreconditionError
from slr_pipeline.state import PRISMACounts, Record, initial_state
from slr_pipeline.human_resolution import StaleReviewError

from ft_screening import CriteriaSet, CriterionSpec, ScreeningConfig

import slr_pipeline.nodes.fulltext as fulltext_module
from slr_pipeline.nodes.fulltext import fulltext_node, resolve_full_text_review_queue

_REAL_TEXT = "This is real extracted full-text content. " * 30  # well over 800 chars, no stub markers
_STUB_TEXT = "This is a preview only. " * 5  # trips the "preview only" stub marker


def _criteria_set() -> CriteriaSet:
    return CriteriaSet(
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


def _config(tmp_path, **overrides) -> ReviewConfig:
    defaults = dict(
        title="Test review",
        output_dir=str(tmp_path / "outputs"),
        full_text_criteria_set=_criteria_set().model_dump(mode="json"),
    )
    defaults.update(overrides)
    return ReviewConfig(**defaults)


def _record(**overrides) -> Record:
    defaults = dict(id="rec-0001", source="test", source_id="0001", doi=None, title="A study", abstract=None)
    defaults.update(overrides)
    return Record(**defaults)


def _state(cfg: ReviewConfig, records: list[Record]) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["included_records"] = [r.to_dict() for r in records]
    # Stage 3: fulltext_node/resolve_full_text_review_queue() now resolve
    # review_run_id via slr_pipeline.run_identity -- these tests call node
    # functions directly (no LangGraph invocation, no thread_id), so the
    # fixture supplies one explicitly.
    st["review_run_id"] = "fulltext-test-run"
    return st


def _place_local_pdf(pdf_dir: Path, record: Record, content: bytes) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / f"{record.source}_{record.source_id}.pdf"
    dest.write_bytes(content)
    return dest


def _fake_run_ft_screening(
    *, status_by_key=None, requires_review_by_key=None, fail_document_ids=None, calls=None, call_log=None,
):
    """Factory: returns a drop-in replacement for ft_screening.run_ft_screening
    that forwards the REAL request into FT_SCREENING's own private
    _run_ft_screening using fakes -- a fully real, schema-valid
    ScreeningRunResult, not a hand-rolled stand-in."""

    def _fake(request):
        from ft_screening.components import Components
        from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
        from ft_screening.orchestrator import _run_ft_screening
        from ft_screening.store import FileScreeningStore

        if calls is not None:
            calls.append(request)
        # The orchestrator's own boundary check (finding 4, FT_SCREENING's
        # own review history) rejects an adjudicator whose reported
        # model/prompt/schema identity doesn't match request.config's --
        # the fake must echo request.config's REAL declared identity
        # (chapter_screening_config()'s gpt-5.6-terra/prompt-6/schema-2),
        # not its own default "fake-1"/"1"/"1".
        components = Components(
            parser=FakeDocumentParser(fail_document_ids=fail_document_ids or set()),
            retriever=FakeEvidenceRetriever(),
            adjudicator=FakeCriterionAdjudicator(
                status_by_key=status_by_key or {}, requires_review_by_key=requires_review_by_key or {},
                model=request.config.adjudicator_model,
                prompt_template_version=request.config.prompt_template_version,
                output_schema_version=request.config.output_schema_version,
                call_log=call_log,
            ),
            store=FileScreeningStore(output_dir=request.output_dir),
        )
        return _run_ft_screening(request, components)

    return _fake


def _always_real_text(pdf_path, max_chars=120_000):
    return _REAL_TEXT


# ---------------------------------------------------------------------------
# include / exclude / needs_human_review -- one record each, in one run.
# ---------------------------------------------------------------------------


def test_fulltext_node_include_exclude_and_needs_human_review(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    include_rec = _record(id="rec-include", source_id="include")
    exclude_rec = _record(id="rec-exclude", source_id="exclude")
    review_rec = _record(id="rec-review", source_id="review")

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, include_rec, b"include content A")
    _place_local_pdf(pdf_dir, exclude_rec, b"exclude content B")
    _place_local_pdf(pdf_dir, review_rec, b"review content C")

    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={
                ("rec-include", "population"): "satisfied",
                ("rec-exclude", "population"): "violated",
                ("rec-review", "population"): "unclear",
            },
            requires_review_by_key={("rec-review", "population"): True},
        ),
    )

    state = _state(cfg, [include_rec, exclude_rec, review_rec])
    result = fulltext_node(state)

    included_ids = {r["id"] for r in result["full_text_records"]}
    assert included_ids == {"rec-include"}

    queue_by_id = {e["record_id"]: e for e in result["full_text_review_queue"]}
    assert queue_by_id["rec-review"]["stage"] == "ft_screening"
    assert queue_by_id["rec-review"]["item_result"]["execution_status"] == "completed"
    assert "rec-exclude" not in queue_by_id  # excluded records aren't queued, only counted

    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.assessed_full_text == 3
    assert prisma.excluded_full_text == 1
    assert prisma.excluded_full_text_reasons == {"wrong_population": 1}

    assert result["ft_screening_run_id"] is not None
    assert result["ft_screening_execution_fingerprint"] is not None
    assert result["ft_screening_artifact_paths"]["run_result_json"]


# ---------------------------------------------------------------------------
# missing PDF -- never reaches FT_SCREENING at all.
# ---------------------------------------------------------------------------


def test_fulltext_node_missing_pdf_never_reaches_ft_screening(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    calls = []
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening(calls=calls))

    missing_rec = _record(id="rec-missing", source_id="missing")  # no local pdf placed, no doi
    cfg = _config(tmp_path)
    state = _state(cfg, [missing_rec])
    result = fulltext_node(state)

    assert result["full_text_records"] == []
    assert calls == []  # FT_SCREENING never invoked -- nothing was confirmed

    queue = result["full_text_review_queue"]
    assert len(queue) == 1
    assert queue[0]["stage"] == "acquisition"
    assert queue[0]["record_id"] == "rec-missing"
    assert queue[0]["full_text_status"] == "no_doi"

    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.assessed_full_text == 0
    assert prisma.fulltext_retrieval == {"no_doi": 1}
    assert result["ft_screening_run_id"] is None


# ---------------------------------------------------------------------------
# full_text_acquisition_failure_policy == "exclude" -- terminal "not
# retrieved" outcome instead of a full_text_review_queue entry (required
# for a fully autonomous run; see config.py's field docstring and
# track3_run_child.py's hard abort on a non-empty full_text_review_queue).
# This must NOT increment PRISMACounts.excluded_full_text -- that box is
# reserved for FT_SCREENING-evaluated exclusions with a ledger decision
# behind them (see reporting.py's ledger-vs-PRISMA reconciliation), and an
# acquisition failure is already fully represented via
# PRISMACounts.fulltext_retrieval.
# ---------------------------------------------------------------------------


def test_fulltext_node_acquisition_failure_policy_exclude_records_prisma_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    calls = []
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening(calls=calls))

    missing_rec = _record(id="rec-missing", source_id="missing")  # no local pdf placed, no doi
    cfg = _config(tmp_path, full_text_acquisition_failure_policy="exclude")
    state = _state(cfg, [missing_rec])
    result = fulltext_node(state)

    assert result["full_text_records"] == []
    assert calls == []  # FT_SCREENING never invoked -- nothing was confirmed

    # The defining behavior: NOT queued, unlike the default "review" policy.
    assert result["full_text_review_queue"] == []
    assert result["requires_human_review"] is False

    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.assessed_full_text == 0
    assert prisma.fulltext_retrieval == {"no_doi": 1}
    # No PRISMA exclusion -- this is a "not retrieved" terminal outcome,
    # not an evaluated-but-ineligible one.
    assert prisma.excluded_full_text == 0
    assert prisma.excluded_full_text_reasons == {}


def test_fulltext_node_acquisition_failure_policy_exclude_covers_stub_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", lambda pdf_path, max_chars=120_000: _STUB_TEXT)
    calls = []
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening(calls=calls))

    stub_rec = _record(id="rec-stub", source_id="stub")
    cfg = _config(tmp_path, full_text_acquisition_failure_policy="exclude")
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, stub_rec, b"stub content")

    state = _state(cfg, [stub_rec])
    result = fulltext_node(state)

    assert calls == []
    assert result["full_text_review_queue"] == []
    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.fulltext_retrieval == {"rejected_stub": 1}
    assert prisma.excluded_full_text == 0
    assert prisma.excluded_full_text_reasons == {}


def test_fulltext_node_acquisition_failure_policy_exclude_combines_with_ft_screening_exclusions(
    tmp_path, monkeypatch,
):
    """Mixed batch: one acquisition-stage failure (policy="exclude", never
    reaches FT_SCREENING) plus one real FT_SCREENING exclusion in the same
    run -- only the genuine FT_SCREENING exclusion counts toward
    excluded_full_text/excluded_full_text_reasons; the acquisition failure
    is represented solely via fulltext_retrieval. Queue stays empty (full
    autonomy) either way."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    missing_rec = _record(id="rec-missing", source_id="missing")
    exclude_rec = _record(id="rec-exclude", source_id="exclude")
    cfg = _config(tmp_path, full_text_acquisition_failure_policy="exclude")
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, exclude_rec, b"exclude content")

    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-exclude", "population"): "violated"}),
    )

    state = _state(cfg, [missing_rec, exclude_rec])
    result = fulltext_node(state)

    assert result["full_text_review_queue"] == []
    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.fulltext_retrieval == {"no_doi": 1, "parsed": 1}
    assert prisma.excluded_full_text == 1
    assert prisma.excluded_full_text_reasons == {"wrong_population": 1}


# ---------------------------------------------------------------------------
# parser/provider failure (FT_SCREENING processing_failed) -- never becomes
# include.
# ---------------------------------------------------------------------------


def test_fulltext_node_processing_failure_never_becomes_include(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    fails_rec = _record(id="rec-fails", source_id="fails")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, fails_rec, b"fails content")

    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(fail_document_ids={"rec-fails"}),
    )

    state = _state(cfg, [fails_rec])
    result = fulltext_node(state)

    assert result["full_text_records"] == []
    queue = result["full_text_review_queue"]
    assert len(queue) == 1
    assert queue[0]["stage"] == "ft_screening"
    assert queue[0]["item_result"]["execution_status"] == "processing_failed"
    assert queue[0]["item_result"]["errors"][0]["code"] == "parse_failed"


# ---------------------------------------------------------------------------
# Stub detection preserved.
# ---------------------------------------------------------------------------


def test_fulltext_node_stub_detection_prevents_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", lambda pdf_path, max_chars=120_000: _STUB_TEXT)
    calls = []
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening(calls=calls))

    stub_rec = _record(id="rec-stub", source_id="stub")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, stub_rec, b"stub content")

    state = _state(cfg, [stub_rec])
    result = fulltext_node(state)

    assert calls == []
    queue = result["full_text_review_queue"]
    assert queue[0]["full_text_status"] == "rejected_stub"
    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.fulltext_retrieval == {"rejected_stub": 1}


# ---------------------------------------------------------------------------
# Exact PRISMA counts across a mixed batch.
# ---------------------------------------------------------------------------


def test_fulltext_node_exact_prisma_counts_mixed_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    include_rec = _record(id="rec-include", source_id="include")
    exclude_rec = _record(id="rec-exclude", source_id="exclude")
    missing_rec = _record(id="rec-missing", source_id="missing")

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, include_rec, b"include bytes")
    _place_local_pdf(pdf_dir, exclude_rec, b"exclude bytes")

    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={
            ("rec-include", "population"): "satisfied",
            ("rec-exclude", "population"): "violated",
        }),
    )

    state = _state(cfg, [include_rec, exclude_rec, missing_rec])
    result = fulltext_node(state)

    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.assessed_full_text == 2          # only the two confirmed PDFs, NOT all 3 records
    assert prisma.excluded_full_text == 1
    assert prisma.excluded_full_text_reasons == {"wrong_population": 1}
    assert prisma.fulltext_retrieval == {"parsed": 2, "no_doi": 1}
    assert len(result["full_text_records"]) == 1
    assert len(result["full_text_review_queue"]) == 1  # the missing-PDF record only


# ---------------------------------------------------------------------------
# Zero abstract fallback: a record with no confirmed PDF structurally never
# reaches FT_SCREENING, abstract or not.
# ---------------------------------------------------------------------------


def test_fulltext_node_zero_abstract_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    confirmed_rec = _record(id="rec-confirmed", source_id="confirmed")
    distinctive_abstract = "UNIQUE_ABSTRACT_MARKER_should_never_reach_screening"
    unconfirmed_rec = _record(id="rec-unconfirmed", source_id="unconfirmed", abstract=distinctive_abstract)

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, confirmed_rec, b"confirmed bytes")
    # unconfirmed_rec: no local pdf, no doi -- acquisition never confirms it

    calls = []
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-confirmed", "population"): "satisfied"}, calls=calls),
    )

    state = _state(cfg, [confirmed_rec, unconfirmed_rec])
    fulltext_node(state)

    assert len(calls) == 1
    sent_document_ids = {item.document_id for item in calls[0].items}
    assert sent_document_ids == {"rec-confirmed"}  # the unconfirmed record's abstract never entered the request


# ---------------------------------------------------------------------------
# Duplicate record ids -- FT_SCREENING's own ScreeningRequest validator
# raises, unwrapped, propagating out of the node.
# ---------------------------------------------------------------------------


def test_fulltext_node_duplicate_record_ids_propagate_ft_screening_validation_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening())

    rec_a = _record(id="rec-dup", source="test", source_id="dup_a")
    rec_b = _record(id="rec-dup", source="test", source_id="dup_b")  # same id, different source_id

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec_a, b"dup a bytes")
    _place_local_pdf(pdf_dir, rec_b, b"dup b bytes")

    state = _state(cfg, [rec_a, rec_b])
    with pytest.raises(Exception):  # pydantic.ValidationError from ScreeningRequest itself
        fulltext_node(state)


# ---------------------------------------------------------------------------
# Unsafe record ids -- ChapterAdapterError, from record_to_screening_item()
# via _run_full_text_eligibility(), unwrapped.
# ---------------------------------------------------------------------------


def test_fulltext_node_unsafe_record_id_raises_chapter_adapter_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening())

    unsafe_rec = _record(id="has spaces/and:slash", source="test", source_id="unsafe")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, unsafe_rec, b"unsafe bytes")

    state = _state(cfg, [unsafe_rec])
    with pytest.raises(ChapterAdapterError):
        fulltext_node(state)


# ---------------------------------------------------------------------------
# Missing full_text_criteria_set -- fails loudly rather than silently
# skipping eligibility screening.
# ---------------------------------------------------------------------------


def test_fulltext_node_requires_explicit_criteria_set_when_pdfs_are_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    rec = _record(id="rec-a", source_id="a")
    cfg = _config(tmp_path, full_text_criteria_set=None)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"content")

    state = _state(cfg, [rec])
    with pytest.raises(ValueError, match="full_text_criteria_set"):
        fulltext_node(state)


def test_fulltext_node_skips_ft_screening_entirely_when_nothing_confirmed_even_without_criteria_set(tmp_path, monkeypatch):
    """No confirmed PDF at all -> full_text_criteria_set is never even
    consulted; nothing needs to be screened."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    rec = _record(id="rec-a", source_id="a")  # no local pdf, no doi
    cfg = _config(tmp_path, full_text_criteria_set=None)
    state = _state(cfg, [rec])
    result = fulltext_node(state)  # must not raise
    assert result["full_text_records"] == []


# ---------------------------------------------------------------------------
# requires_human_review flag (codex review, Phase B finding 1).
# ---------------------------------------------------------------------------


def test_fulltext_node_sets_requires_human_review_when_queue_nonempty(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-review", "population"): "unclear"},
                                requires_review_by_key={("rec-review", "population"): True}),
    )
    rec = _record(id="rec-review", source_id="review")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"review bytes")

    result = fulltext_node(_state(cfg, [rec]))
    assert result["requires_human_review"] is True


def test_fulltext_node_requires_human_review_false_when_all_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-include", "population"): "satisfied"}),
    )
    rec = _record(id="rec-include", source_id="include")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"include bytes")

    result = fulltext_node(_state(cfg, [rec]))
    assert result["requires_human_review"] is False


# ---------------------------------------------------------------------------
# Explicit empty/malformed screening-config override must fail loudly, not
# silently fall back to the chapter preset (codex review, Phase B finding 3).
# ---------------------------------------------------------------------------


def test_fulltext_node_rejects_explicit_empty_screening_config_override(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    rec = _record(id="rec-a", source_id="a")
    cfg = _config(tmp_path, full_text_screening_config={})  # explicit, empty -- must NOT silently default
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"content")

    state = _state(cfg, [rec])
    with pytest.raises(Exception):  # pydantic.ValidationError -- {} is missing every required ScreeningConfig field
        fulltext_node(state)


def test_fulltext_node_none_screening_config_still_uses_chapter_preset(tmp_path, monkeypatch):
    """None (the actual default) is the ONE value that still means "use
    the chapter preset" -- distinguishing this from the rejected {} case
    above proves the fix checks `is not None`, not truthiness."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    calls = []
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening(
        status_by_key={("rec-a", "population"): "satisfied"}, calls=calls,
    ))
    rec = _record(id="rec-a", source_id="a")
    cfg = _config(tmp_path, full_text_screening_config=None)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"content")

    fulltext_node(_state(cfg, [rec]))
    assert calls[0].config.adjudicator_model == "gpt-5.6-terra"  # chapter_screening_config()'s value


# ---------------------------------------------------------------------------
# Stable run identity + resume: an interrupted run's already-completed
# items must not be re-called on retry (codex review, Phase B finding 2).
# ---------------------------------------------------------------------------


def test_fulltext_node_reuses_same_run_id_across_identical_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-a", "population"): "satisfied"}),
    )
    rec = _record(id="rec-a", source_id="a")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"content")

    state = _state(cfg, [rec])
    first = fulltext_node(state)
    second = fulltext_node(state)  # identical retry -- e.g. after a crash
    assert first["ft_screening_run_id"] == second["ft_screening_run_id"]


def test_fulltext_node_resume_does_not_recall_already_completed_items(tmp_path, monkeypatch):
    """The exact scenario the review named: a process interruption after
    FT_SCREENING durably persisted some items but before this node
    returned to LangGraph. Retrying with the IDENTICAL confirmed batch
    must resume (same run_id) and must NOT re-call the adjudicator for an
    item that already has a completed result."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)

    rec1 = _record(id="rec-1", source_id="one")
    rec2 = _record(id="rec-2", source_id="two")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec1, b"one bytes")
    _place_local_pdf(pdf_dir, rec2, b"two bytes")

    call_log: list = []
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-1", "population"): "satisfied", ("rec-2", "population"): "satisfied"},
            call_log=call_log,
        ),
    )

    state = _state(cfg, [rec1, rec2])
    first = fulltext_node(state)
    first_call_count = len(call_log)
    assert first_call_count == 2  # both items adjudicated once each

    # Identical retry (same criteria_set/config/items) -- simulates the
    # pipeline being re-invoked after an interruption, once FT_SCREENING's
    # own durable results from the first attempt are already on disk.
    second = fulltext_node(state)

    assert second["ft_screening_run_id"] == first["ft_screening_run_id"]
    rec1_calls = [c for c in call_log if c[0] == ("rec-1", "population")]
    rec2_calls = [c for c in call_log if c[0] == ("rec-2", "population")]
    assert len(rec1_calls) == 1  # NOT called again on the resumed run
    assert len(rec2_calls) == 1
    assert len(second["full_text_records"]) == 2


def test_fulltext_node_different_criteria_set_gets_a_different_run_id(tmp_path, monkeypatch):
    """A genuinely different configuration must NOT collide with (or try
    to resume) an unrelated prior run under the same output_dir."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-a", "population"): "satisfied"}),
    )
    rec = _record(id="rec-a", source_id="a")
    pdf_dir_cfg = _config(tmp_path)
    pdf_dir = Path(pdf_dir_cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec, b"content")

    other_criteria = CriteriaSet(
        criteria_set_id="a-different-criteria-set", version="1",
        criteria=[CriterionSpec(
            criterion_id="design", version="1", eligibility_requirement="RCT design",
            missing_evidence_policy="review", exclusion_reason_code="wrong_design",
            exclusion_reason_label="Wrong design", priority=1,
        )],
    )
    cfg2 = _config(tmp_path, full_text_criteria_set=other_criteria.model_dump(mode="json"))

    result1 = fulltext_node(_state(pdf_dir_cfg, [rec]))
    result2 = fulltext_node(_state(cfg2, [rec]))
    assert result1["ft_screening_run_id"] != result2["ft_screening_run_id"]


# ---------------------------------------------------------------------------
# _stable_run_id() -- must cover the COMPLETE request identity FT_SCREENING
# itself binds to (document_id/report_id/study_id/citation/pdf path/pdf
# content), not just (record.id, pdf_sha256) (codex review, Phase B
# round-3: the round-2 hash still missed fields FT_SCREENING's own
# request_fingerprint/provisional binding cover, so a citation correction
# -- e.g. a fixed title/doi/year -- silently computed the SAME run_id,
# which FT_SCREENING would then correctly reject as an incompatible
# resume).
# ---------------------------------------------------------------------------


def _fake_screening_config() -> ScreeningConfig:
    return ScreeningConfig(
        parser_name="fake", parser_version="1", chunker_version="1",
        retriever_name="fake", retriever_config_version="1",
        adjudicator_provider="fake", adjudicator_model="fake-1", model_config_id="cfg-1",
        prompt_template_version="1", output_schema_version="1", aggregation_policy_version="1",
    )


def test_stable_run_id_changes_when_citation_changes(tmp_path):
    """The reviewer's exact repro: identical record.id and identical PDF
    bytes, but a corrected title (title lives inside ScreeningItem.citation,
    which FT_SCREENING's own request_fingerprint includes in full)."""
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"identical bytes")
    criteria_set = _criteria_set()
    config = _fake_screening_config()

    original = record_to_screening_item(_record(id="rec-a", title="Original title"), pdf_path=pdf_path)
    corrected = record_to_screening_item(_record(id="rec-a", title="Corrected title"), pdf_path=pdf_path)

    run_id_original = fulltext_module._stable_run_id([original], criteria_set, config)
    run_id_corrected = fulltext_module._stable_run_id([corrected], criteria_set, config)
    assert run_id_original != run_id_corrected


@pytest.mark.parametrize("field,value", [("doi", "10.1000/corrected"), ("year", 2099), ("source_id", "corrected-id")])
def test_stable_run_id_changes_when_any_citation_field_changes(tmp_path, field, value):
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"identical bytes")
    criteria_set = _criteria_set()
    config = _fake_screening_config()

    base_kwargs = dict(id="rec-a", doi="10.1000/original", year=2020, source_id="0001")
    original = record_to_screening_item(_record(**base_kwargs), pdf_path=pdf_path)
    changed_kwargs = {**base_kwargs, field: value}
    changed = record_to_screening_item(_record(**changed_kwargs), pdf_path=pdf_path)

    run_id_original = fulltext_module._stable_run_id([original], criteria_set, config)
    run_id_changed = fulltext_module._stable_run_id([changed], criteria_set, config)
    assert run_id_original != run_id_changed


def test_stable_run_id_changes_when_same_bytes_move_to_a_different_path(tmp_path):
    criteria_set = _criteria_set()
    config = _fake_screening_config()
    rec = _record(id="rec-a")

    path_a = tmp_path / "location_a" / "a.pdf"
    path_a.parent.mkdir(parents=True, exist_ok=True)
    path_a.write_bytes(b"identical bytes")
    path_b = tmp_path / "location_b" / "a.pdf"
    path_b.parent.mkdir(parents=True, exist_ok=True)
    path_b.write_bytes(b"identical bytes")  # SAME content, different path

    item_a = record_to_screening_item(rec, pdf_path=path_a)
    item_b = record_to_screening_item(rec, pdf_path=path_b)

    run_id_a = fulltext_module._stable_run_id([item_a], criteria_set, config)
    run_id_b = fulltext_module._stable_run_id([item_b], criteria_set, config)
    assert run_id_a != run_id_b


def test_stable_run_id_identical_for_a_completely_identical_request(tmp_path):
    criteria_set = _criteria_set()
    config = _fake_screening_config()
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"identical bytes")
    rec = _record(id="rec-a", doi="10.1000/x", title="Same title", year=2020)

    item_1 = record_to_screening_item(rec, pdf_path=pdf_path)
    item_2 = record_to_screening_item(rec, pdf_path=pdf_path)  # rebuilt fresh, same everything

    run_id_1 = fulltext_module._stable_run_id([item_1], criteria_set, config)
    run_id_2 = fulltext_module._stable_run_id([item_2], criteria_set, config)
    assert run_id_1 == run_id_2


def test_fulltext_node_growing_confirmed_batch_completes_rather_than_conflicting(tmp_path, monkeypatch):
    """codex review, Phase B round-2 finding 1's exact repro: PDF A is
    confirmed and screened; PDF B is unavailable and lands in the review
    queue; the human later supplies PDF B via the Tier-3 manual worklist;
    fulltext_node is re-invoked with A+B. The run identity must reflect
    the GROWN batch (a different run_id from the A-only attempt) so this
    completes as a fresh run rather than FT_SCREENING correctly rejecting
    it as an incompatible resume of the A-only run (ResumeConflictError).
    Reprocessing A is accepted here (the review's own stated acceptable
    baseline)."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    call_log: list = []
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-a", "population"): "satisfied", ("rec-b", "population"): "satisfied"},
            call_log=call_log,
        ),
    )
    rec_a = _record(id="rec-a", source_id="a")
    rec_b = _record(id="rec-b", source_id="b")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec_a, b"A bytes")
    # rec_b: no local PDF yet on the first pass -- unavailable.

    result1 = fulltext_node(_state(cfg, [rec_a, rec_b]))
    assert {r["id"] for r in result1["full_text_records"]} == {"rec-a"}
    assert any(e["record_id"] == "rec-b" and e["stage"] == "acquisition" for e in result1["full_text_review_queue"])

    # Tier 3: a human supplies rec_b's PDF later.
    _place_local_pdf(pdf_dir, rec_b, b"B bytes")

    result2 = fulltext_node(_state(cfg, [rec_a, rec_b]))  # must NOT raise ResumeConflictError
    assert {r["id"] for r in result2["full_text_records"]} == {"rec-a", "rec-b"}
    assert result2["ft_screening_run_id"] != result1["ft_screening_run_id"]  # different batch -> different run


# ---------------------------------------------------------------------------
# resolve_full_text_review_queue() -- the resolution/resume path a human
# uses to clear queued records (codex review, Phase B finding 1).
# ---------------------------------------------------------------------------


def _resolution(action: str, **extra) -> dict:
    base = dict(action=action, reviewer="R. Rauleckas", note="manual review", resolved_at="2026-08-21T12:00:00Z")
    base.update(extra)
    return base


def _ft_screening_queue_entry(record: Record, *, reviewed_baseline_decision_id=None) -> dict:
    """A stage="ft_screening" queue entry with the mutated record (as
    fulltext_node itself would preserve it -- full_text_path set).
    `reviewed_baseline_decision_id` is None by default (no real ai ledger
    entry exists for these hand-built, resolve_full_text_review_queue()-
    only fixtures) -- the key must still be explicitly PRESENT (Stage 3
    requires it), just as fulltext_node's own real queue entries always
    carry it."""
    mutated = record.model_copy(update={"full_text_path": f"/pdfs/{record.id}.pdf", "full_text_status": "parsed"})
    return {
        "stage": "ft_screening", "record_id": record.id, "record": mutated.to_dict(),
        "reviewed_baseline_decision_id": reviewed_baseline_decision_id,
    }


def test_resolve_full_text_review_queue_ft_screening_include_preserves_full_text_path(tmp_path):
    rec = _record(id="rec-review", source_id="review", title="Queued study")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [_ft_screening_queue_entry(rec)]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts(assessed_full_text=1).model_dump()

    update = resolve_full_text_review_queue(state, {"rec-review": _resolution("include")})
    assert [r["id"] for r in update["full_text_records"]] == ["rec-review"]
    assert update["full_text_records"][0]["full_text_path"] == "/pdfs/rec-review.pdf"
    assert update["full_text_records"][0]["full_text_status"] == "parsed"
    assert update["full_text_review_queue"] == []
    assert update["requires_human_review"] is False
    assert update["full_text_review_resolutions"][0]["action"] == "include"
    assert update["full_text_review_resolutions"][0]["reviewer"] == "R. Rauleckas"


def test_resolve_full_text_review_queue_ft_screening_exclude_requires_reason(tmp_path):
    rec = _record(id="rec-review", source_id="review")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [_ft_screening_queue_entry(rec)]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts(assessed_full_text=1).model_dump()

    with pytest.raises(ValueError, match="reason"):
        resolve_full_text_review_queue(state, {"rec-review": _resolution("exclude")})  # no "reason"

    update = resolve_full_text_review_queue(
        state, {"rec-review": _resolution("exclude", reason="does not meet inclusion criteria on manual read")},
    )
    assert update["full_text_records"] == []
    assert update["full_text_review_queue"] == []
    prisma = PRISMACounts(**update["prisma_counts"])
    assert prisma.excluded_full_text == 1
    assert prisma.excluded_full_text_reasons == {"does not meet inclusion criteria on manual read": 1}
    assert update["full_text_review_resolutions"][0]["reason"] == "does not meet inclusion criteria on manual read"


def test_resolve_full_text_review_queue_acquisition_stage_rejects_include_and_exclude(tmp_path):
    """A record whose PDF was never retrieved has no assessed report --
    eligibility include/exclude must be structurally refused, not just
    discouraged (codex review, Phase B round-2 finding 2's exact repro)."""
    rec = _record(id="rec-missing", source_id="missing")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [{"stage": "acquisition", "record_id": "rec-missing", "full_text_status": "no_doi"}]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts().model_dump()

    with pytest.raises(ValueError, match="not_retrieved"):
        resolve_full_text_review_queue(state, {"rec-missing": _resolution("include")})
    with pytest.raises(ValueError, match="not_retrieved"):
        resolve_full_text_review_queue(state, {"rec-missing": _resolution("exclude", reason="x")})


def test_resolve_full_text_review_queue_acquisition_stage_not_retrieved_makes_no_prisma_change(tmp_path):
    rec = _record(id="rec-missing", source_id="missing")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [{"stage": "acquisition", "record_id": "rec-missing", "full_text_status": "no_doi"}]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts(fulltext_retrieval={"no_doi": 1}).model_dump()

    update = resolve_full_text_review_queue(state, {"rec-missing": _resolution("not_retrieved")})
    assert update["full_text_records"] == []
    assert update["full_text_review_queue"] == []
    prisma = PRISMACounts(**update["prisma_counts"])
    assert prisma.excluded_full_text == 0  # no eligibility fact was ever established
    assert prisma.assessed_full_text == 0
    assert update["full_text_review_resolutions"][0]["action"] == "not_retrieved"


def test_resolve_full_text_review_queue_partial_resolution_leaves_the_rest_queued(tmp_path):
    rec1 = _record(id="rec-1", source_id="one")
    rec2 = _record(id="rec-2", source_id="two")
    state = _state(_config(tmp_path), [rec1, rec2])
    state["full_text_review_queue"] = [
        _ft_screening_queue_entry(rec1),
        {"stage": "acquisition", "record_id": "rec-2", "full_text_status": "no_doi"},
    ]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts(assessed_full_text=1).model_dump()

    update = resolve_full_text_review_queue(state, {"rec-1": _resolution("include")})
    assert [r["id"] for r in update["full_text_records"]] == ["rec-1"]
    assert [e["record_id"] for e in update["full_text_review_queue"]] == ["rec-2"]
    assert update["requires_human_review"] is True


def test_resolve_full_text_review_queue_rejects_unknown_record_id(tmp_path):
    rec = _record(id="rec-review", source_id="review")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [_ft_screening_queue_entry(rec)]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts().model_dump()

    with pytest.raises(KeyError):
        resolve_full_text_review_queue(state, {"not-queued": _resolution("include")})


def test_resolve_full_text_review_queue_rejects_invalid_action(tmp_path):
    rec = _record(id="rec-review", source_id="review")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [_ft_screening_queue_entry(rec)]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts().model_dump()

    with pytest.raises(ValueError):
        resolve_full_text_review_queue(state, {"rec-review": _resolution("maybe")})


def test_resolve_full_text_review_queue_requires_reviewer_note_and_timestamp(tmp_path):
    rec = _record(id="rec-review", source_id="review")
    state = _state(_config(tmp_path), [rec])
    state["full_text_review_queue"] = [_ft_screening_queue_entry(rec)]
    state["full_text_records"] = []
    state["prisma_counts"] = PRISMACounts().model_dump()

    with pytest.raises(ValueError):
        resolve_full_text_review_queue(state, {"rec-review": {"action": "include"}})  # missing reviewer/note/resolved_at


# ---------------------------------------------------------------------------
# Stage 3 (the Stage 3 node-instrumentation design record): the three authoritative
# full-text ledger scenarios named explicitly in the plan's own acceptance
# matrix -- completed outcome -> ai baseline -> human supersession;
# non-completed outcome -> no ai entry -> human root; growing confirmed
# batch -> one connected chain, not disconnected roots. Plus stale-baseline
# rejection via resolve_full_text_review_queue().
# ---------------------------------------------------------------------------


def _ledger_store(cfg: ReviewConfig, review_run_id: str) -> SelectionDecisionStore:
    return SelectionDecisionStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)


def test_completed_outcome_gets_ai_baseline_then_human_supersession(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    review_rec = _record(id="rec-review", source_id="review")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, review_rec, b"review content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-review", "population"): "unclear"},
            requires_review_by_key={("rec-review", "population"): True},
        ),
    )

    state = _state(cfg, [review_rec])
    result = fulltext_node(state)
    queue = {e["record_id"]: e for e in result["full_text_review_queue"]}
    entry = queue["rec-review"]
    assert entry["stage"] == "ft_screening"
    baseline_id = entry["reviewed_baseline_decision_id"]
    assert baseline_id is not None

    store = _ledger_store(cfg, state["review_run_id"])
    tail = store.project_current_tails()[("rec-review", "full_text_eligibility")]
    assert tail.decision_id == baseline_id
    assert tail.decision_maker == "ai"
    assert tail.decision == "needs_human_review"

    resolution_update = resolve_full_text_review_queue(
        {**state, **result}, {"rec-review": _resolution("include")},
    )
    assert [r["id"] for r in resolution_update["full_text_records"]] == ["rec-review"]

    tail = store.project_current_tails()[("rec-review", "full_text_eligibility")]
    assert tail.decision_maker == "human"
    assert tail.decision == "include"
    assert tail.supersedes_decision_id == baseline_id


def test_non_completed_outcome_gets_no_ai_entry_human_decision_is_root(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    fails_rec = _record(id="rec-fails", source_id="fails")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, fails_rec, b"fails content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening", _fake_run_ft_screening(fail_document_ids={"rec-fails"}),
    )

    state = _state(cfg, [fails_rec])
    result = fulltext_node(state)
    entry = result["full_text_review_queue"][0]
    assert entry["item_result"]["execution_status"] == "processing_failed"
    assert entry["reviewed_baseline_decision_id"] is None

    store = _ledger_store(cfg, state["review_run_id"])
    # No ledger entry at all for this record/stage -- the ai never
    # actually rendered a judgment.
    assert store.load_key(record_id="rec-fails", stage="full_text_eligibility") == []

    resolution_update = resolve_full_text_review_queue(
        {**state, **result}, {"rec-fails": _resolution("include")},
    )
    assert [r["id"] for r in resolution_update["full_text_records"]] == ["rec-fails"]

    chain = store.load_key(record_id="rec-fails", stage="full_text_eligibility")
    assert len(chain) == 1
    assert chain[0].decision_maker == "human"
    assert chain[0].supersedes_decision_id is None   # this human decision IS the chain's root


def test_growing_confirmed_batch_produces_one_connected_ledger_chain(tmp_path, monkeypatch):
    """Extends test_fulltext_node_growing_confirmed_batch_completes_rather_than_conflicting's
    own scenario: rec-a is screened alone, then re-screened once rec-b's
    PDF also becomes available -- ft_screening_run_id changes (a new
    producer_artifact_id), producing a SECOND ledger entry for rec-a, and
    the loader must see one CONNECTED two-decision chain, never
    disconnected roots."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("rec-a", "population"): "satisfied", ("rec-b", "population"): "satisfied"}),
    )
    rec_a = _record(id="rec-a", source_id="a")
    rec_b = _record(id="rec-b", source_id="b")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, rec_a, b"A bytes")

    state = _state(cfg, [rec_a, rec_b])
    result1 = fulltext_node(state)

    _place_local_pdf(pdf_dir, rec_b, b"B bytes")
    result2 = fulltext_node(state)
    assert result2["ft_screening_run_id"] != result1["ft_screening_run_id"]

    store = _ledger_store(cfg, state["review_run_id"])
    chain = store.load_key(record_id="rec-a", stage="full_text_eligibility")   # raises if disconnected
    assert len(chain) == 2
    roots = [r for r in chain if r.supersedes_decision_id is None]
    assert len(roots) == 1
    tail = store.project_current_tails()[("rec-a", "full_text_eligibility")]
    assert tail.decision == "include"


def test_resolve_full_text_review_queue_rejects_stale_baseline(tmp_path, monkeypatch):
    """A queued entry's own reviewed_baseline_decision_id is no longer the
    current tail (something else superseded it since the queue entry was
    created) -- resolving against it must raise, never silently supersede
    a decision this resolution never actually reviewed."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    review_rec = _record(id="rec-review", source_id="review")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, review_rec, b"review content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-review", "population"): "unclear"},
            requires_review_by_key={("rec-review", "population"): True},
        ),
    )
    state = _state(cfg, [review_rec])
    result = fulltext_node(state)
    merged = {**state, **result}

    # Something else (e.g. a differently-configured re-run) supersedes
    # the baseline before the human resolution is ever applied.
    store = _ledger_store(cfg, state["review_run_id"])
    from slr_pipeline.selection_ledger import emit_selection_decision
    emit_selection_decision(
        store=store, record_id="rec-review", stage="full_text_eligibility",
        decision="exclude", decision_maker="ai", source_event_id="race-evt-1",
    )

    with pytest.raises((StaleReviewError, SelectionLedgerPreconditionError)):
        resolve_full_text_review_queue(merged, {"rec-review": _resolution("include")})


def test_confirmed_pdf_all_ft_screening_failed_no_ai_origin_claimed(tmp_path, monkeypatch):
    """codex review, finding 4a: a confirmed PDF whose FT_SCREENING itself
    fails must not be labelled automation_origin="ai" -- no ai ledger
    entry was ever produced for it."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    fails_rec = _record(id="rec-fails", source_id="fails")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, fails_rec, b"fails content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening", _fake_run_ft_screening(fail_document_ids={"rec-fails"}),
    )
    state = _state(cfg, [fails_rec])
    result = fulltext_node(state)

    assert "full_text_eligibility" not in result["automation_origin"]
    assert "full_text_eligibility" not in result["oversight_mode"]


def test_ft_screening_resolution_log_carries_resolution_identity(tmp_path, monkeypatch):
    """codex review, finding 4c: resolution_id/reviewed_baseline_decision_id
    must be present on ft_screening-stage audit entries, durably linking
    the audit trail back to the ledger's own human producer."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    review_rec = _record(id="rec-review", source_id="review")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, review_rec, b"review content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-review", "population"): "unclear"},
            requires_review_by_key={("rec-review", "population"): True},
        ),
    )
    state = _state(cfg, [review_rec])
    result = fulltext_node(state)
    merged = {**state, **result}

    resolution_update = resolve_full_text_review_queue(merged, {"rec-review": _resolution("include")})
    entry = resolution_update["full_text_review_resolutions"][0]
    assert entry["resolution_id"] is not None
    assert entry["reviewed_baseline_decision_id"] is not None


def test_identical_ft_screening_resubmission_does_not_duplicate_audit_log(tmp_path, monkeypatch):
    """codex review, finding 5 (same class of bug applies to fulltext's
    own resolution log): replaying an identical resolution must not grow
    full_text_review_resolutions."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    review_rec = _record(id="rec-review", source_id="review")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, review_rec, b"review content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-review", "population"): "unclear"},
            requires_review_by_key={("rec-review", "population"): True},
        ),
    )
    state = _state(cfg, [review_rec])
    result = fulltext_node(state)
    merged = {**state, **result}

    first = resolve_full_text_review_queue(merged, {"rec-review": _resolution("include")})
    assert len(first["full_text_review_resolutions"]) == 1

    # Re-submit the SAME record against the SAME (still-queued, per this
    # merged snapshot) entry with an IDENTICAL resolution -- simulates a
    # caller retry against stale state. The ledger's own replay makes this
    # idempotent; the audit log must be too.
    second = resolve_full_text_review_queue(
        {**merged, "full_text_review_resolutions": first["full_text_review_resolutions"]},
        {"rec-review": _resolution("include")},
    )
    assert len(second["full_text_review_resolutions"]) == 1


def test_full_text_origin_and_oversight_absent_with_no_ledger_entries(tmp_path):
    from slr_pipeline.nodes.fulltext import _full_text_origin_and_oversight
    store = SelectionDecisionStore(output_dir=_config(tmp_path).ensure_output_dir(), review_run_id="empty-run")
    assert _full_text_origin_and_oversight(store) == (None, None)


def test_ai_baseline_then_human_supersession_origin_stays_ai(tmp_path, monkeypatch):
    """codex review, second round: after a genuine ai -> human chain, the
    only ledger TAIL is the human entry -- automation_origin must still
    report "ai" (an ai decision genuinely was produced this run), derived
    from ALL entries, never just the current tail."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    review_rec = _record(id="rec-review", source_id="review")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, review_rec, b"review content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(
            status_by_key={("rec-review", "population"): "unclear"},
            requires_review_by_key={("rec-review", "population"): True},
        ),
    )
    state = _state(cfg, [review_rec])
    result = fulltext_node(state)
    merged = {**state, **result}

    resolution_update = resolve_full_text_review_queue(merged, {"rec-review": _resolution("include")})
    assert resolution_update["automation_origin"]["full_text_eligibility"] == "ai"
    assert resolution_update["oversight_mode"]["full_text_eligibility"] == "human_resolved"


def test_human_root_after_failed_assessment_origin_is_no_automated_output(tmp_path, monkeypatch):
    """codex review, second round: a human decision that is the chain's
    own ROOT (no ai baseline ever existed, per a processing failure) must
    report automation_origin="no_automated_output" (not "ai", not
    absent), with oversight_mode="human_resolved"."""
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    fails_rec = _record(id="rec-fails", source_id="fails")
    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, fails_rec, b"fails content")
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening", _fake_run_ft_screening(fail_document_ids={"rec-fails"}),
    )
    state = _state(cfg, [fails_rec])
    result = fulltext_node(state)
    assert "full_text_eligibility" not in result.get("automation_origin", {})

    queue_entry = result["full_text_review_queue"][0]
    assert queue_entry["stage"] == "ft_screening"   # confirmed PDF, real assessment attempted, but failed
    assert queue_entry["reviewed_baseline_decision_id"] is None   # no ai baseline was ever emitted for it
    merged = {**state, **result}
    resolution_update = resolve_full_text_review_queue(merged, {"rec-fails": _resolution("include")})

    assert resolution_update["automation_origin"]["full_text_eligibility"] == "no_automated_output"
    assert resolution_update["oversight_mode"]["full_text_eligibility"] == "human_resolved"
