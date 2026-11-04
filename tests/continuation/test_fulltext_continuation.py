"""Unit C -- `slr_pipeline.nodes.fulltext_continuation.continuation_fulltext_node`.

Two boundaries are monkeypatched, and ONLY these two -- everything else
(acquisition routing, the FT_SCREENING request/outcome mapping) runs for
real, mirroring `tests/nodes/test_fulltext_node.py`'s own discipline
exactly:
  - `_extract_text` -- replaced with a controllable in-memory string.
  - `run_ft_screening` -- replaced with a factory that forwards the REAL
    request into FT_SCREENING's own private `_run_ft_screening` using
    fakes.

No network calls happen anywhere in this file: every retrieved-bucket PDF
is pre-placed locally under the exact naming convention
`_find_local_pdf()`/`_dest_path()` both use, so `acquire_and_confirm_
records()`'s local-hit path always wins before the OA cascade is ever
reached; `_resolve_oa_url`/`_download_pdf` are additionally monkeypatched
to raise if called at all, as an explicit proof, not just an assumption.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import slr_pipeline.nodes.fulltext as fulltext_module
from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.partition import derive_continuation_partition
from slr_pipeline.nodes.fulltext_continuation import continuation_fulltext_node
from slr_pipeline.state import PRISMACounts, Record, initial_state

from ft_screening import CriteriaSet, CriterionSpec, ScreeningConfig

from .conftest import load_real_fixture

_REAL_TEXT = "This is real extracted full-text content. " * 30
_NNA_ID = "scopus-fb08897e9d4f238ce11c"


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
        title="Test continuation",
        output_dir=str(tmp_path / "outputs"),
        full_text_criteria_set=_criteria_set().model_dump(mode="json"),
    )
    defaults.update(overrides)
    return ReviewConfig(**defaults)


def _fake_run_ft_screening(*, status_by_key=None, requires_review_by_key=None, calls=None):
    def _fake(request):
        from ft_screening.components import Components
        from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
        from ft_screening.orchestrator import _run_ft_screening
        from ft_screening.store import FileScreeningStore

        if calls is not None:
            calls.append(request)
        components = Components(
            parser=FakeDocumentParser(fail_document_ids=set()),
            retriever=FakeEvidenceRetriever(),
            adjudicator=FakeCriterionAdjudicator(
                status_by_key=status_by_key or {}, requires_review_by_key=requires_review_by_key or {},
                model=request.config.adjudicator_model,
                prompt_template_version=request.config.prompt_template_version,
                output_schema_version=request.config.output_schema_version,
            ),
            store=FileScreeningStore(output_dir=request.output_dir),
        )
        return _run_ft_screening(request, components)

    return _fake


def _always_real_text(pdf_path, max_chars=120_000):
    return _REAL_TEXT


def _seed_state(
    cfg: ReviewConfig, included_records, partition, *,
    preserved_included_records=None, parent_prisma: dict | None = None,
) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["included_records"] = [r.to_dict() for r in included_records]
    st["review_run_id"] = "continuation-test-run"
    st["continuation_partition"] = partition.model_dump()
    st["preserved_included_records"] = [r.to_dict() for r in (preserved_included_records or [])]
    if parent_prisma is not None:
        st["prisma_counts"] = parent_prisma
    return st


def _place_local_pdf(pdf_dir: Path, record: Record, content: bytes) -> Path:
    dest = fulltext_module._dest_path(record, pdf_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


def _never_call_oa(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("OA cascade must never be reached for a pre-staged local PDF")

    monkeypatch.setattr(fulltext_module, "_resolve_oa_url", _boom)
    monkeypatch.setattr(fulltext_module, "_download_pdf", _boom)


# ---------------------------------------------------------------------------
# Module-source checks (round-26 / round-28 findings): the node must never
# reference launcher-scope-only objects it structurally cannot receive.
# ---------------------------------------------------------------------------


def test_module_source_never_references_launcher_scope_partition_objects():
    source = inspect.getsource(continuation_fulltext_node)
    for forbidden in ("parent_snapshot", "parent_bundle_path", "derive_continuation_partition"):
        assert forbidden not in source


def test_module_source_never_references_continuation_manifest_or_copy_logic():
    source = inspect.getsource(continuation_fulltext_node)
    for forbidden in ("continuation_manifest", "pdfs_dir", "shutil.copy", "file_sha256"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Real 14/3/10/1 fixture -- the corrected (round-36) additive/recompute tail.
# ---------------------------------------------------------------------------


def test_continuation_fulltext_node_real_fixture_recomputed_histogram_totals_14(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    _never_call_oa(monkeypatch)
    calls = []
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening(calls=calls))

    included_records, parent_snapshot, continuation_manifest = load_real_fixture()
    partition = derive_continuation_partition(included_records, parent_snapshot, continuation_manifest)
    assert len(partition.retrieved_ids) == 10
    assert partition.not_retrieved_ids == [_NNA_ID]

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    retrieved_by_id = {r.id: r for r in included_records if r.id in set(partition.retrieved_ids)}
    for record in retrieved_by_id.values():
        _place_local_pdf(pdf_dir, record, f"content for {record.id}".encode())

    parent_prisma = PRISMACounts(
        assessed_full_text=3, excluded_full_text=3,
        excluded_full_text_reasons={"wrong_population": 3},
        fulltext_retrieval={"parsed": 3, "not_retrieved": 11},
        after_dedup=14, screened_title_abstract=14,
    ).model_dump()

    state = _seed_state(cfg, included_records, partition, parent_prisma=parent_prisma)
    result = continuation_fulltext_node(state)

    called_ids = {item.document_id for req in calls for item in req.items}
    assert called_ids == set(partition.retrieved_ids)

    assert {r["id"] for r in result["full_text_records"]} == set(partition.retrieved_ids)

    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.assessed_full_text == 3 + 10
    assert prisma.excluded_full_text == 3
    assert prisma.excluded_full_text_reasons == {"wrong_population": 3}
    assert prisma.fulltext_retrieval == {"parsed": 13, "not_retrieved": 1}
    assert sum(prisma.fulltext_retrieval.values()) == len(included_records) == 14
    # The pre-fix (key-wise-sum) design would have produced 24 here -- the
    # regression this test exists to guard against.
    assert sum(prisma.fulltext_retrieval.values()) != 24

    assert result["ft_screening_run_id"] is not None
    assert result["oversight_mode"]["full_text_eligibility"] is not None
    assert result["automation_origin"]["full_text_eligibility"] is not None


# ---------------------------------------------------------------------------
# Synthetic fixture with a preserved *inclusion* -- proves the include path
# reaches full_text_records too (round-2 finding 3's own concern, carried
# into Unit C).
# ---------------------------------------------------------------------------


def test_continuation_fulltext_node_synthetic_fixture_preserved_inclusion_and_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    _never_call_oa(monkeypatch)
    calls = []
    monkeypatch.setattr(
        fulltext_module, "run_ft_screening",
        _fake_run_ft_screening(status_by_key={("r-new", "population"): "satisfied"}, calls=calls),
    )

    r_inc = Record(id="r-inc", source="test", source_id="inc", doi=None)
    r_exc = Record(id="r-exc", source="test", source_id="exc", doi=None)
    r_new = Record(id="r-new", source="test", source_id="new", doi=None)
    r_missing = Record(id="r-missing", source="test", source_id="missing", doi=None)
    included_records = [r_inc, r_exc, r_new, r_missing]

    from slr_pipeline.continuation.partition import ContinuationPartition
    partition = ContinuationPartition(
        preserved_included_ids=["r-inc"], preserved_excluded_ids=["r-exc"],
        retrieved_ids=["r-new"], not_retrieved_ids=["r-missing"],
    )

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    _place_local_pdf(pdf_dir, r_new, b"new content")

    r_inc_preserved = Record(
        id="r-inc", source="test", source_id="inc", doi=None,
        full_text_path=str(tmp_path / "parent_pdfs" / "r-inc.pdf"), full_text_status="parsed",
    )

    parent_prisma = PRISMACounts(
        assessed_full_text=2, excluded_full_text=1,
        excluded_full_text_reasons={"wrong_population": 1},
        fulltext_retrieval={"parsed": 2, "not_retrieved": 2},
        after_dedup=4, screened_title_abstract=4,
    ).model_dump()

    state = _seed_state(
        cfg, included_records, partition,
        preserved_included_records=[r_inc_preserved], parent_prisma=parent_prisma,
    )
    result = continuation_fulltext_node(state)

    called_ids = {item.document_id for req in calls for item in req.items}
    assert called_ids == {"r-new"}

    full_text_ids = {r["id"] for r in result["full_text_records"]}
    assert full_text_ids == {"r-inc", "r-new"}

    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.assessed_full_text == 2 + 1
    assert prisma.excluded_full_text == 1
    assert prisma.excluded_full_text_reasons == {"wrong_population": 1}
    assert prisma.fulltext_retrieval == {"parsed": 3, "not_retrieved": 1}
    assert sum(prisma.fulltext_retrieval.values()) == len(included_records) == 4


# ---------------------------------------------------------------------------
# Minimal seeded state, zero retrieved/preserved records -- proves the node
# runs successfully with no parent_snapshot/bundle object anywhere in scope,
# and that _run_full_text_eligibility() is never called when `confirmed` is
# empty (round-34's own guard).
# ---------------------------------------------------------------------------


def test_continuation_fulltext_node_minimal_seeded_state_all_not_retrieved(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("FT_SCREENING must never be called when confirmed is empty")

    monkeypatch.setattr(fulltext_module, "run_ft_screening", _boom)

    r_missing = Record(id="r-missing", source="test", source_id="missing", doi=None)
    included_records = [r_missing]

    from slr_pipeline.continuation.partition import ContinuationPartition
    partition = ContinuationPartition(
        preserved_included_ids=[], preserved_excluded_ids=[], retrieved_ids=[], not_retrieved_ids=["r-missing"],
    )

    cfg = _config(tmp_path)
    parent_prisma = PRISMACounts(
        assessed_full_text=0, excluded_full_text=0, after_dedup=1, screened_title_abstract=1,
    ).model_dump()
    state = _seed_state(cfg, included_records, partition, parent_prisma=parent_prisma)

    result = continuation_fulltext_node(state)

    assert result["full_text_records"] == []
    assert result["full_text_review_queue"] == []
    assert result["ft_screening_run_id"] is None
    assert result["oversight_mode"] == {}
    assert result["automation_origin"] == {}
    prisma = PRISMACounts(**result["prisma_counts"])
    assert prisma.fulltext_retrieval == {"not_retrieved": 1}


# ---------------------------------------------------------------------------
# retrieved_pdf_manifest-seeded state (round-28 finding 2): zero filesystem
# copy performed by the node itself -- the file is already at its final
# destination before the node ever runs.
# ---------------------------------------------------------------------------


def test_continuation_fulltext_node_never_copies_pdfs_itself(tmp_path, monkeypatch):
    import shutil

    def _boom_copy(*args, **kwargs):
        raise AssertionError("continuation_fulltext_node must never copy a PDF itself")

    monkeypatch.setattr(shutil, "copy", _boom_copy)
    monkeypatch.setattr(shutil, "copy2", _boom_copy)
    monkeypatch.setattr(fulltext_module, "_extract_text", _always_real_text)
    _never_call_oa(monkeypatch)
    monkeypatch.setattr(fulltext_module, "run_ft_screening", _fake_run_ft_screening())

    r_new = Record(id="r-new", source="test", source_id="new", doi=None)
    included_records = [r_new]

    from slr_pipeline.continuation.partition import ContinuationPartition
    partition = ContinuationPartition(
        preserved_included_ids=[], preserved_excluded_ids=[], retrieved_ids=["r-new"], not_retrieved_ids=[],
    )

    cfg = _config(tmp_path)
    pdf_dir = Path(cfg.output_dir) / "pdfs"
    dest = _place_local_pdf(pdf_dir, r_new, b"pre-staged content")

    parent_prisma = PRISMACounts(assessed_full_text=0, excluded_full_text=0).model_dump()
    state = _seed_state(cfg, included_records, partition, parent_prisma=parent_prisma)
    # Unit A item 15's own seeded field -- present on state, but the node's
    # own module source never reads it (see the module-source checks above).
    state["retrieved_pdf_manifest"] = {"r-new": {"dest_path": str(dest), "file_sha256": "a" * 64}}

    result = continuation_fulltext_node(state)

    assert {r["id"] for r in result["full_text_records"]} == {"r-new"}
