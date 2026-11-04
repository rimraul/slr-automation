"""Unit tests for `slr_pipeline.replay.fixture_search_adapter` -- the seven
`fx_*` search adapters (`track2_pipeline_readiness_plan_2026-08-30.md`
Sec 2.5/2.6/6/7.1, step 10). End-to-end wiring through `nodes/search.py`'s
real `_ADAPTER_REGISTRY` is covered separately by
`tests/nodes/test_search_node.py::test_mapped_dispatch_reaches_real_fixture_adapter`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slr_pipeline.replay.fixture_search_adapter import (
    FIXTURE_ADAPTER_CLASSES,
    FIXTURE_DATABASES,
    FixtureCorpusError,
    corpus_root,
)
from slr_pipeline.search_backends.base import SearchBackendRequest


def _request(max_results: int = 50) -> SearchBackendRequest:
    return SearchBackendRequest(query="anything", max_results=max_results,
                                 date_from=None, date_to=None, language=None, mailto=None)


def _write_corpus(tmp_path, db: str, records: list) -> None:
    (tmp_path / "native").mkdir(parents=True, exist_ok=True)
    (tmp_path / "native" / f"fx_{db}.json").write_text(json.dumps(records), encoding="utf-8")


def test_all_seven_fixture_databases_registered():
    assert set(FIXTURE_ADAPTER_CLASSES) == {f"fx_{db}" for db in FIXTURE_DATABASES}
    assert len(FIXTURE_DATABASES) == 7


def test_corpus_root_defaults_and_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SLR_FIXTURE_CORPUS_ROOT", raising=False)
    assert str(corpus_root()) == str(Path("tests/reference_run/fixtures/corpus"))
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    assert corpus_root() == tmp_path


def test_authored_record_id_and_source_id_preserved_verbatim(monkeypatch, tmp_path):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    _write_corpus(tmp_path, "pubmed", [
        {"id": "fx-0009", "native_key": "pm-t2-0009", "title": "Rural disaster MBSE study",
         "doi": None, "year": 2021, "authors": ["A. One", "B. Two"], "journal": "J",
         "abstract": "abs", "url": "https://example.org", "cited_by": 3},
    ])
    adapter = FIXTURE_ADAPTER_CLASSES["fx_pubmed"]()
    result = adapter.search(_request())
    assert result.retrieval_status == "exhaustive"
    assert result.adapter == "fx_pubmed"
    assert len(result.records) == 1
    rec = result.records[0]
    assert rec.id == "fx-0009"            # authored id, never re-derived
    assert rec.source == "fx_pubmed"
    assert rec.source_id == "pm-t2-0009"  # authored native_key, verbatim
    assert rec.doi is None
    assert rec.year == 2021


def test_max_results_truncates_but_reports_full_total_hits(monkeypatch, tmp_path):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    _write_corpus(tmp_path, "scopus", [
        {"id": f"fx-{i:04d}", "native_key": f"scopus-t2-{i:04d}", "title": f"study {i}"}
        for i in range(5)
    ])
    adapter = FIXTURE_ADAPTER_CLASSES["fx_scopus"]()
    result = adapter.search(_request(max_results=2))
    assert result.returned == 2
    assert result.requested == 2
    assert result.total_hits == 5
    assert result.retrieval_status == "capped"


def test_empty_corpus_file_is_a_clean_exhaustive_zero_hit_result(monkeypatch, tmp_path):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    _write_corpus(tmp_path, "cinahl", [])
    adapter = FIXTURE_ADAPTER_CLASSES["fx_cinahl"]()
    result = adapter.search(_request())
    assert result.retrieval_status == "exhaustive"
    assert result.returned == 0
    assert result.failure_detail is None
    assert result.skipped_malformed == 0


def test_missing_corpus_file_raises_fixture_corpus_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    adapter = FIXTURE_ADAPTER_CLASSES["fx_acm"]()
    with pytest.raises(FixtureCorpusError, match="cannot read"):
        adapter.search(_request())


def test_malformed_json_raises_fixture_corpus_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "native").mkdir(parents=True)
    (tmp_path / "native" / "fx_acm.json").write_text("not json", encoding="utf-8")
    adapter = FIXTURE_ADAPTER_CLASSES["fx_acm"]()
    with pytest.raises(FixtureCorpusError, match="not valid JSON"):
        adapter.search(_request())


def test_corpus_file_not_a_list_raises_fixture_corpus_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "native").mkdir(parents=True)
    (tmp_path / "native" / "fx_acm.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    adapter = FIXTURE_ADAPTER_CLASSES["fx_acm"]()
    with pytest.raises(FixtureCorpusError, match="JSON list"):
        adapter.search(_request())


@pytest.mark.parametrize("bad_record", [
    {"native_key": "k1"},                       # missing id
    {"id": "fx-0001"},                           # missing native_key
    {"id": "", "native_key": "k1"},              # empty id
    {"id": "fx-0001", "native_key": ""},         # empty native_key
    "not-a-dict",
])
def test_malformed_record_raises_fixture_corpus_error(monkeypatch, tmp_path, bad_record):
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    _write_corpus(tmp_path, "engineering_village", [bad_record])
    adapter = FIXTURE_ADAPTER_CLASSES["fx_engineering_village"]()
    with pytest.raises(FixtureCorpusError):
        adapter.search(_request())


def test_a_single_malformed_record_is_never_silently_skipped(monkeypatch, tmp_path):
    """Plan Sec 2.4: unlike a real adapter's skip-and-continue tolerance, a
    fixture corpus with one bad record among good ones still raises --
    it never degrades to `partial`/`skipped_malformed`."""
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(tmp_path))
    _write_corpus(tmp_path, "business_source_complete", [
        {"id": "fx-0001", "native_key": "bsc-1", "title": "good"},
        {"id": "fx-0002"},  # missing native_key
    ])
    adapter = FIXTURE_ADAPTER_CLASSES["fx_business_source_complete"]()
    with pytest.raises(FixtureCorpusError):
        adapter.search(_request())
