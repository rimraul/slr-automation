"""Offline tests for the S5 multi-database search solution."""

from __future__ import annotations

import inspect

from examples.chapter.s05_multi_db_search import solution as sol

_SOURCES = ("openalex", "pubmed", "crossref", "scopus")


def test_fetch_returns_raw_body_text():
    for name in _SOURCES:
        body = sol.FixtureAdapter(name).fetch("q")
        assert isinstance(body, str) and body.strip()
        if name == "pubmed":
            assert body.lstrip().startswith("<")
        else:
            import json

            json.loads(body)  # must be valid JSON text


def test_every_normalised_record_has_the_exact_g1_shape():
    for name in _SOURCES:
        records = sol.normalise(name, sol.FixtureAdapter(name).fetch("q"))
        assert records
        for rec in records:
            assert set(rec) == sol.EXPECTED_KEYS, (name, set(rec))
            assert rec["id"] == f"{rec['source']}:{rec['source_id']}"
            assert rec["source"] == name
            assert isinstance(rec["source_id"], str) and rec["source_id"]
            assert rec["doi"] is None or isinstance(rec["doi"], str)
            assert isinstance(rec["title"], str)
            assert isinstance(rec["abstract"], str)  # never None
            assert rec["year"] is None or isinstance(rec["year"], int)
            assert isinstance(rec["authors"], list)
            assert all(isinstance(a, str) for a in rec["authors"])


def test_pubmed_doi_is_inconsistently_present():
    records = {r["source_id"]: r for r in sol.normalise("pubmed", sol.FixtureAdapter("pubmed").fetch("q"))}
    assert records["40000002"]["doi"] is None
    assert records["40000001"]["doi"] == "10.2000/s5.001"


def test_openalex_doi_url_is_stripped():
    records = sol.normalise("openalex", sol.FixtureAdapter("openalex").fetch("q"))
    for rec in records:
        if rec["doi"] is not None:
            assert not rec["doi"].startswith("http")


def test_run_search_merges_all_four_sources():
    records = sol.run_search(sol.default_adapters(), "q")
    assert len(records) == 17
    by_source = {}
    for rec in records:
        by_source.setdefault(rec["source"], 0)
        by_source[rec["source"]] += 1
    assert by_source == {"openalex": 4, "pubmed": 4, "crossref": 5, "scopus": 4}


def test_crossref_record_without_a_doi_still_satisfies_g1():
    records = sol.normalise("crossref", sol.FixtureAdapter("crossref").fetch("q"))
    no_doi = [r for r in records if r["doi"] is None]
    assert len(no_doi) == 1
    rec = no_doi[0]
    assert set(rec) == sol.EXPECTED_KEYS
    assert rec["source_id"]  # a non-empty deterministic fallback
    assert rec["id"] == f"crossref:{rec['source_id']}"


def test_same_study_survives_as_two_entries_across_sources():
    records = sol.run_search(sol.default_adapters(), "q")
    pubmed_no_doi = next(r for r in records if r["source"] == "pubmed" and r["source_id"] == "40000002")
    crossref_with_doi = next(r for r in records if r["source"] == "crossref" and r["doi"] == "10.2000/s5.002")
    assert pubmed_no_doi["doi"] is None
    assert crossref_with_doi["doi"] == "10.2000/s5.002"
    # both present -> S6's job to reconcile by title
    assert pubmed_no_doi["id"] != crossref_with_doi["id"]


def test_solution_does_not_import_a_network_library():
    src = inspect.getsource(sol)
    assert "import requests" not in src
    assert "urllib.request" not in src
    assert "http.client" not in src
