"""Track 2 Step 12A -- verifies `slr_pipeline.reference_fixture_answer_key`
(the ONE structured answer key) against the REAL pipeline code it must
agree with: `dedup.py::_dedup_doi_only`'s actual duplicate-removal
semantics, `doi.py::normalise_title`'s actual normalisation, and
`fixture_search_adapter.py`'s actual native-record loading contract.

None of this depends on `tests/reference_run/fixtures/corpus/` existing on
disk yet -- these tests build `Record` objects and native-JSON payloads
directly from the answer key module in memory, and (where a native file IS
needed) write it to a temporary directory via `dump_native_fixture_files`,
never touching the real fixture corpus path.
"""
from __future__ import annotations

import os

import pytest

from slr_pipeline import reference_fixture_answer_key as ak
from slr_pipeline.doi import normalise_title
from slr_pipeline.nodes.dedup import _dedup_doi_only
from slr_pipeline.replay import fixture_search_adapter as fsa
from slr_pipeline.state import Record


def _records_from_answer_key() -> list[Record]:
    records = []
    for db in ak.CANONICAL_DATABASES:
        for native in ak.native_payload(db):
            records.append(Record(
                id=native["id"], source=f"fx_{db}", source_id=native["native_key"],
                doi=native["doi"], title=native["title"], abstract=native["abstract"],
                authors=native["authors"], year=native["year"], journal=native["journal"],
                cited_by=native["cited_by"], url=native["url"],
            ))
    return records


def test_answer_key_topology_is_internally_consistent():
    ak.verify_topology()


def test_raw_records_ordering_places_every_canonical_before_its_duplicate():
    position = {r["id"]: i for i, r in enumerate(ak._RAW_RECORDS)}
    for mapping in ak.duplicate_mappings():
        assert position[mapping["canonical_id"]] < position[mapping["dup_id"]], (
            f"{mapping['canonical_id']} must appear before {mapping['dup_id']} in raw_records order"
        )


def test_real_dedup_doi_only_produces_exactly_the_designed_22_survivors():
    """The real `_dedup_doi_only` (no replay/LLM machinery involved) run
    over the answer key's 30 records must drop exactly the 8 authored
    duplicates and keep exactly the 22 authored canonical ids -- proof the
    corpus's exact-title duplicate design actually works against the real
    dedup code, not merely by the author's own claim."""
    records = _records_from_answer_key()
    assert len(records) == 30

    survivors = _dedup_doi_only(records)
    survivor_ids = sorted(r.id for r in survivors)
    assert survivor_ids == sorted(ak.canonical_ids())

    dropped_ids = sorted(set(ak.all_ids()) - set(survivor_ids))
    assert dropped_ids == sorted(ak.duplicate_ids())


def test_every_duplicate_title_normalises_identically_to_its_canonical():
    for mapping in ak.duplicate_mappings():
        dup = ak.record(mapping["dup_id"])
        canonical = ak.record(mapping["canonical_id"])
        assert normalise_title(dup["title"]) == normalise_title(canonical["title"]), (
            f"{mapping['dup_id']} does not normalise-match its canonical {mapping['canonical_id']}"
        )


def test_no_two_unrelated_canonical_titles_accidentally_collide():
    """A design-integrity check the other tests do not otherwise cover:
    no two DIFFERENT canonical (non-duplicate-pair) records may
    accidentally normalise to the same title, which would silently merge
    two intended-distinct records at dedup."""
    seen: dict[str, str] = {}
    for rid in ak.canonical_ids():
        norm = normalise_title(ak.record(rid)["title"])
        assert norm not in seen, f"{rid} and {seen[norm]} accidentally normalise to the same title"
        seen[norm] = rid


def test_all_fixture_records_are_doi_less():
    for db in ak.CANONICAL_DATABASES:
        for native in ak.native_payload(db):
            assert native["doi"] is None


def test_native_fixture_files_load_without_error_via_the_real_fixture_adapter(tmp_path):
    """Writes the 7 native files to a temp corpus root and loads each one
    through the real, already-shipped `fixture_search_adapter` machinery
    (`SLR_FIXTURE_CORPUS_ROOT` env override) -- proves the answer key's
    `native_payload()` shape is accepted by `_record_from_native`, not
    just by this test's own hand-rolled `Record(...)` construction above."""
    ak.dump_native_fixture_files(root=tmp_path)

    old = os.environ.get(fsa._ENV_CORPUS_ROOT)
    os.environ[fsa._ENV_CORPUS_ROOT] = str(tmp_path)
    try:
        assert fsa.corpus_root() == tmp_path
        all_records: list[Record] = []
        for db in fsa.FIXTURE_DATABASES:
            backend_cls = fsa.FIXTURE_ADAPTER_CLASSES[f"fx_{db}"]
            backend = backend_cls()
            from slr_pipeline.search_backends.base import SearchBackendRequest
            request = SearchBackendRequest(
                query="ignored", max_results=50, date_from=None, date_to=None,
                language=None, mailto=None,
            )
            result = backend.search(request)
            assert result.retrieval_status == "exhaustive"
            assert result.skipped_malformed == 0
            all_records.extend(result.records)
    finally:
        if old is None:
            os.environ.pop(fsa._ENV_CORPUS_ROOT, None)
        else:
            os.environ[fsa._ENV_CORPUS_ROOT] = old

    assert sorted(r.id for r in all_records) == sorted(ak.all_ids())
    # Fixture adapters preserve the authored id/source/source_id verbatim
    # (plan §2.6) -- spot-check one record from each end of the corpus.
    by_id = {r.id: r for r in all_records}
    assert by_id["fx-0001"].source == "fx_pubmed"
    assert by_id["fx-0030"].source == "fx_business_source_complete"
    assert by_id["fx-0030"].source_id == "NK-fx-0030"


def test_scenario_table_has_one_row_per_raw_record_and_is_deterministic():
    rows = ak.scenario_table()
    assert len(rows) == 30
    assert [r["id"] for r in rows] == ak.all_ids()
    assert ak.scenario_table_markdown() == ak.scenario_table_markdown()


def test_extraction_answers_cover_exactly_the_full_text_eligible_records():
    assert set(ak.EXTRACTION_ANSWERS) == set(ak.FULL_TEXT_ELIGIBLE)
    expected_fields = {
        "country", "study_design", "population", "concept", "context",
        "limitations", "key_findings", "funding_sources", "funder_role",
    }
    for rid, values in ak.EXTRACTION_ANSWERS.items():
        assert set(values) == expected_fields, rid


def test_full_text_exclusion_reason_code_is_one_of_the_frozen_criteria_codes():
    frozen_codes = {"no_mbse_simulation", "not_rural_healthcare", "not_disaster_context"}
    for rid in ak.FULL_TEXT_EXCLUSIONS:
        code = ak.FULL_TEXT_ELIGIBILITY[rid]["exclusion_reason_code"]
        assert code in frozen_codes


def test_funding_statement_variants_cover_five_distinct_branches():
    """The 5 extraction-reaching records must exercise 5 distinct funding
    presentations (present-with-role, exact 'No external funding', exact
    'The funder had no role in the study.', absent, present-with-
    ambiguous-role) -- the diversity the revision proposal asks for."""
    funding_sources = {rid: tuple(v["funding_sources"]) for rid, v in ak.EXTRACTION_ANSWERS.items()}
    funder_roles = {rid: v["funder_role"] for rid, v in ak.EXTRACTION_ANSWERS.items()}

    assert funding_sources["fx-0005"] == ("No external funding",)
    assert funder_roles["fx-0009"] == "The funder had no role in the study."
    assert funding_sources["fx-0013"] == ()
    assert funder_roles["fx-0013"] is None
    assert funding_sources["fx-0002"] not in ((), ("No external funding",))
    assert funder_roles["fx-0002"] not in (None, "The funder had no role in the study.")
    assert funding_sources["fx-0024"] not in ((), ("No external funding",))
    assert funder_roles["fx-0024"] not in (None, "The funder had no role in the study.")
