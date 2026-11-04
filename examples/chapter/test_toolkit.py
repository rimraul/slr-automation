"""Offline tests for examples.chapter.toolkit."""

from __future__ import annotations

import json

import numpy as np
import pytest

from examples.chapter import toolkit


# --- call_llm -------------------------------------------------------------

def test_call_llm_raises_without_a_key():
    with pytest.raises(RuntimeError, match="offline path"):
        toolkit.call_llm([{"role": "user", "content": "hi"}])


def test_call_llm_accepts_the_full_keyword_signature():
    # It must still raise (offline), but only after accepting every kwarg the
    # builder's output dict spreads into it.
    with pytest.raises(RuntimeError):
        toolkit.call_llm(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
            model="whatever",
            temperature=0.0,
        )


# --- scripted_llm --------------------------------------------------------

def test_scripted_llm_dispatches_on_key_fn():
    fake = toolkit.scripted_llm({"a": "res-a", "b": "res-b"}, key_fn=lambda m: m[-1]["content"])
    assert fake([{"role": "user", "content": "a"}]) == "res-a"
    assert fake([{"role": "user", "content": "b"}], temperature=0.0) == "res-b"


def test_scripted_llm_raises_keyerror_on_miss():
    fake = toolkit.scripted_llm({"a": "res-a"}, key_fn=lambda m: m[-1]["content"])
    with pytest.raises(KeyError):
        fake([{"role": "user", "content": "missing"}])


# --- load_fixture ------------------------------------------------------

def test_load_fixture_json():
    records = toolkit.load_fixture("bib_records.json")
    assert isinstance(records, list) and records


def test_load_fixture_markdown_normalises_newlines():
    text = toolkit.load_fixture("papers/relevant.md")
    assert isinstance(text, str)
    assert "\r" not in text


def test_load_fixture_xml_is_text():
    text = toolkit.load_fixture("raw_responses/pubmed.xml")
    assert text.lstrip().startswith("<")


def test_load_fixture_npy_is_array():
    arr = toolkit.load_fixture("topic_corpus_embeddings.npy")
    assert isinstance(arr, np.ndarray)


def test_load_fixture_rejects_unknown_extension():
    with pytest.raises(ValueError):
        toolkit.load_fixture("LICENSE_NOTES.md.bogus")


# --- DEMO_CRITERIA ----------------------------------------------------

def test_demo_criteria_is_frozen_and_populated():
    assert toolkit.DEMO_CRITERIA.population
    assert len(toolkit.DEMO_CRITERIA.inclusion_criteria) >= 3
    assert len(toolkit.DEMO_CRITERIA.exclusion_criteria) >= 3
    with pytest.raises(Exception):
        toolkit.DEMO_CRITERIA.population = "changed"


# --- fake_downloader / mock_oa_resolvers -----------------------------

def _pdf_url(payload: dict) -> str | None:
    if "best_oa_location" in payload:
        loc = payload.get("best_oa_location") or {}
        return loc.get("url_for_pdf")
    if "openAccessPdf" in payload:
        oa = payload.get("openAccessPdf") or {}
        return oa.get("url")
    return None


def test_fake_downloader_maps_every_resolver_url():
    table = toolkit.load_fixture("oa_resolver_responses.json")
    seen = 0
    for entry in table.values():
        for payload in entry.values():
            if payload is None:
                continue
            url = _pdf_url(payload)
            if not url:
                continue
            text = toolkit.fake_downloader(url)
            assert isinstance(text, str) and text
            seen += 1
    assert seen > 0


def test_resolvers_carry_a_name():
    assert [r.name for r in toolkit.mock_oa_resolvers] == ["unpaywall", "semantic_scholar"]


# --- build_screening_request ----------------------------------------

def _sample_records(n: int) -> list[dict]:
    return [
        {"id": f"rec-{i:02d}", "title": f"Title {i}", "abstract": f"Abstract body {i}."}
        for i in range(1, n + 1)
    ]


def test_build_screening_request_modes_produce_byte_identical_messages():
    records = _sample_records(3)
    offline = toolkit.build_screening_request(records, toolkit.DEMO_CRITERIA, mode="offline")
    live = toolkit.build_screening_request(records, toolkit.DEMO_CRITERIA, mode="live")
    assert offline["messages"] == live["messages"]
    assert json.dumps(offline["messages"]) == json.dumps(live["messages"])
    assert offline["model"] != live["model"]
    assert live["model"] == "gpt-5.4-mini"


def test_build_screening_request_dict_has_exactly_four_keys():
    request = toolkit.build_screening_request(_sample_records(1), toolkit.DEMO_CRITERIA, mode="offline")
    assert set(request) == {"model", "messages", "response_format", "temperature"}
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0.0


def test_build_screening_request_rejects_more_than_three_records():
    with pytest.raises(ValueError, match="at most 3"):
        toolkit.build_screening_request(_sample_records(4), toolkit.DEMO_CRITERIA, mode="offline")


def test_build_screening_request_truncates_abstracts_to_1000_chars():
    long_abstract = "x" * 5000
    records = [{"id": "rec-01", "title": "T", "abstract": long_abstract}]
    request = toolkit.build_screening_request(records, toolkit.DEMO_CRITERIA, mode="offline")
    user_content = request["messages"][1]["content"]
    assert "x" * 1000 in user_content
    assert "x" * 1001 not in user_content


def test_build_screening_request_is_directly_spreadable_into_call_llm():
    request = toolkit.build_screening_request(_sample_records(2), toolkit.DEMO_CRITERIA, mode="offline")
    with pytest.raises(RuntimeError):
        toolkit.call_llm(**request)
