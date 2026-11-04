"""Offline tests for the S8 full-text solution."""

from __future__ import annotations

import pytest

from examples.chapter import toolkit
from examples.chapter.s08_fulltext import solution as sol

_ANSWER_KEY = toolkit.load_fixture("papers/answer_key.json")
_RESOLVERS = toolkit.load_fixture("oa_resolver_responses.json")
_OA_ONLY = toolkit.load_fixture("oa_retrieval_records.json")
_SCRIPTED = toolkit.load_scripted("s08")


def _paper_signature(stem: str) -> str:
    return toolkit.load_fixture(f"papers/{stem}.md").strip().splitlines()[0]


def _scripted_by_stem():
    def key_fn(messages):
        content = messages[-1]["content"]
        for stem in _SCRIPTED["papers"]:
            if _paper_signature(stem) in content:
                return stem
        return "?"

    return toolkit.scripted_llm(_SCRIPTED["papers"], key_fn=key_fn)


# --- retrieval cascade ------------------------------------------

def test_discover_none_is_no_doi():
    assert sol.discover(None).status == "no_doi"


def test_all_four_paper_dois_are_found_including_editorial_fall_through():
    for stem, entry in _ANSWER_KEY.items():
        result = sol.discover(entry["doi"])
        assert result.status == "found", stem
    editorial = sol.discover(_ANSWER_KEY["editorial"]["doi"])
    assert editorial.source == "semantic_scholar"  # unpaywall was null; cascade advanced


def test_oa_retrieval_records_never_resolve():
    for record in _OA_ONLY:
        assert sol.discover(record["doi"]).status == "no_oa_fulltext_found"


def test_download_returns_the_right_paper_per_url():
    for stem, entry in _ANSWER_KEY.items():
        found = sol.discover(entry["doi"])
        text = sol.download(found.pdf_url)
        assert _paper_signature(stem) in text


def test_looks_like_stub_catches_truncated_text():
    assert sol.looks_like_stub("Abstract. This is a preview only.") is True
    assert sol.looks_like_stub("x" * 50) is True
    assert sol.looks_like_stub(toolkit.load_fixture("papers/relevant.md")) is False


def test_failed_discovery_never_triggers_screen_fulltext():
    def exploding_llm(*a, **k):
        raise AssertionError("screen_fulltext must not run without full text")

    found, decision = sol.retrieve_and_screen(_OA_ONLY[0]["doi"], toolkit.DEMO_CRITERIA, call_llm=exploding_llm)
    assert found.status == "no_oa_fulltext_found"
    assert decision is None
    found2, decision2 = sol.retrieve_and_screen(None, toolkit.DEMO_CRITERIA, call_llm=exploding_llm)
    assert found2.status == "no_doi" and decision2 is None


# --- cross-fixture integrity (G3) ------------------------------

def test_cross_fixture_integrity():
    for stem, entry in _ANSWER_KEY.items():
        doi = entry["doi"]
        assert doi in _RESOLVERS, stem
        found = sol.discover(doi)
        assert found.status == "found"
        assert _paper_signature(stem) in sol.download(found.pdf_url)
    for record in _OA_ONLY:
        assert record["doi"] in _RESOLVERS
        payloads = _RESOLVERS[record["doi"]]
        assert payloads["unpaywall"] is None
        assert payloads["semantic_scholar"] is None


# --- direct eligibility unit ----------------------------------

def test_screen_fulltext_reproduces_answer_key_for_all_four_papers():
    scripted = _scripted_by_stem()
    for stem, entry in _ANSWER_KEY.items():
        text = toolkit.load_fixture(f"papers/{stem}.md")
        decision = sol.screen_fulltext(text, toolkit.DEMO_CRITERIA, call_llm=scripted)
        assert decision.decision == entry["s08"], stem


def test_prompt_names_every_criterion_label_it_will_validate():
    seen = {}

    def spy(messages, **kwargs):
        seen["content"] = messages[-1]["content"]
        return _SCRIPTED["papers"]["relevant"]

    sol.screen_fulltext(toolkit.load_fixture("papers/relevant.md"), toolkit.DEMO_CRITERIA, call_llm=spy)
    for label in sol._CRITERION_LABELS:
        assert label in seen["content"], f"prompt never mentions the required label {label!r}"


@pytest.mark.parametrize(
    "variant",
    ["malformed_eligibility_json", "unknown_decision", "incomplete_criteria", "bad_criterion_value"],
)
def test_bad_eligibility_responses_raise(variant):
    fake = toolkit.scripted_llm({"k": _SCRIPTED["_bad"][variant]}, key_fn=lambda m: "k")
    with pytest.raises(ValueError):
        sol.screen_fulltext("some text", toolkit.DEMO_CRITERIA, call_llm=fake)


def test_editorial_falls_through_a_responding_but_urlless_unpaywall():
    editorial_doi = _ANSWER_KEY["editorial"]["doi"]
    assert _RESOLVERS[editorial_doi]["unpaywall"] is not None  # it responds...
    assert _RESOLVERS[editorial_doi]["unpaywall"]["best_oa_location"] is None  # ...but with no PDF
    assert sol.discover(editorial_doi).source == "semantic_scholar"
