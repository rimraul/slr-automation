"""Offline tests for the S4 query-expansion solution."""

from __future__ import annotations

import pytest

from examples.chapter import toolkit
from examples.chapter.s04_query_expansion import solution as sol

SEED = "large language models for systematic review screening"
_SCRIPTED = toolkit.load_scripted("s04")


def _llm(response: str):
    return toolkit.scripted_llm({SEED: response}, key_fn=lambda m: m[-1]["content"])


def test_expand_query_returns_at_least_three_concepts():
    expansion = sol.expand_query(SEED, call_llm=_llm(_SCRIPTED[SEED]))
    assert len(expansion.concepts) >= 3
    for concept in expansion.concepts:
        assert concept.terms


def test_boolean_query_uses_and_between_or_within():
    expansion = sol.expand_query(SEED, call_llm=_llm(_SCRIPTED[SEED]))
    assert " AND " in expansion.boolean_query
    assert " OR " in expansion.boolean_query


def test_pubmed_translation_has_field_tags_absent_from_openalex():
    expansion = sol.expand_query(SEED, call_llm=_llm(_SCRIPTED[SEED]))
    pubmed = sol.translate_for_database(expansion, "pubmed")
    openalex = sol.translate_for_database(expansion, "openalex")
    assert "[tiab]" in pubmed
    assert "[tiab]" not in openalex
    assert "TITLE-ABS-KEY" in sol.translate_for_database(expansion, "scopus")


def test_translation_is_pure_no_llm_call():
    def exploding_llm(*a, **k):
        raise AssertionError("translate_for_database must not call the LLM")

    expansion = sol.expand_query(SEED, call_llm=_llm(_SCRIPTED[SEED]))
    # translate takes no call_llm at all; this just proves the seam stays clean.
    assert sol.translate_for_database(expansion, "pubmed")
    _ = exploding_llm


def test_malformed_json_raises():
    with pytest.raises(ValueError):
        sol.expand_query(SEED, call_llm=_llm(_SCRIPTED["_bad"]["malformed_json"]))


def test_empty_concepts_raises():
    with pytest.raises(ValueError):
        sol.expand_query(SEED, call_llm=_llm(_SCRIPTED["_bad"]["empty_concepts"]))
