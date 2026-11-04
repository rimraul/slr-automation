"""Offline tests for the S10 extraction solution."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict, StrictStr

from examples.chapter import toolkit
from examples.chapter.s10_extraction import solution as sol

_SCRIPTED = toolkit.load_scripted("s10")
_PAPER = toolkit.load_fixture("papers/relevant.md")
_ANSWER_KEY = toolkit.load_fixture("papers/answer_key.json")["relevant"]["s10"]


def _llm(response: str):
    return toolkit.scripted_llm({"relevant": response}, key_fn=lambda m: "relevant")


def test_extract_reproduces_answer_key():
    result = sol.extract(_PAPER, sol.ExtractionSchema, call_llm=_llm(_SCRIPTED["papers"]["relevant"]))
    assert result.review_queue == []
    assert result.data is not None
    for name, expected in _ANSWER_KEY.items():
        assert result.fields[name]["value"] == expected["value"]
        assert result.fields[name]["confidence"] == "verified"


def test_fabricated_quote_queues_only_that_field():
    result = sol.extract(_PAPER, sol.ExtractionSchema, call_llm=_llm(_SCRIPTED["_bad"]["fabricated_quote"]))
    queued = {q["field"] for q in result.review_queue}
    assert "main_finding" in queued
    assert "study_design" in result.fields
    assert result.data is None


def test_coercible_int_queues_sample_size_and_keeps_the_other_three():
    result = sol.extract(_PAPER, sol.ExtractionSchema, call_llm=_llm(_SCRIPTED["_bad"]["coercible_int"]))
    queued = {q["field"] for q in result.review_queue}
    assert queued == {"sample_size"}
    assert set(result.fields) == {"study_design", "country", "main_finding"}
    for name in result.fields:
        assert result.fields[name]["confidence"] == "verified"
    assert result.data is None


def test_schema_is_strict_about_int():
    with pytest.raises(Exception):
        sol.ExtractionSchema(study_design="x", main_finding="y", sample_size="7", country=None)


def test_all_four_fields_are_required_but_two_are_nullable():
    required = {n for n, f in sol.ExtractionSchema.model_fields.items() if f.is_required()}
    assert required == {"study_design", "sample_size", "country", "main_finding"}
    # nullable-but-required: None is accepted, absence is not
    ok = sol.ExtractionSchema(study_design="x", sample_size=None, country=None, main_finding="y")
    assert ok.sample_size is None
    with pytest.raises(Exception):
        sol.ExtractionSchema(study_design="x", main_finding="y")


class _SchemaWithOptional(BaseModel):
    model_config = ConfigDict(strict=True)
    study_design: StrictStr
    notes: StrictStr | None = None  # genuinely optional: has a default


def test_optional_field_failure_does_not_block_data():
    response = json.dumps({
        "fields": {
            "study_design": {"value": "retrospective evaluation study",
                             "evidence_quote": "We conducted a retrospective evaluation study"},
            "notes": {"value": 123, "evidence_quote": "We conducted a retrospective evaluation study"},
        }
    })
    fake = toolkit.scripted_llm({"relevant": response}, key_fn=lambda m: "relevant")
    result = sol.extract(_PAPER, _SchemaWithOptional, call_llm=fake)
    assert {q["field"] for q in result.review_queue} == {"notes"}
    assert "study_design" in result.fields
    assert result.data is not None  # the only required field passed
    assert result.data.notes is None
