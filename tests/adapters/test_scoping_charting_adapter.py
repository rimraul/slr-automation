"""Offline tests for slr_pipeline.adapters.scoping_charting_adapter (Stage
2 of quality_appraisal_synthesis_plan_2026-08-22.md: "Charting via the
Extraction Engine"). Pure/no-LLM: schema validity, `derive_author_year`
unit tests, and `scoping_extraction_model_config`'s explicitness are all
checkable without any network/credential/PDF activity.

Section 2.3's empirical check ("does key_findings' atomic evidence-backed
structure actually produce useful extracted/not_reported outcomes")
belongs here, run EARLY per plan Section 5 point 2 -- but a genuine
against-a-real-PDF run requires a real OpenAI call this test suite must
never make. The schema-shape assertions below (six TEXT/single fields,
key_findings as TEXT_LIST/multiple, real non-blank question/description
text on every field) are the structural precondition for that check to
even be meaningful; the actual against-Berg-2023 empirical run is a
follow-up the user runs manually (see this file's own module-level note
at the bottom) -- it is not, and cannot be, part of an offline test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.adapters.scoping_charting_adapter import (
    SCOPING_CHART_SCHEMA_ID,
    SCOPING_MODEL_CONFIG_ID,
    derive_author_year,
    scoping_chart_schema,
    scoping_extraction_model_config,
)
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.extraction.schema import ExtractionSchema, ValueKind
from slr_pipeline.state import Record


# ---------------------------------------------------------------------------
# scoping_chart_schema()
# ---------------------------------------------------------------------------

_SINGLE_TEXT_FIELDS = {"country", "study_design", "population", "concept", "context", "limitations"}


def test_scoping_chart_schema_is_valid_extraction_schema():
    schema = scoping_chart_schema()
    assert isinstance(schema, ExtractionSchema)
    assert schema.schema_id == SCOPING_CHART_SCHEMA_ID


def test_scoping_chart_schema_has_seven_fields_not_nine():
    schema = scoping_chart_schema()
    field_ids = {f.field_id for f in schema.fields}
    assert field_ids == _SINGLE_TEXT_FIELDS | {"key_findings"}
    # author_year is derived deterministically, never an extraction field;
    # relevance_notes is reviewer interpretation produced AFTER extraction
    # -- neither belongs in the schema (plan Section 2.2).
    assert "author_year" not in field_ids
    assert "relevance_notes" not in field_ids


def test_scoping_chart_schema_six_fields_are_text_single():
    schema = scoping_chart_schema()
    by_id = schema.field_map()
    for field_id in _SINGLE_TEXT_FIELDS:
        field = by_id[field_id]
        assert field.value_kind == ValueKind.TEXT
        assert field.cardinality == "single"


def test_scoping_chart_schema_key_findings_is_text_list_multiple():
    schema = scoping_chart_schema()
    key_findings = schema.field_map()["key_findings"]
    assert key_findings.value_kind == ValueKind.TEXT_LIST
    assert key_findings.cardinality == "multiple"


def test_scoping_chart_schema_every_field_has_real_question_and_description():
    schema = scoping_chart_schema()
    for field in schema.fields:
        assert field.question.strip()
        assert field.description.strip()
        assert len(field.description.strip()) > 20  # not a placeholder stub


def test_scoping_chart_schema_field_ids_are_stable_across_calls():
    """Schema construction is deterministic -- two calls produce byte-
    identical schemas (needed for run-identity reproducibility, exactly
    like extraction_node's own document-set/schema-hash binding)."""
    first = scoping_chart_schema()
    second = scoping_chart_schema()
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


# ---------------------------------------------------------------------------
# scoping_extraction_model_config()
# ---------------------------------------------------------------------------

def test_scoping_extraction_model_config_returns_valid_config():
    config = scoping_extraction_model_config()
    assert isinstance(config, OpenAIExtractionConfig)
    assert config.model_config_id == SCOPING_MODEL_CONFIG_ID


def test_scoping_extraction_model_config_has_no_silently_defaulted_field():
    config = scoping_extraction_model_config()
    assert config.model.strip()
    assert config.model_config_id.strip()
    assert config.expected_provider_model.strip()
    assert config.reasoning_effort is not None
    assert config.max_completion_tokens > 0


def test_scoping_extraction_model_config_takes_no_parameters():
    import inspect

    sig = inspect.signature(scoping_extraction_model_config)
    assert list(sig.parameters) == []


# ---------------------------------------------------------------------------
# derive_author_year()
# ---------------------------------------------------------------------------

def test_derive_author_year_with_authors_and_year():
    record = Record(
        id="r1", source="s", source_id="r1", title="A study",
        authors=["Berg, A.", "Someone Else"], year=2023,
    )
    assert derive_author_year(record) == "Berg, A. et al., 2023"


def test_derive_author_year_with_authors_no_year():
    record = Record(id="r1", source="s", source_id="r1", title="A study", authors=["Berg, A."])
    assert derive_author_year(record) == "Berg, A. et al., n.d."


def test_derive_author_year_no_authors_falls_back_to_title():
    record = Record(
        id="r1", source="s", source_id="r1",
        title="A very long title that will be truncated for the citation fallback",
        year=2019,
    )
    result = derive_author_year(record)
    assert result.startswith("A very long title that will be tr")
    assert "2019" in result


def test_derive_author_year_no_authors_no_title():
    record = Record(id="r1", source="s", source_id="r1", title="", year=2020)
    assert derive_author_year(record) == "Unknown author, 2020"


def test_derive_author_year_is_deterministic_and_pure():
    record = Record(id="r1", source="s", source_id="r1", title="t", authors=["A"], year=2021)
    assert derive_author_year(record) == derive_author_year(record)


# ---------------------------------------------------------------------------
# Section 2.3 empirical-check follow-up note (not itself a test).
#
# A real run of scoping_chart_schema()/scoping_extraction_model_config()
# against Berg 2023's actual PDF (test_reviews/candidate_reviews/
# s13753-023-00492-z) requires a real, credentialed OpenAI call this
# offline suite deliberately never makes. The user should run this
# manually via extraction_node() directly (or a compiled scoping graph
# invocation) once OPENAI_API_KEY is set, and inspect the resulting
# checkpoint's per-field status distribution -- especially key_findings'
# own extracted/not_reported/unclear rate -- to confirm the plan Section
# 2.3 "not free" empirical assumption before treating Stage 2 as fully
# validated end to end.
# ---------------------------------------------------------------------------
