from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.extraction.schema import (
    AllowedValue, ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind,
    load_extraction_schema,
)


def _spec(**overrides) -> ExtractionFieldSpec:
    defaults = dict(
        field_id="study_design", question="What study design was used?", description="...",
        value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    defaults.update(overrides)
    return ExtractionFieldSpec(**defaults)


def test_valid_field_spec_constructs():
    spec = _spec()
    assert spec.field_id == "study_design"


@pytest.mark.parametrize("bad_id", ["", "has spaces", "has/slash", "a" * 129])
def test_rejects_unsafe_field_id(bad_id):
    with pytest.raises(Exception):
        _spec(field_id=bad_id)


def test_categorical_requires_allowed_values():
    with pytest.raises(Exception, match="allowed_values"):
        _spec(value_kind=ValueKind.CATEGORICAL, allowed_values=None)


def test_categorical_rejects_empty_allowed_values():
    with pytest.raises(Exception):
        _spec(value_kind=ValueKind.CATEGORICAL, allowed_values=())


def test_categorical_rejects_duplicate_values():
    with pytest.raises(Exception, match="duplicate"):
        _spec(
            value_kind=ValueKind.CATEGORICAL,
            allowed_values=(AllowedValue(value="rct", definition="d1"), AllowedValue(value="rct", definition="d2")),
        )


def test_non_categorical_rejects_allowed_values():
    with pytest.raises(Exception, match="categorical"):
        _spec(value_kind=ValueKind.TEXT, allowed_values=(AllowedValue(value="x", definition="y"),))


def test_categorical_allowed_values_carry_definitions():
    spec = _spec(
        value_kind=ValueKind.CATEGORICAL,
        allowed_values=(AllowedValue(value="rct", definition="Randomized controlled trial"),),
    )
    assert spec.allowed_values[0].definition == "Randomized controlled trial"


def test_allowed_value_requires_non_empty_definition():
    with pytest.raises(Exception):
        AllowedValue(value="rct", definition="")


def test_schema_rejects_duplicate_field_ids():
    with pytest.raises(Exception, match="duplicate"):
        ExtractionSchema(schema_id="s", version="1", fields=[_spec(), _spec()])


def test_schema_requires_at_least_one_field():
    with pytest.raises(Exception):
        ExtractionSchema(schema_id="s", version="1", fields=[])


def test_schema_field_map():
    schema = ExtractionSchema(schema_id="s", version="1", fields=[_spec(field_id="a"), _spec(field_id="b")])
    assert set(schema.field_map()) == {"a", "b"}


def test_field_provenance_records_origin_and_rationale():
    spec = _spec(
        provenance=FieldProvenance(
            origin=FieldOrigin.INDUCTIVELY_DERIVED, version="2",
            rationale="added after calibration round 2 found systematic disagreement",
        ),
    )
    assert spec.provenance.origin == FieldOrigin.INDUCTIVELY_DERIVED
    assert "calibration round 2" in spec.provenance.rationale


def test_load_extraction_schema_from_json_file(tmp_path):
    import json

    payload = {
        "schema_id": "chapter-scoping-v1", "version": "1",
        "fields": [
            {
                "field_id": "study_design", "question": "What study design?", "description": "...",
                "value_kind": "text", "cardinality": "single",
                "provenance": {"origin": "theory_derived", "version": "1"},
            },
        ],
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    schema = load_extraction_schema(path)
    assert schema.schema_id == "chapter-scoping-v1"
    assert schema.fields[0].field_id == "study_design"


def test_schema_is_frozen():
    schema = ExtractionSchema(schema_id="s", version="1", fields=[_spec()])
    with pytest.raises(Exception):
        schema.version = "2"
