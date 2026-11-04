"""`slr_pipeline.reference_charting_adapter` -- the reference run's own
nine-field charting schema + calibrated extraction model config
(`track2_pipeline_readiness_plan_2026-08-30.md` §4.6, step 11).

Fixtures live under `tests/reference_run/fixtures/spec/`:
  - `extraction_schema.json` / `extraction_model_config.json` -- byte-checked
    dumps (plan §0.4's own CI check convention, same as step 8's
    `scientific_projection.json`).

Their hash-target registrations live in `scripts/register_reference_hash_
targets.py`, not in `slr_pipeline.reference_charting_adapter` itself
(Track-2 packaging-boundary fix, CI run 34155102694): that module is
wheel-packaged and must not assume it sits inside a git checkout with
`tests/` as a sibling."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import scripts.register_reference_hash_targets as reg
from slr_pipeline.reference_charting_adapter import (
    REFERENCE_EXTRACTION_SCHEMA_ID,
    REFERENCE_EXTRACTION_SCHEMA_VERSION,
    REFERENCE_MODEL_CONFIG_ID,
    reference_extraction_model_config,
    reference_extraction_schema,
)
from slr_pipeline.extraction.schema import ValueKind
from slr_pipeline.hash_registry import TARGET_REGISTRY

FIXTURES = Path(__file__).parent / "reference_run" / "fixtures" / "spec"


def test_schema_has_exactly_nine_fields_in_order():
    schema = reference_extraction_schema()
    field_ids = [f.field_id for f in schema.fields]
    assert field_ids == [
        "country", "study_design", "population", "concept", "context",
        "limitations", "key_findings", "funding_sources", "funder_role",
    ]
    assert schema.schema_id == REFERENCE_EXTRACTION_SCHEMA_ID
    assert schema.version == REFERENCE_EXTRACTION_SCHEMA_VERSION


def test_key_findings_and_funding_sources_are_text_list_multiple():
    schema = reference_extraction_schema()
    by_id = {f.field_id: f for f in schema.fields}
    for multi_field in ("key_findings", "funding_sources"):
        assert by_id[multi_field].value_kind == ValueKind.TEXT_LIST
        assert by_id[multi_field].cardinality == "multiple"
    for single_field in ("country", "study_design", "population", "concept",
                         "context", "limitations", "funder_role"):
        assert by_id[single_field].value_kind == ValueKind.TEXT
        assert by_id[single_field].cardinality == "single"


def test_model_config_matches_plan_preset():
    cfg = reference_extraction_model_config()
    assert cfg.model == "gpt-5.6-terra"
    assert cfg.model_config_id == REFERENCE_MODEL_CONFIG_ID
    assert cfg.reasoning_effort == "medium"
    assert cfg.max_completion_tokens == 4096
    assert cfg.expected_provider_model == "gpt-5.6-terra"


def test_extraction_schema_fixture_matches_on_disk_bytes():
    on_disk = (FIXTURES / "extraction_schema.json").read_text(encoding="utf-8")
    regenerated = json.dumps(reference_extraction_schema().model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    assert on_disk == regenerated
    assert reg.EXTRACTION_SCHEMA_FIXTURE == FIXTURES / "extraction_schema.json"


def test_extraction_model_config_fixture_matches_on_disk_bytes():
    on_disk = (FIXTURES / "extraction_model_config.json").read_text(encoding="utf-8")
    regenerated = json.dumps(dataclasses.asdict(reference_extraction_model_config()), indent=2, sort_keys=True) + "\n"
    assert on_disk == regenerated
    assert reg.EXTRACTION_MODEL_CONFIG_FIXTURE == FIXTURES / "extraction_model_config.json"


def test_hash_targets_are_registered():
    assert "extraction_schema_sha256" in TARGET_REGISTRY
    assert "extraction_model_config_sha256" in TARGET_REGISTRY
    assert TARGET_REGISTRY["extraction_schema_sha256"]() == reg.EXTRACTION_SCHEMA_FIXTURE
    assert TARGET_REGISTRY["extraction_model_config_sha256"]() == reg.EXTRACTION_MODEL_CONFIG_FIXTURE
