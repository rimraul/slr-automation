"""`slr_pipeline.run_mode` -- the strict `run_mode.json` schema
(`track2_pipeline_readiness_plan_2026-08-30.md` §11.3, step 11). Schema/
validation only; no `run_mode.json` is produced by this step (a later
step's driver builds the real document once the reference cassette/run
actually exists) -- these tests validate the schema against the plan's
own §11.3 worked example."""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from slr_pipeline.run_mode import EXPECTED_TOTAL_MODEL_CALLS, RunModeDocument

# The plan's own §11.3 worked example, verbatim.
_EXAMPLE = {
    "replay_cassette_sha256": "a" * 64,
    "fixture_corpus_digest": "b" * 64,
    "model_calls": {"expected": 50, "served": 50},
    "per_stage_backend": {
        "search_strategy_query_expansion": {"transport": "replay", "access_info": "http://127.0.0.1:0/v1", "calls": 1},
        "search_strategy_db_translation":  {"transport": "replay", "access_info": "http://127.0.0.1:0/v1", "calls": 1},
        "title_abstract_screening":        {"transport": "replay", "access_info": "http://127.0.0.1:0/v1", "calls": 5},
        "screening_qc_second_rater":       {"transport": "replay", "access_info": "http://127.0.0.1:0/v1", "calls": 6},
        "synthesis_claim_map":             {"transport": "replay", "access_info": "http://127.0.0.1:0/v1", "calls": 10},
        "synthesis_claim_refine":          {"transport": "replay", "access_info": "http://127.0.0.1:0/v1", "calls": 4},
    },
    "non_model_tool_events": {"dedup_exact": {"tool_kind": "non_llm_tool", "model_calls": 0}},
    "non_instrumented_model_calls": {
        "full_text_eligibility": {"transport": "replay", "count": 18},
        "extraction":            {"transport": "replay", "count": 5},
    },
}


def test_plan_worked_example_validates():
    doc = RunModeDocument(**_EXAMPLE)
    assert doc.model_calls.expected == EXPECTED_TOTAL_MODEL_CALLS == 50
    assert doc.run_mode_schema_version == "1"
    assert doc.mode_profile == "track2_offline"


def test_call_total_must_equal_expected():
    bad = copy.deepcopy(_EXAMPLE)
    bad["model_calls"] = {"expected": 51, "served": 51}
    with pytest.raises(ValidationError, match="!= model_calls.expected"):
        RunModeDocument(**bad)


def test_expected_and_served_must_match():
    bad = copy.deepcopy(_EXAMPLE)
    bad["model_calls"] = {"expected": 50, "served": 49}
    with pytest.raises(ValidationError, match="expected.*!=.*served"):
        RunModeDocument(**bad)


def test_expected_total_is_fixed_at_fifty():
    bad = copy.deepcopy(_EXAMPLE)
    bad["model_calls"] = {"expected": 49, "served": 49}
    bad["per_stage_backend"]["title_abstract_screening"]["calls"] = 4
    with pytest.raises(ValidationError, match="must be 50"):
        RunModeDocument(**bad)


def test_per_stage_backend_must_have_exactly_six_keys():
    bad = copy.deepcopy(_EXAMPLE)
    del bad["per_stage_backend"]["synthesis_claim_refine"]
    with pytest.raises(ValidationError, match="per_stage_backend must have exactly"):
        RunModeDocument(**bad)


def test_non_model_tool_events_must_be_exactly_dedup_exact():
    bad = copy.deepcopy(_EXAMPLE)
    bad["non_model_tool_events"] = {"other_tool": {"tool_kind": "non_llm_tool", "model_calls": 0}}
    with pytest.raises(ValidationError, match="non_model_tool_events must have exactly"):
        RunModeDocument(**bad)


def test_non_model_tool_events_model_calls_must_be_zero():
    bad = copy.deepcopy(_EXAMPLE)
    bad["non_model_tool_events"] = {"dedup_exact": {"tool_kind": "non_llm_tool", "model_calls": 1}}
    with pytest.raises(ValidationError):
        RunModeDocument(**bad)


def test_non_instrumented_keys_must_be_exact():
    bad = copy.deepcopy(_EXAMPLE)
    bad["non_instrumented_model_calls"] = {"extraction": {"transport": "replay", "count": 23}}
    with pytest.raises(ValidationError, match="non_instrumented_model_calls must have exactly"):
        RunModeDocument(**bad)


def test_hash_fields_must_be_sha256_hex():
    bad = copy.deepcopy(_EXAMPLE)
    bad["replay_cassette_sha256"] = "not-hex"
    with pytest.raises(ValidationError, match="not a sha256 hex digest"):
        RunModeDocument(**bad)


def test_extra_field_forbidden():
    bad = copy.deepcopy(_EXAMPLE)
    bad["bogus_field"] = "x"
    with pytest.raises(ValidationError):
        RunModeDocument(**bad)
