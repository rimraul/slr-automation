"""Strict schema for `run_mode.json` (`track2_pipeline_readiness_plan_
2026-08-30.md` §11.3, step 11).

Schema/validation only -- no computation or writing logic here. Building
the actual `run_mode.json` for a real reference run (counting per-stage
`AIUseEvent`s, stamping the live loopback port into `access_info`, etc.)
is a later step's driver, once the reference cassette/run actually exists.
This module exists so that driver has something strict to validate its
output against, and so tests can assert the shape (including the §11.3
`sum(calls) == 50` invariant) before that driver is written.
"""
from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, model_validator

_HEX64 = r"^[0-9a-f]{64}$"

# The exactly-six LLM model-call stages (round-11 finding 1 -- these are
# both `AIUseEvent.stage` values AND LLM calls).
PER_STAGE_BACKEND_STAGES: tuple = (
    "search_strategy_query_expansion",
    "search_strategy_db_translation",
    "title_abstract_screening",
    "screening_qc_second_rater",
    "synthesis_claim_map",
    "synthesis_claim_refine",
)

# The seventh AIUseEvent stage this run produces -- a non-LLM tool, zero
# model calls.
NON_MODEL_TOOL_EVENT_KEYS: tuple = ("dedup_exact",)

# Model calls with no AIUseEvent stage at all.
NON_INSTRUMENTED_MODEL_CALL_KEYS: tuple = ("full_text_eligibility", "extraction")

EXPECTED_TOTAL_MODEL_CALLS = 50


class ModelCallsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected: int
    served: int


class StageBackendEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transport: Literal["replay"]
    access_info: str
    calls: int


class NonModelToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_kind: Literal["non_llm_tool"]
    model_calls: Literal[0] = 0


class NonInstrumentedModelCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transport: Literal["replay"]
    count: int


def _hex64(value: str, *, field: str) -> str:
    import re
    if not re.match(_HEX64, value):
        raise ValueError(f"{field} is not a sha256 hex digest: {value!r}")
    return value


class RunModeDocument(BaseModel):
    """The strict shape of `run_mode.json` (§11.3). Every field here is a
    frozen `Literal` or a validated hash/count -- no free-text/unconstrained
    field exists on this document (mirrors `ReferenceRunSpec`'s own
    discipline in `slr_pipeline.reference_spec`)."""

    model_config = ConfigDict(extra="forbid")

    run_mode_schema_version: Literal["1"] = "1"
    mode_profile: Literal["track2_offline"] = "track2_offline"
    pipeline_network: Literal["loopback_only"] = "loopback_only"
    llm_transport: Literal["replay"] = "replay"
    replay_cassette_sha256: str
    search_source: Literal["fixture"] = "fixture"
    fixture_corpus_digest: str
    full_text_acquisition: Literal["local_prepositioned"] = "local_prepositioned"
    bertopic: Literal["not_run"] = "not_run"
    dedup_mode: Literal["exact"] = "exact"
    spend_usd: Literal[0.0] = 0.0
    model_calls: ModelCallsSummary
    per_stage_backend: Dict[str, StageBackendEntry]
    non_model_tool_events: Dict[str, NonModelToolEvent]
    non_instrumented_model_calls: Dict[str, NonInstrumentedModelCall]

    @model_validator(mode="after")
    def _check_hashes(self) -> "RunModeDocument":
        _hex64(self.replay_cassette_sha256, field="replay_cassette_sha256")
        _hex64(self.fixture_corpus_digest, field="fixture_corpus_digest")
        return self

    @model_validator(mode="after")
    def _check_per_stage_backend_keys(self) -> "RunModeDocument":
        expected = set(PER_STAGE_BACKEND_STAGES)
        actual = set(self.per_stage_backend)
        if actual != expected:
            raise ValueError(
                f"per_stage_backend must have exactly the keys {sorted(expected)}, got {sorted(actual)}"
            )
        return self

    @model_validator(mode="after")
    def _check_non_model_tool_events(self) -> "RunModeDocument":
        expected = set(NON_MODEL_TOOL_EVENT_KEYS)
        actual = set(self.non_model_tool_events)
        if actual != expected:
            raise ValueError(
                f"non_model_tool_events must have exactly the keys {sorted(expected)}, got {sorted(actual)}"
            )
        for key, entry in self.non_model_tool_events.items():
            if entry.model_calls != 0:
                raise ValueError(f"non_model_tool_events[{key!r}].model_calls must be 0, got {entry.model_calls}")
        return self

    @model_validator(mode="after")
    def _check_non_instrumented_keys(self) -> "RunModeDocument":
        expected = set(NON_INSTRUMENTED_MODEL_CALL_KEYS)
        actual = set(self.non_instrumented_model_calls)
        if actual != expected:
            raise ValueError(
                f"non_instrumented_model_calls must have exactly the keys {sorted(expected)}, got {sorted(actual)}"
            )
        return self

    @model_validator(mode="after")
    def _check_call_total(self) -> "RunModeDocument":
        stage_total = sum(entry.calls for entry in self.per_stage_backend.values())
        non_instrumented_total = sum(entry.count for entry in self.non_instrumented_model_calls.values())
        total = stage_total + non_instrumented_total
        if total != self.model_calls.expected:
            raise ValueError(
                f"sum(per_stage_backend calls) + sum(non_instrumented counts) = {total} "
                f"!= model_calls.expected ({self.model_calls.expected})"
            )
        if self.model_calls.expected != self.model_calls.served:
            raise ValueError(
                f"model_calls.expected ({self.model_calls.expected}) != "
                f"model_calls.served ({self.model_calls.served})"
            )
        if self.model_calls.expected != EXPECTED_TOTAL_MODEL_CALLS:
            raise ValueError(
                f"model_calls.expected must be {EXPECTED_TOTAL_MODEL_CALLS} "
                f"(§11.3's fixed reference-run total), got {self.model_calls.expected}"
            )
        return self
