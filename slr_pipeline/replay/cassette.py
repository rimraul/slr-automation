"""
slr_pipeline.replay.cassette -- the strict `REPLAY_CASSETTE_VERSION = "1"`
schema the replay transport (`transport.py`, Track 2 plan Sec 6.1) matches
recorded model calls against.

The cassette file itself (`tests/reference_run/fixtures/corpus/replay_cassette.json`,
Sec 7.1, a step-12 deliverable) is a bare JSON list of entries -- there is
no wrapping object/version key in the file; `REPLAY_CASSETTE_VERSION` is a
Python-side pin for future schema evolution, not a JSON field.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, constr, model_validator

REPLAY_CASSETTE_VERSION = "1"

_HEX64 = constr(pattern=r"^[0-9a-f]{64}$")

# The 8 call-making surfaces (plan Sec 4.3, round-11 finding 1) -- explicitly
# no `ft_retrieval`: the lexical FT retriever is non-model.
ReplaySurface = Literal[
    "query_expansion",
    "db_translation",
    "title_abstract_screening",
    "qc_second_rater",
    "ft_adjudication",
    "extraction_batch",
    "synthesis_map",
    "synthesis_refine",
]

# The frozen Sec 4.1 model-call topology this cassette's `expected_calls`
# must sum to, per surface, on the Track-2 reference run
# (`test_cassette_surface_totals`). Sum = 50 = `CORPUS_ORACLE.total_replay_calls`.
REFERENCE_SURFACE_CALL_COUNTS: Dict[str, int] = {
    "query_expansion": 1,
    "db_translation": 1,
    "title_abstract_screening": 5,
    "qc_second_rater": 6,
    "ft_adjudication": 18,
    "extraction_batch": 5,
    "synthesis_map": 10,
    "synthesis_refine": 4,
}


class ReplayCassetteEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_id: str
    surface: ReplaySurface
    signature: _HEX64
    endpoint: Literal["/v1/chat/completions"]
    request_preview: str          # first 200 chars of the user message -- human triage only
    response_body: Dict[str, Any]
    expected_calls: PositiveInt


class ReplayCassette(BaseModel):
    """The in-memory, validated form of a `replay_cassette.json` list.
    Enforced beyond the per-entry schema: `entry_id` and `signature` are
    each unique across the cassette -- the transport matches purely by
    `signature`, so a duplicate would make a served response ambiguous."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: List[ReplayCassetteEntry]

    @model_validator(mode="after")
    def _unique_ids_and_signatures(self) -> "ReplayCassette":
        ids = [e.entry_id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("cassette entry_id values must be unique")
        sigs = [e.signature for e in self.entries]
        if len(sigs) != len(set(sigs)):
            raise ValueError("cassette signature values must be unique")
        return self

    @classmethod
    def from_list(cls, raw: List[Dict[str, Any]]) -> "ReplayCassette":
        return cls(entries=[ReplayCassetteEntry.model_validate(item) for item in raw])

    def by_signature(self) -> Dict[str, ReplayCassetteEntry]:
        return {e.signature: e for e in self.entries}

    def surface_totals(self) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for entry in self.entries:
            totals[entry.surface] = totals.get(entry.surface, 0) + entry.expected_calls
        return totals
