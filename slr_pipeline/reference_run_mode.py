"""Track 2 plan §11.3: the `run_mode.json` strict schema + its derived
`run_mode_statement.md` prose file -- HANDOFF §1 check 8 ("a written
statement of which stages ran in real vs. synthetic mode").

This module is intentionally split out of both its producer
(`reference_run_child.py`, which writes these two files at the end of a
real reference run) and its future consumer (`reference_acceptance.py`,
step 16, which re-validates `run_mode.json` against this same schema and
checks `run_mode_statement.md`'s regeneration equality) so neither has to
duplicate the schema.

`RunModeDocument` enforces the plan's four structural rules verbatim
(§11.3, round-11 finding 1):
  1. `per_stage_backend` has exactly the six LLM model-call stages (the
     `AIUseEvent.stage` values that are also LLM calls).
  2. `non_model_tool_events` has exactly one key, `dedup_exact` -- the
     seventh `AIUseEvent` stage, a non-LLM tool with zero model calls.
  3. `non_instrumented_model_calls` has exactly `full_text_eligibility` +
     `extraction` -- model calls with no `AIUseEvent` stage.
  4. `sum(per_stage_backend[*].calls) + sum(non_instrumented[*].count) ==
     model_calls.expected`, and `non_model_tool_events` contribute 0 (that
     last part is structural: `NonModelToolEvent.model_calls` is pinned to
     the literal `0`, not merely asserted).
  5. `replay_cassette_sha256`/`fixture_corpus_digest` are mode-aware
     (Track 3 launcher-wiring plan, Unit B): both required (non-`None`)
     for `mode_profile == "track2_offline"` (today's guarantee, preserved
     exactly); both forbidden (must be `None`) for `track3_live`, which
     has no cassette or synthetic-fixture provenance to report.
  6. `resolution_suggestion_calls`/`resolution_suggestion_interrupted`
     (LLM-resolution plan §3.4, round-4/5 fixes) are a Track-3-live-only
     sibling of rule 1, not a seventh member of it: `track2_offline` always
     carries `resolution_suggestion_calls == {}` and
     `resolution_suggestion_interrupted == 0` (there is no cassette
     `ReplaySurface` for this stage); `track3_live`'s
     `resolution_suggestion_calls` keys are a subset of
     `{"extraction_resolution_suggestion"}`, and both fields are folded
     into rule 4's grand-total identity.

`render_run_mode_statement_md()` is a pure function of a `RunModeDocument`
-- no other input -- so a downstream checker can regenerate it from a
bundled `run_mode.json` and byte-compare (plan §10.3 row: "regeneration
equality from `run_mode.json`").
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, conint, constr, model_validator

from slr_pipeline.extraction.resolution_schemas import EXTRACTION_RESOLUTION_SUGGESTION_STAGE

_HEX64 = r"^[0-9a-f]{64}$"

# The one Track-3-live-only stage `resolution_suggestion_calls` (below) can
# ever carry a key for -- not one of the 6 `SURFACE_TO_PER_STAGE_KEY` stages,
# since it has no Track-2/cassette-replay `ReplaySurface` counterpart.
_RESOLUTION_SUGGESTION_STAGE_KEYS = frozenset({EXTRACTION_RESOLUTION_SUGGESTION_STAGE})

# The 8 `ReplaySurface` values (`slr_pipeline.replay.cassette.ReplaySurface`)
# split into the 6 that get their own `per_stage_backend` entry (each an
# `AIUseEvent.stage` *and* an LLM call) and the 2 that are LLM calls with no
# `AIUseEvent` stage of their own (§11.3's `non_instrumented_model_calls`).
# `ft_retrieval` does not appear on either side -- the lexical FT retriever
# is non-model (cassette.py's own comment).
SURFACE_TO_PER_STAGE_KEY: Dict[str, str] = {
    "query_expansion": "search_strategy_query_expansion",
    "db_translation": "search_strategy_db_translation",
    "title_abstract_screening": "title_abstract_screening",
    "qc_second_rater": "screening_qc_second_rater",
    "synthesis_map": "synthesis_claim_map",
    "synthesis_refine": "synthesis_claim_refine",
}

SURFACE_TO_NON_INSTRUMENTED_KEY: Dict[str, str] = {
    "ft_adjudication": "full_text_eligibility",
    "extraction_batch": "extraction",
}

_NON_MODEL_TOOL_EVENT_KEYS = frozenset({"dedup_exact"})


class RunModeBuildError(RuntimeError):
    """Raised by `build_run_mode_document()` when `served_calls` does not
    match the cassette's own per-surface `expected_calls` exactly, and by
    `build_run_mode_document_live()` when `ai_use_events` contains a stage
    with no `run_mode.json` schema slot.

    `RunModeDocument`'s own `_check_shape` validator only checks the GRAND
    total (`sum(per_stage_backend[*].calls) + sum(non_instrumented[*].count)
    == model_calls.expected`) -- two compensating errors on different
    surfaces (one under-served, another over-served by the same amount)
    would sum to the right total while still mis-stating each individual
    stage's call count. This per-surface check closes that gap; it is the
    real source of the "fail closed on an under/mis-served run" guarantee,
    not the grand-total check alone. `build_run_mode_document_live()`'s own
    unknown-stage check closes a related gap: silently dropping an
    unrecognized stage's calls would also sum to a wrong-but-internally-
    consistent total."""


def _check_served_matches_cassette_per_surface(cassette, surface_totals: Dict[str, int]) -> None:
    expected_surface_totals = cassette.surface_totals()
    all_surfaces = set(expected_surface_totals) | set(surface_totals)
    mismatches = {
        surface: {"served": surface_totals.get(surface, 0), "expected": expected_surface_totals.get(surface, 0)}
        for surface in all_surfaces
        if surface_totals.get(surface, 0) != expected_surface_totals.get(surface, 0)
    }
    if mismatches:
        raise RunModeBuildError(
            f"served_calls does not match the cassette's own per-surface expected_calls -- "
            f"mismatches: {mismatches}. This should be impossible after "
            f"ReplayServer.assert_complete() has already passed; call this only after that."
        )


class ModelCallsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    expected: conint(ge=0)
    served: conint(ge=0)


class PerStageBackendEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    transport: str
    access_info: str
    calls: conint(ge=0)


class NonModelToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tool_kind: str
    # `Literal[0]`, not merely a checked bound -- a non-LLM tool event can
    # never itself carry a model call (plan §11.3 rule 4, second half).
    # `conint(ge=0, le=0)` would say the same thing but reads as an
    # ordinary small-range bound rather than a pin, inviting a future
    # "widen le=0 a little" edit that would silently defeat the invariant.
    model_calls: Literal[0]


class NonInstrumentedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    transport: str
    count: conint(ge=0)


class RunModeDocument(BaseModel):
    """The exact `run_mode.json` shape (plan §11.3). A strict, comment-free
    JSON document -- any explanatory prose belongs in the separate,
    derived `run_mode_statement.md`, not here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_mode_schema_version: str = "1"
    mode_profile: str
    pipeline_network: str
    llm_transport: str
    replay_cassette_sha256: Optional[constr(pattern=_HEX64)] = None
    search_source: str
    fixture_corpus_digest: Optional[constr(pattern=_HEX64)] = None
    full_text_acquisition: str
    bertopic: str
    dedup_mode: str
    spend_usd: float
    model_calls: ModelCallsSummary
    per_stage_backend: Dict[str, PerStageBackendEntry]
    non_model_tool_events: Dict[str, NonModelToolEvent]
    non_instrumented_model_calls: Dict[str, NonInstrumentedEntry]
    # Track-3-live-only sibling of `per_stage_backend` (round-4 fix, plan
    # §3.4): a `track2_offline` document always carries `{}` here -- there
    # is no cassette `ReplaySurface` for this stage, so it must never be
    # forced into `per_stage_backend`'s fixed six-key set.
    resolution_suggestion_calls: Dict[str, PerStageBackendEntry] = Field(default_factory=dict)
    # Count of `...attempt-*.interrupted_unknown.json` archives (round-5
    # addition): the one recovery path whose true call outcome is genuinely
    # unknown. Forbidden nonzero for `track2_offline`, same as
    # `resolution_suggestion_calls`.
    resolution_suggestion_interrupted: conint(ge=0) = 0

    @model_validator(mode="after")
    def _check_shape(self) -> "RunModeDocument":
        if self.mode_profile == "track2_offline":
            if self.replay_cassette_sha256 is None or self.fixture_corpus_digest is None:
                raise ValueError(
                    "track2_offline requires both replay_cassette_sha256 and "
                    "fixture_corpus_digest to be present"
                )
            if self.resolution_suggestion_calls:
                raise ValueError(
                    "track2_offline must not carry resolution_suggestion_calls -- "
                    f"got {sorted(self.resolution_suggestion_calls)}; Track 2 never "
                    "produces the extraction_resolution_suggestion stage"
                )
            if self.resolution_suggestion_interrupted != 0:
                raise ValueError(
                    "track2_offline must not carry a nonzero resolution_suggestion_interrupted, "
                    f"got {self.resolution_suggestion_interrupted}"
                )
        elif self.mode_profile == "track3_live":
            if self.replay_cassette_sha256 is not None or self.fixture_corpus_digest is not None:
                raise ValueError(
                    "track3_live must not carry a replay_cassette_sha256 or "
                    "fixture_corpus_digest -- a live run has no synthetic-fixture "
                    "or cassette provenance"
                )
            if not set(self.resolution_suggestion_calls) <= _RESOLUTION_SUGGESTION_STAGE_KEYS:
                raise ValueError(
                    f"resolution_suggestion_calls keys must be a subset of "
                    f"{sorted(_RESOLUTION_SUGGESTION_STAGE_KEYS)}, got {sorted(self.resolution_suggestion_calls)}"
                )
        else:
            raise ValueError(
                f"unknown mode_profile {self.mode_profile!r} -- no run_mode shape rule defined for it"
            )
        expected_stage_keys = set(SURFACE_TO_PER_STAGE_KEY.values())
        if set(self.per_stage_backend) != expected_stage_keys:
            raise ValueError(
                f"per_stage_backend keys must be exactly {sorted(expected_stage_keys)}, "
                f"got {sorted(self.per_stage_backend)}"
            )
        if set(self.non_model_tool_events) != _NON_MODEL_TOOL_EVENT_KEYS:
            raise ValueError(
                f"non_model_tool_events keys must be exactly {sorted(_NON_MODEL_TOOL_EVENT_KEYS)}, "
                f"got {sorted(self.non_model_tool_events)}"
            )
        expected_non_instr_keys = set(SURFACE_TO_NON_INSTRUMENTED_KEY.values())
        if set(self.non_instrumented_model_calls) != expected_non_instr_keys:
            raise ValueError(
                f"non_instrumented_model_calls keys must be exactly {sorted(expected_non_instr_keys)}, "
                f"got {sorted(self.non_instrumented_model_calls)}"
            )
        total = (
            sum(e.calls for e in self.per_stage_backend.values())
            + sum(e.count for e in self.non_instrumented_model_calls.values())
            + sum(e.calls for e in self.resolution_suggestion_calls.values())
        )
        if total != self.model_calls.expected:
            raise ValueError(
                f"sum(per_stage_backend[*].calls) + sum(non_instrumented_model_calls[*].count) == {total}, "
                f"must equal model_calls.expected == {self.model_calls.expected}"
            )
        return self


def _surface_totals_from_served(cassette, served_calls: Dict[str, int]) -> Dict[str, int]:
    """Re-keys `ReplayServer.served_calls()` (a `Dict[entry_id, count]`) by
    `surface` instead, the same aggregation `ReplayCassette.surface_totals()`
    does for `expected_calls`. Not sourced from `surface_totals()` itself:
    this must reflect what the server actually served, not what the
    cassette merely promised -- the two are only guaranteed equal because
    the caller has already run `ReplayServer.assert_complete()` first."""
    surface_by_entry_id = {e.entry_id: e.surface for e in cassette.entries}
    totals: Dict[str, int] = {}
    for entry_id, count in served_calls.items():
        surface = surface_by_entry_id[entry_id]
        totals[surface] = totals.get(surface, 0) + count
    return totals


def build_run_mode_document(
    *,
    cassette,
    served_calls: Dict[str, int],
    replay_cassette_sha256: str,
    fixture_corpus_digest: str,
    reporting_context: Dict[str, object],
    port: int,
) -> RunModeDocument:
    """Builds the one `RunModeDocument` a Track-2 reference run produces.

    `reporting_context` is `cfg.reference_reporting_context` (already built
    by `reference_spec.reference_reporting_context()` and threaded onto the
    `ReviewConfig` at spec-build time) -- `llm_transport`/`search_source`/
    `bertopic`/`dedup_mode` are read from there rather than re-derived, so
    this document cannot silently drift from the one already embedded in
    the reporting tree (HANDOFF check 8: "consistent with
    `ai_traceability.json`")."""
    surface_totals = _surface_totals_from_served(cassette, served_calls)
    _check_served_matches_cassette_per_surface(cassette, surface_totals)
    access_info = f"http://127.0.0.1:{port}/v1"

    per_stage_backend = {
        schema_key: PerStageBackendEntry(
            transport="replay",
            access_info=access_info,
            calls=surface_totals.get(surface, 0),
        )
        for surface, schema_key in SURFACE_TO_PER_STAGE_KEY.items()
    }
    non_instrumented_model_calls = {
        schema_key: NonInstrumentedEntry(
            transport="replay",
            count=surface_totals.get(surface, 0),
        )
        for surface, schema_key in SURFACE_TO_NON_INSTRUMENTED_KEY.items()
    }
    non_model_tool_events = {
        "dedup_exact": NonModelToolEvent(tool_kind="non_llm_tool", model_calls=0),
    }
    served_total = sum(served_calls.values())
    expected_total = sum(e.expected_calls for e in cassette.entries)

    return RunModeDocument(
        mode_profile="track2_offline",
        pipeline_network="loopback_only",
        llm_transport=reporting_context["llm_transport"],
        replay_cassette_sha256=replay_cassette_sha256,
        search_source=reporting_context["search_source"],
        fixture_corpus_digest=fixture_corpus_digest,
        full_text_acquisition="local_prepositioned",
        bertopic=reporting_context["bertopic"],
        dedup_mode=reporting_context["dedup_mode"],
        spend_usd=0.0,
        model_calls=ModelCallsSummary(expected=expected_total, served=served_total),
        per_stage_backend=per_stage_backend,
        non_model_tool_events=non_model_tool_events,
        non_instrumented_model_calls=non_instrumented_model_calls,
    )


def _per_stage_counts_from_live_events(ai_use_events) -> Dict[str, int]:
    """Tallies real `AIUseEvent`s by their own `.stage` attribute. Unlike
    the Track-2 builder's cassette-`surface`-then-`SURFACE_TO_PER_STAGE_KEY`
    remap, a live run's `AIUseEvent.stage` values ARE already exactly the
    six `per_stage_backend` schema keys verbatim (confirmed against every
    real stage= call site: `nodes/search_strategy_builder.py`,
    `nodes/reporting.py`, `nodes/synthesis.py`, `nodes/screening_qc.py`) --
    no surface-to-schema-key indirection exists or is needed here."""
    counts: Dict[str, int] = {}
    for event in ai_use_events:
        counts[event.stage] = counts.get(event.stage, 0) + 1
    return counts


def build_run_mode_document_live(
    *,
    ai_use_events,
    extraction_call_count: int,
    full_text_eligibility_call_count: int,
    spend_usd: float,
    full_text_acquisition: str,
    bertopic: str,
    port: int,
    resolution_suggestion_interrupted: int = 0,
) -> RunModeDocument:
    """Builds the one `RunModeDocument` a Track-3 live run produces.

    `ai_use_events` is any iterable of real `AIUseEvent`s (or objects
    exposing the same `.stage` attribute) already loaded by the caller --
    e.g. `[vf.model for vf in _load_verified_events(event_store)]`
    (`nodes/reporting.py`), the same durable, re-verified source
    `finalize_bundle()` itself uses. `extraction_call_count` /
    `full_text_eligibility_call_count` likewise come from the caller's own
    already-loaded `_load_extraction_bundle()` / `capture_ft_native_
    provenance()` results (neither stage has an `AIUseEvent` of its own,
    per `ai_traceability_report.py`'s `_STAGE_EVENT_STAGES` mapping) --
    this function never re-derives them from a call log, so it cannot
    silently drift from what `finalize_bundle()` will later independently
    re-verify and stage.

    There is no pre-declared cassette promise for a live run, so
    `model_calls.expected == model_calls.served` always, by construction
    -- this is a live-run-specific semantic, not a bug.

    Scope boundary, deliberately not addressed here: `nodes/quality_
    appraisal.py` makes real, zero-`AIUseEvent`-instrumented OpenAI calls
    (`ai_traceability_report.py`'s own `_NOT_INSTRUMENTED_REASONS
    ['quality_appraisal']`), but `non_instrumented_model_calls` has no
    schema slot for it (`_check_shape` locks that dict to exactly
    `{full_text_eligibility, extraction}`) and this builder has no
    parameter for it either. `quality_appraisal_node` is wired into
    `graphs/slr.py`/`graphs/rapid.py`, never `graphs/scoping.py` -- the
    Track 3 MVP plan runs the scoping graph only, so this gap does not
    affect the planned MVP run. It would need a schema + builder
    extension before any future Track 3 run used a non-scoping
    `review_type`.

    Preflight note for Unit C (launcher-wiring): this builder's
    unaccounted-stage check (below) fail-closes on `stage="dedup_embedding"`
    exactly like any other unrecognized stage, since `known_stage_keys`
    only ever names `dedup_exact`, the one non-model tool event this
    schema has a slot for (`nodes/dedup.py`'s `cfg.dedup.mode == "auto"`
    default dispatches to the embedding path, `stage="dedup_embedding"`,
    unless `mode == "exact"` explicitly -- Track 2 structurally pins this
    via `reference_spec.py`'s `dedup_mode: Literal["exact"]`; no
    equivalent Track 3 pin exists yet). That is correct fail-closed
    behavior for THIS function, but it only runs after `graph.invoke()`
    has already returned -- i.e. after real spend has already occurred.
    Unit C's launcher must assert `cfg.dedup.mode == "exact"` as its own
    preflight check, before `graph.invoke()`, rather than relying on this
    end-of-run check to catch a misconfigured live run."""
    access_info = f"http://127.0.0.1:{port}/v1"
    stage_counts = _per_stage_counts_from_live_events(ai_use_events)

    known_stage_keys = (
        set(SURFACE_TO_PER_STAGE_KEY.values())
        | _NON_MODEL_TOOL_EVENT_KEYS
        | _RESOLUTION_SUGGESTION_STAGE_KEYS
    )
    unaccounted = {stage: count for stage, count in stage_counts.items() if stage not in known_stage_keys}
    if unaccounted:
        raise RunModeBuildError(
            f"ai_use_events contains AIUseEvent stage(s) with no run_mode.json schema slot -- "
            f"{unaccounted}. Left unhandled, these calls would silently vanish from "
            f"model_calls.expected/served and from spend accounting rather than surfacing an "
            f"error (e.g. stage='synthesis_imrad_narrative' if "
            f"cfg.synthesis.generate_legacy_imrad_narrative is ever True for a live run -- "
            f"Track 2 structurally pins that flag to False via reference_spec.py, but nothing "
            f"does so for Track 3 today)."
        )

    per_stage_backend = {
        schema_key: PerStageBackendEntry(
            transport="live",
            access_info=access_info,
            calls=stage_counts.get(schema_key, 0),
        )
        for schema_key in SURFACE_TO_PER_STAGE_KEY.values()
    }
    non_instrumented_model_calls = {
        "full_text_eligibility": NonInstrumentedEntry(transport="live", count=full_text_eligibility_call_count),
        "extraction": NonInstrumentedEntry(transport="live", count=extraction_call_count),
    }
    resolution_suggestion_calls = {
        schema_key: PerStageBackendEntry(
            transport="live",
            access_info=access_info,
            calls=stage_counts.get(schema_key, 0),
        )
        for schema_key in _RESOLUTION_SUGGESTION_STAGE_KEYS
    }
    non_model_tool_events = {
        "dedup_exact": NonModelToolEvent(tool_kind="non_llm_tool", model_calls=0),
    }
    total_calls = (
        sum(e.calls for e in per_stage_backend.values())
        + sum(e.count for e in non_instrumented_model_calls.values())
        + sum(e.calls for e in resolution_suggestion_calls.values())
    )

    return RunModeDocument(
        mode_profile="track3_live",
        pipeline_network="internet_via_cost_cap_proxy",
        llm_transport="live",
        search_source="live",
        full_text_acquisition=full_text_acquisition,
        bertopic=bertopic,
        dedup_mode="exact",
        spend_usd=spend_usd,
        model_calls=ModelCallsSummary(expected=total_calls, served=total_calls),
        per_stage_backend=per_stage_backend,
        non_model_tool_events=non_model_tool_events,
        non_instrumented_model_calls=non_instrumented_model_calls,
        resolution_suggestion_calls=resolution_suggestion_calls,
        resolution_suggestion_interrupted=resolution_suggestion_interrupted,
    )


def render_run_mode_statement_md(doc: RunModeDocument) -> str:
    """Pure function of `doc` alone -- a downstream checker regenerates
    this from a bundled `run_mode.json` and byte-compares (plan §10.3
    row: "regeneration equality from `run_mode.json`")."""
    title = "Track 2 reference run" if doc.mode_profile == "track2_offline" else "Track 3 live run"
    lines = [
        f"# {title} -- mode statement",
        "",
        "Derived, byte-for-byte reproducibly, from `run_mode.json` "
        f"(schema version `{doc.run_mode_schema_version}`). States which "
        "stages ran in real vs. synthetic mode for this run.",
        "",
        f"- Mode profile: `{doc.mode_profile}`",
        f"- Pipeline network: `{doc.pipeline_network}`",
        f"- LLM transport: `{doc.llm_transport}`",
        f"- Search source: `{doc.search_source}`",
        f"- Full-text acquisition: `{doc.full_text_acquisition}`",
        f"- BERTopic: `{doc.bertopic}`",
        f"- Deduplication mode: `{doc.dedup_mode}`",
        f"- Spend (USD): `{doc.spend_usd:.2f}`",
        f"- Replay cassette sha256: `{doc.replay_cassette_sha256 if doc.replay_cassette_sha256 is not None else 'not applicable (live run)'}`",
        f"- Fixture corpus digest: `{doc.fixture_corpus_digest if doc.fixture_corpus_digest is not None else 'not applicable (live run)'}`",
        "",
        "## Model calls",
        "",
        f"Expected: {doc.model_calls.expected}. Served: {doc.model_calls.served}.",
        "",
        "| Stage | Transport | Access info | Calls |",
        "|---|---|---|---|",
    ]
    for key in sorted(doc.per_stage_backend):
        entry = doc.per_stage_backend[key]
        lines.append(f"| {key} | {entry.transport} | {entry.access_info} | {entry.calls} |")
    lines += [
        "",
        "## Non-model tool events",
        "",
        "| Event | Tool kind | Model calls |",
        "|---|---|---|",
    ]
    for key in sorted(doc.non_model_tool_events):
        entry = doc.non_model_tool_events[key]
        lines.append(f"| {key} | {entry.tool_kind} | {entry.model_calls} |")
    lines += [
        "",
        "## Non-instrumented model calls",
        "",
        "| Stage | Transport | Count |",
        "|---|---|---|",
    ]
    for key in sorted(doc.non_instrumented_model_calls):
        entry = doc.non_instrumented_model_calls[key]
        lines.append(f"| {key} | {entry.transport} | {entry.count} |")
    if doc.mode_profile == "track3_live":
        lines += [
            "",
            "## Resolution suggestion calls",
            "",
            "| Stage | Transport | Access info | Calls |",
            "|---|---|---|---|",
        ]
        for key in sorted(doc.resolution_suggestion_calls):
            entry = doc.resolution_suggestion_calls[key]
            lines.append(f"| {key} | {entry.transport} | {entry.access_info} | {entry.calls} |")
    if doc.resolution_suggestion_interrupted:
        lines += [
            "",
            f"{doc.resolution_suggestion_interrupted} interrupted/unknown-outcome calls, "
            "excluded from the totals above and never auto-retried.",
        ]
    lines.append("")
    return "\n".join(lines)
