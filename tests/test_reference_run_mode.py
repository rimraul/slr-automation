"""Unit tests for `slr_pipeline.reference_run_mode` -- the `run_mode.json`
strict schema (plan §11.3) and its derived `run_mode_statement.md`."""
from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from slr_pipeline.reference_run_mode import (
    ModelCallsSummary,
    NonInstrumentedEntry,
    NonModelToolEvent,
    PerStageBackendEntry,
    RunModeBuildError,
    RunModeDocument,
    SURFACE_TO_NON_INSTRUMENTED_KEY,
    SURFACE_TO_PER_STAGE_KEY,
    build_run_mode_document,
    build_run_mode_document_live,
    render_run_mode_statement_md,
)
from slr_pipeline.replay.cassette import REFERENCE_SURFACE_CALL_COUNTS, ReplayCassette
from slr_pipeline.extraction.resolution_schemas import EXTRACTION_RESOLUTION_SUGGESTION_STAGE

_HEX64_A = "a" * 64
_HEX64_B = "b" * 64


def _entry(entry_id: str, surface: str, calls: int):
    return {
        "entry_id": entry_id,
        "surface": surface,
        "signature": hashlib.sha256(entry_id.encode("utf-8")).hexdigest(),
        "endpoint": "/v1/chat/completions",
        "request_preview": "x",
        "response_body": {"ok": True},
        "expected_calls": calls,
    }


def _reference_cassette() -> ReplayCassette:
    """A cassette with one entry per `ReplaySurface`, at exactly the frozen
    Track-2 reference-run counts (`REFERENCE_SURFACE_CALL_COUNTS`, sum 50)."""
    return ReplayCassette.from_list(
        [_entry(f"e-{surface}", surface, calls) for surface, calls in REFERENCE_SURFACE_CALL_COUNTS.items()]
    )


def _served_calls_matching_expected(cassette: ReplayCassette) -> dict:
    return {e.entry_id: e.expected_calls for e in cassette.entries}


def _valid_kwargs(**overrides):
    kwargs = dict(
        mode_profile="track2_offline",
        pipeline_network="loopback_only",
        llm_transport="replay",
        replay_cassette_sha256=_HEX64_A,
        search_source="fixture",
        fixture_corpus_digest=_HEX64_B,
        full_text_acquisition="local_prepositioned",
        bertopic="not_run",
        dedup_mode="exact",
        spend_usd=0.0,
        model_calls=ModelCallsSummary(expected=50, served=50),
        per_stage_backend={
            "search_strategy_query_expansion": PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=1),
            "search_strategy_db_translation": PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=1),
            "title_abstract_screening": PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=5),
            "screening_qc_second_rater": PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=6),
            "synthesis_claim_map": PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=10),
            "synthesis_claim_refine": PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=4),
        },
        non_model_tool_events={
            "dedup_exact": NonModelToolEvent(tool_kind="non_llm_tool", model_calls=0),
        },
        non_instrumented_model_calls={
            "full_text_eligibility": NonInstrumentedEntry(transport="replay", count=18),
            "extraction": NonInstrumentedEntry(transport="replay", count=5),
        },
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# RunModeDocument -- the strict schema itself
# ---------------------------------------------------------------------------

def test_run_mode_document_accepts_the_frozen_reference_run_shape():
    doc = RunModeDocument(**_valid_kwargs())
    assert doc.run_mode_schema_version == "1"
    total = sum(e.calls for e in doc.per_stage_backend.values()) + \
        sum(e.count for e in doc.non_instrumented_model_calls.values())
    assert total == doc.model_calls.expected == 50


def test_run_mode_document_rejects_missing_per_stage_backend_key():
    kwargs = _valid_kwargs()
    per_stage = dict(kwargs["per_stage_backend"])
    del per_stage["synthesis_claim_refine"]
    kwargs["per_stage_backend"] = per_stage
    kwargs["model_calls"] = ModelCallsSummary(expected=46, served=46)
    with pytest.raises(ValidationError, match="per_stage_backend keys must be exactly"):
        RunModeDocument(**kwargs)


def test_run_mode_document_rejects_extra_per_stage_backend_key():
    kwargs = _valid_kwargs()
    per_stage = dict(kwargs["per_stage_backend"])
    per_stage["some_extra_stage"] = PerStageBackendEntry(transport="replay", access_info="http://127.0.0.1:0/v1", calls=1)
    kwargs["per_stage_backend"] = per_stage
    kwargs["model_calls"] = ModelCallsSummary(expected=51, served=51)
    with pytest.raises(ValidationError, match="per_stage_backend keys must be exactly"):
        RunModeDocument(**kwargs)


def test_run_mode_document_rejects_wrong_non_model_tool_event_key():
    kwargs = _valid_kwargs()
    kwargs["non_model_tool_events"] = {"some_other_tool": NonModelToolEvent(tool_kind="non_llm_tool", model_calls=0)}
    with pytest.raises(ValidationError, match="non_model_tool_events keys must be exactly"):
        RunModeDocument(**kwargs)


def test_run_mode_document_rejects_nonzero_non_model_tool_event_calls():
    # Pinned at the field level (conint(ge=0, le=0)) -- fails before the
    # cross-field shape validator even runs.
    with pytest.raises(ValidationError):
        NonModelToolEvent(tool_kind="non_llm_tool", model_calls=1)


def test_run_mode_document_rejects_wrong_non_instrumented_keys():
    kwargs = _valid_kwargs()
    kwargs["non_instrumented_model_calls"] = {
        "full_text_eligibility": NonInstrumentedEntry(transport="replay", count=18),
    }
    kwargs["model_calls"] = ModelCallsSummary(expected=32, served=32)
    with pytest.raises(ValidationError, match="non_instrumented_model_calls keys must be exactly"):
        RunModeDocument(**kwargs)


def test_run_mode_document_rejects_sum_mismatch_against_model_calls_expected():
    kwargs = _valid_kwargs()
    kwargs["model_calls"] = ModelCallsSummary(expected=999, served=999)
    with pytest.raises(ValidationError, match=r"must equal model_calls\.expected"):
        RunModeDocument(**kwargs)


def test_run_mode_document_rejects_non_hex_replay_cassette_sha256():
    kwargs = _valid_kwargs(replay_cassette_sha256="not-hex")
    with pytest.raises(ValidationError):
        RunModeDocument(**kwargs)


def test_run_mode_document_is_frozen_and_forbids_extra_fields():
    doc = RunModeDocument(**_valid_kwargs())
    with pytest.raises(ValidationError):
        doc.spend_usd = 1.0
    with pytest.raises(ValidationError):
        RunModeDocument(**_valid_kwargs(), some_extra_top_level_field="nope")


# ---------------------------------------------------------------------------
# build_run_mode_document()
# ---------------------------------------------------------------------------

def _reporting_context():
    return {"llm_transport": "replay", "search_source": "fixture", "bertopic": "not_run", "dedup_mode": "exact"}


def test_build_run_mode_document_reproduces_the_frozen_reference_run_schema_example():
    cassette = _reference_cassette()
    served = _served_calls_matching_expected(cassette)
    doc = build_run_mode_document(
        cassette=cassette,
        served_calls=served,
        replay_cassette_sha256=_HEX64_A,
        fixture_corpus_digest=_HEX64_B,
        reporting_context=_reporting_context(),
        port=54321,
    )
    assert doc.mode_profile == "track2_offline"
    assert doc.pipeline_network == "loopback_only"
    assert doc.llm_transport == "replay"
    assert doc.search_source == "fixture"
    assert doc.bertopic == "not_run"
    assert doc.dedup_mode == "exact"
    assert doc.full_text_acquisition == "local_prepositioned"
    assert doc.spend_usd == 0.0
    assert doc.replay_cassette_sha256 == _HEX64_A
    assert doc.fixture_corpus_digest == _HEX64_B
    assert doc.model_calls.expected == 50
    assert doc.model_calls.served == 50
    # §11.3's own frozen per-surface counts, translated through the schema-key mapping.
    assert doc.per_stage_backend["search_strategy_query_expansion"].calls == 1
    assert doc.per_stage_backend["search_strategy_db_translation"].calls == 1
    assert doc.per_stage_backend["title_abstract_screening"].calls == 5
    assert doc.per_stage_backend["screening_qc_second_rater"].calls == 6
    assert doc.per_stage_backend["synthesis_claim_map"].calls == 10
    assert doc.per_stage_backend["synthesis_claim_refine"].calls == 4
    assert doc.non_instrumented_model_calls["full_text_eligibility"].count == 18
    assert doc.non_instrumented_model_calls["extraction"].count == 5
    for entry in doc.per_stage_backend.values():
        assert entry.transport == "replay"
        assert entry.access_info == "http://127.0.0.1:54321/v1"
    assert doc.non_model_tool_events["dedup_exact"].model_calls == 0


def test_build_run_mode_document_rejects_under_served_input_fail_closed():
    """`build_run_mode_document()` has no independent notion of "the run
    completed" -- it trusts its `served_calls` argument. If that argument
    reflects a genuinely under-served run (the caller skipped
    `ReplayServer.assert_complete()`, or called this before it passed), the
    per-surface check (`_check_served_matches_cassette_per_surface`) fails
    closed before a `RunModeDocument` is ever constructed."""
    cassette = _reference_cassette()
    served = _served_calls_matching_expected(cassette)
    under_served_entry_id = next(e.entry_id for e in cassette.entries if e.surface == "synthesis_map")
    served[under_served_entry_id] = 3  # cassette says 10 expected; only 3 actually served
    with pytest.raises(RunModeBuildError, match="synthesis_map"):
        build_run_mode_document(
            cassette=cassette, served_calls=served, replay_cassette_sha256=_HEX64_A,
            fixture_corpus_digest=_HEX64_B, reporting_context=_reporting_context(), port=1,
        )


def test_build_run_mode_document_rejects_compensating_multi_surface_miscount():
    """Two surfaces miscounted in OPPOSITE directions by the same amount
    would sum to the right grand total (`model_calls.expected` unaffected)
    -- `RunModeDocument`'s own sum-only validator alone cannot see this.
    The per-surface check in `build_run_mode_document()` must catch it
    anyway, since it compares each surface individually, not just the
    grand total (round-14 self-review finding 1)."""
    cassette = _reference_cassette()
    served = _served_calls_matching_expected(cassette)
    synthesis_map_id = next(e.entry_id for e in cassette.entries if e.surface == "synthesis_map")
    db_translation_id = next(e.entry_id for e in cassette.entries if e.surface == "db_translation")
    served[synthesis_map_id] = 7   # cassette expects 10 -- under by 3
    served[db_translation_id] = 4  # cassette expects 1 -- over by 3 (compensates the grand total)
    assert sum(served.values()) == sum(e.expected_calls for e in cassette.entries)  # grand total unaffected
    with pytest.raises(RunModeBuildError, match="synthesis_map"):
        build_run_mode_document(
            cassette=cassette, served_calls=served, replay_cassette_sha256=_HEX64_A,
            fixture_corpus_digest=_HEX64_B, reporting_context=_reporting_context(), port=1,
        )


# ---------------------------------------------------------------------------
# render_run_mode_statement_md() -- pure function of the document, deterministic
# ---------------------------------------------------------------------------

def test_render_run_mode_statement_md_is_pure_and_deterministic():
    doc = RunModeDocument(**_valid_kwargs())
    first = render_run_mode_statement_md(doc)
    second = render_run_mode_statement_md(doc)
    assert first == second
    assert first.startswith("# Track 2 reference run")
    assert "`track2_offline`" in first
    assert "`replay`" in first
    assert "Expected: 50. Served: 50." in first
    assert "| synthesis_claim_map | replay |" in first
    assert "| dedup_exact | non_llm_tool | 0 |" in first
    assert "| extraction | replay | 5 |" in first


def test_render_run_mode_statement_md_table_rows_are_key_sorted():
    doc = RunModeDocument(**_valid_kwargs())
    md = render_run_mode_statement_md(doc)
    stage_lines = [line for line in md.splitlines() if line.startswith("| search_strategy") or line.startswith("| synthesis_claim") or line.startswith("| screening_qc") or line.startswith("| title_abstract")]
    assert stage_lines == sorted(stage_lines)


# ---------------------------------------------------------------------------
# SURFACE_TO_*_KEY mapping sanity
# ---------------------------------------------------------------------------

def test_surface_mappings_partition_all_eight_replay_surfaces_with_no_overlap():
    per_stage_surfaces = set(SURFACE_TO_PER_STAGE_KEY)
    non_instr_surfaces = set(SURFACE_TO_NON_INSTRUMENTED_KEY)
    assert per_stage_surfaces & non_instr_surfaces == set()
    assert per_stage_surfaces | non_instr_surfaces == set(REFERENCE_SURFACE_CALL_COUNTS)


# ---------------------------------------------------------------------------
# mode-aware schema: replay_cassette_sha256 / fixture_corpus_digest
# ---------------------------------------------------------------------------

def _valid_live_kwargs(**overrides):
    kwargs = dict(
        mode_profile="track3_live",
        pipeline_network="internet_via_cost_cap_proxy",
        llm_transport="live",
        replay_cassette_sha256=None,
        search_source="live",
        fixture_corpus_digest=None,
        full_text_acquisition="live_download",
        bertopic="enabled",
        dedup_mode="exact",
        spend_usd=1.23,
        model_calls=ModelCallsSummary(expected=27, served=27),
        per_stage_backend={
            "search_strategy_query_expansion": PerStageBackendEntry(transport="live", access_info="http://127.0.0.1:0/v1", calls=1),
            "search_strategy_db_translation": PerStageBackendEntry(transport="live", access_info="http://127.0.0.1:0/v1", calls=1),
            "title_abstract_screening": PerStageBackendEntry(transport="live", access_info="http://127.0.0.1:0/v1", calls=5),
            "screening_qc_second_rater": PerStageBackendEntry(transport="live", access_info="http://127.0.0.1:0/v1", calls=6),
            "synthesis_claim_map": PerStageBackendEntry(transport="live", access_info="http://127.0.0.1:0/v1", calls=10),
            "synthesis_claim_refine": PerStageBackendEntry(transport="live", access_info="http://127.0.0.1:0/v1", calls=4),
        },
        non_model_tool_events={
            "dedup_exact": NonModelToolEvent(tool_kind="non_llm_tool", model_calls=0),
        },
        non_instrumented_model_calls={
            "full_text_eligibility": NonInstrumentedEntry(transport="live", count=0),
            "extraction": NonInstrumentedEntry(transport="live", count=0),
        },
    )
    kwargs.update(overrides)
    return kwargs


def test_run_mode_document_track3_live_accepts_none_cassette_and_digest():
    doc = RunModeDocument(**_valid_live_kwargs())
    assert doc.replay_cassette_sha256 is None
    assert doc.fixture_corpus_digest is None
    assert doc.mode_profile == "track3_live"


def test_run_mode_document_track3_live_rejects_non_none_cassette_sha256():
    with pytest.raises(ValidationError, match="track3_live must not carry"):
        RunModeDocument(**_valid_live_kwargs(replay_cassette_sha256=_HEX64_A))


def test_run_mode_document_track3_live_rejects_non_none_fixture_corpus_digest():
    with pytest.raises(ValidationError, match="track3_live must not carry"):
        RunModeDocument(**_valid_live_kwargs(fixture_corpus_digest=_HEX64_B))


def test_run_mode_document_track2_offline_rejects_none_cassette_sha256():
    with pytest.raises(ValidationError, match="track2_offline requires"):
        RunModeDocument(**_valid_kwargs(replay_cassette_sha256=None))


def test_run_mode_document_track2_offline_rejects_none_fixture_corpus_digest():
    with pytest.raises(ValidationError, match="track2_offline requires"):
        RunModeDocument(**_valid_kwargs(fixture_corpus_digest=None))


def test_run_mode_document_rejects_unknown_mode_profile():
    with pytest.raises(ValidationError, match="unknown mode_profile"):
        RunModeDocument(**_valid_live_kwargs(mode_profile="some_other_profile"))


# ---------------------------------------------------------------------------
# build_run_mode_document_live()
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, stage):
        self.stage = stage


def _live_events():
    return [
        _FakeEvent("search_strategy_query_expansion"),
        _FakeEvent("search_strategy_db_translation"),
        _FakeEvent("title_abstract_screening"),
        _FakeEvent("title_abstract_screening"),
        _FakeEvent("screening_qc_second_rater"),
        _FakeEvent("synthesis_claim_map"),
        _FakeEvent("synthesis_claim_map"),
        _FakeEvent("synthesis_claim_refine"),
        _FakeEvent("dedup_exact"),  # non-LLM tool event -- must not be counted anywhere
    ]


def test_build_run_mode_document_live_tallies_events_by_stage():
    doc = build_run_mode_document_live(
        ai_use_events=_live_events(),
        extraction_call_count=3,
        full_text_eligibility_call_count=2,
        spend_usd=4.56,
        full_text_acquisition="live_download",
        bertopic="enabled",
        port=54321,
    )
    assert doc.mode_profile == "track3_live"
    assert doc.pipeline_network == "internet_via_cost_cap_proxy"
    assert doc.llm_transport == "live"
    assert doc.search_source == "live"
    assert doc.dedup_mode == "exact"
    assert doc.replay_cassette_sha256 is None
    assert doc.fixture_corpus_digest is None
    assert doc.spend_usd == 4.56
    assert doc.per_stage_backend["search_strategy_query_expansion"].calls == 1
    assert doc.per_stage_backend["search_strategy_db_translation"].calls == 1
    assert doc.per_stage_backend["title_abstract_screening"].calls == 2
    assert doc.per_stage_backend["screening_qc_second_rater"].calls == 1
    assert doc.per_stage_backend["synthesis_claim_map"].calls == 2
    assert doc.per_stage_backend["synthesis_claim_refine"].calls == 1
    for entry in doc.per_stage_backend.values():
        assert entry.transport == "live"
        assert entry.access_info == "http://127.0.0.1:54321/v1"
    assert doc.non_instrumented_model_calls["extraction"].count == 3
    assert doc.non_instrumented_model_calls["full_text_eligibility"].count == 2
    assert doc.non_model_tool_events["dedup_exact"].model_calls == 0
    total = 1 + 1 + 2 + 1 + 2 + 1 + 3 + 2
    assert doc.model_calls.expected == total
    assert doc.model_calls.served == total


def test_build_run_mode_document_live_fails_closed_on_unaccounted_stage():
    """A stage with no `run_mode.json` schema slot (e.g.
    `synthesis_imrad_narrative`, real when `cfg.synthesis.
    generate_legacy_imrad_narrative` is True) must never be silently
    dropped from the tally -- that would understate both `model_calls`
    and spend accounting with no error raised anywhere."""
    events = _live_events() + [_FakeEvent("synthesis_imrad_narrative")]
    with pytest.raises(RunModeBuildError, match="synthesis_imrad_narrative"):
        build_run_mode_document_live(
            ai_use_events=events,
            extraction_call_count=0,
            full_text_eligibility_call_count=0,
            spend_usd=0.0,
            full_text_acquisition="live_download",
            bertopic="enabled",
            port=1,
        )


def test_build_run_mode_document_live_handles_zero_events():
    doc = build_run_mode_document_live(
        ai_use_events=[],
        extraction_call_count=0,
        full_text_eligibility_call_count=0,
        spend_usd=0.0,
        full_text_acquisition="live_download",
        bertopic="disabled",
        port=1,
    )
    assert doc.model_calls.expected == 0
    assert doc.model_calls.served == 0
    for entry in doc.per_stage_backend.values():
        assert entry.calls == 0
    for entry in doc.non_instrumented_model_calls.values():
        assert entry.count == 0


# ---------------------------------------------------------------------------
# render_run_mode_statement_md() -- track3_live None-guard
# ---------------------------------------------------------------------------

def test_render_run_mode_statement_md_shows_not_applicable_for_live_run():
    doc = RunModeDocument(**_valid_live_kwargs())
    md = render_run_mode_statement_md(doc)
    assert "not applicable (live run)" in md
    assert "`None`" not in md
    assert "`track3_live`" in md
    assert md.startswith("# Track 3 live run")
    assert "# Track 2 reference run" not in md


# ---------------------------------------------------------------------------
# resolution_suggestion_calls / resolution_suggestion_interrupted (plan §3.4)
# ---------------------------------------------------------------------------

def test_run_mode_document_track2_offline_defaults_resolution_suggestion_fields_empty():
    doc = RunModeDocument(**_valid_kwargs())
    assert doc.resolution_suggestion_calls == {}
    assert doc.resolution_suggestion_interrupted == 0


def test_run_mode_document_track2_offline_rejects_nonempty_resolution_suggestion_calls():
    with pytest.raises(ValidationError, match="must not carry resolution_suggestion_calls"):
        RunModeDocument(**_valid_kwargs(
            resolution_suggestion_calls={
                EXTRACTION_RESOLUTION_SUGGESTION_STAGE: PerStageBackendEntry(
                    transport="live", access_info="http://127.0.0.1:0/v1", calls=1,
                ),
            },
        ))


def test_run_mode_document_track2_offline_rejects_nonzero_resolution_suggestion_interrupted():
    with pytest.raises(ValidationError, match="must not carry a nonzero resolution_suggestion_interrupted"):
        RunModeDocument(**_valid_kwargs(resolution_suggestion_interrupted=1))


def test_run_mode_document_rejects_negative_resolution_suggestion_interrupted():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RunModeDocument(**_valid_live_kwargs(resolution_suggestion_interrupted=-3))


def test_run_mode_document_track3_live_accepts_empty_resolution_suggestion_calls():
    doc = RunModeDocument(**_valid_live_kwargs())
    assert doc.resolution_suggestion_calls == {}


def test_run_mode_document_track3_live_accepts_resolution_suggestion_calls_and_folds_into_total():
    doc = RunModeDocument(**_valid_live_kwargs(
        model_calls=ModelCallsSummary(expected=28, served=28),
        resolution_suggestion_calls={
            EXTRACTION_RESOLUTION_SUGGESTION_STAGE: PerStageBackendEntry(
                transport="live", access_info="http://127.0.0.1:0/v1", calls=1,
            ),
        },
    ))
    assert doc.resolution_suggestion_calls[EXTRACTION_RESOLUTION_SUGGESTION_STAGE].calls == 1


def test_run_mode_document_track3_live_rejects_unknown_resolution_suggestion_key():
    with pytest.raises(ValidationError, match="resolution_suggestion_calls keys must be a subset"):
        RunModeDocument(**_valid_live_kwargs(
            model_calls=ModelCallsSummary(expected=28, served=28),
            resolution_suggestion_calls={
                "some_other_stage": PerStageBackendEntry(
                    transport="live", access_info="http://127.0.0.1:0/v1", calls=1,
                ),
            },
        ))


def test_run_mode_document_rejects_sum_mismatch_when_resolution_suggestion_calls_uncounted():
    with pytest.raises(ValidationError, match="must equal model_calls.expected"):
        RunModeDocument(**_valid_live_kwargs(
            resolution_suggestion_calls={
                EXTRACTION_RESOLUTION_SUGGESTION_STAGE: PerStageBackendEntry(
                    transport="live", access_info="http://127.0.0.1:0/v1", calls=1,
                ),
            },
        ))


def test_build_run_mode_document_live_populates_resolution_suggestion_calls():
    events = _live_events() + [_FakeEvent(EXTRACTION_RESOLUTION_SUGGESTION_STAGE)]
    doc = build_run_mode_document_live(
        ai_use_events=events,
        extraction_call_count=0,
        full_text_eligibility_call_count=0,
        spend_usd=0.0,
        full_text_acquisition="live_download",
        bertopic="enabled",
        port=1,
    )
    assert doc.resolution_suggestion_calls[EXTRACTION_RESOLUTION_SUGGESTION_STAGE].calls == 1
    assert doc.resolution_suggestion_calls[EXTRACTION_RESOLUTION_SUGGESTION_STAGE].transport == "live"
    total = 1 + 1 + 2 + 1 + 2 + 1 + 1
    assert doc.model_calls.expected == total
    assert doc.model_calls.served == total


def test_build_run_mode_document_live_defaults_resolution_suggestion_interrupted_zero():
    doc = build_run_mode_document_live(
        ai_use_events=[],
        extraction_call_count=0,
        full_text_eligibility_call_count=0,
        spend_usd=0.0,
        full_text_acquisition="live_download",
        bertopic="disabled",
        port=1,
    )
    assert doc.resolution_suggestion_interrupted == 0
    assert doc.resolution_suggestion_calls[EXTRACTION_RESOLUTION_SUGGESTION_STAGE].calls == 0


def test_build_run_mode_document_live_passes_through_resolution_suggestion_interrupted():
    doc = build_run_mode_document_live(
        ai_use_events=[],
        extraction_call_count=0,
        full_text_eligibility_call_count=0,
        spend_usd=0.0,
        full_text_acquisition="live_download",
        bertopic="disabled",
        port=1,
        resolution_suggestion_interrupted=3,
    )
    assert doc.resolution_suggestion_interrupted == 3


def test_render_run_mode_statement_md_omits_resolution_suggestion_section_for_track2_offline():
    doc = RunModeDocument(**_valid_kwargs())
    md = render_run_mode_statement_md(doc)
    assert "## Resolution suggestion calls" not in md


def test_render_run_mode_statement_md_includes_resolution_suggestion_section_for_track3_live_even_when_zero():
    doc = RunModeDocument(**_valid_live_kwargs())
    md = render_run_mode_statement_md(doc)
    assert "## Resolution suggestion calls" in md


def test_render_run_mode_statement_md_shows_interrupted_line_when_nonzero():
    events = _live_events()
    doc = build_run_mode_document_live(
        ai_use_events=events,
        extraction_call_count=0,
        full_text_eligibility_call_count=0,
        spend_usd=0.0,
        full_text_acquisition="live_download",
        bertopic="enabled",
        port=1,
        resolution_suggestion_interrupted=2,
    )
    md = render_run_mode_statement_md(doc)
    assert "2 interrupted/unknown-outcome calls" in md


def test_render_run_mode_statement_md_omits_interrupted_line_when_zero():
    doc = RunModeDocument(**_valid_live_kwargs())
    md = render_run_mode_statement_md(doc)
    assert "interrupted/unknown-outcome" not in md
