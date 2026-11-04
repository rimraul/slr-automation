"""Tests for `slr_pipeline.reference_acceptance` -- Track 2 pipeline
readiness plan (`track2_pipeline_readiness_plan_2026-08-30.md`) step 16
[C15]: the acceptance harness (HANDOFF §1's 8 checks, §11.1).

Structure:
  1. `prisma_flow()` unit tests -- happy path, the terminal-queue
     precondition, and one failure per each of the six conservation
     groups the ported check enforces. No bundle machinery needed.
  2. A genuine "all 8 checks pass" happy-path fixture
     (`_build_happy_path_bundle`), reusing `tests/nodes/
     test_reporting_node.py::_run_full_pipeline_to_reporting` almost
     unchanged (only its module-global `ReviewConfig` name is
     monkeypatched to inject the two Track 2 reference-run markers that
     function's own shared default cfg does not carry: `dedup.mode ==
     "exact"` and a real `reference_reporting_context`), plus a real,
     schema-valid `run_mode.json` whose per-stage call counts are counted
     from the run's own durable `AIUseEvent`s rather than hand-picked.
  3. Fail-closed coverage against `tests/test_reference_bundle.py`'s own
     `_finalize_from_harness` fixture (a Track-2-context-INCOMPLETE
     bundle -- no `reference_reporting_context`, a non-schema-valid
     `run_mode.json` stub) -- proof a malformed/incomplete bundle produces
     `status="fail"` `AcceptanceCheck`s, never a crash.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import slr_pipeline.reference_acceptance as ra
import slr_pipeline.reference_bundle as rb
from slr_pipeline.ai_traceability import AIUseEventStore, build_ai_use_event
from slr_pipeline.config import DedupConfig, ReviewConfig
from slr_pipeline.export_manifest import ReportingStageEntry
from slr_pipeline.extraction.hashing import file_sha256
from slr_pipeline.extraction.resolution_schemas import EXTRACTION_RESOLUTION_SUGGESTION_STAGE
from slr_pipeline.reference_run_mode import (
    ModelCallsSummary,
    NonInstrumentedEntry,
    NonModelToolEvent,
    PerStageBackendEntry,
    RunModeDocument,
    SURFACE_TO_PER_STAGE_KEY,
    build_run_mode_document_live,
    render_run_mode_statement_md,
)
from slr_pipeline.reference_spec import reference_reporting_context
from slr_pipeline.state import PRISMACounts

REPO_ROOT = Path(__file__).resolve().parent.parent
# `FixtureInputPaths` used to have a `.default()` fallback computed from
# `slr_pipeline.reference_run`'s own `Path(__file__)`-derived constants --
# removed (Track-2 packaging-boundary fix, CI run 34155102694) since that
# assumes `slr_pipeline` sits inside a git checkout. This file's own
# `REPO_ROOT` (a genuine, always-valid checkout-relative constant, since
# pytest always reads `tests/` from the real checkout) supplies the same
# paths explicitly instead.
_FIXTURES = rb.FixtureInputPaths(
    spec_dir=REPO_ROOT / "tests" / "reference_run" / "fixtures" / "spec",
    corpus_dir=REPO_ROOT / "tests" / "reference_run" / "fixtures" / "corpus",
    protocol_path=REPO_ROOT / "track2" / "readiness_protocol.md",
    prisma_scr_matrix_path=REPO_ROOT / "tests" / "reference_run" / "fixtures" / "prisma_scr_matrix.json",
)

# ===========================================================================
# 1. `prisma_flow()` unit tests
# ===========================================================================


def _baseline_prisma(**overrides) -> PRISMACounts:
    base = dict(
        identified={"db": 10}, after_dedup=8, screened_title_abstract=8, excluded_title_abstract=3,
        fulltext_retrieval={"parsed": 5}, assessed_full_text=5, excluded_full_text=2,
        excluded_full_text_reasons={"reason_a": 2}, included=3,
        quality_assessed=None, excluded_quality=None,
    )
    base.update(overrides)
    return PRISMACounts(**base)


def test_prisma_flow_happy_path():
    result = ra.prisma_flow(_baseline_prisma(), full_text_review_queue=[])
    assert result.total_identified == 10
    assert result.duplicates_removed == 2
    assert result.included_after_ta == 5
    assert result.fulltext_sought == 5
    assert result.acquisition_failures == 0
    assert result.included == 3


def test_prisma_flow_requires_empty_full_text_review_queue():
    with pytest.raises(ra.PrismaInconsistencyError, match="not a terminal state"):
        ra.prisma_flow(_baseline_prisma(), full_text_review_queue=[{"id": "r1"}])


def test_prisma_flow_group1_after_dedup_exceeds_total_identified():
    prisma = _baseline_prisma(after_dedup=11, screened_title_abstract=11)
    with pytest.raises(ra.PrismaInconsistencyError, match="group 1"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group1_screened_not_equal_after_dedup():
    prisma = _baseline_prisma(screened_title_abstract=7)
    with pytest.raises(ra.PrismaInconsistencyError, match="screened_title_abstract != after_dedup"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group2_excluded_title_abstract_exceeds_screened():
    prisma = _baseline_prisma(excluded_title_abstract=9)
    with pytest.raises(ra.PrismaInconsistencyError, match="group 2"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group3_retrieval_sum_mismatch():
    prisma = _baseline_prisma(fulltext_retrieval={"parsed": 3})
    with pytest.raises(ra.PrismaInconsistencyError, match="group 3"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group4_assessed_not_equal_parsed():
    prisma = _baseline_prisma(assessed_full_text=4)
    with pytest.raises(ra.PrismaInconsistencyError, match="assessed_full_text != fulltext_retrieval"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group4_assessed_exceeds_included_after_ta():
    # sum(fulltext_retrieval.values()) must still equal included_after_ta (group 3) --
    # achieved here with a negative sibling key, isolating group 4's own
    # `acquisition_failures < 0` branch (assessed_full_text > included_after_ta)
    # without tripping group 3 first.
    prisma = _baseline_prisma(fulltext_retrieval={"parsed": 7, "adjustment": -2}, assessed_full_text=7)
    with pytest.raises(ra.PrismaInconsistencyError, match="exceeds included_after_ta"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group5_excluded_reasons_sum_mismatch():
    prisma = _baseline_prisma(excluded_full_text_reasons={"reason_a": 1})
    with pytest.raises(ra.PrismaInconsistencyError, match="group 5"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group5_included_not_equal_assessed_minus_excluded():
    prisma = _baseline_prisma(included=1)
    with pytest.raises(ra.PrismaInconsistencyError, match="included != assessed_full_text"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


def test_prisma_flow_group6_quality_appraisal_present_in_scoping():
    prisma = _baseline_prisma(quality_assessed=5)
    with pytest.raises(ra.PrismaInconsistencyError, match="group 6"):
        ra.prisma_flow(prisma, full_text_review_queue=[])


# ===========================================================================
# 2 & 3. Bundle-level fixtures -- shared with `tests/test_reference_bundle.py`
# (same `_load_reporting_node_harness()`/`_source_info()` convention, kept
# local rather than cross-imported: neither `tests/` nor `tests/nodes/` is a
# package).
# ===========================================================================


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex(seed: str, n: int = 64) -> str:
    return _raw_sha256(seed.encode("utf-8"))[:n]


def _source_info(**overrides) -> rb.PayloadSourceInfo:
    base = dict(
        git_commit=_hex("git", 40), execution_subtree_digest=_hex("subtree"),
        execution_subtree_provenance="runtime_verified",
        execution_subtree_clean=True, python_version="3.12.12",
        runner_image_os="ubuntu-22.04", runner_image_version="20260101.1",
        base_requirements_lock_sha256=_hex("base_req"), build_requirements_lock_sha256=_hex("build_req"),
        ft_screening_wheel_lock_sha256=_hex("ft_wheel_lock"), slr_pipeline_wheel_sha256=_hex("slr_wheel"),
        workflow_sha256=_hex("workflow"), provision_script_sha256=_hex("provision"),
        build_ft_wheel_script_sha256=_hex("build_ft_wheel"), author_cassette_script_sha256=_hex("author_cassette"),
        ft_screening=rb.FTScreeningSourceInfo(
            filename="ft_screening-1.0.0-py3-none-any.whl", sha256=_hex("ft_wheel"),
            licence_commit=_hex("licence", 40), code_parent="9604dc7", release_tag="track2-assets-v1"),
        docling_assets_manifest_digest=_hex("docling_assets"), docling_archive_sha256=_hex("docling_archive"),
        docling_model_manifest_sha256=_hex("docling_model"), fixture_corpus_digest=_hex("fixture_corpus"),
        cassette_sha256=_hex("cassette"), reference_spec_sha256=_hex("reference_spec"),
        rfc8785_version="0.1.4", mode_profile="track2_offline",
    )
    base.update(overrides)
    return rb.PayloadSourceInfo(**base)


def _load_reporting_node_harness():
    path = Path(__file__).parent / "nodes" / "test_reporting_node.py"
    spec = importlib.util.spec_from_file_location("_reporting_node_harness_for_acceptance_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_reporting_node_harness()


def _review_config_with_track2_overrides(*args, **kwargs):
    kwargs.setdefault("dedup", DedupConfig(mode="exact"))
    kwargs.setdefault("reference_reporting_context", reference_reporting_context().model_dump(mode="json"))
    return ReviewConfig(*args, **kwargs)


def _review_config_with_track3_live_overrides(*args, **kwargs):
    """`track3_live`'s own marker: `dedup.mode == "exact"` (mode-agnostic
    per Sec 2.4), but `reference_reporting_context` deliberately left
    unset -- a live run never sets it (`nodes/reporting.py`'s
    `ARTIFACT_RENDERERS` gate on `reference_context is not None`), which is
    exactly what makes both PRISMA-ScR renderers resolve to `status ==
    "not_applicable"` for this fixture, matching real production
    behavior rather than a hand-picked stub."""
    kwargs.setdefault("dedup", DedupConfig(mode="exact"))
    return ReviewConfig(*args, **kwargs)


def _build_happy_path_bundle(
    harness_mod, monkeypatch, tmp_path, *, pre_finalize_hook=None, review_type: str = "scoping",
) -> rb.Bundle:
    """Genuine "all 8 HANDOFF checks pass" fixture -- see module docstring
    point 2. `pre_finalize_hook(cfg, review_run_id)`, if given, runs after
    the real pipeline has finished writing its raw-layer files (events,
    ledger records, ...) under `cfg.ensure_output_dir()` but BEFORE
    `finalize_bundle()` copies them into the payload and computes
    `payload_manifest.json`'s own per-artifact sha256 -- i.e. any edit it
    makes becomes part of the bundle's own self-consistent "current file
    bytes", the only way to defeat a check's INTERNAL (semantic-hash)
    self-verification without also defeating `open_bundle()`'s FILE-hash
    layer (which a post-finalize edit would trip first, before this
    module's own checks ever ran)."""
    monkeypatch.setattr(harness_mod, "ReviewConfig", _review_config_with_track2_overrides)
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type=review_type)
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]
    if pre_finalize_hook is not None:
        pre_finalize_hook(cfg, review_run_id)

    # Real, per-stage AIUseEvent counts -- never hand-picked -- so
    # `run_mode.json`'s own `per_stage_backend[*].calls` cannot silently
    # drift from what this run actually produced (check 8's own concern).
    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    stage_counts = Counter(event.stage for event in event_store.load_all())

    per_stage_backend = {
        schema_key: PerStageBackendEntry(
            transport="replay", access_info="http://127.0.0.1:0/v1", calls=stage_counts.get(schema_key, 0),
        )
        for schema_key in SURFACE_TO_PER_STAGE_KEY.values()
    }
    non_instrumented_model_calls = {
        "extraction": NonInstrumentedEntry(transport="replay", count=1),
        "full_text_eligibility": NonInstrumentedEntry(transport="replay", count=1),
    }
    non_model_tool_events = {"dedup_exact": NonModelToolEvent(tool_kind="non_llm_tool", model_calls=0)}
    total_calls = (
        sum(entry.calls for entry in per_stage_backend.values())
        + sum(entry.count for entry in non_instrumented_model_calls.values())
    )
    run_mode = RunModeDocument(
        mode_profile="track2_offline", pipeline_network="loopback_only", llm_transport="replay",
        replay_cassette_sha256=_hex("no-cassette-in-this-fixture"), search_source="fixture",
        fixture_corpus_digest=_hex("fixture-corpus-in-this-fixture"), full_text_acquisition="local_prepositioned",
        bertopic="not_run", dedup_mode="exact", spend_usd=0.0,
        model_calls=ModelCallsSummary(expected=total_calls, served=total_calls),
        per_stage_backend=per_stage_backend, non_model_tool_events=non_model_tool_events,
        non_instrumented_model_calls=non_instrumented_model_calls,
    )

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "violations.jsonl").write_text("", encoding="utf-8")
    (workspace / "run_mode.json").write_text(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # `write_bytes`, not `write_text` -- check 8 byte-compares this file
    # against a fresh `render_run_mode_statement_md()` call; on Windows,
    # `Path.write_text()`'s default universal-newline translation would
    # silently turn every `\n` into `\r\n` on disk, diverging from that
    # pure function's own `\n`-only output (the real driver,
    # `reference_run_child.py`, only ever runs on the Linux CI runner,
    # where this translation is a no-op -- this is a Windows-dev-fixture
    # concern, not a production one).
    (workspace / "run_mode_statement.md").write_bytes(render_run_mode_statement_md(run_mode).encode("utf-8"))

    source = _source_info(mode_profile="track2_offline")
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track2_offline",
        fixtures=_FIXTURES,
    )
    return rb.open_bundle(bundle_dir / "payload")


def _build_track3_live_bundle(harness_mod, monkeypatch, tmp_path) -> rb.Bundle:
    """A genuine `track3_live`-mode bundle for exercising check_1/check_5's
    new mode-aware branches: real pipeline run, `dedup.mode == "exact"`,
    `reference_reporting_context` genuinely unset (never a Track-2-only
    stub value) so `nodes/reporting.py`'s own gate really does resolve
    both PRISMA-ScR renderers to `status == "not_applicable"`. `run_mode.
    json` is a schema-agnostic stub here -- Unit B (the `track3_live`
    `RunModeDocument`/builder) is separate work; this fixture is not used
    for check_8 assertions."""
    monkeypatch.setattr(harness_mod, "ReviewConfig", _review_config_with_track3_live_overrides)
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="scoping")
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "cost_cap_call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "cost_cap_ledger.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode.json").write_text(
        json.dumps({"mode_profile": "track3_live", "llm_transport": "live", "search_source": "live"}),
        encoding="utf-8")
    (workspace / "run_mode_statement.md").write_text(
        "# Run mode\n\nLive mode (test harness stub).\n", encoding="utf-8")

    source = _source_info(mode_profile="track3_live", execution_subtree_provenance="retrospective_reconstruction",
                           execution_subtree_clean=None, fixture_corpus_digest=None, cassette_sha256=None)
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track3_live",
    )
    return rb.open_bundle(bundle_dir / "payload")


def _build_track3_live_bundle_with_real_run_mode(harness_mod, monkeypatch, tmp_path) -> rb.Bundle:
    """A `track3_live` bundle whose `run_mode.json` is a genuine,
    schema-valid `RunModeDocument` built by `build_run_mode_document_live()`
    from this run's own real `AIUseEventStore` -- unlike
    `_build_track3_live_bundle` (whose own docstring disclaims check_8
    use), this fixture exists specifically to exercise check_8's
    `track3_live` path end to end. Mirrors `_build_happy_path_bundle`'s own
    real-per-stage-counts discipline: `per_stage_backend[*].calls` is never
    hand-picked, it comes from this run's real events, so check_8's
    cross-check against `snapshot.events` cannot pass by coincidence."""
    monkeypatch.setattr(harness_mod, "ReviewConfig", _review_config_with_track3_live_overrides)
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="scoping")
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    run_mode = build_run_mode_document_live(
        ai_use_events=event_store.load_all(),
        extraction_call_count=1,
        full_text_eligibility_call_count=1,
        spend_usd=0.0,
        full_text_acquisition="local_prepositioned",
        bertopic="not_run",
        port=0,
    )

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "cost_cap_call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "cost_cap_ledger.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode.json").write_bytes(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8") + b"\n")
    # `write_bytes`, not `write_text` -- check 8 byte-compares this file
    # against a fresh `render_run_mode_statement_md()` call (see the
    # matching comment in `_build_happy_path_bundle`).
    (workspace / "run_mode_statement.md").write_bytes(render_run_mode_statement_md(run_mode).encode("utf-8"))

    source = _source_info(mode_profile="track3_live", execution_subtree_provenance="retrospective_reconstruction",
                           execution_subtree_clean=None, fixture_corpus_digest=None, cassette_sha256=None)
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track3_live",
    )
    return rb.open_bundle(bundle_dir / "payload")


def _publish_resolution_suggestion_event(event_store: AIUseEventStore, *, work_item_id: str) -> None:
    """Publishes one minimal, real `extraction_resolution_suggestion`
    `AIUseEvent` into `event_store` -- the same real, self-hash-verified
    publish path every other stage uses, so a test injecting one is
    exercising check_8's genuine cross-check against bundled events, not a
    hand-typed stand-in."""
    event = build_ai_use_event(
        review_run_id=event_store.review_run_id,
        stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE,
        work_item_id=work_item_id,
        attempt=0,
        tool_kind="llm",
        model="test-model",
        provider="test-provider",
        component_versions={},
        task_purpose="test",
        model_config_id="test-config",
        input_artifact_refs=[],
        output_artifact_refs=[],
        call_status="success",
        prompt_template_version="test-v1",
        prompt_template_text="test prompt",
        rendered_prompt_hash=hashlib.sha256(b"test prompt").hexdigest(),
    )
    event_store.publish(event)


def _build_track3_live_bundle_with_resolution_suggestion_event(
    harness_mod, monkeypatch, tmp_path, *, include_in_run_mode: bool,
) -> rb.Bundle:
    """Like `_build_track3_live_bundle_with_real_run_mode`, but also
    publishes one real `extraction_resolution_suggestion` `AIUseEvent`.
    `finalize_bundle()` always bundles whatever is in the store at
    finalize time (it opens its own fresh `AIUseEventStore`), regardless
    of what `run_mode.json` itself declares -- so publishing the event
    AFTER `build_run_mode_document_live()` has already run produces a
    `RunModeDocument` that under-counts a real, bundled event
    (`include_in_run_mode=False`), the exact mismatch check_8's new
    `_require` block must catch."""
    monkeypatch.setattr(harness_mod, "ReviewConfig", _review_config_with_track3_live_overrides)
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="scoping")
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)

    if include_in_run_mode:
        _publish_resolution_suggestion_event(event_store, work_item_id="extrun-1:rec-1:field-1")
        run_mode = build_run_mode_document_live(
            ai_use_events=event_store.load_all(),
            extraction_call_count=1,
            full_text_eligibility_call_count=1,
            spend_usd=0.0,
            full_text_acquisition="local_prepositioned",
            bertopic="not_run",
            port=0,
        )
    else:
        run_mode = build_run_mode_document_live(
            ai_use_events=event_store.load_all(),
            extraction_call_count=1,
            full_text_eligibility_call_count=1,
            spend_usd=0.0,
            full_text_acquisition="local_prepositioned",
            bertopic="not_run",
            port=0,
        )
        _publish_resolution_suggestion_event(event_store, work_item_id="extrun-1:rec-1:field-1")

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "cost_cap_call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "cost_cap_ledger.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode.json").write_bytes(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8") + b"\n")
    (workspace / "run_mode_statement.md").write_bytes(render_run_mode_statement_md(run_mode).encode("utf-8"))

    source = _source_info(mode_profile="track3_live", execution_subtree_provenance="retrospective_reconstruction",
                           execution_subtree_clean=None, fixture_corpus_digest=None, cassette_sha256=None)
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track3_live",
    )
    return rb.open_bundle(bundle_dir / "payload")


def _build_track3_live_bundle_with_crlf_statement(
    harness_mod, monkeypatch, tmp_path, *, statement_text: str = None,
) -> rb.Bundle:
    """Regression fixture for the real Track 3 live run's own launcher bug
    (`track3_run_child.py` used to `write_text()` `run_mode_statement.md`,
    which silently translates every `\\n` to `\\r\\n` on a Windows host --
    confirmed to be the actual cause of the real live-run bundle's
    check_8 failure before both the launcher and check_8 were fixed).
    Identical to `_build_track3_live_bundle_with_real_run_mode` except the
    bundled statement file is deliberately written with `\\r\\n` line
    endings (or, if `statement_text` is given, that exact -- possibly
    genuinely wrong -- content) instead of the pure-`\\n` bytes
    `render_run_mode_statement_md()` itself produces."""
    monkeypatch.setattr(harness_mod, "ReviewConfig", _review_config_with_track3_live_overrides)
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type="scoping")
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    run_mode = build_run_mode_document_live(
        ai_use_events=event_store.load_all(),
        extraction_call_count=1,
        full_text_eligibility_call_count=1,
        spend_usd=0.0,
        full_text_acquisition="local_prepositioned",
        bertopic="not_run",
        port=0,
    )

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "cost_cap_call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "cost_cap_ledger.json").write_text("{}", encoding="utf-8")
    (workspace / "run_mode.json").write_bytes(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8") + b"\n")
    if statement_text is None:
        # simulate write_text()'s Windows CRLF translation exactly, without
        # depending on the actual host OS the test suite runs on
        statement_text = render_run_mode_statement_md(run_mode).replace("\n", "\r\n")
    (workspace / "run_mode_statement.md").write_bytes(statement_text.encode("utf-8"))

    source = _source_info(mode_profile="track3_live", execution_subtree_provenance="retrospective_reconstruction",
                           execution_subtree_clean=None, fixture_corpus_digest=None, cassette_sha256=None)
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track3_live",
    )
    return rb.open_bundle(bundle_dir / "payload")


def _finalize_stub_bundle(harness_mod, monkeypatch, tmp_path, *, review_type: str = "scoping") -> rb.Bundle:
    """The Track-2-context-INCOMPLETE fixture (`tests/test_reference_bundle.
    py::_finalize_from_harness`'s own convention, reproduced here): no
    `reference_reporting_context`, dedup mode left at the harness's own
    default (embedding, not exact -- see `slr_pipeline/nodes/dedup.py`'s
    `mode == "auto"` fallback), a non-schema-valid `run_mode.json` stub.
    Used for fail-closed coverage, never for an "all pass" assertion."""
    state = harness_mod._run_full_pipeline_to_reporting(monkeypatch, tmp_path, review_type=review_type)
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = state["review_run_id"]

    workspace = tmp_path / "driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = {
        "terminal_graph_snapshot_version": "1", "thread_id": "test-thread", "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "violations.jsonl").write_text("", encoding="utf-8")
    (workspace / "run_mode.json").write_text(
        json.dumps({"llm_transport": "replay", "search_source": "fixture"}), encoding="utf-8")
    (workspace / "run_mode_statement.md").write_text(
        "# Run mode\n\nReplay/fixture mode (test harness).\n", encoding="utf-8")

    source = _source_info(mode_profile="track2_offline")
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=review_run_id,
        reference_spec_sha256=_hex("test-reference-spec"), source=source,
        bundle_parent=tmp_path / "bundles", mode_profile="track2_offline",
        fixtures=_FIXTURES,
    )
    return rb.open_bundle(bundle_dir / "payload")


# ===========================================================================
# 2. Happy path -- all 8 checks pass
# ===========================================================================


def test_build_acceptance_report_happy_path_all_checks_pass(monkeypatch, tmp_path, harness):
    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    failing = [c for c in report.checks if c.status != "pass"]
    assert not failing, f"expected all 8 checks to pass, got failures: {failing!r}"
    assert [c.id for c in report.checks] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert report.acceptance_report_hash == report.self_hash()


def test_check_6_passes_when_evidence_map_legitimately_not_applicable():
    """Regression for the real Track 3 live MVP run (0 included studies):
    `synthesis.py` legitimately produces `evidence_map is None` whenever
    there are no records to chart, even for `review_type="scoping"`
    (`if cfg.synthesis.generate_evidence_map and records: ...`), which the
    real renderer gate (`nodes/reporting.py`'s `ARTIFACT_RENDERERS`) then
    stages as a `not_applicable` entry. Driving this through a full
    `review_type="rapid"` pipeline run is the WRONG way to test this --
    the whole bundle/acceptance framework is structurally scoping-only
    (`reference_bundle.py::from_bundle()` explicitly rejects any bundle
    whose terminal_state carries `quality_appraisal_run_id`), so this is
    a direct, lightweight unit test of `check_6_evidence_map()` itself
    instead, duck-typing only the two attributes it actually reads in the
    not_applicable branch (`bundle.reporting_stage_manifest.entries` and
    `snapshot.evidence_map`) -- confirmed sufficient by reading the
    function body: `bundle.resolve()` is never called on this branch."""
    entry = ReportingStageEntry(
        entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/evidence_map_csv",
        status="not_applicable", reason="generate_evidence_map is off, or full_text_records is empty",
        name="evidence_map_csv",
    )
    bundle = SimpleNamespace(reporting_stage_manifest=SimpleNamespace(entries=[entry]))
    snapshot = SimpleNamespace(evidence_map=None)

    check_6 = ra.check_6_evidence_map(bundle, snapshot)
    assert check_6.status == "pass"
    assert "not_applicable" in check_6.detail


def test_check_6_fails_when_not_applicable_status_contradicts_snapshot():
    """Same not_applicable entry, but `snapshot.evidence_map` is NOT None
    -- check_6 must not blindly trust the stored status; it cross-verifies
    against the snapshot and fails closed on a contradiction."""
    entry = ReportingStageEntry(
        entry_kind="renderer_artifact", logical_path="slr://reporting/artifact/evidence_map_csv",
        status="not_applicable", reason="generate_evidence_map is off, or full_text_records is empty",
        name="evidence_map_csv",
    )
    bundle = SimpleNamespace(reporting_stage_manifest=SimpleNamespace(entries=[entry]))
    snapshot = SimpleNamespace(evidence_map=object())

    check_6 = ra.check_6_evidence_map(bundle, snapshot)
    assert check_6.status == "fail"


# ===========================================================================
# 3. Fail-closed coverage against the Track-2-context-incomplete fixture
# ===========================================================================


def test_check_1_fails_closed_without_reference_context(monkeypatch, tmp_path, harness):
    bundle = _finalize_stub_bundle(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    check_1 = next(c for c in report.checks if c.id == 1)
    assert check_1.status == "fail"
    assert "reference_context" in check_1.detail


def test_check_1_passes_track3_live_without_reference_context(monkeypatch, tmp_path, harness):
    """The new `track3_live` branch: `reference_context` genuinely absent
    (a live run's honest state, not a Track-2-context-incomplete stub) --
    unlike the `track2_offline` case above, this must PASS, not fail."""
    bundle = _build_track3_live_bundle(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    check_1 = next(c for c in report.checks if c.id == 1)
    assert check_1.status == "pass"


def test_check_1_fails_closed_track3_live_with_reference_context_present(monkeypatch, tmp_path, harness):
    """A `track3_live` bundle whose snapshot somehow still carries a
    `reference_context` (should never happen for a real live run) must
    fail closed, not silently pass -- reuse the genuine track2_offline
    happy-path bundle (its snapshot has a real `reference_context`) and
    only tamper the field check_1 itself reads: `payload_manifest.
    mode_profile`. This never claims the tampered object is a
    self-consistent release (`verify_release_structure()` is not called
    here) -- it isolates check_1's own mode-branch logic."""
    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    tampered_pm = bundle.payload_manifest.model_copy(update={"mode_profile": "track3_live"})
    tampered = dataclasses.replace(bundle, payload_manifest=tampered_pm)
    report = ra.build_acceptance_report(tampered)
    check_1 = next(c for c in report.checks if c.id == 1)
    assert check_1.status == "fail"
    assert "reference_context must be absent" in check_1.detail


def test_check_5_passes_track3_live_not_applicable(monkeypatch, tmp_path, harness):
    bundle = _build_track3_live_bundle(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    check_5 = next(c for c in report.checks if c.id == 5)
    assert check_5.status == "pass"
    assert "not_applicable" in check_5.detail


def test_check_1_fails_closed_on_unknown_mode_profile(monkeypatch, tmp_path, harness):
    """A third/bogus `mode_profile` value must fail closed, never silently
    fall into either the track2_offline or track3_live branch -- exactly
    the gap code review round 1 flagged about a bare `else`. Call the
    check function directly (not via `build_acceptance_report()`, whose
    own `AcceptanceReport.mode_profile` field is Literal-typed and would
    raise `ValidationError` one layer further out -- a real bundle read
    off disk can never carry a non-Literal `mode_profile` in the first
    place, since `open_bundle()` parses `payload_manifest.json` through
    the same Literal-typed model; this test isolates check_1's own
    internal defense in depth from that outer, structurally-redundant
    guard)."""
    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    tampered_pm = bundle.payload_manifest.model_copy(update={"mode_profile": "bogus_mode"})
    tampered = dataclasses.replace(bundle, payload_manifest=tampered_pm)
    snapshot = ra.from_bundle(tampered)
    terminal_state = json.loads(tampered.resolve("slr://state/snapshot/terminal_state").read_bytes())
    terminal_graph_snapshot = json.loads(
        tampered.resolve("slr://state/snapshot/terminal_graph_snapshot").read_bytes())
    check_1 = ra.check_1_full_autonomy_terminal_state(
        tampered, snapshot, terminal_state=terminal_state, terminal_graph_snapshot=terminal_graph_snapshot)
    assert check_1.status == "fail"
    assert "unknown mode_profile" in check_1.detail

    with pytest.raises(ValidationError):
        ra.build_acceptance_report(tampered)  # the outer, structurally-redundant guard


def test_check_5_fails_closed_on_unknown_mode_profile(monkeypatch, tmp_path, harness):
    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    tampered_pm = bundle.payload_manifest.model_copy(update={"mode_profile": "bogus_mode"})
    tampered = dataclasses.replace(bundle, payload_manifest=tampered_pm)
    snapshot = ra.from_bundle(tampered)
    check_5 = ra.check_5_prisma_scr_report(tampered, snapshot)
    assert check_5.status == "fail"
    assert "unknown mode_profile" in check_5.detail


def test_check_8_fails_closed_on_non_schema_valid_run_mode(monkeypatch, tmp_path, harness):
    bundle = _finalize_stub_bundle(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    check_8 = next(c for c in report.checks if c.id == 8)
    assert check_8.status == "fail"
    assert "run_mode.json failed schema validation" in check_8.detail


def test_check_8_passes_track3_live_run_mode_built_by_live_builder(monkeypatch, tmp_path, harness):
    """End-to-end: `build_run_mode_document_live()`'s output, bundled as a
    real `track3_live` run, must satisfy check_8's own per-stage-count and
    ai_traceability crosswalk cross-checks -- not merely `RunModeDocument`
    schema validity (already covered by `tests/test_reference_run_mode.
    py`). Closes the plan's own stated Unit B test plan item ("Extend
    check_8_run_mode_statement's tests for a track3_live document")."""
    bundle = _build_track3_live_bundle_with_real_run_mode(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    check_8 = next(c for c in report.checks if c.id == 8)
    assert check_8.status == "pass"


def test_check_8_tolerates_crlf_run_mode_statement(monkeypatch, tmp_path, harness):
    """Regression for the real Track 3 live run's launcher bug: a bundled
    `run_mode_statement.md` written with Windows `\\r\\n` line endings must
    still pass check_8's byte-compare against `render_run_mode_statement_md()`'s
    pure-`\\n` output, so long as the content is otherwise identical."""
    bundle = _build_track3_live_bundle_with_crlf_statement(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    check_8 = next(c for c in report.checks if c.id == 8)
    assert check_8.status == "pass"


def test_check_8_still_fails_on_genuine_content_mismatch(monkeypatch, tmp_path, harness):
    """The CRLF/LF normalization in check_8 must not mask a genuine content
    divergence -- only line-ending differences are tolerated."""
    bundle = _build_track3_live_bundle_with_crlf_statement(
        harness, monkeypatch, tmp_path, statement_text="# Wrong content entirely\r\n")
    report = ra.build_acceptance_report(bundle)
    check_8 = next(c for c in report.checks if c.id == 8)
    assert check_8.status == "fail"
    assert "does not byte-match" in check_8.detail


def test_check_8_passes_with_matching_resolution_suggestion_event_count(monkeypatch, tmp_path, harness):
    """A bundled `extraction_resolution_suggestion` `AIUseEvent`, correctly
    counted in `run_mode.resolution_suggestion_calls`, must pass check_8's
    new per-stage cross-check (plan §3.4 round-4)."""
    bundle = _build_track3_live_bundle_with_resolution_suggestion_event(
        harness, monkeypatch, tmp_path, include_in_run_mode=True)
    report = ra.build_acceptance_report(bundle)
    check_8 = next(c for c in report.checks if c.id == 8)
    assert check_8.status == "pass"


def test_check_8_fails_on_mismatched_resolution_suggestion_event_count(monkeypatch, tmp_path, harness):
    """A real, bundled `extraction_resolution_suggestion` event that
    `run_mode.resolution_suggestion_calls` fails to count must fail
    closed, exactly like the six pre-existing per-stage cross-checks."""
    bundle = _build_track3_live_bundle_with_resolution_suggestion_event(
        harness, monkeypatch, tmp_path, include_in_run_mode=False)
    report = ra.build_acceptance_report(bundle)
    check_8 = next(c for c in report.checks if c.id == 8)
    assert check_8.status == "fail"
    assert "resolution_suggestion_calls" in check_8.detail


def test_build_acceptance_report_never_raises_on_incomplete_bundle(monkeypatch, tmp_path, harness):
    """The whole point of `_run()`'s per-check try/except: a malformed or
    Track-2-incomplete bundle produces a `status="fail"` report, never an
    uncaught exception."""
    bundle = _finalize_stub_bundle(harness, monkeypatch, tmp_path)
    report = ra.build_acceptance_report(bundle)
    assert len(report.checks) == 8
    assert any(c.status == "fail" for c in report.checks)


def test_check_4_fails_closed_on_internally_inconsistent_ledger_record(monkeypatch, tmp_path, harness):
    """`finalize_bundle()` itself already re-verifies every raw-layer
    `SelectionDecisionRecord`'s hashes while building the payload
    (`_load_verified_ledger_records` -> `SelectionDecisionStore.
    load_validated()` -> `load_all()` -> `record.verify_hashes()`) -- a
    tampered-on-disk record therefore can never survive INTO a finalized
    bundle in the first place, which check 4's own docstring undersells
    slightly ("`open_bundle()` never calls this") without also naming that
    even earlier build-time guard. This test proves check 4's independent
    `verify_hashes()` layer directly, in isolation, over a real happy-path
    `snapshot` with one `ledger_records` entry's `.model` swapped for a
    `.model_copy(update=...)` whose `reason` no longer matches its own
    (unchanged, now-stale) `semantic_hash`/`content_hash` fields -- never
    touching disk, so no earlier finalize-time guard intercepts it."""
    import dataclasses

    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    snapshot = rb.from_bundle(bundle)
    assert snapshot.ledger_records, "expected at least one ledger record in the happy-path bundle"
    tampered_record = snapshot.ledger_records[0]
    tampered_model = tampered_record.model.model_copy(
        update={"reason": (tampered_record.model.reason or "") + " (tampered, hash left stale)"}
    )
    tampered_ledger_records = [
        dataclasses.replace(tampered_record, model=tampered_model), *snapshot.ledger_records[1:],
    ]
    tampered_snapshot = dataclasses.replace(snapshot, ledger_records=tampered_ledger_records)

    check_4 = ra.check_4_ledger_hash_integrity(bundle, tampered_snapshot)
    assert check_4.status == "fail"
    assert "SelectionDecisionRecord" in check_4.detail


def test_check_4_fails_closed_on_internally_inconsistent_event(monkeypatch, tmp_path, harness):
    """Mirror of `test_check_4_fails_closed_on_internally_inconsistent_ledger_record`
    above, but over check 4's OTHER `verify_hashes()` call
    (`for vf in snapshot.events: vf.model.verify_hashes()`) -- an
    `AIUseEvent` whose `task_purpose` (not part of `event_id`'s own identity
    formula, so `_check_event_id_matches_identity` does not reject the
    tampered copy at construction time) no longer matches its own stale
    `semantic_hash`/`content_hash`. Without this test, a bug that made
    check 4 skip or no-op the `snapshot.events` loop (as opposed to the
    `snapshot.ledger_records` loop the test above already covers) could
    regress silently."""
    import dataclasses

    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    snapshot = rb.from_bundle(bundle)
    assert snapshot.events, "expected at least one AIUseEvent in the happy-path bundle"
    tampered_event = snapshot.events[0]
    tampered_model = tampered_event.model.model_copy(
        update={"task_purpose": tampered_event.model.task_purpose + " (tampered, hash left stale)"}
    )
    tampered_events = [dataclasses.replace(tampered_event, model=tampered_model), *snapshot.events[1:]]
    tampered_snapshot = dataclasses.replace(snapshot, events=tampered_events)

    check_4 = ra.check_4_ledger_hash_integrity(bundle, tampered_snapshot)
    assert check_4.status == "fail"
    assert "AIUseEvent" in check_4.detail


def test_relocate_and_refinalize(monkeypatch, tmp_path, harness):
    """Step 16's named test (plan §10.6/§13 step 16): relocates a real
    happy-path bundle to a fresh directory, DELETES the original driver
    workspace entirely (so nothing downstream can accidentally fall back to
    it), refinalizes the reporting stage from scratch against ONLY the
    relocated payload root, and confirms (a) every regenerated artifact's
    on-disk SHA-256 matches its own regenerated manifest entry, (b) every
    regenerated artifact's SHA-256 also matches the bundle's ORIGINALLY
    recorded value, and (c) the rebuilt `ReportingStageManifest` is
    entry-for-entry equal to the bundle's own recorded one -- modulo the
    one documented exception: `ai_traceability.json`/`.md`'s
    `full_text_eligibility_native_provenance.ft_screening_artifact_paths`
    re-anchors to the relocated bundle's own current `raw/ft_screening/`
    location on every `from_bundle()` reconstruction (see
    `reference_bundle.refinalize_reporting_from_bundle`'s own docstring,
    and `reference_acceptance.check_3_ai_traceability_shape`'s
    `_without_ft_native_provenance_paths`), which legitimately changes
    those two entries' `file_sha256` (and therefore `manifest_hash`)."""
    import shutil

    workspace = tmp_path / "driver_workspace"
    bundle = _build_happy_path_bundle(harness, monkeypatch, tmp_path)
    assert workspace.exists(), "expected the driver workspace this test is about to delete to exist"
    shutil.rmtree(workspace)

    relocated_root = tmp_path / "relocated_payload"
    shutil.copytree(bundle.payload_root, relocated_root)
    shutil.rmtree(bundle.payload_root)

    out_dir = tmp_path / "refinalized_reporting"
    refinalized_manifest = rb.refinalize_reporting_from_bundle(relocated_root, out_dir=out_dir)

    original_manifest = bundle.reporting_stage_manifest.model_dump(mode="json")
    reanchored_names = {"ai_traceability_json", "ai_traceability_md"}
    reanchored_logical_paths = {
        entry["logical_path"] for entry in original_manifest["entries"] if entry.get("name") in reanchored_names
    }

    def _normalize(manifest: dict) -> dict:
        manifest = json.loads(json.dumps(manifest))  # cheap deep copy
        for entry in manifest["entries"]:
            if entry["logical_path"] in reanchored_logical_paths:
                entry["file_sha256"] = None
        manifest["manifest_hash"] = None
        return manifest

    assert _normalize(refinalized_manifest) == _normalize(original_manifest), (
        "rebuilt ReportingStageManifest differs from the bundle's recorded one, beyond the "
        "documented ai_traceability re-anchoring exception"
    )

    for entry in refinalized_manifest["entries"]:
        if entry["entry_kind"] != "renderer_artifact" or entry["status"] != "present":
            continue
        filename = Path(entry["live_relpath"]).name
        on_disk_sha256 = file_sha256(out_dir / filename)
        assert on_disk_sha256 == entry["file_sha256"], (
            f"{filename}: on-disk SHA-256 does not match its own regenerated manifest entry")
        if entry["logical_path"] not in reanchored_logical_paths:
            original_entry = next(
                e for e in original_manifest["entries"] if e["logical_path"] == entry["logical_path"])
            assert on_disk_sha256 == original_entry["file_sha256"], (
                f"{filename}: regenerated SHA-256 does not match the bundle's originally recorded value")

    # The PRISMA diagram is singled out by name in the plan: it is, in this
    # codebase's actual implementation, a deterministic re-render (not a
    # byte-copy -- see `ARTIFACT_RENDERERS`'s `_render_prisma_diagram_png`),
    # but it must still reproduce byte-for-byte, which the generic loop
    # above already covers; assert its presence explicitly too.
    prisma_entry = next(e for e in refinalized_manifest["entries"] if e.get("name") == "prisma_diagram")
    assert prisma_entry["status"] == "present"


# ===========================================================================
# The determinism grep (plan §10.6/§13 step 16), styled after
# `tests/test_track2_packaging.py::test_no_test_side_path_injection`
# ===========================================================================


def test_renderer_determinism_grep():
    """Static scan, via `inspect.getsource`, of every function actually
    registered as a renderer in `ARTIFACT_RENDERERS` (`slr_pipeline/nodes/
    reporting.py`) plus the two PRISMA-ScR renderers it gates on
    `reference_context is not None` (`slr_pipeline/prisma_scr.py`) -- the
    exact set of functions whose output `test_relocate_and_refinalize`
    (above) and `test_two_independent_runs_project_equal` (elsewhere) both
    require to be byte-identical across independent runs/re-renders over
    the same logical content. The renderer name lists below are derived
    programmatically from `ARTIFACT_RENDERERS` itself (`_referenced_functions`,
    scanning each entry's own render_fn source one level deep -- enough to
    also pick up `_render_prisma_diagram_png`'s own call to
    `_draw_prisma_diagram`, since that entry's render_fn IS
    `_render_prisma_diagram_png` directly, not lambda-wrapped), not
    hand-maintained, so a future renderer added there without a matching
    update here cannot silently fall out of scope.

    Two mechanically-checkable determinism hazards, named verbatim in the
    plan (§10.6): a bare `datetime.now(`/`date.today(` call (wall-clock
    read -- these renderers must derive every date-shaped field from the
    snapshot, never the render-time clock: `reporting.py:2201` already
    carries the fix this test guards against regressing), and direct
    iteration over one of the three `VerifiedFile`-wrapped
    `ReportingSourceSnapshot` collections whose insertion order is not
    contractually stable (`snapshot.events`, `snapshot.ledger_records`,
    `snapshot.current_screening_batch_results` -- populated by
    `_from_bundle_events`/`_from_bundle_ledger`/
    `_from_bundle_screening_batches` on a `from_bundle()` reconstruction,
    or by whatever order the live graph itself appended them on a fresh
    run) without an enclosing `sorted(` on the same line, e.g.
    `_render_ai_traceability_dict`'s own `events_sorted = sorted((... for
    vf in snapshot.events), key=...)` -- the pattern every renderer that
    touches one of these three collections must follow.

    Caveat, honestly stated rather than papered over: the "no unsorted
    iteration" check below is a textual substring/lookback-window heuristic
    over each renderer's own source, not real data-flow analysis -- it would
    miss the collection being aliased to another name first (`events =
    snapshot.events; for vf in events: ...`) and iterated unsorted from
    there. Every renderer currently registered is clean under the direct
    (non-aliased) form this heuristic actually checks; a real fix would need
    an AST-based data-flow pass, judged disproportionate here given the
    current renderer set's own straightforward style.
    """
    import inspect
    import re

    from slr_pipeline.nodes import reporting as reporting_mod
    from slr_pipeline import prisma_scr as prisma_scr_mod

    def _referenced_functions(fn, modules):
        """Regex-extract every `name(` call site in `fn`'s own source and keep
        the ones that resolve to a real (non-lambda) function on one of
        `modules` -- used below to derive the renderer name lists straight
        from `ARTIFACT_RENDERERS` itself instead of hand-maintaining a
        parallel list, so a newly-added renderer is picked up automatically
        rather than silently skipped by this determinism check because
        someone forgot to also update the list here. Deliberately NOT
        restricted to names actually DEFINED on `module` (as opposed to
        merely imported into its namespace) -- `_render_ai_traceability_dict`/
        `_render_ai_traceability_md` are genuine renderers this check must
        cover, yet both live in `slr_pipeline.ai_traceability_report` and are
        only imported into `reporting.py`'s namespace; the price of that
        looseness is a handful of harmless non-renderer names (e.g.
        `canonical_json_bytes`) also getting swept in and redundantly
        scanned below, which cannot cause a false pass -- it only ever adds
        scrutiny, never removes it."""
        src = inspect.getsource(fn)
        found = set()
        for name in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", src):
            for module in modules:
                candidate = getattr(module, name, None)
                if candidate is not None and inspect.isfunction(candidate) and candidate.__name__ != "<lambda>":
                    found.add((module, name))
        return found

    _modules = (reporting_mod, prisma_scr_mod)
    seen: set = set()
    for entry in reporting_mod.ARTIFACT_RENDERERS:
        render_fn = entry[3]
        # `_render_prisma_diagram_png` is registered directly (not lambda-wrapped) --
        # its own name must be captured too, not just what it calls (`_draw_prisma_diagram`,
        # picked up below since that call literally appears in `render_fn`'s own source).
        if inspect.isfunction(render_fn) and render_fn.__name__ != "<lambda>":
            seen.add((reporting_mod, render_fn.__name__))
        seen |= _referenced_functions(render_fn, _modules)

    renderer_names_reporting = sorted(name for module, name in seen if module is reporting_mod)
    renderer_names_prisma_scr = sorted(name for module, name in seen if module is prisma_scr_mod)
    assert renderer_names_reporting, "derivation produced nothing -- ARTIFACT_RENDERERS shape must have changed"
    assert renderer_names_prisma_scr, "derivation produced nothing -- ARTIFACT_RENDERERS shape must have changed"

    _CLOCK_NEEDLES = ("datetime" + ".now(", "date" + ".today(")
    _UNSTABLE_COLLECTIONS = ("snapshot.events", "snapshot.ledger_records", "snapshot.current_screening_batch_results")
    _SORTED_LOOKBACK_CHARS = 200  # `sorted(...)` may open a line or two above the collection reference itself

    def _code_only(fn) -> str:
        """`inspect.getsource(fn)` minus its own leading docstring (a prose
        mention of `date.today()` describing what NOT to do, as in
        `_render_methods_md`'s own docstring, must not itself trip the
        wall-clock-call check)."""
        import ast
        import textwrap

        src = inspect.getsource(fn)
        tree = ast.parse(textwrap.dedent(src))
        func_node = tree.body[0]
        if ast.get_docstring(func_node) is None or not func_node.body:
            return src
        doc_node = func_node.body[0]
        lines = src.splitlines(keepends=True)
        return "".join(lines[doc_node.end_lineno:])  # everything after the docstring statement

    offenders: list[str] = []
    sources = [(reporting_mod, name) for name in renderer_names_reporting]
    sources += [(prisma_scr_mod, name) for name in renderer_names_prisma_scr]
    for module, name in sources:
        fn = getattr(module, name)
        code = _code_only(fn)
        qualname = f"{module.__name__}.{name}"
        for needle in _CLOCK_NEEDLES:
            idx = code.find(needle)
            if idx != -1:
                lineno = code.count("\n", 0, idx) + 1
                offenders.append(f"{qualname}:{lineno}: forbidden wall-clock read near {needle!r}")
        for collection in _UNSTABLE_COLLECTIONS:
            search_from = 0
            while True:
                idx = code.find(collection, search_from)
                if idx == -1:
                    break
                search_from = idx + len(collection)
                window = code[max(0, idx - _SORTED_LOOKBACK_CHARS):idx]
                if "sorted(" not in window:
                    lineno = code.count("\n", 0, idx) + 1
                    offenders.append(f"{qualname}:{lineno}: {collection} referenced without a preceding sorted(")

    assert not offenders, "renderer determinism grep found:\n" + "\n".join(offenders)
