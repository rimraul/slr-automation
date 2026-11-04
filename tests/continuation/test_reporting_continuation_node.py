"""Unit E -- `slr_pipeline.nodes.reporting_continuation.continuation_reporting_node`
and the full publish (`finalize_bundle(cumulative_snapshot=...)`) / accept
(`build_acceptance_report(parent_bundle=...)`) round trip for a genuine
continuation bundle."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

import slr_pipeline.reference_bundle as rb
from slr_pipeline.config import DedupConfig, ReviewConfig, ScreeningConfig
from slr_pipeline.continuation.cumulative_snapshot import build_cumulative_reporting_snapshot
from slr_pipeline.continuation.partition import ContinuationPartition
from slr_pipeline.extraction.hashing import file_sha256
from slr_pipeline.graphs.scoping_continuation import build_scoping_continuation_graph
from slr_pipeline.nodes.reporting_continuation import continuation_reporting_node
from slr_pipeline.reference_acceptance import build_acceptance_report
from slr_pipeline.reference_run_mode import (
    ModelCallsSummary,
    NonInstrumentedEntry,
    NonModelToolEvent,
    PerStageBackendEntry,
    RunModeDocument,
    SURFACE_TO_PER_STAGE_KEY,
    render_run_mode_statement_md,
)
from slr_pipeline.reference_spec import reference_reporting_context
from slr_pipeline.selection_ledger import _tails_by_key
from slr_pipeline.state import initial_state
from tests.continuation.conftest import _load_reference_bundle_test_module, release_verify_bundle


def _ft_citations(parent_snapshot):
    tails = _tails_by_key(
        [vf.model for vf in parent_snapshot.ledger_records if vf.model.stage == "full_text_eligibility"])
    return [
        {"record_id": record_id, "decision_id": rec.decision_id, "content_hash": rec.content_hash}
        for (record_id, _stage), rec in tails.items()
    ]


def _review_config_with_track2_overrides(*args, **kwargs):
    kwargs.setdefault("dedup", DedupConfig(mode="exact"))
    kwargs.setdefault("reference_reporting_context", reference_reporting_context().model_dump(mode="json"))
    return ReviewConfig(*args, **kwargs)


def _build_parent_bundle_with_exact_dedup(tmp_path, monkeypatch):
    """Same real, offline, fixture-backed pipeline run `build_parent_bundle()`
    uses, but with `dedup.mode == "exact"` + a real `reference_reporting_
    context` injected (mirrors `test_reference_acceptance.py`'s own
    `_review_config_with_track2_overrides`) -- required for the published
    continuation bundle's cumulative snapshot to carry a `dedup_exact`
    event (check_3) and a `present` PRISMA-ScR report (check_5)."""
    rb_test_module = _load_reference_bundle_test_module()
    harness = rb_test_module._load_reporting_node_harness()
    monkeypatch.setattr(harness, "ReviewConfig", _review_config_with_track2_overrides)
    bundle_dir, state, _cfg = rb_test_module._finalize_from_harness(harness, monkeypatch, tmp_path)
    bundle = rb.open_bundle(bundle_dir / "payload")
    # Code-review corrective pass, round 3: a genuinely release-verified
    # bundle (not merely a fabricated `run_registry_entry` bolted on) --
    # `from_bundle()` now calls `verify_release_structure()` on any parent
    # it is handed.
    bundle = release_verify_bundle(bundle)
    return bundle, state


def _pre_reporting_continuation_state(cfg, thread_id, payload_root, citations, monkeypatch):
    def _passthrough(state):
        return {}

    monkeypatch.setattr(
        "slr_pipeline.graphs.scoping_continuation.continuation_reporting_node", _passthrough)
    graph = build_scoping_continuation_graph()

    partition = ContinuationPartition(
        preserved_included_ids=[], preserved_excluded_ids=[], retrieved_ids=[], not_retrieved_ids=[])
    seeded = dict(initial_state(cfg.to_dict()))
    seeded["included_records"] = []
    seeded["continuation_partition"] = partition.model_dump()
    seeded["preserved_included_records"] = []
    seeded["retrieved_pdf_manifest"] = {}
    seeded["parent_bundle_payload_root"] = payload_root
    seeded["preserved_decision_citations"] = citations
    run_config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(seeded, run_config)


def _publish_continuation_bundle(tmp_path, monkeypatch):
    parent_bundle, _parent_state = _build_parent_bundle_with_exact_dedup(tmp_path / "parent", monkeypatch)
    parent_snapshot = rb.from_bundle(parent_bundle)
    citations = _ft_citations(parent_snapshot)

    cfg = ReviewConfig(
        title="Unit E continuation publish", output_dir=str(tmp_path / "continuation_outputs"),
        screening=ScreeningConfig(uncertain_threshold=0.0),
        reference_reporting_context=reference_reporting_context().model_dump(mode="json"),
    )
    thread_id = "unit-e-continuation-publish-run"
    pre_reporting_state = _pre_reporting_continuation_state(
        cfg, thread_id, str(parent_bundle.payload_root), citations, monkeypatch)

    # Real node call (not the graph) -- writes cfg.ensure_output_dir()/
    # "reporting"/review_run_id/ for real, exactly as `finalize_bundle()`'s
    # own docstring requires before it is invoked.
    reporting_update = continuation_reporting_node(pre_reporting_state)
    terminal_state = {**pre_reporting_state, **reporting_update}

    cumulative_snapshot = build_cumulative_reporting_snapshot(
        parent_bundle=parent_bundle, continuation_state=terminal_state, cfg=cfg,
        review_run_id=thread_id, preserved_decision_citations=citations,
    )

    # Real, per-stage `AIUseEvent` counts over the CUMULATIVE snapshot (not
    # just this run's own new events) -- check_8 compares `run_mode.json`'s
    # `per_stage_backend[*].calls` against `snapshot.events` counts on the
    # bundle it is validating, which for a continuation is the unioned set.
    stage_counts = Counter(vf.model.stage for vf in cumulative_snapshot.events)
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

    workspace = tmp_path / "continuation_driver_workspace"
    workspace.mkdir()
    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(json.dumps(terminal_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot_doc = {
        "terminal_graph_snapshot_version": "1", "thread_id": thread_id, "next": [],
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": thread_id, "captured_utc": rb._now_iso(),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workspace / "call_log.jsonl").write_text("", encoding="utf-8")
    (workspace / "violations.jsonl").write_text("", encoding="utf-8")
    (workspace / "run_mode.json").write_text(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # `write_bytes`, not `write_text` -- check 8 byte-compares this file
    # against a fresh `render_run_mode_statement_md()` call (see
    # `test_reference_acceptance.py`'s own identical note on Windows
    # newline translation).
    (workspace / "run_mode_statement.md").write_bytes(render_run_mode_statement_md(run_mode).encode("utf-8"))

    # Unit J: `finalize_bundle(parent_bundle_id=...)` requires these two
    # workspace files to already exist -- a genuine `publish_continuation_
    # bundle()` call would have written them (see `slr_pipeline.continuation.
    # publish_bundle`); written directly here since this test builds its
    # workspace by hand rather than through that entry point.
    (workspace / "preserved_decisions.json").write_text(
        json.dumps(citations, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from slr_pipeline.continuation.lineage import write_continuation_lineage
    # `_build_parent_bundle_with_exact_dedup()` now returns a genuinely
    # release-verified `parent_bundle` (via `release_verify_bundle()`), so
    # its real `run_registry_entry` is used directly here -- no fabricated
    # stand-in needed.
    write_continuation_lineage(
        path=workspace / "continuation_lineage.json", parent_bundle_id=parent_bundle.bundle_id,
        parent_bundle=parent_bundle, parent_snapshot=parent_snapshot, parent_workspace=None,
        continuation_manifest_path=tmp_path / "continuation_manifest.json",
        continuation_manifest_digest=_hex("test-continuation-manifest"), created_utc=rb._now_iso(),
    )

    rb_test_module = _load_reference_bundle_test_module()
    source = rb_test_module._source_info(mode_profile="track2_offline")
    bundle_dir = rb.finalize_bundle(
        workspace=workspace, cfg=cfg, review_run_id=thread_id,
        reference_spec_sha256=rb_test_module._hex("test-continuation-reference-spec"), source=source,
        bundle_parent=tmp_path / "continuation_bundles", mode_profile="track2_offline",
        fixtures=rb_test_module._FIXTURES, cumulative_snapshot=cumulative_snapshot,
        parent_bundle_id=parent_bundle.bundle_id,
    )
    continuation_bundle = rb.open_bundle(bundle_dir / "payload")
    return parent_bundle, continuation_bundle


def _hex(seed: str, n: int = 64) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:n]


def test_continuation_bundle_publishes_and_opens(tmp_path, monkeypatch):
    parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    assert continuation_bundle.payload_manifest.mode_profile == "track2_offline"
    assert len(continuation_bundle.payload_manifest.artifacts) > 0
    assert continuation_bundle.payload_manifest.parent_bundle_id == parent_bundle.bundle_id
    logical_paths = {a.logical_path for a in continuation_bundle.payload_manifest.artifacts}
    assert "slr://continuation/lineage/main" in logical_paths
    assert "slr://continuation/preserved_decisions/main" in logical_paths


def test_build_acceptance_report_parent_bundle_round_trip(tmp_path, monkeypatch):
    parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    report = build_acceptance_report(continuation_bundle, parent_bundle=parent_bundle)
    assert len(report.checks) == 8
    failing = [c for c in report.checks if c.status != "pass"]
    assert not failing, f"expected all 8 checks to pass: {failing}"


# ---------------------------------------------------------------------------
# Finding 6 (code-review corrective pass, post-95c9667): `from_bundle()`
# enforces an exact `parent_bundle.bundle_id` match against the declared
# `parent_bundle_id`, and cross-checks `continuation_lineage.json` against
# the actual parent's registry hashes -- a caller can no longer silently
# reconstruct a continuation's snapshot against the wrong (or no) parent.
# ---------------------------------------------------------------------------


def test_from_bundle_rejects_wrong_parent_bundle(tmp_path, monkeypatch):
    parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    unrelated_bundle, _ = _build_parent_bundle_with_exact_dedup(tmp_path / "unrelated", monkeypatch)

    with pytest.raises(rb.ReferenceBundleError, match="parent_bundle_id"):
        rb.from_bundle(continuation_bundle, parent_bundle=unrelated_bundle)


def test_from_bundle_requires_parent_bundle_when_declared(tmp_path, monkeypatch):
    _parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)

    with pytest.raises(rb.ReferenceBundleError, match="parent_bundle is required"):
        rb.from_bundle(continuation_bundle)


def test_from_bundle_rejects_parent_bundle_when_not_a_continuation(tmp_path, monkeypatch):
    parent_bundle, _parent_state = _build_parent_bundle_with_exact_dedup(tmp_path / "parent", monkeypatch)

    with pytest.raises(rb.ReferenceBundleError, match="does not declare a parent_bundle_id"):
        rb.from_bundle(parent_bundle, parent_bundle=parent_bundle)


def test_build_acceptance_report_rejects_wrong_parent_bundle(tmp_path, monkeypatch):
    """`build_acceptance_report()`'s own presence-only check is only a fast,
    early failure -- the real enforcement is inside `from_bundle()`, which
    `build_acceptance_report()` calls. A wrong-but-present parent_bundle is
    not silently accepted."""
    _parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    unrelated_bundle, _ = _build_parent_bundle_with_exact_dedup(tmp_path / "unrelated", monkeypatch)

    with pytest.raises(rb.ReferenceBundleError, match="parent_bundle_id"):
        build_acceptance_report(continuation_bundle, parent_bundle=unrelated_bundle)


# ---------------------------------------------------------------------------
# Finding 2 (code-review corrective pass, round 2): the lineage hash
# cross-check must never be silently skipped just because the caller opened
# `parent_bundle` without wiring a `run_registry_entry` through -- a
# continuation's parent must always be a release-verified registry entry.
# ---------------------------------------------------------------------------


def test_from_bundle_requires_registry_entry_on_parent(tmp_path, monkeypatch):
    import dataclasses

    parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    # Otherwise identical/correct (same bundle_id, same lineage-matching
    # hashes) -- only `run_registry_entry` itself is stripped off, exactly
    # the gap `open_bundle(payload_root)` (no `run_registry_entry=`
    # keyword) leaves open for any caller other than `continuation.
    # identity.resolve_parent_bundle()`.
    parent_without_registry_entry = dataclasses.replace(parent_bundle, run_registry_entry=None)

    with pytest.raises(rb.ReleaseStructureInvalid, match="run_registry_entry"):
        rb.from_bundle(continuation_bundle, parent_bundle=parent_without_registry_entry)


def test_build_acceptance_report_requires_registry_entry_on_parent(tmp_path, monkeypatch):
    import dataclasses

    parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    parent_without_registry_entry = dataclasses.replace(parent_bundle, run_registry_entry=None)

    with pytest.raises(rb.ReleaseStructureInvalid, match="run_registry_entry"):
        build_acceptance_report(continuation_bundle, parent_bundle=parent_without_registry_entry)


# ---------------------------------------------------------------------------
# Code-review corrective pass, round 3: a `run_registry_entry` merely being
# *present* proves nothing -- it must match a genuinely release-verified
# bundle's own sidecars. A fabricated entry attached to an otherwise
# unverified parent (no real acceptance_report/reviewer_attestation/
# release_manifest) must be rejected exactly like a missing one.
# ---------------------------------------------------------------------------


def test_from_bundle_rejects_fabricated_registry_entry_without_real_sidecars(tmp_path, monkeypatch):
    import dataclasses

    parent_bundle, continuation_bundle = _publish_continuation_bundle(tmp_path, monkeypatch)
    parent_with_fabricated_entry = dataclasses.replace(
        parent_bundle,
        acceptance_report=None, reviewer_attestation=None, release_manifest=None,
        run_registry_entry=rb.RunRegistryEntry(
            payload_hash=parent_bundle.payload_manifest.payload_hash, release_hash=_hex("stand-in-release-hash"),
            acceptance_report_hash=_hex("stand-in-acceptance-report-hash"),
            attestation_hash=_hex("stand-in-attestation-hash"),
            mode_profile=parent_bundle.payload_manifest.mode_profile, decision="APPROVED"))

    with pytest.raises(rb.ReleaseStructureInvalid):
        rb.from_bundle(continuation_bundle, parent_bundle=parent_with_fabricated_entry)
