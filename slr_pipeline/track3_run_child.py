"""Track 3 live-run launcher plan (track3_live_run_plan_2026-09-06.md),
Unit C (squishy-petting-starlight.md, C2/C5): the live-run **child**
process.

Invoked by `track3_run.py` (the parent driver) as a **subprocess**, with
the fresh env `track3_run.py` builds -- notably `OPENAI_BASE_URL=
http://127.0.0.1:<port>/v1` pointed at a `CostCapProxy` this process
itself starts, and the real `OPENAI_API_KEY` passed through. Unlike
`reference_run_child.py`, this module arms no process-wide network guard
(C3, an explicit, documented scope decision -- see `track3_run.py`'s own
module docstring) -- a live run legitimately needs PubMed/Scopus/OpenAI
network access, which a single-`(host,port)`-pair guard cannot express.

Responsibilities, in order:
  1. Parse the port from `OPENAI_BASE_URL` (pure stdlib `urllib.parse`,
     no network I/O) -- needed only so the `CostCapProxy` this process
     starts binds the SAME port the parent already committed to.
  2. Load `--spec` into a `ReviewConfig`; independently re-run the SAME
     preflight assertions `track3_run.py::run_track3()` already ran
     (`assert_databases_allowed`, `assert_no_execution_adapter_map`,
     `assert_dedup_mode_exact`, `assert_search_credentials_present`) --
     defense in depth (C2) means two independent checks, one per process,
     never one check that a second process merely trusts the first already
     made.
  3. Start `slr_pipeline.cost_cap.proxy.CostCapProxy` bound to that port.
  4. `graph = build_scoping_graph(); state = initial_state(cfg.to_dict());
     graph.invoke(state, config)`.
  5. Terminal-state checks, mode-agnostic, identical in spirit to
     `reference_run_child.py`'s own: both review queues empty,
     `graph_state.next == []`.
  6. Write `terminal_state.json` / `terminal_graph_snapshot.json` as flat
     files at the workspace root.
  7. Build the real `AIUseEvent`/extraction-call-count/full-text-
     eligibility-call-count inputs `build_run_mode_document_live()` needs,
     from the SAME durable, re-verified artifacts `finalize_bundle()`
     itself reads (`AIUseEventStore`/`_load_verified_events`, extraction
     checkpoints' own `DocumentExtractionResult.batch_executions`,
     FT_SCREENING's own `ScreeningRunResult.items`) -- never re-derived
     from the cost-cap proxy's own call log, so this document cannot
     silently drift from what a future `finalize_bundle()` call will
     independently re-verify.
  8. `build_run_mode_document_live(...)`; write `run_mode.json` /
     `run_mode_statement.md`.

Dropped relative to `reference_run_child.py` (C3's own scope decision,
see step 1 above and `track3_run.py`'s module docstring): no
`ReferenceNetGuard` arming, no `violations.jsonl`.

Dropped relative to `reference_run_child.py` (no cassette/replay concept
for a live run): no `ReplayServer`/`assert_complete()` (the ledger itself
naturally enforces "no `BudgetExceededError` raised" as the live
equivalent -- `graph.invoke()` completing at all already means the cap
was never exceeded); no fixed extraction-output-dir lock -- `cfg.
output_dir` is workspace-relative and already unique per run (the
parent's own `allocate_workspace()`), so no cross-run collision is
possible and no lock is needed.

`finalize_bundle()` is explicitly **not** called here -- bundle
publication is out of scope for this plan (see `track3_run.py`'s own
module docstring)."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def _parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path,
                         help="Harness dir for terminal_state.json / terminal_graph_snapshot.json "
                              "/ cost_cap_call_log.jsonl / run_mode.json / run_mode_statement.md.")
    parser.add_argument("--spec", required=True, type=Path, help="ReviewConfig-shaped JSON spec file.")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="cfg.output_dir -- workspace-relative, unique per run.")
    parser.add_argument("--cost-cap-path", required=True, type=Path,
                         help="Fresh, per-run CostLedger path (a CostLedger may only ever be held "
                              "open by one process on a given path).")
    parser.add_argument("--cap-usd", required=True, type=float)
    parser.add_argument("--thread-id", default="track3-live-run")
    return parser.parse_args(argv)


def _proxy_port_from_env() -> int:
    """Pure stdlib, no network I/O. `track3_run.py` sets `OPENAI_BASE_URL`
    to `http://127.0.0.1:<port>/v1` in this process's own environment
    before spawning it; this parses that same port back out so the
    `CostCapProxy` this process starts binds the one port the parent
    already committed to."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL must be set in the child environment")
    parsed = urlparse(base_url)
    if parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise RuntimeError(f"OPENAI_BASE_URL must be http://127.0.0.1:<port>/v1, got {base_url!r}")
    return parsed.port


def main(argv: list) -> int:
    args = _parse_args(argv)
    workspace: Path = args.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    port = _proxy_port_from_env()
    return _run(args, port=port, workspace=workspace)


def _run(args: argparse.Namespace, *, port: int, workspace: Path) -> int:
    from slr_pipeline.config import ReviewConfig
    from slr_pipeline.track3_run import (
        assert_databases_allowed,
        assert_no_execution_adapter_map,
        assert_dedup_mode_exact,
        assert_search_credentials_present,
    )

    cfg = ReviewConfig.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
    # C2, defense in depth: re-checked here, independent of the parent's own
    # already-passed check -- see module docstring step 2.
    assert_databases_allowed(cfg.search.databases)
    assert_no_execution_adapter_map(cfg.search.execution_adapter_map)
    assert_dedup_mode_exact(cfg.dedup.mode)
    assert_search_credentials_present(cfg.search.databases, base_env=os.environ)

    cfg_dict = cfg.to_dict()
    cfg_dict["output_dir"] = str(args.output_dir)
    cfg = ReviewConfig.from_dict(cfg_dict)
    cfg.ensure_output_dir()

    return _invoke_graph(args, cfg, port=port, workspace=workspace)


def _open_checkpointer(workspace: Path):
    """Workspace-relative, durable graph checkpoint DB -- monkeypatchable test
    seam. A crash-then-restart against the same workspace/thread_id resumes
    from the last completed node instead of repeating already-paid stages
    (unlike the prior MemorySaver default, which loses all state on process
    exit)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(str(workspace / "graph_checkpoints.sqlite"))


def _invoke_graph(args: argparse.Namespace, cfg, *, port: int, workspace: Path) -> int:
    from slr_pipeline.cost_cap.ledger import CostLedger
    from slr_pipeline.cost_cap.proxy import CostCapProxy
    from slr_pipeline.extraction.hashing import file_sha256
    from slr_pipeline.graphs.scoping import build_scoping_graph
    from slr_pipeline.state import initial_state

    ledger = CostLedger(args.cost_cap_path, cap_usd=args.cap_usd)
    call_log_path = workspace / "cost_cap_call_log.jsonl"

    proxy = CostCapProxy(ledger, call_log_path, port=port)
    proxy.start()
    try:
        with _open_checkpointer(workspace) as checkpointer:
            graph = build_scoping_graph(checkpointer=checkpointer)
            run_config = {"configurable": {"thread_id": args.thread_id}}
            existing_state = graph.get_state(run_config)
            if existing_state.values:
                # Resuming a prior checkpoint for this thread_id -- do not
                # re-invoke already-completed nodes (proven empirically: a
                # completed node's function is not called again on resume).
                terminal_state = graph.invoke(None, run_config)
            else:
                state = initial_state(cfg.to_dict())
                terminal_state = graph.invoke(state, run_config)
            graph_state = graph.get_state(run_config)
    finally:
        proxy.stop()

    if terminal_state.get("full_text_review_queue"):
        raise RuntimeError("full_text_review_queue is non-empty at graph termination -- "
                            "a live Track 3 run requires a fully-automated pass")
    if terminal_state.get("extraction_review_queue"):
        raise RuntimeError("extraction_review_queue is non-empty at graph termination -- "
                            "a live Track 3 run requires a fully-automated pass")

    next_nodes = list(graph_state.next)
    if next_nodes:
        raise RuntimeError(f"graph did not reach a true terminal state -- next={next_nodes!r}")

    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps(terminal_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    review_run_id = terminal_state.get("review_run_id")
    if not review_run_id:
        raise RuntimeError("terminal_state['review_run_id'] is unset -- search_strategy_builder_node "
                            "(the first node in the graph, which unconditionally sets review_run_id) "
                            "did not run as expected on this run")

    snapshot = {
        "terminal_graph_snapshot_version": "1",
        "thread_id": args.thread_id,
        "next": next_nodes,
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_mode = _build_run_mode(cfg, terminal_state, review_run_id=review_run_id, ledger=ledger, port=port)
    (workspace / "run_mode.json").write_text(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from slr_pipeline.reference_run_mode import render_run_mode_statement_md
    # write_bytes, not write_text -- check_8 byte-compares this file against
    # a fresh render_run_mode_statement_md() call, whose own output is pure
    # '\n'; write_text()'s universal-newline translation would silently turn
    # every '\n' into '\r\n' on a Windows host (this launcher, unlike Track
    # 2's Linux-CI-only driver, runs locally on Windows).
    (workspace / "run_mode_statement.md").write_bytes(
        render_run_mode_statement_md(run_mode).encode("utf-8")
    )
    return 0


def _build_run_mode(cfg, terminal_state, *, review_run_id: str, ledger, port: int):
    from slr_pipeline.ai_traceability import AIUseEventStore
    from slr_pipeline.nodes.reporting import _load_verified_events
    from slr_pipeline.reference_run_mode import build_run_mode_document_live

    event_store = AIUseEventStore(output_dir=cfg.ensure_output_dir(), review_run_id=review_run_id)
    ai_use_events = [vf.model for vf in _load_verified_events(event_store)]

    bertopic = "enabled" if cfg.synthesis.run_bertopic else "disabled"

    return build_run_mode_document_live(
        ai_use_events=ai_use_events,
        extraction_call_count=_extraction_call_count(terminal_state, cfg),
        full_text_eligibility_call_count=_full_text_eligibility_call_count(terminal_state, cfg),
        spend_usd=ledger.snapshot()["committed_usd"],
        full_text_acquisition="live_download",
        bertopic=bertopic,
        port=port,
        resolution_suggestion_interrupted=_resolution_suggestion_interrupted_count(terminal_state, cfg),
    )


def _resolution_suggestion_interrupted_count(terminal_state, cfg) -> int:
    """Real, on-disk count of this run's `*.interrupted_unknown.json`
    extraction-resolution-suggestion archives
    (`RunModeDocument.resolution_suggestion_interrupted`, LLM-resolution
    plan sec. 3.4 round-5 addition) -- an interrupted attempt never
    publishes an `AIUseEvent` (nodes/reporting.py's own
    `_load_extraction_review_artifacts` forbids one), so it is invisible
    to `ai_use_events` and every count derived from it above. Same
    `suggestions_dir` location `nodes/reporting.py::_load_extraction_bundle`
    resolves via `ExtractionCheckpointStore(...).run_dir` for the same
    `extraction_run_id`. Shared by both Track 3 launcher paths --
    `track3_continuation_run.py` calls this same `_build_run_mode()`
    rather than duplicating it."""
    from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
    from slr_pipeline.extraction.resolution_suggestions import count_interrupted_suggestions

    run_id = terminal_state.get("extraction_run_id")
    if not run_id:
        return 0
    store = ExtractionCheckpointStore(output_dir=cfg.ensure_output_dir() / "extraction_runs", run_id=run_id)
    return count_interrupted_suggestions(store.run_dir / "extraction_resolution_suggestions")


def _extraction_call_count(terminal_state, cfg) -> int:
    """Real OpenAI call count for the extraction stage: `checkpoints` (one
    per extracted document) is NOT the call count -- a document's fields
    may be split across multiple batches, each batch its own real call
    (`extraction/results.py::DocumentExtractionResult.batch_executions`,
    one `BatchExecutionRecord` per real provider call). Summing
    `len(batch_executions)` across every document's checkpoint is the true
    per-run call count, matching Track 2's own `extraction_batch` cassette-
    surface semantics (one cassette entry per (document, batch) pair)."""
    from slr_pipeline.extraction.results import DocumentExtractionResult
    from slr_pipeline.nodes.reporting import _load_extraction_bundle

    bundle = _load_extraction_bundle(terminal_state, cfg)
    if bundle is None:
        return 0
    total = 0
    for component in bundle.checkpoints:
        doc_result = DocumentExtractionResult.model_validate_json(component.path.read_text(encoding="utf-8"))
        total += len(doc_result.batch_executions)
    return total


def _full_text_eligibility_call_count(terminal_state, cfg) -> int:
    """`run_result.items` is one line per item **submitted** to FT_SCREENING
    (`nodes/reporting.py`'s own `capture_ft_native_provenance()` docstring),
    NOT one per real OpenAI call -- an item can resolve via
    `input_unavailable`/`duplicate_input`/`processing_failed`/
    `not_processed` (no call at all) or via a cache hit (`execution_status
    == "completed"` but zero provider calls), so `len(run_result.items)`
    overcounts. `ScreeningItemResult.provenance` is only ever set for
    `execution_status == "completed"` items (confirmed:
    `orchestrator.py::_terminal_result()` always passes `provenance=None`
    for every other status); its `total_provider_calls` (orchestrator.py:
    `sum(u.provider_call_count for u in usage_records)`) is the exact real
    per-item provider-call count, already 0 for a pure cache hit. Summing
    that field across all items, not counting items, is the true call
    count -- EXCEPT for items with `provenance.cache_status ==
    "content_reuse"`: when `duplicate_content_policy="reuse"` is set (a
    spec-controllable `ScreeningConfig` field, reachable from a Track 3
    spec's `full_text_screening_config`), `orchestrator.py::_clone_for_reuse()`
    clones the canonical (real-call) item's `provenance` verbatim onto every
    content-duplicate item, including its non-zero `total_provider_calls` --
    that value describes the ONE real call already made for the canonical
    item, not an additional call made for the duplicate. Summing it again per
    duplicate would count the same real call multiple times, so
    `content_reuse` items are excluded from the sum (their real call is
    already counted via the canonical item's own line in `run_result.items`)."""
    from ft_screening import ScreeningRunResult
    from slr_pipeline.nodes.reporting import capture_ft_native_provenance

    ft_native = capture_ft_native_provenance(terminal_state, cfg)
    if ft_native is None:
        return 0
    run_result_path = Path(ft_native["ft_screening_artifact_paths"]["run_result_json"])
    run_result = ScreeningRunResult.model_validate_json(run_result_path.read_text(encoding="utf-8"))
    return sum(
        item.provenance.total_provider_calls
        for item in run_result.items
        if item.provenance is not None and item.provenance.cache_status != "content_reuse"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
