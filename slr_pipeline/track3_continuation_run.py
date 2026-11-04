"""Track 3 continuation launcher (Units A/B/F/G/H). Wires together the
already-implemented building blocks -- parent-bundle resolution (Unit A),
partition derivation (Unit B), PDF re-acquisition (Unit A items 12/15), the
whole-pipeline worst-case cost estimate (Unit F), the cap/authorization
gate around a real continuation graph run (Unit G), and the pinned
synthesis configuration (Unit H) -- behind a single CLI entry point.

`cfg.synthesis` is pinned via `_pin_continuation_synthesis_config()`
immediately after loading `--spec`, before either the estimate or the real
run sees it -- the JSON spec's own `synthesis` block, if any, is discarded
in favor of this continuation's fixed values (see that function's own
docstring for the exact pinned values and why).

`--estimate-only` prints a `ContinuationCostEstimate` breakdown and exits,
WITHOUT ever importing `slr_pipeline.cost_cap.ledger`/
`slr_pipeline.cost_cap.proxy` (only `cost_estimate.py`'s own import of
`cost_cap.pricing` runs) and WITHOUT constructing a `CostLedger` or opening
a `CostCapProxy`.

The real (non-`--estimate-only`) path requires `--cap-usd` explicitly (no
default -- never silently defaulted, and never accepted from the JSON spec,
matching the parent live-run launcher's own standing constraint), prints
the SAME Unit F breakdown first (conservative planning information for the
human to weigh, not a value this gate itself trusts as exact), then
constructs exactly ONE `CostLedger`/`CostCapProxy` pair for the whole run
and points `OPENAI_BASE_URL` at it before invoking the continuation graph --
the same env-var-redirect mechanism `track3_run.py`/`track3_run_child.py`
already use, reused unmodified here (`cost_cap/ledger.py`/`cost_cap/
proxy.py` are neither modified nor reimplemented). Every real OpenAI-calling
code path the continuation graph reaches -- `extraction/openai_runner.py`
(extraction), `nodes/synthesis.py` (synthesis), and the external
`ft_screening` package's `adjudication.py` (full-text eligibility, via
`continuation_fulltext_node`) -- constructs its client with no explicit
`base_url=`, so all three transparently pick up the proxy through this one
env var. Bundle publish (Unit J) is a separate, later, out-of-process step
(`slr_pipeline.continuation.publish_bundle.publish_continuation_bundle()`)
-- this launcher only reaches a true terminal graph state and writes the
workspace artifacts that step requires (`terminal_state.json`,
`terminal_graph_snapshot.json`, `run_mode.json`, `run_mode_statement.md`);
it never calls `finalize_bundle()` itself.

Code-review corrective pass (post-commit-95c9667, round 1, all in
`run_continuation()` unless noted): `state["prisma_counts"]` is now seeded
from the parent's own real cumulative PRISMA counts; the graph is invoked
against a workspace-persisted `thread_id.txt` + workspace `SqliteSaver`
checkpointer (resume-vs-fresh, matching `track3_run_child.py`) instead of a
fresh random thread_id against the default in-memory checkpointer; a
non-empty review queue or a graph that did not reach a true terminal state
now raises instead of silently returning as if successful; and `--pdf-dir`
no longer exists as an independent flag -- `cfg.output_dir` (and therefore
`pdf_dir`) is forced under `--workspace` in `main()`, so it can never
diverge from what `continuation_fulltext_node()` itself derives.

Code-review corrective pass, round 2: a genuinely zero-spend run (every
record preserved, nothing newly retrieved) now still leaves a real
`cost_cap_ledger.json`/`cost_cap_call_log.jsonl` pair on disk (`CostLedger`
only persists itself after a first committed spend, and
`cost_cap_call_log.jsonl` is only ever opened lazily on the first logged
call -- see `run_continuation()`'s own call to `ledger.ensure_persisted()`)
so a zero-call continuation can still be published; resuming an existing
checkpoint now compares the newly-resolved parent/config/partition/manifest
against what the checkpoint was actually seeded with, and refuses to
resume on any mismatch (`_assert_seed_matches_checkpoint()`) -- previously
`graph.invoke(None, run_config)` would silently keep resuming whatever the
ORIGINAL invocation seeded regardless of what this invocation resolved,
while this launcher went on to write cost-cap/run-mode artifacts derived
from the (unused) current, mismatched inputs.

Real-data preflight corrective pass: a real `--estimate-only` invocation
attempted live Hugging Face access while resolving Docling's parsing
models -- this launcher never configured offline model resolution the way
`track3_run.py`'s own MVP launcher does for its child subprocess. Unlike
that launcher (which deliberately leaves `HF_HUB_OFFLINE`/
`TRANSFORMERS_OFFLINE` unset for its child, since a live run may
legitimately need real network to populate a fresh `HF_HOME` for
`bertopic`), this continuation launcher always pins
`cfg.synthesis.run_bertopic=False` (`_pin_continuation_synthesis_config()`)
and never runs a `dedup`/`search_strategy_builder` node
(`graphs/scoping_continuation.py` never registers either) -- the only two
production code paths that construct a `SentenceTransformer` -- so forcing
full offline mode is safe here, unlike in the MVP launcher.
`_docling_offline_env()` now validates the existing `DOCLING_CACHE_DIR`
asset bundle (reusing `track3_run.py`'s own
`assert_docling_artifacts_valid()`/`DOCLING_CACHE_DIR`, the identical
bundle Track 2's `reference_run.py` and Track 3's `track3_run.py` already
rely on) and sets `DOCLING_CACHE_DIR`/`HF_HUB_OFFLINE`/
`TRANSFORMERS_OFFLINE` for the duration of both the cost estimate
(`main()`, around the `estimate_continuation_cost()` call --
`parse_pdf()` runs for real there) and the actual graph execution
(`run_continuation()`, around the checkpointer/graph-invoke block),
restoring each variable to its prior value (or absence) afterward in both
places -- this process's environment is never left mutated once either
call returns.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from slr_pipeline import credentials
from slr_pipeline.config import ReviewConfig, SynthesisConfig
from slr_pipeline.continuation.cost_estimate import estimate_continuation_cost
from slr_pipeline.continuation.identity import check_parent_workspace_consistency, resolve_parent_bundle
from slr_pipeline.continuation.manifest import load_continuation_manifest
from slr_pipeline.continuation.partition import ContinuationPartition, derive_continuation_partition
from slr_pipeline.continuation.preserved_pdfs import acquire_preserved_included_pdfs
from slr_pipeline.continuation.retrieved_pdfs import acquire_retrieved_pdfs
from slr_pipeline.graphs.scoping_continuation import build_scoping_continuation_graph
from slr_pipeline.reference_bundle import Bundle, from_bundle
from slr_pipeline.selection_ledger import _tails_by_key
from slr_pipeline.state import Record, initial_state
from slr_pipeline.track3_run import DOCLING_CACHE_DIR, Track3ConfigError, assert_docling_artifacts_valid


def _pin_continuation_synthesis_config(cfg: ReviewConfig) -> ReviewConfig:
    """Unit H (squishy-petting-starlight.md, codex point 8): the
    continuation's `cfg.synthesis` is pinned explicitly, never left at
    whatever the JSON spec happens to say -- `generate_claim_backed_results`
    stays `False` (the extra `"imrad:results"` target is out of MVP scope;
    ordinary per-RQ synthesis runs regardless of this flag,
    `nodes/synthesis.py:1741-1743`), `generate_evidence_map=True` (pure
    local aggregation, no LLM call, `_build_evidence_map()`),
    `generate_legacy_imrad_narrative=False` (off by default per its own
    documented convention), `run_bertopic=False` (unconditional MVP
    decision for this continuation -- `assert_bertopic_corpus_size_or_
    disabled()` in `track3_run.py` remains the general mechanism for any
    future caller, but is never exercised here since this pin never enables
    it). `refine_batch_size` is left at `SynthesisConfig`'s own default (5)."""
    return cfg.model_copy(update={
        "synthesis": SynthesisConfig(
            run_bertopic=False,
            generate_evidence_map=True,
            generate_claim_backed_results=False,
            generate_legacy_imrad_narrative=False,
        )
    })


def _load_included_records(parent_bundle: Bundle) -> list:
    terminal_state_path = parent_bundle.resolve("slr://state/snapshot/terminal_state")
    terminal_state = json.loads(terminal_state_path.read_bytes())
    return [Record.from_dict(r) for r in terminal_state["included_records"]]


class _ResolvedContinuationInputs:
    def __init__(
        self, *, included_records: List[Record], parent_snapshot, partition: ContinuationPartition,
        preserved_included_parent_records: List[Record], retrieved_manifest_docs: Dict[str, Dict[str, str]],
    ) -> None:
        self.included_records = included_records
        self.parent_snapshot = parent_snapshot
        self.partition = partition
        self.preserved_included_parent_records = preserved_included_parent_records
        self.retrieved_manifest_docs = retrieved_manifest_docs


def _resolve_continuation_inputs(
    *,
    parent_bundle: Bundle,
    parent_workspace: Optional[Path],
    continuation_manifest_path: Path,
    retrieved_pdfs_dir: Path,
    pdf_dir: Path,
) -> _ResolvedContinuationInputs:
    """Shared by `--estimate-only` and the real run: derives the partition
    and re-acquires every PDF a run would need. Reused verbatim by both so
    the real run never re-decides the partition or re-acquires PDFs a
    second, possibly different, time from what was estimated."""
    check_parent_workspace_consistency(parent_workspace)

    included_records = _load_included_records(parent_bundle)
    parent_snapshot = from_bundle(parent_bundle)
    continuation_manifest = load_continuation_manifest(continuation_manifest_path)
    partition: ContinuationPartition = derive_continuation_partition(
        included_records, parent_snapshot, continuation_manifest)

    if partition.preserved_included_ids and parent_workspace is None:
        raise Track3ConfigError(
            "--parent-workspace is required: the parent snapshot has "
            f"{len(partition.preserved_included_ids)} preserved-included record(s), whose PDFs can only "
            "be re-acquired from the parent's own workspace")

    preserved_included_parent_records = acquire_preserved_included_pdfs(
        partition.preserved_included_ids, parent_snapshot, parent_workspace, pdf_dir,
    ) if partition.preserved_included_ids else []

    retrieved_manifest_docs = acquire_retrieved_pdfs(
        partition.retrieved_ids, included_records, continuation_manifest, retrieved_pdfs_dir, pdf_dir,
    ) if partition.retrieved_ids else {}

    return _ResolvedContinuationInputs(
        included_records=included_records, parent_snapshot=parent_snapshot, partition=partition,
        preserved_included_parent_records=preserved_included_parent_records,
        retrieved_manifest_docs=retrieved_manifest_docs,
    )


def _preserved_decision_citations(parent_snapshot, partition: ContinuationPartition) -> List[Dict[str, Any]]:
    """One citation per record this continuation is NOT re-deciding (both
    preserved-included and preserved-excluded) -- `cumulative_snapshot.py`'s
    `_union_ledger_records()` independently re-verifies each against the
    parent bundle's own ledger before trusting it, and raises if a
    retrieved/decided-this-session record is cited here too."""
    preserved_ids = set(partition.preserved_included_ids) | set(partition.preserved_excluded_ids)
    tails = _tails_by_key(
        [vf.model for vf in parent_snapshot.ledger_records if vf.model.stage == "full_text_eligibility"])
    return [
        {"record_id": record_id, "decision_id": rec.decision_id, "content_hash": rec.content_hash}
        for (record_id, _stage), rec in tails.items() if record_id in preserved_ids
    ]


_THREAD_ID_FILENAME = "thread_id.txt"

_DOCLING_OFFLINE_ENV_VARS = ("DOCLING_CACHE_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@contextlib.contextmanager
def _docling_offline_env():
    """Real-data preflight corrective pass (see module docstring): validates
    the shared `DOCLING_CACHE_DIR` asset bundle (`track3_run.py`'s own
    `assert_docling_artifacts_valid()`, same bundle/manifest that launcher
    already relies on) BEFORE mutating any environment variable, then sets
    `DOCLING_CACHE_DIR`/`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` for the
    duration of the `with` block, restoring each variable to whatever it
    was immediately beforehand (including fully unsetting it, if it was
    absent) once the block exits -- by exception or otherwise. Every real
    call site inside this module (`estimate_continuation_cost()` in
    `main()`, the graph-invoke block in `run_continuation()`) parses at
    least one real PDF via Docling; without this, both resolve Docling's
    model cache against whatever the ambient process environment happens
    to be, which may attempt live Hugging Face Hub access instead of the
    offline, digest-verified bundle this repo already ships."""
    assert_docling_artifacts_valid(DOCLING_CACHE_DIR / "models")
    previous = {name: os.environ.get(name) for name in _DOCLING_OFFLINE_ENV_VARS}
    os.environ["DOCLING_CACHE_DIR"] = str(DOCLING_CACHE_DIR.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _open_checkpointer(workspace: Path):
    """Same convention as `track3_run_child.py::_open_checkpointer()`:
    workspace-relative, durable graph checkpoint DB -- a crash-then-restart
    against the same workspace resumes from the last completed node
    instead of silently repeating already-paid stages (unlike the graph's
    default in-memory `MemorySaver`, which loses all state on process
    exit)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(str(workspace / "graph_checkpoints.sqlite"))


_SEED_CONSISTENCY_KEYS = (
    "config",
    "parent_bundle_payload_root",
    "continuation_partition",
    "preserved_included_records",
    "retrieved_pdf_manifest",
    "preserved_decision_citations",
)


def _assert_seed_matches_checkpoint(
    checkpointed_values: Dict[str, Any], newly_resolved_state: Dict[str, Any],
) -> None:
    """Code-review corrective pass, round 2 (finding 3): resuming a
    checkpoint (`existing_state.values` non-empty) previously trusted it
    unconditionally -- `graph.invoke(None, run_config)` ignores the
    freshly-built `state` dict entirely and resumes with whatever the
    ORIGINAL invocation seeded instead. If the parent bundle, config,
    partition, or manifest resolved for THIS invocation differ from that
    original one (e.g. the same `--workspace` reused against a different
    `--parent-bundle-id`, or an edited `--spec`/`--continuation-manifest`),
    the graph would silently keep resuming the unrelated original run
    while this launcher went on to write NEW cost-cap/run-mode artifacts
    derived from the current (unused) inputs -- a real mismatch between
    what actually executed and what gets reported as having executed.
    Raises before any `graph.invoke()` call on a mismatch, so nothing is
    resumed or (re)written."""
    mismatched = [
        key for key in _SEED_CONSISTENCY_KEYS
        if checkpointed_values.get(key) != newly_resolved_state.get(key)
    ]
    if mismatched:
        raise RuntimeError(
            "refusing to resume: this workspace's checkpoint was seeded from different inputs than "
            f"the current invocation resolved -- mismatched field(s): {mismatched!r}. A checkpoint "
            "may only ever be resumed with the exact same parent bundle, configuration, partition, "
            "and manifest it was originally started with; use a fresh --workspace for a different run")


def _build_continuation_seed_state(
    *, cfg: ReviewConfig, parent_bundle: Bundle, resolved: _ResolvedContinuationInputs,
) -> Dict[str, Any]:
    """Build the immutable seed shared by a fresh run and resume checks."""
    state = dict(initial_state(cfg.to_dict()))
    state["included_records"] = [r.to_dict() for r in resolved.included_records]
    state["continuation_partition"] = resolved.partition.model_dump()
    state["preserved_included_records"] = [
        r.to_dict() for r in resolved.preserved_included_parent_records
    ]
    state["retrieved_pdf_manifest"] = resolved.retrieved_manifest_docs
    state["parent_bundle_payload_root"] = str(parent_bundle.payload_root)
    state["preserved_decision_citations"] = _preserved_decision_citations(
        resolved.parent_snapshot, resolved.partition,
    )
    state["prisma_counts"] = resolved.parent_snapshot.prisma.model.model_dump()
    return state


def _load_resolution_checkpoint_state(
    *, cfg: ReviewConfig, parent_bundle: Bundle,
    resolved: _ResolvedContinuationInputs, workspace: Path,
) -> Dict[str, Any]:
    """Read and seed-verify the stopped continuation without invoking it."""
    thread_id_path = workspace / _THREAD_ID_FILENAME
    if not thread_id_path.is_file():
        raise RuntimeError(f"continuation checkpoint thread file is absent: {thread_id_path}")
    thread_id = thread_id_path.read_text(encoding="utf-8").strip()
    if not thread_id:
        raise RuntimeError(f"continuation checkpoint thread file is blank: {thread_id_path}")
    seed = _build_continuation_seed_state(cfg=cfg, parent_bundle=parent_bundle, resolved=resolved)
    with _open_checkpointer(workspace) as checkpointer:
        graph = build_scoping_continuation_graph(checkpointer=checkpointer)
        run_config = {"configurable": {"thread_id": thread_id}}
        checkpoint = graph.get_state(run_config)
        if not checkpoint.values:
            raise RuntimeError("continuation workspace has no checkpoint state to resolve")
        _assert_seed_matches_checkpoint(checkpoint.values, seed)
        return dict(checkpoint.values)


def run_continuation(
    *,
    cfg: ReviewConfig,
    parent_bundle: Bundle,
    resolved: _ResolvedContinuationInputs,
    cap_usd: float,
    workspace: Path,
) -> Dict[str, Any]:
    """Unit G: the real (non-`--estimate-only`) continuation run. Takes the
    SAME resolved partition/PDF inputs `--estimate-only` printed a breakdown
    for (never re-derived a second, possibly different, time), then
    constructs exactly ONE `CostLedger`/`CostCapProxy` pair for the whole
    run and points `OPENAI_BASE_URL` at it before `graph.invoke(...)` --
    covering FT_SCREENING (`continuation_fulltext_node`), extraction, and
    synthesis in a single `graph.invoke()` call, so one proxy instance
    genuinely covers every paid stage, not one per stage.

    Code-review corrective pass (post-95c9667): seeds `state["prisma_counts"]`
    from the parent's own real cumulative PRISMA counts (`continuation_
    fulltext_node()` only ADDS to this dict, so an unseeded run silently
    produced an invalid cumulative report); uses a workspace-persisted
    `thread_id.txt` + a workspace `SqliteSaver` checkpointer instead of a
    fresh random thread_id against the graph's default in-memory
    checkpointer (matching `track3_run_child.py`'s own resume-vs-fresh
    convention, so a crash can actually resume instead of silently risking
    a re-spend); raises instead of returning success when either review
    queue is non-empty or the graph did not reach a true terminal state
    (mirroring `track3_run_child.py`'s own hard-abort checks -- a
    continuation run also requires a fully-automated pass); and writes the
    same terminal-artifact files `publish_continuation_bundle()` requires
    already exist (`terminal_state.json`, `terminal_graph_snapshot.json`,
    `run_mode.json`, `run_mode_statement.md`) -- without these, a real
    launcher run could never actually be published."""
    from slr_pipeline.cost_cap.ledger import CostLedger
    from slr_pipeline.cost_cap.proxy import CostCapProxy

    workspace.mkdir(parents=True, exist_ok=True)
    state = _build_continuation_seed_state(
        cfg=cfg, parent_bundle=parent_bundle, resolved=resolved,
    )

    thread_id_path = workspace / _THREAD_ID_FILENAME
    if thread_id_path.is_file():
        thread_id = thread_id_path.read_text(encoding="utf-8").strip()
    else:
        thread_id = f"track3-continuation-{uuid.uuid4()}"
        thread_id_path.write_text(thread_id, encoding="utf-8")

    ledger = CostLedger(workspace / "cost_cap_ledger.json", cap_usd=cap_usd)
    call_log_path = workspace / "cost_cap_call_log.jsonl"
    proxy = CostCapProxy(ledger, call_log_path, port=0)
    proxy.start()
    previous_base_url = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = proxy.base_url
    try:
        # Real-data preflight corrective pass: the graph invocation below
        # parses real PDFs via Docling (full-text screening/extraction) --
        # see `_docling_offline_env()`'s own docstring for why forcing
        # offline resolution is safe for this launcher specifically.
        with _docling_offline_env(), _open_checkpointer(workspace) as checkpointer:
            graph = build_scoping_continuation_graph(checkpointer=checkpointer)
            run_config = {"configurable": {"thread_id": thread_id}}
            existing_state = graph.get_state(run_config)
            if existing_state.values:
                # Resuming a prior checkpoint for this thread_id -- refuse
                # unless this invocation resolved the exact same seed
                # inputs the checkpoint was originally started with (see
                # `_assert_seed_matches_checkpoint()`'s own docstring),
                # then do not re-invoke already-completed nodes.
                _assert_seed_matches_checkpoint(existing_state.values, state)
                terminal_state = graph.invoke(None, run_config)
            else:
                terminal_state = graph.invoke(state, run_config)
            graph_state = graph.get_state(run_config)
    finally:
        proxy.stop()
        if previous_base_url is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = previous_base_url

    # Finding 1 (code-review corrective pass, round 2): force both files to
    # exist on disk even on a genuinely zero-spend run (e.g. every record
    # preserved, nothing newly retrieved this session) -- CostLedger only
    # persists after a first committed spend, and the call-log file is only
    # ever opened lazily on the first logged call, so a zero-call run would
    # otherwise leave neither file behind, and reference_bundle.py's
    # track3_live run-evidence staging requires both to exist regardless.
    ledger.ensure_persisted()
    call_log_path.touch(exist_ok=True)

    if terminal_state.get("full_text_review_queue"):
        raise RuntimeError(
            "full_text_review_queue is non-empty at graph termination -- a continuation run "
            "requires a fully-automated pass; the checkpoint under this workspace remains valid "
            "for a future resume")
    if terminal_state.get("extraction_review_queue"):
        raise RuntimeError(
            "extraction_review_queue is non-empty at graph termination -- a continuation run "
            "requires a fully-automated pass; the checkpoint under this workspace remains valid "
            "for a future resume")
    next_nodes = list(graph_state.next)
    if next_nodes:
        raise RuntimeError(
            f"graph did not reach a true terminal state -- next={next_nodes!r}; the checkpoint "
            "under this workspace remains valid for a future resume")

    _write_terminal_artifacts(
        cfg=cfg, workspace=workspace, terminal_state=terminal_state, thread_id=thread_id,
        next_nodes=next_nodes, ledger=ledger, proxy_port=proxy.port,
    )

    return terminal_state


def _write_terminal_artifacts(
    *, cfg: ReviewConfig, workspace: Path, terminal_state: Dict[str, Any], thread_id: str,
    next_nodes: list, ledger, proxy_port: int,
) -> None:
    """The §12 files `publish_continuation_bundle()` requires to already
    exist -- `terminal_state.json`/`terminal_graph_snapshot.json`/
    `run_mode.json`/`run_mode_statement.md` -- written exactly the way
    `track3_run_child.py::_invoke_graph()` writes its own, reusing that
    module's `_build_run_mode()` so the real `AIUseEventStore`/extraction-
    checkpoint/FT_SCREENING-run-result-derived call counts a future
    `finalize_bundle()` independently re-verifies can never drift from what
    this launcher itself records."""
    import datetime

    from slr_pipeline.extraction.hashing import file_sha256
    from slr_pipeline.track3_run_child import _build_run_mode

    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps(terminal_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review_run_id = terminal_state.get("review_run_id")
    if not review_run_id:
        raise RuntimeError("terminal_state['review_run_id'] is unset")

    snapshot = {
        "terminal_graph_snapshot_version": "1",
        "thread_id": thread_id,
        "next": next_nodes,
        "terminal_state_sha256": file_sha256(terminal_state_path),
        "review_run_id": review_run_id,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (workspace / "terminal_graph_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_mode = _build_run_mode(cfg, terminal_state, review_run_id=review_run_id, ledger=ledger, port=proxy_port)
    (workspace / "run_mode.json").write_text(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from slr_pipeline.reference_run_mode import render_run_mode_statement_md
    # write_bytes, not write_text -- run_mode_statement.md is byte-compared
    # during acceptance checks; write_text()'s universal-newline translation
    # would silently corrupt it on this Windows host.
    (workspace / "run_mode_statement.md").write_bytes(
        render_run_mode_statement_md(run_mode).encode("utf-8"))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="ReviewConfig-shaped JSON spec file.")
    parser.add_argument("--bundles-root", required=True, type=Path)
    parser.add_argument("--parent-bundle-id", required=True)
    parser.add_argument("--parent-workspace", type=Path, default=None,
                         help="Required if the derived partition has any preserved-included records.")
    parser.add_argument("--continuation-manifest", required=True, type=Path)
    parser.add_argument("--retrieved-pdfs-dir", required=True, type=Path,
                         help="Directory holding this continuation's freshly-downloaded PDFs.")
    parser.add_argument("--cap-usd", type=float, default=None,
                         help="Total dollar spend cap for the real (non-estimate-only) run. Required "
                              "(no default) unless --estimate-only.")
    parser.add_argument("--workspace", required=True, type=Path,
                         help="Directory for cost_cap_ledger.json/cost_cap_call_log.jsonl/graph "
                              "checkpoints/cfg.output_dir (forced to <workspace>/review_output, "
                              "discarding any output_dir the JSON spec declares) -- required even "
                              "for --estimate-only, since estimate_continuation_cost() itself parses "
                              "real PDFs under <workspace>/review_output/pdfs.")
    parser.add_argument("--estimate-only", action="store_true",
                         help="Print a worst-case cost estimate and exit -- no ledger/proxy/graph execution.")
    parser.add_argument(
        "--suggest-extraction-resolutions", action="store_true",
        help="Operate on the stopped extraction-review checkpoint: persist/reopen its review packet, "
             "estimate only the remaining LLM suggestions, and (unless --estimate-only) dispatch them "
             "through this workspace's existing ledger and proxy. Does not resume the graph.",
    )
    parser.add_argument(
        "--retry-suggestion-work-item", action="append", default=[],
        help="Explicitly authorize retry of one prior non-validated suggestion work_item_id. Repeatable.",
    )
    parser.add_argument(
        "--apply-extraction-resolutions", type=Path,
        help="Offline-only: apply a version-1 human decision batch to the stopped extraction checkpoint, "
             "checkpointing each audited decision. Does not invoke/resume the graph or load credentials.",
    )
    args = parser.parse_args(argv)
    if args.retry_suggestion_work_item and not args.suggest_extraction_resolutions:
        parser.error("--retry-suggestion-work-item requires --suggest-extraction-resolutions")
    if args.apply_extraction_resolutions and args.suggest_extraction_resolutions:
        parser.error("--apply-extraction-resolutions and --suggest-extraction-resolutions are separate modes")
    if args.apply_extraction_resolutions and args.estimate_only:
        parser.error("--estimate-only does not apply to the offline decision-application mode")

    cfg = ReviewConfig.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
    cfg = _pin_continuation_synthesis_config(cfg)
    # Finding 4 (code-review corrective pass, post-95c9667): cfg.output_dir
    # is forced under this run's own --workspace -- a reused spec can no
    # longer write into an earlier run's workspace, and pdf_dir is now
    # derived the SAME way `continuation_fulltext_node()` derives it
    # (`cfg.ensure_output_dir() / "pdfs"`), never an independent
    # --pdf-dir flag that could silently diverge from it.
    cfg_dict = cfg.to_dict()
    cfg_dict["output_dir"] = str(args.workspace / "review_output")
    cfg = ReviewConfig.from_dict(cfg_dict)
    pdf_dir = cfg.ensure_output_dir() / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    parent_bundle = resolve_parent_bundle(args.parent_bundle_id, args.bundles_root)

    resolved = _resolve_continuation_inputs(
        parent_bundle=parent_bundle, parent_workspace=args.parent_workspace,
        continuation_manifest_path=args.continuation_manifest,
        retrieved_pdfs_dir=args.retrieved_pdfs_dir, pdf_dir=pdf_dir,
    )
    if args.apply_extraction_resolutions:
        from slr_pipeline.extraction.resolution_runtime import (
            apply_resolution_decision_batch,
            load_resolution_decision_batch,
            prepare_review_packet,
        )

        thread_id_path = args.workspace / _THREAD_ID_FILENAME
        if not thread_id_path.is_file():
            raise RuntimeError(f"continuation checkpoint thread file is absent: {thread_id_path}")
        thread_id = thread_id_path.read_text(encoding="utf-8").strip()
        if not thread_id:
            raise RuntimeError(f"continuation checkpoint thread file is blank: {thread_id_path}")
        seed = _build_continuation_seed_state(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved,
        )
        batch = load_resolution_decision_batch(args.apply_extraction_resolutions)
        with _open_checkpointer(args.workspace) as checkpointer:
            graph = build_scoping_continuation_graph(checkpointer=checkpointer)
            run_config = {"configurable": {"thread_id": thread_id}}
            checkpoint = graph.get_state(run_config)
            if not checkpoint.values:
                raise RuntimeError("continuation workspace has no checkpoint state to resolve")
            _assert_seed_matches_checkpoint(checkpoint.values, seed)
            checkpoint_state = dict(checkpoint.values)
            if checkpoint_state.get("full_text_review_queue"):
                raise RuntimeError("full-text review must be resolved before extraction decisions can apply")
            prepared = prepare_review_packet(checkpoint_state)
            applied = apply_resolution_decision_batch(
                checkpoint_state,
                prepared.packet,
                batch,
                on_state_update=lambda update: graph.update_state(
                    run_config, update, as_node="extraction",
                ),
            )
        print(
            f"offline extraction decisions complete: applied={applied.applied}, "
            f"already_applied={applied.already_applied}, "
            f"remaining_queue={applied.remaining_queue}. "
            "Run the ordinary cap-authorized continuation command to resume the graph."
        )
        return 0

    if args.suggest_extraction_resolutions:
        if not cfg.extraction_model_config:
            parser.error("--suggest-extraction-resolutions requires cfg.extraction_model_config")
        from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
        from slr_pipeline.extraction.resolution_runtime import (
            estimate_suggestion_cost,
            prepare_review_packet,
            run_paid_suggestion_pass,
        )

        model_cfg = OpenAIExtractionConfig(**cfg.extraction_model_config)
        checkpoint_state = _load_resolution_checkpoint_state(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, workspace=args.workspace,
        )
        if checkpoint_state.get("full_text_review_queue"):
            raise RuntimeError("full-text review must be resolved before extraction suggestions can run")
        if not checkpoint_state.get("extraction_review_queue"):
            raise RuntimeError("checkpoint has no extraction_review_queue requiring suggestions")
        prepared = prepare_review_packet(checkpoint_state)
        estimate = estimate_suggestion_cost(
            prepared.packet,
            suggestions_dir=prepared.run_dir / "extraction_resolution_suggestions",
            ledger_path=args.workspace / "cost_cap_ledger.json",
            model=model_cfg.model,
            max_completion_tokens=model_cfg.max_completion_tokens,
            retry_work_item_ids=args.retry_suggestion_work_item,
        )
        print(estimate.render_breakdown())
        if args.estimate_only:
            return 0
        if args.cap_usd is None:
            parser.error("--cap-usd is required for a real suggestion pass")
        if abs(args.cap_usd - estimate.ledger_cap_usd) > 1e-9:
            parser.error(
                f"--cap-usd must equal this workspace's existing ledger cap "
                f"({estimate.ledger_cap_usd:.12g})"
            )
        with credentials.loaded_credentials(("OPENAI_API_KEY",)):
            suggestion_result = run_paid_suggestion_pass(
                checkpoint_state,
                prepared.packet,
                cfg=cfg,
                workspace=args.workspace,
                cap_usd=args.cap_usd,
                model=model_cfg.model,
                model_config_id=model_cfg.model_config_id,
                max_completion_tokens=model_cfg.max_completion_tokens,
                retry_work_item_ids=args.retry_suggestion_work_item,
            )
        dispatched = sum(item.status == "dispatched" for item in suggestion_result.engine.items)
        print(
            f"suggestion pass complete: dispatched={dispatched}, "
            f"remaining_usd=${suggestion_result.remaining_usd:.6f}"
        )
        return 0

    # Real-data preflight corrective pass: `estimate_continuation_cost()`
    # parses real PDFs via Docling -- see `_docling_offline_env()`'s own
    # docstring.
    with _docling_offline_env():
        estimate = estimate_continuation_cost(
            cfg=cfg,
            retrieved_manifest_docs=resolved.retrieved_manifest_docs,
            preserved_included_parent_records=resolved.preserved_included_parent_records,
        )
    print(estimate.render_breakdown())

    if args.estimate_only:
        return 0

    if args.cap_usd is None:
        parser.error("--cap-usd is required for a real (non-estimate-only) run")

    with credentials.loaded_credentials(("OPENAI_API_KEY",)):
        terminal_state = run_continuation(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved,
            cap_usd=args.cap_usd, workspace=args.workspace,
        )
    print(f"continuation run terminal review_run_id={terminal_state.get('review_run_id')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
