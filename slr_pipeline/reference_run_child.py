"""Track 2 plan Step 13 [C12]: the reference-run **child** process.

Invoked by `reference_run.py` (the parent driver) as a **subprocess**, with
the fresh env `reference_run.py` builds per plan §12.3 -- notably
`OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` and
`OPENAI_API_KEY=sk-replay-not-a-real-key`, both already set in this
process's environment before Python even starts. This module's own first
statements -- before any pipeline-internal LOGIC runs, and before any
third-party import -- parse that port out of `OPENAI_BASE_URL` (pure stdlib
`urllib.parse`, no network I/O) and arm
`slr_pipeline.reference_netguard.ReferenceNetGuard` on exactly that
`(host, port)` pair, so no code that runs afterwards can ever open a socket
to anything except the replay server this same process starts on that same,
already-committed port. The parent -- not this module -- decided the port
number and is the one with the small, accepted bind/close-then-rebind TOCTOU
race (§12.3's own env is authored before this process is even spawned).

Round-13 self-review finding 1: `import slr_pipeline.reference_netguard`
itself necessarily runs `slr_pipeline/__init__.py` first (Python always
imports a dotted path's parent package before the submodule -- true whether
invoked via `python -m slr_pipeline.reference_run_child` or any other way),
and that `__init__.py` imports `.config`/`.state`. So the real, achievable
invariant is not "zero pipeline imports before `guard.arm()`" (impossible --
arming the guard IS a pipeline import) but "`slr_pipeline/__init__.py`'s own
import graph opens no live socket" -- verified by
`test_reference_run_child.py::test_slr_pipeline_package_init_time_import_makes_no_network_call`,
which fresh-subprocess-imports bare `slr_pipeline` with `socket.socket.connect`
replaced by an always-raising stub installed before that import, so any
future import-time side effect anywhere in that transitive graph would fail
the test immediately, not just this module's own after-arm code.

Responsibilities, in order:
  1. Parse the port from `OPENAI_BASE_URL` and arm the net guard (before
     importing anything else).
  2. Read the already-committed, frozen `reference_spec.json` and
     `replay_cassette.json` off disk (paths supplied by the parent via CLI
     args) and build a `ReviewConfig` from the spec (`reference_run_config`).
  3. §6.3: hash-verified PDF pre-positioning into `cfg.output_dir/"pdfs"`,
     via the shared `slr_pipeline.reference_pdf_placement.place_pdfs`.
     `cfg.output_dir` (`--extraction-output-dir`) is deliberately **not**
     the harness workspace `reference_run.py` allocated for this run's own
     terminal artifacts -- see `reference_pdf_placement.
     REFERENCE_EXTRACTION_WORKSPACE_PARTS`'s own comment: it must be the
     exact FIXED path `author_reference_cassette.py` used to author the
     cassette, or the `pdf_path`-derived `extraction_run_id`/`evidence_id`/
     `answer_id` strings this run produces will not byte-match the ones
     already frozen into the cassette's synthesis_map/synthesis_refine
     entries, and replay will latch-fatal on "unrecorded signature".
  4. Start the real `slr_pipeline.replay.transport.ReplayServer` bound to
     the exact port already named in `OPENAI_BASE_URL`.
  5. `graph = build_scoping_graph(); state = initial_state(cfg.to_dict());
     graph.invoke(state, config)`. No `interrupt()` exists anywhere in this
     graph (`graphs/scoping.py`), so a single `invoke()` call runs the whole
     pipeline to completion, provided `full_text_review_queue`/
     `extraction_review_queue` both stay empty on this fixture corpus (true
     by construction -- see `CORPUS_ORACLE.json`).
  6. `graph.get_state(config)` -> the §10.9 `terminal_graph_snapshot.json`.
  7. `ReplayServer.assert_complete()` -- teardown verification that every
     cassette entry's `served_calls == expected_calls`.
  8. Write `terminal_state.json` / `terminal_graph_snapshot.json` /
     `call_log.jsonl` (via the server) / `violations.jsonl` (via the guard,
     empty file if never triggered) / `run_mode.json` / `run_mode_statement.md`
     (via `slr_pipeline.reference_run_mode`, §11.3) as **flat** files at the
     workspace root (plan §13 step 12's own bullet, verbatim -- the latter
     two files were not part of the original step-12 bullet list but are a
     necessary addition: `finalize_bundle()` (step 15, already committed)
     already requires both to exist via `_collect_run_mode()`). The
     `raw/state/...`- and `raw/run_evidence/...`-prefixed paths §10.2's
     raw-layer table shows are the PAYLOAD's own bundle-relative layout --
     step 15's `finalize_bundle()` relocates these flat workspace files
     there; this step's job ends once they are written.

`finalize_bundle()` is explicitly **not** called here (plan §13 step 12 note
-- "it needs the manifests + publication machinery of step 15")."""
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
                              "/ call_log.jsonl / violations.jsonl (plan §12.4).")
    parser.add_argument("--extraction-output-dir", required=True, type=Path,
                         help="cfg.output_dir -- MUST be the FIXED path "
                              "reference_pdf_placement.REFERENCE_EXTRACTION_WORKSPACE_PARTS resolves to "
                              "on this checkout, or cassette replay will not signature-match.")
    parser.add_argument("--spec", required=True, type=Path, help="reference_spec.json")
    parser.add_argument("--cassette", required=True, type=Path, help="replay_cassette.json")
    parser.add_argument("--corpus-oracle", required=True, type=Path, help="CORPUS_ORACLE.json")
    parser.add_argument("--pdf-source-dir", required=True, type=Path,
                         help="tests/reference_run/fixtures/corpus/pdfs")
    parser.add_argument("--fixture-corpus-root", required=True, type=Path,
                         help="SLR_FIXTURE_CORPUS_ROOT value the search adapters read -- set by "
                              "this process itself (not part of §12.3's parent-authored env dict).")
    parser.add_argument("--thread-id", default="track2-reference-run")
    return parser.parse_args(argv)


def _replay_port_from_env() -> int:
    """Pure stdlib, no network I/O -- safe to call before the net guard
    exists. `reference_run.py` sets `OPENAI_BASE_URL` to
    `http://127.0.0.1:<port>/v1` in this process's own environment before
    spawning it (plan §12.3); this parses that same port back out so the
    guard and the replay server both bind to the one port the parent
    already committed to."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL must be set in the child environment (plan §12.3)")
    parsed = urlparse(base_url)
    if parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise RuntimeError(f"OPENAI_BASE_URL must be http://127.0.0.1:<port>/v1, got {base_url!r}")
    return parsed.port


def main(argv: list) -> int:
    args = _parse_args(argv)
    workspace: Path = args.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    violations_path = workspace / "violations.jsonl"

    # ---- Step 1: net guard, before ANY pipeline/third-party import ----
    port = _replay_port_from_env()
    from slr_pipeline.reference_netguard import ReferenceNetGuard  # noqa: E402 -- stdlib-only module

    guard = ReferenceNetGuard(("127.0.0.1", port), violations_path)
    guard.arm()
    try:
        return _run(args, port=port, workspace=workspace)
    finally:
        guard.disarm()
        if not violations_path.exists():
            # plan §11.2: "violations.jsonl non-empty -> run failure" implies its absence/
            # emptiness is itself a checked artifact -- write an empty file so a downstream
            # checker never has to distinguish "never armed" from "armed, never violated".
            violations_path.touch()


def _run(args: argparse.Namespace, *, port: int, workspace: Path) -> int:
    # ---- Steps 2-3: everything pipeline/third-party-dependent, only now imported ----
    from slr_pipeline.config import ReviewConfig
    from slr_pipeline.reference_pdf_placement import place_pdfs
    from slr_pipeline.reference_spec import ReferenceRunSpec, reference_run_config

    # SLR_FIXTURE_CORPUS_ROOT is not part of §12.3's parent-authored env dict (it is a
    # fixture-only test affordance, not a real-run credential) -- this process sets it for
    # itself, from its own CLI arg, before the search adapters can read it.
    os.environ["SLR_FIXTURE_CORPUS_ROOT"] = str(args.fixture_corpus_root)

    # `args.extraction_output_dir` is FIXED (not the per-run harness `workspace`) -- see
    # module docstring step 3. A stale prior run's checkpoints/ft_screening_runs there could
    # silently be resumed/reused instead of genuinely replayed, so it is cleared here, at the
    # START of every run, under an O_CREAT|O_EXCL lock (same discipline as
    # author_reference_cassette.py's own FIXED_WORKSPACE) -- guards against two concurrent
    # reference runs racing on the one shared fixed directory. Unlike that script, this
    # directory is **not** deleted again at the end: its contents (extraction checkpoints,
    # ft_screening_runs, selection_decisions, ...) are load-bearing raw-layer artifacts step
    # 15's finalize_bundle() reads later, not throwaway generation scratch.
    extraction_output_dir = args.extraction_output_dir
    extraction_output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = extraction_output_dir.parent / (extraction_output_dir.name + ".lock")
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{lock_path} already exists -- another reference_run_child.py run is in progress "
            f"against this FIXED extraction-output-dir (or a previous run crashed without "
            f"cleaning up); remove it if stale"
        )
    os.close(lock_fd)
    try:
        import shutil

        shutil.rmtree(extraction_output_dir, ignore_errors=True)

        spec = ReferenceRunSpec(**json.loads(args.spec.read_text(encoding="utf-8")))
        cfg = reference_run_config(spec)
        cfg_dict = cfg.to_dict()
        cfg_dict["output_dir"] = str(extraction_output_dir)
        cfg = ReviewConfig.from_dict(cfg_dict)
        output_dir = cfg.ensure_output_dir()

        oracle = json.loads(args.corpus_oracle.read_text(encoding="utf-8"))
        place_pdfs(output_dir / "pdfs", oracle, args.pdf_source_dir)
        return _invoke_graph(args, cfg, spec, port=port, workspace=workspace)
    finally:
        lock_path.unlink(missing_ok=True)


def _invoke_graph(args: argparse.Namespace, cfg, spec, *, port: int, workspace: Path) -> int:
    from slr_pipeline.extraction.hashing import file_sha256
    from slr_pipeline.graphs.scoping import build_scoping_graph
    from slr_pipeline.reference_run_mode import build_run_mode_document, render_run_mode_statement_md
    from slr_pipeline.replay.cassette import ReplayCassette
    from slr_pipeline.replay.transport import ReplayServer
    from slr_pipeline.state import initial_state

    cassette = ReplayCassette(**json.loads(args.cassette.read_text(encoding="utf-8")))
    call_log_path = workspace / "call_log.jsonl"

    server = ReplayServer(cassette, call_log_path, port=port)
    server.start()
    try:
        graph = build_scoping_graph()
        state = initial_state(cfg.to_dict())
        run_config = {"configurable": {"thread_id": args.thread_id}}
        terminal_state = graph.invoke(state, run_config)
        graph_state = graph.get_state(run_config)
    finally:
        server.stop()

    if server.latched_fatal:
        raise RuntimeError(f"replay server latched fatal: {server.latch_reason}")
    server.assert_complete()

    if terminal_state.get("full_text_review_queue"):
        raise RuntimeError("full_text_review_queue is non-empty at graph termination -- "
                            "reference run requires a fully-automated pass on this fixture corpus")
    if terminal_state.get("extraction_review_queue"):
        raise RuntimeError("extraction_review_queue is non-empty at graph termination -- "
                            "reference run requires a fully-automated pass on this fixture corpus")

    # §10.9: "next MUST be exactly [] for a terminal run" -- checked here, not
    # just assumed from "this graph has no interrupt()" (round-13 self-review
    # finding 3: that invariant lives in graphs/scoping.py, outside this
    # file's own control, and a future interrupt point there would otherwise
    # exit 0 with a snapshot silently violating the §10.9 contract).
    next_nodes = list(graph_state.next)
    if next_nodes:
        raise RuntimeError(
            f"graph did not reach a true terminal state -- next={next_nodes!r} "
            f"(plan §10.9 requires exactly [] for a terminal run)"
        )

    terminal_state_path = workspace / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps(terminal_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    review_run_id = terminal_state.get("review_run_id")
    if not review_run_id:
        # fulltext_node (slr_pipeline.run_identity.resolve_review_run_id()) sets this
        # unconditionally once it runs, which it always does on this fixture corpus
        # (full_text_eligible is non-empty -- see CORPUS_ORACLE.json). A missing value here
        # means the graph did not actually reach fulltext, not a value worth papering over
        # with a freshly-generated uuid -- that would silently break run-to-run reproducibility
        # (plan §11.2 "two independent runs project equal") rather than surface the real defect.
        raise RuntimeError("terminal_state['review_run_id'] is unset -- fulltext_node did not run "
                            "as expected on this fixture corpus")

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

    run_mode = build_run_mode_document(
        cassette=cassette,
        served_calls=server.served_calls(),
        replay_cassette_sha256=file_sha256(args.cassette),
        fixture_corpus_digest=spec.fixture_corpus_digest,
        reporting_context=cfg.reference_reporting_context,
        port=port,
    )
    (workspace / "run_mode.json").write_text(
        json.dumps(run_mode.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (workspace / "run_mode_statement.md").write_text(
        render_run_mode_statement_md(run_mode), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
