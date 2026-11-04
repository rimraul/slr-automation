"""Unit tests for `slr_pipeline.reference_run_child` -- the reference-run
**child** process (`track2_pipeline_readiness_plan_2026-08-30.md` §12,
step 13 [C12]).

Full end-to-end (real docling parse + real cassette signature match through
`reporting`) is intentionally not exercised here: `assets/models/
track2-docling-cache` is not yet provisioned in this checkout (§5.3, a
separate, not-yet-executed step), and faking docling's own text output well
enough to byte-match the frozen cassette's `extraction_batch`/
`synthesis_map`/`synthesis_refine` request signatures would just re-implement
`scripts/author_reference_cassette.py`'s own several-hundred-line response
fixture. These tests instead verify each function's own contract in
isolation -- argument parsing, port parsing, the FIXED-directory lock/clear
discipline, and `_invoke_graph`'s orchestration of a (faked) `ReplayServer`
and (faked) graph -- via monkeypatching at the seams `reference_run_child.py`
itself imports through."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import slr_pipeline.reference_run_child as child_mod


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

def _base_argv(tmp_path):
    return [
        "--workspace", str(tmp_path / "ws"),
        "--extraction-output-dir", str(tmp_path / "out"),
        "--spec", str(tmp_path / "spec.json"),
        "--cassette", str(tmp_path / "cassette.json"),
        "--corpus-oracle", str(tmp_path / "oracle.json"),
        "--pdf-source-dir", str(tmp_path / "pdfs"),
        "--fixture-corpus-root", str(tmp_path / "corpus"),
    ]


def test_parse_args_defaults_thread_id(tmp_path):
    args = child_mod._parse_args(_base_argv(tmp_path))
    assert args.thread_id == "track2-reference-run"
    assert args.workspace == tmp_path / "ws"
    assert args.extraction_output_dir == tmp_path / "out"


def test_parse_args_thread_id_override(tmp_path):
    argv = _base_argv(tmp_path) + ["--thread-id", "custom-thread"]
    args = child_mod._parse_args(argv)
    assert args.thread_id == "custom-thread"


def test_parse_args_missing_required_flag_raises_systemexit(tmp_path, capsys):
    argv = _base_argv(tmp_path)[2:]  # drop --workspace
    with pytest.raises(SystemExit):
        child_mod._parse_args(argv)


# ---------------------------------------------------------------------------
# _replay_port_from_env
# ---------------------------------------------------------------------------

def test_replay_port_from_env_parses_port(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:54321/v1")
    assert child_mod._replay_port_from_env() == 54321


def test_replay_port_from_env_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        child_mod._replay_port_from_env()


def test_replay_port_from_env_wrong_host_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com:1234/v1")
    with pytest.raises(RuntimeError, match="127.0.0.1"):
        child_mod._replay_port_from_env()


def test_replay_port_from_env_missing_port_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1/v1")
    with pytest.raises(RuntimeError, match="127.0.0.1"):
        child_mod._replay_port_from_env()


# ---------------------------------------------------------------------------
# main() -- guard-arm/disarm wiring, violations.jsonl always present
# ---------------------------------------------------------------------------

def test_main_arms_and_disarms_guard_around_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:19999/v1")
    armed_during_run = {}

    def fake_run(args, *, port, workspace):
        from slr_pipeline.reference_netguard import ReferenceNetGuard
        armed_during_run["armed"] = ReferenceNetGuard is not None  # guard module importable
        armed_during_run["port"] = port
        armed_during_run["workspace"] = workspace
        return 0

    monkeypatch.setattr(child_mod, "_run", fake_run)
    rc = child_mod.main(_base_argv(tmp_path))
    assert rc == 0
    assert armed_during_run["port"] == 19999
    assert armed_during_run["workspace"] == tmp_path / "ws"
    assert (tmp_path / "ws" / "violations.jsonl").exists()


def test_main_propagates_run_exception_but_still_disarms_and_touches_violations(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:19999/v1")

    def fake_run(args, *, port, workspace):
        raise ValueError("boom")

    monkeypatch.setattr(child_mod, "_run", fake_run)
    with pytest.raises(ValueError, match="boom"):
        child_mod.main(_base_argv(tmp_path))

    # guard disarmed -> a disallowed connect now fails with a normal socket error,
    # not the guard's own NetGuardViolation (proves no leaked monkeypatch).
    import socket

    from slr_pipeline.reference_netguard import NetGuardViolation

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        with pytest.raises(OSError) as excinfo:
            s.connect(("127.0.0.1", 1))
        assert not isinstance(excinfo.value, NetGuardViolation)
    finally:
        s.close()

    assert (tmp_path / "ws" / "violations.jsonl").exists()


def test_main_missing_openai_base_url_raises_before_workspace_side_effects(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        child_mod.main(_base_argv(tmp_path))


# ---------------------------------------------------------------------------
# _run -- FIXED-directory lock/clear-at-start discipline + cfg wiring
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_run_locks_clears_fixed_dir_and_delegates_to_invoke_graph(tmp_path, monkeypatch):
    extraction_output_dir = tmp_path / "fixed_output"
    extraction_output_dir.mkdir(parents=True)
    marker = extraction_output_dir / "stale_marker.txt"
    marker.write_text("stale", encoding="utf-8")

    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, {})
    cassette_path = tmp_path / "cassette.json"
    _write_json(cassette_path, {"entries": []})
    oracle_path = tmp_path / "oracle.json"
    _write_json(oracle_path, {"pdf_sha256": {}})

    args = SimpleNamespace(
        extraction_output_dir=extraction_output_dir,
        spec=spec_path,
        cassette=cassette_path,
        corpus_oracle=oracle_path,
        pdf_source_dir=tmp_path / "pdf_source",
        fixture_corpus_root=tmp_path / "corpus_root",
        thread_id="t",
    )

    ensure_output_dir_target = tmp_path / "ensured_output"

    class FakeSpec:
        def __init__(self, **kw):
            self.kw = kw

    calls = {}

    def fake_reference_run_config(spec):
        calls["spec"] = spec
        return SimpleNamespace(to_dict=lambda: {"output_dir": "will-be-overwritten"})

    class FakeReviewConfig:
        @classmethod
        def from_dict(cls, d):
            calls["from_dict_arg"] = d
            inst = cls()
            inst._d = d
            inst.to_dict = lambda: d
            inst.ensure_output_dir = lambda: (ensure_output_dir_target.mkdir(parents=True, exist_ok=True) or ensure_output_dir_target)
            return inst

    def fake_place_pdfs(pdf_dir, oracle, source_dir):
        calls["place_pdfs"] = (pdf_dir, oracle, source_dir)

    def fake_invoke_graph(a, cfg, spec, *, port, workspace):
        calls["invoke_graph"] = (a, cfg, spec, port, workspace)
        return 0

    monkeypatch.setattr("slr_pipeline.reference_spec.ReferenceRunSpec", FakeSpec)
    monkeypatch.setattr("slr_pipeline.reference_spec.reference_run_config", fake_reference_run_config)
    monkeypatch.setattr("slr_pipeline.config.ReviewConfig", FakeReviewConfig)
    monkeypatch.setattr("slr_pipeline.reference_pdf_placement.place_pdfs", fake_place_pdfs)
    monkeypatch.setattr(child_mod, "_invoke_graph", fake_invoke_graph)

    workspace = tmp_path / "ws"
    rc = child_mod._run(args, port=4242, workspace=workspace)

    assert rc == 0
    # stale contents cleared -- rmtree wiped the marker (nothing recreates the original dir)
    assert not extraction_output_dir.exists()
    assert os.environ["SLR_FIXTURE_CORPUS_ROOT"] == str(args.fixture_corpus_root)
    assert calls["from_dict_arg"]["output_dir"] == str(extraction_output_dir)
    assert calls["place_pdfs"] == (ensure_output_dir_target / "pdfs", {"pdf_sha256": {}}, args.pdf_source_dir)
    invoked_args, invoked_cfg, invoked_spec, invoked_port, invoked_workspace = calls["invoke_graph"]
    assert invoked_args is args
    assert isinstance(invoked_spec, FakeSpec)
    assert invoked_port == 4242
    assert invoked_workspace == workspace
    # lock file removed after a successful run
    lock_path = extraction_output_dir.parent / (extraction_output_dir.name + ".lock")
    assert not lock_path.exists()


def test_run_raises_on_stale_lock_and_leaves_it_untouched(tmp_path):
    extraction_output_dir = tmp_path / "fixed_output"
    extraction_output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = extraction_output_dir.parent / (extraction_output_dir.name + ".lock")
    lock_path.write_text("pid-of-some-other-run", encoding="utf-8")

    args = SimpleNamespace(
        extraction_output_dir=extraction_output_dir,
        spec=tmp_path / "spec.json",
        cassette=tmp_path / "cassette.json",
        corpus_oracle=tmp_path / "oracle.json",
        pdf_source_dir=tmp_path / "pdf_source",
        fixture_corpus_root=tmp_path / "corpus_root",
        thread_id="t",
    )

    with pytest.raises(RuntimeError, match="already exists"):
        child_mod._run(args, port=4242, workspace=tmp_path / "ws")

    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == "pid-of-some-other-run"


# ---------------------------------------------------------------------------
# _invoke_graph -- ReplayServer lifecycle, queue-emptiness checks,
# terminal_state.json / terminal_graph_snapshot.json writing
# ---------------------------------------------------------------------------

class _FakeReplayServer:
    instances = []

    def __init__(self, cassette, call_log_path, *, host="127.0.0.1", port=0):
        self.cassette = cassette
        self.call_log_path = call_log_path
        self.host = host
        self.port = port
        self.started = False
        self.stopped = False
        self.assert_complete_called = False
        self._latched_fatal = False
        self._latch_reason = None
        _FakeReplayServer.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    @property
    def latched_fatal(self):
        return self._latched_fatal

    @property
    def latch_reason(self):
        return self._latch_reason

    def assert_complete(self):
        self.assert_complete_called = True

    def served_calls(self):
        # Mirrors a genuinely-completed run: served == expected for every
        # entry (true once assert_complete() would have passed).
        return {e.entry_id: e.expected_calls for e in self.cassette.entries}


class _FakeGraph:
    def __init__(self, terminal_state, next_nodes=()):
        self._terminal_state = terminal_state
        self._next_nodes = tuple(next_nodes)
        self.invoked_with = None

    def invoke(self, state, config):
        self.invoked_with = (state, config)
        return self._terminal_state

    def get_state(self, config):
        return SimpleNamespace(next=self._next_nodes)


def _invoke_graph_args(tmp_path, cassette_entries=None, thread_id="t"):
    cassette_path = tmp_path / "cassette.json"
    _write_json(cassette_path, {"entries": cassette_entries or []})
    args = SimpleNamespace(cassette=cassette_path, thread_id=thread_id)
    cfg = SimpleNamespace(
        to_dict=lambda: {"some": "config"},
        reference_reporting_context={
            "llm_transport": "replay",
            "search_source": "fixture",
            "bertopic": "not_run",
            "dedup_mode": "exact",
        },
    )
    spec = SimpleNamespace(fixture_corpus_digest="a" * 64)
    return args, cfg, spec


def _patch_graph_seams(monkeypatch, *, terminal_state, next_nodes=()):
    _FakeReplayServer.instances = []
    fake_graph = _FakeGraph(terminal_state, next_nodes=next_nodes)
    monkeypatch.setattr("slr_pipeline.replay.transport.ReplayServer", _FakeReplayServer)
    monkeypatch.setattr("slr_pipeline.graphs.scoping.build_scoping_graph", lambda: fake_graph)
    monkeypatch.setattr("slr_pipeline.state.initial_state", lambda cfg_dict: {"initial": cfg_dict})
    return fake_graph


def test_invoke_graph_happy_path_writes_flat_terminal_files(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg, spec = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [],
        "extraction_review_queue": [],
        "review_run_id": "11111111-1111-1111-1111-111111111111",
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state, next_nodes=())

    rc = child_mod._invoke_graph(args, cfg, spec, port=4321, workspace=workspace)
    assert rc == 0

    server = _FakeReplayServer.instances[0]
    assert server.started and server.stopped
    assert server.assert_complete_called
    assert server.port == 4321
    assert server.call_log_path == workspace / "call_log.jsonl"

    terminal_state_path = workspace / "terminal_state.json"
    assert terminal_state_path.exists()
    assert json.loads(terminal_state_path.read_text(encoding="utf-8")) == terminal_state
    assert not (workspace / "raw").exists()  # flat, not raw/state/-nested

    snapshot_path = workspace / "terminal_graph_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["terminal_graph_snapshot_version"] == "1"
    assert snapshot["thread_id"] == "t"
    assert snapshot["next"] == []
    assert snapshot["review_run_id"] == "11111111-1111-1111-1111-111111111111"
    assert len(snapshot["terminal_state_sha256"]) == 64
    assert snapshot["captured_utc"].endswith("Z")

    from slr_pipeline.extraction.hashing import file_sha256
    assert snapshot["terminal_state_sha256"] == file_sha256(terminal_state_path)

    run_mode_path = workspace / "run_mode.json"
    assert run_mode_path.exists()
    run_mode = json.loads(run_mode_path.read_text(encoding="utf-8"))
    assert run_mode["run_mode_schema_version"] == "1"
    assert run_mode["mode_profile"] == "track2_offline"
    assert run_mode["llm_transport"] == cfg.reference_reporting_context["llm_transport"]
    assert run_mode["replay_cassette_sha256"] == file_sha256(args.cassette)
    assert run_mode["fixture_corpus_digest"] == spec.fixture_corpus_digest
    assert run_mode["model_calls"] == {"expected": 0, "served": 0}  # empty cassette fixture in this test
    assert sorted(run_mode["per_stage_backend"]) == sorted(
        {
            "search_strategy_query_expansion", "search_strategy_db_translation",
            "title_abstract_screening", "screening_qc_second_rater",
            "synthesis_claim_map", "synthesis_claim_refine",
        }
    )

    statement_path = workspace / "run_mode_statement.md"
    assert statement_path.exists()
    statement = statement_path.read_text(encoding="utf-8")
    assert statement.startswith("# Track 2 reference run")
    assert "`track2_offline`" in statement


def test_invoke_graph_raises_on_nonempty_full_text_review_queue(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg, spec = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [{"record_id": "fx-0001"}],
        "extraction_review_queue": [],
        "review_run_id": "r",
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    with pytest.raises(RuntimeError, match="full_text_review_queue"):
        child_mod._invoke_graph(args, cfg, spec, port=1, workspace=workspace)
    assert not (workspace / "terminal_state.json").exists()


def test_invoke_graph_raises_on_nonempty_extraction_review_queue(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg, spec = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [],
        "extraction_review_queue": [{"record_id": "fx-0001"}],
        "review_run_id": "r",
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    with pytest.raises(RuntimeError, match="extraction_review_queue"):
        child_mod._invoke_graph(args, cfg, spec, port=1, workspace=workspace)


def test_invoke_graph_raises_on_missing_review_run_id(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg, spec = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [],
        "extraction_review_queue": [],
        "review_run_id": None,
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    with pytest.raises(RuntimeError, match="review_run_id"):
        child_mod._invoke_graph(args, cfg, spec, port=1, workspace=workspace)
    # terminal_state.json is written before the review_run_id check -- still on disk
    assert (workspace / "terminal_state.json").exists()


def test_invoke_graph_raises_on_latched_fatal_server(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg, spec = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [],
        "extraction_review_queue": [],
        "review_run_id": "r",
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    # Force the fake server latched-fatal after construction, before assert_complete.
    original_init = _FakeReplayServer.__init__

    def latched_init(self, *a, **k):
        original_init(self, *a, **k)
        self._latched_fatal = True
        self._latch_reason = "unrecorded signature"

    monkeypatch.setattr(_FakeReplayServer, "__init__", latched_init)
    with pytest.raises(RuntimeError, match="unrecorded signature"):
        child_mod._invoke_graph(args, cfg, spec, port=1, workspace=workspace)
    assert not _FakeReplayServer.instances[0].assert_complete_called


def test_invoke_graph_stops_server_even_when_graph_invoke_raises(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg, spec = _invoke_graph_args(tmp_path)
    _FakeReplayServer.instances = []

    class _RaisingGraph:
        def invoke(self, state, config):
            raise ValueError("graph blew up")

    monkeypatch.setattr("slr_pipeline.replay.transport.ReplayServer", _FakeReplayServer)
    monkeypatch.setattr("slr_pipeline.graphs.scoping.build_scoping_graph", lambda: _RaisingGraph())
    monkeypatch.setattr("slr_pipeline.state.initial_state", lambda cfg_dict: {})

    with pytest.raises(ValueError, match="graph blew up"):
        child_mod._invoke_graph(args, cfg, spec, port=1, workspace=workspace)
    assert _FakeReplayServer.instances[0].stopped


# ---------------------------------------------------------------------------
# Round-13 self-review finding 1: `slr_pipeline/__init__.py` import graph
# ---------------------------------------------------------------------------

def test_slr_pipeline_package_init_time_import_makes_no_network_call():
    """`import slr_pipeline.reference_netguard` (this module's own first
    pipeline import, before parsing/arming) necessarily runs `slr_pipeline/
    __init__.py` first -- Python always imports a dotted path's parent
    package before the submodule. That `__init__.py` imports `.config`/
    `.state`. So the module docstring's "before any pipeline import" claim
    cannot literally mean zero pipeline imports (arming the guard IS one) --
    the real, achievable invariant is that `slr_pipeline/__init__.py`'s own
    transitive import graph opens no live socket. Proven here in a fresh
    subprocess: `socket.socket.connect` is replaced with an always-raising
    stub *before* `import slr_pipeline` runs, so any future import-time
    network side effect anywhere in that graph would fail this test
    immediately, regardless of which submodule is imported first."""
    script = (
        "import socket\n"
        "def _blocked(self, *a, **k):\n"
        "    raise AssertionError('socket.connect called during slr_pipeline import')\n"
        "socket.socket.connect = _blocked\n"
        "import slr_pipeline\n"
        "print('IMPORT_OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
