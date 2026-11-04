"""Unit tests for `slr_pipeline.track3_run_child` -- Track 3 live-run
**child** process (`track3_live_run_plan_2026-09-06.md`, Unit C, plan
`squishy-petting-starlight.md`). Mirrors `tests/test_reference_run_child.py`'s
own seam-level monkeypatching discipline: `graph.invoke()`, `CostCapProxy`,
`_open_checkpointer`, and `_build_run_mode` are faked -- no real network
call, no real Docling asset, and no real OpenAI/PubMed/Scopus transport is
ever exercised. One exception, by design: the kill/restart resume test uses
a real (offline, file-local) `SqliteSaver` against a throwaway `StateGraph`
to empirically prove completed nodes are not re-invoked after a simulated
crash+restart."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import slr_pipeline.track3_run_child as child_mod


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

def _base_argv(tmp_path):
    return [
        "--workspace", str(tmp_path / "ws"),
        "--spec", str(tmp_path / "spec.json"),
        "--output-dir", str(tmp_path / "out"),
        "--cost-cap-path", str(tmp_path / "ledger.json"),
        "--cap-usd", "30.0",
    ]


def test_parse_args_defaults_thread_id(tmp_path):
    args = child_mod._parse_args(_base_argv(tmp_path))
    assert args.thread_id == "track3-live-run"
    assert args.workspace == tmp_path / "ws"
    assert args.output_dir == tmp_path / "out"
    assert args.cap_usd == 30.0


def test_parse_args_thread_id_override(tmp_path):
    argv = _base_argv(tmp_path) + ["--thread-id", "custom-thread"]
    args = child_mod._parse_args(argv)
    assert args.thread_id == "custom-thread"


def test_parse_args_missing_required_flag_raises_systemexit(tmp_path):
    argv = _base_argv(tmp_path)[2:]  # drop --workspace
    with pytest.raises(SystemExit):
        child_mod._parse_args(argv)


# ---------------------------------------------------------------------------
# _proxy_port_from_env
# ---------------------------------------------------------------------------

def test_proxy_port_from_env_parses_port(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:54321/v1")
    assert child_mod._proxy_port_from_env() == 54321


def test_proxy_port_from_env_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        child_mod._proxy_port_from_env()


def test_proxy_port_from_env_wrong_host_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com:1234/v1")
    with pytest.raises(RuntimeError, match="127.0.0.1"):
        child_mod._proxy_port_from_env()


def test_proxy_port_from_env_missing_port_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1/v1")
    with pytest.raises(RuntimeError, match="127.0.0.1"):
        child_mod._proxy_port_from_env()


# ---------------------------------------------------------------------------
# main() -- port parsing + workspace creation + delegation to _run
# ---------------------------------------------------------------------------

def test_main_creates_workspace_and_delegates_to_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:19999/v1")
    captured = {}

    def fake_run(args, *, port, workspace):
        captured["port"] = port
        captured["workspace"] = workspace
        return 0

    monkeypatch.setattr(child_mod, "_run", fake_run)
    rc = child_mod.main(_base_argv(tmp_path))
    assert rc == 0
    assert captured["port"] == 19999
    assert captured["workspace"] == tmp_path / "ws"
    assert (tmp_path / "ws").is_dir()


def test_main_missing_openai_base_url_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        child_mod.main(_base_argv(tmp_path))


# ---------------------------------------------------------------------------
# _run -- defense-in-depth preflight re-checks (C2/dedup/credentials),
# independent of whatever the parent already checked; cfg output_dir wiring
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _spec_dict(databases=("pubmed", "scopus"), dedup_mode="exact", execution_adapter_map=None):
    search = {"databases": list(databases)}
    if execution_adapter_map is not None:
        search["execution_adapter_map"] = execution_adapter_map
    return {"search": search, "dedup": {"mode": dedup_mode}}


def test_run_rechecks_wos_allowlist_independently(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, _spec_dict(databases=("pubmed", "wos")))
    args = SimpleNamespace(spec=spec_path, output_dir=tmp_path / "out",
                            cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id="t")

    def fail_if_called(*a, **k):
        raise AssertionError("_invoke_graph must never be called for a WoS-containing config")

    monkeypatch.setattr(child_mod, "_invoke_graph", fail_if_called)
    from slr_pipeline.track3_run import Track3ConfigError
    with pytest.raises(Track3ConfigError, match="wos|WoS"):
        child_mod._run(args, port=1, workspace=tmp_path / "ws")


def test_run_rechecks_execution_adapter_map_independently(tmp_path, monkeypatch):
    """Regression: the child must independently re-reject a fixture-
    redirecting execution_adapter_map too, not just re-check WoS -- defense
    in depth (C2), same as the parent's own assert_no_execution_adapter_map."""
    spec_path = tmp_path / "spec.json"
    _write_json(
        spec_path,
        _spec_dict(execution_adapter_map={"pubmed": "fx_pubmed", "scopus": "fx_scopus"}),
    )
    args = SimpleNamespace(spec=spec_path, output_dir=tmp_path / "out",
                            cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id="t")

    def fail_if_called(*a, **k):
        raise AssertionError("_invoke_graph must never be called for an execution_adapter_map override")

    monkeypatch.setattr(child_mod, "_invoke_graph", fail_if_called)
    from slr_pipeline.track3_run import Track3ConfigError
    with pytest.raises(Track3ConfigError, match="execution_adapter_map"):
        child_mod._run(args, port=1, workspace=tmp_path / "ws")


def test_run_rechecks_dedup_mode_independently(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, _spec_dict(dedup_mode="semantic"))
    args = SimpleNamespace(spec=spec_path, output_dir=tmp_path / "out",
                            cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id="t")

    def fail_if_called(*a, **k):
        raise AssertionError("_invoke_graph must never be called for a non-exact dedup mode")

    monkeypatch.setattr(child_mod, "_invoke_graph", fail_if_called)
    from slr_pipeline.track3_run import Track3ConfigError
    with pytest.raises(Track3ConfigError, match="exact"):
        child_mod._run(args, port=1, workspace=tmp_path / "ws")


def test_run_rechecks_scopus_credential_independently(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, _spec_dict(databases=("scopus",)))
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    args = SimpleNamespace(spec=spec_path, output_dir=tmp_path / "out",
                            cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id="t")

    def fail_if_called(*a, **k):
        raise AssertionError("_invoke_graph must never be called without SCOPUS_API_KEY")

    monkeypatch.setattr(child_mod, "_invoke_graph", fail_if_called)
    from slr_pipeline.track3_run import Track3ConfigError
    with pytest.raises(Track3ConfigError, match="SCOPUS_API_KEY"):
        child_mod._run(args, port=1, workspace=tmp_path / "ws")


def test_run_rechecks_scopus_institutional_credential_independently(tmp_path, monkeypatch):
    """Regression: SCOPUS_API_KEY alone is no longer sufficient -- the child
    must independently re-reject a missing SCOPUS_INSTITUTIONAL_KEY too,
    same defense-in-depth rationale as every other re-check in this file."""
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, _spec_dict(databases=("scopus",)))
    monkeypatch.setenv("SCOPUS_API_KEY", "k")
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    args = SimpleNamespace(spec=spec_path, output_dir=tmp_path / "out",
                            cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id="t")

    def fail_if_called(*a, **k):
        raise AssertionError("_invoke_graph must never be called without SCOPUS_INSTITUTIONAL_KEY")

    monkeypatch.setattr(child_mod, "_invoke_graph", fail_if_called)
    from slr_pipeline.track3_run import Track3ConfigError
    with pytest.raises(Track3ConfigError, match="SCOPUS_INSTITUTIONAL_KEY"):
        child_mod._run(args, port=1, workspace=tmp_path / "ws")


def test_run_sets_output_dir_and_delegates_to_invoke_graph(tmp_path, monkeypatch):
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, _spec_dict())
    monkeypatch.setenv("SCOPUS_API_KEY", "k")
    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "i")
    output_dir = tmp_path / "out"
    args = SimpleNamespace(spec=spec_path, output_dir=output_dir,
                            cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id="t")

    calls = {}

    def fake_invoke_graph(a, cfg, *, port, workspace):
        calls["cfg_output_dir"] = cfg.to_dict()["output_dir"]
        calls["port"] = port
        calls["workspace"] = workspace
        return 0

    monkeypatch.setattr(child_mod, "_invoke_graph", fake_invoke_graph)
    rc = child_mod._run(args, port=4242, workspace=tmp_path / "ws")
    assert rc == 0
    assert calls["cfg_output_dir"] == str(output_dir)
    assert calls["port"] == 4242
    assert output_dir.is_dir()  # ensure_output_dir() actually created it


# ---------------------------------------------------------------------------
# _invoke_graph -- CostCapProxy lifecycle, queue-emptiness checks,
# terminal_state.json / terminal_graph_snapshot.json writing
# ---------------------------------------------------------------------------

class _FakeCostCapProxy:
    instances = []

    def __init__(self, ledger, call_log_path, *, port=0):
        self.ledger = ledger
        self.call_log_path = call_log_path
        self.port = port
        self.started = False
        self.stopped = False
        _FakeCostCapProxy.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeGraph:
    def __init__(self, terminal_state, next_nodes=(), existing_values=None):
        self._terminal_state = terminal_state
        self._next_nodes = tuple(next_nodes)
        self._existing_values = existing_values if existing_values is not None else {}
        self.invoked_with = None
        self.invoke_call_count = 0

    def invoke(self, state, config):
        self.invoke_call_count += 1
        self.invoked_with = (state, config)
        return self._terminal_state

    def get_state(self, config):
        return SimpleNamespace(next=self._next_nodes, values=self._existing_values)


class _FakeCheckpointerCM:
    """Stand-in for `SqliteSaver.from_conn_string(...)`'s context manager --
    `_open_checkpointer()` itself is monkeypatched, so no real sqlite file is
    ever created by the seam-mocked tests below."""

    def __init__(self, checkpointer=None):
        self._checkpointer = checkpointer if checkpointer is not None else object()

    def __enter__(self):
        return self._checkpointer

    def __exit__(self, *exc_info):
        return False


class _FakeLedger:
    def __init__(self, path, *, cap_usd):
        self.path = path
        self.cap_usd = cap_usd

    def snapshot(self):
        return {"committed_usd": 1.23}


def _invoke_graph_args(tmp_path, thread_id="t"):
    args = SimpleNamespace(cost_cap_path=tmp_path / "ledger.json", cap_usd=30.0, thread_id=thread_id)
    cfg = SimpleNamespace(to_dict=lambda: {"some": "config"})
    return args, cfg


def _patch_graph_seams(monkeypatch, *, terminal_state, next_nodes=(), run_mode_doc=None,
                        existing_values=None):
    _FakeCostCapProxy.instances = []
    fake_graph = _FakeGraph(terminal_state, next_nodes=next_nodes, existing_values=existing_values)
    monkeypatch.setattr("slr_pipeline.cost_cap.ledger.CostLedger", _FakeLedger)
    monkeypatch.setattr("slr_pipeline.cost_cap.proxy.CostCapProxy", _FakeCostCapProxy)
    monkeypatch.setattr("slr_pipeline.graphs.scoping.build_scoping_graph",
                         lambda checkpointer=None: fake_graph)
    monkeypatch.setattr(child_mod, "_open_checkpointer", lambda workspace: _FakeCheckpointerCM())
    monkeypatch.setattr("slr_pipeline.state.initial_state", lambda cfg_dict: {"initial": cfg_dict})
    if run_mode_doc is not None:
        monkeypatch.setattr(child_mod, "_build_run_mode", lambda *a, **k: run_mode_doc)
    return fake_graph


class _FakeRunModeDoc:
    def model_dump(self, mode="json"):
        return {"run_mode_schema_version": "1", "mode_profile": "track3_live"}


def test_invoke_graph_happy_path_writes_flat_terminal_files(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [],
        "extraction_review_queue": [],
        "review_run_id": "11111111-1111-1111-1111-111111111111",
    }
    run_mode_doc = _FakeRunModeDoc()
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state, next_nodes=(), run_mode_doc=run_mode_doc)
    monkeypatch.setattr("slr_pipeline.reference_run_mode.render_run_mode_statement_md",
                         lambda doc: "# Track 3 live run\nfake statement")

    rc = child_mod._invoke_graph(args, cfg, port=4321, workspace=workspace)
    assert rc == 0

    proxy = _FakeCostCapProxy.instances[0]
    assert proxy.started and proxy.stopped
    assert proxy.port == 4321
    assert proxy.call_log_path == workspace / "cost_cap_call_log.jsonl"

    terminal_state_path = workspace / "terminal_state.json"
    assert terminal_state_path.exists()
    assert json.loads(terminal_state_path.read_text(encoding="utf-8")) == terminal_state
    assert not (workspace / "raw").exists()  # flat, not nested

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
    assert json.loads(run_mode_path.read_text(encoding="utf-8")) == run_mode_doc.model_dump()

    statement_path = workspace / "run_mode_statement.md"
    assert statement_path.exists()
    assert "Track 3 live run" in statement_path.read_text(encoding="utf-8")

    # no violations.jsonl -- C3's own explicit scope decision (no network guard for Track 3)
    assert not (workspace / "violations.jsonl").exists()


def test_invoke_graph_raises_on_nonempty_full_text_review_queue(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [{"record_id": "fx-0001"}],
        "extraction_review_queue": [],
        "review_run_id": "r",
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    with pytest.raises(RuntimeError, match="full_text_review_queue"):
        child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)
    assert not (workspace / "terminal_state.json").exists()


def test_invoke_graph_raises_on_nonempty_extraction_review_queue(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {
        "full_text_review_queue": [],
        "extraction_review_queue": [{"record_id": "fx-0001"}],
        "review_run_id": "r",
    }
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    with pytest.raises(RuntimeError, match="extraction_review_queue"):
        child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)


def test_invoke_graph_raises_on_nonempty_next_nodes(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {"full_text_review_queue": [], "extraction_review_queue": [], "review_run_id": "r"}
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state, next_nodes=("some_node",))
    with pytest.raises(RuntimeError, match="next="):
        child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)


def test_invoke_graph_raises_on_missing_review_run_id(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {"full_text_review_queue": [], "extraction_review_queue": [], "review_run_id": None}
    _patch_graph_seams(monkeypatch, terminal_state=terminal_state)
    with pytest.raises(RuntimeError, match="review_run_id"):
        child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)
    # terminal_state.json is written before the review_run_id check -- still on disk
    assert (workspace / "terminal_state.json").exists()


def test_invoke_graph_stops_proxy_even_when_graph_invoke_raises(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    _FakeCostCapProxy.instances = []

    class _RaisingGraph:
        def get_state(self, config):
            return SimpleNamespace(next=(), values={})

        def invoke(self, state, config):
            raise ValueError("graph blew up")

    monkeypatch.setattr("slr_pipeline.cost_cap.ledger.CostLedger", _FakeLedger)
    monkeypatch.setattr("slr_pipeline.cost_cap.proxy.CostCapProxy", _FakeCostCapProxy)
    monkeypatch.setattr("slr_pipeline.graphs.scoping.build_scoping_graph",
                         lambda checkpointer=None: _RaisingGraph())
    monkeypatch.setattr(child_mod, "_open_checkpointer", lambda workspace: _FakeCheckpointerCM())
    monkeypatch.setattr("slr_pipeline.state.initial_state", lambda cfg_dict: {})

    with pytest.raises(ValueError, match="graph blew up"):
        child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)
    assert _FakeCostCapProxy.instances[0].stopped


# ---------------------------------------------------------------------------
# _invoke_graph -- resume-vs-fresh branching on `graph.get_state(...).values`
# ---------------------------------------------------------------------------

def test_invoke_graph_fresh_start_invokes_with_initial_state(tmp_path, monkeypatch):
    """No prior checkpoint for this thread_id (`.values == {}`) -> a fresh
    `initial_state(cfg.to_dict())` is passed to `graph.invoke()`, exactly as
    before this checkpointer was introduced."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {"full_text_review_queue": [], "extraction_review_queue": [], "review_run_id": "r"}
    fake_graph = _patch_graph_seams(monkeypatch, terminal_state=terminal_state, existing_values={},
                                     run_mode_doc=_FakeRunModeDoc())
    monkeypatch.setattr("slr_pipeline.reference_run_mode.render_run_mode_statement_md", lambda doc: "stmt")

    child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)

    assert fake_graph.invoke_call_count == 1
    state, run_config = fake_graph.invoked_with
    assert state == {"initial": cfg.to_dict()}
    assert run_config == {"configurable": {"thread_id": "t"}}


def test_invoke_graph_resume_invokes_with_none_not_initial_state(tmp_path, monkeypatch):
    """A non-empty prior checkpoint (`.values` non-empty) for this thread_id
    -> `graph.invoke(None, run_config)` resumes from the last completed node
    instead of re-running `initial_state(...)` from scratch."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {"full_text_review_queue": [], "extraction_review_queue": [], "review_run_id": "r"}
    fake_graph = _patch_graph_seams(
        monkeypatch, terminal_state=terminal_state, existing_values={"already": "progressed"},
        run_mode_doc=_FakeRunModeDoc(),
    )
    monkeypatch.setattr("slr_pipeline.reference_run_mode.render_run_mode_statement_md", lambda doc: "stmt")

    child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)

    assert fake_graph.invoke_call_count == 1
    state, run_config = fake_graph.invoked_with
    assert state is None
    assert run_config == {"configurable": {"thread_id": "t"}}


def test_invoke_graph_passes_checkpointer_from_open_checkpointer_seam(tmp_path, monkeypatch):
    """`build_scoping_graph()` must receive the checkpointer yielded by
    `_open_checkpointer(workspace)`'s context manager, not the default
    (in-memory) checkpointer."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args, cfg = _invoke_graph_args(tmp_path)
    terminal_state = {"full_text_review_queue": [], "extraction_review_queue": [], "review_run_id": "r"}
    _FakeCostCapProxy.instances = []
    sentinel_checkpointer = object()
    fake_graph = _FakeGraph(terminal_state)
    received = {}

    monkeypatch.setattr("slr_pipeline.cost_cap.ledger.CostLedger", _FakeLedger)
    monkeypatch.setattr("slr_pipeline.cost_cap.proxy.CostCapProxy", _FakeCostCapProxy)

    def fake_build_scoping_graph(checkpointer=None):
        received["checkpointer"] = checkpointer
        return fake_graph

    monkeypatch.setattr("slr_pipeline.graphs.scoping.build_scoping_graph", fake_build_scoping_graph)
    monkeypatch.setattr(child_mod, "_open_checkpointer",
                         lambda workspace: _FakeCheckpointerCM(sentinel_checkpointer))
    monkeypatch.setattr("slr_pipeline.state.initial_state", lambda cfg_dict: {"initial": cfg_dict})
    monkeypatch.setattr(child_mod, "_build_run_mode", lambda *a, **k: _FakeRunModeDoc())
    monkeypatch.setattr("slr_pipeline.reference_run_mode.render_run_mode_statement_md", lambda doc: "stmt")

    child_mod._invoke_graph(args, cfg, port=1, workspace=workspace)
    assert received["checkpointer"] is sentinel_checkpointer


def test_invoke_graph_kill_restart_does_not_reinvoke_completed_node(tmp_path):
    """Real `SqliteSaver` (not mocked) + a small standalone `StateGraph`
    (mirrors `graphs/scoping.py`'s own `build_scoping_graph(checkpointer=...)`
    shape, not the full real scoping graph -- see plan item 5): a node that
    raises partway through leaves a durable checkpoint for everything before
    it; reopening a brand-new `SqliteSaver` against the same workspace/
    thread_id (simulating a process restart) and calling
    `graph.invoke(None, run_config)` resumes without re-invoking the already-
    completed node -- proving a crashed, paid stage is not repeated."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_config = {"configurable": {"thread_id": "kill-restart-thread"}}
    calls = {"node_a": 0, "node_b": 0}

    class _State(TypedDict):
        log: list

    def node_a(state):
        calls["node_a"] += 1
        return {"log": state["log"] + ["a"]}

    should_fail = {"value": True}

    def node_b(state):
        calls["node_b"] += 1
        if should_fail["value"]:
            raise RuntimeError("simulated crash mid-graph")
        return {"log": state["log"] + ["b"]}

    def _build():
        builder = StateGraph(_State)
        builder.add_node("node_a", node_a)
        builder.add_node("node_b", node_b)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", END)
        return builder

    with child_mod._open_checkpointer(workspace) as checkpointer:
        graph = _build().compile(checkpointer=checkpointer)
        with pytest.raises(RuntimeError, match="simulated crash"):
            graph.invoke({"log": []}, run_config)

    assert calls == {"node_a": 1, "node_b": 1}
    assert (workspace / "graph_checkpoints.sqlite").exists()

    should_fail["value"] = False
    # Brand-new SqliteSaver instance against the same workspace/thread_id --
    # simulates a fresh process, not a reused in-memory object.
    with child_mod._open_checkpointer(workspace) as checkpointer:
        graph = _build().compile(checkpointer=checkpointer)
        existing_state = graph.get_state(run_config)
        assert existing_state.values  # checkpoint survived across the "restart"
        terminal_state = graph.invoke(None, run_config)

    assert terminal_state["log"] == ["a", "b"]
    # node_a was NOT re-invoked on resume -- still exactly 1 call.
    assert calls == {"node_a": 1, "node_b": 2}


# ---------------------------------------------------------------------------
# _extraction_call_count -- sums batch_executions per checkpoint, NOT
# len(checkpoints) (a document's extraction may span multiple real calls)
# ---------------------------------------------------------------------------

class _FakeCheckpoint:
    def __init__(self, path):
        self.path = path


class _FakeBundle:
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints


class _FakeDocResult:
    def __init__(self, batch_executions):
        self.batch_executions = batch_executions


def test_extraction_call_count_sums_batch_executions_not_checkpoint_count(tmp_path, monkeypatch):
    doc1_path = tmp_path / "doc1.json"
    doc1_path.write_text(json.dumps({"marker": "doc1"}), encoding="utf-8")
    doc2_path = tmp_path / "doc2.json"
    doc2_path.write_text(json.dumps({"marker": "doc2"}), encoding="utf-8")

    bundle = _FakeBundle(checkpoints=[_FakeCheckpoint(doc1_path), _FakeCheckpoint(doc2_path)])
    monkeypatch.setattr("slr_pipeline.nodes.reporting._load_extraction_bundle", lambda ts, cfg: bundle)

    import slr_pipeline.extraction.results as results_mod

    def fake_validate(text):
        marker = json.loads(text)["marker"]
        return {"doc1": _FakeDocResult([1, 2, 3]), "doc2": _FakeDocResult([1])}[marker]  # 3 + 1 real calls

    monkeypatch.setattr(results_mod.DocumentExtractionResult, "model_validate_json", staticmethod(fake_validate))

    count = child_mod._extraction_call_count(terminal_state={}, cfg=SimpleNamespace())
    assert count == 4  # 3 + 1, NOT len(checkpoints) == 2


def test_extraction_call_count_zero_when_no_bundle(monkeypatch):
    monkeypatch.setattr("slr_pipeline.nodes.reporting._load_extraction_bundle", lambda ts, cfg: None)
    assert child_mod._extraction_call_count(terminal_state={}, cfg=SimpleNamespace()) == 0


# ---------------------------------------------------------------------------
# _full_text_eligibility_call_count -- one real call per FT_SCREENING item
# ---------------------------------------------------------------------------

def test_full_text_eligibility_call_count_zero_when_no_ft_native(monkeypatch):
    monkeypatch.setattr("slr_pipeline.nodes.reporting.capture_ft_native_provenance", lambda ts, cfg: None)
    assert child_mod._full_text_eligibility_call_count(terminal_state={}, cfg=SimpleNamespace()) == 0


def _fake_item(*, provenance):
    """Schema-faithful stand-in for `ScreeningItemResult`: the real function
    under test only ever reads `.provenance` (None or an object exposing
    `.total_provider_calls`), matching `orchestrator.py`'s own real
    invariant that `provenance` is set if and only if `execution_status ==
    "completed"` (`_terminal_result()` always passes `provenance=None` for
    every other status)."""
    return SimpleNamespace(provenance=provenance)


def _fake_provenance(total_provider_calls, *, cache_status="miss"):
    return SimpleNamespace(total_provider_calls=total_provider_calls, cache_status=cache_status)


def test_full_text_eligibility_call_count_sums_provenance_calls_not_item_count(tmp_path, monkeypatch):
    run_result_path = tmp_path / "run_result.json"
    run_result_path.write_text("{}", encoding="utf-8")
    ft_native = {"ft_screening_artifact_paths": {"run_result_json": str(run_result_path)}}
    monkeypatch.setattr("slr_pipeline.nodes.reporting.capture_ft_native_provenance", lambda ts, cfg: ft_native)

    import ft_screening

    fake_result = SimpleNamespace(
        items=[
            _fake_item(provenance=_fake_provenance(1)),  # completed, one real call
            _fake_item(provenance=_fake_provenance(0)),  # completed, cache hit -- no real call
            _fake_item(provenance=_fake_provenance(2)),  # completed, retried -- two real calls
            _fake_item(provenance=None),  # input_unavailable/duplicate_input/processing_failed/not_processed
            _fake_item(provenance=None),
        ]
    )
    monkeypatch.setattr(ft_screening.ScreeningRunResult, "model_validate_json",
                         staticmethod(lambda text: fake_result))

    count = child_mod._full_text_eligibility_call_count(terminal_state={}, cfg=SimpleNamespace())
    # 1 + 0 + 2 == 3 real calls, NOT len(items) == 5.
    assert count == 3


def test_full_text_eligibility_call_count_excludes_content_reuse_clones(tmp_path, monkeypatch):
    """Regression: orchestrator.py's _clone_for_reuse() (duplicate_content_policy
    == "reuse") copies the canonical item's non-zero total_provider_calls
    verbatim onto every content-duplicate item (cache_status="content_reuse")
    -- summing those too would count the same real call multiple times."""
    run_result_path = tmp_path / "run_result.json"
    run_result_path.write_text("{}", encoding="utf-8")
    ft_native = {"ft_screening_artifact_paths": {"run_result_json": str(run_result_path)}}
    monkeypatch.setattr("slr_pipeline.nodes.reporting.capture_ft_native_provenance", lambda ts, cfg: ft_native)

    import ft_screening

    fake_result = SimpleNamespace(
        items=[
            _fake_item(provenance=_fake_provenance(1, cache_status="miss")),  # canonical, one real call
            # two content-duplicates of the canonical item above, each cloned
            # with the canonical's own total_provider_calls == 1
            _fake_item(provenance=_fake_provenance(1, cache_status="content_reuse")),
            _fake_item(provenance=_fake_provenance(1, cache_status="content_reuse")),
        ]
    )
    monkeypatch.setattr(ft_screening.ScreeningRunResult, "model_validate_json",
                         staticmethod(lambda text: fake_result))

    count = child_mod._full_text_eligibility_call_count(terminal_state={}, cfg=SimpleNamespace())
    # Only the canonical item's one real call, NOT 1 + 1 + 1 == 3.
    assert count == 1


# ---------------------------------------------------------------------------
# _resolution_suggestion_interrupted_count -- real, on-disk
# `.interrupted_unknown.json` count, invisible to ai_use_events (LLM-
# resolution plan sec. 3.4 round-5 addition)
# ---------------------------------------------------------------------------

def test_resolution_suggestion_interrupted_count_zero_when_no_extraction_run():
    count = child_mod._resolution_suggestion_interrupted_count(
        terminal_state={}, cfg=SimpleNamespace(ensure_output_dir=lambda: Path("unused")),
    )
    assert count == 0


def test_resolution_suggestion_interrupted_count_reads_real_archives(tmp_path):
    from slr_pipeline.ai_traceability import compute_event_id
    from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore
    from slr_pipeline.extraction.resolution_ids import safe_key
    from slr_pipeline.extraction.resolution_schemas import (
        EXTRACTION_RESOLUTION_SUGGESTION_STAGE, build_interrupted_call_record, compute_expected_work_item_id,
    )

    run_id = "run-1"
    record_id, field_id = "doc-1", "field-1"
    store = ExtractionCheckpointStore(output_dir=tmp_path / "extraction_runs", run_id=run_id)
    suggestions_dir = store.run_dir / "extraction_resolution_suggestions"
    suggestions_dir.mkdir(parents=True)

    work_item_id = compute_expected_work_item_id(
        extraction_run_id=run_id, record_id=record_id, field_id=field_id,
    )
    interrupted = build_interrupted_call_record(
        extraction_run_id=run_id, record_id=record_id, field_id=field_id, attempt=1,
        ai_use_event_id=compute_event_id(
            review_run_id="r", stage=EXTRACTION_RESOLUTION_SUGGESTION_STAGE, work_item_id=work_item_id, attempt=1,
        ),
        review_run_id="r", work_item_id=work_item_id,
        reserved_at="2026-09-19T00:00:00Z", archived_at="2026-09-19T00:01:00Z",
    )
    key = safe_key(run_id, record_id, field_id)
    marker_path = suggestions_dir / f"{key}__attempt-1.interrupted_unknown.json"
    marker_path.write_text(interrupted.model_dump_json(), encoding="utf-8")

    cfg = SimpleNamespace(ensure_output_dir=lambda: tmp_path)
    terminal_state = {"extraction_run_id": run_id}
    count = child_mod._resolution_suggestion_interrupted_count(terminal_state, cfg)
    assert count == 1


def test_build_run_mode_passes_through_interrupted_count(monkeypatch):
    monkeypatch.setattr(child_mod, "_extraction_call_count", lambda ts, cfg: 0)
    monkeypatch.setattr(child_mod, "_full_text_eligibility_call_count", lambda ts, cfg: 0)
    monkeypatch.setattr(child_mod, "_resolution_suggestion_interrupted_count", lambda ts, cfg: 3)
    monkeypatch.setattr("slr_pipeline.ai_traceability.AIUseEventStore", lambda **kw: SimpleNamespace())
    monkeypatch.setattr("slr_pipeline.nodes.reporting._load_verified_events", lambda store: [])

    received = {}

    def fake_build_run_mode_document_live(**kwargs):
        received.update(kwargs)
        return _FakeRunModeDoc()

    monkeypatch.setattr(
        "slr_pipeline.reference_run_mode.build_run_mode_document_live", fake_build_run_mode_document_live
    )

    cfg = SimpleNamespace(
        ensure_output_dir=lambda: Path("unused"),
        synthesis=SimpleNamespace(run_bertopic=False),
    )
    ledger = SimpleNamespace(snapshot=lambda: {"committed_usd": 0.0})
    child_mod._build_run_mode(cfg, terminal_state={}, review_run_id="r", ledger=ledger, port=1)
    assert received["resolution_suggestion_interrupted"] == 3
