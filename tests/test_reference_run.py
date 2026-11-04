"""Unit tests for `slr_pipeline.reference_run` -- the reference-run
**parent** driver (`track2_pipeline_readiness_plan_2026-08-30.md` §12,
step 13 [C12]). `subprocess.run` is mocked throughout -- no real Docling
assets or a real child process are needed to verify the driver's own
env/argv/workspace-file-check logic."""
from __future__ import annotations

import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import slr_pipeline.reference_run as run_mod


def test_reserve_loopback_port_returns_bindable_port():
    port = run_mod._reserve_loopback_port()
    assert 0 < port < 65536
    # the port is free again immediately after (closed before returning)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    finally:
        s.close()


def test_allocate_workspace_creates_unique_dir(tmp_path):
    ws1 = run_mod.allocate_workspace(tmp_path)
    ws2 = run_mod.allocate_workspace(tmp_path)
    assert ws1.is_dir()
    assert ws2.is_dir()
    assert ws1 != ws2
    assert ws1.name.startswith("reference_run_")


def test_build_child_env_only_named_vars_and_allowlist(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    base_env = {
        "SystemRoot": r"C:\Windows",
        "OPENAI_API_KEY": "sk-real-secret-should-not-leak",
        "SOME_RANDOM_VAR": "should-not-leak",
        "HTTP_PROXY": "http://proxy.example.com",
    }
    env = run_mod.build_child_env(
        workspace=workspace, replay_port=12345, python_executable=r"C:\fake\Scripts\python.exe",
        base_env=base_env,
    )
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:12345/v1"
    assert env["OPENAI_API_KEY"] == "sk-replay-not-a-real-key"  # never the real inherited one
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["NO_PROXY"] == "127.0.0.1"
    assert env["SCREENING_BATCH_SIZE"] == "5"
    assert env["SCREENING_WORKERS"] == "1"
    assert "SOME_RANDOM_VAR" not in env
    assert "HTTP_PROXY" not in env
    assert env["SystemRoot"] == r"C:\Windows"  # allowlisted, copied through

    caches_dir = workspace / "caches"
    assert env["TEMP"] == str(caches_dir)
    assert env["TMP"] == str(caches_dir)
    assert env["HF_HOME"] == str(caches_dir / "huggingface")
    assert env["TORCH_HOME"] == str(caches_dir / "torch")
    assert env["MPLCONFIGDIR"] == str(caches_dir / "matplotlib")
    assert env["XDG_CACHE_HOME"] == str(caches_dir)
    assert env["HOME"] == str(caches_dir / "home")
    assert env["USERPROFILE"] == str(caches_dir / "home")


def test_build_child_env_path_is_scripts_dir_plus_system32_only(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    python_exe = tmp_path / "venv" / "Scripts" / "python.exe"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_bytes(b"")
    env = run_mod.build_child_env(
        workspace=workspace, replay_port=1, python_executable=str(python_exe),
        base_env={"SystemRoot": r"C:\Windows"},
    )
    parts = env["PATH"].split(os.pathsep)
    assert str(python_exe.parent) in parts
    assert str(Path(r"C:\Windows") / "System32") in parts
    assert len(parts) == 2


def test_build_child_env_windows_allowlist_vars_absent_when_not_in_base(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = run_mod.build_child_env(
        workspace=workspace, replay_port=1, python_executable=r"C:\fake\Scripts\python.exe",
        base_env={},  # nothing to copy through
    )
    for name in run_mod._WINDOWS_RUNTIME_ALLOWLIST:
        assert name not in env


def test_build_child_argv_shape(tmp_path):
    argv = run_mod.build_child_argv(
        python_executable="PY", workspace=tmp_path / "ws", extraction_output_dir=tmp_path / "out",
        spec_path=tmp_path / "s.json", cassette_path=tmp_path / "c.json",
        corpus_oracle_path=tmp_path / "o.json", pdf_source_dir=tmp_path / "pdfs",
        fixture_corpus_root=tmp_path / "corpus", thread_id="my-thread",
    )
    assert argv[:3] == ["PY", "-m", "slr_pipeline.reference_run_child"]
    assert "--workspace" in argv and str(tmp_path / "ws") in argv
    assert "--thread-id" in argv and "my-thread" in argv


def test_run_reference_success_returns_workspace(tmp_path, monkeypatch):
    calls = {}

    def fake_run(argv, *, env, cwd, capture_output, text):
        calls["argv"] = argv
        calls["env"] = env
        calls["cwd"] = cwd
        workspace = Path(cwd)
        for name in run_mod._WORKSPACE_FILES:
            (workspace / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    workspace = run_mod.run_reference(base_tmp_dir=tmp_path, extraction_output_dir=tmp_path / "out")
    assert workspace.is_dir()
    assert calls["cwd"] == str(workspace)
    for name in run_mod._WORKSPACE_FILES:
        assert (workspace / name).exists()


def test_run_reference_nonzero_exit_raises_and_keeps_workspace_by_default(tmp_path, monkeypatch):
    def fake_run(argv, *, env, cwd, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    with pytest.raises(run_mod.ReferenceRunError, match="exited 1"):
        run_mod.run_reference(base_tmp_dir=tmp_path, extraction_output_dir=tmp_path / "out")
    workspaces = list(tmp_path.glob("reference_run_*"))
    assert len(workspaces) == 1
    assert workspaces[0].is_dir()  # kept, keep_workspace defaults True


def test_run_reference_nonzero_exit_removes_workspace_when_keep_workspace_false(tmp_path, monkeypatch):
    def fake_run(argv, *, env, cwd, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    with pytest.raises(run_mod.ReferenceRunError):
        run_mod.run_reference(
            base_tmp_dir=tmp_path, extraction_output_dir=tmp_path / "out", keep_workspace=False,
        )
    workspaces = list(tmp_path.glob("reference_run_*"))
    assert workspaces == []


def test_run_reference_zero_exit_but_missing_workspace_file_raises(tmp_path, monkeypatch):
    def fake_run(argv, *, env, cwd, capture_output, text):
        workspace = Path(cwd)
        # write only 3 of the 4 required files
        for name in run_mod._WORKSPACE_FILES[:-1]:
            (workspace / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    with pytest.raises(run_mod.ReferenceRunError, match="missing workspace file"):
        run_mod.run_reference(base_tmp_dir=tmp_path, extraction_output_dir=tmp_path / "out")


def test_run_reference_passes_replay_port_into_env_and_argv(tmp_path, monkeypatch):
    captured = {}

    def fake_reserve_port():
        return 55555

    def fake_run(argv, *, env, cwd, capture_output, text):
        captured["argv"] = argv
        captured["env"] = env
        workspace = Path(cwd)
        for name in run_mod._WORKSPACE_FILES:
            (workspace / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_mod, "_reserve_loopback_port", fake_reserve_port)
    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    run_mod.run_reference(base_tmp_dir=tmp_path, extraction_output_dir=tmp_path / "out")
    assert captured["env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:55555/v1"


def test_main_prints_workspace_and_returns_zero(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "some_workspace"
    workspace.mkdir()
    monkeypatch.setattr(run_mod, "run_reference", lambda: workspace)
    rc = run_mod.main()
    assert rc == 0
    captured = capsys.readouterr()
    assert str(workspace) in captured.out
