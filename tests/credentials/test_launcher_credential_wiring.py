"""Launcher-level credential-wiring tests (plan §2.3, amber-quilting-osprey.md):
`track3_run.py`/`track3_continuation_run.py` request exactly the right
credential names, and never touch the credential store before authorization
(--cap-usd / --estimate-only) has been validated. All offline: subprocess.run
is mocked.

`resolve_credential`/`credential_status`/`loaded_credentials`/etc. default
`keyring_backend` to the real `keyring` module -- a default parameter value
bound once at `credentials.py` import time, so a test can never swap in a
fake by reassigning `credentials.keyring` (that name is never looked up
again). Instead, these tests patch `keyring.get_password` itself, on the
one real module object every default already points at."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import keyring as real_keyring
import pytest

import slr_pipeline.track3_continuation_run as continuation_mod
import slr_pipeline.track3_run as run_mod
from slr_pipeline import credentials


class _RecordingGetPassword:
    def __init__(self, passwords=None, raise_on_any_call=False):
        self._passwords = dict(passwords or {})
        self._raise_on_any_call = raise_on_any_call
        self.calls = []

    def __call__(self, service, name):
        self.calls.append((service, name))
        if self._raise_on_any_call:
            raise AssertionError(f"keyring must not be consulted for {name!r} here")
        return self._passwords.get(name)


def _write_spec(tmp_path, *, databases=("pubmed",), dedup_mode="exact"):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "search": {"databases": list(databases)},
        "dedup": {"mode": dedup_mode},
    }), encoding="utf-8")
    return spec_path


@pytest.fixture(autouse=True)
def _fake_docling_validate(monkeypatch):
    from ft_screening import docling_artifacts
    monkeypatch.setattr(docling_artifacts, "validate", lambda path, expected_tree_digest=None: "fake" * 16)


# ---------------------------------------------------------------------------
# track3_run.py -- requests Scopus names only when Scopus is configured
# ---------------------------------------------------------------------------

def test_run_track3_requests_openai_only_when_scopus_not_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    spec_path = _write_spec(tmp_path, databases=("pubmed",))
    captured_names = {}

    real_loaded_credentials = credentials.loaded_credentials

    def spy_loaded_credentials(names, **kwargs):
        captured_names["names"] = tuple(names)
        return real_loaded_credentials(names, **kwargs)

    monkeypatch.setattr(run_mod.credentials, "loaded_credentials", spy_loaded_credentials)

    def fake_run(argv, *, env, cwd, capture_output, text, timeout=None):
        for name in run_mod._WORKSPACE_FILES:
            (Path(cwd) / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    run_mod.run_track3(spec_path=spec_path, base_tmp_dir=tmp_path)
    assert captured_names["names"] == ("OPENAI_API_KEY",)


def test_run_track3_requests_all_three_names_when_scopus_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("SCOPUS_API_KEY", "sk-fake-scopus")
    monkeypatch.setenv("SCOPUS_INSTITUTIONAL_KEY", "sk-fake-scopus-inst")
    spec_path = _write_spec(tmp_path, databases=("pubmed", "scopus"))
    captured_names = {}

    real_loaded_credentials = credentials.loaded_credentials

    def spy_loaded_credentials(names, **kwargs):
        captured_names["names"] = tuple(names)
        return real_loaded_credentials(names, **kwargs)

    monkeypatch.setattr(run_mod.credentials, "loaded_credentials", spy_loaded_credentials)

    def fake_run(argv, *, env, cwd, capture_output, text, timeout=None):
        for name in run_mod._WORKSPACE_FILES:
            (Path(cwd) / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    run_mod.run_track3(spec_path=spec_path, base_tmp_dir=tmp_path)
    assert captured_names["names"] == ("OPENAI_API_KEY", "SCOPUS_API_KEY", "SCOPUS_INSTITUTIONAL_KEY")


def test_run_track3_never_asks_keyring_for_scopus_names_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    spec_path = _write_spec(tmp_path, databases=("pubmed",))
    recorder = _RecordingGetPassword(passwords={"OPENAI_API_KEY": "sk-vault"})
    monkeypatch.setattr(real_keyring, "get_password", recorder)

    def fake_run(argv, *, env, cwd, capture_output, text, timeout=None):
        for name in run_mod._WORKSPACE_FILES:
            (Path(cwd) / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    run_mod.run_track3(spec_path=spec_path, base_tmp_dir=tmp_path)
    requested_names = {name for _service, name in recorder.calls}
    assert requested_names == {"OPENAI_API_KEY"}


# ---------------------------------------------------------------------------
# track3_continuation_run.py -- estimate-only / --cap-usd gate the credential
# store, requests only OPENAI_API_KEY
# ---------------------------------------------------------------------------

def _continuation_argv(tmp_path, *, extra=()):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"search": {"databases": ["pubmed"]}, "dedup": {"mode": "exact"}}),
                          encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    argv = [
        "--spec", str(spec_path),
        "--bundles-root", str(tmp_path / "bundles"),
        "--parent-bundle-id", "fake-parent",
        "--continuation-manifest", str(tmp_path / "manifest.json"),
        "--retrieved-pdfs-dir", str(tmp_path / "pdfs"),
        "--workspace", str(workspace),
    ]
    return list(argv) + list(extra)


def _stub_continuation_pipeline(monkeypatch):
    monkeypatch.setattr(continuation_mod, "resolve_parent_bundle", lambda *a, **k: object())
    monkeypatch.setattr(continuation_mod, "_resolve_continuation_inputs", lambda **k: SimpleNamespace(
        retrieved_manifest_docs=[], preserved_included_parent_records=[],
    ))
    monkeypatch.setattr(continuation_mod, "estimate_continuation_cost",
                         lambda **k: SimpleNamespace(render_breakdown=lambda: "estimate"))


def test_continuation_estimate_only_never_touches_keyring(tmp_path, monkeypatch):
    recorder = _RecordingGetPassword(raise_on_any_call=True)
    monkeypatch.setattr(real_keyring, "get_password", recorder)
    _stub_continuation_pipeline(monkeypatch)

    def fail_if_called(*a, **k):
        raise AssertionError("run_continuation must never be called for --estimate-only")

    monkeypatch.setattr(continuation_mod, "run_continuation", fail_if_called)
    rc = continuation_mod.main(_continuation_argv(tmp_path, extra=["--estimate-only"]))
    assert rc == 0
    assert recorder.calls == []


def test_continuation_missing_cap_usd_errors_before_keyring_touched(tmp_path, monkeypatch):
    recorder = _RecordingGetPassword(raise_on_any_call=True)
    monkeypatch.setattr(real_keyring, "get_password", recorder)
    _stub_continuation_pipeline(monkeypatch)

    def fail_if_called(*a, **k):
        raise AssertionError("run_continuation must never be called without --cap-usd")

    monkeypatch.setattr(continuation_mod, "run_continuation", fail_if_called)
    with pytest.raises(SystemExit):
        continuation_mod.main(_continuation_argv(tmp_path))
    assert recorder.calls == []


def test_continuation_only_requests_openai_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    _stub_continuation_pipeline(monkeypatch)

    captured_names = {}
    real_loaded_credentials = credentials.loaded_credentials

    def spy_loaded_credentials(names, **kwargs):
        captured_names["names"] = tuple(names)
        return real_loaded_credentials(names, **kwargs)

    monkeypatch.setattr(continuation_mod.credentials, "loaded_credentials", spy_loaded_credentials)
    monkeypatch.setattr(continuation_mod, "run_continuation",
                         lambda **k: {"review_run_id": "fake-run"})
    rc = continuation_mod.main(_continuation_argv(tmp_path, extra=["--cap-usd", "5.0"]))
    assert rc == 0
    assert captured_names["names"] == ("OPENAI_API_KEY",)


def test_continuation_keyring_never_asked_for_scopus_names(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    recorder = _RecordingGetPassword(passwords={"OPENAI_API_KEY": "sk-vault"})
    monkeypatch.setattr(real_keyring, "get_password", recorder)
    _stub_continuation_pipeline(monkeypatch)
    monkeypatch.setattr(continuation_mod, "run_continuation",
                         lambda **k: {"review_run_id": "fake-run"})
    rc = continuation_mod.main(_continuation_argv(tmp_path, extra=["--cap-usd", "5.0"]))
    assert rc == 0
    requested_names = {name for _service, name in recorder.calls}
    assert requested_names == {"OPENAI_API_KEY"}
