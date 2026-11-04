"""Regression tests for slr_pipeline.run_identity.resolve_review_run_id --
Stage 3 node instrumentation (codex review round: this resolver had no
dedicated test file at all)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.run_identity import resolve_review_run_id


def test_package_absolute_import_works_with_only_repo_root_on_path():
    """codex review: run_identity.py previously used a bare `from state
    import ReviewState`, which only works when slr_pipeline/ itself is
    separately on sys.path -- failing a normal `import
    slr_pipeline.run_identity` with only the repo root on sys.path."""
    import slr_pipeline.run_identity as ri
    assert ri.resolve_review_run_id is resolve_review_run_id


def test_state_only_allow_config_true_returns_state_value():
    assert resolve_review_run_id({"review_run_id": "run-abc"}, allow_config=True) == "run-abc"


def test_absent_review_run_id_allow_config_true_raises_naming_both_sources():
    with pytest.raises(ValueError, match="thread_id"):
        resolve_review_run_id({"review_run_id": None}, allow_config=True)


def test_absent_review_run_id_allow_config_false_raises_without_touching_get_config():
    with pytest.raises(ValueError, match="review_run_id"):
        resolve_review_run_id({"review_run_id": None}, allow_config=False)


def test_allow_config_false_never_calls_get_config(monkeypatch):
    """A resolver called between graph invocations (resolve_full_text_review_queue())
    must never touch get_config() at all -- there is no active runnable
    context to read one from."""
    import slr_pipeline.run_identity as ri

    def _boom():
        raise AssertionError("get_config() must not be called when allow_config=False")
    monkeypatch.setattr(ri, "get_config", _boom)
    assert resolve_review_run_id({"review_run_id": "run-xyz"}, allow_config=False) == "run-xyz"


def test_unsafe_token_rejected_state_only():
    with pytest.raises(ValueError, match="safe token"):
        resolve_review_run_id({"review_run_id": "../escaped"}, allow_config=False)


def test_unsafe_token_rejected_allow_config_true():
    with pytest.raises(ValueError, match="safe token"):
        resolve_review_run_id({"review_run_id": "not a valid token!"}, allow_config=True)


def test_get_config_runtime_error_outside_runnable_context_is_not_fatal_by_itself():
    """Calling resolve_review_run_id(allow_config=True) directly in a unit
    test (no active LangGraph invocation) -- get_config() raises
    RuntimeError, which must be caught and treated as "no config source,"
    not propagate as-is, when state already has a value."""
    assert resolve_review_run_id({"review_run_id": "run-direct-call"}, allow_config=True) == "run-direct-call"


def test_mismatch_between_state_and_config_raises(monkeypatch):
    import slr_pipeline.run_identity as ri
    monkeypatch.setattr(ri, "get_config", lambda: {"configurable": {"thread_id": "run-from-config"}})
    with pytest.raises(ValueError, match="mismatch"):
        resolve_review_run_id({"review_run_id": "run-from-state"}, allow_config=True)


def test_config_only_source_used_when_state_has_none(monkeypatch):
    import slr_pipeline.run_identity as ri
    monkeypatch.setattr(ri, "get_config", lambda: {"configurable": {"thread_id": "run-from-config"}})
    assert resolve_review_run_id({"review_run_id": None}, allow_config=True) == "run-from-config"


def test_identical_state_and_config_values_do_not_raise(monkeypatch):
    import slr_pipeline.run_identity as ri
    monkeypatch.setattr(ri, "get_config", lambda: {"configurable": {"thread_id": "same-run"}})
    assert resolve_review_run_id({"review_run_id": "same-run"}, allow_config=True) == "same-run"
