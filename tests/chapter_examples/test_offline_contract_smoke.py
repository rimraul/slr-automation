"""Proof-of-mechanism for the offline-by-default testing contract
(chapter_teaching_design.md Section 6/7) -- NOT a real teaching exercise.

This is Phase E.2a's own deliverable: proving the `live` marker/skip
mechanism, the credential-clearing/network-blocking enforcement, and the
BERTopic offline strategy all actually work, before Phase E.2b builds any
real exercise on top of them. If this file's own tests didn't behave as
asserted below, every "Core" row in the teaching-design's example inventory
would be resting on an unverified promise.

Round-2 review (blocking findings 1-2) added everything below except
`test_a_normal_offline_example_always_runs_and_passes`, which predates it.

Round-3 review (blocking findings 1, 2, 4) fixed the live sentinel's
assertion (it previously failed unconditionally, so even the CORRECT opt-in
invocation reported red, not green), added the module-level import-time
proof, and moved the BERTopic test under its own `slow` marker (see
conftest.py's module docstring for why).

Round-5 review (finding 2) made the collection-time regression actually
deterministic: the network-blocking assertion now checks the exact exception
type (`pytest_socket.SocketBlockedError`), not just "any exception happened"
(which would have also accepted a DNS failure/timeout/firewall denial as a
false-positive proof), and a new subprocess-based test proves credential-
clearing removes a REAL pre-existing value rather than merely observing one
that was already absent in the ambient environment.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_socket

from chapter_offline_contract_plugin import _clear_provider_credentials

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Round-4 review finding 2: this import must happen HERE, at this file's own
# module (top) level -- not inside a test function body -- so it executes
# during pytest's COLLECTION phase (when pytest imports this file to discover
# its tests), which happens strictly after `pytest_configure` but BEFORE any
# `pytest_runtest_setup` hook or per-test autouse fixture has ever run for
# ANY test in the session. The prior version imported it inside the test
# function instead, which meant it ran AFTER those per-test protections were
# already active for that specific test -- so the test would have kept
# passing even if `pytest_configure`'s collection-time-specific protection
# were completely removed, without actually proving that protection exists.
# Verified during this fix: temporarily commenting out `pytest_configure`'s
# body in chapter_offline_contract_plugin.py made
# `test_module_level_network_and_credential_access_is_blocked_even_at_import_time`
# below fail (network call succeeded, credential value non-None) -- proving
# this import genuinely exercises the collection-time guard, not a
# coincidentally-already-protected later stage.
_probe_path = Path(__file__).with_name("module_level_network_probe.py")
_probe_spec = importlib.util.spec_from_file_location(
    "chapter_module_level_network_probe", _probe_path
)
assert _probe_spec is not None and _probe_spec.loader is not None
module_level_network_probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(module_level_network_probe)


def test_a_normal_offline_example_always_runs_and_passes():
    """Stands in for a real 'Core' exercise's test -- must run and pass
    under the default offline command."""
    assert 1 + 1 == 2


@pytest.mark.live
def test_a_live_example_passes_only_when_explicitly_opted_into(request: pytest.FixtureRequest):
    """Stands in for the one 'optional live' variant (S7's real-API-key
    rerun). Round-3 review finding 1: this test previously called
    `pytest.fail(...)` unconditionally, so even the documented, CORRECT
    opt-in command (`--run-live-chapter`) reported a failure -- a reader
    following the docs would see red on the one path that's supposed to
    work. Fixed: this now asserts `--run-live-chapter` was genuinely active
    (a real backstop, not a decorative failure) and otherwise passes.
    Running plain `pytest tests/chapter_examples/` (no flags at all) must
    still report this test SKIPPED, never executed -- confirmed by
    conftest.py's own `pytest_collection_modifyitems` auto-skip, which
    applies unconditionally unless `--run-live-chapter` is passed."""
    assert request.config.getoption("--run-live-chapter") is True, (
        "this test must never actually execute its body unless --run-live-chapter "
        "was passed -- if you are seeing this failure, the live/offline auto-skip "
        "contract is broken (the test ran without the flag that's supposed to gate it)"
    )


def test_clear_provider_credentials_actually_clears_a_set_value(monkeypatch):
    """Round-2 review finding 2: proves `_clear_provider_credentials()` (the
    function the autouse fixture calls for every non-`live` test) genuinely
    REMOVES an existing credential, not merely that it happens to observe one
    that was already unset. Deliberately sets a sentinel via `monkeypatch`
    first, then calls the clearing function directly and unambiguously --
    independent of pytest's own fixture-execution ordering, which would
    otherwise make "was it cleared, or never set" impossible to distinguish
    from inside an ordinary autouse-fixture-covered test."""
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-real-looking-key-should-be-cleared")
    assert os.environ.get("OPENAI_API_KEY") == "sentinel-real-looking-key-should-be-cleared"
    _clear_provider_credentials(monkeypatch)
    assert os.environ.get("OPENAI_API_KEY") is None


def test_a_core_test_never_sees_a_provider_credential():
    """Integration-level check (complements the unit-level proof above): by
    the time any ordinary, non-`live` chapter-example test body runs, the
    autouse `_chapter_example_offline_contract` fixture has already cleared
    every provider credential env var -- regardless of what the real
    environment had set (e.g. `my_api_keys.py`'s own documented import-time
    side effect, tracked as a separate hardening item)."""
    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_a_core_test_attempting_real_network_access_is_blocked():
    """Round-2 review finding 2 (mechanism replaced round 3, finding 2):
    proves the network-blocking half of the offline contract. Any attempt to
    open a real socket connection from a 'core' (non-`live`) test raises
    immediately, rather than silently succeeding via a real network call the
    test should have mocked instead. Round 3 switched from a hand-patched
    `socket.socket.connect` (which left `connect_ex`/UDP `sendto`/etc. open)
    to `pytest-socket`'s `disable_socket()`, applied process-wide in
    `conftest.py::pytest_configure` -- comprehensive, well-established
    network denial rather than patching individual methods one at a time.
    `socket.create_connection` (what `requests`/urllib3 actually call
    internally) is used here so this exercises the same code path a real
    accidental HTTP call would take. Raises `SocketBlockedError` specifically (not
    `SocketConnectBlockedError`) because `create_connection()` calls `socket.getaddrinfo()`
    for DNS resolution BEFORE it ever reaches a `connect()` call -- confirmed by testing
    `pytest_socket.disable_socket()` standalone outside pytest during this fix."""
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.create_connection(("example.com", 80), timeout=2)


def test_module_level_network_and_credential_access_is_blocked_even_at_import_time():
    """Round-3 review finding 2 (import mechanism fixed round 4): the offline
    contract must be active BEFORE any test module is even imported, not
    merely once a per-test fixture's setup phase runs -- otherwise a real
    E.2b exercise module doing something network/credential-touching at ITS
    OWN module level (e.g. constructing a real client, or reading a key, at
    import time in a shared fixture module) would leak before any per-test
    guard existed.

    Round-4 review finding 2: the previous version of this test imported
    `module_level_network_probe` INSIDE this function body -- by then,
    `pytest_runtest_setup` and the autouse fixture had already run for THIS
    test, so the assertion below would have kept passing even with
    `pytest_configure`'s collection-time protection completely removed (the
    per-test guards alone would have been enough to protect an in-body
    import). Fixed by moving the import to this FILE's own top level (see
    above) -- `module_level_network_probe`'s module-level network/credential
    access therefore now runs during pytest's COLLECTION of this file
    itself, strictly before any `pytest_runtest_setup`/autouse fixture has
    ever executed in the session, genuinely isolating and proving
    `pytest_configure`'s specific contribution. This function now only
    asserts on the outcome already captured at that collection-time
    import.

    Round-5 review finding 2 (part 1): the previous version accepted ANY
    exception type as proof of "blocked" -- a DNS failure, timeout, firewall
    denial, or plain connection refusal (any of which can occur in a
    sandboxed/offline environment with zero involvement from pytest-socket)
    would have been wrongly accepted as proof THIS mechanism specifically
    did the blocking. Fixed: asserts the exact exception type is
    `pytest_socket.SocketBlockedError`; any other exception type now fails
    this test loudly, with the real type/message shown, instead of silently
    passing."""
    exc = module_level_network_probe.IMPORT_TIME_NETWORK_EXCEPTION
    assert exc is not None, "network call succeeded during collection -- should have been blocked"
    assert isinstance(exc, pytest_socket.SocketBlockedError), (
        f"blocked by an unexpected mechanism, not pytest-socket specifically: {type(exc).__name__}: {exc}"
    )
    assert module_level_network_probe.IMPORT_TIME_CREDENTIAL_VALUE is None


def test_module_level_credential_clearing_removes_a_real_preexisting_credential_subprocess(
    tmp_path: Path,
):
    """Round-5 review finding 2 (part 2): the test above only asserts
    `IMPORT_TIME_CREDENTIAL_VALUE is None` -- which holds trivially in an
    environment where `OPENAI_API_KEY` was never set to begin with, so it
    never actually proved `_clear_provider_credentials_process_wide()`
    removes anything. Made deterministic by launching a genuinely separate
    pytest SUBPROCESS with a distinctive, obviously-fake credential value
    explicitly set in THAT subprocess's own environment before its
    collection begins -- the subprocess runs the exact same collection-time
    assertion above against a REAL starting value this time. If credential-
    clearing were removed, that subprocess's own assertion would then fail
    (observing the sentinel instead of `None`), and this outer test would
    catch it via the subprocess's nonzero exit code."""
    sentinel = "sk-test-sentinel-DO-NOT-USE-9f3a7c21b1e04d5a"
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = sentinel

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "--import-mode=importlib",
            f"{_REPO_ROOT / 'tests' / 'chapter_examples' / 'test_offline_contract_smoke.py'}::"
            "test_module_level_network_and_credential_access_is_blocked_even_at_import_time",
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"nested pytest subprocess failed with a real OPENAI_API_KEY={sentinel!r} pre-set in its "
        f"environment -- credential-clearing did not remove a real starting value:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

# The Phase E.2a BERTopic offline-strategy smoke test that used to live here
# (`test_bertopic_runs_fully_offline_with_precomputed_embeddings`) was removed
# in Phase E.2b: it is superseded by
# `examples/chapter/s11_synthesis_topics/test_s11_synthesis_topics.py::
# test_bertopic_offline_deterministic`, which exercises the same strategy on
# the real 12-document teaching corpus and its committed precomputed
# embeddings, with the determinism assertion the smoke test lacked. See the
# "## Erratum (Phase E.2b)" section of chapter_teaching_design.md.
