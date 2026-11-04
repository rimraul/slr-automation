"""Shared pytest plugin enforcing the chapter teaching examples' offline-by-
default contract (Track 1 Phase E.2a, see chapter_teaching_design.md).

Round-4 review finding 1: this logic previously lived ONLY inside
`tests/chapter_examples/conftest.py` -- but the actual `test_solution.py`-style
exercise files are designed (per the code-source-of-truth convention) to live
under `examples/chapter/`, a SEPARATE directory tree pytest does not share a
conftest.py with by default. Codex reproduced the gap directly: running
`pytest examples/chapter --run-slow-chapter --collect-only` failed with
"unrecognized arguments" because the CLI options/marker-skip/credential-
clearing/network-blocking hooks were never loaded for that tree at all -- a
direct per-exercise run (or even collection ordering during a root-wide run)
could happen with zero offline protection.

Fix: extract every hook/helper into this ONE shared, plain (non-conftest)
module at the repo root, importable by both `examples/chapter/conftest.py`
and `tests/chapter_examples/conftest.py` alike -- so both trees enforce
identically, from one source of truth, never two independently-maintained
copies that could silently drift apart. A conftest.py's hook functions
(`pytest_addoption`, `pytest_configure`, etc.) and fixtures are recognized by
pytest via ordinary attribute lookup on the conftest module's own namespace --
importing them by name into that namespace (`from chapter_offline_contract_
plugin import pytest_configure, ...`) makes pytest see them exactly as if they
had been defined there directly.
"""

from __future__ import annotations

import os
import weakref

import pytest
import pytest_socket

# ---------------------------------------------------------------------------
# Auto-skip machinery for the `live` and `slow` markers -- two independent
# opt-in flags, since "requires real network/credentials/spend" (live) and
# "just slow" (bertopic import) are different kinds of risk and a reader may
# reasonably want to opt into one without the other.
# ---------------------------------------------------------------------------

_AUTO_SKIP_MARKERS = {
    "live": (
        "--run-live-chapter",
        "live chapter-example test -- requires real network/credentials/spend; "
        "never run by default. Pass --run-live-chapter to opt in explicitly.",
    ),
    "slow": (
        "--run-slow-chapter",
        "slow chapter-example test (e.g. `import bertopic` alone takes ~25s-5min "
        "depending on environment) -- excluded from the default quick-smoke path. "
        "Pass --run-slow-chapter to opt in explicitly.",
    ),
}


# Round-5 review finding 1: the previous version caught `ValueError`
# unconditionally around `parser.addoption(...)` to swallow the known "already
# registered" case (both `examples/chapter/conftest.py` and `tests/
# chapter_examples/conftest.py` import this same function into a single pytest
# session) -- but that also silently hides a GENUINELY different problem: a
# real conflicting plugin, an invalid option definition, or a future typo in
# this file's own `add_argument` calls would raise the identical exception
# type and be swallowed just as silently. Fixed with an explicit once-per-
# parser registration guard instead: track which `parser` objects this
# function has already registered options on, and simply skip re-registering
# for a parser already in that set -- never attempt-then-catch. Any
# `ValueError` that still escapes past this guard is now something genuinely
# unexpected and is allowed to propagate/raise normally.
_PARSERS_WITH_OPTIONS_REGISTERED: "weakref.WeakSet[pytest.Parser]" = weakref.WeakSet()


def pytest_addoption(parser: pytest.Parser) -> None:
    if parser in _PARSERS_WITH_OPTIONS_REGISTERED:
        return
    _PARSERS_WITH_OPTIONS_REGISTERED.add(parser)
    for marker, (flag, help_text) in _AUTO_SKIP_MARKERS.items():
        parser.addoption(flag, action="store_true", default=False, help=help_text)


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    for marker, (flag, help_text) in _AUTO_SKIP_MARKERS.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=help_text)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# The offline contract itself: credential-clearing + network-blocking,
# enforced process-wide from before collection begins, not only per-test.
# ---------------------------------------------------------------------------

# Every provider-credential env var this project's nodes read from, per
# config.py/search.py -- kept as one explicit list so it's obvious what's
# covered (and what to add to if a new provider integration appears).
_PROVIDER_CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PUBMED_API_KEY",
    "MEDPUB_API_KEY",
    "SPRINGER_API_KEY",
    "IEEE_API_KEY",
    "S2_API_KEY",
    "S2_API_Key",
)


def _clear_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test variant, using `monkeypatch` (auto-restored at test teardown).
    Extracted as its own function so a smoke test can exercise it directly
    against a deliberately-set sentinel value, proving it actually clears an
    existing credential rather than merely observing one that was never set."""
    for var in _PROVIDER_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


_saved_env_for_unconfigure: dict[str, str] = {}


def _clear_provider_credentials_process_wide() -> None:
    """Session-wide variant, used in `pytest_configure` -- runs before
    collection/import begins, so no test module (including a real E.2b
    exercise doing something at its own module level) can ever observe a
    real credential during this session. Saves original values so
    `pytest_unconfigure` can restore them, leaving the wider environment
    (e.g. a monorepo test run covering other suites too) undisturbed."""
    for var in _PROVIDER_CREDENTIAL_ENV_VARS:
        if var in os.environ:
            _saved_env_for_unconfigure[var] = os.environ.pop(var)


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--run-live-chapter"):
        # Live mode explicitly requested -- at least one test in this session
        # needs the real environment; do not sanitize it process-wide. Core
        # (non-live) tests running in the same session still get the
        # per-test autouse fixture below, an acceptable narrower guarantee
        # for this explicitly-opted-into-real-calls session.
        return
    _clear_provider_credentials_process_wide()
    pytest_socket.disable_socket(allow_unix_socket=False)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Re-applies the network block before EVERY test, not just once at
    session start. Necessary because `pytest-socket`'s OWN plugin registers a
    `pytest_runtest_teardown` hook that unconditionally calls
    `_remove_restrictions()` after every single test (its own internal
    cleanup, unrelated to any of our markers/options). `pytest_configure`
    alone still correctly protects the entire collection/import phase
    (nothing tears anything down before the first test runs); this hook
    additionally keeps every individual test's own execution protected too."""
    if "live" in item.keywords:
        return
    pytest_socket.disable_socket(allow_unix_socket=False)


def pytest_unconfigure(config: pytest.Config) -> None:
    for var, value in _saved_env_for_unconfigure.items():
        os.environ[var] = value
    _saved_env_for_unconfigure.clear()
    pytest_socket.enable_socket()


@pytest.fixture(autouse=True)
def _chapter_example_offline_contract(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Per-test credential-clearing -- kept alongside the process-wide
    `pytest_configure` guard above as a defensive second layer (belt-and-
    suspenders): even if a future change ever runs this file's tests outside
    a normal `pytest_configure` lifecycle, individual tests are still
    protected. Network blocking itself is NOT duplicated here -- it's a
    single process-wide `pytest_socket.disable_socket()` call already made in
    `pytest_configure`/`pytest_runtest_setup` for the whole non-live session.
    A `live`-marked test (which only ever runs under the explicit
    `--run-live-chapter` opt-in, per the skip logic above) is exempt -- it
    needs the real network/credentials by definition."""
    if "live" in request.keywords:
        yield
        return
    _clear_provider_credentials(monkeypatch)
    yield
