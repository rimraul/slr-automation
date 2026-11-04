"""Proof-of-mechanism that the shared offline contract (`chapter_offline_
contract_plugin.py`) is genuinely active for THIS directory tree too, not
only under `tests/chapter_examples/` -- round-4 review finding 1: Codex
reproduced that `examples/chapter/` (where real E.2b exercise/solution files
are designed to live) had zero credential/network protection before this
fix, since the enforcement conftest previously lived only under
`tests/chapter_examples/`.

NOT a real teaching exercise -- Phase E.2a's own infrastructure-verification
deliverable, mirroring (not duplicating in full) the equivalent proof already
established in `tests/chapter_examples/test_offline_contract_smoke.py`. Full
mechanism history lives in `chapter_offline_contract_plugin.py`'s own
docstring.
"""

from __future__ import annotations

import os
import socket

import pytest
import pytest_socket


def test_a_core_example_here_never_sees_a_provider_credential():
    """By the time any ordinary (non-`live`) test body under `examples/
    chapter/` runs, the autouse `_chapter_example_offline_contract` fixture
    (loaded via this directory's own `conftest.py` -> the shared plugin) has
    already cleared every provider credential env var."""
    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_a_core_example_here_attempting_real_network_access_is_blocked():
    """Proves the network-blocking half of the offline contract is active
    for a test collected under `examples/chapter/`, not only under
    `tests/chapter_examples/`. `socket.create_connection` is what `requests`/
    urllib3 actually call internally, so this exercises the same code path a
    real accidental HTTP call would take."""
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.create_connection(("example.com", 80), timeout=2)


@pytest.mark.live
def test_a_live_marked_test_here_is_still_gated_by_the_shared_opt_in(request: pytest.FixtureRequest):
    """Proves the `live` marker's auto-skip-unless-`--run-live-chapter`
    contract is enforced for this tree too -- plain `pytest examples/chapter`
    (no flags) must report this test SKIPPED, never executed."""
    assert request.config.getoption("--run-live-chapter") is True, (
        "this test must never actually execute its body unless --run-live-chapter "
        "was passed -- if you are seeing this failure, the shared live/offline "
        "auto-skip contract is broken for the examples/chapter/ tree specifically"
    )
