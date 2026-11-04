"""Loads the same shared offline-contract plugin as `tests/chapter_examples/
conftest.py` (see `chapter_offline_contract_plugin.py` at the repo root for
the full rationale/history) -- so a direct per-exercise run under this
directory (`pytest examples/chapter/...`), where the real E.2b exercise/
solution files are designed to live per the code-source-of-truth convention,
gets identical credential-clearing/network-blocking/live-slow-marker-skip
protection to a run under `tests/chapter_examples/`. One shared source of
truth, never two independently-maintained copies that could drift apart
(round-4 review finding 1)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chapter_offline_contract_plugin import (  # noqa: F401,E402
    _chapter_example_offline_contract,
    _clear_provider_credentials,
    pytest_addoption,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_setup,
    pytest_unconfigure,
)
