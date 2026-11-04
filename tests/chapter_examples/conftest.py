"""Test infrastructure for the chapter's reader-facing teaching examples
(Track 1 Phase E.2a/E.2b, see chapter_teaching_design.md).

Phase E.2a (this phase) only proves the offline/live marker contract works
(see test_offline_contract_smoke.py) -- no real exercise fixtures live here
yet. Phase E.2b adds the actual per-section fixtures once the exercises
themselves are built under examples/chapter/.

Round-4 review finding 1: all the actual enforcement logic (CLI options, the
`live`/`slow` auto-skip, credential-clearing, network-blocking) has moved to
`chapter_offline_contract_plugin.py` at the repo root, a plain shared module
BOTH this file and `examples/chapter/conftest.py` import from -- Codex found
that `examples/chapter/`, where the real exercise files are designed to live,
had none of this protection at all (`pytest examples/chapter
--run-slow-chapter --collect-only` failed with "unrecognized arguments"
because the options were never registered there). See that module's own
docstring for the full revision history (rounds 1-3) this file used to carry
directly, before the extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# append, not insert(0, ...): chapter_offline_contract_plugin is a
# repo-root-only module with no installed-package equivalent, so it is
# fine to fall back to the checkout for it -- but never ahead of an
# installed slr_pipeline (wheel or -e .), which is what insert(0, ...)
# at the front of sys.path would do (round-16/17 foreign-cwd contract,
# see track2_pipeline_readiness_plan_2026-08-30.md §2.3).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

# Re-exported so pytest's own conftest hook/fixture discovery (plain
# attribute lookup on this module's namespace) finds them here. Tests import
# shared helpers from chapter_offline_contract_plugin directly so they also
# work under Track 2 CI's `--import-mode=importlib` foreign-CWD invocation.
from chapter_offline_contract_plugin import (  # noqa: F401,E402
    _chapter_example_offline_contract,
    _clear_provider_credentials,
    pytest_addoption,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_setup,
    pytest_unconfigure,
)
