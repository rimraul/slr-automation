"""Root pytest conftest -- Track 2 import-origin gate.

track2_pipeline_readiness_plan_2026-08-30.md step 2 (round-17/18): asserts
`slr_pipeline`'s resolved import origin is never inside this checkout while
`TRACK2_REQUIRE_WHEEL_ORIGIN=1` is set -- CI's `unit`/`acceptance` jobs
(§5.5) set it, so a wheel-install-only slip is caught. Local `pip install
-e .` development leaves the variable unset, so the gate is a no-op there --
an editable install's whole *purpose* is to make `slr_pipeline` resolve
inside REPO_ROOT, so an unconditional gate would fail every local run.

The gate's actual logic lives in `scripts/import_origin_gate.py` (kept
`pytest`-free so it can also run standalone inside an isolated install venv
that only has the pinned build toolchain, not the dev/test dependencies).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# append, not insert(0, ...): scripts/ (and chapter_offline_contract_plugin,
# see tests/chapter_examples/conftest.py) is a repo-root-only module tree
# with no installed-package equivalent, so it is fine to fall back to the
# checkout for it -- but never ahead of an installed slr_pipeline (wheel or
# -e .), which is what insert(0, ...) at the front of sys.path would do
# (track2_pipeline_readiness_plan_2026-08-30.md §2.3 foreign-cwd contract).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.import_origin_gate import ImportOriginError, import_origin_gate  # noqa: E402

# re-exported under the names the plan/tests refer to
_import_origin_gate = import_origin_gate


@pytest.fixture(scope="session", autouse=True)
def _track2_import_origin_gate():
    _import_origin_gate()
    yield
