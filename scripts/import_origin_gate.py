"""Track 2 import-origin gate -- pure logic, no `pytest` dependency.

track2_pipeline_readiness_plan_2026-08-30.md step 2 (round-17/18): asserts
`slr_pipeline`'s resolved import origin is never inside this checkout while
`TRACK2_REQUIRE_WHEEL_ORIGIN=1` is set. Kept dependency-free (no `pytest`
import) so it can also run standalone inside an isolated install venv that
only has the pinned build toolchain (`track2/build-requirements.lock`), not
the dev/test dependencies -- `tests/conftest.py`'s pytest fixture wraps this
same function for the real, in-process test-suite gate.
"""
from __future__ import annotations

import inspect
import os
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class ImportOriginError(RuntimeError):
    pass


def import_origin_gate(repo_root: Path = REPO_ROOT) -> None:
    if os.environ.get("TRACK2_REQUIRE_WHEEL_ORIGIN") != "1":
        return
    import slr_pipeline

    origin = Path(inspect.getfile(slr_pipeline)).resolve()

    # CI's actual layout (`uv venv .venv` run inside the checkout) puts the
    # venv's site-packages *underneath* repo_root, not beside it -- a plain
    # "is origin outside repo_root" check (the previous logic) rejects that
    # real wheel install just as hard as it rejects the checkout source
    # (round-19 finding: reproduced by a repository-local-venv regression).
    # Check the running interpreter's own purelib/platlib first: any origin
    # under either is a real site-packages install, wherever it physically
    # sits, and is accepted regardless of repo_root.
    site_package_dirs = {
        Path(sysconfig.get_paths()["purelib"]).resolve(),
        Path(sysconfig.get_paths()["platlib"]).resolve(),
    }
    for site_dir in site_package_dirs:
        try:
            origin.relative_to(site_dir)
        except ValueError:
            continue
        return

    # Not a site-packages install by the check above -- reject only an
    # origin that actually resolves inside the checkout's own source tree.
    source_tree = (repo_root / "slr_pipeline").resolve()
    try:
        origin.relative_to(source_tree)
    except ValueError:
        return  # neither a known site-packages dir nor the checkout source -- accept
    raise ImportOriginError(
        f"slr_pipeline imported from repo checkout ({origin}) while "
        "TRACK2_REQUIRE_WHEEL_ORIGIN=1 -- CI must install it from a wheel, "
        "never resolve the checkout copy."
    )
