"""Track 2 Step 12A, sub-step (c) -- authors `tests/reference_run/fixtures/
prisma_scr_matrix.json`: a static transcription of `track2_pipeline_
readiness_plan_2026-08-30.md` §9's frozen 22-row PRISMA-ScR item matrix.

This file lives at the fixtures tree's own root (`tests/reference_run/
fixtures/`), not under `fixtures/corpus/` -- plan §9/§11's own citations of
this exact path, and round-14 finding 5's correction of an earlier draft
that wrongly placed it under `fixtures/spec/`. Because of that, it is
genuinely independent of `fixture_corpus_digest`'s "every file under
corpus/" ordering constraint (§7.1) and can be authored any time before
step 12(f) hashes it.

Every row is static descriptive prose (which run-state fields feed this
PRISMA-ScR item, and what rule decides `reported` vs
`optional_not_conducted`) transcribed verbatim from the plan's own table
-- not a value computed from a real run. `slr_pipeline/prisma_scr.py`
(step 14) is the single canonical source for the table itself (`ITEMS`/
`PRISMA_SCR_VERSION`/`PRISMA_SCR_MATRIX_VERSION`, plus the renderers that
actually consume it at runtime); this script imports from there rather
than keeping its own copy, the same anti-duplication move step 13 made
for `reference_pdf_placement.py`. `tests/test_prisma_scr.py::
test_prisma_matrix_byte_equality` cross-checks this fixture against that
module's own `matrix_payload()`.

Usage:
    python scripts/author_prisma_scr_matrix.py --write
    python scripts/author_prisma_scr_matrix.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Direct invocation (`python scripts/author_prisma_scr_matrix.py`) sets
# `sys.path[0]` to this file's own directory (`scripts/`), not the repo
# root, so `scripts` (this file's own parent package, needed below for
# `from scripts.hash_reference_inputs import register_target`) would not
# otherwise be importable -- unlike `slr_pipeline`, which is pip-installed
# and importable regardless of invocation style.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slr_pipeline.prisma_scr import matrix_text

MATRIX_PATH = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "prisma_scr_matrix.json"


def write_matrix() -> None:
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.write_text(matrix_text(), encoding="utf-8")


def check_matrix() -> None:
    on_disk = MATRIX_PATH.read_text(encoding="utf-8")
    expected = matrix_text()
    if on_disk != expected:
        raise ValueError(f"{MATRIX_PATH} does not match the authored matrix content (re-run --write)")


def _register_targets() -> None:
    from scripts.hash_reference_inputs import register_target

    @register_target("prisma_scr_matrix_sha256")
    def _prisma_scr_matrix_target() -> Path:
        return MATRIX_PATH


_register_targets()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        write_matrix()
        print(f"wrote {MATRIX_PATH.relative_to(REPO_ROOT)}")
    else:
        check_matrix()
        print("OK: prisma_scr_matrix.json matches authored content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
