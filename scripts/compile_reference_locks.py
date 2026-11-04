"""Compile a `track2/<name>.in` requirements file into its hash-locked
`track2/<name>.lock` via `uv pip compile`.

track2_pipeline_readiness_plan_2026-08-30.md §5.1: each of Track 2's three
locks (`build-requirements`, `base-requirements`, `ft-screening-wheel`) is
generated the same way, at its own owning step, never hand-edited. This
script is the one shared generator for the `.in` -> `.lock` locks (the
`ft-screening-wheel.lock`, a bare `{filename, sha256}` pair, is instead
written by `build_ft_screening_wheel.py --write-lock`, step 5 -- not this
script).

Usage:
    python scripts/compile_reference_locks.py <name>

`<name>` is the lock's basename under `track2/`, e.g. `build-requirements`
reads `track2/build-requirements.in` and writes
`track2/build-requirements.lock`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACK2_DIR = REPO_ROOT / "track2"

PYTHON_VERSION = "3.12.12"
PYTHON_PLATFORM = "x86_64-pc-windows-msvc"


def compile_lock(name: str, *, uv_executable: str = "uv", python_executable: str | None = None) -> Path:
    """Compile `track2/<name>.in` into `track2/<name>.lock`. Returns the
    written lock file's path.

    `python_executable`, if given, is passed as `--python` -- a concrete
    interpreter uv uses to build sdist metadata for any package this
    lock's `.in` pulls in that has no wheel for `PYTHON_PLATFORM`. This is
    distinct from `--python-version`/`--python-platform` below, which only
    set the resolution's *target* platform tags; a real interpreter is
    still needed locally to run build backends."""
    in_path = TRACK2_DIR / f"{name}.in"
    if not in_path.exists():
        raise FileNotFoundError(f"no such requirements input: {in_path}")
    out_path = TRACK2_DIR / f"{name}.lock"
    cmd = [
        uv_executable, "pip", "compile",
        "--generate-hashes",
        "--python-version", PYTHON_VERSION,
        "--python-platform", PYTHON_PLATFORM,
        "--output-file", str(out_path),
        str(in_path),
    ]
    if python_executable:
        cmd[3:3] = ["--python", python_executable]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="lock basename under track2/, e.g. build-requirements")
    parser.add_argument("--uv", default="uv", help="uv executable to invoke (default: 'uv' on PATH)")
    parser.add_argument("--python", default=None, help="concrete interpreter for build-backend metadata")
    args = parser.parse_args(argv)

    out_path = compile_lock(args.name, uv_executable=args.uv, python_executable=args.python)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
