"""Build (and, eventually, lock/verify) the `slr-pipeline` wheel.

track2_pipeline_readiness_plan_2026-08-30.md §0.4/§2.2/§13 step 19: this
script owns two release artifacts that are committed **at step 19 only**,
once CN's code is final --

    track2/SLR_PIPELINE_WHEEL.lock              (--write-lock)
    tests/reference_run/fixtures/wheel_contents.txt   (--write-contents)

and one CI check, run from step 19's `acceptance` job onward --

    --check <wheel path>   verifies a just-built wheel's {filename, sha256}
                            against SLR_PIPELINE_WHEEL.lock and its exact
                            `zipfile.namelist()` against wheel_contents.txt.

At step 2 (this commit), neither lock file exists yet, so only the plain
build step below is exercised -- by `test_wheel_licence_scope`
(`tests/test_track2_packaging.py`), which builds a wheel via `build_wheel()`
in a temporary directory (a "foreign cwd") and inspects its licence
metadata directly, with no dependency on either not-yet-committed file.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "track2" / "SLR_PIPELINE_WHEEL.lock"
CONTENTS_PATH = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "wheel_contents.txt"


def _assert_canonical_working_tree() -> None:
    """Guard against the 2026-09-13 CRLF incident: `git add`/commit runs the
    `eol=lf` clean filter and stores an LF blob, but git never rewrites an
    already-tracked working-tree file back to LF on its own -- only a fresh
    checkout (re-)applies the smudge filter. A long-lived Windows working
    copy can therefore carry CRLF bytes on disk for a file whose committed
    (and CI-checked-out) blob is pure LF, with `git status`/`git diff`
    reporting no change (both normalize through the same clean filter
    before comparing). `build_wheel()` zips up raw bytes straight off disk,
    so it would silently package those CRLF bytes and freeze a wheel
    sha256 that no canonical checkout (including CI's) can ever reproduce.
    Compares working-tree bytes directly against the git INDEX blob (not
    HEAD, so a legitimate staged-but-uncommitted edit is not flagged) for
    every tracked `slr_pipeline/**/*.py` file, bypassing git's own
    comparison filters entirely."""
    all_tracked = subprocess.run(
        ["git", "ls-files", "--", "slr_pipeline"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    # `git ls-files -- "slr_pipeline/**/*.py"` under-matches (its pathspec
    # glob requires at least one directory between `slr_pipeline/` and the
    # filename, silently skipping top-level files like `state.py`,
    # `track3_run.py`) even though `git check-attr` confirms the
    # .gitattributes `eol=lf` PATTERN (a different, gitignore-style
    # matcher) does cover them -- filter by suffix directly instead.
    tracked = [p for p in all_tracked if p.endswith(".py")]
    offenders = []
    for rel_path in tracked:
        working_bytes = (REPO_ROOT / rel_path).read_bytes()
        indexed_bytes = subprocess.run(
            ["git", "show", f":{rel_path}"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        ).stdout
        if working_bytes != indexed_bytes:
            offenders.append(rel_path)
    if offenders:
        raise RuntimeError(
            "refusing to build/lock the slr_pipeline wheel: the following "
            "tracked file(s) have working-tree bytes that differ from git's "
            "own index blob (e.g. a stale CRLF working copy despite an "
            "`eol=lf` .gitattributes pin) -- a wheel built from these bytes "
            "would not match a canonical `git archive`/checkout build, "
            "including CI's own: " + ", ".join(offenders) + ". Rewrite "
            "each file's line endings to match its index blob (or "
            "`git checkout -- <path>` after removing and re-adding it) "
            "before building."
        )


def build_wheel(dist_dir: Path, *, python_executable: str = sys.executable) -> Path:
    """`python -m build --wheel --no-isolation` into `dist_dir`, matching
    the CI `unit`/`acceptance` jobs' own build step (§5.5) exactly. Returns
    the single built wheel's path."""
    _assert_canonical_working_tree()
    dist_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [python_executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(dist_dir)],
        cwd=REPO_ROOT, check=True,
    )
    wheels = sorted(dist_dir.glob("slr_pipeline-*.whl"))
    if not wheels:
        raise RuntimeError(f"build produced no slr_pipeline-*.whl under {dist_dir}")
    if len(wheels) > 1:
        raise RuntimeError(f"build produced more than one wheel under {dist_dir}: {wheels}")
    return wheels[0]


def wheel_sha256(wheel_path: Path) -> str:
    return hashlib.sha256(wheel_path.read_bytes()).hexdigest()


def wheel_namelist(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as zf:
        return sorted(zf.namelist())


def _write_lock(wheel_path: Path) -> None:
    """**Step 19 only** -- not invoked by this commit."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        f"filename = {wheel_path.name!r}\nsha256 = {wheel_sha256(wheel_path)!r}\n",
        encoding="utf-8",
    )


def _write_contents(wheel_path: Path) -> None:
    """**Step 19 only** -- not invoked by this commit."""
    CONTENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTENTS_PATH.write_text("\n".join(wheel_namelist(wheel_path)) + "\n", encoding="utf-8")


def _check(wheel_path: Path) -> None:
    """**Step 19+ only** -- both `track2/SLR_PIPELINE_WHEEL.lock` and
    `tests/reference_run/fixtures/wheel_contents.txt` must exist; this
    commit does not create either."""
    if not LOCK_PATH.exists():
        raise FileNotFoundError(f"{LOCK_PATH} does not exist yet (written at step 19)")
    if not CONTENTS_PATH.exists():
        raise FileNotFoundError(f"{CONTENTS_PATH} does not exist yet (written at step 19)")
    lock_ns: dict = {}
    exec(LOCK_PATH.read_text(encoding="utf-8"), {}, lock_ns)
    if wheel_path.name != lock_ns["filename"]:
        raise ValueError(f"wheel filename {wheel_path.name!r} != locked {lock_ns['filename']!r}")
    actual_sha256 = wheel_sha256(wheel_path)
    if actual_sha256 != lock_ns["sha256"]:
        raise ValueError(f"wheel sha256 {actual_sha256!r} != locked {lock_ns['sha256']!r}")
    expected_names = CONTENTS_PATH.read_text(encoding="utf-8").splitlines()
    actual_names = wheel_namelist(wheel_path)
    if actual_names != expected_names:
        raise ValueError(
            f"wheel contents mismatch vs {CONTENTS_PATH.name}:\n"
            f"  only in wheel: {sorted(set(actual_names) - set(expected_names))}\n"
            f"  only in fixture: {sorted(set(expected_names) - set(actual_names))}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-lock", action="store_true", help="step 19 only")
    mode.add_argument("--write-contents", action="store_true", help="step 19 only")
    mode.add_argument("--check", metavar="WHEEL", help="step 19+ only: verify an already-built wheel")
    args = parser.parse_args(argv)

    if args.check:
        _check(Path(args.check))
        print(f"OK: {args.check} matches {LOCK_PATH.name} and {CONTENTS_PATH.name}")
        return 0

    wheel_path = build_wheel(REPO_ROOT / "dist")
    print(f"built {wheel_path.relative_to(REPO_ROOT)}")

    if args.write_lock:
        _write_lock(wheel_path)
        print(f"wrote {LOCK_PATH.relative_to(REPO_ROOT)}")
    if args.write_contents:
        _write_contents(wheel_path)
        print(f"wrote {CONTENTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
