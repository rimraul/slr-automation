"""Classify a git diff's changed files into Track 2's three licence-manifest
sections.

track2_pipeline_readiness_plan_2026-08-30.md §12.5 (round-14 finding 8,
round-16 finding 6, round-17 finding 4): every path changed within the
closed authored-code range `C1^..CN` belongs to exactly one of three
sections --

    mit         -- authored-code default: "(c) 2026 Rimantas Rauleckas, MIT"
    cc_by_4_0   -- carve-out 1: tests/reference_run/fixtures/corpus/**
                   (licensed under tests/reference_run/fixtures/corpus/LICENSE.md)
    manuscript  -- carve-out 2: manuscript/HANDOFF.md (Phase E's own licence,
                   not Track 2's)

This script is authored at **step 2**; it is only *invoked* against the real
closed range `C1^..CN` at step 18 (commit `CM`), which replaces the
`track2/LICENSE-NOTES.md` stub body with this script's rendered manifest and
touches no other file.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path

CC_BY_PATTERN = "tests/reference_run/fixtures/corpus/*"
MANUSCRIPT_PATH = "manuscript/HANDOFF.md"

SECTION_NAMES = ("mit", "cc_by_4_0", "manuscript")


def changed_paths(rev_range: str, *, cwd: Path | None = None) -> list[str]:
    """`git diff --name-only <rev_range>` (e.g. `"C1^..CN"`), as a list of
    repo-relative paths."""
    result = subprocess.run(
        ["git", "diff", "--name-only", rev_range],
        cwd=cwd, check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def classify(paths: list[str]) -> dict[str, list[str]]:
    """Partition `paths` into `{"mit": [...], "cc_by_4_0": [...],
    "manuscript": [...]}`, exhaustively and disjointly -- every path lands
    in exactly one section."""
    sections: dict[str, list[str]] = {name: [] for name in SECTION_NAMES}
    for path in paths:
        if path == MANUSCRIPT_PATH:
            sections["manuscript"].append(path)
        elif fnmatch.fnmatch(path, CC_BY_PATTERN):
            sections["cc_by_4_0"].append(path)
        else:
            sections["mit"].append(path)
    return sections


def render_manifest(sections: dict[str, list[str]]) -> str:
    lines = ["# Track-2 licence manifest", ""]
    lines.append("## mit -- (c) 2026 Rimantas Rauleckas, MIT (LICENSE-TRACK2)")
    lines.extend(f"- {p}" for p in sorted(sections["mit"]))
    lines.append("")
    lines.append(
        "## cc_by_4_0 -- tests/reference_run/fixtures/corpus/LICENSE.md"
    )
    lines.extend(f"- {p}" for p in sorted(sections["cc_by_4_0"]))
    lines.append("")
    lines.append("## manuscript -- Phase E's own licence, not Track 2's")
    lines.extend(f"- {p}" for p in sorted(sections["manuscript"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rev_range", help='git diff rev-range, e.g. "C1^..CN"')
    parser.add_argument("--cwd", default=None, help="repo root to run git in (default: cwd)")
    args = parser.parse_args(argv)

    paths = changed_paths(args.rev_range, cwd=Path(args.cwd) if args.cwd else None)
    sections = classify(paths)
    print(render_manifest(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
