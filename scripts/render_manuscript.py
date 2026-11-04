"""Phase F renderer: Pandoc + Mermaid pass over the expanded manuscript.

Builds `build/chapter.expanded.md` (via build_manuscript.expand(), same code
path as `build_manuscript.py --write`) and then shells out to `pandoc` to
produce a reader-facing document (PDF/DOCX/HTML).

Per the Phase E/F plan: `--csl` and `--bibliography` are **never** passed on
the Pandoc command line. Both paths come solely from the YAML front matter
in `manuscript/00_metadata.md`, which `build_manuscript.expand()` carries
through verbatim as the first block of `chapter.expanded.md` -- Pandoc reads
its own `csl:`/`bibliography:` metadata keys from that block. `--citeproc` is
the single bibliography owner; `s15_literatura.md`'s `::: {#refs} :::` div is
where it inserts the reference list.

Mermaid: the S2 figure (`manuscript/figures/pipeline-overview.md`) carries a
fenced ```mermaid block inside a `::: {.figure #fig:...}` div. Pandoc has no
native Mermaid renderer, so this script adds `--filter mermaid-filter`
(https://github.com/raghur/mermaid-filter, `npm install -g mermaid-filter`)
whenever the expanded text contains a ```mermaid fence. mermaid-filter shells
out to a headless Chromium via puppeteer; if PDF output is requested in an
environment without a usable Chromium, pass --no-mermaid to skip the filter
and leave the raw fenced block in place.

    python scripts/render_manuscript.py --format pdf
    python scripts/render_manuscript.py --format docx --out build/chapter.docx
    python scripts/render_manuscript.py --format html --no-mermaid

KNOWN LIMITATION (2026-10-01): neither `pandoc` nor `mermaid-filter`/`mmdc`
is installed on this machine, so this script has not been run end-to-end
here. It has only been exercised up to the point where it shells out to
`pandoc`; the `pandoc`/`mermaid-filter` invocation itself is unverified.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_manuscript import BuildError, check_invariants, expand

BUILD_DIR = REPO_ROOT / "build"
EXPANDED_PATH = BUILD_DIR / "chapter.expanded.md"

_EXT_BY_FORMAT = {"pdf": "pdf", "docx": "docx", "html": "html"}


def _write_expanded() -> Path:
    check_invariants()
    text = expand(allow_pending=False)
    BUILD_DIR.mkdir(exist_ok=True)
    EXPANDED_PATH.write_bytes(text.encode("utf-8"))
    return EXPANDED_PATH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=sorted(_EXT_BY_FORMAT), default="pdf")
    ap.add_argument("--out", type=Path, default=None, help="output path (default: build/chapter.<format>)")
    ap.add_argument("--no-mermaid", action="store_true", help="skip the mermaid-filter pass; leave raw ```mermaid fences in the output")
    args = ap.parse_args()

    pandoc = shutil.which("pandoc")
    if pandoc is None:
        print(
            "render_manuscript: 'pandoc' not found on PATH. Install it "
            "(https://pandoc.org/installing.html) and re-run.",
            file=sys.stderr,
        )
        return 1

    try:
        expanded_path = _write_expanded()
    except BuildError as exc:
        print(f"render_manuscript: build step failed: {exc}", file=sys.stderr)
        return 1

    out_path = args.out or (BUILD_DIR / f"chapter.{_EXT_BY_FORMAT[args.format]}")

    has_mermaid = "```mermaid" in expanded_path.read_text(encoding="utf-8")
    use_mermaid_filter = has_mermaid and not args.no_mermaid
    mermaid_filter = shutil.which("mermaid-filter")  # resolves mermaid-filter.cmd on Windows
    if use_mermaid_filter and mermaid_filter is None:
        print(
            "render_manuscript: expanded manuscript contains a ```mermaid "
            "block but 'mermaid-filter' is not on PATH. Install it "
            "(npm install -g mermaid-filter) or pass --no-mermaid to render "
            "with the raw fenced block left in place.",
            file=sys.stderr,
        )
        return 1

    cmd = [pandoc, str(expanded_path), "--citeproc", "-o", str(out_path)]
    if use_mermaid_filter:
        cmd[1:1] = ["--filter", mermaid_filter]

    print("render_manuscript: running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"render_manuscript: pandoc exited {result.returncode}", file=sys.stderr)
        return result.returncode

    print(f"render_manuscript: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
