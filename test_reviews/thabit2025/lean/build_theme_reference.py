"""
One-time build: extract Thabit et al. 2025's own Findings-section prose per
theme (Table 2's 6 themes) from `_thabit_full.txt` (Docling extraction of
the published PDF), cleaned of page furniture (running headers, bare page
numbers, the Table 2 block itself, omitted-picture markers). Used as the
qualitative reference text against which `run_theme_synthesis.py`'s
LLM-generated cross-study synthesis can be read side by side -- not scored
(narrative prose isn't precision/recall-comparable), a reading aid only.

Line ranges below are the Findings section's own '## <Theme>' headers in
`_thabit_full.txt`, read directly off the file (1-indexed, matches the Read
tool's own line numbers) -- not guessed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path(__file__).parent.parent.parent.parent / "_thabit_full.txt"
OUT = Path(__file__).parent / "theme_reference_text.json"

# (theme name -- matches ft_codebook.CODEBOOK's theme strings exactly,
#  first line, last line) -- 1-indexed, inclusive.
THEME_RANGES = [
    ("Engagement and authorization", 168, 250),
    ("Operational capabilities and assets", 251, 284),
    ("Broader environment and public sphere", 285, 300),
    ("Instrumental public values (process-oriented)", 301, 332),
    ("Instrumental and prime public values", 333, 360),
    ("Prime public values (output-oriented)", 361, 386),
]

NOISE_EXACT = {
    "STRATEGIC PUBLIC VALUE(S) GOVERNANCE",
    "— PUBLIC ADMINISTRATION REVIEW",
    "PUBLIC ADMINISTRATION REVIEW",
}


def clean_range(lines: list[str], start: int, end: int) -> str:
    """start/end are 1-indexed inclusive line numbers."""
    raw = lines[start - 1:end]
    kept: list[str] = []
    skip_table = False
    for ln in raw:
        s = ln.strip()
        if s.startswith("T A B L E"):
            skip_table = True
            continue
        if skip_table:
            if not s or s.startswith("|"):
                continue
            skip_table = False  # first non-blank, non-pipe line ends the table
        if not s:
            continue
        if s in NOISE_EXACT:
            continue
        if re.fullmatch(r"\d{2,3}", s):
            continue
        if s.startswith("**==>") or s.startswith("**----"):
            continue
        kept.append(s)
    return "\n\n".join(kept)


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    ref = {}
    for theme, start, end in THEME_RANGES:
        text = clean_range(lines, start, end)
        ref[theme] = text
        print(f"{theme!r}: {len(text)} chars")

    OUT.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
