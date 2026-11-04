"""Phase E manuscript assembler (plan v6 §3.3).

Pure-Python directive expansion of the ordered `manuscript/` section files
into `build/chapter.expanded.md`. **No Pandoc, no Mermaid** -- Phase F runs
those on the expanded file.

    python scripts/build_manuscript.py --check     # in-memory only, no writes
    python scripts/build_manuscript.py --write     # validate, then write build/

`--check` and `--write` are mutually exclusive.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MANUSCRIPT = REPO_ROOT / "manuscript"
BUILD_DIR = REPO_ROOT / "build"

# The canonical ordered file list (plan §3.2; mirrored in manuscript/README.md,
# which lint check 1 verifies against this constant).
SECTION_ORDER: list[str] = [
    "00_metadata.md",
    "s01_ivadas.md",
    "s02_teoriniai_aspektai.md",
    "s03_protokolas.md",
    "s04_paieskos_strategija.md",
    "s05_daugiabaze_paieska.md",
    "s06_deduplikacija.md",
    "s07_atranka.md",
    "s08_pilno_teksto_tinkamumas.md",
    "s09_kokybes_vertinimas.md",
    "s10_duomenu_isgava.md",
    "s11_sinteze.md",
    "s12_ataskaitu_teikimas.md",
    "s13_praktinis_pavyzdys.md",
    "s14_baigiamosios_izvalgos.md",
    "s15_literatura.md",
]

_LISTINGS_MANIFEST = REPO_ROOT / "examples" / "chapter" / "listings_manifest.json"


class BuildError(Exception):
    pass


def load_listings_manifest() -> dict[str, dict]:
    rows = json.loads(_LISTINGS_MANIFEST.read_text(encoding="utf-8"))["listings"]
    return {row["id"]: row for row in rows}


def _extract_listing(source_rel: str, listing_id: str) -> str:
    from scripts.extract_listings import extract_listing  # frozen 2-arg signature

    return extract_listing(REPO_ROOT / source_rel, listing_id)


def _figure_block(fig_id: str) -> str:
    fig_path = MANUSCRIPT / "figures" / f"{fig_id}.md"
    if not fig_path.is_file():
        raise BuildError(f"unknown figure id: {fig_id}")
    text = fig_path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise BuildError(f"{fig_path}: missing YAML front-matter")
    front, body = m.group(1), m.group(2).strip()
    caption = _yaml_scalar(front, "caption")
    alt = _yaml_scalar(front, "alt")
    if not caption or not alt:
        raise BuildError(f"{fig_path}: front-matter needs both 'caption' and 'alt'")
    if "```mermaid" not in body:
        raise BuildError(f"{fig_path}: body has no ```mermaid block")
    return f"::: {{.figure #fig:{fig_id} alt=\"{alt}\"}}\n{body}\n\n{caption}\n:::"


def _yaml_scalar(front: str, key: str) -> str | None:
    m = re.search(rf'^{re.escape(key)}:\s*"(.*)"\s*$', front, re.M)
    if m:
        return m.group(1)
    m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", front, re.M)
    return m.group(1) if m else None


# --- directive patterns --------------------------------------------------

_RE_SECTION = re.compile(r"^<!--\s*section:(S\d{2})\s*-->\s*$")
_RE_CLAIM = re.compile(r"^<!--\s*claim:(C\d{2})\s*-->\s*$")
_RE_SIDEBAR = re.compile(r"^<!--\s*sidebar:([A-Za-z0-9-]+)\s*-->\s*$")
_RE_LISTING = re.compile(r"^<!--\s*listing:([a-z0-9-]+)\s*-->\s*$")
_RE_ASSIGN_START = re.compile(r"^<!--\s*assignment:(s\d{2})\s+start\s*-->\s*$")
_RE_ASSIGN_END = re.compile(r"^<!--\s*assignment:(s\d{2})\s+end\s*-->\s*$")
_RE_FIGURE = re.compile(r"^<!--\s*figure:([a-z0-9-]+)\s*-->\s*$")
_RE_FRAGMENT = re.compile(r"^<!--\s*fragment: illustrative, not executed\s*-->\s*$")
_RE_NARROWED = re.compile(r"^<!--\s*narrowed:.*-->\s*$")
_RE_RUNREF = re.compile(r"^<!--\s*run-ref:.*-->\s*$")
_RE_MITNOTICE = re.compile(r"^<!--\s*mit-notice\s*-->\s*$")
_RE_TRACK3 = re.compile(r"<!--\s*TRACK3-PENDING:.*-->")
_RE_PHASEF = re.compile(r"<!--\s*PHASEF-PENDING:.*-->")


def section_source_files() -> list[str]:
    """Every `manuscript/s*.md` on disk, sorted."""
    return sorted(p.name for p in MANUSCRIPT.glob("s*.md"))


def raw_manuscript_markdown() -> dict[str, str]:
    """Every distributable Markdown file under manuscript/ (NOT internal/,
    NOT the vendored CSL). Keys are repo-relative posix paths."""
    out: dict[str, str] = {}
    for p in sorted(MANUSCRIPT.rglob("*.md")):
        if "internal" in p.relative_to(MANUSCRIPT).parts:
            continue
        out[f"manuscript/{p.relative_to(MANUSCRIPT).as_posix()}"] = p.read_text(encoding="utf-8")
    return out


_DRIVE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")          # X:\  X:/
_UNC_BACKSLASH_RE = re.compile(r"\\\\[A-Za-z0-9._$-]+\\[A-Za-z0-9._$-]+")  # \\host\share
_MSYS_DRIVE_RE = re.compile(r"(?<![\w:/])/[a-zA-Z]/(?=[a-zA-Z])")   # /c/...  /C:/...
# scheme://host/seg or bare //host/seg -- the ONLY exempt schemes are http/https
_SLASHSLASH_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9+.-]*:)?//([A-Za-z0-9._-]+)/[A-Za-z0-9._-]"
)


def _abs_path_hits(text: str) -> list[str]:
    hits: list[str] = []
    for rx in (_DRIVE_RE, _UNC_BACKSLASH_RE, _MSYS_DRIVE_RE):
        m = rx.search(text)
        if m:
            hits.append(m.group(0))
    for m in _SLASHSLASH_RE.finditer(text):
        scheme = (m.group(1) or "").lower()
        if scheme in ("http:", "https:"):
            continue
        hits.append(m.group(0))
        break
    return hits


def leak_scan(text: str, *, proj_commits: dict, artifact_hashes: set) -> list[str]:
    """Return a list of leak descriptions (empty == clean). Detects any literal
    `proj:` and any absolute Windows/UNC path regardless of surrounding
    punctuation; only `http://` / `https://` URLs are exempt from the
    `//host/...` check (a `file://` / `ftp://` path is still a leak)."""
    problems: list[str] = []
    deny = set()
    for h in list(proj_commits.values()) + list(artifact_hashes):
        for L in range(7, len(h) + 1):
            deny.add(h[:L])
    for token in sorted(deny, key=len, reverse=True):
        if token in text:
            problems.append(f"internal hash prefix {token!r}")
            break
    if "proj:" in text:
        problems.append("literal 'proj:' identifier")
    for hit in _abs_path_hits(text):
        problems.append(f"absolute filesystem path near {hit!r}")
        break
    return problems


def check_invariants() -> None:
    """The `--check` assertions that are the assembler's own responsibility
    (plan v6 §3.3): exact source-file coverage + a leak scan of every raw
    Markdown file and of the in-memory expansion."""
    on_disk = set(section_source_files()) | {"00_metadata.md"}
    declared = set(SECTION_ORDER)
    if on_disk != declared:
        raise BuildError(f"SECTION_ORDER != files on disk: extra={declared - on_disk}, missing={on_disk - declared}")

    inv = (REPO_ROOT / "chapter_project_inventory.md").read_text(encoding="utf-8")
    proj_commits = dict(re.findall(r"\|\s*`(proj:[a-z_]+)`\s*\|[^|]*\|[^|]*\|\s*`([0-9a-f]{40})`", inv))
    artifact_hashes = set(re.findall(r"sha256:([0-9a-f]{64})", inv))

    targets = dict(raw_manuscript_markdown())
    targets["build(expanded)"] = expand(allow_pending=True)
    for name, text in targets.items():
        problems = leak_scan(text, proj_commits=proj_commits, artifact_hashes=artifact_hashes)
        if problems:
            raise BuildError(f"leak scan: {name}: {problems[0]}")


def expand(*, allow_pending: bool = True) -> str:
    """Expand the ordered section files into one string. Raises BuildError."""
    manifest = load_listings_manifest()
    out: list[str] = []
    listing_counter = 0
    track3_seen = 0

    for fname in SECTION_ORDER:
        path = MANUSCRIPT / fname
        if not path.is_file():
            raise BuildError(f"missing section file: {fname}")
        chunk, listing_counter, seen = _expand_chunk(
            path.read_text(encoding="utf-8"), fname, manifest, listing_counter)
        track3_seen += seen
        out.append(chunk)
        out.append("")

    result = "\n".join(out).rstrip("\n") + "\n"
    if not allow_pending and track3_seen:
        raise BuildError(f"{track3_seen} TRACK3-PENDING token(s) remain but --allow-pending=false")
    return result


def expand_chunk(text: str, *, source_name: str = "fixture", start_count: int = 0) -> str:
    """Expand one markdown chunk (used by the golden-fixture tests)."""
    chunk, _count, _seen = _expand_chunk(text, source_name, load_listings_manifest(), start_count)
    return chunk.rstrip("\n") + "\n"


def _expand_chunk(text: str, fname: str, manifest: dict[str, dict], listing_counter: int):
    out: list[str] = []
    track3_seen = 0
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        if _RE_SECTION.match(line) or _RE_CLAIM.match(line) or _RE_SIDEBAR.match(line) \
                or _RE_NARROWED.match(line) or _RE_RUNREF.match(line) or _RE_MITNOTICE.match(line):
            i += 1
            continue

        m = _RE_LISTING.match(line)
        if m:
            lid = m.group(1)
            if lid not in manifest:
                raise BuildError(f"{fname}:{i+1}: unknown listing id {lid!r}")
            k = i + 1
            cap_lines: list[str] = []
            while k < n and lines[k].strip() and not lines[k].lstrip().startswith("<!--"):
                cap_lines.append(lines[k].strip())
                k += 1
            cap = " ".join(cap_lines)
            if not cap:
                raise BuildError(f"{fname}:{i+1}: <!-- listing:{lid} --> needs a caption line after it")
            if re.match(r"^\*\*\s*\d", cap):
                raise BuildError(f"{fname}:{i+2}: caption must not be pre-numbered by the author")
            listing_counter += 1
            row = manifest[lid]
            code = _extract_listing(row["source"], lid)
            out.append(f"**{listing_counter} sąrašas.** {cap}")
            out.append("")
            out.append("```python")
            out.append(code.rstrip("\n"))
            out.append("```")
            i = k
            continue

        m = _RE_ASSIGN_START.match(line)
        if m:
            sid = m.group(1)
            j = i + 1
            body: list[str] = []
            while j < n and not _RE_ASSIGN_END.match(lines[j]):
                body.append(lines[j])
                j += 1
            if j >= n:
                raise BuildError(f"{fname}:{i+1}: <!-- assignment:{sid} start --> has no matching end")
            if _RE_ASSIGN_END.match(lines[j]).group(1) != sid:
                raise BuildError(f"{fname}:{j+1}: assignment end id != start id")
            if not any(b.strip() for b in body):
                raise BuildError(f"{fname}:{i+1}: assignment {sid} has no authored content")
            out.append("> **Užduotis.**")
            out.append(">")
            for b in body:
                out.append(f"> {b}".rstrip() if b.strip() else ">")
            if sid != "s03":
                num = int(sid[1:])
                slug = _slug_for(num)
                out.append("")
                out.append("> **Kaip paleisti.**")
                out.append(f"> Saugyklos kelias: `examples/chapter/{slug}/`.")
                out.append(f"> Paleidimas: `python -m examples.chapter.{slug}.exercise`.")
                out.append(
                    f"> Pavyzdinį sprendimą galima patikrinti: "
                    f"`python -m pytest examples/chapter/{slug} -q` (rakto ir tinklo nereikia). "
                    f"Šie testai tikrina pavyzdinį (instruktoriaus) sprendimą `solution.py`, "
                    f"o ne automatiškai vertina jūsų `exercise.py`."
                )
            i = j + 1
            continue

        m = _RE_FIGURE.match(line)
        if m:
            out.append(_figure_block(m.group(1)))
            i += 1
            continue

        if _RE_TRACK3.search(line):
            track3_seen += 1
            out.append(line)
            i += 1
            continue

        # fragment markers, PHASEF-PENDING, and ordinary content pass through
        out.append(line)
        i += 1

    return "\n".join(out), listing_counter, track3_seen


_SLUGS = {
    4: "s04_query_expansion", 5: "s05_multi_db_search", 6: "s06_deduplication",
    7: "s07_screening", 8: "s08_fulltext", 9: "s09_quality_appraisal",
    10: "s10_extraction", 11: "s11_synthesis_topics", 12: "s12_reporting",
}


def _slug_for(num: int) -> str:
    if num not in _SLUGS:
        raise BuildError(f"no examples/chapter package for section S{num:02d}")
    return _SLUGS[num]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="in-memory expansion + assertions, no writes")
    g.add_argument("--write", action="store_true", help="validate, then write build/chapter.expanded.md")
    ap.add_argument("--allow-pending", choices=["true", "false"], default="true")
    args = ap.parse_args()

    try:
        check_invariants()
        expanded = expand(allow_pending=(args.allow_pending == "true"))
    except BuildError as exc:
        print(f"build_manuscript: {exc}", file=sys.stderr)
        return 1

    if args.write:
        BUILD_DIR.mkdir(exist_ok=True)
        # explicit UTF-8/LF bytes -- Path.write_text would translate LF->CRLF on Windows
        (BUILD_DIR / "chapter.expanded.md").write_bytes(expanded.encode("utf-8"))
        print(f"wrote {BUILD_DIR / 'chapter.expanded.md'} ({len(expanded)} bytes)")
    else:
        print(f"build_manuscript --check OK ({len(expanded)} bytes in memory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
