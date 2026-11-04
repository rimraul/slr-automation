"""Phase E manuscript drafting-time lint (plan v6 §8, + implementation-review
round-1 amendments).

Offline, no network, no key.

    python scripts/lint_manuscript.py --check              # globals + completeness-for-complete (CI)
    python scripts/lint_manuscript.py --write              # regenerate manuscript/PLACEHOLDERS.md
    python scripts/lint_manuscript.py --no-regress <ref>   # STATUS.json cannot regress vs <ref>
    python scripts/lint_manuscript.py --final              # the pre-final-review gate

`--check` and `--write` are mutually exclusive. Neither `--check` nor
`--no-regress` writes anything. Not Phase F: no literature re-search, no
source re-verification, no code execution, no Pandoc/Mermaid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_manuscript import (  # noqa: E402
    SECTION_ORDER, expand, leak_scan, load_listings_manifest,
    raw_manuscript_markdown, section_source_files,
)

MANUSCRIPT = REPO_ROOT / "manuscript"
_CSL_SHA256 = "1ece4fb3c295e66d04b4394e295aa58a87741ceeef1658192437eb9953c2f13e"
_SECTION_FILES = [f for f in SECTION_ORDER if f != "00_metadata.md"]
_ALL_SIDS = [f"S{n:02d}" for n in range(1, 16)]
_STATE_ENUM = {"skeleton", "drafting", "complete"}
_GOLDEN_FIXTURES = [
    "listing_in.md", "listing.expected.md", "assignment_in.md", "assignment.expected.md",
    "figure_in.md", "figure.expected.md", "refs_in.md", "refs.expected.md",
]

_FROZEN_TITLES: dict[str, tuple[int, str]] = {
    "S01": (2, "Įvadas"),
    "S02": (2, "Teoriniai aspektai"),
    "S03": (3, "Protokolo apibrėžimas ir apžvalgos šeimos pasirinkimas"),
    "S04": (3, "Paieškos strategija: užklausos plėtimas su DKM pagalba"),
    "S05": (3, "Daugiabazė paieška"),
    "S06": (3, "Dubliavimosi pašalinimas (deduplikacija)"),
    "S07": (3, "Pavadinimų ir santraukų atranka bei tarpvertintojų patikimumas"),
    "S08": (3, "Pilno teksto tinkamumo vertinimas"),
    "S09": (3, "Kokybės vertinimas"),
    "S10": (3, "Duomenų išgava (chartavimas)"),
    "S11": (3, "Sintezė: įrodymų žemėlapiai, teiginiais paremti rezultatai ir BERTopic"),
    "S12": (3, "Ataskaitų teikimas, PRISMA diagramos ir DI naudojimo atsekamumas"),
    "S13": (2, "Praktinis automatizuotos tiriamosios (scoping) apžvalgos pavyzdys"),
    "S14": (2, "Baigiamosios įžvalgos"),
    "S15": (2, "Literatūra"),
}

# plan v6 §5.matrix -- the EXACT per-section sets.
_LISTING_MATRIX: dict[str, list[str]] = {
    "S04": ["s04-expand", "s04-translate"],
    "S05": ["s05-fixture-adapter", "s05-normalise", "s05-run-search"],
    "S06": ["s06-normalise-doi", "s06-dedupe"],
    "S07": ["s07-screen-batch", "s07-partition", "s07-irr"],
    "S08": ["s08-discover", "s08-download", "s08-looks-like-stub", "s08-screen-fulltext"],
    "S09": ["s09-checklist", "s09-appraise"],
    "S10": ["s10-schema", "s10-extract"],
    "S11": ["s11-run-topics", "s11-map-study", "s11-refine"],
    "S12": ["s12-prisma-flow", "s12-classify"],
}
_SIDEBAR_MATRIX: dict[str, list[str]] = {
    "S03": ["S03-a"],
    "S05": ["S05-a", "S05-b"],
    "S07": ["S07-a", "S07-b", "S07-c", "S07-d"],
    "S08": ["S08-a", "S08-b", "S08-c"],
    "S10": ["S10-a"],
    "S11": ["S11-a", "S11-b", "S11-c"],
}
_ASSIGNMENT_SECTIONS = {f"S{n:02d}" for n in range(3, 13)}
_ILLUSTRATION_SECTIONS = {f"S{n:02d}" for n in range(3, 13)}  # each carries one run-derived illustration

_S01_REQUIRED_KEYS = {
    "higgins2019cochranehandbook", "peters2024jbiscopingreviews", "tricco2018prismascr",
    "garritty2024rapidreviews", "wang2025llmboolquery", "yuan2025llmnondeterminism",
    "schroeder2025llmextractionhitl",
}

# The three prose-licence notices (single source of truth). Compared normalized
# (NFC, collapsed whitespace, dashes unified) against LICENSE-NOTES.md and S1.
_PROSE_NOTICE = {
    "pending": (
        "© 2026 Rimantas Rauleckas. Šis tekstas rengiamas leidybai; numatoma licencija - "
        "Creative Commons Attribution 4.0 (CC BY 4.0) - įsigalios tik gavus leidėjo "
        "raštišką patvirtinimą. Iki tol jokia atviroji licencija nesuteikiama."
    ),
    "granted": (
        "Except where otherwise noted, the manuscript prose is licensed under CC BY 4.0. "
        "The 23 Python code listings are separately licensed under the MIT License. "
        "Third-party materials retain their stated licences."
    ),
    "refused": "© 2026 Rimantas Rauleckas. Visos teisės saugomos.",
}

_GLOSSARY_PRIMARY = [
    "tiriamoji literatūros apžvalga", "deduplikacija", "chartavimas",
    "pilno teksto tinkamumo vertinimas", "tarpvertintojų patikimumas", "atsekamumas",
]
_GLOSSARY_ALTERNATIVES = ["apžvalginė", "žvalgomoji", "provaizdžio šablonas"]

_CITE_RE = re.compile(r"(?:\[|;)\s*@([A-Za-z0-9_]+)")
_VISIBLE_SID_RE = re.compile(r"(?<![A-Za-z0-9])S0?(?:[1-9]|1[0-5])(?![A-Za-z0-9])")
_FENCE_RE = re.compile(r"^```")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"(?m)^[ \t]*>[ \t]?", "", s)  # strip blockquote markers
    return re.sub(r"\s+", " ", s).strip()


def cited_keys(text: str) -> set[str]:
    return set(_CITE_RE.findall(text))


def paragraph_after(text: str, marker_re: str) -> str | None:
    m = re.search(marker_re + r"[ \t]*\n(?!\n)((?:[^\n]+\n?)+?)(?:\n|$)", text)
    return m.group(1).strip() if m else None


def strip_comments_and_fences(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, in_fence = [], False
    for line in text.split("\n"):
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def canon(sid: str) -> str:
    return "S" + str(int(sid[1:]))


def check_no_regress(baseline: dict, current: dict) -> list[str]:
    return [f"{sid} was 'complete' in the baseline but is now {current.get(sid)!r}"
            for sid, st in baseline.items()
            if sid.startswith("S") and st == "complete" and current.get(sid) != "complete"]


class Lint:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.report: list[str] = []
        self.claims = json.loads((REPO_ROOT / "chapter_claims.json").read_text(encoding="utf-8"))["claims"]
        self.rels = json.loads((REPO_ROOT / "chapter_claim_sources.json").read_text(encoding="utf-8"))["relationships"]
        self.bibkeys = set(re.findall(r"@\w+\{([^,]+),", (REPO_ROOT / "chapter_references.bib").read_text(encoding="utf-8")))
        import csv
        self.src2bib = {r["source_id"]: r["bibliography_key"]
                        for r in csv.DictReader((REPO_ROOT / "chapter_evidence_sources.csv").open(encoding="utf-8"))}
        inv = (REPO_ROOT / "chapter_project_inventory.md").read_text(encoding="utf-8")
        self.proj_commits = dict(re.findall(r"\|\s*`(proj:[a-z_]+)`\s*\|[^|]*\|[^|]*\|\s*`([0-9a-f]{40})`", inv))
        self.artifact_hashes = set(re.findall(r"sha256:([0-9a-f]{64})", inv))
        self.manifest_ids = set(load_listings_manifest())
        self.sidebars = json.loads((MANUSCRIPT / "internal" / "sidebars.json").read_text(encoding="utf-8"))["sidebars"]
        self.status_raw = json.loads((MANUSCRIPT / "STATUS.json").read_text(encoding="utf-8"))
        self.status = ({k: self.status_raw[k] for k in _ALL_SIDS if k in self.status_raw}
                       if isinstance(self.status_raw, dict) else {})
        self.notes_text = (MANUSCRIPT / "LICENSE-NOTES.md").read_text(encoding="utf-8")
        m = re.search(r"clearance:\s*(\w+)", self.notes_text)
        self.clearance = m.group(1) if m else "?"
        self.files = {f: (MANUSCRIPT / f).read_text(encoding="utf-8") for f in _SECTION_FILES}
        self.registry_state = self._registry_state()

    def err(self, m: str) -> None:
        self.errors.append(m)

    def _registry_state(self) -> str:
        want = {"C27", "C28", "C29", "C30", "C31"}
        statuses = {c["claim_id"]: c["evidence_status"] for c in self.claims if c["claim_id"] in want}
        if set(statuses) != want:
            self.err(f"registry: missing evidence_status for {sorted(want - set(statuses))}")
            return "pending_track_3"
        vals = set(statuses.values())
        if vals == {"verified"}:
            return "verified"
        if vals == {"pending_track_3"}:
            return "pending_track_3"
        self.err(f"registry: C27-C31 evidence_status is mixed/invalid: {statuses}")
        return "pending_track_3"

    def claim_citesets(self, cid: str):
        s_lit, q_lit, i_lit = set(), set(), set()
        for r in self.rels:
            if r["claim_id"] != cid or not r["source_id"].startswith("lit:"):
                continue
            key = self.src2bib.get(r["source_id"])
            {"supports": s_lit, "qualifies": q_lit, "illustrates": i_lit}.get(r["relationship"], set()).add(key)
        return s_lit, q_lit, i_lit

    def section_claims(self, sid: str) -> list[str]:
        return [c["claim_id"] for c in self.claims
                if canon(c["section_id"]) == canon(sid) and c["evidence_status"] == "verified"]

    # ================= GLOBAL CHECKS =================

    def c_status_json(self) -> None:
        if not isinstance(self.status_raw, dict):
            self.err("status: STATUS.json is not a JSON object")
            return
        allowed = set(_ALL_SIDS) | {"_comment"}
        extra = set(self.status_raw) - allowed
        if extra:
            self.err(f"status: STATUS.json has unexpected key(s) {sorted(extra)} (only S01..S15 and _comment are allowed)")
        missing = set(_ALL_SIDS) - set(self.status_raw)
        if missing:
            self.err(f"status: STATUS.json is missing key(s) {sorted(missing)}")
        if "_comment" in self.status_raw and not isinstance(self.status_raw["_comment"], str):
            self.err("status: STATUS.json _comment must be a string")
        for sid in _ALL_SIDS:
            if sid not in self.status_raw:
                continue
            val = self.status_raw[sid]
            if not isinstance(val, str) or val not in _STATE_ENUM:
                self.err(f"status: {sid} value {val!r} not one of {sorted(_STATE_ENUM)}")

    def c1_section_markers(self) -> None:
        readme = (MANUSCRIPT / "README.md").read_text(encoding="utf-8")
        for f in SECTION_ORDER:
            if f not in readme:
                self.err(f"check1: README ordered list is missing {f}")
        if set(section_source_files()) | {"00_metadata.md"} != set(SECTION_ORDER):
            self.err("check1: SECTION_ORDER does not match manuscript/s*.md on disk")
        for f in _SECTION_FILES:
            markers = re.findall(r"<!--\s*section:(S\d{2})\s*-->", self.files[f])
            want = "S" + f[1:3]
            if markers != [want]:
                self.err(f"check1: {f} section marker(s) {markers} != [{want}]")

    def c2_titles(self) -> None:
        for f in _SECTION_FILES:
            sid = "S" + f[1:3]
            lvl, title = _FROZEN_TITLES[sid]
            headings = re.findall(r"^(#{1,6})\s+(.*?)\s*$", self.files[f], re.M)
            if not headings:
                self.err(f"check2: {f} has no heading")
                continue
            if sid == "S03":
                if headings[0] != ("##", "Metodologiniai taikymo orientyrai"):
                    self.err(f"check2: {f} first heading must be '## Metodologiniai taikymo orientyrai'")
                if len(headings) < 2 or headings[1] != ("#" * lvl, title):
                    self.err(f"check2: {f} second heading must be '{'#'*lvl} {title}'")
            elif headings[0] != ("#" * lvl, title):
                self.err(f"check2: {f} first heading {headings[0]} != ('{'#'*lvl}', '{title}')")

    def c3_c27_31(self) -> None:
        for f in _SECTION_FILES:
            anchors = set(re.findall(r"<!--\s*claim:(C2[7-9]|C3[01])\s*-->", self.files[f]))
            if self.registry_state == "pending_track_3":
                if anchors:
                    self.err(f"check3: C27-C31 anchor(s) {sorted(anchors)} in {f} while pending_track_3")
            elif f != "s13_praktinis_pavyzdys.md" and anchors:
                self.err(f"check3: C27-C31 anchor outside S13 in {f}")
        if self.registry_state == "verified":
            s13a = set(re.findall(r"<!--\s*claim:(C\d{2})\s*-->", self.files["s13_praktinis_pavyzdys.md"]))
            if not {"C27", "C28", "C29", "C30", "C31"} <= s13a:
                self.err(f"check3: verified state needs C27-C31 anchors in S13; have {sorted(s13a)}")

    def c4_citations(self) -> None:
        for f in _SECTION_FILES:
            for key in cited_keys(self.files[f]):
                if key not in self.bibkeys:
                    self.err(f"check4: {f} cites [@{key}] -- not in chapter_references.bib")

    def c_visible_sids(self) -> None:
        for f in _SECTION_FILES:
            for m in _VISIBLE_SID_RE.finditer(strip_comments_and_fences(self.files[f])):
                self.err(f"check(visible-sid): {f} has a bare section id {m.group(0)!r} in reader-facing text")

    def c10_fences(self) -> None:
        # Raw source never hand-writes ```python / ```mermaid blocks -- the builder
        # inserts those from <!-- listing --> / <!-- figure --> directives. The only
        # fenced blocks allowed in raw source are (a) the S1 MIT-notice block
        # (S1 ONLY -- see check 12) and (b) blocks preceded by
        # <!-- fragment: illustrative, not executed -->.
        for f in _SECTION_FILES:
            allowed_prev = {"<!-- fragment: illustrative, not executed -->"}
            if f == "s01_ivadas.md":
                allowed_prev.add("<!-- mit-notice -->")
            lines = self.files[f].split("\n")
            open_at: int | None = None
            for idx, line in enumerate(lines):
                s = line.strip()
                if not _FENCE_RE.match(s):
                    continue
                info = s[3:].strip()
                if open_at is None:
                    if info:
                        self.err(f"check10: {f}:{idx+1} raw source has a ```{info} fence "
                                 "(listings/figures are inserted by build_manuscript, not hand-written)")
                    prev = next((lines[j].strip() for j in range(idx - 1, -1, -1) if lines[j].strip()), "")
                    if prev not in allowed_prev:
                        self.err(f"check10: {f}:{idx+1} unauthorized fenced block "
                                 "(only the S1 <!-- mit-notice --> block or a "
                                 "<!-- fragment: illustrative, not executed --> block)")
                    open_at = idx
                else:
                    if info:
                        self.err(f"check10: {f}:{idx+1} fence close carries an info string "
                                 f"{info!r} (nested or malformed fence)")
                    open_at = None
            if open_at is not None:
                self.err(f"check10: {f}:{open_at+1} fenced block is never closed")

    def c11_leak(self) -> None:
        try:
            targets = dict(raw_manuscript_markdown())
            targets["build(expanded)"] = expand(allow_pending=True)
        except Exception as exc:  # noqa: BLE001
            self.err(f"check11: expansion failed: {exc}")
            return
        for name, text in targets.items():
            for p in leak_scan(text, proj_commits=self.proj_commits, artifact_hashes=self.artifact_hashes):
                self.err(f"check11: {name}: {p}")

    def c12_licensing(self) -> None:
        named = set(re.findall(r"`([^`]+)`", self.notes_text))

        def is_named(*candidates: str) -> bool:
            return any(c in named for c in candidates)

        # Coverage is validated against the FIRST cell of each "Per-file coverage"
        # table row only -- normalized path, exactly once. Cross-references
        # elsewhere in the file and names in other sections do not count.
        cov = self.notes_text.split("## Per-file coverage", 1)[-1].split("\n## ", 1)[0]
        cov_cells: list[str] = []
        for line in cov.splitlines():
            if line.startswith("|") and "`" in line:
                cov_cells += re.findall(r"`([^`]+?)`", line.split("|")[1])

        def coverage_count(tok: str) -> int:
            return cov_cells.count(tok)

        for p in sorted(MANUSCRIPT.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(MANUSCRIPT).as_posix()
            n = coverage_count(f"manuscript/{rel}")
            if n == 0:
                self.err(f"check12: manuscript/{rel} has no `manuscript/...` coverage row in LICENSE-NOTES.md")
            elif n > 1:
                self.err(f"check12: manuscript/{rel} appears in {n} coverage rows (expected exactly one)")
        for label, tok in (("STATUS.json", "manuscript/STATUS.json"), (".gitattributes", ".gitattributes")):
            if coverage_count(tok) != 1:
                self.err(f"check12: {label} not named exactly once in the LICENSE-NOTES.md coverage table")
        for tool in ("scripts/lint_manuscript.py", "scripts/build_manuscript.py",
                     "tests/chapter_examples/test_manuscript_lint.py"):
            if not is_named(tool):
                self.err(f"check12: {tool} not named in LICENSE-NOTES.md")
        # each of D29-D36 named in LICENSE-CODE's MIT-covered scope (before the
        # "It does NOT cover:" section -- a name in the excluded list must not pass)
        licence_code = (REPO_ROOT / "examples" / "chapter" / "LICENSE-CODE").read_text(encoding="utf-8")
        covered_scope = licence_code.split("It does NOT cover:", 1)[0]
        for fx in _GOLDEN_FIXTURES:
            if covered_scope.count(fx) != 1:
                self.err(f"check12: golden fixture {fx!r} not named exactly once in LICENSE-CODE's covered scope")
        # exactly one <!-- mit-notice --> marker, and only in S1
        mit_hits = {f: len(re.findall(r"<!--\s*mit-notice\s*-->", t)) for f, t in self.files.items()}
        total_mit = sum(mit_hits.values())
        if total_mit != 1:
            self.err(f"check12: expected exactly one <!-- mit-notice --> marker, found {total_mit}")
        outside = sorted(f for f, n in mit_hits.items() if n and f != "s01_ivadas.md")
        if outside:
            self.err(f"check12: <!-- mit-notice --> must appear only in S1, also found in {outside}")
        digest = hashlib.sha256((MANUSCRIPT / "apa-7.csl").read_bytes()).hexdigest()
        if digest != _CSL_SHA256:
            self.err(f"check12: apa-7.csl sha256 {digest} != frozen")
        if _CSL_SHA256 not in (MANUSCRIPT / "THIRD_PARTY.md").read_text(encoding="utf-8"):
            self.err("check12: THIRD_PARTY.md does not record the frozen CSL SHA-256")
        # all three prose notices recorded verbatim (normalized) in LICENSE-NOTES.md
        notes_n = _norm(self.notes_text)
        for state, txt in _PROSE_NOTICE.items():
            if _norm(txt) not in notes_n:
                self.err(f"check12: LICENSE-NOTES.md is missing the {state!r} prose-licence notice")
        if self.clearance not in _PROSE_NOTICE:
            self.err(f"check12: clearance value {self.clearance!r} invalid")
        # S1 MIT notice body + active prose notice, when S01 is drafted
        s01 = self.files["s01_ivadas.md"]
        mit_body = re.split(r"\n-{10,}\n", licence_code, maxsplit=1)[0].strip()
        m = re.search(r"<!--\s*mit-notice\s*-->\s*\n```\n(.*?)\n```", s01, re.S)
        if m:
            if m.group(1).strip() != mit_body:
                self.err("check12: S1's MIT notice block != the LICENSE-CODE notice body")
        elif self.status.get("S01") == "complete":
            self.err("check12: S01 is complete but has no <!-- mit-notice --> block")
        if self.status.get("S01") == "complete":
            if self.clearance in _PROSE_NOTICE and _norm(_PROSE_NOTICE[self.clearance]) not in _norm(s01):
                self.err(f"check12: S1's prose-licence box != the {self.clearance!r} notice (normalized)")

    def c13_s13(self) -> None:
        s13 = self.files["s13_praktinis_pavyzdys.md"]
        blob = "\n".join(self.files.values())
        if self.registry_state == "pending_track_3":
            if len(s13.encode("utf-8")) > 1400:
                self.err("check13: S13 placeholder exceeds the byte ceiling")
            if re.search(r"<!--\s*(listing|claim|assignment|figure|sidebar):", s13):
                self.err("check13: S13 placeholder carries a forbidden directive")
            if re.search(r"\bTrack ?3?\b", strip_comments_and_fences(s13), re.I):
                self.err("check13: S13 placeholder visible text contains a 'Track' roadmap term")
            if len(re.findall(r"<!--\s*TRACK3-PENDING: S13", s13)) != 1:
                self.err("check13: S13 must carry exactly one 'TRACK3-PENDING: S13' token")
            if re.search(r"<!--\s*run-ref:", blob):
                self.err("check13: a <!-- run-ref: --> directive is present while the registry is pending_track_3")
            return
        if self.registry_state == "verified":
            if re.search(r"<!--\s*TRACK3-PENDING:", blob):
                self.err("check13: verified registry state but a TRACK3-PENDING token remains in the body")
            if len(s13.encode("utf-8")) <= 1400:
                self.err("check13: verified registry state but S13 is still placeholder-sized (expected full prose)")
            for f in _SECTION_FILES:
                if ("S" + f[1:3]) not in _ILLUSTRATION_SECTIONS:
                    continue
                lines = self.files[f].split("\n")
                ref_idx = [i for i, ln in enumerate(lines) if re.search(r"<!--\s*run-ref:", ln)]
                if len(ref_idx) != 1:
                    self.err(f"check13: {f} must carry exactly one <!-- run-ref: --> directive, found {len(ref_idx)}")
                    continue
                i = ref_idx[0]
                loc = re.search(r"<!--\s*run-ref:\s*(.*?)\s*-->", lines[i]).group(1).strip()
                if not loc:
                    self.err(f"check13: {f} <!-- run-ref: --> has an empty locator")

                def _is_prose(ln: str) -> bool:
                    t = ln.strip()
                    return bool(t) and not t.startswith(("<!--", "#", "```", "|", ">"))

                nbr = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
                if not any(_is_prose(x) for x in nbr):
                    self.err(f"check13: {f} <!-- run-ref: --> is not adjacent to its illustration paragraph "
                             "(must sit immediately before or after the run-derived sentences)")

    def c14_bibliography(self) -> None:
        s15 = self.files["s15_literatura.md"].strip()
        if s15 != "<!-- section:S15 -->\n## Literatūra\n\n::: {#refs}\n:::":
            self.err("check14: s15_literatura.md body is not exactly heading + the two-line #refs div")
        cited = set()
        for f in _SECTION_FILES:
            cited |= cited_keys(self.files[f])
        unused = self.bibkeys - cited
        if unused:
            self.report.append(f"check14 (report): {len(unused)} bib entries not yet cited: {sorted(unused)}")

    def c15_encoding(self) -> None:
        for p in list(MANUSCRIPT.rglob("*")) + [REPO_ROOT / "scripts" / "lint_manuscript.py",
                                                REPO_ROOT / "scripts" / "build_manuscript.py"]:
            if p.is_dir() or p.name == "apa-7.csl":
                continue
            b = p.read_bytes()
            if b[:3] == b"\xef\xbb\xbf":
                self.err(f"check15: {p} has a UTF-8 BOM")
            if b"\r" in b:
                self.err(f"check15: {p} contains a CR byte")
            try:
                b.decode("utf-8")
            except UnicodeDecodeError:
                self.err(f"check15: {p} not valid UTF-8")

    def c16_terminology(self) -> None:
        for term in _GLOSSARY_PRIMARY:
            n = sum(t.lower().count(term.lower()) for t in self.files.values())
            self.report.append(f"check16: primary term {term!r} appears {n}x")
        for alt in _GLOSSARY_ALTERNATIVES:
            for f, t in self.files.items():
                for i, line in enumerate(t.split("\n"), 1):
                    if alt.lower() in line.lower():
                        self.report.append(f"check16: alt term {alt!r} at {f}:{i} -- confirm it is used AS the alternative")

    # ================= COMPLETENESS CHECKS =================

    def c5_claim_anchors(self, sid: str, f: str) -> None:
        text = self.files[f]
        for cid in self.section_claims(sid):
            if len(re.findall(rf"<!--\s*claim:{cid}\s*-->", text)) != 1:
                self.err(f"check5: {f} does not have exactly one anchor for {cid}")
                continue
            para = paragraph_after(text, rf"<!--\s*claim:{cid}\s*-->")
            if not para:
                self.err(f"check5: {f} {cid} anchor not immediately followed by a paragraph")
                continue
            s_lit, q_lit, i_lit = self.claim_citesets(cid)
            cited = cited_keys(para)
            if s_lit - cited:
                self.err(f"check5: {f} {cid} paragraph missing supports citation(s): {sorted(s_lit - cited)}")
            if cited - (s_lit | q_lit | i_lit):
                self.err(f"check5: {f} {cid} paragraph has non-registry citation(s): {sorted(cited - (s_lit | q_lit | i_lit))}")

    def c6_listings(self, sid: str, f: str) -> None:
        ids = re.findall(r"<!--\s*listing:([a-z0-9-]+)\s*-->", self.files[f])
        want = _LISTING_MATRIX.get(sid, [])
        if sorted(ids) != sorted(want):
            self.err(f"check6: {f} listing ids {sorted(ids)} != the matrix set {sorted(want)}")
        for lid in ids:
            if lid not in self.manifest_ids:
                self.err(f"check6: {f} references unknown listing id {lid!r}")

    def c7_sidebars(self, sid: str, f: str) -> None:
        placed = re.findall(r"<!--\s*sidebar:([A-Za-z0-9-]+)\s*-->", self.files[f])
        want = _SIDEBAR_MATRIX.get(sid, [])
        if sorted(placed) != sorted(want):
            self.err(f"check7: {f} sidebar ids {sorted(placed)} != the matrix set {sorted(want)}")
        rows = {s["id"]: s for s in self.sidebars}
        for mid in placed:
            if mid not in rows:
                self.err(f"check7: {f} sidebar {mid!r} not in internal/sidebars.json")
                continue
            row = rows[mid]
            if canon(row["section"]) != canon(sid):
                self.err(f"check7: sidebar {mid} row.section {row['section']} != {sid}")
            full = self.proj_commits.get(row["proj_id"])
            if not full:
                self.err(f"check7: sidebar {mid} proj_id {row['proj_id']} not in the inventory")
            elif len(row["commit_prefix"]) < 7 or not full.startswith(row["commit_prefix"]):
                self.err(f"check7: sidebar {mid} commit_prefix {row['commit_prefix']!r} does not prefix {full}")

    def c8_assignments(self, sid: str, f: str) -> None:
        text = self.files[f]
        want = f"s{sid[1:]}"
        starts = re.findall(r"<!--\s*assignment:(s\d{2})\s+start\s*-->", text)
        ends = re.findall(r"<!--\s*assignment:(s\d{2})\s+end\s*-->", text)
        if starts != [want] or ends != [want]:
            self.err(f"check8: {f} assignment markers start={starts} end={ends} (want ['{want}'])")
            return
        body = re.search(rf"<!--\s*assignment:{want}\s+start\s*-->\n(.*?)<!--\s*assignment:{want}\s+end\s*-->",
                         text, re.S)
        if not body or not body.group(1).strip():
            self.err(f"check8: {f} assignment {want} has no authored content")
        if sid == "S03":
            if re.search(r"Kaip paleisti|python -m examples", text):
                self.err(f"check8: {f} (S03) must not carry a run/test command")
        else:
            from scripts.build_manuscript import _SLUGS
            n = int(sid[1:])
            if n in _SLUGS and not (REPO_ROOT / "examples" / "chapter" / _SLUGS[n]).is_dir():
                self.err(f"check8: examples/chapter/{_SLUGS[n]}/ missing")

    def c9_s1_citations(self, f: str) -> None:
        missing = _S01_REQUIRED_KEYS - cited_keys(self.files[f])
        if missing:
            self.err(f"check9: s01 missing required citation key(s): {sorted(missing)}")

    # ================= DRIVER =================

    def run(self, *, final: bool = False) -> None:
        self.c_status_json()
        self.c1_section_markers()
        self.c2_titles()
        self.c3_c27_31()
        self.c4_citations()
        self.c_visible_sids()
        self.c10_fences()
        self.c11_leak()
        self.c12_licensing()
        self.c13_s13()
        self.c14_bibliography()
        self.c15_encoding()
        self.c16_terminology()

        for f in _SECTION_FILES:
            sid = "S" + f[1:3]
            if self.status.get(sid) == "complete" or final:
                if self.section_claims(sid):
                    self.c5_claim_anchors(sid, f)
                if sid in _LISTING_MATRIX:
                    self.c6_listings(sid, f)
                if sid in _SIDEBAR_MATRIX:
                    self.c7_sidebars(sid, f)
                if sid in _ASSIGNMENT_SECTIONS:
                    self.c8_assignments(sid, f)
                if sid == "S01":
                    self.c9_s1_citations(f)

        if final:
            self._final()

    def _final(self) -> None:
        for sid in _ALL_SIDS:
            if self.status.get(sid) != "complete":
                self.err(f"--final: section {sid} is not 'complete' in STATUS.json")
        blob = "\n".join(self.files.values())
        anchors = re.findall(r"<!--\s*claim:(C\d{2})\s*-->", blob)
        want = [f"C{n:02d}" for n in range(1, 27)]
        if self.registry_state == "verified":
            want += [f"C{n}" for n in range(27, 32)]
        if sorted(anchors) != sorted(want):
            self.err(f"--final: claim anchor set != expected for state {self.registry_state}")
        listing_ids = re.findall(r"<!--\s*listing:([a-z0-9-]+)\s*-->", blob)
        if sorted(listing_ids) != sorted(self.manifest_ids):
            self.err(f"--final: listing multiset != the {len(self.manifest_ids)} manifest ids")
        placed = {m for f in _SECTION_FILES for m in re.findall(r"<!--\s*sidebar:([A-Za-z0-9-]+)\s*-->", self.files[f])}
        if placed != {s["id"] for s in self.sidebars}:
            self.err("--final: placed sidebars != sidebars.json rows")
        t3 = len(re.findall(r"<!--\s*TRACK3-PENDING:", blob))
        pf = len(re.findall(r"<!--\s*PHASEF-PENDING:", blob))
        if self.registry_state == "pending_track_3" and (t3, pf) != (11, 1):
            self.err(f"--final: expected 11 Track-3 + 1 Phase-F tokens in pending state, found {t3}/{pf}")
        if self.registry_state == "verified" and (t3, pf) != (0, 1):
            self.err(f"--final: expected 0 Track-3 + 1 Phase-F token in verified state, found {t3}/{pf}")


def generate_placeholders(files: dict[str, str]) -> str:
    t3, pf, frag = [], [], []
    for f in _SECTION_FILES:
        for i, line in enumerate(files[f].split("\n"), 1):
            m = re.search(r"<!--\s*TRACK3-PENDING:\s*(.*?)\s*-->", line)
            if m:
                t3.append((m.group(1).split(" — ")[0].strip(), f"manuscript/{f}", i, m.group(1)))
            m = re.search(r"<!--\s*PHASEF-PENDING:\s*(.*?)\s*-->", line)
            if m:
                pf.append((m.group(1), f"manuscript/{f}", i))
            if "fragment: illustrative, not executed" in line:
                frag.append((f"manuscript/{f}", i))
    out = ["<!-- GENERATED by scripts/lint_manuscript.py --write. Do not hand-edit. -->", "",
           "# Placeholders", "", "## Track-3 pending", "",
           "| scope | file | line | description |", "|---|---|---|---|"]
    out += [f"| {s} | {p} | {i} | {d} |" for s, p, i, d in t3]
    out += ["", "## Phase-F pending", "", "| what | file | line |", "|---|---|---|"]
    out += [f"| {w} | {p} | {i} |" for w, p, i in pf]
    out += ["", "## Permanent fragment annotations", "", "| file | line |", "|---|---|"]
    out += [f"| {p} | {i} |" for p, i in frag]
    out.append("")
    return "\n".join(out)


def _resolve_ref(ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                          cwd=REPO_ROOT, capture_output=True).returncode == 0


class BaselineError(Exception):
    """The STATUS.json at <ref> is present but not a usable JSON object."""


def _baseline_status(ref: str) -> dict | None:
    """None -> invalid ref; {} -> valid ref, file genuinely absent; dict -> the
    parsed file. Raises BaselineError if the file is present but malformed --
    a malformed baseline must NOT be silently treated as an absent one."""
    if not _resolve_ref(ref):
        return None
    r = subprocess.run(["git", "show", f"{ref}:manuscript/STATUS.json"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"STATUS.json at {ref} is present but not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BaselineError(f"STATUS.json at {ref} is present but is not a JSON object")
    return {k: v for k, v in data.items() if k.startswith("S")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-regress", metavar="REF")
    args = ap.parse_args()

    if args.no_regress is not None:
        try:
            baseline = _baseline_status(args.no_regress)
        except BaselineError as exc:
            print(f"lint --no-regress: {exc}", file=sys.stderr)
            return 1
        if baseline is None:
            print(f"lint --no-regress: {args.no_regress!r} is not a valid git ref", file=sys.stderr)
            return 1
        current = {k: v for k, v in json.loads((MANUSCRIPT / "STATUS.json").read_text(encoding="utf-8")).items()
                   if k.startswith("S")}
        problems = check_no_regress(baseline, current)
        if problems:
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"lint --no-regress {args.no_regress}: OK "
              f"({'valid ref, no STATUS.json (baseline = all skeleton)' if not baseline else 'baseline compared'})")
        return 0

    lint = Lint()
    placeholders = generate_placeholders(lint.files)
    ph = MANUSCRIPT / "PLACEHOLDERS.md"
    want_ph = (placeholders.rstrip("\n") + "\n").encode("utf-8")

    if args.write:
        lint.run(final=False)
        if lint.errors:
            print(f"\nlint_manuscript --write: {len(lint.errors)} error(s); "
                  "PLACEHOLDERS.md left unchanged:", file=sys.stderr)
            for e in lint.errors:
                print("  " + e.encode("ascii", "replace").decode("ascii"), file=sys.stderr)
            return 1
        if not ph.is_file() or ph.read_bytes() != want_ph:
            ph.write_bytes(want_ph)
            print(f"wrote {ph}")
        else:
            print(f"{ph} already current")
        return 0

    if (ph.read_text(encoding="utf-8") if ph.is_file() else "").rstrip("\n") != placeholders.rstrip("\n"):
        lint.err("PLACEHOLDERS.md is stale -- run `lint_manuscript.py --write` and commit it")
    lint.run(final=args.final)

    for line in lint.report:
        print("  - " + line.encode("ascii", "replace").decode("ascii"))
    if lint.errors:
        print(f"\nlint_manuscript: {len(lint.errors)} error(s):", file=sys.stderr)
        for e in lint.errors:
            print("  " + e.encode("ascii", "replace").decode("ascii"), file=sys.stderr)
        return 1
    print(f"\nlint_manuscript {'--final ' if args.final else ''}OK "
          f"(registry: {lint.registry_state}; clearance: {lint.clearance})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
