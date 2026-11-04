"""
Chapter demonstration: deductive data extraction, simulating Thabit et al.
2025's own coding process as closely as a single LLM coder can.

Their method (see NOTES.md's "Data extraction methodology" section, ported
from `_thabit_full.txt`'s "Data collection and analysis"): thematic,
abductive coding against a FIXED codebook (Table 2 -- 1 aggregate dimension,
6 themes, 13 components, 37 key concepts), applied by 3 human coders with
every article dual-coded and disagreements resolved by discussion. This
script applies the SAME codebook (`ft_codebook.CODEBOOK`, transcribed
verbatim from Table 2) deductively with ONE LLM coder per article -- not a
reproduction of their inter-rater process, a single-annotator approximation
of it. That gap (1 LLM pass vs. 3 human coders + calibration + dual-coding +
disagreement resolution) is itself the chapter's teaching point, not glossed
over.

For each of the 10 known_included pilot full texts, one call asks the model
to walk EVERY ONE of the 37 concepts and decide, for each: present in this
article's text or not, with a supporting quote + page if present. Compared
against Table C1's own gold reference list (`ft_codebook.gold_refs()`) via
`compare_to_gold.py`.

Usage:
  PYEXE="python"
  $PYEXE run_deductive_extraction.py           # skips already-cached records
  $PYEXE run_deductive_extraction.py --force   # re-runs everything (re-pays)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"c:/projects")
from SEARCHING.keys import get_key

import diskcache
from openai import OpenAI

from ft_codebook import CODEBOOK, REC_ID_TO_REF

HERE = Path(__file__).parent
PILOT_DIR = HERE / "fulltext_pilot"
TEXT_DIR = PILOT_DIR / "text"

MODEL = "gpt-5.4-mini"
MAX_CHARS = 100_000

# v1 (first pass) used a low bar -- "the passage must actually support the
# concept's meaning as named" -- which any brief, passing mention already
# satisfies. Against Table C1 gold this produced near-ceiling recall (0.92)
# but poor precision (0.41): the LLM flagged ~33/37 concepts per article on
# average vs. the paper's own coders' ~15.
#
# v2 tightened the threshold wording alone (require SUBSTANTIVE findings/
# discussion treatment, not background/passing mention). Measured effect
# against gold: precision 0.409->0.436, recall 0.921->0.875, F1 0.567->0.582.
# FP only dropped 202->172; LLM still flagged ~30/37 per article. Wording
# alone does not close the gap -- treated as a structural over-coding
# tendency, not a prompt-adjective problem.
#
# v3 adds two more levers: (a) few-shot anchoring -- two concrete worked
# examples (one found, one not-found) showing the SAME borderline case
# resolved both ways, since abstract wording ("substantive" vs "passing")
# is itself vague without an anchor; (b) a grounded cardinality calibration
# note -- across all 58 scoreable articles in Thabit et al.'s own Table C1,
# concepts-found-per-article has median=13, mean=13.7, range 2-25 (computed
# directly from `ft_codebook.gold_refs()`, not invented) -- given as a
# real-world prior a calibrated coder would hold, mirroring the function of
# Thabit et al.'s own calibration phase (6 jointly-reviewed practice
# articles), not as a hard cap to game the eval metric against.
# Measured: v3 vs v2, F1 0.582->0.581 -- flat, noise-level. Wording-only
# levers (bar phrasing, worked examples, a soft numeric prior) are exhausted:
# independent per-concept thresholding lets ~30/37 pass regardless of how
# the bar is phrased, because the model evaluates each concept in isolation
# and a plausible-sounding passage exists for most of the 37 in any
# ~8000-word article. v4 changes the MECHANISM, not the wording: force
# relative ranking across all 37 (not independent absolute yes/no per
# concept) plus a hard cap on how many can be marked found, so the model
# must trade off weaker candidates against stronger ones instead of passing
# every candidate that individually clears a vague bar.
PROMPT_VERSION = "v4"
CACHE_DIR = PILOT_DIR / f"deductive_cache_{PROMPT_VERSION}"
OUT_PATH = PILOT_DIR / f"deductive_extracted_{PROMPT_VERSION}.jsonl"

MAX_FOUND = 18  # v4 hard cap, enforced in code below (not left to model arithmetic)

SYSTEM_PROMPT = """\
You are a systematic-review coder performing DEDUCTIVE thematic coding \
against a FIXED, pre-defined codebook -- you do not invent new codes, \
rename concepts, or merge/split them. For EVERY concept in the codebook, \
decide whether the article's text provides evidence for it, AND rate how \
strong that evidence is relative to the other 36 concepts.

Apply a STRICT bar, matching a calibrated human coder, not a keyword search:
a concept is "found" ONLY if the article's own findings, results, or \
discussion substantively examines that concept as something the studied \
case(s) actually exhibit or address -- e.g. the article presents evidence, \
analysis, or a specific example bearing on it. It is NOT found if the \
concept appears only in the literature review/theoretical background, only \
as a passing single-sentence mention with no elaboration, or only as a \
generic term used in a different sense than the concept's own definition. \
When genuinely uncertain whether a mention rises to this bar, mark it NOT \
found -- false positives (over-coding) are the failure mode to avoid.

Worked examples (illustrative only, not from the article you will code):
- Concept "Trust" -- FOUND, strength 90: the results section states \
"Interviewees repeatedly described how early joint wins built mutual trust \
between the municipality and the housing association, which later let them \
skip formal sign-off steps" -- this is the case's own finding about trust, \
with a specific mechanism and consequence.
- Concept "Equal representation and inclusion" -- NOT found, strength 10: \
the introduction states "prior work stresses that governance arrangements \
should be equitable and inclusive of all affected groups" and the concept \
is never revisited in the findings or discussion -- background framing, \
not something the case is shown to exhibit, even though the words \
"equitable" and "inclusive" appear.

After judging all 37 independently, ALSO assign each a "strength" score \
1-100: how central and well-evidenced this concept's treatment is in the \
article RELATIVE to the other 36 concepts you just judged (not an absolute \
probability). Only a small minority of articles' concepts should score \
above 70 -- reserve high scores for the concepts the article's own \
discussion or conclusion treats as central, not every concept that \
technically clears the strict bar.

Return ONLY a valid JSON object: {"codes": [...]} with exactly one entry per \
codebook row, each: {"concept": "<exact concept name as given>", "found": \
true|false, "strength": <integer 1-100>, "evidence_quote": "<verbatim quote \
<=40 words, or null if not found>", "page": <integer, nearest preceding \
[PAGE N] marker, or null>}. No markdown, no explanation outside the JSON \
object.\
"""

USER_TEMPLATE = """\
Codebook ({n} concepts, grouped by theme > component):
{codebook_listing}

Article title: {title}

Article full text (page-anchored via [PAGE N] markers, may be truncated at \
{max_chars} characters):
{text}
"""


def _cache() -> diskcache.Cache:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(CACHE_DIR))


def _cache_key(rec_id: str) -> str:
    return f"deductive:{PROMPT_VERSION}:{MODEL}:{rec_id}"


def _codebook_listing() -> str:
    lines = []
    last_theme = last_component = None
    for theme, component, concept in CODEBOOK:
        if theme != last_theme:
            lines.append(f"- {theme}")
            last_theme = theme
            last_component = None
        if component != last_component:
            lines.append(f"  - {component}")
            last_component = component
        lines.append(f"    - {concept}")
    return "\n".join(lines)


def _apply_cap(codes: list[dict], max_found: int) -> list[dict]:
    """Keep 'found' only for the top-`max_found` codes by strength; force the
    rest to found=False. Codes are cached/stored uncapped so MAX_FOUND can be
    retuned without re-paying for the underlying model call."""
    found = [c for c in codes if c.get("found")]
    found_sorted = sorted(found, key=lambda c: c.get("strength", 0), reverse=True)
    keep_ids = {id(c) for c in found_sorted[:max_found]}
    return [
        {**c, "found": c.get("found") and id(c) in keep_ids}
        for c in codes
    ]


def _title_for(rec_id: str) -> str:
    import json as _json

    manifest = _json.loads((PILOT_DIR / "tier1_manifest.json").read_text(encoding="utf-8"))
    for rec in manifest:
        if rec["rec_id"] == rec_id:
            return rec["title"]
    return rec_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache, re-run and re-pay for everything")
    args = ap.parse_args()

    client = OpenAI(api_key=get_key("openai"))
    cache = _cache()
    codebook_listing = _codebook_listing()

    results: dict[str, dict] = {}
    if OUT_PATH.exists() and not args.force:
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["rec_id"]] = row

    for rec_id, ref in REC_ID_TO_REF.items():
        if rec_id in results and not args.force:
            print(f"skip  {rec_id} ({ref}) -- already extracted")
            continue

        cache_key = _cache_key(rec_id)
        cached = None if args.force else cache.get(cache_key)
        if cached is not None:
            print(f"cache {rec_id} ({ref})")
            title = _title_for(rec_id)
            results[rec_id] = {"rec_id": rec_id, "ref": ref, "title": title, "codes": _apply_cap(cached, MAX_FOUND)}
            continue

        safe = re.sub(r"[^A-Za-z0-9]+", "_", rec_id)
        text_path = TEXT_DIR / f"{safe}.txt"
        text = text_path.read_text(encoding="utf-8", errors="replace")[:MAX_CHARS]
        title = _title_for(rec_id)

        prompt = USER_TEMPLATE.format(
            n=len(CODEBOOK), codebook_listing=codebook_listing,
            title=title, max_chars=MAX_CHARS, text=text,
        )

        print(f"call  {rec_id} ({ref}) -- {title[:70]!r} live paid call")
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_completion_tokens=6000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        codes = json.loads(content)["codes"]

        cache.set(cache_key, codes)
        results[rec_id] = {"rec_id": rec_id, "ref": ref, "title": title, "codes": _apply_cap(codes, MAX_FOUND)}

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for rec_id in REC_ID_TO_REF:
            if rec_id in results:
                out_f.write(json.dumps(results[rec_id], ensure_ascii=False) + "\n")

    print(f"\n-> {OUT_PATH} ({len(results)}/{len(REC_ID_TO_REF)} records)")


if __name__ == "__main__":
    main()
