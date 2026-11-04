"""
Theme-level narrative synthesis -- the data-charting/synthesis stage beyond
per-article deductive extraction (`run_deductive_extraction.py`). Thabit et
al. 2025's own Findings section isn't a per-article found/quote table, it's
cross-study aggregative prose per theme: what's convergent across cases,
what's divergent, named outliers, illustrative quotes cited to specific
studies (see NOTES.md's "Deductive data extraction pilot" section and
`theme_reference_text.json`, built by `build_theme_reference.py`).

This script does the same aggregation step, one LLM call per theme (6
total), using ONLY the evidence this pipeline already extracted (v2's
per-article found/quote pairs across the 10 known_included pilot articles --
`fulltext_pilot/deductive_extracted_v2.jsonl`, chosen over v1/v3/v4 because
it has no artificial cap and the tightened v2 bar is the best signal-to-noise
ratio measured against gold, see NOTES.md). The model is given the raw
evidence bag per theme (concept, article title/ref, quote, page) and asked
to write cross-study synthesis prose in the same aggregative style --
explicitly NOT asked to reproduce or match Thabit et al.'s own Figure 3
"unifying framework" (that is the authors' own theorizing contribution, not
an extraction/synthesis target, see NOTES.md).

Usage:
  PYEXE="python"
  $PYEXE run_theme_synthesis.py           # skips already-cached themes
  $PYEXE run_theme_synthesis.py --force   # re-runs everything (re-pays)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"c:/projects")
from SEARCHING.keys import get_key

import diskcache
from openai import OpenAI

from ft_codebook import CODEBOOK, REC_ID_TO_REF

HERE = Path(__file__).parent
PILOT_DIR = HERE / "fulltext_pilot"
EXTRACTED_PATH = PILOT_DIR / "deductive_extracted_v2.jsonl"
CACHE_DIR = HERE / "theme_synthesis_cache"
OUT_PATH = HERE / "theme_synthesis.json"

MODEL = "gpt-5.4-mini"

# Table 2's own 6 themes, in order -- matches ft_codebook.CODEBOOK's theme
# strings exactly.
THEMES: list[str] = []
for theme, _, _ in CODEBOOK:
    if theme not in THEMES:
        THEMES.append(theme)
assert len(THEMES) == 6

SYSTEM_PROMPT = """\
You are a systematic-review author drafting the cross-study Findings \
narrative for ONE theme of a thematic synthesis. You are given, for a \
single theme, all the (concept, article, quote) evidence a prior deductive \
coding pass extracted from a small set of included articles.

Write 2-4 paragraphs of cross-study synthesis prose in the aggregative \
style of a published qualitative systematic review Findings section: \
identify what is convergent across the articles (patterns most cases \
agree on), what is divergent (tensions, competing approaches, named \
outliers -- e.g. "however, X et al. shows a different pattern"), and use \
short illustrative quotes attributed to the specific article's title (not \
invented citation keys, since you were not given the articles' bibliographic \
citation format). Organize loosely by the theme's components (given below), \
but the paragraphs need not map 1:1 to components. Base every claim ONLY on \
the evidence given -- do not invent findings, and if the evidence for a \
concept is thin (few or no articles), say so plainly rather than padding.

Do NOT attempt to build or reference a wider theoretical model, framework \
diagram, or cross-theme integration -- that is out of scope for this task, \
which is per-theme aggregative synthesis only.

Return ONLY a valid JSON object: {"synthesis": "<the prose, as a single \
string with paragraphs separated by blank lines>"}. No markdown fencing, \
no explanation outside the JSON object.\
"""

USER_TEMPLATE = """\
Theme: {theme}

Components and concepts under this theme:
{components_listing}

Evidence extracted from {n_articles} articles (concept -- article title \
(ref) -- quote (page)):
{evidence_listing}
"""


def _cache() -> diskcache.Cache:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(CACHE_DIR))


def _cache_key(theme: str) -> str:
    return f"theme_synthesis:v1:{MODEL}:{theme}"


def _components_listing(theme: str) -> str:
    lines = []
    last_component = None
    for t, component, concept in CODEBOOK:
        if t != theme:
            continue
        if component != last_component:
            lines.append(f"- {component}")
            last_component = component
        lines.append(f"  - {concept}")
    return "\n".join(lines)


def _load_extracted() -> dict[str, dict]:
    rows = {}
    for line in EXTRACTED_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["rec_id"]] = row
    return rows


def _evidence_listing(theme: str, extracted: dict[str, dict]) -> tuple[str, int]:
    theme_concepts = {concept for t, _, concept in CODEBOOK if t == theme}
    lines = []
    articles_with_evidence = set()
    for rec_id, ref in REC_ID_TO_REF.items():
        row = extracted.get(rec_id)
        if not row:
            continue
        title = row.get("title", rec_id)
        for code in row["codes"]:
            if code.get("concept") not in theme_concepts or not code.get("found"):
                continue
            quote = code.get("evidence_quote") or ""
            page = code.get("page")
            lines.append(f"- {code['concept']} -- {title!r} ({ref}) -- \"{quote}\" (p.{page})")
            articles_with_evidence.add(rec_id)
    if not lines:
        return "(no evidence found for any concept under this theme across the 10 articles)", 0
    return "\n".join(lines), len(articles_with_evidence)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache, re-run and re-pay for everything")
    args = ap.parse_args()

    client = OpenAI(api_key=get_key("openai"))
    cache = _cache()
    extracted = _load_extracted()

    results: dict[str, str] = {}
    if OUT_PATH.exists() and not args.force:
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    for theme in THEMES:
        if theme in results and not args.force:
            print(f"skip  {theme!r} -- already synthesized")
            continue

        cache_key = _cache_key(theme)
        cached = None if args.force else cache.get(cache_key)
        if cached is not None:
            print(f"cache {theme!r}")
            results[theme] = cached
            continue

        components_listing = _components_listing(theme)
        evidence_listing, n_articles = _evidence_listing(theme, extracted)

        prompt = USER_TEMPLATE.format(
            theme=theme, components_listing=components_listing,
            n_articles=n_articles, evidence_listing=evidence_listing,
        )

        print(f"call  {theme!r} -- {n_articles} articles with evidence -- live paid call")
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_completion_tokens=3000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        synthesis = json.loads(content)["synthesis"]

        cache.set(cache_key, synthesis)
        results[theme] = synthesis

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        json.dump({t: results[t] for t in THEMES if t in results}, out_f, ensure_ascii=False, indent=2)

    print(f"\n-> {OUT_PATH} ({len(results)}/{len(THEMES)} themes)")


if __name__ == "__main__":
    main()
