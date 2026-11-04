"""
screen_candidate_reviews.py — Quick feasibility screen for candidate review PDFs.

For each PDF in test_reviews/candidate_reviews/, extracts:
  - Review type (scoping / rapid / SLR)
  - Number of included studies
  - Search databases used
  - Date range
  - Whether full-text or abstract-only analysis is reported

Outputs a console table + candidate_reviews_screen.csv with a replicability
assessment:
  FULL        — n ≤ 80, all databases accessible via API
  PARTIAL     — n 81–200, or some databases unavailable
  SEARCH_ONLY — n > 200, or only proprietary databases (no free API)
  SKIP        — cannot determine n, or clearly non-replicable

Usage:
    python screen_candidate_reviews.py [--model gpt-5.4-mini] [--folder test_reviews/candidate_reviews]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent / "slr_pipeline"))
sys.path.insert(0, r"c:/projects")
import my_api_keys  # sets OPENAI_API_KEY and other keys as env vars  # noqa: F401

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Databases accessible via free/open API in our pipeline
FREE_API_DATABASES = {
    "openalex", "pubmed", "medline", "semantic scholar", "arxiv",
    "crossref", "europe pmc", "europepmc", "pmc",
}

# Proprietary databases (partial replication only)
PROPRIETARY_DATABASES = {
    "web of science", "wos", "scopus", "embase", "cinahl", "psycinfo",
    "psychinfo", "psycinfo", "cochrane", "ebsco", "proquest",
    "springerlink", "ieee xplore", "acm digital library",
    "business source", "academic search",
}

SCREEN_PROMPT = """You are screening a published systematic or scoping review to assess pipeline replication feasibility.

Read the text below (from the paper's abstract and methods section) and extract these fields.
Return ONLY valid JSON — no explanations.

Text:
{text}

JSON to return:
{{
  "review_type": "scoping" | "rapid" | "slr" | "other" | "unknown",
  "title": "short paper title (max 80 chars)",
  "first_author_year": "Smith 2023",
  "n_included": <integer number of studies finally included, or null if not found>,
  "n_identified": <integer total records identified before screening, or null>,
  "databases": ["exact database names as reported in paper"],
  "date_from": "YYYY or null",
  "date_to": "YYYY or null",
  "language_restriction": ["English"] or [],
  "has_prisma": true | false,
  "has_fulltext_extraction": true | false,
  "notes": "anything unusual about this review that affects replication"
}}

For n_included: look for phrases like "X studies were included", "final sample of X",
"X articles met inclusion criteria". For databases: list them exactly as named in the paper.
"""

# ---------------------------------------------------------------------------
# Free API availability assessment
# ---------------------------------------------------------------------------

def _api_availability(databases: List[str]) -> Dict[str, str]:
    """
    Return dict: database_name → "free_api" | "proprietary" | "unknown"
    """
    out = {}
    for db in databases:
        key = db.lower().strip()
        if any(f in key for f in FREE_API_DATABASES):
            out[db] = "free_api"
        elif any(p in key for p in PROPRIETARY_DATABASES):
            out[db] = "proprietary"
        else:
            out[db] = "unknown"
    return out


def _search_replicability(db_availability: Dict[str, str]) -> str:
    """High / Partial / Low based on database coverage."""
    vals = list(db_availability.values())
    if not vals:
        return "unknown"
    free = vals.count("free_api")
    prop = vals.count("proprietary")
    total = len(vals)
    if prop == 0:
        return "high"
    elif free >= prop:
        return "partial"
    else:
        return "low"


def _replication_tier(n_included: Optional[int], search_rep: str) -> str:
    """
    FULL        — n ≤ 80, search replicability high/partial
    PARTIAL     — n 81–200, or search replicability partial
    SEARCH_ONLY — n > 200, or search replicability low
    MANUAL      — n present but very large (> 500)
    UNKNOWN     — cannot determine
    """
    if n_included is None:
        return "UNKNOWN"
    if n_included > 500:
        return "SEARCH_ONLY (n>500)"
    if n_included > 200:
        if search_rep == "low":
            return "SEARCH_ONLY"
        return "SEARCH_ONLY (large n)"
    if n_included > 80:
        return "PARTIAL"
    # n ≤ 80
    if search_rep == "low":
        return "PARTIAL"
    return "FULL"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_text_for_screening(pdf_path: Path, max_chars: int = 20000) -> str:
    """Extract first ~20k chars of text from PDF using pymupdf4llm page chunks."""
    try:
        import pymupdf4llm
        pages = pymupdf4llm.to_markdown(
            doc=str(pdf_path),
            page_chunks=True,
            table_strategy="lines",
            write_images=False,
            margins=0,
        )
        # First 10 pages covers abstract + methods for most reviews
        text = "\n\n".join(p.get("text", "") for p in pages[:10])
        return text[:max_chars]
    except Exception as e:
        print(f"    [WARN] PDF extraction failed: {e}")
        return ""


def _screen_pdf(pdf_path: Path, client: OpenAI, model: str) -> Dict[str, Any]:
    """Run LLM screening on a single PDF."""
    print(f"  Processing: {pdf_path.name}")
    text = _extract_text_for_screening(pdf_path)
    if not text.strip():
        return {"error": "could not extract text", "pdf": pdf_path.name}

    prompt = SCREEN_PROMPT.format(text=text[:18000])
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        result["pdf"] = pdf_path.name
        result["pdf_path"] = str(pdf_path)
        return result
    except Exception as e:
        print(f"    [WARN] LLM call failed: {e}")
        return {"error": str(e), "pdf": pdf_path.name}


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_WIDTH = {
    "File": 35, "Type": 8, "n incl.": 8, "n ident.": 9,
    "Search rep.": 12, "Tier": 22, "Databases": 45, "Years": 11,
}

def _trunc(s: str, n: int) -> str:
    s = str(s) if s is not None else "—"
    return s if len(s) <= n else s[:n-1] + "…"


def _print_header() -> None:
    cols = list(_COL_WIDTH.items())
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    sep = "  ".join("-" * w for _, w in cols)
    print("\n" + sep)
    print(header)
    print(sep)


def _print_row(r: Dict[str, Any]) -> None:
    db_avail = _api_availability(r.get("databases") or [])
    search_rep = _search_replicability(db_avail)
    tier = _replication_tier(r.get("n_included"), search_rep)
    db_str = ", ".join(r.get("databases") or []) or "—"

    years = "—"
    if r.get("date_from") or r.get("date_to"):
        years = f"{r.get('date_from','?')}–{r.get('date_to','?')}"

    print("  ".join([
        _trunc(r.get("pdf", ""), _COL_WIDTH["File"]),
        f"{_trunc(r.get('review_type','?'), _COL_WIDTH['Type']):<{_COL_WIDTH['Type']}}",
        f"{str(r.get('n_included') or '?'):>{_COL_WIDTH['n incl.']-1}} ",
        f"{str(r.get('n_identified') or '?'):>{_COL_WIDTH['n ident.']-1}} ",
        f"{search_rep:<{_COL_WIDTH['Search rep.']}}",
        f"{tier:<{_COL_WIDTH['Tier']}}",
        _trunc(db_str, _COL_WIDTH["Databases"]),
        _trunc(years, _COL_WIDTH["Years"]),
    ]))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "pdf", "first_author_year", "title", "review_type",
    "n_included", "n_identified", "databases_raw", "databases_free_api",
    "databases_proprietary", "search_replicability", "replication_tier",
    "date_from", "date_to", "language_restriction",
    "has_prisma", "has_fulltext_extraction", "notes", "error",
]


def _to_csv_row(r: Dict[str, Any]) -> Dict[str, Any]:
    dbs = r.get("databases") or []
    avail = _api_availability(dbs)
    free = [d for d, v in avail.items() if v == "free_api"]
    prop = [d for d, v in avail.items() if v == "proprietary"]
    search_rep = _search_replicability(avail)
    tier = _replication_tier(r.get("n_included"), search_rep)

    return {
        "pdf": r.get("pdf", ""),
        "first_author_year": r.get("first_author_year", ""),
        "title": r.get("title", ""),
        "review_type": r.get("review_type", ""),
        "n_included": r.get("n_included", ""),
        "n_identified": r.get("n_identified", ""),
        "databases_raw": "; ".join(dbs),
        "databases_free_api": "; ".join(free),
        "databases_proprietary": "; ".join(prop),
        "search_replicability": search_rep,
        "replication_tier": tier,
        "date_from": r.get("date_from", ""),
        "date_to": r.get("date_to", ""),
        "language_restriction": "; ".join(r.get("language_restriction") or []),
        "has_prisma": r.get("has_prisma", ""),
        "has_fulltext_extraction": r.get("has_fulltext_extraction", ""),
        "notes": r.get("notes", ""),
        "error": r.get("error", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def screen_candidate_reviews(
    folder: Path,
    model: str = "gpt-5.4-mini",
    output_csv: Optional[Path] = None,
) -> List[Dict[str, Any]]:

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"\nNo PDF files found in: {folder}")
        print("Drop your candidate review PDFs there and re-run.")
        return []

    print(f"\n{'='*60}")
    print(f"Screening {len(pdfs)} candidate review(s) in: {folder.name}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    results = []

    for pdf_path in pdfs:
        r = _screen_pdf(pdf_path, client, model)
        results.append(r)

    # --- Print table ---
    _print_header()
    for r in results:
        _print_row(r)
    print()

    # --- Tier summary ---
    print("Replication tier legend:")
    print("  FULL          — n ≤ 80, search replicable via free APIs")
    print("  PARTIAL       — n 81–200, or some databases unavailable")
    print("  SEARCH_ONLY   — n > 200, or mostly proprietary databases")
    print("  UNKNOWN       — could not determine n or databases")
    print()

    # --- Per-review recommendation ---
    print("Recommendations:")
    for r in results:
        if "error" in r:
            print(f"  [FAIL] {r['pdf']}: extraction failed — check PDF")
            continue
        dbs = r.get("databases") or []
        avail = _api_availability(dbs)
        prop = [d for d, v in avail.items() if v == "proprietary"]
        search_rep = _search_replicability(avail)
        tier = _replication_tier(r.get("n_included"), search_rep)

        n = r.get("n_included")
        label = r.get("first_author_year") or r.get("pdf", "")
        print(f"  {label}")
        print(f"    Type: {r.get('review_type','?')}  |  Included: {n or '?'}  |  Tier: {tier}")
        if prop:
            print(f"    [WARN] Proprietary DBs (no free API): {', '.join(prop)}")
            print(f"       → Search stage will miss records from these sources")
        if n and n > 150:
            print(f"    [WARN] Large corpus ({n} papers) — LLM full-text analysis will be costly")
            print(f"       → Recommend --skip-ft for initial run; full-text optional")
        if n and n <= 80 and search_rep in ("high", "partial"):
            print(f"    [OK] Good candidate for full pipeline reproduction")
        print()

    # --- Save CSV ---
    if output_csv is None:
        output_csv = folder.parent / "candidate_reviews_screen.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_to_csv_row(r) for r in results)

    print(f"Saved: {output_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen candidate review PDFs for replication feasibility")
    parser.add_argument(
        "--folder",
        default="test_reviews/candidate_reviews",
        help="Folder containing candidate review PDFs",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Screen a single PDF file (path relative to folder or absolute)",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = Path(__file__).parent / folder

    out_csv = Path(args.output_csv) if args.output_csv else None

    if args.file:
        # Single-file mode: screen one PDF and append to existing CSV
        pdf_path = Path(args.file)
        if not pdf_path.is_absolute():
            pdf_path = folder / args.file
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        r = _screen_pdf(pdf_path, client, args.model)
        _print_header()
        _print_row(r)
        print()
        # Append to CSV
        if out_csv is None:
            out_csv = folder.parent / "candidate_reviews_screen.csv"
        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(_to_csv_row(r))
        print(f"Appended to: {out_csv}")
        # Print recommendation
        dbs = r.get("databases") or []
        avail = _api_availability(dbs)
        prop = [d for d, v in avail.items() if v == "proprietary"]
        search_rep = _search_replicability(avail)
        tier = _replication_tier(r.get("n_included"), search_rep)
        n = r.get("n_included")
        print(f"  {r.get('first_author_year') or pdf_path.name}")
        print(f"    Type: {r.get('review_type','?')}  |  Included: {n or '?'}  |  Tier: {tier}")
        if prop:
            print(f"    [WARN] Proprietary DBs: {', '.join(prop)}")
        if n and n <= 80 and search_rep in ("high", "partial"):
            print(f"    [OK] Good candidate for full pipeline reproduction")
    else:
        screen_candidate_reviews(folder, model=args.model, output_csv=out_csv)
