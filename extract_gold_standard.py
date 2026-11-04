"""
extract_gold_standard.py — Extract included papers from a review PDF.

Usage:
    python extract_gold_standard.py <review_dir> [--model MODEL] [--pages N]

Where <review_dir> contains review.pdf and optionally:
  - included_papers.csv  (user-curated; takes priority)
  - supplementary.xlsx / supplementary.csv / supplementary.pdf

Outputs to <review_dir>/:
  - gold_standard.json   (GoldStandard model for evaluation)
  - extracted_papers.csv (auto-extracted; rename to included_papers.csv after review)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- API keys ---
_PROJECTS = r"c:/projects"
if _PROJECTS not in sys.path:
    sys.path.insert(0, _PROJECTS)
try:
    import my_api_keys  # noqa: F401
except ImportError:
    pass

import fitz  # PyMuPDF
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def parse_pdf_pages(pdf_path: Path) -> List[str]:
    """Extract text from each page using fitz (no OCR)."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        pages.append(text)
    doc.close()
    return pages


def pages_to_text(pages: List[str], max_chars: int = 40000) -> str:
    """Join pages, truncate to max_chars."""
    full = "\n\n".join(pages)
    return full[:max_chars]


def find_reference_section(pages: List[str]) -> str:
    """
    Return text from 'Included studies' appendix or reference section.
    Prefers sections explicitly labelled 'Included studies' or 'References'.
    """
    markers = [
        r"included\s+studies",
        r"appendix[\s\w]*included",
        r"list\s+of\s+included",
        r"references",
        r"bibliography",
    ]
    full_text = "\n\n".join(pages)
    best_pos = len(full_text)
    best_match = None
    for pattern in markers:
        for m in re.finditer(pattern, full_text, re.IGNORECASE):
            if m.start() < best_pos:
                # Prefer "included studies" over generic "references"
                if best_match is None or "included" in pattern or "list" in pattern:
                    best_pos = m.start()
                    best_match = m
    if best_match:
        return full_text[best_match.start():][:20000]
    return ""


def extract_dois_from_text(text: str) -> List[str]:
    """Regex DOI extraction as last resort."""
    pattern = r'\b10\.\d{4,}(?:\.\d+)*/\S+'
    dois = re.findall(pattern, text)
    # Clean trailing punctuation
    cleaned = []
    for doi in dois:
        doi = re.sub(r'[.,;)\]]+$', '', doi)
        if doi not in cleaned:
            cleaned.append(doi)
    return cleaned


# ---------------------------------------------------------------------------
# LLM extraction helpers
# ---------------------------------------------------------------------------

def _str(v: Any, default: str = "") -> str:
    if v is None or v == "null":
        return default
    return str(v)


def _list(v: Any, default: Optional[List] = None) -> List:
    if v is None or v == "null":
        return default if default is not None else []
    if isinstance(v, list):
        return v
    return [v]


METADATA_PROMPT = """\
You are reviewing a systematic review paper. Extract the following metadata.

Return JSON only:
{{
  "title": "...",
  "review_type": "scoping | rapid | slr",
  "research_questions": ["..."],
  "population": "...",
  "concept": "...",
  "context": "...",
  "intervention": null,
  "comparator": null,
  "outcome": null,
  "inclusion_criteria": ["..."],
  "exclusion_criteria": ["..."],
  "databases": ["PubMed", "Scopus", ...],
  "search_query": "...",
  "date_range": "2010-2023",
  "language_filter": ["English"],
  "n_identified": 974,
  "n_included": 89,
  "original_query": "..."
}}

Paper text (truncated):
{text}
"""

INCLUDED_PAPERS_PROMPT = """\
You are extracting the list of papers that were INCLUDED in this systematic review.

IMPORTANT:
- Extract ONLY papers that the review authors selected and included in their analysis
- Do NOT extract general references or citations from the review text
- Look for a numbered appendix, "Included studies" section, or reference list labelled as included papers
- If there is no explicit list of included papers, return an empty array

Return JSON only:
{{
  "papers": [
    {{
      "doi": "10.xxxx/...",
      "title": "Full paper title",
      "authors": "Lastname1 et al.",
      "year": 2020,
      "extraction_note": "from appendix or reference list"
    }}
  ],
  "extraction_note": "brief note on where the list was found or why it is empty"
}}

Text to analyse:
{text}
"""


def extract_methods_metadata(text: str, client: OpenAI, model: str) -> Dict[str, Any]:
    """Extract review metadata via LLM."""
    prompt = METADATA_PROMPT.format(text=text[:8000])
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
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  [WARN] Metadata extraction failed: {e}")
        return {}


def extract_included_papers_llm(text: str, client: OpenAI, model: str) -> List[Dict]:
    """Extract included papers list via LLM."""
    prompt = INCLUDED_PAPERS_PROMPT.format(text=text[:10000])
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
        raw = json.loads(resp.choices[0].message.content)
        papers = raw.get("papers", [])
        note = raw.get("extraction_note", "")
        if note:
            print(f"  [INFO] LLM note: {note}")
        return papers if isinstance(papers, list) else []
    except Exception as e:
        print(f"  [WARN] Included papers extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Supplementary file loader
# ---------------------------------------------------------------------------

def load_supplementary_csv(path: Path) -> List[Dict]:
    """Load included papers from CSV/XLSX."""
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        papers = []
        for _, row in df.iterrows():
            doi_raw = row.get("doi")
            doi = None if (doi_raw is None or pd.isna(doi_raw) or str(doi_raw).strip() in ("", "nan", "None")) else str(doi_raw).strip()
            papers.append({
                "doi": doi,
                "title": str(row.get("title", "") or "").strip(),
                "authors": str(row.get("authors", "") or "").strip(),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            })
        return papers
    except Exception as e:
        print(f"  [WARN] Failed to load {path.name}: {e}")
        return []


# ---------------------------------------------------------------------------
# GoldStandard builder
# ---------------------------------------------------------------------------

def build_gold_standard(
    review_dir: Path,
    metadata: Dict[str, Any],
    included_papers: List[Dict],
) -> Dict[str, Any]:
    """
    Build the gold_standard.json dict using the GoldStandard Pydantic schema
    (slr_pipeline/evaluation/gold_standard.py).
    """
    included_dois = [
        p["doi"] for p in included_papers
        if p.get("doi") and str(p["doi"]).strip() not in ("", "null", "None", "nan")
    ]
    included_titles = [
        p["title"] for p in included_papers
        if p.get("title") and str(p["title"]).strip()
    ]
    doi_coverage = len(included_dois) / len(included_papers) if included_papers else 0.0
    return {
        # Required by GoldStandard model
        "review_id": review_dir.name,
        "review_citation": _str(metadata.get("title"), "Unknown review"),
        "review_type": _str(metadata.get("review_type"), "scoping"),

        # Search strategy
        "original_query": _str(metadata.get("search_query") or metadata.get("original_query")),
        "original_databases": _list(metadata.get("databases")),
        "original_date_from": None,
        "original_date_to": None,
        "original_language_filter": _list(metadata.get("language_filter")),

        # Protocol
        "research_questions": _list(metadata.get("research_questions")),
        "inclusion_criteria": _list(metadata.get("inclusion_criteria")),
        "exclusion_criteria": _list(metadata.get("exclusion_criteria")),
        "pcc_population": _str(metadata.get("population")),
        "pcc_concept": _str(metadata.get("concept")),
        "pcc_context": _str(metadata.get("context")),

        # PRISMA counts
        "prisma_identified": metadata.get("n_identified"),
        "prisma_after_dedup": None,
        "prisma_screened": None,
        "prisma_assessed_fulltext": None,
        "prisma_included": metadata.get("n_included") or len(included_papers),

        # Gold standard papers
        "included_dois": included_dois,
        "included_titles": included_titles,
        "doi_coverage": round(doi_coverage, 2),
        "notes": [],

        # Extra (not in GoldStandard model but useful for debugging)
        "included_papers": included_papers,
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_gold_standard(review_dir_path: str, model: str = "gpt-5.4-mini") -> Dict:
    review_dir = Path(review_dir_path)
    review_pdf = review_dir / "review.pdf"

    if not review_pdf.exists():
        raise FileNotFoundError(f"review.pdf not found in {review_dir}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("  Parsing PDF...")
    pages = parse_pdf_pages(review_pdf)
    print(f"  -> {len(pages)} pages")

    full_text = pages_to_text(pages, max_chars=40000)
    ref_text = find_reference_section(pages)

    # --- Extract methods metadata ---
    print("  Extracting methods metadata...")
    metadata = extract_methods_metadata(full_text, client, model)

    # --- Extract included papers ---
    print("  Extracting included paper list...")
    included_papers: List[Dict] = []

    # Priority 1: user-provided supplementary files (manually curated)
    for supp_name in ("supplementary.xlsx", "supplementary.xls", "supplementary.csv",
                      "included_papers.csv"):
        supp_path = review_dir / supp_name
        if supp_path.exists():
            print(f"  -> Loading from {supp_name} (user-provided)")
            included_papers = load_supplementary_csv(supp_path)
            print(f"  -> {len(included_papers)} papers from supplementary file")
            break

    # Priority 2: supplementary PDF
    if not included_papers:
        supp_pdf = review_dir / "supplementary.pdf"
        if supp_pdf.exists():
            print("  -> Parsing supplementary PDF...")
            supp_pages = parse_pdf_pages(supp_pdf)
            supp_text = pages_to_text(supp_pages)
            included_papers = extract_included_papers_llm(supp_text, client, model)
            print(f"  -> {len(included_papers)} papers from supplementary PDF")

    # Priority 3: appendix / reference section of main PDF
    if not included_papers and ref_text:
        print("  -> Attempting extraction from reference/appendix section...")
        included_papers = extract_included_papers_llm(ref_text, client, model)
        print(f"  -> {len(included_papers)} papers from reference section")

    # Priority 4: DOI regex from full text (last resort)
    if not included_papers:
        print("  -> Fallback: DOI regex extraction (may include non-included papers)")
        dois = extract_dois_from_text(ref_text or full_text)
        included_papers = [{"doi": d, "title": "", "authors": "", "year": None}
                           for d in dois]
        print(f"  -> {len(included_papers)} DOIs extracted by regex (review manually!)")

    # --- Build and save GoldStandard ---
    gs = build_gold_standard(review_dir, metadata, included_papers)

    gs_path = review_dir / "gold_standard.json"
    with open(gs_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved gold_standard.json ({len(included_papers)} included papers)")

    # --- Save extracted papers as CSV for review ---
    if included_papers:
        csv_path = review_dir / "extracted_papers.csv"
        pd.DataFrame(included_papers).to_csv(csv_path, index=False)
        print(f"  [OK] Saved extracted_papers.csv (rename to included_papers.csv after review)")

    return gs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gold standard from a review PDF")
    parser.add_argument("review_dir", help="Path to review folder containing review.pdf")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model to use")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Extracting gold standard from: {Path(args.review_dir).name}")
    print("=" * 60)

    gs = extract_gold_standard(args.review_dir, model=args.model)

    print()
    print("Summary:")
    print(f"  Title:       {gs['review_citation']}")
    print(f"  Review type: {gs['review_type']}")
    print(f"  Databases:   {', '.join(gs['original_databases'])}")
    print(f"  n_identified:{gs['prisma_identified']}")
    print(f"  n_included:  {gs['prisma_included']}")
    print(f"  DOIs found:  {len(gs['included_dois'])}")
    print(f"  Titles found:{len(gs['included_titles'])}")
