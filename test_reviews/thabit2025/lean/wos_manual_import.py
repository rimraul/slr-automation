"""
Lean prototype: parse a WoS GUI manual export (Thabit et al. 2025 reproduction)
into Record-shaped dicts, ready to append into raw_records.json before dedup_node.

Reused directly from il_outcomes/run_adapters/merge_wos_abstracts.py and
auto-mare/lit_review/merge_wos_abstracts_v2.py: column-alias resolution
(full-name export headers vs. short field-tag headers), UT as the exact
match/id key. No live API call -- WoS is manually exported via the web UI
and dropped into test_reviews/thabit2025/lean/raw_exports/.

Deliberately NOT wired into slr_pipeline/search_backends/registry.py (WoS
must never be a registered live adapter there -- see registry.py's own
comment). This script is out-of-band, appends into raw_records.json
directly.

Prototype only -- no pydantic Record validation, no dedup (dedup_node
handles cross-source dedup downstream via DOI/title matching, see
slr_pipeline/nodes/dedup.py::_dedup_doi_only). Cut corners here freely;
harden only the pieces that turn out to matter.

Usage:
    PYEXE="python"
    $PYEXE wos_manual_import.py <export1.xls> [<export2.txt> ...] --out wos_raw.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Column aliases: canonical field -> candidate header names, full-name export first.
ALIASES = {
    "ut":       ["UT (Unique WOS ID)", "UT"],
    "title":    ["Article Title", "TI"],
    "abstract": ["Abstract", "AB"],
    "authors":  ["Authors", "AU"],
    "year":     ["Publication Year", "PY"],
    "journal":  ["Source Title", "SO"],
    "kw_author": ["Author Keywords", "DE"],
    "kw_plus":   ["Keywords Plus", "ID"],
    "doi":      ["DOI", "DI"],
    "cited_by": ["Times Cited, WoS Core", "Times Cited, All Databases", "Z9", "TC"],
    "oa":       ["Open Access Designations", "OA"],
}


def _resolve(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _clean(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s or None


def _record_id(source: str, native_key: str) -> str:
    digest = hashlib.sha256(native_key.encode("utf-8")).hexdigest()[:20]
    return f"{source}-{digest}"


def _load_export(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def parse_export(path: Path) -> List[Dict[str, Any]]:
    df = _load_export(path)
    cols = {k: _resolve(df, v) for k, v in ALIASES.items()}
    if cols["ut"] is None:
        raise ValueError(f"{path.name}: no UT/UT (Unique WOS ID) column found -- cannot key records")

    records = []
    for _, row in df.iterrows():
        ut = _clean(row.get(cols["ut"]))
        if not ut:
            continue

        authors_raw = _clean(row.get(cols["authors"])) if cols["authors"] else None
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()] if authors_raw else []

        kw = []
        for kw_key in ("kw_author", "kw_plus"):
            raw = _clean(row.get(cols[kw_key])) if cols[kw_key] else None
            if raw:
                kw.extend(k.strip() for k in raw.split(";") if k.strip())

        year_raw = _clean(row.get(cols["year"])) if cols["year"] else None
        year = None
        if year_raw:
            try:
                year = int(float(year_raw))
            except ValueError:
                year = None

        cited_raw = _clean(row.get(cols["cited_by"])) if cols["cited_by"] else None
        cited_by = None
        if cited_raw:
            try:
                cited_by = int(float(cited_raw))
            except ValueError:
                cited_by = None

        doi = _clean(row.get(cols["doi"])) if cols["doi"] else None
        doi = doi.lower() if doi else None

        oa_designation = _clean(row.get(cols["oa"])) if cols["oa"] else None
        # "Green Submitted"/"Green Accepted"/"Green Published" alone only mean
        # an author manuscript is deposited in a repository somewhere else --
        # the publisher's own page (what a DOI resolves to) stays paywalled.
        # Only gold/hybrid/bronze mean the publisher's page itself is free.
        # A bare designation lumping these together (bool(oa_designation))
        # falsely counted green-only papers as OA -- confirmed by a Tier 2
        # pilot run where 8/20 "OA" papers were paywalled at their DOI, all
        # 8 green-only.
        oa_lower = (oa_designation or "").lower()
        has_real_oa_route = any(tok in oa_lower for tok in ("gold", "hybrid", "bronze"))
        is_green_only = bool(oa_designation) and not has_real_oa_route

        rec = {
            "id": _record_id("wos", ut),
            "source": "wos",
            "source_id": ut,
            "doi": doi,
            "title": _clean(row.get(cols["title"])) if cols["title"] else "",
            "abstract": _clean(row.get(cols["abstract"])) if cols["abstract"] else None,
            "authors": authors,
            "year": year,
            "journal": _clean(row.get(cols["journal"])) if cols["journal"] else None,
            "keywords": kw,
            "cited_by": cited_by,
            "url": f"https://doi.org/{doi}" if doi else None,
            "is_open_access": has_real_oa_route if oa_designation is not None else None,
            "is_green_only": is_green_only,
            "open_access_designation": oa_designation,
        }
        rec["title"] = rec["title"] or ""
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", help="WoS GUI export file(s) (.xls/.xlsx/.txt)")
    ap.add_argument("--out", default="wos_raw.json", help="output JSON path")
    ap.add_argument("--merge-into", default=None, help="existing raw_records.json to append into")
    args = ap.parse_args()

    by_ut: Dict[str, Dict[str, Any]] = {}
    for export_str in args.exports:
        path = Path(export_str)
        recs = parse_export(path)
        print(f"{path.name}: parsed {len(recs)} records")
        for r in recs:
            by_ut[r["source_id"]] = r  # last file wins on UT collision across exports

    wos_records = list(by_ut.values())
    n_with_abstract = sum(1 for r in wos_records if r["abstract"])
    print(f"Total unique WoS records (by UT): {len(wos_records)} ({n_with_abstract} with abstract)")

    out_path = Path(args.out)
    if args.merge_into:
        existing = json.loads(Path(args.merge_into).read_text(encoding="utf-8"))
        combined = existing + wos_records
        out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Merged {len(wos_records)} WoS + {len(existing)} existing -> {out_path} "
              f"({len(combined)} total, cross-source dedup happens in dedup_node, not here)")
    else:
        out_path.write_text(json.dumps(wos_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(wos_records)} records -> {out_path}")


if __name__ == "__main__":
    main()
