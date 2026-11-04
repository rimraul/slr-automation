"""S5 sprendimas -- daugiabazė paieška ir heterogeniškų atsakymų
normalizavimas į vieną įrašo schemą.

Mokomoji versija. Realus etapas -- ``slr_pipeline/nodes/search.py``.

Kiekviena duomenų bazė grąžina savo formos atsakymą (PubMed -- XML,
OpenAlex/Crossref/Scopus -- JSON, bet skirtingais raktais). ``fetch`` grąžina
NEAPDOROTĄ atsakymo tekstą; visas parsinimas -- viename sluoksnyje
(``normalise``).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Protocol

_DATA = Path(__file__).resolve().parents[1] / "data" / "raw_responses"

EXPECTED_KEYS = {"id", "source", "source_id", "doi", "title", "abstract", "year", "authors"}

_DOI_PREFIXES = ("https://doi.org/", "http://dx.doi.org/")


def _norm_doi(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    for prefix in _DOI_PREFIXES:
        text = text.removeprefix(prefix)
    return text or None


class Adapter(Protocol):
    name: str

    def fetch(self, query: str) -> str: ...


# --- listing:s05-fixture-adapter start ---
class FixtureAdapter:
    """Vietoj tikro tinklo -- nuskaito iškirptą (fixture) atsakymą iš
    ``data/raw_responses/<name>.{xml|json}`` ir grąžina jį kaip tekstą."""

    def __init__(self, name: str) -> None:
        self.name = name
        suffix = "xml" if name == "pubmed" else "json"
        self._path = _DATA / f"{name}.{suffix}"

    def fetch(self, query: str) -> str:  # noqa: ARG002 -- fixture'as užklausos nenaudoja
        return self._path.read_text(encoding="utf-8").replace("\r\n", "\n")
# --- listing:s05-fixture-adapter end ---


def _uninvert_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _normalise_openalex(body: str) -> list[dict]:
    out = []
    for item in json.loads(body).get("results", []):
        source_id = item["id"].removeprefix("https://openalex.org/")
        out.append({
            "id": f"openalex:{source_id}",
            "source": "openalex",
            "source_id": source_id,
            "doi": _norm_doi(item.get("doi")),
            "title": item.get("title") or "",
            "abstract": _uninvert_abstract(item.get("abstract_inverted_index")),
            "year": item.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in item.get("authorships", [])],
        })
    return out


def _normalise_pubmed(body: str) -> list[dict]:
    root = ET.fromstring(body)
    out = []
    for article in root.findall("PubmedArticle"):
        pmid = article.findtext("MedlineCitation/PMID") or ""
        title = article.findtext("MedlineCitation/Article/ArticleTitle") or ""
        abstract_parts = [
            (node.text or "")
            for node in article.findall("MedlineCitation/Article/Abstract/AbstractText")
        ]
        doi = None
        for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
        year_text = article.findtext("MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
        authors = []
        for author in article.findall("MedlineCitation/Article/AuthorList/Author"):
            fore = author.findtext("ForeName") or ""
            last = author.findtext("LastName") or ""
            authors.append(" ".join(part for part in (fore, last) if part))
        out.append({
            "id": f"pubmed:{pmid}",
            "source": "pubmed",
            "source_id": pmid,
            "doi": _norm_doi(doi),
            "title": title,
            "abstract": " ".join(part for part in abstract_parts if part),
            "year": int(year_text) if year_text else None,
            "authors": authors,
        })
    return out


def _strip_jats(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _normalise_crossref(body: str) -> list[dict]:
    out = []
    for item in json.loads(body).get("message", {}).get("items", []):
        titles = item.get("title") or [""]
        date_parts = item.get("published", {}).get("date-parts", [[None]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        authors = [
            " ".join(part for part in (a.get("given", ""), a.get("family", "")) if part)
            for a in item.get("author", [])
        ]
        doi = item.get("DOI")
        # Crossref's native id IS the DOI; when a record has none, fall back to
        # a deterministic title slug so `id == f"{source}:{source_id}"` still
        # holds (G1) and S6 can still reconcile it by title.
        source_id = doi if doi else f"title:{_slugify(titles[0])}"
        out.append({
            "id": f"crossref:{source_id}",
            "source": "crossref",
            "source_id": source_id,
            "doi": _norm_doi(doi),
            "title": titles[0],
            "abstract": _strip_jats(item.get("abstract", "")),
            "year": year,
            "authors": authors,
        })
    return out


def _normalise_scopus(body: str) -> list[dict]:
    out = []
    for entry in json.loads(body).get("search-results", {}).get("entry", []):
        eid = entry.get("eid") or entry.get("dc:identifier") or ""
        cover = entry.get("prism:coverDate", "")
        year = int(cover[:4]) if cover[:4].isdigit() else None
        creator = entry.get("dc:creator")
        out.append({
            "id": f"scopus:{eid}",
            "source": "scopus",
            "source_id": eid,
            "doi": _norm_doi(entry.get("prism:doi")),
            "title": entry.get("dc:title") or "",
            "abstract": entry.get("dc:description") or "",
            "year": year,
            "authors": [creator] if creator else [],
        })
    return out


_NORMALISERS = {
    "openalex": _normalise_openalex,
    "pubmed": _normalise_pubmed,
    "crossref": _normalise_crossref,
    "scopus": _normalise_scopus,
}


# --- listing:s05-normalise start ---
def normalise(name: str, raw_body: str) -> list[dict]:
    """Paverčia vienos duomenų bazės neapdorotą atsakymo tekstą į standartinių
    įrašų sąrašą. Kiekvienas įrašas turi TIKSLIAI raktus ``EXPECTED_KEYS``.

    ``pubmed`` -- XML (``xml.etree.ElementTree``); kiti -- JSON.
    """
    try:
        normaliser = _NORMALISERS[name]
    except KeyError:
        raise ValueError(f"unknown source: {name}") from None
    return normaliser(raw_body)
# --- listing:s05-normalise end ---


# --- listing:s05-run-search start ---
def run_search(adapters: list[Adapter], query: str) -> list[dict]:
    """Kviečia kiekvieną adapterį, normalizuoja jo atsakymą ir sujungia visus
    įrašus į vieną sąrašą. Dublikatai NEŠALINAMI -- tai S6 užduotis."""
    merged: list[dict] = []
    for adapter in adapters:
        merged.extend(normalise(adapter.name, adapter.fetch(query)))
    return merged
# --- listing:s05-run-search end ---


def default_adapters() -> list[Adapter]:
    return [FixtureAdapter(n) for n in ("openalex", "pubmed", "crossref", "scopus")]


def main() -> None:
    records = run_search(default_adapters(), "review automation")
    print("S5 multi-database search (fixture demo)")
    print(f"merged {len(records)} records from 4 sources")
    by_source: dict[str, int] = {}
    for rec in records:
        by_source[rec["source"]] = by_source.get(rec["source"], 0) + 1
    print(f"per source: {by_source}")


if __name__ == "__main__":
    main()
