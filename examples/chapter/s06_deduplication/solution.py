"""S6 sprendimas -- deduplikacija dviejų pakopų kaskada.

Mokomoji, savarankiška versija. Realus etapas
(``slr_pipeline/nodes/dedup.py``) papildomai naudoja sakinių įterpinių
(embedding) panašumą; čia to nedarome -- apsiribojame standartine biblioteka
(``re`` + ``difflib``), o įterpinių kelias paminimas kaip gamybinis
patobulinimas.

Kaskada:
  1 pakopa -- tikslus sutapimas pagal normalizuotą DOI.
  2 pakopa -- neaiškus (fuzzy) pavadinimų sutapimas tiems įrašams, kurie
              praėjo 1 pakopą (dažniausiai -- įrašai be DOI).
"""

from __future__ import annotations

import difflib
import re
from typing import Literal

from pydantic import BaseModel


class DuplicatePair(BaseModel):
    """Vienas rastas dublikatas: ``dup_id`` yra pašalinamo įrašo id,
    ``canonical_id`` -- pasiliekamo (pirmojo pastebėto) įrašo id."""

    dup_id: str
    canonical_id: str
    matched_on: Literal["doi", "title"]


class DedupeResult(BaseModel):
    unique: list[dict]
    duplicates: list[DuplicatePair]


# --- listing:s06-normalise-doi start ---
def normalise_doi(doi: str | None) -> str | None:
    """Grąžina normalizuotą DOI (mažosiomis raidėmis, be URL prefikso) arba
    ``None``.

    Pastaba: realus kodas (``dedup.py:230``) čia naudoja
    ``s.lstrip("https://doi.org/")``. Tai klaida: ``str.lstrip`` argumentas
    yra simbolių AIBĖ, ne prefiksas. Su ``https://doi.org/...`` jis suveikia
    atsitiktinai (DOI prasideda "10.", o "1" nėra toje aibėje), bet kitą
    formą, pvz. ``http://dx.doi.org/...``, jis sugadintų -- liktų
    "x.doi.org/...". Teisingas įrankis -- ``str.removeprefix``.
    """
    if doi is None:
        return None
    text = doi.strip().lower()
    text = text.removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
    return text or None
# --- listing:s06-normalise-doi end ---


def _normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


# --- listing:s06-dedupe start ---
def dedupe(records: list[dict], *, title_threshold: float = 0.90) -> DedupeResult:
    """Pašalina dublikatus dviejų pakopų kaskada ir grąžina ``DedupeResult``.

    ``title_threshold`` -- mažiausias ``difflib`` panašumo santykis, nuo kurio
    du normalizuoti pavadinimai laikomi tuo pačiu darbu.
    """
    unique: list[dict] = []
    duplicates: list[DuplicatePair] = []
    seen_doi: dict[str, str] = {}

    for record in records:
        norm_doi = normalise_doi(record.get("doi"))
        if norm_doi is not None and norm_doi in seen_doi:
            duplicates.append(
                DuplicatePair(dup_id=record["id"], canonical_id=seen_doi[norm_doi], matched_on="doi")
            )
            continue

        norm_title = _normalise_title(record["title"])
        title_match = None
        for kept in unique:
            ratio = difflib.SequenceMatcher(None, norm_title, _normalise_title(kept["title"])).ratio()
            if ratio >= title_threshold:
                title_match = kept["id"]
                break
        if title_match is not None:
            duplicates.append(
                DuplicatePair(dup_id=record["id"], canonical_id=title_match, matched_on="title")
            )
            continue

        unique.append(record)
        if norm_doi is not None:
            seen_doi[norm_doi] = record["id"]

    return DedupeResult(unique=unique, duplicates=duplicates)
# --- listing:s06-dedupe end ---
