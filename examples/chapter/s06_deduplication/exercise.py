"""S6 pratimas -- deduplikacija.

Užduotis
--------
Faile ``examples/chapter/data/bib_records.json`` yra 18 bibliografinių įrašų,
surinktų iš kelių duomenų bazių. Kai kurie iš jų yra to paties darbo
dublikatai.

1. Įgyvendinkite ``normalise_doi`` -- normalizuokite DOI (mažosios raidės,
   pašalintas ``https://doi.org/`` arba ``http://dx.doi.org/`` prefiksas).
   Naudokite ``str.removeprefix``, NE ``str.lstrip``.
2. Įgyvendinkite ``dedupe`` -- dviejų pakopų kaskadą:
   pirma pagal normalizuotą DOI, paskui pagal neaiškų (fuzzy) pavadinimų
   sutapimą (``difflib.SequenceMatcher``).
3. Deduplikuokite ``bib_records.json``. Turėtų likti 15 unikalių įrašų
   (pašalinami 3 dublikatai: 2 pagal DOI, 1 pagal pavadinimą).
4. Trumpai atsakykite: kurią porą praleistų vien DOI paremtas metodas, o
   kurią -- naivus tikslaus pavadinimo sutapimo metodas?

Paleidimas: ``python -m examples.chapter.s06_deduplication.exercise``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DuplicatePair(BaseModel):
    dup_id: str
    canonical_id: str
    matched_on: Literal["doi", "title"]


class DedupeResult(BaseModel):
    unique: list[dict]
    duplicates: list[DuplicatePair]


def normalise_doi(doi: str | None) -> str | None:
    """TODO: normalizuokite DOI arba grąžinkite None."""
    raise NotImplementedError


def dedupe(records: list[dict], *, title_threshold: float = 0.90) -> DedupeResult:
    """TODO: dviejų pakopų deduplikacijos kaskada."""
    raise NotImplementedError


def main() -> None:
    from examples.chapter import toolkit

    records = toolkit.load_fixture("bib_records.json")
    print("S6 deduplication exercise")
    print(f"loaded {len(records)} records; target unique count after dedupe: 15")
    try:
        result = dedupe(records)
    except NotImplementedError:
        print("dedupe() is not implemented yet -- fill it in and re-run.")
        return
    print(f"your dedupe() kept {len(result.unique)} unique records")
    for pair in result.duplicates:
        print(f"  {pair.dup_id} -> {pair.canonical_id} (matched on {pair.matched_on})")


if __name__ == "__main__":
    main()
