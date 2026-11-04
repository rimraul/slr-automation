"""S5 pratimas -- daugiabazė paieška.

Užduotis
--------
Kiekviena duomenų bazė grąžina savo formos atsakymą:
  * ``raw_responses/pubmed.xml``    -- PubMed EFetch XML;
  * ``raw_responses/openalex.json`` -- OpenAlex (santrauka „inverted index",
    DOI kaip URL);
  * ``raw_responses/crossref.json`` -- Crossref (pavadinimas -- sąrašas,
    santrauka su JATS žymomis);
  * ``raw_responses/scopus.json``   -- Scopus (raktai su vardų sritimis).

1. Įgyvendinkite ``normalise(name, raw_body)`` OpenAlex, PubMed ir Crossref
   šaltiniams. Kiekvienas įrašas privalo turėti TIKSLIAI raktus
   ``EXPECTED_KEYS``; ``abstract`` niekada nėra ``None`` (naudokite ``""``);
   ``doi`` -- normalizuotas (be URL prefikso) arba ``None``.
2. Įgyvendinkite ``run_search(adapters, query)`` -- sujunkite visų adapterių
   įrašus (dublikatų NEŠALINKITE).
3. SAVO užduotis: pridėkite ketvirtąjį adapterį (Scopus) savo ``exercise.py``.
4. Pastebėkite: tas pats darbas gali ateiti iš PubMed be DOI ir iš Crossref
   su DOI -- tai lieka du įrašai, kuriuos suderins S6.

Paleidimas: ``python -m examples.chapter.s05_multi_db_search.exercise``
"""

from __future__ import annotations

from pathlib import Path

EXPECTED_KEYS = {"id", "source", "source_id", "doi", "title", "abstract", "year", "authors"}

_DATA = Path(__file__).resolve().parents[1] / "data" / "raw_responses"


class FixtureAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        suffix = "xml" if name == "pubmed" else "json"
        self._path = _DATA / f"{name}.{suffix}"

    def fetch(self, query: str) -> str:  # noqa: ARG002
        return self._path.read_text(encoding="utf-8").replace("\r\n", "\n")


def normalise(name: str, raw_body: str) -> list[dict]:
    """TODO: OpenAlex + PubMed + Crossref (Scopus -- jūsų užduotis)."""
    raise NotImplementedError


def run_search(adapters, query: str) -> list[dict]:
    """TODO: sujunkite visų adapterių normalizuotus įrašus."""
    raise NotImplementedError


def main() -> None:
    adapters = [FixtureAdapter(n) for n in ("openalex", "pubmed", "crossref")]
    print("S5 multi-database search exercise")
    try:
        records = run_search(adapters, "review automation")
    except NotImplementedError:
        print("normalise()/run_search() not implemented yet -- fill them in and re-run.")
        return
    print(f"merged {len(records)} records from {len(adapters)} sources")


if __name__ == "__main__":
    main()
