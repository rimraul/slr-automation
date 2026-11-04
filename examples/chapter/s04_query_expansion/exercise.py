"""S4 pratimas -- paieškos užklausos plėtimas su DKM (LLM).

Užduotis
--------
1. Įgyvendinkite ``expand_query(seed, *, call_llm)`` -- nusiųskite pradinę
   užklausą DKM, gaukite JSON su koncepcijomis ir jų sinonimais, patikrinkite
   jį ``QueryExpansion`` modeliu. Iškelkite ``ValueError``, jei JSON netinkamas
   arba nėra nė vienos koncepcijos.
2. Įgyvendinkite ``translate_for_database(expansion, database)`` -- BE DKM,
   grynomis taisyklėmis išverskite išraišką į ``pubmed`` / ``openalex`` /
   ``scopus`` sintaksę.
3. Paimkite SAVO pradinę užklausą, išplėskite ją, išverskite į visas tris
   duomenų bazes ir palyginkite. Vienu sakiniu įvardykite reikšmingiausią
   skirtumą tarp bazių.
4. Prisiminkite: DKM pasiūlyta užklausa VISADA reikalauja nepriklausomo
   žmogaus įvertinimo (skyriaus teiginys C07).

Paleidimas: ``python -m examples.chapter.s04_query_expansion.exercise``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr

from examples.chapter import toolkit


class Concept(BaseModel):
    model_config = ConfigDict(strict=True)

    label: StrictStr
    terms: list[StrictStr]


class QueryExpansion(BaseModel):
    model_config = ConfigDict(strict=True)

    concepts: list[Concept]
    boolean_query: StrictStr


def expand_query(seed: str, *, call_llm=toolkit.call_llm) -> QueryExpansion:
    """TODO: paklauskite DKM ir patikrinkite atsakymą."""
    raise NotImplementedError


def translate_for_database(
    expansion: QueryExpansion, database: Literal["pubmed", "openalex", "scopus"]
) -> str:
    """TODO: grynos taisyklės, jokio DKM."""
    raise NotImplementedError


def main() -> None:
    seed = "large language models for systematic review screening"
    scripted = toolkit.scripted_llm(
        {seed: toolkit.SCRIPTED_RESPONSES["s04"][seed]},
        key_fn=lambda messages: messages[-1]["content"],
    )
    print("S4 query expansion exercise")
    try:
        expansion = expand_query(seed, call_llm=scripted)
    except NotImplementedError:
        print("expand_query() is not implemented yet -- fill it in and re-run.")
        return
    print(f"concepts: {[c.label for c in expansion.concepts]}")
    for db in ("pubmed", "openalex", "scopus"):
        try:
            print(f"[{db}] {translate_for_database(expansion, db)}")
        except NotImplementedError:
            print(f"[{db}] translate_for_database() not implemented yet")


if __name__ == "__main__":
    main()
