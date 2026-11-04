"""S8 pratimas -- pilno teksto gavimas ir tinkamumo vertinimas.

Užduotis
--------
(a) Įgyvendinkite ``discover(doi, *, resolvers)`` -- OA gavimo kaskadą
    (``discover(None)`` -> ``no_doi``; nė vienas resolveris nerado ->
    ``no_oa_fulltext_found``); ``download`` -- atskirą atsisiuntimo žingsnį;
    ``looks_like_stub`` -- peržiūros fragmento aptikimą.
(b) Įgyvendinkite ``screen_fulltext(text, criteria, *, call_llm)`` --
    „include" / „exclude" / „needs_review", kriterijus po kriterijaus.

Klasifikuokite kiekvieno DOI *gavimo* baigtį (4 straipsniai +
``oa_retrieval_records.json`` DOI), atsisiųskite gaunamus, ir kiekvienam iš 4
straipsnių pateikite tinkamumo sprendimą. Paaiškinkite, kodėl
``editorial.md`` yra **exclude**, o ``ambiguous.md`` -- **needs_review**.

Svarbu: ``screen_fulltext`` NEKVIEČIAMAS, jei pilno teksto gauti nepavyko.

Paleidimas: ``python -m examples.chapter.s08_fulltext.exercise``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from examples.chapter import toolkit


class Discovery(BaseModel):
    status: Literal["found", "no_oa_fulltext_found", "no_doi"]
    source: str | None = None
    pdf_url: str | None = None


class FullTextDecision(BaseModel):
    decision: Literal["include", "exclude", "needs_review"]
    per_criterion: dict[str, str]
    rationale: str


def discover(doi: str | None, *, resolvers=toolkit.mock_oa_resolvers) -> Discovery:
    """TODO: OA gavimo kaskada."""
    raise NotImplementedError


def download(pdf_url: str, *, downloader=toolkit.fake_downloader) -> str:
    """TODO: atsisiuntimo žingsnis."""
    raise NotImplementedError


def looks_like_stub(text: str) -> bool:
    """TODO: peržiūros fragmento aptikimas."""
    raise NotImplementedError


def screen_fulltext(text: str, criteria, *, call_llm=toolkit.call_llm) -> FullTextDecision:
    """TODO: tinkamumo vertinimas kriterijus po kriterijaus."""
    raise NotImplementedError


def main() -> None:
    key = toolkit.load_fixture("papers/answer_key.json")
    print("S8 full-text exercise")
    try:
        for stem, entry in key.items():
            print(f"  {stem}: discovery={discover(entry['doi']).status}")
    except NotImplementedError:
        print("discover() is not implemented yet -- fill it in and re-run.")


if __name__ == "__main__":
    main()
