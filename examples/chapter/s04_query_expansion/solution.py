"""S4 sprendimas -- paieškos užklausos plėtimas su DKM (LLM) pagalba.

Mokomoji versija. Realus etapas --
``slr_pipeline/nodes/search_strategy_builder.py``.

Idėja: DKM pasiūlo koncepcijas ir jų sinonimus, o Boolean užklausą surenkame
patys deterministiškai. Vertimas į konkrečios duomenų bazės sintaksę --
grynas (be DKM) žingsnis, nes tai taisyklės, ne interpretacija. Bet kuri DKM
pasiūlyta užklausa vis tiek reikalauja nepriklausomo žmogaus įvertinimo.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr

from examples.chapter import toolkit

_SYSTEM = (
    "You expand a literature-search seed query. Return only JSON: "
    '{"concepts": [{"label": str, "terms": [str, ...]}, ...], "boolean_query": str}. '
    "Give at least three concepts, each with several synonyms."
)


class Concept(BaseModel):
    model_config = ConfigDict(strict=True)

    label: StrictStr
    terms: list[StrictStr]


class QueryExpansion(BaseModel):
    model_config = ConfigDict(strict=True)

    concepts: list[Concept]
    boolean_query: StrictStr


# --- listing:s04-expand start ---
def expand_query(seed: str, *, call_llm=toolkit.call_llm) -> QueryExpansion:
    """Paprašo DKM išplėsti ``seed`` užklausą ir grąžina patikrintą
    ``QueryExpansion``.

    Iškelia ``ValueError``, jei atsakymas nėra tinkamas JSON arba jei
    negrąžinta nė viena koncepcija.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": seed},
    ]
    raw = call_llm(messages, response_format={"type": "json_object"}, temperature=0.0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {exc}") from exc

    expansion = QueryExpansion.model_validate(payload)
    if not expansion.concepts:
        raise ValueError("model returned zero concepts")
    return expansion
# --- listing:s04-expand end ---


def _render_term(term: str, tag: str | None) -> str:
    quoted = f'"{term}"' if " " in term else term
    return f"{quoted}{tag}" if tag else quoted


def _concept_group(terms: list[str], *, tag: str | None) -> str:
    return "(" + " OR ".join(_render_term(t, tag) for t in terms) + ")"


# --- listing:s04-translate start ---
def translate_for_database(
    expansion: QueryExpansion, database: Literal["pubmed", "openalex", "scopus"]
) -> str:
    """Deterministiškai išverčia ``QueryExpansion`` į vienos duomenų bazės
    Boolean sintaksę. DKM čia nenaudojamas.

    - ``pubmed``: kiekvienas terminas gauna ``[tiab]`` lauko žymą.
    - ``openalex``: paprasta Boolean išraiška be lauko žymų.
    - ``scopus``: visa išraiška apgaubiama ``TITLE-ABS-KEY(...)``.
    """
    if database == "pubmed":
        groups = [_concept_group(c.terms, tag="[tiab]") for c in expansion.concepts]
        return " AND ".join(groups)
    if database == "openalex":
        groups = [_concept_group(c.terms, tag=None) for c in expansion.concepts]
        return " AND ".join(groups)
    if database == "scopus":
        groups = [_concept_group(c.terms, tag=None) for c in expansion.concepts]
        return "TITLE-ABS-KEY(" + " AND ".join(groups) + ")"
    raise ValueError(f"unknown database: {database}")
# --- listing:s04-translate end ---


def main() -> None:
    seed = "large language models for systematic review screening"
    scripted = toolkit.scripted_llm(
        {seed: toolkit.SCRIPTED_RESPONSES["s04"][seed]},
        key_fn=lambda messages: messages[-1]["content"],
    )
    expansion = expand_query(seed, call_llm=scripted)
    print("S4 query expansion (scripted demo)")
    print(f"concepts: {[c.label for c in expansion.concepts]}")
    for db in ("pubmed", "openalex", "scopus"):
        print(f"\n[{db}]\n{translate_for_database(expansion, db)}")


if __name__ == "__main__":
    main()
