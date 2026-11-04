"""S8 sprendimas -- pilno teksto gavimas ir pilno teksto tinkamumo vertinimas.

Mokomoji versija. Realus etapas -- ``slr_pipeline/nodes/fulltext.py``:
  1 sluoksnis -- atviros prieigos (OA) gavimo kaskada + „stub" (peržiūros)
                 aptikimas;
  2 sluoksnis -- tinkamumo vertinimas; „include" TIK esant aiškiam
                 sprendimui, viskas kita -> peržiūros eilė (niekada
                 „įtraukti dėl klaidos").

Svarbu: ``screen_fulltext`` NIEKADA nekviečiamas įrašui, kuriam nepavyko
gauti pilno teksto (``no_oa_fulltext_found`` / ``no_doi``).
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from examples.chapter import toolkit

_SYSTEM = (
    "Decide full-text eligibility: 'include', 'exclude', or 'needs_review'. "
    'Return JSON {"decision", "per_criterion": {<label>: "met"|"not_met"|"unclear"}, "rationale"}.'
)

_STUB_MARKERS = ("preview only", "this is a preview", "purchase to read", "access denied")
_MIN_FULLTEXT_CHARS = 400
_VALID_DECISIONS = ("include", "exclude", "needs_review")
_CRITERION_LABELS = ("names_a_review_step", "automation_is_the_subject", "method_detail_sufficient")
_CRITERION_VALUES = ("met", "not_met", "unclear")


class Discovery(BaseModel):
    status: Literal["found", "no_oa_fulltext_found", "no_doi"]
    source: str | None = None
    pdf_url: str | None = None


class FullTextDecision(BaseModel):
    decision: Literal["include", "exclude", "needs_review"]
    per_criterion: dict[str, str]
    rationale: str


def _pdf_url_from(resolver_name: str, payload: dict) -> str | None:
    """Kiekvienas OA teikėjas PDF nuorodą pateikia savo vietoje: Unpaywall --
    ``best_oa_location.url_for_pdf``, Semantic Scholar -- ``openAccessPdf.url``.
    """
    if resolver_name == "unpaywall":
        location = payload.get("best_oa_location") or {}
        return location.get("url_for_pdf")
    if resolver_name == "semantic_scholar":
        oa_pdf = payload.get("openAccessPdf") or {}
        return oa_pdf.get("url")
    return None


# --- listing:s08-discover start ---
def discover(doi: str | None, *, resolvers=toolkit.mock_oa_resolvers) -> Discovery:
    """Bando gauti nemokamą pilną tekstą einant per resolverius iš eilės.

    ``discover(None)`` -> ``no_doi``. Jei nė vienas resolveris negrąžina PDF
    nuorodos (net jei atsakė, bet be nuorodos) -> ``no_oa_fulltext_found``.
    """
    if not doi:
        return Discovery(status="no_doi")
    for resolver in resolvers:
        payload = resolver(doi)
        if not payload:
            continue
        pdf_url = _pdf_url_from(resolver.name, payload)
        if pdf_url:
            return Discovery(status="found", source=resolver.name, pdf_url=pdf_url)
    return Discovery(status="no_oa_fulltext_found")
# --- listing:s08-discover end ---


# --- listing:s08-download start ---
def download(pdf_url: str, *, downloader=toolkit.fake_downloader) -> str:
    """Atsisiunčia PDF turinį kaip tekstą (atskiras žingsnis nuo
    ``discover``)."""
    return downloader(pdf_url)
# --- listing:s08-download end ---


# --- listing:s08-looks-like-stub start ---
def looks_like_stub(text: str) -> bool:
    """Ar tekstas atrodo kaip peržiūros fragmentas, ne pilnas straipsnis --
    per trumpas arba su būdingomis „preview" frazėmis."""
    if len(text.strip()) < _MIN_FULLTEXT_CHARS:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _STUB_MARKERS)
# --- listing:s08-looks-like-stub end ---


# --- listing:s08-screen-fulltext start ---
def screen_fulltext(text: str, criteria: toolkit.Criteria, *, call_llm=toolkit.call_llm) -> FullTextDecision:
    """Vertina pilną tekstą kriterijus po kriterijaus. Grąžina „include",
    „exclude" arba „needs_review". Iškelia ``ValueError`` neteisingam
    atsakymui -- niekada nenutyli.
    """
    if len(_CRITERION_LABELS) != len(criteria.inclusion_criteria):
        raise ValueError("criterion label count does not match the protocol's inclusion criteria")
    inclusion = "\n".join(
        f"{label}: {crit}" for label, crit in zip(_CRITERION_LABELS, criteria.inclusion_criteria)
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"INCLUSION CRITERIA (answer per_criterion under EXACTLY these labels)\n{inclusion}\n\n"
                f"FULL TEXT\n{text}"
            ),
        },
    ]
    try:
        payload = json.loads(call_llm(messages, response_format={"type": "json_object"}, temperature=0.0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {exc}") from exc
    if payload.get("decision") not in _VALID_DECISIONS:
        raise ValueError(f"invalid eligibility decision: {payload.get('decision')!r}")
    per_criterion = payload.get("per_criterion", {})
    if set(per_criterion) != set(_CRITERION_LABELS):
        raise ValueError(f"per_criterion must cover exactly {_CRITERION_LABELS}, got {sorted(per_criterion)}")
    bad_values = {v for v in per_criterion.values() if v not in _CRITERION_VALUES}
    if bad_values:
        raise ValueError(f"per_criterion values must be one of {_CRITERION_VALUES}, got {bad_values}")
    return FullTextDecision(
        decision=payload["decision"],
        per_criterion=per_criterion,
        rationale=payload.get("rationale", ""),
    )
# --- listing:s08-screen-fulltext end ---


def retrieve_and_screen(doi: str | None, criteria: toolkit.Criteria, *, call_llm=toolkit.call_llm):
    """Sujungia abu sluoksnius: TIK gavus pilną tekstą kviečiamas vertinimas."""
    found = discover(doi)
    if found.status != "found":
        return found, None
    text = download(found.pdf_url)
    if looks_like_stub(text):
        return Discovery(status="no_oa_fulltext_found"), None
    return found, screen_fulltext(text, criteria, call_llm=call_llm)


def main() -> None:
    key = toolkit.load_fixture("papers/answer_key.json")
    scripted_map = toolkit.SCRIPTED_RESPONSES["s08"]["papers"]

    def key_fn(messages):
        content = messages[-1]["content"]
        for stem in scripted_map:
            paper = toolkit.load_fixture(f"papers/{stem}.md")
            if paper[:200] in content:
                return stem
        return "?"

    scripted = toolkit.scripted_llm(scripted_map, key_fn=key_fn)
    print("S8 full-text retrieval + eligibility (scripted demo)")
    for stem, entry in key.items():
        found, decision = retrieve_and_screen(entry["doi"], toolkit.DEMO_CRITERIA, call_llm=scripted)
        verdict = decision.decision if decision else found.status
        print(f"  {stem}: discovery={found.status}, eligibility={verdict}")


if __name__ == "__main__":
    main()
