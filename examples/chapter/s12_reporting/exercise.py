"""S12 pratimas -- ataskaitų teikimas: PRISMA srautas ir DI atsekamumas.

Užduotis
--------
Duota galutinė tiriamosios (scoping) apžvalgos būsena
(``examples/chapter/data/pipeline_state_fixture.json``) ir DI naudojimo
atsekamumo ataskaita (``ai_traceability_fixture.json``).

1. Įgyvendinkite ``prisma_flow(state)`` -- sudarykite PRISMA srauto skaičius
   ir patikrinkite penkias tvermės lygčių grupes bei tiriamojo režimo
   tinkamumo patikrą (kokybės vertinimas praleistas). Jei būsena nėra
   galutinė (netuščia pilno teksto peržiūros eilė) -- iškelkite klaidą.
2. Įgyvendinkite ``classify_stage_oversight(...)`` -- kiekvienam etapui
   grąžinkite tris nepriklausomus matmenis: ``execution`` (ar vyko),
   ``automation_origin`` (kas automatizavo) ir ``human_reviewed`` (ar
   peržiūrėjo žmogus).
3. Paaiškinkite žodyno skirtumą tarp „vienas etapas automatizuotas" ir
   „visiška (viso pipeline'o) autonomija".

Paleidimas: ``python -m examples.chapter.s12_reporting.exercise``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PrismaInconsistencyError(ValueError):
    pass


class StageOversight(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution: Literal["run", "not_run", "unknown_or_not_instrumented"]
    automation_origin: Literal["ai", "rule_based_automation"] | None
    human_reviewed: bool


class PrismaFlow(BaseModel):
    total_identified: int
    duplicates_removed: int
    after_dedup: int
    screened_title_abstract: int
    excluded_title_abstract: int
    included_after_ta: int
    fulltext_sought: int
    assessed_full_text: int
    acquisition_failures: int
    excluded_full_text: int
    excluded_full_text_reasons: dict[str, int]
    included: int

    def to_ascii(self) -> str:
        raise NotImplementedError


def prisma_flow(state: dict) -> PrismaFlow:
    """TODO: sudarykite srautą ir patikrinkite tvermės grupes."""
    raise NotImplementedError


def classify_stage_oversight(traceability: dict, prisma_counts: dict, oversight_mode: dict) -> dict:
    """TODO: trys nepriklausomi priežiūros matmenys kiekvienam etapui."""
    raise NotImplementedError


def main() -> None:
    from examples.chapter import toolkit

    state = toolkit.load_fixture("pipeline_state_fixture.json")
    print("S12 reporting exercise")
    print(f"review type: {state['review_type']}; FT review queue size: {len(state['full_text_review_queue'])}")
    try:
        flow = prisma_flow(state)
    except NotImplementedError:
        print("prisma_flow() is not implemented yet -- fill it in and re-run.")
        return
    print(flow.to_ascii())


if __name__ == "__main__":
    main()
