"""S7 pratimas -- pavadinimo/santraukos atranka ir tarpvertintojų patikimumas.

Užduotis
--------
1. Įgyvendinkite ``screen_batch(records, criteria, *, batch_size, mode,
   call_llm)``:
   - patikrinkite, kad ``1 <= batch_size <= 3`` (kitaip ``ValueError``);
   - kiekvienam paketui sudarykite užklausą per
     ``toolkit.build_screening_request`` ir kvieskite ``call_llm(**request)``;
   - patikrinkite, kad grąžinta po vieną galiojantį sprendimą kiekvienam
     įrašui; klaidos atveju iškelkite ``ValueError`` (NIEKADA nerašykite
     „exclude" iš atsargumo).
2. Įgyvendinkite ``partition`` -- „borderline" visada į žmogaus eilę.
3. Įgyvendinkite ``inter_rater_reliability`` -- dvireikšmis sutraukimas,
   Cohen's kappa, jos juostos etiketė.
4. Atrinkite ``bib_records.json``, apskaičiuokite kappa prieš
   ``second_rater_decisions.json`` ir interpretuokite juostą.
5. Neprivaloma (live): pakartokite vieną paketą su realiu API raktu.

Paleidimas: ``python -m examples.chapter.s07_screening.exercise``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from examples.chapter import toolkit


class ScreenDecision(BaseModel):
    record_id: str
    decision: Literal["include", "borderline", "exclude"]
    reason: str = ""


class Partition(BaseModel):
    auto_included: list[str]
    auto_excluded: list[str]
    human_queue: list[str]


class IRRReport(BaseModel):
    n_pairs: int
    pct_agreement: float
    cohens_kappa: float
    kappa_label: str


def screen_batch(records, criteria, *, batch_size=3, mode="offline", call_llm=toolkit.call_llm):
    """TODO: paketinė atranka su tikrinimu."""
    raise NotImplementedError


def partition(decisions) -> Partition:
    """TODO: „borderline" -> žmogaus eilė."""
    raise NotImplementedError


def inter_rater_reliability(first: dict, second: dict) -> IRRReport:
    """TODO: dvireikšmis sutraukimas + Cohen's kappa."""
    raise NotImplementedError


def main() -> None:
    import re

    records = toolkit.load_fixture("bib_records.json")
    scripted = toolkit.scripted_llm(
        toolkit.SCRIPTED_RESPONSES["s07"]["batches"],
        key_fn=lambda m: ",".join(re.findall(r"\[(rec-\d+)\]", m[-1]["content"])),
    )
    print("S7 screening exercise")
    try:
        decisions = screen_batch(records, toolkit.DEMO_CRITERIA, call_llm=scripted)
    except NotImplementedError:
        print("screen_batch() is not implemented yet -- fill it in and re-run.")
        return
    print(f"got {len(decisions)} decisions")


if __name__ == "__main__":
    main()
