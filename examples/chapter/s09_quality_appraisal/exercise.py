"""S9 pratimas -- kokybės vertinimas.

Užduotis
--------
1. Įgyvendinkite ``build_teaching_checklist()`` -- apie 6 elementus,
   atsakomus „yes" / „no" / „unclear".
2. Įgyvendinkite ``appraise(paper_text, checklist, *, call_llm)``:
   - paklauskite DKM kiekvieno elemento su įrodymo citata;
   - PAŽODŽIUI patikrinkite, ar citata yra tekste;
   - patvirtintas „no" -> ``provisional_pending_human_confirmation=True``
     (bet studija NEATMETAMA);
   - nepatvirtinta citata arba „unclear" be įrodymo -> suvienodinti į
     „unclear", ``provisional=False``, įrašyti į ``review_queue`` su
     priežastimi.
3. Įvertinkite ``papers/relevant.md``. Paaiškinkite skirtumą tarp
   provizorinio elemento (patvirtintas „no") ir sprendimų eilės elemento
   (nepatvirtinama), ir kodėl nė vienas neatmeta studijos automatiškai.

Paleidimas: ``python -m examples.chapter.s09_quality_appraisal.exercise``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from examples.chapter import toolkit


class ChecklistItem(BaseModel):
    item_id: str
    question: str


class AppraisalItem(BaseModel):
    item_id: str
    status: Literal["yes", "no", "unclear"]
    provisional_pending_human_confirmation: bool
    evidence_quote: str


class QueueEntry(BaseModel):
    item_id: str
    queue_reason: str


class AppraisalResult(BaseModel):
    items: list[AppraisalItem]
    n_provisional: int
    review_queue: list[QueueEntry]
    requires_human_action: bool


def build_teaching_checklist() -> list[ChecklistItem]:
    """TODO: apie 6 kontrolinio sąrašo elementai."""
    raise NotImplementedError


def appraise(paper_text: str, checklist, *, call_llm=toolkit.call_llm) -> AppraisalResult:
    """TODO: vertinimas su pažodiniu įrodymų tikrinimu ir dviem keliais."""
    raise NotImplementedError


def main() -> None:
    paper = toolkit.load_fixture("papers/relevant.md")
    scripted = toolkit.scripted_llm(
        {"relevant": toolkit.SCRIPTED_RESPONSES["s09"]["papers"]["relevant"]},
        key_fn=lambda m: "relevant",
    )
    print("S9 quality appraisal exercise")
    try:
        result = appraise(paper, build_teaching_checklist(), call_llm=scripted)
    except NotImplementedError:
        print("appraise()/build_teaching_checklist() not implemented yet -- fill them in and re-run.")
        return
    print(f"provisional: {result.n_provisional}; queue: {len(result.review_queue)}")


if __name__ == "__main__":
    main()
