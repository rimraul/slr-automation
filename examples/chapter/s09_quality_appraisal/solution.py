"""S9 sprendimas -- kokybės vertinimas su įrodymų tikrinimu.

Mokomoji versija. Realus etapas --
``slr_pipeline/nodes/quality_appraisal.py``.

Du atskiri keliai (kaip realiame kode, ``quality_appraisal.py:909``):
  * 1 kelias -- patvirtintas atsakymas "no": elementas pažymimas
    ``provisional_pending_human_confirmation=True`` (laukia žmogaus
    patvirtinimo), bet studija NEATMETAMA.
  * 2 kelias -- įrodymo citata nepatvirtinta arba atsakymas "unclear" be
    įrodymo: atsakymas suvienodinamas į "unclear", ``provisional=False``, o
    elementas patenka į sprendimų eilę su priežastimi.

Kokybės vertinimas niekada nekeičia tinkamumo (įtraukimo) sprendimo ir
neduoda vieno bendro „balo".
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from examples.chapter import toolkit

_ALLOWED_ANSWERS = ("yes", "no", "unclear")

_SYSTEM = (
    "You appraise a study against a checklist. For each item return "
    '{"item_id", "answer" in {"yes","no","unclear"}, "evidence_quote" a verbatim '
    'span from the paper or ""}. Return JSON {"items": [ ... ]}.'
)

_QUEUE_VERIFICATION_FAILED = "evidence_verification_failed"
_QUEUE_NO_EVIDENCE = "no_evidence_found"


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


# --- listing:s09-checklist start ---
def build_teaching_checklist() -> list[ChecklistItem]:
    """Grąžina ~6 elementų mokomąjį kontrolinį sąrašą. Kiekvienas atsakomas
    „yes", „no" arba „unclear"."""
    return [
        ChecklistItem(item_id="objective_stated", question="Ar aiškiai suformuluotas tyrimo tikslas?"),
        ChecklistItem(item_id="eligibility_criteria_prespecified", question="Ar atrankos kriterijai apibrėžti iš anksto?"),
        ChecklistItem(item_id="search_reproducible", question="Ar paieška aprašyta taip, kad ją būtų galima pakartoti?"),
        ChecklistItem(item_id="dual_independent_screening", question="Ar atranką atliko du nepriklausomi vertintojai?"),
        ChecklistItem(item_id="limitations_discussed", question="Ar aptarti tyrimo apribojimai?"),
        ChecklistItem(item_id="conflicts_of_interest_declared", question="Ar deklaruoti interesų konfliktai?"),
    ]
# --- listing:s09-checklist end ---


TEACHING_APPRAISAL_V1 = build_teaching_checklist()


# --- listing:s09-appraise start ---
def appraise(paper_text: str, checklist: list[ChecklistItem], *, call_llm=toolkit.call_llm) -> AppraisalResult:
    """Įvertina ``paper_text`` pagal ``checklist``, patikrindama kiekvieną
    įrodymo citatą pažodžiui ir nukreipdama neaiškius elementus į sprendimų
    eilę."""
    questions = "\n".join(f"{item.item_id}: {item.question}" for item in checklist)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"CHECKLIST\n{questions}\n\nPAPER\n{paper_text}"},
    ]
    payload = json.loads(call_llm(messages, response_format={"type": "json_object"}, temperature=0.0))
    rows = payload["items"]

    returned_ids = [row["item_id"] for row in rows]
    expected_ids = [item.item_id for item in checklist]
    if sorted(returned_ids) != sorted(expected_ids):
        raise ValueError(f"checklist coverage mismatch: got {returned_ids}, expected each of {expected_ids} once")
    by_id = {row["item_id"]: row for row in rows}

    items: list[AppraisalItem] = []
    review_queue: list[QueueEntry] = []
    n_provisional = 0

    for item in checklist:
        row = by_id[item.item_id]
        answer = row["answer"]
        if answer not in _ALLOWED_ANSWERS:
            raise ValueError(f"unsupported appraisal answer for {item.item_id!r}: {answer!r}")
        quote = row.get("evidence_quote", "") or ""
        quote_ok = bool(quote) and quote in paper_text

        if quote and not quote_ok:
            # išgalvota / nepatikrinama įrodymo citata
            review_queue.append(QueueEntry(item_id=item.item_id, queue_reason=_QUEUE_VERIFICATION_FAILED))
            items.append(AppraisalItem(item_id=item.item_id, status="unclear",
                                       provisional_pending_human_confirmation=False, evidence_quote=""))
            continue

        if not quote_ok:
            # bet koks apibrėžtas atsakymas („yes"/„no") be patikrintos citatos,
            # taip pat ir „unclear" be įrodymo, suvienodinamas į „unclear" ir
            # patenka į sprendimų eilę
            review_queue.append(QueueEntry(item_id=item.item_id, queue_reason=_QUEUE_NO_EVIDENCE))
            items.append(AppraisalItem(item_id=item.item_id, status="unclear",
                                       provisional_pending_human_confirmation=False, evidence_quote=""))
            continue

        if answer == "no":
            n_provisional += 1
            items.append(AppraisalItem(item_id=item.item_id, status="no",
                                       provisional_pending_human_confirmation=True, evidence_quote=quote))
            continue

        items.append(AppraisalItem(item_id=item.item_id, status=answer,
                                   provisional_pending_human_confirmation=False, evidence_quote=quote))

    return AppraisalResult(
        items=items,
        n_provisional=n_provisional,
        review_queue=review_queue,
        requires_human_action=n_provisional > 0 or bool(review_queue),
    )
# --- listing:s09-appraise end ---


def main() -> None:
    paper = toolkit.load_fixture("papers/relevant.md")
    scripted = toolkit.scripted_llm(
        {"relevant": toolkit.SCRIPTED_RESPONSES["s09"]["papers"]["relevant"]},
        key_fn=lambda m: "relevant",
    )
    result = appraise(paper, TEACHING_APPRAISAL_V1, call_llm=scripted)
    print("S9 quality appraisal (scripted demo)")
    print(f"provisional items: {result.n_provisional}; review queue: {len(result.review_queue)}")
    print(f"requires human action: {result.requires_human_action}")


if __name__ == "__main__":
    main()
