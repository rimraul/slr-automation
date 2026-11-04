"""S7 sprendimas -- pavadinimo/santraukos atranka + tarpvertintojų
patikimumas (Cohen's kappa).

Mokomoji versija. Realūs etapai -- ``slr_pipeline/nodes/screening.py`` (DKM
paketais, atsarginis raktažodžių variantas, ``interrupt()``) ir
``slr_pipeline/nodes/screening_qc.py`` (``cohen_kappa_score``, PABAK, Gwet
AC1; „include"/„borderline" -> „include" prieš skaičiuojant).

Ta pati užklausos sudarymo funkcija (``toolkit.build_screening_request``)
naudojama IR neprisijungus (offline), IR realiu (live) režimu -- skiriasi tik
modelio identifikatorius.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score

from examples.chapter import toolkit

_DECISIONS = ("include", "borderline", "exclude")


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


# --- listing:s07-screen-batch start ---
def screen_batch(
    records: list[dict],
    criteria: toolkit.Criteria,
    *,
    batch_size: int = 3,
    mode: Literal["offline", "live"] = "offline",
    call_llm=toolkit.call_llm,
) -> list[ScreenDecision]:
    """Atrenka ``records`` paketais. Kiekvienam paketui sudaro užklausą per
    ``toolkit.build_screening_request`` ir kviečia ``call_llm``.

    Iškelia ``ValueError``, jei ``batch_size`` nėra tarp 1 ir 3. Kiekviename
    pakete tikrina, kad grąžinta po lygiai vieną sprendimą kiekvienam įrašui
    ir kad sprendimo reikšmė galiojanti -- niekada nenutyli klaidos ir
    niekada „iš atsargumo" nerašo „exclude".
    """
    if not 1 <= batch_size <= 3:
        raise ValueError("batch_size must be between 1 and 3")

    decisions: list[ScreenDecision] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        request = toolkit.build_screening_request(batch, criteria, mode=mode)
        raw = call_llm(**request)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model did not return JSON for batch starting at {start}: {exc}") from exc

        rows = payload.get("decisions", [])
        returned_ids = [row["record_id"] for row in rows]
        expected_ids = [rec["id"] for rec in batch]
        if sorted(returned_ids) != sorted(expected_ids):
            raise ValueError(
                f"batch decision id mismatch: expected {expected_ids}, got {returned_ids}"
            )
        for row in rows:
            if row["decision"] not in _DECISIONS:
                raise ValueError(f"invalid decision value: {row['decision']!r}")
            decisions.append(ScreenDecision(**row))
    return decisions
# --- listing:s07-screen-batch end ---


# --- listing:s07-partition start ---
def partition(decisions: list[ScreenDecision]) -> Partition:
    """Suskirsto sprendimus: „include" -> automatiškai įtraukti, „exclude" ->
    automatiškai atmesti, „borderline" -> žmogaus peržiūros eilė."""
    result = Partition(auto_included=[], auto_excluded=[], human_queue=[])
    for decision in decisions:
        if decision.decision == "include":
            result.auto_included.append(decision.record_id)
        elif decision.decision == "exclude":
            result.auto_excluded.append(decision.record_id)
        else:
            result.human_queue.append(decision.record_id)
    return result
# --- listing:s07-partition end ---


def _collapse(decision: str) -> str:
    return "include" if decision in ("include", "borderline") else "exclude"


def _kappa_label(k: float) -> str:
    if k != k:  # NaN
        return "n/a"
    if k < 0:
        return "Poor"
    if k <= 0.20:
        return "Slight"
    if k <= 0.40:
        return "Fair"
    if k <= 0.60:
        return "Moderate"
    if k <= 0.80:
        return "Substantial"
    return "Almost perfect"


# --- listing:s07-irr start ---
def inter_rater_reliability(first: dict, second: dict) -> IRRReport:
    """Palygina du vertintojus tik ten, kur abu turi sprendimą. Prieš
    skaičiuojant „include"/„borderline" suvienodinama į „include" (dvireikšmis
    sutraukimas, kaip ``screening_qc.py:335``). Antraštinis rodiklis -- Cohen's
    kappa, ne procentinis sutapimas, nes kappa įvertina atsitiktinį sutapimą.
    """
    shared = sorted(set(first) & set(second))
    a = [_collapse(first[k]) for k in shared]
    b = [_collapse(second[k]) for k in shared]
    agree = sum(x == y for x, y in zip(a, b))
    kappa = float(cohen_kappa_score(a, b))
    return IRRReport(
        n_pairs=len(shared),
        pct_agreement=agree / len(shared) if shared else 0.0,
        cohens_kappa=kappa,
        kappa_label=_kappa_label(kappa),
    )
# --- listing:s07-irr end ---


def _batch_key(messages: list[dict]) -> str:
    return ",".join(re.findall(r"\[(rec-\d+)\]", messages[-1]["content"]))


def main() -> None:
    records = toolkit.load_fixture("bib_records.json")
    scripted = toolkit.scripted_llm(
        toolkit.SCRIPTED_RESPONSES["s07"]["batches"], key_fn=_batch_key
    )
    decisions = screen_batch(records, toolkit.DEMO_CRITERIA, call_llm=scripted)
    part = partition(decisions)
    irr = inter_rater_reliability(
        toolkit.load_fixture("expected_screening.json"),
        toolkit.load_fixture("second_rater_decisions.json"),
    )
    print("S7 screening (scripted demo)")
    print(f"included={len(part.auto_included)} excluded={len(part.auto_excluded)} queue={len(part.human_queue)}")
    print(f"kappa={irr.cohens_kappa:.2f} ({irr.kappa_label}) over {irr.n_pairs} pairs")


if __name__ == "__main__":
    main()
