"""S12 sprendimas -- PRISMA srauto skaičiai su tvermės (konservacijos)
tikrinimu ir DI naudojimo atsekamumo ataskaitos skaitymas.

Mokomoji versija. Realiai tai atlieka ``slr_pipeline/nodes/reporting.py`` ir
``slr_pipeline/ai_traceability_report.py``; čia braižymo (matplotlib) nėra --
tik skaičiai ir jų tarpusavio darna.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PrismaInconsistencyError(ValueError):
    """PRISMA skaičiai nesutampa tarpusavyje arba būsena nėra galutinė."""


class StageOversight(BaseModel):
    """Vieno etapo priežiūros pjūvis trimis nepriklausomais matmenimis."""

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
        reasons = ", ".join(f"{k}={v}" for k, v in self.excluded_full_text_reasons.items())
        return (
            f"Identified: {self.total_identified}\n"
            f"Duplicates removed: {self.duplicates_removed}\n"
            f"After dedup / screened (title-abstract): {self.after_dedup} / {self.screened_title_abstract}\n"
            f"Excluded at title-abstract: {self.excluded_title_abstract}\n"
            f"Included after title-abstract: {self.included_after_ta}\n"
            f"Full text sought: {self.fulltext_sought}\n"
            f"Full text assessed: {self.assessed_full_text}\n"
            f"Full text not acquired: {self.acquisition_failures}\n"
            f"Excluded at full text: {self.excluded_full_text} ({reasons})\n"
            f"Included: {self.included}"
        )


# Stage -> the prisma_counts key whose `None` value is a positive "did not run"
# signal. Absence of AI events is NOT such a signal (round-2 finding 5).
_NOT_RUN_SIGNAL: dict[str, str] = {"quality_appraisal": "quality_assessed"}


# --- listing:s12-prisma-flow start ---
def prisma_flow(state: dict) -> PrismaFlow:
    """Sudaro ``PrismaFlow`` ir patikrina penkias tvermės lygčių grupes bei
    vieną tiriamojo (scoping) režimo tinkamumo patikrą.

    Pirmiausia reikalaujama, kad ``state["full_text_review_queue"]`` būtų
    tuščias -- kitaip būsena nėra galutinė ir srauto skaičiuoti negalima
    (``slr_pipeline/graphs/routing.py:43``).
    """
    if "full_text_review_queue" not in state:
        raise PrismaInconsistencyError("state is missing the required key 'full_text_review_queue'")
    if state["full_text_review_queue"] != []:
        raise PrismaInconsistencyError(
            "not a terminal state -- the full-text review queue must be exactly [] "
            f"(got {state['full_text_review_queue']!r})"
        )

    pc = state["prisma_counts"]
    for required_key in ("quality_assessed", "excluded_quality"):
        if required_key not in pc:
            raise PrismaInconsistencyError(f"prisma_counts is missing the required key {required_key!r}")
    total_identified = sum(pc["identified"].values())
    duplicates_removed = pc["duplicates_removed"]
    after_dedup = pc["after_dedup"]
    screened = pc["screened_title_abstract"]
    excluded_ta = pc["excluded_title_abstract"]
    retrieval = pc["fulltext_retrieval"]
    assessed = pc["assessed_full_text"]
    excluded_ft = pc["excluded_full_text"]
    excluded_ft_reasons = pc["excluded_full_text_reasons"]
    included = pc["included"]

    included_after_ta = screened - excluded_ta
    acquisition_failures = included_after_ta - assessed

    # 1 grupė -- identifikavimo / deduplikacijos / atrankos perdavimas
    if duplicates_removed < 0 or after_dedup != total_identified - duplicates_removed:
        raise PrismaInconsistencyError("group 1: after_dedup != total_identified - duplicates_removed")
    if screened != after_dedup:
        raise PrismaInconsistencyError("group 1: screened_title_abstract != after_dedup")
    # 2 grupė -- atmetimai pavadinimo/santraukos etape
    if included_after_ta < 0:
        raise PrismaInconsistencyError("group 2: excluded_title_abstract exceeds screened count")
    # 3 grupė -- kiekvienam įtrauktam įrašui bandyta gauti pilną tekstą
    if sum(retrieval.values()) != included_after_ta:
        raise PrismaInconsistencyError("group 3: sum(fulltext_retrieval) != included_after_ta")
    # 4 grupė -- pilno teksto gavimas
    if assessed != retrieval.get("parsed", 0):
        raise PrismaInconsistencyError("group 4: assessed_full_text != fulltext_retrieval['parsed']")
    if acquisition_failures < 0:
        raise PrismaInconsistencyError("group 4: assessed_full_text exceeds included_after_ta")
    # 5 grupė -- atmetimai pilno teksto etape
    if sum(excluded_ft_reasons.values()) != excluded_ft:
        raise PrismaInconsistencyError("group 5: sum(excluded_full_text_reasons) != excluded_full_text")
    if included != assessed - excluded_ft or included < 0:
        raise PrismaInconsistencyError("group 5: included != assessed_full_text - excluded_full_text")
    # 6 grupė -- tiriamojo (scoping) režimo tinkamumas: kokybės vertinimas praleistas
    if pc["quality_assessed"] is not None or pc["excluded_quality"] is not None:
        raise PrismaInconsistencyError("group 6: quality appraisal counts present in a scoping review")

    return PrismaFlow(
        total_identified=total_identified,
        duplicates_removed=duplicates_removed,
        after_dedup=after_dedup,
        screened_title_abstract=screened,
        excluded_title_abstract=excluded_ta,
        included_after_ta=included_after_ta,
        fulltext_sought=sum(retrieval.values()),
        assessed_full_text=assessed,
        acquisition_failures=acquisition_failures,
        excluded_full_text=excluded_ft,
        excluded_full_text_reasons=dict(excluded_ft_reasons),
        included=included,
    )
# --- listing:s12-prisma-flow end ---


# --- listing:s12-classify start ---
def classify_stage_oversight(
    traceability: dict, prisma_counts: dict, oversight_mode: dict
) -> dict[str, StageOversight]:
    """Kiekvienam atsekamumo ataskaitos etapui grąžina ``StageOversight`` su
    trimis nepriklausomais matmenimis: ar etapas vyko, kas jį automatizavo, ir
    ar jį peržiūrėjo žmogus.

    Svarbu: ``execution="not_run"`` nustatomas tik iš teigiamo signalo
    (``prisma_counts`` laukas lygus ``None``), o ne iš to, kad nėra DI įvykių.
    """
    result: dict[str, StageOversight] = {}
    for stage, entry in traceability["stages"].items():
        signal_key = _NOT_RUN_SIGNAL.get(stage)
        if signal_key is not None and signal_key not in prisma_counts:
            raise ValueError(
                f"stage {stage!r} needs prisma_counts[{signal_key!r}] to decide execution, "
                "but the key is absent"
            )
        if signal_key is not None and prisma_counts[signal_key] is None:
            execution = "not_run"
        elif entry.get("instrumentation_status") == "instrumented":
            execution = "run"
        else:
            execution = "unknown_or_not_instrumented"

        result[stage] = StageOversight(
            execution=execution,
            automation_origin=entry.get("automation_origin"),
            human_reviewed=oversight_mode.get(stage) == "human_resolved",
        )
    return result
# --- listing:s12-classify end ---
