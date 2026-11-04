"""S11 pratimas -- sintezė ir temų modeliavimas.

Užduotis
--------
(a) Įgyvendinkite ``run_topics(texts, embeddings)`` -- BERTopic su iš anksto
    apskaičiuotais įterpiniais, fiksuota UMAP sėkla ir konfigūruotu HDBSCAN.
    ``bertopic``/``umap``/``hdbscan`` importuokite TIK funkcijos viduje.
    Paleiskite ant 12 dokumentų korpuso ir palyginkite rastų temų skaičių su
    projektuotomis 3 temomis.
(b) Įgyvendinkite ``map_study(study_id, rq, charted, *, call_llm)`` --
    ``study_id`` imkite IŠ PARAMETRO, ne iš modelio atsakymo; iškelkite
    ``ValueError``, jei ``cited_fields`` tuščias arba nurodo nesamą lauką.
    Tada ``refine_synthesis(rq, answers, ...)`` -- reikalaukite TIKSLIOS
    studijų aprėpties ir iškelkite A-vs-B prieštarą.

Paleidimas: ``python -m examples.chapter.s11_synthesis_topics.exercise``
"""

from __future__ import annotations

from pydantic import BaseModel

from examples.chapter import toolkit


class StudyAnswer(BaseModel):
    study_id: str
    answer_text: str
    cited_fields: list[str]


class Synthesis(BaseModel):
    narrative: str
    cited_study_ids: list[str]
    contradictions: list[str]


class TopicResult(BaseModel):
    assignments: list[int]
    topic_terms: dict[int, list[str]]


def run_topics(texts: list[str], embeddings) -> TopicResult:
    """TODO: BERTopic su iš anksto apskaičiuotais įterpiniais (lazy import)."""
    raise NotImplementedError


def map_study(study_id: str, rq: str, charted: dict, *, call_llm=toolkit.call_llm) -> StudyAnswer:
    """TODO: viena studija, tapatybė iš parametro."""
    raise NotImplementedError


def refine_synthesis(rq: str, answers, *, batch_size: int = 2, call_llm=toolkit.call_llm) -> Synthesis:
    """TODO: sulieti atsakymus, reikalauti tikslios aprėpties."""
    raise NotImplementedError


def main() -> None:
    data = toolkit.load_fixture("synthesis_studies.json")
    s11 = toolkit.SCRIPTED_RESPONSES["s11"]
    map_fake = toolkit.scripted_llm(
        s11["map"], key_fn=lambda m: m[-1]["content"].split("STUDY_ID: ", 1)[1].split("\n", 1)[0]
    )
    print("S11 synthesis exercise")
    try:
        answers = [
            map_study(s["study_id"], data["rq"], s["charted"], call_llm=map_fake)
            for s in data["studies"]
        ]
    except NotImplementedError:
        print("map_study() is not implemented yet -- fill it in and re-run.")
        return
    print(f"mapped {len(answers)} studies")


if __name__ == "__main__":
    main()
