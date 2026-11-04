"""S11 sprendimas -- teiginiais paremta sintezė (map/refine) ir temų
modeliavimas (BERTopic).

Mokomoji versija. Realus etapas -- ``slr_pipeline/nodes/synthesis.py``.

Esminė mintis (kaip ``synthesis.py:432``): studijos tapatybė (``study_id``)
prijungiama PATIKIMAME kode iš parametro, o ne imama iš modelio atsakymo.
Modelis prisideda tik prie ``answer_text`` ir ``cited_fields``.

Temų modeliavimas veikia neprisijungus: naudojami iš anksto apskaičiuoti
įterpiniai (``embeddings``), o ``bertopic``/``umap``/``hdbscan`` importuojami
TIK funkcijos viduje, kad importo kaina nekristų ant kiekvieno testo.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from examples.chapter import toolkit

_MAP_SYSTEM = (
    "Answer the review question for ONE study using ONLY its charted fields. "
    'Return JSON {"answer_text": str, "cited_fields": [field names you used]}.'
)
_REFINE_SYSTEM = (
    "Fold the per-study answers into one narrative for the review question. "
    'Return JSON {"narrative": str, "cited_study_ids": [every study id], '
    '"contradictions": [short descriptions of any disagreement]}.'
)


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


# --- listing:s11-run-topics start ---
def run_topics(texts: list[str], embeddings) -> TopicResult:
    """Sugrupuoja ``texts`` į temas naudodama iš anksto apskaičiuotus
    ``embeddings``. Deterministiška: UMAP ir HDBSCAN su fiksuota sėkla /
    konfigūracija, jokio įterpinių modelio (``embedding_model=None``)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric="euclidean", cluster_selection_method="eom")
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
    )
    assignments, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    topic_terms = {
        int(topic_id): [word for word, _score in topic_model.get_topic(topic_id) or []]
        for topic_id in set(int(a) for a in assignments)
    }
    return TopicResult(assignments=[int(a) for a in assignments], topic_terms=topic_terms)
# --- listing:s11-run-topics end ---


# --- listing:s11-map-study start ---
def map_study(study_id: str, rq: str, charted: dict, *, call_llm=toolkit.call_llm) -> StudyAnswer:
    """Sukuria vienos studijos atsakymą, pagrįstą TIK jos ``charted`` laukais.

    ``study_id`` prijungiamas iš parametro (patikimas kodas). Iškelia
    ``ValueError``, jei ``cited_fields`` tuščias arba nurodo lauką, kurio nėra
    ``charted``.
    """
    messages = [
        {"role": "system", "content": _MAP_SYSTEM},
        {"role": "user", "content": f"STUDY_ID: {study_id}\nRQ: {rq}\nCHARTED: {json.dumps(charted)}"},
    ]
    payload = json.loads(call_llm(messages, response_format={"type": "json_object"}, temperature=0.0))
    cited_fields = payload.get("cited_fields", [])
    if not cited_fields:
        raise ValueError(f"{study_id}: model cited no charted fields")
    unknown = set(cited_fields) - set(charted)
    if unknown:
        raise ValueError(f"{study_id}: model cited fields not in charted data: {sorted(unknown)}")
    return StudyAnswer(study_id=study_id, answer_text=payload["answer_text"], cited_fields=list(cited_fields))
# --- listing:s11-map-study end ---


def _render_answers(answers: list[StudyAnswer]) -> str:
    return "\n".join(f"[{a.study_id}] {a.answer_text}" for a in answers)


def _refine_key(messages: list[dict]) -> str:
    """Dispatch key for a scripted refine fake: the sorted set of study ids
    visible in the fold's prompt (current narrative + new answers)."""
    return ",".join(sorted(set(re.findall(r"study-[A-Za-z0-9]+", messages[-1]["content"]))))


# --- listing:s11-refine start ---
def refine_synthesis(
    rq: str, answers: list[StudyAnswer], *, batch_size: int = 2, call_llm=toolkit.call_llm
) -> Synthesis:
    """Iteratyviai sulieja per-studijos atsakymus į vieną naratyvą.

    Aprėptis tikrinama PO KIEKVIENO suliejimo: ``set(cited_study_ids)`` privalo
    lygiai sutapti su iki tol MATYTŲ studijų aibe. Taip pagaunama ir tarpinė
    haliucinacija (dar nematytos studijos citata pirmame žingsnyje), ir
    galutinis praleidimas ar išgalvota citata.
    """
    if not answers:
        raise ValueError("no answers to synthesise")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    narrative = ""
    seen_ids: set[str] = set()
    result: Synthesis | None = None
    for start in range(0, len(answers), batch_size):
        batch = answers[start:start + batch_size]
        seen_ids |= {a.study_id for a in batch}
        messages = [
            {"role": "system", "content": _REFINE_SYSTEM},
            {"role": "user", "content": f"RQ: {rq}\nCURRENT NARRATIVE: {narrative}\nNEW ANSWERS:\n{_render_answers(batch)}"},
        ]
        payload = json.loads(call_llm(messages, response_format={"type": "json_object"}, temperature=0.0))
        result = Synthesis(**payload)
        narrative = result.narrative
        if set(result.cited_study_ids) != seen_ids:
            raise ValueError(
                f"fold citation coverage mismatch: cited {sorted(set(result.cited_study_ids))}, "
                f"studies seen so far {sorted(seen_ids)}"
            )
    return result
# --- listing:s11-refine end ---


def main() -> None:
    data = toolkit.load_fixture("synthesis_studies.json")
    s11 = toolkit.SCRIPTED_RESPONSES["s11"]
    map_fake = toolkit.scripted_llm(
        s11["map"], key_fn=lambda m: m[-1]["content"].split("STUDY_ID: ", 1)[1].split("\n", 1)[0]
    )
    refine_fake = toolkit.scripted_llm(s11["refine"], key_fn=_refine_key)

    answers = [
        map_study(study["study_id"], data["rq"], study["charted"], call_llm=map_fake)
        for study in data["studies"]
    ]
    synthesis = refine_synthesis(data["rq"], answers, call_llm=refine_fake)
    print("S11 synthesis (scripted demo)")
    print(f"cited studies: {synthesis.cited_study_ids}")
    print(f"contradictions: {synthesis.contradictions}")


if __name__ == "__main__":
    main()
