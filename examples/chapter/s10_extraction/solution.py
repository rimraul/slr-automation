"""S10 sprendimas -- duomenų išgava su lauko lygmens tikrinimu.

Mokomoji versija. Realus etapas -- ``slr_pipeline/nodes/extraction.py`` su
užšaldyta ``slr_pipeline/extraction/`` varikliu.

Principas: kiekvienas laukas tikrinamas ATSKIRAI. Vieno lauko klaida
nepanaikina kitų, teisingai ištrauktų laukų -- jie lieka ``fields``, o
sugedę patenka į ``review_queue``. Pilnas ``schema`` objektas sukuriamas tik
tada, kai visi laukai praėjo patikrą.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, TypeAdapter

from examples.chapter import toolkit

_SYSTEM = (
    "Extract the requested fields from the paper. For each field return "
    '{"value": ..., "evidence_quote": a verbatim span from the paper}. '
    'Return JSON {"fields": {<name>: {"value", "evidence_quote"}}}.'
)


# --- listing:s10-schema start ---
class ExtractionSchema(BaseModel):
    """Keturių laukų mokomoji išgavos schema. ``strict=True`` reiškia, kad
    tipai netraukiami per prievartą: ``sample_size`` privalo būti tikras
    ``int``, ne ``"42"``.

    Visi keturi laukai yra PRIVALOMI (neturi numatytosios reikšmės);
    ``sample_size`` ir ``country`` gali būti ``None``, bet turi būti aiškiai
    nurodyti. Privalomumas nustatomas iš ``model_fields[...].is_required()``.
    """

    model_config = ConfigDict(strict=True)

    study_design: StrictStr
    sample_size: StrictInt | None
    country: StrictStr | None
    main_finding: StrictStr
# --- listing:s10-schema end ---


class ExtractionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: dict[str, dict]
    data: BaseModel | None
    review_queue: list[dict]


# --- listing:s10-extract start ---
def extract(paper_text: str, schema: type[BaseModel], *, call_llm=toolkit.call_llm) -> ExtractionResult:
    """Ištraukia ``schema`` laukus iš ``paper_text``, tikrindama kiekvieną
    lauką atskirai. Grąžina ``ExtractionResult`` su patikrintais laukais,
    sprendimų eile ir -- jei viskas praėjo -- pilnu ``schema`` objektu.
    """
    field_names = list(schema.model_fields)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"FIELDS: {', '.join(field_names)}\n\nPAPER\n{paper_text}"},
    ]
    payload = json.loads(call_llm(messages, response_format={"type": "json_object"}, temperature=0.0))
    returned = payload["fields"]

    fields: dict[str, dict] = {}
    review_queue: list[dict] = []
    collected: dict[str, object] = {}

    for name, field_info in schema.model_fields.items():
        row = returned.get(name)
        if row is None:
            review_queue.append({"field": name, "reason": "field_missing_from_response"})
            continue
        value = row.get("value")
        quote = row.get("evidence_quote", "") or ""
        if not quote or quote not in paper_text:
            review_queue.append({"field": name, "reason": "evidence_not_verbatim"})
            continue
        try:
            TypeAdapter(field_info.annotation).validate_python(value, strict=True)
        except Exception:  # noqa: BLE001 -- bet kokia tipo klaida siunčia lauką į eilę
            review_queue.append({"field": name, "reason": "value_failed_type_check"})
            continue
        fields[name] = {"value": value, "confidence": "verified", "evidence_quote": quote}
        collected[name] = value

    # Pilnas objektas kuriamas tik jei praėjo VISI privalomi laukai
    # (``is_required()``); neprivalomo lauko klaida jo neblokuoja.
    required = {n for n, f in schema.model_fields.items() if f.is_required()}
    data = schema(**collected) if required <= set(collected) else None
    return ExtractionResult(fields=fields, data=data, review_queue=review_queue)
# --- listing:s10-extract end ---


def main() -> None:
    paper = toolkit.load_fixture("papers/relevant.md")
    scripted = toolkit.scripted_llm(
        {"relevant": toolkit.SCRIPTED_RESPONSES["s10"]["papers"]["relevant"]},
        key_fn=lambda m: "relevant",
    )
    result = extract(paper, ExtractionSchema, call_llm=scripted)
    print("S10 extraction (scripted demo)")
    print(f"verified fields: {list(result.fields)}")
    print(f"review queue: {result.review_queue}")
    print(f"data built: {result.data is not None}")


if __name__ == "__main__":
    main()
