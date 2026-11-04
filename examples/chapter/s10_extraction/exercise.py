"""S10 pratimas -- duomenų išgava (chartavimas).

Užduotis
--------
1. Apibrėžkite ``ExtractionSchema`` -- 4 laukai, ``strict=True``
   (``study_design``, ``sample_size: int | None``, ``country: str | None``,
   ``main_finding``).
2. Įgyvendinkite ``extract(paper_text, schema, *, call_llm)``:
   - paklauskite DKM kiekvieno lauko su įrodymo citata;
   - kiekvieną lauką tikrinkite ATSKIRAI (citata pažodžiui + tipas
     ``TypeAdapter(...).validate_python(value, strict=True)``);
   - sugedę laukai -> ``review_queue``, teisingi -> ``fields``;
   - pilną ``schema`` objektą kurkite tik jei VISI laukai praėjo.
3. Ištraukite 4 laukus iš ``papers/relevant.md``. Paaiškinkite, ką daryti su
   nepatikrinama reikšme (užuomina: neatmesti visos ekstrakcijos).

Paleidimas: ``python -m examples.chapter.s10_extraction.exercise``
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from examples.chapter import toolkit


class ExtractionSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    study_design: StrictStr
    sample_size: StrictInt | None
    country: StrictStr | None
    main_finding: StrictStr


class ExtractionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: dict[str, dict]
    data: BaseModel | None
    review_queue: list[dict]


def extract(paper_text: str, schema: type[BaseModel], *, call_llm=toolkit.call_llm) -> ExtractionResult:
    """TODO: lauko lygmens ekstrakcija su citatų ir tipų tikrinimu."""
    raise NotImplementedError


def main() -> None:
    paper = toolkit.load_fixture("papers/relevant.md")
    scripted = toolkit.scripted_llm(
        {"relevant": toolkit.SCRIPTED_RESPONSES["s10"]["papers"]["relevant"]},
        key_fn=lambda m: "relevant",
    )
    print("S10 extraction exercise")
    try:
        result = extract(paper, ExtractionSchema, call_llm=scripted)
    except NotImplementedError:
        print("extract() is not implemented yet -- fill it in and re-run.")
        return
    print(f"verified fields: {list(result.fields)}; queue: {result.review_queue}")


if __name__ == "__main__":
    main()
