"""Generates one Colab notebook per chapter exercise (S3-S12) into
examples/chapter/notebooks/. Notebooks are derived from each section's
exercise.py; never edit them by hand -- re-run this script.

    python scripts/build_colab_notebooks.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_demos  # noqa: E402
CHAPTER = ROOT / "examples" / "chapter"
OUT = CHAPTER / "notebooks"
# Placeholder until the code-release URL is fixed (see manuscript PHASEF-PENDING).
REPO_URL = "https://github.com/rimraul/slr-automation.git"

SETUP = f'''import os, subprocess, sys

REPO_URL = "{REPO_URL}"
REPO_DIR = REPO_URL.rsplit("/", 1)[-1].removesuffix(".git")
if "google.colab" in sys.modules and not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
if os.path.isdir(REPO_DIR):
    os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())
for req in {{REQS}}:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("pip install nepavyko (zr. klaida aukščiau): " + req)
print("Paruosta:", os.getcwd())'''

NOTICE = (
    "### Prieš pradedant\n\n"
    "1. **Pirmas paleidimas.** Paleidus pirmą kodo langelį, Colab gali parodyti įspėjimą "
    "*„Notebook was not authored by Google“*. Tai normalu (sąsiuvinis atkeltas iš GitHub): "
    "spauskite **Run anyway**.\n"
    "2. **Pirmas langelis** atsisiunčia kodą ir įdiegia bibliotekas (maždaug 1-2 min.). "
    "Paleiskite jį pirmą ir palaukite, kol pasirodys `Paruosta:`.\n"
    "3. **Tikras DKM nebūtinas.** Jei norite jį išbandyti: kairėje Colab juostoje spauskite "
    "rakto piktogramą (*Secrets*), *Add new secret*, pavadinimas lygiai **`OPENAI_API_KEY`**, "
    "reikšmė -- jūsų OpenAI raktas, ir įjunkite *Notebook access*. Raktą rašykite tik čia, "
    "niekada į kodo langelį. Kreipiniai apmokami."
)

LIVE = '''# NEPRIVALOMA: tikri kreipiniai i DKM (sumoka uz naudojimą). Raktą laikykite
# Colab "Secrets" (rakto piktograma kairėje) pavadinimu OPENAI_API_KEY; niekada ne užrašykite kode.
# from google.colab import userdata
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")'''


def cell(kind: str, text: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


S04_SEED = "large language models for systematic review screening"

S04_PARAMS = r'''#@title 1. Pasirinkite užklausą ir duomenų bazes
SEED = "__SEED__"  #@param {type:"string"}
PUBMED = True  #@param {type:"boolean"}
OPENALEX = True  #@param {type:"boolean"}
SCOPUS = True  #@param {type:"boolean"}
# Pagal nutylėjimą DKM NEnaudojamas (nemokama): koncepcijas redaguojate patys (2 langelis),
# o SEED čia tik pavyzdys. SEED veikia tik kai USE_LIVE įjungtas.
# Pažymėję USE_LIVE, DKM išplės JŪSŲ užklausą (reikia rakto Colab "Secrets";
# kreipinys apmokamas, paprastai kainuoja centus).
USE_LIVE = False  #@param {type:"boolean"}'''

S04_EDIT_HEAD = '''#@title 2. Redaguokite koncepcijas ir sinonimus (kai USE_LIVE išjungtas)
# Kiekviena eilutė -- viena koncepcija; sąrašas -- jos sinonimai.
# Pridėkite, ištrinkite ar pakeiskite terminus ir paleiskite 3 langelį iš naujo.
'''

S04_RUN = r'''#@title 3. Išplėsti ir išversti
from examples.chapter.s04_query_expansion import solution as s4

if USE_LIVE:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("Kreipinys į DKM (apmokamas). Užklausa:", SEED)
    expansion = s4.expand_query(SEED)  # tikras DKM kreipinys
else:
    expansion = s4.QueryExpansion(
        concepts=[s4.Concept(label=k, terms=v) for k, v in CONCEPTS.items()],
        boolean_query=" AND ".join("(" + " OR ".join(v) + ")" for v in CONCEPTS.values()))

print("KONCEPCIJOS IR SINONIMAI")
for c in expansion.concepts:
    print(f"  {c.label}: {', '.join(c.terms)}")

chosen = [d for d, on in (("pubmed", PUBMED), ("openalex", OPENALEX), ("scopus", SCOPUS)) if on]
print("\nVERTIMAS Į DUOMENŲ BAZIŲ SINTAKSĘ (be DKM, grynos taisyklės)")
for db in chosen:
    print(f"\n[{db}]\n{s4.translate_for_database(expansion, db)}")
print("\nSkirtumai: PubMed prideda [tiab] prie kiekvieno termino; OpenAlex -- be lauko žymų; "
      "Scopus apgaubia visą išraišką TITLE-ABS-KEY(...).")
print("Prisiminkite: DKM pasiūlytą užklausą visada turi peržiūrėti žmogus.")'''

S04_PROMPT = r'''#@title (nebūtina) Ką DKM gauna iš tikrųjų?
from examples.chapter.s04_query_expansion import solution as s4
print("SISTEMOS INSTRUKCIJA:\n" + s4._SYSTEM)
print("\nVARTOTOJO ŽINUTĖ:\n" + SEED)'''

S04_INTRO = (
    "# S04 -- užklausos plėtimas ir vertimas\n\n"
    "**A kelias (be programavimo).** Paleiskite 1-3 langelius iš eilės. Matysite, "
    "kaip pradinė užklausa išskaidoma į koncepcijas ir sinonimus ir kaip ta pati paieška "
    "užrašoma PubMed, OpenAlex ir Scopus sintakse. Be rakto galite ranka keisti koncepcijas ir sinonimus (2 langelis) ir matyti, kaip keičiasi "
    "vertimas. Užklausos laukelis (SEED) naudojamas tik įjungus USE_LIVE: tada tikras DKM "
    "išplečia JŪSŲ užklausą (reikia savo OpenAI rakto, kreipinys apmokamas).\n\n"
    "**B kelias (programuotojams)** -- žemiau, po demonstracija: patys įgyvendinkite funkcijas."
)


def demo_s04(setup_cell: dict) -> list[dict]:
    """Track A for S4: no coding. Edit concepts, see the database translations."""
    fx = json.loads((CHAPTER / "data" / "scripted_responses" / "s04.json").read_text(encoding="utf-8"))
    default = json.loads(fx[S04_SEED])
    concepts = "CONCEPTS = {\n" + "".join(
        f"    {c['label']!r}: {c['terms']!r},\n" for c in default["concepts"]) + "}"
    return [
        cell("markdown", S04_INTRO),
        cell("markdown", NOTICE),
        setup_cell,
        cell("code", S04_PARAMS.replace("__SEED__", S04_SEED)),
        cell("code", S04_EDIT_HEAD + concepts),
        cell("code", S04_RUN),
        cell("code", S04_PROMPT),
        cell("markdown", "---\n## B kelias: parašykite kodą patys\n\n"
             "Toliau esantis pratimas skirtas tiems, kas nori išmokti programuoti."),
    ]


def build(sec: Path) -> dict:
    src = (sec / "exercise.py").read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or sec.name
    body = re.sub(r'^if __name__ == "__main__":\n\s+main\(\)\s*$', "main()", src, flags=re.M)
    body = re.sub(r'^""".*?"""\n', "", body, count=1, flags=re.S)
    # notebooks have no __file__; the setup cell chdirs to the repo root
    body = body.replace('Path(__file__).resolve().parents[1] / "data"', 'Path("examples/chapter") / "data"')
    sid = sec.name.split("_")[0].upper()
    ref = f"examples/chapter/{sec.name}"
    reqs = ["examples/chapter/requirements-colab.txt"]
    if sid == "S11":  # topic modelling only: heavy bertopic/torch chain
        reqs.append("examples/chapter/requirements-colab-topics.txt")
    setup_cell = cell("code", SETUP.replace("{REQS}", repr(reqs)))
    head = []
    if sid == "S04":
        head = demo_s04(setup_cell)
    elif sid in colab_demos.DEMOS:
        head = colab_demos.DEMOS[sid](setup_cell, cell, {"notice": cell("markdown", NOTICE)})
    cells = head + [
        cell("markdown", f"# {sid} pratimas\n\n```\n{doc}\n```\n\n"
             "1. Paleiskite paruošimo langelį. 2. Užpildykite `TODO` funkcijas "
             "pratimo langelyje ir paleiskite jį. 3. Palyginkite su pavyzdiniu sprendimu."),
        *([] if head else [cell("markdown", NOTICE), setup_cell]),
        cell("code", body),
        cell("markdown", "## Pavyzdinis sprendimas\n\nPažiūrėkite tik pabandę patys. "
             "Žemiau esantys testai tikrina **pavyzdinį sprendimą**, ne jūsų kodą."),
        cell("code", f"!cat {ref}/solution.py"),
        cell("code", f"!python -m pytest {ref} -q"),
        cell("code", LIVE),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for sec in sorted(CHAPTER.glob("s[0-9][0-9]_*")):
        if not (sec / "exercise.py").is_file():
            continue
        path = OUT / f"{sec.name}.ipynb"
        path.write_text(json.dumps(build(sec), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
