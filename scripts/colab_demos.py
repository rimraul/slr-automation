"""No-code "A path" demo cells for the Colab notebooks (S05-S07).

Used by build_colab_notebooks.py. Cell code is stored as raw strings so the
backslash escapes stay literal inside the generated notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "chapter" / "data"

B_PATH = (
    "---\n## B kelias: parašykite kodą patys\n\n"
    "Toliau esantis pratimas skirtas tiems, kas nori išmokti programuoti."
)

LIVE_NOTE = (
    "**Tikras DKM (nebūtina, apmokama).** Kairėje Colab juostoje spauskite rakto piktogramą "
    "(*Secrets*) -> *Add new secret* -> pavadinimas lygiai `OPENAI_API_KEY`, reikšmė -- jūsų "
    "OpenAI raktas, ir įjunkite *Notebook access*. Tada pažymėkite `USE_LIVE`."
)


# ---------------------------------------------------------------- S05 -------

S05_INTRO = (
    "# S05 -- daugiabazė paieška\n\n"
    "**A kelias (be programavimo).** Kiekviena duomenų bazė grąžina atsakymą sava forma "
    "(PubMed -- XML, kitos -- JSON su skirtingais laukais). Šis demonstracinis langelis "
    "parodo neapdorotą atsakymą ir tai, kaip jis paverčiamas vienu standartiniu įrašų "
    "sąrašu. Pasirinkite šaltinius ir paleiskite 1-2 langelius.\n\n"
    "Pastaba: čia naudojami **iš anksto išsaugoti** atsakymai, todėl užklausos tekstas "
    "rezultato nekeičia; tikra paieška reikalautų kiekvienos duomenų bazės API rakto.\n\n"
    "**B kelias (programuotojams)** -- žemiau."
)

S05_PARAMS = r'''#@title 1. Pasirinkite šaltinius
OPENALEX = True  #@param {type:"boolean"}
PUBMED = True  #@param {type:"boolean"}
CROSSREF = True  #@param {type:"boolean"}
SCOPUS = True  #@param {type:"boolean"}
SHOW_RAW_CHARS = 500  #@param {type:"slider", min:0, max:2000, step:100}'''

S05_RUN = r'''#@title 2. Gauti, standartizuoti ir sujungti
from examples.chapter.s05_multi_db_search import solution as s5

names = [n for n, on in (("openalex", OPENALEX), ("pubmed", PUBMED),
                         ("crossref", CROSSREF), ("scopus", SCOPUS)) if on]
adapters = [s5.FixtureAdapter(n) for n in names]
if SHOW_RAW_CHARS:
    print("NEAPDOROTI ATSAKYMAI (pradžia)")
    for a in adapters:
        print(f"\n--- {a.name} ---\n{a.fetch('').strip()[:SHOW_RAW_CHARS]} ...")

records = s5.run_search(adapters, "review automation")
print(f"\nSUJUNGTA: {len(records)} įrašai iš {len(adapters)} šaltinių (dublikatai NEšalinami)\n")
print(f"{'šaltinis':9} {'metai':5} {'DOI':28} pavadinimas")
for r in records:
    print(f"{r['source']:9} {str(r['year']):5} {str(r['doi'] or '-')[:28]:28} {r['title'][:70]}")

from collections import defaultdict
by_title = defaultdict(list)
for r in records:
    by_title[" ".join(r["title"].lower().split())].append(r["source"])
same = {t: s for t, s in by_title.items() if len(s) > 1}
print(f"\nTAS PATS DARBAS keliuose šaltiniuose (tikslus pavadinimo sutapimas): {len(same)}")
for t, s in same.items():
    print(f"  {t[:70]}  <- {', '.join(s)}")
print("Panašūs, bet ne identiški pavadinimai ar įrašai be DOI lieka atskirai -- tai S6 darbas.")'''


def demo_s05(setup_cell: dict, cell, common: dict) -> list[dict]:
    return [
        cell("markdown", S05_INTRO), common["notice"], setup_cell,
        cell("code", S05_PARAMS), cell("code", S05_RUN), cell("markdown", B_PATH),
    ]


# ---------------------------------------------------------------- S06 -------

S06_INTRO = (
    "# S06 -- deduplikacija\n\n"
    "**A kelias (be programavimo).** 18 bibliografinių įrašų iš kelių duomenų bazių; kai "
    "kurie yra to paties darbo dublikatai. Paleiskite 1-3 langelius ir žiūrėkite, kuriuos "
    "įrašus pašalina kaskada: pirma pagal DOI, paskui pagal panašų pavadinimą. "
    "Slankikliu keiskite panašumo slenkstį ir matykite, kaip tai keičia rezultatą.\n\n"
    "**B kelias (programuotojams)** -- žemiau."
)

S06_PARAMS = r'''#@title 1. Pasirinkite panašumo slenkstį ir DOI patikrą
THRESHOLD = 0.90  #@param {type:"slider", min:0.5, max:1.0, step:0.01}
DOI_TEST = "HTTP://DX.DOI.ORG/10.1000/Rev.001"  #@param {type:"string"}'''

S06_EDIT = r'''#@title 2. (nebūtina) Pridėkite savo įrašų
# Pridėkite savo įrašus (id, title, doi) -- jie bus deduplikuojami kartu su 18 pavyzdinių.
# Pavyzdys (atkomentuokite): vienas tikras dublikatas ir vienas panašus, bet skirtingas darbas.
EXTRA_RECORDS = [
    # {"id": "my-1", "title": "Large language models for title and abstract screening in systematic reviews", "doi": None},
    # {"id": "my-2", "title": "Large language models for full text screening in systematic reviews", "doi": None},
]'''

S06_RUN = r'''#@title 3. Deduplikuoti
import difflib
from examples.chapter import toolkit
from examples.chapter.s06_deduplication import solution as s6

print("DOI normalizavimas:", repr(DOI_TEST), "->", repr(s6.normalise_doi(DOI_TEST)))
records = toolkit.load_fixture("bib_records.json") + EXTRA_RECORDS
by_id = {r["id"]: r for r in records}
res = s6.dedupe(records, title_threshold=THRESHOLD)
print(f"\nĮRAŠŲ: {len(records)}  ->  UNIKALIŲ: {len(res.unique)}  (slenkstis {THRESHOLD:.2f})\n")
for p in res.duplicates:
    a, b = by_id[p.dup_id], by_id[p.canonical_id]
    ratio = difflib.SequenceMatcher(None, s6._normalise_title(a["title"]), s6._normalise_title(b["title"])).ratio()
    print(f"[{p.matched_on:5}] {p.dup_id} pašalintas, paliktas {p.canonical_id}  (pavadinimų panašumas {ratio:.2f})")
    print(f"        {a['title'][:80]}\n        {b['title'][:80]}")
print("\nKAIP KEIČIASI REZULTATAS SU SLENKSČIU")
for t in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00):
    n = len(s6.dedupe(records, title_threshold=t).duplicates)
    print(f"  slenkstis {t:.2f}: pašalinta {n}")
print("Pavyzdinėje aibėje tikri dublikatai yra 3. Ties 0.50 pašalinamas 4-as įrašas -- tai jau SKIRTINGAS darbas (klaidingas sujungimas); "
      "ties 0.95 praleidžiamas tikras dublikatas.")'''


def demo_s06(setup_cell: dict, cell, common: dict) -> list[dict]:
    return [
        cell("markdown", S06_INTRO), common["notice"], setup_cell,
        cell("code", S06_PARAMS), cell("code", S06_EDIT), cell("code", S06_RUN),
        cell("markdown", B_PATH),
    ]


# ---------------------------------------------------------------- S07 -------

S07_INTRO = (
    "# S07 -- atranka pagal pavadinimą ir santrauką\n\n"
    "**A kelias (be programavimo).** Pagal protokolo kriterijus 18 įrašų atrenkami "
    "„include / borderline / exclude“. Pagal nutylėjimą rodomi **iš anksto išsaugoti** "
    "DKM sprendimai (nemokama). Matysite, kuriuos įrašus sistema palieka žmogui "
    "(*borderline*) ir kaip sutaria du vertintojai (Cohen kappa). Redaguokite antrojo "
    "vertintojo sprendimus (2 langelis) ir žiūrėkite, kaip keičiasi kappa.\n\n"
    "Pažymėjus `USE_LIVE`, tikras DKM atrenka pirmus kelis įrašus pagal JŪSŲ kriterijus.\n\n"
    "**B kelias (programuotojams)** -- žemiau."
)

S07_PARAMS = r'''#@title 1. Pasirinkite režimą
USE_LIVE = False  #@param {type:"boolean"}
# Tik USE_LIVE: kiek pirmų įrašų atrinkti (po 3 viename pakete; kiekvienas paketas -- vienas apmokamas kreipinys)
N_LIVE = 3  #@param {type:"slider", min:3, max:9, step:3}'''

S07_EDIT_HEAD = '''#@title 2. Redaguokite (second_rater -- visada; kriterijai -- tik kai USE_LIVE)
# Antrojo vertintojo sprendimai: "include", "borderline" arba "exclude".
# Pakeiskite kelias reikšmes ir paleiskite 3 langelį iš naujo -- kappa pasikeis.
'''

S07_CRITERIA = r'''
# Kriterijai DKM (veikia tik su USE_LIVE; pagal nutylėjimą -- mokomasis protokolas)
POPULATION = "Studies about the conduct of systematic, scoping or rapid literature reviews."
CONCEPT = "Automation of one or more review steps using LLMs, machine learning or rule-based tools."
CONTEXT = "Academic or applied research; any discipline; English; 2015 onward."
INCLUSION = [
    "The work describes or evaluates an automated or semi-automated method for a distinct review step.",
    "Automation of the named step is the subject of the report, not an incidental mention.",
]
EXCLUSION = [
    "The work is an opinion piece or editorial with no method and no evaluation.",
    "The work is about primary research automation rather than literature-review automation.",
    "Full text is not in English.",
]'''

S07_RUN = r'''#@title 3. Atrinkti ir palyginti vertintojus
import re
from examples.chapter import toolkit
from examples.chapter.s07_screening import solution as s7

records = toolkit.load_fixture("bib_records.json")
if USE_LIVE:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    criteria = toolkit.Criteria(population=POPULATION, concept=CONCEPT, context=CONTEXT,
                                inclusion_criteria=tuple(INCLUSION), exclusion_criteria=tuple(EXCLUSION))
    records = records[:N_LIVE]
    print(f"Kreipiniai į DKM (apmokami): {len(records) // 3} pakeit(-as/-ų), {len(records)} įrašai")
    decisions = s7.screen_batch(records, criteria, mode="live")
else:
    scripted = toolkit.scripted_llm(toolkit.SCRIPTED_RESPONSES["s07"]["batches"],
                                    key_fn=lambda m: ",".join(re.findall(r"\[(rec-\d+)\]", m[-1]["content"])))
    decisions = s7.screen_batch(records, toolkit.DEMO_CRITERIA, call_llm=scripted)

by_id = {r["id"]: r for r in records}
print(f"{'id':7} {'DKM':11} {'2 vertintojas':13} pavadinimas")
for d in decisions:
    print(f"{d.record_id:7} {d.decision:11} {SECOND_RATER.get(d.record_id, '-'):13} {by_id[d.record_id]['title'][:60]}")

part = s7.partition(decisions)
print(f"\nĮtraukta automatiškai: {len(part.auto_included)} | atmesta automatiškai: {len(part.auto_excluded)}"
      f" | ŽMOGAUS EILĖ (borderline): {len(part.human_queue)} {part.human_queue}")

irr = s7.inter_rater_reliability({d.record_id: d.decision for d in decisions}, SECOND_RATER)
print(f"\nDKM prieš 2 vertintoją: kappa={irr.cohens_kappa:.2f} ({irr.kappa_label}), "
      f"sutapimas {irr.pct_agreement:.0%}, {irr.n_pairs} porų")
ref = s7.inter_rater_reliability(toolkit.load_fixture("expected_screening.json"), SECOND_RATER)
print(f"Etaloninis sąrašas prieš 2 vertintoją: kappa={ref.cohens_kappa:.2f} ({ref.kappa_label})")
print("Prisiminkite: ‘borderline’ visada tikrina žmogus; DKM sprendimas -- ne galutinis.")'''

S07_SYSTEM = r'''#@title (nebūtina) Ką DKM gauna iš tikrųjų?
from examples.chapter import toolkit
req = toolkit.build_screening_request(toolkit.load_fixture("bib_records.json")[:1], toolkit.DEMO_CRITERIA, mode="offline")
for m in req["messages"]:
    print(f"[{m['role']}]\n{m['content']}\n")'''


def demo_s07(setup_cell: dict, cell, common: dict) -> list[dict]:
    second = json.loads((DATA / "second_rater_decisions.json").read_text(encoding="utf-8"))
    sr = "SECOND_RATER = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in second.items()) + "}"
    return [
        cell("markdown", S07_INTRO), common["notice"], setup_cell,
        cell("code", S07_PARAMS), cell("code", S07_EDIT_HEAD + sr + "\n" + S07_CRITERIA),
        cell("code", S07_RUN), cell("code", S07_SYSTEM), cell("markdown", B_PATH),
    ]


DEMOS = {"S05": demo_s05, "S06": demo_s06, "S07": demo_s07}
