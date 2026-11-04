"""
Lean prototype: abstract-only title/abstract screening test (deliberate
deviation from Thabit et al. 2025, who screened on titles+abstracts+
keywords -- see NOTES.md).

Reuses the same include/borderline/exclude batch-LLM pattern as
slr_pipeline/nodes/screening.py's BATCH_PROMPT_TEMPLATE, but abstract-only
(no title shown to the model) and against Thabit's own Appendix A criteria
(reconstructed verbatim from the PDF below).

Two-part test, since no published excluded-title list exists in the paper
(only Appendix B = 59 included; Appendix C = framework categorization,
not an excluded-examples table):
  1. Known included: the 51 Appendix B titles that matched our corpus
     (appendix_b_check.py's fuzzy match), re-derived here with record ids.
  2. Known excluded (proxy): a random sample of journal-filter "no" records
     -- off-topic by construction, not paper-confirmed exclusions, but the
     closest available negative control.

count=small, gpt-5.4-mini, temperature=0.0 -- live paid calls.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent

MODEL = "gpt-5.4-mini"

# Richness level: "R0" = naive, verbatim Appendix A criteria + checklist
# structuring only, no added guidance/examples. "R1" (default) = R0 plus
# GUIDANCE_BLOCK (generic vocab-masking guidance + invented contrastive
# examples, no known-label access). "R2" = R1 plus EXTRA_RULES_R2 below,
# mined by open-coding actual weak-vs-strong-vs-true-label disagreements on
# known_included/known_excluded (oracle/ceiling test -- see NOTES.md "R2
# ceiling analysis": this is a deliberate, disclosed use of the known-label
# pool to test whether the original authors' decisions are recoverable at
# all with maximal prompt tuning, not a held-out generalization estimate).
RICHNESS = "R1"

EXTRA_RULES_R2 = """
Additional rules (mined from disagreement analysis against this review's
own known included/excluded studies):
- field_fit is about ANALYTIC LENS, not topical domain: a paper studying
  SDGs, sustainability, digital government, health, or another substantive
  domain still satisfies field_fit=yes if its actual analytic focus is
  governance/management processes (multi-actor coordination, partnership
  structures, institutional roles) -- do not mark field_fit=no just because
  the domain label is SDGs/environment/digital/health rather than the
  words "public administration" appearing verbatim.
- mechanism_specificity should be judged generously for short abstracts:
  if the abstract names a concrete case, program, or setting (e.g. "digital
  co-production in Denmark", "neighbourhood governance during COVID-19
  lockdown") that is enough to satisfy mechanism_specificity=yes, even if
  it does not spell out exactly who collaborates with whom in procedural
  detail -- abstracts are short by construction and omit detail that full
  text would supply. Reserve "no"/"unclear" for abstracts that stay at a
  purely generic/theoretical level with no named case or setting at all.
- field_fit=no is still correct when the paper's substantive empirical
  domain is primarily public health service delivery, sanitation/WASH,
  museums/cultural institutions, or individual-level welfare/social-work
  casework, and public administration is only a peripheral framing device
  -- "public value" or "co-production" vocabulary alone does not override
  this.
"""

# Reconstructed verbatim from Appendix A (PDF page 17, column-region
# extraction) -- wording cleaned of PDF word-join artifacts, content
# unchanged.
INCLUSION_CRITERIA = [
    "Articles that offer empirical evidence around collaboration practices for public value creation.",
    "Articles that focus on any type of cross-sector or same-sector collaboration (e.g., formal partnerships, informal collaborations, small or large number of partners) and their contribution to public value creation.",
    "Articles that focus on value creation processes among multiple actors (e.g., how democratic value was enacted, value controversies) with specific mention to collaboration mechanisms.",
]

EXCLUSION_CRITERIA = [
    "Books, book chapters, conference proceedings, theses, and articles not written in English.",
    "Articles published in journals not related to the public administration field.",
    "Articles whose focus is not on specific collaborative mechanisms for public value creation, in action or observed within its configuration -- e.g., articles that focus on general topics around collaboration without describing specific mechanisms for public value, or that focus on general approaches to value creation without specific mention of collaboration mechanisms.",
    "Theoretical articles that do not offer empirical evidence about collaboration mechanisms for public value creation.",
]

BATCH_SYSTEM = (
    "You are a systematic review screening assistant. "
    "Return only valid JSON -- an object with key \"items\", one entry per paper, in the same order as the input."
)

GUIDANCE_BLOCK = """
Important -- surface vocabulary is not sufficient for inclusion:
The presence of words like "co-creation", "collaboration", "collaborative
governance", "partnership", or "public value" does NOT by itself satisfy the
inclusion criteria. Before deciding "include", check BOTH:
(a) Field fit: the paper's own disciplinary orientation must plausibly be
    public administration / public management / public policy -- not
    primarily public health, marketing/branding, urban or environmental
    planning, education, or another field that only touches collaboration
    in passing (exclusion criterion 2).
(b) Mechanism specificity: the abstract must describe a SPECIFIC
    collaboration mechanism in action (who collaborates, how, in what
    configuration) contributing to public value creation -- not just a
    general topic mention of collaboration or value creation without that
    detail (exclusion criterion 3).
If (a) or (b) is genuinely unclear from the abstract alone, decide
"borderline", not a confident "include" or "exclude".

Example -- EXCLUDE despite matching vocabulary: an abstract about a city's
social-media brand community using "collaborative practices" to build
"collective value" is about branding/marketing, not public-administration
collaboration for public value -- exclude, field fit fails.
Example -- BORDERLINE despite matching vocabulary: an abstract about
university-led multi-stakeholder co-creation of SDG solutions, with no
public-administration framing stated -- borderline, field fit unclear.
"""

BATCH_PROMPT_TEMPLATE = """\
Screen the following {n} papers for inclusion in a systematic review, using ONLY the abstract (no title is provided).

Inclusion criteria:
{inc}

Exclusion criteria:
{exc}
{guidance}
Before deciding, answer each checklist item for "yes"/"no"/"unclear" based
ONLY on what the abstract states -- do not let topical vocabulary alone
answer "yes":
- field_fit: is the paper's own disciplinary orientation public
  administration / public management / public policy (not public health,
  marketing/branding, urban/environmental planning, education, or another
  field that only touches collaboration in passing)?
- mechanism_specificity: does the abstract describe a SPECIFIC
  collaboration mechanism in action (who collaborates, how, in what
  configuration) contributing to public value creation?
- empirical_evidence: does the abstract report empirical evidence (not a
  purely theoretical/conceptual argument)?

Then derive the decision from the checklist: "include" only if all three
are "yes"; "exclude" if any is "no"; "borderline" if any is "unclear" and
none is "no".
{extra_rules}
Papers:
{papers}

Return a JSON object with key "items" containing exactly {n} objects, one per paper, in input order:
{{
  "items": [
    {{"idx": <1-based integer>,
      "checklist": {{"field_fit": "yes"|"no"|"unclear", "mechanism_specificity": "yes"|"no"|"unclear", "empirical_evidence": "yes"|"no"|"unclear"}},
      "decision": "include" | "borderline" | "exclude",
      "reason": "<one sentence>"}},
    ...
  ]
}}
"""


def tokens(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return set(w for w in s.split() if len(w) > 2)


APPENDIX_B = [
    "The Role of Universities in SDGs Solution Co-creation and Implementation: A Human-Centered Design and Shared-Action Learning Process",
    "A Decision Theory Perspective on Wicked Problems, SDGs and Stakeholders: The Case of Deforestation",
    "Value Conflicts in Co-production: Governing Public Values in Multi-Actor Settings",
    "Co-Designing a Citizen Science Program for Malaria Control in Rwanda",
    "Brand Innovation and Value-Creating Practices of the City of Johannesburg's Twitter Community",
    "Partnerships and the Good-Governance Agenda: Improving Service Delivery Through State-NGO Collaborations",
    "Strengthening Public Service Production, Administrative Problem Solving, and Political Leadership Through Co-creation of Innovative Public Value Outcomes",
    "Stakeholder Salience in Public Sector Value co-creation",
    "Better Public Sector Governance Through Partnership With The Private Sector and Civil Society: The Case of Guatemala's Forest Administration",
    "Mobilizing Change in Public Services: Insights from a Systems Leadership Development Intervention",
    "New Community Spaces: Regional Governance in the Public Interest in the Greater Toronto Area",
    "Revitalizing the Global Alliances for Sustainable Development: Analyzing the Viability of Sustainable Development Goal 17-A Multi-Actor Governance Approach",
    "How does digital technology impact on the co-production of local services? Evidence from a childcare experience",
    "Public Values and Public Participation: A Case of Collaborative Governance of a Planning Process",
    "Boundary-spanning in Public Value Co-creation Through The Lens Of Multilevel Governance",
    "Having it all: can Collaborative Governance be Both Legitimate and Accountable",
    "Coproduction in the U.S. Department of Defense: Examining How the Evolution of Geographic Information Systems (GIS) Expands Non-Traditional Partner Engagement",
    "The Role of Soft Skills to Leverage Co-creation in Living Labs: Insights from Spain",
    "Living Labs as an Innovation Tool for Public Value Creation: Possibilities and Pitfalls",
    "Public Value Co-Creation in Living Labs-Results from Three Case Studies",
    "Crowdfunding Public Projects: Collaborative Governance for Achieving Citizen Co-funding of Public Goods",
    "Realizing Public Values: Enhancement or Obstruction? Exploring Value Tensions and Coping Strategies in the Co-Production of Social Care",
    "Does Co-Production Lead to the Creation of Public Value? Balancing the Dimensions of Public Value Creation in Urban Mobility Planning",
    "Organizational Maturity for Co-creation: Towards a Multi-attribute Decision Support Model for Public Organizations",
    "The Governance Games of Citizens and Stakeholders' Engagement: Longitudinal Narratives",
    "Knowledge Co-production in Contested Spaces: An Evaluation of the North Slope Borough-Shell Baseline Studies Program",
    "The Collaborative Realization of Public Values and Business Goals: Governance and Infrastructure of Public-Private Information Platforms",
    "Creating Value Through Data Collaboratives Balancing Innovation and Control",
    "Citizen Science in the Field: Co-experimentation at Pilot Scale for Sustainable Use of Natural Resources",
    "Institutional Design of Public Agencies and Coproduction: A Study of Irrigation Associations in Taiwan",
    "Achieving Sustainable Development Goals Through Collaborative Innovation: Evidence from Four European Initiatives",
    "How Does Open Government Data Driven Co-creation Occur? Six Factors and a Perfect Storm insights from Chicago's Food Inspection Forecasting Model",
    "Catalyzing Transformational Partnerships for the SDGs: Effectiveness and Impact of the Multi-Stakeholder Initiative El dia despues",
    "Feedback Loops and Facilitation: Catalyzing Transformational Multi-Stakeholder Refugee Response Partnerships",
    "Integrative public leadership: Catalyzing collaboration to create public value",
    "Behavioral Effects of Public Service Motivation Among Citizens: Testing the Case of Digital Co-Production",
    "Leading the Way: the Integral Role of Local Government within A Multisector Partnership Delivering a Large Infrastructure Project in an Australian Growth Region",
    "Strategic Management as an Enabler of Co-creation in Public Services",
    "Sustaining Public-Private Partnerships for Public Service Provision Through Democratically Accountable Practices",
    "Paradox and Collaboration in Network Management",
    "Ambidexterity in Cross-Sector Collaborations Involving Public Organizations",
    "Coping with Value Conflicts in Interorganizational Collaborations",
    "Leading for Public Value in Multi-agency Collaboration",
    "Local Responses to Global Sustainability Agendas: Learning from Experimenting with the Urban Sustainable Development Goal in Cape Town",
    "Innovative Co-Creative Participatory Methodologies for a Dreamt-of Quality Education in Europe",
    "Applying Elinor Ostrom's Design Principles to Guide Co-Design in Health(care) Improvement: A Case Study with Citizens Returning to the Community from Jail in Los Angeles County",
    "Co-production in digital transformation of public administration and public value creation: The case of Denmark",
    "Community Foundations as Network Conveners: Structuring Collective Agency for Child Education and Development System Impact",
    "Public Services Management and Co-Production in Multi-Level Governance Settings",
    "Co-creating Ambitious Climate Change Mitigation Goals: The Copenhagen Experience",
    "Co-initiation of Collaborative Innovation in Urban Spaces",
    "Establishing and Implementing Data Collaborations for Public Good: A Critical Factor Analysis to Scale Up the Practice",
    "Neoliberal Governing Through Social Enterprise: Exploring the Neglected Roles of Deviance and Ignorance in Public Value Creation",
    "Constructing a Joint University-Municipality Platform for Networked Co-creation of Innovative Urban Solutions: A Norwegian Experience",
    "Cross-Sector Partnerships for Sustainability: How Mission-Driven Conveners Drive Change in National Coffee Platforms",
    "The Co-production of Public Value in Community Development: Can Street-Level Professionals Make a Difference",
    "Paradoxes of Collaborative Governance: Investigating the Real-Life Dynamics of Multi-Agency Collaborations using a Quasi-Experimental Action-Research Approach",
    "Co-production, Governance and Practice: The Dynamics and Effects of User Voice Prison Councils",
    "Neighbourhood Governance During the COVID-19 Lockdown in Hangzhou: Coproduction Based on Digital Technologies",
]


def best_match(target: str, pool: list) -> tuple:
    t_tokens = tokens(target)
    best_rec, best_score = None, 0.0
    for rec in pool:
        r_tokens = tokens(rec.get("title") or "")
        if not r_tokens:
            continue
        inter = len(t_tokens & r_tokens)
        union = len(t_tokens | r_tokens)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score, best_rec = score, rec
    return best_rec, best_score


def build_known_included(dedup: list) -> list:
    out = []
    matched_ids = set()
    for title in APPENDIX_B:
        rec, score = best_match(title, dedup)
        if rec is not None and score >= 0.5 and (rec.get("abstract") or "").strip():
            out.append(rec)
            matched_ids.add(rec["id"])
    missing8 = json.loads((HERE / "missing8_records.json").read_text(encoding="utf-8"))
    out.extend(r for r in missing8 if r.get("abstract"))
    return out, matched_ids


def build_known_excluded(journal_yes: list, matched_ids: set, n: int, seed: int = 42) -> list:
    """Records that survived search+dedup+journal-filter (same pre-abstract-
    screening pool the paper's own 291 came from) but are NOT among the
    paper's 59 final-included studies -- i.e., candidates the original
    authors screened OUT somewhere downstream, not a random off-topic
    sample."""
    pool = [
        r for r in journal_yes
        if r["id"] not in matched_ids and (r.get("abstract") or "").strip()
    ]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def screen_batch(client: OpenAI, records: list) -> list:
    papers = "\n\n".join(
        f"[{i+1}] Abstract: {(r.get('abstract') or '').strip()[:900]}"
        for i, r in enumerate(records)
    )
    prompt = BATCH_PROMPT_TEMPLATE.format(
        n=len(records),
        inc="\n".join(f"- {c}" for c in INCLUSION_CRITERIA),
        exc="\n".join(f"- {c}" for c in EXCLUSION_CRITERIA),
        guidance=GUIDANCE_BLOCK if RICHNESS in ("R1", "R2") else "",
        extra_rules=EXTRA_RULES_R2 if RICHNESS == "R2" else "",
        papers=papers,
    )
    call_kwargs = dict(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": BATCH_SYSTEM}, {"role": "user", "content": prompt}],
    )
    if MODEL.startswith("gpt-5.6"):
        # reasoning-family model: no temperature param, use reasoning_effort
        # instead (same pairing as slr_pipeline's own gpt-5.6-terra callers).
        call_kwargs["reasoning_effort"] = "medium"
    else:
        call_kwargs["temperature"] = 0.0
    resp = client.chat.completions.create(**call_kwargs)
    raw = json.loads(resp.choices[0].message.content)
    items = raw.get("items") or []
    items_by_idx = {item["idx"]: item for item in items}
    return [items_by_idx[i] for i in range(1, len(records) + 1)]


def load_env(path: str) -> dict:
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def run_group(client: OpenAI, name: str, records: list, batch_size: int = 20) -> list:
    results = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        items = screen_batch(client, batch)
        for r, item in zip(batch, items):
            results.append({"title": r.get("title"), "journal": r.get("journal"), **item})
    counts = {"include": 0, "borderline": 0, "exclude": 0}
    for r in results:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(f"{name}: n={len(results)}  include={counts['include']} borderline={counts['borderline']} exclude={counts['exclude']}")
    return results


def main() -> None:
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    journal_yes = json.loads((HERE / "journal_filtered_records.json").read_text(encoding="utf-8"))

    known_included, matched_ids = build_known_included(dedup)
    known_excluded = build_known_excluded(journal_yes, matched_ids, n=len(known_included))
    print(f"known_included n={len(known_included)}  known_excluded n={len(known_excluded)}")

    env = load_env("../../../.env")
    import os
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    client = OpenAI()

    inc_results = run_group(client, "known_included (Appendix B, 59/59)", known_included)
    exc_results = run_group(client, "known_excluded (journal=yes, not in Appendix B)", known_excluded)

    (HERE / "screening_test_results.json").write_text(
        json.dumps({"known_included": inc_results, "known_excluded": exc_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("-> screening_test_results.json")


if __name__ == "__main__":
    main()
