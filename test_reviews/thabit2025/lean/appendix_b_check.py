"""
Lean prototype: check whether the paper's own 59 final-included studies
(Appendix B) survive our pipeline's journal-filter ("yes") group, versus
only making it into deduplicated_records.json or being missing entirely
from raw_records.json.

Fuzzy title matching (token-set overlap, not exact) since Appendix B
titles were manually retyped from the PDF and will differ in punctuation
from the DB-native title strings.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent

# Appendix B, 59 references (title only, manually transcribed from the PDF).
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


def tokens(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return set(w for w in s.split() if len(w) > 2)


def best_match(target: str, pool: list) -> tuple:
    """Returns (best_record_or_None, score) via Jaccard token overlap."""
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


def main() -> None:
    raw = json.loads((HERE / "raw_records.json").read_text(encoding="utf-8"))
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    journal_yes = json.loads((HERE / "journal_filtered_records.json").read_text(encoding="utf-8"))

    dedup_ids = {r["id"] for r in dedup}
    yes_ids = {r["id"] for r in journal_yes}

    THRESHOLD = 0.5
    in_raw = in_dedup = in_yes = not_found = 0
    rows = []
    for title in APPENDIX_B:
        rec, score = best_match(title, raw)
        if rec is None or score < THRESHOLD:
            not_found += 1
            rows.append(("NOT FOUND", f"{score:.2f}" if rec else "-", title[:70], rec.get("title", "")[:70] if rec else ""))
            continue
        in_raw += 1
        stage = "raw only"
        if rec["id"] in dedup_ids:
            in_dedup += 1
            stage = "dedup, journal=NO"
            if rec["id"] in yes_ids:
                in_yes += 1
                stage = "journal=YES"
        rows.append((stage, f"{score:.2f}", title[:70], rec.get("journal", "")))

    (HERE / "appendix_b_check_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"of 59: matched_in_raw={in_raw} (of which dedup={in_dedup}, journal_filter_YES={in_yes}) not_found={not_found}")
    print("-> appendix_b_check_results.json")


if __name__ == "__main__":
    main()
