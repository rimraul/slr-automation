"""
Lean prototype: FT_SCREENING CriteriaSet for Thabit et al. 2025's full-text
eligibility step (see NOTES.md "Full-text screening criteria" section).

Thabit's own text (_thabit_full.txt line 128, confirmed the fullest of two
in-paper mentions -- Appendix A line 705 restates the same 125-article
exclusion but drops the third reason) gives three full-text exclusion
reasons: (1) nonempirical articles, (2) no information on "governance
practices, dynamics, actions, or outcomes described in the previous
section", (3) 22 articles focused on public value FAILURES rather than
successes.

Reason (2) points at the paper's own THEORETICAL BACKDROP + Table 2 ("Data
structure: themes, components, key concepts", _thabit_full.txt line 176) --
that table is the operationalization used below: each CriterionSpec's
eligibility_requirement embeds the theme/component/key-concept definitions
and worked examples directly in its text, since FT_SCREENING's adjudication
prompt (ft_screening/adjudication.py::_build_prompt) sends
eligibility_requirement to the LLM as the ENTIRE criterion -- there is no
separate schema field for definitions/examples.

CriteriaSet is intentionally hand-authored per project convention (see
slr_pipeline/adapters/ft_screening_adapter.py's module docstring: never
derived from Protocol.inclusion_criteria/exclusion_criteria).

missing_evidence_policy="review" on all three -- full-text screening favors
recall (see [[project_chapter_outline]]'s Cochrane/JBI rationale); an
unclear/not_reported adjudication should go to a human, not auto-exclude.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"c:/projects")
from ft_screening import CriteriaSet, CriterionSpec

GOVERNANCE_TAXONOMY = """\
GOVERNANCE PRACTICES, DYNAMICS, ACTIONS (how public value is pursued):
1. Engagement and authorization -- meaningful participation and engagement \
(spaces/channels for engagement, empowerment of stakeholders, number/type of \
stakeholders involved); dialogue and negotiation (deliberation, decision-making \
processes among partners); shared value proposition (joint value definitions, \
coping with value tensions).
2. Operational capabilities and assets -- individual and collective assets/resources \
(skills, knowledge, funding, capacity mixes); integrative leadership and strategic \
management (leadership capabilities, process management skills); institutional and \
collaboration design (flexibility/autonomy levels, hierarchy/centrality, formality \
of the collaboration structure).
3. Broader environment and public sphere -- the broader political, social, or \
institutional context (administrative architecture, legislative/policy context, \
historical/cultural context) shaping the collaboration.

PUBLIC VALUE OUTCOMES (what public value is generated):
4. Instrumental public values (process-oriented) -- procedural legitimacy \
(procedural rationality, procedural justice, operational control) or relational \
value (trust, social networks/social capital, social cohesion).
5. Instrumental-and-prime public values -- democratic accountability (vertical/\
horizontal accountability, equal representation and inclusion) or equity and \
reciprocity (equity of benefits, equity of payment).
6. Prime public values (output-oriented) -- performance value (effectiveness, \
efficiency, durability) or problem-solving capacity (behavioral shifts, innovation, \
skills development/learning, increased knowledge/awareness, data/information gains).
"""


def thabit_criteria_set() -> CriteriaSet:
    return CriteriaSet(
        criteria_set_id="thabit2025-fulltext-v1",
        version="1",
        criteria=[
            CriterionSpec(
                criterion_id="empirical_evidence",
                version="1",
                eligibility_requirement=(
                    "The article reports ORIGINAL EMPIRICAL EVIDENCE (qualitative, "
                    "quantitative, or mixed-methods data collected and analyzed by the "
                    "authors -- e.g. case studies, interviews, surveys, document/content "
                    "analysis of a real collaboration) about one or more real-world "
                    "multi-actor collaborations for public value creation. A purely "
                    "theoretical, conceptual, normative, or literature-review article -- "
                    "one that develops or discusses ideas about public value governance "
                    "without reporting original data from a real collaboration -- does "
                    "NOT satisfy this criterion, even if it cites empirical studies by "
                    "other authors.\n"
                    "Example SATISFYING: a case study of a living-lab co-creation "
                    "initiative analyzing interview data from participants.\n"
                    "Example NOT satisfying: a conceptual paper proposing a new typology "
                    "of public value with no empirical data of its own."
                ),
                missing_evidence_policy="review",
                exclusion_reason_code="nonempirical",
                exclusion_reason_label="Nonempirical article",
                priority=1,
            ),
            CriterionSpec(
                criterion_id="governance_practice_or_outcome_info",
                version="1",
                eligibility_requirement=(
                    "The article reports information on at least ONE of the following "
                    "governance practices/dynamics/actions OR public value outcomes, as "
                    "operationalized by this review's own theoretical framework (Table 2, "
                    "'Data structure: themes, components, key concepts'):\n\n"
                    + GOVERNANCE_TAXONOMY
                    + "\nAn article satisfies this criterion if it provides concrete "
                    "information on at least one component above (1-6), in relation to a "
                    "multi-actor collaboration -- not just a bare mention of the theme "
                    "name. An article that discusses multi-actor collaboration only in "
                    "general terms, without describing any specific governance practice/"
                    "dynamic/action (1-3) or any specific public value outcome (4-6), does "
                    "NOT satisfy this criterion."
                ),
                missing_evidence_policy="review",
                exclusion_reason_code="no_governance_practice_or_outcome_info",
                exclusion_reason_label="No governance practice, dynamic, action, or outcome information",
                priority=2,
            ),
            CriterionSpec(
                criterion_id="not_failure_focused",
                version="1",
                eligibility_requirement=(
                    "The article's focus is on public value(s) SUCCESSFULLY generated "
                    "through a multi-actor collaboration, and/or the governance practices "
                    "that led to their achievement. Articles whose PRIMARY focus is on "
                    "PUBLIC VALUE FAILURES -- cases where collaboration failed to generate "
                    "public value, value destruction, or the mechanisms behind "
                    "unsuccessful/negative outcomes -- do NOT satisfy this criterion, since "
                    "this review's research questions are restricted to types of public "
                    "value(s) successfully generated and the practices that led to their "
                    "achievement.\n"
                    "Example NOT satisfying: a case study primarily analyzing why a "
                    "public-private partnership failed to deliver value, or documenting "
                    "institutional value destruction.\n"
                    "Example SATISFYING: a case study that discusses challenges alongside "
                    "successful outcomes, where the article's primary focus/contribution "
                    "is the successfully generated value."
                ),
                missing_evidence_policy="review",
                exclusion_reason_code="public_value_failure_focus",
                exclusion_reason_label="Focus on public value failures",
                priority=3,
            ),
        ],
    )


if __name__ == "__main__":
    cs = thabit_criteria_set()
    print(f"OK  {cs.criteria_set_id} v{cs.version}  {len(cs.criteria)} criteria")
    for c in cs.criteria:
        print(f"  [{c.priority}] {c.criterion_id}  ({len(c.eligibility_requirement)} chars)  "
              f"-> {c.exclusion_reason_code}")
