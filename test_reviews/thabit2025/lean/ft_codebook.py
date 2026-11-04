"""
Thabit et al. 2025's own deductive codebook (Table 2, "Data structure: Overview
of themes, components, and key concepts") plus the gold-standard concept-to-
article mapping from Appendix C (Table C1, "Information sources by components
and key concepts") -- transcribed directly from `_thabit_full.txt` (Docling
extraction of the published PDF), not paraphrased or reconstructed.

CODEBOOK is the 37-row (theme, component, concept) hierarchy in the paper's
own Table 2 order. GOLD_REFS_RAW is the corresponding Table C1 raw reference
list per concept (R1..R59 = Appendix B's numbered list of 59 included
studies); parsed into sets of ref strings by `gold_refs()` below. One entry
(Equal representation and inclusion) has a bare "30" in the source table with
the "R" prefix dropped -- fixed here, noted inline, not silently "corrected"
without a trace.

This is what the review authors say they actually coded for (see
NOTES.md's planned "extraction methodology" section) -- our LLM-based
deductive extraction (`run_deductive_extraction.py`) uses this SAME codebook,
concept-for-concept, so its output is directly comparable to Table C1's own
per-article ground truth.
"""
from __future__ import annotations

import re

# (theme, component, concept) -- Table 2's own order, 37 rows.
CODEBOOK: list[tuple[str, str, str]] = [
    ("Engagement and authorization", "Meaningful participation and engagement", "Spaces and channels for engagement"),
    ("Engagement and authorization", "Meaningful participation and engagement", "Empowerment of actors and community groups"),
    ("Engagement and authorization", "Meaningful participation and engagement", "Number and type of stakeholders"),
    ("Engagement and authorization", "Dialogue and negotiation", "Dialogue and deliberation"),
    ("Engagement and authorization", "Dialogue and negotiation", "Negotiation and decision making"),
    ("Engagement and authorization", "Shared value proposition", "Joint value definitions"),
    ("Engagement and authorization", "Shared value proposition", "Value tensions and coping strategies"),
    ("Operational capabilities and assets", "Individual and collective assets and resources", "Individual assets and resources"),
    ("Operational capabilities and assets", "Individual and collective assets and resources", "Collective assets and resources"),
    ("Operational capabilities and assets", "Individual and collective assets and resources", "Capacity mixes"),
    ("Operational capabilities and assets", "Integrative leadership and strategic management", "Integrative leadership capabilities"),
    ("Operational capabilities and assets", "Integrative leadership and strategic management", "Strategic management capabilities"),
    ("Operational capabilities and assets", "Institutional and collaboration design", "Flexibility and autonomy levels"),
    ("Operational capabilities and assets", "Institutional and collaboration design", "Hierarchy and (de)centrality levels"),
    ("Operational capabilities and assets", "Institutional and collaboration design", "Formality levels"),
    ("Broader environment and public sphere", "Broader political and social context", "Broader administrative architecture"),
    ("Broader environment and public sphere", "Broader political and social context", "Legislative and policy context"),
    ("Broader environment and public sphere", "Broader political and social context", "Historical and cultural context"),
    ("Instrumental public values (process-oriented)", "Procedural legitimacy", "Procedural rationality"),
    ("Instrumental public values (process-oriented)", "Procedural legitimacy", "Procedural justice"),
    ("Instrumental public values (process-oriented)", "Procedural legitimacy", "Operational control"),
    ("Instrumental public values (process-oriented)", "Relational value", "Trust"),
    ("Instrumental public values (process-oriented)", "Relational value", "Social networks and social capital"),
    ("Instrumental public values (process-oriented)", "Relational value", "Social cohesion"),
    ("Instrumental and prime public values", "Democratic accountability", "Vertical democratic accountability"),
    ("Instrumental and prime public values", "Democratic accountability", "Horizontal democratic accountability"),
    ("Instrumental and prime public values", "Democratic accountability", "Equal representation and inclusion"),
    ("Instrumental and prime public values", "Equity and reciprocity", "Equity of benefits"),
    ("Instrumental and prime public values", "Equity and reciprocity", "Equity of payment"),
    ("Prime public values (output-oriented)", "Performance value", "Efficient performance"),
    ("Prime public values (output-oriented)", "Performance value", "Effective performance"),
    ("Prime public values (output-oriented)", "Performance value", "Durable solutions"),
    ("Prime public values (output-oriented)", "Problem-solving capacity", "Behavioral shifts"),
    ("Prime public values (output-oriented)", "Problem-solving capacity", "Innovation"),
    ("Prime public values (output-oriented)", "Problem-solving capacity", "Skills development and learning"),
    ("Prime public values (output-oriented)", "Problem-solving capacity", "Increased knowledge and awareness"),
    ("Prime public values (output-oriented)", "Problem-solving capacity", "Data and information"),
]

assert len(CODEBOOK) == 37, f"expected 37 key concepts per Table 2, got {len(CODEBOOK)}"

# Table C1's raw "# References (Appendix B)" cell per concept, in the SAME
# order as CODEBOOK above. Transcribed verbatim except one fix, noted below.
GOLD_REFS_RAW: list[str] = [
    "R1, R2, R4, R5, R8, R9, R11, R12, R13, R14, R17, R18, R19, R20, R22, R24, R25, R29, R31, R33, R34, R35, R37, R38, R39, R40, R41, R42, R43, R44, R45, R47, R48, R49, R50, R51, R54, R56, R59",
    "R6, R9, R10, R14, R18, R19, R20, R22, R24, R28, R29, R33, R37, R38, R45, R49, R50, R51, R56",
    "R9, R16, R18, R31, R45, R51",
    "R1, R2, R3, R5, R7, R8, R9, R10, R12, R13, R14, R15, R16, R17, R18, R23, R24, R26, R29, R29, R30, R31, R32, R33, R35, R36, R37, R38, R39, R40, R43, R45, R48, R49, R50, R51, R54, R56",
    "R2, R9, R11, R14, R16, R18, R19, R20, R21, R26, R27, R29, R29, R30, R35, R40, R41, R43, R45, R46, R47, R48, R53, R57",
    "R2, R7, R8, R9, R10, R11, R12, R13, R14, R16, R19, R21, R22, R23, R24, R27, R28, R29, R29, R31, R33, R34, R35, R37, R38, R40, R41, R42, R43, R46, R47, R48, R49, R51, R52, R53, R54, R55",
    "R3, R8, R10, R11, R14, R22, R23, R35, R41, R56, R57",
    "R1, R6, R9, R12, R18, R29, R33, R43, R47, R49, R57",
    "R6, R8, R10, R11, R20, R26, R32, R33, R34, R35, R38, R44, R46, R47, R48, R49, R52, R54, R55",
    "R4, R11, R14, R23, R29, R33, R34, R44, R54, R55",
    "R6, R7, R8, R10, R11, R16, R20, R29, R30, R33, R35, R37, R38, R40, R43, R46, R49, R50, R51, R54, R56, R57",
    "R8, R10, R11, R31, R35, R38, R40, R43, R49, R50, R51, R55",
    "R1, R3, R6, R9, R10, R11, R14, R15, R18, R19, R20, R21, R22, R26, R28, R29, R29, R30, R32, R33, R34, R35, R38, R40, R41, R43, R44, R46, R47, R48, R50, R51, R52, R53, R55, R56, R57, R59",
    "R10, R11, R17, R27, R30, R31, R37, R47, R49, R50, R54",
    "R6, R11, R13, R15, R16, R28, R33, R34, R46, R47, R50, R54, R59",
    "R6, R9, R15, R28, R32, R35, R38, R50, R55, R57",
    "R9, R10, R15, R16, R17, R21, R27, R28, R31, R38, R39, R46, R50, R51, R54, R55, R57",
    "R3, R7, R9, R10, R28, R29, R38, R50, R57",
    "R1, R2, R3, R6, R7, R8, R12, R13, R14, R16, R19, R20, R23, R26, R28, R29, R30, R32, R34, R35, R38, R39, R41, R42, R43, R44, R45, R46, R48, R57, R59",
    "R1, R2, R7, R8, R9, R10, R13, R15, R16, R19, R20, R21, R23, R26, R29, R32, R37, R39, R40, R43, R45, R47, R48, R52, R53, R56, R59",
    "R1, R2, R3, R6, R11, R13, R14, R16, R17, R18, R26, R27, R28, R29, R30, R33, R34, R35, R37, R38, R39, R41, R42, R43, R44, R47, R49, R51, R53, R54, R57",
    "R1, R6, R7, R8, R9, R10, R13, R14, R16, R17, R18, R20, R21, R23, R24, R26, R28, R29, R29, R33, R34, R35, R37, R45, R46, R48, R49, R50, R52, R54, R56, R57",
    "R5, R6, R10, R14, R18, R20, R22, R23, R24, R25, R29, R35, R44, R45, R48, R50, R54, R55, R56",
    "R4, R5, R14, R20, R23, R29, R31, R33, R40, R43, R49, R50, R56",
    "R1, R2, R8, R15, R16, R20, R23, R26, R29, R30, R35, R39, R40, R41, R44, R47, R49, R51, R57",
    "R1, R3, R4, R7, R8, R11, R12, R13, R15, R16, R18, R19, R20, R22, R24, R26, R27, R29, R29, R32, R33, R34, R35, R37, R38, R39, R40, R41, R42, R43, R45, R48, R50, R51, R55",
    # source table has a bare "30" (missing "R") as the first token here -- fixed to "R30".
    "R30, R10, R11, R13, R14, R16, R19, R22, R26, R29, R29, R37, R40, R41, R45, R46, R48, R49, R56",
    "R1, R5, R8, R10, R12, R17, R20, R21, R22, R23, R27, R28, R29, R29, R30, R31, R33, R35, R37, R38, R41, R43, R44, R45, R46, R50, R51, R52, R53, R54",
    "R1, R6, R7, R8, R9, R15, R16, R17, R18, R21, R22, R23, R26, R27, R29, R31, R33, R34, R35, R41, R43, R46, R47, R49, R51, R54, R55, R59",
    "R6, R8, R10, R17, R20, R27, R30, R31, R32, R33, R39, R42, R43, R46, R47, R50, R51, R52, R59",
    "R1, R3, R4, R6, R7, R8, R9, R10, R11, R12, R14, R15, R18, R20, R21, R22, R23, R24, R26, R27, R28, R29, R29, R30, R31, R33, R35, R36, R37, R38, R39, R40, R42, R43, R46, R47, R49, R50, R51, R52, R53, R56, R57, R59",
    "R1, R6, R10, R11, R14, R18, R20, R24, R28, R29, R32, R38, R43, R50",
    "R6, R10, R17, R20, R25, R31, R35, R38, R39, R43, R44, R46, R48, R49, R50, R51, R55",
    "R5, R7, R11, R14, R15, R17, R18, R19, R20, R27, R28, R31, R32, R33, R34, R42, R47, R49, R50, R51, R54, R56, R57",
    "R1, R3, R4, R10, R15, R18, R19, R20, R25, R29, R31, R33, R34, R39, R44, R49, R50, R55, R56, R57",
    "R1, R4, R8, R15, R17, R19, R20, R26, R28, R29, R30, R31, R32, R34, R35, R45, R48, R49, R50, R51, R52, R54, R56, R57",
    "R20, R27, R28, R32, R39, R44, R47, R48, R51, R52, R54, R57",
]

assert len(GOLD_REFS_RAW) == 37

_REF_RE = re.compile(r"R\d+")


def gold_refs() -> dict[str, set[str]]:
    """concept name -> set of Appendix B ref IDs ("R1".."R59") Table C1
    credits with that concept. Deduplicates the source table's own repeated
    refs within a cell (e.g. "R29, R29") via set()."""
    return {
        CODEBOOK[i][2]: set(_REF_RE.findall(GOLD_REFS_RAW[i]))
        for i in range(len(CODEBOOK))
    }


# Our 10-record known_included pilot sample, matched by exact title against
# Appendix B (`_thabit_full.txt` lines 734-880) -- manual one-time lookup,
# not automatable (Appendix B has no DOIs).
REC_ID_TO_REF: dict[str, str] = {
    "wos-212c4fe1e566d8b5a503": "R1",   # Agusdinata -- universities/SDGs co-creation
    "wos-2dbac6562e9fa94dcbe7": "R8",   # Best, Moffett & McAdam -- stakeholder salience
    "wos-15de79f98e1e4757cf96": "R12",  # Bulmer & del Prado-Higuera -- SDG 17 alliances
    "wos-93a6e9f90f555e17a764": "R20",  # Haug & Mergel -- living labs
    "wos-92b0efcd6c4d07e481a1": "R22",  # Jaspers & Steen -- realizing public values
    "wos-a799cafa17fa9259a377": "R47",  # Scupola & Mergel -- co-production Denmark
    "wos-a159f93cd244636d67d7": "R45",  # Peso, Gonzalez & Milosevic -- co-creative participatory methods
    "wos-b1db8d1123755a488b64": "R39",  # Onyoin & Bovis -- sustaining PPPs
    "wos-1e158b40efa0486eb5ef": "R57",  # Waardenburg et al. -- paradoxes of collaborative governance
    "wos-553431b20bbbf9b271a1": "R43",  # Parker, Hartley, Beashel & Vo -- leading for public value
}

assert len(REC_ID_TO_REF) == 10
assert len(set(REC_ID_TO_REF.values())) == 10, "ref IDs must be unique"
