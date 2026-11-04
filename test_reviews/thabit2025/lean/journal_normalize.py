"""
Lean prototype: cheap journal-name normalization to collapse WoS/Scopus
title-variant duplicates (e.g. "J. of Public Admin. Research & Theory" vs
"Journal of Public Administration Research and Theory") before/after LLM
classification, so the same journal isn't scored twice under different
strings.

No LLM calls -- pure string normalization. Import from journal_filter.py
and journal_reclassify.py.
"""
from __future__ import annotations

import re

ABBREV = {
    "j": "journal", "jour": "journal",
    "int": "international", "intl": "international",
    "admin": "administration", "adm": "administration",
    "rev": "review",
    "mgmt": "management", "mgt": "management",
    "res": "research",
    "dev": "development",
    "govt": "government", "gov": "government",
    "soc": "society", "socy": "society",
    "econ": "economic", "econ.": "economic",
    "pol": "political", "polit": "political",
    "assoc": "association", "assn": "association",
    "q": "quarterly", "qtly": "quarterly",
    "natl": "national", "nat": "national",
    "amer": "american",
    "sci": "science",
    "publ": "public",
    "pub": "public",
    "org": "organization", "organisation": "organization",
    "org.": "organization",
    "comp": "comparative",
    "affairs": "affairs",
    "std": "studies", "stud": "studies",
}

STOPWORDS = {"the", "of", "and", "for", "in", "on", "a", "an"}


def normalise_journal(name: str) -> str:
    """Lowercase, strip punctuation, expand common scholarly abbreviations,
    drop stopwords -- collapses most WoS/Scopus title-variant duplicates
    into the same key. Not fuzzy matching -- pure deterministic rewrite."""
    if not name:
        return ""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = s.split()
    expanded = [ABBREV.get(t, t) for t in tokens]
    kept = [t for t in expanded if t not in STOPWORDS]
    return " ".join(kept)
