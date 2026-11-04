"""
doi.py -- the shared DOI/title normalisation helpers (Track 2 plan §3.3).

`normalise_doi` is the ONE owner of DOI-resolver-URL stripping, replacing
four independent copies (`dedup.py`, `fulltext.py`, `evaluation/evaluator.py`,
`evaluation/gold_standard.py`) that each used `s.lstrip("https://doi.org/")`.
`str.lstrip(chars)` strips a leading CHARACTER SET, not a prefix --
`"http://dx.doi.org/10.1000/x".lstrip("https://doi.org/")` does not remove
the substring `"http://dx.doi.org/"`; it strips any leading run of the
characters `{h, t, p, s, :, /, d, o, i, .}` instead, silently mangling the
string into `"x.doi.org/10.1000/x"` rather than raising or no-op-ing. All
four call sites now use `str.removeprefix`, which has none of this
character-set ambiguity.

`normalise_title` is the sibling helper for `dedup.py`'s title-match stage
(§3.1) -- exact-equality comparison after casefolding, NFKC normalisation,
and whitespace collapsing, deliberately not a fuzzy/`difflib` comparison.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://dx.doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
)


def normalise_doi(doi: Optional[str]) -> Optional[str]:
    """Case-folds a DOI and strips any of the four common resolver-URL
    wrappers. Returns `None` for `None` (never raises on a missing DOI --
    callers that need a bare-string DOI already guard with `if r.doi`)."""
    if doi is None:
        return None
    normalised = doi.casefold().strip()
    for prefix in _DOI_URL_PREFIXES:
        normalised = normalised.removeprefix(prefix)
    return normalised


def normalise_title(title: Optional[str]) -> Optional[str]:
    """Exact-equality title normalisation: casefold, NFKC-normalise,
    collapse internal whitespace to one space, strip. Two titles are
    either identical after this or they are not -- no fuzzy matching."""
    if title is None:
        return None
    folded = unicodedata.normalize("NFKC", title.casefold())
    return re.sub(r"\s+", " ", folded).strip()
