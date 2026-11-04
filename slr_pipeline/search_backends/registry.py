"""slr_pipeline.search_backends.registry -- the real adapter classes that
feed `nodes/search.py`'s own `_ADAPTER_REGISTRY` (Track 2 plan §2.5; the
eighth key, `scopus`, was added by the Track 3 launcher-wiring plan's
Unit C, `squishy-petting-starlight.md`).

This module supplies only `REAL_ADAPTER_CLASSES`. `nodes/search.py` is the
actual owner of `_ADAPTER_REGISTRY` (plan §2.5:
"search_node resolves a key ... `_ADAPTER_REGISTRY`"), which combines this
table with the seven `fx_*` fixture-adapter keys from
`replay/fixture_search_adapter.py::FIXTURE_ADAPTER_CLASSES` (step 10).
Module-level, no side effects: every value here is a class (not an
instance), so importing this module never opens a network connection or
reads an environment variable."""
from __future__ import annotations

from typing import Dict, Type

from slr_pipeline.search_backends.arxiv import ArxivBackend
from slr_pipeline.search_backends.crossref import CrossrefBackend
from slr_pipeline.search_backends.ieee import IeeeBackend
from slr_pipeline.search_backends.openalex import OpenAlexBackend
from slr_pipeline.search_backends.pubmed import PubMedBackend
from slr_pipeline.search_backends.scopus import ScopusBackend
from slr_pipeline.search_backends.semantic_scholar import SemanticScholarBackend
from slr_pipeline.search_backends.springer import SpringerBackend

REAL_ADAPTER_CLASSES: Dict[str, Type] = {
    "openalex": OpenAlexBackend,
    "pubmed": PubMedBackend,
    "semantic_scholar": SemanticScholarBackend,
    "arxiv": ArxivBackend,
    "crossref": CrossrefBackend,
    "springer": SpringerBackend,
    "ieee": IeeeBackend,
    "scopus": ScopusBackend,
}
# NEVER add "wos"/"web_of_science" here -- this omission is the primary
# structural WoS-impossibility guarantee for the Track 3 MVP (plan Unit C,
# §2.2): `nodes/search.py::_resolve_adapter_keys()` fail-closed raises
# `SearchConfigError` for any db key not in `_ADAPTER_REGISTRY` or a
# matching `execution_adapter_map` entry.
