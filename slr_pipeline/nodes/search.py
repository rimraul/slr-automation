"""
search_node — Step 1 of the SLR pipeline.

Dispatches one `SearchBackendRequest` per configured database to the
package-native adapters in `slr_pipeline/search_backends/*` (Track 2 plan
§2.4-§2.6) and assembles `DatabaseSearchLog` v2 entries (§2.8) from each
adapter's `SearchBackendResult`.

Replaces the earlier `sys.path.insert(0, "c:/projects/screening")`-based
wrappers: every adapter here is self-contained, HTTPS-only, and reads any
credential it needs from `os.environ` at call time (never from config,
never serialised into a checkpoint -- see `SearchConfig`'s deprecated
`pubmed_email`/`springer_api_key`/`ieee_api_key` fields, §2.7).

Node input keys:  config
Node output keys: raw_records, search_log, prisma_counts
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Type

from slr_pipeline.config import ReviewConfig, SearchConfig
from slr_pipeline.search_backends.base import (
    DATE_UNSUPPORTED,
    LANGUAGE_UNSUPPORTED,
    SearchCheckpointHooks,
    SearchBackendRequest,
    SearchBackendResult,
    SearchConfigError,
)
from slr_pipeline.replay.fixture_search_adapter import FIXTURE_ADAPTER_CLASSES
from slr_pipeline.search_backends.registry import REAL_ADAPTER_CLASSES
from slr_pipeline.state import DatabaseSearchLog, PRISMACounts, Record, ReviewState, SearchLog

logger = logging.getLogger(__name__)

# `_ADAPTER_REGISTRY` (plan §2.5): the eight real keys (Track 3 Unit C
# added `scopus`) plus the seven
# `fx_*` fixture keys (step 10, `replay/fixture_search_adapter.py`).
# Mapped dispatch against an `execution_adapter_map` pointing at any other
# `fx_*`-shaped key (e.g. `fx_openalex` -- not one of the seven canonical
# Berg database names) still correctly raises `SearchConfigError`.
_ADAPTER_REGISTRY: Dict[str, Type] = {**REAL_ADAPTER_CLASSES, **FIXTURE_ADAPTER_CLASSES}

# Per-adapter env var read for `SearchBackendRequest.mailto` (plan §2.6 --
# only openalex/crossref's "polite pool" support it; every other adapter's
# request.mailto is simply unused).
_MAILTO_ENV_VAR = {"openalex": "OPENALEX_MAILTO", "crossref": "CROSSREF_MAILTO"}

# Databases that understand PubMed field tag syntax ([Title], [MeSH], etc.)
_PUBMED_SYNTAX_DBS = {"pubmed"}

# Databases for which a missing/failed per-database translation must be a
# pre-network SearchConfigError, never a silent substitute -- neither the
# raw PubMed-syntax query (search_strategy_builder.py's own
# call_status="heuristic_fallback" copies the untranslated query into
# EVERY requested database, including this one) nor `_sanitize_query()`'s
# lossy quoted-phrase-OR rebuild (which discards Boolean AND structure
# entirely). Confirmed root cause of a real live-run producing an invalid
# 0-hit Scopus result: a failed LLM translation silently reached Scopus as
# verbatim PubMed field-tag syntax, which Scopus's API does not understand.
_STRICT_TRANSLATION_DBS = {"scopus"}

# PubMed field-tag syntax, e.g. `[tiab]`, `[MeSH]`, `[la]` -- a query bound
# for a `_STRICT_TRANSLATION_DBS` member must never contain this; its
# presence is a strong, adapter-agnostic signal that a raw PubMed query
# leaked through untranslated.
_PUBMED_FIELD_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9 /]*\]")


def _sanitize_query(query: str, db: str) -> str:
    """
    Adapt a query for different database APIs.

    PubMed-syntax queries use field tags like [Title], [MeSH] and nested
    Boolean structure that breaks OpenAlex/Semantic Scholar.  For those
    databases we extract quoted phrases and bare keywords and rebuild a
    simpler query that their search APIs can handle.
    """
    if db in _PUBMED_SYNTAX_DBS:
        return query
    import re

    # 1. Extract all quoted phrases (keep them as key concepts)
    quoted = re.findall(r'"([^"]+)"', query)

    # 2. Remove PubMed field tags
    cleaned = re.sub(r'\[[^\]]+\]', '', query)

    # 3. Remove punctuation-only tokens left behind, collapse whitespace
    cleaned = re.sub(r'[()]+', ' ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    # 4. Remove dangling boolean operators at start/end
    cleaned = re.sub(r'(?i)^\s*(AND|OR|NOT)\b\s*', '', cleaned)
    cleaned = re.sub(r'(?i)\s*\b(AND|OR|NOT)\s*$', '', cleaned).strip()

    if quoted:
        # Rebuild: quoted phrases joined by OR, giving the database the key concepts
        # This is more reliable than passing the full nested Boolean structure
        simplified = " OR ".join(f'"{p}"' for p in quoted[:6])  # cap at 6 phrases
        return simplified

    return cleaned or query


def _resolve_query_for_db(db: str, db_queries: Dict[str, str], default_query: str) -> str:
    """The single query-resolution choke point every dispatch/log call site
    must go through -- no other path in this module may pick a per-db
    query string. `_STRICT_TRANSLATION_DBS` members never fall back to the
    raw PubMed-syntax `default_query` or to `_sanitize_query()`'s lossy
    regex rebuild: a missing/failed translation is a pre-network
    `SearchConfigError` instead."""
    if db in _PUBMED_SYNTAX_DBS:
        return db_queries.get(db) or default_query

    if db in _STRICT_TRANSLATION_DBS:
        query = db_queries.get(db)
        if not query or not query.strip():
            raise SearchConfigError(
                f"{db}: no per-database query translation available; refusing to "
                f"substitute the raw PubMed-syntax query or a lossy regex fallback for {db}"
            )
        if query == default_query:
            raise SearchConfigError(
                f"{db}: translated query is identical to the untranslated PubMed-syntax "
                f"query, which means translation was missing or failed and "
                f"search_strategy_builder.py's heuristic fallback copied the raw query "
                f"through -- {db} requires a genuine per-database translation, never the "
                f"raw PubMed query"
            )
        tag_match = _PUBMED_FIELD_TAG_RE.search(query)
        if tag_match:
            raise SearchConfigError(
                f"{db}: resolved query still contains PubMed field-tag syntax "
                f"({tag_match.group()!r}); refusing to send PubMed-only syntax to {db}"
            )
        return query

    return db_queries.get(db) or _sanitize_query(default_query, db)


def _resolve_adapter_keys(cfg: SearchConfig) -> Dict[str, str]:
    """`db` -> `_ADAPTER_REGISTRY` key, per plan §2.5. Both dispatch paths
    raise `SearchConfigError` before any network call -- this function never
    touches the network."""
    databases = cfg.databases
    mapping = cfg.execution_adapter_map

    if mapping is not None:
        # Mapped dispatch: keys must equal set(databases) EXACTLY.
        if set(mapping) != set(databases):
            raise SearchConfigError(
                f"execution_adapter_map keys {sorted(mapping)} do not exactly "
                f"equal cfg.search.databases {sorted(databases)}"
            )
        unknown = sorted({v for v in mapping.values() if v not in _ADAPTER_REGISTRY})
        if unknown:
            raise SearchConfigError(
                f"execution_adapter_map maps to unknown adapter key(s) "
                f"not in _ADAPTER_REGISTRY: {unknown}"
            )
        return {db: mapping[db] for db in databases}

    # Unmapped dispatch: db itself must be a known registry key.
    resolved: Dict[str, str] = {}
    for db in databases:
        if db not in _ADAPTER_REGISTRY:
            raise SearchConfigError(
                f"unknown database {db!r}: not in _ADAPTER_REGISTRY and no "
                "execution_adapter_map is configured"
            )
        resolved[db] = db
    return resolved


def _dispatch_one(
    db: str, adapter_key: str, query: str, cfg: SearchConfig,
    checkpoint_hooks: Optional[Callable[[str], Optional[SearchCheckpointHooks]]] = None,
) -> SearchBackendResult:
    """Builds the request and runs one adapter. Called from a worker thread
    -- must raise nothing but the adapter's own (Search adapters catch their
    own network/parsing errors internally and report them via
    `SearchBackendResult.failure_detail`, never by raising).

    `checkpoint_hooks`, if given, is a factory called with `db` that returns
    the `SearchCheckpointHooks` for restart-safe pagination on that database,
    or `None` to run without checkpointing -- additive, keyword-only, mirrors
    `screening_node`'s `client_factory` injection precedent. Only passed
    through to adapters whose `.search()` actually declares a `checkpoint`
    parameter (PubMed, OpenAlex, Scopus as of Checkpoint 4); every other
    adapter is called exactly as before, unaffected by this parameter."""
    adapter_cls = _ADAPTER_REGISTRY[adapter_key]
    request = SearchBackendRequest(
        query=query,
        max_results=cfg.max_results_per_database,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        language=cfg.language,
        mailto=os.environ.get(_MAILTO_ENV_VAR[db]) if db in _MAILTO_ENV_VAR else None,
    )
    adapter = adapter_cls()
    if checkpoint_hooks is not None and "checkpoint" in inspect.signature(adapter.search).parameters:
        return adapter.search(request, checkpoint=checkpoint_hooks(db))
    return adapter.search(request)


def search_node(
    state: ReviewState,
    *,
    checkpoint_hooks: Optional[Callable[[str], Optional[SearchCheckpointHooks]]] = None,
) -> Dict[str, Any]:
    """
    Queries all configured databases in parallel via `search_backends/*`.

    `checkpoint_hooks`, if given, is a factory from database name to that
    database's `SearchCheckpointHooks` (Checkpoint 4 restart-safe pagination)
    -- additive, keyword-only, defaults to `None` (no checkpointing, today's
    behavior unchanged). Threaded straight through to `_dispatch_one()`.

    Returns:
        raw_records    — all retrieved records as dicts
        search_log     — per-database query log (DatabaseSearchLog v2)
        prisma_counts  — updated with 'identified' counts
    """
    cfg = ReviewConfig.from_dict(state["config"])
    search_cfg = cfg.search

    # Per-DB query variants from search_strategy_builder (preferred)
    # Falls back to expanded_query, then seed query
    strategy = state.get("search_strategy") or {}
    db_queries: dict = strategy.get("db_queries") or {}
    default_query = strategy.get("expanded_query") or search_cfg.query
    if strategy.get("expanded_query") and strategy["expanded_query"] != search_cfg.query:
        logger.info("Using LLM-expanded query: %r", default_query[:120])

    databases = search_cfg.databases
    today = search_cfg.as_of_date or date.today().isoformat()

    # Unknown-adapter fatality (plan §2.5) -- resolved BEFORE any network call.
    adapter_keys = _resolve_adapter_keys(search_cfg)

    # Per-db query resolution (incl. the `_STRICT_TRANSLATION_DBS` fail-closed
    # check) -- resolved BEFORE any network call, same as adapter_keys above,
    # so a bad Scopus translation blocks dispatch to every database, not just
    # Scopus.
    resolved_queries: Dict[str, str] = {
        db: _resolve_query_for_db(db, db_queries, default_query) for db in databases
    }

    identified: Dict[str, int] = {}

    logger.info(f"Searching {len(databases)} databases for: {default_query!r}")

    # Run adapters in parallel (I/O-bound: network requests), but collect
    # each db's result keyed by db first -- as_completed() yields results in
    # non-deterministic arrival order, which would otherwise make
    # raw_records/search_log's element order depend on network timing.
    # Track 2 plan §8: reference-run reproducibility requires a fixed
    # ordering, so the final lists below are assembled in
    # `cfg.search.databases`' CONFIGURED order, after the parallel phase
    # completes entirely.
    per_db_result: Dict[str, SearchBackendResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(databases))) as pool:
        futures = {
            pool.submit(
                _dispatch_one,
                db,
                adapter_keys[db],
                resolved_queries[db],
                search_cfg,
                checkpoint_hooks,
            ): db
            for db in databases
        }
        for future in as_completed(futures):
            db = futures[future]
            adapter_key = adapter_keys[db]
            result: SearchBackendResult = future.result()

            # Request<->result binding assertions (plan §2.4, round-12
            # finding 2) -- a mismatch is a SearchConfigError, not a silently
            # accepted result.
            if result.adapter != adapter_key:
                raise SearchConfigError(
                    f"{db}: dispatched adapter {adapter_key!r} returned a "
                    f"result bound to adapter {result.adapter!r}"
                )
            if result.requested != search_cfg.max_results_per_database:
                raise SearchConfigError(
                    f"{db}: result.requested ({result.requested}) != "
                    f"cfg.search.max_results_per_database "
                    f"({search_cfg.max_results_per_database})"
                )

            per_db_result[db] = result

    all_records: List[Record] = []
    log_entries: List[DatabaseSearchLog] = []

    for db in databases:
        adapter_key = adapter_keys[db]
        # Review finding P1 (2026-09-24): a search log's `query` field must
        # describe what actually ran, not the pre-adapter resolved query --
        # `result.effective_query` is the adapter's own report of the actual
        # executed expression (e.g. OpenAlex's `title_and_abstract.search:...`
        # filter) when it differs from the resolved query; `None` means the
        # adapter executed the resolved query verbatim.
        result = per_db_result[db]
        per_db_query = (
            result.effective_query if result.effective_query is not None else resolved_queries[db]
        )

        all_records.extend(result.records)
        identified[db] = len(result.records)

        # unsupported_filter_deviations (plan §2.6's language/date matrix
        # columns, via base.py's frozen LANGUAGE_UNSUPPORTED/
        # DATE_UNSUPPORTED tables) -- keyed on the CANONICAL db name, not
        # the adapter_key, so this stays correct once fx_* fixture
        # adapters (step 10) are dispatched through execution_adapter_map.
        deviations: List[str] = []
        if db in LANGUAGE_UNSUPPORTED and search_cfg.language:
            deviations.append(
                f"language filter ({search_cfg.language!r}) not supported server-side by {db}"
            )
        if db in DATE_UNSUPPORTED and (search_cfg.date_from or search_cfg.date_to):
            deviations.append(f"date filter not supported server-side by {db}")

        # total_hits -> DatabaseSearchLog.hits mapping (round-11 finding 2):
        # no source total known -> hits == retrieved.
        hits = result.total_hits if result.total_hits is not None else result.returned

        log_entries.append(
            DatabaseSearchLog(
                database=db,
                execution_adapter=adapter_key,
                adapter_config_version=result.adapter_config_version,
                query=per_db_query,
                date=today,
                retrieval_status=result.retrieval_status,
                termination_reason=result.termination_reason,
                failure_detail=result.failure_detail,
                unsupported_filter_deviations=deviations,
                skipped_malformed=result.skipped_malformed,
                requested=result.requested,
                hits=hits,
                access_mode=result.access_mode,
                rate_limit_remaining=result.rate_limit_remaining,
                rate_limit_reset_seconds=result.rate_limit_reset_seconds,
                retrieved=result.returned,
            )
        )
        logger.info(
            f"  {db} ({adapter_key}): {len(result.records)} records, "
            f"status={result.retrieval_status}"
        )

    logger.info(f"Total retrieved: {len(all_records)}")

    # Update PRISMA counts
    prisma = PRISMACounts(**state["prisma_counts"])
    prisma.identified = identified

    return {
        "raw_records": [r.to_dict() for r in all_records],
        "search_log": SearchLog(entries=log_entries).model_dump(),
        "prisma_counts": prisma.model_dump(),
    }
