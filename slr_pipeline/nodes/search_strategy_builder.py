"""
search_strategy_builder_node — Step 0 of the SLR pipeline (optional).

Uses an LLM to expand a seed query into a structured Boolean search strategy:
  - Groups seed terms into concept blocks (population, concept, context, …)
  - Adds synonyms and controlled vocabulary (MeSH-style)
  - Produces a candidate Boolean query string

The expanded query is stored in ReviewState['search_strategy']['expanded_query']
and picked up by search_node. If disabled (strategy_builder.enabled=False),
the seed query passes through unchanged.

Ported from codex_implementation with import style adapted to the
canonical `slr_pipeline.*` package convention.

Node input keys:  config
Node output keys: search_strategy, review_run_id, oversight_mode,
                   automation_origin

Stage 3 instrumentation (the Stage 3 node-instrumentation design record): both
call sites (query expansion, db translation) are durable `AIUseEvent`s --
`search_strategy` has no selection-ledger records at all (not a selection
stage). Query expansion has no computed fallback (a failure just keeps
the seed query unchanged, `call_status="failure"`); db translation's
same-query-for-every-database substitute IS a real computed fallback
(`call_status="heuristic_fallback"`).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from slr_pipeline.state import ReviewState, SearchStrategy
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key
from slr_pipeline.nodes.llm_backends import get_llm_backend
from slr_pipeline.run_identity import resolve_review_run_id

from slr_pipeline.ai_traceability import (
    AIUseEventStore, build_ai_use_event, classify_automation_origin_optional,
    compute_prompt_template_hash, reserved_attempt,
)
from slr_pipeline.extraction.hashing import sha256_hex

logger = logging.getLogger(__name__)

QUERY_EXPANSION_STAGE = "search_strategy_query_expansion"
DB_TRANSLATION_STAGE = "search_strategy_db_translation"
COMBINED_STATE_KEY = "search_strategy"   # oversight_mode/automation_origin key covering both stages above


def _protocol_config_ref(cfg: ReviewConfig) -> str:
    seed_terms = {
        "population": cfg.population, "concept": cfg.concept, "context": cfg.context,
        "intervention": cfg.intervention, "comparator": cfg.comparator, "outcome": cfg.outcome,
    }
    payload = {
        "research_questions": cfg.research_questions, "inclusion_criteria": cfg.inclusion_criteria,
        "exclusion_criteria": cfg.exclusion_criteria, "seed_terms": seed_terms,
    }
    return f"protocol_config:sha256:{sha256_hex(payload)}"


# ---------------------------------------------------------------------------
# Per-database query translation
# ---------------------------------------------------------------------------

_DB_DESCRIPTIONS = {
    # Checkpoint 3 query harmonization (2026-09-24): the adapter itself
    # (openalex.py) now sends this query through `filter=
    # title_and_abstract.search:...`, not the fulltext-matching default
    # `search=` param -- described here so the LLM never assumes fulltext
    # terms (e.g. a methods-section-only phrase) will match.
    "openalex":        "OpenAlex REST API (title/abstract-only keyword search -- fulltext is NOT matched, supports AND/OR/NOT and quoted phrases, no field tags)",
    "semantic_scholar":"Semantic Scholar API (keyword search, AND/OR, quoted phrases, no field tags)",
    "arxiv":           "ArXiv API (simple keyword search, no complex Boolean syntax)",
    "crossref":        "Crossref API (keyword search, basic AND/OR)",
    "springer":        "Springer Link API (keyword search, AND/OR, quoted phrases)",
    "ieee":            "IEEE Xplore API (supports AND/OR/NOT and quoted phrases, field tags differ from PubMed)",
    "pubmed":          "PubMed Entrez API (full PubMed Boolean syntax including [Title], [MeSH] etc.)",
    # Track-3 authorization (2026-09-12): this entry was missing entirely,
    # so scopus silently fell back to the generic ".get(db, 'keyword
    # search')" default below -- confirmed root cause (together with the
    # fixed 7-db JSON example this same authorization removes) of a real
    # live run's db-translation call twice omitting the required `scopus`
    # key from its JSON response at temperature 0.0.
    # Checkpoint 3 query harmonization (2026-09-24): TITLE-ABS-KEY also
    # matches Scopus's author keywords and indexed (controlled-vocabulary)
    # keywords, not just the title/abstract text itself -- broader
    # coverage than PubMed's [tiab] or OpenAlex's title_and_abstract.search,
    # named explicitly so this is never treated as equivalent scoping.
    "scopus":          "Scopus Search API (TITLE-ABS-KEY(...) field scoping -- matches title, abstract, AND author/indexed keywords, i.e. broader coverage than plain title/abstract; Boolean AND/OR/AND NOT, quoted phrases -- NOT PubMed-style [tag] field syntax)",
}

_TRANSLATE_SYSTEM = "Return only valid JSON."

_TRANSLATE_PROMPT = """\
You are adapting a PubMed literature search query for use with multiple databases.

Original PubMed query:
{query}

For each database listed below, produce an adapted search query that:
- Preserves the search intent and key concepts
- Uses only syntax the database supports (see descriptions)
- For APIs without field tags: use plain Boolean (AND, OR, NOT) with quoted phrases

Databases:
{db_list}

Return a JSON object with exactly one key per database listed above, for example:
{json_example}

Include only databases listed above. Keep queries concise but complete.
"""


def compute_db_translation_call_identity(*, query: str, databases: List[str], model: str) -> Dict[str, Any]:
    """The complete immutable call-identity fields `_generate_db_queries()`
    itself publishes onto its `AIUseEvent` -- extracted as a shared builder
    (Codex round-8 review, finding 2) so `reporting.py`'s verification can
    reconstruct the SAME values from live config/state and compare them
    field-by-field against a loaded event, rather than checking only
    `call_status`/`tool_kind`/`model_config_id`/`output_artifact_refs` (which
    cannot distinguish a substituted event from a DIFFERENT prompt/config
    whose call_status and output happen to coincide -- e.g. two different
    failure events are indistinguishable by output alone, since a failure/
    fallback branch's output is either empty or a deterministic function of
    `query` alone). `query` must be the CURRENT invocation's own actual input
    to db translation (`SearchStrategy.expanded_query`) -- the real outcome
    of this invocation's own Step 1, never independently re-derived.

    Codex round-9 review, finding 3, P2: widened beyond work_item_id/
    rendered_prompt_hash/prompt_parameters/input_artifact_refs to cover
    EVERY other deterministic field this call site publishes -- model,
    provider, tool_kind, task_purpose, model_config_id, prompt_template_
    version/text/hash. `component_versions`/`access_info`/`attempt`/
    `timestamp` are deliberately EXCLUDED: they depend on runtime/
    environment facts (whether a response was actually received, the
    live client's base_url, retry sequencing, wall-clock time), not on
    the call's own deterministic identity, and binding them would make
    a genuine, unmodified event fail this check."""
    # Always include pubmed so the original query is preserved as-is.
    # Sorted, not a bare set -- an unsorted set's iteration order is not
    # guaranteed stable across runs, which would make the rendered prompt
    # (and its rendered_prompt_hash) non-deterministic across otherwise-
    # identical calls.
    dbs_needed = sorted(set(databases) | {"pubmed"})
    db_list = "\n".join(
        f"- {db}: {_DB_DESCRIPTIONS.get(db, 'keyword search')}"
        for db in dbs_needed
    )
    # Track-3 authorization (2026-09-12): the prompt's JSON example used to
    # be a FIXED literal listing exactly the original 7 databases
    # (never scopus), regardless of `dbs_needed` -- confirmed root cause of
    # a real live run's model twice omitting `scopus` from its response at
    # temperature 0.0: the model saw an authoritative-looking example that
    # implicitly signalled "these are the only expected keys," overriding
    # the separate `Databases:` list above. Generated fresh from
    # `dbs_needed` every call so the example can never silently diverge
    # from what is actually being requested.
    json_example = json.dumps({db: "..." for db in dbs_needed}, indent=2)
    prompt = _TRANSLATE_PROMPT.format(query=query, db_list=db_list, json_example=json_example)
    messages = [
        {"role": "system", "content": _TRANSLATE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    # Track-3 authorization (2026-09-12): `response_format={"type":
    # "json_object"}` only guarantees syntactically valid JSON, not any
    # particular set of keys -- confirmed root cause (together with the
    # prompt fixes above) of the same real live run's translation twice
    # coming back without a required key. OpenAI's Structured Outputs
    # (`type: "json_schema"`, `strict: True`, `additionalProperties:
    # False`) makes a missing configured database key structurally
    # impossible rather than merely validated after the fact -- built
    # dynamically from `dbs_needed` for the same reason the JSON example
    # above is, never a fixed literal. `_generate_db_queries()`'s own
    # missing/empty-key validation and `heuristic_fallback` substitute are
    # preserved unchanged as defense-in-depth: Structured Outputs makes
    # this failure mode rare, not provably impossible (a non-conforming or
    # malformed response is still possible in principle).
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "db_query_translation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {db: {"type": "string"} for db in dbs_needed},
                "required": dbs_needed,
                "additionalProperties": False,
            },
        },
    }
    canonical_payload = {"messages": messages, "model": model, "temperature": 0.0, "response_format": response_format}
    work_item_id = f"{DB_TRANSLATION_STAGE}:{sha256_hex(canonical_payload)}"
    prompt_template_text = json.dumps([{"role": "system", "content": _TRANSLATE_SYSTEM}, {"role": "user", "content": _TRANSLATE_PROMPT}])
    return {
        "messages": messages,
        "work_item_id": work_item_id,
        "rendered_prompt_hash": sha256_hex(messages),
        "prompt_parameters": {"temperature": 0.0, "response_format": response_format},
        "input_artifact_refs": [
            f"query:sha256:{sha256_hex(query)}",
            f"databases:sha256:{sha256_hex(dbs_needed)}",
        ],
        "model": model,
        "provider": "openai",
        "tool_kind": "llm",
        "task_purpose": "Translate the search query into per-database syntax variants",
        # Track-3 authorization (2026-09-12): bumped v1 -> v2, the prompt
        # text (dynamic JSON example) and response_format (Structured
        # Outputs schema) both changed -- reporting.py's INV-R7 hardcoded
        # literal check must be updated to match (see reporting.py).
        "model_config_id": "search_strategy_db_translation_v2",
        "prompt_template_version": "v2",
        "prompt_template_text": prompt_template_text,
        "prompt_template_hash": compute_prompt_template_hash(prompt_template_text),
        "response_format": response_format,
        # Round-1 `/code-review` finding (Track-3 proxy-fix cycle):
        # `_generate_db_queries()` used to independently recompute
        # `sorted(set(databases) | {"pubmed"})` itself -- deterministic and
        # can't diverge given identical inputs today, but exposing THIS
        # function's own already-computed value here means the schema's
        # `required`/`properties` keys and the response-validation logic
        # are provably the same list, not just two calls that happen to
        # agree, closing off any future risk of the two derivations being
        # changed independently and silently disagreeing.
        "dbs_needed": dbs_needed,
    }


def _generate_db_queries(
    query: str,
    databases: List[str],
    model: str,
    client: OpenAI,
    event_store: AIUseEventStore,
    review_run_id: str,
) -> Tuple[Dict[str, str], str]:
    """
    Use an LLM to produce per-database query variants from a seed query.
    Always includes pubmed (unchanged) as the baseline.
    Returns (db_name -> adapted_query dict, this call's own published event_id).
    """
    identity = compute_db_translation_call_identity(query=query, databases=databases, model=model)
    dbs_needed = identity["dbs_needed"]
    messages = identity["messages"]
    work_item_id = identity["work_item_id"]
    response_format = identity["response_format"]

    with reserved_attempt(event_store, stage=DB_TRANSLATION_STAGE, work_item_id=work_item_id) as attempt:
        response_received = False
        resp_model: Optional[str] = None
        parsed: Optional[Dict[str, str]] = None
        failure_exc: Optional[Exception] = None
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format=response_format,
                messages=messages,
                temperature=0.0,
            )
            response_received = True
            resp_model = resp.model
            raw = json.loads(resp.choices[0].message.content)
            raw["pubmed"] = query   # ensure pubmed always keeps the original
            # codex review (second round): str(q) coerced ANY value --
            # null, an array, an object, a number, a boolean -- into a
            # truthy-looking string ("None", "[]", "True", ...) that then
            # passed the non-empty check below. Only a genuine string is
            # ever accepted; anything else is excluded from `candidate`
            # entirely, so it's caught by the "missing database" check
            # right below, not silently stringified into something that
            # looks like a real query.
            candidate = {db: q for db, q in raw.items() if db in dbs_needed and isinstance(q, str)}
            # Semantic validation, not just shape (codex review): an
            # incomplete response (missing configured databases) or one
            # with empty/whitespace query strings must never be accepted
            # as "success" -- it would silently omit real databases from
            # the eventual search. Every requested database must be
            # present with a real, non-empty query. `parsed` itself is
            # only ever assigned AFTER every check passes (never before,
            # and never reset in `except` below) -- assigning it earlier
            # and only resetting on failure is the exact ordering bug
            # this fix itself once had: a raise after an early assignment
            # leaves `parsed` non-None even though validation rejected it.
            missing = set(dbs_needed) - set(candidate.keys())
            if missing:
                raise ValueError(f"DB translation response is missing database(s): {sorted(missing)}")
            empty = [db for db, q in candidate.items() if not q.strip()]
            if empty:
                raise ValueError(f"DB translation response has empty quer{'y' if len(empty) == 1 else 'ies'} for: {sorted(empty)}")
            parsed = candidate
            logger.info("DB query variants generated for: %s", list(parsed.keys()))
        except Exception as e:
            logger.warning("DB query translation failed (%s); using seed query for all DBs", e)
            failure_exc = e

        # F23: a malformed/unparseable response still has a real computed
        # fallback here (the same query for every DB) -- call_status is
        # "heuristic_fallback", and component_versions still reports
        # resp.model whenever a response WAS received (independent axes).
        if parsed is not None:
            call_status = "success"
            db_queries = parsed
        else:
            call_status = "heuristic_fallback"
            db_queries = {db: query for db in dbs_needed}
        component_versions = {model: resp_model} if response_received else {}

        event = build_ai_use_event(
            review_run_id=review_run_id, stage=DB_TRANSLATION_STAGE, work_item_id=work_item_id, attempt=attempt,
            tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
            component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose=identity["task_purpose"],
            model_config_id=identity["model_config_id"],
            prompt_template_version=identity["prompt_template_version"],
            prompt_template_text=identity["prompt_template_text"],
            prompt_parameters=identity["prompt_parameters"],
            rendered_prompt_hash=identity["rendered_prompt_hash"],
            input_artifact_refs=identity["input_artifact_refs"],
            output_artifact_refs=[f"sha256:{sha256_hex(db_queries)}"],
            call_status=call_status, failure_detail=(str(failure_exc) if failure_exc is not None else None),
        )
        event = event_store.publish(event)

    return db_queries, event.event_id


class SearchConcept(BaseModel):
    label: str
    seed_terms: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    controlled_terms: List[str] = Field(default_factory=list)


class SearchStrategyResponse(BaseModel):
    concepts: List[SearchConcept] = Field(default_factory=list)
    candidate_query: str
    notes: List[str] = Field(default_factory=list)


_EXPANSION_SYSTEM = "Return only valid JSON."

_STRATEGY_PROMPT_TEMPLATE = """\
You are designing a transparent database search strategy for a literature review.

Review type: {review_type}
Title: {title}

Research questions:
{rqs}

Framework terms:
{seed_text}

Inclusion criteria:
{inc}

Exclusion criteria:
{exc}

Current seed query:
{seed_query}

Instructions:
- Build concept groups suitable for OpenAlex, PubMed, Crossref, Semantic Scholar, and Scopus-like Boolean search.
- Keep each concept focused; avoid irrelevant buzzwords.
- Suggest at most {limit} synonyms per concept.
- Include controlled vocabulary terms only when plausible (e.g. MeSH).
- {field_hint}
- Return JSON only.

JSON schema:
{{
  "concepts": [
    {{
      "label": "Population | Concept | Context | Outcome | Other",
      "seed_terms": ["..."],
      "synonyms": ["..."],
      "controlled_terms": ["..."]
    }}
  ],
  "candidate_query": "(term1 OR term2) AND (term3 OR term4)",
  "notes": ["brief methodological notes"]
}}
"""


def _build_prompt(cfg: ReviewConfig) -> str:
    rqs = "\n".join(f"- {rq}" for rq in cfg.research_questions) or "- Not provided"
    inc = "\n".join(f"- {c}" for c in cfg.inclusion_criteria) or "- Not provided"
    exc = "\n".join(f"- {c}" for c in cfg.exclusion_criteria) or "- Not provided"
    seed_terms = {
        "population": cfg.population,
        "concept": cfg.concept,
        "context": cfg.context,
        "intervention": cfg.intervention,
        "comparator": cfg.comparator,
        "outcome": cfg.outcome,
    }
    seed_text = "\n".join(
        f"- {k}: {v}" for k, v in seed_terms.items() if v
    ) or "- None"
    limit = cfg.search.strategy_builder.max_synonyms_per_concept
    field_hint = (
        "Use TITLE-ABS-KEY style field scoping where appropriate."
        if cfg.search.strategy_builder.prefer_title_abstract_fields
        else "Do not require field-scoped syntax."
    )
    return _STRATEGY_PROMPT_TEMPLATE.format(
        review_type=cfg.review_type, title=cfg.title, rqs=rqs, seed_text=seed_text,
        inc=inc, exc=exc, seed_query=cfg.search.query or "<empty>", limit=limit, field_hint=field_hint,
    )


def canonical_query_expansion_output(*, expanded_query: str, concepts: List[Dict[str, Any]], notes: List[str]) -> Dict[str, Any]:
    """The exact canonical payload query expansion's own success
    `output_artifact_refs` hashes -- a shared helper (Codex round-7 review,
    finding 2, P2) so the producer's own publish site and reporting.py's
    own verification can never independently drift out of sync with each
    other. Reconstructs the same shape as `SearchStrategyResponse.model_dump
    (mode="json")` (`{"concepts", "candidate_query", "notes"}`) from
    `SearchStrategy`'s own already-stored state fields, so reporting.py
    never needs to import `SearchStrategyResponse` itself -- `sha256_hex()`
    canonicalizes via sorted keys, so key order/dict identity doesn't
    matter, only content."""
    return {"concepts": concepts, "candidate_query": expanded_query, "notes": notes}


def compute_query_expansion_call_identity(cfg: ReviewConfig) -> Dict[str, Any]:
    """The complete immutable call-identity fields `_run_query_expansion()`
    itself publishes onto its `AIUseEvent` -- extracted as a shared builder
    (Codex round-8 review, finding 2) so `reporting.py`'s verification can
    reconstruct the SAME values from live config and compare them field-by-
    field against a loaded event, rather than checking only `call_status`/
    `tool_kind`/`model_config_id`/`output_artifact_refs` (which cannot
    distinguish a substituted event from a DIFFERENT prompt/config whose
    call_status and output happen to coincide -- e.g. every failure event
    for this stage has the identical empty output, `[]`, by construction).

    Codex round-9 review, finding 3, P2: widened beyond work_item_id/
    rendered_prompt_hash/prompt_parameters/input_artifact_refs to cover
    EVERY other deterministic field this call site publishes -- see
    `compute_db_translation_call_identity()`'s own docstring for why
    `component_versions`/`access_info`/`attempt`/`timestamp` are
    deliberately excluded."""
    model = cfg.search.strategy_builder.llm_model
    prompt = _build_prompt(cfg)
    messages = [
        {"role": "system", "content": _EXPANSION_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    canonical_payload = {"messages": messages, "model": model, "temperature": 0.2, "response_format": {"type": "json_object"}}
    work_item_id = f"{QUERY_EXPANSION_STAGE}:{sha256_hex(canonical_payload)}"
    prompt_template_text = json.dumps([{"role": "system", "content": _EXPANSION_SYSTEM}, {"role": "user", "content": _STRATEGY_PROMPT_TEMPLATE}])
    return {
        "messages": messages,
        "work_item_id": work_item_id,
        "rendered_prompt_hash": sha256_hex(messages),
        "prompt_parameters": {"temperature": 0.2, "response_format": {"type": "json_object"}},
        "input_artifact_refs": [_protocol_config_ref(cfg)],
        "model": model,
        "provider": "openai",
        "tool_kind": "llm",
        "task_purpose": "Expand seed search query into concept groups and a candidate boolean query",
        "model_config_id": "search_strategy_query_expansion_v1",
        "prompt_template_version": "v1",
        "prompt_template_text": prompt_template_text,
        "prompt_template_hash": compute_prompt_template_hash(prompt_template_text),
    }


def _run_query_expansion(
    cfg: ReviewConfig, client: OpenAI, model: str, event_store: AIUseEventStore, review_run_id: str,
):
    """Returns (expanded_query, concepts, notes, this call's own published
    event_id). No computed fallback exists at this call site -- any error
    (transport or parse) is `call_status="failure"` and simply keeps the
    seed query unchanged."""
    identity = compute_query_expansion_call_identity(cfg)
    messages = identity["messages"]
    work_item_id = identity["work_item_id"]

    expanded_query = cfg.search.query
    concepts: List[Dict[str, Any]] = []
    notes: List[str] = []

    with reserved_attempt(event_store, stage=QUERY_EXPANSION_STAGE, work_item_id=work_item_id) as attempt:
        response_received = False
        resp_model: Optional[str] = None
        parsed: Optional[SearchStrategyResponse] = None
        failure_exc: Optional[Exception] = None
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.2,
            )
            response_received = True
            resp_model = response.model
            raw = json.loads(response.choices[0].message.content)
            parsed = SearchStrategyResponse(**raw)
            expanded_query = parsed.candidate_query
            concepts = [c.model_dump() for c in parsed.concepts]
            notes = parsed.notes
            logger.info("Query expanded: %d concepts", len(concepts))
        except Exception as exc:
            logger.warning("Query expansion failed: %s; using seed query", exc)
            failure_exc = exc
            notes.append(f"Expansion failed: {exc}")

        call_status = "success" if parsed is not None else "failure"

        # Codex round-8 review, finding 1, P1: this banner line used to be
        # prepended by the CALLER (search_strategy_builder_node), AFTER this
        # function returned -- but this function's own success-path output
        # hash (below) is computed from `notes` BEFORE that prepend, while
        # `state["search_strategy"]["notes"]` (what reporting.py's
        # `canonical_query_expansion_output()` reconstruction hashes) is the
        # FINAL, prepended list. The two hashes could therefore never match
        # for a real, unmodified successful run. Fixed by prepending HERE,
        # before the hash is computed, so the value hashed into
        # `output_artifact_refs` is byte-identical to the final `notes` this
        # function returns (and that ends up in `SearchStrategy.notes`) --
        # the prepended TEXT itself is unchanged, only ITS TIMING relative to
        # hashing moved. (The caller's own "Strategy builder disabled; using
        # seed query." prepend, used when expansion never runs at all, stays
        # in the caller -- there is no event to bind a hash to in that case.)
        notes.insert(0, f"Query expanded with {len(concepts)} concept groups.")

        output_artifact_refs = (
            [f"sha256:{sha256_hex(canonical_query_expansion_output(expanded_query=expanded_query, concepts=concepts, notes=notes))}"]
            if parsed is not None else []
        )
        component_versions = {model: resp_model} if response_received else {}

        event = build_ai_use_event(
            review_run_id=review_run_id, stage=QUERY_EXPANSION_STAGE, work_item_id=work_item_id, attempt=attempt,
            tool_kind=identity["tool_kind"], model=identity["model"], provider=identity["provider"],
            component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose=identity["task_purpose"],
            model_config_id=identity["model_config_id"],
            prompt_template_version=identity["prompt_template_version"],
            prompt_template_text=identity["prompt_template_text"],
            prompt_parameters=identity["prompt_parameters"],
            rendered_prompt_hash=identity["rendered_prompt_hash"],
            input_artifact_refs=identity["input_artifact_refs"],
            output_artifact_refs=output_artifact_refs,
            call_status=call_status, failure_detail=(str(failure_exc) if failure_exc is not None else None),
        )
        event = event_store.publish(event)

    return expanded_query, concepts, notes, event.event_id


def search_strategy_builder_node(state: ReviewState) -> Dict[str, Any]:
    """
    Builds a search strategy for the pipeline. Always runs (even when disabled)
    to produce per-database query variants via LLM translation.

    When strategy_builder.enabled=True: expands the seed query (adds synonyms/concepts).
    Always: translates the final query into per-DB variants stored in db_queries.
    """
    cfg = ReviewConfig.from_dict(state["config"])
    strategy_cfg = cfg.search.strategy_builder
    model = strategy_cfg.llm_model
    databases = cfg.search.databases

    client = OpenAI(api_key=resolve_openai_api_key())

    output_dir = cfg.ensure_output_dir()
    review_run_id = resolve_review_run_id(state, allow_config=True)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)

    seed_query = cfg.search.query
    expanded_query = seed_query
    concepts: List[Dict[str, Any]] = []
    notes: List[str] = []
    # Codex round-5 review, finding 3: the exact event_id(s) THIS
    # invocation published -- persisted on SearchStrategy so reporting's
    # own cross-check can verify against this specific invocation's
    # events, not every historical event ever published under this
    # review_run_id for these stage names.
    producing_event_ids: List[str] = []

    # --- Step 1: Optional query expansion ---
    if strategy_cfg.enabled:
        expanded_query, concepts, notes, expansion_event_id = _run_query_expansion(
            cfg, client, model, event_store, review_run_id)
        producing_event_ids.append(expansion_event_id)

    # --- Step 2: Always translate to per-DB variants ---
    print(f"  Translating query for {len(databases)} databases...")
    db_queries, translation_event_id = _generate_db_queries(expanded_query, databases, model, client, event_store, review_run_id)
    producing_event_ids.append(translation_event_id)
    for db, q in db_queries.items():
        if db != "pubmed":
            logger.info("  %s: %s", db, q[:100])

    # Codex round-8 review, finding 1: the "Query expanded with N concept
    # groups." banner is now prepended INSIDE _run_query_expansion() itself,
    # before that function computes/publishes its own output hash (see its
    # docstring note) -- so it must not be prepended again here. Only the
    # "disabled" banner (no event exists to bind a hash to in that case)
    # still belongs in this node.
    if not strategy_cfg.enabled:
        notes.insert(0, "Strategy builder disabled; using seed query.")

    # automation_origin/oversight_mode: no selection-ledger records exist
    # for this stage (not a selection stage) -- derived directly from
    # BOTH call sites' own AIUseEvents this run, combined under one
    # "search_strategy" state key. Failure events are excluded entirely
    # (INV-06) -- a failure produced no automated output.
    events_this_run = event_store.load_all()
    relevant_stages = {QUERY_EXPANSION_STAGE, DB_TRANSLATION_STAGE}
    events_this_node = [e for e in events_this_run if e.stage in relevant_stages]
    stage_origin = classify_automation_origin_optional(events_this_node)

    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    oversight_mode[COMBINED_STATE_KEY] = "automation_unreviewed"   # no human-review step exists for this stage
    if stage_origin is not None:
        automation_origin[COMBINED_STATE_KEY] = stage_origin

    return {
        "search_strategy": SearchStrategy(
            seed_query=seed_query,
            expanded_query=expanded_query,
            db_queries=db_queries,
            concepts=concepts,
            notes=notes,
            backend=cfg.llm_backend,
            llm_model=model,
            producing_event_ids=producing_event_ids,
        ).model_dump(),
        "review_run_id": review_run_id,
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
    }
