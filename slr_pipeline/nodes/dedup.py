"""
dedup_node — Step 2 of the SLR pipeline.

Removes duplicate records using a 3-stage approach:
  1. Exact DOI match
  2. Title semantic similarity  (threshold 0.85)
  3. Abstract semantic similarity (threshold 0.90)

Wraps the existing DeduplicationPipeline from c:/projects/screening/.

Node input keys:  raw_records, prisma_counts
Node output keys: deduplicated_records, prisma_counts
"""

from __future__ import annotations

import platform
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

# Temporary: removed once step 3's package-native search_backends/* rewrite
# (Track 2 plan §2.6/§13 step 3) replaces this legacy screening-adapter import.
SCREENING_DIR = r"c:/projects/screening"
if SCREENING_DIR not in sys.path:
    sys.path.insert(0, SCREENING_DIR)

from slr_pipeline.state import Record, PRISMACounts, ReviewState
from slr_pipeline.config import ReviewConfig
from slr_pipeline.doi import normalise_doi, normalise_title
from slr_pipeline.run_identity import resolve_review_run_id

from slr_pipeline.ai_traceability import AIUseEvent, AIUseEventStore, build_ai_use_event, reserved_attempt
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.selection_ledger import SelectionDecisionStore, derive_source_event_id, emit_selection_decision

logger = logging.getLogger(__name__)


def _dedup_reason(event: AIUseEvent) -> str:
    """Ledger `reason` text for a dedup decision, keyed on which event
    actually produced it (Track 2 plan §3.1's `"exact"` mode has its own
    fixed reason string, distinct from the embedding path's two)."""
    if event.stage == "dedup_exact":
        return "exact DOI/normalised-title deduplication"
    if event.call_status == "success":
        return "embedding-similarity dedup pipeline"
    return f"DOI/title fallback dedup (embedding pipeline failed: {event.failure_detail})"


def _record_content_ref(r: Record) -> str:
    """Content-addressed reference to a record's decision-relevant payload
    -- used as an `input_artifact_ref`, not a bare record id, so the
    reference is bound to WHAT was fed in, not just an opaque id that
    could point to since-changed content."""
    return f"record:{r.id}:sha256:{sha256_hex({'title': r.title, 'abstract': r.abstract, 'doi': r.doi})}"


def dedup_node(state: ReviewState) -> Dict[str, Any]:
    """
    Deduplicates raw_records and returns the unique set.

    Falls back to DOI-only exact dedup if the embedding-based pipeline
    is unavailable (e.g., when sentence-transformers is not installed) --
    always a real attempted-and-failed embedding call in the current code
    (never skipped straight to DOI-only), so this fallback is instrumented
    as a genuine `heuristic_fallback` AIUseEvent, never "no event" (Stage
    3, the Stage 3 node-instrumentation design record F9).

    Every record -- unique or duplicate -- gets a `SelectionDecisionRecord`
    (`stage="dedup"`, `decision_maker="rule_based_automation"` always: the
    embedding-similarity/DOI-match decision itself is deterministic given
    its inputs, never an AI judgment, even though the embedding STEP
    itself is a real tool call worth tracing via its own `AIUseEvent`).
    """
    cfg = ReviewConfig.from_dict(state["config"])
    records = [Record.from_dict(r) for r in state["raw_records"]]

    logger.info(f"Deduplicating {len(records)} records …")

    output_dir = cfg.ensure_output_dir()
    review_run_id = resolve_review_run_id(state, allow_config=True)
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)

    event, deduplicated = _dedup_with_instrumentation(records, cfg, event_store, review_run_id)

    n_dupes = len(records) - len(deduplicated)
    logger.info(f"Removed {n_dupes} duplicates → {len(deduplicated)} unique records")

    unique_ids = {r.id for r in deduplicated}
    # Membership in dedup_node's own returned unique set, NOT r.is_duplicate
    # -- _dedup_with_pipeline() doesn't reliably set that field on the
    # records it drops (a separate, smaller pre-existing gap, noted here,
    # not fixed in this pass).
    for record in records:
        decision = "unique" if record.id in unique_ids else "duplicate"
        source_event_id = derive_source_event_id(
            producer_artifact_id=event.event_id, record_id=record.id, stage="dedup",
        )
        emit_selection_decision(
            store=ledger_store, record_id=record.id, stage="dedup", decision=decision,
            decision_maker="rule_based_automation", source_event_id=source_event_id,
            reason=_dedup_reason(event),
            producer_artifact_id=event.event_id,
        )

    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    oversight_mode["dedup"] = "automation_unreviewed"
    automation_origin["dedup"] = "rule_based_automation"

    prisma = PRISMACounts(**state["prisma_counts"])
    prisma.after_dedup = len(deduplicated)

    return {
        "deduplicated_records": [r.to_dict() for r in deduplicated],
        "prisma_counts": prisma.model_dump(),
        "review_run_id": review_run_id,
        "selection_decision_ledger_dir": str(ledger_store.decisions_dir),
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
    }


def _dedup_with_instrumentation(
    records: List[Record], cfg: ReviewConfig, event_store: AIUseEventStore, review_run_id: str,
) -> Tuple[AIUseEvent, List[Record]]:
    """Dispatches on `cfg.dedup.mode` (Track 2 plan §3.1):
      - `"exact"`  -- deterministic DOI/title-exact path only, its own
        `dedup_exact` `AIUseEvent`, never attempts the embedding pipeline.
      - `"semantic"` -- embedding pipeline only, fail-closed (raises,
        rather than falling back, if it is unavailable).
      - `"auto"` (default) -- current production behavior: always
        attempts the embedding pipeline, falls back to DOI/title-exact on
        any exception.
    """
    mode = cfg.dedup.mode
    if mode == "exact":
        return _dedup_exact_with_instrumentation(records, cfg, event_store, review_run_id)
    return _dedup_embedding_with_instrumentation(
        records, cfg, event_store, review_run_id, fail_closed=(mode == "semantic"),
    )


def _dedup_embedding_with_instrumentation(
    records: List[Record], cfg: ReviewConfig, event_store: AIUseEventStore, review_run_id: str,
    *, fail_closed: bool,
) -> Tuple[AIUseEvent, List[Record]]:
    """Runs the ONE real embedding-pipeline attempt (or its DOI-only
    fallback, unless `fail_closed`) for this node's whole run under one
    concurrency-safe attempt allocation, and publishes the single
    `AIUseEvent` that represents it. `work_item_id`'s fingerprint binds
    decision-relevant input content, the embedding model, thresholds, and
    method -- so a genuine retry with unchanged input is idempotent, while
    a genuinely different input set (e.g. a growing raw_records corpus)
    gets a fresh identity rather than silently colliding.

    `fail_closed=True` (mode `"semantic"`) lets the pipeline's exception
    propagate uncaught -- no fallback, no event published for the failed
    attempt (the reserved attempt slot is simply never written, so a
    retry reuses it) -- a deliberate Track-3 fail-closed preflight rather
    than silently degrading to the exact path."""
    canonical_payload = {
        "records": sorted(
            ((r.id, r.title, r.abstract, r.doi) for r in records), key=lambda t: t[0],
        ),
        "embedding_model": cfg.screening.embedding_model,
        "thresholds": {"doi": 1.0, "title": 0.85, "abstract": 0.90},
    }
    stage = "dedup_embedding"
    work_item_id = f"{stage}:{sha256_hex(canonical_payload)}"

    with reserved_attempt(event_store, stage=stage, work_item_id=work_item_id) as attempt:
        try:
            deduplicated = _dedup_with_pipeline(records, cfg)
        except Exception as e:
            if fail_closed:
                raise
            logger.warning(f"Full dedup pipeline failed ({e}); falling back to DOI dedup")
            deduplicated = _dedup_doi_only(records)
            call_status = "heuristic_fallback"
            failure_detail: Optional[str] = str(e)
            component_versions: Dict[str, str] = {}
        else:
            call_status = "success"
            failure_detail = None
            try:
                import sentence_transformers  # already imported successfully inside _dedup_with_pipeline
                component_versions = {"sentence-transformers": sentence_transformers.__version__}
            except Exception:
                # F21: an unimportable package on the success path would be
                # a contradiction (the pipeline just used it) -- defensive
                # only, never a second crash either way.
                component_versions = {}

        output_payload = sorted(r.id for r in deduplicated)
        event = build_ai_use_event(
            review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=attempt,
            tool_kind="non_llm_tool", model=cfg.screening.embedding_model, provider="sentence-transformers",
            component_versions=component_versions, access_info=None,
            task_purpose="Deduplicate records via sentence-embedding similarity",
            model_config_id="dedup_embedding_v1",
            input_artifact_refs=[_record_content_ref(r) for r in records],
            output_artifact_refs=[f"sha256:{sha256_hex(output_payload)}"],
            call_status=call_status, failure_detail=failure_detail,
        )
        event = event_store.publish(event)
    return event, deduplicated


def _dedup_exact_with_instrumentation(
    records: List[Record], cfg: ReviewConfig, event_store: AIUseEventStore, review_run_id: str,
) -> Tuple[AIUseEvent, List[Record]]:
    """`cfg.dedup.mode == "exact"` (Track 2 plan §3.1): skips the
    embedding pipeline entirely, runs `_dedup_doi_only` directly, and
    publishes one `dedup_exact` `AIUseEvent` -- a non-LLM tool event with
    zero model calls, `call_status` always `"success"` (the deterministic
    match itself cannot fail; a genuinely broken input would fail earlier,
    at `Record` validation)."""
    stage = "dedup_exact"
    canonical_payload = sorted((r.id, r.title, r.abstract, r.doi) for r in records)
    work_item_id = f"{stage}:{sha256_hex(canonical_payload)}"

    with reserved_attempt(event_store, stage=stage, work_item_id=work_item_id) as attempt:
        deduplicated = _dedup_doi_only(records)
        event = build_ai_use_event(
            review_run_id=review_run_id, stage=stage, work_item_id=work_item_id, attempt=attempt,
            tool_kind="non_llm_tool", model="doi_title_exact_match", provider="slr_pipeline",
            component_versions={"python": platform.python_version()}, access_info=None,
            task_purpose="Deduplicate by exact normalised DOI then exact normalised title",
            model_config_id="dedup_exact_v1",
            input_artifact_refs=[f"records:sha256:{sha256_hex(canonical_payload)}"],
            output_artifact_refs=[f"sha256:{sha256_hex([r.id for r in deduplicated])}"],
            call_status="success", failure_detail=None,
        )
        event = event_store.publish(event)
    return event, deduplicated


# ---------------------------------------------------------------------------
# Strategy 1: full semantic dedup via existing pipeline
# ---------------------------------------------------------------------------

def _dedup_with_pipeline(records: List[Record], cfg: ReviewConfig) -> List[Record]:
    """Use the existing DeduplicationPipeline (embedding-based)."""
    from unified_screening_pipeline_v1 import DeduplicationPipeline
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(cfg.screening.embedding_model)
    pipeline = DeduplicationPipeline(
        embed_model=embed_model,
        doi_threshold=1.0,
        title_threshold=0.85,
        abstract_threshold=0.90,
    )

    # Convert Record → Article dataclass expected by existing pipeline
    articles = [_record_to_article(r) for r in records]
    deduped_df, _ = pipeline.deduplicate(articles)

    # Map back to Record, marking duplicates
    unique_ids = set(
        deduped_df[deduped_df["is_duplicate"] == False]["article_id"].tolist()
    )
    return [r for r in records if r.source_id in unique_ids]


# ---------------------------------------------------------------------------
# Strategy 2: lightweight DOI-only exact dedup (fallback)
# ---------------------------------------------------------------------------

def _dedup_doi_only(records: List[Record]) -> List[Record]:
    """
    Fallback dedup: exact normalised-DOI match, then exact normalised-title
    match (Track 2 plan §3.1) -- handles the majority of duplicates
    without embeddings. Both stages are exact-equality only, never fuzzy
    (see `slr_pipeline.doi.normalise_doi`/`normalise_title`).
    """
    seen_dois: set = set()
    seen_titles: set = set()
    unique: List[Record] = []

    for r in records:
        # Stage 1: exact normalised DOI
        if r.doi:
            doi_norm = normalise_doi(r.doi)
            if doi_norm in seen_dois:
                r.is_duplicate = True
                continue
            seen_dois.add(doi_norm)

        # Stage 2: exact normalised title (catches cross-database duplicates without DOIs)
        title_norm = normalise_title(r.title)
        if title_norm and len(title_norm) > 10:
            if title_norm in seen_titles:
                r.is_duplicate = True
                continue
            seen_titles.add(title_norm)

        unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Helper: Record → Article dataclass
# ---------------------------------------------------------------------------

def _record_to_article(r: Record):
    """Convert our Pydantic Record to the Article dataclass used by the
    existing DeduplicationPipeline."""
    try:
        from unified_screening_pipeline_v1 import Article
        return Article(
            source=r.source,
            article_id=r.source_id,
            doi=r.doi,
            title=r.title,
            abstract=r.abstract,
            authors="; ".join(r.authors),
            year=r.year,
            journal=r.journal,
            keywords=", ".join(r.keywords),
            cited_by=r.cited_by,
            url=r.url,
            extra_metadata={},
        )
    except ImportError:
        raise
