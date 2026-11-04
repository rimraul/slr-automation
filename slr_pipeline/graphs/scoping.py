"""
Scoping Review Graph
====================

Pipeline:
  search → dedup → screening → fulltext → extraction → synthesis → reporting

Key differences from SLR:
  - Quality appraisal is SKIPPED (standard for scoping reviews per JBI/Arksey)
  - Charting is now performed via `extraction_node` (the frozen Phase A
    structured-extraction engine, slr_pipeline/extraction/*), using the
    seven-field schema `slr_pipeline.adapters.scoping_charting_adapter.
    scoping_chart_schema()` -- NOT the legacy nine-field descriptive
    `charting_node` (moved to examples/legacy/charting.py; see
    quality_appraisal_synthesis_plan_2026-08-22.md Section 2). This gets
    scoping the same durable manifest+checkpoints+resolutions artifact,
    evidence verification, and no-confirmed-PDF-raises guarantees rapid/
    SLR extraction already has, by construction -- it is the same,
    already-nine-times-reviewed machinery, not a re-derived one.
  - Synthesis includes evidence map
  - Reporting generates PRISMA-ScR diagram

A caller must supply `ReviewConfig.extraction_schema`/
`.extraction_model_config` explicitly before this graph's "extraction"
node can run over any confirmed full-text PDF -- `scoping_review_config()`
sets neither by default (same "no silently-mirrored default" discipline
`rapid_review_config()`/`slr_config()` already follow). Use
`scoping_charting_adapter.scoping_chart_schema().model_dump(mode="json")`/
`dataclasses.asdict(scoping_charting_adapter.scoping_extraction_model_config())`
for the chapter's own calibrated teaching preset.

Usage:
    import dataclasses
    import uuid
    from slr_pipeline.graphs.scoping import build_scoping_graph
    from slr_pipeline.config import scoping_review_config, SearchConfig
    from slr_pipeline.adapters.scoping_charting_adapter import scoping_chart_schema, scoping_extraction_model_config
    from slr_pipeline.state import initial_state

    # extraction_schema/extraction_model_config are mandatory (no default
    # -- Section 1's own "no silently chosen model" discipline) once any
    # confirmed PDF reaches "extraction". scoping_chart_schema()/
    # scoping_extraction_model_config() are this chapter's own calibrated
    # teaching preset (Stage 2 of quality_appraisal_synthesis_plan_
    # 2026-08-22.md); author a different schema/model config directly if
    # this review needs different charting fields.
    cfg = scoping_review_config(
        title="AI in education: a scoping review",
        research_questions=["What AI tools are used in higher education?"],
        population="University students and educators",
        concept="Artificial intelligence tools",
        context="Higher education institutions",
        inclusion_criteria=["Empirical studies", "Published 2015–2025", "English"],
        exclusion_criteria=["Grey literature", "Conference abstracts only"],
        search=SearchConfig(query="artificial intelligence AND higher education"),
        extraction_schema=scoping_chart_schema().model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(scoping_extraction_model_config()),
    )
    # No openai_api_key kwarg -- set the OPENAI_API_KEY environment variable
    # instead; every OpenAI-client construction site resolves it exclusively
    # from there (config.resolve_openai_api_key()).

    graph = build_scoping_graph()
    state = initial_state(cfg.to_dict())
    # A configurable.thread_id is required -- Stage 3's AIUseEvent/
    # SelectionDecisionRecord stores are keyed by it (review_run_id), not
    # just LangGraph's own checkpointer; invoking without one raises.
    # run_scoping_review() below generates one automatically.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(state, config)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import uuid
from typing import Optional

from slr_pipeline.state import ReviewState
from slr_pipeline.nodes.search_strategy_builder import search_strategy_builder_node
from slr_pipeline.nodes.search import search_node
from slr_pipeline.nodes.dedup import dedup_node
from slr_pipeline.nodes.screening import screening_node
from slr_pipeline.nodes.screening_qc import screening_qc_node
from slr_pipeline.nodes.fulltext import fulltext_node
from slr_pipeline.nodes.extraction import extraction_node
from slr_pipeline.nodes.synthesis import synthesis_node
from slr_pipeline.nodes.reporting import reporting_node
from slr_pipeline.graphs.routing import make_post_fulltext_router, make_post_extraction_router


def build_scoping_graph(checkpointer=None) -> StateGraph:
    """
    Compile and return the scoping review StateGraph.

    The graph stops at "fulltext" (never reaches extraction/synthesis/
    reporting) while `state["full_text_review_queue"]` is non-empty --
    see graphs/routing.py. Resolve queued records with
    `nodes.fulltext.resolve_full_text_review_queue()`, apply the result
    with `graph.update_state(config, ...)` on the SAME thread_id, then
    resume with `graph.invoke(None, config)`.

    Also stops at "extraction" while `state["extraction_review_queue"]`
    is non-empty -- resolve with
    `nodes.extraction.resolve_extraction_review_queue()`, same
    update_state/resume sequence, exactly parallel to the fulltext gate
    above and to rapid.py/slr.py's own identical extraction gate.

    Args:
        checkpointer: LangGraph checkpointer for persistence.
                      Defaults to MemorySaver (in-memory).

    Returns:
        Compiled LangGraph runnable.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(ReviewState)

    # --- Register nodes ---
    builder.add_node("search_strategy_builder", search_strategy_builder_node)
    builder.add_node("search", search_node)
    builder.add_node("dedup", dedup_node)
    builder.add_node("screening", screening_node)
    builder.add_node("screening_qc", screening_qc_node)
    builder.add_node("fulltext", fulltext_node)
    builder.add_node("extraction", extraction_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("reporting", reporting_node)

    # --- Edges (sequential; no quality appraisal) ---
    builder.add_edge(START, "search_strategy_builder")
    builder.add_edge("search_strategy_builder", "search")
    builder.add_edge("search", "dedup")
    builder.add_edge("dedup", "screening")
    builder.add_edge("screening", "screening_qc")
    builder.add_edge("screening_qc", "fulltext")
    # Conditional, not unconditional: stop before extraction/synthesis/
    # reporting while any full-text record still has no final include/
    # exclude disposition (codex review, Phase B finding 1). Resume by
    # resolving the queue (nodes.fulltext.resolve_full_text_review_queue())
    # and re-invoking this same compiled graph on the same thread_id.
    builder.add_conditional_edges(
        "fulltext", make_post_fulltext_router("extraction"), {"extraction": "extraction", "END": END},
    )
    # Conditional, not unconditional: stop before synthesis/reporting
    # while any extraction field still has no final human-reviewed
    # disposition -- same gate rapid.py/slr.py already have one stage
    # later, reused via the same router factory (Stage 2 of
    # quality_appraisal_synthesis_plan_2026-08-22.md Section 2.2).
    builder.add_conditional_edges(
        "extraction", make_post_extraction_router("synthesis"), {"synthesis": "synthesis", "END": END},
    )
    builder.add_edge("synthesis", "reporting")
    builder.add_edge("reporting", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Quick-run helper
# ---------------------------------------------------------------------------

def run_scoping_review(config_dict: dict, thread_id: Optional[str] = None) -> dict:
    """
    Run the full scoping review pipeline.

    Args:
        config_dict: ReviewConfig serialised as dict (use cfg.to_dict())
        thread_id:   Unique ID for checkpointing (resume on failure) --
                     also Stage 3's own durable `review_run_id` (every
                     `AIUseEvent`/`SelectionDecisionRecord` this run
                     produces is stored under it). Defaults to a freshly
                     generated UUID4 per call, never a fixed literal --
                     two separate calls sharing one fixed default would
                     blend both their LangGraph checkpoints AND their
                     Stage 3 event/ledger stores into the same directory
                     (codex review). Pass an explicit, stable `thread_id`
                     only when you deliberately want to resume the SAME
                     run later.

    Returns:
        Final ReviewState as dict
    """
    from slr_pipeline.state import initial_state

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    graph = build_scoping_graph()
    state = initial_state(config_dict)
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(state, config)
    return result
