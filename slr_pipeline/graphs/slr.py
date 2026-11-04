"""
Full Systematic Literature Review (SLR) Graph
==============================================

Pipeline:
  search → dedup → screening → screening_qc → fulltext → extraction
         → quality_appraisal → synthesis → reporting

Key differences from rapid review:
  - All 7 databases searched (maximum sensitivity)
  - RoB2-inspired quality appraisal (rob2_inspired_study_level_teaching_v1,
    5-domain signalling-question shape -- no numeric threshold, never
    gates eligibility; see quality_appraisal_synthesis_plan_2026-08-22.md
    Section 1.3)
  - screening_qc: independent second-rater IRR (kappa/PABAK/Gwet's AC1) on a
    calibration sample, plus topic-saturation analysis, computed after the
    full screening pass (not a pre-run pilot -- see nodes/screening_qc.py)
  - Structured, evidence-verified extraction via the frozen Phase A engine
    (slr_pipeline/extraction/*, wired in through nodes/extraction.py's
    extraction_node) -- not charting.py's own Pydantic/chunking methods.
    The graph stops before "meta_analysis"/"synthesis" while
    state["extraction_review_queue"] is non-empty -- see graphs/routing.py's
    make_post_extraction_router() and nodes/extraction.py's
    resolve_extraction_review_queue() for the resolve/update_state/resume
    sequence, exactly parallel to the fulltext gate above.
  - IMRAD synthesis + optional meta-analysis stubs
  - PRISMA 2020 (not PRISMA-ScR)

Note on meta-analysis:
  A meta_analysis_node is stubbed but not yet implemented.
  It would consume charted_data and produce pooled effect sizes.
  Requires: homogeneous outcomes, effect size data in the extraction schema.

Usage:
    from slr_pipeline.graphs.slr import build_slr_graph, run_slr
    from slr_pipeline.config import slr_config

    cfg = slr_config(
        title="Effect of spaced repetition on vocabulary retention: SLR",
        research_questions=["What is the effect of spaced repetition on L2 vocabulary retention?"],
        search=SearchConfig(query="spaced repetition AND vocabulary AND second language"),
        extraction_schema=my_extraction_schema.model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(my_openai_extraction_config),
        quality_appraisal=QualityAppraisalConfig(checklist_id="rob2_inspired_study_level_teaching_v1"),
        quality_appraisal_model_config=dataclasses.asdict(my_qa_model_config),
    )
    # No openai_api_key kwarg -- set the OPENAI_API_KEY environment variable
    # instead; every OpenAI-client construction site resolves it exclusively
    # from there (config.resolve_openai_api_key()).
    result = run_slr(cfg.to_dict())
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
from slr_pipeline.nodes.quality_appraisal import quality_appraisal_node
from slr_pipeline.nodes.extraction import extraction_node
from slr_pipeline.nodes.synthesis import synthesis_node
from slr_pipeline.nodes.reporting import reporting_node
from slr_pipeline.graphs.routing import make_post_fulltext_router, make_post_extraction_router, make_post_quality_appraisal_router


# ---------------------------------------------------------------------------
# Stub: meta-analysis node (future work)
# ---------------------------------------------------------------------------

def meta_analysis_node(state: ReviewState) -> dict:
    """
    Stub for quantitative meta-analysis.

    When implemented, this node will:
      1. Extract effect sizes from charted_data
      2. Compute pooled effects (random/fixed effects model)
      3. Generate forest plots
      4. Assess heterogeneity (I², Cochran's Q)
      5. Run subgroup analyses and sensitivity analyses

    Libraries to use: PyMeta, statsmodels, rpy2 (metafor).
    This node is only connected when cfg.synthesis.run_meta_analysis=True.
    """
    return {}   # no-op until implemented


def build_slr_graph(checkpointer=None, include_meta_analysis: bool = False) -> StateGraph:
    """
    Compile and return the full SLR StateGraph.

    Stops at "fulltext" while `state["full_text_review_queue"]` is
    non-empty -- see graphs/routing.py and build_scoping_graph()'s own
    docstring for the resolve/update_state/resume sequence. Also stops
    at "extraction" while `state["extraction_review_queue"]` is
    non-empty -- see this module's own header docstring.

    Args:
        checkpointer:        LangGraph checkpointer (default: MemorySaver)
        include_meta_analysis: Connect the meta_analysis_node stub
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(ReviewState)

    builder.add_node("search_strategy_builder", search_strategy_builder_node)
    builder.add_node("search", search_node)
    builder.add_node("dedup", dedup_node)
    builder.add_node("screening", screening_node)
    builder.add_node("screening_qc", screening_qc_node)
    builder.add_node("fulltext", fulltext_node)
    builder.add_node("quality_appraisal", quality_appraisal_node)
    builder.add_node("extraction", extraction_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("reporting", reporting_node)

    builder.add_edge(START, "search_strategy_builder")
    builder.add_edge("search_strategy_builder", "search")
    builder.add_edge("search", "dedup")
    builder.add_edge("dedup", "screening")
    builder.add_edge("screening", "screening_qc")
    builder.add_edge("screening_qc", "fulltext")
    # Conditional, not unconditional -- see graphs/routing.py's docstring
    # and scoping.py's own identical gate (codex review, Phase B finding 1).
    # Order is fulltext -> extraction -> quality_appraisal ->
    # [meta_analysis] -> synthesis (Section 1.9 of
    # quality_appraisal_synthesis_plan_2026-08-22.md, matching
    # extraction/PROTOCOL.md's own documented canonical sequence -- the
    # prior quality_appraisal -> extraction order was drift, not a
    # deliberate design choice). The meta_analysis stub keeps its own
    # prior position relative to whatever comes immediately before
    # synthesis -- now quality_appraisal instead of extraction.
    builder.add_conditional_edges(
        "fulltext", make_post_fulltext_router("extraction"),
        {"extraction": "extraction", "END": END},
    )
    # Conditional, not unconditional: stop before quality_appraisal while
    # any extraction field still has no final human-reviewed disposition
    # (Extraction Phase B requirement 7).
    builder.add_conditional_edges(
        "extraction", make_post_extraction_router("quality_appraisal"),
        {"quality_appraisal": "quality_appraisal", "END": END},
    )

    # Conditional, not unconditional: stop before meta_analysis/synthesis
    # while any quality-appraisal leaf still has no final human-reviewed
    # disposition (Section 1.10).
    if include_meta_analysis:
        builder.add_node("meta_analysis", meta_analysis_node)
        builder.add_conditional_edges(
            "quality_appraisal", make_post_quality_appraisal_router("meta_analysis"),
            {"meta_analysis": "meta_analysis", "END": END},
        )
        builder.add_edge("meta_analysis", "synthesis")
    else:
        builder.add_conditional_edges(
            "quality_appraisal", make_post_quality_appraisal_router("synthesis"),
            {"synthesis": "synthesis", "END": END},
        )

    builder.add_edge("synthesis", "reporting")
    builder.add_edge("reporting", END)

    return builder.compile(checkpointer=checkpointer)


def run_slr(config_dict: dict, thread_id: Optional[str] = None,
            meta_analysis: bool = False) -> dict:
    """`thread_id` also serves as Stage 3's own durable `review_run_id` --
    defaults to a freshly generated UUID4 per call, never a fixed
    literal (two calls sharing one fixed default would blend both their
    LangGraph checkpoints and their AIUseEvent/SelectionDecisionRecord
    stores into the same directory). Pass an explicit, stable value only
    to deliberately resume the SAME run later."""
    from slr_pipeline.state import initial_state
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    graph = build_slr_graph(include_meta_analysis=meta_analysis)
    state = initial_state(config_dict)
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config)
