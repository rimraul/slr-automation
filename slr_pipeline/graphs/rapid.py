"""
Rapid Review Graph
==================

Pipeline:
  search → dedup → screening → fulltext → extraction → quality_appraisal
         → synthesis → reporting

Extraction (not charting) is used here -- structured, evidence-verified
field extraction via the frozen Phase A engine (slr_pipeline/extraction/*),
wired in through nodes/extraction.py's extraction_node. Quality appraisal
runs AFTER extraction (Section 1.9 of quality_appraisal_synthesis_plan_
2026-08-22.md, matching extraction/PROTOCOL.md's own documented canonical
sequence). The graph stops before "quality_appraisal" while
state["extraction_review_queue"] is non-empty, and before "synthesis"
while state["quality_appraisal_review_queue"] is non-empty -- see
graphs/routing.py's make_post_extraction_router()/
make_post_quality_appraisal_router() and nodes/extraction.py's
resolve_extraction_review_queue()/nodes/quality_appraisal.py's
resolve_quality_appraisal_review_queue() for the resolve/update_state/
resume sequences, exactly parallel to the fulltext gate above.

Key differences from scoping review:
  - Reduced database set (see rapid_review_config defaults)
  - Quality appraisal IS included (teaching_appraisal_v1 by default -- see
    quality_appraisal_synthesis_plan_2026-08-22.md Section 1.3; there is
    no numeric quality_threshold any more, appraisal never excludes)
  - Single screener acceptable (but borderlines still go to human queue)
  - Charting is structured extraction via extraction_node, same as SLR

Key differences from full SLR:
  - Fewer databases (speed trade-off, must be documented)
  - Simplified quality appraisal (teaching_appraisal_v1 vs
    rob2_inspired_study_level_teaching_v1)
  - No meta-analysis

Usage:
    import dataclasses
    from slr_pipeline.graphs.rapid import build_rapid_graph, run_rapid_review
    from slr_pipeline.config import rapid_review_config, SearchConfig, QualityAppraisalConfig
    from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
    from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind

    # extraction_schema/extraction_model_config are mandatory (Section 1
    # of quality_appraisal_synthesis_plan_2026-08-22.md's own "no silently
    # chosen model" discipline) -- there is no default; author the fields
    # this specific review actually needs. quality_appraisal_model_config
    # is equally mandatory once quality_appraisal_node runs.
    schema = ExtractionSchema(
        schema_id="rapid-clinical-decision-support-v1", version="1",
        fields=[ExtractionFieldSpec(
            field_id="study_design", question="What study design was used?",
            description="The paper's own stated research design.",
            value_kind=ValueKind.TEXT, cardinality="single",
            provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
        )],
    )
    extraction_model_config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="rapid-teaching-preset-v1",
        reasoning_effort="medium", max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )
    qa_model_config = OpenAIExtractionConfig(
        model="gpt-5.6-terra", model_config_id="rapid-qa-teaching-preset-v1",
        reasoning_effort="medium", max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )

    cfg = rapid_review_config(
        title="LLMs in clinical decision support: a rapid review",
        research_questions=["How are LLMs used in clinical decision support?"],
        search=SearchConfig(query="large language model AND clinical decision support"),
        extraction_schema=schema.model_dump(mode="json"),
        extraction_model_config=dataclasses.asdict(extraction_model_config),
        quality_appraisal=QualityAppraisalConfig(checklist_id="teaching_appraisal_v1"),
        quality_appraisal_model_config=dataclasses.asdict(qa_model_config),
    )
    # No openai_api_key kwarg -- set the OPENAI_API_KEY environment variable
    # instead; every OpenAI-client construction site resolves it exclusively
    # from there (config.resolve_openai_api_key()).
    result = run_rapid_review(cfg.to_dict())
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


def build_rapid_graph(checkpointer=None) -> StateGraph:
    """Compile and return the rapid review StateGraph.

    Stops at "fulltext" while `state["full_text_review_queue"]` is
    non-empty -- see graphs/routing.py and build_scoping_graph()'s own
    docstring for the resolve/update_state/resume sequence. Also stops at
    "extraction" while `state["extraction_review_queue"]` is non-empty --
    see this module's own header docstring."""
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
    # Order is fulltext -> extraction -> quality_appraisal -> synthesis
    # (Section 1.9 of quality_appraisal_synthesis_plan_2026-08-22.md,
    # matching extraction/PROTOCOL.md's own documented canonical sequence
    # -- the prior quality_appraisal -> extraction order was drift, not a
    # deliberate design choice).
    builder.add_conditional_edges(
        "fulltext", make_post_fulltext_router("extraction"),
        {"extraction": "extraction", "END": END},
    )
    # Conditional, not unconditional: stop before quality_appraisal while
    # any extraction field still has no final human-reviewed disposition
    # (Extraction Phase B requirement 7, mirroring make_post_fulltext_router's
    # own gate one stage later).
    builder.add_conditional_edges(
        "extraction", make_post_extraction_router("quality_appraisal"),
        {"quality_appraisal": "quality_appraisal", "END": END},
    )
    # Conditional, not unconditional: stop before synthesis/reporting while
    # any quality-appraisal leaf still has no final human-reviewed
    # disposition (Section 1.10).
    builder.add_conditional_edges(
        "quality_appraisal", make_post_quality_appraisal_router("synthesis"),
        {"synthesis": "synthesis", "END": END},
    )
    builder.add_edge("synthesis", "reporting")
    builder.add_edge("reporting", END)

    return builder.compile(checkpointer=checkpointer)


def run_rapid_review(config_dict: dict, thread_id: Optional[str] = None) -> dict:
    """`thread_id` also serves as Stage 3's own durable `review_run_id` --
    defaults to a freshly generated UUID4 per call, never a fixed
    literal (two calls sharing one fixed default would blend both their
    LangGraph checkpoints and their AIUseEvent/SelectionDecisionRecord
    stores into the same directory). Pass an explicit, stable value only
    to deliberately resume the SAME run later."""
    from slr_pipeline.state import initial_state
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    graph = build_rapid_graph()
    state = initial_state(config_dict)
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config)
