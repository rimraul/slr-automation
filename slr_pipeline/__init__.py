"""
slr_pipeline — Automated literature review pipeline using LangGraph.

Three review families:
  - scoping  : search → dedup → screening → fulltext → extraction → synthesis → reporting
  - rapid    : scoping + quality_appraisal (teaching_appraisal_v1)
  - slr      : rapid + all databases + RoB2-inspired QA (rob2_inspired_study_level_teaching_v1) + optional meta-analysis

Quick start:
    from slr_pipeline.config import scoping_review_config
    from slr_pipeline.graphs.scoping import run_scoping_review

    cfg = scoping_review_config(
        title="My scoping review",
        research_questions=["What is known about X?"],
        search={"query": "X AND Y", "databases": ["openalex", "pubmed"]},
    )
    # No openai_api_key kwarg -- set the OPENAI_API_KEY environment variable
    # instead; every OpenAI-client construction site resolves it exclusively
    # from there (config.resolve_openai_api_key()).
    result = run_scoping_review(cfg.to_dict())
"""

from .config import ReviewConfig, scoping_review_config, rapid_review_config, slr_config
from .state import ReviewState, Record, PRISMACounts, initial_state

__all__ = [
    "ReviewConfig",
    "scoping_review_config",
    "rapid_review_config",
    "slr_config",
    "ReviewState",
    "Record",
    "PRISMACounts",
    "initial_state",
]
