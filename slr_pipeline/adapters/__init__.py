"""Adapters between slr_pipeline's own models and external systems.

Phase A of the FT_SCREENING integration (full-text eligibility
screening): offline mapping/config code only. See
ft_screening_adapter.py's module docstring for scope and the
methodological caveat. No graph/node wiring lives here yet -- that is
Phase B, and only starts once Phase A's own gate closes.
"""

from .ft_screening_adapter import (
    CHAPTER_MODEL_CONFIG_ID,
    ChapterAdapterError,
    ChapterFullTextOutcome,
    ComponentResolutionError,
    CriteriaSet,
    CriterionSpec,
    build_full_text_screening_request,
    chapter_screening_config,
    map_run_result_to_chapter_outcomes,
    record_to_screening_item,
)

__all__ = [
    "CHAPTER_MODEL_CONFIG_ID",
    "ChapterAdapterError",
    "ChapterFullTextOutcome",
    "ComponentResolutionError",
    "CriteriaSet",
    "CriterionSpec",
    "build_full_text_screening_request",
    "chapter_screening_config",
    "map_run_result_to_chapter_outcomes",
    "record_to_screening_item",
]
