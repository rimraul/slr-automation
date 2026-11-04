"""Shared conditional-routing helpers for graphs -- one implementation,
wired into all three graphs (scoping/rapid/slr), never three separate
copies (codex review, Phase B finding 1's "wire the same adapter into
scoping, rapid, and SLR graphs without three separate implementations"
requirement, applied to this routing gate too).
"""

from __future__ import annotations

from typing import Callable

from slr_pipeline.state import ReviewState


def make_post_fulltext_router(next_node: str) -> Callable[[ReviewState], str]:
    """Returns a LangGraph conditional-edge function for the edge out of
    "fulltext": routes to `next_node` when `full_text_review_queue` is
    empty, else to the sentinel string `"END"` (map it to the real `END`
    object in the caller's `add_conditional_edges(..., {"END": END, ...})`
    path map -- LangGraph's own API takes a dict of return-value ->
    node/END, not the router function returning END directly).

    This is the "conditionally terminate every graph before downstream
    analysis" half of codex review Phase B finding 1: quality_appraisal/
    charting/synthesis/reporting must never run over a corpus that still
    has records with no final include/exclude disposition -- those
    records would otherwise silently vanish from every downstream stage
    while reporting still produced apparently-final PRISMA counts/output
    (the exact defect the review reproduced). `next_node` differs per
    graph (scoping goes straight to "charting"; rapid/slr go to
    "quality_appraisal" first) -- one factory, parameterized, rather than
    three hand-written near-duplicate router functions.

    The "resume" half lives in `nodes.fulltext.resolve_full_text_review_
    queue()`: once a human has resolved every queued record (or the
    caller decides no more will be forthcoming), re-invoking the SAME
    compiled graph on the SAME thread_id re-evaluates this router against
    the updated state and proceeds into `next_node` -- LangGraph's
    checkpointer means `fulltext` itself is never re-run, only the
    routing decision.
    """

    def _route(state: ReviewState) -> str:
        if state.get("full_text_review_queue"):
            return "END"
        return next_node

    return _route


def make_post_extraction_router(next_node: str) -> Callable[[ReviewState], str]:
    """Returns a LangGraph conditional-edge function for the edge out of
    "extraction" (rapid/SLR graphs only -- scoping never has this node):
    routes to `next_node` when `extraction_review_queue` is empty, else to
    the sentinel string `"END"` (see `make_post_fulltext_router`'s own
    docstring for why the caller maps that string to the real `END`
    object rather than this function returning `END` directly).

    Same "conditionally terminate before downstream analysis" contract as
    `make_post_fulltext_router`, one stage later: `synthesis`/
    `meta_analysis`/`reporting` must never run over extraction results
    that still have an unresolved field-level review entry -- doing so
    would let charted_data (built by `nodes.extraction.extraction_node`
    only once the queue is empty -- see that module's own docstring)
    either be silently stale/absent or, worse, get built prematurely from
    an incomplete resolution set. `next_node` differs per graph (rapid
    goes to "synthesis"; SLR goes to "meta_analysis" when that stub is
    wired in, else "synthesis") -- one factory, parameterized, matching
    `make_post_fulltext_router`'s own precedent rather than two
    hand-written near-duplicate router functions.

    Resume path: `nodes.extraction.resolve_extraction_review_queue()`
    followed by `graph.update_state(config, ...)` and
    `graph.invoke(None, config)` on the SAME thread_id -- `extraction`
    itself is never re-run, only this routing decision is re-evaluated.
    """

    def _route(state: ReviewState) -> str:
        if state.get("extraction_review_queue"):
            return "END"
        return next_node

    return _route


def make_post_quality_appraisal_router(next_node: str) -> Callable[[ReviewState], str]:
    """Returns a LangGraph conditional-edge function for the edge out of
    "quality_appraisal" (rapid/SLR graphs only -- scoping never has this
    node): routes to `next_node` when `quality_appraisal_review_queue` is
    empty, else to the sentinel string `"END"` (see
    `make_post_fulltext_router`'s own docstring for why the caller maps
    that string to the real `END` object rather than this function
    returning `END` directly).

    Section 1.10 of quality_appraisal_synthesis_plan_2026-08-22.md: under
    `negative_high_risk_policy="allow_provisional_ai_only"`, provisional
    negative/high-risk leaves live in `quality_appraisal_provisional_ledger`,
    never `quality_appraisal_review_queue` -- so this router's queue-emptiness
    check is unaffected by them, which is the concrete mechanism that lets a
    fully-AI-only run reach `next_node` even with a genuine negative finding.
    Under the default `require_human_confirmation` policy, a provisional
    leaf DOES occupy a queue entry (queue_kind="confirmation") and blocks
    this router until `nodes.quality_appraisal.
    confirm_quality_appraisal_provisional_item()` clears it.

    Resume path: `nodes.quality_appraisal.resolve_quality_appraisal_review_
    queue()` / `confirm_quality_appraisal_provisional_item()` followed by
    `graph.update_state(config, ...)` and `graph.invoke(None, config)` on
    the SAME thread_id -- `quality_appraisal` itself is never re-run, only
    this routing decision is re-evaluated.
    """

    def _route(state: ReviewState) -> str:
        if state.get("quality_appraisal_review_queue"):
            return "END"
        return next_node

    return _route
