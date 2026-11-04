"""
Track 3 continuation plan (squishy-petting-starlight.md) Unit D --
continuation graph.

Entry point is `continuation_fulltext_node` (Unit C), wired to the SAME
downstream edges `graphs/scoping.py` uses (`make_post_fulltext_router`,
`extraction_node`, `make_post_extraction_router`, `synthesis_node`).
`search`/`dedup`/`screening`/`screening_qc` are never registered as nodes
in this graph -- a continuation never re-searches/re-dedups/re-screens;
`state["included_records"]` is seeded verbatim from the parent bundle's
own `terminal_state.json` by the launcher (Unit A item 7), not produced
by any node in this graph.

Reporting: `synthesis -> reporting -> END`, where "reporting" is
`continuation_reporting_node` (Unit E) -- it builds the cumulative
`ReportingSourceSnapshot` (parent bundle's own published snapshot UNIONed
with this run's own live state, see `continuation/cumulative_snapshot.py`)
and hands it to the ordinary `reporting_node(state, cumulative_snapshot=
...)` for the real report-writing/bundle-staging work.

Invoked via `graph.invoke(seeded_state, run_config)` on a fresh
`thread_id` + fresh `SqliteSaver` -- never a resume of the parent's own
checkpoint (that checkpoint belongs to a different, already-terminal,
already-published run). `seeded_state` must include `continuation_partition`
(Unit A item 13), `preserved_included_records` (Unit A item 14), and
`retrieved_pdf_manifest` (Unit A item 15) alongside `included_records`
(Unit A item 7) and the other fields `initial_state()` already provides.

`ContinuationReviewState(ReviewState)` declares those three fields as
genuine TypedDict fields -- LangGraph's `_get_channels(schema)`
(`langgraph/graph/state.py`) builds exactly one channel per
`get_type_hints(schema, include_extras=True)` entry on whatever schema is
passed to `StateGraph(...)`; `ReviewState` itself declares none of the
three, so seeding them alone (without this subclass) would leave them
silently dropped by LangGraph's own input-mapping (`langgraph/pregel/
_io.py::map_input()`), a non-fatal warning, never an exception, and
`continuation_fulltext_node` would `KeyError` on its very first read.
`get_type_hints()` walks the MRO, so this subclass's channels include
every inherited `ReviewState` field plus these three new ones. This
module's own `builder = StateGraph(ContinuationReviewState)` call is
scoped to this one module -- `ReviewState`/`graphs/scoping.py` and every
other graph/node in the codebase are unaffected.
"""
from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from slr_pipeline.state import ReviewState
from slr_pipeline.nodes.fulltext_continuation import continuation_fulltext_node
from slr_pipeline.nodes.extraction import extraction_node
from slr_pipeline.nodes.synthesis import synthesis_node
from slr_pipeline.nodes.reporting_continuation import continuation_reporting_node
from slr_pipeline.graphs.routing import make_post_fulltext_router, make_post_extraction_router


class ContinuationReviewState(ReviewState):
    continuation_partition: Dict[str, Any]
    preserved_included_records: List[Dict[str, Any]]
    retrieved_pdf_manifest: Dict[str, Dict[str, str]]
    # Unit E: seeded once by the launcher, never mutated by any node.
    # `parent_bundle_payload_root` is the parent bundle's own `payload/`
    # directory path (the exact argument shape `open_bundle()` itself
    # takes) -- not `continuation_lineage.json`'s path (Unit A, a
    # different durable artifact one level up) -- so
    # `continuation_reporting_node` can `open_bundle()` it directly with
    # zero extra path arithmetic.
    parent_bundle_payload_root: str
    # Independently re-verified against the parent bundle's own ledger by
    # `_union_ledger_records()` (`continuation/cumulative_snapshot.py`) --
    # never trusted from this list alone.
    preserved_decision_citations: List[Dict[str, Any]]


def build_scoping_continuation_graph(checkpointer=None) -> StateGraph:
    """Compile and return the continuation StateGraph.

    Entry node is "fulltext" (`continuation_fulltext_node`), not "search"
    -- there is no search_strategy_builder/search/dedup/screening/
    screening_qc stage in this graph at all.

    Args:
        checkpointer: LangGraph checkpointer for persistence. Defaults to
                      MemorySaver (in-memory) -- a real launcher passes a
                      fresh `SqliteSaver` on a fresh path, never the
                      parent run's own checkpoint DB.

    Returns:
        Compiled LangGraph runnable.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(ContinuationReviewState)

    # input_schema=ContinuationReviewState explicitly on every node
    # (**fix, this round**): LangGraph's own `add_node()` infers a node's
    # input schema from its function signature's first-parameter type hint
    # when `input_schema` isn't given (`langgraph/graph/state.py`'s
    # `inferred_input_schema` logic) -- `continuation_fulltext_node`/
    # `extraction_node`/`synthesis_node` are all annotated `state:
    # ReviewState` (the base class, reused unmodified from `graphs/
    # scoping.py`), so without this override LangGraph would silently
    # narrow each node's own input mapping to ONLY `ReviewState`'s
    # declared channels, dropping `continuation_partition`/
    # `preserved_included_records`/`retrieved_pdf_manifest` before
    # `continuation_fulltext_node` ever sees them -- reproduced directly:
    # `state["continuation_partition"]` raised `KeyError` even though the
    # graph schema itself (`ContinuationReviewState`) and the seeded input
    # dict both carried the key correctly.
    builder.add_node("fulltext", continuation_fulltext_node, input_schema=ContinuationReviewState)
    builder.add_node("extraction", extraction_node, input_schema=ContinuationReviewState)
    builder.add_node("synthesis", synthesis_node, input_schema=ContinuationReviewState)
    builder.add_node("reporting", continuation_reporting_node, input_schema=ContinuationReviewState)

    builder.add_edge(START, "fulltext")
    builder.add_conditional_edges(
        "fulltext", make_post_fulltext_router("extraction"), {"extraction": "extraction", "END": END},
    )
    builder.add_conditional_edges(
        "extraction", make_post_extraction_router("synthesis"), {"synthesis": "synthesis", "END": END},
    )
    builder.add_edge("synthesis", "reporting")
    builder.add_edge("reporting", END)

    return builder.compile(checkpointer=checkpointer)
