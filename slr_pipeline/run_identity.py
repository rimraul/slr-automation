"""Shared `review_run_id` resolution -- Stage 3 node instrumentation
(the Stage 3 node-instrumentation design record, section 2.4). One place every
node and every out-of-graph resolver derives/validates the LangGraph
`thread_id` that identifies the whole pipeline invocation, so no node
repeats this logic (and its edge cases) independently.
"""
from __future__ import annotations

import re
from typing import Optional

from langgraph.config import get_config

from slr_pipeline.state import ReviewState

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def resolve_review_run_id(state: ReviewState, *, allow_config: bool) -> str:
    """The one place `review_run_id` is derived and validated.

    `allow_config=True` (every LangGraph node): tries `state["review_run_id"]`
    first, then `langgraph.config.get_config()["configurable"]["thread_id"]`
    -- `get_config()` raises `RuntimeError` outside an active runnable
    context (e.g. a node function called directly in a unit test); that is
    caught and treated as "no config source available," not fatal by
    itself. If BOTH sources are present and disagree, raises -- state
    should never silently diverge from the graph's own active thread. If
    NEITHER is available, raises naming both sources that were checked.

    `allow_config=False` (`resolve_full_text_review_queue()` and any other
    resolver called between graph invocations, with only `state` in hand):
    only ever reads `state["review_run_id"]`; never touches `get_config()`
    at all, since there is no active runnable context to read one from.

    Either way, the resolved value is validated against the same safe-
    token regex `AIUseEventStore`/`SelectionDecisionStore` already
    enforce, before being handed back to a caller that will use it to
    construct a store.
    """
    state_value: Optional[str] = state.get("review_run_id")

    if not allow_config:
        if not state_value:
            raise ValueError(
                "resolve_review_run_id(allow_config=False) found no review_run_id in state -- this "
                "resolver runs outside an active LangGraph invocation and cannot fall back to "
                "get_config(); the state passed in must already carry review_run_id from a prior node"
            )
        _validate_safe_token(state_value)
        return state_value

    config_value: Optional[str] = None
    try:
        config = get_config()
    except RuntimeError:
        config = None
    if config is not None:
        config_value = (config.get("configurable") or {}).get("thread_id")

    if state_value and config_value and state_value != config_value:
        raise ValueError(
            f"review_run_id mismatch: state carries {state_value!r}, but the active LangGraph "
            f"thread_id is {config_value!r} -- refusing to silently prefer one over the other"
        )

    review_run_id = state_value or config_value
    if not review_run_id:
        raise ValueError(
            "resolve_review_run_id() found no review_run_id: neither state['review_run_id'] nor an "
            "active LangGraph config's configurable.thread_id is available -- this node must be "
            "invoked either with review_run_id already in state, or from within a real graph "
            "invocation that provides a thread_id"
        )
    _validate_safe_token(review_run_id)
    return review_run_id


def _validate_safe_token(review_run_id: str) -> None:
    if not _SAFE_RUN_ID_RE.fullmatch(review_run_id):
        raise ValueError(
            f"review_run_id {review_run_id!r} is not a safe token -- must match [A-Za-z0-9_-]{{1,128}} "
            "(no path separators, no '.', no '..')"
        )
