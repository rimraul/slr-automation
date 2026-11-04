from __future__ import annotations

import pytest

from slr_pipeline.search_backends.base import SearchBackendRequest, SearchBackendResult, finalize_result
from slr_pipeline.state import Record


def _request(**overrides):
    base = dict(query="cats", max_results=1000, date_from=None, date_to=None, language=None, mailto=None)
    base.update(overrides)
    return SearchBackendRequest(**base)


def _record(n: int) -> Record:
    return Record(id=f"src-{n}", source="src", source_id=str(n), title=f"t{n}")


def test_exhaustive_with_max_results_cap_reached_is_rejected():
    """Review finding P2 (2026-09-24): 'exhaustive' + 'max_results_cap_reached'
    is self-contradictory -- exhaustive implies no cap was needed -- and must
    be rejected by the termination_reason/retrieval_status compatibility
    matrix, not merely accepted as arbitrary text."""
    with pytest.raises(ValueError, match="not compatible with retrieval_status"):
        SearchBackendResult(
            adapter="src", records=[], retrieval_status="exhaustive",
            termination_reason="max_results_cap_reached", requested=10, returned=0,
        )


def test_finalize_result_unexplained_gap_becomes_partial_not_capped():
    """Review finding P1 (2026-09-24): a known total_hits exceeding returned,
    with request.max_results not reached either, is an unexplained gap --
    must never default to 'capped'."""
    request = _request(max_results=1000)
    records = [_record(i) for i in range(300)]

    result = finalize_result("src", request, records, 0, 303, None)

    assert result.retrieval_status == "partial"
    assert result.termination_reason == "unexplained_retrieval_gap"
    assert "303" in result.failure_detail
    assert "300" in result.failure_detail


def test_finalize_result_unexplained_gap_with_zero_records_is_failed():
    request = _request(max_results=1000)

    result = finalize_result("src", request, [], 0, 5, None)

    assert result.retrieval_status == "failed"
    assert result.termination_reason == "unexplained_retrieval_gap"


def test_finalize_result_cap_reached_stays_capped_despite_larger_total_hits():
    request = _request(max_results=2)
    records = [_record(0), _record(1)]

    result = finalize_result("src", request, records, 0, 10, None)

    assert result.retrieval_status == "capped"
    assert result.termination_reason == "not_yet_instrumented"


def test_finalize_result_unknown_total_hits_stays_capped():
    request = _request(max_results=1000)
    records = [_record(0)]

    result = finalize_result("src", request, records, 0, None, None)

    assert result.retrieval_status == "capped"


def test_finalize_result_source_exhausted_true_wins_over_gap_default():
    request = _request(max_results=1000)
    records = [_record(i) for i in range(5)]

    result = finalize_result(
        "src", request, records, 0, 5, None,
        source_exhausted=True, termination_reason="total_hits_equals_retrieved",
    )

    assert result.retrieval_status == "exhaustive"
