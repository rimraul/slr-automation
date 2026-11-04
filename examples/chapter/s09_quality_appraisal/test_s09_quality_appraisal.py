"""Offline tests for the S9 quality-appraisal solution."""

from __future__ import annotations

import pytest

from examples.chapter import toolkit
from examples.chapter.s09_quality_appraisal import solution as sol

_SCRIPTED = toolkit.load_scripted("s09")
_PAPER = toolkit.load_fixture("papers/relevant.md")
_ANSWER_KEY = toolkit.load_fixture("papers/answer_key.json")["relevant"]["s09"]


def _llm(response: str):
    return toolkit.scripted_llm({"relevant": response}, key_fn=lambda m: "relevant")


def test_checklist_shape():
    checklist = sol.build_teaching_checklist()
    assert len(checklist) >= 6
    assert {i.item_id for i in checklist} == set(_ANSWER_KEY)


def test_appraise_reproduces_answer_key_exactly():
    result = sol.appraise(_PAPER, sol.TEACHING_APPRAISAL_V1, call_llm=_llm(_SCRIPTED["papers"]["relevant"]))
    by_id = {i.item_id: i for i in result.items}
    for item_id, expected in _ANSWER_KEY.items():
        assert by_id[item_id].status == expected["answer"]  # exact, no "unclear" escape hatch
    assert by_id["dual_independent_screening"].provisional_pending_human_confirmation is True
    assert result.n_provisional == 1
    assert result.review_queue == []
    assert result.requires_human_action is True


def test_fabricated_quote_goes_to_review_queue_not_provisional():
    result = sol.appraise(
        _PAPER, sol.TEACHING_APPRAISAL_V1, call_llm=_llm(_SCRIPTED["_bad"]["fabricated_quote"])
    )
    queued = {q.item_id: q.queue_reason for q in result.review_queue}
    assert queued.get("search_reproducible") == "evidence_verification_failed"
    by_id = {i.item_id: i for i in result.items}
    assert by_id["search_reproducible"].provisional_pending_human_confirmation is False
    assert by_id["search_reproducible"].status == "unclear"


def test_evidence_free_yes_is_coerced_to_unclear_and_queued():
    result = sol.appraise(
        _PAPER, sol.TEACHING_APPRAISAL_V1, call_llm=_llm(_SCRIPTED["_bad"]["evidence_free_yes"])
    )
    queued = {q.item_id: q.queue_reason for q in result.review_queue}
    assert queued.get("objective_stated") == "no_evidence_found"
    by_id = {i.item_id: i for i in result.items}
    assert by_id["objective_stated"].status == "unclear"
    assert by_id["objective_stated"].provisional_pending_human_confirmation is False


def test_invalid_status_raises():
    with pytest.raises(ValueError, match="unsupported appraisal answer"):
        sol.appraise(_PAPER, sol.TEACHING_APPRAISAL_V1, call_llm=_llm(_SCRIPTED["_bad"]["invalid_status"]))


@pytest.mark.parametrize("variant", ["missing_item", "duplicate_item"])
def test_checklist_coverage_violations_raise(variant):
    with pytest.raises(ValueError, match="coverage mismatch"):
        sol.appraise(_PAPER, sol.TEACHING_APPRAISAL_V1, call_llm=_llm(_SCRIPTED["_bad"][variant]))


def test_result_has_no_score_field():
    assert not hasattr(sol.AppraisalResult, "score")
    assert "score" not in sol.AppraisalResult.model_fields
