"""Offline tests for the S12 reporting solution."""

from __future__ import annotations

import copy

import pytest

from examples.chapter import toolkit
from examples.chapter.s12_reporting import solution as sol


@pytest.fixture
def state():
    return toolkit.load_fixture("pipeline_state_fixture.json")


@pytest.fixture
def traceability():
    return toolkit.load_fixture("ai_traceability_fixture.json")


# --- prisma_flow ----------------------------------------------------

def test_prisma_flow_known_correct(state):
    flow = sol.prisma_flow(state)
    assert flow.total_identified == 100
    assert flow.after_dedup == 82
    assert flow.included_after_ta == 20
    assert flow.assessed_full_text == 16
    assert flow.acquisition_failures == 4
    assert flow.included == 10


def test_to_ascii_contains_every_number(state):
    flow = sol.prisma_flow(state)
    text = flow.to_ascii()
    for number in (100, 18, 82, 62, 20, 16, 4, 6, 10):
        assert str(number) in text


def test_non_terminal_state_is_rejected_first(state):
    broken = copy.deepcopy(state)
    broken["full_text_review_queue"] = [{"record_id": "rec-99"}]
    with pytest.raises(sol.PrismaInconsistencyError, match="terminal"):
        sol.prisma_flow(broken)


def test_missing_queue_key_is_rejected(state):
    broken = copy.deepcopy(state)
    del broken["full_text_review_queue"]
    with pytest.raises(sol.PrismaInconsistencyError, match="missing the required key"):
        sol.prisma_flow(broken)


def test_missing_quality_key_is_rejected(state):
    broken = copy.deepcopy(state)
    del broken["prisma_counts"]["quality_assessed"]
    with pytest.raises(sol.PrismaInconsistencyError, match="missing the required key"):
        sol.prisma_flow(broken)


def test_absent_quality_key_is_not_silently_read_as_not_run(state, traceability):
    broken_pc = copy.deepcopy(state["prisma_counts"])
    del broken_pc["quality_assessed"]
    with pytest.raises(ValueError, match="absent"):
        sol.classify_stage_oversight(traceability, broken_pc, state["oversight_mode"])


def test_broken_dedup_conservation_names_group_one(state):
    broken = copy.deepcopy(state)
    broken["prisma_counts"]["after_dedup"] = 999
    with pytest.raises(sol.PrismaInconsistencyError, match="group 1"):
        sol.prisma_flow(broken)


def test_broken_ft_reason_sum_names_group_five(state):
    broken = copy.deepcopy(state)
    broken["prisma_counts"]["excluded_full_text_reasons"]["wrong_population"] = 99
    with pytest.raises(sol.PrismaInconsistencyError, match="group 5"):
        sol.prisma_flow(broken)


def test_quality_counts_present_names_group_six(state):
    broken = copy.deepcopy(state)
    broken["prisma_counts"]["quality_assessed"] = 5
    with pytest.raises(sol.PrismaInconsistencyError, match="group 6"):
        sol.prisma_flow(broken)


# --- classify_stage_oversight -------------------------------------

def test_classify_stage_oversight_dimensions(state, traceability):
    pc = state["prisma_counts"]
    om = state["oversight_mode"]
    result = sol.classify_stage_oversight(traceability, pc, om)

    ta = result["title_abstract_screening"]
    assert ta.execution == "run"
    assert ta.automation_origin == "ai"
    assert ta.human_reviewed is True

    dedup = result["dedup"]
    assert dedup.execution == "run"
    assert dedup.automation_origin == "rule_based_automation"
    assert dedup.human_reviewed is False

    qa = result["quality_appraisal"]
    assert qa.execution == "not_run"


def test_stage_oversight_is_frozen(state, traceability):
    result = sol.classify_stage_oversight(
        traceability, state["prisma_counts"], state["oversight_mode"]
    )
    with pytest.raises(Exception):
        result["dedup"].execution = "not_run"


def test_not_run_comes_from_prisma_field_not_absent_events(state, traceability):
    # Even if we fake AI events for QA, execution stays not_run while the
    # PRISMA field is None.
    tweaked = copy.deepcopy(traceability)
    tweaked_qa = tweaked["stages"]["quality_appraisal"]
    tweaked_qa["instrumentation_status"] = "instrumented"
    tweaked_qa["event_count"] = 3
    result = sol.classify_stage_oversight(tweaked, state["prisma_counts"], state["oversight_mode"])
    assert result["quality_appraisal"].execution == "not_run"
