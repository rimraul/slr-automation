"""Offline tests for the S7 screening solution (plus one gated live test)."""

from __future__ import annotations

import pytest

from examples.chapter import toolkit
from examples.chapter.s07_screening import solution as sol

_SCRIPTED = toolkit.load_scripted("s07")
_RECORDS = toolkit.load_fixture("bib_records.json")
_EXPECTED = toolkit.load_fixture("expected_screening.json")
_SECOND = toolkit.load_fixture("second_rater_decisions.json")


def _scripted_batches():
    return toolkit.scripted_llm(_SCRIPTED["batches"], key_fn=sol._batch_key)


def _fixed(response: str):
    return toolkit.scripted_llm({"k": response}, key_fn=lambda m: "k")


# --- core screening ------------------------------------------------

def test_screen_batch_reproduces_expected_screening():
    decisions = sol.screen_batch(_RECORDS, toolkit.DEMO_CRITERIA, call_llm=_scripted_batches())
    got = {d.record_id: d.decision for d in decisions}
    assert got == _EXPECTED


def test_partition_sends_every_borderline_to_the_human_queue():
    decisions = sol.screen_batch(_RECORDS, toolkit.DEMO_CRITERIA, call_llm=_scripted_batches())
    part = sol.partition(decisions)
    borderline = {rid for rid, d in _EXPECTED.items() if d == "borderline"}
    assert set(part.human_queue) == borderline
    assert borderline.isdisjoint(part.auto_included)
    assert borderline.isdisjoint(part.auto_excluded)


def test_inter_rater_reliability_band():
    report = sol.inter_rater_reliability(_EXPECTED, _SECOND)
    assert report.n_pairs == len(_SECOND)
    assert 0.3 < report.cohens_kappa < 0.85
    assert report.kappa_label in {"Fair", "Moderate"}


# --- batch_size validation (G2) ----------------------------------

@pytest.mark.parametrize("bad_size", [0, 4])
def test_batch_size_out_of_range_raises_before_any_call(bad_size):
    def exploding_llm(**kwargs):
        raise AssertionError("call_llm must not be reached")

    with pytest.raises(ValueError, match="between 1 and 3"):
        sol.screen_batch(_RECORDS, toolkit.DEMO_CRITERIA, batch_size=bad_size, call_llm=exploding_llm)


# --- shared request path ----------------------------------------

def test_build_screening_request_is_shared_between_modes():
    batch = _RECORDS[:3]
    offline = toolkit.build_screening_request(batch, toolkit.DEMO_CRITERIA, mode="offline")
    live = toolkit.build_screening_request(batch, toolkit.DEMO_CRITERIA, mode="live")
    assert offline["messages"] == live["messages"]
    assert set(offline) == {"model", "messages", "response_format", "temperature"}
    assert offline["model"] != live["model"]


def test_build_screening_request_rejects_four_records_no_truncation():
    with pytest.raises(ValueError):
        toolkit.build_screening_request(_RECORDS[:4], toolkit.DEMO_CRITERIA, mode="offline")


def test_abstracts_are_truncated_to_1000_chars():
    rec = {"id": "rec-01", "title": "T", "abstract": "y" * 4000}
    request = toolkit.build_screening_request([rec], toolkit.DEMO_CRITERIA, mode="offline")
    content = request["messages"][1]["content"]
    assert "y" * 1000 in content and "y" * 1001 not in content


def test_screen_batch_actually_calls_through_the_shared_builder():
    seen = []

    def spy(**kwargs):
        seen.append(kwargs)
        rid_line = kwargs["messages"][1]["content"]
        import re

        ids = re.findall(r"\[(rec-\d+)\]", rid_line)
        return '{"decisions": [' + ",".join(
            f'{{"record_id": "{i}", "decision": "exclude", "reason": "x"}}' for i in ids
        ) + "]}"

    sol.screen_batch(_RECORDS[:3], toolkit.DEMO_CRITERIA, call_llm=spy)
    assert len(seen) == 1
    assert set(seen[0]) == {"model", "messages", "response_format", "temperature"}


# --- rejection cases ------------------------------------------

@pytest.mark.parametrize("variant", ["missing_record_id", "duplicated_record_id", "bad_enum", "short_batch", "non_json"])
def test_malformed_batch_responses_raise_value_error(variant):
    with pytest.raises(ValueError):
        sol.screen_batch(_RECORDS[:3], toolkit.DEMO_CRITERIA, call_llm=_fixed(_SCRIPTED["_bad"][variant]))


# --- gated live test (never executed this phase) --------------

@pytest.mark.live
def test_s07_real_screening_smoke(request):
    """Needs OPENAI_API_KEY, network access, and real (use-time) spend.
    Auto-skipped unless --run-live-chapter is passed."""
    assert request.config.getoption("--run-live-chapter") is True
    three = _RECORDS[:3]
    decisions = sol.screen_batch(three, toolkit.DEMO_CRITERIA, mode="live", call_llm=toolkit.call_llm)
    assert all(d.decision in ("include", "borderline", "exclude") for d in decisions)
