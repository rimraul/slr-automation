"""Offline tests for the S11 synthesis + topic-modelling solution."""

from __future__ import annotations

import socket

import pytest
import pytest_socket

from examples.chapter import toolkit
from examples.chapter.s11_synthesis_topics import solution as sol

_DATA = toolkit.load_fixture("synthesis_studies.json")
_S11 = toolkit.load_scripted("s11")


def _map_fake(mapping):
    return toolkit.scripted_llm(
        mapping, key_fn=lambda m: m[-1]["content"].split("STUDY_ID: ", 1)[1].split("\n", 1)[0]
    )


def _refine_fake_ok():
    return toolkit.scripted_llm(_S11["refine"], key_fn=sol._refine_key)


def _refine_fake_fixed(response):
    return toolkit.scripted_llm({"k": response}, key_fn=lambda m: "k")


def _answers():
    fake = _map_fake(_S11["map"])
    return [
        sol.map_study(s["study_id"], _DATA["rq"], s["charted"], call_llm=fake)
        for s in _DATA["studies"]
    ]


# --- map / refine -------------------------------------------------

def test_map_study_binds_id_from_the_parameter():
    fake = _map_fake(_S11["map"])
    answer = sol.map_study("study-A", _DATA["rq"], _DATA["studies"][0]["charted"], call_llm=fake)
    assert answer.study_id == "study-A"
    assert set(answer.cited_fields) <= set(_DATA["studies"][0]["charted"])


def test_map_then_refine_covers_every_study_and_surfaces_the_contradiction():
    synthesis = sol.refine_synthesis(_DATA["rq"], _answers(), call_llm=_refine_fake_ok())
    assert set(synthesis.cited_study_ids) == {"study-A", "study-B", "study-C"}
    assert synthesis.contradictions


def test_premature_citation_of_an_unseen_study_raises_at_the_first_fold():
    # batch_size=2: the first fold has only study-A, study-B in view, but the
    # scripted response already cites study-C.
    with pytest.raises(ValueError, match="seen so far"):
        sol.refine_synthesis(_DATA["rq"], _answers(), call_llm=_refine_fake_fixed(_S11["_bad"]["premature_citation"]))


def test_dropped_citation_raises():
    with pytest.raises(ValueError):
        sol.refine_synthesis(_DATA["rq"], _answers(), call_llm=_refine_fake_fixed(_S11["_bad"]["dropped_citation"]))


def test_invented_citation_raises():
    with pytest.raises(ValueError):
        sol.refine_synthesis(_DATA["rq"], _answers(), call_llm=_refine_fake_fixed(_S11["_bad"]["invented_citation"]))


@pytest.mark.parametrize("bad_size", [0, -1])
def test_non_positive_batch_size_is_rejected(bad_size):
    def exploding_llm(*a, **k):
        raise AssertionError("call_llm must not be reached")

    with pytest.raises(ValueError, match=">= 1"):
        sol.refine_synthesis(_DATA["rq"], _answers(), batch_size=bad_size, call_llm=exploding_llm)


def test_unknown_cited_field_raises_in_map_study():
    fake = _map_fake({"study-A": _S11["_bad"]["unknown_cited_field"]})
    with pytest.raises(ValueError):
        sol.map_study("study-A", _DATA["rq"], _DATA["studies"][0]["charted"], call_llm=fake)


# --- topic modelling (slow) -----------------------------------

@pytest.mark.slow
def test_bertopic_offline_deterministic():
    texts = toolkit.load_fixture("topic_corpus.json")["documents"]
    embeddings = toolkit.load_fixture("topic_corpus_embeddings.npy")

    # network provably blocked
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.create_connection(("example.com", 80), timeout=2)

    first = sol.run_topics(texts, embeddings)
    second = sol.run_topics(texts, embeddings)
    assert first.assignments == second.assignments

    blocks = [first.assignments[0:4], first.assignments[4:8], first.assignments[8:12]]
    for block in blocks:
        assert len(set(block)) == 1, block
    labels = [block[0] for block in blocks]
    assert len(set(labels)) == 3
    assert len({t for t in first.assignments if t != -1}) == 3
