"""Integration tests for `slr_pipeline.replay.transport.ReplayServer`
(`track2_pipeline_readiness_plan_2026-08-30.md` Sec 6.1, step 10) -- the
loopback `POST /v1/chat/completions` server real `OpenAI` clients will
eventually talk to (Sec 6.2 wiring is a later step; here the server is
exercised directly with `requests`)."""
from __future__ import annotations

import json

import pytest
import requests

from slr_pipeline.replay.cassette import ReplayCassette
from slr_pipeline.replay.signature import REPLAY_ENDPOINT_PATH, compute_replay_signature
from slr_pipeline.replay.transport import ReplayServer, ReplayTransportError

_AUTH = "Bearer sk-replay-not-a-real-key"


def _body(content: str = "hi") -> dict:
    return {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": content}]}


def _cassette_for(*bodies_and_calls):
    entries = []
    for i, (body, calls) in enumerate(bodies_and_calls):
        entries.append({
            "entry_id": f"e{i}",
            "surface": "query_expansion",
            "signature": compute_replay_signature(REPLAY_ENDPOINT_PATH, body),
            "endpoint": REPLAY_ENDPOINT_PATH,
            "request_preview": "preview",
            "response_body": {"echo": body, "index": i},
            "expected_calls": calls,
        })
    return ReplayCassette.from_list(entries)


@pytest.fixture()
def server(tmp_path):
    cassette = _cassette_for((_body("hi"), 2), (_body("bye"), 1))
    srv = ReplayServer(cassette, tmp_path / "call_log.jsonl")
    with srv:
        yield srv


def _post(srv: ReplayServer, body: dict, *, auth: str = _AUTH, path: str = REPLAY_ENDPOINT_PATH,
          host_header: str | None = None) -> requests.Response:
    headers = {"Authorization": auth, "Content-Type": "application/json"}
    kwargs = {}
    if host_header is not None:
        headers["Host"] = host_header
    return requests.post(f"http://127.0.0.1:{srv.port}{path}", json=body, headers=headers)


def test_matched_request_returns_200_and_response_body_verbatim(server):
    resp = _post(server, _body("hi"))
    assert resp.status_code == 200
    assert resp.json() == {"echo": _body("hi"), "index": 0}


def test_call_log_jsonl_records_one_line_per_call(server, tmp_path):
    _post(server, _body("hi"))
    _post(server, _body("hi"))
    _post(server, _body("bye"))
    lines = (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert records[0]["matched_entry_id"] == "e0"
    assert records[2]["matched_entry_id"] == "e1"
    assert records[0]["surface"] == "query_expansion"
    assert all("ts_monotonic" in r for r in records)


def test_assert_complete_passes_when_all_expected_calls_served(server):
    _post(server, _body("hi"))
    _post(server, _body("hi"))
    _post(server, _body("bye"))
    server.assert_complete()  # does not raise


def test_assert_complete_fails_on_under_serving(server):
    _post(server, _body("hi"))  # only 1 of 2 expected, and "bye" never served
    with pytest.raises(ReplayTransportError):
        server.assert_complete()


def test_unrecorded_signature_returns_422_and_latches(server):
    resp = _post(server, _body("never recorded"))
    assert resp.status_code == 422
    assert server.latched_fatal is True

    # every subsequent request, even a previously-good one, now also 422s.
    resp2 = _post(server, _body("hi"))
    assert resp2.status_code == 422
    assert "latched fatal" in resp2.json()["error"]


def test_wrong_endpoint_returns_422_and_latches(server):
    resp = requests.post(f"http://127.0.0.1:{server.port}/v1/other",
                          json=_body("hi"), headers={"Authorization": _AUTH})
    assert resp.status_code == 422
    assert server.latched_fatal is True


def test_get_method_returns_422_and_latches(server):
    resp = requests.get(f"http://127.0.0.1:{server.port}{REPLAY_ENDPOINT_PATH}",
                         headers={"Authorization": _AUTH})
    assert resp.status_code == 422
    assert server.latched_fatal is True


def test_wrong_authorization_returns_422_and_latches(server):
    resp = _post(server, _body("hi"), auth="Bearer wrong-key")
    assert resp.status_code == 422
    assert server.latched_fatal is True


def test_wrong_host_header_returns_422_and_latches(server):
    resp = _post(server, _body("hi"), host_header="127.0.0.1:9")
    assert resp.status_code == 422
    assert server.latched_fatal is True


def test_served_calls_starts_at_zero_for_unserved_entries(server):
    assert server.served_calls() == {"e0": 0, "e1": 0}
    _post(server, _body("hi"))
    assert server.served_calls()["e0"] == 1
