"""Integration tests for `slr_pipeline.cost_cap.proxy.CostCapProxy` --
exercised with `requests` directly against a fake upstream (never the real
`https://api.openai.com`), mirroring how `tests/test_replay_transport.py`
exercises `ReplayServer`. No network to the real OpenAI API and no
`OPENAI_API_KEY`/`my_api_keys.py` involved anywhere in this file."""
from __future__ import annotations

import json
import http.client
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
import requests

from slr_pipeline.cost_cap import pricing
from slr_pipeline.cost_cap import proxy as proxy_module
from slr_pipeline.cost_cap.ledger import CostLedger
from slr_pipeline.cost_cap.proxy import (
    _CHAT_COMPLETIONS_PATH,
    _MAX_RESPONSE_FORMAT_SCHEMA_TOKENS,
    _CostCapRequestHandler,
    CostCapProxy,
    count_prompt_tokens,
    estimate_response_format_tokens,
)


def _usage_response(model="gpt-4o", prompt_tokens=100, completion_tokens=50, cached_tokens=0):
    return {
        "model": model,
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b""
        self.server.received_bodies.append(json.loads(raw.decode("utf-8")))
        status, resp_body = self.server.next_response()
        payload = json.dumps(resp_body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _FakeUpstreamServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler_cls, responses):
        super().__init__(address, handler_cls)
        self._responses = list(responses)
        self._default = (200, _usage_response())
        self.received_bodies = []
        self._lock = threading.Lock()

    def next_response(self):
        with self._lock:
            if self._responses:
                return self._responses.pop(0)
            return self._default


@pytest.fixture()
def fake_upstream():
    responses = []
    httpd = _FakeUpstreamServer(("127.0.0.1", 0), _FakeUpstreamHandler, responses)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def ledger(tmp_path):
    return CostLedger(tmp_path / "ledger.json", cap_usd=10.0)


@pytest.fixture()
def proxy(ledger, fake_upstream, tmp_path):
    upstream_url = f"http://127.0.0.1:{fake_upstream.server_address[1]}"
    p = CostCapProxy(ledger, tmp_path / "call_log.jsonl", upstream_url=upstream_url)
    with p:
        yield p


def _chat_url(proxy: CostCapProxy) -> str:
    return f"http://127.0.0.1:{proxy.port}/v1/chat/completions"


def test_allowed_call_forwards_and_returns_upstream_body_verbatim(proxy, fake_upstream):
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=100, completion_tokens=50)))
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    assert resp.json()["usage"]["prompt_tokens"] == 100
    assert len(fake_upstream.received_bodies) == 1


def test_allowed_call_commits_actual_not_worst_case_cost(proxy, ledger, fake_upstream):
    fake_upstream._responses.append((200, _usage_response(model="gpt-4o", prompt_tokens=100, completion_tokens=50)))
    before = ledger.remaining_usd()
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                           "max_tokens": 4096},
                  headers={"Authorization": "Bearer sk-fake"})
    after = ledger.remaining_usd()
    # actual cost: 100 prompt + 50 completion tokens at gpt-4o prices, NOT
    # the worst-case reservation sized for max_tokens=4096
    expected_actual = (100 / 1_000_000) * 2.50 + (50 / 1_000_000) * 10.00
    assert before - after == pytest.approx(expected_actual, abs=1e-6)


def test_unpriced_model_refused_without_forwarding(proxy, fake_upstream, ledger):
    resp = requests.post(_chat_url(proxy),
                          json={"model": "not-a-real-model", "messages": [{"role": "user", "content": "hi"}]},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 402
    assert len(fake_upstream.received_bodies) == 0  # never forwarded
    assert ledger.remaining_usd() == pytest.approx(10.0)  # untouched


def test_budget_exceeded_refused_without_forwarding(ledger, fake_upstream, tmp_path):
    ledger.reserve("pre-existing", 9.999999)  # leave almost nothing
    upstream_url = f"http://127.0.0.1:{fake_upstream.server_address[1]}"
    with CostCapProxy(ledger, tmp_path / "call_log.jsonl", upstream_url=upstream_url) as p:
        resp = requests.post(_chat_url(p),
                              json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 10_000}],
                                    "max_tokens": 4096},
                              headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 402
    assert len(fake_upstream.received_bodies) == 0  # never forwarded to the real/fake API at all


def test_reserve_internal_valueerror_refused_500_not_connection_crash(proxy, ledger, fake_upstream):
    """Round-5 `/code-review` finding: `reserve()` also raises a bare
    `ValueError` for internal-invariant failures that are NOT a
    caller-facing budget refusal (a non-finite `worst_case_usd` reaching
    it, or a `call_id` collision) -- this used to propagate uncaught out
    of `_handle_post`, crashing the connection instead of returning a
    clean response. Round-6 `/code-review`: that new 500 path had no test
    of its own."""
    with patch.object(CostLedger, "reserve", side_effect=ValueError("simulated internal invariant violation")):
        resp = requests.post(_chat_url(proxy),
                              json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                              headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 500
    assert len(fake_upstream.received_bodies) == 0  # never forwarded


def test_upstream_error_releases_reservation_and_relays_status(proxy, ledger, fake_upstream):
    fake_upstream._responses.append((500, {"error": {"message": "upstream boom"}}))
    before = ledger.remaining_usd()
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 500
    # nothing was actually spent -- the reservation must be fully released
    assert ledger.remaining_usd() == pytest.approx(before)


def test_wrong_path_refused_422(proxy):
    resp = requests.post(f"http://127.0.0.1:{proxy.port}/v1/embeddings", json={"model": "gpt-4o"},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422


def test_call_log_records_both_allowed_and_refused_attempts(proxy, fake_upstream, tmp_path):
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer sk-fake"})
    requests.post(_chat_url(proxy), json={"model": "not-a-real-model", "messages": []},
                  headers={"Authorization": "Bearer sk-fake"})
    lines = (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
    decisions = [json.loads(line)["decision"] for line in lines]
    assert decisions == ["allowed", "refused_unpriced"]


def test_missing_max_tokens_gets_default_ceiling_injected_into_forwarded_body(proxy, fake_upstream):
    """Round-1 `/code-review` finding: 4 of the 8 real call sites never set
    `max_tokens`/`max_completion_tokens` themselves -- the proxy must inject
    the same default ceiling the worst-case reservation was sized for into
    the ACTUAL request forwarded upstream, or the reservation is a hopeful
    upper bound rather than an enforced one."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == 4096


def test_caller_supplied_max_tokens_is_left_untouched(proxy, fake_upstream):
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                           "max_tokens": 123},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_tokens"] == 123
    assert "max_completion_tokens" not in forwarded


def test_missing_usage_field_charges_full_worst_case_not_near_zero(proxy, ledger, fake_upstream):
    """Round-1 `/code-review` finding: a 200 response with no/incomplete
    `usage` must never be committed as near-zero cost -- the safe direction
    is to charge the full worst-case reservation, never to under-charge a
    call that (for all the proxy can verify) really was billed."""
    fake_upstream._responses.append((200, {"model": "gpt-4o", "choices": [{"message": {"content": "ok"}}]}))
    before = ledger.remaining_usd()
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    after = ledger.remaining_usd()
    assert resp.status_code == 200
    expected_worst_case = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000,
    )
    assert before - after == pytest.approx(expected_worst_case, rel=1e-6)


def test_missing_usage_field_logs_unverified_decision(proxy, fake_upstream, tmp_path):
    fake_upstream._responses.append((200, {"model": "gpt-4o", "choices": [{"message": {"content": "ok"}}]}))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer sk-fake"})
    lines = (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
    decisions = [json.loads(line)["decision"] for line in lines]
    assert decisions == ["allowed_usage_unverified_charged_worst_case"]


def test_missing_usage_field_records_completion_tokens_matching_worst_case_charge(proxy, ledger, fake_upstream):
    """Round-42 `/code-review` finding, reproduced live: worst-case-charge
    commits used to hardcode `completion_tokens=0` while `actual_usd` was
    the worst-case dollar figure (which prices `max_output_tokens * n`
    output tokens) -- an internally self-contradictory `_calls[]` audit
    record (nonzero dollars, zero completion tokens) unreconstructable by
    anyone auditing `ledger.json`. `completion_tokens` must equal the same
    token count the dollar figure was actually priced from."""
    fake_upstream._responses.append((200, {"model": "gpt-4o", "choices": [{"message": {"content": "ok"}}]}))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                           "max_tokens": 777},
                  headers={"Authorization": "Bearer sk-fake"})
    record = ledger._calls[-1]
    assert record.completion_tokens == 777  # n defaults to 1
    assert record.actual_usd == pytest.approx(
        pricing.actual_cost_usd(model="gpt-4o", prompt_tokens=record.prompt_tokens,
                                 completion_tokens=record.completion_tokens, cached_tokens=0))


def test_malformed_non_json_200_body_also_charges_worst_case(proxy, ledger, fake_upstream):
    """Same safe-direction rule, for the case where the 200 body isn't even
    valid JSON (not just missing `usage`) -- must not raise out of the
    handler, and must still charge the worst-case reservation rather than
    silently committing nothing."""
    fake_upstream._responses.append((200, "not a json object, just a bare string"))
    before = ledger.remaining_usd()
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    after = ledger.remaining_usd()
    assert resp.status_code == 200
    assert before > after  # something was charged, not nothing


def test_huge_int_usage_field_charges_full_worst_case_not_lost_entirely(proxy, ledger, fake_upstream):
    """Round-9 `/code-review` finding, reproduced live: `usage.
    prompt_tokens` as a huge JSON integer (e.g. `10**400`) used to make
    `is_finite_number()` raise `OverflowError` instead of returning
    `False`, escaping the surrounding `except ValueError:` and propagating
    out of the usage-verification block entirely -- for a call the real
    API DID actually complete (a 200 with a usage block, however
    malformed), the outer `finally` would then call `ledger.release()`
    instead of `ledger.commit()`, silently erasing real spend from the
    cap's running total. Same "safe direction" rule as the missing/
    malformed-usage tests above: an unverifiable usage field must charge
    the full worst-case reservation, never release it uncharged."""
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=10 ** 400, completion_tokens=50)))
    before = ledger.remaining_usd()
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    after = ledger.remaining_usd()
    assert resp.status_code == 200
    expected_worst_case = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000,
    )
    assert before - after == pytest.approx(expected_worst_case, rel=1e-6)


def test_huge_int_max_completion_tokens_falls_back_to_default_ceiling_not_a_crash(proxy, fake_upstream):
    """Round-9 `/code-review` finding, reproduced live: a syntactically
    valid but astronomically large JSON integer (`10**400`) in
    `max_completion_tokens` used to crash the handler thread with an
    uncaught `OverflowError` from `math.isfinite()` deep inside
    `resolve_max_output_tokens()` -- the client saw a broken connection
    with zero response bytes instead of the clean fallback-to-default
    behavior every other invalid value (`0`, negative, NaN, inf) already
    gets."""
    fake_upstream._responses.append((200, _usage_response()))
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_completion_tokens": 10 ** 400}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == pricing.DEFAULT_MAX_OUTPUT_TOKENS_CEILING


def test_stop_drains_in_flight_request_before_closing_socket(ledger, tmp_path):
    """`daemon_threads = True` means `server_close()` alone does not wait
    for an in-flight `do_POST` thread. `stop()` must block until that
    thread's ledger commit has actually happened, or a subprocess-per-run
    driver exiting right after `stop()` returns could kill an already-
    billed call before its cost was ever recorded."""
    release_upstream = threading.Event()
    reached_upstream = threading.Event()

    class _SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or "0")
            self.rfile.read(length) if length else b""
            reached_upstream.set()
            release_upstream.wait(timeout=5)
            payload = json.dumps(_usage_response()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    slow_upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    slow_upstream.daemon_threads = True
    upstream_thread = threading.Thread(target=slow_upstream.serve_forever, daemon=True)
    upstream_thread.start()
    try:
        upstream_url = f"http://127.0.0.1:{slow_upstream.server_address[1]}"
        p = CostCapProxy(ledger, tmp_path / "call_log.jsonl", upstream_url=upstream_url, drain_timeout_s=10.0)
        p.start()

        result = {}

        def make_request():
            result["resp"] = requests.post(
                _chat_url(p), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer sk-fake"},
            )

        req_thread = threading.Thread(target=make_request)
        req_thread.start()
        assert reached_upstream.wait(timeout=5), "request never reached fake upstream"

        stop_thread = threading.Thread(target=p.stop)
        stop_thread.start()
        # stop() must still be blocked (draining), since the handler thread
        # hasn't been released to finish yet.
        stop_thread.join(timeout=1)
        assert stop_thread.is_alive(), "stop() returned before the in-flight request finished"

        release_upstream.set()
        stop_thread.join(timeout=10)
        req_thread.join(timeout=10)

        assert result["resp"].status_code == 200
        assert ledger.remaining_usd() < 10.0  # the completed call's cost was committed
    finally:
        release_upstream.set()
        slow_upstream.shutdown()
        slow_upstream.server_close()


def test_malformed_non_dict_body_refused_422(proxy, fake_upstream):
    """Round-3 `/code-review` finding: a JSON body that parses fine but
    isn't a JSON object (e.g. a bare array) used to crash the handler with
    an uncaught AttributeError on `body.get(...)` before any response was
    written -- the real SDK client saw a broken connection instead of a
    clean 4xx. Nothing here has touched `ledger.reserve()` yet."""
    resp = requests.post(_chat_url(proxy), json=["not", "a", "dict"], headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert len(fake_upstream.received_bodies) == 0


def test_malformed_non_list_messages_refused_422(proxy, fake_upstream):
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": "not-a-list"},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert len(fake_upstream.received_bodies) == 0


@pytest.mark.parametrize("falsy_non_list_messages", [0, "", False])
def test_malformed_falsy_non_list_messages_refused_422_not_silently_defaulted(
    proxy, fake_upstream, falsy_non_list_messages,
):
    """Round-8 `/code-review` finding, reproduced live: `messages =
    body.get("messages") or []` treated ANY falsy-but-present value the
    same as an absent key, silently defaulting to `[]` and skipping the
    `isinstance(messages, list)` check entirely -- a malformed
    `messages: 0` body was accepted (and its pre-flight cost estimate
    silently computed from an empty list) instead of being refused."""
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": falsy_non_list_messages},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert len(fake_upstream.received_bodies) == 0


def test_malformed_explicit_null_messages_refused_422_not_silently_defaulted(proxy, fake_upstream):
    """Round-9 `/code-review` finding, reproduced live: checking
    `body.get("messages") is None` cannot distinguish an explicit JSON
    `"messages": null` from a genuinely ABSENT key -- both produce `None`
    from `.get()`. An explicit `null` must be refused as a present-but-
    wrong-type value, the same as any other non-list `messages`, not
    silently defaulted to `[]`."""
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": None},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert len(fake_upstream.received_bodies) == 0


def test_malformed_non_numeric_max_tokens_refused_422(proxy, fake_upstream):
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [], "max_tokens": "lots"},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert len(fake_upstream.received_bodies) == 0


def test_malformed_non_numeric_value_in_the_dropped_key_does_not_refuse(proxy, fake_upstream):
    """Round-8 `/code-review` finding, reproduced live: the type-
    validation loop used to check BOTH `max_completion_tokens` and
    `max_tokens` regardless of which one is authoritative -- so
    `{"max_completion_tokens": 500, "max_tokens": "unlimited"}` was
    refused with a 422 even though `max_tokens` is always dropped, unread,
    and therefore harmless once `max_completion_tokens` is present. Only
    the authoritative key's value should ever be validated."""
    fake_upstream._responses.append((200, _usage_response()))
    resp = requests.post(_chat_url(proxy),
                          json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                "max_completion_tokens": 500, "max_tokens": "unlimited"},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == 500
    assert "max_tokens" not in forwarded


def test_commit_persist_failure_does_not_block_response_or_mislabel_decision(proxy, ledger, fake_upstream, tmp_path):
    """Round-3 `/code-review` finding: an `OSError` writing the ledger's
    on-disk audit trail during `commit()` must not propagate out of
    `commit()` -- if it did, the proxy's own `finally` block would treat an
    already-committed (already-billed) call as unresolved and call
    `release()` on it, which would both give the client a broken connection
    for a call that really happened AND falsely log the call as released
    rather than allowed."""
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=100, completion_tokens=50)))
    with patch.object(CostLedger, "_persist_locked", side_effect=OSError("disk full (simulated)")):
        resp = requests.post(_chat_url(proxy),
                              json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                              headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    lines = (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
    decisions = [json.loads(line)["decision"] for line in lines]
    assert decisions == ["allowed"]
    # the in-memory running total (the actual cap enforcement) is still
    # correct despite the on-disk audit trail write having failed
    assert ledger.remaining_usd() < 10.0


def test_thread_spawn_failure_does_not_leak_active_requests_counter(proxy, fake_upstream):
    """Round-3 `/code-review` finding: `ThreadingMixIn.process_request`
    spawns the worker thread and returns immediately -- if `Thread.start()`
    itself raises (FD/thread exhaustion), the compensating `end_request()`
    living inside the spawned thread's own `finally` never runs, since the
    thread never started. `_CostCapHTTPServer.process_request()` must catch
    that and decrement its own counter, or every future `stop()` blocks for
    the full drain timeout for no reason."""
    real_start = threading.Thread.start
    calls = {"n": 0}

    def flaky_start(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated thread spawn failure")
        return real_start(self)

    with patch.object(threading.Thread, "start", flaky_start):
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": []},
                          headers={"Authorization": "Bearer sk-fake"}, timeout=5)

    assert proxy._httpd.wait_for_idle(timeout=2), "_active_requests leaked after a simulated Thread.start() failure"


def test_socket_read_timeout_does_not_leak_active_requests_counter(ledger, fake_upstream, tmp_path):
    """Round-3 `/code-review` finding: a client that declares a
    `Content-Length` and then stalls mid-body used to block the handler
    thread forever in `self.rfile.read(length)`, with `begin_request()`
    already counted -- `stop()` would then always wait the full
    `drain_timeout_s` for a request that never even reached
    `ledger.reserve()`. A short handler-level socket timeout bounds this."""
    upstream_url = f"http://127.0.0.1:{fake_upstream.server_address[1]}"
    with patch.object(_CostCapRequestHandler, "timeout", 0.2):
        p = CostCapProxy(ledger, tmp_path / "call_log.jsonl", upstream_url=upstream_url)
        with p:
            sock = socket.create_connection(("127.0.0.1", p.port), timeout=5)
            try:
                body = b'{"model": "gpt-4o", "messages": []}'
                # Declare more bytes than are actually sent, then never send
                # the rest -- the handler thread blocks reading the body.
                request = (
                    f"POST {_CHAT_COMPLETIONS_PATH} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1\r\n"
                    f"Content-Length: {len(body) + 100}\r\n"
                    f"Content-Type: application/json\r\n\r\n"
                ).encode("ascii") + body
                sock.sendall(request)
                # Round-4 `/code-review` finding: a fixed `sleep()` before
                # asserting is a flaky margin under scheduler contention --
                # `wait_for_idle()` already polls internally (it's a
                # `threading.Event.wait()`), so give it a generous budget
                # directly instead of gambling on a short sleep finishing
                # first.
                assert p._httpd.wait_for_idle(timeout=5), (
                    "_active_requests leaked after a socket read timeout"
                )
            finally:
                sock.close()


def test_upstream_network_error_charges_worst_case_not_released(ledger, tmp_path):
    """Round-12 `/code-review` finding: a `requests.RequestException` here
    does not prove the upstream never received/processed/billed the
    request -- a `ReadTimeout` in particular fires only AFTER the request
    was fully sent and the server may have already started (or finished)
    generating a response. The old `release()` here assumed "never
    billed," which could silently under-track real spend for the rest of
    the run. Same fail-safe direction as the unverifiable-response-body
    branch: when the true outcome can't be verified, charge the worst-case
    reservation rather than assume nothing was spent."""
    # A closed loopback port: requests.post() raises a genuine
    # requests.exceptions.ConnectionError (a RequestException subclass),
    # no fake upstream server involved.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()
    upstream_url = f"http://127.0.0.1:{closed_port}"
    with CostCapProxy(ledger, tmp_path / "call_log.jsonl", upstream_url=upstream_url) as p:
        before = ledger.remaining_usd()
        resp = requests.post(_chat_url(p), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 1000},
                              headers={"Authorization": "Bearer sk-fake"})
        after = ledger.remaining_usd()
        assert resp.status_code == 502
        expected_worst_case = pricing.estimate_worst_case_cost_usd(
            model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
            max_output_tokens=1000,
        )
        assert before - after == pytest.approx(expected_worst_case, rel=1e-6)


def _client_post(proxy: CostCapProxy, payload: dict) -> int:
    """Raw `http.client` request, deliberately NOT `requests` -- these two
    tests patch `proxy_module.requests.post` (the proxy's OWN outbound
    call to the upstream). Since both this test module and `proxy.py` did
    `import requests` (the same module object, in the same process), a
    `requests`-based client call here would be patched too, defeating the
    point of only faking the proxy's upstream leg. Returns the HTTP status
    code."""
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    try:
        conn.request("POST", _CHAT_COMPLETIONS_PATH, body=body,
                     headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
        return conn.getresponse().status
    finally:
        conn.close()


def test_connect_timeout_releases_without_charging(proxy, ledger):
    """Round-13 `/code-review` finding: the round-12 blanket "charge
    worst case for any RequestException" fix has its own failure
    scenario -- a sustained connect-level outage makes every call fail
    this way, and charging worst-case for each one could exhaust the
    entire cap on calls that provably sent zero bytes. `ConnectTimeout`
    is the one `RequestException` subclass `requests` raises only when
    the TCP connect itself times out, before any request bytes could
    have been sent -- safe to release without reintroducing the
    under-counting risk the round-12 fix exists to close."""
    before = ledger.remaining_usd()
    with patch.object(proxy_module.requests, "post",
                       side_effect=requests.exceptions.ConnectTimeout("simulated connect timeout")):
        status = _client_post(proxy, {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                       "max_tokens": 1000})
    after = ledger.remaining_usd()
    assert status == 502
    assert after == pytest.approx(before)


def test_read_timeout_still_charges_worst_case_not_released(proxy, ledger):
    """Companion to the above: a `ReadTimeout` (fires only after the
    request was fully sent) is NOT split out the same way as
    `ConnectTimeout` -- it must stay in the fail-safe "charge worst case"
    branch, since the request may already have been billed server-side."""
    before = ledger.remaining_usd()
    with patch.object(proxy_module.requests, "post",
                       side_effect=requests.exceptions.ReadTimeout("simulated read timeout")):
        status = _client_post(proxy, {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                       "max_tokens": 1000})
    after = ledger.remaining_usd()
    assert status == 502
    expected_worst_case = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000,
    )
    assert before - after == pytest.approx(expected_worst_case, rel=1e-6)


def test_count_prompt_tokens_exception_refused_500_not_connection_crash(proxy, ledger):
    """Round-13 `/code-review` finding: `count_prompt_tokens()` calls
    `tiktoken.encoding_for_model()`/`get_encoding()`, which lazily
    downloads its BPE data file over the network if not already cached --
    in an environment where general internet access is restricted (this
    proxy exists specifically to fence off/monitor LLM egress), that
    download could fail. An uncaught exception here used to crash the
    handler thread before any HTTP response was written."""
    before = ledger.remaining_usd()
    with patch.object(proxy_module, "count_prompt_tokens", side_effect=RuntimeError("simulated tiktoken failure")):
        status = _client_post(proxy, {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    after = ledger.remaining_usd()
    assert status == 500
    assert after == pytest.approx(before)


def test_worst_case_charged_paths_persist_usage_verified_false(proxy, ledger, fake_upstream, tmp_path):
    """Round-14 `/code-review` finding: `commit()` calls made when the
    real outcome couldn't be verified (unparseable/incomplete response
    body) must persist `usage_verified: false` in the ledger's own audit
    trail, distinguishing them from a genuinely `usage`-verified record."""
    fake_upstream._responses.append((200, {"model": "gpt-4o", "choices": [{"message": {"content": "ok"}}]}))
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    on_disk = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert on_disk["calls"][0]["usage_verified"] is False


def test_normal_allowed_call_persists_usage_verified_true(proxy, fake_upstream, tmp_path):
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    on_disk = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert on_disk["calls"][0]["usage_verified"] is True


def test_invalid_content_length_header_refused_422_not_connection_crash(proxy):
    """Round-4 `/code-review` finding, reproduced live: a non-numeric
    `Content-Length` raised an uncaught `ValueError` from `int(...)`
    before the round-3 body-shape validation ever ran -- the client saw a
    broken connection with zero response bytes instead of a clean 4xx."""
    sock = socket.create_connection(("127.0.0.1", proxy.port), timeout=5)
    try:
        request = (
            f"POST {_CHAT_COMPLETIONS_PATH} HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\n"
            f"Content-Length: not-a-number\r\n"
            f"Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        sock.sendall(request)
        response = sock.recv(4096)
        assert response.startswith(b"HTTP/1.1 422"), response
    finally:
        sock.close()


def test_oversized_content_length_refused_413_before_reading_body(proxy):
    """Round-5 `/code-review` finding: a syntactically valid but absurdly
    large `Content-Length` used to reach `self.rfile.read(length)`
    unbounded. Declare a length far past `_MAX_REQUEST_BODY_BYTES` and
    confirm the handler refuses with a clean 413 BEFORE attempting to
    read that many bytes -- never send the (huge) body itself, so this
    test stays fast; if the refusal didn't happen before the read, the
    handler would hang waiting for bytes that never arrive instead of
    responding at all."""
    sock = socket.create_connection(("127.0.0.1", proxy.port), timeout=5)
    try:
        oversized = proxy_module._MAX_REQUEST_BODY_BYTES + 1
        request = (
            f"POST {_CHAT_COMPLETIONS_PATH} HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\n"
            f"Content-Length: {oversized}\r\n"
            f"Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        sock.sendall(request)
        response = sock.recv(4096)
        assert response.startswith(b"HTTP/1.1 413"), response
    finally:
        sock.close()


@pytest.mark.parametrize("bogus_max_tokens", [0, -5, float("nan"), float("inf")])
def test_invalid_max_tokens_forwarded_value_matches_what_the_reservation_was_sized_for(
    proxy, ledger, fake_upstream, bogus_max_tokens,
):
    """Round-4 `/code-review` finding, reproduced live: the default-ceiling
    injection only fired for `max_tokens is None`, so a non-finite,
    negative, or zero value was forwarded to the (fake, here) upstream API
    completely unmodified while the reservation was quietly sized using
    the 4096-token default ceiling -- violating the module's own stated
    invariant that the real call is bounded to match what was reserved."""
    fake_upstream._responses.append((200, _usage_response()))
    # `requests`' own `json=` param encodes with `allow_nan=False` and
    # refuses to serialize NaN/inf at all -- send a hand-encoded body via
    # `data=` instead (Python's own `json.dumps` allows NaN/Infinity by
    # default, and `_handle_post`'s `json.loads` accepts them symmetrically)
    # so this test can actually reach the proxy with those values.
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": bogus_max_tokens}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_tokens"] == pricing.DEFAULT_MAX_OUTPUT_TOKENS_CEILING


def test_valid_positive_max_tokens_still_left_untouched_after_the_fix(proxy, fake_upstream):
    """Companion to the above -- confirms the fix didn't regress the
    already-covered case of a genuinely valid caller-supplied value."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                           "max_tokens": 123},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_tokens"] == 123


@pytest.mark.parametrize("bogus_max_completion_tokens", [0, -5, float("nan"), float("inf")])
def test_invalid_max_completion_tokens_only_forwarded_value_matches_the_reservation(
    proxy, fake_upstream, bogus_max_completion_tokens,
):
    """Round-8 `/code-review` finding: the parametrized invalid-value test
    above only ever exercises `max_tokens` alone -- nothing separately
    covers the `max_completion_tokens`-only branch, which since round 7
    is a textually distinct code path (the `max_output_field is None`
    vs. `max_output_field == "max_completion_tokens"` branches). A future
    edit to that branch alone could silently stop correcting an invalid
    completion-tokens-only value without any test failing."""
    fake_upstream._responses.append((200, _usage_response()))
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_completion_tokens": bogus_max_completion_tokens}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == pricing.DEFAULT_MAX_OUTPUT_TOKENS_CEILING


def test_both_max_tokens_keys_present_max_tokens_is_dropped_completion_key_resolved(proxy, fake_upstream):
    """Round-7 `/code-review` finding: forwarding BOTH
    `max_completion_tokens` and `max_tokens` in one request (round 6's
    fix, forcing both to match) is itself risky -- some model families
    reject requests carrying both parameters, and none of this codebase's
    8 real call sites ever set `max_tokens` in the first place. When both
    keys are present, `max_completion_tokens` is authoritative and
    `max_tokens` must be dropped from the forwarded body entirely, not
    forced to match. Here `max_completion_tokens: None` resolves to the
    module default, and `max_tokens: 5000` (the non-authoritative,
    legacy key) never reaches the upstream API at all."""
    fake_upstream._responses.append((200, _usage_response()))
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_completion_tokens": None, "max_tokens": 5000}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == pricing.DEFAULT_MAX_OUTPUT_TOKENS_CEILING
    assert "max_tokens" not in forwarded


def test_both_max_tokens_keys_present_max_completion_tokens_value_wins_unchanged(proxy, fake_upstream):
    """Companion to the above: when `max_completion_tokens` already
    carries a valid, positive value, that value is left untouched (NOT
    upsized to match a larger `max_tokens` sitting alongside it -- round-6
    `/code-review`'s "take the max" fix could silently inflate a caller's
    intentionally small, cost-conscious limit; dropping the non-
    authoritative key instead avoids that entirely) and `max_tokens` is
    still dropped."""
    fake_upstream._responses.append((200, _usage_response()))
    resp = requests.post(_chat_url(proxy),
                          json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                "max_completion_tokens": 100, "max_tokens": 5000},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == 100
    assert "max_tokens" not in forwarded


def test_both_max_tokens_keys_present_nan_in_the_dropped_key_never_forwarded(proxy, fake_upstream):
    """Companion to the above: a literal NaN sitting in the
    non-authoritative `max_tokens` key must never reach the upstream API
    -- since that key is unconditionally dropped whenever
    `max_completion_tokens` is also present, its value never even needs
    inspecting to achieve that."""
    fake_upstream._responses.append((200, _usage_response()))
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_completion_tokens": 100, "max_tokens": float("nan")}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_completion_tokens"] == 100
    assert "max_tokens" not in forwarded


def test_call_log_write_failure_does_not_block_an_already_committed_response(proxy, ledger, fake_upstream,
                                                                              tmp_path):
    """Round-4 `/code-review` finding, reproduced live: `_CallLogWriter
    .append()` for the "allowed" decision runs AFTER `ledger.commit()` has
    already recorded real spend, immediately before the 200 is written
    back. An `OSError` there used to propagate and prevent the response --
    the client would see a broken connection for a call that had already
    succeeded and been billed, risking a client-side retry that pays for
    the same completion twice."""
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=100, completion_tokens=50)))
    before = ledger.remaining_usd()
    # Patch the underlying `open()` call `append()` uses, not `append()`
    # itself -- mocking the whole method would bypass the internal
    # try/except this test exists to exercise and give a false pass.
    with patch("slr_pipeline.cost_cap.proxy.open", side_effect=OSError("disk full (simulated)")):
        resp = requests.post(_chat_url(proxy),
                              json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                              headers={"Authorization": "Bearer sk-fake"})
    after = ledger.remaining_usd()
    assert resp.status_code == 200
    assert before > after  # the call was still billed despite the log-write failure


def test_concurrent_requests_never_overshoot_a_tight_cap(ledger, fake_upstream, tmp_path):
    """Mirrors `screening.py`'s `ThreadPoolExecutor` concurrency pattern:
    many simultaneous calls against a budget that can only actually afford
    a handful of them must never let more through than the cap allows."""
    fake_upstream._default = (200, _usage_response(prompt_tokens=1, completion_tokens=1))
    tiny_ledger = CostLedger(tmp_path / "tiny_ledger.json", cap_usd=0.0001)
    upstream_url = f"http://127.0.0.1:{fake_upstream.server_address[1]}"
    with CostCapProxy(tiny_ledger, tmp_path / "call_log.jsonl", upstream_url=upstream_url) as p:
        results = []
        lock = threading.Lock()
        total_attempts = 0

        def worker():
            # A loopback socket serving 20 concurrent short-lived
            # HTTP/1.1-close connections occasionally hits a transport-level
            # `ConnectionResetError` on this machine (observed independently
            # of any cap-enforcement logic -- a Windows/localhost TCP-RST
            # race under socket churn, not a proxy bug: the invariant this
            # test actually cares about, never overshooting the cap, is
            # asserted below via `tiny_ledger.remaining_usd()` regardless of
            # whether any one client happened to see its own response). One
            # retry is harmless here: worst case it consumes a second
            # `reserve()`/`release()` cycle against an already-refused call.
            #
            # Round-5 `/code-review` finding: the retry itself, unbounded,
            # could quietly mask a systemic double-billing regression (each
            # worker silently retrying real, billed calls). Count every
            # attempt across all workers and assert a hard ceiling on the
            # total below, so that kind of regression would still fail
            # this test even though it wouldn't change `allowed`/`refused`.
            nonlocal total_attempts
            for attempt in range(2):
                with lock:
                    total_attempts += 1
                try:
                    r = requests.post(_chat_url(p), json={"model": "gpt-4o",
                                                            "messages": [{"role": "user", "content": "hi"}],
                                                            "max_tokens": 1000},
                                       headers={"Authorization": "Bearer sk-fake"}, timeout=10)
                    with lock:
                        results.append(r.status_code)
                    return
                except requests.exceptions.RequestException:
                    if attempt == 1:
                        raise

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    allowed = results.count(200)
    refused = results.count(402)
    assert allowed + refused == 20
    assert allowed < 20  # the tiny cap must have actually refused some
    assert tiny_ledger.remaining_usd() >= 0.0
    # 20 workers x at most 2 attempts each = 40 hard ceiling; a healthy run
    # should see very few (if any) retries, so 25 leaves headroom for the
    # known transport flake without masking a real double-billing bug.
    assert total_attempts <= 25


def test_tokenization_internal_error_logs_loudly(proxy, caplog):
    """Round-15 `/code-review` finding: round 14 added a `logger.error()`
    call on the tokenization/pricing internal-error path so an unattended
    live run's console/log stream would show a systemic failure here (a
    corrupted local `tiktoken` cache, say) instead of it only ever showing
    up as a silent stream of 500s in the call-log JSONL -- but no test
    actually asserted the log call happens, mirroring the gap
    `test_cost_cap_ledger.py::test_commit_actual_usd_exceeding_reservation_still_commits_but_warns`
    already closed for the ledger's own `logger.warning()` call."""
    with patch.object(proxy_module, "count_prompt_tokens", side_effect=RuntimeError("simulated tiktoken failure")), \
            caplog.at_level("ERROR"):
        status = _client_post(proxy, {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 500
    assert any("internal error sizing the reservation" in rec.message for rec in caplog.records)


def test_response_format_estimation_internal_error_logs_loudly(proxy, caplog):
    """Round-1 `/code-review` finding (Track-3 proxy-fix cycle): sibling of
    `test_tokenization_internal_error_logs_loudly` above, for the newer
    `estimate_response_format_tokens()` code path -- a non-`ValueError`
    failure there (e.g. the same tiktoken network/OS error round 13 found
    for `count_prompt_tokens()`) must be caught, logged loudly, and
    surfaced as a clean 500, never left to crash the handler thread."""
    with patch.object(proxy_module, "estimate_response_format_tokens",
                       side_effect=RuntimeError("simulated tiktoken failure")), \
            caplog.at_level("ERROR"):
        status = _client_post(proxy, {
            "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "response_format": _STRICT_JSON_SCHEMA_RESPONSE_FORMAT,
        })
    assert status == 500
    assert any("internal error estimating response_format tokens" in rec.message for rec in caplog.records)


def test_reserve_internal_valueerror_logs_loudly(proxy, caplog):
    """Companion to the above, for the sibling `reserve()` `ValueError`
    internal-error path round 14 also added a `logger.error()` call to."""
    with patch.object(CostLedger, "reserve", side_effect=ValueError("simulated internal invariant violation")), \
            caplog.at_level("ERROR"):
        status = _client_post(proxy, {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 500
    assert any("internal error reserving budget" in rec.message for rec in caplog.records)


def test_n_greater_than_one_multiplies_worst_case_output_cost(fake_upstream, tmp_path):
    """Round-15 `/code-review` finding: OpenAI bills `n` independently-billed
    completions, each at the full output rate -- the pre-flight reservation
    used to assume `n=1` unconditionally, so a request with e.g. `n=4` was
    sized/gated as if it were a single completion, letting the real call's
    worst case exceed what was actually reserved. Prove the reservation
    itself changed (not just the eventually-committed actual cost, which is
    always the same small `usage`-verified figure regardless of `n`): a cap
    sized strictly between the `n=1` and `n=4` worst cases must admit an
    `n=1` request but refuse the identical request with `n=4`."""
    worst_case_n1 = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000, n=1,
    )
    worst_case_n4 = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000, n=4,
    )
    assert worst_case_n4 > worst_case_n1  # sanity check on the fixture math itself
    cap_usd = (worst_case_n1 + worst_case_n4) / 2
    upstream_url = f"http://127.0.0.1:{fake_upstream.server_address[1]}"

    fake_upstream._responses.append((200, _usage_response()))
    ledger_n1 = CostLedger(tmp_path / "ledger_n1.json", cap_usd=cap_usd)
    with CostCapProxy(ledger_n1, tmp_path / "call_log_n1.jsonl", upstream_url=upstream_url) as p:
        resp_n1 = requests.post(_chat_url(p), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                     "max_tokens": 1000, "n": 1},
                                 headers={"Authorization": "Bearer sk-fake"})
    assert resp_n1.status_code == 200  # n=1's worst case fits the cap

    fake_upstream._responses.append((200, _usage_response()))
    ledger_n4 = CostLedger(tmp_path / "ledger_n4.json", cap_usd=cap_usd)
    with CostCapProxy(ledger_n4, tmp_path / "call_log_n4.jsonl", upstream_url=upstream_url) as p:
        resp_n4 = requests.post(_chat_url(p), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                     "max_tokens": 1000, "n": 4},
                                 headers={"Authorization": "Bearer sk-fake"})
    assert resp_n4.status_code == 402  # the identical request with n=4 does not fit the same cap


def test_invalid_n_forwarded_value_matches_what_the_reservation_was_sized_for(proxy, fake_upstream):
    """Companion to `test_invalid_max_tokens_forwarded_value_matches_...`:
    an `n` that `resolve_n()` treats as invalid (non-finite, non-positive)
    must be rewritten in the forwarded body to the value actually reserved
    (1), not left forwarded unmodified."""
    fake_upstream._responses.append((200, _usage_response()))
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1000, "n": -3}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    assert resp.status_code == 200
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["n"] == 1


def test_non_numeric_n_refused_422(proxy, fake_upstream):
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                                  "n": "many"},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert len(fake_upstream.received_bodies) == 0


def test_unexpected_exception_after_200_charges_worst_case_not_released(proxy, ledger, fake_upstream):
    """Round-15 `/code-review` finding: the outer `finally` block used to
    unconditionally `release()` whenever nothing had set `resolved = True`
    yet -- correct for an exception before the upstream ever responded, but
    wrong for one AFTER a real 200 came back (the call may genuinely have
    been billed server-side). Simulate that by making `actual_cost_usd()`
    itself blow up with something the `usage_verified` try/except (which
    only catches `ValueError`) does not catch."""
    # Nothing in `_handle_post` converts an exception raised THIS late
    # (after the response-writing section starts) into a clean HTTP
    # status -- unlike the tokenization/`reserve()` internal-error paths,
    # this one isn't wrapped in its own try/except, so the connection is
    # simply torn down without a response ever being written. That's a
    # pre-existing, separate gap (a clean 500 here would be nicer) -- this
    # test only asserts the ledger bookkeeping this round's fix is about:
    # worst-case charged, not released, even though the client sees a
    # broken connection rather than a response.
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=100, completion_tokens=50)))
    before = ledger.remaining_usd()
    with patch.object(proxy_module.pricing_mod, "actual_cost_usd",
                       side_effect=RuntimeError("simulated unexpected post-200 failure")):
        with pytest.raises(requests.exceptions.RequestException):
            requests.post(_chat_url(proxy), json={"model": "gpt-4o",
                                                   "messages": [{"role": "user", "content": "hi"}],
                                                   "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    after = ledger.remaining_usd()
    expected_worst_case = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000,
    )
    # worst-case was CHARGED, not released -- spend must have decreased by
    # roughly the worst-case amount, not stayed at `before`.
    assert before - after == pytest.approx(expected_worst_case, rel=1e-6)


def test_unexpected_exception_after_200_logs_and_records_the_real_exception(
    proxy, fake_upstream, tmp_path, caplog
):
    """Round-42 `/code-review` finding: the `finally` block's post-200
    fail-safe branch used to charge worst-case and log a decision string,
    but never captured or logged the actual exception that put the
    handler here -- unlike every other internal-error path in this
    module, which all call `logger.error(..., exc_info=True)` and record
    `error=str(exc)`. `sys.exc_info()` must still see the in-flight
    exception from inside this bare `finally:` (no enclosing `except`)."""
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=100, completion_tokens=50)))
    with caplog.at_level("ERROR"):
        with patch.object(proxy_module.pricing_mod, "actual_cost_usd",
                           side_effect=RuntimeError("simulated unexpected post-200 failure")):
            with pytest.raises(requests.exceptions.RequestException):
                requests.post(_chat_url(proxy), json={"model": "gpt-4o",
                                                       "messages": [{"role": "user", "content": "hi"}],
                                                       "max_tokens": 1000},
                              headers={"Authorization": "Bearer sk-fake"})
    assert any("simulated unexpected post-200 failure" in rec.message for rec in caplog.records)
    lines = (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    charged = next(e for e in entries if e["decision"] == "charged_worst_case_unexpected_exception_after_200")
    assert "simulated unexpected post-200 failure" in charged["error"]


def test_extreme_max_output_and_n_overflow_refused_500_not_connection_crash(proxy, ledger, fake_upstream):
    """Round-17 `/code-review` finding: `estimate_worst_case_cost_usd()`'s
    round-16 arithmetic-overflow `ValueError` (huge-but-individually-valid
    `max_completion_tokens`/`n` multiplying out to `inf`) only had a
    `pricing.py`-level unit test -- unlike every OTHER internal-error path
    in this handler (the `reserve()` ValueError, the tokenization
    exception), which each have a dedicated request-level test pinning
    that the `except Exception` block around this call still converts it
    into a clean 500, not a torn-down connection."""
    before = ledger.remaining_usd()
    raw_body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                            "max_completion_tokens": 1e300, "n": 1e300}).encode("utf-8")
    resp = requests.post(_chat_url(proxy), data=raw_body,
                          headers={"Authorization": "Bearer sk-fake", "Content-Type": "application/json"})
    after = ledger.remaining_usd()
    assert resp.status_code == 500
    assert len(fake_upstream.received_bodies) == 0  # never forwarded
    assert after == pytest.approx(before)  # nothing reserved/committed


def test_missing_n_stays_absent_from_forwarded_body(proxy, fake_upstream):
    """Round-18 `/code-review` finding: nothing pinned the exact invariant
    round 17's revert restores -- that a caller who never supplied `n` at
    all gets `n` left OUT of the forwarded body (trusting the real
    upstream's own default of 1), not injected. Without this test, a
    future edit reintroducing round-16's "always inject n" behavior would
    pass the rest of the suite unnoticed."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert "n" not in forwarded


def test_finally_second_commit_after_partial_commit_does_not_double_charge(proxy, ledger, fake_upstream, caplog):
    """Round-19 `/code-review` finding: if `commit()` raises something
    AFTER it has already popped the reservation and updated the ledger's
    in-memory committed total, `resolved` stays False and the outer
    `finally` (seeing `upstream_returned_200=True`) tries to `commit()`
    AGAIN for the same `call_id` -- which used to raise its OWN
    `ValueError('... no open reservation')` since the first
    (partially-succeeded) call already popped it, REPLACING the original
    exception Python was propagating out of `finally` and masking the
    real root cause. Simulate the partial-success-then-raise shape by
    monkeypatching `CostLedger.commit()` itself, not `_persist_locked()`:
    round 43's fix wrapped `commit()`'s own persist call in `except
    Exception`, so a persist-originated exception (e.g. `_persist_
    locked()` raising `RecursionError`/`MemoryError`) can no longer
    escape a real `commit()` call -- this test's raise is a stand-in for
    some other, non-persistence cause of the same partial-success shape,
    matching how `test_commit_persist_exotic_exception_does_not_escape_
    or_lose_committed_spend` in `test_cost_cap_ledger.py` now covers the
    persist-specific case directly against `_persist_locked()`."""
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=100, completion_tokens=50)))
    real_commit = CostLedger.commit
    call_count = {"n": 0}

    def flaky_commit(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_commit(self, *args, **kwargs)  # really pops the reservation, updates committed spend
            raise RecursionError("simulated failure outside commit()'s own exception handling")
        return real_commit(self, *args, **kwargs)

    with patch.object(CostLedger, "commit", flaky_commit), caplog.at_level("ERROR"):
        with pytest.raises(requests.exceptions.RequestException):
            requests.post(_chat_url(proxy), json={"model": "gpt-4o",
                                                   "messages": [{"role": "user", "content": "hi"}],
                                                   "max_tokens": 1000},
                          headers={"Authorization": "Bearer sk-fake"})
    # the FIRST (real) commit already charged the actual usage-verified
    # cost -- the finally block's second attempt must not double-charge,
    # and must not itself crash trying.
    expected_actual = pricing.actual_cost_usd(model="gpt-4o", prompt_tokens=100, completion_tokens=50,
                                              cached_tokens=0)
    assert ledger.remaining_usd() == pytest.approx(10.0 - expected_actual)
    assert call_count["n"] == 2  # the finally block's second attempt did run
    assert any("no open reservation" in rec.message for rec in caplog.records)


def test_negative_upstream_usage_clamped_before_persisting_to_ledger(proxy, ledger, fake_upstream):
    """Round-20 `/code-review` finding: `is_finite_number()` rejects NaN/
    inf/huge-int/wrong-type but NOT a negative value -- a negative
    `usage.prompt_tokens`/`completion_tokens`/`prompt_tokens_details.
    cached_tokens` from a buggy or adversarial upstream used to reach
    `ledger.commit()` unclamped, so `actual_cost_usd()`'s own internal
    clamp (round-2 fix) kept the DOLLAR figure correct while the ledger's
    persisted audit-trail record (`_calls[]`) still stored the raw
    negative counts verbatim -- corrupting the exact record this module
    exists to produce. The clamp must happen before `ledger.commit()` is
    called, not just inside `actual_cost_usd()`."""
    fake_upstream._responses.append((200, _usage_response(prompt_tokens=-999999, completion_tokens=-50,
                                                           cached_tokens=-10)))
    resp = requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                          headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    record = ledger._calls[-1]
    assert record.prompt_tokens == 0
    assert record.completion_tokens == 0
    # actual_usd must match what a genuinely all-zero usage would cost --
    # confirms the clamp, not just a coincidentally-correct dollar figure.
    assert record.actual_usd == pytest.approx(pricing.actual_cost_usd(model="gpt-4o", prompt_tokens=0,
                                                                      completion_tokens=0, cached_tokens=0))


def test_fractional_max_output_tokens_resolved_to_int_before_forwarding(proxy, fake_upstream):
    """Round-20 `/code-review` finding: `resolve_max_output_tokens()` used
    to return a valid-but-fractional value (e.g. `500.7`) unchanged --
    `proxy.py` only rewrites the forwarded body's `max_output_field` when
    it differs from the resolved value, so `500.7 == 500.7` meant the
    fractional value was forwarded byte-for-byte to the real OpenAI API,
    which expects an integer token count and would likely reject it with
    a 400 -- a spurious mid-live-run failure for an otherwise legitimate
    call. The resolved value forwarded upstream must always be an int.
    Round-23 `/code-review` finding changed the rounding direction from
    truncation to `math.ceil` (the safe-for-a-worst-case-reservation
    direction, and the fix for a separate zero-collapse bug) -- `500.7`
    now resolves to `501`, not `500`."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                          "max_tokens": 500.7},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_tokens"] == 501
    assert isinstance(forwarded["max_tokens"], int)


def test_whole_number_float_max_tokens_still_coerced_to_int(proxy, fake_upstream):
    """Round-21 `/code-review` finding, reproduced live: round 20's `!=`-
    based rewrite check missed a caller-supplied WHOLE-NUMBER float (e.g.
    `4096.0`) that already compares numerically equal to the resolved int
    (`4096.0 != 4096` is False in Python), so `needs_rewrite` stayed False
    and the original float literal was forwarded verbatim -- the exact
    "real API expects an integer, gets a float, spurious 400" failure
    mode round 20's fractional-value fix was supposed to close entirely.
    Companion to `test_fractional_max_output_tokens_resolved_to_int_
    before_forwarding` above, which only covered a genuinely-fractional
    value and did not catch this."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                          "max_tokens": 4096.0},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_tokens"] == 4096
    assert isinstance(forwarded["max_tokens"], int)


def test_whole_number_float_n_still_coerced_to_int(proxy, fake_upstream):
    """Same whole-number-float gap as the `max_tokens` case above, for the
    `n` field's rewrite check (round-21 `/code-review` finding)."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                          "n": 1.0},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["n"] == 1
    assert isinstance(forwarded["n"], int)


def test_chunked_request_body_refused_with_clear_message_not_misread_as_empty(proxy):
    """Round-21 `/code-review` finding: a `Transfer-Encoding: chunked`
    request has no `Content-Length` header -- the `or "0"` fallback used
    to silently treat that as a zero-length body, failing further down
    with a confusing, unrelated 'request body is not valid JSON' 422
    instead of naming the real problem. Drive this directly against the
    raw socket (the `requests` library would just chunk-encode transparently
    and this proxy's real call sites never send chunked bodies, so there's
    no higher-level client to reproduce this through)."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        body_chunk = json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8")
        conn.send(b"%x\r\n%s\r\n0\r\n\r\n" % (len(body_chunk), body_chunk))
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 422
        assert "chunked" in payload["error"]["message"].lower()
    finally:
        conn.close()


def test_chunked_via_two_separate_transfer_encoding_header_lines_also_refused(proxy):
    """Round-22 `/code-review` finding: `self.headers.get("Transfer-
    Encoding")` only returns the FIRST value of a header name that
    legally appears on more than one line (e.g. `Transfer-Encoding: gzip`
    followed by a separate `Transfer-Encoding: chunked` line) -- round
    21's single-`.get()` check missed this two-header-line case entirely,
    falling through to the pre-round-21 confusing 'not valid JSON' 422.
    `.get_all()` must be used instead."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Transfer-Encoding", "gzip")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        body_chunk = json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8")
        conn.send(b"%x\r\n%s\r\n0\r\n\r\n" % (len(body_chunk), body_chunk))
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 422
        assert "chunked" in payload["error"]["message"].lower()
    finally:
        conn.close()


def test_multimodal_list_content_refused_not_silently_underestimated(proxy):
    """Round-22 `/code-review` finding: `count_prompt_tokens()` stringifies
    non-string message `content` (e.g. a multimodal list of parts
    carrying an `image_url`) via `json.dumps()` and tokenizes that short
    JSON/URL text -- which does not reflect OpenAI's real, much higher,
    resolution-dependent image-token billing, silently turning the
    pre-flight worst-case estimate into an UNDER-estimate. No real call
    site sends this shape today; refuse it outright rather than reserve
    an estimate this module can't make safely."""
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o",
              "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "list" in resp.json()["error"]["message"].lower()


@pytest.mark.parametrize("field", ["tools", "functions"])
def test_top_level_tools_or_functions_refused_not_silently_underestimated(proxy, field):
    """Round-23 `/code-review` finding: a top-level `tools`/`functions`
    field (tool/function definitions, billed as real input tokens --
    sometimes hundreds to thousands of them for a large JSON schema) was
    never read by `count_prompt_tokens()` at all, silently under-
    estimating the pre-flight reservation. No real call site sends this
    field today; refuse it outright rather than guess."""
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
              field: [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "tools" in resp.json()["error"]["message"].lower() or "functions" in resp.json()["error"]["message"].lower()


@pytest.mark.parametrize("field", ["tool_calls", "function_call"])
def test_message_level_tool_calls_refused_not_silently_underestimated(proxy, field):
    """Round-23 `/code-review` finding: an assistant message with
    `content=null` and a `tool_calls`/`function_call` field (real, billed
    prompt tokens whenever that message is replayed in a later turn) was
    tokenized as if its content were the 4-character string `"null"` --
    the real tool-call payload's tokens were never counted. No real
    call site sends this shape today; refuse it outright."""
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o",
              "messages": [{"role": "assistant", "content": None,
                            field: [{"function": {"name": "f", "arguments": "{}"}}]}]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "tool_calls" in resp.json()["error"]["message"].lower() or "function_call" in resp.json()["error"]["message"].lower()


def test_message_name_field_counted_towards_prompt_tokens(proxy, fake_upstream):
    """Round-23 `/code-review` finding: a message's `name` key bills its
    own real prompt tokens on OpenAI's side but was never read by
    `count_prompt_tokens()` at all -- silently under-counting the
    pre-flight estimate by however many tokens the identifier is worth."""
    with_name = count_prompt_tokens("gpt-4o", [{"role": "user", "name": "a_fairly_long_identifier", "content": "hi"}])
    without_name = count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}])
    assert with_name > without_name


def test_fractional_max_output_tokens_below_one_rounds_up_not_down_to_zero(proxy, fake_upstream):
    """Round-23 `/code-review` finding, reproduced live: round 20's
    truncating `int(x)` fix rounded any `x` in `(0, 1)` (e.g. `0.5`) DOWN
    to `0` -- forwarded upstream as `max_tokens=0`, which a real API is
    documented to reject outright (minimum 1), a spurious 400 for what
    looked like a safely-over-reserved call. Also, re-resolving that
    already-resolved `0` a SECOND time (as `estimate_worst_case_cost_usd()`
    does internally) hit the `<= 0` fallback branch and silently sized the
    reservation off the 4096-token default instead -- a forwarded-vs-
    reserved divergence. Rounding UP (floored at 1) instead fixes both:
    the forwarded value is always a real API-valid positive integer, and
    resolving it a second time is a no-op."""
    fake_upstream._responses.append((200, _usage_response()))
    requests.post(_chat_url(proxy), json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                          "max_tokens": 0.5},
                  headers={"Authorization": "Bearer sk-fake"})
    forwarded = fake_upstream.received_bodies[0]
    assert forwarded["max_tokens"] == 1


@pytest.mark.parametrize("field", ["tools", "functions"])
def test_top_level_tools_refused_even_when_messages_key_absent(proxy, field):
    """Round-24 `/code-review` finding, reproduced live: round 23's
    `tools`/`functions` refusal used to live inside the `else:` branch of
    the `messages`-presence check, so it only ran when `"messages"` was
    present in the body at all -- a request carrying `tools`/`functions`
    but NO `messages` key skipped the refusal entirely and was priced off
    an empty message list, the exact silent-under-estimate the refusal
    exists to close, just reached via a different path. This body
    deliberately omits `messages` entirely."""
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", field: [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "tools" in resp.json()["error"]["message"].lower() or "functions" in resp.json()["error"]["message"].lower()


_STRICT_JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "db_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"pubmed": {"type": "string"}, "scopus": {"type": "string"}},
            "required": ["pubmed", "scopus"],
            "additionalProperties": False,
        },
    },
}


def test_response_format_json_schema_forwarded_unchanged_and_counted(proxy, fake_upstream):
    """Track-3 authorization (2026-09-12): round 25's blanket refusal of
    `response_format={"type": "json_schema", ...}` is superseded now that a
    real call site needs Structured Outputs -- a well-formed strict schema
    must be forwarded to the real upstream byte-for-byte (never rewritten,
    unlike `max_tokens`/`n`), and its tokens must be counted, not ignored."""
    fake_upstream._responses.append((200, _usage_response()))
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": _STRICT_JSON_SCHEMA_RESPONSE_FORMAT,
    }
    resp = requests.post(_chat_url(proxy), json=body, headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 200
    assert len(fake_upstream.received_bodies) == 1
    assert fake_upstream.received_bodies[0]["response_format"] == _STRICT_JSON_SCHEMA_RESPONSE_FORMAT


def test_response_format_json_schema_reservation_increases_by_estimated_schema_cost(ledger, fake_upstream, tmp_path):
    """Gate: the reservation for a call carrying a `json_schema`
    `response_format` must be sized at least as large as
    `count_prompt_tokens(...)` alone PLUS `estimate_response_format_tokens(
    ...)` -- proven here by setting the ledger cap exactly between the two
    worst-case costs, so the plain call is allowed and the schema-carrying,
    otherwise-identical call is refused for exceeding the (larger)
    reservation."""
    model = "gpt-4o"
    messages = [{"role": "user", "content": "hi"}]
    plain_tokens = count_prompt_tokens(model, messages)
    schema_tokens = estimate_response_format_tokens(model, _STRICT_JSON_SCHEMA_RESPONSE_FORMAT)
    assert schema_tokens > 0
    max_output = 16  # tiny, real default ceiling is irrelevant to this comparison
    plain_worst_case = pricing.estimate_worst_case_cost_usd(
        model=model, prompt_tokens=plain_tokens, max_output_tokens=max_output, n=1,
    )
    schema_worst_case = pricing.estimate_worst_case_cost_usd(
        model=model, prompt_tokens=plain_tokens + schema_tokens, max_output_tokens=max_output, n=1,
    )
    assert schema_worst_case > plain_worst_case
    cap = (plain_worst_case + schema_worst_case) / 2
    boundary_ledger = CostLedger(tmp_path / "boundary_ledger.json", cap_usd=cap)
    upstream_url = f"http://127.0.0.1:{fake_upstream.server_address[1]}"
    with CostCapProxy(boundary_ledger, tmp_path / "boundary_call_log.jsonl", upstream_url=upstream_url) as p:
        fake_upstream._responses.append((200, _usage_response()))
        plain_resp = requests.post(
            _chat_url(p),
            json={"model": model, "messages": messages, "max_tokens": max_output},
            headers={"Authorization": "Bearer sk-fake"},
        )
        assert plain_resp.status_code == 200

        schema_resp = requests.post(
            _chat_url(p),
            json={
                "model": model, "messages": messages, "max_tokens": max_output,
                "response_format": _STRICT_JSON_SCHEMA_RESPONSE_FORMAT,
            },
            headers={"Authorization": "Bearer sk-fake"},
        )
        assert schema_resp.status_code == 402
        assert len(fake_upstream.received_bodies) == 1  # the schema-carrying call was never forwarded


@pytest.mark.parametrize(
    "bad_json_schema, expected_message_fragment",
    [
        ({"schema": {}}, "name"),
        ({"name": "x"}, "schema"),
        ({"name": "x", "schema": {}, "strict": "yes"}, "strict"),
        ({"name": 123, "schema": {}}, "name"),
        ({"name": "x", "schema": "not-an-object"}, "schema"),
        ({"name": "x", "schema": {"$ref": "#/$defs/thing"}}, "$ref"),
        ({"name": "x", "schema": {"$defs": {"thing": {"type": "string"}}}}, "$ref"),
        ({"name": "x", "schema": {"type": "object",
                                   "properties": {"a": {"$ref": "#/$defs/thing"}}}}, "$ref"),
        # Round-3 `/code-review` finding: a property literally named
        # "properties" (or "definitions"/"patternProperties") whose own
        # subschema genuinely uses `$ref` must still be caught -- the
        # name-map flag must reset to keyword-parsing mode one level
        # below a caller-chosen name, not stay armed by string match.
        ({"name": "x", "schema": {"properties": {"properties": {"$ref": "#/foo"}}}}, "$ref"),
        ({"name": "x", "schema": {"properties": {"definitions": {"$ref": "#/foo"}}}}, "$ref"),
    ],
)
def test_response_format_json_schema_malformed_refused_before_dispatch(
    proxy, fake_upstream, bad_json_schema, expected_message_fragment,
):
    """Gate: a malformed `json_schema` object (missing/wrong-typed `name`,
    `schema`, or `strict`) must be refused before the request ever reaches
    the real upstream -- never a 500, never silently forwarded."""
    resp = requests.post(
        _chat_url(proxy),
        json={
            "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": bad_json_schema},
        },
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert expected_message_fragment in resp.json()["error"]["message"].lower()
    assert len(fake_upstream.received_bodies) == 0


@pytest.mark.parametrize(
    "benign_schema",
    [
        # Round-2 `/code-review` finding (Track-3 proxy-fix cycle): a
        # property literally named "definitions" (e.g. an extracted-text
        # section), unrelated to JSON Schema's `$defs`/legacy
        # `definitions` reference keyword, must NOT be refused -- the
        # earlier `json.dumps()`-substring check false-positived on this.
        {"type": "object", "properties": {"definitions": {"type": "string"}},
         "required": ["definitions"]},
        # An enum whose legitimate values happen to equal exactly "$ref"
        # / "$defs" as plain strings, not as object keys.
        {"type": "object", "properties": {"tag": {"enum": ["$ref", "$defs", "other"]}}},
        # Round-3 `/code-review` finding: `dependentSchemas` has the same
        # "children are names, not keywords" shape as `properties` -- a
        # property literally named "definitions" under it is benign.
        {"type": "object", "dependentSchemas": {"definitions": {"type": "string"}}},
    ],
)
def test_response_format_json_schema_benign_reference_lookalike_not_refused(
    proxy, fake_upstream, benign_schema,
):
    """A schema that merely uses "$ref"/"$defs"/"definitions" as an
    ordinary property name or string value (not an actual JSON Schema
    reference key) must be forwarded normally, not refused."""
    resp = requests.post(
        _chat_url(proxy),
        json={
            "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": benign_schema, "strict": True},
            },
        },
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 200
    assert len(fake_upstream.received_bodies) == 1


def test_response_format_json_schema_oversized_refused_before_dispatch(proxy, fake_upstream):
    """Gate: a schema so large its conservatively-padded token estimate
    exceeds `_MAX_RESPONSE_FORMAT_SCHEMA_TOKENS` is refused outright
    (can't be safely estimated) rather than trusting the padding
    multiplier to cover an unbounded size -- never forwarded."""
    huge_schema = {
        "type": "object",
        "properties": {"field": {"type": "string", "description": "x " * (_MAX_RESPONSE_FORMAT_SCHEMA_TOKENS * 4)}},
    }
    resp = requests.post(
        _chat_url(proxy),
        json={
            "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "big", "schema": huge_schema}},
        },
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "too large" in resp.json()["error"]["message"].lower()
    assert len(fake_upstream.received_bodies) == 0


def test_response_format_json_object_still_allowed(proxy, fake_upstream):
    """Regression: `response_format={"type": "json_object"}` (the shape
    every real call site used before Structured Outputs) must remain
    allowed and unaffected by the `json_schema` estimation path added
    above -- it carries no embedded schema, so no extra tokens are added."""
    fake_upstream._responses.append((200, _usage_response()))
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
              "response_format": {"type": "json_object"}},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 200


def test_empty_content_length_header_gives_clear_diagnostic_not_misread_as_empty_body(proxy):
    """Round-25 `/code-review` finding: `self.headers.get("Content-Length")
    or "0"` conflated an ABSENT header with one PRESENT but empty
    (`Content-Length: `) -- the same falsy-present-value-treated-as-
    absent class rounds 8/9 already fixed for `messages`. An
    empty-but-present header used to silently read as a zero-length body
    instead of hitting the numeric-parse `ValueError` branch, which gives
    a clearer diagnostic."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Content-Length", "")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 422
        assert "content-length" in payload["error"]["message"].lower()
    finally:
        conn.close()


def test_duplicate_conflicting_content_length_headers_refused_not_silently_first_wins(proxy):
    """Round-26 `/code-review` finding: `self.headers["Content-Length"]`
    only returns the FIRST value of a header name that legally appears on
    more than one line, the same gap round 22 already fixed for
    `Transfer-Encoding` via `.get_all()`. Two CONFLICTING `Content-Length`
    values must be refused outright rather than silently picking one."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Content-Length", "5")
        conn.putheader("Content-Length", "500")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        conn.send(json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8"))
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 422
        assert "content-length" in payload["error"]["message"].lower()
    finally:
        conn.close()


def test_duplicate_identical_content_length_headers_not_refused(proxy, fake_upstream):
    """Companion to the above: two IDENTICAL `Content-Length` values are
    not actually conflicting -- must still be accepted, not refused."""
    fake_upstream._responses.append((200, _usage_response()))
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Authorization", "Bearer sk-fake")
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        assert resp.status == 200
    finally:
        conn.close()


def test_duplicate_conflicting_authorization_headers_refused(proxy):
    """Round-27 `/code-review` finding: `self.headers.get("Authorization",
    "")` only returns the FIRST value of a header name that legally
    appears on more than one line, the same gap rounds 22/26 already
    fixed for `Transfer-Encoding`/`Content-Length`. Two CONFLICTING
    `Authorization` values must be refused outright rather than silently
    forwarding just the first and dropping the second."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Authorization", "Bearer sk-one")
        conn.putheader("Authorization", "Bearer sk-two")
        conn.putheader("Content-Type", "application/json")
        body = json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 422
        assert "authorization" in payload["error"]["message"].lower()
    finally:
        conn.close()


def test_duplicate_identical_authorization_headers_not_refused(proxy, fake_upstream):
    """Companion to the above: two IDENTICAL `Authorization` values are
    not actually conflicting -- must still be accepted."""
    fake_upstream._responses.append((200, _usage_response()))
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Authorization", "Bearer sk-fake")
        conn.putheader("Authorization", "Bearer sk-fake")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        assert resp.status == 200
    finally:
        conn.close()


def test_content_length_and_chunked_transfer_encoding_both_present_refused(proxy, fake_upstream):
    """Round-28 `/code-review` finding: a request carrying BOTH
    `Content-Length` and a chunked `Transfer-Encoding` is the classic
    HTTP request-smuggling precondition (RFC 7230 SS3.3.3 mandates such
    a message be rejected, since the two headers disagree about where
    the body ends). Must be refused outright, not silently resolved by
    picking `Content-Length` and ignoring `Transfer-Encoding`."""
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        assert resp.status == 422
        resp.read()
    finally:
        conn.close()
    assert len(fake_upstream.received_bodies) == 0  # never forwarded


def test_content_length_present_without_chunked_transfer_encoding_still_allowed(proxy, fake_upstream):
    """Companion to the above: a non-chunked `Transfer-Encoding` value
    alongside `Content-Length` is not the smuggling precondition and
    must still be accepted."""
    fake_upstream._responses.append((200, _usage_response()))
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(body)))
        conn.putheader("Transfer-Encoding", "identity")
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        assert resp.status == 200
    finally:
        conn.close()


def test_dict_content_refused_not_silently_underestimated(proxy):
    """Round-29 `/code-review` finding: the round-22 refusal only checked
    `isinstance(content, list)`. A `content` value that is some OTHER
    non-string JSON shape (a bare dict, e.g. a single multimodal part
    sent without the documented list wrapper) slipped past that check
    and fell through to `count_prompt_tokens()`'s `json.dumps()`-and-
    tokenize fallback -- the same silent under-estimate the list check
    exists to prevent, just a different container type. Must be refused
    the same way."""
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o",
              "messages": [{"role": "user", "content": {"type": "image_url", "image_url": {"url": "https://x/y.png"}}}]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "content" in resp.json()["error"]["message"].lower()


def test_null_content_still_allowed_not_refused(proxy, fake_upstream):
    """Companion to the above: an explicit `content: null` (distinct from
    a dict/list) is not a billing-relevant shape by itself and must still
    be accepted -- the refusal must not overreach into refusing `None`."""
    fake_upstream._responses.append((200, _usage_response()))
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": None}]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 200


def test_unexpected_non_request_exception_charges_worst_case_not_released(proxy, ledger):
    """Round-29 `/code-review` finding: the `except ConnectTimeout` /
    `except requests.RequestException` pair above only catches those two
    hierarchies. Any OTHER exception type raised by `requests.post()`
    (simulated here with a raw `OSError`, standing in for e.g. an
    unwrapped socket-level error) used to fall through to the outer
    `finally` block with `resolved` still False, which RELEASES the
    reservation -- correct only if request bytes were never sent, but
    this exception could just as easily fire after the request was
    transmitted and billed server-side. Same fail-safe direction as the
    `RequestException` branch: charge worst-case rather than assume
    nothing was spent."""
    before = ledger.remaining_usd()
    with patch.object(proxy_module.requests, "post", side_effect=OSError("simulated unwrapped socket error")):
        status = _client_post(proxy, {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                                       "max_tokens": 1000})
    after = ledger.remaining_usd()
    assert status == 502
    expected_worst_case = pricing.estimate_worst_case_cost_usd(
        model="gpt-4o", prompt_tokens=count_prompt_tokens("gpt-4o", [{"role": "user", "content": "hi"}]),
        max_output_tokens=1000,
    )
    assert before - after == pytest.approx(expected_worst_case, rel=1e-6)


def test_stream_true_refused_not_forwarded_as_broken_response(proxy, fake_upstream):
    """Round-30 `/code-review` finding: `stream: true` used to be
    forwarded unmodified. It does not under-count the cap (the
    unparseable-SSE-body fallback already charges worst-case), but the
    caller would get back a 200 with a raw SSE blob under an
    `application/json` Content-Type instead of a clean, understandable
    error -- refuse it outright."""
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 422
    assert "stream" in resp.json()["error"]["message"].lower()
    assert len(fake_upstream.received_bodies) == 0  # never forwarded


def test_stream_false_still_allowed(proxy, fake_upstream):
    """Companion to the above: `stream: false` (or absent) is not a
    streaming request and must still be accepted."""
    fake_upstream._responses.append((200, _usage_response()))
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 200


def test_refuse_write_failure_after_worst_case_commit_does_not_corrupt_call_log(ledger, tmp_path):
    """Round-31 `/code-review` finding, reproduced live: `_charge_worst_
    case_and_refuse()` used to set `resolved = True` only AFTER its own
    `self._refuse()` call returned. If that call's socket write raised
    (e.g. the client already disconnected), the outer `finally` block
    saw `resolved` still False and RELEASED the reservation `commit()`
    had just made, plus appended a SECOND, contradictory `call_log.jsonl`
    entry (`released_unexpected_exception`) for a call already charged
    worst-case -- corrupting the audit trail, even though no dollar
    amount was actually lost (`release()` on an already-committed
    call_id is a harmless no-op). `resolved` must be set right after the
    commit, inside the helper, before the write that can fail."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()
    upstream_url = f"http://127.0.0.1:{closed_port}"
    call_log_path = tmp_path / "call_log.jsonl"
    with CostCapProxy(ledger, call_log_path, upstream_url=upstream_url) as p:
        with patch.object(proxy_module._CostCapRequestHandler, "_refuse",
                           side_effect=OSError("simulated broken client connection")):
            try:
                requests.post(_chat_url(p), json={"model": "gpt-4o",
                                                    "messages": [{"role": "user", "content": "hi"}],
                                                    "max_tokens": 1000},
                              headers={"Authorization": "Bearer sk-fake"}, timeout=5)
            except requests.exceptions.RequestException:
                pass  # server-side write failed mid-response; client-side outcome not under test
    lines = call_log_path.read_text(encoding="utf-8").splitlines()
    decisions = [json.loads(line)["decision"] for line in lines]
    assert decisions.count("upstream_network_error_charged_worst_case") == 1
    assert "released_unexpected_exception" not in decisions


def test_refusal_before_body_read_does_not_intermittently_reset_connection(proxy):
    """Round-32 `/code-review` finding, reproduced live: refusal paths
    that fire before `raw_body = self.rfile.read(length)` (e.g. the
    duplicate-`Authorization` check) used to close the connection with
    the client's already-sent body bytes still sitting unread in the
    kernel receive buffer -- TCP then sends a RST instead of a clean FIN
    (reliably reproducible on Windows loopback), so the client's read of
    the 422 response intermittently failed with `ConnectionResetError`
    instead of receiving it (~1-in-5 to 1-in-3 runs empirically, not
    deterministic within a single request). Repeat the request enough
    times in one test to make a regression here fail reliably rather
    than flake past CI; `_refuse()` now drains the pending body first."""
    body = json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8")
    for _ in range(20):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
        try:
            conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
            conn.putheader("Authorization", "Bearer sk-one")
            conn.putheader("Authorization", "Bearer sk-two")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()
            conn.send(body)
            resp = conn.getresponse()
            payload = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 422
            assert "authorization" in payload["error"]["message"].lower()
        finally:
            conn.close()


@pytest.mark.parametrize("field", ["audio", "modalities"])
def test_audio_modalities_refused_not_silently_underestimated(proxy, field):
    """Round-32 `/code-review` finding: `modalities`/`audio` (gpt-4o
    audio input/output) bills a substantially higher, entirely separate
    audio-token rate `pricing.PRICING` has no entry for at all -- the
    pre-flight estimate would only ever account the text portion,
    silently under-estimating the true cost the same way `tools`/
    `response_format`/list-content did before rounds 22-25 closed them."""
    payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    if field == "audio":
        payload["audio"] = {"voice": "alloy", "format": "wav"}
    else:
        payload["modalities"] = ["text", "audio"]
    resp = requests.post(_chat_url(proxy), json=payload, headers={"Authorization": "Bearer sk-fake"})
    assert resp.status_code == 422
    assert "audio" in resp.json()["error"]["message"].lower()


def test_modalities_text_only_still_allowed(proxy, fake_upstream):
    """Companion to the above: `modalities: ["text"]` (no audio) is not
    the billing-relevant shape and must still be accepted."""
    fake_upstream._responses.append((200, _usage_response()))
    resp = requests.post(
        _chat_url(proxy),
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "modalities": ["text"]},
        headers={"Authorization": "Bearer sk-fake"},
    )
    assert resp.status_code == 200


def test_non_post_refusal_with_body_does_not_intermittently_reset_connection(proxy):
    """Round-33 `/code-review` finding, reproduced live: `_handle_non_
    post()` (backing `do_GET`/`do_PUT`/`do_DELETE`/`do_PATCH`/`do_HEAD`/
    `do_OPTIONS`) called `_refuse()` without ever setting `self._request_
    body_drained = False` first -- that flag was only initialized inside
    `_handle_post`, so `_refuse()`'s `getattr(..., True)` default
    silently skipped the round-32 drain fix on this sibling code path,
    reintroducing the same intermittent-RST bug for e.g. a `PUT` with a
    body. Repeat enough times in one test to catch a regression
    reliably rather than flaking past CI."""
    body = json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8")
    for _ in range(20):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
        try:
            conn.putrequest("PUT", _CHAT_COMPLETIONS_PATH)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()
            conn.send(body)
            resp = conn.getresponse()
            payload = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 422
            assert "PUT" in payload["error"]["message"]
        finally:
            conn.close()


def test_drain_fallback_has_overall_deadline_not_indefinite(proxy):
    """Round-33 `/code-review` finding, reproduced live: the fallback
    branch of `_drain_request_body_best_effort()` bounded only each
    individual `recv()` call to 0.2s, with no OVERALL deadline -- a
    client trickling bytes faster than that window kept the handler
    thread blocked in the drain loop indefinitely, contradicting the
    docstring's "short bounded" claim. Trickle bytes for well longer than
    the fallback deadline and assert the refusal response still arrives
    within a bounded time, not after the trickle finally stops."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=15)
    stop_trickle = threading.Event()
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()

        def trickle():
            try:
                while not stop_trickle.is_set():
                    conn.sock.sendall(b"0")
                    time.sleep(0.05)
            except OSError:
                pass

        thread = threading.Thread(target=trickle, daemon=True)
        thread.start()
        try:
            start = time.monotonic()
            resp = conn.getresponse()
            elapsed = time.monotonic() - start
            assert resp.status == 422
        finally:
            stop_trickle.set()
            thread.join(timeout=2)
        # The fallback deadline is proxy_module._DRAIN_FALLBACK_DEADLINE_S
        # (2.0s); generous slack allows for scheduling/test-runner
        # overhead, but this must stay well under "however long the
        # trickle keeps going" -- the whole point of the round-33 fix.
        assert elapsed < proxy_module._DRAIN_FALLBACK_DEADLINE_S + 5.0
    finally:
        stop_trickle.set()
        conn.close()


def test_drain_fallback_survives_a_quiet_period_shorter_than_deadline(proxy):
    """Round-34 `/code-review` finding, reproduced live: `except OSError:
    pass` used to wrap the ENTIRE fallback while-loop in
    `_drain_request_body_best_effort()`, not each iteration individually
    -- `socket.timeout` (an expected `OSError` subclass, the normal
    outcome of a quiet period shorter than the overall deadline) on any
    single `recv()` aborted the WHOLE drain immediately instead of
    retrying until `_DRAIN_FALLBACK_DEADLINE_S`. A client with any pause
    longer than the ~0.2s per-iteration poll window (ordinary network
    jitter, or simply a slow-but-alive sender) got abandoned almost
    instantly -- reproduced live: response arrived after only ~0.2s,
    before a 0.6s-later send even happened. Send one byte, pause 0.6s
    (longer than one poll window, far short of the 2.0s deadline), then
    signal end-of-body via a half-close -- the response must not arrive
    before that pause ends."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=15)
    try:
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        conn.send(b"0")
        start = time.monotonic()
        time.sleep(0.6)
        conn.sock.shutdown(socket.SHUT_WR)  # half-close: signal end-of-body, keep reading the response
        resp = conn.getresponse()
        elapsed = time.monotonic() - start
        payload = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 422
        assert "chunked" in payload["error"]["message"].lower()
        assert elapsed >= 0.55, (
            f"response arrived after only {elapsed:.3f}s -- the drain loop aborted on the first "
            "recv() timeout instead of surviving the quiet period until the overall deadline"
        )
    finally:
        conn.close()


def test_drain_with_known_content_length_still_bounded_by_overall_deadline(proxy):
    """Round-35 `/code-review` finding, reproduced live: the `Content-
    Length`-known branch of `_drain_request_body_best_effort()` used to be
    a single blocking `self.rfile.read(length)` call bounded only by the
    per-`recv` SOCKET timeout (`self.timeout`, 60s in production) -- which
    RESETS on every byte received, so a client trickling the DECLARED body
    slower than each poll gap but faster than 60s per gap could keep this
    call blocked far past the intended `_DRAIN_FALLBACK_DEADLINE_S`
    best-effort bound (reproduced live at ~7.8s against an intended 2.0s
    cap). Declare a large `Content-Length`, trickle far fewer bytes than
    declared, slower than the response would need to wait for under the
    fix, and assert the refusal still arrives within the overall deadline
    instead of blocking for the trickle's full duration."""
    body_declared_length = 1_000_000
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=15)
    stop_trickle = threading.Event()
    try:
        # Missing Authorization header, refused before the real body read
        # -- this exercises the same drain call as the other refusal paths.
        conn.putrequest("POST", _CHAT_COMPLETIONS_PATH)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(body_declared_length))
        conn.putheader("Authorization", "Bearer one")
        conn.putheader("Authorization", "Bearer two")
        conn.endheaders()

        def trickle():
            try:
                for _ in range(40):
                    if stop_trickle.is_set():
                        return
                    conn.sock.sendall(b"0")
                    time.sleep(0.3)
            except OSError:
                pass

        thread = threading.Thread(target=trickle, daemon=True)
        thread.start()
        try:
            start = time.monotonic()
            resp = conn.getresponse()
            elapsed = time.monotonic() - start
            assert resp.status == 422
        finally:
            stop_trickle.set()
            thread.join(timeout=2)
        assert elapsed < proxy_module._DRAIN_FALLBACK_DEADLINE_S + 5.0, (
            f"response arrived after {elapsed:.3f}s -- the Content-Length-known drain branch is "
            "still blocking on the socket timeout instead of the overall drain deadline"
        )
    finally:
        stop_trickle.set()
        conn.close()


def test_drain_skipped_immediately_when_nothing_was_ever_declared(proxy):
    """Round-35 `/code-review` finding, reproduced live: the fallback
    branch of `_drain_request_body_best_effort()` used to unconditionally
    wait out the full `_DRAIN_FALLBACK_DEADLINE_S` even when NEITHER
    `Content-Length` NOR a chunked `Transfer-Encoding` was ever declared
    -- i.e. when there is provably nothing to drain, the common shape for
    a bodiless GET/HEAD/OPTIONS/etc. refusal. Reproduced live: a plain
    bodiless GET measured at exactly 2.0s before its 422, instead of an
    immediate response. Assert the refusal now arrives near-instantly."""
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=15)
    try:
        start = time.monotonic()
        conn.request("GET", _CHAT_COMPLETIONS_PATH)
        resp = conn.getresponse()
        elapsed = time.monotonic() - start
        resp.read()
        assert resp.status == 422
        assert elapsed < proxy_module._DRAIN_FALLBACK_DEADLINE_S, (
            f"response arrived after {elapsed:.3f}s -- a bodiless refusal with nothing declared "
            "to drain should not wait out the fallback drain deadline at all"
        )
    finally:
        conn.close()


def test_unregistered_http_method_still_gets_the_422_refusal_contract(proxy):
    """Round-36 `/code-review` finding, reproduced live: only the six
    `do_*` methods explicitly registered on `_CostCapRequestHandler`
    (plus `do_POST`) were ever wired up -- any OTHER HTTP method token
    (`TRACE`, `CONNECT`, `PROPFIND`, ...) is dispatched by
    `BaseHTTPRequestHandler.handle_one_request()` via `getattr(self,
    "do_" + self.command)`, which used to fall through to
    `BaseHTTPRequestHandler`'s own stock 501 handler for any
    unrecognized `do_*` name -- bypassing `_handle_non_post` entirely,
    so a request body (if any) was never drained (the round-32 RST-risk
    class) and the response was a raw 501 HTML page instead of this
    module's documented 422 JSON refusal contract. Send a `TRACE` with a
    body and assert it gets the same 422 JSON contract as every other
    unsupported method, repeated to also catch a drain/RST regression."""
    body = json.dumps({"model": "gpt-4o", "messages": []}).encode("utf-8")
    for _ in range(20):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port)
        try:
            conn.putrequest("TRACE", _CHAT_COMPLETIONS_PATH)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()
            conn.send(body)
            resp = conn.getresponse()
            payload = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 422
            assert "TRACE" in payload["error"]["message"]
        finally:
            conn.close()
