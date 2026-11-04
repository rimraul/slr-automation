"""slr_pipeline.cost_cap.proxy -- the loopback HTTP proxy every real
`OpenAI(api_key=...)` client in the graph transparently talks to once
`OPENAI_BASE_URL` points at it, mirroring how `slr_pipeline.replay.transport
.ReplayServer` already does the same trick for offline reproducibility
(confirmed: none of the 8 production call sites pass an explicit
`base_url=`, and the `openai` SDK itself falls back to
`os.environ.get("OPENAI_BASE_URL")` when one isn't given -- so pointing
this env var at the proxy wires every call site with zero node-code
changes).

Unlike `ReplayServer`, this does NOT serve canned responses: every allowed
request is forwarded to the real `https://api.openai.com`, and the
Authorization header (holding the real API key) is relayed verbatim,
untouched and unread by this module -- it never itself resolves
`OPENAI_API_KEY` or reads `my_api_keys.py`.

Contract, `POST /v1/chat/completions` only (the only endpoint any of the
8 real call sites use, confirmed by grep):
  1. Parse the request body; look up its `model` in
     `pricing.PRICING`. Unpriced model -> refuse (402), never forwarded.
  2. Count prompt tokens (tiktoken) and the request's own
     `max_completion_tokens`/`max_tokens` (or a conservative default
     ceiling if neither is given); compute the worst-case dollar cost of
     this ONE call. If BOTH keys are present, `max_completion_tokens` (the
     modern parameter, and the only one any of the 8 real call sites ever
     sets) is authoritative and `max_tokens` is dropped from the
     forwarded body entirely -- this proxy never forwards both keys in
     one request, since some model families reject that outright, and
     the reservation is always sized for the exact single value actually
     forwarded. The request's own `n` (number of independently-billed
     completions OpenAI returns, default 1) is validated/resolved the
     same way (round-15 `/code-review` finding) and multiplies only the
     output-token half of the worst-case estimate -- an invalid
     (non-finite/non-positive) `n` is rewritten in the forwarded body to
     the resolved value, exactly like `max_completion_tokens`/`max_tokens`
     are, so the reservation can never diverge from what is actually sent.
  3. `ledger.reserve(call_id, worst_case_usd)` -- if this would push
     committed+reserved spend past the cap, refuse (402), never
     forwarded. This is the literal "stop before any request that could
     exceed the remaining cap" the user's authorization requires: the
     check happens BEFORE the real API is ever contacted.
  4. Otherwise forward the exact request to the real API. On a real 200
     response, read `usage` and `ledger.commit()` the ACTUAL cost
     (normally less than the worst-case reservation); return the response
     body verbatim. On upstream failure, the ledger is resolved by
     verifiability, not uniformly: a `ConnectTimeout` (by definition zero
     bytes could have been sent) or a non-200 status `ledger.release()`s
     the reservation, but any OTHER `requests.RequestException` (timeout
     after the request was sent, connection reset -- both ambiguous about
     whether the upstream already billed the call), and a 200 whose body
     is unparseable/missing a usable `usage` field, instead
     `ledger.commit()` the WORST-CASE reservation (`usage_verified=False`
     in the persisted record) -- round-12/13/14 `/code-review` findings:
     when the true outcome can't be verified, this module's policy is to
     charge the safe upper bound, never to assume nothing was spent. Only
     the `ConnectTimeout`/non-200 cases relay the real upstream failure
     (or a proxy-synthesized refusal) back to the caller; the SDK's own
     error handling (all 8 sites use `max_retries=0` or the SDK default)
     still sees a normal HTTP error response either way. An unexpected
     exception raised AFTER a real 200 has already come back (e.g. a bug
     in the `usage`-parsing/`actual_cost_usd()`/`commit()` code itself,
     distinct from the already-covered "body is unparseable" case) is
     caught by neither of the above -- the outer `finally`'s
     `upstream_returned_200` flag (round-15 `/code-review` finding) makes
     the same call: `ledger.commit()` the worst case rather than release,
     since the upstream may already have billed it. That exception is
     NOT itself converted into a clean response by this proxy (no
     `_refuse()` runs for it); the connection is simply torn down, same
     as any other uncaught exception outside this handler's own
     `try`/`except` blocks.

Any other method/path is refused (422) without touching the ledger.
"""
from __future__ import annotations

import json
import logging
import math
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import requests
import tiktoken

from .ledger import BudgetExceededError, CostLedger, UnpricedModelError
from . import pricing as pricing_mod

logger = logging.getLogger(__name__)

_REAL_OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Round-5 `/code-review` finding: a syntactically valid but absurdly large
# `Content-Length` (malicious or simply malformed client) would otherwise
# reach `self.rfile.read(length)` unbounded, forcing this handler thread to
# allocate/block on however many bytes the header claims. No legitimate
# `chat/completions` request body -- even with a long system prompt and
# many few-shot messages -- should ever approach this; refuse before the
# read instead of after.
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024

# Round-33 `/code-review` finding: the fallback branch of
# `_drain_request_body_best_effort()` used to bound only each individual
# `recv()` call, not the loop's total wall-clock time -- a client
# trickling bytes faster than this window kept the handler thread
# blocked indefinitely. This is the hard overall cap on how long that
# best-effort drain is allowed to run, regardless of how often chunks
# arrive.
_DRAIN_FALLBACK_DEADLINE_S = 2.0

# tiktoken has no encoding registered for these project-specific model IDs
# (they postdate tiktoken's built-in model->encoding table). o200k_base is
# the encoding OpenAI's own gpt-4o/gpt-4o-mini family uses and is the
# closest available match for token-count purposes -- this is a PRE-FLIGHT
# ESTIMATE feeding a safety check, not the billed figure (the real
# `usage.prompt_tokens` from the response is what actually gets committed
# to the ledger in `commit()`).
_FALLBACK_ENCODING_NAME = "o200k_base"

# OpenAI cookbook's documented per-message/per-conversation chat-format
# overhead (role/name/separator tokens the raw content-only token count
# misses) -- pads the pre-flight estimate up rather than down, keeping it a
# genuine worst-case upper bound.
_TOKENS_PER_MESSAGE = 4
_TOKENS_PER_CONVERSATION = 2

# Track-3 authorization (2026-09-12): the round-25 blocklist entry refused
# `response_format={"type": "json_schema", ...}` outright rather than count
# it, because OpenAI bills the whole `response_format` object (type/name/
# strict/schema) as real input tokens the same way a `tools` definition is,
# and this proxy had no way to size that safely. That assumption no longer
# holds now that a real call site needs Structured Outputs -- this estimates
# it instead of refusing it, using the raw JSON-serialized-and-tokenized
# size padded well past 1x so it can never under-reserve (OpenAI's internal
# schema-to-token accounting is not published, so a raw `json.dumps` token
# count is treated as a lower bound, not the true figure).
_RESPONSE_FORMAT_TOKEN_PADDING_MULTIPLIER = 2.0
_RESPONSE_FORMAT_TOKEN_PADDING_FIXED = 32
# A schema this large is treated as unsafe to estimate at all (refused
# outright, same "can't safely estimate it, so don't guess" rule as the
# other blocklist entries in this module) rather than trusting the padding
# multiplier to cover an unbounded size.
_MAX_RESPONSE_FORMAT_SCHEMA_TOKENS = 4000

# Round-1 `/code-review` finding (Track-3 proxy-fix cycle): the schema-
# reference refusal below must only fire on these names in *key* position
# (JSON Schema's actual `$ref`/`$defs`/legacy `definitions` keywords),
# never on a plain string match, or it false-positives on a legitimate
# schema that merely uses one of these words as a property name or an
# enum/string value (round-2 `/code-review` finding, against the earlier
# `json.dumps()`-substring version of this check).
_SCHEMA_REFERENCE_KEYS = frozenset({"$ref", "$defs", "definitions"})
# Containers whose immediate child keys are caller-chosen names (property
# names, or sub-schema names under `$defs`/`definitions`), not JSON Schema
# keywords -- a child key here must never be checked against
# `_SCHEMA_REFERENCE_KEYS` itself, only recursed into as an ordinary value.
# Round-2 `/code-review` finding: without this, a property literally named
# "definitions" (e.g. `{"properties": {"definitions": {"type": "string"}}}`,
# an extracted-text section, unrelated to the JSON Schema keyword) was
# wrongly flagged.
# Round-3 `/code-review` finding: `dependentSchemas` has the identical
# "children are caller-chosen property names, not keywords" shape as
# `properties`/`patternProperties`/`$defs`/`definitions` and was missing,
# false-positiving on e.g. `{"dependentSchemas": {"definitions": {...}}}`.
_SCHEMA_NAME_MAP_CONTAINERS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)


def _contains_schema_reference_key(node, _in_name_map: bool = False) -> bool:
    """True if `node` (a parsed JSON value) contains any of
    `_SCHEMA_REFERENCE_KEYS` in actual JSON-Schema-keyword key position, at
    any depth (never inside a caller-chosen name, e.g. a `properties` key)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if not _in_name_map and key in _SCHEMA_REFERENCE_KEYS:
                return True
            # Round-3 `/code-review` finding: whether `value`'s own keys are
            # names-not-keywords depends on `key` being a real *keyword* in
            # the current dict -- which is only true when the current dict's
            # own keys are keywords (`not _in_name_map`). Inside an already-
            # entered name-map (e.g. a property literally named
            # "properties"), `key` is just a caller-chosen name, and `value`
            # is a fresh subschema whose own keys are keywords again -- NOT
            # a second name-map, however `key` happens to spell. Without the
            # `not _in_name_map` guard, a schema like
            # `{"properties": {"properties": {"$ref": "..."}}}` wrongly
            # re-armed name-map mode for the inner `$ref`'s dict and missed
            # a genuine reference-keyword use.
            child_in_name_map = (not _in_name_map) and key in _SCHEMA_NAME_MAP_CONTAINERS
            if _contains_schema_reference_key(value, _in_name_map=child_in_name_map):
                return True
        return False
    if isinstance(node, list):
        return any(_contains_schema_reference_key(item) for item in node)
    return False


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING_NAME)


def count_prompt_tokens(model: str, messages: list) -> int:
    enc = _encoding_for_model(model)
    total = _TOKENS_PER_CONVERSATION
    for message in messages:
        total += _TOKENS_PER_MESSAGE
        # Round-2 `/code-review` finding: a malformed `messages` entry
        # (not a dict -- e.g. a bare string) used to raise an uncaught
        # AttributeError here, crashing the handler thread before any
        # HTTP response was ever written back to the caller. This call
        # happens BEFORE `ledger.reserve()`, so no budget leak resulted,
        # but the caller (the OpenAI SDK client) saw a broken connection
        # instead of a clean 422 its own error handling expects. Treat a
        # malformed entry the same as an empty one rather than crash.
        if not isinstance(message, dict):
            content, role, name = "", "", ""
        else:
            content = message.get("content", "")
            role = message.get("role", "")
            name = message.get("name", "")
        # Chat messages are usually a plain string; some call sites may
        # send structured content (list of parts) -- stringify defensively
        # rather than crash the proxy on an unexpected but legitimate shape.
        # (`proxy.py`'s own request-shape validation refuses a list
        # `content` outright before this is ever reached -- round-22
        # `/code-review` finding -- so this fallback only ever fires for
        # some other non-str, non-list shape, e.g. a stray number.)
        if not isinstance(content, str):
            content = json.dumps(content)
        if not isinstance(role, str):
            role = json.dumps(role)
        # Round-23 `/code-review` finding: a message's `name` key (used
        # for function/tool-role messages and named participants) bills
        # its own real prompt tokens on OpenAI's side -- `name` was never
        # read here at all, silently under-counting the pre-flight
        # estimate by however many tokens the identifier is worth.
        if not isinstance(name, str):
            name = json.dumps(name)
        total += len(enc.encode(content))
        total += len(enc.encode(role))
        total += len(enc.encode(name))
    return total


def estimate_response_format_tokens(model: str, response_format: dict) -> int:
    """Conservative worst-case token estimate for a whole `response_format`
    object (type/name/strict/schema together, not just the nested `schema`
    -- OpenAI bills the object as a unit, and `name`/`strict` are real,
    if small, additional tokens). Raises `ValueError` if `response_format`
    cannot be JSON-serialized at all (caller treats that as a malformed
    request, refused before dispatch, never as an internal error)."""
    enc = _encoding_for_model(model)
    try:
        serialized = json.dumps(response_format, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"response_format is not JSON-serializable: {exc}") from exc
    raw_tokens = len(enc.encode(serialized))
    return math.ceil(raw_tokens * _RESPONSE_FORMAT_TOKEN_PADDING_MULTIPLIER) + _RESPONSE_FORMAT_TOKEN_PADDING_FIXED


class _CallLogWriter:
    """Append-only `cost_cap_call_log.jsonl` -- one line per HTTP attempt
    this proxy handled, allowed or refused (the ledger's own JSON only
    records committed successes; this is the fuller audit trail, part of
    "preserve results ... for later use")."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, **fields) -> int:
        # Round-42 `/code-review` finding, considered and deliberately left
        # as-is: this lock is held across the actual disk write, unlike
        # `CostLedger`'s persist lock (round 39 moved ITS retry loop
        # outside its own lock specifically because that lock also gates
        # `reserve()`/`commit()`/`release()` -- unrelated critical
        # sections a slow disk would otherwise stall). `self._lock` here
        # has no other caller anywhere in this class; concurrent
        # `append()` calls must still serialize on the write itself
        # regardless of lock structure, to avoid interleaved bytes from
        # two threads sharing one file handle -- splitting seq-allocation
        # from the write into two locks would not shorten any thread's
        # wait, only add a second lock with no unblocked use case behind
        # it. Left combined.
        write_exc: Optional[BaseException] = None
        with self._lock:
            self._seq += 1
            seq = self._seq
            line = {"seq": seq, "ts": time.time(), **fields}
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, sort_keys=True, default=str) + "\n")
                    f.flush()
            except (OSError, TypeError, ValueError) as exc:
                # Round-4 `/code-review` finding, reproduced live: every
                # call site that reaches `append()` for an "allowed" call
                # does so AFTER `ledger.commit()` has already recorded the
                # real spend, immediately before writing the HTTP 200 back
                # to the client. Before this fix, letting an `OSError` here
                # (disk full, same fault class round 3 hardened
                # `commit()` against) propagate meant the 200 was never
                # sent -- the client saw a broken connection for a call
                # that had genuinely succeeded and been billed, and the
                # SDK's retry logic could then pay for the same completion
                # twice. The call log is a best-effort audit trail, same
                # as the ledger's own on-disk snapshot; it must never be
                # allowed to block the response for a call whose outcome
                # is already decided.
                #
                # Round-5 `/code-review` finding: `json.dumps(..., default=
                # str)` above can itself raise `TypeError`/`ValueError` for
                # pathological `**fields` values (e.g. a circular reference
                # inside a nested dict), not just disk I/O failures -- a
                # narrower `except OSError` alone missed that case.
                #
                # Round-6 `/code-review` finding: the `logger.error()` call
                # used to run right here, still holding `self._lock` -- a
                # slow log handler would stall every other thread's
                # `append()`. Stash the exception and log it after the
                # `with self._lock` block exits instead; the exception
                # object itself is enough for `exc_info=` to render the
                # full traceback, no need to still be inside the `except`.
                write_exc = exc
            return_seq = seq
        if write_exc is not None:
            logger.error(
                "_CallLogWriter.append(): failed to write call-log entry seq=%d to %s -- "
                "continuing without blocking the caller's response",
                return_seq, self._path, exc_info=write_exc,
            )
        return return_seq


class _CostCapHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler_cls, *, ledger: CostLedger, call_log: _CallLogWriter,
                 upstream_url: str, request_timeout_s: float) -> None:
        super().__init__(address, handler_cls)
        self.ledger = ledger
        self.call_log = call_log
        self.upstream_url = upstream_url
        self.request_timeout_s = request_timeout_s
        # Round-1 `/code-review` finding: `daemon_threads = True` (needed
        # so a hung request never blocks process exit) means
        # `server_close()` does NOT wait for in-flight `do_POST` handler
        # threads -- if the parent process exits right after `stop()`
        # returns (the subprocess-per-run pattern this proxy is meant to
        # run under), a request that already spent real money but hadn't
        # reached `ledger.commit()` yet would be killed mid-flight and
        # never get recorded. `stop()` now waits for this counter to hit
        # zero (bounded by a timeout) before closing the socket.
        self._active_requests = 0
        self._active_requests_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()

    def next_call_id(self) -> str:
        # Round-2 `/code-review` finding: a sequential per-process counter
        # (`call-1`, `call-2`, ...) mints the SAME call_id again if a proxy
        # is ever restarted against a `CostLedger` resumed from disk (the
        # ledger's own cap-tolerance check exists specifically to support
        # resuming) -- two distinct real API calls would then share one
        # call_id in the persisted audit trail. A random id has no such
        # cross-process collision.
        return f"call-{uuid.uuid4().hex}"

    def begin_request(self) -> None:
        with self._active_requests_lock:
            self._active_requests += 1
            self._idle.clear()

    def end_request(self) -> None:
        with self._active_requests_lock:
            self._active_requests -= 1
            if self._active_requests <= 0:
                self._idle.set()

    def wait_for_idle(self, timeout: float) -> bool:
        """Returns True if all in-flight requests finished within
        `timeout` seconds, False if the wait timed out with requests
        still outstanding."""
        return self._idle.wait(timeout=timeout)

    def process_request(self, request, client_address) -> None:
        # Round-2 `/code-review` finding: the previous drain fix called
        # `begin_request()` from INSIDE the spawned handler thread's
        # `do_POST`, so a connection whose thread had been started but not
        # yet scheduled to run was invisible to `wait_for_idle()` --
        # `shutdown()` could return while such a thread was still about to
        # make the real upstream call, and a caller exiting right after
        # `stop()` returns could kill it before it ever reached
        # `ledger.reserve()`. Counting the request here, on the ACCEPTING
        # thread, before the worker thread is even spawned, closes that
        # gap for every request (not just POST).
        self.begin_request()
        try:
            super().process_request(request, client_address)
        except BaseException:
            # Round-3 `/code-review` finding: `ThreadingMixIn.process_request`
            # spawns the worker thread and returns immediately -- if
            # `threading.Thread.start()` itself raises (e.g. thread/FD
            # exhaustion during a long overnight run), `process_request_thread`
            # (which holds the compensating `end_request()`) never runs at
            # all, permanently leaking this increment and making every
            # future `stop()` block for the full drain timeout for no
            # reason. Compensate here, then let the exception continue
            # propagating exactly as it would have without this override.
            self.end_request()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.end_request()


class _CostCapRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _CostCapHTTPServer

    # Round-3 `/code-review` finding: `socketserver.BaseRequestHandler`
    # applies no read timeout by default, so a client that declares a
    # `Content-Length` and then stalls mid-body (hung/suspended process,
    # slow network) leaves this handler thread blocked in
    # `self.rfile.read(length)` forever, with `begin_request()` already
    # counted -- `CostCapProxy.stop()` would then always wait the full
    # `drain_timeout_s` for a request that never even reached
    # `ledger.reserve()`. A finite socket timeout bounds this; the
    # resulting `socket.timeout` propagates up through the normal
    # exception path, and `process_request_thread`'s `finally` (see
    # `_CostCapHTTPServer` above) still calls `end_request()` correctly.
    timeout = 60.0

    def setup(self) -> None:
        # `self.request` (the raw socket) is set by `BaseRequestHandler
        # .__init__` before `setup()` runs; `self.connection` doesn't
        # exist yet until `StreamRequestHandler.setup()` (called via
        # `super()` below) assigns it -- settimeout() must go on
        # `self.request` here, not `self.connection`.
        self.request.settimeout(self.timeout)
        super().setup()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- stdlib signature
        pass

    def _close_after_this_response(self) -> None:
        # Round-2 `/code-review` follow-up (found while re-testing after
        # the round-2 fixes, not one of the reviewer's own 8 findings):
        # `protocol_version = "HTTP/1.1"` defaults to persistent
        # connections, so `BaseHTTPRequestHandler`'s request loop sits
        # blocked reading the NEXT request on the same socket after every
        # response this proxy sends -- it never calls `end_request()`
        # until the CLIENT closes its end. That happened to occur quickly
        # in most manual/test runs (CPython's cyclic GC eventually
        # collecting urllib3's connection objects and closing the
        # underlying socket), but is not a real guarantee -- a
        # short-lived driver script with too little allocation to trigger
        # a GC cycle left the connection open indefinitely, reproduced
        # directly: `_active_requests` stuck at 1 well past the drain
        # timeout with the request itself long since answered. This proxy
        # is a one-shot-per-call server, never a long-lived multiplexed
        # one -- forcing the server side to close after every response
        # removes the dependency on client-side behavior entirely.
        self.send_header("Connection", "close")
        self.close_connection = True

    def _respond_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._close_after_this_response()
        self.end_headers()
        self.wfile.write(payload)

    def _respond_raw(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._close_after_this_response()
        self.end_headers()
        self.wfile.write(body)

    def _drain_request_body_best_effort(self) -> None:
        """Round-32 `/code-review` finding, reproduced live: several
        refusal paths in `_handle_post` (the `Authorization`-conflict
        check, the chunked-without-`Content-Length` check, the `Content-
        Length`-plus-chunked-`Transfer-Encoding` check) close the
        connection before ever reading the client's already-sent request
        body (`raw_body = self.rfile.read(length)` only happens once the
        request shape is judged valid enough to reach it). If those bytes
        are still sitting in the kernel receive buffer when the socket
        closes, TCP sends a RST instead of a clean FIN -- reliably
        reproducible on Windows loopback -- which can make the client's
        read of THIS refusal's own 4xx response intermittently fail with
        a connection-reset error instead of receiving it, silently
        undermining this module's documented promise that a malformed
        request "still sees a normal HTTP error response". Best-effort
        drain whatever the client already sent -- bounded by a single,
        unambiguous `Content-Length` value when known, else unbounded by
        byte count for the absent/ambiguous/chunked cases where no exact
        length can be trusted. Purely a response-delivery reliability fix
        -- no call site that reaches this has reserved anything yet, so
        nothing here affects budget accounting.

        Round-33 `/code-review` finding, reproduced live: an earlier
        version bounded only each individual `recv()` call to 0.2s, with
        no OVERALL deadline -- a client trickling data faster than every
        0.2s (even 1 byte at a time) kept this handler thread blocked
        here indefinitely. Capped with an explicit wall-clock deadline
        (`_DRAIN_FALLBACK_DEADLINE_S`) and a total-bytes cap, independent
        of how often individual chunks arrive.

        Round-34 `/code-review` finding, reproduced live: an earlier
        version's `except OSError: pass` wrapped the entire polling loop
        instead of each iteration -- `socket.timeout` (an `OSError`
        subclass) is the NORMAL, expected outcome of a quiet period
        shorter than the overall deadline, not a fatal error. Wrapping
        the whole loop meant the first quiet period longer than a single
        poll window (ordinary network jitter, or simply a slow-but-alive
        sender) aborted the ENTIRE drain immediately, abandoning most of
        the deadline's budget while the client was still mid-
        transmission. Now caught per iteration: `socket.timeout` just
        `continue`s the poll; only a genuine non-timeout `OSError` (the
        connection actually dropping) or a clean peer-closed (`recv()`/
        `read1()` returning `b""`) ends the loop early.

        Round-35 `/code-review` finding, reproduced live, closing two
        more gaps in this same block:
        (a) the `Content-Length`-known case used to be a single
        `self.rfile.read(length)` call bounded only by the per-`recv`
        SOCKET timeout (`self.timeout`, 60s in production) -- which
        RESETS on every byte received, so a client trickling the
        declared body slower than each poll interval but faster than
        60s per gap could keep this call blocked far past the intended
        best-effort bound, the same "indefinite blocking" failure class
        round 33 fixed for the fallback branch, left open here. Folded
        into the SAME deadline-bounded polling loop as the fallback
        case below (reading via `self.rfile.read1()` so any bytes
        already buffered from header parsing are consumed correctly,
        while still respecting the per-iteration `socket.timeout`
        handling from round 34), rather than a single unbounded call.
        (b) the fallback path used to run the full ~2.0s deadline even
        when NEITHER `Content-Length` NOR a chunked `Transfer-Encoding`
        was ever declared at all -- i.e. when there is provably nothing
        to drain, the single most common shape for the bodiless GET/
        HEAD/OPTIONS/DELETE/PATCH requests `_handle_non_post()` refuses.
        Skip the drain entirely in that case instead of waiting out the
        deadline for no reason.
        """
        target_bytes = None
        try:
            content_length_values = self.headers.get_all("Content-Length") or []
            if len(set(content_length_values)) == 1:
                length = int(content_length_values[0])
                if 0 <= length <= _MAX_REQUEST_BODY_BYTES:
                    target_bytes = length
        except (ValueError, OSError):
            target_bytes = None

        if target_bytes is None:
            transfer_encoding_values = self.headers.get_all("Transfer-Encoding") or []
            chunked = any("chunked" in v.lower() for v in transfer_encoding_values)
            if not chunked and "Content-Length" not in self.headers:
                return  # nothing was ever declared as a body -- nothing to drain

        deadline = time.monotonic() + _DRAIN_FALLBACK_DEADLINE_S
        drained_bytes = 0
        read1 = getattr(self.rfile, "read1", None)
        while True:
            if target_bytes is not None and drained_bytes >= target_bytes:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or drained_bytes >= _MAX_REQUEST_BODY_BYTES:
                break
            read_size = 65536 if target_bytes is None else min(65536, target_bytes - drained_bytes)
            try:
                self.connection.settimeout(min(0.2, remaining))
                chunk = read1(read_size) if read1 is not None else self.connection.recv(read_size)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            drained_bytes += len(chunk)
        try:
            self.connection.settimeout(self.timeout)
        except OSError:
            pass

    def _refuse(self, status: int, reason: str, **extra) -> None:
        if not getattr(self, "_request_body_drained", True):
            self._drain_request_body_best_effort()
            self._request_body_drained = True
        self._respond_json(status, {"error": {"message": reason, "type": "cost_cap_refusal", **extra}})

    def _handle_non_post(self) -> None:
        # Round-33 `/code-review` finding, reproduced live: this sibling
        # code path to `_handle_post` never set `self._request_body_
        # drained = False` before refusing, so `_refuse()`'s `getattr(...,
        # True)` default silently skipped the round-32 drain fix here --
        # reintroducing the exact same intermittent-RST bug for any
        # non-POST request that happens to carry a body (e.g. a `PUT`
        # with a JSON payload). Match `_handle_post`'s own entry-point
        # convention.
        self._request_body_drained = False
        self._refuse(422, f"unsupported method {self.command!r} (only POST is permitted)")

    do_GET = _handle_non_post
    do_PUT = _handle_non_post
    do_DELETE = _handle_non_post
    do_PATCH = _handle_non_post
    do_HEAD = _handle_non_post
    do_OPTIONS = _handle_non_post

    def __getattr__(self, name: str):
        # Round-36 `/code-review` finding, reproduced live: only the six
        # `do_*` methods explicitly registered above (plus `do_POST`)
        # were ever wired up -- any OTHER HTTP method token
        # (`TRACE`, `CONNECT`, `PROPFIND`, a mis-cased `post`, ...) is
        # dispatched by `BaseHTTPRequestHandler.handle_one_request()` via
        # `getattr(self, "do_" + self.command)`, which fell through to
        # its own stock 501 handler for any unrecognized `do_*` name --
        # bypassing `_handle_non_post` entirely, so the request's body
        # (if any) was never drained (reintroducing the round-32
        # RST-risk class) and the response was a raw 501 HTML page
        # instead of this module's documented 422 JSON refusal contract
        # ("any other method/path is refused (422) without touching the
        # ledger"). Answer any `do_*` lookup that isn't already an
        # explicit class attribute with `_handle_non_post` instead of
        # letting it fail into `BaseHTTPRequestHandler`'s own fallback.
        if name.startswith("do_"):
            return self._handle_non_post
        raise AttributeError(name)

    def do_POST(self) -> None:
        self._handle_post()

    def _handle_post(self) -> None:
        server = self.server
        # Round-32 `/code-review` finding: several refusal paths below
        # close the connection before ever reading the client's already-
        # sent request body -- see `_drain_request_body_best_effort()`'s
        # docstring for why that matters. This flag lets `_refuse()` tell
        # whether the body has been consumed yet.
        self._request_body_drained = False
        if self.path != _CHAT_COMPLETIONS_PATH:
            self._refuse(422, f"unknown endpoint {self.path!r}, expected {_CHAT_COMPLETIONS_PATH!r}")
            return

        # Round-27 `/code-review` finding: the `Authorization` header
        # forwarded upstream further below used to be read with plain
        # `self.headers.get("Authorization", "")`, which -- like
        # `Transfer-Encoding` (round 22) and `Content-Length` (round 26)
        # before it -- only returns the FIRST value of a header name that
        # legally appears on more than one line, silently dropping a
        # second one. No real call site sends two `Authorization` headers
        # (the OpenAI SDK sends exactly one) and the failure direction was
        # already fail-safe (forwarding the wrong/stale key yields an
        # upstream 401, which the non-200 branch already `release()`s for
        # -- no overspend), but this closes the same header-parsing
        # consistency gap the last two rounds closed for the other two
        # headers, before the value is ever used.
        authorization_values = self.headers.get_all("Authorization") or []
        if len(set(authorization_values)) > 1:
            self._refuse(422, "conflicting Authorization header values")
            return

        # Round-4 `/code-review` finding: a non-numeric or negative
        # `Content-Length` header (malformed client, not something the
        # round-3 body-shape validation below could ever see, since it
        # runs after this line) raised an uncaught `ValueError` here,
        # crashing the handler before any HTTP response was written --
        # the same "broken connection instead of a clean 4xx" failure
        # mode round 3 fixed for the body itself, just one field earlier.
        # Round-21 `/code-review` finding: a `Transfer-Encoding: chunked`
        # request has no `Content-Length` header at all -- the `or "0"`
        # fallback below used to silently treat that as a zero-length
        # body, which then failed with the confusing, unrelated "request
        # body is not valid JSON" 422 further down, leaving the actual
        # chunked body bytes unread on the socket. None of this repo's
        # real call sites currently send a chunked JSON body (the OpenAI
        # SDK's httpx client always sets `Content-Length` for a bounded
        # JSON payload), so this isn't a live gap today, but the failure
        # mode is confusing enough to name explicitly rather than let it
        # masquerade as a JSON-parsing error.
        # Round-22 `/code-review` finding: `self.headers.get(...)` only
        # returns the FIRST value of a header line that legally appears
        # more than once (e.g. `Transfer-Encoding: gzip` followed by a
        # separate `Transfer-Encoding: chunked` line) -- `.get_all()`
        # returns every value actually present. A single `.get()` call
        # here missed exactly this two-header-line case, falling through
        # to the same confusing "not valid JSON" 422 round 21 was fixing.
        transfer_encoding_values = self.headers.get_all("Transfer-Encoding") or []
        transfer_encoding_chunked = any("chunked" in v.lower() for v in transfer_encoding_values)
        if "Content-Length" not in self.headers and transfer_encoding_chunked:
            self._refuse(422, "chunked request bodies are not supported; send Content-Length")
            return
        # Round-28 `/code-review` finding: a request carrying BOTH
        # `Content-Length` and a chunked `Transfer-Encoding` is the
        # classic HTTP request-smuggling precondition (RFC 7230 SS3.3.3
        # mandates such a message be rejected, since the two headers
        # disagree about where the body ends). Before this check, the
        # combination fell through to whichever length `Content-Length`
        # claimed, silently ignoring `Transfer-Encoding` -- still
        # fail-safe against overspend (no real call site sends this
        # combination, and it either reads a truncated body that fails
        # `json.loads` or a body that happens to parse), but confusing,
        # and exactly the shape every other CL/TE inconsistency in this
        # handler is already refused for. Named and closed here for the
        # same reason those were.
        if "Content-Length" in self.headers and transfer_encoding_chunked:
            self._refuse(422, "Content-Length and chunked Transfer-Encoding must not both be present")
            return
        # Round-25 `/code-review` finding: `self.headers.get(...) or "0"`
        # conflates an ABSENT `Content-Length` header with one that is
        # PRESENT but empty (`Content-Length: `) -- the same "falsy
        # present value treated as absent" class rounds 8/9 already fixed
        # for `messages`. An empty-but-present header used to silently
        # read as a zero-length body (masking a real JSON body already
        # sitting on the socket, with the misleading "not valid JSON"
        # 422 further down) instead of hitting the numeric-parse
        # `ValueError` branch immediately below, which gives a clearer
        # "invalid Content-Length header" diagnostic. Check presence with
        # `in self.headers` instead.
        # Round-26 `/code-review` finding: `self.headers["Content-Length"]`
        # (like plain `.get()`) only returns the FIRST value of a header
        # name that appears on more than one line -- the same gap round
        # 22 already fixed for `Transfer-Encoding` via `.get_all()`, just
        # left unaddressed here. A duplicate `Content-Length` (two
        # different values) can't cause overspend today (the read length
        # comes from whichever value is picked, so the body either reads
        # short and fails `json.loads` -- refused before `reserve()` --
        # or reads the true length and parses fine), but silently picking
        # one of two conflicting values rather than refusing outright is
        # the same inconsistency class, closed here for the same reason.
        content_length_values = self.headers.get_all("Content-Length") or []
        if len(set(content_length_values)) > 1:
            self._refuse(422, f"conflicting Content-Length header values: {content_length_values!r}")
            return
        content_length_raw = content_length_values[0] if content_length_values else "0"
        try:
            length = int(content_length_raw)
        except ValueError:
            self._refuse(422, f"invalid Content-Length header: {content_length_raw!r}")
            return
        if length < 0:
            self._refuse(422, f"Content-Length must be non-negative, got {length}")
            return
        if length > _MAX_REQUEST_BODY_BYTES:
            self._refuse(413, f"Content-Length {length} exceeds the {_MAX_REQUEST_BODY_BYTES}-byte limit")
            return
        raw_body = self.rfile.read(length) if length else b""
        self._request_body_drained = True
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._refuse(422, f"request body is not valid JSON: {exc}")
            return

        # Round-3 `/code-review` finding: round 2 hardened how the
        # UPSTREAM response is read, but the INBOUND request from the
        # caller was never shape-checked the same way -- a non-dict top-
        # level body, a non-list `messages`, or a non-numeric
        # `max_tokens`/`max_completion_tokens` each crashed the handler
        # thread with an unhandled AttributeError/TypeError before any
        # HTTP response was written, leaving the real OpenAI SDK client
        # to see a broken connection instead of the clean 4xx its
        # `max_retries=0` handling is documented to expect. None of this
        # has touched `ledger.reserve()` yet, so refusing here costs
        # nothing in budget terms.
        if not isinstance(body, dict):
            self._refuse(422, f"request body must be a JSON object, got {type(body).__name__}")
            return
        # Round-8 `/code-review` finding, reproduced live: `messages =
        # body.get("messages") or []` treated ANY falsy-but-present value
        # (`0`, `""`, `False`) the same as an absent key, silently
        # defaulting to `[]` and skipping the `isinstance` check below
        # entirely -- a malformed `messages: 0` body was accepted rather
        # than refused, and the prompt-token estimate was silently
        # computed from an empty list instead. Only an actually ABSENT
        # key should default; anything else present must be validated.
        #
        # Round-9 `/code-review` finding, reproduced live: switching to
        # `body.get("messages")` and checking `is None` still couldn't
        # distinguish an actually-absent key from an explicit
        # `"messages": null` in the JSON body -- both produce `None` from
        # `.get()`. Check key PRESENCE with `in body` instead, so an
        # explicit `null` is treated as a present-but-wrong-type value
        # (refused) rather than silently defaulted to `[]`.
        if "messages" not in body:
            messages = []
        else:
            messages = body["messages"]
            if not isinstance(messages, list):
                self._refuse(422, f"'messages' must be a JSON array, got {type(messages).__name__}")
                return
            # Round-22 `/code-review` finding: `count_prompt_tokens()`
            # below stringifies non-string `content` (e.g. a multimodal
            # list-of-parts message carrying an `image_url`) via
            # `json.dumps()` and tokenizes THAT text -- a short URL string
            # tokenizes to a handful of tokens, while OpenAI actually
            # bills image content at a much higher, resolution-dependent
            # token count. That silently turns the pre-flight worst-case
            # estimate into a severe UNDER-estimate for exactly the kind
            # of call this module's whole design promises never to
            # under-count. No real call site in this repo sends
            # list-of-parts content today (verified) -- refuse it
            # outright rather than reserve an estimate this module has no
            # way to make safely, the same "refuse what can't be safely
            # estimated" rule `is_priced()` already enforces for an
            # unknown model.
            for message in messages:
                # Round-29 `/code-review` finding: the check below used to
                # test `isinstance(..., list)` only. A `content` value that
                # is some OTHER non-string JSON shape (e.g. a bare dict --
                # `{"type": "image_url", "image_url": {...}}` instead of
                # the documented list-of-parts) slipped past this refusal
                # and fell through to `count_prompt_tokens()`'s silent
                # `json.dumps()`-and-tokenize fallback, the exact
                # silent-under-estimate this refusal exists to prevent --
                # same failure class as the list case, just a different
                # container type. Refuse any non-string, non-missing
                # `content` outright rather than special-case each shape
                # one container type at a time.
                if (
                    isinstance(message, dict)
                    and "content" in message
                    and message["content"] is not None
                    and not isinstance(message["content"], str)
                ):
                    self._refuse(
                        422,
                        "message 'content' as a non-string, non-list value (e.g. multimodal/image content) "
                        "is not supported -- this proxy's pre-flight token/dollar estimate cannot safely "
                        "account for it",
                    )
                    return
                # Round-23 `/code-review` finding: same silent-under-
                # estimate class as the multimodal-list refusal just
                # above, for tool/function calling instead of images. An
                # assistant message can carry `tool_calls`/`function_call`
                # (function name + arguments -- real, sometimes large,
                # billed prompt tokens whenever that message is replayed
                # in a later conversation turn) with `content=null`;
                # `count_prompt_tokens()` only ever reads `content`/`role`,
                # so a `null` content is tokenized as the 4-character
                # string `"null"` while the real tool-call payload's
                # tokens are never counted at all. No real in-repo call
                # site sends this shape today (verified) -- refuse it
                # outright for the same "can't safely estimate it, so
                # don't guess" reason as the list-content refusal.
                if isinstance(message, dict) and ("tool_calls" in message or "function_call" in message):
                    self._refuse(
                        422,
                        "message 'tool_calls'/'function_call' fields are not supported -- this proxy's "
                        "pre-flight token/dollar estimate cannot safely account for them",
                    )
                    return
        # Round-23 `/code-review` finding: a top-level `tools`/
        # `functions` field (tool/function definitions -- OpenAI bills
        # these as real input tokens, sometimes hundreds to thousands of
        # them for a large JSON schema) is never read by
        # `count_prompt_tokens()` at all, the same silent-under-estimate
        # gap as the two message-level cases above, at the request level
        # instead of the message level. No real in-repo call site sends
        # `tools`/`functions` today (verified) -- refuse for the same
        # reason.
        #
        # Round-24 `/code-review` finding, reproduced live: this check
        # used to live INSIDE the `else:` branch of `if "messages" not in
        # body:` above (at the same indent as the `for message in
        # messages:` loop it follows), so it only ever ran when
        # `"messages"` was present in the body at all -- a request
        # carrying `tools`/`functions` but no `messages` key skipped the
        # refusal entirely and was priced/forwarded with `prompt_tokens`
        # estimated from an empty message list, the exact silent-under-
        # estimate this check exists to close, just reachable via a
        # different path. Moved outside (dedented past) the `if/else`
        # entirely so it runs unconditionally, regardless of whether
        # `messages` was given.
        if "tools" in body or "functions" in body:
            self._refuse(
                422,
                "'tools'/'functions' are not supported -- this proxy's pre-flight token/dollar "
                "estimate cannot safely account for tool/function definitions",
            )
            return
        # Round-25 `/code-review` finding (superseded, see below): `tools`/
        # `functions`, list-content, and `tool_calls`/`function_call` were
        # each found and blocked ONE AT A TIME across rounds 22-24 -- an
        # ever-growing blocklist of billing-relevant fields, rather than a
        # general allowlist, so any future OpenAI request field that bills
        # real extra tokens can silently slip through uncounted the same
        # way `tools`/`functions` did before round 23. Round 25 closed the
        # one concrete gap named at the time by refusing `response_format`
        # outright whenever its `type` is `json_schema`.
        #
        # Track-3 authorization (2026-09-12): that refusal's own premise --
        # "no real call site uses `json_schema`" -- no longer holds; a real
        # call site (the db-translation prompt in
        # `search_strategy_builder.py`) now needs OpenAI Structured Outputs
        # specifically to make missing required keys structurally
        # impossible rather than merely validated after the fact. Refusing
        # it outright is no longer the safe default; ESTIMATING it safely
        # is. Only the shape is checked here (malformed json_schema is
        # still refused, before dispatch, same as before); the actual
        # token estimate is folded into `prompt_tokens_estimate` below,
        # alongside `count_prompt_tokens()`, once `model` has been
        # resolved and validated.
        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema_obj = response_format.get("json_schema")
            if not isinstance(json_schema_obj, dict):
                self._refuse(422, "response_format.json_schema must be a JSON object")
                return
            schema_name = json_schema_obj.get("name")
            if not isinstance(schema_name, str) or not schema_name:
                self._refuse(422, "response_format.json_schema.name must be a non-empty string")
                return
            schema_obj = json_schema_obj.get("schema")
            if not isinstance(schema_obj, dict):
                self._refuse(422, "response_format.json_schema.schema must be a JSON object")
                return
            strict_flag = json_schema_obj.get("strict")
            if strict_flag is not None and not isinstance(strict_flag, bool):
                self._refuse(422, "response_format.json_schema.strict must be a boolean")
                return
            # Round-1 `/code-review` finding (Track-3 proxy-fix cycle):
            # `estimate_response_format_tokens()` below sizes a schema from
            # its raw `json.dumps()` length, padded 2x+32 -- conservative
            # for the flat, no-reference schema every real call site sends
            # today, but a schema using JSON Schema `$ref`/`$defs`/
            # `definitions` to reuse a substructure multiple times could
            # serialize small while OpenAI's actual (undocumented) internal
            # schema-to-token accounting bills the fully-expanded form,
            # which the padding multiplier has no principled reason to
            # cover. Refuse it outright now, before it becomes a real gap,
            # rather than trust an unproven multiplier against a shape this
            # module cannot safely estimate -- same "can't safely estimate
            # it, so don't guess" rule as every other blocklist entry here.
            # Checked by walking the parsed structure and looking for these
            # names in *key* position only (recursing into dict values and
            # list items), so a `$ref`/`$defs`/`definitions` key nested
            # anywhere inside the schema (e.g. under a `properties` entry)
            # is caught, but a schema that merely happens to use one of
            # these words as an ordinary property name or string/enum
            # value (round-2 `/code-review` finding: the prior substring-
            # on-`json.dumps()` check false-positived on exactly that) is
            # not.
            try:
                json.dumps(schema_obj)
            except (TypeError, ValueError) as exc:
                self._refuse(422, f"response_format.json_schema.schema is not JSON-serializable: {exc}")
                return
            if _contains_schema_reference_key(schema_obj):
                self._refuse(
                    422,
                    "response_format.json_schema.schema must not use '$ref'/'$defs'/'definitions' -- "
                    "this proxy's token estimate cannot safely account for schema reference expansion",
                )
                return
        # Round-30 `/code-review` finding: `stream: true` was never
        # detected or refused. This does not cause a cap under-count --
        # `requests.post()` below is never called with `stream=True`, so
        # the full SSE body is buffered before `upstream_resp.json()`
        # runs, that parse fails on the `data: ...\n\n` text, and the
        # existing `usage_verified=False` fallback correctly charges the
        # worst-case reservation, same as any other unparseable response
        # -- but the caller then gets back a 200 with a raw SSE blob as
        # its body under an `application/json` Content-Type, which no
        # real client can parse, breaking the call functionally. No real
        # in-repo call site sends `stream: true` today (verified), but
        # refuse it outright now rather than let a future caller hit a
        # confusing broken-response failure mode instead of a clear 422.
        if body.get("stream"):
            self._refuse(422, "'stream': true is not supported by this proxy")
            return
        # Round-32 `/code-review` finding: `modalities`/`audio` (gpt-4o
        # audio input/output) bills a substantially higher, entirely
        # separate audio-token rate that `pricing.PRICING` has no entry
        # for at all -- `count_prompt_tokens()`/`estimate_worst_case_
        # cost_usd()` would only ever account the text portion, silently
        # under-estimating the true cost the same way `tools`/`response_
        # format`/list-content did before rounds 22-25 closed them. No
        # real in-repo call site sends this shape today (verified), but
        # named and refused now for the same "can't safely estimate it,
        # so don't guess" reason as those.
        if "audio" in body or (isinstance(body.get("modalities"), list) and "audio" in body["modalities"]):
            self._refuse(
                422,
                "'audio'/'modalities' with audio input or output is not supported -- this proxy's "
                "pre-flight token/dollar estimate has no pricing data for the audio-token rate",
            )
            return
        model = body.get("model")
        if not isinstance(model, str):
            self._refuse(422, f"'model' must be a string, got {type(model).__name__}")
            return

        # Round-1 `/code-review` finding (Track-3 proxy-fix cycle): `call_id`
        # used to be generated further below (right before the `is_priced()`
        # check), but the new `response_format` sizing block just below this
        # needs one too (to log/call-log an internal-error refusal the same
        # way every other internal-error path in this handler does) -- moved
        # up here, once, so both blocks share the same single id for this
        # request instead of generating two or leaving the earlier one unset.
        call_id = server.next_call_id()

        # Track-3 authorization (2026-09-12): the shape of `response_format`
        # was already validated above (before `model` was resolved); the
        # actual worst-case token estimate needs a validated model string
        # (for `_encoding_for_model()`), so it is sized here, oversized/
        # unserializable schemas are refused before dispatch (never
        # forwarded), and a valid estimate is added to
        # `prompt_tokens_estimate` alongside `count_prompt_tokens()` below.
        response_format_tokens = 0
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            try:
                response_format_tokens = estimate_response_format_tokens(model, response_format)
            except ValueError as exc:
                self._refuse(422, f"response_format could not be safely estimated: {exc}")
                return
            except Exception as exc:
                # Round-1 `/code-review` finding (Track-3 proxy-fix cycle):
                # `estimate_response_format_tokens()` calls the same
                # `_encoding_for_model()` helper `count_prompt_tokens()`
                # uses, which round-13 already found can raise a non-
                # `ValueError` (e.g. a network/OS error while lazily
                # downloading tiktoken's BPE data the first time an
                # encoding is used) -- only `ValueError` was caught here,
                # so that failure mode propagated uncaught and crashed the
                # handler thread instead of returning a clean response,
                # reproducing round-13's exact bug class for this newer
                # code path. Same "log loudly, surface a clean 500" fix.
                logger.error("proxy: internal error estimating response_format tokens for "
                             "call_id=%s model=%s: %s", call_id, model, exc, exc_info=True)
                server.call_log.append(call_id=call_id, model=model, decision="internal_error", error=str(exc))
                self._refuse(500, f"internal error estimating response_format tokens: {exc}")
                return
            if response_format_tokens > _MAX_RESPONSE_FORMAT_SCHEMA_TOKENS:
                self._refuse(
                    422,
                    f"response_format schema is too large to safely estimate "
                    f"({response_format_tokens} padded tokens exceeds the "
                    f"{_MAX_RESPONSE_FORMAT_SCHEMA_TOKENS}-token limit)",
                )
                return
        # Round-7 `/code-review` finding: when both `max_completion_tokens`
        # and `max_tokens` are present, `max_completion_tokens` (the
        # modern parameter, the only one any of the 8 real call sites
        # ever sets) is authoritative and `max_tokens` is unconditionally
        # dropped from the forwarded body below -- so its VALUE is never
        # forwarded and never needs validating here.
        #
        # Round-8 `/code-review` finding, reproduced live: this used to
        # validate BOTH keys' types regardless of which one (if either)
        # is actually authoritative, so `{"max_completion_tokens": 500,
        # "max_tokens": "unlimited"}` was refused with a 422 even though
        # `max_tokens` was always going to be dropped, unread, and
        # therefore harmless. Determine which single key is authoritative
        # FIRST, and only validate that one.
        if "max_completion_tokens" in body:
            max_output_field = "max_completion_tokens"
        elif "max_tokens" in body:
            max_output_field = "max_tokens"
        else:
            max_output_field = None
        max_output_raw = body[max_output_field] if max_output_field is not None else None
        if max_output_raw is not None and (isinstance(max_output_raw, bool)
                                            or not isinstance(max_output_raw, (int, float))):
            self._refuse(422, f"'{max_output_field}' must be numeric, got {type(max_output_raw).__name__}")
            return
        # Round-15 `/code-review` finding: OpenAI's `n` field requests `n`
        # independently-billed completions -- must be validated the same
        # way `max_output_field` is, before the worst-case reservation
        # below is sized from it.
        n_raw = body.get("n")
        if n_raw is not None and (isinstance(n_raw, bool) or not isinstance(n_raw, (int, float))):
            self._refuse(422, f"'n' must be numeric, got {type(n_raw).__name__}")
            return

        if not pricing_mod.is_priced(model):
            # Round-2 `/code-review` finding: `UnpricedModelError` was
            # defined and re-exported but never actually raised anywhere
            # -- dead public API a maintainer could mistake for the real
            # enforcement mechanism. It IS that mechanism now: the refusal
            # message below is generated from the exception itself.
            exc = UnpricedModelError(model)
            server.call_log.append(call_id=call_id, model=model, decision="refused_unpriced")
            self._refuse(402, str(exc), model=model)
            return

        # Round-1 `/code-review` finding: 7 of the real call sites
        # (screening.py, screening_qc.py, search_strategy_builder.py,
        # synthesis.py) never set max_tokens/max_completion_tokens on
        # their own request, so the pre-flight worst-case estimate below
        # was a hopeful upper bound, not an enforced one -- an
        # unexpectedly large real completion could commit MORE than the
        # reservation held, overshooting the cap after the fact (this was
        # reproduced offline by the reviewer). Fix: when the caller didn't
        # cap its own output, THIS proxy injects the same default ceiling
        # into the actual forwarded request, so the real API call is
        # bounded to match the number the reservation was sized for.
        #
        # Round-4 `/code-review` finding: the fix above only checked for
        # `max_output is None`, but `pricing.resolve_max_output_tokens()`
        # (the SAME function the reservation below is sized with) also
        # falls back to the default ceiling for non-finite, negative, or
        # zero values -- those were reproduced live going out to the fake
        # upstream completely unmodified (`max_tokens=-5`, `nan`, `inf`,
        # `0`) while the reservation was quietly sized using 4096. Compare
        # against the SAME resolved value the reservation uses, and
        # overwrite whichever key(s) the caller actually supplied (falling
        # back to `max_completion_tokens` only when neither was given), so
        # the forwarded request can never diverge from what was reserved.
        #
        # Round-5 `/code-review` finding: a body carrying BOTH
        # `max_completion_tokens` and `max_tokens` (e.g. `{"max_completion_
        # tokens": None, "max_tokens": 5000}`) only had the FIRST-checked
        # key corrected, since `max_output` above is resolved from
        # whichever key `.get()`'s fallback chain picks -- the other key
        # was left forwarded unmodified, and it's undefined (and not this
        # proxy's job to know) which of the two a real upstream API would
        # actually honor. Rewrite EVERY output-limiting key present to the
        # same resolved value, so the forwarded request can't diverge from
        # the reservation no matter which key wins upstream.
        # Round-7 `/code-review` finding: round 6's fix forced BOTH keys
        # to the LARGER of the two resolved ceilings whenever either was
        # present -- but forwarding both `max_completion_tokens` AND
        # `max_tokens` in the same request is itself undocumented/risky
        # behavior some OpenAI model families are known to reject
        # outright, and "take the max" could silently upsize a caller's
        # intentionally small, cost-conscious `max_completion_tokens`
        # (the parameter all 8 real call sites in this codebase actually
        # use -- none of them ever set `max_tokens`) to match a stale or
        # unrelated `max_tokens` value sitting alongside it. Never forward
        # both keys: when both are present, treat `max_completion_tokens`
        # (the modern parameter) as authoritative, resolve/validate it
        # normally, and drop `max_tokens` from the forwarded body
        # entirely -- so the real API call is bounded by the SAME single
        # value the reservation is sized for, with no ambiguity about
        # which of two keys upstream honors.
        # Round-8 `/code-review` finding: this used to be a 4-way
        # if/elif/elif/else, one branch per (both-present, completion-
        # only, tokens-only, neither) combination, each hand-rewriting
        # "compare then assign then mark needs_rewrite" slightly
        # differently -- exactly the shape of duplication that let
        # rounds 5-7 each find a case one branch handled and another
        # didn't. `max_output_field`/`max_output_raw` above already
        # picked the one authoritative key (or `None`, `None` if
        # neither was given); there is now only ONE key left to ever
        # write, so there is only one path.
        resolved_max_output = pricing_mod.resolve_max_output_tokens(max_output_raw)
        needs_rewrite = False
        if max_output_field is None:
            body["max_completion_tokens"] = resolved_max_output
            needs_rewrite = True
        else:
            # Round-21 `/code-review` finding, reproduced live: `!=` alone
            # missed a whole-number FLOAT that already numerically equals
            # the resolved int (e.g. caller sends `4096.0`,
            # `resolve_max_output_tokens()` resolves to int `4096` --
            # `4096.0 != 4096` is False in Python, so `needs_rewrite`
            # stayed False and the original float literal was forwarded
            # verbatim). Round-20's int-coercion fix only helps once the
            # VALUE differs; a whole-number float slips through this
            # equality check untouched. Also rewrite whenever the
            # forwarded value isn't already a plain `int` (bools were
            # already refused above by the type check on `max_output_raw`,
            # so `isinstance(..., int)` here can't be fooled by one).
            if not isinstance(body[max_output_field], int) or body[max_output_field] != resolved_max_output:
                body[max_output_field] = resolved_max_output
                needs_rewrite = True
            other_field = "max_tokens" if max_output_field == "max_completion_tokens" else "max_completion_tokens"
            if other_field in body:
                del body[other_field]
                needs_rewrite = True
        # Round-15 `/code-review` finding: same "forwarded request must
        # match what the reservation was sized for" reasoning as
        # `max_output` above, applied to `n` -- a caller-supplied `n` that
        # `resolve_n()` treats as invalid (non-finite, non-positive) must
        # be rewritten to the value actually reserved, not left forwarded
        # unmodified, or the real API call could request more completions
        # than the reservation accounted for.
        # Round-16 `/code-review` finding proposed always injecting `n`
        # (even when absent) the same way `max_output_field` always
        # injects its default -- round-17 `/code-review` REVERTED that:
        # confirmed live -- the call sites routed through
        # `openai_runner.py` (`graphs/rapid.py`,
        # `adapters/scoping_charting_adapter.py`,
        # `reference_charting_adapter.py`) AND the ones that construct
        # their own `OpenAI()` client directly (`nodes/screening.py`,
        # `nodes/screening_qc.py`, `nodes/search_strategy_builder.py`,
        # `nodes/synthesis.py`, `nodes/quality_appraisal.py` -- round-18
        # `/code-review` widened the audit to these after round 17's
        # comment only named the `openai_runner.py`-routed ones) -- that
        # NONE of them ever send `n` at all. Round-19 `/code-review`
        # correction: `adapters/ft_screening_adapter.py` does NOT itself
        # call OpenAI (its own docstring says so) -- the real call for
        # that flow is `nodes/fulltext.py`'s `run_ft_screening()` ->
        # `resolve_components()` -> the separate FT_SCREENING package's
        # `ft_screening/adjudication.py`'s `OpenAIChatCompletionTransport`,
        # OUTSIDE this repo entirely. It doesn't send `n` today either
        # (checked directly in that package), but it is NOT covered by
        # this proxy's own test suite or this audit's guarantees the way
        # the in-repo call sites are -- a future change there could add
        # `n` with zero warning from anything in THIS repo. Likewise
        # `nodes/llm_backends.py`'s `_backend_openai()` constructs an
        # `OpenAI()` client but is currently dead code (imported, never
        # called, in-repo) -- not a live gap today, but not "audited and
        # verified inert" either if it's ever wired up. This codebase
        # already special-cases
        # `max_tokens`/`max_completion_tokens` for the `gpt-5.6-terra`
        # reasoning-effort model family specifically because reasoning
        # models have a real history of rejecting certain sampling
        # parameters outright, even at their documented default value.
        # Injecting a novel `n` key into every real request risked
        # breaking production extraction/QC/charting calls the test
        # suite can't catch (it only talks to a fake upstream that
        # doesn't validate `n`) -- an enforcement gap that is only ever
        # reachable by a caller explicitly setting an invalid `n` is a
        # smaller, more acceptable risk than a live 400 on every real
        # call. Only rewrite `n` when the caller actually supplied one.
        resolved_n = pricing_mod.resolve_n(n_raw)
        # Round-21 `/code-review` finding: same whole-number-float gap as
        # `max_output_field` just above -- `body["n"] != resolved_n` alone
        # misses a caller-supplied `n=1.0` comparing equal to resolved
        # int `1`, forwarding the float unmodified. Same fix.
        if "n" in body and (not isinstance(body["n"], int) or body["n"] != resolved_n):
            body["n"] = resolved_n
            needs_rewrite = True
        if needs_rewrite:
            raw_body = json.dumps(body).encode("utf-8")
        max_output = resolved_max_output

        try:
            # Round-13 `/code-review` finding: `count_prompt_tokens()`
            # calls `tiktoken.encoding_for_model()`/`get_encoding()`, which
            # lazily downloads its BPE data file over the network the
            # first time a given encoding is used if it isn't already
            # cached locally -- this proxy exists specifically to fence
            # off/monitor all LLM network egress for the live run, so
            # general internet access in that environment may well be
            # restricted. An uncaught exception here (or from the pricing
            # arithmetic just below) used to crash the handler thread
            # before any HTTP response was written -- the same "broken
            # connection instead of a clean 4xx/5xx" class of bug every
            # earlier round (2, 3, 4, 7, 8, 9) fixed for other inputs,
            # left unguarded here.
            prompt_tokens_estimate = count_prompt_tokens(model, messages) + response_format_tokens
            worst_case_usd = pricing_mod.estimate_worst_case_cost_usd(
                model=model, prompt_tokens=prompt_tokens_estimate, max_output_tokens=max_output, n=resolved_n,
            )
        except Exception as exc:
            # Round-14 `/code-review` finding: this failure (and the
            # `reserve()` `ValueError` one below) used to be recorded ONLY
            # into the call-log JSONL, unlike every other failure path in
            # this module (persist failures, call-log write failures,
            # reservation overshoot), which all also log loudly via
            # `logger`. During an unattended live run, a systemic bug here
            # (e.g. a corrupted local tiktoken cache) would otherwise
            # silently refuse every request with no console/stderr trace
            # at all -- nobody watching the run's log stream would notice
            # until it visibly stalled or finished with far fewer allowed
            # calls than expected.
            logger.error("proxy: internal error sizing the reservation for call_id=%s model=%s: %s",
                          call_id, model, exc, exc_info=True)
            server.call_log.append(call_id=call_id, model=model, decision="internal_error", error=str(exc))
            self._refuse(500, f"internal error sizing the reservation: {exc}")
            return

        try:
            server.ledger.reserve(call_id, worst_case_usd)
        except BudgetExceededError as exc:
            server.call_log.append(
                call_id=call_id, model=model, decision="refused_budget",
                estimated_usd=worst_case_usd, remaining_usd=exc.remaining_usd, cap_usd=exc.cap_usd,
            )
            self._refuse(402, str(exc), estimated_usd=worst_case_usd, remaining_usd=exc.remaining_usd,
                         cap_usd=exc.cap_usd)
            return
        except ValueError as exc:
            # Round-5 `/code-review` finding: `reserve()` also raises a bare
            # `ValueError` for two internal-invariant failures that are NOT
            # a legitimate caller-facing budget refusal -- a non-finite
            # `worst_case_usd` reaching this point (should be impossible,
            # `estimate_worst_case_cost_usd()`/`resolve_max_output_tokens()`
            # above are supposed to prevent it) or a `call_id` collision
            # (this handler generates a fresh UUID per request; a collision
            # would mean that generation itself is broken). Before this
            # fix, this `ValueError` propagated uncaught out of
            # `_handle_post`, crashing the connection instead of returning
            # a clean response -- surface it as a 500, distinct from the
            # 402/422 refusal contract, since it reflects a bug in this
            # proxy, not a problem with the caller's request.
            # Round-14 `/code-review` finding: same "must log loudly, not
            # just to the call-log JSONL" reasoning as the tokenization
            # `except Exception` above.
            logger.error("proxy: internal error reserving budget for call_id=%s model=%s: %s",
                          call_id, model, exc, exc_info=True)
            server.call_log.append(call_id=call_id, model=model, decision="internal_error", error=str(exc))
            self._refuse(500, f"internal error reserving budget: {exc}")
            return

        # Round-1 `/code-review` finding: every exit below this point MUST
        # resolve `call_id`'s reservation exactly once (commit or release)
        # or budget capacity leaks forever. `resolved` + `finally` is the
        # safety net for any path that forgets to, or for an exception
        # raised somewhere unexpected between here and the last explicit
        # resolution.
        resolved = False
        # Round-15 `/code-review` finding: the `finally` safety net below
        # used to unconditionally `release()` whenever `resolved` was still
        # False -- correct for an exception BEFORE the upstream ever
        # returned a response (nothing could have been billed), but wrong
        # for an exception AFTER a real 200 came back (e.g. `actual_cost_
        # usd()`/`ledger.commit()` itself raising something not already
        # caught by the `usage_verified` try/except above): the call may
        # genuinely have been billed, so releasing would under-count real
        # spend, the exact failure mode every other fail-safe branch in
        # this handler (round 1's unverifiable-usage fallback, round 12's
        # `RequestException` fallback) exists to avoid. This flag lets the
        # `finally` block tell the two situations apart.
        upstream_returned_200 = False

        # Round-30 `/code-review` finding: the `RequestException` branch
        # (round 12) and the catch-all `Exception` branch (round 29) below
        # ran the identical commit(worst_case, usage_verified=False) /
        # call_log.append / refuse(502) sequence, differing only in the
        # decision string and error message -- exactly the kind of
        # near-duplicated branch where rounds 5-8 and 12-14 previously
        # found divergence bugs after only one copy got updated. Factored
        # into one helper so any future change to the fail-safe sequence
        # only has to be made once.
        def _commit_worst_case_and_log(decision: str, error: Optional[str] = None) -> None:
            # Round-42 `/code-review` finding: this commit+log pair used
            # to be duplicated verbatim in two places -- inside
            # `_charge_worst_case_and_refuse()` below (pre-response error
            # paths, which also sends a 502) and again, inline, in the
            # outer `finally` block's post-200 branch (which CANNOT call
            # `_refuse()` there, since a 200 was already sent -- the two
            # call sites' charge+log halves are identical, only the
            # refuse half differs). Same near-duplicated-branch class
            # rounds 5-8, 12-14, and 30 already found and fixed elsewhere
            # in this file. Factored out so a future change to the
            # fail-safe charge+log sequence only has to be made once.
            #
            # Round-31 `/code-review` finding: this used to leave `resolved
            # = True` to the caller, set only AFTER this whole helper
            # returned -- unlike every other resolved-path in this
            # handler, which sets it immediately after the ledger commit/
            # release call and before any later step that can raise (e.g.
            # `self._refuse()`'s socket write, if the caller already
            # disconnected). Reproduced live: an `OSError` from that write
            # used to propagate past the caller's unreached `resolved =
            # True` line into the outer `finally` block, which then saw
            # `resolved=False` and called `ledger.release()` (a harmless
            # no-op post-commit -- no money lost or duplicated) plus a
            # SECOND, contradictory `call_log.jsonl` entry for the same
            # call_id, corrupting the audit trail this module exists to
            # keep trustworthy. Set `resolved` here, right after the
            # commit that actually matters, matching every other branch's
            # ordering.
            nonlocal resolved
            # Round-42 `/code-review` finding: `completion_tokens=0` here
            # while `actual_usd=worst_case_usd` (which prices
            # `max_output * resolved_n` output tokens, per
            # `estimate_worst_case_cost_usd()`'s own formula) made the
            # persisted `_calls[]` audit record self-contradictory -- a
            # nonzero dollar figure attributed to zero completion tokens,
            # unreconstructable by anyone auditing `ledger.json`. Record
            # the same token count the dollar figure was actually priced
            # from.
            server.ledger.commit(
                call_id, model=model, prompt_tokens=prompt_tokens_estimate,
                completion_tokens=max_output * resolved_n,
                cached_tokens=0, actual_usd=worst_case_usd, usage_verified=False,
            )
            resolved = True
            kwargs = {"error": error} if error is not None else {}
            server.call_log.append(call_id=call_id, model=model, decision=decision,
                                    estimated_usd=worst_case_usd, actual_usd=worst_case_usd, **kwargs)

        def _charge_worst_case_and_refuse(exc: Exception, decision: str, detail: str) -> None:
            _commit_worst_case_and_log(decision, error=str(exc))
            self._refuse(502, detail)

        try:
            # Round-28 `/code-review` finding: this used to re-fetch
            # `Authorization` with a fresh `self.headers.get(...)` call
            # instead of reusing `authorization_values` from the round-27
            # check above -- safe today only because that check already
            # guarantees no conflict exists by the time this line runs,
            # but not structurally coupled to it, so a future edit that
            # moved or removed the check could silently reintroduce the
            # drop-second-header bug. Reuse the validated list, mirroring
            # the `content_length_values[0]` pattern used two blocks down.
            forwarded_headers = {
                "Authorization": authorization_values[0] if authorization_values else "",
                "Content-Type": "application/json",
            }
            try:
                upstream_resp = requests.post(
                    server.upstream_url, data=raw_body, headers=forwarded_headers,
                    timeout=server.request_timeout_s,
                )
            except requests.exceptions.ConnectTimeout as exc:
                # Round-13 `/code-review` finding: the round-12 blanket
                # "charge worst case for any RequestException" fix has its
                # own failure scenario -- a sustained connect-level outage
                # (DNS failure, firewall drop, target unreachable) makes
                # EVERY call fail this way, and charging worst-case for
                # each one can exhaust the entire $30 cap on calls that
                # provably sent zero bytes to the upstream, halting the run
                # with no real OpenAI spend at all. `ConnectTimeout` is the
                # one `RequestException` subclass `requests` raises ONLY
                # when the TCP connect itself times out -- by definition,
                # before any request bytes could have been sent -- so it is
                # safe to release here without reintroducing the
                # under-counting risk the round-12 fix exists to close.
                # A plain `ConnectionError` (connection refused, DNS
                # failure) is NOT split out the same way: `requests`
                # doesn't reliably distinguish "refused before any bytes
                # sent" from "reset mid-transfer after the request was
                # already delivered," so it stays in the fail-safe
                # (charge worst case) branch below rather than risk
                # silently under-counting a call that really was billed.
                #
                # Round-14 `/code-review` finding, confirmed via
                # `requests.exceptions.ConnectTimeout.__mro__`: plain
                # `ConnectionError` is NOT a subclass of `ConnectTimeout`
                # (only the reverse), so this carve-out does NOT catch the
                # more common real-world outage shape (immediate
                # connection-refused/DNS-failure `ConnectionError`, no
                # slow timeout involved) -- a sustained outage of THAT
                # kind can still exhaust the cap on zero real spend. This
                # is a deliberate, informed trade-off, not an oversight:
                # `requests`/`urllib3` don't expose a reliable, forward-
                # compatible way to tell a pre-send `ConnectionError` from
                # a mid-transfer one without inspecting private exception-
                # chain internals, and the two failure directions are NOT
                # symmetric -- exhausting the cap on a real outage is
                # visible and recoverable (re-authorize once the network is
                # back), while silently under-counting a call that WAS
                # billed is neither. Kept conservative on purpose.
                server.ledger.release(call_id)
                resolved = True
                server.call_log.append(call_id=call_id, model=model, decision="upstream_connect_timeout",
                                        error=str(exc))
                self._refuse(502, f"upstream request failed: {exc}")
                return
            except requests.RequestException as exc:
                # Round-12 `/code-review` finding: a network exception here
                # (timeout, connection reset) does not prove the upstream
                # never received/processed/billed the request -- a
                # `ReadTimeout` in particular fires only AFTER the request
                # was fully sent and the server had already started (or
                # finished) generating a response. The old `release()` here
                # assumed "never billed," which could silently under-track
                # real spend for the rest of the run if the call actually
                # was billed server-side. Same fail-safe direction already
                # used a few lines below for an unparseable/unverifiable
                # response body: when the true outcome can't be verified,
                # charge the worst-case reservation rather than assume
                # nothing was spent -- the ledger may then overstate spend
                # slightly, never understate it.
                # `resolved` is set inside `_charge_worst_case_and_refuse()`
                # itself now (round-31 fix), immediately after the commit
                # -- not here, after the potentially-raising `_refuse()`
                # call the helper also makes.
                _charge_worst_case_and_refuse(
                    exc, "upstream_network_error_charged_worst_case", f"upstream request failed: {exc}",
                )
                return
            except Exception as exc:
                # Round-29 `/code-review` finding: the two `except` clauses
                # above only catch `ConnectTimeout` (provably pre-send) and
                # `requests.RequestException` (the whole family `requests`
                # normally raises for network/protocol failures). Any
                # exception type OUTSIDE that hierarchy -- e.g. a raw
                # socket-level error surfacing through urllib3 without
                # being wrapped, per this round's review -- fell straight
                # through to the outer `finally` block below with
                # `resolved` still False, which RELEASES the reservation:
                # correct only if request bytes were never sent, but this
                # exception could just as easily fire AFTER the request was
                # transmitted and billed server-side. Same fail-safe
                # direction as the `RequestException` branch just above:
                # when the true outcome can't be verified, charge the
                # worst-case reservation rather than assume nothing was
                # spent.
                # `resolved` is set inside the helper itself -- see the
                # comment on the call site above.
                _charge_worst_case_and_refuse(
                    exc, "upstream_unexpected_error_charged_worst_case",
                    f"upstream request failed unexpectedly: {exc}",
                )
                return

            if upstream_resp.status_code != 200:
                server.ledger.release(call_id)
                resolved = True
                server.call_log.append(
                    call_id=call_id, model=model, decision="upstream_error",
                    upstream_status=upstream_resp.status_code,
                )
                self._respond_raw(upstream_resp.status_code, upstream_resp.content,
                                   upstream_resp.headers.get("Content-Type", "application/json"))
                return

            # Round-15 `/code-review` finding: from here on, the upstream
            # HAS returned a real 200 -- billing may already have occurred
            # server-side regardless of what happens next in this handler.
            upstream_returned_200 = True

            # Round-1 `/code-review` finding: a 200 whose body is
            # unparseable, or whose `usage` field is missing/incomplete,
            # gives no reliable actual token count -- committing a
            # near-zero fallback (the old behavior) silently under-charges
            # real spend. Safe direction: when actual usage can't be
            # verified, charge the FULL worst-case reservation rather than
            # guess low -- the ledger may then overstate spend slightly,
            # never understate it.
            # Round-2 `/code-review` finding: the previous version only
            # guarded the "is this valid JSON, and is it a dict" shape --
            # a `usage` field that IS present but isn't itself a dict (a
            # string, number, list), or `prompt_tokens`/`completion_tokens`
            # values that aren't numeric, still raised an uncaught
            # AttributeError/ValueError from `.get()`/`int()` further down,
            # which propagated to the outer `finally` and RELEASED the
            # reservation instead of charging the safe worst-case fallback
            # -- exactly the under-charge this whole branch exists to
            # prevent. Every field this block reads is now type-checked
            # before use, so no value from an untrusted response can ever
            # escape this `try` as anything other than "usage_verified
            # False, charge worst case".
            try:
                response_json = upstream_resp.json()
                if not isinstance(response_json, dict):
                    raise ValueError("response body is valid JSON but not a JSON object")
                usage = response_json.get("usage")
                if not isinstance(usage, dict):
                    raise ValueError("usage field is missing or not an object")
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if not pricing_mod.is_finite_number(prompt_tokens):
                    raise ValueError("usage.prompt_tokens is missing, not numeric, or not finite")
                if not pricing_mod.is_finite_number(completion_tokens):
                    raise ValueError("usage.completion_tokens is missing, not numeric, or not finite")
                cached_details = usage.get("prompt_tokens_details")
                cached_raw = cached_details.get("cached_tokens", 0) if isinstance(cached_details, dict) else 0
                cached_tokens = int(cached_raw) if pricing_mod.is_finite_number(cached_raw) else 0
                prompt_tokens = int(prompt_tokens)
                completion_tokens = int(completion_tokens)
                # Round-20 `/code-review` finding: `is_finite_number()` only
                # rejects NaN/inf/huge-int/wrong-type -- it does NOT reject a
                # negative value. `actual_cost_usd()` clamps negative counts
                # to 0 internally (round-2 fix), so the DOLLAR amount was
                # always safe -- but the raw, still-negative `prompt_tokens`/
                # `completion_tokens`/`cached_tokens` were forwarded verbatim
                # to `ledger.commit()` below and persisted byte-for-byte into
                # the ledger's own audit trail (`calls[]` in ledger.json),
                # corrupting the exact record this module exists to produce
                # for later use. Clamp here too, so the audit trail agrees
                # with the dollar figure computed from it.
                prompt_tokens = max(prompt_tokens, 0)
                completion_tokens = max(completion_tokens, 0)
                cached_tokens = max(cached_tokens, 0)
                cached_tokens = min(cached_tokens, prompt_tokens)
                usage_verified = True
            except ValueError:
                usage_verified = False

            if usage_verified:
                actual_usd = pricing_mod.actual_cost_usd(
                    model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                )
                server.ledger.commit(
                    call_id, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens, actual_usd=actual_usd,
                )
                resolved = True
                server.call_log.append(
                    call_id=call_id, model=model, decision="allowed", estimated_usd=worst_case_usd,
                    actual_usd=actual_usd, prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            else:
                # Round-43 `/code-review` finding: this used to duplicate
                # `_commit_worst_case_and_log()`'s commit+log sequence by
                # hand instead of calling the helper created in round 42
                # specifically to prevent this class of divergence (already
                # bitten twice: rounds 5-8/12-14/30 found one duplicated
                # copy fixed while another was missed, then round 42 itself
                # repeated that pattern here). No behavior change --
                # `decision` and `error` (omitted) match exactly what this
                # inline code produced.
                _commit_worst_case_and_log("allowed_usage_unverified_charged_worst_case")
            self._respond_raw(200, upstream_resp.content, upstream_resp.headers.get("Content-Type", "application/json"))
        finally:
            if not resolved:
                # Round-42 `/code-review` finding: neither branch below
                # used to capture or log the actual exception that put the
                # handler into this `finally` path -- unlike every other
                # internal-error path in this module (tokenization,
                # `reserve()`, the `RequestException`/generic-`Exception`
                # handlers), which all call `logger.error(...,
                # exc_info=True)` and record `error=str(exc)`. During an
                # unattended live run, only Python's default stderr
                # traceback (not part of this module's own structured
                # audit trail) would show what actually happened.
                # `sys.exc_info()` still sees the in-flight exception here
                # because a bare `finally:` (no enclosing `except`) runs
                # during unwind, before the exception is cleared.
                _, exc_value, _ = sys.exc_info()
                if upstream_returned_200:
                    # Round-15 `/code-review` finding: an unexpected
                    # exception AFTER a real 200 came back must not be
                    # treated as "nothing was spent" -- same fail-safe
                    # direction as every other unverifiable-outcome branch
                    # in this handler (charge the worst-case reservation,
                    # `usage_verified=False`, rather than silently
                    # under-count a call that may genuinely have been
                    # billed).
                    logger.error(
                        "proxy: unexpected exception after upstream 200 for call_id=%s model=%s: %s",
                        call_id, model, exc_value, exc_info=True,
                    )
                    try:
                        _commit_worst_case_and_log(
                            "charged_worst_case_unexpected_exception_after_200",
                            error=str(exc_value) if exc_value is not None else None,
                        )
                    except ValueError:
                        # Round-19 `/code-review` finding: `commit()` here
                        # raising "no open reservation" means an EARLIER
                        # `commit()` call (the one that got us into this
                        # `finally` block in the first place, e.g. the
                        # `usage_verified` branch above) already popped the
                        # reservation and updated the ledger's committed
                        # total before something else made execution fall
                        # through to this `finally` block with `resolved`
                        # still False. The cap accounting is already
                        # correct -- there is nothing left to charge.
                        # Swallowing this secondary `ValueError` here,
                        # rather than letting it propagate, matters because
                        # an exception raised inside a `finally` block
                        # REPLACES whatever exception Python was already
                        # propagating out of it, which would otherwise mask
                        # the real root cause from whoever ends up
                        # debugging the resulting connection teardown.
                        #
                        # Round-44 `/code-review` finding: this comment
                        # used to name a `RecursionError`/`MemoryError`
                        # escaping `_persist_locked()` inside that earlier
                        # `commit()` call as the illustrative "something
                        # else" -- round 43's fix (`ledger.py`'s `commit()`
                        # now wraps its own persist call in `except
                        # Exception`) closed that specific pathway; a
                        # persist failure of any kind can no longer make
                        # `commit()` raise. The `except ValueError` here
                        # stays: the general shape (an earlier `commit()`
                        # for this call_id already resolved the reservation
                        # before something unrelated to persistence lands
                        # execution here) is not tied to persistence and
                        # remains possible -- only the specific example
                        # named in this comment was stale.
                        logger.error(
                            "proxy: finally-block commit() for call_id=%s found no open reservation -- "
                            "an earlier commit() attempt already resolved it before something else made "
                            "IT raise too; the ledger's own accounting should already be correct",
                            call_id,
                        )
                else:
                    logger.error(
                        "proxy: unexpected exception for call_id=%s model=%s before any upstream response -- "
                        "releasing reservation (no request was ever billed): %s",
                        call_id, model, exc_value, exc_info=True,
                    )
                    server.ledger.release(call_id)
                    error_kwargs = {"error": str(exc_value)} if exc_value is not None else {}
                    server.call_log.append(
                        call_id=call_id, model=model, decision="released_unexpected_exception", **error_kwargs
                    )


class CostCapProxy:
    """Context-manager wrapper, same shape as `ReplayServer`: binds a
    loopback socket (port 0 -> OS-assigned), serves in a daemon thread."""

    def __init__(self, ledger: CostLedger, call_log_path: Path, *, host: str = "127.0.0.1", port: int = 0,
                 upstream_url: str = _REAL_OPENAI_CHAT_COMPLETIONS_URL, request_timeout_s: float = 180.0,
                 drain_timeout_s: float = 200.0) -> None:
        self._ledger = ledger
        self._call_log = _CallLogWriter(call_log_path)
        self._host = host
        self._requested_port = port
        self._upstream_url = upstream_url
        self._request_timeout_s = request_timeout_s
        self._drain_timeout_s = drain_timeout_s
        self._httpd: Optional[_CostCapHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "CostCapProxy":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        self._httpd = _CostCapHTTPServer(
            (self._host, self._requested_port), _CostCapRequestHandler,
            ledger=self._ledger, call_log=self._call_log,
            upstream_url=self._upstream_url, request_timeout_s=self._request_timeout_s,
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            # shutdown() stops the accept loop -- no NEW connections are
            # dispatched after this -- but does nothing about handler
            # threads already dispatched for in-flight requests (they run
            # independently, `daemon_threads = True`). Wait for those to
            # finish (round-1 `/code-review` finding) before closing the
            # socket, so a request that already spent real money always
            # reaches `ledger.commit()`/`release()` before the caller
            # (e.g. a subprocess-per-run driver) can exit and kill it.
            self._httpd.shutdown()
            if not self._httpd.wait_for_idle(self._drain_timeout_s):
                logger.warning(
                    "CostCapProxy.stop(): %d request(s) still in flight after %.0fs drain timeout -- "
                    "closing anyway; any in-flight spend may not be recorded in the ledger",
                    self._httpd._active_requests, self._drain_timeout_s,
                )
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def port(self) -> int:
        assert self._httpd is not None, "CostCapProxy.start() was never called"
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self.port}/v1"
