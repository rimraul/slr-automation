"""
slr_pipeline.replay.transport -- the loopback HTTP replay server (Track 2
plan Sec 6.1) every real `OpenAI(api_key=...)` client in the graph talks to
once `OPENAI_BASE_URL` points at it (plan Sec 6.2 -- wiring the subprocess
child's environment is a later step; this module is the server itself).

Contract (plan Sec 6.1, verbatim):
- `POST /v1/chat/completions` only -- exact path, no other method/path.
  Anything else -> latched fatal.
- Headers validated, not signed: `Host == 127.0.0.1:<port>`,
  `Authorization == "Bearer sk-replay-not-a-real-key"`. Mismatch -> latched
  fatal.
- A match -> HTTP 200, `response_body` verbatim. Any contract violation ->
  HTTP 422 (the SDK does not retry 422) AND a latched fatal state (every
  subsequent request also 422 + recorded) -- several nodes catch provider
  errors and silently fall back to heuristics, so a transport that failed
  once must never look "successful" again.
- `call_log.jsonl`: a genuine JSONL file, one JSON object per line,
  appended under a lock with `f.flush()` + `os.fsync()` per line. Each
  line: `{seq, surface, signature, matched_entry_id, ts_monotonic}`.

Record-gaps authoring (`scripts/author_reference_cassette.py`) is a later,
step-12 deliverable built on top of this server -- out of scope here.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from slr_pipeline.replay.cassette import ReplayCassette, ReplayCassetteEntry
from slr_pipeline.replay.signature import REPLAY_ENDPOINT_PATH, compute_replay_signature

_EXPECTED_AUTH = "Bearer sk-replay-not-a-real-key"


class ReplayTransportError(RuntimeError):
    """Raised by `ReplayServer` bookkeeping (never sent over the wire) --
    e.g. `assert_complete()` finding an entry whose `served_calls !=
    expected_calls` (plan Sec 6.1/11.2 teardown)."""


class _CallLogWriter:
    """Thread-locked, append-only `call_log.jsonl` writer (plan Sec 6.1 --
    "a genuine JSONL file ... appended with a thread lock + f.flush() +
    os.fsync() per line")."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, surface: str, signature: str, matched_entry_id: Optional[str]) -> int:
        with self._lock:
            self._seq += 1
            line = {
                "seq": self._seq,
                "surface": surface,
                "signature": signature,
                "matched_entry_id": matched_entry_id,
                "ts_monotonic": time.monotonic(),
            }
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return self._seq


class _ReplayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler_cls, *, cassette: ReplayCassette, call_log: _CallLogWriter) -> None:
        super().__init__(address, handler_cls)
        self.cassette = cassette
        self.call_log = call_log
        self.by_signature: Dict[str, ReplayCassetteEntry] = cassette.by_signature()
        self.served_calls: Dict[str, int] = {e.entry_id: 0 for e in cassette.entries}
        self.host_header: str = f"{address[0]}:{self.server_address[1]}"
        self.latched_fatal = False
        self.latch_reason: Optional[str] = None
        self.state_lock = threading.Lock()


class _ReplayRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _ReplayHTTPServer  # narrows the inherited `self.server` type

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- stdlib signature
        pass  # silence default stderr request logging

    def _respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _latch(self, status: int, reason: str) -> None:
        self.server.latched_fatal = True
        self.server.latch_reason = reason
        self._respond(status, {"error": reason})

    def _handle_non_post(self) -> None:
        with self.server.state_lock:
            self._latch(422, f"unsupported method {self.command!r} (only POST is permitted)")

    do_GET = _handle_non_post
    do_PUT = _handle_non_post
    do_DELETE = _handle_non_post
    do_PATCH = _handle_non_post
    do_HEAD = _handle_non_post
    do_OPTIONS = _handle_non_post

    def do_POST(self) -> None:
        server = self.server
        with server.state_lock:
            if server.latched_fatal:
                self._respond(422, {"error": f"latched fatal: {server.latch_reason}"})
                return
            if self.path != REPLAY_ENDPOINT_PATH:
                self._latch(422, f"unknown endpoint {self.path!r}, expected {REPLAY_ENDPOINT_PATH!r}")
                return
            if self.headers.get("Host") != server.host_header:
                self._latch(422, f"Host header mismatch: got {self.headers.get('Host')!r}, "
                                  f"expected {server.host_header!r}")
                return
            if self.headers.get("Authorization") != _EXPECTED_AUTH:
                self._latch(422, "Authorization header mismatch")
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._latch(422, f"request body is not valid JSON: {exc}")
                return
            signature = compute_replay_signature(self.path, body)
            entry = server.by_signature.get(signature)
            if entry is None:
                self._latch(422, f"unrecorded signature {signature}")
                return
            server.served_calls[entry.entry_id] += 1
            server.call_log.append(surface=entry.surface, signature=signature, matched_entry_id=entry.entry_id)
            response_body = entry.response_body
        self._respond(200, response_body)


class ReplayServer:
    """Context-manager wrapper around `_ReplayHTTPServer`: binds a loopback
    socket (port 0 => OS-assigned, the default for tests), serves in a
    daemon thread, and exposes the bookkeeping the reference-run harness
    needs after the run (plan Sec 6.1/11.2 teardown)."""

    def __init__(self, cassette: ReplayCassette, call_log_path: Path, *, host: str = "127.0.0.1", port: int = 0):
        self._cassette = cassette
        self._call_log = _CallLogWriter(call_log_path)
        self._host = host
        self._requested_port = port
        self._httpd: Optional[_ReplayHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ReplayServer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        self._httpd = _ReplayHTTPServer(
            (self._host, self._requested_port), _ReplayRequestHandler,
            cassette=self._cassette, call_log=self._call_log,
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def port(self) -> int:
        assert self._httpd is not None, "ReplayServer.start() was never called"
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self.port}/v1"

    @property
    def latched_fatal(self) -> bool:
        assert self._httpd is not None
        return self._httpd.latched_fatal

    @property
    def latch_reason(self) -> Optional[str]:
        assert self._httpd is not None
        return self._httpd.latch_reason

    def served_calls(self) -> Dict[str, int]:
        assert self._httpd is not None
        return dict(self._httpd.served_calls)

    def assert_complete(self) -> None:
        """Plan Sec 6.1/11.2 teardown: any cassette entry with
        `served_calls != expected_calls` is a harness failure."""
        assert self._httpd is not None
        mismatches = {
            entry.entry_id: {"served": self._httpd.served_calls[entry.entry_id], "expected": entry.expected_calls}
            for entry in self._cassette.entries
            if self._httpd.served_calls[entry.entry_id] != entry.expected_calls
        }
        if mismatches:
            raise ReplayTransportError(f"cassette entries with served_calls != expected_calls: {mismatches}")
