"""
slr_pipeline.reference_netguard -- the socket allowlist Track 2's
reference-run child process installs before any pipeline/third-party
import (plan `track2_pipeline_readiness_plan_2026-08-30.md` Sec 12.3):
"The socket allowlist is installed before any pipeline / third-party
import (`reference_run_child.py`'s first statements), permits only
`127.0.0.1:<replay_port>`, and appends to `<ws>/violations.jsonl` before
raising."

Once armed, only a `connect()` to the single allowed `(host, port)` pair
succeeds; any other attempt is BLOCKED, appended to `violations.jsonl`
(plan Sec 11.2: "a non-loopback `connect` -> `violations.jsonl` non-empty
-> run failure"), and re-raised as `NetGuardViolation` so the caller sees a
genuine connection failure rather than a silent stall or a fallback to a
real network call.

This module owns the mechanism only. Wiring it into
`reference_run_child.py`'s first statements is a later step; here it is a
self-contained, directly arm/disarm-able guard any test or driver can use.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

_real_socket_connect = socket.socket.connect


class NetGuardViolation(ConnectionError):
    """Raised in place of the caller's original `connect()` when a
    non-allowlisted address is attempted while a `ReferenceNetGuard` is
    armed."""


class ReferenceNetGuard:
    """Patches `socket.socket.connect` process-wide while armed. Only one
    guard may be armed at a time (a second `arm()` while already armed
    raises `RuntimeError`) -- it patches a class-level method, so nesting
    would silently clobber the outer guard's restore point."""

    def __init__(self, allowed: Tuple[str, int], violations_path: Path) -> None:
        self._allowed = allowed
        self._violations_path = Path(violations_path)
        self._write_lock = threading.Lock()
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        if self._armed:
            raise RuntimeError("ReferenceNetGuard.arm() called while already armed")
        guard = self

        def _guarded_connect(sock: socket.socket, address):
            host = address[0] if isinstance(address, tuple) else address
            port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
            if (host, port) != guard._allowed:
                guard._record_violation(address)
                raise NetGuardViolation(
                    f"blocked non-loopback connect to {address!r}; only "
                    f"{guard._allowed!r} is permitted under the reference net guard"
                )
            return _real_socket_connect(sock, address)

        socket.socket.connect = _guarded_connect
        self._armed = True

    def disarm(self) -> None:
        if not self._armed:
            return
        socket.socket.connect = _real_socket_connect
        self._armed = False

    def __enter__(self) -> "ReferenceNetGuard":
        self.arm()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.disarm()

    def _record_violation(self, address: object) -> None:
        line = {
            "blocked_address": list(address) if isinstance(address, tuple) else address,
            "ts_monotonic": time.monotonic(),
        }
        with self._write_lock:
            self._violations_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._violations_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
