"""Unit tests for `slr_pipeline.reference_netguard` -- the socket allowlist
Track 2's reference-run child arms before any pipeline/third-party import
(`track2_pipeline_readiness_plan_2026-08-30.md` Sec 12.3/11.2, step 10)."""
from __future__ import annotations

import json
import socket
import threading

import pytest

from slr_pipeline.reference_netguard import NetGuardViolation, ReferenceNetGuard


@pytest.fixture()
def loopback_listener():
    """A real, bound loopback listener the guard is allowed to reach."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def _accept_forever():
        try:
            while True:
                conn, _ = srv.accept()
                conn.close()
        except OSError:
            pass  # socket closed at teardown

    thread = threading.Thread(target=_accept_forever, daemon=True)
    thread.start()
    yield port
    srv.close()


def test_allowed_connect_succeeds_while_armed(loopback_listener, tmp_path):
    guard = ReferenceNetGuard(("127.0.0.1", loopback_listener), tmp_path / "violations.jsonl")
    with guard:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("127.0.0.1", loopback_listener))
        finally:
            s.close()
    assert not (tmp_path / "violations.jsonl").exists()


def test_disallowed_connect_blocked_and_logged(tmp_path):
    guard = ReferenceNetGuard(("127.0.0.1", 65535), tmp_path / "violations.jsonl")
    with guard:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetGuardViolation):
                s.connect(("127.0.0.1", 1))  # not the allowed port
        finally:
            s.close()

    lines = (tmp_path / "violations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["blocked_address"] == ["127.0.0.1", 1]
    assert "ts_monotonic" in entry


def test_disarm_restores_real_connect(tmp_path):
    original = socket.socket.connect
    guard = ReferenceNetGuard(("127.0.0.1", 1), tmp_path / "violations.jsonl")
    guard.arm()
    assert socket.socket.connect is not original
    guard.disarm()
    assert socket.socket.connect is original


def test_double_arm_raises(tmp_path):
    guard = ReferenceNetGuard(("127.0.0.1", 1), tmp_path / "violations.jsonl")
    guard.arm()
    try:
        with pytest.raises(RuntimeError, match="already armed"):
            guard.arm()
    finally:
        guard.disarm()


def test_disarm_when_not_armed_is_a_noop(tmp_path):
    guard = ReferenceNetGuard(("127.0.0.1", 1), tmp_path / "violations.jsonl")
    guard.disarm()  # must not raise
    assert guard.armed is False


def test_multiple_violations_append_multiple_lines(tmp_path):
    guard = ReferenceNetGuard(("127.0.0.1", 65535), tmp_path / "violations.jsonl")
    with guard:
        for port in (1, 2, 3):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with pytest.raises(NetGuardViolation):
                    s.connect(("127.0.0.1", port))
            finally:
                s.close()
    lines = (tmp_path / "violations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
