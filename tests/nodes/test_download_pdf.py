"""slr_pipeline.nodes.fulltext._download_pdf -- unlike every other test in
tests/nodes/test_fulltext_node.py, this file does NOT monkeypatch
`_download_pdf` away. Its own docstring says every fixture record has
doi=None so `_resolve_oa_url` short-circuits before `_download_pdf` is ever
called -- meaning, before this file, `_download_pdf` had zero test
coverage at all. `/code-review` (Track 2 step-20 round 3) found this and
two real connection-leak/uncaught-exception bugs inside it as a direct
consequence: nothing exercised its real body."""
from __future__ import annotations

import time

import pytest
import requests

from slr_pipeline.nodes import fulltext as fulltext_module

_download_pdf = fulltext_module._download_pdf


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    # _download_pdf has real polite-throttle + exponential-backoff sleeps;
    # none of that is under test here.
    monkeypatch.setattr(time, "sleep", lambda *a, **kw: None)
    monkeypatch.setattr(fulltext_module.random, "uniform", lambda a, b: 0.0)


class _FakeResponse:
    def __init__(self, status_code=200, content_type="application/pdf",
                 chunks=(b"%PDF-1.4 fake pdf body that is over one kilobyte " + b"x" * 1100,)):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self._chunks = list(chunks)
        self.closed = False

    def iter_content(self, chunk_size=8192):
        for c in self._chunks:
            yield c

    def close(self):
        self.closed = True


def test_download_pdf_closes_response_on_5xx_before_retry(monkeypatch, tmp_path):
    responses = [_FakeResponse(status_code=500), _FakeResponse(status_code=200)]
    monkeypatch.setattr(requests, "get", lambda *a, **kw: responses.pop(0))

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=2, base_wait=0.0)

    assert ok is True
    assert responses == []  # both fake responses were consumed


def test_download_pdf_closes_response_on_final_non_200_status(monkeypatch, tmp_path):
    resp = _FakeResponse(status_code=404)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True


def test_download_pdf_closes_response_on_html_content_type(monkeypatch, tmp_path):
    resp = _FakeResponse(content_type="text/html")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True


def test_download_pdf_closes_response_on_empty_body(monkeypatch, tmp_path):
    resp = _FakeResponse(chunks=())
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True


def test_download_pdf_closes_response_on_success(monkeypatch, tmp_path):
    resp = _FakeResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is True
    assert resp.closed is True
    assert dest.read_bytes().startswith(b"%PDF")


def test_download_pdf_mid_stream_error_is_treated_as_failure_not_crash(monkeypatch, tmp_path):
    class _BrokenResponse(_FakeResponse):
        def iter_content(self, chunk_size=8192):
            yield b"%PDF-1.4 partial"
            raise requests.exceptions.ChunkedEncodingError("connection dropped mid-body")

    resp = _BrokenResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    dest = tmp_path / "out.pdf"
    # Must degrade to a normal "not retrievable" result, exactly like every
    # other failure path in this function -- NOT propagate the network
    # exception and abort whatever loop over records is calling this.
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True
    assert not dest.exists()


def test_download_pdf_error_on_very_first_chunk_is_treated_as_failure_not_crash(monkeypatch, tmp_path):
    # /code-review round 4: round 3's fix only caught RequestException on
    # the SECOND-or-later next(content_iter) call (inside the write loop),
    # not the FIRST one -- a server that resets the connection right after
    # headers, before any body byte, raises on that first next() instead.
    class _BrokenFromStartResponse(_FakeResponse):
        def iter_content(self, chunk_size=8192):
            raise requests.exceptions.ChunkedEncodingError("connection dropped before any body")
            yield b""  # pragma: no cover -- unreachable, keeps this a generator

    resp = _BrokenFromStartResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True
    assert not dest.exists()


def test_download_pdf_oserror_writing_file_is_treated_as_failure_not_crash(monkeypatch, tmp_path):
    # A disk-full/permission-denied/invalid-path OSError while writing must
    # degrade the same way a network error does -- not crash the batch.
    resp = _FakeResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))

    dest = tmp_path / "out.pdf"
    ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True


def test_download_pdf_permission_error_on_cleanup_unlink_does_not_crash(monkeypatch, tmp_path, caplog):
    # A stream error triggers the cleanup dest.unlink() -- if THAT itself
    # raises (e.g. a transient AV/indexer lock on Windows), the intended
    # graceful-failure return must still win, not a fresh uncaught crash.
    #
    # /code-review round 5: round 4's `except OSError: pass` here was itself
    # a new bug -- it silently left the stray/truncated file on disk with
    # no diagnostic trail, and `_find_local_pdf` (existence-only check)
    # would pick it up forever on every subsequent run. The fix logs a
    # warning instead of swallowing silently; assert BOTH that stray file
    # is left in place (the unlink genuinely did fail, that's real) AND
    # that a warning was logged pointing at it.
    class _BrokenResponse(_FakeResponse):
        def iter_content(self, chunk_size=8192):
            yield b"%PDF-1.4 partial"
            raise requests.exceptions.ChunkedEncodingError("connection dropped mid-body")

    resp = _BrokenResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    real_unlink = type(tmp_path).unlink

    def _flaky_unlink(self, *a, **kw):
        raise PermissionError("[WinError 32] file in use by another process")

    monkeypatch.setattr(type(tmp_path), "unlink", _flaky_unlink)
    try:
        dest = tmp_path / "out.pdf"
        with caplog.at_level("WARNING", logger=fulltext_module.logger.name):
            ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)
    finally:
        monkeypatch.setattr(type(tmp_path), "unlink", real_unlink)

    assert ok is False
    assert resp.closed is True
    assert dest.exists()   # unlink genuinely failed -- the stray file IS still there
    assert any("stray" in rec.message.lower() for rec in caplog.records)


def test_download_pdf_stat_oserror_on_size_check_does_not_crash(monkeypatch, tmp_path, caplog):
    # /code-review round 5: the too-small-file size check's dest.stat()
    # call had no failure handling at all before round 4's fix; the fix
    # added `except OSError: return False` with no log line. Assert it now
    # degrades gracefully AND logs, same standard as the sibling branch.
    resp = _FakeResponse()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    real_stat = type(tmp_path).stat

    def _flaky_stat(self, *a, **kw):
        if self.name == "out.pdf":
            raise OSError("transient stat failure")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(type(tmp_path), "stat", _flaky_stat)

    dest = tmp_path / "out.pdf"
    with caplog.at_level("WARNING", logger=fulltext_module.logger.name):
        ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)

    assert ok is False
    assert resp.closed is True
    assert any("verify size" in rec.message.lower() for rec in caplog.records)


def test_download_pdf_permission_error_on_undersized_cleanup_unlink_does_not_crash(monkeypatch, tmp_path, caplog):
    # Same "log instead of swallow" standard for the OTHER cleanup unlink
    # (the too-small-file branch), which round 4 also left as bare `pass`.
    resp = _FakeResponse(chunks=(b"%PDF-1.4 tiny",))   # under the 1024-byte floor
    monkeypatch.setattr(requests, "get", lambda *a, **kw: resp)

    real_unlink = type(tmp_path).unlink

    def _flaky_unlink(self, *a, **kw):
        raise PermissionError("[WinError 32] file in use by another process")

    monkeypatch.setattr(type(tmp_path), "unlink", _flaky_unlink)
    try:
        dest = tmp_path / "out.pdf"
        with caplog.at_level("WARNING", logger=fulltext_module.logger.name):
            ok = _download_pdf("https://example.test/x.pdf", dest, max_retries=1, base_wait=0.0)
    finally:
        monkeypatch.setattr(type(tmp_path), "unlink", real_unlink)

    assert ok is False
    assert resp.closed is True
    assert dest.exists()
    assert any("undersized" in rec.message.lower() for rec in caplog.records)
