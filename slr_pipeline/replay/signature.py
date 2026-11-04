"""
slr_pipeline.replay.signature -- the replay-transport request signature
(Track 2 plan Sec 6.1) and the ONE place `rfc8785_jcs()` is defined for the
whole codebase.

Round-18 clarification (plan Sec 6.1): every other `rfc8785_jcs(...)`
citation in the plan (Sec 4.3, 10.1, 10.6, 10.7) imports this module's
`rfc8785_jcs` rather than redefining it -- e.g. `reference_bundle.py`'s
`_sha256_rfc8785_jcs()` (a later step) does
`from slr_pipeline.replay.signature import rfc8785_jcs`.

`rfc8785==0.1.4`'s real public API (confirmed against the installed wheel)
is `rfc8785.dumps(obj) -> bytes`, not a module-level `rfc8785_jcs` -- this
module is that thin, single wrapper, named to match every citation above.
"""
from __future__ import annotations

import hashlib
from typing import Any

import rfc8785

REPLAY_SIGNATURE_VERSION = "1"

# The one, fixed request path the replay transport ever signs against
# (plan Sec 6.1: "POST /v1/chat/completions only -- exact path").
REPLAY_ENDPOINT_PATH = "/v1/chat/completions"


def rfc8785_jcs(obj: Any) -> bytes:
    """RFC 8785 JSON Canonicalization Scheme serialisation of `obj`,
    returned as UTF-8 bytes. `test_jcs_conformance` (tests/test_replay_signature.py)
    runs the official RFC 8785 Appendix B/Sections 3.2.2-3.2.4/Appendix E
    vectors against this exact call."""
    return rfc8785.dumps(obj)


def compute_replay_signature(request_target_path: str, body: dict) -> str:
    """The Sec 6.1 signature formula, verbatim:
    `sha256(b"POST\\n" + request_target_path.encode() + b"\\n" + rfc8785_jcs(body))`.

    `body` must be the complete request JSON exactly as physically
    received -- every field, `messages` strings verbatim, no whitespace
    normalisation: `model`, `messages`, `response_format`/`json_schema`,
    `max_completion_tokens`, `reasoning_effort`, `temperature`, `tools`,
    `tool_choice`, `seed`, `n`. Callers pass the exact path they received
    (never lower-cased/normalised) -- the transport itself only ever
    receives `REPLAY_ENDPOINT_PATH`, but this function stays path-agnostic
    so `test_jcs_conformance`/unit tests can exercise it directly."""
    payload = b"POST\n" + request_target_path.encode("utf-8") + b"\n" + rfc8785_jcs(body)
    return hashlib.sha256(payload).hexdigest()
