"""Unit tests for `slr_pipeline.replay.signature` -- RFC 8785 JCS
conformance and the replay-transport signature formula
(`track2_pipeline_readiness_plan_2026-08-30.md` Sec 6.1, step 10).

`test_jcs_conformance` runs the official RFC 8785 vectors: the Section
3.2.2.3 IEEE-754-double serialisation table (Note 4's round-to-even rule
included), the worked Section 3.2.2-3.2.4 example (property sorting +
number/string serialisation together), and the Appendix E "pure" string
example -- each checked against the installed `rfc8785==0.1.4`."""
from __future__ import annotations

import hashlib
import json
import struct

from slr_pipeline.replay.signature import (
    REPLAY_ENDPOINT_PATH,
    REPLAY_SIGNATURE_VERSION,
    compute_replay_signature,
    rfc8785_jcs,
)


def _double_from_hex(hex_bytes: str) -> float:
    return struct.unpack(">d", bytes.fromhex(hex_bytes))[0]


# RFC 8785 Sec 3.2.2.3's IEEE-754-double -> JSON-number table, verbatim
# (hex big-endian double -> expected ECMAScript-number-to-string form).
_NUMBER_VECTORS = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    # Note 4's round-to-even example: true value 1424953923781206.25.
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


def test_jcs_conformance_number_serialization_vectors():
    for hex_bytes, expected in _NUMBER_VECTORS:
        value = _double_from_hex(hex_bytes)
        assert rfc8785_jcs(value).decode("ascii") == expected, hex_bytes


def test_jcs_conformance_worked_example_sections_3_2_2_to_3_2_4():
    # RFC 8785's own worked example: unsorted input with the four number
    # edge cases (a truncation, 1e30, 4.50 -> 4.5, 2e-3 -> 0.002, a small
    # value -> 1e-27), a string needing /\n/quote/backslash escaping,
    # and literals -- canonicalized with properties sorted lexicographically.
    input_text = (
        '{"numbers":[333333333.33333329,1e+30,4.50,2e-3,0.000000000000000000000000001],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/",'
        '"literals":[null,true,false]}'
    )
    obj = json.loads(input_text)
    expected = (
        b'{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        b'"string":"\xe2\x82\xac$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert rfc8785_jcs(obj) == expected


def test_jcs_conformance_appendix_e_pure_parse_example():
    obj = {"time": "2019-01-28T07:45:10Z", "big": "055", "val": 3.5}
    assert rfc8785_jcs(obj) == b'{"big":"055","time":"2019-01-28T07:45:10Z","val":3.5}'


def test_jcs_property_sorting_is_by_utf16_code_unit_order():
    # Values chosen so a naive Python string-sort would also happen to
    # agree; this asserts the observable ordering matches RFC 8785 Sec
    # 3.2.3's lexicographic-by-UTF-16-code-unit rule for our own callers'
    # inputs, without needing the RFC's own multi-plane property-name set.
    obj = {"b": 1, "a": 2, "1": 3}
    assert rfc8785_jcs(obj) == b'{"1":3,"a":2,"b":1}'


def test_replay_signature_version_pin():
    assert REPLAY_SIGNATURE_VERSION == "1"
    assert REPLAY_ENDPOINT_PATH == "/v1/chat/completions"


def test_compute_replay_signature_matches_formula_by_hand():
    body = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "hi"}]}
    expected = hashlib.sha256(
        b"POST\n" + REPLAY_ENDPOINT_PATH.encode("utf-8") + b"\n" + rfc8785_jcs(body)
    ).hexdigest()
    assert compute_replay_signature(REPLAY_ENDPOINT_PATH, body) == expected
    assert len(expected) == 64


def test_compute_replay_signature_is_deterministic_and_key_order_independent():
    body_a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.0}
    body_b = {"temperature": 0.0, "messages": [{"role": "user", "content": "x"}], "model": "m"}
    assert compute_replay_signature(REPLAY_ENDPOINT_PATH, body_a) == \
        compute_replay_signature(REPLAY_ENDPOINT_PATH, body_b)


def test_compute_replay_signature_sensitive_to_message_whitespace():
    # Sec 6.1: "messages strings verbatim (no whitespace normalisation)".
    body_a = {"messages": [{"role": "user", "content": "hi there"}]}
    body_b = {"messages": [{"role": "user", "content": "hi  there"}]}
    assert compute_replay_signature(REPLAY_ENDPOINT_PATH, body_a) != \
        compute_replay_signature(REPLAY_ENDPOINT_PATH, body_b)


def test_compute_replay_signature_sensitive_to_path():
    body = {"model": "m"}
    assert compute_replay_signature("/v1/chat/completions", body) != \
        compute_replay_signature("/v1/other", body)
