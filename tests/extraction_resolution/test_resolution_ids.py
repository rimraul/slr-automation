from __future__ import annotations

import re

from slr_pipeline.extraction.resolution_ids import candidate_id, evidence_id, safe_key

_HEX24 = re.compile(r"^[0-9a-f]{24}$")


def test_evidence_id_is_24_hex_chars_and_deterministic():
    a = evidence_id("doc-1", "field-1", "chunk-0019", "excerpt text", "4", "Results")
    b = evidence_id("doc-1", "field-1", "chunk-0019", "excerpt text", "4", "Results")
    assert a == b
    assert _HEX24.match(a)


def test_evidence_id_distinguishes_none_page_from_string_none():
    a = evidence_id("doc-1", "field-1", "chunk-1", "excerpt", None, "Results")
    b = evidence_id("doc-1", "field-1", "chunk-1", "excerpt", "None", "Results")
    assert a != b


def test_evidence_id_same_chunk_different_excerpt_differs():
    a = evidence_id("doc-1", "field-1", "chunk-0019", "excerpt one", "4", "Results")
    b = evidence_id("doc-1", "field-1", "chunk-0019", "excerpt two", "4", "Results")
    assert a != b


def test_candidate_id_is_24_hex_chars_and_deterministic():
    eids = (evidence_id("d", "f", "c", "e", "1", "s"),)
    a = candidate_id("12", 12, "value", eids)
    b = candidate_id("12", 12, "value", eids)
    assert a == b
    assert _HEX24.match(a)


def test_candidate_id_order_independent_evidence_ids():
    e1 = evidence_id("d", "f", "c1", "e1", "1", "s")
    e2 = evidence_id("d", "f", "c2", "e2", "2", "s")
    a = candidate_id("v", None, "value", (e1, e2))
    b = candidate_id("v", None, "value", (e2, e1))
    assert a == b


def test_candidate_id_distinguishes_source_category():
    eids = (evidence_id("d", "f", "c", "e", "1", "s"),)
    value_id = candidate_id("v", None, "value", eids)
    invalid_id = candidate_id("v", None, "invalid_value", eids)
    assert value_id != invalid_id


def test_safe_key_is_24_hex_chars_and_deterministic():
    a = safe_key("run-1", "record with spaces/../weird", "field-1")
    b = safe_key("run-1", "record with spaces/../weird", "field-1")
    assert a == b
    assert _HEX24.match(a)


def test_safe_key_distinguishes_run_record_field():
    base = safe_key("run-1", "record-1", "field-1")
    assert base != safe_key("run-2", "record-1", "field-1")
    assert base != safe_key("run-1", "record-2", "field-1")
    assert base != safe_key("run-1", "record-1", "field-2")
