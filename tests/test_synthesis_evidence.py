"""Tests for slr_pipeline/synthesis_evidence.py -- the shared evidence-index
construction module for the Stage 3 claim-construction slice of
quality_appraisal_synthesis_plan_2026-08-22.md's own §3.2. Its own
`make_evidence_id()` deliberately supersedes that parent document's own
narrower, normalized formula -- see the supersession note in that document's
own §3.2 for why."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slr_pipeline.synthesis_evidence import (
    EvidenceIndexEntry, build_evidence_index_for_field, make_evidence_id, _human_attested_chunk_id,
)


def _raw(**overrides):
    base = dict(chunk_id="c1", quotation="A B", page="1", section="Methods", locator=None)
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# make_evidence_id() -- raw hashing, no normalization.
# ---------------------------------------------------------------------------

def test_evidence_id_hashes_raw_values_not_normalized():
    """Two quotations differing only in whitespace ("A  B" vs "A B") normalize identically but must
    NOT collide on evidence_id -- the hash is over the RAW stored fields."""
    id1 = make_evidence_id(artifact_kind="extraction", artifact_run_id="r1", record_id="rec1",
                            field_id="f1", chunk_id="c1", quotation="A  B", page="1", section=None, locator=None)
    id2 = make_evidence_id(artifact_kind="extraction", artifact_run_id="r1", record_id="rec1",
                            field_id="f1", chunk_id="c1", quotation="A B", page="1", section=None, locator=None)
    assert id1 != id2


def test_evidence_id_byte_identical_inputs_collide_and_dedupe():
    entries = build_evidence_index_for_field(
        [_raw(), _raw()], artifact_kind="extraction", artifact_run_id="r1", record_id="rec1", field_id="f1")
    assert len(entries) == 1


def test_evidence_id_rejects_invalid_artifact_kind():
    with pytest.raises(ValueError, match="artifact_kind"):
        make_evidence_id(artifact_kind="bogus", artifact_run_id="r1", record_id="rec1",
                          field_id="f1", chunk_id="c1", quotation="q", page=None, section=None, locator=None)


# ---------------------------------------------------------------------------
# _build_evidence_index_entry() / build_evidence_index_for_field() -- page/
# section/locator are REQUIRED keyword arguments, no default.
# ---------------------------------------------------------------------------

def test_build_evidence_index_for_field_requires_page_section_locator_explicitly():
    with pytest.raises(TypeError):
        build_evidence_index_for_field(
            [{"chunk_id": "c1", "quotation": "q"}],  # page/section/locator omitted entirely
            artifact_kind="extraction", artifact_run_id="r1", record_id="rec1", field_id="f1")


def test_build_evidence_index_for_field_accepts_none_for_inapplicable_fields():
    entries = build_evidence_index_for_field(
        [_raw(page=None, section=None, locator="p.3")],
        artifact_kind="quality_appraisal", artifact_run_id="qa1", record_id="rec1", field_id="f1")
    assert len(entries) == 1
    assert entries[0].page is None and entries[0].locator == "p.3"


def test_evidence_id_format_validated():
    with pytest.raises(ValueError, match="not well-formed"):
        EvidenceIndexEntry(evidence_id="evidence-not-hex", chunk_id="c1", quotation="q")


# ---------------------------------------------------------------------------
# _human_attested_chunk_id() -- deterministic, locator-derived.
# ---------------------------------------------------------------------------

def test_human_attested_chunk_id_deterministic_and_locator_sensitive():
    id1 = _human_attested_chunk_id("page 3")
    id2 = _human_attested_chunk_id("page 3")
    id3 = _human_attested_chunk_id("page 4")
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("human-attested:")
