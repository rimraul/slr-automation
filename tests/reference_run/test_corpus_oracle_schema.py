"""Track 2 Step 12A -- `slr_pipeline.corpus_oracle.CorpusOracle`'s schema
validation, in particular the revision proposal's explicit acceptance-
harness assertion: a `CORPUS_ORACLE.json` with a missing or wrong
`oracle_kind` MUST fail loudly, never be silently accepted.

Builds its base payload from the real
`reference_fixture_answer_key.corpus_oracle_payload()` (never a hand-typed
second copy of the same facts) plus a placeholder `pdf_sha256` mapping,
since these tests do not depend on the synthetic PDFs actually existing on
disk.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from slr_pipeline import reference_fixture_answer_key as ak
from slr_pipeline.corpus_oracle import CORPUS_ORACLE_VERSION, ORACLE_KIND, CorpusOracle


def _valid_payload() -> dict:
    placeholder_pdf_sha256 = {rid: "0" * 64 for rid in ak.FULL_TEXT_SOUGHT}
    return ak.corpus_oracle_payload(placeholder_pdf_sha256)


def test_valid_payload_from_the_answer_key_round_trips():
    oracle = CorpusOracle.model_validate(_valid_payload())
    assert oracle.oracle_kind == ORACLE_KIND
    assert oracle.corpus_oracle_version == CORPUS_ORACLE_VERSION
    assert oracle.raw_records == 30
    assert oracle.unique_after_dedup == 22
    assert oracle.total_replay_calls == 50


def test_missing_oracle_kind_is_rejected():
    payload = _valid_payload()
    del payload["oracle_kind"]
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


def test_wrong_oracle_kind_value_is_rejected():
    payload = _valid_payload()
    payload["oracle_kind"] = "scientific_finding"
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


def test_wrong_corpus_oracle_version_is_rejected():
    payload = _valid_payload()
    payload["corpus_oracle_version"] = "1"
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


def test_extra_field_is_rejected():
    payload = _valid_payload()
    payload["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


@pytest.mark.parametrize("field, mutate", [
    ("unique_after_dedup", lambda v: v + 1),
    ("duplicates_removed", lambda v: v + 1),
    ("total_replay_calls", lambda v: v + 1),
])
def test_internal_consistency_checks_reject_a_broken_count(field, mutate):
    payload = _valid_payload()
    payload[field] = mutate(payload[field])
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


def test_full_text_exclusions_and_eligible_must_partition_inclusions():
    payload = _valid_payload()
    payload["full_text_eligible"] = list(payload["full_text_eligible"]) + list(payload["full_text_exclusions"])
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


def test_pdf_sha256_keys_must_match_title_abstract_inclusions():
    payload = _valid_payload()
    payload["pdf_sha256"] = {k: v for k, v in payload["pdf_sha256"].items() if k != ak.FULL_TEXT_SOUGHT[0]}
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)


def test_map_relevant_by_rq_cannot_reference_an_ineligible_record():
    payload = _valid_payload()
    payload["map_relevant_by_rq"] = dict(payload["map_relevant_by_rq"])
    payload["map_relevant_by_rq"]["rq_0"] = list(payload["map_relevant_by_rq"]["rq_0"]) + [
        next(iter(payload["full_text_exclusions"]))
    ]
    with pytest.raises(ValidationError):
        CorpusOracle.model_validate(payload)
