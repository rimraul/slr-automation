"""Track 2 plan §2.8 -- DatabaseSearchLog v3 (Revision 5 point 1): `_migrate_legacy`
(a legacy v1 doc is fully upgraded, never left as a hybrid shape; a v2 doc is
downgraded `"complete"` -> `"capped"`, never upgraded to `"exhaustive"`) and
`_conditional` (the full cross-field contract mirroring SearchBackendResult's own
invariants, plus retrieved<=requested and hits>=retrieved)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from slr_pipeline.state import DatabaseSearchLog


def _v3_kwargs(**overrides):
    base = dict(
        database="openalex",
        execution_adapter="openalex",
        adapter_config_version="1",
        query="cats",
        date="2026-09-01",
        retrieval_status="capped",
        termination_reason="max_results_cap_reached",
        failure_detail=None,
        unsupported_filter_deviations=[],
        skipped_malformed=0,
        requested=10,
        hits=3,
        retrieved=3,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# test_database_search_log_v1_load -- legacy doc fully upgraded, never hybrid
# ---------------------------------------------------------------------------

def test_database_search_log_v1_load_no_schema_version():
    v1 = {"database": "openalex", "query": "cats", "date": "2026-09-01", "hits": 5, "retrieved": 5}
    log = DatabaseSearchLog.model_validate(v1)
    assert log.schema_version == "database_search_log_v3"
    assert log.database == "openalex"
    assert log.execution_adapter == "openalex"     # migrated: same as database
    assert log.adapter_config_version == "unknown"
    # A v1 doc never recorded an exhaustion proof -- migrated to "capped", not "exhaustive".
    assert log.retrieval_status == "capped"
    assert log.termination_reason == "migrated_from_v2_unspecified"
    assert log.failure_detail is None
    assert log.unsupported_filter_deviations == []
    assert log.skipped_malformed == 0
    assert log.requested == 5                       # migrated from retrieved
    assert log.hits == 5
    assert log.retrieved == 5


def test_database_search_log_v1_load_explicit_v1_tag():
    v1 = {"schema_version": "v1", "database": "pubmed", "query": "dogs",
          "date": "2026-01-01", "hits": 2, "retrieved": 1}
    log = DatabaseSearchLog.model_validate(v1)
    assert log.schema_version == "database_search_log_v3"
    assert log.hits == 2
    assert log.retrieved == 1
    assert log.requested == 1                        # migrated from retrieved, NOT hits


def test_database_search_log_v1_load_missing_hits_defaults_to_retrieved():
    v1 = {"database": "arxiv", "query": "q", "date": "2026-01-01", "retrieved": 4}
    log = DatabaseSearchLog.model_validate(v1)
    assert log.hits == 4
    assert log.requested == 4


# ---------------------------------------------------------------------------
# `/code-review` (Track 2 step-20 round 2) found `_migrate_v1`'s guard only
# checked `set(data) <= allowed` -- a SUBSET check, which passes trivially
# even when required v1 fields are simply absent (e.g. `{}` is a subset of
# any set). A hand-edited/truncated legacy checkpoint missing "retrieved"
# used to blow up with a raw, unhandled KeyError on `data["retrieved"]`
# instead of the clean ValidationError this codebase guarantees everywhere
# else on malformed input.
# ---------------------------------------------------------------------------

def test_database_search_log_v1_load_missing_retrieved_raises_validation_error():
    v1 = {"database": "openalex", "query": "cats", "date": "2026-09-01", "hits": 5}
    with pytest.raises(ValidationError):
        DatabaseSearchLog.model_validate(v1)


def test_database_search_log_v1_load_missing_database_raises_validation_error():
    v1 = {"query": "cats", "date": "2026-09-01", "retrieved": 5}
    with pytest.raises(ValidationError):
        DatabaseSearchLog.model_validate(v1)


def test_database_search_log_v1_load_empty_dict_raises_validation_error():
    with pytest.raises(ValidationError):
        DatabaseSearchLog.model_validate({})


# ---------------------------------------------------------------------------
# v2 -> v3 migration (Revision 5 point 1): the already-persisted
# berg2023-update-20260923 artifact is v2 -- this is the exact upgrade path
# any code loading it goes through, in memory only, never rewriting the file.
# ---------------------------------------------------------------------------

def test_database_search_log_v2_doc_is_migrated_complete_downgrades_to_capped():
    v2 = {
        "schema_version": "database_search_log_v2", "database": "openalex",
        "execution_adapter": "openalex", "adapter_config_version": "1", "query": "cats",
        "date": "2026-09-01", "retrieval_status": "complete", "failure_detail": None,
        "unsupported_filter_deviations": [], "skipped_malformed": 0,
        "requested": 1000, "hits": 267400, "retrieved": 1000,
    }
    log = DatabaseSearchLog.model_validate(v2)
    assert log.schema_version == "database_search_log_v3"
    assert log.retrieval_status == "capped"
    assert log.termination_reason == "migrated_from_v2_unspecified"


def test_database_search_log_v2_doc_with_unexplained_gap_becomes_partial_not_capped():
    """Review finding P1 (2026-09-24): the preserved PubMed 300/303 v2 entry
    (requested=1000, far above both hits and retrieved) must not be blindly
    downgraded to a clean 'capped' -- its own counts prove an unexplained
    partial retrieval, which migration must surface, not hide."""
    v2 = {
        "schema_version": "database_search_log_v2", "database": "pubmed",
        "execution_adapter": "pubmed", "adapter_config_version": "1", "query": "cats",
        "date": "2026-09-01", "retrieval_status": "complete", "failure_detail": None,
        "unsupported_filter_deviations": [], "skipped_malformed": 0,
        "requested": 1000, "hits": 303, "retrieved": 300,
    }
    log = DatabaseSearchLog.model_validate(v2)
    assert log.retrieval_status == "partial"
    assert log.termination_reason == "unexplained_retrieval_gap"
    assert "303" in log.failure_detail
    assert "300" in log.failure_detail


def test_database_search_log_v3_capped_with_unexplained_gap_is_rejected():
    """Review finding P1 (2026-09-24): a directly constructed v3 doc must not
    be able to claim 'capped' when its own counts show an unexplained gap
    (hits > retrieved and retrieved < requested)."""
    with pytest.raises(ValidationError):
        DatabaseSearchLog.model_validate({
            "database": "pubmed", "execution_adapter": "pubmed", "adapter_config_version": "1",
            "query": "cats", "date": "2026-09-01", "retrieval_status": "capped",
            "termination_reason": "max_results_cap_reached", "failure_detail": None,
            "unsupported_filter_deviations": [], "skipped_malformed": 0,
            "requested": 1000, "hits": 303, "retrieved": 300,
        })


def test_database_search_log_v2_doc_preserves_partial_and_failed():
    v2 = {
        "schema_version": "database_search_log_v2", "database": "scopus",
        "execution_adapter": "scopus", "adapter_config_version": "1", "query": "cats",
        "date": "2026-09-01", "retrieval_status": "failed", "failure_detail": "boom",
        "unsupported_filter_deviations": [], "skipped_malformed": 0,
        "requested": 10, "hits": 0, "retrieved": 0,
    }
    log = DatabaseSearchLog.model_validate(v2)
    assert log.retrieval_status == "failed"
    assert log.termination_reason == "migrated_from_v2_unspecified"


def test_database_search_log_v3_doc_is_not_migrated():
    v3 = _v3_kwargs()
    log = DatabaseSearchLog.model_validate(v3)
    assert log.adapter_config_version == "1"          # NOT "unknown" -- migration did not run
    assert log.termination_reason == "max_results_cap_reached"


# ---------------------------------------------------------------------------
# test_database_search_log_conditional_rules -- the full cross-field contract
# ---------------------------------------------------------------------------

def test_database_search_log_conditional_rules_capped_is_valid():
    DatabaseSearchLog(**_v3_kwargs())  # does not raise


def test_database_search_log_conditional_rules_exhaustive_requires_hits_equal_retrieved():
    DatabaseSearchLog(**_v3_kwargs(
        retrieval_status="exhaustive", termination_reason="total_hits_equals_retrieved",
        requested=10, hits=3, retrieved=3,
    ))  # does not raise


def test_exhaustive_requires_hits_equal_retrieved():
    with pytest.raises(ValidationError, match="'exhaustive' requires hits == retrieved"):
        DatabaseSearchLog(**_v3_kwargs(
            retrieval_status="exhaustive", termination_reason="total_hits_equals_retrieved",
            requested=10, hits=5, retrieved=3,
        ))


def test_exhaustive_forbids_not_yet_instrumented_reason():
    with pytest.raises(ValidationError, match="not_yet_instrumented"):
        DatabaseSearchLog(**_v3_kwargs(
            retrieval_status="exhaustive", termination_reason="not_yet_instrumented",
            requested=10, hits=3, retrieved=3,
        ))


def test_conditional_skipped_forces_non_exhaustive_or_capped():
    with pytest.raises(ValidationError, match="skipped_malformed"):
        DatabaseSearchLog(**_v3_kwargs(skipped_malformed=1, retrieval_status="capped"))


def test_conditional_skipped_and_zero_retrieved_forces_failed():
    with pytest.raises(ValidationError, match="'failed'"):
        DatabaseSearchLog(**_v3_kwargs(
            skipped_malformed=1, retrieved=0, hits=0, retrieval_status="partial",
            termination_reason="partial_retrieval_with_skips_or_errors",
            failure_detail="skipped 1 malformed source record(s)",
        ))


def test_conditional_failed_must_have_zero_retrieved():
    with pytest.raises(ValidationError, match="failed database search retrieved 0"):
        DatabaseSearchLog(**_v3_kwargs(
            retrieval_status="failed", termination_reason="retrieval_error",
            retrieved=1, hits=1, failure_detail="boom",
        ))


def test_conditional_partial_requires_at_least_one_retrieved():
    with pytest.raises(ValidationError, match="partial requires"):
        DatabaseSearchLog(**_v3_kwargs(
            retrieval_status="partial", termination_reason="partial_retrieval_with_skips_or_errors",
            retrieved=0, hits=0, failure_detail="boom",
        ))


def test_conditional_capped_forbids_failure_detail():
    with pytest.raises(ValidationError, match="exhaustive/capped search has no failure_detail"):
        DatabaseSearchLog(**_v3_kwargs(retrieval_status="capped", failure_detail="oops"))


def test_conditional_partial_or_failed_require_failure_detail():
    with pytest.raises(ValidationError, match="MUST carry a non-empty failure_detail"):
        DatabaseSearchLog(**_v3_kwargs(
            retrieval_status="partial", termination_reason="partial_retrieval_with_skips_or_errors",
            retrieved=1, hits=1, failure_detail=None,
        ))


def test_conditional_retrieved_cannot_exceed_requested():
    with pytest.raises(ValidationError, match="retrieved cannot exceed requested"):
        DatabaseSearchLog(**_v3_kwargs(requested=1, retrieved=2, hits=2))


def test_conditional_hits_cannot_be_less_than_retrieved():
    with pytest.raises(ValidationError, match="hits cannot be < retrieved"):
        DatabaseSearchLog(**_v3_kwargs(requested=10, retrieved=5, hits=2))


def test_conditional_failed_requires_zero_retrieved_and_valid_failure_detail():
    DatabaseSearchLog(**_v3_kwargs(
        retrieval_status="failed", termination_reason="retrieval_error",
        retrieved=0, hits=0, requested=10,
        failure_detail="SPRINGER_API_KEY not set",
    ))  # does not raise
