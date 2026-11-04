"""Offline tests for slr_pipeline.nodes.search -- Track 2 plan §2.5/§2.8
step 3 (search_node dispatch rewrite onto slr_pipeline/search_backends/*).

No real network call is ever made here: every adapter dispatched is a fake
registered into a monkeypatched `_ADAPTER_REGISTRY`, so `HttpClient` never
runs. `test_unknown_database_is_fatal` covers plan §2.5's both dispatch
paths; the rest cover the request<->result binding assertions (§2.4,
round-12 finding 2) and DatabaseSearchLog v2 population (§2.8)."""
from __future__ import annotations

import json

import pytest

import slr_pipeline.nodes.search as search_module
from slr_pipeline.config import ReviewConfig, SearchConfig, scoping_review_config
from slr_pipeline.search_backends.base import SearchBackendRequest, SearchBackendResult, SearchConfigError
from slr_pipeline.state import Record, initial_state


def _state_with_databases(databases, execution_adapter_map=None, **search_kwargs):
    cfg = scoping_review_config(title="t")
    cfg.search = SearchConfig(
        databases=databases,
        query="cats",
        execution_adapter_map=execution_adapter_map,
        **search_kwargs,
    )
    return initial_state(cfg.to_dict())


def _make_fake_adapter(*, adapter_name, records=None, retrieval_status="capped",
                        termination_reason="not_yet_instrumented",
                        failure_detail=None, skipped_malformed=0, total_hits=None):
    records = records or []

    class _FakeAdapter:
        def search(self, request: SearchBackendRequest) -> SearchBackendResult:
            return SearchBackendResult(
                adapter=adapter_name,
                records=records,
                retrieval_status=retrieval_status,
                termination_reason=termination_reason,
                failure_detail=failure_detail,
                skipped_malformed=skipped_malformed,
                requested=request.max_results,
                returned=len(records),
                total_hits=total_hits,
            )

    return _FakeAdapter


def _fake_record(source: str, key: str) -> Record:
    return Record(source=source, source_id=key, title=f"title for {key}")


# ---------------------------------------------------------------------------
# test_unknown_database_is_fatal -- both dispatch paths (plan §2.5)
# ---------------------------------------------------------------------------

def test_unknown_database_is_fatal_unmapped_dispatch():
    """Unmapped dispatch (execution_adapter_map is None): an unrecognised
    db raises SearchConfigError -- never the old silent drop."""
    state = _state_with_databases(["openalex", "not_a_real_db"])
    with pytest.raises(SearchConfigError, match="not_a_real_db"):
        search_module.search_node(state)


def test_unknown_database_is_fatal_mapped_dispatch_key_mismatch():
    """Mapped dispatch: execution_adapter_map's keys must equal
    set(cfg.search.databases) EXACTLY."""
    state = _state_with_databases(
        ["openalex", "pubmed"],
        execution_adapter_map={"openalex": "openalex"},  # missing 'pubmed'
    )
    with pytest.raises(SearchConfigError, match="execution_adapter_map"):
        search_module.search_node(state)


def test_unknown_database_is_fatal_mapped_dispatch_unknown_value():
    """Mapped dispatch: every value must be a real _ADAPTER_REGISTRY key.
    'fx_openalex' is fx_*-shaped but not one of the seven canonical Berg
    database names step 10's fixture adapters register (see
    tests/adapters/test_fixture_search_adapter.py) -- so it stays fatal
    even after step 10 lands."""
    state = _state_with_databases(
        ["openalex"],
        execution_adapter_map={"openalex": "fx_openalex"},
    )
    with pytest.raises(SearchConfigError, match="fx_openalex"):
        search_module.search_node(state)


def test_unknown_database_fatal_before_any_dispatch(monkeypatch):
    """Neither dispatch path may perform any network call / adapter
    invocation before validation completes -- a registered adapter for the
    KNOWN db in the same request must never be invoked either."""
    called = []

    class _SpyAdapter:
        def search(self, request):
            called.append(True)
            raise AssertionError("adapter must not be dispatched")

    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"openalex": _SpyAdapter})
    state = _state_with_databases(["openalex", "not_a_real_db"])
    with pytest.raises(SearchConfigError):
        search_module.search_node(state)
    assert called == []


# ---------------------------------------------------------------------------
# Request<->result binding assertions (plan §2.4, round-12 finding 2)
# ---------------------------------------------------------------------------

def test_result_adapter_mismatch_is_fatal(monkeypatch):
    fake = _make_fake_adapter(adapter_name="wrong_adapter_name")
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"openalex": fake})
    state = _state_with_databases(["openalex"])
    with pytest.raises(SearchConfigError, match="wrong_adapter_name"):
        search_module.search_node(state)


# ---------------------------------------------------------------------------
# DatabaseSearchLog v2 population (plan §2.8) + PRISMA/raw_records assembly
# ---------------------------------------------------------------------------

def test_happy_path_dispatch_populates_log_and_records(monkeypatch):
    openalex_records = [_fake_record("openalex", "W1"), _fake_record("openalex", "W2")]
    pubmed_records = [_fake_record("pubmed", "111")]

    monkeypatch.setattr(
        search_module,
        "_ADAPTER_REGISTRY",
        {
            "openalex": _make_fake_adapter(
                adapter_name="openalex", records=openalex_records, total_hits=57
            ),
            "pubmed": _make_fake_adapter(
                adapter_name="pubmed", records=pubmed_records, total_hits=None
            ),
        },
    )
    # max_results_per_database=2 makes openalex's "capped" result genuinely
    # valid (retrieved == requested == the configured cap reached) -- review
    # finding P1 (2026-09-24): 'capped' with retrieved < requested and a known
    # larger total_hits is an unexplained gap, not a clean cap.
    state = _state_with_databases(["openalex", "pubmed"], max_results_per_database=2)
    result = search_module.search_node(state)

    assert len(result["raw_records"]) == 3
    assert result["prisma_counts"]["identified"] == {"openalex": 2, "pubmed": 1}

    entries = {e["database"]: e for e in result["search_log"]["entries"]}
    assert entries["openalex"]["execution_adapter"] == "openalex"
    assert entries["openalex"]["retrieval_status"] == "capped"
    assert entries["openalex"]["hits"] == 57          # total_hits known -> hits == total_hits
    assert entries["openalex"]["retrieved"] == 2
    assert entries["pubmed"]["hits"] == 1              # total_hits None -> hits == retrieved
    assert entries["pubmed"]["retrieved"] == 1


def test_mapped_dispatch_records_execution_adapter_distinct_from_database(monkeypatch):
    """When execution_adapter_map is set, DatabaseSearchLog.database is the
    CANONICAL name and .execution_adapter is the mapped registry key --
    verified distinct here even though both point at the same fake class."""
    fake = _make_fake_adapter(adapter_name="pubmed", records=[_fake_record("pubmed", "1")])
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"openalex": fake, "pubmed": fake})
    state = _state_with_databases(["openalex"], execution_adapter_map={"openalex": "pubmed"})
    result = search_module.search_node(state)
    entry = result["search_log"]["entries"][0]
    assert entry["database"] == "openalex"
    assert entry["execution_adapter"] == "pubmed"


def test_mapped_dispatch_reaches_real_fixture_adapter(tmp_path, monkeypatch):
    """End-to-end wiring check (step 10): a genuine `fx_pubmed` mapped
    dispatch -- the REAL `_ADAPTER_REGISTRY` merge, not a monkeypatched
    one -- reaches `replay.fixture_search_adapter`'s registered class and
    returns its corpus records. Per-record contract details (authored id
    preservation, FixtureCorpusError, etc.) are covered in
    tests/adapters/test_fixture_search_adapter.py; this test only proves
    nodes/search.py actually dispatches to it."""
    corpus_root = tmp_path / "corpus"
    (corpus_root / "native").mkdir(parents=True)
    (corpus_root / "native" / "fx_pubmed.json").write_text(
        json.dumps([{"id": "fx-0001", "native_key": "pm-t2-0001", "title": "A rural study",
                     "doi": None, "year": 2020}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLR_FIXTURE_CORPUS_ROOT", str(corpus_root))

    state = _state_with_databases(["pubmed"], execution_adapter_map={"pubmed": "fx_pubmed"})
    result = search_module.search_node(state)

    entry = result["search_log"]["entries"][0]
    assert entry["database"] == "pubmed"
    assert entry["execution_adapter"] == "fx_pubmed"
    assert entry["retrieval_status"] == "exhaustive"
    assert len(result["raw_records"]) == 1
    assert result["raw_records"][0]["id"] == "fx-0001"
    assert result["raw_records"][0]["source"] == "fx_pubmed"
    assert result["raw_records"][0]["source_id"] == "pm-t2-0001"


def test_unsupported_filter_deviations_recorded(monkeypatch):
    """openalex is LANGUAGE_UNSUPPORTED, arxiv is DATE_UNSUPPORTED (plan
    §2.6's language/date matrix columns, base.py's frozen tables) --
    search_node records the deviation, keyed on canonical db name."""
    monkeypatch.setattr(
        search_module,
        "_ADAPTER_REGISTRY",
        {
            "openalex": _make_fake_adapter(adapter_name="openalex"),
            "arxiv": _make_fake_adapter(adapter_name="arxiv"),
        },
    )
    state = _state_with_databases(
        ["openalex", "arxiv"], language="fr", date_from="2020", date_to="2021",
    )
    result = search_module.search_node(state)
    entries = {e["database"]: e for e in result["search_log"]["entries"]}
    assert any("language" in d for d in entries["openalex"]["unsupported_filter_deviations"])
    assert any("date" in d for d in entries["arxiv"]["unsupported_filter_deviations"])


def test_partial_retrieval_status_and_skipped_malformed_propagate(monkeypatch):
    fake = _make_fake_adapter(
        adapter_name="openalex",
        records=[_fake_record("openalex", "W1")],
        retrieval_status="partial",
        termination_reason="partial_retrieval_with_skips_or_errors",
        failure_detail="skipped 2 malformed source record(s)",
        skipped_malformed=2,
    )
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"openalex": fake})
    state = _state_with_databases(["openalex"])
    result = search_module.search_node(state)
    entry = result["search_log"]["entries"][0]
    assert entry["retrieval_status"] == "partial"
    assert entry["skipped_malformed"] == 2
    assert entry["failure_detail"]


# ---------------------------------------------------------------------------
# _sanitize_query -- unchanged legacy behaviour, still exercised through
# the real dispatch path (PubMed field-tag syntax vs. the OpenAlex/S2 style)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Track 2 plan §8: determinism fixes -- as_of_date override, configured-order
# aggregation independent of as_completed()'s arrival order.
# ---------------------------------------------------------------------------

def test_as_of_date_none_falls_back_to_today(monkeypatch):
    class _FixedDate:
        @staticmethod
        def today():
            import datetime as _dt
            return _dt.date(2024, 1, 1)

    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"openalex": _make_fake_adapter(adapter_name="openalex")})
    monkeypatch.setattr(search_module, "date", _FixedDate)
    state = _state_with_databases(["openalex"])
    result = search_module.search_node(state)
    assert result["search_log"]["entries"][0]["date"] == "2024-01-01"


def test_as_of_date_explicit_overrides_today(monkeypatch):
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"openalex": _make_fake_adapter(adapter_name="openalex")})
    state = _state_with_databases(["openalex"], as_of_date="2023-06-15")
    result = search_module.search_node(state)
    assert result["search_log"]["entries"][0]["date"] == "2023-06-15"


def test_configured_order_aggregation_independent_of_completion_order(monkeypatch):
    """The db that finishes LAST (an artificial delay) must still appear
    FIRST in raw_records/search_log if it's first in cfg.search.databases --
    as_completed()'s non-deterministic arrival order must never leak into
    output ordering (Track 2 plan §8)."""
    import time

    class _SlowAdapter:
        def search(self, request):
            time.sleep(0.05)
            return SearchBackendResult(
                adapter="slow", records=[_fake_record("slow", "S1")],
                retrieval_status="capped", termination_reason="not_yet_instrumented",
                failure_detail=None,
                skipped_malformed=0, requested=request.max_results, returned=1,
            )

    fast = _make_fake_adapter(adapter_name="fast", records=[_fake_record("fast", "F1")])
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"slow": _SlowAdapter, "fast": fast})
    state = _state_with_databases(["slow", "fast"])
    result = search_module.search_node(state)

    assert [e["database"] for e in result["search_log"]["entries"]] == ["slow", "fast"]
    assert [r["source"] for r in result["raw_records"]] == ["slow", "fast"]


def test_sanitize_query_passes_pubmed_syntax_through_unchanged():
    query = '"disaster"[Title] AND "healthcare"[MeSH]'
    assert search_module._sanitize_query(query, "pubmed") == query


def test_sanitize_query_simplifies_for_non_pubmed_db():
    query = '"disaster"[Title] AND "healthcare"[MeSH]'
    simplified = search_module._sanitize_query(query, "openalex")
    assert simplified == '"disaster" OR "healthcare"'


# ---------------------------------------------------------------------------
# _STRICT_TRANSLATION_DBS (scopus): a missing/failed per-database
# translation must be a pre-network SearchConfigError, never the raw
# PubMed-syntax query or `_sanitize_query()`'s lossy regex rebuild.
# Confirmed root cause of a real live run producing an invalid 0-hit
# Scopus result -- see slr_pipeline/nodes/search.py::_resolve_query_for_db.
# ---------------------------------------------------------------------------

def test_scopus_missing_translation_is_pre_network_failure(monkeypatch):
    called = []

    class _SpyAdapter:
        def search(self, request):
            called.append(True)
            raise AssertionError("adapter must not be dispatched")

    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"pubmed": _SpyAdapter, "scopus": _SpyAdapter})
    state = _state_with_databases(["pubmed", "scopus"])   # no search_strategy -> db_queries == {}
    with pytest.raises(SearchConfigError, match="scopus"):
        search_module.search_node(state)
    assert called == []   # not even pubmed was dispatched -- resolved pre-network, before the pool


def test_scopus_heuristic_fallback_raw_pubmed_query_is_pre_network_failure(monkeypatch):
    """search_strategy_builder.py's own `call_status='heuristic_fallback'`
    copies the untranslated PubMed-syntax query into EVERY requested
    database, including scopus -- db_queries['scopus'] is then identical
    to the seed/expanded query. That must never reach the Scopus adapter."""
    called = []

    class _SpyAdapter:
        def search(self, request):
            called.append(True)
            raise AssertionError("adapter must not be dispatched")

    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"pubmed": _SpyAdapter, "scopus": _SpyAdapter})
    state = _state_with_databases(["pubmed", "scopus"])
    state["search_strategy"] = {"db_queries": {"pubmed": "cats", "scopus": "cats"}}
    with pytest.raises(SearchConfigError, match="scopus"):
        search_module.search_node(state)
    assert called == []


def test_pubmed_field_tag_cannot_reach_scopus_even_via_execution_adapter_map(monkeypatch):
    """A resolved scopus query that still carries PubMed field-tag syntax
    (e.g. `[tiab]`) -- however it got there -- must be a pre-network
    failure, not a dispatched request."""
    called = []

    class _SpyAdapter:
        def search(self, request):
            called.append(True)
            raise AssertionError("adapter must not be dispatched")

    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"scopus": _SpyAdapter})
    state = _state_with_databases(["scopus"])
    state["search_strategy"] = {
        "db_queries": {"scopus": '"disaster"[tiab] AND "healthcare"[tiab]'}
    }
    with pytest.raises(SearchConfigError, match=r"\[tiab\]"):
        search_module.search_node(state)
    assert called == []


def test_scopus_genuine_translation_dispatches_normally(monkeypatch):
    scopus_records = [_fake_record("scopus", "S1")]
    monkeypatch.setattr(
        search_module, "_ADAPTER_REGISTRY", {"scopus": _make_fake_adapter(adapter_name="scopus", records=scopus_records)}
    )
    state = _state_with_databases(["scopus"])
    state["search_strategy"] = {
        "db_queries": {"scopus": 'TITLE-ABS-KEY("disaster" OR "healthcare")'}
    }
    result = search_module.search_node(state)
    assert len(result["raw_records"]) == 1
    entry = result["search_log"]["entries"][0]
    assert entry["query"] == 'TITLE-ABS-KEY("disaster" OR "healthcare")'


def test_builder_heuristic_fallback_scopus_blocked_by_real_producer(monkeypatch, tmp_path):
    """Gate 3 (reviewer finding): `test_scopus_genuine_translation_dispatches_normally`
    (above) and the `_STRICT_TRANSLATION_DBS` tests earlier in this file
    all hand-construct `state["search_strategy"]["db_queries"]` -- they
    prove the guard works, but not that it is actually reachable from the
    REAL translation producer. This test runs the real, unmodified
    `search_strategy_builder_node` (with a mocked OpenAI transport that
    fails DB translation, forcing its own real `call_status=
    "heuristic_fallback"` path in `_generate_db_queries()`) and feeds its
    real output directly into `search_node`, confirming the guard blocks
    scopus end-to-end through the actual producer chain, not just a
    fixture that assumes the producer's shape.

    Residual risk, explicitly accepted (not silently glossed over): a live
    run whose DB translation call fails this way still spends that one
    (tiny) OpenAI call before `search_node` aborts pre-network on scopus --
    the guard prevents an invalid Scopus dispatch, it does not prevent the
    translation call itself from being attempted and billed once."""
    from unittest.mock import MagicMock

    import slr_pipeline.nodes.search_strategy_builder as ssb_module
    from slr_pipeline.nodes.search_strategy_builder import search_strategy_builder_node

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeResp:
        def __init__(self, content, model="gpt-5.4-mini-2026-01-01"):
            self.choices = [_FakeChoice(content)]
            self.model = model

    def _create(*, model, response_format, messages, temperature):
        return _FakeResp("not valid json{{{")   # forces the real heuristic_fallback path

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create
    fake_client.base_url = "https://api.openai.com/v1/"
    monkeypatch.setattr(ssb_module, "OpenAI", lambda **kw: fake_client)

    raw_query = '"disaster"[tiab]'
    cfg = scoping_review_config(title="t", output_dir=str(tmp_path / "outputs"))
    cfg.search = SearchConfig(databases=["pubmed", "scopus"], query=raw_query)
    cfg.search.strategy_builder.enabled = False   # isolate db translation
    state = dict(initial_state(cfg.to_dict()))
    state["review_run_id"] = "gate3-integration-run"

    builder_result = search_strategy_builder_node(state)
    # Confirmed real root cause reproduced via the real producer, not
    # assumed: heuristic_fallback really does leak the raw PubMed-syntax
    # query into db_queries['scopus'].
    assert builder_result["search_strategy"]["db_queries"]["scopus"] == raw_query

    state.update(builder_result)
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {"pubmed": None, "scopus": None})
    with pytest.raises(SearchConfigError, match="scopus"):
        search_module.search_node(state)


def test_builder_structured_outputs_valid_response_reaches_both_adapters(monkeypatch, tmp_path):
    """Track-3 authorization (2026-09-12) gate: a valid, schema-conforming
    fake response (including a real, distinct Scopus-syntax value for the
    required `scopus` key) must flow all the way through the REAL,
    unmodified `search_strategy_builder_node` -> `search_node` chain
    (same producer-to-consumer shape as
    `test_builder_heuristic_fallback_scopus_blocked_by_real_producer` above,
    just the happy-path sibling) without tripping `_STRICT_TRANSLATION_DBS`,
    and both pubmed and scopus adapters must actually be dispatched."""
    from unittest.mock import MagicMock

    import slr_pipeline.nodes.search_strategy_builder as ssb_module
    from slr_pipeline.nodes.search_strategy_builder import search_strategy_builder_node

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeResp:
        def __init__(self, content, model="gpt-5.4-mini-2026-01-01"):
            self.choices = [_FakeChoice(content)]
            self.model = model

    captured_response_formats = []

    def _create(*, model, response_format, messages, temperature):
        captured_response_formats.append(response_format)
        return _FakeResp(json.dumps({
            "pubmed": '"disaster"[tiab]',
            "scopus": 'TITLE-ABS-KEY("disaster" OR "healthcare")',
        }))

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _create
    fake_client.base_url = "https://api.openai.com/v1/"
    monkeypatch.setattr(ssb_module, "OpenAI", lambda **kw: fake_client)

    raw_query = '"disaster"[tiab]'
    cfg = scoping_review_config(title="t", output_dir=str(tmp_path / "outputs"))
    cfg.search = SearchConfig(databases=["pubmed", "scopus"], query=raw_query)
    cfg.search.strategy_builder.enabled = False   # isolate db translation
    state = dict(initial_state(cfg.to_dict()))
    state["review_run_id"] = "structured-outputs-happy-path-run"

    builder_result = search_strategy_builder_node(state)
    assert builder_result["search_strategy"]["db_queries"]["scopus"] == 'TITLE-ABS-KEY("disaster" OR "healthcare")'

    # The actual request sent to the (fake) OpenAI transport used Structured
    # Outputs with a schema requiring both configured databases.
    sent_response_format = captured_response_formats[0]
    assert sent_response_format["type"] == "json_schema"
    schema = sent_response_format["json_schema"]["schema"]
    assert sent_response_format["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"pubmed", "scopus"}

    pubmed_records = [_fake_record("pubmed", "P1")]
    scopus_records = [_fake_record("scopus", "S1")]
    state.update(builder_result)
    monkeypatch.setattr(search_module, "_ADAPTER_REGISTRY", {
        "pubmed": _make_fake_adapter(adapter_name="pubmed", records=pubmed_records),
        "scopus": _make_fake_adapter(adapter_name="scopus", records=scopus_records),
    })
    result = search_module.search_node(state)
    dispatched = {entry["database"] for entry in result["search_log"]["entries"]}
    assert dispatched == {"pubmed", "scopus"}
