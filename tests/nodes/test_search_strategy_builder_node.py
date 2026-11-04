"""Offline tests for slr_pipeline.nodes.search_strategy_builder --
Stage 3 node instrumentation: query expansion (no fallback) and db
translation (has a real fallback) are each a durable AIUseEvent;
call_status/output-existence and response-version availability are
independent axes (F23) -- a malformed response at the FALLBACK site still
retains resp.model even though call_status="heuristic_fallback"."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from slr_pipeline.ai_traceability import AIUseEventStore
from slr_pipeline.config import ReviewConfig
from slr_pipeline.state import initial_state

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


def _config(tmp_path, **overrides) -> ReviewConfig:
    cfg = ReviewConfig(
        title="Test review", output_dir=str(tmp_path / "outputs"), research_questions=["q"],
    )
    cfg.search.databases = ["openalex", "arxiv"]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _state(cfg: ReviewConfig) -> dict:
    st = dict(initial_state(cfg.to_dict()))
    st["review_run_id"] = "ssb-test-run"
    return st


def _install_fake_client(create_fn):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = create_fn
    fake_client.base_url = "https://api.openai.com/v1/"
    ssb_module.OpenAI = lambda **kw: fake_client
    return fake_client


def _dispatch(user_content: str, expand_resp, translate_resp):
    if "adapting a PubMed" in user_content:
        return translate_resp
    return expand_resp


def test_query_expansion_and_db_translation_both_succeed(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        user = messages[1]["content"]
        if "adapting a PubMed" in user:
            return _FakeResp(json.dumps({"openalex": "x", "arxiv": "y", "pubmed": "orig"}))
        return _FakeResp(json.dumps({"concepts": [], "candidate_query": "(a OR b)", "notes": ["ok"]}))
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.strategy_builder.enabled = True
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["expanded_query"] == "(a OR b)"
    assert result["oversight_mode"]["search_strategy"] == "automation_unreviewed"
    assert result["automation_origin"]["search_strategy"] == "ai"

    output_dir = cfg.ensure_output_dir()
    events = AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()
    assert len(events) == 2
    assert all(e.call_status == "success" for e in events)
    assert all(e.access_info == "https://api.openai.com/v1/" for e in events)


def test_query_expansion_transport_failure_no_fallback_empty_component_versions(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        raise RuntimeError("simulated transport failure")
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.strategy_builder.enabled = True
    result = search_strategy_builder_node(_state(cfg))

    # no substitute exists for query expansion -- the seed query passes
    # through unchanged, and db translation's OWN call is separate and
    # also fails transport-wise here, so db_queries falls back too.
    assert result["search_strategy"]["expanded_query"] == cfg.search.query

    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    expansion = events["search_strategy_query_expansion"]
    assert expansion.call_status == "failure"
    assert expansion.output_artifact_refs == []
    assert expansion.component_versions == {}


def test_query_expansion_parse_failure_retains_response_model(tmp_path):
    """F23: a malformed response at the no-fallback query-expansion site
    is still call_status='failure', but component_versions retains
    resp.model since a response WAS received."""
    def _create(*, model, response_format, messages, temperature):
        user = messages[1]["content"]
        if "adapting a PubMed" in user:
            return _FakeResp(json.dumps({"openalex": "x", "pubmed": "orig"}))
        return _FakeResp("not valid json{{{", model="gpt-5.4-mini-parsefail")
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.strategy_builder.enabled = True
    search_strategy_builder_node(_state(cfg))

    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    expansion = events["search_strategy_query_expansion"]
    assert expansion.call_status == "failure"
    assert expansion.output_artifact_refs == []
    assert expansion.component_versions == {"gpt-5.4-mini": "gpt-5.4-mini-parsefail"}


def test_db_translation_malformed_json_produces_heuristic_fallback_with_response_model(tmp_path):
    """F23's exact regression: db translation has a REAL fallback (same
    query for every DB), so a malformed response is call_status=
    'heuristic_fallback' with a non-empty output hash AND component_versions
    still carrying resp.model -- not {}, the bug an earlier plan round had."""
    def _create(*, model, response_format, messages, temperature):
        return _FakeResp("not valid json{{{", model="gpt-5.4-mini-parsefail")
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.strategy_builder.enabled = False   # skip expansion, isolate db translation
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["db_queries"]["pubmed"] == cfg.search.query
    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    translation = events["search_strategy_db_translation"]
    assert translation.call_status == "heuristic_fallback"
    assert translation.output_artifact_refs != []
    assert translation.component_versions == {"gpt-5.4-mini": "gpt-5.4-mini-parsefail"}


def test_db_translation_transport_failure_empty_component_versions(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        raise RuntimeError("simulated transport failure")
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.strategy_builder.enabled = False
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["db_queries"]["openalex"] == cfg.search.query
    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    translation = events["search_strategy_db_translation"]
    assert translation.call_status == "heuristic_fallback"
    assert translation.component_versions == {}


def test_db_list_is_deterministically_sorted_across_identical_calls(tmp_path):
    """dbs_needed = sorted(...) -- an unsorted set's iteration order isn't
    guaranteed stable, which would make rendered_prompt_hash (and the
    rendered prompt text itself) non-deterministic across otherwise-
    identical calls."""
    captured_prompts = []

    def _create(*, model, response_format, messages, temperature):
        captured_prompts.append(messages[1]["content"])
        return _FakeResp(json.dumps({"pubmed": "orig"}))
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.databases = ["ieee", "arxiv", "crossref", "openalex"]
    cfg.search.strategy_builder.enabled = False
    search_strategy_builder_node(_state(cfg))

    prompt = captured_prompts[0]
    order = [db for db in ("arxiv", "crossref", "ieee", "openalex", "pubmed") if db in prompt]
    positions = [prompt.index(f"- {db}:") for db in order]
    assert positions == sorted(positions)   # alphabetical order in the rendered db_list


def test_automation_origin_mixed_when_expansion_succeeds_and_translation_falls_back(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        user = messages[1]["content"]
        if "adapting a PubMed" in user:
            raise RuntimeError("translation transport failure")
        return _FakeResp(json.dumps({"concepts": [], "candidate_query": "(a)", "notes": []}))
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.strategy_builder.enabled = True
    result = search_strategy_builder_node(_state(cfg))
    assert result["automation_origin"]["search_strategy"] == "mixed"


# ---------------------------------------------------------------------------
# Semantic validation (codex review, finding 1): an incomplete DB-
# translation response (missing a configured database) must never be
# accepted as "success" -- it would silently omit that database from the
# eventual search.
# ---------------------------------------------------------------------------

def test_db_translation_missing_database_falls_back(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        return _FakeResp(json.dumps({"pubmed": "orig", "openalex": "x"}))   # "arxiv" is missing
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.databases = ["openalex", "arxiv"]
    cfg.search.strategy_builder.enabled = False
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["db_queries"]["arxiv"] == cfg.search.query   # fallback filled it in
    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    assert events["search_strategy_db_translation"].call_status == "heuristic_fallback"


def test_db_translation_empty_query_value_falls_back(tmp_path):
    def _create(*, model, response_format, messages, temperature):
        return _FakeResp(json.dumps({"pubmed": "orig", "openalex": "   "}))   # whitespace-only
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.databases = ["openalex"]
    cfg.search.strategy_builder.enabled = False
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["db_queries"]["openalex"] == cfg.search.query
    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    assert events["search_strategy_db_translation"].call_status == "heuristic_fallback"


# ---------------------------------------------------------------------------
# codex review (second round): str(q) coerced null/array/object/number/
# boolean values into truthy-looking strings that passed validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [None, [], {}, 42, True, False])
def test_db_translation_non_string_value_falls_back(tmp_path, bad_value):
    def _create(*, model, response_format, messages, temperature):
        return _FakeResp(json.dumps({"pubmed": "orig", "openalex": bad_value}))
    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.databases = ["openalex"]
    cfg.search.strategy_builder.enabled = False
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["db_queries"]["openalex"] == cfg.search.query   # fallback, not stringified garbage
    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    assert events["search_strategy_db_translation"].call_status == "heuristic_fallback"


# ---------------------------------------------------------------------------
# Track-3 authorization (2026-09-12): Structured Outputs schema must
# require EVERY configured database key (never a fixed literal list), with
# additionalProperties: False -- proven directly against the actual request
# sent to the OpenAI transport, not just the eventual db_queries output.
# ---------------------------------------------------------------------------

def test_db_translation_schema_requires_every_configured_database(tmp_path):
    captured_response_formats = []

    def _create(*, model, response_format, messages, temperature):
        captured_response_formats.append(response_format)
        return _FakeResp(json.dumps({"pubmed": "orig", "scopus": 'TITLE-ABS-KEY("x")'}))

    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.databases = ["scopus"]
    cfg.search.strategy_builder.enabled = False
    search_strategy_builder_node(_state(cfg))

    response_format = captured_response_formats[0]
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"pubmed", "scopus"}
    assert set(schema["properties"].keys()) == {"pubmed", "scopus"}


def test_db_translation_missing_key_from_nonconforming_response_still_falls_back(tmp_path):
    """Structured Outputs makes a missing key RARE, not provably
    impossible (a non-conforming response is still possible in principle)
    -- `_generate_db_queries()`'s own missing/empty-key validation and
    `heuristic_fallback` substitute must remain intact as defense-in-depth
    even after the schema change, for scopus specifically (not just the
    openalex/arxiv examples already covered above)."""
    def _create(*, model, response_format, messages, temperature):
        return _FakeResp(json.dumps({"pubmed": "orig"}))   # "scopus" missing despite the schema requiring it

    _install_fake_client(_create)

    cfg = _config(tmp_path)
    cfg.search.databases = ["scopus"]
    cfg.search.strategy_builder.enabled = False
    result = search_strategy_builder_node(_state(cfg))

    assert result["search_strategy"]["db_queries"]["scopus"] == cfg.search.query   # fallback filled it in
    output_dir = cfg.ensure_output_dir()
    events = {e.stage: e for e in AIUseEventStore(output_dir=output_dir, review_run_id="ssb-test-run").load_all()}
    assert events["search_strategy_db_translation"].call_status == "heuristic_fallback"
