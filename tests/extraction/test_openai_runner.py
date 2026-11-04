"""Offline tests for openai_runner.py. No real network call: the OpenAI
client is faked by patching `openai.OpenAI` itself (the function does a
LAZY `from openai import OpenAI` inside its own body, so patching the
attribute on the `openai` module -- not a name inside openai_runner's own
namespace, since it never binds one at module level -- is what actually
takes effect)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import openai

from slr_pipeline.extraction.openai_runner import (
    ExtractionCallError, OpenAIExtractionConfig, RawBatchResponse, build_batch_prompt, run_openai_batch_extraction,
)
from slr_pipeline.extraction.parsing import ParsedChunk
from slr_pipeline.extraction.schema import AllowedValue, ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind


def _schema() -> ExtractionSchema:
    return ExtractionSchema(
        schema_id="s", version="1",
        fields=[
            ExtractionFieldSpec(
                field_id="study_design", question="What design?", description="d",
                value_kind=ValueKind.CATEGORICAL, cardinality="single",
                allowed_values=(AllowedValue(value="rct", definition="Randomized controlled trial"),),
                provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
            ),
        ],
    )


def _chunk() -> ParsedChunk:
    return ParsedChunk(chunk_id="doc-chunk-0000", order_index=0, section="Methods", page_min=3, page_max=3, text="We used a randomized controlled trial design.")


def _config(**overrides) -> OpenAIExtractionConfig:
    defaults = dict(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium", max_completion_tokens=4096,
        expected_provider_model="gpt-5.6-terra",
    )
    defaults.update(overrides)
    return OpenAIExtractionConfig(**defaults)


def test_config_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        _config(model="  ")


def test_config_rejects_empty_model_config_id():
    with pytest.raises(ValueError, match="model_config_id"):
        _config(model_config_id="")


def test_config_rejects_empty_expected_provider_model():
    with pytest.raises(ValueError, match="expected_provider_model"):
        _config(expected_provider_model="")


def test_config_rejects_invalid_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning_effort"):
        _config(reasoning_effort="ultra")  # type: ignore[arg-type]


def test_config_rejects_zero_max_completion_tokens():
    with pytest.raises(ValueError, match="max_completion_tokens"):
        _config(max_completion_tokens=0)


def test_config_rejects_negative_max_completion_tokens():
    with pytest.raises(ValueError, match="max_completion_tokens"):
        _config(max_completion_tokens=-10)


def test_build_batch_prompt_includes_controlled_values_and_definitions():
    prompt = build_batch_prompt(_schema(), [_chunk()])
    assert '"rct": Randomized controlled trial' in prompt


def test_build_batch_prompt_includes_chunk_id_and_page():
    prompt = build_batch_prompt(_schema(), [_chunk()])
    assert "doc-chunk-0000" in prompt
    assert "page: 3" in prompt


def test_build_batch_prompt_never_asks_model_to_assert_a_page_number():
    prompt = build_batch_prompt(_schema(), [_chunk()])
    assert "quotation" in prompt
    assert '"page"' not in prompt  # the response schema never asks for a page field from the model


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50, total_tokens=150) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, content: str, model: str = "gpt-5.6-terra") -> None:
        self.choices = [_FakeChoice(content)]
        self.model = model
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, response_content: str, model: str = "gpt-5.6-terra", raise_exc: Exception | None = None) -> None:
        self._content = response_content
        self._model = model
        self._raise = raise_exc
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise:
            raise self._raise
        return _FakeResponse(self._content, model=self._model)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, response_content: str = "", model: str = "gpt-5.6-terra", raise_exc: Exception | None = None) -> None:
        self.chat = _FakeChat(_FakeCompletions(response_content, model=model, raise_exc=raise_exc))

    def __call__(self, *args, **kwargs):  # so `OpenAI()` construction itself returns this instance
        return self


def _install_fake_openai(monkeypatch, **kwargs):
    fake_client = _FakeOpenAIClient(**kwargs)
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **kw: fake_client)
    return fake_client


def _install_fake_openai_capturing_constructor_kwargs(monkeypatch, **kwargs):
    """Round-6 (calibration cap review) regression: captures the kwargs
    the `OpenAI(...)` CONSTRUCTOR itself was called with -- distinct from
    `_install_fake_openai`'s own `_FakeCompletions.last_kwargs`, which
    only captures the LATER `.chat.completions.create(...)` call's own
    kwargs. `max_retries` is a constructor argument, never a per-call
    one, so it can only be observed here."""
    fake_client = _FakeOpenAIClient(**kwargs)
    constructor_calls: list = []

    def _fake_constructor(*args, **kw):
        constructor_calls.append(kw)
        return fake_client

    monkeypatch.setattr(openai, "OpenAI", _fake_constructor)
    return fake_client, constructor_calls


VALID_JSON_RESPONSE = (
    '{"fields": {"study_design": {"outcome": "found", "findings": '
    '[{"value": "rct", "confidence": "high", "rationale": "explicit", '
    '"evidence": [{"chunk_id": "doc-chunk-0000", "quotation": "randomized controlled trial"}]}]}}}'
)


def test_run_openai_batch_extraction_success(monkeypatch):
    _install_fake_openai(monkeypatch, response_content=VALID_JSON_RESPONSE)
    raw, telemetry = run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())
    assert isinstance(raw, RawBatchResponse)
    assert raw.fields["study_design"].outcome == "found"
    assert telemetry.provider_model == "gpt-5.6-terra"
    assert telemetry.prompt_tokens == 100


def test_run_openai_batch_extraction_passes_explicit_model_and_reasoning_effort(monkeypatch):
    fake_client = _install_fake_openai(monkeypatch, response_content=VALID_JSON_RESPONSE)
    run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config(model="gpt-5.6-terra", reasoning_effort="medium", max_completion_tokens=4096))
    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-5.6-terra"
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["max_completion_tokens"] == 4096
    assert kwargs["response_format"] == {"type": "json_object"}


def test_run_openai_batch_extraction_omits_reasoning_effort_when_none(monkeypatch):
    fake_client = _install_fake_openai(monkeypatch, response_content=VALID_JSON_RESPONSE)
    run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config(reasoning_effort=None))
    kwargs = fake_client.chat.completions.last_kwargs
    assert "reasoning_effort" not in kwargs


def test_run_openai_batch_extraction_call_failure_raises_extraction_call_error(monkeypatch):
    _install_fake_openai(monkeypatch, raise_exc=RuntimeError("connection reset"))
    with pytest.raises(ExtractionCallError):
        run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())


def test_run_openai_batch_extraction_malformed_json_raises_extraction_call_error(monkeypatch):
    _install_fake_openai(monkeypatch, response_content="not valid json at all {{{")
    with pytest.raises(ExtractionCallError):
        run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())


def test_run_openai_batch_extraction_schema_violating_json_raises_extraction_call_error(monkeypatch):
    _install_fake_openai(monkeypatch, response_content='{"unexpected_key": true}')
    with pytest.raises(ExtractionCallError):
        run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())


def test_run_openai_batch_extraction_requires_no_credentials_setup_by_this_module(monkeypatch):
    """This function never reads OPENAI_API_KEY itself -- the (faked)
    OpenAI() constructor is the only thing that would; confirmed here by
    NOT setting the env var at all and still succeeding against the fake."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install_fake_openai(monkeypatch, response_content=VALID_JSON_RESPONSE)
    raw, _ = run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())
    assert raw.fields["study_design"].outcome == "found"


def test_run_openai_batch_extraction_provider_model_mismatch_raises(monkeypatch):
    """The provider returned a DIFFERENT model than this run's explicit
    `expected_provider_model` policy committed to -- exact string
    comparison, no alias normalization; a real substitution (silent
    downgrade, A/B routing, etc.) must never pass through as a clean
    result."""
    _install_fake_openai(monkeypatch, response_content=VALID_JSON_RESPONSE, model="gpt-5.6-terra-2026-08-01")
    with pytest.raises(ExtractionCallError, match="provider model mismatch"):
        run_openai_batch_extraction(
            schema=_schema(), batch_chunks=[_chunk()], config=_config(expected_provider_model="gpt-5.6-terra"),
        )


def test_run_openai_batch_extraction_disables_sdk_retries(monkeypatch):
    """The installed openai SDK (2.7.2) defaults `OpenAI()`'s own
    `max_retries` to 2 -- a bare `OpenAI()` would transparently retry
    transient failures up to twice at the HTTP layer, below this
    function's own visibility, making its "one attempt" docstring claim
    false (up to 3 real, billable HTTP attempts per logical call). This
    is the direct regression: the constructor must be called with
    `max_retries=0` explicitly."""
    fake_client, constructor_calls = _install_fake_openai_capturing_constructor_kwargs(monkeypatch, response_content=VALID_JSON_RESPONSE)
    run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())
    assert len(constructor_calls) == 1
    assert constructor_calls[0].get("max_retries") == 0


def test_run_openai_batch_extraction_uses_resolved_api_key(monkeypatch):
    """Track 1 Phase A follow-up (Codex review, finding 3): this site must
    be wired through the shared `resolve_openai_api_key()` resolver, not
    left to the SDK's own implicit environment lookup -- an explicit,
    auditable, testable contract."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-value")
    fake_client, constructor_calls = _install_fake_openai_capturing_constructor_kwargs(monkeypatch, response_content=VALID_JSON_RESPONSE)
    run_openai_batch_extraction(schema=_schema(), batch_chunks=[_chunk()], config=_config())
    assert len(constructor_calls) == 1
    assert constructor_calls[0].get("api_key") == "sk-real-env-value"


def test_run_openai_batch_extraction_provider_model_exact_match_succeeds(monkeypatch):
    """A caller that KNOWS the provider echoes a dated alias sets
    `expected_provider_model` to that exact alias -- the check is then an
    exact match against the caller's own explicit policy, not a
    hardcoded assumption inside this module."""
    _install_fake_openai(monkeypatch, response_content=VALID_JSON_RESPONSE, model="gpt-5.6-terra-2026-08-01")
    raw, telemetry = run_openai_batch_extraction(
        schema=_schema(), batch_chunks=[_chunk()],
        config=_config(expected_provider_model="gpt-5.6-terra-2026-08-01"),
    )
    assert telemetry.provider_model == "gpt-5.6-terra-2026-08-01"
