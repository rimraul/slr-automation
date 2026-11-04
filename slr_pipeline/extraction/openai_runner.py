"""The OpenAI extraction runner -- ONE explicit provider path (requirement
10). This module is deliberately the ONLY place that constructs an OpenAI
client or reads `OPENAI_API_KEY`; a later local-model runner
(`local_runner.py`, not built in this phase) will be a SEPARATE module
with its own config/prompt/cache/checkpoint identity, never a branch
inside this one and never something this module falls back to. See this
package's own `__init__.py` docstring for the full "shared contract,
provider-specific execution, no auto-detection" rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .parsing import ParsedChunk
from .schema import ExtractionSchema

PROMPT_TEMPLATE_VERSION = "extraction-openai-prompt-1"
OUTPUT_SCHEMA_VERSION = "extraction-openai-schema-1"

_SYSTEM_MESSAGE = (
    "You are a meticulous systematic-review data-extraction assistant. "
    "You extract structured data from one batch of a scientific paper's text. "
    "Return only valid JSON -- no markdown, no explanation outside the JSON object."
)


class ExtractionCallError(Exception):
    """A single batch's LLM call failed -- network/API error, malformed
    JSON, or schema-validation failure. There is no fallback: the caller
    marks this batch's outcome as "failed" and moves on (or stops); this
    function never retries with a different model or provider."""


class ProviderModelMismatchError(ExtractionCallError):
    """The provider returned a DIFFERENT model than this run's
    `expected_provider_model` policy committed to. Carries the ACTUAL
    returned model and token usage (sanitized -- identifiers and counts
    only, never prompt/response content) so the caller can still record
    accurate telemetry on the batch's own failed `BatchExecutionRecord`
    before rejecting the substituted response's CONTENT (review finding
    P: "a real provider substitution is rejected but its returned
    identity is discarded" -- without this, a run's own manifest could
    report `provider_model_consistent=True` vacuously, since the batch
    that actually witnessed the substitution left no telemetry trace at
    all)."""

    def __init__(
        self, *, expected_provider_model: str, actual_provider_model: str,
        prompt_tokens: Optional[int], completion_tokens: Optional[int], total_tokens: Optional[int],
    ) -> None:
        super().__init__(f"provider model mismatch: expected {expected_provider_model!r}, got {actual_provider_model!r}")
        self.actual_provider_model = actual_provider_model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


# --------------------------------------------------------------------------- #
# Raw (pre-verification) response shape -- validated, but evidence within it
# is NOT yet trusted; evidence.verify_evidence() is the trust boundary,
# applied by the caller (engine.py), not here.
# --------------------------------------------------------------------------- #


class RawEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    quotation: str


class RawFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: Optional[Literal["high", "medium", "low"]] = None
    rationale: Optional[str] = None
    evidence: list[RawEvidenceItem] = Field(default_factory=list)


class RawFieldResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["found", "not_found", "unclear"]
    findings: list[RawFinding] = Field(default_factory=list)
    rationale: Optional[str] = None


class RawBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: dict[str, RawFieldResponse]


_VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class OpenAIExtractionConfig:
    """Explicit, no defaults on the fields that determine what actually
    runs (requirement 10: "explicit model, model configuration ID,
    reasoning effort and token budget") -- mirrors
    `build_supported_screening_config()`'s own "never silently choose a
    model/reasoning-effort/token-budget" discipline from the FT_SCREENING
    integration, reimplemented fresh here since this package does not
    import FT_SCREENING.

    `expected_provider_model` is the run-bound policy this config commits
    to for provider-substitution detection (review constraint: "an
    explicit, run-bound expected_provider_model policy... Do not assume
    the requested name must always equal the returned name or silently
    normalize dated/alias variants"). It is deliberately a SEPARATE field
    from `model` -- `model` is what is SENT in the API call;
    `expected_provider_model` is what the caller has decided counts as an
    acceptable RETURNED `response.model` for this run (which may differ
    from `model` for a provider that is known to echo back a dated/
    versioned alias, e.g. requesting "gpt-5.6-terra" and always receiving
    "gpt-5.6-terra-2026-08-01" back) -- but that mapping is the CALLER's
    explicit, recorded decision, never something this module infers or
    string-normalizes on its own."""

    model: str
    model_config_id: str
    reasoning_effort: Optional[Literal["none", "low", "medium", "high", "xhigh", "max"]]
    max_completion_tokens: int
    expected_provider_model: str
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    output_schema_version: str = OUTPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not self.model_config_id.strip():
            raise ValueError("model_config_id must be a non-empty string")
        if not self.expected_provider_model.strip():
            raise ValueError("expected_provider_model must be a non-empty string")
        if self.reasoning_effort is not None and self.reasoning_effort not in _VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort {self.reasoning_effort!r} is not one of {sorted(_VALID_REASONING_EFFORTS)}"
            )
        if self.max_completion_tokens <= 0:
            raise ValueError(f"max_completion_tokens must be positive, got {self.max_completion_tokens!r}")
        if not self.prompt_template_version.strip():
            raise ValueError("prompt_template_version must be a non-empty string")
        if not self.output_schema_version.strip():
            raise ValueError("output_schema_version must be a non-empty string")


@dataclass(frozen=True)
class BatchCallTelemetry:
    """Provider-RETURNED identity/usage (requirement 10: "provider-
    returned model and token telemetry recorded") -- `provider_model` is
    what the API actually reports back, not `config.model` echoed --
    catches a silent model substitution on the provider's own side."""

    provider_model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


def build_batch_prompt(schema: ExtractionSchema, batch_chunks: list[ParsedChunk]) -> str:
    """Every controlled value and its definition goes into the prompt
    (lit_review_protocol requirement) -- not just the bare allowed-values
    list. Trusted chunk_id/section/page are shown for each chunk so the
    model can cite them back; the model is never asked to invent or
    report a page number itself."""
    field_blocks = []
    for field in schema.fields:
        block = (
            f'- field_id: "{field.field_id}"\n'
            f"  question: {field.question}\n"
            f"  description: {field.description}\n"
            f"  value_kind: {field.value_kind.value}\n"
            f"  cardinality: {field.cardinality}"
        )
        if field.allowed_values:
            values_text = "\n".join(f'    * "{av.value}": {av.definition}' for av in field.allowed_values)
            block += f"\n  allowed_values:\n{values_text}"
        field_blocks.append(block)
    fields_text = "\n\n".join(field_blocks)

    chunk_blocks = []
    for chunk in batch_chunks:
        page_label = chunk.page_range or "unknown"
        chunk_blocks.append(
            f"[chunk_id: {chunk.chunk_id} | section: {chunk.section} | page: {page_label}]\n{chunk.text}"
        )
    chunks_text = "\n\n".join(chunk_blocks)

    return f"""You are extracting structured data from ONE batch of a scientific paper's text for a systematic review. Answer ONLY using the text in this batch below -- never use outside knowledge, and never guess a value from a field's name/description alone.

For EACH field listed, report exactly one outcome:
- "found": this batch's text directly supports a specific value. Give one or more "findings", each with:
    - "value": the extracted value (a string; for categorical fields, exactly one of the allowed_values).
    - "confidence": "high", "medium", or "low".
    - "rationale": one short sentence.
    - "evidence": a list of {{"chunk_id": ..., "quotation": ...}} pairs. "quotation" MUST be copied VERBATIM
      from the exact chunk_id's text below -- do not paraphrase, translate, summarize, or fix typos in it.
      "chunk_id" MUST be one of the chunk_id values shown below -- never invent one.
- "unclear": this batch's text seems to touch the field, but no confident value can be stated. You may still
  cite evidence for the ambiguous passage in "findings" (value can describe the ambiguity), and/or a top-level
  "rationale".
- "not_found": this batch's text does not address the field at all. No findings needed.

Fields:
{fields_text}

Text batch (each block is one trusted, citable chunk):
{chunks_text}

Respond with exactly one JSON object of this shape, covering EVERY field_id listed above:
{{"fields": {{"<field_id>": {{"outcome": "found"|"not_found"|"unclear", "findings": [...], "rationale": null}}, ...}}}}"""


def run_openai_batch_extraction(
    *, schema: ExtractionSchema, batch_chunks: list[ParsedChunk], config: OpenAIExtractionConfig,
) -> tuple[RawBatchResponse, BatchCallTelemetry]:
    """The ONE call this whole module makes. Credentials are read lazily,
    by the `openai` SDK itself, from `OPENAI_API_KEY` in the process
    environment -- this function never reads, holds, or is passed a key
    (requirement 10: "credentials loaded lazily from the environment...
    no key in ReviewConfig or persisted state"). No retry, no fallback:
    one attempt, and any failure raises `ExtractionCallError` for the
    caller to record as this batch's outcome.

    `max_retries=0` is passed explicitly to the SDK client -- the "one
    attempt" claim above was FALSE until this was added: the installed
    `openai` SDK (2.7.2) defaults `OpenAI()`'s own `max_retries` to `2`,
    meaning a bare `OpenAI()` transparently retried transient failures
    (connection errors, 429/5xx) up to twice at the HTTP layer BELOW this
    function's own visibility -- so what this module called "one attempt"
    could silently become up to three real HTTP attempts per logical
    call, each one a genuine billable request. `max_retries=0` makes the
    SDK's own behavior match what this module has always claimed."""
    from openai import OpenAI  # lazy import -- never at module import time
    from slr_pipeline.config import resolve_openai_api_key  # lazy import, same reason -- see below

    prompt = build_batch_prompt(schema, batch_chunks)
    call_kwargs: dict = dict(
        model=config.model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=config.max_completion_tokens,
    )
    if config.reasoning_effort is not None:
        call_kwargs["reasoning_effort"] = config.reasoning_effort

    try:
        # Codex review (Phase A follow-up, finding 3): explicitly wired
        # through the shared resolver rather than relying on the SDK's own
        # implicit `OPENAI_API_KEY` lookup when `api_key` is omitted -- same
        # runtime behavior, but now an explicit, auditable, testable
        # contract instead of an implicit coincidence. `resolve_openai_api_key()`
        # is zero-argument and reads the environment exclusively, so it needs
        # no `ReviewConfig` in scope here (this module intentionally never
        # sees one -- only `OpenAIExtractionConfig`).
        client = OpenAI(api_key=resolve_openai_api_key(), max_retries=0)
        response = client.chat.completions.create(**call_kwargs)
    except Exception as exc:
        raise ExtractionCallError(f"OpenAI call failed: {type(exc).__name__}") from exc

    provider_model = response.model
    if provider_model != config.expected_provider_model:
        # Provider substitution -- the API silently answered with a
        # DIFFERENT model than this run committed to. Exact string
        # comparison against the caller's own recorded policy, never a
        # normalized/alias-aware comparison (review constraint: "Do not
        # assume the requested name must always equal the returned name
        # or silently normalize dated/alias variants" -- a caller that
        # KNOWS the provider echoes a dated alias sets
        # `expected_provider_model` to that alias explicitly; this
        # function itself never guesses that mapping). The mismatch
        # exception itself carries the returned model/usage (review
        # finding P) -- the CONTENT is rejected, but its identity is not
        # discarded.
        mismatch_usage = getattr(response, "usage", None)
        raise ProviderModelMismatchError(
            expected_provider_model=config.expected_provider_model, actual_provider_model=provider_model,
            prompt_tokens=getattr(mismatch_usage, "prompt_tokens", None) if mismatch_usage else None,
            completion_tokens=getattr(mismatch_usage, "completion_tokens", None) if mismatch_usage else None,
            total_tokens=getattr(mismatch_usage, "total_tokens", None) if mismatch_usage else None,
        )

    try:
        raw_json = json.loads(response.choices[0].message.content or "")
        parsed = RawBatchResponse.model_validate(raw_json)
    except Exception as exc:
        raise ExtractionCallError(f"response validation failed: {type(exc).__name__}") from exc

    usage = getattr(response, "usage", None)
    telemetry = BatchCallTelemetry(
        provider_model=provider_model,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
    )
    return parsed, telemetry
