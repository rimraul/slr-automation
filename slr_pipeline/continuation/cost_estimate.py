"""Track 3 continuation plan Unit F -- whole-pipeline worst-case cost
estimate for a continuation run.

Reframed (plan v22, direct user code-review finding 4): this module's
output is CONSERVATIVE PLANNING INFORMATION for the human choosing
`--cap-usd`, not a mathematically exact upper bound and not itself an
enforcement mechanism. Its own tokenizer is an approximate fallback for 2
of the 3 priced models (see `TOKENIZER_SAFETY_MARGIN_MULTIPLIER` below),
so "hard ceiling" is the wrong word for what this estimate is. The actual
ENFORCED hard cap is, and remains, Unit G's pre-existing, unmodified
`cost_cap/ledger.py`/`cost_cap/proxy.py` -- nothing in this module ever
constructs a `CostLedger` or opens a `CostCapProxy`.

Every call's dollar figure goes through the existing, unmodified
`cost_cap.pricing.estimate_worst_case_cost_usd()`; this module is only the
aggregation/enumeration layer over the review's REAL, already-configured
inputs (real `CriteriaSet`/`ScreeningConfig`/`ExtractionSchema`/
`OpenAIExtractionConfig`/`SynthesisConfig`, real `build_batches()`/
`build_batch_prompt()`, real FT_SCREENING prompt assembly), reusing every
one of those unmodified rather than reimplementing an approximation of any
of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tiktoken
from ft_screening import CriteriaSet, EvidenceSpan
from ft_screening import ScreeningConfig as FTScreeningConfig
# `_build_prompt` is module-private in FT_SCREENING (not part of its public
# `ft_screening`/`ft_screening.api` surface) -- reused anyway, deliberately
# and read-only, because it is FT_SCREENING's own real, pure, deterministic
# prompt-assembly function, and the plan's own explicit instruction is to
# reuse it "if importable read-only" rather than hand-mirror its template
# and risk silent drift. No other FT_SCREENING internal is imported here.
from ft_screening.adjudication import _build_prompt as _ft_build_prompt

from slr_pipeline.adapters.ft_screening_adapter import chapter_screening_config
from slr_pipeline.config import ReviewConfig, SynthesisConfig
from slr_pipeline.cost_cap.pricing import estimate_worst_case_cost_usd, is_priced
from slr_pipeline.extraction.batching import build_batches
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig, _SYSTEM_MESSAGE, build_batch_prompt
from slr_pipeline.extraction.parsing import ParsedDocument, parse_pdf
from slr_pipeline.extraction.schema import ExtractionSchema
from slr_pipeline.nodes.extraction import _load_model_config, _load_parser_options
from slr_pipeline.extraction import verified_bundle
from slr_pipeline.nodes.synthesis import (
    SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET,
    _SYNTHESIS_MAP_SYSTEM_PROMPT,
    _SYNTHESIS_MAP_USER_TEMPLATE,
    _SYNTHESIS_REFINE_SYSTEM_PROMPT,
    _SYNTHESIS_REFINE_USER_TEMPLATE,
    _target_display_text,
)
from slr_pipeline.state import Record

# Round-1 code-review finding, confirmed empirically: `tiktoken.
# encoding_for_model()` only resolves "gpt-4o" of the 3 models `cost_cap.
# pricing.PRICING` actually prices -- "gpt-5.4-mini"/"gpt-5.6-terra" both
# raise KeyError and fall back to "o200k_base" (the same encoding gpt-4o
# itself resolves to). Because that fallback is a best-available proxy,
# not each model's real vocabulary, every token count this module produces
# is inflated by this documented safety margin before costing -- this,
# plus the fallback itself, is the reason this estimate is conservative
# planning information, never an exact figure (see module docstring).
TOKENIZER_SAFETY_MARGIN_MULTIPLIER = 1.15


class UnpricedModelError(ValueError):
    """A stage's configured model has no entry in `cost_cap.pricing.
    PRICING`. Per Unit A's own "unpriced or unavailable model IDs must be
    marked unresolved, not silently substituted" rule, this estimate
    refuses to guess a price for that stage rather than silently omitting
    or zeroing it."""


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, *, model: str) -> int:
    """Real tokenizer count of `text` for `model`, inflated by
    `TOKENIZER_SAFETY_MARGIN_MULTIPLIER` and rounded UP -- a worst-case
    count must never round down."""
    encoding = _encoding_for_model(model)
    raw = len(encoding.encode(text))
    return math.ceil(raw * TOKENIZER_SAFETY_MARGIN_MULTIPLIER)


@dataclass(frozen=True)
class StageCostEstimate:
    stage: str
    model: str
    calls: int
    total_prompt_tokens: int
    worst_case_usd: float


def _empty_stage(stage: str) -> StageCostEstimate:
    return StageCostEstimate(stage=stage, model="(no calls)", calls=0, total_prompt_tokens=0, worst_case_usd=0.0)


@dataclass(frozen=True)
class ContinuationCostEstimate:
    stages: Tuple[StageCostEstimate, ...]

    @property
    def grand_total_usd(self) -> float:
        return sum(s.worst_case_usd for s in self.stages)

    def render_breakdown(self) -> str:
        lines = [
            "Continuation worst-case cost estimate -- CONSERVATIVE PLANNING "
            "INFORMATION for choosing --cap-usd. Not a mathematically exact "
            "figure and not itself an enforcement mechanism: the actual "
            "enforced hard cap is the cost-cap proxy/ledger at run time.",
        ]
        for s in self.stages:
            lines.append(
                f"  {s.stage:<26s} model={s.model:<16s} calls={s.calls:>5d} "
                f"prompt_tokens={s.total_prompt_tokens:>10d} worst_case_usd={s.worst_case_usd:>10.4f}"
            )
        lines.append(f"  {'TOTAL':<26s} worst_case_usd={self.grand_total_usd:.4f}")
        return "\n".join(lines)


def _parsed_documents(
    retrieved_manifest_docs: Dict[str, Dict[str, str]],
    preserved_included_parent_records: List[Record],
    *,
    parser_options,
) -> Tuple[Dict[str, ParsedDocument], Dict[str, ParsedDocument]]:
    """Returns (retrieved_parsed, full_text_parsed). Each document is
    parsed exactly once via the same real, local, free, deterministic
    `parse_pdf()` the production pipeline itself uses -- never a bundle
    read (see this unit's plan notes: the release bundle never stages PDF
    bytes for `track3_live`, and no durable already-parsed chunk data
    exists anywhere for a preserved-included record)."""
    retrieved_parsed = {
        record_id: parse_pdf(doc["dest_path"], options=parser_options)
        for record_id, doc in retrieved_manifest_docs.items()
    }
    preserved_parsed = {
        record.id: parse_pdf(record.full_text_path, options=parser_options)
        for record in preserved_included_parent_records
    }
    full_text_parsed = {**preserved_parsed, **retrieved_parsed}
    return retrieved_parsed, full_text_parsed


def _estimate_ft_screening(
    retrieved_parsed: Dict[str, ParsedDocument], criteria_set: CriteriaSet, screening_config: FTScreeningConfig,
) -> StageCostEstimate:
    """One call per (document, criterion), scoped to `retrieved_ids` only
    -- confirmed by Unit C, `preserved_included_ids` are never passed into
    `_run_full_text_eligibility()`, so no FT_SCREENING call is ever made
    for them. Worst case per document: treats the WHOLE document's parsed
    chunks as the evidence set for every criterion -- a safe over-estimate,
    since real per-criterion retrieval (FT_SCREENING's own retriever)
    returns only a subset of a document's chunks."""
    if not retrieved_parsed:
        return _empty_stage("ft_screening_eligibility")

    model = screening_config.adjudicator_model
    if not is_priced(model):
        raise UnpricedModelError(
            f"FT_SCREENING adjudicator_model {model!r} has no cost_cap.pricing.PRICING entry"
        )
    max_output_tokens = screening_config.model_options.get("max_completion_tokens")

    calls = 0
    total_prompt_tokens = 0
    worst_case_usd = 0.0
    for record_id, parsed in retrieved_parsed.items():
        evidence = [
            EvidenceSpan(
                document_id=record_id, document_sha256=parsed.document_sha256, chunk_id=chunk.chunk_id,
                section=chunk.section, page=chunk.page_min, excerpt=chunk.text, stance="relevant",
                retrieval_method="cost_estimate_worst_case", retrieval_query_version="1",
            )
            for chunk in parsed.chunks
        ]
        for criterion in criteria_set.criteria:
            system_prompt, user_prompt = _ft_build_prompt(criterion=criterion, evidence=evidence)
            prompt_tokens = count_tokens(system_prompt, model=model) + count_tokens(user_prompt, model=model)
            worst_case_usd += estimate_worst_case_cost_usd(
                model=model, prompt_tokens=prompt_tokens, max_output_tokens=max_output_tokens,
            )
            total_prompt_tokens += prompt_tokens
            calls += 1
    return StageCostEstimate(
        stage="ft_screening_eligibility", model=model, calls=calls,
        total_prompt_tokens=total_prompt_tokens, worst_case_usd=worst_case_usd,
    )


def _estimate_extraction(
    full_text_parsed: Dict[str, ParsedDocument], schema: ExtractionSchema, model_config: OpenAIExtractionConfig,
    max_batch_chars: int,
) -> StageCostEstimate:
    """One call per real `build_batches()`-derived batch, over every
    document in `full_text_records` (`retrieved_ids` UNION
    `preserved_included_ids`) -- production extraction batches parsed
    chunks and makes one call per BATCH, never one call per field."""
    if not full_text_parsed:
        return _empty_stage("extraction")

    model = model_config.model
    if not is_priced(model):
        raise UnpricedModelError(f"extraction model {model!r} has no cost_cap.pricing.PRICING entry")

    calls = 0
    total_prompt_tokens = 0
    worst_case_usd = 0.0
    for parsed in full_text_parsed.values():
        batches = build_batches(parsed.chunks, max_batch_chars=max_batch_chars)
        chunk_map = parsed.chunk_map()
        for batch in batches:
            batch_chunks = [chunk_map[chunk_id] for chunk_id in batch.chunk_ids]
            prompt_text = build_batch_prompt(schema, batch_chunks)
            prompt_tokens = count_tokens(_SYSTEM_MESSAGE, model=model) + count_tokens(prompt_text, model=model)
            worst_case_usd += estimate_worst_case_cost_usd(
                model=model, prompt_tokens=prompt_tokens, max_output_tokens=model_config.max_completion_tokens,
            )
            total_prompt_tokens += prompt_tokens
            calls += 1
    return StageCostEstimate(
        stage="extraction", model=model, calls=calls,
        total_prompt_tokens=total_prompt_tokens, worst_case_usd=worst_case_usd,
    )


def _estimate_synthesis_map(
    full_text_parsed: Dict[str, ParsedDocument], research_questions: List[str], synthesis_cfg: SynthesisConfig,
) -> StageCostEstimate:
    """One per-RQ study-mapping call per document, over the same
    `full_text_records` corpus as extraction. Worst-case prompt size: the
    real `_build_map_context()` caps its own rendered context at
    `SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET` chars -- sizing every call at that
    cap (rather than from any one document's real extracted data) is a
    safe upper bound on every real call's actual size."""
    n_docs = len(full_text_parsed)
    if n_docs == 0 or not research_questions:
        return _empty_stage("synthesis_map")

    model = synthesis_cfg.llm_model
    if not is_priced(model):
        raise UnpricedModelError(f"synthesis llm_model {model!r} has no cost_cap.pricing.PRICING entry")

    worst_case_context = "x" * SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET
    calls = 0
    total_prompt_tokens = 0
    worst_case_usd = 0.0
    for rq in research_questions:
        user_content = _SYNTHESIS_MAP_USER_TEMPLATE.format(
            target=_target_display_text(rq), rendered_context=worst_case_context,
        )
        prompt_tokens = count_tokens(_SYNTHESIS_MAP_SYSTEM_PROMPT, model=model) + count_tokens(user_content, model=model)
        per_call_usd = estimate_worst_case_cost_usd(model=model, prompt_tokens=prompt_tokens, max_output_tokens=None)
        calls += n_docs
        total_prompt_tokens += prompt_tokens * n_docs
        worst_case_usd += per_call_usd * n_docs
    return StageCostEstimate(
        stage="synthesis_map", model=model, calls=calls,
        total_prompt_tokens=total_prompt_tokens, worst_case_usd=worst_case_usd,
    )


def _estimate_synthesis_refine(
    full_text_parsed: Dict[str, ParsedDocument], research_questions: List[str], synthesis_cfg: SynthesisConfig,
) -> StageCostEstimate:
    """`ceil(n_eligible_documents / refine_batch_size)` refine calls per
    research question, over the same `full_text_records` corpus. Each
    call's prompt is bounded by `refine_batch_size` new answers plus the
    running `existing_claims_block` accumulated so far in that RQ's own
    refine sequence -- modeled here as growing by `refine_batch_size`
    worst-case-length answers per prior batch, a safe upper bound on the
    real accumulator (claims can only merge, never exceed one per prior
    answer). Each individual answer/claim is sized at
    `SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET` chars -- the same worst-case bound
    `_estimate_synthesis_map` uses for a map call's own context, itself
    already a generous over-estimate of a real "1-2 sentence" map answer."""
    n_docs = len(full_text_parsed)
    if n_docs == 0 or not research_questions:
        return _empty_stage("synthesis_refine")

    model = synthesis_cfg.llm_model
    if not is_priced(model):
        raise UnpricedModelError(f"synthesis llm_model {model!r} has no cost_cap.pricing.PRICING entry")

    refine_batch_size = synthesis_cfg.refine_batch_size
    n_batches = math.ceil(n_docs / refine_batch_size)
    worst_case_text = "x" * SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET

    calls = 0
    total_prompt_tokens = 0
    worst_case_usd = 0.0
    for rq in research_questions:
        for batch_index in range(n_batches):
            prior_claim_count = batch_index * refine_batch_size
            batch_size_here = min(refine_batch_size, n_docs - prior_claim_count)
            existing_claims_block = "\n".join(
                f"[{i}] {worst_case_text}\n  supported by: a{i} (r{i}): {worst_case_text}"
                for i in range(prior_claim_count)
            ) or "(none yet)"
            new_answers_block = "\n".join(
                f"a{i} (r{i}): {worst_case_text}" for i in range(batch_size_here)
            )
            user_content = _SYNTHESIS_REFINE_USER_TEMPLATE.format(
                target=_target_display_text(rq), existing_claims_block=existing_claims_block,
                new_answers_block=new_answers_block,
            )
            prompt_tokens = (
                count_tokens(_SYNTHESIS_REFINE_SYSTEM_PROMPT, model=model) + count_tokens(user_content, model=model)
            )
            worst_case_usd += estimate_worst_case_cost_usd(
                model=model, prompt_tokens=prompt_tokens, max_output_tokens=None,
            )
            total_prompt_tokens += prompt_tokens
            calls += 1
    return StageCostEstimate(
        stage="synthesis_refine", model=model, calls=calls,
        total_prompt_tokens=total_prompt_tokens, worst_case_usd=worst_case_usd,
    )


def estimate_continuation_cost(
    *,
    cfg: ReviewConfig,
    retrieved_manifest_docs: Dict[str, Dict[str, str]],
    preserved_included_parent_records: List[Record],
) -> ContinuationCostEstimate:
    """Whole-pipeline worst-case cost estimate for a continuation run.
    Sources every stage's real config (`CriteriaSet`/`ScreeningConfig`/
    `ExtractionSchema`/`OpenAIExtractionConfig`/`SynthesisConfig`/
    `research_questions`) from `cfg` the identical way the real nodes do
    (`_run_full_text_eligibility()`, `extraction_node()`) -- a single `cfg`
    parameter, rather than each of those forwarded separately, so this
    estimator can never drift from what a real run would actually use.

    `retrieved_manifest_docs` is Unit A item 15's own
    `acquire_retrieved_pdfs()` return value (`Dict[str, Dict[str, str]]`,
    keyed by `record_id`, each entry `{"dest_path": str, "file_sha256":
    str}`). `preserved_included_parent_records` is Unit A item 12's own
    `acquire_preserved_included_pdfs()` return value. Both are always
    populated from their own output by the time this estimator runs,
    regardless of `--estimate-only` -- this function performs no PDF
    acquisition/copy of its own, only local, free, deterministic
    `parse_pdf()` calls against each already-acquired local path."""
    parser_options = _load_parser_options(cfg)
    retrieved_parsed, full_text_parsed = _parsed_documents(
        retrieved_manifest_docs, preserved_included_parent_records, parser_options=parser_options,
    )

    if retrieved_parsed:
        if not cfg.full_text_criteria_set:
            raise ValueError(
                "ReviewConfig.full_text_criteria_set is not set, but retrieved_manifest_docs is "
                "non-empty -- a full-text CriteriaSet must be authored explicitly before a cost "
                "estimate involving FT_SCREENING eligibility calls can be computed."
            )
        criteria_set = CriteriaSet.model_validate(cfg.full_text_criteria_set)
        screening_config = (
            FTScreeningConfig.model_validate(cfg.full_text_screening_config)
            if cfg.full_text_screening_config is not None
            else chapter_screening_config()
        )
        ft_screening_stage = _estimate_ft_screening(retrieved_parsed, criteria_set, screening_config)
    else:
        ft_screening_stage = _empty_stage("ft_screening_eligibility")

    if full_text_parsed:
        schema = verified_bundle.load_schema(cfg)
        model_config = _load_model_config(cfg)
        extraction_stage = _estimate_extraction(full_text_parsed, schema, model_config, cfg.extraction_max_batch_chars)
    else:
        extraction_stage = _empty_stage("extraction")

    synthesis_map_stage = _estimate_synthesis_map(full_text_parsed, cfg.research_questions, cfg.synthesis)
    synthesis_refine_stage = _estimate_synthesis_refine(full_text_parsed, cfg.research_questions, cfg.synthesis)

    # Evidence-map generation is deliberately NOT enumerated here:
    # `_build_evidence_map()` is pure local aggregation over
    # already-extracted/charted data -- no OpenAI client, no prompt, no
    # network call anywhere in its body. It costs nothing.
    return ContinuationCostEstimate(
        stages=(ft_screening_stage, extraction_stage, synthesis_map_stage, synthesis_refine_stage),
    )
