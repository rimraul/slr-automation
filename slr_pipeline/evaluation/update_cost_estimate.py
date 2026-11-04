"""Workstream C Phase 0 -- offline cost estimator for the Berg post-2023
evidence-update wave (approved plan Sec 0.4). No network access, no live
search, no paid LLM call: every number here is either a direct token count
over real, already-on-disk data, an explicitly labeled assumption, or the
Track 3 MVP's coarse whole-run sanity bound.

Round-2 review established that a single ``worst_case_usd`` per stage
conflated two different questions: "what would this scenario plausibly
cost" (an assumption-priced *expected* figure) and "what is the largest
amount a hard cap must cover" (pricing the same prompt tokens against the
real/provider-default MAXIMUM output-token ceiling, via
``cost_cap.pricing.estimate_worst_case_cost_usd()``). Every ``StageCostLine``
reports both, separately.

Round-4 review found ``scenario_max_output_usd`` (formerly named
``cap_worst_case_usd``) is still NOT an unconditional hard cap: it is the
maximum-OUTPUT cost conditional on THIS SCENARIO's own workload-volume
assumptions (candidacy fraction, extraction batches/doc, resolution rate).
A genuinely workload-bound-free ceiling is computed separately by
``estimate_absolute_cost_ceiling()`` -- see its docstring for the
Checkpoint-3 enforcement requirement that figure depends on. Round-4 also
found: ``screening_qc`` (a real, always-registered graph node the draft
config enables) was missing entirely; full-text-eligibility's model/
ceiling were hardcoded rather than sourced from the real
``chapter_screening_config()``; and resolution-suggestions silently
inherited extraction's model/ceiling even though the real
``run_paid_suggestion_pass()`` receives its own model/ceiling
independently -- this module now declares its own explicit, separately
labeled assumption for those two values instead.

Per-stage calibration honesty:

- ``title_abstract_rescreen``: DIRECT token count for prompts. Real parent
  ``deduplicated_records`` (accepted only via an already-verified
  ``VerifiedReproductionArtifactSet``, never a bare list), partitioned by
  the real ``compute_current_screening_batches()``/``effective_batch_size()``,
  rendered through the real ``BATCH_SYSTEM``/``_format_batch()``, counted
  with the real ``count_tokens()``. Prompt tokens are identical across
  low/base/high; completion tokens vary by scenario's own assumed
  per-record rate.
- ``title_abstract_new_records``: ASSUMPTION for record COUNT
  (scenario-varying; ``high`` is the zero-dedup ceiling: 3 search adapters
  x ``search.max_results_per_database``). Per-record token SIZE reuses a
  real parent record at the corpus's own median (low/base) or p95 (high)
  formatted-entry-length percentile -- deterministic, not order-dependent.
- ``screening_qc_second_rater``: real ``screening_qc.enabled``/
  ``calibration_sample_size`` from the draft config bound the call count;
  the real ``compute_qc_second_rater_call_identity()`` builds the exact
  single-record prompt shape. Per-call token SIZE reuses the same
  representative-record percentile as the new-records line (the real
  ``random.sample()`` draw is still not reproducible offline even though
  ``calibration_seed`` is now pinned: no real screened/eligible-record pool
  exists yet in this offline preflight, since the update wave's own
  title/abstract screening has not run). Completion tokens are an
  assumption.
- ``full_text_eligibility``, ``extraction``, ``resolution_suggestions``:
  ASSUMPTION throughout (candidate document counts, per-call prompt-token
  budgets, resolution-required rate) -- no verified full-text/extraction
  parent and no parsed PDF exists anywhere offline for this review.
  ``full_text_eligibility``'s model/ceiling are sourced from the real
  ``chapter_screening_config()``, not a hardcoded duplicate.
  ``resolution_suggestions`` derives its call count from extraction's own
  candidate-doc count times the real extraction schema's field count times
  an explicit assumed resolution-required rate; its model/ceiling are this
  module's own declared assumption (see ``RESOLUTION_SUGGESTION_MODEL_
  ASSUMPTION``), not inherited from ``extraction_model_config``.
- ``synthesis``: not an assumption -- DISABLED. ``review_config_draft.json``
  sets every synthesis-triggering flag ``False`` (Sec 0.5's out-of-scope
  gate). If any synthesis flag is enabled this estimator refuses outright
  (``ValueError``) rather than silently reporting zero cost for a stage
  that would actually run.
- Track 3 MVP's whole-run total (~$0.2956) is cited once, separately, as a
  coarse, unlinked sanity bound -- never distributed across these stages.

Round-6 review found ``scenario_max_output_usd``/the absolute ceiling still
priced exactly ONE attempt per logical call, ignoring provider/SDK-level
retries -- a real gap, since a retried call is a separate, genuinely
billable HTTP request the caller's own code never sees. Every stage now
declares a ``max_attempts`` multiplier sourced from that stage's REAL
client-construction code, not a guess:

- ``title_abstract_rescreen``/``title_abstract_new_records`` (screening.py)
  and ``screening_qc_second_rater`` (screening_qc.py) both construct a bare
  ``OpenAI(api_key=...)`` with no ``max_retries`` override. The installed
  ``openai`` SDK's own documented default is ``max_retries=2`` (see
  ``extraction/openai_runner.py``'s docstring, which names this exact
  default while explaining why IT overrides it) -- i.e. up to 1 + 2 = 3
  real attempts per logical call. Priced at 3 attempts, not 1.
- ``full_text_eligibility`` sources its attempt count from the real
  ``chapter_screening_config().model_options["max_retries"]`` (currently 2
  -> 3 attempts), not a duplicated constant -- if that preset's retry
  policy changes, this figure tracks it automatically.
- ``extraction`` (``extraction/openai_runner.py``) and
  ``resolution_suggestions`` (``extraction/resolution_runtime.py``'s
  ``_default_client_factory()``) both explicitly pass ``max_retries=0`` in
  their real production client construction -- priced at 1 attempt, no
  multiplier needed.
- This prices TODAY's actual retry behavior, not a hoped-for future one.
  ``screening.py``/``screening_qc.py`` are not patched here to also pin
  ``max_retries=0`` -- that would be a production-node change beyond this
  offline estimator's scope; it remains a real option for whoever builds
  Checkpoint 3 to tighten this bound, alongside binding the driver's
  resolution model/ceiling/retry-policy to this module's own declared
  constants (``RESOLUTION_SUGGESTION_MODEL_ASSUMPTION``,
  ``RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION``,
  ``RESOLUTION_SUGGESTION_MAX_ATTEMPTS``) or refusing execution when they
  differ. The multiplier applies only to ``scenario_max_output_usd``
  (the cap-safe worst case): ``expected_usd`` still assumes no retry is
  needed, matching this module's existing expected-vs-cap-safe split.

Round-6 also found ``UpdateCostScenario.scenario`` typed as a bare ``str``,
excluding the real ``"absolute_ceiling"`` label from any static check. It
is now ``ScenarioLabel``, a genuine widened ``Literal`` covering all four
values this module actually returns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Literal, Optional, Sequence, Tuple

from slr_pipeline.adapters.ft_screening_adapter import chapter_screening_config
from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.cost_estimate import count_tokens
from slr_pipeline.cost_cap.pricing import (
    DEFAULT_MAX_OUTPUT_TOKENS_CEILING,
    PRICING,
    estimate_worst_case_cost_usd,
    is_priced,
)
from slr_pipeline.evaluation.bundle_artifacts import VerifiedReproductionArtifactSet
from slr_pipeline.nodes.screening import (
    BATCH_SYSTEM,
    _format_batch,
    compute_current_screening_batches,
)
from slr_pipeline.nodes.screening_qc import compute_qc_second_rater_call_identity
from slr_pipeline.state import Record

Scenario = Literal["low", "base", "high"]
ScenarioLabel = Literal["low", "base", "high", "absolute_ceiling"]
CalibrationSource = Literal["direct_token_count", "assumption", "disabled_in_config"]

# Review finding 5 (2026-09-25, second round): bumped whenever
# decide_census_vs_sampling()'s pricing/estimation logic changes in a way
# that could change a recommendation for the SAME inputs (e.g. adding QC
# cost to the gate, review finding 4) -- persisted into
# UpdateWaveCensusDecisionRecord so a stale decision computed under a prior
# estimator version is never silently trusted as still valid.
CENSUS_VS_SAMPLING_ESTIMATOR_VERSION = "census_vs_sampling_v2"

TRACK3_MVP_WHOLE_RUN_SANITY_BOUND_USD = 0.2956

# Cap-relevant worst case: 3 search adapters (pubmed/openalex/scopus) each
# independently return up to search.max_results_per_database records
# BEFORE cross-database dedup -- 3 * 1000 = 3000 is a real ceiling from the
# review_config_draft.json's own configured value, not a guess. low/base
# remain smaller, labeled assumptions about the post-dedup net-new slice.
NEW_RECORD_COUNT_ASSUMPTIONS: dict[Scenario, int] = {"low": 30, "base": 120, "high": 3000}

# No real screening response has ever been generated under this draft.
# ScreeningResponse's own fields (decision, one-sentence reason,
# relevant_topics, confidence) bound plausible completion length; these
# per-record token counts are a stated assumption, not a measurement.
COMPLETION_TOKENS_PER_RECORD_ASSUMPTIONS: dict[Scenario, int] = {"low": 60, "base": 100, "high": 160}

# No real second-rater response has ever been generated under this draft
# either; QC's response shape (a single decision, no batch) is smaller.
QC_COMPLETION_TOKENS_PER_CALL_ASSUMPTIONS: dict[Scenario, int] = {"low": 40, "base": 60, "high": 100}

# Berg's own real narrow-criteria survivor rate is 20/483 (~4.1%). The
# update wave's broader criteria could plausibly pass more; these
# per-scenario fractions of the total re-screened population (parent +
# new) are an explicit assumption, not a projection from any real decision.
FULL_TEXT_CANDIDATE_FRACTION_ASSUMPTIONS: dict[Scenario, float] = {"low": 0.02, "base": 0.05, "high": 0.12}

# No parsed full-text document exists offline for this review. Per-call
# (document, criterion) prompt-token budget, assumed directly (not derived
# from any real parse_pdf()/chunk output).
FT_ELIGIBILITY_PROMPT_TOKENS_PER_CALL_ASSUMPTIONS: dict[Scenario, int] = {"low": 1000, "base": 2500, "high": 6000}

# Worst-case assumption: every full-text-eligibility candidate proceeds to
# extraction (no full-text exclusions) -- see module docstring.
# Per-document assumed extraction-batch count and per-batch prompt-token
# budget (extraction_max_batch_chars=12000 from the real
# extraction_model_config, divided by an assumed ~4 chars/token heuristic
# -- not a real tokenizer count, since no batch text exists to tokenize).
# nodes/extraction.py has NO code-enforced maximum batches-per-document --
# a real document could in principle need more than this assumption; see
# estimate_absolute_cost_ceiling()'s docstring for the Checkpoint-3
# enforcement this figure depends on.
EXTRACTION_BATCHES_PER_DOC_ASSUMPTIONS: dict[Scenario, int] = {"low": 2, "base": 4, "high": 8}
_ASSUMED_CHARS_PER_TOKEN = 4
_EXTRACTION_MAX_BATCH_CHARS_FALLBACK = 12000
_EXTRACTION_MAX_COMPLETION_TOKENS_FALLBACK = DEFAULT_MAX_OUTPUT_TOKENS_CEILING

# Interrupted/unresolved extraction-resolution-suggestion attempts (sec.
# 3.7): derived from extraction's own worst-case candidate-doc count times
# the real extraction schema's field count (one resolution opportunity per
# extracted field), times an explicit assumed resolution-required rate --
# not a fixed small constant, so "high" scales with the same worst-case
# document volume extraction itself uses.
RESOLUTION_REQUIRED_RATE_ASSUMPTIONS: dict[Scenario, float] = {"low": 0.01, "base": 0.05, "high": 0.25}
RESOLUTION_SUGGESTION_PROMPT_TOKENS_PER_CALL_ASSUMPTIONS: dict[Scenario, int] = {"low": 1500, "base": 3000, "high": 6000}
RESOLUTION_SUGGESTION_COMPLETION_TOKENS_PER_CALL_ASSUMPTIONS: dict[Scenario, int] = {"low": 200, "base": 400, "high": 800}

# extraction/resolution_runtime.py's run_paid_suggestion_pass() takes
# `model`/`max_completion_tokens` as its OWN independent parameters -- it
# never derives them from ReviewConfig.extraction_model_config. This
# module therefore declares its own explicit assumption for both, which
# the eventual Checkpoint 3 driver's actual resolution-pass call MUST
# match for this cost line to remain valid; it is not "real" the way
# extraction's own model/ceiling (read directly from
# extraction_model_config) are.
RESOLUTION_SUGGESTION_MODEL_ASSUMPTION = "gpt-5.6-terra"
RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION = 4096

# Absolute-ceiling workload bounds (see estimate_absolute_cost_ceiling()):
# these are POLICY REQUIREMENTS the Checkpoint 3 driver must enforce for
# the resulting dollar figure to be a true hard cap, not facts already
# guaranteed by the current code.
ABSOLUTE_CEILING_FULL_TEXT_CANDIDATE_FRACTION = 1.0
ABSOLUTE_CEILING_RESOLUTION_REQUIRED_RATE = 1.0
ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE = EXTRACTION_BATCHES_PER_DOC_ASSUMPTIONS["high"]

# Round-6 review: scenario_max_output_usd/the absolute ceiling must price
# provider/SDK-level retries, since each retried attempt is a separate,
# genuinely billable HTTP request. This is the installed openai SDK's own
# documented default max_retries (see extraction/openai_runner.py's
# docstring, which names this exact value while explaining why it
# overrides it for that one call site) -- cited here, not re-derived.
_OPENAI_SDK_DEFAULT_MAX_RETRIES = 2

# nodes/screening.py and nodes/screening_qc.py both construct a bare
# OpenAI(api_key=...) with no max_retries override for their real,
# already-registered production call sites -> the SDK default applies:
# up to 1 + 2 = 3 real attempts per logical call.
SCREENING_MAX_ATTEMPTS = 1 + _OPENAI_SDK_DEFAULT_MAX_RETRIES
SCREENING_QC_MAX_ATTEMPTS = 1 + _OPENAI_SDK_DEFAULT_MAX_RETRIES

# extraction/openai_runner.py and extraction/resolution_runtime.py's
# _default_client_factory() both explicitly pass max_retries=0 in their
# real production client construction -> exactly 1 attempt, no multiplier.
EXTRACTION_MAX_ATTEMPTS = 1

# Checkpoint 3 binding contract: the driver's resolution pass MUST
# construct its client with max_retries=0 (matching resolution_runtime.py's
# own _default_client_factory() pattern) for this figure to remain valid --
# same discipline as RESOLUTION_SUGGESTION_MODEL_ASSUMPTION/
# RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION above.
RESOLUTION_SUGGESTION_MAX_ATTEMPTS = 1


class UnpricedModelError(ValueError):
    """A stage's configured model has no ``cost_cap.pricing.PRICING`` entry.
    This estimator refuses to guess a price rather than silently omit or
    zero it, mirroring ``continuation.cost_estimate``'s own rule."""


class SampleBudgetExceededError(RuntimeError):
    """Review finding 4 (2026-09-25, second round): raised by
    ``decide_census_vs_sampling()`` when census screening's retry-priced
    worst-case cost exceeds ``cap_usd`` AND the requested sample's own
    retry-priced worst-case cost ALSO exceeds it -- there is no affordable
    option to recommend, so this must be rejected here, offline, before any
    LLM call is ever made, rather than silently recommending a sample that
    could exhaust the cap mid-run."""


def _price_usd(*, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Assumption-priced *expected* cost: real per-token rates, but the
    completion side uses this stage's assumed completion-token count, not
    any provider-enforced ceiling."""
    if not is_priced(model):
        raise UnpricedModelError(f"model {model!r} has no cost_cap.pricing.PRICING entry")
    pricing = PRICING[model]
    return (
        (prompt_tokens / 1_000_000) * pricing.input_per_million
        + (completion_tokens / 1_000_000) * pricing.output_per_million
    )


def _scenario_max_output_usd(
    *, model: str, per_call_prompt_tokens: List[int], max_output_tokens: int, max_attempts: int = 1,
) -> float:
    """Maximum-output cost CONDITIONAL on the caller's own call-count/
    prompt-token workload: each call's real/assumed PROMPT tokens, priced
    with the OUTPUT side pinned to the real (or provider-default) maximum
    completion-token ceiling, via the same
    ``cost_cap.pricing.estimate_worst_case_cost_usd()`` the live cost-cap
    proxy itself uses. This is NOT an unconditional hard cap: the call
    count and prompt-token sizes it sums over still come from this
    scenario's own workload-volume assumptions (candidacy fraction,
    batches/doc, resolution rate) -- see ``estimate_absolute_cost_ceiling()``
    for a bound that does not depend on any single scenario's workload
    assumption.

    ``max_attempts`` (Round-6 review) multiplies every call's worst-case
    price by that stage's real/documented maximum provider/SDK attempt
    count (1 when the real client pins ``max_retries=0``, more otherwise)
    -- a retried attempt is a separate, genuinely billable HTTP request
    the caller's own code never sees, so a cap that ignores it is not
    actually a cap."""
    if not is_priced(model):
        raise UnpricedModelError(f"model {model!r} has no cost_cap.pricing.PRICING entry")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
    return max_attempts * sum(
        estimate_worst_case_cost_usd(model=model, prompt_tokens=t, max_output_tokens=max_output_tokens)
        for t in per_call_prompt_tokens
    )


@dataclass(frozen=True)
class StageCostLine:
    stage: str
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    expected_usd: float
    scenario_max_output_usd: float
    max_attempts: int
    calibration: CalibrationSource
    note: str


@dataclass(frozen=True)
class UpdateCostScenario:
    scenario: ScenarioLabel
    stages: Tuple[StageCostLine, ...]

    @property
    def expected_total_usd(self) -> float:
        return sum(s.expected_usd for s in self.stages)

    @property
    def scenario_max_output_total_usd(self) -> float:
        return sum(s.scenario_max_output_usd for s in self.stages)

    def render(self) -> str:
        lines = [f"Update-wave cost scenario: {self.scenario}"]
        for s in self.stages:
            lines.append(
                f"  {s.stage:<28s} model={s.model:<14s} calls={s.calls:>5d} "
                f"prompt_tokens={s.prompt_tokens:>9d} completion_tokens={s.completion_tokens:>8d} "
                f"expected_usd={s.expected_usd:>9.4f} scenario_max_output_usd={s.scenario_max_output_usd:>10.4f} "
                f"max_attempts={s.max_attempts:>2d} calibration={s.calibration}"
            )
        lines.append(f"  {'TOTAL':<28s} expected_usd={self.expected_total_usd:.4f} "
                      f"scenario_max_output_usd={self.scenario_max_output_total_usd:.4f}")
        lines.append(
            "  (scenario_max_output_usd is conditional on this scenario's own workload-volume "
            "assumptions, NOT an unconditional hard cap -- see estimate_absolute_cost_ceiling())"
        )
        lines.append(
            "  (scenario_max_output_usd already multiplies by each stage's max_attempts -- the "
            "real/documented provider/SDK maximum attempt count for that stage's real client)"
        )
        lines.append(
            f"  (sanity bound only, not a per-stage figure: Track 3 MVP whole run cost "
            f"${TRACK3_MVP_WHOLE_RUN_SANITY_BOUND_USD:.4f} for an unrelated review/config)"
        )
        return "\n".join(lines)


def _real_batch_prompt_token_list(records: List[Record], cfg: ReviewConfig, *, model: str) -> List[int]:
    """Exact per-batch prompt-token count (system + user) for real batches
    of ``records`` under ``cfg``'s real batch size/criteria/research
    questions, using the real ``BATCH_SYSTEM``/``_format_batch()``/
    ``count_tokens()``."""
    batches = compute_current_screening_batches(records=records, cfg=cfg)
    system_tokens = count_tokens(BATCH_SYSTEM, model=model)
    return [
        system_tokens + count_tokens(
            _format_batch(batch, cfg.inclusion_criteria, cfg.exclusion_criteria, cfg.research_questions),
            model=model,
        )
        for batch in batches
    ]


def _representative_record_by_percentile(records: List[Record], percentile: float) -> Record:
    """Deterministic stand-in for a hypothetical new record's title/abstract
    length: the real parent record whose OWN formatted-paper-entry length
    sits at ``percentile`` of the real corpus's length distribution
    (measured in characters, since sorting doesn't need a tokenizer call
    per candidate). Not order-dependent on the input list's position --
    only on each record's own text length."""
    def _entry_len(r: Record) -> int:
        return len(f"Title: {r.title}\n    Abstract: {(r.abstract or '')[:600]}")

    ranked = sorted(records, key=_entry_len)
    index = round(percentile * (len(ranked) - 1))
    return ranked[index]


def _estimate_title_abstract_rescreen(
    parent_records: List[Record], cfg: ReviewConfig, scenario: Scenario,
) -> StageCostLine:
    model = cfg.screening.llm_model
    per_batch_tokens = _real_batch_prompt_token_list(parent_records, cfg, model=model)
    prompt_tokens = sum(per_batch_tokens)
    completion_tokens = COMPLETION_TOKENS_PER_RECORD_ASSUMPTIONS[scenario] * len(parent_records)
    return StageCostLine(
        stage=f"title_abstract_rescreen_{len(parent_records)}_parent",
        model=model,
        calls=len(per_batch_tokens),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_usd=_price_usd(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        scenario_max_output_usd=_scenario_max_output_usd(
            model=model, per_call_prompt_tokens=per_batch_tokens,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS_CEILING, max_attempts=SCREENING_MAX_ATTEMPTS,
        ),
        max_attempts=SCREENING_MAX_ATTEMPTS,
        calibration="direct_token_count",
        note=(
            f"Real {len(parent_records)} parent deduplicated_records, real batch_size="
            f"{cfg.screening.batch_size}, real BATCH_SYSTEM/_format_batch()/count_tokens() -- "
            "prompt tokens identical across scenarios. completion_tokens/expected_usd use this "
            "scenario's assumed per-record rate; scenario_max_output_usd prices the same real "
            "prompt tokens against the provider-default max-output-token ceiling (no configured "
            f"screening completion ceiling exists), multiplied by max_attempts="
            f"{SCREENING_MAX_ATTEMPTS} (nodes/screening.py builds a bare OpenAI(...) client with "
            "no max_retries override -> the openai SDK's own documented default applies)."
        ),
    )


def _estimate_title_abstract_new_records(
    parent_records: List[Record], cfg: ReviewConfig, scenario: Scenario,
) -> StageCostLine:
    model = cfg.screening.llm_model
    new_count = NEW_RECORD_COUNT_ASSUMPTIONS[scenario]
    if new_count == 0 or not parent_records:
        return StageCostLine(
            stage="title_abstract_first_screen_new_records", model=model, calls=0,
            prompt_tokens=0, completion_tokens=0, expected_usd=0.0, scenario_max_output_usd=0.0,
            max_attempts=SCREENING_MAX_ATTEMPTS,
            calibration="assumption", note="assumed new-record count is 0 for this scenario",
        )
    percentile = 0.95 if scenario == "high" else 0.5
    representative = _representative_record_by_percentile(parent_records, percentile)
    sample = [representative] * new_count
    per_batch_tokens = _real_batch_prompt_token_list(sample, cfg, model=model)
    prompt_tokens = sum(per_batch_tokens)
    completion_tokens = COMPLETION_TOKENS_PER_RECORD_ASSUMPTIONS[scenario] * new_count
    return StageCostLine(
        stage="title_abstract_first_screen_new_records",
        model=model,
        calls=len(per_batch_tokens),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_usd=_price_usd(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        scenario_max_output_usd=_scenario_max_output_usd(
            model=model, per_call_prompt_tokens=per_batch_tokens,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS_CEILING, max_attempts=SCREENING_MAX_ATTEMPTS,
        ),
        max_attempts=SCREENING_MAX_ATTEMPTS,
        calibration="assumption",
        note=(
            f"ASSUMED {new_count} new records this scenario. low/base are a post-dedup net-new "
            "slice assumption; high is the zero-dedup ceiling (3 search adapters x "
            f"search.max_results_per_database={cfg.search.max_results_per_database} = "
            f"{3 * cfg.search.max_results_per_database}). Per-record token SIZE is the real "
            f"parent record at the {percentile:.0%} percentile of the real corpus's own "
            "formatted-entry-length distribution (median for low/base, p95 for high), not an "
            f"order-dependent sample. scenario_max_output_usd multiplied by max_attempts="
            f"{SCREENING_MAX_ATTEMPTS} (same bare OpenAI(...) client as the rescreen line)."
        ),
    )


def _estimate_screening_qc(
    parent_records: List[Record], cfg: ReviewConfig, scenario: Scenario, total_rescreened: int,
) -> StageCostLine:
    qc_cfg = cfg.screening_qc
    if not qc_cfg.enabled:
        return StageCostLine(
            stage="screening_qc_second_rater", model="(no calls)", calls=0,
            prompt_tokens=0, completion_tokens=0, expected_usd=0.0, scenario_max_output_usd=0.0,
            max_attempts=0,
            calibration="disabled_in_config",
            note="review_config_draft.json sets screening_qc.enabled=False: zero calls, not an assumption.",
        )
    model = qc_cfg.second_rater_model or cfg.screening.llm_model
    calls = min(qc_cfg.calibration_sample_size, total_rescreened)
    percentile = 0.95 if scenario == "high" else 0.5
    representative = _representative_record_by_percentile(parent_records, percentile)
    identity = compute_qc_second_rater_call_identity(cfg=cfg, record=representative, model=model)
    per_call_prompt_tokens = sum(count_tokens(m["content"], model=model) for m in identity["messages"])
    prompt_tokens = per_call_prompt_tokens * calls
    completion_tokens = QC_COMPLETION_TOKENS_PER_CALL_ASSUMPTIONS[scenario] * calls
    return StageCostLine(
        stage="screening_qc_second_rater",
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_usd=_price_usd(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        scenario_max_output_usd=_scenario_max_output_usd(
            model=model, per_call_prompt_tokens=[per_call_prompt_tokens] * calls,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS_CEILING, max_attempts=SCREENING_QC_MAX_ATTEMPTS,
        ),
        max_attempts=SCREENING_QC_MAX_ATTEMPTS,
        calibration="assumption",
        note=(
            f"Real screening_qc.enabled=True, real calibration_sample_size="
            f"{qc_cfg.calibration_sample_size} bounds calls=min(sample_size, "
            f"total_rescreened={total_rescreened})={calls}; real "
            "compute_qc_second_rater_call_identity() builds the exact single-record prompt "
            "shape and model (second_rater_model or screening.llm_model fallback). Per-call "
            "prompt token SIZE uses the same representative-record percentile as the "
            "new-records line, not the true random.sample() draw (calibration_seed is now "
            "pinned, but no real screened/eligible-record pool exists yet in this offline "
            "preflight, since the update wave's own title/abstract screening has not run, so "
            "that draw is still not reproducible offline). Completion tokens are an assumption. "
            "scenario_max_output_usd uses the provider-default ceiling (ScreeningQCConfig has "
            f"no output-ceiling field), multiplied by max_attempts={SCREENING_QC_MAX_ATTEMPTS} "
            "(nodes/screening_qc.py builds a bare OpenAI(...) client with no max_retries "
            "override -> the openai SDK's own documented default applies)."
        ),
    )


def _estimate_full_text_eligibility(
    cfg: ReviewConfig, scenario: Scenario, total_rescreened: int,
    *, candidate_fraction_override: Optional[float] = None,
) -> Tuple[StageCostLine, int]:
    ft_preset = chapter_screening_config()
    model = ft_preset.adjudicator_model
    max_completion_tokens = ft_preset.model_options["max_completion_tokens"]
    ft_max_attempts = 1 + ft_preset.model_options.get("max_retries", 0)
    n_criteria = len((cfg.full_text_criteria_set or {}).get("criteria", []))
    candidate_fraction = (
        candidate_fraction_override
        if candidate_fraction_override is not None
        else FULL_TEXT_CANDIDATE_FRACTION_ASSUMPTIONS[scenario]
    )
    candidate_docs = round(total_rescreened * candidate_fraction)
    calls = candidate_docs * n_criteria
    per_call_prompt_tokens = FT_ELIGIBILITY_PROMPT_TOKENS_PER_CALL_ASSUMPTIONS[scenario]
    prompt_tokens = calls * per_call_prompt_tokens
    completion_tokens = calls * 150  # short structured adjudication response, assumed
    line = StageCostLine(
        stage="full_text_eligibility",
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_usd=_price_usd(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if calls else 0.0,
        scenario_max_output_usd=_scenario_max_output_usd(
            model=model, per_call_prompt_tokens=[per_call_prompt_tokens] * calls,
            max_output_tokens=max_completion_tokens, max_attempts=ft_max_attempts,
        ),
        max_attempts=ft_max_attempts,
        calibration="assumption",
        note=(
            f"ASSUMED {candidate_docs} full-text candidates ({candidate_fraction:.0%} of "
            f"{total_rescreened} re-screened, vs. Berg's real 20/483~=4.1% under narrower "
            f"criteria) x {n_criteria} real full_text_criteria_set criteria; per-call "
            "prompt-token budget assumed (no parsed PDF exists offline for this review). "
            f"model={model!r}/max_completion_tokens={max_completion_tokens} are sourced from "
            "the real chapter_screening_config() preset, not a hardcoded duplicate. "
            f"scenario_max_output_usd multiplied by max_attempts={ft_max_attempts} = 1 + "
            "chapter_screening_config().model_options['max_retries'] (a real, config-declared "
            "value for the FT-eligibility adjudicator; not independently verified against the "
            "external ft_screening package's own client construction, since that package is "
            "out-of-tree)."
        ),
    )
    return line, candidate_docs


def _estimate_extraction(
    cfg: ReviewConfig, scenario: Scenario, candidate_docs: int,
    *, batches_per_doc_override: Optional[int] = None,
) -> StageCostLine:
    model = cfg.extraction_model_config.get("model") if cfg.extraction_model_config else "gpt-5.6-terra"
    max_batch_chars = cfg.extraction_max_batch_chars or _EXTRACTION_MAX_BATCH_CHARS_FALLBACK
    max_completion_tokens = (
        cfg.extraction_model_config.get("max_completion_tokens") if cfg.extraction_model_config else None
    ) or _EXTRACTION_MAX_COMPLETION_TOKENS_FALLBACK
    batches_per_doc = (
        batches_per_doc_override if batches_per_doc_override is not None
        else EXTRACTION_BATCHES_PER_DOC_ASSUMPTIONS[scenario]
    )
    calls = candidate_docs * batches_per_doc
    prompt_tokens_per_call = max_batch_chars // _ASSUMED_CHARS_PER_TOKEN
    prompt_tokens = calls * prompt_tokens_per_call
    completion_tokens = calls * 800  # structured field-extraction JSON response, assumed
    return StageCostLine(
        stage="extraction",
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_usd=_price_usd(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if calls else 0.0,
        scenario_max_output_usd=_scenario_max_output_usd(
            model=model, per_call_prompt_tokens=[prompt_tokens_per_call] * calls,
            max_output_tokens=max_completion_tokens, max_attempts=EXTRACTION_MAX_ATTEMPTS,
        ),
        max_attempts=EXTRACTION_MAX_ATTEMPTS,
        calibration="assumption",
        note=(
            f"WORST CASE: assumes all {candidate_docs} full-text-eligibility candidates proceed "
            f"to extraction (no full-text exclusions), {batches_per_doc} assumed batches/doc "
            "(nodes/extraction.py enforces no maximum batches/doc -- see "
            "estimate_absolute_cost_ceiling() for the Checkpoint-3 enforcement this bound "
            f"depends on), prompt tokens from real extraction_max_batch_chars={max_batch_chars} "
            f"/ {_ASSUMED_CHARS_PER_TOKEN} assumed chars-per-token (no batch text exists to "
            f"tokenize for real). scenario_max_output_usd uses the real configured "
            f"max_completion_tokens={max_completion_tokens}, multiplied by max_attempts="
            f"{EXTRACTION_MAX_ATTEMPTS} (extraction/openai_runner.py's real client pins "
            "max_retries=0 explicitly)."
        ),
    )


def _estimate_resolution_suggestions(
    cfg: ReviewConfig, scenario: Scenario, candidate_docs: int,
    *, resolution_rate_override: Optional[float] = None,
) -> StageCostLine:
    model = RESOLUTION_SUGGESTION_MODEL_ASSUMPTION
    max_completion_tokens = RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION
    n_fields = len(cfg.extraction_schema.get("fields", [])) if cfg.extraction_schema else 0
    attempts = candidate_docs * n_fields
    resolution_rate = (
        resolution_rate_override if resolution_rate_override is not None
        else RESOLUTION_REQUIRED_RATE_ASSUMPTIONS[scenario]
    )
    calls = round(attempts * resolution_rate)
    per_call_prompt_tokens = RESOLUTION_SUGGESTION_PROMPT_TOKENS_PER_CALL_ASSUMPTIONS[scenario]
    prompt_tokens = calls * per_call_prompt_tokens
    completion_tokens = calls * RESOLUTION_SUGGESTION_COMPLETION_TOKENS_PER_CALL_ASSUMPTIONS[scenario]
    return StageCostLine(
        stage="resolution_suggestions",
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        expected_usd=_price_usd(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if calls else 0.0,
        scenario_max_output_usd=_scenario_max_output_usd(
            model=model, per_call_prompt_tokens=[per_call_prompt_tokens] * calls,
            max_output_tokens=max_completion_tokens, max_attempts=RESOLUTION_SUGGESTION_MAX_ATTEMPTS,
        ) if calls else 0.0,
        max_attempts=RESOLUTION_SUGGESTION_MAX_ATTEMPTS if calls else 0,
        calibration="assumption",
        note=(
            f"ASSUMED {attempts} extraction attempts ({candidate_docs} worst-case extraction "
            f"candidate docs x {n_fields} real extraction_schema fields) x "
            f"{resolution_rate:.0%} assumed resolution-required rate (sec. 3.7); no real "
            f"telemetry for this review. model={model!r}/max_completion_tokens="
            f"{max_completion_tokens} are this module's OWN declared assumption -- "
            "extraction/resolution_runtime.py's run_paid_suggestion_pass() receives model/"
            "max_completion_tokens as independent parameters, never derived from "
            "extraction_model_config, so the eventual driver's actual resolution-pass call "
            "must match these values for this line to remain valid. max_attempts="
            f"{RESOLUTION_SUGGESTION_MAX_ATTEMPTS}: resolution_runtime.py's real "
            "_default_client_factory() pins max_retries=0 -- the Checkpoint 3 driver must "
            "match this or refuse execution."
        ),
    )


def _estimate_synthesis(cfg: ReviewConfig) -> StageCostLine:
    enabled_flags = [
        name for name, value in (
            ("run_bertopic", cfg.synthesis.run_bertopic),
            ("generate_evidence_map", cfg.synthesis.generate_evidence_map),
            ("generate_claim_backed_results", cfg.synthesis.generate_claim_backed_results),
            ("generate_legacy_imrad_narrative", cfg.synthesis.generate_legacy_imrad_narrative),
        ) if value
    ]
    if enabled_flags:
        raise ValueError(
            f"synthesis flags {enabled_flags} are enabled -- this estimator only prices a "
            "config with every synthesis flag disabled (plan Sec 0.5's out-of-scope gate); it "
            "refuses to silently report zero cost for a stage that would actually run"
        )
    return StageCostLine(
        stage="synthesis",
        model="(no calls)",
        calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        expected_usd=0.0,
        scenario_max_output_usd=0.0,
        max_attempts=0,
        calibration="disabled_in_config",
        note=(
            "review_config_draft.json disables every synthesis-triggering flag "
            "(run_bertopic/generate_evidence_map/generate_claim_backed_results/"
            "generate_legacy_imrad_narrative) per Sec 0.5's out-of-scope gate: zero calls, "
            "not an assumption."
        ),
    )


def estimate_update_wave_cost(
    *, parent: VerifiedReproductionArtifactSet, cfg: ReviewConfig, scenario: Scenario,
) -> UpdateCostScenario:
    """Whole-Phase-0 low/base/high cost scenario for the Berg update wave.

    ``parent`` must come from
    ``slr_pipeline.evaluation.bundle_artifacts.verify_reproduction_artifact_set()``
    -- an unverified/tampered record set can never be priced here. ``cfg``
    is the update wave's ``ReviewConfig`` (``review_config_draft.json``).
    """
    parent_records = list(parent.stage_corpus.deduplicated_records)
    rescreen_line = _estimate_title_abstract_rescreen(parent_records, cfg, scenario)
    new_records_line = _estimate_title_abstract_new_records(parent_records, cfg, scenario)
    total_rescreened = len(parent_records) + NEW_RECORD_COUNT_ASSUMPTIONS[scenario]
    qc_line = _estimate_screening_qc(parent_records, cfg, scenario, total_rescreened)
    ft_line, candidate_docs = _estimate_full_text_eligibility(cfg, scenario, total_rescreened)
    extraction_line = _estimate_extraction(cfg, scenario, candidate_docs)
    resolution_line = _estimate_resolution_suggestions(cfg, scenario, candidate_docs)
    synthesis_line = _estimate_synthesis(cfg)
    return UpdateCostScenario(
        scenario=scenario,
        stages=(
            rescreen_line, new_records_line, qc_line, ft_line, extraction_line,
            resolution_line, synthesis_line,
        ),
    )


def estimate_absolute_cost_ceiling(
    *, parent: VerifiedReproductionArtifactSet, cfg: ReviewConfig,
) -> UpdateCostScenario:
    """Workload-bound-free dollar ceiling. Unlike low/base/high (each
    conditional on that scenario's own candidacy-fraction/batches-per-doc/
    resolution-rate assumption), every downstream stage here is priced at
    ``ABSOLUTE_CEILING_FULL_TEXT_CANDIDATE_FRACTION`` = 100% candidacy,
    ``ABSOLUTE_CEILING_RESOLUTION_REQUIRED_RATE`` = 100% resolution demand,
    and ``ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE``
    batches per document, with output tokens pinned to each stage's real/
    configured ceiling throughout, and the "high" scenario's already-worst
    new-record volume (the zero-dedup 3-adapter ceiling).

    This is a TRUE dollar ceiling only if the Checkpoint 3 driver actually
    enforces the extraction per-document batch stop: nodes/extraction.py
    has no code-level maximum batches-per-document today, so
    ``ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE`` also
    states that enforcement as an explicit Checkpoint 3 requirement, not a
    fact already guaranteed by the current code. Every other bound here
    (100% candidacy, 100% resolution demand) is a real logical maximum
    (a fraction/rate cannot exceed 1.0) and needs no further enforcement.

    Round-6 review: every stage's ``scenario_max_output_usd`` here is also
    multiplied by that stage's real/documented ``max_attempts`` (provider/
    SDK retry count) -- see the module docstring and
    ``SCREENING_MAX_ATTEMPTS``/``SCREENING_QC_MAX_ATTEMPTS``/
    ``EXTRACTION_MAX_ATTEMPTS``/``RESOLUTION_SUGGESTION_MAX_ATTEMPTS`` --
    since an unpriced retry is a genuinely billable request this figure
    would otherwise silently miss.
    """
    parent_records = list(parent.stage_corpus.deduplicated_records)
    scenario: Scenario = "high"
    rescreen_line = _estimate_title_abstract_rescreen(parent_records, cfg, scenario)
    new_records_line = _estimate_title_abstract_new_records(parent_records, cfg, scenario)
    total_rescreened = len(parent_records) + NEW_RECORD_COUNT_ASSUMPTIONS[scenario]
    qc_line = _estimate_screening_qc(parent_records, cfg, scenario, total_rescreened)
    ft_line, candidate_docs = _estimate_full_text_eligibility(
        cfg, scenario, total_rescreened,
        candidate_fraction_override=ABSOLUTE_CEILING_FULL_TEXT_CANDIDATE_FRACTION,
    )
    extraction_line = _estimate_extraction(
        cfg, scenario, candidate_docs,
        batches_per_doc_override=ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE,
    )
    resolution_line = _estimate_resolution_suggestions(
        cfg, scenario, candidate_docs,
        resolution_rate_override=ABSOLUTE_CEILING_RESOLUTION_REQUIRED_RATE,
    )
    synthesis_line = _estimate_synthesis(cfg)
    return UpdateCostScenario(
        scenario="absolute_ceiling",
        stages=(
            rescreen_line, new_records_line, qc_line, ft_line, extraction_line,
            resolution_line, synthesis_line,
        ),
    )


@dataclass(frozen=True)
class CensusVsSamplingDecision:
    """Revision 5's census-vs-sampling decision gate (plan Sec 0.4
    addition): compares the real cost of screening the ENTIRE true
    deduplicated frame (census) against the real cost of screening a
    fixed-seed random subset of that SAME real frame (sample), both
    priced through the exact production batching/tokenization path --
    never an assumed per-record token size, since by this point the true
    deduplicated frame already exists (unlike ``_estimate_title_abstract_
    new_records``'s pre-search assumption). ``recommendation`` names
    which option fits ``cap_usd``; the actual sample size to run is a
    human decision made from these real numbers, never chosen
    automatically."""

    true_deduplicated_volume: int
    cap_usd: float
    census: StageCostLine
    census_qc: StageCostLine
    sample_size: int
    sample_seed: int
    sample: StageCostLine
    sample_qc: StageCostLine
    recommendation: Literal["census", "sample"]
    rationale: str

    @property
    def census_total_expected_usd(self) -> float:
        return self.census.expected_usd + self.census_qc.expected_usd

    @property
    def census_total_scenario_max_output_usd(self) -> float:
        return self.census.scenario_max_output_usd + self.census_qc.scenario_max_output_usd

    @property
    def sample_total_expected_usd(self) -> float:
        return self.sample.expected_usd + self.sample_qc.expected_usd

    @property
    def sample_total_scenario_max_output_usd(self) -> float:
        return self.sample.scenario_max_output_usd + self.sample_qc.scenario_max_output_usd

    def render(self) -> str:
        lines = [
            f"Census-vs-sampling decision: true_deduplicated_volume={self.true_deduplicated_volume}, "
            f"cap_usd={self.cap_usd:.4f}",
            f"  census title-screen: {self.census.stage} calls={self.census.calls} "
            f"scenario_max_output_usd={self.census.scenario_max_output_usd:.4f} "
            f"(expected_usd={self.census.expected_usd:.4f}, descriptive only)",
            f"  census QC: {self.census_qc.stage} calls={self.census_qc.calls} "
            f"scenario_max_output_usd={self.census_qc.scenario_max_output_usd:.4f} "
            f"(expected_usd={self.census_qc.expected_usd:.4f}, descriptive only)",
            f"  census combined worst-case: ${self.census_total_scenario_max_output_usd:.4f} "
            f"(expected ${self.census_total_expected_usd:.4f}, descriptive only)",
            f"  sample title-screen (size={self.sample_size}, seed={self.sample_seed}): "
            f"{self.sample.stage} calls={self.sample.calls} "
            f"scenario_max_output_usd={self.sample.scenario_max_output_usd:.4f} "
            f"(expected_usd={self.sample.expected_usd:.4f}, descriptive only)",
            f"  sample QC: {self.sample_qc.stage} calls={self.sample_qc.calls} "
            f"scenario_max_output_usd={self.sample_qc.scenario_max_output_usd:.4f} "
            f"(expected_usd={self.sample_qc.expected_usd:.4f}, descriptive only)",
            f"  sample combined worst-case: ${self.sample_total_scenario_max_output_usd:.4f} "
            f"(expected ${self.sample_total_expected_usd:.4f}, descriptive only)",
            f"  recommendation: {self.recommendation}",
            f"  rationale: {self.rationale}",
        ]
        return "\n".join(lines)


def draw_sample_records(
    records: Sequence[Record], *, sample_size: int, sample_seed: int,
) -> Tuple[Record, ...]:
    """The one place a census-vs-sampling sample is drawn from a real
    record list -- ``decide_census_vs_sampling()`` prices this exact draw,
    and ``run_update_wave()`` (review finding 3) screens this exact draw
    when the decision is ``"sample"``, so the priced sample and the
    screened sample can never silently diverge."""
    return tuple(random.Random(sample_seed).sample(list(records), sample_size))


def decide_census_vs_sampling(
    *,
    deduplicated_records: List[Record],
    cfg: ReviewConfig,
    cap_usd: float,
    sample_size: int,
    sample_seed: int,
    scenario: Scenario = "base",
) -> CensusVsSamplingDecision:
    """Prices screening ``deduplicated_records`` in full (census) and a
    fixed-seed ``random.Random(sample_seed).sample()`` subset of size
    ``sample_size`` drawn from that SAME real list (sample), both via
    ``_estimate_title_abstract_rescreen``'s real batching/token-counting
    path. Recommends ``"census"`` only when its retry-priced worst-case
    ceiling (``scenario_max_output_usd``) is within ``cap_usd`` (review
    finding 4: gating on ``expected_usd`` alone could recommend census
    even when the worst-case retry-priced cost would blow the cap) --
    ``expected_usd`` is reported alongside purely as descriptive
    information, never as the basis for the recommendation. Otherwise
    ``"sample"``, reporting the requested sample's real cost for
    comparison rather than silently resizing it. Both title-screening AND
    ``screening_qc`` (review finding 4: QC is a real, separately-priced,
    separately-billed call when ``cfg.screening_qc.enabled``, and omitting
    it understated the true cost of the selected workload) are priced and
    combined before gating. Raises ``SampleBudgetExceededError`` if census
    exceeds ``cap_usd`` AND the requested sample's own combined worst-case
    cost ALSO exceeds it -- there is no affordable option to recommend, so
    this is rejected here, offline, before any LLM call is ever made. Also
    raises ``SampleBudgetExceededError`` (review finding 2, 2026-09-25 third
    round) when ``sample_size=0`` and census is unaffordable: a zero-record
    sample is always priced at $0.00 and would otherwise always look
    "affordable," letting a screening run succeed by screening nothing.
    Raises ``ValueError`` if ``sample_size`` exceeds the true volume."""
    volume = len(deduplicated_records)
    if sample_size > volume:
        raise ValueError(f"sample_size={sample_size} exceeds true_deduplicated_volume={volume}")

    census_line = replace(
        _estimate_title_abstract_rescreen(deduplicated_records, cfg, scenario),
        stage=f"census_title_abstract_screen_{volume}_of_{volume}",
    )
    census_qc_line = replace(
        _estimate_screening_qc(deduplicated_records, cfg, scenario, total_rescreened=volume),
        stage=f"census_screening_qc_second_rater_{volume}_of_{volume}",
    )

    census_total_max = census_line.scenario_max_output_usd + census_qc_line.scenario_max_output_usd
    census_total_expected = census_line.expected_usd + census_qc_line.expected_usd

    if sample_size == 0:
        # Review finding 2 (2026-09-25, third round): a sample_size=0 line is
        # unconditionally priced at $0.00, so if this were built before
        # comparing against cap_usd, an unaffordable census would always be
        # "rescued" by recommending a zero-record sample -- a screening run
        # that succeeds by screening nothing. Only legitimate when census
        # itself is affordable (sample_size=0 there is a genuine no-sample-
        # needed informational line); refuse outright when census is not.
        if census_total_max > cap_usd:
            raise SampleBudgetExceededError(
                f"census combined worst-case cost ${census_total_max:.4f} exceeds cap ${cap_usd:.4f}, and "
                "sample_size=0 requests zero records -- a zero-record sample can never be a meaningful "
                "affordable fallback. Raise cap_usd, choose a nonzero sample_size, or disable "
                "screening_qc before proceeding. No LLM call has been made."
            )
        sample_line = StageCostLine(
            stage=f"sample_title_abstract_screen_0_of_{volume}", model=cfg.screening.llm_model, calls=0,
            prompt_tokens=0, completion_tokens=0, expected_usd=0.0, scenario_max_output_usd=0.0,
            max_attempts=SCREENING_MAX_ATTEMPTS, calibration="direct_token_count",
            note="sample_size=0 -- no records selected",
        )
        sample_qc_line = StageCostLine(
            stage=f"sample_screening_qc_second_rater_0_of_{volume}", model="(no calls)", calls=0,
            prompt_tokens=0, completion_tokens=0, expected_usd=0.0, scenario_max_output_usd=0.0,
            max_attempts=0, calibration="direct_token_count",
            note="sample_size=0 -- no records selected, no QC calls",
        )
    else:
        sample_records = draw_sample_records(deduplicated_records, sample_size=sample_size, sample_seed=sample_seed)
        sample_line = replace(
            _estimate_title_abstract_rescreen(sample_records, cfg, scenario),
            stage=f"sample_title_abstract_screen_{sample_size}_of_{volume}",
        )
        sample_qc_line = replace(
            _estimate_screening_qc(sample_records, cfg, scenario, total_rescreened=sample_size),
            stage=f"sample_screening_qc_second_rater_{sample_size}_of_{volume}",
        )

    sample_total_max = sample_line.scenario_max_output_usd + sample_qc_line.scenario_max_output_usd
    sample_total_expected = sample_line.expected_usd + sample_qc_line.expected_usd

    if census_total_max <= cap_usd:
        recommendation: Literal["census", "sample"] = "census"
        rationale = (
            f"census combined worst-case retry-priced cost (title ${census_line.scenario_max_output_usd:.4f} "
            f"+ QC ${census_qc_line.scenario_max_output_usd:.4f} = ${census_total_max:.4f}) is within cap "
            f"${cap_usd:.4f} (expected cost ${census_total_expected:.4f}, descriptive only) -- screen the "
            "full true deduplicated frame, no sampling needed."
        )
    else:
        if sample_total_max > cap_usd:
            raise SampleBudgetExceededError(
                f"census combined worst-case cost ${census_total_max:.4f} exceeds cap ${cap_usd:.4f}, and "
                f"the requested sample (size={sample_size}, seed={sample_seed}) combined worst-case cost "
                f"(title ${sample_line.scenario_max_output_usd:.4f} + QC "
                f"${sample_qc_line.scenario_max_output_usd:.4f} = ${sample_total_max:.4f}) ALSO exceeds it "
                "-- no affordable option exists. Raise cap_usd, lower sample_size, or disable "
                "screening_qc before proceeding. No LLM call has been made."
            )
        recommendation = "sample"
        rationale = (
            f"census combined worst-case retry-priced cost (title ${census_line.scenario_max_output_usd:.4f} "
            f"+ QC ${census_qc_line.scenario_max_output_usd:.4f} = ${census_total_max:.4f}) exceeds cap "
            f"${cap_usd:.4f} (expected cost ${census_total_expected:.4f}, descriptive only); a seed="
            f"{sample_seed} sample of size={sample_size} combined worst-case cost (title "
            f"${sample_line.scenario_max_output_usd:.4f} + QC ${sample_qc_line.scenario_max_output_usd:.4f} "
            f"= ${sample_total_max:.4f}, expected ${sample_total_expected:.4f}) fits the cap -- the final "
            "sample size is a human decision made against these real numbers, not chosen automatically here."
        )

    return CensusVsSamplingDecision(
        true_deduplicated_volume=volume, cap_usd=cap_usd, census=census_line, census_qc=census_qc_line,
        sample_size=sample_size, sample_seed=sample_seed, sample=sample_line, sample_qc=sample_qc_line,
        recommendation=recommendation, rationale=rationale,
    )


__all__ = [
    "Scenario",
    "ScenarioLabel",
    "CalibrationSource",
    "StageCostLine",
    "UpdateCostScenario",
    "UnpricedModelError",
    "SampleBudgetExceededError",
    "CENSUS_VS_SAMPLING_ESTIMATOR_VERSION",
    "TRACK3_MVP_WHOLE_RUN_SANITY_BOUND_USD",
    "NEW_RECORD_COUNT_ASSUMPTIONS",
    "ABSOLUTE_CEILING_FULL_TEXT_CANDIDATE_FRACTION",
    "ABSOLUTE_CEILING_RESOLUTION_REQUIRED_RATE",
    "ABSOLUTE_CEILING_EXTRACTION_BATCHES_PER_DOC_DRIVER_MUST_ENFORCE",
    "RESOLUTION_SUGGESTION_MODEL_ASSUMPTION",
    "RESOLUTION_SUGGESTION_MAX_COMPLETION_TOKENS_ASSUMPTION",
    "SCREENING_MAX_ATTEMPTS",
    "SCREENING_QC_MAX_ATTEMPTS",
    "EXTRACTION_MAX_ATTEMPTS",
    "RESOLUTION_SUGGESTION_MAX_ATTEMPTS",
    "estimate_update_wave_cost",
    "estimate_absolute_cost_ceiling",
    "CensusVsSamplingDecision",
    "decide_census_vs_sampling",
    "draw_sample_records",
]
