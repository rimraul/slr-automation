"""slr_pipeline.cost_cap.pricing -- the ONE place a dollar-per-token price
is hardcoded for the live-run cost cap (`track3_live_run_plan_2026-09-06.md`
Sec 5.1/5.3). Every price below was independently re-fetched from
`developers.openai.com/api/docs/models/<id>` twice (the plan's own Sec 5.1
draft, then again in `/code-review` round 25) and matched both times, dated
2026-09-08. A model ID not in this table is, by design, UNPRICED -- callers
must refuse the call rather than guess, mirroring the same rule the plan's
Sec 5 estimate itself was required to follow ("unpriced or unavailable
model IDs must be marked unresolved, not silently substituted").

Prices are USD per 1,000,000 tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


# Dated 2026-09-08, source developers.openai.com/api/docs/models/<id>.
# gpt-5.6-terra dropped from launch $2.50/$15.00 to $2.00/$12.00 on
# 2026-07-30 (independently corroborated via press coverage in round 25) --
# this table reflects the current, post-cut price.
PRICING: Dict[str, ModelPricing] = {
    "gpt-5.4-mini": ModelPricing(input_per_million=0.75, cached_input_per_million=0.075, output_per_million=4.50),
    "gpt-4o": ModelPricing(input_per_million=2.50, cached_input_per_million=1.25, output_per_million=10.00),
    "gpt-5.6-terra": ModelPricing(input_per_million=2.00, cached_input_per_million=0.20, output_per_million=12.00),
}

# No `max_completion_tokens`/`max_tokens` given on the request AND the model
# isn't one of the two `gpt-5.6-terra` sites that always pin 4096 (plan Sec
# 5's `ft_screening_adapter.py`/`scoping_charting_adapter.py` citations) --
# fall back to this ceiling for the pre-flight worst-case estimate rather
# than assume an unbounded response. Deliberately generous (a genuine worst
# case is more likely lower), so the estimate stays a safe upper bound.
DEFAULT_MAX_OUTPUT_TOKENS_CEILING = 4096


def is_priced(model: str) -> bool:
    return model in PRICING


def is_finite_number(value: object) -> bool:
    """True only for a genuine, non-boolean, finite int/float. Round-4
    `/code-review` finding: this exact isinstance/isfinite check was
    hand-duplicated with slightly different phrasing in three separate
    places (here, `actual_cost_usd`'s per-field loop, and `proxy.py`'s
    usage-field verification) -- a round-3 fix widening one copy's
    fallback conditions did NOT get mirrored into `proxy.py`'s own
    from-scratch max-output-tokens check, leaving the forwarded request
    unbounded for exactly the non-finite/negative/zero values the
    reservation itself already treats as invalid. One shared predicate,
    used everywhere a dollar-or-token-count value crosses a trust
    boundary, so a future widening only has to happen once.

    Round-9 `/code-review` finding, reproduced live: `math.isfinite()`
    does not just return `False` for a value it considers non-finite --
    for a Python `int` too large to convert to a C `double` (e.g. a
    syntactically valid `10**400` in an inbound JSON body's
    `max_completion_tokens`, or in an untrusted upstream response's
    `usage.prompt_tokens`), it raises `OverflowError` instead. Every
    caller of this predicate treats it as a boolean test, not something
    that can raise -- an uncaught `OverflowError` from a huge-but-valid
    JSON integer crashed the handler thread the same way NaN/inf used to
    before rounds 3-5, and worse, on the upstream-response path it meant
    a call that WAS actually billed by the real API fell through to
    `ledger.release()` instead of `ledger.commit()` in the outer
    `finally`, silently erasing real spend from the cap's running total."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def resolve_max_output_tokens(max_output_tokens: Optional[float]) -> int:
    """The single source of truth for "is this a usable output-token
    ceiling, or must we fall back to the module default": `None`, wrong
    type, non-finite (NaN/inf), or non-positive all fall back to
    `DEFAULT_MAX_OUTPUT_TOKENS_CEILING`. Used both to size the worst-case
    reservation below AND, in `proxy.py`, to decide what value must
    actually be forwarded upstream -- so the real API call is bounded to
    match the number the reservation was sized for, for every input this
    function treats as invalid, not just `None`.

    Round-3 `/code-review` finding (the reason for the isfinite check): a
    NaN/inf `max_output_tokens` makes EVERY comparison against it (`<= 0`,
    and later `estimated_usd > remaining` in the ledger) evaluate to
    False, since NaN/inf comparisons never behave as a normal caller
    would expect -- silently disabling the cap for that call.

    Round-20 `/code-review` finding: unlike `resolve_n()` (which always
    `int()`s a valid value), this function used to return a valid-but-
    fractional `max_output_tokens` (e.g. `500.7`) unchanged. `proxy.py`
    only rewrites the forwarded body's `max_output_field` when it differs
    from the resolved value, so a caller-supplied fractional value would
    compare equal to itself and be forwarded byte-for-byte to the real
    OpenAI API -- which expects an integer token count and is likely to
    reject it with a 400, turning a legitimate call into a spurious
    refusal mid live-run. Coercing to an int here makes the value both a
    valid API request and a stable fixed-point for the `needs_rewrite`
    comparison.

    Round-23 `/code-review` finding, reproduced live: round 20's fix used
    plain truncation (`int(x)`), which rounds any `x` in the open
    interval `(0, 1)` (e.g. `0.5`) DOWN to `0` -- a value that then fails
    the `<= 0` check above if fed back through this SAME function a
    second time (as `estimate_worst_case_cost_usd()` does, re-resolving
    the already-resolved value), silently falling back to
    `DEFAULT_MAX_OUTPUT_TOKENS_CEILING` for the reservation while `0`
    itself was already forwarded upstream -- a real API is documented to
    reject `max_completion_tokens=0` outright (minimum 1), a spurious
    400 for what looked like a safely-over-reserved call. Round UP
    (`math.ceil`, floored at 1) instead of truncating: this can only ever
    make the resolved value LARGER than a plain truncation would, which
    is the safe direction for a worst-case reservation (never smaller
    than what a truncating caller might have intended), guarantees the
    result is always a real API-valid positive integer, and — because
    `math.ceil()` of an already-integer value returns that same value
    unchanged — makes this function idempotent under repeated
    resolution, closing the two-call divergence at its root instead of
    special-casing zero."""
    if max_output_tokens is None or not is_finite_number(max_output_tokens) or max_output_tokens <= 0:
        return DEFAULT_MAX_OUTPUT_TOKENS_CEILING
    return max(1, math.ceil(max_output_tokens))


def resolve_n(n: Optional[float]) -> int:
    """Round-15 `/code-review` finding: OpenAI's `n` request field asks for
    `n` independent completions, each separately billed at the full output
    rate (input/prompt tokens are billed once regardless of `n`). A request
    with `n=8` costs roughly 8x the output-token component of `n=1` --
    `estimate_worst_case_cost_usd()` used to assume `n=1` unconditionally,
    so a caller-supplied `n` above 1 could exceed the reservation it was
    gated against while still passing `reserve()`. Same fallback shape as
    `resolve_max_output_tokens()`: `None`, wrong type, non-finite, or
    non-positive all fall back to the safe default of 1 (never higher --
    a bad/missing `n` must never SHRINK the worst-case estimate)."""
    if n is None or not is_finite_number(n) or n < 1:
        return 1
    return int(n)


def estimate_worst_case_cost_usd(*, model: str, prompt_tokens: int, max_output_tokens: Optional[int],
                                  n: Optional[int] = None) -> float:
    """Pre-flight cost ceiling for ONE call: full (non-cached) input price
    on `prompt_tokens` (billed once, regardless of `n`) plus output price on
    `max_output_tokens` (or the module default ceiling if the caller didn't
    cap it, or gave an unusable value), multiplied by `n` -- OpenAI bills
    each of the `n` requested completions separately at the output rate.
    Never applies the cached-input discount here -- pre-flight is a worst
    case, not an estimate of what will actually happen."""
    pricing = PRICING[model]  # raises KeyError -- caller's job to have checked is_priced() first
    out_tokens = resolve_max_output_tokens(max_output_tokens)
    resolved_n = resolve_n(n)
    # Round-12 `/code-review` finding: `max(prompt_tokens, 0)` alone does
    # NOT clamp a non-finite value -- `max(nan, 0)` returns `nan` (Python's
    # `max()` keeps its first argument whenever the comparison is False,
    # and NaN comparisons are always False) and `max(inf, 0)` returns
    # `inf` unclamped, producing a non-finite `worst_case_usd`. Every other
    # dollar-or-token-count value that crosses a trust boundary in this
    # module goes through `is_finite_number()` first (including this
    # function's own `max_output_tokens` parameter, one line above) --
    # `prompt_tokens` must not be the one exception to that rule.
    if not is_finite_number(prompt_tokens):
        prompt_tokens = 0
    prompt_tokens = max(prompt_tokens, 0)
    worst_case_usd = ((prompt_tokens / 1_000_000) * pricing.input_per_million
                      + resolved_n * (out_tokens / 1_000_000) * pricing.output_per_million)
    # Round-16 `/code-review` finding: `max_output_tokens` and `n` can each
    # individually pass `resolve_max_output_tokens()`/`resolve_n()` (both
    # finite, both positive) and still overflow the ARITHMETIC here once
    # multiplied together -- e.g. `max_output_tokens=1e300, n=1e300` sizes
    # an output-cost term far past `sys.float_info.max`, producing an
    # `inf` `worst_case_usd` despite every individual input being "valid".
    # Every other dollar-or-token value that crosses a trust boundary in
    # this module is finiteness-checked; the arithmetic RESULT must not be
    # the one exception, or a non-finite reservation size reaches
    # `ledger.reserve()` (which then raises `ValueError` -- a confusing
    # 500 "internal error" for what is really adversarial/malformed
    # caller input, not a bug in this proxy).
    if not is_finite_number(worst_case_usd):
        raise ValueError(
            f"estimate_worst_case_cost_usd() produced a non-finite worst_case_usd={worst_case_usd!r} "
            f"from max_output_tokens={max_output_tokens!r}, n={n!r} -- refusing rather than reserving "
            f"an unusable amount"
        )
    return worst_case_usd


def actual_cost_usd(*, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Real cost of a completed call, from the response's own `usage`
    field. `cached_tokens` (from `usage.prompt_tokens_details.cached_tokens`
    when the provider reports it) is billed at the cached-input rate; the
    remainder of `prompt_tokens` at the full input rate."""
    pricing = PRICING[model]
    # Round-2 `/code-review` finding: an untrusted `usage` field reporting a
    # negative token count (garbage upstream, buggy mock, future non-OpenAI
    # target) must never produce a negative dollar amount -- a "credit"
    # committed to the ledger would increase `remaining_usd()` beyond the
    # true cap, letting a LATER call bypass the spend limit entirely.
    # Clamp every count to zero before any arithmetic. Round-3 companion
    # finding: also refuse a non-finite (NaN/inf) count outright, rather
    # than let it silently poison the ledger's running total the same way
    # a non-finite `max_output_tokens` could -- there's no sane way to
    # clamp NaN, so this is the one input actual_cost_usd rejects instead
    # of sanitizing.
    for name, value in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens),
                        ("cached_tokens", cached_tokens)):
        if not is_finite_number(value):
            raise ValueError(f"actual_cost_usd() refuses non-finite/non-numeric {name}={value!r}")
    prompt_tokens = max(prompt_tokens, 0)
    completion_tokens = max(completion_tokens, 0)
    cached_tokens = max(cached_tokens, 0)
    cached_tokens = min(cached_tokens, prompt_tokens)
    uncached_tokens = prompt_tokens - cached_tokens
    return (
        (uncached_tokens / 1_000_000) * pricing.input_per_million
        + (cached_tokens / 1_000_000) * pricing.cached_input_per_million
        + (completion_tokens / 1_000_000) * pricing.output_per_million
    )
