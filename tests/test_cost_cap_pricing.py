"""Unit tests for `slr_pipeline.cost_cap.pricing` -- arithmetic only, no
network. The dollar figures themselves are the ones independently
re-verified against `developers.openai.com` in
`track3_live_run_plan_2026-09-06.md` Sec 5.1 and again in round 25's
`/code-review`; this file checks the module computes correctly from them,
not that the prices are right (that's a manual, dated, external check)."""
from __future__ import annotations

import pytest

from slr_pipeline.cost_cap.pricing import (
    DEFAULT_MAX_OUTPUT_TOKENS_CEILING,
    actual_cost_usd,
    estimate_worst_case_cost_usd,
    is_finite_number,
    is_priced,
    resolve_max_output_tokens,
    resolve_n,
)


def test_is_priced_true_for_known_models():
    assert is_priced("gpt-5.4-mini")
    assert is_priced("gpt-4o")
    assert is_priced("gpt-5.6-terra")


def test_is_priced_false_for_unknown_model():
    assert not is_priced("gpt-99-not-a-real-model")
    assert not is_priced(None)


def test_estimate_worst_case_uses_explicit_max_output_tokens():
    # gpt-4o: $2.50/$10.00 per 1M in/out
    cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=1_000_000, max_output_tokens=1_000_000)
    assert cost == pytest.approx(2.50 + 10.00)


def test_estimate_worst_case_falls_back_to_default_ceiling_when_no_max_given():
    cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=None)
    expected = (DEFAULT_MAX_OUTPUT_TOKENS_CEILING / 1_000_000) * 10.00
    assert cost == pytest.approx(expected)


def test_estimate_worst_case_unpriced_model_raises_keyerror():
    with pytest.raises(KeyError):
        estimate_worst_case_cost_usd(model="not-a-real-model", prompt_tokens=100, max_output_tokens=100)


def test_actual_cost_no_caching():
    # gpt-5.4-mini: $0.75/$4.50 per 1M in/out
    cost = actual_cost_usd(model="gpt-5.4-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_tokens=0)
    assert cost == pytest.approx(0.75 + 4.50)


def test_actual_cost_with_cached_tokens_is_cheaper():
    # gpt-5.6-terra: input $2.00, cached $0.20, output $12.00 per 1M
    all_cached = actual_cost_usd(model="gpt-5.6-terra", prompt_tokens=1_000_000, completion_tokens=0,
                                  cached_tokens=1_000_000)
    none_cached = actual_cost_usd(model="gpt-5.6-terra", prompt_tokens=1_000_000, completion_tokens=0,
                                   cached_tokens=0)
    assert all_cached == pytest.approx(0.20)
    assert none_cached == pytest.approx(2.00)
    assert all_cached < none_cached


def test_actual_cost_cached_tokens_clamped_to_prompt_tokens():
    """A caller-reported `cached_tokens` larger than `prompt_tokens` (should
    never happen from a real OpenAI response, but the module must not
    silently produce a NEGATIVE uncached-token count / cost)."""
    cost = actual_cost_usd(model="gpt-4o", prompt_tokens=100, completion_tokens=0, cached_tokens=10_000)
    fully_cached_equivalent = actual_cost_usd(model="gpt-4o", prompt_tokens=100, completion_tokens=0,
                                               cached_tokens=100)
    assert cost == pytest.approx(fully_cached_equivalent)


def test_actual_cost_negative_token_counts_clamped_to_zero():
    """Round-2 `/code-review` finding: a negative count from an untrusted
    `usage` field must never produce a negative (credit) dollar amount --
    that would let a later call bypass the cap by increasing
    `remaining_usd()` beyond the true cap."""
    cost = actual_cost_usd(model="gpt-4o", prompt_tokens=100, completion_tokens=-1_000_000, cached_tokens=0)
    assert cost >= 0
    assert cost == pytest.approx(actual_cost_usd(model="gpt-4o", prompt_tokens=100, completion_tokens=0,
                                                  cached_tokens=0))


def test_actual_cost_non_finite_token_count_raises():
    """Round-3 `/code-review` finding: NaN/inf can't be clamped like a
    negative number -- must be refused outright, never silently let
    through to poison the ledger's running total."""
    with pytest.raises(ValueError):
        actual_cost_usd(model="gpt-4o", prompt_tokens=100, completion_tokens=float("nan"), cached_tokens=0)
    with pytest.raises(ValueError):
        actual_cost_usd(model="gpt-4o", prompt_tokens=float("inf"), completion_tokens=0, cached_tokens=0)


def test_estimate_worst_case_nan_max_tokens_falls_back_to_default_ceiling_not_bypassed():
    """Round-3 `/code-review` finding, directly reproduced by the reviewer:
    a NaN `max_output_tokens` makes `max_output_tokens <= 0` evaluate to
    False (NaN comparisons are always False), so the old code treated NaN
    as "a real value that happens to not need the default" and multiplied
    NaN through to a NaN worst-case cost -- which then bypassed both the
    `reserve()` gate (NaN > remaining is also always False) and the ledger's
    running total. NaN must be treated exactly like "no value given"."""
    nan_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=float("nan"))
    default_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=None)
    assert nan_cost == pytest.approx(default_cost)
    inf_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=float("inf"))
    assert inf_cost == pytest.approx(default_cost)


def test_estimate_worst_case_negative_max_tokens_falls_back_to_default_ceiling():
    negative_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=-5)
    default_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=None)
    assert negative_cost == pytest.approx(default_cost)


def test_is_finite_number_huge_int_returns_false_not_overflowerror():
    """Round-9 `/code-review` finding, reproduced live: `math.isfinite()`
    raises `OverflowError` (not `False`) for a Python `int` too large to
    convert to a C `double` -- a case distinct from NaN/inf that every
    caller of `is_finite_number()` (the max-output-token resolver in
    `proxy.py`, and the upstream `usage` field verification also in
    `proxy.py`) treats as a plain boolean test, not something that can
    raise. A syntactically valid, well-formed JSON integer like `10**400`
    used to crash the handler thread; it must be treated the same as any
    other non-finite value."""
    huge_int = 10 ** 400
    assert is_finite_number(huge_int) is False
    assert is_finite_number(-huge_int) is False


def test_estimate_worst_case_non_finite_prompt_tokens_clamped_not_propagated():
    """Round-12 `/code-review` finding: `max(prompt_tokens, 0)` alone does
    NOT clamp a non-finite value -- `max(nan, 0)` returns `nan` (Python's
    `max()` keeps its first argument whenever the comparison is False, and
    NaN comparisons are always False) and `max(inf, 0)` returns `inf`
    unclamped, producing a non-finite `worst_case_usd`. `prompt_tokens`
    must go through the same `is_finite_number()` guard as every other
    dollar-or-token-count value in this module, including this function's
    own `max_output_tokens` parameter."""
    zero_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=None)
    nan_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=float("nan"), max_output_tokens=None)
    inf_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=float("inf"), max_output_tokens=None)
    huge_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=10 ** 400, max_output_tokens=None)
    assert nan_cost == pytest.approx(zero_cost)
    assert inf_cost == pytest.approx(zero_cost)
    assert huge_cost == pytest.approx(zero_cost)


def test_estimate_worst_case_huge_int_max_tokens_falls_back_to_default_ceiling():
    """Companion to the above at the level `proxy.py` actually calls
    through: a huge-int `max_output_tokens` must fall back to the default
    ceiling exactly like NaN/inf/negative do, not crash."""
    huge_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=10 ** 400)
    default_cost = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=None)
    assert huge_cost == pytest.approx(default_cost)


def test_resolve_n_none_defaults_to_one():
    assert resolve_n(None) == 1


@pytest.mark.parametrize("bad_n", [0, -3, float("nan"), float("inf"), 10 ** 400])
def test_resolve_n_invalid_falls_back_to_one_never_lower(bad_n):
    """Round-16 `/code-review` finding: `resolve_n()` had no unit test of
    its own, unlike every one of `resolve_max_output_tokens()`'s parallel
    edge cases (NaN, inf, huge-int, negative, zero) above. A bad/missing
    `n` must fall back to 1, not something that could ever UNDER-size the
    worst-case reservation."""
    assert resolve_n(bad_n) == 1


def test_resolve_n_valid_positive_passthrough():
    assert resolve_n(4) == 4
    assert resolve_n(1) == 1


def test_estimate_worst_case_n_multiplies_only_output_cost_not_input():
    # gpt-4o: $2.50/$10.00 per 1M in/out
    n1 = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=1_000_000, max_output_tokens=1_000_000, n=1)
    n4 = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=1_000_000, max_output_tokens=1_000_000, n=4)
    assert n1 == pytest.approx(2.50 + 10.00)
    # input cost (2.50) billed once regardless of n; only the 10.00 output
    # term is multiplied by n=4.
    assert n4 == pytest.approx(2.50 + 4 * 10.00)


def test_estimate_worst_case_n_none_defaults_to_one_unchanged_from_before_the_parameter_existed():
    without_n = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=1_000_000, max_output_tokens=1_000_000)
    with_explicit_n1 = estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=1_000_000, max_output_tokens=1_000_000,
                                                     n=1)
    assert without_n == pytest.approx(with_explicit_n1)


def test_estimate_worst_case_extreme_max_output_and_n_overflow_raises_valueerror_not_inf():
    """Round-16 `/code-review` finding, reproduced live: `max_output_tokens`
    and `n` can each individually pass their own resolver's finiteness/
    positivity check and still overflow the ARITHMETIC once multiplied
    together -- e.g. both `1e300` sizes the output-cost term far past
    `sys.float_info.max`, silently producing `worst_case_usd = inf`
    instead of refusing outright. Every other dollar-or-token value that
    crosses a trust boundary in this module is finiteness-checked; the
    arithmetic RESULT must not be the one exception, or a non-finite
    reservation size reaches `ledger.reserve()` as a confusing internal
    error instead of a clear refusal at the source."""
    with pytest.raises(ValueError):
        estimate_worst_case_cost_usd(model="gpt-4o", prompt_tokens=0, max_output_tokens=1e300, n=1e300)


def test_resolve_max_output_tokens_open_interval_below_one_rounds_up_to_one_not_down_to_zero():
    """Round-23 `/code-review` finding: `int(0.5) == 0`, and `0` then
    fails this function's own `<= 0` check if fed back through it a
    second time -- exactly what `estimate_worst_case_cost_usd()` does
    internally. Rounding up avoids ever producing `0` for a value that
    passed the `> 0` gate above it."""
    assert resolve_max_output_tokens(0.5) == 1
    assert resolve_max_output_tokens(0.001) == 1


def test_resolve_max_output_tokens_idempotent_under_repeated_resolution():
    """Round-23 `/code-review` finding: `estimate_worst_case_cost_usd()`
    calls `resolve_max_output_tokens()` on an ALREADY-resolved value
    (the one `proxy.py` also forwards upstream) -- resolving that value a
    second time must be a no-op, or the reservation and the forwarded
    request silently diverge."""
    for raw in (0.5, 500.7, 4096.0, 1, 4096, DEFAULT_MAX_OUTPUT_TOKENS_CEILING):
        once = resolve_max_output_tokens(raw)
        twice = resolve_max_output_tokens(once)
        assert once == twice


def test_resolve_max_output_tokens_rounds_up_fractional_above_one():
    assert resolve_max_output_tokens(500.7) == 501
    assert resolve_max_output_tokens(500.0) == 500
