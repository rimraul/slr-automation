"""slr_pipeline.cost_cap.ledger -- the running-total half of the live-run
spend cap (`track3_live_run_plan_2026-09-06.md` Sec 5, user authorization
2026-09-08: "up to USD 30 total ... including retries ... must stop before
any request that could exceed the remaining cap"). USD 30 is a
Codex-calculated MVP-scope figure, separate from Sec 5.4's own USD 50
recommended-ceiling analysis.

`CostLedger` is deliberately reserve/commit/release, not a single
"charge()" call, because several node modules (`screening.py`) dispatch
calls concurrently via `ThreadPoolExecutor` -- without a reservation held
for the lifetime of the in-flight HTTP request, N threads could all pass a
"would this exceed the cap?" check before any of them records its actual
spend, overshooting the cap by up to N-1 calls' worth. `reserve()` is the
gate; `commit()`/`release()` resolve it once the real outcome (success or
failure) is known.

Only COMMITTED spend is persisted across process restarts -- an open
reservation is inherently mid-flight, in-process state; if the process
dies with one outstanding, the next run's ledger starts without it (the
real OpenAI call either happened, in which case it will show up on the
next OpenAI dashboard/usage check the user is separately relying on, or it
didn't, in which case there is nothing to account for). This mirrors the
"pipeline's own cap is a second, independent guard, not a substitute for"
relationship the plan's Sec 2.5 already documents for the OpenAI
project-level dashboard limit.

REQUIRED CALLER INVARIANT, not enforced by this class: exactly ONE process
may ever hold a `CostLedger` open on a given `path` at a time. This is
deliberately a launcher-level responsibility (never start a second live
run pointed at an in-progress run's ledger file), not a lock this class
takes out itself -- but the failure mode if that invariant is violated is
worse than simple double-enforcement of the cap. `_persist_locked()`
unconditionally overwrites the WHOLE `calls[]`/`total_committed_usd`
payload from only the writing process's own in-memory state, not a merge;
two processes committing against the same file can silently erase each
other's committed-spend history (whichever writes last wins), and both
would write to the identical `<path>.tmp` name, so their "atomic" writes
can corrupt each other too. Round-13 `/code-review` finding: this
docstring originally undersold the risk as "each process independently
enforces the cap, so combined spend could roughly double" -- the real risk
is silent, retroactive data loss on the ledger's own source of truth.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .pricing import is_finite_number

logger = logging.getLogger(__name__)

# Round-38 `/code-review` finding: bounds the retry-with-backoff loop
# `_persist_with_retry_locked()` runs against a transient `OSError` (the
# Windows antivirus/exclusive-handle threat vector already named
# elsewhere in this file) before giving up and marking the ledger dirty.
_PERSIST_RETRY_ATTEMPTS = 3
_PERSIST_RETRY_DELAY_S = 0.05

# Round-39 `/code-review` finding: throttles the "ledger still dirty"
# opportunistic-flush-failure warning to at most once per this many
# seconds, so a sustained outage doesn't flood the log.
_DIRTY_LOG_THROTTLE_S = 30.0


class BudgetExceededError(RuntimeError):
    """Raised by `CostLedger.reserve()` when even the WORST-CASE estimate
    of one call would push cumulative committed+reserved spend past the
    cap. The caller (the proxy) must refuse the call -- not forward it --
    when this is raised."""

    def __init__(self, *, estimated_usd: float, remaining_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"call's worst-case cost ${estimated_usd:.4f} exceeds remaining budget "
            f"${remaining_usd:.4f} (cap ${cap_usd:.4f})"
        )
        self.estimated_usd = estimated_usd
        self.remaining_usd = remaining_usd
        self.cap_usd = cap_usd


class UnpricedModelError(RuntimeError):
    """Raised when a call names a model not in `pricing.PRICING`. Mirrors
    the plan Sec 5 rule at runtime: an unpriced model is refused, never
    silently let through un-costed."""

    def __init__(self, model: str) -> None:
        super().__init__(f"model {model!r} has no entry in slr_pipeline.cost_cap.pricing.PRICING -- refusing")
        self.model = model


@dataclass
class _CallRecord:
    call_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    actual_usd: float
    ts: float
    # Round-14 `/code-review` finding: `commit()` is used both for
    # genuinely `usage`-verified calls and for the fail-safe "charge the
    # worst-case reservation because the real outcome couldn't be
    # verified" paths (unparseable/incomplete response body, ambiguous
    # network exception) -- without this flag, both kinds of record look
    # identical in shape, so this ledger's own "preserve results for
    # later use" audit trail couldn't tell a real, billed token count from
    # a worst-case guess without cross-referencing the separate call-log
    # file. Defaults to `True` (the common, normal-completion case) so
    # every existing caller keeps working unchanged.
    usage_verified: bool = True


class CostLedger:
    """Thread-safe, disk-persisted running total against `cap_usd`. One
    instance per live run; `path` is the JSON file both the running total
    and the full per-call record (the "preserve results ... for later use"
    the user asked for) live in."""

    def __init__(self, path: Path, cap_usd: float) -> None:
        # Round-10 `/code-review` finding: `cap_usd` itself was the one
        # numeric value in this module that never crossed the
        # `is_finite_number()` gate rounds 3-9 hardened everywhere else.
        # A NaN `cap_usd` made `estimated_usd > remaining` always False
        # in `reserve()` -- silently disabling the entire spend cap with
        # zero refusals ever -- and a huge-int `cap_usd` (e.g. `10**400`)
        # raised an uncaught `OverflowError` from plain float arithmetic
        # in `_remaining_locked()`, permanently breaking every request
        # the proxy ever handles instead of failing fast, once, at
        # construction time.
        if not is_finite_number(cap_usd) or cap_usd < 0:
            # Round-13 `/code-review` finding: negative was accepted by
            # the round-10 finiteness check alone -- it happened to "fail
            # safe" (every `reserve()` is immediately refused since
            # `remaining_usd()` starts negative), but silently violates
            # this class's own documented invariant that `cap_usd` is a
            # non-negative authorized spend limit, the same "fail fast
            # with a friendly message rather than a confusing downstream
            # symptom" reasoning as every other sign check in this module.
            raise ValueError(f"CostLedger requires a finite, non-negative cap_usd, got {cap_usd!r}")
        self._path = Path(path)
        self._cap_usd = cap_usd
        self._lock = threading.Lock()
        self._reserved: Dict[str, float] = {}
        self._committed_usd = 0.0
        self._calls: List[_CallRecord] = []
        # Round-38 `/code-review` finding: set whenever `_persist_locked()`
        # fails and stays stale on disk -- see `_persist_with_retry()`.
        self._persist_dirty = False
        # Round-39 `/code-review` finding: throttles repeat "still dirty"
        # warnings -- see `_should_log_dirty_locked()`.
        self._last_dirty_log_monotonic: Optional[float] = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self._path.exists():
            return
        try:
            raw_text = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # Round-36 `/code-review` finding, reproduced live: this raw
            # read had no guard against `OSError` (e.g. Windows
            # `PermissionError` from a transient antivirus scan or another
            # process briefly holding an exclusive handle on this same
            # file), unlike every other corrupted-ledger scenario a few
            # lines below -- the same "fail fast with a clear message, not
            # an uncaught internal exception" pattern rounds 10-11 already
            # established for JSON-decode/shape errors on this exact path.
            #
            # Round-37 `/code-review` finding, reproduced live: the guard
            # above only caught `OSError`, but a ledger.json with invalid
            # UTF-8 bytes (a hand-edit mistake, or disk-level bit
            # corruption -- both threat vectors this method's own
            # docstring already names) makes this same `read_text()` call
            # raise `UnicodeDecodeError` instead, which is a `ValueError`
            # subclass, not an `OSError` subclass -- so it slipped past
            # the round-36 guard uncaught. `proxy.py` already catches
            # `(UnicodeDecodeError, json.JSONDecodeError)` together for
            # the identical decode-then-parse shape; name it here too.
            raise ValueError(f"existing ledger at {self._path} could not be read: {exc}") from exc
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            # Round-11 `/code-review` finding: a syntactically-broken
            # ledger.json (truncated write outside the atomic
            # tmp-file-then-os.replace() path, hand-editing mistake, etc.)
            # used to raise a bare `json.JSONDecodeError` instead of this
            # module's own documented, friendly `ValueError` -- the same
            # "fail fast with a clear message, not an uncaught internal
            # exception" pattern round-10 just hardened for the numeric
            # fields below.
            raise ValueError(f"existing ledger at {self._path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            # Round-11 `/code-review` finding: same reasoning, for the
            # top-level JSON shape -- a ledger.json that parses but isn't a
            # JSON object (e.g. a bare list or number) must not crash the
            # `.get()` calls below with an uncaught `AttributeError`.
            raise ValueError(f"existing ledger at {self._path} must contain a JSON object, got {type(data).__name__}")
        persisted_cap = data.get("cap_usd")
        # Round-1 `/code-review` finding: exact float `!=` is fragile to
        # last-ULP JSON round-tripping differences between logically
        # identical caps (e.g. a future launcher computing cap_usd via
        # arithmetic rather than a literal) -- compare with tolerance
        # instead of spuriously refusing to resume an in-progress run.
        #
        # Round-10 `/code-review` finding, reproduced live: `float(
        # persisted_cap)` had no `is_finite_number()` guard, unlike every
        # other numeric value read across a trust boundary in this module
        # -- a corrupted/hand-edited ledger file with a huge-int
        # `cap_usd` (e.g. `10**400`) raised an uncaught `OverflowError`
        # here instead of the deliberate, friendly `ValueError` this
        # cap-mismatch check exists to provide. Check finiteness FIRST so
        # `or`'s short-circuit never reaches the unguarded `float()` call.
        if persisted_cap is None or not is_finite_number(persisted_cap) \
                or abs(float(persisted_cap) - self._cap_usd) > 1e-9:
            raise ValueError(
                f"existing ledger at {self._path} was created with cap_usd={persisted_cap!r}, "
                f"this run requested cap_usd={self._cap_usd!r} -- refusing to silently change the cap "
                f"of an in-progress run's ledger"
            )
        persisted_committed = data.get("total_committed_usd")
        # Round-12 `/code-review` finding: `is_finite_number()` accepts
        # any finite value regardless of sign -- a hand-corrupted (or
        # buggy-writer) `total_committed_usd` of e.g. `-20.0` passed this
        # check, then `_remaining_locked()` computed `cap_usd - (-20.0)`,
        # inflating the resumed run's remaining budget above the
        # authorized cap and silently defeating the whole mechanism for
        # the rest of the run. The persisted running total can never
        # legitimately be negative (every value ever added to it via
        # `commit()` is itself required to be `>= 0`), so reject it here
        # exactly like `commit()` rejects a negative `actual_usd` inline.
        if not is_finite_number(persisted_committed) or persisted_committed < 0:
            raise ValueError(
                f"existing ledger at {self._path} has an invalid total_committed_usd="
                f"{persisted_committed!r} -- refusing to resume from a corrupted ledger file"
            )
        self._committed_usd = float(persisted_committed)
        persisted_calls = data.get("calls", [])
        # Round-11 `/code-review` finding: a malformed 'calls' array (not a
        # list at all, or containing a record missing one of the required
        # keys) used to raise an uncaught TypeError/KeyError from this
        # comprehension instead of the friendly ValueError this whole
        # module's pattern exists to provide for a corrupted ledger file.
        if not isinstance(persisted_calls, list):
            raise ValueError(
                f"existing ledger at {self._path} has a non-list 'calls' field: "
                f"{type(persisted_calls).__name__}"
            )
        try:
            self._calls = [
                _CallRecord(
                    call_id=c["call_id"], model=c["model"], prompt_tokens=c["prompt_tokens"],
                    completion_tokens=c["completion_tokens"], cached_tokens=c["cached_tokens"],
                    actual_usd=c["actual_usd"], ts=c["ts"],
                    # `.get(..., True)`, not `c["usage_verified"]`: a
                    # ledger.json written before this field existed must
                    # still reload cleanly, defaulting every pre-existing
                    # record to the common "verified" case rather than
                    # refusing to resume an otherwise-valid in-progress run.
                    usage_verified=c.get("usage_verified", True),
                )
                for c in persisted_calls
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"existing ledger at {self._path} has a malformed 'calls' record: {exc}") from exc

    def _persist_locked(self) -> None:
        """Caller must already hold `self._lock`. Atomic write: tmp file +
        `os.replace()`, so a crash mid-write never leaves a truncated/
        corrupt ledger file behind."""
        payload = {
            "cap_usd": self._cap_usd,
            "total_committed_usd": self._committed_usd,
            "call_count": len(self._calls),
            "calls": [
                {
                    "call_id": c.call_id, "model": c.model, "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens, "cached_tokens": c.cached_tokens,
                    "actual_usd": c.actual_usd, "ts": c.ts, "usage_verified": c.usage_verified,
                }
                for c in self._calls
            ],
            "updated_at": time.time(),
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self._path)

    def _persist_once_locked(self) -> Optional[Exception]:
        """Caller must already hold `self._lock`. Single persist attempt,
        no retry/backoff -- see `_persist_with_retry()` for the retrying
        driver `commit()` itself uses. `TypeError`/`ValueError` (a
        non-JSON-serializable value reaching `json.dumps()`) and `OSError`
        (disk/permission problems) are all treated the same here: on
        failure, marks `self._persist_dirty = True` and returns the
        exception; on success, clears the flag and returns `None`."""
        try:
            self._persist_locked()
        except (OSError, TypeError, ValueError) as exc:
            self._persist_dirty = True
            return exc
        self._persist_dirty = False
        return None

    def _persist_with_retry(self, max_attempts: int) -> Optional[Exception]:
        """Round-38 `/code-review` finding, reproduced live: a
        `_persist_locked()` failure on the LAST `commit()` of a run
        before the process exits (or crashes) used to be swallowed and
        logged with no retry at all -- if no LATER commit()/reserve()/
        release() in that same process ever gave persistence another
        chance, that call's real, already-billed `actual_usd` silently
        never reached disk. A resumed run then reloads only the stale,
        lower on-disk `total_committed_usd`, under-counting true
        cumulative spend and letting `remaining_usd()` report more budget
        than truly remains -- exactly the overshoot this module's cap
        exists to prevent.

        Retry up to `max_attempts` times on `OSError` (the transient
        Windows antivirus/exclusive-handle lock threat vector this file's
        `_load_if_exists()` already names) with a short backoff -- most
        such locks clear within well under a second. `TypeError`/
        `ValueError` are deterministic; retrying identical input cannot
        help, so those fail immediately without burning attempts.

        Round-39 `/code-review` finding, reproduced live: an earlier
        version held `self._lock` across the WHOLE retry loop, including
        the `time.sleep()` between attempts -- against a SUSTAINED
        (non-transient) disk problem, every other `reserve()`/`commit()`/
        `release()` call in the meantime blocked on that same lock for
        the full retry budget (measured ~100ms/call, fully serializing
        5 threads x 10 calls into ~5s wall-clock instead of overlapping),
        collapsing the `ThreadPoolExecutor`-based concurrency this ledger
        exists to support into near-serial throughput for as long as the
        outage lasts. Each attempt now takes `self._lock` only for the
        single `_persist_once_locked()` call itself; the backoff sleep
        between attempts happens with the lock released, so a concurrent
        caller's own attempt (which always persists the CURRENT
        in-memory state, not a stale snapshot) can interleave freely."""
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            with self._lock:
                last_exc = self._persist_once_locked()
            if last_exc is None:
                return None
            if isinstance(last_exc, OSError) and attempt < max_attempts - 1:
                time.sleep(_PERSIST_RETRY_DELAY_S)
                continue
            break
        return last_exc

    def _flush_dirty_persist_if_any(self) -> None:
        """Must be called with `self._lock` NOT held (it acquires the lock
        itself). Round-38 `/code-review` finding: opportunistic self-heal
        -- if a previous persist left the ledger dirty, every later
        `reserve()`/`release()` call is a free chance to flush it before
        the process potentially exits, without waiting for another
        `commit()`.

        Round-39 `/code-review` finding, reproduced live: this used to run
        `_persist_with_retry_locked()`'s full multi-attempt retry loop
        INSIDE the caller's own critical section, which (a) serialized
        every `reserve()`/`release()` call for the full retry budget
        during a sustained outage (see `_persist_with_retry()`'s
        docstring) and (b) never logged a repeat failure, so once the
        ONE original `commit()` failure scrolled by, an operator running
        an unattended live run had no further signal that the on-disk
        audit trail was still stale. Fixed on both counts: a single quick
        attempt (no retry/backoff -- `commit()`'s own `_persist_with_
        retry()` call is what actually retries with backoff), and a
        throttled `logger.warning()` on failure so a sustained outage is
        still visible in the logs periodically, not just once."""
        should_log = False
        exc: Optional[Exception] = None
        with self._lock:
            if self._persist_dirty:
                exc = self._persist_once_locked()
                if exc is not None:
                    should_log = self._should_log_dirty_locked()
        if should_log:
            logger.warning(
                "CostLedger: ledger at %s is still dirty -- an earlier persist failed and this "
                "opportunistic retry just failed again, so the on-disk audit trail remains stale. "
                "The in-memory running total (the actual cap enforcement for THIS process) is still "
                "correct; only the persisted record a RESUMED run would reload from is affected.",
                self._path, exc_info=exc,
            )

    def _should_log_dirty_locked(self) -> bool:
        """Caller must already hold `self._lock`. Round-39 `/code-review`
        finding: throttles repeat "still dirty" warnings to at most one
        per `_DIRTY_LOG_THROTTLE_S` so a sustained outage doesn't flood
        the log at whatever rate `reserve()`/`release()` happen to be
        called, while still giving an operator periodic visibility beyond
        the single initial `commit()` failure log line."""
        now = time.monotonic()
        if self._last_dirty_log_monotonic is None or now - self._last_dirty_log_monotonic >= _DIRTY_LOG_THROTTLE_S:
            self._last_dirty_log_monotonic = now
            return True
        return False

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    def remaining_usd(self) -> float:
        with self._lock:
            return self._remaining_locked()

    def _remaining_locked(self) -> float:
        return self._cap_usd - self._committed_usd - sum(self._reserved.values())

    def reserve(self, call_id: str, estimated_usd: float) -> None:
        """Gate. Raises `BudgetExceededError` and reserves NOTHING if
        `estimated_usd` would push committed+reserved spend past the cap.
        Otherwise holds `estimated_usd` against the cap until `commit()`
        or `release()` resolves it."""
        if not is_finite_number(estimated_usd):
            # Round-3 `/code-review` finding: a NaN/inf `estimated_usd`
            # would make `estimated_usd > remaining` evaluate to False
            # (NaN comparisons are always False), silently letting a
            # reservation of unknown size through the gate this method
            # exists to enforce. Refuse before it ever reaches `_reserved`.
            #
            # Round-5 `/code-review` finding: this used to be a bare
            # `math.isfinite(estimated_usd)` call, which itself raises an
            # undocumented `TypeError` for a non-numeric (e.g. string)
            # input instead of this method's own documented `ValueError`
            # -- switched to the shared `pricing.is_finite_number()`
            # predicate, which returns `False` (not an exception) for any
            # non-numeric input, so the `ValueError` below is raised
            # uniformly regardless of what kind of bad value came in.
            raise ValueError(f"reserve() refuses non-finite estimated_usd={estimated_usd!r} for call_id {call_id!r}")
        if estimated_usd < 0:
            # Round-11 `/code-review` finding: `is_finite_number()` checks
            # type and finiteness, not sign -- a negative `estimated_usd`
            # would pass both that check and the `estimated_usd > remaining`
            # gate trivially, then inflate `_remaining_locked()`'s reported
            # remaining budget for every OTHER call for as long as this
            # reservation stays open. Nothing upstream currently produces a
            # negative estimate (both `pricing.py` cost functions clamp to
            # >= 0), but this is the module's own public enforcement
            # chokepoint and must not rely on callers upholding that.
            raise ValueError(f"reserve() refuses negative estimated_usd={estimated_usd!r} for call_id {call_id!r}")
        # Round-38 `/code-review` finding: opportunistic self-heal -- a
        # previous commit()'s persist failed and left the on-disk file
        # stale; every later call is a free chance to try again before
        # the process actually exits.
        #
        # Round-39 `/code-review` finding, reproduced live: this used to
        # run inside THIS method's own `with self._lock:` block below,
        # using the same multi-attempt retry-with-backoff helper `commit(
        # )` uses -- during a sustained (non-transient) disk problem that
        # serialized every `reserve()` call for the full retry budget
        # (measured ~100ms/call, collapsing 5 threads x 10 calls each
        # into ~5s of near-serial wall-clock instead of overlapping), and
        # a repeat failure here was completely silent (only the ORIGINAL
        # `commit()` failure was ever logged). `_flush_dirty_persist_if_
        # any()` takes its own separate, brief lock for a single quick
        # attempt (no retry/backoff) and logs (throttled) on failure --
        # called here, BEFORE this method's own lock, so it never nests
        # acquisitions of the same non-reentrant `threading.Lock`.
        self._flush_dirty_persist_if_any()
        with self._lock:
            remaining = self._remaining_locked()
            if estimated_usd > remaining:
                raise BudgetExceededError(estimated_usd=estimated_usd, remaining_usd=remaining, cap_usd=self._cap_usd)
            if call_id in self._reserved:
                raise ValueError(f"call_id {call_id!r} already has an open reservation")
            self._reserved[call_id] = estimated_usd

    def commit(self, call_id: str, *, model: str, prompt_tokens: int, completion_tokens: int,
               cached_tokens: int, actual_usd: float, usage_verified: bool = True) -> None:
        """The call actually completed -- move its reservation from
        reserved into committed at the REAL cost (which may be less than
        the worst-case estimate, e.g. the response used fewer than
        `max_completion_tokens`).

        `usage_verified=False` (round-14 `/code-review` finding) records
        that `prompt_tokens`/`completion_tokens`/`actual_usd` here are the
        pre-flight WORST-CASE reservation, not a real `usage` field from
        the upstream response -- the caller couldn't verify what actually
        happened (network exception after the request was sent, or an
        unparseable/incomplete response body) and is charging the safe
        upper bound instead of guessing low. Kept in the persisted record
        itself so this ledger's own audit trail can distinguish real
        telemetry from a worst-case guess without cross-referencing a
        separate call-log file."""
        if not is_finite_number(actual_usd):
            # Round-3 `/code-review` finding: a NaN/inf `actual_usd` (which
            # can only reach here if a NaN/inf worst-case estimate was
            # accepted by `reserve()` upstream -- see the matching
            # `pricing.py` finiteness guards) would poison
            # `remaining_usd()` for the rest of the run, since NaN
            # comparisons are always False and silently defeat both the
            # `estimated_usd > remaining` gate and any `<= 0` check. Refuse
            # outright rather than let a non-finite value ever enter the
            # running total.
            #
            # Round-5 `/code-review` finding: same bare-`math.isfinite`-vs
            # -shared-predicate fix as `reserve()` above, for the same
            # documented-`ValueError`-not-bare-`TypeError` reason.
            raise ValueError(f"commit() refuses non-finite actual_usd={actual_usd!r} for call_id {call_id!r}")
        if actual_usd < 0:
            # Round-11 `/code-review` finding: same reasoning as `reserve()`
            # above -- a negative `actual_usd` committed to the running
            # total would act as a "credit," increasing `remaining_usd()`
            # beyond the true cap for every later call.
            raise ValueError(f"commit() refuses negative actual_usd={actual_usd!r} for call_id {call_id!r}")
        # Round-12 `/code-review` finding: both `logger.warning()` below
        # (reservation-overshoot) and `logger.error()` (persist-failure)
        # used to execute while still holding `self._lock` -- reintroducing
        # the same "logging under lock stalls every other concurrent
        # `reserve()`/`commit()`/`release()`/`remaining_usd()` caller for
        # the duration of the log call" liveness hazard round 6 already
        # fixed for `_CallLogWriter.append()` elsewhere in this module.
        # Capture what needs logging while under the lock, release it, THEN
        # log -- the log calls themselves don't need any lock-protected
        # state, only values already computed by this point.
        overshoot_amount: Optional[float] = None
        with self._lock:
            if call_id not in self._reserved:
                raise ValueError(f"commit() called for call_id {call_id!r} with no open reservation")
            reserved_amount = self._reserved.pop(call_id)
            if actual_usd > reserved_amount:
                # Round-11 `/code-review` finding: the real cost of a
                # completed call can exceed its own worst-case reservation
                # if the pre-flight token estimate undercounted (e.g. the
                # `tiktoken` `o200k_base` fallback for a model with no
                # dedicated encoding, `proxy.py`'s prompt-token counter,
                # differs from the real tokenizer OpenAI billed against).
                # The call already happened and already cost real money --
                # `commit()` must record the true cost, not refuse or clamp
                # it -- but this is exactly the scenario where cumulative
                # committed+reserved spend can end up exceeding `cap_usd`
                # despite every individual `reserve()` call having passed
                # its own gate, so it must be visible, loudly, to whoever
                # is watching the run.
                overshoot_amount = actual_usd - reserved_amount
            self._committed_usd += actual_usd
            self._calls.append(_CallRecord(
                call_id=call_id, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cached_tokens=cached_tokens, actual_usd=actual_usd, ts=time.time(),
                usage_verified=usage_verified,
            ))
            # Round-40 `/code-review` finding, reproduced live: since
            # round 39 moved the persist-retry loop OUTSIDE this method's
            # own lock (so a sustained outage in `_persist_with_retry()`
            # doesn't hold `self._lock` across its backoff sleeps), there
            # is now a window between mutating state here and this
            # method's own first persist attempt a few lines below where
            # `self._persist_dirty` still read `False` even though the
            # on-disk file does not yet reflect this commit -- a process
            # death or scheduling delay in exactly that window left the
            # round-38/39 staleness signal blind to "committed but not
            # yet even attempted to persist," as distinct from "persist
            # attempted and failed." Mark dirty pre-emptively, in the
            # SAME lock acquisition as the mutation itself, so any
            # observer polling `snapshot()['persist_dirty']` during that
            # window sees an accurate (if briefly pessimistic) `True`;
            # `_persist_once_locked()` clears it back to `False` once the
            # first attempt actually succeeds.
            self._persist_dirty = True
        # Round-3 `/code-review` finding: the in-memory running total
        # above (the actual $30 cap enforcement) is already correct at
        # this point -- a disk failure writing the audit-trail JSON must
        # not make `commit()` raise, or the caller (the proxy) would fall
        # through to its own exception handler and RELEASE a reservation
        # that was already, correctly, committed in memory: the client
        # would see a broken connection for a call that had already
        # succeeded and already cost real money, and the call log would
        # falsely claim nothing was charged. Logging loudly and
        # continuing keeps the cap enforcement correct; the on-disk file
        # is merely stale until persistence next succeeds.
        #
        # Round-13 `/code-review` finding: originally only caught
        # `OSError`, unlike the structurally identical `json.dumps()`
        # risk in this same module's sibling `_CallLogWriter.append()`,
        # which round 5 widened to also catch `TypeError`/`ValueError`
        # (a non-JSON-serializable value reaching `json.dumps()` raises
        # `TypeError`, and some encoders raise `ValueError` for
        # out-of-range floats) -- mirrored here for the same reason:
        # `CostLedger` is a general-purpose, reusable class, not
        # something only the proxy's own already-validated values ever
        # reach. Delegated to `_persist_with_retry()` (round-38
        # `/code-review` finding) so this, the call most likely to be the
        # LAST persist attempt this process ever makes, gets a bounded
        # retry against a transient failure instead of giving up on the
        # first attempt.
        #
        # Round-39 `/code-review` finding, reproduced live: this used to
        # run the whole retry-with-backoff loop INSIDE the `with self.
        # _lock:` block above -- holding the lock across each retry's
        # `time.sleep()` fully serialized every other `reserve()`/
        # `commit()`/`release()` call for the entire retry budget during a
        # sustained (non-transient) disk problem. Moved outside the state
        # mutation's own lock: `_persist_with_retry()` now takes the lock
        # only for each individual attempt, releasing it during the
        # backoff sleep between attempts.
        # Round-43 `/code-review` finding: `_persist_with_retry()` only
        # guards `(OSError, TypeError, ValueError)` (see
        # `_persist_once_locked()`) -- an exotic exception outside that set
        # (e.g. `RecursionError`/`MemoryError` from `json.dumps()` on a
        # pathological payload) escaped uncaught from HERE, AFTER the state
        # mutation above (`_committed_usd`, `_calls`, `_reserved.pop()`) had
        # already completed, breaking this method's own documented
        # invariant just above ("a disk failure ... must not make commit()
        # raise"). A caller relying on that invariant (every pre-200
        # fail-safe handler in `proxy.py` commits via this method with no
        # surrounding try/except of its own, unlike the post-200 `finally`
        # path's belt-and-suspenders `except ValueError`) would see
        # `commit()` raise despite the ledger's own in-memory accounting
        # already being correct -- landing in its own fail-safe path,
        # RE-RELEASING an already-committed reservation and logging a
        # contradictory "no request was ever billed" audit entry for a call
        # that, per the ledger, really was billed. `self._persist_dirty` is
        # already `True` from the pre-emptive round-40 write above, so
        # nothing here needs to change it.
        try:
            persist_exc = self._persist_with_retry(_PERSIST_RETRY_ATTEMPTS)
        except Exception as exc:  # noqa: BLE001 -- deliberately broader than _persist_once_locked's own catch
            persist_exc = exc
        if overshoot_amount is not None:
            logger.warning(
                "CostLedger.commit(): call_id=%s actual cost $%.6f exceeded its own reservation of "
                "$%.6f by $%.6f -- the pre-flight worst-case estimate undercounted this call; "
                "cumulative spend may now be closer to (or over) cap_usd=$%.6f than reserve() gates alone "
                "would suggest",
                call_id, actual_usd, reserved_amount, overshoot_amount, self._cap_usd,
            )
        if persist_exc is not None:
            logger.error(
                "CostLedger.commit(): failed to persist ledger to %s after committing $%.6f for "
                "call_id=%s -- the in-memory running total is still correct and the cap is still "
                "enforced, but the on-disk audit trail is now stale until a later commit() succeeds",
                self._path, actual_usd, call_id, exc_info=persist_exc,
            )

    def release(self, call_id: str) -> None:
        """The call did NOT complete (upstream error, network failure) --
        drop its reservation without charging anything. Idempotent-safe to
        skip if the call_id was never reserved (e.g. it was refused by
        `reserve()` itself and the proxy calls `release()` defensively in
        a `finally` block)."""
        # Round-38/39 `/code-review` findings: same opportunistic self-heal
        # as `reserve()` above, and the same round-39 fix (single quick
        # attempt via a separate lock acquisition, called before this
        # method's own, rather than a multi-attempt retry loop nested
        # inside it) -- see `_flush_dirty_persist_if_any()`'s docstring.
        # `release()` doesn't itself change any persisted state, but it's
        # still a free chance to flush a previous commit()'s stale write
        # before the process potentially exits.
        self._flush_dirty_persist_if_any()
        with self._lock:
            self._reserved.pop(call_id, None)

    def ensure_persisted(self) -> None:
        """Force a persist even if this ledger has committed zero calls --
        the retry-with-backoff machinery `_persist_with_retry()` gives
        `commit()` is otherwise only ever reached from inside `commit()`'s
        own success path, so a genuinely zero-spend run (e.g. a Track 3
        continuation whose entire partition was preserved, nothing left to
        pay for) would leave no ledger file on disk at all, even though
        downstream tooling (`reference_bundle.py`'s `track3_live`
        run-evidence staging) requires the file to exist unconditionally,
        regardless of spend. Idempotent and safe to call even after real
        commits -- just re-persists the current state.

        Round-3 `/code-review` finding: an earlier version called
        `_persist_locked()` directly under `self._lock` -- bypassing
        `_persist_with_retry()`'s retry/backoff entirely (a single
        transient `OSError`, e.g. the Windows antivirus/exclusive-handle
        lock this module already names elsewhere, failed outright with no
        second attempt), never marking `self._persist_dirty` on failure
        (so `_flush_dirty_persist_if_any()`'s later opportunistic self-heal
        would never even notice anything needed re-persisting), and
        silently swallowed any failure instead of surfacing it. Unlike
        `commit()` (which must never raise on a persist failure -- the
        in-memory cap enforcement is already correct, and the call already
        happened and already cost real money), this method's entire
        purpose IS guaranteeing a file lands on disk, so a failure after
        every retry is exhausted is raised here, not merely logged."""
        try:
            persist_exc = self._persist_with_retry(_PERSIST_RETRY_ATTEMPTS)
        except Exception as exc:  # noqa: BLE001 -- mirrors commit()'s own broad catch
            persist_exc = exc
        if persist_exc is not None:
            raise RuntimeError(
                f"CostLedger.ensure_persisted(): failed to persist ledger to {self._path} after "
                f"{_PERSIST_RETRY_ATTEMPTS} attempts"
            ) from persist_exc

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cap_usd": self._cap_usd,
                "committed_usd": self._committed_usd,
                "reserved_usd": sum(self._reserved.values()),
                "remaining_usd": self._remaining_locked(),
                "call_count": len(self._calls),
                # Round-39 `/code-review` finding: without this, an
                # operator polling `snapshot()` (this class's one public
                # status method) had no way to notice the on-disk ledger
                # has been stale since an earlier persist failure -- the
                # only signal was a single log line at the moment of the
                # original failure, easy to miss in a long unattended run.
                "persist_dirty": self._persist_dirty,
            }
