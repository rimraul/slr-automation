"""
slr_pipeline.search_backends.base — the shared request/result contract every
real adapter in this package implements (Track 2 plan §2.4, verbatim).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.state import Record, TERMINATION_REASONS_BY_STATUS, TerminationReason


class SearchConfigError(RuntimeError):
    """Raised by `search_node` when a configured database cannot be
    dispatched at all -- an unmapped `execution_adapter_map` entry, an
    unknown adapter key, or a request/result binding mismatch (plan §2.5,
    §2.4's "Request<->result binding" bullet). Raised before any network
    call for the mapped-dispatch validation path; raised at dispatch time
    for the unmapped path."""


@dataclass(frozen=True)
class SearchBackendRequest:
    """One database query, fully self-contained. No secret is ever a field
    here -- a backend that needs an API key reads it from `os.environ` by
    the documented name (plan §2.7) inside its own `.search()`, never from
    this request, never storing it on `self`."""

    query: str
    max_results: int
    date_from: Optional[str]          # "YYYY" | None
    date_to: Optional[str]            # "YYYY" | None
    language: Optional[str]           # BCP-47 primary subtag | None
    mailto: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.max_results, int) or isinstance(self.max_results, bool):
            raise TypeError("max_results must be a plain int")
        if not (1 <= self.max_results <= 100_000):
            raise ValueError("max_results must be in [1, 100_000]")
        for f in (self.date_from, self.date_to):
            if f is not None and not re.fullmatch(r"[12][0-9]{3}", f):
                raise ValueError("date bounds must be a 4-digit year or None")
        if self.language is not None and not re.fullmatch(r"[A-Za-z]{2,3}", self.language):
            raise ValueError("language must be a 2-3 letter primary subtag or None")


class SearchBackendResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["search_backend_result_v1"] = "search_backend_result_v1"
    adapter: str
    # Free-form, like `DatabaseSearchLog.adapter_config_version` (state.py) --
    # was a closed `Literal["1"]` until review finding P1 (2026-09-24): a
    # material retrieval-semantics change (e.g. OpenAlex's title/abstract-only
    # scoping, Checkpoint 3) must be able to report a version other than "1"
    # so old and new searches are distinguishable in the persisted log.
    adapter_config_version: str = "1"
    records: List[Record]
    # The actual field-scoped expression executed against the adapter's API,
    # when it differs from `request.query` verbatim (review finding P1,
    # 2026-09-24: OpenAlex executes `title_and_abstract.search:<query>`, but
    # the search log/immutable execution spec must record what actually ran,
    # not the unscoped raw query). `None` means the adapter executed
    # `request.query` unmodified -- callers fall back to the resolved query
    # text in that case.
    effective_query: Optional[str] = None
    # "exhaustive" is used ONLY when the adapter positively proved source
    # exhaustion (Revision 5 point 1) -- never inferred merely from "the
    # request succeeded" or "the last page was short". "capped" is reserved
    # for an INTENTIONAL request-limit cap (`returned >= request.max_results`)
    # or a genuinely unknown/unverifiable total volume; a known `total_hits`
    # exceeding `returned` that the request cap does not explain is
    # `partial`/`failed` instead (review finding P1, 2026-09-24 -- the
    # PubMed 300/303 case).
    retrieval_status: Literal["exhaustive", "capped", "partial", "failed"]
    # Persisted regardless of status; see `TERMINATION_REASONS_BY_STATUS`
    # (slr_pipeline.state) for the closed set of values compatible with each
    # `retrieval_status` -- e.g. "exhaustive" can never carry
    # "max_results_cap_reached" (review finding P2, 2026-09-24).
    termination_reason: TerminationReason
    failure_detail: Optional[str] = None      # sanitised: never a URL query string, never a token
    skipped_malformed: int = Field(default=0, ge=0)
    requested: int = Field(ge=1)
    returned: int = Field(ge=0)
    total_hits: Optional[int] = None
    # Live-run finding (2026-09-22): Scopus's `view` (COMPLETE vs STANDARD)
    # governs how much abstract metadata a request actually gets back, and
    # can silently downgrade mid-run (a 401 on the institutional token drops
    # to STANDARD for the rest of the run) -- with nothing persisted, a
    # report could not tell whether a run had full institutional access or
    # a bare personal key. None for every adapter with no such distinction.
    access_mode: Optional[str] = None
    # OpenAlex-specific rate-limit provenance (2026-09-23 live-run finding):
    # a keyless/under-quota 429 carries `X-RateLimit-Remaining`/
    # `X-RateLimit-Reset` headers that say exactly how long until the daily
    # budget resets. None for every adapter with no such distinction, and
    # None even for OpenAlex on a `complete`/`partial` result -- only
    # recorded on the 429 failure path itself, since that is the only case
    # this metadata is decision-relevant for.
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset_seconds: Optional[int] = None

    @model_validator(mode="after")
    def _invariants(self) -> "SearchBackendResult":
        if self.returned != len(self.records):
            raise ValueError("returned must equal len(records)")
        if self.requested < self.returned:
            raise ValueError("returned <= requested")
        if self.total_hits is not None:
            if self.total_hits < 0:
                raise ValueError("total_hits must be non-negative when known")
            if self.total_hits < self.returned:
                raise ValueError("total_hits must be >= returned when known")
        if self.skipped_malformed > 0 and self.retrieval_status in ("exhaustive", "capped"):
            raise ValueError("skipped_malformed > 0 forces retrieval_status in {partial, failed}")
        if self.skipped_malformed > 0 and self.returned == 0 and self.retrieval_status != "failed":
            raise ValueError("skipped >= 1 and 0 usable records => 'failed' "
                             "(a partial requires >= 1 usable record)")
        if self.retrieval_status == "failed":
            if self.records or self.returned != 0:
                raise ValueError("failed => zero records, returned == 0")
        if self.retrieval_status == "partial" and self.returned < 1:
            raise ValueError("partial requires >= 1 usable retrieved record")
        if self.retrieval_status in ("exhaustive", "capped") and self.failure_detail is not None:
            raise ValueError("exhaustive/capped retrieval has no failure_detail")
        if self.retrieval_status in ("partial", "failed") and not self.failure_detail:
            raise ValueError("partial/failed retrieval MUST carry a non-empty failure_detail")
        allowed_reasons = TERMINATION_REASONS_BY_STATUS[self.retrieval_status]
        if self.termination_reason not in allowed_reasons:
            raise ValueError(
                f"termination_reason {self.termination_reason.value!r} is not compatible with "
                f"retrieval_status {self.retrieval_status!r} -- allowed: "
                f"{sorted(r.value for r in allowed_reasons)}"
            )
        if self.retrieval_status == "exhaustive" and self.total_hits is not None and self.total_hits != self.returned:
            raise ValueError("'exhaustive' requires total_hits == returned when total_hits is known")
        return self


def finalize_result(
    adapter: str,
    request: "SearchBackendRequest",
    records: List[Record],
    skipped_malformed: int,
    total_hits: Optional[int],
    failure_detail: Optional[str],
    access_mode: Optional[str] = None,
    rate_limit_remaining: Optional[int] = None,
    rate_limit_reset_seconds: Optional[int] = None,
    *,
    source_exhausted: Optional[bool] = None,
    termination_reason: Optional[str] = None,
    effective_query: Optional[str] = None,
    adapter_config_version: str = "1",
) -> SearchBackendResult:
    """Shared `retrieval_status` derivation every adapter's `.search()` ends
    with (plan §2.4, Revision 5 point 1, corrected by review finding P1,
    2026-09-24): a clean, error-free result is `exhaustive` when the caller
    passes `source_exhausted=True` (a positive proof of source exhaustion --
    e.g. `total_hits == retrieved`, or a pagination cursor's own genuine
    end-of-results signal). Otherwise, if `total_hits` is known and exceeds
    `len(records)` in a way the request's own `max_results` cap does NOT
    explain (`len(records) < request.max_results`), the shortfall is an
    UNEXPLAINED gap -- e.g. PubMed reporting `total_hits=303` but returning
    300 records against a `max_results=1000` request -- and must never be
    silently reported as `capped`; it becomes `partial` (or `failed` if zero
    records were retrieved). Only when the shortfall is explained by hitting
    the cap, or `total_hits` is unknown entirely, does `capped` apply --
    the honest default for every adapter not yet audited for real exhaustion
    proof. Any skip forces at least `partial`; an unrecovered network/parsing
    error with zero usable records is `failed`; the same error with >=1
    usable record is `partial`. `termination_reason` is always persisted;
    an unset value falls back to a generic, honest default per branch
    (`"not_yet_instrumented"` for a clean, genuinely-capped/unknown-volume
    result the caller didn't classify)."""
    if failure_detail is None and skipped_malformed == 0:
        returned = len(records)
        unexplained_gap = (
            total_hits is not None
            and returned < total_hits
            and returned < request.max_results
        )
        status: Literal["exhaustive", "capped", "partial", "failed"]
        if source_exhausted is True:
            status = "exhaustive"
            detail: Optional[str] = None
            reason = termination_reason or "not_yet_instrumented"
        elif unexplained_gap:
            status = "failed" if returned == 0 else "partial"
            detail = (
                f"unexplained retrieval gap: total_hits={total_hits} exceeds returned={returned}, "
                f"and returned did not reach the configured max_results cap ({request.max_results})"
            )
            reason = termination_reason or "unexplained_retrieval_gap"
        else:
            status = "capped"
            detail = None
            reason = termination_reason or "not_yet_instrumented"
    elif not records:
        status = "failed"
        detail = failure_detail or f"skipped {skipped_malformed} malformed source record(s)"
        reason = termination_reason or "retrieval_error"
    else:
        status = "partial"
        parts = []
        if skipped_malformed:
            parts.append(f"skipped {skipped_malformed} malformed source record(s)")
        if failure_detail:
            parts.append(failure_detail)
        detail = "; ".join(parts)
        reason = termination_reason or "partial_retrieval_with_skips_or_errors"

    return SearchBackendResult(
        adapter=adapter,
        adapter_config_version=adapter_config_version,
        records=records,
        effective_query=effective_query,
        retrieval_status=status,
        termination_reason=reason,
        failure_detail=detail,
        skipped_malformed=skipped_malformed,
        requested=request.max_results,
        returned=len(records),
        total_hits=total_hits,
        access_mode=access_mode,
        rate_limit_remaining=rate_limit_remaining,
        rate_limit_reset_seconds=rate_limit_reset_seconds,
    )


# The frozen per-adapter filter-support table (plan §2.6's "language" and
# "date params" matrix columns) -- `search_node` reads this to populate
# `DatabaseSearchLog.unsupported_filter_deviations`, not any single
# adapter's own runtime output (a static contract fact, not a live one).
LANGUAGE_UNSUPPORTED = frozenset({"openalex", "semantic_scholar", "arxiv", "crossref", "ieee", "scopus"})
DATE_UNSUPPORTED = frozenset({"arxiv"})


def record_id(source: str, native_key: str) -> str:
    """The deterministic, collision-safe, `_SAFE_ID_RE`-safe `Record.id`
    formula every real adapter uses (plan §2.6, "Deterministic Record.id"):
    `f"{source}-{sha256(native_key)[:20]}"`. `source` is already `[a-z_]`,
    the separator is `-`, the digest is 20 lowercase hex -- always matches
    `ft_screening_adapter.py`'s `_SAFE_ID_RE = ^[A-Za-z0-9_-]{1,128}$`.
    The raw `native_key` is never lost -- callers set `Record.source_id =
    native_key` verbatim alongside this id."""
    import hashlib
    digest = hashlib.sha256(native_key.encode("utf-8")).hexdigest()[:20]
    return f"{source}-{digest}"


def compute_request_fingerprint(
    adapter: str,
    request: "SearchBackendRequest",
    adapter_config_version: str,
    *,
    identity_extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Identity key for a restart-safe pagination checkpoint (Checkpoint 4,
    bullets 1-2). Covers every field that changes what results a search
    would return -- `adapter`, `adapter_config_version`, `query`,
    `max_results`, `date_from`, `date_to`, `language` -- so a changed
    configuration can never silently resume stale progress. Deliberately
    excludes `mailto`, which does not affect result identity.

    `identity_extra` lets one adapter bind additional identity-affecting
    state that has no field on `SearchBackendRequest` itself -- e.g.
    review finding 6 (2026-09-25): Scopus's requested access tier
    (COMPLETE via an institutional token vs. STANDARD) changes what
    metadata a resumed run can produce, so a checkpoint started under one
    tier must never silently resume under a different one."""
    payload = {
        "adapter": adapter,
        "adapter_config_version": adapter_config_version,
        "query": request.query,
        "max_results": request.max_results,
        "date_from": request.date_from,
        "date_to": request.date_to,
        "language": request.language,
    }
    if identity_extra:
        payload["identity_extra"] = identity_extra
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


SEARCH_CHECKPOINT_ENVELOPE_SCHEMA_VERSION = "search_page_checkpoint_envelope_v1"


@dataclass(frozen=True)
class SearchPageCheckpoint:
    """Restart-safe pagination progress for one `(review_run_id, database)`
    pair (Checkpoint 4, bullets 1-2). Saved after every successfully fetched
    page with `completed=False`; a final `completed=True` save happens ONLY
    on a clean, error-free pagination termination -- an exception path
    leaves the last per-page checkpoint untouched so a restart resumes
    pagination instead of treating a failed run as finished.

    `cursor_state` is opaque per-adapter continuation state (e.g. an
    OpenAlex/Scopus `next_cursor` string, or PubMed's `web_env`/`query_key`/
    `offset`/`have_hits` dict). `records_so_far` are `Record.to_dict()`
    payloads. `source_exhausted`/`termination_reason`/`effective_query` are
    populated only when `completed=True`, so a fully-cached short-circuit
    return can correctly rebuild a `finalize_result()` call with zero
    network calls."""

    request_fingerprint: str
    adapter: str
    adapter_config_version: str
    cursor_state: Dict[str, Any] = field(default_factory=dict)
    records_so_far: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    total_hits: Optional[int] = None
    page_count: int = 0
    completed: bool = False
    source_exhausted: Optional[bool] = None
    termination_reason: Optional[str] = None
    effective_query: Optional[str] = None

    @property
    def records_hash(self) -> str:
        """A verifiable hash of `records_so_far` ONLY -- narrower than
        `envelope_hash` below. Review finding 5 (2026-09-25): a checkpoint-
        integrity check covering only this property lets `cursor_state`,
        `total_hits`, `completed`, `termination_reason`,
        `adapter_config_version`, or `effective_query` change undetected,
        potentially skipping pages while still producing an apparently
        exhaustive result. Kept for callers that specifically need a
        records-only hash; `FileSearchCheckpointStore` uses `envelope_hash`
        for its integrity check, not this."""
        blob = json.dumps(list(self.records_so_far), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @property
    def envelope_hash(self) -> str:
        """Versioned hash over the COMPLETE checkpoint envelope -- every
        field that determines whether a resumed run's pagination state is
        trustworthy, not just `records_so_far`. Review finding 5: covers
        `cursor_state`, `total_hits`, `page_count`, `completed`,
        `source_exhausted`, `termination_reason`, `effective_query`,
        `adapter`, and `adapter_config_version` alongside `records_so_far`,
        so tampering with (or silently losing) any one of them is
        detected, not just tampering with the record payloads."""
        payload = {
            "schema_version": SEARCH_CHECKPOINT_ENVELOPE_SCHEMA_VERSION,
            "request_fingerprint": self.request_fingerprint,
            "adapter": self.adapter,
            "adapter_config_version": self.adapter_config_version,
            "cursor_state": self.cursor_state,
            "records_so_far": list(self.records_so_far),
            "total_hits": self.total_hits,
            "page_count": self.page_count,
            "completed": self.completed,
            "source_exhausted": self.source_exhausted,
            "termination_reason": self.termination_reason,
            "effective_query": self.effective_query,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class SearchCheckpointHooks:
    """Storage-agnostic injection point for restart-safe pagination
    (mirrors `screening_node`'s `client_factory` precedent -- an additive,
    keyword-only, backward-compatible parameter). `load` returns the
    checkpoint for a given `request_fingerprint`, or `None` if none exists
    yet. `save` persists a checkpoint, overwriting any prior state for the
    same fingerprint. A concrete file-backed implementation lives in
    `slr_pipeline.evaluation.update_driver.FileSearchCheckpointStore`."""

    load: Callable[[str], Optional[SearchPageCheckpoint]]
    save: Callable[[SearchPageCheckpoint], None]
