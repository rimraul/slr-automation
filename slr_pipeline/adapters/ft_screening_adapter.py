"""slr_pipeline <-> FT_SCREENING (full-text eligibility screening) adapter.

Phase A scope (offline adapter only -- see the review that scoped this):
this module maps slr_pipeline's own models to/from FT_SCREENING's public
request/result models and builds the chapter's calibrated pilot LLM
config. It does NOT execute a screening run (no `resolve_components()`/
`run_ft_screening()` call lives here), does NOT touch `ReviewState`, and
does NOT wire into any graph or node -- that is Phase B, which only
starts once this module's own review gate closes.

Imports ONLY the public FT_SCREENING surface (`ft_screening`/
`ft_screening.api`) -- never `ft_screening.orchestrator`,
`ft_screening.document`, `ft_screening.fakes`, or any other internal
module. Tests for this module are free to reach into FT_SCREENING's
private testing surface (fakes + `_run_ft_screening`) to exercise the
pure functions here end-to-end without a real LLM; that is a property of
the TEST file, not of this production module.

CriteriaSet is never derived from `Protocol.inclusion_criteria`/
`exclusion_criteria`. Those are free-text, title/abstract-screening-era
strings from `slr_pipeline.state.Protocol` -- they are not phrased as
per-criterion eligibility requirements, and have no id/version/missing-
evidence-policy/exclusion-code/priority/allowed-sources of their own.
There is deliberately no function anywhere in this module that reads
`Protocol` or `ReviewConfig.inclusion_criteria`/`exclusion_criteria` --
a full-text `CriteriaSet` must always be authored explicitly, as a real
`ft_screening.CriteriaSet` of real `ft_screening.CriterionSpec` objects,
by whoever configures a review. `CriteriaSet`/`CriterionSpec` are
re-exported from here purely so a caller doesn't need a second import
line to build one.

METHODOLOGICAL CAVEAT (see `chapter_screening_config()`): the model
configuration this module provides is the ONE calibrated pilot
configuration this book's own examples were run and validated under. It
is a validated starting point for reproducing the book's examples, not a
production-accuracy guarantee for an arbitrary reader's own review topic
-- exactly the same caveat FT_SCREENING's own Arm 2b carries for its
retrospectively-curated evidence packages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ft_screening import (
    ComponentResolutionError,
    CriteriaSet,
    CriterionSpec,
    RunArtifacts,
    ScreeningConfig,
    ScreeningItem,
    ScreeningItemResult,
    ScreeningRequest,
    ScreeningRunResult,
    build_supported_screening_config,
)

from slr_pipeline.state import Record

__all__ = [
    "CHAPTER_MODEL_CONFIG_ID",
    "ChapterAdapterError",
    "ChapterFullTextOutcome",
    "ComponentResolutionError",
    "CriteriaSet",
    "CriterionSpec",
    "build_full_text_screening_request",
    "chapter_screening_config",
    "map_run_result_to_chapter_outcomes",
    "record_to_screening_item",
]

# Versioned so a future change to the chapter's teaching preset (a
# different model/effort/token-budget triple) is itself a new,
# distinguishable model_config_id -- never a silent redefinition of what
# "v1" meant, matching build_supported_screening_config()'s own
# model_config_id contract.
CHAPTER_MODEL_CONFIG_ID = "slr-chapter-teaching-preset-v1"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ChapterAdapterError(ValueError):
    """A Record/PDF pairing cannot be mapped into a valid FT_SCREENING
    request -- currently: an unsafe or empty `Record.id`. Distinct from
    FT_SCREENING's own `pydantic.ValidationError` (malformed
    ScreeningItem/CriteriaSet/etc. field values) and
    `ComponentResolutionError` (bad ScreeningConfig identity/options),
    both of which still propagate unwrapped from this module -- this
    class covers only adapter-level preconditions FT_SCREENING itself has
    no field to reject."""


def chapter_screening_config() -> ScreeningConfig:
    """The chapter's ONE calibrated pilot LLM configuration: `gpt-5.6-terra`,
    `reasoning_effort="medium"`, `max_completion_tokens=4096` -- the exact
    triple FT_SCREENING's own calibration (`terra_calibration_v4.py`) and
    matched Arm 1/Arm 2 runs validated, not an arbitrary example value.
    Takes no parameters on purpose: every value is fixed and named, so a
    reader following the book's examples reproduces the calibrated
    configuration exactly. A different configuration is a different,
    explicitly-named preset, not an optional argument to this one.

    Locked component identities/versions still come from
    `build_supported_screening_config()` itself (docling/lexical_bm25/
    openai, live version constants) -- this function adds nothing beyond
    naming the model/effort/token-budget/model_config_id.
    """
    return build_supported_screening_config(
        adjudicator_model="gpt-5.6-terra",
        model_config_id=CHAPTER_MODEL_CONFIG_ID,
        model_options={"max_completion_tokens": 4096, "reasoning_effort": "medium"},
    )


def _safe_document_id(record_id: str) -> str:
    if not _SAFE_ID_RE.match(record_id):
        raise ChapterAdapterError(
            f"Record.id {record_id!r} is not a safe identifier token "
            "([A-Za-z0-9_-], 1-128 chars, no path separators/dots) -- refusing to use it as an "
            "FT_SCREENING document_id/report_id"
        )
    return record_id


def record_to_screening_item(record: Record, *, pdf_path: Path) -> ScreeningItem:
    """Deterministic `Record` + a CONFIRMED `pdf_path` -> one FT_SCREENING
    `ScreeningItem`. Pure: no I/O, no PDF existence/content check beyond
    what `ScreeningItem` itself validates, no acquisition logic --
    `pdf_path` must already point at a real, locally confirmed PDF; Phase
    B's node is responsible for running its own OA-download-cascade/stub-
    detection acquisition layer BEFORE calling this, never after.

    `document_id == report_id == record.id` (validated as a safe token,
    raising `ChapterAdapterError` otherwise): this integration sends
    exactly one document per record, and `report_id` is FT_SCREENING's
    own stable join key back to the originating record (see
    api_contract_v1.md's explicit slr_pipeline mapping note) -- so the two
    naturally coincide here rather than needing a separate identifier
    scheme. Two records sharing the same (unsafe-token-valid) `id` is a
    slr_pipeline-side bug (duplicate/undeduplicated records reaching this
    stage), and surfaces as `ScreeningRequest`'s own
    `pydantic.ValidationError` for duplicate item identifiers when such
    items are batched together -- deliberately not pre-checked or
    silently disambiguated here.

    `citation` carries only display/audit metadata (never sent to the
    adjudication prompt -- see FT_SCREENING's own evidence-redaction
    design): doi, title, source, source_id, year. `None`-valued fields are
    omitted rather than sent as JSON `null`.
    """
    document_id = _safe_document_id(record.id)
    citation = {
        "record_id": record.id,
        "doi": record.doi,
        "title": record.title,
        "source": record.source,
        "source_id": record.source_id,
        "year": record.year,
    }
    citation = {k: v for k, v in citation.items() if v is not None}
    return ScreeningItem(
        document_id=document_id,
        report_id=document_id,
        pdf_path=pdf_path,
        citation=citation,
    )


def build_full_text_screening_request(
    *,
    items: list[ScreeningItem],
    criteria_set: CriteriaSet,
    config: ScreeningConfig,
    output_dir: Path,
    run_id: str | None = None,
    resume_run_id: str | None = None,
) -> ScreeningRequest:
    """Explicit, pure assembly of a `ScreeningRequest` -- `criteria_set`
    MUST be a real, hand-authored FT_SCREENING `CriteriaSet` (see this
    module's own docstring); this function never constructs, defaults, or
    derives one. All validation (unique item identifiers, criteria-set
    invariants, at-most-one of run_id/resume_run_id, ...) is FT_SCREENING's
    own -- this function adds none of its own beyond field assembly.
    `resume_run_id` exists so a caller can re-enter an interrupted
    FT_SCREENING run (see `nodes.fulltext._resolve_run_identity()`) rather
    than always starting fresh."""
    return ScreeningRequest(
        run_id=run_id, resume_run_id=resume_run_id, items=items, criteria_set=criteria_set,
        config=config, output_dir=output_dir,
    )


class ChapterFullTextOutcome(BaseModel):
    """One record's full-text eligibility outcome -- carries the COMPLETE
    `ScreeningItemResult` FT_SCREENING produced for it, not a stripped
    summary (codex review of the first Phase A draft, finding 1: a
    fields-only summary discarded criterion-level status/evidence/
    rationale/requires_human_review, processing-error detail, document
    coverage, and execution provenance -- exactly the information a human
    reviewer needs to understand WHY a record needs review or failed, not
    just THAT it does). `item_result.assessments`/`.errors`/`.coverage`/
    `.provenance` are all still there for Phase B/a human reviewer to read
    directly; `run_artifacts` is the path back to the complete, immutable,
    authoritative frozen run files (`run_result.json`/`item_results.jsonl`/
    `ledger.jsonl`) those came from -- this model REFERENCES that artifact,
    it never copies or simplifies its contents (Phase B's own artifact-
    immutability requirement).

    `record_id`/`decision`/`execution_status`/`primary_exclusion_reason_code`/
    `exclusion_reason_codes`/`ft_screening_result_id` are plain `@property`
    values DERIVED LIVE from `item_result`, not independently stored
    fields -- codex review finding 2: a plain stored `decision` field could
    be constructed inconsistently with `execution_status`/exclusion codes
    (e.g. `decision="include"` alongside `execution_status="processing_failed"`),
    even though the mapping function itself never produced such a
    combination -- the public model still allowed deserializing one into
    graph state. Deriving these fields from `item_result` on every access
    makes that class of state STRUCTURALLY unreachable rather than merely
    validated against: there is no field to independently set, so there is
    nothing to be inconsistent. This also means FT_SCREENING's own
    invariants (`eligibility` present iff `execution_status == "completed"`;
    `exclusion_reason_codes`/`primary_exclusion_reason_code` populated only
    when `decision == "exclude"`; primary always a member of
    `exclusion_reason_codes`) are inherited automatically, not re-checked
    -- `item_result`/`EligibilityResult`/`aggregation.py` already guarantee
    every one of them upstream.

    Pydantic (not a plain dataclass) to match this codebase's own
    JSON-serializable-state convention (`Record`, `PRISMACounts`, ...) --
    Phase B can `.model_dump()` this directly into state.

    The six derived values below are ordinary `@property` methods, NOT
    `@computed_field`s (codex review finding, round 3: `@computed_field`
    IS included in `model_dump()`, but is NOT a constructor field, so
    `ChapterFullTextOutcome.model_validate(outcome.model_dump())` failed
    with 6 validation errors under `extra="forbid"` -- a persisted
    outcome could be written into `ReviewState` but never reconstructed
    from it again, exactly the resume/artifact-loading path Phase B
    needs). Plain `@property` is never touched by `model_dump()`/
    `model_validate()` at all -- only `ft_screening_run_id`/
    `run_artifacts`/`item_result` are ever serialized, and every derived
    value is recomputed fresh from `item_result` after reconstruction,
    so there is nothing to keep in sync and no duplicated state in
    `ReviewState` either.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ft_screening_run_id: str
    run_artifacts: RunArtifacts
    item_result: ScreeningItemResult

    @property
    def record_id(self) -> str:
        """`item_result.report_id` (== the original `Record.id`, per
        `record_to_screening_item()`'s own mapping), not `document_id` --
        `report_id` is FT_SCREENING's documented join key back to the
        originating record, and the two happen to be equal here only
        because of how this adapter constructs items, not because they
        mean the same thing in general."""
        return self.item_result.report_id

    @property
    def execution_status(
        self,
    ) -> Literal["completed", "input_unavailable", "duplicate_input", "processing_failed", "not_processed"]:
        return self.item_result.execution_status

    @property
    def decision(self) -> Literal["include", "exclude", "needs_human_review", "not_assessed"]:
        """"not_assessed" covers every non-"completed" execution_status
        (missing PDF / duplicate input / parser or provider failure /
        not_processed under fail_fast) -- FT_SCREENING's own
        ScreeningItemResult invariant guarantees eligibility is present if
        and only if execution_status == "completed", so this never has to
        guess. Phase B's own conservative-mapping rule (missing PDF or
        processing failure -> review queue, never included) is a Phase B
        concern, not this property's -- execution_status is exposed
        unchanged so Phase B can apply that rule itself."""
        elig = self.item_result.eligibility
        return elig.final_decision if elig is not None else "not_assessed"

    @property
    def primary_exclusion_reason_code(self) -> str | None:
        elig = self.item_result.eligibility
        return elig.primary_exclusion_reason_code if elig is not None else None

    @property
    def exclusion_reason_codes(self) -> list[str]:
        elig = self.item_result.eligibility
        return list(elig.exclusion_reason_codes) if elig is not None else []

    @property
    def ft_screening_result_id(self) -> str | None:
        elig = self.item_result.eligibility
        return elig.result_id if elig is not None else None


def map_run_result_to_chapter_outcomes(run_result: ScreeningRunResult) -> list[ChapterFullTextOutcome]:
    """Pure: one `ChapterFullTextOutcome` per `ScreeningItemResult` in
    `run_result.items`, in the same order -- each wraps that COMPLETE item
    result plus a reference to the run's own artifact paths. See
    `ChapterFullTextOutcome`'s own docstring for why nothing is
    summarized/discarded here."""
    return [
        ChapterFullTextOutcome(
            ft_screening_run_id=run_result.run_id,
            run_artifacts=run_result.artifacts,
            item_result=item,
        )
        for item in run_result.items
    ]
