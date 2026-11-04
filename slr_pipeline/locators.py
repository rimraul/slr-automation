"""
Logical locator grammar (`track2_pipeline_readiness_plan_2026-08-30.md`
Sec 10.3) -- the `slr://<kind>/<subkind>/<id>[/<subid>]` naming scheme used
at the reporting boundary (`slr_pipeline.export_manifest`) to name a live/
logical source instead of an absolute filesystem path (Sec 10.5).

Scope note (step 9 of the readiness plan): this module implements exactly
the grammar and registry needed by `ReportingStageEntry`/`ReportingStageManifest`
-- `LogicalLocator`, `LocatorGlob`, `_parse_and_check_locator`, `expand_glob`.
Deliberately NOT implemented here (later steps, per the plan's own step list):

  - `ISO8601_Z` and the `PayloadArtifact`/`PayloadManifestV1`/
    `FTScreeningSourceInfo`/`PayloadSourceInfo` models (Sec 10.1(a)) --
    owned by `reference_bundle.py::finalize_bundle()`, a step-15 concern.
    `POSIX_RELPATH` is the one "shared strict primitive" pulled forward
    into this module, because `ReportingStageEntry.live_relpath` (step 9,
    in scope) needs it now.
  - The `run:<bundle_id>@<payload_hash>/<artifact_id>[#<pointer>]`
    addressing scheme's `resolve()` mechanics (Sec 10.6) -- they resolve
    through `PayloadManifestV1.artifacts`, which does not exist yet.
    `RUN_LOCATOR_VERSION` is pinned here as a version constant only.
  - `expand_glob()`'s eventual `PayloadManifestV1`-typed call site (Sec
    10.3: "`expand_glob(glob, PayloadManifestV1)`") -- this module's
    `expand_glob` instead takes a plain iterable of candidate locator
    strings, which is exactly what step 9's own caller
    (`ReportingStageManifest.entries`) can supply; a later step can widen
    the call site to pass `PayloadManifestV1.artifacts`' own locators
    without changing this function's contract.
  - The `kind`s reserved for `PayloadManifestV1` alone (`state`,
    `run_evidence`, `fixture`, `bundle`) are still registered in
    `LogicalLocator._SUBKINDS` (Sec 10.3's registry is one shared table
    for both manifests), but no code in this step ever constructs one --
    `ReportingStageEntry`'s 7 `entry_kind`s only ever need `reporting`,
    `selection_ledger`, `screening_batch`, `extraction`,
    `quality_appraisal`, `ai_use_events`, `ft_screening`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Annotated, Iterable, List, Optional

from pydantic import AfterValidator, constr

LOCATOR_GRAMMAR_VERSION = "1"

# Sec 10.6 -- grammar version pin only; `resolve()` needs `PayloadManifestV1`
# (not yet built, see module docstring), so no `run:`-parsing code lives here.
RUN_LOCATOR_VERSION = "1"


def _check_posix_relpath_nfc(v: str) -> str:
    """Plan Sec 10.1, round-15/round-16 findings verbatim: NFC-normalization
    is enforced (not just documented), and a `.`/`..` path segment is
    rejected here, in the semantic validator, because pydantic-core's Rust
    regex engine has no look-around support (a lookaround shape regex
    raises `SchemaError` at class-body execution time, not at validation
    time -- reproduced in the plan against real pydantic 2.12.4)."""
    if unicodedata.normalize("NFC", v) != v:
        raise ValueError(f"{v!r} is not Unicode NFC-normalized")
    for segment in v.split("/"):
        if segment in (".", ".."):
            raise ValueError(f"{v!r} contains a '.' or '..' path segment")
    return v


POSIX_RELPATH = Annotated[
    constr(pattern=r"^[\w.-]+(/[\w.-]+)*$"),
    AfterValidator(_check_posix_relpath_nfc),
]

_LOCATOR_RE = re.compile(
    r"^slr://"
    r"(?P<kind>[a-z_]+)/"
    r"(?P<subkind>[a-z_]+)/"
    r"(?P<id>[A-Za-z0-9][A-Za-z0-9._-]{0,127})"
    r"(?:/(?P<subid>[A-Za-z0-9][A-Za-z0-9._-]{0,127}))?$"
)
_GLOB_RE = re.compile(r"^slr://(?P<kind>[a-z_]+)/(?P<subkind>[a-z_]+)/\*$")


@dataclass(frozen=True)
class LogicalLocator:
    """`kind`/`subkind` are plain `str` here (not `Literal[...]`) because
    this is a bare `@dataclass`, not a pydantic model -- a `Literal` type
    hint on a plain dataclass field is never enforced at construction time
    (only static type-checkers read it), so direct construction of a
    `LogicalLocator` with an unregistered kind would silently succeed. Real
    validation lives in `_parse_and_check_locator()`/`.parse()`, both of
    which check against `_SUBKINDS` explicitly."""

    kind: str
    subkind: str
    id: str  # ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$  (never '*' -- see LocatorGlob)
    subid: Optional[str] = None

    _SUBKINDS = {
        "reporting": {"artifact"},
        "selection_ledger": {"decision"},  # id=<key_digest 32hex>, subid=<sef_stem 64hex>
        "screening_batch": {"result"},
        "extraction": {
            "manifest", "checkpoint", "resolution",  # checkpoint: subid=<document_id>
            "review_packet", "queue_snapshot", "suggestion",
            "suggestions_index", "decision_journal",
        },
        "quality_appraisal": {"manifest", "checkpoint", "resolution", "confirmation"},
        "ai_use_events": {"event"},
        "ft_screening": {"run_result", "item_results", "ledger"},
        "state": {"snapshot"},  # PayloadManifestV1 only -- id in {terminal_state, terminal_graph_snapshot}
        "run_evidence": {"replay", "netguard", "cost_cap"},  # PayloadManifestV1 only -- id in {call_log, violations, ledger}
        "fixture": {"pdf", "native", "cassette", "spec", "protocol"},  # PayloadManifestV1 only
        "bundle": {"run_mode"},  # PayloadManifestV1 only -- id in {main, statement}
        "continuation": {"lineage", "preserved_decisions"},  # PayloadManifestV1 only -- id=main
        "reproduction": {
            "spec", "gold_standard", "stage_corpus", "pipeline_evaluation",
            "gate_report", "adjudication_journal",
        },
    }

    def __str__(self) -> str:
        return f"slr://{self.kind}/{self.subkind}/{self.id}" + (f"/{self.subid}" if self.subid else "")

    @classmethod
    def parse(cls, value: str) -> "LogicalLocator":
        """Round-trips `value` through the grammar and the `_SUBKINDS`
        registry, raising `ValueError` on either failure -- the same checks
        `_parse_and_check_locator()` performs, returned as a structured
        object instead of the bare string."""
        m = _LOCATOR_RE.match(value)
        if m is None:
            raise ValueError(f"{value!r} is not a well-formed slr:// locator")
        kind, subkind, loc_id, subid = m.group("kind"), m.group("subkind"), m.group("id"), m.group("subid")
        if kind not in cls._SUBKINDS:
            raise ValueError(f"{value!r}: {kind!r} is not a registered LogicalLocator kind")
        if subkind not in cls._SUBKINDS[kind]:
            raise ValueError(f"{value!r}: {subkind!r} is not a registered subkind of kind {kind!r}")
        return cls(kind=kind, subkind=subkind, id=loc_id, subid=subid)


def _parse_and_check_locator(value: str, *, kind: Optional[str] = None, subkind: Optional[str] = None) -> str:
    """Plan Sec 10.1 verbatim contract: always raises `ValueError` on a
    malformed grammar or an unregistered kind/subkind, regardless of
    whether `kind`/`subkind` are passed. `kind=None`/`subkind=None` (used
    by `ReportingStageEntry.logical_path`'s own `field_validator`, which
    runs before `entry_kind` is necessarily available) means "any
    registered kind/subkind -- grammar and registration only, no
    cross-check". Passing `kind` (used once `entry_kind` is known, by
    `ReportingStageEntry`'s `model_validator`) additionally asserts the
    locator's own `kind` segment equals it. Returns `value` unchanged on
    success (a pydantic `AfterValidator`/`field_validator` convention)."""
    m = _LOCATOR_RE.match(value)
    if m is None:
        raise ValueError(f"{value!r} is not a well-formed slr:// locator")
    parsed_kind, parsed_subkind = m.group("kind"), m.group("subkind")
    if parsed_kind not in LogicalLocator._SUBKINDS:
        raise ValueError(f"{value!r}: {parsed_kind!r} is not a registered LogicalLocator kind")
    if parsed_subkind not in LogicalLocator._SUBKINDS[parsed_kind]:
        raise ValueError(f"{value!r}: {parsed_subkind!r} is not a registered subkind of kind {parsed_kind!r}")
    if kind is not None and parsed_kind != kind:
        raise ValueError(f"{value!r}: expected kind {kind!r}, got {parsed_kind!r}")
    if subkind is not None and parsed_subkind != subkind:
        raise ValueError(f"{value!r}: expected subkind {subkind!r}, got {parsed_subkind!r}")
    return value


@dataclass(frozen=True)
class LocatorGlob:
    """`slr://<kind>/<subkind>/*` -- a SEPARATE type from `LogicalLocator`
    (no `id`/`subid` at all, not `id="*"`), permitted only in
    `ReportingStageEntry.built_from` (never in a manifest's own
    `logical_path`)."""

    kind: str
    subkind: str

    def __str__(self) -> str:
        return f"slr://{self.kind}/{self.subkind}/*"

    @classmethod
    def parse(cls, value: str) -> "LocatorGlob":
        m = _GLOB_RE.match(value)
        if m is None:
            raise ValueError(f"{value!r} is not a well-formed LocatorGlob (slr://<kind>/<subkind>/*)")
        kind, subkind = m.group("kind"), m.group("subkind")
        if kind not in LogicalLocator._SUBKINDS or subkind not in LogicalLocator._SUBKINDS.get(kind, set()):
            raise ValueError(f"{value!r}: {kind!r}/{subkind!r} is not a registered LocatorGlob kind/subkind")
        return cls(kind=kind, subkind=subkind)


def is_glob(value: str) -> bool:
    """`built_from` entries (Sec 10.3) distinguish a concrete locator from
    a glob purely by the literal `/*` suffix -- the same test
    `ReportingStageEntry._fields_match_entry_kind` (export_manifest.py)
    uses to decide which parser to hand a `built_from` string to."""
    return value.endswith("/*")


def expand_glob(glob: LocatorGlob, candidate_locators: Iterable[str]) -> List[str]:
    """The sorted, de-duplicated subset of `candidate_locators` whose kind
    and subkind match `glob`. Each candidate must itself be a well-formed,
    registered `LogicalLocator` string (malformed entries raise, they are
    never silently skipped -- an unexpectedly malformed manifest entry is
    a bug, not a glob-non-match).

    Scope note: the plan (Sec 10.3) writes this as
    `expand_glob(glob, PayloadManifestV1)`, expanding against a bundle's
    complete artifact inventory -- that type does not exist yet (see the
    module docstring). This implementation instead takes a plain iterable
    of locator strings, which is exactly what step 9's own caller
    (`ReportingStageManifest.entries`, via their `logical_path`s) can
    supply; a later step can widen the call site without changing this
    function's contract.
    """
    matches = {
        candidate
        for candidate in candidate_locators
        if (parsed := LogicalLocator.parse(candidate)).kind == glob.kind and parsed.subkind == glob.subkind
    }
    return sorted(matches)
