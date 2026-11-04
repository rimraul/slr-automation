"""
Track 2 plan (`track2_pipeline_readiness_plan_2026-08-30.md`) step 16
([C15]): the acceptance harness -- the 8 HANDOFF §1 checks (§11.1),
implemented as functions over an already-`open_bundle()`-verified `Bundle`
plus its `from_bundle()`-reconstructed `ReportingSourceSnapshot`.

This module deliberately does NOT re-derive what `open_bundle()`/
`from_bundle()` already establish (payload_manifest self-hash, complete
file-inventory match, per-artifact sha256, export_manifest.json's own
manifest_hash self-check and its per-entry cross-check against
payload_manifest -- see `reference_bundle.py`'s own module docstring for
the full four-layer breakdown). Its job is the NARROWER, check-specific
work on top of an already-tamper-evident bundle: per-content-model hash
self-verification (`AIUseEvent.verify_hashes()`/
`SelectionDecisionRecord.verify_hashes()`, which `open_bundle()` never
calls -- it only verifies FILE bytes against a recorded sha256, never a
model's own internal semantic_hash/content_hash self-consistency),
re-rendering `ai_traceability.json`/`prisma_scr_conformance.json`/
`run_mode_statement.md` from the reconstructed snapshot and byte-comparing
against the bundled files (catches renderer-vs-raw-layer divergence a pure
hash check cannot), the S12 PRISMA conservation check (`prisma_flow`,
ported from `examples/chapter/s12_reporting/solution.py`), and the narrow
structural assertions each HANDOFF check's own wording names (exactly 3
not-instrumented stages, item 22's conformance status, etc.).

Every check function is independently callable and independently
try/excepts its own body (see `_run()`) -- a malformed or tampered bundle
must produce a `status="fail"` `AcceptanceCheck` with a useful `detail`,
never crash the whole acceptance run. `build_acceptance_report()` is the
one entry point that runs all 8 and assembles the self-hashed
`AcceptanceReport` (§10.7's `AcceptanceReport.self_hash()`
construct-with-placeholder-then-`model_copy` pattern, same as
`tests/test_reference_bundle.py`'s own `AcceptanceReport` fixtures).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from slr_pipeline.ai_traceability_report import _render_ai_traceability_dict
from slr_pipeline.export_manifest import ReportingStageEntry
from slr_pipeline.extraction.hashing import canonical_json_bytes
from slr_pipeline.locators import LogicalLocator
from slr_pipeline.nodes.reporting import ReportingSourceSnapshot
from slr_pipeline.prisma_scr import render_prisma_scr_conformance
from slr_pipeline.reference_bundle import (
    AcceptanceCheck, AcceptanceReport, Bundle, ReferenceBundleError, apply_parent_oversight_union, from_bundle,
)
from slr_pipeline.extraction.resolution_schemas import EXTRACTION_RESOLUTION_SUGGESTION_STAGE
from slr_pipeline.reference_run_mode import (
    RunModeDocument,
    SURFACE_TO_NON_INSTRUMENTED_KEY,
    SURFACE_TO_PER_STAGE_KEY,
    render_run_mode_statement_md,
)
from slr_pipeline.state import PRISMACounts

# ---------------------------------------------------------------------------
# §11.2 group 2 -- the ported S12 PRISMA conservation check.
# ---------------------------------------------------------------------------


class PrismaInconsistencyError(ValueError):
    """Production counterpart of
    `examples/chapter/s12_reporting/solution.py::PrismaInconsistencyError`
    -- operates on the real `PRISMACounts` model rather than a teaching
    dict."""


class PrismaFlowResult(BaseModel):
    """Same six conservation groups, same field set, as the teaching
    version's `PrismaFlow` -- see
    `examples/chapter/s12_reporting/solution.py`."""

    model_config = ConfigDict(frozen=True)
    total_identified: int
    duplicates_removed: int
    after_dedup: int
    screened_title_abstract: int
    excluded_title_abstract: int
    included_after_ta: int
    fulltext_sought: int
    assessed_full_text: int
    acquisition_failures: int
    excluded_full_text: int
    excluded_full_text_reasons: Dict[str, int]
    included: int


def prisma_flow(prisma: PRISMACounts, *, full_text_review_queue: List[Dict[str, Any]]) -> PrismaFlowResult:
    """Ported from `examples/chapter/s12_reporting/solution.py::prisma_flow`
    onto the real production `PRISMACounts` model. Two differences from the
    teaching version, both load-bearing:

    - `PRISMACounts` has no `duplicates_removed` field (the teaching dict
      does) -- it is computed here as `total_identified - after_dedup`,
      never read.
    - `full_text_review_queue` is passed in explicitly rather than pulled
      off a raw `state` dict -- a `ReportingSourceSnapshot` already carries
      it verified, as `snapshot.full_text_review_queue.model`. The
      teaching version's own "must be exactly `[]`" precondition still
      applies unchanged: a non-terminal run has no conserved flow to
      check.
    """
    if full_text_review_queue != []:
        raise PrismaInconsistencyError(
            "not a terminal state -- full_text_review_queue must be exactly [] "
            f"(got {full_text_review_queue!r})"
        )

    total_identified = prisma.total_identified
    # Derived, not read (see docstring) -- group 1's first condition below
    # is therefore definitionally satisfied by construction; it is kept
    # anyway as the direct expression of the conservation law, and the
    # `duplicates_removed < 0` half remains a genuine guard (it fires iff
    # `after_dedup > total_identified`, an impossible dedup outcome).
    duplicates_removed = total_identified - prisma.after_dedup
    after_dedup = prisma.after_dedup
    screened = prisma.screened_title_abstract
    excluded_ta = prisma.excluded_title_abstract
    retrieval = prisma.fulltext_retrieval
    assessed = prisma.assessed_full_text
    excluded_ft = prisma.excluded_full_text
    excluded_ft_reasons = prisma.excluded_full_text_reasons
    included = prisma.included

    included_after_ta = screened - excluded_ta
    acquisition_failures = included_after_ta - assessed

    # 1: identification / dedup / screening handoff
    if duplicates_removed < 0 or after_dedup != total_identified - duplicates_removed:
        raise PrismaInconsistencyError("group 1: after_dedup != total_identified - duplicates_removed")
    if screened != after_dedup:
        raise PrismaInconsistencyError("group 1: screened_title_abstract != after_dedup")
    # 2: title/abstract exclusions
    if included_after_ta < 0:
        raise PrismaInconsistencyError("group 2: excluded_title_abstract exceeds screened count")
    # 3: every title/abstract inclusion was sought for full text
    if sum(retrieval.values()) != included_after_ta:
        raise PrismaInconsistencyError("group 3: sum(fulltext_retrieval) != included_after_ta")
    # 4: full-text acquisition
    if assessed != retrieval.get("parsed", 0):
        raise PrismaInconsistencyError("group 4: assessed_full_text != fulltext_retrieval['parsed']")
    if acquisition_failures < 0:
        raise PrismaInconsistencyError("group 4: assessed_full_text exceeds included_after_ta")
    # 5: full-text exclusions
    if sum(excluded_ft_reasons.values()) != excluded_ft:
        raise PrismaInconsistencyError("group 5: sum(excluded_full_text_reasons) != excluded_full_text")
    if included != assessed - excluded_ft or included < 0:
        raise PrismaInconsistencyError("group 5: included != assessed_full_text - excluded_full_text")
    # 6: scoping-mode fitness -- quality appraisal must have been skipped
    if prisma.quality_assessed is not None or prisma.excluded_quality is not None:
        raise PrismaInconsistencyError("group 6: quality appraisal counts present in a scoping review")

    return PrismaFlowResult(
        total_identified=total_identified, duplicates_removed=duplicates_removed, after_dedup=after_dedup,
        screened_title_abstract=screened, excluded_title_abstract=excluded_ta, included_after_ta=included_after_ta,
        fulltext_sought=sum(retrieval.values()), assessed_full_text=assessed,
        acquisition_failures=acquisition_failures, excluded_full_text=excluded_ft,
        excluded_full_text_reasons=dict(excluded_ft_reasons), included=included,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class AcceptanceCheckFailed(RuntimeError):
    """Raised inside a check body to fail that check with a specific
    message. Any OTHER exception (KeyError, pydantic ValidationError,
    `PrismaInconsistencyError`, an `AIEventIntegrityError`, ...) is caught
    just as readily by `_run()` -- this class exists only so a check body
    can state a failure in its own words without inventing a bespoke
    exception type per check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceCheckFailed(message)


def _run(check_id: int, body: Callable[[], str]) -> AcceptanceCheck:
    """Runs one check body; ANY exception closes the check to
    `status="fail"` with the exception's own message as `detail`, rather
    than aborting the whole acceptance run -- a harness that could itself
    crash on a malformed bundle would defeat its own purpose."""
    try:
        detail = body()
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        return AcceptanceCheck(id=check_id, status="fail", detail=f"{type(exc).__name__}: {exc}")
    return AcceptanceCheck(id=check_id, status="pass", detail=detail)


def _renderer_artifact_entry(bundle: Bundle, name: str) -> ReportingStageEntry:
    for entry in bundle.reporting_stage_manifest.entries:
        if entry.entry_kind == "renderer_artifact" and entry.name == name:
            return entry
    raise AcceptanceCheckFailed(f"no renderer_artifact entry named {name!r} in export_manifest.json")


# ---------------------------------------------------------------------------
# The 8 HANDOFF §1 checks (§11.1)
# ---------------------------------------------------------------------------


def check_1_full_autonomy_terminal_state(
    bundle: Bundle, snapshot: ReportingSourceSnapshot, *,
    terminal_state: Dict[str, Any], terminal_graph_snapshot: Dict[str, Any],
) -> AcceptanceCheck:
    """Review family = scoping, full autonomy, quality appraisal skipped,
    terminal state reached; Track-2 additions: zero human involvement
    anywhere, full-text stage automation, exact dedup mode with no
    external import, every database search complete."""

    def body() -> str:
        cp = snapshot.config_projection
        _require(cp.review_type == "scoping", f"review_type must be 'scoping', got {cp.review_type!r}")
        _require(
            cp.screening_uncertain_threshold == 0.0,
            f"screening_uncertain_threshold must be 0.0 (full autonomy), got {cp.screening_uncertain_threshold!r}",
        )
        prisma = snapshot.prisma.model
        _require(
            prisma.quality_assessed is None and prisma.excluded_quality is None,
            "quality appraisal must be skipped (quality_assessed/excluded_quality both None)",
        )
        _require(
            terminal_graph_snapshot.get("next") == [],
            f"terminal_graph_snapshot.next must be [] (terminal), got {terminal_graph_snapshot.get('next')!r}",
        )
        _require(
            terminal_state.get("extraction_review_queue") == [],
            "terminal_state.extraction_review_queue must be [] (no queued human review)",
        )
        _require(
            snapshot.full_text_review_queue.model == [],
            "full_text_review_queue must be [] (no queued human review)",
        )
        _require(
            snapshot.screening_review_resolutions.model == [] and snapshot.full_text_review_resolutions.model == [],
            "zero human-resolution entries required in screening_review_resolutions/full_text_review_resolutions",
        )
        human_ledger_entries = [vf.model.decision_id for vf in snapshot.ledger_records if vf.model.decision_maker == "human"]
        _require(not human_ledger_entries, f"selection ledger has human decision_maker entries: {human_ledger_entries!r}")
        human_oversight_stages = [
            stage for stage, mode in snapshot.oversight_mode_state.items()
            if mode in ("human_resolved", "human_generated")
        ]
        _require(not human_oversight_stages, f"oversight_mode has human-involved stage(s): {human_oversight_stages!r}")
        _require(
            snapshot.oversight_mode_state.get("full_text_eligibility") == "automation_unreviewed",
            "full_text_eligibility oversight_mode must be 'automation_unreviewed', got "
            f"{snapshot.oversight_mode_state.get('full_text_eligibility')!r}",
        )
        _require(
            snapshot.automation_origin_state.get("full_text_eligibility") == "ai",
            "full_text_eligibility automation_origin must be 'ai', got "
            f"{snapshot.automation_origin_state.get('full_text_eligibility')!r}",
        )
        mode_profile = bundle.payload_manifest.mode_profile
        if mode_profile == "track2_offline":
            _require(snapshot.reference_context is not None, "reference_context must be present on a Track 2 bundle")
            _require(
                snapshot.reference_context.dedup_mode == "exact",
                f"reference_context.dedup_mode must be 'exact', got {snapshot.reference_context.dedup_mode!r}",
            )
        elif mode_profile == "track3_live":
            _require(
                snapshot.reference_context is None,
                "reference_context must be absent on a track3_live bundle -- it is a "
                "Track-2-reference-run-only field",
            )
        else:
            _require(False, f"unknown mode_profile {mode_profile!r} -- no check_1 rule defined for it")
        dedup_exact_events = [vf for vf in snapshot.events if vf.model.stage == "dedup_exact"]
        _require(dedup_exact_events, "no dedup_exact AIUseEvent present")
        _require(
            len(snapshot.raw_records.model) == prisma.total_identified,
            "raw_records count must equal prisma.total_identified (no external dedup import) -- got "
            f"{len(snapshot.raw_records.model)} vs {prisma.total_identified}",
        )
        incomplete = [
            e.database for e in snapshot.search_log.model.entries
            if e.retrieval_status not in ("exhaustive", "capped")
        ]
        _require(not incomplete, f"database search(es) neither 'exhaustive' nor 'capped': {incomplete!r}")
        # 'capped' means error-free, NOT a claim of comprehensive evidence-frame
        # coverage -- this check only rejects a genuine retrieval failure
        # (partial/failed); it must never report a capped search under the
        # same "complete" label as a genuinely exhaustive one (review finding
        # P1, 2026-09-24). Evidence-frame comprehensiveness for a real
        # screening run is a separate, stricter gate
        # (`update_driver.enforce_exhaustive_retrieval`), not this function's job.
        non_exhaustive = sorted({
            e.database for e in snapshot.search_log.model.entries if e.retrieval_status == "capped"
        })
        search_log_claim = (
            "exhaustive-search-log" if not non_exhaustive
            else f"capped-search-log (error-free but non-exhaustive: {non_exhaustive!r})"
        )
        return (
            "scoping/full-autonomy/terminal-state/zero-human-involvement/full-text-automation/"
            f"exact-dedup/{search_log_claim} all confirmed"
        )

    return _run(1, body)


def check_2_prisma_conservation(bundle: Bundle, snapshot: ReportingSourceSnapshot) -> AcceptanceCheck:
    """`prisma_counts` populated and conserved across all six groups
    (§11.1 item 2), fed through the ported `prisma_flow`."""

    def body() -> str:
        flow = prisma_flow(snapshot.prisma.model, full_text_review_queue=snapshot.full_text_review_queue.model)
        return f"prisma_flow conserved: total_identified={flow.total_identified}, included={flow.included}"

    return _run(2, body)


def _without_ft_native_provenance_paths(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strips the one documented, deliberate re-render divergence
    (`reference_bundle.refinalize_reporting_from_bundle`'s own docstring):
    `full_text_eligibility_native_provenance.ft_screening_artifact_paths`
    re-anchors to THIS bundle's current `raw/ft_screening/` location on
    every `from_bundle()` reconstruction, never byte-identical to whatever
    transient host path the original live pipeline run happened to write
    to -- and `content_hash` (computed OVER that field, see
    `ai_traceability_report.py`'s own `sha256_hex(body)` call) therefore
    changes right along with it. Both are excluded here, not silently
    ignored: the exclusion itself is the assertion that nothing ELSE in
    the document differs."""
    doc = dict(doc)
    doc.pop("content_hash", None)
    provenance = doc.get("full_text_eligibility_native_provenance")
    if isinstance(provenance, dict) and "ft_screening_artifact_paths" in provenance:
        provenance = dict(provenance)
        provenance.pop("ft_screening_artifact_paths", None)
        doc["full_text_eligibility_native_provenance"] = provenance
    return doc


def check_3_ai_traceability_shape(bundle: Bundle, snapshot: ReportingSourceSnapshot) -> AcceptanceCheck:
    """`ai_traceability.json`'s real shape, re-rendered from the
    reconstructed snapshot and compared against the bundled file (modulo
    the one documented re-anchoring exception, see
    `_without_ft_native_provenance_paths`) -- exactly 3 not-instrumented
    stages, `full_text_eligibility` instrumented with zero events, dedup
    row recognizes `dedup_exact`."""

    def body() -> str:
        rendered = _render_ai_traceability_dict(snapshot)
        bundled_path = bundle.resolve(str(LogicalLocator("reporting", "artifact", "ai_traceability_json")))
        bundled = json.loads(bundled_path.read_bytes())
        _require(
            _without_ft_native_provenance_paths(rendered) == _without_ft_native_provenance_paths(bundled),
            "re-rendered ai_traceability.json does not match the bundled file (beyond the documented "
            "ft_screening_artifact_paths/content_hash re-anchoring exception)",
        )
        stages = rendered["stages"]
        not_instrumented = {name for name, entry in stages.items() if entry["instrumentation_status"] == "not_instrumented"}
        _require(
            not_instrumented == {"synthesis_bertopic", "extraction", "quality_appraisal"},
            f"expected exactly 3 not-instrumented stages, got {sorted(not_instrumented)!r}",
        )
        ft = stages["full_text_eligibility"]
        _require(
            ft["instrumentation_status"] == "instrumented" and ft["event_count"] == 0,
            f"full_text_eligibility must be instrumented with event_count 0, got {ft!r}",
        )
        dedup = stages["dedup"]
        _require(dedup["event_count"] >= 1, "dedup stage has zero events")
        _require(
                any(vf.model.stage == "dedup_exact" for vf in snapshot.events),
                "no dedup_exact event contributes to the dedup stage row",
        )
        return "ai_traceability.json shape + byte-equality confirmed"

    return _run(3, body)


def check_4_ledger_hash_integrity(bundle: Bundle, snapshot: ReportingSourceSnapshot) -> AcceptanceCheck:
    """Complete `AIUseEvent`/`SelectionDecisionRecord` ledger, persisted +
    hashed -- per-model internal hash self-verification (`verify_hashes()`,
    which `open_bundle()` never calls) plus mapping equality against
    `ReportingStageManifest`'s own `ai_use_event`/`selection_decision`
    entries."""

    def body() -> str:
        for vf in snapshot.events:
            vf.model.verify_hashes()
        for vf in snapshot.ledger_records:
            vf.model.verify_hashes()

        manifest_events = {
            e.event_id: e for e in bundle.reporting_stage_manifest.entries if e.entry_kind == "ai_use_event"
        }
        _require(
            set(manifest_events) == {vf.model.event_id for vf in snapshot.events},
            "ai_use_event ids in export_manifest.json do not match the bundled event set",
        )
        for vf in snapshot.events:
            entry = manifest_events[vf.model.event_id]
            _require(
                entry.file_sha256 == vf.file_sha256 and entry.internal_hash == vf.internal_hash
                and entry.stage == vf.model.stage and entry.work_item_id == vf.model.work_item_id,
                f"ai_use_event {vf.model.event_id!r} disagrees between export_manifest.json and the bundled file",
            )

        manifest_decisions = {
            e.decision_id: e for e in bundle.reporting_stage_manifest.entries if e.entry_kind == "selection_decision"
        }
        _require(
            set(manifest_decisions) == {vf.model.decision_id for vf in snapshot.ledger_records},
            "selection_decision ids in export_manifest.json do not match the bundled ledger record set",
        )
        for vf in snapshot.ledger_records:
            entry = manifest_decisions[vf.model.decision_id]
            _require(
                entry.file_sha256 == vf.file_sha256 and entry.internal_hash == vf.internal_hash
                and entry.record_id == vf.model.record_id and entry.stage == vf.model.stage,
                f"selection_decision {vf.model.decision_id!r} disagrees between export_manifest.json and the bundled file",
            )
        return (
            f"{len(snapshot.events)} AIUseEvent(s) + {len(snapshot.ledger_records)} SelectionDecisionRecord(s) "
            "self-verified and manifest-mapped"
        )

    return _run(4, body)


def check_5_prisma_scr_report(bundle: Bundle, snapshot: ReportingSourceSnapshot) -> AcceptanceCheck:
    """PRISMA-ScR-conformant report present, all 22 matrix items resolved,
    item 22 genuinely 'reported'. `open_bundle()` already self-hash- and
    per-entry-verified both files; this check is narrowly the semantic
    shape HANDOFF item 5 names (semantic REVIEW status, i.e.
    `ReviewerAttestation.decision`, is deliberately out of scope here --
    see `AcceptanceReport`'s own docstring)."""

    def body() -> str:
        mode_profile = bundle.payload_manifest.mode_profile
        _require(mode_profile in ("track2_offline", "track3_live"),
                 f"unknown mode_profile {mode_profile!r} -- no check_5 rule defined for it")
        report_entry = _renderer_artifact_entry(bundle, "prisma_scr_report_md")
        if mode_profile == "track3_live":
            conformance_entry = _renderer_artifact_entry(bundle, "prisma_scr_conformance_json")
            _require(
                report_entry.status == "not_applicable",
                f"prisma_scr_report_md status must be 'not_applicable' for track3_live, got {report_entry.status!r}",
            )
            _require(
                conformance_entry.status == "not_applicable",
                "prisma_scr_conformance_json status must be 'not_applicable' for track3_live, got "
                f"{conformance_entry.status!r}",
            )
            return "prisma_scr_report_md/prisma_scr_conformance_json correctly not_applicable (Track-2-reference-run-only artifacts)"

        _require(report_entry.status == "present", f"prisma_scr_report_md status must be 'present', got {report_entry.status!r}")

        conformance = render_prisma_scr_conformance(snapshot)
        items = conformance["items"]
        _require(len(items) == 22, f"prisma_scr_conformance must have exactly 22 items, got {len(items)}")
        item_22 = next((i for i in items if i["item_id"] == 22), None)
        _require(item_22 is not None, "prisma_scr_conformance has no item_id 22")
        _require(item_22["status"] == "reported", f"item 22 status must be 'reported', got {item_22['status']!r}")

        bundled_path = bundle.resolve(str(LogicalLocator("reporting", "artifact", "prisma_scr_conformance_json")))
        _require(
            canonical_json_bytes(conformance) == bundled_path.read_bytes(),
            "re-rendered prisma_scr_conformance.json does not byte-match the bundled file",
        )
        return "prisma_scr_report.md present, 22/22 items resolved, item 22 reported"

    return _run(5, body)


def check_6_evidence_map(bundle: Bundle, snapshot: ReportingSourceSnapshot) -> AcceptanceCheck:
    """Hashed evidence-map artifact present and hash-verified. Deliberately
    NOT re-deriving the CSV content -- HANDOFF item 6 says "reused, not
    re-derived"; `bundle.resolve()` already re-verifies the file's sha256
    against `payload_manifest.json`. `evidence_map_csv` is legitimately
    `not_applicable` whenever `snapshot.evidence_map is None` (renderer's
    own gate: `generate_evidence_map` is off, or there were zero included
    records to chart) -- a genuine negative/access-limited scientific
    outcome, not a pipeline failure, so this check cross-verifies that
    condition against the snapshot itself rather than trusting the stored
    entry's status blindly."""

    def body() -> str:
        entry = _renderer_artifact_entry(bundle, "evidence_map_csv")
        if entry.status == "not_applicable":
            _require(snapshot.evidence_map is None,
                      "evidence_map_csv status is 'not_applicable' but snapshot.evidence_map is not None")
            return "evidence_map.csv correctly not_applicable (no included records to chart)"
        _require(entry.status == "present", f"evidence_map_csv status must be 'present', got {entry.status!r}")
        bundle.resolve(str(LogicalLocator("reporting", "artifact", "evidence_map_csv")))  # raises on hash mismatch
        return "evidence_map.csv present and hash-verified"

    return _run(6, body)


def check_7_model_ids_and_config_snapshot(bundle: Bundle, snapshot: ReportingSourceSnapshot) -> AcceptanceCheck:
    """Model ids / prompt versions / frozen config snapshot (§12.2)
    present and sufficient. The Docling tree digest's own live re-
    verification is deliberately out of scope here -- `payload_manifest.
    source.docling_assets_manifest_digest`'s presence is schema-guaranteed
    at parse time; re-checking it against a live, re-hashed asset tree is
    step 19's (the asset-lock commit's) job, not this run-level check's."""

    def body() -> str:
        source = bundle.payload_manifest.source
        _require(bool(source.docling_assets_manifest_digest), "docling_assets_manifest_digest missing")
        _require(bool(source.docling_archive_sha256), "docling_archive_sha256 missing")
        missing_model = [vf.model.event_id for vf in snapshot.events if vf.model.tool_kind == "llm" and not vf.model.model]
        _require(not missing_model, f"LLM event(s) with an empty model id: {missing_model!r}")
        missing_prompt_version = [
            vf.model.event_id for vf in snapshot.events
            if vf.model.tool_kind == "llm" and not vf.model.prompt_template_version
        ]
        _require(not missing_prompt_version, f"LLM event(s) with an empty prompt_template_version: {missing_prompt_version!r}")
        return (
            "model ids + prompt template versions present on every LLM event; frozen config snapshot "
            "(payload_manifest.source) present -- Docling tree re-hash is step 19's own job, not re-checked here"
        )

    return _run(7, body)


def check_8_run_mode_statement(
    bundle: Bundle, snapshot: ReportingSourceSnapshot, *, run_mode: RunModeDocument,
) -> AcceptanceCheck:
    """Written statement of real vs. synthetic mode -- `run_mode.json`
    valid against the §11.3 strict schema (already validated by the
    caller, via `RunModeDocument.model_validate()` -- a validation failure
    there surfaces as this check's own `AcceptanceCheckFailed`, see
    `build_acceptance_report()`), consistent with `ai_traceability.json`,
    and `run_mode_statement.md` byte-reproducible from it."""

    def body() -> str:
        for surface, schema_key in SURFACE_TO_PER_STAGE_KEY.items():
            event_count = sum(1 for vf in snapshot.events if vf.model.stage == schema_key)
            expected = run_mode.per_stage_backend[schema_key].calls
            _require(
                event_count == expected,
                f"run_mode.per_stage_backend[{schema_key!r}].calls == {expected}, but "
                f"{event_count} AIUseEvent(s) with stage=={schema_key!r} are bundled (surface {surface!r})",
            )

        resolution_event_count = sum(
            1 for vf in snapshot.events if vf.model.stage == EXTRACTION_RESOLUTION_SUGGESTION_STAGE
        )
        resolution_entry = run_mode.resolution_suggestion_calls.get(EXTRACTION_RESOLUTION_SUGGESTION_STAGE)
        resolution_expected = resolution_entry.calls if resolution_entry is not None else 0
        _require(
            resolution_event_count == resolution_expected,
            f"run_mode.resolution_suggestion_calls[{EXTRACTION_RESOLUTION_SUGGESTION_STAGE!r}].calls == "
            f"{resolution_expected}, but {resolution_event_count} AIUseEvent(s) with "
            f"stage=={EXTRACTION_RESOLUTION_SUGGESTION_STAGE!r} are bundled",
        )

        rendered = _render_ai_traceability_dict(snapshot)
        stages = rendered["stages"]
        extraction_key = SURFACE_TO_NON_INSTRUMENTED_KEY["extraction_batch"]
        ft_key = SURFACE_TO_NON_INSTRUMENTED_KEY["ft_adjudication"]
        _require(
            stages[extraction_key]["instrumentation_status"] == "not_instrumented",
            f"run_mode's non_instrumented_model_calls names {extraction_key!r}, but ai_traceability.json "
            f"marks it {stages[extraction_key]['instrumentation_status']!r}",
        )
        _require(
            stages[ft_key]["instrumentation_status"] == "instrumented" and stages[ft_key]["event_count"] == 0,
            f"run_mode's non_instrumented_model_calls names {ft_key!r}, but ai_traceability.json does not "
            "show it as instrumented-with-zero-events",
        )

        regenerated = render_run_mode_statement_md(run_mode).encode("utf-8")
        bundled_path = bundle.resolve(str(LogicalLocator("bundle", "run_mode", "statement")))
        # render_run_mode_statement_md() always emits bare '\n' (Python's
        # "\n".join()); a run whose launcher wrote the file in Windows text
        # mode legitimately has '\r\n' line endings on disk instead -- a
        # platform artifact of how the bytes were written, not a content
        # difference, so both sides are newline-normalized before comparing
        # (same CRLF/LF class of issue this codebase already root-caused for
        # committed wheel-source hashes via .gitattributes eol=lf).
        _require(
            regenerated.replace(b"\r\n", b"\n") == bundled_path.read_bytes().replace(b"\r\n", b"\n"),
            "regenerated run_mode_statement.md does not byte-match the bundled file (after CRLF/LF normalization)",
        )
        return "run_mode.json schema-valid, per-stage call counts + non-instrumented crosswalk consistent, statement.md reproduces"

    return _run(8, body)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def build_acceptance_report(bundle: Bundle, *, parent_bundle: Optional[Bundle] = None) -> AcceptanceReport:
    """Runs all 8 HANDOFF §1 checks against `bundle` and assembles the
    self-hashed `AcceptanceReport`. `bundle` must already be the return
    value of `reference_bundle.open_bundle()` -- this function performs no
    inventory/self-hash verification of its own beyond what that call
    already did.

    `parent_bundle` (Unit E addition): required for a Track 3 continuation
    bundle -- `from_bundle(bundle)` alone can only ever reconstruct the
    continuation's own narrow `oversight_mode_state`/`automation_origin_
    state`/`screening_qc_state`, never the parent-inclusive union
    `check_3_ai_traceability_shape()` (and `check_1`/`check_8`, which read
    the same 3 fields off this same shared `snapshot`) needs to reproduce
    what was originally published. Default `None` preserves today's
    non-continuation behavior unchanged. Unit J: required (hard error)
    whenever `bundle.payload_manifest.parent_bundle_id` is set -- a
    continuation bundle can never be correctly re-verified without its
    parent. The presence check below is only a fast, early failure --
    `from_bundle()` itself is what actually enforces an exact
    `parent_bundle.bundle_id` match against the declared `parent_bundle_id`
    and cross-checks `continuation_lineage.json` against the parent's real
    registry hashes; a caller cannot pass an unrelated bundle as
    `parent_bundle` and have it silently accepted."""
    if bundle.payload_manifest.parent_bundle_id is not None and parent_bundle is None:
        raise ReferenceBundleError(
            f"build_acceptance_report(): bundle {bundle.bundle_id!r} has parent_bundle_id "
            f"{bundle.payload_manifest.parent_bundle_id!r} set -- parent_bundle is required")
    snapshot = from_bundle(bundle, parent_bundle=parent_bundle)
    if parent_bundle is not None:
        snapshot = apply_parent_oversight_union(snapshot, from_bundle(parent_bundle))
    terminal_state = json.loads(bundle.resolve("slr://state/snapshot/terminal_state").read_bytes())
    terminal_graph_snapshot = json.loads(bundle.resolve("slr://state/snapshot/terminal_graph_snapshot").read_bytes())
    run_mode_dict = json.loads(bundle.resolve(str(LogicalLocator("bundle", "run_mode", "main"))).read_bytes())

    def run_mode_check() -> AcceptanceCheck:
        try:
            run_mode = RunModeDocument.model_validate(run_mode_dict)
        except Exception as exc:  # noqa: BLE001 -- schema failure IS a check-8 failure, not a crash
            return AcceptanceCheck(id=8, status="fail", detail=f"run_mode.json failed schema validation: {exc}")
        return check_8_run_mode_statement(bundle, snapshot, run_mode=run_mode)

    checks = [
        check_1_full_autonomy_terminal_state(
            bundle, snapshot, terminal_state=terminal_state, terminal_graph_snapshot=terminal_graph_snapshot,
        ),
        check_2_prisma_conservation(bundle, snapshot),
        check_3_ai_traceability_shape(bundle, snapshot),
        check_4_ledger_hash_integrity(bundle, snapshot),
        check_5_prisma_scr_report(bundle, snapshot),
        check_6_evidence_map(bundle, snapshot),
        check_7_model_ids_and_config_snapshot(bundle, snapshot),
        run_mode_check(),
    ]

    ar_unsigned = AcceptanceReport(
        payload_hash=bundle.payload_manifest.payload_hash,
        mode_profile=bundle.payload_manifest.mode_profile,
        source=bundle.payload_manifest.source,
        checks=checks,
        acceptance_report_hash="0" * 64,
    )
    return ar_unsigned.model_copy(update={"acceptance_report_hash": ar_unsigned.self_hash()})
