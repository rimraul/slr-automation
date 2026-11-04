"""Scientific projection classifier (Track 2 plan §8 / step 8, item 8).

This module owns the vocabulary the two-run determinism comparison
(`test_two_independent_runs_project_equal`, step 16) will eventually use to
decide, leaf by leaf, how a given `ReviewState` (or external-store) field
must be compared across two independent runs of the SAME reference corpus.
Per plan §8.1, this file is deliberately split into two things that never
share fixtures:

  (A) schema coverage -- `test_projection_classifies_every_model_field`:
      the classifier below reaches every leaf of a branch-complete exemplar
      object with zero unclassified / obsolete / doubly-classified paths.
  (B) the real oracle assertions on Track 2's own terminal state
      (`test_reference_run_state_shape`) and the real two-run comparison
      (`test_two_independent_runs_project_equal`) both require the replay
      transport + synthetic corpus (steps 10-12) and the driver (step 16) --
      NEITHER is implemented here (round-11 finding 7, restated in the plan's
      own step 8 bullet). This module only builds the classifier + the
      derivation registry + the run-id canonicalisation used by
      `test_projection_fk_isomorphism`'s SYNTHETIC paired objects.

Scope deliberately NOT covered by this module yet (documented, not silently
dropped -- each lands in the step that actually produces the artifact it
needs):
  - The full `input_artifact_refs`/`output_artifact_refs` per-variant grammar
    (§8.2's `artifact_ref` sub-table, `test_every_artifact_ref_grammar`) --
    classified here only at the coarse `ARTIFACT_REF` level.
  - External-store model classification beyond `AIUseEvent`,
    `SelectionDecisionRecord`, and `ScreeningBatchResult` (RunManifest /
    DocumentCheckpoint / the complete FT_SCREENING schema / the run-evidence
    store, §8.8) -- those artifacts do not exist inside `ReviewState` itself
    and this step has no real reference run to source them from yet.
  - `SelectionDecisionRecord` supersession-chain pairing (§8.4 points 3-5) --
    needs a real chain in the reference corpus (steps 12/16); only the
    whole-run id bijection (§8.4 points 1-2) is implemented here.
  - Scanning `derived_id`s embedded in `rq_synthesis` / `imrad_*` prose
    (§8.2's `templated_text` treatment for those fields) -- classified here
    at the coarser `EXACT` level; only `methods_text` and (step 14)
    `prisma_scr_report_md` get the dedicated `TEMPLATED_TEXT` class -- both
    are long, free-form, state-templated prose bodies where a scan for
    embedded `derived_id`s would be worth the cost, unlike this step's other
    EXACT-classified fields.

`SCIENTIFIC_PROJECTION_VERSION` bumps whenever `CLASSES` changes shape --
`tests/reference_run/fixtures/scientific_projection.json` is a byte-checked
dump of `CLASSES` (plan §0.4 row for this file).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

# Real, already-reviewed derivation functions -- never reinvented here.
from slr_pipeline.state import compute_answer_id, _compute_claim_id
from slr_pipeline.ai_traceability import compute_event_id
from slr_pipeline.selection_ledger import derive_source_event_id
from slr_pipeline.synthesis_evidence import make_evidence_id
from slr_pipeline.nodes.screening import compute_screening_work_item_id

SCIENTIFIC_PROJECTION_VERSION = "2"  # bumped: step 14 ([C13]) added the 4 prisma_scr_* review_state patterns

# ---------------------------------------------------------------------------
# 8.2 -- the classes
# ---------------------------------------------------------------------------

EXACT = "exact"
STABLE_ID = "stable_id"
DERIVED_ID = "derived_id"
CANONICAL_ID = "canonical_id"
STRUCTURED_REF = "structured_ref"
DERIVED_HASH = "derived_hash"
PROVENANCE_HASH = "provenance_hash"
ENDPOINT_URL = "endpoint_url"
ARTIFACT_REF = "artifact_ref"
TEMPLATED_TEXT = "templated_text"
PERMITTED_DYNAMIC = "permitted_dynamic"

ALL_CLASSES: FrozenSet[str] = frozenset({
    EXACT, STABLE_ID, DERIVED_ID, CANONICAL_ID, STRUCTURED_REF, DERIVED_HASH,
    PROVENANCE_HASH, ENDPOINT_URL, ARTIFACT_REF, TEMPLATED_TEXT, PERMITTED_DYNAMIC,
})


def _build_patterns(pairs: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    """Dict-literal-with-a-duplicate-key-check -- a pattern string typo'd
    twice with two different classes would otherwise just silently keep the
    last one; this raises immediately instead."""
    out: Dict[str, str] = {}
    for pattern, cls in pairs:
        if pattern in out:
            raise ValueError(f"duplicate pattern {pattern!r} in the same table")
        if cls not in ALL_CLASSES:
            raise ValueError(f"pattern {pattern!r} uses unknown class {cls!r}")
        out[pattern] = cls
    return out


def _record_field_patterns(prefix: str) -> List[Tuple[str, str]]:
    """`Record` per §8.5, reused under every one of the five `List[Record]`
    `ReviewState` keys (`raw_records`/`deduplicated_records`/
    `included_records`/`screened_records_full`/`full_text_records`) --
    written once here rather than five times."""
    return [
        (f"{prefix}.*", EXACT),  # ambient default for every other Record field
        (f"{prefix}.*.id", STABLE_ID),
        (f"{prefix}.*.duplicate_of", STABLE_ID),
        (f"{prefix}.*.full_text_path", STRUCTURED_REF),
    ]


# --- ReviewState (§8.3) + its nested models (§8.5, §8.7) -------------------

_REVIEW_STATE_PAIRS: List[Tuple[str, str]] = [
    ("config", EXACT),
    ("config.output_dir", STRUCTURED_REF),
    ("review_type", EXACT),
    ("protocol", EXACT),
    ("search_strategy", EXACT),
    ("search_strategy.producing_event_ids.*", DERIVED_ID),

    *_record_field_patterns("raw_records"),
    *_record_field_patterns("deduplicated_records"),
    *_record_field_patterns("included_records"),
    *_record_field_patterns("screened_records_full"),
    *_record_field_patterns("full_text_records"),

    # ChartedRecord
    ("charted_data.*", EXACT),
    ("charted_data.*.record_id", STABLE_ID),

    # full_text_review_queue -- (A) exemplar branch (§8.8's
    # branch-complete-exemplar list); empty this run per (B).
    ("full_text_review_queue.*", EXACT),
    ("full_text_review_queue.*.record_id", STABLE_ID),

    # Generic audit-trail / queue lists the plan does not itemise beyond
    # "human resolution trail" -- classified as one opaque EXACT unit rather
    # than recursed into, since no per-field id/hash is called out for them.
    ("full_text_review_resolutions", EXACT),
    ("extraction_review_resolutions", EXACT),
    ("quality_appraisal_review_queue", EXACT),
    ("quality_appraisal_resolutions", EXACT),
    ("quality_appraisal_confirmations", EXACT),
    ("quality_appraisal_provisional_ledger", EXACT),
    ("pipeline_errors", EXACT),
    ("screening_review_resolutions", EXACT),

    ("ft_screening_run_id", CANONICAL_ID),
    ("ft_screening_execution_fingerprint", DERIVED_HASH),
    ("ft_screening_artifact_paths.*", STRUCTURED_REF),

    # No bare "extraction_results.*" ambient here -- unlike other
    # per-element rows in this table, EVERY direct key of a
    # `DocumentExtractionResult` dump (document_id/fields/batch_executions/
    # planned_chunk_ids_by_batch) already has its own explicit rule below,
    # so a top-level ambient default would never actually be reached and
    # would show up as a permanently-obsolete pattern.
    ("extraction_results.*.document_id", STABLE_ID),
    ("extraction_results.*.fields.*", EXACT),
    ("extraction_results.*.fields.*.field_id", STABLE_ID),
    ("extraction_results.*.batch_executions.*", EXACT),
    ("extraction_results.*.batch_executions.*.chunk_ids.*", STABLE_ID),
    ("extraction_results.*.batch_executions.*.prompt_tokens", PERMITTED_DYNAMIC),
    ("extraction_results.*.batch_executions.*.completion_tokens", PERMITTED_DYNAMIC),
    ("extraction_results.*.batch_executions.*.total_tokens", PERMITTED_DYNAMIC),
    ("extraction_results.*.planned_chunk_ids_by_batch.*.*", STABLE_ID),

    # extraction_review_queue -- (A) exemplar branch.
    ("extraction_review_queue.*", EXACT),
    ("extraction_review_queue.*.record_id", STABLE_ID),
    ("extraction_review_queue.*.field_id", STABLE_ID),

    ("extraction_run_id", CANONICAL_ID),
    ("extraction_manifest_path", STRUCTURED_REF),
    ("extraction_manifest_hash", DERIVED_HASH),
    ("extraction_artifact_paths.*", STRUCTURED_REF),
    ("extraction_schema_hash", PROVENANCE_HASH),

    # quality_appraisal_results -- (A) exemplar branch (empty/None per (B)).
    ("quality_appraisal_results.*", EXACT),
    ("quality_appraisal_results.*.run_id", CANONICAL_ID),
    ("quality_appraisal_results.*.manifest_hash", DERIVED_HASH),

    ("quality_appraisal_run_id", CANONICAL_ID),
    ("quality_appraisal_manifest_path", STRUCTURED_REF),
    ("quality_appraisal_manifest_hash", DERIVED_HASH),
    ("quality_appraisal_artifact_paths.*", STRUCTURED_REF),

    # screening_qc -- {"irr": {...}, "saturation": {...}} (screening_qc.py:573)
    ("screening_qc", EXACT),
    ("screening_qc.irr.producing_event_ids.*", DERIVED_ID),
    ("screening_qc.irr.sampled_record_ids.*", STABLE_ID),
    ("screening_qc.saturation", EXACT),

    # SynthesisOutput (§8.7)
    ("synthesis", EXACT),
    ("synthesis.extraction_run_id", CANONICAL_ID),
    ("synthesis.quality_appraisal_run_id", CANONICAL_ID),

    ("synthesis.map_outcome_records.*", EXACT),
    ("synthesis.map_outcome_records.*.record_id", STABLE_ID),
    ("synthesis.map_outcome_records.*.producing_event_id", DERIVED_ID),

    ("synthesis.map_answers.*", EXACT),
    ("synthesis.map_answers.*.answer_id", DERIVED_ID),
    ("synthesis.map_answers.*.record_id", STABLE_ID),
    ("synthesis.map_answers.*.producing_event_id", DERIVED_ID),
    ("synthesis.map_answers.*.selectors.*", EXACT),
    ("synthesis.map_answers.*.selectors.*.field_id", STABLE_ID),
    ("synthesis.map_answers.*.evidence_references.*", EXACT),
    ("synthesis.map_answers.*.evidence_references.*.artifact_run_id", CANONICAL_ID),
    ("synthesis.map_answers.*.evidence_references.*.record_id", STABLE_ID),
    ("synthesis.map_answers.*.evidence_references.*.field_id", STABLE_ID),
    ("synthesis.map_answers.*.evidence_references.*.evidence_id", DERIVED_ID),

    ("synthesis.claims.*", EXACT),
    ("synthesis.claims.*.claim_id", DERIVED_ID),
    ("synthesis.claims.*.producing_event_id", DERIVED_ID),
    ("synthesis.claims.*.supporting_answer_ids.*", DERIVED_ID),
    ("synthesis.claims.*.references.*", EXACT),
    ("synthesis.claims.*.references.*.artifact_run_id", CANONICAL_ID),
    ("synthesis.claims.*.references.*.record_id", STABLE_ID),
    ("synthesis.claims.*.references.*.field_id", STABLE_ID),
    ("synthesis.claims.*.references.*.evidence_id", DERIVED_ID),

    ("synthesis.rejected_targets.*", EXACT),

    ("synthesis.refine_batch_outcomes.*", EXACT),
    ("synthesis.refine_batch_outcomes.*.producing_event_id", DERIVED_ID),
    ("synthesis.refine_batch_outcomes.*.batch_answer_ids.*", DERIVED_ID),
    ("synthesis.refine_batch_outcomes.*.pre_batch_claim_states.*", EXACT),
    ("synthesis.refine_batch_outcomes.*.pre_batch_claim_states.*.producing_event_id", DERIVED_ID),
    ("synthesis.refine_batch_outcomes.*.pre_batch_claim_states.*.supporting_answer_ids.*", DERIVED_ID),
    ("synthesis.refine_batch_outcomes.*.applied_updates.*", EXACT),
    ("synthesis.refine_batch_outcomes.*.applied_updates.*.final_support_answer_ids.*", DERIVED_ID),

    ("synthesis.claim_construction_coverage.*", EXACT),
    ("synthesis.claim_construction_coverage.*.unincorporated_answer_ids.*", DERIVED_ID),

    # imrad_section_outcomes -- (A) exemplar branch only (IMRAD does not
    # fire in Track 2).
    ("synthesis.imrad_section_outcomes.*", EXACT),
    ("synthesis.imrad_section_outcomes.*.producing_event_id", DERIVED_ID),

    ("prisma_counts", EXACT),

    ("search_log.entries.*", EXACT),

    ("requires_human_review", EXACT),
    ("human_feedback", PERMITTED_DYNAMIC),
    ("prisma_diagram_path", STRUCTURED_REF),
    ("methods_text", TEMPLATED_TEXT),
    ("manual_worklist_path", STRUCTURED_REF),

    ("review_run_id", CANONICAL_ID),
    ("selection_decision_ledger_dir", STRUCTURED_REF),

    ("oversight_mode.*", EXACT),
    ("automation_origin.*", EXACT),

    ("ai_traceability_json_path", STRUCTURED_REF),
    ("ai_traceability_md_path", STRUCTURED_REF),
    ("quality_appraisal_csv_path", STRUCTURED_REF),
    ("quality_appraisal_json_path", STRUCTURED_REF),
    ("evidence_map_csv_path", STRUCTURED_REF),
    ("synthesis_claims_json_path", STRUCTURED_REF),
    ("synthesis_json_path", STRUCTURED_REF),
    ("synthesis_md_path", STRUCTURED_REF),
    ("export_manifest_path", STRUCTURED_REF),

    # Step 14 ([C13], PRISMA-ScR renderers) -- prisma_scr_report_md mirrors
    # methods_text exactly (templated prose over run-state values, same
    # renderer family). prisma_scr_conformance is a small ambient dict
    # (`{"items": [...]}`, same treatment as screening_qc above) -- EXACT
    # except its one optional embedded locator field, which gets the same
    # STRUCTURED_REF override every other embedded-path field in this table
    # gets.
    ("prisma_scr_report_md", TEMPLATED_TEXT),
    ("prisma_scr_conformance", EXACT),
    ("prisma_scr_conformance.items.*", EXACT),
    ("prisma_scr_conformance.items.*.locator", STRUCTURED_REF),
    ("prisma_scr_report_sha256", DERIVED_HASH),
    ("prisma_scr_conformance_sha256", DERIVED_HASH),
]

# --- External stores actually implemented at this step (§8.6, §8.8.3, §8.9) -

_AI_USE_EVENT_PAIRS: List[Tuple[str, str]] = [
    ("", EXACT),
    ("review_run_id", CANONICAL_ID),
    ("event_id", DERIVED_ID),
    ("work_item_id", DERIVED_ID),
    ("access_info", ENDPOINT_URL),
    ("prompt_template_hash", PROVENANCE_HASH),
    ("rendered_prompt_hash", DERIVED_HASH),
    ("input_artifact_refs.*", ARTIFACT_REF),
    ("output_artifact_refs.*", ARTIFACT_REF),
    ("semantic_hash", DERIVED_HASH),
    ("content_hash", DERIVED_HASH),
    ("timestamp", PERMITTED_DYNAMIC),
]

_SELECTION_DECISION_RECORD_PAIRS: List[Tuple[str, str]] = [
    ("", EXACT),
    ("review_run_id", CANONICAL_ID),
    ("decision_id", CANONICAL_ID),
    ("source_event_id", DERIVED_ID),
    ("producer_artifact_id", DERIVED_ID),
    ("record_id", STABLE_ID),
    ("supersedes_decision_id", CANONICAL_ID),
    ("timestamp", PERMITTED_DYNAMIC),
    ("semantic_hash", DERIVED_HASH),
    ("content_hash", DERIVED_HASH),
]

_SCREENING_BATCH_RESULT_PAIRS: List[Tuple[str, str]] = [
    ("", EXACT),
    ("review_run_id", CANONICAL_ID),
    ("work_item_id", DERIVED_ID),
    ("batch_fingerprint", DERIVED_ID),
    ("artifact_id", DERIVED_ID),
    ("record_ids.*", STABLE_ID),
    ("producing_event_id", DERIVED_ID),
    ("content_hash", DERIVED_HASH),
]

REVIEW_STATE_PATTERNS = _build_patterns(_REVIEW_STATE_PAIRS)
AI_USE_EVENT_PATTERNS = _build_patterns(_AI_USE_EVENT_PAIRS)
SELECTION_DECISION_RECORD_PATTERNS = _build_patterns(_SELECTION_DECISION_RECORD_PAIRS)
SCREENING_BATCH_RESULT_PATTERNS = _build_patterns(_SCREENING_BATCH_RESULT_PAIRS)

# Namespaced so `scientific_projection.json` is ONE frozen artifact even
# though coverage is checked per namespace against that namespace's own
# synthetic/exemplar corpus (a table entry nothing in this step's corpus
# ever visits would otherwise show up as spuriously "obsolete").
CLASSES: Dict[str, Dict[str, str]] = {
    "review_state": REVIEW_STATE_PATTERNS,
    "ai_use_event": AI_USE_EVENT_PATTERNS,
    "selection_decision_record": SELECTION_DECISION_RECORD_PATTERNS,
    "screening_batch_result": SCREENING_BATCH_RESULT_PATTERNS,
}


# ---------------------------------------------------------------------------
# The recursive classifier -- "most specific match wins, nearest-enclosing-
# default otherwise" (see module docstring for why this needs to be more
# than a flat leaf-path lookup: `config` is `exact` EXCEPT its nested
# `output_dir`, and a flat table can't express "exact, except...").
# ---------------------------------------------------------------------------

def _split(pattern: str) -> Tuple[str, ...]:
    return tuple(pattern.split(".")) if pattern else ()


@dataclass
class ProjectionResult:
    leaves: List[Tuple[Tuple[str, ...], Optional[str]]] = field(default_factory=list)
    used_patterns: set = field(default_factory=set)

    @property
    def unclassified(self) -> List[Tuple[str, ...]]:
        return [path for path, cls in self.leaves if cls is None]


def _match(pattern: Tuple[str, ...], path: Tuple[str, ...]) -> bool:
    if len(pattern) != len(path):
        return False
    return all(p == "*" or p == a for p, a in zip(pattern, path))


def _own_rule(patterns: Dict[Tuple[str, ...], str], path: Tuple[str, ...]) -> Optional[Tuple[Tuple[str, ...], str]]:
    candidates = [(pat, cls) for pat, cls in patterns.items() if _match(pat, path)]
    if not candidates:
        return None
    candidates.sort(key=lambda pc: pc[0].count("*"))
    if len(candidates) > 1 and candidates[0][0].count("*") == candidates[1][0].count("*"):
        raise ValueError(f"path {'.'.join(path)!r} is doubly-classified by "
                          f"{'.'.join(candidates[0][0])!r} and {'.'.join(candidates[1][0])!r}")
    return candidates[0]


def _has_deeper_rule(patterns: Dict[Tuple[str, ...], str], path: Tuple[str, ...]) -> bool:
    n = len(path)
    return any(len(pat) > n and _match(pat[:n], path) for pat in patterns)


def classify(namespace_patterns: Dict[str, str], obj: Any) -> ProjectionResult:
    """Walk `obj` (a plain JSON-shaped dict/list/scalar tree -- an exemplar
    fixture, or a `model_dump(mode="json")` of a real model instance) and
    classify every leaf using `namespace_patterns` (one of `CLASSES`'
    values). List indices AND dynamic dict keys both collapse to the SAME
    `"*"` wildcard segment when matching, since JSON arrays carry no key
    names and several `ReviewState` dicts (`oversight_mode`, `screening_qc`
    sub-blocks, `*_artifact_paths`) are keyed by a set that isn't fixed in
    the schema."""
    patterns = {_split(p): cls for p, cls in namespace_patterns.items()}
    result = ProjectionResult()

    def walk(value: Any, path: Tuple[str, ...], ambient: Optional[Tuple[Tuple[str, ...], str]]) -> None:
        own = _own_rule(patterns, path)
        effective = own if own is not None else ambient
        if own is not None and not _has_deeper_rule(patterns, path):
            result.used_patterns.add(own[0])
            result.leaves.append((path, own[1]))
            return
        if isinstance(value, dict) and value:
            for k, v in value.items():
                walk(v, path + (k,), effective)
            return
        if isinstance(value, list) and value:
            for item in value:
                walk(item, path + ("*",), effective)
            return
        if isinstance(value, list) and not value:
            # An empty list has no element to walk into, but a per-element
            # rule one level down (the common `"path.*"` shape) still tells
            # us what an element WOULD be classified as -- use it rather
            # than falling back to the enclosing ambient default, so an
            # exemplar where e.g. `deduplicated_records` happens to be
            # empty doesn't misclassify it as its container's `exact`
            # default instead of "no Records here, but if there were any,
            # they'd classify per `_record_field_patterns`".
            elem_rule = _own_rule(patterns, path + ("*",))
            if elem_rule is not None:
                result.used_patterns.add(elem_rule[0])
                result.leaves.append((path, elem_rule[1]))
                return
        # scalar, or an empty dict/list with no per-element rule -- a
        # genuine leaf for classification purposes (nothing further to
        # recurse into).
        if effective is not None:
            result.used_patterns.add(effective[0])
            result.leaves.append((path, effective[1]))
        else:
            result.leaves.append((path, None))

    walk(obj, (), None)
    return result


def assert_full_coverage(namespace_patterns: Dict[str, str], *objs: Any) -> ProjectionResult:
    """Runs `classify` over every `objs` entry and asserts (per §8.1's (A))
    zero unclassified leaves and zero obsolete patterns across the COMBINED
    corpus -- a pattern only one exemplar in a multi-exemplar corpus visits
    is fine; a pattern NONE of them visit is a stale rule."""
    combined = ProjectionResult()
    for obj in objs:
        r = classify(namespace_patterns, obj)
        combined.leaves.extend(r.leaves)
        combined.used_patterns |= r.used_patterns
    unclassified = combined.unclassified
    if unclassified:
        raise AssertionError("unclassified leaf path(s): " + ", ".join(".".join(p) for p in unclassified))
    all_patterns = {_split(p) for p in namespace_patterns}
    obsolete = all_patterns - combined.used_patterns
    if obsolete:
        raise AssertionError("obsolete pattern(s), never matched: " + ", ".join(".".join(p) for p in sorted(obsolete)))
    return combined


# ---------------------------------------------------------------------------
# 8.4 (points 1-2 only -- see module docstring) -- run-identity
# canonicalisation, the part `test_projection_fk_isomorphism` needs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunIdentity:
    review_run_id: str
    extraction_run_id: Optional[str] = None
    quality_appraisal_run_id: Optional[str] = None
    ft_screening_run_id: Optional[str] = None


def canonicalize_run_ids(run: RunIdentity) -> Dict[str, str]:
    """One run's own raw ids -> their shared `"CID:*"` placeholder (§8.4
    points 1-2). `review_run_id` IS `thread_id` (verified `run_identity.py:19`
    -- see this module's own docstring) -- there is exactly one placeholder
    for both."""
    mapping = {run.review_run_id: "CID:run"}
    if run.extraction_run_id is not None:
        mapping[run.extraction_run_id] = "CID:extraction_run"
    if run.quality_appraisal_run_id is not None:
        mapping[run.quality_appraisal_run_id] = "CID:qa_run"
    if run.ft_screening_run_id is not None:
        mapping[run.ft_screening_run_id] = "CID:ft_run"
    return mapping


def pair_canonical_ids(run_a: RunIdentity, run_b: RunIdentity) -> Dict[str, str]:
    """`objs_A -> objs_B` direct pairing for the whole-run identity fields
    (§8.4 points 1-2) -- the two runs must agree on WHICH optional run ids
    are present (both `None` or both set), or the projection fails (a real
    determinism violation, per §8.4's own "a length or semantic-identity
    mismatch -> the projection fails")."""
    for name, a, b in (
        ("extraction_run_id", run_a.extraction_run_id, run_b.extraction_run_id),
        ("quality_appraisal_run_id", run_a.quality_appraisal_run_id, run_b.quality_appraisal_run_id),
        ("ft_screening_run_id", run_a.ft_screening_run_id, run_b.ft_screening_run_id),
    ):
        if (a is None) != (b is None):
            raise ValueError(f"{name} presence disagrees between the two runs: {a!r} vs {b!r}")
    pairs = {run_a.review_run_id: run_b.review_run_id}
    if run_a.extraction_run_id is not None:
        pairs[run_a.extraction_run_id] = run_b.extraction_run_id
    if run_a.quality_appraisal_run_id is not None:
        pairs[run_a.quality_appraisal_run_id] = run_b.quality_appraisal_run_id
    if run_a.ft_screening_run_id is not None:
        pairs[run_a.ft_screening_run_id] = run_b.ft_screening_run_id
    return pairs


# ---------------------------------------------------------------------------
# 8.10 -- the named derivation functions, reused verbatim (never
# reimplemented) so "recomputes" means calling the SAME function production
# calls, not a parallel copy that could silently drift from it.
# ---------------------------------------------------------------------------

DERIVATIONS: Dict[str, Any] = {
    "MapAnswer.answer_id": compute_answer_id,
    "SynthesisClaim.claim_id": _compute_claim_id,
    "AIUseEvent.event_id": compute_event_id,
    "EvidenceReference.evidence_id": make_evidence_id,
    "ScreeningBatchResult.work_item_id": compute_screening_work_item_id,
    "SelectionDecisionRecord.source_event_id": derive_source_event_id,
}
