"""
synthesis_node -- Step 6 of the SLR pipeline.

Structured, evidence-verified claims and traceability -- the Stage 3 claim-
construction slice of quality_appraisal_synthesis_plan_2026-08-22.md's own
§3.2 ("Structural traceability, not claim validation"). Complementary
outputs:
  1. Evidence map -- descriptive/numerical summary of study characteristics
     (PRISMA-ScR's required "map of the field" for scoping reviews).
     UNCHANGED, pre-existing.
  2. BERTopic thematic clustering -- inductive, exploratory complement to
     the deductive per-target synthesis below. UNCHANGED, pre-existing.
  3. Structured, claim-backed synthesis -- a map/refine pipeline over
     research questions (and, optionally, an "imrad:results" target): each
     included study is asked to answer each target on its own (map step,
     grounded only in that study's own charted/appraised data, citing the
     exact extraction/quality-appraisal field(s) it drew on), then the
     per-study answers are folded into a running set of claim candidates
     one batch at a time (refine step). Every claim is bound to its own
     supporting evidence and independently source-verified at the end of
     the node (verify_synthesis_output_sources()) against the real,
     durable AIUseEvent each claim/answer/batch/section attempt actually
     produced -- never trusted merely because the persisted output is
     internally self-consistent.
  4. (Optional, off by default) legacy free-text IMRAD narrative --
     Introduction/Methods/Discussion, kept for review families that want
     traditional full-paper prose rather than claim-backed synthesis.

Node input keys:  config, full_text_records, extraction_run_id (used to
                   reload the verified extraction resolution artifact --
                   see nodes.extraction.load_verified_extraction_resolution_
                   artifact()), quality_appraisal_run_id (rapid/SLR only)
Node output keys: synthesis, oversight_mode, automation_origin, review_run_id
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError, field_validator, model_validator

from slr_pipeline.state import (ReviewState, SynthesisOutput, Record, MapAnswer, SynthesisClaim, RejectedSynthesisTarget,
                    MapOutcomeRecord, RefineBatchOutcomeRecord, ImradSectionOutcomeRecord, build_map_answer,
                    build_synthesis_claim, compute_answer_id, SupportingFieldSelector, EvidenceReference,
                    ClaimConstructionCoverage, RefineClaimSnapshot, RefineAppliedUpdate)
from slr_pipeline.config import ReviewConfig, SynthesisConfig, resolve_openai_api_key
from slr_pipeline.synthesis_evidence import make_evidence_id
from slr_pipeline.ai_traceability import AIUseEvent, AIUseEventStore, build_ai_use_event, reserved_attempt
from slr_pipeline.extraction.hashing import sha256_hex
from slr_pipeline.run_identity import resolve_review_run_id
from slr_pipeline.nodes.extraction import load_verified_extraction_resolution_artifact
from slr_pipeline.extraction import verified_bundle
from slr_pipeline.nodes.quality_appraisal import (load_verified_quality_appraisal_resolution_artifact, DocumentAppraisal,
                                      TEACHING_APPRAISAL_ITEMS, ROB2_DOMAINS)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception taxonomy, protected response-reading helpers, and the shared
# internal-failure wrapper
# ---------------------------------------------------------------------------

class MalformedModelResponseError(ValueError):
    """The ONE exception type meaning 'the provider's own response, or its content, did not fit
    what was expected' -- a genuinely expected, routine data condition."""


class HallucinatedSelectorError(MalformedModelResponseError):
    """Raised, and ONLY raised, at the exact point a MapResponse's own supporting_selectors is
    found not to be a subset of citeable_selectors."""


class SynthesisInternalConsistencyError(RuntimeError):
    """The ONE exception type meaning 'a structural invariant that should be impossible under
    correct operation was violated, or an unexpected internal defect occurred while processing a
    response the provider genuinely returned.' A call site that catches this still publishes its
    own physical call's event first -- honestly, as a genuine FAILURE, never a fabricated success
    -- then re-raises to abort the node."""


def _extract_response_snapshot(resp: Any) -> Optional[str]:
    try:
        model = resp.model
    except Exception:
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    return model


def _read_response_content(resp: Any) -> str:
    """The ONE place resp.choices[0].message.content is ever read. ANY failure here, OR a
    successfully-read but non-str value (pydantic's own model_validate_json() silently accepts
    bytes/bytearray too, which would otherwise let a genuinely anomalous response shape through
    unflagged), means the response object itself is malformed -- a genuine
    MalformedModelResponseError."""
    try:
        content = resp.choices[0].message.content
    except Exception as exc:
        raise MalformedModelResponseError(f"could not read response content: {exc}") from exc
    if not isinstance(content, str):
        raise MalformedModelResponseError(f"response content is not a string (got {type(content).__name__})")
    return content


def _compute_output_refs_or_internal_failure(
    *, compute_fn: Callable[[], List[str]], existing_internal_exc: Optional[SynthesisInternalConsistencyError],
    work_item_id: str,
) -> Tuple[List[str], str, Optional[str], Optional[SynthesisInternalConsistencyError]]:
    """The shared wrapper around output-ref CANONICALIZATION at both the map and refine call sites.
    `compute_fn` is a zero-arg closure building the real output_artifact_refs for a genuinely
    successful outcome. Returns (output_refs, call_status, failure_detail, internal_exc). On
    success: (compute_fn(), "success", None, existing_internal_exc). On an unexpected exception
    from compute_fn() itself: ([], "failure", f"internal_error: {exc}", a set internal_exc) --
    NEVER a raw-content hash under a fabricated "success"."""
    try:
        return compute_fn(), "success", None, existing_internal_exc
    except Exception as exc:
        internal_exc = existing_internal_exc or SynthesisInternalConsistencyError(
            f"unexpected internal error computing output_artifact_refs for work_item_id={work_item_id!r}: {exc}")
        return [], "failure", f"internal_error: {exc}", internal_exc


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET = 3000


def _chunk(items: List[Any], size: int) -> List[List[Any]]:
    if type(size) is not int or size < 1:
        raise ValueError(f"_chunk: size must be a positive integer, got {size!r} -- "
                          "SynthesisConfig.refine_batch_size should already reject this at "
                          "config-construction time; this is defense in depth")
    return [items[i:i + size] for i in range(0, len(items), size)]


_SYNTHESIS_RESULTS_TARGET_ID = "imrad:results"
_SYNTHESIS_RESULTS_TARGET_INSTRUCTION = (
    "Summarize this study's own key characteristics and findings that a systematic review's "
    "Results section should report -- for example its study design, population, and the "
    "outcomes it actually measured or reported."
)


def _target_display_text(rq_or_section: str) -> str:
    if rq_or_section == _SYNTHESIS_RESULTS_TARGET_ID:
        return _SYNTHESIS_RESULTS_TARGET_INSTRUCTION
    return rq_or_section


def _qa_field_label(checklist_id: str, field_id: str) -> str:
    if checklist_id == "teaching_appraisal_v1":
        item = next(i for i in TEACHING_APPRAISAL_ITEMS if i["item_id"] == field_id)
        return item["question"]
    domain_id, question_id = field_id.split(".", 1)
    domain = next(d for d in ROB2_DOMAINS if d["domain_id"] == domain_id)
    question = next(q for q in domain["questions"] if q["question_id"] == question_id)
    return question["question"]


# ---------------------------------------------------------------------------
# Canonical corpus ordering -- extracted verbatim from synthesis_node's own
# coverage-validation-and-reorder logic (aggregation/export/reporting plan
# v17, §2.1/X2/Y3). The ONE authoritative implementation: synthesis_node
# itself calls this (below) instead of its own former inline copy, and
# reporting's own evidence-map reconstruction (slr_pipeline/nodes/
# reporting.py) reuses it identically, parameterized on `error_prefix` so
# each caller's own error messages stay distinct while the underlying
# validation SEMANTICS are shared.
# ---------------------------------------------------------------------------

def canonicalize_corpus_order(*, full_text_records: List[Record], charted: List[Dict[str, Any]],
                               error_prefix: str) -> List[Record]:
    """Rejects duplicate record_ids on either side and any missing/foreign
    id, then reorders `full_text_records` into `charted`'s own canonical
    order. `error_prefix`: `synthesis_node`'s own call site passes its
    EXISTING `f"synthesis_node (review_type={cfg.review_type!r})"` string,
    so every message it raises is byte-identical to before this
    extraction; reporting's own call site passes a different prefix
    naming itself -- only the validation semantics (which fixture
    triggers which category of error) are shared across callers, not the
    exact message text."""
    record_ids = [r.id for r in full_text_records]
    duplicate_record_ids = sorted({rid for rid in record_ids if record_ids.count(rid) > 1})
    if duplicate_record_ids:
        raise ValueError(f"{error_prefix}: full_text_records contains duplicate record_id(s): {duplicate_record_ids}")
    charted_ids = [c["record_id"] for c in charted]
    duplicate_charted_ids = sorted({cid for cid in charted_ids if charted_ids.count(cid) > 1})
    if duplicate_charted_ids:
        raise ValueError(f"{error_prefix}: charted_data contains duplicate record_id(s): {duplicate_charted_ids}")
    record_id_set, charted_id_set = set(record_ids), set(charted_ids)
    missing, foreign = sorted(record_id_set - charted_id_set), sorted(charted_id_set - record_id_set)
    if missing or foreign:
        raise ValueError(f"{error_prefix}: charted_data must have EXACTLY one entry per full_text_records "
                          f"record_id -- missing={missing!r} foreign={foreign!r}")
    records_by_id = {r.id: r for r in full_text_records}
    return [records_by_id[c["record_id"]] for c in charted]


# ---------------------------------------------------------------------------
# Evidence map (PRISMA-ScR descriptive summary) -- UNCHANGED, pre-existing
# ---------------------------------------------------------------------------

def _citation_for(record: Record, raw_extraction: Dict[str, Any]) -> str:
    """Short traceable citation -- prefer the charted author_year, fall back to record metadata."""
    author_year = raw_extraction.get("author_year")
    if author_year:
        return str(author_year)

    from slr_pipeline.adapters.scoping_charting_adapter import derive_author_year
    return derive_author_year(record)


def _build_evidence_map(records: List[Record], charted_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive/numerical summary of included studies -- the standard PRISMA-ScR
    "characteristics of included sources" deliverable for scoping reviews."""
    by_year: Counter = Counter()
    by_design: Counter = Counter()
    by_country: Counter = Counter()
    characteristics_table: List[Dict[str, Any]] = []

    for record in records:
        raw = charted_by_id.get(record.id, {})
        by_year[record.year or "unknown"] += 1
        by_design[raw.get("study_design") or "unreported"] += 1
        by_country[raw.get("country") or "unreported"] += 1
        characteristics_table.append({
            "citation": _citation_for(record, raw),
            "year": record.year,
            "country": raw.get("country"),
            "study_design": raw.get("study_design"),
            "population": raw.get("population"),
            "concept": raw.get("concept"),
            "context": raw.get("context"),
        })

    return {
        "n_studies": len(records),
        "by_year": dict(sorted(by_year.items(), key=lambda kv: str(kv[0]))),
        "by_study_design": dict(by_design.most_common()),
        "by_country": dict(by_country.most_common()),
        "characteristics_table": characteristics_table,
    }


# ---------------------------------------------------------------------------
# BERTopic thematic clustering -- UNCHANGED, pre-existing
# ---------------------------------------------------------------------------

def _run_bertopic(records: List[Record], n_topics: int, texts: List[str]) -> List[Dict[str, Any]]:
    """Cluster the given per-record `texts` with BERTopic; return topic summaries. `texts` must
    already be built from verified extracted content."""
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=max(5, len(records) // 20),
                                metric="euclidean", prediction_data=True)

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        result = []
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            result.append({
                "topic_id": int(tid),
                "label": str(row.get("Name", f"Topic {tid}")),
                "count": int(row["Count"]),
                "keywords": topic_model.get_topic(tid),
                "representative_docs": topic_model.get_representative_docs(tid)[:2],
            })
        return result
    except Exception as e:
        logger.warning(f"BERTopic failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Shared, deterministic map-input builder -- genuinely two-pass
# ---------------------------------------------------------------------------

def _build_map_context(*, record_id: str, raw_extraction: Dict[str, Any], extraction_field_order: List[str],
                        extraction_index: Dict[str, Dict[str, List[Any]]], doc_appraisal: Optional[DocumentAppraisal],
                        qa_index: Dict[str, Dict[str, List[Any]]],
                        char_budget: int) -> Tuple[List[SupportingFieldSelector], str]:
    candidates: List[Tuple[SupportingFieldSelector, str]] = []
    for field_id in extraction_field_order:
        value = raw_extraction.get(field_id)
        if value in (None, "", []):
            continue
        if not extraction_index.get(record_id, {}).get(field_id):
            continue
        candidates.append((SupportingFieldSelector(artifact_kind="extraction", field_id=field_id),
                            f"{field_id}: {value}"))
    if doc_appraisal is not None:
        for field_id, leaf in doc_appraisal.leaves().items():
            if not qa_index.get(record_id, {}).get(field_id):
                continue
            label = _qa_field_label(doc_appraisal.checklist_id, field_id)
            answer_or_status = getattr(leaf, "answer", None) or getattr(leaf, "status", None)
            candidates.append((SupportingFieldSelector(artifact_kind="quality_appraisal", field_id=field_id),
                                f"{field_id} ({label}): answer={answer_or_status}"))

    # PASS 1: pure greedy admit-if-it-fits, in deterministic candidate order. No truncation
    # happens here -- a candidate that doesn't fit is simply SKIPPED, never truncated, so it can
    # never consume budget a later, shorter candidate would have needed.
    selectors: List[SupportingFieldSelector] = []
    lines: List[str] = []
    total_len = 0
    for selector, line in candidates:
        added_len = len(line) + (1 if lines else 0)
        if total_len + added_len <= char_budget:
            selectors.append(selector)
            lines.append(line)
            total_len += added_len

    # PASS 2: fires ONLY if pass 1 found nothing at all -- i.e. every candidate, individually,
    # exceeds the entire budget. Truncates the FIRST candidate (same deterministic order) as a
    # last-resort fallback, so a record with real evidence is never mislabeled no_citeable_evidence.
    if not selectors and candidates:
        selector, line = candidates[0]
        selectors.append(selector)
        lines.append(line[:char_budget])

    return selectors, "\n".join(lines)


# ---------------------------------------------------------------------------
# Map call sequence
# ---------------------------------------------------------------------------

_SYNTHESIS_MAP_PROMPT_VERSION = "v1"
_SYNTHESIS_MAP_SYSTEM_PROMPT = (
    "You are assisting a systematic literature review. You will be shown one study's own "
    "extracted or appraised data, labeled by field id, and a research question or synthesis "
    "target. Decide whether this study's own data provides evidence relevant to the target. "
    'Respond with a JSON object of exactly this shape: {"relevant": true or false, "answer": '
    "a 1-2 sentence string present if and only if relevant is true, \"supporting_selectors\": a "
    'list of {"artifact_kind": "extraction" or "quality_appraisal", "field_id": the exact field '
    'id string} objects, present if and only if relevant is true}. Cite ONLY field ids that were '
    "actually shown to you below -- never invent one. If the study's data does not bear on the "
    "target, respond with relevant=false and omit answer and supporting_selectors entirely."
)
_SYNTHESIS_MAP_USER_TEMPLATE = (
    "Research question or synthesis target: {target}\n\n"
    "Study data (field_id: value):\n{rendered_context}"
)


class MapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevant: StrictBool
    answer: Optional[str] = None
    supporting_selectors: List[SupportingFieldSelector] = Field(default_factory=list)

    @field_validator("supporting_selectors")
    @classmethod
    def _no_duplicate_selectors(cls, v: List[SupportingFieldSelector]) -> List[SupportingFieldSelector]:
        keys = [(s.artifact_kind, s.field_id) for s in v]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate selector in supporting_selectors")
        return v

    @model_validator(mode="after")
    def _check(self) -> "MapResponse":
        if self.relevant:
            if not self.answer or not self.answer.strip():
                raise ValueError("relevant=True requires a non-blank answer")
            if not self.supporting_selectors:
                raise ValueError("relevant=True requires at least one supporting_selectors entry")
        elif self.answer is not None or self.supporting_selectors:
            raise ValueError("relevant=False must not carry answer/supporting_selectors")
        return self


def _canonical_map_response_payload(*, answer_text: str, selectors: List[SupportingFieldSelector]) -> Dict[str, Any]:
    ordered = sorted(selectors, key=lambda s: (s.artifact_kind, s.field_id))
    return {"relevant": True, "answer": answer_text,
            "supporting_selectors": [{"artifact_kind": s.artifact_kind, "field_id": s.field_id} for s in ordered]}


_SYNTHESIS_MAP_MODEL_CONFIG_ID = "slr-chapter-synthesis-claim-map-v1"


def _build_map_work_item(*, target: str, record_id: str, rendered_context: str, model: str
                          ) -> Tuple[str, List[Dict[str, str]], str, Dict[str, Any]]:
    """Builds the map call's own messages/rendered_prompt_hash/work_item_id -- the ONE authoritative
    implementation, reused identically by `_run_one_map_call()` (the real call site) and
    `verify_synthesis_output_sources()` (the verifier), so the two can never silently drift apart.
    Returns (work_item_id, messages, rendered_prompt_hash, prompt_parameters)."""
    messages = [{"role": "system", "content": _SYNTHESIS_MAP_SYSTEM_PROMPT},
                {"role": "user", "content": _SYNTHESIS_MAP_USER_TEMPLATE.format(
                    target=_target_display_text(target), rendered_context=rendered_context)}]
    prompt_parameters = {"temperature": 0.0, "response_format": {"type": "json_object"}}
    rendered_prompt_hash = sha256_hex(messages)
    expected_schema_hash = sha256_hex(MapResponse.model_json_schema())
    work_item_payload = {"model": model, "model_config_id": _SYNTHESIS_MAP_MODEL_CONFIG_ID,
                          "prompt_parameters": prompt_parameters, "prompt_template_version": _SYNTHESIS_MAP_PROMPT_VERSION,
                          "expected_schema_hash": expected_schema_hash, "rendered_prompt_hash": rendered_prompt_hash,
                          "rq_or_section": target, "record_id": record_id}
    work_item_id = f"synthesis_claim_map:{sha256_hex(work_item_payload)}"
    return work_item_id, messages, rendered_prompt_hash, prompt_parameters


def _run_one_map_call(*, target: str, record: Record, citeable_selectors: List[SupportingFieldSelector],
                       rendered_context: str, client: OpenAI, cfg: ReviewConfig, review_run_id: str,
                       event_store: AIUseEventStore
                       ) -> Tuple[str, Optional[MapResponse], bool, "AIUseEvent", Optional[SynthesisInternalConsistencyError]]:
    """Returns (outcome, parsed_or_None, hallucinated, published_event, internal_exc_or_None). The
    event is ALWAYS published before this function returns, in every reachable code path."""
    work_item_id, messages, rendered_prompt_hash, prompt_parameters = _build_map_work_item(
        target=target, record_id=record.id, rendered_context=rendered_context, model=cfg.synthesis.llm_model)

    with reserved_attempt(event_store, stage="synthesis_claim_map", work_item_id=work_item_id) as attempt:
        resp_model: Optional[str] = None
        parsed: Optional[MapResponse] = None
        hallucinated = False
        internal_exc: Optional[SynthesisInternalConsistencyError] = None

        try:
            resp = client.chat.completions.create(model=cfg.synthesis.llm_model, messages=messages,
                                                    temperature=0.0, response_format={"type": "json_object"})
        except Exception as exc:
            outcome, failure_detail = "call_failed", f"transport_error: {exc}"
        else:
            resp_model = _extract_response_snapshot(resp)
            try:
                raw_content = _read_response_content(resp)
            except MalformedModelResponseError as exc:
                outcome, failure_detail = "invalid_response", f"invalid_response: {exc}"
            else:
                try:
                    parsed = MapResponse.model_validate_json(raw_content)
                    selector_keys = {(s.artifact_kind, s.field_id) for s in parsed.supporting_selectors}
                    citeable_keys = {(s.artifact_kind, s.field_id) for s in citeable_selectors}
                    if parsed.relevant and not (selector_keys <= citeable_keys):
                        raise HallucinatedSelectorError(
                            f"supporting_selectors {sorted(selector_keys - citeable_keys)} not a "
                            "subset of citeable_selectors")
                except HallucinatedSelectorError as exc:
                    outcome, hallucinated, parsed = "invalid_response", True, None
                    failure_detail = f"invalid_response: hallucinated_selector: {exc}"
                except (ValidationError, json.JSONDecodeError) as exc:
                    outcome, parsed = "invalid_response", None
                    failure_detail = f"invalid_response: {exc}"
                except Exception as exc:
                    # A genuine, unexpected internal defect during PARSING is NOT a successful
                    # call -- classified as a real failure, never a fabricated success carrying a
                    # raw-content hash.
                    outcome, parsed = "call_failed", None
                    failure_detail = f"internal_error: {exc}"
                    internal_exc = SynthesisInternalConsistencyError(
                        f"unexpected internal error processing a received response for "
                        f"work_item_id={work_item_id!r}: {exc}")
                else:
                    outcome = "relevant" if parsed.relevant else "not_relevant"
                    failure_detail = None

        component_versions = {cfg.synthesis.llm_model: resp_model} if resp_model is not None else {}
        if outcome == "relevant":
            output_refs, call_status, refs_failure_detail, internal_exc = _compute_output_refs_or_internal_failure(
                compute_fn=lambda: [f"sha256:{sha256_hex(_canonical_map_response_payload(answer_text=parsed.answer, selectors=parsed.supporting_selectors))}"],
                existing_internal_exc=internal_exc, work_item_id=work_item_id)
            if refs_failure_detail is not None:
                failure_detail = refs_failure_detail
        elif outcome == "not_relevant":
            output_refs, call_status, refs_failure_detail, internal_exc = _compute_output_refs_or_internal_failure(
                compute_fn=lambda: [f"sha256:{sha256_hex({'relevant': False})}"],
                existing_internal_exc=internal_exc, work_item_id=work_item_id)
            if refs_failure_detail is not None:
                failure_detail = refs_failure_detail
        else:
            call_status, output_refs = "failure", []

        event = build_ai_use_event(
            review_run_id=review_run_id, stage="synthesis_claim_map", work_item_id=work_item_id, attempt=attempt,
            tool_kind="llm", model=cfg.synthesis.llm_model, provider="openai", component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose="Determine whether one study's own extracted/appraised evidence answers one "
                         "research question or IMRAD target, citing which fields it drew on",
            model_config_id=_SYNTHESIS_MAP_MODEL_CONFIG_ID, prompt_template_version=_SYNTHESIS_MAP_PROMPT_VERSION,
            prompt_template_text=json.dumps([{"role": "system", "content": _SYNTHESIS_MAP_SYSTEM_PROMPT},
                                              {"role": "user", "content": _SYNTHESIS_MAP_USER_TEMPLATE}]),
            prompt_parameters=prompt_parameters, rendered_prompt_hash=rendered_prompt_hash,
            input_artifact_refs=[f"target:{target}", f"record:{record.id}:sha256:{sha256_hex(rendered_context)}"],
            output_artifact_refs=output_refs, call_status=call_status, failure_detail=failure_detail,
        )
        event = event_store.publish(event)
    return outcome, parsed, hallucinated, event, internal_exc


# ---------------------------------------------------------------------------
# _references_for_selectors() -- the fatal-vs-rejected boundary
# ---------------------------------------------------------------------------

def _references_for_selectors(selectors: List[SupportingFieldSelector], record_id: str, extraction_index, qa_index,
                               extraction_run_id: str, qa_run_id: Optional[str]) -> List[EvidenceReference]:
    refs = []
    for selector in selectors:
        index = qa_index if selector.artifact_kind == "quality_appraisal" else extraction_index
        run_id = qa_run_id if selector.artifact_kind == "quality_appraisal" else extraction_run_id
        entries = index.get(record_id, {}).get(selector.field_id, [])
        if not entries:
            raise SynthesisInternalConsistencyError(
                f"selector {selector!r} was offered as citeable for record {record_id!r} moments ago "
                "in this same map call but resolved to zero evidence_index entries")
        refs.extend(EvidenceReference(artifact_kind=selector.artifact_kind, artifact_run_id=run_id,
                                       record_id=record_id, field_id=selector.field_id,
                                       evidence_id=e.evidence_id) for e in entries)
    return refs


# ---------------------------------------------------------------------------
# Refine call sequence
# ---------------------------------------------------------------------------

_SYNTHESIS_REFINE_PROMPT_VERSION = "v1"
_SYNTHESIS_REFINE_MODEL_CONFIG_ID = "slr-chapter-synthesis-claim-refine-v1"
_SYNTHESIS_REFINE_PROMPT_PARAMETERS = {"temperature": 0.2, "response_format": {"type": "json_object"}}
_SYNTHESIS_REFINE_SYSTEM_PROMPT = (
    "You are drafting the synthesis section of a systematic literature review, one batch of new "
    "per-study evidence at a time. You will see the CURRENT set of claim candidates (their own "
    "index, current text, and current supporting evidence) and a NEW batch of per-study answers, "
    "each with its own answer_id. For each new answer, decide whether it extends an EXISTING claim "
    "-- cite its claim_index and restate that claim's text plus its COMPLETE final list of "
    "supporting answer_ids, drawn only from that claim's own previously-shown supporting answer_ids "
    "plus this batch's own new answer_ids -- or starts a NEW claim (claim_index null, "
    "final_support_answer_ids drawn only from this batch's own new answer_ids). You do not need to "
    "update every existing claim, and may respond with an empty claim_updates list if this batch's "
    'evidence does not warrant any claim change. Respond with a JSON object of exactly this shape: '
    '{"claim_updates": [{"claim_index": an integer or null, "claim_text": a string, '
    '"final_support_answer_ids": a list of answer_id strings}, ...]} (claim_updates may be empty).'
)
_SYNTHESIS_REFINE_USER_TEMPLATE = (
    "Research question or synthesis target: {target}\n\n"
    "Current claim candidates:\n{existing_claims_block}\n\n"
    "New batch of per-study answers:\n{new_answers_block}"
)


class RefineClaimUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_index: Optional[StrictInt] = None
    claim_text: str = Field(min_length=1)
    final_support_answer_ids: List[str] = Field(min_length=1)

    @field_validator("claim_text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim_text must not be blank/whitespace-only")
        return v

    @field_validator("final_support_answer_ids")
    @classmethod
    def _no_dupes(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("duplicate answer_id in final_support_answer_ids")
        return v


class RefineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_updates: List[RefineClaimUpdate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> "RefineResponse":
        targets = [u.claim_index for u in self.claim_updates if u.claim_index is not None]
        if len(targets) != len(set(targets)):
            raise ValueError("two updates in one response target the same existing claim_index")
        return self


def _canonical_refine_update_payload(claim_text: str, supporting_answer_ids: List[str]) -> Dict[str, Any]:
    """Reused identically at publish time (per-update hash) and at verification time (reconstructing
    the expected hash from a persisted claim's own CURRENT state) -- the ONE canonical shape for
    'this exact claim content was produced by this exact refine response'."""
    return {"claim_text": claim_text, "final_support_answer_ids": sorted(supporting_answer_ids)}


def _canonical_refine_response_payload(claim_updates: List[Any]) -> Dict[str, Any]:
    """The COMPLETE canonical response -- claim_index, update ORDERING, and the genuinely-empty
    case are all preserved here, unlike _canonical_refine_update_payload() above, which
    deliberately discards ordering/claim_index. This is the ONE hash every successful refine
    event's own output_artifact_refs begins with, whether or not the batch produced any updates.
    Accepts any list of objects exposing .claim_index/.claim_text/.final_support_answer_ids --
    reused identically for the live RefineResponse.claim_updates (publish time) and the persisted
    RefineAppliedUpdate list (verification time, reconstructed from RefineBatchOutcomeRecord)."""
    return {"claim_updates": [{"claim_index": u.claim_index, "claim_text": u.claim_text,
                                "final_support_answer_ids": sorted(u.final_support_answer_ids)}
                               for u in claim_updates]}


class _ClaimAccumulator:
    def __init__(self, claim_text: str, supporting_answer_ids: List[str], producing_event_id: Optional[str] = None):
        self.claim_text = claim_text
        self.supporting_answer_ids = supporting_answer_ids   # always kept SORTED
        self.producing_event_id = producing_event_id   # the LAST refine event's own event_id that
                                                          # set THIS accumulator's current state --
                                                          # None only before any batch has ever
                                                          # touched it (impossible for a fully-
                                                          # finalized accumulator, see below)


def _validate_and_simulate(response: RefineResponse, batch_answer_ids: set,
                            accumulators: List[_ClaimAccumulator]) -> Tuple[List[Tuple[str, tuple]], set]:
    """PHASE 1 -- pure validation and simulation, ZERO mutation. Returns (simulated_state,
    touched_indices) -- touched_indices is exactly the set of indices in the RESULTING simulated
    list that THIS batch's own response created or modified (used to set producing_event_id only
    for those accumulators -- an untouched accumulator keeps its own PRIOR producing_event_id)."""
    for u in response.claim_updates:
        if u.claim_index is None:
            allowed = batch_answer_ids
        else:
            if not (0 <= u.claim_index < len(accumulators)):
                raise MalformedModelResponseError(f"claim_index {u.claim_index} out of range 0..{len(accumulators) - 1}")
            allowed = batch_answer_ids | set(accumulators[u.claim_index].supporting_answer_ids)
        bad = set(u.final_support_answer_ids) - allowed
        if bad:
            raise MalformedModelResponseError(f"final_support_answer_ids references id(s) {bad} outside the allowed set")

    simulated = [(acc.claim_text, tuple(sorted(acc.supporting_answer_ids))) for acc in accumulators]
    touched_indices: set = set()
    for u in response.claim_updates:
        entry = (u.claim_text, tuple(sorted(set(u.final_support_answer_ids))))
        if u.claim_index is None:
            simulated.append(entry)
            touched_indices.add(len(simulated) - 1)
        else:
            simulated[u.claim_index] = entry
            touched_indices.add(u.claim_index)

    dedup_check = [(text, frozenset(ids)) for text, ids in simulated]
    if len(dedup_check) != len(set(dedup_check)):
        raise MalformedModelResponseError("this batch's updates would produce two claims with identical "
                                           "claim_text and final_support_answer_ids")
    return simulated, touched_indices


def _commit_simulated_state(accumulators: List[_ClaimAccumulator], simulated: List[Tuple[str, tuple]],
                             touched_indices: set, producing_event_id: str) -> None:
    """PHASE 4 -- pure mutation, applying an ALREADY-VALIDATED simulated state. Only accumulators
    at a TOUCHED index get this batch's own producing_event_id; every other, untouched accumulator
    keeps whatever producing_event_id it already had from a PRIOR batch."""
    new_accumulators = []
    for i, (text, ids) in enumerate(simulated):
        if i in touched_indices:
            new_accumulators.append(_ClaimAccumulator(text, list(ids), producing_event_id=producing_event_id))
        else:
            new_accumulators.append(_ClaimAccumulator(text, list(ids), producing_event_id=accumulators[i].producing_event_id))
    accumulators[:] = new_accumulators


def _citation_for_map_answer(answer: MapAnswer) -> str:
    """NOT the same function as, and never calling or being called by, the pre-existing,
    unmodified _citation_for(record, raw_extraction) above."""
    return answer.record_id


def _build_refine_work_item(*, target: str, batch_index: int, batch: List[MapAnswer],
                             pre_batch_claim_states: List[Tuple[str, List[str]]],
                             answer_lookup: Dict[str, MapAnswer], model: str
                             ) -> Tuple[str, List[Dict[str, str]], str, Dict[str, Any]]:
    """Builds the refine call's own messages/rendered_prompt_hash/work_item_id -- the ONE
    authoritative implementation, reused identically by `_run_one_refine_batch()` (the real call
    site, passing the LIVE `accumulators` list's own current state) and
    `verify_synthesis_output_sources()` (the verifier, passing the persisted
    `RefineBatchOutcomeRecord.pre_batch_claim_states`), so the two can never silently drift apart --
    closing codex code-review round 2 (2026-08-26) finding 1 (refine provenance not bound to the
    actual rendered call). `pre_batch_claim_states` is a plain, POSITIONAL (claim_text,
    supporting_answer_ids) tuple list -- one entry per accumulator, in the same order both callers
    use; `answer_lookup` must resolve every answer_id referenced by EITHER pre_batch_claim_states or
    `batch` (a target-wide lookup is sufficient -- answer_ids are globally unique). Returns
    (work_item_id, messages, rendered_prompt_hash, prompt_parameters)."""
    batch_answer_ids = {a.answer_id for a in batch}
    existing_claims_block = "\n".join(
        f"[{i}] {claim_text}\n  supported by: " + "; ".join(
            f"{aid} ({_citation_for_map_answer(answer_lookup[aid])}): {answer_lookup[aid].answer_text}"
            for aid in sorted(supporting_answer_ids))
        for i, (claim_text, supporting_answer_ids) in enumerate(pre_batch_claim_states)
    )
    new_answers_block = "\n".join(f"{a.answer_id} ({_citation_for_map_answer(a)}): {a.answer_text}" for a in batch)
    messages = [{"role": "system", "content": _SYNTHESIS_REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": _SYNTHESIS_REFINE_USER_TEMPLATE.format(
                    target=_target_display_text(target), existing_claims_block=existing_claims_block or "(none yet)",
                    new_answers_block=new_answers_block)}]
    prompt_parameters = _SYNTHESIS_REFINE_PROMPT_PARAMETERS
    rendered_prompt_hash = sha256_hex(messages)
    expected_schema_hash = sha256_hex(RefineResponse.model_json_schema())
    work_item_payload = {"model": model, "model_config_id": _SYNTHESIS_REFINE_MODEL_CONFIG_ID,
                          "prompt_parameters": prompt_parameters, "prompt_template_version": _SYNTHESIS_REFINE_PROMPT_VERSION,
                          "expected_schema_hash": expected_schema_hash, "rendered_prompt_hash": rendered_prompt_hash,
                          "rq_or_section": target, "batch_index": batch_index, "batch_answer_ids": sorted(batch_answer_ids)}
    work_item_id = f"synthesis_claim_refine:{sha256_hex(work_item_payload)}"
    return work_item_id, messages, rendered_prompt_hash, prompt_parameters


def _run_one_refine_batch(*, target: str, batch_index: int, batch: List[MapAnswer], answer_lookup: Dict[str, MapAnswer],
                           accumulators: List[_ClaimAccumulator], client: OpenAI, cfg: ReviewConfig,
                           review_run_id: str, event_store: AIUseEventStore
                           ) -> Tuple[str, "AIUseEvent", Optional[SynthesisInternalConsistencyError], bool,
                                      List[RefineClaimSnapshot], List[RefineAppliedUpdate]]:
    """Runs ONE refine batch. Returns (outcome, published_event, internal_exc_or_None, is_no_op,
    pre_batch_claim_states, applied_updates); mutates `accumulators` IN PLACE only for
    outcome == 'applied' AND internal_exc is None."""
    batch_answer_ids = {a.answer_id for a in batch}
    pre_batch_claim_states = [RefineClaimSnapshot(claim_text=acc.claim_text,
                                                    supporting_answer_ids=list(acc.supporting_answer_ids),
                                                    producing_event_id=acc.producing_event_id)
                               for acc in accumulators]
    work_item_id, messages, rendered_prompt_hash, prompt_parameters = _build_refine_work_item(
        target=target, batch_index=batch_index, batch=batch,
        pre_batch_claim_states=[(s.claim_text, s.supporting_answer_ids) for s in pre_batch_claim_states],
        answer_lookup=answer_lookup, model=cfg.synthesis.llm_model)

    with reserved_attempt(event_store, stage="synthesis_claim_refine", work_item_id=work_item_id) as attempt:
        resp_model: Optional[str] = None
        parsed: Optional[RefineResponse] = None
        simulated: Optional[List[Tuple[str, tuple]]] = None
        touched_indices: set = set()
        internal_exc: Optional[SynthesisInternalConsistencyError] = None
        is_no_op = False
        applied_updates: List[RefineAppliedUpdate] = []

        try:
            resp = client.chat.completions.create(model=cfg.synthesis.llm_model, messages=messages,
                                                    temperature=0.2, response_format={"type": "json_object"})
        except Exception as exc:
            outcome, failure_detail = "call_failed", f"transport_error: {exc}"
        else:
            resp_model = _extract_response_snapshot(resp)
            try:
                raw_content = _read_response_content(resp)
            except MalformedModelResponseError as exc:
                outcome, failure_detail = "invalid_response", f"invalid_response: {exc}"
            else:
                try:
                    parsed = RefineResponse.model_validate_json(raw_content)
                    simulated, touched_indices = _validate_and_simulate(parsed, batch_answer_ids, accumulators)
                except MalformedModelResponseError as exc:
                    outcome, failure_detail, simulated = "invalid_response", f"invalid_response: {exc}", None
                except (ValidationError, json.JSONDecodeError) as exc:
                    outcome, failure_detail, simulated = "invalid_response", f"invalid_response: {exc}", None
                except Exception as exc:
                    # A genuine, unexpected internal defect during PARSING is NOT a successful
                    # call -- classified as a real failure, never a fabricated success.
                    outcome, simulated = "call_failed", None
                    failure_detail = f"internal_error: {exc}"
                    internal_exc = SynthesisInternalConsistencyError(
                        f"unexpected internal error processing a received response for "
                        f"work_item_id={work_item_id!r}: {exc}")
                else:
                    outcome, failure_detail = "applied", None
                    is_no_op = (len(parsed.claim_updates) == 0)
                    applied_updates = [RefineAppliedUpdate(claim_index=u.claim_index, claim_text=u.claim_text,
                                                             final_support_answer_ids=u.final_support_answer_ids)
                                        for u in parsed.claim_updates]

        component_versions = {cfg.synthesis.llm_model: resp_model} if resp_model is not None else {}
        if outcome == "applied" and parsed is not None:
            # whole-response hash FIRST, then per-update hashes. Canonicalization failure is a
            # real failure, not a fabricated success -- via _compute_output_refs_or_internal_failure().
            def _compute_refine_output_refs() -> List[str]:
                response_hash = f"response:sha256:{sha256_hex(_canonical_refine_response_payload(parsed.claim_updates))}"
                update_hashes = [f"update:sha256:{sha256_hex(_canonical_refine_update_payload(u.claim_text, u.final_support_answer_ids))}"
                                  for u in parsed.claim_updates]
                return [response_hash] + update_hashes
            output_refs, call_status, refs_failure_detail, internal_exc = _compute_output_refs_or_internal_failure(
                compute_fn=_compute_refine_output_refs, existing_internal_exc=internal_exc, work_item_id=work_item_id)
            if refs_failure_detail is not None:
                failure_detail = refs_failure_detail
        else:
            call_status, output_refs = "failure", []

        event = build_ai_use_event(
            review_run_id=review_run_id, stage="synthesis_claim_refine", work_item_id=work_item_id, attempt=attempt,
            tool_kind="llm", model=cfg.synthesis.llm_model, provider="openai", component_versions=component_versions,
            access_info=str(client.base_url),
            task_purpose="Fold one batch of per-study claim-relevant answers into the running set of claim "
                         "candidates for one research question or IMRAD target",
            model_config_id=_SYNTHESIS_REFINE_MODEL_CONFIG_ID, prompt_template_version=_SYNTHESIS_REFINE_PROMPT_VERSION,
            prompt_template_text=json.dumps([{"role": "system", "content": _SYNTHESIS_REFINE_SYSTEM_PROMPT},
                                              {"role": "user", "content": _SYNTHESIS_REFINE_USER_TEMPLATE}]),
            prompt_parameters=prompt_parameters, rendered_prompt_hash=rendered_prompt_hash,
            input_artifact_refs=(
                # An explicit batch:{batch_index} reference -- without it, two RefineBatchOutcomeRecord
                # rows could swap their own recorded batch_index values while keeping their real
                # events/answer sets, and nothing would ever catch it.
                [f"target:{target}", f"batch:{batch_index}"] + [f"answer:{a.answer_id}" for a in batch]
                # Hash the accumulator's FULL state {claim_text, supporting_answer_ids} -- not
                # claim_text alone -- since the actual rendered prompt (existing_claims_block,
                # above) shows each accumulator's own supporting answer set, not just its text.
                + [f"claim:{i}:sha256:{sha256_hex({'claim_text': acc.claim_text, 'supporting_answer_ids': sorted(acc.supporting_answer_ids)})}"
                   for i, acc in enumerate(accumulators)]
            ),
            output_artifact_refs=output_refs, call_status=call_status, failure_detail=failure_detail,
        )
        event = event_store.publish(event)   # `accumulators` is STILL completely unmutated here in
                                               # every case, including the internal_exc case

    if internal_exc is None and outcome == "applied":
        _commit_simulated_state(accumulators, simulated, touched_indices, producing_event_id=event.event_id)
    return outcome, event, internal_exc, is_no_op, pre_batch_claim_states, applied_updates


def _finalize_target(*, target: str, map_answers: List[MapAnswer], cfg: ReviewConfig, client: OpenAI,
                      review_run_id: str, event_store: AIUseEventStore
                      ) -> Tuple[List[SynthesisClaim], Optional[RejectedSynthesisTarget], ClaimConstructionCoverage,
                                 List["AIUseEvent"], List[RefineBatchOutcomeRecord]]:
    target_answers = [a for a in map_answers if a.rq_or_section == target]
    answer_lookup = {a.answer_id: a for a in target_answers}
    accumulators: List[_ClaimAccumulator] = []
    refine_events: List[AIUseEvent] = []
    refine_batch_outcomes: List[RefineBatchOutcomeRecord] = []
    refine_batches_total = refine_batches_applied = refine_batches_applied_no_op = 0
    refine_batches_call_failed = refine_batches_invalid_response = 0

    for batch_index, batch in enumerate(_chunk(target_answers, cfg.synthesis.refine_batch_size)):
        outcome, event, internal_exc, is_no_op, pre_batch_claim_states, applied_updates = _run_one_refine_batch(
            target=target, batch_index=batch_index, batch=batch, answer_lookup=answer_lookup,
            accumulators=accumulators, client=client, cfg=cfg, review_run_id=review_run_id,
            event_store=event_store)
        refine_events.append(event)
        if internal_exc is not None:
            raise internal_exc   # abort the node -- AFTER the event is durably published
        # One addressable record per attempt, regardless of outcome -- built from the SAME
        # event.event_id already durably published above. batch_answer_ids preserves `batch`'s own
        # REAL order (the exact slice _chunk() produced) -- never sorted, since order is what the
        # topology validator's own exact-reconstruction check verifies against.
        # pre_batch_claim_states/applied_updates (codex code-review round 2 finding 1): the exact
        # accumulator state this batch's own call was rendered against, plus the model's own real
        # per-update response -- lets SynthesisOutput's own topology validator prove this batch's
        # own event was genuinely produced against THIS target's own real, prior batch sequence.
        refine_batch_outcomes.append(RefineBatchOutcomeRecord(
            rq_or_section=target, batch_index=batch_index,
            batch_answer_ids=[a.answer_id for a in batch],
            outcome=outcome, is_no_op=is_no_op, producing_event_id=event.event_id,
            pre_batch_claim_states=pre_batch_claim_states, applied_updates=applied_updates,
        ))
        refine_batches_total += 1
        if outcome == "applied":
            refine_batches_applied += 1
            if is_no_op:
                # An EXPLICIT flag returned by _run_one_refine_batch(), computed directly from
                # len(parsed.claim_updates) == 0. NEVER inferred from event.output_artifact_refs
                # == [] -- that shape is now impossible for a successful "applied" outcome, since
                # output_artifact_refs always carries at least the whole-response hash.
                refine_batches_applied_no_op += 1
        elif outcome == "call_failed":
            refine_batches_call_failed += 1
        else:
            refine_batches_invalid_response += 1

    all_relevant_answer_ids = {a.answer_id for a in target_answers}
    incorporated_answer_ids = {aid for acc in accumulators for aid in acc.supporting_answer_ids}
    unincorporated_answer_ids = sorted(all_relevant_answer_ids - incorporated_answer_ids)

    coverage = ClaimConstructionCoverage(
        rq_or_section=target, refine_batches_total=refine_batches_total,
        refine_batches_applied=refine_batches_applied, refine_batches_applied_no_op=refine_batches_applied_no_op,
        refine_batches_call_failed=refine_batches_call_failed, refine_batches_invalid_response=refine_batches_invalid_response,
        unincorporated_answer_ids=unincorporated_answer_ids,
    )

    if not accumulators:
        return [], RejectedSynthesisTarget(rq_or_section=target), coverage, refine_events, refine_batch_outcomes

    claims_for_target = []
    for acc in accumulators:
        contributing = [answer_lookup[aid] for aid in acc.supporting_answer_ids]
        claims_for_target.append(build_synthesis_claim(claim_text=acc.claim_text, contributing_answers=contributing,
                                                         rq_or_section=target, producing_event_id=acc.producing_event_id))
    return claims_for_target, None, coverage, refine_events, refine_batch_outcomes


# ---------------------------------------------------------------------------
# IMRAD legacy narrative (off by default)
# ---------------------------------------------------------------------------

_IMRAD_NARRATIVE_PROMPT_VERSION = "v1"
_IMRAD_NARRATIVE_SYSTEM_TEMPLATE = "You are writing the {section} section of a publication-quality literature review."
_IMRAD_NARRATIVE_USER_TEMPLATE = (
    "Write the {section_upper} section of a scoping/rapid literature review.\n\n"
    "Research questions:\n{rqs}\n\n"
    "Evidence summary:\n{rendered_evidence_context}\n\n"
    "Guidelines:\n"
    "- Academic, third-person writing\n"
    "- 500-800 words\n"
    "- Reference findings from the evidence summary\n"
    "- Use hedged language appropriate for a review\n"
)


def _build_imrad_narrative_prompt(section: str, evidence_context: str,
                                   canonical_research_questions: List[str]) -> Tuple[List[Dict[str, str]], str]:
    rendered_evidence_context = evidence_context[:6000]
    rqs = "\n".join(f"- {q}" for q in canonical_research_questions)
    messages = [
        {"role": "system", "content": _IMRAD_NARRATIVE_SYSTEM_TEMPLATE.format(section=section)},
        {"role": "user", "content": _IMRAD_NARRATIVE_USER_TEMPLATE.format(
            section_upper=section.upper(), rqs=rqs, rendered_evidence_context=rendered_evidence_context)},
    ]
    return messages, rendered_evidence_context


def _validate_imrad_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise MalformedModelResponseError(f"IMRAD generation returned no usable text (got {text!r})")
    return text


_IMRAD_NARRATIVE_MODEL_CONFIG_ID = "slr-chapter-synthesis-imrad-narrative-v1"


def _build_imrad_work_item(*, section: str, evidence_context: str, canonical_research_questions: List[str],
                            model: str) -> Tuple[str, List[Dict[str, str]], str, str, Dict[str, Any]]:
    """Builds the IMRAD narrative call's own messages/rendered_prompt_hash/work_item_id -- the ONE
    authoritative implementation, reused identically by `_run_one_imrad_section()` (the real call
    site) and `verify_synthesis_output_sources()` (the verifier). Returns (work_item_id, messages,
    rendered_evidence_context, rendered_prompt_hash, prompt_parameters)."""
    messages, rendered_evidence_context = _build_imrad_narrative_prompt(section, evidence_context,
                                                                         canonical_research_questions)
    prompt_parameters = {"temperature": 0.3}
    rendered_prompt_hash = sha256_hex(messages)
    work_item_payload = {"model": model, "model_config_id": _IMRAD_NARRATIVE_MODEL_CONFIG_ID,
                          "prompt_parameters": prompt_parameters, "prompt_template_version": _IMRAD_NARRATIVE_PROMPT_VERSION,
                          "rendered_prompt_hash": rendered_prompt_hash, "section": section}
    work_item_id = f"synthesis_imrad_narrative:{sha256_hex(work_item_payload)}"
    return work_item_id, messages, rendered_evidence_context, rendered_prompt_hash, prompt_parameters


def _run_one_imrad_section(*, section: str, evidence_context: str, canonical_research_questions: List[str],
                            client: OpenAI, cfg: ReviewConfig, review_run_id: str,
                            event_store: AIUseEventStore) -> Tuple[str, Optional[str], Optional[str], "AIUseEvent"]:
    model = cfg.synthesis.llm_model
    work_item_id, messages, rendered_evidence_context, rendered_prompt_hash, prompt_parameters = _build_imrad_work_item(
        section=section, evidence_context=evidence_context,
        canonical_research_questions=canonical_research_questions, model=model)

    with reserved_attempt(event_store, stage="synthesis_imrad_narrative", work_item_id=work_item_id) as attempt:
        resp_model: Optional[str] = None
        published_text: Optional[str] = None
        internal_exc: Optional[SynthesisInternalConsistencyError] = None

        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
        except Exception as exc:
            outcome, failure_detail = "failure", f"transport_error: {exc}"
        else:
            resp_model = _extract_response_snapshot(resp)
            try:
                raw_text = _read_response_content(resp)
                validated_text = _validate_imrad_text(raw_text)
            except MalformedModelResponseError as exc:
                outcome, failure_detail, published_text = "failure", f"invalid_response: {exc}", None
            except Exception as exc:
                # An unexpected internal defect during VALIDATION is NOT a successful,
                # trustworthy narrative section -- a real failure, never a fabricated success
                # carrying the unvalidated raw text.
                outcome, published_text = "failure", None
                failure_detail = f"internal_error: {exc}"
                internal_exc = SynthesisInternalConsistencyError(
                    f"unexpected internal error validating IMRAD text for section={section!r}: {exc}")
            else:
                outcome, failure_detail, published_text = "success", None, validated_text

        event = build_ai_use_event(
            review_run_id=review_run_id, stage="synthesis_imrad_narrative", work_item_id=work_item_id,
            attempt=attempt, tool_kind="llm", model=model, provider="openai",
            component_versions=({model: resp_model} if resp_model is not None else {}),
            access_info=str(client.base_url),
            task_purpose="Draft a free-text, non-claim-backed IMRAD section",
            model_config_id=_IMRAD_NARRATIVE_MODEL_CONFIG_ID,
            prompt_template_version=_IMRAD_NARRATIVE_PROMPT_VERSION,
            prompt_template_text=json.dumps([{"role": "system", "content": _IMRAD_NARRATIVE_SYSTEM_TEMPLATE},
                                              {"role": "user", "content": _IMRAD_NARRATIVE_USER_TEMPLATE}]),
            prompt_parameters=prompt_parameters, rendered_prompt_hash=rendered_prompt_hash,
            # An explicit section:<name> reference, the SAME "bind the record to an unambiguous
            # identity, not just a content hash" fix already applied to refine's own batch: reference.
            input_artifact_refs=[f"section:{section}", f"sha256:{sha256_hex(rendered_evidence_context)}"],
            output_artifact_refs=([f"sha256:{sha256_hex(published_text)}"] if published_text is not None else []),
            call_status=("success" if outcome == "success" else "failure"), failure_detail=failure_detail,
        )
        event = event_store.publish(event)

    if internal_exc is not None:
        raise internal_exc
    return outcome, (published_text if outcome == "success" else None), failure_detail, event


# ---------------------------------------------------------------------------
# Source-backed verification
# ---------------------------------------------------------------------------

_OUTPUT_HASH_RE = re.compile(r"^(response|update):sha256:[0-9a-f]{64}$")
# [0-9]/[1-9] are LITERAL character ranges -- codepoint-restricted regardless of Unicode mode --
# so this is ASCII-safe without needing re.ASCII, unlike \d (which matches Unicode decimal digits
# in Python's default regex mode). The (0|[1-9][0-9]*) alternation additionally forbids a
# leading-zero spelling like "00" for index 0, matching exactly what f"claim:{i}:..."/
# f"batch:{i}" (a real Python int formatted directly) can ever produce.
_INPUT_CLAIM_HASH_RE = re.compile(r"^claim:(0|[1-9][0-9]*):sha256:[0-9a-f]{64}$")
_INPUT_BATCH_REF_RE = re.compile(r"^batch:(0|[1-9][0-9]*)$")


def _canonical_map_response_payload_for_answer(a: MapAnswer) -> Dict[str, Any]:
    return _canonical_map_response_payload(answer_text=a.answer_text, selectors=a.selectors)


def _verify_refine_input_shape(event: "AIUseEvent", *, label: str, expected_target: str,
                                known_answers_by_id: Dict[str, MapAnswer],
                                expected_llm_model: str,
                                expected_batch_index: Optional[int] = None,
                                expected_answer_ids: Optional[List[str]] = None) -> None:
    """Validates input_artifact_refs (applies regardless of whether the call succeeded or failed,
    since these refs are constructed BEFORE the call is even attempted) AND the call's own STABLE,
    non-batch-dependent metadata (model, model_config_id, prompt_template_version, prompt_parameters)
    -- the four fields fully known ahead of time regardless of accumulator state. Refine's own
    rendered_prompt_hash/work_item_id are NOT checked HERE -- this shared helper has no access to a
    batch's own persisted pre_batch_claim_states (needed to reconstruct the full prompt), since the
    lighter of its two call sites (the `claims` loop) only has a claim's own final content, not the
    batch record it came from. The `refine_batch_outcomes` loop (which DOES have the full
    RefineBatchOutcomeRecord, pre_batch_claim_states included) performs the COMPLETE rendered_prompt_
    hash/work_item_id reconstruction itself, immediately after calling this helper, via the SAME
    _build_refine_work_item() the real call site uses -- see verify_synthesis_output_sources()'s own
    refine_batch_outcomes loop, immediately below this function's own call site there. Shared by the
    claims loop and the refine_batch_outcomes loop -- one authoritative implementation, never two
    that could silently drift apart."""
    if event.model != expected_llm_model:
        raise RuntimeError(f"{label}: producing event's own model {event.model!r} does not match "
                            f"the real, live cfg.synthesis.llm_model {expected_llm_model!r}")
    if event.model_config_id != _SYNTHESIS_REFINE_MODEL_CONFIG_ID:
        raise RuntimeError(f"{label}: producing event's own model_config_id {event.model_config_id!r} "
                            f"does not match the expected {_SYNTHESIS_REFINE_MODEL_CONFIG_ID!r}")
    if event.prompt_template_version != _SYNTHESIS_REFINE_PROMPT_VERSION:
        raise RuntimeError(f"{label}: producing event's own prompt_template_version "
                            f"{event.prompt_template_version!r} does not match the expected "
                            f"{_SYNTHESIS_REFINE_PROMPT_VERSION!r}")
    if event.prompt_parameters != _SYNTHESIS_REFINE_PROMPT_PARAMETERS:
        raise RuntimeError(f"{label}: producing event's own prompt_parameters "
                            f"{event.prompt_parameters!r} does not match the expected "
                            f"{_SYNTHESIS_REFINE_PROMPT_PARAMETERS!r}")
    if len(event.input_artifact_refs) != len(set(event.input_artifact_refs)):
        raise RuntimeError(f"{label}: producing event's own input_artifact_refs contains duplicate entries")
    unknown_refs = [ref for ref in event.input_artifact_refs
                     if not ref.startswith(("target:", "batch:", "answer:", "claim:"))]
    if unknown_refs:
        raise RuntimeError(f"{label}: producing event's own input_artifact_refs contains unrecognized "
                            f"reference type(s) {unknown_refs!r}")
    target_refs = [ref for ref in event.input_artifact_refs if ref.startswith("target:")]
    if target_refs != [f"target:{expected_target}"]:
        raise RuntimeError(f"{label}: producing event does not have EXACTLY one target: reference "
                            f"matching the expected target (found {target_refs!r})")

    batch_refs = [ref for ref in event.input_artifact_refs if ref.startswith("batch:")]
    if expected_batch_index is not None:
        if batch_refs != [f"batch:{expected_batch_index}"]:
            raise RuntimeError(f"{label}: producing event's own batch: reference does not exactly "
                                f"equal batch:{expected_batch_index} (found {batch_refs!r})")
    else:
        if len(batch_refs) != 1:
            raise RuntimeError(f"{label}: producing event does not have EXACTLY one batch: reference "
                                f"(found {batch_refs!r})")
        if not _INPUT_BATCH_REF_RE.fullmatch(batch_refs[0]):
            raise RuntimeError(f"{label}: producing event's own batch: reference {batch_refs[0]!r} is "
                                "not in canonical ASCII digit form (no leading zeros, no Unicode digit variants)")

    for ref in event.input_artifact_refs:
        if not ref.startswith("answer:"):
            continue
        referenced_answer = known_answers_by_id.get(ref[len("answer:"):])
        if referenced_answer is None:
            raise RuntimeError(f"{label}: producing event's own input_artifact_refs references unknown "
                                f"or malformed answer_id in {ref!r}")
        if referenced_answer.rq_or_section != expected_target:
            raise RuntimeError(f"{label}: producing event's own input_artifact_refs references answer "
                                f"{ref!r} belonging to a DIFFERENT target ({referenced_answer.rq_or_section!r})")

    # Real production ALWAYS emits exactly [target, batch] + [answer, answer, ...] + [claim, claim,
    # ...], in that order, with nothing interleaved and nothing after the claim: entries.
    refs = event.input_artifact_refs
    if len(refs) < 2 or not refs[0].startswith("target:") or not refs[1].startswith("batch:"):
        raise RuntimeError(f"{label}: producing event's own input_artifact_refs does not begin with "
                            "[target:, batch:] in that exact order")
    remainder = refs[2:]
    answer_refs_in_order = [ref for ref in remainder if ref.startswith("answer:")]
    claim_refs_in_remainder_order = [ref for ref in remainder if ref.startswith("claim:")]
    if remainder != answer_refs_in_order + claim_refs_in_remainder_order:
        raise RuntimeError(f"{label}: producing event's own input_artifact_refs does not follow the "
                            "one real layout [target, batch] + answers + claims -- entries are "
                            f"interleaved or out of place (found {remainder!r})")

    if expected_answer_ids is not None:
        actual_answer_ids_in_order = [ref[len("answer:"):] for ref in answer_refs_in_order]
        if actual_answer_ids_in_order != expected_answer_ids:
            raise RuntimeError(f"{label}: producing event's own answer: references, in order, "
                                f"{actual_answer_ids_in_order!r} do not exactly match the expected "
                                f"batch_answer_ids {expected_answer_ids!r}")

    claim_refs_in_order = [ref for ref in event.input_artifact_refs if ref.startswith("claim:")]
    for ref in claim_refs_in_order:
        if not _INPUT_CLAIM_HASH_RE.fullmatch(ref):
            raise RuntimeError(f"{label}: producing event's own input_artifact_refs contains a malformed "
                                f"claim-state entry {ref!r}")
    claim_indices = [int(ref.split(":")[1]) for ref in claim_refs_in_order]
    if claim_indices != list(range(len(claim_indices))):
        raise RuntimeError(f"{label}: producing event's own claim: references are not contiguously, "
                            f"uniquely indexed from 0 (found indices {claim_indices!r})")


def _verify_refine_output_shape(event: "AIUseEvent", *, label: str) -> None:
    """Validates output_artifact_refs for a CLEAN SUCCESS only -- exactly one leading response:
    hash, every remaining entry a unique update: hash."""
    for ref in event.output_artifact_refs:
        if not _OUTPUT_HASH_RE.fullmatch(ref):
            raise RuntimeError(f"{label}: producing event's own output_artifact_refs contains a malformed "
                                f"entry {ref!r}")
    response_refs = [ref for ref in event.output_artifact_refs if ref.startswith("response:")]
    if len(response_refs) != 1 or not event.output_artifact_refs or event.output_artifact_refs[0] != response_refs[0]:
        raise RuntimeError(f"{label}: producing event's own output_artifact_refs does not contain EXACTLY "
                            f"one leading response: entry (found {response_refs!r})")
    update_refs = event.output_artifact_refs[1:]
    if any(not ref.startswith("update:") for ref in update_refs):
        raise RuntimeError(f"{label}: producing event's own output_artifact_refs contains an entry after "
                            "the leading response: hash that is not update:-prefixed")
    if len(update_refs) != len(set(update_refs)):
        raise RuntimeError(f"{label}: producing event's own output_artifact_refs contains duplicate "
                            "update: hashes")


def build_synthesis_verification_context(
    *, cfg: ReviewConfig, records: List[Record], extraction_raw: Dict[str, Any],
    qa_raw: Optional[Dict[str, Any]], review_run_id: str, event_store: AIUseEventStore,
) -> Dict[str, Any]:
    """Rebuilds every index/closure/expected_* value
    `verify_synthesis_output_sources()` needs, from ALREADY-LOADED
    `extraction_raw`/`qa_raw` dicts (aggregation/export/reporting plan
    v17, §2.7) -- extracted, behavior-preserving, from `synthesis_node`'s
    own prior inline construction of the identical values. `records` must
    already be in `charted`'s own canonical order (see
    `canonicalize_corpus_order()`) -- this function does not reorder.

    Returns EXACTLY the keyword arguments `verify_synthesis_output_sources()`
    accepts (besides `output` itself) -- a caller invokes it as
    `verify_synthesis_output_sources(output, **build_synthesis_verification_context(...))`.
    `synthesis_node` itself calls this at its own verify call site instead
    of manually re-inlining this construction a second time; reporting's
    own `load_and_verify_synthesis_output()` calls it fresh from durable,
    re-verified artifacts with no live orchestration state at all."""
    charted = list(extraction_raw.get("resolved_projection", [])) if extraction_raw else []
    charted_by_id: Dict[str, Dict[str, Any]] = {c["record_id"]: c.get("raw_extraction", {}) for c in charted}
    extraction_index: Dict[str, Dict[str, List[Any]]] = extraction_raw.get("evidence_index", {}) if extraction_raw else {}

    extraction_schema_field_order: List[str] = []
    if records:
        extraction_schema = verified_bundle.load_schema(cfg)
        extraction_schema_field_order = [f.field_id for f in extraction_schema.fields]

    qa_doc_appraisals_by_id: Dict[str, DocumentAppraisal] = {}
    qa_index: Dict[str, Dict[str, List[Any]]] = {}
    if qa_raw is not None:
        qa_index = qa_raw.get("evidence_index", {})
        for proj in qa_raw.get("resolved_quality_projection", []):
            qa_doc_appraisals_by_id[proj["record_id"]] = DocumentAppraisal.model_validate(proj)

    canonical_research_questions = [rq.strip() for rq in cfg.research_questions]
    requested_targets = list(canonical_research_questions) + (
        ["imrad:results"] if cfg.synthesis.generate_claim_backed_results else []
    )
    synthesis_record_ids = [r.id for r in records]

    map_context_by_record: Dict[str, Tuple[List[SupportingFieldSelector], str]] = {
        record.id: _build_map_context(
            record_id=record.id, raw_extraction=charted_by_id.get(record.id, {}),
            extraction_field_order=extraction_schema_field_order, extraction_index=extraction_index,
            doc_appraisal=qa_doc_appraisals_by_id.get(record.id), qa_index=qa_index,
            char_budget=SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET)
        for record in records
    }

    def build_map_context_fn(record_id: str) -> Tuple[List[SupportingFieldSelector], str]:
        return map_context_by_record[record_id]

    entries_for_imrad = [(r.id, charted_by_id.get(r.id, {})) for r in records]
    evidence_lines: List[str] = []
    any_real_evidence = False
    for record_id, raw in entries_for_imrad:
        citation = raw.get("author_year") or record_id
        fields_text = "; ".join(f"{field_id}: {value}" for field_id, value in raw.items()
                                 if value not in (None, "", []))
        if fields_text:
            any_real_evidence = True
            evidence_lines.append(f"- {citation}: {fields_text}")
        else:
            evidence_lines.append(f"- {citation}: (no extracted data)")
    evidence_context = "\n".join(evidence_lines) if evidence_lines else (
        "\n".join(f"- {r.title} ({r.year})" for r in records[:50])
    )

    return {
        "review_run_id": review_run_id,
        "build_map_context_fn": build_map_context_fn,
        "extraction_index": extraction_index,
        "qa_index": qa_index,
        "event_store": event_store,
        "expected_refine_batch_size": cfg.synthesis.refine_batch_size,
        "expected_requested_targets": requested_targets,
        "expected_synthesis_record_ids": synthesis_record_ids,
        "expected_generate_legacy_imrad_narrative": cfg.synthesis.generate_legacy_imrad_narrative,
        "expected_imrad_records_exist": bool(records),
        "expected_imrad_any_real_evidence": any_real_evidence,
        "expected_evidence_context": evidence_context,
        "expected_canonical_research_questions": canonical_research_questions,
        "expected_llm_model": cfg.synthesis.llm_model,
    }


def verify_synthesis_output_sources(output: SynthesisOutput, *, review_run_id: str,
                                     build_map_context_fn: Callable[[str], Tuple[List[SupportingFieldSelector], str]],
                                     extraction_index: Dict[str, Dict[str, List[Any]]],
                                     qa_index: Dict[str, Dict[str, List[Any]]],
                                     event_store: AIUseEventStore,
                                     expected_refine_batch_size: int,
                                     expected_requested_targets: List[str],
                                     expected_synthesis_record_ids: List[str],
                                     expected_generate_legacy_imrad_narrative: bool,
                                     expected_imrad_records_exist: bool,
                                     expected_imrad_any_real_evidence: bool,
                                     expected_evidence_context: str,
                                     expected_canonical_research_questions: List[str],
                                     expected_llm_model: str) -> None:
    """Called once at the end of synthesis_node, before returning. Every `expected_*` parameter is
    the REAL, live value synthesis_node() itself just computed, supplied a second time as an
    independent expected value -- this verifier proves CONSISTENCY with caller-supplied ground
    truth, never independently rediscovers historical truth from nothing.
    `expected_llm_model`: the REAL, live cfg.synthesis.llm_model for THIS run -- without it, an
    event with the correct target/context/output hash but an UNRELATED model/prompt-parameters/
    template/config-id would still pass every other check (input_artifact_refs and output_artifact_refs
    say nothing about which model or prompt configuration actually produced them). Map's own prompt
    is fully reconstructable, so its own verification below recomputes the COMPLETE call identity --
    rendered_prompt_hash, model, prompt_parameters, model_config_id, prompt_template_version, and the
    recomputed work_item_id itself -- via the SAME _build_map_work_item() the real call site uses
    (reused, not reimplemented). IMRAD's own prompt is likewise fully reconstructable and gets the
    identical full-identity treatment via _build_imrad_work_item(). Refine's own STABLE (non-batch-
    dependent) metadata -- model/model_config_id/prompt_template_version/prompt_parameters -- is
    checked the same way inside _verify_refine_input_shape(). Refine's own rendered_prompt_hash/
    work_item_id ARE ALSO fully recomputed (codex code-review round 2, 2026-08-26 finding 1) -- via
    the SAME _build_refine_work_item() the real call site uses, fed each batch's own persisted
    RefineBatchOutcomeRecord.pre_batch_claim_states, immediately after the refine_batch_outcomes
    loop's own call to _verify_refine_input_shape() below. That persisted pre-batch state is itself
    proven non-fabricated by SynthesisOutput's own topology validator, which walks each target's own
    batch sequence from empty state and requires each batch's own pre_batch_claim_states to equal
    the state genuinely produced by every real batch before it -- so this verifier's own
    work_item_id/rendered_prompt_hash check ties that PROVEN-real state to the event that actually
    claims to have been produced from it, closing the "valid event substituted from a different
    accumulator context" gap."""
    if output.requested_targets != expected_requested_targets:
        raise RuntimeError(f"output.requested_targets={output.requested_targets!r} does not match "
                            f"the real, live requested_targets={expected_requested_targets!r} for this run "
                            "-- a target may have been deleted, added, or reordered")
    if output.synthesis_record_ids != expected_synthesis_record_ids:
        raise RuntimeError(f"output.synthesis_record_ids={output.synthesis_record_ids!r} does not match "
                            f"the real, live synthesis_record_ids={expected_synthesis_record_ids!r} for this "
                            "run -- a record may have been deleted, added, or reordered")
    if output.refine_batch_size != expected_refine_batch_size:
        raise RuntimeError(f"output.refine_batch_size={output.refine_batch_size!r} does not match "
                            f"the real, live cfg.synthesis.refine_batch_size={expected_refine_batch_size!r} "
                            "for this run")
    # Bind imrad_section_outcomes' own section coverage to the REAL, live flag -- closes the gap
    # where SynthesisOutput's own model checks could only prove internal mutual consistency, never
    # that the right sections (or none at all) were actually attempted.
    if expected_generate_legacy_imrad_narrative:
        expected_imrad_sections = {"introduction", "methods", "discussion"}
        actual_imrad_sections = {r.section for r in output.imrad_section_outcomes}
        if actual_imrad_sections != expected_imrad_sections:
            raise RuntimeError(f"imrad_section_outcomes covers {actual_imrad_sections!r}, expected exactly "
                                f"{expected_imrad_sections!r} -- generate_legacy_imrad_narrative is enabled "
                                "for this run")
        # The coverage check above only proves the RIGHT NUMBER of rows exist -- never that a
        # row's own claimed outcome="no_call" was actually legitimate. Independently recompute the
        # SAME three-way branch selection synthesis_node() itself used from the real, live booleans
        # supplied by the caller, and require every section's own outcome to match it exactly.
        if not expected_imrad_records_exist:
            expected_no_call_reason = "no full_text_records available to synthesize"
        elif not expected_imrad_any_real_evidence:
            expected_no_call_reason = "no full_text_records contain any extracted data to synthesize"
        else:
            expected_no_call_reason = None   # eligible corpus -- every section must be event-backed

        for r in output.imrad_section_outcomes:
            if expected_no_call_reason is not None:
                if r.outcome != "no_call" or r.failure_detail != expected_no_call_reason:
                    raise RuntimeError(f"IMRAD section {r.section!r}: this run has no eligible evidence -- "
                                        f"expected outcome='no_call' with failure_detail="
                                        f"{expected_no_call_reason!r}, got outcome={r.outcome!r} "
                                        f"failure_detail={r.failure_detail!r}")
            elif r.outcome == "no_call":
                raise RuntimeError(f"IMRAD section {r.section!r}: outcome='no_call' but the live corpus "
                                    "genuinely has usable evidence for this run -- a real call was required")
    elif output.imrad_section_outcomes or output.imrad_narrative_sections or output.imrad_narrative_failures:
        raise RuntimeError("generate_legacy_imrad_narrative is disabled for this run but "
                            "imrad_section_outcomes/imrad_narrative_sections/imrad_narrative_failures "
                            "is non-empty")
    answer_by_pair = {(a.rq_or_section, a.record_id): a for a in output.map_answers}
    known_answers_by_id = {a.answer_id: a for a in output.map_answers}
    context_by_record: Dict[str, Tuple[List[SupportingFieldSelector], str]] = {}

    for r in output.map_outcome_records:
        if r.record_id not in context_by_record:
            context_by_record[r.record_id] = build_map_context_fn(r.record_id)
        reconstructed_selectors, reconstructed_context = context_by_record[r.record_id]

        if not reconstructed_selectors:
            if r.outcome != "no_citeable_evidence":
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): reconstructed ZERO "
                                    f"citeable selectors but the persisted outcome is {r.outcome!r}")
            continue
        elif r.outcome == "no_citeable_evidence":
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): reconstructed "
                                f"{len(reconstructed_selectors)} citeable selector(s), but the persisted "
                                "outcome is no_citeable_evidence")

        event = event_store.load_event(r.producing_event_id)
        if event.review_run_id != review_run_id:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event belongs to "
                                "a different review_run_id")
        if event.stage != "synthesis_claim_map":
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event is not a "
                                "synthesis_claim_map event")

        expected_input_refs = [f"target:{r.rq_or_section}",
                                f"record:{r.record_id}:sha256:{sha256_hex(reconstructed_context)}"]
        if event.input_artifact_refs != expected_input_refs:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own "
                                "input_artifact_refs does not match the freshly reconstructed canonical input")

        # Map's own prompt is fully reconstructable -- verify the COMPLETE current call identity,
        # not just input/output refs. Without this, an event with the right target/context/output
        # hash but an UNRELATED model, prompt_parameters, template version, or model_config_id would
        # still pass every check above: input_artifact_refs/output_artifact_refs say nothing about
        # WHICH model or prompt configuration actually produced them. Reuses _build_map_work_item()
        # (the SAME function the live orchestration calls) rather than reimplementing the formula.
        expected_work_item_id, _expected_messages, expected_rendered_prompt_hash, expected_prompt_parameters = (
            _build_map_work_item(target=r.rq_or_section, record_id=r.record_id,
                                  rendered_context=reconstructed_context, model=expected_llm_model))
        if event.work_item_id != expected_work_item_id:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own "
                                f"work_item_id {event.work_item_id!r} does not match the freshly recomputed "
                                f"{expected_work_item_id!r} -- model/prompt_parameters/template/config-id/"
                                "rendered content may not genuinely match what this event claims")
        if event.model != expected_llm_model:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own model "
                                f"{event.model!r} does not match the real, live cfg.synthesis.llm_model "
                                f"{expected_llm_model!r}")
        if event.model_config_id != _SYNTHESIS_MAP_MODEL_CONFIG_ID:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own "
                                f"model_config_id {event.model_config_id!r} does not match the expected "
                                f"{_SYNTHESIS_MAP_MODEL_CONFIG_ID!r}")
        if event.prompt_template_version != _SYNTHESIS_MAP_PROMPT_VERSION:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own "
                                f"prompt_template_version {event.prompt_template_version!r} does not match "
                                f"the expected {_SYNTHESIS_MAP_PROMPT_VERSION!r}")
        if event.prompt_parameters != expected_prompt_parameters:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own "
                                f"prompt_parameters {event.prompt_parameters!r} does not match the expected "
                                f"{expected_prompt_parameters!r}")
        if event.rendered_prompt_hash != expected_rendered_prompt_hash:
            raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's own "
                                "rendered_prompt_hash does not match the freshly reconstructed prompt")

        if r.outcome == "relevant":
            a = answer_by_pair[(r.rq_or_section, r.record_id)]
            if a.producing_event_id != r.producing_event_id:
                raise RuntimeError(f"MapAnswer {a.answer_id} and its own MapOutcomeRecord disagree on "
                                    "producing_event_id")
            answer_selector_keys = {(s.artifact_kind, s.field_id) for s in a.selectors}
            citeable_keys = {(s.artifact_kind, s.field_id) for s in reconstructed_selectors}
            if not (answer_selector_keys <= citeable_keys):
                raise RuntimeError(f"MapAnswer {a.answer_id}: cites selector(s) "
                                    f"{sorted(answer_selector_keys - citeable_keys)} that were NOT among the "
                                    "selectors reconstructed as citeable for this record -- would have been "
                                    "truncated out of, or never offered in, the actual budget-bounded prompt")
            if event.call_status != "success" or event.failure_detail is not None:
                raise RuntimeError(f"MapAnswer {a.answer_id}: producing event is not a clean success")
            expected_output = [f"sha256:{sha256_hex(_canonical_map_response_payload_for_answer(a))}"]
            if event.output_artifact_refs != expected_output:
                raise RuntimeError(f"MapAnswer {a.answer_id}: producing event's output_artifact_refs does not "
                                    "exactly match this answer's own canonical content")
        elif r.outcome == "not_relevant":
            if event.call_status != "success" or event.failure_detail is not None:
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): expected a clean success")
            expected_output = [f"sha256:{sha256_hex({'relevant': False})}"]
            if event.output_artifact_refs != expected_output:
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): producing event's "
                                    "output_artifact_refs does not match the expected not_relevant payload")
        elif r.outcome == "call_failed":
            if event.call_status != "failure" or not event.failure_detail or \
                    not event.failure_detail.startswith("transport_error: "):
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): expected a "
                                    "transport_error-prefixed failure")
            if event.output_artifact_refs != []:
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): call_failed must have "
                                    "no output_artifact_refs")
        elif r.outcome == "invalid_response":
            if event.call_status != "failure" or not event.failure_detail or \
                    not event.failure_detail.startswith("invalid_response: "):
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): expected an "
                                    "invalid_response-prefixed failure")
            if event.output_artifact_refs != []:
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): invalid_response must "
                                    "have no output_artifact_refs")
            has_marker = event.failure_detail.startswith("invalid_response: hallucinated_selector: ")
            if r.hallucinated_selector != has_marker:
                raise RuntimeError(f"outcome ({r.rq_or_section!r}, {r.record_id!r}): hallucinated_selector="
                                    f"{r.hallucinated_selector!r} disagrees with the event's own recorded "
                                    "failure_detail prefix")

    # Reconstruct the COMPLETE, deterministic expected reference set for every answer, via the
    # SAME _references_for_selectors() the live orchestration itself calls -- reused, never
    # reimplemented.
    for a in output.map_answers:
        try:
            expected_references = _references_for_selectors(
                a.selectors, a.record_id, extraction_index, qa_index,
                output.extraction_run_id, output.quality_appraisal_run_id)
        except SynthesisInternalConsistencyError as exc:
            raise RuntimeError(f"MapAnswer {a.answer_id}: could not reconstruct its own expected "
                                f"evidence_references -- {exc}")
        expected_sorted = sorted(expected_references,
                                  key=lambda r: (r.artifact_kind, r.artifact_run_id, r.record_id, r.field_id, r.evidence_id))
        if a.evidence_references != expected_sorted:
            raise RuntimeError(f"MapAnswer {a.answer_id}: evidence_references does not exactly match the "
                                "deterministic complete set _references_for_selectors() produces for its "
                                "own selectors -- silent citation loss (or fabrication)")

    # Independently verify EVERY refine batch ATTEMPT's own record -- not only the batches whose
    # accumulator update happens to still be reachable via a surviving claim's own producing_event_id.
    for rbo in output.refine_batch_outcomes:
        event = event_store.load_event(rbo.producing_event_id)
        if event.review_run_id != review_run_id:
            raise RuntimeError(f"refine batch ({rbo.rq_or_section!r}, {rbo.batch_index}): producing_event_id "
                                "belongs to a different review_run_id")
        if event.stage != "synthesis_claim_refine":
            raise RuntimeError(f"refine batch ({rbo.rq_or_section!r}, {rbo.batch_index}): producing_event_id "
                                "is not a synthesis_claim_refine event")

        batch_label = f"refine batch ({rbo.rq_or_section!r}, {rbo.batch_index})"
        _verify_refine_input_shape(event, label=batch_label, expected_target=rbo.rq_or_section,
                                    known_answers_by_id=known_answers_by_id, expected_llm_model=expected_llm_model,
                                    expected_batch_index=rbo.batch_index,
                                    expected_answer_ids=rbo.batch_answer_ids)

        # Codex code-review round 2 (2026-08-26) finding 1: refine's own prompt IS fully
        # reconstructable after all, from this batch's own persisted pre_batch_claim_states -- verify
        # the COMPLETE call identity (work_item_id, rendered_prompt_hash), via the SAME
        # _build_refine_work_item() the real call site uses, exactly mirroring how map/IMRAD are
        # already verified above. Without this, a genuinely valid same-run refine event produced
        # against a DIFFERENT prior accumulator state could be substituted here as long as target/
        # batch answers/output-hash shape happened to match (SynthesisOutput's own topology
        # validator already proves pre_batch_claim_states itself is not a fabricated/stale
        # accumulator snapshot -- see state.py -- so this check ties that PROVEN-real state to the
        # event that actually claims to have been produced from it).
        try:
            batch_for_reconstruction = [known_answers_by_id[aid] for aid in rbo.batch_answer_ids]
            expected_pre_batch_claim_states = [(s.claim_text, s.supporting_answer_ids)
                                                 for s in rbo.pre_batch_claim_states]
            expected_work_item_id, _expected_messages, expected_rendered_prompt_hash, _expected_prompt_parameters = (
                _build_refine_work_item(target=rbo.rq_or_section, batch_index=rbo.batch_index,
                                         batch=batch_for_reconstruction,
                                         pre_batch_claim_states=expected_pre_batch_claim_states,
                                         answer_lookup=known_answers_by_id, model=expected_llm_model))
        except KeyError as exc:
            raise RuntimeError(f"{batch_label}: pre_batch_claim_states references an unknown answer_id "
                                f"{exc} -- could not reconstruct the expected refine prompt")
        if event.work_item_id != expected_work_item_id:
            raise RuntimeError(f"{batch_label}: producing event's own work_item_id {event.work_item_id!r} "
                                f"does not match the freshly recomputed {expected_work_item_id!r} -- this "
                                "event may have been produced against a DIFFERENT prior accumulator state")
        if event.rendered_prompt_hash != expected_rendered_prompt_hash:
            raise RuntimeError(f"{batch_label}: producing event's own rendered_prompt_hash does not match "
                                "the prompt freshly reconstructed from this batch's own persisted "
                                "pre_batch_claim_states -- this event may have been produced against a "
                                "DIFFERENT prior accumulator state")

        if rbo.outcome == "applied":
            if event.call_status != "success" or event.failure_detail is not None:
                raise RuntimeError(f"{batch_label}: outcome='applied' but producing event is not a clean success")
            _verify_refine_output_shape(event, label=batch_label)
            # Exact-value check (not just shape) against this batch's own persisted applied_updates
            # -- reused via the SAME _canonical_refine_response_payload()/_canonical_refine_update_
            # payload() the live orchestration hashes at publish time.
            expected_output_refs = (
                [f"response:sha256:{sha256_hex(_canonical_refine_response_payload(rbo.applied_updates))}"]
                + [f"update:sha256:{sha256_hex(_canonical_refine_update_payload(u.claim_text, u.final_support_answer_ids))}"
                   for u in rbo.applied_updates]
            )
            if event.output_artifact_refs != expected_output_refs:
                raise RuntimeError(f"{batch_label}: producing event's own output_artifact_refs does not "
                                    "exactly match the canonical hash of this batch's own persisted "
                                    "applied_updates")
        elif rbo.outcome == "call_failed":
            # ONLY transport_error: is representable here -- the live orchestration always raises
            # internal_exc IMMEDIATELY after publishing an internal_error: event and BEFORE ever
            # appending a RefineBatchOutcomeRecord for it.
            if event.call_status != "failure" or not event.failure_detail or \
                    not event.failure_detail.startswith("transport_error: "):
                raise RuntimeError(f"{batch_label}: expected a transport_error-prefixed failure -- "
                                    "internal_error: can never legitimately appear on a persisted "
                                    "RefineBatchOutcomeRecord (see this class's own docstring)")
            if event.output_artifact_refs != []:
                raise RuntimeError(f"{batch_label}: call_failed must have no output_artifact_refs")
        else:   # invalid_response
            if event.call_status != "failure" or not event.failure_detail or \
                    not event.failure_detail.startswith("invalid_response: "):
                raise RuntimeError(f"{batch_label}: expected an invalid_response-prefixed failure")
            if event.output_artifact_refs != []:
                raise RuntimeError(f"{batch_label}: invalid_response must have no output_artifact_refs")

    # Independently verify EVERY legacy IMRAD section ATTEMPT's own record -- the SAME "one record
    # per attempt, every attempt checked against its own real event" pattern just applied above to
    # refine_batch_outcomes. outcome="no_call" has no event to check at all; its own execution-mode
    # legitimacy was already checked above, before this loop.
    for r in output.imrad_section_outcomes:
        if r.outcome == "no_call":
            continue
        imrad_label = f"IMRAD section {r.section!r}"
        event = event_store.load_event(r.producing_event_id)
        if event.review_run_id != review_run_id:
            raise RuntimeError(f"{imrad_label}: producing_event_id belongs to a different review_run_id")
        if event.stage != "synthesis_imrad_narrative":
            raise RuntimeError(f"{imrad_label}: producing_event_id is not a synthesis_imrad_narrative event")
        # The explicit section:<name> reference is what makes this binding exact -- but a plain
        # membership test is not itself exact: an event carrying MULTIPLE section: references
        # could satisfy every section's own record simultaneously, and membership alone says
        # nothing about whether the event's own evidence-context hash (or its model/prompt-
        # parameters/template/config-id) matches the CURRENT run's identity. IMRAD's own prompt is
        # fully reconstructable, exactly like map's -- reconstruct the ONE real, current expected
        # call identity for this section via _build_imrad_work_item() (the SAME function the live
        # orchestration calls), and require EXACT equality for input_artifact_refs, model,
        # model_config_id, prompt_template_version, prompt_parameters, rendered_prompt_hash, AND the
        # recomputed work_item_id itself.
        expected_work_item_id, _expected_messages, expected_rendered_evidence_context, expected_rendered_prompt_hash, expected_prompt_parameters = (
            _build_imrad_work_item(section=r.section, evidence_context=expected_evidence_context,
                                    canonical_research_questions=expected_canonical_research_questions,
                                    model=expected_llm_model))
        expected_imrad_input_refs = [f"section:{r.section}", f"sha256:{sha256_hex(expected_rendered_evidence_context)}"]
        if event.input_artifact_refs != expected_imrad_input_refs:
            raise RuntimeError(f"{imrad_label}: producing event's own input_artifact_refs "
                                f"{event.input_artifact_refs!r} does not exactly match the freshly "
                                f"reconstructed current expected input {expected_imrad_input_refs!r} -- "
                                "possible section swap or stale evidence context")
        if event.work_item_id != expected_work_item_id:
            raise RuntimeError(f"{imrad_label}: producing event's own work_item_id {event.work_item_id!r} "
                                f"does not match the freshly recomputed {expected_work_item_id!r} -- model/"
                                "prompt_parameters/template/config-id/rendered content may not genuinely "
                                "match what this event claims")
        if event.model != expected_llm_model:
            raise RuntimeError(f"{imrad_label}: producing event's own model {event.model!r} does not match "
                                f"the real, live cfg.synthesis.llm_model {expected_llm_model!r}")
        if event.model_config_id != _IMRAD_NARRATIVE_MODEL_CONFIG_ID:
            raise RuntimeError(f"{imrad_label}: producing event's own model_config_id "
                                f"{event.model_config_id!r} does not match the expected "
                                f"{_IMRAD_NARRATIVE_MODEL_CONFIG_ID!r}")
        if event.prompt_template_version != _IMRAD_NARRATIVE_PROMPT_VERSION:
            raise RuntimeError(f"{imrad_label}: producing event's own prompt_template_version "
                                f"{event.prompt_template_version!r} does not match the expected "
                                f"{_IMRAD_NARRATIVE_PROMPT_VERSION!r}")
        if event.prompt_parameters != expected_prompt_parameters:
            raise RuntimeError(f"{imrad_label}: producing event's own prompt_parameters "
                                f"{event.prompt_parameters!r} does not match the expected "
                                f"{expected_prompt_parameters!r}")
        if event.rendered_prompt_hash != expected_rendered_prompt_hash:
            raise RuntimeError(f"{imrad_label}: producing event's own rendered_prompt_hash does not match "
                                "the freshly reconstructed current expected prompt -- possible stale "
                                "research-question set or tampered prompt")

        if r.outcome == "success":
            if event.call_status != "success" or event.failure_detail is not None:
                raise RuntimeError(f"{imrad_label}: outcome='success' but producing event is not a clean success")
            expected_output = [f"sha256:{sha256_hex(output.imrad_narrative_sections[r.section])}"]
            if event.output_artifact_refs != expected_output:
                raise RuntimeError(f"{imrad_label}: producing event's output_artifact_refs does not exactly "
                                    "match this section's own current narrative text -- possible swapped "
                                    "or tampered text")
        elif r.outcome == "call_failed":
            if event.call_status != "failure" or not event.failure_detail or \
                    not event.failure_detail.startswith("transport_error: "):
                raise RuntimeError(f"{imrad_label}: expected a transport_error-prefixed failure -- "
                                    "internal_error: can never legitimately appear on a persisted "
                                    "ImradSectionOutcomeRecord (see this class's own docstring)")
            if event.output_artifact_refs != []:
                raise RuntimeError(f"{imrad_label}: call_failed must have no output_artifact_refs")
            if r.failure_detail != event.failure_detail:
                raise RuntimeError(f"{imrad_label}: record's own failure_detail {r.failure_detail!r} does "
                                    f"not exactly match its producing event's own failure_detail "
                                    f"{event.failure_detail!r}")
        else:   # invalid_response
            if event.call_status != "failure" or not event.failure_detail or \
                    not event.failure_detail.startswith("invalid_response: "):
                raise RuntimeError(f"{imrad_label}: expected an invalid_response-prefixed failure")
            if event.output_artifact_refs != []:
                raise RuntimeError(f"{imrad_label}: invalid_response must have no output_artifact_refs")
            if r.failure_detail != event.failure_detail:
                raise RuntimeError(f"{imrad_label}: record's own failure_detail {r.failure_detail!r} does "
                                    f"not exactly match its producing event's own failure_detail "
                                    f"{event.failure_detail!r}")

    # Every claim's OWN refine-event lineage, not just its map-answer lineage. This check proves a
    # claim's CURRENT content matches a real "update:" entry inside a real, successful,
    # correctly-and-completely-shaped refine event -- it does NOT reconstruct or replay that
    # event's own prior batch/accumulator composition (refine's own input is a whole prior
    # accumulator list, evolving batch-by-batch, not persisted anywhere this module captures).
    for c in output.claims:
        event = event_store.load_event(c.producing_event_id)
        if event.review_run_id != review_run_id:
            raise RuntimeError(f"claim {c.claim_id}: producing_event_id belongs to a different review_run_id")
        if event.stage != "synthesis_claim_refine":
            raise RuntimeError(f"claim {c.claim_id}: producing_event_id is not a synthesis_claim_refine event")
        if event.call_status != "success" or event.failure_detail is not None:
            raise RuntimeError(f"claim {c.claim_id}: producing event is not a clean success")

        claim_label = f"claim {c.claim_id}"
        _verify_refine_input_shape(event, label=claim_label, expected_target=c.rq_or_section,
                                    known_answers_by_id=known_answers_by_id, expected_llm_model=expected_llm_model)
        _verify_refine_output_shape(event, label=claim_label)

        expected_update_hash = f"update:sha256:{sha256_hex(_canonical_refine_update_payload(c.claim_text, c.supporting_answer_ids))}"
        if expected_update_hash not in event.output_artifact_refs:
            raise RuntimeError(f"claim {c.claim_id}: no update in its producing event's own output matches "
                                "this claim's current claim_text/supporting_answer_ids")


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def _automation_origin_for_stage(events: List[AIUseEvent]) -> str:
    if any(e.call_status == "success" for e in events):
        return "ai"
    return "no_automated_output"


def synthesis_node(state: ReviewState) -> Dict[str, Any]:
    cfg = ReviewConfig.from_dict(state["config"])
    review_run_id = resolve_review_run_id(state, allow_config=True)
    output_dir = cfg.ensure_output_dir()
    event_store = AIUseEventStore(output_dir=output_dir, review_run_id=review_run_id)

    # --- canonical corpus + extraction coverage (UNCHANGED from today's real, existing code) ---
    extraction_artifact = load_verified_extraction_resolution_artifact(state)
    charted = list(extraction_artifact["resolved_projection"]) if extraction_artifact else []
    charted_by_id: Dict[str, Dict[str, Any]] = {c["record_id"]: c.get("raw_extraction", {}) for c in charted}
    records = [Record.from_dict(r) for r in state["full_text_records"]]
    records = canonicalize_corpus_order(
        full_text_records=records, charted=charted,
        error_prefix=f"synthesis_node (review_type={cfg.review_type!r})",
    )

    extraction_index: Dict[str, Dict[str, List[Any]]] = extraction_artifact["evidence_index"] if extraction_artifact else {}
    extraction_run_id: Optional[str] = extraction_artifact["run_id"] if extraction_artifact else None

    extraction_schema_field_order: List[str] = []
    if records:
        extraction_schema = verified_bundle.load_schema(cfg)
        extraction_schema_field_order = [f.field_id for f in extraction_schema.fields]

    # --- QA artifact + evidence index ---
    qa_doc_appraisals_by_id: Dict[str, DocumentAppraisal] = {}
    qa_index: Dict[str, Dict[str, List[Any]]] = {}
    qa_run_id: Optional[str] = None
    qa_artifact: Optional[Dict[str, Any]] = None
    if cfg.review_type in ("rapid", "slr") and records:
        qa_artifact = load_verified_quality_appraisal_resolution_artifact(state)
        if qa_artifact is None:
            raise ValueError(f"synthesis_node (review_type={cfg.review_type!r}): quality appraisal is "
                              "required for this review type but no verified QA artifact was found")
        qa_projection_ids = [p["record_id"] for p in qa_artifact["resolved_quality_projection"]]
        duplicate_qa_ids = sorted({rid for rid in qa_projection_ids if qa_projection_ids.count(rid) > 1})
        if duplicate_qa_ids:
            raise ValueError(f"quality-appraisal artifact contains duplicate record_id(s): {duplicate_qa_ids}")
        qa_id_set, corpus_id_set = set(qa_projection_ids), {r.id for r in records}
        qa_missing, qa_foreign = sorted(corpus_id_set - qa_id_set), sorted(qa_id_set - corpus_id_set)
        if qa_missing or qa_foreign:
            raise ValueError(f"quality-appraisal artifact coverage mismatch: missing={qa_missing!r} "
                              f"foreign={qa_foreign!r}")
        qa_run_id = qa_artifact["qa_run_id"]
        qa_index = qa_artifact["evidence_index"]
        for proj in qa_artifact["resolved_quality_projection"]:
            qa_doc_appraisals_by_id[proj["record_id"]] = DocumentAppraisal.model_validate(proj)

    # --- 1. Evidence map (UNCHANGED, pre-existing) ---
    evidence_map: Optional[Dict[str, Any]] = None
    if cfg.synthesis.generate_evidence_map and records:
        evidence_map = _build_evidence_map(records, charted_by_id)

    # --- 2. BERTopic (UNCHANGED, pre-existing) ---
    topics: List[Dict] = []
    bertopic_excluded_empty_evidence_count = 0
    if cfg.synthesis.run_bertopic and records:
        bertopic_records, bertopic_texts = [], []
        for record in records:
            raw = charted_by_id.get(record.id, {})
            text = "; ".join(f"{k}: {v}" for k, v in raw.items() if v not in (None, "", []))
            if text:
                bertopic_records.append(record)
                bertopic_texts.append(text)
            else:
                bertopic_excluded_empty_evidence_count += 1
        if bertopic_records:
            topics = _run_bertopic(bertopic_records, cfg.synthesis.n_topics, bertopic_texts)

    # --- 3. Target identity ---
    canonical_research_questions = [rq.strip() for rq in cfg.research_questions]
    if any(not rq for rq in canonical_research_questions):
        raise ValueError("synthesis_node: cfg.research_questions must not contain blank entries")
    if len(canonical_research_questions) != len(set(canonical_research_questions)):
        raise ValueError("synthesis_node: cfg.research_questions must not contain duplicate entries "
                          "(after stripping whitespace)")
    if any(rq.startswith("imrad:") for rq in canonical_research_questions):
        raise ValueError("synthesis_node: a research question must not start with the reserved 'imrad:' prefix")
    requested_targets = list(canonical_research_questions) + (
        ["imrad:results"] if cfg.synthesis.generate_claim_backed_results else []
    )
    synthesis_record_ids = [r.id for r in records]

    # Precompute EVERY record's own map context up front, before deciding whether a client is
    # needed at all -- this is the only way to know, without guessing, whether any map call could
    # genuinely occur.
    map_context_by_record: Dict[str, Tuple[List[SupportingFieldSelector], str]] = {
        record.id: _build_map_context(
            record_id=record.id, raw_extraction=charted_by_id.get(record.id, {}),
            extraction_field_order=extraction_schema_field_order, extraction_index=extraction_index,
            doc_appraisal=qa_doc_appraisals_by_id.get(record.id), qa_index=qa_index,
            char_budget=SYNTHESIS_MAP_CONTEXT_CHAR_BUDGET)
        for record in records
    }
    any_record_has_evidence = any(bool(selectors) for selectors, _ in map_context_by_record.values())

    # Build the IMRAD evidence context -- and know whether it is genuinely usable -- up front too,
    # for the identical reason.
    entries_for_imrad = [(r.id, charted_by_id.get(r.id, {})) for r in records]
    evidence_lines: List[str] = []
    any_real_evidence = False
    for record_id, raw in entries_for_imrad:
        citation = raw.get("author_year") or record_id
        fields_text = "; ".join(f"{field_id}: {value}" for field_id, value in raw.items()
                                 if value not in (None, "", []))
        if fields_text:
            any_real_evidence = True
            evidence_lines.append(f"- {citation}: {fields_text}")
        else:
            evidence_lines.append(f"- {citation}: (no extracted data)")
    evidence_context = "\n".join(evidence_lines) if evidence_lines else (
        "\n".join(f"- {r.title} ({r.year})" for r in records[:50])
    )

    # Construct the OpenAI client only when a real call could genuinely occur -- map/refine need
    # at least one record with non-empty citeable_selectors for some requested target; legacy
    # IMRAD needs at least one record with genuinely-extracted (non-placeholder) evidence.
    need_map_client = bool(requested_targets) and any_record_has_evidence
    need_imrad_client = cfg.synthesis.generate_legacy_imrad_narrative and bool(records) and any_real_evidence
    client: Optional[OpenAI] = OpenAI(api_key=resolve_openai_api_key()) if (need_map_client or need_imrad_client) else None

    # --- 4. Map step: every (target, record) pair ---
    map_outcome_records: List[MapOutcomeRecord] = []
    map_answers: List[MapAnswer] = []
    map_events: List[AIUseEvent] = []

    for target in requested_targets:
        for record in records:
            citeable_selectors, rendered_context = map_context_by_record[record.id]

            if not citeable_selectors:
                map_outcome_records.append(MapOutcomeRecord(rq_or_section=target, record_id=record.id,
                                                              outcome="no_citeable_evidence"))
                continue

            outcome, parsed, hallucinated, event, internal_exc = _run_one_map_call(
                target=target, record=record, citeable_selectors=citeable_selectors,
                rendered_context=rendered_context, client=client, cfg=cfg, review_run_id=review_run_id,
                event_store=event_store)
            map_events.append(event)
            if internal_exc is not None:
                raise internal_exc   # abort -- AFTER the event is durably published

            if outcome == "relevant":
                evidence_references = _references_for_selectors(parsed.supporting_selectors, record.id,
                                                                  extraction_index, qa_index,
                                                                  extraction_run_id, qa_run_id)
                map_answers.append(build_map_answer(rq_or_section=target, record_id=record.id, answer_text=parsed.answer,
                                                     selectors=parsed.supporting_selectors, evidence_references=evidence_references,
                                                     producing_event_id=event.event_id))
                map_outcome_records.append(MapOutcomeRecord(rq_or_section=target, record_id=record.id,
                                                              outcome="relevant", producing_event_id=event.event_id))
            else:
                map_outcome_records.append(MapOutcomeRecord(
                    rq_or_section=target, record_id=record.id, outcome=outcome,
                    hallucinated_selector=(outcome == "invalid_response" and hallucinated),
                    producing_event_id=event.event_id))

    # --- 5. Refine + finalize, per target ---
    claims: List[SynthesisClaim] = []
    rejected_targets: List[RejectedSynthesisTarget] = []
    claim_construction_coverage: List[ClaimConstructionCoverage] = []
    refine_events: List[AIUseEvent] = []
    refine_batch_outcomes: List[RefineBatchOutcomeRecord] = []

    for target in requested_targets:
        target_claims, rejected, coverage, target_refine_events, target_refine_batch_outcomes = _finalize_target(
            target=target, map_answers=map_answers, cfg=cfg, client=client,
            review_run_id=review_run_id, event_store=event_store)
        claims.extend(target_claims)
        if rejected is not None:
            rejected_targets.append(rejected)
        claim_construction_coverage.append(coverage)
        refine_events.extend(target_refine_events)
        refine_batch_outcomes.extend(target_refine_batch_outcomes)

    # --- 6. IMRAD legacy narrative ---
    imrad_narrative_sections: Dict[str, str] = {}
    imrad_narrative_failures: Dict[str, str] = {}
    imrad_events: List[AIUseEvent] = []
    imrad_section_outcomes: List[ImradSectionOutcomeRecord] = []
    if cfg.synthesis.generate_legacy_imrad_narrative:
        if not records:
            for section in ("introduction", "methods", "discussion"):
                reason = "no full_text_records available to synthesize"
                imrad_narrative_failures[section] = reason
                # No physical call was ever attempted for this section -- outcome="no_call", no
                # producing_event_id, matching MapOutcomeRecord's own established "no_citeable_evidence
                # means no event" contract.
                imrad_section_outcomes.append(ImradSectionOutcomeRecord(
                    section=section, outcome="no_call", producing_event_id=None, failure_detail=reason))
        elif not any_real_evidence:
            # Records exist but every one is an unextracted placeholder -- there is nothing
            # genuinely usable to synthesize from either, so still never call the model.
            for section in ("introduction", "methods", "discussion"):
                reason = "no full_text_records contain any extracted data to synthesize"
                imrad_narrative_failures[section] = reason
                imrad_section_outcomes.append(ImradSectionOutcomeRecord(
                    section=section, outcome="no_call", producing_event_id=None, failure_detail=reason))
        else:
            for section in ("introduction", "methods", "discussion"):
                outcome, text, failure_detail, event = _run_one_imrad_section(
                    section=section, evidence_context=evidence_context,
                    canonical_research_questions=canonical_research_questions, client=client, cfg=cfg,
                    review_run_id=review_run_id, event_store=event_store)
                imrad_events.append(event)
                if outcome == "success" and text is not None:
                    imrad_narrative_sections[section] = text
                    imrad_section_outcomes.append(ImradSectionOutcomeRecord(
                        section=section, outcome="success", producing_event_id=event.event_id, failure_detail=None))
                else:
                    imrad_narrative_failures[section] = failure_detail
                    # failure_detail's own real prefix (never "internal_error: " -- that path
                    # always raises inside _run_one_imrad_section() itself, immediately after
                    # publishing, before ever returning normally, so this branch can only ever see
                    # the two ordinary, expected failure categories) determines the persisted
                    # outcome, mirroring MapOutcomeRecord's own literal outcome vocabulary.
                    record_outcome = "call_failed" if failure_detail.startswith("transport_error: ") else "invalid_response"
                    imrad_section_outcomes.append(ImradSectionOutcomeRecord(
                        section=section, outcome=record_outcome, producing_event_id=event.event_id,
                        failure_detail=failure_detail))

    # --- 7. Derived views ---
    grouped: Dict[str, List[str]] = {}
    for c in claims:
        grouped.setdefault(c.rq_or_section, []).append(c.claim_text)
    joined = {k: "\n\n".join(v) for k, v in grouped.items()}
    rq_synthesis = {k: v for k, v in joined.items() if not k.startswith("imrad:")}
    imrad_claim_sections = {k[len("imrad:"):]: v for k, v in joined.items() if k.startswith("imrad:")}
    sections = dict(imrad_claim_sections)
    accepted_targets = {c.rq_or_section for c in claims}
    target_coverage = (None if not claim_construction_coverage
                        else len(accepted_targets) / len(claim_construction_coverage))

    # --- 8. Construct + verify the output ---
    output = SynthesisOutput(
        topics=topics, evidence_map=evidence_map, bertopic_excluded_empty_evidence_count=bertopic_excluded_empty_evidence_count,
        extraction_run_id=extraction_run_id, quality_appraisal_run_id=qa_run_id,
        requested_targets=requested_targets, synthesis_record_ids=synthesis_record_ids,
        map_outcome_records=map_outcome_records, map_answers=map_answers, claims=claims,
        rejected_targets=rejected_targets, refine_batch_outcomes=refine_batch_outcomes,
        # the effective batch size for THIS run, so the topology validator can reconstruct the
        # exact expected _chunk() output per target.
        refine_batch_size=cfg.synthesis.refine_batch_size,
        claim_construction_coverage=claim_construction_coverage,
        target_coverage=target_coverage, rq_synthesis=rq_synthesis, imrad_claim_sections=imrad_claim_sections,
        imrad_narrative_sections=imrad_narrative_sections, imrad_narrative_failures=imrad_narrative_failures,
        imrad_section_outcomes=imrad_section_outcomes,
        sections=sections,
    )

    # build_synthesis_verification_context() independently RECOMPUTES the
    # same expected_*/index/closure values this function computed inline
    # above (steps 3-6) from the SAME extraction_artifact/qa_artifact/cfg/
    # records this run actually used -- reused here (rather than manually
    # re-threading each value into the verify call individually) so the
    # identical reconstruction logic also backs reporting's own
    # load_and_verify_synthesis_output(), which has no live orchestration
    # state to draw from at all.
    verification_ctx = build_synthesis_verification_context(
        cfg=cfg, records=records, extraction_raw=(extraction_artifact or {}), qa_raw=qa_artifact,
        review_run_id=review_run_id, event_store=event_store,
    )
    verify_synthesis_output_sources(output, **verification_ctx)

    # --- 9. oversight_mode / automation_origin ---
    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    if requested_targets:
        oversight_mode["synthesis_claim_map"] = "automation_unreviewed"
        automation_origin["synthesis_claim_map"] = _automation_origin_for_stage(map_events)
        oversight_mode["synthesis_claim_refine"] = "automation_unreviewed"
        automation_origin["synthesis_claim_refine"] = _automation_origin_for_stage(refine_events)
    if cfg.synthesis.generate_legacy_imrad_narrative:
        oversight_mode["synthesis_imrad_narrative"] = "automation_unreviewed"
        automation_origin["synthesis_imrad_narrative"] = _automation_origin_for_stage(imrad_events)

    return {
        "review_run_id": review_run_id,
        "synthesis": output.model_dump(mode="json"),
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
    }
