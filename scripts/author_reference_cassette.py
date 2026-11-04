"""Track 2 plan Step 12A sub-step (d): authors `fixtures/corpus/replay_
cassette.json` (+ `cassette_sha256`) and `fixtures/spec/*.request.json`
(+ `fixtures/spec/request_snapshots.lock`) by actually running the real
scoping-review graph (`slr_pipeline.graphs.scoping.build_scoping_graph`)
against the real fixture corpus, with the OpenAI SDK's HTTP calls routed
(via `OPENAI_BASE_URL`) to a **generation-mode** variant of `ReplayServer`
that, instead of latching fatal on an unrecorded signature (the real
`slr_pipeline.replay.transport.ReplayServer`'s behaviour), synthesizes a
correct response for every one of the 8 replay surfaces from the frozen
`slr_pipeline.reference_fixture_answer_key` oracle, records it as a new
cassette entry, and serves it.

Usage:
    python scripts/author_reference_cassette.py

Writes:
    tests/reference_run/fixtures/corpus/replay_cassette.json
    tests/reference_run/fixtures/spec/<surface>.request.json  (x8)
    tests/reference_run/fixtures/spec/request_snapshots.lock
and upserts `cassette_sha256` into track2/reference_hashes.lock.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import sys
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slr_pipeline import reference_fixture_answer_key as ak  # noqa: E402
from slr_pipeline.config import ReviewConfig  # noqa: E402
from slr_pipeline.extraction.openai_runner import _SYSTEM_MESSAGE as _EXTRACTION_SYSTEM  # noqa: E402
from slr_pipeline.graphs.scoping import build_scoping_graph  # noqa: E402
from slr_pipeline.nodes.screening import BATCH_SYSTEM as _SCREENING_SYSTEM  # noqa: E402
from slr_pipeline.nodes.screening_qc import QC_SECOND_RATER_SYSTEM as _QC_SYSTEM  # noqa: E402
from slr_pipeline.nodes.synthesis import (  # noqa: E402
    _SYNTHESIS_MAP_SYSTEM_PROMPT,
    _SYNTHESIS_REFINE_SYSTEM_PROMPT,
)
from slr_pipeline.reference_pdf_placement import (  # noqa: E402
    REFERENCE_EXTRACTION_WORKSPACE_PARTS as _REFERENCE_EXTRACTION_WORKSPACE_PARTS,
    place_pdfs as _place_pdfs_shared,
)
from slr_pipeline.reference_spec import (  # noqa: E402
    EXTRACTION_PARSER_OPTIONS,
    FULL_TEXT_CRITERIA_SET,
    REFERENCE_REQUEST_SNAPSHOT_SURFACES,
    ReferenceRunSpec,
    _CANONICAL_DATABASES,
    _EXCLUSION_CRITERIA,
    _FUNDING_STATEMENT,
    _INCLUSION_CRITERIA,
    _DATA_GOVERNANCE_DECLARATION,
    _PCC_CONCEPT,
    _PCC_CONTEXT,
    _PCC_POPULATION,
    _RESEARCH_QUESTIONS,
    _REVIEW_RATIONALE,
    _REVIEW_TITLE,
    _SEED_QUERY,
    reference_run_config,
)
from slr_pipeline.replay.cassette import (  # noqa: E402
    REFERENCE_SURFACE_CALL_COUNTS,
    ReplayCassette,
    ReplayCassetteEntry,
)
from slr_pipeline.replay.signature import compute_replay_signature  # noqa: E402
from slr_pipeline.state import initial_state  # noqa: E402

import ft_screening.adjudication as _ftadj  # noqa: E402

from scripts.hash_reference_inputs import register_target, write_targets  # noqa: E402

FIXTURES_CORPUS_DIR = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "corpus"
FIXTURES_SPEC_DIR = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "spec"
CASSETTE_PATH = FIXTURES_CORPUS_DIR / "replay_cassette.json"
CORPUS_ORACLE_PATH = FIXTURES_CORPUS_DIR / "CORPUS_ORACLE.json"
REQUEST_SNAPSHOTS_LOCK_PATH = FIXTURES_SPEC_DIR / "request_snapshots.lock"

# FIXED (not tempfile.mkdtemp()-derived) workspace root. cfg.output_dir == this path, and
# fulltext.py derives its own local-PDF search directory as `output_dir / "pdfs"` -- so this
# path's own resolved string is what actually reaches extraction/run_identity.py::compute_run_id()
# as part of each document's pdf_path. compute_run_id() hashes pdf_path verbatim into
# extraction_run_id, which is embedded (via make_evidence_id()'s artifact_run_id field) into
# every evidence_id a synthesis map/refine call cites, and therefore into compute_answer_id()'s
# own hash input. A random per-run mkdtemp() workspace therefore made extraction_run_id -- and
# downstream, every synthesis_refine answer_id and its cascading request signature -- differ
# between two otherwise-identical runs of this script (discovered via a reproducibility diff:
# 46/50 cassette entries were byte-identical across two runs, the other 4 -- exactly the
# synthesis_refine entries -- were not). A fixed, repo-relative workspace instead makes
# pdf_path (and therefore extraction_run_id/evidence_id/answer_id) reproduce identically on
# the same checkout. Cleared at the start of every run (see main()) so stale checkpoints from
# a previous, possibly-aborted run can never leak in. The path itself now lives in
# `slr_pipeline.reference_pdf_placement.REFERENCE_EXTRACTION_WORKSPACE_PARTS` -- step 13's
# `reference_run_child.py` MUST resolve `cfg.output_dir` to this exact same absolute path to
# replay against the cassette this script produces (see that module's own comment) -- one
# source of truth, not two literals that could silently drift apart.
FIXED_WORKSPACE = REPO_ROOT.joinpath(*_REFERENCE_EXTRACTION_WORKSPACE_PARTS)

_PLACEHOLDER_HEX64 = "0" * 64
_SENTINEL = "<<RECORD-DEPENDENT>>"


@register_target("cassette_sha256")
def _cassette_target_path() -> Path:
    return CASSETTE_PATH


# --------------------------------------------------------------------------
# text-normalization helper, matching slr_pipeline.extraction.evidence's own
# NFC + whitespace-collapse substring-containment discipline
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------
# corpus lookups
# --------------------------------------------------------------------------

def _title_lookup() -> Dict[str, str]:
    return {ak.record(rid)["title"].strip(): rid for rid in ak.canonical_ids()}


def _record_signatures(rid: str) -> List[str]:
    """Distinctive phrases known to appear verbatim, somewhere, in `rid`'s
    fixture PDF -- title (always), every `EXTRACTION_ANSWERS` string field
    (5 of the 6 `FULL_TEXT_SOUGHT` records: not `fx-0018`, which is excluded
    before extraction ever runs), and every criterion's `FULL_TEXT_
    ELIGIBILITY` evidence phrase (embedded verbatim only in `fx-0018`'s PDF,
    empirically -- the other 5 records' PDFs paraphrase `Methods` as
    `Study design: {study_design}` instead). Used because FT_SCREENING's
    lexical_bm25 retriever returns only a handful of chunks per call, so no
    single field is reliably present in every call's EVIDENCE_JSON."""
    signatures = [ak.record(rid)["title"]]
    answers = ak.EXTRACTION_ANSWERS.get(rid)
    if answers:
        signatures.extend(v for v in answers.values() if isinstance(v, str) and v)
    signatures.extend(c["evidence"] for c in ak.FULL_TEXT_ELIGIBILITY[rid]["criteria"].values())
    return signatures


def _record_id_for_any_signature(haystack: str, candidate_ids: Tuple[str, ...]) -> str:
    normalized_haystack = _norm(haystack)
    for rid in candidate_ids:
        for sig in _record_signatures(rid):
            if _norm(sig) in normalized_haystack:
                return rid
    raise RuntimeError(f"could not identify record from content (candidates={candidate_ids}): {haystack[:300]!r}")


def _record_id_for_concept_substring(haystack: str, candidate_ids: Tuple[str, ...]) -> str:
    """Used only where `haystack` is `_build_map_context()`'s field_id:value
    rendering (synthesis_map) or a raw-PDF-chunk extraction batch limited to
    `FULL_TEXT_ELIGIBLE` records -- both always have a non-empty `concept`
    field (`EXTRACTION_ANSWERS` is keyed by `FULL_TEXT_ELIGIBLE` only, i.e.
    never `fx-0018`), unlike `ft_adjudication`, which also sees `fx-0018`
    (title-matched instead, see `_record_id_for_title_substring`)."""
    normalized_haystack = _norm(haystack)
    for rid in candidate_ids:
        concept = ak.EXTRACTION_ANSWERS[rid]["concept"]
        if _norm(concept) in normalized_haystack:
            return rid
    raise RuntimeError(f"could not identify record from content (candidates={candidate_ids}): {haystack[:300]!r}")


# --------------------------------------------------------------------------
# per-surface responders -- each takes the parsed request body dict and
# returns the dict that becomes json.dumps(...)'d into choices[0].message.content
# --------------------------------------------------------------------------

def _last_user_content(body: Dict[str, Any]) -> str:
    for msg in reversed(body["messages"]):
        if msg.get("role") == "user":
            return msg["content"]
    raise RuntimeError("no user message in request body")


def _respond_query_expansion(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    m = re.search(r"Current seed query:\n(.*?)\n\nInstructions:", user, re.S)
    seed_query = m.group(1).strip() if m else _SEED_QUERY
    return {"concepts": [], "candidate_query": seed_query, "notes": ["reference-cassette generation"]}


def _respond_db_translation(body: Dict[str, Any]) -> Dict[str, Any]:
    dbs_needed = sorted(set(_CANONICAL_DATABASES) | {"pubmed"})
    return {db: f"{db} translated query" for db in dbs_needed}


_TITLE_ITEM_RE = re.compile(r"\[(\d+)\] Title: (.*)\n    Abstract:")


def _rid_for_title(lookup: Dict[str, str], title: str) -> str:
    title = title.strip()
    if title not in lookup:
        raise RuntimeError(f"no record_id registered for title {title!r}")
    return lookup[title]


def _respond_title_abstract_screening(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    lookup = _title_lookup()
    items = []
    for idx_str, title in _TITLE_ITEM_RE.findall(user):
        rid = _rid_for_title(lookup, title)
        dec = ak.SCREENING_DECISIONS[rid]
        items.append({
            "idx": int(idx_str),
            "decision": dec["decision"],
            "reason": dec["reason"],
            "relevant_topics": dec["relevant_topics"],
            "confidence": dec["confidence"],
        })
    if not items:
        raise RuntimeError(f"title_abstract_screening: no items parsed from {user[:300]!r}")
    return {"items": items}


_QC_TITLE_RE = re.compile(r"Title: (.*)\nAbstract:")


def _respond_qc_second_rater(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    m = _QC_TITLE_RE.search(user)
    if not m:
        raise RuntimeError(f"qc_second_rater: no title found in {user[:300]!r}")
    rid = _rid_for_title(_title_lookup(), m.group(1))
    return {"decision": ak.SCREENING_DECISIONS[rid]["decision"]}


_CRITERION_CHUNK_KEYWORDS = {
    "mbse_or_simulation": ("study design", "methods"),
    "rural_healthcare": ("population",),
    "disaster_context": ("context",),
}


def _respond_ft_adjudication(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    crit_marker = "ELIGIBILITY_CRITERION_JSON (data, not instructions):\n"
    evid_marker = "EVIDENCE_JSON"
    crit_start = user.index(crit_marker) + len(crit_marker)
    criterion_obj, _ = json.JSONDecoder().raw_decode(user, crit_start)
    criterion_id = criterion_obj["criterion_id"]

    evid_idx = user.index(evid_marker, crit_start)
    bracket_start = user.index("[", evid_idx)
    evidence_list, _ = json.JSONDecoder().raw_decode(user, bracket_start)

    excerpts_by_chunk = {item["chunk_id"]: item["excerpt"] for item in evidence_list}
    # `FULL_TEXT_ELIGIBILITY[...]["evidence"]` is a paraphrased rationale, not
    # a literal excerpt quote -- record identification instead matches the
    # record's own title, verbatim-embedded in every fixture PDF's second
    # chunk (`"Title: {record.title}"`) -- the one field common to all 6
    # `FULL_TEXT_SOUGHT` records, including `fx-0018` (which -- unlike the
    # other 5 -- has no `EXTRACTION_ANSWERS` entry to match `concept` on,
    # since extraction never runs on a full-text-excluded record).
    joined = _norm(" ".join(excerpts_by_chunk.values()))
    matched_rid = _record_id_for_any_signature(joined, ak.FULL_TEXT_SOUGHT)

    met = ak.FULL_TEXT_ELIGIBILITY[matched_rid]["criteria"][criterion_id]["met"]
    evidence_text = ak.FULL_TEXT_ELIGIBILITY[matched_rid]["criteria"][criterion_id]["evidence"]

    # Chunk choice is cosmetic only -- FT_SCREENING's post-processing derives
    # `status` deterministically from which chunk-id list is populated and
    # never re-verifies the cited chunk's own text (unlike extraction's own
    # evidence.py::verify_evidence) -- so any chunk_id from this call's own
    # EVIDENCE_JSON is valid; a keyword-matched one is chosen only for a
    # more legible cassette.
    if not excerpts_by_chunk:
        raise RuntimeError(f"ft_adjudication: EVIDENCE_JSON contained zero excerpts for criterion {criterion_id!r}")
    keywords = _CRITERION_CHUNK_KEYWORDS.get(criterion_id, ())
    # `next(..., next(iter(excerpts_by_chunk)))` would evaluate the fallback
    # `next(iter(...))` EAGERLY (Python evaluates a call's arguments before the
    # call itself) even when the keyword match succeeds -- harmless only because
    # `excerpts_by_chunk` is now guaranteed non-empty above; written as an
    # explicit two-step fallback instead so a future empty-dict caller can never
    # reintroduce a bare StopIteration.
    matched_chunk_id = next(
        (cid for cid, excerpt in excerpts_by_chunk.items() if any(kw in excerpt.lower() for kw in keywords)),
        None,
    )
    if matched_chunk_id is None:
        matched_chunk_id = next(iter(excerpts_by_chunk))

    if met:
        supporting, contradicting, status = [matched_chunk_id], [], "satisfied"
    else:
        supporting, contradicting, status = [], [matched_chunk_id], "violated"
    return {
        "supporting_chunk_ids": supporting,
        "contradicting_chunk_ids": contradicting,
        "proposed_status": status,
        "rationale": evidence_text,
        "requires_human_review": False,
    }


_CHUNK_BLOCK_RE = re.compile(
    # The lookahead must also stop at the trailing "Respond with exactly one JSON
    # object..." instructions that follow the LAST chunk block in openai_runner.py's
    # own prompt template -- without this alternative, the non-greedy `.*?` for the
    # final chunk fell through to `\Z` instead, gluing that whole boilerplate tail
    # (including the JSON-shape schema) onto the last chunk's own captured text.
    r"\[chunk_id: (?P<id>[^|]+) \| section: (?P<section>[^|]+) \| page: (?P<page>[^\]]+)\]\n"
    r"(?P<text>.*?)(?=\n\n\[chunk_id:|\n\nRespond with exactly one JSON object|\Z)",
    re.S,
)

_EXTRACTION_FIELD_ORDER = (
    "country", "study_design", "population", "concept", "context",
    "limitations", "key_findings", "funding_sources", "funder_role",
)


def _respond_extraction_batch(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    chunks = [(m.group("id").strip(), _norm(m.group("text"))) for m in _CHUNK_BLOCK_RE.finditer(user)]
    if not chunks:
        raise RuntimeError(f"extraction_batch: no chunks parsed from {user[:300]!r}")
    joined = "\n".join(text for _, text in chunks)
    rid = _record_id_for_concept_substring(joined, ak.FULL_TEXT_ELIGIBLE)
    answers = ak.EXTRACTION_ANSWERS[rid]

    def _evidence_for(value: str) -> Tuple[str, str]:
        """Returns (chunk_id, quotation) where `quotation` is guaranteed to
        be a literal (normalized-)substring of that chunk's text --
        `evidence.py::verify_evidence()`'s only requirement. `value` (the
        answer-key text, always used as `RawFinding.value` regardless of
        this) usually appears verbatim too, but a handful of the synthetic
        fixture PDFs (e.g. fx-0024's long sentences) have their reading
        order locally scrambled by PDF text-reflow/OCR, splitting one
        answer-key sentence across two non-adjacent lines -- so the exact
        full string is sometimes not contiguous in any single chunk. Falls
        back to the longest contiguous run shared between `value` and a
        chunk's text (via `difflib`), which is always at least a long
        clause even when the whole sentence isn't reassembled."""
        norm_value = _norm(value)
        for chunk_id, text in chunks:
            if norm_value in text:
                return chunk_id, value
        best_chunk_id, best_span, best_len = chunks[0][0], "", 0
        for chunk_id, text in chunks:
            matcher = difflib.SequenceMatcher(None, norm_value, text, autojunk=False)
            match = matcher.find_longest_match(0, len(norm_value), 0, len(text))
            if match.size > best_len:
                best_chunk_id, best_span, best_len = chunk_id, norm_value[match.a:match.a + match.size], match.size
        if best_len < 15:
            # Unverified (chunk_id, value) pair here would fail far downstream,
            # deep inside evidence.py::verify_evidence() during graph.invoke(),
            # with this diagnostic already scrolled past. Fail at point of detection.
            raise RuntimeError(
                f"extraction_batch: no usable quotation found for {rid}: {value!r} "
                f"(best difflib match only {best_len} chars)"
            )
        return best_chunk_id, best_span.strip()

    fields: Dict[str, Any] = {}
    for field_id in _EXTRACTION_FIELD_ORDER:
        value = answers[field_id]
        if value is None or value == [] or value == "":
            fields[field_id] = {"outcome": "not_found", "findings": [], "rationale": None}
            continue
        items = value if isinstance(value, list) else [value]
        findings = []
        for item in items:
            chunk_id, quotation = _evidence_for(item)
            findings.append({
                "value": item,
                "confidence": "high",
                "rationale": None,
                "evidence": [{"chunk_id": chunk_id, "quotation": quotation}],
            })
        fields[field_id] = {"outcome": "found", "findings": findings, "rationale": None}
    return {"fields": fields}


_MAP_TARGET_RE = re.compile(
    r"Research question or synthesis target: (.*?)\n\nStudy data \(field_id: value\):\n(.*)", re.S
)


def _respond_synthesis_map(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    m = _MAP_TARGET_RE.match(user)
    if not m:
        raise RuntimeError(f"synthesis_map: could not parse {user[:300]!r}")
    target, rendered_context = m.group(1).strip(), m.group(2)
    if target == _RESEARCH_QUESTIONS[0].strip():
        rq_index = 0
    elif target == _RESEARCH_QUESTIONS[1].strip():
        rq_index = 1
    else:
        raise RuntimeError(f"synthesis_map: target matches neither known research question: {target!r}")
    rid = _record_id_for_concept_substring(rendered_context, ak.FULL_TEXT_ELIGIBLE)
    relevant = rid in ak.SYNTHESIS_MAP_RELEVANCE[f"rq_{rq_index}"]
    if not relevant:
        return {"relevant": False}
    present_fields = [fid for fid in _EXTRACTION_FIELD_ORDER if f"{fid}: " in rendered_context]
    answer = f"{ak.record(rid)['title']} provides evidence relevant to this research question."
    return {
        "relevant": True,
        "answer": answer,
        "supporting_selectors": [{"artifact_kind": "extraction", "field_id": fid} for fid in present_fields],
    }


_REFINE_TARGET_RE = re.compile(
    # Matches synthesis.py's real _SYNTHESIS_REFINE_USER_TEMPLATE verbatim: "New batch
    # of per-study answers:", not "New batch of new per-study answers:" -- an earlier,
    # dead-code alternative regex assumed the latter and was removed (never matched).
    r"Research question or synthesis target: (.*?)\n\n"
    r"Current claim candidates:\n(.*?)\n\n"
    r"New batch of per-study answers:\n(.*)",
    re.S,
)
_REFINE_ANSWER_LINE_RE = re.compile(r"^(\S+) \(([^)]+)\): (.*)$", re.M)


def _respond_synthesis_refine(body: Dict[str, Any]) -> Dict[str, Any]:
    user = _last_user_content(body)
    m = _REFINE_TARGET_RE.match(user)
    if not m:
        raise RuntimeError(f"synthesis_refine: could not parse {user[:300]!r}")
    target, new_answers_block = m.group(1).strip(), m.group(3)
    answer_lines = _REFINE_ANSWER_LINE_RE.findall(new_answers_block)
    if not answer_lines:
        raise RuntimeError(f"synthesis_refine: no answer lines parsed from {new_answers_block[:300]!r}")
    answer_ids = [aid for aid, _record, _text in answer_lines]
    citations = sorted({rec for _aid, rec, _text in answer_lines})
    claim_text = f"Evidence from {', '.join(citations)} addresses: {target[:150]}"
    return {
        "claim_updates": [
            {
                "claim_index": None,
                "claim_text": claim_text,
                "final_support_answer_ids": answer_ids,
            }
        ]
    }


_RESPONDERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "query_expansion": _respond_query_expansion,
    "db_translation": _respond_db_translation,
    "title_abstract_screening": _respond_title_abstract_screening,
    "qc_second_rater": _respond_qc_second_rater,
    "ft_adjudication": _respond_ft_adjudication,
    "extraction_batch": _respond_extraction_batch,
    "synthesis_map": _respond_synthesis_map,
    "synthesis_refine": _respond_synthesis_refine,
}

_QUERY_EXPANSION_USER_PREFIX = "You are designing a transparent database search strategy"
_DB_TRANSLATION_USER_PREFIX = "You are adapting a PubMed literature search query"


def _dispatch_surface(body: Dict[str, Any]) -> str:
    system = ""
    user = ""
    for msg in body["messages"]:
        if msg.get("role") == "system" and not system:
            system = msg["content"]
        if msg.get("role") == "user":
            user = msg["content"]
    if system == _SCREENING_SYSTEM:
        return "title_abstract_screening"
    if system == _QC_SYSTEM:
        return "qc_second_rater"
    if system == _ftadj._SYSTEM_PROMPT:
        return "ft_adjudication"
    if system == _EXTRACTION_SYSTEM:
        return "extraction_batch"
    if system == _SYNTHESIS_MAP_SYSTEM_PROMPT:
        return "synthesis_map"
    if system == _SYNTHESIS_REFINE_SYSTEM_PROMPT:
        return "synthesis_refine"
    if user.startswith(_QUERY_EXPANSION_USER_PREFIX):
        return "query_expansion"
    if user.startswith(_DB_TRANSLATION_USER_PREFIX):
        return "db_translation"
    raise RuntimeError(f"could not dispatch request to a surface: system={system[:120]!r} user={user[:120]!r}")


# --------------------------------------------------------------------------
# generation-mode HTTP server
# --------------------------------------------------------------------------

class _CassetteEntryRecord:
    __slots__ = ("entry_id", "surface", "signature", "endpoint", "request_preview", "response_body", "served_calls")

    def __init__(self, entry_id: str, surface: str, signature: str, request_preview: str, response_body: Dict[str, Any]):
        self.entry_id = entry_id
        self.surface = surface
        self.signature = signature
        self.endpoint = "/v1/chat/completions"
        self.request_preview = request_preview
        self.response_body = response_body
        self.served_calls = 0


class _GenerationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.entries_by_signature: Dict[str, _CassetteEntryRecord] = {}
        self.surface_counts: Dict[str, int] = {name: 0 for name in _RESPONDERS}
        self.first_request_by_surface: Dict[str, Dict[str, Any]] = {}
        self.error: Optional[BaseException] = None


class _GenerationHandler(BaseHTTPRequestHandler):
    server: _GenerationHTTPServer

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            with self.server.lock:
                surface = _dispatch_surface(body)
                if surface not in self.server.first_request_by_surface:
                    self.server.first_request_by_surface[surface] = body
                self.server.surface_counts[surface] += 1
                responder = _RESPONDERS[surface]
                response_obj = responder(body)
                signature = compute_replay_signature(self.path, body)
                entry = self.server.entries_by_signature.get(signature)
                content_str = json.dumps(response_obj)
                envelope = {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content_str, "refusal": None},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                if entry is None:
                    entry_id = f"{surface}-{self.server.surface_counts[surface]:03d}"
                    entry = _CassetteEntryRecord(
                        entry_id=entry_id,
                        surface=surface,
                        signature=signature,
                        # cassette.py's ReplayCassetteEntry contract: "first 200 chars
                        # of the user message -- human triage only", not the full body.
                        request_preview=_last_user_content(body)[:200],
                        response_body=envelope,
                    )
                    self.server.entries_by_signature[signature] = entry
                entry.served_calls += 1
            payload = json.dumps(envelope).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except BaseException as exc:  # noqa: BLE001
            self.server.error = exc
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


# --------------------------------------------------------------------------
# PDF placement table (plan §6.3)
# --------------------------------------------------------------------------

# Table + copy-and-verify logic now live in `slr_pipeline.reference_pdf_
# placement` (a shipped module, unlike this script, imported at the top of
# this file) -- step 13's `reference_run_child.py` needs the exact same
# table to reproduce the same `pdf_path` strings this script baked into the
# frozen cassette, so it is a single source of truth, not a second copy.
# See that module's docstring for the "why extracted, not duplicated"
# rationale.


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _place_pdfs(pdf_dir: Path, oracle: Dict[str, Any]) -> None:
    _place_pdfs_shared(pdf_dir, oracle, FIXTURES_CORPUS_DIR / "pdfs")


# --------------------------------------------------------------------------
# request-snapshot sentinel templating (plan §4.3)
# --------------------------------------------------------------------------

_SNAPSHOT_KEYS = ("model", "messages", "response_format", "json_schema", "temperature",
                   "reasoning_effort", "max_completion_tokens", "seed", "n")


def _sentinel_user_content(surface: str, content: str) -> str:
    if surface == "title_abstract_screening":
        idx = content.find("Papers:\n")
        if idx == -1:
            return content
        tail_idx = content.find("\n\nReturn", idx)
        prefix = content[: idx + len("Papers:\n")]
        suffix = content[tail_idx:] if tail_idx != -1 else ""
        return prefix + _SENTINEL + suffix
    if surface == "qc_second_rater":
        return _QC_TITLE_RE.sub(f"Title: {_SENTINEL}\nAbstract:", content, count=1)
    if surface == "ft_adjudication":
        idx = content.find("ELIGIBILITY_CRITERION_JSON")
        return content[:idx] + _SENTINEL if idx != -1 else content
    if surface == "extraction_batch":
        idx = content.find("[chunk_id:")
        if idx == -1:
            return content
        tail_idx = content.find("\n\nRespond", idx)
        suffix = content[tail_idx:] if tail_idx != -1 else ""
        return content[:idx] + _SENTINEL + suffix
    if surface == "synthesis_map":
        idx = content.find("Study data (field_id: value):\n")
        if idx == -1:
            return content
        prefix = content[: idx + len("Study data (field_id: value):\n")]
        return prefix + _SENTINEL
    if surface == "synthesis_refine":
        idx = content.find("Current claim candidates:\n")
        return content[:idx] + "Current claim candidates:\n" + _SENTINEL if idx != -1 else content
    return content


def _build_request_snapshot(surface: str, body: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = {k: body[k] for k in _SNAPSHOT_KEYS if k in body}
    messages = []
    for msg in snapshot.get("messages", []):
        msg = dict(msg)
        if msg.get("role") == "user":
            msg["content"] = _sentinel_user_content(surface, msg["content"])
        messages.append(msg)
    snapshot["messages"] = messages
    return snapshot


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _build_generation_spec() -> ReferenceRunSpec:
    from scripts.hash_reference_inputs import _load_lock
    locked = _load_lock()

    def _h(name: str) -> str:
        return locked[name]["sha256"]

    databases = list(_CANONICAL_DATABASES)
    return ReferenceRunSpec(
        review_title=_REVIEW_TITLE,
        review_rationale=_REVIEW_RATIONALE,
        research_questions=_RESEARCH_QUESTIONS,
        pcc_population=_PCC_POPULATION,
        pcc_concept=_PCC_CONCEPT,
        pcc_context=_PCC_CONTEXT,
        inclusion_criteria=_INCLUSION_CRITERIA,
        exclusion_criteria=_EXCLUSION_CRITERIA,
        seed_query=_SEED_QUERY,
        protocol_sha256=_h("protocol_sha256"),
        databases=databases,
        execution_adapter_map={db: f"fx_{db}" for db in databases},
        qc_saturation_levels=[0.80, 0.90, 0.95],
        extraction_schema_sha256=_h("extraction_schema_sha256"),
        extraction_model_config_sha256=_h("extraction_model_config_sha256"),
        extraction_parser_options=EXTRACTION_PARSER_OPTIONS,
        full_text_criteria_set=FULL_TEXT_CRITERIA_SET,
        ft_screening_config_sha256=_h("ft_screening_config_sha256"),
        request_snapshot_sha256={name: _PLACEHOLDER_HEX64 for name in REFERENCE_REQUEST_SNAPSHOT_SURFACES},
        funding_statement=_FUNDING_STATEMENT,
        data_governance_declaration=_DATA_GOVERNANCE_DECLARATION,
        reference_reporting_context_sha256=_h("reference_reporting_context_sha256"),
        fixture_corpus_digest=_PLACEHOLDER_HEX64,
        prisma_scr_matrix_sha256=_h("prisma_scr_matrix_sha256"),
    )


def main() -> int:
    import shutil

    oracle = json.loads(CORPUS_ORACLE_PATH.read_text(encoding="utf-8"))

    # Fixed, not tempfile.mkdtemp()-derived -- see FIXED_WORKSPACE's own docstring comment above.
    # That fixedness means two concurrent invocations would race on rmtree/mkdir/PDF
    # placement and silently corrupt each other's output -- an O_CREAT|O_EXCL sentinel
    # makes a second concurrent run fail loudly instead.
    workspace = FIXED_WORKSPACE
    workspace.parent.mkdir(parents=True, exist_ok=True)
    lock_path = workspace.parent / ".cassette_generation_workspace.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{lock_path} already exists -- another author_reference_cassette.py run is in "
            f"progress (or a previous run crashed without cleaning up); remove it if stale"
        )
    os.close(lock_fd)
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        pdf_dir = workspace / "pdfs"
        _place_pdfs(pdf_dir, oracle)

        spec = _build_generation_spec()
        cfg = reference_run_config(spec)
        cfg_dict = cfg.to_dict()
        cfg_dict["output_dir"] = str(workspace)
        cfg = ReviewConfig.from_dict(cfg_dict)

        server = _GenerationHTTPServer(("127.0.0.1", 0), _GenerationHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        env_backup = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SLR_FIXTURE_CORPUS_ROOT")}
        os.environ["OPENAI_API_KEY"] = "sk-replay-not-a-real-key"
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["SLR_FIXTURE_CORPUS_ROOT"] = str(FIXTURES_CORPUS_DIR)
        try:
            graph = build_scoping_graph()
            state = initial_state(cfg.to_dict())
            config = {"configurable": {"thread_id": "track2-cassette-generation"}}
            graph.invoke(state, config)
        finally:
            server.shutdown()
            thread.join(timeout=10)
            for k, v in env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        if server.error is not None:
            raise server.error

        expected_counts = dict(REFERENCE_SURFACE_CALL_COUNTS)
        if server.surface_counts != expected_counts:
            raise RuntimeError(
                f"served call counts do not match REFERENCE_SURFACE_CALL_COUNTS:\n"
                f"  got:      {server.surface_counts}\n"
                f"  expected: {expected_counts}"
            )

        entries = sorted(server.entries_by_signature.values(), key=lambda e: e.entry_id)
        cassette = ReplayCassette(entries=[
            ReplayCassetteEntry(
                entry_id=e.entry_id,
                surface=e.surface,
                signature=e.signature,
                endpoint=e.endpoint,
                request_preview=e.request_preview,
                response_body=e.response_body,
                expected_calls=e.served_calls,
            )
            for e in entries
        ])
        CASSETTE_PATH.write_text(
            json.dumps(cassette.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        FIXTURES_SPEC_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_hashes: Dict[str, str] = {}
        for surface in REFERENCE_REQUEST_SNAPSHOT_SURFACES:
            body = server.first_request_by_surface.get(surface)
            if body is None:
                raise RuntimeError(f"surface {surface!r} was never called -- cannot build request snapshot")
            snapshot = _build_request_snapshot(surface, body)
            path = FIXTURES_SPEC_DIR / f"{surface}.request.json"
            path.write_text(_canonical_json_dumps(snapshot), encoding="utf-8")
            snapshot_hashes[surface] = _sha256_file(path)
        REQUEST_SNAPSHOTS_LOCK_PATH.write_text(_canonical_json_dumps(snapshot_hashes), encoding="utf-8")

        write_targets(["cassette_sha256"])

        print(f"wrote {CASSETTE_PATH.relative_to(REPO_ROOT)} ({len(entries)} entries)")
        print(f"wrote {len(REFERENCE_REQUEST_SNAPSHOT_SURFACES)} request snapshot(s) + {REQUEST_SNAPSHOTS_LOCK_PATH.relative_to(REPO_ROOT)}")
        print(f"surface call counts: {server.surface_counts}")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
