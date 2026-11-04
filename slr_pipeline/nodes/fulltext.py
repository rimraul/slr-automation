"""
fulltext_node -- Step 4a of the SLR pipeline.

Two clearly separated layers:

  1. ACQUISITION (this module, unchanged in substance from the pre-
     FT_SCREENING version): locate-or-download a PDF, extract text, and
     apply a content-quality gate (stub/preview detection). Produces, per
     record, a `full_text_status` and -- only when status == "parsed" --
     a locally CONFIRMED PDF path. A record that never gets a confirmed PDF
     is either queued into `full_text_review_queue` (default,
     `cfg.full_text_acquisition_failure_policy == "review"`) or dropped as a
     terminal "report not retrieved" outcome (`"exclude"`, required for a
     fully autonomous run) -- the latter is NOT a PRISMA `excluded_full_text`
     count (that box is reserved for FT_SCREENING-evaluated exclusions with
     a ledger decision behind them); it is represented solely via
     `prisma.fulltext_retrieval`'s non-"parsed" counts -- see config.py's
     field docstring.
  2. FULL-TEXT ELIGIBILITY (delegated entirely to FT_SCREENING via
     `slr_pipeline.adapters.ft_screening_adapter`): every confirmed PDF,
     and ONLY a confirmed PDF, is sent through a real FT_SCREENING run
     against an explicitly-authored `CriteriaSet`
     (`cfg.full_text_criteria_set` -- see config.py; never derived from
     `cfg.inclusion_criteria`/`exclusion_criteria`). No in-node LLM call,
     no abstract/title fallback, no include-on-error: a record only ever
     reaches `full_text_records` via an explicit FT_SCREENING
     `final_decision == "include"`. Everything else (needs_human_review,
     processing/provider failure, missing/duplicate input, AND records
     that never got a confirmed PDF in layer 1 at all) lands in
     `full_text_review_queue` instead -- never silently dropped, never
     silently included.

Retrieval (layer 1) follows the same three-tier pattern used across the
group's other review projects (see c:/projects/DOWNLOADING/CLAUDE.md for
the full methodology and known bugs):

  Tier 1 -- fully automated OA cascade (implemented here): Unpaywall ->
            Semantic Scholar -> EuropePMC -> arXiv/direct URL. Measured yield
            on real reviews is ~30% of DOI-bearing records across two
            independent corpora -- plan for it as a real number, not a guess.
  Tier 2 -- semi-automated browser download for papers that need an
            institutional login. Not reproducible inside a book example
            (it drives a live Chrome window); left as a described extension,
            not implemented here.
  Tier 3 -- manual: this module writes a worklist
            (manual_download_worklist.csv) for every record Tier 1 could not
            confirm, with the exact filename to save as so a
            manually-downloaded PDF is picked up automatically on the next
            run -- no fuzzy matching needed.

Node input keys:  included_records, config, prisma_counts
Node output keys: full_text_records, full_text_review_queue, prisma_counts,
                   manual_worklist_path, ft_screening_run_id,
                   ft_screening_execution_fingerprint,
                   ft_screening_artifact_paths

Credentials: THIS MODULE never reads, holds, or serialises an API key --
`run_ft_screening()` (via `resolve_components()`) reads `OPENAI_API_KEY`
directly from the process environment, and neither this module nor
FT_SCREENING's own request/artifact models have any field a credential
could be serialised into. Track 2's packaging de-hack (§2.3/§2.7) removed
the `my_api_keys`/`c:/projects` loader from `nodes/search.py` entirely --
credentials are read from `os.environ` only, everywhere in the package, so
importing `nodes.fulltext` (or any other `slr_pipeline.nodes.*` module) no
longer has any transitive credential-loading side effect to document.
`ReviewConfig.to_dict()`/`model_dump()` no longer serialise
`openai_api_key` at all (Track 1 Phase A credential-safety fix,
`config.py` -- the field is `Field(exclude=True, repr=False)` and is never
used for authentication; `config.resolve_openai_api_key()` reads
`OPENAI_API_KEY` from the environment exclusively).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from slr_pipeline.state import Record, PRISMACounts, ReviewState
from slr_pipeline.config import ReviewConfig
from slr_pipeline.doi import normalise_doi
from slr_pipeline.adapters.ft_screening_adapter import (
    build_full_text_screening_request,
    chapter_screening_config,
    map_run_result_to_chapter_outcomes,
    record_to_screening_item,
)
from ft_screening import CriteriaSet, ScreeningConfig, ScreeningItem, run_ft_screening

from slr_pipeline.run_identity import resolve_review_run_id
from slr_pipeline.human_resolution import (
    append_resolution_log_entry_deduplicated,
    compute_resolution_id,
    preflight_check_baseline_still_current,
)
from slr_pipeline.selection_ledger import (
    SelectionDecisionStore, derive_source_event_id, emit_selection_decision,
    is_fulltext_exclusion_decision, ledger_automation_origin_and_oversight,
)

logger = logging.getLogger(__name__)

FULL_TEXT_ELIGIBILITY_STAGE = "full_text_eligibility"


def _full_text_origin_and_oversight(ledger_store: SelectionDecisionStore) -> Tuple[Optional[str], Optional[str]]:
    """Thin, stage-bound wrapper over the shared
    `selection_ledger.ledger_automation_origin_and_oversight()`
    (aggregation/export/reporting plan v17, §2.8 -- extracted from this
    function's own prior body, now reused by reporting's PRISMA-adjacent
    re-derivation too). Kept as a local name since every call site in
    this module already refers to it this way."""
    return ledger_automation_origin_and_oversight(ledger_store, FULL_TEXT_ELIGIBILITY_STAGE)

# Content-quality gate: a downloaded PDF that LOOKS like a success but isn't
# real full text (book-preview PDFs, "abstract only" stubs). Caught here
# rather than trusted post-hoc, because a stub silently used as "full text"
# poisons every stage downstream (FT_SCREENING, extraction, synthesis).
_STUB_MARKERS = (
    "abstract only", "preview only", "this is a preview", "book preview",
    "sample pages", "no full text available",
)


# ---------------------------------------------------------------------------
# Tier 1 -- open-access PDF location resolvers
# ---------------------------------------------------------------------------

def _normalize_doi(doi: str) -> str:
    """Strip a 'https://doi.org/' URL wrapper down to the bare DOI. Some
    sources (OpenAlex) return doi as a full URL rather than a bare string --
    left un-normalized, the ':' in 'https:' ends up in any filename built
    from it, which Windows/NTFS silently reinterprets as an alternate-data-
    stream separator instead of raising an error. Delegates to the shared
    `slr_pipeline.doi.normalise_doi` (Track 2 plan §3.3) -- previously an
    independent copy using `str.lstrip(prefix)`, a character-set strip, not
    a prefix strip; see that module's docstring for the exact failure case."""
    return normalise_doi(doi)


def _resolve_unpaywall(doi: str, email: str) -> str:
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        with urllib.request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read())
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or ""
    except Exception as e:
        # Unpaywall rejects placeholder addresses (e.g. anything @example.com)
        # with a 422 "use your own email" -- surfaced here instead of swallowed,
        # since a silent failure here looks identical to "genuinely not OA".
        logger.debug(f"Unpaywall lookup failed for {doi}: {e}")
        return ""


def _resolve_semantic_scholar(doi: str) -> str:
    try:
        req = urllib.request.Request(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read())
        return (data.get("openAccessPdf") or {}).get("url") or ""
    except Exception:
        return ""


def _resolve_europepmc(doi: str) -> str:
    try:
        q = urllib.parse.quote(f'DOI:"{doi}"')
        url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
               f"?query={q}&format=json&resultType=core")
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        results = (data.get("resultList") or {}).get("result") or []
        for r in results[:1]:
            for ft in (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
                if ft.get("documentStyle") == "pdf" and ft.get("availability") == "OA":
                    return ft.get("url") or ""
    except Exception:
        pass
    return ""


def _resolve_oa_url(record: Record, contact_email: Optional[str]) -> Tuple[str, str]:
    """Try each Tier-1 resolver in priority order. Returns (pdf_url, source_name)
    or ("", "") if none of them found anything."""
    if record.doi:
        doi = _normalize_doi(record.doi)
        resolvers = []
        if contact_email:
            resolvers.append((lambda d: _resolve_unpaywall(d, contact_email), "unpaywall"))
        resolvers += [
            (_resolve_semantic_scholar, "semantic_scholar"),
            (_resolve_europepmc, "europepmc"),
        ]
        for resolver, source in resolvers:
            url = resolver(doi)
            if url:
                return url, source

    if record.url and record.url.lower().endswith(".pdf"):
        return record.url, "direct_url"

    if record.source == "arxiv" and record.source_id:
        arxiv_id = record.source_id.replace("http://arxiv.org/abs/", "")
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv"

    return "", ""


# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------

def _find_local_pdf(record: Record, pdf_dir: Path) -> Optional[Path]:
    """Check if a PDF already exists locally (by DOI or source id) -- this is
    also where a manually- or Tier-2-downloaded file gets picked up, since
    both use the same safe_id(record) naming convention as the destination
    path computed below."""
    if record.doi:
        doi_slug = _normalize_doi(record.doi).replace("/", "_").replace(".", "_")
        candidate = pdf_dir / f"{doi_slug}.pdf"
        if candidate.exists():
            return candidate
    candidate = pdf_dir / f"{record.source}_{record.source_id}.pdf"
    if candidate.exists():
        return candidate
    return None


def _dest_path(record: Record, pdf_dir: Path) -> Path:
    doi_slug = (_normalize_doi(record.doi) if record.doi else f"{record.source}_{record.source_id}")
    doi_slug = doi_slug.replace("/", "_").replace(".", "_")
    return pdf_dir / f"{doi_slug}.pdf"


def _download_pdf(
    pdf_url: str,
    dest: Path,
    max_retries: int = 3,
    base_wait: float = 4.0,
) -> bool:
    """Robust PDF download with exponential backoff and content validation.
    Handles rate-limiting (429), server errors (5xx), HTML-instead-of-PDF
    responses, empty bodies, and too-small files -- publisher/API responses
    fail in all of these ways in practice, not just with a clean 404."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    for attempt in range(max_retries):
        time.sleep(random.uniform(1.0, 2.0))   # polite throttle
        try:
            resp = requests.get(pdf_url, headers=headers, timeout=45,
                                 stream=True, allow_redirects=True)
        except Exception as e:
            logger.debug(f"Request error: {e}")
            resp = None

        if resp is None or resp.status_code in (429, 500, 502, 503, 504):
            if resp is not None:
                resp.close()
            time.sleep(base_wait * (2 ** attempt))
            continue

        # From here on this attempt is final (return, not continue) -- every
        # exit path below must close `resp` itself before leaving, since a
        # `stream=True` response otherwise keeps its urllib3 connection
        # checked out until GC. A batch fulltext run resolves OA URLs for
        # dozens-to-hundreds of records, so this is not a theoretical leak.
        try:
            if resp.status_code != 200:
                return False

            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                return False

            content_iter = resp.iter_content(chunk_size=8192)
            try:
                first_chunk = next(content_iter)
            except StopIteration:
                return False
            except requests.exceptions.RequestException as e:
                # Connection dropped before any body bytes arrived (server
                # reset right after headers) -- treat like any other "not
                # retrievable" outcome, not an exception the caller loop
                # over records has to handle.
                logger.debug(f"Stream error while downloading {pdf_url!r}: {e}")
                return False
            if not first_chunk.startswith(b"%PDF") and "pdf" not in content_type:
                return False

            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(first_chunk)
                    for chunk in content_iter:
                        if chunk:
                            f.write(chunk)
            except (requests.exceptions.RequestException, OSError) as e:
                # RequestException: connection dropped or was truncated
                # mid-stream (publisher servers reset connections routinely).
                # OSError: disk full / permission denied / invalid path --
                # a real risk across "dozens-to-hundreds" of PDFs in one
                # batch run. Either way, treat exactly like any other "this
                # record's PDF wasn't retrievable" failure, not an exception
                # that aborts the whole batch.
                logger.debug(f"Download error for {pdf_url!r}: {e}")
                try:
                    dest.unlink(missing_ok=True)
                except OSError as unlink_err:
                    # e.g. a transient AV/indexer lock on Windows. Swallowing
                    # this silently would leave a stray/truncated file at
                    # `dest`; `_find_local_pdf` only checks existence, so on
                    # every future run it would be picked up as-is and
                    # `_resolve_oa_url`/`_download_pdf` would never be
                    # retried for this record again. Log loudly so an
                    # operator can find and remove it.
                    logger.warning(
                        f"Could not remove stray/truncated download {dest} after a "
                        f"failed attempt for {pdf_url!r} ({unlink_err}) -- this file "
                        f"will be picked up as-is on the next run unless removed manually."
                    )
                return False

            try:
                too_small = dest.stat().st_size < 1024
            except OSError as stat_err:
                # A fully-written file that can't be stat()'d right after
                # close (same AV/indexer-lock class as above) is left in
                # place unvalidated -- log it for the same reason.
                logger.warning(
                    f"Could not verify size of {dest} after download from "
                    f"{pdf_url!r} ({stat_err}) -- leaving it in place unvalidated."
                )
                return False
            if too_small:
                try:
                    dest.unlink(missing_ok=True)
                except OSError as unlink_err:
                    logger.warning(
                        f"Could not remove undersized download {dest} from "
                        f"{pdf_url!r} ({unlink_err}) -- this file will be picked up "
                        f"as-is on the next run unless removed manually."
                    )
                return False
            return True
        finally:
            resp.close()

    return False


def _extract_text(pdf_path: Path, max_chars: int = 120_000) -> str:
    """Used ONLY for the stub-detection gate below -- FT_SCREENING does its
    own real (Docling) parsing for the actual eligibility decision; this
    extraction never feeds a screening prompt."""
    try:
        import pymupdf4llm
    except ImportError:
        logger.warning("pymupdf4llm not installed -- cannot extract PDF text")
        return ""
    try:
        text = pymupdf4llm.to_markdown(doc=str(pdf_path), write_images=False, margins=0)
    except Exception as e:
        logger.debug(f"PDF text extraction failed for {pdf_path}: {e}")
        return ""
    return (text or "").strip()[:max_chars]


def _looks_like_stub(text: str, pdf_path: Path) -> Optional[str]:
    """Returns a reason string if this looks like a false-positive "success"
    (preview/stub, not real full text), else None. Does NOT try to catch
    genuine partial content (e.g. one real open-access chapter of a
    multi-chapter book) -- that's a different, legitimate category that
    needs its own disclosed label and can't be told apart from a stub by
    content alone."""
    lower = text.lower()[:2000]
    for marker in _STUB_MARKERS:
        if marker in lower:
            return f"stub marker: {marker!r}"
    if len(text) < 800:
        return f"too short to be real full text ({len(text)} chars)"
    if re.search(r"^\d{9,13}_preview", pdf_path.stem.lower()):
        return "generic ISBN-preview filename pattern"
    return None


# ---------------------------------------------------------------------------
# Tier 3 -- manual download worklist
# ---------------------------------------------------------------------------

def _write_manual_worklist(records: List[Record], out_path: Path) -> None:
    """One row per record that Tier 1 could not confirm. "save_as" gives the
    exact filename a human should save the PDF under so _find_local_pdf picks
    it up automatically on the next pipeline run -- no fuzzy title matching
    needed, as long as the record has a DOI or source id."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "doi", "reason", "save_as", "doi_link", "scholar_search"])
        for r in records:
            doi = _normalize_doi(r.doi) if r.doi else ""
            slug = (doi or f"{r.source}_{r.source_id}").replace("/", "_").replace(".", "_")
            writer.writerow([
                r.title,
                doi,
                r.full_text_status or "",
                f"{slug}.pdf",
                f"https://doi.org/{doi}" if doi else "",
                f"https://scholar.google.com/scholar?q={urllib.parse.quote(r.title)}",
            ])


# ---------------------------------------------------------------------------
# FT_SCREENING full-text eligibility (layer 2) -- confirmed PDFs only
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    """Streamed (1 MiB chunks, matching FT_SCREENING's own file-hashing
    convention) -- never reads a whole PDF into memory just to compute a
    run-identity hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_run_id(items: List[ScreeningItem], criteria_set: CriteriaSet, config: ScreeningConfig) -> str:
    """A deterministic run identity for THIS EXACT full-text eligibility
    request: the SAME ordered items + criteria_set + config always hash
    to the SAME run_id. This is what makes a retry after an interrupted
    process resumable (codex review, Phase B finding 2): if the process
    dies mid-run, LangGraph never received this node's return value, so
    there is nothing to read back from `ReviewState` on retry --
    recomputing the identical hash from the identical inputs is the only
    way to rediscover "this is the same request as before."

    Built from the ACTUAL ordered `ScreeningItem` values FT_SCREENING
    itself will bind this run to, plus a real, freshly-streamed PDF
    content hash per item (codex review, Phase B round-3 finding: hashing
    only `(record.id, pdf_sha256)` -- round-2's fix for the item-batch gap
    -- still missed fields FT_SCREENING's own `request_fingerprint`/
    provisional binding actually cover. Reproduced directly: same
    record.id, same PDF bytes, but a corrected `citation["title"]`
    produced the IDENTICAL run_id here, while FT_SCREENING's real
    `request_fingerprint` -- which includes the complete citation --
    would differ and correctly reject the "resume" with
    `ResumeConflictError`. `report_id`/`study_id`/DOI/year/source/
    source_id corrections have the same problem, since all of them ride
    inside `citation`.):
      - `document_id`, `report_id`, `study_id` -- FT_SCREENING's own
        per-item identity fields.
      - the COMPLETE `citation` dict -- not just title; any citation field
        (doi, year, source, source_id, ...) is bound into FT_SCREENING's
        real `request_fingerprint`, so any of them changing must change
        this hash too.
      - the item's PDF path, NORMALIZED (`Path(...).resolve()`) -- covers
        FT_SCREENING's provisional-manifest binding, which uses the
        declared path before content hashing completes; the same bytes
        moved to a different path is a different request from
        FT_SCREENING's own point of view during that window, even though
        the eventual content hash would match.
      - the item's actual PDF content, freshly streamed and hashed here
        (never trusted from a prior computation) -- covers the eventual,
        confirmed content identity.
      - `criteria_set` and `config`.

    Order matters, not just the set -- matches FT_SCREENING's own
    `request_fingerprint`, which is itself computed over ORDERED item
    bindings. A genuinely different batch (round-2's A-then-A+B case)
    still hashes differently and completes as a fresh run rather than
    colliding with an incompatible resume; an IDENTICAL retry -- same
    items, same citation, same path, same bytes -- still hashes
    identically and genuinely resumes.
    """
    items_payload = [
        {
            "document_id": item.document_id,
            "report_id": item.report_id,
            "study_id": item.study_id,
            "citation": item.citation,
            "pdf_path": str(Path(item.pdf_path).resolve()),
            "pdf_sha256": _file_sha256(Path(item.pdf_path)),
        }
        for item in items
    ]
    payload = {
        "items": items_payload,
        "criteria_set": criteria_set.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"ftrun-{digest[:32]}"


def _resolve_run_identity(
    ft_output_dir: Path, items: List[ScreeningItem], criteria_set: CriteriaSet, config: ScreeningConfig,
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (run_id, resume_run_id) -- exactly one is set, matching
    ScreeningRequest's own at-most-one contract. A run directory already
    existing under this stable id means a PRIOR attempt at this EXACT
    request (this run or an interrupted one) already started it -- resume
    rather than declaring a fresh run_id, which FT_SCREENING would
    otherwise reject as a collision."""
    stable_id = _stable_run_id(items, criteria_set, config)
    run_dir = ft_output_dir / "runs" / stable_id
    if run_dir.exists():
        return None, stable_id
    return stable_id, None


def _run_full_text_eligibility(
    confirmed: List[Tuple[Record, Path]],
    cfg: ReviewConfig,
    output_dir: Path,
    ledger_store: SelectionDecisionStore,
) -> Tuple[List[Record], List[Dict[str, Any]], int, Dict[str, int], Optional[str], Optional[str], Optional[Dict[str, str]]]:
    """Runs ONE real FT_SCREENING request over every confirmed (record, pdf_path)
    pair and returns (passing_records, ft_screening_queue_entries,
    excluded_count, excluded_reason_counts, run_id, execution_fingerprint,
    artifact_paths). Never called with an empty `confirmed` list by the
    caller below (nothing to screen -> nothing to configure/run).

    Stage 3 (the Stage 3 node-instrumentation design record): every record whose
    FT_SCREENING `execution_status == "completed"` (decision genuinely
    include/exclude/needs_human_review -- a real assessment happened,
    whatever its outcome) gets an `ai` `SelectionDecisionRecord`
    (`stage="full_text_eligibility"`), using the ITEM-level
    `outcome.ft_screening_result_id` as `producer_artifact_id` (not the
    shared batch `run_id` -- the more precise, already-available audit
    link). A record whose outcome never reached "completed" gets NO `ai`
    entry at all -- a later human decision for it is the chain's own
    root, never a supersession of a fictional AI decision that never
    happened. Every `ft_screening`-stage queue entry carries its own
    `reviewed_baseline_decision_id` (the just-emitted `ai` entry's
    `decision_id`), captured HERE at queue-creation time, never
    recomputed later at resolution time (F12)."""
    if not cfg.full_text_criteria_set:
        raise ValueError(
            "ReviewConfig.full_text_criteria_set is not set, but "
            f"{len(confirmed)} record(s) have a confirmed full-text PDF ready for eligibility "
            "screening. A full-text CriteriaSet must be authored explicitly (see "
            "slr_pipeline.adapters.ft_screening_adapter's module docstring) -- it is never "
            "derived from ReviewConfig.inclusion_criteria/exclusion_criteria."
        )
    criteria_set = CriteriaSet.model_validate(cfg.full_text_criteria_set)
    # `is not None`, not truthiness: an explicit but EMPTY/malformed
    # override ({} or a dict missing required fields) must reach
    # ScreeningConfig.model_validate() and fail loudly, never be silently
    # treated as "no override, use the chapter preset" the way `if
    # cfg.full_text_screening_config:` would (codex review, Phase B
    # finding 3).
    screening_config = (
        ScreeningConfig.model_validate(cfg.full_text_screening_config)
        if cfg.full_text_screening_config is not None
        else chapter_screening_config()
    )

    items = [record_to_screening_item(record, pdf_path=pdf_path) for record, pdf_path in confirmed]
    ft_output_dir = output_dir / "ft_screening_runs"
    run_id, resume_run_id = _resolve_run_identity(ft_output_dir, items, criteria_set, screening_config)
    request = build_full_text_screening_request(
        items=items, criteria_set=criteria_set, config=screening_config,
        output_dir=ft_output_dir, run_id=run_id, resume_run_id=resume_run_id,
    )
    # Public entry point only -- resolve_components() reads OPENAI_API_KEY
    # from the process environment itself; no credential is read, held, or
    # passed by this node.
    run_result = run_ft_screening(request)
    outcomes = map_run_result_to_chapter_outcomes(run_result)
    outcome_by_record_id = {o.record_id: o for o in outcomes}

    passing: List[Record] = []
    queue_entries: List[Dict[str, Any]] = []
    excluded_count = 0
    excluded_reasons: Dict[str, int] = {}

    for record, _pdf_path in confirmed:
        outcome = outcome_by_record_id[record.id]

        # Stage 3: an `ai` ledger baseline ONLY for a genuine completed
        # assessment -- execution_status != "completed" means the AI
        # never actually rendered a judgment for this record at all
        # (attempted and failed, or never got the chance to).
        ai_baseline_decision_id: Optional[str] = None
        if outcome.execution_status == "completed":
            producer_artifact_id = outcome.ft_screening_result_id
            assert producer_artifact_id is not None  # guaranteed by the "completed" invariant
            source_event_id = derive_source_event_id(
                producer_artifact_id=producer_artifact_id, record_id=record.id,
                stage=FULL_TEXT_ELIGIBILITY_STAGE,
            )
            if is_fulltext_exclusion_decision(outcome.decision):
                ai_reason = outcome.primary_exclusion_reason_code or "unknown"
            elif outcome.decision == "include":
                ai_reason = "FT_SCREENING assessed eligible"
            else:
                ai_reason = "FT_SCREENING needs_human_review"
            emitted = emit_selection_decision(
                store=ledger_store, record_id=record.id, stage=FULL_TEXT_ELIGIBILITY_STAGE,
                decision=outcome.decision, decision_maker="ai", source_event_id=source_event_id,
                reason=ai_reason, producer_artifact_id=producer_artifact_id,
            )
            ai_baseline_decision_id = emitted.decision_id

        if outcome.decision == "include":
            passing.append(record)
        elif is_fulltext_exclusion_decision(outcome.decision):
            excluded_count += 1
            reason_code = outcome.primary_exclusion_reason_code or "unknown"
            excluded_reasons[reason_code] = excluded_reasons.get(reason_code, 0) + 1
        else:
            # needs_human_review OR not_assessed (processing_failed /
            # input_unavailable / duplicate_input) -- NEVER included.
            # "record": the MUTATED record (full_text_path/full_text_status=
            # "parsed"/full_text_source already set by the acquisition loop
            # above) -- codex review, Phase B finding 2: resolving this entry
            # later must not have to reconstruct the record from
            # included_records (the PRE-fulltext snapshot), which would lose
            # exactly the confirmed PDF path that makes this entry
            # eligible-for-inclusion in the first place.
            queue_entries.append({
                "stage": "ft_screening",
                "record_id": outcome.record_id,
                "record": record.to_dict(),
                **outcome.model_dump(mode="json"),
                "reviewed_baseline_decision_id": ai_baseline_decision_id,
            })

    artifact_paths = {
        "run_result_json": str(run_result.artifacts.run_result_json),
        "item_results_jsonl": str(run_result.artifacts.item_results_jsonl),
        "ledger_jsonl": str(run_result.artifacts.ledger_jsonl),
    }
    return (
        passing, queue_entries, excluded_count, excluded_reasons,
        run_result.run_id, run_result.execution_fingerprint, artifact_paths,
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def acquire_and_confirm_records(
    records: List[Record], cfg: ReviewConfig, pdf_dir: Path, contact_email: Optional[str],
) -> Tuple[List[Tuple[Record, Path]], List[Record], Dict[str, int], List[Dict[str, Any]]]:
    """Track 3 continuation plan Unit C (codex point 5) -- the per-record
    acquisition/eligibility-gate loop `fulltext_node()` used to run inline,
    lifted out VERBATIM (no branch removed or added) so
    `nodes/fulltext_continuation.py::continuation_fulltext_node` can reuse
    it unmodified against a smaller input set, rather than duplicating this
    logic. `output_dir` is deliberately not a parameter: neither this loop
    body nor any helper it calls (`_find_local_pdf`, `_resolve_oa_url`,
    `_download_pdf`, `_extract_text`, `_looks_like_stub`) references it --
    it is only used just after this loop, for the manual worklist CSV's own
    path, outside this function's scope. Returns `(confirmed,
    manual_worklist, retrieval_counts, review_queue_entries)` -- the exact
    four values the loop below builds."""
    retrieval_counts: Dict[str, int] = {}
    manual_worklist: List[Record] = []
    review_queue: List[Dict[str, Any]] = []
    confirmed: List[Tuple[Record, Path]] = []

    for record in records:
        pdf_path = _find_local_pdf(record, pdf_dir)
        source = "local" if pdf_path else ""

        if pdf_path is None:
            pdf_url, source = _resolve_oa_url(record, contact_email)
            if pdf_url:
                dest = _dest_path(record, pdf_dir)
                if _download_pdf(pdf_url, dest):
                    pdf_path = dest

        full_text = _extract_text(pdf_path) if pdf_path else ""

        if full_text:
            stub_reason = _looks_like_stub(full_text, pdf_path)
            if stub_reason:
                record.full_text_status = "rejected_stub"
                pdf_path.unlink(missing_ok=True)
                pdf_path = None
            else:
                record.full_text_path = str(pdf_path)
                record.full_text_status = "parsed"
                record.full_text_source = source
        else:
            record.full_text_status = "no_doi" if not record.doi else "no_oa_fulltext_found"
            pdf_path = None

        retrieval_counts[record.full_text_status] = retrieval_counts.get(record.full_text_status, 0) + 1

        if record.full_text_status == "parsed" and pdf_path is not None:
            confirmed.append((record, pdf_path))
        elif cfg.full_text_acquisition_failure_policy == "exclude":
            # Terminal "not retrieved" -- no queue entry, no PRISMA exclusion
            # count. retrieval_counts[record.full_text_status] above already
            # captured this record.
            pass
        else:
            manual_worklist.append(record)
            review_queue.append({
                "stage": "acquisition",
                "record_id": record.id,
                "full_text_status": record.full_text_status,
                # No ai ledger baseline ever exists for an acquisition-stage
                # entry (FT_SCREENING never ran) -- present and None, not
                # omitted, so every queue entry uniformly carries this key.
                "reviewed_baseline_decision_id": None,
            })

    return confirmed, manual_worklist, retrieval_counts, review_queue


def fulltext_node(state: ReviewState) -> Dict[str, Any]:
    """
    For each included (title/abstract-passed) record:
      1. Look for a local PDF (incl. one placed there by a human via the
         Tier 3 worklist from a previous run).
      2. Otherwise run the Tier 1 automated OA cascade and download.
      3. Extract text and run it through the stub-detection gate.
      4. Write a Tier 3 manual worklist for every record still unconfirmed.
    Then, in one batch, send every CONFIRMED PDF (and only those) through a
    real FT_SCREENING full-text eligibility run and map its decisions
    conservatively back onto `full_text_records`/`full_text_review_queue`
    (see this module's own docstring for the exact mapping).
    """
    cfg = ReviewConfig.from_dict(state["config"])
    records = [Record.from_dict(r) for r in state["included_records"]]

    output_dir = cfg.ensure_output_dir()
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    review_run_id = resolve_review_run_id(state, allow_config=True)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)

    # Track 2 plan §2.7: SearchConfig.pubmed_email is deprecated and always
    # discarded now (never serialised, never read) -- the contact email
    # comes from the documented PUBMED_EMAIL environment variable instead,
    # the same source `search_backends/pubmed.py` reads.
    contact_email = os.environ.get("PUBMED_EMAIL")
    if not contact_email or contact_email.endswith("@example.com"):
        # Unpaywall rejects placeholder/example.com addresses outright (422
        # "use your own email in API calls") -- sending one anyway would
        # silently zero out every Unpaywall lookup, which looks identical to
        # "genuinely not open access". Skip that resolver rather than lie
        # about why nothing was found.
        logger.warning(
            "No real contact email configured (PUBMED_EMAIL) -- "
            "skipping Unpaywall lookups; set a real email to enable them."
        )
        contact_email = None

    confirmed, manual_worklist, retrieval_counts, review_queue = acquire_and_confirm_records(
        records=records, cfg=cfg, pdf_dir=pdf_dir, contact_email=contact_email)

    worklist_path = output_dir / "manual_download_worklist.csv"
    _write_manual_worklist(manual_worklist, worklist_path)

    passing: List[Record] = []
    excluded_count = 0
    excluded_reasons: Dict[str, int] = {}
    ft_run_id: Optional[str] = None
    ft_execution_fingerprint: Optional[str] = None
    ft_artifact_paths: Optional[Dict[str, str]] = None

    if confirmed:
        (
            passing, ft_queue_entries, ft_excluded_count, ft_excluded_reasons,
            ft_run_id, ft_execution_fingerprint, ft_artifact_paths,
        ) = _run_full_text_eligibility(confirmed, cfg, output_dir, ledger_store)
        review_queue.extend(ft_queue_entries)
        excluded_count += ft_excluded_count
        for reason, count in ft_excluded_reasons.items():
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + count

    logger.info(
        f"Full-text retrieval: {retrieval_counts} "
        f"({len(manual_worklist)} records need Tier 2/3 follow-up)"
    )
    logger.info(
        f"Full-text eligibility (FT_SCREENING): {len(passing)} included, {excluded_count} excluded, "
        f"{len(review_queue)} in review queue"
    )

    prisma = PRISMACounts(**state["prisma_counts"])
    prisma.assessed_full_text = len(confirmed)   # only documents actually assessed from a PDF
    prisma.excluded_full_text = excluded_count
    prisma.excluded_full_text_reasons = excluded_reasons
    prisma.fulltext_retrieval = retrieval_counts

    # Derived from the ledger's own current tails, never hardcoded from
    # whether any PDF was confirmed -- see _full_text_origin_and_oversight().
    origin, oversight = _full_text_origin_and_oversight(ledger_store)
    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    if oversight is not None:
        oversight_mode[FULL_TEXT_ELIGIBILITY_STAGE] = oversight
    if origin is not None:
        automation_origin[FULL_TEXT_ELIGIBILITY_STAGE] = origin

    return {
        "full_text_records": [r.to_dict() for r in passing],
        "full_text_review_queue": review_queue,
        # A non-empty queue means at least one record has NO final
        # include/exclude disposition -- graphs/routing.py's
        # make_post_fulltext_router() reads full_text_review_queue
        # directly (not this flag) to decide whether to stop before
        # downstream analysis, but requires_human_review is also set here
        # so it's visible/queryable the same way every other
        # human-in-the-loop signal in this state shape already is.
        "requires_human_review": bool(review_queue),
        "prisma_counts": prisma.model_dump(),
        "manual_worklist_path": str(worklist_path),
        "ft_screening_run_id": ft_run_id,
        "ft_screening_execution_fingerprint": ft_execution_fingerprint,
        "ft_screening_artifact_paths": ft_artifact_paths,
        "review_run_id": review_run_id,
        "selection_decision_ledger_dir": str(ledger_store.decisions_dir),
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
    }


# ---------------------------------------------------------------------------
# Human-review resolution -- the "resume path" for a non-empty
# full_text_review_queue. Not a LangGraph node: called between graph
# invocations (e.g. from a notebook or a CLI) once a human has looked at
# the queued entries, producing a state-update dict in the same "only the
# changed keys" shape every node returns -- apply it via
# `graph.update_state(config, resolve_full_text_review_queue(state, {...}))`
# then resume with `graph.invoke(None, config)` (see
# tests/nodes/test_fulltext_graph_routing.py for a full worked example).
# ---------------------------------------------------------------------------

def resolve_full_text_review_queue(
    state: ReviewState, resolutions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Applies human decisions to queued records by `record_id`, WITH
    stage-specific rules (codex review, Phase B round-2 finding 2: the
    previous version treated `stage="acquisition"` and
    `stage="ft_screening"` entries identically, which let a record whose
    PDF was NEVER retrieved -- no report ever assessed -- be resolved
    straight to "include", reintroducing exactly the no-PDF/abstract-
    fallback risk this whole integration exists to eliminate; it also
    reconstructed "included" records from `included_records` -- the
    PRE-fulltext snapshot -- discarding the confirmed `full_text_path`
    fulltext_node had already set on them).

    `resolutions` maps record_id -> a resolution dict:
      - "action": REQUIRED.
          * `stage="acquisition"` entries (no confirmed report -- no_doi/
            no_oa_fulltext_found/rejected_stub): ONLY `"not_retrieved"` is
            valid. Never "include"/"exclude" -- there is no report to be
            eligible or ineligible. Closing as `"not_retrieved"` removes
            it from the queue and makes NO PRISMA change (its acquisition
            outcome was already counted in
            `PRISMACounts.fulltext_retrieval` the moment it first failed
            to retrieve; this is a queue-state closure, not a new fact).
            To actually RETRY acquisition instead of closing it, do NOT
            call this function -- just place the PDF (Tier 3) and
            re-invoke `fulltext_node` with the SAME record still in
            `included_records`; fulltext_node re-attempts acquisition for
            every record it's given on every call, and a newly-confirmed
            PDF naturally supersedes the old queue entry (see
            `_stable_run_id()`'s own docstring on why a growing confirmed
            batch is safe).
          * `stage="ft_screening"` entries (a report WAS confirmed and
            assessed by FT_SCREENING -- needs_human_review, or a
            processing/provider failure): `"include"` or `"exclude"`.
            "include" restores the record FT_SCREENING actually assessed
            -- preserving `full_text_path`/`full_text_status`/
            `full_text_source` -- from the queue entry's own preserved
            `"record"` field (never re-derived from `included_records`).
            "exclude" requires a `"reason"` (a real, meaningful string --
            not a fixed generic code, since a human override isn't
            attributable to one specific FT_SCREENING criterion) and
            increments `PRISMACounts.excluded_full_text` under that exact
            reason string.
      - "reviewer", "note", "resolved_at": REQUIRED for every action --
        who decided, why, and when. Appended to the new
        `full_text_review_resolutions` audit-trail state field (never
        embedded into `Record` itself, which has no fields for this and
        is used far beyond the full-text stage).
      - "reason": REQUIRED only when action == "exclude".

    Every key in `resolutions` must currently be present in
    `state["full_text_review_queue"]` -- resolving a record that was
    never queued (typo, or already resolved) is a caller bug and raises
    `KeyError`. Records queued but NOT mentioned in `resolutions` stay
    queued -- resolution can be partial across multiple calls, and
    `requires_human_review`/the conditional routing gate stay True/
    blocking until the queue is fully empty.

    Stage 3 (the Stage 3 node-instrumentation design record): validated in two
    passes -- every entry's own fields, action legality, and (for
    `ft_screening`-stage entries) ledger-baseline staleness are all
    checked BEFORE any entry is applied, so one malformed/stale entry in
    a multi-record submission causes zero human-resolution writes, not a
    partial application (this is a claim about ORDINARY input-validation
    failures; it is not a global-transaction guarantee against a
    concurrent writer changing a tail mid-application -- the per-record
    `expected_current_tail_decision_id` precondition inside
    `emit_selection_decision()` is what makes that specific race safe,
    identifying exactly which record's own emission failed). A human
    `include`/`exclude` resolution supersedes the `ai` baseline the
    corresponding queue entry preserved at creation time
    (`reviewed_baseline_decision_id`, read here, never recomputed from
    the live tail); `not_retrieved` (acquisition-stage) never touches the
    ledger at all -- no assessment ever happened for that record.
    """
    queue = list(state["full_text_review_queue"])
    queue_by_id = {entry["record_id"]: entry for entry in queue}
    unknown = set(resolutions) - set(queue_by_id)
    if unknown:
        raise KeyError(f"record_id(s) not in full_text_review_queue: {sorted(unknown)}")

    cfg = ReviewConfig.from_dict(state["config"])
    output_dir = cfg.ensure_output_dir()
    review_run_id = resolve_review_run_id(state, allow_config=False)
    ledger_store = SelectionDecisionStore(output_dir=output_dir, review_run_id=review_run_id)

    # --- Pass 1: validate every entry's own fields/action legality, and
    # (ft_screening-stage only) preflight-check ledger-baseline staleness
    # -- zero side effects yet.
    validated: List[Dict[str, Any]] = []
    for record_id, resolution in resolutions.items():
        entry = queue_by_id[record_id]
        action = resolution.get("action")
        reviewer = resolution.get("reviewer")
        note = resolution.get("note")
        resolved_at = resolution.get("resolved_at")
        if not reviewer or not note or not resolved_at:
            raise ValueError(f"resolution for {record_id!r} must set reviewer, note, and resolved_at")

        stage = entry.get("stage")
        reason: Optional[str] = None

        if stage == "acquisition":
            if action != "not_retrieved":
                raise ValueError(
                    f"resolution for {record_id!r}: an acquisition-stage entry (no confirmed report) "
                    f"can only be resolved as 'not_retrieved', got {action!r} -- eligibility include/"
                    "exclude requires a confirmed, FT_SCREENING-assessed report"
                )
        elif stage == "ft_screening":
            if action == "include":
                if entry.get("record") is None:
                    raise KeyError(
                        f"ft_screening-stage queue entry for {record_id!r} is missing its preserved "
                        "'record' -- cannot include without the confirmed full_text_path"
                    )
            elif action == "exclude":
                reason = resolution.get("reason")
                if not reason:
                    raise ValueError(f"resolution for {record_id!r}: 'exclude' requires a meaningful 'reason'")
            else:
                raise ValueError(
                    f"resolution for {record_id!r}: an ft_screening-stage entry must be resolved "
                    f"'include' or 'exclude', got {action!r}"
                )
        else:
            raise ValueError(f"queue entry for {record_id!r} has unknown stage {stage!r}")

        item: Dict[str, Any] = {
            "record_id": record_id, "entry": entry, "stage": stage, "action": action,
            "reviewer": reviewer, "note": note, "resolved_at": resolved_at, "reason": reason,
        }

        if stage == "ft_screening":
            if "reviewed_baseline_decision_id" not in entry:
                # Codex review: .get() silently accepted a MISSING key as
                # an intentional None baseline, indistinguishable from a
                # genuinely-checked "no prior ai decision" case. Every
                # queue entry this codebase's own fulltext_node produces
                # carries this key explicitly (None is a valid value for
                # acquisition-stage/not-yet-assessed entries) -- a key
                # that's simply ABSENT means a malformed or externally-
                # constructed entry, which must raise, not be silently
                # treated as an empty-chain baseline.
                raise KeyError(
                    f"ft_screening-stage queue entry for {record_id!r} is missing "
                    "'reviewed_baseline_decision_id' -- every queue entry must explicitly carry this "
                    "key (None is a valid value when no ai baseline exists, but the key itself must "
                    "be present)"
                )
            reviewed_baseline_decision_id = entry["reviewed_baseline_decision_id"]
            resolution_id = compute_resolution_id(
                stage=FULL_TEXT_ELIGIBILITY_STAGE, record_id=record_id, action=action, reviewer=reviewer,
                note=note, resolved_at=resolved_at, reviewed_baseline_decision_id=reviewed_baseline_decision_id,
                exclusion_reason=reason,
            )
            source_event_id = derive_source_event_id(
                producer_artifact_id=resolution_id, record_id=record_id, stage=FULL_TEXT_ELIGIBILITY_STAGE,
            )
            # Fast, unlocked preflight -- fails an obviously-stale entry
            # early. NOT the source of truth: emit_selection_decision()'s
            # own expected_current_tail_decision_id (applied below, Pass
            # 2) is the atomic, authoritative guarantee (INV-03).
            preflight_check_baseline_still_current(
                store=ledger_store, record_id=record_id, stage=FULL_TEXT_ELIGIBILITY_STAGE,
                reviewed_baseline_decision_id=reviewed_baseline_decision_id,
                resolution_source_event_id=source_event_id,
            )
            item["reviewed_baseline_decision_id"] = reviewed_baseline_decision_id
            item["resolution_id"] = resolution_id
            item["source_event_id"] = source_event_id

        validated.append(item)

    # --- Pass 2: every entry passed validation (+ preflight for
    # ft_screening-stage ones) -- apply.
    passing = [Record.from_dict(r) for r in state["full_text_records"]]
    prisma = PRISMACounts(**state["prisma_counts"])
    resolution_log = list(state.get("full_text_review_resolutions") or [])
    resolved_ids = {v["record_id"] for v in validated}
    remaining_queue = [entry for entry in queue if entry["record_id"] not in resolved_ids]

    for v in validated:
        record_id, entry, stage, action = v["record_id"], v["entry"], v["stage"], v["action"]
        log_entry: Dict[str, Any] = {
            "record_id": record_id, "stage": stage, "action": action,
            "reviewer": v["reviewer"], "note": v["note"], "resolved_at": v["resolved_at"],
        }

        if stage == "acquisition":
            resolution_log.append(log_entry)   # No PRISMA change, no ledger entry, no resolution_id -- no assessment ever happened (see docstring).
        else:  # stage == "ft_screening"
            emit_selection_decision(
                store=ledger_store, record_id=record_id, stage=FULL_TEXT_ELIGIBILITY_STAGE, decision=action,
                decision_maker="human", source_event_id=v["source_event_id"], reason=v["note"],
                producer_artifact_id=v["resolution_id"],
                expected_current_tail_decision_id=v["reviewed_baseline_decision_id"],
            )
            if action == "include":
                passing.append(Record.from_dict(entry["record"]))
            else:
                reason = v["reason"]
                prisma.excluded_full_text += 1
                prisma.excluded_full_text_reasons[reason] = prisma.excluded_full_text_reasons.get(reason, 0) + 1
                log_entry["reason"] = reason
            # Durable link back to the human ledger producer (codex
            # review: these were previously omitted, breaking the audit
            # trail's own connection to the emitted decision), and
            # deduplicated by resolution_id -- an identical resubmission
            # is idempotent at the ledger level and must not also grow
            # this audit log with a duplicate entry.
            log_entry["resolution_id"] = v["resolution_id"]
            log_entry["reviewed_baseline_decision_id"] = v["reviewed_baseline_decision_id"]
            resolution_log = append_resolution_log_entry_deduplicated(
                resolution_log, log_entry, resolution_id=v["resolution_id"],
            )

    # Derived from the ledger's own current tails, never hardcoded --
    # see _full_text_origin_and_oversight().
    origin, oversight = _full_text_origin_and_oversight(ledger_store)
    oversight_mode = dict(state.get("oversight_mode") or {})
    automation_origin = dict(state.get("automation_origin") or {})
    if oversight is not None:
        oversight_mode[FULL_TEXT_ELIGIBILITY_STAGE] = oversight
    if origin is not None:
        automation_origin[FULL_TEXT_ELIGIBILITY_STAGE] = origin

    return {
        "full_text_records": [r.to_dict() for r in passing],
        "full_text_review_queue": remaining_queue,
        "full_text_review_resolutions": resolution_log,
        "requires_human_review": bool(remaining_queue),
        "prisma_counts": prisma.model_dump(),
        "review_run_id": review_run_id,
        "selection_decision_ledger_dir": str(ledger_store.decisions_dir),
        "oversight_mode": oversight_mode,
        "automation_origin": automation_origin,
    }
