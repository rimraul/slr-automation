"""
Chapter demonstration: OpenAI async Batch API for data extraction -- ported
from iasia_test/layer2_fulltext.py's Phase 2/3 (submit/collect), the one
genuine gap found there: nothing else in this codebase uses the Batch API
(50% cheaper than sync chat.completions, up to 24h turnaround). Everywhere
else (slr_pipeline/nodes/screening.py, extraction/batching.py) "batch"
means local document/record chunking for synchronous calls -- a different
concept. This script demonstrates the async-job pattern as a teaching
example, not a production addition to slr_pipeline.

Three phases, each resumable/idempotent:
  build   -- read fulltext_pilot/tier1_manifest.json + parsed text, build
             batch_input.jsonl for records not already cached
  submit  -- upload batch_input.jsonl, create the batch job, save state
             (guards against re-submitting a pending batch)
  collect -- poll until done, cache results (diskcache, keyed by doi/title
             so re-runs cost $0), write batch_extracted.jsonl

Input:  fulltext_pilot/tier1_manifest.json + fulltext_pilot/text/*.txt
        (both produced by fulltext_pilot.py; text now page-anchored via
        DOWNLOADING.tier1_auto.pdf_to_text's [PAGE N] markers)
Output: fulltext_pilot/batch_input.jsonl
        fulltext_pilot/batch_state.json
        fulltext_pilot/batch_cache/        (diskcache dir)
        fulltext_pilot/batch_extracted.jsonl

Usage:
  PYEXE="python"
  $PYEXE batch_extract_demo.py --phase build
  $PYEXE batch_extract_demo.py --phase submit   # live paid call, submits the job
  $PYEXE batch_extract_demo.py --phase collect  # polls until done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"c:/projects")
from DOWNLOADING.tier1_auto import safe_id
from SEARCHING.keys import get_key

import diskcache
from openai import OpenAI

from screening_test import HERE

MODEL = "gpt-5.4-mini"
MAX_CHARS = 60_000

PILOT_DIR    = HERE / "fulltext_pilot"
MANIFEST_PATH = PILOT_DIR / "tier1_manifest.json"
TEXT_DIR      = PILOT_DIR / "text"

INPUT_PATH  = PILOT_DIR / "batch_input.jsonl"
STATE_PATH  = PILOT_DIR / "batch_state.json"
CACHE_DIR   = PILOT_DIR / "batch_cache"
OUT_PATH    = PILOT_DIR / "batch_extracted.jsonl"

client = OpenAI(api_key=get_key("openai"))

EXTRACT_SYSTEM = (
    "You are a systematic review methodologist. Extract structured "
    "information from the provided public-value-governance review paper "
    "text. Return ONLY valid JSON. No markdown, no explanation."
)

EXTRACT_PROMPT = """\
Title: {title}
DOI: {doi}

Full text (page-anchored, may be truncated at {max_chars} chars):
{text}

---

Extract this JSON schema:

{{
  "study_design": "<systematic_review|conceptual|case_study|survey|mixed|other>",
  "sample_size": <integer or null>,
  "country": "<country or region, or null if not specified>",
  "public_value_dimension": "<one short phrase, e.g. accountability, co-creation, legitimacy>",
  "main_finding": "<one sentence, the paper's core substantive claim>",
  "evidence_quote": "<direct quote <=40 words supporting main_finding>",
  "evidence_page": <page integer this quote came from, or null>
}}

Rules:
- evidence_quote must be an exact substring from the text above
- evidence_page: read the nearest preceding [PAGE N] marker in the text
"""


def _cache() -> diskcache.Cache:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(CACHE_DIR))


def _cache_key(rec: dict) -> str:
    return f"batchdemo:{MODEL}:{rec.get('doi') or rec['title'][:60]}"


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def phase_build() -> None:
    manifest = _load_manifest()
    parsed = [m for m in manifest if m.get("status") == "parsed"]
    print(f"Manifest: {len(manifest)} records, {len(parsed)} parsed (have text)")

    cache = _cache()
    done_keys: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_keys.add(json.loads(line)["rec_id"])

    requests: list[dict] = []
    for rec in parsed:
        if rec["rec_id"] in done_keys:
            continue
        if cache.get(_cache_key(rec)) is not None:
            continue
        text_path = TEXT_DIR / f"{safe_id(rec['rec_id'])}.txt"
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")[:MAX_CHARS]
        prompt = EXTRACT_PROMPT.format(
            title=rec["title"], doi=rec.get("doi") or "", text=text, max_chars=MAX_CHARS,
        )
        requests.append({
            "custom_id": rec["rec_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "temperature": 0,
                "max_completion_tokens": 800,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
        })

    if not requests:
        print("Nothing to build -- everything already cached or done.")
        return

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"-> {INPUT_PATH} ({len(requests)} requests)")


def phase_submit() -> None:
    if not INPUT_PATH.exists():
        print("No batch_input.jsonl -- run --phase build first.")
        return
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("collected_at") is None:
            print(f"Batch {state['batch_id']} already submitted, not yet collected. "
                  f"Run --phase collect, or delete {STATE_PATH} to resubmit.")
            return

    n_requests = sum(1 for _ in INPUT_PATH.open(encoding="utf-8"))
    print(f"Submitting live paid batch job: {n_requests} requests, model={MODEL}")

    with open(INPUT_PATH, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted: {batch.id}  status={batch.status}")

    state = {
        "batch_id": batch.id,
        "file_id": file_obj.id,
        "n_requests": n_requests,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collected_at": None,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def phase_collect(poll_interval: int = 30, max_wait_minutes: int = 120) -> None:
    if not STATE_PATH.exists():
        print("No batch_state.json -- run --phase submit first.")
        return
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("collected_at"):
        print(f"Already collected at {state['collected_at']}.")
        return

    batch_id = state["batch_id"]
    deadline = time.time() + max_wait_minutes * 60
    poll = poll_interval
    while True:
        batch = client.batches.retrieve(batch_id)
        cnt = batch.request_counts
        print(f"  status={batch.status:<12} completed={cnt.completed} failed={cnt.failed} total={cnt.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        if time.time() > deadline:
            print(f"Max wait ({max_wait_minutes}min) reached -- run --phase collect again later.")
            return
        time.sleep(poll)
        poll = min(poll * 1.5, 300)

    if batch.status != "completed":
        print(f"Batch ended with status: {batch.status}")
        return

    result_bytes = client.files.content(batch.output_file_id).content
    cache = _cache()
    manifest_by_id = {m["rec_id"]: m for m in _load_manifest()}

    results: dict[str, dict] = {}
    for line in result_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj["custom_id"]
        if obj.get("error"):
            print(f"  error for {cid}: {obj['error']}")
            continue
        content = obj["response"]["body"]["choices"][0]["message"]["content"]
        results[cid] = json.loads(content)

    print(f"Parsed {len(results)} batch results")

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for rec_id, extracted in results.items():
            rec = manifest_by_id.get(rec_id, {})
            cache.set(_cache_key(rec), extracted)
            row = {"rec_id": rec_id, "title": rec.get("title"), "doi": rec.get("doi"), **extracted}
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    state["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"-> {OUT_PATH} ({len(results)} rows appended)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch API extraction demo")
    ap.add_argument("--phase", choices=["build", "submit", "collect"], required=True)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--max-wait", type=int, default=120)
    args = ap.parse_args()

    if args.phase == "build":
        phase_build()
    elif args.phase == "submit":
        phase_submit()
    elif args.phase == "collect":
        phase_collect(poll_interval=args.poll_interval, max_wait_minutes=args.max_wait)


if __name__ == "__main__":
    main()
