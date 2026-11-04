"""
Lean prototype: classify unique journal names as belonging to the public
administration field (Thabit et al. 2025's 4 Scimago categories: Sociology
and Political Science; Public Administration; Political Science; Political
Science and International Relations), one LLM call per UNIQUE journal name
(not per record) to keep cost down -- proxy for the paper's own WoS/Scopus/
Scimago category lookup (no recoverable 119-journal allowlist exists).

Reuses the shared OpenAI credential resolver + model default from
slr_pipeline/config.py, but is a standalone lean script, not a pipeline node.

Live paid calls -- run only with explicit per-invocation authorization.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, r"C:\projects\SLR_chapter")

from openai import OpenAI
from slr_pipeline.config import resolve_openai_api_key

HERE = Path(__file__).parent
MODEL = "gpt-5.4-mini"

PROMPT = """You are screening academic journals for a systematic literature review in public administration.

Journal name: "{journal}"

Question: Does this journal belong to any of these fields (per Scimago/Scopus subject categories)?
- Sociology and Political Science
- Public Administration
- Political Science
- Political Science and International Relations

Answer with a JSON object: {{"answer": "yes"}} or {{"answer": "no"}}. If uncertain, use your best judgement based on the journal's typical scope."""


def load_env(path: str) -> dict:
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def classify_journal(client: OpenAI, journal: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(journal=journal)}],
        temperature=0.0,
    )
    data = json.loads(resp.choices[0].message.content)
    answer = str(data.get("answer", "")).strip().lower()
    return "yes" if answer == "yes" else "no"


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        env = load_env(str(HERE / "../../../.env"))
        if env.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]

    records = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))

    journals_original_case: Dict[str, str] = {}
    for r in records:
        j = (r.get("journal") or "").strip()
        if j:
            journals_original_case.setdefault(j.lower(), j)

    unique_journals = sorted(journals_original_case.values())
    print(f"{len(records)} records, {len(unique_journals)} unique journals to classify")

    cache_path = HERE / "journal_classification.json"
    cache: Dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"resuming: {len(cache)} already classified")

    client = OpenAI(api_key=resolve_openai_api_key())

    for i, journal in enumerate(unique_journals):
        if journal.lower() in cache:
            continue
        answer = classify_journal(client, journal)
        cache[journal.lower()] = answer
        if (i + 1) % 25 == 0:
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {i + 1}/{len(unique_journals)} classified (checkpoint saved)")

    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    n_yes = sum(1 for v in cache.values() if v == "yes")
    n_no = len(cache) - n_yes
    print(f"\nclassified: {len(cache)} journals -> yes={n_yes} no={n_no}")

    kept = [r for r in records if cache.get((r.get("journal") or "").strip().lower()) == "yes"]
    dropped = len(records) - len(kept)
    print(f"records kept={len(kept)} dropped={dropped}")

    (HERE / "journal_filtered_records.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"-> journal_filtered_records.json")


if __name__ == "__main__":
    main()
