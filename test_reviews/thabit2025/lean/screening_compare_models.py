"""
il_outcomes-style disagreement-rationale check (screen_abstracts_reproduction.py
v2, 2026-09-19): run the SAME known_included/known_excluded pools through a
stronger model (gpt-5.6-terra, this project's calibrated flagship/adjudicator
model elsewhere) alongside the mini model already run in screening_test.py,
then open-code where they disagree to find a GENERALIZABLE error pattern --
never tune the prompt to flip a specific known title.

Reuses screening_test.py's pool-construction + batch-screening machinery
verbatim (same prompt, same criteria) so the only variable is the model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from screening_test import (
    build_known_included, build_known_excluded, run_group, load_env, HERE,
)

FLAGSHIP_MODEL = "gpt-5.6-terra"


def main() -> None:
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    journal_yes = json.loads((HERE / "journal_filtered_records.json").read_text(encoding="utf-8"))

    known_included, matched_ids = build_known_included(dedup)
    known_excluded = build_known_excluded(journal_yes, matched_ids, n=len(known_included))

    env = load_env("../../../.env")
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    client = OpenAI()

    import screening_test
    screening_test.MODEL = FLAGSHIP_MODEL

    inc_flagship = run_group(client, f"known_included ({FLAGSHIP_MODEL})", known_included)
    exc_flagship = run_group(client, f"known_excluded ({FLAGSHIP_MODEL})", known_excluded)

    (HERE / "screening_test_results_flagship.json").write_text(
        json.dumps({"known_included": inc_flagship, "known_excluded": exc_flagship}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("-> screening_test_results_flagship.json")

    # --- Disagreement diff against the mini-model results already on disk ---
    mini = json.loads((HERE / "screening_test_results.json").read_text(encoding="utf-8"))

    def index_by_title(rows):
        return {r["title"]: r for r in rows}

    disagreements = []
    for group in ("known_included", "known_excluded"):
        mini_by_title = index_by_title(mini[group])
        flag_by_title = index_by_title(
            (inc_flagship if group == "known_included" else exc_flagship)
        )
        for title, m in mini_by_title.items():
            f = flag_by_title.get(title)
            if f is None:
                continue
            if m["decision"] != f["decision"]:
                disagreements.append({
                    "group": group, "title": title,
                    "mini_decision": m["decision"], "mini_reason": m["reason"],
                    "flagship_decision": f["decision"], "flagship_reason": f["reason"],
                })

    (HERE / "screening_disagreements.json").write_text(
        json.dumps(disagreements, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{len(disagreements)} disagreements out of {len(mini['known_included']) + len(mini['known_excluded'])} total")
    print("-> screening_disagreements.json")


if __name__ == "__main__":
    main()
