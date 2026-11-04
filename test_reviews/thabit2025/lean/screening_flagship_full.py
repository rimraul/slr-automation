"""
Full-118 flagship (gpt-5.6-terra) pass on the current R1 prompt (checklist
structuring + extended richness: vocab-masking guidance + invented
contrastive examples, no known-label leakage). Fills the gap left by the
cascade run, which only covered the 60 records mini escalated -- needed so
weak-vs-strong-vs-true disagreement open-coding for R2 has full coverage,
not just mini's uncertain subset.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from screening_test import build_known_included, build_known_excluded, run_group, load_env, HERE

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

    inc = run_group(client, f"known_included ({FLAGSHIP_MODEL}, R1 full)", known_included)
    exc = run_group(client, f"known_excluded ({FLAGSHIP_MODEL}, R1 full)", known_excluded)

    (HERE / "screening_test_results_flagship_r1_full.json").write_text(
        json.dumps({"known_included": inc, "known_excluded": exc}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("-> screening_test_results_flagship_r1_full.json")


if __name__ == "__main__":
    main()
