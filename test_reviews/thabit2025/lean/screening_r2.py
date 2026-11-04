"""
R2 (tuned richness) ceiling test: EXTRA_RULES_R2 in screening_test.py was
mined by open-coding real weak (mini)-vs-strong (flagship)-vs-true-label
disagreements on the full known_included/known_excluded pool -- see
NOTES.md "R2 ceiling analysis" for why this is a deliberate oracle/ceiling
test (full known-label access, no held-out split), not a held-out
generalization estimate. Runs both weak and strong models on the SAME
118-record pool used to derive the rules, to test whether the original
authors' decisions are recoverable at all given maximal prompt tuning.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from screening_test import build_known_included, build_known_excluded, run_group, load_env, HERE

WEAK_MODEL = "gpt-5.4-mini"
STRONG_MODEL = "gpt-5.6-terra"


def main() -> None:
    dedup = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    journal_yes = json.loads((HERE / "journal_filtered_records.json").read_text(encoding="utf-8"))

    known_included, matched_ids = build_known_included(dedup)
    known_excluded = build_known_excluded(journal_yes, matched_ids, n=len(known_included))

    env = load_env("../../../.env")
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    client = OpenAI()

    import screening_test
    screening_test.RICHNESS = "R2"

    for model, tag in ((WEAK_MODEL, "weak"), (STRONG_MODEL, "strong")):
        screening_test.MODEL = model
        inc = run_group(client, f"known_included ({model}, R2)", known_included)
        exc = run_group(client, f"known_excluded ({model}, R2)", known_excluded)
        (HERE / f"screening_test_results_r2_{tag}.json").write_text(
            json.dumps({"known_included": inc, "known_excluded": exc}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"-> screening_test_results_r2_{tag}.json")


if __name__ == "__main__":
    main()
