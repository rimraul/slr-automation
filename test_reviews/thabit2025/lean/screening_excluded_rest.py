"""
Screens the 261 known_excluded-proxy records NOT used in the R1/R2 runs
(journal_yes pool minus Appendix B minus the 59 already sampled), across
all three richness levels (R0 naive, R1 extended, R2 tuned) for both weak
(mini) and strong (flagship) models. Gives a held-out check on the
aggregate lenient-rate/specificity axis for R2 specifically (R2's rules
were mined only from the already-used 59, never touch this remainder), and
fills the missing R0 richness level across the board -- see NOTES.md "R2
ceiling analysis" for why known_excluded is a proxy label, not confirmed
per-record ground truth (so this is a base-rate/consistency check, not a
per-record accuracy measurement).
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
    used59 = build_known_excluded(journal_yes, matched_ids, n=len(known_included))
    used_titles = {r["title"] for r in used59}

    pool = [
        r for r in journal_yes
        if r["id"] not in matched_ids and (r.get("abstract") or "").strip()
    ]
    rest = [r for r in pool if r["title"] not in used_titles]
    print(f"remaining known_excluded pool: n={len(rest)}")

    env = load_env("../../../.env")
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    client = OpenAI()

    import screening_test
    results = {}
    for model, tag in ((WEAK_MODEL, "weak"), (STRONG_MODEL, "strong")):
        screening_test.MODEL = model
        for richness in ("R0", "R1", "R2"):
            screening_test.RICHNESS = richness
            key = f"{tag}_{richness}"
            rows = run_group(client, f"known_excluded_rest {key}", rest)
            results[key] = rows
            (HERE / f"screening_test_results_excluded_rest_{key}.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"-> screening_test_results_excluded_rest_{key}.json")

    (HERE / "screening_test_results_excluded_rest_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("-> screening_test_results_excluded_rest_all.json")


if __name__ == "__main__":
    main()
