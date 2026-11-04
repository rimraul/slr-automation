"""
Recall-first cascade: mini (checklist prompt) screens all 118 known
records; only mini's "exclude" and "borderline" decisions escalate to
flagship (gpt-5.6-terra) for a second look, same checklist prompt. Mini's
"include" decisions stand unescalated. Final cascade decision = mini's for
unescalated records, flagship's for escalated ones.

Bounds flagship cost to the escalated subset instead of a full second pass
over all 118 records.
"""
from __future__ import annotations

import json
import os

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

    mini = json.loads((HERE / "screening_test_results_checklist.json").read_text(encoding="utf-8"))

    def records_by_title(records):
        return {r.get("title"): r for r in records}

    pools = {"known_included": records_by_title(known_included), "known_excluded": records_by_title(known_excluded)}

    escalated = {"known_included": [], "known_excluded": []}
    for group in ("known_included", "known_excluded"):
        pool = pools[group]
        for m in mini[group]:
            if m["decision"] in ("exclude", "borderline"):
                rec = pool.get(m["title"])
                if rec is not None:
                    escalated[group].append(rec)

    print(f"escalating known_included: {len(escalated['known_included'])}/{len(mini['known_included'])}")
    print(f"escalating known_excluded: {len(escalated['known_excluded'])}/{len(mini['known_excluded'])}")

    env = load_env("../../../.env")
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    client = OpenAI()

    import screening_test
    screening_test.MODEL = FLAGSHIP_MODEL

    flagship_results = {}
    for group in ("known_included", "known_excluded"):
        if escalated[group]:
            flagship_results[group] = run_group(client, f"escalated {group} ({FLAGSHIP_MODEL})", escalated[group])
        else:
            flagship_results[group] = []

    (HERE / "screening_test_results_cascade_flagship.json").write_text(
        json.dumps(flagship_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("-> screening_test_results_cascade_flagship.json")

    # --- Merge: mini's include stands, escalated records get flagship's decision ---
    final = {"known_included": [], "known_excluded": []}
    for group in ("known_included", "known_excluded"):
        flagship_by_title = {r["title"]: r for r in flagship_results[group]}
        for m in mini[group]:
            if m["decision"] == "include":
                final[group].append(m)
            else:
                f = flagship_by_title.get(m["title"])
                final[group].append(f if f is not None else m)

    (HERE / "screening_test_results_cascade_final.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("-> screening_test_results_cascade_final.json")

    def counts(rows):
        c = {"include": 0, "borderline": 0, "exclude": 0}
        for r in rows:
            c[r["decision"]] += 1
        return c

    ci, ce = counts(final["known_included"]), counts(final["known_excluded"])
    n_inc, n_exc = len(final["known_included"]), len(final["known_excluded"])
    print(f"\nFINAL CASCADE known_included: {ci}  strict_recall={ci['include']}/{n_inc}  retained={ci['include']+ci['borderline']}/{n_inc}")
    print(f"FINAL CASCADE known_excluded: {ce}  strict_specificity={ce['exclude']}/{n_exc}  rejected={ce['exclude']+ce['borderline']}/{n_exc}")


if __name__ == "__main__":
    main()
