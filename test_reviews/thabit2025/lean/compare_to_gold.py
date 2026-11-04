"""
Compare `run_deductive_extraction.py`'s LLM-coded concepts against Table
C1's own gold reference list (`ft_codebook.gold_refs()`) -- per publication
and in aggregate. Answers: does a single LLM coder, applying Thabit et al.'s
own 37-concept codebook deductively, recover the same codes the paper's
3-human-coder dual-coding process assigned to the same article?

Usage:
  PYEXE="python"
  $PYEXE compare_to_gold.py            # latest version (v2)
  $PYEXE compare_to_gold.py v1         # a specific version
  $PYEXE compare_to_gold.py --diff     # v1 vs v2 delta view
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ft_codebook import CODEBOOK, REC_ID_TO_REF, gold_refs

HERE = Path(__file__).parent
PILOT_DIR = HERE / "fulltext_pilot"

CONCEPTS = [c for _, _, c in CODEBOOK]


def extracted_path(version: str) -> Path:
    return PILOT_DIR / f"deductive_extracted_{version}.jsonl"


def load_extracted(version: str) -> dict[str, dict]:
    path = extracted_path(version)
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["rec_id"]] = row
    return rows


def score(extracted: dict[str, dict]) -> tuple[dict, list[tuple]]:
    gold = gold_refs()
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_pub_rows = []

    for rec_id, ref in REC_ID_TO_REF.items():
        if rec_id not in extracted:
            print(f"MISSING extraction for {rec_id}")
            continue

        found_by_llm = {
            code["concept"] for code in extracted[rec_id]["codes"] if code.get("found")
        }
        gold_for_pub = {concept for concept in CONCEPTS if ref in gold[concept]}

        tp = found_by_llm & gold_for_pub
        fp = found_by_llm - gold_for_pub
        fn = gold_for_pub - found_by_llm
        tn = set(CONCEPTS) - found_by_llm - gold_for_pub

        totals["tp"] += len(tp)
        totals["fp"] += len(fp)
        totals["fn"] += len(fn)
        totals["tn"] += len(tn)

        precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else float("nan")
        recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

        title = extracted[rec_id].get("title", "")
        per_pub_rows.append((rec_id, ref, title, len(gold_for_pub), len(found_by_llm), len(tp), len(fp), len(fn), precision, recall, f1))

    return totals, per_pub_rows


def micro_avg(totals: dict) -> tuple[float, float, float]:
    p = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else float("nan")
    r = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else float("nan")
    f1 = (2 * p * r / (p + r)) if (p + r) else float("nan")
    return p, r, f1


def print_report(version: str, extracted: dict[str, dict]) -> None:
    totals, per_pub_rows = score(extracted)

    print(f"\n=== {version} ===")
    print(f"{'rec_id':30s} {'ref':5s} {'gold':>5s} {'llm':>5s} {'tp':>4s} {'fp':>4s} {'fn':>4s} {'prec':>6s} {'rec':>6s} {'f1':>6s}  title")
    for rec_id, ref, title, n_gold, n_llm, tp, fp, fn, p, r, f1 in per_pub_rows:
        print(f"{rec_id:30s} {ref:5s} {n_gold:5d} {n_llm:5d} {tp:4d} {fp:4d} {fn:4d} {p:6.2f} {r:6.2f} {f1:6.2f}  {title[:50]}")

    p_all, r_all, f1_all = micro_avg(totals)
    print(f"\nMicro-averaged over {len(per_pub_rows)} publications x 37 concepts "
          f"({len(per_pub_rows) * 37} decisions):")
    print(f"  TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} TN={totals['tn']}")
    print(f"  precision={p_all:.3f} recall={r_all:.3f} f1={f1_all:.3f}")

    print("\nPer-concept confusion (concepts the LLM most often adds or misses across all 10 pubs):")
    concept_fp = {c: 0 for c in CONCEPTS}
    concept_fn = {c: 0 for c in CONCEPTS}
    gold = gold_refs()
    for rec_id, ref in REC_ID_TO_REF.items():
        if rec_id not in extracted:
            continue
        found_by_llm = {code["concept"] for code in extracted[rec_id]["codes"] if code.get("found")}
        gold_for_pub = {concept for concept in CONCEPTS if ref in gold[concept]}
        for c in found_by_llm - gold_for_pub:
            concept_fp[c] += 1
        for c in gold_for_pub - found_by_llm:
            concept_fn[c] += 1

    print(f"  {'concept':45s} {'FP (LLM over-codes)':>20s} {'FN (LLM misses)':>18s}")
    for c in CONCEPTS:
        if concept_fp[c] or concept_fn[c]:
            print(f"  {c:45s} {concept_fp[c]:20d} {concept_fn[c]:18d}")


def print_diff(v_old: str, v_new: str) -> None:
    ext_old = load_extracted(v_old)
    ext_new = load_extracted(v_new)
    totals_old, rows_old = score(ext_old)
    totals_new, rows_new = score(ext_new)
    p1, r1, f1_1 = micro_avg(totals_old)
    p2, r2, f2 = micro_avg(totals_new)

    print(f"\n=== {v_old} -> {v_new} delta ===")
    print(f"{'':10s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s} {'prec':>7s} {'rec':>7s} {'f1':>7s}")
    print(f"{v_old:10s} {totals_old['tp']:5d} {totals_old['fp']:5d} {totals_old['fn']:5d} {totals_old['tn']:5d} {p1:7.3f} {r1:7.3f} {f1_1:7.3f}")
    print(f"{v_new:10s} {totals_new['tp']:5d} {totals_new['fp']:5d} {totals_new['fn']:5d} {totals_new['tn']:5d} {p2:7.3f} {r2:7.3f} {f2:7.3f}")
    print(f"{'delta':10s} {totals_new['tp']-totals_old['tp']:+5d} {totals_new['fp']-totals_old['fp']:+5d} "
          f"{totals_new['fn']-totals_old['fn']:+5d} {totals_new['tn']-totals_old['tn']:+5d} "
          f"{p2-p1:+7.3f} {r2-r1:+7.3f} {f2-f1_1:+7.3f}")

    print(f"\nPer-publication precision/recall, {v_old} -> {v_new}:")
    old_by_rec = {r[0]: r for r in rows_old}
    new_by_rec = {r[0]: r for r in rows_new}
    print(f"{'rec_id':30s} {'ref':5s} {'gold':>5s} "
          f"{v_old+' llm':>10s} {v_new+' llm':>10s} "
          f"{v_old+' P':>8s} {v_new+' P':>8s} {v_old+' R':>8s} {v_new+' R':>8s}")
    for rec_id in REC_ID_TO_REF:
        if rec_id not in old_by_rec or rec_id not in new_by_rec:
            continue
        _, ref, _, n_gold, n_llm_o, _, _, _, p_o, r_o, _ = old_by_rec[rec_id]
        _, _, _, _, n_llm_n, _, _, _, p_n, r_n, _ = new_by_rec[rec_id]
        print(f"{rec_id:30s} {ref:5s} {n_gold:5d} {n_llm_o:10d} {n_llm_n:10d} "
              f"{p_o:8.2f} {p_n:8.2f} {r_o:8.2f} {r_n:8.2f}")


def main() -> None:
    args = sys.argv[1:]
    if "--diff" in args:
        rest = [a for a in args if a != "--diff"]
        v_old = rest[0] if len(rest) > 0 else "v1"
        v_new = rest[1] if len(rest) > 1 else "v2"
        print_diff(v_old, v_new)
        return

    version = args[0] if args else "v2"
    extracted = load_extracted(version)
    print_report(version, extracted)


if __name__ == "__main__":
    main()
