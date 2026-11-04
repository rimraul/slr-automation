"""
Lean prototype: BERTopic on the 871 deduplicated abstracts, cross-tabbed
against the LLM journal yes/no classification -- QA cross-check to see
where topic modelling agrees/disagrees with the journal-level filter
(e.g. a topic dominated by "no" journals that still contains PA-relevant
abstracts, or vice versa).

Offline, no paid API calls -- local sentence-transformer embeddings +
BERTopic clustering only.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent


def main() -> None:
    records = json.loads((HERE / "deduplicated_records.json").read_text(encoding="utf-8"))
    cache = json.loads((HERE / "journal_classification.json").read_text(encoding="utf-8"))

    docs, journals, decisions, titles = [], [], [], []
    for r in records:
        abstract = (r.get("abstract") or "").strip()
        if not abstract:
            continue
        journal = (r.get("journal") or "").strip().lower()
        docs.append(abstract)
        journals.append(journal)
        decisions.append(cache.get(journal, "unknown"))
        titles.append(r.get("title") or "")

    print(f"{len(docs)} abstracts going into BERTopic")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, show_progress_bar=True)

    topic_model = BERTopic(min_topic_size=10, calculate_probabilities=False, verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings)

    info = topic_model.get_topic_info()
    print("\n=== Topic overview ===")
    print(info.to_string())

    # Cross-tab: per topic, how many docs came from journal-filter yes/no/unknown
    by_topic = {}
    for t, dec in zip(topics, decisions):
        by_topic.setdefault(t, Counter())[dec] += 1

    print("\n=== Per-topic journal-filter yes/no split ===")
    rows = []
    for t in sorted(by_topic, key=lambda t: -sum(by_topic[t].values())):
        c = by_topic[t]
        total = sum(c.values())
        words = ", ".join(w for w, _ in topic_model.get_topic(t)[:6]) if t != -1 else "(outlier)"
        row = f"topic {t:>4} (n={total:>3})  yes={c['yes']:>3} no={c['no']:>3}  [{words}]"
        print(row)
        rows.append(row)

    out = {
        "n_docs": len(docs),
        "topic_info": info.to_dict(orient="records"),
        "per_topic_split": {
            str(t): dict(c) for t, c in by_topic.items()
        },
    }
    (HERE / "bertopic_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n-> bertopic_results.json")

    # Flag disagreement cases: topics that are majority "yes" journals but
    # small "no" minority (or vice versa) -- worth a manual look.
    print("\n=== Topics with meaningful yes/no mix (possible misclassifications) ===")
    for t, c in by_topic.items():
        if t == -1:
            continue
        total = c["yes"] + c["no"]
        if total < 5:
            continue
        minority = min(c["yes"], c["no"])
        if minority >= 3:
            words = ", ".join(w for w, _ in topic_model.get_topic(t)[:6])
            print(f"topic {t}: yes={c['yes']} no={c['no']}  [{words}]")


if __name__ == "__main__":
    main()
