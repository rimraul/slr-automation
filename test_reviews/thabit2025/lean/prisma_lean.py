"""
Lean prototype: build PRISMACounts for the Thabit reproduction directly
(no LangGraph run) and render the PRISMA flow diagram, reusing the real
pipeline's dedup logic and diagram renderer as-is.

identified counts come from wos_raw.json (642) / scopus_raw.json (674);
after_dedup comes from running the same _dedup_doi_only() the pipeline's
dedup_node uses on raw_records.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\projects\SLR_chapter")

from slr_pipeline.state import Record, PRISMACounts
from slr_pipeline.nodes.dedup import _dedup_doi_only
from slr_pipeline.nodes.reporting import _draw_prisma_diagram

HERE = Path(__file__).parent


def main() -> None:
    wos = json.loads((HERE / "wos_raw.json").read_text(encoding="utf-8"))
    scopus = json.loads((HERE / "scopus_raw.json").read_text(encoding="utf-8"))
    raw = json.loads((HERE / "raw_records.json").read_text(encoding="utf-8"))

    records = [Record(**r) for r in raw]
    deduplicated = _dedup_doi_only(records)

    prisma = PRISMACounts(
        identified={"wos": len(wos), "scopus": len(scopus)},
        after_dedup=len(deduplicated),
    )

    print(f"identified: wos={len(wos)} scopus={len(scopus)} total={prisma.total_identified}")
    print(f"after_dedup: {prisma.after_dedup} (removed {prisma.total_identified - prisma.after_dedup})")

    (HERE / "prisma_counts.json").write_text(
        json.dumps(prisma.model_dump(), indent=2), encoding="utf-8"
    )
    (HERE / "deduplicated_records.json").write_text(
        json.dumps([r.to_dict() for r in deduplicated], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig = _draw_prisma_diagram(prisma, {})
    out_png = HERE / "prisma_flow.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"-> {out_png}")


if __name__ == "__main__":
    main()
