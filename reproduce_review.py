"""
reproduce_review.py — Reproduce a published review with the SLR pipeline.

Loads gold_standard.json from a test review folder, builds a ReviewConfig
from the extracted search strategy and criteria, runs the pipeline stage by
stage capturing record counts, then evaluates recall/precision at each stage
against the gold standard included paper list.

Usage:
    python reproduce_review.py <review_folder> [options]

Options:
    --review-type  scoping | rapid | slr  (overrides gold_standard.review_type)
    --model        OpenAI model name (default: gpt-5.4-mini)
    --skip-search  Use cached raw_records.json instead of running live search
    --skip-ft      Skip full-text download (evaluate search+dedup+screening only)
    --max-results  Max results per database (default: 500)
    --no-report    Skip HTML report generation

Example:
    python reproduce_review.py test_reviews/jones2022_ai_gov
    python reproduce_review.py test_reviews/smith2023_ml_health --model gpt-4o --max-results 1000

Output:
    test_reviews/{review_id}/reproduction/
        raw_records.json            search hits (all databases)
        deduplicated_records.json
        screened_records.json       title/abstract pass
        fulltext_records.json       full-text pass
        evaluation_report.html      stage-by-stage recall/precision
        evaluation_report.json      machine-readable results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Pipeline path setup — only add slr_pipeline root; import evaluation as a package
PIPELINE_DIR = Path(__file__).parent / "slr_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, r"c:/projects")
try:
    import my_api_keys  # noqa: F401 — sets OPENAI_API_KEY, PUBMED_API_KEY, etc.
except ImportError:
    pass

from evaluation.berg2023_gates import evaluate_berg2023_gates
from evaluation.bundle_artifacts import ReproductionArtifactPaths, verify_reproduction_artifact_set
from evaluation.evaluator import PipelineEvaluation
from evaluation.gold_standard import GoldStandard, load_gold_standard
from evaluation.reproduction_spec import ReproductionSpec
from evaluation.stage_corpus import StageCorpus
from evaluation.workflow import evaluate_with_adjudication_journal
from slr_pipeline.state import Record


# ---------------------------------------------------------------------------
# Config builder from gold standard
# ---------------------------------------------------------------------------

def _map_database_names(gs_databases: List[str]) -> List[str]:
    """Map human-readable database names to API adapter keys."""
    mapping = {
        "pubmed": "pubmed",
        "medline": "pubmed",
        "embase": "openalex",       # no free API; use openalex as proxy
        "web of science": "openalex",
        "wos": "openalex",
        "scopus": "openalex",
        "openalex": "openalex",
        "semantic scholar": "semantic_scholar",
        "s2": "semantic_scholar",
        "arxiv": "arxiv",
        "crossref": "crossref",
        "springer": "springer",
        "springerlink": "springer",
        "ieee": "ieee",
        "ieee xplore": "ieee",
        "cinahl": "pubmed",
        "psycinfo": "pubmed",
        "psychinfo": "pubmed",
        "cochrane": "pubmed",
        "eric": "openalex",
    }
    out: List[str] = []
    for db in gs_databases:
        key = db.lower().strip()
        matched = None
        for pattern, api_name in mapping.items():
            if pattern in key:
                matched = api_name
                break
        if matched and matched not in out:
            out.append(matched)
    if not out:
        out = ["openalex", "pubmed", "semantic_scholar"]
    # Always supplement with openalex + semantic_scholar: broad open-access coverage
    # that proxies much of WoS/Scopus/Google Scholar without API restrictions
    for fallback in ("openalex", "semantic_scholar"):
        if fallback not in out:
            out.append(fallback)
    return out


def build_config_from_gold_standard(
    gs: GoldStandard,
    output_dir: Path,
    model: str = "gpt-5.4-mini",
    max_results: int = 500,
    review_type_override: Optional[str] = None,
) -> dict:
    """Convert a GoldStandard into a ReviewConfig dict for the pipeline."""
    import dataclasses

    from config import (
        ReviewConfig, SearchConfig, SearchStrategyConfig,
        ScreeningConfig, SynthesisConfig,
        QualityAppraisalConfig,
    )
    # Package-absolute import (needs the REPO ROOT on sys.path, unlike the
    # bare nodes/*.py-style imports above) -- reproduce_review.py lives AT
    # the repo root, so its own directory (implicitly sys.path[0] when run
    # directly) already satisfies this; see nodes/fulltext.py's own
    # docstring note on this exact two-convention split.
    _repo_root = str(Path(__file__).resolve().parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from slr_pipeline.adapters.scoping_charting_adapter import (
        scoping_chart_schema, scoping_extraction_model_config,
    )

    review_type = review_type_override or gs.review_type
    if review_type not in ("scoping", "rapid", "slr"):
        review_type = "scoping"

    databases = _map_database_names(gs.original_databases)
    language = gs.original_language_filter[0] if gs.original_language_filter else "en"

    # Scoping charting now runs through extraction_node (Stage 2 of
    # quality_appraisal_synthesis_plan_2026-08-22.md), never the legacy
    # ChartingConfig/charting_node -- the chapter's own calibrated
    # teaching-preset schema/model config. rapid/slr are UNCHANGED here
    # (a pre-existing, out-of-scope gap: this script never set
    # extraction_schema/extraction_model_config for them either, before
    # or after Stage 2 -- a review-specific structured-extraction schema
    # is the review author's own explicit choice, never silently
    # defaulted to the scoping charting schema).
    extraction_kwargs: Dict[str, Any] = {}
    if review_type == "scoping":
        extraction_kwargs = dict(
            extraction_schema=scoping_chart_schema().model_dump(mode="json"),
            extraction_model_config=dataclasses.asdict(scoping_extraction_model_config()),
        )

    cfg = ReviewConfig(
        review_type=review_type,
        title=f"Reproduction: {gs.review_citation}",
        output_dir=str(output_dir),
        research_questions=gs.research_questions,
        population=gs.pcc_population,
        concept=gs.pcc_concept,
        context=gs.pcc_context,
        inclusion_criteria=gs.inclusion_criteria,
        exclusion_criteria=gs.exclusion_criteria,
        search=SearchConfig(
            databases=databases,
            query=gs.original_query,
            date_from=gs.original_date_from,
            date_to=gs.original_date_to,
            language=language,
            max_results_per_database=max_results,
            strategy_builder=SearchStrategyConfig(enabled=False),
        ),
        screening=ScreeningConfig(
            llm_model=model,
            auto_include_threshold=0.0,   # all decisions are auto-accepted in reproduction
            uncertain_threshold=0.0,       # no human-review queue during automated run
        ),
        synthesis=SynthesisConfig(
            llm_model=model,
            run_bertopic=False,   # skip heavy synthesis in reproduction
            generate_claim_backed_results=False,
            generate_legacy_imrad_narrative=False,
        ),
        quality_appraisal=QualityAppraisalConfig(),
        # No openai_api_key here (Track 1 Phase A credential-safety fix) --
        # every OpenAI-client construction site resolves the credential
        # exclusively from the OPENAI_API_KEY environment variable via
        # config.resolve_openai_api_key(); a real key must never be placed
        # on ReviewConfig/ReviewState/its persisted JSON again.
        **extraction_kwargs,
    )
    return cfg.to_dict()


# ---------------------------------------------------------------------------
# Stage-by-stage capture using graph.stream()
# ---------------------------------------------------------------------------

def _run_pipeline_stages(
    config_dict: dict,
    review_type: str,
    repro_dir: Path,
    skip_fulltext: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Run the pipeline and capture records at each stage.

    Returns a dict: stage_name -> list of record dicts.
    Stages: "search", "dedup", "screening", "fulltext", "included"
    """
    from state import initial_state

    # Choose graph
    if review_type == "slr":
        from graphs.slr import build_slr_graph as build_graph
    elif review_type == "rapid":
        from graphs.rapid import build_rapid_graph as build_graph
    else:
        from graphs.scoping import build_scoping_graph as build_graph

    graph = build_graph()
    state = initial_state(config_dict)
    thread_cfg = {"configurable": {"thread_id": f"repro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}

    stages: Dict[str, List[Dict]] = {}

    # Stream node-by-node to capture intermediate state
    print("  Running pipeline (streaming node by node)...")
    for step in graph.stream(state, config=thread_cfg, stream_mode="updates"):
        for node_name, node_updates in step.items():
            print(f"    [OK] {node_name}")

            if node_name in ("search_node", "search"):
                records = node_updates.get("raw_records", [])
                stages["search"] = records
                _save_json(repro_dir / "raw_records.json", records)
                print(f"      -> {len(records)} records")

            elif node_name in ("dedup_node", "dedup"):
                records = node_updates.get("deduplicated_records", [])
                stages["dedup"] = records
                _save_json(repro_dir / "deduplicated_records.json", records)
                print(f"      -> {len(records)} after dedup")

            elif node_name in ("screening_node", "screening"):
                # included_records = passed title/abstract screening
                records = node_updates.get("included_records", [])
                stages["screening"] = records
                _save_json(repro_dir / "screened_records.json", records)
                print(f"      -> {len(records)} after screening")

            elif node_name in ("fulltext_node", "fulltext"):
                records = node_updates.get("full_text_records", [])
                stages["fulltext"] = records
                _save_json(repro_dir / "fulltext_records.json", records)
                print(f"      -> {len(records)} after full-text")

            elif node_name in ("quality_appraisal_node", "quality_appraisal"):
                # Quality appraisal never filters/mutates full_text_records
                # (Section 1.2 of quality_appraisal_synthesis_plan_2026-08-22.md)
                # -- "included" here is simply whatever full-text screening
                # already decided; quality_appraisal_results/-review_queue
                # are reported separately for visibility only.
                records = stages.get("fulltext", [])
                qa_results = node_updates.get("quality_appraisal_results") or []
                stages["included"] = records
                _save_json(repro_dir / "included_records.json", records)
                print(f"      -> {len(records)} after quality appraisal ({len(qa_results)} appraised)")

            # Stop early if full-text skipped
            if skip_fulltext and node_name in ("screening_node", "screening"):
                print("  (--skip-ft set, stopping after screening)")
                raise _StopEarly()

    # If no quality appraisal (scoping), "included" = full_text_records
    if "included" not in stages and "fulltext" in stages:
        stages["included"] = stages["fulltext"]

    return stages


class _StopEarly(Exception):
    """Internal signal to break out of graph.stream() early."""


# ---------------------------------------------------------------------------
# Load cached stage data
# ---------------------------------------------------------------------------

def _load_cached_stages(repro_dir: Path) -> Dict[str, List[Dict]]:
    """Load previously saved stage snapshots for re-evaluation without re-running."""
    stages: Dict[str, List[Dict]] = {}
    file_map = {
        "search": "raw_records.json",
        "dedup": "deduplicated_records.json",
        "screening": "screened_records.json",
        "fulltext": "fulltext_records.json",
        "included": "included_records.json",
    }
    for stage, fname in file_map.items():
        p = repro_dir / fname
        if p.exists():
            stages[stage] = json.loads(p.read_text(encoding="utf-8"))
            print(f"  Loaded cached {stage}: {len(stages[stage])} records")
    return stages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main reproduction function
# ---------------------------------------------------------------------------

def _stage_corpus_from_snapshots(stages: Dict[str, List[Dict]]) -> StageCorpus:
    """Build a hash-bound corpus from captured stage snapshots."""
    missing = [name for name in ("search", "dedup", "screening") if name not in stages]
    if missing:
        raise RuntimeError(
            "reproduction evaluation requires raw, deduplicated, and screened "
            f"stage snapshots; missing: {', '.join(missing)}"
        )

    def records(name: str) -> tuple[Record, ...]:
        return tuple(Record.model_validate(item) for item in stages[name])

    return StageCorpus.build(
        raw_records=records("search"),
        deduplicated_records=records("dedup"),
        screened_records=records("screening"),
        full_text_records=records("fulltext") if "fulltext" in stages else None,
        included_records=records("included") if "included" in stages else None,
    )

def reproduce_review(
    review_dir: Path,
    review_type_override: Optional[str] = None,
    model: str = "gpt-5.4-mini",
    skip_search: bool = False,
    skip_fulltext: bool = False,
    max_results: int = 500,
    generate_report: bool = True,
) -> PipelineEvaluation:
    print(f"\n{'='*60}")
    print(f"Reproducing review: {review_dir.name}")
    print(f"{'='*60}")

    # --- 1. Load gold standard ---
    gs = load_gold_standard(review_dir)
    print(f"  Gold standard loaded: {gs.n_included} included papers")
    print(f"  DOI coverage: {gs.doi_coverage:.0%}")
    if gs.doi_coverage < 0.5:
        print("  ⚠  Low DOI coverage — matching will rely on title fuzzy match")

    # --- 2. Prepare output directory ---
    repro_dir = review_dir / "reproduction"
    repro_dir.mkdir(exist_ok=True)

    # --- 3. Run pipeline or load cached ---
    if skip_search and _has_cached_stages(repro_dir):
        print("  Using cached stage data (--skip-search)")
        stages = _load_cached_stages(repro_dir)
    else:
        config_dict = build_config_from_gold_standard(
            gs, repro_dir, model=model, max_results=max_results,
            review_type_override=review_type_override,
        )
        _save_json(repro_dir / "reproduction_config.json", config_dict)
        print(f"  Review type: {config_dict['review_type']}")
        print(f"  Search query: {config_dict['search']['query'][:100]}...")
        print(f"  Databases: {config_dict['search']['databases']}")
        print(f"  Date range: {config_dict['search']['date_from']} – {config_dict['search']['date_to']}")

        try:
            stages = _run_pipeline_stages(
                config_dict,
                review_type=config_dict["review_type"],
                repro_dir=repro_dir,
                skip_fulltext=skip_fulltext,
            )
        except _StopEarly:
            stages = _load_cached_stages(repro_dir)

    # --- 4. Evaluate ---
    print("\n  Evaluating against gold standard...")
    spec_path = repro_dir / "reproduction_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"{spec_path} is required. Author a versioned ReproductionSpec "
            "with explicit unknowns before evaluating this attempt."
        )
    reproduction_spec = ReproductionSpec.load(spec_path)
    stage_corpus = _stage_corpus_from_snapshots(stages)
    journal_path = repro_dir / "evaluation_adjudications.jsonl"
    eval_result = evaluate_with_adjudication_journal(
        reproduction_spec,
        gs,
        stage_corpus,
        journal_path=journal_path,
    )
    gate_report = evaluate_berg2023_gates(eval_result, reproduction_spec, gs, stage_corpus)

    # --- 5. Save results ---
    report_json = repro_dir / "evaluation_report.json"
    _save_json(report_json, eval_result.model_dump(mode="json"))
    print(f"  Saved: {report_json}")

    stage_corpus_path = repro_dir / "stage_corpus.json"
    _save_json(stage_corpus_path, stage_corpus.model_dump(mode="json"))
    pipeline_evaluation_path = repro_dir / "pipeline_evaluation.json"
    _save_json(pipeline_evaluation_path, eval_result.model_dump(mode="json"))
    gate_report_path = repro_dir / "gate_report.json"
    _save_json(gate_report_path, gate_report.model_dump(mode="json"))
    print(f"  Saved: {stage_corpus_path}")
    print(f"  Saved: {pipeline_evaluation_path}")
    print(f"  Saved: {gate_report_path}")

    # --- 5b. Verify the complete reproduction artifact set is internally
    # consistent and publication-ready. NOTE: "publication-ready" here means
    # the six files on disk are complete and mutually cross-verified via
    # verify_reproduction_artifact_set() -- this script does NOT publish them
    # into a ReferenceBundle. finalize_bundle(reproduction_artifacts=...) has
    # no production driver: this script never runs the LangGraph pipeline
    # (no workspace/terminal_state.json/PayloadSourceInfo/reporting artifacts
    # exist for it to bundle), and Track 3 continuation's publish_bundle.py
    # is architecturally unrelated (new live reviews, not published-review
    # reproductions). A real reproduction-bundle publication driver is a
    # separate, larger feature (its own workspace contract, provenance
    # model, CLI, tests, publication lifecycle) -- not implemented here.
    if not journal_path.exists():
        # No pending adjudications were recorded; an empty journal is valid.
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.touch()
    verify_reproduction_artifact_set(
        ReproductionArtifactPaths(
            reproduction_spec=spec_path,
            gold_standard=review_dir / "gold_standard.json",
            stage_corpus=stage_corpus_path,
            pipeline_evaluation=pipeline_evaluation_path,
            gate_report=gate_report_path,
            adjudication_journal=journal_path,
        )
    )
    print("  Reproduction artifact set verified (not published to a bundle)")

    # --- 6. Print summary ---
    eval_result.print_summary()

    # --- 7. Generate HTML report ---
    if generate_report:
        try:
            from evaluation.reporter import generate_html_report
            html_path = repro_dir / "evaluation_report.html"
            generate_html_report(eval_result, gs, html_path, gate_report=gate_report)
            print(f"  Saved: {html_path}")
        except ImportError as e:
            print(f"  ⚠  HTML report skipped (import error): {e}")

    return eval_result


def _has_cached_stages(repro_dir: Path) -> bool:
    return (repro_dir / "raw_records.json").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce a published review and evaluate pipeline")
    parser.add_argument("review_folder", help="Path to review folder (contains gold_standard.json)")
    parser.add_argument("--review-type", choices=["scoping", "rapid", "slr"], default=None)
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model for screening/charting")
    parser.add_argument("--skip-search", action="store_true",
                        help="Use cached raw_records.json (skip live search)")
    parser.add_argument("--skip-ft", action="store_true",
                        help="Stop after screening (skip full-text download)")
    parser.add_argument("--max-results", type=int, default=500,
                        help="Max results per database")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")

    args = parser.parse_args()

    review_dir = Path(args.review_folder)
    if not review_dir.is_absolute():
        review_dir = Path(__file__).parent / review_dir

    if not review_dir.exists():
        print(f"Error: {review_dir} does not exist")
        sys.exit(1)

    result = reproduce_review(
        review_dir=review_dir,
        review_type_override=args.review_type,
        model=args.model,
        skip_search=args.skip_search,
        skip_fulltext=args.skip_ft,
        max_results=args.max_results,
        generate_report=not args.no_report,
    )
