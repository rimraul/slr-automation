"""Berg post-2023 evidence-update wave launcher (Workstream C, user-named
findings 2026-09-22, both rounds): a ``search`` subcommand that performs the
real, restart-safe PubMed/OpenAlex/Scopus search only (no LLM call, no
dollar cap needed; Scopus credentials scoped via
``credentials.loaded_credentials()``, any ``retrieval_status`` outside
``{"exhaustive", "capped"}`` refused) and persists a hashed, provenance-bound
``UpdateWaveSearchArtifact`` (``update_driver.run_update_wave_search()``),
then mints a separate, immutable execution spec from the draft protocol
spec plus the search artifact's real per-database execution dates/executed
queries (``update_driver.build_update_execution_spec()``); a ``plan-screen``
subcommand (review finding 2, 2026-09-25 second round) that loads that
persisted artifact and execution spec and calls ``update_driver.
plan_update_wave()`` to reserve/print the census-vs-sampling decision --
entirely offline, no LLM call, no dollar spend -- for a human to review;
and a ``screen`` subcommand that loads the same persisted artifact/spec,
requires an ALREADY-PLANNED decision (``plan-screen`` must run first), and
wraps ``update_driver.run_update_wave()`` in exactly one ``CostLedger``/
``CostCapProxy`` pair enforcing ``--cap-usd`` -- mirroring
``slr_pipeline.track3_continuation_run``'s own ``run_continuation()``/
``main()`` launcher pattern (same ledger/proxy construction, same single
failure-safe finally block doing proxy shutdown + env-var restore +
``ledger.ensure_persisted()`` + ``call_log_path.touch(exist_ok=True)``, same
``credentials.loaded_credentials(("OPENAI_API_KEY",))`` scoping around the
one real call).

An ``estimate-screen`` subcommand (review finding 3, 2026-09-25 third round)
previews the census-vs-sampling decision for ANY candidate ``--cap-usd``/
``--sample-size``/``--sample-seed`` combination -- it loads the same real
search artifact/execution spec/parent and prices both options via
``update_driver.estimate_update_wave()``, but persists NOTHING (no census
decision file, no assembly manifest) and may be run any number of times
with different candidate values before a human settles on the one to
actually commit via ``plan-screen``.

``plan-screen`` and ``screen`` both refuse to run unless a search artifact
AND a persisted execution spec already exist for the given
``--review-run-id`` -- neither triggers its own search, so a misconfigured
or repeated invocation can never reach a live search endpoint, and can
never screen under the draft's placeholder ``search_dates="unknown"``.
``screen`` additionally refuses to run unless ``plan-screen`` has already
persisted a decision for this exact review_run_id/corpus/config/cap/
sample/scenario -- it never creates one itself.

Neither subcommand is authorized to run by this module's mere existence:
the standing engagement constraint (`inherited-sniffing-boole.md`) requires
a separate, explicit go-ahead naming the exact command before either is
actually invoked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from slr_pipeline import credentials
from slr_pipeline.config import ReviewConfig
from slr_pipeline.evaluation.bundle_artifacts import (
    ReproductionArtifactPaths,
    VerifiedReproductionArtifactSet,
    verify_reproduction_artifact_set,
)
from slr_pipeline.evaluation.reproduction_spec import ReproductionSpec
from slr_pipeline.evaluation.update_driver import (
    UpdateWaveExecutionSpecStore,
    UpdateWaveSearchArtifact,
    UpdateWaveSearchArtifactStore,
    build_update_execution_spec,
    estimate_update_wave,
    plan_update_wave,
    run_update_wave,
    run_update_wave_search,
    verify_search_artifact_and_execution_spec_before_screening,
)

_DEFAULT_BERG_ROOT = Path("test_reviews/berg2023")


def _load_parent(*, parent_dir: Path, gold_standard: Path) -> VerifiedReproductionArtifactSet:
    """Builds and hash-verifies the Berg parent bundle from a reproduction
    directory (``reproduction_spec.json``/``stage_corpus.json``/
    ``pipeline_evaluation.json``/``gate_report.json``/
    ``evaluation_adjudications.jsonl``) plus its ``gold_standard.json`` --
    the same layout ``tests/evaluation/test_update_driver.py``'s own
    ``_load_verified_berg_parent()`` uses."""
    paths = ReproductionArtifactPaths(
        reproduction_spec=parent_dir / "reproduction_spec.json",
        gold_standard=gold_standard,
        stage_corpus=parent_dir / "stage_corpus.json",
        pipeline_evaluation=parent_dir / "pipeline_evaluation.json",
        gate_report=parent_dir / "gate_report.json",
        adjudication_journal=parent_dir / "evaluation_adjudications.jsonl",
    )
    return verify_reproduction_artifact_set(paths)


def run_capped_update_wave_screening(
    *,
    sample_size: int,
    sample_seed: int,
    cost_scenario: str = "base",
    cfg: ReviewConfig,
    parent: VerifiedReproductionArtifactSet,
    update_spec: ReproductionSpec,
    search_artifact: UpdateWaveSearchArtifact,
    review_id: str,
    review_run_id: str,
    cap_usd: float,
    workspace: Path,
    upstream_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Wraps ``run_update_wave()`` in exactly one ``CostLedger``/
    ``CostCapProxy`` pair for the whole call, pointing ``OPENAI_BASE_URL``
    at the proxy before invoking it -- the same mechanism
    ``track3_continuation_run.run_continuation()`` uses, reused unmodified
    (``cost_cap/ledger.py``/``cost_cap/proxy.py`` are neither modified nor
    reimplemented here). ``search_artifact.records()`` supplies
    ``new_raw_records`` -- this function never calls ``search_node()``
    itself.

    ``upstream_url``, if given, overrides the proxy's default real-OpenAI
    upstream -- test-only, so an offline end-to-end test can point a real
    ``CostCapProxy`` at a fake local upstream server instead.

    Proxy shutdown, ``OPENAI_BASE_URL`` restoration, ``ledger.
    ensure_persisted()``, and call-log creation all happen in ONE
    ``finally`` block (user-named finding: these last two used to sit
    AFTER the try/finally, so an exception from ``run_update_wave()`` would
    skip them entirely, leaving no ledger file and no call-log file on
    disk for a failed/interrupted run). ``proxy.stop()`` itself is wrapped
    in its own nested ``try``/``finally`` (user-named finding, round 3): if
    ``proxy.stop()`` raised, the outer ``finally``'s remaining statements
    (env-var restoration, ``ensure_persisted()``, call-log creation) would
    otherwise be skipped too."""
    from slr_pipeline.cost_cap.ledger import CostLedger
    from slr_pipeline.cost_cap.proxy import CostCapProxy

    workspace.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(workspace / "cost_cap_ledger.json", cap_usd=cap_usd)
    call_log_path = workspace / "cost_cap_call_log.jsonl"
    proxy_kwargs: Dict[str, Any] = {} if upstream_url is None else {"upstream_url": upstream_url}
    proxy = CostCapProxy(ledger, call_log_path, port=0, **proxy_kwargs)
    proxy.start()
    previous_base_url = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = proxy.base_url
    try:
        state = run_update_wave(
            cfg=cfg, parent=parent, update_spec=update_spec,
            new_raw_records=search_artifact.records(), review_id=review_id, review_run_id=review_run_id,
            cap_usd=cap_usd, sample_size=sample_size, sample_seed=sample_seed, cost_scenario=cost_scenario,
        )
    finally:
        try:
            proxy.stop()
        finally:
            if previous_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = previous_base_url
            # Same zero-spend-still-leaves-a-file discipline as
            # track3_continuation_run.run_continuation(): CostLedger only
            # persists after a first committed spend, and the call-log file
            # is only ever opened lazily on the first logged call -- both
            # must still happen even when run_update_wave() raised, or
            # proxy.stop() itself raised.
            ledger.ensure_persisted()
            call_log_path.touch(exist_ok=True)
    return state


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search", help="Run the real, restart-safe search-only stage. No LLM call, no --cap-usd."
    )
    search_parser.add_argument("--review-config", required=True, type=Path)
    search_parser.add_argument("--review-run-id", required=True)
    search_parser.add_argument(
        "--db-queries", type=Path, default=None,
        help="JSON file of {database: translated_query} overrides (e.g. db_queries_draft.json) -- "
             "required for any database in search.py's _STRICT_TRANSLATION_DBS (e.g. scopus).",
    )
    search_parser.add_argument(
        "--draft-update-spec", required=True, type=Path,
        help="The draft ReproductionSpec (search_dates='unknown') this update wave's protocol was authored "
             "as -- immediately after a real search completes, its real per-database execution dates and "
             "each backend's actual executed query are minted into a separate, immutable execution spec; "
             "the draft itself is never mutated.",
    )

    def _add_common_decision_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--review-config", required=True, type=Path)
        sub.add_argument("--review-run-id", required=True)
        sub.add_argument("--review-id", required=True)
        sub.add_argument(
            "--parent-dir", type=Path, default=_DEFAULT_BERG_ROOT / "reproduction",
            help="Directory holding the verified parent's reproduction_spec.json/stage_corpus.json/"
                 "pipeline_evaluation.json/gate_report.json/evaluation_adjudications.jsonl.",
        )
        sub.add_argument(
            "--gold-standard", type=Path, default=_DEFAULT_BERG_ROOT / "gold_standard.json",
        )
        sub.add_argument(
            "--cap-usd", type=float, required=True,
            help="Total dollar spend cap for the census-vs-sampling decision and (if 'screen') the "
                 "screening run itself. Required (no default).",
        )
        sub.add_argument(
            "--sample-size", type=int, required=True,
            help="Size of the fixed-seed sample decide_census_vs_sampling() prices/draws as the "
                 "alternative to census screening. Required (no default).",
        )
        sub.add_argument(
            "--sample-seed", type=int, required=True,
            help="Seed for the deterministic sample draw. Required (no default).",
        )
        sub.add_argument(
            "--cost-scenario", choices=["low", "base", "high"], default="base",
            help="Pricing scenario for the census-vs-sampling decision. Defaults to 'base'.",
        )

    estimate_screen_parser = subparsers.add_parser(
        "estimate-screen",
        help="Offline, non-persisting: preview the census-vs-sampling decision for candidate "
             "--cap-usd/--sample-size/--sample-seed values. Writes nothing to disk -- try as many "
             "candidate values as needed before committing one via 'plan-screen'.",
    )
    _add_common_decision_args(estimate_screen_parser)

    plan_screen_parser = subparsers.add_parser(
        "plan-screen",
        help="Offline: plan (and persist) the census-vs-sampling decision for a review_run_id, "
             "without screening a single record or making any LLM call. Review its printed decision "
             "before running 'screen' to consume it.",
    )
    _add_common_decision_args(plan_screen_parser)

    screen_parser = subparsers.add_parser(
        "screen",
        help="Screen an already-planned (via 'plan-screen') decision under an enforced dollar cap. "
             "Refuses to run if no decision has been planned yet.",
    )
    _add_common_decision_args(screen_parser)

    args = parser.parse_args(argv)
    cfg = ReviewConfig.model_validate(json.loads(args.review_config.read_text(encoding="utf-8")))

    if args.command == "search":
        db_queries = None
        if args.db_queries is not None:
            db_queries = json.loads(args.db_queries.read_text(encoding="utf-8"))
        artifact = run_update_wave_search(cfg=cfg, review_run_id=args.review_run_id, db_queries=db_queries)
        draft_spec = ReproductionSpec.load(args.draft_update_spec)
        execution_spec = build_update_execution_spec(draft_spec=draft_spec, search_artifact=artifact)
        spec_store = UpdateWaveExecutionSpecStore(output_dir=cfg.ensure_output_dir(), review_run_id=args.review_run_id)
        spec_store.persist(execution_spec)
        print(
            f"search complete: per_database_counts={artifact.per_database_counts}, "
            f"total_records={artifact.total_records}, execution_spec={spec_store._path()}"
        )
        return 0

    store = UpdateWaveSearchArtifactStore(output_dir=cfg.ensure_output_dir(), review_run_id=args.review_run_id)
    search_artifact = store.load_if_exists()
    if search_artifact is None:
        parser.error(
            f"no persisted search artifact for review_run_id={args.review_run_id!r} -- run the 'search' "
            "subcommand first; 'plan-screen'/'screen' never trigger their own search"
        )
    spec_store = UpdateWaveExecutionSpecStore(output_dir=cfg.ensure_output_dir(), review_run_id=args.review_run_id)
    update_spec = spec_store.load_if_exists()
    if update_spec is None:
        parser.error(
            f"no persisted execution spec for review_run_id={args.review_run_id!r} -- run the 'search' "
            "subcommand first; 'plan-screen'/'screen' never use the draft spec directly"
        )
    verify_search_artifact_and_execution_spec_before_screening(
        cfg=cfg, search_artifact=search_artifact, update_spec=update_spec,
    )
    parent = _load_parent(parent_dir=args.parent_dir, gold_standard=args.gold_standard)

    if args.command == "estimate-screen":
        decision = estimate_update_wave(
            cfg=cfg, parent=parent, update_spec=update_spec, new_raw_records=search_artifact.records(),
            review_id=args.review_id, cap_usd=args.cap_usd, sample_size=args.sample_size,
            sample_seed=args.sample_seed, cost_scenario=args.cost_scenario,
        )
        print(decision.render())
        return 0

    if args.command == "plan-screen":
        decision = plan_update_wave(
            cfg=cfg, parent=parent, update_spec=update_spec, new_raw_records=search_artifact.records(),
            review_id=args.review_id, review_run_id=args.review_run_id, cap_usd=args.cap_usd,
            sample_size=args.sample_size, sample_seed=args.sample_seed, cost_scenario=args.cost_scenario,
        )
        print(decision.render())
        return 0

    workspace = cfg.ensure_output_dir() / "update_wave_cost_cap" / args.review_run_id

    with credentials.loaded_credentials(("OPENAI_API_KEY",)):
        state = run_capped_update_wave_screening(
            cfg=cfg, parent=parent, update_spec=update_spec, search_artifact=search_artifact,
            review_id=args.review_id, review_run_id=args.review_run_id, cap_usd=args.cap_usd,
            sample_size=args.sample_size, sample_seed=args.sample_seed, cost_scenario=args.cost_scenario,
            workspace=workspace,
        )
    print(f"screening complete: included={len(state.get('included_records') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
