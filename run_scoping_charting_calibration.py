"""
run_scoping_charting_calibration.py -- Permanent, reusable calibration
runner for Stage 2's (charting via the extraction engine) real-model
empirical check, per Section 2.3/2.4 of
quality_appraisal_synthesis_plan_2026-08-22.md. Reused for Berg 2023's
own charting_calibration/ package (see test_reviews/berg2023/
charting_calibration/INTERPRETATION.md); reusable for any future
scoping-review PDF that needs the same check.

SAFETY DESIGN (all deliberate, learned from the Berg 2023 calibration's
own cap-review round):
  - Makes ZERO API calls unless invoked with --execute. Without that
    flag, only the no-cost dry-run preflight (real Docling parse, real
    chunk/batch counts, expected call count) runs.
  - Reads credentials ONLY from OPENAI_API_KEY in the process
    environment -- never imports a personal credential-loading module
    (this project's own my_api_keys convention, used by reproduce_review.py,
    is a personal, machine-specific loader outside this repo; a
    permanent, shareable script must not depend on it).
  - Requires --output-dir to be a genuinely NEW or EMPTY directory --
    refuses to run against an existing populated run directory, which
    would otherwise let run_extraction() silently RESUME from existing
    checkpoints (zero new calls), giving a false impression that a fresh
    calibration ran.
  - Requires --max-calls with --execute. The real, immediately-preceding
    Docling parse must fit within that caller-authorized ceiling, and an
    unexpected call beyond the parsed batch count raises immediately.
  - No SDK retries: inherited automatically from
    extraction/openai_runner.py's own run_openai_batch_extraction(),
    which passes max_retries=0 to the OpenAI() client explicitly (fixed
    2026-08-22, commit f96adaf, found during this exact calibration's
    own cap review -- the installed SDK otherwise defaults to 2
    transparent HTTP-layer retries per call).

Usage:
  Dry run (default, zero cost):
    python run_scoping_charting_calibration.py --pdf test_reviews/berg2023/review.pdf --output-dir <new_or_empty_dir>

  Real run (spends real money -- requires explicit opt-in, an authorized
  call ceiling, and OPENAI_API_KEY set). PowerShell example:
    $env:OPENAI_API_KEY = "<your-key>"
    python run_scoping_charting_calibration.py --pdf test_reviews/berg2023/review.pdf --output-dir <new_or_empty_dir> --execute --max-calls 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from slr_pipeline.adapters.scoping_charting_adapter import scoping_chart_schema, scoping_extraction_model_config
from slr_pipeline.extraction.batching import build_batches
from slr_pipeline.extraction.checkpoint import ExtractionCheckpointStore, VerifiedRunBundle
from slr_pipeline.extraction.engine import run_extraction
from slr_pipeline.extraction.openai_runner import run_openai_batch_extraction
from slr_pipeline.extraction.parsing import DoclingParserOptions, parse_pdf

DEFAULT_MAX_BATCH_CHARS = 12_000
# gpt-5.6-terra pricing, confirmed 2026-08-22 via OpenAI's own pricing
# announcement (https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/),
# effective 2026-07-30. Re-verify before relying on this for a future run --
# pricing changes over time and this script does not look it up live.
INPUT_PRICE_PER_M = 2.0
OUTPUT_PRICE_PER_M = 12.0


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dry_run_preflight(pdf_path: Path, *, max_batch_chars: int):
    """The no-OpenAI-cost half: real Docling parse and real batching.

    It never touches the OpenAI API. Docling itself may initialize or
    download a parser model when that dependency is not already cached,
    so "no-cost" here means no billable model call, not necessarily zero
    network/cache activity on a first-ever Docling run.
    """
    parsed = parse_pdf(pdf_path, options=DoclingParserOptions())
    batches = build_batches(parsed.chunks, max_batch_chars=max_batch_chars)
    return parsed, batches


def check_output_dir_reusable(output_dir: Path) -> None:
    """Finding (packaging review): refuses a populated existing directory
    -- a populated one would let `run_extraction()` silently RESUME from
    existing checkpoints (zero new calls), giving a false impression that
    a fresh calibration ran."""
    if output_dir.exists():
        existing_entries = list(output_dir.rglob("*"))
        if existing_entries:
            raise SystemExit(
                f"REFUSING TO PROCEED: --output-dir {output_dir} already exists and is not empty "
                f"({len(existing_entries)} entries found). A populated run directory would let "
                "run_extraction() silently RESUME from existing checkpoints (zero new calls), giving a "
                "false impression that a fresh calibration ran. Pass a new or genuinely empty directory."
            )


def execute_calibration(
    *, pdf_path: Path, output_dir: Path, document_id: str, max_batch_chars: int, max_calls: int,
    schema, model_config, preflight_parsed, preflight_batches,
) -> dict:
    """The real, billable half. Runs the SAME `run_extraction()` a
    production scoping charting run would use, hard-capped at
    `max_calls`, and builds the SAME audit-grade report structure the
    archived Berg 2023 calibration package carries (schema/parser
    config, per-field results with evidence, review statuses) -- never a
    weaker summary than what gets manually preserved. Raises (after
    still writing the report -- money was spent, evidence of what
    happened must never be discarded) if:
      - the real parse's own batch count exceeds `max_calls` (checked
        BEFORE any API access, so this specific case raises before ANY
        money is spent);
      - the actual number of calls made does not exactly equal the
        expected batch count;
      - the executed checkpoint's own `pdf_sha256` does not match the
        preflight's own `document_sha256` (proves the SAME bytes were
        actually charged/appraised, not a swapped-out file mid-run);
      - any document's `DocumentExtractionResult.is_complete` is False
        (a batch failed -- the run is charged but not usable)."""
    parsed = preflight_parsed
    expected_calls = len(preflight_batches)

    if max_calls <= 0:
        raise SystemExit(f"REFUSING TO PROCEED: max_calls must be positive, got {max_calls!r}.")
    if expected_calls <= 0:
        raise SystemExit("REFUSING TO PROCEED: the real parse produced zero batches, so there is nothing to calibrate.")
    if expected_calls > max_calls:
        raise SystemExit(
            f"REFUSING TO PROCEED: the real parse produced {expected_calls} batches (= {expected_calls} "
            f"expected API calls), which EXCEEDS the authorized max_calls={max_calls}. Aborting before "
            "any API access."
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "REFUSING TO PROCEED: OPENAI_API_KEY is not set in the environment. This script reads "
            "credentials ONLY from the environment -- set it explicitly before running with --execute."
        )

    # Re-check immediately before the paid operation when this function
    # is called directly rather than through main(). The directory is
    # created only after every no-cost authorization/credential check has
    # passed, so a refused run leaves no misleading empty run directory.
    check_output_dir_reusable(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    call_count = {"n": 0}
    def call_batch_fn(batch_chunks):
        call_count["n"] += 1
        if call_count["n"] > expected_calls:
            raise RuntimeError(
                f"REFUSING call #{call_count['n']}: hard-capped at exactly {expected_calls} calls "
                "(confirmed by the dry-run preflight immediately above). Aborting rather than spending "
                "beyond the confirmed cap."
            )
        print(f"  -> making REAL call {call_count['n']}/{expected_calls} ({len(batch_chunks)} chunks)...")
        return run_openai_batch_extraction(schema=schema, batch_chunks=batch_chunks, config=model_config)

    print(f"\nAuthorization check OK: {expected_calls} expected call(s) <= max_calls={max_calls}.")
    print(f"Making {expected_calls} REAL, billable API call(s) now...\n")
    run_id, _ = run_extraction(
        documents=[(document_id, str(pdf_path))],
        schema=schema, call_batch_fn=call_batch_fn, output_dir=output_dir,
        model_config=model_config, parser_options=DoclingParserOptions(), max_batch_chars=max_batch_chars,
    )
    print(f"\nrun_extraction() complete. run_id={run_id}, calls made={call_count['n']}")

    # The verified artifact is the source of truth for reporting. If its
    # own integrity verification fails, its manifest/checkpoint files are
    # already preserved on disk and the error propagates immediately. The
    # three calibration postconditions below are recorded in a report and
    # then raised after that report is written.
    store = ExtractionCheckpointStore(output_dir=output_dir, run_id=run_id)
    bundle = VerifiedRunBundle.from_store(store)
    bundle.verify()
    print("VerifiedRunBundle.verify() passed.")

    checkpoint = bundle.checkpoints[0]
    batch_executions = list(checkpoint.result.batch_executions)
    telemetry_log = [
        {
            "call_index": batch.batch_index + 1,
            "status": batch.status,
            "error": batch.error,
            "provider_model": batch.provider_model,
            "prompt_tokens": batch.prompt_tokens,
            "completion_tokens": batch.completion_tokens,
            "total_tokens": batch.total_tokens,
        }
        for batch in batch_executions
    ]
    total_prompt = bundle.manifest.telemetry_summary.total_prompt_tokens
    total_completion = bundle.manifest.telemetry_summary.total_completion_tokens
    real_cost = total_prompt / 1_000_000 * INPUT_PRICE_PER_M + total_completion / 1_000_000 * OUTPUT_PRICE_PER_M
    usage_complete = all(
        batch.prompt_tokens is not None and batch.completion_tokens is not None
        for batch in batch_executions
    )

    result = bundle.results[0]
    per_field_summary = {}
    for field in result.fields:
        per_field_summary[field.field_id] = {
            "status": field.status.value,
            "requires_human_review": field.requires_human_review,
            "n_values": len(field.values),
            "values": [
                {
                    "value": v.value,
                    "evidence": [
                        {"chunk_id": e.chunk_id, "quotation": e.excerpt, "locator": e.page, "section": e.section}
                        for e in v.evidence
                    ],
                }
                for v in field.values
            ],
        }
    total_evidence_spans = sum(
        len(value.evidence)
        for field in result.fields
        for value in (*field.values, *field.provisional_values)
    ) + sum(len(field.review_evidence) for field in result.fields)

    executed_pdf_sha256 = checkpoint.pdf_sha256
    is_complete = result.is_complete

    report = {
        "purpose": (
            "Stage 2 scoping-charting empirical calibration using the production extraction engine."
        ),
        "run_id": run_id,
        "document_sha256": executed_pdf_sha256,
        "document_sha256_preflight": parsed.document_sha256,
        "document_sha256_executed_checkpoint": executed_pdf_sha256,
        "pdf_sha256_match": executed_pdf_sha256 == parsed.document_sha256,
        "pdf_path_relative_to_repo_root": _relative_or_absolute(pdf_path.resolve()),
        "expected_calls": expected_calls, "calls_made": call_count["n"],
        "call_count_matches_expected": call_count["n"] == expected_calls,
        "is_complete": is_complete,
        "calibration_status": "complete" if is_complete else "INCOMPLETE -- at least one batch failed",
        "schema_id": bundle.manifest.extraction_schema_id, "schema_version": bundle.manifest.extraction_schema_version,
        "model": bundle.manifest.model, "model_config_id": bundle.manifest.model_config_id,
        "reasoning_effort": bundle.manifest.reasoning_effort, "max_completion_tokens": bundle.manifest.max_completion_tokens,
        "parser_name": bundle.manifest.parser_name, "parser_version": bundle.manifest.parser_version,
        "telemetry_per_call": telemetry_log,
        "telemetry_summary": bundle.manifest.telemetry_summary.model_dump(mode="json"),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "usage_complete": usage_complete,
        "cost_status": (
            "exact_from_complete_provider_telemetry"
            if usage_complete else
            "lower_bound_from_batches_with_available_provider_telemetry"
        ),
        "real_cost_usd": round(real_cost, 5),
        "total_evidence_spans": total_evidence_spans,
        "per_field_summary": per_field_summary,
        "artifact_paths_relative_to_repo_root": {
            "manifest": _relative_or_absolute(store.manifest_path()),
            "checkpoints_dir": _relative_or_absolute(store.checkpoints_dir),
        },
        "reproduction_note": (
            "Re-verify at zero API cost with ExtractionCheckpointStore(output_dir=<this output dir>, "
            f"run_id={run_id!r}) and VerifiedRunBundle.from_store(store).verify()."
        ),
    }
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REAL cost (from actual telemetry): ${real_cost:.5f}")
    print(f"Report written to: {report_path}")

    # Raise AFTER writing the report -- an integrity failure must be
    # loud and unmissable, but the report itself (proof of what actually
    # happened, since money was already spent) is never discarded.
    if call_count["n"] != expected_calls:
        raise RuntimeError(
            f"CALIBRATION INTEGRITY FAILURE: expected exactly {expected_calls} calls, but "
            f"{call_count['n']} were actually made. See {report_path} for the full record."
        )
    if not report["pdf_sha256_match"]:
        raise RuntimeError(
            f"CALIBRATION INTEGRITY FAILURE: the executed checkpoint's own pdf_sha256 "
            f"({executed_pdf_sha256!r}) does not match the preflight's own document_sha256 "
            f"({parsed.document_sha256!r}) -- the file may have changed between preflight and execution. "
            f"See {report_path} for the full record."
        )
    if not is_complete:
        raise RuntimeError(
            f"CALIBRATION FAILED: at least one batch did not complete successfully -- this run is charged "
            f"but not usable for interpretation. See {report_path} for the full record."
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the PDF to charting-calibrate against.")
    parser.add_argument("--output-dir", required=True, type=Path, help="A NEW or EMPTY directory for this run's own manifest/checkpoints. Refuses an existing, populated one.")
    parser.add_argument("--document-id", default="calibration-doc", help="record_id to use for this document (default: calibration-doc).")
    parser.add_argument("--max-batch-chars", type=int, default=DEFAULT_MAX_BATCH_CHARS)
    parser.add_argument("--execute", action="store_true", help="Actually spend money and make real API calls. Without this flag, only the no-cost dry-run preflight runs.")
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help=(
            "REQUIRED with --execute: the caller's own explicit authorization ceiling on the number of "
            "real API calls this run may make. This is an AUTHORIZATION cap, distinct from the internal "
            "hard cap derived from the PDF's own real batch count -- the run aborts BEFORE any API access "
            "if the real parse produces MORE batches than --max-calls authorizes, rather than silently "
            "executing however many calls the PDF happens to need."
        ),
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    if args.execute and args.max_calls is None:
        raise SystemExit(
            "REFUSING TO PROCEED: --execute requires --max-calls, an explicit authorization ceiling on "
            "the number of real API calls this run may make. Run WITHOUT --execute first (the no-cost "
            "dry-run preflight) to see how many batches/calls the PDF actually produces, then re-run with "
            "--execute --max-calls <N> once you have explicitly authorized that many calls."
        )
    if args.max_calls is not None and args.max_calls <= 0:
        raise SystemExit(f"REFUSING TO PROCEED: --max-calls must be a positive integer, got {args.max_calls!r}.")

    check_output_dir_reusable(args.output_dir)

    schema = scoping_chart_schema()
    model_config = scoping_extraction_model_config()

    print(f"PDF: {args.pdf}")
    print("Parsing with real Docling (local, no API cost)...")
    parsed, batches = dry_run_preflight(args.pdf, max_batch_chars=args.max_batch_chars)
    expected_calls = len(batches)

    print(f"document_sha256: {parsed.document_sha256}")
    print(f"Chunks: {len(parsed.chunks)}, Batches: {expected_calls}")
    print(f"Model: {model_config.model} (reasoning_effort={model_config.reasoning_effort}, max_completion_tokens={model_config.max_completion_tokens})")
    print(f"Expected API calls: {expected_calls}")
    print(
        f"Worst-case max output tokens (hard cap from config): {expected_calls * model_config.max_completion_tokens} "
        f"(output-only ceiling at the recorded ${OUTPUT_PRICE_PER_M:g}/M rate: "
        f"${expected_calls * model_config.max_completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_M:.5f}) -- "
        "input tokens are only known exactly after a real call; run with --execute for telemetry-based cost."
    )

    if not args.execute:
        print("\nDry run only (no --execute passed). No API calls made. Exiting.")
        return

    execute_calibration(
        pdf_path=args.pdf, output_dir=args.output_dir, document_id=args.document_id,
        max_batch_chars=args.max_batch_chars, max_calls=args.max_calls,
        schema=schema, model_config=model_config,
        preflight_parsed=parsed, preflight_batches=batches,
    )


if __name__ == "__main__":
    main()
