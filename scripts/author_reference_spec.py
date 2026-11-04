"""Track 2 plan Step 12A sub-step (g): assembles the **real** `ReferenceRunSpec`
-- every field a genuine value, no placeholder -- and calls
`slr_pipeline.reference_spec.dump_reference_spec()` to write the canonical
`reference_spec.json`.

This is deliberately a *different* spec-builder from
`author_reference_cassette.py::_build_generation_spec()`: that one exists
to bootstrap a `ReviewConfig` capable of *driving* cassette generation
itself (step 12(d)), before `request_snapshot_sha256`/`fixture_corpus_digest`
exist, so it fills both with `_PLACEHOLDER_HEX64`. By sub-step (g), both are
real (12(d)'s `request_snapshots.lock`, 12(f)'s locked `fixture_corpus_digest`)
-- this script reads the *real* values and produces the one `reference_spec.json`
every later step (13's driver, 15's bundle, the acceptance harness) treats as
canonical. Two builders, not one reused with an `if placeholder:` branch,
because the generation-time spec's `ReviewConfig` projection is consumed
immediately and discarded (never dumped to `reference_spec.json`), while this
script's spec is the one frozen artifact -- conflating them would make it
easy to accidentally freeze a placeholder-bearing spec by forgetting a
follow-up "real" call, per round-16 finding 4's `fixture_corpus_digest`
ordering lesson (§13 step 12, top of the sub-step list).

Usage:
    python scripts/author_reference_spec.py

Reads:
    track2/reference_hashes.lock (protocol_sha256, extraction_schema_sha256,
        extraction_model_config_sha256, ft_screening_config_sha256,
        reference_reporting_context_sha256, fixture_corpus_digest,
        prisma_scr_matrix_sha256 -- all already locked by steps 11/12(b/c/e/f))
    tests/reference_run/fixtures/spec/request_snapshots.lock (12(d)'s 8
        per-surface sha256 digests, plain JSON despite the `.lock` extension
        -- unlike `track2/reference_hashes.lock`, which is a Python literal)

Writes:
    tests/reference_run/fixtures/spec/reference_spec.json
and registers (but does not itself write) `reference_spec_sha256` --
sub-step (h)'s own `hash_reference_inputs.py --write --target
reference_spec_sha256` invocation computes and locks it, once this file
exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Direct invocation (`python scripts/author_reference_spec.py`) sets
# `sys.path[0]` to this file's own directory (`scripts/`), not the repo
# root -- same fix as every other script in this family
# (author_fixture_corpus.py, author_prisma_scr_matrix.py,
# author_reference_cassette.py, hash_reference_inputs.py).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slr_pipeline.reference_spec import (  # noqa: E402
    EXTRACTION_PARSER_OPTIONS,
    FULL_TEXT_CRITERIA_SET,
    ReferenceRunSpec,
    _CANONICAL_DATABASES,
    _DATA_GOVERNANCE_DECLARATION,
    _EXCLUSION_CRITERIA,
    _FUNDING_STATEMENT,
    _INCLUSION_CRITERIA,
    _PCC_CONCEPT,
    _PCC_CONTEXT,
    _PCC_POPULATION,
    _RESEARCH_QUESTIONS,
    _REVIEW_RATIONALE,
    _REVIEW_TITLE,
    _SEED_QUERY,
    dump_reference_spec,
)

FIXTURES_SPEC_DIR = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "spec"
REQUEST_SNAPSHOTS_LOCK_PATH = FIXTURES_SPEC_DIR / "request_snapshots.lock"
REFERENCE_SPEC_PATH = FIXTURES_SPEC_DIR / "reference_spec.json"

# Every build-output hash `ReferenceRunSpec` embeds directly by name from
# `track2/reference_hashes.lock`, keyed identically to `ReferenceRunSpec`'s
# own field names -- both `_h()` below and `ReferenceRunSpec`'s constructor
# use the same string, so a typo here fails loudly (KeyError / pydantic
# validation) rather than silently embedding the wrong hash.
_LOCKED_HASH_FIELDS = (
    "protocol_sha256",
    "extraction_schema_sha256",
    "extraction_model_config_sha256",
    "ft_screening_config_sha256",
    "reference_reporting_context_sha256",
    "fixture_corpus_digest",
    "prisma_scr_matrix_sha256",
)


def build_reference_spec() -> ReferenceRunSpec:
    from scripts.hash_reference_inputs import _load_lock

    locked = _load_lock()
    missing = [name for name in _LOCKED_HASH_FIELDS if name not in locked]
    if missing:
        raise KeyError(
            f"track2/reference_hashes.lock is missing {missing} -- "
            f"run the owning sub-step's `hash_reference_inputs.py --write` first"
        )
    hashes = {name: locked[name]["sha256"] for name in _LOCKED_HASH_FIELDS}

    if not REQUEST_SNAPSHOTS_LOCK_PATH.exists():
        raise FileNotFoundError(
            f"{REQUEST_SNAPSHOTS_LOCK_PATH} does not exist yet -- run "
            f"author_reference_cassette.py (step 12(d)) first"
        )
    request_snapshot_sha256 = json.loads(REQUEST_SNAPSHOTS_LOCK_PATH.read_text(encoding="utf-8"))

    databases = list(_CANONICAL_DATABASES)
    return ReferenceRunSpec(
        review_title=_REVIEW_TITLE,
        review_rationale=_REVIEW_RATIONALE,
        research_questions=_RESEARCH_QUESTIONS,
        pcc_population=_PCC_POPULATION,
        pcc_concept=_PCC_CONCEPT,
        pcc_context=_PCC_CONTEXT,
        inclusion_criteria=_INCLUSION_CRITERIA,
        exclusion_criteria=_EXCLUSION_CRITERIA,
        seed_query=_SEED_QUERY,
        protocol_sha256=hashes["protocol_sha256"],
        databases=databases,
        execution_adapter_map={db: f"fx_{db}" for db in databases},
        qc_saturation_levels=[0.80, 0.90, 0.95],
        extraction_schema_sha256=hashes["extraction_schema_sha256"],
        extraction_model_config_sha256=hashes["extraction_model_config_sha256"],
        extraction_parser_options=EXTRACTION_PARSER_OPTIONS,
        full_text_criteria_set=FULL_TEXT_CRITERIA_SET,
        ft_screening_config_sha256=hashes["ft_screening_config_sha256"],
        request_snapshot_sha256=request_snapshot_sha256,
        funding_statement=_FUNDING_STATEMENT,
        data_governance_declaration=_DATA_GOVERNANCE_DECLARATION,
        reference_reporting_context_sha256=hashes["reference_reporting_context_sha256"],
        fixture_corpus_digest=hashes["fixture_corpus_digest"],
        prisma_scr_matrix_sha256=hashes["prisma_scr_matrix_sha256"],
    )


def _register_targets() -> None:
    from scripts.hash_reference_inputs import register_target

    # Step 12(h): whole-file sha256 of `reference_spec.json`, computed only
    # once this script has already written it (this same sub-step (g)).
    # Registered here, unconditionally at import time, so `--check` can
    # discover it later -- same idiom as every other target in this family
    # (author_fixture_corpus.py, author_prisma_scr_matrix.py). The
    # generator itself raises (via `_resolve_target_sha256`'s own
    # `FileNotFoundError`) if invoked before this script has run.
    @register_target("reference_spec_sha256")
    def _reference_spec_target() -> Path:
        return REFERENCE_SPEC_PATH


_register_targets()


def main() -> int:
    spec = build_reference_spec()
    dump_reference_spec(spec, REFERENCE_SPEC_PATH)
    print(f"wrote {REFERENCE_SPEC_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
