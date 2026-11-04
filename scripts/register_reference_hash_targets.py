"""Registers the five `slr_pipeline`-produced reference-run fixture hash
targets (`extraction_schema_sha256`, `extraction_model_config_sha256`,
`ft_screening_config_sha256`, `reference_reporting_context_sha256`,
`protocol_sha256`) with `slr_pipeline.hash_registry`.

Moved here (Track-2 packaging-boundary fix, CI run 34155102694): these five
targets' generators used to live inside `slr_pipeline.reference_charting_
adapter`/`slr_pipeline.reference_spec` themselves, each importing
`register_target` from this repo's own `scripts.hash_reference_inputs` and
computing its fixture path via `Path(__file__).resolve().parent.parent` --
both assume the importing module sits inside a git checkout with `tests/`
and `track2/` as true siblings. Once `slr_pipeline` is installed as a real,
non-editable wheel (a genuine site-packages install, not an editable
checkout-relative one), neither assumption holds: `scripts` is never
packaged into the wheel (`ModuleNotFoundError` on import), and
`Path(__file__)` resolves inside site-packages, not the checkout, so the
computed paths point nowhere real.

`slr_pipeline`'s own modules now own only the *pure content*
(`reference_extraction_schema()`, `chapter_screening_config()`, ...); this
repository-only script owns the on-disk paths that content gets dumped to
and registers them as CLI hash targets, following `scripts/hash_reference_
inputs.py`'s own discovery-module convention -- import this module (directly,
or via `TRACK2_HASH_DISCOVERY_MODULES=scripts.register_reference_hash_
targets`) before `--write`/`--check`ing any of these five targets."""
from __future__ import annotations

from pathlib import Path

from slr_pipeline.hash_registry import register_target

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_SPEC_DIR = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "spec"

EXTRACTION_SCHEMA_FIXTURE = FIXTURES_SPEC_DIR / "extraction_schema.json"
EXTRACTION_MODEL_CONFIG_FIXTURE = FIXTURES_SPEC_DIR / "extraction_model_config.json"
FT_SCREENING_CONFIG_FIXTURE = FIXTURES_SPEC_DIR / "ft_screening_config.json"
REFERENCE_REPORTING_CONTEXT_FIXTURE = FIXTURES_SPEC_DIR / "reference_reporting_context.json"
# Real, human-authored prose describing this rehearsal's methodological
# choices -- not a generated fixture -- so it lives at `track2/readiness_
# protocol.md`, not under `tests/reference_run/fixtures/` (matching
# `ReferenceRunSpec.protocol_uri`'s frozen literal).
READINESS_PROTOCOL_PATH = REPO_ROOT / "track2" / "readiness_protocol.md"


@register_target("extraction_schema_sha256")
def _extraction_schema_target() -> Path:
    return EXTRACTION_SCHEMA_FIXTURE


@register_target("extraction_model_config_sha256")
def _extraction_model_config_target() -> Path:
    return EXTRACTION_MODEL_CONFIG_FIXTURE


@register_target("ft_screening_config_sha256")
def _ft_screening_config_target() -> Path:
    return FT_SCREENING_CONFIG_FIXTURE


@register_target("reference_reporting_context_sha256")
def _reference_reporting_context_target() -> Path:
    return REFERENCE_REPORTING_CONTEXT_FIXTURE


@register_target("protocol_sha256")
def _protocol_target() -> Path:
    return READINESS_PROTOCOL_PATH
