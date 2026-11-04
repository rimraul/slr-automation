"""Local/dev preflight -- mirrors the CI `probe` job's three-asset-lock
readiness check (§5.5) as a reusable, importable Python check, plus
whether FT_SCREENING is actually importable.

track2_pipeline_readiness_plan_2026-08-30.md §13 step 5 authors this file
with this asset-readiness check only. **Step 11 adds the `ReferenceRunSpec`
context check** (line 4236 -- asserting `ft_screening_config_sha256` in
the reference spec matches what's actually installed, §4.2), now that
`reference_spec.py` exists. `run_preflight()`'s `reference_spec` parameter
is optional: `reference_spec.json` itself is not written until step 12
(see `slr_pipeline.reference_spec`'s own module docstring), so a caller
that only has the four hash-locked step-11 fixtures and not yet a full
`ReferenceRunSpec` simply omits it and gets `None` for the new field
rather than a spurious failure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from slr_pipeline.reference_spec import ReferenceRunSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
FT_SCREENING_WHEEL_LOCK = REPO_ROOT / "track2" / "ft-screening-wheel.lock"
SLR_PIPELINE_WHEEL_LOCK = REPO_ROOT / "track2" / "SLR_PIPELINE_WHEEL.lock"
DOCLING_ASSETS_LOCK = REPO_ROOT / "assets" / "models" / "DOCLING_ASSETS.lock"


@dataclass(frozen=True)
class PreflightReport:
    ft_screening_wheel_lock_present: bool
    slr_pipeline_wheel_lock_present: bool
    docling_assets_lock_present: bool
    ft_screening_importable: bool
    # None when no `reference_spec` was passed to `run_preflight()` (no
    # ReferenceRunSpec available yet -- §11.3's driver doesn't exist until
    # step 12). True/False once a spec is available: whether the
    # LIVE `chapter_screening_config().model_dump()` still hashes to
    # exactly `reference_spec.ft_screening_config_sha256` -- catching a
    # since-drifted `ft_screening_adapter.chapter_screening_config()`
    # before a reference run silently screens with the wrong config.
    ft_screening_config_matches_reference_spec: Optional[bool] = None

    @property
    def all_step19_assets_ready(self) -> bool:
        """Mirrors the CI `probe` job's `all_assets_ready` output exactly
        (§5.5 -- three independent checks, ANDed, not a single `hashFiles`
        OR-across-patterns call)."""
        return (
            self.ft_screening_wheel_lock_present
            and self.slr_pipeline_wheel_lock_present
            and self.docling_assets_lock_present
        )


def _live_ft_screening_config_sha256() -> str:
    """The same `json.dumps(..., indent=2, sort_keys=True) + "\\n"` byte
    convention `slr_pipeline.reference_spec.dump_ft_screening_config_
    fixture()` writes to disk with -- hashed in memory here rather than
    round-tripped through that file, so this check still works even when
    the fixture file itself is stale or missing."""
    from slr_pipeline.adapters.ft_screening_adapter import chapter_screening_config

    text = json.dumps(chapter_screening_config().model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_preflight(reference_spec: "Optional[ReferenceRunSpec]" = None) -> PreflightReport:
    matches: Optional[bool] = None
    if reference_spec is not None:
        matches = _live_ft_screening_config_sha256() == reference_spec.ft_screening_config_sha256
    return PreflightReport(
        ft_screening_wheel_lock_present=FT_SCREENING_WHEEL_LOCK.exists(),
        slr_pipeline_wheel_lock_present=SLR_PIPELINE_WHEEL_LOCK.exists(),
        docling_assets_lock_present=DOCLING_ASSETS_LOCK.exists(),
        ft_screening_importable=importlib.util.find_spec("ft_screening") is not None,
        ft_screening_config_matches_reference_spec=matches,
    )


def main(argv=None) -> int:
    report = run_preflight()
    for field in ("ft_screening_wheel_lock_present", "slr_pipeline_wheel_lock_present",
                  "docling_assets_lock_present", "ft_screening_importable",
                  "ft_screening_config_matches_reference_spec"):
        print(f"{field}: {getattr(report, field)}")
    print(f"all_step19_assets_ready: {report.all_step19_assets_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
