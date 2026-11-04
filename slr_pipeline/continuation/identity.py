"""Unit A items 1-2 -- parent identity via signed release bundle, not a
workspace path.

Item 1: `resolve_parent_bundle()` looks `parent_bundle_id` up in
`run_registry.json`, opens the payload via `open_bundle()` (with
`run_registry_entry`/`run_registry_key` both supplied -- v41/round-38
finding 1, omitting `run_registry_key` leaves `bundle.run_registry_key` at
its default `None` and `verify_release_structure()`'s check 6,
`reference_bundle.py:734-737`, unconditionally fails), then calls
`verify_release_structure(bundle)`, which itself is what enforces
`decision == "APPROVED"` (its own final check, after all 6 structural
checks pass -- `reference_bundle.py:739-740`). `entry.decision` is
deliberately NOT pre-checked here before `open_bundle()`/
`verify_release_structure()` run: `verify_release_structure()`'s own
docstring guarantee is that "a genuinely audit-closed REJECTED bundle
stays distinguishable from a malformed one" -- that guarantee only holds
if a REJECTED entry is still run through every structural check first, so
a REJECTED-but-tampered bundle correctly raises `ReleaseStructureInvalid`
rather than being reported as merely "not approved."

Item 2: `check_parent_workspace_consistency()` is a NARROWER check than the
plan's own item 2 describes. Independently re-verified against live source
while implementing this: no durable workspace-side artifact anywhere in
`slr_pipeline` (`track3_run.py`/`track3_run_child.py`'s own
`terminal_state.json`/`thread_id.txt`/`graph_checkpoints.sqlite`/
`run_mode.json`) ever records `git_commit`/`execution_subtree_digest` --
both fields exist ONLY on `PayloadSourceInfo`, computed by
`finalize_bundle()`/publish tooling at bundle-PUBLISH time (a separate,
later, out-of-process step per Unit A item 6's own `build_payload()`
discussion), never written into a running workspace. The
`git_commit`/`execution_subtree_digest` cross-check Unit A item 2
describes is therefore not implementable against any value that exists
today -- this function instead checks only that the three expected
workspace files are present (existence/shape, not identity), and this gap
is flagged back to the plan/user rather than silently invented against
made-up workspace metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from slr_pipeline.reference_bundle import (
    Bundle,
    RunRegistry,
    open_bundle,
    verify_release_structure,
)
from slr_pipeline.track3_run import Track3ConfigError

_THREAD_ID_FILENAME = "thread_id.txt"  # track3_run.py's own module-private constant, same value


def resolve_parent_bundle(parent_bundle_id: str, bundles_root: Path) -> Bundle:
    """`bundles_root` is the directory holding `run_registry.json` and one
    `<bundle_id>/payload/` subdirectory per published bundle (confirmed
    against the real fixture layout,
    `track3_runs/bundles/t2-a24952b7a9805008f408c049/payload/...`)."""
    bundles_root = Path(bundles_root)
    registry_path = bundles_root / "run_registry.json"
    if not registry_path.is_file():
        raise Track3ConfigError(f"--parent-bundle-id {parent_bundle_id}: no run_registry.json at {registry_path}")
    registry = RunRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
    entry = registry.runs.get(parent_bundle_id)
    if entry is None:
        raise Track3ConfigError(
            f"--parent-bundle-id {parent_bundle_id}: no such entry in {registry_path}")

    payload_root = bundles_root / parent_bundle_id / "payload"
    bundle = open_bundle(payload_root, run_registry_entry=entry, run_registry_key=parent_bundle_id)
    verify_release_structure(bundle)  # raises ReleaseStructureInvalid/ReleaseNotApproved on any failure
    return bundle


def check_parent_workspace_consistency(parent_workspace: Optional[Path]) -> None:
    """No-op if `parent_workspace` is omitted (item 2: optional unless
    Unit B's `preserved_included_ids` is non-empty -- that conditional
    requirement is enforced by the launcher, not here, since it depends on
    Unit B's own derived partition). When given, only checks the three
    expected files exist -- see module docstring for why the
    `git_commit`/`execution_subtree_digest` cross-check itself is not
    implementable yet."""
    if parent_workspace is None:
        return
    parent_workspace = Path(parent_workspace)
    if not parent_workspace.is_dir():
        raise Track3ConfigError(f"--parent-workspace {parent_workspace}: not a directory")
    for name in ("terminal_state.json", _THREAD_ID_FILENAME, "graph_checkpoints.sqlite"):
        if not (parent_workspace / name).is_file():
            raise Track3ConfigError(f"--parent-workspace {parent_workspace}: missing expected file {name!r}")
