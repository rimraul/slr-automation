"""The shared hash-computation mechanism for the reference-run's many
named build-output hashes (`protocol_sha256`, `extraction_schema_sha256`,
`prisma_scr_matrix_sha256`, `fixture_corpus_digest`, `reference_spec_
sha256`, `corpus_oracle_pdf_sha256`, ... -- see §0.4's full name->
generator->step mapping).

track2_pipeline_readiness_plan_2026-08-30.md §13 step 5 authors this
**mechanism only** -- a pluggable target registry plus a `--write --target
a,b,c` / `--check [--target a,b,c]` CLI, writing every computed hash into
one shared `track2/reference_hashes.lock` (JSON, `{target: {"path":
relpath, "sha256": ...}}`). No concrete target is registered here: each
target's owning step (11, 12(c)/(e)/(f)/(h), ...) registers its own
generator via `register_target()`, right next to the code that produces
the file being hashed -- the same "script has no other owning step; only
its invocation against real inputs happens at its using step" split
already established for `emit_licence_manifest.py` (§12.5) and
`provision_reference_assets.py` (§5.3).

The registry mechanism itself (`register_target`/`TARGET_REGISTRY`/
`sha256_of`) lives in `slr_pipeline.hash_registry`, re-exported below --
moved out (Track-2 packaging-boundary fix, CI run 34155102694) so
`slr_pipeline` runtime modules that produce a hashed fixture can self-
register their own target without importing this repository-only module
(never packaged into the `slr_pipeline` wheel). Step 11's two targets that
used to self-register inside `slr_pipeline.reference_charting_adapter`/
`slr_pipeline.reference_spec` are now registered by `scripts/register_
reference_hash_targets.py` instead, alongside step 11's other three
targets -- that script owns the repo-relative fixture paths those two
wheel-packaged modules must no longer assume.

Usage (once targets exist):
    python scripts/hash_reference_inputs.py --write --target protocol_sha256
    python scripts/hash_reference_inputs.py --write --target fixture_corpus_digest,prisma_scr_matrix_sha256
    python scripts/hash_reference_inputs.py --check
    python scripts/hash_reference_inputs.py --check --target protocol_sha256
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from slr_pipeline.hash_registry import TARGET_REGISTRY, register_target, sha256_of  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
# Direct invocation (`python scripts/hash_reference_inputs.py`) sets
# `sys.path[0]` to this file's own directory (`scripts/`), not the repo
# root, so `discover_targets()`'s `importlib.import_module("scripts.
# author_prisma_scr_matrix")` (and every other `scripts.*` discovery
# target) fails with `ModuleNotFoundError` unless the caller happens to
# already have the repo root on `PYTHONPATH` -- `python -m scripts.hash_
# reference_inputs` is unaffected (module invocation puts the repo root
# on `sys.path[0]` instead), which is why that form always worked. Same
# fix as `scripts/author_prisma_scr_matrix.py`'s own top-of-file guard.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HASHES_LOCK_PATH = REPO_ROOT / "track2" / "reference_hashes.lock"

# TARGET_REGISTRY/register_target/sha256_of now live in `slr_pipeline.
# hash_registry` (imported above, re-exported) -- see this file's own
# docstring. `write_targets`/`check_targets` dispatch on a generator's
# return type: a `Path` is hashed via `sha256_of()` (the original,
# still-default contract); a `str` (step 12(f)'s `fixture_corpus_digest`
# -- a multi-file recursive digest per plan §7.1, not a single file's
# bytes) is recorded/compared as-is.

# Round-19 finding: a fresh CLI process (`python scripts/hash_reference_
# inputs.py ...`) never itself imports any target's owning module, so
# TARGET_REGISTRY was always empty at CLI time -- and `--check` with no
# `--target` silently "passed" having verified zero targets, even once a
# lock file with real entries existed. `main()` imports every module named
# here before resolving `--write`/`--check`'s target list, so the owning
# `@register_target(...)` calls actually run. Empty now -- no owning step
# exists yet (steps 11, 12(c)/(e)/(f)/(h), ... register their own targets
# next to the code that produces the file being hashed); each such step
# appends its own dotted module path to this tuple as it's authored.
# `TRACK2_HASH_DISCOVERY_MODULES` (comma-separated) extends this list for
# tests, without needing to hand-edit this file.
#
# Step 12(b) tried appending `slr_pipeline.reference_spec`/`slr_pipeline.
# reference_charting_adapter` here directly and broke `test_cli_check_
# passes_when_discovered_target_matches_written_lock`: that test copies
# just `scripts/hash_reference_inputs.py` into an isolated throwaway repo
# to prove discovery-via-env-var works in total isolation from this
# repo's own production targets -- but `slr_pipeline` is pip-installed
# into this environment, so the copied script could still import it
# regardless of the throwaway repo's own `PYTHONPATH`, silently pulling in
# every real target (`extraction_model_config_sha256`, etc.) that
# throwaway lock never wrote. Left empty here, matching step 11's own
# precedent: `track2/reference_hashes.lock`'s existing entries were
# likewise written by one-off `--write` invocations with
# `TRACK2_HASH_DISCOVERY_MODULES` set by hand for that invocation, not by
# a permanently-registered static module.
_STATIC_DISCOVERY_MODULES: tuple[str, ...] = ()


def _discovery_modules() -> tuple[str, ...]:
    extra = tuple(m for m in os.environ.get("TRACK2_HASH_DISCOVERY_MODULES", "").split(",") if m)
    return _STATIC_DISCOVERY_MODULES + extra


def discover_targets() -> None:
    """Imports every module in `_discovery_modules()` so its module-level
    `@register_target(...)` calls run, populating `TARGET_REGISTRY` --
    called by the CLI (`main()`) only; library callers that already
    imported their own target-owning modules directly don't need this."""
    for module_name in _discovery_modules():
        importlib.import_module(module_name)


def _resolve_target_sha256(name: str) -> tuple[str, str]:
    """Calls target `name`'s generator and returns `(sha256_hex,
    path_label)` -- `path_label` is the real relative path for a
    file-hashing target, or a descriptive (non-hashed) string for a
    digest-string target, recorded in the lock purely for human
    provenance (never read back for verification -- the generator itself
    is always re-invoked)."""
    result = TARGET_REGISTRY[name]()
    if isinstance(result, str):
        if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
            raise ValueError(
                f"target {name!r}'s generator returned {result!r}, which is not a "
                f"well-formed lowercase 64-char sha256 hex digest"
            )
        return result, f"<computed digest -- see {name}'s own generator>"
    path = result
    if not path.exists():
        raise FileNotFoundError(f"target {name!r}'s input does not exist yet: {path}")
    return sha256_of(path), str(path.relative_to(REPO_ROOT))


def _load_lock() -> dict:
    if not HASHES_LOCK_PATH.exists():
        return {}
    namespace: dict = {}
    exec(HASHES_LOCK_PATH.read_text(encoding="utf-8"), {}, namespace)
    return namespace.get("targets", {})


def _save_lock(targets: dict) -> None:
    HASHES_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["targets = {"]
    for name, meta in sorted(targets.items()):
        lines.append(f"    {name!r}: {{'path': {meta['path']!r}, 'sha256': {meta['sha256']!r}}},")
    lines.append("}")
    HASHES_LOCK_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_targets(names: list[str] | None) -> list[str]:
    if names is not None:
        unknown = [n for n in names if n not in TARGET_REGISTRY]
        if unknown:
            raise KeyError(f"unregistered target(s): {unknown} (registered: {sorted(TARGET_REGISTRY)})")
        return names
    return sorted(TARGET_REGISTRY)


def write_targets(names: list[str]) -> dict:
    """Computes each named target's current sha256 and upserts it into
    `HASHES_LOCK_PATH`. Errors immediately (via the target generator's own
    `FileNotFoundError`) if a named target's input file does not exist yet."""
    resolved = _resolve_targets(names)
    targets = _load_lock()
    for name in resolved:
        digest, path_label = _resolve_target_sha256(name)
        targets[name] = {"path": path_label, "sha256": digest}
    _save_lock(targets)
    return targets


def check_targets(names: list[str] | None) -> None:
    """Read-only. Recomputes and asserts equality against
    `HASHES_LOCK_PATH`'s recorded hashes; never writes. `names=None` checks
    the union of every *currently registered* target and every target the
    lock records (round-19 finding: checking only the lock's own keys let
    a registered-but-never-locked target silently pass, and let a fresh
    CLI process's then-empty `TARGET_REGISTRY` make `--check` silently
    "pass" against a missing lock having verified nothing; checking only
    `TARGET_REGISTRY` in isolation would, symmetrically, silently ignore a
    stale lock entry for a target no longer registered anywhere -- the
    union catches both directions and fails closed on either)."""
    locked = _load_lock()
    check_names = names if names is not None else sorted(set(TARGET_REGISTRY) | set(locked))
    for name in check_names:
        if name not in TARGET_REGISTRY:
            raise KeyError(f"target {name!r} is recorded in the lock but no longer registered")
        if name not in locked:
            raise KeyError(f"{HASHES_LOCK_PATH} has no recorded hash for registered target {name!r} -- run --write --target {name} first")
        actual, _ = _resolve_target_sha256(name)
        expected = locked[name]["sha256"]
        if actual != expected:
            raise ValueError(f"target {name!r} sha256 mismatch: expected {expected!r}, got {actual!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--target", default=None, help="comma-separated target name(s); default: all")
    args = parser.parse_args(argv)
    names = args.target.split(",") if args.target else None

    discover_targets()

    if args.write:
        if names is None:
            parser.error("--write requires --target")
        write_targets(names)
        print(f"wrote {len(names)} target(s) to {HASHES_LOCK_PATH.relative_to(REPO_ROOT)}")
    else:
        check_targets(names)
        print("OK: reference hashes match track2/reference_hashes.lock")
    return 0


if __name__ == "__main__":
    # Round-19 finding: run directly (`python scripts/hash_reference_
    # inputs.py`), this file executes as `__main__` -- a distinct module
    # object from `scripts.hash_reference_inputs`, with its own separate
    # `TARGET_REGISTRY` dict. `discover_targets()` imports each owning
    # module by its `scripts.hash_reference_inputs`-qualified name (the
    # same way tests and every other caller import this file), and each
    # of those calls `register_target` on *that* module object -- never
    # this one -- so a naive run would always see an empty registry
    # regardless of `discover_targets()`. Aliasing this already-executing
    # `__main__` module under its real dotted name in `sys.modules`
    # *before* `main()` runs means a later `import scripts.hash_reference_
    # inputs` anywhere else in this same process returns this exact module
    # object (Python's import cache short-circuits on the name, never
    # re-executing the file), so registrations land in the one registry
    # `main()` itself reads from.
    import sys as _sys
    _sys.modules.setdefault("scripts.hash_reference_inputs", _sys.modules["__main__"])
    raise SystemExit(main())
