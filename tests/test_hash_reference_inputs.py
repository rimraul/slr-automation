"""`scripts/hash_reference_inputs.py`'s registry/write/check mechanism
(track2_pipeline_readiness_plan_2026-08-30.md §13 step 5) -- exercised
against throwaway targets registered inline; no target this repo actually
uses (those are steps 11/12's own, registered next to the code that
produces the file being hashed) is registered here."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.hash_reference_inputs as hri
import slr_pipeline.hash_registry as hash_registry

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch, tmp_path):
    """Every test gets an empty `TARGET_REGISTRY` and its own lock path --
    registering a target under its real, global name in one test must
    never leak into another.

    `register_target` is defined in `slr_pipeline.hash_registry` (Track-2
    packaging-boundary fix, CI run 34155102694); `hri.TARGET_REGISTRY` is
    just a re-exported name bound to the SAME dict object at import time.
    `register_target()`'s body mutates the dict via `slr_pipeline.hash_
    registry`'s own globals, never via `hri`'s -- so patching only `hri.
    TARGET_REGISTRY` no longer isolates anything; both bindings must be
    monkeypatched to one shared fresh dict."""
    fresh: dict = {}
    monkeypatch.setattr(hash_registry, "TARGET_REGISTRY", fresh)
    monkeypatch.setattr(hri, "TARGET_REGISTRY", fresh)
    monkeypatch.setattr(hri, "HASHES_LOCK_PATH", tmp_path / "reference_hashes.lock")
    monkeypatch.setattr(hri, "REPO_ROOT", tmp_path)


def test_register_target_then_write_then_check_round_trips(tmp_path):
    target_file = tmp_path / "protocol.md"
    target_file.write_text("v1", encoding="utf-8")
    hri.register_target("protocol_sha256")(lambda: target_file)

    hri.write_targets(["protocol_sha256"])
    hri.check_targets(["protocol_sha256"])  # must not raise
    hri.check_targets(None)  # checks all recorded targets


def test_check_detects_drift_after_input_changes(tmp_path):
    target_file = tmp_path / "protocol.md"
    target_file.write_text("v1", encoding="utf-8")
    hri.register_target("protocol_sha256")(lambda: target_file)
    hri.write_targets(["protocol_sha256"])

    target_file.write_text("v2 -- drifted", encoding="utf-8")
    with pytest.raises(ValueError):
        hri.check_targets(["protocol_sha256"])


def test_write_two_targets_in_one_call_preserves_a_third_already_written(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    hri.register_target("a_sha256")(lambda: a)
    hri.register_target("b_sha256")(lambda: b)
    hri.write_targets(["a_sha256"])
    hri.write_targets(["b_sha256"])  # a separate --write call, later
    hri.check_targets(None)  # both still verify


def test_write_unregistered_target_raises_key_error():
    with pytest.raises(KeyError):
        hri.write_targets(["does_not_exist"])


def test_write_target_with_missing_input_raises_file_not_found(tmp_path):
    missing = tmp_path / "does-not-exist-yet.json"
    hri.register_target("future_target")(lambda: missing)
    with pytest.raises(FileNotFoundError):
        hri.write_targets(["future_target"])


def test_register_target_twice_raises():
    hri.register_target("dup")(lambda: None)
    with pytest.raises(ValueError):
        hri.register_target("dup")(lambda: None)


def test_check_without_lock_file_is_a_no_op_for_empty_target_set(tmp_path):
    hri.check_targets(None)  # no lock file, no targets named -- nothing to check, must not raise


def test_check_without_target_fails_closed_on_registered_but_unlocked_target():
    """Round-19 finding: checking only the lock's own keys let a
    registered-but-never-locked target silently pass. Not a subprocess
    test -- exercises `check_targets` directly against the isolated
    in-process registry/lock the autouse fixture already provides."""
    hri.register_target("unlocked_target")(lambda: Path(__file__))
    with pytest.raises(KeyError, match="no recorded hash for registered target"):
        hri.check_targets(None)


# ---------------------------------------------------------------------------
# CLI, real subprocess -- proves target *discovery* actually happens in a
# fresh process, not just that in-process TARGET_REGISTRY mutation works
# ---------------------------------------------------------------------------

def test_cli_check_fails_closed_when_discovered_target_has_no_lock_entry(tmp_path):
    """Round-19 finding: a fresh CLI process (`python scripts/hash_
    reference_inputs.py --check`) previously started with an empty
    `TARGET_REGISTRY`, so `--check` with no `--target` silently "passed"
    having verified nothing, even once real targets existed elsewhere in
    the codebase. A fixture module registers a real target (discovered via
    `TRACK2_HASH_DISCOVERY_MODULES`, not present in the real repo's
    `track2/reference_hashes.lock`) -- the fresh subprocess must fail
    closed, not silently succeed."""
    fixture_pkg = tmp_path / "fixture_discovery_pkg"
    fixture_pkg.mkdir()
    (fixture_pkg / "__init__.py").write_text("", encoding="utf-8")
    target_file = tmp_path / "fixture_target_input.txt"
    target_file.write_text("hello", encoding="utf-8")
    # Leading "_" (not just "cli_..."): this subprocess runs `--check` with no
    # `--target` against the REAL repo's `track2/reference_hashes.lock`, whose
    # every entry is unregistered-in-this-subprocess too (no real target's
    # owning module is on `TRACK2_HASH_DISCOVERY_MODULES` here) --
    # check_targets(None) raises on the alphabetically-FIRST name in the
    # union, so this fixture target's own name must sort before every real
    # target name for the assertion below to observe ITS failure rather than
    # a real target's. "_" (0x5F) sorts before every lowercase letter, so
    # this holds regardless of which/how many real targets get registered.
    (fixture_pkg / "register_mod.py").write_text(
        "import scripts.hash_reference_inputs as hri\n"
        "hri.register_target('_cli_discovery_fixture_target')"
        f"(lambda: __import__('pathlib').Path(r'{target_file}'))\n",
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        TRACK2_HASH_DISCOVERY_MODULES="fixture_discovery_pkg.register_mod",
        PYTHONPATH=str(tmp_path) + os.pathsep + str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "hash_reference_inputs.py"), "--check"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "_cli_discovery_fixture_target" in result.stderr


def test_cli_check_passes_when_discovered_target_matches_written_lock(tmp_path, monkeypatch):
    """Same discovery mechanism, but `--write` then `--check` in two
    subprocesses against an isolated lock path (`TRACK2_TEST_HASHES_LOCK`
    is not read by production code -- this test instead points
    `TRACK2_HASH_DISCOVERY_MODULES` at a fixture module and runs both CLI
    invocations with `cwd` set to a throwaway repo copy of just the
    `scripts/` package, so the real repo's own lock is never touched)."""
    throwaway_repo = tmp_path / "throwaway_repo"
    (throwaway_repo / "scripts").mkdir(parents=True)
    (throwaway_repo / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    real_script = (REPO_ROOT / "scripts" / "hash_reference_inputs.py").read_text(encoding="utf-8")
    (throwaway_repo / "scripts" / "hash_reference_inputs.py").write_text(real_script, encoding="utf-8")

    fixture_pkg = throwaway_repo / "fixture_discovery_pkg"
    fixture_pkg.mkdir()
    (fixture_pkg / "__init__.py").write_text("", encoding="utf-8")
    target_file = throwaway_repo / "fixture_target_input.txt"
    target_file.write_text("hello", encoding="utf-8")
    (fixture_pkg / "register_mod.py").write_text(
        "import scripts.hash_reference_inputs as hri\n"
        "hri.register_target('cli_discovery_fixture_target')"
        f"(lambda: __import__('pathlib').Path(r'{target_file}'))\n",
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        TRACK2_HASH_DISCOVERY_MODULES="fixture_discovery_pkg.register_mod",
        PYTHONPATH=str(throwaway_repo),
    )
    script_path = throwaway_repo / "scripts" / "hash_reference_inputs.py"
    write_result = subprocess.run(
        [sys.executable, str(script_path), "--write", "--target", "cli_discovery_fixture_target"],
        cwd=throwaway_repo, env=env, capture_output=True, text=True,
    )
    assert write_result.returncode == 0, write_result.stdout + write_result.stderr
    assert (throwaway_repo / "track2" / "reference_hashes.lock").exists()

    check_result = subprocess.run(
        [sys.executable, str(script_path), "--check"],
        cwd=throwaway_repo, env=env, capture_output=True, text=True,
    )
    assert check_result.returncode == 0, check_result.stdout + check_result.stderr


def test_cli_direct_script_invocation_discovers_scripts_targets_without_pythonpath():
    """Bug found in review of b9b8001: `discover_targets()` imports
    `scripts.*`-qualified module names, but direct invocation (`python
    scripts/hash_reference_inputs.py`, as the module's own docstring
    documents) sets `sys.path[0]` to this file's own directory
    (`scripts/`), not the repo root -- so `scripts` (its own parent
    package) was unimportable and discovery raised `ModuleNotFoundError`
    unless the caller's environment already had the repo root on
    `PYTHONPATH`. `python -m scripts.hash_reference_inputs` was never
    affected (module invocation puts the repo root on `sys.path[0]`
    instead), which is why this went unnoticed. Exercises the exact
    documented direct-invocation form against this repo's own real lock
    and a real registered target, with `PYTHONPATH` stripped so the
    repo-root-on-sys.path fix is the only thing that can make it pass."""
    env = dict(os.environ, TRACK2_HASH_DISCOVERY_MODULES="scripts.author_prisma_scr_matrix")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "hash_reference_inputs.py"),
         "--check", "--target", "prisma_scr_matrix_sha256"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
