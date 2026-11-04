"""`slr_pipeline.execution_subtree_digest` -- computes `PayloadSourceInfo.
execution_subtree_digest` from a `git` commit's own object store, never the
live working tree. Tested against small, real `tmp_path`-local git repos
(never this repo's own history -- that would make the test's expected
digest a moving target)."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from slr_pipeline.execution_subtree_digest import compute_execution_subtree_digest


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")
    return repo_root


def _commit_all(repo_root: Path, message: str) -> str:
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-q", "-m", message)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True)
    return result.stdout.decode("utf-8").strip()


def test_raises_value_error_if_commit_has_no_tracked_file_under_subtree(tmp_path):
    repo_root = _init_repo(tmp_path)
    (repo_root / "other.txt").write_text("unrelated", encoding="utf-8")
    commit = _commit_all(repo_root, "no slr_pipeline subtree")
    with pytest.raises(ValueError):
        compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit, subtree="slr_pipeline")


def test_raises_called_process_error_if_commit_does_not_exist(tmp_path):
    repo_root = _init_repo(tmp_path)
    (repo_root / "slr_pipeline").mkdir()
    (repo_root / "slr_pipeline" / "a.py").write_text("x = 1", encoding="utf-8")
    _commit_all(repo_root, "init")
    with pytest.raises(subprocess.CalledProcessError):
        compute_execution_subtree_digest(repo_root=repo_root, git_commit="0" * 40, subtree="slr_pipeline")


def test_deterministic_across_two_calls(tmp_path):
    repo_root = _init_repo(tmp_path)
    (repo_root / "slr_pipeline").mkdir()
    (repo_root / "slr_pipeline" / "a.py").write_text("x = 1", encoding="utf-8")
    (repo_root / "slr_pipeline" / "b.py").write_text("y = 2", encoding="utf-8")
    commit = _commit_all(repo_root, "init")

    first = compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit, subtree="slr_pipeline")
    second = compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit, subtree="slr_pipeline")
    assert first == second
    assert len(first) == 64  # hex sha256


def test_reads_committed_bytes_not_the_live_working_tree(tmp_path):
    """The whole point of reading via `git show <commit>:<path>` instead
    of the filesystem: a dirty working tree (unrelated in-progress edits)
    must never perturb a digest computed for an already-recorded commit."""
    repo_root = _init_repo(tmp_path)
    (repo_root / "slr_pipeline").mkdir()
    target = repo_root / "slr_pipeline" / "a.py"
    target.write_text("x = 1", encoding="utf-8")
    commit = _commit_all(repo_root, "init")

    before = compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit, subtree="slr_pipeline")

    # Dirty the working tree without committing -- must not change the
    # digest for the already-recorded commit.
    target.write_text("x = 999  # dirtied, not committed", encoding="utf-8")
    (repo_root / "slr_pipeline" / "new_untracked.py").write_text("z = 3", encoding="utf-8")

    after = compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit, subtree="slr_pipeline")
    assert before == after


def test_digest_changes_across_commits_when_content_changes(tmp_path):
    repo_root = _init_repo(tmp_path)
    (repo_root / "slr_pipeline").mkdir()
    target = repo_root / "slr_pipeline" / "a.py"
    target.write_text("x = 1", encoding="utf-8")
    first_commit = _commit_all(repo_root, "init")

    target.write_text("x = 2", encoding="utf-8")
    second_commit = _commit_all(repo_root, "change")

    first_digest = compute_execution_subtree_digest(repo_root=repo_root, git_commit=first_commit, subtree="slr_pipeline")
    second_digest = compute_execution_subtree_digest(repo_root=repo_root, git_commit=second_commit, subtree="slr_pipeline")
    assert first_digest != second_digest


def test_matches_manual_reimplementation_of_the_spec(tmp_path):
    """Independent re-derivation of the algorithm straight from `git`'s own
    plumbing commands, cross-checked against the real function."""
    repo_root = _init_repo(tmp_path)
    (repo_root / "slr_pipeline").mkdir()
    (repo_root / "slr_pipeline" / "nodes.py").write_text("def f(): pass\n", encoding="utf-8")
    (repo_root / "slr_pipeline" / "state.py").write_text("class State: pass\n", encoding="utf-8")
    (repo_root / "other").mkdir()
    (repo_root / "other" / "ignored.py").write_text("should not be included", encoding="utf-8")
    commit = _commit_all(repo_root, "init")

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", "slr_pipeline"],
        cwd=repo_root, check=True, capture_output=True,
    ).stdout.decode("utf-8")
    pairs = []
    for path in listing.splitlines():
        if not path:
            continue
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repo_root, check=True, capture_output=True,
        ).stdout
        pairs.append((path, hashlib.sha256(blob).hexdigest()))
    pairs.sort(key=lambda pair: pair[0].encode("utf-8"))

    expected = hashlib.sha256()
    expected.update(b"execution_subtree_digest_v1\n")
    for path, blob_sha256 in pairs:
        expected.update(f"{path}\x00{blob_sha256}\n".encode("utf-8"))

    assert compute_execution_subtree_digest(
        repo_root=repo_root, git_commit=commit, subtree="slr_pipeline") == expected.hexdigest()


def test_default_subtree_is_slr_pipeline(tmp_path):
    repo_root = _init_repo(tmp_path)
    (repo_root / "slr_pipeline").mkdir()
    (repo_root / "slr_pipeline" / "a.py").write_text("x = 1", encoding="utf-8")
    commit = _commit_all(repo_root, "init")

    default_call = compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit)
    explicit_call = compute_execution_subtree_digest(repo_root=repo_root, git_commit=commit, subtree="slr_pipeline")
    assert default_call == explicit_call
