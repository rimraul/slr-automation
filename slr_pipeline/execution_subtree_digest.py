"""`PayloadSourceInfo.execution_subtree_digest` -- a versioned digest over
the exact source-code identity of a `git` subtree (`slr_pipeline/` by
default) AT A GIVEN COMMIT, computed straight from `git`'s own object
store rather than the live working tree.

Reading from the commit (via `git ls-tree`/`git show`) rather than disk is
deliberate, not incidental: a caller reconstructing provenance for a run
that did not itself record `git_commit`/`execution_subtree_digest` at
runtime (see `reference_bundle.PayloadSourceInfo.execution_subtree_
provenance == "retrospective_reconstruction"`) can still recompute an
honest digest of "what this commit's `slr_pipeline/` subtree actually
contains" without that computation being perturbed by whatever unrelated
edits happen to sit in the current working tree when the digest is
computed after the fact.

**Algorithm (versioned, mirrors `slr_pipeline.fixture_corpus_digest`'s
existing convention as closely as the different data source allows)**:
- file set: every path `git ls-tree -r --name-only <git_commit> --
  <subtree>` lists -- i.e. every blob `git` actually recorded under
  `subtree` in that exact commit, not whatever happens to be on disk now;
- path key: the POSIX-style relative path `git ls-tree` itself reports
  (git always stores/reports tree paths `/`-separated, so no further
  normalisation is needed the way `fixture_corpus_digest`'s live
  filesystem walk requires);
- content hash: `sha256` of the exact committed blob bytes, read via
  `git show <git_commit>:<path>` -- never the working-tree copy of that
  path, which may differ from what was actually committed;
- ordering: the list of `(path, sha256_hex(blob_bytes))` pairs, sorted by
  `path` (byte order of its UTF-8 encoding);
- serialisation: `digest = sha256(b"execution_subtree_digest_v1\\n" +
  b"".join(f"{path}\\x00{blob_sha256}\\n".encode("utf-8") for ... in
  sorted list)).hexdigest()`.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_DIGEST_PREFIX = b"execution_subtree_digest_v1\n"


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True,
    )
    return result.stdout


def compute_execution_subtree_digest(*, repo_root: Path, git_commit: str, subtree: str = "slr_pipeline") -> str:
    """Recomputes the versioned digest of `subtree` as recorded at
    `git_commit`. Raises `subprocess.CalledProcessError` if `git_commit`
    does not exist in `repo_root`'s object store (e.g. an unfetched or
    typo'd commit), and `ValueError` if `git_commit` records no tracked
    file at all under `subtree` (almost certainly a wrong `subtree`
    argument, not a legitimately empty subtree)."""
    listing = _git(repo_root, "ls-tree", "-r", "--name-only", git_commit, "--", subtree).decode("utf-8")
    paths = [line for line in listing.splitlines() if line]
    if not paths:
        raise ValueError(f"git commit {git_commit!r} has no tracked file under subtree {subtree!r}")

    pairs: list[tuple[str, str]] = []
    for path in paths:
        blob = _git(repo_root, "show", f"{git_commit}:{path}")
        pairs.append((path, hashlib.sha256(blob).hexdigest()))
    pairs.sort(key=lambda pair: pair[0].encode("utf-8"))

    digest = hashlib.sha256()
    digest.update(_DIGEST_PREFIX)
    for path, blob_sha256 in pairs:
        digest.update(f"{path}\x00{blob_sha256}\n".encode("utf-8"))
    return digest.hexdigest()
