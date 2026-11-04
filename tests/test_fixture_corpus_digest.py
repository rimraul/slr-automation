"""`slr_pipeline.fixture_corpus_digest` -- track2_pipeline_readiness_plan_
2026-08-30.md §7.1's `fixture_corpus_digest` algorithm, tested in
isolation against synthetic `tmp_path` corpora (never the real fixture
corpus -- that round-trip is `tests/test_author_fixture_corpus.py`'s
job)."""
from __future__ import annotations

import hashlib

import pytest

from slr_pipeline.fixture_corpus_digest import compute_corpus_digest


def test_raises_if_corpus_root_does_not_exist(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        compute_corpus_digest(missing)


def test_deterministic_across_two_calls(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world", encoding="utf-8")
    first = compute_corpus_digest(tmp_path)
    second = compute_corpus_digest(tmp_path)
    assert first == second
    assert len(first) == 64  # hex sha256


def test_digest_changes_when_a_file_s_content_changes(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    before = compute_corpus_digest(tmp_path)
    target.write_text("hello, drifted", encoding="utf-8")
    after = compute_corpus_digest(tmp_path)
    assert before != after


def test_digest_changes_when_a_file_is_renamed(tmp_path):
    """The path key is part of what's hashed -- renaming without changing
    content must still change the digest."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    before = compute_corpus_digest(tmp_path)
    (tmp_path / "a.txt").rename(tmp_path / "renamed.txt")
    after = compute_corpus_digest(tmp_path)
    assert before != after


def test_digest_is_independent_of_filesystem_iteration_order(tmp_path):
    """§7.1 sorts by path_key explicitly -- two directories with the same
    files, populated in opposite name order, must produce the same
    digest."""
    for name in ("zzz.txt", "aaa.txt", "mmm.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    forward = compute_corpus_digest(tmp_path)

    tmp_path2 = tmp_path.parent / (tmp_path.name + "_reordered")
    tmp_path2.mkdir()
    for name in ("mmm.txt", "zzz.txt", "aaa.txt"):
        (tmp_path2 / name).write_text(name, encoding="utf-8")
    reordered = compute_corpus_digest(tmp_path2)
    assert forward == reordered


def test_matches_manual_reimplementation_of_the_spec(tmp_path):
    """Independent re-derivation of §7.1's algorithm (not importing
    `compute_corpus_digest`'s own internals) over a small synthetic tree,
    cross-checked against the real function."""
    (tmp_path / "native").mkdir()
    (tmp_path / "native" / "x.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "LICENSE.md").write_text("CC BY 4.0", encoding="utf-8")

    pairs = []
    for path in tmp_path.rglob("*"):
        if path.is_file():
            rel = path.relative_to(tmp_path).as_posix()
            pairs.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))
    pairs.sort(key=lambda pair: pair[0].encode("utf-8"))
    expected = hashlib.sha256()
    expected.update(b"fixture_corpus_digest_v1\n")
    for path_key, file_sha256 in pairs:
        expected.update(f"{path_key}\x00{file_sha256}\n".encode("utf-8"))

    assert compute_corpus_digest(tmp_path) == expected.hexdigest()


def test_empty_directory_has_a_stable_nonempty_digest(tmp_path):
    """An existing-but-empty corpus_root is a legitimate (if degenerate)
    input -- must not raise, and must still hash just the fixed prefix."""
    digest = compute_corpus_digest(tmp_path)
    assert digest == hashlib.sha256(b"fixture_corpus_digest_v1\n").hexdigest()
