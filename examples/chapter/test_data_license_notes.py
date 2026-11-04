"""Every file under examples/chapter/data/ must have exactly one row in
LICENSE_NOTES.md with one of the four recognised provenance categories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"
_NOTES = _DATA / "LICENSE_NOTES.md"
_MANIFEST = _DATA / "topic_corpus_embeddings.manifest.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO_ROOT / "scripts" / "gen_topic_embeddings.py"

_RECOGNISED_CATEGORIES = {
    "author-created-synthetic",
    "model-derived",
    "generated-metadata",
    "authored-metadata",
}


def _parse_notes_table() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in _NOTES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        name, category = cells[0], cells[1]
        if name in ("file", "---") or set(name) <= {"-"}:
            continue
        if name in rows:
            raise AssertionError(f"LICENSE_NOTES.md has more than one row for {name!r}")
        rows[name] = category
    return rows


def test_license_notes_has_no_duplicate_rows():
    _parse_notes_table()  # raises on a duplicate filename


def _tracked_data_files() -> list[str]:
    out: list[str] = []
    for path in sorted(_DATA.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts:
            continue
        out.append(path.relative_to(_DATA).as_posix())
    return out


def test_every_data_file_has_a_recognised_license_row():
    table = _parse_notes_table()
    missing = []
    bad_category = []
    for rel in _tracked_data_files():
        if rel not in table:
            missing.append(rel)
        elif table[rel] not in _RECOGNISED_CATEGORIES:
            bad_category.append((rel, table[rel]))
    assert not missing, f"files under data/ with no LICENSE_NOTES.md row: {missing}"
    assert not bad_category, f"rows with an unrecognised category: {bad_category}"


def test_no_license_row_points_at_a_missing_file():
    table = _parse_notes_table()
    tracked = set(_tracked_data_files())
    stale = [name for name in table if name not in tracked]
    assert not stale, f"LICENSE_NOTES.md rows for files that do not exist: {stale}"


def test_all_four_categories_are_used():
    used = set(_parse_notes_table().values())
    assert _RECOGNISED_CATEGORIES <= used, f"unused categories: {_RECOGNISED_CATEGORIES - used}"


# --- embedding manifest integrity -----------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_embedding_manifest_is_complete_and_hashes_match():
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    required = {
        "corpus_file", "corpus_sha256", "generator", "generator_sha256",
        "embedding_model_id", "embedding_model_revision", "embedding_model_license",
        "embedding_model_attribution", "embeddings_sha256", "embeddings_shape",
        "embeddings_dtype", "sentence_transformers_version", "torch_version",
        "numpy_version", "generated_on", "platform",
    }
    assert required <= set(manifest), f"manifest missing keys: {required - set(manifest)}"

    # a concrete pinned HF snapshot commit, not null / a branch name
    revision = manifest["embedding_model_revision"]
    assert isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)

    assert manifest["corpus_sha256"] == _sha256(_DATA / manifest["corpus_file"])
    assert manifest["generator_sha256"] == _sha256(_GENERATOR), (
        "gen_topic_embeddings.py changed without the manifest being regenerated"
    )
    npy_path = _DATA / "topic_corpus_embeddings.npy"
    assert manifest["embeddings_sha256"] == _sha256(npy_path)
    assert manifest["embeddings_shape"] == [12, 384]
    assert manifest["embeddings_dtype"] == "float32"


def test_embedding_npy_matches_manifest_shape_and_dtype():
    import numpy as np

    arr = np.load(_DATA / "topic_corpus_embeddings.npy")
    assert arr.shape == (12, 384)
    assert arr.dtype == np.float32
