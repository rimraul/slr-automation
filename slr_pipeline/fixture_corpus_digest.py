"""Track 2 `track2_pipeline_readiness_plan_2026-08-30.md` §7.1's
`fixture_corpus_digest` algorithm -- one reusable implementation so step
12(f) (locking the digest via `hash_reference_inputs.py --write`) and the
later bundle relocation test (step 15,
`test_bundle_reproduces_fixture_corpus_digest`, which recomputes this same
digest against `payload/fixture_inputs/corpus/` -- the once-relocated
copy -- and asserts it equals the frozen value in `reference_spec.json`)
both compute the identical digest from two different `corpus/` directory
instances via the same code, not two independently-transcribed copies of
the algorithm that could silently drift apart.

**Algorithm (§7.1, verbatim)**:
- file set: every regular file under `corpus_root`, recursively;
- path key: POSIX path relative to `corpus_root`, NFC-normalised, `/`
  separators;
- ordering: the list of `(path_key, sha256_hex(file_bytes))` pairs, sorted
  by `path_key` (byte order of the UTF-8 encoding);
- serialisation: `digest = sha256(b"fixture_corpus_digest_v1\\n" +
  b"".join(f"{path_key}\\x00{file_sha256}\\n".encode("utf-8") for ... in
  sorted list)).hexdigest()`.
"""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

_DIGEST_PREFIX = b"fixture_corpus_digest_v1\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_corpus_digest(corpus_root: Path) -> str:
    """§7.1's `fixture_corpus_digest`, computed over every regular file
    found (recursively) under `corpus_root` right now. Raises
    `FileNotFoundError` if `corpus_root` itself does not exist -- an
    empty-but-existing directory is not the same failure (it would
    silently produce the empty-corpus digest, masking a caller bug that
    passed the wrong path)."""
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"corpus_root does not exist or is not a directory: {corpus_root}")
    pairs: list[tuple[str, str]] = []
    for path in corpus_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(corpus_root)
        path_key = unicodedata.normalize("NFC", rel.as_posix())
        pairs.append((path_key, _sha256_file(path)))
    pairs.sort(key=lambda pair: pair[0].encode("utf-8"))

    digest = hashlib.sha256()
    digest.update(_DIGEST_PREFIX)
    for path_key, file_sha256 in pairs:
        digest.update(f"{path_key}\x00{file_sha256}\n".encode("utf-8"))
    return digest.hexdigest()
