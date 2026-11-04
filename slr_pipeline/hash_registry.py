"""Generic hash-target registry -- `register_target`/`TARGET_REGISTRY`/
`sha256_of`.

Moved here from `scripts/hash_reference_inputs.py` (Track-2 packaging-
boundary fix, CI run 34155102694): `slr_pipeline` runtime modules that
produce a reference-run fixture (e.g. `reference_charting_adapter.py`'s
extraction schema, `reference_spec.py`'s FT_SCREENING config snapshot) need
to self-register a hash target for it, but `scripts/hash_reference_inputs.py`
is a repository-only module, never packaged into the `slr_pipeline` wheel --
`from scripts.hash_reference_inputs import register_target` at
`slr_pipeline` import time raised `ModuleNotFoundError` under a genuine
site-packages install. This module holds only the pluggable, pure
mechanism (no repo-relative paths, no lock file); `scripts/hash_reference_
inputs.py` re-exports these three names and owns everything repo-specific
(the lock file, discovery, the CLI). Concrete fixture-path *registrations*
for `slr_pipeline`-produced targets now live in `scripts/register_
reference_hash_targets.py`, a repository-only script that legitimately
knows the checkout layout -- not inside `slr_pipeline` itself."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

# name -> zero-arg callable returning either the Path whose bytes this target
# hashes, or an already-computed hex sha256 string (for a target whose own
# algorithm isn't "hash one file's bytes", e.g. a multi-file digest).
TARGET_REGISTRY: dict[str, Callable[[], "Path | str"]] = {}


def register_target(name: str):
    """Decorator: `@register_target("protocol_sha256")` on a zero-arg
    function returning either the `Path` to hash (the common case), or a
    pre-computed 64-char hex sha256 `str`. Registered by each target's
    owning module, next to the code that produces the file being hashed
    (or, for a wheel-packaged producer, by the repo-only script that owns
    that producer's on-disk fixture path)."""
    def _decorator(generator: Callable[[], "Path | str"]) -> Callable[[], "Path | str"]:
        if name in TARGET_REGISTRY:
            raise ValueError(f"target {name!r} already registered")
        TARGET_REGISTRY[name] = generator
        return generator
    return _decorator


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
