"""`scripts.author_reference_spec` -- Track 2 Step 12A sub-step (g)/(h):
assembles the real `ReferenceRunSpec` (every field genuine, no placeholder)
and writes `reference_spec.json`, then registers `reference_spec_sha256` as
a `hash_reference_inputs.py` target (locked by sub-step (h)'s own
`--write --target reference_spec_sha256` invocation). See §0.4 / §13 step
12(g)/(h). `ReferenceRunSpec`'s own field/Literal validation is
`tests/test_reference_spec.py`'s job; this file covers only this script's
assembly (real hash wiring, request_snapshot_sha256 wiring) and the
hash-target mechanism sub-step (h) adds."""
from __future__ import annotations

import json
from pathlib import Path

import scripts.author_reference_spec as ars
from slr_pipeline.reference_spec import ReferenceRunSpec

_LOCK = Path(__file__).parent.parent / "track2" / "reference_hashes.lock"


def _locked_hashes() -> dict:
    namespace: dict = {}
    exec(_LOCK.read_text(encoding="utf-8"), {}, namespace)
    return namespace["targets"]


def test_build_reference_spec_embeds_real_locked_hashes():
    locked = _locked_hashes()
    spec = ars.build_reference_spec()
    assert isinstance(spec, ReferenceRunSpec)
    for field in ars._LOCKED_HASH_FIELDS:
        assert getattr(spec, field) == locked[field]["sha256"]


def test_build_reference_spec_embeds_real_request_snapshot_hashes():
    on_disk = json.loads(ars.REQUEST_SNAPSHOTS_LOCK_PATH.read_text(encoding="utf-8"))
    spec = ars.build_reference_spec()
    assert spec.request_snapshot_sha256 == on_disk
    # No placeholder leaked through -- every request-snapshot digest is a
    # real, distinct 64-char hex string, not author_reference_cassette.py's
    # generation-time `_PLACEHOLDER_HEX64` ("0" * 64).
    assert "0" * 64 not in spec.request_snapshot_sha256.values()
    assert spec.fixture_corpus_digest != "0" * 64


def test_reference_spec_json_on_disk_round_trips_to_the_same_spec():
    on_disk = json.loads(ars.REFERENCE_SPEC_PATH.read_text(encoding="utf-8"))
    rebuilt = ReferenceRunSpec(**on_disk)
    assert rebuilt == ars.build_reference_spec()


def test_reference_spec_sha256_target_registered_and_locked():
    from scripts.hash_reference_inputs import TARGET_REGISTRY, sha256_of

    assert "reference_spec_sha256" in TARGET_REGISTRY
    assert TARGET_REGISTRY["reference_spec_sha256"]() == ars.REFERENCE_SPEC_PATH
    locked = _locked_hashes()
    assert locked["reference_spec_sha256"]["sha256"] == sha256_of(ars.REFERENCE_SPEC_PATH)
