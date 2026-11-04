"""`scripts.author_fixture_corpus` -- Track 2 Step 12A sub-step (e):
registers `corpus_oracle_pdf_sha256` as a `hash_reference_inputs.py`
target (whole-file sha256 of `CORPUS_ORACLE.json`, computed once
`pdf_sha256[*]` and `total_replay_calls` are both in place -- sub-step
(a), see §0.4 / §13 step 12(e)) and locks it. Sub-step (a)'s own content
(native fixtures, PDFs, LICENSE.md, etc.) is covered by
`tests/reference_run/test_fixture_pdf_content.py`; this file covers only
the hash-target mechanism this sub-step adds."""
from __future__ import annotations

from pathlib import Path

import scripts.author_fixture_corpus as afc

_LOCK = Path(__file__).parent.parent / "track2" / "reference_hashes.lock"


def _locked_hashes() -> dict:
    namespace: dict = {}
    exec(_LOCK.read_text(encoding="utf-8"), {}, namespace)
    return namespace["targets"]


def test_corpus_oracle_pdf_sha256_target_registered_and_locked():
    from scripts.hash_reference_inputs import TARGET_REGISTRY, sha256_of

    oracle_path = afc.CORPUS_ROOT / "CORPUS_ORACLE.json"
    assert "corpus_oracle_pdf_sha256" in TARGET_REGISTRY
    assert TARGET_REGISTRY["corpus_oracle_pdf_sha256"]() == oracle_path
    locked = _locked_hashes()
    assert locked["corpus_oracle_pdf_sha256"]["sha256"] == sha256_of(oracle_path)


def test_fixture_corpus_digest_target_registered_and_locked():
    from slr_pipeline.fixture_corpus_digest import compute_corpus_digest
    from scripts.hash_reference_inputs import TARGET_REGISTRY

    assert "fixture_corpus_digest" in TARGET_REGISTRY
    live = TARGET_REGISTRY["fixture_corpus_digest"]()
    assert isinstance(live, str) and len(live) == 64
    assert live == compute_corpus_digest(afc.CORPUS_ROOT)
    locked = _locked_hashes()
    assert locked["fixture_corpus_digest"]["sha256"] == live


def test_corpus_oracle_is_content_final_pdf_sha256_and_total_replay_calls_present():
    """§13 step 12(e)'s own precondition: by the time this target is
    hashed, `CORPUS_ORACLE.json` must already carry real `pdf_sha256[*]`
    entries for every sought PDF and a `total_replay_calls` count --
    both written by sub-step (a), neither dependent on a live run."""
    import json

    from slr_pipeline import reference_fixture_answer_key as ak

    oracle = json.loads((afc.CORPUS_ROOT / "CORPUS_ORACLE.json").read_text(encoding="utf-8"))
    assert set(oracle["pdf_sha256"]) == set(ak.FULL_TEXT_SOUGHT)
    assert all(len(v) == 64 for v in oracle["pdf_sha256"].values())
    assert isinstance(oracle["total_replay_calls"], int) and oracle["total_replay_calls"] > 0
