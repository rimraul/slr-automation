"""`scripts.author_prisma_scr_matrix` -- Track 2 Step 12A sub-step (c):
`tests/reference_run/fixtures/prisma_scr_matrix.json` (plan §9's frozen
22-row PRISMA-ScR item matrix)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.author_prisma_scr_matrix import MATRIX_PATH, check_matrix
from slr_pipeline.prisma_scr import ITEMS as _ITEMS
from slr_pipeline.prisma_scr import PRISMA_SCR_MATRIX_VERSION, PRISMA_SCR_VERSION
from slr_pipeline.prisma_scr import matrix_text as _matrix_text

_LOCK = Path(__file__).parent.parent / "track2" / "reference_hashes.lock"


def _locked_hashes() -> dict:
    namespace: dict = {}
    exec(_LOCK.read_text(encoding="utf-8"), {}, namespace)
    return namespace["targets"]


def test_matrix_has_22_items_in_order():
    assert len(_ITEMS) == 22
    assert [row["item_id"] for row in _ITEMS] == list(range(1, 23))


def test_essential_optional_split_is_exactly_all_except_12_and_16():
    essential = {row["item_id"] for row in _ITEMS if row["tier"] == "ess"}
    optional = {row["item_id"] for row in _ITEMS if row["tier"] == "opt"}
    assert essential == set(range(1, 23)) - {12, 16}
    assert optional == {12, 16}


def test_item_ids_are_unique_and_tier_is_ess_or_opt():
    ids = [row["item_id"] for row in _ITEMS]
    assert len(ids) == len(set(ids))
    assert {row["tier"] for row in _ITEMS} == {"ess", "opt"}


def test_matrix_on_disk_matches_authored_content():
    check_matrix()  # raises on mismatch


def test_matrix_json_matches_authored_bytes_exactly():
    on_disk = MATRIX_PATH.read_text(encoding="utf-8")
    assert on_disk == _matrix_text()


def test_matrix_payload_shape():
    payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    assert payload["prisma_scr_matrix_version"] == PRISMA_SCR_MATRIX_VERSION
    assert payload["prisma_scr_version"] == PRISMA_SCR_VERSION
    assert len(payload["items"]) == 22


def test_hash_target_registered_and_locked():
    from scripts.hash_reference_inputs import TARGET_REGISTRY, sha256_of

    assert "prisma_scr_matrix_sha256" in TARGET_REGISTRY
    assert TARGET_REGISTRY["prisma_scr_matrix_sha256"]() == MATRIX_PATH
    locked = _locked_hashes()
    assert locked["prisma_scr_matrix_sha256"]["sha256"] == sha256_of(MATRIX_PATH)
