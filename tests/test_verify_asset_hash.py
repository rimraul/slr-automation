"""`scripts/verify_asset_hash.py` -- the one shared, single-purpose
asset-hash re-verification step (track2_pipeline_readiness_plan_2026-08-
30.md §5.2 item 5 / §5.3 / §5.5)."""
from __future__ import annotations

import pytest

from scripts.verify_asset_hash import sha256_of, verify_asset_hash


def _write_lock(path, **fields):
    path.write_text("\n".join(f"{k} = {v!r}" for k, v in fields.items()) + "\n", encoding="utf-8")


def test_verify_asset_hash_passes_on_matching_hash(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"some bytes")
    lock = tmp_path / "asset.lock"
    _write_lock(lock, filename="asset.bin", sha256=sha256_of(asset))
    verify_asset_hash(asset, lock, field="sha256")  # must not raise


def test_verify_asset_hash_supports_a_named_non_default_field(tmp_path):
    """`DOCLING_ASSETS.lock` verifies `archive_sha256`, not `sha256`."""
    asset = tmp_path / "docling-assets-v1.tar"
    asset.write_bytes(b"tar bytes")
    lock = tmp_path / "DOCLING_ASSETS.lock"
    _write_lock(lock, archive_sha256=sha256_of(asset))
    verify_asset_hash(asset, lock, field="archive_sha256")  # must not raise


def test_verify_asset_hash_raises_on_mismatch(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"some bytes")
    lock = tmp_path / "asset.lock"
    _write_lock(lock, sha256="0" * 64)
    with pytest.raises(ValueError):
        verify_asset_hash(asset, lock, field="sha256")


def test_verify_asset_hash_raises_on_missing_file(tmp_path):
    lock = tmp_path / "asset.lock"
    _write_lock(lock, sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        verify_asset_hash(tmp_path / "does-not-exist.bin", lock, field="sha256")


def test_verify_asset_hash_raises_on_missing_lock_field(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"some bytes")
    lock = tmp_path / "asset.lock"
    _write_lock(lock, filename="asset.bin")
    with pytest.raises(KeyError):
        verify_asset_hash(asset, lock, field="sha256")


def test_verify_asset_hash_raises_on_missing_lock_file(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"some bytes")
    with pytest.raises(FileNotFoundError):
        verify_asset_hash(asset, tmp_path / "does-not-exist.lock", field="sha256")
