"""Track 2 step 5 [C4]: `scripts/provision_reference_assets.py`'s
orchestration logic (track2_pipeline_readiness_plan_2026-08-30.md §5.3) --
against a **mocked** `ft_screening.docling_artifacts`, never a real
bootstrap (real Docling model downloads are step 19's own scope, gated on
`DOCLING_REDISTRIBUTION_NOTE.md`'s not-yet-existing user authorization).
"""
from __future__ import annotations

import subprocess

import pytest

import scripts.provision_reference_assets as pra


class _FakeDoclingArtifacts:
    """A tiny stand-in for `ft_screening.docling_artifacts` -- `bootstrap`
    writes one fake Hugging-Face-style asset directory (matching
    `PINNED_REVISIONS`' single fake entry) plus the always-present
    `RapidOcr` directory `_write_model_manifest_and_licences` also expects
    (`_EXTRA_UPSTREAM_ASSETS` is a fixed, real, non-test-configurable list
    in production code). The fake HF repo gets a `README.md` whose
    frontmatter declares `license: Apache-2.0` (matching the fixture's
    `_UPSTREAM_LICENCE_INFO` below) -- real HF repos ship no standalone
    LICENSE file at all, only this frontmatter field (see the comment
    above `_EXTRA_UPSTREAM_ASSETS` in production code), so the fake mirrors
    that exactly rather than the old, since-corrected assumption. RapidOcr
    gets no local licence file, since production code now fetches its
    real one over the network (`_fetch_rapidocr_licence`, mocked by the
    `fake_docling` fixture below). `bootstrap`'s returned digest is
    computed the same way `compute_tree_digest` does, so `validate`/
    `check`'s recomputation is exercised for real too."""

    PINNED_REVISIONS = {"fake-org/fake-repo": "deadbeef0000000000000000000000000000000"}

    def bootstrap(self, output_dir, *, progress=False, force=False):
        (output_dir / "fake-org--fake-repo").mkdir(parents=True, exist_ok=True)
        (output_dir / "fake-org--fake-repo" / "fake_model.bin").write_bytes(b"fake model bytes")
        (output_dir / "fake-org--fake-repo" / "README.md").write_bytes(
            b"---\nlicense: Apache-2.0\n---\n\n# Fake Repo\n"
        )
        (output_dir / "RapidOcr").mkdir(parents=True, exist_ok=True)
        (output_dir / "RapidOcr" / "fake_rapidocr_model.bin").write_bytes(b"fake rapidocr bytes")
        return self.compute_tree_digest(output_dir)

    def validate(self, artifacts_path, *, expected_tree_digest=None):
        actual = self.compute_tree_digest(artifacts_path)
        if expected_tree_digest is not None and actual != expected_tree_digest:
            raise ValueError(f"tree digest mismatch: {actual} != {expected_tree_digest}")
        return actual

    def compute_tree_digest(self, artifacts_path):
        import hashlib
        digest = hashlib.sha256()
        for relpath, meta in sorted(pra._hash_tree(artifacts_path).items()):
            digest.update(f"{relpath}:{meta['sha256']}".encode("utf-8"))
        return digest.hexdigest()


@pytest.fixture
def fake_docling(monkeypatch, tmp_path):
    monkeypatch.setattr(pra, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pra, "CACHE_ROOT", tmp_path / "assets" / "models" / "track2-docling-cache")
    monkeypatch.setattr(pra, "MODELS_DIR", tmp_path / "assets" / "models" / "track2-docling-cache" / "models")
    monkeypatch.setattr(pra, "ASSETS_LOCK_PATH", tmp_path / "assets" / "models" / "DOCLING_ASSETS.lock")
    monkeypatch.setattr(pra, "UPSTREAM_LICENSES_DIR", tmp_path / "assets" / "models" / "upstream_licenses")
    monkeypatch.setattr(pra, "MODEL_MANIFEST_PATH", tmp_path / "assets" / "models" / "DOCLING_MODEL_MANIFEST.md")
    monkeypatch.setattr(pra, "ARCHIVE_PATH", tmp_path / "docling-assets-v1.tar")
    # Test-only licence-info table -- decoupled from production's real
    # values so a future correction to the real upstream licences never
    # breaks this fixture; "rapidocr" is required because
    # `_EXTRA_UPSTREAM_ASSETS` is a fixed, non-test-configurable list.
    monkeypatch.setattr(pra, "_UPSTREAM_LICENCE_INFO", {
        "fake-org/fake-repo": {"url": "https://example.invalid/fake-repo", "licence": "Apache-2.0"},
        "rapidocr": {"url": "https://example.invalid/rapidocr", "licence": "Apache-2.0"},
    })
    # RapidOCR's licence is now fetched over the network (its real GitHub
    # LICENSE file, not a local frontmatter field) -- offline tests must
    # never touch the network (pytest-socket blocks it anyway), so this
    # seam is mocked at the same level `_import_docling_artifacts` already
    # is, rather than mocking `requests.get` itself.
    monkeypatch.setattr(pra, "_fetch_rapidocr_licence", lambda url: b"fake rapidocr Apache-2.0 licence text\n")
    fake = _FakeDoclingArtifacts()
    monkeypatch.setattr(pra, "_import_docling_artifacts", lambda: fake)
    return fake


def test_write_lock_then_check_round_trips(fake_docling):
    pra.write_lock()
    assert pra.ASSETS_LOCK_PATH.exists()
    pra.check(skip_archive_hash=True)  # must not raise


def test_check_fails_without_lock(fake_docling):
    with pytest.raises(FileNotFoundError):
        pra.check(skip_archive_hash=True)


def test_write_lock_writes_manifest_and_licence_tree(fake_docling):
    """§5.3: `--write-lock` authors `DOCLING_MODEL_MANIFEST.md` and bundles
    each asset's real licence source into `upstream_licenses/<local_dir>/`
    itself -- neither is required to pre-exist. The HF repo's source is
    its README.md (frontmatter-declared licence); RapidOcr's is its real
    fetched LICENSE (mocked here, see `fake_docling`)."""
    pra.write_lock()
    manifest_text = pra.MODEL_MANIFEST_PATH.read_text(encoding="utf-8")
    assert "fake-org/fake-repo" in manifest_text
    assert "rapidocr" in manifest_text
    assert "Apache-2.0" in manifest_text
    fake_repo_licence = pra.UPSTREAM_LICENSES_DIR / "fake-org--fake-repo" / "README.md"
    rapidocr_licence = pra.UPSTREAM_LICENSES_DIR / "RapidOcr" / "LICENSE"
    assert fake_repo_licence.read_bytes() == b"---\nlicense: Apache-2.0\n---\n\n# Fake Repo\n"
    assert rapidocr_licence.read_bytes() == b"fake rapidocr Apache-2.0 licence text\n"


def test_write_lock_fails_closed_on_unregistered_licence_info(fake_docling, monkeypatch):
    monkeypatch.setattr(pra, "_UPSTREAM_LICENCE_INFO", {})
    with pytest.raises(ValueError, match="no licence info registered"):
        pra.write_lock()


def test_write_lock_fails_closed_on_unsupported_licence(fake_docling, monkeypatch):
    monkeypatch.setattr(pra, "_UPSTREAM_LICENCE_INFO", {
        "fake-org/fake-repo": {"url": "https://example.invalid/fake-repo", "licence": "GPL-3.0"},
        "rapidocr": {"url": "https://example.invalid/rapidocr", "licence": "Apache-2.0"},
    })
    with pytest.raises(ValueError, match="not in ALLOWED_UPSTREAM_LICENCES"):
        pra.write_lock()


def test_write_lock_fails_closed_on_missing_licence_file(fake_docling):
    """`bootstrap()` runs first (writing the fake asset dirs), so delete
    the fake repo's README.md (its only licence source) before
    `write_lock`'s own manifest/licence-bundle step runs -- exercised via
    a bootstrap override that then removes the file, since `write_lock`
    calls bootstrap itself."""
    original_bootstrap = fake_docling.bootstrap

    def _bootstrap_without_licence(output_dir, *, progress=False, force=False):
        digest = original_bootstrap(output_dir, progress=progress, force=force)
        (output_dir / "fake-org--fake-repo" / "README.md").unlink()
        return fake_docling.compute_tree_digest(output_dir)

    fake_docling.bootstrap = _bootstrap_without_licence
    with pytest.raises(FileNotFoundError, match="no README.md found"):
        pra.write_lock()


def test_write_lock_fails_closed_on_licence_field_mismatch(fake_docling):
    """The README's declared licence must agree with `_UPSTREAM_LICENCE_
    INFO`'s hardcoded label -- a genuine upstream licence change must be
    caught, not silently trusted from either source alone."""
    original_bootstrap = fake_docling.bootstrap

    def _bootstrap_with_wrong_licence(output_dir, *, progress=False, force=False):
        digest = original_bootstrap(output_dir, progress=progress, force=force)
        (output_dir / "fake-org--fake-repo" / "README.md").write_bytes(b"---\nlicense: MIT\n---\n\n# Fake Repo\n")
        return fake_docling.compute_tree_digest(output_dir)

    fake_docling.bootstrap = _bootstrap_with_wrong_licence
    with pytest.raises(ValueError, match="declares license 'MIT', expected 'Apache-2.0'"):
        pra.write_lock()


def test_readme_licence_bundle_fails_closed_on_missing_field(tmp_path):
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    (asset_dir / "README.md").write_text("# no frontmatter here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no `license:` field"):
        pra._readme_licence_bundle(asset_dir, "Apache-2.0")


def test_fetch_rapidocr_licence_rejects_empty_response(monkeypatch):
    class _FakeResponse:
        content = b""

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda url, timeout: _FakeResponse())
    with pytest.raises(ValueError, match="was empty"):
        pra._fetch_rapidocr_licence("https://example.invalid/LICENSE")


def test_check_detects_tampered_model_file(fake_docling):
    pra.write_lock()
    (pra.MODELS_DIR / "fake-org--fake-repo" / "fake_model.bin").write_bytes(b"tampered bytes")
    with pytest.raises(ValueError):
        pra.check(skip_archive_hash=True)


def test_write_archive_then_check_verifies_archive_hash(fake_docling):
    pra.write_lock()
    pra.write_archive()
    assert pra.ARCHIVE_PATH.exists()
    pra.check(skip_archive_hash=False)  # must not raise


def test_check_without_skip_requires_archive_present(fake_docling):
    """`write_lock()` builds the archive too (so the lock and the archive
    it references start out consistent) -- this test simulates the actual
    CI-cache-hit scenario `--skip-archive-hash` exists for: the lock is
    committed to the repo, but the (large, uncommitted) archive itself
    isn't present on this particular checkout."""
    pra.write_lock()
    pra.ARCHIVE_PATH.unlink()
    with pytest.raises(FileNotFoundError):
        pra.check(skip_archive_hash=False)


def test_archive_is_deterministic_across_two_writes(fake_docling):
    pra.write_lock()
    pra.write_archive()
    first = pra.ARCHIVE_PATH.read_bytes()
    pra.write_archive()
    second = pra.ARCHIVE_PATH.read_bytes()
    assert first == second


def test_archive_extracts_matching_workflow_command_no_double_nesting(fake_docling, tmp_path):
    """Round-19 finding: `.github/workflows/track2.yml`'s `acceptance` job
    extracts with `tar -xf docling-assets-v1.tar -C
    assets/models/track2-docling-cache` -- i.e. straight into CACHE_ROOT,
    not into `assets/models`. Reproduces that exact command against a
    fresh target directory standing in for a cache-miss CACHE_ROOT, and
    asserts the models land directly at `<target>/models/...` (matching
    what `MODELS_DIR` then expects), not doubly nested under
    `<target>/track2-docling-cache/models/...` (the previous bug)."""
    pra.write_lock()
    pra.write_archive()
    fresh_cache_root = tmp_path / "fresh-cache-root"
    fresh_cache_root.mkdir()
    subprocess.run(["tar", "-xf", str(pra.ARCHIVE_PATH), "-C", str(fresh_cache_root)], check=True)
    assert (fresh_cache_root / "models" / "fake-org--fake-repo" / "fake_model.bin").exists()
    assert (fresh_cache_root / "models" / "RapidOcr" / "fake_rapidocr_model.bin").exists()
    assert not (fresh_cache_root / "track2-docling-cache").exists()
