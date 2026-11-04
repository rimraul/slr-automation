"""Real-data preflight corrective pass: `track3_continuation_run.py`'s
`_docling_offline_env()` -- validates the shared Docling asset bundle,
forces offline model resolution for the duration of the `with` block, and
restores every touched environment variable to its exact prior state
afterward (including fully unsetting one that was previously absent), even
when the block itself raises. Entirely offline -- no network, no OpenAI
client, no ledger/proxy; `assert_docling_artifacts_valid()` itself is
monkeypatched in every test here (never the real, slow, bundle-hashing
implementation)."""
from __future__ import annotations

import os

import pytest

import slr_pipeline.track3_continuation_run as launcher_module
import slr_pipeline.track3_run as track3_run_module


def test_docling_offline_env_sets_expected_vars(monkeypatch):
    monkeypatch.setattr(launcher_module, "assert_docling_artifacts_valid", lambda models_dir: None)
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with launcher_module._docling_offline_env():
        assert os.environ["DOCLING_CACHE_DIR"] == str(launcher_module.DOCLING_CACHE_DIR.resolve())
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_docling_offline_env_restores_previously_set_values(monkeypatch):
    monkeypatch.setattr(launcher_module, "assert_docling_artifacts_valid", lambda models_dir: None)
    monkeypatch.setenv("DOCLING_CACHE_DIR", "/some/previous/cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    with launcher_module._docling_offline_env():
        pass

    assert os.environ["DOCLING_CACHE_DIR"] == "/some/previous/cache"
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_docling_offline_env_fully_unsets_previously_absent_vars(monkeypatch):
    monkeypatch.setattr(launcher_module, "assert_docling_artifacts_valid", lambda models_dir: None)
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with launcher_module._docling_offline_env():
        pass

    assert "DOCLING_CACHE_DIR" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_docling_offline_env_restores_even_when_block_raises(monkeypatch):
    monkeypatch.setattr(launcher_module, "assert_docling_artifacts_valid", lambda models_dir: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with pytest.raises(RuntimeError):
        with launcher_module._docling_offline_env():
            raise RuntimeError("boom")

    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_docling_offline_env_validates_bundle_before_mutating_environment(monkeypatch):
    """Validation failure propagates BEFORE any environment variable is
    touched -- a broken/missing bundle must never leave the process
    pinned to offline mode."""
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def _raise(models_dir):
        raise track3_run_module.Track3ConfigError("fake validation failure")

    monkeypatch.setattr(launcher_module, "assert_docling_artifacts_valid", _raise)

    with pytest.raises(track3_run_module.Track3ConfigError):
        with launcher_module._docling_offline_env():
            pass

    assert "DOCLING_CACHE_DIR" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_docling_offline_env_validates_models_subdirectory_of_cache_dir(monkeypatch):
    captured = {}

    def _capture(models_dir):
        captured["models_dir"] = models_dir

    monkeypatch.setattr(launcher_module, "assert_docling_artifacts_valid", _capture)

    with launcher_module._docling_offline_env():
        pass

    assert captured["models_dir"] == launcher_module.DOCLING_CACHE_DIR / "models"
