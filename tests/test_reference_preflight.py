"""`scripts/reference_preflight.py`'s step-5 asset-readiness check
(track2_pipeline_readiness_plan_2026-08-30.md §5.5's `probe` job, mirrored
locally)."""
from __future__ import annotations

import scripts.reference_preflight as rp


def test_all_step19_assets_ready_is_false_when_any_lock_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "FT_SCREENING_WHEEL_LOCK", tmp_path / "ft-screening-wheel.lock")
    monkeypatch.setattr(rp, "SLR_PIPELINE_WHEEL_LOCK", tmp_path / "SLR_PIPELINE_WHEEL.lock")
    monkeypatch.setattr(rp, "DOCLING_ASSETS_LOCK", tmp_path / "DOCLING_ASSETS.lock")
    report = rp.run_preflight()
    assert report.all_step19_assets_ready is False


def test_all_step19_assets_ready_is_true_only_when_all_three_present(tmp_path, monkeypatch):
    ft_lock = tmp_path / "ft-screening-wheel.lock"
    slr_lock = tmp_path / "SLR_PIPELINE_WHEEL.lock"
    docling_lock = tmp_path / "DOCLING_ASSETS.lock"
    for p in (ft_lock, slr_lock, docling_lock):
        p.write_text("x", encoding="utf-8")
    monkeypatch.setattr(rp, "FT_SCREENING_WHEEL_LOCK", ft_lock)
    monkeypatch.setattr(rp, "SLR_PIPELINE_WHEEL_LOCK", slr_lock)
    monkeypatch.setattr(rp, "DOCLING_ASSETS_LOCK", docling_lock)
    report = rp.run_preflight()
    assert report.all_step19_assets_ready is True


def test_all_step19_assets_ready_is_false_when_exactly_two_of_three_present(tmp_path, monkeypatch):
    """The AND, not an OR-across-patterns `hashFiles` false positive
    (round-14 finding 1) -- two present, one missing must still be False."""
    ft_lock = tmp_path / "ft-screening-wheel.lock"
    slr_lock = tmp_path / "SLR_PIPELINE_WHEEL.lock"
    docling_lock = tmp_path / "DOCLING_ASSETS.lock"
    ft_lock.write_text("x", encoding="utf-8")
    slr_lock.write_text("x", encoding="utf-8")
    monkeypatch.setattr(rp, "FT_SCREENING_WHEEL_LOCK", ft_lock)
    monkeypatch.setattr(rp, "SLR_PIPELINE_WHEEL_LOCK", slr_lock)
    monkeypatch.setattr(rp, "DOCLING_ASSETS_LOCK", docling_lock)
    report = rp.run_preflight()
    assert report.all_step19_assets_ready is False


def test_ft_screening_importable_reflects_real_environment():
    """This repo's own dev env (lamai) has FT_SCREENING installed
    editable -- confirms `find_spec` actually detects a real install,
    not just that the attribute exists."""
    report = rp.run_preflight()
    assert report.ft_screening_importable is True
