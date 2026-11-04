"""Safety and report-contract tests for the paid calibration runner.

Every provider path is mocked. These tests must never read credentials,
contact OpenAI, or invoke Docling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_scoping_charting_calibration as calibration


def _pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"fixture")
    return path


def _preflight(batch_count: int = 2, *, document_sha256: str = "pdf-sha"):
    parsed = SimpleNamespace(document_sha256=document_sha256, chunks=tuple(range(batch_count)))
    batches = [SimpleNamespace(batch_index=i) for i in range(batch_count)]
    return parsed, batches


def _model_config():
    return SimpleNamespace(
        model="gpt-5.6-terra", reasoning_effort="medium", max_completion_tokens=4096,
    )


def test_dry_run_never_executes_provider_or_creates_output(monkeypatch, tmp_path, capsys):
    pdf = _pdf(tmp_path)
    output_dir = tmp_path / "dry-output"
    parsed, batches = _preflight()

    monkeypatch.setattr(calibration, "dry_run_preflight", lambda *a, **k: (parsed, batches))
    monkeypatch.setattr(calibration, "scoping_chart_schema", lambda: object())
    monkeypatch.setattr(calibration, "scoping_extraction_model_config", _model_config)
    monkeypatch.setattr(
        calibration, "execute_calibration",
        lambda **kwargs: pytest.fail("dry run must never enter the paid execution function"),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["run_scoping_charting_calibration.py", "--pdf", str(pdf), "--output-dir", str(output_dir)],
    )

    calibration.main()

    assert not output_dir.exists()
    assert "No API calls made" in capsys.readouterr().out


def test_populated_output_is_refused_before_preflight(monkeypatch, tmp_path):
    pdf = _pdf(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "old-artifact.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        calibration, "dry_run_preflight",
        lambda *a, **k: pytest.fail("populated output must be rejected before parsing"),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_scoping_charting_calibration.py", "--pdf", str(pdf), "--output-dir", str(output_dir)],
    )

    with pytest.raises(SystemExit, match="already exists and is not empty"):
        calibration.main()


def test_execute_without_max_calls_is_refused_before_preflight(monkeypatch, tmp_path):
    pdf = _pdf(tmp_path)
    monkeypatch.setattr(
        calibration, "dry_run_preflight",
        lambda *a, **k: pytest.fail("missing authorization must be rejected before parsing"),
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "run_scoping_charting_calibration.py", "--pdf", str(pdf),
            "--output-dir", str(tmp_path / "out"), "--execute",
        ],
    )

    with pytest.raises(SystemExit, match="--execute requires --max-calls"):
        calibration.main()


def test_missing_credentials_is_refused_without_provider_call_or_output(monkeypatch, tmp_path):
    parsed, batches = _preflight()
    output_dir = tmp_path / "missing-key"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        calibration, "run_extraction",
        lambda **kwargs: pytest.fail("missing credentials must prevent paid execution"),
    )

    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        calibration.execute_calibration(
            pdf_path=tmp_path / "paper.pdf", output_dir=output_dir, document_id="d1",
            max_batch_chars=12_000, max_calls=2, schema=object(), model_config=_model_config(),
            preflight_parsed=parsed, preflight_batches=batches,
        )

    assert not output_dir.exists()


def test_authorized_call_limit_is_checked_before_credentials_or_provider(monkeypatch, tmp_path):
    parsed, batches = _preflight(batch_count=3)
    output_dir = tmp_path / "over-cap"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        calibration, "run_extraction",
        lambda **kwargs: pytest.fail("over-cap run must never reach the extraction engine"),
    )

    with pytest.raises(SystemExit, match="EXCEEDS the authorized max_calls=2"):
        calibration.execute_calibration(
            pdf_path=tmp_path / "paper.pdf", output_dir=output_dir, document_id="d1",
            max_batch_chars=12_000, max_calls=2, schema=object(), model_config=_model_config(),
            preflight_parsed=parsed, preflight_batches=batches,
        )

    assert not output_dir.exists()


class _FakeTelemetrySummary:
    total_prompt_tokens = 20
    total_completion_tokens = 10
    total_tokens = 30
    distinct_provider_models = ("gpt-5.6-terra",)
    provider_model_consistent = True
    provider_mismatch_count = 0

    def model_dump(self, *, mode: str):
        assert mode == "json"
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "distinct_provider_models": list(self.distinct_provider_models),
            "provider_model_consistent": self.provider_model_consistent,
            "provider_mismatch_count": self.provider_mismatch_count,
        }


def _fake_bundle(*, expected_batches: int, pdf_sha256: str = "pdf-sha", complete: bool = True):
    evidence = SimpleNamespace(chunk_id="chunk-1", excerpt="quoted evidence", page="1", section="Results")
    value = SimpleNamespace(value="finding", evidence=(evidence,))
    field = SimpleNamespace(
        field_id="key_findings", status=SimpleNamespace(value="extracted"),
        requires_human_review=False, values=(value,), provisional_values=(), review_evidence=(),
    )
    batch_executions = tuple(
        SimpleNamespace(
            batch_index=i, status="completed", error=None, provider_model="gpt-5.6-terra",
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        for i in range(expected_batches)
    )
    result = SimpleNamespace(fields=(field,), is_complete=complete, batch_executions=batch_executions)
    manifest = SimpleNamespace(
        extraction_schema_id="scoping-charting-v1", extraction_schema_version="1",
        model="gpt-5.6-terra", model_config_id="preset-v1", reasoning_effort="medium",
        max_completion_tokens=4096, parser_name="docling", parser_version="2.61.2",
        telemetry_summary=_FakeTelemetrySummary(),
    )
    checkpoint = SimpleNamespace(pdf_sha256=pdf_sha256, result=result)
    return SimpleNamespace(
        manifest=manifest, checkpoints=(checkpoint,), results=(result,), verify=lambda: None,
    )


def _install_fake_paid_path(monkeypatch, *, bundle, actual_calls: int):
    provider_calls = []

    def fake_provider_call(**kwargs):
        provider_calls.append(kwargs)
        return object(), SimpleNamespace(
            provider_model="gpt-5.6-terra", prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )

    def fake_run_extraction(**kwargs):
        for _ in range(actual_calls):
            kwargs["call_batch_fn"]([object()])
        return "run-1", []

    class FakeStore:
        def __init__(self, *, output_dir, run_id):
            self.output_dir = Path(output_dir)
            self.run_id = run_id
            self.checkpoints_dir = self.output_dir / "runs" / run_id / "checkpoints"

        def manifest_path(self):
            return self.output_dir / "runs" / self.run_id / "manifest.json"

    class FakeVerifiedRunBundle:
        @classmethod
        def from_store(cls, store):
            return bundle

    monkeypatch.setattr(calibration, "run_openai_batch_extraction", fake_provider_call)
    monkeypatch.setattr(calibration, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(calibration, "ExtractionCheckpointStore", FakeStore)
    monkeypatch.setattr(calibration, "VerifiedRunBundle", FakeVerifiedRunBundle)
    return provider_calls


def test_exact_call_count_mismatch_is_reported_then_raised(monkeypatch, tmp_path):
    parsed, batches = _preflight(batch_count=2)
    output_dir = tmp_path / "mismatch"
    bundle = _fake_bundle(expected_batches=1)
    provider_calls = _install_fake_paid_path(monkeypatch, bundle=bundle, actual_calls=1)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")

    with pytest.raises(RuntimeError, match="expected exactly 2 calls"):
        calibration.execute_calibration(
            pdf_path=tmp_path / "paper.pdf", output_dir=output_dir, document_id="d1",
            max_batch_chars=12_000, max_calls=2, schema=object(), model_config=_model_config(),
            preflight_parsed=parsed, preflight_batches=batches,
        )

    report = json.loads((output_dir / "calibration_report.json").read_text(encoding="utf-8"))
    assert len(provider_calls) == 1
    assert report["expected_calls"] == 2
    assert report["calls_made"] == 1
    assert report["call_count_matches_expected"] is False


def test_success_report_has_audit_grade_structure(monkeypatch, tmp_path):
    parsed, batches = _preflight(batch_count=2)
    pdf = _pdf(tmp_path)
    output_dir = tmp_path / "success"
    bundle = _fake_bundle(expected_batches=2)
    provider_calls = _install_fake_paid_path(monkeypatch, bundle=bundle, actual_calls=2)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")

    report = calibration.execute_calibration(
        pdf_path=pdf, output_dir=output_dir, document_id="d1",
        max_batch_chars=12_000, max_calls=2, schema=object(), model_config=_model_config(),
        preflight_parsed=parsed, preflight_batches=batches,
    )

    persisted = json.loads((output_dir / "calibration_report.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert len(provider_calls) == 2
    assert report["calibration_status"] == "complete"
    assert report["call_count_matches_expected"] is True
    assert report["pdf_sha256_match"] is True
    assert report["usage_complete"] is True
    assert report["cost_status"] == "exact_from_complete_provider_telemetry"
    assert report["schema_id"] == "scoping-charting-v1"
    assert report["telemetry_summary"]["provider_model_consistent"] is True
    assert report["per_field_summary"]["key_findings"]["n_values"] == 1
    assert report["total_evidence_spans"] == 1
    assert set(report["artifact_paths_relative_to_repo_root"]) == {"manifest", "checkpoints_dir"}
