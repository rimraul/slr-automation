"""Unit H (squishy-petting-starlight.md, codex point 8): the continuation
launcher pins `cfg.synthesis` explicitly, regardless of whatever the JSON
spec says. All offline -- no network, no OpenAI client, no ledger/proxy."""
from __future__ import annotations

import pytest

import slr_pipeline.track3_run as track3_run_module
from slr_pipeline.config import ReviewConfig, SynthesisConfig
from slr_pipeline.track3_continuation_run import _pin_continuation_synthesis_config


def test_pin_continuation_synthesis_config_pins_exact_values():
    cfg = ReviewConfig()
    pinned = _pin_continuation_synthesis_config(cfg)

    assert pinned.synthesis.run_bertopic is False
    assert pinned.synthesis.generate_evidence_map is True
    assert pinned.synthesis.generate_claim_backed_results is False
    assert pinned.synthesis.generate_legacy_imrad_narrative is False
    assert pinned.synthesis.refine_batch_size == 5


def test_pin_continuation_synthesis_config_overrides_spec_supplied_values():
    """A spec file that requests the opposite of every pinned value is
    still forced to the pinned values -- the continuation never trusts
    the spec's own `synthesis` block."""
    cfg = ReviewConfig.from_dict({
        "synthesis": {
            "run_bertopic": True,
            "generate_evidence_map": False,
            "generate_claim_backed_results": True,
            "generate_legacy_imrad_narrative": True,
            "refine_batch_size": 3,
        }
    })
    pinned = _pin_continuation_synthesis_config(cfg)

    assert pinned.synthesis.run_bertopic is False
    assert pinned.synthesis.generate_evidence_map is True
    assert pinned.synthesis.generate_claim_backed_results is False
    assert pinned.synthesis.generate_legacy_imrad_narrative is False
    assert pinned.synthesis.refine_batch_size == 5


def test_pin_continuation_synthesis_config_leaves_other_fields_untouched():
    cfg = ReviewConfig.from_dict({"title": "My Continuation Review", "research_questions": ["RQ1"]})
    pinned = _pin_continuation_synthesis_config(cfg)

    assert pinned.title == "My Continuation Review"
    assert pinned.research_questions == ["RQ1"]


def test_pinned_generate_claim_backed_results_false_excludes_imrad_results_target():
    """Mirrors the real formula `nodes/synthesis.py:1741-1743` computes
    (`requested_targets = list(canonical_research_questions) + (["imrad:
    results"] if cfg.synthesis.generate_claim_backed_results else [])`),
    generically proven for both flag values by
    `tests/nodes/test_synthesis_node.py::test_synthesis_node_imrad_flag_
    combinations`. This test only confirms the CONTINUATION's own pinned
    value (`False`) feeds that already-proven formula to the expected
    result for a fixture set of research questions."""
    canonical_research_questions = ["What designs were used?", "What outcomes were reported?"]
    cfg = ReviewConfig.from_dict({"research_questions": canonical_research_questions})
    pinned = _pin_continuation_synthesis_config(cfg)

    requested_targets = list(canonical_research_questions) + (
        ["imrad:results"] if pinned.synthesis.generate_claim_backed_results else []
    )
    assert requested_targets == canonical_research_questions
    assert "imrad:results" not in requested_targets


def test_assert_bertopic_corpus_size_or_disabled_raises_below_threshold():
    cfg = ReviewConfig(synthesis=SynthesisConfig(run_bertopic=True))
    with pytest.raises(track3_run_module.Track3ConfigError):
        track3_run_module.assert_bertopic_corpus_size_or_disabled(
            cfg, track3_run_module.BERTOPIC_MIN_CORPUS_SIZE - 1)


def test_assert_bertopic_corpus_size_or_disabled_passes_at_or_above_threshold():
    cfg = ReviewConfig(synthesis=SynthesisConfig(run_bertopic=True))
    track3_run_module.assert_bertopic_corpus_size_or_disabled(cfg, track3_run_module.BERTOPIC_MIN_CORPUS_SIZE)


def test_assert_bertopic_corpus_size_or_disabled_passes_when_disabled_regardless_of_corpus_size():
    cfg = ReviewConfig(synthesis=SynthesisConfig(run_bertopic=False))
    track3_run_module.assert_bertopic_corpus_size_or_disabled(cfg, 0)


def test_main_pins_synthesis_config_before_estimate(monkeypatch, tmp_path, capsys):
    """End-to-end through `main(...--estimate-only)`: a spec requesting
    `run_bertopic=True` never reaches `estimate_continuation_cost()` with
    that value -- the pin fires first. Reuses the same `_FakeBundle`/
    manifest/spec fixture shape `test_cost_estimate.py`'s own Unit G gate
    tests use (real `Bundle`/`resolve_parent_bundle()` correctness is
    already covered by `test_identity.py`/`test_reference_bundle.py`)."""
    import json

    import slr_pipeline.continuation.cost_estimate as cost_estimate_module
    import slr_pipeline.track3_continuation_run as launcher_module

    from .test_cost_estimate import _chunk, _cfg, _minimal_launcher_argv, _parsed_document, _stub_bundle_resolution

    fake_bundle, argv = _minimal_launcher_argv(tmp_path)
    _stub_bundle_resolution(monkeypatch, fake_bundle)
    chunks = (_chunk(0, "worst case document body text " * 10),)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: _parsed_document(chunks))

    cfg = _cfg(tmp_path, synthesis=SynthesisConfig(run_bertopic=True))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    argv = ["--spec", str(spec_path)] + argv[2:]

    captured_cfgs = []
    real_estimate = launcher_module.estimate_continuation_cost

    def _capturing_estimate(*, cfg, **kwargs):
        captured_cfgs.append(cfg)
        return real_estimate(cfg=cfg, **kwargs)

    monkeypatch.setattr(launcher_module, "estimate_continuation_cost", _capturing_estimate)

    exit_code = launcher_module.main(argv + ["--workspace", str(tmp_path / "ws"), "--estimate-only"])

    assert exit_code == 0
    assert len(captured_cfgs) == 1
    assert captured_cfgs[0].synthesis.run_bertopic is False
    assert "Continuation worst-case cost estimate" in capsys.readouterr().out
