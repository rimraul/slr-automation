"""Unit F -- `slr_pipeline.continuation.cost_estimate` and the
`--estimate-only` launcher path; Unit G -- the cap/authorization gate
around the real (non-`--estimate-only`) continuation run (both in
`slr_pipeline.track3_continuation_run`)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import List

import pytest
import tiktoken

from ft_screening import CriteriaSet, CriterionSpec, ScreeningConfig as FTScreeningConfig

import slr_pipeline.continuation.cost_estimate as cost_estimate_module
import slr_pipeline.cost_cap.ledger as ledger_module
import slr_pipeline.cost_cap.proxy as proxy_module
import slr_pipeline.reference_bundle as rb
import slr_pipeline.track3_continuation_run as launcher_module
from slr_pipeline.config import ReviewConfig
from slr_pipeline.continuation.cost_estimate import (
    TOKENIZER_SAFETY_MARGIN_MULTIPLIER,
    count_tokens,
    estimate_continuation_cost,
)
from slr_pipeline.continuation.manifest import ContinuationManifest, ContinuationManifestDocument
from slr_pipeline.continuation.partition import ContinuationPartition
from slr_pipeline.extraction.batching import build_batches
from slr_pipeline.extraction.parsing import DoclingParserOptions, ParsedChunk, ParsedDocument
from slr_pipeline.extraction.schema import ExtractionFieldSpec, ExtractionSchema, FieldOrigin, FieldProvenance, ValueKind
from slr_pipeline.extraction.openai_runner import OpenAIExtractionConfig
from slr_pipeline.nodes.reporting import _state_value
from slr_pipeline.selection_ledger import _tails_by_key
from slr_pipeline.state import PRISMACounts, Record, initial_state

from .conftest import build_parent_bundle, minimal_reporting_source_snapshot


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _chunk(index: int, text: str, section: str = "Methods") -> ParsedChunk:
    return ParsedChunk(
        chunk_id=f"doc-chunk-{index:04d}", order_index=index, section=section,
        page_min=index + 1, page_max=index + 1, text=text,
    )


def _parsed_document(chunks: tuple, *, document_sha256: str = "a" * 64) -> ParsedDocument:
    return ParsedDocument(
        document_sha256=document_sha256, parser_name="docling", parser_version="2.61.2",
        parser_options=DoclingParserOptions(), chunks=chunks,
    )


def _schema(**overrides) -> ExtractionSchema:
    field = ExtractionFieldSpec(
        field_id="study_design", question="q", description="d", value_kind=ValueKind.TEXT, cardinality="single",
        provenance=FieldProvenance(origin=FieldOrigin.THEORY_DERIVED, version="1"),
    )
    defaults = dict(schema_id="s", version="1", fields=[field])
    defaults.update(overrides)
    return ExtractionSchema(**defaults)


def _model_config(**overrides) -> OpenAIExtractionConfig:
    defaults = dict(
        model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
    )
    defaults.update(overrides)
    return OpenAIExtractionConfig(**defaults)


def _criteria_set() -> CriteriaSet:
    return CriteriaSet(
        criteria_set_id="unit-f-pilot", version="1",
        criteria=[
            CriterionSpec(
                criterion_id="population", version="1",
                eligibility_requirement="Adult human participants",
                missing_evidence_policy="review", exclusion_reason_code="wrong_population",
                exclusion_reason_label="Wrong population", priority=1,
            ),
        ],
    )


def _cfg(tmp_path, **overrides) -> ReviewConfig:
    defaults = dict(
        title="Unit F cost estimate", output_dir=str(tmp_path / "outputs"),
        extraction_schema=_schema().model_dump(mode="json"),
        extraction_model_config=dict(
            model="gpt-5.6-terra", model_config_id="cfg-1", reasoning_effort="medium",
            max_completion_tokens=4096, expected_provider_model="gpt-5.6-terra",
        ),
        full_text_criteria_set=_criteria_set().model_dump(mode="json"),
    )
    defaults.update(overrides)
    return ReviewConfig(**defaults)


# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------

def test_count_tokens_matches_hand_computed_expected():
    text = "hello world, this is a small fixture prompt for Unit F."
    raw = len(tiktoken.get_encoding("o200k_base").encode(text))
    expected = math.ceil(raw * TOKENIZER_SAFETY_MARGIN_MULTIPLIER)
    assert count_tokens(text, model="gpt-5.6-terra") == expected
    # gpt-4o resolves via encoding_for_model() directly to the same encoding.
    assert count_tokens(text, model="gpt-4o") == expected


# ---------------------------------------------------------------------------
# _estimate_extraction() call count == build_batches()'s own real output
# ---------------------------------------------------------------------------

def test_extraction_call_count_matches_build_batches(tmp_path, monkeypatch):
    chunks = tuple(_chunk(i, "x" * 5_000) for i in range(3))  # 3 chunks, 5000 chars each
    parsed = _parsed_document(chunks)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: parsed)

    cfg = _cfg(tmp_path, extraction_max_batch_chars=6_000)
    expected_batches = build_batches(chunks, max_batch_chars=6_000)
    assert len(expected_batches) == 3  # one chunk per batch at this budget -- sanity on the fixture itself

    estimate = estimate_continuation_cost(
        cfg=cfg, retrieved_manifest_docs={"r1": {"dest_path": "unused", "file_sha256": "x"}},
        preserved_included_parent_records=[],
    )
    extraction_stage = next(s for s in estimate.stages if s.stage == "extraction")
    assert extraction_stage.calls == len(expected_batches) == 3
    assert extraction_stage.worst_case_usd > 0.0


def test_preserved_included_records_are_genuinely_consumed(tmp_path, monkeypatch):
    chunks = tuple(_chunk(i, "x" * 5_000) for i in range(3))
    parsed = _parsed_document(chunks)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: parsed)

    cfg = _cfg(tmp_path, extraction_max_batch_chars=6_000)

    retrieved_only = estimate_continuation_cost(
        cfg=cfg, retrieved_manifest_docs={"r1": {"dest_path": "unused", "file_sha256": "x"}},
        preserved_included_parent_records=[],
    )
    preserved_record = Record(id="r2", source="test", source_id="2", full_text_path="unused2.pdf")
    with_preserved = estimate_continuation_cost(
        cfg=cfg, retrieved_manifest_docs={"r1": {"dest_path": "unused", "file_sha256": "x"}},
        preserved_included_parent_records=[preserved_record],
    )

    extraction_only = next(s for s in retrieved_only.stages if s.stage == "extraction")
    extraction_with = next(s for s in with_preserved.stages if s.stage == "extraction")
    assert extraction_with.calls == 2 * extraction_only.calls
    assert with_preserved.grand_total_usd > retrieved_only.grand_total_usd


def test_unpriced_extraction_model_raises(tmp_path, monkeypatch):
    chunks = (_chunk(0, "hello"),)
    parsed = _parsed_document(chunks)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: parsed)

    cfg = _cfg(tmp_path, extraction_model_config=dict(
        model="not-a-priced-model", model_config_id="cfg-1", reasoning_effort="medium",
        max_completion_tokens=4096, expected_provider_model="not-a-priced-model",
    ))
    with pytest.raises(cost_estimate_module.UnpricedModelError):
        estimate_continuation_cost(
            cfg=cfg, retrieved_manifest_docs={"r1": {"dest_path": "unused", "file_sha256": "x"}},
            preserved_included_parent_records=[],
        )


def test_no_documents_is_all_empty_stages(tmp_path):
    cfg = _cfg(tmp_path)
    estimate = estimate_continuation_cost(
        cfg=cfg, retrieved_manifest_docs={}, preserved_included_parent_records=[])
    assert estimate.grand_total_usd == 0.0
    assert all(s.calls == 0 for s in estimate.stages)


def test_render_breakdown_mentions_planning_information(tmp_path):
    cfg = _cfg(tmp_path)
    estimate = estimate_continuation_cost(
        cfg=cfg, retrieved_manifest_docs={}, preserved_included_parent_records=[])
    text = estimate.render_breakdown()
    assert "CONSERVATIVE PLANNING" in text
    assert "TOTAL" in text


# ---------------------------------------------------------------------------
# --estimate-only launcher: never constructs a CostLedger/CostCapProxy
# ---------------------------------------------------------------------------

class _FakeBundle:
    """Stand-in for `Bundle` -- only `.resolve()` is exercised by the
    launcher, and only for the one locator it actually needs. Real
    `Bundle`/`resolve_parent_bundle()` correctness is already covered by
    `test_identity.py`/`test_reference_bundle.py`; stubbing it here keeps
    this test's own focus on the launcher's own wiring and on the
    ledger/proxy non-construction property."""

    def __init__(self, terminal_state_path: Path):
        self._terminal_state_path = terminal_state_path

    def resolve(self, locator: str) -> Path:
        assert locator == "slr://state/snapshot/terminal_state"
        return self._terminal_state_path


def _never_construct(name):
    def _raise(self, *a, **k):
        raise AssertionError(f"{name} must never be constructed by --estimate-only")
    return _raise


def test_estimate_only_never_constructs_ledger_or_proxy(tmp_path, monkeypatch, capsys):
    import slr_pipeline.cost_cap.ledger as ledger_module
    import slr_pipeline.cost_cap.proxy as proxy_module

    monkeypatch.setattr(ledger_module.CostLedger, "__init__", _never_construct("CostLedger"))
    monkeypatch.setattr(proxy_module.CostCapProxy, "__init__", _never_construct("CostCapProxy"))
    monkeypatch.setattr(proxy_module.CostCapProxy, "start", _never_construct("CostCapProxy.start"))

    record = Record(id="r1", source="test", source_id="1", title="Study 1")
    terminal_state_path = tmp_path / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps({"included_records": [record.to_dict()]}), encoding="utf-8")
    fake_bundle = _FakeBundle(terminal_state_path)
    monkeypatch.setattr(launcher_module, "resolve_parent_bundle", lambda bundle_id, bundles_root: fake_bundle)

    parent_snapshot = minimal_reporting_source_snapshot(full_text_records=[], ledger_records=[])
    monkeypatch.setattr(launcher_module, "from_bundle", lambda bundle: parent_snapshot)

    chunks = (_chunk(0, "worst case document body text " * 10),)
    parsed = _parsed_document(chunks)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: parsed)

    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake\n"
    (pdfs_dir / "r1.pdf").write_bytes(pdf_bytes)
    import hashlib
    doc = ContinuationManifestDocument(
        worklist_id="r1", save_as="r1.pdf",
        file_sha256=hashlib.sha256(pdf_bytes).hexdigest(), status="parsed", identity_flag="OK")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        ContinuationManifest(continuation_workspace=str(tmp_path), documents=[doc]).model_dump_json(),
        encoding="utf-8")

    cfg = _cfg(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")

    rc = launcher_module.main([
        "--spec", str(spec_path),
        "--bundles-root", str(tmp_path / "bundles-root-unused"),
        "--parent-bundle-id", "irrelevant-under-the-stub",
        "--continuation-manifest", str(manifest_path),
        "--retrieved-pdfs-dir", str(pdfs_dir),
        "--workspace", str(tmp_path / "ws"),
        "--estimate-only",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Continuation worst-case cost estimate" in out
    assert "extraction" in out


def test_estimate_only_sets_and_restores_docling_offline_env(tmp_path, monkeypatch, capsys):
    """Real-data preflight corrective pass: `main()`'s `--estimate-only`
    path wraps `estimate_continuation_cost()` (which parses real PDFs via
    Docling) in `_docling_offline_env()`. A prior, unrelated
    `HF_HUB_OFFLINE`/absent `TRANSFORMERS_OFFLINE`/absent
    `DOCLING_CACHE_DIR` environment is captured mid-call (inside the
    monkeypatched `parse_pdf()`, the real call site) and restored exactly
    afterward."""
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    record = Record(id="r1", source="test", source_id="1", title="Study 1")
    terminal_state_path = tmp_path / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps({"included_records": [record.to_dict()]}), encoding="utf-8")
    fake_bundle = _FakeBundle(terminal_state_path)
    monkeypatch.setattr(launcher_module, "resolve_parent_bundle", lambda bundle_id, bundles_root: fake_bundle)

    parent_snapshot = minimal_reporting_source_snapshot(full_text_records=[], ledger_records=[])
    monkeypatch.setattr(launcher_module, "from_bundle", lambda bundle: parent_snapshot)

    chunks = (_chunk(0, "worst case document body text " * 10),)
    parsed = _parsed_document(chunks)
    captured = {}

    def _capturing_parse_pdf(*a, **k):
        captured["docling_cache_dir"] = os.environ.get("DOCLING_CACHE_DIR")
        captured["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return parsed

    monkeypatch.setattr(cost_estimate_module, "parse_pdf", _capturing_parse_pdf)

    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake\n"
    (pdfs_dir / "r1.pdf").write_bytes(pdf_bytes)
    doc = ContinuationManifestDocument(
        worklist_id="r1", save_as="r1.pdf",
        file_sha256=hashlib.sha256(pdf_bytes).hexdigest(), status="parsed", identity_flag="OK")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        ContinuationManifest(continuation_workspace=str(tmp_path), documents=[doc]).model_dump_json(),
        encoding="utf-8")

    cfg = _cfg(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")

    rc = launcher_module.main([
        "--spec", str(spec_path),
        "--bundles-root", str(tmp_path / "bundles-root-unused"),
        "--parent-bundle-id", "irrelevant-under-the-stub",
        "--continuation-manifest", str(manifest_path),
        "--retrieved-pdfs-dir", str(pdfs_dir),
        "--workspace", str(tmp_path / "ws"),
        "--estimate-only",
    ])

    assert rc == 0
    assert captured["docling_cache_dir"] == str(launcher_module.DOCLING_CACHE_DIR.resolve())
    assert captured["hf_hub_offline"] == "1"
    assert captured["transformers_offline"] == "1"
    # Restored to the exact prior state once `main()` returns.
    assert "DOCLING_CACHE_DIR" not in os.environ
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_main_forces_output_dir_and_pdf_dir_under_workspace(tmp_path, monkeypatch, capsys):
    """Finding 4 (code-review corrective pass, post-95c9667): there is no
    independent `--pdf-dir` flag anymore -- `cfg.output_dir` is forced to
    `<workspace>/review_output`, and PDFs are copied into `cfg.
    ensure_output_dir() / "pdfs"`, the SAME expression `continuation_
    fulltext_node()` itself uses -- so a real run can never search a
    different directory than the one this launcher populated."""
    record = Record(id="r1", source="test", source_id="1", title="Study 1")
    terminal_state_path = tmp_path / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps({"included_records": [record.to_dict()]}), encoding="utf-8")
    fake_bundle = _FakeBundle(terminal_state_path)
    monkeypatch.setattr(launcher_module, "resolve_parent_bundle", lambda bundle_id, bundles_root: fake_bundle)
    parent_snapshot = minimal_reporting_source_snapshot(full_text_records=[], ledger_records=[])
    monkeypatch.setattr(launcher_module, "from_bundle", lambda bundle: parent_snapshot)

    chunks = (_chunk(0, "worst case document body text " * 10),)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: _parsed_document(chunks))

    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake\n"
    (pdfs_dir / "r1.pdf").write_bytes(pdf_bytes)
    doc = ContinuationManifestDocument(
        worklist_id="r1", save_as="r1.pdf",
        file_sha256=hashlib.sha256(pdf_bytes).hexdigest(), status="parsed", identity_flag="OK")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        ContinuationManifest(continuation_workspace=str(tmp_path), documents=[doc]).model_dump_json(),
        encoding="utf-8")

    cfg = _cfg(tmp_path, output_dir=str(tmp_path / "some-other-unrelated-output-dir"))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")

    workspace = tmp_path / "ws"
    rc = launcher_module.main([
        "--spec", str(spec_path),
        "--bundles-root", str(tmp_path / "bundles-root-unused"),
        "--parent-bundle-id", "irrelevant-under-the-stub",
        "--continuation-manifest", str(manifest_path),
        "--retrieved-pdfs-dir", str(pdfs_dir),
        "--workspace", str(workspace),
        "--estimate-only",
    ])

    assert rc == 0
    # The spec's own output_dir is discarded; PDFs land under the
    # workspace-derived directory instead of the spec's original one.
    assert not (tmp_path / "some-other-unrelated-output-dir").exists()
    copied_pdfs = list((workspace / "review_output" / "pdfs").glob("*.pdf"))
    assert len(copied_pdfs) == 1


# ---------------------------------------------------------------------------
# Unit G -- cap/authorization gate around the real (non-`--estimate-only`) run
# ---------------------------------------------------------------------------


def _minimal_launcher_argv(tmp_path):
    """Same fixture shape `test_estimate_only_never_constructs_ledger_or_proxy`
    builds, factored out so the `--cap-usd`/`--workspace` gate tests below
    can drive `main()` for real up through the point the gate fires,
    without duplicating the whole `_FakeBundle`/manifest/spec setup."""
    record = Record(id="r1", source="test", source_id="1", title="Study 1")
    terminal_state_path = tmp_path / "terminal_state.json"
    terminal_state_path.write_text(
        json.dumps({"included_records": [record.to_dict()]}), encoding="utf-8")
    fake_bundle = _FakeBundle(terminal_state_path)

    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake\n"
    (pdfs_dir / "r1.pdf").write_bytes(pdf_bytes)
    doc = ContinuationManifestDocument(
        worklist_id="r1", save_as="r1.pdf",
        file_sha256=hashlib.sha256(pdf_bytes).hexdigest(), status="parsed", identity_flag="OK")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        ContinuationManifest(continuation_workspace=str(tmp_path), documents=[doc]).model_dump_json(),
        encoding="utf-8")

    cfg = _cfg(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")

    argv = [
        "--spec", str(spec_path),
        "--bundles-root", str(tmp_path / "bundles-root-unused"),
        "--parent-bundle-id", "irrelevant-under-the-stub",
        "--continuation-manifest", str(manifest_path),
        "--retrieved-pdfs-dir", str(pdfs_dir),
    ]
    return fake_bundle, argv


def _stub_bundle_resolution(monkeypatch, fake_bundle):
    monkeypatch.setattr(launcher_module, "resolve_parent_bundle", lambda bundle_id, bundles_root: fake_bundle)
    parent_snapshot = minimal_reporting_source_snapshot(full_text_records=[], ledger_records=[])
    monkeypatch.setattr(launcher_module, "from_bundle", lambda bundle: parent_snapshot)


def test_cap_usd_required_for_real_run(tmp_path, monkeypatch, capsys):
    fake_bundle, argv = _minimal_launcher_argv(tmp_path)
    _stub_bundle_resolution(monkeypatch, fake_bundle)
    chunks = (_chunk(0, "worst case document body text " * 10),)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: _parsed_document(chunks))

    with pytest.raises(SystemExit):
        launcher_module.main(argv + ["--workspace", str(tmp_path / "ws")])

    # The breakdown is printed BEFORE the gate check fires -- a human
    # deciding --cap-usd must see it even when they omitted the flag.
    assert "Continuation worst-case cost estimate" in capsys.readouterr().out


def test_workspace_always_required(tmp_path, monkeypatch, capsys):
    """Unlike `--cap-usd`, `--workspace` is now an argparse-level required
    flag (finding 4, code-review corrective pass post-95c9667) -- even
    `--estimate-only` needs it, since `cfg.output_dir`/`pdf_dir` are
    derived from it before the estimate itself can be computed. Omitting
    it fails argparse's own parsing, before any breakdown is printed."""
    fake_bundle, argv = _minimal_launcher_argv(tmp_path)
    _stub_bundle_resolution(monkeypatch, fake_bundle)
    chunks = (_chunk(0, "worst case document body text " * 10),)
    monkeypatch.setattr(cost_estimate_module, "parse_pdf", lambda *a, **k: _parsed_document(chunks))

    with pytest.raises(SystemExit):
        launcher_module.main(argv + ["--cap-usd", "5.0"])

    assert "Continuation worst-case cost estimate" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_continuation(): fake CostLedger/CostCapProxy/graph -- proves exactly
# ONE ledger/proxy pair is constructed per run, OPENAI_BASE_URL is pointed
# at that SAME proxy instance's own base_url before graph.invoke(), and the
# proxy is stopped (and the env var restored) even if the graph raises.
# ---------------------------------------------------------------------------


class _FakeBundleWithPayloadRoot:
    def __init__(self, payload_root: Path):
        self.payload_root = payload_root


class _FakeLedger:
    instances: List["_FakeLedger"] = []

    def __init__(self, path, cap_usd):
        self.path = path
        self.cap_usd = cap_usd
        self.ensure_persisted_calls = 0
        _FakeLedger.instances.append(self)

    def ensure_persisted(self):
        self.ensure_persisted_calls += 1


class _FakeProxy:
    instances: List["_FakeProxy"] = []

    def __init__(self, ledger, call_log_path, port=0):
        self.ledger = ledger
        self.call_log_path = call_log_path
        self.started = False
        self.stopped = False
        _FakeProxy.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    @property
    def port(self):
        return 65432

    @property
    def base_url(self):
        return "http://127.0.0.1:65432/v1"


def _resolved_fixture():
    r_missing = Record(id="r-missing", source="test", source_id="missing", doi=None)
    partition = ContinuationPartition(
        preserved_included_ids=[], preserved_excluded_ids=[], retrieved_ids=[],
        not_retrieved_ids=["r-missing"],
    )
    parent_snapshot = minimal_reporting_source_snapshot(full_text_records=[], ledger_records=[])
    # `minimal_reporting_source_snapshot()`'s own `prisma` is `_state_value({})`
    # -- a plain empty dict, cheapest valid placeholder for code that never
    # reads it. `run_continuation()` DOES read `.prisma.model.model_dump()`
    # (finding 1, code-review corrective pass post-95c9667), so this fixture
    # overrides it with a genuine `PRISMACounts` instance.
    parent_snapshot = dataclasses.replace(parent_snapshot, prisma=_state_value(PRISMACounts(included=3)))
    return launcher_module._ResolvedContinuationInputs(
        included_records=[r_missing], parent_snapshot=parent_snapshot, partition=partition,
        preserved_included_parent_records=[], retrieved_manifest_docs={},
    )


def test_run_continuation_constructs_ledger_and_proxy_exactly_once_and_routes_env(tmp_path, monkeypatch):
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # Real-data preflight corrective pass: `_docling_offline_env()` sets
    # these for the duration of `run_continuation()`'s graph invocation --
    # a prior, unrelated value must be restored afterward, same as
    # `OPENAI_BASE_URL` below.
    monkeypatch.delenv("DOCLING_CACHE_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    captured = {}

    class _FakeGraphState:
        values = {}
        next = ()

    class _FakeGraph:
        def get_state(self, run_config):
            return _FakeGraphState()

        def invoke(self, state, run_config):
            captured["base_url"] = os.environ.get("OPENAI_BASE_URL")
            captured["docling_cache_dir"] = os.environ.get("DOCLING_CACHE_DIR")
            captured["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
            captured["transformers_offline"] = os.environ.get("TRANSFORMERS_OFFLINE")
            captured["state"] = state
            return {"review_run_id": run_config["configurable"]["thread_id"]}

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _FakeGraph())
    # `_write_terminal_artifacts()` needs a real `AIUseEventStore`/ledger
    # `.snapshot()`/`_build_run_mode()` chain -- out of scope for this
    # test's own focus (ledger/proxy construction + env routing, per its
    # name); covered instead by `test_publish_bundle.py` (Unit J) and the
    # real end-to-end test below.
    written = {}
    monkeypatch.setattr(launcher_module, "_write_terminal_artifacts", lambda **kw: written.update(kw))

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()
    workspace = tmp_path / "ws"

    terminal_state = launcher_module.run_continuation(
        cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=workspace,
    )
    assert written["terminal_state"] is terminal_state
    assert (workspace / "thread_id.txt").is_file()

    assert len(_FakeLedger.instances) == 1
    assert len(_FakeProxy.instances) == 1
    ledger = _FakeLedger.instances[0]
    proxy = _FakeProxy.instances[0]
    assert ledger.cap_usd == 5.0
    assert ledger.path == workspace / "cost_cap_ledger.json"
    assert proxy.ledger is ledger
    assert proxy.started and proxy.stopped
    # Finding 1 (code-review corrective pass, round 2): both files forced
    # onto disk on the success path, even for this fake (zero-real-spend)
    # ledger/proxy pair.
    assert ledger.ensure_persisted_calls == 1
    assert (workspace / "cost_cap_call_log.jsonl").is_file()
    # The ONE proxy instance's own base_url is what the graph invocation
    # actually saw via OPENAI_BASE_URL -- the mechanism every real
    # OpenAI-client-constructing call site (extraction/openai_runner.py,
    # nodes/synthesis.py, ft_screening's adjudication.py -- none pass an
    # explicit base_url=) transparently picks up.
    assert captured["base_url"] == proxy.base_url
    assert terminal_state["review_run_id"]
    # Restored to unset afterward -- this run must not leak env state.
    assert "OPENAI_BASE_URL" not in os.environ
    # `_docling_offline_env()`'s 3 vars were visible to the real graph
    # invocation, and are restored to their exact prior state afterward
    # (2 unset -> fully unset again, 1 set to "0" -> restored to "0", never
    # left at the forced "1").
    assert captured["docling_cache_dir"] == str(launcher_module.DOCLING_CACHE_DIR.resolve())
    assert captured["hf_hub_offline"] == "1"
    assert captured["transformers_offline"] == "1"
    assert "DOCLING_CACHE_DIR" not in os.environ
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    # Seeded state carries the resolved partition/records through.
    assert captured["state"]["continuation_partition"] == resolved.partition.model_dump()
    # Finding 1 (code-review corrective pass, post-95c9667): the parent's
    # real cumulative PRISMA counts are seeded, not left at
    # `initial_state()`'s own all-zeros default -- `continuation_
    # fulltext_node()` only ADDS to this dict.
    assert captured["state"]["prisma_counts"] == resolved.parent_snapshot.prisma.model.model_dump()


def test_run_continuation_stops_proxy_and_restores_env_even_if_graph_raises(tmp_path, monkeypatch):
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://previous-value.invalid/v1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    class _BoomGraphState:
        values = {}
        next = ()

    class _BoomGraph:
        def get_state(self, run_config):
            return _BoomGraphState()

        def invoke(self, state, run_config):
            raise RuntimeError("boom")

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _BoomGraph())

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()

    with pytest.raises(RuntimeError):
        launcher_module.run_continuation(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=tmp_path / "ws",
        )

    assert _FakeProxy.instances[0].stopped
    assert os.environ["OPENAI_BASE_URL"] == "http://previous-value.invalid/v1"
    # `_docling_offline_env()`'s own `finally` restores even when the graph
    # invocation itself raises -- not only on the success path.
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


# ---------------------------------------------------------------------------
# Findings 2/5 (code-review corrective pass, post-95c9667): stable
# thread_id persistence across calls against the same workspace, and
# hard-abort (never silent success) on a non-empty review queue or a graph
# that did not reach a true terminal state.
# ---------------------------------------------------------------------------


class _FakeGraphState:
    def __init__(self, *, values=None, next_nodes=()):
        self.values = values if values is not None else {}
        self.next = tuple(next_nodes)


def test_run_continuation_persists_and_reuses_thread_id_across_calls(tmp_path, monkeypatch):
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    seen_thread_ids = []

    class _FakeGraph:
        def get_state(self, run_config):
            return _FakeGraphState()

        def invoke(self, state, run_config):
            seen_thread_ids.append(run_config["configurable"]["thread_id"])
            return {"review_run_id": "r1"}

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _FakeGraph())
    monkeypatch.setattr(launcher_module, "_write_terminal_artifacts", lambda **kw: None)

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()
    workspace = tmp_path / "ws"

    launcher_module.run_continuation(
        cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=workspace)
    launcher_module.run_continuation(
        cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=workspace)

    assert len(seen_thread_ids) == 2
    # SAME thread_id both calls -- a fresh random thread_id each call would
    # make a real SqliteSaver checkpoint from a prior crash unreachable,
    # silently forcing a from-scratch (re-spending) run.
    assert seen_thread_ids[0] == seen_thread_ids[1]
    assert (workspace / "thread_id.txt").read_text(encoding="utf-8").strip() == seen_thread_ids[0]


# ---------------------------------------------------------------------------
# Finding 3 (code-review corrective pass, round 2): a resume must be refused
# unless the newly-resolved parent/config/partition/manifest match what the
# checkpoint under this workspace was originally seeded with.
# ---------------------------------------------------------------------------


def test_assert_seed_matches_checkpoint_passes_on_identical_seed():
    seed = {"config": {"a": 1}, "continuation_partition": {"x": [1]}, "parent_bundle_payload_root": "p"}
    launcher_module._assert_seed_matches_checkpoint(dict(seed), dict(seed))


def test_assert_seed_matches_checkpoint_raises_on_mismatched_partition():
    checkpointed = {"config": {"a": 1}, "continuation_partition": {"x": [1]}, "parent_bundle_payload_root": "p"}
    current = {"config": {"a": 1}, "continuation_partition": {"x": [2]}, "parent_bundle_payload_root": "p"}
    with pytest.raises(RuntimeError, match="continuation_partition"):
        launcher_module._assert_seed_matches_checkpoint(checkpointed, current)


def test_assert_seed_matches_checkpoint_raises_on_mismatched_parent():
    checkpointed = {"parent_bundle_payload_root": "parent-a"}
    current = {"parent_bundle_payload_root": "parent-b"}
    with pytest.raises(RuntimeError, match="parent_bundle_payload_root"):
        launcher_module._assert_seed_matches_checkpoint(checkpointed, current)


def _seeded_state_for(cfg, parent_bundle, resolved):
    """The exact fields `run_continuation()` itself seeds into `state`,
    reconstructed here to stand in as 'what a prior invocation checkpointed'."""
    seeded = dict(initial_state(cfg.to_dict()))
    seeded["included_records"] = [r.to_dict() for r in resolved.included_records]
    seeded["continuation_partition"] = resolved.partition.model_dump()
    seeded["preserved_included_records"] = [r.to_dict() for r in resolved.preserved_included_parent_records]
    seeded["retrieved_pdf_manifest"] = resolved.retrieved_manifest_docs
    seeded["parent_bundle_payload_root"] = str(parent_bundle.payload_root)
    seeded["preserved_decision_citations"] = launcher_module._preserved_decision_citations(
        resolved.parent_snapshot, resolved.partition)
    seeded["prisma_counts"] = resolved.parent_snapshot.prisma.model.model_dump()
    return seeded


def test_run_continuation_resumes_when_seed_matches_checkpoint(tmp_path, monkeypatch):
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(launcher_module, "_write_terminal_artifacts", lambda **kw: None)

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()
    seeded_state = _seeded_state_for(cfg, parent_bundle, resolved)

    invoked_with = []

    class _FakeGraph:
        def get_state(self, run_config):
            return _FakeGraphState(values=seeded_state)

        def invoke(self, state, run_config):
            invoked_with.append(state)
            return {"review_run_id": "r1"}

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _FakeGraph())

    launcher_module.run_continuation(
        cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=tmp_path / "ws")

    # A genuine resume: `graph.invoke(None, ...)`, never re-seeded.
    assert invoked_with == [None]


def test_run_continuation_refuses_to_resume_on_seed_mismatch(tmp_path, monkeypatch):
    """The core finding-3 scenario: a workspace reused with different
    inputs than whatever it was originally seeded with (e.g. a different
    --parent-bundle-id, or an edited --spec/--continuation-manifest) must
    never silently resume the ORIGINAL run's unrelated checkpoint."""
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    write_calls = []
    monkeypatch.setattr(launcher_module, "_write_terminal_artifacts", lambda **kw: write_calls.append(kw))

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()
    seeded_state = _seeded_state_for(cfg, parent_bundle, resolved)
    # Simulate a DIFFERENT invocation's config resolving for this same
    # (reused) workspace -- e.g. an edited --spec.
    seeded_state["config"] = {**seeded_state["config"], "title": "a different spec entirely"}

    invoked_with = []

    class _FakeGraph:
        def get_state(self, run_config):
            return _FakeGraphState(values=seeded_state)

        def invoke(self, state, run_config):
            invoked_with.append(state)
            return {"review_run_id": "r1"}

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _FakeGraph())

    workspace = tmp_path / "ws"
    with pytest.raises(RuntimeError, match="refusing to resume"):
        launcher_module.run_continuation(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=workspace)

    # `graph.invoke()` never called at all -- neither a resume nor a fresh
    # re-seed happened; nothing was executed under mismatched inputs.
    assert invoked_with == []
    assert write_calls == []
    assert not (workspace / "terminal_state.json").exists()


@pytest.mark.parametrize("queue_key", ["full_text_review_queue", "extraction_review_queue"])
def test_run_continuation_raises_on_nonempty_review_queue_and_preserves_checkpoint(
    tmp_path, monkeypatch, queue_key,
):
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    class _FakeGraph:
        def get_state(self, run_config):
            return _FakeGraphState()

        def invoke(self, state, run_config):
            return {"review_run_id": "r1", queue_key: ["r-missing"]}

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _FakeGraph())
    write_calls = []
    monkeypatch.setattr(launcher_module, "_write_terminal_artifacts", lambda **kw: write_calls.append(kw))

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()
    workspace = tmp_path / "ws"

    with pytest.raises(RuntimeError, match=queue_key):
        launcher_module.run_continuation(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=workspace)

    # Never silently reported as a success: no terminal artifacts written.
    assert write_calls == []
    assert not (workspace / "terminal_state.json").exists()
    # thread_id.txt IS preserved -- a future resume against this same
    # workspace must reuse it, not silently start a fresh (re-spending) run.
    assert (workspace / "thread_id.txt").is_file()
    # Ledger/call-log audit trail is still recorded even on this hard-abort
    # path -- real committed spend so far is genuine run evidence
    # regardless of whether the graph ever reaches a true terminal state.
    assert (workspace / "cost_cap_call_log.jsonl").is_file()


def test_run_continuation_raises_when_graph_state_next_is_nonempty(tmp_path, monkeypatch):
    _FakeLedger.instances.clear()
    _FakeProxy.instances.clear()
    monkeypatch.setattr(ledger_module, "CostLedger", _FakeLedger)
    monkeypatch.setattr(proxy_module, "CostCapProxy", _FakeProxy)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    class _FakeGraph:
        def get_state(self, run_config):
            return _FakeGraphState(next_nodes=("synthesis",))

        def invoke(self, state, run_config):
            return {"review_run_id": "r1"}

    monkeypatch.setattr(launcher_module, "build_scoping_continuation_graph", lambda checkpointer=None: _FakeGraph())
    write_calls = []
    monkeypatch.setattr(launcher_module, "_write_terminal_artifacts", lambda **kw: write_calls.append(kw))

    cfg = _cfg(tmp_path)
    parent_bundle = _FakeBundleWithPayloadRoot(tmp_path / "parent-payload")
    resolved = _resolved_fixture()
    workspace = tmp_path / "ws"

    with pytest.raises(RuntimeError, match="did not reach a true terminal state"):
        launcher_module.run_continuation(
            cfg=cfg, parent_bundle=parent_bundle, resolved=resolved, cap_usd=5.0, workspace=workspace)

    assert write_calls == []
    assert not (workspace / "terminal_state.json").exists()


# ---------------------------------------------------------------------------
# Real end-to-end: genuine `CostLedger`/`CostCapProxy` + genuine continuation
# graph, on the same zero-OpenAI-call cheap path Unit D's own tests use
# (empty partition, all `not_retrieved`) -- proves the wiring works against
# the real mechanism, not just fakes, while making zero network calls.
# ---------------------------------------------------------------------------


def test_real_run_continuation_routes_openai_base_url_through_real_proxy(tmp_path, monkeypatch):
    bundle, _parent_state = build_parent_bundle(tmp_path / "parent", monkeypatch)
    parent_snapshot = rb.from_bundle(bundle)

    # Genuine launcher-shaped input: this run's own `included_records` is
    # the parent's real full-text-eligibility candidate corpus (loaded the
    # same way `_load_included_records()` would), and the continuation
    # manifest supplies no new documents -- every record is already
    # decided in the parent, so `derive_continuation_partition()` puts all
    # of them in `preserved_included_ids`/`preserved_excluded_ids`, none in
    # `retrieved_ids`/`not_retrieved_ids`. This exercises the SAME "nothing
    # new retrieved yet" real path a report-regeneration-only continuation
    # would take, with real `preserved_decision_citations` genuinely
    # matching the parent bundle's own ledger.
    included_records = launcher_module._load_included_records(bundle)
    empty_manifest = ContinuationManifest(continuation_workspace=str(tmp_path), documents=[])
    from slr_pipeline.continuation.partition import derive_continuation_partition
    partition = derive_continuation_partition(included_records, parent_snapshot, empty_manifest)
    assert partition.retrieved_ids == [] and partition.not_retrieved_ids == []

    resolved = launcher_module._ResolvedContinuationInputs(
        included_records=included_records, parent_snapshot=parent_snapshot, partition=partition,
        preserved_included_parent_records=[], retrieved_manifest_docs={},
    )
    cfg = ReviewConfig(title="Unit G real run", output_dir=str(tmp_path / "continuation-outputs"))

    captured = {}
    import slr_pipeline.nodes.fulltext_continuation as fulltext_continuation_module
    real_node = fulltext_continuation_module.continuation_fulltext_node

    def _recording_node(state):
        captured["base_url_seen_by_fulltext_node"] = os.environ.get("OPENAI_BASE_URL")
        return real_node(state)

    monkeypatch.setattr(
        "slr_pipeline.graphs.scoping_continuation.continuation_fulltext_node", _recording_node)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    # Spy on the REAL CostLedger (still fully functional -- forwards to the
    # genuine implementation via super().__init__()) so the single real
    # instance this run constructs can be inspected afterward. A plain
    # CostLedger never persists `cost_cap_ledger.json` to disk on its own
    # until the FIRST committed spend (`track3_run.py`'s own documented
    # behavior) -- this path makes zero real OpenAI calls, so without
    # finding 1's `ledger.ensure_persisted()` fix the file would never
    # appear; asserted below that it now does regardless of spend.
    real_cost_ledger_cls = ledger_module.CostLedger
    captured_ledgers = []

    class _SpyLedger(real_cost_ledger_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured_ledgers.append(self)

    monkeypatch.setattr(ledger_module, "CostLedger", _SpyLedger)

    workspace = tmp_path / "continuation-workspace"
    terminal_state = launcher_module.run_continuation(
        cfg=cfg, parent_bundle=bundle, resolved=resolved, cap_usd=7.5, workspace=workspace,
    )

    assert terminal_state["full_text_records"] == []
    base_url_seen = captured["base_url_seen_by_fulltext_node"]
    assert base_url_seen is not None and base_url_seen.startswith("http://127.0.0.1:")
    assert "OPENAI_BASE_URL" not in os.environ  # restored after the run

    assert len(captured_ledgers) == 1
    snapshot = captured_ledgers[0].snapshot()
    assert snapshot["committed_usd"] == 0.0  # zero real OpenAI calls made on this path
    assert snapshot["cap_usd"] == 7.5

    # Finding 1 (code-review corrective pass, round 2): both files exist on
    # disk even though this run made zero real OpenAI calls -- required for
    # a report-regeneration-only continuation to ever be publishable via
    # `publish_continuation_bundle()` -> `finalize_bundle(mode_profile=
    # "track3_live")` -> `reference_bundle.py`'s run-evidence staging.
    assert (workspace / "cost_cap_ledger.json").is_file()
    persisted_ledger = json.loads((workspace / "cost_cap_ledger.json").read_text(encoding="utf-8"))
    assert persisted_ledger["total_committed_usd"] == 0.0
    assert (workspace / "cost_cap_call_log.jsonl").is_file()
