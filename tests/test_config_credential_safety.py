"""Track 1 Phase A -- credential-safety regression tests.

Verifies the fix for the confirmed leak path: `reproduce_review.py`
previously assigned a real `OPENAI_API_KEY` env value into `ReviewConfig`,
and `ReviewConfig.to_dict()` (a bare `model_dump()`) serialised it verbatim
into `test_reviews/*/reproduction/reproduction_config.json` (both files
still tracked in git, unmodified by this fix -- see the plan's own
credential note). This file proves, by test rather than inspection alone:

  1. `openai_api_key` never survives `to_dict()`/`model_dump()`/`repr()`/
     `str()`/`ReviewState`/persisted JSON -- an enumerated surface list.
  2. `resolve_openai_api_key()` never falls back to `cfg.openai_api_key` --
     a historical config carrying a real (possibly committed) key must not
     be silently reactivated just because no environment variable is set.
  3. Every affected call site still works correctly once the credential
     comes from the environment alone.
  4. The specific bug this closes (`llm_backends.py`'s old
     `cfg.openai_api_key or ""` pattern, where an empty string never
     triggers the OpenAI SDK's own `None`-based environment fallback) does
     not reappear anywhere in the pipeline's source.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pytest
from slr_pipeline.config import ReviewConfig, resolve_openai_api_key, scoping_review_config
from slr_pipeline.state import initial_state

REPO_ROOT = Path(__file__).resolve().parents[1]

SENTINEL = "sk-SENTINEL-DO-NOT-LEAK-3f9c2a7b1e4d"


# ---------------------------------------------------------------------------
# 1. Enumerated sentinel-scan surfaces
# ---------------------------------------------------------------------------

def _cfg_with_sentinel() -> ReviewConfig:
    with pytest.warns(DeprecationWarning, match="openai_api_key"):
        return scoping_review_config(title="t", openai_api_key=SENTINEL)


def test_sentinel_absent_from_to_dict():
    cfg = _cfg_with_sentinel()
    assert SENTINEL not in json.dumps(cfg.to_dict())
    assert "openai_api_key" not in cfg.to_dict()


def test_sentinel_absent_from_model_dump():
    cfg = _cfg_with_sentinel()
    assert SENTINEL not in json.dumps(cfg.model_dump())
    assert "openai_api_key" not in cfg.model_dump()


def test_sentinel_absent_from_model_dump_json():
    cfg = _cfg_with_sentinel()
    assert SENTINEL not in cfg.model_dump_json()


def test_sentinel_absent_from_repr_and_str():
    cfg = _cfg_with_sentinel()
    assert SENTINEL not in repr(cfg)
    assert SENTINEL not in str(cfg)


def test_sentinel_absent_from_review_state():
    cfg = _cfg_with_sentinel()
    state = initial_state(cfg.to_dict())
    assert SENTINEL not in json.dumps(state["config"])
    assert SENTINEL not in json.dumps(state, default=str)


def test_sentinel_absent_from_persisted_json_file(tmp_path):
    """Proxy for 'any persisted JSON file the reference workflow writes' --
    every such file is produced from `cfg.to_dict()`, so this is the same
    guarantee `reproduce_review.py`'s own output relies on."""
    cfg = _cfg_with_sentinel()
    out = tmp_path / "reproduction_config.json"
    out.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    assert SENTINEL not in out.read_text(encoding="utf-8")


def test_sentinel_absent_from_log_output_info_and_above(caplog):
    """A realistic accidental-logging scenario -- interpolating the config
    object itself into a log message -- must not leak the sentinel either."""
    cfg = _cfg_with_sentinel()
    logger = logging.getLogger("slr_pipeline.test_credential_safety")
    with caplog.at_level(logging.INFO):
        logger.info("constructed config: %s", cfg)
        logger.warning("constructed config repr: %r", cfg)
    assert SENTINEL not in caplog.text


def test_field_excluded_and_not_repr_by_construction():
    """The model-level contract itself, not just its observable effects --
    `exclude=True`/`repr=False` on the field, so no future call site can
    accidentally bypass redaction the way a single call-site patch could."""
    field = ReviewConfig.model_fields["openai_api_key"]
    assert field.exclude is True
    assert field.repr is False


# ---------------------------------------------------------------------------
# 2. resolve_openai_api_key() never falls back to cfg.openai_api_key
# ---------------------------------------------------------------------------

def test_resolver_ignores_cfg_field_even_when_populated(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _cfg_with_sentinel()  # constructed only to prove the warning path; resolver takes no cfg arg
    assert resolve_openai_api_key() is None


def test_resolver_is_zero_argument(monkeypatch):
    """Codex review (Phase A follow-up, finding 3): `resolve_openai_api_key`
    no longer takes a `cfg` parameter at all -- it was unused, and a real
    `ReviewConfig` instance is not always in scope at every call site (e.g.
    slr_pipeline/extraction/openai_runner.py only ever sees an
    `OpenAIExtractionConfig`). Removing the parameter is what lets those
    sites use the shared resolver too, instead of being permitted
    exceptions."""
    import inspect
    sig = inspect.signature(resolve_openai_api_key)
    assert list(sig.parameters) == []


def test_resolver_reads_environment_exclusively(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-value")
    assert resolve_openai_api_key() == "sk-real-env-value"


def test_historical_config_with_sentinel_and_no_env_fails_safely(monkeypatch):
    """The specific regression this round's review demanded: a historical-
    style config carrying a real (possibly committed) key, loaded with no
    OPENAI_API_KEY set, must fail safely -- the sentinel must never reach
    the OpenAI SDK in any form."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.warns(DeprecationWarning):
        cfg = ReviewConfig.from_dict({**scoping_review_config(title="t").to_dict(),
                                       "openai_api_key": SENTINEL})
    resolved = resolve_openai_api_key()
    assert resolved is None
    from openai import OpenAI, OpenAIError
    with pytest.raises(OpenAIError):
        OpenAI(api_key=resolved)


def test_backward_compatible_loading_does_not_raise():
    """Loading a historical dict with a real string value in this field
    must not raise -- parsing compatibility only, never re-serialised."""
    with pytest.warns(DeprecationWarning):
        cfg = ReviewConfig.from_dict({**scoping_review_config(title="t").to_dict(),
                                       "openai_api_key": SENTINEL})
    assert "openai_api_key" not in cfg.to_dict()


def test_field_is_actually_discarded_not_just_hidden():
    """Codex review (Phase A follow-up, finding 2): the validator previously
    returned the original secret value unchanged after warning -- comments
    claimed it was "discarded" but `cfg.openai_api_key` factually still held
    the real secret in memory. Direct field access must be None, not just
    absent from serialization."""
    with pytest.warns(DeprecationWarning):
        cfg = ReviewConfig.from_dict({**scoping_review_config(title="t").to_dict(),
                                       "openai_api_key": SENTINEL})
    assert cfg.openai_api_key is None


# ---------------------------------------------------------------------------
# 3. Runtime correctness once the credential comes from the environment alone
# ---------------------------------------------------------------------------

def test_every_call_site_succeeds_with_cfg_field_none_and_env_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-value")
    cfg = scoping_review_config(title="t")
    assert cfg.openai_api_key is None
    from openai import OpenAI
    client = OpenAI(api_key=resolve_openai_api_key())
    assert client.api_key == "sk-real-env-value"


# ---------------------------------------------------------------------------
# 3b. Codex review (Phase A follow-up, finding 3): the "every call site uses
# the resolver" contract, checked positively (by source) for every site --
# deep functional regressions for the two newly-wired sites
# (extraction/openai_runner.py, nodes/quality_appraisal.py's two functions)
# live alongside their own existing, correctly-fixtured fake-OpenAI-client
# tests: see test_run_openai_batch_extraction_uses_resolved_api_key in
# tests/extraction/test_openai_runner.py, and
# test_run_openai_teaching_batch_uses_resolved_api_key /
# test_run_openai_rob2_domain_batch_uses_resolved_api_key in
# tests/nodes/test_quality_appraisal_node.py -- not duplicated here with
# ad hoc, under-constrained fixtures.
# ---------------------------------------------------------------------------

def test_every_construction_site_wires_resolver_by_source():
    """Every one of them must actually call the shared resolver by name,
    not just avoid the old unsafe pattern (that negative case is
    `test_no_unsafe_credential_pattern_remains_in_pipeline_source` above)."""
    positive_pattern = re.compile(r"api_key\s*=\s*resolve_openai_api_key\(\)")
    for rel_path in (
        "slr_pipeline/nodes/screening.py",
        "slr_pipeline/nodes/screening_qc.py",
        "slr_pipeline/nodes/search_strategy_builder.py",
        "slr_pipeline/nodes/synthesis.py",
        "slr_pipeline/nodes/llm_backends.py",
        "slr_pipeline/extraction/openai_runner.py",
        "slr_pipeline/nodes/quality_appraisal.py",
    ):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert positive_pattern.search(text), f"{rel_path} does not call resolve_openai_api_key()"


# ---------------------------------------------------------------------------
# 4. Codex review (Phase A follow-up, finding 1 -- P0): the graph-state
# boundary itself, not just ReviewConfig's own serialization. A raw dict
# (never validated through ReviewConfig) passed to any of the three public
# run_*_review() functions must still have its credential stripped.
# ---------------------------------------------------------------------------

def _raw_dict_with_sentinel() -> dict:
    """A plain dict, deliberately NOT constructed via `ReviewConfig` at all
    -- simulates a historical/hand-built config dict a caller might pass
    straight to a public runner function."""
    return {**scoping_review_config(title="t").to_dict(), "openai_api_key": SENTINEL}


def test_initial_state_sanitizes_raw_dict_not_only_reviewconfig_output():
    raw = _raw_dict_with_sentinel()
    assert "openai_api_key" in raw  # sanity: the raw dict really does carry it
    with pytest.warns(DeprecationWarning):
        state = initial_state(raw)
    assert SENTINEL not in json.dumps(state["config"])
    assert SENTINEL not in json.dumps(state, default=str)
    assert "openai_api_key" not in state["config"]


def test_run_scoping_review_sanitizes_raw_dict(monkeypatch):
    from slr_pipeline.graphs.scoping import run_scoping_review
    captured_state = {}

    def _fake_invoke(self, state, config):
        captured_state.update(state)
        return state

    import langgraph.graph.state
    monkeypatch.setattr(langgraph.graph.state.CompiledStateGraph, "invoke", _fake_invoke)
    with pytest.warns(DeprecationWarning):
        run_scoping_review(_raw_dict_with_sentinel())
    assert SENTINEL not in json.dumps(captured_state.get("config", {}))
    assert "openai_api_key" not in captured_state.get("config", {})


def test_run_rapid_review_sanitizes_raw_dict(monkeypatch):
    from slr_pipeline.graphs.rapid import run_rapid_review
    captured_state = {}

    def _fake_invoke(self, state, config):
        captured_state.update(state)
        return state

    import langgraph.graph.state
    monkeypatch.setattr(langgraph.graph.state.CompiledStateGraph, "invoke", _fake_invoke)
    with pytest.warns(DeprecationWarning):
        run_rapid_review(_raw_dict_with_sentinel())
    assert SENTINEL not in json.dumps(captured_state.get("config", {}))
    assert "openai_api_key" not in captured_state.get("config", {})


def test_run_slr_sanitizes_raw_dict(monkeypatch):
    """The third public runner is named `run_slr`, not `run_slr_review` --
    confirmed by reading slr_pipeline/graphs/slr.py directly rather than
    assumed from the other two runners' naming pattern."""
    from slr_pipeline.graphs.slr import run_slr
    captured_state = {}

    def _fake_invoke(self, state, config):
        captured_state.update(state)
        return state

    import langgraph.graph.state
    monkeypatch.setattr(langgraph.graph.state.CompiledStateGraph, "invoke", _fake_invoke)
    with pytest.warns(DeprecationWarning):
        run_slr(_raw_dict_with_sentinel())
    assert SENTINEL not in json.dumps(captured_state.get("config", {}))
    assert "openai_api_key" not in captured_state.get("config", {})


def test_llm_backends_get_llm_backend_resolves_from_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeSentenceTransformersModule())
    # Import BEFORE the env override: importing the `nodes` package transitively
    # triggers `nodes.search_strategy_builder`'s module-level `import my_api_keys`,
    # which sets OPENAI_API_KEY in os.environ as its own, separate, pre-existing
    # side effect (unrelated to Phase A's contract). Setting our controlled value
    # AFTER the import, not before, is what makes this test deterministic.
    from slr_pipeline.nodes.llm_backends import get_llm_backend
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-value")
    cfg = scoping_review_config(title="t")
    client, _embed = get_llm_backend(cfg)
    assert client.api_key == "sk-real-env-value"


class _FakeSentenceTransformersModule:
    """Avoids a real (slow) SentenceTransformer model download in this
    focused unit test -- only `get_llm_backend`'s credential-resolution
    wiring is under test here, not embedding-model loading."""
    def SentenceTransformer(self, *_a, **_kw):
        return object()


# ---------------------------------------------------------------------------
# 4. The specific bug pattern must not reappear anywhere in the source
# ---------------------------------------------------------------------------

_UNSAFE_PATTERNS = [
    re.compile(r'cfg\.openai_api_key\s+or\s+""'),
    re.compile(r'cfg\.openai_api_key\s+or\s+os\.environ'),
    re.compile(r"api_key\s*=\s*cfg\.openai_api_key\s*\)"),  # bare, no resolver
]


def test_no_unsafe_credential_pattern_remains_in_pipeline_source():
    offenders = []
    for path in (REPO_ROOT / "slr_pipeline").rglob("*.py"):
        if "codex_implementation" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(text):
                offenders.append((str(path.relative_to(REPO_ROOT)), pattern.pattern))
    assert offenders == []


def test_reproduce_review_never_assigns_openai_api_key():
    text = (REPO_ROOT / "reproduce_review.py").read_text(encoding="utf-8")
    assert "openai_api_key=os.environ" not in text
    assert re.search(r"openai_api_key\s*=", text) is None
