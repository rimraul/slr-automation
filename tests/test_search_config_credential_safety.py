"""Track 2 plan §2.7 -- SearchConfig credential-field deprecation, the same
treatment as ReviewConfig.openai_api_key (see
tests/test_config_credential_safety.py). Scoped to SearchConfig's three
fields: pubmed_email, springer_api_key, ieee_api_key."""
from __future__ import annotations

import json

import pytest

from slr_pipeline.config import ReviewConfig, SearchConfig, scoping_review_config
from slr_pipeline.state import initial_state

SENTINEL = "SEARCH-CREDENTIAL-SENTINEL-DO-NOT-LEAK-91a2"

_FIELDS = ("pubmed_email", "springer_api_key", "ieee_api_key")


def _cfg_with_sentinel() -> SearchConfig:
    with pytest.warns(DeprecationWarning):
        return SearchConfig(**{f: SENTINEL for f in _FIELDS})


def test_search_config_never_serializes_credentials():
    """`test_search_config_never_serializes_credentials` (plan §2.7): a
    config with all three set -> to_dict()/model_dump()/initial_state(...)
    -> no credential key, no credential value, anywhere."""
    cfg = _cfg_with_sentinel()
    for field in _FIELDS:
        assert field not in cfg.model_dump()
    assert SENTINEL not in json.dumps(cfg.model_dump())
    assert SENTINEL not in cfg.model_dump_json()

    review_cfg = scoping_review_config(title="t")
    review_cfg.search = cfg
    to_dict = review_cfg.to_dict()
    assert SENTINEL not in json.dumps(to_dict)
    for field in _FIELDS:
        assert field not in to_dict["search"]

    state = initial_state(review_cfg.to_dict())
    assert SENTINEL not in json.dumps(state["config"])
    assert SENTINEL not in json.dumps(state, default=str)


def test_search_config_fields_excluded_and_not_repr_by_construction():
    for field in _FIELDS:
        model_field = SearchConfig.model_fields[field]
        assert model_field.exclude is True
        assert model_field.repr is False


def test_search_config_fields_actually_discarded_not_just_hidden():
    cfg = _cfg_with_sentinel()
    for field in _FIELDS:
        assert getattr(cfg, field) is None


def test_search_config_sentinel_absent_from_repr_and_str():
    cfg = _cfg_with_sentinel()
    assert SENTINEL not in repr(cfg)
    assert SENTINEL not in str(cfg)


def test_search_config_backward_compatible_loading_does_not_raise():
    with pytest.warns(DeprecationWarning):
        cfg = SearchConfig.model_validate({"pubmed_email": SENTINEL})
    assert cfg.pubmed_email is None
