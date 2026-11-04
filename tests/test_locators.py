"""Unit tests for `slr_pipeline.locators` -- the `slr://` logical-locator
grammar (`track2_pipeline_readiness_plan_2026-08-30.md` Sec 10.3), used at
the reporting boundary (`slr_pipeline.export_manifest`)."""

from __future__ import annotations

import pytest

from slr_pipeline.locators import (
    LOCATOR_GRAMMAR_VERSION, RUN_LOCATOR_VERSION, LocatorGlob, LogicalLocator,
    _parse_and_check_locator, expand_glob, is_glob,
)


def test_grammar_version_pins_are_strings():
    assert LOCATOR_GRAMMAR_VERSION == "1"
    assert RUN_LOCATOR_VERSION == "1"


def test_logical_locator_str_roundtrip_without_subid():
    loc = LogicalLocator("reporting", "artifact", "methods_md")
    assert str(loc) == "slr://reporting/artifact/methods_md"
    assert LogicalLocator.parse(str(loc)) == loc


def test_logical_locator_str_roundtrip_with_subid():
    loc = LogicalLocator("extraction", "checkpoint", "extrun-abc", "rec-a")
    assert str(loc) == "slr://extraction/checkpoint/extrun-abc/rec-a"
    assert LogicalLocator.parse(str(loc)) == loc


def test_logical_locator_parse_rejects_malformed_grammar():
    with pytest.raises(ValueError, match="not a well-formed"):
        LogicalLocator.parse("slr://reporting/artifact")
    with pytest.raises(ValueError, match="not a well-formed"):
        LogicalLocator.parse("not-a-locator")


def test_logical_locator_parse_rejects_unregistered_kind():
    with pytest.raises(ValueError, match="not a registered LogicalLocator kind"):
        LogicalLocator.parse("slr://nonexistent_kind/whatever/id1")


def test_logical_locator_parse_rejects_unregistered_subkind():
    with pytest.raises(ValueError, match="not a registered subkind"):
        LogicalLocator.parse("slr://reporting/wrong_subkind/id1")


def test_parse_and_check_locator_kind_cross_check():
    value = str(LogicalLocator("selection_ledger", "decision", "abc123", "def456"))
    assert _parse_and_check_locator(value, kind="selection_ledger") == value
    with pytest.raises(ValueError, match="expected kind"):
        _parse_and_check_locator(value, kind="reporting")


def test_parse_and_check_locator_subkind_cross_check():
    value = str(LogicalLocator("extraction", "manifest", "extrun-abc"))
    assert _parse_and_check_locator(value, subkind="manifest") == value
    with pytest.raises(ValueError, match="expected subkind"):
        _parse_and_check_locator(value, subkind="checkpoint")


def test_locator_glob_str_roundtrip():
    glob = LocatorGlob("ai_use_events", "event")
    assert str(glob) == "slr://ai_use_events/event/*"
    assert LocatorGlob.parse(str(glob)) == glob


def test_locator_glob_parse_rejects_non_glob():
    with pytest.raises(ValueError, match="not a well-formed LocatorGlob"):
        LocatorGlob.parse("slr://ai_use_events/event/some-id")


def test_locator_glob_parse_rejects_unregistered_kind_subkind():
    with pytest.raises(ValueError, match="not a registered LocatorGlob"):
        LocatorGlob.parse("slr://ai_use_events/nonexistent/*")


def test_is_glob_distinguishes_glob_from_locator():
    assert is_glob("slr://ai_use_events/event/*") is True
    assert is_glob("slr://ai_use_events/event/some-id") is False


def test_expand_glob_filters_by_kind_and_subkind():
    glob = LocatorGlob("ai_use_events", "event")
    candidates = [
        "slr://ai_use_events/event/aiuse-2",
        "slr://ai_use_events/event/aiuse-1",
        "slr://extraction/manifest/extrun-abc",
    ]
    assert expand_glob(glob, candidates) == [
        "slr://ai_use_events/event/aiuse-1", "slr://ai_use_events/event/aiuse-2",
    ]


def test_expand_glob_dedupes_and_sorts():
    glob = LocatorGlob("selection_ledger", "decision")
    candidates = [
        "slr://selection_ledger/decision/aaa/bbb",
        "slr://selection_ledger/decision/aaa/bbb",
        "slr://selection_ledger/decision/aaa/ccc",
    ]
    assert expand_glob(glob, candidates) == [
        "slr://selection_ledger/decision/aaa/bbb", "slr://selection_ledger/decision/aaa/ccc",
    ]


def test_expand_glob_raises_on_malformed_candidate():
    glob = LocatorGlob("ai_use_events", "event")
    with pytest.raises(ValueError, match="not a well-formed"):
        expand_glob(glob, ["not-a-locator"])
