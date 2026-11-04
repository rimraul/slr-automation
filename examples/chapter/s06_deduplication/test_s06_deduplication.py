"""Offline tests for the S6 deduplication solution."""

from __future__ import annotations

import ast
import difflib
import inspect
import re
import textwrap

import pytest

from examples.chapter import toolkit
from examples.chapter.s06_deduplication import solution as sol


@pytest.fixture
def records():
    return toolkit.load_fixture("bib_records.json")


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


# --- normalise_doi ---------------------------------------------------

def test_normalise_doi_strips_prefix_and_lowercases():
    assert sol.normalise_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert sol.normalise_doi("http://dx.doi.org/10.1000/abc") == "10.1000/abc"
    assert sol.normalise_doi("  10.1000/ABC  ") == "10.1000/abc"
    assert sol.normalise_doi(None) is None


def _executable_body_src(func) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    fn = tree.body[0]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop the docstring
    return "\n".join(ast.unparse(node) for node in body)


def test_normalise_doi_uses_prefix_removal_not_lstrip_charset():
    # str.lstrip(prefix) treats the argument as a CHARACTER SET, not a prefix.
    # It happens to work for https://doi.org/10... (a DOI starts with '1', not
    # in the set) but silently corrupts other prefix forms.
    code = _executable_body_src(sol.normalise_doi)
    assert ".lstrip(" not in code, "use str.removeprefix, not str.lstrip (charset semantics)"
    assert ".removeprefix(" in code, "prefix stripping must be prefix-based"
    # A naive s.lstrip('https://doi.org/') on the dx.doi.org form yields
    # 'x.doi.org/...'; prefix-based removal yields the bare DOI.
    assert sol.normalise_doi("http://dx.doi.org/10.1000/x") == "10.1000/x"
    assert "https://doi.org/10.1000/x".lstrip("https://doi.org/") == "10.1000/x"  # the accidental case
    assert "http://dx.doi.org/10.1000/x".lstrip("https://doi.org/") == "x.doi.org/10.1000/x"  # the bug


# --- dedupe counts --------------------------------------------------

def test_dedupe_counts(records):
    result = sol.dedupe(records)
    assert len(result.unique) == 15
    assert len(result.duplicates) == 3


def test_dedupe_matched_on_reasons(records):
    result = sol.dedupe(records)
    by_reason = {}
    for pair in result.duplicates:
        by_reason.setdefault(pair.matched_on, []).append(pair.dup_id)
    assert sorted(by_reason["doi"]) == ["rec-02", "rec-04"]
    assert by_reason["title"] == ["rec-06"]


def test_doi_only_pass_leaves_sixteen(records):
    # An impossible title threshold disables stage 2 -> only the DOI pairs go.
    result = sol.dedupe(records, title_threshold=1.01)
    assert len(result.unique) == 16


def test_strict_title_threshold_misses_the_fuzzy_pair(records):
    result = sol.dedupe(records, title_threshold=0.99)
    assert len(result.unique) == 16
    assert all(p.matched_on == "doi" for p in result.duplicates)


# --- fixture structural properties (asserted here per plan 2.1) -----

def test_fixture_has_exactly_two_doi_duplicate_pairs(records):
    by_doi = {}
    for r in records:
        nd = sol.normalise_doi(r["doi"])
        if nd:
            by_doi.setdefault(nd, []).append(r)
    groups = [v for v in by_doi.values() if len(v) > 1]
    assert len(groups) == 2
    for group in groups:
        assert len({r["source"] for r in group}) == len(group), "duplicate pair should span sources"


def test_fixture_has_one_fuzzy_pair_and_no_other_high_ratio_pair(records):
    high = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if sol.normalise_doi(a["doi"]) and sol.normalise_doi(a["doi"]) == sol.normalise_doi(b["doi"]):
                continue  # a DOI pair, handled by stage 1
            ratio = difflib.SequenceMatcher(None, _norm_title(a["title"]), _norm_title(b["title"])).ratio()
            if ratio >= 0.80:
                high.append((a["id"], b["id"], round(ratio, 3)))
    assert len(high) == 1, high
    a_id, b_id, ratio = high[0]
    assert {a_id, b_id} == {"rec-05", "rec-06"}
    assert 0.90 <= ratio < 0.99
    assert any(records_by_id(records, x)["doi"] is None for x in (a_id, b_id))


def test_expected_screening_label_balance():
    exp = toolkit.load_fixture("expected_screening.json")
    counts = {"include": 0, "borderline": 0, "exclude": 0}
    for v in exp.values():
        counts[v] += 1
    assert counts["include"] >= 6
    assert counts["exclude"] >= 6
    assert counts["borderline"] >= 3


def records_by_id(records, rid):
    return next(r for r in records if r["id"] == rid)
