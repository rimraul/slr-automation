"""Offline tests for slr_pipeline.doi -- the shared DOI/title normalisation
helper (Track 2 plan §3.3). Regression cases target the exact bug the four
`str.lstrip(prefix)` copies had: `lstrip` treats its argument as a
character set, not a prefix."""

from __future__ import annotations

from slr_pipeline.doi import normalise_doi, normalise_title


def test_normalise_doi_strips_https_doi_org_prefix():
    assert normalise_doi("https://doi.org/10.1000/x") == "10.1000/x"


def test_normalise_doi_strips_http_dx_doi_org_prefix():
    # The old `str.lstrip("https://doi.org/")` mangled this one: none of
    # the leading characters "h","t","t","p",":","/","/","d","x","." are
    # a prefix match, so lstrip strips a run of the WRONG characters,
    # producing "x.doi.org/10.1000/x" instead of "10.1000/x".
    assert normalise_doi("http://dx.doi.org/10.1000/x") == "10.1000/x"


def test_normalise_doi_strips_http_doi_org_and_https_dx_doi_org_prefixes():
    assert normalise_doi("http://doi.org/10.1000/x") == "10.1000/x"
    assert normalise_doi("https://dx.doi.org/10.1000/x") == "10.1000/x"


def test_normalise_doi_bare_doi_passes_through_unchanged():
    assert normalise_doi("10.1000/x") == "10.1000/x"


def test_normalise_doi_casefolds_and_strips_whitespace():
    assert normalise_doi("  10.1000/X  ") == "10.1000/x"


def test_normalise_doi_none_returns_none():
    assert normalise_doi(None) is None


def test_normalise_doi_does_not_use_lstrip():
    import inspect
    src = inspect.getsource(normalise_doi)
    assert ".lstrip(" not in src, "must use str.removeprefix, not str.lstrip (character-set semantics)"


def test_normalise_title_casefolds_nfkc_and_collapses_whitespace():
    assert normalise_title("  Some   Title\there  ") == "some title here"


def test_normalise_title_exact_equality_after_normalisation():
    assert normalise_title("A Study of Cats") == normalise_title("a study   of CATS")


def test_normalise_title_distinct_titles_stay_distinct():
    assert normalise_title("A Study of Cats") != normalise_title("A Study of Dogs")


def test_normalise_title_none_returns_none():
    assert normalise_title(None) is None
