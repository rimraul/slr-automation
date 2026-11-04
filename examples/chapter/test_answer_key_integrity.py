"""papers/answer_key.json must stay honest: every evidence quote it names is a
verbatim substring of the paper it belongs to, and every declared DOI is
resolvable through the S8 resolver fixture."""

from __future__ import annotations

import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"
_PAPERS = _DATA / "papers"

_ANSWER_KEY = json.loads((_PAPERS / "answer_key.json").read_text(encoding="utf-8"))
_RESOLVERS = json.loads((_DATA / "oa_resolver_responses.json").read_text(encoding="utf-8"))

_VALID_S08 = {"include", "exclude", "needs_review"}


def _paper_text(stem: str) -> str:
    return (_PAPERS / f"{stem}.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def test_every_paper_has_a_markdown_file_and_a_valid_s08_and_doi():
    for stem, entry in _ANSWER_KEY.items():
        assert (_PAPERS / f"{stem}.md").is_file(), stem
        assert entry["s08"] in _VALID_S08, (stem, entry["s08"])
        assert entry["doi"], stem


def test_every_evidence_quote_is_a_verbatim_substring():
    problems = []
    for stem, entry in _ANSWER_KEY.items():
        text = _paper_text(stem)
        for section in ("s09", "s10"):
            for item_id, item in entry.get(section, {}).items():
                quote = item.get("evidence_quote", "")
                if quote and quote not in text:
                    problems.append((stem, section, item_id, quote))
    assert not problems, f"non-verbatim quotes: {problems}"


def test_every_answer_key_doi_resolves_in_the_s8_fixture():
    for stem, entry in _ANSWER_KEY.items():
        doi = entry["doi"]
        assert doi in _RESOLVERS, (stem, doi)
        payloads = _RESOLVERS[doi]
        assert any(payloads.get(k) for k in ("unpaywall", "semantic_scholar")), (
            f"{stem}: answer-key DOI {doi} has no resolvable full text"
        )
