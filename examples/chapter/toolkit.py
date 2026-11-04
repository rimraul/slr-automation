"""Shared helpers for the Track 1 chapter teaching exercises (Phase E.2b).

This module is infrastructure, not a reader exercise -- its docstrings and
comments are English. Everything reader-facing (``exercise.md``,
``solution.md``, and every ``exercise.py``/``solution.py`` docstring and
in-listing comment) is Lithuanian per the chapter's language rule.

Public surface
--------------
* ``call_llm(messages, *, response_format=None, model=None, temperature=None)``
  -- the ONLY place ``openai`` is imported and an API key is read. Raises
  ``RuntimeError`` on the offline path (no key). Solutions take ``call_llm``
  as a keyword parameter defaulting to this function; tests inject a scripted
  fake with the identical signature.
* ``build_screening_request(records, criteria, *, mode)`` -- the single
  request-construction path shared by the offline and live screening code.
* ``scripted_llm(mapping, *, key_fn)`` -- build a fake ``call_llm``.
* ``load_fixture(name)`` / ``load_scripted(section)`` / ``SCRIPTED_RESPONSES``.
* ``DEMO_CRITERIA`` (a ``Criteria``) -- the one fixed teaching protocol.
* ``mock_oa_resolvers`` / ``fake_downloader`` -- S8 retrieval mocks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, StrictStr

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"

SCREEN_DECISIONS: tuple[str, str, str] = ("include", "borderline", "exclude")


def load_fixture(name: str) -> Any:
    """Resolve ``data/<name>`` relative to this module and decode by suffix.

    ``.json`` -> parsed object; ``.md``/``.xml``/``.txt`` -> text with
    newlines normalised to ``\\n``; ``.npy`` -> ``numpy.ndarray``.
    """
    path = _DATA_DIR / name
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in (".md", ".xml", ".txt"):
        return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if suffix == ".npy":
        import numpy as np

        return np.load(path)
    raise ValueError(f"load_fixture: unsupported extension for {name!r}")


def load_scripted(section: str) -> dict:
    """Load ``data/scripted_responses/<section>.json`` (the scripted fake-LLM
    payloads for one chapter section, including its ``_bad`` variants)."""
    return load_fixture(f"scripted_responses/{section}.json")


def _discover_scripted() -> dict[str, dict]:
    out: dict[str, dict] = {}
    scripted_dir = _DATA_DIR / "scripted_responses"
    if scripted_dir.is_dir():
        for path in sorted(scripted_dir.glob("*.json")):
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


SCRIPTED_RESPONSES: dict[str, dict] = _discover_scripted()


# ---------------------------------------------------------------------------
# The review protocol / screening criteria
# ---------------------------------------------------------------------------

class Criteria(BaseModel):
    """A fixed teaching protocol expressed as PCC (Population, Concept,
    Context) plus explicit inclusion / exclusion criteria. Strict and frozen
    so it is safe to share across every exercise."""

    model_config = ConfigDict(strict=True, frozen=True)

    population: StrictStr
    concept: StrictStr
    context: StrictStr
    inclusion_criteria: tuple[StrictStr, ...]
    exclusion_criteria: tuple[StrictStr, ...]


DEMO_CRITERIA = Criteria(
    population=(
        "Studies, reports, or tool descriptions concerning the conduct of "
        "systematic, scoping, or rapid literature reviews."
    ),
    concept=(
        "Automation of one or more review steps (search, deduplication, "
        "screening, full-text retrieval, data extraction, or synthesis) using "
        "large language models, machine learning, or rule-based tooling."
    ),
    context=(
        "Academic or applied research settings; any discipline; English-language "
        "reports published from 2015 onward."
    ),
    inclusion_criteria=(
        "The work describes or evaluates an automated or semi-automated method "
        "for at least one distinct review step.",
        "The review step is named and its automation is the subject of the "
        "report, not an incidental mention.",
        "Enough method detail is given for a reader to understand what was "
        "automated and how it was assessed.",
    ),
    exclusion_criteria=(
        "The work is about primary research automation (e.g. trial registries, "
        "data collection) rather than literature-review automation.",
        "The work is an opinion piece, editorial, or news item with no method "
        "and no evaluation.",
        "Full text is not in English.",
    ),
)


# ---------------------------------------------------------------------------
# The `call_llm` seam
# ---------------------------------------------------------------------------

CallLLM = Callable[..., str]

_OFFLINE_MODEL = "offline-fixture-model-not-a-real-endpoint"
_LIVE_SCREENING_MODEL = "gpt-5.4-mini"


def call_llm(
    messages: list[dict],
    *,
    response_format: dict | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Real OpenAI chat call. The only function in the chapter that imports
    ``openai`` or reads ``OPENAI_API_KEY``.

    On the offline path (no key) this raises immediately -- every offline
    test and every ``__main__`` demo must inject a scripted fake instead.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "no OPENAI_API_KEY -- this is the offline path; pass a scripted call_llm"
        )
    from openai import OpenAI  # lazy: never imported on the offline path

    client = OpenAI(api_key=api_key)
    kwargs: dict[str, Any] = {"model": model or _LIVE_SCREENING_MODEL, "messages": messages}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def scripted_llm(mapping: dict[str, str], *, key_fn: Callable[[list[dict]], str]) -> CallLLM:
    """Build a fake ``call_llm`` with the real signature. It computes
    ``key_fn(messages)`` and returns ``mapping[key]``; a miss raises
    ``KeyError`` (never a silent default)."""

    def _fake(
        messages: list[dict],
        *,
        response_format: dict | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        key = key_fn(messages)
        if key not in mapping:
            raise KeyError(f"scripted_llm: no scripted response for key {key!r}")
        return mapping[key]

    return _fake


# ---------------------------------------------------------------------------
# The single screening-request builder (shared offline + live path)
# ---------------------------------------------------------------------------

_SCREENING_SYSTEM = (
    "You are a screening reviewer for a literature review. For each record, "
    "decide 'include', 'borderline', or 'exclude' against the criteria. "
    'Return only JSON of the form {"decisions": [{"record_id": ..., '
    '"decision": ..., "reason": ...}]} with exactly one object per record.'
)

_ABSTRACT_TRUNCATE_CHARS = 1000


def _render_screening_user_message(records: list[dict], criteria: Criteria) -> str:
    lines: list[str] = []
    lines.append("PROTOCOL")
    lines.append(f"Population: {criteria.population}")
    lines.append(f"Concept: {criteria.concept}")
    lines.append(f"Context: {criteria.context}")
    lines.append("Inclusion criteria:")
    for item in criteria.inclusion_criteria:
        lines.append(f"  - {item}")
    lines.append("Exclusion criteria:")
    for item in criteria.exclusion_criteria:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("RECORDS")
    for record in records:
        abstract = (record.get("abstract") or "")[:_ABSTRACT_TRUNCATE_CHARS]
        lines.append(f"[{record['id']}] {record.get('title', '')}")
        lines.append(f"Abstract: {abstract}")
    return "\n".join(lines)


def build_screening_request(
    records: list[dict],
    criteria: Criteria,
    *,
    mode: Literal["offline", "live"],
) -> dict:
    """Construct the ``call_llm`` kwargs for one screening batch.

    Same ``messages`` byte-for-byte regardless of ``mode`` -- ``mode`` only
    selects the ``model`` id. Raises ``ValueError`` for more than 3 records
    (the batch cap) -- it never silently truncates the record list. Abstract
    *text* is truncated to 1000 chars (content shaping, not record dropping).
    The returned dict is directly ``**``-spreadable into ``call_llm``.
    """
    if mode not in ("offline", "live"):
        raise ValueError(f"mode must be 'offline' or 'live', got {mode!r}")
    if len(records) > 3:
        raise ValueError(
            f"build_screening_request accepts at most 3 records per batch, got {len(records)}"
        )
    model = _LIVE_SCREENING_MODEL if mode == "live" else _OFFLINE_MODEL
    messages = [
        {"role": "system", "content": _SCREENING_SYSTEM},
        {"role": "user", "content": _render_screening_user_message(records, criteria)},
    ]
    return {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }


# ---------------------------------------------------------------------------
# S8 retrieval mocks
# ---------------------------------------------------------------------------

class _NamedResolver:
    """A callable ``doi -> dict | None`` that also carries a ``.name`` so
    ``discover`` can record which resolver produced a hit."""

    def __init__(self, name: str, table_key: str) -> None:
        self.name = name
        self._table_key = table_key

    def __call__(self, doi: str) -> dict | None:
        table = load_fixture("oa_resolver_responses.json")
        entry = table.get(doi)
        if not entry:
            return None
        return entry.get(self._table_key)


mock_oa_resolvers: list[_NamedResolver] = [
    _NamedResolver("unpaywall", "unpaywall"),
    _NamedResolver("semantic_scholar", "semantic_scholar"),
]


def fake_downloader(url: str) -> str:
    """Map a resolver PDF URL to the corresponding synthetic paper text. The
    URL's last path segment (minus extension) is the paper stem."""
    stem = url.rstrip("/").split("/")[-1]
    stem = stem.rsplit(".", 1)[0]
    return load_fixture(f"papers/{stem}.md")
