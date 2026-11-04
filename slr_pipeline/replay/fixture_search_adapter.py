"""
slr_pipeline.replay.fixture_search_adapter -- the seven `fx_*` adapter
classes `nodes/search.py::_ADAPTER_REGISTRY` dispatches Track 2's
`execution_adapter_map` entries to (plan Sec 2.5/2.6, Sec 6, Sec 7.1).

Each fixture adapter reads its database's synthetic corpus file --
`<corpus_root>/native/fx_<db>.json`, a JSON list of natively-shaped fixture
records -- and returns every record it holds as a `SearchBackendResult`,
ignoring `SearchBackendRequest.query`/`date_from`/`date_to`/`language`
(the corpus is small, fully deterministic, and already scoped to the
reference run; there is nothing live to filter). Per plan Sec 2.6's frozen
fixture-adapter contract: `Record.id` is the fixture's own authored id
verbatim (`fx-0001` .. `fx-0030`), never re-derived via `record_id()`;
`Record.source = "fx_<db>"`; `Record.source_id` = the authored
`native_key`.

A malformed/incomplete fixture record is an AUTHORING BUG, not a live
adapter's every-day partial failure -- `finalize_result`'s skip-tolerant
`partial`/`failed` derivation (`search_backends/base.py`) is deliberately
NOT used here; any defect raises `FixtureCorpusError` immediately (plan
Sec 2.4: "The fixture adapters raise `FixtureCorpusError` instead -- a
malformed fixture record is an authoring bug, never tolerated").

The corpus root defaults to `tests/reference_run/fixtures/corpus`
(plan Sec 7.1, a step-12 deliverable, not yet present) and is overridable
via the `SLR_FIXTURE_CORPUS_ROOT` environment variable so a later step's
`reference_run_child.py` can point at the corpus copied into its
workspace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import ValidationError

from slr_pipeline.search_backends.base import SearchBackendRequest, SearchBackendResult
from slr_pipeline.state import Record

# The seven canonical Berg-review database names Track 2's
# `execution_adapter_map = {c: f"fx_{c}" for c in ...}` maps to `fx_<c>`
# (plan Sec 2.5, Sec 4.2's `ReferenceRunSpec.databases`).
FIXTURE_DATABASES = (
    "pubmed",
    "web_of_science",
    "cinahl",
    "scopus",
    "acm",
    "engineering_village",
    "business_source_complete",
)

_ENV_CORPUS_ROOT = "SLR_FIXTURE_CORPUS_ROOT"
_DEFAULT_CORPUS_ROOT = Path("tests/reference_run/fixtures/corpus")


class FixtureCorpusError(RuntimeError):
    """Raised on any malformed/incomplete fixture record or unreadable
    corpus file -- a fixture-corpus authoring bug, never a runtime
    condition a fixture adapter tolerates (plan Sec 2.4)."""


def corpus_root() -> Path:
    override = os.environ.get(_ENV_CORPUS_ROOT)
    return Path(override) if override else _DEFAULT_CORPUS_ROOT


def _load_native_records(db: str, root: Path) -> List[Dict[str, Any]]:
    path = root / "native" / f"fx_{db}.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FixtureCorpusError(f"fx_{db}: cannot read corpus file {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FixtureCorpusError(f"fx_{db}: {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise FixtureCorpusError(f"fx_{db}: {path} must contain a JSON list of native records")
    return payload


def _record_from_native(db: str, native: Any) -> Record:
    source = f"fx_{db}"
    if not isinstance(native, dict):
        raise FixtureCorpusError(f"{source}: a native record must be a JSON object, got {type(native).__name__}")
    record_id = native.get("id")
    native_key = native.get("native_key")
    if not record_id or not isinstance(record_id, str):
        raise FixtureCorpusError(f"{source}: native record is missing a non-empty string 'id'")
    if not native_key or not isinstance(native_key, str):
        raise FixtureCorpusError(f"{source}: native record {record_id!r} is missing a non-empty string 'native_key'")
    try:
        return Record(
            id=record_id,
            source=source,
            source_id=native_key,
            doi=native.get("doi"),
            title=native.get("title") or "",
            abstract=native.get("abstract"),
            authors=list(native.get("authors") or []),
            year=native.get("year"),
            journal=native.get("journal"),
            cited_by=native.get("cited_by"),
            url=native.get("url"),
        )
    except ValidationError as exc:
        raise FixtureCorpusError(f"{source}: native record {record_id!r} failed Record validation: {exc}") from exc


class _FixtureBackend:
    """One `fx_<db>` adapter. `db` is one of `FIXTURE_DATABASES`; `name`
    (plan Sec 2.6's `source`) is `f"fx_{db}"`."""

    db: str = ""

    @property
    def name(self) -> str:
        return f"fx_{self.db}"

    def search(self, request: SearchBackendRequest) -> SearchBackendResult:
        natives = _load_native_records(self.db, corpus_root())
        records = [_record_from_native(self.db, native) for native in natives]
        # Deterministic, already-scoped corpus: every held record is
        # returned (subject to max_results), never partially -- see the
        # module docstring on `FixtureCorpusError` vs. skip-tolerance.
        returned = records[: request.max_results]
        # The whole fixture corpus is held on disk -- `len(records)` IS the
        # true total, so hitting it exactly is a real exhaustion proof, not
        # an inference; falling short of it (max_results cap) is `capped`.
        exhaustive = len(returned) == len(records)
        return SearchBackendResult(
            adapter=self.name,
            records=returned,
            retrieval_status="exhaustive" if exhaustive else "capped",
            termination_reason="total_hits_equals_retrieved" if exhaustive else "max_results_cap_reached",
            failure_detail=None,
            skipped_malformed=0,
            requested=request.max_results,
            returned=len(returned),
            total_hits=len(records),
        )


def _make_fixture_backend(db: str) -> Type[_FixtureBackend]:
    class_name = "Fixture" + "".join(part.capitalize() for part in db.split("_")) + "Backend"
    return type(class_name, (_FixtureBackend,), {"db": db})


# `nodes/search.py::_ADAPTER_REGISTRY` merges this in alongside
# `search_backends.registry.REAL_ADAPTER_CLASSES` (plan Sec 2.5).
FIXTURE_ADAPTER_CLASSES: Dict[str, Type[_FixtureBackend]] = {
    f"fx_{db}": _make_fixture_backend(db) for db in FIXTURE_DATABASES
}
