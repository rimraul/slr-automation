"""
`ReportingStageManifest` / `ReportingStageEntry` -- the live/logical-source
manifest `reporting_node` writes to
`<output_dir>/reporting/<review_run_id>/export_manifest.json`
(`track2_pipeline_readiness_plan_2026-08-30.md` Sec 10.1(b)).

This is deliberately NOT `PayloadManifestV1` (Sec 10.1(a)): that model is
`payload/`-relative-path-owning, bundle-level, and owned by
`reference_bundle.py::finalize_bundle()` (a later, post-graph step). This
manifest is created DURING the graph, by `reporting_node`, and never carries
a `path`/`relative_path` field -- only `slr://` logical locators (see
`slr_pipeline.locators`), an optional `live_relpath` for a human/live
re-check, and content hashes. A bundle consumer resolves every file through
`PayloadManifestV1` alone; this manifest is evidence for cross-checking a
`PayloadManifestV1`-resolved reporting file against what `reporting_node`
itself recorded, never a resolver.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, constr, field_validator, model_validator

from slr_pipeline.locators import POSIX_RELPATH, LocatorGlob, _parse_and_check_locator, is_glob

_HEX64 = constr(pattern=r"^[0-9a-f]{64}$")

# entry_kind -> the LogicalLocator.kind its logical_path must carry.
_ENTRY_KIND_LOCATOR = {
    "renderer_artifact": "reporting",
    "selection_decision": "selection_ledger",
    "screening_batch": "screening_batch",
    "extraction_component": "extraction",
    "qa_component": "quality_appraisal",
    "ai_use_event": "ai_use_events",
    "ft_run_artifact": "ft_screening",
}

# entry_kind -> fields that are UNCONDITIONALLY required (non-None) on it.
_ENTRY_KIND_REQUIRED: Dict[str, Set[str]] = {
    "renderer_artifact": {"name"},  # built_from is CONDITIONALLY required (status == "present") -- see below
    "selection_decision": {"record_id", "stage", "decision_id", "internal_hash"},
    "screening_batch": {"artifact_id", "batch_fingerprint", "internal_hash"},
    "extraction_component": {"run_id", "component", "internal_hash"},
    "qa_component": {"run_id", "component", "internal_hash"},
    "ai_use_event": {"event_id", "work_item_id", "stage", "internal_hash"},
    "ft_run_artifact": {"ft_screening_run_id", "component"},
}

# entry_kind -> fields that are ALLOWED but only conditionally required (a
# field that is merely conditionally required must still be in the ALLOWED
# set, or "extra = present - required" would classify it as forbidden
# before the model_validator's own conditional check ever runs).
_ENTRY_KIND_ALLOWED_OPTIONAL: Dict[str, Set[str]] = {
    "renderer_artifact": {"built_from"},
    "extraction_component": {"document_id"},
    "qa_component": {"document_id"},
}

_ALL_PER_KIND_FIELDS = {
    "name", "built_from", "record_id", "stage", "decision_id", "internal_hash",
    "artifact_id", "batch_fingerprint", "run_id", "component", "document_id",
    "event_id", "work_item_id", "ft_screening_run_id",
}

_COMPONENT_VOCAB: Dict[str, Set[str]] = {
    "extraction_component": {
        "manifest", "checkpoint", "resolution", "review_packet", "queue_snapshot",
        "suggestion", "suggestions_index", "decision_journal",
    },
    "qa_component": {"manifest", "checkpoint", "resolution", "confirmation"},
    "ft_run_artifact": {"run_result", "item_results", "ledger"},
}


class ReportingStageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_kind: Literal[
        "renderer_artifact", "selection_decision", "screening_batch",
        "extraction_component", "qa_component", "ai_use_event", "ft_run_artifact",
    ]
    logical_path: str  # canonical slr:// NAME (slr_pipeline.locators)
    status: Literal["present", "not_applicable", "failed"] = "present"
    reason: Optional[str] = None
    file_sha256: Optional[_HEX64] = None  # required (non-None) exactly when status == "present"
    internal_hash: Optional[_HEX64] = None
    built_from: Optional[List[str]] = None  # renderer_artifact only; else None
    live_relpath: Optional[POSIX_RELPATH] = None  # relative to <output_dir>, human/live re-check only; not used in the bundle

    # per-kind fields -- every field below is None unless its owning entry_kind (table above) requires/allows it
    name: Optional[str] = None  # renderer_artifact
    record_id: Optional[str] = None  # selection_decision
    stage: Optional[str] = None  # selection_decision, ai_use_event
    decision_id: Optional[str] = None  # selection_decision
    artifact_id: Optional[str] = None  # screening_batch
    batch_fingerprint: Optional[str] = None  # screening_batch
    run_id: Optional[str] = None  # extraction_component, qa_component
    component: Optional[str] = None  # extraction_component, qa_component, ft_run_artifact
    document_id: Optional[str] = None  # extraction_component/qa_component, component == "checkpoint" only
    event_id: Optional[str] = None  # ai_use_event
    work_item_id: Optional[str] = None  # ai_use_event
    ft_screening_run_id: Optional[str] = None  # ft_run_artifact

    @field_validator("logical_path")
    @classmethod
    def _valid_locator(cls, v: str) -> str:
        return _parse_and_check_locator(v)  # kind cross-check happens below, once entry_kind is known

    @model_validator(mode="after")
    def _fields_match_entry_kind(self) -> "ReportingStageEntry":
        required = _ENTRY_KIND_REQUIRED[self.entry_kind]
        allowed = required | _ENTRY_KIND_ALLOWED_OPTIONAL.get(self.entry_kind, set())
        present = {f for f in _ALL_PER_KIND_FIELDS if getattr(self, f) is not None}
        missing = required - present
        if missing:
            raise ValueError(f"{self.entry_kind} entry missing required field(s): {sorted(missing)}")
        extra = present - allowed
        if extra:
            raise ValueError(f"{self.entry_kind} entry sets field(s) not applicable to it: {sorted(extra)}")
        if self.document_id is not None and self.component != "checkpoint":
            raise ValueError("document_id is only set when component == 'checkpoint'")
        if self.component == "checkpoint" and self.document_id is None:
            raise ValueError("document_id is required when component == 'checkpoint'")
        if self.component is not None and self.entry_kind in _COMPONENT_VOCAB:
            if self.component not in _COMPONENT_VOCAB[self.entry_kind]:
                raise ValueError(f"{self.entry_kind}.component must be one of {sorted(_COMPONENT_VOCAB[self.entry_kind])}")
        _parse_and_check_locator(self.logical_path, kind=_ENTRY_KIND_LOCATOR[self.entry_kind])
        if self.status != "present" and self.file_sha256 is not None:
            raise ValueError("file_sha256 must be None when status != 'present'")
        if self.status == "present" and self.file_sha256 is None:
            raise ValueError("file_sha256 is required when status == 'present'")
        if self.entry_kind == "renderer_artifact" and self.status != "present" and not self.reason:
            raise ValueError("a non-present renderer_artifact entry requires a reason")
        if self.entry_kind == "renderer_artifact" and self.status == "present" and self.reason is not None:
            raise ValueError("reason must be None on a present renderer_artifact entry")
        if self.entry_kind != "renderer_artifact" and self.reason is not None:
            raise ValueError(f"reason only applies to renderer_artifact entries, not {self.entry_kind}")
        if self.entry_kind == "renderer_artifact" and self.status == "present" and self.built_from is None:
            raise ValueError("a present renderer_artifact entry requires built_from")
        if self.entry_kind == "renderer_artifact" and self.status != "present" and self.built_from is not None:
            raise ValueError("built_from must be None on a non-present renderer_artifact entry")
        for loc in self.built_from or []:  # built_from is a validated List[str], never a bare List[str]
            if is_glob(loc):
                LocatorGlob.parse(loc)  # raises ValueError on an unregistered glob kind/subkind
            else:
                _parse_and_check_locator(loc)
        return self


class M1ProtocolInfo(BaseModel):
    """Verified against the real producer, `ReportingConfigProjection`
    (`slr_pipeline/nodes/reporting.py`). Every field here is `Optional`
    because `ReviewConfig`'s own `protocol_*`/`ai_use_prespecified` fields
    are all `Optional[...] = None` (config.py) -- a real, common case is a
    review that has not (yet) registered a protocol. The plan's Sec 10.1
    snippet writes `protocol_hash: constr(pattern=hex64)`; that would
    reject every such config, so this model instead matches
    `ReportingConfigProjection`'s real, already-verified field types."""

    model_config = ConfigDict(extra="forbid")
    protocol_uri: Optional[str]
    protocol_hash: Optional[str]
    protocol_registration_id: Optional[str]
    ai_use_prespecified: Optional[bool]
    protocol_deviations: List[str]


class ReportingStageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reporting_stage_manifest_version: Literal["1"] = "1"
    review_run_id: str
    review_type: Literal["scoping", "rapid", "slr"]
    m1_protocol: M1ProtocolInfo
    m10_data_governance: Optional[str]
    entries: List[ReportingStageEntry]
    manifest_hash: _HEX64  # sha256(canonical_json(this minus manifest_hash)) -- see reporting.py's _hash_value;
    # rfc8785 JCS migration (plan Sec 10.7/17-3) is a step-10 dependency, not yet installed -- deferred.

    @model_validator(mode="after")
    def _entries_sorted_and_unique(self) -> "ReportingStageManifest":
        paths = [e.logical_path for e in self.entries]
        if paths != sorted(paths):
            raise ValueError("entries must be sorted by logical_path")
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate logical_path in entries")
        return self
