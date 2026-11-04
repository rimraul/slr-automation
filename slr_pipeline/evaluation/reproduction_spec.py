"""Versioned description of an external review reproduction attempt.

The specification records both the protocol reported by the source review and
what this pipeline actually did. It deliberately stays separate from the gold
standard: the latter describes the source review's outcomes, while this model
describes one attempt to reproduce them.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slr_pipeline.extraction.hashing import sha256_hex


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_DATABASE_ALIASES = {
    "pub_med": "pubmed",
    "pubmed_gov": "pubmed",
    "web_of_science": "web_of_science",
    "wos": "web_of_science",
    "acm_digital_library": "acm",
    "business_source_complete": "business_source_complete",
}


def normalize_database_name(value: str) -> str:
    """Return a stable token for spelling variants of one named database.

    The aliases are intentionally narrow. Related database/platform identities
    remain distinct: PubMed, MEDLINE, Ovid MEDLINE, and EBSCO MEDLINE are not
    interchangeable; neither are databases that happen to share a vendor.
    """

    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not token:
        raise ValueError("database names must not be blank")
    return _DATABASE_ALIASES.get(token, token)


class DatabaseCoverage(BaseModel):
    """Normalized database coverage for one reproduction attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_databases: tuple[str, ...]
    unsupported_databases: tuple[str, ...]
    partial_reproduction: bool


class ReproductionSpec(BaseModel):
    """Protocol and execution facts for one reproduction attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: Literal[1] = 1
    reproduction_mode: Literal[
        "search_reproduction", "screening_replay", "review_update"
    ]
    source_mode: Literal["archived_exports", "live_search"]
    citation: str
    original_databases: list[str]
    reproduced_databases: list[str]
    exact_queries: dict[str, str]
    search_dates: dict[str, str]
    filters: list[str]
    manual_additions: list[str]
    reported_prisma_counts: dict[str, int]
    unknowns: list[str]
    cutoff_policy: Literal[
        "latest_search_date", "earliest_search_date"
    ] = "latest_search_date"

    review_id: str
    review_type: str
    search_ordering: Optional[str] = None
    pagination_notes: Optional[str] = None
    original_dedup_rules: Optional[str] = None
    title_abstract_eligibility_criteria: str
    full_text_eligibility_criteria: str
    study_design_constraints: list[str] = Field(default_factory=list)
    language_constraints: list[str] = Field(default_factory=list)
    gold_exclusion_refs: list[str] = Field(default_factory=list)
    original_charting_fields_refs: list[str] = Field(default_factory=list)
    queries_executed: dict[str, str] = Field(default_factory=dict)

    gold_standard_ref: str
    expected_gold_standard_hash: str

    @field_validator(
        "citation",
        "review_id",
        "review_type",
        "title_abstract_eligibility_criteria",
        "full_text_eligibility_criteria",
        "gold_standard_ref",
        "expected_gold_standard_hash",
    )
    @classmethod
    def _require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("original_databases", "reproduced_databases")
    @classmethod
    def _validate_database_lists(cls, values: list[str]) -> list[str]:
        for value in values:
            normalize_database_name(value)
        return values

    @field_validator("expected_gold_standard_hash")
    @classmethod
    def _validate_gold_standard_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(
                "expected_gold_standard_hash must be a lowercase SHA-256 hex digest"
            )
        return value

    @field_validator("search_dates")
    @classmethod
    def _validate_search_dates(cls, values: dict[str, str]) -> dict[str, str]:
        normalized_dates: dict[str, str] = {}
        for database, value in values.items():
            normalized = normalize_database_name(database)
            if value != "unknown":
                if not _ISO_DATE_RE.fullmatch(value):
                    raise ValueError("search dates must use YYYY-MM-DD or unknown")
                date.fromisoformat(value)
            previous = normalized_dates.get(normalized)
            if previous is not None and previous != value:
                raise ValueError(
                    "conflicting search dates for database spelling variants "
                    f"{database!r}"
                )
            normalized_dates[normalized] = value
        return values

    def spec_hash(self) -> str:
        """Hash the complete versioned specification deterministically."""

        return sha256_hex(self.model_dump(mode="json"))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ReproductionSpec":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def coverage(self) -> DatabaseCoverage:
        """Compare normalized reproduced databases with the source protocol."""

        original = {normalize_database_name(v) for v in self.original_databases}
        reproduced = {
            normalize_database_name(v) for v in self.reproduced_databases
        }
        missing = tuple(sorted(original - reproduced))
        unsupported = tuple(sorted(reproduced - original))
        return DatabaseCoverage(
            missing_databases=missing,
            unsupported_databases=unsupported,
            partial_reproduction=bool(missing),
        )

    def effective_cutoff(self) -> date | None:
        """Combine known per-database search dates under the stated policy."""

        known_dates = [
            date.fromisoformat(value)
            for value in self.search_dates.values()
            if value != "unknown"
        ]
        if not known_dates:
            return None
        if self.cutoff_policy == "earliest_search_date":
            return min(known_dates)
        return max(known_dates)

    def cutoff_for_database(self, database: str | None) -> date | None:
        """Use a known source-specific date, otherwise the effective cutoff."""

        if database:
            target = normalize_database_name(database)
            for name, value in self.search_dates.items():
                if normalize_database_name(name) == target and value != "unknown":
                    return date.fromisoformat(value)
        return self.effective_cutoff()


__all__ = [
    "DatabaseCoverage",
    "ReproductionSpec",
    "normalize_database_name",
]
