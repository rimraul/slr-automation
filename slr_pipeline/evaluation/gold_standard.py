"""Structured ground truth for a published review.

The gold standard records the source review's reported outcomes. Attempt-level
facts such as search source mode belong in :mod:`reproduction_spec` instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slr_pipeline.doi import normalise_doi, normalise_title
from slr_pipeline.extraction.hashing import sha256_hex


class GoldPaper(BaseModel):
    """One included paper with a deterministic, verified identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = ""
    doi: Optional[str] = None
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    other_persistent_ids: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_and_verify_identity(self) -> "GoldPaper":
        normalized_doi = normalise_doi(self.doi)
        if normalized_doi:
            expected = normalized_doi
        else:
            normalized_title = normalise_title(self.title)
            if not normalized_title:
                raise ValueError("a gold paper without a DOI must have a title")
            expected = "title:" + sha256_hex(
                {"title": normalized_title, "year": self.year}
            )[:16]
        if self.paper_id and self.paper_id != expected:
            raise ValueError("paper_id does not match the paper's derived identity")
        if not self.paper_id:
            object.__setattr__(self, "paper_id", expected)

        normalized_sources: dict[str, str] = {}
        from .reproduction_spec import normalize_database_name

        for source, persistent_id in self.other_persistent_ids.items():
            normalized_source = normalize_database_name(source)
            if not persistent_id.strip():
                raise ValueError("other persistent IDs must not be blank")
            previous = normalized_sources.get(normalized_source)
            if previous is not None and previous != persistent_id:
                raise ValueError(
                    "conflicting persistent IDs for source spelling variants"
                )
            normalized_sources[normalized_source] = persistent_id
        return self


class GoldStandard(BaseModel):
    # Bibliographic identity of the source review
    review_id: str
    review_citation: str = ""
    review_doi: Optional[str] = None
    prospero_id: Optional[str] = None
    review_type: str = "scoping"

    # Original search strategy
    original_query: str = ""
    original_databases: List[str] = Field(default_factory=list)
    original_date_from: Optional[str] = None
    original_date_to: Optional[str] = None
    original_language_filter: List[str] = Field(default_factory=list)

    # Protocol / criteria
    research_questions: List[str] = Field(default_factory=list)
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    pcc_population: Optional[str] = None
    pcc_concept: Optional[str] = None
    pcc_context: Optional[str] = None

    # Published PRISMA counts (for comparison)
    prisma_identified: Optional[int] = None
    prisma_after_dedup: Optional[int] = None
    prisma_screened: Optional[int] = None
    prisma_assessed_fulltext: Optional[int] = None
    prisma_included: Optional[int] = None

    # Structured included papers. The two legacy lists remain load-compatible,
    # but they are independent and must never be paired by position.
    included_papers: List[GoldPaper] = Field(default_factory=list)
    included_dois: List[str] = Field(default_factory=list)
    included_titles: List[str] = Field(default_factory=list)
    structurally_ambiguous: bool = False

    # Quality flags
    doi_coverage: float = 0.0
    notes: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_included_lists(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("included_papers"):
            return data

        dois = [doi for doi in data.get("included_dois", []) if doi]
        titles = [title for title in data.get("included_titles", []) if title]
        if not dois and not titles:
            return data

        # Independent lists cannot be honestly zipped: synthesize distinct
        # entries and mark the resulting per-paper identity as unknowable.
        papers: list[dict[str, Any]] = [
            {"doi": doi, "title": ""} for doi in dois
        ]
        papers.extend({"doi": None, "title": title} for title in titles)
        data["included_papers"] = papers
        data["structurally_ambiguous"] = True
        return data

    @model_validator(mode="after")
    def _paper_ids_are_unique(self) -> "GoldStandard":
        paper_ids = [paper.paper_id for paper in self.included_papers]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("included_papers contains duplicate derived paper_id values")
        return self

    @property
    def n_included(self) -> int:
        if not self.structurally_ambiguous and self.included_papers:
            return len(self.included_papers)
        return max(
            len(self.included_dois),
            len(self.included_titles),
            self.prisma_included or 0,
        )

    def normalise_doi(self, doi: str) -> str:
        return normalise_doi(doi) or ""

    def doi_set(self) -> set[str]:
        if not self.structurally_ambiguous and self.included_papers:
            return {
                normalized
                for paper in self.included_papers
                if (normalized := normalise_doi(paper.doi))
            }
        return {self.normalise_doi(d) for d in self.included_dois if d}

    def content_hash(self) -> str:
        """Bind evaluations to the complete parsed gold-standard content."""

        return sha256_hex(self.model_dump(mode="json"))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "GoldStandard":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_gold_standard(review_dir: str | Path) -> GoldStandard:
    """Load ``gold_standard.json`` from a review folder."""

    path = Path(review_dir) / "gold_standard.json"
    if not path.exists():
        raise FileNotFoundError(
            f"gold_standard.json not found in {review_dir}. "
            "Run extract_gold_standard.py first."
        )
    return GoldStandard.load(path)


__all__ = ["GoldPaper", "GoldStandard", "load_gold_standard"]
