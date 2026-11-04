"""Deterministic one-to-one matching of pipeline records to gold papers."""

from __future__ import annotations

from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from slr_pipeline.doi import normalise_doi, normalise_title
from slr_pipeline.state import Record

from .adjudication import (
    AdjudicationRecord,
    CandidatePairScore,
    build_pending_adjudication,
)
from .gold_standard import GoldPaper, GoldStandard
from .reproduction_spec import normalize_database_name


MatchTier = Literal["doi_exact", "other_persistent_id_exact", "title_exact", "adjudicated"]


class MatchedPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_paper_id: str
    pipeline_record_id: str
    match_method: MatchTier
    score: float = Field(ge=0.0, le=1.0)


class DuplicateManifestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_record_id: str
    duplicate_manifestation_of: str
    match_method: Literal[
        "doi_exact", "other_persistent_id_exact", "title_exact", "fuzzy_title"
    ]
    score: float = Field(ge=0.0, le=1.0)


class GoldMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matched_pairs: tuple[MatchedPair, ...]
    unmatched_gold_paper_ids: tuple[str, ...]
    unmatched_pipeline_record_ids: tuple[str, ...]
    adjudications: tuple[AdjudicationRecord, ...]
    duplicate_manifestations: tuple[DuplicateManifestation, ...]


class _Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_id: str
    record_id: str
    score: float
    corroborating_fields: tuple[Literal["year", "first_author_surname"], ...] = ()


def _records(values: Sequence[Record | Mapping[str, Any]]) -> dict[str, Record]:
    records: dict[str, Record] = {}
    for value in values:
        if isinstance(value, Mapping) and not str(value.get("id", "")).strip():
            raise ValueError("pipeline records used for reproduction matching require stable ids")
        record = value if isinstance(value, Record) else Record.model_validate(value)
        if not record.id.strip():
            raise ValueError("pipeline records used for reproduction matching require stable ids")
        if record.id in records:
            raise ValueError(f"duplicate pipeline record id {record.id!r}")
        records[record.id] = record
    return records


def _persistent_id_matches(gold: GoldPaper, record: Record) -> bool:
    if not record.source or not record.source_id:
        return False
    source = normalize_database_name(record.source)
    return any(
        normalize_database_name(gold_source) == source
        and persistent_id == record.source_id
        for gold_source, persistent_id in gold.other_persistent_ids.items()
    )


def _author_tokens(value: str) -> set[str]:
    tokens = [
        token
        for token in normalise_title(value).split()
        if len(token) > 1 and token not in {"et", "al"}
    ]
    if not tokens:
        return set()
    return {tokens[0], tokens[-1]}


def _corroborating_fields(
    gold: GoldPaper, record: Record
) -> tuple[Literal["year", "first_author_surname"], ...]:
    fields: list[Literal["year", "first_author_surname"]] = []
    if gold.year is not None and record.year is not None and gold.year == record.year:
        fields.append("year")
    if gold.authors and record.authors:
        if _author_tokens(gold.authors) & _author_tokens(record.authors[0]):
            fields.append("first_author_surname")
    return tuple(fields)


def _connected_components(edges: Iterable[_Edge]) -> list[tuple[_Edge, ...]]:
    edge_list = tuple(edges)
    by_gold: dict[str, list[_Edge]] = defaultdict(list)
    by_record: dict[str, list[_Edge]] = defaultdict(list)
    for edge in edge_list:
        by_gold[edge.gold_id].append(edge)
        by_record[edge.record_id].append(edge)

    unseen = {(edge.gold_id, edge.record_id) for edge in edge_list}
    components: list[tuple[_Edge, ...]] = []
    while unseen:
        start = min(unseen)
        queue: deque[tuple[str, str]] = deque([start])
        component_pairs: set[tuple[str, str]] = set()
        while queue:
            pair = queue.popleft()
            if pair in component_pairs:
                continue
            component_pairs.add(pair)
            gold_id, record_id = pair
            for edge in by_gold[gold_id] + by_record[record_id]:
                other = (edge.gold_id, edge.record_id)
                if other not in component_pairs:
                    queue.append(other)
        unseen -= component_pairs
        by_pair = {(edge.gold_id, edge.record_id): edge for edge in edge_list}
        components.append(tuple(by_pair[pair] for pair in sorted(component_pairs)))
    return sorted(
        components,
        key=lambda component: tuple(
            (edge.gold_id, edge.record_id) for edge in component
        ),
    )


def _unique_maximum_matching(edges: tuple[_Edge, ...]) -> tuple[_Edge, ...] | None:
    """Return the unique maximum-cardinality assignment, else ``None``.

    Exact-tier collision components are normally tiny. Components larger than
    twelve vertices fail conservatively into adjudication instead of risking an
    exponential search or an arbitrary solver tie-break.
    """

    gold_ids = sorted({edge.gold_id for edge in edges})
    record_ids = {edge.record_id for edge in edges}
    if len(gold_ids) + len(record_ids) > 12:
        return None
    by_gold: dict[str, list[_Edge]] = defaultdict(list)
    for edge in edges:
        by_gold[edge.gold_id].append(edge)
    for values in by_gold.values():
        values.sort(key=lambda edge: edge.record_id)

    best_size = -1
    best_solutions: set[tuple[tuple[str, str], ...]] = set()

    def visit(index: int, used: set[str], chosen: list[_Edge]) -> None:
        nonlocal best_size
        if len(chosen) + len(gold_ids) - index < best_size:
            return
        if index == len(gold_ids):
            pairs = tuple(sorted((edge.gold_id, edge.record_id) for edge in chosen))
            size = len(pairs)
            if size > best_size:
                best_size = size
                best_solutions.clear()
            if size == best_size and len(best_solutions) < 2:
                best_solutions.add(pairs)
            return
        gold_id = gold_ids[index]
        visit(index + 1, used, chosen)
        for edge in by_gold[gold_id]:
            if edge.record_id in used:
                continue
            used.add(edge.record_id)
            chosen.append(edge)
            visit(index + 1, used, chosen)
            chosen.pop()
            used.remove(edge.record_id)

    visit(0, set(), [])
    if len(best_solutions) != 1:
        return None
    only = next(iter(best_solutions))
    edge_map = {(edge.gold_id, edge.record_id): edge for edge in edges}
    return tuple(edge_map[pair] for pair in only)


def _score_entries(edges: tuple[_Edge, ...]) -> tuple[CandidatePairScore, ...]:
    return tuple(
        CandidatePairScore(
            gold_paper_id=edge.gold_id,
            pipeline_record_id=edge.record_id,
            score=edge.score,
            corroborating_fields=edge.corroborating_fields,
        )
        for edge in edges
    )


def _duplicate_hit(
    gold: GoldPaper, record: Record
) -> tuple[str, float] | None:
    gold_doi = normalise_doi(gold.doi)
    record_doi = normalise_doi(record.doi)
    if gold_doi and record_doi and gold_doi == record_doi:
        return "doi_exact", 1.0
    if _persistent_id_matches(gold, record):
        return "other_persistent_id_exact", 1.0
    gold_title = normalise_title(gold.title)
    record_title = normalise_title(record.title)
    if gold_title and record_title and gold_title == record_title:
        return "title_exact", 1.0
    if gold_title and record_title:
        ratio = SequenceMatcher(None, gold_title, record_title).ratio()
        if ratio >= 0.85 and _corroborating_fields(gold, record):
            return "fuzzy_title", ratio
    return None


def match_records_to_gold(
    records: Sequence[Record | Mapping[str, Any]],
    gold_standard: GoldStandard,
    *,
    run_id: str,
    adjudication_projection: Mapping[str, AdjudicationRecord] | None = None,
) -> GoldMatchResult:
    """Match records one-to-one and surface every real ambiguity for review."""

    if gold_standard.structurally_ambiguous:
        raise ValueError(
            "structured per-paper matching is unavailable for a structurally "
            "ambiguous legacy gold standard"
        )
    if not run_id.strip():
        raise ValueError("run_id must not be blank")

    gold = {paper.paper_id: paper for paper in gold_standard.included_papers}
    pipeline = _records(records)
    unmatched_gold = set(gold)
    unmatched_records = set(pipeline)
    matches: list[MatchedPair] = []
    adjudications: list[AdjudicationRecord] = []
    blocked_gold: set[str] = set()
    blocked_records: set[str] = set()
    projection = dict(adjudication_projection or {})

    for record in projection.values():
        if record.run_id != run_id:
            raise ValueError("adjudication projection contains a different run_id")
        if record.resolution != "matched":
            continue
        gold_id = record.resolved_gold_paper_id
        pipeline_id = record.resolved_pipeline_record_id
        if gold_id in unmatched_gold and pipeline_id in unmatched_records:
            matches.append(
                MatchedPair(
                    gold_paper_id=gold_id,
                    pipeline_record_id=pipeline_id,
                    match_method="adjudicated",
                    score=next(
                        score.score
                        for score in record.scores
                        if score.gold_paper_id == gold_id
                        and score.pipeline_record_id == pipeline_id
                    ),
                )
            )
            unmatched_gold.remove(gold_id)
            unmatched_records.remove(pipeline_id)

    exact_tiers = (
        (
            "doi_collision",
            "doi_exact",
            lambda paper, record: bool(
                normalise_doi(paper.doi)
                and normalise_doi(paper.doi) == normalise_doi(record.doi)
            ),
        ),
        (
            "other_persistent_id_collision",
            "other_persistent_id_exact",
            _persistent_id_matches,
        ),
        (
            "exact_title_collision",
            "title_exact",
            lambda paper, record: bool(
                normalise_title(paper.title)
                and normalise_title(paper.title) == normalise_title(record.title)
            ),
        ),
    )

    for adjudication_tier, match_method, predicate in exact_tiers:
        edges = tuple(
            _Edge(gold_id=gold_id, record_id=record_id, score=1.0)
            for gold_id in sorted(unmatched_gold - blocked_gold)
            for record_id in sorted(unmatched_records - blocked_records)
            if predicate(gold[gold_id], pipeline[record_id])
        )
        for component in _connected_components(edges):
            assignment = _unique_maximum_matching(component)
            if assignment is not None:
                for edge in assignment:
                    if edge.gold_id not in unmatched_gold or edge.record_id not in unmatched_records:
                        continue
                    matches.append(
                        MatchedPair(
                            gold_paper_id=edge.gold_id,
                            pipeline_record_id=edge.record_id,
                            match_method=match_method,
                            score=1.0,
                        )
                    )
                    unmatched_gold.remove(edge.gold_id)
                    unmatched_records.remove(edge.record_id)
                continue

            pending = build_pending_adjudication(
                run_id=run_id,
                tier=adjudication_tier,
                scores=_score_entries(component),
            )
            prior = projection.get(pending.adjudication_id)
            if prior is None or prior.resolution == "pending":
                adjudications.append(prior or pending)
                blocked_gold.update(pending.candidate_gold_paper_ids)
                blocked_records.update(pending.candidate_pipeline_record_ids)
            # A resolved no_match suppresses this component at this tier. A
            # resolved match was applied before tier matching.

    fuzzy_edges = tuple(
        _Edge(
            gold_id=gold_id,
            record_id=record_id,
            score=ratio,
            corroborating_fields=_corroborating_fields(
                gold[gold_id], pipeline[record_id]
            ),
        )
        for gold_id in sorted(unmatched_gold - blocked_gold)
        for record_id in sorted(unmatched_records - blocked_records)
        if (ratio := SequenceMatcher(
            None,
            normalise_title(gold[gold_id].title),
            normalise_title(pipeline[record_id].title),
        ).ratio()) >= 0.85
        and normalise_title(gold[gold_id].title)
        and normalise_title(pipeline[record_id].title)
    )
    for component in _connected_components(fuzzy_edges):
        pending = build_pending_adjudication(
            run_id=run_id,
            tier="fuzzy_title",
            scores=_score_entries(component),
        )
        prior = projection.get(pending.adjudication_id)
        if prior is None or prior.resolution == "pending":
            adjudications.append(prior or pending)

    matched_gold_ids = {pair.gold_paper_id for pair in matches}
    duplicate_manifestations: list[DuplicateManifestation] = []
    for record_id in sorted(unmatched_records - blocked_records):
        candidates: list[tuple[float, str, str]] = []
        for gold_id in sorted(matched_gold_ids):
            hit = _duplicate_hit(gold[gold_id], pipeline[record_id])
            if hit:
                method, score = hit
                candidates.append((score, gold_id, method))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        best_score = candidates[0][0]
        best = [candidate for candidate in candidates if candidate[0] == best_score]
        if len(best) != 1:
            continue
        score, gold_id, method = best[0]
        duplicate_manifestations.append(
            DuplicateManifestation(
                pipeline_record_id=record_id,
                duplicate_manifestation_of=gold_id,
                match_method=method,
                score=score,
            )
        )

    return GoldMatchResult(
        matched_pairs=tuple(
            sorted(matches, key=lambda pair: (pair.gold_paper_id, pair.pipeline_record_id))
        ),
        unmatched_gold_paper_ids=tuple(sorted(unmatched_gold)),
        unmatched_pipeline_record_ids=tuple(sorted(unmatched_records)),
        adjudications=tuple(
            sorted(adjudications, key=lambda record: record.adjudication_id)
        ),
        duplicate_manifestations=tuple(duplicate_manifestations),
    )


__all__ = [
    "DuplicateManifestation",
    "GoldMatchResult",
    "MatchedPair",
    "match_records_to_gold",
]
