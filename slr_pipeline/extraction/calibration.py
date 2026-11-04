"""Two DELIBERATELY SEPARATE calibration contracts (review refinement:
"Keep reusable human ground truth separate from run-specific evidence
auditing... Otherwise changing model output invalidates the calibration
corpus itself"):

  - `CalibrationCorpus`: reusable human ground truth -- correct status,
    correct values, and REFERENCE evidence (what a human reviewer
    actually found in the publication, independent of any system run).
    Never invalidated by a new model, a new run, or a changed prompt --
    it describes the PUBLICATION, not any particular extraction attempt.
    This is what makes evidence RECALL measurable: did the system find
    evidence a human reviewer already knows exists?

  - `EvidenceAudit`: run-bound judgments about the SYSTEM's own returned
    quotations for one specific checkpoint (bound by content hash) --
    does each system-returned quotation actually SUPPORT its claimed
    value, and is its page/section locator correct? This is what makes
    evidence PRECISION measurable, and it must be re-done whenever the
    system's own output changes; it is explicitly NOT part of the
    reusable corpus.

Neither module ships populated data: selecting real, deliberately
difficult academic papers and writing real human judgments against them
is work only a human reviewer can do (the same reason FT_SCREENING's own
GoldSet was built through an interactive, chat-driven, row-by-row human
review rather than fabricated). What this module provides is the
CONTAINER that work goes into, plus worksheet generators that turn real
extraction output into human-reviewable rows.

See this package's PROTOCOL.md for the staged calibration ladder
(n=1 -> n~5-10 deliberately difficult cases -> n~20-30 -> n~50-100 -> full
run) `CalibrationCorpus` is meant to support.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .checkpoint import DocumentCheckpoint
from .results import DocumentExtractionResult, EvidenceSpanModel, FieldStatus
from .schema import ExtractionSchema

DifficultyCategory = Literal[
    "plumbing",  # n=1 -- nothing methodologically interesting, just proves the pipeline runs
    "decisive_late_page",
    "genuinely_not_reported",
    "contradictory_values",
    "multilingual",
    "table_figure_dependent",
    "long_document",
    "prompt_injection",
    "repeated_values_different_sections",
    "routine",  # a case with no special difficulty -- a calibration corpus needs a baseline too
]


class ReferenceEvidence(BaseModel):
    """Human-identified evidence FROM THE PUBLICATION supporting one
    correct value -- independent of any system run. `supports_value`
    names which one of the judgment's own `correct_values` this specific
    span backs (a judgment may cite several spans across several
    values)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    excerpt: str = Field(min_length=1)
    supports_value: str = Field(min_length=1)
    page: Optional[str] = None
    section: Optional[str] = None


class FieldJudgment(BaseModel):
    """One human judgment about ONE field for ONE calibration document --
    the ground truth `reports.py`'s audited-accuracy section compares
    extraction output against. `correct_values`/`correct_status` are
    the human reviewer's OWN determination from reading the actual
    publication -- never derived from, or compared against, another AI's
    output (this project's own explicit rule: "AI-vs-AI or OpenAI-vs-local
    agreement" is agreement, not correctness).

    Coherence between `correct_status`, `correct_values`, and
    `reference_evidence` is enforced STRUCTURALLY (construction fails,
    not just a property callers might forget to check -- review finding
    5: "invalid judgments construct successfully because is_valid is
    only a property")."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str
    correct_status: FieldStatus
    correct_values: tuple[str, ...] = ()
    reference_evidence: tuple[ReferenceEvidence, ...] = ()
    reviewer: str = Field(min_length=1)
    note: Optional[str] = None

    @model_validator(mode="after")
    def _values_match_status(self) -> "FieldJudgment":
        if self.correct_status in (FieldStatus.EXTRACTED, FieldStatus.CONFLICTING):
            if not self.correct_values:
                raise ValueError(
                    f"correct_status={self.correct_status.value!r} requires at least one correct_value"
                )
        else:
            if self.correct_values:
                raise ValueError(f"correct_status={self.correct_status.value!r} must not carry correct_values")
            if self.reference_evidence:
                raise ValueError(f"correct_status={self.correct_status.value!r} must not carry reference_evidence")
        return self

    @model_validator(mode="after")
    def _reference_evidence_supports_a_correct_value(self) -> "FieldJudgment":
        if self.reference_evidence:
            known_values = set(self.correct_values)
            unknown = sorted({e.supports_value for e in self.reference_evidence if e.supports_value not in known_values})
            if unknown:
                raise ValueError(
                    f"reference_evidence.supports_value {unknown} not present in correct_values {sorted(known_values)}"
                )
        return self


class CalibrationCase(BaseModel):
    """One document in the calibration corpus. `pdf_sha256` binds the
    case to a SPECIFIC PDF's content, not just a path string (review
    refinement) -- a corpus case survives a file being moved, but not the
    underlying document being silently swapped for a different one at
    the same path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    pdf_path: str
    pdf_sha256: str = Field(min_length=64, max_length=64)
    difficulty_category: DifficultyCategory
    notes: Optional[str] = None
    field_judgments: tuple[FieldJudgment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_field_ids(self) -> "CalibrationCase":
        ids = [j.field_id for j in self.field_judgments]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate field_id(s) in field_judgments: {dupes}")
        return self

    def judgment_map(self) -> dict[str, FieldJudgment]:
        return {j.field_id: j for j in self.field_judgments}


class CalibrationBindingError(ValueError):
    """A `CalibrationCorpus`'s own `extraction_schema_hash` or a
    `CalibrationCase`'s own `pdf_sha256` does not match the audited
    run/checkpoint it is being compared against. Raised by
    `reports.build_audited_accuracy_report()` and
    `reports.build_evidence_quality_report()` BEFORE any metric is
    computed -- review finding I: storing these bindings but never
    enforcing them "turns the fields described as scientific identity
    bindings into passive metadata: an audit against a revised schema or
    substituted PDF can look perfect." An incompatible corpus/run pairing
    must fail explicitly, never be silently compared."""


class CalibrationCorpus(BaseModel):
    """A versioned, stratified set of calibration cases for one
    extraction schema. `extraction_schema_hash` binds the corpus to the
    EXACT schema content (review refinement) -- a schema revision that
    keeps the same `schema_id`/`version` label but changes field
    definitions would otherwise silently invalidate old judgments without
    that being detectable. `version` must be bumped whenever a case is
    added/changed -- comparing extraction runs against two different
    corpus versions is a caller error this model cannot prevent by
    itself, but every report this package produces records which corpus
    version it was audited against (see `reports.py`).

    `expected_field_ids` freezes the schema's OWN field_id set at corpus-
    authoring time (review finding N: "`CalibrationCorpus` binds a schema
    hash but does not retain the schema's expected field IDs"). Every
    case must judge EXACTLY this set -- neither fewer (a cherry-picked
    subset that could make `coverage_complete=True` describe only the
    easy fields) nor more (a judgment for a field the schema doesn't
    have). This models the ORDINARY, full-calibration path only:
    deliberately partial/subset calibration against a smaller declared
    matrix is out of scope for this v1 model -- a corpus either judges
    every field of the schema it is bound to, or it does not construct.

    `content_hash` is a canonical hash of the corpus's own complete
    content (review finding N: "Persist a canonical corpus content hash
    (not only caller-maintained ID/version labels)") -- always freshly
    computed from the live object (never a stored, independently-
    tamperable field), so it cannot itself go stale or be forged; reports
    bind it directly (see `reports.py`) so a reader can tell exactly
    which corpus content a report was actually audited against, not just
    a caller-typed id/version label that might not match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str
    version: str
    extraction_schema_id: str
    extraction_schema_version: str
    extraction_schema_hash: str = Field(min_length=64, max_length=64)
    expected_field_ids: tuple[str, ...] = Field(min_length=1)
    cases: tuple[CalibrationCase, ...] = Field(min_length=1)

    @classmethod
    def build(
        cls, *, corpus_id: str, version: str, schema: ExtractionSchema, cases: tuple[CalibrationCase, ...],
    ) -> "CalibrationCorpus":
        """The RECOMMENDED way to construct a `CalibrationCorpus`:
        derives `extraction_schema_id`/`extraction_schema_version`/
        `extraction_schema_hash`/`expected_field_ids` ALL from a real
        `ExtractionSchema` object, rather than accepting independently-
        authored copies that could drift from what the schema actually
        contains (review finding N-residual: "`expected_field_ids`
        remains independent of the schema it claims to describe... The
        original cherry-picking failure therefore still exists with one
        additional self-consistent label"). Direct
        `CalibrationCorpus(...)` construction remains available --
        chiefly for deserializing a previously-built, already-derived
        corpus, which does not need a live `ExtractionSchema` object at
        load time -- but authoring a NEW corpus should always go through
        here."""
        from .hashing import sha256_hex

        return cls(
            corpus_id=corpus_id, version=version,
            extraction_schema_id=schema.schema_id, extraction_schema_version=schema.version,
            extraction_schema_hash=sha256_hex(schema.model_dump(mode="json")),
            expected_field_ids=tuple(f.field_id for f in schema.fields),
            cases=cases,
        )

    @model_validator(mode="after")
    def _unique_document_ids(self) -> "CalibrationCorpus":
        ids = [c.document_id for c in self.cases]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate document_id(s) in cases: {dupes}")
        return self

    @model_validator(mode="after")
    def _expected_field_ids_are_unique(self) -> "CalibrationCorpus":
        if len(set(self.expected_field_ids)) != len(self.expected_field_ids):
            raise ValueError("expected_field_ids must not contain duplicates")
        return self

    @model_validator(mode="after")
    def _every_case_covers_exactly_the_expected_fields(self) -> "CalibrationCorpus":
        """Review finding N's central adversarial reproduction: "With a
        two-field extraction schema/checkpoint and a corpus judging only
        one easy field, the report returned coverage_complete=True and
        accuracy 1.0." Structurally impossible once every case is
        required to judge EXACTLY `expected_field_ids` -- neither a
        cherry-picked subset nor an unknown extra field_id."""
        expected = set(self.expected_field_ids)
        for case in self.cases:
            judged = {j.field_id for j in case.field_judgments}
            missing, extra = expected - judged, judged - expected
            if missing or extra:
                raise ValueError(
                    f"case document_id={case.document_id!r} does not judge exactly "
                    f"expected_field_ids -- missing={sorted(missing)!r} extra={sorted(extra)!r}"
                )
        return self

    @property
    def content_hash(self) -> str:
        from .hashing import sha256_hex

        return sha256_hex(self.model_dump(mode="json"))

    @property
    def difficulty_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.difficulty_category] = counts.get(case.difficulty_category, 0) + 1
        return counts

    def case_map(self) -> dict[str, CalibrationCase]:
        return {c.document_id: c for c in self.cases}


def build_calibration_worksheet_rows(result: DocumentExtractionResult) -> list[dict]:
    """Turns ONE document's real extraction output into human-reviewable
    worksheet rows -- one row per field, with the system's own extracted
    status/value(s)/evidence, and BLANK columns for a human to fill in
    (`human_correct_status`, `human_correct_values`, `human_reviewer`,
    `human_note`). Per the protocol requirement: "it must assess
    individual values and supporting evidence -- not merely whether each
    document produced JSON" -- this is a per-FIELD worksheet, not a
    per-document one."""
    rows: list[dict] = []
    for field in result.fields:
        values_text = "; ".join(v.value for v in field.values) if field.values else ""
        evidence_text = " | ".join(
            f"[{e.chunk_id} p.{e.page or '?'}] {e.excerpt}" for v in field.values for e in v.evidence
        )
        rows.append({
            "document_id": result.document_id,
            "field_id": field.field_id,
            "system_status": field.status.value,
            "system_values": values_text,
            "system_confidence": field.confidence or "",
            "system_evidence": evidence_text,
            "system_requires_human_review": field.requires_human_review,
            "human_correct_status": "",
            "human_correct_values": "",
            "human_reviewer": "",
            "human_note": "",
        })
    return rows


class QuotationJudgment(BaseModel):
    """One human judgment about ONE system-returned evidence span for one
    field/claimed-value of one document, bound (via
    `EvidenceAudit.checkpoint_content_hash`) to the EXACT system output it
    was reviewed against. `claimed_value` is the value this quotation was
    offered as evidence FOR (`""` for a `review_evidence` span, which is
    not evidence for any specific value) -- review finding B: "evidence
    support is a relation between a quotation and a claim; dropping the
    claim makes the frozen judgment ambiguous and can collapse two uses
    of the same quotation for different values."

    `supports_claimed_value` is `Optional[bool]`, not a plain `bool`
    (review finding F): a review-only span (`claimed_value == ""`) is not
    evidence FOR any claim at all, so "does it support the claim" is not
    a meaningful question for it -- `None` means exactly that
    inapplicability, structurally distinct from `False` ("it was judged
    and found NOT to support its claim"). A claim-bearing span
    (`claimed_value != ""`) must always receive an explicit `True`/`False`
    verdict -- `None` is not allowed to mean "not yet judged" there.
    Exact-substring verification (`evidence.verify_evidence`) only proves
    the text EXISTS, never that it supports the claim (review finding 5)
    -- this field is the human judgment closing that gap.

    `locator_correct` remains a plain `bool` -- every span, claim-bearing
    or not, was attributed to SOME page/section, so locator correctness
    is always a meaningful question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str
    claimed_value: str
    chunk_id: str
    excerpt: str
    supports_claimed_value: Optional[bool]
    locator_correct: bool
    reviewer: str = Field(min_length=1)
    note: Optional[str] = None

    @model_validator(mode="after")
    def _supports_claimed_value_matches_claim_presence(self) -> "QuotationJudgment":
        if self.claimed_value == "":
            if self.supports_claimed_value is not None:
                raise ValueError(
                    "a review-only span (claimed_value='') is not evidence for any claim -- "
                    "supports_claimed_value must be None, not a True/False verdict"
                )
        elif self.supports_claimed_value is None:
            raise ValueError(
                f"claimed_value={self.claimed_value!r} requires an explicit supports_claimed_value "
                "(True/False) -- None is only valid for a review-only span"
            )
        return self


class EvidenceAuditIdentityMismatchError(ValueError):
    """An `EvidenceAudit` claims `run_id`/`document_id`/
    `checkpoint_content_hash` that do NOT match the `DocumentCheckpoint`
    it is being reported against. Raised by `reports.build_evidence_quality_report()`
    when re-verifying a directly-deserialized audit (one that bypassed
    `EvidenceAudit.build()`'s own binding-at-construction-time guarantee)
    -- review finding F: "direct `EvidenceAudit(...)` construction
    remains public and `build_evidence_quality_report()` explicitly
    trusts such objects without re-verification." """


class EvidenceAuditCoverageError(ValueError):
    """A proposed `EvidenceAudit`'s `quotation_judgments` do not exactly
    cover every evidence span in the checkpoint's own result (via
    `evidence_projection`). Missing spans would let precision be
    computed over a cherry-picked subset -- review finding B's
    adversarial reproduction: "a one-row audit selecting only one good
    quotation reports precision == 1.0, regardless of additional
    unaudited quotations." Extra/unknown spans would let stale judgments
    from a earlier/different result silently survive into a new audit."""


class EvidenceAudit(BaseModel):
    """Run-bound evidence-quality judgments -- explicitly SEPARATE from
    `CalibrationCorpus` (module docstring). `checkpoint_content_hash`
    binds this audit to the EXACT `checkpoint.DocumentCheckpoint` its
    quotations were reviewed from -- an audit performed against output
    that has since changed (a re-run, a prompt revision) is detectably
    stale rather than silently compared against different system output
    than what was actually reviewed.

    `quotation_judgments` MAY be empty -- a result with zero evidence
    spans at all (e.g. every field `not_reported`) is a real, valid
    "nothing to audit" case (review refinement: "An empty/no-quotation
    system result must remain representable and report zero yield rather
    than making an audit impossible"), not a construction error.

    Direct construction (`EvidenceAudit(...)`) only checks INTERNAL
    judgment-key uniqueness -- it is the path used to DESERIALIZE a
    previously-built, already-coverage-verified audit, and does not by
    itself require a `DocumentExtractionResult` to check against. The
    RECOMMENDED construction path is the `build()` classmethod below,
    which additionally enforces that `quotation_judgments` exactly covers
    the checkpoint's own evidence projection -- neither missing spans
    (an incomplete, cherry-picked audit) nor extra ones (stale judgments
    from different output)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: str
    run_id: str
    document_id: str
    checkpoint_content_hash: str = Field(min_length=64, max_length=64)
    quotation_judgments: tuple[QuotationJudgment, ...] = ()

    @model_validator(mode="after")
    def _unique_judgment_keys(self) -> "EvidenceAudit":
        keys = [(j.field_id, j.claimed_value, j.chunk_id, j.excerpt) for j in self.quotation_judgments]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate (field_id, claimed_value, chunk_id, excerpt) judgment in quotation_judgments")
        return self

    @classmethod
    def build(
        cls, *, audit_id: str, checkpoint: DocumentCheckpoint, quotation_judgments: tuple[QuotationJudgment, ...],
    ) -> "EvidenceAudit":
        """The RECOMMENDED way to construct an `EvidenceAudit`: takes a
        VERIFIED `DocumentCheckpoint` directly (never independent
        `run_id`/`document_id`/`checkpoint_content_hash` copies -- review
        finding F: "`EvidenceAudit.build()` accepts independent `run_id`,
        `document_id`, and `checkpoint_content_hash` arguments without
        verifying them against the supplied result/checkpoint"). Identity
        is READ from the checkpoint, not supplied and cross-checked --
        there is no longer any parameter through which a caller could
        pass mismatched metadata. Also verifies `quotation_judgments`
        exactly covers `checkpoint.result`'s own evidence projection (via
        `evidence_projection`, the SAME function that generates the
        human worksheet), raising `EvidenceAuditCoverageError` on any
        missing or extra span. Also calls `checkpoint.verify()` FIRST --
        review finding H: a `DocumentCheckpoint` object being passed in
        does not by itself prove it is a trustworthy artifact (a
        deserialized checkpoint with a tampered `content_hash` still
        constructs successfully as a Pydantic object); this call is what
        actually establishes that."""
        checkpoint.verify()
        expected = {
            (field_id, claimed_value, evidence.chunk_id, evidence.excerpt)
            for field_id, claimed_value, evidence in evidence_projection(checkpoint.result)
        }
        provided = {(j.field_id, j.claimed_value, j.chunk_id, j.excerpt) for j in quotation_judgments}
        missing = expected - provided
        extra = provided - expected
        if missing or extra:
            raise EvidenceAuditCoverageError(
                f"quotation_judgments do not exactly cover this result's evidence spans -- "
                f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
            )
        return cls(
            audit_id=audit_id, run_id=checkpoint.run_id, document_id=checkpoint.document_id,
            checkpoint_content_hash=checkpoint.content_hash, quotation_judgments=tuple(quotation_judgments),
        )

    @property
    def precision(self) -> float:
        """Fraction of judged, CLAIM-BEARING system quotations that
        actually support their claimed value (evidence PRECISION --
        distinct from `CalibrationCorpus`-based accuracy, which measures
        whether the VALUE was correct, not whether its evidence
        substantively supports it). Review-only spans
        (`supports_claimed_value is None`) are excluded from both the
        numerator AND denominator -- they are not evidence for any claim,
        so they can neither support nor fail to support one (review
        finding F: "exclude non-claim spans from the support-precision
        denominator while retaining them for locator accuracy")."""
        claim_bearing = [j for j in self.quotation_judgments if j.supports_claimed_value is not None]
        if not claim_bearing:
            return 0.0
        supported = sum(1 for j in claim_bearing if j.supports_claimed_value)
        return supported / len(claim_bearing)

    @property
    def locator_accuracy(self) -> float:
        if not self.quotation_judgments:
            return 0.0
        correct = sum(1 for j in self.quotation_judgments if j.locator_correct)
        return correct / len(self.quotation_judgments)


def evidence_projection(result: DocumentExtractionResult) -> list[tuple[str, str, "EvidenceSpanModel"]]:
    """Every `(field_id, claimed_value, evidence_span)` triple a human
    should be able to audit for THIS result -- final `values`,
    `provisional_values` (an INCOMPLETE document's candidate findings --
    review finding B: "the worksheet builder also omits
    provisional_values, despite claiming to cover every span returned by
    the system"), and `review_evidence` (ambiguous spans that were never
    erased just because a value was also found -- see `aggregation.py`).
    `claimed_value` is `""` for `review_evidence` spans, since those are
    not evidence FOR any specific value. This is the SAME projection used
    both to generate the human worksheet and to compute
    `EvidenceAudit.build()`'s required exact coverage -- one definition
    of "every span the system returned," never two that could drift
    apart."""
    projection: list[tuple[str, str, "EvidenceSpanModel"]] = []
    for field in result.fields:
        for value in field.values:
            for evidence in value.evidence:
                projection.append((field.field_id, value.value, evidence))
        for value in field.provisional_values:
            for evidence in value.evidence:
                projection.append((field.field_id, value.value, evidence))
        for evidence in field.review_evidence:
            projection.append((field.field_id, "", evidence))
    return projection


def build_evidence_audit_worksheet_rows(
    *, document_id: str, checkpoint_content_hash: str, result: DocumentExtractionResult,
) -> list[dict]:
    """Turns ONE checkpoint's real extraction output into human-reviewable
    per-QUOTATION worksheet rows (evidence PRECISION review) -- every span
    from `evidence_projection` above, so a reviewer sees every span the
    system actually returned, not just the ones behind a clean answer."""
    rows: list[dict] = []
    for field_id, claimed_value, evidence in evidence_projection(result):
        rows.append({
            "document_id": document_id,
            "checkpoint_content_hash": checkpoint_content_hash,
            "field_id": field_id,
            "claimed_value": claimed_value,
            "chunk_id": evidence.chunk_id,
            "excerpt": evidence.excerpt,
            "page": evidence.page or "",
            "section": evidence.section,
            "supports_claimed_value": "",
            "locator_correct": "",
            "reviewer": "",
            "note": "",
        })
    return rows
