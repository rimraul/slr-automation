"""Extraction field-specification contract -- the ONE thing every provider
(OpenAI now, a local-model runner later) shares (see this package's own
`__init__.py` module docstring for the full architecture rationale: the
scientific contract is shared, execution is provider-specific and never
auto-detected/silently substituted).

A fixed, transparent Pydantic schema, not arbitrary imported Python --
`charting.py`'s `cfg.charting.schema_path` (`importlib.util.spec_from_file_
location` executing a caller-supplied `.py` file to get an `ExtractionSchema`
class) is exactly the code-execution surface this package does not carry
forward. An `ExtractionSchema` here is DATA: load it from JSON, validate it,
done -- nothing in this module ever imports or executes a caller-supplied
module.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_INTEGER_RE = re.compile(r"^-?\d+$")


class ValueKind(str, Enum):
    """The value shapes a field can take. Deliberately small and closed --
    a v1 teaching schema, not a general type system."""

    TEXT = "text"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    TEXT_LIST = "text_list"


class FieldOrigin(str, Enum):
    """Where a field/value-list came from -- required per-field provenance
    (lit_review_protocol addition): a reader should always be able to tell
    whether a controlled-value list reflects an a priori theoretical
    framework, was built up inductively while reading the corpus, or is a
    deliberate mix of both (and therefore needs its own disclosed
    limitations)."""

    THEORY_DERIVED = "theory_derived"
    INDUCTIVELY_DERIVED = "inductively_derived"
    DUAL_SCHEMA = "dual_schema"


class AllowedValue(BaseModel):
    """One controlled value AND its definition -- lit_review_protocol's
    explicit requirement: "require every controlled value and its
    definition in the actual model-visible prompt." A bare enum of value
    strings with no definition is exactly what this model refuses to be;
    `prompts.py`'s builder reads `definition` directly into the prompt, so
    an empty/missing definition here is a schema-authoring bug caught at
    validation time, not discovered later in a malformed prompt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class FieldProvenance(BaseModel):
    """Per-field provenance block (lit_review_protocol addition: "The
    extraction output should consequently include a small provenance
    block per field specification, not just its label and question").
    `version` is THIS FIELD's own version -- deliberately independent of
    `ExtractionSchema.version` (the whole schema's version), so revising
    one field's definition after a calibration disagreement doesn't force
    every other, unchanged field to claim a new version too. `rationale`
    is free text: why this field exists / what prompted it (e.g. "added
    after calibration round 2 found systematic disagreement on X" --
    lit_review_protocol's "version any prompt/schema revision motivated
    by disagreements" requirement lives here, not as a side note).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: FieldOrigin
    version: str = Field(min_length=1)
    rationale: str | None = None


class ExtractionFieldSpec(BaseModel):
    """One field of a versioned extraction schema. `field_id` is a stable,
    safe token -- it is what evidence/results/checkpoints key on, and MUST
    stay stable across schema versions for the same conceptual field (a
    field rename is a NEW field_id, and the old one's results simply stop
    appearing -- there is no implicit renaming/migration in v1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value_kind: ValueKind
    allowed_values: tuple[AllowedValue, ...] | None = None
    cardinality: Literal["single", "multiple"]
    provenance: FieldProvenance

    @model_validator(mode="after")
    def _validate_field_id(self) -> "ExtractionFieldSpec":
        if not _SAFE_ID_RE.match(self.field_id):
            raise ValueError(
                f"field_id {self.field_id!r} is not a safe token "
                "([A-Za-z0-9_-], 1-128 chars, no path separators/dots)"
            )
        return self

    @model_validator(mode="after")
    def _validate_allowed_values_match_kind(self) -> "ExtractionFieldSpec":
        if self.value_kind == ValueKind.CATEGORICAL:
            if not self.allowed_values:
                raise ValueError("value_kind='categorical' requires a non-empty allowed_values list")
            values = [av.value for av in self.allowed_values]
            if len(set(values)) != len(values):
                raise ValueError(f"allowed_values contains duplicate values: {values}")
        elif self.allowed_values is not None:
            raise ValueError(
                f"allowed_values is only meaningful for value_kind='categorical', "
                f"got value_kind={self.value_kind.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_cardinality_matches_kind(self) -> "ExtractionFieldSpec":
        """The value-kind/cardinality compatibility matrix (explicit
        design decision, not left implicit): `text_list` is inherently a
        list of separately-supported strings, so it must be
        `cardinality="multiple"` -- a single-valued `text_list` field is a
        schema-authoring bug caught here, not a silently-accepted
        confusion with plain `text`. `categorical` MAY be single or
        multiple (a multi-select controlled-vocabulary field, e.g. "study
        designs used", is a legitimate single field with several
        supported values) -- no restriction. `integer`/`number`/`boolean`
        similarly have no cardinality restriction; only their ACCEPTED
        LEXICAL FORMS are constrained, by `validate_value_for_field`
        below, not their cardinality."""
        if self.value_kind == ValueKind.TEXT_LIST and self.cardinality != "multiple":
            raise ValueError("value_kind='text_list' requires cardinality='multiple'")
        return self


class ExtractionSchema(BaseModel):
    """A versioned, complete set of extraction fields for one review's
    charting stage. Loaded from a plain JSON file
    (`ExtractionSchema.model_validate(json.load(...))`) -- never Python
    code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1)
    fields: tuple[ExtractionFieldSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_schema_id(self) -> "ExtractionSchema":
        if not _SAFE_ID_RE.match(self.schema_id):
            raise ValueError(f"schema_id {self.schema_id!r} is not a safe token")
        return self

    @model_validator(mode="after")
    def _unique_field_ids(self) -> "ExtractionSchema":
        ids = [f.field_id for f in self.fields]
        if len(set(ids)) != len(ids):
            dupes = sorted({fid for fid in ids if ids.count(fid) > 1})
            raise ValueError(f"duplicate field_id(s) in schema: {dupes}")
        return self

    def field_map(self) -> dict[str, ExtractionFieldSpec]:
        return {f.field_id: f for f in self.fields}


def load_extraction_schema(path: str | Path) -> ExtractionSchema:
    """The ONLY supported way to load a schema for extraction -- reads and
    validates plain JSON. There is deliberately no equivalent of
    `charting.py`'s `schema_path` (arbitrary `.py` import) anywhere in
    this package."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExtractionSchema.model_validate(data)


TypedValue = Union[str, int, float, bool]


def validate_value_for_field(field_spec: ExtractionFieldSpec, value: str) -> tuple[bool, Optional[TypedValue]]:
    """The value-validation boundary (review finding 3): checks a raw
    model-returned string against `field_spec.value_kind` (and, for
    categorical fields, `allowed_values`) BEFORE that value is ever
    allowed to become a "found" observation. Returns
    `(is_valid, typed_value)`:
      - `is_valid=False` -- the caller must route this to review (never a
        clean extraction); `typed_value` is always `None` in this case.
      - `is_valid=True` -- `typed_value` is the canonicalized typed form
        (`int`/`float`/`bool` for integer/number/boolean; the matched
        string itself for categorical) when one is meaningful, or `None`
        for `text`/`text_list` (the raw string already IS the value --
        there is no separate "typed" form to compute, and no
        canonicalization/rewriting happens here per requirement 9).

    The RAW string is never discarded or rewritten by this function --
    callers keep it as `ExtractedValue.value` regardless of `typed_value`
    (review constraint: "Preserve the raw string alongside any validated
    typed representation").

    Accepted lexical forms (the compatibility matrix, explicit and
    closed, not left to caller convention):
      - categorical: EXACT (case-sensitive) match against one of
        `allowed_values`' own `.value` strings -- never a fuzzy or
        case-folded match, consistent with this whole package's
        exact-string-equality convention.
      - boolean: `"true"`/`"false"`, case-insensitive after stripping
        whitespace -- nothing else (`"1"`/`"0"`/`"yes"`/`"no"` are NOT
        accepted; a model returning those fails validation and is routed
        to review, not silently coerced).
      - integer: an optional leading `-` followed by one or more ASCII
        digits, after stripping whitespace -- no thousands separators, no
        decimal point.
      - number: anything Python's own `float()` accepts after stripping
        whitespace (covers integers, decimals, and scientific notation).
      - text / text_list: any non-blank string (after stripping) --
        there is no further constraint; a `text_list` field's
        `cardinality="multiple"` requirement is enforced structurally on
        `ExtractionFieldSpec` itself, not here.
    """
    stripped = value.strip()

    if field_spec.value_kind == ValueKind.CATEGORICAL:
        allowed = {av.value for av in field_spec.allowed_values or ()}
        if value in allowed:
            return True, value
        return False, None

    if field_spec.value_kind == ValueKind.BOOLEAN:
        lowered = stripped.lower()
        if lowered == "true":
            return True, True
        if lowered == "false":
            return True, False
        return False, None

    if field_spec.value_kind == ValueKind.INTEGER:
        if _INTEGER_RE.match(stripped):
            return True, int(stripped)
        return False, None

    if field_spec.value_kind == ValueKind.NUMBER:
        try:
            parsed = float(stripped)
        except ValueError:
            return False, None
        if not math.isfinite(parsed):
            # float() also accepts "nan"/"inf"/"-infinity" -- not a
            # reportable scientific value in a data-extraction context
            # (a paper never "reports NaN" as a finding); reject rather
            # than silently accepting a non-finite typed value.
            return False, None
        return True, parsed

    # TEXT / TEXT_LIST: any non-blank string, no further constraint.
    return (bool(stripped), None)
