# Structured extraction — protocol section (Phase A)

Filled against `lit_review_protocol/template_v5.md` §3.3's "Structured
extraction" placeholder (`[goal / method / alternatives / output schema /
cards / checks]`), using Card C (LLM structured extraction against a
typed ontology) and Card D (disagreement-driven schema revision) from
§3.2. This document is the "reportable" artifact behind the extraction
MVP — it belongs alongside a future `template_v6.md`'s own "Structured
extraction" section and a registered SLR_chapter protocol instance (see
"Protocol drift found" at the end of this file), not as a substitute for
either.

**Scope note:** this covers Phase A (the extraction engine:
`slr_pipeline/extraction/`) only. No graph/node integration exists yet
(Phase B); nothing in this document should be read as claiming the
chapter's graphs call this code.

---

## Goal

Given a title/abstract-included, full-text-eligible record with a
confirmed PDF (see "Input boundary" below), produce a versioned,
evidence-bearing answer for every field in a user-authored extraction
schema: one of `extracted` / `not_reported` / `unclear` / `conflicting` /
`extraction_failed`, with every claimed value backed by a verbatim
quotation from a specific, trusted page/section — never a value asserted
without checkable evidence, and never a `not_reported` claim that could
instead mean "we never actually finished looking."

## Method

**Input boundary.** Only `state["full_text_records"]`'s own PDFs are
ever parsed — the same records FT_SCREENING already confirmed as
`decision == "include"` with a real, retrieved PDF (Phase B's own
contract). No abstract/title fallback exists anywhere in this package;
there is no code path that could read `record.abstract` at all.

**Parsing** (`parsing.py`): Docling, directly, deterministic
section-aware chunking (`_group_items_into_chunks` — fixed-size-with-
natural-boundaries, never mid-item, never semantic). No fallback parser:
`DocumentParseError` propagates to the caller on any failure. Installed
Docling version and parser options (`do_ocr`, `do_table_structure`,
`max_chunk_chars`) are recorded and bound into run identity.

**Batching** (`batching.py`): the complete parsed document, split into
deterministic, document-order, fixed-character-budget batches. No
truncation (an oversized single chunk still gets its own batch), no
semantic splitter, no vector index, no top-k retrieval.

**Execution** (`openai_runner.py`, this phase's ONLY provider): one
explicit model / `model_config_id` / `reasoning_effort` / token budget
per run; credentials read lazily from the environment by the `openai`
SDK itself (never held by this module, never in `ReviewConfig` or
persisted state); `response_format={"type": "json_object"}` validated
against a fixed Pydantic response schema; provider-returned model
string and token usage recorded verbatim (`BatchCallTelemetry`); no
retry, no fallback to a different model or provider — a call failure is
recorded as that batch's own terminal outcome.

**Verification** (`evidence.py`): every model-cited `chunk_id` must
match a real parsed chunk; every quotation must appear verbatim in that
chunk's text after ONLY whitespace-collapse + Unicode NFC normalization
(no case-folding, no fuzzy/word-overlap matching); page/section are
always read from the trusted chunk, never from anything the model
asserted. A claim surviving with zero verified evidence becomes
`unclear`, never a silently-accepted value and never a silently-dropped
one.

**Aggregation** (`aggregation.py`): deterministic, across every batch
that touched a field. Identical values (exact normalized-string
equality) are deduplicated; a single-valued field with more than one
distinct supported value becomes `conflicting`; a multiple-valued field
retains every distinct value. No semantic rewriting or construct
harmonization happens during extraction (see "Harmonization boundary"
below).

**Completeness** (`results.py`): `not_reported` is permitted only when
every batch for that document completed — enforced structurally by
`DocumentExtractionResult`'s own validator, not by caller discipline. A
document with any failed/skipped batch reports `extraction_failed` for
whatever it never successfully addressed instead.

**Run identity & resumability** (`run_identity.py`/`checkpoint.py`): a
run's identity is a hash of the ACTUAL ordered document identities sent
— each one `(record_id, normalized PDF path, freshly-streamed PDF
content hash)`, i.e. exactly `run_identity.DocumentIdentity`'s three
fields, not a full `ScreeningItem`-shaped citation/report/study record
(Phase A's own API accepts a bare `(document_id, pdf_path)` pair per
document; it has no citation/report/study fields to bind even if it
wanted to — narrowed here after a review correctly flagged the earlier
draft of this section as overstating that binding) — together with the
extraction schema hash, parser/chunking identity, the deterministic
**batching** identity (`max_batch_chars`, a separate boundary from
parsing's own `max_chunk_chars`: it determines which chunks are
COMBINED into one model call, so two runs differing only in this value
must never share a run_id or reuse each other's checkpoints), prompt/
output-schema versions, and the complete model configuration (including
`expected_provider_model`, the run's own explicit provider-substitution
policy — checked by EXACT match against the API's own returned model,
never alias-normalized) — mirroring `slr_pipeline.nodes.fulltext`'s own
`_stable_run_id()` design (itself converged on this shape across three
review rounds). Checkpoints are per-document, namespaced by run_id, and
are themselves content-hashed, self-describing envelopes
(`checkpoint.DocumentCheckpoint`) binding the run/document/PDF/schema/
config identity alongside the result — not a bare result trusted purely
because a plausible file exists; loading one re-verifies its content
hash and re-checks its own claimed `document_id` against the checkpoint
being requested, raising `CheckpointIntegrityError` on any mismatch
(tampering, corruption, or a copied-from-elsewhere file). An identical
retry resumes without re-calling the provider for already-checkpointed
documents; a changed input (schema, model config, batching, or even one
document's path/content) computes a different run_id and starts a fresh
run rather than attempting an incompatible resume. The final
`RunManifest` reports checkpointed/complete/incomplete document coverage
as three explicit, non-overlapping sets (a checkpointed-but-incomplete
document — some batch failed — is a real, distinct outcome, never
folded into "completed" just because a checkpoint file exists) and binds
every checkpoint's own content hash, refusing to silently overwrite an
already-published manifest with a contradictory hash for the same
document.

Provider-returned telemetry (`response.model`, prompt/completion/total
token usage) is recorded per batch on `BatchExecutionRecord` — part of
`DocumentExtractionResult.batch_executions`, and therefore persisted
inside every checkpoint, not merely present in a transient in-memory
value during a run.

## Alternatives considered (and why rejected for v1)

- **Whole-document single-call extraction** (`charting.py`'s existing
  `pydantic` mode, and `pydantic_extract`'s general approach): simpler,
  but truncates long documents and has no page-aware provenance at all.
  Rejected — this package processes the complete document in batches
  specifically to avoid truncation, and provenance is the whole point.
- **Semantic-splitter chunking** (`charting.py`'s existing `chunking`
  mode via `llama_index`/`sentence_transformers`): the requirement to
  avoid a semantic splitter in v1 is explicit (batching must be
  deterministic and auditable, not embedding-dependent) — the existing
  code's own known failure mode (page/chunk provenance collapses to `[]`
  after semantic re-splitting, degrading every excerpt attributed after
  that point to section-level only) was the concrete evidence for
  rejecting this approach, not a hypothetical concern.
- **Fuzzy/word-overlap quotation matching** (`charting.py`'s
  `_find_page_of_excerpt`, `pydantic_extract`'s general permissiveness):
  rejected in favor of exact-after-normalization matching. A fuzzy
  matcher accepts paraphrase as "close enough"; this package's whole
  evidence-verification boundary exists specifically to NOT do that.
- **Reusing FT_SCREENING's own Docling parser instance directly**:
  considered (identical parser class, already artifact-manifest-
  validated) and rejected for this phase — it is an internal
  FT_SCREENING component, not part of the public surface the SLR_chapter
  integration has otherwise restricted itself to; re-implementing a
  small parser here keeps this package's dependency boundary the one the
  acceptance criteria specified ("do not import FT_SCREENING
  internals"). Revisit only if FT_SCREENING ever publishes a stable,
  public, complete-parsed-document artifact (see "Reusing FT_SCREENING
  evidence" below).
- **Auto-detecting provider/backend from model name or environment**
  (`pydantic_extract`'s `llm.py`, and the exact failure mode
  `il_outcomes` hit in production — see "Prior art" below): rejected
  outright. This package has exactly one runner in this phase, and any
  future local-model runner will be a separately selected, separately
  configured module, never something detected or substituted silently.

## Output schema

See `schema.py` (`ExtractionSchema`/`ExtractionFieldSpec`) and
`results.py` (`DocumentExtractionResult`/`FieldResult`/`ExtractedValue`/
`EvidenceSpanModel`) for the authoritative Pydantic definitions. Every
field specification carries a `FieldProvenance` block
(`origin: theory_derived | inductively_derived | dual_schema`, its own
independent `version`, an optional `rationale`) — required per Card A's
value-grounding bullet and Card D's revision-logging requirement, not
just a label and a question. Every controlled (categorical) value
carries its own `definition`, and `openai_runner.build_batch_prompt()`
puts every one of them, verbatim, into the actual model-visible prompt
— never just a bare enum.

## Cards

- **Card C** (LLM structured extraction against a typed ontology) is
  the primary card this stage instantiates. Per Card C's own audit
  requirement: `reports.CoverageReport.zero_occurrence_field_ids`
  surfaces exactly the "theoretically-grounded field with zero
  occurrences in real output" signal Card C asks to disclose, not treat
  as neutral.
- **Card A**'s per-record-decomposes-into-per-field-instantiations
  pattern (already used for FT_SCREENING's own per-criterion eligibility
  decisions) is the same shape this stage uses per FIELD rather than per
  criterion — each field gets its own status taxonomy, its own
  evidence, its own review flag; a document-level "did this succeed"
  label never substitutes for it.
- **Card D** governs any future schema revision motivated by a
  calibration disagreement: `FieldProvenance.rationale` and
  `ExtractionFieldSpec.provenance.version` are exactly where that gets
  logged, and per Card D's own requirement, already-processed documents
  must be traceable to which schema version they ran under (the run
  identity's `extraction_schema_hash` binding already guarantees this —
  a schema revision computes a different run_id, never silently mixed
  with the prior version's results).

## Checks (validation / verification / sensitivity, beyond the
cross-cutting rules)

- **Structural, not just documented**: `not_reported` requiring batch
  completeness, and `requires_human_review` matching its own status, are
  Pydantic validators — a violating result cannot be constructed, not
  merely discouraged (rules 1/2: doc ≠ code, and instrument every
  default/fallback — both satisfied by making the invalid state
  unconstructable rather than trusting caller discipline).
- **Yield vs. quality, never blended** (rule 7): `reports.py`'s
  `CoverageReport` (execution telemetry only) and `AuditedAccuracyReport`
  (requires a real `CalibrationCorpus`) are separate models; the latter
  is `None`, never fabricated, when no calibration data exists.
- **Finest-grain validation** (rule 8): calibration judgments
  (`calibration.FieldJudgment`) are per FIELD per DOCUMENT, with
  correct status AND correct values — never a document-level "did it
  produce JSON" check. `build_calibration_worksheet_rows()` produces
  one worksheet row per field for exactly this reason.
- **Staged ladder** (rule 9, §4.0's default shape): mandatory before
  running extraction across the chapter corpus --- n=1 (plumbing) → n≈5–10
  (deliberately difficult: decisive value on a late page, genuinely
  `not_reported`, contradictory values across sections, multilingual
  text, table/figure-dependent evidence, a long document, a prompt-
  injection attempt, the same value repeated in different sections) →
  n≈20–30 (first quantitative field-level audit) → n≈50–100 (only if
  justified by corpus size/claims) → full run only once no new failure
  class has appeared for a while. `calibration.DifficultyCategory`
  encodes exactly this stratification, not a random sample.
- **Sensitivity** (rule 10): not yet run in Phase A (no populated
  calibration corpus exists to run it against) — flagged as required
  BEFORE any results from this stage support a claim in the chapter,
  not optional polish. Candidate sensitivity axes once a corpus exists:
  `max_chunk_chars` (does batch granularity change what gets found),
  `reasoning_effort`, and any schema field-definition wording revised
  under Card D.

---

## Harmonization boundary

Per the acceptance criteria's explicit boundary, this package produces
exactly two kinds of raw output and NOTHING further:

- **raw extracted value** (`ExtractedValue.value` — the model's literal
  string, `"RCT"` and `"Randomized controlled trial"` stay two distinct,
  unmerged values if a document uses both).
- **evidence** (`EvidenceSpanModel` — trusted chunk/page/section +
  verbatim excerpt).

A **canonical value**, a **mapping method and version**, and a
**confidence** in that mapping are all explicitly OUT OF SCOPE for this
package — they belong to a strictly later, separate harmonization stage
that reads these raw values as its own input. `il_outcomes`'s
harmonization work (`term_normalization_20260319_v1.py`) is the
concrete prior-art example of what that later stage looks like structurally
(raw term kept alongside canonical term, mapping method, confidence, and
human review flags, all as separate fields) — worth reusing that SHAPE
when a harmonization stage is eventually built, not something this
extraction stage does itself.

## Reusing FT_SCREENING evidence

Allowed, narrowly, per the protocol's own efficiency-gain note:
`ChapterFullTextOutcome`'s already-verified per-criterion evidence
(from Phase B) MAY be supplied to this package as clearly-marked
`eligibility_seed_evidence` — useful as a cross-check for an overlapping
extraction field, never as a substitute for full-document extraction
coverage, and never as license to infer `not_reported` from its
absence (FT_SCREENING's own retrieval was optimized for eligibility
criteria, not this schema's fields — silence there proves nothing about
this schema). Any such seed evidence must still pass this package's own
`evidence.verify_evidence()` before it can back a value here. Not
implemented in Phase A; noted as a real, bounded future integration
point.

## Prior art -- what five other projects in this codebase taught this
design, and what NOT to copy

| Project | Pattern kept | Pattern rejected |
|---|---|---|
| `pydantic_extract` | The `value`/`confidence`/`evidence_quote`/`evidence_page`/`rationale` per-field evidence shape (`models.py`'s `_evidence_model`) | Project-specific Excel-driven config, the ontology/knowledge-graph pipeline, hardcoded credential paths, multi-provider auto-detection dispatch |
| `auto-mare` | Per-document resume; keeping raw-response auditing separate from later entity merging; keeping extraction separate from downstream harmonization (`models.py:61`) | Silent fallback to another model; an empty extraction result that could mean either "nothing found" or "it failed" (this package's whole `not_reported`-requires-completeness rule exists to never repeat this); unverified quotations |
| `egpa_test` | OpenAI and local extraction as genuinely separate scripts sharing one scientific schema; matched-provider comparison, not reconciliation (`fulltext_extraction_local_v1.py`) | Coercing an invalid value into a plausible default; a reconciliation rule that just prefers OpenAI whenever it's available |
| `iasia_test` | Page markers; a truncation flag; batch state; explicit `ok`/`no_text`/`extract_error` statuses (`drr_review_schema.py`) | Declaring a Pydantic schema and then never actually validating Layer-2 responses against it; model-asserted (unverified) pages/quotations; a flat 60,000-character truncation |
| `il_outcomes` | Keeping raw term, canonical term, mapping method+version, confidence, and human-review flags as separate, never-merged fields (`term_normalization_20260319_v1.py:369`) | Automatic backend fallback that made a run LOOK unified while a different method silently ran underneath — the single clearest warning this whole survey produced, and the direct reason this package has one explicit runner and will have separately-selected runners later, never auto-detection |

`merge_screening_data_extraction` was inspected only for audit/
validation patterns (per the survey's own scope): its post-hoc
faithfulness-audit technique (cross-tabulating extracted values against
an independent signal — there, the screening decision — to catch
systematic over-extraction) is a real, useful, LATER technique a reader
could apply once real extraction + calibration data exist; not
something this package builds in for v1.

---

## Known protocol drift (found during this survey, not fixed here)

Per `lit_review_protocol`'s own rule 5 ("version artifacts, never edit
in place"), none of the following are edited here — this section only
records what a future revision needs to fix:

- The canonical stage sequence in `template_v5.md` §3.3 lists
  "Structured extraction" BEFORE "Full-text / richer-data collection"
  and "Full-text eligibility screening," even though its own one-line
  placeholder text ("goal / method / ...") gives no stage-order
  guidance and "Structured extraction" (per this very document's own
  Method section) assumes inclusion is already decided. A future
  `template_v6.md` should reorder §3.3 to: Search → title/abstract
  screening → full-text collection → full-text eligibility → structured
  extraction → quality appraisal → synthesis → writing, and add proper
  per-stage templates for quality appraisal and synthesis (currently no
  dedicated subsection exists for either).
- `lit_review_protocol/INSTANCES.md` still describes FT_SCREENING as
  "planned" and points to `context_protocol_v1.md`.
- `FT_SCREENING/context_protocol_v4.md` still says FT_SCREENING's
  implementation is "planned" and references `api_contract_v2.md` — the
  actual project has since reached `api_contract_v4.md`, is fully
  implemented and calibrated, and is now integrated into SLR_chapter
  (Phase A + Phase B both closed, per that project's own persistent
  memory).
- Once a real update is warranted, the correct artifacts are
  `FT_SCREENING/context_protocol_v5.md` (recording the implemented/
  validated state), `lit_review_protocol/template_v6.md` (corrected
  stage order + the two missing per-stage templates), an updated
  `INSTANCES.md`, and a registered SLR_chapter protocol instance derived
  from template v6 — none of which this survey creates.
