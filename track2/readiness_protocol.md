Track 2 pipeline-readiness rehearsal protocol — this is not a real systematic-review protocol and was never registered with any registry. `protocol_registration_id` is the fixed literal `TRACK2-READINESS-REHEARSAL-NONE` for exactly that reason. Every methodological choice below is real (the pipeline is genuinely configured this way), but the run itself replays canned model responses over a synthetic corpus (see `tests/reference_run/fixtures/corpus/PROVENANCE.md`) to demonstrate that the chapter's automation pipeline produces every methodological artifact it claims to, before the audited run on real data (Step 12B). No finding, country, or funder anywhere downstream of this protocol is real.

## 1. Rationale

This run rehearses the chapter's end-to-end scoping-review automation pipeline on a synthetic corpus with replayed model calls, to demonstrate that every methodological artifact the chapter cites is produced and internally consistent before the audited Track-3 run on real data.

## 2. Review type and title

Review type: scoping review (JBI/PRISMA-ScR methodology).

Title: *Pipeline-readiness rehearsal: MBSE and computer-simulation methods for rural healthcare disaster preparedness (scoping review)*.

## 3. Research questions

1. What is the current body of evidence on the application of computer simulation based on model-based systems engineering approaches to disaster preparedness and response for rural healthcare systems?
2. What MBSE methods, healthcare settings, disaster types, and contexts have been studied in rural healthcare disaster planning?

## 4. Population, Concept, Context (PCC)

- **Population:** Rural healthcare systems and rural communities involved in disaster preparedness and response.
- **Concept:** Application of model-based systems engineering (MBSE), including computer simulation and systems thinking, to rural healthcare disaster planning.
- **Context:** Disaster preparedness and response planning in rural healthcare settings.

## 5. Eligibility criteria

**Inclusion:**
1. Studies addressing computer simulation or MBSE approaches applied to disaster preparedness or response.
2. Studies focused on rural healthcare systems or rural healthcare settings.
3. Studies relevant to disaster planning, preparedness, or response.

**Exclusion:**
1. Studies not involving rural healthcare systems.
2. Studies not using MBSE or computer simulation approaches.
3. Studies not related to disaster preparedness or response.

No date, language, or publication-status restriction is applied (`date_from`/`date_to`/`language` are all unset).

## 6. Information sources and search strategy

Seed query (translated per database by the pipeline's own strategy builder, `gpt-5.4-mini`, up to 5 synonyms/concept, title/abstract fields preferred):

```
((rural OR non-urban OR frontier) AND (healthcare OR "health care") AND
(disaster OR pandemic OR epidemic) AND (simulation OR "agent based model"
OR "system dynamics" OR "machine learning" OR "artificial intelligence"))
```

Databases (7, each executed here against its fixture adapter `fx_<database>`, not the real database): `pubmed`, `web_of_science`, `cinahl`, `scopus`, `acm`, `engineering_village`, `business_source_complete`. Up to 50 results/database, `as_of_date` 2026-01-01.

## 7. Deduplication

Exact mode: casefold + NFKC-normalise + collapse-whitespace + strip title match (plus exact-DOI match, not exercised here since every fixture record has `doi: null`, see `known_real_doi_denylist.txt`).

## 8. Title/abstract screening

Model `gpt-5.4-mini`, batch size 5, 1 worker, auto-include threshold 0.80, uncertain threshold 0.0. Second-rater QC enabled (reuses the screening model), calibration sample size 6, calibration seed 20260101, 200 bootstrap resamples, saturation levels 0.80/0.90/0.95.

## 9. Full-text eligibility

Three criteria, each independently adjudicated, `missing_evidence_policy: review` on inconclusive evidence:

1. `mbse_or_simulation` — the study applies computer simulation or model-based systems engineering (MBSE) methods (discrete-event, agent-based, or system-dynamics simulation, or systems modelling) to a disaster-planning problem.
2. `rural_healthcare` — the study concerns rural healthcare systems, rural healthcare facilities, or rural communities' healthcare, in a disaster context.
3. `disaster_context` — the study addresses disaster preparedness, planning, or response (natural disaster, pandemic, epidemic, or mass-casualty event).

## 10. Data extraction (charting)

9-field extraction schema (`reference-charting-v1`), Docling parsing with OCR and table-structure detection both disabled, max chunk 3000 chars, max extraction batch 12,000 chars. Every extracted value is evidence-verified against a quoted, page-located span; unclear/conflicting fields go to a human review queue (empty this run).

## 11. Synthesis

Map/refine narrative synthesis (refine batch size 3), evidence map and claim-backed results both generated; legacy IMRaD narrative and BERTopic topic modelling both disabled for this rehearsal.

## 12. AI use and reporting

AI use is prespecified in full above (`ai_use_prespecified: true`) — every model call this protocol authorises is listed in §6, §8, §9, §10, §11. No protocol deviations occurred (`protocol_deviations: []`). Funding statement: "This pipeline-readiness rehearsal received no external funding." Data-governance declaration: "Track 2 pipeline-readiness rehearsal: synthetic corpus, replayed model calls, no human subjects, no external data-governance obligations."

## 13. Registration

Not registered anywhere. `protocol_registration_id` is fixed to the literal `TRACK2-READINESS-REHEARSAL-NONE` to make that fact machine-checkable rather than merely asserted in prose.
