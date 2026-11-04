Track 2 pipeline conformance scenario — synthetic test data, not scientific evidence. Every finding, funding statement, and country named below is fabricated to exercise a specific pipeline branch (see SCENARIO_TABLE.md) and must never be cited as a real study.

## What this corpus is

This directory is an entirely synthetic bibliographic corpus (30 records across 7 fixture databases), authored solely to exercise specific branches of the chapter's scoping-review automation pipeline: exact-title deduplication, title/abstract screening, full-text eligibility adjudication (including its one full-text-only exclusion), 9-field extraction (including 5 distinct funding-statement presentations), and map/refine synthesis. No record describes a real study, no country, funder, or finding named anywhere in this corpus is real, and no PDF in `pdfs/` should ever be read as reporting an actual result.

The single structured answer key this corpus is generated from is `slr_pipeline/reference_fixture_answer_key.py`; `SCENARIO_TABLE.md` renders it into one row per record, naming the pipeline branch each record exists to exercise.

## Licence

CC BY 4.0 -- see `LICENSE.md`.

## Real-DOI exclusion

`known_real_doi_denylist.txt` lists every real DOI this corpus must never contain (Berg2023's real included-study DOIs, plus every DOI in `chapter_claim_sources.json`); every fixture record has `doi: null`.
