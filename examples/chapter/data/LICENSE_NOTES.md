# Provenance and licensing of the chapter fixture data

Every file under `examples/chapter/data/` is listed below with one of four
categories. `examples/chapter/test_data_license_notes.py` walks this directory
and fails if any file is missing a row or carries an unrecognised category.

Categories:

1. **author-created-synthetic** — invented by the chapter authors for teaching.
   No real study, person, or API response. Licensed CC BY 4.0.
2. **model-derived** — a numeric derivation of an author-created-synthetic file
   through a third-party model. Provenance is in the manifest.
3. **generated-metadata** — a machine-written record about a model-derived file.
4. **authored-metadata** — an authored index / notes file about this directory
   (this file).

| file | category | provenance | license | attribution / notes |
|---|---|---|---|---|
| bib_records.json | author-created-synthetic | invented | CC BY 4.0 | 18 fake bibliographic records; DOIs are non-resolving `10.1000/...` placeholders |
| oa_retrieval_records.json | author-created-synthetic | invented | CC BY 4.0 | 2 metadata-only records for the S8 no-open-access branch |
| expected_screening.json | author-created-synthetic | invented | CC BY 4.0 | first-rater screening key for `bib_records.json` |
| second_rater_decisions.json | author-created-synthetic | invented | CC BY 4.0 | second-rater key; disagreements chosen to put Cohen's kappa near 0.4 |
| synthesis_studies.json | author-created-synthetic | invented | CC BY 4.0 | 3 charted study records; study-B contradicts study-A by design |
| topic_corpus.json | author-created-synthetic | invented | CC BY 4.0 | 12 abstracts, 3 latent themes by construction |
| topic_corpus_embeddings.npy | model-derived | see manifest | CC BY 4.0 (derivation of an author-created corpus) | MiniLM (`all-MiniLM-L6-v2`, Apache-2.0) embeddings of `topic_corpus.json` |
| topic_corpus_embeddings.manifest.json | generated-metadata | written by `scripts/gen_topic_embeddings.py` | CC BY 4.0 | hashes, model id/revision, package versions |
| pipeline_state_fixture.json | author-created-synthetic | invented | CC BY 4.0 | terminal scoping-review state; PRISMA numbers satisfy the S12 conservation groups |
| ai_traceability_fixture.json | author-created-synthetic | invented | CC BY 4.0 | trimmed `ai_traceability.json` shape |
| LICENSE_NOTES.md | authored-metadata | authored | CC BY 4.0 | this file |
| raw_responses/pubmed.xml | author-created-synthetic | invented | CC BY 4.0 | realistic PubMed EFetch XML envelope; PMIDs are fake |
| raw_responses/openalex.json | author-created-synthetic | invented | CC BY 4.0 | realistic OpenAlex response shape (inverted-index abstracts, DOI URLs) |
| raw_responses/crossref.json | author-created-synthetic | invented | CC BY 4.0 | realistic Crossref response shape (title list, JATS abstract) |
| raw_responses/scopus.json | author-created-synthetic | invented | CC BY 4.0 | realistic Scopus response shape (namespaced keys) |
| papers/relevant.md | author-created-synthetic | invented | CC BY 4.0 | fake paper; authors, journal, numbers all invented |
| papers/wrong_population.md | author-created-synthetic | invented | CC BY 4.0 | fake paper; primary-research topic (clear exclude) |
| papers/editorial.md | author-created-synthetic | invented | CC BY 4.0 | fake opinion piece; no method, no data |
| papers/ambiguous.md | author-created-synthetic | invented | CC BY 4.0 | fake experience report; reasonably callable either way |
| papers/answer_key.json | author-created-synthetic | invented | CC BY 4.0 | expected S8/S9/S10 answers; every quote is a verbatim substring of its paper |
| oa_resolver_responses.json | author-created-synthetic | invented | CC BY 4.0 | fake Unpaywall / Semantic Scholar payloads keyed by normalised DOI |
| scripted_responses/s04.json | author-created-synthetic | invented | CC BY 4.0 | scripted fake-LLM payloads for S4, incl. `_bad` variants |
| scripted_responses/s07.json | author-created-synthetic | invented | CC BY 4.0 | scripted fake-LLM payloads for S7, incl. `_bad` variants |
| scripted_responses/s08.json | author-created-synthetic | invented | CC BY 4.0 | scripted fake-LLM payloads for S8, incl. `_bad` variants |
| scripted_responses/s09.json | author-created-synthetic | invented | CC BY 4.0 | scripted fake-LLM payloads for S9, incl. `_bad` variants |
| scripted_responses/s10.json | author-created-synthetic | invented | CC BY 4.0 | scripted fake-LLM payloads for S10, incl. `_bad` variants |
| scripted_responses/s11.json | author-created-synthetic | invented | CC BY 4.0 | scripted fake-LLM payloads for S11, incl. `_bad` variants |
