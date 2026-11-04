# Thabit 2025 reproduction — running notes

Lean prototype notes for the chapter's SLR-automation reproduction case study.
Not yet integrated into `manuscript/` (Track-3-adjacent, out of current Phase E
scope) — kept here until a manuscript section is designated for it.

## Deviation: title/abstract screening fields

Thabit et al. 2025 (Appendix A) state screening used "titles, abstracts, and
keywords." This project's convention is abstract-only (title/keywords usually
redundant with abstract content) — **deliberate deviation**, not an omission.
Note for chapter text: state this explicitly when presenting the
reproduction, since it is one source of expected funnel divergence from the
original paper's reported counts.

## Journal-filter proxy (119-journal Scimago allowlist)

Paper restricts to journals in 4 Scimago categories (Sociology and Political
Science; Public Administration; Political Science; Political Science and
International Relations) — exact 119-journal list not published, unrecoverable.
Proxy: LLM yes/no classifier per unique journal name (`journal_filter.py`),
368/871 kept vs paper's 291. BERTopic cross-check (`bertopic_explore.py`)
broadly validates the filter directionally. Journal diversity (548 unique
across 871 records) confirmed genuine, not a title-variant artifact
(`journal_normalize.py`: only 4 groups merge).

## Appendix B (59 final-included studies) reproduction check

Of 59: 51 matched in raw records and survive dedup; of those 45 reach
journal=YES, 6 are journal-classifier false negatives (interdisciplinary
journals a strict 4-category prompt plausibly misses — Sustainability
Science x2, Communicatio, Arctic, Leadership Quarterly, International
Journal of Integrated Care). 8 titles not found in our search corpus at all.

### The 8 not-found titles: expert-addition hypothesis

Live exact-title lookups against both source databases for all 8:

| # | Title (short) | WoS | Scopus |
|---|---|---|---|
| 1 | A Decision Theory Perspective on Wicked Problems... | present (2022) | present (2022) |
| 2 | Organizational Maturity for Co-creation... | present (2022) | present (2022) |
| 3 | Achieving SDGs Through Collaborative Innovation... | present (2022) | present (2022) |
| 4 | Paradox and Collaboration in Network Management | present (2010) | present (2010) |
| 5 | Ambidexterity in Cross-Sector Collaborations... | present (2021) | present (2021) |
| 6 | Coping with Value Conflicts in Interorganizational Collaborations | **absent** | present (2018) |
| 7 | Community Foundations as Network Conveners... | present (2022) | present (2022) |
| 8 | Co-initiation of Collaborative Innovation in Urban Spaces | present (2018) | present (2018) |

**8/8 confirmed present in at least one source database** (7/8 in both WoS
and Scopus; #6 only in Scopus). None are genuinely absent from the
literature — every one is indexed, just not retrieved by our calibrated
`TS=(THEME1) AND TS=(THEME2)` query.

**Conclusion**: these 8 are very likely part of the paper's own 27
expert-recommended additions (Appendix A), added after the 291-study
title/abstract screening stage, explicitly *outside* the systematic
WoS/Scopus search. This is structural, not a bug in our query calibration:
by construction, no search-string tuning can ever retrieve expert-added
studies, since the original authors' own search terms also missed them —
that is precisely why expert consultation was needed. Chapter takeaway:
search-based SLR reproduction has a hard ceiling short of 100% recall
whenever the original review used non-systematic supplementary methods
(expert consultation, snowballing, hand-search); this should be presented
as a general limitation of automated/semi-automated SLR pipelines, not
specific to this reproduction.

Scripts: `appendix_b_check.py` (fuzzy Appendix B match),
`appendix_b_wos_title_lookup.txt` (WoS live check),
`appendix_b_scopus_title_lookup.py` / `.json` (Scopus live check, this turn).

## Full-text pilot (20-record OA sample): green-only false positive, caught and reverted

`fulltext_pilot.py`'s known_excluded OA sample included `wos-0be9828c8c8318cd9b72`
("How public leaders can promote public value through co-creation") via
`oa_by_id.get(id) is True`, but its actual `open_access_designation` is
"Green Submitted" only -- exactly the naive-bool-check false positive
`wos_manual_import.py` documents (gold/hybrid/bronze = publisher page free;
green = self-archived manuscript elsewhere, publisher page still paywalled).
Tier 1 auto and Tier 2 Selenium both correctly failed to find it (no OA PDF
at the DOI, by design).

Swapped it for a randomly-drawn replacement (`wos-0e5e5a2f17916a2d5909`,
"Prefiguring Sustainability...") from the corrected `is_open_access is True`
pool, to keep the pilot's "OA" label methodologically honest. But a manual
web search then turned up the original paper's own accepted-manuscript
docx (a legitimate green-OA repository deposit, not a bypass) -- so real,
usable full text existed for it after all, just not auto-discoverable.
**Reverted the swap**: original 20-record sample restored, docx parsed via
`python-docx` (no PDF conversion needed) and saved as
`fulltext_pilot/text/wos_0be9828c8c8318cd9b72.txt` (55,328 chars).

Final: 20/20 full text retrieved -- 4 Tier 1 auto, 10 Tier 2 Selenium
(`fulltext_pilot_reconcile.py`), 5 Tier 3 manual PDF, 1 Tier 3 manual docx.
Chapter takeaway: OA metadata flags (green vs. gold/hybrid/bronze) matter
for *automated* retrieval yield stats, but don't by themselves prove content
is unobtainable -- Tier 3 (human search) can still legitimately recover
green-OA content an automated cascade will correctly skip.

## Deductive data extraction pilot: single-LLM-pass F1 ceiling vs Table C1 gold

`ft_codebook.py` transcribes Thabit et al.'s own Table 2 codebook (37
concepts) and Table C1's gold concept-to-article mapping. `run_deductive_
extraction.py` applies that SAME codebook deductively to the 10
known_included pilot full texts, one LLM call per article judging all 37
concepts; `compare_to_gold.py` scores against Table C1 (per-publication and
micro-averaged precision/recall/F1, plus `--diff va vb` deltas).

Four prompt variants tried, each a different lever, none a wording tweak
alone after v1:
- **v1** (loose bar: "passage supports the concept as named"): P=0.409
  R=0.921 F1=0.567. Near-ceiling recall, heavy over-coding (~33/37 flagged
  per article vs. gold's ~15).
- **v2** (strict bar: concept must be *substantively* treated in the
  article's own findings/discussion, not background/passing mention):
  P=0.436 R=0.875 F1=0.582. FP only dropped 202->172 -- wording alone
  barely moves it.
- **v3** (v2 + two worked few-shot examples + a grounded cardinality prior,
  "median 13, range 2-25 concepts/article" computed directly from all 58
  scoreable Table C1 refs): P=0.437 R=0.868 F1=0.581. Flat vs v2,
  noise-level. Anchoring/priors-as-wording don't move it either --
  independent per-concept judgment lets ~30/37 pass regardless of phrasing.
- **v4** (mechanism change: every concept also gets a 1-100 "strength"
  score; hard cap keeps only the top-18 by strength, forcing relative
  ranking instead of 37 independent absolute judgments): P=0.539 R=0.638
  F1=0.584. Real precision gain (+0.10) but recall drops (-0.23) roughly
  1:1 in F1 terms -- the fixed cap truncates genuine TPs on richer
  articles (e.g. gold=22 for R20, capped to 18).

Post-hoc, zero-cost sweep over v4's cached (concept, strength) pairs --
both a strength-threshold cutoff (found AND strength>=T, T=30..70) and a
top-K-by-strength cutoff (K=12..28) -- traces the full precision/recall
tradeoff curve without re-paying. Result: **F1 stays in a flat 0.52-0.60
band across the entire sweep**; the single best point found (K=28,
P=0.465 R=0.842 F1=0.600) barely beats the unconstrained v2/v3 baseline
(~0.58). No cutoff, threshold, or ranking mechanism gets meaningfully
closer to gold than the others.

**Conclusion**: this is a real ceiling, not an undertuned prompt. Four
independently different levers (bar wording, few-shot anchoring, a
grounded cardinality prior, and forced relative ranking with a hard cap)
all land within the same F1~0.55-0.60 band. The gap to Thabit et al.'s
own gold Table C1 most likely reflects the process gap the module
docstring already names: one LLM pass vs. their 3-human-coder calibration
+ dual-coding + disagreement-resolution pipeline, not a fixable prompt
defect. Chapter takeaway: single-pass deductive LLM coding against a rich
37-concept codebook is a plausible teaching demo of the *mechanics* of
deductive extraction, but readers should not expect it to reproduce a
calibrated multi-coder qualitative process's agreement level without
something structurally different (e.g. ensemble/self-consistency voting,
or a second verification pass) -- worth stating explicitly as a
limitation rather than chasing further prompt tuning.

Artifacts: `deductive_extracted_v1.jsonl`..`_v4.jsonl` +
`deductive_cache_v1/`..`_v4/` (each version's raw model output preserved,
including v4's uncapped strength scores, so any future cutoff analysis is
a $0 re-analysis of cached data, not a re-run).

## Quality appraisal: skipped by design

This reproduction never ran a quality-appraisal stage. Thabit et al. 2025 is
itself a scoping review (broad-mapping intent, no effect-size synthesis) --
per the pipeline's own scoping-mode convention, quality/risk-of-bias
appraisal is an optional, explicitly skippable stage for that review type,
not a mandatory one (contrast with a systematic review of effect
estimates, where it would be required). `prisma_counts.json`'s
`quality_assessed`/`excluded_quality` fields are left `null` for this
reason -- `null` means "step skipped by design," not "not yet run."

## Theme-level narrative synthesis: the data-charting/synthesis stage beyond extraction

Per-article deductive extraction stops at found/not-found + quote per
concept. Thabit et al.'s own Findings section is a further, cross-study
aggregative-synthesis step: prose per theme identifying what's convergent
across cases, what's divergent (named outliers), illustrative quotes
attributed to specific studies -- built here as `run_theme_synthesis.py`
(one LLM call per theme, 6 total, live paid call), using ONLY v2's
per-article found/quote evidence (chosen over v1/v3/v4 -- no artificial
cap, best measured signal-to-noise vs gold). `build_theme_reference.py`
programmatically extracts Thabit's own cleaned Findings-section prose per
theme (`theme_reference_text.json`) as a qualitative reading-comparison
aid, not a scored gold reference -- narrative prose isn't
precision/recall-comparable the way found/not-found decisions are.

Compared all 6 themes to Thabit's own Findings-section prose (not just
the initial "Broader environment and public sphere" spot-check). Holds
across the full set: structural coverage (which sub-concepts/components
get discussed) tracks Thabit's own component breakdown closely, and no
theme shows a qualitative failure (hallucinated pattern, wrong direction,
fabricated quote). Two named divergent/outlier cases the model surfaced
on its own initiative -- R12 (marine conservation) contradicting
procedural-justice convergence in "Instrumental public values", and thin
equity-of-payment evidence flagged plainly in "Instrumental and prime
public values" -- mirror the same rhetorical move ("however, X et al.
shows...") Thabit's own text uses, and only appear where the underlying
evidence actually supports them (not forced onto themes with cleaner
convergence, e.g. "Operational capabilities and assets").

Two systematic, expected (non-bug) differences present in every theme:
(1) our evidence pool is 10 of Thabit's 59 articles, so different
specific illustrative examples surface than in the original; (2)
citations are by full article title, not author-year citation keys
(pilot's extracted text carries no bibliographic metadata) -- and
relatedly, Thabit's prose cites external theoretical literature (e.g.
Bozeman & Jorgensen, Emerson et al. 2011, Page et al. 2015) our version
has no access to, since the pipeline only sees the 10 pilot full texts,
not Thabit's own literature review.

Explicitly out of scope: Thabit's own Figure 3 "unifying framework"
(cross-theme theoretical integration) is the authors' own theorizing
contribution, not a data-charting/synthesis target -- the system prompt
tells the model not to attempt it.

Artifacts: `theme_reference_text.json` (Thabit's own text, cleaned),
`theme_synthesis.json` (generated, 6 themes), `theme_synthesis_cache/`
(diskcache, so cutoff/prompt tuning here is also re-runnable at $0 for
unchanged themes).
