# Large language models as a second screener: a retrospective evaluation

## Abstract

The objective of this study was to measure whether a large language model can replace one of two human screeners during title and abstract screening without lowering recall. We conducted a retrospective evaluation study using four previously completed systematic reviews as reference standards. The model reproduced the human inclusion decisions with 0.96 recall while halving screening time. We conclude that the model is a credible second screener but not yet a safe sole screener.

## Methods

Eligibility criteria were fixed in advance and recorded in a protocol before any data were seen. The full search string and the date of each database query are reported in Appendix A, so that the searches can be re-run. For each of the four reference reviews we retrieved the full set of records that the original teams had screened, together with their final inclusion decisions.

Each record was presented to the model in batches of five with the review's own inclusion and exclusion criteria in the prompt. The model returned one of three labels per record — include, borderline, or exclude — with a short reason. Only a single human reviewer screened each record in the present study, so we could not measure inter-rater reliability between two humans; instead we treated the original review teams' published decisions as the reference standard.

All screening was performed at a single centre in Lithuania. Analysis code is available on request from the corresponding author.

## Results

Across the four reviews the model achieved 0.96 recall against the published inclusion decisions, missing 3 of 71 included studies. Screening time, measured as reviewer-hours logged against the task, fell by roughly half. Precision was lower, so a human still had to confirm the model's "include" and "borderline" calls, but the volume of records requiring a human read dropped by 58 percent.

The three missed studies shared a feature: each described its method only in a supplementary file, which was not part of the abstract the model saw.

## Discussion

Several limitations temper these results. The reference standard is itself imperfect, since the original teams may have made errors. The four reviews are all in health services research and the finding may not transfer to other fields. The authors declare no competing interests, though funding sources are not itemised in the original grant records.

We recommend that the model be used only alongside a human reviewer, with the human reading every record the model marks include or borderline and a sample of the excludes.

## References

1. Petrauskas A, Nguyen L. Large language models for title and abstract screening. J Res Synth Methods. 2023.
2. O'Brien J, Yamada T. A rapid review workflow combining machine learning and human screening. Syst Rev. 2024.
3. Lindqvist H, Farooq N. Reporting guidance for AI-assisted systematic reviews. 2024.
