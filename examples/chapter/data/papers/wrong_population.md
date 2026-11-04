# Automated detection of outcome switching in trial registries

## Abstract

We built an automated pipeline that compares the primary outcomes registered for a clinical trial with the primary outcomes reported in the trial's published article, and flags discrepancies. The pipeline processed 11,842 trials and flagged 27 percent as having at least one switched primary outcome. This work is about primary trial records and their registry entries, not about the conduct of literature reviews.

## Methods

Registered outcomes were downloaded from two trial registries. Published outcomes were extracted from the linked articles with a rule-based parser followed by a language-model check. A discrepancy was recorded when a registered primary outcome did not appear as a primary outcome in the publication, or when a new primary outcome appeared that had not been registered.

Two researchers manually verified a random sample of 300 flagged trials to estimate the pipeline's precision.

## Results

The pipeline flagged 27 percent of trials. On the verification sample its precision was 0.91. Switching was more common in trials without a data monitoring committee and in trials with industry funding.

## Discussion

Automated registry-to-publication comparison can make outcome switching visible at a scale no manual audit could reach. The method operates entirely on trial records and does not touch systematic-review screening, extraction, or synthesis. Applying it would be a governance tool for trialists and journal editors.

## References

1. Grigalyte I. Trial registry data mining for outcome reporting bias. 2021.
2. Mockus C, Schmidt E. A randomized trial of a mobile app for smoking cessation. 2022.
