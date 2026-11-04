# Scenario table

| fx id | source db | branch |
|---|---|---|
| fx-0009 | pubmed | title/abstract include -> full-text eligible -> extracted; funding: funder-had-no-role canned phrase |
| fx-0001 | pubmed | title/abstract exclude — off-topic (industrial predictive maintenance) |
| fx-0003 | pubmed | title/abstract exclude — wrong setting (urban ED); duplicate-of-title canonical (fx-0007 dup) |
| fx-0004 | pubmed | title/abstract exclude — qualitative only, no MBSE/simulation; duplicate-of-title canonical (fx-0011 dup) |
| fx-0006 | pubmed | title/abstract exclude — no disaster context; duplicate-of-title canonical (fx-0030 dup) |
| fx-0007 | pubmed | dedup exact-title duplicate of fx-0003 (basis: title) |
| fx-0002 | web_of_science | title/abstract include -> full-text eligible -> extracted; funding: present with a substantive stated role |
| fx-0008 | web_of_science | title/abstract exclude — wrong method (blockchain, not simulation/MBSE); duplicate-of-title canonical (fx-0023 dup) |
| fx-0010 | web_of_science | title/abstract exclude — entirely off-topic (wearable glucose monitoring) |
| fx-0011 | web_of_science | dedup exact-title duplicate of fx-0004 (basis: title) |
| fx-0012 | web_of_science | title/abstract exclude — simulation present, wrong domain (airport security); duplicate-of-title canonical (fx-0028 dup) |
| fx-0013 | cinahl | title/abstract include -> full-text eligible -> extracted; funding statement ABSENT |
| fx-0014 | cinahl | title/abstract exclude — no simulation/MBSE, no disaster context; duplicate-of-title canonical (fx-0021 dup) |
| fx-0015 | cinahl | title/abstract exclude — rural+healthcare present but BOTH other criteria fail (harder near-miss, lower confidence) |
| fx-0016 | cinahl | dedup exact-title duplicate of fx-0009 (basis: title) |
| fx-0005 | scopus | title/abstract include -> full-text eligible -> extracted; funding: 'No external funding' canned phrase |
| fx-0017 | scopus | title/abstract exclude — entirely off-topic (autonomous vehicle perception) |
| fx-0019 | scopus | title/abstract exclude — off-topic (SCADA cybersecurity); duplicate-of-title canonical (fx-0026 dup) |
| fx-0020 | scopus | title/abstract exclude — descriptive/statistical, no simulation/MBSE method |
| fx-0021 | scopus | dedup exact-title duplicate of fx-0014 (basis: title) |
| fx-0022 | acm | title/abstract exclude — entirely off-topic (serious game for CS education) |
| fx-0023 | acm | dedup exact-title duplicate of fx-0008 (basis: title) |
| fx-0024 | acm | title/abstract include -> full-text eligible -> extracted; funding: present, ambiguous non-canned role text |
| fx-0025 | acm | title/abstract exclude — entirely off-topic (video-game terrain generation) |
| fx-0018 | engineering_village | title/abstract include -> FULL-TEXT EXCLUDED (reason: not_rural_healthcare) |
| fx-0026 | engineering_village | dedup exact-title duplicate of fx-0019 (basis: title) |
| fx-0027 | engineering_village | title/abstract exclude — simulation present, wrong domain (structural engineering); duplicate-of-title canonical (fx-0028's own db-pair, see fx-0012) |
| fx-0028 | business_source_complete | dedup exact-title duplicate of fx-0012 (basis: title) |
| fx-0029 | business_source_complete | title/abstract exclude — entirely off-topic (retail supply-chain strategy) |
| fx-0030 | business_source_complete | dedup exact-title duplicate of fx-0006 (basis: title) |
