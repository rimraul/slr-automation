**1 sąrašas.** Deduplikacijos kaskada `dedupe`.

```python
def dedupe(records: list[dict], *, title_threshold: float = 0.90) -> DedupeResult:
    """Pašalina dublikatus dviejų pakopų kaskada ir grąžina ``DedupeResult``.

    ``title_threshold`` -- mažiausias ``difflib`` panašumo santykis, nuo kurio
    du normalizuoti pavadinimai laikomi tuo pačiu darbu.
    """
    unique: list[dict] = []
    duplicates: list[DuplicatePair] = []
    seen_doi: dict[str, str] = {}

    for record in records:
        norm_doi = normalise_doi(record.get("doi"))
        if norm_doi is not None and norm_doi in seen_doi:
            duplicates.append(
                DuplicatePair(dup_id=record["id"], canonical_id=seen_doi[norm_doi], matched_on="doi")
            )
            continue

        norm_title = _normalise_title(record["title"])
        title_match = None
        for kept in unique:
            ratio = difflib.SequenceMatcher(None, norm_title, _normalise_title(kept["title"])).ratio()
            if ratio >= title_threshold:
                title_match = kept["id"]
                break
        if title_match is not None:
            duplicates.append(
                DuplicatePair(dup_id=record["id"], canonical_id=title_match, matched_on="title")
            )
            continue

        unique.append(record)
        if norm_doi is not None:
            seen_doi[norm_doi] = record["id"]

    return DedupeResult(unique=unique, duplicates=duplicates)
```
