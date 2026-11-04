# Search adapter matrix

Frozen contract for the seven real `slr_pipeline/search_backends/*` adapters
(Track 2 plan `track2_pipeline_readiness_plan_2026-08-30.md` §2.6, step 3).
This file is the table from §2.6 **verbatim** — step 3 implements it, adding
nothing. A live-doc mismatch discovered during implementation requires a new
plan revision (§0.3), not a silent implementation choice.

## Global rules (all six non-PubMed rewrites)

HTTPS; `timeout=(5,30)`; `Retry(3)`; 25 MiB response cap;
`adapter_config_version="1"`; `date_from`/`date_to` applied as the per-adapter
parameter named below, **omitted entirely when `None`**; `language` applied
only where the row below says the API supports it, otherwise recorded in
`DatabaseSearchLog.unsupported_filter_deviations`; a zero-hit response →
`retrieval_status="complete"`.

PubMed's self-contained `requests`+`xmltodict` implementation (formerly
`nodes/search.py::_search_pubmed`) is refactored into the same class shape
(`search_backends/pubmed.py`), its esearch/efetch logic unchanged.

## Deterministic `Record.id` — a safe-token, collision-safe formula

```
Record.id      = f"{source}-{hashlib.sha256(native_key.encode('utf-8')).hexdigest()[:20]}"
Record.source  = "<adapter name>"          # already [a-z_]-safe
Record.source_id = native_key               # the RAW value, verbatim (recoverable)
```

`source` is `[a-z_]`, the separator is `-`, the digest is 20 lowercase hex ⇒
`Record.id` always matches `ft_screening_adapter.py`'s
`_SAFE_ID_RE = ^[A-Za-z0-9_-]{1,128}$`. Collision-safe: two distinct
`native_key`s → different sha256 (prefix-truncated to 20 hex = 80 bits, ample
for a single review's corpus). The raw `native_key` lives in `Record.source_id`.

A source record whose `native_key` field is missing/empty is **skipped**
(counted in `skipped_malformed`) — the uuid fallback is never reached.
Fixture adapters (step 10) preserve their authored `Record.id` verbatim
(`fx-0001` … `fx-0030` — already `_SAFE_ID_RE`-safe); their `source` =
`fx_<db>`, their `source_id` = the authored `native_key` (`wos-t2-0002` etc).

| adapter | `native_key` (→ `Record.source_id`, RAW) |
|---|---|
| `openalex` | the OpenAlex work id, `id` with `https://openalex.org/` stripped (e.g. `W2755950973`) |
| `pubmed` | the PMID |
| `semantic_scholar` | `paperId` |
| `arxiv` | the arXiv id from `entry/id` (e.g. `2401.01234v1`) |
| `crossref` | the lowercased DOI (may contain `/`) |
| `springer` | the lowercased DOI |
| `ieee` | `article_number` |

## The complete adapter matrix

| adapter | endpoint (HTTPS) | auth (env var) | query param | pagination | page cap | date params | language | `Record` field map (source JSON → Record) |
|---|---|---|---|---|---|---|---|---|
| `openalex` | `https://api.openalex.org/works` | none; `OPENALEX_MAILTO` → `mailto` (polite pool) | `search=<query>` (as a `filter` component is also valid; this implementation uses the top-level `search` param) | **`cursor`** — first request `cursor=*`, then `cursor=<meta.next_cursor>` until `next_cursor` is null; **`per_page`** (underscore). **Never `page`** — OpenAlex basic `page` paging is capped at 10 000 results; cursor paging has no cap. | **`per_page=100`** (OpenAlex max is 100, not 200) | **inside `filter=`**: `filter=from_publication_date:{Y}-01-01,to_publication_date:{Y}-12-31` (comma-joined with the `search` filter if `search` is used as a filter; this implementation sends `search` top-level + `filter=` for dates). Omitted entirely when `date_from`/`date_to` are `None`. | not filtered; deviation recorded | `id`(strip `https://openalex.org/`)→`source_id` & `native_key`; `doi`(strip `https://doi.org/`)→`doi`; `title`→`title`; `abstract_inverted_index`→`abstract` (reconstructed); `authorships[].author.display_name`→`authors`; `publication_year`→`year`; `primary_location.source.display_name`→`journal`; `cited_by_count`→`cited_by`; `id`→`url`; `source="openalex"` |
| `pubmed` | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi` then `.../efetch.fcgi` | `PUBMED_API_KEY`→`api_key` (opt); `PUBMED_EMAIL`→`email` (opt) | `term` | esearch `retstart`/`retmax`, then efetch by PMID batch | 200 | `mindate="{Y}/01/01"` / `maxdate="{Y}/12/31"` + `datetype=pdat` | `term` gets ` AND {lang}[lang]` when set (supported) | efetch XML: `PMID`→`source_id`; `ELocationID[@EIdType="doi"]` or `ArticleId[@IdType="doi"]`→`doi`; `ArticleTitle`→`title`; `Abstract/AbstractText` (joined)→`abstract`; `Author` (`LastName`+" "+`Initials`)→`authors`; `PubDate/Year`→`year`; `Journal/Title`→`journal`; `None`→`cited_by`; `https://pubmed.ncbi.nlm.nih.gov/{PMID}/`→`url`; `source="pubmed"` |
| `semantic_scholar` | `https://api.semanticscholar.org/graph/v1/paper/search/bulk` | `S2_API_KEY`→header `x-api-key` (opt) | `query` | continuation `token` (response `token` → next request `token`); `fields=externalIds,title,abstract,authors,year,venue,citationCount,url` | 1000 | `year="{from}-{to}"` (or `"{from}-"` / `"-{to}"`) | not filtered; deviation recorded | `paperId`→`source_id`; `externalIds.DOI`→`doi`; `title`→`title`; `abstract`→`abstract`; `authors[].name`→`authors`; `year`→`year`; `venue`→`journal`; `citationCount`→`cited_by`; `url`→`url`; `source="semantic_scholar"` |
| `arxiv` | `https://export.arxiv.org/api/query` **(legacy `http://` — HTTPS rewrite)** | none | `search_query` | `start` (0-based), `max_results` | 100 | not supported → client-side post-filter on parsed `published` year; deviation recorded | not supported; deviation recorded | Atom XML: `entry/id`→`source_id`; `entry/arxiv:doi` or `None`→`doi`; `entry/title`→`title`; `entry/summary`→`abstract`; `entry/author/name`→`authors`; `entry/published`[:4]→`year`; `"arXiv"`→`journal`; `None`→`cited_by`; `entry/id`→`url`; `source="arxiv"` |
| `crossref` | `https://api.crossref.org/works` | none; `CROSSREF_MAILTO` → `mailto` | `query.bibliographic` | `cursor` (`*` → response `message.next-cursor`), `rows` | 200 | `filter=from-pub-date:{Y}-01-01,until-pub-date:{Y}-12-31` | not filtered; deviation recorded | `DOI`→`doi` & `source_id`; `title[0]`→`title`; `abstract` (JATS-stripped) or `None`→`abstract`; `author[]` (`given`+" "+`family`)→`authors`; `published.date-parts[0][0]`→`year`; `container-title[0]`→`journal`; `is-referenced-by-count`→`cited_by`; `URL`→`url`; `source="crossref"` |
| `springer` | `https://api.springernature.com/meta/v2/json` **(legacy `http://` — HTTPS rewrite)** | **`SPRINGER_API_KEY` (required)** → `api_key`; absent → `retrieval_status="failed"`, `failure_detail="SPRINGER_API_KEY not set"`, zero records | `q` | `s` (1-based start), `p` (page size) | 100 | `q` gets ` onlinedatefrom:{Y}-01-01 onlinedateto:{Y}-12-31` | `q` gets ` language:{language}` when set (supported) | `records[].doi`→`doi` & `source_id`; `records[].title`→`title`; `records[].abstract`→`abstract`; `records[].creators[].creator`→`authors`; `records[].publicationDate`[:4]→`year`; `records[].publicationName`→`journal`; `None`→`cited_by`; `records[].url[0].value`→`url`; `source="springer"` |
| `ieee` | `https://ieeexploreapi.ieee.org/api/v1/search/articles` **(legacy `http://` — HTTPS rewrite)** | **`IEEE_API_KEY` (required)** → `apikey`; absent → `retrieval_status="failed"`, `failure_detail="IEEE_API_KEY not set"`, zero records | `querytext` | `start_record` (1-based), `max_records`; response `total_records`→`total_hits` | 200 | `start_year={Y}` / `end_year={Y}` | not filtered; deviation recorded | `articles[].article_number`→`source_id`; `articles[].doi`→`doi`; `articles[].title`→`title`; `articles[].abstract`→`abstract`; `articles[].authors.authors[].full_name`→`authors`; `articles[].publication_year`→`year`; `articles[].publication_title`→`journal`; `articles[].citing_paper_count`→`cited_by`; `articles[].html_url`→`url`; `source="ieee"` |

## Live-doc re-verification (step 3, performed 2026-09-01)

OpenAlex/Semantic Scholar/Crossref/arXiv were already verified against
current public API docs at plan step 1. Springer and IEEE were re-verified
during step 3 implementation:

- **Springer**: `api.springernature.com/meta/v2/json`, `q`/`s`/`p` params,
  `api_key` auth param, `records[].{doi,title,abstract,publicationDate}`,
  `records[].creators[].creator`→authors, `records[].publicationName`→journal
  (confirmed against SearXNG's production Springer-parsing source, which
  actually calls this API), `records[].url[...].value` list-of-dicts shape —
  all cells confirmed correct, no plan contradiction found.
- **IEEE**: `article_number`, `authors.authors[].full_name`,
  `citing_paper_count`, `publication_title`, `apikey`/`querytext`/
  `start_record`/`max_records` (max 200)/`start_year`/`end_year`/
  `total_records` — all confirmed via IEEE's official field definitions, no
  plan contradiction found.

No plan revision was required for step 3.
