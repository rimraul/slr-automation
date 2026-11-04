"""slr_pipeline.cost_cap -- the Track 3 live-run dollar-spend cap
(`track3_live_run_plan_2026-09-06.md` Sec 5, user authorization
2026-09-08, up to USD 30 total including retries, no reruns beyond that
without separate approval). This USD 30 cap is a Codex-calculated
MVP-scope authorization, distinct from -- and lower than -- Sec 5.4's
own USD 50 recommended-ceiling analysis; the pipeline's configured cap
must track the USD 30 figure actually authorized, not Sec 5.4's larger
number. (Date corrected 2026-09-10: this docstring previously read
2026-08-08, a typo for the actual 2026-09-08 authorization date --
`ledger.py`'s docstring already had it right.)

Two pieces:
  - `pricing`: the hardcoded, dated, independently-verified-twice
    USD-per-token table for the three real model IDs this pipeline
    actually calls (`gpt-5.4-mini`, `gpt-4o`, `gpt-5.6-terra`).
  - `ledger.CostLedger`: thread-safe reserve/commit/release running total
    against the cap, persisted to JSON.
  - `proxy.CostCapProxy`: a loopback HTTP server every real
    `OpenAI(api_key=...)` client in the graph transparently talks to via
    `OPENAI_BASE_URL`, enforcing the ledger's reservation gate BEFORE
    forwarding any request to the real OpenAI API.
"""
from .ledger import BudgetExceededError, CostLedger, UnpricedModelError
from .pricing import PRICING, actual_cost_usd, estimate_worst_case_cost_usd, is_priced
from .proxy import CostCapProxy

__all__ = [
    "BudgetExceededError",
    "CostLedger",
    "UnpricedModelError",
    "PRICING",
    "actual_cost_usd",
    "estimate_worst_case_cost_usd",
    "is_priced",
    "CostCapProxy",
]
