"""Structured data extraction for FT-included, full-text-confirmed
records -- Phase A: the engine only, not yet wired into any graph or
node (see PROTOCOL.md's own scope note, and Phase B in that same
document for what comes after this gate closes).

ARCHITECTURE: shared scientific contract, provider-specific execution,
NEVER auto-detection or silent substitution. This was the explicit
lesson drawn from surveying five other projects in this codebase before
writing a line of this package (see PROTOCOL.md's "prior art" section
for the full survey): il_outcomes' provider auto-detection made a run
LOOK unified while a local backend silently fell through to a different
method underneath, and that opacity is exactly what this package
refuses to repeat.

Shared across every provider (this module tree, used by every runner):
  - `schema.py` -- the extraction field-specification contract.
  - `parsing.py` -- the Docling parsed-chunk representation.
  - `evidence.py` -- evidence-validation rules (chunk trust, quotation
    verification).
  - `results.py`/`aggregation.py` -- the result and artifact schemas,
    and how per-batch observations combine into one final answer.
  - `checkpoint.py`/`run_identity.py` -- resumable checkpoints keyed by a
    run identity that binds every input that determines the science.
  - `calibration.py`/`reports.py` -- the calibration corpus contract and
    the coverage/yield-vs-audited-accuracy report split.

Provider-specific, never shared (one module per provider, never a branch
inside a shared one):
  - `openai_runner.py` -- the ONLY provider implemented in this phase.
    Explicit model/model_config_id/reasoning_effort/token budget,
    lazy environment-only credentials, structured output, provider-
    returned telemetry, no retry, no fallback to another model or
    provider.
  - A later `local_runner.py` (NOT built in this phase) would be a
    SEPARATE module with its own config/prompt/cache/checkpoint
    identity -- selected explicitly by whichever caller wires it in
    (e.g. a distinct `local_extraction` graph node, once Phase B
    exists), never auto-detected, never something this package falls
    back to when the OpenAI path is unavailable.

Provider results MAY be compared (a matched-corpus comparison between
providers is real, useful evidence) but are NEVER reconciled by a rule
like "OpenAI wins" -- a disagreement stays visible and goes to human
review, exactly like any other field-level disagreement this package
already produces.
"""
