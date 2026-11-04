"""
slr_pipeline.replay -- the offline replay-transport boundary (Track 2 plan
`track2_pipeline_readiness_plan_2026-08-30.md` Sec 6, step 10):

- `signature.py` -- RFC 8785 JCS (`rfc8785_jcs()`, the ONE definition every
  other `rfc8785_jcs(...)` citation in the plan imports, Sec 6.1
  round-18 clarification) and the request-signature formula.
- `cassette.py` -- the strict `REPLAY_CASSETTE_VERSION = "1"` schema.
- `transport.py` -- the loopback `POST /v1/chat/completions` server.
- `fixture_search_adapter.py` -- the seven `fx_*` search adapters that
  complete `nodes/search.py::_ADAPTER_REGISTRY` (Sec 2.5).
"""
from __future__ import annotations

from slr_pipeline.replay.cassette import (
    REPLAY_CASSETTE_VERSION,
    REFERENCE_SURFACE_CALL_COUNTS,
    ReplayCassette,
    ReplayCassetteEntry,
    ReplaySurface,
)
from slr_pipeline.replay.signature import (
    REPLAY_SIGNATURE_VERSION,
    compute_replay_signature,
    rfc8785_jcs,
)
from slr_pipeline.replay.transport import ReplayServer, ReplayTransportError

__all__ = [
    "REPLAY_CASSETTE_VERSION",
    "REFERENCE_SURFACE_CALL_COUNTS",
    "ReplayCassette",
    "ReplayCassetteEntry",
    "ReplaySurface",
    "REPLAY_SIGNATURE_VERSION",
    "compute_replay_signature",
    "rfc8785_jcs",
    "ReplayServer",
    "ReplayTransportError",
]
