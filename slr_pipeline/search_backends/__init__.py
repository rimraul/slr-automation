"""
slr_pipeline.search_backends — package-native rewrites of the seven real
literature-database adapters (Track 2 plan §2.4-§2.6, step 3), plus the
Track 3 plan's Unit C `scopus` port (eight real adapters total).

Replaces the legacy `sys.path.insert(0, "c:/projects/screening")` imports
that `nodes/search.py` used to reach out-of-repo adapter modules. Every
backend here is self-contained: HTTPS-only, `HttpClient`-based, reads any
credential it needs from `os.environ` at call time (never from a request
object, never stored on `self`), and returns a `SearchBackendResult`.
"""
from __future__ import annotations

from slr_pipeline.search_backends.base import (
    SearchBackendRequest,
    SearchBackendResult,
    SearchConfigError,
)
from slr_pipeline.search_backends.registry import REAL_ADAPTER_CLASSES

__all__ = [
    "SearchBackendRequest",
    "SearchBackendResult",
    "SearchConfigError",
    "REAL_ADAPTER_CLASSES",
]
