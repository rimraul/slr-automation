"""A module that deliberately performs a real socket connection attempt AND
reads a provider credential at IMPORT TIME (module level, not inside a
function) -- simulating a badly-written E.2b exercise module that touches
the network/credentials as a side effect of merely being imported, before
any per-test fixture's setup phase would ever run.

Not itself a pytest test file (its name matches neither `test_*.py` nor
`*_test.py`), so it is never auto-collected -- `test_offline_contract_smoke.py`
imports it at ITS OWN module level (see that file's own docstring for why),
to prove that the offline contract (established process-wide in
`chapter_offline_contract_plugin.py::pytest_configure`, which runs before
collection/import begins for the whole session) is already active by the
time this import happens, module-level code included.

Round-5 review finding 2: the previous version pre-classified the outcome
into a "blocked: ..."/"succeeded (BUG...)" string here, accepting ANY
exception type as proof of "blocked" -- a DNS failure, timeout, firewall
denial, or plain connection refusal (any of which can occur in a sandboxed/
offline CI environment with zero involvement from pytest-socket) would have
been wrongly accepted as proof. Fixed: this module now stores the raw
exception OBJECT (or `None` if the call unexpectedly succeeded), and the
consuming test asserts its exact type is `pytest_socket.SocketBlockedError`
-- not just "some exception happened."
"""

from __future__ import annotations

import os
import socket

try:
    socket.create_connection(("example.com", 80), timeout=2)
except Exception as exc:  # capture whatever actually happened; the exact TYPE is asserted by the consuming test, not here
    IMPORT_TIME_NETWORK_EXCEPTION: "Exception | None" = exc
else:
    IMPORT_TIME_NETWORK_EXCEPTION = None

IMPORT_TIME_CREDENTIAL_VALUE = os.environ.get("OPENAI_API_KEY")
