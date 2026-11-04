"""Track 3 live-run launcher plan (track3_live_run_plan_2026-09-06.md),
Unit C (squishy-petting-starlight.md, C2/C5): the live-run **parent**
driver.

Architecturally cloned from `reference_run.py`/`reference_run_child.py`
(Track 2's own parent/child pair), with the simplifications a genuinely
live run allows and the safety additions it requires:

  - No cassette/corpus-oracle/pdf-source/fixture-corpus-root arguments --
    those are Track-2-only replay affordances. The child fetches full text
    over the real network (`nodes/fulltext.py`'s own Unpaywall/Semantic
    Scholar/EuropePMC resolution + download), so `full_text_acquisition`
    in the produced `run_mode.json` is `"live_download"`, not
    `"local_prepositioned"`.
  - `OPENAI_BASE_URL` points at a `slr_pipeline.cost_cap.proxy.CostCapProxy`
    (real forwarding to `https://api.openai.com`, ledger-enforced dollar
    cap), not a `ReplayServer` (canned responses, cassette-bound).
  - C2 -- WoS-impossible, defense in depth (plan Unit C): this module
    asserts `set(cfg.search.databases) <= {"pubmed", "scopus"}` via
    `assert_databases_allowed()` BEFORE any subprocess/port/proxy setup --
    "the single most safety-critical check in this unit" per the plan.
    This is layer 2; layer 1 is `search_backends/registry.py` never
    registering a `"wos"`/`"web_of_science"` adapter class at all --
    `nodes/search.py::_resolve_adapter_keys()` fail-closed raises
    `SearchConfigError` for any db key not in that registry, so even a
    hypothetical bug in THIS module's own assertion still cannot reach a
    live WoS call. The child (`track3_run_child.py`) independently
    re-runs the same assertion against the same spec file -- never trusts
    that the parent already checked it (defense in depth means two
    independent checks, not one check two processes both rely on).
  - C3 -- explicit scope decision: this launcher does NOT arm a
    process-wide network guard (unlike `reference_run_child.py`'s
    `ReferenceNetGuard`, a single-`(host,port)`-pair design incompatible
    with a live run legitimately needing PubMed/Scopus/OpenAI hosts). The
    two C2 layers above already make a WoS call structurally impossible
    from this pipeline's own code paths; a hostname-pattern-aware
    multi-host guard variant is a real, but explicitly deferred, post-MVP
    hardening candidate -- not a silent omission. Consequently there is no
    `violations.jsonl` (that file was `ReferenceNetGuard`-specific); the
    workspace's terminal artifacts are `terminal_state.json`,
    `terminal_graph_snapshot.json`, `cost_cap_call_log.jsonl`,
    `run_mode.json`, `run_mode_statement.md`.
  - Cost-cap enforcement does not depend on a network guard at all -- it
    depends solely on no real call site ever passing an explicit
    `base_url=` override to the OpenAI SDK (C4). Scoped precisely to
    what THIS launcher can reach -- `track3_run_child.py` invokes only
    `graphs/scoping.py::build_scoping_graph()`, never `graphs/rapid.py`
    or `graphs/slr.py` -- confirmed: all 5 real OpenAI call sites
    reachable from `build_scoping_graph()`'s own node list --
    `search_strategy_builder.py`, `screening.py`, `screening_qc.py`,
    `extraction/openai_runner.py`, `synthesis.py` -- use
    `OpenAI(api_key=resolve_openai_api_key())` with no `base_url=`, so
    `OPENAI_BASE_URL` transparently redirects every real call through the
    proxy with zero node-code changes. `nodes/quality_appraisal.py` also
    constructs `OpenAI(api_key=resolve_openai_api_key(), max_retries=0)`
    directly (no `base_url=` either) and IS one of `cost_cap/proxy.py`'s
    own round-19-audited in-repo sites -- but `quality_appraisal_node` is
    never added to `build_scoping_graph()`'s node list (confirmed
    directly), so it is unreachable from any Track 3 live run by
    construction, not merely omitted from this count.

    There is a SIXTH real, reachable call site OUTSIDE this repo:
    `fulltext_node -> run_ft_screening() -> resolve_components()`
    dispatches into the separate FT_SCREENING package's
    `ft_screening/adjudication.py` (`client = OpenAI(api_key=api_key)`,
    also no `base_url=` today, confirmed directly against that file).
    This launcher's own `OPENAI_BASE_URL` env var still redirects it the
    same way (it's a plain env var read by the OpenAI SDK, not something
    this repo's code has to opt into per call site) -- but
    `cost_cap/proxy.py`'s own round-19 comment already flags that this
    site is NOT covered by this proxy's own test suite or this audit's
    guarantees the way the in-repo sites are: a future FT_SCREENING
    change adding a `base_url=` there would silently escape the cap with
    zero warning from anything in THIS repo. Flagged as a known, accepted
    gap for this MVP launcher, not a functional bypass today.

Scope boundary (identical to Track 2's own step-13 boundary,
`reference_run.py`'s own docstring): this driver's job ends once the
child exits 0 and its workspace files are confirmed present. It does
**not** call `finalize_bundle()`/`create_payload()` -- bundle publication
is out of scope for this plan, and the MVP live run itself is gated on a
separate user go-ahead (dashboard-limit confirmation + these launcher
acceptance gates passing) -- not part of this deliverable.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

from slr_pipeline import credentials
from slr_pipeline.reference_run import _reserve_loopback_port

REPO_ROOT = Path(__file__).resolve().parent.parent
# Same read-only bundle Track 2's own `reference_run.py:98` uses -- gitignored
# (`.gitignore:30`; ~1.2GB, reconstructed via `ft_screening.docling_artifacts.
# bootstrap()` or a release-asset download, never `git clone` alone), but a
# shared, verified-digest asset relative to THIS checkout, not per-run
# ephemeral state -- so pointing a live run's DOCLING_CACHE_DIR here does not
# compromise the per-run `caches/` isolation `build_child_env()` otherwise
# maintains.
DOCLING_CACHE_DIR = REPO_ROOT / "assets" / "models" / "track2-docling-cache"

_WORKSPACE_FILES = (
    "terminal_state.json", "terminal_graph_snapshot.json", "cost_cap_call_log.jsonl",
    "run_mode.json", "run_mode_statement.md",
)

# Records the thread_id a workspace was allocated under, so `resume_workspace=`
# can fail fast on a mismatch instead of silently starting a fresh, re-spending
# run (see `run_track3()`'s `resume_workspace` docstring paragraph).
_THREAD_ID_FILENAME = "thread_id.txt"

# C2 (plan Unit C): the ONLY two databases this MVP launcher will ever pass
# to the child -- WoS is structurally excluded here, in addition to never
# being registered in search_backends/registry.py (layer 1).
ALLOWED_DATABASES = frozenset({"pubmed", "scopus"})

# Unit H (squishy-petting-starlight.md): a placeholder minimum-viable-corpus
# threshold for enabling `cfg.synthesis.run_bertopic`, gating the general
# `assert_bertopic_corpus_size_or_disabled()` mechanism below. Not yet
# confirmed against topic-modelling literature/BERTopic's own documented
# guidance -- deliberately provisional, to be revisited by a future caller
# who actually wants `run_bertopic=True` on a real corpus.
BERTOPIC_MIN_CORPUS_SIZE = 20

_WINDOWS_RUNTIME_ALLOWLIST = (
    "SystemRoot", "windir", "ComSpec", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)

# Search-credential env vars threaded through to the child, when present in
# the parent's own environment (plan §2.2's resolution) -- never required
# (PubMed's own adapter treats both as optional, NCBI etiquette only).
# SCOPUS_API_KEY and SCOPUS_INSTITUTIONAL_KEY are BOTH required whenever
# "scopus" is requested -- see `assert_search_credentials_present()` below --
# so neither appears in this optional list.
_OPTIONAL_SEARCH_CREDENTIAL_ENV_VARS = ("PUBMED_API_KEY", "PUBMED_EMAIL")


class Track3ConfigError(ValueError):
    """Raised by a preflight check, before any subprocess/port/proxy setup."""


def assert_databases_allowed(databases) -> None:
    """C2, plan Unit C -- "the single most safety-critical check in this
    unit". Raises BEFORE any subprocess is spawned, any port reserved, or
    any proxy started, if `databases` contains anything other than
    `pubmed`/`scopus` (in particular: `wos`/`web_of_science`, under any
    spelling that could appear in a hand-authored spec file). Also rejects
    an empty list -- a Track 3 run with zero databases configured is a
    misconfiguration, not a valid degenerate case."""
    requested = set(databases)
    if not requested or not requested <= ALLOWED_DATABASES:
        raise Track3ConfigError(
            f"Track 3 MVP launcher only allows databases {sorted(ALLOWED_DATABASES)} -- "
            f"got {sorted(requested)}. WoS is never permitted for this launcher."
        )


def assert_no_execution_adapter_map(execution_adapter_map: Optional[Dict[str, str]]) -> None:
    """Defense in depth alongside `assert_databases_allowed()`: that check
    only validates `cfg.search.databases` (e.g. `["pubmed", "scopus"]`), but
    `nodes/search.py::_resolve_adapter_keys()` supports a separate mapped-
    dispatch mode via `cfg.search.execution_adapter_map` that can point any
    database name at ANY `_ADAPTER_REGISTRY` key -- including a fixture
    adapter (`fx_pubmed`, `fx_scopus`, even `fx_web_of_science`), which reads
    local fixture files and never touches the real network. Without this
    check, a spec with `databases=["pubmed", "scopus"]` (passing
    `assert_databases_allowed()`) but
    `execution_adapter_map={"pubmed": "fx_pubmed", "scopus": "fx_scopus"}`
    would silently run entirely against local fixture data -- zero real
    spend, zero real search -- while `build_run_mode_document_live()` still
    unconditionally records `search_source="live"`, falsely certifying the
    run. A genuine Track 3 live run has no legitimate reason to set this
    field at all (it exists for Track 2 replay specs), so it is rejected
    outright rather than merely value-checked."""
    if execution_adapter_map is not None:
        raise Track3ConfigError(
            "Track 3 MVP launcher does not permit cfg.search.execution_adapter_map "
            "(Track 2 replay-only field) -- it can silently redirect real databases "
            "to fixture adapters, bypassing the live search this launcher exists to run"
        )


def assert_search_credentials_present(databases, *, base_env: Dict[str, str]) -> None:
    """Scopus fails closed with a silent 0-record search when its API key is
    missing (`search_backends/scopus.py`'s own `finalize_result(...,
    failure_detail=...)` path) -- for a real, money-spending run that would
    waste the whole run's spend on a search that structurally cannot see
    Scopus coverage. Checked eagerly, before any subprocess/port/proxy
    setup, same as `assert_databases_allowed()`. PubMed has no equivalent
    hard requirement (its own adapter treats both PUBMED_API_KEY/
    PUBMED_EMAIL as optional -- NCBI etiquette, not an auth requirement).

    BOTH SCOPUS_API_KEY and SCOPUS_INSTITUTIONAL_KEY are required whenever
    'scopus' is requested -- user-mandated dual-credential gate (2026-09-11):
    an institution-scoped Scopus subscription's real content coverage
    depends on the institutional token being sent, not just a bare API key,
    so a run missing it would silently under-cover Scopus results for a
    real, money-spending run without any visible failure."""
    missing = []
    if "scopus" in set(databases):
        if not base_env.get("SCOPUS_API_KEY"):
            missing.append("SCOPUS_API_KEY")
        if not base_env.get("SCOPUS_INSTITUTIONAL_KEY"):
            missing.append("SCOPUS_INSTITUTIONAL_KEY")
    if missing:
        raise Track3ConfigError(
            f"{' and '.join(missing)} must be set in the environment when 'scopus' is a "
            "requested database"
        )


def assert_dedup_mode_exact(dedup_mode: str) -> None:
    """`reference_run_mode.py::build_run_mode_document_live()`'s own
    docstring (Unit B's preflight note): that builder fail-closes on an
    unrecognized `AIUseEvent` stage (e.g. `dedup_embedding`, which
    `nodes/dedup.py` dispatches to whenever `cfg.dedup.mode != "exact"`) --
    but only AFTER `graph.invoke()` has already returned, i.e. after real
    spend has already occurred. This launcher's own preflight check catches
    a misconfigured `dedup.mode` BEFORE any spend, not after."""
    if dedup_mode != "exact":
        raise Track3ConfigError(f"Track 3 MVP launcher requires cfg.dedup.mode == 'exact', got {dedup_mode!r}")


def assert_bertopic_corpus_size_or_disabled(cfg, n_eligible_documents: int) -> None:
    """Unit H (squishy-petting-starlight.md): general preflight mechanism for
    any caller enabling `cfg.synthesis.run_bertopic` -- raises BEFORE any
    subprocess/port/proxy setup, same discipline as the other `assert_*`
    checks in this module, if the corpus is too small for a meaningful
    topic model. A no-op whenever `run_bertopic` is `False` regardless of
    corpus size (the flag being off makes the check moot, not violated)."""
    if cfg.synthesis.run_bertopic and n_eligible_documents < BERTOPIC_MIN_CORPUS_SIZE:
        raise Track3ConfigError(
            f"cfg.synthesis.run_bertopic=True requires at least {BERTOPIC_MIN_CORPUS_SIZE} eligible "
            f"documents for a meaningful topic model, got {n_eligible_documents}"
        )


def assert_docling_artifacts_valid(models_dir: Path) -> None:
    """Checked eagerly, before any subprocess/port/proxy setup, same as the
    other preflight asserts above. `build_child_env()` points the child's
    `DOCLING_CACHE_DIR` at `DOCLING_CACHE_DIR` (this module's constant);
    FT_SCREENING's own `DoclingDocumentParser` (no `artifacts_path`
    override configurable from `ScreeningConfig` today) resolves its
    default artifacts directory from Docling's own `AppSettings.cache_dir`
    -- a pydantic `BaseSettings` with `env_prefix="DOCLING_"`, so that same
    env var steers FT_SCREENING's resolution too, not only
    `extraction/parsing.py`'s separate Track-2 manifest check. Without this
    preflight, a missing/corrupt bundle is only discovered lazily, inside
    `_ensure_artifacts_resolved()`, on the first real PDF parse -- i.e.
    after search and screening have already spent real money. Runs the
    identical content-hash check `_ensure_artifacts_resolved()` runs later
    (`ft_screening.docling_artifacts.validate()`), just eagerly. Accepted,
    deliberate tradeoff: the child's own `DoclingDocumentParser` still
    re-hashes the same bundle once more on its own first real PDF parse --
    FT_SCREENING exposes no way to inject a pre-verified digest or skip its
    own check (v0.1's `DoclingDocumentParser()` has no such config surface,
    and changing that is out of scope -- FT_SCREENING is a separate,
    read-only-reference project here). That's one extra ~1.2GB tree-hash
    pass (~seconds) per live run, negligible against the run's own overall
    duration, in exchange for failing BEFORE any paid call instead of after
    (the original bug this preflight exists to close). Loads the manifest
    itself first, rather than letting `validate()` do it internally, so a
    broken/corrupt manifest resource (`load_manifest()` raises `OSError`
    on a missing file or `ValueError` -- `json.JSONDecodeError` is a
    subclass -- on invalid JSON; subscripting `["expected_tree_digest"]`
    raises `KeyError` on a manifest missing/renaming that key, or
    `TypeError` if the manifest JSON parses to something other than an
    object (e.g. a list or scalar) -- e.g. a broken `ft_screening`
    install) is reported with its own distinct,
    accurate `Track3ConfigError` message, never confused with an actual
    content-hash mismatch OR with an unrelated `OSError` while reading a
    bundle file during hashing (e.g. an antivirus lock or a
    permission-denied file, not necessarily a bad/incomplete bundle) --
    that second case gets its own distinct message too, both fail fast
    with a friendly `Track3ConfigError`, never a raw exception, matching
    every sibling assert above."""
    from ft_screening import docling_artifacts
    try:
        manifest = docling_artifacts.load_manifest()
        expected_tree_digest = manifest["expected_tree_digest"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Track3ConfigError(
            f"Docling artifacts manifest could not be loaded from FT_SCREENING's packaged "
            f"resource -- likely a broken/incomplete ft_screening install: {exc}"
        ) from exc
    try:
        docling_artifacts.validate(models_dir, expected_tree_digest=expected_tree_digest)
    except docling_artifacts.ArtifactValidationError as exc:
        raise Track3ConfigError(
            f"Docling model artifacts at {models_dir!s} failed manifest validation "
            f"against FT_SCREENING's committed manifest: {exc}"
        ) from exc
    except OSError as exc:
        raise Track3ConfigError(
            f"Docling model artifacts at {models_dir!s} could not be read while computing "
            f"their content hash (e.g. a locked or permission-denied file, not necessarily a "
            f"bad bundle): {exc}"
        ) from exc


def assert_openai_credential_present(*, base_env: Dict[str, str]) -> None:
    """Checked eagerly, before any subprocess/port/proxy/workspace setup,
    same as the other three preflight asserts above -- `build_child_env()`
    also re-checks this (defense in depth, same rationale as C2), but that
    check runs too late to be a true preflight: it fires only after
    `allocate_workspace()` has already created a real directory and
    `_reserve_loopback_port()` has already bound/closed a real socket. That
    late `Track3ConfigError` IS still cleaned up when `keep_workspace=False`
    (via `run_track3()`'s own broad exception handling around the child
    invocation), but only this earlier preflight avoids creating the
    directory/socket in the first place."""
    if not base_env.get("OPENAI_API_KEY"):
        raise Track3ConfigError("OPENAI_API_KEY must be set in the environment for a Track 3 live run")


def allocate_workspace(base_tmp_dir: Path) -> Path:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = base_tmp_dir / f"track3_run_{ts}_{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def build_child_env(*, workspace: Path, proxy_port: int, python_executable: str,
                     base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Fresh dict, not `os.environ.copy()` (same discipline as Track 2's own
    `reference_run.py::build_child_env()`) -- every value either a fixed
    constant, derived from `workspace`, or copied from `base_env` (defaults
    to the real `os.environ`) for the small, explicit allowlist below.
    `cost_cap_path` is NOT an env var -- it's threaded to the child only via
    `build_child_argv()`'s own `--cost-cap-path` flag. Unlike Track 2, the
    real `OPENAI_API_KEY` IS passed through (the proxy relays the real
    Authorization header verbatim, never reading it itself --
    `cost_cap/proxy.py`'s own docstring), and real search credentials are
    threaded through explicitly."""
    base_env = os.environ if base_env is None else base_env
    assert_openai_credential_present(base_env=base_env)
    caches_dir = workspace / "caches"
    for sub in ("home", "huggingface", "torch", "matplotlib"):
        (caches_dir / sub).mkdir(parents=True, exist_ok=True)
    env: Dict[str, str] = {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{proxy_port}/v1",
        "OPENAI_API_KEY": base_env["OPENAI_API_KEY"],
        # DOCLING_CACHE_DIR points at the shared, read-only, digest-verified
        # bundle at DOCLING_CACHE_DIR (module constant) -- same bundle Track 2's
        # own reference_run.py:98 uses. Docling's own AppSettings (pydantic
        # BaseSettings, env_prefix="DOCLING_") means this env var steers
        # FT_SCREENING's DoclingDocumentParser default resolution too, not only
        # extraction/parsing.py's separate Track-2 manifest check -- without it,
        # every child gets a fresh HOME (below) with no docling cache, so
        # DoclingDocumentParser fails at first real PDF parse. No
        # HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE still (unlike Track 2): forcing
        # offline mode would break a `bertopic="enabled"` live run -- `nodes/
        # synthesis.py` loads a real `SentenceTransformer("all-MiniLM-L6-v2")`,
        # and this run's own HF_HOME below is a fresh, empty per-run directory
        # -- a live run legitimately needs real network to populate it, same
        # as its PubMed/Scopus/OpenAI traffic.
        "DOCLING_CACHE_DIR": str(DOCLING_CACHE_DIR.resolve()),
        "PYTHONHASHSEED": "0",
        "NO_PROXY": "127.0.0.1",
        "SCREENING_BATCH_SIZE": "5",
        "SCREENING_WORKERS": "1",
        "TEMP": str(caches_dir),
        "TMP": str(caches_dir),
        "HF_HOME": str(caches_dir / "huggingface"),
        "TORCH_HOME": str(caches_dir / "torch"),
        "MPLCONFIGDIR": str(caches_dir / "matplotlib"),
        "XDG_CACHE_HOME": str(caches_dir),
        "HOME": str(caches_dir / "home"),
        "USERPROFILE": str(caches_dir / "home"),
    }
    if base_env.get("SCOPUS_API_KEY"):
        env["SCOPUS_API_KEY"] = base_env["SCOPUS_API_KEY"]
    if base_env.get("SCOPUS_INSTITUTIONAL_KEY"):
        env["SCOPUS_INSTITUTIONAL_KEY"] = base_env["SCOPUS_INSTITUTIONAL_KEY"]
    for name in _OPTIONAL_SEARCH_CREDENTIAL_ENV_VARS:
        if base_env.get(name):
            env[name] = base_env[name]
    for name in _WINDOWS_RUNTIME_ALLOWLIST:
        if name in base_env:
            env[name] = base_env[name]
    venv_scripts = str(Path(python_executable).resolve().parent)
    system32 = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32")
    env["PATH"] = os.pathsep.join([venv_scripts, system32])
    return env


def build_child_argv(*, python_executable: str, workspace: Path, spec_path: Path, output_dir: Path,
                      cost_cap_path: Path, cap_usd: float, thread_id: str) -> list:
    """No cassette/corpus-oracle/pdf-source/fixture-corpus-root flags
    (Track-2-only replay affordances). `--databases` is deliberately NOT
    passed here -- the child derives it from `--spec` itself and
    independently re-checks `assert_databases_allowed()` against that same
    value (defense in depth, see module docstring)."""
    return [
        python_executable, "-m", "slr_pipeline.track3_run_child",
        "--workspace", str(workspace),
        "--spec", str(spec_path),
        "--output-dir", str(output_dir),
        "--cost-cap-path", str(cost_cap_path),
        "--cap-usd", str(cap_usd),
        "--thread-id", thread_id,
    ]


class Track3RunError(RuntimeError):
    """Raised when the child subprocess exits non-zero, or exits zero but is
    missing one of its required workspace files."""


def _cleanup_workspace_if_requested(workspace: Path, *, keep_workspace: bool) -> None:
    """Single cleanup helper so every failure branch in `run_track3()` honors
    `keep_workspace=False` identically -- prevents future edits from updating
    only some of the failure branches."""
    if not keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)


def run_track3(*, spec_path: Path, base_tmp_dir: Optional[Path] = None, cap_usd: float = 30.0,
                thread_id: str = "track3-live-run", python_executable: str = sys.executable,
                keep_workspace: bool = True, timeout_s: Optional[float] = None,
                resume_workspace: Optional[Path] = None) -> Path:
    """Runs the child subprocess to completion and returns the workspace
    path. Six preflight assertions (C2's WoS-impossible allowlist,
    no-execution-adapter-map, dedup-mode-exact, Scopus-credential-present,
    OpenAI-credential-present, Docling-artifacts-valid) all run here, in-process, BEFORE any
    subprocess is spawned, any port reserved, any proxy started, or any
    workspace directory created -- a failing config never gets as far as
    touching the network or the
    filesystem beyond reading `spec_path` itself.

    `spec_path` is resolved to an absolute path immediately, before any of
    the preflight asserts run: the child subprocess is spawned with
    `cwd=str(workspace)` (a different directory than wherever this function
    was called from), so a caller-supplied relative path (a realistic CLI
    invocation, e.g. `--spec my_review.json` from a project directory)
    would otherwise resolve correctly here but fail in the child with an
    opaque `FileNotFoundError` -> non-zero exit -> `Track3RunError`.
    `base_tmp_dir` gets the same resolve-to-absolute treatment (below, right
    before `allocate_workspace()`) for the identical reason: `workspace`/
    `output_dir`/`cost_cap_path` are all derived from it and passed to the
    child verbatim via `--workspace`/`--output-dir`/`--cost-cap-path`; since
    the child's own `cwd` is already set to `workspace`, a relative
    `base_tmp_dir` would otherwise make the child re-resolve an
    already-relative-from-the-parent path a second time, against its own
    new `cwd`, producing a doubled/nonexistent nested directory instead of
    the real workspace. `base_tmp_dir` has no CLI flag today (defaults to
    `TEMP`/`TMP`, always absolute on Windows), so this is a latent gap in
    `run_track3()`'s keyword-argument contract, not a live CLI-reachable one
    -- fixed anyway, on the same reasoning as the `spec_path` fix above.

    `python_executable` gets the same resolve-to-absolute treatment (below,
    alongside `spec_path`) for the identical reason: `build_child_argv()`
    uses it verbatim as `argv[0]`, and the child subprocess is spawned with
    `cwd=str(workspace)` -- a relative `python_executable` (e.g.
    `"venv\\Scripts\\python.exe"`, a legitimate keyword-argument use, same
    as `base_tmp_dir` before its own fix) would resolve fine against the
    caller's cwd but be re-resolved by the OS against the child's `cwd`
    instead, raising a raw `FileNotFoundError`/`WinError 2` rather than a
    working run. Defaults to `sys.executable`, which is always absolute, so
    this is a latent gap in the keyword-argument contract only, no CLI flag
    exposes it today -- fixed anyway, same reasoning as `spec_path` and
    `base_tmp_dir` above. Note `build_child_env()` separately does its own
    `Path(python_executable).resolve()` to compute the child's `PATH`
    env var -- that call was already safe on its own since `.resolve()`
    doesn't mutate its input, but relied on this function passing an
    already-correct value into `build_child_argv()` too, which it did not
    do before this fix.

    `timeout_s` defaults to `None` (unbounded), a deliberate scope decision
    like C3's own network-guard call: a genuinely live SLR run's true
    duration (real PubMed/Scopus/full-text-download/OpenAI latency across
    however many records it finds) isn't knowable in advance, so this
    launcher does not impose an arbitrary cutoff that risks killing a
    legitimate long-running run. The parameter exists so a caller who does
    want a circuit breaker (e.g. a smoke test) can opt in explicitly.

    Once `allocate_workspace()` has created a real directory, EVERY
    exception (including `BaseException` subclasses like `KeyboardInterrupt`,
    not just `Exception`) from `_reserve_loopback_port()`/`build_child_env()`/
    `build_child_argv()`/`subprocess.run()` (including
    `subprocess.TimeoutExpired` and a stray `Track3ConfigError` from
    `build_child_env()`'s own defense-in-depth re-check, e.g. if
    `OPENAI_API_KEY` were removed from the environment in the narrow window
    after this function's own earlier preflight check) honors
    `keep_workspace=False`, same as the two explicit failure branches
    below.

    `resume_workspace`, when given, reuses that existing directory (from a
    prior `keep_workspace=True` run of this same function, e.g. after a
    crash) instead of calling `allocate_workspace()` for a brand-new one --
    the actual operator-facing entry point into the crash-resume machinery
    in `track3_run_child.py::_invoke_graph()` (its workspace-relative
    `graph_checkpoints.sqlite`, `CostLedger`'s own on-disk resume, both
    keyed off files inside this same directory). Verified before reuse: a
    `graph_checkpoints.sqlite` file must already exist in the directory --
    opening the sqlite checkpointer creates this file immediately, before
    any graph node runs, so its presence is a reliable signal the
    directory was genuinely used by a prior `track3_run_child.py`
    invocation (unlike `cost_cap_ledger.json`, which `CostLedger` only
    writes after the FIRST committed spend, so would not yet exist after a
    very-early crash).
    `thread_id` MUST match the original run's `thread_id` for the graph
    checkpoint to actually be found by `graph.get_state(...)` -- a mismatch
    would otherwise silently start a fresh graph run instead of resuming
    (no error), re-spending on already-paid-for stages. To close that off,
    every freshly allocated workspace records its `thread_id` to a
    `thread_id.txt` file; resuming re-reads that file and raises
    `Track3ConfigError` up front (before any subprocess/port/proxy setup)
    on a missing file or a mismatch, rather than silently degrading to a
    fresh, re-spending run. A resumed workspace is also never deleted by
    this function's own failure-cleanup paths, even when
    `keep_workspace=False` -- the crashed run's checkpoint/ledger state
    that made resuming possible in the first place must survive a second
    failed attempt too, so cleanup-on-failure only ever applies to a
    workspace this same call freshly allocated."""
    from slr_pipeline.config import ReviewConfig

    spec_path = Path(spec_path).resolve()
    python_executable = str(Path(python_executable).resolve())
    cfg = ReviewConfig.from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
    assert_databases_allowed(cfg.search.databases)
    assert_no_execution_adapter_map(cfg.search.execution_adapter_map)
    assert_dedup_mode_exact(cfg.dedup.mode)

    # Credential names requested mirror track3_run.py:208's own dual-key
    # preflight check exactly -- Scopus credentials are only requested when
    # Scopus is actually configured. loaded_credentials() wraps everything
    # downstream that actually spends: the credential preflight asserts,
    # workspace setup, and the subprocess.run() call that performs the live
    # run -- entered only after the (unchanged) --cap-usd/authorization
    # validation in main() has already passed.
    names = (
        ("OPENAI_API_KEY", "SCOPUS_API_KEY", "SCOPUS_INSTITUTIONAL_KEY")
        if "scopus" in set(cfg.search.databases)
        else ("OPENAI_API_KEY",)
    )
    with credentials.loaded_credentials(names):
        assert_search_credentials_present(cfg.search.databases, base_env=os.environ)
        assert_openai_credential_present(base_env=os.environ)
        assert_docling_artifacts_valid(DOCLING_CACHE_DIR / "models")

        resumed = resume_workspace is not None
        if resumed:
            workspace = Path(resume_workspace).resolve()
            if not workspace.is_dir():
                raise Track3ConfigError(f"resume_workspace {workspace!s} is not an existing directory")
            thread_id_path = workspace / _THREAD_ID_FILENAME
            if not thread_id_path.is_file():
                raise Track3ConfigError(
                    f"resume_workspace {workspace!s} has no {_THREAD_ID_FILENAME} recorded from its "
                    "original run -- cannot verify thread_id matches, refusing to risk a silent "
                    "fresh (re-spending) run instead of a true resume"
                )
            recorded_thread_id = thread_id_path.read_text(encoding="utf-8").strip()
            if recorded_thread_id != thread_id:
                raise Track3ConfigError(
                    f"thread_id mismatch resuming {workspace!s}: original run used "
                    f"{recorded_thread_id!r}, got {thread_id!r} -- the graph checkpoint would "
                    "silently NOT be found under the wrong thread_id, re-spending on stages "
                    "already paid for; pass the original --thread-id"
                )
            if not (workspace / "graph_checkpoints.sqlite").is_file():
                raise Track3ConfigError(
                    f"resume_workspace {workspace!s} has no graph_checkpoints.sqlite -- it was never "
                    "actually used by a prior track3_run_child.py invocation (opening the sqlite "
                    "checkpointer creates this file immediately, before any graph node runs), "
                    "refusing to treat an arbitrary directory as a valid resume target"
                )
        else:
            base_tmp_dir = Path(base_tmp_dir or os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp").resolve()
            workspace = allocate_workspace(base_tmp_dir)
        output_dir = workspace / "review_output"
        cost_cap_path = workspace / "cost_cap_ledger.json"
        # A resumed workspace holds a prior crashed run's checkpoint/ledger state --
        # never let this call's own failure-cleanup delete it, even if the caller
        # passed keep_workspace=False for THIS attempt.
        keep_workspace = keep_workspace or resumed

        try:
            if not resumed:
                # Inside try/except BaseException (below) so a failure writing this
                # file still honors keep_workspace=False -- allocate_workspace()
                # above has already created a real directory by this point.
                (workspace / _THREAD_ID_FILENAME).write_text(thread_id, encoding="utf-8")
            proxy_port = _reserve_loopback_port()
            child_env = build_child_env(
                workspace=workspace, proxy_port=proxy_port, python_executable=python_executable,
            )
            argv = build_child_argv(
                python_executable=python_executable, workspace=workspace, spec_path=spec_path,
                output_dir=output_dir, cost_cap_path=cost_cap_path, cap_usd=cap_usd, thread_id=thread_id,
            )
            result = subprocess.run(
                argv, env=child_env, cwd=str(workspace), capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            _cleanup_workspace_if_requested(workspace, keep_workspace=keep_workspace)
            raise Track3RunError(
                f"track3_run_child.py did not exit within timeout_s={timeout_s!r} seconds"
                + (f" -- workspace preserved at {workspace!s} (resumable via resume_workspace=)" if keep_workspace else "")
            ) from exc
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt/SystemExit during
            # port reservation, env/argv building, or while subprocess.run() waits
            # on the child must still honor keep_workspace=False -- the cleanup
            # itself is non-suppressing (bare `raise` below), so this never
            # swallows or delays signal delivery, it only ensures cleanup runs
            # first. The workspace-preserved hint is printed to stderr rather than
            # folded into the exception message (unlike the sibling branches
            # below): this branch must re-raise the ORIGINAL exception type/
            # traceback verbatim (e.g. a real KeyboardInterrupt), not wrap it in a
            # Track3RunError, so external tooling handling that exception type
            # still sees it unchanged.
            if keep_workspace:
                print(
                    f"track3_run: workspace preserved at {workspace!s} (resumable via resume_workspace=)",
                    file=sys.stderr,
                )
            _cleanup_workspace_if_requested(workspace, keep_workspace=keep_workspace)
            raise
        if result.returncode != 0:
            _cleanup_workspace_if_requested(workspace, keep_workspace=keep_workspace)
            raise Track3RunError(
                f"track3_run_child.py exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                + (f"\nworkspace preserved at {workspace!s} (resumable via resume_workspace=)" if keep_workspace else "")
            )
        missing = [name for name in _WORKSPACE_FILES if not (workspace / name).exists()]
        if missing:
            _cleanup_workspace_if_requested(workspace, keep_workspace=keep_workspace)
            raise Track3RunError(
                f"child exited 0 but is missing workspace file(s): {missing}"
                + (f" -- workspace preserved at {workspace!s} (resumable via resume_workspace=)" if keep_workspace else "")
            )
        return workspace


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="ReviewConfig-shaped JSON spec file.")
    parser.add_argument("--cap-usd", type=float, default=30.0,
                         help="Total dollar spend cap for this run (user-authorized default: $30).")
    parser.add_argument("--thread-id", default="track3-live-run")
    parser.add_argument("--timeout-s", type=float, default=None,
                         help="Optional subprocess timeout in seconds (default: unbounded).")
    parser.add_argument(
        "--keep-workspace", action=argparse.BooleanOptionalAction, default=True,
        help="Keep the workspace directory on failure (default: True). Pass "
             "--no-keep-workspace to delete it on any failure branch, same as "
             "run_track3(keep_workspace=False).",
    )
    parser.add_argument(
        "--resume-workspace", type=Path, default=None,
        help="Resume a prior crashed run's preserved workspace directory instead of "
             "allocating a new one -- requires the SAME --thread-id used originally; a "
             "mismatch is detected and rejected up front with an error, not silently "
             "started fresh.",
    )
    args = parser.parse_args(argv)
    workspace = run_track3(
        spec_path=args.spec, cap_usd=args.cap_usd, thread_id=args.thread_id, timeout_s=args.timeout_s,
        keep_workspace=args.keep_workspace, resume_workspace=args.resume_workspace,
    )
    print(f"track3 live run workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
