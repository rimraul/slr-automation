"""Track 2 plan Step 13 [C12]: the reference-run **parent** driver.

Allocates a mutable workspace (plan §12.4), builds the child's fresh,
minimal environment (§12.3), and spawns `reference_run_child.py` as a
**subprocess** (`python -m slr_pipeline.reference_run_child ...`) -- never
in-process, so the child's own socket-guard/import-order discipline (see
that module's docstring) is real process isolation, not merely a
convention this process could accidentally violate.

Scope note (plan §13 step 12's own bullet): this driver's job, at this
step, ends once the child exits 0 and the four workspace files it must
have written (`terminal_state.json`, `terminal_graph_snapshot.json`,
`call_log.jsonl`, `violations.jsonl`) are confirmed present. It does
**not** run the §12.1 execution-subtree clean-check, does **not** write
`payload_manifest.json`/`acceptance_report.json` (§12.2), and does **not**
call `finalize_bundle()` -- all three need "the manifests + publication
machinery of step 15" (round-13 finding 7) and are out of scope here.
"""
from __future__ import annotations

import datetime
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

from slr_pipeline.reference_pdf_placement import REFERENCE_EXTRACTION_WORKSPACE_PARTS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_CORPUS_DIR = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "corpus"
FIXTURES_SPEC_DIR = REPO_ROOT / "tests" / "reference_run" / "fixtures" / "spec"
DEFAULT_SPEC_PATH = FIXTURES_SPEC_DIR / "reference_spec.json"
DEFAULT_CASSETTE_PATH = FIXTURES_CORPUS_DIR / "replay_cassette.json"
DEFAULT_CORPUS_ORACLE_PATH = FIXTURES_CORPUS_DIR / "CORPUS_ORACLE.json"
DEFAULT_PDF_SOURCE_DIR = FIXTURES_CORPUS_DIR / "pdfs"
DEFAULT_FIXTURE_CORPUS_ROOT = FIXTURES_CORPUS_DIR

# The FIXED extraction-output-dir every child MUST use, on this checkout, to reproduce
# the same pdf_path strings baked into the frozen cassette (see reference_pdf_placement's
# own comment on REFERENCE_EXTRACTION_WORKSPACE_PARTS -- the same constant
# author_reference_cassette.py's FIXED_WORKSPACE resolves against).
DEFAULT_EXTRACTION_OUTPUT_DIR = REPO_ROOT.joinpath(*REFERENCE_EXTRACTION_WORKSPACE_PARTS)

_WORKSPACE_FILES = ("terminal_state.json", "terminal_graph_snapshot.json", "call_log.jsonl", "violations.jsonl")

# Minimal, non-secret Windows runtime allowlist (plan §12.3, verbatim list).
_WINDOWS_RUNTIME_ALLOWLIST = (
    "SystemRoot", "windir", "ComSpec", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)


def _reserve_loopback_port() -> int:
    """Binds an ephemeral port on `127.0.0.1`, reads it back, closes the
    socket. This process (the parent) picks the port and commits to it via
    `OPENAI_BASE_URL` in the child's env, before the child even starts --
    the small bind/close-then-rebind TOCTOU race is accepted here, not in
    the sandboxed child (see `reference_run_child.py`'s own docstring)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def allocate_workspace(base_tmp_dir: Path) -> Path:
    """Plan §12.4: `<tmp>/reference_run_<ts>_<uuid>/`."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = base_tmp_dir / f"reference_run_{ts}_{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def build_child_env(*, workspace: Path, replay_port: int, python_executable: str,
                     base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Plan §12.3: a fresh dict, not `os.environ.copy()` -- every value
    either a fixed constant, derived from `workspace`, or copied from
    `base_env` (defaults to the real `os.environ`) for the small,
    explicitly-named Windows runtime allowlist only. No inherited
    provider/search credentials or proxies."""
    base_env = os.environ if base_env is None else base_env
    caches_dir = workspace / "caches"
    # Round-13 self-review finding 2: TEMP/TMP/HOME/etc. pointed at these
    # subdirectories but nothing created them -- on Windows an absent TEMP/TMP
    # is silently skipped by `tempfile.gettempdir()`'s candidate search,
    # falling back to `os.getcwd()` (the workspace root, since the child is
    # spawned with `cwd=str(workspace)`), defeating the isolation this env is
    # for. Created eagerly, here, so `build_child_env`'s return value is
    # immediately usable without a separate caller-side setup step.
    for sub in ("home", "huggingface", "torch", "matplotlib"):
        (caches_dir / sub).mkdir(parents=True, exist_ok=True)
    env: Dict[str, str] = {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{replay_port}/v1",
        "OPENAI_API_KEY": "sk-replay-not-a-real-key",
        "DOCLING_CACHE_DIR": str((REPO_ROOT / "assets" / "models" / "track2-docling-cache").resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
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
    for name in _WINDOWS_RUNTIME_ALLOWLIST:
        if name in base_env:
            env[name] = base_env[name]
    venv_scripts = str(Path(python_executable).resolve().parent)
    system32 = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32")
    env["PATH"] = os.pathsep.join([venv_scripts, system32])
    return env


def build_child_argv(*, python_executable: str, workspace: Path, extraction_output_dir: Path,
                      spec_path: Path, cassette_path: Path, corpus_oracle_path: Path,
                      pdf_source_dir: Path, fixture_corpus_root: Path, thread_id: str) -> list:
    return [
        python_executable, "-m", "slr_pipeline.reference_run_child",
        "--workspace", str(workspace),
        "--extraction-output-dir", str(extraction_output_dir),
        "--spec", str(spec_path),
        "--cassette", str(cassette_path),
        "--corpus-oracle", str(corpus_oracle_path),
        "--pdf-source-dir", str(pdf_source_dir),
        "--fixture-corpus-root", str(fixture_corpus_root),
        "--thread-id", thread_id,
    ]


class ReferenceRunError(RuntimeError):
    """Raised when the child subprocess exits non-zero, or exits zero but
    is missing one of the four workspace files it must have written."""


def run_reference(*, base_tmp_dir: Optional[Path] = None, extraction_output_dir: Optional[Path] = None,
                   spec_path: Path = DEFAULT_SPEC_PATH, cassette_path: Path = DEFAULT_CASSETTE_PATH,
                   corpus_oracle_path: Path = DEFAULT_CORPUS_ORACLE_PATH,
                   pdf_source_dir: Path = DEFAULT_PDF_SOURCE_DIR,
                   fixture_corpus_root: Path = DEFAULT_FIXTURE_CORPUS_ROOT,
                   thread_id: str = "track2-reference-run",
                   python_executable: str = sys.executable, keep_workspace: bool = True) -> Path:
    """Runs the child subprocess to completion and returns the workspace
    path. `keep_workspace` defaults `True` here (unlike plan §12.4's
    default for the *full* driver, which deletes on success after publish
    -- step 15's job): at this step there is no publish/re-verify stage
    yet, so deleting an unpublished, otherwise-inspectable workspace would
    just destroy the only artifacts this step produces."""
    extraction_output_dir = extraction_output_dir or DEFAULT_EXTRACTION_OUTPUT_DIR
    base_tmp_dir = base_tmp_dir or Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")
    workspace = allocate_workspace(base_tmp_dir)

    replay_port = _reserve_loopback_port()
    child_env = build_child_env(workspace=workspace, replay_port=replay_port, python_executable=python_executable)
    argv = build_child_argv(
        python_executable=python_executable, workspace=workspace, extraction_output_dir=extraction_output_dir,
        spec_path=spec_path, cassette_path=cassette_path, corpus_oracle_path=corpus_oracle_path,
        pdf_source_dir=pdf_source_dir, fixture_corpus_root=fixture_corpus_root, thread_id=thread_id,
    )
    result = subprocess.run(argv, env=child_env, cwd=str(workspace), capture_output=True, text=True)
    if result.returncode != 0:
        if not keep_workspace:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)
        raise ReferenceRunError(
            f"reference_run_child.py exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    missing = [name for name in _WORKSPACE_FILES if not (workspace / name).exists()]
    if missing:
        raise ReferenceRunError(f"child exited 0 but is missing workspace file(s): {missing}")
    return workspace


def main() -> int:
    """No CLI-argument passthrough at this step -- unlike the child's own
    `_parse_args`, this entrypoint always runs `run_reference()` with its
    all-default spec/cassette/thread-id (round-13 self-review finding 4:
    a prior `argv` parameter here was accepted but never consulted)."""
    workspace = run_reference()
    print(f"reference run workspace: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

