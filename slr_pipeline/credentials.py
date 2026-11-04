"""Credential policy: env > OS keyring > credential-file > absent.

Only `loaded_credentials()` is ever allowed to mutate `os.environ`, and only
for the duration of its `with` block. Low-level resolvers
(`config.py::resolve_openai_api_key()`, `search_backends/scopus.py`'s reads)
stay environment-only and unchanged -- they must never reach into the OS
credential store, or an offline test that clears an env var to prove a
failure path could be silently rescued by a real stored secret, and
`--estimate-only` (which does no network I/O) could end up touching the
credential store for no reason.

Callers, exhaustively: `track3_run.py`'s live-run entry point,
`track3_continuation_run.py::main()`, `update_driver.run_update_wave_search()`
(Scopus and OpenAlex credentials), and `update_wave_run.main()`'s `screen`
subcommand (OpenAI credentials) -- all entered only after cost-cap /
authorization checks have already passed. Nothing else may call
`loaded_credentials()`.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Iterable, Literal, Optional

import keyring
import keyring.errors

_CREDENTIAL_NAMES = (
    "OPENAI_API_KEY",
    "SCOPUS_API_KEY",
    "SCOPUS_INSTITUTIONAL_KEY",
    "OPENALEX_API_KEY",
)
_SERVICE_NAME = "slr_pipeline"

# Distinct from "" -- a name present but blank is a real, intentional prior
# state (e.g. a launcher's own child-environment allowlist) and must be
# restored to that exact blank value on exit, never deleted as if absent.
_MISSING = object()

CredentialSource = Literal["env", "keyring", "credential_file", "absent"]


class CredentialFileError(RuntimeError):
    """Raised when `SLR_PIPELINE_CREDENTIAL_FILE` points at a misconfigured file."""


def _assert_known_name(name: str) -> None:
    if name not in _CREDENTIAL_NAMES:
        raise ValueError(f"unknown credential name: {name!r}")


def _repository_root() -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return Path(result.stdout.strip()).resolve()


def _assert_not_inside_repository(path: Path) -> None:
    repo_root = _repository_root()
    if repo_root is None:
        return
    try:
        path.relative_to(repo_root)
    except ValueError:
        return  # not inside the repository -- fine
    raise CredentialFileError(f"credential file must not live inside the repository: {path}")


def _windows_file_grants_broad_access(path: Path) -> bool:
    """Best-effort, advisory only -- Windows ACL enforcement from Python is
    inherently weaker than POSIX mode bits. Scans `icacls` output for
    well-known broad-access principals."""
    try:
        result = subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True)
    except Exception:
        return False
    broad_principals = ("Everyone:", "BUILTIN\\Users:", "NT AUTHORITY\\Authenticated Users:")
    return any(principal in result.stdout for principal in broad_principals)


def _file_has_unsafe_permissions(path: Path) -> bool:
    if os.name == "nt":
        return _windows_file_grants_broad_access(path)
    mode = path.stat().st_mode
    return bool(mode & (stat.S_IRGRP | stat.S_IROTH))


def _assert_file_not_broadly_readable(path: Path) -> None:
    if _file_has_unsafe_permissions(path):
        raise CredentialFileError(f"credential file has unsafe permissions (world/group-readable): {path}")


def _assert_credential_file_safe(path: Path) -> None:
    _assert_not_inside_repository(path)
    _assert_file_not_broadly_readable(path)


def _read_credential_file(name: str) -> Optional[str]:
    file_path_str = os.environ.get("SLR_PIPELINE_CREDENTIAL_FILE")
    if not file_path_str:
        return None
    file_path = Path(file_path_str).resolve()
    if not file_path.exists():
        return None
    _assert_credential_file_safe(file_path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    value = data.get(name)
    return value if value else None


def _resolve_with_source(name: str, *, keyring_backend) -> tuple[Optional[str], CredentialSource]:
    """The only place that decides env-vs-keyring-vs-credential_file-vs-absent.

    An unavailable/failing keyring backend is treated identically to "no
    value for this name" -- resolution falls through to the credential-file
    tier rather than aborting before it. The credential-file tier itself is
    NOT wrapped this way: a misconfigured file (inside the repo, or
    world/group-readable) raises `CredentialFileError` rather than being
    silently skipped.
    """
    env_value = os.environ.get(name)
    if env_value:
        return env_value, "env"
    try:
        keyring_value = keyring_backend.get_password(_SERVICE_NAME, name)
    except Exception:
        keyring_value = None
    if keyring_value:
        return keyring_value, "keyring"
    file_value = _read_credential_file(name)
    if file_value:
        return file_value, "credential_file"
    return None, "absent"


def resolve_credential(name: str, *, keyring_backend=keyring) -> Optional[str]:
    """`keyring_backend` is an injection point so tests never touch the real
    OS vault -- default is the real `keyring` module, tests pass a fake."""
    value, _source = _resolve_with_source(name, keyring_backend=keyring_backend)
    return value


def credential_status(names: Iterable[str] = _CREDENTIAL_NAMES, *, keyring_backend=keyring) -> dict:
    """Returns {name: source label}, never the value -- purely read-only,
    never mutates os.environ."""
    names = tuple(names)
    for name in names:
        _assert_known_name(name)
    return {name: _resolve_with_source(name, keyring_backend=keyring_backend)[1] for name in names}


@contextlib.contextmanager
def loaded_credentials(names: Iterable[str], *, keyring_backend=keyring):
    """Populates os.environ, for JUST the given names, for JUST the
    duration of the with block. Only injects a value for a name that is
    genuinely absent or blank; a name already non-blank before entry is
    never touched. On exit (normal return or exception), every name this
    call itself injected is restored to its exact prior state -- deleted if
    it was absent, restored to blank if it was blank."""
    names = tuple(names)
    for name in names:
        _assert_known_name(name)
    injected = []
    priors = {}
    try:
        for name in names:
            prior = os.environ.get(name, _MISSING)
            if prior is _MISSING or prior == "":
                priors[name] = prior
                value, _source = _resolve_with_source(name, keyring_backend=keyring_backend)
                if value:
                    os.environ[name] = value
                    injected.append(name)
        yield
    finally:
        for name in injected:
            prior = priors[name]
            if prior is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior


def set_credential(name: str, value: str, *, keyring_backend=keyring) -> None:
    _assert_known_name(name)
    keyring_backend.set_password(_SERVICE_NAME, name, value)


def delete_credential(name: str, *, keyring_backend=keyring) -> None:
    _assert_known_name(name)
    try:
        keyring_backend.delete_password(_SERVICE_NAME, name)
    except keyring.errors.PasswordDeleteError:
        pass  # tolerate absent


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m slr_pipeline.credentials")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("name", choices=_CREDENTIAL_NAMES)

    subparsers.add_parser("status")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("name", choices=_CREDENTIAL_NAMES)

    args = parser.parse_args()
    if args.command == "set":
        value = getpass.getpass(f"Value for {args.name}: ")
        set_credential(args.name, value)
    elif args.command == "status":
        for name, source in credential_status().items():
            print(f"{name}: {source}")
    elif args.command == "delete":
        delete_credential(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
