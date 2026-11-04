"""Tests for `slr_pipeline.credentials` (plan §2.3, amber-quilting-osprey.md).

All offline. A fake in-memory keyring backend is injected everywhere the
real vault would otherwise be consulted -- `keyring.WinVaultKeyring` is
never touched by any test in this file."""
from __future__ import annotations

import json
import re
import stat
import subprocess
from pathlib import Path

import keyring.errors
import pytest

from slr_pipeline import credentials

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeKeyring:
    """In-memory stand-in for the `keyring` module's public surface."""

    def __init__(self, passwords=None, get_password_raises=None):
        self._passwords = dict(passwords or {})
        self._get_password_raises = get_password_raises
        self.get_password_calls = []
        self.set_password_calls = []
        self.delete_password_calls = []

    def get_password(self, service, name):
        self.get_password_calls.append((service, name))
        if self._get_password_raises is not None:
            raise self._get_password_raises
        return self._passwords.get(name)

    def set_password(self, service, name, value):
        self.set_password_calls.append((service, name, value))
        self._passwords[name] = value

    def delete_password(self, service, name):
        self.delete_password_calls.append((service, name))
        if name not in self._passwords:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._passwords[name]


# ---------------------------------------------------------------------------
# resolve_credential / credential_status source precedence
# ---------------------------------------------------------------------------

def test_env_present_keyring_never_consulted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-value")
    fake = FakeKeyring()
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) == "sk-env-value"
    assert fake.get_password_calls == []


def test_env_absent_keyring_has_value(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "sk-keyring-value"})
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) == "sk-keyring-value"


def test_env_absent_keyring_absent_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeKeyring()
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) is None


def test_credential_status_distinguishes_sources_with_env_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "same-value")
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    monkeypatch.delenv("SCOPUS_INSTITUTIONAL_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "same-value", "SCOPUS_API_KEY": "k"})
    status = credentials.credential_status(keyring_backend=fake)
    assert status["OPENAI_API_KEY"] == "env"  # same value in both -- source precedence, not value precedence
    assert status["SCOPUS_API_KEY"] == "keyring"
    assert status["SCOPUS_INSTITUTIONAL_KEY"] == "absent"
    assert status["OPENALEX_API_KEY"] == "absent"


def test_openalex_api_key_env_precedence_over_keyring(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "sk-env-value")
    fake = FakeKeyring(passwords={"OPENALEX_API_KEY": "sk-keyring-value"})
    assert credentials.resolve_credential("OPENALEX_API_KEY", keyring_backend=fake) == "sk-env-value"
    assert fake.get_password_calls == []


def test_openalex_api_key_falls_back_to_keyring_when_env_absent(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENALEX_API_KEY": "sk-keyring-value"})
    assert credentials.resolve_credential("OPENALEX_API_KEY", keyring_backend=fake) == "sk-keyring-value"


# ---------------------------------------------------------------------------
# loaded_credentials
# ---------------------------------------------------------------------------

def test_loaded_credentials_injects_only_missing_names_in_given_tuple(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "sk-vault", "SCOPUS_API_KEY": "should-not-be-requested"})
    with credentials.loaded_credentials(("OPENAI_API_KEY",), keyring_backend=fake):
        assert __import__("os").environ["OPENAI_API_KEY"] == "sk-vault"
        assert "SCOPUS_API_KEY" not in __import__("os").environ
    assert ("slr_pipeline", "SCOPUS_API_KEY") not in fake.get_password_calls


def test_loaded_credentials_injects_openalex_api_key(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENALEX_API_KEY": "sk-vault"})
    with credentials.loaded_credentials(("OPENALEX_API_KEY",), keyring_backend=fake):
        assert __import__("os").environ["OPENALEX_API_KEY"] == "sk-vault"
    assert "OPENALEX_API_KEY" not in __import__("os").environ


def test_loaded_credentials_never_overwrites_already_set_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env")
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "sk-different-vault-value"})
    with credentials.loaded_credentials(("OPENAI_API_KEY",), keyring_backend=fake):
        assert __import__("os").environ["OPENAI_API_KEY"] == "sk-real-env"
    assert fake.get_password_calls == []


def test_loaded_credentials_removes_injected_name_on_normal_exit(monkeypatch):
    import os
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "sk-vault"})
    with credentials.loaded_credentials(("OPENAI_API_KEY",), keyring_backend=fake):
        assert "OPENAI_API_KEY" in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_loaded_credentials_removes_injected_name_on_exception(monkeypatch):
    import os
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "sk-vault"})
    with pytest.raises(RuntimeError):
        with credentials.loaded_credentials(("OPENAI_API_KEY",), keyring_backend=fake):
            assert "OPENAI_API_KEY" in os.environ
            raise RuntimeError("boom")
    assert "OPENAI_API_KEY" not in os.environ


def test_loaded_credentials_never_removes_a_name_that_was_already_present(monkeypatch):
    import os
    monkeypatch.setenv("OPENAI_API_KEY", "sk-already-here")
    fake = FakeKeyring()
    with credentials.loaded_credentials(("OPENAI_API_KEY",), keyring_backend=fake):
        pass
    assert os.environ["OPENAI_API_KEY"] == "sk-already-here"


def test_loaded_credentials_restores_blank_value_not_deletes_it(monkeypatch):
    import os
    monkeypatch.setenv("OPENAI_API_KEY", "")
    fake = FakeKeyring(passwords={"OPENAI_API_KEY": "sk-vault"})
    with credentials.loaded_credentials(("OPENAI_API_KEY",), keyring_backend=fake):
        assert os.environ["OPENAI_API_KEY"] == "sk-vault"
    assert "OPENAI_API_KEY" in os.environ
    assert os.environ["OPENAI_API_KEY"] == ""


def test_loaded_credentials_rejects_unknown_name(monkeypatch):
    fake = FakeKeyring()
    with pytest.raises(ValueError):
        with credentials.loaded_credentials(("NOT_A_KNOWN_CREDENTIAL",), keyring_backend=fake):
            pass


# ---------------------------------------------------------------------------
# set_credential / delete_credential
# ---------------------------------------------------------------------------

def test_set_and_delete_credential_roundtrip_fake_backend():
    fake = FakeKeyring()
    credentials.set_credential("OPENAI_API_KEY", "sk-new-value", keyring_backend=fake)
    assert fake.set_password_calls == [("slr_pipeline", "OPENAI_API_KEY", "sk-new-value")]
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) == "sk-new-value"
    credentials.delete_credential("OPENAI_API_KEY", keyring_backend=fake)
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) is None


def test_delete_credential_tolerates_absent():
    fake = FakeKeyring()
    credentials.delete_credential("OPENAI_API_KEY", keyring_backend=fake)  # must not raise


def test_set_credential_rejects_unknown_name():
    fake = FakeKeyring()
    with pytest.raises(ValueError):
        credentials.set_credential("NOT_A_KNOWN_CREDENTIAL", "x", keyring_backend=fake)


# ---------------------------------------------------------------------------
# Credential-file fallback tier
# ---------------------------------------------------------------------------

def test_credential_file_fallback_returns_value_and_status(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    file_path = tmp_path / "creds.json"
    file_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-from-file"}), encoding="utf-8")
    monkeypatch.setenv("SLR_PIPELINE_CREDENTIAL_FILE", str(file_path))
    fake = FakeKeyring()
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) == "sk-from-file"
    assert credentials.credential_status(keyring_backend=fake)["OPENAI_API_KEY"] == "credential_file"


def test_credential_file_env_precedence_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-wins")
    file_path = tmp_path / "creds.json"
    file_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-from-file"}), encoding="utf-8")
    monkeypatch.setenv("SLR_PIPELINE_CREDENTIAL_FILE", str(file_path))
    fake = FakeKeyring()
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) == "sk-env-wins"


def test_credential_file_world_readable_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    file_path = tmp_path / "creds.json"
    file_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-from-file"}), encoding="utf-8")
    if __import__("os").name == "nt":
        subprocess.run(["icacls", str(file_path), "/grant", "Everyone:R"], capture_output=True, text=True, check=True)
    else:
        file_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    monkeypatch.setenv("SLR_PIPELINE_CREDENTIAL_FILE", str(file_path))
    fake = FakeKeyring()
    with pytest.raises(credentials.CredentialFileError):
        credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake)


def test_credential_file_inside_repository_raises_even_with_safe_permissions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    file_path = REPO_ROOT / "tmp_test_credential_file_inside_repo.json"
    try:
        file_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-from-file"}), encoding="utf-8")
        if __import__("os").name != "nt":
            file_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        monkeypatch.setenv("SLR_PIPELINE_CREDENTIAL_FILE", str(file_path))
        fake = FakeKeyring()
        with pytest.raises(credentials.CredentialFileError, match="repository"):
            credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake)
    finally:
        file_path.unlink(missing_ok=True)


def test_keyring_failure_falls_through_to_credential_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    file_path = tmp_path / "creds.json"
    file_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-from-file"}), encoding="utf-8")
    monkeypatch.setenv("SLR_PIPELINE_CREDENTIAL_FILE", str(file_path))
    fake = FakeKeyring(get_password_raises=RuntimeError("no OS credential store available"))
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) == "sk-from-file"
    assert credentials.credential_status(keyring_backend=fake)["OPENAI_API_KEY"] == "credential_file"


def test_credential_file_absent_env_var_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SLR_PIPELINE_CREDENTIAL_FILE", raising=False)
    fake = FakeKeyring()
    assert credentials.resolve_credential("OPENAI_API_KEY", keyring_backend=fake) is None


# ---------------------------------------------------------------------------
# Isolation: config.py / scopus.py never consult keyring
# ---------------------------------------------------------------------------

def test_resolve_openai_api_key_never_consults_keyring(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import keyring as real_keyring
    monkeypatch.setattr(real_keyring, "get_password", lambda *a, **k: "sk-vault-value")
    from slr_pipeline.config import resolve_openai_api_key
    assert resolve_openai_api_key() is None


def test_scopus_backend_never_consults_keyring(monkeypatch):
    import slr_pipeline.search_backends.scopus as scopus_module
    from slr_pipeline.search_backends.base import SearchBackendRequest

    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    import keyring as real_keyring
    monkeypatch.setattr(real_keyring, "get_password", lambda *a, **k: "sk-vault-value")

    class _NeverCalledHttpClient:
        def __call__(self):
            return self

        def get(self, *a, **kw):
            raise AssertionError("HttpClient.get must not be called")

    monkeypatch.setattr(scopus_module, "HttpClient", _NeverCalledHttpClient())
    request = SearchBackendRequest(
        query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None,
    )
    result = scopus_module.ScopusBackend().search(request)
    assert result.retrieval_status == "failed"
    assert result.failure_detail == "SCOPUS_API_KEY not set"


def test_openalex_backend_never_consults_keyring(monkeypatch):
    import slr_pipeline.search_backends.openalex as openalex_module
    from slr_pipeline.search_backends.base import SearchBackendRequest

    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    import keyring as real_keyring
    monkeypatch.setattr(real_keyring, "get_password", lambda *a, **k: "sk-vault-value")

    class _FakeResponse:
        def json(self):
            return {"meta": {"count": 0, "next_cursor": None}, "results": []}

    class _FakeHttpClient:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            return self

        def get(self, url, *, params=None, headers=None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            return _FakeResponse()

    client = _FakeHttpClient()
    monkeypatch.setattr(openalex_module, "HttpClient", client)
    request = SearchBackendRequest(
        query="cats", max_results=5, date_from=None, date_to=None, language=None, mailto=None,
    )
    result = openalex_module.OpenAlexBackend().search(request)
    # api_key is optional for OpenAlex (unlike Scopus's hard-required key) --
    # an absent OPENALEX_API_KEY must never be silently rescued from the OS
    # keyring, it must simply proceed without an api_key param.
    assert result.retrieval_status == "exhaustive"
    assert "api_key" not in client.calls[0]["params"]
    assert client.calls[0]["headers"] is None


def test_config_module_does_not_import_credentials():
    text = (REPO_ROOT / "slr_pipeline" / "config.py").read_text(encoding="utf-8")
    assert "slr_pipeline.credentials" not in text
    assert re.search(r"^\s*(from|import)\s+.*\bcredentials\b", text, re.MULTILINE) is None


def test_scopus_module_does_not_import_credentials():
    text = (REPO_ROOT / "slr_pipeline" / "search_backends" / "scopus.py").read_text(encoding="utf-8")
    assert "slr_pipeline.credentials" not in text
    assert re.search(r"^\s*(from|import)\s+.*\bcredentials\b", text, re.MULTILINE) is None


def test_openalex_module_does_not_import_credentials():
    text = (REPO_ROOT / "slr_pipeline" / "search_backends" / "openalex.py").read_text(encoding="utf-8")
    assert "slr_pipeline.credentials" not in text
    assert re.search(r"^\s*(from|import)\s+.*\bcredentials\b", text, re.MULTILINE) is None


# ---------------------------------------------------------------------------
# No credential ever formatted into a log/print/exception message
# ---------------------------------------------------------------------------

def test_no_function_formats_a_value_into_logging_print_or_exception():
    text = (REPO_ROOT / "slr_pipeline" / "credentials.py").read_text(encoding="utf-8")
    assert "logging" not in text
    for match in re.finditer(r"print\((.*)\)", text):
        assert "value" not in match.group(1)
