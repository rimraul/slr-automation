"""Tests for scripts/extract_listings.py and the listing manifest.

This file deliberately carries NO listing markers of its own.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import textwrap
from pathlib import Path

import pytest

from scripts.extract_listings import extract_all, extract_listing, list_listings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "examples" / "chapter" / "listings_manifest.json"
_MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
_SOLUTION_FILES = sorted((_REPO_ROOT / "examples" / "chapter").glob("s*/solution.py"))

_LISTING_RE = re.compile(r"#\s*---\s*listing:([A-Za-z0-9_-]+)\s+(start|end)\s*---")


# --- 1. extractor mechanics -------------------------------------

def test_extract_two_flat_regions(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text(
        "# --- listing:a start ---\n"
        "def a():\n"
        "    return 1\n"
        "# --- listing:a end ---\n"
        "\n"
        "# --- listing:b start ---\n"
        "def b():\n"
        "    return 2\n"
        "# --- listing:b end ---\n",
        encoding="utf-8",
    )
    assert list_listings(src) == ["a", "b"]
    assert extract_listing(src, "a") == "def a():\n    return 1\n"
    assert extract_listing(src, "b") == "def b():\n    return 2\n"


def test_dedent_is_applied(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(
        "    # --- listing:x start ---\n"
        "    def x():\n"
        "        return 1\n"
        "    # --- listing:x end ---\n",
        encoding="utf-8",
    )
    assert extract_listing(src, "x") == "def x():\n    return 1\n"


def test_unknown_id_raises_keyerror(tmp_path):
    src = tmp_path / "s.py"
    src.write_text("# --- listing:a start ---\nx = 1\n# --- listing:a end ---\n", encoding="utf-8")
    with pytest.raises(KeyError):
        extract_listing(src, "nope")


def test_unterminated_marker_raises_valueerror(tmp_path):
    src = tmp_path / "s.py"
    src.write_text("# --- listing:a start ---\nx = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="never closed"):
        list_listings(src)


def test_nested_marker_raises_valueerror(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(
        "# --- listing:a start ---\n# --- listing:b start ---\nx=1\n"
        "# --- listing:b end ---\n# --- listing:a end ---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nested"):
        list_listings(src)


def test_duplicate_id_in_file_raises_valueerror(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(
        "# --- listing:a start ---\nx=1\n# --- listing:a end ---\n"
        "# --- listing:a start ---\ny=2\n# --- listing:a end ---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        list_listings(src)


# --- 2/4. manifest completeness and manifest <-> reality --------

def test_manifest_ids_extract_from_their_source():
    extracted = extract_all(_MANIFEST_PATH)
    assert len(extracted) == len(_MANIFEST["listings"])
    for text in extracted.values():
        assert text.strip()


def test_manifest_id_set_equals_the_marker_id_set_in_the_tree():
    tree_ids = set()
    for path in _SOLUTION_FILES:
        for match in _LISTING_RE.finditer(path.read_text(encoding="utf-8")):
            if match.group(2) == "start":
                tree_ids.add(match.group(1))
    manifest_ids = {row["id"] for row in _MANIFEST["listings"]}
    assert manifest_ids == tree_ids


# --- 3. global uniqueness -------------------------------------

def test_no_listing_id_appears_in_two_files():
    seen: dict[str, Path] = {}
    for path in _SOLUTION_FILES:
        for match in _LISTING_RE.finditer(path.read_text(encoding="utf-8")):
            if match.group(2) != "start":
                continue
            lid = match.group(1)
            assert lid not in seen, f"{lid} in both {seen.get(lid)} and {path}"
            seen[lid] = path


# --- 5. whole-symbol parity ----------------------------------

def _module_for(source: str):
    package = Path(source).parent.name
    return importlib.import_module(f"examples.chapter.{package}.solution")


@pytest.mark.parametrize("row", _MANIFEST["listings"], ids=lambda r: r["id"])
def test_whole_symbol_parity(row):
    if row.get("fragment"):
        pytest.skip("fragment listing -- whole-symbol parity does not apply")
    module = _module_for(row["source"])
    symbol = getattr(module, row["symbol"])
    extracted = extract_listing(_REPO_ROOT / row["source"], row["id"])
    from_source = textwrap.dedent(inspect.getsource(symbol)).rstrip("\n") + "\n"
    assert extracted == from_source


# --- 6. node-id binding via AST -----------------------------

def _resolve_symbol_names(test_tree: ast.Module, package: str, symbol: str) -> set[str]:
    """Names by which `symbol` (from examples.chapter.<package>.solution) can be
    called in the test module: a bare name (from ... import symbol) or
    <alias>.symbol (import solution as alias)."""
    names: set[str] = set()
    want_module = f"examples.chapter.{package}.solution"
    want_pkg = f"examples.chapter.{package}"
    for node in ast.walk(test_tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == want_module:
                for alias in node.names:
                    if alias.name == symbol:
                        names.add(alias.asname or alias.name)
            if node.module == want_pkg:
                for alias in node.names:
                    if alias.name == "solution":
                        names.add((alias.asname or "solution") + "." + symbol)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == want_module:
                    names.add((alias.asname or alias.name) + "." + symbol)
    return names


def _function_calls_symbol(func: ast.FunctionDef, call_names: set[str]) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id in call_names:
            return True
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
            if f"{callee.value.id}.{callee.attr}" in call_names:
                return True
    return False


@pytest.mark.parametrize("row", _MANIFEST["listings"], ids=lambda r: r["id"])
def test_node_id_binding_is_a_real_call(row):
    package = Path(row["source"]).parent.name
    for node_id in row["test_node_ids"]:
        file_part, _, test_name = node_id.partition("::")
        test_path = _REPO_ROOT / file_part
        assert test_path.is_file(), node_id
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == test_name]
        assert funcs, f"{node_id}: test function not found"
        call_names = _resolve_symbol_names(tree, package, row["symbol"])
        assert call_names, f"{node_id}: could not resolve an import for {row['symbol']}"
        assert any(_function_calls_symbol(f, call_names) for f in funcs), (
            f"{node_id}: no ast.Call to {row['symbol']} in the bound test"
        )


# --- 7. transformation contract ----------------------------

def test_only_common_whitespace_is_stripped(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(
        "        # --- listing:z start ---\n"
        "        def z():\n"
        "            x = {'a': 1}\n"
        "            return x\n"
        "        # --- listing:z end ---\n",
        encoding="utf-8",
    )
    assert extract_listing(src, "z") == "def z():\n    x = {'a': 1}\n    return x\n"
