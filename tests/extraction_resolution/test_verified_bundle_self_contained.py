"""Static self-containment check for slr_pipeline/extraction/verified_bundle.py.

Per amber-quilting-osprey.md sec. 3.0, this module owns bundle verification
and `nodes` imports it -- never the reverse. An import from
`slr_pipeline.nodes` here would put the dependency back in the wrong
direction.
"""
from __future__ import annotations

import ast
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "slr_pipeline" / "extraction" / "verified_bundle.py"


def _imported_module_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_verified_bundle_module_exists():
    assert MODULE_PATH.is_file(), MODULE_PATH


def test_verified_bundle_imports_nothing_from_nodes():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    imported = _imported_module_names(tree)
    offenders = [name for name in imported if name == "slr_pipeline.nodes" or name.startswith("slr_pipeline.nodes.")]
    assert offenders == [], f"verified_bundle.py must not import from slr_pipeline.nodes, found: {offenders}"


def test_verified_bundle_exports_expected_names():
    from slr_pipeline.extraction import verified_bundle

    assert hasattr(verified_bundle, "REVIEW_STATUSES")
    assert hasattr(verified_bundle, "load_schema")
    assert hasattr(verified_bundle, "bundle_review_fields")
    assert hasattr(verified_bundle, "load_and_verify_bundle")
    assert verified_bundle.REVIEW_STATUSES == {"unclear", "conflicting", "extraction_failed"}
