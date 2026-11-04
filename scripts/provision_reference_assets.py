"""Provision the Docling reference-run asset cache (immutable, hashed).

track2_pipeline_readiness_plan_2026-08-30.md §5.3/§13 step 19: this
script's actual invocation (real model downloads, real manifest/licence
authoring) is **step 19 only**, once `ft_screening.docling_artifacts` is
installed and the redistribution authorization
(`assets/models/DOCLING_REDISTRIBUTION_NOTE.md`) exists -- mirroring
`build_slr_pipeline_wheel.py`'s own step-19-only `--write-lock`/
`--write-contents`. The script itself is authored **now, at step 5**
(§13 step 5's own deliverable list), since `assets/models/DOCLING_ASSETS.
lock`'s mere *presence* already gates the CI `probe` job's three-lock
readiness check (§5.5) from this step onward -- the file's absence, not
this script's absence, is what keeps `acceptance` skipped.

Usage (step 19):
    python scripts/provision_reference_assets.py --write-lock
    python scripts/provision_reference_assets.py --write-archive
    python scripts/provision_reference_assets.py --check [--skip-archive-hash]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = REPO_ROOT / "assets" / "models" / "track2-docling-cache"
MODELS_DIR = CACHE_ROOT / "models"
ASSETS_LOCK_PATH = REPO_ROOT / "assets" / "models" / "DOCLING_ASSETS.lock"
UPSTREAM_LICENSES_DIR = REPO_ROOT / "assets" / "models" / "upstream_licenses"
MODEL_MANIFEST_PATH = REPO_ROOT / "assets" / "models" / "DOCLING_MODEL_MANIFEST.md"
ARCHIVE_PATH = REPO_ROOT / "docling-assets-v1.tar"
ALLOWED_UPSTREAM_LICENCES = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause", "CDLA-Permissive-2.0"})

# Round-19 finding: `--write-lock` previously required
# `DOCLING_MODEL_MANIFEST.md` to already exist and never touched
# `UPSTREAM_LICENSES_DIR`/`ALLOWED_UPSTREAM_LICENCES` at all, contradicting
# §5.3 ("--write-lock copies each upstream repo's actual LICENSE... into
# this tree... and writes DOCLING_MODEL_MANIFEST.md"). The four Hugging
# Face repos' commits come from `docling_artifacts.PINNED_REVISIONS`
# itself (imported, never retyped, so a revision bump there can't drift
# out of sync here); RapidOCR is not an HF repo (Docling fetches it via
# `RapidOcrModel.download_models` from versioned ModelScope release URLs)
# so it is listed here explicitly. Licence + URL values below were checked
# directly against each upstream repository's own licence field on
# 2026-09-02 -- an SPDX label alone does not satisfy a redistribution
# condition (round-11 finding 10).
#
# First real --write-lock run (2026-09-06) found that no downloaded asset
# directory actually contains a standalone LICENSE/LICENSE.txt/LICENSE.md/
# NOTICE file: confirmed via `HfApi.list_repo_files` that none of the four
# HF repos ship one, at their pinned revision or at `main` -- each only
# declares its licence via the `license:` field in README.md's YAML
# frontmatter (HF's own platform treats this as the authoritative,
# machine-readable licence assertion). RapidOCR is fetched via Docling's
# own downloader (individual weight files from a release URL), so nothing
# licence-related is ever part of that download either -- but RapidOCR's
# actual GitHub repository does ship a real LICENSE file at the matching
# tag, confirmed reachable at `licence_fetch_url` below. So the two asset
# families use two different, and equally non-fabricated, licence
# sources: `_readme_licence_bundle()` bundles the whole HF README.md
# (after checking its declared licence matches the table below) and
# `_fetch_rapidocr_licence()` fetches RapidOCR's real upstream LICENSE
# file directly. Neither ever invents licence text.
_EXTRA_UPSTREAM_ASSETS: tuple[dict[str, str], ...] = (
    {
        "repo_id": "rapidocr",
        "revision": "v3.4.0",
        "local_dir": "RapidOcr",
        "licence_fetch_url": "https://raw.githubusercontent.com/RapidAI/RapidOCR/v3.4.0/LICENSE",
    },
)
_UPSTREAM_LICENCE_INFO: dict[str, dict[str, str]] = {
    "docling-project/docling-layout-heron": {
        "url": "https://huggingface.co/docling-project/docling-layout-heron",
        "licence": "Apache-2.0",
    },
    "docling-project/CodeFormulaV2": {
        "url": "https://huggingface.co/docling-project/CodeFormulaV2",
        "licence": "CDLA-Permissive-2.0",
    },
    "docling-project/docling-models": {
        "url": "https://huggingface.co/docling-project/docling-models",
        "licence": "CDLA-Permissive-2.0",
    },
    "docling-project/DocumentFigureClassifier": {
        "url": "https://huggingface.co/docling-project/DocumentFigureClassifier",
        "licence": "MIT",
    },
    "rapidocr": {
        "url": "https://github.com/RapidAI/RapidOCR",
        "licence": "Apache-2.0",
    },
}
_LICENCE_FIELD_RE = re.compile(r"^license:\s*(\S+)", re.MULTILINE | re.IGNORECASE)


def _local_dir_name(repo_id: str) -> str:
    """Docling's own local-directory naming convention for a pinned HF
    repo id -- verified against `ft_screening.docling_artifacts`'s actual
    (private) `_LOCAL_DIR_NAMES` values: `"owner/repo"` -> `"owner--repo"`."""
    return repo_id.replace("/", "--")


def _hash_tree(root: Path) -> dict[str, dict]:
    """Per-file `(relpath, sha256, size)` for every regular file under
    `root`, sorted by relpath -- the same shape `DOCLING_ASSETS.lock`
    records."""
    entries: dict[str, dict] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relpath = path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        entries[relpath] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
    return entries


def _import_docling_artifacts():
    """Deferred import -- `ft_screening` is an optional dependency of this
    script (only needed at step 19, when it is actually installed and
    invoked); importing it eagerly at module scope would make every other
    caller of this file (e.g. the step-5 `probe`-mirroring checks in
    `reference_preflight.py`) depend on FT_SCREENING for no reason."""
    from ft_screening import docling_artifacts
    return docling_artifacts


def _readme_licence_bundle(asset_dir: Path, expected_licence: str) -> bytes:
    """Hugging Face model repos declare their licence only via the
    `license:` field in README.md's YAML frontmatter -- confirmed via
    `HfApi.list_repo_files` against all four pinned repos, at both their
    pinned revision and `main`, that none ships a standalone LICENSE/
    LICENSE.txt/LICENSE.md/NOTICE file. Bundles the whole README.md
    verbatim (not just the frontmatter fragment, to avoid any string-
    slicing misjudgement) -- but only after confirming it actually
    contains a `license:` field matching `expected_licence`; fails closed
    if the field is missing or has drifted from `_UPSTREAM_LICENCE_INFO`'s
    hardcoded label."""
    readme_path = asset_dir / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"no README.md found in downloaded asset directory {asset_dir}")
    text = readme_path.read_text(encoding="utf-8")
    match = _LICENCE_FIELD_RE.search(text)
    if match is None:
        raise ValueError(f"{readme_path} has no `license:` field in its frontmatter")
    declared = match.group(1).strip()
    if declared.lower() != expected_licence.lower():
        raise ValueError(
            f"{readme_path} declares license {declared!r}, expected {expected_licence!r} "
            "(per _UPSTREAM_LICENCE_INFO) -- update the hardcoded table if this is a genuine upstream change"
        )
    return readme_path.read_bytes()


def _fetch_rapidocr_licence(url: str) -> bytes:
    """RapidOCR's model weights are fetched via Docling's own downloader
    (individual weight files from a versioned release URL, never a repo
    tree), so no licence file is ever part of that download -- unlike the
    four HF repos, RapidOCR's actual GitHub repository does ship a real
    LICENSE file at the matching tag, so this fetches it directly instead
    of falling back to any local frontmatter."""
    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"fetched licence from {url} was empty")
    return response.content


def _write_model_manifest_and_licences(docling_artifacts) -> None:
    """§5.3: for every upstream asset `bootstrap()` fetched -- the four
    pinned Hugging Face repos (`docling_artifacts.PINNED_REVISIONS`) plus
    `_EXTRA_UPSTREAM_ASSETS` (RapidOCR) -- obtain its real licence source
    (`_readme_licence_bundle()` for the HF repos, `_fetch_rapidocr_licence()`
    for RapidOCR; see the comment above `_EXTRA_UPSTREAM_ASSETS`), bundle
    it verbatim into `UPSTREAM_LICENSES_DIR/<local_dir>/`, and write
    `DOCLING_MODEL_MANIFEST.md` recording id/revision/licence/url/bundled-
    licence-path for each. Fails closed: raises if an asset has no
    registered licence info, if that licence is not in
    `ALLOWED_UPSTREAM_LICENCES`, or if its licence source is missing or
    disagrees with the registered label -- an SPDX label alone is never
    sufficient."""
    assets = [
        {"repo_id": repo_id, "revision": revision, "local_dir": _local_dir_name(repo_id)}
        for repo_id, revision in docling_artifacts.PINNED_REVISIONS.items()
    ]
    assets.extend(dict(asset) for asset in _EXTRA_UPSTREAM_ASSETS)

    UPSTREAM_LICENSES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = []
    for asset in sorted(assets, key=lambda a: a["local_dir"]):
        info = _UPSTREAM_LICENCE_INFO.get(asset["repo_id"])
        if info is None:
            raise ValueError(
                f"no licence info registered for upstream asset {asset['repo_id']!r} -- "
                "add an entry to _UPSTREAM_LICENCE_INFO before --write-lock can proceed"
            )
        licence = info["licence"]
        if licence not in ALLOWED_UPSTREAM_LICENCES:
            raise ValueError(
                f"upstream asset {asset['repo_id']!r} licence {licence!r} is not in "
                f"ALLOWED_UPSTREAM_LICENCES {sorted(ALLOWED_UPSTREAM_LICENCES)}"
            )

        asset_dir = MODELS_DIR / asset["local_dir"]
        if asset["repo_id"] == "rapidocr":
            licence_bytes = _fetch_rapidocr_licence(asset["licence_fetch_url"])
            licence_filename = "LICENSE"
        else:
            licence_bytes = _readme_licence_bundle(asset_dir, licence)
            licence_filename = "README.md"

        bundled_dir = UPSTREAM_LICENSES_DIR / asset["local_dir"]
        bundled_dir.mkdir(parents=True, exist_ok=True)
        bundled_path = bundled_dir / licence_filename
        bundled_path.write_bytes(licence_bytes)
        bundled_relpath = bundled_path.relative_to(REPO_ROOT / "assets" / "models").as_posix()

        manifest_rows.append(f"| {asset['repo_id']} | {asset['revision']} | {licence} | {info['url']} | `{bundled_relpath}` |")

    manifest_lines = [
        "# Docling upstream model manifest",
        "",
        "Generated by `scripts/provision_reference_assets.py --write-lock`. One row",
        "per upstream asset `ft_screening.docling_artifacts.bootstrap()` fetches.",
        "",
        "| Repository | Revision | Licence | URL | Bundled licence file |",
        "|---|---|---|---|---|",
        *manifest_rows,
        "",
    ]
    MODEL_MANIFEST_PATH.write_text("\n".join(manifest_lines), encoding="utf-8")


def write_lock(*, progress: bool = False) -> None:
    """**Step 19 only** -- not invoked by this commit. Bootstraps the real
    Docling model set into `MODELS_DIR`, validates it, writes
    `DOCLING_MODEL_MANIFEST.md` + `upstream_licenses/<asset>/` (see
    `_write_model_manifest_and_licences`, §5.3), and writes
    `DOCLING_ASSETS.lock`."""
    docling_artifacts = _import_docling_artifacts()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tree_digest = docling_artifacts.bootstrap(MODELS_DIR, progress=progress)
    docling_artifacts.validate(MODELS_DIR, expected_tree_digest=tree_digest)

    _write_model_manifest_and_licences(docling_artifacts)

    entries = _hash_tree(MODELS_DIR)
    # Build the archive here too (not only via a separate --write-archive
    # call) so DOCLING_ASSETS.lock's own `archive_sha256` is always
    # consistent with the tar actually on disk when this function returns
    # -- a standalone --write-archive re-run later (e.g. to prove
    # byte-for-byte determinism, test_docling_archive_deterministic) is
    # still fully supported and produces identical bytes.
    archive_sha256 = _build_archive().hexdigest()
    ASSETS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["files = {"]
    for relpath, meta in sorted(entries.items()):
        lines.append(f"    {relpath!r}: {{'sha256': {meta['sha256']!r}, 'size': {meta['size']!r}}},")
    lines.append("}")
    lines.append(f"tree_digest = {tree_digest!r}")
    lines.append(f"archive_sha256 = {archive_sha256!r}")
    ASSETS_LOCK_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_archive() -> "hashlib._Hash":
    """`docling-assets-v1.tar` -- `tarfile` over `MODELS_DIR`,
    `UPSTREAM_LICENSES_DIR`, and `MODEL_MANIFEST_PATH`; entries sorted by
    name, every member `mtime=0`/`uid=gid=0`/`uname=gname=""`, mode
    normalised to `0o644`/`0o755`, `PAX_FORMAT`, no compression (§5.3).
    Returns the archive's sha256."""
    members: list[Path] = []
    if MODELS_DIR.exists():
        members.extend(p for p in MODELS_DIR.rglob("*") if p.is_file())
    if UPSTREAM_LICENSES_DIR.exists():
        members.extend(p for p in UPSTREAM_LICENSES_DIR.rglob("*") if p.is_file())
    if MODEL_MANIFEST_PATH.exists():
        members.append(MODEL_MANIFEST_PATH)

    def _reset(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
        tarinfo.mtime = 0
        tarinfo.uid = 0
        tarinfo.gid = 0
        tarinfo.uname = ""
        tarinfo.gname = ""
        tarinfo.mode = 0o755 if tarinfo.isdir() else 0o644
        return tarinfo

    with tarfile.open(ARCHIVE_PATH, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(members, key=lambda p: p.as_posix()):
            # Round-19 finding: the workflow (§5.5) extracts this archive
            # with `-C assets/models/track2-docling-cache` (CACHE_ROOT), not
            # `-C assets/models`. A member under MODELS_DIR must therefore
            # be named relative to CACHE_ROOT ("models/...") so it lands
            # directly at MODELS_DIR after extraction -- naming it relative
            # to `assets/models` ("track2-docling-cache/models/...", the
            # previous behaviour) doubly-nests it under CACHE_ROOT instead.
            # upstream_licenses/ and the manifest live outside CACHE_ROOT
            # entirely (they're committed to git directly, not consumed via
            # this extraction path), so they keep their `assets/models`-
            # relative names unchanged.
            if path.is_relative_to(MODELS_DIR):
                arcname = "models" / path.relative_to(MODELS_DIR)
            else:
                arcname = path.relative_to(REPO_ROOT / "assets" / "models")
            tar.add(path, arcname=arcname.as_posix(), filter=_reset)

    digest = hashlib.sha256()
    with ARCHIVE_PATH.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest


def write_archive() -> None:
    """**Step 19 only** -- not invoked by this commit."""
    _build_archive()


def check(*, skip_archive_hash: bool = False) -> None:
    """Read-only. Recomputes `compute_tree_digest(MODELS_DIR)` and asserts
    it equals `DOCLING_ASSETS.lock`'s recorded `tree_digest`; never writes.
    Without `--skip-archive-hash`, also requires `ARCHIVE_PATH` to be
    present on disk and recomputes its sha256 against the lock's
    `archive_sha256` (§5.3)."""
    if not ASSETS_LOCK_PATH.exists():
        raise FileNotFoundError(f"{ASSETS_LOCK_PATH} does not exist yet (written at step 19, --write-lock)")
    lock_ns: dict = {}
    exec(ASSETS_LOCK_PATH.read_text(encoding="utf-8"), {}, lock_ns)

    docling_artifacts = _import_docling_artifacts()
    actual_digest = docling_artifacts.compute_tree_digest(MODELS_DIR)
    if actual_digest != lock_ns["tree_digest"]:
        raise ValueError(f"tree digest mismatch: expected {lock_ns['tree_digest']!r}, got {actual_digest!r}")

    if not skip_archive_hash:
        if not ARCHIVE_PATH.exists():
            raise FileNotFoundError(f"{ARCHIVE_PATH} not present -- pass --skip-archive-hash on a cache-hit run")
        digest = hashlib.sha256()
        with ARCHIVE_PATH.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != lock_ns["archive_sha256"]:
            raise ValueError("archive sha256 mismatch against DOCLING_ASSETS.lock")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-lock", action="store_true")
    mode.add_argument("--write-archive", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--skip-archive-hash", action="store_true", help="--check only: skip the archive-bytes hash (cache-hit path)")
    parser.add_argument("--progress", action="store_true", help="--write-lock only: show bootstrap download progress")
    args = parser.parse_args(argv)

    if args.write_lock:
        write_lock(progress=args.progress)
        print(f"wrote {ASSETS_LOCK_PATH.relative_to(REPO_ROOT)}")
    elif args.write_archive:
        write_archive()
        print(f"wrote {ARCHIVE_PATH.relative_to(REPO_ROOT)}")
    elif args.check:
        check(skip_archive_hash=args.skip_archive_hash)
        print("OK: Docling asset tree matches DOCLING_ASSETS.lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
