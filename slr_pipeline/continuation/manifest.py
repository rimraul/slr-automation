"""`ContinuationManifest` -- typed model for the `manifest.json` an offline
PDF-retrieval/parse pass writes for a continuation (Unit A items 15,
round-30/v31 fix; round-29 finding 2 -- dot-access only, no dict-bracket
outlier anywhere consuming this object). Confirmed against the real,
already-completed
`track3_runs/track3_continuation_20260914T144702Z/manifest.json`: only
`continuation_workspace` (top level) and `worklist_id`/`save_as`/
`file_sha256`/`status`/`identity_flag` (per document) are read by this
package -- every other real field on that fixture (`expected_doi`,
`document_sha256`, `parser_name`, ...) is informational, produced by the
external offline-parse tooling, and is neither required nor rejected here
(`extra="allow"`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, ValidationError

from slr_pipeline.track3_run import Track3ConfigError


class ContinuationManifestDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    worklist_id: str
    save_as: str
    file_sha256: str
    status: str
    identity_flag: str


class ContinuationManifest(BaseModel):
    model_config = ConfigDict(extra="allow")
    continuation_workspace: str
    documents: List[ContinuationManifestDocument]


def load_continuation_manifest(path: Path) -> ContinuationManifest:
    """Loads and validates `--continuation-manifest`. Round-29 finding 3:
    every one of `FileNotFoundError`/`json.JSONDecodeError`/pydantic's
    `ValidationError` is converted into `Track3ConfigError` naming `path` --
    never an unhandled exception of any of these three types."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return ContinuationManifest.model_validate(parsed)
    except FileNotFoundError as exc:
        raise Track3ConfigError(f"--continuation-manifest {path}: file not found") from exc
    except json.JSONDecodeError as exc:
        raise Track3ConfigError(f"--continuation-manifest {path}: malformed JSON ({exc})") from exc
    except ValidationError as exc:
        raise Track3ConfigError(f"--continuation-manifest {path}: does not match the manifest schema ({exc})") from exc
