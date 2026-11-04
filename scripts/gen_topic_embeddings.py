"""One-shot generator for ``examples/chapter/data/topic_corpus_embeddings.npy``
and its manifest. NOT run in CI -- it needs the ``all-MiniLM-L6-v2`` model,
which is downloaded from the Hugging Face Hub on first use.

Run once, from the repo root, on a machine with the chapter's
``examples/chapter/requirements.txt`` installed:

    python scripts/gen_topic_embeddings.py

The committed ``.npy`` is the authoritative artifact -- the chapter's S11
exercise and its ``slow`` determinism test run against THIS file, never a
freshly generated one. Re-running the generator on the pinned model revision
reproduces the embeddings closely, but NOT necessarily byte-for-byte: the
float output of MiniLM inference depends on the exact torch build (CPU kernels
differ between torch versions), and the shipped ``requirements.txt`` lock may
resolve a different torch than the machine that produced the committed file.
If you need bit-exact regeneration, do it inside an environment that matches
the ``torch_version`` recorded in the manifest.

Dependency note (G5): ``torch_version`` is read via a metadata lookup, not by
importing torch. If a future revision imports torch directly, add ``torch`` as
an explicit line in ``examples/chapter/requirements.in``.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "examples" / "chapter" / "data"
_CORPUS = _DATA / "topic_corpus.json"
_NPY = _DATA / "topic_corpus_embeddings.npy"
_MANIFEST = _DATA / "topic_corpus_embeddings.manifest.json"
_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned Hugging Face snapshot commit -- passed explicitly to
# SentenceTransformer so regeneration is reproducible even if `main` moves.
_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from sentence_transformers import SentenceTransformer

    corpus = json.loads(_CORPUS.read_text(encoding="utf-8"))
    texts = corpus["documents"]

    model = SentenceTransformer(_MODEL_ID, revision=_MODEL_REVISION, device="cpu")
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
    np.save(_NPY, embeddings)

    manifest = {
        "corpus_file": _CORPUS.name,
        "corpus_sha256": _sha256(_CORPUS),
        "generator": Path(__file__).name,
        "generator_sha256": _sha256(Path(__file__)),
        "embedding_model_id": _MODEL_ID,
        "embedding_model_revision": _MODEL_REVISION,
        "embedding_model_license": "Apache-2.0",
        "embedding_model_attribution": (
            "Reimers & Gurevych, Sentence-BERT; model card at "
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
        ),
        "embeddings_sha256": _sha256(_NPY),
        "embeddings_shape": list(embeddings.shape),
        "embeddings_dtype": str(embeddings.dtype),
        "sentence_transformers_version": version("sentence-transformers"),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
        "generated_on": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
    }
    _MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {_NPY.name} {embeddings.shape} and {_MANIFEST.name}")


if __name__ == "__main__":
    main()
