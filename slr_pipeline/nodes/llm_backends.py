"""
LLM backend abstraction for the SLR pipeline.

Provides a unified interface for switching between OpenAI, Ollama, and
HuggingFace backends. Adapted from single_doc_20251105.ipynb.

The notebook used LlamaIndex's global Settings object (Settings.llm,
Settings.embed_model) which creates global state — problematic in a
multi-threaded pipeline. Here we return explicit client objects instead.

Usage:
    client, embed_model = get_llm_backend(cfg)
    # client is always an openai.OpenAI-compatible object
    # embed_model is always a SentenceTransformer (for consistency)

NOTE: HuggingFace backend currently requires CUDA for reasonable speed.
Ollama backend requires a running Ollama server (http://localhost:11434).
For the textbook, OpenAI is the recommended default.

Known gaps from NB2 (single_doc_20251105.ipynb):
  - BitsAndBytes quantization (4-bit) not implemented here — NB2 had this
    for GPU memory efficiency; add if running local models on GPU
  - Direct HF generation pipeline not included — NB2 bypassed LlamaIndex
    for this; not needed when using OpenAI-compatible local servers
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend: OpenAI (default)
# ---------------------------------------------------------------------------

def _backend_openai(
    api_key: Optional[str],
    llm_model: str = "gpt-4o-mini",
    embed_model_name: str = "all-MiniLM-L6-v2",
):
    """Returns (openai.OpenAI client, SentenceTransformer embed model)."""
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    client = OpenAI(api_key=api_key)
    embed_model = SentenceTransformer(embed_model_name)
    logger.info(f"Backend: OpenAI ({llm_model})")
    return client, embed_model


# ---------------------------------------------------------------------------
# Backend: Ollama (local, OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _backend_ollama(
    llm_model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434/v1",
    embed_model_name: str = "all-MiniLM-L6-v2",
):
    """
    Ollama exposes an OpenAI-compatible endpoint, so we can use the
    standard openai.OpenAI client pointed at localhost.

    Adapted from use_hf_ollama() in single_doc_20251105.ipynb, but
    without LlamaIndex dependency.

    Requires: Ollama server running locally with the model pulled.
      ollama pull llama3.2:3b
    """
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    client = OpenAI(base_url=base_url, api_key="ollama")
    embed_model = SentenceTransformer(embed_model_name)
    logger.info(f"Backend: Ollama ({llm_model} @ {base_url})")
    return client, embed_model


# ---------------------------------------------------------------------------
# Backend: HuggingFace (local inference, OpenAI-compatible via llama-cpp-python
# or text-generation-inference server)
# ---------------------------------------------------------------------------

def _backend_huggingface_server(
    base_url: str,
    llm_model: str = "tgi-model",
    embed_model_name: str = "all-MiniLM-L6-v2",
):
    """
    Connects to a Text Generation Inference (TGI) or llama-cpp-python server
    that exposes an OpenAI-compatible endpoint.

    Start TGI server: docker run ... ghcr.io/huggingface/text-generation-inference
    Start llama.cpp:  python -m llama_cpp.server --model model.gguf

    This avoids the LlamaIndex global state issue in NB2 (use_hf_remote).
    NOTE: Full HF model loading without a server is not implemented here;
    use the Ollama backend with a GGUF model for simplest local inference.
    """
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    client = OpenAI(base_url=base_url, api_key="hf")
    embed_model = SentenceTransformer(embed_model_name)
    logger.info(f"Backend: HuggingFace server ({llm_model} @ {base_url})")
    return client, embed_model


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

def get_llm_backend(cfg, llm_model: Optional[str] = None) -> Tuple[Any, Any]:
    """
    llm_model overrides cfg.screening.llm_model when provided.
    Useful when a specific node needs a different model than the default
    (e.g. search_strategy_builder may use gpt-4o-mini while charting uses gpt-4o).
    """
    """
    Return (llm_client, embed_model) based on ReviewConfig.llm_backend.

    All backends return an openai.OpenAI-compatible client so that node
    code doesn't need to branch on backend type.

    Config keys (add to ReviewConfig if needed):
      llm_backend: "openai" | "ollama" | "hf_server"
      ollama_base_url: str  (default "http://localhost:11434/v1")
      hf_server_url: str
    """
    backend = getattr(cfg, "llm_backend", "openai")
    model = llm_model or cfg.screening.llm_model

    if backend == "ollama":
        return _backend_ollama(
            llm_model=model,
            base_url=getattr(cfg, "ollama_base_url", "http://localhost:11434/v1"),
        )

    if backend == "hf_server":
        return _backend_huggingface_server(
            base_url=getattr(cfg, "hf_server_url", "http://localhost:8080/v1"),
            llm_model=model,
        )

    # Default: OpenAI
    from slr_pipeline.config import resolve_openai_api_key

    return _backend_openai(
        api_key=resolve_openai_api_key(),
        llm_model=model,
    )
