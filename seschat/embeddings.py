# seschat/embeddings.py
"""
embeddings.py — Stage 6 of Seschat: turns text into vectors, local-first
with an automatic hosted fallback. Same dual-backend shape as llm.py,
kept in its own file rather than folded into llm.py because embedding
calls are a different API/method entirely (embed vs chat), and because
a query embedding sometimes needs to be FORCED onto a specific backend
(see embed_texts' force_provider) — a distinction llm.py never needs.
"""

from __future__ import annotations

import array
import os
from dataclasses import dataclass

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from google import genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-001"


class EmbeddingError(Exception):
    """Base class for anything that goes wrong turning text into vectors."""


class EmbeddingNotConfiguredError(EmbeddingError):
    """A backend's dependency isn't installed, or its server/credentials aren't reachable."""


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    provider: str   # "ollama" | "gemini"
    model: str
    dim: int


def _resolve_ollama_model(model: str | None) -> str:
    return model or os.environ.get("SESCHAT_OLLAMA_EMBED_MODEL") or DEFAULT_OLLAMA_EMBED_MODEL


def _resolve_ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def _resolve_gemini_model(model: str | None) -> str:
    return model or os.environ.get("SESCHAT_GEMINI_EMBED_MODEL") or DEFAULT_GEMINI_EMBED_MODEL


def _call_ollama_embed(texts: list[str], model: str | None, host: str) -> EmbeddingBatch:
    if not OLLAMA_AVAILABLE:
        raise EmbeddingNotConfiguredError("The 'ollama' package isn't installed.")

    resolved_model = _resolve_ollama_model(model)
    client = ollama.Client(host=host)

    try:
        response = client.embed(model=resolved_model, input=texts)
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            raise EmbeddingNotConfiguredError(
                f"model {resolved_model!r} isn't pulled "
                f"(run `ollama pull {resolved_model}`)"
            ) from e
        raise EmbeddingNotConfiguredError(f"{getattr(e, 'error', e)}") from e
    except Exception as e:
        raise EmbeddingNotConfiguredError(f"couldn't reach Ollama at {host}: {e}") from e

    vectors = [list(v) for v in response.embeddings]
    if not vectors:
        raise EmbeddingError("Ollama returned no embeddings.")
    return EmbeddingBatch(vectors=vectors, provider="ollama", model=resolved_model, dim=len(vectors[0]))


def _call_gemini_embed(texts: list[str], model: str | None) -> EmbeddingBatch:
    if not GEMINI_AVAILABLE:
        raise EmbeddingNotConfiguredError(
            "the 'google-genai' package isn't installed (`pip install -e .` "
            "or `pip install google-genai`)"
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingNotConfiguredError(
            "GEMINI_API_KEY isn't set (get one at https://aistudio.google.com/ "
            "and `export GEMINI_API_KEY=...`)"
        )

    resolved_model = _resolve_gemini_model(model)
    client = genai.Client(api_key=api_key)

    try:
        result = client.models.embed_content(model=resolved_model, contents=texts)
    except Exception as e:
        raise EmbeddingError(f"Gemini embedding API error: {e}") from e

    vectors = [list(e.values) for e in result.embeddings]
    if not vectors:
        raise EmbeddingError("Gemini returned no embeddings.")
    return EmbeddingBatch(vectors=vectors, provider="gemini", model=resolved_model, dim=len(vectors[0]))


def embed_texts(
    texts: list[str], model: str | None = None, force_provider: str | None = None
) -> EmbeddingBatch:
    """
    Turn a batch of texts into vectors in ONE call (much cheaper than one
    round-trip per file). Local-first with automatic hosted fallback,
    same shape as llm.call_llm() — UNLESS force_provider is given
    ("ollama" or "gemini"), which locks to that single backend and raises
    instead of falling back.

    force_provider exists because mixing vectors from two different
    embedding models/dimensions in one cosine-similarity comparison is
    meaningless. semantic.py always re-embeds a query with whichever
    provider actually built the stored index — never "whichever backend
    happens to be reachable right now" — so an index built with Ollama
    can't silently get queried with mismatched Gemini vectors.
    """
    if not texts:
        raise EmbeddingError("embed_texts() called with an empty text list.")

    if force_provider == "ollama":
        return _call_ollama_embed(texts, model, _resolve_ollama_host())
    if force_provider == "gemini":
        return _call_gemini_embed(texts, model)
    if force_provider is not None:
        raise EmbeddingError(f"Unknown force_provider {force_provider!r} (expected 'ollama' or 'gemini').")

    try:
        return _call_ollama_embed(texts, model, _resolve_ollama_host())
    except EmbeddingError as ollama_error:
        try:
            return _call_gemini_embed(texts, model=None)
        except EmbeddingError as gemini_error:
            raise EmbeddingNotConfiguredError(
                f"No usable embedding backend.\n"
                f"  Ollama: {ollama_error}\n"
                f"  Gemini: {gemini_error}\n"
                f"Fix one of these — start Ollama locally and pull "
                f"'{DEFAULT_OLLAMA_EMBED_MODEL}', or set GEMINI_API_KEY."
            ) from gemini_error


# --------------------------------------------------------------------------
# Vector <-> BLOB serialization. Kept here (not in db.py) so db.py never
# needs to know how a vector is encoded — it just stores/returns bytes.
# --------------------------------------------------------------------------

def encode_vector(vector: list[float]) -> bytes:
    """Serialize a vector to raw float32 bytes for SQLite BLOB storage."""
    return array.array("f", vector).tobytes()


def decode_vector(blob: bytes) -> list[float]:
    """Inverse of encode_vector()."""
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)
