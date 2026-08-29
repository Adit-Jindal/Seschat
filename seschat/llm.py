"""
llm.py — Stage 5 of Seschat: talks to a model, local-first with an
automatic hosted fallback.

This is deliberately the *only* file in Seschat that talks to a model.
Every other module (scanner, metadata, db, search) has no idea an LLM
exists — that separation is the point of Stage 5's pipeline: "the LLM
doesn't search, your software searches" (roadmap.md). This module's job
starts and ends at "given a finished prompt, get an answer back."

Two backends, tried in order:
    1. Ollama (local)   — no API key, no network, no per-token cost.
                           Requires the daemon running and the model
                           already pulled.
    2. Gemini (hosted)  — automatic fallback if Ollama isn't usable for
                           any reason. This is what makes `seschat ask`
                           work out of the box for someone who clones
                           this repo without any local model set up —
                           they just need GEMINI_API_KEY.

If BOTH backends fail, call_llm() raises LLMNotConfiguredError with both
underlying reasons folded into the message, so it's obvious which one(s)
to fix.

Configuration (no config file yet — that's Stage 13 polish):
    SESCHAT_OLLAMA_MODEL   optional. Overrides DEFAULT_OLLAMA_MODEL.
    OLLAMA_HOST           optional. Overrides DEFAULT_OLLAMA_HOST.
    GEMINI_API_KEY        required for the Gemini fallback to work.
    SESCHAT_GEMINI_MODEL   optional. Overrides DEFAULT_GEMINI_MODEL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_MAX_TOKENS = 1024


class LLMError(Exception):
    """Base class for anything that goes wrong talking to a model."""


class LLMNotConfiguredError(LLMError):
    """A backend's dependency isn't installed, or its server/credentials aren't reachable."""


@dataclass
class LLMResponse:
    text: str
    provider: str   # "ollama" | "gemini" — whichever one actually answered
    model: str


def _resolve_ollama_model(model: str | None) -> str:
    return model or os.environ.get("SESCHAT_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL


def _resolve_ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def _resolve_gemini_model() -> str:
    return os.environ.get("SESCHAT_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


# --------------------------------------------------------------------------
# Backend 1: local Ollama.
# --------------------------------------------------------------------------

def _call_ollama(
    prompt: str, system: str | None, model: str | None, max_tokens: int
) -> LLMResponse:
    if not OLLAMA_AVAILABLE:
        raise LLMNotConfiguredError("The 'ollama' package isn't installed.")

    resolved_model = _resolve_ollama_model(model)
    host = _resolve_ollama_host()
    client = ollama.Client(host=host)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat(
            model=resolved_model,
            messages=messages,
            options={"num_predict": max_tokens},
        )
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            raise LLMNotConfiguredError(
                f"model {resolved_model!r} isn't pulled "
                f"(run `ollama pull {resolved_model}`)"
            ) from e
        raise LLMNotConfiguredError(f"{getattr(e, 'error', e)}") from e
    except Exception as e:
        # Almost always "daemon not running" / connection refused. Treated
        # as LLMNotConfiguredError (not a hard LLMError) specifically so
        # call_llm() below falls back to Gemini instead of giving up.
        raise LLMNotConfiguredError(f"couldn't reach Ollama at {host}: {e}") from e

    return LLMResponse(
        text=response.message.content.strip(), provider="ollama", model=resolved_model
    )


# --------------------------------------------------------------------------
# Backend 2: Gemini API (fallback).
# --------------------------------------------------------------------------

def _call_gemini(prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
    if not GEMINI_AVAILABLE:
        raise LLMNotConfiguredError(
            "the 'google-genai' package isn't installed (`pip install -e .` "
            "or `pip install google-genai`)"
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMNotConfiguredError(
            "GEMINI_API_KEY isn't set (get one at https://aistudio.google.com/ "
            "and `export GEMINI_API_KEY=...`)"
        )

    resolved_model = _resolve_gemini_model()
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
    except Exception as e:
        raise LLMError(f"Gemini API error: {e}") from e

    return LLMResponse(text=(response.text or "").strip(), provider="gemini", model=resolved_model)


# --------------------------------------------------------------------------
# Public entry point: local-first, automatic hosted fallback.
# --------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LLMResponse:
    """
    Try the local Ollama model first. If it isn't installed, isn't
    running, or doesn't have the model pulled, fall back automatically
    to the Gemini API — so someone who clones this repo without any
    local setup still gets a working `seschat ask`, as long as
    GEMINI_API_KEY is set.

    `model`, if given, only overrides the *Ollama* model name — an
    Ollama tag like "qwen3:8b" passed straight to the Gemini API would
    just error. The Gemini model is controlled separately via
    $SESCHAT_GEMINI_MODEL. This asymmetry is a deliberate simplification
    for a two-backend, no-config-file setup; revisit if a third backend
    gets added.

    Raises LLMNotConfiguredError only if *both* backends fail — the
    Ollama failure reason is folded into that message so you're not left
    guessing why the fallback triggered.
    """
    try:
        return _call_ollama(prompt, system, model, max_tokens)
    except LLMError as ollama_error:
        try:
            return _call_gemini(prompt, system, max_tokens)
        except LLMError as gemini_error:
            raise LLMNotConfiguredError(
                f"No usable LLM backend.\n"
                f"  Ollama: {ollama_error}\n"
                f"  Gemini: {gemini_error}\n"
                f"Fix one of these — start Ollama locally and pull a model, "
                f"or set GEMINI_API_KEY."
            ) from gemini_error
