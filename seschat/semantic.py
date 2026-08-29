# seschat/semantic.py
"""
semantic.py — Stage 6 of Seschat: semantic search over the repository,
complementing Stage 4's keyword search (search.py) with embedding-based
similarity. Where `seschat query` only matches literal substrings,
`seschat semantic` can surface a file about authentication even if the
word "authentication" never appears in it — the whole point of
embeddings (roadmap.md's Stage 6 goal).

Two things live here:
    1. Building the index — build_and_store_embeddings(): read every
       scanned file's extracted metadata + source, turn each into one
       embedding vector, store it via db.write_embeddings().
    2. Querying the index — semantic_search(): embed the query with the
       SAME model that built the index, and rank every stored file by
       cosine similarity.

Granularity: Stage 6 embeds one vector PER FILE, not per function/class.
Function-level chunking would give tighter retrieval, but metadata.py
doesn't currently record where in the file a function/class starts and
ends (only its name) — only the whole file's source can be read back off
disk. File-level embeddings are a deliberate, documented simplification;
per-chunk embeddings with byte-range tracking are a natural upgrade,
and Stage 7's RAG pipeline (which explicitly has a "chunking" step) is
the obvious place to revisit this.

This module never talks to a model directly — embeddings.py is the only
file that does (mirrors llm.py's role for `seschat ask`). This module's
only direct SQL is a single read query; db.py still owns schema/writes.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from seschat.db import write_embeddings
from seschat.embeddings import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    decode_vector,
    embed_texts,
    encode_vector,
)
from seschat.metadata import FileMetadata

__all__ = [
    "build_embedding_text",
    "build_and_store_embeddings",
    "semantic_search",
    "SemanticMatch",
    "EmbeddingIndexResult",
    "EmbeddingsNotFoundError",
]

MAX_SOURCE_CHARS = 4000  # rough cap so one huge file doesn't dominate its own embedding text


class EmbeddingsNotFoundError(Exception):
    """Raised when there's no usable embedding index (missing db or empty embeddings table)."""


# --------------------------------------------------------------------------
# Step 1: build the text that actually gets embedded, per file.
# --------------------------------------------------------------------------

def build_embedding_text(meta: FileMetadata, root: Path) -> str | None:
    """
    Construct the text representation of one file that gets turned into
    a vector. For parsed files this leads with the extracted structure
    (classes/functions/imports/comments, read straight off FileMetadata —
    no DB round-trip needed) since names carry a lot of semantic signal
    cheaply, then appends a source excerpt for whatever the structure
    doesn't capture (docstring prose, overall shape). For files with no
    extracted structure (not_applicable languages like Markdown, or
    unsupported/unrecognized files), the source excerpt is all there is —
    which is fine, since a README's prose is exactly what semantic
    search should still be able to find.

    Returns None if there's nothing usable (source unreadable and no
    extracted structure at all) — the caller skips those files rather
    than embedding an empty/near-empty string.
    """
    parts: list[str] = [f"File: {meta.path}", f"Language: {meta.language}"]

    if meta.classes:
        parts.append("Classes: " + ", ".join(meta.classes))
    if meta.functions:
        parts.append("Functions: " + ", ".join(meta.functions))
    if meta.imports:
        parts.append("Imports: " + ", ".join(meta.imports))
    if meta.comments:
        parts.append("Comments: " + " | ".join(meta.comments))

    try:
        source = (root / meta.path).read_text(errors="replace")
    except OSError:
        source = None

    if source:
        if len(source) > MAX_SOURCE_CHARS:
            source = source[:MAX_SOURCE_CHARS]
        parts.append("Source:\n" + source)

    if len(parts) <= 2:  # only the File:/Language: header — nothing to embed
        return None

    return "\n".join(parts)


# --------------------------------------------------------------------------
# Step 2: embed every file's text and persist the vectors.
# --------------------------------------------------------------------------

@dataclass
class EmbeddingIndexResult:
    embedded_count: int
    skipped_count: int  # files with nothing usable to embed
    provider: str | None
    model: str | None


def build_and_store_embeddings(
    file_metadata: list[FileMetadata], root: Path, db_path: Path, model: str | None = None
) -> EmbeddingIndexResult:
    """
    For every file with something usable to embed, build its embedding
    text, embed the whole batch in ONE call (one API/daemon round-trip,
    not one per file), and write the resulting vectors to the
    `embeddings` table via db.write_embeddings().

    Raises EmbeddingNotConfiguredError/EmbeddingError (from embeddings.py)
    if neither backend is usable — same failure shape `seschat ask` already
    surfaces, handled the same way by the CLI.
    """
    paths: list[str] = []
    texts: list[str] = []
    skipped = 0

    for meta in file_metadata:
        text = build_embedding_text(meta, root)
        if text is None:
            skipped += 1
            continue
        paths.append(meta.path)
        texts.append(text)

    if not texts:
        return EmbeddingIndexResult(embedded_count=0, skipped_count=skipped, provider=None, model=None)

    batch = embed_texts(texts, model=model)

    rows = [
        (path, batch.provider, batch.model, batch.dim, encode_vector(vector))
        for path, vector in zip(paths, batch.vectors)
    ]
    inserted = write_embeddings(db_path, rows)

    return EmbeddingIndexResult(
        embedded_count=inserted, skipped_count=skipped, provider=batch.provider, model=batch.model
    )


# --------------------------------------------------------------------------
# Step 3: query the stored vectors.
# --------------------------------------------------------------------------

@dataclass
class SemanticMatch:
    path: str
    language: str
    score: float  # cosine similarity


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_index(db_path: Path):
    """
    Load every stored embedding plus which provider/model built the
    index. All rows in a given repo's index come from a single
    build_and_store_embeddings() call, so provider/model is uniform
    across the table — read off the first row rather than needing a
    separate index-metadata table.
    """
    if not db_path.exists():
        raise EmbeddingsNotFoundError(
            f"No index found at {db_path}. Run `seschat index <path>` first."
        )

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT f.path, f.language, e.provider, e.model, e.vector
                FROM embeddings e JOIN files f ON f.id = e.file_id
                """
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []  # embeddings table doesn't exist — pre-Stage-6 db
    finally:
        conn.close()

    if not rows:
        raise EmbeddingsNotFoundError(
            "No embeddings found in the index. Run `seschat index <path>` "
            "without --no-embeddings (and without --no-db) first."
        )

    return rows[0][2], rows[0][3], rows  # provider, model, rows


def semantic_search(db_path: Path, query: str, limit: int = 10) -> list[SemanticMatch]:
    """
    Embed `query` with the SAME provider/model that built the stored
    index (forced via force_provider/model — never "whichever backend
    happens to be up right now"), then rank every file by cosine
    similarity to that query vector.

    Raises EmbeddingsNotFoundError if the repo hasn't been embedded yet,
    and EmbeddingError/EmbeddingNotConfiguredError if the matching
    backend isn't reachable right now (e.g. the index was built with
    Ollama but Ollama isn't running anymore).
    """
    provider, model, rows = _load_index(db_path)

    query_vector = embed_texts([query], model=model, force_provider=provider).vectors[0]

    matches = [
        SemanticMatch(path=path, language=language, score=_cosine_similarity(query_vector, decode_vector(blob)))
        for path, language, _row_provider, _row_model, blob in rows
    ]
    matches.sort(key=lambda m: -m.score)
    return matches[:limit]
