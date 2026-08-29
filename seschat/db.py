"""
db.py — Stage 3 of Seschat: persist scanned repository metadata into SQLite,
instead of only writing a JSON blob.

Schema (five tables — four from roadmap.md's Stage 3 sketch, plus comments
since Stage 2 already extracts that data):

    files       one row per scanned file (path, language, status, detail)
    classes     one row per class/struct/interface definition, FK -> files
    functions   one row per function/method definition, FK -> files
    imports     one row per import/include/use statement, FK -> files
    comments    one row per comment, FK -> files

Why drop-and-rebuild instead of diffing against the previous run?
Incremental indexing is explicitly called out as Stage 13 ("Polish")
work. For now, every `seschat index` run DROPs and recreates all five
tables and re-inserts everything from scratch — simple and always
correct (no stale rows left behind for files that were renamed or
deleted since the last run), if not maximally efficient on huge repos
yet.

This module deliberately doesn't replace metadata.write_metadata_index():
the JSON index still gets written too. The DB is an additional,
queryable representation of the same data (that's the whole point of
Stage 3 — "now you can query the repository").
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from seschat.metadata import FileMetadata

# seschat/db.py — changed sections only

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    path     TEXT NOT NULL UNIQUE,
    language TEXT NOT NULL,
    status   TEXT NOT NULL,
    detail   TEXT
);

CREATE TABLE IF NOT EXISTS classes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS functions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    statement TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    text    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id  INTEGER NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_classes_name    ON classes(name);
CREATE INDEX IF NOT EXISTS idx_functions_name  ON functions(name);
CREATE INDEX IF NOT EXISTS idx_files_path      ON files(path);
CREATE INDEX IF NOT EXISTS idx_embeddings_file ON embeddings(file_id);
"""


def write_database(file_metadata: list[FileMetadata], root: Path) -> Path:
    out_dir = root / ".seschat"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "index.db"

    conn = sqlite3.connect(out_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()

        # Drop children before parents. embeddings dropped here too, even
        # though Stage 6 populates it via a separate call after this
        # function returns — otherwise a rebuilt `files` table (new
        # AUTOINCREMENT ids) could leave stale/mismatched embedding rows
        # behind. Re-embedding every `seschat index` run is the same
        # drop-and-rebuild tradeoff Stage 3 already made for the rest of
        # the db; caching embeddings across runs is Stage 13 work.
        cur.executescript(
            "DROP TABLE IF EXISTS embeddings;"
            "DROP TABLE IF EXISTS comments;"
            "DROP TABLE IF EXISTS imports;"
            "DROP TABLE IF EXISTS functions;"
            "DROP TABLE IF EXISTS classes;"
            "DROP TABLE IF EXISTS files;"
        )
        cur.executescript(SCHEMA)

        for m in file_metadata:
            cur.execute(
                "INSERT INTO files (path, language, status, detail) VALUES (?, ?, ?, ?)",
                (m.path, m.language, m.status, m.detail),
            )
            file_id = cur.lastrowid

            if m.classes:
                cur.executemany(
                    "INSERT INTO classes (file_id, name) VALUES (?, ?)",
                    [(file_id, name) for name in m.classes],
                )
            if m.functions:
                cur.executemany(
                    "INSERT INTO functions (file_id, name) VALUES (?, ?)",
                    [(file_id, name) for name in m.functions],
                )
            if m.imports:
                cur.executemany(
                    "INSERT INTO imports (file_id, statement) VALUES (?, ?)",
                    [(file_id, stmt) for stmt in m.imports],
                )
            if m.comments:
                cur.executemany(
                    "INSERT INTO comments (file_id, text) VALUES (?, ?)",
                    [(file_id, text) for text in m.comments],
                )

        conn.commit()
    finally:
        conn.close()

    return out_path


def write_embeddings(db_path: Path, embeddings: list[tuple[str, str, str, int, bytes]]) -> int:
    """
    Stage 6: persist one embedding row per file.

    `embeddings` is a list of (path, provider, model, dim, vector_blob)
    tuples — deliberately raw bytes for the vector, not a list of
    floats, since db.py has no reason to know how vectors are encoded
    (that's embeddings.encode_vector()'s job); it just stores BLOBs.

    Looks up each file's id by path (must already exist via
    write_database(), which runs first). Paths not found are skipped,
    not raised, since the caller may have filtered which files to embed.
    Returns the number of rows inserted.
    """
    conn = sqlite3.connect(db_path)
    inserted = 0
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()
        for path, provider, model, dim, vector_blob in embeddings:
            row = cur.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
            if row is None:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO embeddings (file_id, provider, model, dim, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                (row[0], provider, model, dim, vector_blob),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def get_counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        counts = {}
        for table in ("files", "classes", "functions", "imports", "comments", "embeddings"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        return counts
    finally:
        conn.close()
