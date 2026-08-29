"""
search.py — Stage 4 of Seschat: keyword search over the SQLite index built
in Stage 3, with no AI involved. Given a term, find every file whose
path, class names, function names, imports, or comments mention it.

Why not embeddings/semantic search yet?
That's Stage 6. This stage is deliberately "dumb" — plain substring
matching via SQL LIKE — because the roadmap's whole point here is to
build the retrieval half of RAG *before* ever calling an LLM, so later
stages can plug a model in on top of something that already works.

No ranking yet either (that's flagged as a stretch item in roadmap.md's
Stage 4 reading list — TF-IDF/inverted indexes). Results are grouped by
file and ordered by how many matches that file has, which is a cheap
proxy for relevance without actually implementing TF-IDF.

Stage 7 addition: get_file_structure() — a direct by-path lookup of a
file's classes/functions/imports/comments, used by ask.py's hybrid
retrieval to fill in structure for files that matched via Stage 6
semantic search but had no keyword hits (and so no Match objects to
derive structure from the way keyword-matched files used to).

Stage 8 addition: get_file_record() — a direct by-path lookup of a
file's (language, status, detail), used by explain.py so it can tell
the model (and the user) when a file's extracted structure is
incomplete for a documented reason rather than genuinely having zero
classes/functions.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Match:
    """A single hit: something in `kind` matched the search term."""
    kind: str          # "path" | "class" | "function" | "import" | "comment"
    text: str          # the matched name/statement/comment text itself


@dataclass
class FileResult:
    """All matches found within one file, grouped together for display."""
    path: str
    language: str
    matches: list[Match] = field(default_factory=list)

    @property
    def match_count(self) -> int:
        return len(self.matches)


class IndexNotFoundError(Exception):
    """Raised when .seschat/index.db doesn't exist — repo hasn't been indexed yet."""


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise IndexNotFoundError(
            f"No index found at {db_path}. Run `seschat index <path>` first."
        )
    return sqlite3.connect(db_path)


def search_repository(db_path: Path, term: str, limit: int = 50) -> list[FileResult]:
    """
    Search files/classes/functions/imports/comments for `term` (case-
    insensitive substring match) and return results grouped by file,
    ordered by descending match count (files with more hits first).

    `limit` caps the number of *files* returned, not the number of raw
    matches — a file with 10 matching functions still counts as 1 toward
    the limit, so results stay grouped and readable.
    """
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        like_term = f"%{term}%"

        files_by_path: dict[str, FileResult] = {}

        def ensure_file(path: str, language: str) -> FileResult:
            if path not in files_by_path:
                files_by_path[path] = FileResult(path=path, language=language)
            return files_by_path[path]

        # Filename/path matches.
        for path, language in cur.execute(
            "SELECT path, language FROM files WHERE path LIKE ? COLLATE NOCASE",
            (like_term,),
        ):
            ensure_file(path, language).matches.append(Match("path", path))

        # Class name matches.
        for path, language, name in cur.execute(
            """
            SELECT f.path, f.language, c.name
            FROM classes c JOIN files f ON f.id = c.file_id
            WHERE c.name LIKE ? COLLATE NOCASE
            """,
            (like_term,),
        ):
            ensure_file(path, language).matches.append(Match("class", name))

        # Function name matches.
        for path, language, name in cur.execute(
            """
            SELECT f.path, f.language, fn.name
            FROM functions fn JOIN files f ON f.id = fn.file_id
            WHERE fn.name LIKE ? COLLATE NOCASE
            """,
            (like_term,),
        ):
            ensure_file(path, language).matches.append(Match("function", name))

        # Import statement matches.
        for path, language, statement in cur.execute(
            """
            SELECT f.path, f.language, i.statement
            FROM imports i JOIN files f ON f.id = i.file_id
            WHERE i.statement LIKE ? COLLATE NOCASE
            """,
            (like_term,),
        ):
            ensure_file(path, language).matches.append(Match("import", statement))

        # Comment matches.
        for path, language, text in cur.execute(
            """
            SELECT f.path, f.language, cm.text
            FROM comments cm JOIN files f ON f.id = cm.file_id
            WHERE cm.text LIKE ? COLLATE NOCASE
            """,
            (like_term,),
        ):
            ensure_file(path, language).matches.append(Match("comment", text))

        results = sorted(
            files_by_path.values(), key=lambda r: (-r.match_count, r.path)
        )
        return results[:limit]
    finally:
        conn.close()


def get_file_structure(db_path: Path, path: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Stage 7: direct lookup of one file's full extracted classes/
    functions/imports/comments by path, independent of any search term.

    Used by ask.py's hybrid retrieval so every retrieved file — whether
    it surfaced via keyword search, semantic search, or both — gets the
    same complete structural context in the LLM prompt, rather than
    keyword-matched files getting only their *matched* names (the old
    Stage 5 behavior) while semantic-only files got none at all.

    Returns ([], [], [], []) if the path isn't in the `files` table
    (shouldn't normally happen — callers only pass paths that came from
    search_repository or semantic_search — but this fails soft rather
    than raising, since a stale/renamed file shouldn't crash `ask`).
    """
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return [], [], [], []
        file_id = row[0]

        classes = [r[0] for r in cur.execute(
            "SELECT name FROM classes WHERE file_id = ? ORDER BY name", (file_id,)
        )]
        functions = [r[0] for r in cur.execute(
            "SELECT name FROM functions WHERE file_id = ? ORDER BY name", (file_id,)
        )]
        imports = [r[0] for r in cur.execute(
            "SELECT statement FROM imports WHERE file_id = ? ORDER BY statement", (file_id,)
        )]
        comments = [r[0] for r in cur.execute(
            "SELECT text FROM comments WHERE file_id = ? ORDER BY text", (file_id,)
        )]
        return classes, functions, imports, comments
    finally:
        conn.close()

def get_file_record(db_path: Path, path: str) -> tuple[str, str, str | None] | None:
    """
    Stage 8: look up one file's (language, status, detail) by path —
    the piece get_file_structure() doesn't cover, since Stage 7 only
    needed classes/functions/imports/comments for prompt context.
    explain.py needs status/detail too, so it can tell the model (and
    the user) when a file's extracted structure is incomplete because
    it's "not_applicable" (e.g. Markdown), "unsupported", etc., rather
    than genuinely having zero classes/functions.

    Returns None if the path isn't in the `files` table at all —
    distinct from get_file_structure()'s soft "return empty lists"
    behavior, because explain.py needs to tell "this file has no
    classes" apart from "this file was never indexed": explain has no
    fallback-to-empty-context behavior the way ask.py does for a
    no-match question, so a missing file is a hard error here.
    """
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT language, status, detail FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row if row else None
    finally:
        conn.close()
