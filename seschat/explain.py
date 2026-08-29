"""
explain.py — Stage 8 of Seschat: `seschat explain <path> <file>`.

Pipeline (roadmap.md's Stage 8 sketch, implemented literally):

    read file
      -> retrieve metadata      (search.py — direct by-path lookup, Stage 7's get_file_structure)
      -> read source            (this module — same disk-read pattern ask.py already uses)
      -> construct prompt       (this module)
      -> LLM                    (llm.py)
      -> summary

Unlike `ask`, there's no retrieval step here in the Stage 4/6/7 sense —
the user names the exact file they want explained, so "search" is just
a direct lookup of that one file's already-extracted metadata plus a
fresh read of its source off disk. The LLM still never sees anything
Python didn't hand it first: metadata comes from the SQLite index
(Stage 3), source comes straight from disk, and the model's job is
purely to synthesize a human-readable summary from both — same
LLM-as-reader principle as `ask.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seschat.llm import call_llm
from seschat.search import IndexNotFoundError, get_file_record, get_file_structure

__all__ = [
    "FileNotIndexedError",
    "ExplainResult",
    "build_prompt",
    "explain_file",
]

MAX_SOURCE_CHARS = 6000  # explain gets a bigger budget than ask's per-file cap
                          # (MAX_CHARS_PER_FILE=3000 in ask.py) since a single
                          # file is the entire context here, not one of several.


class FileNotIndexedError(Exception):
    """Raised when the requested path isn't in the repository's index at all."""


def _normalize_rel_path(root: Path, path: str) -> str:
    """
    Accept either a path relative to the repo root (as stored in the DB,
    e.g. "src/cache.cpp") or an absolute path pointing at the same file,
    and return the DB's stored form. `files.path` is always written
    relative to `root` (scanner.py does `full.relative_to(root)`), so an
    absolute path needs converting before it'll match anything.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            return path  # outside root entirely — let the lookup fail naturally
    return Path(path).as_posix()


def _load_source(root: Path, rel_path: str) -> tuple[str | None, bool, str | None]:
    """Best-effort read of the file's actual source, same truncation pattern as ask.py."""
    full = root / rel_path
    try:
        text = full.read_text(errors="replace")
    except OSError as e:
        return None, False, f"could not read file: {e}"

    if len(text) > MAX_SOURCE_CHARS:
        return text[:MAX_SOURCE_CHARS], True, None
    return text, False, None


SYSTEM_PROMPT = (
    "You are a code documentation assistant. You will be given one "
    "file's extracted structure (classes, functions, imports, comments) "
    "and its source code. Using ONLY that information, write a clear "
    "explanation covering: (1) a one-paragraph summary of what this "
    "file does, (2) its main responsibilities, (3) the important "
    "classes/functions and what each does, (4) what it depends on "
    "(its imports/dependencies) and, if apparent from the source, what "
    "depends on it, (5) any suggestions for improvement (readability, "
    "structure, missing error handling, etc.) — mark suggestions "
    "clearly as opinions, not facts about the code. Do not invent "
    "behavior that isn't supported by the provided structure or "
    "source. Use short headers or a brief list for each section and "
    "be concise."
)


def build_prompt(
    rel_path: str,
    language: str,
    status: str,
    detail: str | None,
    classes: list[str],
    functions: list[str],
    imports: list[str],
    comments: list[str],
    source: str | None,
    source_truncated: bool,
    read_error: str | None,
) -> str:
    """Assemble the user-turn prompt: file identity, extracted structure, then source."""
    sections = [f"File: {rel_path}", f"Language: {language}"]

    if status != "parsed":
        note = f"Note: this file's status is {status!r}"
        if detail:
            note += f" ({detail})"
        note += " — structural metadata below may be incomplete or empty."
        sections.append(note)

    if classes:
        sections.append("Classes: " + ", ".join(classes))
    if functions:
        sections.append("Functions: " + ", ".join(functions))
    if imports:
        sections.append("Imports: " + ", ".join(imports))
    if comments:
        sections.append("Comments: " + " | ".join(comments))

    if source is not None:
        note = " (truncated)" if source_truncated else ""
        sections.append(f"Source{note}:\n{source}")
    elif read_error:
        sections.append(f"[source unavailable: {read_error}]")

    return "\n".join(sections)


@dataclass
class ExplainResult:
    explanation: str
    path: str
    language: str
    status: str
    classes: list[str]
    functions: list[str]
    imports: list[str]
    comments: list[str]
    model: str
    source_truncated: bool = False
    read_error: str | None = None


def explain_file(
    root: Path,
    db_path: Path,
    path: str,
    model: str | None = None,
) -> ExplainResult:
    """
    Run the Stage 8 pipeline for a single file: look up its indexed
    metadata by path, read its source fresh off disk, build a prompt,
    and hand both to the LLM for a human-readable explanation.

    Raises IndexNotFoundError if the repo hasn't been indexed at all
    (propagated from search.py), and FileNotIndexedError if the repo
    IS indexed but this particular path has no entry in it (never
    scanned, wrong path, or scanned after the last `seschat index` run).
    Raises LLMError/LLMNotConfiguredError (from llm.py) if the model
    call itself can't happen.
    """
    rel_path = _normalize_rel_path(root, path)

    record = get_file_record(db_path, rel_path)
    if record is None:
        raise FileNotIndexedError(
            f"{rel_path!r} isn't in the index. Run `seschat index {root}` "
            f"first, or check the path is relative to the repo root."
        )
    language, status, detail = record

    classes, functions, imports, comments = get_file_structure(db_path, rel_path)
    source, truncated, read_error = _load_source(root, rel_path)

    prompt = build_prompt(
        rel_path, language, status, detail,
        classes, functions, imports, comments,
        source, truncated, read_error,
    )

    response = call_llm(prompt, system=SYSTEM_PROMPT, model=model)

    return ExplainResult(
        explanation=response.text,
        path=rel_path,
        language=language,
        status=status,
        classes=classes,
        functions=functions,
        imports=imports,
        comments=comments,
        model=f"{response.provider}:{response.model}",
        source_truncated=truncated,
        read_error=read_error,
    )
