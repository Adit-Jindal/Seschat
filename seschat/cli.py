"""
cli.py — Seschat command-line interface.

Stage 1: `seschat index <path>` scans a repo and reports what's in it.
Stage 2: the same command also extracts per-file metadata (classes,
functions, imports, comments) via tree-sitter and writes it to
`<path>/.seschat/index.json`.
Stage 3: the same command also persists that metadata into a queryable
SQLite database at `<path>/.seschat/index.db`.
Stage 4: a new `seschat query <path> <term>` command does keyword search
over that database — no AI, just SQL LIKE across files/classes/
functions/imports/comments.
Stage 5: `seschat ask <path> "<question>"` answers natural-language
questions using keyword retrieval + an LLM.
Stage 6: `seschat semantic <path> "<query>"` does embedding-based
similarity search, and `index` gains an embedding step.
Stage 7: `seschat ask` now retrieves via BOTH keyword and semantic
search, merged — see ask.py. New `--no-semantic` flag falls back to
Stage 5's keyword-only behavior.
Stage 8: `seschat explain <path> <file>` explains a single indexed file
(summary, responsibilities, key classes/functions, dependencies,
suggestions) using that file's extracted metadata + real source — no
retrieval step, the user names the file directly.
"""

from pathlib import Path

import typer

from seschat.db import get_counts, write_database
from seschat.metadata import write_metadata_index
from seschat.scanner import scan_repository
from seschat.search import IndexNotFoundError, search_repository
from seschat.ask import ask_repository
from seschat.explain import ExplainResult, FileNotIndexedError, explain_file
from seschat.llm import LLMError, LLMNotConfiguredError
from seschat.embeddings import EmbeddingError, EmbeddingNotConfiguredError
from seschat.semantic import EmbeddingsNotFoundError, build_and_store_embeddings, semantic_search

app = typer.Typer(help="Seschat — a repository intelligence tool.")


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to the repository to scan."),
    no_gitignore: bool = typer.Option(
        False,
        "--no-gitignore",
        help="Don't read .gitignore files; only use Seschat's built-in ignore list.",
    ),
    no_metadata: bool = typer.Option(
        False,
        "--no-metadata",
        help="Skip Stage 2 metadata extraction; just do the Stage 1 file count. Implies --no-db and --no-embeddings.",
    ),
    no_db: bool = typer.Option(
        False,
        "--no-db",
        help="Skip writing the SQLite database (.seschat/index.db); implies --no-embeddings. .seschat/index.json is still written.",
    ),
    no_embeddings: bool = typer.Option(
        False,
        "--no-embeddings",
        help="Skip Stage 6 embedding generation. Makes one embedding-backend call per index run when enabled.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show full ignored/unrecognized lists instead of a summary count.",
    ),
):
    """
    Scan a repository, print a summary of the files it contains, and
    (unless disabled) write a structural metadata index to
    .seschat/index.json, a queryable database to .seschat/index.db, and
    semantic embeddings into that same database.
    """
    typer.echo(f"Scanning: {path}")

    try:
        result = scan_repository(
            path,
            respect_gitignore=not no_gitignore,
            extract_file_metadata=not no_metadata,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # ---- Found -----------------------------------------------------------
    if result.language_counts:
        typer.echo("Found:")
        for language, count in sorted(
            result.language_counts.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            noun = "file" if count == 1 else "files"
            typer.echo(f"  • {count} {language} {noun}")
    else:
        typer.echo("Found: (no recognized source files)")

    # ---- Ignored (built-in list + .gitignore) -----------------------------
    total_ignored_dirs = len(result.ignored_dirs_found) + len(result.gitignored_dirs)
    if total_ignored_dirs:
        if verbose:
            typer.echo("Ignored:")
            if result.ignored_dirs_found:
                typer.echo("  Built-in rules:")
                for d in sorted(result.ignored_dirs_found):
                    typer.echo(f"    • {d}")
            if result.gitignored_dirs:
                typer.echo("  From .gitignore:")
                for d in sorted(result.gitignored_dirs):
                    typer.echo(f"    • {d}")
        else:
            noun = "directory" if total_ignored_dirs == 1 else "directories"
            typer.echo(
                f"Ignored: {total_ignored_dirs} {noun} "
                f"({len(result.ignored_dirs_found)} built-in, "
                f"{len(result.gitignored_dirs)} via .gitignore) "
                f"— use --verbose to list"
            )

    if result.gitignored_files:
        if verbose:
            typer.echo("Ignored files (.gitignore):")
            for f in sorted(result.gitignored_files):
                typer.echo(f"  • {f}")
        else:
            n = len(result.gitignored_files)
            typer.echo(
                f"Ignored {n} file{'s' if n != 1 else ''} via .gitignore "
                f"— use --verbose to list"
            )

    # ---- Unrecognized extensions -------------------------------------------
    if result.unrecognized_extensions:
        total_unknown = sum(result.unrecognized_extensions.values())
        if verbose:
            typer.echo("Other extensions seen (not counted):")
            for ext, count in sorted(
                result.unrecognized_extensions.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                typer.echo(f"  • {ext} ({count})")
        else:
            n_types = len(result.unrecognized_extensions)
            typer.echo(
                f"Skipped {total_unknown} file(s) with {n_types} unrecognized "
                f"extension(s) — use --verbose to list"
            )

    if result.gitignore_files_used and verbose:
        typer.echo(f".gitignore files applied: {len(result.gitignore_files_used)}")
        for g in sorted(result.gitignore_files_used):
            typer.echo(f"  • {g}")

    # ---- Stage 2: metadata -------------------------------------------------
    if not no_metadata:
        total_classes = sum(len(m.classes) for m in result.file_metadata)
        total_functions = sum(len(m.functions) for m in result.file_metadata)
        total_imports = sum(len(m.imports) for m in result.file_metadata)
        total_comments = sum(len(m.comments) for m in result.file_metadata)

        typer.echo("Metadata:")
        typer.echo(
            f"  • Parsed {result.parsed_file_count} file(s): "
            f"{total_classes} classes, {total_functions} functions, "
            f"{total_imports} imports, {total_comments} comments"
        )

        if result.not_applicable_file_count:
            typer.echo(
                f"  • {result.not_applicable_file_count} file(s) skipped — not applicable "
                f"(markup/data formats like Markdown, JSON, YAML have no classes/functions/imports)"
            )

        if result.unsupported_file_count:
            typer.echo(
                f"  • {result.unsupported_file_count} file(s) not parsed — language support "
                f"not written yet "
                f"{'(use --verbose for details)' if not verbose else ''}"
            )

        if result.unavailable_file_count:
            typer.echo(
                f"  • {result.unavailable_file_count} file(s) not parsed — tree-sitter isn't "
                f"installed (run `pip install -e .`)"
            )

        if result.error_file_count:
            typer.echo(
                f"  • {result.error_file_count} file(s) failed to parse "
                f"{'(use --verbose for details)' if not verbose else ''}"
            )

        if result.unrecognized_file_count:
            typer.echo(
                f"  • {result.unrecognized_file_count} file(s) skipped — unrecognized "
                f"file type "
                f"{'(use --verbose for details)' if not verbose else ''}"
            )

        if verbose:
            for label, status in (
                ("Not applicable", "not_applicable"),
                ("Unsupported", "unsupported"),
                ("Errors", "error"),
                ("Unrecognized", "unrecognized"),
            ):
                seen: dict[str, str] = {}
                for m in result.file_metadata:
                    if m.status == status and m.language not in seen:
                        seen[m.language] = m.detail
                if seen:
                    typer.echo(f"  {label}:")
                    for lang, detail in sorted(seen.items()):
                        typer.echo(f"    • {lang}: {detail}")

        index_path = write_metadata_index(result.file_metadata, result.root)
        typer.echo(f"  • Wrote index: {index_path}")

        # ---- Stage 3: database ---------------------------------------------
        if not no_db:
            db_path = write_database(result.file_metadata, result.root)
            counts = get_counts(db_path)
            typer.echo("Database:")
            typer.echo(
                f"  • Stored {counts['files']} file(s): "
                f"{counts['classes']} classes, {counts['functions']} functions, "
                f"{counts['imports']} imports, {counts['comments']} comments"
            )
            typer.echo(f"  • Wrote db: {db_path}")

            # ---- Stage 6: embeddings ----------------------------------------
            if not no_embeddings:
                try:
                    embed_result = build_and_store_embeddings(result.file_metadata, result.root, db_path)
                except (EmbeddingNotConfiguredError, EmbeddingError) as e:
                    typer.secho(f"Warning: embeddings not generated — {e}", fg=typer.colors.YELLOW, err=True)
                else:
                    typer.echo("Embeddings:")
                    skip_note = (
                        f" ({embed_result.skipped_count} file(s) had nothing to embed)"
                        if embed_result.skipped_count
                        else ""
                    )
                    typer.echo(
                        f"  • Embedded {embed_result.embedded_count} file(s) via "
                        f"{embed_result.provider}:{embed_result.model}{skip_note}"
                    )


@app.command()
def query(
    path: str = typer.Argument(..., help="Path to the repository (must already be indexed)."),
    term: str = typer.Argument(..., help="Keyword to search for (case-insensitive substring match)."),
    limit: int = typer.Option(
        20, "--limit", "-n", help="Max number of files to show (files with more matches rank higher)."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show every individual match, not just the count, per file."
    ),
):
    """
    Keyword search (Stage 4): checks filenames, class names, function
    names, imports, and comments via plain substring matching. For
    similarity-based search that doesn't require exact word matches, see
    `seschat semantic`.
    """
    root = Path(path).resolve()
    db_path = root / ".seschat" / "index.db"

    try:
        results = search_repository(db_path, term, limit=limit)
    except IndexNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if not results:
        typer.echo(f"No matches for {term!r}.")
        return

    typer.echo(f"Matches for {term!r}:")
    for file_result in results:
        typer.echo(f"  {file_result.path} ({file_result.language}) — {file_result.match_count} match(es)")
        if verbose:
            for m in file_result.matches:
                shown = m.text if len(m.text) <= 80 else m.text[:77] + "..."
                typer.echo(f"    [{m.kind}] {shown}")

    if not verbose:
        typer.echo("  (use --verbose to see individual matches)")


@app.command()
def semantic(
    path: str = typer.Argument(..., help="Path to the repository (must already be indexed with embeddings)."),
    query: str = typer.Argument(..., help="Natural-language query to search for semantically."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max number of files to show."),
):
    """
    Semantic search (Stage 6): rank files by embedding similarity to a
    natural-language query, rather than literal keyword matching — can
    surface a file even if none of the query's words appear in it
    verbatim. Requires `seschat index <path>` to have been run without
    --no-db or --no-embeddings.
    """
    root = Path(path).resolve()
    db_path = root / ".seschat" / "index.db"

    try:
        matches = semantic_search(db_path, query, limit=limit)
    except EmbeddingsNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except (EmbeddingNotConfiguredError, EmbeddingError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if not matches:
        typer.echo("No files in the index.")
        return

    typer.echo(f"Semantic matches for {query!r}:")
    for m in matches:
        typer.echo(f"  {m.path} ({m.language}) — similarity {m.score:.3f}")


@app.command()
def ask(
    path: str = typer.Argument(..., help="Path to the repository (must already be indexed)."),
    question: str = typer.Argument(..., help="A natural-language question about the repository."),
    max_files: int = typer.Option(
        6, "--max-files", "-k", help="Max number of files to retrieve as context for the LLM."
    ),
    model: str = typer.Option(
        None, "--model", help="Override the Ollama chat model (default: $SESCHAT_OLLAMA_MODEL or qwen3:8b)."
    ),
    no_semantic: bool = typer.Option(
        False,
        "--no-semantic",
        help="Retrieve using Stage 4 keyword search only, skipping Stage 6/7 semantic retrieval.",
    ),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the retrieved keywords/files/model before the answer."
    ),
):
    """
    Ask a natural-language question about a previously-indexed repository.
    Stage 7: retrieval merges keyword search (Stage 4) with semantic
    search (Stage 6) into one ranked candidate list before the LLM sees
    anything — the model answers from what retrieval found, it doesn't
    search on its own. Run `seschat index <path>` first if this errors
    out. Falls back to keyword-only retrieval automatically (with a
    note under --show-context) if the repo has no embedding index.
    """
    root = Path(path).resolve()
    db_path = root / ".seschat" / "index.db"

    try:
        result = ask_repository(
            root, db_path, question, max_files=max_files, model=model, use_semantic=not no_semantic
        )
    except IndexNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except (LLMNotConfiguredError, LLMError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if show_context:
        typer.echo(f"Keywords searched: {', '.join(result.keywords) or '(none)'}")
        if result.semantic_note:
            typer.echo(f"Semantic retrieval: {result.semantic_note}")
        if result.contexts:
            typer.echo("Retrieved files:")
            for ctx in result.contexts:
                typer.echo(
                    f"  • {ctx.path} ({ctx.language}) — via {', '.join(ctx.sources)} "
                    f"[kw {ctx.keyword_score:.2f} / sem {ctx.semantic_score:.2f} "
                    f"/ combined {ctx.combined_score:.2f}]"
                )
        else:
            typer.echo("Retrieved files: (none matched)")
        typer.echo(f"Model: {result.model}")
        typer.echo("")

    typer.echo(result.answer)


@app.command()
def explain(
    path: str = typer.Argument(..., help="Path to the repository (must already be indexed)."),
    file: str = typer.Argument(..., help="Path to the file to explain, relative to the repo root (or absolute)."),
    model: str = typer.Option(
        None, "--model", help="Override the Ollama chat model (default: $SESCHAT_OLLAMA_MODEL or qwen3:8b)."
    ),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the file's extracted metadata before the explanation."
    ),
):
    """
    Explain a single indexed file (Stage 8): summary, responsibilities,
    key classes/functions, dependencies, and suggestions, generated
    from that file's extracted metadata (Stage 2/3) and real source —
    there's no search step, the LLM just reads what's already indexed
    for the file you name. Run `seschat index <path>` first if this
    errors out.
    """
    root = Path(path).resolve()
    db_path = root / ".seschat" / "index.db"

    try:
        result = explain_file(root, db_path, file, model=model)
    except IndexNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except FileNotIndexedError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except (LLMNotConfiguredError, LLMError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if show_context:
        typer.echo(f"File: {result.path} ({result.language})")
        typer.echo(f"Status: {result.status}")
        if result.classes:
            typer.echo(f"Classes: {', '.join(result.classes)}")
        if result.functions:
            typer.echo(f"Functions: {', '.join(result.functions)}")
        if result.imports:
            typer.echo(f"Imports: {', '.join(result.imports)}")
        if result.comments:
            typer.echo(f"Comments: {', '.join(result.comments)}")
        if result.read_error:
            typer.echo(f"Source: unavailable ({result.read_error})")
        elif result.source_truncated:
            typer.echo("Source: read (truncated)")
        typer.echo(f"Model: {result.model}")
        typer.echo("")

    typer.echo(result.explanation)


def main():
    app()


if __name__ == "__main__":
    main()
