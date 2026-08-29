# Changelog

All notable changes to Seschat are recorded here, stage by stage
(see `roadmap.md` for what each stage means). Dates are approximate to
when the stage was completed during development, not necessarily the
PyPI release date.

## [Unreleased]

- Stage 9–12 (compiler-error assistant, verified test generation,
  PR assistant, architecture diagrams) — not started.
- Chunk-level (function/class-level) embeddings — deferred, documented
  as safe to retrofit later; audited going into Stage 8.
- Incremental indexing (currently drop-and-rebuild every `seschat index`
  run) — planned for Stage 13.

## [0.1.0b1] — Beta

First public beta. Stages 1–8 of the roadmap:

- **Stage 1 — Scanner.** `seschat index` walks a repository, respecting
  a built-in ignore list plus scoped `.gitignore` rules, and tallies
  source files by language.
- **Stage 2 — Metadata extraction.** Every recognized source file is
  parsed with tree-sitter to extract its classes, functions, imports,
  and comments — real AST parsing, not regex.
- **Stage 3 — SQLite persistence.** Extracted metadata is written to a
  queryable `.seschat/index.db` (five tables: files, classes, functions,
  imports, comments), in addition to `.seschat/index.json`.
- **Stage 4 — Keyword search.** `seschat query` does substring search
  across extracted metadata fields, no AI involved.
- **Stage 5 — First LLM integration.** `seschat ask` answers
  natural-language questions using keyword retrieval + an LLM
  (Ollama local-first, automatic Gemini fallback).
- **Stage 6 — Embeddings.** `seschat semantic` ranks files by meaning,
  not literal word match, via an embedding index built into
  `.seschat/index.db`.
- **Stage 7 — Hybrid RAG.** `seschat ask` now merges keyword and
  semantic retrieval into one ranked candidate list before the LLM
  ever sees anything.
- **Stage 8 — Explain code.** `seschat explain <file>` summarizes a
  single indexed file: purpose, responsibilities, key classes/
  functions, dependencies, and improvement suggestions.

### Known limitations (see README "Status" section)

- Embeddings are per-file, not per-function/class.
- Indexing is drop-and-rebuild, not incremental.
- The two chat/embedding LLM backends (Ollama, Gemini) have been
  verified against hand-built fixtures but not exhaustively against
  live servers in every environment — see README for what to smoke-test
  after installing.
