# Seschat

A CLI that actually understands your codebase. It parses every file into a
real syntax tree (via tree-sitter, not regex), indexes the result in
SQLite, and lets you search, semantically browse, and ask questions about
your repository — grounded in what's actually in the code, not a model's
best guess.

> **Beta.** Seschat works and is actively maintained, but it's early —
> expect rough edges, and expect the internals to keep moving for a while.
> See [Known limitations](#known-limitations) below before you point it at
> anything you depend on. Issues and PRs are welcome.

```bash
seschat index ./my-project
seschat ask ./my-project "Where is authentication handled?"
seschat explain ./my-project src/auth.py
```

## Why

Most "AI for your codebase" tools quietly let the model do the searching —
you ask a question, the model reads what it can fit in context, and hopes
for the best. Seschat splits that apart on purpose: retrieval (keyword
search, semantic search, structural lookups) happens entirely in Python
and SQL, *before* the model ever sees a token. The model's only job is to
read what retrieval already found and write a grounded answer. If you turn
on `--show-context`, you can see exactly which files were retrieved, by
which method, and why — nothing about what reaches the model is hidden.

## What it does

| Command | What it does | Talks to a model? |
|---|---|---|
| `seschat index <path>` | Scans a repo, parses every file's structure (classes, functions, imports, comments) with tree-sitter, stores it in SQLite | No |
| `seschat query <path> <term>` | Keyword search over that extracted structure | No |
| `seschat semantic <path> <query>` | Embedding-based similarity search — finds related files even with no shared vocabulary | Embeddings only |
| `seschat ask <path> "<question>"` | Hybrid keyword + semantic retrieval, then an LLM answers and cites files | Yes |
| `seschat explain <path> <file>` | Summarizes one file: purpose, responsibilities, key classes/functions, dependencies, suggestions | Yes |

Structural parsing currently covers Python, C, C++, Java, Go, Rust, Ruby,
JavaScript/JSX, TypeScript/TSX, C#, PHP, Kotlin, and Scala. Markdown,
JSON, YAML, TOML, XML, HTML, CSS, and SCSS are indexed and searchable but
correctly reported as having no classes or functions — they're not code.
Anything else still gets counted and flagged rather than silently
dropped, so `seschat index` never leaves a file unaccounted for.

## Install

```bash
pip install seschat
```
Requires Python 3.10+.

## Quick start

```bash
seschat index ./my-project
seschat query ./my-project Cache
seschat semantic ./my-project "how does the app handle user login"
seschat ask ./my-project "Where is authentication handled?"
seschat explain ./my-project src/auth.py
```

`index`, `query`, and `semantic` work with nothing beyond `pip install` —
`semantic` needs an embedding backend, covered below. `ask` and `explain`
need a chat model.

## Setting up a model backend

`ask` and `explain` are local-first with an automatic hosted fallback —
there's nothing you *must* configure, but pick one:

**Local, no API key (Ollama):**
```bash
ollama serve
ollama pull qwen3:8b
```

**Hosted (Gemini):**
```bash
export GEMINI_API_KEY=...   # https://aistudio.google.com/
```

Seschat tries Ollama first and falls back to Gemini automatically if it
isn't running, isn't installed, or doesn't have the model pulled. If
neither is usable, the error message tells you exactly what to fix rather
than a raw traceback.

### Embeddings (for `semantic`, and the semantic half of `ask`)

Same shape, separate model:

```bash
ollama pull nomic-embed-text     # local
# or reuse the same GEMINI_API_KEY above for hosted embeddings
```

If no embedding backend is available when you index, indexing still
succeeds — you just won't have `seschat semantic`, and `ask` quietly
falls back to keyword-only retrieval (visible under `--show-context`,
never silent).

Full environment variable reference is in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## How retrieval actually works

- **`query`** — plain SQL `LIKE` across extracted file paths, class
  names, function names, imports, and comments. It doesn't see inside
  function bodies or string literals; it only searches what got pulled
  *out* of the file during indexing.
- **`semantic`** — one embedding vector per file (its structure plus a
  source excerpt), ranked by cosine similarity against your query.
- **`ask`** — runs both of the above, normalizes each to a 0–1 score, and
  merges them with a weighted sum. Every retrieved file's provenance and
  score is inspectable via `--show-context`.
- **`explain`** — no retrieval at all. You name the file; it reads that
  file's indexed structure and a fresh copy of its source, and asks the
  model to summarize only that.

## About the name

Seschat is Seshat, morphed with chat. Seshat was the
ancient Egyptian goddess of writing, measurement, and record-keeping —
credited with inventing writing itself and keeping the pharaoh's library.
A tool that reads your code, keeps a structured record of it, and lets
you talk to that record felt like a reasonable namesake. Say it out loud
and it should land closer to "seshat" than "session-chat."

## Known limitations

- **Embeddings are per file, not per function or class.** A large file's
  embedding can dilute a small relevant section inside it. Chunk-level
  embeddings are on the roadmap; they need byte-range tracking in the
  metadata layer first.
- **Indexing is drop-and-rebuild, not incremental.** Every `seschat
  index` run re-scans and re-embeds the whole repository. Fine for
  small-to-medium repos; slower than it needs to be on large ones you
  re-index often.
- **Keyword ranking is match-count, not TF-IDF.** It's a cheap proxy for
  relevance, not real ranking.
- **`.gitignore` support is close, not exact.** Rules are matched with
  `fnmatch`, scoped per directory the way git actually applies them, but
  it doesn't reproduce git's `**` wildcard semantics precisely.
- **Test your own model setup after installing.** The retrieval and
  prompting logic is well-tested; the two live backends (Ollama, Gemini)
  will behave slightly differently depending on your local setup. Run
  `ask`/`explain`/`semantic` once against a real repo after install to
  confirm your backend is actually reachable, and once with it
  deliberately down to confirm the fallback kicks in.

## Project layout

```
seschat/
├── cli.py         # index, query, semantic, ask, explain
├── scanner.py     # traversal, ignore rules, extension → language mapping
├── gitignore.py   # scoped .gitignore parsing
├── metadata.py    # tree-sitter parsing → classes/functions/imports/comments
├── db.py          # SQLite schema and writes (metadata + embeddings)
├── search.py      # keyword search plus by-path structure/record lookups
├── ask.py         # hybrid retrieval → prompt → LLM → answer
├── explain.py     # file path → metadata + source → prompt → LLM → explanation
├── llm.py         # the only module that talks to a chat model
└── embeddings.py  # the only module that talks to an embedding model
```

Every module besides `cli.py` has no idea a CLI framework exists — `cli.py`
is a thin layer on top of plain function calls.

## What's next

A compiler/test-failure assistant, verified test generation, a PR-review
assistant, architecture diagram generation, and general hardening
(incremental indexing, a config file, caching) are planned. The full
history and reasoning behind each stage of this project so far is in
[`roadmap.md`](roadmap.md), kept around because the "why" behind a
decision is usually more useful later than the decision itself.

## Contributing

Issues and PRs are welcome. This is under active development, so expect
things to move between releases — check `CHANGELOG.md` for what changed
and why.

## License

MIT — see [`LICENSE`](LICENSE).
