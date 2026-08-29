# Configuration reference

Config file planned for Stage 13. Everything is environment
variables and CLI flags.

## Environment variables

| Variable | Purpose | Used by |
|---|---|---|
| `OLLAMA_HOST` | Override the local Ollama server address (default `http://localhost:11434`) | `ask`, `explain`, `semantic`, `index` |
| `SESCHAT_OLLAMA_MODEL` | Override the Ollama **chat** model tag (default `qwen3:8b`) | `ask`, `explain` |
| `SESCHAT_OLLAMA_EMBED_MODEL` | Override the Ollama **embedding** model tag (default `nomic-embed-text`) | `semantic`, `index` |
| `GEMINI_API_KEY` | Required for the Gemini fallback (chat and embeddings) | `ask`, `explain`, `semantic`, `index` |
| `SESCHAT_GEMINI_MODEL` | Override the Gemini **chat** model (default `gemini-3.6-flash`) | `ask`, `explain` |
| `SESCHAT_GEMINI_EMBED_MODEL` | Override the Gemini **embedding** model (default `gemini-embedding-001`) | `semantic`, `index` |

## CLI flags by command

### `seschat index <path>`

- `--no-gitignore` — ignore only Seschat's built-in directory list, skip `.gitignore` files.
- `--no-metadata` — skip tree-sitter extraction, just count files. Implies `--no-db --no-embeddings`.
- `--no-db` — skip writing `.seschat/index.db`. Implies `--no-embeddings`.
- `--no-embeddings` — skip the embedding step (one backend call per index run when enabled).
- `--verbose` / `-v` — show full ignored/unrecognized lists instead of summary counts.

### `seschat query <path> <term>`

- `--limit` / `-n` — max files shown (default 20).
- `--verbose` / `-v` — show every individual match, not just per-file counts.

### `seschat semantic <path> <query>`

- `--limit` / `-n` — max files shown (default 10).

### `seschat ask <path> "<question>"`

- `--max-files` / `-k` — max files retrieved as LLM context (default 6).
- `--no-semantic` — keyword-only retrieval, skipping the semantic half entirely.
- `--model` — override the Ollama chat model for this call.
- `--show-context` — print keywords, retrieval provenance/scores per file, and which backend answered, before the answer.

### `seschat explain <path> <file>`

- `--model` — override the Ollama chat model for this call.
- `--show-context` — print the file's extracted metadata and which backend answered, before the explanation.

## Running without installing

```bash
python -m seschat.cli index ./my-project
python -m seschat.cli query ./my-project Cache
python -m seschat.cli semantic ./my-project "how does login work"
python -m seschat.cli ask ./my-project "Where is authentication?"
python -m seschat.cli explain ./my-project src/cache.cpp
```
