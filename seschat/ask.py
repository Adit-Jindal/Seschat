"""
ask.py — Stage 5 + Stage 7 of Seschat: `seschat ask <path> "<question>"`.

Stage 5 pipeline (roadmap.md's sketch, implemented literally):

    question
      -> search repository         (search.py — plain SQL, no AI)
      -> collect relevant files    (this module)
      -> construct prompt          (this module)
      -> LLM                       (llm.py)
      -> answer

Stage 7 changes the *retrieval* step only: instead of keyword search
alone, `ask` now runs BOTH Stage 4's keyword search and Stage 6's
semantic (embedding) search and merges them into one ranked candidate
list — roadmap.md's Stage 7 pipeline:

    question -> embedding -> vector search -> relevant chunks
             -> (also) keyword search
             -> merged, ranked candidates -> prompt builder -> LLM -> answer

Retrieval still happens entirely outside the model. The LLM never
decides what to look at — it only reads what Python already selected,
now by two complementary signals instead of one: keyword search catches
exact identifier/name matches; semantic search catches conceptually
related files that don't share any words with the question. If the
repo has no embedding index (or the embedding backend isn't reachable),
retrieval degrades to keyword-only and that degradation is reported via
AskResult.semantic_note rather than failing silently or hard-erroring —
`ask` should still work on a repo indexed with `--no-embeddings`.

"Chunks" in the roadmap sketch are still whole files, not sub-file
spans — same file-level granularity Stage 6 already documented as a
deliberate simplification (metadata.py doesn't track byte ranges yet).
Stage 7 improves *which* files get retrieved, not the granularity of
what's retrieved; per-chunk embeddings remain a Stage 6/7 follow-up
noted in the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from seschat.embeddings import EmbeddingError, EmbeddingNotConfiguredError
from seschat.llm import call_llm
from seschat.search import IndexNotFoundError, Match, get_file_structure, search_repository
from seschat.semantic import EmbeddingsNotFoundError, semantic_search

__all__ = [
    "extract_keywords",
    "collect_relevant_files",
    "retrieve_relevant_files",
    "build_file_contexts",
    "build_prompt",
    "ask_repository",
    "AskResult",
    "FileContext",
    "RankedFile",
    "RetrievedFile",
    "IndexNotFoundError",
]

# --------------------------------------------------------------------------
# Step 1: turn a natural-language question into search keywords.
# --------------------------------------------------------------------------

# Small stopword list — just enough to filter out the words that show up
# in almost every question ("where", "is", "the", ...) and would otherwise
# get searched as if they were meaningful repo terms. Not linguistically
# rigorous; good enough for Stage 5's keyword extraction. Real ranking
# (TF-IDF, etc.) is still deferred, same as Stage 4.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "where", "what", "when", "who", "how", "why", "which", "does", "do",
    "did", "can", "could", "would", "should", "will", "shall",
    "this", "that", "these", "those", "there", "here",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "it", "its", "i", "we", "you",
    "my", "our", "me", "us", "about", "into", "if", "so", "than",
}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_keywords(question: str) -> list[str]:
    """
    Pull search-worthy tokens out of a natural-language question.

    Deliberately simple (split on word boundaries, drop stopwords and
    very short tokens, de-duplicate preserving order) rather than real
    NLP — this is still "no AI" territory per roadmap.md; the model only
    gets involved *after* retrieval. Identifiers like `CamelCase` or
    `snake_case` survive intact since they're exactly the kind of term
    likely to match a class/function name.
    """
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(question):
        if word.lower() in _STOPWORDS or len(word) < 3:
            continue
        seen.setdefault(word, None)
    return list(seen.keys())


# --------------------------------------------------------------------------
# Step 2a: Stage 4 keyword retrieval (unchanged logic, still used
# standalone by --no-semantic and as one half of Stage 7's merge).
# --------------------------------------------------------------------------

@dataclass
class RankedFile:
    """One file's aggregated keyword relevance across every keyword searched."""
    path: str
    language: str
    total_matches: int = 0
    matched_keywords: set[str] = field(default_factory=set)
    per_keyword_results: list = field(default_factory=list)  # list[FileResult]


def collect_relevant_files(
    db_path: Path, question: str, max_files: int = 6
) -> tuple[list[str], list[RankedFile]]:
    """
    Run Stage 4's search_repository() once per extracted keyword and merge
    the results: a file's overall rank is the sum of its match counts
    across every keyword that hit it, so a file matching several
    question-keywords outranks one matching only one. Raises
    IndexNotFoundError (from search.py) if the repo hasn't been indexed —
    checked on the first search call, so an un-indexed repo fails fast
    even if the question has several keywords.
    """
    keywords = extract_keywords(question)
    ranked: dict[str, RankedFile] = {}

    for keyword in keywords:
        for file_result in search_repository(db_path, keyword, limit=50):
            entry = ranked.setdefault(
                file_result.path,
                RankedFile(path=file_result.path, language=file_result.language),
            )
            entry.total_matches += file_result.match_count
            entry.matched_keywords.add(keyword)
            entry.per_keyword_results.append(file_result)

    ordered = sorted(ranked.values(), key=lambda r: (-r.total_matches, r.path))
    return keywords, ordered[:max_files]


# --------------------------------------------------------------------------
# Step 2b: Stage 7 — merge keyword ranking with Stage 6 semantic ranking.
# --------------------------------------------------------------------------

_KEYWORD_WEIGHT = 0.5
_SEMANTIC_WEIGHT = 0.5
_CANDIDATE_MULTIPLIER = 3  # pull more candidates than max_files from each
                            # signal before merging, so a file that's #2 on
                            # keywords and #2 on semantics but never #1 on
                            # either still has a shot at the final top-k.


@dataclass
class RetrievedFile:
    """
    One file's merged relevance, Stage 7. Unlike RankedFile (keyword-only),
    this tracks a score per retrieval signal plus which signal(s) actually
    found it, so --show-context can explain *why* a file was retrieved.
    """
    path: str
    language: str
    keyword_score: float = 0.0     # normalized 0-1: this file's matches / top file's matches
    semantic_score: float = 0.0    # cosine similarity, already ~0-1
    combined_score: float = 0.0
    matched_keywords: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)  # {"keyword"} / {"semantic"} / both


def _normalized_keyword_scores(ranked: list[RankedFile]) -> dict[str, float]:
    """Scale raw match counts to 0-1 so they're comparable to cosine similarity."""
    if not ranked:
        return {}
    top = max(r.total_matches for r in ranked) or 1
    return {r.path: r.total_matches / top for r in ranked}


def retrieve_relevant_files(
    db_path: Path,
    question: str,
    max_files: int = 6,
    use_semantic: bool = True,
) -> tuple[list[str], list[RetrievedFile], str | None]:
    """
    Stage 7's hybrid retrieval: run keyword search (Stage 4) and, unless
    disabled or unavailable, semantic search (Stage 6) over the same
    question, normalize each signal to 0-1, and merge into one ranked
    candidate list via a simple weighted sum
    (0.5 * keyword_score + 0.5 * semantic_score).

    Returns (keywords_searched, retrieved_files, semantic_note).
    `semantic_note` is None when semantic search ran and contributed
    normally; otherwise it's a short human-readable explanation of why
    it didn't (disabled via --no-semantic, no embedding index yet, or
    the embedding backend that built the index isn't reachable right
    now) — surfaced by `--show-context` so a keyword-only fallback is
    visible rather than silent.

    Raises IndexNotFoundError if the repo hasn't been indexed at all
    (propagated from collect_relevant_files/search_repository, checked
    before semantic search runs — there's no point calling an embedding
    backend against a database that doesn't exist).
    """
    keywords, keyword_ranked = collect_relevant_files(
        db_path, question, max_files=max_files * _CANDIDATE_MULTIPLIER
    )
    kw_scores = _normalized_keyword_scores(keyword_ranked)

    combined: dict[str, RetrievedFile] = {}
    for rf in keyword_ranked:
        entry = combined.setdefault(rf.path, RetrievedFile(path=rf.path, language=rf.language))
        entry.keyword_score = kw_scores.get(rf.path, 0.0)
        entry.matched_keywords = set(rf.matched_keywords)
        entry.sources.add("keyword")

    semantic_note: str | None = None
    if not use_semantic:
        semantic_note = "semantic search disabled (--no-semantic)"
    else:
        try:
            sem_matches = semantic_search(db_path, question, limit=max_files * _CANDIDATE_MULTIPLIER)
        except EmbeddingsNotFoundError as e:
            semantic_note = f"semantic search skipped — {e}"
        except (EmbeddingNotConfiguredError, EmbeddingError) as e:
            semantic_note = f"semantic search unavailable — {e}"
        else:
            for m in sem_matches:
                entry = combined.setdefault(m.path, RetrievedFile(path=m.path, language=m.language))
                entry.semantic_score = max(entry.semantic_score, m.score)
                entry.sources.add("semantic")

    for entry in combined.values():
        entry.combined_score = (
            _KEYWORD_WEIGHT * entry.keyword_score + _SEMANTIC_WEIGHT * entry.semantic_score
        )

    ordered = sorted(combined.values(), key=lambda r: (-r.combined_score, r.path))
    return keywords, ordered[:max_files], semantic_note


# --------------------------------------------------------------------------
# Step 3: attach real source excerpts + full structure, construct the prompt.
# --------------------------------------------------------------------------

MAX_CHARS_PER_FILE = 3000  # rough cap so one huge file can't crowd out the rest


@dataclass
class FileContext:
    path: str
    language: str
    classes: list[str]
    functions: list[str]
    imports: list[str]
    comments: list[str]
    source_excerpt: str | None
    source_truncated: bool
    read_error: str | None = None
    # Stage 7 additions — how this file was retrieved, for transparency.
    sources: list[str] = field(default_factory=list)
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0


def _load_source_excerpt(root: Path, rel_path: str) -> tuple[str | None, bool, str | None]:
    """
    Best-effort read of a file's actual source, for prompt grounding. The
    DB only stores *extracted* structure (Stage 2/3), never raw file
    bytes, so getting real source means reading it fresh off disk —
    which also means this can fail if the file moved/was deleted since
    the last `seschat index` run; that's reported, not raised.
    """
    full = root / rel_path
    try:
        text = full.read_text(errors="replace")
    except OSError as e:
        return None, False, f"could not read file: {e}"

    if len(text) > MAX_CHARS_PER_FILE:
        return text[:MAX_CHARS_PER_FILE], True, None
    return text, False, None


def build_file_contexts(
    root: Path, db_path: Path, retrieved_files: list[RetrievedFile]
) -> list[FileContext]:
    """
    For each retrieved file, look up its FULL extracted structure via
    search.get_file_structure() — a change from Stage 5, which only
    reused whatever partial structure happened to match the search
    keywords. That worked when retrieval was keyword-only (a matched
    class name implied that class was relevant), but semantic-only
    files have no matched names to reuse at all, so Stage 7 always does
    the direct-by-path lookup for every file, keyword-matched or not.
    Source excerpts are still read fresh off disk, same as Stage 5.
    """
    contexts = []
    for rf in retrieved_files:
        classes, functions, imports, comments = get_file_structure(db_path, rf.path)
        source, truncated, err = _load_source_excerpt(root, rf.path)
        contexts.append(
            FileContext(
                path=rf.path,
                language=rf.language,
                classes=classes,
                functions=functions,
                imports=imports,
                comments=comments,
                source_excerpt=source,
                source_truncated=truncated,
                read_error=err,
                sources=sorted(rf.sources),
                keyword_score=rf.keyword_score,
                semantic_score=rf.semantic_score,
                combined_score=rf.combined_score,
            )
        )
    return contexts


SYSTEM_PROMPT = (
    "You are a repository question-answering assistant. Answer the "
    "question using ONLY the file excerpts and extracted metadata "
    "provided below — they were retrieved by keyword and/or semantic "
    "search over the repository, not chosen by you. Each file notes "
    "which retrieval method(s) found it; a file found only by semantic "
    "search may be conceptually related even if it shares no words with "
    "the question. If the provided context doesn't contain enough "
    "information to answer confidently, say so plainly instead of "
    "guessing. Cite specific file paths in your answer when you "
    "reference something. Be concise."
)


def build_prompt(question: str, keywords: list[str], contexts: list[FileContext]) -> str:
    """
    Assemble the final user-turn prompt: the question, which keywords
    were searched, then the retrieved file context — in that order, so
    the model sees the question again right before it starts reading.
    Each file section now also states how it was retrieved (Stage 7),
    so the model can weigh an exact keyword hit differently from a
    semantic-only match if that matters to the answer.
    """
    if not contexts:
        return (
            f"Question: {question}\n\n"
            f"Keywords searched: {', '.join(keywords) or '(none extracted)'}\n\n"
            "No files in the repository index matched any of these keywords "
            "or were found via semantic search. Tell the user plainly that "
            "retrieval found nothing relevant, rather than guessing at an "
            "answer."
        )

    sections = [f"Question: {question}", f"Keywords searched: {', '.join(keywords)}", ""]
    for ctx in contexts:
        sections.append(f"--- {ctx.path} ({ctx.language}) ---")
        sections.append(
            f"Retrieved via: {', '.join(ctx.sources)} "
            f"(keyword score {ctx.keyword_score:.2f}, semantic score {ctx.semantic_score:.2f})"
        )
        if ctx.classes:
            sections.append(f"Classes: {', '.join(ctx.classes)}")
        if ctx.functions:
            sections.append(f"Functions: {', '.join(ctx.functions)}")
        if ctx.imports:
            sections.append(f"Imports: {', '.join(ctx.imports)}")
        if ctx.comments:
            sections.append(f"Comments: {' | '.join(ctx.comments)}")
        if ctx.source_excerpt is not None:
            note = " (truncated)" if ctx.source_truncated else ""
            sections.append(f"Source{note}:\n{ctx.source_excerpt}")
        elif ctx.read_error:
            sections.append(f"[source unavailable: {ctx.read_error}]")
        sections.append("")

    return "\n".join(sections)


# --------------------------------------------------------------------------
# Step 4: tie it all together.
# --------------------------------------------------------------------------

@dataclass
class AskResult:
    answer: str
    question: str
    keywords: list[str]
    contexts: list[FileContext]
    model: str
    semantic_note: str | None = None


def ask_repository(
    root: Path,
    db_path: Path,
    question: str,
    max_files: int = 6,
    model: str | None = None,
    use_semantic: bool = True,
) -> AskResult:
    """
    Run the full Stage 7 pipeline: hybrid retrieval (keyword + semantic)
    -> collect files -> build prompt -> call the LLM -> return the
    answer plus everything that fed it (so the CLI can optionally show
    its work with --show-context).

    Raises IndexNotFoundError if the repo hasn't been indexed
    (propagated from retrieve_relevant_files), and LLMError/
    LLMNotConfiguredError (from llm.py) if the model call itself can't
    happen. A missing/unreachable *embedding* backend is NOT raised
    here — it's reported via AskResult.semantic_note and retrieval
    degrades to keyword-only, since `ask` should still work on a repo
    indexed with --no-embeddings.
    """
    keywords, retrieved_files, semantic_note = retrieve_relevant_files(
        db_path, question, max_files=max_files, use_semantic=use_semantic
    )
    contexts = build_file_contexts(root, db_path, retrieved_files)
    prompt = build_prompt(question, keywords, contexts)

    response = call_llm(prompt, system=SYSTEM_PROMPT, model=model)

    return AskResult(
        answer=response.text,
        question=question,
        keywords=keywords,
        contexts=contexts,
        model=f'{response.provider}:{response.model}',
        semantic_note=semantic_note,
    )
