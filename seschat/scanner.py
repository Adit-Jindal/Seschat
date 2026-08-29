"""
scanner.py — Stage 1 + Stage 2 of Seschat: repository scanning and, for
each recognized source file, structural metadata extraction.

Stage 1 (file counting, ignore rules) is unchanged in spirit; Stage 2
just adds a call to metadata.extract_metadata() for every counted file.
"""

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from seschat.gitignore import GitignoreRule, is_ignored, parse_gitignore
from seschat.metadata import FileMetadata, extract_metadata, make_unrecognized_metadata


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Directories we always ignore, regardless of .gitignore. Matched by exact
# folder name, anywhere in the tree.
IGNORED_DIRS = {
    ".git",
    "node_modules",
    "build",
    "dist",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".idea",
    ".vscode",
    "target",       # Rust/Java build output
    "egg-info",
    ".egg-info",
    ".seschat",      # Seschat's own metadata output — never scan our own index
}

# Extension -> language/file-type label.
# This is the single place to add support for a new language later.
EXTENSION_MAP = {
    ".py": "Python",
    ".ipynb": "Jupyter Notebook",
    ".c": "C",
    ".h": "C Header (or C++)",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".java": "Java",
    ".kt": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".js": "JavaScript",
    ".jsx": "JavaScript (JSX)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (TSX)",
    ".swift": "Swift",
    ".m": "Objective-C",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
}


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class ScanResult:
    root: Path
    language_counts: Counter = field(default_factory=Counter)
    total_files_scanned: int = 0

    # Built-in ignore list (name-based, e.g. ".git", "node_modules")
    ignored_dirs_found: set[str] = field(default_factory=set)

    # .gitignore-based ignores — stored as paths (relative to repo root)
    # since the same *name* can be ignored in one folder and not another.
    gitignored_dirs: set[str] = field(default_factory=set)
    gitignored_files: set[str] = field(default_factory=set)
    gitignore_files_used: list[str] = field(default_factory=list)

    unrecognized_extensions: Counter = field(default_factory=Counter)

    # Stage 2: one FileMetadata per counted source file (same order as
    # they were encountered during the walk).
    file_metadata: list[FileMetadata] = field(default_factory=list)

    @property
    def total_source_files(self) -> int:
        return sum(self.language_counts.values())

    @property
    def parsed_file_count(self) -> int:
        return sum(1 for m in self.file_metadata if m.status == "parsed")

    @property
    def not_applicable_file_count(self) -> int:
        return sum(1 for m in self.file_metadata if m.status == "not_applicable")

    @property
    def unsupported_file_count(self) -> int:
        return sum(1 for m in self.file_metadata if m.status == "unsupported")

    @property
    def unavailable_file_count(self) -> int:
        return sum(1 for m in self.file_metadata if m.status == "unavailable")

    @property
    def error_file_count(self) -> int:
        return sum(1 for m in self.file_metadata if m.status == "error")

    @property
    def unrecognized_file_count(self) -> int:
        return sum(1 for m in self.file_metadata if m.status == "unrecognized")


# --------------------------------------------------------------------------
# Core scanning function
# --------------------------------------------------------------------------

def scan_repository(
    root: str | Path,
    respect_gitignore: bool = True,
    extract_file_metadata: bool = True,
) -> ScanResult:
    """
    Recursively walk `root`, skipping ignored directories, and tally
    source files by language based on file extension.

    If `respect_gitignore` is True (default), any .gitignore files found
    are parsed and applied *scoped to the directory they live in* — a
    .gitignore in a subfolder never affects files outside that subfolder,
    matching git's own behavior.

    If `extract_file_metadata` is True (default), each recognized source
    file is also parsed with tree-sitter to pull out its classes,
    functions, imports, and comments (see metadata.py). Set this to False
    to get Stage 1's fast, metadata-free counting behavior.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    result = ScanResult(root=root)

    # effective_rules[dir] = the full ordered rule list that applies inside
    # `dir`: every ancestor .gitignore's rules, followed by dir's own.
    effective_rules: dict[Path, list[GitignoreRule]] = {}

    if respect_gitignore:
        root_gitignore = root / ".gitignore"
        if root_gitignore.is_file():
            effective_rules[root] = parse_gitignore(root_gitignore)
            result.gitignore_files_used.append(str(root_gitignore))
        else:
            effective_rules[root] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        current_rules = effective_rules.pop(dirpath_p, [])

        # Prune ignored directories IN PLACE so os.walk never descends
        # into them. This is more efficient than walking in and discarding.
        kept_dirs = []
        for d in dirnames:
            full = dirpath_p / d

            if d in IGNORED_DIRS or d.endswith(".egg-info"):
                result.ignored_dirs_found.add(d)
                continue

            if respect_gitignore and is_ignored(full, True, current_rules):
                result.gitignored_dirs.add(str(full.relative_to(root)))
                continue

            kept_dirs.append(d)

            # Compute the rule set that will apply *inside* this child dir:
            # everything active here, plus its own .gitignore if it has one.
            child_rules = current_rules
            if respect_gitignore:
                child_gitignore = full / ".gitignore"
                if child_gitignore.is_file():
                    own_rules = parse_gitignore(child_gitignore)
                    if own_rules:
                        child_rules = current_rules + own_rules
                        result.gitignore_files_used.append(str(child_gitignore))
            effective_rules[full] = child_rules

        dirnames[:] = kept_dirs

        for filename in filenames:
            full = dirpath_p / filename

            if respect_gitignore and is_ignored(full, False, current_rules):
                result.gitignored_files.add(str(full.relative_to(root)))
                continue

            result.total_files_scanned += 1
            ext = Path(filename).suffix.lower()

            if ext in EXTENSION_MAP:
                language = EXTENSION_MAP[ext]
                result.language_counts[language] += 1

                if extract_file_metadata:
                    rel = full.relative_to(root)
                    try:
                        source_bytes = full.read_bytes()
                    except OSError as e:
                        result.file_metadata.append(
                            FileMetadata(
                                path=str(rel),
                                language=language,
                                status="error",
                                detail=f"could not read file: {e}",
                            )
                        )
                    else:
                        meta = extract_metadata(rel, language, source_bytes)
                        result.file_metadata.append(meta)

            else:
                if ext:
                    result.unrecognized_extensions[ext] += 1
                # Files with no extension (Dockerfile, Makefile, LICENSE, ...)
                # aren't tallied in unrecognized_extensions (there's no
                # extension to tally), but still get a metadata fallback
                # entry below so they aren't silently missing from the index.

                if extract_file_metadata:
                    rel = full.relative_to(root)
                    result.file_metadata.append(make_unrecognized_metadata(rel, ext))

    return result
