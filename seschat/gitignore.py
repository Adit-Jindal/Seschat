"""
gitignore.py — minimal, *scoped* .gitignore support for the Seschat scanner.

Important: .gitignore rules are NOT global. A .gitignore file only affects
the directory it lives in, plus everything below it — and a rule defined
deeper in the tree can override (or un-ignore, via `!`) a rule from a
parent .gitignore. This module models that scoping explicitly via
`base_dir` on each rule, rather than treating patterns as one flat list.

This is intentionally a lightweight implementation, not a full
re-implementation of git's wildmatch algorithm. It supports the patterns
people actually write 95% of the time:
    build/          -> directory-only match
    *.pyc           -> glob match on the basename, any depth
    /dist           -> anchored to the .gitignore's own directory
    !important.log  -> negation (un-ignores a previously-matched path)

It does NOT implement double-star (`**`) semantics precisely, character
classes edge cases, or a few other git wildmatch subtleties. Good enough
for Stage 1; worth revisiting if it starts giving wrong answers on real
repos.
"""

import fnmatch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitignoreRule:
    pattern: str        # pattern text, leading/trailing slashes stripped
    negation: bool       # True if the line started with '!'
    dir_only: bool       # True if the line ended with '/'
    anchored: bool       # True if the pattern contains a '/' (not just trailing)
    base_dir: Path        # directory the owning .gitignore lives in


def parse_gitignore(gitignore_path: Path) -> list[GitignoreRule]:
    """
    Parse a single .gitignore file into a list of rules scoped to its
    containing directory. Returns an empty list if the file can't be read.
    """
    try:
        text = gitignore_path.read_text(errors="ignore")
    except OSError:
        return []

    base_dir = gitignore_path.parent
    rules: list[GitignoreRule] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        negation = line.startswith("!")
        if negation:
            line = line[1:]

        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]

        if not line:
            continue

        # A pattern is "anchored" (relative to base_dir) if it contains a
        # slash anywhere except a single trailing one we already stripped.
        anchored = "/" in line

        pattern = line.lstrip("/")
        if not pattern:
            continue

        rules.append(
            GitignoreRule(
                pattern=pattern,
                negation=negation,
                dir_only=dir_only,
                anchored=anchored,
                base_dir=base_dir,
            )
        )

    return rules


def is_ignored(full_path: Path, is_dir: bool, rules: list[GitignoreRule]) -> bool:
    """
    Apply an ordered list of rules (parent .gitignore rules first, then
    rules from deeper .gitignore files) to a path. Later matching rules
    win, and a negated match un-ignores — this mirrors git's own
    last-match-wins behavior.
    """
    ignored = False

    for rule in rules:
        if rule.dir_only and not is_dir:
            continue

        try:
            rel_path = full_path.relative_to(rule.base_dir).as_posix()
        except ValueError:
            # This rule belongs to a .gitignore outside full_path's tree.
            continue

        name = full_path.name
        candidate = rel_path if rule.anchored else name

        if fnmatch.fnmatch(candidate, rule.pattern):
            ignored = not rule.negation

    return ignored
