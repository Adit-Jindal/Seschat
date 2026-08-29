"""
metadata.py — Stage 2 of Seschat: extract structural metadata from source
files using tree-sitter, instead of just counting them by extension.

For each source file we try to answer:
    - What classes does it define?
    - What functions does it define?
    - What does it import?
    - What comments does it contain?

Why not regex?
A pattern like ``^\\s*def\\s+(\\w+)`` looks like it would find every Python
function... until it meets ``def`` inside a string, a comment, or a
decorator/type-annotation that spans multiple lines. Regex has no concept
of "am I inside a string right now?" — it just matches text. Tree-sitter
instead parses the file into a real Abstract Syntax Tree (a structured
representation of *what the code means*, not just what it looks like), so
we can ask "find every node of type function_definition" and get the
right answer even in code that would trip regex up.

This module deliberately keeps parsing and querying separate from
scanning (scanner.py) and presentation (cli.py):
    scanner.py  -> decides *which* files exist and what language they are
    metadata.py -> given a file's bytes + language, extracts structure
    cli.py      -> formats/writes the result

Every file gets one of five statuses (FileMetadata.status):
    "parsed"          — extraction ran and (maybe) found things.
    "not_applicable"  — this language isn't code in the relevant sense
                         (Markdown, JSON, YAML, ...) — there's no such
                         thing as a "function" in a YAML file, so we don't
                         try, and it's not an error that we didn't.
    "unsupported"      — a real programming language where classes/
                         functions/imports *do* make sense, but Seschat
                         hasn't wired up a NodeTypeSpec for it yet.
    "unavailable"       — tree-sitter / tree-sitter-language-pack isn't
                         installed at all, so nothing could be parsed.
    "error"            — we tried to parse and it failed (bad grammar
                         load, genuine syntax error, etc).
`FileMetadata.detail` carries a human-readable explanation for any status
other than "parsed".

Extending to a new language:
    1. Add an extension -> language-label entry in scanner.EXTENSION_MAP
       (if not already there).
    2. If it's a markup/data/config format with no classes/functions/
       imports (like another YAML-ish format), add it to
       NOT_APPLICABLE_LANGUAGES with a one-line reason instead of steps
       3-4 below.
    3. Otherwise, add language-label -> tree-sitter grammar name in
       TREE_SITTER_LANGUAGE_NAMES below.
    4. Add a NodeTypeSpec for that grammar name in LANGUAGE_SPECS,
       describing which node types are classes/functions/imports/comments
       in *that* grammar's tree. (Grammars don't agree on node-type names
       with each other, so this is a per-language table, not one clever
       generic query.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from tree_sitter_language_pack import get_parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


# --------------------------------------------------------------------------
# Status constants for FileMetadata.status.
# --------------------------------------------------------------------------
STATUS_PARSED = "parsed"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_UNSUPPORTED = "unsupported"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"
# For a file whose extension isn't in scanner.EXTENSION_MAP at all — Seschat
# doesn't know what language it is, let alone how to parse it. Distinct
# from STATUS_UNSUPPORTED, where the language *is* known but not wired up.
STATUS_UNRECOGNIZED = "unrecognized"


# --------------------------------------------------------------------------
# Languages (as labeled in scanner.EXTENSION_MAP) where classes/functions/
# imports genuinely don't apply — markup, data, and config formats. These
# are reported as "not_applicable", not as a missing feature.
# --------------------------------------------------------------------------
NOT_APPLICABLE_LANGUAGES: dict[str, str] = {
    "Markdown": "Markdown is prose/markup, not code — there's no such thing as a class or import to extract.",
    "reStructuredText": "reStructuredText is prose/markup, not code — there's no such thing as a class or import to extract.",
    "JSON": "JSON is a data format with no classes, functions, or imports.",
    "YAML": "YAML is a data format with no classes, functions, or imports.",
    "TOML": "TOML is a data format with no classes, functions, or imports.",
    "XML": "XML is a markup/data format with no classes, functions, or imports.",
    "HTML": "HTML is markup, not code. (Embedded <script>/<style> blocks aren't parsed separately yet.)",
    "CSS": "CSS has no classes, functions, or imports in the programming-language sense.",
    "SCSS": "SCSS has no classes, functions, or imports in the programming-language sense.",
}

# Languages that *are* code — and so could have classes/functions/imports
# worth extracting — but don't have a NodeTypeSpec wired up yet. Listed
# explicitly so the message can say why, and to distinguish "we haven't
# gotten to this" from the NOT_APPLICABLE case above. Anything else that
# falls through gets a generic fallback message.
UNSUPPORTED_LANGUAGES: dict[str, str] = {
    "Jupyter Notebook": "Notebook cells contain real code, but Seschat doesn't parse .ipynb cell contents yet.",
    "Swift": "Swift has a tree-sitter grammar, but Seschat hasn't added a NodeTypeSpec for it yet.",
    "Objective-C": "Objective-C has a tree-sitter grammar, but Seschat hasn't added a NodeTypeSpec for it yet.",
    "Shell": "Shell scripts can define functions, but Seschat hasn't added a NodeTypeSpec for it yet.",
    "SQL": "SQL has a tree-sitter grammar, but Seschat hasn't added a NodeTypeSpec for it yet.",
}


# --------------------------------------------------------------------------
# Language label -> tree-sitter grammar name (tree-sitter-language-pack).
# Only languages listed here attempt real structural parsing.
# --------------------------------------------------------------------------
TREE_SITTER_LANGUAGE_NAMES: dict[str, str] = {
    "Python": "python",
    "C": "c",
    "C++": "cpp",
    "C Header (or C++)": "cpp",
    "C++ Header": "cpp",
    "Java": "java",
    "Go": "go",
    "Rust": "rust",
    "Ruby": "ruby",
    "JavaScript": "javascript",
    "JavaScript (JSX)": "javascript",
    "TypeScript": "typescript",
    "TypeScript (TSX)": "tsx",
    "C#": "c_sharp",
    "PHP": "php",
    "Kotlin": "kotlin",
    "Scala": "scala",
}


@dataclass
class FileMetadata:
    """Structural metadata extracted from a single source file."""

    path: str
    language: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)

    # One of the STATUS_* constants above. classes/functions/etc. will be
    # empty for anything other than "parsed" — `detail` explains why, so
    # callers can tell "genuinely has none of these" apart from "we didn't
    # look" apart from "we can't look yet".
    status: str = STATUS_PARSED
    detail: str | None = None

    @property
    def parsed(self) -> bool:
        """Convenience boolean: True only for status == "parsed"."""
        return self.status == STATUS_PARSED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["parsed"] = self.parsed
        return d


# --------------------------------------------------------------------------
# Per-language node-type tables — the piece to extend for a new language.
# --------------------------------------------------------------------------
@dataclass
class NodeTypeSpec:
    class_nodes: tuple[str, ...] = ()
    function_nodes: tuple[str, ...] = ()
    import_nodes: tuple[str, ...] = ()
    comment_nodes: tuple[str, ...] = ("comment",)


LANGUAGE_SPECS: dict[str, NodeTypeSpec] = {
    "python": NodeTypeSpec(
        class_nodes=("class_definition",),
        function_nodes=("function_definition",),
        import_nodes=("import_statement", "import_from_statement"),
        comment_nodes=("comment",),
    ),
    "c": NodeTypeSpec(
        class_nodes=(),  # C has no classes
        function_nodes=("function_definition",),
        import_nodes=("preproc_include",),
        comment_nodes=("comment",),
    ),
    "cpp": NodeTypeSpec(
        class_nodes=("class_specifier", "struct_specifier"),
        function_nodes=("function_definition",),
        import_nodes=("preproc_include",),
        comment_nodes=("comment",),
    ),
    "java": NodeTypeSpec(
        class_nodes=("class_declaration", "interface_declaration"),
        function_nodes=("method_declaration", "constructor_declaration"),
        import_nodes=("import_declaration",),
        comment_nodes=("line_comment", "block_comment"),
    ),
    "go": NodeTypeSpec(
        class_nodes=("type_declaration",),  # closest Go analogue (structs)
        function_nodes=("function_declaration", "method_declaration"),
        import_nodes=("import_declaration",),
        comment_nodes=("comment",),
    ),
    "rust": NodeTypeSpec(
        class_nodes=("struct_item", "enum_item", "trait_item"),
        function_nodes=("function_item",),
        import_nodes=("use_declaration",),
        comment_nodes=("line_comment", "block_comment"),
    ),
    "ruby": NodeTypeSpec(
        class_nodes=("class", "module"),
        function_nodes=("method",),
        # require/require_relative/load show up as ordinary method calls in
        # Ruby's grammar, not a dedicated "import" node — filtered below.
        import_nodes=("call",),
        comment_nodes=("comment",),
    ),
    "javascript": NodeTypeSpec(
        class_nodes=("class_declaration",),
        function_nodes=("function_declaration", "method_definition"),
        import_nodes=("import_statement",),
        comment_nodes=("comment",),
    ),
    "typescript": NodeTypeSpec(
        class_nodes=("class_declaration", "interface_declaration"),
        function_nodes=("function_declaration", "method_definition"),
        import_nodes=("import_statement",),
        comment_nodes=("comment",),
    ),
    "tsx": NodeTypeSpec(
        class_nodes=("class_declaration", "interface_declaration"),
        function_nodes=("function_declaration", "method_definition"),
        import_nodes=("import_statement",),
        comment_nodes=("comment",),
    ),
    "c_sharp": NodeTypeSpec(
        class_nodes=("class_declaration", "interface_declaration", "struct_declaration"),
        function_nodes=("method_declaration",),
        import_nodes=("using_directive",),
        comment_nodes=("comment",),
    ),
    "php": NodeTypeSpec(
        class_nodes=("class_declaration", "interface_declaration"),
        function_nodes=("function_definition", "method_declaration"),
        import_nodes=("namespace_use_declaration",),
        comment_nodes=("comment",),
    ),
    "kotlin": NodeTypeSpec(
        class_nodes=("class_declaration",),
        function_nodes=("function_declaration",),
        import_nodes=("import_header",),
        comment_nodes=("comment", "line_comment", "multiline_comment"),
    ),
    "scala": NodeTypeSpec(
        class_nodes=("class_definition", "object_definition", "trait_definition"),
        function_nodes=("function_definition",),
        import_nodes=("import_declaration",),
        comment_nodes=("comment",),
    ),
}

# Ruby's "call" nodes are only imports when they're one of these methods.
_RUBY_IMPORT_METHODS = {"require", "require_relative", "load", "autoload"}

# Cache of loaded parsers, since building one per file would be wasteful.
_parser_cache: dict[str, Any] = {}


def _get_cached_parser(ts_name: str):
    if ts_name not in _parser_cache:
        _parser_cache[ts_name] = get_parser(ts_name)
    return _parser_cache[ts_name]


def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _node_name(node, source: bytes) -> str | None:
    """Pull the identifier out of a definition node's `name` field, if any."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node, source)
    return None


def _find_declarator_identifier(node, source: bytes) -> str | None:
    """
    C/C++ function names aren't a simple `name` field — the identifier is
    nested inside the declarator (which may itself be wrapped in pointer /
    reference / qualified-name declarators, e.g. `Cache::insert`, `int
    *foo()`). Walk down to find it.

    This handles the common cases, not every declarator shape C/C++ grammar
    allows (e.g. function pointers returning function pointers). Good
    enough for Stage 2; worth revisiting if it misidentifies real code.
    """
    if node is None:
        return None
    if node.type in ("identifier", "field_identifier", "operator_name", "destructor_name"):
        return _text(node, source)
    if node.type == "qualified_identifier":
        # `Namespace::Class::method` — the actual method name is the last
        # segment, held in the qualified_identifier's own `name` field.
        inner = node.child_by_field_name("name")
        return _find_declarator_identifier(inner, source)
    inner = node.child_by_field_name("declarator")
    if inner is not None:
        return _find_declarator_identifier(inner, source)
    return None


def _c_like_function_name(node, source: bytes) -> str | None:
    declarator = node.child_by_field_name("declarator")
    return _find_declarator_identifier(declarator, source)


def _definition_name(node, source: bytes, ts_name: str) -> str | None:
    if ts_name in ("c", "cpp") and node.type == "function_definition":
        return _c_like_function_name(node, source)
    return _node_name(node, source)


def extract_metadata(path: Path, language: str, source_bytes: bytes) -> FileMetadata:
    """
    Parse one file's contents and pull out its classes, functions, imports,
    and comments. Never raises — every outcome (success, not-applicable,
    unsupported, tree-sitter missing, genuine parse error) is reported via
    `FileMetadata.status`/`detail` so one bad file can't crash a whole scan.
    """
    meta = FileMetadata(path=str(path), language=language)

    # Markup/data/config formats: nothing to extract, and that's expected —
    # check this before anything else, regardless of whether tree-sitter is
    # even installed.
    if language in NOT_APPLICABLE_LANGUAGES:
        meta.status = STATUS_NOT_APPLICABLE
        meta.detail = NOT_APPLICABLE_LANGUAGES[language]
        return meta

    if not TREE_SITTER_AVAILABLE:
        meta.status = STATUS_UNAVAILABLE
        meta.detail = "tree-sitter / tree-sitter-language-pack isn't installed (pip install -e .)"
        return meta

    ts_name = TREE_SITTER_LANGUAGE_NAMES.get(language)
    if ts_name is None:
        meta.status = STATUS_UNSUPPORTED
        meta.detail = UNSUPPORTED_LANGUAGES.get(
            language, f"No tree-sitter grammar mapped for {language!r} yet."
        )
        return meta

    spec = LANGUAGE_SPECS.get(ts_name)
    if spec is None:
        meta.status = STATUS_UNSUPPORTED
        meta.detail = f"Grammar {ts_name!r} is mapped, but no NodeTypeSpec has been written for it yet."
        return meta

    try:
        parser = _get_cached_parser(ts_name)
        tree = parser.parse(source_bytes)
    except Exception as e:  # tree-sitter grammar/runtime issues, bad input, etc.
        meta.status = STATUS_ERROR
        meta.detail = f"parse failed: {e}"
        return meta

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children)

        if node.type in spec.class_nodes:
            name = _definition_name(node, source_bytes, ts_name)
            if name:
                meta.classes.append(name)

        elif node.type in spec.function_nodes:
            name = _definition_name(node, source_bytes, ts_name)
            if name:
                meta.functions.append(name)

        elif node.type in spec.import_nodes:
            if ts_name == "ruby" and node.type == "call":
                method_node = node.child_by_field_name("method")
                if method_node is None or _text(method_node, source_bytes) not in _RUBY_IMPORT_METHODS:
                    continue
            line = _text(node, source_bytes).strip()
            if line:
                meta.imports.append(line.splitlines()[0])

        elif node.type in spec.comment_nodes:
            text = _text(node, source_bytes).strip()
            if text:
                meta.comments.append(text)

    meta.status = STATUS_PARSED
    return meta


def make_unrecognized_metadata(path: Path, ext: str) -> FileMetadata:
    """
    Build the fallback entry for a file whose extension isn't in
    scanner.EXTENSION_MAP at all — a genuinely new/unmapped file type, or
    a file with no extension (Dockerfile, Makefile, LICENSE, ...).

    scanner.py calls this for every such file so the metadata index always
    has one entry per scanned file, never a silent gap. To move a file
    type out of this bucket, add it to EXTENSION_MAP in scanner.py — from
    there it'll fall into "not_applicable" or "unsupported"/"parsed"
    depending on whether it's registered in this module.
    """
    if ext:
        label = ext
        detail = (
            f"No language mapping for {ext!r} yet — add it to "
            f"scanner.EXTENSION_MAP to enable counting and metadata extraction."
        )
    else:
        label = "(no extension)"
        detail = "File has no extension, so Seschat can't infer a language for it yet."

    return FileMetadata(path=str(path), language=label, status=STATUS_UNRECOGNIZED, detail=detail)


def write_metadata_index(file_metadata: list[FileMetadata], root: Path) -> Path:
    """
    Write the collected per-file metadata to `<root>/.seschat/index.json`.
    A single index file (rather than one file per source file) keeps
    later stages (query, explain) simple: one JSON blob to load.
    """
    import json

    out_dir = root / ".seschat"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "index.json"

    payload = {
        "root": str(root),
        "files": [m.to_dict() for m in file_metadata],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path
