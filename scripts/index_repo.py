"""Code Repository Indexer — build symbol index using tree-sitter.

Scans configured code_repos, extracts functions/classes/types/components,
builds a searchable index stored in .deer-flow/indexes/.

Usage:
    python scripts/index_repo.py                    # Index all configured repos
    python scripts/index_repo.py search <query>     # Search the index
    python scripts/index_repo.py stats              # Show index statistics
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.app import PROJECT_ROOT, load_agent_config, load_global_config, resolve_agent_name


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    name: str
    kind: str           # function, class, interface, type, enum, const/fn, const/call, component
    file: str           # relative path within repo
    line_start: int
    line_end: int
    exported: bool = False

    @property
    def span(self) -> int:
        return self.line_end - self.line_start


@dataclass
class FileEntry:
    path: str           # relative to repo root
    symbols: list[Symbol] = field(default_factory=list)
    lines: int = 0
    size_bytes: int = 0

    # Pre-computed for code graph (phase 2)
    imports: list[str] = field(default_factory=list)  # imported module paths


@dataclass
class RepoIndex:
    name: str
    root_path: str
    languages: list[str]
    files: dict[str, FileEntry] = field(default_factory=dict)  # path → FileEntry
    built_at: str = ""
    build_seconds: float = 0

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_symbols(self) -> int:
        return sum(len(f.symbols) for f in self.files.values())


# ---------------------------------------------------------------------------
# Tree-sitter extraction
# ---------------------------------------------------------------------------

_parsers = {}


def _get_parser(lang: str):
    """Lazy-init parser for a language."""
    if lang in _parsers:
        return _parsers[lang]

    from tree_sitter import Language, Parser

    if lang in ("typescript", "tsx"):
        import tree_sitter_typescript as ts_lang
        language = Language(ts_lang.language_tsx())
    elif lang == "javascript":
        import tree_sitter_javascript as js_lang
        language = Language(js_lang.language())
    else:
        return None

    parser = Parser(language)
    _parsers[lang] = parser
    return parser


def _extract_symbols(code: bytes, parser, file_path: str) -> tuple[list[Symbol], list[str]]:
    """Extract symbols and imports from a parsed file."""
    tree = parser.parse(code)
    symbols = []
    imports = []

    def _visit(node, exported=False):
        if node.type == "import_statement":
            # Extract import source
            source = node.child_by_field_name("source")
            if source:
                imports.append(source.text.decode().strip("'\""))
            return

        if node.type == "export_statement":
            for child in node.children:
                _visit(child, exported=True)
            return

        name_node = None
        kind = None

        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            kind = "function"

        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            kind = "class"

        elif node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            kind = "interface"

        elif node.type == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            kind = "type"

        elif node.type == "enum_declaration":
            name_node = node.child_by_field_name("name")
            kind = "enum"

        elif node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    n = child.child_by_field_name("name")
                    v = child.child_by_field_name("value")
                    if n and v:
                        if v.type in ("arrow_function", "function_expression"):
                            kind = "const/fn"
                        elif v.type == "call_expression":
                            kind = "const/call"
                        elif v.type == "jsx_element" or v.type == "jsx_fragment":
                            kind = "component"
                        if kind:
                            symbols.append(Symbol(
                                name=n.text.decode(),
                                kind=kind,
                                file=file_path,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                exported=exported,
                            ))
                            kind = None  # reset for next declarator
            return  # handled children inline

        if name_node and kind:
            symbols.append(Symbol(
                name=name_node.text.decode(),
                kind=kind,
                file=file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                exported=exported,
            ))

        for child in node.children:
            _visit(child, exported=False)

    _visit(tree.root_node)
    return symbols, imports


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _get_git_files(repo_path: str, extensions: list[str]) -> list[str]:
    """Get tracked files from git."""
    patterns = [f"*.{ext}" for ext in extensions]
    try:
        result = subprocess.check_output(
            ["git", "ls-files"] + patterns,
            cwd=repo_path,
            stderr=subprocess.DEVNULL,
        )
        return result.decode().strip().split("\n")
    except subprocess.CalledProcessError:
        return []


def build_index(repo_name: str, repo_path: str, languages: list[str]) -> RepoIndex:
    """Build symbol index for a repository."""
    start = time.time()

    ext_map = {
        "typescript": ["ts", "tsx"],
        "tsx": ["tsx"],
        "javascript": ["js", "jsx"],
    }
    extensions = []
    for lang in languages:
        extensions.extend(ext_map.get(lang, [lang]))

    files = _get_git_files(repo_path, list(set(extensions)))

    # Filter out test/mock/dist files
    skip_patterns = ["/node_modules/", "/__tests__/", "/__test__/", "/__mocks__/",
                     "/dist/", "/build/", ".test.", ".spec.", "/stories/", ".d.ts"]
    files = [f for f in files if f and not any(p in f for p in skip_patterns)]

    parser = _get_parser(languages[0])
    if parser is None:
        print(f"  No parser for {languages[0]}")
        return RepoIndex(name=repo_name, root_path=repo_path, languages=languages)

    index = RepoIndex(name=repo_name, root_path=repo_path, languages=languages)

    for i, rel_path in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  ... {i}/{len(files)} files processed")

        full_path = os.path.join(repo_path, rel_path)
        try:
            code = open(full_path, "rb").read()
        except (OSError, IOError):
            continue

        symbols, imports = _extract_symbols(code, parser, rel_path)

        index.files[rel_path] = FileEntry(
            path=rel_path,
            symbols=symbols,
            lines=code.count(b"\n") + 1,
            size_bytes=len(code),
            imports=imports,
        )

    index.build_seconds = time.time() - start
    from datetime import datetime, timezone
    index.built_at = datetime.now(timezone.utc).isoformat()

    return index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _index_path(repo_name: str) -> Path:
    p = PROJECT_ROOT / ".deer-flow" / "indexes"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{repo_name}.json"


def save_index(index: RepoIndex) -> Path:
    """Save index to JSON."""
    path = _index_path(index.name)

    data = {
        "name": index.name,
        "root_path": index.root_path,
        "languages": index.languages,
        "built_at": index.built_at,
        "build_seconds": index.build_seconds,
        "files": {},
    }
    for fpath, entry in index.files.items():
        data["files"][fpath] = {
            "lines": entry.lines,
            "size": entry.size_bytes,
            "imports": entry.imports,
            "symbols": [asdict(s) for s in entry.symbols],
        }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return path


def load_index(repo_name: str) -> RepoIndex | None:
    """Load index from JSON."""
    path = _index_path(repo_name)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    index = RepoIndex(
        name=data["name"],
        root_path=data["root_path"],
        languages=data["languages"],
        built_at=data.get("built_at", ""),
        build_seconds=data.get("build_seconds", 0),
    )
    for fpath, fdata in data.get("files", {}).items():
        symbols = [Symbol(**s) for s in fdata.get("symbols", [])]
        index.files[fpath] = FileEntry(
            path=fpath,
            symbols=symbols,
            lines=fdata.get("lines", 0),
            size_bytes=fdata.get("size", 0),
            imports=fdata.get("imports", []),
        )
    return index


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_index(index: RepoIndex, query: str, limit: int = 20) -> list[dict]:
    """Search symbols by name (case-insensitive substring match)."""
    query_lower = query.lower()
    results = []

    for fpath, entry in index.files.items():
        for sym in entry.symbols:
            # Match on symbol name
            if query_lower in sym.name.lower():
                results.append({
                    "name": sym.name,
                    "kind": sym.kind,
                    "file": sym.file,
                    "line": sym.line_start,
                    "span": sym.span,
                    "exported": sym.exported,
                })
            # Also match on file path
            elif query_lower in fpath.lower():
                results.append({
                    "name": sym.name,
                    "kind": sym.kind,
                    "file": sym.file,
                    "line": sym.line_start,
                    "span": sym.span,
                    "exported": sym.exported,
                })

    # Sort: exact name match first, then by file path
    results.sort(key=lambda r: (0 if query_lower == r["name"].lower() else 1, r["file"]))
    return results[:limit]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_build():
    """Build indexes for all configured repos."""
    global_cfg = load_global_config()
    agent_name = resolve_agent_name(None, global_cfg)
    agent_cfg = load_agent_config(agent_name)
    repos = agent_cfg.get("code_repos", [])

    if not repos:
        print("No code_repos configured in agent.yaml")
        return

    for repo in repos:
        name = repo["name"]
        path = repo["path"]
        languages = repo.get("languages", ["typescript", "tsx"])

        print(f"\nIndexing {name} ({path})...")
        index = build_index(name, path, languages)
        saved = save_index(index)

        print(f"  Files:   {index.total_files}")
        print(f"  Symbols: {index.total_symbols}")
        print(f"  Time:    {index.build_seconds:.1f}s")
        print(f"  Saved:   {saved}")

    print("\nDone.")


def cmd_search(query: str):
    """Search across all indexes."""
    global_cfg = load_global_config()
    agent_name = resolve_agent_name(None, global_cfg)
    agent_cfg = load_agent_config(agent_name)
    repos = agent_cfg.get("code_repos", [])

    for repo in repos:
        index = load_index(repo["name"])
        if index is None:
            print(f"  Index not found for {repo['name']}. Run: python scripts/index_repo.py")
            continue

        results = search_index(index, query)
        if results:
            print(f"\n{repo['name']} — {len(results)} matches for \"{query}\":\n")
            for r in results:
                exp = "export " if r["exported"] else "       "
                print(f"  {exp}{r['kind']:12s} {r['name']:40s} {r['file']}:{r['line']} ({r['span']} lines)")
        else:
            print(f"\n{repo['name']} — no matches for \"{query}\"")


def cmd_stats():
    """Show index statistics."""
    global_cfg = load_global_config()
    agent_name = resolve_agent_name(None, global_cfg)
    agent_cfg = load_agent_config(agent_name)
    repos = agent_cfg.get("code_repos", [])

    for repo in repos:
        index = load_index(repo["name"])
        if index is None:
            print(f"  {repo['name']}: not indexed")
            continue

        print(f"\n{index.name}:")
        print(f"  Root:      {index.root_path}")
        print(f"  Built:     {index.built_at}")
        print(f"  Files:     {index.total_files}")
        print(f"  Symbols:   {index.total_symbols}")
        print(f"  Build time: {index.build_seconds:.1f}s")

        # Breakdown by kind
        kinds = {}
        for entry in index.files.values():
            for sym in entry.symbols:
                kinds[sym.kind] = kinds.get(sym.kind, 0) + 1
        for kind, count in sorted(kinds.items(), key=lambda x: -x[1]):
            print(f"    {kind:15s} {count:>6}")

        # Top directories by symbol count
        dirs = {}
        for entry in index.files.values():
            d = "/".join(entry.path.split("/")[:3])
            dirs[d] = dirs.get(d, 0) + len(entry.symbols)
        print(f"\n  Top directories:")
        for d, count in sorted(dirs.items(), key=lambda x: -x[1])[:10]:
            print(f"    {d:50s} {count:>5} symbols")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "build":
        cmd_build()
    elif sys.argv[1] == "search":
        if len(sys.argv) < 3:
            print("Usage: index_repo.py search <query>")
            return
        cmd_search(sys.argv[2])
    elif sys.argv[1] == "stats":
        cmd_stats()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
