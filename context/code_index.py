from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import os
import re

from utils.paths import is_binary_file
from utils.text import truncate_text

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".next",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

SOURCE_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".hpp": "c++",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

MAX_INDEXED_FILES = 250
MAX_FILE_BYTES = 256 * 1024
MAX_SYMBOLS = 800
MAX_SUMMARY_TOKENS = 1200


@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    language: str
    path: Path
    line: int
    signature: str | None = None

    def display_location(self, root: Path) -> str:
        try:
            rel_path = self.path.relative_to(root)
        except ValueError:
            rel_path = self.path
        return f"{rel_path}:{self.line}"


@dataclass
class WorkspaceCodeIndex:
    root: Path
    symbols: list[CodeSymbol]
    indexed_files: int
    language_files: dict[str, int]
    truncated: bool = False

    def render_summary(self, model_name: str) -> str:
        lines = [
            f"- Root: {self.root}",
            f"- Indexed source files: {self.indexed_files}"
            + (" (scan limited)" if self.truncated else ""),
            f"- Symbols extracted: {len(self.symbols)}",
        ]

        if self.language_files:
            lines.append(
                "- Languages: "
                + ", ".join(
                    f"{language} ({count})"
                    for language, count in sorted(
                        self.language_files.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:8]
                )
            )

        kind_counts = Counter(symbol.kind for symbol in self.symbols)
        if kind_counts:
            lines.append(
                "- Symbol kinds: "
                + ", ".join(
                    f"{kind} ({count})"
                    for kind, count in kind_counts.most_common(6)
                )
            )

        lines.extend(["", "## Notable Symbols"])

        notable = self.notable_symbols(limit=12)
        if notable:
            for symbol in notable:
                lines.append(
                    f"- {symbol.kind} {symbol.name} [{symbol.language}]"
                    f" - {symbol.display_location(self.root)}"
                )
        else:
            lines.append("- No symbols extracted from the current workspace yet.")

        lines.extend(
            [
                "",
                "Use the `find_symbol` tool to locate exact definitions quickly.",
            ]
        )

        return truncate_text("\n".join(lines), model_name, MAX_SUMMARY_TOKENS)

    def notable_symbols(self, limit: int = 10) -> list[CodeSymbol]:
        ranked = sorted(
            self.symbols,
            key=lambda symbol: (
                -_symbol_priority(symbol),
                len(symbol.path.parts),
                symbol.path.as_posix(),
                symbol.line,
            ),
        )
        return ranked[:limit]

    def find(
        self,
        query: str,
        *,
        kind: str | None = None,
        language: str | None = None,
        limit: int = 8,
    ) -> list[CodeSymbol]:
        normalized_query = query.strip().lower()
        normalized_kind = kind.strip().lower() if kind else None
        normalized_language = language.strip().lower() if language else None

        filtered_symbols = [
            symbol
            for symbol in self.symbols
            if (not normalized_kind or symbol.kind.lower() == normalized_kind)
            and (not normalized_language or symbol.language.lower() == normalized_language)
        ]

        if normalized_query in {"", "*", "all"}:
            matches = sorted(
                filtered_symbols,
                key=lambda symbol: (
                    -_symbol_priority(symbol),
                    len(symbol.path.parts),
                    symbol.path.as_posix(),
                    symbol.line,
                ),
            )[:limit]
        else:
            scored: list[tuple[int, CodeSymbol]] = []
            best_quality = 0
            for symbol in filtered_symbols:
                quality = _match_quality(symbol, normalized_query)
                if quality <= 0:
                    continue
                best_quality = max(best_quality, quality)
                score = _match_score(symbol, normalized_query)
                scored.append((score, symbol))

            if best_quality >= 2:
                scored = [
                    item
                    for item in scored
                    if _match_quality(item[1], normalized_query) >= 2
                ]

            scored.sort(
                key=lambda item: (
                    -item[0],
                    item[1].path.as_posix(),
                    item[1].line,
                )
            )
            matches = [symbol for _, symbol in scored[:limit]]

        return matches[:limit]


def build_workspace_code_index(cwd: Path) -> WorkspaceCodeIndex | None:
    root = cwd.resolve()
    if not root.exists() or not root.is_dir():
        return None

    symbols: list[CodeSymbol] = []
    seen_symbols: set[tuple[str, str, str, int]] = set()
    language_files: Counter[str] = Counter()
    indexed_files = 0
    truncated = False

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [directory for directory in sorted(dirs) if directory not in IGNORED_DIRS]

        for filename in sorted(filenames):
            path = Path(current_root) / filename
            language = SOURCE_LANGUAGES.get(path.suffix.lower())

            if not language or _should_skip_file(path, root):
                continue

            indexed_files += 1
            language_files[language] += 1

            text = _read_text(path)
            if text is None:
                continue

            for symbol in _extract_symbols(text, path, language):
                key = (symbol.name, symbol.kind, symbol.path.as_posix(), symbol.line)
                if key in seen_symbols:
                    continue
                seen_symbols.add(key)
                symbols.append(symbol)
                if len(symbols) >= MAX_SYMBOLS:
                    truncated = True
                    return WorkspaceCodeIndex(
                        root=root,
                        symbols=symbols,
                        indexed_files=indexed_files,
                        language_files=dict(language_files),
                        truncated=truncated,
                    )

            if indexed_files >= MAX_INDEXED_FILES:
                truncated = True
                return WorkspaceCodeIndex(
                    root=root,
                    symbols=symbols,
                    indexed_files=indexed_files,
                    language_files=dict(language_files),
                    truncated=truncated,
                )

    return WorkspaceCodeIndex(
        root=root,
        symbols=symbols,
        indexed_files=indexed_files,
        language_files=dict(language_files),
        truncated=truncated,
    )


def _should_skip_file(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
        return True

    if path.name.startswith("."):
        return True

    if path.is_symlink():
        return True

    try:
        size = path.stat().st_size
    except OSError:
        return True

    if size > MAX_FILE_BYTES:
        return True

    if is_binary_file(path):
        return True

    return False


def _read_text(path: Path) -> str | None:
    try:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")
    except OSError:
        return None


def _extract_symbols(text: str, path: Path, language: str) -> list[CodeSymbol]:
    patterns = _patterns_for_language(language)
    if not patterns:
        return []

    symbols: list[CodeSymbol] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue

            name = _match_name(match)
            if not name or name in {"if", "for", "while", "switch", "return"}:
                continue

            symbols.append(
                CodeSymbol(
                    name=name,
                    kind=kind,
                    language=language,
                    path=path,
                    line=line_number,
                    signature=_clean_signature(line),
                )
            )
            break

    return symbols


def _patterns_for_language(language: str) -> list[tuple[str, re.Pattern[str]]]:
    if language == "python":
        return [
            ("class", re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
            (
                "function",
                re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            ),
        ]

    if language in {"javascript", "typescript"}:
        return [
            (
                "class",
                re.compile(
                    r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"
                ),
            ),
            (
                "function",
                re.compile(
                    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
                ),
            ),
            (
                "function",
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
                ),
            ),
            (
                "interface",
                re.compile(
                    r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"
                ),
            ),
            (
                "type",
                re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
            ),
            (
                "enum",
                re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
            ),
        ]

    if language in {"c", "c++"}:
        return [
            ("macro", re.compile(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
            (
                "struct",
                re.compile(
                    r"^\s*(?:typedef\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)?\b"
                ),
            ),
            (
                "enum",
                re.compile(r"^\s*(?:typedef\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)?\b"),
            ),
            (
                "function",
                re.compile(
                    r"^\s*(?:static\s+|inline\s+|extern\s+|const\s+|unsigned\s+|signed\s+|long\s+|short\s+|volatile\s+)*[A-Za-z_][A-Za-z0-9_\s\*\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:\{|;)?\s*$"
                ),
            ),
        ]

    if language == "java":
        return [
            (
                "class",
                re.compile(
                    r"^\s*(?:public|private|protected)?\s*(?:abstract\s+|final\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b"
                ),
            ),
            (
                "interface",
                re.compile(
                    r"^\s*(?:public|private|protected)?\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)\b"
                ),
            ),
            (
                "enum",
                re.compile(
                    r"^\s*(?:public|private|protected)?\s*enum\s+([A-Za-z_][A-Za-z0-9_]*)\b"
                ),
            ),
            (
                "function",
                re.compile(
                    r"^\s*(?:public|private|protected|static|final|synchronized|native|abstract|\s)+[A-Za-z_][A-Za-z0-9_<>\[\],\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*(?:\{|;)?\s*$"
                ),
            ),
        ]

    if language == "go":
        return [
            (
                "function",
                re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            ),
            (
                "type",
                re.compile(
                    r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface|map|chan|\[)"
                ),
            ),
        ]

    if language == "rust":
        return [
            (
                "function",
                re.compile(
                    r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
                ),
            ),
            (
                "struct",
                re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
            ),
            (
                "enum",
                re.compile(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
            ),
            (
                "trait",
                re.compile(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
            ),
        ]

    return []


def _match_name(match: re.Match[str]) -> str | None:
    for group in match.groups():
        if group:
            return group
    return None


def _clean_signature(line: str) -> str:
    signature = re.sub(r"\s+", " ", line).strip()
    if len(signature) > 120:
        return signature[:117].rstrip() + "..."
    return signature


def _symbol_priority(symbol: CodeSymbol) -> int:
    kind_priority = {
        "class": 120,
        "interface": 115,
        "trait": 112,
        "struct": 108,
        "enum": 104,
        "type": 96,
        "function": 90,
        "macro": 82,
    }
    name_bonus = {
        "main": 20,
        "app": 12,
        "agent": 12,
        "cli": 10,
        "run": 8,
        "execute": 8,
    }
    return kind_priority.get(symbol.kind, 70) + name_bonus.get(symbol.name.lower(), 0)


def _match_score(symbol: CodeSymbol, query: str) -> int:
    quality = _match_quality(symbol, query)
    if quality <= 0:
        return 0
    return quality * 1000 + _symbol_priority(symbol)


def _match_quality(symbol: CodeSymbol, query: str) -> int:
    name = symbol.name.lower()
    signature = (symbol.signature or "").lower()
    location = symbol.path.as_posix().lower()

    if name == query:
        return 4
    if name.startswith(query):
        return 3
    if query in name:
        return 2
    if query in location:
        return 1
    if query in signature:
        return 1
    return 0
