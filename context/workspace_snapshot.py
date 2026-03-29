from __future__ import annotations

from collections import Counter
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

SENSITIVE_FILE_PATTERNS = (
    re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
    re.compile(r".*secret.*", re.IGNORECASE),
    re.compile(r".*token.*", re.IGNORECASE),
    re.compile(r".*password.*", re.IGNORECASE),
    re.compile(r".*credential.*", re.IGNORECASE),
    re.compile(r".*private.*key.*", re.IGNORECASE),
)

KEY_FILE_BONUS = {
    "readme.md": 120,
    "agent.md": 115,
    "main.py": 110,
    "app.py": 105,
    "index.html": 100,
    "package.json": 98,
    "pyproject.toml": 98,
    "requirements.txt": 96,
    "dockerfile": 95,
    "compose.yaml": 94,
    "docker-compose.yml": 94,
    "makefile": 92,
    "cargo.toml": 90,
    "go.mod": 90,
}

TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

MAX_SCANNED_FILES = 400
MAX_TOP_LEVEL_ENTRIES = 16
MAX_KEY_FILES = 12
MAX_PREVIEW_BYTES = 4096
MAX_SUMMARY_TOKENS = 1800


def build_workspace_snapshot(cwd: Path, model_name: str) -> str | None:
    root = cwd.resolve()
    if not root.exists() or not root.is_dir():
        return None

    top_level_entries = _list_top_level(root)
    discovered_files, scan_truncated = _scan_files(root)

    if not discovered_files and not top_level_entries:
        summary = (
            f"- Root: {root}\n"
            "- The workspace is currently empty.\n"
        )
        return truncate_text(summary, model_name, MAX_SUMMARY_TOKENS)

    extension_counts = Counter(_extension_label(path) for path in discovered_files)
    key_files = sorted(
        discovered_files,
        key=lambda path: _file_priority(path, root),
        reverse=True,
    )[:MAX_KEY_FILES]

    lines = [
        f"- Root: {root}",
        f"- Files scanned: {len(discovered_files)}"
        + (" (scan limited)" if scan_truncated else ""),
    ]

    if top_level_entries:
        lines.append(
            "- Top-level entries: " + ", ".join(top_level_entries[:MAX_TOP_LEVEL_ENTRIES])
        )

    if extension_counts:
        type_summary = ", ".join(
            f"{name} ({count})"
            for name, count in extension_counts.most_common(8)
        )
        lines.append(f"- Common file types: {type_summary}")

    lines.extend(
        [
            "",
            "## Key Files",
        ]
    )

    if key_files:
        for path in key_files:
            rel_path = path.relative_to(root)
            preview = _describe_file(path)
            lines.append(f"- {rel_path}: {preview}")
    else:
        lines.append("- No key files detected yet.")

    lines.extend(
        [
            "",
            "Use file tools to inspect exact contents before editing. "
            "This snapshot is a compact map, not a replacement for read_file.",
        ]
    )

    summary = "\n".join(lines)
    return truncate_text(summary, model_name, MAX_SUMMARY_TOKENS)


def _list_top_level(root: Path) -> list[str]:
    entries = []
    try:
        items = sorted(root.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower()))
    except OSError:
        return entries

    for item in items:
        if item.name in IGNORED_DIRS:
            continue
        suffix = "/" if item.is_dir() else ""
        entries.append(f"{item.name}{suffix}")
        if len(entries) >= MAX_TOP_LEVEL_ENTRIES:
            break

    return entries


def _scan_files(root: Path) -> tuple[list[Path], bool]:
    files: list[Path] = []

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [
            directory
            for directory in sorted(dirs)
            if directory not in IGNORED_DIRS
        ]

        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if _should_skip_file(path, root):
                continue
            files.append(path)
            if len(files) >= MAX_SCANNED_FILES:
                return files, True

    return files, False


def _should_skip_file(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
        return True

    if path.name.startswith(".") and path.name not in {".gitignore", ".env", ".env.example"}:
        if ".ai-agent" not in relative.parts:
            return True

    if path.is_symlink():
        return True

    try:
        size = path.stat().st_size
    except OSError:
        return True

    if size > 1024 * 1024:
        return True

    if is_binary_file(path):
        return True

    return False


def _file_priority(path: Path, root: Path) -> tuple[int, int, int, str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    name = path.name.lower()
    extension = path.suffix.lower()
    depth = len(relative.parts)

    score = KEY_FILE_BONUS.get(name, 0)

    if depth == 1:
        score += 45
    elif depth == 2:
        score += 20

    if extension in TEXT_EXTENSIONS:
        score += 20

    if ".ai-agent" in relative.parts:
        score += 15

    return (score, -depth, -len(name), str(relative))


def _describe_file(path: Path) -> str:
    if _is_sensitive_file(path):
        return "Sensitive file present; contents intentionally omitted."

    special = _special_file_description(path)
    if special:
        return special

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="latin-1")
        except OSError:
            return "Could not read file preview."
    except OSError:
        return "Could not read file preview."

    if not raw.strip():
        return "Empty file."

    snippet = raw[:MAX_PREVIEW_BYTES]
    if path.suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".rs", ".go", ".java"}:
        preview = _extract_code_hint(snippet) or _first_useful_line(snippet)
    else:
        preview = (
            _extract_markdown_heading(snippet)
            or _extract_html_title(snippet)
            or _extract_config_key(snippet)
            or _extract_code_hint(snippet)
            or _first_useful_line(snippet)
        )

    preview = re.sub(r"\s+", " ", preview).strip()
    if len(preview) > 120:
        preview = preview[:117].rstrip() + "..."

    return preview or "Text file."


def _special_file_description(path: Path) -> str | None:
    name = path.name.lower()
    if name == "dockerfile":
        return "Docker build recipe"
    if name in {"compose.yaml", "docker-compose.yml", "docker-compose.yaml"}:
        return "Container composition config"
    if name == "requirements.txt":
        return "Python dependency list"
    if name == ".gitignore":
        return "Git ignore rules"
    return None


def _is_sensitive_file(path: Path) -> bool:
    name = path.name
    return any(pattern.match(name) for pattern in SENSITIVE_FILE_PATTERNS)


def _extract_markdown_heading(text: str) -> str | None:
    match = re.search(r"^\s{0,3}#\s+(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1)
    return None


def _extract_html_title(text: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if match and match.group(1).strip():
        return f"HTML page titled '{match.group(1).strip()}'"
    if "<!DOCTYPE html" in text or "<html" in text.lower():
        return "HTML document"
    return None


def _extract_config_key(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", ";", "[")):
            continue
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip().strip('"').strip("'")
            if key:
                return f"Configuration file with key '{key}'"
        if ":" in stripped and not stripped.startswith(("http://", "https://")):
            key = stripped.split(":", 1)[0].strip().strip('"').strip("'")
            if key:
                return f"Configuration file with key '{key}'"
    return None


def _extract_code_hint(text: str) -> str | None:
    patterns = (
        r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*async\s+def\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*export\s+(?:default\s+)?(?:function|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*int\s+main\s*\(",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            if match.lastindex:
                return f"Defines {match.group(1)}"
            return "Program entrypoint"
    return None


def _first_useful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "//", "/*", "*", ";")):
            continue
        return stripped
    return "Text file."


def _extension_label(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix:
        return suffix
    return "[no extension]"
