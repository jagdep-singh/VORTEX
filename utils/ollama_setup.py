from __future__ import annotations

import inspect
import ipaddress
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse

import httpx

OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_BASE_URL = "http://localhost:11434/api"
OLLAMA_SPACE_RESERVE_BYTES = 512 * 1024 * 1024
OLLAMA_INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"
OLLAMA_WINDOWS_INSTALL_SCRIPT_URL = "https://ollama.com/install.ps1"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
OLLAMA_LINUX_DOCS_URL = "https://docs.ollama.com/linux"
OLLAMA_MACOS_DOCS_URL = "https://docs.ollama.com/macos"
OLLAMA_WINDOWS_DOCS_URL = "https://docs.ollama.com/windows"


@dataclass(frozen=True)
class LocalModelOption:
    choice: str
    model_name: str
    title: str
    description: str
    size_bytes: int


LOCAL_MODEL_OPTIONS: tuple[LocalModelOption, ...] = (
    LocalModelOption(
        choice="1",
        model_name="qwen2.5-coder:1.5b",
        title="Fast + light",
        description=(
            "Smaller download, quicker startup, easier on laptops."
        ),
        size_bytes=986 * 1024 * 1024,
    ),
    LocalModelOption(
        choice="2",
        model_name="qwen2.5-coder:3b",
        title="Better coding quality",
        description=(
            "Larger download, slower than 1.5b, but usually better edits and code fixes."
        ),
        size_bytes=int(1.9 * 1024 * 1024 * 1024),
    ),
)


def get_local_model_option(choice: str) -> LocalModelOption | None:
    for option in LOCAL_MODEL_OPTIONS:
        if option.choice == choice:
            return option
    return None


def get_local_model_option_by_name(model_name: str) -> LocalModelOption | None:
    for option in LOCAL_MODEL_OPTIONS:
        if option.model_name == model_name:
            return option
    return None


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def supports_automatic_ollama_install() -> bool:
    return (
        sys.platform == "darwin"
        or sys.platform.startswith("linux")
        or os.name == "nt"
    )


def ollama_install_docs_url() -> str:
    if sys.platform == "darwin":
        return OLLAMA_MACOS_DOCS_URL
    if os.name == "nt":
        return OLLAMA_WINDOWS_DOCS_URL
    if sys.platform.startswith("linux"):
        return OLLAMA_LINUX_DOCS_URL
    return OLLAMA_DOWNLOAD_URL


def build_ollama_install_shell_command() -> str | None:
    if not supports_automatic_ollama_install():
        return None
    if os.name == "nt":
        return (
            'powershell -NoProfile -ExecutionPolicy Bypass -Command '
            f'"irm {OLLAMA_WINDOWS_INSTALL_SCRIPT_URL} | iex"'
        )
    return (
        'tmp="$(mktemp -t vortex-ollama-install.XXXXXX 2>/dev/null || mktemp)"; '
        'trap \'rm -f "$tmp"\' EXIT; '
        f'curl --fail --show-error --location "{OLLAMA_INSTALL_SCRIPT_URL}" -o "$tmp"; '
        'sh "$tmp"'
    )


def is_ollama_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    normalized = base_url.strip().rstrip("/")
    if normalized == OLLAMA_OPENAI_BASE_URL:
        return True

    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    if not hostname or (parsed.port not in {11434, None}):
        return False
    if parsed.path.rstrip("/") != "/v1":
        return False
    if hostname in {"localhost", "0.0.0.0", "::1", "host.docker.internal"}:
        return True

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return address.is_loopback or address.is_private or address.is_link_local


def resolve_ollama_models_dir() -> Path:
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path.home() / ".ollama" / "models"
    if os.name == "nt":
        return Path.home() / ".ollama" / "models"
    return Path("/usr/share/ollama/.ollama/models")


def _disk_usage_target(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else Path.home()


def get_free_space_bytes(path: Path | None = None) -> int:
    target = _disk_usage_target(path or resolve_ollama_models_dir())
    return shutil.disk_usage(target).free


def has_enough_space(
    *,
    required_bytes: int,
    free_bytes: int,
    reserve_bytes: int = OLLAMA_SPACE_RESERVE_BYTES,
) -> bool:
    return free_bytes >= (required_bytes + reserve_bytes)


def format_bytes(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


async def list_installed_models(api_base_url: str = OLLAMA_API_BASE_URL) -> list[str]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{api_base_url}/tags")
        response.raise_for_status()
        payload = response.json()

    models = payload.get("models", [])
    result: list[str] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if isinstance(name, str) and name:
            result.append(name)
    return result


async def pull_model(
    model_name: str,
    *,
    api_base_url: str = OLLAMA_API_BASE_URL,
    progress_callback: Callable[[str], None | Awaitable[None]] | None = None,
) -> None:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{api_base_url}/pull",
            json={"model": model_name, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                data = json.loads(line)
                if not isinstance(data, dict):
                    continue

                if data.get("error"):
                    raise RuntimeError(str(data["error"]))

                status = str(data.get("status", "")).strip()
                completed = data.get("completed")
                total = data.get("total")

                message = status or "Downloading model"
                if isinstance(completed, int) and isinstance(total, int) and total > 0:
                    message += (
                        f" ({format_bytes(completed)} / {format_bytes(total)})"
                    )

                if progress_callback:
                    maybe_awaitable = progress_callback(message)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
