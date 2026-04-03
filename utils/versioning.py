from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from config.loader import get_data_dir

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


PACKAGE_NAME = "vortex-agent-cli"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"


def _version_key(version: str) -> tuple[tuple[int, Any], ...]:
    cleaned = version.strip().lstrip("v")
    tokens = re.findall(r"\d+|[A-Za-z]+", cleaned)
    if not tokens:
        return ((1, cleaned.lower()),)

    key: list[tuple[int, Any]] = []
    for token in tokens:
        if token.isdigit():
            key.append((0, int(token)))
        else:
            key.append((1, token.lower()))
    return tuple(key)


def is_newer_version(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def recommended_update_instruction(install_mode: str) -> str:
    if install_mode in {"editable", "source"}:
        return "pull the latest repo changes"
    return "run vortex --update"


def _read_pyproject_version(project_root: Path) -> str:
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project", {})
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("No project.version value found in pyproject.toml")
    return version.strip()


def get_current_version(project_root: Path | None = None) -> str:
    try:
        return importlib_metadata.version(PACKAGE_NAME)
    except importlib_metadata.PackageNotFoundError:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        return _read_pyproject_version(project_root)


def detect_install_mode(package_name: str = PACKAGE_NAME) -> str:
    try:
        distribution = importlib_metadata.distribution(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "source"

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {}
        dir_info = direct_url.get("dir_info")
        if isinstance(dir_info, dict) and dir_info.get("editable"):
            return "editable"

    prefix_text = str(Path(sys.prefix).resolve())
    if "pipx" in prefix_text and "venvs" in prefix_text:
        return "pipx"

    return "pip"


@dataclass
class ReleaseInfo:
    current_version: str
    latest_version: str | None = None
    checked_at: str | None = None
    source: str = "none"
    error: str | None = None
    install_mode: str = "pip"

    @property
    def update_available(self) -> bool:
        if not self.latest_version:
            return False
        return is_newer_version(self.latest_version, self.current_version)


@dataclass
class UpdateCommand:
    argv: list[str]
    display: str


@dataclass
class UpdateResult:
    success: bool
    message: str
    command: str | None = None
    returncode: int | None = None


class ReleaseInfoStore:
    FILE_NAME = "release_info.json"

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or get_data_dir()
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.file_path = self.data_dir / self.FILE_NAME

    def load(self) -> dict[str, Any] | None:
        if not self.file_path.exists():
            return None

        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None
        return data

    def save(self, *, latest_version: str, checked_at: str) -> None:
        payload = {
            "latest_version": latest_version,
            "checked_at": checked_at,
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError:
            return


class VersionManager:
    def __init__(
        self,
        *,
        package_name: str = PACKAGE_NAME,
        pypi_json_url: str = PYPI_JSON_URL,
        data_dir: Path | None = None,
        project_root: Path | None = None,
        current_version: str | None = None,
        install_mode: str | None = None,
        cache_ttl: timedelta = timedelta(hours=6),
        request_timeout: float = 1.5,
    ) -> None:
        self.package_name = package_name
        self.pypi_json_url = pypi_json_url
        self.store = ReleaseInfoStore(data_dir=data_dir)
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.current_version = current_version or get_current_version(self.project_root)
        self.install_mode = install_mode or detect_install_mode(self.package_name)
        self.cache_ttl = cache_ttl
        self.request_timeout = request_timeout

    def fetch_latest_version(self) -> str:
        with urllib_request.urlopen(self.pypi_json_url, timeout=self.request_timeout) as response:
            payload = json.load(response)

        info = payload.get("info", {})
        version = info.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("PyPI response did not include a valid version")
        return version.strip()

    def _cached_release_info(self) -> ReleaseInfo | None:
        cached = self.store.load()
        if not cached:
            return None

        latest_version = cached.get("latest_version")
        checked_at = cached.get("checked_at")
        if not isinstance(latest_version, str) or not latest_version.strip():
            return None
        if checked_at is not None and not isinstance(checked_at, str):
            checked_at = None

        return ReleaseInfo(
            current_version=self.current_version,
            latest_version=latest_version.strip(),
            checked_at=checked_at,
            source="cache",
            install_mode=self.install_mode,
        )

    def _cache_is_fresh(self, checked_at: str | None) -> bool:
        if not checked_at:
            return False

        try:
            checked = datetime.fromisoformat(checked_at)
        except ValueError:
            return False

        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)

        return datetime.now(timezone.utc) - checked <= self.cache_ttl

    def get_release_info(self, *, force_refresh: bool = False) -> ReleaseInfo:
        cached = self._cached_release_info()
        if cached and not force_refresh and self._cache_is_fresh(cached.checked_at):
            return cached

        try:
            latest_version = self.fetch_latest_version()
        except (
            OSError,
            ValueError,
            urllib_error.URLError,
            urllib_error.HTTPError,
            TimeoutError,
        ) as exc:
            if cached is not None:
                cached.error = str(exc)
                return cached

            return ReleaseInfo(
                current_version=self.current_version,
                latest_version=None,
                checked_at=None,
                source="none",
                error=str(exc),
                install_mode=self.install_mode,
            )

        checked_at = datetime.now(timezone.utc).isoformat()
        self.store.save(latest_version=latest_version, checked_at=checked_at)
        return ReleaseInfo(
            current_version=self.current_version,
            latest_version=latest_version,
            checked_at=checked_at,
            source="live",
            install_mode=self.install_mode,
        )

    def resolve_update_command(self) -> UpdateCommand | None:
        if self.install_mode in {"editable", "source"}:
            return None

        if self.install_mode == "pipx":
            pipx_path = shutil.which("pipx")
            if pipx_path:
                return UpdateCommand(
                    argv=[pipx_path, "upgrade", self.package_name],
                    display=f"pipx upgrade {self.package_name}",
                )

        return UpdateCommand(
            argv=[sys.executable, "-m", "pip", "install", "--upgrade", self.package_name],
            display=f"{Path(sys.executable).name} -m pip install --upgrade {self.package_name}",
        )

    def perform_self_update(self) -> UpdateResult:
        command = self.resolve_update_command()
        if command is None:
            return UpdateResult(
                success=True,
                message=(
                    "This VORTEX instance is running from a local checkout and already "
                    "points at your current source tree. Pull the latest repo changes instead."
                ),
                command=None,
                returncode=0,
            )

        completed = subprocess.run(command.argv, check=False)
        if completed.returncode == 0:
            return UpdateResult(
                success=True,
                message="VORTEX update finished successfully.",
                command=command.display,
                returncode=completed.returncode,
            )

        return UpdateResult(
            success=False,
            message=f"VORTEX update failed with exit code {completed.returncode}.",
            command=command.display,
            returncode=completed.returncode,
        )
