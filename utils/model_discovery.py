from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from config.config import Config
from config.loader import get_data_dir
from utils.model_health import ModelHealthRecord


MODEL_STATUS_ORDER = (
    "working",
    "quota",
    "rate-limited",
    "auth-error",
    "unavailable",
    "offline",
    "error",
    "missing-key",
    "cached",
    "unknown",
)

MODEL_BUCKET_ORDER = (
    "working",
    "quota",
    "not-working",
)


@dataclass(frozen=True)
class ModelCatalogSource:
    profile_name: str | None
    display_name: str
    base_url: str
    api_key: str | None
    key_source_label: str
    is_active: bool


@dataclass
class ModelCatalogResult:
    source: ModelCatalogSource
    models: list[str]
    checked_at: str
    error: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class ModelCatalogEntry:
    index: int
    profile_name: str | None
    display_name: str
    model_name: str
    base_url: str
    key_source_label: str
    checked_at: str
    is_active: bool
    status: str
    note: str | None = None


def model_status_bucket(status: str | None) -> str:
    normalized = (status or "unknown").strip().lower()
    if normalized == "working":
        return "working"
    if normalized in {"quota", "rate-limited"}:
        return "quota"
    return "not-working"


def group_model_catalog_entries_by_bucket(
    entries: list[ModelCatalogEntry],
) -> list[tuple[str, list[ModelCatalogEntry]]]:
    grouped: dict[str, list[ModelCatalogEntry]] = {}
    for entry in entries:
        grouped.setdefault(model_status_bucket(entry.status), []).append(entry)

    return [
        (bucket, grouped[bucket])
        for bucket in MODEL_BUCKET_ORDER
        if bucket in grouped
    ]


class ModelCatalogStore:
    def __init__(self) -> None:
        self.data_dir = get_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "model_catalog_cache.json"
        self._records = self._load()

    def _source_key(self, source: ModelCatalogSource) -> str:
        payload = json.dumps(
            {
                "profile_name": source.profile_name,
                "display_name": source.display_name,
                "base_url": source.base_url,
                "api_key_sha256": hashlib.sha256(
                    (source.api_key or "").encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, dict[str, object]]:
        if not self.file_path.exists():
            return {}

        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        try:
            self.file_path.write_text(
                json.dumps(self._records, indent=2),
                encoding="utf-8",
            )
            os.chmod(self.file_path, 0o600)
        except OSError:
            return

    def save(self, result: ModelCatalogResult) -> None:
        if result.error:
            return

        self._records[self._source_key(result.source)] = {
            "models": list(result.models),
            "checked_at": result.checked_at,
        }
        self._save()

    def get(self, source: ModelCatalogSource) -> ModelCatalogResult | None:
        cached = self._records.get(self._source_key(source))
        if not isinstance(cached, dict):
            return None

        models = cached.get("models")
        checked_at = cached.get("checked_at")
        if not isinstance(models, list) or not isinstance(checked_at, str):
            return None

        return ModelCatalogResult(
            source=source,
            models=[str(model) for model in models],
            checked_at=checked_at,
            cached=True,
        )


def build_model_catalog_sources(config: Config) -> list[ModelCatalogSource]:
    sources: list[ModelCatalogSource] = []

    for name, _profile in config.list_model_profiles():
        sources.append(
            ModelCatalogSource(
                profile_name=name,
                display_name=name,
                base_url=config.resolve_profile_base_url(name),
                api_key=config.resolve_profile_api_key(name),
                key_source_label=config.resolve_profile_key_source_label(name),
                is_active=config.active_model_profile == name,
            )
        )

    if not sources:
        sources.append(
            ModelCatalogSource(
                profile_name=None,
                display_name="default",
                base_url=config.resolve_profile_base_url(None),
                api_key=config.resolve_profile_api_key(None),
                key_source_label=config.resolve_profile_key_source_label(None),
                is_active=config.active_model_profile is None,
            )
        )

    return sources


def flatten_model_catalog(
    results: list[ModelCatalogResult],
    health_records: dict[tuple[str, str], ModelHealthRecord] | None = None,
) -> list[ModelCatalogEntry]:
    entries: list[ModelCatalogEntry] = []

    for result in results:
        for model_name in result.models:
            record = None
            if health_records is not None:
                record = health_records.get((result.source.display_name, model_name))

            entries.append(
                ModelCatalogEntry(
                    index=len(entries) + 1,
                    profile_name=result.source.profile_name,
                    display_name=result.source.display_name,
                    model_name=model_name,
                    base_url=result.source.base_url,
                    key_source_label=result.source.key_source_label,
                    checked_at=record.checked_at if record else result.checked_at,
                    is_active=result.source.is_active,
                    status=(
                        record.status
                        if record
                        else "cached"
                        if result.cached
                        else "unknown"
                    ),
                    note=result.error if result.cached else None,
                )
            )

    return entries


def select_best_working_model(
    results: list[ModelCatalogResult],
    *,
    health_records: dict[tuple[str, str], ModelHealthRecord] | None = None,
    preferred_profile_name: str | None = None,
    preferred_model_name: str | None = None,
) -> ModelCatalogEntry | None:
    working_entries = [
        entry
        for entry in flatten_model_catalog(results, health_records=health_records)
        if entry.status == "working"
    ]
    if not working_entries:
        return None

    def _rank(entry: ModelCatalogEntry) -> tuple[int, int, int, int, int]:
        is_exact_match = (
            entry.profile_name == preferred_profile_name
            and entry.model_name == preferred_model_name
        )
        is_same_profile = entry.profile_name == preferred_profile_name
        is_same_model_name = entry.model_name == preferred_model_name
        return (
            1 if is_exact_match else 0,
            1 if is_same_profile else 0,
            1 if is_same_model_name else 0,
            1 if entry.is_active else 0,
            -entry.index,
        )

    return max(working_entries, key=_rank)


def model_status_rank(status: str | None) -> int:
    normalized = (status or "unknown").strip().lower()
    try:
        return MODEL_STATUS_ORDER.index(normalized)
    except ValueError:
        return len(MODEL_STATUS_ORDER)


def order_model_catalog_entries(
    entries: list[ModelCatalogEntry],
    *,
    active_profile_name: str | None = None,
    active_model_name: str | None = None,
) -> list[ModelCatalogEntry]:
    def _sort_key(entry: ModelCatalogEntry) -> tuple[int, int, int, int, int, str, str]:
        is_exact_active = (
            entry.profile_name == active_profile_name
            and entry.model_name == active_model_name
        )
        is_same_profile = entry.profile_name == active_profile_name
        is_same_model_name = entry.model_name == active_model_name
        return (
            model_status_rank(entry.status),
            0 if is_exact_active else 1,
            0 if is_same_profile else 1,
            0 if is_same_model_name else 1,
            0 if entry.is_active else 1,
            entry.display_name.lower(),
            entry.model_name.lower(),
        )

    ordered = sorted(entries, key=_sort_key)
    return [replace(entry, index=index) for index, entry in enumerate(ordered, start=1)]


def group_model_catalog_entries(
    entries: list[ModelCatalogEntry],
) -> list[tuple[str, list[ModelCatalogEntry]]]:
    grouped: dict[str, list[ModelCatalogEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.status or "unknown", []).append(entry)

    ordered_statuses = [
        status for status in MODEL_STATUS_ORDER if status in grouped
    ]
    ordered_statuses.extend(
        sorted(status for status in grouped if status not in MODEL_STATUS_ORDER)
    )

    return [(status, grouped[status]) for status in ordered_statuses]


def _dedupe_model_ids(model_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    models: list[str] = []

    for model_id in sorted(model_ids, key=str.lower):
        candidate = model_id.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        models.append(candidate)

    return models


def _format_error(exc: Exception) -> str:
    detail = " ".join(str(exc).split())

    if isinstance(exc, APIConnectionError):
        return f"Connection error: {detail or 'Unable to reach the provider.'}"
    if isinstance(exc, RateLimitError):
        return f"Rate limited while listing models: {detail or 'Too many requests.'}"
    if isinstance(exc, APIError):
        return f"Provider error while listing models: {detail or 'Unknown API error.'}"
    return detail or "Failed to list models."


async def discover_models_for_source(source: ModelCatalogSource) -> ModelCatalogResult:
    checked_at = datetime.now(timezone.utc).isoformat()

    if not source.api_key:
        return ModelCatalogResult(
            source=source,
            models=[],
            checked_at=checked_at,
            error=f"No API key is configured for '{source.display_name}'.",
        )

    client = AsyncOpenAI(
        api_key=source.api_key,
        base_url=source.base_url,
        timeout=30.0,
        max_retries=0,
    )

    try:
        response = await client.models.list()
        data = getattr(response, "data", response)
        models = _dedupe_model_ids(
            [
                item.get("id", "")
                if isinstance(item, dict)
                else getattr(item, "id", "")
                for item in data
            ]
        )
        return ModelCatalogResult(
            source=source,
            models=models,
            checked_at=checked_at,
        )
    except Exception as exc:
        return ModelCatalogResult(
            source=source,
            models=[],
            checked_at=checked_at,
            error=_format_error(exc),
        )
    finally:
        await client.close()


async def discover_model_catalog(config: Config) -> list[ModelCatalogResult]:
    sources = build_model_catalog_sources(config)
    return list(await asyncio.gather(*(discover_models_for_source(source) for source in sources)))
