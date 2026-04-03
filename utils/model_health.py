from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from config.loader import get_data_dir


@dataclass
class ModelHealthRecord:
    provider: str
    model: str
    status: str
    checked_at: str
    detail: str | None = None

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        model: str,
        status: str,
        detail: str | None = None,
    ) -> ModelHealthRecord:
        return cls(
            provider=provider,
            model=model,
            status=status,
            checked_at=datetime.now(timezone.utc).isoformat(),
            detail=detail,
        )

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> ModelHealthRecord:
        return cls(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            status=str(data.get("status", "unknown")),
            checked_at=str(data.get("checked_at", "")),
            detail=data.get("detail"),
        )


def classify_error_text(error_text: str) -> tuple[str, str]:
    detail = " ".join(error_text.strip().split())
    lowered = detail.lower()

    if not detail:
        return "error", "Unknown error"
    if "insufficient_quota" in lowered:
        return "quota", detail
    if "more credits" in lowered or "upgrade to a paid account" in lowered:
        return "quota", detail
    if "exceeded your current quota" in lowered:
        return "quota", detail
    if "error code: 402" in lowered:
        return "quota", detail
    if "error code: 401" in lowered or "unauthorized" in lowered:
        return "auth-error", detail
    if "user not found" in lowered:
        return "auth-error", detail
    if "error code: 404" in lowered or "not found" in lowered:
        return "unavailable", detail
    if "does not exist" in lowered or "unknown model" in lowered:
        return "unavailable", detail
    if "rate limit" in lowered or "error code: 429" in lowered:
        return "rate-limited", detail
    if "connection error" in lowered:
        return "offline", detail

    return "error", detail


class ModelHealthStore:
    def __init__(self) -> None:
        self.data_dir = get_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "model_health.json"
        self._records = self._load()

    def _key(self, provider: str, model: str) -> str:
        return f"{provider}:{model}"

    def _load(self) -> dict[str, ModelHealthRecord]:
        if not self.file_path.exists():
            return {}

        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        if not isinstance(data, dict):
            return {}

        records: dict[str, ModelHealthRecord] = {}
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            try:
                records[key] = ModelHealthRecord.from_dict(value)
            except Exception:
                continue
        return records

    def _save(self) -> None:
        payload = {key: asdict(record) for key, record in self._records.items()}
        try:
            self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.chmod(self.file_path, 0o600)
        except OSError:
            return

    def get(self, provider: str, model: str) -> ModelHealthRecord | None:
        return self._records.get(self._key(provider, model))

    def get_many(
        self,
        provider: str,
        models: list[str],
    ) -> dict[str, ModelHealthRecord]:
        result: dict[str, ModelHealthRecord] = {}
        for model in models:
            record = self.get(provider, model)
            if record:
                result[model] = record
        return result

    def save_record(self, record: ModelHealthRecord) -> None:
        self._records[self._key(record.provider, record.model)] = record
        self._save()

    def save_records(self, records: list[ModelHealthRecord]) -> None:
        for record in records:
            self._records[self._key(record.provider, record.model)] = record
        self._save()


class ModelHealthChecker:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str | None,
        store: ModelHealthStore | None = None,
        timeout_sec: float = 20.0,
        concurrency: int = 4,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.store = store or ModelHealthStore()
        self.timeout_sec = timeout_sec
        self.concurrency = max(1, concurrency)

    async def probe_models(
        self,
        models: list[str],
        *,
        progress_callback: Callable[[int, int, ModelHealthRecord], None | Awaitable[None]]
        | None = None,
    ) -> list[ModelHealthRecord]:
        if not models:
            return []

        if not self.api_key:
            records = [
                ModelHealthRecord.create(
                    provider=self.provider,
                    model=model,
                    status="missing-key",
                    detail="No API key is configured for this provider.",
                )
                for model in models
            ]
            self.store.save_records(records)
            return records

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_sec,
            max_retries=0,
        )
        semaphore = asyncio.Semaphore(self.concurrency)
        records: list[ModelHealthRecord | None] = [None] * len(models)
        total = len(models)

        async def probe_one(index: int, model: str) -> None:
            async with semaphore:
                record = await self._probe_model(client, model)
                records[index] = record
                self.store.save_record(record)
                if progress_callback:
                    callback_result = progress_callback(index + 1, total, record)
                    if inspect.isawaitable(callback_result):
                        await callback_result

        try:
            await asyncio.gather(
                *(probe_one(index, model) for index, model in enumerate(models))
            )
        finally:
            await client.close()

        return [record for record in records if record is not None]

    async def probe_model(self, model: str) -> ModelHealthRecord:
        if not model:
            record = ModelHealthRecord.create(
                provider=self.provider,
                model="",
                status="error",
                detail="No model was specified.",
            )
            self.store.save_record(record)
            return record

        if not self.api_key:
            record = ModelHealthRecord.create(
                provider=self.provider,
                model=model,
                status="missing-key",
                detail="No API key is configured for this provider.",
            )
            self.store.save_record(record)
            return record

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_sec,
            max_retries=0,
        )

        try:
            record = await self._probe_model(client, model)
            self.store.save_record(record)
            return record
        finally:
            await client.close()

    async def _probe_model(
        self,
        client: AsyncOpenAI,
        model: str,
    ) -> ModelHealthRecord:
        try:
            await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with OK.",
                    }
                ],
                max_tokens=1,
            )
            return ModelHealthRecord.create(
                provider=self.provider,
                model=model,
                status="working",
            )
        except APIConnectionError as exc:
            return ModelHealthRecord.create(
                provider=self.provider,
                model=model,
                status="offline",
                detail=" ".join(str(exc).split()),
            )
        except RateLimitError as exc:
            status, detail = classify_error_text(str(exc))
            return ModelHealthRecord.create(
                provider=self.provider,
                model=model,
                status=status,
                detail=detail,
            )
        except APIError as exc:
            status, detail = classify_error_text(str(exc))
            return ModelHealthRecord.create(
                provider=self.provider,
                model=model,
                status=status,
                detail=detail,
            )
        except Exception as exc:
            return ModelHealthRecord.create(
                provider=self.provider,
                model=model,
                status="error",
                detail=" ".join(str(exc).split()),
            )
