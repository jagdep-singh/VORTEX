from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _dedupe_model_ids(model_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for model_id in model_ids:
        candidate = model_id.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)

    return result


def _parse_lines(text: str) -> list[str]:
    return _dedupe_model_ids(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_model_catalog_text(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_lines(text)

    if isinstance(parsed, dict):
        data = parsed.get("data")
        if isinstance(data, list):
            return _dedupe_model_ids(
                item.get("id", "")
                for item in data
                if isinstance(item, dict)
            )

    if isinstance(parsed, list):
        return _dedupe_model_ids(
            item if isinstance(item, str) else item.get("id", "")
            for item in parsed
            if isinstance(item, (str, dict))
        )

    return _parse_lines(text)


def load_model_catalog(paths: Iterable[Path]) -> list[str]:
    collected: list[str] = []

    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        collected.extend(parse_model_catalog_text(text))

    return _dedupe_model_ids(collected)
