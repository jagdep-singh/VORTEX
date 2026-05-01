from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

LOCAL_API_KEY_PLACEHOLDER = "local-no-auth-required"
_LOCAL_HOST_ALIASES = {
    "localhost",
    "0.0.0.0",
    "::1",
    "host.docker.internal",
}


def _normalize_base_url(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().rstrip("/")


def is_gemini_openai_compat_url(base_url: str | None) -> bool:
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return False

    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    return (
        hostname.endswith("generativelanguage.googleapis.com")
        and "/openai" in path
    )


def base_url_allows_missing_api_key(base_url: str | None) -> bool:
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return False

    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False

    if hostname in _LOCAL_HOST_ALIASES or hostname.endswith(".localhost"):
        return True

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return address.is_loopback or address.is_private or address.is_link_local


def resolve_client_api_key(api_key: str | None, base_url: str | None) -> str | None:
    if api_key:
        return api_key
    if base_url_allows_missing_api_key(base_url):
        return LOCAL_API_KEY_PLACEHOLDER
    return None
