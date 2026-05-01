from __future__ import annotations
from enum import Enum
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator
from utils.provider_auth import base_url_allows_missing_api_key, resolve_client_api_key


class ModelConfig(BaseModel):
    name: str = "openrouter/free"
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256_000
    max_output_tokens: int = Field(default=8192, ge=256)


class GeminiThinkingConfig(BaseModel):
    include_thoughts: bool | None = None
    thinking_budget: int | str | None = None
    thinking_level: str | None = None

    @model_validator(mode="after")
    def validate_thinking_controls(self) -> GeminiThinkingConfig:
        if self.thinking_budget is not None and self.thinking_level:
            raise ValueError(
                "Gemini thinking_config cannot set both thinking_budget and thinking_level."
            )
        return self


class GeminiCompatConfig(BaseModel):
    reasoning_effort: str | None = None
    cached_content: str | None = None
    thinking_config: GeminiThinkingConfig = Field(
        default_factory=GeminiThinkingConfig
    )

    @model_validator(mode="after")
    def validate_reasoning_controls(self) -> GeminiCompatConfig:
        if self.reasoning_effort and (
            self.thinking_config.thinking_budget is not None
            or self.thinking_config.thinking_level
        ):
            raise ValueError(
                "Gemini reasoning_effort cannot be combined with thinking_budget or thinking_level."
            )
        return self

    def build_request_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if self.reasoning_effort:
            overrides["reasoning_effort"] = self.reasoning_effort

        google: dict[str, Any] = {}
        if self.cached_content:
            google["cached_content"] = self.cached_content

        thinking: dict[str, Any] = {}
        if self.thinking_config.include_thoughts is not None:
            thinking["include_thoughts"] = self.thinking_config.include_thoughts
        if self.thinking_config.thinking_budget is not None:
            thinking["thinking_budget"] = self.thinking_config.thinking_budget
        if self.thinking_config.thinking_level:
            thinking["thinking_level"] = self.thinking_config.thinking_level
        if thinking:
            google["thinking_config"] = thinking

        if google:
            overrides["extra_body"] = {"google": google}

        return overrides


class ModelProfileConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    gemini: GeminiCompatConfig = Field(default_factory=GeminiCompatConfig)

    @property
    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    @property
    def key_source_label(self) -> str:
        if self.api_key:
            return "inline"
        if self.api_key_env:
            return f"env:{self.api_key_env}"
        return "env:API_KEY"


class ShellEnvironmentPolicy(BaseModel):
    ignore_default_excludes: bool = False
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*KEY*", "*TOKEN*", "*SECRET*"]
    )
    set_vars: dict[str, str] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    enabled: bool = True
    startup_timeout_sec: float = 10

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None

    # http/sse transport
    url: str | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        has_command = self.command is not None
        has_url = self.url is not None

        if not has_command and not has_url:
            raise ValueError(
                "MCP Server must have either 'command' (stdio) or 'url' (http/sse)"
            )

        if has_command and has_url:
            raise ValueError(
                "MCP Server cannot have both 'command' (stdio) and 'url' (http/sse)"
            )

        return self


class ApprovalPolicy(str, Enum):
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    AUTO = "auto"
    AUTO_EDIT = "auto-edit"
    NEVER = "never"
    YOLO = "yolo"


class HookTrigger(str, Enum):
    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_ERROR = "on_error"


class HookConfig(BaseModel):
    name: str
    trigger: HookTrigger
    command: str | None = None  # python3 tests.py
    script: str | None = None  # *.sh
    timeout_sec: float = 30
    enabled: bool = True

    @model_validator(mode="after")
    def validate_hook(self) -> HookConfig:
        if not self.command and not self.script:
            raise ValueError("Hook must either have 'command' or 'script'")
        return self


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)

    def _workspace():
        path = Path.cwd() / "workspace"
        path.mkdir(exist_ok=True)
        return path

    cwd: Path = Field(default_factory=_workspace)
    active_model_profile: str | None = None
    models: dict[str, ModelProfileConfig] = Field(default_factory=dict)

    shell_environment: ShellEnvironmentPolicy = Field(
        default_factory=ShellEnvironmentPolicy
    )
    hooks_enabled: bool = False
    hooks: list[HookConfig] = Field(default_factory=list)
    approval: ApprovalPolicy = ApprovalPolicy.ON_REQUEST
    max_turns: int = 100
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    allowed_tools: list[str] | None = Field(
        None,
        description="If set, only these tools will be available to the agent",
    )

    developer_instructions: str | None = None
    user_instructions: str | None = None

    debug: bool = False

    @model_validator(mode="after")
    def validate_model_profiles(self) -> Config:
        if self.active_model_profile and self.active_model_profile not in self.models:
            raise ValueError(
                f"Unknown active_model_profile: {self.active_model_profile}"
            )

        # If the user defined provider profiles and didn't choose one explicitly,
        # prefer the first profile when no legacy API_KEY is present.
        if self.active_model_profile is None and self.models and not os.environ.get(
            "API_KEY"
        ):
            self.active_model_profile = next(iter(self.models))

        if self.active_model_profile is None and os.environ.get("MODEL_NAME"):
            self.model.name = os.environ["MODEL_NAME"]

        return self

    def get_model_profile(self, name: str) -> ModelProfileConfig | None:
        return self.models.get(name)

    def ensure_model_profile(
        self,
        name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> ModelProfileConfig:
        profile = self.models.get(name)
        if profile is None:
            profile = ModelProfileConfig(
                base_url=base_url,
                api_key=api_key,
                api_key_env=api_key_env,
            )
            self.models[name] = profile
            return profile

        if base_url and not profile.base_url:
            profile.base_url = base_url
        if api_key and not profile.api_key:
            profile.api_key = api_key
        if api_key_env and not profile.api_key_env:
            profile.api_key_env = api_key_env
        return profile

    @property
    def active_profile(self) -> ModelProfileConfig | None:
        if not self.active_model_profile:
            return None
        return self.models.get(self.active_model_profile)

    def resolve_profile_api_key(self, profile_name: str | None) -> str | None:
        if profile_name is None:
            return os.environ.get("API_KEY")

        profile = self.models.get(profile_name)
        if profile is None:
            return None

        resolved = profile.resolved_api_key
        if resolved:
            return resolved
        if profile.api_key or profile.api_key_env:
            return None
        return os.environ.get("API_KEY")

    def resolve_profile_base_url(self, profile_name: str | None) -> str:
        if profile_name is None:
            return os.environ.get("BASE_URL") or "https://api.openai.com/v1"

        profile = self.models.get(profile_name)
        if profile and profile.base_url:
            return profile.base_url
        return os.environ.get("BASE_URL") or "https://api.openai.com/v1"

    def resolve_profile_key_source_label(self, profile_name: str | None) -> str:
        base_url = self.resolve_profile_base_url(profile_name)

        if profile_name is None:
            if os.environ.get("API_KEY"):
                return "env:API_KEY"
            if base_url_allows_missing_api_key(base_url):
                return "local:no-key-required"
            return "missing:API_KEY"

        profile = self.models.get(profile_name)
        if profile is None:
            return "missing"

        if profile.api_key:
            return f"profile:{profile_name}.api_key"

        if profile.api_key_env:
            if os.environ.get(profile.api_key_env):
                return f"env:{profile.api_key_env}"
            if base_url_allows_missing_api_key(base_url):
                return "local:no-key-required"
            return f"missing:{profile.api_key_env}"

        if os.environ.get("API_KEY"):
            return "env:API_KEY"
        if base_url_allows_missing_api_key(base_url):
            return "local:no-key-required"
        return "missing:API_KEY"

    @property
    def current_model_config(self) -> ModelConfig:
        profile = self.active_profile
        if profile:
            return profile.model
        return self.model

    @property
    def api_key(self) -> str | None:
        return self.resolve_profile_api_key(self.active_model_profile)

    @property
    def client_api_key(self) -> str | None:
        return self.resolve_profile_client_api_key(self.active_model_profile)

    @property
    def base_url(self) -> str | None:
        return self.resolve_profile_base_url(self.active_model_profile)

    @property
    def model_name(self) -> str:
        return self.current_model_config.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.current_model_config.name = value

    @property
    def temperature(self) -> float:
        return self.current_model_config.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.current_model_config.temperature = value

    @property
    def max_output_tokens(self) -> int:
        return self.current_model_config.max_output_tokens

    @max_output_tokens.setter
    def max_output_tokens(self, value: int) -> None:
        self.current_model_config.max_output_tokens = value

    @property
    def api_key_source_label(self) -> str:
        return self.resolve_profile_key_source_label(self.active_model_profile)

    @property
    def active_model_label(self) -> str:
        if self.active_model_profile:
            return f"{self.active_model_profile} ({self.model_name})"
        return self.model_name

    def switch_model_profile(self, profile_name: str) -> None:
        if profile_name not in self.models:
            raise ValueError(f"Unknown model profile: {profile_name}")
        self.active_model_profile = profile_name

    def list_model_profiles(self) -> list[tuple[str, ModelProfileConfig]]:
        return list(self.models.items())

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key and self.requires_api_key_for_profile(
            self.active_model_profile
        ):
            if self.active_model_profile:
                errors.append(
                    "No API key found for active model profile "
                    f"'{self.active_model_profile}'. Configure api_key/api_key_env "
                    "for that profile or set the matching environment variable."
                )
            else:
                errors.append(
                    "No API key found. Set API_KEY or configure "
                    "[models.<name>] with api_key/api_key_env."
                )

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors

    def allows_missing_api_key_for_profile(self, profile_name: str | None) -> bool:
        return base_url_allows_missing_api_key(
            self.resolve_profile_base_url(profile_name)
        )

    def requires_api_key_for_profile(self, profile_name: str | None) -> bool:
        return not self.allows_missing_api_key_for_profile(profile_name)

    def resolve_profile_client_api_key(self, profile_name: str | None) -> str | None:
        return resolve_client_api_key(
            self.resolve_profile_api_key(profile_name),
            self.resolve_profile_base_url(profile_name),
        )

    def profile_uses_gemini_openai_compat(self, profile_name: str | None) -> bool:
        from utils.provider_auth import is_gemini_openai_compat_url

        return is_gemini_openai_compat_url(
            self.resolve_profile_base_url(profile_name)
        )

    def resolve_profile_request_overrides(
        self,
        profile_name: str | None,
    ) -> dict[str, Any]:
        if not self.profile_uses_gemini_openai_compat(profile_name):
            return {}

        profile = self.models.get(profile_name) if profile_name is not None else None
        if profile is None:
            return {}

        return profile.gemini.build_request_overrides()

    @property
    def request_overrides(self) -> dict[str, Any]:
        return self.resolve_profile_request_overrides(self.active_model_profile)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
