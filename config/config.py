from __future__ import annotations
from enum import Enum
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "nvidia/nemotron-3-super-120b-a12b:free"
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256_000


class ModelProfileConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None

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

        return self

    def get_model_profile(self, name: str) -> ModelProfileConfig | None:
        return self.models.get(name)

    @property
    def active_profile(self) -> ModelProfileConfig | None:
        if not self.active_model_profile:
            return None
        return self.models.get(self.active_model_profile)

    @property
    def current_model_config(self) -> ModelConfig:
        profile = self.active_profile
        if profile:
            return profile.model
        return self.model

    @property
    def api_key(self) -> str | None:
        profile = self.active_profile
        if profile:
            resolved = profile.resolved_api_key
            if resolved:
                return resolved
        return os.environ.get("API_KEY")

    @property
    def base_url(self) -> str | None:
        profile = self.active_profile
        if profile and profile.base_url:
            return profile.base_url
        return os.environ.get("BASE_URL")

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
    def api_key_source_label(self) -> str:
        profile = self.active_profile
        if profile:
            if profile.api_key:
                return f"profile:{self.active_model_profile}.api_key"
            if profile.api_key_env and os.environ.get(profile.api_key_env):
                return f"env:{profile.api_key_env}"
        if os.environ.get("API_KEY"):
            return "env:API_KEY"
        if profile and profile.api_key_env:
            return f"missing:{profile.api_key_env}"
        return "missing"

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

        if not self.api_key:
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

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
