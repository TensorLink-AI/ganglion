"""Centralized configuration — all env vars and settings in one place.

Every configurable value must be defined here. No scattered os.getenv() calls.
Validates all required variables at import time and fails fast with a clear message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env file (if present) so env vars are available before from_env() runs.
load_dotenv()


@dataclass(frozen=True)
class GanglionConfig:
    """Immutable configuration loaded from environment variables."""

    # LLM
    llm_provider_api_key: str = ""
    llm_provider_base_url: str = ""
    llm_model: str = "gpt-4o"
    llm_max_retries: int = 5
    llm_base_delay: float = 1.0
    llm_max_delay: float = 60.0
    llm_request_timeout: float = 120.0

    # Server
    server_host: str = "127.0.0.1"
    server_port: int = 8899

    # CORS
    cors_allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])

    # Rate limiting
    rate_limit_requests_per_minute: int = 60

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Knowledge store
    max_patterns: int = 500
    max_antipatterns: int = 500

    # Request limits
    max_request_body_bytes: int = 10 * 1024 * 1024  # 10MB

    # Compute backends
    basilica_token: str = ""

    # MCP server (outbound — expose Ganglion tools via MCP)
    mcp_server_enabled: bool = False
    mcp_server_transport: str = "stdio"  # "stdio" or "sse"
    mcp_server_sse_port: int = 8900

    @classmethod
    def from_env(cls) -> GanglionConfig:
        """Load configuration from environment variables.

        Uses GANGLION_ prefix for all variables.
        """

        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"GANGLION_{key}", default)

        def _get_int(key: str, default: int) -> int:
            raw = _get(key, str(default))
            try:
                return int(raw)
            except ValueError:
                return default

        def _get_float(key: str, default: float) -> float:
            raw = _get(key, str(default))
            try:
                return float(raw)
            except ValueError:
                return default

        def _get_list(key: str, default: str = "") -> list[str]:
            raw = _get(key, default)
            if not raw:
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]

        llm_provider_api_key = os.environ.get("LLM_PROVIDER_API_KEY", "")
        llm_provider_base_url = os.environ.get("LLM_PROVIDER_BASE_URL", "")

        # Default to OPEN-API-KEY when using Chutes
        if "chutes" in llm_provider_base_url.lower() and not llm_provider_api_key:
            llm_provider_api_key = "OPEN-API-KEY"

        return cls(
            llm_provider_api_key=llm_provider_api_key,
            llm_provider_base_url=llm_provider_base_url,
            llm_model=_get("LLM_MODEL", "gpt-4o"),
            llm_max_retries=_get_int("LLM_MAX_RETRIES", 5),
            llm_base_delay=_get_float("LLM_BASE_DELAY", 1.0),
            llm_max_delay=_get_float("LLM_MAX_DELAY", 60.0),
            llm_request_timeout=_get_float("LLM_REQUEST_TIMEOUT", 120.0),
            server_host=_get("HOST", "127.0.0.1"),
            server_port=_get_int("PORT", 8899),
            cors_allowed_origins=_get_list("CORS_ORIGINS", "http://localhost:3000"),
            rate_limit_requests_per_minute=_get_int("RATE_LIMIT_RPM", 60),
            log_level=_get("LOG_LEVEL", "INFO"),
            log_format=_get("LOG_FORMAT", "json"),
            max_patterns=_get_int("MAX_PATTERNS", 500),
            max_antipatterns=_get_int("MAX_ANTIPATTERNS", 500),
            max_request_body_bytes=_get_int("MAX_REQUEST_BODY_BYTES", 10 * 1024 * 1024),
            basilica_token=_get("BASILICA_TOKEN", ""),
            mcp_server_enabled=_get("MCP_SERVER_ENABLED", "").lower() in ("1", "true", "yes"),
            mcp_server_transport=_get("MCP_SERVER_TRANSPORT", "stdio"),
            mcp_server_sse_port=_get_int("MCP_SERVER_SSE_PORT", 8900),
        )

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors (empty = valid)."""
        errors: list[str] = []

        if self.llm_max_retries < 0:
            errors.append("GANGLION_LLM_MAX_RETRIES must be >= 0")
        if self.llm_base_delay <= 0:
            errors.append("GANGLION_LLM_BASE_DELAY must be > 0")
        if self.llm_request_timeout <= 0:
            errors.append("GANGLION_LLM_REQUEST_TIMEOUT must be > 0")
        if self.server_port < 1 or self.server_port > 65535:
            errors.append("GANGLION_PORT must be between 1 and 65535")
        if self.rate_limit_requests_per_minute < 1:
            errors.append("GANGLION_RATE_LIMIT_RPM must be >= 1")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(
                "GANGLION_LOG_LEVEL must be one of"
                f" DEBUG/INFO/WARNING/ERROR/CRITICAL, got '{self.log_level}'"
            )
        if self.max_request_body_bytes < 1024:
            errors.append("GANGLION_MAX_REQUEST_BODY_BYTES must be >= 1024")
        if self.mcp_server_transport not in ("stdio", "sse"):
            errors.append(
                "GANGLION_MCP_SERVER_TRANSPORT must be 'stdio' or 'sse',"
                f" got '{self.mcp_server_transport}'"
            )
        if self.mcp_server_sse_port < 1 or self.mcp_server_sse_port > 65535:
            errors.append("GANGLION_MCP_SERVER_SSE_PORT must be between 1 and 65535")

        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ValueError with all errors if invalid."""
        errors = self.validate()
        if errors:
            msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)
