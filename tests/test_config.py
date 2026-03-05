"""Tests for the centralized configuration module."""

import json
import os

import pytest

from ganglion.config import GanglionConfig, LLMBackendConfig


class TestGanglionConfig:
    def test_defaults(self):
        config = GanglionConfig()
        assert config.llm_model == "gpt-4o"
        assert config.server_host == "127.0.0.1"
        assert config.server_port == 8899
        assert config.log_level == "INFO"
        assert config.rate_limit_requests_per_minute == 60

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("GANGLION_LLM_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("GANGLION_PORT", "9000")
        monkeypatch.setenv("GANGLION_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GANGLION_CORS_ORIGINS", "http://a.com,http://b.com")

        config = GanglionConfig.from_env()
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.server_port == 9000
        assert config.log_level == "DEBUG"
        assert config.cors_allowed_origins == ["http://a.com", "http://b.com"]

    def test_validate_valid(self):
        config = GanglionConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_port(self):
        config = GanglionConfig(server_port=99999)
        errors = config.validate()
        assert any("PORT" in e for e in errors)

    def test_validate_invalid_log_level(self):
        config = GanglionConfig(log_level="VERBOSE")
        errors = config.validate()
        assert any("LOG_LEVEL" in e for e in errors)

    def test_validate_invalid_rate_limit(self):
        config = GanglionConfig(rate_limit_requests_per_minute=0)
        errors = config.validate()
        assert any("RATE_LIMIT" in e for e in errors)

    def test_validate_or_raise(self):
        config = GanglionConfig(server_port=-1)
        with pytest.raises(ValueError, match="Configuration validation failed"):
            config.validate_or_raise()

    def test_validate_or_raise_valid(self):
        config = GanglionConfig()
        config.validate_or_raise()  # Should not raise

    def test_from_env_defaults(self, monkeypatch):
        # Clear any env vars that might be set
        for key in list(os.environ.keys()):
            if key.startswith("GANGLION_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("LLM_PROVIDER_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER_BASE_URL", raising=False)

        config = GanglionConfig.from_env()
        assert config.llm_model == "gpt-4o"
        assert config.server_port == 8899

    def test_frozen(self):
        config = GanglionConfig()
        with pytest.raises(AttributeError):
            config.server_port = 9000  # type: ignore[misc]

    def test_default_backend_always_created(self, monkeypatch):
        for key in list(os.environ.keys()):
            if key.startswith("GANGLION_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("LLM_PROVIDER_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER_BASE_URL", raising=False)

        config = GanglionConfig.from_env()
        assert "default" in config.llm_backends
        assert config.llm_backends["default"].model == "gpt-4o"

    def test_default_backend_uses_legacy_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_API_KEY", "sk-test-123")
        monkeypatch.setenv("LLM_PROVIDER_BASE_URL", "https://custom.api.com/v1")
        monkeypatch.setenv("GANGLION_LLM_MODEL", "gpt-3.5-turbo")

        config = GanglionConfig.from_env()
        default = config.llm_backends["default"]
        assert default.api_key == "sk-test-123"
        assert default.base_url == "https://custom.api.com/v1"
        assert default.model == "gpt-3.5-turbo"

    def test_named_backends_from_json(self, monkeypatch):
        backends = {
            "fast": {"api_key": "sk-fast", "model": "gpt-4o-mini"},
            "reasoning": {
                "api_key": "sk-reason",
                "base_url": "https://reason.api.com/v1",
                "model": "o1",
            },
        }
        monkeypatch.setenv("GANGLION_LLM_BACKENDS", json.dumps(backends))

        config = GanglionConfig.from_env()
        assert "fast" in config.llm_backends
        assert "reasoning" in config.llm_backends
        assert "default" in config.llm_backends

        fast = config.llm_backends["fast"]
        assert fast.api_key == "sk-fast"
        assert fast.model == "gpt-4o-mini"
        assert fast.name == "fast"

        reasoning = config.llm_backends["reasoning"]
        assert reasoning.base_url == "https://reason.api.com/v1"
        assert reasoning.model == "o1"

    def test_named_backends_inherit_retry_defaults(self, monkeypatch):
        monkeypatch.setenv("GANGLION_LLM_MAX_RETRIES", "10")
        monkeypatch.setenv(
            "GANGLION_LLM_BACKENDS",
            json.dumps({"fast": {"api_key": "sk-fast", "model": "gpt-4o-mini"}}),
        )

        config = GanglionConfig.from_env()
        assert config.llm_backends["fast"].max_retries == 10

    def test_invalid_backends_json_ignored(self, monkeypatch):
        monkeypatch.setenv("GANGLION_LLM_BACKENDS", "not-valid-json")
        config = GanglionConfig.from_env()
        # Should still have the default backend
        assert "default" in config.llm_backends
        assert len(config.llm_backends) == 1


class TestLLMBackendConfig:
    def test_frozen(self):
        cfg = LLMBackendConfig(name="test")
        with pytest.raises(AttributeError):
            cfg.api_key = "new"  # type: ignore[misc]

    def test_defaults(self):
        cfg = LLMBackendConfig(name="test")
        assert cfg.model == "gpt-4o"
        assert cfg.max_retries == 5
        assert cfg.api_key == ""
        assert cfg.base_url == ""
