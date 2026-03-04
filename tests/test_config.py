"""Tests for the centralized configuration module."""

import os

import pytest

from ganglion.config import GanglionConfig


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
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        config = GanglionConfig.from_env()
        assert config.llm_model == "gpt-4o"
        assert config.server_port == 8899

    def test_frozen(self):
        config = GanglionConfig()
        with pytest.raises(AttributeError):
            config.server_port = 9000  # type: ignore[misc]
