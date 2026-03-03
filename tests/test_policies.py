"""Tests for Layer 4: Policies."""

import pytest

from ganglion.policies.retry import (
    AttemptConfig,
    NoRetry,
    FixedRetry,
    EscalatingRetry,
    ModelEscalationRetry,
)
from ganglion.policies.stall import (
    ConfigComparisonStallDetector,
    OutputHashStallDetector,
)
from ganglion.runtime.types import AgentResult


def make_result(success: bool = False, structured: dict | None = None, raw_text: str = "") -> AgentResult:
    return AgentResult(success=success, structured=structured, raw_text=raw_text)


class TestNoRetry:
    def test_single_attempt(self):
        policy = NoRetry()
        config = policy.configure_attempt(0, None)
        assert config is not None
        assert isinstance(config, AttemptConfig)

    def test_no_second_attempt(self):
        policy = NoRetry()
        config = policy.configure_attempt(1, make_result())
        assert config is None


class TestFixedRetry:
    def test_retries_up_to_max(self):
        policy = FixedRetry(max_attempts=3)
        assert policy.configure_attempt(0, None) is not None
        assert policy.configure_attempt(1, make_result()) is not None
        assert policy.configure_attempt(2, make_result()) is not None
        assert policy.configure_attempt(3, make_result()) is None

    def test_default_max(self):
        policy = FixedRetry()
        assert policy.configure_attempt(2, None) is not None
        assert policy.configure_attempt(3, None) is None


class TestEscalatingRetry:
    def test_temperature_increases(self):
        policy = EscalatingRetry(max_attempts=3, base_temp=0.1, temp_step=0.2)
        c0 = policy.configure_attempt(0, None)
        c1 = policy.configure_attempt(1, make_result())
        c2 = policy.configure_attempt(2, make_result())
        assert c0.temperature == pytest.approx(0.1)
        assert c1.temperature == pytest.approx(0.3)
        assert c2.temperature == pytest.approx(0.5)

    def test_stops_after_max(self):
        policy = EscalatingRetry(max_attempts=2)
        assert policy.configure_attempt(2, make_result()) is None

    def test_stops_on_success(self):
        policy = EscalatingRetry(max_attempts=5)
        # Success is not retryable
        config = policy.configure_attempt(1, make_result(success=True))
        assert config is None

    def test_stall_detection(self):
        detector = ConfigComparisonStallDetector(
            extract_config=lambda r: r.structured or {}
        )
        policy = EscalatingRetry(max_attempts=5, stall_detector=detector)

        result1 = make_result(structured={"lr": 0.01})
        config1 = policy.configure_attempt(1, result1)
        assert config1.extra_system_context is None

        # Same config again => stall detected
        result2 = make_result(structured={"lr": 0.01})
        config2 = policy.configure_attempt(2, result2)
        assert config2.extra_system_context is not None
        assert "CRITICAL" in config2.extra_system_context


class TestModelEscalationRetry:
    def test_model_ladder(self):
        policy = ModelEscalationRetry(
            model_ladder=["haiku", "sonnet", "opus"],
            attempts_per_model=2,
        )
        c0 = policy.configure_attempt(0, None)
        c1 = policy.configure_attempt(1, make_result())
        c2 = policy.configure_attempt(2, make_result())
        c3 = policy.configure_attempt(3, make_result())
        c4 = policy.configure_attempt(4, make_result())

        assert c0.model == "haiku"
        assert c1.model == "haiku"
        assert c2.model == "sonnet"
        assert c3.model == "sonnet"
        assert c4.model == "opus"

    def test_exhausted(self):
        policy = ModelEscalationRetry(
            model_ladder=["haiku"],
            attempts_per_model=1,
        )
        assert policy.configure_attempt(0, None) is not None
        assert policy.configure_attempt(1, make_result()) is None


class TestConfigComparisonStallDetector:
    def test_no_stall_on_different_configs(self):
        detector = ConfigComparisonStallDetector(
            extract_config=lambda r: r.structured or {}
        )
        assert detector.is_stalled(0, make_result(structured={"a": 1})) is False
        assert detector.is_stalled(1, make_result(structured={"a": 2})) is False

    def test_stall_on_same_config(self):
        detector = ConfigComparisonStallDetector(
            extract_config=lambda r: r.structured or {}
        )
        assert detector.is_stalled(0, make_result(structured={"a": 1})) is False
        assert detector.is_stalled(1, make_result(structured={"a": 1})) is True

    def test_divergence_prompt(self):
        detector = ConfigComparisonStallDetector(
            extract_config=lambda r: r.structured or {}
        )
        prompt = detector.divergence_prompt()
        assert "CRITICAL" in prompt

    def test_reset(self):
        detector = ConfigComparisonStallDetector(
            extract_config=lambda r: r.structured or {}
        )
        detector.is_stalled(0, make_result(structured={"a": 1}))
        detector.reset()
        # Should not detect stall after reset
        assert detector.is_stalled(0, make_result(structured={"a": 1})) is False


class TestOutputHashStallDetector:
    def test_no_stall_on_different_outputs(self):
        detector = OutputHashStallDetector(max_repeats=2)
        assert detector.is_stalled(0, make_result(raw_text="output 1")) is False
        assert detector.is_stalled(1, make_result(raw_text="output 2")) is False

    def test_stall_after_max_repeats(self):
        detector = OutputHashStallDetector(max_repeats=1)
        assert detector.is_stalled(0, make_result(raw_text="same")) is False
        assert detector.is_stalled(1, make_result(raw_text="same")) is True
