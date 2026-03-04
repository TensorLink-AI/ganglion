"""Error taxonomy for the framework."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all framework errors."""

    is_retryable: bool = True


class EnvironmentError(AgentError):
    """Missing dependency, broken import, misconfiguration. NOT retryable."""

    is_retryable = False


class InfrastructureError(AgentError):
    """GPU timeout, API rate limit, network failure. Retryable — transient issue."""

    is_retryable = True


class ValidationError(AgentError):
    """Output doesn't match expected spec. Retryable — agent might succeed on retry."""

    is_retryable = True


class StallError(AgentError):
    """Agent producing the same failing output repeatedly. Retryable with escalation."""

    is_retryable = True


class PipelineValidationError(AgentError):
    """Pipeline definition is invalid. NOT retryable."""

    is_retryable = False


class PipelineOperationError(AgentError):
    """Error applying a pipeline mutation operation."""

    is_retryable = False


class ConcurrentMutationError(AgentError):
    """Attempted mutation while pipeline is running."""

    is_retryable = False


class ToolAlreadyRegisteredError(AgentError):
    """Tool name is already registered."""

    is_retryable = False


class ToolNotFoundError(AgentError):
    """Tool not found in registry."""

    is_retryable = False


class AgentNotFoundError(AgentError):
    """Agent not found in registry."""

    is_retryable = False


class AgentValidationError(AgentError):
    """Agent class did not pass validation."""

    is_retryable = False
