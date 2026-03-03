"""Error taxonomy for the framework."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all framework errors."""

    retryable: bool = True


class EnvironmentError(AgentError):
    """Missing dependency, broken import, misconfiguration. NOT retryable."""

    retryable = False


class InfrastructureError(AgentError):
    """GPU timeout, API rate limit, network failure. Retryable — transient issue."""

    retryable = True


class ValidationError(AgentError):
    """Output doesn't match expected spec. Retryable — agent might succeed on retry."""

    retryable = True


class StallError(AgentError):
    """Agent producing the same failing output repeatedly. Retryable with escalation."""

    retryable = True


class PipelineValidationError(AgentError):
    """Pipeline definition is invalid. NOT retryable."""

    retryable = False


class PipelineOperationError(AgentError):
    """Error applying a pipeline mutation operation."""

    retryable = False


class ConcurrentMutationError(AgentError):
    """Attempted mutation while pipeline is running."""

    retryable = False


class ToolAlreadyRegisteredError(AgentError):
    """Tool name is already registered."""

    retryable = False


class ToolNotFoundError(AgentError):
    """Tool not found in registry."""

    retryable = False


class AgentNotFoundError(AgentError):
    """Agent not found in registry."""

    retryable = False


class AgentValidationError(AgentError):
    """Agent class did not pass validation."""

    retryable = False
