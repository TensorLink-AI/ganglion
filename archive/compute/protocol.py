"""ComputeBackend protocol — the contract all compute providers implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class JobStatus(Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class JobSpec:
    """What to run and what it needs."""

    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    gpu_type: str | None = None
    gpu_count: int = 0
    cpu_cores: int = 2
    memory_gb: int = 8
    timeout_seconds: int = 3600
    artifacts_dir: str = "/outputs"
    upload_paths: list[str] = field(default_factory=list)


@dataclass
class DockerPrefab:
    """Named container template for pipeline stages.

    Subnet authors declare these in SubnetConfig to provide pre-built
    container images. A prefab seeds a JobSpec — the command is supplied
    at call time and overrides take precedence.
    """

    name: str
    image: str
    env: dict[str, str] = field(default_factory=dict)
    gpu_type: str | None = None
    gpu_count: int = 0
    cpu_cores: int = 2
    memory_gb: int = 8
    timeout_seconds: int = 3600
    artifacts_dir: str = "/outputs"

    def to_job_spec(self, command: list[str], **overrides: Any) -> JobSpec:
        """Create a JobSpec from this prefab with optional overrides."""
        return JobSpec(
            image=overrides.get("image", self.image),
            command=command,
            env={**self.env, **overrides.get("env", {})},
            gpu_type=overrides.get("gpu_type", self.gpu_type),
            gpu_count=overrides.get("gpu_count", self.gpu_count),
            cpu_cores=overrides.get("cpu_cores", self.cpu_cores),
            memory_gb=overrides.get("memory_gb", self.memory_gb),
            timeout_seconds=overrides.get("timeout_seconds", self.timeout_seconds),
            artifacts_dir=overrides.get("artifacts_dir", self.artifacts_dir),
            upload_paths=overrides.get("upload_paths", []),
        )


@dataclass
class JobHandle:
    """Opaque reference to a submitted job."""

    job_id: str
    backend_name: str
    status: JobStatus = JobStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """What comes back when a job finishes."""

    job_id: str
    status: JobStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, bytes] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    cost_usd: float | None = None


class ComputeBackend(Protocol):
    """The only interface compute providers must implement."""

    @property
    def name(self) -> str: ...

    async def submit(self, spec: JobSpec) -> JobHandle: ...

    async def poll(self, handle: JobHandle) -> JobHandle: ...

    async def cancel(self, handle: JobHandle) -> None: ...

    async def collect(self, handle: JobHandle) -> JobResult: ...

    async def cleanup(self, handle: JobHandle) -> None: ...


@dataclass
class BuildResult:
    """What comes back when an image build completes."""

    image_ref: str
    success: bool
    error: str = ""
    duration_seconds: float = 0.0


class BuildBackend(Protocol):
    """The interface image builders must implement.

    Parallel to ComputeBackend: the bot declares what it needs (a Dockerfile),
    infrastructure decides whether it's permitted and handles credentials.
    """

    @property
    def name(self) -> str: ...

    async def validate(self, dockerfile: str) -> list[str]:
        """Lint the Dockerfile. Return a list of errors (empty = valid)."""
        ...

    async def build(self, dockerfile: str, tag: str) -> BuildResult:
        """Build an image from a Dockerfile string. Returns the image ref."""
        ...

    async def push(self, tag: str) -> str:
        """Push a built image to a registry. Returns the full registry URI."""
        ...

    async def build_and_push(self, dockerfile: str, tag: str) -> BuildResult:
        """Validate, build, and push in one call."""
        ...
