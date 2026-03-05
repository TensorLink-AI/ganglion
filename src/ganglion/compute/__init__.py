"""Ganglion compute — backend-agnostic job execution layer."""

from ganglion.compute.artifacts import ArtifactStore, LocalArtifactStore
from ganglion.compute.job_manager import JobManager
from ganglion.compute.protocol import (
    ComputeBackend,
    JobHandle,
    JobResult,
    JobSpec,
    JobStatus,
)
from ganglion.compute.router import ComputeRoute, ComputeRouter

__all__ = [
    "ArtifactStore",
    "ComputeBackend",
    "ComputeRoute",
    "ComputeRouter",
    "JobHandle",
    "JobManager",
    "JobResult",
    "JobSpec",
    "JobStatus",
    "LocalArtifactStore",
]
