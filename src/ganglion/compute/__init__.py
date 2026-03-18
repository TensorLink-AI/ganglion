"""Ganglion compute — backend-agnostic job execution layer."""

from ganglion.compute.artifacts import ArtifactMeta, ArtifactStore, LocalArtifactStore
from ganglion.compute.backends.registry import BackendRegistry, get_backend_registry
from ganglion.compute.job_manager import JobManager
from ganglion.compute.protocol import (
    BuildBackend,
    BuildResult,
    ComputeBackend,
    DockerPrefab,
    JobHandle,
    JobResult,
    JobSpec,
    JobStatus,
)
from ganglion.compute.router import ComputeRoute, ComputeRouter

__all__ = [
    "ArtifactMeta",
    "ArtifactStore",
    "BackendRegistry",
    "BuildBackend",
    "BuildResult",
    "ComputeBackend",
    "ComputeRoute",
    "ComputeRouter",
    "DockerPrefab",
    "JobHandle",
    "JobManager",
    "JobResult",
    "JobSpec",
    "JobStatus",
    "LocalArtifactStore",
    "get_backend_registry",
]


def get_s3_artifact_store(**kwargs):
    """Lazy import of S3ArtifactStore to avoid hard boto3 dependency."""
    from ganglion.compute.artifacts_s3 import S3ArtifactStore

    return S3ArtifactStore(**kwargs)
