"""RunPodBackend — submit jobs to RunPod on-demand GPU instances."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from ganglion.compute.protocol import JobHandle, JobResult, JobSpec, JobStatus

logger = logging.getLogger(__name__)


@dataclass
class RunPodConfig:
    """RunPod API configuration. Credentials are server-side only."""

    api_key: str
    template_id: str | None = None
    preferred_gpu: str = "NVIDIA A100 80GB"
    cloud_type: str = "COMMUNITY"
    max_bid_per_gpu: float = 0.5


# RunPod GraphQL status -> JobStatus mapping
_RUNPOD_STATUS_MAP: dict[str, JobStatus] = {
    "CREATED": JobStatus.PROVISIONING,
    "RUNNING": JobStatus.RUNNING,
    "EXITED": JobStatus.SUCCEEDED,
    "ERROR": JobStatus.FAILED,
}


class RunPodBackend:
    """Submit jobs to RunPod serverless or pods.

    Credentials (api_key) are loaded from config at startup and never
    exposed to the agent layer.
    """

    def __init__(self, config: RunPodConfig, name: str = "runpod"):
        self._config = config
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _resolve_gpu(self, gpu_type: str | None) -> str:
        """Map abstract GPU type to RunPod GPU ID."""
        if gpu_type and gpu_type.upper() != "ANY":
            return gpu_type
        return self._config.preferred_gpu

    def _create_pod_mutation(self, pod_config: dict[str, Any]) -> str:
        """Build the GraphQL mutation for creating a pod."""
        return """
        mutation {
            podFindAndDeployOnDemand(input: {
                name: "%(name)s",
                imageName: "%(imageName)s",
                gpuTypeId: "%(gpuTypeId)s",
                gpuCount: %(gpuCount)d,
                containerDiskInGb: %(containerDiskInGb)d,
                dockerArgs: "%(dockerArgs)s",
                cloudType: "%(cloudType)s"
            }) {
                id
                desiredStatus
                imageName
                machine { gpuDisplayName }
            }
        }
        """ % {
            **pod_config,
            "cloudType": self._config.cloud_type,
        }

    async def _api_request(self, query: str) -> dict[str, Any]:
        """Make a GraphQL request to the RunPod API."""
        try:
            import aiohttp
        except ImportError as e:
            raise ImportError(
                "aiohttp is required for RunPodBackend. Install with: pip install aiohttp"
            ) from e

        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                "https://api.runpod.io/graphql",
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                json={"query": query},
            )
            resp.raise_for_status()
            return await resp.json()

    async def submit(self, spec: JobSpec) -> JobHandle:
        pod_config = {
            "name": f"ganglion-{uuid.uuid4().hex[:6]}",
            "imageName": spec.image,
            "gpuTypeId": self._resolve_gpu(spec.gpu_type),
            "gpuCount": max(spec.gpu_count, 1),
            "containerDiskInGb": 20,
            "dockerArgs": " ".join(spec.command),
        }

        data = await self._api_request(self._create_pod_mutation(pod_config))

        pod_data = data.get("data", {}).get("podFindAndDeployOnDemand", {})
        pod_id = pod_data.get("id", f"runpod-{uuid.uuid4().hex[:8]}")

        return JobHandle(
            job_id=pod_id,
            backend_name=self._name,
            status=JobStatus.PROVISIONING,
            metadata={"pod_id": pod_id, "env": spec.env},
        )

    async def _get_pod_status(self, pod_id: str) -> str:
        """Query the current status of a RunPod pod."""
        query = """
        query {
            pod(input: { podId: "%s" }) {
                id
                desiredStatus
                runtime { uptimeInSeconds gpus { id } }
            }
        }
        """ % pod_id
        data = await self._api_request(query)
        pod = data.get("data", {}).get("pod", {})
        return pod.get("desiredStatus", "UNKNOWN")

    async def poll(self, handle: JobHandle) -> JobHandle:
        pod_status = await self._get_pod_status(handle.metadata["pod_id"])
        handle.status = _RUNPOD_STATUS_MAP.get(pod_status, JobStatus.PENDING)
        return handle

    async def _terminate_pod(self, pod_id: str) -> None:
        """Terminate a RunPod pod."""
        query = """
        mutation {
            podTerminate(input: { podId: "%s" })
        }
        """ % pod_id
        try:
            await self._api_request(query)
        except Exception:
            logger.warning("Failed to terminate RunPod pod %s", pod_id)

    async def cancel(self, handle: JobHandle) -> None:
        await self._terminate_pod(handle.metadata["pod_id"])
        handle.status = JobStatus.CANCELLED

    async def collect(self, handle: JobHandle) -> JobResult:
        # RunPod artifact collection would typically use their file API
        # or rsync from the pod. Placeholder for now.
        return JobResult(
            job_id=handle.job_id,
            status=handle.status,
        )

    async def cleanup(self, handle: JobHandle) -> None:
        await self._terminate_pod(handle.metadata["pod_id"])
