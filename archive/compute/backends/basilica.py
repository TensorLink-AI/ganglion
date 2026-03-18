"""BasilicaBackend — submit batch jobs to the Basilica decentralized GPU marketplace.

Basilica (Bittensor Subnet 39) deploys long-running containerised services with
a public URL.  Ganglion's ComputeBackend protocol models batch jobs (submit →
poll → collect → cleanup).  This module bridges the two models by generating a
Python wrapper script that executes the requested command inside a Basilica
deployment, captures stdout/stderr/exit-code, writes a JSON status file to a
shared Basilica Volume, then keeps the container alive briefly for result
collection.  A ``ttl_seconds`` timer ensures orphaned containers are
automatically cleaned up.
"""

from __future__ import annotations

import logging
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ganglion.compute.protocol import JobHandle, JobResult, JobSpec, JobStatus

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────


@dataclass
class BasilicaConfig:
    """Basilica deployment configuration."""

    api_token: str | None = None
    default_image: str = "python:3.11-slim"
    artifacts_volume: str = "ganglion-artifacts"
    preferred_gpu: str = "A100"
    min_gpu_memory_gb: int = 16
    ttl_seconds: int = 3600
    deploy_timeout: int = 600
    storage_mount: str = "/outputs"
    extra_pip_packages: list[str] = field(default_factory=list)


# ── Basilica state → JobStatus mapping ───────────────────────

_BASILICA_STATUS_MAP: dict[str, JobStatus] = {
    "pending": JobStatus.PENDING,
    "deploying": JobStatus.PROVISIONING,
    "starting": JobStatus.PROVISIONING,
    "running": JobStatus.RUNNING,
    "ready": JobStatus.RUNNING,
    "stopping": JobStatus.RUNNING,
    "stopped": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "error": JobStatus.FAILED,
}

# ── Image shorthand map ──────────────────────────────────────

_IMAGE_SHORTHANDS: dict[str, str] = {
    "pytorch": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "tensorflow": "tensorflow/tensorflow:2.14.0-gpu",
    "vllm": "vllm/vllm-openai:latest",
    "sglang": "lmsysorg/sglang:latest",
    "nvidia": "nvidia/cuda:12.1-runtime-ubuntu22.04",
}

# ── Terminal states ───────────────────────────────────────────

_TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT}
)


# ── Helpers ───────────────────────────────────────────────────


def _resolve_image(image: str, default_image: str) -> str:
    """Map a JobSpec image to a Basilica-compatible image reference.

    * If *image* already contains ``/`` or ``:`` it is assumed to be a fully
      qualified reference and is returned as-is.
    * Otherwise look up the shorthand map (``pytorch`` → full pytorch image).
    * Fall back to *default_image*.
    """
    if "/" in image or ":" in image:
        return image
    return _IMAGE_SHORTHANDS.get(image.lower(), default_image)


def _build_wrapper_script(spec: JobSpec) -> str:
    """Generate a Python wrapper that runs *spec.command* and writes results."""
    cmd_list = repr(spec.command)
    timeout = spec.timeout_seconds
    return textwrap.dedent(f"""\
        import json, os, subprocess, time

        # Wait for FUSE volume to be ready
        fuse_marker = "/outputs/.fuse_ready"
        for _ in range(60):
            if os.path.exists(fuse_marker):
                break
            time.sleep(1)

        start = time.time()
        try:
            proc = subprocess.run(
                {cmd_list},
                capture_output=True,
                text=True,
                timeout={timeout},
            )
            duration = time.time() - start
            result = {{
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "duration": duration,
                "status": "succeeded" if proc.returncode == 0 else "failed",
            }}
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            result = {{
                "exit_code": -1,
                "stdout": "",
                "stderr": "Command timed out",
                "duration": duration,
                "status": "timeout",
            }}
        except Exception as exc:
            duration = time.time() - start
            result = {{
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "duration": duration,
                "status": "failed",
            }}

        status_path = "/outputs/job_status.json"
        with open(status_path, "w") as f:
            json.dump(result, f)

        # Keep container alive briefly for result collection
        time.sleep(30)
    """)


def _build_deploy_kwargs(
    spec: JobSpec,
    config: BasilicaConfig,
    deployment_name: str,
    wrapper_script: str,
) -> dict[str, Any]:
    """Translate a JobSpec + config into keyword arguments for ``client.deploy()``."""
    kwargs: dict[str, Any] = {
        "name": deployment_name,
        "source": wrapper_script,
        "image": _resolve_image(spec.image, config.default_image),
        "port": 8000,
        "cpu": f"{spec.cpu_cores * 1000}m",
        "memory": f"{spec.memory_gb}Gi",
        "env": spec.env,
        "ttl_seconds": config.ttl_seconds,
        "timeout": config.deploy_timeout,
        "storage": config.storage_mount,
    }

    if config.extra_pip_packages:
        kwargs["pip_packages"] = list(config.extra_pip_packages)

    # GPU configuration — only set when the spec actually requests GPUs
    if spec.gpu_count and spec.gpu_count > 0:
        kwargs["gpu_count"] = spec.gpu_count
        gpu_type = spec.gpu_type
        if not gpu_type or gpu_type.upper() == "ANY":
            gpu_type = config.preferred_gpu
        kwargs["gpu_models"] = [gpu_type]
        kwargs["min_gpu_memory_gb"] = config.min_gpu_memory_gb

    return kwargs


# ── Backend ───────────────────────────────────────────────────


class BasilicaBackend:
    """Submit batch jobs to the Basilica decentralized GPU marketplace.

    Basilica deploys long-running services; this backend bridges to Ganglion's
    batch-job model by wrapping the requested command in a helper script that
    captures results and writes them to a shared Volume.
    """

    def __init__(self, config: BasilicaConfig | None = None, name: str = "basilica"):
        self._config = config or BasilicaConfig()
        self._name = name
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_client(self) -> Any:
        """Lazily create a BasilicaClient, raising ImportError if the SDK is missing."""
        if self._client is not None:
            return self._client
        try:
            from basilica import BasilicaClient
        except ImportError as e:
            raise ImportError(
                "basilica-sdk is required for BasilicaBackend. "
                "Install with: pip install basilica-sdk"
            ) from e

        kwargs: dict[str, Any] = {}
        if self._config.api_token:
            kwargs["api_token"] = self._config.api_token
        self._client = BasilicaClient(**kwargs)
        return self._client

    async def submit(self, spec: JobSpec) -> JobHandle:
        job_id = f"basilica-{uuid.uuid4().hex[:8]}"
        deployment_name = f"ganglion-{uuid.uuid4().hex[:6]}"

        wrapper_script = _build_wrapper_script(spec)
        deploy_kwargs = _build_deploy_kwargs(spec, self._config, deployment_name, wrapper_script)

        try:
            client = self._ensure_client()
            deployment = client.deploy(**deploy_kwargs)
            deployment_name = deployment.name
            url = getattr(deployment, "url", None)
        except Exception as exc:
            logger.warning("Basilica deploy failed for job %s: %s", job_id, exc)
            return JobHandle(
                job_id=job_id,
                backend_name=self._name,
                status=JobStatus.FAILED,
                metadata={
                    "deployment_name": deployment_name,
                    "error": str(exc),
                    "submitted_at": time.time(),
                },
            )

        return JobHandle(
            job_id=job_id,
            backend_name=self._name,
            status=JobStatus.PROVISIONING,
            metadata={
                "deployment_name": deployment_name,
                "url": url,
                "submitted_at": time.time(),
            },
        )

    async def poll(self, handle: JobHandle) -> JobHandle:
        # Skip polling for terminal states
        if handle.status in _TERMINAL_STATUSES:
            return handle

        deployment_name = handle.metadata.get("deployment_name")
        if not deployment_name:
            logger.warning("No deployment_name in handle metadata for job %s", handle.job_id)
            handle.status = JobStatus.FAILED
            handle.metadata["error"] = "Missing deployment_name in metadata"
            return handle

        try:
            client = self._ensure_client()
            deployment = client.get(deployment_name)
            status_obj = deployment.status()

            if getattr(status_obj, "is_failed", False):
                handle.status = JobStatus.FAILED
                return handle

            state = getattr(status_obj, "state", "unknown")
            handle.status = _BASILICA_STATUS_MAP.get(state, JobStatus.PENDING)
        except Exception as exc:
            # Check for timeout: if provisioning too long, mark as TIMEOUT
            submitted_at = handle.metadata.get("submitted_at", 0)
            if time.time() - submitted_at > self._config.deploy_timeout:
                handle.status = JobStatus.TIMEOUT
                handle.metadata["error"] = f"Deploy timeout exceeded: {exc}"
            else:
                logger.warning("Basilica poll error for job %s: %s", handle.job_id, exc)

        return handle

    async def cancel(self, handle: JobHandle) -> None:
        deployment_name = handle.metadata.get("deployment_name")
        if deployment_name:
            try:
                client = self._ensure_client()
                deployment = client.get(deployment_name)
                deployment.delete()
            except Exception:
                logger.warning("Failed to delete Basilica deployment %s", deployment_name)
        handle.status = JobStatus.CANCELLED

    async def collect(self, handle: JobHandle) -> JobResult:
        stdout = ""
        stderr = handle.metadata.get("error", "")

        deployment_name = handle.metadata.get("deployment_name")
        if deployment_name:
            try:
                client = self._ensure_client()
                deployment = client.get(deployment_name)
                stdout = deployment.logs(tail=500)
            except Exception:
                logger.warning("Failed to collect logs for job %s", handle.job_id)

        return JobResult(
            job_id=handle.job_id,
            status=handle.status,
            stdout=stdout,
            stderr=stderr,
        )

    async def cleanup(self, handle: JobHandle) -> None:
        await self.cancel(handle)
