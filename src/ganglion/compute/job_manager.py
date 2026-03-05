"""JobManager — lifecycle management for compute jobs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from ganglion.compute.protocol import JobHandle, JobResult, JobSpec, JobStatus
from ganglion.compute.router import ComputeRouter

logger = logging.getLogger(__name__)


class JobManager:
    """Manages the lifecycle of compute jobs.

    Tools submit jobs through here. The manager handles:
    - Routing to the right backend via ComputeRouter
    - Polling with exponential backoff
    - Timeout enforcement
    - Early stopping based on metric thresholds
    - Artifact collection
    """

    def __init__(self, router: ComputeRouter):
        self._router = router
        self._active_jobs: dict[str, tuple[JobHandle, str]] = {}  # job_id -> (handle, stage)
        self._results_cache: dict[str, JobResult] = {}

    @property
    def router(self) -> ComputeRouter:
        return self._router

    async def submit_and_wait(
        self,
        stage: str,
        spec: JobSpec,
        progress_callback: Callable[[JobHandle], None] | None = None,
        early_stop: Callable[[JobHandle], bool] | None = None,
    ) -> JobResult:
        """Submit a job and poll until completion."""
        backend, spec = self._router.resolve_with_overrides(stage, spec)
        handle = await backend.submit(spec)
        self._active_jobs[handle.job_id] = (handle, stage)

        # Poll with exponential backoff
        interval = 5.0
        max_interval = 60.0
        elapsed = 0.0

        try:
            while handle.status in (
                JobStatus.PENDING, JobStatus.PROVISIONING, JobStatus.RUNNING
            ):
                await asyncio.sleep(interval)
                elapsed += interval

                handle = await backend.poll(handle)
                self._active_jobs[handle.job_id] = (handle, stage)

                if progress_callback:
                    progress_callback(handle)

                if early_stop and early_stop(handle):
                    await backend.cancel(handle)
                    handle.status = JobStatus.CANCELLED
                    break

                if elapsed > spec.timeout_seconds:
                    await backend.cancel(handle)
                    handle.status = JobStatus.TIMEOUT
                    break

                interval = min(interval * 1.5, max_interval)

            result = await backend.collect(handle)
            await backend.cleanup(handle)
        except Exception:
            logger.error("Job %s failed unexpectedly", handle.job_id, exc_info=True)
            result = JobResult(
                job_id=handle.job_id,
                status=JobStatus.FAILED,
                stderr="Job failed with an unexpected error",
            )
        finally:
            self._active_jobs.pop(handle.job_id, None)

        self._results_cache[handle.job_id] = result
        return result

    async def submit_batch(
        self,
        stage: str,
        specs: list[JobSpec],
        max_concurrent: int = 5,
    ) -> list[JobResult]:
        """Submit multiple jobs and wait for all. For sweep/parallel training."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(spec: JobSpec) -> JobResult:
            async with semaphore:
                return await self.submit_and_wait(stage, spec)

        return list(await asyncio.gather(*[run_one(spec) for spec in specs]))

    def list_active(self) -> list[JobHandle]:
        """Return handles for all currently active jobs."""
        return [handle for handle, _ in self._active_jobs.values()]

    def get_result(self, job_id: str) -> JobResult | None:
        """Retrieve a cached result by job ID."""
        return self._results_cache.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific active job. Returns True if found and cancelled."""
        entry = self._active_jobs.get(job_id)
        if entry is None:
            return False
        handle, stage = entry
        backend = self._router.resolve(stage)
        await backend.cancel(handle)
        return True

    async def cancel_all(self) -> int:
        """Cancel all active jobs. Returns the number cancelled."""
        count = 0
        for job_id in list(self._active_jobs):
            if await self.cancel_job(job_id):
                count += 1
        return count

    def status(self) -> dict[str, Any]:
        """Summary of job manager state for observation endpoints."""
        return {
            "active_jobs": [
                {
                    "job_id": handle.job_id,
                    "backend": handle.backend_name,
                    "status": handle.status.value,
                    "stage": stage,
                }
                for handle, stage in self._active_jobs.values()
            ],
            "cached_results": len(self._results_cache),
        }
