"""LocalBackend — run jobs as local subprocesses."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

from ganglion.compute.protocol import JobHandle, JobResult, JobSpec, JobStatus


class LocalBackend:
    """Run jobs as local subprocesses. For development and cheap stages."""

    def __init__(self, name: str = "local", workdir: Path | None = None):
        self._name = name
        self._workdir = workdir or Path("./compute-runs")
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._start_times: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    async def submit(self, spec: JobSpec) -> JobHandle:
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        run_dir = self._workdir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create artifacts dir inside run dir
        artifacts_path = run_dir / spec.artifacts_dir.lstrip("/")
        artifacts_path.mkdir(parents=True, exist_ok=True)

        env = {**os.environ, **spec.env}
        proc = await asyncio.create_subprocess_exec(
            *spec.command,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(run_dir),
        )
        self._processes[job_id] = proc
        self._start_times[job_id] = time.monotonic()

        return JobHandle(
            job_id=job_id,
            backend_name=self._name,
            status=JobStatus.RUNNING,
            metadata={"run_dir": str(run_dir)},
        )

    async def poll(self, handle: JobHandle) -> JobHandle:
        proc = self._processes.get(handle.job_id)
        if proc is None:
            handle.status = JobStatus.FAILED
        elif proc.returncode is not None:
            handle.status = JobStatus.SUCCEEDED if proc.returncode == 0 else JobStatus.FAILED
        else:
            handle.status = JobStatus.RUNNING
        return handle

    async def cancel(self, handle: JobHandle) -> None:
        proc = self._processes.get(handle.job_id)
        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                proc.kill()
        handle.status = JobStatus.CANCELLED

    async def collect(self, handle: JobHandle) -> JobResult:
        proc = self._processes.get(handle.job_id)
        if proc is None:
            return JobResult(
                job_id=handle.job_id,
                status=JobStatus.FAILED,
                stderr="Process not found",
            )

        stdout_bytes, stderr_bytes = await proc.communicate()
        duration = time.monotonic() - self._start_times.get(handle.job_id, time.monotonic())
        artifacts = self._gather_artifacts(handle.job_id)

        status = handle.status
        if status not in (JobStatus.CANCELLED, JobStatus.TIMEOUT):
            status = JobStatus.SUCCEEDED if proc.returncode == 0 else JobStatus.FAILED

        return JobResult(
            job_id=handle.job_id,
            status=status,
            exit_code=proc.returncode,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            artifacts=artifacts,
            duration_seconds=duration,
        )

    async def cleanup(self, handle: JobHandle) -> None:
        self._processes.pop(handle.job_id, None)
        self._start_times.pop(handle.job_id, None)

    def _gather_artifacts(self, job_id: str) -> dict[str, bytes]:
        """Collect output files from the job's run directory."""
        run_dir = self._workdir / job_id
        artifacts: dict[str, bytes] = {}
        outputs_dir = run_dir / "outputs"
        if outputs_dir.is_dir():
            for path in outputs_dir.rglob("*"):
                if path.is_file():
                    try:
                        artifacts[path.name] = path.read_bytes()
                    except OSError:
                        pass
        return artifacts
