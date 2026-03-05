"""SSHBackend — run jobs on remote machines over SSH."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ganglion.compute.protocol import JobHandle, JobResult, JobSpec, JobStatus

logger = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    """Connection details for a remote machine."""

    host: str
    user: str
    key_path: str | None = None
    port: int = 22
    remote_workdir: str = "/tmp/ganglion-jobs"
    gpu_type: str | None = None
    gpu_count: int = 0
    labels: dict[str, str] = field(default_factory=dict)


class SSHBackend:
    """Run jobs on a remote machine over SSH.

    Uses asyncssh under the hood. Supports:
    - File upload (training scripts, data)
    - Job execution (nohup detached)
    - Artifact download
    - Process monitoring via PID tracking
    """

    def __init__(self, config: SSHConfig, name: str | None = None):
        self._config = config
        self._name = name or f"ssh-{config.host}"
        self._conn: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    async def _ensure_connected(self) -> Any:
        """Lazily connect to the remote host."""
        if self._conn is not None:
            return self._conn
        try:
            import asyncssh
        except ImportError as e:
            raise ImportError(
                "asyncssh is required for SSHBackend. Install with: pip install asyncssh"
            ) from e

        connect_kwargs: dict[str, Any] = {
            "host": self._config.host,
            "port": self._config.port,
            "username": self._config.user,
            "known_hosts": None,
        }
        if self._config.key_path:
            connect_kwargs["client_keys"] = [self._config.key_path]

        self._conn = await asyncssh.connect(**connect_kwargs)
        return self._conn

    async def submit(self, spec: JobSpec) -> JobHandle:
        conn = await self._ensure_connected()
        job_id = f"ssh-{uuid.uuid4().hex[:8]}"
        remote_dir = f"{self._config.remote_workdir}/{job_id}"

        await conn.run(f"mkdir -p {remote_dir}", check=True)

        # Upload files
        if spec.upload_paths:
            async with conn.start_sftp_client() as sftp:
                for local_path in spec.upload_paths:
                    remote_path = f"{remote_dir}/{Path(local_path).name}"
                    await sftp.put(local_path, remote_path)

        # Launch detached with nohup, write PID to file
        cmd = " ".join(spec.command)
        env_str = " ".join(f"{k}={v}" for k, v in spec.env.items())
        launch = (
            f"cd {remote_dir} && "
            f"{env_str} nohup {cmd} "
            f"> stdout.log 2> stderr.log & echo $! > pid"
        )
        await conn.run(launch, check=False)

        return JobHandle(
            job_id=job_id,
            backend_name=self._name,
            status=JobStatus.RUNNING,
            metadata={"remote_dir": remote_dir},
        )

    async def poll(self, handle: JobHandle) -> JobHandle:
        conn = await self._ensure_connected()
        remote_dir = handle.metadata["remote_dir"]

        # Check if process is still running
        result = await conn.run(
            f"kill -0 $(cat {remote_dir}/pid) 2>/dev/null && echo RUNNING || echo DONE",
            check=False,
        )
        output = result.stdout.strip() if result.stdout else ""

        if "RUNNING" in output:
            handle.status = JobStatus.RUNNING
        else:
            # Check exit code from the exit_code file we write
            exit_result = await conn.run(
                f"cat {remote_dir}/exit_code 2>/dev/null || echo -1",
                check=False,
            )
            exit_code_str = (exit_result.stdout or "").strip()
            try:
                exit_code = int(exit_code_str)
            except ValueError:
                exit_code = -1
            handle.status = JobStatus.SUCCEEDED if exit_code == 0 else JobStatus.FAILED
            handle.metadata["exit_code"] = exit_code

        return handle

    async def cancel(self, handle: JobHandle) -> None:
        conn = await self._ensure_connected()
        remote_dir = handle.metadata["remote_dir"]
        await conn.run(
            f"kill $(cat {remote_dir}/pid) 2>/dev/null",
            check=False,
        )
        handle.status = JobStatus.CANCELLED

    async def collect(self, handle: JobHandle) -> JobResult:
        conn = await self._ensure_connected()
        remote_dir = handle.metadata["remote_dir"]

        # Read stdout and stderr logs
        stdout_result = await conn.run(f"cat {remote_dir}/stdout.log 2>/dev/null", check=False)
        stderr_result = await conn.run(f"cat {remote_dir}/stderr.log 2>/dev/null", check=False)

        stdout = stdout_result.stdout or ""
        stderr = stderr_result.stdout or ""
        exit_code = handle.metadata.get("exit_code")

        # Download artifacts
        artifacts: dict[str, bytes] = {}
        try:
            async with conn.start_sftp_client() as sftp:
                artifacts_dir = f"{remote_dir}/outputs"
                try:
                    for entry in await sftp.listdir(artifacts_dir):
                        remote_path = f"{artifacts_dir}/{entry}"
                        data = await sftp.read(remote_path)
                        if isinstance(data, bytes):
                            artifacts[entry] = data
                except (OSError, Exception):
                    pass  # No artifacts dir or read error
        except Exception:
            logger.warning("Failed to download artifacts for job %s", handle.job_id)

        return JobResult(
            job_id=handle.job_id,
            status=handle.status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            artifacts=artifacts,
        )

    async def cleanup(self, handle: JobHandle) -> None:
        try:
            conn = await self._ensure_connected()
            remote_dir = handle.metadata["remote_dir"]
            await conn.run(f"rm -rf {remote_dir}", check=False)
        except Exception:
            logger.warning("Failed to clean up remote dir for job %s", handle.job_id)

    async def disconnect(self) -> None:
        """Close the SSH connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
