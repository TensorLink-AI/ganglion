"""Tests for the compute module — protocol, backends, router, job manager."""

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ganglion.compute.artifacts import LocalArtifactStore
from ganglion.compute.backends.docker_build import (
    DockerBuildBackend,
    DockerBuildConfig,
    _match_glob,
)
from ganglion.compute.backends.local import LocalBackend
from ganglion.compute.backends.runpod import RunPodBackend, RunPodConfig
from ganglion.compute.backends.ssh import SSHBackend, SSHConfig
from ganglion.compute.job_manager import JobManager
from ganglion.compute.mcp_tools import _render_dockerfile, register_compute_tools
from ganglion.compute.protocol import (
    BuildResult,
    DockerPrefab,
    JobHandle,
    JobResult,
    JobSpec,
    JobStatus,
)
from ganglion.compute.router import ComputeRoute, ComputeRouter

# ── Protocol / data class tests ────────────────────────────


class TestJobSpec:
    def test_defaults(self):
        spec = JobSpec(image="test:latest", command=["echo", "hi"])
        assert spec.gpu_count == 0
        assert spec.cpu_cores == 2
        assert spec.memory_gb == 8
        assert spec.timeout_seconds == 3600
        assert spec.artifacts_dir == "/outputs"
        assert spec.upload_paths == []

    def test_custom(self):
        spec = JobSpec(
            image="train:v2",
            command=["python", "train.py"],
            gpu_type="A100",
            gpu_count=2,
            timeout_seconds=7200,
        )
        assert spec.gpu_type == "A100"
        assert spec.gpu_count == 2


class TestJobHandle:
    def test_defaults(self):
        h = JobHandle(job_id="test-1", backend_name="local")
        assert h.status == JobStatus.PENDING
        assert h.metadata == {}

    def test_status_mutation(self):
        h = JobHandle(job_id="test-1", backend_name="local")
        h.status = JobStatus.RUNNING
        assert h.status == JobStatus.RUNNING


class TestJobResult:
    def test_defaults(self):
        r = JobResult(job_id="test-1", status=JobStatus.SUCCEEDED)
        assert r.exit_code is None
        assert r.stdout == ""
        assert r.artifacts == {}
        assert r.cost_usd is None


class TestJobStatus:
    def test_all_values(self):
        assert len(JobStatus) == 7
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.TIMEOUT.value == "timeout"


class TestDockerPrefab:
    def test_to_job_spec_defaults(self):
        prefab = DockerPrefab(name="train", image="registry/train:v1")
        spec = prefab.to_job_spec(["python", "train.py"])
        assert spec.image == "registry/train:v1"
        assert spec.command == ["python", "train.py"]
        assert spec.gpu_count == 0
        assert spec.cpu_cores == 2
        assert spec.memory_gb == 8
        assert spec.timeout_seconds == 3600
        assert spec.artifacts_dir == "/outputs"
        assert spec.upload_paths == []
        assert spec.env == {}

    def test_to_job_spec_with_prefab_fields(self):
        prefab = DockerPrefab(
            name="gpu-train",
            image="registry/train:v2",
            gpu_type="A100",
            gpu_count=2,
            memory_gb=64,
            timeout_seconds=7200,
            env={"CUDA_VISIBLE_DEVICES": "all"},
        )
        spec = prefab.to_job_spec(["python", "train.py"])
        assert spec.image == "registry/train:v2"
        assert spec.gpu_type == "A100"
        assert spec.gpu_count == 2
        assert spec.memory_gb == 64
        assert spec.timeout_seconds == 7200
        assert spec.env == {"CUDA_VISIBLE_DEVICES": "all"}

    def test_to_job_spec_overrides(self):
        prefab = DockerPrefab(
            name="train",
            image="registry/train:v1",
            gpu_type="A100",
            gpu_count=1,
            memory_gb=16,
        )
        spec = prefab.to_job_spec(
            ["python", "train.py"],
            gpu_count=4,
            timeout_seconds=9000,
        )
        assert spec.gpu_type == "A100"  # from prefab
        assert spec.gpu_count == 4  # overridden
        assert spec.timeout_seconds == 9000  # overridden
        assert spec.memory_gb == 16  # from prefab

    def test_env_merging(self):
        prefab = DockerPrefab(
            name="train",
            image="registry/train:v1",
            env={"BASE_KEY": "base", "SHARED": "from_prefab"},
        )
        spec = prefab.to_job_spec(
            ["python", "train.py"],
            env={"SHARED": "from_override", "NEW_KEY": "new"},
        )
        assert spec.env["BASE_KEY"] == "base"
        assert spec.env["SHARED"] == "from_override"
        assert spec.env["NEW_KEY"] == "new"

    def test_image_override(self):
        prefab = DockerPrefab(name="train", image="registry/train:v1")
        spec = prefab.to_job_spec(["echo"], image="registry/train:v2")
        assert spec.image == "registry/train:v2"


# ── MockBackend for testing ─────────────────────────────────


class MockBackend:
    """A mock compute backend for testing."""

    def __init__(self, name: str = "mock", fail: bool = False):
        self._name = name
        self._fail = fail
        self._submitted: list[JobSpec] = []
        self._cancelled: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def submit(self, spec: JobSpec) -> JobHandle:
        self._submitted.append(spec)
        return JobHandle(
            job_id=f"mock-{len(self._submitted)}",
            backend_name=self._name,
            status=JobStatus.RUNNING,
        )

    async def poll(self, handle: JobHandle) -> JobHandle:
        if self._fail:
            handle.status = JobStatus.FAILED
        else:
            handle.status = JobStatus.SUCCEEDED
        return handle

    async def cancel(self, handle: JobHandle) -> None:
        self._cancelled.append(handle.job_id)
        handle.status = JobStatus.CANCELLED

    async def collect(self, handle: JobHandle) -> JobResult:
        return JobResult(
            job_id=handle.job_id,
            status=handle.status,
            exit_code=0 if not self._fail else 1,
            metrics={"loss": 0.5},
        )

    async def cleanup(self, handle: JobHandle) -> None:
        pass


# ── Router tests ────────────────────────────────────────────


class TestComputeRouter:
    def test_resolve_exact_match(self):
        mock = MockBackend("gpu-rig")
        local = MockBackend("local")
        router = ComputeRouter(
            backends={"local": local, "gpu-rig": mock},
            routes=[
                ComputeRoute(pattern="train", backend="gpu-rig"),
                ComputeRoute(pattern="default", backend="local"),
            ],
        )
        assert router.resolve("train").name == "gpu-rig"
        assert router.resolve("plan").name == "local"

    def test_resolve_glob_pattern(self):
        mock = MockBackend("cloud")
        local = MockBackend("local")
        router = ComputeRouter(
            backends={"local": local, "cloud": mock},
            routes=[
                ComputeRoute(pattern="train*", backend="cloud"),
                ComputeRoute(pattern="default", backend="local"),
            ],
        )
        assert router.resolve("train").name == "cloud"
        assert router.resolve("train_v2").name == "cloud"
        assert router.resolve("plan").name == "local"

    def test_resolve_with_overrides(self):
        mock = MockBackend("gpu")
        router = ComputeRouter(
            backends={"gpu": mock},
            routes=[
                ComputeRoute(
                    pattern="train",
                    backend="gpu",
                    overrides={"gpu_count": 4, "timeout_seconds": 9000},
                ),
            ],
        )
        spec = JobSpec(image="test:latest", command=["train"])
        backend, new_spec = router.resolve_with_overrides("train", spec)
        assert backend.name == "gpu"
        assert new_spec.gpu_count == 4
        assert new_spec.timeout_seconds == 9000
        # Original spec unchanged
        assert spec.gpu_count == 0

    def test_fallback_to_local(self):
        local = MockBackend("local")
        router = ComputeRouter(backends={"local": local}, routes=[])
        assert router.resolve("anything").name == "local"

    def test_add_remove_backend(self):
        local = MockBackend("local")
        router = ComputeRouter(backends={"local": local}, routes=[])
        cloud = MockBackend("cloud")
        router.add_backend("cloud", cloud)
        assert "cloud" in router.backends
        removed = router.remove_backend("cloud")
        assert removed is not None
        assert "cloud" not in router.backends

    def test_to_dict(self):
        local = MockBackend("local")
        router = ComputeRouter(
            backends={"local": local},
            routes=[ComputeRoute(pattern="default", backend="local")],
        )
        d = router.to_dict()
        assert len(d["backends"]) == 1
        assert d["backends"][0]["name"] == "local"
        assert len(d["routes"]) == 1

    def test_set_routes(self):
        local = MockBackend("local")
        router = ComputeRouter(backends={"local": local}, routes=[])
        router.set_routes([ComputeRoute(pattern="x", backend="local")])
        assert len(router.routes) == 1


# ── LocalBackend tests ──────────────────────────────────────


class TestLocalBackend:
    @pytest.fixture
    def tmp_workdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.mark.asyncio
    async def test_submit_and_collect(self, tmp_workdir):
        backend = LocalBackend(workdir=tmp_workdir)
        spec = JobSpec(
            image="unused",
            command=["echo", "hello world"],
        )
        handle = await backend.submit(spec)
        assert handle.status == JobStatus.RUNNING
        assert handle.backend_name == "local"
        assert handle.job_id.startswith("local-")

        result = await backend.collect(handle)
        assert result.status == JobStatus.SUCCEEDED
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_failed_command(self, tmp_workdir):
        backend = LocalBackend(workdir=tmp_workdir)
        spec = JobSpec(
            image="unused",
            command=["false"],
        )
        handle = await backend.submit(spec)
        result = await backend.collect(handle)
        assert result.status == JobStatus.FAILED
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_poll(self, tmp_workdir):
        backend = LocalBackend(workdir=tmp_workdir)
        spec = JobSpec(
            image="unused",
            command=["sleep", "0.1"],
        )
        handle = await backend.submit(spec)

        # Should be running immediately
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.RUNNING

        # Wait for it to finish
        await asyncio.sleep(0.3)
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_cancel(self, tmp_workdir):
        backend = LocalBackend(workdir=tmp_workdir)
        spec = JobSpec(
            image="unused",
            command=["sleep", "10"],
        )
        handle = await backend.submit(spec)
        await backend.cancel(handle)
        assert handle.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cleanup(self, tmp_workdir):
        backend = LocalBackend(workdir=tmp_workdir)
        spec = JobSpec(image="unused", command=["echo", "x"])
        handle = await backend.submit(spec)
        await backend.collect(handle)
        await backend.cleanup(handle)
        # Process should be gone
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.FAILED  # not found

    @pytest.mark.asyncio
    async def test_env_vars(self, tmp_workdir):
        backend = LocalBackend(workdir=tmp_workdir)
        spec = JobSpec(
            image="unused",
            command=["env"],
            env={"GANGLION_TEST": "hello123"},
        )
        handle = await backend.submit(spec)
        result = await backend.collect(handle)
        assert "GANGLION_TEST=hello123" in result.stdout


# ── JobManager tests ────────────────────────────────────────


class TestJobManager:
    @pytest.mark.asyncio
    async def test_submit_and_wait(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)

        spec = JobSpec(image="test:latest", command=["train"])
        result = await manager.submit_and_wait("train", spec)
        assert result.status == JobStatus.SUCCEEDED
        assert result.metrics == {"loss": 0.5}
        assert len(mock._submitted) == 1

    @pytest.mark.asyncio
    async def test_submit_and_wait_failure(self):
        mock = MockBackend(fail=True)
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)

        spec = JobSpec(image="test:latest", command=["train"])
        result = await manager.submit_and_wait("train", spec)
        assert result.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_submit_batch(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)

        specs = [
            JobSpec(image="test:latest", command=["train", f"--lr={lr}"])
            for lr in [0.001, 0.01, 0.1]
        ]
        results = await manager.submit_batch("sweep", specs, max_concurrent=2)
        assert len(results) == 3
        assert all(r.status == JobStatus.SUCCEEDED for r in results)
        assert len(mock._submitted) == 3

    @pytest.mark.asyncio
    async def test_get_result_cached(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)

        spec = JobSpec(image="test:latest", command=["train"])
        result = await manager.submit_and_wait("train", spec)
        cached = manager.get_result(result.job_id)
        assert cached is not None
        assert cached.job_id == result.job_id

    @pytest.mark.asyncio
    async def test_status(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)
        status = manager.status()
        assert status["active_jobs"] == []
        assert status["cached_results"] == 0

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)

        callbacks = []
        spec = JobSpec(image="test:latest", command=["train"])
        await manager.submit_and_wait(
            "train", spec, progress_callback=lambda h: callbacks.append(h.status)
        )
        assert len(callbacks) > 0

    @pytest.mark.asyncio
    async def test_early_stop(self):
        mock = MockBackend()
        # Override poll to keep returning RUNNING
        original_poll = mock.poll

        call_count = 0

        async def slow_poll(handle):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                handle.status = JobStatus.RUNNING
                return handle
            return await original_poll(handle)

        mock.poll = slow_poll

        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)

        spec = JobSpec(image="test:latest", command=["train"])
        result = await manager.submit_and_wait("train", spec, early_stop=lambda h: True)
        assert result.status == JobStatus.CANCELLED


# ── ArtifactStore tests ─────────────────────────────────────


class TestLocalArtifactStore:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as d:
            yield LocalArtifactStore(root=Path(d))

    @pytest.mark.asyncio
    async def test_put_and_get(self, store):
        await store.put("model.pt", b"model data")
        data = await store.get("model.pt")
        assert data == b"model data"

    @pytest.mark.asyncio
    async def test_get_missing(self, store):
        data = await store.get("nope.pt")
        assert data is None

    @pytest.mark.asyncio
    async def test_list(self, store):
        await store.put("run1/model.pt", b"a")
        await store.put("run1/metrics.json", b"b")
        await store.put("run2/model.pt", b"c")

        all_files = await store.list()
        assert len(all_files) == 3

        run1_files = await store.list("run1")
        assert len(run1_files) == 2

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.put("temp.pt", b"data")
        assert await store.delete("temp.pt")
        assert await store.get("temp.pt") is None
        assert not await store.delete("nope.pt")

    @pytest.mark.asyncio
    async def test_nested_paths(self, store):
        await store.put("deep/nested/file.txt", b"content")
        data = await store.get("deep/nested/file.txt")
        assert data == b"content"


# ── SSHBackend tests (mocked asyncssh) ──────────────────────


def _make_ssh_run_result(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, returncode=returncode)


class TestSSHBackend:
    def test_config_defaults(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        assert cfg.port == 22
        assert cfg.remote_workdir == "/tmp/ganglion-jobs"
        assert cfg.gpu_count == 0
        assert cfg.labels == {}

    def test_name_default(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        assert backend.name == "ssh-10.0.0.1"

    def test_name_custom(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg, name="my-rig")
        assert backend.name == "my-rig"

    @pytest.mark.asyncio
    async def test_submit(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=_make_ssh_run_result())
        backend._conn = mock_conn

        spec = JobSpec(image="train:v1", command=["python", "train.py"])
        handle = await backend.submit(spec)
        assert handle.status == JobStatus.RUNNING
        assert handle.backend_name == "ssh-10.0.0.1"
        assert handle.job_id.startswith("ssh-")
        assert "remote_dir" in handle.metadata
        assert mock_conn.run.call_count == 2  # mkdir + launch

    @pytest.mark.asyncio
    async def test_submit_with_uploads(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=_make_ssh_run_result())
        mock_sftp = AsyncMock()
        mock_conn.start_sftp_client = MagicMock(return_value=mock_sftp)
        mock_sftp.__aenter__ = AsyncMock(return_value=mock_sftp)
        mock_sftp.__aexit__ = AsyncMock(return_value=False)
        backend._conn = mock_conn

        spec = JobSpec(
            image="train:v1",
            command=["python", "train.py"],
            upload_paths=["/data/dataset.csv"],
        )
        handle = await backend.submit(spec)
        assert handle.status == JobStatus.RUNNING
        mock_sftp.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_running(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=_make_ssh_run_result("RUNNING"))
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.RUNNING,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_poll_done_success(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        # First call: kill -0 returns DONE
        # Second call: cat exit_code returns 0
        mock_conn.run = AsyncMock(
            side_effect=[
                _make_ssh_run_result("DONE"),
                _make_ssh_run_result("0"),
            ]
        )
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.RUNNING,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_poll_done_failed(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(
            side_effect=[
                _make_ssh_run_result("DONE"),
                _make_ssh_run_result("1"),
            ]
        )
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.RUNNING,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_invalid_exit_code(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(
            side_effect=[
                _make_ssh_run_result("DONE"),
                _make_ssh_run_result("not-a-number"),
            ]
        )
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.RUNNING,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=_make_ssh_run_result())
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.RUNNING,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        await backend.cancel(handle)
        assert handle.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_collect(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(
            side_effect=[
                _make_ssh_run_result("stdout output"),
                _make_ssh_run_result("stderr output"),
            ]
        )
        # Mock SFTP that raises to skip artifact download
        mock_sftp = AsyncMock()
        mock_sftp.__aenter__ = AsyncMock(side_effect=Exception("no sftp"))
        mock_sftp.__aexit__ = AsyncMock(return_value=False)
        mock_conn.start_sftp_client = MagicMock(return_value=mock_sftp)
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.SUCCEEDED,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test", "exit_code": 0},
        )
        result = await backend.collect(handle)
        assert result.stdout == "stdout output"
        assert result.stderr == "stderr output"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_cleanup(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=_make_ssh_run_result())
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.SUCCEEDED,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        await backend.cleanup(handle)
        mock_conn.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_error_handled(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(side_effect=Exception("connection lost"))
        backend._conn = mock_conn

        handle = JobHandle(
            job_id="ssh-test",
            backend_name="ssh-10.0.0.1",
            status=JobStatus.SUCCEEDED,
            metadata={"remote_dir": "/tmp/ganglion-jobs/ssh-test"},
        )
        # Should not raise
        await backend.cleanup(handle)

    @pytest.mark.asyncio
    async def test_disconnect(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = MagicMock()
        backend._conn = mock_conn
        await backend.disconnect()
        mock_conn.close.assert_called_once()
        assert backend._conn is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        await backend.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_ensure_connected_import_error(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        with (
            patch.dict("sys.modules", {"asyncssh": None}),
            pytest.raises(ImportError, match="asyncssh"),
        ):
            await backend._ensure_connected()

    @pytest.mark.asyncio
    async def test_ensure_connected_reuses(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        backend._conn = mock_conn
        result = await backend._ensure_connected()
        assert result is mock_conn

    @pytest.mark.asyncio
    async def test_submit_with_env(self):
        cfg = SSHConfig(host="10.0.0.1", user="miner")
        backend = SSHBackend(cfg)
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=_make_ssh_run_result())
        backend._conn = mock_conn

        spec = JobSpec(
            image="train:v1",
            command=["python", "train.py"],
            env={"LR": "0.001"},
        )
        handle = await backend.submit(spec)
        assert handle.status == JobStatus.RUNNING
        # Check that env vars were included in the launch command
        launch_call = mock_conn.run.call_args_list[-1]
        assert "LR=0.001" in launch_call[0][0]


# ── RunPodBackend tests (mocked aiohttp) ────────────────────


class TestRunPodBackend:
    def test_config_defaults(self):
        cfg = RunPodConfig(api_key="test-key")
        assert cfg.preferred_gpu == "NVIDIA A100 80GB"
        assert cfg.cloud_type == "COMMUNITY"
        assert cfg.max_bid_per_gpu == 0.5

    def test_name(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        assert backend.name == "runpod"
        backend2 = RunPodBackend(cfg, name="runpod-us")
        assert backend2.name == "runpod-us"

    def test_resolve_gpu_any(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        assert backend._resolve_gpu("any") == "NVIDIA A100 80GB"
        assert backend._resolve_gpu(None) == "NVIDIA A100 80GB"
        assert backend._resolve_gpu("H100") == "H100"

    def test_create_pod_mutation(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        query = backend._create_pod_mutation(
            {
                "name": "test-pod",
                "imageName": "train:v1",
                "gpuTypeId": "A100",
                "gpuCount": 1,
                "containerDiskInGb": 20,
                "dockerArgs": "python train.py",
            }
        )
        assert "podFindAndDeployOnDemand" in query
        assert "test-pod" in query
        assert "train:v1" in query
        assert "COMMUNITY" in query

    @pytest.mark.asyncio
    async def test_api_request_import_error(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        with (
            patch.dict("sys.modules", {"aiohttp": None}),
            pytest.raises(ImportError, match="aiohttp"),
        ):
            await backend._api_request("query { test }")

    @pytest.mark.asyncio
    async def test_submit(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(
            return_value={"data": {"podFindAndDeployOnDemand": {"id": "pod-abc123"}}}
        )

        spec = JobSpec(
            image="train:v1",
            command=["python", "train.py"],
            gpu_type="A100",
            gpu_count=1,
        )
        handle = await backend.submit(spec)
        assert handle.job_id == "pod-abc123"
        assert handle.status == JobStatus.PROVISIONING
        assert handle.backend_name == "runpod"

    @pytest.mark.asyncio
    async def test_submit_fallback_id(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(return_value={"data": {}})

        spec = JobSpec(image="train:v1", command=["python", "train.py"])
        handle = await backend.submit(spec)
        assert handle.job_id.startswith("runpod-")

    @pytest.mark.asyncio
    async def test_poll_running(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(
            return_value={"data": {"pod": {"desiredStatus": "RUNNING"}}}
        )

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.PROVISIONING,
            metadata={"pod_id": "pod-abc"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_poll_exited(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(
            return_value={"data": {"pod": {"desiredStatus": "EXITED"}}}
        )

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.RUNNING,
            metadata={"pod_id": "pod-abc"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_poll_error(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(return_value={"data": {"pod": {"desiredStatus": "ERROR"}}})

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.RUNNING,
            metadata={"pod_id": "pod-abc"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_poll_unknown_status(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(
            return_value={"data": {"pod": {"desiredStatus": "UNKNOWN_NEW_STATUS"}}}
        )

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.RUNNING,
            metadata={"pod_id": "pod-abc"},
        )
        handle = await backend.poll(handle)
        assert handle.status == JobStatus.PENDING  # fallback

    @pytest.mark.asyncio
    async def test_cancel(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(return_value={})

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.RUNNING,
            metadata={"pod_id": "pod-abc"},
        )
        await backend.cancel(handle)
        assert handle.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_collect(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.SUCCEEDED,
            metadata={"pod_id": "pod-abc"},
        )
        result = await backend.collect(handle)
        assert result.job_id == "pod-abc"
        assert result.status == JobStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_cleanup(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(return_value={})

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.SUCCEEDED,
            metadata={"pod_id": "pod-abc"},
        )
        await backend.cleanup(handle)
        backend._api_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_error_handled(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(side_effect=Exception("API error"))

        handle = JobHandle(
            job_id="pod-abc",
            backend_name="runpod",
            status=JobStatus.RUNNING,
            metadata={"pod_id": "pod-abc"},
        )
        # Should not raise
        await backend.cleanup(handle)

    @pytest.mark.asyncio
    async def test_get_pod_status_missing_pod(self):
        cfg = RunPodConfig(api_key="test-key")
        backend = RunPodBackend(cfg)
        backend._api_request = AsyncMock(return_value={"data": {"pod": None}})
        status = await backend._get_pod_status("missing-pod")
        assert status == "UNKNOWN"


# ── MCP Tools tests ─────────────────────────────────────────


class TestMCPTools:
    def _make_state_with_compute(self):
        """Create a FrameworkState with compute configured."""
        from ganglion.orchestration.pipeline import PipelineDef, StageDef
        from ganglion.orchestration.task_context import MetricDef, OutputSpec, SubnetConfig, TaskDef
        from ganglion.state.agent_registry import AgentRegistry
        from ganglion.state.framework_state import FrameworkState
        from ganglion.state.tool_registry import ToolRegistry

        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        return FrameworkState(
            subnet_config=SubnetConfig(
                netuid=99,
                name="Test",
                metrics=[MetricDef("acc", "maximize")],
                tasks={"main": TaskDef("main")},
                output_spec=OutputSpec(format="test"),
            ),
            pipeline_def=PipelineDef(name="test", stages=[StageDef(name="train", agent="Trainer")]),
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            compute_router=router,
        )

    def test_register_compute_tools(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)
        assert state.tool_registry.has("compute_status")
        assert state.tool_registry.has("compute_jobs")
        assert state.tool_registry.has("compute_job_detail")
        assert state.tool_registry.has("compute_routes")

    def test_register_skips_without_compute(self):
        from ganglion.orchestration.pipeline import PipelineDef, StageDef
        from ganglion.orchestration.task_context import MetricDef, OutputSpec, SubnetConfig, TaskDef
        from ganglion.state.agent_registry import AgentRegistry
        from ganglion.state.framework_state import FrameworkState
        from ganglion.state.tool_registry import ToolRegistry

        state = FrameworkState(
            subnet_config=SubnetConfig(
                netuid=99,
                name="Test",
                metrics=[MetricDef("acc", "maximize")],
                tasks={"main": TaskDef("main")},
                output_spec=OutputSpec(format="test"),
            ),
            pipeline_def=PipelineDef(name="test", stages=[StageDef(name="train", agent="Trainer")]),
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
        )
        register_compute_tools(state)
        assert not state.tool_registry.has("compute_status")

    def test_register_idempotent(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)
        register_compute_tools(state)
        assert state.tool_registry.has("compute_status")

    @pytest.mark.asyncio
    async def test_compute_status_tool(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)
        tool = state.tool_registry.get("compute_status")
        result = await tool.func()
        data = json.loads(result)
        assert "backends" in data

    @pytest.mark.asyncio
    async def test_compute_jobs_tool(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)
        tool = state.tool_registry.get("compute_jobs")
        result = await tool.func()
        data = json.loads(result)
        assert "active_jobs" in data

    @pytest.mark.asyncio
    async def test_compute_job_detail_not_found(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)
        tool = state.tool_registry.get("compute_job_detail")
        result = await tool.func(job_id="nonexistent")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_compute_routes_tool(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)
        tool = state.tool_registry.get("compute_routes")
        result = await tool.func()
        data = json.loads(result)
        assert "routes" in data
        assert len(data["routes"]) == 1

    @pytest.mark.asyncio
    async def test_compute_job_detail_cached_result(self):
        state = self._make_state_with_compute()
        register_compute_tools(state)

        # Submit and wait for a job to populate cached results
        spec = JobSpec(image="test:latest", command=["train"])
        result = await state.job_manager.submit_and_wait("train", spec)

        tool = state.tool_registry.get("compute_job_detail")
        detail = await tool.func(job_id=result.job_id)
        data = json.loads(detail)
        assert data["job_id"] == result.job_id
        assert data["status"] == "succeeded"


# ── Router edge case tests ──────────────────────────────────


class TestComputeRouterEdgeCases:
    def test_resolve_with_overrides_no_overrides(self):
        mock = MockBackend("gpu")
        router = ComputeRouter(
            backends={"gpu": mock},
            routes=[ComputeRoute(pattern="default", backend="gpu")],
        )
        spec = JobSpec(image="test:latest", command=["train"])
        backend, new_spec = router.resolve_with_overrides("train", spec)
        assert new_spec is spec  # Same object, no copy needed

    def test_resolve_no_local_fallback(self):
        mock = MockBackend("cloud")
        router = ComputeRouter(backends={"cloud": mock}, routes=[])
        assert router.resolve("anything").name == "cloud"

    def test_resolve_with_overrides_fallback(self):
        mock = MockBackend("cloud")
        router = ComputeRouter(backends={"cloud": mock}, routes=[])
        spec = JobSpec(image="test:latest", command=["train"])
        backend, new_spec = router.resolve_with_overrides("unknown", spec)
        assert backend.name == "cloud"
        assert new_spec is spec

    def test_add_route(self):
        local = MockBackend("local")
        router = ComputeRouter(backends={"local": local}, routes=[])
        router.add_route(ComputeRoute(pattern="train", backend="local"))
        assert len(router.routes) == 1
        router.add_route(ComputeRoute(pattern="plan", backend="local"), index=0)
        assert router.routes[0].pattern == "plan"


# ── JobManager edge case tests ──────────────────────────────


class TestJobManagerEdgeCases:
    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)
        result = await manager.cancel_job("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_empty(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)
        count = await manager.cancel_all()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_result_not_found(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)
        assert manager.get_result("nonexistent") is None

    @pytest.mark.asyncio
    async def test_router_property(self):
        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        manager = JobManager(router)
        assert manager.router is router


# ── ArtifactStore edge cases ────────────────────────────────


class TestLocalArtifactStoreEdgeCases:
    @pytest.mark.asyncio
    async def test_list_nonexistent_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=Path(d))
            result = await store.list("nonexistent")
            assert result == []

    @pytest.mark.asyncio
    async def test_delete_directory(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=Path(d))
            await store.put("dir/file1.txt", b"a")
            await store.put("dir/file2.txt", b"b")
            assert await store.delete("dir")
            assert await store.list("dir") == []

    @pytest.mark.asyncio
    async def test_default_root(self):
        store = LocalArtifactStore()
        assert store._root == Path("./artifacts")


# ── BuildResult tests ────────────────────────────────────────


class TestBuildResult:
    def test_defaults(self):
        r = BuildResult(image_ref="ghcr.io/test:v1", success=True)
        assert r.error == ""
        assert r.duration_seconds == 0.0

    def test_failed(self):
        r = BuildResult(image_ref="", success=False, error="build failed")
        assert not r.success
        assert r.error == "build failed"


# ── DockerBuildBackend tests ─────────────────────────────────


class TestMatchGlob:
    def test_exact(self):
        assert _match_glob("python:3.11", "python:3.11")
        assert not _match_glob("python:3.11", "python:3.12")

    def test_wildcard_tag(self):
        assert _match_glob("python:*", "python:3.11")
        assert _match_glob("python:*", "python:3.11-slim")
        assert _match_glob("nvidia/pytorch:*", "nvidia/pytorch:24.01")
        assert not _match_glob("nvidia/pytorch:*", "nvidia/cuda:12.0")

    def test_bare_name_matches_wildcard(self):
        assert _match_glob("python:*", "python")

    def test_prefix_glob(self):
        assert _match_glob("nvidia/*", "nvidia/pytorch:24.01")


class TestDockerBuildConfig:
    def test_defaults(self):
        cfg = DockerBuildConfig()
        assert cfg.registry == "ghcr.io"
        assert cfg.registry_user == ""
        assert cfg.registry_token == ""
        assert cfg.max_dockerfile_lines == 200
        assert cfg.build_timeout_seconds == 600
        assert len(cfg.allowed_base_images) > 0


class TestDockerBuildBackend:
    def test_name(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        assert backend.name == "docker-build"
        backend2 = DockerBuildBackend(cfg, name="custom-builder")
        assert backend2.name == "custom-builder"

    @pytest.mark.asyncio
    async def test_validate_empty(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        errors = await backend.validate("")
        assert any("empty" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_validate_no_from(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        errors = await backend.validate("RUN echo hello")
        assert any("FROM" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_allowed_base_image(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        errors = await backend.validate("FROM python:3.11\nRUN pip install torch")
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_disallowed_base_image(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        errors = await backend.validate("FROM malicious/image:latest\nRUN echo hi")
        assert any("not in allowed list" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_scratch_always_allowed(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        errors = await backend.validate("FROM scratch\nCOPY binary /app")
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_too_long(self):
        cfg = DockerBuildConfig(max_dockerfile_lines=5)
        backend = DockerBuildBackend(cfg)
        dockerfile = "FROM python:3.11\n" + "\n".join(f"RUN echo {i}" for i in range(10))
        errors = await backend.validate(dockerfile)
        assert any("too long" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_validate_privileged_instructions(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        errors = await backend.validate("FROM python:3.11\nUSER root\nRUN echo hi")
        assert any("privileged" in e.lower() or "Privileged" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_multistage_build(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        dockerfile = (
            "FROM python:3.11 AS builder\n"
            "RUN pip install build\n"
            "FROM python:3.11-slim\n"
            "COPY --from=builder /app /app\n"
        )
        errors = await backend.validate(dockerfile)
        assert errors == []

    @pytest.mark.asyncio
    async def test_build_validation_failure(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        result = await backend.build("FROM malicious/image\nRUN echo hi", "test:v1")
        assert not result.success
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_build_docker_not_installed(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        # Mock subprocess to simulate docker not found
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await backend.build("FROM python:3.11\nRUN echo hi", "test:v1")
        assert not result.success
        assert "not installed" in result.error

    @pytest.mark.asyncio
    async def test_build_timeout(self):
        cfg = DockerBuildConfig(build_timeout_seconds=0)
        backend = DockerBuildBackend(cfg)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.build("FROM python:3.11\nRUN echo hi", "test:v1")
        assert not result.success
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_build_failure(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error: something broke"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.build("FROM python:3.11\nRUN echo hi", "test:v1")
        assert not result.success
        assert "something broke" in result.error

    @pytest.mark.asyncio
    async def test_build_success(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="testorg")
        backend = DockerBuildBackend(cfg)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"Successfully built", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.build("FROM python:3.11\nRUN echo hi", "train:v1")
        assert result.success
        assert result.image_ref == "ghcr.io/testorg/train:v1"

    @pytest.mark.asyncio
    async def test_push_docker_not_installed(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        with (
            patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError),
            pytest.raises(RuntimeError, match="not installed"),
        ):
            await backend.push("test:v1")

    @pytest.mark.asyncio
    async def test_push_success_no_auth(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        backend = DockerBuildBackend(cfg)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"Pushed", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            uri = await backend.push("train:v1")
        assert uri == "ghcr.io/org/train:v1"

    @pytest.mark.asyncio
    async def test_push_with_auth(self):
        cfg = DockerBuildConfig(
            registry="ghcr.io",
            namespace="org",
            registry_user="user",
            registry_token="secret-token",
        )
        backend = DockerBuildBackend(cfg)

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"ok", b""))
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            uri = await backend.push("train:v1")
        assert uri == "ghcr.io/org/train:v1"
        assert call_count == 2  # login + push

    @pytest.mark.asyncio
    async def test_push_failure(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        backend = DockerBuildBackend(cfg)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"denied"))
        mock_proc.returncode = 1

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(RuntimeError, match="Push failed"),
        ):
            await backend.push("train:v1")

    @pytest.mark.asyncio
    async def test_build_and_push_success(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        backend = DockerBuildBackend(cfg)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await backend.build_and_push("FROM python:3.11\nRUN echo hi", "train:v1")
        assert result.success
        assert result.image_ref == "ghcr.io/org/train:v1"

    @pytest.mark.asyncio
    async def test_build_and_push_build_fails(self):
        cfg = DockerBuildConfig()
        backend = DockerBuildBackend(cfg)
        result = await backend.build_and_push("FROM malicious/img\nRUN echo", "train:v1")
        assert not result.success
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_build_and_push_push_fails(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        backend = DockerBuildBackend(cfg)

        # Build succeeds, push fails
        build_proc = AsyncMock()
        build_proc.communicate = AsyncMock(return_value=(b"ok", b""))
        build_proc.returncode = 0

        push_proc = AsyncMock()
        push_proc.communicate = AsyncMock(return_value=(b"", b"denied"))
        push_proc.returncode = 1

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return build_proc
            return push_proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await backend.build_and_push("FROM python:3.11\nRUN echo hi", "train:v1")
        assert not result.success
        assert "Push failed" in result.error

    def test_full_tag(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        backend = DockerBuildBackend(cfg)
        assert backend._full_tag("train:v1") == "ghcr.io/org/train:v1"

    def test_full_tag_already_qualified(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        backend = DockerBuildBackend(cfg)
        assert backend._full_tag("ghcr.io/org/train:v1") == "ghcr.io/org/train:v1"

    def test_full_tag_no_namespace(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="")
        backend = DockerBuildBackend(cfg)
        assert backend._full_tag("train:v1") == "ghcr.io/train:v1"


# ── Render Dockerfile tests ──────────────────────────────────


class TestRenderDockerfile:
    def test_basic(self):
        result = _render_dockerfile("python:3.11", ["torch", "numpy"], "python train.py")
        assert "FROM python:3.11" in result
        assert "pip install --no-cache-dir torch numpy" in result
        assert "ENTRYPOINT" in result
        assert "python" in result
        assert "train.py" in result

    def test_no_dependencies(self):
        result = _render_dockerfile("python:3.11", [], "python train.py")
        assert "FROM python:3.11" in result
        assert "pip install" not in result

    def test_custom_workdir(self):
        result = _render_dockerfile("python:3.11", [], "echo hi", workdir="/workspace")
        assert "WORKDIR /workspace" in result

    def test_env_vars(self):
        result = _render_dockerfile("python:3.11", [], "echo hi", env={"CUDA_VISIBLE_DEVICES": "0"})
        assert "ENV CUDA_VISIBLE_DEVICES=0" in result


# ── MCP Tools: write_dockerfile and build_image ──────────────


class TestBuildMCPTools:
    def _make_state_with_build(self, build_backend=None):
        """Create a FrameworkState with compute and optional build backend."""
        from ganglion.orchestration.pipeline import PipelineDef, StageDef
        from ganglion.orchestration.task_context import MetricDef, OutputSpec, SubnetConfig, TaskDef
        from ganglion.state.agent_registry import AgentRegistry
        from ganglion.state.framework_state import FrameworkState
        from ganglion.state.tool_registry import ToolRegistry

        mock = MockBackend()
        router = ComputeRouter(
            backends={"mock": mock},
            routes=[ComputeRoute(pattern="default", backend="mock")],
        )
        return FrameworkState(
            subnet_config=SubnetConfig(
                netuid=99,
                name="Test",
                metrics=[MetricDef("acc", "maximize")],
                tasks={"main": TaskDef("main")},
                output_spec=OutputSpec(format="test"),
            ),
            pipeline_def=PipelineDef(name="test", stages=[StageDef(name="train", agent="Trainer")]),
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            compute_router=router,
            build_backend=build_backend,
        )

    def test_register_build_tools(self):
        state = self._make_state_with_build()
        register_compute_tools(state)
        assert state.tool_registry.has("write_dockerfile")
        assert state.tool_registry.has("validate_dockerfile")
        assert state.tool_registry.has("build_image")

    @pytest.mark.asyncio
    async def test_write_dockerfile_no_build_backend(self):
        state = self._make_state_with_build(build_backend=None)
        register_compute_tools(state)
        tool = state.tool_registry.get("write_dockerfile")
        result = await tool.func(
            base_image="python:3.11",
            dependencies="torch,numpy",
            entrypoint="python train.py",
            tag="test:v1",
        )
        data = json.loads(result)
        assert data["valid"] is True
        assert "FROM python:3.11" in data["dockerfile"]
        assert data["tag"] == "test:v1"

    @pytest.mark.asyncio
    async def test_write_dockerfile_with_validation(self):
        cfg = DockerBuildConfig()
        build = DockerBuildBackend(cfg)
        state = self._make_state_with_build(build_backend=build)
        register_compute_tools(state)
        tool = state.tool_registry.get("write_dockerfile")

        # Allowed base image
        result = await tool.func(
            base_image="nvidia/pytorch:24.01",
            dependencies="transformers",
            entrypoint="python train.py",
            tag="exp:v1",
        )
        data = json.loads(result)
        assert data["valid"] is True
        assert data["validation_errors"] == []

    @pytest.mark.asyncio
    async def test_write_dockerfile_validation_fails(self):
        cfg = DockerBuildConfig()
        build = DockerBuildBackend(cfg)
        state = self._make_state_with_build(build_backend=build)
        register_compute_tools(state)
        tool = state.tool_registry.get("write_dockerfile")

        result = await tool.func(
            base_image="evil/image:latest",
            dependencies="torch",
            entrypoint="python train.py",
            tag="exp:v1",
        )
        data = json.loads(result)
        assert data["valid"] is False
        assert len(data["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_build_image_no_backend(self):
        state = self._make_state_with_build(build_backend=None)
        register_compute_tools(state)
        tool = state.tool_registry.get("build_image")
        result = await tool.func(dockerfile="FROM python:3.11", tag="test:v1")
        data = json.loads(result)
        assert data["success"] is False
        assert "No build backend" in data["error"]

    @pytest.mark.asyncio
    async def test_build_image_success(self):
        cfg = DockerBuildConfig(registry="ghcr.io", namespace="org")
        build = DockerBuildBackend(cfg)
        build.build_and_push = AsyncMock(
            return_value=BuildResult(
                image_ref="ghcr.io/org/train:v1",
                success=True,
                duration_seconds=45.2,
            )
        )
        state = self._make_state_with_build(build_backend=build)
        register_compute_tools(state)
        tool = state.tool_registry.get("build_image")
        result = await tool.func(dockerfile="FROM python:3.11\nRUN echo hi", tag="train:v1")
        data = json.loads(result)
        assert data["success"] is True
        assert data["image_ref"] == "ghcr.io/org/train:v1"
