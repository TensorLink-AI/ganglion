"""Tests for the compute module — protocol, backends, router, job manager."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from ganglion.compute.artifacts import LocalArtifactStore
from ganglion.compute.backends.local import LocalBackend
from ganglion.compute.job_manager import JobManager
from ganglion.compute.protocol import JobHandle, JobResult, JobSpec, JobStatus
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
