"""Tests for S3ArtifactStore — uses mocked boto3 client."""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from ganglion.compute.artifacts import ArtifactMeta

# ── Mock boto3 so tests run without it installed ──────────

_mock_boto3 = ModuleType("boto3")
_mock_boto3.client = MagicMock()  # will be overridden per-test


def _make_mock_client():
    """Build a mock S3 client with an in-memory object store."""
    client = MagicMock()
    objects: dict[str, bytes] = {}

    def put_object(Bucket, Key, Body, **kwargs):
        objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(Bucket, Key, **kwargs):
        if Key not in objects:
            raise client.exceptions.NoSuchKey(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject"
            )
        body = MagicMock()
        body.read.return_value = objects[Key]
        return {"Body": body}

    def head_object(Bucket, Key, **kwargs):
        if Key not in objects:
            raise Exception("NoSuchKey")
        return {"ContentLength": len(objects[Key]), "ContentType": "application/octet-stream"}

    def delete_object(Bucket, Key, **kwargs):
        objects.pop(Key, None)

    def list_objects_v2(Bucket, Prefix="", **kwargs):
        matching = [k for k in objects if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in matching]}

    paginator = MagicMock()
    paginator.paginate.side_effect = lambda Bucket, Prefix="", **kw: [
        list_objects_v2(Bucket, Prefix)
    ]

    def generate_presigned_url(method, Params, ExpiresIn=3600):
        return f"https://presigned.example.com/{Params['Key']}?expires={ExpiresIn}"

    client.put_object = put_object
    client.get_object = get_object
    client.head_object = head_object
    client.delete_object = delete_object
    client.get_paginator = MagicMock(return_value=paginator)
    client.generate_presigned_url = generate_presigned_url
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

    return client, objects


def _make_store(client, **kwargs):
    """Create an S3ArtifactStore with a mocked boto3 client."""
    mock_boto3 = ModuleType("boto3")
    mock_boto3.client = MagicMock(return_value=client)

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        # Force re-import so the mock takes effect
        if "ganglion.compute.artifacts_s3" in sys.modules:
            del sys.modules["ganglion.compute.artifacts_s3"]
        from ganglion.compute.artifacts_s3 import S3ArtifactStore

        return S3ArtifactStore(bucket="test-bucket", **kwargs)


class TestS3ArtifactStore:
    @pytest.fixture
    def setup(self):
        client, objects = _make_mock_client()
        store = _make_store(client)
        return store, client, objects

    @pytest.mark.asyncio
    async def test_put_and_get(self, setup):
        store, client, objects = setup
        await store.put("run-1/model.pt", b"fake weights")

        data = await store.get("run-1/model.pt")
        assert data == b"fake weights"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, setup):
        store, _, _ = setup
        data = await store.get("nope")
        assert data is None

    @pytest.mark.asyncio
    async def test_put_stores_meta_sidecar(self, setup):
        store, _, objects = setup
        meta = ArtifactMeta(key="run-1/model.pt", run_id="run-1", stage="train")
        await store.put("run-1/model.pt", b"weights", meta)

        meta_key = "__meta__/run-1/model.pt.json"
        assert meta_key in objects
        parsed = json.loads(objects[meta_key])
        assert parsed["run_id"] == "run-1"
        assert parsed["stage"] == "train"
        assert parsed["size_bytes"] == len(b"weights")

    @pytest.mark.asyncio
    async def test_get_meta(self, setup):
        store, _, _ = setup
        meta = ArtifactMeta(key="run-1/config.json", run_id="run-1", source_bot="bot-1")
        await store.put("run-1/config.json", b"{}", meta)

        retrieved = await store.get_meta("run-1/config.json")
        assert retrieved is not None
        assert retrieved.run_id == "run-1"
        assert retrieved.source_bot == "bot-1"

    @pytest.mark.asyncio
    async def test_get_meta_no_sidecar_synthesizes(self, setup):
        store, client, objects = setup
        objects["run-1/raw.bin"] = b"raw data"

        retrieved = await store.get_meta("run-1/raw.bin")
        assert retrieved is not None
        assert retrieved.key == "run-1/raw.bin"

    @pytest.mark.asyncio
    async def test_get_meta_missing(self, setup):
        store, _, _ = setup
        assert await store.get_meta("nope") is None

    @pytest.mark.asyncio
    async def test_list_excludes_meta(self, setup):
        store, _, _ = setup
        await store.put("run-1/model.pt", b"weights")
        await store.put("run-1/config.json", b"{}")

        keys = await store.list()
        assert all("__meta__" not in k for k in keys)
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_list_with_prefix(self, setup):
        store, _, _ = setup
        await store.put("run-1/model.pt", b"weights")
        await store.put("run-2/model.pt", b"other")

        keys = await store.list("run-1")
        assert all(k.startswith("run-1") for k in keys)

    @pytest.mark.asyncio
    async def test_list_meta(self, setup):
        store, _, _ = setup
        await store.put(
            "run-1/a.pt", b"x",
            ArtifactMeta(key="run-1/a.pt", run_id="run-1", experiment_id="exp-1"),
        )
        await store.put(
            "run-1/b.pt", b"y",
            ArtifactMeta(key="run-1/b.pt", run_id="run-1", experiment_id="exp-1"),
        )

        metas = await store.list_meta("run-1")
        assert len(metas) == 2
        assert all(m.experiment_id == "exp-1" for m in metas)

    @pytest.mark.asyncio
    async def test_delete(self, setup):
        store, _, objects = setup
        await store.put("temp.txt", b"data")
        assert await store.delete("temp.txt")

        assert await store.get("temp.txt") is None
        assert all("temp.txt" not in k for k in objects)

    @pytest.mark.asyncio
    async def test_source_bot_roundtrip(self, setup):
        store, _, _ = setup
        meta = ArtifactMeta(key="run-1/model.pt", source_bot="claw-bot")
        await store.put("run-1/model.pt", b"data", meta)

        retrieved = await store.get_meta("run-1/model.pt")
        assert retrieved is not None
        assert retrieved.source_bot == "claw-bot"


class TestS3ArtifactStoreURLs:
    @pytest.mark.asyncio
    async def test_public_url_base(self):
        client, _ = _make_mock_client()
        store = _make_store(client, public_url_base="https://cdn.example.com")
        await store.put("run-1/model.pt", b"weights")

        url = await store.get_url("run-1/model.pt")
        assert url == "https://cdn.example.com/run-1/model.pt"

    @pytest.mark.asyncio
    async def test_public_url_in_meta(self):
        client, _ = _make_mock_client()
        store = _make_store(client, public_url_base="https://cdn.example.com")
        await store.put("run-1/model.pt", b"weights")

        meta = await store.get_meta("run-1/model.pt")
        assert meta is not None
        assert meta.url == "https://cdn.example.com/run-1/model.pt"

    @pytest.mark.asyncio
    async def test_presigned_url_fallback(self):
        client, _ = _make_mock_client()
        store = _make_store(client)  # no public_url_base
        await store.put("run-1/model.pt", b"weights")

        url = await store.get_url("run-1/model.pt")
        assert url is not None
        assert "presigned" in url

    @pytest.mark.asyncio
    async def test_url_nonexistent_returns_none(self):
        client, _ = _make_mock_client()
        store = _make_store(client)

        url = await store.get_url("nope/nothing")
        assert url is None

    @pytest.mark.asyncio
    async def test_prefix_in_url(self):
        client, _ = _make_mock_client()
        store = _make_store(client, prefix="ganglion/", public_url_base="https://cdn.example.com")
        await store.put("run-1/model.pt", b"weights")

        url = await store.get_url("run-1/model.pt")
        assert url == "https://cdn.example.com/ganglion/run-1/model.pt"

    @pytest.mark.asyncio
    async def test_local_store_get_url_returns_none(self):
        """LocalArtifactStore.get_url always returns None."""
        import tempfile
        from pathlib import Path

        from ganglion.compute.artifacts import LocalArtifactStore

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=Path(d))
            await store.put("test.txt", b"hello")
            assert await store.get_url("test.txt") is None


class TestArtifactMetaURL:
    def test_url_field_in_to_dict(self):
        meta = ArtifactMeta(key="test", url="https://example.com/test")
        d = meta.to_dict()
        assert d["url"] == "https://example.com/test"

    def test_url_field_from_dict(self):
        meta = ArtifactMeta.from_dict({
            "key": "test",
            "url": "https://example.com/test",
        })
        assert meta.url == "https://example.com/test"

    def test_url_none_by_default(self):
        meta = ArtifactMeta(key="test")
        assert meta.url is None
