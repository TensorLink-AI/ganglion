"""S3ArtifactStore — S3-compatible artifact storage backend.

Works with any S3-compatible service:
  - Cloudflare R2
  - AWS S3
  - Minio
  - Hippius (S3-compatible gateway)

Requires ``boto3``::

    pip install ganglion[s3]
    # or: pip install boto3

Configuration in ``config.py``::

    from ganglion.compute.artifacts_s3 import S3ArtifactStore

    artifact_store = S3ArtifactStore(
        bucket="my-ganglion-artifacts",
        endpoint_url="https://<account>.r2.cloudflarestorage.com",
        access_key_id="...",
        secret_access_key="...",
        public_url_base="https://artifacts.my-domain.com",  # optional
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ganglion.compute.artifacts import ArtifactMeta

logger = logging.getLogger(__name__)

# Meta objects are stored as JSON under a parallel __meta__/ prefix
_META_PREFIX = "__meta__/"


class S3ArtifactStore:
    """Store artifacts in an S3-compatible bucket.

    Keys map directly to S3 object keys within the configured prefix.
    Metadata sidecars are stored as ``__meta__/{key}.json``.

    URLs are constructed from ``public_url_base`` if provided, otherwise
    presigned URLs are generated with configurable expiry.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        public_url_base: str | None = None,
        presign_expiry: int = 3600,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.public_url_base = public_url_base.rstrip("/") if public_url_base else None
        self.presign_expiry = presign_expiry

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3ArtifactStore. "
                "Install it with: pip install ganglion[s3]  (or: pip install boto3)"
            ) from exc

        kwargs: dict[str, Any] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region_name:
            kwargs["region_name"] = region_name
        if access_key_id:
            kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            kwargs["aws_secret_access_key"] = secret_access_key

        self._client = boto3.client("s3", **kwargs)

    def _obj_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _meta_key(self, key: str) -> str:
        return f"{self.prefix}{_META_PREFIX}{key}.json"

    # ── Core protocol ──────────────────────────────────────

    async def put(self, key: str, data: bytes, meta: ArtifactMeta | None = None) -> None:
        obj_key = self._obj_key(key)

        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": obj_key,
            "Body": data,
        }
        if meta and meta.content_type:
            put_kwargs["ContentType"] = meta.content_type

        self._client.put_object(**put_kwargs)

        # Build and store metadata sidecar
        m = meta or ArtifactMeta(key=key)
        m.key = key
        m.size_bytes = len(data)
        if not m.created_at:
            m.created_at = time.time()
        m.url = self._build_url(key)

        meta_body = json.dumps(m.to_dict()).encode()
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._meta_key(key),
            Body=meta_body,
            ContentType="application/json",
        )

    async def get(self, key: str) -> bytes | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._obj_key(key))
            return resp["Body"].read()
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            if "NoSuchKey" in str(type(e).__name__) or "404" in str(e):
                return None
            logger.warning("S3 get failed for '%s': %s", key, e)
            return None

    async def get_url(self, key: str) -> str | None:
        """Return a URL for the artifact.

        Uses ``public_url_base`` if configured, otherwise generates a
        presigned URL valid for ``presign_expiry`` seconds.
        """
        url = self._build_url(key)
        if url:
            return url
        # No public URL — check the object exists before presigning
        meta = await self.get_meta(key)
        if meta is None:
            return None
        return self._presign_url(key)

    async def get_meta(self, key: str) -> ArtifactMeta | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._meta_key(key))
            data = json.loads(resp["Body"].read())
            return ArtifactMeta.from_dict(data)
        except Exception:
            # No sidecar — check if the object exists and synthesize
            try:
                head = self._client.head_object(Bucket=self.bucket, Key=self._obj_key(key))
                return ArtifactMeta(
                    key=key,
                    size_bytes=head.get("ContentLength", 0),
                    content_type=head.get("ContentType", ""),
                    url=self._build_url(key),
                )
            except Exception:
                return None

    async def list(self, prefix: str = "") -> list[str]:
        full_prefix = self._obj_key(prefix)
        keys: list[str] = []

        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                obj_key = obj["Key"]
                # Skip metadata sidecars
                if f"/{_META_PREFIX}" in obj_key or obj_key.startswith(f"{self.prefix}{_META_PREFIX}"):
                    continue
                # Strip prefix to get the artifact key
                rel_key = obj_key[len(self.prefix) :] if self.prefix else obj_key
                keys.append(rel_key)

        return keys

    async def list_meta(self, prefix: str = "") -> list[ArtifactMeta]:
        keys = await self.list(prefix)
        metas = []
        for key in keys:
            meta = await self.get_meta(key)
            if meta:
                metas.append(meta)
        return metas

    async def delete(self, key: str) -> bool:
        deleted = False
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._obj_key(key))
            deleted = True
        except Exception as e:
            logger.warning("S3 delete failed for '%s': %s", key, e)

        # Clean up sidecar
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._meta_key(key))
        except Exception:
            pass

        return deleted

    # ── URL helpers ────────────────────────────────────────

    def _build_url(self, key: str) -> str | None:
        """Build a public URL if public_url_base is configured."""
        if not self.public_url_base:
            return None
        obj_key = self._obj_key(key)
        return f"{self.public_url_base}/{obj_key}"

    def _presign_url(self, key: str) -> str:
        """Generate a presigned URL for temporary access."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._obj_key(key)},
            ExpiresIn=self.presign_expiry,
        )
