"""DockerBuildBackend — build and push container images for compute jobs.

The bot writes a Dockerfile (pure text generation). This backend handles
the privileged operations: validation, build, and push. Registry credentials
are server-side only, never exposed to the agent layer.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ganglion.compute.protocol import BuildResult

logger = logging.getLogger(__name__)

# Dockerfile instructions that can introduce security risks
_PRIVILEGED_INSTRUCTIONS = re.compile(
    r"^\s*(USER\s+root|--privileged|--security-opt|--cap-add)",
    re.MULTILINE | re.IGNORECASE,
)

# Matches FROM lines to extract base image
_FROM_PATTERN = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class DockerBuildConfig:
    """Configuration for the Docker build backend.

    Credentials are server-side only — the bot never sees registry tokens.
    """

    registry: str = "ghcr.io"
    registry_user: str = ""
    registry_token: str = ""
    namespace: str = ""
    allowed_base_images: list[str] = field(default_factory=lambda: [
        "python:*",
        "nvidia/cuda:*",
        "nvidia/pytorch:*",
        "nvidia/tensorflow:*",
        "pytorch/pytorch:*",
        "ubuntu:*",
        "debian:*",
    ])
    max_dockerfile_lines: int = 200
    build_timeout_seconds: int = 600


def _match_glob(pattern: str, value: str) -> bool:
    """Simple glob matching for base image whitelist."""
    if pattern == value:
        return True
    if pattern.endswith(":*"):
        prefix = pattern[:-2]
        # Match "nvidia/pytorch" against "nvidia/pytorch:24.01"
        # Also match "nvidia/pytorch:latest"
        return value == prefix or value.startswith(prefix + ":")
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return False


class DockerBuildBackend:
    """Build and push Docker images for compute jobs.

    Flow:
    1. Bot generates Dockerfile text (via write_dockerfile tool)
    2. validate() checks base image whitelist + lint rules
    3. build() writes Dockerfile to temp dir, runs `docker build`
    4. push() tags and pushes to the configured registry

    The bot never touches steps 3-4 directly. Credentials stay server-side.
    """

    def __init__(self, config: DockerBuildConfig, name: str = "docker-build"):
        self._config = config
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def validate(self, dockerfile: str) -> list[str]:
        """Validate a Dockerfile before building.

        Checks:
        - Not empty
        - Not too long
        - All FROM images are in the allowed list
        - No privileged instructions
        """
        errors: list[str] = []

        if not dockerfile.strip():
            return ["Dockerfile is empty"]

        lines = dockerfile.strip().splitlines()
        if len(lines) > self._config.max_dockerfile_lines:
            errors.append(
                f"Dockerfile too long ({len(lines)} lines, max {self._config.max_dockerfile_lines})"
            )

        # Check base images against whitelist
        from_matches = _FROM_PATTERN.findall(dockerfile)
        if not from_matches:
            errors.append("No FROM instruction found")
        else:
            for base_image in from_matches:
                if base_image.lower() == "scratch":
                    continue  # scratch is always allowed
                if not any(
                    _match_glob(allowed, base_image)
                    for allowed in self._config.allowed_base_images
                ):
                    errors.append(
                        f"Base image '{base_image}' not in allowed list. "
                        f"Allowed: {self._config.allowed_base_images}"
                    )

        # Check for privileged instructions
        privileged = _PRIVILEGED_INSTRUCTIONS.findall(dockerfile)
        if privileged:
            errors.append(
                f"Privileged instructions not allowed: {privileged}"
            )

        return errors

    async def build(self, dockerfile: str, tag: str) -> BuildResult:
        """Build an image from a Dockerfile string.

        Runs validate() first. If validation fails, returns a failed BuildResult
        without attempting the build.
        """
        validation_errors = await self.validate(dockerfile)
        if validation_errors:
            return BuildResult(
                image_ref="",
                success=False,
                error=f"Validation failed: {'; '.join(validation_errors)}",
            )

        start = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="ganglion-build-") as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile)

            full_tag = self._full_tag(tag)
            cmd = ["docker", "build", "-t", full_tag, "-f", str(dockerfile_path), tmpdir]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._config.build_timeout_seconds,
                )
            except asyncio.TimeoutError:
                return BuildResult(
                    image_ref="",
                    success=False,
                    error=f"Build timed out after {self._config.build_timeout_seconds}s",
                    duration_seconds=time.monotonic() - start,
                )
            except FileNotFoundError:
                return BuildResult(
                    image_ref="",
                    success=False,
                    error="Docker is not installed or not in PATH",
                    duration_seconds=time.monotonic() - start,
                )

        duration = time.monotonic() - start

        if proc.returncode != 0:
            error_msg = (stderr or b"").decode(errors="replace")[-500:]
            return BuildResult(
                image_ref="",
                success=False,
                error=f"Build failed (exit {proc.returncode}): {error_msg}",
                duration_seconds=duration,
            )

        logger.info("Built image %s in %.1fs", full_tag, duration)
        return BuildResult(
            image_ref=full_tag,
            success=True,
            duration_seconds=duration,
        )

    async def push(self, tag: str) -> str:
        """Push a built image to the configured registry.

        Handles registry login if credentials are configured.
        Returns the full registry URI on success, raises on failure.
        """
        full_tag = self._full_tag(tag)

        # Login if credentials are available
        if self._config.registry_token:
            login_cmd = [
                "docker", "login", self._config.registry,
                "-u", self._config.registry_user or "_token",
                "--password-stdin",
            ]
            try:
                login_proc = await asyncio.create_subprocess_exec(
                    *login_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await login_proc.communicate(input=self._config.registry_token.encode())
            except FileNotFoundError:
                raise RuntimeError("Docker is not installed or not in PATH")

            if login_proc.returncode != 0:
                raise RuntimeError(f"Docker login to {self._config.registry} failed")

        # Push
        push_cmd = ["docker", "push", full_tag]
        try:
            proc = await asyncio.create_subprocess_exec(
                *push_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._config.build_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Push timed out after {self._config.build_timeout_seconds}s"
            )
        except FileNotFoundError:
            raise RuntimeError("Docker is not installed or not in PATH")

        if proc.returncode != 0:
            error_msg = (stderr or b"").decode(errors="replace")[-500:]
            raise RuntimeError(f"Push failed (exit {proc.returncode}): {error_msg}")

        logger.info("Pushed image %s", full_tag)
        return full_tag

    async def build_and_push(self, dockerfile: str, tag: str) -> BuildResult:
        """Convenience: validate, build, and push in one call.

        This is the primary method the framework calls. The bot never
        calls this directly — it writes a Dockerfile, and infrastructure
        invokes this.
        """
        result = await self.build(dockerfile, tag)
        if not result.success:
            return result

        try:
            uri = await self.push(tag)
            return BuildResult(
                image_ref=uri,
                success=True,
                duration_seconds=result.duration_seconds,
            )
        except RuntimeError as e:
            return BuildResult(
                image_ref="",
                success=False,
                error=str(e),
                duration_seconds=result.duration_seconds,
            )

    def _full_tag(self, tag: str) -> str:
        """Build the full image reference including registry and namespace."""
        parts = [self._config.registry]
        if self._config.namespace:
            parts.append(self._config.namespace)
        # If tag already contains the registry prefix, use as-is
        if tag.startswith(self._config.registry):
            return tag
        return "/".join(parts) + f"/{tag}" if parts[0] else tag
