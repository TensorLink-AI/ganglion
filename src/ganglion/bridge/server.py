"""FastAPI HTTP bridge exposing the framework to OpenClaw and external tools."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ganglion.knowledge.types import KnowledgeQuery
from ganglion.state.framework_state import FrameworkState

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ganglion Bridge",
    version="0.1.0",
    docs_url="/v1/docs",
    openapi_url="/v1/openapi.json",
)

# State is set at startup via configure()
_state: FrameworkState | None = None
_config: Any = None

# ── Rate limiter (in-memory, per-IP) ──────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW_SECONDS = 60


def _check_rate_limit(client_ip: str, max_requests: int) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS
    requests = _rate_limit_store[client_ip]
    # Prune old entries
    _rate_limit_store[client_ip] = [t for t in requests if t > window_start]
    if len(_rate_limit_store[client_ip]) >= max_requests:
        return False
    _rate_limit_store[client_ip].append(now)
    return True


# ── Middleware ─────────────────────────────────────────────


@app.middleware("http")
async def request_middleware(
    request: Request,
    call_next: Callable[[Request], Coroutine[Any, Any, Response]],
) -> Response:
    """Add request ID, security headers, rate limiting, timing, and size limits."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.monotonic()

    # Request size limit
    if _config and request.headers.get("content-length"):
        content_length = int(request.headers.get("content-length", 0))
        if content_length > _config.max_request_body_bytes:
            return Response(status_code=413, content="Request body too large")

    # Rate limiting
    if _config:
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip, _config.rate_limit_requests_per_minute):
            logger.warning(
                "Rate limit exceeded",
                extra={"request_id": request_id, "client_ip": client_ip},
            )
            return Response(status_code=429, content="Rate limit exceeded")

    # Process request
    response = await call_next(request)

    # Timing
    elapsed_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    # Security headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response


# ── Configuration ──────────────────────────────────────────


def configure(state: FrameworkState, config: Any = None) -> None:
    """Configure the bridge with a FrameworkState instance."""
    global _state, _config
    _state = state
    _config = config


def setup_cors(allowed_origins: list[str] | None = None) -> None:
    """Set up CORS middleware. Must be called before app starts."""
    origins = allowed_origins or ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )


def _get_state() -> FrameworkState:
    if _state is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "NOT_CONFIGURED",
                    "message": "Bridge not configured. Call configure() first.",
                }
            },
        )
    return _state


# ── Response helpers ───────────────────────────────────────


def _success_response(data: Any) -> dict[str, Any]:
    """Wrap successful response in standard envelope."""
    return {"data": data}


def _error_response(code: str, message: str, status_code: int = 400) -> None:
    """Raise HTTPException with standard error envelope."""
    raise HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message}},
    )


# ── Request models ──────────────────────────────────────────


class WriteToolRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., min_length=1, max_length=100_000)
    category: str = Field(default="general", max_length=100)
    test_code: str | None = Field(default=None, max_length=100_000)


class WriteAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., min_length=1, max_length=100_000)
    test_task: dict[str, Any] | None = None


class WriteComponentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., min_length=1, max_length=100_000)
    component_type: str = Field(default="general", max_length=100)


class WritePromptRequest(BaseModel):
    agent_name: str = Field(..., min_length=1, max_length=200)
    prompt_section: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=50_000)


class PatchPipelineRequest(BaseModel):
    operations: list[dict[str, Any]] = Field(..., min_length=1, max_length=50)


class SwapPolicyRequest(BaseModel):
    retry_policy: dict[str, Any]


class RunPipelineRequest(BaseModel):
    overrides: dict[str, Any] | None = None


class RunStageRequest(BaseModel):
    context: dict[str, Any] | None = None


class RunExperimentRequest(BaseModel):
    config: dict[str, Any]


class ConnectMCPServerRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    transport: str = Field(default="stdio", pattern="^(stdio|sse)$")
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None
    tool_prefix: str = Field(default="", max_length=100)
    category: str = Field(default="mcp", max_length=100)
    timeout: float = Field(default=30.0, gt=0)


# ── Health endpoints ───────────────────────────────────────


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/readyz")
async def readiness_check() -> dict[str, str]:
    """Readiness probe — returns 200 if the bridge is configured and ready."""
    if _state is None:
        raise HTTPException(
            status_code=503,
            detail={"error": {"code": "NOT_READY", "message": "Bridge not yet configured"}},
        )
    return {"status": "ready"}


# ── Observation endpoints (v1) ─────────────────────────────


@app.get("/v1/status")
async def get_status() -> dict[str, Any]:
    """Full framework state snapshot."""
    state = _get_state()
    return _success_response(await state.describe())


@app.get("/v1/pipeline")
async def get_pipeline() -> dict[str, Any]:
    """Current pipeline definition."""
    return _success_response(_get_state().pipeline_def.to_dict())


@app.get("/v1/tools")
async def get_tools(category: str | None = None) -> dict[str, Any]:
    """Registered tools."""
    return _success_response(_get_state().tool_registry.list_all(category=category))


@app.get("/v1/agents")
async def get_agents() -> dict[str, Any]:
    """Registered agents."""
    return _success_response(_get_state().agent_registry.list_all())


@app.get("/v1/runs")
async def get_run_history(n: int = 10) -> dict[str, Any]:
    """Past pipeline runs."""
    if n < 1 or n > 1000:
        _error_response("INVALID_PARAM", "n must be between 1 and 1000")
    state = _get_state()
    if state.persistence is None:
        return _success_response([])
    return _success_response(await state.persistence.load_run_history(n=n))


@app.get("/v1/metrics")
async def get_metrics(experiment_id: str | None = None) -> dict[str, Any]:
    """Experiment metrics."""
    state = _get_state()
    if state.persistence is None:
        return _success_response([])
    return _success_response(await state.persistence.query_metrics(experiment_id=experiment_id))


@app.get("/v1/leaderboard")
async def get_leaderboard() -> dict[str, Any]:
    """Current Bittensor subnet leaderboard."""
    state = _get_state()
    subnet_client = getattr(state, "subnet_client", None)
    if subnet_client and hasattr(subnet_client, "get_leaderboard"):
        return _success_response(await subnet_client.get_leaderboard())
    return _success_response([])


@app.get("/v1/knowledge")
async def get_knowledge(capability: str | None = None, max_entries: int = 20) -> dict[str, Any]:
    """Knowledge store contents."""
    if max_entries < 1 or max_entries > 1000:
        _error_response("INVALID_PARAM", "max_entries must be between 1 and 1000")
    state = _get_state()
    if not state.knowledge:
        return _success_response({"patterns": [], "antipatterns": [], "summary": None})
    query = KnowledgeQuery(capability=capability, max_entries=max_entries)
    return _success_response(
        {
            "patterns": [p.__dict__ for p in await state.knowledge.backend.query_patterns(query)],
            "antipatterns": [
                a.__dict__ for a in await state.knowledge.backend.query_antipatterns(query)
            ],
            "summary": await state.knowledge.summary(),
        }
    )


@app.get("/v1/source/{path:path}")
async def get_source(path: str) -> dict[str, Any]:
    """Read source code of any file in the project."""
    state = _get_state()
    # Sanitize path to prevent directory traversal
    if ".." in path or path.startswith("/"):
        _error_response("INVALID_PATH", "Path must be relative and cannot contain '..'")
    full_path = state.project_root / path
    try:
        resolved = full_path.resolve()
        if not str(resolved).startswith(str(state.project_root.resolve())):
            _error_response("INVALID_PATH", "Path escapes project root", status_code=403)
    except (OSError, ValueError):
        _error_response("INVALID_PATH", "Invalid path")
    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Not found: {path}",
                }
            },
        )
    try:
        content = full_path.read_text()
    except OSError as exc:
        _error_response("READ_ERROR", f"Failed to read file: {exc}", status_code=500)
    return _success_response({"path": path, "content": content})


@app.get("/v1/components")
async def get_components() -> dict[str, Any]:
    """Available model components in the training framework."""
    state = _get_state()
    training_framework = getattr(state, "training_framework", None)
    if training_framework and hasattr(training_framework, "list_components"):
        return _success_response(training_framework.list_components())
    return _success_response([])


# ── Mutation endpoints (v1) ────────────────────────────────


@app.post("/v1/tools", status_code=201)
async def write_tool(body: WriteToolRequest) -> dict[str, Any]:
    """Write and register a new tool."""
    state = _get_state()
    result = await state.write_and_register_tool(
        body.name, body.code, body.category, body.test_code
    )
    if not result.success:
        _error_response("VALIDATION_FAILED", "; ".join(result.errors))
    return _success_response({"path": result.path})


@app.post("/v1/agents", status_code=201)
async def write_agent(body: WriteAgentRequest) -> dict[str, Any]:
    """Write and register a new agent."""
    state = _get_state()
    result = await state.write_and_register_agent(body.name, body.code, body.test_task)
    if not result.success:
        _error_response("VALIDATION_FAILED", "; ".join(result.errors))
    return _success_response({"path": result.path})


@app.post("/v1/components", status_code=201)
async def write_component(body: WriteComponentRequest) -> dict[str, Any]:
    """Write a new model component (backbone, head, loss, etc.)."""
    state = _get_state()
    training_framework = getattr(state, "training_framework", None)
    if training_framework and hasattr(training_framework, "write_component"):
        result = training_framework.write_component(body.name, body.code, body.component_type)
        return _success_response({"result": result})
    path = state.project_root / "components" / f"{body.name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(body.code)
    except OSError as exc:
        _error_response("WRITE_ERROR", f"Failed to write component: {exc}", status_code=500)
    return _success_response({"path": str(path)})


@app.post("/v1/prompts")
async def write_prompt(body: WritePromptRequest) -> dict[str, Any]:
    """Write or replace a prompt section for an existing agent."""
    state = _get_state()
    result = await state.update_prompt(body.agent_name, body.prompt_section, body.content)
    if not result.success:
        _error_response("VALIDATION_FAILED", "; ".join(result.errors))
    return _success_response({"path": result.path})


@app.patch("/v1/pipeline")
async def patch_pipeline(body: PatchPipelineRequest) -> dict[str, Any]:
    """Apply pipeline modifications."""
    state = _get_state()
    result = await state.apply_pipeline_patch(body.operations)
    if not result.success:
        _error_response("PIPELINE_ERROR", "; ".join(result.errors))
    return _success_response({"pipeline": result.pipeline})


@app.put("/v1/policies/{stage_name}")
async def swap_policy(stage_name: str, body: SwapPolicyRequest) -> dict[str, Any]:
    """Swap retry policy for a stage."""
    state = _get_state()
    result = await state.swap_policy(
        stage_name if stage_name != "default" else None,
        body.retry_policy,
    )
    if not result.success:
        _error_response("POLICY_ERROR", "; ".join(result.errors))
    return _success_response(None)


# ── Execution endpoints (v1) ──────────────────────────────


@app.post("/v1/run/pipeline")
async def run_pipeline(body: RunPipelineRequest | None = None) -> dict[str, Any]:
    """Execute full pipeline."""
    state = _get_state()
    try:
        result = await state.run_pipeline(overrides=body.overrides if body else None)
    except Exception as exc:
        logger.error("Pipeline execution failed", exc_info=True)
        _error_response("EXECUTION_ERROR", str(exc), status_code=500)
    return _success_response(result.to_dict())


@app.post("/v1/run/stage/{stage_name}")
async def run_stage(stage_name: str, body: RunStageRequest | None = None) -> dict[str, Any]:
    """Execute a single pipeline stage."""
    state = _get_state()
    try:
        result = await state.run_single_stage(stage_name, body.context if body else None)
    except Exception as exc:
        logger.error("Stage execution failed: %s", stage_name, exc_info=True)
        _error_response("EXECUTION_ERROR", str(exc), status_code=500)
    return _success_response(result.to_dict())


@app.post("/v1/run/experiment")
async def run_experiment(body: RunExperimentRequest) -> dict[str, Any]:
    """Run a single experiment directly (bypass pipeline)."""
    state = _get_state()
    try:
        result = await state.run_direct_experiment(body.config)
    except Exception as exc:
        logger.error("Experiment execution failed", exc_info=True)
        _error_response("EXECUTION_ERROR", str(exc), status_code=500)
    return _success_response(result)


# ── Rollback endpoints (v1) ───────────────────────────────


@app.post("/v1/rollback/last")
async def rollback_last() -> dict[str, Any]:
    """Undo the most recent mutation."""
    state = _get_state()
    result = await state.rollback_last()
    if not result.success:
        _error_response("ROLLBACK_ERROR", "; ".join(result.errors))
    return _success_response(None)


@app.post("/v1/rollback/{index}")
async def rollback_to(index: int) -> dict[str, Any]:
    """Undo all mutations back to the given index."""
    if index < 0:
        _error_response("INVALID_PARAM", "Index must be >= 0")
    state = _get_state()
    result = await state.rollback_to(index)
    if not result.success:
        _error_response("ROLLBACK_ERROR", "; ".join(result.errors))
    return _success_response(None)


# ── MCP endpoints (v1) ────────────────────────────────────


@app.get("/v1/mcp")
async def get_mcp_status() -> dict[str, Any]:
    """MCP integration status — connected servers and their tools."""
    state = _get_state()
    return _success_response(state._describe_mcp())


@app.post("/v1/mcp/servers", status_code=201)
async def connect_mcp_server(body: ConnectMCPServerRequest) -> dict[str, Any]:
    """Dynamically connect to an external MCP server and register its tools."""
    from ganglion.mcp.config import MCPClientConfig

    state = _get_state()
    config = MCPClientConfig(
        name=body.name,
        transport=body.transport,
        command=body.command,
        url=body.url,
        env=body.env,
        tool_prefix=body.tool_prefix or body.name,
        category=body.category,
        timeout=body.timeout,
    )
    result = await state.connect_mcp_server(config)
    if not result.success:
        _error_response("MCP_CONNECTION_ERROR", "; ".join(result.errors))
    return _success_response(state._describe_mcp())


@app.delete("/v1/mcp/servers/{name}")
async def disconnect_mcp_server(name: str) -> dict[str, Any]:
    """Disconnect from an MCP server and unregister its tools."""
    state = _get_state()
    result = await state.disconnect_mcp_server(name)
    if not result.success:
        _error_response("MCP_DISCONNECT_ERROR", "; ".join(result.errors))
    return _success_response(state._describe_mcp())


@app.post("/v1/mcp/servers/{name}/reconnect")
async def reconnect_mcp_server(name: str) -> dict[str, Any]:
    """Reconnect to a failed MCP server."""
    state = _get_state()
    if name not in state._mcp_bridges:
        _error_response("NOT_FOUND", f"MCP server '{name}' not connected", status_code=404)

    bridge = state._mcp_bridges[name]
    config = bridge.config

    # Disconnect and reconnect
    disconnect_result = await state.disconnect_mcp_server(name)
    if not disconnect_result.success:
        _error_response("MCP_DISCONNECT_ERROR", "; ".join(disconnect_result.errors))

    connect_result = await state.connect_mcp_server(config)
    if not connect_result.success:
        _error_response("MCP_CONNECTION_ERROR", "; ".join(connect_result.errors))

    return _success_response(state._describe_mcp())


# ── Artifact endpoints (v1) ────────────────────────────────


class StoreArtifactRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    encoding: str = Field(default="utf-8", pattern="^(utf-8|base64)$")
    run_id: str = Field(default="", max_length=200)
    experiment_id: str = Field(default="", max_length=200)
    stage: str = Field(default="", max_length=200)
    content_type: str = Field(default="", max_length=200)


@app.get("/v1/artifacts")
async def list_artifacts(
    run_id: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """List artifacts, optionally filtered by run or experiment ID."""
    state = _get_state()
    if state.artifact_store is None:
        return _success_response({"artifacts": [], "count": 0})

    prefix = run_id or ""
    metas = await state.artifact_store.list_meta(prefix)

    if experiment_id:
        metas = [m for m in metas if m.experiment_id == experiment_id]

    # Enrich with URLs for remote stores
    artifacts = []
    for m in metas:
        entry = m.to_dict()
        if not entry.get("url"):
            url = await state.artifact_store.get_url(m.key)
            if url:
                entry["url"] = url
        artifacts.append(entry)

    return _success_response({
        "artifacts": artifacts,
        "count": len(artifacts),
    })


@app.get("/v1/artifacts/{key:path}")
async def get_artifact(key: str, encoding: str = "utf-8") -> dict[str, Any]:
    """Retrieve a single artifact by key.

    Use encoding=base64 for binary artifacts (model weights).
    Use encoding=utf-8 for text artifacts (code, configs).
    """
    state = _get_state()
    if state.artifact_store is None:
        _error_response("NO_ARTIFACTS", "No artifact store configured", status_code=404)

    data = await state.artifact_store.get(key)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": f"Artifact '{key}' not found"}},
        )

    meta = await state.artifact_store.get_meta(key)

    if encoding == "base64":
        import base64
        content = base64.b64encode(data).decode("ascii")
    else:
        try:
            content = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            import base64
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"

    result: dict[str, Any] = {
        "key": key,
        "encoding": encoding,
        "size_bytes": len(data),
        "content": content,
    }
    if meta:
        result["meta"] = meta.to_dict()
    url = (meta.url if meta and meta.url else None) or await state.artifact_store.get_url(key)
    if url:
        result["url"] = url
    return _success_response(result)


@app.post("/v1/artifacts", status_code=201)
async def store_artifact(body: StoreArtifactRequest) -> dict[str, Any]:
    """Store an artifact."""
    state = _get_state()
    if state.artifact_store is None:
        _error_response("NO_ARTIFACTS", "No artifact store configured", status_code=404)

    if body.encoding == "base64":
        import base64
        try:
            data = base64.b64decode(body.content)
        except Exception:
            _error_response("INVALID_ENCODING", "Invalid base64 content")
            return {}  # unreachable
    else:
        data = body.content.encode(body.encoding)

    await state.store_artifact(
        key=body.key,
        data=data,
        run_id=body.run_id,
        experiment_id=body.experiment_id,
        stage=body.stage,
        content_type=body.content_type,
    )
    result: dict[str, Any] = {"key": body.key, "size_bytes": len(data)}
    url = await state.artifact_store.get_url(body.key)
    if url:
        result["url"] = url
    return _success_response(result)


# ── Compute endpoints (v1) ─────────────────────────────────


class ComputeSubmitRequest(BaseModel):
    stage: str = Field(..., min_length=1, max_length=200)
    image: str = Field(..., min_length=1, max_length=500)
    command: list[str] = Field(..., min_length=1, max_length=50)
    env: dict[str, str] = Field(default_factory=dict)
    gpu_type: str | None = None
    gpu_count: int = Field(default=0, ge=0, le=16)
    timeout_seconds: int = Field(default=3600, gt=0, le=86400)


class UpdateRoutesRequest(BaseModel):
    routes: list[dict[str, Any]] = Field(..., min_length=1, max_length=100)


@app.get("/v1/compute/backends")
async def get_compute_backends() -> dict[str, Any]:
    """List compute backends and their status."""
    state = _get_state()
    return _success_response(await state.compute_status())


@app.get("/v1/compute/jobs")
async def get_compute_jobs() -> dict[str, Any]:
    """List active compute jobs."""
    state = _get_state()
    if state.job_manager is None:
        return _success_response({"active_jobs": [], "cached_results": 0})
    return _success_response(state.job_manager.status())


@app.get("/v1/compute/jobs/{job_id}")
async def get_compute_job(job_id: str) -> dict[str, Any]:
    """Get details for a specific compute job."""
    state = _get_state()
    if state.job_manager is None:
        _error_response("NO_COMPUTE", "Compute not configured", status_code=404)
        return {}  # unreachable, satisfies type checker

    # Check active jobs
    for handle in state.job_manager.list_active():
        if handle.job_id == job_id:
            return _success_response(
                {
                    "job_id": handle.job_id,
                    "backend": handle.backend_name,
                    "status": handle.status.value,
                }
            )

    # Check cached results
    result = state.job_manager.get_result(job_id)
    if result:
        return _success_response(
            {
                "job_id": result.job_id,
                "status": result.status.value,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "cost_usd": result.cost_usd,
                "metrics": result.metrics,
            }
        )

    raise HTTPException(
        status_code=404,
        detail={"error": {"code": "NOT_FOUND", "message": f"Job '{job_id}' not found"}},
    )


@app.get("/v1/compute/routes")
async def get_compute_routes() -> dict[str, Any]:
    """Current stage-to-backend routing table."""
    state = _get_state()
    if state.compute_router is None:
        return _success_response({"backends": [], "routes": []})
    return _success_response(state.compute_router.to_dict())


@app.post("/v1/compute/jobs/{job_id}/cancel")
async def cancel_compute_job(job_id: str) -> dict[str, Any]:
    """Cancel a running compute job."""
    state = _get_state()
    if state.job_manager is None:
        _error_response("NO_COMPUTE", "Compute not configured", status_code=404)
        return {}
    cancelled = await state.job_manager.cancel_job(job_id)
    if not cancelled:
        _error_response("NOT_FOUND", f"Active job '{job_id}' not found", status_code=404)
    return _success_response({"job_id": job_id, "status": "cancelled"})


@app.put("/v1/compute/routes")
async def update_compute_routes(body: UpdateRoutesRequest) -> dict[str, Any]:
    """Update the compute routing table."""
    from ganglion.compute.router import ComputeRoute

    state = _get_state()
    if state.compute_router is None:
        _error_response("NO_COMPUTE", "Compute not configured", status_code=404)
        return {}

    routes = []
    for r in body.routes:
        pattern = r.get("pattern")
        backend = r.get("backend")
        if not pattern or not backend:
            _error_response("INVALID_ROUTE", "Each route needs 'pattern' and 'backend'")
            continue
        routes.append(
            ComputeRoute(
                pattern=str(pattern),
                backend=str(backend),
                overrides=r.get("overrides", {}),
            )
        )
    state.compute_router.set_routes(routes)
    return _success_response(state.compute_router.to_dict())


@app.delete("/v1/compute/backends/{name}")
async def remove_compute_backend(name: str) -> dict[str, Any]:
    """Remove a compute backend."""
    state = _get_state()
    result = await state.remove_backend(name)
    if not result.success:
        _error_response("BACKEND_ERROR", "; ".join(result.errors))
    return _success_response(await state.compute_status())


# ── Backward compatibility (unversioned routes) ───────────
# These mirror v1 routes for backward compatibility during migration.


@app.get("/status")
async def get_status_compat() -> dict[str, Any]:
    """Full framework state snapshot (deprecated: use /v1/status)."""
    return await get_status()  # type: ignore[no-any-return]


@app.get("/pipeline")
async def get_pipeline_compat() -> dict[str, Any]:
    """Current pipeline definition (deprecated: use /v1/pipeline)."""
    return await get_pipeline()  # type: ignore[no-any-return]


@app.get("/tools")
async def get_tools_compat(category: str | None = None) -> dict[str, Any]:
    """Registered tools (deprecated: use /v1/tools)."""
    return await get_tools(category)  # type: ignore[no-any-return]


@app.get("/agents")
async def get_agents_compat() -> dict[str, Any]:
    """Registered agents (deprecated: use /v1/agents)."""
    return await get_agents()  # type: ignore[no-any-return]


@app.get("/knowledge")
async def get_knowledge_compat(
    capability: str | None = None,
    max_entries: int = 20,
) -> dict[str, Any]:
    """Knowledge store (deprecated: use /v1/knowledge)."""
    return await get_knowledge(capability, max_entries)  # type: ignore[no-any-return]


@app.post("/tools")
async def write_tool_compat(body: WriteToolRequest) -> dict[str, Any]:
    """Write tool (deprecated: use /v1/tools)."""
    return await write_tool(body)  # type: ignore[no-any-return]


@app.post("/agents")
async def write_agent_compat(body: WriteAgentRequest) -> dict[str, Any]:
    """Write agent (deprecated: use /v1/agents)."""
    return await write_agent(body)  # type: ignore[no-any-return]


@app.patch("/pipeline")
async def patch_pipeline_compat(body: PatchPipelineRequest) -> dict[str, Any]:
    """Patch pipeline (deprecated: use /v1/pipeline)."""
    return await patch_pipeline(body)  # type: ignore[no-any-return]


@app.post("/run/pipeline")
async def run_pipeline_compat(body: RunPipelineRequest | None = None) -> dict[str, Any]:
    """Run pipeline (deprecated: use /v1/run/pipeline)."""
    return await run_pipeline(body)  # type: ignore[no-any-return]


@app.post("/rollback/last")
async def rollback_last_compat() -> dict[str, Any]:
    """Rollback (deprecated: use /v1/rollback/last)."""
    return await rollback_last()  # type: ignore[no-any-return]
