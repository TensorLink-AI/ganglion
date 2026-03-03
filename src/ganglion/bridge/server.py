"""FastAPI HTTP bridge exposing the framework to OpenClaw and external tools."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ganglion.state.framework_state import FrameworkState
from ganglion.knowledge.types import KnowledgeQuery

app = FastAPI(title="Ganglion Bridge", version="0.1.0")

# State is set at startup via configure()
_state: FrameworkState | None = None


def configure(state: FrameworkState) -> None:
    """Configure the bridge with a FrameworkState instance."""
    global _state
    _state = state


def _get_state() -> FrameworkState:
    if _state is None:
        raise HTTPException(500, "Bridge not configured. Call configure() first.")
    return _state


# ── Request models ──────────────────────────────────────────


class WriteToolRequest(BaseModel):
    name: str
    code: str
    category: str = "general"
    test_code: str | None = None


class WriteAgentRequest(BaseModel):
    name: str
    code: str
    test_task: dict | None = None


class PatchPipelineRequest(BaseModel):
    operations: list[dict]


class SwapPolicyRequest(BaseModel):
    retry_policy: dict


class RunPipelineRequest(BaseModel):
    overrides: dict | None = None


class RunStageRequest(BaseModel):
    context: dict | None = None


class RunExperimentRequest(BaseModel):
    config: dict


# ── Observation endpoints ───────────────────────────────────


@app.get("/status")
async def get_status():
    """Full framework state snapshot."""
    return _get_state().describe()


@app.get("/pipeline")
async def get_pipeline():
    """Current pipeline definition."""
    return _get_state().pipeline_def.to_dict()


@app.get("/tools")
async def get_tools(category: str | None = None):
    """Registered tools."""
    return _get_state().tool_registry.list_all(category=category)


@app.get("/agents")
async def get_agents():
    """Registered agents."""
    return _get_state().agent_registry.list_all()


@app.get("/runs")
async def get_run_history(n: int = 10):
    """Past pipeline runs."""
    state = _get_state()
    if state.persistence is None:
        return []
    return await state.persistence.load_run_history(n=n)


@app.get("/metrics")
async def get_metrics(experiment_id: str | None = None):
    """Experiment metrics."""
    state = _get_state()
    if state.persistence is None:
        return []
    return await state.persistence.query_metrics(experiment_id=experiment_id)


@app.get("/knowledge")
async def get_knowledge(capability: str | None = None, max_entries: int = 20):
    """Knowledge store contents."""
    state = _get_state()
    if not state.knowledge:
        return {"patterns": [], "antipatterns": [], "summary": None}
    query = KnowledgeQuery(capability=capability, max_entries=max_entries)
    return {
        "patterns": [
            p.__dict__
            for p in state.knowledge.backend.query_patterns(query)
        ],
        "antipatterns": [
            a.__dict__
            for a in state.knowledge.backend.query_antipatterns(query)
        ],
        "summary": state.knowledge.summary(),
    }


@app.get("/source/{path:path}")
async def get_source(path: str):
    """Read source code of any file in the project."""
    state = _get_state()
    full_path = state.project_root / path
    if not full_path.exists():
        raise HTTPException(404, f"Not found: {path}")
    return {"path": path, "content": full_path.read_text()}


# ── Mutation endpoints ──────────────────────────────────────


@app.post("/tools")
async def write_tool(body: WriteToolRequest):
    """Write and register a new tool."""
    state = _get_state()
    result = await state.write_and_register_tool(
        body.name, body.code, body.category, body.test_code
    )
    if not result.success:
        raise HTTPException(400, {"errors": result.errors})
    return {"success": True, "path": result.path}


@app.post("/agents")
async def write_agent(body: WriteAgentRequest):
    """Write and register a new agent."""
    state = _get_state()
    result = await state.write_and_register_agent(
        body.name, body.code, body.test_task
    )
    if not result.success:
        raise HTTPException(400, {"errors": result.errors})
    return {"success": True, "path": result.path}


@app.patch("/pipeline")
async def patch_pipeline(body: PatchPipelineRequest):
    """Apply pipeline modifications."""
    state = _get_state()
    result = await state.apply_pipeline_patch(body.operations)
    if not result.success:
        raise HTTPException(400, {"errors": result.errors})
    return {"success": True, "pipeline": result.pipeline}


@app.put("/policies/{stage_name}")
async def swap_policy(stage_name: str, body: SwapPolicyRequest):
    """Swap retry policy for a stage."""
    state = _get_state()
    result = await state.swap_policy(
        stage_name if stage_name != "default" else None,
        body.retry_policy,
    )
    if not result.success:
        raise HTTPException(400, {"errors": result.errors})
    return {"success": True}


# ── Execution endpoints ─────────────────────────────────────


@app.post("/run/pipeline")
async def run_pipeline(body: RunPipelineRequest | None = None):
    """Execute full pipeline."""
    state = _get_state()
    result = await state.run_pipeline(
        overrides=body.overrides if body else None
    )
    return result.to_dict()


@app.post("/run/stage/{stage_name}")
async def run_stage(stage_name: str, body: RunStageRequest | None = None):
    """Execute a single pipeline stage."""
    state = _get_state()
    result = await state.run_single_stage(
        stage_name, body.context if body else None
    )
    return result.to_dict()


# ── Rollback endpoints ──────────────────────────────────────


@app.post("/rollback/last")
async def rollback_last():
    """Undo the most recent mutation."""
    state = _get_state()
    result = await state.rollback_last()
    if not result.success:
        raise HTTPException(400, {"errors": result.errors})
    return {"success": True}


@app.post("/rollback/{index}")
async def rollback_to(index: int):
    """Undo all mutations back to the given index."""
    state = _get_state()
    result = await state.rollback_to(index)
    if not result.success:
        raise HTTPException(400, {"errors": result.errors})
    return {"success": True}
