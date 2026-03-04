"""CLI entry point for Ganglion.

Usage:
    # Scaffold a new subnet project
    ganglion init ./my-subnet --subnet sn9 --netuid 9

    # Start the HTTP bridge server (remote mode)
    ganglion serve ./my-subnet --bot-id alpha --port 8899

    # Local mode — no server needed (same machine as OpenClaw)
    ganglion status ./my-subnet
    ganglion tools ./my-subnet
    ganglion agents ./my-subnet
    ganglion knowledge ./my-subnet
    ganglion run ./my-subnet
    ganglion run ./my-subnet --stage plan
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import types
from typing import Any

logger = logging.getLogger("ganglion")


def _setup_logging(level: str = "INFO") -> None:
    """Configure structured logging with appropriate level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _setup_signal_handlers() -> None:
    """Install graceful shutdown handlers for SIGTERM and SIGINT."""

    def _handle_shutdown(signum: int, frame: types.FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down gracefully", sig_name)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)


def main(argv: list[str] | None = None) -> None:
    _setup_signal_handlers()

    parser = argparse.ArgumentParser(
        prog="ganglion",
        description="Domain-specific execution engine for Bittensor subnet mining",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── init ───────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold a new subnet project directory",
    )
    init_parser.add_argument(
        "target_dir",
        help="Directory to create (will be created if it doesn't exist)",
    )
    init_parser.add_argument(
        "--subnet",
        default="generic",
        help="Subnet name or built-in template (default: generic)",
    )
    init_parser.add_argument(
        "--netuid",
        type=int,
        default=0,
        help="Subnet netuid (default: 0)",
    )

    # ── serve ──────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the HTTP bridge server for OpenClaw integration",
    )
    serve_parser.add_argument(
        "project_dir",
        help="Path to the subnet project directory (must contain config.py)",
    )
    serve_parser.add_argument(
        "--bot-id",
        default=None,
        help="Bot identifier for multi-bot shared knowledge",
    )
    serve_parser.add_argument(
        "--host",
        default=None,
        help="Host to bind the server to (default: from config or 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the server to (default: from config or 8899)",
    )

    # ── Local-mode commands (no server needed) ─────────────

    _project_help = "Path to the subnet project directory"

    status_parser = subparsers.add_parser("status", help="Show framework state")
    status_parser.add_argument("project_dir", help=_project_help)
    status_parser.add_argument("--bot-id", default=None)

    tools_parser = subparsers.add_parser("tools", help="List registered tools")
    tools_parser.add_argument("project_dir", help=_project_help)
    tools_parser.add_argument("--category", default=None)

    agents_parser = subparsers.add_parser("agents", help="List registered agents")
    agents_parser.add_argument("project_dir", help=_project_help)

    knowledge_parser = subparsers.add_parser("knowledge", help="Show knowledge store")
    knowledge_parser.add_argument("project_dir", help=_project_help)
    knowledge_parser.add_argument("--bot-id", default=None)
    knowledge_parser.add_argument("--capability", default=None)
    knowledge_parser.add_argument("--max-entries", type=int, default=20)

    pipeline_parser = subparsers.add_parser("pipeline", help="Show pipeline definition")
    pipeline_parser.add_argument("project_dir", help=_project_help)

    run_parser = subparsers.add_parser("run", help="Run pipeline or a single stage")
    run_parser.add_argument("project_dir", help=_project_help)
    run_parser.add_argument("--bot-id", default=None)
    run_parser.add_argument("--stage", default=None, help="Run only this stage")
    run_parser.add_argument(
        "--overrides",
        default=None,
        help="JSON string of overrides for the pipeline run",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _setup_logging()

    commands = {
        "init": _run_init,
        "serve": _run_serve,
        "status": _run_status,
        "tools": _run_tools,
        "agents": _run_agents,
        "knowledge": _run_knowledge,
        "pipeline": _run_pipeline,
        "run": _run_run,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


# ── Helpers ────────────────────────────────────────────────


def _load_state(project_dir: str, bot_id: str | None = None) -> Any:
    from ganglion.state.framework_state import FrameworkState

    return FrameworkState.load(project_dir, bot_id=bot_id)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _async_run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── init ───────────────────────────────────────────────────


def _run_init(args: argparse.Namespace) -> None:
    from dataclasses import replace
    from pathlib import Path

    from ganglion.templates import get_template

    template = get_template(args.subnet)
    is_builtin = template.name != args.subnet or args.subnet == "generic"

    # Apply CLI overrides — only override name/slug for unregistered templates
    if args.netuid != 0:
        template = replace(template, netuid=args.netuid)
    if args.subnet != "generic" and not is_builtin:
        template = replace(template, name=args.subnet, slug=args.subnet.lower().replace(" ", "-"))

    target = Path(args.target_dir)
    if (target / "config.py").exists():
        logger.error("Refusing to overwrite existing config at %s/config.py", target)
        sys.exit(1)

    created = template.scaffold(target)

    logger.info("Scaffolded project at %s", target.resolve())
    for path in created:
        logger.info("  Created: %s", path)
    logger.info(
        "Next: edit %s/config.py, then run: ganglion serve %s --bot-id my-bot",
        target,
        target,
    )


# ── serve ──────────────────────────────────────────────────


def _run_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from ganglion.bridge.server import app, configure, setup_cors
    from ganglion.config import GanglionConfig

    config = GanglionConfig.from_env()
    config.validate_or_raise()

    state = _load_state(args.project_dir, bot_id=args.bot_id)
    configure(state, config)
    setup_cors(config.cors_allowed_origins)

    host = args.host or config.server_host
    port = args.port or config.server_port

    logger.info(
        "Ganglion bridge starting on %s:%d (project=%s, pipeline=%s, tools=%d, agents=%d)",
        host,
        port,
        state.project_root.resolve(),
        state.pipeline_def.name,
        len(state.tool_registry.list_all()),
        len(state.agent_registry.list_all()),
    )

    uvicorn.run(app, host=host, port=port)


# ── Local-mode commands ────────────────────────────────────


def _run_status(args: argparse.Namespace) -> None:
    state = _load_state(args.project_dir, bot_id=getattr(args, "bot_id", None))
    _print_json(_async_run(state.describe()))


def _run_tools(args: argparse.Namespace) -> None:
    state = _load_state(args.project_dir)
    tools = state.tool_registry.list_all(category=args.category)
    _print_json(tools)


def _run_agents(args: argparse.Namespace) -> None:
    state = _load_state(args.project_dir)
    _print_json(state.agent_registry.list_all())


def _run_knowledge(args: argparse.Namespace) -> None:
    from ganglion.knowledge.types import KnowledgeQuery

    state = _load_state(args.project_dir, bot_id=getattr(args, "bot_id", None))
    if not state.knowledge:
        _print_json({"patterns": [], "antipatterns": [], "summary": None})
        return

    query = KnowledgeQuery(capability=args.capability, max_entries=args.max_entries)

    async def _gather() -> dict[str, Any]:
        return {
            "patterns": [p.__dict__ for p in await state.knowledge.backend.query_patterns(query)],
            "antipatterns": [
                a.__dict__ for a in await state.knowledge.backend.query_antipatterns(query)
            ],
            "summary": await state.knowledge.summary(),
        }

    _print_json(_async_run(_gather()))


def _run_pipeline(args: argparse.Namespace) -> None:
    state = _load_state(args.project_dir)
    _print_json(state.pipeline_def.to_dict())


def _run_run(args: argparse.Namespace) -> None:
    state = _load_state(args.project_dir, bot_id=getattr(args, "bot_id", None))

    overrides = None
    if args.overrides:
        try:
            overrides = json.loads(args.overrides)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON for --overrides: %s", exc)
            sys.exit(1)

    if args.stage:
        result = _async_run(state.run_single_stage(args.stage, overrides))
    else:
        result = _async_run(state.run_pipeline(overrides=overrides))

    _print_json(result.to_dict())


if __name__ == "__main__":
    main()
