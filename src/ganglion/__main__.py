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
import sys


def main(argv: list[str] | None = None) -> None:
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
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8899,
        help="Port to bind the server to (default: 8899)",
    )

    # ── Local-mode commands (no server needed) ─────────────

    _project_arg = {"help": "Path to the subnet project directory"}

    status_parser = subparsers.add_parser("status", help="Show framework state")
    status_parser.add_argument("project_dir", **_project_arg)
    status_parser.add_argument("--bot-id", default=None)

    tools_parser = subparsers.add_parser("tools", help="List registered tools")
    tools_parser.add_argument("project_dir", **_project_arg)
    tools_parser.add_argument("--category", default=None)

    agents_parser = subparsers.add_parser("agents", help="List registered agents")
    agents_parser.add_argument("project_dir", **_project_arg)

    knowledge_parser = subparsers.add_parser("knowledge", help="Show knowledge store")
    knowledge_parser.add_argument("project_dir", **_project_arg)
    knowledge_parser.add_argument("--bot-id", default=None)
    knowledge_parser.add_argument("--capability", default=None)
    knowledge_parser.add_argument("--max-entries", type=int, default=20)

    pipeline_parser = subparsers.add_parser("pipeline", help="Show pipeline definition")
    pipeline_parser.add_argument("project_dir", **_project_arg)

    run_parser = subparsers.add_parser("run", help="Run pipeline or a single stage")
    run_parser.add_argument("project_dir", **_project_arg)
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


def _load_state(project_dir: str, bot_id: str | None = None):
    from ganglion.state.framework_state import FrameworkState

    return FrameworkState.load(project_dir, bot_id=bot_id)


def _print_json(data):
    print(json.dumps(data, indent=2, default=str))


def _async_run(coro):
    return asyncio.run(coro)


# ── init ───────────────────────────────────────────────────


def _run_init(args: argparse.Namespace) -> None:
    from dataclasses import replace
    from pathlib import Path

    from ganglion.templates import get_template

    template = get_template(args.subnet)

    # Apply CLI overrides
    if args.netuid != 0:
        template = replace(template, netuid=args.netuid)
    if args.subnet != "generic":
        template = replace(template, name=args.subnet, slug=args.subnet.lower().replace(" ", "-"))

    target = Path(args.target_dir)
    if (target / "config.py").exists():
        print(f"Error: {target}/config.py already exists. Refusing to overwrite.")
        sys.exit(1)

    created = template.scaffold(target)

    print(f"Scaffolded project at {target.resolve()}")
    print()
    for path in created:
        print(f"  {path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {target}/config.py with your subnet details")
    print(f"  2. Replace the starter tool in {target}/tools/run_experiment.py")
    print(f"  3. Start the bridge:  ganglion serve {target} --bot-id my-bot")
    print(f"  4. Or use local mode: ganglion status {target}")
    print(f"  5. Copy {target}/skill/SKILL.md to your OpenClaw skills directory")
    print()


# ── serve ──────────────────────────────────────────────────


def _run_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from ganglion.bridge.server import app, configure

    state = _load_state(args.project_dir, bot_id=args.bot_id)
    configure(state)

    print(f"Ganglion bridge starting on {args.host}:{args.port}")
    print(f"  Project:  {state.project_root.resolve()}")
    print(f"  Pipeline: {state.pipeline_def.name}")
    print(f"  Tools:    {len(state.tool_registry.list_all())}")
    print(f"  Agents:   {len(state.agent_registry.list_all())}")
    if args.bot_id:
        print(f"  Bot ID:   {args.bot_id}")
    print()
    print("OpenClaw can connect at:")
    print(f"  http://{args.host}:{args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


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

    async def _gather():
        return {
            "patterns": [
                p.__dict__
                for p in await state.knowledge.backend.query_patterns(query)
            ],
            "antipatterns": [
                a.__dict__
                for a in await state.knowledge.backend.query_antipatterns(query)
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
        overrides = json.loads(args.overrides)

    if args.stage:
        result = _async_run(state.run_single_stage(args.stage, overrides))
    else:
        result = _async_run(state.run_pipeline(overrides=overrides))

    _print_json(result.to_dict())


if __name__ == "__main__":
    main()
