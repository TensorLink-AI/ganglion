"""CLI entry point for Ganglion.

Usage:
    # Scaffold a new subnet project
    ganglion init ./my-subnet --subnet sn9 --netuid 9

    # Start the HTTP bridge server for OpenClaw integration
    ganglion serve ./my-subnet --bot-id alpha --port 8899

    # Or via python -m
    python -m ganglion init ./my-subnet --subnet sn9 --netuid 9
    python -m ganglion serve ./my-subnet --bot-id alpha
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ganglion",
        description="Domain-specific execution engine for Bittensor subnet mining",
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ───────────────────────────────────────────────
    init_parser = sub.add_parser(
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
    serve_parser = sub.add_parser(
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

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "init":
        _run_init(args)
    elif args.command == "serve":
        _run_serve(args)


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
    print(f"  4. Copy {target}/skill/SKILL.md to your OpenClaw skills directory")
    print()


def _run_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from ganglion.bridge.server import app, configure
    from ganglion.state.framework_state import FrameworkState

    state = FrameworkState.load(args.project_dir, bot_id=args.bot_id)
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


if __name__ == "__main__":
    main()
