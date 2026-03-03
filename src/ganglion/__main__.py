"""CLI entry point for Ganglion.

Usage:
    # Start the HTTP bridge server for OpenClaw integration
    ganglion serve ./my-subnet --bot-id alpha --port 8899

    # Or via python -m
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

    if args.command == "serve":
        _run_serve(args)


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
