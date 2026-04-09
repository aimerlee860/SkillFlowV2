"""serve 子命令 CLI。"""

from __future__ import annotations


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("serve", help="启动 Web UI 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.set_defaults(func=_run)


def _run(args) -> None:
    import uvicorn

    from .app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
