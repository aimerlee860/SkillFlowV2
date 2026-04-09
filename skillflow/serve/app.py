"""FastAPI 应用工厂。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.create import router as create_router
from .api.eval import router as eval_router
from .api.evolve import router as evolve_router
from .api.filesystem import router as fs_router
from .tasks import TaskManager

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="SkillFlow", docs_url=None, redoc_url=None)

    # CORS — 本地开发工具，允许所有来源
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局 TaskManager 实例
    app.state.task_manager = TaskManager()

    # API 路由
    app.include_router(fs_router, prefix="/api")
    app.include_router(create_router, prefix="/api")
    app.include_router(eval_router, prefix="/api")
    app.include_router(evolve_router, prefix="/api")

    # 静态文件（index.html）
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app
