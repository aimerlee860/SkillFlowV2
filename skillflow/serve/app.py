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
from .api.tasks import router as tasks_router
from .task_manager import TaskManager
from .task_store import TaskStore

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
    store = TaskStore()
    task_manager = TaskManager(store=store)
    app.state.task_manager = task_manager
    app.state.task_store = store

    # API 路由
    app.include_router(fs_router, prefix="/api")
    app.include_router(tasks_router, prefix="/api")
    app.include_router(create_router, prefix="/api")
    app.include_router(eval_router, prefix="/api")
    app.include_router(evolve_router, prefix="/api")

    # 静态文件（index.html）
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    # 启动时处理中断的任务（不自动恢复）
    @app.on_event("startup")
    async def handle_interrupted_tasks():
        """处理服务停止时的未完成任务，标记为中断状态。"""
        # 将所有 running/queued/paused 状态改为 interrupted
        for status in ("running", "queued", "paused"):
            tasks = store.list(status=status)
            for task in tasks:
                store.update_status(
                    task.id,
                    "interrupted",
                    error="服务重启导致任务中断，请点击重试继续执行",
                )

    return app