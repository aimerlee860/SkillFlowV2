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

    # 启动时恢复中断的任务
    @app.on_event("startup")
    async def recover_tasks():
        """恢复服务停止时的运行任务。"""
        # 将所有 running 状态改为 paused
        running_tasks = store.list(status="running")
        for task in running_tasks:
            store.update_status(task.id, "paused")

        # 统计 paused 任务
        paused_tasks = store.list(status="paused")

        if not paused_tasks:
            return

        # 按创建时间排序，恢复最早的任务
        paused_tasks.sort(key=lambda t: t.created_at)

        # 只恢复不超过并发限制的任务
        for task in paused_tasks[:task_manager.MAX_CONCURRENT_TASKS]:
            # 检查技能冲突
            if task.skill:
                running_for_skill = store.get_running_for_skill(task.skill)
                if running_for_skill and running_for_skill.id != task.id:
                    continue  # 同技能已有任务恢复，跳过

            # 恢复执行
            task_manager.submit(task.id)

    return app