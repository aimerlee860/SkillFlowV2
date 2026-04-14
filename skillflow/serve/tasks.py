"""后台任务管理：线程池执行 + SSE 进度追踪。"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    task_type: str  # "create" | "eval" | "evolve"
    skill: str | None = None  # 正在操作的技能名
    status: TaskStatus = TaskStatus.PENDING
    progress_events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "skill": self.skill,
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at,
        }


class TaskManager:
    """管理后台任务的生命周期和进度追踪。"""

    def __init__(self, max_workers: int = 8):
        self._tasks: dict[str, Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_task(self, task_type: str, skill: str | None = None) -> Task:
        task_id = uuid.uuid4().hex[:8]
        task = Task(id=task_id, task_type=task_type, skill=skill)
        self._tasks[task_id] = task
        return task

    def is_skill_in_use(self, skill_name: str) -> bool:
        """检查技能是否正在被任务使用。"""
        for task in self._tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                if task.skill == skill_name:
                    return True
        return False

    def submit(
        self,
        task: Task,
        func: Callable,
        *,
        progress_file: Path | None = None,
        **kwargs,
    ) -> None:
        """提交同步函数到后台线程执行，可选轮询进度文件。"""

        def _worker():
            task.status = TaskStatus.RUNNING
            try:
                result = func(**kwargs)
                task.result = result if isinstance(result, dict) else {"path": str(result)}
                task.status = TaskStatus.COMPLETED
                task.progress_events.append(
                    {"type": "done", "status": "completed", "result": task.result}
                )
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.progress_events.append(
                    {"type": "done", "status": "failed", "error": str(e)}
                )

        task.status = TaskStatus.RUNNING
        self._executor.submit(_worker)

        # 如果有进度文件，启动轮询
        if progress_file is not None:
            self._poll_progress(task, progress_file)

    def submit_with_watcher(
        self,
        task: Task,
        func: Callable,
        progress_file: Path | None = None,
        **kwargs,
    ) -> None:
        """提交后台任务，同时启动进度文件轮询。"""

        def _worker():
            task.status = TaskStatus.RUNNING
            try:
                result = func(**kwargs)
                task.result = result if isinstance(result, dict) else {"path": str(result)}
                task.status = TaskStatus.COMPLETED
                task.progress_events.append(
                    {"type": "done", "status": "completed", "result": task.result}
                )
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.progress_events.append(
                    {"type": "done", "status": "failed", "error": str(e)}
                )

        self._executor.submit(_worker)

        if progress_file:
            self._start_file_watcher(task, progress_file)

    def _start_file_watcher(self, task: Task, progress_file: Path) -> None:
        """在后台线程轮询进度文件，将新内容注入 task.progress_events。"""

        def _watch():
            seen_lines = 0
            while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                try:
                    if progress_file.exists():
                        lines = progress_file.read_text(encoding="utf-8").strip().split("\n")
                        new_lines = lines[seen_lines:]
                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                task.progress_events.append(
                                    {"type": "progress", "data": record}
                                )
                            except json.JSONDecodeError:
                                pass
                        seen_lines = len(lines)
                except Exception:
                    pass
                time.sleep(1)

        self._executor.submit(_watch)

    def emit_progress(self, task_id: str, event: dict) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.progress_events.append(event)

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        return [t.to_dict() for t in self._tasks.values()]
