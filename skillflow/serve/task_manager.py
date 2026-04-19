"""任务调度器：队列管理 + 并发限制 + 技能冲突检测。"""

from __future__ import annotations

import shutil
import uuid
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

from .task_store import TaskStore, TaskRecord


class TaskManager:
    """任务调度器。

    功能：
    - 全局并发限制（最多 5 个 running 任务）
    - 技能冲突检测（同技能任务串行）
    - 队列管理（超限任务排队等待）
    - 状态持久化（SQLite）
    """

    MAX_CONCURRENT_TASKS = 5  # 全局并发上限

    def __init__(self, store: Optional[TaskStore] = None, max_workers: int = 8):
        self.store = store or TaskStore()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue: deque[str] = deque()  # task_id 队列
        self._running_ids: set[str] = set()  # 当前运行的任务 ID
        self._lock = Lock()

    def create_task(
        self,
        task_type: str,
        skill: Optional[str],
        params: dict,
        output_dir: Optional[str] = None,
    ) -> TaskRecord:
        """创建任务并持久化。"""
        task_id = uuid.uuid4().hex[:8]
        task = TaskRecord(
            id=task_id,
            task_type=task_type,
            skill=skill,
            status="pending",
            params=params,
            output_dir=output_dir,
            created_at=time.time(),
            progress_current=0,
            progress_total=0,
        )
        self.store.save(task)
        return task

    def submit(self, task_id: str, executor_func: Optional[Callable[[TaskRecord], dict]] = None) -> str:
        """提交任务执行。

        Args:
            task_id: 任务 ID
            executor_func: 可选的执行函数，如果为 None 则使用内部 TaskExecutor

        Returns:
            实际状态: 'running' | 'queued'
        """
        task = self.store.load(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        with self._lock:
            # 1. 检查全局并发限制
            if len(self._running_ids) >= self.MAX_CONCURRENT_TASKS:
                self._enqueue(task_id)
                return "queued"

            # 2. 检查技能冲突限制
            if task.skill:
                running = self.store.get_running_for_skill(task.skill)
                if running and running.id != task_id:
                    self._enqueue(task_id)
                    return "queued"

            # 3. 立即执行
            self._running_ids.add(task_id)
            self.store.update_status(task_id, "running")

        # 清除旧进度事件（新执行）
        self.store.clear_events(task_id)

        # 创建带进度回调的 executor
        def _emit_progress(tid: str, event_type: str, data: dict) -> None:
            self.emit_progress(tid, event_type, data)

        from .task_executor import TaskExecutor
        executor = TaskExecutor(emit_progress=_emit_progress, store=self.store)

        # 启动 worker 线程
        def _worker():
            try:
                if executor_func:
                    result = executor_func(task)
                else:
                    result = executor.execute(task)
                self.store.update_status(task_id, "completed", result=result)
            except Exception as e:
                self.store.update_status(task_id, "failed", error=str(e))
            finally:
                with self._lock:
                    self._running_ids.discard(task_id)
                self._try_next_queued()

        self._executor.submit(_worker)
        return "running"

    def _enqueue(self, task_id: str) -> None:
        """加入队列。"""
        self.store.update_status(task_id, "queued")
        self._queue.append(task_id)

    def _try_next_queued(self) -> None:
        """任务完成后尝试执行下一个队列任务。"""
        with self._lock:
            # 检查是否还有空位
            if len(self._running_ids) >= self.MAX_CONCURRENT_TASKS:
                return

            # 按顺序查找可执行的队列任务
            to_execute = None
            for queued_id in list(self._queue):
                queued_task = self.store.load(queued_id)

                # 跳过无效/非排队状态的任务
                if not queued_task or queued_task.status != "queued":
                    self._queue.remove(queued_id)
                    continue

                # 检查技能冲突
                if queued_task.skill:
                    running = self.store.get_running_for_skill(queued_task.skill)
                    if running and running.id != queued_id:
                        continue  # 同技能还有运行任务，跳过

                # 可以执行
                to_execute = queued_id
                self._queue.remove(queued_id)
                break

            if not to_execute:
                return

            # 标记为运行
            self._running_ids.add(to_execute)
            self.store.update_status(to_execute, "running")
            self.store.clear_events(to_execute)

        # 在锁外提交执行（避免死锁）
        task = self.store.load(to_execute)

        # 创建带进度回调的 executor
        def _emit_progress(tid: str, event_type: str, data: dict) -> None:
            self.emit_progress(tid, event_type, data)

        def _worker():
            try:
                from .task_executor import TaskExecutor
                executor = TaskExecutor(emit_progress=_emit_progress, store=self.store)
                result = executor.execute(task, recovery=True)
                self.store.update_status(to_execute, "completed", result=result)
            except Exception as e:
                self.store.update_status(to_execute, "failed", error=str(e))
            finally:
                with self._lock:
                    self._running_ids.discard(to_execute)
                self._try_next_queued()

        self._executor.submit(_worker)

    def pause(self, task_id: str) -> bool:
        """暂停任务（仅标记状态，实际暂停需 executor 支持）。"""
        task = self.store.load(task_id)
        if not task or task.status != "running":
            return False
        with self._lock:
            self._running_ids.discard(task_id)
        self.store.update_status(task_id, "paused")
        return True

    def resume(self, task_id: str) -> str:
        """恢复暂停的任务。"""
        task = self.store.load(task_id)
        if not task or task.status != "paused":
            raise ValueError(f"Task not paused: {task_id}")
        # 使用内部 TaskExecutor 恢复执行
        return self.submit(task_id)

    def cancel(self, task_id: str) -> bool:
        """取消任务。"""
        task = self.store.load(task_id)
        if not task:
            return False

        if task.status in ("queued", "pending", "paused"):
            with self._lock:
                self._running_ids.discard(task_id)
                if task_id in self._queue:
                    self._queue.remove(task_id)
            self.store.update_status(task_id, "cancelled")
            return True

        # running 状态暂时不支持取消（需要 executor 支持中断）
        return False

    def retry(self, task_id: str) -> str:
        """重试失败或中断的任务。"""
        task = self.store.load(task_id)
        if not task or task.status not in ("failed", "interrupted"):
            raise ValueError(f"Task not failed/interrupted: {task_id}")
        self.store.update_status(task_id, "pending")
        return self.submit(task_id)

    def delete(self, task_id: str) -> bool:
        """删除任务及其关联的结果目录。"""
        task = self.store.load(task_id)
        if not task:
            return False
        # 只允许删除已完成/失败/取消/中断的任务
        if task.status not in ("completed", "failed", "cancelled", "interrupted"):
            return False

        # 删除关联的结果目录
        if task.output_dir:
            output_path = Path(task.output_dir)
            # 安全检查：确保路径在 results/ 下，防止误删其他目录
            cwd = Path.cwd()
            results_dir = cwd / "results"
            try:
                resolved_base = output_path.resolve()
                if str(resolved_base).startswith(str(results_dir.resolve())):
                    if task.task_type == "evolve" and task.result and task.result.get("run_id"):
                        run_id = task.result["run_id"]
                        # 新格式: output_dir 已含时间戳; 旧格式: output_dir 是父目录
                        actual_path = resolved_base if resolved_base.name == run_id else resolved_base / run_id
                    else:
                        actual_path = resolved_base

                    if actual_path.exists() and actual_path.is_dir():
                        shutil.rmtree(actual_path)
            except Exception:
                pass  # 目录删除失败不影响任务删除

        return self.store.delete(task_id)

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """获取单个任务。"""
        return self.store.load(task_id)

    def list_tasks(
        self,
        status: Optional[str] = None,
        skill: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[TaskRecord]:
        """查询任务列表。"""
        return self.store.list(status=status, skill=skill, task_type=task_type, limit=limit)

    def is_skill_in_use(self, skill: str) -> bool:
        """检查技能是否正在被使用。"""
        running = self.store.get_running_for_skill(skill)
        return running is not None and running.status in ("running", "queued")

    def get_concurrent_status(self) -> dict:
        """获取并发状态。"""
        with self._lock:
            return {
                "running_count": len(self._running_ids),
                "running_ids": list(self._running_ids),
                "max_concurrent": self.MAX_CONCURRENT_TASKS,
                "queue_length": len(self._queue),
                "queued_ids": list(self._queue),
                "available_slots": self.MAX_CONCURRENT_TASKS - len(self._running_ids),
            }

    def emit_progress(self, task_id: str, event_type: str, event_data: dict) -> None:
        """发送进度事件。"""
        self.store.append_event(task_id, event_type, event_data)

    def get_progress_events(self, task_id: str, after_seq: int = -1) -> list[dict]:
        """获取进度事件。"""
        return self.store.get_events(task_id, after_seq)