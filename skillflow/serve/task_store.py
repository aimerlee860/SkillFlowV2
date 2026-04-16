"""任务持久化层：SQLite 存储。"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

DB_PATH = Path.home() / ".skillflow" / "tasks.db"


@dataclass
class TaskRecord:
    """任务记录数据类。"""
    id: str
    task_type: str  # create | eval | evolve
    skill: Optional[str]
    status: str  # pending | queued | running | paused | completed | failed | cancelled
    params: dict  # 完整任务参数（JSON 可恢复）
    output_dir: Optional[str]  # 输出目录路径
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress_current: int = 0  # 当前进度值
    progress_total: int = 0  # 总进度值
    error: Optional[str] = None
    result: Optional[dict] = None  # 最终结果摘要

    def to_dict(self) -> dict:
        """转换为字典（用于 API 返回）。"""
        return asdict(self)


class TaskStore:
    """SQLite 任务存储。"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    skill TEXT,
                    status TEXT NOT NULL,
                    params TEXT NOT NULL,
                    output_dir TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    progress_current INTEGER DEFAULT 0,
                    progress_total INTEGER DEFAULT 0,
                    error TEXT,
                    result TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_skill ON tasks(skill);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);

                CREATE TABLE IF NOT EXISTS progress_events (
                    task_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (task_id, seq)
                );

                CREATE INDEX IF NOT EXISTS idx_events_task ON progress_events(task_id);
            """)

    def save(self, task: TaskRecord) -> None:
        """保存/更新任务。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, task_type, skill, status, params, output_dir,
                    created_at, started_at, finished_at,
                    progress_current, progress_total, error, result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.task_type,
                task.skill,
                task.status,
                json.dumps(task.params, ensure_ascii=False),
                task.output_dir,
                task.created_at,
                task.started_at,
                task.finished_at,
                task.progress_current,
                task.progress_total,
                task.error,
                json.dumps(task.result, ensure_ascii=False) if task.result else None,
            ))

    def load(self, task_id: str) -> Optional[TaskRecord]:
        """加载单个任务。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def list(
        self,
        status: Optional[str] = None,
        skill: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        """查询任务列表。"""
        sql = "SELECT * FROM tasks WHERE 1=1"
        args: list[Any] = []

        if status:
            sql += " AND status = ?"
            args.append(status)
        if skill:
            sql += " AND skill = ?"
            args.append(skill)
        if task_type:
            sql += " AND task_type = ?"
            args.append(task_type)

        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, args).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_running_for_skill(self, skill: str) -> Optional[TaskRecord]:
        """获取指定技能正在执行/排队的任务。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT * FROM tasks
                WHERE skill = ? AND status IN ('pending', 'queued', 'running', 'paused')
                ORDER BY created_at ASC LIMIT 1
            """, (skill,)).fetchone()
            if row:
                return self._row_to_record(row)
            return None

    def count_running(self) -> int:
        """统计当前运行任务数。"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]

    def update_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
        result: Optional[dict] = None,
    ) -> None:
        """更新任务状态。"""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET
                    status = ?,
                    started_at = COALESCE(started_at, ?),
                    finished_at = CASE
                        WHEN ? IN ('completed', 'failed', 'cancelled') THEN ?
                        ELSE finished_at
                    END,
                    error = ?,
                    result = ?
                WHERE id = ?
            """, (
                status,
                now if status == "running" else None,
                status,
                now if status in ("completed", "failed", "cancelled") else None,
                error,
                json.dumps(result, ensure_ascii=False) if result else None,
                task_id,
            ))

    def update_progress(self, task_id: str, current: int, total: int) -> None:
        """更新进度值。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET progress_current = ?, progress_total = ?
                WHERE id = ?
            """, (current, total, task_id))

    def append_event(self, task_id: str, event_type: str, event_data: dict) -> None:
        """追加进度事件。"""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM progress_events WHERE task_id = ?",
                (task_id,)
            ).fetchone()[0]
            conn.execute("""
                INSERT INTO progress_events (task_id, seq, event_type, event_data, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (task_id, seq, event_type, json.dumps(event_data, ensure_ascii=False), now))

    def get_events(self, task_id: str, after_seq: int = -1) -> list[dict]:
        """获取进度事件（用于 SSE 流恢复）。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT seq, event_type, event_data, created_at
                FROM progress_events
                WHERE task_id = ? AND seq > ?
                ORDER BY seq ASC
            """, (task_id, after_seq)).fetchall()
            return [
                {
                    "seq": r[0],
                    "type": r[1],
                    "data": json.loads(r[2]),
                    "time": r[3],
                }
                for r in rows
            ]

    def clear_events(self, task_id: str) -> None:
        """清除任务的进度事件。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM progress_events WHERE task_id = ?", (task_id,))

    def delete(self, task_id: str) -> bool:
        """删除任务及其进度事件。"""
        with sqlite3.connect(self.db_path) as conn:
            # 先检查任务是否存在
            row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                return False
            # 删除进度事件
            conn.execute("DELETE FROM progress_events WHERE task_id = ?", (task_id,))
            # 删除任务
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return True

    def _row_to_record(self, row: tuple) -> TaskRecord:
        """数据库行转 TaskRecord。"""
        return TaskRecord(
            id=row[0],
            task_type=row[1],
            skill=row[2],
            status=row[3],
            params=json.loads(row[4]) if row[4] else {},
            output_dir=row[5],
            created_at=row[6],
            started_at=row[7],
            finished_at=row[8],
            progress_current=row[9] or 0,
            progress_total=row[10] or 0,
            error=row[11],
            result=json.loads(row[12]) if row[12] else None,
        )