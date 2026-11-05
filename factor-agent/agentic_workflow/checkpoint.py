"""SQLite checkpoint lifecycle helpers."""
# 保存和恢复LangGraph的运行进度
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def open_sqlite_checkpointer(
    database_path: str | Path,
) -> Iterator[SqliteSaver]:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("agentic_workflow.state", "ManagerAction"),
            ("agentic_workflow.state", "WorkflowStage"),
        ]
    )
    try:
        yield SqliteSaver(connection, serde=serializer)
    finally:
        connection.close()


def checkpoint_config(thread_id: str) -> dict[str, dict[str, str]]:
    if not thread_id.strip():
        raise ValueError("thread_id cannot be empty")
    return {"configurable": {"thread_id": thread_id}}
