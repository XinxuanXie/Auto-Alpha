"""Workflow Tool that persists experiment records to SQLite."""

import sqlite3

from agentic_workflow.state import WorkflowState
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.base import ToolResult


class SQLiteRecordTool:
    name = "sqlite_record"

    def __init__(self, repository: SQLiteResearchRepository) -> None:
        self.repository = repository

    def run(self, state: WorkflowState) -> ToolResult:
        try: #调用save方法将state存储到sqlite数据库中
            record = self.repository.save(state)
        except (OSError, ValueError, TypeError, sqlite3.Error) as error:
            return ToolResult.fail(f"Failed to persist research record: {error}")
        return ToolResult.ok({"record": record})
