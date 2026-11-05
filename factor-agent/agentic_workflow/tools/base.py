"""Common interface for workflow tools."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_workflow.state import WorkflowState


@dataclass(frozen=True)
class ToolResult:
    success: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, payload: dict[str, Any]) -> "ToolResult":
        return cls(success=True, payload=payload)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)


class WorkflowTool(Protocol):
    name: str

    def run(
        self, 
        state: WorkflowState
        ) -> ToolResult:
        """Execute one bounded operation without changing workflow routing."""

