"""SQLite repository for immutable attempt-level research memory."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from agentic_workflow.hypothesis import hypothesis_signature
from agentic_workflow.state import WorkflowState


class SQLiteResearchRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_records (
                    record_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    hypothesis_id TEXT,
                    hypothesis_signature TEXT,
                    factor_id TEXT,
                    evaluation_status TEXT,
                    recommended_action TEXT,
                    next_action TEXT,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, iteration, attempt)
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(research_records)"
                ).fetchall()
            }
            required_columns = {
                "attempt",
                "hypothesis_signature",
                "factor_id",
                "recommended_action",
            }
            missing_columns = sorted(required_columns - columns)
            if missing_columns:
                raise RuntimeError(
                    "research.db uses an obsolete schema; delete it before "
                    f"starting the attempt-level memory: {', '.join(missing_columns)}"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_records_run "
                "ON research_records(run_id, iteration, attempt)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_research_records_hypothesis_signature "
                "ON research_records(hypothesis_signature)"
            )

    def save(self, state: WorkflowState) -> dict[str, Any]:
        record_id = (
            f"{state['run_id']}:{state['iteration']}:{state['attempt']}"
        )
        hypothesis = state["hypothesis"]
        factor_spec = state["factor_spec"]
        evaluation = state["evaluation"]
        hypothesis_id = (
            hypothesis.get("hypothesis_id")
            if isinstance(hypothesis, dict)
            else None
        )
        signature = (
            hypothesis_signature(hypothesis)
            if isinstance(hypothesis, dict)
            else None
        )
        factor_id = (
            factor_spec.get("factor_id")
            if isinstance(factor_spec, dict)
            else None
        )
        evaluation_status = (
            evaluation.get("status")
            if isinstance(evaluation, dict)
            else None
        )
        recommended_action = (
            evaluation.get("recommended_action")
            if isinstance(evaluation, dict)
            else None
        )
        next_action = (
            state["next_action"].value
            if state["next_action"] is not None
            else None
        )
        state_json = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_records (
                    record_id,
                    run_id,
                    iteration,
                    attempt,
                    stage,
                    hypothesis_id,
                    hypothesis_signature,
                    factor_id,
                    evaluation_status,
                    recommended_action,
                    next_action,
                    state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    stage = excluded.stage,
                    hypothesis_id = excluded.hypothesis_id,
                    hypothesis_signature = excluded.hypothesis_signature,
                    factor_id = excluded.factor_id,
                    evaluation_status = excluded.evaluation_status,
                    recommended_action = excluded.recommended_action,
                    next_action = excluded.next_action,
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record_id,
                    state["run_id"],
                    state["iteration"],
                    state["attempt"],
                    state["stage"].value,
                    hypothesis_id,
                    signature,
                    factor_id,
                    evaluation_status,
                    recommended_action,
                    next_action,
                    state_json,
                ),
            )

        return {
            "record_id": record_id,
            "run_id": state["run_id"],
            "iteration": state["iteration"],
            "attempt": state["attempt"],
            "hypothesis_id": hypothesis_id,
            "factor_id": factor_id,
            "hypothesis_signature": signature,
            "evaluation_status": evaluation_status,
            "database_path": str(self.database_path),
        }

    def fetch(self, record_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def fetch_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM research_records
                WHERE run_id = ?
                ORDER BY iteration ASC, attempt ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_experiments(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    record_id,
                    run_id,
                    iteration,
                    attempt,
                    evaluation_status,
                    recommended_action,
                    next_action,
                    state_json
                FROM research_records
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._summarize_row(row) for row in rows]

    def relevant_experiments(
        self,
        state: WorkflowState,
        limit: int = 10,
        candidate_limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if candidate_limit < limit:
            raise ValueError("candidate_limit must be at least limit")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    record_id,
                    run_id,
                    iteration,
                    attempt,
                    evaluation_status,
                    recommended_action,
                    next_action,
                    state_json
                FROM research_records
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (candidate_limit,),
            ).fetchall()

        candidates = [self._summarize_row(row) for row in rows]
        query = self._build_retrieval_query(state, candidates)
        scored = []
        for recency_index, candidate in enumerate(candidates):
            score, reasons = self._score_relevance(
                candidate,
                query,
                recency_index,
            )
            candidate["retrieval"] = {
                "score": round(score, 6),
                "match_reasons": reasons,
            }
            scored.append(candidate)
        scored.sort(
            key=lambda candidate: (
                -candidate["retrieval"]["score"],
                candidate["record_id"],
            )
        )
        return scored[:limit]

    @staticmethod
    def _build_retrieval_query(
        state: WorkflowState,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        hypothesis = state.get("hypothesis")
        evaluation = state.get("evaluation")
        if isinstance(hypothesis, dict):
            return {
                "run_id": state["run_id"],
                "hypothesis": hypothesis,
                "failure_profile": SQLiteResearchRepository._failure_profile(
                    evaluation if isinstance(evaluation, dict) else {}
                ),
            }
        for candidate in candidates:
            if candidate["run_id"] == state["run_id"]:
                return {
                    "run_id": state["run_id"],
                    "hypothesis": candidate["hypothesis"],
                    "failure_profile": candidate["failure_profile"],
                }
        return {
            "run_id": state["run_id"],
            "hypothesis": {},
            "failure_profile": [],
        }

    @staticmethod
    def _score_relevance(
        candidate: dict[str, Any],
        query: dict[str, Any],
        recency_index: int,
    ) -> tuple[float, list[str]]:
        score = 1.0 / (recency_index + 1)
        reasons = ["recency"]
        query_hypothesis = query.get("hypothesis") or {}
        candidate_hypothesis = candidate.get("hypothesis") or {}
        query_concepts = set(query_hypothesis.get("required_concepts") or [])
        candidate_concepts = set(
            candidate_hypothesis.get("required_concepts") or []
        )
        concept_overlap = sorted(query_concepts & candidate_concepts)
        if query_concepts and concept_overlap:
            score += 5.0 * len(concept_overlap) / len(query_concepts)
            reasons.append(f"concept_overlap:{','.join(concept_overlap)}")

        query_direction = query_hypothesis.get("prediction_direction")
        if query_direction and query_direction == candidate_hypothesis.get(
            "prediction_direction"
        ):
            score += 0.75
            reasons.append("same_prediction_direction")

        query_failures = set(query.get("failure_profile") or [])
        candidate_failures = set(candidate.get("failure_profile") or [])
        failure_overlap = sorted(query_failures & candidate_failures)
        if query_failures and failure_overlap:
            score += 2.0 * len(failure_overlap) / len(query_failures)
            reasons.append(f"failure_overlap:{','.join(failure_overlap)}")

        if candidate["run_id"] == query["run_id"]:
            score += 1.5
            reasons.append("same_run")
        if candidate.get("evaluation_status") == "PASS":
            score += 1.5
            reasons.append("historical_pass")

        rank_ic = candidate.get("rank_ic")
        if isinstance(rank_ic, (int, float)) and rank_ic > 0:
            score += min(rank_ic / 0.02, 1.0)
            reasons.append("positive_rank_ic")
        net_return = candidate.get("net_return")
        if isinstance(net_return, (int, float)) and net_return > 0:
            score += 0.25
            reasons.append("positive_net_return")
        return score, reasons

    @staticmethod
    def _failure_profile(evaluation: dict[str, Any]) -> list[str]:
        rule_checks = evaluation.get("rule_checks") or {}
        failed_checks = [
            name
            for name, check in rule_checks.items()
            if isinstance(check, dict) and check.get("passed") is False
        ]
        status = evaluation.get("status")
        if status and status != "PASS":
            failed_checks.append(f"status:{status}")
        return sorted(set(failed_checks))

    @staticmethod
    def _summarize_row(row: sqlite3.Row) -> dict[str, Any]:
        state = json.loads(row["state_json"])
        financial_intuition = state.get("financial_intuition") or {}
        hypothesis = state.get("hypothesis") or {}
        factor_spec = state.get("factor_spec") or {}
        evaluation = state.get("evaluation") or {}
        backtest_result = state.get("backtest_result") or {}
        manager_decision = state.get("manager_decision") or {}
        return {
            "record_id": row["record_id"],
            "run_id": row["run_id"],
            "iteration": row["iteration"],
            "attempt": row["attempt"],
            "hypothesis": {
                "title": hypothesis.get("title"),
                "claim": hypothesis.get("claim"),
                "prediction_direction": hypothesis.get(
                    "prediction_direction"
                ),
                "required_concepts": hypothesis.get("required_concepts"),
                "sources": hypothesis.get("sources", []),
                "factor_references": hypothesis.get("factor_references", []),
            },
            "financial_intuition": {
                "intuition_title": financial_intuition.get("intuition_title"),
                "core_mechanism": financial_intuition.get("core_mechanism"),
                "factor_components": financial_intuition.get(
                    "factor_components", []
                ),
            },
            "factor": {
                "factor_id": factor_spec.get("factor_id"),
                "name": factor_spec.get("name"),
                "description": factor_spec.get("description"),
            },
            "evaluation_status": row["evaluation_status"],
            "evaluation_reasons": evaluation.get("reasons", []),
            "failure_profile": SQLiteResearchRepository._failure_profile(
                evaluation
            ),
            "diagnosis": evaluation.get("diagnosis"),
            "evidence": evaluation.get("evidence", []),
            "revision_guidance": evaluation.get("revision_guidance", []),
            "rank_ic": backtest_result.get(
                "rank_ic", backtest_result.get("mean_rank_ic")
            ),
            "icir": backtest_result.get(
                "icir", backtest_result.get("rank_icir")
            ),
            "net_return": backtest_result.get(
                "net_return", backtest_result.get("net_excess_return")
            ),
            "turnover": backtest_result.get("turnover"),
            "max_drawdown": backtest_result.get("max_drawdown"),
            "recommended_action": row["recommended_action"],
            "next_action": row["next_action"],
            "manager_rationale": manager_decision.get("rationale"),
            "manager_instructions": manager_decision.get("instructions"),
        }

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS record_count FROM research_records"
            ).fetchone()
        return int(row["record_count"])

    def contains_hypothesis(self, hypothesis: dict[str, Any]) -> bool:
        signature = hypothesis_signature(hypothesis)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM research_records "
                "WHERE hypothesis_signature = ? LIMIT 1",
                (signature,),
            ).fetchone()
        return row is not None
