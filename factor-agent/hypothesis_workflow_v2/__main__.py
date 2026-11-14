"""Run the complete V2 workflow with mock or real tools."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from .artifacts import ArtifactRepository
from .contracts import ArtifactRef
from .manager import build_manager
from .state import create_initial_state
from .tools import MockV2Tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("mock", "real-smoke", "real-full"),
        default="mock",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--env-file", type=Path, default=Path("../.env"))
    parser.add_argument(
        "--panel-path", type=Path, default=Path("../factor-data/panel.parquet")
    )
    parser.add_argument(
        "--membership-path",
        type=Path,
        default=Path("../factor-data/membership_intervals.parquet"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")

    output_dir = args.output_dir or Path(
        {
            "mock": ".runtime/hypothesis_workflow_v2_mock",
            "real-smoke": ".runtime/hypothesis_workflow_v2_real_smoke",
            "real-full": ".runtime/hypothesis_workflow_v2_real_full",
        }[args.mode]
    )
    if args.mode in {"real-smoke", "real-full"}:
        from .real_tools import RealV2Tools

        load_dotenv(args.env_file, override=False)
        full_run = args.mode == "real-full"
        tools = RealV2Tools(
            args.panel_path,
            args.membership_path,
            min_hypotheses_per_idea=1,
            max_hypotheses_per_idea=4 if full_run else 1,
            max_experiments_per_hypothesis=5 if full_run else 2,
            cache_dir=output_dir / ".factor_spec_cache",
        )
    else:
        tools = MockV2Tools()

    repository = ArtifactRepository(output_dir)
    initial_state = create_initial_state(
        {
            "mode": args.mode,
            "llm_concurrency": 1 if args.mode != "mock" else 4,
            "backtest_concurrency": 1,
        },
        run_id=args.run_id,
    )
    run_id = initial_state["run_id"]
    print(json.dumps({"run_id": run_id, "mode": args.mode}), flush=True)
    checkpoint_path = output_dir / "checkpoints.sqlite"
    with sqlite3.connect(checkpoint_path, check_same_thread=False) as connection:
        manager = build_manager(
            tools,
            repository,
            checkpointer=SqliteSaver(connection),
        )
        config = {"configurable": {"thread_id": run_id}}
        result = manager.invoke(None if args.resume else initial_state, config=config)
    summary_ref = result["artifact_refs"].get("run_summary")
    summary = (
        repository.read_json(ArtifactRef.model_validate(summary_ref))
        if summary_ref
        else None
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "stage": result["stage"],
                "summary": summary,
                "events": result["events"],
                "error": result["error"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
