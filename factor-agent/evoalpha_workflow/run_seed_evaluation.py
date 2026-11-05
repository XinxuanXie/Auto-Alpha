"""Command-line entry point for real Alpha158 seed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .seed_evaluation import Alpha158SeedEvaluationConfig, Alpha158SeedEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 157 executable Alpha158 seeds")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("factor-data/panel.parquet"),
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001"),
    )
    parser.add_argument("--top-k", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise ValueError("top-k must be at least 1")

    def report_progress(index: int, total: int, factor_id: str) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(f"[{index}/{total}] {factor_id}", flush=True)

    evaluator = Alpha158SeedEvaluator(
        Alpha158SeedEvaluationConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            output_dir=args.output_dir,
        ),
        progress_callback=report_progress,
    )
    candidates = evaluator.evaluate_seeds()
    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.passed_hard_filters
            and candidate.composite_score is not None
        ),
        key=lambda candidate: (-candidate.composite_score, candidate.factor_id),
    )
    if len(ranked) < args.top_k:
        raise ValueError(
            f"only {len(ranked)} valid seeds are available; cannot select Top {args.top_k}"
        )
    top_candidates = ranked[: args.top_k]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"alpha158_top{args.top_k}.json").write_text(
        json.dumps(
            [candidate.model_dump(mode="json") for candidate in top_candidates],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "evaluated": len(candidates),
        "passed_hard_filters": len(ranked),
        "rejected_or_failed": len(candidates) - len(ranked),
        "selected": len(top_candidates),
        "best_factor_id": top_candidates[0].factor_id,
        "best_composite_score": top_candidates[0].composite_score,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
