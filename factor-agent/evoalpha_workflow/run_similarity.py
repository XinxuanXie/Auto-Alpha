"""CLI for Top-30 training-period factor similarity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .similarity import SimilarityConfig, TrainingFactorSimilarityAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Top-30 factor similarity")
    parser.add_argument("--panel", type=Path, default=Path("factor-data/panel.parquet"))
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("factor-data/membership_intervals.parquet"),
    )
    parser.add_argument(
        "--top-candidates",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001/alpha158_top30.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("factor-agent/results/evoalpha_seed_2001"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def report_progress(index: int, total: int, factor_id: str) -> None:
        if index == 1 or index % 5 == 0 or index == total:
            print(f"[{index}/{total}] {factor_id}", flush=True)

    result = TrainingFactorSimilarityAnalyzer(
        SimilarityConfig(
            panel_path=args.panel,
            membership_path=args.membership,
            output_dir=args.output_dir,
        ),
        progress_callback=report_progress,
    ).analyze_top_candidates(args.top_candidates)
    print(
        json.dumps(
            {
                "factor_count": len(result.factor_ids),
                "pair_count": len(result.pairs),
                "eligible_pairs": result.eligible_pairs,
                "excluded_pairs": result.excluded_pairs,
                "lowest_similarity_pair": as_pair(result.pairs[0]),
                "highest_similarity_pair": as_pair(result.pairs[-1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def as_pair(pair) -> dict[str, object]:
    return {
        "factor_a": pair.factor_a,
        "factor_b": pair.factor_b,
        "median_absolute_spearman": pair.median_absolute_spearman,
        "median_signed_spearman": pair.median_signed_spearman,
    }


if __name__ == "__main__":
    raise SystemExit(main())
