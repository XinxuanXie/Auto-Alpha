from __future__ import annotations

import polars as pl

from .compiler import compile_factor_spec
from .config import FactorConfig
from .spec import FactorSpec, UniverseFilterSpec, abnormal_volume_reversal_spec


def _cross_sectional_zscore(column: str, output: str) -> pl.Expr:
    centered = pl.col(column) - pl.col(column).mean().over("date")
    standard_deviation = pl.col(column).std(ddof=0).over("date")
    return (
        pl.when(standard_deviation > 0)
        .then(centered / standard_deviation)
        .otherwise(None)
        .alias(output)
    )


def build_abnormal_volume_reversal(
    panel: pl.LazyFrame, config: FactorConfig
) -> pl.LazyFrame:
    return build_factor_from_spec(
        panel,
        abnormal_volume_reversal_spec(),
        config,
    )


def build_factor_from_spec(
    panel: pl.LazyFrame,
    spec: FactorSpec,
    config: FactorConfig,
) -> pl.LazyFrame:
    compiled = compile_factor_spec(panel, spec)
    if spec.universe_filters is not None:
        compiled = apply_universe_filters(compiled, spec.universe_filters)
    return clean_and_size_neutralize(compiled, config)


def apply_universe_filters(
    frame: pl.LazyFrame,
    filters: UniverseFilterSpec,
) -> pl.LazyFrame:
    conditions = []
    if filters.min_price is not None:
        conditions.append(pl.col("close") >= filters.min_price)
    if filters.min_liquidity_percentile is not None:
        frame = frame.with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(
                window_size=filters.liquidity_lookback_days,
                min_samples=filters.liquidity_lookback_days,
            )
            .over("permno")
            .alias("_average_dollar_volume")
        ).with_columns(
            (
                pl.col("_average_dollar_volume")
                .rank(method="average")
                .over("date")
                / pl.len().over("date")
            ).alias("_liquidity_percentile")
        )
        conditions.append(
            pl.col("_liquidity_percentile")
            >= filters.min_liquidity_percentile
        )
    if filters.min_market_cap_percentile is not None:
        frame = frame.with_columns(
            (
                pl.col("mktcap").rank(method="average").over("date")
                / pl.len().over("date")
            ).alias("_market_cap_percentile")
        )
        conditions.append(
            pl.col("_market_cap_percentile")
            >= filters.min_market_cap_percentile
        )
    combined = conditions[0]
    for condition in conditions[1:]:
        combined = combined & condition
    return frame.filter(combined)


def clean_and_size_neutralize(
    frame: pl.LazyFrame, config: FactorConfig
) -> pl.LazyFrame:
    lower = pl.col("raw_factor").quantile(
        config.winsor_lower_quantile, interpolation="linear"
    ).over("date")
    upper = pl.col("raw_factor").quantile(
        config.winsor_upper_quantile, interpolation="linear"
    ).over("date")
    frame = (
        frame.filter(
            pl.col("raw_factor").is_finite()
            & pl.col("mktcap").is_finite()
            & (pl.col("mktcap") > 0)
        )
        .with_columns(
            pl.len().over("date").alias("cross_section_size"),
            pl.col("raw_factor").clip(lower, upper).alias("winsorized_factor"),
            pl.col("mktcap").log().alias("log_mktcap"),
        )
        .filter(pl.col("cross_section_size") >= config.minimum_cross_section)
        .with_columns(
            _cross_sectional_zscore("winsorized_factor", "standardized_factor"),
            (
                pl.col("log_mktcap")
                - pl.col("log_mktcap").mean().over("date")
            ).alias("_centered_log_mktcap"),
        )
        .with_columns(
            (
                (
                    pl.col("standardized_factor")
                    * pl.col("_centered_log_mktcap")
                ).mean().over("date")
                / pl.col("_centered_log_mktcap")
                .pow(2)
                .mean()
                .over("date")
            ).alias("_size_beta")
        )
        .with_columns(
            (
                pl.col("standardized_factor")
                - pl.col("_size_beta") * pl.col("_centered_log_mktcap")
            ).alias("size_neutral_factor")
        )
        .with_columns(
            _cross_sectional_zscore("size_neutral_factor", "factor")
        )
        .filter(pl.col("factor").is_finite())
    )
    return frame.select(
        "date",
        "permno",
        "open",
        "close",
        "mktcap",
        "raw_factor",
        "winsorized_factor",
        "size_neutral_factor",
        "factor",
        "cross_section_size",
    )

