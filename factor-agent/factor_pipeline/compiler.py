from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .spec import FactorSpec, FormulaNode, FormulaOperator


class FactorCompilationError(ValueError):
    pass


@dataclass
class _CompilationContext:
    frame: pl.LazyFrame
    counter: int = 0
    time_index_column: str | None = None

    def add_expression(self, expression: pl.Expr) -> str:
        self.counter += 1
        column = f"_dsl_{self.counter:04d}"
        self.frame = self.frame.with_columns(expression.alias(column))
        return column

    def ensure_time_index(self) -> str:
        if self.time_index_column is None:
            self.time_index_column = self.add_expression(
                pl.col("date").cum_count().over("permno").cast(pl.Float64)
            )
        return self.time_index_column


def compile_factor_spec(panel: pl.LazyFrame, spec: FactorSpec) -> pl.LazyFrame:
    schema = panel.collect_schema()
    missing = {
        field.value for field in spec.required_fields if field.value not in schema
    }
    if missing: # 确保数据完整性
        raise FactorCompilationError(
            f"panel is missing factor fields: {sorted(missing)}"
        )
    context = _CompilationContext(panel.sort("permno", "date")) # 按股票编号&日期排序
    output_column = _compile_node(context, spec.formula)
    raw_factor = pl.col(output_column)
    if not spec.higher_is_better:
        raw_factor = -raw_factor # 因子方向
    return context.frame.with_columns(raw_factor.alias("raw_factor"))


def _compile_node(context: _CompilationContext, node: FormulaNode) -> str:
    if node.op == FormulaOperator.FIELD:
        return node.field.value
    if node.op == FormulaOperator.CONSTANT:
        return context.add_expression(pl.lit(node.value, dtype=pl.Float64))

    operand_columns = [_compile_node(context, argument) for argument in node.args]
    if node.field is not None:
        operand_columns.append(node.field.value)

    if node.op == FormulaOperator.LAG:
        expression = pl.col(operand_columns[0]).shift(node.window).over("permno")
    elif node.op == FormulaOperator.RETURN:
        current = pl.col(operand_columns[0])
        lagged = current.shift(node.window).over("permno")
        expression = current / lagged - 1.0
    elif node.op == FormulaOperator.MEAN:
        expression = (
            pl.col(operand_columns[0])
            .rolling_mean(
                window_size=node.window,
                min_samples=_minimum_periods(node),
            )
            .over("permno")
        )
    elif node.op == FormulaOperator.STD:
        expression = (
            pl.col(operand_columns[0])
            .rolling_std(
                window_size=node.window,
                min_samples=_minimum_periods(node),
                ddof=0 if node.ddof is None else node.ddof,
            )
            .over("permno")
        )
    elif node.op == FormulaOperator.SUM:
        expression = (
            pl.col(operand_columns[0])
            .rolling_sum(
                window_size=node.window,
                min_samples=_minimum_periods(node),
            )
            .over("permno")
        )
    elif node.op == FormulaOperator.ROLLING_MAX:
        expression = (
            pl.col(operand_columns[0])
            .rolling_max(
                window_size=node.window,
                min_samples=_minimum_periods(node),
            )
            .over("permno")
        )
    elif node.op == FormulaOperator.ROLLING_MIN:
        expression = (
            pl.col(operand_columns[0])
            .rolling_min(
                window_size=node.window,
                min_samples=_minimum_periods(node),
            )
            .over("permno")
        )
    elif node.op == FormulaOperator.QUANTILE:
        expression = (
            pl.col(operand_columns[0])
            .rolling_quantile(
                quantile=node.quantile,
                interpolation="linear",
                window_size=node.window,
                min_samples=_minimum_periods(node),
            )
            .over("permno")
        )
    elif node.op == FormulaOperator.TS_RANK:
        expression = _rolling_map(node, operand_columns[0], _percentile_rank)
    elif node.op == FormulaOperator.IDX_MAX:
        expression = _rolling_map(node, operand_columns[0], _index_max)
    elif node.op == FormulaOperator.IDX_MIN:
        expression = _rolling_map(node, operand_columns[0], _index_min)
    elif node.op == FormulaOperator.SLOPE:
        expression = _rolling_slope(
            node,
            context.ensure_time_index(),
            operand_columns[0],
        )
    elif node.op == FormulaOperator.RSQUARE:
        expression = _rolling_r_square(
            node,
            context.ensure_time_index(),
            operand_columns[0],
        )
    elif node.op == FormulaOperator.RESIDUAL:
        time_index = context.ensure_time_index()
        slope = _rolling_slope(node, time_index, operand_columns[0])
        mean_x = _rolling_mean(node, time_index)
        mean_y = _rolling_mean(node, operand_columns[0])
        expression = pl.col(operand_columns[0]) - (
            mean_y + slope * (pl.col(time_index) - mean_x)
        )
    elif node.op == FormulaOperator.CORRELATION:
        expression = pl.rolling_corr(
            pl.col(operand_columns[0]),
            pl.col(operand_columns[1]),
            window_size=node.window,
            min_samples=_minimum_periods(node),
            ddof=1,
        ).over("permno")
    
    # 横截面操作
    elif node.op == FormulaOperator.ZSCORE:
        value = pl.col(operand_columns[0])
        mean = value.mean().over("date")
        standard_deviation = value.std(ddof=0).over("date")
        expression = (
            pl.when(standard_deviation > 0)
            .then((value - mean) / standard_deviation)
            .otherwise(None)
        )
    elif node.op == FormulaOperator.RANK:
        value = pl.col(operand_columns[0])
        expression = value.rank(method="average").over("date") / pl.len().over(
            "date"
        )
    elif node.op == FormulaOperator.NEGATE:
        expression = -pl.col(operand_columns[0])
    elif node.op == FormulaOperator.ABS:
        expression = pl.col(operand_columns[0]).abs()
    elif node.op == FormulaOperator.LOG:
        value = pl.col(operand_columns[0])
        expression = pl.when(value > 0).then(value.log()).otherwise(None)
    elif node.op in {
        FormulaOperator.ADD,
        FormulaOperator.SUBTRACT,
        FormulaOperator.MULTIPLY,
        FormulaOperator.DIVIDE,
        FormulaOperator.GREATER_THAN,
        FormulaOperator.LESS_THAN,
        FormulaOperator.ELEMENTWISE_MAX,
        FormulaOperator.ELEMENTWISE_MIN,
    }:
        left = pl.col(operand_columns[0])
        right = pl.col(operand_columns[1])
        if node.op == FormulaOperator.ADD:
            expression = left + right
        elif node.op == FormulaOperator.SUBTRACT:
            expression = left - right
        elif node.op == FormulaOperator.MULTIPLY:
            expression = left * right
        elif node.op == FormulaOperator.DIVIDE:
            expression = (
                pl.when(right.abs() > 1e-12).then(left / right).otherwise(None)
            )
        elif node.op == FormulaOperator.GREATER_THAN:
            expression = left > right
        elif node.op == FormulaOperator.LESS_THAN:
            expression = left < right
        elif node.op == FormulaOperator.ELEMENTWISE_MAX:
            expression = pl.max_horizontal(left, right)
        else:
            expression = pl.min_horizontal(left, right)
    elif node.op == FormulaOperator.WHERE:
        condition = pl.col(operand_columns[0])
        expression = (
            pl.when(condition)
            .then(pl.col(operand_columns[1]))
            .otherwise(pl.col(operand_columns[2]))
        )
    else:
        raise FactorCompilationError(f"unsupported operator: {node.op.value}")
    return context.add_expression(expression)


def _minimum_periods(node: FormulaNode) -> int:
    return node.window if node.min_periods is None else node.min_periods


def _rolling_map(
    node: FormulaNode,
    column: str,
    function,
) -> pl.Expr:
    return (
        pl.col(column)
        .rolling_map(
            function,
            window_size=node.window,
            min_samples=_minimum_periods(node),
        )
        .over("permno")
    )


def _rolling_mean(node: FormulaNode, column: str) -> pl.Expr:
    return (
        pl.col(column)
        .rolling_mean(
            window_size=node.window,
            min_samples=_minimum_periods(node),
        )
        .over("permno")
    )


def _rolling_slope(node: FormulaNode, x_column: str, y_column: str) -> pl.Expr:
    covariance = pl.rolling_cov(
        pl.col(x_column),
        pl.col(y_column),
        window_size=node.window,
        min_samples=_minimum_periods(node),
        ddof=1,
    ).over("permno")
    variance = (
        pl.col(x_column)
        .rolling_var(
            window_size=node.window,
            min_samples=_minimum_periods(node),
            ddof=1,
        )
        .over("permno")
    )
    return pl.when(variance.abs() > 1e-12).then(covariance / variance).otherwise(None)


def _rolling_r_square(node: FormulaNode, x_column: str, y_column: str) -> pl.Expr:
    correlation = pl.rolling_corr(
        pl.col(x_column),
        pl.col(y_column),
        window_size=node.window,
        min_samples=_minimum_periods(node),
        ddof=1,
    ).over("permno")
    return correlation.pow(2)


def _finite_values(series: pl.Series) -> np.ndarray | None:
    values = series.cast(pl.Float64).to_numpy()
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def _percentile_rank(series: pl.Series) -> float:
    values = _finite_values(series)
    if values is None:
        return float("nan")
    latest = values[-1]
    lower = np.count_nonzero(values < latest)
    equal = np.count_nonzero(values == latest)
    return float((lower + (equal + 1.0) / 2.0) / values.size)


def _index_max(series: pl.Series) -> float:
    values = _finite_values(series)
    return float("nan") if values is None else float(np.argmax(values) + 1)


def _index_min(series: pl.Series) -> float:
    values = _finite_values(series)
    return float("nan") if values is None else float(np.argmin(values) + 1)


