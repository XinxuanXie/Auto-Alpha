from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataField(StrEnum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    MKTCAP = "mktcap"


class FormulaOperator(StrEnum):
    FIELD = "field"
    CONSTANT = "constant"
    LAG = "lag"
    RETURN = "return"
    MEAN = "mean"
    STD = "std"
    SUM = "sum"
    ROLLING_MAX = "rolling_max"
    ROLLING_MIN = "rolling_min"
    QUANTILE = "quantile"
    TS_RANK = "ts_rank"
    IDX_MAX = "idx_max"
    IDX_MIN = "idx_min"
    CORRELATION = "correlation"
    SLOPE = "slope"
    RSQUARE = "rsquare"
    RESIDUAL = "residual"
    ZSCORE = "zscore"
    RANK = "rank"
    ABS = "abs"
    LOG = "log"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    NEGATE = "negate"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    ELEMENTWISE_MAX = "elementwise_max"
    ELEMENTWISE_MIN = "elementwise_min"
    WHERE = "where"


ROLLING_OPERATORS = {
    FormulaOperator.MEAN,
    FormulaOperator.STD,
    FormulaOperator.SUM,
    FormulaOperator.ROLLING_MAX,
    FormulaOperator.ROLLING_MIN,
    FormulaOperator.QUANTILE,
    FormulaOperator.TS_RANK,
    FormulaOperator.IDX_MAX,
    FormulaOperator.IDX_MIN,
    FormulaOperator.SLOPE,
    FormulaOperator.RSQUARE,
    FormulaOperator.RESIDUAL,
}
WINDOWED_UNARY_OPERATORS = {
    FormulaOperator.LAG,
    FormulaOperator.RETURN,
    *ROLLING_OPERATORS,
}
UNARY_OPERATORS = {
    *WINDOWED_UNARY_OPERATORS,
    FormulaOperator.ZSCORE,
    FormulaOperator.RANK,
    FormulaOperator.NEGATE,
    FormulaOperator.ABS,
    FormulaOperator.LOG,
}
BINARY_OPERATORS = {
    FormulaOperator.ADD,
    FormulaOperator.SUBTRACT,
    FormulaOperator.MULTIPLY,
    FormulaOperator.DIVIDE,
    FormulaOperator.GREATER_THAN,
    FormulaOperator.LESS_THAN,
    FormulaOperator.ELEMENTWISE_MAX,
    FormulaOperator.ELEMENTWISE_MIN,
}
ROLLING_BINARY_OPERATORS = {FormulaOperator.CORRELATION}
TERNARY_OPERATORS = {FormulaOperator.WHERE}


class FormulaNode(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    op: FormulaOperator
    field: DataField | None = None
    value: float | None = None
    window: int | None = Field(default=None, ge=1, le=504)
    min_periods: int | None = Field(default=None, ge=1, le=504)
    quantile: float | None = Field(default=None, ge=0.0, le=1.0)
    ddof: Literal[0, 1] | None = None
    args: list["FormulaNode"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self) -> Self: # 不同operator对应的args不同，检查
        if self.op == FormulaOperator.FIELD:
            if self.field is None or self.args or self.value is not None:
                raise ValueError("field requires one field and no args or value")
            if any(
                parameter is not None
                for parameter in (
                    self.window,
                    self.min_periods,
                    self.quantile,
                    self.ddof,
                )
            ):
                raise ValueError("field does not accept rolling parameters")
            return self

        if self.op == FormulaOperator.CONSTANT:
            if self.value is None or self.field is not None or self.args:
                raise ValueError("constant requires one value")
            if any(
                parameter is not None
                for parameter in (
                    self.window,
                    self.min_periods,
                    self.quantile,
                    self.ddof,
                )
            ):
                raise ValueError("constant does not accept rolling parameters")
            return self

        if self.op in UNARY_OPERATORS:
            operand_count = len(self.args) + int(self.field is not None)
            if operand_count != 1 or self.value is not None:
                raise ValueError(
                    f"{self.op.value} requires exactly one field or argument"
                )
            if self.op in WINDOWED_UNARY_OPERATORS and self.window is None:
                raise ValueError(f"{self.op.value} requires window")
            if self.op not in WINDOWED_UNARY_OPERATORS and self.window is not None:
                raise ValueError(f"{self.op.value} does not accept window")
            self._validate_rolling_parameters()
            return self

        if self.op in BINARY_OPERATORS:
            if (
                len(self.args) != 2
                or self.field is not None
                or self.value is not None
                or self.window is not None
                or self.min_periods is not None
                or self.quantile is not None
                or self.ddof is not None
            ):
                raise ValueError(f"{self.op.value} requires exactly two args")
            return self

        if self.op in ROLLING_BINARY_OPERATORS:
            if len(self.args) != 2 or self.field is not None or self.value is not None:
                raise ValueError(f"{self.op.value} requires exactly two args")
            if self.window is None:
                raise ValueError(f"{self.op.value} requires window")
            self._validate_rolling_parameters()
            return self

        if self.op in TERNARY_OPERATORS:
            if (
                len(self.args) != 3
                or self.field is not None
                or self.value is not None
                or self.window is not None
                or self.min_periods is not None
                or self.quantile is not None
                or self.ddof is not None
            ):
                raise ValueError(f"{self.op.value} requires exactly three args")
            return self

        raise ValueError(f"unsupported operator: {self.op}")

    def _validate_rolling_parameters(self) -> None:
        if self.min_periods is not None:
            if self.op not in ROLLING_OPERATORS | ROLLING_BINARY_OPERATORS:
                raise ValueError(f"{self.op.value} does not accept min_periods")
            if self.window is not None and self.min_periods > self.window:
                raise ValueError("min_periods cannot exceed window")
        if self.quantile is not None and self.op != FormulaOperator.QUANTILE:
            raise ValueError(f"{self.op.value} does not accept quantile")
        if self.op == FormulaOperator.QUANTILE and self.quantile is None:
            raise ValueError("quantile requires quantile")
        if self.ddof is not None and self.op != FormulaOperator.STD:
            raise ValueError(f"{self.op.value} does not accept ddof")

    def referenced_fields(self) -> set[DataField]:
        fields = {self.field} if self.field is not None else set()
        for argument in self.args:
            fields.update(argument.referenced_fields())
        return fields

    def required_history(self) -> int:
        child_history = max(
            (argument.required_history() for argument in self.args),
            default=0,
        )
        if self.op in {FormulaOperator.LAG, FormulaOperator.RETURN}:
            return child_history + int(self.window or 0)
        if self.op in ROLLING_OPERATORS | ROLLING_BINARY_OPERATORS:
            return child_history + int(self.window or 1) - 1
        return child_history


class UniverseFilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min_price: float | None = Field(default=None, gt=0)
    min_liquidity_percentile: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    liquidity_lookback_days: int = Field(default=20, ge=1, le=504)
    min_market_cap_percentile: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    @model_validator(mode="after")
    def require_filter(self) -> Self:
        if (
            self.min_price is None
            and self.min_liquidity_percentile is None
            and self.min_market_cap_percentile is None
        ):
            raise ValueError("universe_filters must enable at least one filter")
        return self

    def required_history(self) -> int:
        if self.min_liquidity_percentile is None:
            return 0
        return self.liquidity_lookback_days - 1


class PortfolioRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: Literal[20] = 20
    exit_rank: Literal[30] = 30


class FactorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    factor_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1_500)
    hypothesis_id: str | None = Field(default=None, max_length=100)
    formula: FormulaNode
    required_fields: list[DataField]
    lookback_days: int = Field(ge=0, le=504)
    higher_is_better: bool = True
    rebalance_frequency: str = Field(default="weekly", pattern="^weekly$")
    universe_filters: UniverseFilterSpec | None = None
    portfolio_rules: PortfolioRules = Field(default_factory=PortfolioRules)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ValueError("required_fields cannot contain duplicates")
        declared = set(self.required_fields)
        referenced = self.formula.referenced_fields()
        if declared != referenced:
            raise ValueError(
                "required_fields must exactly match formula fields: "
                f"expected {sorted(field.value for field in referenced)}"
            )
        required_history = max(
            self.formula.required_history(),
            (
                self.universe_filters.required_history()
                if self.universe_filters is not None
                else 0
            ),
        )
        if self.lookback_days < required_history:
            raise ValueError(
                f"lookback_days must be at least {required_history} for formula"
            )
        return self


def abnormal_volume_reversal_spec() -> FactorSpec: #接入implementation agent之前用该函数验证工作流是否顺畅
    return FactorSpec.model_validate(
        {
            "factor_id": "F-ABNORMAL-VOLUME-REVERSAL",
            "name": "异常放量短期反转",
            "description": "近期价格变化与异常成交量共同刻画短期过度反应。",
            "hypothesis_id": "H-ABNORMAL-VOLUME-REVERSAL",
            "formula": {
                "op": "negate",
                "args": [
                    {
                        "op": "multiply",
                        "args": [
                            {"op": "return", "field": "close", "window": 5},
                            {
                                "op": "zscore",
                                "args": [
                                    {
                                        "op": "divide",
                                        "args": [
                                            {"op": "field", "field": "volume"},
                                            {
                                                "op": "mean",
                                                "field": "volume",
                                                "window": 20,
                                            },
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            "required_fields": ["close", "volume"],
            "lookback_days": 20,
            "higher_is_better": True,
            "rebalance_frequency": "weekly",
        }
    )
