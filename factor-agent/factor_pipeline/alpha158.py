"""Qlib Alpha158 formula catalog and safe FormulaSpec conversion."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .spec import DataField, FactorSpec, FormulaNode


@dataclass(frozen=True)
class Alpha158SupportRecord:
    factor_id: str
    formula: str
    supported: bool
    reason: str | None = None


def alpha158_formula_catalog() -> tuple[tuple[str, str], ...]:
    expressions = [
        ("KMID", "($close-$open)/$open"),
        ("KLEN", "($high-$low)/$open"),
        ("KMID2", "($close-$open)/($high-$low+1e-12)"),
        ("KUP", "($high-Greater($open,$close))/$open"),
        ("KUP2", "($high-Greater($open,$close))/($high-$low+1e-12)"),
        ("KLOW", "(Less($open,$close)-$low)/$open"),
        ("KLOW2", "(Less($open,$close)-$low)/($high-$low+1e-12)"),
        ("KSFT", "(2*$close-$high-$low)/$open"),
        ("KSFT2", "(2*$close-$high-$low)/($high-$low+1e-12)"),
        ("OPEN0", "$open/$close"),
        ("HIGH0", "$high/$close"),
        ("LOW0", "$low/$close"),
        ("VWAP0", "$vwap/$close"),
    ]
    templates = {
        "ROC": "Ref($close,{d})/$close",
        "MA": "Mean($close,{d})/$close",
        "STD": "Std($close,{d})/$close",
        "BETA": "Slope($close,{d})/$close",
        "RSQR": "Rsquare($close,{d})",
        "RESI": "Resi($close,{d})/$close",
        "MAX": "Max($high,{d})/$close",
        "MIN": "Min($low,{d})/$close",
        "QTLU": "Quantile($close,{d},0.8)/$close",
        "QTLD": "Quantile($close,{d},0.2)/$close",
        "RANK": "Rank($close,{d})",
        "RSV": "($close-Min($low,{d}))/(Max($high,{d})-Min($low,{d})+1e-12)",
        "IMAX": "IdxMax($high,{d})/{d}",
        "IMIN": "IdxMin($low,{d})/{d}",
        "IMXD": "(IdxMax($high,{d})-IdxMin($low,{d}))/{d}",
        "CORR": "Corr($close,Log($volume+1),{d})",
        "CORD": "Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),{d})",
        "CNTP": "Mean($close>Ref($close,1),{d})",
        "CNTN": "Mean($close<Ref($close,1),{d})",
        "CNTD": "Mean($close>Ref($close,1),{d})-Mean($close<Ref($close,1),{d})",
        "SUMP": "Sum(Greater($close-Ref($close,1),0),{d})/(Sum(Abs($close-Ref($close,1)),{d})+1e-12)",
        "SUMN": "Sum(Greater(Ref($close,1)-$close,0),{d})/(Sum(Abs($close-Ref($close,1)),{d})+1e-12)",
        "SUMD": "(Sum(Greater($close-Ref($close,1),0),{d})-Sum(Greater(Ref($close,1)-$close,0),{d}))/(Sum(Abs($close-Ref($close,1)),{d})+1e-12)",
        "VMA": "Mean($volume,{d})/($volume+1e-12)",
        "VSTD": "Std($volume,{d})/($volume+1e-12)",
        "WVMA": "Std(Abs($close/Ref($close,1)-1)*$volume,{d})/(Mean(Abs($close/Ref($close,1)-1)*$volume,{d})+1e-12)",
        "VSUMP": "Sum(Greater($volume-Ref($volume,1),0),{d})/(Sum(Abs($volume-Ref($volume,1)),{d})+1e-12)",
        "VSUMN": "Sum(Greater(Ref($volume,1)-$volume,0),{d})/(Sum(Abs($volume-Ref($volume,1)),{d})+1e-12)",
        "VSUMD": "(Sum(Greater($volume-Ref($volume,1),0),{d})-Sum(Greater(Ref($volume,1)-$volume,0),{d}))/(Sum(Abs($volume-Ref($volume,1)),{d})+1e-12)",
    }
    for prefix, template in templates.items():
        expressions.extend(
            (f"{prefix}{window}", template.format(d=window))
            for window in (5, 10, 20, 30, 60)
        )
    if len(expressions) != 158:
        raise ValueError(f"expected 158 Alpha158 formulas, found {len(expressions)}")
    return tuple(expressions)


def executable_alpha158_specs() -> tuple[FactorSpec, ...]:
    return tuple(
        alpha158_factor_spec(record.factor_id.removeprefix("ALPHA158-"), record.formula)
        for record in alpha158_support_manifest()
        if record.supported
    )


def alpha158_support_manifest() -> tuple[Alpha158SupportRecord, ...]:
    return tuple(
        Alpha158SupportRecord(
            factor_id=f"ALPHA158-{name}",
            formula=formula,
            supported=name != "VWAP0",
            reason=(
                None
                if name != "VWAP0"
                else "requires true VWAP; daily OHLCV cannot reconstruct it exactly"
            ),
        )
        for name, formula in alpha158_formula_catalog()
    )


def alpha158_factor_spec(name: str, formula: str) -> FactorSpec:
    if name == "VWAP0" or "$vwap" in formula.casefold():
        raise ValueError("VWAP0 requires true VWAP data and is not executable")
    root = _Alpha158Parser().parse(formula)
    fields = sorted(root.referenced_fields(), key=lambda field: field.value)
    return FactorSpec(
        factor_id=f"ALPHA158-{name}",
        name=f"Qlib Alpha158 {name}",
        description=f"Qlib Alpha158 formula: {formula}",
        formula=root,
        required_fields=fields,
        lookback_days=root.required_history(),
        higher_is_better=True,
        rebalance_frequency="weekly",
    )


class _Alpha158Parser:
    _field_pattern = re.compile(r"\$(open|high|low|close|volume)\b")

    def parse(self, formula: str) -> FormulaNode:
        normalized = self._field_pattern.sub(r"\1", formula)
        try:
            expression = ast.parse(normalized, mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"invalid Alpha158 formula: {formula}") from error
        return self._convert(expression)

    def _convert(self, node: ast.AST) -> FormulaNode:
        if isinstance(node, ast.Name):
            try:
                return FormulaNode(op="field", field=DataField(node.id))
            except ValueError as error:
                raise ValueError(f"unsupported field: {node.id}") from error
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return FormulaNode(op="constant", value=float(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return FormulaNode(op="negate", args=[self._convert(node.operand)])
        if isinstance(node, ast.BinOp):
            operators = {
                ast.Add: "add",
                ast.Sub: "subtract",
                ast.Mult: "multiply",
                ast.Div: "divide",
            }
            operator = operators.get(type(node.op))
            if operator is None:
                raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
            return FormulaNode(
                op=operator,
                args=[self._convert(node.left), self._convert(node.right)],
            )
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            operators = {ast.Gt: "greater_than", ast.Lt: "less_than"}
            operator = operators.get(type(node.ops[0]))
            if operator is None:
                raise ValueError(f"unsupported comparison: {type(node.ops[0]).__name__}")
            return FormulaNode(
                op=operator,
                args=[self._convert(node.left), self._convert(node.comparators[0])],
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self._convert_call(node.func.id, node.args)
        raise ValueError(f"unsupported Alpha158 syntax: {ast.dump(node)}")

    def _convert_call(self, name: str, arguments: list[ast.expr]) -> FormulaNode:
        unary = {"Abs": "abs", "Log": "log"}
        elementwise = {"Greater": "elementwise_max", "Less": "elementwise_min"}
        rolling = {
            "Mean": "mean",
            "Std": "std",
            "Sum": "sum",
            "Max": "rolling_max",
            "Min": "rolling_min",
            "Rank": "ts_rank",
            "IdxMax": "idx_max",
            "IdxMin": "idx_min",
            "Slope": "slope",
            "Rsquare": "rsquare",
            "Resi": "residual",
        }
        if name in unary and len(arguments) == 1:
            return FormulaNode(op=unary[name], args=[self._convert(arguments[0])])
        if name in elementwise and len(arguments) == 2:
            return FormulaNode(
                op=elementwise[name],
                args=[self._convert(argument) for argument in arguments],
            )
        if name == "Ref" and len(arguments) == 2:
            return FormulaNode(
                op="lag",
                args=[self._convert(arguments[0])],
                window=self._integer(arguments[1]),
            )
        if name in rolling and len(arguments) == 2:
            parameters = {
                "op": rolling[name],
                "args": [self._convert(arguments[0])],
                "window": self._integer(arguments[1]),
                "min_periods": 1,
            }
            if name == "Std":
                parameters["ddof"] = 1
            return FormulaNode.model_validate(parameters)
        if name == "Quantile" and len(arguments) == 3:
            return FormulaNode(
                op="quantile",
                args=[self._convert(arguments[0])],
                window=self._integer(arguments[1]),
                min_periods=1,
                quantile=self._number(arguments[2]),
            )
        if name == "Corr" and len(arguments) == 3:
            return FormulaNode(
                op="correlation",
                args=[self._convert(arguments[0]), self._convert(arguments[1])],
                window=self._integer(arguments[2]),
                min_periods=1,
            )
        raise ValueError(f"unsupported Alpha158 function: {name}/{len(arguments)}")

    @staticmethod
    def _number(node: ast.AST) -> float:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float)):
            raise ValueError("rolling parameter must be numeric")
        return float(node.value)

    def _integer(self, node: ast.AST) -> int:
        value = self._number(node)
        if not value.is_integer():
            raise ValueError("rolling window must be an integer")
        return int(value)
