"""Local factor knowledge base for Research Agent inspiration."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

class FactorKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    economic_rationale: str = Field(min_length=1)
    required_data: list[str]
    concepts: list[str] = Field(min_length=1)
    implementation_status: str
    source_url: HttpUrl


QLIB_SOURCE = (
    "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py"
)
ALPHA101_SOURCE = "https://arxiv.org/abs/1601.00991"
FRENCH_SOURCE = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
)


class FactorKnowledgeBase:
    def __init__(self, alpha101_path: str | Path | None = None) -> None:
        path = (
            Path(alpha101_path)
            if alpha101_path is not None
            else Path(__file__).resolve().parents[1]
            / "library"
            / "alpha101_formulas.tsv"
        )
        self.entries = tuple(
            [*_load_alpha101(path), *_build_alpha158(), *_academic_factors()]
        )
        identifiers = [entry.factor_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("factor knowledge base contains duplicate factor_id values")

    def search(
        self,
        concepts: list[str] | tuple[str, ...],
        *,
        iteration: int,
        limit: int = 12,
    ) -> list[dict[str, object]]:
        if iteration < 1:
            raise ValueError("iteration must be at least 1")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        requested = set(concepts)
        by_family: dict[str, list[tuple[int, FactorKnowledge]]] = defaultdict(list)
        for entry in self.entries:
            overlap = len(requested & set(entry.concepts))
            if overlap:
                by_family[entry.family].append((overlap, entry))

        selected = []
        families = ("alpha101", "alpha158", "academic")
        family_limit = max(1, limit // len(families))
        for family in families:
            ranked = sorted(
                by_family.get(family, []),
                key=lambda item: (-item[0], item[1].factor_id),
            )
            if not ranked:
                continue
            offset = ((iteration - 1) * family_limit) % len(ranked)
            rotated = ranked[offset:] + ranked[:offset]
            selected.extend(entry for _, entry in rotated[:family_limit])

        if len(selected) < limit:
            selected_ids = {entry.factor_id for entry in selected}
            remaining = sorted(
                (
                    entry
                    for entry in self.entries
                    if entry.factor_id not in selected_ids
                    and requested & set(entry.concepts)
                ),
                key=lambda entry: entry.factor_id,
            )
            selected.extend(remaining[: limit - len(selected)])
        return [entry.model_dump(mode="json") for entry in selected[:limit]]

    def count_by_family(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            counts[entry.family] += 1
        return dict(sorted(counts.items()))


def _load_alpha101(path: Path) -> list[FactorKnowledge]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        number, formula = line.split("\t", 1)
        required_data = _required_data(formula)
        concepts = _infer_concepts(formula)
        status = _implementation_status(formula, required_data)
        entries.append(
            FactorKnowledge(
                factor_id=f"ALPHA101-{number}",
                name=f"WorldQuant Alpha {int(number)}",
                family="alpha101",
                definition=formula,
                economic_rationale=(
                    "原论文提供公式而未逐条声明经济解释；可作为"
                    f"{_concept_label(concepts)}结构、条件关系和算子组合的研究模板。"
                ),
                required_data=required_data,
                concepts=concepts,
                implementation_status=status,
                source_url=ALPHA101_SOURCE,
            )
        )
    if len(entries) != 101:
        raise ValueError(f"expected 101 Alpha101 formulas, found {len(entries)}")
    return entries


def _build_alpha158() -> list[FactorKnowledge]:
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
    windows = (5, 10, 20, 30, 60)
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
            for window in windows
        )
    entries = []
    for name, formula in expressions:
        required_data = _required_data(formula)
        concepts = _alpha158_concepts(name)
        entries.append(
            FactorKnowledge(
                factor_id=f"ALPHA158-{name}",
                name=f"Qlib Alpha158 {name}",
                family="alpha158",
                definition=formula,
                economic_rationale=_alpha158_rationale(name),
                required_data=required_data,
                concepts=concepts,
                implementation_status=_implementation_status(
                    formula,
                    required_data,
                ),
                source_url=QLIB_SOURCE,
            )
        )
    if len(entries) != 158:
        raise ValueError(f"expected 158 Alpha158 factors, found {len(entries)}")
    return entries


def _academic_factors() -> list[FactorKnowledge]:
    rows = [
        ("MOM-12-1", "12-1 Momentum", "Return from month -12 to -2", "Information diffuses gradually and past winners may continue outperforming.", ["close"], ["price_momentum", "recent_return"], ALPHA101_SOURCE),
        ("REV-1M", "Short-Term Reversal", "Negative return over the most recent month", "Temporary price pressure and liquidity shocks may reverse over short horizons.", ["close"], ["price_reversal", "recent_return"], ALPHA101_SOURCE),
        ("LOWVOL", "Low Volatility", "Negative rolling standard deviation of daily returns", "Low-risk stocks may earn stronger risk-adjusted returns than high-volatility stocks.", ["close"], ["volatility"], FRENCH_SOURCE),
        ("IDIOVOL", "Idiosyncratic Volatility", "Residual return volatility after market-factor adjustment", "Stocks with extreme idiosyncratic volatility may be overpriced or difficult to arbitrage.", ["close", "market_return"], ["volatility"], FRENCH_SOURCE),
        ("AMIHUD", "Amihud Illiquidity", "Mean absolute return divided by dollar volume", "Larger price response per unit of trading indicates illiquidity and potential compensation.", ["close", "volume"], ["liquidity", "price_volume_interaction"], "https://doi.org/10.1016/S0304-405X(01)00024-6"),
        ("TURNOVER", "Share Turnover", "Trading volume divided by shares outstanding", "Trading intensity can proxy for liquidity, attention and disagreement.", ["volume", "shares_outstanding"], ["liquidity", "volume_trend"], FRENCH_SOURCE),
        ("SIZE", "Equity Size", "Negative log market capitalization", "Smaller firms may command compensation for risk, information frictions or limited attention.", ["mktcap"], ["market_cap"], FRENCH_SOURCE),
        ("VALUE-BM", "Book-to-Market", "Book equity divided by market equity", "Cheap firms may earn a distress or mispricing premium.", ["book_equity", "mktcap"], ["market_cap"], FRENCH_SOURCE),
        ("VALUE-EP", "Earnings Yield", "Earnings divided by market capitalization", "High earnings yield identifies inexpensive firms relative to current profits.", ["earnings", "mktcap"], ["market_cap"], FRENCH_SOURCE),
        ("PROFIT", "Operating Profitability", "Operating profits divided by book equity", "More profitable firms may sustain stronger future cash flows and returns.", ["operating_profit", "book_equity"], ["market_cap"], FRENCH_SOURCE),
        ("INVEST", "Conservative Investment", "Negative annual asset growth", "Aggressive corporate investment may reflect overexpansion or lower expected returns.", ["total_assets"], ["market_cap"], FRENCH_SOURCE),
        ("ACCRUAL", "Operating Accruals", "Negative accrual component of earnings scaled by assets", "Accrual-heavy earnings may be less persistent than cash-flow earnings.", ["financial_statements"], ["market_cap"], "https://doi.org/10.2307/2491082"),
        ("MAX", "Maximum Daily Return", "Negative maximum daily return during the previous month", "Lottery-like stocks with extreme recent gains may become overpriced.", ["close"], ["volatility", "recent_return"], "https://doi.org/10.1093/rfs/hhn034"),
        ("RANGE", "High-Low Range Volatility", "Rolling mean of (high-low)/close", "Wide intraday ranges capture uncertainty, trading frictions and disagreement.", ["high", "low", "close"], ["intraday_range", "volatility"], FRENCH_SOURCE),
        ("VOL-SHOCK", "Abnormal Volume", "Current volume divided by rolling mean volume", "Unexpected trading activity can reveal attention, information arrival or crowded trading.", ["volume"], ["volume_abnormality", "volume_trend"], ALPHA101_SOURCE),
        ("PV-DIVERGENCE", "Price-Volume Divergence", "Price trend minus standardized volume trend", "A price move without participation confirmation may be fragile and prone to reversal.", ["close", "volume"], ["price_volume_interaction", "price_momentum", "volume_trend"], ALPHA101_SOURCE),
        ("PV-CONFIRM", "Price-Volume Confirmation", "Price trend multiplied by volume trend", "Aligned price and volume trends may indicate broader participation and information diffusion.", ["close", "volume"], ["price_volume_interaction", "price_momentum", "volume_trend"], ALPHA101_SOURCE),
        ("GAP-REV", "Overnight Gap Reversal", "Negative open-to-previous-close return", "Overnight information shocks can overshoot before intraday liquidity restores prices.", ["open", "close"], ["price_reversal", "recent_return"], ALPHA101_SOURCE),
        ("INTRADAY-MOM", "Intraday Momentum", "Close-to-open return", "Persistent order flow during the session can create short-horizon continuation.", ["open", "close"], ["price_momentum", "recent_return"], ALPHA101_SOURCE),
        ("RANGE-POS", "Close Location Value", "(2*close-high-low)/(high-low)", "The close location within the daily range summarizes late-session buying or selling pressure.", ["high", "low", "close"], ["intraday_range", "price_momentum"], QLIB_SOURCE),
    ]
    entries = []
    available = {"open", "high", "low", "close", "volume", "mktcap"}
    for factor_id, name, definition, rationale, required, concepts, source in rows:
        status = (
            "available_now"
            if set(required) <= available
            else "requires_additional_data"
        )
        entries.append(
            FactorKnowledge(
                factor_id=f"ACADEMIC-{factor_id}",
                name=name,
                family="academic",
                definition=definition,
                economic_rationale=rationale,
                required_data=required,
                concepts=concepts,
                implementation_status=status,
                source_url=source,
            )
        )
    return entries


def _required_data(formula: str) -> list[str]:
    lowered = formula.casefold()
    required = set()
    for field in ("open", "high", "low", "close", "volume", "vwap"):
        if re.search(rf"\b{field}\b", lowered):
            required.add(field)
    if "returns" in lowered:
        required.add("close")
    if re.search(r"\badv\d+\b", lowered):
        required.update({"close", "volume"})
    if re.search(r"\bcap\b", lowered):
        required.add("mktcap")
    if "indclass" in lowered:
        required.add("industry_classification")
    return sorted(required)


def _infer_concepts(formula: str) -> list[str]:
    lowered = formula.casefold()
    concepts = set()
    has_price = any(field in lowered for field in ("close", "open", "high", "low", "returns"))
    has_volume = "volume" in lowered or "adv" in lowered
    if has_price:
        concepts.update({"recent_return", "price_momentum", "price_reversal"})
    if has_volume:
        concepts.update({"volume_trend", "volume_abnormality"})
    if has_price and has_volume:
        concepts.add("price_volume_interaction")
    if any(token in lowered for token in ("stddev", "high - low", "high-low")):
        concepts.update({"volatility", "intraday_range"})
    if "cap" in lowered:
        concepts.add("market_cap")
    if "adv" in lowered:
        concepts.add("liquidity")
    return sorted(concepts or {"recent_return"})


def _implementation_status(formula: str, required_data: list[str]) -> str:
    available = {"open", "high", "low", "close", "volume", "mktcap"}
    if not set(required_data) <= available:
        return "requires_additional_data"
    functions = {
        name.casefold()
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", formula)
    }
    supported = {"rank", "ref", "mean", "std"}
    return "available_now" if functions <= supported else "requires_dsl_extension"


def _alpha158_concepts(name: str) -> list[str]:
    prefix = re.sub(r"\d+$", "", name)
    mapping = {
        "KMID": ["recent_return", "intraday_range"],
        "KLEN": ["intraday_range", "volatility"],
        "KUP": ["intraday_range", "price_reversal"],
        "KLOW": ["intraday_range", "price_reversal"],
        "KSFT": ["intraday_range", "price_momentum"],
        "OPEN": ["recent_return"], "HIGH": ["intraday_range"],
        "LOW": ["intraday_range"], "VWAP": ["price_volume_interaction"],
        "ROC": ["price_momentum", "recent_return"],
        "MA": ["price_momentum"], "STD": ["volatility"],
        "BETA": ["price_momentum"], "RSQR": ["price_momentum"],
        "RESI": ["price_reversal"], "MAX": ["price_momentum"],
        "MIN": ["price_reversal"], "QTLU": ["price_momentum"],
        "QTLD": ["price_reversal"], "RANK": ["price_momentum"],
        "RSV": ["price_momentum", "price_reversal"],
        "IMAX": ["price_momentum"], "IMIN": ["price_reversal"],
        "IMXD": ["price_momentum", "price_reversal"],
        "CORR": ["price_volume_interaction"],
        "CORD": ["price_volume_interaction"],
        "CNTP": ["price_momentum"], "CNTN": ["price_reversal"],
        "CNTD": ["price_momentum", "price_reversal"],
        "SUMP": ["price_momentum"], "SUMN": ["price_reversal"],
        "SUMD": ["price_momentum", "price_reversal"],
        "VMA": ["volume_trend"], "VSTD": ["volume_abnormality"],
        "WVMA": ["price_volume_interaction", "volatility"],
        "VSUMP": ["volume_trend"], "VSUMN": ["volume_trend"],
        "VSUMD": ["volume_trend"],
    }
    if prefix in mapping:
        return mapping[prefix]
    for candidate in sorted(mapping, key=len, reverse=True):
        if prefix.startswith(candidate):
            return mapping[candidate]
    return ["recent_return"]


def _alpha158_rationale(name: str) -> str:
    concepts = _alpha158_concepts(name)
    return (
        f"Qlib Alpha158 的 {_concept_label(concepts)}特征，"
        "用于提供标准化时间窗口、归一化方式和价格成交量结构参考。"
    )


def _concept_label(concepts: list[str]) -> str:
    labels = {
        "price_momentum": "价格动量",
        "price_reversal": "价格反转",
        "recent_return": "近期收益",
        "volume_trend": "成交量趋势",
        "volume_abnormality": "异常成交量",
        "price_volume_interaction": "量价交互",
        "volatility": "波动率",
        "intraday_range": "日内区间",
        "liquidity": "流动性",
        "market_cap": "规模",
    }
    return "、".join(labels.get(concept, concept) for concept in concepts[:3])
