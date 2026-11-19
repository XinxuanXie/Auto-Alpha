"""Real LLM, checker, and backtest tools for the V2 smoke workflow."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_workflow.factor_library import FactorKnowledgeBase
from factor_pipeline.compiler import compile_factor_spec
from factor_pipeline.spec import (
    BINARY_OPERATORS,
    ROLLING_BINARY_OPERATORS,
    ROLLING_OPERATORS,
    TERNARY_OPERATORS,
    WINDOWED_UNARY_OPERATORS,
    DataField,
    FactorSpec,
    FormulaNode,
    FormulaOperator,
    PortfolioRules,
)

from .contracts import (
    CheckIssue,
    DevelopedHypothesis,
    ExperimentVariantType,
    FactorExperimentSpec,
    FactorFormulaSpecV2,
    HypothesisCheckResult,
    InsightAnalysis,
    InsightDecision,
    MeasurementPlan,
    MeasurementProxy,
    PreflightResult,
    ResearchDirection,
    ResearchIdea,
    SplitMetrics,
    SplitName,
    passes_selection_gate,
)
from .real_backtest import RealBacktestConfig, RealV2BacktestEngine


SUPPORTED_CONCEPTS = frozenset(
    {
        "intraday_range",
        "liquidity",
        "liquidity_proxy",
        "market_cap",
        "price_momentum",
        "price_reversal",
        "price_volume_confirmation",
        "price_volume_interaction",
        "realized_volatility",
        "recent_return",
        "volatility",
        "volume_abnormality",
        "volume_trend",
    }
)
LEAKAGE_CONCEPTS = frozenset(
    {
        "forward_return",
        "future_price",
        "future_return",
        "future_volume",
        "label",
        "next_day_price",
        "target",
    }
)

# FormulaNode 的形状规则。Pydantic 的 model_json_schema() 表达不了
# spec.py 里 validate_shape() 的参数个数校验，所以必须用文字明确告诉
# LLM，否则它会把 field 当成二元算子的一个操作数（本次 Smoke 报错的
# 根因：{"op": "add", "field": "close", "args": []}）。
# 生成实验、修复实验和 _validated_json_completion 的重试提示共用同一份规则。
FORMULA_NODE_SHAPE_RULES = (
    "公式节点 FormulaNode 的形状规则（必须严格遵守，JSON Schema 里看不到这些规则）："
    "field 节点只能是 {\"op\": \"field\", \"field\": \"close\"}，不能带 args、value 或窗口参数；"
    "constant 节点只能是 {\"op\": \"constant\", \"value\": 1.0}，不能带 field 或 args；"
    "一元算子 lag、return、mean、std、sum、rolling_max、rolling_min、quantile、ts_rank、"
    "idx_max、idx_min、slope、rsquare、residual、zscore、rank、abs、log、negate 恰好有 1 个操作数，"
    "在 field 或单个 args 子节点之间二选一，绝不能同时填写 field 和 args；"
    "其中 lag、return、mean、std、sum、rolling_max、rolling_min、quantile、ts_rank、idx_max、"
    "idx_min、slope、rsquare、residual 必须设置 window，其余一元算子不能设置 window；"
    "二元算子 add、subtract、multiply、divide、greater_than、less_than、"
    "elementwise_max、elementwise_min 恰好有 2 个 args 子节点，field、value、window、"
    "min_periods、quantile、ddof 全部必须为空，field 不是操作数；"
    "correlation 也是二元算子，恰好有 2 个 args 且必须设置 window；"
    "correlation 不接受 ddof，std 是唯一能带 ddof 的算子；"
    "min_periods 只有 mean、std、sum、rolling_max、rolling_min、quantile、"
    "ts_rank、idx_max、idx_min、slope、rsquare、residual、correlation 这些滚动算子能带；"
    "return、lag 不接受 min_periods，只设置 window；field、constant、zscore、"
    "rank、abs、log、negate 不能带任何窗口参数；"
    "三元算子 where 恰好有 3 个 args 子节点，field、value、window 等全部必须为空；"
    "where 的第一个参数必须是布尔条件，也就是 greater_than 或 less_than 的结果，"
    "不能是数值序列——elementwise_max、mean 这类数值结果不能直接当条件用；"
    "想写“条件成立取 X、否则取 0”时必须写满 3 个 args：where(条件, X, 0.0)，"
    "不要只写 2 个 args；"
    "正确示例：add 是 {\"op\": \"add\", \"args\": [{\"op\": \"field\", \"field\": \"close\"}, "
    "{\"op\": \"constant\", \"value\": 1.0}]}；"
    "错误示例：{\"op\": \"add\", \"field\": \"close\", \"args\": []} 是非法的，"
    "因为 add 需要 2 个 args 且不能带 field。"
)

MAX_MEASUREMENT_PROXIES = 5

# 预测周期下限（日）。13ac24bc 重跑出来的 Reversal 基线选了 1 日持有：
# 每日全仓换手（turnover 1.0），10bps 成本一年吃掉约 50%，10 年毛收益
# 7.52 最后净收益是 -0.31——信号本身 IC 0.0143 并不差，是成本把它杀掉
# 了。1–3 日的周期在 Top 20 每日调仓 + 10bps 的成本模型下基本过不了
# 净收益门槛，等于白占一个实验名额，所以生成实验时就拦住。
MIN_PREDICTION_HORIZON_DAYS = 5

# Preflight 信号去重：两个实验在 2008–2010 样本上的横截面 rank 相关系数
# 绝对值达到该阈值、且共同观测数足够时，判为实质同一个信号。3d2eeec5 的教训：
# 不同方向、不同假设的公式塌缩成同一个信号——Reversal 基线 return(close,5)
# 与 Liquidity 基线 negate(return(close,5))，Momentum 基线与 Price-Volume
# 基线一字不差都是 return(close,20)；Preflight 原来只查公式文本完全相同的
# exact_duplicate，对取负、乘一个接近 1 的量比、where 套壳这类“换个写法、
# 同一个信号”的情况完全没反应。5 日收益 vs 10 日收益这类合理相关的信号
# 横截面相关通常在 0.7–0.85 之间。0.90 的依据是 4fa040cd 的实测分布：
# 漏网的那对 REVERSAL-02-01 vs PRICE_VOLUME-02-04 在去重样本上的 rank 相关
# 是 0.944，train IC 同为 0.0236——实质同一个信号，0.95 没拦住；同轮
# |corr| >= 0.80 的 33 对里，0.93–1.00 是一簇（同一信号的不同写法），
# 0.88 以下才开始出现相对独立的信号，0.90 正好落在断层上。
DUPLICATE_SIGNAL_MIN_CORRELATION = 0.90
DUPLICATE_SIGNAL_MIN_OVERLAP = 500

# Preflight 退化因子检查：一个因子在横截面上必须有区分度。0f45e744 的教训：
# REVERSAL-03-02 的公式里有一项 mean(where(volume<1, 1, 0), 20)，而本 panel
# 里 volume<1 的行是 0——这一项恒为 0，乘法链把整个因子变成恒 0，所有股票
# 每天都是一样的值，train 算不出 Rank IC、整轮崩掉。Preflight 原来只要求
# “至少算出 1 个有限值”，0 也是有限值，所以放行了。退化按有区分度的日期
# 数判：2008–2010 样本约 780 个交易日，正常因子几乎每天都有区分度；
# 只在不到 10 天有区分度的因子，基本是恒为常数或只在极少数日期有值。
DEGENERATE_SIGNAL_MIN_VARIED_DATES = 10

# 数据事实：本 panel 是 S&P 500 成分股，成交量从不为 0——全表 5,260,971 行里
# volume<1 的有 0 行。写进生成提示，免得 LLM 再编出以零成交量为条件的构造。
PANEL_ZERO_VOLUME_NOTE = (
    "数据事实：本 panel 是 S&P 500 成分股，成交量从不为 0——全表 5,260,971 行里 "
    "volume<1 的有 0 行；不要用 volume<1、volume==0 或零成交量天数占比这类构造，"
    "任何以零成交量为条件的项恒为 0，会让整个因子失去区分度。"
)

# Measurement Plan 的可实现性规则。13ac24bc 那轮的教训：Plan 一次列 7、8 个
# proxy，Preflight 又要求 full_mechanism 覆盖全部 proxy、公式用到每个 proxy 的
# 全部 required_fields，于是 repair 两头堵——简化就掉 proxy，全声明又做不出来
# （Reversal P4 的 required_fields 里有 open，但它自己的定义 CLV=(close-low)/
# (high-low) 根本用不到 open；Price-Volume P3 的定义要算“未来 K 日”收盘价，
# DSL 明文禁止用未来数据；Liquidity P1 要算全市场换手率中位数，DSL 里没有
# 全市场统计量）。所以 Plan 必须在生成时就只写能实现的东西，修 Plan 的责任
# 不能推给修公式。
MEASUREMENT_PROXY_RULES = (
    "Measurement Plan 的可实现性规则（Preflight 会逐条核对，违反的 Plan 修出的公式一定过不了 Preflight）："
    "proxy 最多 5 个，超过的一律是写多了，先砍掉最弱的；"
    "每个 proxy 必须只用给定字段（open/high/low/close/volume/mktcap）和 DSL 算子、"
    "只用截至当日（含当日）的历史数据就能算出来；observable_definition 里不准出现“未来”，"
    "不准用未来 K 日的数据，不准用全市场中位数/均值、买卖价差、真实订单流方向这类 DSL 算不出来的东西；"
    "required_fields 必须恰好是 observable_definition 里实际用到的字段，不多不少，"
    "定义里写了 close/low/high 就必须列 close/low/high，定义里没用 open 就不准列 open；"
    "分类型 proxy 必须给出可计算的判定规则，例如“收盘价高于过去 20 日最高价记为 1，否则记为 0”，"
    "不能只说“分为稳固与回吐两类”而不给判定方法；"
    "实在观测不到的机制不要硬编成 proxy，写进 unobservable_aspects；"
    "baseline_logic 与 full_mechanism_logic 只能引用上面已定义的 proxy，不能引入新机制。"
)

# 各研究方向的经典价格类异象文献。只列本数据能算的机制：OHLCV + 市值，
# 没有基本面、财报、分析师数据，所以价值/盈利/投资类文献不在此列。
# 作用（Patch 9 机制优先）：文献是佐证和对照，不是必选项——研究 agent 先把
# 机制说精确，再引用 0 到 2 篇文献说明是沿用、组合还是不同于经典构造；没有
# 单篇文献能概括的机制也可以输出，不要为了凑一篇而硬锚定。每个条目给文献、
# 机制、经典构造、窗口与已知坑；构造都只用 DSL 能表达的算子（return/mean/
# std/rolling_max/abs/log/divide/multiply/where 等）。
_DIRECTION_LITERATURE = {
    ResearchDirection.MOMENTUM: (
        "本方向经典文献："
        "1）Jegadeesh & Titman 1993 中期动量：过去 6–12 个月收益排序，跳过最近 1 个月，"
        "机制是信息扩散慢、投资者反应不足；坑：最近 1 个月是反转方向，不能混进动量窗口。"
        "2）Novy-Marx 2012：7–12 个月前的动量比近期动量更强，近期收益里含反转噪声。"
        "3）George & Hwang 2004，52 周最高价接近度：收盘价离过去 252 日最高价越近，"
        "未来收益越高，机制是投资者对历史高点的锚定；可表达为 close / rolling_max(high, 252)。"
    ),
    ResearchDirection.REVERSAL: (
        "本方向经典文献："
        "1）Jegadeesh 1990 / Lehmann 1990 短期反转：过去 1 周–1 个月收益反转，"
        "机制是流动性冲击后的价格过度反应与修复；坑：持有期太短会被换手成本吃掉，"
        "本流程持有不短于 5 日。"
        "2）De Bondt & Thaler 1985 长期反转：过去 3–5 年输家跑赢赢家，"
        "机制是对长期业绩的过度反应；可表达为长窗口 return(close, 756) 一类。"
        "3）注意：短期反转与中期动量方向相反，窗口选择决定方向，"
        "同一个过去收益不能既当动量又当反转。"
    ),
    ResearchDirection.LIQUIDITY_VOLATILITY: (
        "本方向经典文献："
        "1）Amihud 2002 非流动性溢价：|日收益| / 日成交额的均值越高，未来收益越高，"
        "机制是流动性风险补偿；可表达为 mean(abs(return(close, 1)) / volume, 20)。"
        "2）Ang, Hodrick, Xing & Zhang 2006 特质波动率异象：过去一个月特质波动率越高的股票，"
        "未来收益越低，方向是负的；可用 std(return(close, 1), 20) 近似。"
        "3）Bali, Cakici & Whitelaw 2011 MAX 彩票偏好：过去一个月最大单日收益越高，"
        "未来收益越低；可表达为 rolling_max(return(close, 1), 20)。"
        "4）Datar, Naik & Radcliffe 1998 换手率：低换手股票未来收益更高；"
        "可表达为 volume / mktcap 或 volume / mean(volume, 20)。"
    ),
    ResearchDirection.PRICE_VOLUME: (
        "本方向经典文献："
        "1）Lee & Swaminathan 2000 量价动量生命周期：成交量确认动量，低成交量动量更持久、"
        "高成交量动量更快反转；机制是投资者对信息的反应速度差异。"
        "2）Gervais, Kaniel & Mingelgrin 2001 高成交量溢价：成交量异常放大的股票短期后续收益更高；"
        "机制是注意力与可见性；可表达为 volume / mean(volume, 20) 的异常部分。"
        "3）Parkinson 1980 日内振幅波动率：(high - low) / close 反映日内波动与信息到达强度，"
        "振幅异常可作波动状态变量。"
        "4）量价背离：价格创新高但成交量萎缩，趋势可能衰竭——这是技术分析的经典说法，"
        "必须写成可证伪的假设，不能只当格言用。"
    ),
    ResearchDirection.OVERNIGHT_INTRADAY: (
        "本方向经典文献："
        "1）Lou, Polk & Skouras 2019 隔夜收益：美股大部分股权溢价在隔夜时段实现，"
        "隔夜动量与日内反转方向相反——隔夜收益延续、日内收益反转；机制是机构投资者"
        "在开盘前消化隔夜信息、散户在日内交易，二者行为不同。"
        "2）Heston, Korajczyk & Sadka 2010 日内季节性：同一股票在一天内固定时段的"
        "收益有持续性，信息到达与交易行为按时段聚集。"
        "3）DSL 表达：隔夜收益 = open / 前一日 close − 1，可写成 "
        "subtract(divide(field open, lag(field close, 1)), 1)；日内收益 = close / open − 1，"
        "可写成 subtract(divide(field close, field open), 1)。open 字段此前六轮几乎没被"
        "用作主信号，是本 panel 里未开发的信息。"
        "4）坑：隔夜与日内是日收益的两个负相关分量，假设必须说清是哪一段携带信号、"
        "为什么是那一段，不能把两段混成日收益再讲一遍动量或反转的故事。"
    ),
}

# 文献锚定改机制优先（Patch 9，用户决定）：机制先说精确，文献是佐证不是
# 必选项。Patch 7 的教训反过来了——硬要求每个假设锚定一篇文献，治好了想法层
# 塌成通用技术指标，但假设容易变成经典构造的轻微改写（0784f5f3 的 Amihud 假设
# 就是例子：点了 Amihud 2002，完整机制与基线的 rank 相关仍有 0.95）。所以
# 引用放宽到 0 到 2 篇：沿用、组合或不同于文献都可以，没有单篇文献能概括的
# 机制也可以输出；守住的底线是机制必须精确、可证伪，且审查仍要查“是否只是
# 已有因子的重新表述”。
MECHANISM_FIRST_LITERATURE_RULES = (
    "机制优先，文献作佐证：先把核心机制说精确——谁承担什么风险、什么摩擦或"
    "谁的行为导致横截面收益差异，以及在什么条件下成立或失效；经典文献可以引用"
    " 0 到 2 篇，说明是沿用、组合还是不同于该文献的经典构造，引用了就要说清区别；"
    "没有单篇文献能概括的机制也可以输出，但不要为了凑一篇而硬锚定，也不要编造"
    "既无精确机制、又无文献或因子库依据的洞察。"
)

# 生成实验时的防坍缩规则。3d2eeec5 与 4fa040cd 两轮的教训：40 个实验实际只有
# 5、6 个不同信号——非 baseline 实验的主信号经常就是过去收益本身，ablation 与
# full 退化成 baseline 的改写（取负、乘量比、where 套壳），最后全靠 Preflight
# 去重来拦。生成时就要求主信号不同，比事后去重要便宜。
EXPERIMENT_DIVERSITY_RULES = (
    "每个实验必须有明确的主信号：baseline 的主信号就是它检验的机制变量；"
    "ablation、full_mechanism、robustness 的主信号必须与 baseline 不同——"
    "只取负、只乘一个接近 1 的量比、只套 where 条件而不改变主变量，都不算不同的实验，"
    "会被 Preflight 按重复信号拒掉。"
    "非 baseline 实验的公式里，成交量、波动率、振幅、流动性变量不能只是装饰："
    "如果去掉这些变量后公式退化成过去收益本身，这个实验就是不合格的。"
)

# 换手成本意识（Patch 11，manager 诊断）。0784f5f3 与 0f45e744 的教训：
# 2007–08 块里完整机制、P2、P3 的 IC 全是正的，净收益却全是负的
# （-0.384 / -0.381 / -0.358），基线 -0.569——信号本身没错，是换手把
# 收益吃掉了；0f45e744 里 1 日持有的反转基线换手率 1.0，10 年毛收益 7.52
# 被 10bps 成本吃到净 -0.31。生成时就要信号慢下来，而不是事后靠门槛筛。
TURNOVER_RULES = (
    "换手成本意识：本流程按 10bps 收费、持有 5–60 日；信号必须慢——"
    "调节项优先用 20 日窗口，少用 5 日窗口；不要构造每天翻转的信号，"
    "不要用 1–2 日的快速均线差作主信号；一个 IC 为正但每天换手的因子，"
    "净收益一定是负的。"
)

# repair 里 duplicate_signal 的修法。4fa040cd 的教训：6 个被判重复的实验修完后
# 仍是重复（rank corr 0.98–1.00），因为提示只说“必须具有实质差异”，LLM 不知道
# 具体该改什么，只能改窗口、取负、乘量比——这些恰好都不算差异。修重复必须换主信号。
DUPLICATE_REPAIR_RULES = (
    "若错误是 duplicate_signal，必须换主信号族：先看 measurement_plan 里有哪些 proxy"
    "还没被对照实验用过，改用其中一个作主信号；被拒公式的主信号如果是过去收益，"
    "改用成交量、换手率、波动率或日内振幅为主信号；主信号如果是成交量比，"
    "改用波动率或价格位置。只改窗口、只取负、只乘量比、只加 where 包装都不算实质差异，"
    "修完仍会被判重复。checker_issues 里已给出重复对照实验与相关系数，"
    "修完的公式与它的横截面 rank 相关必须低于 0.90。"
)

_FIELD_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(open|high|low|close|volume|mktcap)(?![A-Za-z])",
    re.IGNORECASE,
)
# Plan 是中文写的，字段常以中文出现：\b 在中文字符和英文字母之间不算边界，
# 所以不能用 \b；上面用 ASCII 前后瞻断词。中文同义词也要算进去，否则
# “日收益”这种写法会被误判为没用 close（13ac24bc 的 Reversal P6、Liquidity P7
# 都是这样被误伤的）。
_FIELD_SYNONYMS = {
    "开盘价": {"open"},
    "最高价": {"high"},
    "最低价": {"low"},
    "收盘价": {"close"},
    "成交量": {"volume"},
    "成交额": {"close", "volume"},
    "换手率": {"volume", "mktcap"},
    "市值": {"mktcap"},
    "日收益": {"close"},
    "收益率": {"close"},
}


def _mentioned_fields(definition: str) -> set[str]:
    mentioned = {token.lower() for token in _FIELD_TOKEN_RE.findall(definition)}
    for phrase, fields in _FIELD_SYNONYMS.items():
        if phrase in definition:
            mentioned |= fields
    return mentioned
_PROXY_ID_RE = re.compile(r"(?<![A-Za-z])P[1-9][0-9]*(?![0-9])")
_FUTURE_WORDS = ("未来",)
_MARKET_WIDE_PHRASES = ("全市场", "市场中位数", "市场均值")


def _validate_measurement_plan(plan) -> list[str]:
    """Deterministic implementability checks for a MeasurementPlan draft.

    Returns one violation string per failed rule, in plain language the
    repair prompt can act on. This is deliberately a heuristic, not a
    semantic proof: the field-consistency check only runs when the
    observable_definition names at least one English field token, because
    a purely Chinese definition cannot be checked this way.
    """
    violations: list[str] = []
    proxies = list(plan.proxies)
    if len(proxies) > MAX_MEASUREMENT_PROXIES:
        violations.append(
            f"proxy 数量为 {len(proxies)}，超过上限 {MAX_MEASUREMENT_PROXIES} 个"
        )
    seen: set[str] = set()
    for proxy in proxies:
        if proxy.proxy_id in seen:
            violations.append(f"proxy_id {proxy.proxy_id} 重复")
        seen.add(proxy.proxy_id)
    required_by_id = {
        proxy.proxy_id: {field.value for field in proxy.required_fields}
        for proxy in proxies
    }
    definitions_by_id = {
        proxy.proxy_id: proxy.observable_definition or "" for proxy in proxies
    }
    for proxy in proxies:
        definition = proxy.observable_definition or ""
        if any(word in definition for word in _FUTURE_WORDS):
            violations.append(
                f"{proxy.proxy_id} 的 observable_definition 出现“未来”，用了未来数据"
            )
        if any(phrase in definition for phrase in _MARKET_WIDE_PHRASES):
            violations.append(
                f"{proxy.proxy_id} 的 observable_definition 用了 DSL 算不出来的全市场/市场统计量"
            )
        # 定义里常写“结合 P1 和 P2”而不重写字段：这类组合 proxy 继承被引用
        # proxy 的 required_fields（传递闭包），否则 P4 这种“P1+P2 的组合”
        # 会被误判为 required_fields 写多了。
        mentioned = _mentioned_fields(definition)
        referenced = set(_PROXY_ID_RE.findall(definition)) - {proxy.proxy_id}
        queue = list(referenced)
        expanded_refs: set[str] = set()
        while queue:
            ref_id = queue.pop()
            if ref_id in expanded_refs or ref_id not in required_by_id:
                continue
            expanded_refs.add(ref_id)
            mentioned |= required_by_id[ref_id]
            queue.extend(
                set(_PROXY_ID_RE.findall(definitions_by_id[ref_id]))
                - expanded_refs
            )
        if mentioned:
            required = required_by_id[proxy.proxy_id]
            extra = sorted(required - mentioned)
            missing = sorted(mentioned - required)
            if extra:
                violations.append(
                    f"{proxy.proxy_id} 的 required_fields 包含定义中未提到的字段："
                    + "、".join(extra)
                )
            if missing:
                violations.append(
                    f"{proxy.proxy_id} 的定义中提到了字段却未列入 required_fields："
                    + "、".join(missing)
                )
    for field_name in ("baseline_logic", "full_mechanism_logic"):
        logic = getattr(plan, field_name, "") or ""
        for proxy_id in sorted(set(_PROXY_ID_RE.findall(logic))):
            if proxy_id not in seen:
                violations.append(f"{field_name} 引用了未定义的 {proxy_id}")
    return violations


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _InsightReviewDraft(_StrictModel):
    decision: InsightDecision
    supporting_arguments: list[str] = Field(min_length=1, max_length=12)
    objections: list[str] = Field(min_length=1, max_length=12)
    alternative_explanations: list[str] = Field(min_length=1, max_length=12)
    final_research_idea: str = Field(min_length=20, max_length=10_000)
    analysis_text: str = Field(min_length=20, max_length=30_000)


class _HypothesisDraft(_StrictModel):
    title: str = Field(min_length=1, max_length=240)
    claim: str = Field(min_length=10, max_length=4_000)
    economic_rationale: str = Field(min_length=10, max_length=4_000)
    core_mechanism: str = Field(min_length=10, max_length=4_000)
    counterargument: str = Field(min_length=10, max_length=4_000)
    falsification_criteria: list[str] = Field(min_length=1, max_length=12)
    distinction_from_siblings: str = Field(min_length=10, max_length=2_000)
    prediction_horizons_days: list[int] = Field(min_length=1, max_length=4)
    required_concepts: list[str] = Field(min_length=1, max_length=12)
    market_conditions: list[str] = Field(default_factory=list, max_length=12)
    failure_conditions: list[str] = Field(default_factory=list, max_length=12)


class _HypothesisBatch(_StrictModel):
    hypotheses: list[_HypothesisDraft] = Field(min_length=1, max_length=4)


class _MeasurementPlanDraft(_StrictModel):
    mechanism_summary: str = Field(min_length=10, max_length=4_000)
    proxies: list[MeasurementProxy] = Field(min_length=1, max_length=8)
    baseline_logic: str = Field(min_length=10, max_length=2_000)
    full_mechanism_logic: str = Field(min_length=10, max_length=4_000)
    unobservable_aspects: list[str] = Field(default_factory=list, max_length=12)


class _ExperimentDraft(_StrictModel):
    name: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2_000)
    variation_reason: str = Field(min_length=10, max_length=2_000)
    variant_type: ExperimentVariantType
    represented_proxy_ids: list[str] = Field(min_length=1, max_length=8)
    formula: FormulaNode
    prediction_horizon_days: int = Field(gt=0)


class _ExperimentBatch(_StrictModel):
    experiments: list[_ExperimentDraft] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_required_variants(self):
        variants = {item.variant_type for item in self.experiments}
        required = {
            ExperimentVariantType.BASELINE,
            ExperimentVariantType.FULL_MECHANISM,
        }
        if not required.issubset(variants):
            raise ValueError("experiment batch requires baseline and full_mechanism")
        return self


class _ExperimentRepairDraft(_StrictModel):
    name: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=2_000)
    variation_reason: str = Field(min_length=10, max_length=2_000)
    formula: FormulaNode


class _ExperimentRepairBatch(_StrictModel):
    experiments: list[_ExperimentRepairDraft] = Field(min_length=1, max_length=1)


def _derived_proxy_ids(
    experiment: FactorExperimentSpec, plan: MeasurementPlan
) -> set[str]:
    """Proxies whose required fields the formula actually uses.

    This is what preflight judges coverage by: the LLM's declared
    represented_proxy_ids is a claim, this is the evidence. 3d2eeec5
    showed the claim is often wrong in the conservative direction — a
    full declares P1-P3 while its formula contains return(close, 5) for
    P4 and std(1-day returns, 20) for P5.
    """
    used_fields = experiment.factor.formula.referenced_fields()
    return {
        proxy.proxy_id
        for proxy in plan.proxies
        if set(proxy.required_fields).issubset(used_fields)
    }


def _normalize_represented_proxies(
    experiments: list[FactorExperimentSpec],
    measurement_plans: list[MeasurementPlan],
) -> None:
    """Rewrite every variant's declaration to match its formula.

    The declared list states what the formula actually implements, not
    what the LLM intended: ec610c9d lost 9 of 13 preflight rejections to
    measurement_proxy_not_implemented, and all 9 were repaired -R1
    ablations/robustness whose formulas no longer implement the proxies
    the (inherited) declaration claims — e.g. a volume-only formula
    still declaring [P1..P4], or an overnight ablation whose repair
    dropped the open field while still declaring the gap proxy P1.
    Repair rewrites the formula but copies the old declaration, so the
    declaration is stale by construction; deriving it from the formula's
    referenced fields makes the label honest and stops the pipeline
    throwing away genuine experiments over bookkeeping. A full mechanism
    still needs the derived set to equal the whole plan — that coverage
    check lives in preflight and is untouched. Only normalizes when the
    derived set is non-empty; an empty derivation means the formula uses
    none of the plan's fields, and preflight will reject it with a
    specific message instead.
    """
    plan_by_id = {plan.plan_id: plan for plan in measurement_plans}
    for experiment in experiments:
        plan = plan_by_id.get(experiment.measurement_plan_id)
        if plan is None:
            continue
        derived = _derived_proxy_ids(experiment, plan)
        if derived:
            experiment.represented_proxy_ids = [
                proxy.proxy_id for proxy in plan.proxies if proxy.proxy_id in derived
            ]


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, int]:
    """Pearson correlation of two cross-sectional rank vectors.

    Both arrays are aligned on the same (date, permno) grid with NaN where
    the formula produced no finite value. Returns (correlation, overlap);
    correlation is None when the shared observations are too few, either
    side is (near-)constant, or the result is not finite.
    """
    mask = ~(np.isnan(left) | np.isnan(right))
    overlap = int(mask.sum())
    if overlap < DUPLICATE_SIGNAL_MIN_OVERLAP:
        return None, overlap
    x = left[mask]
    y = right[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None, overlap
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return None, overlap
    return corr, overlap


def _is_same_hypothesis_full_vs_baseline(
    earlier: FactorExperimentSpec, later: FactorExperimentSpec
) -> bool:
    """True when the pair is a full mechanism against its OWN hypothesis's
    baseline. A full is defined as its baseline plus modulation, so a high
    rank correlation with its own baseline is the design, not duplication —
    and the backtest, not the correlation, must decide whether the
    modulation adds value. Evidence: the hand-built Amihud fulls (M1/M2)
    correlated 0.95-0.96 with their baseline and beat it out of sample
    (validation 0.026/0.030 vs baseline 0.019, test 0.030 p~0.01), while
    the fulls that survived dedup (corr < 0.90) were violent modulations
    that scrambled the ranking and lost to their baselines in every run
    (0d53bd4b: 4 of 6 fulls were built on their baseline yet lost, the
    worst 0.0339 -> 0.0021). The rule as written selects against gentle
    modulations. The 0.90 threshold is unchanged for every other pair,
    including a full against another hypothesis's baseline.
    """
    if earlier.hypothesis_id != later.hypothesis_id:
        return False
    pair = {earlier.variant_type, later.variant_type}
    return pair == {
        ExperimentVariantType.BASELINE,
        ExperimentVariantType.FULL_MECHANISM,
    }


def _count_varied_cross_section_dates(
    values: np.ndarray, dates: Any
) -> int:
    """Count dates where the factor distinguishes at least two stocks.

    ``values`` are cross-sectional ranks aligned on the shared (date,
    permno) grid, NaN where the formula produced no finite value. A
    constant factor gives every stock the same average rank on a date,
    so that date does not count. Returns 0 when no finite value exists.
    """
    finite = np.isfinite(values)
    if not finite.any():
        return 0
    _, inverse = np.unique(np.asarray(dates), return_inverse=True)
    varied = 0
    for group in range(int(inverse.max()) + 1):
        group_values = values[(inverse == group) & finite]
        if len(np.unique(group_values)) > 1:
            varied += 1
    return varied


class RealV2Tools:
    def __init__(
        self,
        panel_path: str | Path,
        membership_path: str | Path,
        *,
        min_hypotheses_per_idea: int = 1,
        max_hypotheses_per_idea: int = 1,
        max_experiments_per_hypothesis: int = 2,
        deepseek_model: str | None = None,
        openai_model: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not 1 <= min_hypotheses_per_idea <= max_hypotheses_per_idea <= 4:
            raise ValueError(
                "hypotheses per idea must satisfy 1 <= minimum <= maximum <= 4"
            )
        if not 2 <= max_experiments_per_hypothesis <= 5:
            raise ValueError("max_experiments_per_hypothesis must be between 2 and 5")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.min_hypotheses_per_idea = min_hypotheses_per_idea
        self.max_hypotheses_per_idea = max_hypotheses_per_idea
        self.max_experiments_per_hypothesis = max_experiments_per_hypothesis
        self.deepseek_model = deepseek_model or os.getenv(
            "DEEPSEEK_RESEARCH_MODEL",
            os.getenv("DEEPSEEK_REFINEMENT_MODEL", "deepseek-chat"),
        )
        self.openai_model = openai_model or os.getenv(
            "OPENAI_MODEL", "gpt-5.6-luna"
        )
        self.deepseek = OpenAI(
            api_key=deepseek_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=180.0,
            max_retries=2,
        )
        self.school_openai = OpenAI(
            api_key=openai_key,
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://hkust.azure-api.net/hkust-genai/v1",
            ),
            default_headers={"api-key": openai_key},
            timeout=180.0,
            max_retries=2,
        )
        self.panel_path = Path(panel_path)
        self.membership_path = Path(membership_path)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.factor_library = FactorKnowledgeBase()
        self.backtest_engine = RealV2BacktestEngine(
            RealBacktestConfig(
                panel_path=self.panel_path,
                membership_path=self.membership_path,
            )
        )

    def load_context(self, state: dict[str, Any]) -> dict[str, Any]:
        del state
        entries = [
            entry.model_dump(mode="json") for entry in self.factor_library.entries
        ]
        return {
            "available_fields": [field.value for field in DataField],
            "data_frequency": "daily",
            "universe": "historical point-in-time S&P 500 constituents",
            "factor_library_count": len(entries),
            "factor_library": entries,
            "factor_backtest_metrics_included": False,
            "web_search_enabled": False,
        }

    def generate_research_ideas(
        self, state: dict[str, Any], context: dict[str, Any]
    ) -> list[ResearchIdea]:
        prefix = str(state["run_id"])[:8].upper()
        ideas = []
        for direction in ResearchDirection:
            system_prompt = (
                f"你是专注于 {direction.value} 的量化金融 Research Agent。"
                f"研究范围：{_direction_focus(direction)}。"
                f"本方向的经典文献：{_direction_literature(direction)}"
                f"{MECHANISM_FIRST_LITERATURE_RULES}"
                "提出一个抽象但有金融意义的研究洞察。只讨论市场现象、金融或行为机制、"
                "横截面收益差异的形成路径及可能成立或失效的市场条件。"
                "不要生成公式、具体窗口、DSL 算子、FactorSpec 或回测结论。"
                "可以参考完整因子定义库中的机制，但不要直接复制已有公式。"
                "只返回一段聚焦于单一核心洞察的中文文本。"
            )
            user_payload = {
                "task": "提出本方向一个新的抽象研究洞察。",
                "available_data": {
                    "fields": context["available_fields"],
                    "frequency": context["data_frequency"],
                    "universe": context["universe"],
                },
                "factor_library": context["factor_library"],
                "historical_experience": [],
            }
            text = self._deepseek_text(system_prompt, user_payload)
            ideas.append(
                ResearchIdea(
                    idea_id=f"I-{prefix}-{direction.value.upper()}",
                    run_id=str(state["run_id"]),
                    direction=direction,
                    idea_text=text,
                    model=self.deepseek_model,
                    prompt_version="v2.1-critical-development",
                )
            )
        return ideas

    def analyze_insights(
        self, state: dict[str, Any], ideas: list[ResearchIdea]
    ) -> list[InsightAnalysis]:
        del state
        analyses = []
        for idea in ideas:
            review = self._school_json(
                (
                    "你是 Hypothesis Development Agent 的批判性审查与重构阶段。"
                    "不要顺着原始 Idea 扩写；必须同时提出支持理由、反对理由和替代解释，"
                    "检查机制是否混淆相关性与因果、是否只是已有因子的重新表述"
                    "（对照本方向经典文献）、是否具有"
                    "清晰的可证伪边界。再多查两点：第一，这个机制的收益是否可能只是"
                    "日历或月份效应（例如 12 月避税卖盘、1 月反弹）——如果机制只在"
                    "特定月份或特定市场状态下成立，必须写明，不能当成全年稳定的溢价；"
                    "第二，机制隐含的交易频率与换手成本是否匹配——每天翻转的信号在"
                    "10bps 成本下活不下来，机制必须慢到能持有 5 日以上。你可以 ACCEPT、REVISE 或 REPLACE。REPLACE 时必须"
                    "在同一研究方向提出更合理的新 Idea；REVISE 时必须实质性修正机制。"
                    "本阶段保持纯金融分析，不讨论数据字段、DSL、公式或实现。"
                    "最终给出唯一的 final_research_idea，并返回符合 output_schema 的 JSON。"
                ),
                {
                    "research_direction": idea.direction.value,
                    "research_idea": idea.idea_text,
                    "direction_literature": _direction_literature(idea.direction),
                    "decision_definitions": {
                        "accept": "Idea is already coherent after adversarial review.",
                        "revise": "Keep the core intuition but materially rewrite it.",
                        "replace": "Reject it and create a better idea in the same direction.",
                    },
                    "output_schema": _InsightReviewDraft.model_json_schema(),
                },
                _InsightReviewDraft,
            )
            analyses.append(
                InsightAnalysis(
                    analysis_id=idea.idea_id.replace("I-", "A-", 1),
                    idea_id=idea.idea_id,
                    **review.model_dump(),
                    model=self.openai_model,
                    prompt_version="v2.1-critical-development",
                )
            )
        return analyses

    def expand_hypotheses(
        self,
        state: dict[str, Any],
        ideas: list[ResearchIdea],
        analyses: list[InsightAnalysis],
    ) -> list[DevelopedHypothesis]:
        del state
        analysis_map = {item.idea_id: item for item in analyses}
        hypotheses = []
        batch_model = _HypothesisBatch
        for idea in ideas:
            analysis = analysis_map.get(idea.idea_id)
            if analysis is None:
                continue
            batch = self._school_json(
                (
                    "你是 Hypothesis Development Agent 的 Hypothesis Expansion 阶段。"
                    "只根据批判性审查后的 final_research_idea 生成明确、可证伪的横截面假设。"
                    "每个假设必须代表不同的经济机制或相互竞争的可证伪解释。"
                    f"{MECHANISM_FIRST_LITERATURE_RULES}"
                    "禁止仅通过更换预测周期、市场状态、流动性分组或公司规模分组来凑数量。"
                    "如果只有一个高质量机制，就只输出一个假设。每个假设必须说明核心机制、"
                    "最强反驳、明确的证伪标准，以及它与同批其他假设的实质区别。"
                    "不要生成公式。预测周期可以是任意正整数；不声明预测方向，方向由训练集确定。"
                    "required_concepts 只能从给定 supported_concepts 中选择。"
                    f"必须生成 {self.min_hypotheses_per_idea} 至 "
                    f"{self.max_hypotheses_per_idea} 个假设，具体数量由研究价值决定。"
                    "只返回符合 output_schema 的 JSON。"
                ),
                {
                    "research_direction": idea.direction.value,
                    "research_idea": idea.idea_text,
                    "critical_review": analysis.model_dump(mode="json"),
                    "final_research_idea": analysis.final_research_idea,
                    "direction_literature": _direction_literature(idea.direction),
                    "supported_concepts": sorted(SUPPORTED_CONCEPTS),
                    "output_schema": batch_model.model_json_schema(),
                },
                batch_model,
            )
            for index, draft in enumerate(
                batch.hypotheses[: self.max_hypotheses_per_idea], start=1
            ):
                hypotheses.append(
                    DevelopedHypothesis(
                        hypothesis_id=f"H-{idea.idea_id[2:]}-{index:02d}",
                        idea_id=idea.idea_id,
                        direction=idea.direction,
                        **draft.model_dump(),
                    )
                )
        return hypotheses

    def check_hypotheses(
        self,
        hypotheses: list[DevelopedHypothesis],
        generation_round: int,
    ) -> list[HypothesisCheckResult]:
        signatures: Counter[str] = Counter(
            _hypothesis_signature(item) for item in hypotheses
        )
        results = []
        for hypothesis in hypotheses:
            issues = []
            unsupported = sorted(
                set(hypothesis.required_concepts) - SUPPORTED_CONCEPTS
            )
            for concept in unsupported:
                issues.append(
                    CheckIssue(
                        code="unsupported_concept",
                        message=f"Unsupported concept: {concept}",
                        field="required_concepts",
                    )
                )
            leaked = sorted(
                set(hypothesis.required_concepts) & LEAKAGE_CONCEPTS
            )
            for concept in leaked:
                issues.append(
                    CheckIssue(
                        code="future_information_leakage",
                        message=f"Leakage concept: {concept}",
                        field="required_concepts",
                    )
                )
            if signatures[_hypothesis_signature(hypothesis)] > 1:
                issues.append(
                    CheckIssue(
                        code="exact_duplicate_hypothesis",
                        message="An identical normalized hypothesis exists in this run.",
                    )
                )
            results.append(
                HypothesisCheckResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    accepted=not issues,
                    issues=issues,
                    generation_round=generation_round,
                )
            )
        return results

    def repair_hypotheses(
        self,
        rejected: list[DevelopedHypothesis],
        issues_by_id: dict[str, list[CheckIssue]],
    ) -> list[DevelopedHypothesis]:
        repaired = []
        for hypothesis in rejected:
            batch = self._school_json(
                (
                    "修正一个未通过确定性检查的假设。保留原经济机制，但 required_concepts "
                    "只能使用 supported_concepts，且不得复制输入中的完整 claim。"
                    "恰好返回一个假设 JSON。"
                ),
                {
                    "rejected_hypothesis": hypothesis.model_dump(mode="json"),
                    "checker_issues": [
                        item.model_dump(mode="json")
                        for item in issues_by_id.get(hypothesis.hypothesis_id, [])
                    ],
                    "supported_concepts": sorted(SUPPORTED_CONCEPTS),
                    "output_schema": _HypothesisBatch.model_json_schema(),
                },
                _HypothesisBatch,
            )
            draft = batch.hypotheses[0]
            repaired.append(
                DevelopedHypothesis(
                    hypothesis_id=f"{hypothesis.hypothesis_id}-R1",
                    idea_id=hypothesis.idea_id,
                    direction=hypothesis.direction,
                    **draft.model_dump(),
                )
            )
        return repaired

    def design_measurement_plans(
        self, hypotheses: list[DevelopedHypothesis]
    ) -> list[MeasurementPlan]:
        plans = []
        for hypothesis in hypotheses:
            system_prompt = (
                "你是 Measurement Planning Agent。先不要生成公式。把纯金融假设拆成"
                "可由当前日频字段观测的机制组件和代理变量。必须明确哪些机制无法被"
                "OHLCV/市值忠实观测，并为简单 baseline 与完整机制分别给出测量逻辑。"
                "代理必须使用给定字段，proxy_id 使用 P1、P2 等稳定编号。"
                "不要声称代理等于真实机制；必须列出每个代理的局限。"
                f"{MEASUREMENT_PROXY_RULES}"
                "只返回符合 output_schema 的 JSON。"
            )
            draft = self._deepseek_json(
                system_prompt,
                {
                    "hypothesis": hypothesis.model_dump(mode="json"),
                    "available_fields": [field.value for field in DataField],
                    "output_schema": _MeasurementPlanDraft.model_json_schema(),
                },
                _MeasurementPlanDraft,
            )
            violations = _validate_measurement_plan(draft)
            if violations:
                # 只修一轮 Plan：修 Plan 的责任在生成 Plan 这一步，不能推给
                # 后面修公式。修 Plan 本身失败也不让整轮崩掉，留原稿继续走。
                try:
                    repaired_draft = self._deepseek_json(
                        system_prompt,
                        {
                            "hypothesis": hypothesis.model_dump(mode="json"),
                            "available_fields": [
                                field.value for field in DataField
                            ],
                            "output_schema": _MeasurementPlanDraft.model_json_schema(),
                            "previous_draft": draft.model_dump(mode="json"),
                            "plan_violations": violations,
                            "correction_instruction": (
                                "上一版 Measurement Plan 违反了可实现性规则，"
                                "必须逐条修正后返回完整 Plan：删掉不可实现的 proxy、"
                                "把 required_fields 改成定义里实际用到的字段、"
                                "把未来数据和全市场统计量改成只用历史数据能算的定义。"
                                "不要只改文字不改结构。"
                            ),
                        },
                        _MeasurementPlanDraft,
                    )
                except ValueError:
                    repaired_draft = None
                if repaired_draft is not None:
                    repaired_violations = _validate_measurement_plan(repaired_draft)
                    if len(repaired_violations) <= len(violations):
                        draft, violations = repaired_draft, repaired_violations
            # 修完仍有问题也继续走，但要让它可观测：记进 unobservable_aspects，
            # 下游 Preflight 拦公式时能看到 Plan 本身就有问题，而不是只怪公式。
            unobservable = list(draft.unobservable_aspects)
            for violation in violations:
                note = f"PLAN_CHECK: {violation}"
                if len(unobservable) < 12 and note not in unobservable:
                    unobservable.append(note)
            data = draft.model_dump()
            data["unobservable_aspects"] = unobservable
            plans.append(
                MeasurementPlan(
                    plan_id=f"MP-{hypothesis.hypothesis_id[2:]}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    **data,
                )
            )
        return plans

    def generate_experiments(
        self,
        hypotheses: list[DevelopedHypothesis],
        measurement_plans: list[MeasurementPlan],
    ) -> list[FactorExperimentSpec]:
        plan_by_hypothesis = {
            plan.hypothesis_id: plan for plan in measurement_plans
        }
        experiments = []
        for hypothesis in hypotheses:
            plan = plan_by_hypothesis.get(hypothesis.hypothesis_id)
            if plan is None:
                continue
            batch = self._read_cached_experiment_batch(hypothesis.hypothesis_id)
            if batch is None:
                batch = self._deepseek_json(
                    (
                        "你是 FactorSpec Agent。把一个结构化假设转换成可执行的结构化因子实验。"
                        "必须严格遵循 measurement_plan，不得用一个方便的技术指标替代整个机制。"
                        "只能使用给定字段和 DSL 算子，不得生成 Python。"
                        "每个实验必须对应一个预测周期，且周期来自 hypothesis。"
                        f"预测周期不得短于 {MIN_PREDICTION_HORIZON_DAYS} 日："
                        "1 日持有意味着每日全仓换手，10bps 成本一年吃掉约 50%，"
                        "再强的信号也过不了净收益门槛；只从 hypothesis 里不短于"
                        f"{MIN_PREDICTION_HORIZON_DAYS} 日的周期中选，若 hypothesis"
                        "里没有，取其中最长的。"
                        "full_mechanism 必须以 baseline 公式为骨架，用机制项做乘法"
                        "调节（例如 baseline × 机制调节项），或者用 where 做条件筛选"
                        "（机制条件成立时取 baseline 信号，否则取 0），两种结构选对机制"
                        "更合理的那种，不要两种都拼在一个公式里；不要把全部 proxy"
                        "的项简单加总：加总会稀释核心信号，13ac24bc 里 Momentum"
                        "完整机制的 IC（0.0078）就低于基线（0.0102）。"
                        f"{PANEL_ZERO_VOLUME_NOTE}"
                        f"{EXPERIMENT_DIVERSITY_RULES}"
                        f"{TURNOVER_RULES}"
                        f"{FORMULA_NODE_SHAPE_RULES}"
                        "参数约束：return 与 lag 只设置 window；mean/std/sum/rolling_max/"
                        "rolling_min/quantile/ts_rank 等滚动算子才可设置 min_periods；"
                        "field 与 constant 不设置 window 或 min_periods。"
                        "本 DSL 的 residual、slope、rsquare 都是单序列相对时间趋势的滚动统计，"
                        "只接收一个输入，不是两个变量之间的回归。"
                        "必须生成一个 baseline 和一个 full_mechanism；其余只能是 ablation 或"
                        "robustness。禁止仅更换预测周期来制造新实验。每个实验必须列出实际"
                        "表达的 proxy_id。"
                        f"总数为 2 至 {self.max_experiments_per_hypothesis} 个实验。"
                        "只返回一个符合 output_schema 的 JSON 对象。"
                    ),
                    {
                        "hypothesis": hypothesis.model_dump(mode="json"),
                        "measurement_plan": plan.model_dump(mode="json"),
                        "available_fields": [field.value for field in DataField],
                        "available_operators": [item.value for item in FormulaOperator],
                        "formula_node_schema": FormulaNode.model_json_schema(),
                        "output_schema": _ExperimentBatch.model_json_schema(),
                    },
                    _ExperimentBatch,
                )
                self._write_cached_experiment_batch(
                    hypothesis.hypothesis_id, batch
                )
            accepted_drafts = batch.experiments[
                : self.max_experiments_per_hypothesis
            ]
            for index, draft in enumerate(accepted_drafts, start=1):
                horizon = draft.prediction_horizon_days
                if horizon not in hypothesis.prediction_horizons_days:
                    horizon = hypothesis.prediction_horizons_days[
                        (index - 1) % len(hypothesis.prediction_horizons_days)
                    ]
                # 确定性下限：prompt 只是建议，LLM 仍可能选 1 日；这里强制执行。
                if horizon < MIN_PREDICTION_HORIZON_DAYS:
                    eligible = [
                        item
                        for item in hypothesis.prediction_horizons_days
                        if item >= MIN_PREDICTION_HORIZON_DAYS
                    ]
                    horizon = (
                        eligible[0]
                        if eligible
                        else max(hypothesis.prediction_horizons_days)
                    )
                formula = draft.formula
                factor_id = f"F-{hypothesis.hypothesis_id[2:]}-{index:02d}"
                factor = FactorFormulaSpecV2(
                    factor_id=factor_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    name=draft.name,
                    description=draft.description,
                    formula=formula,
                    required_fields=sorted(
                        formula.referenced_fields(), key=lambda item: item.value
                    ),
                    lookback_days=formula.required_history(),
                )
                experiments.append(
                    FactorExperimentSpec(
                        experiment_id=f"E-{factor_id[2:]}",
                        hypothesis_id=hypothesis.hypothesis_id,
                        variation_reason=draft.variation_reason,
                        measurement_plan_id=plan.plan_id,
                        variant_type=draft.variant_type,
                        represented_proxy_ids=draft.represented_proxy_ids,
                        factor=factor,
                        prediction_horizon_days=horizon,
                    )
                )
        # Full-mechanism declarations come from the LLM and are often
        # incomplete even when the formula covers every proxy; rewrite
        # them from the formulas so the stored record says what the
        # formula actually does (preflight judges by the same derivation).
        _normalize_represented_proxies(experiments, measurement_plans)
        return experiments

    def _experiment_cache_path(self, hypothesis_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hypothesis_id}.json"

    def _read_cached_experiment_batch(
        self, hypothesis_id: str
    ) -> _ExperimentBatch | None:
        path = self._experiment_cache_path(hypothesis_id)
        if path is None or not path.is_file():
            return None
        return _ExperimentBatch.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_cached_experiment_batch(
        self, hypothesis_id: str, batch: _ExperimentBatch
    ) -> None:
        path = self._experiment_cache_path(hypothesis_id)
        if path is None:
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(batch.model_dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _preflight_sample_grid(self, sample_panel: pl.LazyFrame) -> pl.DataFrame:
        """(date, permno) grid of the preflight sample, with a row index.

        Cached on the instance: the sample is deterministic (2008–2010,
        first 64 permnos), and round 1 re-preflights the round-0 accepted
        experiments, so their rank vectors must line up on the same grid.
        """
        grid = getattr(self, "_preflight_grid", None)
        if grid is None:
            grid = (
                sample_panel.select("date", "permno")
                .collect(engine="streaming")
                .with_row_index("idx")
            )
            self._preflight_grid = grid
        return grid

    def _preflight_rank_values(
        self, compiled: pl.LazyFrame, experiment_id: str, grid: pl.DataFrame
    ) -> np.ndarray | None:
        """Cross-sectional ranks of one compiled formula on the sample grid.

        Ranks are computed per date (average ranks for ties) and placed on
        the shared grid, NaN where the formula produced no finite value.
        Cached by experiment_id: round 1 re-preflights accepted experiments
        whose values were already computed in round 0. Returns None when
        the values cannot be computed; dedup is best-effort and a failure
        here must not fail preflight itself.
        """
        cache = getattr(self, "_preflight_rank_cache", None)
        if cache is None:
            cache = self._preflight_rank_cache = {}
        if experiment_id in cache:
            return cache[experiment_id]
        try:
            frame = (
                compiled.select("date", "permno", "raw_factor")
                .filter(pl.col("raw_factor").is_finite())
                .collect(engine="streaming")
            )
            if frame.is_empty():
                return None
            ranked = frame.with_columns(
                pl.col("raw_factor").rank("average").over("date").alias("rank")
            ).select("date", "permno", "rank")
            joined = grid.join(ranked, on=["date", "permno"], how="left").sort("idx")
            values = joined["rank"].to_numpy().astype(float)
        except Exception:
            return None
        cache[experiment_id] = values
        return values

    def preflight_experiments(
        self,
        experiments: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
        correction_round: int,
    ) -> list[PreflightResult]:
        signatures: Counter[str] = Counter(
            _experiment_signature(item) for item in experiments
        )
        per_hypothesis = Counter(item.hypothesis_id for item in experiments)
        results = []
        plan_by_id = {plan.plan_id: plan for plan in measurement_plans}
        panel = pl.scan_parquet(self.panel_path)
        sample_start = pl.datetime(2008, 1, 1)
        sample_end = pl.datetime(2010, 12, 31)
        sample_ids = (
            panel.filter(pl.col("date").is_between(sample_start, sample_end))
            .select("permno")
            .unique()
            .sort("permno")
            .limit(64)
            .collect(engine="streaming")
            .get_column("permno")
            .to_list()
        )
        sample_panel = panel.filter(
            pl.col("date").is_between(sample_start, sample_end)
            & pl.col("permno").is_in(sample_ids)
        )
        grid = self._preflight_sample_grid(sample_panel)
        values_by_id: dict[str, np.ndarray] = {}
        issues_by_experiment: list[list[CheckIssue]] = []
        for experiment in experiments:
            issues = []
            plan = plan_by_id.get(experiment.measurement_plan_id)
            if plan is None or plan.hypothesis_id != experiment.hypothesis_id:
                issues.append(
                    CheckIssue(
                        code="invalid_measurement_plan_lineage",
                        message="Experiment does not reference its hypothesis measurement plan.",
                        field="measurement_plan_id",
                    )
                )
            else:
                plan_proxy_ids = {proxy.proxy_id for proxy in plan.proxies}
                represented = set(experiment.represented_proxy_ids)
                if not represented.issubset(plan_proxy_ids):
                    issues.append(
                        CheckIssue(
                            code="unknown_measurement_proxy",
                            message="Experiment references a proxy outside its measurement plan.",
                            field="represented_proxy_ids",
                        )
                    )
                # Coverage is judged by what the formula actually uses,
                # not by the declared list: 3d2eeec5 fulls routinely
                # implemented proxies they did not declare, and were
                # rejected for paperwork while their formulas were
                # complete. The issue names the missing proxies so repair
                # has a concrete job instead of re-doing the diff itself.
                derived = _derived_proxy_ids(experiment, plan)
                if (
                    experiment.variant_type == ExperimentVariantType.FULL_MECHANISM
                    and derived != plan_proxy_ids
                ):
                    missing = sorted(plan_proxy_ids - derived)
                    issues.append(
                        CheckIssue(
                            code="incomplete_full_mechanism",
                            message=(
                                "Full-mechanism experiment does not implement "
                                f"proxies {', '.join(missing)}: their required "
                                "fields are not all used by the formula."
                            ),
                            field="represented_proxy_ids",
                        )
                    )
                proxy_fields = {
                    proxy.proxy_id: set(proxy.required_fields)
                    for proxy in plan.proxies
                }
                used_fields = experiment.factor.formula.referenced_fields()
                missing_by_proxy = {}
                for proxy_id in sorted(represented):
                    required = proxy_fields.get(proxy_id)
                    if required is None:
                        continue
                    missing_fields = sorted(
                        field.value for field in required - used_fields
                    )
                    if missing_fields:
                        missing_by_proxy[proxy_id] = missing_fields
                if missing_by_proxy:
                    detail = "; ".join(
                        f"{proxy_id} needs {', '.join(fields)}"
                        for proxy_id, fields in missing_by_proxy.items()
                    )
                    issues.append(
                        CheckIssue(
                            code="measurement_proxy_not_implemented",
                            message=(
                                "Formula does not use every field required by "
                                "its declared measurement proxies: "
                                f"{detail}."
                            ),
                            field="factor.formula",
                        )
                    )
            if per_hypothesis[experiment.hypothesis_id] > self.max_experiments_per_hypothesis:
                issues.append(
                    CheckIssue(
                        code="experiment_limit_exceeded",
                        message="Hypothesis generated too many experiment configurations.",
                    )
                )
            if signatures[_experiment_signature(experiment)] > 1:
                issues.append(
                    CheckIssue(
                        code="exact_duplicate_experiment",
                        message="An identical formula and horizon exists in this run.",
                    )
                )
            for message in self.backtest_engine.validate_sample_capacity(
                experiment.prediction_horizon_days
            ):
                issues.append(
                    CheckIssue(
                        code="insufficient_sample_capacity",
                        message=message,
                        field="prediction_horizon_days",
                    )
                )
            try:
                compiled = compile_factor_spec(
                    sample_panel, _legacy_factor_spec(experiment)
                )
                schema = compiled.collect_schema()
                if "raw_factor" not in schema:
                    raise ValueError("compiled formula did not produce raw_factor")
                sample = (
                    compiled
                    .select("date", "permno", "raw_factor")
                    .filter(pl.col("raw_factor").is_finite())
                    .limit(1)
                    .collect(engine="streaming")
                )
                if sample.is_empty():
                    raise ValueError("formula produced no finite value in preflight sample")
                values = self._preflight_rank_values(
                    compiled, experiment.experiment_id, grid
                )
                if values is not None:
                    values_by_id[experiment.experiment_id] = values
                    # Degenerate-factor check (Patch 8, 0f45e744): a factor
                    # that is constant within each cross-section passes
                    # the "at least one finite value" check above (0 is
                    # finite) but cannot produce a Rank IC, and train used
                    # to crash on it. Ranks make the check cheap: a
                    # constant factor gives every stock the same average
                    # rank, so the date has only one distinct value.
                    varied_dates = _count_varied_cross_section_dates(
                        values, grid["date"].to_numpy()
                    )
                    if varied_dates < DEGENERATE_SIGNAL_MIN_VARIED_DATES:
                        issues.append(
                            CheckIssue(
                                code="degenerate_signal",
                                message=(
                                    "Factor has no cross-sectional "
                                    "dispersion: it distinguishes at least "
                                    "two stocks on only "
                                    f"{varied_dates} dates in the 2008-2010 "
                                    "preflight sample. A factor that is "
                                    "constant within each cross-section "
                                    "cannot produce a Rank IC."
                                ),
                                field="factor.formula",
                            )
                        )
            except Exception as error:
                issues.append(
                    CheckIssue(
                        code="formula_compile_failed",
                        message=str(error),
                        field="formula",
                    )
                )
            issues_by_experiment.append(issues)
        # Phase 2: signal dedup. Compare each experiment against the earlier
        # ones in list order; the first of a near-identical pair is kept.
        # An earlier experiment only counts as a reference when it is
        # otherwise clean: if it is already rejected for something else it
        # goes to repair (or is dropped), and round 1 re-checks the pair if
        # both survive. Flagging the later one either way would kill a clean
        # experiment for duplicating a doomed one.
        self._duplicate_pairs = {}
        for later_index, experiment in enumerate(experiments):
            later_values = values_by_id.get(experiment.experiment_id)
            if later_values is None:
                continue
            for earlier_index, earlier in enumerate(experiments[:later_index]):
                if issues_by_experiment[earlier_index]:
                    continue
                earlier_values = values_by_id.get(earlier.experiment_id)
                if earlier_values is None:
                    continue
                corr, overlap = _rank_correlation(earlier_values, later_values)
                if (
                    corr is not None
                    and overlap >= DUPLICATE_SIGNAL_MIN_OVERLAP
                    and abs(corr) >= DUPLICATE_SIGNAL_MIN_CORRELATION
                    and not _is_same_hypothesis_full_vs_baseline(
                        earlier, experiment
                    )
                ):
                    issues_by_experiment[later_index].append(
                        CheckIssue(
                            code="duplicate_signal",
                            message=(
                                "Formula produces a signal nearly identical to "
                                f"experiment {earlier.experiment_id} "
                                "(cross-sectional rank correlation "
                                f"{corr:.2f} on the 2008-2010 preflight sample). "
                                "Redesign the formula to express a genuinely "
                                "different mechanism from that experiment."
                            ),
                            field="factor.formula",
                        )
                    )
                    self._duplicate_pairs[experiment.experiment_id] = (
                        earlier.experiment_id,
                        corr,
                    )
                    break
        for experiment, issues in zip(experiments, issues_by_experiment):
            results.append(
                PreflightResult(
                    experiment_id=experiment.experiment_id,
                    accepted=not issues,
                    issues=issues,
                    correction_round=correction_round,
                )
            )
        return results

    def repair_experiments(
        self,
        rejected: list[FactorExperimentSpec],
        issues_by_id: dict[str, list[CheckIssue]],
        accepted: list[FactorExperimentSpec],
        measurement_plans: list[MeasurementPlan],
    ) -> list[FactorExperimentSpec]:
        repaired = []
        plan_by_id = {plan.plan_id: plan for plan in measurement_plans}
        duplicate_pairs = getattr(self, "_duplicate_pairs", {}) or {}
        for experiment in rejected:
            plan = plan_by_id.get(experiment.measurement_plan_id)
            # duplicate 的对照实验可能在另一个假设里：existing_experiments
            # 原来只给同假设的已通过实验，跨假设的重复对照就看不到，LLM
            # 不知道要跟谁区分开。把本次被拒实验的重复对照（preflight 在
            # _duplicate_pairs 里记的）也加进去，修的时候才能照着改出
            # 真正不同的公式。
            paired_ids = set()
            if experiment.experiment_id in duplicate_pairs:
                paired_ids.add(duplicate_pairs[experiment.experiment_id][0])
            try:
                batch = self._deepseek_json(
                    (
                        "修正一个未通过 Preflight 的 FactorSpec。只使用允许的 OHLCV/市值字段"
                        "和 DSL 算子，返回一个更简单、可执行且经济含义一致的实验。"
                        "必须针对 checker_issues 中的真实错误修复，不得猜测失败原因。"
                        f"{DUPLICATE_REPAIR_RULES}"
                        f"{FORMULA_NODE_SHAPE_RULES}"
                        "只返回符合 output_schema 的完整 JSON 对象。"
                    ),
                {
                    "rejected_experiment": experiment.model_dump(mode="json"),
                    "measurement_plan": (
                        plan.model_dump(mode="json") if plan is not None else None
                    ),
                    "checker_issues": [
                        item.model_dump(mode="json")
                        for item in issues_by_id.get(experiment.experiment_id, [])
                    ],
                    "existing_experiments": [
                        {
                            "experiment_id": item.experiment_id,
                            "prediction_horizon_days": item.prediction_horizon_days,
                            "formula": item.factor.formula.model_dump(mode="json"),
                        }
                        for item in accepted
                        if item.hypothesis_id == experiment.hypothesis_id
                        or item.experiment_id in paired_ids
                    ],
                    "available_fields": [field.value for field in DataField],
                    "available_operators": [item.value for item in FormulaOperator],
                    "formula_node_schema": FormulaNode.model_json_schema(),
                    "output_schema": _ExperimentRepairBatch.model_json_schema(),
                },
                _ExperimentRepairBatch,
            )
            except ValueError:
                # 单个实验 3 次重试仍修不好时只丢弃该实验，不要让整轮崩掉：
                # Manager 会把已通过的实验 + 其余修好的实验拿去做第二轮 Preflight，
                # 再进 train。丢弃本身是可观测的：该实验不会出现在新的
                # experiment_specs 里，也就不会进 train_metrics。
                continue
            draft = batch.experiments[0]
            formula = draft.formula
            repaired_factor = FactorFormulaSpecV2(
                factor_id=f"{experiment.factor.factor_id}-R1",
                hypothesis_id=experiment.hypothesis_id,
                name=draft.name,
                description=draft.description,
                formula=formula,
                required_fields=sorted(
                    formula.referenced_fields(), key=lambda item: item.value
                ),
                lookback_days=formula.required_history(),
            )
            repaired.append(
                FactorExperimentSpec(
                    experiment_id=f"{experiment.experiment_id}-R1",
                    hypothesis_id=experiment.hypothesis_id,
                    variation_reason=draft.variation_reason,
                    measurement_plan_id=experiment.measurement_plan_id,
                    variant_type=experiment.variant_type,
                    represented_proxy_ids=experiment.represented_proxy_ids,
                    factor=repaired_factor,
                    prediction_horizon_days=experiment.prediction_horizon_days,
                )
            )
        # Same normalization as generate: a repaired full inherits the old
        # (incomplete) declaration, so rewrite it from the new formula.
        _normalize_represented_proxies(repaired, measurement_plans)
        return repaired

    def run_backtests(
        self,
        experiments: list[FactorExperimentSpec],
        split: SplitName,
        train_metrics: dict[str, SplitMetrics] | None = None,
    ) -> list[SplitMetrics]:
        return self.backtest_engine.evaluate(experiments, split, train_metrics)

    def build_memory_summary(
        self,
        train_metrics: list[SplitMetrics],
        validation_metrics: list[SplitMetrics],
        test_metrics: list[SplitMetrics],
    ) -> dict[str, Any]:
        train_pass = [
            item for item in train_metrics if passes_selection_gate(item)
        ]
        validation_pass = [
            item for item in validation_metrics if passes_selection_gate(item)
        ]
        return {
            "summary_version": "v2.1-real-1",
            "train_experiments": len(train_metrics),
            "train_passed": len(train_pass),
            "validation_experiments": len(validation_metrics),
            "validation_passed": len(validation_pass),
            "test_experiments": len(test_metrics),
            "research_experience": [
                (
                    "Some generated mechanisms passed the training threshold."
                    if train_pass
                    else "No generated mechanism passed the training threshold."
                ),
                (
                    "At least one signal remained valid in frozen validation."
                    if validation_pass
                    else "No training signal remained valid in frozen validation."
                ),
            ],
        }

    def _deepseek_text(self, system_prompt: str, payload: dict[str, Any]) -> str:
        return self._completion_content(
            self.deepseek,
            self.deepseek_model,
            system_prompt,
            payload,
            max_tokens=2_048,
            provider_name="DeepSeek",
        )

    def _deepseek_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        model: type[BaseModel],
    ) -> Any:
        return self._validated_json_completion(
            self.deepseek,
            self.deepseek_model,
            system_prompt,
            payload,
            model,
            provider_name="DeepSeek",
        )

    def _school_text(self, system_prompt: str, payload: dict[str, Any]) -> str:
        return self._completion_content(
            self.school_openai,
            self.openai_model,
            system_prompt,
            payload,
            max_tokens=8_192,
            provider_name="School OpenAI endpoint",
        )

    def _school_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        model: type[BaseModel],
    ) -> Any:
        return self._validated_json_completion(
            self.school_openai,
            self.openai_model,
            system_prompt,
            payload,
            model,
            provider_name="School OpenAI endpoint",
        )

    def _validated_json_completion(
        self,
        client: OpenAI,
        model_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        output_model: type[BaseModel],
        *,
        provider_name: str,
    ) -> Any:
        request_payload = dict(payload)
        last_error: ValidationError | json.JSONDecodeError | None = None
        for schema_attempt in range(3):
            content = self._completion_content(
                client,
                model_name,
                system_prompt,
                request_payload,
                max_tokens=8_192,
                provider_name=provider_name,
                response_format={"type": "json_object"},
            )
            try:
                try:
                    raw_payload = json.loads(content)
                except json.JSONDecodeError:
                    # Patch 15: try a deterministic repair first — the
                    # model cannot surgically fix one missing comma in a
                    # 6KB batch (a5a57aea resume crashed on this three
                    # times). A failed salvage re-raises and the LLM
                    # retry proceeds as before; this inner try sits
                    # inside the outer one so a salvaged parse failure
                    # still lands in the retry logic below.
                    salvaged = _salvage_json(content)
                    if salvaged is None:
                        raise
                    raw_payload = json.loads(salvaged)
                if output_model in {_ExperimentBatch, _ExperimentRepairBatch}:
                    raw_payload = _normalize_experiment_batch(raw_payload)

                return output_model.model_validate(raw_payload)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = error

                if isinstance(error, ValidationError):
                    # include_input=True 是有意的：上一次的反馈只给了 loc + message，
                    # LLM 看不到自己写错的节点（例如带 field 的 add），3 次重试都在
                    # 重复同一个错误。input 只保留截断后的预览，避免整批 payload 过大。
                    validation_errors = [
                        {
                            **item,
                            "input": _truncated_error_input(item.get("input")),
                        }
                        if "input" in item
                        else item
                        for item in error.errors(
                            include_url=False,
                            include_input=True,
                        )
                    ]
                else:
                    validation_errors = [
                        {
                            "type": "json_decode_error",
                            "message": str(error),
                            "line": error.lineno,
                            "column": error.colno,
                            "position": error.pos,
                        }
                    ]

                if schema_attempt < 2:
                    request_payload = {
                        **payload,
                        "previous_invalid_output": content,
                        "validation_errors": validation_errors,
                        "correction_instruction": (
                            "Return one complete syntactically valid JSON object. Correct every "
                            "listed JSON or schema error. Return the full required experiment "
                            "batch, including baseline and full_mechanism when the output schema "
                            "requires them. Preserve valid economic content. "
                            f"{FORMULA_NODE_SHAPE_RULES}"
                        ),
                    }
        raise ValueError(
            f"{provider_name} returned schema-invalid JSON after three attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _completion_content(
        client: OpenAI,
        model: str,
        system_prompt: str,
        payload: dict[str, Any],
        *,
        max_tokens: int,
        provider_name: str,
        response_format: dict[str, str] | None = None,
        attempts: int = 3,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        finish_reasons: list[str] = []
        last_bad_request: BadRequestError | None = None
        for _ in range(attempts):
            options: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                options["response_format"] = response_format
            try:
                response = client.chat.completions.create(**options)
            except BadRequestError as error:
                # 0d53bd4b 的教训：HKUST 网关曾对完全合法的请求返回无响应体
                # 的 400，原样重放同一请求全部通过——是一时网关问题，不是
                # payload 问题。400 不进 SDK 的自动重试（只重试 408/409/
                # 429/5xx），这里补一次有界重试；持续 400 仍会原样抛出，
                # 不会无限重试。
                last_bad_request = error
                finish_reasons.append("bad_request")
                time.sleep(5)
                continue
            last_bad_request = None
            finish_reasons.append(
                str(response.choices[0].finish_reason)
                if response.choices
                else "no_choice"
            )
            content = (
                response.choices[0].message.content if response.choices else None
            )
            if content and content.strip():
                return content.strip()
        if last_bad_request is not None:
            raise last_bad_request
        raise ValueError(
            f"{provider_name} returned empty content in {attempts} consecutive "
            f"responses; finish_reasons={finish_reasons}"
        )


def _direction_focus(direction: ResearchDirection) -> str:
    return {
        ResearchDirection.MOMENTUM: "信息扩散、趋势延续与拥挤过滤",
        ResearchDirection.REVERSAL: "过度反应、价格冲击与卖压衰竭",
        ResearchDirection.LIQUIDITY_VOLATILITY: "流动性变化、风险补偿与波动状态",
        ResearchDirection.PRICE_VOLUME: "价格与成交量的确认、背离和非线性交互",
        ResearchDirection.OVERNIGHT_INTRADAY: "隔夜收益与日内收益的分解、开盘缺口与隔夜信息到达",
    }[direction]


def _direction_literature(direction: ResearchDirection) -> str:
    return _DIRECTION_LITERATURE[direction]


def _hypothesis_signature(hypothesis: DevelopedHypothesis) -> str:
    payload = {
        "title": " ".join(hypothesis.title.split()).casefold(),
        "claim": " ".join(hypothesis.claim.split()).casefold(),
        "economic_rationale": " ".join(
            hypothesis.economic_rationale.split()
        ).casefold(),
        "core_mechanism": " ".join(hypothesis.core_mechanism.split()).casefold(),
        "horizons": hypothesis.prediction_horizons_days,
        "concepts": hypothesis.required_concepts,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _experiment_signature(experiment: FactorExperimentSpec) -> str:
    return json.dumps(
        {
            "formula": experiment.factor.formula.model_dump(mode="json"),
            "prediction_horizon_days": experiment.prediction_horizon_days,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _legacy_factor_spec(experiment: FactorExperimentSpec) -> FactorSpec:
    factor = experiment.factor
    return FactorSpec(
        factor_id=factor.factor_id,
        hypothesis_id=factor.hypothesis_id,
        name=factor.name,
        description=factor.description,
        formula=factor.formula,
        required_fields=factor.required_fields,
        lookback_days=factor.lookback_days,
        higher_is_better=True,
        rebalance_frequency="weekly",
        universe_filters=factor.universe_filters,
        portfolio_rules=PortfolioRules(),
    )


_UNARY_OPERATOR_NAMES = {
    item.value
    for item in (
        FormulaOperator.LAG,
        FormulaOperator.RETURN,
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
        FormulaOperator.ZSCORE,
        FormulaOperator.RANK,
        FormulaOperator.ABS,
        FormulaOperator.LOG,
        FormulaOperator.NEGATE,
    )
}


def _salvage_json(content: str, max_repairs: int = 8) -> str | None:
    """Deterministically repair common JSON syntax errors, or None.

    Patch 15 (a5a57aea resume crash): DeepSeek returned malformed JSON
    ("Expecting ',' delimiter" at char 6435) three attempts in a row.
    The retry already sends the broken output back with the error
    position, but the model cannot surgically fix one comma in a 6KB+
    batch — it regenerates the whole batch and breaks elsewhere. So
    repair it here, using the parser's own error position: a missing
    comma is inserted where the parser stopped, a trailing comma before
    } or ] is removed, a missing colon likewise. Each repair is
    re-parsed, so a wrong repair just fails again and returns None.
    The salvaged text still goes through schema validation afterwards,
    so a repair that changes the shape cannot smuggle bad data in —
    it fails validation and the LLM retry proceeds as before.
    """
    text = content
    for _ in range(max_repairs):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as error:
            if "Expecting ',' delimiter" in error.msg:
                text = text[: error.pos] + "," + text[error.pos :]
            elif "Expecting ':' delimiter" in error.msg:
                text = text[: error.pos] + ":" + text[error.pos :]
            elif "Expecting property name enclosed in double quotes" in error.msg:
                before = text[: error.pos].rstrip()
                if not before.endswith(","):
                    return None
                text = before[:-1] + text[error.pos :]
            elif "Extra data" in error.msg:
                text = text[: error.pos]
            else:
                return None
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


def _truncated_error_input(value: Any, limit: int = 2_000) -> str:
    """把 Pydantic 校验错误的 input 截断成字符串预览，供 LLM 重试时定位。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…(truncated)"


def _normalize_experiment_batch(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        return payload
    normalized = dict(payload)
    normalized["experiments"] = [
        {
            **item,
            "formula": _normalize_formula_node(item.get("formula")),
        }
        if isinstance(item, dict)
        else item
        for item in experiments
    ]
    return normalized


def _normalize_formula_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    normalized = dict(node)
    args = normalized.get("args")
    if isinstance(args, list):
        normalized["args"] = [_normalize_formula_node(item) for item in args]
    op = normalized.get("op")
    # Patch 12: strip rolling parameters the op cannot accept, mirroring
    # spec.validate_shape — before the early returns below, because the
    # ec610c9d crash nodes carry BOTH a stray field and min_periods.
    # DeepSeek put min_periods on return nodes three attempts in a row
    # (the prompt already says return/lag take only window); the retry
    # feedback names the error but the LLM reintroduces it elsewhere in
    # the batch. min_periods on a return/lag node is meaningless — return
    # is a point-to-point ratio, not a rolling window — so dropping it
    # cannot change the computed values. Same for window on non-windowed
    # unary ops, quantile off quantile, ddof off std, and any rolling
    # parameter on field/constant nodes.
    if op in ("field", "constant"):
        for key in ("window", "min_periods", "quantile", "ddof"):
            normalized.pop(key, None)
    elif op in _UNARY_OPERATOR_NAMES:
        if op not in ROLLING_OPERATORS:
            normalized.pop("min_periods", None)
        if op not in WINDOWED_UNARY_OPERATORS:
            normalized.pop("window", None)
        if op != "quantile":
            normalized.pop("quantile", None)
        if op != "std":
            normalized.pop("ddof", None)
    elif op in ROLLING_BINARY_OPERATORS:
        if op != "quantile":
            normalized.pop("quantile", None)
        if op != "std":
            normalized.pop("ddof", None)
    if op in ROLLING_OPERATORS | ROLLING_BINARY_OPERATORS:
        window = normalized.get("window")
        min_periods = normalized.get("min_periods")
        if (
            window is not None
            and min_periods is not None
            and min_periods > window
        ):
            # validate_shape rejects min_periods > window; clamp to the
            # window instead of failing the whole batch on one node.
            normalized["min_periods"] = window
    field = normalized.get("field")
    args = normalized.get("args") if isinstance(normalized.get("args"), list) else []
    if (
        op in _UNARY_OPERATOR_NAMES
        and field is not None
        and len(args) == 1
    ):
        # 一元算子把 field 当成了第一个操作数：保留 args 里的那个操作数。
        normalized.pop("field", None)
        return normalized
    if op in BINARY_OPERATORS | ROLLING_BINARY_OPERATORS:
        # 二元算子把 field 当成了一个操作数（本次 Smoke 的 add 报错）：
        # field + 1 个 args 时，field 就是缺失的第一个操作数，把它还原成
        # {"op": "field", "field": ...} 补进 args；args 已满 2 个时 field
        # 是多余的，直接去掉；args 为空时补不齐，留给校验报错、走重试反馈。
        if field is not None:
            if len(args) == 1:
                normalized["args"] = [
                    {"op": "field", "field": field},
                    *args,
                ]
            normalized.pop("field", None)
        normalized.pop("value", None)
        if op not in ROLLING_BINARY_OPERATORS:
            for key in ("window", "min_periods", "quantile", "ddof"):
                normalized.pop(key, None)
        else:
            for key in ("value", "quantile", "ddof"):
                normalized.pop(key, None)
        return normalized
    if op in TERNARY_OPERATORS:
        if field is not None:
            # where 的第一个参数必须是布尔条件（greater_than/less_than 的
            # 结果），一个 bare field 永远当不了条件，所以这里不能像二元
            # 算子那样把 field 还原成第一个 args——还原了也会在校验里被
            # where 的条件检查打回来。直接去掉 field，让校验报“需要 3 个
            # args”，重试反馈能把缺失说清楚。
            normalized.pop("field", None)
        normalized.pop("value", None)
        for key in ("window", "min_periods", "quantile", "ddof"):
            normalized.pop(key, None)
        # Patch 14: a two-arg where whose first arg is a comparison is
        # the "filter" idiom — where(cond, X) means X where cond holds,
        # else 0, the else-branch the LLM omitted. Complete it with a
        # constant 0.0, the else value this codebase uses everywhere
        # (e.g. where(corr > 0, P1, 0.0)). Only when args[0] is a
        # comparison: a two-arg where without a condition cannot be
        # recovered, so it stays invalid and goes through retry feedback.
        ternary_args = (
            normalized.get("args")
            if isinstance(normalized.get("args"), list)
            else []
        )
        if (
            len(ternary_args) == 2
            and isinstance(ternary_args[0], dict)
            and ternary_args[0].get("op") in ("greater_than", "less_than")
        ):
            normalized["args"] = [
                *ternary_args,
                {"op": "constant", "value": 0.0},
            ]
        return normalized
    return normalized


def _valid_experiment_drafts(payload: Any) -> list[_ExperimentDraft]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("experiments"), list
    ):
        return []
    valid = []
    for item in payload["experiments"]:
        try:
            valid.append(_ExperimentDraft.model_validate(item))
        except ValidationError:
            continue
    return valid
