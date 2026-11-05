"""OpenAI Research Agent for structured hypothesis generation."""
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_workflow.factor_library import FactorKnowledgeBase
from agentic_workflow.state import WorkflowState
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.base import ToolResult
from agentic_workflow.tools.web_research import (
    KeyFreeWebResearchProvider,
    WebResearchProvider,
    WebResearchSource,
    build_strategy_search_queries,
)


class SupportedConcept(StrEnum):
    INTRADAY_RANGE = "intraday_range" # 日内(high-low)/close
    LIQUIDITY = "liquidity"
    MARKET_CAP = "market_cap"
    PRICE_MOMENTUM = "price_momentum"
    PRICE_REVERSAL = "price_reversal"
    PRICE_VOLUME_INTERACTION = "price_volume_interaction"
    RECENT_RETURN = "recent_return" # 过去五日收益率
    VOLATILITY = "volatility"
    VOLUME_ABNORMALITY = "volume_abnormality"
    VOLUME_TREND = "volume_trend"


@dataclass(frozen=True)
class ResearchStrategy:
    key: str
    name: str
    focus: str
    allowed_concepts: tuple[SupportedConcept, ...]


RESEARCH_STRATEGIES = (
    ResearchStrategy(
        key="momentum",
        name="Momentum Research Agent",
        focus=(
            "研究价格趋势延续与信息扩散。优先提出由中短期价格趋势主导、"
            "成交量趋势仅用于确认或区分市场状态的假设。"
        ),
        allowed_concepts=(
            SupportedConcept.PRICE_MOMENTUM,
            SupportedConcept.RECENT_RETURN,
            SupportedConcept.VOLUME_TREND,
        ),
    ),
    ResearchStrategy(
        key="reversal",
        name="Reversal Research Agent",
        focus=(
            "研究短期过度反应与价格均值回归。优先区分普通下跌与具有异常成交量、"
            "日内振幅或吸收特征的暂时性价格冲击。"
        ),
        allowed_concepts=(
            SupportedConcept.PRICE_REVERSAL,
            SupportedConcept.RECENT_RETURN,
            SupportedConcept.VOLUME_ABNORMALITY,
            SupportedConcept.INTRADAY_RANGE,
        ),
    ),
    ResearchStrategy(
        key="liquidity_volatility",
        name="Liquidity and Volatility Research Agent",
        focus=(
            "研究流动性、波动率和规模相关的风险补偿。每个假设选择一个主要风险机制，"
            "其他概念只作为条件，避免把规模、流动性和波动率机械相乘。"
        ),
        allowed_concepts=(
            SupportedConcept.LIQUIDITY,
            SupportedConcept.VOLATILITY,
            SupportedConcept.INTRADAY_RANGE,
            SupportedConcept.MARKET_CAP,
        ),
    ),
    ResearchStrategy(
        key="price_volume",
        name="Price-Volume Research Agent",
        focus=(
            "研究价格与成交量之间的确认或背离关系。假设必须描述价格信号在何种"
            "成交量状态下增强或反转，避免仅提出单独的价格或成交量效应。"
        ),
        allowed_concepts=(
            SupportedConcept.PRICE_VOLUME_INTERACTION,
            SupportedConcept.VOLUME_ABNORMALITY,
            SupportedConcept.VOLUME_TREND,
            SupportedConcept.RECENT_RETURN,
        ),
    ),
)

class SourceReferenceProposal(BaseModel):
    source_id: str = Field(pattern=r"^WEB-\d{2}$")
    relevance: str = Field(min_length=1, max_length=500)


class FactorReferenceProposal(BaseModel):
    factor_id: str = Field(min_length=1, max_length=100)
    adaptation: str = Field(min_length=1, max_length=500)


class HypothesisProposal(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    claim: str = Field(min_length=1, max_length=500)
    economic_rationale: str = Field(min_length=1, max_length=500)
    prediction_direction: Literal["positive", "negative"]
    prediction_horizon_days: Literal[5]
    required_concepts: list[SupportedConcept] = Field(
        min_length=1,
        max_length=4,
    )
    source_references: list[SourceReferenceProposal] = Field(
        min_length=1,
        max_length=3,
    )
    factor_references: list[FactorReferenceProposal] = Field(
        min_length=1,
        max_length=3,
    )
    
class ResearchAgentTool:
    name = "research_agent"
    
    def __init__(
        self, 
        repository: SQLiteResearchRepository,
        model: str| None = None,
        history_limit: int = 10,
        strategy: ResearchStrategy | None = None,
        web_research: WebResearchProvider | None = None,
        factor_library: FactorKnowledgeBase | None = None,
        *,
        client: Any | None = None,
    )-> None:
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        
        self.repository = repository
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.history_limit = history_limit
        self.strategy = strategy
        self.web_research = web_research or KeyFreeWebResearchProvider()
        self.factor_library = factor_library or FactorKnowledgeBase()
        self.name = (
            f"research_agent_{strategy.key}" if strategy else "research_agent"
        )
        
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is not set in environment variable 'OPENAI_API_KEY'")
            
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_key,
                base_url = os.getenv(
                    "OPENAI_BASE_URL",  
                    "https://hkust.azure-api.net/hkust-genai/v1",
                    ),
                default_headers = {"api-key": api_key,},
            )
        self.client = client
    
    def run(self,state: WorkflowState) -> ToolResult:
        try:
            web_sources = self.web_research.search(
                build_strategy_search_queries(
                    self.strategy.key if self.strategy is not None else None
                )
            )
            library_concepts = [
                concept.value
                for concept in (
                    self.strategy.allowed_concepts
                    if self.strategy is not None
                    else tuple(SupportedConcept)
                )
            ]
            factor_examples = self.factor_library.search(
                library_concepts,
                iteration=state["iteration"],
            )
            retrieval_state = state
            if self.strategy is not None:
                retrieval_state = {
                    **state,
                    "hypothesis": {
                        "required_concepts": [
                            concept.value
                            for concept in self.strategy.allowed_concepts
                        ]
                    },
                }
            relevant_experiments = self.repository.relevant_experiments(
                state=retrieval_state,
                limit=self.history_limit,
            )
            research_context = build_research_context(
                state = state,
                relevant_experiments = relevant_experiments,
                strategy=self.strategy,
                web_sources=web_sources,
                factor_library_examples=factor_examples,
                )
            user_message = {
                "task": "根据研究上下文提出一个新的结构化量化研究假设。",
                "research_context": research_context,
                "output_schema": HypothesisProposal.model_json_schema(),
            }
            response = self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {
                        "role": "system",
                        "content": build_system_prompt(self.strategy),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(user_message, ensure_ascii=False),
                    }
                ],
                response_format = {
                    "type": "json_object",
                },
            )
            
            if not response.choices:
                return ToolResult.fail(
                    "Research Agent returned no choices in the response."
                )
                
            response_content = response.choices[0].message.content
            if not response_content:
                return ToolResult.fail(
                    "Research Agent returned an empty content in the response."
                )
            proposal = HypothesisProposal.model_validate_json(response_content) #pydantic校验
            sources_by_id = {
                source.source_id: source for source in web_sources
            }
            factors_by_id = {
                str(factor["factor_id"]): factor for factor in factor_examples
            }
            unknown_sources = sorted(
                {
                    reference.source_id
                    for reference in proposal.source_references
                    if reference.source_id not in sources_by_id
                }
            )
            if unknown_sources:
                return ToolResult.fail(
                    "Research Agent cited unknown web sources: "
                    + ", ".join(unknown_sources)
                )
            unknown_factors = sorted(
                {
                    reference.factor_id
                    for reference in proposal.factor_references
                    if reference.factor_id not in factors_by_id
                }
            )
            if unknown_factors:
                return ToolResult.fail(
                    "Research Agent cited unknown factor-library entries: "
                    + ", ".join(unknown_factors)
                )
            hypothesis = proposal.model_dump(
                mode="json",
                exclude={"source_references", "factor_references"},
            )
            hypothesis["sources"] = [
                {
                    "title": sources_by_id[reference.source_id].title,
                    "url": str(sources_by_id[reference.source_id].url),
                    "relevance": reference.relevance,
                }
                for reference in proposal.source_references
            ]
            hypothesis["factor_references"] = [
                {
                    "factor_id": reference.factor_id,
                    "name": factors_by_id[reference.factor_id]["name"],
                    "family": factors_by_id[reference.factor_id]["family"],
                    "implementation_status": factors_by_id[reference.factor_id][
                        "implementation_status"
                    ],
                    "adaptation": reference.adaptation,
                }
                for reference in proposal.factor_references
            ]
            if self.strategy is not None:
                allowed_concepts = {
                    concept.value for concept in self.strategy.allowed_concepts
                }
                selected_concepts = set(hypothesis["required_concepts"])
                unsupported = sorted(selected_concepts - allowed_concepts)
                if unsupported:
                    return ToolResult.fail(
                        f"{self.strategy.name} selected concepts outside its "
                        f"research scope: {', '.join(unsupported)}"
                    )
                hypothesis["research_strategy"] = self.strategy.key
            hypothesis["hypothesis_id"] = self._build_hypothesis_id(state)
            
            return ToolResult.ok(
                {"hypothesis": hypothesis,}
            )
            
        except Exception as error:
            return ToolResult.fail(
                f"Research Agent encountered an error: {error}"
            )
    def _build_hypothesis_id(self, state: WorkflowState) -> str:
        run_prefix = state["run_id"][:8].upper()
        suffix = f"-{self.strategy.key.upper()}" if self.strategy else ""
        return f"H-{run_prefix}-{state['iteration']:04d}{suffix}"
    
    
AVAILABLE_DATA_FIELDS = (
            "open", "high", "low", "close", "volume", "mktcap",
        )
    
def build_system_prompt(strategy: ResearchStrategy | None = None) -> str:
    concepts = (
        strategy.allowed_concepts
        if strategy is not None
        else tuple(SupportedConcept)
    )
    supported_concepts = ", ".join(concept.value for concept in concepts)
    strategy_instruction = (
        f"你的身份是 {strategy.name}。{strategy.focus}"
        if strategy is not None
        else ""
    )
    
    return (
        "你是自动化量化研究系统中的 Research Agent。"
        f"{strategy_instruction}"
        "你的职责是提出一个具有经济含义、可以使用现有数据直接证伪的横截面选股假设。"
        "每次只提出一个假设，并使用一个核心经济机制。"
        "claim 必须明确说明哪些可观测特征较高或较低的股票，预期未来五个交易日"
        "横截面收益更高或更低，prediction_direction 必须与该表述一致。"
        "允许灵活组合一至三个概念，探索条件效应、状态依赖、确认关系和背离关系。"
        "组合必须围绕一个可解释的核心机制，其他概念作为条件、确认或反向证据，"
        "避免无经济逻辑地堆叠多个弱信号。"
        "required_concepts 仅从当前研究方向提供的概念中选择。"
        "输出内容聚焦研究假设，公式、FactorSpec 和程序代码由后续模块生成。"
        "系统支持最低价格、滚动成交额分位数和市值分位数股票池筛选，"
        "并支持 Top 20 入选、跌出 Top 30 退出的持仓缓冲。"
        "不要提出除此之外的行业中性、基本面字段、动态调仓频率或其他未支持规则。"
        "必须把 relevant_experiments 当作经验约束。若相似概念、方向或经济机制已经"
        "因为 Rank IC 较低而失败，不得只改标题、措辞、阈值或附加另一个概念后重复提出。"
        "只有在改变核心条件机制、预测方向或信号时间结构，并且该改变能回应历史诊断时，"
        "才可以继续研究相近主题。历史组合净收益为正但 Rank IC 失败时，仍视为预测关系"
        "未得到支持，不能据此重复该结构。"
        "web_sources 是本轮实时网页检索结果。把这些资料作为研究灵感和外部证据，"
        "允许参考论文、机构文章、论坛和策略博客，但需要独立判断其经济逻辑。"
        "source_references 必须引用一至三个 web_sources 中真实存在的 source_id，"
        "并简要说明每项资料如何支持或启发当前假设；不得编造来源或 URL。"
        "factor_library_examples 来自 Alpha101、Qlib Alpha158 和经典学术因子库。"
        "允许参考当前不可实现因子的经济机制，但最终假设只能使用本方向允许的概念和"
        "现有 OHLCV/市值数据。factor_references 必须引用一至三个真实 factor_id，"
        "adaptation 说明如何改造、组合或简化该因子，而不是机械复制原公式。"
        "优先选择经济方向清晰、预期横截面关系单调、周频下具有一定持续性且能够由"
        "后续 FactorSpec 简洁实现的假设。"
        "结合研究目标、Manager 指令和历史失败证据，提出实质差异化的新假设。"
        "仅返回符合 output_schema 的 JSON 对象。"
        f"当前支持的概念包括：{supported_concepts}。"
    )
    
def build_research_context(
    state: WorkflowState,
    relevant_experiments: list[dict[str, object]],
    strategy: ResearchStrategy | None = None,
    web_sources: list[WebResearchSource] | None = None,
    factor_library_examples: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    manager_decision = state["manager_decision"] or {}
    
    return {
         "research_goal": state["research_goal"],
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"],
        "available_data_fields": list(AVAILABLE_DATA_FIELDS),
        "execution_capabilities": {
            "universe_filters": [
                "minimum price",
                "minimum rolling dollar-volume percentile",
                "minimum market-cap percentile",
            ],
            "portfolio_rules": {
                "entry": "top 20",
                "exit_buffer": "exit below rank 30",
                "rebalance_frequency": "weekly",
            },
        },
        "research_quality_criteria": [
            "one primary economic mechanism",
            "clear monotonic cross-sectional prediction",
            "combine one to three concepts as mechanism, condition, or confirmation",
            "weekly-persistent signal compatible with Top 20 / Exit 30",
            "simple implementation with supported OHLCV and market-cap data",
        ],
        "web_sources": [
            source.model_dump(mode="json") for source in web_sources or []
        ],
        "factor_library_examples": factor_library_examples or [],
        "historical_learning_policy": {
            "rank_ic_failure": (
                "Treat low Rank IC as rejection of the predictive relationship even "
                "when portfolio net return is positive."
            ),
            "similar_failed_hypothesis": (
                "Do not reuse the same concepts, direction, and mechanism through "
                "renaming or minor threshold changes."
            ),
            "permitted_reuse": (
                "Reuse a theme only after changing its core conditioning mechanism, "
                "prediction direction, or signal timing in response to diagnosis."
            ),
        },
        "manager_rationale": manager_decision.get("rationale"),
        "manager_instructions": manager_decision.get("instructions"),
        "research_strategy": (
            {
                "key": strategy.key,
                "name": strategy.name,
                "focus": strategy.focus,
                "allowed_concepts": [
                    concept.value for concept in strategy.allowed_concepts
                ],
            }
            if strategy is not None
            else None
        ),
        "relevant_experiments": relevant_experiments,
    }
