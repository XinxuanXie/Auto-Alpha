import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from agentic_workflow.state import create_initial_state
from agentic_workflow.storage import SQLiteResearchRepository
from agentic_workflow.tools.research_agent import (
    RESEARCH_STRATEGIES,
    ResearchAgentTool,
)
from agentic_workflow.tools.web_research import (
    DDGSWebResearchProvider,
    KeyFreeWebResearchProvider,
    WebResearchSource,
    build_strategy_search_query,
    build_strategy_search_queries,
)


class FakeSearchClient:
    def __init__(self, **kwargs):
        self.timeout = kwargs["timeout"]

    def text(self, query, max_results):
        return [
            {
                "title": "Momentum and volume research",
                "href": "https://example.com/research",
                "body": "An empirical study of momentum and trading volume.",
            },
            {
                "title": "Duplicate",
                "href": "https://example.com/research",
                "body": "Duplicate URL must be removed.",
            },
            {"title": "Missing snippet", "href": "https://example.com/empty"},
        ]


class FakeWebResearch:
    def search(self, query):
        return [
            WebResearchSource(
                source_id="WEB-01",
                title="Momentum and volume research",
                url="https://example.com/research",
                snippet="An empirical study of momentum and trading volume.",
            )
        ]


class FakeFactorLibrary:
    def search(self, concepts, *, iteration, limit=12):
        return [
            {
                "factor_id": "ALPHA158-ROC5",
                "name": "Qlib Alpha158 ROC5",
                "family": "alpha158",
                "definition": "Ref($close,5)/$close",
                "economic_rationale": "五日价格变化结构。",
                "required_data": ["close"],
                "concepts": ["price_momentum", "recent_return"],
                "implementation_status": "available_now",
                "source_url": "https://github.com/microsoft/qlib",
            }
        ]


class FailedProvider:
    def search(self, query):
        raise RuntimeError("primary search unavailable")


class SuccessfulProvider:
    def search(self, query):
        return FakeWebResearch().search(query)


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        proposal = {
            "title": "成交量确认的价格动量",
            "claim": "价格上涨且成交量趋势增强的股票未来五日收益更高。",
            "economic_rationale": "成交量确认表明价格变化包含持续信息。",
            "prediction_direction": "positive",
            "prediction_horizon_days": 5,
            "required_concepts": ["price_momentum", "volume_trend"],
            "source_references": [
                {
                    "source_id": "WEB-01",
                    "relevance": "该资料启发使用成交量确认价格趋势。",
                }
            ],
            "factor_references": [
                {
                    "factor_id": "ALPHA158-ROC5",
                    "adaptation": "使用短期收益并由成交量趋势确认。",
                }
            ],
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(proposal, ensure_ascii=False)
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class WebResearchTests(unittest.TestCase):
    def test_normalizes_and_deduplicates_search_results(self):
        provider = DDGSWebResearchProvider(
            max_results=5,
            timeout_seconds=7,
            client_factory=FakeSearchClient,
        )

        sources = provider.search(" momentum   volume ")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_id, "WEB-01")
        self.assertEqual(str(sources[0].url), "https://example.com/research")

    def test_builds_distinct_strategy_queries(self):
        momentum = build_strategy_search_query("momentum")
        reversal = build_strategy_search_query("reversal")

        self.assertIn("momentum", momentum)
        self.assertIn("reversal", reversal)
        self.assertNotEqual(momentum, reversal)
        self.assertEqual(len(build_strategy_search_queries("momentum")), 2)

    def test_key_free_provider_falls_back_after_search_failure(self):
        provider = KeyFreeWebResearchProvider(
            providers=[FailedProvider(), SuccessfulProvider()]
        )

        sources = provider.search("momentum research")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_id, "WEB-01")

    def test_research_agent_preserves_verified_source_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = SQLiteResearchRepository(
                Path(temporary_directory) / "research.db"
            )
            client = FakeClient()
            agent = ResearchAgentTool(
                repository=repository,
                strategy=RESEARCH_STRATEGIES[0],
                client=client,
                web_research=FakeWebResearch(),
                factor_library=FakeFactorLibrary(),
            )
            state = create_initial_state()
            state["iteration"] = 1

            result = agent.run(state)

        self.assertTrue(result.success, result.error)
        source = result.payload["hypothesis"]["sources"][0]
        self.assertEqual(source["title"], "Momentum and volume research")
        self.assertEqual(source["url"], "https://example.com/research")
        factor_reference = result.payload["hypothesis"]["factor_references"][0]
        self.assertEqual(factor_reference["factor_id"], "ALPHA158-ROC5")
        request = json.loads(
            client.chat.completions.request["messages"][1]["content"]
        )
        self.assertEqual(
            request["research_context"]["web_sources"][0]["source_id"],
            "WEB-01",
        )
        self.assertEqual(
            request["research_context"]["factor_library_examples"][0][
                "factor_id"
            ],
            "ALPHA158-ROC5",
        )


if __name__ == "__main__":
    unittest.main()
