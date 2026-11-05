import unittest

from agentic_workflow.factor_library import FactorKnowledgeBase


class FactorKnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = FactorKnowledgeBase()

    def test_contains_complete_factor_families(self):
        self.assertEqual(
            self.library.count_by_family(),
            {"academic": 20, "alpha101": 101, "alpha158": 158},
        )

    def test_retrieval_balances_families_and_rotates_by_iteration(self):
        concepts = ["price_momentum", "recent_return", "volume_trend"]

        first = self.library.search(concepts, iteration=1, limit=9)
        second = self.library.search(concepts, iteration=2, limit=9)

        self.assertEqual({entry["family"] for entry in first}, {
            "alpha101",
            "alpha158",
            "academic",
        })
        self.assertNotEqual(
            [entry["factor_id"] for entry in first],
            [entry["factor_id"] for entry in second],
        )

    def test_marks_unavailable_data_without_hiding_factor(self):
        value = next(
            entry
            for entry in self.library.entries
            if entry.factor_id == "ACADEMIC-VALUE-BM"
        )

        self.assertEqual(value.implementation_status, "requires_additional_data")
        self.assertIn("book_equity", value.required_data)


if __name__ == "__main__":
    unittest.main()
