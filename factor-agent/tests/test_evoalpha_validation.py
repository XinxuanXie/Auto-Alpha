import unittest
from datetime import datetime

import polars as pl

from evoalpha_workflow.run_single_factor_validation import (
    _apply_frozen_direction,
    _json_safe,
)


class EvoAlphaValidationTests(unittest.TestCase):
    def test_applies_negative_frozen_direction_to_factor_outputs(self):
        frame = pl.DataFrame(
            {
                "factor": [1.5, -2.0],
                "raw_factor": [3.0, -4.0],
            }
        ).lazy()

        result = _apply_frozen_direction(frame, -1).collect()

        self.assertEqual(result.get_column("factor").to_list(), [-1.5, 2.0])
        self.assertEqual(result.get_column("raw_factor").to_list(), [-3.0, 4.0])

    def test_preserves_positive_frozen_direction(self):
        frame = pl.DataFrame(
            {"factor": [1.0], "raw_factor": [2.0]}
        ).lazy()

        result = _apply_frozen_direction(frame, 1).collect()

        self.assertEqual(result.get_column("factor").to_list(), [1.0])
        self.assertEqual(result.get_column("raw_factor").to_list(), [2.0])

    def test_serializes_data_audit_timestamps(self):
        value = datetime(2002, 1, 2, 3, 4, 5)

        self.assertEqual(_json_safe({"timestamp": value}), {"timestamp": str(value)})


if __name__ == "__main__":
    unittest.main()
