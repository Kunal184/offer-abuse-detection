import unittest
from pathlib import Path

import pandas as pd

from features.ingestion import load_raw_dataset, validate_and_clean_table


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class IngestionTest(unittest.TestCase):
    def test_dataset_loading_and_validation(self):
        dataset, reports = load_raw_dataset(DATA_DIR)

        self.assertIn("customers", dataset)
        self.assertIn("orders", dataset)
        self.assertIn("offer_redemptions", dataset)

        for name, report in reports.items():
            self.assertTrue(report.is_valid, f"Validation failed for table {name}: {report.errors}")
            self.assertGreater(report.valid_rows, 0)
            self.assertEqual(report.dropped_rows, 0)

    def test_malformed_data_handling(self):
        malformed_df = pd.DataFrame([
            {"customer_id": "c1", "created_at": "2026-01-01 10:00:00"},
            {"customer_id": "c2", "created_at": "invalid-timestamp"},
            {"customer_id": None, "created_at": "2026-01-02 10:00:00"},
        ])

        cleaned, report = validate_and_clean_table(malformed_df, "customers")
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.valid_rows, 1)
        self.assertEqual(report.dropped_rows, 2)
        self.assertEqual(cleaned.iloc[0]["customer_id"], "c1")


if __name__ == "__main__":
    unittest.main()
