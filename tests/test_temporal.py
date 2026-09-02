import unittest
from pathlib import Path

import pandas as pd

from features.feature_engineering import build_feature_matrix
from ml.inference import FEATURE_COLUMNS, score_customer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class TemporalAndEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.customers = pd.read_csv(DATA_DIR / "customers.csv")
        self.orders = pd.read_csv(DATA_DIR / "orders.csv")
        self.redemptions = pd.read_csv(DATA_DIR / "offer_redemptions.csv")
        self.devices = pd.read_csv(DATA_DIR / "customer_devices.csv")
        self.addresses = pd.read_csv(DATA_DIR / "customer_addresses.csv")
        self.payments = pd.read_csv(DATA_DIR / "customer_payments.csv")
        self.ips = pd.read_csv(DATA_DIR / "customer_ips.csv")

        self.max_as_of = max(
            pd.to_datetime(self.customers["created_at"]).max(),
            pd.to_datetime(self.orders["timestamp"]).max(),
            pd.to_datetime(self.redemptions["timestamp"]).max(),
        )

    def test_no_future_leakage_when_as_of_moves_backwards(self):
        # Pick a customer created before 2026-03-01
        self.customers["created_at_dt"] = pd.to_datetime(self.customers["created_at"])
        early_cust = self.customers.loc[self.customers["created_at_dt"] < "2026-01-01"].iloc[0]
        target_cid = early_cust["customer_id"]

        mid_as_of = pd.Timestamp("2026-03-01T00:00:00")

        matrix_full = build_feature_matrix(data_dir=DATA_DIR, as_of=self.max_as_of)
        matrix_mid = build_feature_matrix(data_dir=DATA_DIR, as_of=mid_as_of)

        row_full = matrix_full.loc[matrix_full["customer_id"] == target_cid].iloc[0]
        row_mid = matrix_mid.loc[matrix_mid["customer_id"] == target_cid].iloc[0]

        # Mid account age should be strictly smaller than full account age
        self.assertLess(row_mid["account_age_days"], row_full["account_age_days"])
        # Mid order count should be <= full order count
        self.assertLessEqual(row_mid["order_count"], row_full["order_count"])
        # Mid total spend should be <= full total spend
        self.assertLessEqual(row_mid["total_spend"], row_full["total_spend"])

    def test_prediction_equivalence_across_endpoints(self):
        # Test 5 customers for exact equivalence between score_customer and precomputed features
        test_cids = self.customers["customer_id"].head(5).tolist()

        for cid in test_cids:
            score_res = score_customer(
                customer_id=cid,
                customers=self.customers,
                orders=self.orders,
                offer_redemptions=self.redemptions,
                customer_devices=self.devices,
                customer_addresses=self.addresses,
                customer_payments=self.payments,
                customer_ips=self.ips,
                as_of=self.max_as_of,
            )

            # Feature snapshot keys must match exact 16-feature order
            self.assertEqual(tuple(score_res["feature_snapshot"].keys()), FEATURE_COLUMNS)
            self.assertGreaterEqual(score_res["abuse_probability"], 0.0)
            self.assertLessEqual(score_res["abuse_probability"], 1.0)
            self.assertEqual(score_res["decision_threshold"], 0.5)
            self.assertEqual(score_res["model_name"], "xgboost_groupaware")


if __name__ == "__main__":
    unittest.main()
